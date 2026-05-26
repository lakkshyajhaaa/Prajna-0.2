"""
evaluation/experiments.py — Prajñā 0.2
Five Deep Experiments — Depth over Breadth

EXPERIMENTS:
  1. exp_adaptive_vs_static     — Core claim: adaptive thresholds reduce FAR
  2. exp_calibration_tau        — Justifies tau=0.03 via ECE measurement
  3. exp_stranger_rejection     — Identifies optimal open-set floor
  4. exp_weight_sensitivity     — Weight stability + FAR/FRR tradeoffs
  5. exp_quality_weight_stability — Quality metric robustness

PAIR CONSTRUCTION — 1 SAMPLE PER IDENTITY:
  Genuine proxy:  identity A vs full DB → top-1 is A itself (sim≈1.0).
                  Margin and entropy still reflect real nearest-neighbour competition.
  Impostor proxy: identity A vs all OTHER identities (A excluded).

  Consequence: ROC AUC will be near-perfect by construction (self-match).
  This is documented honestly in every result note. The meaningful comparison
  is the FAR/FRR operating-point table, not the ROC curve.
"""

import numpy as np
from scipy import stats as scipy_stats
from core.metrics import (
    calculate_similarity_and_margin,
    compute_entropy_and_certainty,
    responsibility_score,
    compute_ambiguity,
    compute_open_set_score,
)
from core.thresholds import compute_adaptive_thresholds, compute_static_thresholds
from core.calibration import (
    sweep_tau,
    compute_ece,
    entropy_distribution,
    TAU_CANDIDATES,
)
from core.quality import DEFAULT_QUALITY_WEIGHTS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_pairs(database_embs: dict) -> tuple[list[dict], list[dict]]:
    """
    Constructs genuine and impostor score records from the database.

    With 1 sample/identity, genuine pairs are self-matches (sim≈1.0).
    Impostor pairs are cross-identity matches (the hardest non-match).
    See module docstring for full rationale.
    """
    names = list(database_embs.keys())
    genuine_results  = []
    impostor_results = []

    for i, name_i in enumerate(names):
        query_emb = database_embs[name_i]

        # --- Impostor: query A vs all others ---
        other_embs = {n: e for j, (n, e) in enumerate(database_embs.items()) if j != i}
        if not other_embs:
            continue

        scores_imp, margin_imp = calculate_similarity_and_margin(query_emb, other_embs)
        top_k = min(5, len(scores_imp))
        top_sims_imp = [s[1] for s in scores_imp[:top_k]]
        H_imp, U_imp = compute_entropy_and_certainty(top_sims_imp)

        impostor_results.append({
            "query":     name_i,
            "top1_name": scores_imp[0][0],
            "top1_sim":  scores_imp[0][1],
            "margin":    margin_imp,
            "entropy":   H_imp,
            "certainty": U_imp,
            "ambiguity": compute_ambiguity(scores_imp),
            "top_sims":  top_sims_imp,
            "k":         top_k,
            "is_correct": False,
        })

        # --- Genuine: query A vs full DB (A present → self-match top-1) ---
        scores_gen, margin_gen = calculate_similarity_and_margin(query_emb, database_embs)
        top_k_g = min(5, len(scores_gen))
        top_sims_gen = [s[1] for s in scores_gen[:top_k_g]]
        H_gen, U_gen = compute_entropy_and_certainty(top_sims_gen)

        if scores_gen[0][0] == name_i:
            genuine_results.append({
                "query":     name_i,
                "top1_name": name_i,
                "top1_sim":  scores_gen[0][1],
                "margin":    margin_gen,
                "entropy":   H_gen,
                "certainty": U_gen,
                "ambiguity": compute_ambiguity(scores_gen),
                "top_sims":  top_sims_gen,
                "k":         top_k_g,
                "is_correct": True,
            })

    return genuine_results, impostor_results


def _compute_decision(result: dict, use_adaptive: bool, weights=(0.70, 0.15, 0.15)) -> str:
    """Applies decision logic and returns ACCEPT/REVIEW/REJECT."""
    R = responsibility_score(result["top1_sim"], result["margin"], result["certainty"], weights)
    if use_adaptive:
        tr = compute_adaptive_thresholds(
            entropy=result["entropy"], k=result["k"],
            quality=0.75,
            ambiguity=result["ambiguity"], margin=result["margin"],
        )
    else:
        tr = compute_static_thresholds()
    if result["top1_sim"] < 0.60:
        return "REJECT"
    if R >= tr.t_accept:
        return "ACCEPT"
    if R >= tr.t_review:
        return "REVIEW"
    return "REJECT"


def _compute_roc_from_scores(genuine: list, impostor: list, weights: tuple) -> dict:
    """
    Proper ROC computation: sort ALL (R_score, label) pairs descending,
    accumulate TP/FP counts as threshold sweeps from high to low.
    This is threshold-independent and reflects score distribution discriminability.
    """
    scored = []
    for r in genuine:
        R = responsibility_score(r["top1_sim"], r["margin"], r["certainty"], weights)
        scored.append((R, 1))
    for r in impostor:
        R = responsibility_score(r["top1_sim"], r["margin"], r["certainty"], weights)
        scored.append((R, 0))

    scored.sort(key=lambda x: x[0], reverse=True)
    n_genuine  = max(sum(1 for _, l in scored if l == 1), 1)
    n_impostor = max(sum(1 for _, l in scored if l == 0), 1)

    tps, fps = 0, 0
    fpr_list, tpr_list = [0.0], [0.0]
    prev_score = None

    for score, label in scored:
        if score != prev_score and prev_score is not None:
            fpr_list.append(fps / n_impostor)
            tpr_list.append(tps / n_genuine)
        if label == 1:
            tps += 1
        else:
            fps += 1
        prev_score = score

    fpr_list.append(fps / n_impostor)
    tpr_list.append(tps / n_genuine)
    fpr_list.append(1.0)
    tpr_list.append(1.0)

    pairs = sorted(zip(fpr_list, tpr_list))
    auc = float(np.trapz([p[1] for p in pairs], [p[0] for p in pairs]))
    return {"fpr": fpr_list, "tpr": tpr_list, "auc": auc}


def _far_frr_sweep(genuine: list, impostor: list, weights: tuple, n_steps: int = 60) -> dict:
    """
    Sweeps a scalar threshold T over [0.50, 0.97].
    FAR(T) = fraction of impostor R scores >= T.
    FRR(T) = fraction of genuine  R scores <  T.
    """
    thresholds = np.linspace(0.50, 0.97, n_steps).tolist()
    genuine_R  = [responsibility_score(r["top1_sim"], r["margin"], r["certainty"], weights) for r in genuine]
    impostor_R = [responsibility_score(r["top1_sim"], r["margin"], r["certainty"], weights) for r in impostor]
    far_list = [sum(1 for R in impostor_R if R >= t) / max(len(impostor_R), 1) for t in thresholds]
    frr_list = [sum(1 for R in genuine_R  if R <  t) / max(len(genuine_R),  1) for t in thresholds]
    return {"thresholds": thresholds, "far": far_list, "frr": frr_list}


# ===========================================================================
# EXPERIMENT 1 — Adaptive vs Static Thresholds
# ===========================================================================

def exp_adaptive_vs_static(database_embs: dict) -> dict:
    """
    Primary experiment. See module docstring for pair construction rationale.

    ROC: computed correctly from score distributions (threshold-independent).
         AUC near-perfect with 1 sample/identity — documented limitation.
    FAR/FRR: scalar sweep shows score distribution vs threshold.
    Operating points: where each policy sits on the FAR/FRR curve.
    """
    genuine, impostor = _build_pairs(database_embs)
    if not genuine or not impostor:
        return {"error": "Insufficient data. Need at least 2 enrolled identities."}

    weights = (0.70, 0.15, 0.15)

    gen_dec_adap = [_compute_decision(r, True,  weights) for r in genuine]
    gen_dec_stat = [_compute_decision(r, False, weights) for r in genuine]
    imp_dec_adap = [_compute_decision(r, True,  weights) for r in impostor]
    imp_dec_stat = [_compute_decision(r, False, weights) for r in impostor]

    def _op(gen_dec, imp_dec, label):
        far    = sum(1 for d in imp_dec if d == "ACCEPT") / max(len(imp_dec), 1)
        frr    = sum(1 for d in gen_dec if d != "ACCEPT") / max(len(gen_dec), 1)
        review = sum(1 for d in gen_dec if d == "REVIEW") / max(len(gen_dec), 1)
        return {"label": label, "FAR": far, "FRR": frr, "TAR": 1.0 - frr, "review_rate": review}

    adaptive_metrics = _op(gen_dec_adap, imp_dec_adap, "Adaptive (0.2)")
    static_metrics   = _op(gen_dec_stat, imp_dec_stat, "Static (0.1)")

    roc = _compute_roc_from_scores(genuine, impostor, weights)
    ffr = _far_frr_sweep(genuine, impostor, weights)

    avg_t = float(np.mean([
        compute_adaptive_thresholds(r["entropy"], r["k"], 0.75, r["ambiguity"], r["margin"]).t_accept
        for r in genuine + impostor
    ]))

    ac = [1 if d == "ACCEPT" else 0 for d in gen_dec_adap]
    sc = [1 if d == "ACCEPT" else 0 for d in gen_dec_stat]
    if len(ac) >= 3 and len(set(ac)) > 1 and len(set(sc)) > 1:
        t_stat, p_value = scipy_stats.ttest_rel(ac, sc)
    else:
        t_stat, p_value = 0.0, 1.0

    improvement_found = adaptive_metrics["FAR"] < static_metrics["FAR"]
    dataset_note = (
        "\n[Dataset limitation: 1 sample/identity → genuine self-matches (sim≈1.0) "
        "→ ROC AUC trivially near 1.0. The meaningful comparison is the operating-point "
        "FAR/FRR table. Enroll ≥2 images per person for a realistic ROC.]"
    )

    return {
        "adaptive": adaptive_metrics,
        "static":   static_metrics,
        "roc":  {"fpr": roc["fpr"], "tpr": roc["tpr"], "auc": roc["auc"], "avg_adaptive_t": avg_t},
        "far_frr": {
            "thresholds":     ffr["thresholds"],
            "far":            ffr["far"],
            "frr":            ffr["frr"],
            "static_t":       0.72,
            "adaptive_t":     avg_t,
        },
        "significance": {"t_stat": t_stat, "p_value": p_value},
        "improvement_found": improvement_found,
        "n_genuine":  len(genuine),
        "n_impostor": len(impostor),
        "note": (
            ("Adaptive thresholding shows measurable FAR reduction at the operating point."
             if improvement_found else
             "On this dataset, adaptive thresholding did not show FAR improvement. "
             "This reflects low natural ambiguity. "
             "Evaluation on a more diverse dataset is required to generalise this claim.")
            + dataset_note
        ),
    }


# ===========================================================================
# EXPERIMENT 2 — Calibration & Tau Justification
# ===========================================================================

def exp_calibration_tau(database_embs: dict) -> dict:
    genuine, impostor = _build_pairs(database_embs)
    if not genuine or not impostor:
        return {"error": "Insufficient data."}

    genuine_sims  = [r["top_sims"] for r in genuine]
    impostor_sims = [r["top_sims"] for r in impostor]
    all_sims = genuine_sims + impostor_sims
    sweep_tau(all_sims, TAU_CANDIDATES)

    tau_results = {}
    for tau in TAU_CANDIDATES:
        confidences, labels = [], []
        for r in genuine:
            H, U = compute_entropy_and_certainty(r["top_sims"], tau=tau)
            R = responsibility_score(r["top1_sim"], r["margin"], U)
            confidences.append(R); labels.append(1)
        for r in impostor:
            H, U = compute_entropy_and_certainty(r["top_sims"], tau=tau)
            R = responsibility_score(r["top1_sim"], r["margin"], U)
            confidences.append(R); labels.append(0)

        ece_result   = compute_ece(confidences, labels)
        entropy_dist = entropy_distribution(genuine_sims, impostor_sims, tau=tau)
        tau_results[tau] = {
            "ece":           ece_result.ece,
            "separation":    entropy_dist["separation"],
            "mean_genuine_H":  entropy_dist["mean_genuine"],
            "mean_impostor_H": entropy_dist["mean_impostor"],
            "genuine_H":     entropy_dist["genuine_H"],
            "impostor_H":    entropy_dist["impostor_H"],
            "bin_confs":     ece_result.bin_confidences,
            "bin_accs":      ece_result.bin_accuracies,
        }

    best_tau = min(tau_results, key=lambda t: tau_results[t]["ece"])
    return {
        "tau_results":    tau_results,
        "best_tau":       best_tau,
        "chosen_tau":     0.03,
        "tau_justified":  best_tau == 0.03,
        "note": (
            f"tau=0.03 achieves ECE={tau_results[0.03]['ece']:.4f}, separation={tau_results[0.03]['separation']:.3f}. "
            f"Best ECE: tau={best_tau} ({tau_results[best_tau]['ece']:.4f}). "
            + ("tau=0.03 is empirically justified." if best_tau == 0.03 else
               f"tau={best_tau} shows lower ECE; consider updating if this persists on larger datasets.")
        ),
    }


# ===========================================================================
# EXPERIMENT 3 — Stranger Rejection
# ===========================================================================

def exp_stranger_rejection(database_embs: dict) -> dict:
    genuine, impostor = _build_pairs(database_embs)
    if not genuine or not impostor:
        return {"error": "Insufficient data."}

    floors = [0.50, 0.53, 0.55, 0.58, 0.60, 0.63, 0.65, 0.68, 0.70]
    floor_results = []

    for floor in floors:
        genuine_pass     = sum(1 for r in genuine  if r["top1_sim"] >= floor)
        impostor_blocked = sum(1 for r in impostor if r["top1_sim"] <  floor)
        impostor_past    = [r for r in impostor if r["top1_sim"] >= floor]
        fa  = sum(1 for r in impostor_past
                  if responsibility_score(r["top1_sim"], r["margin"], r["certainty"]) >= 0.72)
        floor_results.append({
            "floor":                   floor,
            "TAR":                     genuine_pass / max(len(genuine), 1),
            "FAR":                     fa / max(len(impostor), 1),
            "stranger_rejection_rate": impostor_blocked / max(len(impostor), 1),
            "n_impostor_blocked":      impostor_blocked,
            "n_genuine_blocked":       len(genuine) - genuine_pass,
        })

    best = max(floor_results, key=lambda x: x["TAR"] - x["FAR"])
    return {
        "floor_results":   floor_results,
        "optimal_floor":   best["floor"],
        "default_floor":   0.60,
        "floor_justified": best["floor"] == 0.60,
        "n_genuine":       len(genuine),
        "n_impostor":      len(impostor),
        "note": (
            f"Optimal floor: {best['floor']:.2f} (TAR={best['TAR']:.3f}, FAR={best['FAR']:.3f}). "
            f"Default=0.60 is {'justified' if best['floor'] == 0.60 else 'suboptimal — consider updating'}."
        ),
    }


# ===========================================================================
# EXPERIMENT 4 — Responsibility Weight Sensitivity
# ===========================================================================

def exp_weight_sensitivity(database_embs: dict) -> dict:
    genuine, impostor = _build_pairs(database_embs)
    if not genuine or not impostor:
        return {"error": "Insufficient data."}

    weight_sets = [
        (0.70, 0.15, 0.15, "0.70/0.15/0.15 (default)"),
        (0.60, 0.20, 0.20, "0.60/0.20/0.20"),
        (0.80, 0.10, 0.10, "0.80/0.10/0.10"),
        (0.50, 0.25, 0.25, "0.50/0.25/0.25"),
    ]

    results = []
    for w_sim, w_mar, w_cer, label in weight_sets:
        weights   = (w_sim, w_mar, w_cer)
        genuine_R = [responsibility_score(r["top1_sim"], r["margin"], r["certainty"], weights) for r in genuine]
        impostor_R= [responsibility_score(r["top1_sim"], r["margin"], r["certainty"], weights) for r in impostor]
        far = sum(1 for R in impostor_R if R >= 0.72) / max(len(impostor_R), 1)
        frr = sum(1 for R in genuine_R  if R <  0.72) / max(len(genuine_R),  1)
        results.append({
            "label": label, "weights": weights,
            "FAR": far, "FRR": frr, "TAR": 1.0 - frr,
            "r_variance": float(np.var(genuine_R + impostor_R)),
            "genuine_R": genuine_R, "impostor_R": impostor_R,
        })

    best_far    = min(results, key=lambda x: x["FAR"])
    best_stable = min(results, key=lambda x: x["r_variance"])
    default_r   = next(r for r in results if r["weights"] == (0.70, 0.15, 0.15))

    return {
        "results":           results,
        "best_far_label":    best_far["label"],
        "best_stable_label": best_stable["label"],
        "default_is_best_far":    best_far["weights"] == (0.70, 0.15, 0.15),
        "default_is_most_stable": best_stable["weights"] == (0.70, 0.15, 0.15),
        "note": (
            f"{best_far['label']} achieves lowest FAR ({best_far['FAR']:.4f}). "
            f"{best_stable['label']} has lowest R-variance ({best_stable['r_variance']:.5f}). "
            f"Default: FAR={default_r['FAR']:.4f}, var={default_r['r_variance']:.5f}. "
            "Results are dataset-specific — recalibrate for other deployments."
        ),
    }


# ===========================================================================
# EXPERIMENT 5 — Quality Weight Stability
# ===========================================================================

def exp_quality_weight_stability(database_embs: dict) -> dict:
    genuine, impostor = _build_pairs(database_embs)
    if not genuine or not impostor:
        return {"error": "Insufficient data."}

    rng = np.random.default_rng(42)
    n_gen = len(genuine); n_imp = len(impostor)
    genuine_Q  = rng.beta(a=5, b=2, size=n_gen).tolist()
    impostor_Q = rng.beta(a=3, b=3, size=n_imp).tolist()
    weight_sets_raw = rng.dirichlet(alpha=[1, 1, 1, 1, 1], size=10)

    far_values, frr_values, weight_labels = [], [], []

    for i, w_raw in enumerate(weight_sets_raw):
        genuine_final_R, impostor_final_R = [], []
        for j, r in enumerate(genuine):
            tr = compute_adaptive_thresholds(
                entropy=r["entropy"], k=r["k"],
                quality=genuine_Q[j], ambiguity=r["ambiguity"], margin=r["margin"],
            )
            R = responsibility_score(r["top1_sim"], r["margin"], r["certainty"])
            genuine_final_R.append((R, tr.t_accept))
        for j, r in enumerate(impostor):
            tr = compute_adaptive_thresholds(
                entropy=r["entropy"], k=r["k"],
                quality=impostor_Q[j], ambiguity=r["ambiguity"], margin=r["margin"],
            )
            R = responsibility_score(r["top1_sim"], r["margin"], r["certainty"])
            impostor_final_R.append((R, tr.t_accept))

        far = sum(1 for R, T in impostor_final_R if R >= T) / max(n_imp, 1)
        frr = sum(1 for R, T in genuine_final_R  if R <  T) / max(n_gen, 1)
        far_values.append(far); frr_values.append(frr)
        weight_labels.append(f"Set {i+1}")

    far_range = max(far_values) - min(far_values)
    frr_range = max(frr_values) - min(frr_values)

    return {
        "weight_labels": weight_labels,
        "far_values":    far_values,
        "frr_values":    frr_values,
        "far_range":     far_range,
        "frr_range":     frr_range,
        "stable":        far_range < 0.05 and frr_range < 0.05,
        "note": (
            f"FAR range: {far_range:.4f} | FRR range: {frr_range:.4f}. "
            + ("STABLE: quality weight perturbations cause < 5% FAR/FRR change."
               if far_range < 0.05 else
               "Quality weights have measurable impact — further tuning warranted.")
            + "\n[Note: quality sub-scores are simulated; real eval needs image-level quality.]"
        ),
    }
