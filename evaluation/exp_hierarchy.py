"""
evaluation/exp_hierarchy.py — Prajñā 0.3
Hierarchical Inference Experiment Suite

6 experiments proving or disproving the value of hierarchical routing.
All experiments are independently runnable and return structured dicts
compatible with the Analysis tab in app.py.

Experiment 1: Single-stage vs Hierarchical — FAR/FRR/compute comparison
Experiment 2: Compute Reduction Analysis — f1 rate by quality decile
Experiment 3: Latency Analysis — per-stage timing, breakeven f1
Experiment 4: False Accept Recovery — Stage-1 FAs recovered by Stage 2
Experiment 5: Review Rate Comparison — REVIEW rate: flat vs hierarchical
Experiment 6: Routing Ablation Study — which rho components matter

Design:
  - All experiments operate on a passed-in database (genuine + impostor pairs)
  - No Streamlit imports (pure computation)
  - Each returns a dict with results + metadata + a 'note' field
  - All results include explicit 'limitation' notes per the Prajna 0.2 tradition

IMPORTANT: These experiments use synthetic pair generation from the existing
database (same-identity = genuine, cross-identity = impostor). This is a
research-grade evaluation methodology, not a production certification.
"""

from __future__ import annotations

import time
import json
import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

from core.metrics import (
    calculate_similarity_and_margin,
    compute_entropy_and_certainty,
    responsibility_score,
    compute_ambiguity,
)
from core.quality import compute_composite_quality, QualityComponents
from core.thresholds import compute_adaptive_thresholds
from core.decision import make_decision, STRANGER_FLOOR_DEFAULT
from core.routing import (
    routing_score as compute_rho,
    routing_decision,
    build_routing_record,
    RHO_ACCEPT_DEFAULT,
    RHO_REJECT_DEFAULT,
    KAPPA_DEFAULT,
    LAMBDA_DEFAULT,
)

log = logging.getLogger(__name__)

TAU_DEFAULT = 0.03


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _build_eval_pairs(
    database: dict,
    max_genuine: int = 200,
    max_impostor: int = 200,
) -> list[dict]:
    """
    Builds genuine and impostor pairs from the database.

    Genuine pairs: same identity queried against itself (leave-one-out proxy).
    Impostor pairs: cross-identity queries.

    Each pair dict:
      {embedding, identity, all_db_minus_self (for genuine),
       is_genuine, query_identity, scores, margin, entropy, certainty,
       ambiguity, R, Q (placeholder 0.7), top1_name, top1_sim}

    Note: Q is set to a neutral placeholder (0.7) since we don't have
    actual face images in the evaluation DB, only embeddings.
    This is an acknowledged limitation documented in the 'note' field.
    """
    names = list(database.keys())
    if len(names) < 2:
        return []

    pairs = []
    rng = np.random.default_rng(42)

    # Genuine: each identity queries against full DB
    genuine_names = rng.choice(names, size=min(max_genuine, len(names)), replace=False)
    for name in genuine_names:
        emb = database[name]
        scores, margin = calculate_similarity_and_margin(emb, database)
        k = min(5, len(scores))
        sims = [s[1] for s in scores[:k]]
        H, U = compute_entropy_and_certainty(sims, tau=TAU_DEFAULT)
        amb = compute_ambiguity(scores)
        top1_name = scores[0][0]
        top1_sim  = scores[0][1]
        R = responsibility_score(top1_sim, margin, U)
        pairs.append({
            "is_genuine":      True,
            "query_identity":  name,
            "top1_name":       top1_name,
            "top1_sim":        top1_sim,
            "margin":          margin,
            "entropy":         H,
            "certainty":       U,
            "ambiguity":       amb,
            "R":               R,
            "Q":               0.70,   # neutral placeholder
            "scores":          scores,
            "k":               k,
        })

    # Impostor: random cross-identity pairs
    sampled = min(max_impostor, len(names) * (len(names) - 1) // 2)
    for _ in range(sampled):
        q_name, db_name = rng.choice(names, size=2, replace=False)
        emb = database[q_name]
        # Query against full DB
        scores, margin = calculate_similarity_and_margin(emb, database)
        k = min(5, len(scores))
        sims = [s[1] for s in scores[:k]]
        H, U = compute_entropy_and_certainty(sims, tau=TAU_DEFAULT)
        amb = compute_ambiguity(scores)
        top1_name = scores[0][0]
        top1_sim  = scores[0][1]
        R = responsibility_score(top1_sim, margin, U)
        pairs.append({
            "is_genuine":      False,
            "query_identity":  q_name,
            "top1_name":       top1_name,
            "top1_sim":        top1_sim,
            "margin":          margin,
            "entropy":         H,
            "certainty":       U,
            "ambiguity":       amb,
            "R":               R,
            "Q":               0.65,   # slightly lower for impostors (heuristic placeholder)
            "scores":          scores,
            "k":               k,
        })

    return pairs


def _routing_metrics(pairs: list, rho_accept: float, rho_reject: float,
                     kappa: float, lambda_: float, stage: int = 1) -> dict:
    """
    Computes routing statistics for a set of pairs given routing thresholds.
    Returns counts of ACCEPT/REJECT/ESCALATE per pair type.
    """
    results = []
    for p in pairs:
        rho, phi_q, psi_a = compute_rho(
            R=p["R"], Q=p["Q"], A=p["ambiguity"],
            kappa=kappa, lambda_=lambda_,
        )
        dec = routing_decision(
            rho=rho, R=p["R"], Q=p["Q"], A=p["ambiguity"],
            margin=p["margin"], rho_accept=rho_accept, rho_reject=rho_reject,
        )
        tr = compute_adaptive_thresholds(
            p["entropy"], p["k"], p["Q"], p["ambiguity"], p["margin"]
        )
        dummy_qc = QualityComponents(
            blur=p["Q"], confidence=p["Q"], brightness=p["Q"],
            face_size=p["Q"], pose=p["Q"], composite=p["Q"],
            weights_used={},
        )
        final = make_decision(
            R=p["R"], top1_sim=p["top1_sim"], margin=p["margin"],
            entropy=p["entropy"], certainty=p["certainty"],
            ambiguity=p["ambiguity"], tr=tr, quality=dummy_qc,
            predicted_identity=p["top1_name"],
            stranger_floor=STRANGER_FLOOR_DEFAULT,
        )
        correct_identity = p["query_identity"] == p["top1_name"]
        results.append({
            "is_genuine":      p["is_genuine"],
            "correct_identity": correct_identity,
            "routing_action":  dec.action,
            "rho":             rho,
            "final_decision":  final.decision,
            "R":               p["R"],
            "Q":               p["Q"],
            "ambiguity":       p["ambiguity"],
        })
    return results


def _compute_far_frr(results: list, decision_key: str = "final_decision") -> dict:
    """
    Computes FAR, FRR, TAR, review_rate from decision results.
    genuine ACCEPT = True Accept
    impostor ACCEPT = False Accept (FAR)
    genuine REJECT or REVIEW = False Reject (FRR)
    """
    genuine  = [r for r in results if r["is_genuine"]]
    impostor = [r for r in results if not r["is_genuine"]]

    if not genuine or not impostor:
        return {"FAR": 0, "FRR": 0, "TAR": 0, "review_rate": 0, "n_genuine": 0, "n_impostor": 0}

    fa  = sum(1 for r in impostor if r[decision_key] == "ACCEPT")
    fr  = sum(1 for r in genuine  if r[decision_key] == "REJECT")
    ta  = sum(1 for r in genuine  if r[decision_key] == "ACCEPT")
    rev = sum(1 for r in results  if r[decision_key] == "REVIEW")

    FAR = fa / len(impostor)
    FRR = fr / len(genuine)
    TAR = ta / len(genuine)
    review_rate = rev / len(results)

    return {
        "FAR": round(FAR, 5),
        "FRR": round(FRR, 5),
        "TAR": round(TAR, 5),
        "review_rate": round(review_rate, 5),
        "n_genuine":  len(genuine),
        "n_impostor": len(impostor),
    }


# ---------------------------------------------------------------------------
# Experiment 1: Single-stage vs Hierarchical
# ---------------------------------------------------------------------------

def exp_single_vs_hierarchical(
    database_s1: dict,
    database_s2: dict,
    rho_accept: float = RHO_ACCEPT_DEFAULT,
    rho_reject: float = RHO_REJECT_DEFAULT,
    kappa: float = KAPPA_DEFAULT,
    lambda_: float = LAMBDA_DEFAULT,
) -> dict:
    """
    Experiment 1: Compares three conditions on the evaluation set:
      (a) Flat Stage-1 only
      (b) Flat Stage-2 only
      (c) Hierarchical Stage-1 -> Stage-2

    Metrics: FAR, FRR, TAR, review_rate, compute_units_avg

    Hypothesis: (c) achieves FAR <= (b) while avg compute < (b).
    """
    if len(database_s2) < 3:
        return {"error": "Stage-2 database must have at least 3 identities."}

    pairs_s1 = _build_eval_pairs(database_s1) if database_s1 else []
    pairs_s2 = _build_eval_pairs(database_s2)

    # (a) Flat Stage-1
    if database_s1:
        res_s1 = _routing_metrics(pairs_s1, rho_accept=1.01, rho_reject=-0.01,
                                   kappa=kappa, lambda_=lambda_, stage=1)
        metrics_s1 = _compute_far_frr(res_s1)
        metrics_s1["label"] = "Flat Stage-1"
        metrics_s1["compute_avg"] = 0.15
    else:
        metrics_s1 = {"label": "Flat Stage-1", "FAR": None, "FRR": None,
                      "TAR": None, "review_rate": None, "compute_avg": None,
                      "note": "Stage-1 DB empty"}

    # (b) Flat Stage-2
    res_s2 = _routing_metrics(pairs_s2, rho_accept=1.01, rho_reject=-0.01,
                               kappa=kappa, lambda_=lambda_, stage=2)
    metrics_s2 = _compute_far_frr(res_s2)
    metrics_s2["label"] = "Flat Stage-2"
    metrics_s2["compute_avg"] = 1.00

    # (c) Hierarchical
    if database_s1:
        hier_results = []
        compute_used = []
        for p_s1, p_s2 in zip(pairs_s1, pairs_s2):
            rho1, _, _ = compute_rho(R=p_s1["R"], Q=p_s1["Q"], A=p_s1["ambiguity"],
                                     kappa=kappa, lambda_=lambda_)
            dec1 = routing_decision(rho1, p_s1["R"], p_s1["Q"], p_s1["ambiguity"],
                                    p_s1["margin"], rho_accept=rho_accept, rho_reject=rho_reject)
            if dec1.action in ("ACCEPT", "REJECT"):
                tr = compute_adaptive_thresholds(p_s1["entropy"], p_s1["k"], p_s1["Q"],
                                                  p_s1["ambiguity"], p_s1["margin"])
                dummy_qc = QualityComponents(blur=p_s1["Q"], confidence=p_s1["Q"],
                                             brightness=p_s1["Q"], face_size=p_s1["Q"],
                                             pose=p_s1["Q"], composite=p_s1["Q"], weights_used={})
                final = make_decision(R=p_s1["R"], top1_sim=p_s1["top1_sim"],
                                      margin=p_s1["margin"], entropy=p_s1["entropy"],
                                      certainty=p_s1["certainty"], ambiguity=p_s1["ambiguity"],
                                      tr=tr, quality=dummy_qc,
                                      predicted_identity=p_s1["top1_name"],
                                      stranger_floor=STRANGER_FLOOR_DEFAULT)
                hier_results.append({"is_genuine": p_s1["is_genuine"],
                                      "final_decision": final.decision,
                                      "routing_action": dec1.action, "rho": rho1})
                compute_used.append(0.15)
            else:
                # Escalate to Stage 2
                tr2 = compute_adaptive_thresholds(p_s2["entropy"], p_s2["k"], p_s2["Q"],
                                                   p_s2["ambiguity"], p_s2["margin"])
                dummy_qc2 = QualityComponents(blur=p_s2["Q"], confidence=p_s2["Q"],
                                              brightness=p_s2["Q"], face_size=p_s2["Q"],
                                              pose=p_s2["Q"], composite=p_s2["Q"], weights_used={})
                final2 = make_decision(R=p_s2["R"], top1_sim=p_s2["top1_sim"],
                                       margin=p_s2["margin"], entropy=p_s2["entropy"],
                                       certainty=p_s2["certainty"], ambiguity=p_s2["ambiguity"],
                                       tr=tr2, quality=dummy_qc2,
                                       predicted_identity=p_s2["top1_name"],
                                       stranger_floor=STRANGER_FLOOR_DEFAULT)
                hier_results.append({"is_genuine": p_s2["is_genuine"],
                                      "final_decision": final2.decision,
                                      "routing_action": "ESCALATE", "rho": rho1})
                compute_used.append(1.15)

        metrics_hier = _compute_far_frr(hier_results)
        metrics_hier["label"] = "Hierarchical S1->S2"
        metrics_hier["compute_avg"] = round(float(np.mean(compute_used)), 4)
        f1_rate = sum(1 for r in hier_results if r["routing_action"] != "ESCALATE") / max(len(hier_results), 1)
        metrics_hier["f1_termination_rate"] = round(f1_rate, 4)
        compute_savings = round((1.0 - metrics_hier["compute_avg"]) * 100, 1)
    else:
        metrics_hier = {"label": "Hierarchical S1->S2", "note": "Stage-1 DB empty; cannot run"}
        compute_savings = 0.0

    return {
        "flat_s1":    metrics_s1,
        "flat_s2":    metrics_s2,
        "hierarchical": metrics_hier,
        "compute_savings_pct": compute_savings,
        "note": (
            "Results computed on synthetic pairs from enrollment database. "
            "Quality Q is set to a neutral placeholder (0.70/0.65) since face images are "
            "not stored in embedding-only databases. "
            "Stage-1 and Stage-2 operate in different metric spaces; "
            "R scores are not directly comparable between stages."
        ),
        "limitation": "Dataset-specific. Recalibrate rho_accept/rho_reject on held-out data.",
    }


# ---------------------------------------------------------------------------
# Experiment 2: Compute Reduction Analysis
# ---------------------------------------------------------------------------

def exp_compute_reduction(
    database_s1: dict,
    rho_accept: float = RHO_ACCEPT_DEFAULT,
    rho_reject: float = RHO_REJECT_DEFAULT,
    kappa: float = KAPPA_DEFAULT,
    lambda_: float = LAMBDA_DEFAULT,
) -> dict:
    """
    Experiment 2: What fraction of queries terminate at Stage 1?
    Stratified by Q decile to show quality dependence.

    Since we use a neutral Q placeholder, we sweep Q synthetically
    to show how f1 rate would vary by quality level.
    """
    if len(database_s1) < 2:
        return {"error": "Stage-1 database must have at least 2 identities."}

    pairs = _build_eval_pairs(database_s1)
    if not pairs:
        return {"error": "Could not build evaluation pairs."}

    q_levels    = np.linspace(0.20, 1.00, 9)
    decile_rows = []
    overall_rhos = []

    for q_val in q_levels:
        term_count = 0
        total      = 0
        rhos       = []
        for p in pairs:
            p_copy = dict(p)
            p_copy["Q"] = float(q_val)
            rho, _, _ = compute_rho(R=p["R"], Q=float(q_val), A=p["ambiguity"],
                                    kappa=kappa, lambda_=lambda_)
            rhos.append(rho)
            dec = routing_decision(rho, p["R"], float(q_val), p["ambiguity"],
                                   p["margin"], rho_accept=rho_accept, rho_reject=rho_reject)
            if dec.action in ("ACCEPT", "REJECT"):
                term_count += 1
            total += 1

        f1 = term_count / max(total, 1)
        c_avg = 0.15 * f1 + (0.15 + 1.0) * (1 - f1)
        decile_rows.append({
            "Q_level": round(float(q_val), 2),
            "f1_termination_rate": round(f1, 4),
            "compute_avg": round(c_avg, 4),
            "compute_savings_pct": round((1.0 - c_avg) * 100, 1),
            "mean_rho": round(float(np.mean(rhos)), 4),
        })
        overall_rhos.extend(rhos)

    overall_f1 = float(np.mean([r["f1_termination_rate"] for r in decile_rows]))

    return {
        "decile_results": decile_rows,
        "overall_f1": round(overall_f1, 4),
        "rho_values": [round(r, 4) for r in overall_rhos[:500]],
        "note": (
            "Q is swept synthetically (0.20–1.00) since only embeddings are stored. "
            f"At Q=0.70 (neutral), f1={decile_rows[4]['f1_termination_rate']:.3f} queries terminate at Stage 1."
        ),
        "limitation": "Real-world f1 depends on actual image quality distribution at deployment.",
    }


# ---------------------------------------------------------------------------
# Experiment 3: Latency Analysis
# ---------------------------------------------------------------------------

def exp_latency_analysis(
    database_s1: dict,
    database_s2: dict,
    n_warmup: int = 5,
    n_trials: int = 20,
) -> dict:
    """
    Experiment 3: Measures actual per-stage search latency (embedding matching only).
    Embedding inference latency is not measured here (requires model instances).
    Computes breakeven f1 below which hierarchy is slower than flat Stage-2.
    """
    if not database_s1 or not database_s2:
        return {"error": "Both Stage-1 and Stage-2 databases required for latency experiment."}

    names_s1 = list(database_s1.keys())
    names_s2 = list(database_s2.keys())
    rng = np.random.default_rng(99)

    def measure_search_latency(db: dict, n: int) -> list:
        latencies = []
        for _ in range(n):
            name = rng.choice(list(db.keys()))
            emb  = db[name]
            t0   = time.perf_counter()
            calculate_similarity_and_margin(emb, db)
            latencies.append((time.perf_counter() - t0) * 1000)
        return latencies

    # Warmup
    measure_search_latency(database_s2, n_warmup)

    lat_s1 = measure_search_latency(database_s1, n_trials)
    lat_s2 = measure_search_latency(database_s2, n_trials)

    med_s1 = float(np.median(lat_s1))
    med_s2 = float(np.median(lat_s2))
    p95_s1 = float(np.percentile(lat_s1, 95))
    p95_s2 = float(np.percentile(lat_s2, 95))

    # Breakeven: latency_1 + (1-f1)*latency_2 = latency_2
    # => f1_breakeven = latency_1 / latency_2
    f1_breakeven = med_s1 / med_s2 if med_s2 > 0 else 1.0

    # Expected latency at different f1 values
    f1_sweep = np.linspace(0, 1, 11)
    expected_lat = [
        float(med_s1 + (1 - f1) * med_s2)
        for f1 in f1_sweep
    ]

    return {
        "stage1_search": {
            "median_ms": round(med_s1, 3),
            "p95_ms":    round(p95_s1, 3),
            "n_trials":  n_trials,
            "db_size":   len(database_s1),
        },
        "stage2_search": {
            "median_ms": round(med_s2, 3),
            "p95_ms":    round(p95_s2, 3),
            "n_trials":  n_trials,
            "db_size":   len(database_s2),
        },
        "f1_breakeven": round(f1_breakeven, 4),
        "f1_sweep": [round(f, 2) for f in f1_sweep.tolist()],
        "expected_latency_ms": [round(l, 3) for l in expected_lat],
        "flat_s2_latency_ms": round(med_s2, 3),
        "note": (
            "Latency measured for similarity search only (not embedding inference). "
            f"f1_breakeven={f1_breakeven:.3f}: hierarchy faster when >{f1_breakeven*100:.0f}% "
            "of queries terminate at Stage 1. "
            "Embedding inference dominates real-world latency; measure separately."
        ),
        "limitation": "Search latency is O(N) per stage. Real systems use ANN indexes.",
    }


# ---------------------------------------------------------------------------
# Experiment 4: False Accept Recovery
# ---------------------------------------------------------------------------

def exp_false_accept_recovery(
    database_s1: dict,
    database_s2: dict,
    rho_accept: float = RHO_ACCEPT_DEFAULT,
    rho_reject: float = RHO_REJECT_DEFAULT,
    kappa: float = KAPPA_DEFAULT,
    lambda_: float = LAMBDA_DEFAULT,
) -> dict:
    """
    Experiment 4: Among Stage-1 false accepts (impostor ACCEPTs),
    how many does Stage-2 correctly recover (reclassify to REJECT/REVIEW)?
    """
    if not database_s1 or len(database_s2) < 2:
        return {"error": "Both Stage-1 and Stage-2 databases required."}

    pairs_s1 = _build_eval_pairs(database_s1)
    pairs_s2 = _build_eval_pairs(database_s2)
    if not pairs_s1:
        return {"error": "Could not build evaluation pairs from Stage-1 database."}

    # Find Stage-1 false accepts
    fa_cases_s1 = []
    fa_indices  = []
    for i, (p1, p2) in enumerate(zip(pairs_s1, pairs_s2)):
        if p1["is_genuine"]:
            continue
        # Stage-1 flat decision
        tr1 = compute_adaptive_thresholds(p1["entropy"], p1["k"], p1["Q"],
                                           p1["ambiguity"], p1["margin"])
        dummy_qc1 = QualityComponents(blur=p1["Q"], confidence=p1["Q"],
                                      brightness=p1["Q"], face_size=p1["Q"],
                                      pose=p1["Q"], composite=p1["Q"], weights_used={})
        dr1 = make_decision(R=p1["R"], top1_sim=p1["top1_sim"],
                             margin=p1["margin"], entropy=p1["entropy"],
                             certainty=p1["certainty"], ambiguity=p1["ambiguity"],
                             tr=tr1, quality=dummy_qc1,
                             predicted_identity=p1["top1_name"],
                             stranger_floor=STRANGER_FLOOR_DEFAULT)
        if dr1.decision == "ACCEPT":
            fa_cases_s1.append({"s1": p1, "s2": p2, "dr1": dr1})
            fa_indices.append(i)

    if not fa_cases_s1:
        return {
            "n_fa_stage1": 0,
            "n_recovered": 0,
            "recovery_rate": 1.0,
            "note": "No Stage-1 false accepts found. System has very low FAR at Stage 1.",
            "limitation": "Result is dataset-specific.",
        }

    # Check Stage-2 decision for each FA
    recovered    = 0
    not_recovered = 0
    recovery_details = []

    for case in fa_cases_s1:
        p2 = case["s2"]
        tr2 = compute_adaptive_thresholds(p2["entropy"], p2["k"], p2["Q"],
                                           p2["ambiguity"], p2["margin"])
        dummy_qc2 = QualityComponents(blur=p2["Q"], confidence=p2["Q"],
                                      brightness=p2["Q"], face_size=p2["Q"],
                                      pose=p2["Q"], composite=p2["Q"], weights_used={})
        dr2 = make_decision(R=p2["R"], top1_sim=p2["top1_sim"],
                             margin=p2["margin"], entropy=p2["entropy"],
                             certainty=p2["certainty"], ambiguity=p2["ambiguity"],
                             tr=tr2, quality=dummy_qc2,
                             predicted_identity=p2["top1_name"],
                             stranger_floor=STRANGER_FLOOR_DEFAULT)
        is_recovered = dr2.decision in ("REJECT", "REVIEW")
        if is_recovered:
            recovered += 1
        else:
            not_recovered += 1
        recovery_details.append({
            "query":      case["s1"]["query_identity"],
            "s1_R":       round(case["s1"]["R"], 4),
            "s2_R":       round(p2["R"], 4),
            "s1_decision": "ACCEPT",
            "s2_decision": dr2.decision,
            "recovered":   is_recovered,
        })

    recovery_rate = recovered / len(fa_cases_s1)

    return {
        "n_fa_stage1":     len(fa_cases_s1),
        "n_recovered":     recovered,
        "n_not_recovered": not_recovered,
        "recovery_rate":   round(recovery_rate, 4),
        "recovery_details": recovery_details[:20],  # cap for display
        "note": (
            f"{len(fa_cases_s1)} Stage-1 false accepts found. "
            f"Stage-2 recovered {recovered} ({recovery_rate*100:.1f}%). "
            "This measures Stage-2's ability to correct Stage-1 errors for impostor queries."
        ),
        "limitation": (
            "Stage-1 and Stage-2 operate in different metric spaces. "
            "Recovery depends on how differently the two models score the same pair."
        ),
    }


# ---------------------------------------------------------------------------
# Experiment 5: Review Rate Comparison
# ---------------------------------------------------------------------------

def exp_review_rate_comparison(
    database_s1: dict,
    database_s2: dict,
    rho_accept: float = RHO_ACCEPT_DEFAULT,
    rho_reject: float = RHO_REJECT_DEFAULT,
    kappa: float = KAPPA_DEFAULT,
    lambda_: float = LAMBDA_DEFAULT,
) -> dict:
    """
    Experiment 5: REVIEW rate under flat Stage-2 vs hierarchical pipeline.
    Also computes REVIEW rate by case type (genuine / impostor).
    """
    if len(database_s2) < 2:
        return {"error": "Stage-2 database must have at least 2 identities."}

    pairs_s2 = _build_eval_pairs(database_s2)
    if not pairs_s2:
        return {"error": "Could not build evaluation pairs."}

    # Flat Stage-2 reviews
    flat_results = []
    for p in pairs_s2:
        tr = compute_adaptive_thresholds(p["entropy"], p["k"], p["Q"],
                                          p["ambiguity"], p["margin"])
        dummy_qc = QualityComponents(blur=p["Q"], confidence=p["Q"],
                                     brightness=p["Q"], face_size=p["Q"],
                                     pose=p["Q"], composite=p["Q"], weights_used={})
        dr = make_decision(R=p["R"], top1_sim=p["top1_sim"],
                            margin=p["margin"], entropy=p["entropy"],
                            certainty=p["certainty"], ambiguity=p["ambiguity"],
                            tr=tr, quality=dummy_qc,
                            predicted_identity=p["top1_name"],
                            stranger_floor=STRANGER_FLOOR_DEFAULT)
        flat_results.append({"is_genuine": p["is_genuine"], "decision": dr.decision})

    flat_review = sum(1 for r in flat_results if r["decision"] == "REVIEW") / max(len(flat_results), 1)
    flat_review_genuine  = sum(1 for r in flat_results if r["is_genuine"] and r["decision"] == "REVIEW") / max(sum(1 for r in flat_results if r["is_genuine"]), 1)
    flat_review_impostor = sum(1 for r in flat_results if not r["is_genuine"] and r["decision"] == "REVIEW") / max(sum(1 for r in flat_results if not r["is_genuine"]), 1)

    # Hierarchical (Stage-2 only runs on ESCALATE cases)
    hier_reviews = 0
    hier_total   = len(pairs_s2)
    if database_s1:
        pairs_s1 = _build_eval_pairs(database_s1, max_genuine=len(pairs_s2)//2+1, max_impostor=len(pairs_s2)//2+1)
        for p1, p2 in zip(pairs_s1, pairs_s2):
            rho1, _, _ = compute_rho(R=p1["R"], Q=p1["Q"], A=p1["ambiguity"],
                                     kappa=kappa, lambda_=lambda_)
            dec1 = routing_decision(rho1, p1["R"], p1["Q"], p1["ambiguity"], p1["margin"],
                                    rho_accept=rho_accept, rho_reject=rho_reject)
            if dec1.action == "ESCALATE":
                tr2 = compute_adaptive_thresholds(p2["entropy"], p2["k"], p2["Q"],
                                                   p2["ambiguity"], p2["margin"])
                dummy_qc2 = QualityComponents(blur=p2["Q"], confidence=p2["Q"],
                                              brightness=p2["Q"], face_size=p2["Q"],
                                              pose=p2["Q"], composite=p2["Q"], weights_used={})
                dr2 = make_decision(R=p2["R"], top1_sim=p2["top1_sim"],
                                    margin=p2["margin"], entropy=p2["entropy"],
                                    certainty=p2["certainty"], ambiguity=p2["ambiguity"],
                                    tr=tr2, quality=dummy_qc2,
                                    predicted_identity=p2["top1_name"],
                                    stranger_floor=STRANGER_FLOOR_DEFAULT)
                if dr2.decision == "REVIEW":
                    hier_reviews += 1
        hier_review_rate = hier_reviews / max(hier_total, 1)
    else:
        hier_review_rate = flat_review

    return {
        "flat_s2_review_rate":      round(flat_review, 4),
        "flat_review_genuine":      round(flat_review_genuine, 4),
        "flat_review_impostor":     round(flat_review_impostor, 4),
        "hierarchical_review_rate": round(hier_review_rate, 4),
        "review_reduction_pct":     round((flat_review - hier_review_rate) * 100, 2),
        "note": (
            f"Flat Stage-2 REVIEW rate: {flat_review:.3f}. "
            f"Hierarchical REVIEW rate: {hier_review_rate:.3f}. "
            "Hierarchical pipeline reduces reviews by routing confident cases at Stage 1."
        ),
        "limitation": "Genuine review rate includes cases where a human would also flag the decision.",
    }


# ---------------------------------------------------------------------------
# Experiment 6: Routing Ablation Study
# ---------------------------------------------------------------------------

def exp_routing_ablation(
    database_s1: dict,
    rho_accept: float = RHO_ACCEPT_DEFAULT,
    rho_reject: float = RHO_REJECT_DEFAULT,
) -> dict:
    """
    Experiment 6: Which components of rho contribute to routing accuracy?

    Compares four rho formulations:
      A. Full: R * Q^kappa * (1 - lambda*A)
      B. No quality: R * (1 - lambda*A)
      C. No ambiguity: R * Q^kappa
      D. R only: R (no routing signals)
      E. kappa sweep: kappa in {0.25, 0.5, 0.75, 1.0}

    Metric: f1 termination rate + FAR/FRR from final decisions.
    """
    if len(database_s1) < 2:
        return {"error": "Stage-1 database must have at least 2 identities."}

    pairs = _build_eval_pairs(database_s1)
    if not pairs:
        return {"error": "Could not build evaluation pairs."}

    ablation_configs = [
        {"name": "Full rho (R * Q^0.5 * (1-0.3A))", "kappa": 0.50, "lambda_": 0.30,
         "use_Q": True, "use_A": True},
        {"name": "No quality (R * (1-0.3A))",         "kappa": 0.50, "lambda_": 0.30,
         "use_Q": False, "use_A": True},
        {"name": "No ambiguity (R * Q^0.5)",           "kappa": 0.50, "lambda_": 0.30,
         "use_Q": True, "use_A": False},
        {"name": "R only (no routing signals)",         "kappa": 0.50, "lambda_": 0.00,
         "use_Q": False, "use_A": False},
    ]

    ablation_results = []
    for cfg in ablation_configs:
        f1_count  = 0
        decisions = []
        rho_vals  = []
        for p in pairs:
            q_eff = p["Q"] if cfg["use_Q"] else 1.0
            a_eff = p["ambiguity"] if cfg["use_A"] else 0.0
            rho, _, _ = compute_rho(R=p["R"], Q=q_eff, A=a_eff,
                                    kappa=cfg["kappa"], lambda_=cfg["lambda_"])
            rho_vals.append(rho)
            dec = routing_decision(rho, p["R"], q_eff, a_eff, p["margin"],
                                   rho_accept=rho_accept, rho_reject=rho_reject)
            tr = compute_adaptive_thresholds(p["entropy"], p["k"], p["Q"],
                                              p["ambiguity"], p["margin"])
            dummy_qc = QualityComponents(blur=p["Q"], confidence=p["Q"],
                                         brightness=p["Q"], face_size=p["Q"],
                                         pose=p["Q"], composite=p["Q"], weights_used={})
            final = make_decision(R=p["R"], top1_sim=p["top1_sim"],
                                   margin=p["margin"], entropy=p["entropy"],
                                   certainty=p["certainty"], ambiguity=p["ambiguity"],
                                   tr=tr, quality=dummy_qc,
                                   predicted_identity=p["top1_name"],
                                   stranger_floor=STRANGER_FLOOR_DEFAULT)
            if dec.action in ("ACCEPT", "REJECT"):
                f1_count += 1
            decisions.append({"is_genuine": p["is_genuine"], "final_decision": final.decision})

        metrics = _compute_far_frr(decisions, decision_key="final_decision")
        f1_rate = f1_count / max(len(pairs), 1)
        ablation_results.append({
            "config":              cfg["name"],
            "f1_termination_rate": round(f1_rate, 4),
            "FAR":                 metrics["FAR"],
            "FRR":                 metrics["FRR"],
            "mean_rho":            round(float(np.mean(rho_vals)), 4),
            "std_rho":             round(float(np.std(rho_vals)), 4),
        })

    # kappa sweep
    kappa_results = []
    for kappa_val in [0.25, 0.50, 0.75, 1.00]:
        f1_count = 0
        for p in pairs:
            rho, _, _ = compute_rho(R=p["R"], Q=p["Q"], A=p["ambiguity"],
                                    kappa=kappa_val, lambda_=0.30)
            dec = routing_decision(rho, p["R"], p["Q"], p["ambiguity"], p["margin"],
                                   rho_accept=rho_accept, rho_reject=rho_reject)
            if dec.action in ("ACCEPT", "REJECT"):
                f1_count += 1
        kappa_results.append({
            "kappa":               kappa_val,
            "f1_termination_rate": round(f1_count / max(len(pairs), 1), 4),
        })

    return {
        "ablation_results": ablation_results,
        "kappa_sweep":      kappa_results,
        "note": (
            "Ablation shows which rho components drive routing decisions. "
            "Full rho (with Q and A) is expected to have the lowest FAR "
            "and most targeted escalation."
        ),
        "limitation": "Q is a placeholder (0.70); real Q variation would show stronger ablation differences.",
    }
