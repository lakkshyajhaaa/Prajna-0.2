"""
core/metrics.py — Prajñā 0.2
All metric computations, extracted and extended from frt_utils.py.

frt_utils.py is NOT modified — these functions are the 0.2 versions with
configurable parameters (tau, weights) while preserving identical default
behaviour to 0.1.

Key additions vs frt_utils.py:
  - tau is now an argument (default 0.03 preserved)
  - weights are now a configurable tuple (default 0.7/0.15/0.15 preserved)
  - compute_ambiguity() — identity separation ratio (new)
  - compute_open_set_score() — stranger rejection metric (new)
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Preserved from frt_utils.py (identical defaults)
# ---------------------------------------------------------------------------

def calculate_similarity_and_margin(
    query_emb: np.ndarray,
    database_embs: dict,
) -> tuple[list[tuple[str, float]], float]:
    """
    Computes scaled cosine similarity [0, 1] and top-2 margin.
    Identical to frt_utils.calculate_similarity_and_margin().
    Preserved exactly for continuity with 0.1 results.
    """
    scores = []
    query_vec = query_emb.flatten()

    for name, emb in database_embs.items():
        db_vec = emb.flatten()
        cos_sim = np.dot(query_vec, db_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(db_vec) + 1e-9
        )
        scaled_sim = (cos_sim + 1) / 2  # map [-1,1] → [0,1]
        scores.append((name, float(scaled_sim)))

    scores.sort(key=lambda x: x[1], reverse=True)

    margin = 0.0
    if len(scores) > 1:
        margin = scores[0][1] - scores[1][1]

    return scores, float(margin)


def compute_entropy_and_certainty(
    similarities: list[float],
    tau: float = 0.03,
) -> tuple[float, float]:
    """
    Computes Shannon Entropy H and normalized Certainty U from top-K similarities.
    Default tau=0.03 preserved from 0.1.

    tau is now an explicit argument so calibration.sweep_tau() can vary it.
    The 0.1 default is intentionally kept — Experiment 2 justifies this choice.
    """
    if len(similarities) == 0:
        return 1.0, 0.0
    if len(similarities) == 1:
        return 0.0, 1.0

    sims = np.array(similarities)
    exp_sims = np.exp(sims / tau)
    probs = exp_sims / np.sum(exp_sims)

    eps = 1e-9
    probs = np.clip(probs, eps, 1.0)
    H = float(-np.sum(probs * np.log(probs)))
    U = float(1.0 - H / np.log(len(probs)))
    return H, U


def responsibility_score(
    scaled_top1_sim: float,
    margin: float,
    certainty: float,
    weights: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> float:
    """
    Computes final responsibility score R.
    Default weights (0.70, 0.15, 0.15) preserved from 0.1.

    weights is now an explicit argument so Experiment 4 can sweep alternatives.
    """
    w_sim, w_mar, w_cer = weights
    return float(w_sim * scaled_top1_sim + w_mar * margin + w_cer * certainty)


# ---------------------------------------------------------------------------
# New in 0.2
# ---------------------------------------------------------------------------

def compute_ambiguity(scores: list[tuple[str, float]]) -> float:
    """
    Identity separation ratio: how similar is the 2nd-best match to the best.
    High ambiguity (value near 1.0) means two identities are nearly equally likely.
    Low ambiguity (value near 0.0) means the top match is clearly dominant.

    ambiguity = top2_sim / top1_sim   (0 when only one identity in DB)
    Used by the adaptive threshold engine as the ambiguity penalty signal.
    """
    if len(scores) < 2 or scores[0][1] < 1e-9:
        return 0.0
    return float(min(scores[1][1] / scores[0][1], 1.0))


def compute_open_set_score(
    top1_sim: float,
    stranger_floor: float = 0.60,
) -> dict:
    """
    Evaluates whether a query embedding should be treated as a stranger
    (non-enrolled identity) based on a hard similarity floor.

    The floor of 0.60 is determined by Experiment 3 (stranger rejection
    analysis), which identifies the Pareto-optimal floor value on the
    eval dataset. This value is exposed as a parameter so that value can
    be justified experimentally rather than asserted.

    Returns:
        {
            "is_stranger": bool,
            "top1_sim": float,
            "floor_used": float,
            "margin_to_floor": float,   # positive = above floor (not stranger)
        }
    """
    is_stranger = top1_sim < stranger_floor
    return {
        "is_stranger": is_stranger,
        "top1_sim": top1_sim,
        "floor_used": stranger_floor,
        "margin_to_floor": float(top1_sim - stranger_floor),
    }
