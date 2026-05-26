"""
core/calibration.py — Prajñā 0.2
Calibration Analysis Utilities

Provides the experimental machinery to justify tau=0.03 and evaluate
confidence calibration. Used by Experiment 2 and the Analysis tab.

Expected Calibration Error (ECE):
    ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    where B_b is the set of samples whose confidence falls in bin b.

In face recognition context:
  - "confidence" = the responsibility score R (or top-1 similarity)
  - "accuracy" = whether the decision was correct (genuine pair accepted)
  - Bins are computed over the [0,1] score range

Reliability Diagram:
    Plots actual accuracy vs predicted confidence per bin.
    A perfectly calibrated system follows the diagonal.
    Below diagonal → overconfident; above diagonal → underconfident.
"""

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Tau sweep
# ---------------------------------------------------------------------------

TAU_CANDIDATES = [0.01, 0.03, 0.05, 0.10]


def sweep_tau(
    similarities_list: list[list[float]],
    tau_list: list[float] = None,
) -> dict:
    """
    Runs entropy computation across multiple tau values on the same set
    of similarity score distributions.

    Args:
        similarities_list: List of top-K similarity vectors (one per query).
        tau_list: Temperature values to evaluate. Defaults to TAU_CANDIDATES.

    Returns:
        Dict mapping tau → {mean_H, std_H, mean_U, std_U, entropy_list}
    """
    from core.metrics import compute_entropy_and_certainty

    if tau_list is None:
        tau_list = TAU_CANDIDATES

    results = {}
    for tau in tau_list:
        entropies   = []
        certainties = []
        for sims in similarities_list:
            H, U = compute_entropy_and_certainty(sims, tau=tau)
            entropies.append(H)
            certainties.append(U)

        results[tau] = {
            "mean_H":       float(np.mean(entropies)),
            "std_H":        float(np.std(entropies)),
            "mean_U":       float(np.mean(certainties)),
            "std_U":        float(np.std(certainties)),
            "entropy_list": entropies,
        }
    return results


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

@dataclass
class ECEResult:
    ece:             float        # Expected Calibration Error [0,1]; lower = better
    n_bins:          int
    bin_confidences: list[float]  # mean confidence per bin
    bin_accuracies:  list[float]  # fraction correct per bin
    bin_counts:      list[int]    # number of samples per bin
    bin_edges:       list[float]  # bin boundary values


def compute_ece(
    confidences: list[float],
    labels: list[int],        # 1 = genuine correct, 0 = incorrect/impostor
    n_bins: int = 10,
) -> ECEResult:
    """
    Computes Expected Calibration Error.

    Args:
        confidences: List of confidence/score values ∈ [0,1] (e.g., R scores)
        labels:      1 if the decision was correct, 0 otherwise
        n_bins:      Number of equal-width bins over [0,1]

    Returns:
        ECEResult with ECE value and per-bin breakdown for reliability diagram.
    """
    confidences = np.array(confidences)
    labels      = np.array(labels)
    n           = len(confidences)

    bin_edges       = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences = []
    bin_accuracies  = []
    bin_counts      = []

    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)  # include 1.0

        count = int(np.sum(mask))
        if count == 0:
            bin_confidences.append(float((lo + hi) / 2))
            bin_accuracies.append(0.0)
            bin_counts.append(0)
            continue

        mean_conf = float(np.mean(confidences[mask]))
        mean_acc  = float(np.mean(labels[mask]))
        bin_confidences.append(mean_conf)
        bin_accuracies.append(mean_acc)
        bin_counts.append(count)

        ece += (count / n) * abs(mean_acc - mean_conf)

    return ECEResult(
        ece=float(ece),
        n_bins=n_bins,
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
        bin_edges=bin_edges.tolist(),
    )


# ---------------------------------------------------------------------------
# Entropy distribution analysis
# ---------------------------------------------------------------------------

def entropy_distribution(
    genuine_similarities: list[list[float]],
    impostor_similarities: list[list[float]],
    tau: float = 0.03,
) -> dict:
    """
    Computes entropy distributions for genuine and impostor query sets.
    Used to visualize tau's effect on separating known vs unknown queries.

    A good tau should:
      - Keep genuine entropy LOW (high certainty for known faces)
      - Keep impostor entropy HIGH (high uncertainty for strangers)

    Args:
        genuine_similarities:  Top-K sim vectors for same-identity pairs
        impostor_similarities: Top-K sim vectors for cross-identity pairs
        tau: Temperature value to evaluate

    Returns:
        {genuine_H: [...], impostor_H: [...], separation: float}
        separation = mean(impostor_H) - mean(genuine_H)  [higher = better]
    """
    from core.metrics import compute_entropy_and_certainty

    genuine_H  = [compute_entropy_and_certainty(s, tau=tau)[0] for s in genuine_similarities]
    impostor_H = [compute_entropy_and_certainty(s, tau=tau)[0] for s in impostor_similarities]

    separation = float(np.mean(impostor_H) - np.mean(genuine_H)) if genuine_H and impostor_H else 0.0

    return {
        "genuine_H":   genuine_H,
        "impostor_H":  impostor_H,
        "mean_genuine":  float(np.mean(genuine_H)) if genuine_H else 0.0,
        "mean_impostor": float(np.mean(impostor_H)) if impostor_H else 0.0,
        "separation":  separation,
        "tau":         tau,
    }
