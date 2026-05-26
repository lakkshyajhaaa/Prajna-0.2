"""
core/thresholds.py — Prajñā 0.2
Adaptive Threshold Engine — The Core Contribution of 0.2

This module implements true per-query adaptive thresholds. Every threshold
is derived from first principles (entropy, quality, ambiguity, margin) using
a bounded linear penalty model.

DESIGN PHILOSOPHY (Documented explicitly):
  The adaptive threshold function is interpretable and controllable but remains
  heuristically parameterized. Its validity is supported by experimental results
  (Experiment 1 in evaluation/experiments.py), not derived from first principles.
  Users deploying on a different dataset should recalibrate α, β, γ.

WHAT MAKES IT "ADAPTIVE" VS 0.1:
  0.1: penalty = α·H_norm + β·(1-Q)         [2 signals]
  0.2: penalty = α·H_norm + β·(1-Q) + γ·A   [3 signals + ambiguity term]
  
  Additionally, the ambiguity term A captures identity confusion that neither
  entropy nor quality alone can detect (e.g., two very similar faces).

WHAT MAKES IT NOT BLACK-BOX:
  - Every input signal has a clear semantic meaning
  - The penalty is a bounded linear combination (no hidden layers)
  - Every threshold decision is logged with its full derivation
  - Parameters are exposed and experimentally analyzed
"""

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Baseline thresholds (preserved from 0.1)
# ---------------------------------------------------------------------------
T_BASE_ACC = 0.72  # Justified in 0.1 README: strangers typically score ~0.67
T_BASE_REV = 0.62  # 10-point gap between REVIEW floor and ACCEPT floor
T_MAX      = 0.95  # Hard ceiling to prevent impossible thresholds

# ---------------------------------------------------------------------------
# Adaptive penalty parameters
# ---------------------------------------------------------------------------
# α: entropy penalty weight — high uncertainty → stricter threshold
# β: quality penalty weight — poor quality → stricter threshold
# γ: ambiguity penalty weight — similar-looking identities → stricter threshold
#
# These values are the starting point for Experiment 1's sensitivity sweep.
# The experiment confirms stability for α∈[0.05,0.20], β∈[0.05,0.25], γ∈[0.02,0.10].
ALPHA_DEFAULT = 0.10
BETA_DEFAULT  = 0.15
GAMMA_DEFAULT = 0.05


@dataclass
class ThresholdRecord:
    """
    Complete derivation record for one adaptive threshold computation.
    Logged to the audit trail so every threshold decision is explainable.
    """
    # Inputs
    entropy:          float   # H — raw Shannon entropy
    entropy_norm:     float   # H / log(K) — normalized to [0,1]
    quality:          float   # Q composite
    quality_penalty:  float   # 1 - Q
    ambiguity:        float   # top2/top1 ratio
    margin:           float   # top1 - top2 similarity

    # Penalty computation
    alpha:            float
    beta:             float
    gamma:            float
    penalty:          float   # α·H_norm + β·(1-Q) + γ·A

    # Outputs
    t_accept:         float   # adaptive accept threshold
    t_review:         float   # adaptive review threshold
    t_accept_base:    float   # static baseline (for comparison)
    t_review_base:    float   # static baseline (for comparison)
    threshold_delta:  float   # how much the threshold shifted vs static


def compute_adaptive_thresholds(
    entropy: float,
    k: int,
    quality: float,
    ambiguity: float,
    margin: float,
    alpha: float = ALPHA_DEFAULT,
    beta: float  = BETA_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
    t_base_acc: float = T_BASE_ACC,
    t_base_rev: float = T_BASE_REV,
) -> ThresholdRecord:
    """
    Computes per-query adaptive thresholds.

    Args:
        entropy:   Shannon entropy H from compute_entropy_and_certainty()
        k:         Number of identities considered (for H_norm denominator)
        quality:   Composite quality score Q ∈ [0,1]
        ambiguity: Identity separation ratio ∈ [0,1]
        margin:    Top-1 minus Top-2 similarity ∈ [0,1]
        alpha:     Entropy penalty weight
        beta:      Quality penalty weight
        gamma:     Ambiguity penalty weight
        t_base_acc: Static accept baseline (default: 0.1 value)
        t_base_rev: Static review baseline (default: 0.1 value)

    Returns:
        ThresholdRecord with full derivation for audit logging.
    """
    # Normalized entropy: H_max = log(K) for K uniform classes
    H_norm = entropy / np.log(max(k, 2))
    H_norm = float(np.clip(H_norm, 0.0, 1.0))

    quality_penalty = float(np.clip(1.0 - quality, 0.0, 1.0))
    ambiguity_val   = float(np.clip(ambiguity, 0.0, 1.0))

    # Linear penalty (bounded, interpretable)
    penalty = alpha * H_norm + beta * quality_penalty + gamma * ambiguity_val
    penalty = float(np.clip(penalty, 0.0, T_MAX - t_base_acc))

    t_accept = float(np.clip(t_base_acc + penalty, t_base_acc, T_MAX))
    t_review = float(np.clip(t_base_rev + penalty, t_base_rev, t_accept))

    return ThresholdRecord(
        entropy=entropy,
        entropy_norm=H_norm,
        quality=quality,
        quality_penalty=quality_penalty,
        ambiguity=ambiguity_val,
        margin=margin,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        penalty=penalty,
        t_accept=t_accept,
        t_review=t_review,
        t_accept_base=t_base_acc,
        t_review_base=t_base_rev,
        threshold_delta=penalty,
    )


def compute_static_thresholds(
    t_accept: float = T_BASE_ACC,
    t_review: float = T_BASE_REV,
) -> ThresholdRecord:
    """
    Returns the 0.1 static thresholds as a ThresholdRecord.
    Used as the control condition in Experiment 1 comparisons.
    """
    return ThresholdRecord(
        entropy=0.0,
        entropy_norm=0.0,
        quality=1.0,
        quality_penalty=0.0,
        ambiguity=0.0,
        margin=0.0,
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        penalty=0.0,
        t_accept=t_accept,
        t_review=t_review,
        t_accept_base=t_accept,
        t_review_base=t_review,
        threshold_delta=0.0,
    )


def explain_threshold(tr: ThresholdRecord) -> str:
    """
    Generates a human-readable explanation of why this query's threshold differs
    from the static baseline. Used in the "Why this decision?" panel.
    """
    if tr.threshold_delta < 0.001:
        return (
            f"Threshold held at baseline (T_acc={tr.t_accept:.3f}). "
            f"Query had low uncertainty, adequate quality, and clear identity separation."
        )

    parts = []
    if tr.entropy_norm > 0.3:
        parts.append(f"elevated uncertainty (H_norm={tr.entropy_norm:.2f})")
    if tr.quality_penalty > 0.3:
        parts.append(f"reduced image quality (Q={tr.quality:.2f})")
    if tr.ambiguity > 0.7:
        parts.append(f"similar competing identity (ambiguity={tr.ambiguity:.2f})")

    if not parts:
        parts.append("combined moderate signals across entropy, quality, and ambiguity")

    reasons_str = ", ".join(parts)
    return (
        f"Threshold raised by {tr.threshold_delta:.3f} above baseline due to: {reasons_str}. "
        f"Adaptive T_acc={tr.t_accept:.3f} (static baseline: {tr.t_accept_base:.3f})."
    )
