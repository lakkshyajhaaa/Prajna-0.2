"""
core/decision.py — Prajñā 0.2
Decision Engine with Full Audit Trail

Wraps the tri-state decision logic (preserved from 0.1) with:
  1. Open-set hard floor (stranger rejection before R-score evaluation)
  2. Structured REVIEW reason generation
  3. DecisionRecord — single object capturing all inputs + outputs
  4. Integration point for the audit logger

The hard stranger floor is evaluated BEFORE the R-score. If top1_sim is
below the floor, the decision is immediately REJECT regardless of R.
This prevents a high-certainty but low-similarity query from being accepted.
"""

import numpy as np
from dataclasses import dataclass, field
from core.quality import QualityComponents
from core.thresholds import ThresholdRecord


# ---------------------------------------------------------------------------
# Review trigger thresholds (interpretable, documented)
# ---------------------------------------------------------------------------

# Similarity: if R is within this band above T_review, it's "close to threshold"
REVIEW_SIM_BAND         = 0.04

# Entropy: above this level, entropy is "elevated"
REVIEW_ENTROPY_THRESHOLD = 0.30

# Ambiguity: above this, competing identity is "dangerously similar"
REVIEW_AMBIGUITY_THRESHOLD = 0.80

# Quality: below this, image quality is "insufficient"
REVIEW_QUALITY_THRESHOLD = 0.50

# Stranger floor (default; also used in metrics.compute_open_set_score)
STRANGER_FLOOR_DEFAULT = 0.60


@dataclass
class DecisionRecord:
    """
    Complete decision record: all inputs and outputs in one auditable object.
    Passed directly to utils.logging_utils.log_decision().
    """
    # Decision output
    decision:         str           # ACCEPT | REVIEW | REJECT
    review_reasons:   list[str]     # Empty unless decision == REVIEW
    is_stranger:      bool          # True if rejected by open-set floor

    # Core inputs
    responsibility:   float         # R score
    top1_sim:         float         # scaled [0,1] cosine similarity
    margin:           float
    entropy:          float
    certainty:        float
    ambiguity:        float

    # Threshold context
    thresholds:       ThresholdRecord

    # Quality context
    quality:          QualityComponents

    # Identity context
    predicted_identity: str
    stranger_floor:     float

    # Threshold explanation (natural language)
    threshold_explanation: str = ""


def make_review_reasons(
    R:         float,
    tr:        ThresholdRecord,
    entropy:   float,
    ambiguity: float,
    quality:   float,
) -> list[str]:
    """
    Generates a structured list of human-readable reasons for a REVIEW decision.
    Each reason corresponds to a measurable condition, not a vague assessment.
    """
    reasons = []

    gap = R - tr.t_review
    if 0 <= gap < REVIEW_SIM_BAND:
        reasons.append(
            f"Responsibility score ({R:.3f}) is within {REVIEW_SIM_BAND:.2f} of review threshold ({tr.t_review:.3f})"
        )

    if entropy > REVIEW_ENTROPY_THRESHOLD:
        reasons.append(
            f"Elevated entropy (H={entropy:.3f} > {REVIEW_ENTROPY_THRESHOLD:.2f}): high identity uncertainty"
        )

    if ambiguity > REVIEW_AMBIGUITY_THRESHOLD:
        reasons.append(
            f"Weak identity separation (ambiguity={ambiguity:.3f} > {REVIEW_AMBIGUITY_THRESHOLD:.2f}): competing identity is closely similar"
        )

    if quality < REVIEW_QUALITY_THRESHOLD:
        reasons.append(
            f"Moderate image quality (Q={quality:.3f} < {REVIEW_QUALITY_THRESHOLD:.2f}): image may affect embedding reliability"
        )

    if tr.threshold_delta > 0.02:
        reasons.append(
            f"Adaptive threshold raised by {tr.threshold_delta:.3f} above baseline due to combined query signals"
        )

    # Fallback: if in REVIEW band but no specific reason triggered, report generic
    if not reasons:
        reasons.append(
            f"Borderline responsibility score (R={R:.3f}): insufficient confidence for autonomous ACCEPT"
        )

    return reasons


def make_decision(
    R:                  float,
    top1_sim:           float,
    margin:             float,
    entropy:            float,
    certainty:          float,
    ambiguity:          float,
    tr:                 ThresholdRecord,
    quality:            QualityComponents,
    predicted_identity: str,
    stranger_floor:     float = STRANGER_FLOOR_DEFAULT,
    threshold_explanation: str = "",
) -> DecisionRecord:
    """
    Main decision entry point for Prajñā 0.2.

    Evaluation order:
      1. Open-set hard floor — if top1_sim < stranger_floor → REJECT immediately
      2. ACCEPT if R >= T_accept
      3. REVIEW if R >= T_review (with reason generation)
      4. REJECT otherwise

    Args:
        R:                  Responsibility score
        top1_sim:           Top-1 scaled similarity
        margin:             Top-1 minus Top-2
        entropy:            Shannon entropy H
        certainty:          Normalized certainty U
        ambiguity:          Identity separation ratio
        tr:                 ThresholdRecord from thresholds.compute_adaptive_thresholds()
        quality:            QualityComponents from quality.compute_composite_quality()
        predicted_identity: Top-1 identity name
        stranger_floor:     Hard similarity floor for open-set rejection
        threshold_explanation: Pre-computed explanation string

    Returns:
        DecisionRecord with full audit trail.
    """
    # --- Step 1: Open-set hard floor ---
    if top1_sim < stranger_floor:
        return DecisionRecord(
            decision="REJECT",
            review_reasons=[
                f"Similarity ({top1_sim:.3f}) below stranger rejection floor ({stranger_floor:.3f}): "
                f"query does not match any enrolled identity"
            ],
            is_stranger=True,
            responsibility=R,
            top1_sim=top1_sim,
            margin=margin,
            entropy=entropy,
            certainty=certainty,
            ambiguity=ambiguity,
            thresholds=tr,
            quality=quality,
            predicted_identity=predicted_identity,
            stranger_floor=stranger_floor,
            threshold_explanation=threshold_explanation,
        )

    # --- Step 2: ACCEPT ---
    if R >= tr.t_accept:
        return DecisionRecord(
            decision="ACCEPT",
            review_reasons=[],
            is_stranger=False,
            responsibility=R,
            top1_sim=top1_sim,
            margin=margin,
            entropy=entropy,
            certainty=certainty,
            ambiguity=ambiguity,
            thresholds=tr,
            quality=quality,
            predicted_identity=predicted_identity,
            stranger_floor=stranger_floor,
            threshold_explanation=threshold_explanation,
        )

    # --- Step 3: REVIEW ---
    if R >= tr.t_review:
        reasons = make_review_reasons(R, tr, entropy, ambiguity, quality.composite)
        return DecisionRecord(
            decision="REVIEW",
            review_reasons=reasons,
            is_stranger=False,
            responsibility=R,
            top1_sim=top1_sim,
            margin=margin,
            entropy=entropy,
            certainty=certainty,
            ambiguity=ambiguity,
            thresholds=tr,
            quality=quality,
            predicted_identity=predicted_identity,
            stranger_floor=stranger_floor,
            threshold_explanation=threshold_explanation,
        )

    # --- Step 4: REJECT (insufficient R score) ---
    return DecisionRecord(
        decision="REJECT",
        review_reasons=[
            f"Responsibility score ({R:.3f}) below review threshold ({tr.t_review:.3f}): "
            f"confidence insufficient for any positive decision"
        ],
        is_stranger=False,
        responsibility=R,
        top1_sim=top1_sim,
        margin=margin,
        entropy=entropy,
        certainty=certainty,
        ambiguity=ambiguity,
        thresholds=tr,
        quality=quality,
        predicted_identity=predicted_identity,
        stranger_floor=stranger_floor,
        threshold_explanation=threshold_explanation,
    )
