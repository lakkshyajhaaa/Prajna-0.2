"""
core/routing.py — Prajñā 0.3
Responsibility-Guided Routing Engine

Implements the routing score ρ and escalation policy that determine whether
a stage's confidence justifies a terminal decision or whether additional
computation should be invested.

Design rationale (from architecture document):
  The routing score answers: "Does this stage's R justify a terminal decision,
  or should we spend more compute?"

  ρ = R · φ(Q) · ψ(A)

  where:
    R         = responsibility score (existing 0.2 formula)
    φ(Q)      = Q^κ          (quality attenuation; default κ=0.5)
    ψ(A)      = 1 − λ·A    (ambiguity attenuation; default λ=0.3)
    Q         = composite quality score ∈ [0,1]
    A         = ambiguity ratio ∈ [0,1]

Routing policy:
  ρ ≥ ρ_accept  →  ACCEPT  (stage is sufficiently confident; terminate)
  ρ ≤ ρ_reject  →  REJECT  (stranger/insufficient; terminate)
  otherwise     →  ESCALATE (uncertain; spend more compute)

Default thresholds:
  ρ_accept = 0.78
  ρ_reject = 0.42

These are the starting calibration values. Experiment 6 (ablation) sweeps
them to find utility-optimal values on the evaluation dataset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Routing constants
# ---------------------------------------------------------------------------

# Default routing score thresholds
RHO_ACCEPT_DEFAULT: float = 0.78
RHO_REJECT_DEFAULT: float = 0.42

# Default quality attenuation exponent (κ)
KAPPA_DEFAULT: float = 0.5

# Default ambiguity attenuation coefficient (λ)
LAMBDA_DEFAULT: float = 0.3

# Routing actions
RoutingAction = Literal["ACCEPT", "REJECT", "ESCALATE"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RoutingRecord:
    """
    Complete record of routing signal computation for one stage.
    Logged to the audit trail so the routing decision is fully explainable.
    """
    # Stage identifier
    stage:               int           # 1 or 2

    # Input signals
    responsibility:      float         # R score from responsibility_score()
    quality:             float         # Q composite from compute_composite_quality()
    ambiguity:           float         # A ratio from compute_ambiguity()

    # Routing score components
    kappa:               float         # quality exponent used
    lambda_:             float         # ambiguity coefficient used (lambda is reserved)
    phi_q:               float         # φ(Q) = Q^κ
    psi_a:               float         # ψ(A) = 1 − λ·A
    routing_score:       float         # ρ = R · φ(Q) · ψ(A)

    # Decision
    rho_accept:          float         # threshold above which → ACCEPT
    rho_reject:          float         # threshold below which → REJECT
    action:              RoutingAction  # ACCEPT | REJECT | ESCALATE
    escalation_reasons:  list[str] = field(default_factory=list)

    # Timing (set externally)
    latency_ms:          float = 0.0


@dataclass
class RoutingDecision:
    """
    Lightweight summary returned by routing_decision().
    Contains only what downstream code needs; full derivation is in RoutingRecord.
    """
    action:              RoutingAction
    routing_score:       float
    rho_accept:          float
    rho_reject:          float
    escalation_reasons:  list[str]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def routing_score(
    R: float,
    Q: float,
    A: float,
    kappa: float = KAPPA_DEFAULT,
    lambda_: float = LAMBDA_DEFAULT,
) -> tuple[float, float, float]:
    """
    Computes the routing score ρ and its factor components.

    Formula:
        φ(Q) = Q^κ                     (quality attenuation)
        ψ(A) = clip(1 − λ·A, 0, 1)    (ambiguity attenuation)
        ρ    = clip(R · φ(Q) · ψ(A), 0, 1)

    Args:
        R:       Responsibility score ∈ [0, 1]
        Q:       Composite quality score ∈ [0, 1]
        A:       Ambiguity ratio ∈ [0, 1]
        kappa:   Quality exponent. Default 0.5 (concave: penalizes low Q heavily).
        lambda_: Ambiguity coefficient. Default 0.3.

    Returns:
        (rho, phi_q, psi_a) where:
          rho:   Final routing score ∈ [0, 1]
          phi_q: Quality factor φ(Q) = Q^κ
          psi_a: Ambiguity factor ψ(A) = 1 − λ·A
    """
    R      = float(np.clip(R, 0.0, 1.0))
    Q      = float(np.clip(Q, 0.0, 1.0))
    A      = float(np.clip(A, 0.0, 1.0))
    kappa  = float(max(kappa, 1e-6))    # guard against zero exponent
    lambda_ = float(np.clip(lambda_, 0.0, 1.0))

    phi_q = Q ** kappa                             # ∈ [0, 1]
    psi_a = float(np.clip(1.0 - lambda_ * A, 0.0, 1.0))  # ∈ [0.7, 1.0] with defaults
    rho   = float(np.clip(R * phi_q * psi_a, 0.0, 1.0))

    return rho, phi_q, psi_a


def routing_decision(
    rho: float,
    R: float,
    Q: float,
    A: float,
    margin: float,
    rho_accept: float = RHO_ACCEPT_DEFAULT,
    rho_reject: float = RHO_REJECT_DEFAULT,
) -> RoutingDecision:
    """
    Maps routing score ρ to a terminal or escalation action.

    Args:
        rho:        Routing score from routing_score()
        R:          Responsibility score (for reason generation)
        Q:          Quality (for reason generation)
        A:          Ambiguity (for reason generation)
        margin:     Top-1 vs top-2 similarity margin (for reason generation)
        rho_accept: Accept threshold. Default 0.78.
        rho_reject: Reject threshold. Default 0.42.

    Returns:
        RoutingDecision with action, score, and human-readable reasons.
    """
    reasons = _classify_escalation_reasons(rho, R, Q, A, margin, rho_accept, rho_reject)

    if rho >= rho_accept:
        action = "ACCEPT"
    elif rho <= rho_reject:
        action = "REJECT"
    else:
        action = "ESCALATE"

    return RoutingDecision(
        action=action,
        routing_score=rho,
        rho_accept=rho_accept,
        rho_reject=rho_reject,
        escalation_reasons=reasons if action == "ESCALATE" else [],
    )


def build_routing_record(
    stage: int,
    R: float,
    Q: float,
    A: float,
    margin: float,
    rho_accept: float = RHO_ACCEPT_DEFAULT,
    rho_reject: float = RHO_REJECT_DEFAULT,
    kappa: float = KAPPA_DEFAULT,
    lambda_: float = LAMBDA_DEFAULT,
    latency_ms: float = 0.0,
) -> RoutingRecord:
    """
    Full pipeline: computes ρ and builds a RoutingRecord for the audit log.

    Args:
        stage:      Stage number (1 or 2)
        R, Q, A, margin: Inputs from metrics computation
        rho_accept, rho_reject: Routing thresholds
        kappa, lambda_: Routing score parameters
        latency_ms: Embedding + matching latency for this stage

    Returns:
        RoutingRecord with full derivation.
    """
    rho, phi_q, psi_a = routing_score(R, Q, A, kappa=kappa, lambda_=lambda_)
    decision = routing_decision(rho, R, Q, A, margin, rho_accept=rho_accept, rho_reject=rho_reject)

    return RoutingRecord(
        stage=stage,
        responsibility=R,
        quality=Q,
        ambiguity=A,
        kappa=kappa,
        lambda_=lambda_,
        phi_q=phi_q,
        psi_a=psi_a,
        routing_score=rho,
        rho_accept=rho_accept,
        rho_reject=rho_reject,
        action=decision.action,
        escalation_reasons=decision.escalation_reasons,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Explanation helpers
# ---------------------------------------------------------------------------

def _classify_escalation_reasons(
    rho: float,
    R: float,
    Q: float,
    A: float,
    margin: float,
    rho_accept: float,
    rho_reject: float,
) -> list[str]:
    """
    Generates human-readable reasons for why the routing score is in the
    escalation band or why a rejection was triggered.
    """
    reasons: list[str] = []

    if Q < 0.45:
        reasons.append(
            f"Poor image quality (Q={Q:.3f} < 0.45) reduced embedding reliability"
        )
    if A > 0.80:
        reasons.append(
            f"High identity ambiguity (A={A:.3f} > 0.80) — competing identity is closely similar"
        )
    if margin < 0.04:
        reasons.append(
            f"Weak identity separation margin ({margin:.4f} < 0.04) — match is marginal"
        )
    if rho_reject < rho < rho_accept and 0.45 <= R <= 0.80:
        reasons.append(
            f"Borderline responsibility score (R={R:.3f}) in uncertainty band"
        )

    if not reasons:
        reasons.append(
            f"Combined moderate signals: ρ={rho:.3f} in uncertainty band ({rho_reject:.2f}–{rho_accept:.2f})"
        )

    return reasons


def explain_routing(record: RoutingRecord) -> str:
    """
    Generates a one-sentence human-readable explanation of a routing decision.
    Used in the Verify tab and audit log.
    """
    stage_name = f"Stage {record.stage}"

    if record.action == "ACCEPT":
        return (
            f"{stage_name} routing score {record.routing_score:.3f} ≥ ρ_accept={record.rho_accept:.3f}: "
            f"confidence sufficient for terminal ACCEPT decision."
        )
    elif record.action == "REJECT":
        return (
            f"{stage_name} routing score {record.routing_score:.3f} ≤ ρ_reject={record.rho_reject:.3f}: "
            f"similarity too low for any positive decision. Terminal REJECT."
        )
    else:
        reasons_str = "; ".join(record.escalation_reasons[:2]) if record.escalation_reasons else "combined uncertainty"
        return (
            f"{stage_name} routing score {record.routing_score:.3f} in escalation band "
            f"({record.rho_reject:.2f}–{record.rho_accept:.2f}). "
            f"Reason: {reasons_str}. Escalating to Stage {record.stage + 1}."
        )


def compute_responsibility_delta(R1: float, R2: float) -> dict:
    """
    Computes responsibility delta between Stage-1 and Stage-2.

    Returns:
        dict with:
          delta:          R2 - R1
          improved:       bool (Stage 2 was more confident)
          confirmed:      bool (delta approximately 0; stages agreed)
          degraded:       bool (Stage 2 was less confident)
    """
    delta = R2 - R1
    return {
        "delta":     round(float(delta), 5),
        "improved":  delta > 0.02,
        "confirmed": abs(delta) <= 0.02,
        "degraded":  delta < -0.02,
    }
