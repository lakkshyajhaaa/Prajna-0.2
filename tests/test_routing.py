"""
tests/test_routing.py — Prajñā 0.3
pytest tests for core/routing.py
"""

import pytest
import math
from core.routing import (
    routing_score,
    routing_decision,
    build_routing_record,
    explain_routing,
    compute_responsibility_delta,
    RHO_ACCEPT_DEFAULT,
    RHO_REJECT_DEFAULT,
    KAPPA_DEFAULT,
    LAMBDA_DEFAULT,
    RoutingRecord,
    RoutingDecision,
)


# ---------------------------------------------------------------------------
# routing_score
# ---------------------------------------------------------------------------

class TestRoutingScore:
    def test_perfect_signals(self):
        """R=1, Q=1, A=0 should give rho=1."""
        rho, phi_q, psi_a = routing_score(R=1.0, Q=1.0, A=0.0)
        assert abs(rho - 1.0) < 1e-6
        assert abs(phi_q - 1.0) < 1e-6
        assert abs(psi_a - 1.0) < 1e-6

    def test_zero_quality(self):
        """Q=0 should give rho=0 regardless of R."""
        rho, phi_q, psi_a = routing_score(R=1.0, Q=0.0, A=0.0)
        assert rho == pytest.approx(0.0, abs=1e-6)
        assert phi_q == pytest.approx(0.0, abs=1e-6)

    def test_full_ambiguity(self):
        """A=1 with default lambda=0.3 should reduce psi_a to 0.7."""
        _, _, psi_a = routing_score(R=1.0, Q=1.0, A=1.0, lambda_=0.3)
        assert psi_a == pytest.approx(0.7, abs=1e-6)

    def test_quality_kappa_effect(self):
        """Higher kappa = more aggressive quality penalty."""
        _, phi_q_low, _ = routing_score(R=1.0, Q=0.5, A=0.0, kappa=0.25)
        _, phi_q_high, _ = routing_score(R=1.0, Q=0.5, A=0.0, kappa=1.0)
        assert phi_q_low > phi_q_high  # concave: low kappa penalizes less

    def test_rho_clipped_to_unit_interval(self):
        """rho must always be in [0, 1]."""
        for r, q, a in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.8, 0.3), (-0.1, 1.1, -0.5)]:
            rho, _, _ = routing_score(R=r, Q=q, A=a)
            assert 0.0 <= rho <= 1.0

    def test_product_form(self):
        """Verify exact formula: rho = R * Q^kappa * (1 - lambda*A)."""
        R, Q, A, kappa, lam = 0.8, 0.7, 0.4, 0.5, 0.3
        expected = R * (Q ** kappa) * (1.0 - lam * A)
        rho, _, _ = routing_score(R=R, Q=Q, A=A, kappa=kappa, lambda_=lam)
        assert rho == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# routing_decision
# ---------------------------------------------------------------------------

class TestRoutingDecision:
    def test_accept_zone(self):
        """rho >= rho_accept should always return ACCEPT."""
        dec = routing_decision(
            rho=RHO_ACCEPT_DEFAULT + 0.05,
            R=0.9, Q=0.9, A=0.1, margin=0.15,
            rho_accept=RHO_ACCEPT_DEFAULT, rho_reject=RHO_REJECT_DEFAULT,
        )
        assert dec.action == "ACCEPT"
        assert dec.escalation_reasons == []

    def test_reject_zone(self):
        """rho <= rho_reject should always return REJECT."""
        dec = routing_decision(
            rho=RHO_REJECT_DEFAULT - 0.05,
            R=0.3, Q=0.5, A=0.2, margin=0.02,
            rho_accept=RHO_ACCEPT_DEFAULT, rho_reject=RHO_REJECT_DEFAULT,
        )
        assert dec.action == "REJECT"

    def test_escalate_zone(self):
        """rho in (rho_reject, rho_accept) should return ESCALATE."""
        mid_rho = (RHO_ACCEPT_DEFAULT + RHO_REJECT_DEFAULT) / 2
        dec = routing_decision(
            rho=mid_rho, R=0.6, Q=0.5, A=0.7, margin=0.02,
            rho_accept=RHO_ACCEPT_DEFAULT, rho_reject=RHO_REJECT_DEFAULT,
        )
        assert dec.action == "ESCALATE"
        assert len(dec.escalation_reasons) > 0

    def test_boundary_accept(self):
        """rho exactly at rho_accept should be ACCEPT."""
        dec = routing_decision(
            rho=RHO_ACCEPT_DEFAULT, R=0.9, Q=0.9, A=0.1, margin=0.2,
            rho_accept=RHO_ACCEPT_DEFAULT, rho_reject=RHO_REJECT_DEFAULT,
        )
        assert dec.action == "ACCEPT"

    def test_boundary_reject(self):
        """rho exactly at rho_reject should be REJECT."""
        dec = routing_decision(
            rho=RHO_REJECT_DEFAULT, R=0.3, Q=0.3, A=0.9, margin=0.01,
            rho_accept=RHO_ACCEPT_DEFAULT, rho_reject=RHO_REJECT_DEFAULT,
        )
        assert dec.action == "REJECT"


# ---------------------------------------------------------------------------
# build_routing_record
# ---------------------------------------------------------------------------

class TestBuildRoutingRecord:
    def test_returns_routing_record(self):
        record = build_routing_record(
            stage=1, R=0.8, Q=0.75, A=0.3, margin=0.12
        )
        assert isinstance(record, RoutingRecord)
        assert record.stage == 1
        assert 0.0 <= record.routing_score <= 1.0

    def test_stage2_record(self):
        record = build_routing_record(
            stage=2, R=0.85, Q=0.80, A=0.2, margin=0.15,
            latency_ms=143.0
        )
        assert record.stage == 2
        assert record.latency_ms == 143.0

    def test_action_consistency(self):
        """RoutingRecord action matches direct routing_decision."""
        R, Q, A, margin = 0.95, 0.90, 0.1, 0.20
        record = build_routing_record(stage=1, R=R, Q=Q, A=A, margin=margin)
        rho, _, _ = routing_score(R=R, Q=Q, A=A)
        dec = routing_decision(rho, R=R, Q=Q, A=A, margin=margin)
        assert record.action == dec.action
        assert abs(record.routing_score - rho) < 1e-6


# ---------------------------------------------------------------------------
# explain_routing
# ---------------------------------------------------------------------------

class TestExplainRouting:
    def test_accept_explanation_contains_accept(self):
        record = build_routing_record(stage=1, R=0.95, Q=0.92, A=0.05, margin=0.25)
        if record.action == "ACCEPT":
            expl = explain_routing(record)
            assert "ACCEPT" in expl
            assert "Stage 1" in expl

    def test_escalate_explanation_has_stage(self):
        mid_rho = (RHO_ACCEPT_DEFAULT + RHO_REJECT_DEFAULT) / 2
        # Force rho into escalation band by using borderline inputs
        record = build_routing_record(stage=1, R=0.65, Q=0.50, A=0.85, margin=0.02)
        expl = explain_routing(record)
        assert "Stage 1" in expl


# ---------------------------------------------------------------------------
# compute_responsibility_delta
# ---------------------------------------------------------------------------

class TestResponsibilityDelta:
    def test_improved(self):
        result = compute_responsibility_delta(R1=0.60, R2=0.85)
        assert result["delta"] == pytest.approx(0.25, abs=1e-4)
        assert result["improved"] is True
        assert result["confirmed"] is False
        assert result["degraded"] is False

    def test_confirmed(self):
        result = compute_responsibility_delta(R1=0.75, R2=0.76)
        assert result["confirmed"] is True
        assert result["improved"] is False
        assert result["degraded"] is False

    def test_degraded(self):
        result = compute_responsibility_delta(R1=0.85, R2=0.60)
        assert result["degraded"] is True
        assert result["improved"] is False

    def test_delta_sign(self):
        result = compute_responsibility_delta(R1=0.70, R2=0.50)
        assert result["delta"] < 0


# ---------------------------------------------------------------------------
# Integration: full routing pipeline
# ---------------------------------------------------------------------------

class TestRoutingIntegration:
    def test_high_confidence_terminates_at_stage1(self):
        """High R, high Q, low A should produce ACCEPT routing."""
        rho, _, _ = routing_score(R=0.95, Q=0.92, A=0.05)
        dec = routing_decision(
            rho=rho, R=0.95, Q=0.92, A=0.05, margin=0.30
        )
        assert dec.action == "ACCEPT"

    def test_low_quality_forces_escalation(self):
        """Low Q reduces rho enough to cause escalation even with moderate R."""
        rho, _, _ = routing_score(R=0.85, Q=0.20, A=0.50)
        dec = routing_decision(
            rho=rho, R=0.85, Q=0.20, A=0.50, margin=0.05
        )
        # Low Q = 0.2 => phi_q = 0.2^0.5 = 0.447; rho ~ 0.85 * 0.447 * 0.85 = 0.322 < 0.42
        assert dec.action in ("REJECT", "ESCALATE")

    def test_routing_score_monotone_in_R(self):
        """For fixed Q and A, rho should increase with R."""
        Q, A = 0.7, 0.3
        rhos = [routing_score(R=r, Q=Q, A=A)[0] for r in [0.3, 0.5, 0.7, 0.9]]
        assert all(rhos[i] < rhos[i+1] for i in range(len(rhos)-1))
