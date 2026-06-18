"""
core/hierarchy.py — Prajñā 0.3
Hierarchical Inference Orchestrator

This is the primary entry point for 0.3. It runs the full two-stage
hierarchical pipeline and returns a PipelineRecord capturing every
decision, metric, and routing event in a single auditable object.

Pipeline:
  1. Quality gate (pre-stage)
  2. Stage-1 embedding (MobileFaceNet or fallback)
  3. Stage-1 metrics: similarity, margin, entropy, certainty, ambiguity, R
  4. Stage-1 adaptive thresholds
  5. Stage-1 routing score ρ
  6. Routing decision: ACCEPT → terminate | REJECT → terminate | ESCALATE → Stage 2
  7. Stage-2 embedding (InceptionResnetV1) [only if escalated]
  8. Stage-2 metrics
  9. Stage-2 routing score ρ
  10. Final decision: ACCEPT | REJECT | REVIEW (human escalation)

Preserved from 0.2:
  - make_decision() is used as the final decision layer (unchanged)
  - compute_adaptive_thresholds() is called at each stage
  - All QualityComponents from Stage 1 are reused (Q is image-level, not model-level)
  - Multilingual explanation interface is unchanged
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.metrics import (
    calculate_similarity_and_margin,
    compute_entropy_and_certainty,
    responsibility_score,
    compute_ambiguity,
)
from core.quality import QualityComponents
from core.thresholds import ThresholdRecord, compute_adaptive_thresholds
from core.decision import DecisionRecord, make_decision, STRANGER_FLOOR_DEFAULT
from core.routing import (
    RoutingRecord,
    build_routing_record,
    explain_routing,
    compute_responsibility_delta,
    RHO_ACCEPT_DEFAULT,
    RHO_REJECT_DEFAULT,
    KAPPA_DEFAULT,
    LAMBDA_DEFAULT,
)

# ---------------------------------------------------------------------------
# Quality gate constants
# ---------------------------------------------------------------------------

# Below this: hard reject before any model runs
Q_MIN_FLOOR: float = 0.20

# Below this: skip Stage-1 routing decision, force Stage-2 regardless
Q_STAGE2_FORCE_FLOOR: float = 0.40


# ---------------------------------------------------------------------------
# Pipeline record
# ---------------------------------------------------------------------------

@dataclass
class StageMetrics:
    """
    All metric outputs for one inference stage.
    """
    stage:            int
    model_name:       str
    embedding:        Optional[np.ndarray]   # (1, 512); not serialized to JSONL directly
    top1_identity:    str
    top1_sim:         float
    margin:           float
    entropy:          float
    certainty:        float
    ambiguity:        float
    responsibility:   float
    thresholds:       ThresholdRecord
    routing:          RoutingRecord
    latency_ms:       float
    top_scores:       list[tuple[str, float]] = field(default_factory=list)


@dataclass
class PipelineRecord:
    """
    Complete record for one hierarchical inference run.
    Contains all stage metrics, routing decisions, and the final outcome.
    """
    # Pipeline metadata
    stages_run:            list[int]           # e.g. [1] or [1, 2]
    terminal_stage:        int                 # which stage produced the final decision
    hard_rejected_at_gate: bool                # True if Q < Q_MIN_FLOOR
    quality_forced_s2:     bool                # True if Q < Q_STAGE2_FORCE_FLOOR
    quality:               Optional[QualityComponents]

    # Per-stage metrics (None if stage not run)
    stage1:                Optional[StageMetrics]
    stage2:                Optional[StageMetrics]

    # Final decision (from core/decision.py make_decision())
    final_decision:        DecisionRecord

    # Cross-stage analysis (only valid if both stages ran)
    responsibility_delta:  Optional[dict]      # from compute_responsibility_delta()

    # Compute accounting
    compute_units:         float               # relative to Stage-2-only baseline = 1.0
    total_latency_ms:      float

    # Routing explanations (natural language)
    routing_explanation_s1:  str = ""
    routing_explanation_s2:  str = ""


# ---------------------------------------------------------------------------
# Compute cost constants (relative to Stage-2 = 1.0)
# ---------------------------------------------------------------------------

COMPUTE_STAGE1: float = 0.15  # MobileFaceNet vs InceptionResnetV1
COMPUTE_STAGE2: float = 1.00


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def hierarchical_inference(
    image,                        # PIL.Image
    mtcnn_model,                  # MTCNN (from model_utils.load_models)
    resnet_model,                 # InceptionResnetV1 (from model_utils.load_models)
    db_stage1:    dict,           # Stage-1 database from database_manager.load_stage1_database()
    db_stage2:    dict,           # Stage-2 database from database_manager.load_stage2_database()
    stage1_model_state: dict,     # from core.stage1_model.load_stage1_model()
    stranger_floor_s1: float = 0.50,
    stranger_floor_s2: float = STRANGER_FLOOR_DEFAULT,
    rho_accept:    float = RHO_ACCEPT_DEFAULT,
    rho_reject:    float = RHO_REJECT_DEFAULT,
    kappa:         float = KAPPA_DEFAULT,
    lambda_:       float = LAMBDA_DEFAULT,
    tau:           float = 0.03,
    top_k:         int   = 5,
) -> PipelineRecord:
    """
    Runs the full hierarchical inference pipeline.

    Args:
        image:              PIL.Image (RGB)
        mtcnn_model:        MTCNN model for detection
        resnet_model:       InceptionResnetV1 for Stage-2 embedding
        db_stage1:          Stage-1 enrollment database
        db_stage2:          Stage-2 enrollment database
        stage1_model_state: Loaded Stage-1 model dict
        stranger_floor_s1:  Hard similarity floor for Stage 1 open-set rejection
        stranger_floor_s2:  Hard similarity floor for Stage 2 open-set rejection
        rho_accept:         Routing accept threshold
        rho_reject:         Routing reject threshold
        kappa:              Quality attenuation exponent in ρ formula
        lambda_:            Ambiguity attenuation coefficient in ρ formula
        tau:                Softmax temperature for entropy computation
        top_k:              Number of top matches to use for entropy

    Returns:
        PipelineRecord with complete audit trail.
    """
    from model_utils import extract_face_full
    from core.stage1_model import extract_stage1_embedding
    from core.quality import compute_composite_quality

    pipeline_start = time.perf_counter()

    # ----------------------------------------------------------------
    # Step 1: Face detection + quality gate
    # ----------------------------------------------------------------
    t0 = time.perf_counter()
    result = extract_face_full(image, mtcnn_model, resnet_model)

    if result is None:
        # No face detected — hard reject before quality gate
        empty_decision = _make_gate_reject("No face detected by MTCNN")
        return PipelineRecord(
            stages_run=[],
            terminal_stage=0,
            hard_rejected_at_gate=True,
            quality_forced_s2=False,
            quality=None,
            stage1=None,
            stage2=None,
            final_decision=empty_decision,
            responsibility_delta=None,
            compute_units=0.0,
            total_latency_ms=(time.perf_counter() - pipeline_start) * 1000,
        )

    img_arr = np.array(image)
    qc = compute_composite_quality(
        face_rgb=result["face_crop"],
        original_image=img_arr,
        detection_prob=result["prob"],
        landmarks=result["landmarks"],
    )

    # Hard quality gate
    if qc.composite < Q_MIN_FLOOR:
        gate_decision = _make_gate_reject(
            f"Image quality too low for any model (Q={qc.composite:.3f} < {Q_MIN_FLOOR:.2f})"
        )
        return PipelineRecord(
            stages_run=[],
            terminal_stage=0,
            hard_rejected_at_gate=True,
            quality_forced_s2=False,
            quality=qc,
            stage1=None,
            stage2=None,
            final_decision=gate_decision,
            responsibility_delta=None,
            compute_units=0.0,
            total_latency_ms=(time.perf_counter() - pipeline_start) * 1000,
        )

    force_stage2 = qc.composite < Q_STAGE2_FORCE_FLOOR

    # Stage-2 embedding already computed by extract_face_full
    emb_s2 = result["embedding"]

    # ----------------------------------------------------------------
    # Step 2: Stage-1 embedding
    # ----------------------------------------------------------------
    t_s1_start = time.perf_counter()
    stage1_model_name = stage1_model_state.get("model_name", "Stage-1-Unknown")

    if not db_stage1:
        # Stage-1 database empty: skip Stage-1 and go straight to Stage-2
        stage1_metrics = None
        force_stage2 = True
        s1_latency = 0.0
    else:
        emb_s1 = extract_stage1_embedding(
            result["face_crop"], model_state=stage1_model_state
        )
        s1_emb_latency = (time.perf_counter() - t_s1_start) * 1000

        # Step 3: Stage-1 metrics
        if len(db_stage1) > 0:
            scores_s1, margin_s1 = calculate_similarity_and_margin(emb_s1, db_stage1)
        else:
            scores_s1, margin_s1 = [], 0.0

        if not scores_s1:
            force_stage2 = True
            stage1_metrics = None
            s1_latency = s1_emb_latency
        else:
            k1 = min(top_k, len(scores_s1))
            sims_s1 = [s[1] for s in scores_s1[:k1]]
            H1, U1 = compute_entropy_and_certainty(sims_s1, tau=tau)
            amb1 = compute_ambiguity(scores_s1)
            sim1 = scores_s1[0][1]
            name1 = scores_s1[0][0]
            R1 = responsibility_score(sim1, margin_s1, U1)
            tr1 = compute_adaptive_thresholds(H1, k1, qc.composite, amb1, margin_s1)

            s1_total_latency = (time.perf_counter() - t_s1_start) * 1000

            # Step 5: Routing record
            rr1 = build_routing_record(
                stage=1, R=R1, Q=qc.composite, A=amb1, margin=margin_s1,
                rho_accept=rho_accept, rho_reject=rho_reject,
                kappa=kappa, lambda_=lambda_, latency_ms=s1_total_latency,
            )

            stage1_metrics = StageMetrics(
                stage=1, model_name=stage1_model_name,
                embedding=emb_s1, top1_identity=name1,
                top1_sim=sim1, margin=margin_s1,
                entropy=H1, certainty=U1, ambiguity=amb1,
                responsibility=R1, thresholds=tr1, routing=rr1,
                latency_ms=s1_total_latency, top_scores=scores_s1[:k1],
            )
            s1_latency = s1_total_latency

    # ----------------------------------------------------------------
    # Step 6: Stage-1 routing decision
    # ----------------------------------------------------------------
    run_stage2 = force_stage2  # already True if DB empty or quality forced

    if stage1_metrics is not None and not force_stage2:
        rr1 = stage1_metrics.routing
        if rr1.action == "ACCEPT":
            # Terminal ACCEPT at Stage 1
            final_dr = make_decision(
                R=stage1_metrics.responsibility,
                top1_sim=stage1_metrics.top1_sim,
                margin=stage1_metrics.margin,
                entropy=stage1_metrics.entropy,
                certainty=stage1_metrics.certainty,
                ambiguity=stage1_metrics.ambiguity,
                tr=stage1_metrics.thresholds,
                quality=qc,
                predicted_identity=stage1_metrics.top1_identity,
                stranger_floor=stranger_floor_s1,
                threshold_explanation=explain_routing(rr1),
            )
            return PipelineRecord(
                stages_run=[1],
                terminal_stage=1,
                hard_rejected_at_gate=False,
                quality_forced_s2=False,
                quality=qc,
                stage1=stage1_metrics,
                stage2=None,
                final_decision=final_dr,
                responsibility_delta=None,
                compute_units=COMPUTE_STAGE1,
                total_latency_ms=(time.perf_counter() - pipeline_start) * 1000,
                routing_explanation_s1=explain_routing(rr1),
            )
        elif rr1.action == "REJECT":
            # Terminal REJECT at Stage 1
            final_dr = make_decision(
                R=stage1_metrics.responsibility,
                top1_sim=stage1_metrics.top1_sim,
                margin=stage1_metrics.margin,
                entropy=stage1_metrics.entropy,
                certainty=stage1_metrics.certainty,
                ambiguity=stage1_metrics.ambiguity,
                tr=stage1_metrics.thresholds,
                quality=qc,
                predicted_identity=stage1_metrics.top1_identity,
                stranger_floor=stranger_floor_s1,
                threshold_explanation=explain_routing(rr1),
            )
            return PipelineRecord(
                stages_run=[1],
                terminal_stage=1,
                hard_rejected_at_gate=False,
                quality_forced_s2=False,
                quality=qc,
                stage1=stage1_metrics,
                stage2=None,
                final_decision=final_dr,
                responsibility_delta=None,
                compute_units=COMPUTE_STAGE1,
                total_latency_ms=(time.perf_counter() - pipeline_start) * 1000,
                routing_explanation_s1=explain_routing(rr1, name1),
            )
        else:
            # ESCALATE to Stage 2
            run_stage2 = True

    # ----------------------------------------------------------------
    # Step 7: Stage-2 (InceptionResnetV1) — already embedded above
    # ----------------------------------------------------------------
    t_s2_start = time.perf_counter()

    s2_emb_latency_base = (time.perf_counter() - pipeline_start) * 1000

    if not db_stage2:
        # No Stage-2 database — fall through to REVIEW
        gate_decision = _make_gate_reject("Stage-2 database empty; no enrolled identities.")
        total_latency = (time.perf_counter() - pipeline_start) * 1000
        return PipelineRecord(
            stages_run=[1, 2] if stage1_metrics else [2],
            terminal_stage=2,
            hard_rejected_at_gate=False,
            quality_forced_s2=force_stage2,
            quality=qc,
            stage1=stage1_metrics,
            stage2=None,
            final_decision=gate_decision,
            responsibility_delta=None,
            compute_units=COMPUTE_STAGE1 + COMPUTE_STAGE2 if stage1_metrics else COMPUTE_STAGE2,
            total_latency_ms=total_latency,
        )

    scores_s2, margin_s2 = calculate_similarity_and_margin(emb_s2, db_stage2)
    k2 = min(top_k, len(scores_s2))
    sims_s2 = [s[1] for s in scores_s2[:k2]]
    H2, U2 = compute_entropy_and_certainty(sims_s2, tau=tau)
    amb2 = compute_ambiguity(scores_s2)
    sim2 = scores_s2[0][1]
    name2 = scores_s2[0][0]
    R2 = responsibility_score(sim2, margin_s2, U2)
    tr2 = compute_adaptive_thresholds(H2, k2, qc.composite, amb2, margin_s2)

    s2_latency = (time.perf_counter() - t_s2_start) * 1000

    rr2 = build_routing_record(
        stage=2, R=R2, Q=qc.composite, A=amb2, margin=margin_s2,
        rho_accept=rho_accept, rho_reject=rho_reject,
        kappa=kappa, lambda_=lambda_, latency_ms=s2_latency,
    )

    stage2_metrics = StageMetrics(
        stage=2, model_name="InceptionResnetV1/VGGFace2",
        embedding=emb_s2, top1_identity=name2,
        top1_sim=sim2, margin=margin_s2,
        entropy=H2, certainty=U2, ambiguity=amb2,
        responsibility=R2, thresholds=tr2, routing=rr2,
        latency_ms=s2_latency, top_scores=scores_s2[:k2],
    )

    # Step 9: Final decision using Stage-2 metrics
    final_dr = make_decision(
        R=R2, top1_sim=sim2, margin=margin_s2,
        entropy=H2, certainty=U2, ambiguity=amb2,
        tr=tr2, quality=qc, predicted_identity=name2,
        stranger_floor=stranger_floor_s2,
        threshold_explanation=explain_routing(rr2),
    )

    stages_run = ([1, 2] if stage1_metrics is not None else [2])
    r1_val = stage1_metrics.responsibility if stage1_metrics else None
    delta = compute_responsibility_delta(r1_val, R2) if r1_val is not None else None

    compute_used = COMPUTE_STAGE2
    if stage1_metrics is not None:
        compute_used += COMPUTE_STAGE1

    explain_s1 = explain_routing(stage1_metrics.routing, stage1_metrics.top1_identity) if stage1_metrics else ""
    explain_s2 = explain_routing(rr2, name2)

    return PipelineRecord(
        stages_run=stages_run,
        terminal_stage=2,
        hard_rejected_at_gate=False,
        quality_forced_s2=force_stage2,
        quality=qc,
        stage1=stage1_metrics,
        stage2=stage2_metrics,
        final_decision=final_dr,
        responsibility_delta=delta,
        compute_units=compute_used,
        total_latency_ms=(time.perf_counter() - pipeline_start) * 1000,
        routing_explanation_s1=explain_s1,
        routing_explanation_s2=explain_s2,
    )


# ---------------------------------------------------------------------------
# Helper: gate-level rejection record
# ---------------------------------------------------------------------------

def _make_gate_reject(reason: str) -> DecisionRecord:
    """
    Creates a minimal DecisionRecord for gate-level rejections
    (no face detected, image quality below minimum floor).
    """
    from core.thresholds import compute_static_thresholds
    from core.quality import QualityComponents

    dummy_qc = QualityComponents(
        blur=0.0, confidence=0.0, brightness=0.0,
        face_size=0.0, pose=0.0, composite=0.0,
        weights_used={},
    )
    tr = compute_static_thresholds()
    return DecisionRecord(
        decision="REJECT",
        review_reasons=[reason],
        is_stranger=False,
        responsibility=0.0,
        top1_sim=0.0,
        margin=0.0,
        entropy=0.0,
        certainty=0.0,
        ambiguity=0.0,
        thresholds=tr,
        quality=dummy_qc,
        predicted_identity="UNKNOWN",
        stranger_floor=STRANGER_FLOOR_DEFAULT,
        threshold_explanation=reason,
    )
