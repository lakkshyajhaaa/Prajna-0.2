"""
utils/logging_utils_v3.py — Prajñā 0.3
Extended Audit Logger for Hierarchical Inference

Extends the 0.2 AuditRecord/log_decision system with:
  - Per-stage metric logging (StageAuditEntry)
  - Full pipeline logging (PipelineAuditRecord)
  - Routing score and action per stage
  - Responsibility delta between stages
  - Compute units consumed
  - Backward-compatible: 0.2 log_decision() still works unchanged

The 0.2 logging_utils.py is NOT modified. This file is the 0.3 extension.
log_pipeline() writes to a separate file: logs/pipeline_YYYYMMDD.jsonl
This keeps 0.2 and 0.3 audit trails cleanly separated.
"""

import json
import csv
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional


LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


# ---------------------------------------------------------------------------
# Stage-level audit entry
# ---------------------------------------------------------------------------

@dataclass
class StageAuditEntry:
    """
    Audit record for one stage of the hierarchical pipeline.
    One of these is created per stage that was executed.
    """
    stage:            int
    model_name:       str

    # Similarity / matching
    top1_identity:    str
    top1_sim:         float
    margin:           float

    # Uncertainty
    entropy:          float
    certainty:        float
    ambiguity:        float

    # Responsibility
    responsibility:   float

    # Routing
    routing_score:    float       # rho
    phi_q:            float       # quality factor Q^kappa
    psi_a:            float       # ambiguity factor 1-lambda*A
    routing_action:   str         # ACCEPT | REJECT | ESCALATE
    routing_reasons:  list[str] = field(default_factory=list)
    rho_accept:       float = 0.78
    rho_reject:       float = 0.42

    # Thresholds
    t_accept_adaptive: float = 0.0
    t_review_adaptive: float = 0.0
    t_accept_base:     float = 0.72
    t_review_base:     float = 0.62
    threshold_delta:   float = 0.0

    # Performance
    latency_ms:       float = 0.0


# ---------------------------------------------------------------------------
# Full pipeline audit record
# ---------------------------------------------------------------------------

@dataclass
class PipelineAuditRecord:
    """
    Complete audit record for one hierarchical inference run.
    Written as one JSONL line to logs/pipeline_YYYYMMDD.jsonl.
    """
    # Outcome
    final_decision:        str         # ACCEPT | REVIEW | REJECT
    predicted_identity:    str
    is_stranger:           bool
    review_reasons:        list[str] = field(default_factory=list)

    # Pipeline topology
    stages_run:            list[int] = field(default_factory=list)
    terminal_stage:        int = 0
    hard_rejected_at_gate: bool = False
    quality_forced_s2:     bool = False

    # Quality (image-level, computed once)
    quality_composite:     float = 0.0
    quality_blur:          float = 0.0
    quality_confidence:    float = 0.0
    quality_brightness:    float = 0.0
    quality_face_size:     float = 0.0
    quality_pose:          float = 0.0

    # Stage records
    stage1: Optional[dict] = None   # serialized StageAuditEntry or None
    stage2: Optional[dict] = None

    # Cross-stage analysis
    responsibility_delta:  Optional[float] = None  # R2 - R1
    delta_improved:        Optional[bool]  = None
    delta_confirmed:       Optional[bool]  = None
    delta_degraded:        Optional[bool]  = None

    # Compute
    compute_units:         float = 0.0
    total_latency_ms:      float = 0.0

    # Explanations
    routing_explanation_s1: str = ""
    routing_explanation_s2: str = ""

    # Model context
    stage1_model_name:     str = ""
    stage2_model_name:     str = "InceptionResnetV1/VGGFace2"
    detector_model:        str = "MTCNN"

    # Calibration
    stranger_floor:        float = 0.60
    tau:                   float = 0.03
    rho_accept:            float = 0.78
    rho_reject:            float = 0.42
    kappa:                 float = 0.50
    lambda_:               float = 0.30

    # Metadata
    timestamp:             str = ""
    image_filename:        str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Build from PipelineRecord
# ---------------------------------------------------------------------------

def build_pipeline_audit_record(
    pipeline_record,       # core.hierarchy.PipelineRecord
    image_filename: str = "",
    stage1_model_name: str = "",
    rho_accept: float = 0.78,
    rho_reject: float = 0.42,
    kappa: float = 0.50,
    lambda_: float = 0.30,
    tau: float = 0.03,
    stranger_floor: float = 0.60,
) -> PipelineAuditRecord:
    """
    Constructs a PipelineAuditRecord from a core.hierarchy.PipelineRecord.
    This is the primary way to create a 0.3 audit record from the pipeline output.

    Args:
        pipeline_record: PipelineRecord returned by hierarchical_inference()
        image_filename:  Original filename for traceability
        ... (routing parameters for metadata)

    Returns:
        PipelineAuditRecord ready for log_pipeline()
    """
    pr = pipeline_record
    dr = pr.final_decision

    # Quality
    qc = pr.quality
    q_composite  = qc.composite   if qc else 0.0
    q_blur       = qc.blur        if qc else 0.0
    q_confidence = qc.confidence  if qc else 0.0
    q_brightness = qc.brightness  if qc else 0.0
    q_face_size  = qc.face_size   if qc else 0.0
    q_pose       = qc.pose        if qc else 0.0

    # Stage 1
    stage1_entry = None
    if pr.stage1 is not None:
        s1 = pr.stage1
        rr1 = s1.routing
        tr1 = s1.thresholds
        entry = StageAuditEntry(
            stage=1, model_name=s1.model_name,
            top1_identity=s1.top1_identity, top1_sim=s1.top1_sim,
            margin=s1.margin, entropy=s1.entropy, certainty=s1.certainty,
            ambiguity=s1.ambiguity, responsibility=s1.responsibility,
            routing_score=rr1.routing_score, phi_q=rr1.phi_q, psi_a=rr1.psi_a,
            routing_action=rr1.action, routing_reasons=rr1.escalation_reasons,
            rho_accept=rr1.rho_accept, rho_reject=rr1.rho_reject,
            t_accept_adaptive=tr1.t_accept, t_review_adaptive=tr1.t_review,
            t_accept_base=tr1.t_accept_base, t_review_base=tr1.t_review_base,
            threshold_delta=tr1.threshold_delta, latency_ms=s1.latency_ms,
        )
        stage1_entry = asdict(entry)

    # Stage 2
    stage2_entry = None
    if pr.stage2 is not None:
        s2 = pr.stage2
        rr2 = s2.routing
        tr2 = s2.thresholds
        entry2 = StageAuditEntry(
            stage=2, model_name=s2.model_name,
            top1_identity=s2.top1_identity, top1_sim=s2.top1_sim,
            margin=s2.margin, entropy=s2.entropy, certainty=s2.certainty,
            ambiguity=s2.ambiguity, responsibility=s2.responsibility,
            routing_score=rr2.routing_score, phi_q=rr2.phi_q, psi_a=rr2.psi_a,
            routing_action=rr2.action, routing_reasons=rr2.escalation_reasons,
            rho_accept=rr2.rho_accept, rho_reject=rr2.rho_reject,
            t_accept_adaptive=tr2.t_accept, t_review_adaptive=tr2.t_review,
            t_accept_base=tr2.t_accept_base, t_review_base=tr2.t_review_base,
            threshold_delta=tr2.threshold_delta, latency_ms=s2.latency_ms,
        )
        stage2_entry = asdict(entry2)

    # Responsibility delta
    r_delta      = pr.responsibility_delta
    delta_val    = r_delta["delta"]    if r_delta else None
    delta_impr   = r_delta["improved"] if r_delta else None
    delta_conf   = r_delta["confirmed"] if r_delta else None
    delta_degr   = r_delta["degraded"] if r_delta else None

    return PipelineAuditRecord(
        final_decision=dr.decision,
        predicted_identity=dr.predicted_identity,
        is_stranger=dr.is_stranger,
        review_reasons=dr.review_reasons,
        stages_run=pr.stages_run,
        terminal_stage=pr.terminal_stage,
        hard_rejected_at_gate=pr.hard_rejected_at_gate,
        quality_forced_s2=pr.quality_forced_s2,
        quality_composite=q_composite,
        quality_blur=q_blur,
        quality_confidence=q_confidence,
        quality_brightness=q_brightness,
        quality_face_size=q_face_size,
        quality_pose=q_pose,
        stage1=stage1_entry,
        stage2=stage2_entry,
        responsibility_delta=delta_val,
        delta_improved=delta_impr,
        delta_confirmed=delta_conf,
        delta_degraded=delta_degr,
        compute_units=pr.compute_units,
        total_latency_ms=pr.total_latency_ms,
        routing_explanation_s1=pr.routing_explanation_s1,
        routing_explanation_s2=pr.routing_explanation_s2,
        stage1_model_name=stage1_model_name,
        stranger_floor=stranger_floor,
        tau=tau,
        rho_accept=rho_accept,
        rho_reject=rho_reject,
        kappa=kappa,
        lambda_=lambda_,
        image_filename=image_filename,
    )


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _get_pipeline_log_path() -> str:
    os.makedirs(LOGS_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    return os.path.join(LOGS_DIR, f"pipeline_{date_str}.jsonl")


def log_pipeline(record: PipelineAuditRecord) -> None:
    """
    Appends one PipelineAuditRecord to today's pipeline JSONL log file.
    """
    path = _get_pipeline_log_path()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_all_pipeline_logs() -> list:
    """
    Loads all pipeline JSONL log files from the logs/ directory.
    Returns a list of dicts (one per record), newest-file-first.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    records = []
    log_files = sorted(
        [f for f in os.listdir(LOGS_DIR) if f.startswith("pipeline_") and f.endswith(".jsonl")],
        reverse=True,
    )
    for fname in log_files:
        path = os.path.join(LOGS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return records


def export_pipeline_csv(records: list) -> str:
    """
    Exports pipeline audit records to CSV.
    Flattens nested stage dicts to top-level columns.
    """
    if not records:
        return ""
    import io
    buf = io.StringIO()
    flat_records = []
    for r in records:
        row = {k: v for k, v in r.items() if k not in ("stage1", "stage2", "review_reasons", "stages_run")}
        row["review_reasons"] = "; ".join(r.get("review_reasons", []))
        row["stages_run"]     = str(r.get("stages_run", []))
        # Flatten stage1
        s1 = r.get("stage1") or {}
        for k, v in s1.items():
            row[f"s1_{k}"] = v
        s2 = r.get("stage2") or {}
        for k, v in s2.items():
            row[f"s2_{k}"] = v
        flat_records.append(row)
    if not flat_records:
        return ""
    writer = csv.DictWriter(buf, fieldnames=flat_records[0].keys())
    writer.writeheader()
    writer.writerows(flat_records)
    return buf.getvalue()


def export_pipeline_json(records: list) -> str:
    """Returns all pipeline records as formatted JSON."""
    return json.dumps(records, indent=2, ensure_ascii=False)
