"""
utils/logging_utils.py — Prajñā 0.2
Structured audit logger.

Appends one JSONL record per decision to logs/audit_YYYYMMDD.jsonl.
Provides CSV and JSON export for the Audit Log tab.

Every field is documented so the log is self-describing.
"""

import json
import csv
import os
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Optional


LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


@dataclass
class AuditRecord:
    """
    One complete decision audit record.
    All fields map directly to what the system computed — no derived values.
    """
    # Identity context
    predicted_identity:  str
    decision:            str          # ACCEPT | REVIEW | REJECT

    # Core metrics
    similarity:          float        # scaled [0,1]
    margin:              float        # top1 - top2 similarity
    entropy:             float        # H — Shannon entropy
    certainty:           float        # U — normalized certainty
    ambiguity:           float        # top2/top1 ratio
    quality_composite:   float        # Q — composite [0,1]

    # Quality sub-scores
    quality_blur:        float
    quality_confidence:  float
    quality_brightness:  float
    quality_face_size:   float
    quality_pose:        float

    # Threshold context
    t_accept_used:       float        # adaptive accept threshold
    t_review_used:       float        # adaptive review threshold
    t_accept_base:       float        # static baseline
    t_review_base:       float        # static baseline
    stranger_floor:      float        # open-set hard floor

    # Calibration context
    tau:                 float        # softmax temperature used
    weights_sim:         float        # responsibility weight for similarity
    weights_margin:      float        # responsibility weight for margin
    weights_certainty:   float        # responsibility weight for certainty

    # Responsibility score
    responsibility_score: float       # R — final weighted score

    # Model context
    detector_model:      str          # e.g. "MTCNN"
    embedding_model:     str          # e.g. "InceptionResnetV1/VGGFace2"

    # Review explanation (if REVIEW)
    review_reasons:      list[str] = field(default_factory=list)

    # Open-set context
    is_stranger_flag:    bool = False

    # Metadata
    timestamp:           str = ""
    image_filename:      str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def _get_log_path() -> str:
    os.makedirs(LOGS_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    return os.path.join(LOGS_DIR, f"audit_{date_str}.jsonl")


def log_decision(record: AuditRecord) -> None:
    """
    Appends one AuditRecord to today's JSONL log file.
    Each line is a valid JSON object — readable by any JSONL parser.
    """
    path = _get_log_path()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_all_logs() -> list[dict]:
    """
    Loads all JSONL log files from the logs/ directory.
    Returns a list of dicts (one per record), newest-file-first.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    records = []
    log_files = sorted(
        [f for f in os.listdir(LOGS_DIR) if f.endswith(".jsonl")],
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


def export_csv(records: list[dict]) -> str:
    """
    Converts a list of log records to a CSV string.
    Used by the Audit Log tab download button.
    """
    if not records:
        return ""
    import io
    buf = io.StringIO()
    # Flatten list fields (review_reasons → semicolon-separated)
    flat = []
    for r in records:
        row = dict(r)
        row["review_reasons"] = "; ".join(row.get("review_reasons", []))
        flat.append(row)
    writer = csv.DictWriter(buf, fieldnames=flat[0].keys())
    writer.writeheader()
    writer.writerows(flat)
    return buf.getvalue()


def export_json(records: list[dict]) -> str:
    """Returns all records as a formatted JSON string."""
    return json.dumps(records, indent=2, ensure_ascii=False)
