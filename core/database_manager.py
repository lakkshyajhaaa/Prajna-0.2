"""
core/database_manager.py — Prajñā 0.3
Dual-Stage Database Manager

Manages two separate embedding databases:
  Stage-1 database  → database/stage1/  (MobileFaceNet or fallback embeddings)
  Stage-2 database  → database/stage2/  (InceptionResnetV1 embeddings)

Both databases use identical on-disk layout:
  database/stage{k}/{person_name}/embedding.npy
  database/stage{k}/{person_name}/meta.json

Why separate databases:
  MobileFaceNet and InceptionResnetV1 operate in different metric spaces.
  A cosine similarity of 0.75 from MobileFaceNet is NOT comparable to 0.75
  from InceptionResnetV1. Mixing embeddings would produce undefined similarity
  scores. Each stage must match queries against embeddings from the same model.

Enrollment policy:
  enroll_identity() runs BOTH models and saves to BOTH databases in one call.
  This ensures both databases are always in sync.

Quality validation:
  Same quality gate as 0.2 (is_enrollable from core/quality.py).
  An identity rejected by quality is not enrolled in either database.

Backward compatibility:
  The existing database/ (0.2 local DB) is preserved and read as Stage-2
  embeddings. Stage-1 DB is new.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE_DIR     = os.path.dirname(os.path.dirname(__file__))
DB_STAGE1_DIR = os.path.join(_BASE_DIR, "database", "stage1")
DB_STAGE2_DIR = os.path.join(_BASE_DIR, "database", "stage2")
DB_LEGACY_DIR = os.path.join(_BASE_DIR, "database")  # 0.2 local DB


def _stage_dir(stage: int) -> str:
    if stage == 1:
        return DB_STAGE1_DIR
    elif stage == 2:
        return DB_STAGE2_DIR
    raise ValueError(f"Unknown stage: {stage}")


# ---------------------------------------------------------------------------
# Load databases
# ---------------------------------------------------------------------------

def _load_stage_database(stage_dir: str) -> dict:
    """
    Loads embeddings from a stage database directory.
    Returns dict {person_name: np.ndarray (1, 512)}.
    """
    os.makedirs(stage_dir, exist_ok=True)
    db = {}
    for person_name in os.listdir(stage_dir):
        person_dir = os.path.join(stage_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        emb_path = os.path.join(person_dir, "embedding.npy")
        if os.path.exists(emb_path):
            try:
                emb = np.load(emb_path)
                if emb.ndim == 1:
                    emb = emb[np.newaxis, :]  # ensure (1, D)
                db[person_name] = emb
            except Exception as exc:
                log.warning("Failed to load embedding for %s: %s", person_name, exc)
    return db


def load_stage1_database() -> dict:
    """
    Loads the Stage-1 (MobileFaceNet) enrollment database.
    Returns dict {person_name: np.ndarray (1, 512)}.
    Returns empty dict if Stage-1 database does not exist yet.
    """
    return _load_stage_database(DB_STAGE1_DIR)


def load_stage2_database() -> dict:
    """
    Loads the Stage-2 (InceptionResnetV1) enrollment database.
    Merges the new stage2/ directory with the legacy 0.2 database/ folder.
    Legacy embeddings are treated as Stage-2 (InceptionResnetV1) embeddings.
    Returns dict {person_name: np.ndarray (1, 512)}.
    """
    db_new    = _load_stage_database(DB_STAGE2_DIR)
    db_legacy = _load_legacy_database()
    # New entries take precedence over legacy
    merged = {**db_legacy, **db_new}
    return merged


def _load_legacy_database() -> dict:
    """
    Loads embeddings from the 0.2 local database (database/<name>/embedding.npy).
    Excludes stage1/ and stage2/ subdirectories.
    """
    if not os.path.exists(DB_LEGACY_DIR):
        return {}
    db = {}
    excluded = {"stage1", "stage2"}
    for person_name in os.listdir(DB_LEGACY_DIR):
        if person_name in excluded:
            continue
        person_dir = os.path.join(DB_LEGACY_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        emb_path = os.path.join(person_dir, "embedding.npy")
        if os.path.exists(emb_path):
            try:
                emb = np.load(emb_path)
                if emb.ndim == 1:
                    emb = emb[np.newaxis, :]
                db[person_name] = emb
            except Exception:
                pass
    return db


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _cosine_sim_scaled(query: np.ndarray, reference: np.ndarray) -> float:
    """Scaled cosine similarity → [0, 1]."""
    q = query.flatten()
    r = reference.flatten()
    cos = np.dot(q, r) / (np.linalg.norm(q) * np.linalg.norm(r) + 1e-9)
    return float((cos + 1.0) / 2.0)


def search_stage1(
    query_embedding: np.ndarray,
    db: Optional[dict] = None,
) -> tuple[list[tuple[str, float]], float]:
    """
    Searches the Stage-1 database for the closest identity.

    Args:
        query_embedding: np.ndarray (1, 512) — from extract_stage1_embedding()
        db:              Optional pre-loaded Stage-1 database. If None, loads from disk.

    Returns:
        (scores, margin) where scores is a list of (name, scaled_sim) sorted desc.
    """
    if db is None:
        db = load_stage1_database()

    if not db:
        return [], 0.0

    scores = [
        (name, _cosine_sim_scaled(query_embedding, emb))
        for name, emb in db.items()
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    margin = (scores[0][1] - scores[1][1]) if len(scores) > 1 else 0.0
    return scores, float(margin)


def search_stage2(
    query_embedding: np.ndarray,
    db: Optional[dict] = None,
) -> tuple[list[tuple[str, float]], float]:
    """
    Searches the Stage-2 database for the closest identity.

    Args:
        query_embedding: np.ndarray (1, 512) — from InceptionResnetV1
        db:              Optional pre-loaded Stage-2 database. If None, loads from disk.

    Returns:
        (scores, margin).
    """
    if db is None:
        db = load_stage2_database()

    if not db:
        return [], 0.0

    scores = [
        (name, _cosine_sim_scaled(query_embedding, emb))
        for name, emb in db.items()
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    margin = (scores[0][1] - scores[1][1]) if len(scores) > 1 else 0.0
    return scores, float(margin)


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

def enroll_identity(
    person_name: str,
    image_files: list,         # list of PIL.Image objects
    mtcnn_model,               # MTCNN from model_utils
    resnet_model,              # InceptionResnetV1 from model_utils
    stage1_model_state: Optional[dict] = None,
) -> dict:
    """
    Enrolls a new identity into BOTH Stage-1 and Stage-2 databases.

    Pipeline per image:
      1. MTCNN detection + face crop
      2. Quality validation (is_enrollable)
      3. Stage-1 embedding (MobileFaceNet or fallback)
      4. Stage-2 embedding (InceptionResnetV1)
      5. Average embeddings across accepted images
      6. Save to database/stage1/<name>/ and database/stage2/<name>/

    Args:
        person_name:        Identity label
        image_files:        List of PIL.Image objects
        mtcnn_model:        MTCNN model instance
        resnet_model:       InceptionResnetV1 model instance
        stage1_model_state: Optional loaded Stage-1 model dict

    Returns:
        dict with: success, person_name, accepted_count, rejected_count,
                   rejection_log, message, embedding_s1, embedding_s2
    """
    from model_utils import extract_face_full
    from core.quality import compute_composite_quality, is_enrollable
    from core.stage1_model import extract_stage1_embedding, load_stage1_model

    if stage1_model_state is None:
        stage1_model_state = load_stage1_model()

    os.makedirs(os.path.join(DB_STAGE1_DIR, person_name), exist_ok=True)
    os.makedirs(os.path.join(DB_STAGE2_DIR, person_name), exist_ok=True)

    accepted_s1 = []
    accepted_s2 = []
    log_entries = []
    accepted_images = []
    rejected_images = []

    for i, pil_img in enumerate(image_files):
        img_name = f"img_{i+1:03d}.jpg"

        # Step 1: MTCNN detection
        result = extract_face_full(pil_img, mtcnn_model, resnet_model)
        if result is None:
            rejected_images.append(img_name)
            log_entries.append({"image": img_name, "accepted": False,
                                 "reason": "No face detected by MTCNN"})
            continue

        if result["n_faces"] > 1:
            rejected_images.append(img_name)
            log_entries.append({"image": img_name, "accepted": False,
                                 "reason": f"Multiple faces ({result['n_faces']}); single face required"})
            continue

        # Step 2: Quality validation
        img_arr = np.array(pil_img)
        qc = compute_composite_quality(
            face_rgb=result["face_crop"],
            original_image=img_arr,
            detection_prob=result["prob"],
            landmarks=result["landmarks"],
        )
        valid, reasons = is_enrollable(qc)
        if not valid:
            rejected_images.append(img_name)
            log_entries.append({"image": img_name, "accepted": False,
                                 "reason": "; ".join(reasons), "quality": qc.composite})
            continue

        # Step 3: Stage-1 embedding
        try:
            emb_s1 = extract_stage1_embedding(result["face_crop"], model_state=stage1_model_state)
        except Exception as exc:
            log.warning("Stage-1 embedding failed for %s/%s: %s", person_name, img_name, exc)
            rejected_images.append(img_name)
            log_entries.append({"image": img_name, "accepted": False,
                                 "reason": f"Stage-1 embedding error: {exc}"})
            continue

        # Step 4: Stage-2 embedding (already computed by extract_face_full)
        emb_s2 = result["embedding"]  # (1, 512) from InceptionResnetV1

        # Save image copy to stage2 dir (reference)
        try:
            pil_img.save(os.path.join(DB_STAGE2_DIR, person_name, img_name))
        except Exception:
            pass

        accepted_s1.append(emb_s1)
        accepted_s2.append(emb_s2)
        accepted_images.append(img_name)
        log_entries.append({"image": img_name, "accepted": True, "quality": qc.composite})

    if not accepted_s1:
        return {
            "success":        False,
            "person_name":    person_name,
            "accepted_count": 0,
            "rejected_count": len(image_files),
            "rejection_log":  log_entries,
            "message":        "No images passed quality validation. Enrollment failed.",
        }

    # Step 5: Average embeddings
    avg_s1 = np.mean(np.concatenate(accepted_s1, axis=0), axis=0, keepdims=True)  # (1, 512)
    avg_s2 = np.mean(np.concatenate(accepted_s2, axis=0), axis=0, keepdims=True)  # (1, 512)

    # Step 6: Save embeddings
    np.save(os.path.join(DB_STAGE1_DIR, person_name, "embedding.npy"), avg_s1)
    np.save(os.path.join(DB_STAGE2_DIR, person_name, "embedding.npy"), avg_s2)

    # Save metadata
    meta = {
        "person_name":      person_name,
        "enrolled_at":      datetime.now(timezone.utc).isoformat(),
        "n_accepted":       len(accepted_s1),
        "n_rejected":       len(rejected_images),
        "accepted_images":  accepted_images,
        "rejected_images":  rejected_images,
        "stage1_model":     stage1_model_state.get("model_name", "unknown"),
        "stage2_model":     "InceptionResnetV1/VGGFace2",
        "log":              log_entries,
    }
    for stage_d in [DB_STAGE1_DIR, DB_STAGE2_DIR]:
        with open(os.path.join(stage_d, person_name, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    return {
        "success":         True,
        "person_name":     person_name,
        "accepted_count":  len(accepted_s1),
        "rejected_count":  len(rejected_images),
        "rejection_log":   log_entries,
        "embedding_s1":    avg_s1,
        "embedding_s2":    avg_s2,
        "message":         (
            f"Enrolled '{person_name}' with {len(accepted_s1)} image(s). "
            f"Saved to Stage-1 and Stage-2 databases."
        ),
    }


# ---------------------------------------------------------------------------
# Update / Delete
# ---------------------------------------------------------------------------

def update_identity(
    person_name: str,
    new_embedding_s1: np.ndarray,
    new_embedding_s2: np.ndarray,
) -> None:
    """
    Overwrites the stored embeddings for an existing identity.
    Does not change metadata or image files.
    """
    for stage, emb in [(1, new_embedding_s1), (2, new_embedding_s2)]:
        d = _stage_dir(stage)
        os.makedirs(os.path.join(d, person_name), exist_ok=True)
        np.save(os.path.join(d, person_name, "embedding.npy"), emb)


def delete_identity(person_name: str) -> dict:
    """
    Removes an identity from both Stage-1 and Stage-2 databases.
    Returns dict with {deleted_from_stages: list[int], not_found_in_stages: list[int]}.
    """
    import shutil
    deleted = []
    not_found = []
    for stage in [1, 2]:
        person_dir = os.path.join(_stage_dir(stage), person_name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
            deleted.append(stage)
        else:
            not_found.append(stage)
    return {"deleted_from_stages": deleted, "not_found_in_stages": not_found}


def get_database_meta(person_name: str, stage: int = 2) -> dict:
    """Returns enrollment metadata for a person from the specified stage database."""
    meta_path = os.path.join(_stage_dir(stage), person_name, "meta.json")
    if not os.path.exists(meta_path):
        # Fallback to legacy 0.2 metadata
        legacy_meta = os.path.join(DB_LEGACY_DIR, person_name, "meta.json")
        if os.path.exists(legacy_meta):
            with open(legacy_meta) as f:
                return json.load(f)
        return {}
    with open(meta_path) as f:
        return json.load(f)


def list_all_identities() -> dict:
    """
    Returns a summary of all enrolled identities across both stages.
    Returns dict {person_name: {in_stage1: bool, in_stage2: bool, meta: dict}}
    """
    s1_names = set(load_stage1_database().keys())
    s2_names = set(load_stage2_database().keys())
    all_names = s1_names | s2_names
    result = {}
    for name in sorted(all_names):
        result[name] = {
            "in_stage1": name in s1_names,
            "in_stage2": name in s2_names,
            "meta": get_database_meta(name),
        }
    return result
