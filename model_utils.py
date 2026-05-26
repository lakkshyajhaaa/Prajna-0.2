"""
model_utils.py — Prajñā 0.2 (backward-compatible extension of 0.1)

Changes from 0.1:
  - load_models():              UNCHANGED
  - fetch_and_load_database():  UNCHANGED (still loads Kaggle dataset)
  - extract_face_and_embedding(): UNCHANGED signature; now ALSO returns
                                  (detection_prob, landmarks) via new function
  - NEW: extract_face_full()   — returns prob + landmarks for quality module
  - NEW: load_local_database() — loads from database/ folder
  - NEW: enroll_identity()     — adds new identity to database/
  - NEW: save_database_embedding() / load_database_embedding()

All new functions use the same models (MTCNN, InceptionResnetV1) cached by
load_models() so there is zero additional memory overhead.
"""

import torch
import numpy as np
import os
import json
import shutil
from datetime import datetime, timezone
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import streamlit as st

device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to the local enrollment database (separate from Kaggle dataset/)
DATABASE_DIR = os.path.join(os.path.dirname(__file__), "database")


# ---------------------------------------------------------------------------
# Model loading — UNCHANGED from 0.1
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    mtcnn_model = MTCNN(keep_all=True, device=device)   # keep_all=True for multi-face
    resnet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn_model, resnet_model


# ---------------------------------------------------------------------------
# Kaggle dataset DB — UNCHANGED from 0.1 (fetch_and_load_database)
# ---------------------------------------------------------------------------

import kagglehub

@st.cache_data
def fetch_and_load_database(num_classes=None, samples_per_class=1):
    """
    Downloads Kaggle face dataset and builds in-memory embedding database.
    UNCHANGED from Prajñā 0.1. Returns dict of {name: embedding_array (1, 512)}.
    """
    local_dataset_path = "dataset/faces"

    if not os.path.exists(local_dataset_path) or len(os.listdir(local_dataset_path)) == 0:
        try:
            dataset_path = kagglehub.dataset_download("vasukipatel/face-recognition-dataset")
            target_folder = dataset_path
            for root, dirs, files in os.walk(dataset_path):
                if "Original Images" in dirs:
                    target_folder = os.path.join(root, "Original Images")
                    if "Original Images" in os.listdir(target_folder):
                        target_folder = os.path.join(target_folder, "Original Images")
                    break
            shutil.copytree(target_folder, local_dataset_path, dirs_exist_ok=True)
        except Exception as e:
            st.error(f"Failed to download dataset: {e}")
            return {}

    target_folder = local_dataset_path
    database = {}
    classes_loaded = 0
    mtcnn, resnet = load_models()

    if not os.path.exists(target_folder):
        return database

    for person_name in os.listdir(target_folder):
        person_dir = os.path.join(target_folder, person_name)
        if not os.path.isdir(person_dir):
            continue
        images = [f for f in os.listdir(person_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        if not images:
            continue
        success_count = 0
        for img_name in images:
            img_path = os.path.join(person_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                face_tensor = mtcnn(img)
                if face_tensor is not None:
                    # keep_all=True: MTCNN may return (N,3,160,160) for N faces.
                    # Always take only the first (highest-confidence) face so
                    # ResNet always outputs (1, 512), never (N, 512).
                    if isinstance(face_tensor, list):
                        face_tensor = face_tensor[0]          # list of tensors
                    if face_tensor.dim() == 4:
                        face_tensor = face_tensor[0:1]        # (N,3,160,160) → (1,3,160,160)
                    elif face_tensor.dim() == 3:
                        face_tensor = face_tensor.unsqueeze(0)  # (3,160,160) → (1,3,160,160)
                    face_tensor = face_tensor.to(device)
                    with torch.no_grad():
                        emb = resnet(face_tensor).cpu().numpy()  # always (1, 512)

                    if samples_per_class > 1:
                        database[f"{person_name}_{success_count+1}"] = emb
                    else:
                        database[person_name] = emb
                    success_count += 1
                    if success_count >= samples_per_class:
                        break
            except Exception:
                pass
        if success_count > 0:
            classes_loaded += 1
            if num_classes is not None and classes_loaded >= num_classes:
                break

    return database


# ---------------------------------------------------------------------------
# Face extraction — extended for 0.2 (full probe + backward-compatible)
# ---------------------------------------------------------------------------

def extract_face_and_embedding(image, mtcnn_model, resnet_model):
    """
    BACKWARD-COMPATIBLE with 0.1. Returns (cropped_face_np, embedding_np) or (None, None).
    Uses only the first detected face (original 0.1 behavior).
    """
    result = extract_face_full(image, mtcnn_model, resnet_model)
    if result is None:
        return None, None
    return result["face_crop"], result["embedding"]


def extract_face_full(image, mtcnn_model, resnet_model):
    """
    Extended extraction returning all signals needed by quality.py.
    Returns None if no face detected, otherwise a dict with:
      - face_crop:   np.ndarray (H, W, 3) RGB of the primary face
      - embedding:   np.ndarray (1, 512)
      - prob:        float — MTCNN detection confidence for primary face
      - landmarks:   np.ndarray (5, 2) — MTCNN facial landmarks
      - box:         [x1, y1, x2, y2] — bounding box
      - n_faces:     int — total number of detected faces
      - all_boxes:   list of all bounding boxes (for multi-face UI)
      - all_probs:   list of all detection probabilities
    """
    img_arr = np.array(image)
    boxes, probs, landmarks = mtcnn_model.detect(image, landmarks=True)

    if boxes is None or len(boxes) == 0:
        return None

    n_faces = len(boxes)

    # Primary face = highest confidence detection
    primary_idx = int(np.argmax(probs)) if probs is not None else 0

    box  = boxes[primary_idx]
    prob = float(probs[primary_idx]) if probs is not None else 0.5
    lm   = landmarks[primary_idx] if landmarks is not None else None

    h, w = img_arr.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
    face_crop = img_arr[y1:y2, x1:x2]

    # Get embedding via MTCNN preprocessing (returns aligned face tensor)
    face_tensors = mtcnn_model(image)
    if face_tensors is None:
        return None

    # Normalise to a single-face (1, 3, 160, 160) tensor regardless of how
    # many faces MTCNN detected.  keep_all=True may return:
    #   • a list of 3-D tensors  (one per face)
    #   • a stacked 4-D tensor   (N, 3, 160, 160)
    #   • a single 3-D tensor    (3, 160, 160)
    if isinstance(face_tensors, list):
        if len(face_tensors) == 0:
            return None
        face_tensor = face_tensors[primary_idx] if primary_idx < len(face_tensors) else face_tensors[0]
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)   # → (1, 3, 160, 160)
    elif face_tensors.dim() == 4:
        # Stacked multi-face tensor — slice the primary face
        idx = min(primary_idx, face_tensors.shape[0] - 1)
        face_tensor = face_tensors[idx:idx+1]        # → (1, 3, 160, 160)
    else:
        face_tensor = face_tensors.unsqueeze(0)      # (3,160,160) → (1,3,160,160)
    face_tensor = face_tensor.to(device)

    with torch.no_grad():
        embedding = resnet_model(face_tensor).cpu().numpy()

    return {
        "face_crop":  face_crop,
        "embedding":  embedding,
        "prob":       prob,
        "landmarks":  lm,
        "box":        [x1, y1, x2, y2],
        "n_faces":    n_faces,
        "all_boxes":  [b.tolist() for b in boxes],
        "all_probs":  [float(p) for p in probs] if probs is not None else [],
    }


# ---------------------------------------------------------------------------
# Local enrollment database
# ---------------------------------------------------------------------------

def load_local_database() -> dict:
    """
    Loads the persistent enrollment database from database/ folder.
    Returns dict of {person_name: embedding_array (1, 512)}.
    Uses cached embedding.npy if available; otherwise skips (enrollment required first).

    This format is identical to fetch_and_load_database() output so the rest
    of the codebase (metrics, thresholds, decision) works unchanged.
    """
    os.makedirs(DATABASE_DIR, exist_ok=True)
    db = {}
    for person_name in os.listdir(DATABASE_DIR):
        person_dir = os.path.join(DATABASE_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        emb_path = os.path.join(person_dir, "embedding.npy")
        if os.path.exists(emb_path):
            try:
                emb = np.load(emb_path)
                db[person_name] = emb
            except Exception:
                pass
    return db


def save_database_embedding(person_name: str, embedding: np.ndarray) -> None:
    """Saves or overwrites the cached embedding for a person."""
    person_dir = os.path.join(DATABASE_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    np.save(os.path.join(person_dir, "embedding.npy"), embedding)


def enroll_identity(
    person_name: str,
    image_files: list,
    mtcnn_model,
    resnet_model,
    quality_validator=None,
) -> dict:
    """
    Enrolls a new identity into the local database.
    Validates each image before accepting it.
    Average embedding is computed across all accepted images.

    Args:
        person_name:       Name for this identity
        image_files:       List of PIL Images to enroll from
        mtcnn_model:       MTCNN instance
        resnet_model:      InceptionResnetV1 instance
        quality_validator: Optional callable(QualityComponents) → (bool, reasons)

    Returns dict with:
        accepted_count, rejected_count, rejection_log, success (bool), embedding
    """
    from core.quality import compute_composite_quality, is_enrollable

    person_dir = os.path.join(DATABASE_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    accepted_embeddings = []
    rejected_images = []
    accepted_images = []
    log_entries = []

    for i, pil_img in enumerate(image_files):
        img_name = f"img_{i+1:03d}.jpg"
        result = extract_face_full(pil_img, mtcnn_model, resnet_model)

        if result is None:
            rejected_images.append(img_name)
            log_entries.append({
                "image": img_name,
                "accepted": False,
                "reason": "No face detected by MTCNN",
            })
            continue

        # Multi-face check
        if result["n_faces"] > 1:
            rejected_images.append(img_name)
            log_entries.append({
                "image": img_name,
                "accepted": False,
                "reason": f"Multiple faces detected ({result['n_faces']}); enrollment requires single face",
            })
            continue

        # Quality check
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
            log_entries.append({
                "image": img_name,
                "accepted": False,
                "reason": "; ".join(reasons),
                "quality": qc.composite,
            })
            continue

        # Save image copy
        pil_img.save(os.path.join(person_dir, img_name))
        accepted_embeddings.append(result["embedding"])
        accepted_images.append(img_name)
        log_entries.append({
            "image":    img_name,
            "accepted": True,
            "quality":  qc.composite,
        })

    if not accepted_embeddings:
        return {
            "success":         False,
            "person_name":     person_name,
            "accepted_count":  0,
            "rejected_count":  len(image_files),
            "rejection_log":   log_entries,
            "message":         "No images passed quality validation. Enrollment failed.",
        }

    # Average embedding
    avg_emb = np.mean(np.concatenate(accepted_embeddings, axis=0), axis=0, keepdims=True)
    save_database_embedding(person_name, avg_emb)

    # Save metadata
    meta = {
        "person_name":    person_name,
        "enrolled_at":    datetime.now(timezone.utc).isoformat(),
        "n_accepted":     len(accepted_embeddings),
        "n_rejected":     len(rejected_images),
        "accepted_images": accepted_images,
        "rejected_images": rejected_images,
        "log":            log_entries,
    }
    with open(os.path.join(person_dir, "meta.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2)

    return {
        "success":        True,
        "person_name":    person_name,
        "accepted_count": len(accepted_embeddings),
        "rejected_count": len(rejected_images),
        "rejection_log":  log_entries,
        "embedding":      avg_emb,
        "message":        f"Enrolled '{person_name}' with {len(accepted_embeddings)} accepted image(s). Average embedding saved.",
    }


def get_database_meta(person_name: str) -> dict:
    """Loads enrollment metadata for a person."""
    meta_path = os.path.join(DATABASE_DIR, person_name, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        import json
        return json.load(f)
