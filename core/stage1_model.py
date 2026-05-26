"""
core/stage1_model.py — Prajñā 0.3
MobileFaceNet Stage-1 Embedding Wrapper

Provides a lightweight first-pass face embedding using MobileFaceNet,
loaded from an ONNX checkpoint for CPU/GPU portability.

Fallback strategy:
  If the ONNX model is not found at MOBILEFACENET_ONNX_PATH, the module
  automatically falls back to InceptionResnetV1 run at reduced resolution
  (112×112 instead of 160×160). This produces a Stage-1 proxy that is
  faster than full Stage-2 while still exercising the routing machinery.
  The fallback is logged at import time.

Embedding spaces:
  MobileFaceNet:      512-d (MobileFaceNet-512 checkpoint)
  Fallback (IRESNET): 512-d (same architecture as Stage 2, but different
                       preprocessing resolution => different geometry)

All stage-1 embeddings are stored separately from stage-2 embeddings
(see core/database_manager.py). Do NOT mix stage spaces.
"""

import os
import logging
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MOBILEFACENET_ONNX_PATH = os.environ.get(
    "MOBILEFACENET_ONNX_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "mobilefacenet.onnx"),
)

STAGE1_INPUT_SIZE      = 112
FALLBACK_INPUT_SIZE    = 112
STAGE1_EMBEDDING_DIM   = 512
STAGE1_MODEL_NAME_ONNX     = "MobileFaceNet-ONNX"
STAGE1_MODEL_NAME_FALLBACK = "InceptionResnetV1-112px-Fallback"

_ort_session    = None
_fallback_model = None
_using_fallback = None


def _load_onnx_session():
    try:
        import onnxruntime as ort
        onnx_path = os.path.abspath(MOBILEFACENET_ONNX_PATH)
        if not os.path.exists(onnx_path):
            log.info("MobileFaceNet ONNX not found at %s. Using InceptionResnetV1 fallback.", onnx_path)
            return None
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(onnx_path, providers=providers)
        log.info("Stage-1 model loaded: MobileFaceNet ONNX (%s)", onnx_path)
        return session
    except ImportError:
        log.info("onnxruntime not installed. Using InceptionResnetV1 fallback for Stage-1.")
        return None
    except Exception as exc:
        log.warning("Failed to load MobileFaceNet ONNX: %s. Using fallback.", exc)
        return None


def _load_fallback_model():
    from facenet_pytorch import InceptionResnetV1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    log.info("Stage-1 fallback model loaded: InceptionResnetV1/VGGFace2 at %dpx", FALLBACK_INPUT_SIZE)
    return model


def load_stage1_model() -> dict:
    """
    Loads the Stage-1 embedding model with module-level caching.

    Returns dict with:
      - session:    onnxruntime.InferenceSession or None
      - fallback:   InceptionResnetV1 or None
      - using_onnx: bool
      - model_name: str
    """
    global _ort_session, _fallback_model, _using_fallback

    if _using_fallback is not None:
        return {
            "session":    _ort_session,
            "fallback":   _fallback_model,
            "using_onnx": not _using_fallback,
            "model_name": (
                STAGE1_MODEL_NAME_ONNX if not _using_fallback
                else STAGE1_MODEL_NAME_FALLBACK
            ),
        }

    _ort_session = _load_onnx_session()
    if _ort_session is not None:
        _using_fallback = False
        _fallback_model = None
    else:
        _using_fallback = True
        _fallback_model = _load_fallback_model()

    return {
        "session":    _ort_session,
        "fallback":   _fallback_model,
        "using_onnx": not _using_fallback,
        "model_name": (
            STAGE1_MODEL_NAME_ONNX if not _using_fallback
            else STAGE1_MODEL_NAME_FALLBACK
        ),
    }


def _preprocess_face_onnx(face_rgb: np.ndarray) -> np.ndarray:
    import cv2
    face_resized = cv2.resize(face_rgb, (STAGE1_INPUT_SIZE, STAGE1_INPUT_SIZE))
    face_float   = (face_resized.astype(np.float32) - 127.5) / 128.0
    face_chw     = np.transpose(face_float, (2, 0, 1))
    return face_chw[np.newaxis, ...]


def _preprocess_face_fallback(face_rgb: np.ndarray) -> torch.Tensor:
    import cv2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    face_resized = cv2.resize(face_rgb, (FALLBACK_INPUT_SIZE, FALLBACK_INPUT_SIZE))
    face_float   = (face_resized.astype(np.float32) - 127.5) / 128.0
    tensor = torch.from_numpy(
        np.transpose(face_float, (2, 0, 1))[np.newaxis, ...]
    ).to(device)
    return tensor


def extract_stage1_embedding(
    face_rgb: np.ndarray,
    model_state: Optional[dict] = None,
) -> np.ndarray:
    """
    Extracts a Stage-1 L2-normalized embedding from a cropped face.

    Args:
        face_rgb:    uint8 RGB numpy array (H, W, 3), any size.
        model_state: Optional pre-loaded dict from load_stage1_model().

    Returns:
        np.ndarray of shape (1, 512).
    """
    if face_rgb is None or face_rgb.size == 0:
        raise ValueError("extract_stage1_embedding: face_rgb is None or empty")

    if model_state is None:
        model_state = load_stage1_model()

    if model_state["using_onnx"]:
        session    = model_state["session"]
        inp        = _preprocess_face_onnx(face_rgb)
        input_name = session.get_inputs()[0].name
        outputs    = session.run(None, {input_name: inp})
        emb        = outputs[0]
    else:
        fallback = model_state["fallback"]
        tensor   = _preprocess_face_fallback(face_rgb)
        with torch.no_grad():
            emb = fallback(tensor).cpu().numpy()

    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb  = emb / (norm + 1e-9)
    return emb.astype(np.float32)


def batch_stage1_embeddings(
    faces_rgb: list,
    model_state: Optional[dict] = None,
    batch_size: int = 32,
) -> list:
    """
    Extracts Stage-1 embeddings for a list of face crops.

    Args:
        faces_rgb:   List of uint8 RGB numpy arrays.
        model_state: Optional pre-loaded dict.
        batch_size:  ONNX batch size.

    Returns:
        List of np.ndarray, each shape (1, 512).
    """
    if not faces_rgb:
        return []

    if model_state is None:
        model_state = load_stage1_model()

    results = []

    if model_state["using_onnx"]:
        session    = model_state["session"]
        input_name = session.get_inputs()[0].name
        for i in range(0, len(faces_rgb), batch_size):
            batch_faces = faces_rgb[i : i + batch_size]
            batch_inp   = np.concatenate(
                [_preprocess_face_onnx(f) for f in batch_faces], axis=0
            )
            outputs    = session.run(None, {input_name: batch_inp})
            batch_embs = outputs[0]
            norm       = np.linalg.norm(batch_embs, axis=1, keepdims=True)
            batch_embs = batch_embs / (norm + 1e-9)
            for j in range(len(batch_faces)):
                results.append(batch_embs[j : j + 1].astype(np.float32))
    else:
        for face in faces_rgb:
            emb = extract_stage1_embedding(face, model_state=model_state)
            results.append(emb)

    return results


def get_stage1_model_name() -> str:
    """Returns the audit-log name of the currently loaded Stage-1 model."""
    state = load_stage1_model()
    return state["model_name"]
