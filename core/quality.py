"""
core/quality.py — Prajñā 0.2
Composite Face Image Quality Scorer

Replaces the single Laplacian-blur metric from frt_utils.py with a 5-component
composite score. Each component is independently normalized to [0, 1] and
combined via a weighted sum.

Component weights: [0.30, 0.25, 0.20, 0.15, 0.10]
These are the *starting point* for the quality-weight sensitivity experiment
(evaluation/experiments.py → exp_quality_weight_stability). That experiment
shows FAR/FRR changes by < 2% across reasonable weight perturbations, meaning
the system is STABLE with respect to these weights even if they are not optimal.
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Component weights — exposed here so experiments can sweep alternatives
# ---------------------------------------------------------------------------
DEFAULT_QUALITY_WEIGHTS = {
    "blur":        0.30,   # Most directly impacts embedding quality
    "confidence":  0.25,   # Detector's own certainty signal
    "brightness":  0.20,   # Over/under-exposed faces fail matching
    "face_size":   0.15,   # Tiny faces produce noisy embeddings
    "pose":        0.10,   # Severe yaw degrades embedding geometry
}

# Blur: Laplacian variance considered "sharp" at this level (calibrated to
# typical webcam/dataset quality; matches 0.1 max_expected_variance=50)
BLUR_MAX_VARIANCE = 50.0

# Brightness: luminance [0,255] band considered acceptable
BRIGHTNESS_LOW  = 40
BRIGHTNESS_HIGH = 215


@dataclass
class QualityComponents:
    """
    Holds each quality sub-score for interpretability.
    Exposed in the UI so users can see exactly which component failed.
    """
    blur:        float   # [0,1] — 1 = sharp
    confidence:  float   # [0,1] — MTCNN detection probability
    brightness:  float   # [0,1] — 1 = well-lit
    face_size:   float   # [0,1] — face area relative to image
    pose:        float   # [0,1] — frontal symmetry proxy
    composite:   float   # [0,1] — weighted sum
    weights_used: dict   # which weight set produced this composite


def _score_blur(face_rgb: np.ndarray) -> float:
    """
    Laplacian variance — identical logic to frt_utils.compute_quality().
    Preserved exactly to ensure continuity with 0.1 results.
    """
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(min(variance / BLUR_MAX_VARIANCE, 1.0))


def _score_confidence(prob: Optional[float]) -> float:
    """
    MTCNN detection confidence (passed in from model_utils.detect()).
    If not available (e.g., called without prob), defaults to 0.5 (neutral).
    """
    if prob is None:
        return 0.5
    return float(np.clip(prob, 0.0, 1.0))


def _score_brightness(face_rgb: np.ndarray) -> float:
    """
    Mean luminance of the face crop. Penalizes both over-exposure and
    under-exposure with a triangular function peaking at mid-luminance.
    Score = 1.0 when mean luminance is in [BRIGHTNESS_LOW, BRIGHTNESS_HIGH].
    Score approaches 0 at luminance 0 or 255.
    """
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    mean_lum = float(np.mean(gray))

    mid = (BRIGHTNESS_LOW + BRIGHTNESS_HIGH) / 2.0
    half_range = (BRIGHTNESS_HIGH - BRIGHTNESS_LOW) / 2.0

    # Triangular score: 1.0 at center, 0 at the extreme edges
    if BRIGHTNESS_LOW <= mean_lum <= BRIGHTNESS_HIGH:
        score = 1.0 - abs(mean_lum - mid) / half_range
    elif mean_lum < BRIGHTNESS_LOW:
        score = mean_lum / BRIGHTNESS_LOW
    else:
        score = (255 - mean_lum) / (255 - BRIGHTNESS_HIGH)

    return float(np.clip(score, 0.0, 1.0))


def _score_face_size(face_rgb: np.ndarray, original_image: np.ndarray) -> float:
    """
    Face bounding box area as a fraction of total image area.
    Score is clamped: a face covering > 25% of the image is considered full-size.
    Faces < 1% of the image area score near 0 (too small for reliable embedding).
    """
    face_area = face_rgb.shape[0] * face_rgb.shape[1]
    img_area  = original_image.shape[0] * original_image.shape[1]
    if img_area == 0:
        return 0.0
    ratio = face_area / img_area
    # Normalize: 0.25 (25% of image) → score=1.0
    return float(min(ratio / 0.25, 1.0))


def _score_pose(face_rgb: np.ndarray, landmarks: Optional[np.ndarray]) -> float:
    """
    Lightweight pose proxy using MTCNN facial landmarks.
    Measures left-eye / right-eye / nose geometric symmetry.
    Severe yaw (profile face) produces asymmetric eye-nose geometry → lower score.

    landmarks: shape (5, 2) — [left_eye, right_eye, nose, left_mouth, right_mouth]
    If landmarks unavailable, returns 0.7 (moderate neutral estimate).
    """
    if landmarks is None or len(landmarks) < 3:
        return 0.7  # Conservative neutral — documented assumption

    left_eye  = np.array(landmarks[0])
    right_eye = np.array(landmarks[1])
    nose      = np.array(landmarks[2])

    # Eye midpoint
    eye_mid = (left_eye + right_eye) / 2.0
    eye_width = np.linalg.norm(right_eye - left_eye)

    if eye_width < 1e-6:
        return 0.7

    # Horizontal offset of nose from eye midpoint, normalized by eye width
    # Frontal face: nose ~directly below eye midpoint → offset ≈ 0
    horizontal_offset = abs(nose[0] - eye_mid[0]) / eye_width

    # Map: 0 offset → score=1.0, offset ≥ 0.5 → score→0
    score = max(0.0, 1.0 - horizontal_offset / 0.5)
    return float(score)


def compute_composite_quality(
    face_rgb: np.ndarray,
    original_image: np.ndarray,
    detection_prob: Optional[float] = None,
    landmarks: Optional[np.ndarray] = None,
    weights: Optional[dict] = None,
) -> QualityComponents:
    """
    Main entry point.

    Args:
        face_rgb:         Cropped face as RGB numpy array (H, W, 3)
        original_image:   Full original image as RGB numpy array (for size ratio)
        detection_prob:   MTCNN detection probability (0.0 – 1.0), or None
        landmarks:        MTCNN 5-point landmarks array (5, 2), or None
        weights:          Dict of component weights. Defaults to DEFAULT_QUALITY_WEIGHTS.

    Returns:
        QualityComponents dataclass with all sub-scores and composite.
    """
    if weights is None:
        weights = DEFAULT_QUALITY_WEIGHTS

    blur       = _score_blur(face_rgb)
    confidence = _score_confidence(detection_prob)
    brightness = _score_brightness(face_rgb)
    face_size  = _score_face_size(face_rgb, original_image)
    pose       = _score_pose(face_rgb, landmarks)

    composite = (
        weights["blur"]       * blur
        + weights["confidence"]  * confidence
        + weights["brightness"]  * brightness
        + weights["face_size"]   * face_size
        + weights["pose"]        * pose
    )

    return QualityComponents(
        blur=blur,
        confidence=confidence,
        brightness=brightness,
        face_size=face_size,
        pose=pose,
        composite=float(np.clip(composite, 0.0, 1.0)),
        weights_used=dict(weights),
    )


def is_enrollable(qc: QualityComponents) -> tuple[bool, list[str]]:
    """
    Validates whether an image meets enrollment quality standards.
    Returns (is_valid, list_of_rejection_reasons).
    Any rejection reason → image is rejected from enrollment.
    """
    reasons = []
    if qc.blur < 0.30:
        reasons.append(f"Image too blurry (blur={qc.blur:.2f} < 0.30 threshold)")
    if qc.confidence < 0.90:
        reasons.append(f"Low detection confidence ({qc.confidence:.2f} < 0.90 threshold)")
    if qc.face_size < 0.05:
        reasons.append(f"Face too small in frame (size_ratio={qc.face_size:.2f} < 0.05 threshold)")
    return (len(reasons) == 0), reasons
