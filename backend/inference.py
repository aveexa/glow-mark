"""Analyze orchestrator: image bytes → beauty score, feature labels, UI metrics, suggestions.

Pipeline stages (see analyze_image_bytes):
  1. Decode + MediaPipe FaceLandmarker (pass 1), square face normalize (pass 2 remesh)
  2. Realness gate  — CLIP zero-shot, rejects drawings / renders / statues / animals
  3. Pose gate      — Euler decomposition of the pass-2 transformation matrix
  4. Region detection (see region.py) — must precede neutrality
  5. Neutrality gate — blendshapes vs region-weighted thresholds
  6. Beauty MLP (canonicalized 68 landmarks) + display calibration
  7. Geometry features → feature MLP (low/ok/high per feature)
  8. UI metrics, suggestion ranker (fail-soft), JSON payload for /analyze

Gates 2, 3 and 5 read every threshold from data/interim/gate_config.json via gates.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from face_normalize import DEFAULT_OUTPUT_SIZE, square_face_crop
from gates import (
    NEUTRALITY_MODE_GLOBAL,
    autocorrect_roll,
    check_neutrality,
    check_pose,
    check_realness,
    neutrality_mode,
)
from geometry import (
    FEATURE_CONTRACT_VERSION,
    extract_geometry_features,
    GeometryError,
)
from score_calibrate import calibrate_beauty_score, calibration_note
from suggestion_serve import predict_suggestions
from ui_metrics import ui_metrics_from_z

_MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass(frozen=True)
class AnalyzeError(Exception):
    """Structured API error: machine code, HTTP status, optional human details."""

    code: str
    http_status: int = 400
    details: str | None = None
    # Actionable, user-facing text when the generic message is not specific enough
    # (e.g. EXPRESSION_NOT_NEUTRAL -> "Please close your mouth").
    hint: str | None = None


# A commonly-used sparse 68-point subset from MediaPipe Face Mesh (468) for
# compatibility with models trained on 68-landmark pipelines.
MP_468_TO_68: List[int] = [
    162,
    234,
    93,
    58,
    172,
    136,
    149,
    148,
    152,
    377,
    378,
    365,
    397,
    288,
    323,
    454,
    389,
    71,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    301,
    168,
    197,
    5,
    4,
    75,
    97,
    2,
    326,
    305,
    33,
    160,
    158,
    133,
    153,
    144,
    362,
    385,
    387,
    263,
    373,
    380,
    61,
    39,
    37,
    0,
    267,
    269,
    291,
    405,
    314,
    17,
    84,
    181,
    78,
    82,
    13,
    312,
    308,
    317,
    14,
    87,
]

# 17-21  right brow          22-26  left brow
#        36-41 right eye            42-47 left eye
#                 27-30 nose bridge
#                 31-35 nose base
#        48-59 outer mouth
#        60-67 inner mouth
# 0 -------------------- 8 -------------------- 16
#        jaw (ear → chin → ear)


def _mlp_beauty(in_dim: int) -> nn.Module:
    """Helper: beauty MLP topology matching checkpoint keys (0.weight, 3.weight, 6.weight...)."""
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1),
    )


def _mlp_feature(in_dim: int, out_dim: int) -> nn.Module:
    """Helper: feature MLP topology matching checkpoint keys (0.weight, 2.weight, 4.weight...)."""
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, out_dim),
    )


@lru_cache(maxsize=1)
def _load_models():
    """Helper: lazy-load beauty + feature checkpoints with μ/σ (cached once per process)."""
    beauty_ckpt = torch.load(
        _MODELS_DIR / "beauty_landmarks_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    feature_ckpt = torch.load(
        _MODELS_DIR / "feature_geometry_model.pt",
        map_location="cpu",
        weights_only=False,
    )

    beauty_in_dim = int(beauty_ckpt["in_dim"])
    beauty_mu = np.array(beauty_ckpt["mu"], dtype=np.float32).reshape(1, beauty_in_dim)
    beauty_sd = np.array(beauty_ckpt["sd"], dtype=np.float32).reshape(1, beauty_in_dim)
    beauty_model = _mlp_beauty(beauty_in_dim)
    beauty_model.load_state_dict(beauty_ckpt["model_state"])
    beauty_model.eval()
    # Face frame implied by training mu (SCUT ~350px); used to canonicalize serve landmarks.
    mu68 = beauty_mu.reshape(-1, 2)
    beauty_ref_span = float(np.max(mu68.max(axis=0) - mu68.min(axis=0)))
    beauty_ref_center = mu68.mean(axis=0).astype(np.float32)

    feature_feat_cols: List[str] = list(feature_ckpt["feat_cols"])
    feature_label_cols: List[str] = list(feature_ckpt["label_cols"])
    feature_in_dim = len(feature_feat_cols)
    feature_mu = np.array(feature_ckpt["mu"], dtype=np.float32).reshape(1, feature_in_dim)
    feature_sd = np.array(feature_ckpt["sd"], dtype=np.float32).reshape(1, feature_in_dim)

    # Output is 72 = 24 labels * 3 classes (low / ok / high)
    feature_out_dim = int(np.array(list(feature_ckpt["state"].values())[-2]).shape[0])
    feature_model = _mlp_feature(feature_in_dim, feature_out_dim)
    feature_model.load_state_dict(feature_ckpt["state"])
    feature_model.eval()

    return {
        "beauty": {
            "model": beauty_model,
            "mu": beauty_mu,
            "sd": beauty_sd,
            "ref_span": beauty_ref_span,
            "ref_center": beauty_ref_center,
        },
        "feature": {
            "model": feature_model,
            "mu": feature_mu,
            "sd": feature_sd,
            "feat_cols": feature_feat_cols,
            "label_cols": feature_label_cols,
        },
    }


_TASK_PATH = _MODELS_DIR / "face_landmarker_v2_with_blendshapes.task"


@lru_cache(maxsize=1)
def _face_landmarker():
    """Helper: cached MediaPipe FaceLandmarker (Tasks API) for static single-image detection.

    ``num_faces=2`` is deliberate: it is what lets MULTIPLE_FACES_DETECTED fire
    instead of the detector silently picking one face.
    """
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(_TASK_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=2,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.5,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Helper: decode upload bytes to BGR; raise AnalyzeError on corrupt files."""
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise AnalyzeError(code="CORRUPT_FILE", http_status=400, details="Could not decode image.")
    return img


class FaceDetection(NamedTuple):
    """One detected face as returned by the Tasks API.

    ``landmarks`` is (478, 3): indices 0–467 are the canonical mesh topology the
    feature contract uses, 468–477 are iris points appended by this model.
    """

    landmarks: np.ndarray  # (478, 3) normalized [0..1] x/y, model-space z
    blendshapes: Dict[str, float]  # 52 entries keyed by category_name
    matrix: np.ndarray  # (4, 4) facial transformation matrix


def _detect_face(img_bgr: np.ndarray) -> FaceDetection:
    """Helper: detect exactly one face → landmarks + blendshapes + transformation matrix."""
    # The Tasks API expects SRGB; handing it BGR degrades results silently.
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = _face_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    faces = result.face_landmarks or []
    if len(faces) == 0:
        raise AnalyzeError(code="NO_FACE_DETECTED", http_status=400)
    if len(faces) > 1:
        raise AnalyzeError(code="MULTIPLE_FACES_DETECTED", http_status=400)

    lm = faces[0]
    if len(lm) < 468:
        raise AnalyzeError(code="FACE_TOO_ANGLED_OR_SMALL", http_status=400)

    norm = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (478,3)
    blendshapes = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
    matrix = np.array(result.facial_transformation_matrixes[0], dtype=np.float32).reshape(4, 4)
    return FaceDetection(landmarks=norm, blendshapes=blendshapes, matrix=matrix)


def _detect_region(img_bgr: np.ndarray) -> Tuple[Dict[str, float] | None, str]:
    """Helper: region mixture weights for the neutrality gate, or None to use global cuts.

    Fail-soft by design — an unavailable or erroring region model degrades to the
    global thresholds and the request continues.
    """
    if neutrality_mode() == NEUTRALITY_MODE_GLOBAL:
        return None, "Region: skipped (NEUTRALITY_MODE=global)"
    try:
        from region import predict_region_weights
    except Exception as e:  # noqa: BLE001 — region is optional, global cuts still work
        return None, f"Region: unavailable ({e}); using global thresholds"
    try:
        weights = predict_region_weights(img_bgr)
    except Exception as e:  # noqa: BLE001
        return None, f"Region: failed ({e}); using global thresholds"
    top = max(weights, key=weights.__getitem__)
    return weights, f"Region: mixture (top {top} {weights[top]:.2f})"


# SCUT-FBP5500 / beauty checkpoint training canvas (square).
_BEAUTY_CANVAS_SIZE = 350.0


def _canonicalize_beauty_68(
    pts68: np.ndarray,
    ref_span: float,
    ref_center: np.ndarray,
) -> np.ndarray:
    """Align 68 points into the face frame implied by beauty checkpoint mu (span/center)."""
    mn = pts68.min(axis=0)
    mx = pts68.max(axis=0)
    center = (mn + mx) / 2.0
    span = float(max(float((mx - mn).max()), 1e-6))
    return (pts68 - center) * (ref_span / span) + ref_center


def _beauty_features_from_68(
    norm468: np.ndarray,
    ref_span: float,
    ref_center: np.ndarray,
) -> np.ndarray:
    """Build 136-d beauty features from normalized landmarks (resolution-invariant)."""
    # Ignore upload W×H: treat coords as if on the square training canvas, then
    # scale/translate the face bbox to match checkpoint mu span/center.
    pts = norm468[MP_468_TO_68, :2].astype(np.float32) * _BEAUTY_CANVAS_SIZE
    pts = _canonicalize_beauty_68(pts, ref_span, ref_center)
    return pts.reshape(1, -1).astype(np.float32)  # (1,136)


def analyze_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """Run the full analyze pipeline and return the /analyze JSON payload.

    Stages: decode → normalize (fail-soft) → pose gate → beauty + calibrate →
    geometry/feature → UI metrics → suggestions (fail-soft) → response dict.
    """
    models = _load_models()
    img = _decode_image(image_bytes)
    h, w = img.shape[:2]

    # 1. Pass 1: landmarks on original (also used for fail-soft crop bbox).
    det_orig = _detect_face(img)

    normalize_note = "Face normalize: skipped (using original)"
    transform = None
    img_for_score = img
    try:
        cropped = square_face_crop(img, det_orig.landmarks[:468], output_size=DEFAULT_OUTPUT_SIZE)
        if cropped is not None:
            square_bgr, transform = cropped
            # Pass 2: remesh on square crop for isotropic, stable framing.
            # Blendshapes, pose matrix and geometry features all come from here.
            det = _detect_face(square_bgr)
            img_for_score = square_bgr
            normalize_note = f"Face normalize: square {DEFAULT_OUTPUT_SIZE}"
        else:
            det = det_orig
    except Exception:  # noqa: BLE001 — never block analyze on normalize (incl. pass-2 misses)
        det = det_orig
        transform = None
        normalize_note = "Face normalize: skipped (error, using original)"

    # 2. Realness gate — runs on the upload, before any region inference happens,
    #    so non-photographic input never reaches a population model.
    is_real, p_photo = check_realness(img)
    if not is_real:
        raise AnalyzeError(
            code="NOT_A_REAL_FACE",
            http_status=400,
            details=f"CLIP p(photo of a person)={p_photo:.3f} below the configured floor.",
        )

    # 3. Pose gate. Roll inside the limit is corrected by rotating the crop rather
    #    than rejected; the overlay keeps the pre-rotation landmarks.
    pose_ok, pose = check_pose(det.matrix)
    if not pose_ok:
        raise AnalyzeError(
            code="FACE_TOO_ANGLED_OR_SMALL",
            http_status=400,
            details=(
                f"Pose out of frontal band: yaw={pose['yaw_deg']:.1f}, "
                f"pitch={pose['pitch_deg']:.1f}, roll={pose['roll_deg']:.1f}"
            ),
        )

    overlay_det = det
    roll_note = "Roll autocorrect: not needed"
    roll_correction = float(pose["roll_correction_deg"])
    if roll_correction:
        try:
            upright = autocorrect_roll(img_for_score, roll_correction)
            det = _detect_face(upright)
            img_for_score = upright
            roll_note = f"Roll autocorrect: rotated {-roll_correction:.1f} deg"
        except Exception:  # noqa: BLE001 — an uncorrected face still scores
            roll_note = f"Roll autocorrect: skipped (redetect failed, roll={roll_correction:.1f})"

    norm468 = det.landmarks[:468]

    # 4. Region detection, then 5. neutrality gate — region weights pick the
    #    per-population thresholds, so detection has to come first.
    region_weights, region_note = _detect_region(img_for_score)

    neutral, neutrality_hint = check_neutrality(det.blendshapes, region_weights)
    if not neutral:
        raise AnalyzeError(
            code="EXPRESSION_NOT_NEUTRAL",
            http_status=400,
            details="Facial expression is outside the neutral band.",
            hint=neutrality_hint,
        )

    # Beauty score (canonicalized landmarks; independent of upload resolution).
    beauty_x = _beauty_features_from_68(
        norm468,
        models["beauty"]["ref_span"],
        models["beauty"]["ref_center"],
    )  # (1,136)
    beauty_mu = models["beauty"]["mu"]
    beauty_sd = models["beauty"]["sd"]
    beauty_xn = (beauty_x - beauty_mu) / (beauty_sd + 1e-6)
    with torch.no_grad():
        beauty_raw = float(models["beauty"]["model"](torch.from_numpy(beauty_xn)).reshape(-1)[0].item())

    # Geometry features for feature model (shared contract module).
    try:
        feats = extract_geometry_features(norm468)
    except GeometryError as e:
        raise AnalyzeError(code=e.code, http_status=400, details=e.details) from e
    feat_cols: List[str] = models["feature"]["feat_cols"]
    x = np.array([[float(feats[c]) for c in feat_cols]], dtype=np.float32)  # (1,24)
    mu = models["feature"]["mu"]
    sd = models["feature"]["sd"]
    xn = (x - mu) / (sd + 1e-6)

    with torch.no_grad():
        logits = models["feature"]["model"](torch.from_numpy(xn)).cpu().numpy().reshape(-1)

    label_cols: List[str] = models["feature"]["label_cols"]
    if logits.shape[0] % len(label_cols) != 0:
        raise AnalyzeError(
            code="UNKNOWN_ERROR",
            http_status=500,
            details=f"Feature output dim {logits.shape[0]} not compatible with {len(label_cols)} labels.",
        )

    classes_per = logits.shape[0] // len(label_cols)
    scores = logits.reshape(len(label_cols), classes_per)
    probs = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    class_names = ["low", "ok", "high"] if classes_per == 3 else [f"class_{i}" for i in range(classes_per)]

    feature_items = []
    recommendations = []
    for i, name in enumerate(label_cols):
        ci = int(np.argmax(probs[i]))
        conf = float(probs[i][ci])
        cls = class_names[ci]
        feature_items.append({"label": name, "class": cls, "confidence": round(conf, 4)})
        if cls != "ok":
            recommendations.append(f"{name}: {cls}")

    # UI metrics only (soft multi-feature); does not affect model inputs.
    ui = ui_metrics_from_z(xn, feat_cols)

    # Overlay landmarks in original-image normalized space.
    overlay468 = overlay_det.landmarks[:468]
    if transform is not None:
        overlay_norm = transform.remap_landmarks(overlay468)
    else:
        overlay_norm = overlay468
    landmarks = [{"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])} for pt in overlay_norm]

    # Catalog suggestions from ranker when checkpoint is present (Phase 10).
    # One-hots come from percentile cutoffs inside predict_suggestions — not Feature classes.
    try:
        suggestions = predict_suggestions(feats, top_k=4)
    except Exception as e:  # noqa: BLE001 — never break analyze if ranker fails
        suggestions = []
        suggestion_note = f"Suggestion ranker skipped: {e}"
    else:
        suggestion_note = (
            f"Suggestions: ranker top-{len(suggestions)} (percentile one-hots)"
            if suggestions
            else "Suggestions: ranker checkpoint missing (feature strings only)"
        )

    sh, sw = img_for_score.shape[:2]
    beauty_score = calibrate_beauty_score(beauty_raw)
    notes = [
        "Backend inference: MediaPipe FaceLandmarker (478) -> beauty model -> feature model.",
        f"Image size: {w}x{h}",
        f"Score frame: {sw}x{sh}",
        normalize_note,
        calibration_note(),
        f"Feature contract: {FEATURE_CONTRACT_VERSION}",
        f"Pose: yaw={pose['yaw_deg']:.1f}, pitch={pose['pitch_deg']:.1f}, roll={pose['roll_deg']:.1f}",
        roll_note,
        f"Realness: p(photo of a person)={p_photo:.3f}",
        region_note,
        "Neutrality: passed",
        suggestion_note,
    ]

    return {
        "score": beauty_score,
        "score_raw": beauty_raw,
        "metrics": {
            "symmetry": round(ui["symmetry"]),
            "proportions": round(ui["proportions"]),
            "balance": round(ui["balance"]),
        },
        "landmarks": landmarks,
        "overlayTypeHints": {"points": True, "outline": True, "mesh": False},
        "ratios": [{"name": k, "value": float(feats[k]), "idealRange": ""} for k in feat_cols],
        "recommendations": recommendations,
        "recommendation_items": feature_items,
        "suggestions": suggestions,
        "notes": notes,
    }
