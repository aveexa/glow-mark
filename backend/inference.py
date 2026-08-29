"""Analyze orchestrator: image bytes → beauty score, feature labels, UI metrics, suggestions.

Pipeline stages (see analyze_image_bytes):
  1. Decode + MediaPipe FaceLandmarker (pass 1), square face normalize (pass 2 remesh)
  2. Realness gate  — CLIP zero-shot, rejects drawings / renders / statues / animals
  3. Pose gate      — Euler decomposition of the pass-2 transformation matrix
  4. Region detection (see region.py) — must precede neutrality
  5. Neutrality gate — blendshapes vs region-weighted thresholds
  6. Beauty MLP (canonicalized 68 landmarks), region-normalized, then calibrated
  7. Geometry features → low/ok/high from region p20/p80 cutoffs (no Feature MLP)
  8. UI metrics, suggestion ranker (fail-soft), JSON payload for /analyze

Gates 2, 3 and 5 read every threshold from data/interim/gate_config.json via gates.py.
Stages 6–8 read population norms from data/processed/region_reference_stats.csv via
region_stats.py, and fall back to the pre-region behaviour when it is absent.
"""

from __future__ import annotations

import math
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
    NEUTRALITY_THRESHOLD_MODE_REGION,
    autocorrect_roll,
    beauty_region_normalize,
    check_neutrality,
    check_pose,
    check_realness,
    neutrality_threshold_mode,
)
from geometry import (
    FEATURE_COLS,
    FEATURE_CONTRACT_VERSION,
    extract_geometry_features,
    GeometryError,
)
from region_stats import beauty_stats, mixture_stats
from score_calibrate import calibrate_beauty_score, calibration_note
from suggestion_serve import (
    RESPONSE_EXCLUDED_FEATURES,
    classes_from_thresholds,
    load_threshold_rules,
    predict_suggestions,
)
from ui_metrics import ui_metrics_from_features

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


@lru_cache(maxsize=1)
def _load_models():
    """Helper: lazy-load the beauty checkpoint with μ/σ (cached once per process).

    feature_geometry_model.pt is deliberately not loaded. It approximated a
    percentile rule that the region-conditioned p20/p80 cutoffs now compute
    exactly; the checkpoint stays on disk but is off the serve path.
    """
    beauty_ckpt = torch.load(
        _MODELS_DIR / "beauty_landmarks_best.pt",
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

    return {
        "beauty": {
            "model": beauty_model,
            "mu": beauty_mu,
            "sd": beauty_sd,
            "ref_span": beauty_ref_span,
            "ref_center": beauty_ref_center,
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


@lru_cache(maxsize=1)
def _threshold_rules():
    """Helper: static p20/p80 rules CSV, cached — only chin_length_ratio still needs it."""
    return load_threshold_rules()


def _class_confidence(
    value: float,
    cls: str,
    stats: Dict[str, float] | None,
) -> float:
    """How strongly a low/ok/high call holds, as a percentile under the region's norm.

    The class itself is now an exact p20/p80 rule, so this reports position rather
    than model certainty: a value at the 95th percentile of its region reports 0.95
    for "high", and an "ok" value reports how centrally it sits (1.0 at the median).
    Without region statistics there is nothing to be confident about, so 0.5.
    """
    if not stats or stats["sigma"] <= 0.0:
        return 0.5
    z = (float(value) - stats["mu"]) / stats["sigma"]
    percentile = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    if cls == "high":
        return percentile
    if cls == "low":
        return 1.0 - percentile
    return 1.0 - 2.0 * abs(percentile - 0.5)


def _region_normalize_beauty(
    beauty_raw: float,
    region_weights: Dict[str, float] | None,
) -> Tuple[float, str]:
    """Re-express the raw beauty score against the region's own distribution.

    Disabled by default (beauty.region_normalize in gate_config.json). E4b tested it
    against MEBeauty human ratings and it reduced agreement rather than improving it:
    pooled Pearson 0.236 raw against 0.209 normalised, bootstrap 95% CI
    [-0.038, -0.016]. The path is kept, not deleted — one dataset is thin evidence to
    remove a mechanism on, and beauty_score_raw stays in the reference table.

    When enabled, maps the raw score onto the global arm's scale — same z, global mean
    and spread — so the existing display calibration stays valid while the score
    becomes relative to the user's population. Returns the (possibly unchanged) raw
    score and a note for the API payload.

    This governs the beauty score alone. Region conditioning of the geometry features,
    the p20/p80 classes and the UI gauges is a separate mechanism and is unaffected.
    """
    if not beauty_region_normalize():
        return beauty_raw, "Beauty: raw score (region normalisation off; see E4b)"

    region = beauty_stats(region_weights)
    reference = beauty_stats(None)
    if not region or not reference or region["sigma"] <= 0.0:
        return beauty_raw, "Beauty: global calibration (no region reference statistics)"
    z = (beauty_raw - region["mu"]) / region["sigma"]
    return (
        reference["mu"] + z * reference["sigma"],
        f"Beauty: region-normalized (z={z:+.2f} vs region mu={region['mu']:.2f})",
    )


def _detect_region(img_bgr: np.ndarray) -> Tuple[Dict[str, float] | None, str]:
    """Helper: region mixture weights, or None to fall back to the global arm.

    Runs regardless of NEUTRALITY_THRESHOLD_MODE — that variable scopes only the neutrality
    thresholds, while the reference norms behind the classes, gauges and beauty
    score need the mixture either way. Fail-soft by design: an unavailable or
    erroring region model degrades to the global arm and the request continues.
    """
    try:
        from region import predict_region_weights
    except Exception as e:  # noqa: BLE001 — region is optional, the global arm still works
        return None, f"Region: unavailable ({e}); using global norms"
    try:
        weights = predict_region_weights(img_bgr)
    except Exception as e:  # noqa: BLE001
        return None, f"Region: failed ({e}); using global norms"
    top = max(weights, key=weights.__getitem__)
    mode = neutrality_threshold_mode()
    scope = (
        "norms + neutrality" if mode == NEUTRALITY_THRESHOLD_MODE_REGION
        else "norms only (NEUTRALITY_THRESHOLD_MODE=global)"
    )
    return weights, f"Region: mixture (top {top} {weights[top]:.2f}); applied to {scope}"


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

    Stages: decode → normalize (fail-soft) → realness / pose / neutrality gates →
    region mixture → beauty (region-normalized, calibrated) → geometry classes from
    region cutoffs → UI metrics → suggestions (fail-soft) → response dict.
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

    # 6. Beauty score (canonicalized landmarks; independent of upload resolution),
    #    re-expressed against the region's own raw-score distribution before the
    #    display calibration, so the number means "relative to this population".
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

    beauty_for_display, beauty_note = _region_normalize_beauty(beauty_raw, region_weights)

    # 7. Geometry features → low/ok/high straight from the region p20/p80 cutoffs.
    try:
        feats = extract_geometry_features(norm468)
    except GeometryError as e:
        raise AnalyzeError(code=e.code, http_status=400, details=e.details) from e

    feat_cols: List[str] = list(FEATURE_COLS)
    region_norms = mixture_stats(region_weights) or {}
    classes = classes_from_thresholds(feats, _threshold_rules(), region_weights)

    # The contract still carries all 24 features and the ranker still scores all 24;
    # only what the user sees is filtered. See RESPONSE_EXCLUDED_FEATURES.
    reported_cols = [c for c in feat_cols if c not in RESPONSE_EXCLUDED_FEATURES]

    feature_items = []
    recommendations = []
    for name in reported_cols:
        cls = classes[name]
        feature_items.append({
            "label": name,
            "class": cls,
            "confidence": round(_class_confidence(feats[name], cls, region_norms.get(name)), 4),
        })
        if cls != "ok":
            recommendations.append(f"{name}: {cls}")

    # 8. UI metrics only (soft multi-feature); does not affect any model input.
    ui = ui_metrics_from_features(feats, feat_cols, region_weights)

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
        suggestions = predict_suggestions(feats, top_k=4, region_weights=region_weights)
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
    beauty_score = calibrate_beauty_score(beauty_for_display)
    notes = [
        "Backend inference: MediaPipe FaceLandmarker (478) -> beauty model -> region p20/p80 classes.",
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
        beauty_note,
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
        "ratios": [{"name": k, "value": float(feats[k]), "idealRange": ""} for k in reported_cols],
        "recommendations": recommendations,
        "recommendation_items": feature_items,
        "suggestions": suggestions,
        "notes": notes,
    }
