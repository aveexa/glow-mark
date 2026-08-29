"""Orchestrator analyze: MediaPipe + geometry + HTTP calls to model APIs.

SUPERSEDED — v1 pipeline, none of the v2 gates. See the note in app.py.
The ``mp.solutions`` call below only works because requirements.txt pins an old
MediaPipe; backend/inference.py is the maintained equivalent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import httpx
import mediapipe as mp
import numpy as np
import torch

from face_normalize import DEFAULT_OUTPUT_SIZE, square_face_crop
from gcs_utils import resolve_artifact
from geometry import (
    FEATURE_CONTRACT_VERSION,
    assert_frontal,
    estimate_pose,
    extract_geometry_features,
    GeometryError,
)
from score_calibrate import calibrate_beauty_score, calibration_note
from ui_metrics import ui_metrics_from_z


@dataclass(frozen=True)
class AnalyzeError(Exception):
    code: str
    http_status: int = 400
    details: str | None = None


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

_BEAUTY_CANVAS_SIZE = 350.0


def _canonicalize_beauty_68(
    pts68: np.ndarray,
    ref_span: float,
    ref_center: np.ndarray,
) -> np.ndarray:
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
    pts = norm468[MP_468_TO_68, :2].astype(np.float32) * _BEAUTY_CANVAS_SIZE
    pts = _canonicalize_beauty_68(pts, ref_span, ref_center)
    return pts.reshape(1, -1).astype(np.float32)


@lru_cache(maxsize=1)
def _load_stats() -> Dict[str, Any]:
    """Load beauty/feature checkpoints for mu/sd/ref only (no MLP forward)."""
    beauty_path = resolve_artifact(
        local_env="BEAUTY_MODEL_PATH",
        gcs_env="BEAUTY_MODEL_GCS_URI",
        default_dest=Path("/tmp/models/beauty_landmarks_best.pt"),
    )
    feature_path = resolve_artifact(
        local_env="FEATURE_MODEL_PATH",
        gcs_env="FEATURE_MODEL_GCS_URI",
        default_dest=Path("/tmp/models/feature_geometry_model.pt"),
    )
    beauty_ckpt = torch.load(beauty_path, map_location="cpu", weights_only=False)
    feature_ckpt = torch.load(feature_path, map_location="cpu", weights_only=False)

    beauty_in_dim = int(beauty_ckpt["in_dim"])
    beauty_mu = np.array(beauty_ckpt["mu"], dtype=np.float32).reshape(1, beauty_in_dim)
    beauty_sd = np.array(beauty_ckpt["sd"], dtype=np.float32).reshape(1, beauty_in_dim)
    mu68 = beauty_mu.reshape(-1, 2)
    beauty_ref_span = float(np.max(mu68.max(axis=0) - mu68.min(axis=0)))
    beauty_ref_center = mu68.mean(axis=0).astype(np.float32)

    feature_feat_cols: List[str] = list(feature_ckpt["feat_cols"])
    feature_in_dim = len(feature_feat_cols)
    feature_mu = np.array(feature_ckpt["mu"], dtype=np.float32).reshape(1, feature_in_dim)
    feature_sd = np.array(feature_ckpt["sd"], dtype=np.float32).reshape(1, feature_in_dim)

    return {
        "beauty": {
            "mu": beauty_mu,
            "sd": beauty_sd,
            "ref_span": beauty_ref_span,
            "ref_center": beauty_ref_center,
            "in_dim": beauty_in_dim,
        },
        "feature": {
            "mu": feature_mu,
            "sd": feature_sd,
            "feat_cols": feature_feat_cols,
        },
    }


@lru_cache(maxsize=1)
def _mp_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _decode_image(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise AnalyzeError(code="CORRUPT_FILE", http_status=400, details="Could not decode image.")
    return img


def _extract_landmarks_468(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = _mp_face_mesh().process(rgb)

    faces = results.multi_face_landmarks or []
    if len(faces) == 0:
        raise AnalyzeError(code="NO_FACE_DETECTED", http_status=400)
    if len(faces) > 1:
        raise AnalyzeError(code="MULTIPLE_FACES_DETECTED", http_status=400)

    lm = faces[0].landmark
    if len(lm) < 100:
        raise AnalyzeError(code="FACE_TOO_ANGLED_OR_SMALL", http_status=400)

    norm = np.array([[p.x, p.y, getattr(p, "z", 0.0)] for p in lm], dtype=np.float32)
    px = np.empty((len(lm), 2), dtype=np.float32)
    px[:, 0] = norm[:, 0] * float(w)
    px[:, 1] = norm[:, 1] * float(h)
    return norm, px


def _api_urls() -> tuple[str, str, str]:
    beauty = os.environ.get("BEAUTY_URL", "").rstrip("/")
    feature = os.environ.get("FEATURE_URL", "").rstrip("/")
    suggestion = os.environ.get("SUGGESTION_URL", "").rstrip("/")
    if not beauty or not feature or not suggestion:
        raise AnalyzeError(
            code="UNKNOWN_ERROR",
            http_status=500,
            details="Set BEAUTY_URL, FEATURE_URL, and SUGGESTION_URL.",
        )
    return beauty, feature, suggestion


def call_beauty(features_136: list[float]) -> dict:
    beauty_url, _, _ = _api_urls()
    r = httpx.post(
        f"{beauty_url}/v1/beauty/predict",
        json={"features": features_136},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()


def call_feature(features_24: list[float]) -> dict:
    _, feature_url, _ = _api_urls()
    r = httpx.post(
        f"{feature_url}/v1/feature/predict",
        json={"features": features_24},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()


def call_suggestion(feats: dict, top_k: int = 4) -> list:
    _, _, suggestion_url = _api_urls()
    try:
        r = httpx.post(
            f"{suggestion_url}/v1/suggestion/predict",
            json={"features": feats, "top_k": top_k},
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json().get("suggestions") or []
    except Exception:
        return []


def analyze_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    stats = _load_stats()
    img = _decode_image(image_bytes)
    h, w = img.shape[:2]

    # Pass 1: landmarks on original (bbox for fail-soft crop).
    norm_orig, _px468 = _extract_landmarks_468(img)

    normalize_note = "Face normalize: skipped (using original)"
    transform = None
    img_for_score = img
    try:
        cropped = square_face_crop(img, norm_orig, output_size=DEFAULT_OUTPUT_SIZE)
        if cropped is not None:
            square_bgr, transform = cropped
            norm468, _ = _extract_landmarks_468(square_bgr)
            img_for_score = square_bgr
            normalize_note = f"Face normalize: square {DEFAULT_OUTPUT_SIZE}"
        else:
            norm468 = norm_orig
    except Exception:  # noqa: BLE001 — never block analyze on normalize
        norm468 = norm_orig
        transform = None
        normalize_note = "Face normalize: skipped (error, using original)"

    try:
        pose = estimate_pose(norm468)
        assert_frontal(pose)
    except GeometryError as e:
        raise AnalyzeError(code=e.code, http_status=400, details=e.details) from e

    beauty_x = _beauty_features_from_68(
        norm468,
        stats["beauty"]["ref_span"],
        stats["beauty"]["ref_center"],
    )
    beauty_mu = stats["beauty"]["mu"]
    beauty_sd = stats["beauty"]["sd"]
    beauty_xn = (beauty_x - beauty_mu) / (beauty_sd + 1e-6)
    beauty_features = beauty_xn.reshape(-1).astype(float).tolist()

    try:
        feats = extract_geometry_features(norm468)
    except GeometryError as e:
        raise AnalyzeError(code=e.code, http_status=400, details=e.details) from e

    feat_cols: List[str] = stats["feature"]["feat_cols"]
    x = np.array([[float(feats[c]) for c in feat_cols]], dtype=np.float32)
    mu = stats["feature"]["mu"]
    sd = stats["feature"]["sd"]
    xn = (x - mu) / (sd + 1e-6)
    feature_vec = xn.reshape(-1).astype(float).tolist()

    try:
        beauty_out = call_beauty(beauty_features)
        feature_out = call_feature(feature_vec)
    except httpx.HTTPError as e:
        raise AnalyzeError(
            code="UNKNOWN_ERROR",
            http_status=502,
            details=f"Model API call failed: {e}",
        ) from e

    beauty_raw = float(beauty_out.get("score_raw", beauty_out.get("score", 0.0)))
    score = calibrate_beauty_score(beauty_raw)
    recommendation_items = feature_out.get("recommendation_items") or []
    recommendations = feature_out.get("recommendations") or []

    ui = ui_metrics_from_z(xn, feat_cols)

    if transform is not None:
        overlay_norm = transform.remap_landmarks(norm468)
    else:
        overlay_norm = norm468
    landmarks = [{"x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])} for pt in overlay_norm]

    suggestions = call_suggestion({k: float(v) for k, v in feats.items()}, top_k=4)
    suggestion_note = (
        f"Suggestions: ranker top-{len(suggestions)} (percentile one-hots via API)"
        if suggestions
        else "Suggestions: empty or suggestion API unavailable"
    )

    sh, sw = img_for_score.shape[:2]
    notes = [
        "Orchestrator: MediaPipe FaceMesh (468) -> HTTP beauty/feature/suggestion APIs.",
        f"Image size: {w}x{h}",
        f"Score frame: {sw}x{sh}",
        normalize_note,
        calibration_note(),
        f"Feature contract: {FEATURE_CONTRACT_VERSION}",
        f"Pose: yaw={pose['yaw_deg']:.1f}, pitch={pose['pitch_deg']:.1f}",
        suggestion_note,
    ]

    return {
        "score": score,
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
        "recommendation_items": recommendation_items,
        "suggestions": suggestions,
        "notes": notes,
    }
