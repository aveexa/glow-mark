"""Feature contract v1 — pose gate + 24 locked geometry features from MediaPipe 468."""

from __future__ import annotations

import math
from typing import Dict, Mapping, MutableMapping

import numpy as np

FEATURE_CONTRACT_VERSION: str = "v1"
POSE_YAW_MAX_DEG: float = 25.0
POSE_PITCH_MAX_DEG: float = 25.0
EXPECTED_NOSE_EYE_CHIN_FRAC: float = 0.37

FEATURE_COLS: tuple[str, ...] = (
    "symmetry_error",
    "face_aspect_ratio",
    "midface_length_ratio",
    "lowerface_length_ratio",
    "jaw_width_ratio",
    "jaw_angle_sharpness",
    "chin_length_ratio",
    "chin_width_ratio",
    "cheekbone_width_ratio",
    "lower_cheek_ratio",
    "eye_openness_ratio",
    "eye_size_ratio",
    "eye_spacing_ratio",
    "eye_tilt_deg",
    "brow_height_ratio",
    "brow_tilt_deg",
    "nose_width_ratio",
    "nose_length_ratio",
    "nose_tip_deviation_ratio",
    "mouth_width_ratio",
    "mouth_corner_tilt_deg",
    "lip_thickness_ratio",
    "upper_lip_ratio",
    "philtrum_ratio",
)


class GeometryError(Exception):
    """Raised when landmarks are degenerate or pose is out of the frontal band."""

    def __init__(self, code: str, details: str | None = None):
        self.code = code
        self.details = details
        super().__init__(details if details is not None else code)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])
    return float(math.degrees(math.atan2(dy, dx)))


def _triangle_angle_deg(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    v1 = a - p
    v2 = b - p
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))


def estimate_pose(norm468: np.ndarray) -> Dict[str, float]:
    """Estimate yaw/pitch (degrees) from normalized MediaPipe landmarks."""
    if norm468.ndim != 2 or norm468.shape[0] < 468:
        raise GeometryError("FACE_TOO_ANGLED_OR_SMALL", "Expected at least 468 landmarks.")

    p = norm468[:, :2]
    face_left = p[234]
    face_right = p[454]
    face_top = p[10]
    face_bottom = p[152]
    nose_tip = p[4]
    left_eye_outer = p[33]
    right_eye_outer = p[263]

    face_width = _dist(face_left, face_right)
    face_height = _dist(face_top, face_bottom)
    if face_width <= 1e-08 or face_height <= 1e-08:
        raise GeometryError("FACE_TOO_ANGLED_OR_SMALL", "Degenerate face geometry for pose.")

    center_x = float((face_left[0] + face_right[0]) / 2.0)
    half_w = face_width / 2.0

    yaw_ratio = float((nose_tip[0] - center_x) / (half_w + 1e-06))
    yaw_deg = float(np.clip(yaw_ratio, -1.0, 1.0) * 90.0)

    eye_line_y = float((left_eye_outer[1] + right_eye_outer[1]) / 2.0)
    chin_y = float(face_bottom[1])
    eye_chin = float(chin_y - eye_line_y)
    expected_nose_y = eye_line_y + EXPECTED_NOSE_EYE_CHIN_FRAC * eye_chin
    band = abs(eye_chin) / 2.0 + 1e-06
    pitch_ratio = float((float(nose_tip[1]) - expected_nose_y) / band)
    pitch_deg = float(np.clip(pitch_ratio, -1.0, 1.0) * 90.0)

    if norm468.shape[1] >= 3:
        z = norm468[:, 2]
        z_asym = float(z[234] - z[454])
        yaw_deg = float(0.85 * yaw_deg + 0.15 * np.clip(z_asym * 180.0, -90.0, 90.0))

    return {"yaw_deg": yaw_deg, "pitch_deg": pitch_deg}


def assert_frontal(
    pose: Mapping[str, float],
    yaw_max: float = POSE_YAW_MAX_DEG,
    pitch_max: float = POSE_PITCH_MAX_DEG,
) -> None:
    yaw = float(pose["yaw_deg"])
    pitch = float(pose["pitch_deg"])
    if abs(yaw) > yaw_max or abs(pitch) > pitch_max:
        raise GeometryError(
            "FACE_TOO_ANGLED_OR_SMALL",
            f"Pose out of frontal band (|yaw|<={yaw_max}, |pitch|<={pitch_max}): "
            f"yaw={yaw:.1f}, pitch={pitch:.1f}",
        )


def extract_geometry_features(norm468: np.ndarray) -> Dict[str, float]:
    """Extract the 24 locked geometry features from normalized MediaPipe landmarks.

    ``norm468`` shape: (468+, 2) or (468+, 3) with coordinates in roughly [0, 1].
    """
    if norm468.ndim != 2 or norm468.shape[0] < 468:
        raise GeometryError("FACE_TOO_ANGLED_OR_SMALL", "Expected at least 468 landmarks.")

    p = norm468[:, :2]

    face_top = p[10]
    face_bottom = p[152]
    face_left = p[234]
    face_right = p[454]
    nose_tip = p[4]
    nose_bridge = p[6]
    mouth_left = p[61]
    mouth_right = p[291]
    left_eye_outer = p[33]
    right_eye_outer = p[263]
    left_eye_inner = p[133]
    right_eye_inner = p[362]
    left_brow = p[105]
    right_brow = p[334]

    face_width = _dist(face_left, face_right)
    face_height = _dist(face_top, face_bottom)
    if face_height == 0 or face_width == 0:
        raise GeometryError("FACE_TOO_ANGLED_OR_SMALL", "Degenerate face geometry.")

    eye_line_y = float((left_eye_outer[1] + right_eye_outer[1]) / 2.0)
    midface_len = abs(float(nose_tip[1] - eye_line_y))
    lowerface_len = abs(float(face_bottom[1] - nose_tip[1]))

    center_x = float((face_left[0] + face_right[0]) / 2.0)
    pairs = [(33, 263), (133, 362), (61, 291), (234, 454), (105, 334), (10, 10)]
    errs: list[float] = []
    for li, ri in pairs:
        lx = float(p[li][0])
        rx = float(p[ri][0])
        errs.append(abs((lx - center_x) + (rx - center_x)))
    symmetry_error = float(np.mean(errs))

    eye_spacing = _dist(left_eye_inner, right_eye_inner)
    eye_width = _dist(left_eye_outer, left_eye_inner)
    eye_spacing_ratio = (eye_spacing / eye_width) if eye_width > 0 else 1.0

    left_upper_lid = p[159]
    left_lower_lid = p[145]
    eye_openness_ratio = abs(float(left_lower_lid[1] - left_upper_lid[1])) / (eye_width + 1e-06)
    eye_size_ratio = eye_width / face_width
    eye_tilt_deg = _angle_deg(left_eye_outer, right_eye_outer)

    brow_height_ratio = abs(float(left_brow[1] - left_eye_outer[1])) / (face_height + 1e-06)
    brow_tilt_deg = _angle_deg(left_brow, right_brow)

    nose_width = _dist(p[131], p[360])
    nose_length = _dist(nose_bridge, nose_tip)
    mouth_width = _dist(mouth_left, mouth_right)
    nose_tip_deviation_ratio = abs(float(nose_tip[0] - center_x)) / (face_width + 1e-06)
    mouth_corner_tilt_deg = _angle_deg(mouth_left, mouth_right)

    jaw_angle = _triangle_angle_deg(p[152], p[234], p[454])
    jaw_angle_sharpness = 180.0 - jaw_angle

    jaw_width_ratio = _dist(p[172], p[397]) / (face_width + 1e-06)
    chin_length_ratio = abs(float(p[152][1] - nose_tip[1])) / (face_height + 1e-06)
    chin_width_ratio = _dist(p[176], p[400]) / (face_width + 1e-06)
    cheekbone_width_ratio = _dist(p[93], p[323]) / (face_width + 1e-06)
    lower_cheek_ratio = _dist(p[58], p[288]) / (face_width + 1e-06)

    upper_lip = p[13]
    lower_lip = p[14]
    lip_thickness_ratio = abs(float(lower_lip[1] - upper_lip[1])) / (face_height + 1e-06)
    philtrum_ratio = _dist(p[2], p[0]) / (face_height + 1e-06)
    upper_lip_ratio = _dist(p[0], p[13]) / (face_height + 1e-06)

    feats: Dict[str, float] = {
        "symmetry_error": symmetry_error,
        "face_aspect_ratio": face_width / face_height,
        "midface_length_ratio": midface_len / face_height,
        "lowerface_length_ratio": lowerface_len / face_height,
        "jaw_width_ratio": jaw_width_ratio,
        "jaw_angle_sharpness": jaw_angle_sharpness,
        "chin_length_ratio": chin_length_ratio,
        "chin_width_ratio": chin_width_ratio,
        "cheekbone_width_ratio": cheekbone_width_ratio,
        "lower_cheek_ratio": lower_cheek_ratio,
        "eye_openness_ratio": eye_openness_ratio,
        "eye_size_ratio": eye_size_ratio,
        "eye_spacing_ratio": eye_spacing_ratio,
        "eye_tilt_deg": eye_tilt_deg,
        "brow_height_ratio": brow_height_ratio,
        "brow_tilt_deg": brow_tilt_deg,
        "nose_width_ratio": nose_width / face_width,
        "nose_length_ratio": nose_length / face_height,
        "nose_tip_deviation_ratio": nose_tip_deviation_ratio,
        "mouth_width_ratio": mouth_width / face_width,
        "mouth_corner_tilt_deg": mouth_corner_tilt_deg,
        "lip_thickness_ratio": lip_thickness_ratio,
        "upper_lip_ratio": upper_lip_ratio,
        "philtrum_ratio": philtrum_ratio,
    }
    for k in FEATURE_COLS:
        feats[k] = float(feats[k])
    return feats


def features_to_vector(feats: MutableMapping[str, float]) -> np.ndarray:
    return np.array([float(feats[c]) for c in FEATURE_COLS], dtype=np.float32)
