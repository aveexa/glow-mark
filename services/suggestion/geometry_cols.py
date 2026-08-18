"""Feature contract column names only — no MediaPipe/OpenCV."""

from __future__ import annotations

FEATURE_CONTRACT_VERSION: str = "v1"

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
