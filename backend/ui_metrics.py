"""UI-only metric helpers derived from feature z-scores (not model inputs).

Produces symmetry / proportions / balance gauges for the frontend; never feeds
beauty, feature, or suggestion models.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

# Soft curve: z=0 => 100; decays slower than exp(-|z|); floor avoids display 0.
_SOFT_FLOOR = 5.0
_SOFT_SCALE = 0.5

PROPORTION_FEAT_NAMES: tuple[str, ...] = (
    "face_aspect_ratio",
    "midface_length_ratio",
    "lowerface_length_ratio",
    "eye_spacing_ratio",
    "nose_length_ratio",
    "mouth_width_ratio",
)


def z_to_score_soft(z: float) -> float:
    """Map a z-like value to a 5..100 display score via soft |z| decay."""
    raw = 100.0 * math.exp(-_SOFT_SCALE * abs(float(z)))
    return float(max(_SOFT_FLOOR, min(100.0, raw)))


def ui_metrics_from_z(
    z_row: np.ndarray,
    feat_cols: Sequence[str],
) -> dict:
    """Derive symmetry / proportions / balance display metrics from a (1,F) or (F,) z vector."""
    z = np.asarray(z_row, dtype=np.float64).reshape(-1)
    col_index = {name: i for i, name in enumerate(feat_cols)}

    if "symmetry_error" not in col_index:
        symmetry = _SOFT_FLOOR
    else:
        symmetry = z_to_score_soft(float(z[col_index["symmetry_error"]]))

    prop_scores: List[float] = []
    for name in PROPORTION_FEAT_NAMES:
        if name in col_index:
            prop_scores.append(z_to_score_soft(float(z[col_index[name]])))
    if prop_scores:
        proportions = float(np.mean(prop_scores))
    elif "face_aspect_ratio" in col_index:
        proportions = z_to_score_soft(float(z[col_index["face_aspect_ratio"]]))
    else:
        proportions = _SOFT_FLOOR

    balance = float(np.mean([z_to_score_soft(float(v)) for v in z])) if z.size else _SOFT_FLOOR

    return {
        "symmetry": symmetry,
        "proportions": proportions,
        "balance": balance,
    }
