"""Display-only beauty score calibration (no model retrain).

score = clip(SCALE * score_raw + BIAS, 0, 100)

Env (optional):
  BEAUTY_SCORE_SCALE  default 1.2
  BEAUTY_SCORE_BIAS   default 15.0

Set SCALE=1 and BIAS=0 to disable (identity after clip).
"""

from __future__ import annotations

import os
from typing import Tuple


def _read_calibration() -> Tuple[float, float]:
    """Helper: read SCALE/BIAS from env with safe defaults on bad values."""
    try:
        scale = float(os.environ.get("BEAUTY_SCORE_SCALE", "1.2"))
    except ValueError:
        scale = 1.2
    try:
        bias = float(os.environ.get("BEAUTY_SCORE_BIAS", "15"))
    except ValueError:
        bias = 15.0
    return scale, bias


def calibration_params() -> Tuple[float, float]:
    """Expose the current (scale, bias) pair in effect."""
    return _read_calibration()


def calibrate_beauty_score(raw: float) -> float:
    """Map uncalibrated MLP output to display score in [0, 100] (affine; no retrain)."""
    scale, bias = _read_calibration()
    return float(max(0.0, min(100.0, scale * float(raw) + bias)))


def calibration_note() -> str:
    """Human-readable calibration summary for API ``notes``."""
    scale, bias = _read_calibration()
    return f"Beauty score calibrate: scale={scale} bias={bias}"
