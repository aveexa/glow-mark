"""Square face-centered crop for stable MediaPipe / scoring inputs.

Fail-soft: callers should fall back to the original image when normalize returns None.
Does not change beauty/feature/suggestion feature contracts — only framing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# MediaPipe face oval key points used for a coarse bbox.
_BBOX_LANDMARKS = (10, 152, 234, 454)

DEFAULT_MARGIN = 0.35
DEFAULT_OUTPUT_SIZE = 512


@dataclass(frozen=True)
class FaceCropTransform:
    """Hold crop→original mapping so UI overlays can be drawn on the upload image.

    Crop pixels (x_c, y_c) in the resized square relate to the padded crop
    before resize, which sits on a virtual canvas that may extend outside the
    original image (when padding was applied).
    """

    orig_w: int
    orig_h: int
    # Top-left of the square crop in original pixel space (may be negative if padded).
    crop_x0: float
    crop_y0: float
    # Side length of the square in original pixel units (before resize).
    crop_side: float
    output_size: int

    def crop_norm_to_orig_norm(self, nx: float, ny: float) -> Tuple[float, float]:
        """Map one landmark from square-crop [0,1] → original-image [0,1]."""
        # Position in the pre-resize square (original pixel units).
        px = self.crop_x0 + float(nx) * self.crop_side
        py = self.crop_y0 + float(ny) * self.crop_side
        ox = px / float(self.orig_w) if self.orig_w > 0 else 0.0
        oy = py / float(self.orig_h) if self.orig_h > 0 else 0.0
        return ox, oy

    def remap_landmarks(self, norm468: np.ndarray) -> np.ndarray:
        """Batch-remap crop-normalized landmarks to original-normalized coords for UI overlay."""
        out = np.array(norm468, dtype=np.float32, copy=True)
        for i in range(out.shape[0]):
            ox, oy = self.crop_norm_to_orig_norm(float(out[i, 0]), float(out[i, 1]))
            out[i, 0] = ox
            out[i, 1] = oy
        return out


def _bbox_from_landmarks(norm468: np.ndarray, w: int, h: int) -> Optional[Tuple[float, float, float, float]]:
    """Helper: coarse face bbox (min_x, min_y, max_x, max_y) in original pixels, or None if degenerate."""
    if norm468.ndim != 2 or norm468.shape[0] < 155:
        return None
    xs = []
    ys = []
    for idx in _BBOX_LANDMARKS:
        if idx >= norm468.shape[0]:
            return None
        xs.append(float(norm468[idx, 0]) * float(w))
        ys.append(float(norm468[idx, 1]) * float(h))
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x - min_x < 1.0 or max_y - min_y < 1.0:
        return None
    return min_x, min_y, max_x, max_y


def square_face_crop(
    img_bgr: np.ndarray,
    norm468: np.ndarray,
    *,
    margin: float = DEFAULT_MARGIN,
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> Optional[Tuple[np.ndarray, FaceCropTransform]]:
    """Build a square face-centered crop resized to ``output_size``.

    Returns ``(square_bgr, transform)`` or ``None`` on failure (caller fail-softs to original).
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    h, w = img_bgr.shape[:2]
    if h < 8 or w < 8:
        return None

    bbox = _bbox_from_landmarks(norm468, w, h)
    if bbox is None:
        return None

    min_x, min_y, max_x, max_y = bbox
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    face_w = max_x - min_x
    face_h = max_y - min_y
    side = max(face_w, face_h) * (1.0 + 2.0 * float(margin))
    if side < 8.0:
        return None

    x0 = cx - 0.5 * side
    y0 = cy - 0.5 * side
    x1 = x0 + side
    y1 = y0 + side

    # Integer bounds for the region that overlaps the original image.
    src_x0 = int(max(0, np.floor(x0)))
    src_y0 = int(max(0, np.floor(y0)))
    src_x1 = int(min(w, np.ceil(x1)))
    src_y1 = int(min(h, np.ceil(y1)))
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return None

    patch = img_bgr[src_y0:src_y1, src_x0:src_x1]
    if patch.size == 0:
        return None

    # Place patch onto a square canvas of side ``side`` (may include padding).
    side_i = int(max(8, round(side)))
    canvas = np.zeros((side_i, side_i, 3), dtype=img_bgr.dtype)
    # Fill with edge color from patch mean for gentler padding.
    fill = np.median(patch.reshape(-1, 3), axis=0).astype(img_bgr.dtype)
    canvas[:, :] = fill

    # Where the original pixels land on the canvas.
    dst_x0 = int(round(src_x0 - x0))
    dst_y0 = int(round(src_y0 - y0))
    # Clamp destination so the patch fits.
    dst_x0 = max(0, min(side_i - 1, dst_x0))
    dst_y0 = max(0, min(side_i - 1, dst_y0))
    dst_x1 = min(side_i, dst_x0 + (src_x1 - src_x0))
    dst_y1 = min(side_i, dst_y0 + (src_y1 - src_y0))
    copy_w = dst_x1 - dst_x0
    copy_h = dst_y1 - dst_y0
    if copy_w <= 0 or copy_h <= 0:
        return None
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = patch[:copy_h, :copy_w]

    out_size = int(max(64, output_size))
    square = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    transform = FaceCropTransform(
        orig_w=w,
        orig_h=h,
        crop_x0=float(x0),
        crop_y0=float(y0),
        crop_side=float(side),
        output_size=out_size,
    )
    return square, transform
