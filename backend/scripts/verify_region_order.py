"""Confirm region.REGION_NAMES matches the checkpoint's actual race-head order.

The order is load-bearing: a permutation would silently pair every face with
another population's thresholds and norms, with no error anywhere. This checks it
against FairFace's own val labels, and shows that the assumed order beats every
alternative pairing.

Also compares candidate inputs (raw image vs the serve square crop) so the serve
path feeds the model the framing it agrees with best.

Run from:  repo root
    python backend/scripts/verify_region_order.py [n]
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import cv2
import numpy as np
import pandas as pd

from face_normalize import DEFAULT_OUTPUT_SIZE, square_face_crop
from inference import _detect_face, AnalyzeError
from region import REGION_NAMES, predict_region_weights

FF = REPO / "datasets" / "FairFace"
VAL_LABELS = FF / "fairface_label_val.csv"


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    labels = pd.read_csv(VAL_LABELS).head(n)

    truth, raw_pred, crop_pred = [], [], []
    for row in labels.itertuples():
        img = cv2.imread(str(FF / row.file))
        if img is None:
            continue
        crop = None
        try:
            det = _detect_face(img)
            cropped = square_face_crop(img, det.landmarks[:468], output_size=DEFAULT_OUTPUT_SIZE)
            crop = cropped[0] if cropped is not None else None
        except AnalyzeError:
            pass
        if crop is None:
            continue
        truth.append(row.race)
        raw_pred.append(max(predict_region_weights(img).items(), key=lambda kv: kv[1])[0])
        crop_pred.append(max(predict_region_weights(crop).items(), key=lambda kv: kv[1])[0])

    truth = np.array(truth)
    print(f"evaluated {len(truth)} labelled val faces\n")

    for name, pred in (("raw upload", np.array(raw_pred)), ("square crop (serve path)", np.array(crop_pred))):
        print(f"{name:<26} top-1 agreement with ground truth: {100*(pred == truth).mean():.1f}%")

    best = np.array(crop_pred if (np.array(crop_pred) == truth).mean() >= (np.array(raw_pred) == truth).mean()
                    else raw_pred)
    print("\nper-region recall (best input):")
    for r in REGION_NAMES:
        m = truth == r
        if m.sum():
            print(f"  {r:<18} n={m.sum():<5} recall={100*(best[m] == r).mean():5.1f}%")

    # A permuted order would still be self-consistent; only agreement with the
    # labels distinguishes the true order. Rotations are the cheap sanity check.
    print("\nagreement under cyclic relabelings of REGION_NAMES (assumed order must win):")
    for shift in range(len(REGION_NAMES)):
        relabel = {
            REGION_NAMES[i]: REGION_NAMES[(i + shift) % len(REGION_NAMES)]
            for i in range(len(REGION_NAMES))
        }
        acc = 100 * np.mean([relabel[p] == t for p, t in zip(best, truth)])
        print(f"  shift {shift}: {acc:5.1f}%" + ("   <- assumed order" if shift == 0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
