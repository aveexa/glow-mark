"""Sanity-check the FaceLandmarker v2 extraction against the legacy baseline ranges.

This is NOT a parity test (the legacy API and source images are gone). It confirms
the new extraction produces features on the same SCALE as the frozen baseline —
enough to catch gross errors (wrong axis, factor-of-N scale, degenerate landmarks).

It also dumps the blendshape distribution over the sample, which is the input to
calibrating the neutrality thresholds.

Place in:  backend/scripts/sanity_check_extraction.py
Run from:  repo root
    python backend/scripts/sanity_check_extraction.py datasets/FairFace/val 300
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from face_normalize import DEFAULT_OUTPUT_SIZE, square_face_crop
from geometry import FEATURE_COLS, extract_geometry_features

TASK = BACKEND / "models" / "face_landmarker_v2_with_blendshapes.task"
BASELINE = REPO / "data" / "interim" / "geometry_features" / "geometry_features.csv"

# blendshapes that drive the neutrality gate
GATE_BS = [
    "jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker",
    "mouthPressLeft", "mouthPressRight", "mouthFunnel",
    "browInnerUp", "browDownLeft", "browDownRight",
    "browOuterUpLeft", "browOuterUpRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
    "cheekPuff",
]


def make_landmarker():
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(TASK)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=2,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.5,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def detect(landmarker, img_bgr):
    """Return (landmarks_478x3, blendshape_dict, matrix_4x4) or None."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if len(res.face_landmarks) != 1:
        return None
    arr = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], dtype=np.float32)
    bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
    mat = np.array(res.facial_transformation_matrixes[0], dtype=np.float32)
    return arr, bs, mat


def main():
    if not TASK.exists():
        sys.exit(f"Missing: {TASK}")
    if not BASELINE.exists():
        sys.exit(f"Missing baseline: {BASELINE}")

    img_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "datasets/FairFace/val"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    paths = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:n]
    if not paths:
        sys.exit(f"No images in {img_dir}")

    base = pd.read_csv(BASELINE)
    landmarker = make_landmarker()

    rows, bs_rows = [], []
    n_nodetect = n_nocrop = n_geomfail = 0

    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue

        # ---- replicate the SERVE path: detect -> square crop -> re-detect ----
        d1 = detect(landmarker, img)
        if d1 is None:
            n_nodetect += 1
            continue
        norm_orig, _, _ = d1

        cropped = square_face_crop(img, norm_orig[:468], output_size=DEFAULT_OUTPUT_SIZE)
        if cropped is None:
            n_nocrop += 1
            continue
        square_bgr, _ = cropped

        d2 = detect(landmarker, square_bgr)
        if d2 is None:
            n_nodetect += 1
            continue
        norm468, bs, _mat = d2

        try:
            feats = extract_geometry_features(norm468[:468])
        except Exception:
            n_geomfail += 1
            continue

        rows.append(feats)
        bs_rows.append({k: bs.get(k, np.nan) for k in GATE_BS})

    landmarker.close()

    if not rows:
        sys.exit("Nothing extracted — check inputs.")

    new = pd.DataFrame(rows)
    bsdf = pd.DataFrame(bs_rows)

    print(f"\nattempted {len(paths)}   extracted {len(new)}")
    print(f"  no single face : {n_nodetect}")
    print(f"  crop failed    : {n_nocrop}")
    print(f"  geometry error : {n_geomfail}")
    print(f"  yield          : {100*len(new)/len(paths):.1f}%\n")

    # ---- scale comparison -------------------------------------------------
    print("=" * 78)
    print("SCALE CHECK — new median vs legacy baseline (different populations,")
    print("so exact agreement is NOT expected; we are looking for gross errors)")
    print("=" * 78)
    print(f"{'feature':<26}{'base med':>10}{'new med':>10}{'ratio':>8}{'base range':>20}  flag")

    flags = []
    for c in FEATURE_COLS:
        if c not in base.columns or c not in new.columns:
            print(f"{c:<26}{'MISSING':>10}")
            flags.append((c, "missing"))
            continue
        bm = float(base[c].median())
        nm = float(new[c].median())
        lo, hi = float(base[c].min()), float(base[c].max())
        ratio = (nm / bm) if abs(bm) > 1e-9 else float("nan")

        if abs(bm) < 1e-9:
            flag = "~zero base"
        elif not (0.5 <= abs(ratio) <= 2.0):
            flag = "CHECK"
            flags.append((c, f"ratio {ratio:.2f}"))
        elif not (lo <= nm <= hi):
            flag = "outside"
        else:
            flag = "ok"
        print(f"{c:<26}{bm:>10.4f}{nm:>10.4f}{ratio:>8.2f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>20}  {flag}")

    # ---- blendshape distribution (input to threshold calibration) ---------
    print("\n" + "=" * 78)
    print("BLENDSHAPE DISTRIBUTION — use these to set neutrality thresholds")
    print("=" * 78)
    print(f"{'blendshape':<22}{'p50':>8}{'p75':>8}{'p90':>8}{'p95':>8}{'p99':>8}{'max':>8}")
    for c in GATE_BS:
        q = bsdf[c].quantile([.50, .75, .90, .95, .99]).values
        print(f"{c:<22}{q[0]:>8.3f}{q[1]:>8.3f}{q[2]:>8.3f}"
              f"{q[3]:>8.3f}{q[4]:>8.3f}{bsdf[c].max():>8.3f}")

    outdir = REPO / "data" / "interim"
    new.to_csv(outdir / "sanity_features_new.csv", index=False)
    bsdf.to_csv(outdir / "sanity_blendshapes.csv", index=False)
    print(f"\nwrote {outdir/'sanity_features_new.csv'}")
    print(f"wrote {outdir/'sanity_blendshapes.csv'}")

    print("\n" + "=" * 78)
    if flags:
        print("REVIEW NEEDED:")
        for c, why in flags:
            print(f"  {c}: {why}")
    else:
        print("PASS — all 24 features on a plausible scale. Extraction is sound.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
