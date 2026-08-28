"""Two questions, one run.

Q1. Do `symmetry_error` and `nose_tip_deviation_ratio` return to baseline range
    once a pose gate is applied? (The legacy baseline was gated to +/-15 deg;
    the sanity check was not. If yes, the extraction is confirmed sound.)

Q2. Are the neutrality blendshapes region-biased? `eyeSquint` sits at 0.365
    median on ordinary faces, which means MediaPipe is reading eye SHAPE, not
    expression. If that varies by population, one global threshold rejects some
    regions harder than others -- i.e. the fairness gate would itself be unfair.

Place in:  backend/scripts/diagnose_gates.py
Run from:  repo root
    python backend/scripts/diagnose_gates.py 3000
"""
from __future__ import annotations

import math
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
FF = REPO / "datasets" / "FairFace"
VAL_LABELS = FF / "fairface_label_val.csv"

POSE_LIMIT = 15.0  # matches the legacy baseline's observed +/-15 deg range

GATE_BS = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker",
           "mouthPressLeft", "mouthPressRight", "mouthFunnel",
           "browInnerUp", "browDownLeft", "browDownRight",
           "browOuterUpLeft", "browOuterUpRight",
           "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight"]

# candidate thresholds derived from the sanity-check percentiles
THRESH = {"jawOpen": 0.25, "mouthSmileLeft": 0.50, "mouthSmileRight": 0.50,
          "mouthPucker": 0.60, "mouthPressLeft": 0.40, "mouthPressRight": 0.40,
          "mouthFunnel": 0.20, "browInnerUp": 0.60, "browDownLeft": 0.55,
          "browDownRight": 0.55, "browOuterUpLeft": 0.60, "browOuterUpRight": 0.60,
          "eyeBlinkLeft": 0.60, "eyeBlinkRight": 0.60,
          "eyeSquintLeft": 0.70, "eyeSquintRight": 0.70}


def euler_from_matrix(m):
    """Yaw/pitch/roll in degrees from the 4x4 facial transformation matrix."""
    r = m[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch, yaw, roll = math.atan2(-r[1, 2], r[1, 1]), math.atan2(-r[2, 0], sy), 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def make_landmarker():
    return mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(TASK)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5))


def detect(lmk, img):
    res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if len(res.face_landmarks) != 1:
        return None
    arr = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], dtype=np.float32)
    bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
    mat = np.array(res.facial_transformation_matrixes[0], dtype=np.float32)
    return arr, bs, mat


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    lab = pd.read_csv(VAL_LABELS)
    lab["path"] = lab["file"].apply(lambda f: FF / f)
    lab = lab.head(n)

    lmk = make_landmarker()
    recs = []
    n_nodetect = 0

    for i, row in enumerate(lab.itertuples(), 1):
        if i % 500 == 0:
            print(f"  ...{i}/{len(lab)}", flush=True)
        img = cv2.imread(str(row.path))
        if img is None:
            continue
        d1 = detect(lmk, img)
        if d1 is None:
            n_nodetect += 1
            continue
        cropped = square_face_crop(img, d1[0][:468], output_size=DEFAULT_OUTPUT_SIZE)
        if cropped is None:
            continue
        d2 = detect(lmk, cropped[0])
        if d2 is None:
            n_nodetect += 1
            continue
        lm, bs, mat = d2
        try:
            feats = extract_geometry_features(lm[:468])
        except Exception:
            continue
        yaw, pitch, roll = euler_from_matrix(mat)
        recs.append({**feats, **{k: bs.get(k, np.nan) for k in GATE_BS},
                     "race": row.race, "gender": row.gender,
                     "yaw": yaw, "pitch": pitch, "roll": roll})

    lmk.close()
    df = pd.DataFrame(recs)
    print(f"\nextracted {len(df)} / {len(lab)}   (no face: {n_nodetect})\n")

    base = pd.read_csv(BASELINE)
    gated = df[(df.yaw.abs() <= POSE_LIMIT) & (df.pitch.abs() <= POSE_LIMIT)]
    print(f"pose gate |yaw|,|pitch| <= {POSE_LIMIT} deg  ->  "
          f"{len(gated)}/{len(df)} kept ({100*len(gated)/max(len(df),1):.1f}%)\n")

    # ---- Q1 -----------------------------------------------------------
    print("=" * 74)
    print("Q1  Do the flagged features return to baseline range when gated?")
    print("=" * 74)
    print(f"{'feature':<26}{'base med':>10}{'ungated':>10}{'GATED':>10}{'base range':>18}")
    for c in ["symmetry_error", "nose_tip_deviation_ratio",
              "midface_length_ratio", "nose_length_ratio"]:
        bm, lo, hi = base[c].median(), base[c].min(), base[c].max()
        print(f"{c:<26}{bm:>10.4f}{df[c].median():>10.4f}{gated[c].median():>10.4f}"
              f"{f'[{lo:.3f}, {hi:.3f}]':>18}"
              f"  {'RESOLVED' if lo <= gated[c].median() <= hi else 'still out'}")

    # ---- Q2 -----------------------------------------------------------
    print("\n" + "=" * 74)
    print("Q2  Are the neutrality blendshapes region-biased?")
    print("=" * 74)
    print("median value per region (gated faces only)\n")
    key = ["eyeSquintLeft", "eyeSquintRight", "eyeBlinkLeft",
           "jawOpen", "mouthSmileLeft", "browInnerUp"]
    med = gated.groupby("race")[key].median()
    print(med.to_string(float_format=lambda x: f"{x:7.3f}"))
    print("\nspread across regions (max - min of the medians):")
    for c in key:
        sp = med[c].max() - med[c].min()
        print(f"  {c:<18} {sp:6.3f}   {med[c].idxmax():<16} vs {med[c].idxmin()}"
              f"   {'<-- LARGE' if sp > 0.15 else ''}")

    # ---- rejection rate per region under candidate thresholds ----------
    print("\n" + "=" * 74)
    print("Rejection rate per region under candidate thresholds")
    print("=" * 74)
    rej = pd.Series(False, index=gated.index)
    for k, t in THRESH.items():
        if k in gated:
            rej |= gated[k] > t
    out = (gated.assign(rejected=rej)
                .groupby("race")["rejected"]
                .agg(n="size", rejected="sum"))
    out["rate_%"] = 100 * out.rejected / out.n
    print(out.to_string(float_format=lambda x: f"{x:8.1f}"))
    spread = out["rate_%"].max() - out["rate_%"].min()
    print(f"\noverall: {100*rej.mean():.1f}%   spread across regions: {spread:.1f} pts")
    print("A large spread means the neutrality gate is itself region-biased.")

    outp = REPO / "data" / "interim" / "diagnose_gates.csv"
    df.to_csv(outp, index=False)
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())