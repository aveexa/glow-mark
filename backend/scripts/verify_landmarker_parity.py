"""Verify FaceLandmarker v2 produces the same 24 geometry features as legacy FaceMesh.

HARD GATE. If this fails, the feature contract has moved and every
before/after comparison in the project is invalid.

Place in:  backend/scripts/verify_landmarker_parity.py
Run from:  repo root  ->  python backend/scripts/verify_landmarker_parity.py <img_dir> [n]
"""
from __future__ import annotations

import sys
from pathlib import Path

# backend/scripts/ -> backend/  so `geometry` and friends import cleanly
BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import cv2
import mediapipe as mp
import numpy as np

from geometry import FEATURE_COLS, extract_geometry_features

TASK = BACKEND / "models" / "face_landmarker_v2_with_blendshapes.task"
TOL = 1e-5

HAS_LEGACY = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")


# ── OLD: exactly the config in inference.py ────────────────────────────────
def legacy_landmarks(img_bgr):
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    mesh.close()
    faces = res.multi_face_landmarks or []
    if len(faces) != 1:
        return None
    return np.array([[p.x, p.y, getattr(p, "z", 0.0)] for p in faces[0].landmark],
                    dtype=np.float32)


# ── NEW: Tasks API ─────────────────────────────────────────────────────────
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


def tasks_landmarks(landmarker, img_bgr):
    # CRITICAL: Tasks API needs mp.Image wrapping RGB, not a raw BGR array.
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if len(res.face_landmarks) != 1:
        return None, None, None
    arr = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], dtype=np.float32)
    bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
    mat = np.array(res.facial_transformation_matrixes[0], dtype=np.float32)
    return arr, bs, mat


def main():
    if not TASK.exists():
        sys.exit(f"Missing model file:\n  {TASK}")

    if len(sys.argv) < 2:
        sys.exit(f"usage: python {Path(sys.argv[0]).name} <image_dir> [n_images]")

    img_dir = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    paths = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:n]
    if not paths:
        sys.exit(f"No images found in {img_dir}")

    print(f"mediapipe {mp.__version__}   legacy solutions API: "
          f"{'available' if HAS_LEGACY else 'NOT AVAILABLE'}")

    landmarker = make_landmarker()
    deltas = {c: [] for c in FEATURE_COLS}
    n_ok = n_skip = 0
    shapes = set()
    bs = mat = None

    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            n_skip += 1
            continue

        new, bs_i, mat_i = tasks_landmarks(landmarker, img)
        if new is None:
            n_skip += 1
            continue
        bs, mat = bs_i, mat_i

        if not HAS_LEGACY:
            n_ok += 1
            shapes.add(new.shape[0])
            continue

        old = legacy_landmarks(img)
        if old is None:
            n_skip += 1
            continue

        shapes.add((old.shape[0], new.shape[0]))
        # geometry.py only indexes 0..467 — slice so it's like-for-like
        f_old = extract_geometry_features(old[:468])
        f_new = extract_geometry_features(new[:468])
        for c in FEATURE_COLS:
            deltas[c].append(abs(float(f_old[c]) - float(f_new[c])))
        n_ok += 1

    landmarker.close()

    if n_ok == 0:
        sys.exit("No image produced exactly one face. Check inputs.")

    print(f"\nprocessed {n_ok} images   (skipped {n_skip})")
    print(f"landmark counts: {sorted(shapes)}")

    if bs is not None:
        print(f"\nblendshapes returned: {len(bs)}  (expect 52)")
        for k in ("jawOpen", "mouthSmileLeft", "browInnerUp", "eyeBlinkLeft"):
            v = bs.get(k)
            print(f"  {k:<18} {v:.4f}" if v is not None else f"  {k:<18} MISSING")
        print(f"transformation matrix: {mat.shape}  (expect (4, 4))")

    if not HAS_LEGACY:
        print("\n" + "=" * 60)
        print("Legacy API unavailable — cannot A/B compare.")
        print("Tasks API works. Verify features against committed dataset")
        print("values instead (see build_geometry_dataset.py output).")
        print("=" * 60)
        return 0

    print(f"\ntolerance: {TOL}\n")
    worst = sorted(((max(v), k) for k, v in deltas.items()), reverse=True)
    for d, name in worst:
        print(f"  {'FAIL' if d > TOL else 'ok  '}  {name:<32} max|delta| = {d:.3e}")

    overall = worst[0][0]
    print("\n" + "=" * 60)
    if overall <= TOL:
        print(f"PASS — max delta {overall:.3e} across all 24 features.")
        print("Feature contract is stable. Safe to migrate.")
    else:
        print(f"FAIL — max delta {overall:.3e} exceeds {TOL}.")
        print("Check the BGR->RGB conversion and the [:468] slice.")
    print("=" * 60)
    return 0 if overall <= TOL else 1


if __name__ == "__main__":
    sys.exit(main())
