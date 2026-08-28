#!/usr/bin/env python3
"""Batch-extract feature-contract-v1 geometry features from face images.

Usage (from repo root, with backend/.venv activated):

  python backend/scripts/extract_geometry_batch.py
  python backend/scripts/extract_geometry_batch.py --images-dir data/raw/images

Writes:
  data/interim/landmarks_468/{sample_id}.npy
  data/interim/geometry_features/{sample_id}.json
  data/interim/geometry_features/geometry_features.csv
  data/interim/rejects.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from geometry import (  # noqa: E402
    FEATURE_COLS,
    FEATURE_CONTRACT_VERSION,
    GeometryError,
    assert_frontal,
    estimate_pose_from_landmarks,
    extract_geometry_features,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _mp_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_landmarks_468(img_bgr: np.ndarray, face_mesh) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    faces = results.multi_face_landmarks or []
    if len(faces) == 0:
        raise GeometryError("NO_FACE_DETECTED")
    if len(faces) > 1:
        raise GeometryError("MULTIPLE_FACES_DETECTED")
    lm = faces[0].landmark
    if len(lm) < 468:
        raise GeometryError("FACE_TOO_ANGLED_OR_SMALL", f"Only {len(lm)} landmarks.")
    return np.array([[p.x, p.y, getattr(p, "z", 0.0)] for p in lm], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract geometry features (feature contract v1).")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "images",
        help="Directory of face images",
    )
    parser.add_argument(
        "--landmarks-dir",
        type=Path,
        default=REPO_ROOT / "data" / "interim" / "landmarks_468",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=REPO_ROOT / "data" / "interim" / "geometry_features",
    )
    parser.add_argument(
        "--rejects-csv",
        type=Path,
        default=REPO_ROOT / "data" / "interim" / "rejects.csv",
    )
    args = parser.parse_args()

    images_dir: Path = args.images_dir
    landmarks_dir: Path = args.landmarks_dir
    features_dir: Path = args.features_dir
    rejects_csv: Path = args.rejects_csv

    landmarks_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    rejects_csv.parent.mkdir(parents=True, exist_ok=True)

    if not images_dir.is_dir():
        print(f"Images directory not found: {images_dir}", file=sys.stderr)
        return 1

    paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        print(f"No images found in {images_dir}. Add files then re-run.")
        # Still write empty combined CSV header for tooling.
        combined_csv = features_dir / "geometry_features.csv"
        with combined_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sample_id", "image_path", "feature_contract_version", "yaw_deg", "pitch_deg", *FEATURE_COLS],
            )
            writer.writeheader()
        with rejects_csv.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["sample_id", "image_path", "code", "details"]).writeheader()
        return 0

    face_mesh = _mp_face_mesh()
    accepted_rows = []
    rejects = []

    for path in paths:
        sample_id = path.stem
        rel = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        try:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise GeometryError("CORRUPT_FILE", "Could not decode image.")
            norm468 = extract_landmarks_468(img, face_mesh)
            pose = estimate_pose_from_landmarks(norm468)
            assert_frontal(pose)
            feats = extract_geometry_features(norm468)

            np.save(landmarks_dir / f"{sample_id}.npy", norm468)
            payload = {
                "sample_id": sample_id,
                "image_path": rel,
                "feature_contract_version": FEATURE_CONTRACT_VERSION,
                "yaw_deg": pose["yaw_deg"],
                "pitch_deg": pose["pitch_deg"],
                "features": feats,
            }
            (features_dir / f"{sample_id}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            row = {
                "sample_id": sample_id,
                "image_path": rel,
                "feature_contract_version": FEATURE_CONTRACT_VERSION,
                "yaw_deg": pose["yaw_deg"],
                "pitch_deg": pose["pitch_deg"],
                **feats,
            }
            accepted_rows.append(row)
        except GeometryError as e:
            rejects.append(
                {
                    "sample_id": sample_id,
                    "image_path": rel,
                    "code": e.code,
                    "details": e.details or "",
                }
            )
        except Exception as e:  # noqa: BLE001 — batch job should not die on one file
            rejects.append(
                {
                    "sample_id": sample_id,
                    "image_path": rel,
                    "code": "UNKNOWN_ERROR",
                    "details": str(e),
                }
            )

    face_mesh.close()

    combined_csv = features_dir / "geometry_features.csv"
    fieldnames = ["sample_id", "image_path", "feature_contract_version", "yaw_deg", "pitch_deg", *FEATURE_COLS]
    with combined_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in accepted_rows:
            writer.writerow(row)

    with rejects_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image_path", "code", "details"])
        writer.writeheader()
        for row in rejects:
            writer.writerow(row)

    print(
        f"Feature contract {FEATURE_CONTRACT_VERSION}: "
        f"accepted={len(accepted_rows)} rejected={len(rejects)} "
        f"-> {combined_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
