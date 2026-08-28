"""Build the region-conditioned reference statistics from FairFace train.

Replicates the serve path exactly — detect, square crop, re-detect, pose gate,
roll autocorrect, neutrality gate — so the reference population and the serve
population are drawn through the same funnel.

Two deliberate omissions relative to serve:
  * no CLIP realness gate — every FairFace image is already a photograph.
  * no region model — the label CSV is ground truth, so neutrality uses a
    one-hot on the labelled region rather than a predicted mixture.

Alongside the 23 geometry features the table carries one extra row per cell,
``beauty_score_raw``: the raw beauty-MLP output, whose per-region mu/sigma let
serve express the score relative to the user's own population.

Place in:  backend/scripts/build_region_reference.py
Run from:  repo root
    python backend/scripts/build_region_reference.py [--workers 12] [--limit N]

Output:    data/processed/region_reference_stats.csv
"""
from __future__ import annotations

import argparse
import multiprocessing as mp_proc
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import cv2
import numpy as np
import pandas as pd

FF = REPO / "datasets" / "FairFace"
TRAIN_LABELS = FF / "fairface_label_train.csv"
OUT_PATH = REPO / "data" / "processed" / "region_reference_stats.csv"

GLOBAL_ARM = "global"
POOLED_GENDER = "pooled"
# Fixed 50/50 sex weights for the pooled arm. FairFace's Middle Eastern group is
# 69% male; proportional pooling would carry that imbalance into the norms.
POOLED_WEIGHTS = {"male": 0.5, "female": 0.5}
MIN_CELL_N = 200

# Identical in every row (both computed from p4 and p152), so it carries no
# information the lowerface ratio does not already carry. See the brief.
EXCLUDED_FEATURES = ("chin_length_ratio",)

_WORKER_STATE: dict = {}


def _worker_init() -> None:
    """Each process builds its own landmarker; MediaPipe graphs are not fork-safe."""
    from inference import _face_landmarker, _load_models

    _WORKER_STATE["landmarker"] = _face_landmarker()
    _WORKER_STATE["models"] = _load_models()


def _extract(args: tuple[str, str, str]) -> dict | None:
    """Run one image through the serve funnel. Returns a feature row or None if gated out."""
    import torch

    from face_normalize import DEFAULT_OUTPUT_SIZE, square_face_crop
    from gates import autocorrect_roll, check_neutrality, check_pose
    from geometry import extract_geometry_features
    from inference import _beauty_features_from_68, _detect_face
    from region_stats import BEAUTY_STAT

    path, race, gender = args
    try:
        img = cv2.imread(path)
        if img is None:
            return None

        det = _detect_face(img)                                   # pass 1: original
        cropped = square_face_crop(img, det.landmarks[:468], output_size=DEFAULT_OUTPUT_SIZE)
        if cropped is None:
            return None
        det = _detect_face(cropped[0])                            # pass 2: square crop

        passed, pose = check_pose(det.matrix)
        if not passed:
            return None

        roll = float(pose["roll_correction_deg"])
        if roll:
            det = _detect_face(autocorrect_roll(cropped[0], roll))

        # Ground-truth region as a one-hot, so per-region thresholds apply exactly.
        neutral, _ = check_neutrality(det.blendshapes, {race: 1.0})
        if not neutral:
            return None

        landmarks = det.landmarks[:468]
        feats = extract_geometry_features(landmarks)

        # Raw beauty score through the same path serve uses, before calibration.
        beauty = _WORKER_STATE["models"]["beauty"]
        x = _beauty_features_from_68(landmarks, beauty["ref_span"], beauty["ref_center"])
        xn = (x - beauty["mu"]) / (beauty["sd"] + 1e-6)
        with torch.no_grad():
            feats[BEAUTY_STAT] = float(beauty["model"](torch.from_numpy(xn)).reshape(-1)[0].item())
    except Exception:  # noqa: BLE001 — one bad image must not stop the build
        return None

    for name in EXCLUDED_FEATURES:
        feats.pop(name, None)
    return {**feats, "region": race, "gender": gender, "file": path}


def _cell_stats(frame: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    """p20 / p80 / mu / sigma per feature for one (region, gender) cell."""
    return {
        f: {
            "p20": float(frame[f].quantile(0.20)),
            "p80": float(frame[f].quantile(0.80)),
            "mu": float(frame[f].mean()),
            "sigma": float(frame[f].std(ddof=0)),
        }
        for f in features
    }


def _mix(cells: dict[str, dict], weights: dict[str, float], feature: str) -> dict[str, float]:
    """Mixture of per-arm stats. Sigma uses the mixture-variance identity."""
    total = sum(weights.values())
    mu = sum(w * cells[a][feature]["mu"] for a, w in weights.items()) / total
    second = sum(
        w * (cells[a][feature]["sigma"] ** 2 + cells[a][feature]["mu"] ** 2)
        for a, w in weights.items()
    ) / total
    return {
        "p20": sum(w * cells[a][feature]["p20"] for a, w in weights.items()) / total,
        "p80": sum(w * cells[a][feature]["p80"] for a, w in weights.items()) / total,
        "mu": mu,
        "sigma": float(np.sqrt(max(second - mu * mu, 0.0))),
    }


def main() -> int:
    from gates import load_gate_config
    from geometry import FEATURE_CONTRACT_VERSION
    from region_stats import BEAUTY_STAT

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap the number of images")
    args = ap.parse_args()

    labels = pd.read_csv(TRAIN_LABELS)
    if args.limit:
        labels = labels.head(args.limit)
    jobs = [
        (str(FF / row.file), str(row.race), str(row.gender).lower())
        for row in labels.itertuples()
    ]
    print(f"{len(jobs):,} images, {args.workers} workers", flush=True)

    rows: list[dict] = []
    with mp_proc.Pool(args.workers, initializer=_worker_init) as pool:
        for i, rec in enumerate(pool.imap_unordered(_extract, jobs, chunksize=64), 1):
            if rec is not None:
                rows.append(rec)
            if i % 5000 == 0:
                print(f"  {i:,}/{len(jobs):,}  kept {len(rows):,} ({100*len(rows)/i:.1f}%)", flush=True)

    # Sort by source file so the float summations below run in a fixed order and
    # re-running the build reproduces the CSV byte for byte despite imap_unordered.
    df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    print(f"\nextracted {len(df):,} / {len(jobs):,}  ({100*len(df)/len(jobs):.1f}% yield)\n")

    features = [c for c in df.columns if c not in ("region", "gender", "file")]
    regions = sorted(df["region"].unique())

    # gate_version pins the thresholds these statistics were funnelled through, so a
    # recalibration invalidates the table visibly rather than silently.
    provenance = load_gate_config().get("provenance", {})
    gate_version = f"n{provenance.get('n_extracted', 0)}_pose{provenance.get('n_after_pose', 0)}"

    out: list[dict] = []

    def emit(region: str, gender: str, stats: dict[str, dict[str, float]], n: int) -> None:
        for f in features:
            out.append({
                "region": region,
                "gender": gender,
                "feature": f,
                "p20": stats[f]["p20"],
                "p80": stats[f]["p80"],
                "mu": stats[f]["mu"],
                "sigma": stats[f]["sigma"],
                "n": n,
                "gate_version": gate_version,
                "feature_contract_version": FEATURE_CONTRACT_VERSION,
            })

    for region in regions:
        region_df = df[df["region"] == region]
        cells = {}
        for gender in ("male", "female"):
            cell = region_df[region_df["gender"] == gender]
            cells[gender] = _cell_stats(cell, features)
            emit(region, gender, cells[gender], len(cell))
        # Pooled is derived from the two sex cells at fixed 50/50, never from raw counts.
        pooled = {f: _mix(cells, POOLED_WEIGHTS, f) for f in features}
        emit(region, POOLED_GENDER, pooled, len(region_df))

    # Global arm over all gated faces, the baseline for the region-conditioning
    # experiment. Its pooled row is 50/50 by sex for the same reason.
    global_cells = {}
    for gender in ("male", "female"):
        cell = df[df["gender"] == gender]
        global_cells[gender] = _cell_stats(cell, features)
        emit(GLOBAL_ARM, gender, global_cells[gender], len(cell))
    emit(GLOBAL_ARM, POOLED_GENDER,
         {f: _mix(global_cells, POOLED_WEIGHTS, f) for f in features}, len(df))

    result = pd.DataFrame(out).sort_values(["region", "gender", "feature"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)

    print(f"wrote {OUT_PATH}  ({len(result)} rows, {len(features)} features)")
    thin = result[(result.n < MIN_CELL_N)][["region", "gender", "n"]].drop_duplicates()
    print(f"cells below n={MIN_CELL_N}: {len(thin)}")
    if len(thin):
        print(thin.to_string(index=False))
    counts = result[["region", "gender", "n"]].drop_duplicates().pivot(
        index="region", columns="gender", values="n")
    print("\nn per cell:")
    print(counts.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
