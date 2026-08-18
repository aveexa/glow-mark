#!/usr/bin/env python3
"""Build Dataset B from feature-contract-v1 extracts.

- Archive synthetic pre-v1 CSVs once
- Hash-split sample_id → train/val/test (80/10/10)
- Fit p20/p80 thresholds on train only
- Write suggestion_mapping_rules.csv + geometry_dataset.csv

Usage (from repo root):

  python backend/scripts/build_geometry_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from geometry import FEATURE_COLS, FEATURE_CONTRACT_VERSION  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"
INTERIM_CSV = REPO_ROOT / "data" / "interim" / "geometry_features" / "geometry_features.csv"
LABEL_METHOD = "percentile_p20_p80"
CONSENT_FLAG = "research_lfw"
SOURCE = "lfw"


def _archive_synthetics() -> None:
    mapping = {
        "geometry_dataset.csv": "geometry_dataset.synthetic_pre_v1.csv",
        "suggestion_dataset.csv": "suggestion_dataset.synthetic_pre_v1.csv",
        "suggestion_mapping_rules.csv": "suggestion_mapping_rules.synthetic_pre_v1.csv",
    }
    for src_name, dst_name in mapping.items():
        src = PROCESSED / src_name
        dst = PROCESSED / dst_name
        if src.is_file() and not dst.is_file():
            # Only archive if current file looks synthetic (no feature_contract_version col).
            with src.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames or []
            if "feature_contract_version" in fields and src_name == "geometry_dataset.csv":
                print(f"Skip archive {src_name}: already looks like v1 rebuild")
                continue
            if src_name == "suggestion_mapping_rules.csv":
                # Always archive old rules once if archive missing
                pass
            shutil.move(str(src), str(dst))
            print(f"Archived {src_name} -> {dst_name}")
        elif dst.is_file():
            print(f"Archive exists: {dst_name}")


def _split_for_sample(sample_id: str) -> str:
    h = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "test"


def _class_for(value: float, low_t: float, high_t: float) -> str:
    if value < low_t:
        return "low"
    if value > high_t:
        return "high"
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build geometry_dataset.csv (Dataset B).")
    parser.add_argument("--interim-csv", type=Path, default=INTERIM_CSV)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED)
    args = parser.parse_args()

    interim_csv: Path = args.interim_csv
    processed_dir: Path = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not interim_csv.is_file():
        print(f"ERROR: missing interim features: {interim_csv}", file=sys.stderr)
        return 1

    _archive_synthetics()

    with interim_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("ERROR: interim CSV is empty", file=sys.stderr)
        return 1

    for col in FEATURE_COLS:
        if col not in rows[0]:
            print(f"ERROR: missing feature column {col}", file=sys.stderr)
            return 1

    for row in rows:
        row["split"] = _split_for_sample(row["sample_id"])

    train_rows = [r for r in rows if r["split"] == "train"]
    if len(train_rows) < 10:
        print(f"ERROR: train split too small ({len(train_rows)})", file=sys.stderr)
        return 1

    thresholds: dict[str, tuple[float, float]] = {}
    for feat in FEATURE_COLS:
        vals = np.array([float(r[feat]) for r in train_rows], dtype=np.float64)
        p20 = float(np.percentile(vals, 20))
        p80 = float(np.percentile(vals, 80))
        thresholds[feat] = (p20, p80)

    rules_path = processed_dir / "suggestion_mapping_rules.csv"
    with rules_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["feature", "low_threshold_p20", "high_threshold_p80", "label_method"],
        )
        writer.writeheader()
        for feat in FEATURE_COLS:
            p20, p80 = thresholds[feat]
            writer.writerow(
                {
                    "feature": feat,
                    "low_threshold_p20": f"{p20:.6f}", 
                    "high_threshold_p80": f"{p80:.6f}",
                    "label_method": LABEL_METHOD,
                }
            )
    print(f"Wrote {rules_path}")

    out_fields = [
        "sample_id",
        "image_path",
        "split",
        *FEATURE_COLS,
        *[f"y_{c}" for c in FEATURE_COLS],
        "label_method",
        "consent_flag",
        "source",
        "feature_contract_version",
        "yaw_deg",
        "pitch_deg",
    ]

    out_rows = []
    for row in rows:
        out = {
            "sample_id": row["sample_id"],
            "image_path": row["image_path"],
            "split": row["split"],
            "label_method": LABEL_METHOD,
            "consent_flag": CONSENT_FLAG,
            "source": SOURCE,
            "feature_contract_version": row.get("feature_contract_version") or FEATURE_CONTRACT_VERSION,
            "yaw_deg": row.get("yaw_deg", ""),
            "pitch_deg": row.get("pitch_deg", ""),
        }
        for feat in FEATURE_COLS:
            val = float(row[feat])
            out[feat] = f"{val:.6f}"
            p20, p80 = thresholds[feat]
            out[f"y_{feat}"] = _class_for(val, p20, p80)
        out_rows.append(out)

    # Stable sort: split then sample_id
    out_rows.sort(key=lambda r: (r["split"], r["sample_id"]))

    geom_path = processed_dir / "geometry_dataset.csv"
    with geom_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {geom_path} ({len(out_rows)} rows)")

    counts = {s: sum(1 for r in out_rows if r["split"] == s) for s in ("train", "val", "test")}
    print(f"Split counts: {counts}")

    print("Train class rates (expect ~20/60/20):")
    train_out = [r for r in out_rows if r["split"] == "train"]
    for feat in FEATURE_COLS:
        ys = [r[f"y_{feat}"] for r in train_out]
        n = len(ys) or 1
        low = 100.0 * ys.count("low") / n
        ok = 100.0 * ys.count("ok") / n
        high = 100.0 * ys.count("high") / n
        print(f"  {feat}: low={low:.1f}% ok={ok:.1f}% high={high:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
