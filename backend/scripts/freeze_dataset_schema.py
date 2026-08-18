#!/usr/bin/env python3
"""Enrich Dataset B/C with frozen schema columns and export CSV + Parquet.

Usage (from repo root):

  python backend/scripts/freeze_dataset_schema.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from dataset_schema import (  # noqa: E402
    GEOMETRY_DATASET_COLUMNS,
    SUGGESTION_DATASET_COLUMNS,
    compute_train_stats,
    enrich_row,
    reorder_row,
)
from geometry import FEATURE_COLS  # noqa: E402

PROCESSED = REPO_ROOT / "data" / "processed"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(reorder_row(row, columns))


def _write_parquet(path: Path, rows: list[dict], columns: list[str]) -> None:
    ordered = [reorder_row(r, columns) for r in rows]
    df = pd.DataFrame(ordered, columns=columns)
    numeric_cols = set(FEATURE_COLS) | {
        "yaw_deg",
        "pitch_deg",
        "quality_score",
        "num_non_ok_features",
    }
    for col in columns:
        if col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_parquet(path, index=False)


def _enrich_table(rows: list[dict[str, str]], columns: list[str]) -> list[dict]:
    stats = compute_train_stats(rows)
    enriched = [enrich_row(dict(r), stats) for r in rows]
    return [reorder_row(r, columns) for r in enriched]


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze Dataset B/C schema and export Parquet.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED)
    args = parser.parse_args()
    processed: Path = args.processed_dir

    geom_csv = processed / "geometry_dataset.csv"
    sug_csv = processed / "suggestion_dataset.csv"
    if not geom_csv.is_file() or not sug_csv.is_file():
        print("ERROR: geometry_dataset.csv and suggestion_dataset.csv required", file=sys.stderr)
        return 1

    geom_rows = _enrich_table(_load_csv(geom_csv), GEOMETRY_DATASET_COLUMNS)
    sug_rows = _enrich_table(_load_csv(sug_csv), SUGGESTION_DATASET_COLUMNS)

    _write_csv(geom_csv, geom_rows, GEOMETRY_DATASET_COLUMNS)
    _write_csv(sug_csv, sug_rows, SUGGESTION_DATASET_COLUMNS)

    geom_pq = processed / "geometry_dataset.parquet"
    sug_pq = processed / "suggestion_dataset.parquet"
    _write_parquet(geom_pq, geom_rows, GEOMETRY_DATASET_COLUMNS)
    _write_parquet(sug_pq, sug_rows, SUGGESTION_DATASET_COLUMNS)

    print(f"Wrote {geom_csv} ({len(geom_rows)} rows, {len(GEOMETRY_DATASET_COLUMNS)} cols)")
    print(f"Wrote {geom_pq}")
    print(f"Wrote {sug_csv} ({len(sug_rows)} rows, {len(SUGGESTION_DATASET_COLUMNS)} cols)")
    print(f"Wrote {sug_pq}")
    print("GEOMETRY columns:", ", ".join(GEOMETRY_DATASET_COLUMNS))
    print("SUGGESTION columns:", ", ".join(SUGGESTION_DATASET_COLUMNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
