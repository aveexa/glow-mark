#!/usr/bin/env python3
"""Validate frozen Dataset B/C CSV + Parquet schemas.

Usage (from repo root):

  python backend/scripts/validate_dataset_schema.py
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
    ALLOWED_LABEL_METHODS,
    ALLOWED_SPLITS,
    ALLOWED_Y,
    FEATURE_COLS,
    GEOMETRY_DATASET_COLUMNS,
    SUGGESTION_DATASET_COLUMNS,
    Y_COLS,
)

PROCESSED = REPO_ROOT / "data" / "processed"
TOP_K = 4


def _check_columns(actual: list[str], expected: list[str], label: str, errors: list[str]) -> None:
    if actual != expected:
        errors.append(
            f"{label}: column order mismatch.\n"
            f"  expected ({len(expected)}): {expected}\n"
            f"  actual   ({len(actual)}): {actual}"
        )


def _validate_rows(rows: list[dict], *, is_suggestion: bool, label: str, errors: list[str]) -> None:
    for i, row in enumerate(rows):
        sid = row.get("sample_id", f"row{i}")
        if row.get("split") not in ALLOWED_SPLITS:
            errors.append(f"{label}:{sid}: bad split {row.get('split')!r}")
        for ycol in Y_COLS:
            if row.get(ycol) not in ALLOWED_Y:
                errors.append(f"{label}:{sid}: bad {ycol}={row.get(ycol)!r}")
                break
        if is_suggestion:
            lm = row.get("label_method")
            if lm not in ALLOWED_LABEL_METHODS:
                errors.append(f"{label}:{sid}: bad label_method {lm!r}")
            pri = [s for s in str(row.get("priority_order") or "").split("|") if s]
            sug = [s for s in str(row.get("suggestion_ids") or "").split("|") if s]
            if set(pri) != set(sug):
                errors.append(f"{label}:{sid}: suggestion_ids set != priority_order set")
            if len(pri) > TOP_K:
                errors.append(f"{label}:{sid}: more than {TOP_K} suggestion IDs")
        # recommended columns present
        for col in ("num_non_ok_features", "primary_feature", "quality_score", "yaw_deg", "pitch_deg"):
            if col not in row or row[col] in (None, ""):
                errors.append(f"{label}:{sid}: missing {col}")
                break
        if row.get("primary_feature") not in FEATURE_COLS:
            errors.append(f"{label}:{sid}: primary_feature {row.get('primary_feature')!r} not in FEATURE_COLS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate frozen dataset schemas.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED)
    args = parser.parse_args()
    processed: Path = args.processed_dir
    errors: list[str] = []

    checks = [
        ("geometry_dataset.csv", GEOMETRY_DATASET_COLUMNS, False),
        ("suggestion_dataset.csv", SUGGESTION_DATASET_COLUMNS, True),
        ("geometry_dataset.parquet", GEOMETRY_DATASET_COLUMNS, False),
        ("suggestion_dataset.parquet", SUGGESTION_DATASET_COLUMNS, True),
    ]

    for name, expected, is_sug in checks:
        path = processed / name
        if not path.is_file():
            errors.append(f"missing file: {path}")
            continue
        if name.endswith(".csv"):
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                cols = list(reader.fieldnames or [])
                rows = list(reader)
        else:
            df = pd.read_parquet(path)
            cols = list(df.columns)
            rows = df.astype(object).where(pd.notnull(df), "").to_dict(orient="records")
            # normalize values to strings for shared checks
            rows = [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in rows]

        _check_columns(cols, expected, name, errors)
        _validate_rows(rows, is_suggestion=is_sug, label=name, errors=errors)

    if errors:
        for e in errors[:40]:
            print(f"ERROR: {e}", file=sys.stderr)
        if len(errors) > 40:
            print(f"... and {len(errors) - 40} more", file=sys.stderr)
        print(f"FAILED ({len(errors)} errors)", file=sys.stderr)
        return 1

    print("OK dataset schema validation passed")
    print(f"  geometry cols={len(GEOMETRY_DATASET_COLUMNS)}")
    print(f"  suggestion cols={len(SUGGESTION_DATASET_COLUMNS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
