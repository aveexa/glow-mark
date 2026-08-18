#!/usr/bin/env python3
"""Merge primary human submissions into suggestion_dataset.csv (human_v1).

Usage:

  python backend/scripts/merge_human_labels.py \\
    --primary data/labeling/submissions/ann_a.csv \\
    --primary-annotator ann_a
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from suggestion_rules import load_catalog  # noqa: E402

TOP_K = 4


def _load_submissions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _validate_row(row: dict[str, str], catalog: dict[str, dict[str, str]]) -> list[str]:
    errors: list[str] = []
    sid = row.get("sample_id", "").strip()
    if not sid:
        errors.append("empty sample_id")
        return errors
    pri = [s for s in (row.get("priority_order") or "").split("|") if s]
    sug = [s for s in (row.get("suggestion_ids") or "").split("|") if s]
    if not sug:
        sug = list(pri)
    if not pri:
        pri = list(sug)
    if not (1 <= len(pri) <= TOP_K):
        errors.append(f"{sid}: priority_order length {len(pri)} not in 1..{TOP_K}")
    if set(pri) != set(sug):
        errors.append(f"{sid}: suggestion_ids set != priority_order set")
    for i in pri:
        if i not in catalog:
            errors.append(f"{sid}: unknown/inactive id {i}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge human labels into Dataset C.")
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--primary-annotator", type=str, default="")
    parser.add_argument(
        "--secondary",
        type=Path,
        default=None,
        help="Optional secondary submission (stored for agreement, not merged as primary labels)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "suggestion_dataset.csv",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "data" / "catalogs" / "suggestions.csv",
    )
    parser.add_argument(
        "--secondary-out",
        type=Path,
        default=REPO_ROOT / "data" / "labeling" / "submissions" / "secondary_labels.csv",
    )
    args = parser.parse_args()

    if not args.primary.is_file():
        print(f"ERROR: missing primary submissions: {args.primary}", file=sys.stderr)
        return 1
    if not args.dataset.is_file():
        print(f"ERROR: missing dataset: {args.dataset}", file=sys.stderr)
        return 1

    catalog = load_catalog(args.catalog)
    primary_rows = _load_submissions(args.primary)
    if args.primary_annotator:
        primary_rows = [
            r for r in primary_rows if r.get("annotator_id") == args.primary_annotator
        ] or primary_rows

    errors: list[str] = []
    for r in primary_rows:
        errors.extend(_validate_row(r, catalog))
    if errors:
        for e in errors[:30]:
            print(f"ERROR: {e}", file=sys.stderr)
        print(f"FAILED validation ({len(errors)} errors)", file=sys.stderr)
        return 1

    # Backup rules snapshot once
    backup = args.dataset.with_name("suggestion_dataset.rules_v1_backup.csv")
    if not backup.is_file():
        shutil.copy2(args.dataset, backup)
        print(f"Backup → {backup}")

    by_sample = {r["sample_id"]: r for r in primary_rows}

    with args.dataset.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        ds_rows = list(reader)

    updated = 0
    for row in ds_rows:
        hum = by_sample.get(row["sample_id"])
        if not hum:
            continue
        pri = [s for s in (hum.get("priority_order") or "").split("|") if s]
        joined = "|".join(pri)
        row["suggestion_ids"] = joined
        row["priority_order"] = joined
        row["annotator_id"] = hum.get("annotator_id") or args.primary_annotator or "ann_unknown"
        row["label_method"] = "human_v1"
        updated += 1

    with args.dataset.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ds_rows)

    print(f"Merged human_v1 labels into {args.dataset}: updated={updated} total={len(ds_rows)}")

    if args.secondary and args.secondary.is_file():
        sec_rows = _load_submissions(args.secondary)
        for r in sec_rows:
            errors = _validate_row(r, catalog)
            if errors:
                print(f"WARN secondary: {errors[0]}", file=sys.stderr)
        args.secondary_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.secondary, args.secondary_out)
        print(f"Secondary copy → {args.secondary_out} ({len(sec_rows)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
