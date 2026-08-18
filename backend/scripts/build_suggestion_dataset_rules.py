#!/usr/bin/env python3
"""Build Dataset C via catalog rules (no neural net).

Reads geometry_dataset.csv + suggestions.csv, writes suggestion_dataset.csv
with top-k=4 suggestion_ids / priority_order (annotator_id=rules_v1).

Usage (from repo root):

  python backend/scripts/build_suggestion_dataset_rules.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from suggestion_rules import DEFAULT_TOP_K, load_catalog, map_candidates, trigger_index  # noqa: E402

TOP_K = DEFAULT_TOP_K


def main() -> int:
    parser = argparse.ArgumentParser(description="Build suggestion_dataset.csv via rules.")
    parser.add_argument(
        "--geometry-csv",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "geometry_dataset.csv",
    )
    parser.add_argument(
        "--catalog-csv",
        type=Path,
        default=REPO_ROOT / "data" / "catalogs" / "suggestions.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "suggestion_dataset.csv",
    )
    args = parser.parse_args()

    if not args.geometry_csv.is_file():
        print(f"ERROR: missing geometry dataset: {args.geometry_csv}", file=sys.stderr)
        return 1
    if not args.catalog_csv.is_file():
        print(f"ERROR: missing catalog: {args.catalog_csv}", file=sys.stderr)
        return 1

    catalog = load_catalog(args.catalog_csv)
    if not catalog:
        print("ERROR: no active catalog rows", file=sys.stderr)
        return 1
    index = trigger_index(catalog)

    with args.geometry_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        geom_fields = list(reader.fieldnames or [])
        geom_rows = list(reader)

    core = [
        c
        for c in geom_fields
        if c
        not in {
            "label_method",
            "consent_flag",
            "source",
            "feature_contract_version",
            "yaw_deg",
            "pitch_deg",
        }
    ]
    out_fields = core + [
        "suggestion_ids",
        "priority_order",
        "annotator_id",
        "label_method",
        "consent_flag",
        "source",
    ]
    for c in ("feature_contract_version", "yaw_deg", "pitch_deg"):
        if c in geom_fields and c not in out_fields:
            out_fields.append(c)

    id_counter: Counter[str] = Counter()
    length_counter: Counter[int] = Counter()
    out_rows = []

    for row in geom_rows:
        ids = map_candidates(row, catalog, index, top_k=TOP_K)
        if len(ids) > TOP_K:
            raise AssertionError(f"top-k exceeded for {row['sample_id']}")
        for sid in ids:
            if sid not in catalog:
                raise AssertionError(f"unknown suggestion_id {sid}")
            id_counter[sid] += 1
        length_counter[len(ids)] += 1
        joined = "|".join(ids)
        out = dict(row)
        out["suggestion_ids"] = joined
        out["priority_order"] = joined
        out["annotator_id"] = "rules_v1"
        out["label_method"] = "rules_v1"
        out_rows.append(out)

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for row in out_rows:
            writer.writerow({k: row.get(k, "") for k in out_fields})

    print(f"Wrote {args.out_csv} ({len(out_rows)} rows)")
    print(f"Suggestion list length distribution: {dict(sorted(length_counter.items()))}")
    print("Top suggestion_ids:")
    for sid, n in id_counter.most_common(15):
        print(f"  {n:4d} {sid}")

    assert all(len(r["suggestion_ids"].split("|")) <= TOP_K for r in out_rows)
    assert all(
        all(sid in catalog for sid in r["suggestion_ids"].split("|") if sid) for r in out_rows
    )
    print("OK suggestion dataset QA asserts passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
