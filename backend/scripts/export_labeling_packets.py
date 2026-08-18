#!/usr/bin/env python3
"""Export JSONL labeling packets for the Flask labeling UI.

Usage (from repo root):

  python backend/scripts/export_labeling_packets.py --annotator-id ann_a --role primary --limit 50
  python backend/scripts/export_labeling_packets.py --annotator-id ann_b --role secondary --secondary-frac 0.15
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from geometry import FEATURE_COLS  # noqa: E402
from suggestion_rules import load_catalog, map_candidates, trigger_index  # noqa: E402

PACKET_TOP_K = 8


def _seeded_pick(sample_id: str, frac: float, seed: str = "glowmark_double_label_v1") -> bool:
    """Deterministic Bernoulli(frac) using sha1(seed:sample_id)."""
    h = hashlib.sha1(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    # Map first 8 hex digits → [0, 1)
    u = int(h[:8], 16) / 0xFFFFFFFF
    return u < frac


def main() -> int:
    parser = argparse.ArgumentParser(description="Export labeling packets (JSONL).")
    parser.add_argument("--annotator-id", required=True, help="e.g. ann_a")
    parser.add_argument("--role", choices=("primary", "secondary"), default="primary")
    parser.add_argument(
        "--secondary-frac",
        type=float,
        default=0.15,
        help="Fraction of train split to include for secondary role",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max samples (0 = all)")
    parser.add_argument(
        "--suggestion-csv",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "suggestion_dataset.csv",
    )
    parser.add_argument(
        "--catalog-csv",
        type=Path,
        default=REPO_ROOT / "data" / "catalogs" / "suggestions.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "labeling" / "packets",
    )
    args = parser.parse_args()

    if not args.suggestion_csv.is_file():
        print(f"ERROR: missing {args.suggestion_csv}", file=sys.stderr)
        return 1

    catalog = load_catalog(args.catalog_csv)
    index = trigger_index(catalog)

    with args.suggestion_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.role == "secondary":
        rows = [
            r
            for r in rows
            if r.get("split") == "train" and _seeded_pick(r["sample_id"], args.secondary_frac)
        ]
    # primary: all rows (optionally limited)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.annotator_id}_{args.role}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            features = {c: float(row[c]) for c in FEATURE_COLS}
            y_classes = {c: row[f"y_{c}"] for c in FEATURE_COLS}
            candidates = map_candidates(row, catalog, index, top_k=PACKET_TOP_K)
            rules_priority = [
                s for s in (row.get("priority_order") or row.get("suggestion_ids") or "").split("|") if s
            ]
            catalog_texts = {
                sid: catalog[sid]["approved_text"] for sid in candidates if sid in catalog
            }
            # Also include texts for full catalog (UI "add" dropdown)
            all_texts = {sid: catalog[sid]["approved_text"] for sid in catalog}

            rec = {
                "sample_id": row["sample_id"],
                "image_path": row["image_path"],
                "split": row.get("split", ""),
                "features": features,
                "y_classes": y_classes,
                "rules_priority": rules_priority,
                "candidate_ids": candidates,
                "catalog_texts": catalog_texts,
                "all_catalog_ids": list(catalog.keys()),
                "all_catalog_texts": all_texts,
                "role": args.role,
                "annotator_id": args.annotator_id,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path} ({len(rows)} samples, role={args.role})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
