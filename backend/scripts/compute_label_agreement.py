#!/usr/bin/env python3
"""Compute Jaccard agreement between primary and secondary human labels.

Usage:

  python backend/scripts/compute_label_agreement.py \\
    --primary data/labeling/submissions/ann_a.csv \\
    --secondary data/labeling/submissions/ann_b.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_ids(row: dict[str, str]) -> list[str]:
    pri = [s for s in (row.get("priority_order") or "").split("|") if s]
    if pri:
        return pri
    return [s for s in (row.get("suggestion_ids") or "").split("|") if s]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def _spearman_shared(a: list[str], b: list[str]) -> float | None:
    shared = [x for x in a if x in set(b)]
    if len(shared) < 2:
        return None
    # Rank positions in each list
    ra = {x: i for i, x in enumerate(a)}
    rb = {x: i for i, x in enumerate(b)}
    xs = [ra[x] for x in shared]
    ys = [rb[x] for x in shared]
    n = len(shared)
    # Spearman on ranks = Pearson of ranks
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    den_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def main() -> int:
    parser = argparse.ArgumentParser(description="Primary vs secondary label agreement.")
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "labeling" / "agreement",
    )
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    if not args.primary.is_file() or not args.secondary.is_file():
        print("ERROR: primary and secondary CSVs required", file=sys.stderr)
        return 1

    with args.primary.open(newline="", encoding="utf-8") as f:
        primary = {r["sample_id"]: r for r in csv.DictReader(f)}
    with args.secondary.open(newline="", encoding="utf-8") as f:
        secondary = {r["sample_id"]: r for r in csv.DictReader(f)}

    overlap = sorted(set(primary) & set(secondary))
    if not overlap:
        print("WARN: no overlapping sample_ids")
        report = {
            "n_overlap": 0,
            "mean_jaccard": None,
            "mean_spearman_shared": None,
            "pass": False,
            "threshold": args.threshold,
            "pairs": [],
        }
    else:
        pairs = []
        jaccards = []
        spearmans = []
        for sid in overlap:
            a = _parse_ids(primary[sid])
            b = _parse_ids(secondary[sid])
            j = _jaccard(set(a), set(b))
            jaccards.append(j)
            sp = _spearman_shared(a, b)
            if sp is not None:
                spearmans.append(sp)
            pairs.append(
                {
                    "sample_id": sid,
                    "primary": a,
                    "secondary": b,
                    "jaccard": round(j, 4),
                    "spearman_shared": None if sp is None else round(sp, 4),
                }
            )
        mean_j = sum(jaccards) / len(jaccards)
        mean_sp = (sum(spearmans) / len(spearmans)) if spearmans else None
        passed = mean_j >= args.threshold
        report = {
            "n_overlap": len(overlap),
            "mean_jaccard": round(mean_j, 4),
            "mean_spearman_shared": None if mean_sp is None else round(mean_sp, 4),
            "pass": passed,
            "threshold": args.threshold,
            "pairs": pairs,
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "agreement_report.json"
    md_path = args.out_dir / "agreement_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Label agreement report",
        "",
        f"- Overlap samples: **{report['n_overlap']}**",
        f"- Mean Jaccard: **{report['mean_jaccard']}** (threshold {args.threshold})",
        f"- Mean Spearman (shared IDs): **{report['mean_spearman_shared']}**",
        f"- Status: **{'PASS' if report['pass'] else 'FAIL'}**",
        "",
        "| sample_id | jaccard | spearman | primary | secondary |",
        "|-----------|---------|----------|---------|-----------|",
    ]
    for p in report.get("pairs") or []:
        lines.append(
            f"| {p['sample_id']} | {p['jaccard']} | {p['spearman_shared']} | "
            f"`{'|'.join(p['primary'])}` | `{'|'.join(p['secondary'])}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"n_overlap={report['n_overlap']} mean_jaccard={report['mean_jaccard']} "
        f"status={'PASS' if report['pass'] else 'FAIL'}"
    )
    # Always exit 0; print FAIL for operators
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
