#!/usr/bin/env python3
"""Dataset C ML-readiness quality gates (Phase 9).

Writes:
  data/processed/quality_report.json
  data/processed/quality_report.md
  data/labeling/spotcheck/spotcheck_100.csv

Exit 1 if any FAIL; 0 if only WARN/OK.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from geometry import FEATURE_COLS  # noqa: E402
from suggestion_rules import load_catalog  # noqa: E402

Y_COLS = [f"y_{c}" for c in FEATURE_COLS]
ALLOWED_Y = {"low", "ok", "high"}
DUP_L2_THRESH = 1e-3
BALANCE_MIN = 50
SPOTCHECK_N = 100
SPOTCHECK_SEED = 42


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_ids(row: dict[str, str]) -> list[str]:
    pri = [s for s in (row.get("priority_order") or "").split("|") if s]
    if pri:
        return pri
    return [s for s in (row.get("suggestion_ids") or "").split("|") if s]


def _train_mu_sd(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    train = [r for r in rows if r.get("split") == "train"]
    if not train:
        train = rows
    x = np.array([[float(r[c]) for c in FEATURE_COLS] for r in train], dtype=np.float64)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Dataset C quality gates.")
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
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--spotcheck-dir",
        type=Path,
        default=REPO_ROOT / "data" / "labeling" / "spotcheck",
    )
    parser.add_argument("--skip-schema-subprocess", action="store_true")
    args = parser.parse_args()

    fails: list[str] = []
    warns: list[str] = []
    info: list[str] = []

    # Phase 8 schema validator
    if not args.skip_schema_subprocess:
        schema_script = BACKEND_DIR / "scripts" / "validate_dataset_schema.py"
        rc = subprocess.call([sys.executable, str(schema_script)], cwd=str(REPO_ROOT))
        if rc != 0:
            fails.append("validate_dataset_schema.py failed (freeze schema first)")
        else:
            info.append("validate_dataset_schema.py OK")

    if not args.dataset.is_file():
        fails.append(f"missing dataset {args.dataset}")
        _write_reports(args.out_dir, fails, warns, info, {}, [])
        return 1

    rows = _load_rows(args.dataset)
    catalog = load_catalog(args.catalog)
    catalog_ids = set(catalog.keys())

    # Catalog / emptiness / y_*
    id_counter: Counter[str] = Counter()
    for r in rows:
        sid = r.get("sample_id", "")
        ids = _parse_ids(r)
        if not ids:
            fails.append(f"{sid}: empty suggestion_ids/priority_order")
        sug = [s for s in (r.get("suggestion_ids") or "").split("|") if s]
        pri = [s for s in (r.get("priority_order") or "").split("|") if s]
        if set(sug) != set(pri):
            fails.append(f"{sid}: suggestion_ids set != priority_order set")
        for i in ids:
            if i not in catalog_ids:
                fails.append(f"{sid}: unknown/inactive suggestion_id {i}")
            id_counter[i] += 1
        for ycol in Y_COLS:
            if r.get(ycol) not in ALLOWED_Y:
                fails.append(f"{sid}: bad {ycol}={r.get(ycol)!r}")

    # Synthetic mix
    sources = { (r.get("source") or "").lower() for r in rows }
    consents = { (r.get("consent_flag") or "").lower() for r in rows }
    has_real = any("lfw" in s or "propr" in s or s == "proprietary_v1" for s in sources) or any(
        "lfw" in c for c in consents
    )
    has_synth = any("synthetic" in s for s in sources) or any("synthetic" in c for c in consents)
    if has_real and has_synth:
        fails.append("synthetic and real faces mixed in the same Dataset C file")
    if has_synth and not has_real:
        warns.append("Dataset C appears fully synthetic")

    # sample_id leakage across splits
    split_map: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        split_map[r["sample_id"]].add(r.get("split", ""))
    leaked = {sid: sorted(spl) for sid, spl in split_map.items() if len(spl) > 1}
    if leaked:
        fails.append(f"sample_id appears in multiple splits: {list(leaked.items())[:5]}")

    # Identity map missing
    warns.append("identity_map_missing: flat lfw_##### IDs — cannot verify person-level leakage")

    # Balance
    rare = sorted([(i, n) for i, n in id_counter.items() if 0 < n < BALANCE_MIN], key=lambda x: x[1])
    if rare:
        warns.append(
            f"balance: {len(rare)} suggestion_ids have <{BALANCE_MIN} positives "
            f"(train with pos_weight; expand data later). Examples: {rare[:10]}"
        )

    # Dead features
    dead = []
    for feat in FEATURE_COLS:
        ys = Counter(r[f"y_{feat}"] for r in rows)
        if ys.get("low", 0) + ys.get("high", 0) == 0:
            dead.append(feat)
    if dead:
        warns.append(f"dead features (y_* never leaves ok): {dead}")

    # Near-duplicates with conflicting labels
    mu, sd = _train_mu_sd(rows)
    feats = np.array([[float(r[c]) for c in FEATURE_COLS] for r in rows], dtype=np.float64)
    zn = (feats - mu) / sd
    label_sets = [frozenset(_parse_ids(r)) for r in rows]
    dup_conflicts = []
    # O(n^2) fine for ~412
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            dist = float(np.linalg.norm(zn[i] - zn[j]))
            if dist < DUP_L2_THRESH and label_sets[i] != label_sets[j]:
                dup_conflicts.append(
                    {
                        "a": rows[i]["sample_id"],
                        "b": rows[j]["sample_id"],
                        "l2": dist,
                    }
                )
                if len(dup_conflicts) >= 20:
                    break
        if len(dup_conflicts) >= 20:
            break
    if dup_conflicts:
        fails.append(
            f"near-duplicate feature vectors with conflicting labels: {dup_conflicts[:5]}"
        )
    else:
        info.append("near-duplicate conflicting-label check OK")

    # Spot-check export
    args.spotcheck_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SPOTCHECK_SEED)
    sample = list(rows)
    rng.shuffle(sample)
    spot = sample[: min(SPOTCHECK_N, len(sample))]
    spot_path = args.spotcheck_dir / "spotcheck_100.csv"
    spot_fields = [
        "sample_id",
        "image_path",
        "split",
        "suggestion_ids",
        "priority_order",
        "label_method",
        "annotator_id",
        "num_non_ok_features",
        "primary_feature",
        "quality_score",
        "reviewed",
        "reviewer_notes",
    ]
    with spot_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=spot_fields, extrasaction="ignore")
        w.writeheader()
        for r in spot:
            out = {k: r.get(k, "") for k in spot_fields}
            out["reviewed"] = ""
            out["reviewer_notes"] = ""
            w.writerow(out)
    info.append(f"spotcheck export: {spot_path} ({len(spot)} rows)")

    summary = {
        "n_rows": len(rows),
        "n_unique_suggestion_ids": len(id_counter),
        "id_counts": dict(id_counter.most_common()),
        "rare_ids": [{ "id": i, "count": n} for i, n in rare],
        "dead_features": dead,
        "dup_conflicts": dup_conflicts,
        "splits": dict(Counter(r.get("split") for r in rows)),
        "sources": dict(Counter(r.get("source") for r in rows)),
    }
    _write_reports(args.out_dir, fails, warns, info, summary, list(id_counter.most_common(20)))

    print("--- Quality gates summary ---")
    for x in info:
        print(f"INFO: {x}")
    for x in warns:
        print(f"WARN: {x}")
    for x in fails:
        print(f"FAIL: {x}")
    status = "FAIL" if fails else "PASS"
    print(f"STATUS: {status} (fails={len(fails)} warns={len(warns)})")
    return 1 if fails else 0


def _write_reports(
    out_dir: Path,
    fails: list[str],
    warns: list[str],
    info: list[str],
    summary: dict,
    top_ids: list,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "FAIL" if fails else "PASS",
        "fails": fails,
        "warns": warns,
        "info": info,
        "summary": summary,
        "top_suggestion_ids": top_ids,
    }
    (out_dir / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Dataset C quality report",
        "",
        f"**Status:** {report['status']}",
        "",
        "## Fails",
    ]
    if fails:
        lines.extend(f"- {x}" for x in fails)
    else:
        lines.append("- (none)")
    lines.extend(["", "## Warns"])
    if warns:
        lines.extend(f"- {x}" for x in warns)
    else:
        lines.append("- (none)")
    lines.extend(["", "## Info"])
    if info:
        lines.extend(f"- {x}" for x in info)
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            "## Summary",
            f"- rows: {summary.get('n_rows')}",
            f"- unique suggestion_ids: {summary.get('n_unique_suggestion_ids')}",
            f"- splits: {summary.get('splits')}",
            "",
        ]
    )
    (out_dir / "quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'quality_report.json'}")
    print(f"Wrote {out_dir / 'quality_report.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
