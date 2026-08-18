#!/usr/bin/env python3
"""Validate data/catalogs/suggestions.csv against feature contract v0/v1 rules.

Usage (from repo root):

  python backend/scripts/validate_suggestion_catalog.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from geometry import FEATURE_COLS  # noqa: E402

CATALOG_PATH = REPO_ROOT / "data" / "catalogs" / "suggestions.csv"
REQUIRED_COLS = [
    "suggestion_id",
    "feature",
    "trigger_class",
    "category",
    "severity",
    "approved_text",
    "forbidden",
    "active",
]
ALLOWED_FEATURES = set(FEATURE_COLS) | {"general", "capture"}
ALLOWED_CLASSES = {"low", "ok", "high"}
ALLOWED_SEVERITY = {"mild", "info"}
BOOL_LIKE = {"true", "false"}
BANNED_SUBSTRINGS = [
    "surgery",
    "surgical",
    "botox",
    "filler inject",
    "rhinoplasty",
    "implant",
    "guarantee",
    "diagnose",
    "medical",
    "clinical procedure",
]
# Intentionally uncovered for v0 (documented in suggestion_trigger_map.md).
INTENTIONAL_GAPS = {
    ("symmetry_error", "low"),
    ("nose_tip_deviation_ratio", "low"),
}


def _parse_bool(value: str, field: str, sid: str, errors: list[str]) -> None:
    if value.strip().lower() not in BOOL_LIKE:
        errors.append(f"{sid}: {field} must be true|false, got {value!r}")


def main() -> int:
    if not CATALOG_PATH.is_file():
        print(f"ERROR: catalog not found: {CATALOG_PATH}", file=sys.stderr)
        return 1

    with CATALOG_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("ERROR: empty catalog", file=sys.stderr)
            return 1
        missing = [c for c in REQUIRED_COLS if c not in reader.fieldnames]
        if missing:
            print(f"ERROR: missing columns: {missing}", file=sys.stderr)
            return 1
        rows = list(reader)

    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    coverage: dict[tuple[str, str], str] = {}

    for i, row in enumerate(rows, start=2):
        sid = (row.get("suggestion_id") or "").strip()
        if not sid:
            errors.append(f"line {i}: empty suggestion_id")
            continue
        if sid in seen_ids:
            errors.append(f"{sid}: duplicate suggestion_id")
        seen_ids.add(sid)

        feature = (row.get("feature") or "").strip()
        trigger = (row.get("trigger_class") or "").strip()
        severity = (row.get("severity") or "").strip()
        text = (row.get("approved_text") or "").strip()
        forbidden = (row.get("forbidden") or "").strip()
        active = (row.get("active") or "").strip()

        if feature not in ALLOWED_FEATURES:
            errors.append(f"{sid}: feature {feature!r} not in FEATURE_COLS|general|capture")
        if trigger not in ALLOWED_CLASSES:
            errors.append(f"{sid}: trigger_class {trigger!r} invalid")
        if severity not in ALLOWED_SEVERITY:
            errors.append(f"{sid}: severity {severity!r} invalid")
        if not text:
            errors.append(f"{sid}: approved_text empty")
        _parse_bool(forbidden, "forbidden", sid, errors)
        _parse_bool(active, "active", sid, errors)

        low_text = text.lower()
        for banned in BANNED_SUBSTRINGS:
            if banned in low_text:
                errors.append(f"{sid}: banned substring {banned!r} in approved_text")

        if forbidden.strip().lower() == "true" and active.strip().lower() == "true":
            warnings.append(f"{sid}: forbidden=true but active=true (blocked at serve later)")

        if feature in FEATURE_COLS and trigger in {"low", "high"}:
            key = (feature, trigger)
            if key in coverage:
                warnings.append(
                    f"{sid}: duplicate coverage for {feature}/{trigger} "
                    f"(already {coverage[key]})"
                )
            else:
                coverage[key] = sid

    # Coverage matrix
    print(f"Catalog rows: {len(rows)} unique_ids: {len(seen_ids)}")
    print("Coverage matrix (feature × low/high):")
    missing_cells: list[tuple[str, str]] = []
    for feat in FEATURE_COLS:
        low_id = coverage.get((feat, "low"), "—")
        high_id = coverage.get((feat, "high"), "—")
        print(f"  {feat}: low={low_id}  high={high_id}")
        for cls in ("low", "high"):
            if (feat, cls) not in coverage:
                missing_cells.append((feat, cls))

    unexpected_gaps = [c for c in missing_cells if c not in INTENTIONAL_GAPS]
    intentional = [c for c in missing_cells if c in INTENTIONAL_GAPS]
    if intentional:
        print(f"Intentional gaps ({len(intentional)}): {intentional}")
    if unexpected_gaps:
        errors.append(f"Unexpected coverage gaps: {unexpected_gaps}")

    for w in warnings:
        print(f"WARN: {w}")
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)

    if errors:
        print("FAILED", file=sys.stderr)
        return 1
    print("OK catalog validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
