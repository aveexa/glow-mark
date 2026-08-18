"""Catalog loader (vendored for Cloud Run image)."""

from __future__ import annotations

import csv
from pathlib import Path


def load_catalog(path: Path | str) -> dict[str, dict[str, str]]:
    """Load active, non-forbidden suggestion catalog rows keyed by suggestion_id."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("active", "").strip().lower() != "true":
            continue
        if row.get("forbidden", "").strip().lower() == "true":
            continue
        sid = row["suggestion_id"].strip()
        by_id[sid] = row
    return by_id
