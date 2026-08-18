"""Shared catalog rules mapper for Dataset C (rules_v1) and labeling packets.

Maps geometry y_* classes to ranked suggestion_ids using the active catalog.
Used for offline labeling / rules baselines; serve path uses the ranker instead.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

from geometry import FEATURE_COLS

FALLBACK_IDS = ["SUG_OK_KEEP_01", "SUG_OK_SKIN_PREP_01"]
SEVERITY_RANK = {"info": 0, "mild": 1}
DEFAULT_TOP_K = 4


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


def trigger_index(catalog: Mapping[str, Mapping[str, str]]) -> dict[tuple[str, str], list[str]]:
    """Index catalog as (feature, trigger_class) → suggestion_ids (stable catalog order)."""
    index: dict[tuple[str, str], list[str]] = {}
    for sid, row in catalog.items():
        key = (row["feature"].strip(), row["trigger_class"].strip())
        index.setdefault(key, []).append(sid)
    return index


def map_candidates(
    row: Mapping[str, str],
    catalog: Mapping[str, Mapping[str, str]],
    index: dict[tuple[str, str], list[str]] | None = None,
    *,
    top_k: int | None = DEFAULT_TOP_K,
) -> list[str]:
    """Map geometry y_* classes → ranked suggestion_ids (severity + insertion order).

    ``top_k=None`` returns the full ranked candidate list (for labeling packets).
    Falls back to FALLBACK_IDS when no triggers fire.
    """
    if index is None:
        index = trigger_index(catalog)

    ordered: list[str] = []
    seen: set[str] = set()

    def add(sid: str) -> None:
        """Deduped append into the candidate list (catalog membership required)."""
        if sid in catalog and sid not in seen:
            ordered.append(sid)
            seen.add(sid)

    if row.get("y_symmetry_error") == "high":
        for sid in index.get(("symmetry_error", "high"), []):
            add(sid)

    for feat in FEATURE_COLS:
        y = row.get(f"y_{feat}", "ok")
        if y not in {"low", "high"}:
            continue
        if feat == "symmetry_error" and y == "high":
            continue
        for sid in index.get((feat, y), []):
            add(sid)

    if not ordered:
        for sid in FALLBACK_IDS:
            add(sid)

    insertion = {sid: i for i, sid in enumerate(ordered)}
    ordered.sort(
        key=lambda sid: (
            SEVERITY_RANK.get(str(catalog[sid].get("severity", "mild")), 99),
            insertion[sid],
        )
    )

    if top_k is None:
        return ordered
    return ordered[: int(top_k)]
