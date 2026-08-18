"""Per-row label trust weights for suggestion ranker training.

human_v1 priority_order is higher-trust reward/target than rules_v1.
Used offline only — not on the /analyze serve path.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

# Locked Phase 8 weights: human > ensemble > rules.
LABEL_METHOD_WEIGHT: dict[str, float] = {
    "rules_v1": 1.0,
    "ensemble_v1": 2.0,
    "human_v1": 3.0,
}


def label_method_weight(label_method: str | None) -> float:
    """Map Dataset C label_method → sample trust weight (human > ensemble > rules)."""
    key = str(label_method or "rules_v1").strip() or "rules_v1"
    return float(LABEL_METHOD_WEIGHT.get(key, 1.0))


def row_trust_weights(rows: Sequence[Mapping[str, str]]) -> np.ndarray:
    """Build shape (N,) float32 sample weights from each row's label_method column."""
    return np.asarray(
        [label_method_weight(r.get("label_method")) for r in rows],
        dtype=np.float32,
    )


def count_label_methods(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    """Histogram of label_method values — for training reports / dataset audits."""
    counts: dict[str, int] = {}
    for r in rows:
        key = str(r.get("label_method") or "rules_v1").strip() or "rules_v1"
        counts[key] = counts.get(key, 0) + 1
    return counts
