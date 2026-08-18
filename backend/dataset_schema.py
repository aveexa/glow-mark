"""Frozen Dataset B/C schema (v1) + enrich helpers.

Locks column order for geometry (Dataset B) and suggestion (Dataset C) CSVs,
plus derived fields used in offline labeling / training pipelines.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from geometry import FEATURE_COLS, FEATURE_CONTRACT_VERSION

# Allowed categorical values for schema validation / labeling.
ALLOWED_SPLITS = frozenset({"train", "val", "test"})
ALLOWED_Y = frozenset({"low", "ok", "high"})
ALLOWED_LABEL_METHODS = frozenset({"rules_v1", "human_v1", "ensemble_v1"})
ALLOWED_GEOMETRY_LABEL_METHODS = frozenset({"percentile_p20_p80"})

# Per-feature class columns: y_<feature_name>.
Y_COLS: list[str] = [f"y_{c}" for c in FEATURE_COLS]

# Dataset B column order (frozen) — geometry features + labels + enrich fields.
GEOMETRY_DATASET_COLUMNS: list[str] = [
    "sample_id",
    "image_path",
    "split",
    *FEATURE_COLS,
    *Y_COLS,
    "label_method",
    "consent_flag",
    "source",
    "feature_contract_version",
    "yaw_deg",
    "pitch_deg",
    "num_non_ok_features",
    "primary_feature",
    "quality_score",
]

# Dataset C column order (frozen) — B fields + suggestion_ids / priority_order.
SUGGESTION_DATASET_COLUMNS: list[str] = [
    "sample_id",
    "image_path",
    "split",
    *FEATURE_COLS,
    *Y_COLS,
    "suggestion_ids",
    "priority_order",
    "annotator_id",
    "label_method",
    "consent_flag",
    "source",
    "feature_contract_version",
    "yaw_deg",
    "pitch_deg",
    "num_non_ok_features",
    "primary_feature",
    "quality_score",
]


def compute_train_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, tuple[float, float]]:
    """Fit per-feature (mu, sigma) on train split only — used for z-style enrich."""
    train = [r for r in rows if str(r.get("split", "")) == "train"]
    if not train:
        raise ValueError("No train rows for stats")
    stats: dict[str, tuple[float, float]] = {}
    for feat in FEATURE_COLS:
        vals = [float(r[feat]) for r in train]
        n = len(vals)
        mu = sum(vals) / n
        var = sum((v - mu) ** 2 for v in vals) / n
        sd = math.sqrt(var)
        stats[feat] = (mu, sd)
    return stats


def quality_score_from_pose(yaw_deg: float, pitch_deg: float) -> float:
    """Pose quality proxy in [0, 1]: exp(-(|yaw|+|pitch|)/30)."""
    return float(max(0.0, min(1.0, math.exp(-(abs(yaw_deg) + abs(pitch_deg)) / 30.0))))


def num_non_ok_features(row: Mapping[str, Any]) -> int:
    """Count y_* labels that are low or high (non-ok geometry classes)."""
    return sum(1 for f in FEATURE_COLS if str(row.get(f"y_{f}", "ok")) in {"low", "high"})


def primary_feature(row: Mapping[str, Any], stats: Mapping[str, tuple[float, float]]) -> str:
    """Pick the feature with largest |z| vs train stats (main geometric outlier)."""
    best_feat = FEATURE_COLS[0]
    best_z = -1.0
    for feat in FEATURE_COLS:
        mu, sd = stats[feat]
        x = float(row[feat])
        z = abs((x - mu) / sd) if sd > 0 else 0.0
        if z > best_z:
            best_z = z
            best_feat = feat
    return best_feat


def enrich_row(row: MutableMapping[str, Any], stats: Mapping[str, tuple[float, float]]) -> dict[str, Any]:
    """Return a new dict with derived recommended columns filled (contract, pose, primary, quality)."""
    out = dict(row)
    out["feature_contract_version"] = out.get("feature_contract_version") or FEATURE_CONTRACT_VERSION
    yaw = float(out.get("yaw_deg") or 0.0)
    pitch = float(out.get("pitch_deg") or 0.0)
    out["yaw_deg"] = yaw
    out["pitch_deg"] = pitch
    out["num_non_ok_features"] = num_non_ok_features(out)
    out["primary_feature"] = primary_feature(out, stats)
    out["quality_score"] = quality_score_from_pose(yaw, pitch)
    return out


def reorder_row(row: Mapping[str, Any], columns: Iterable[str]) -> dict[str, Any]:
    """Project a row onto frozen column order; missing keys become empty string."""
    return {c: row.get(c, "") for c in columns}
