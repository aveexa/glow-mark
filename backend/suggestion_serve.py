"""Serve path: geometry feats + y_* → suggestion ranker → catalog text.

Called from inference.analyze_image_bytes. Class one-hots come from percentile
p20/p80 cutoffs — the region-conditioned reference statistics when they are
available, else the static rules CSV. Returns [] if no checkpoint is present.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

from geometry import FEATURE_COLS
from region_stats import mixture_stats
from suggestion_model import decode_top_k, encode_features, load_checkpoint
from suggestion_rules import load_catalog

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_CKPT = MODELS_DIR / "suggestion_ranker.pt"
# Prefer production name; fall back to bakeoff / named stage checkpoints on disk.
CKPT_FALLBACKS = (
    DEFAULT_CKPT,
    MODELS_DIR / "suggestion_ranker_listnet.pt",
    MODELS_DIR / "suggestion_ranker_rl_v1.pt",
    MODELS_DIR / "suggestion_ranker_bce_v1.pt",
)
DEFAULT_RULES = REPO_ROOT / "data" / "processed" / "suggestion_mapping_rules.csv"
DEFAULT_CATALOG = REPO_ROOT / "data" / "catalogs" / "suggestions.csv"

# Features held back from every user-facing output.
#
# chin_length_ratio is computed from the same two landmarks as
# lowerface_length_ratio — p4 (nose tip) and p152 (chin), over face height — and is
# numerically identical to it in every row. Surfacing both shows the user the same
# measurement twice, and because the reference table excludes it the duplicate also
# carries no region statistics, so it reports the neutral 0.5 confidence sentinel
# beside features showing 84-95%.
#
# It is filtered *after* ranking, never before:
#   - geometry.py still computes all 24; the feature contract stays v1.
#   - the ranker still receives all 24 positions; dropping one would change the
#     input shape from 96 and require a retrain.
RESPONSE_EXCLUDED_FEATURES: tuple[str, ...] = ("chin_length_ratio",)


def resolve_ranker_checkpoint(ckpt_path: Path | str | None = None) -> Path | None:
    """Pick the first existing ranker .pt (explicit path, else prod then fallbacks)."""
    if ckpt_path is not None:
        p = Path(ckpt_path)
        return p if p.is_file() else None
    for candidate in CKPT_FALLBACKS:
        if candidate.is_file():
            return candidate
    return None


def load_threshold_rules(path: Path | str = DEFAULT_RULES) -> Dict[str, tuple[float, float]]:
    """Load per-feature (p20 low, p80 high) thresholds used to derive y_* classes."""
    path = Path(path)
    if not path.is_file():
        return {}
    rules: Dict[str, tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            feat = row["feature"].strip()
            rules[feat] = (float(row["low_threshold_p20"]), float(row["high_threshold_p80"]))
    return rules


def classes_from_thresholds(
    feats: Mapping[str, float],
    rules: Mapping[str, tuple[float, float]] | None = None,
    region_weights: Mapping[str, float] | None = None,
) -> Dict[str, str]:
    """Map continuous feats → low/ok/high via p20/p80 cutoffs.

    Cutoffs come from the region-conditioned reference statistics blended by
    ``region_weights``. Features the reference table does not cover (currently only
    ``chin_length_ratio``) fall back to the static rules CSV, and anything covered
    by neither is reported ``ok``.
    """
    region = mixture_stats(region_weights) or {}
    rules = rules or {}

    out: Dict[str, str] = {}
    for feat in FEATURE_COLS:
        val = float(feats[feat])
        if feat in region:
            lo, hi = region[feat]["p20"], region[feat]["p80"]
        elif feat in rules:
            lo, hi = rules[feat]
        else:
            out[feat] = "ok"
            continue
        if val < lo:
            out[feat] = "low"
        elif val > hi:
            out[feat] = "high"
        else:
            out[feat] = "ok"
    return out


@lru_cache(maxsize=1)
def _load_ranker_bundle(
    ckpt_path: str,
    catalog_path: str,
    rules_path: str,
) -> Dict[str, Any] | None:
    """Helper: cached bundle of ranker model + catalog + threshold rules."""
    ckpt = Path(ckpt_path)
    if not ckpt.is_file():
        return None
    bundle = load_checkpoint(ckpt)
    catalog = load_catalog(catalog_path)
    rules = load_threshold_rules(rules_path)
    return {
        **bundle,
        "catalog": catalog,
        "rules": rules,
    }


def predict_suggestions(
    feats: Mapping[str, float],
    *,
    top_k: int = 4,
    y_classes: Mapping[str, str] | None = None,
    region_weights: Mapping[str, float] | None = None,
    ckpt_path: Path | str | None = None,
    catalog_path: Path | str = DEFAULT_CATALOG,
    rules_path: Path | str = DEFAULT_RULES,
) -> List[Dict[str, Any]]:
    """Return [{id, text, confidence}] or [] if checkpoint missing.

    Class one-hots default to the region-conditioned p20/p80 cutoffs
    (``classes_from_thresholds``) — callers must pass ``y_classes`` to override.
    """
    resolved = resolve_ranker_checkpoint(ckpt_path)
    if resolved is None:
        return []
    bundle = _load_ranker_bundle(str(resolved), str(catalog_path), str(rules_path))
    if bundle is None:
        return []

    rules = bundle["rules"]
    if y_classes is None:
        y_classes = classes_from_thresholds(feats, rules, region_weights)

    x = encode_features(feats, y_classes, bundle["feat_mu"], bundle["feat_sd"])
    model = bundle["model"]
    with torch.no_grad():
        logits = model(torch.from_numpy(x)).cpu().numpy().reshape(-1)

    catalog = bundle["catalog"]
    out: List[Dict[str, Any]] = []
    # Rank over the whole vocabulary, then drop excluded-feature suggestions and take
    # top_k — truncating first would silently return fewer than k.
    ranked = decode_top_k(logits, bundle["suggestion_ids"], k=len(bundle["suggestion_ids"]))
    for sid, conf in ranked:
        row = catalog.get(sid, {})
        # An id absent from the catalog keeps its prior behaviour (empty text) —
        # only a known excluded-feature row is dropped.
        if row.get("feature", "").strip() in RESPONSE_EXCLUDED_FEATURES:
            continue
        # category, severity and trigger_class come from the catalog row already in
        # hand. suggestion_summary groups on them; carrying them here avoids a second
        # catalog lookup and keeps the grouping keyed to the same row that produced
        # the text.
        out.append({
            "id": sid,
            "text": row.get("approved_text", ""),
            "confidence": round(conf, 4),
            "category": row.get("category", "").strip(),
            "severity": row.get("severity", "").strip(),
            "trigger_class": row.get("trigger_class", "").strip(),
        })
        if len(out) >= top_k:
            break
    return out
