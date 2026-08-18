"""Serve path: geometry feats → percentile one-hots → ranker → catalog text."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch

from geometry_cols import FEATURE_COLS
from suggestion_model import decode_top_k, encode_features, load_checkpoint
from suggestion_rules import load_catalog


def load_threshold_rules(path: Path | str) -> Dict[str, tuple[float, float]]:
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
    rules: Mapping[str, tuple[float, float]],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for feat in FEATURE_COLS:
        val = float(feats[feat])
        if feat in rules:
            lo, hi = rules[feat]
            if val < lo:
                out[feat] = "low"
            elif val > hi:
                out[feat] = "high"
            else:
                out[feat] = "ok"
        else:
            out[feat] = "ok"
    return out


@lru_cache(maxsize=1)
def _load_ranker_bundle(
    ckpt_path: str,
    catalog_path: str,
    rules_path: str,
) -> Dict[str, Any] | None:
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
    ckpt_path: Path | str,
    catalog_path: Path | str,
    rules_path: Path | str,
) -> List[Dict[str, Any]]:
    """Return [{id, text, confidence}] or [] if checkpoint missing.

    Class one-hots default to percentile p20/p80 rules (``classes_from_thresholds``),
    not Feature MLP argmax.
    """
    resolved = Path(ckpt_path)
    if not resolved.is_file():
        return []
    bundle = _load_ranker_bundle(str(resolved), str(catalog_path), str(rules_path))
    if bundle is None:
        return []

    rules = bundle["rules"]
    if y_classes is None:
        y_classes = classes_from_thresholds(feats, rules)

    x = encode_features(feats, y_classes, bundle["feat_mu"], bundle["feat_sd"])
    model = bundle["model"]
    with torch.no_grad():
        logits = model(torch.from_numpy(x)).cpu().numpy().reshape(-1)

    catalog = bundle["catalog"]
    out: List[Dict[str, Any]] = []
    for sid, conf in decode_top_k(logits, bundle["suggestion_ids"], k=top_k):
        text = catalog.get(sid, {}).get("approved_text", "")
        out.append({"id": sid, "text": text, "confidence": round(conf, 4)})
    return out
