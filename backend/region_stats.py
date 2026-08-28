"""Mixture-weighted lookup over the region-conditioned reference statistics.

Reads data/processed/region_reference_stats.csv (built by
backend/scripts/build_region_reference.py) and blends the per-region cells into a
single set of norms for one face, weighted by that face's region mixture:

    p20_f = Σ_r  p_r · p20[r][f]
    p80_f = Σ_r  p_r · p80[r][f]
    mu_f  = Σ_r  p_r · mu[r][f]
    sd_f  = sqrt( Σ_r p_r · (sd[r][f]² + mu[r][f]²) − mu_f² )

Serve always reads the ``pooled`` sex arm, which the build derives at fixed 50/50
weights — the pipeline never infers sex from a photograph.

Thin cells (n < MIN_CELL_N) fall back to the global row for that feature, so a
sparsely-populated region borrows the baseline rather than reporting noise.
Every entry point is fail-soft: a missing CSV yields None and callers keep their
pre-region behaviour.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_STATS_PATH = _REPO_ROOT / "data" / "processed" / "region_reference_stats.csv"

GLOBAL_ARM = "global"
POOLED_GENDER = "pooled"
MIN_CELL_N = 200

# The raw beauty-MLP score is carried as an extra "feature" row so the CSV schema
# stays as specified. It is not a geometry feature and must never reach the
# p20/p80 classifier or the UI gauges — reference_features() excludes it.
BEAUTY_STAT = "beauty_score_raw"

_STAT_KEYS = ("p20", "p80", "mu", "sigma")


@lru_cache(maxsize=1)
def load_reference_stats() -> Dict[str, Dict[str, Dict[str, float]]] | None:
    """``{region: {feature: {p20, p80, mu, sigma, n}}}`` for the pooled arm, or None."""
    import pandas as pd

    if not REFERENCE_STATS_PATH.is_file():
        return None
    frame = pd.read_csv(REFERENCE_STATS_PATH)
    pooled = frame[frame["gender"] == POOLED_GENDER]
    if pooled.empty:
        return None

    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in pooled.itertuples():
        table.setdefault(str(row.region), {})[str(row.feature)] = {
            "p20": float(row.p20),
            "p80": float(row.p80),
            "mu": float(row.mu),
            "sigma": float(row.sigma),
            "n": float(row.n),
        }
    return table


def reference_features() -> tuple[str, ...]:
    """Geometry feature names the table covers (chin_length_ratio and the beauty row excluded)."""
    table = load_reference_stats()
    if not table or GLOBAL_ARM not in table:
        return ()
    return tuple(f for f in table[GLOBAL_ARM] if f != BEAUTY_STAT)


def _cell(table, region: str, feature: str) -> Dict[str, float] | None:
    """Helper: one region's cell for a feature, falling back to global when thin/absent."""
    cell = table.get(region, {}).get(feature)
    if cell is None or cell["n"] < MIN_CELL_N:
        return table.get(GLOBAL_ARM, {}).get(feature)
    return cell


def mixture_stats(
    region_weights: Mapping[str, float] | None = None,
) -> Dict[str, Dict[str, float]] | None:
    """Region-blended ``{feature: {p20, p80, mu, sigma}}``, or None if unavailable.

    ``region_weights=None`` returns the global arm unchanged.
    """
    table = load_reference_stats()
    if not table:
        return None

    features = reference_features()
    if not features:
        return None

    if not region_weights:
        globals_ = table.get(GLOBAL_ARM)
        if not globals_:
            return None
        return {f: {k: globals_[f][k] for k in _STAT_KEYS} for f in features}

    weights = {r: float(w) for r, w in region_weights.items() if r in table and float(w) > 0.0}
    total = sum(weights.values())
    if total <= 0.0:
        return mixture_stats(None)

    out: Dict[str, Dict[str, float]] = {}
    for feature in features:
        cells = {r: _cell(table, r, feature) for r in weights}
        cells = {r: c for r, c in cells.items() if c is not None}
        norm = sum(weights[r] for r in cells)
        if norm <= 0.0:
            continue

        mu = sum(weights[r] * cells[r]["mu"] for r in cells) / norm
        second = sum(
            weights[r] * (cells[r]["sigma"] ** 2 + cells[r]["mu"] ** 2) for r in cells
        ) / norm
        out[feature] = {
            "p20": sum(weights[r] * cells[r]["p20"] for r in cells) / norm,
            "p80": sum(weights[r] * cells[r]["p80"] for r in cells) / norm,
            "mu": mu,
            "sigma": math.sqrt(max(second - mu * mu, 0.0)),
        }
    return out or None


def z_scores(
    feats: Mapping[str, float],
    region_weights: Mapping[str, float] | None = None,
) -> Dict[str, float] | None:
    """Region-normalized z per feature, or None when the reference table is absent."""
    stats = mixture_stats(region_weights)
    if not stats:
        return None
    return {
        f: (float(feats[f]) - s["mu"]) / (s["sigma"] + 1e-06)
        for f, s in stats.items()
        if f in feats
    }


def beauty_stats(
    region_weights: Mapping[str, float] | None = None,
) -> Dict[str, float] | None:
    """Region-blended ``{p20, p80, mu, sigma}`` for the raw beauty score, or None."""
    table = load_reference_stats()
    if not table or BEAUTY_STAT not in table.get(GLOBAL_ARM, {}):
        return None

    if not region_weights:
        cell = table[GLOBAL_ARM][BEAUTY_STAT]
        return {k: cell[k] for k in _STAT_KEYS}

    weights = {r: float(w) for r, w in region_weights.items() if r in table and float(w) > 0.0}
    cells = {r: _cell(table, r, BEAUTY_STAT) for r in weights}
    cells = {r: c for r, c in cells.items() if c is not None}
    norm = sum(weights[r] for r in cells)
    if norm <= 0.0:
        return beauty_stats(None)

    mu = sum(weights[r] * cells[r]["mu"] for r in cells) / norm
    second = sum(weights[r] * (cells[r]["sigma"] ** 2 + cells[r]["mu"] ** 2) for r in cells) / norm
    return {
        "p20": sum(weights[r] * cells[r]["p20"] for r in cells) / norm,
        "p80": sum(weights[r] * cells[r]["p80"] for r in cells) / norm,
        "mu": mu,
        "sigma": math.sqrt(max(second - mu * mu, 0.0)),
    }
