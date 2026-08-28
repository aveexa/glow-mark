"""Acceptance checks for data/processed/region_reference_stats.csv.

  1. Shape: ~500 rows, 7 regions + global, three sex arms, chin_length_ratio absent.
  2. Every region x gender cell has n > 400.
  3. Medians sit in the same neighbourhood as the frozen v1 baseline.
  4. Pooled rows really are the fixed 50/50 mixture of the male and female cells.

Determinism (re-running the build reproduces the file) is checked separately by
rebuilding with a fixed --limit and diffing.

Run from:  repo root
    python backend/scripts/check_region_reference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import numpy as np
import pandas as pd

from region_stats import BEAUTY_STAT, GLOBAL_ARM, POOLED_GENDER

STATS = REPO / "data" / "processed" / "region_reference_stats.csv"
BASELINE = REPO / "data" / "interim" / "geometry_features" / "geometry_features.csv"
MIN_CELL_N = 400
POOLED_TOLERANCE = 1e-09

# Features whose new-vs-legacy scale gap is already documented: the legacy baseline
# came from the removed FaceMesh API on a different, gated population, and
# sanity_check_extraction.py reports the same five. They are compared but not
# allowed to fail this check.
KNOWN_EXTRACTION_DIFFS = {
    "symmetry_error",
    "eye_tilt_deg",
    "nose_tip_deviation_ratio",
    "mouth_corner_tilt_deg",
    "lip_thickness_ratio",
}

ok = True


def check(label: str, passed: bool, detail: str = "") -> None:
    global ok
    ok = ok and passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


def main() -> int:
    df = pd.read_csv(STATS)
    base = pd.read_csv(BASELINE)
    features = sorted(df["feature"].unique())
    geometry = [f for f in features if f != BEAUTY_STAT]

    print("1. shape")
    check("row count in the expected range", 400 <= len(df) <= 700, f"{len(df)} rows")
    check("7 regions + global", df["region"].nunique() == 8, str(sorted(df["region"].unique())))
    check("male / female / pooled arms", set(df["gender"]) == {"male", "female", POOLED_GENDER})
    check("chin_length_ratio excluded", "chin_length_ratio" not in features)
    check("23 geometry features", len(geometry) == 23, f"{len(geometry)} + {BEAUTY_STAT}")

    print("\n2. cell sizes")
    cells = df[["region", "gender", "n"]].drop_duplicates()
    sex_cells = cells[cells["gender"] != POOLED_GENDER]
    check(f"every region x gender cell n > {MIN_CELL_N}", bool((sex_cells["n"] > MIN_CELL_N).all()),
          f"min n = {int(sex_cells['n'].min())}")
    print(cells.pivot(index="region", columns="gender", values="n").to_string())

    print("\n3. medians vs the frozen v1 baseline (different populations, so we")
    print("   check the same neighbourhood, not equality)")
    pooled = df[(df["region"] == GLOBAL_ARM) & (df["gender"] == POOLED_GENDER)].set_index("feature")
    print(f"   {'feature':<26}{'baseline':>11}{'global mu':>11}{'ratio':>8}")
    off = []
    for f in geometry:
        if f not in base.columns:
            continue
        b = float(base[f].median())
        m = float(pooled.loc[f, "mu"])
        ratio = m / b if abs(b) > 1e-09 else float("nan")
        # Signed angles hover near zero, so a ratio is meaningless for them; compare
        # against the baseline's own spread instead. Roll autocorrect is expected to
        # pull them toward 0, since the legacy baseline never de-rotated.
        if f.endswith("_deg") and abs(b) < 5.0:
            near = abs(m - b) <= 2.0 * float(base[f].std())
        else:
            near = 0.5 <= ratio <= 2.0
        flag = "" if near else ("known diff" if f in KNOWN_EXTRACTION_DIFFS else "OFF")
        if not near and f not in KNOWN_EXTRACTION_DIFFS:
            off.append(f)
        print(f"   {f:<26}{b:>11.4f}{m:>11.4f}{ratio:>8.2f}  {flag}")
    check("medians in the same neighbourhood", not off, f"off: {off}" if off else
          "(features marked 'known diff' are the documented legacy-vs-Tasks-API gap)")

    print("\n4. pooled rows are the fixed 50/50 sex mixture")
    worst = 0.0
    for region in df["region"].unique():
        r = df[df["region"] == region]
        m = r[r["gender"] == "male"].set_index("feature")
        f_ = r[r["gender"] == "female"].set_index("feature")
        p = r[r["gender"] == POOLED_GENDER].set_index("feature")
        for feat in features:
            mu = 0.5 * (m.loc[feat, "mu"] + f_.loc[feat, "mu"])
            second = 0.5 * (m.loc[feat, "sigma"] ** 2 + m.loc[feat, "mu"] ** 2
                            + f_.loc[feat, "sigma"] ** 2 + f_.loc[feat, "mu"] ** 2)
            sigma = np.sqrt(max(second - mu * mu, 0.0))
            worst = max(worst, abs(mu - p.loc[feat, "mu"]), abs(sigma - p.loc[feat, "sigma"]))
    check("pooled mu/sigma match the mixture formula", worst < POOLED_TOLERANCE,
          f"max deviation {worst:.2e}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
