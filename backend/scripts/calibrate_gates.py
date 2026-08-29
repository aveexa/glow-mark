"""Calibrate the pose and neutrality gates. Produces the final config + E7 result.

Reads the cached extraction from diagnose_gates.py -- no re-extraction, runs in
seconds. Re-run diagnose_gates.py with a larger n for tighter numbers.

Three outputs:
  1. Pose gate sweep -- what each limit costs in yield.
  2. Global vs region-relative neutrality thresholds -- the E7 result.
  3. gate_config.json -- the calibrated thresholds, ready to load in gates.py.

Place in:  backend/scripts/calibrate_gates.py
Run from:  repo root
    python backend/scripts/calibrate_gates.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
SRC = REPO / "data" / "interim" / "diagnose_gates.csv"
OUT = REPO / "data" / "interim" / "gate_config.json"

# blendshape -> the message shown when it trips
SIGNALS = {
    "jawOpen":          "Please close your mouth",
    "mouthSmileLeft":   "Please relax your mouth",
    "mouthSmileRight":  "Please relax your mouth",
    "mouthPucker":      "Please relax your lips",
    "mouthPressLeft":   "Please relax your lips",
    "mouthPressRight":  "Please relax your lips",
    "mouthFunnel":      "Please relax your lips",
    "browInnerUp":      "Please relax your eyebrows",
    "browDownLeft":     "Please relax your eyebrows",
    "browDownRight":    "Please relax your eyebrows",
    "browOuterUpLeft":  "Please relax your eyebrows",
    "browOuterUpRight": "Please relax your eyebrows",
    "eyeBlinkLeft":     "Please keep your eyes open",
    "eyeBlinkRight":    "Please keep your eyes open",
    "eyeSquintLeft":    "Please relax your eyes",
    "eyeSquintRight":   "Please relax your eyes",
}

# per-signal percentile: how much of the reference population each cut admits.
# jawOpen is stricter because an open mouth genuinely wrecks the jaw/lip features;
# the rest are looser because they mostly track morphology, not expression.
PCTL = {s: 0.98 for s in SIGNALS}
PCTL["jawOpen"] = 0.96
PCTL["mouthSmileLeft"] = 0.97
PCTL["mouthSmileRight"] = 0.97

# CLIP zero-shot cut for the realness gate (softmax mass on the two human prompts).
# Not calibrated here — this script only sees FairFace, which has no negative class.
# The value comes from backend/scripts/calibrate_realness.py; it is mirrored so that
# re-running this script does not silently reset the threshold. Keep the two in sync.
REALNESS_MIN_P_PHOTO = 0.3977


def rejected_mask(df, thr):
    """thr: dict signal -> scalar, or dict signal -> Series aligned to df.index."""
    m = pd.Series(False, index=df.index)
    for s in SIGNALS:
        if s in df:
            t = thr[s]
            m |= df[s] > (t if np.isscalar(t) else t.reindex(df.index).values)
    return m


def main():
    if not SRC.exists():
        sys.exit(f"Missing {SRC}\nRun diagnose_gates.py first.")
    df = pd.read_csv(SRC)
    print(f"loaded {len(df)} extracted faces\n")

    # ── 1. pose sweep ───────────────────────────────────────────────────
    print("=" * 70)
    print("1.  POSE GATE SWEEP")
    print("=" * 70)
    print(f"{'yaw':>6}{'pitch':>7}{'kept':>8}{'yield':>9}")
    best = None
    for yaw, pitch in [(12, 10), (15, 15), (18, 15), (20, 15), (20, 18), (25, 20), (30, 25)]:
        k = ((df.yaw.abs() <= yaw) & (df.pitch.abs() <= pitch)).sum()
        pct = 100 * k / len(df)
        mark = ""
        if best is None and pct >= 70:
            best, mark = (yaw, pitch), "  <- recommended"
        print(f"{yaw:>6}{pitch:>7}{k:>8}{pct:>8.1f}%{mark}")
    if best is None:
        best = (25, 20)
    yaw_lim, pitch_lim = best
    g = df[(df.yaw.abs() <= yaw_lim) & (df.pitch.abs() <= pitch_lim)].copy()
    print(f"\nusing |yaw|<={yaw_lim}, |pitch|<={pitch_lim}  ->  {len(g)} faces")
    print(f"pitch median: {df.pitch.median():.2f} deg  "
          f"(legacy heuristic sat near -10 deg; near 0 means the matrix fixed it)")

    # ── 2. global thresholds ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2.  GLOBAL THRESHOLDS  (percentile of the pooled reference)")
    print("=" * 70)
    glob = {s: float(g[s].quantile(PCTL[s])) for s in SIGNALS if s in g}
    print(f"{'signal':<20}{'pctl':>6}{'thresh':>9}{'rejects':>9}")
    for s, t in glob.items():
        print(f"{s:<20}{PCTL[s]:>6.2f}{t:>9.3f}{100*(g[s] > t).mean():>8.1f}%")

    gm = rejected_mask(g, glob)
    gstat = g.assign(rej=gm).groupby("race")["rej"].agg(n="size", r="sum")
    gstat["rate_%"] = 100 * gstat.r / gstat.n
    g_spread = gstat["rate_%"].max() - gstat["rate_%"].min()
    print(f"\noverall rejection: {100*gm.mean():.1f}%")
    print(gstat[["n", "rate_%"]].to_string(float_format=lambda x: f"{x:7.1f}"))
    print(f"spread across regions: {g_spread:.1f} pts")

    # ── 3. region-relative thresholds ───────────────────────────────────
    print("\n" + "=" * 70)
    print("3.  REGION-RELATIVE THRESHOLDS   (E7)")
    print("=" * 70)
    per = {s: g.groupby("race")[s].quantile(PCTL[s]) for s in SIGNALS if s in g}
    rmap = {s: g["race"].map(per[s]) for s in per}
    rm = rejected_mask(g, rmap)
    rstat = g.assign(rej=rm).groupby("race")["rej"].agg(n="size", r="sum")
    rstat["rate_%"] = 100 * rstat.r / rstat.n
    r_spread = rstat["rate_%"].max() - rstat["rate_%"].min()

    cmp = pd.DataFrame({"n": gstat.n,
                        "global_%": gstat["rate_%"],
                        "region_%": rstat["rate_%"]})
    cmp["delta"] = cmp["region_%"] - cmp["global_%"]
    print(cmp.to_string(float_format=lambda x: f"{x:8.1f}"))
    print(f"\noverall rejection: {100*rm.mean():.1f}%")
    print(f"spread: {g_spread:.1f} pts (global)  ->  {r_spread:.1f} pts (region-relative)")
    print(f"reduction: {g_spread - r_spread:.1f} pts "
          f"({100*(g_spread-r_spread)/max(g_spread,1e-9):.0f}% less biased)")

    print("\nper-region eyeSquintLeft threshold (morphology, not expression):")
    for reg, v in per["eyeSquintLeft"].items():
        print(f"  {reg:<18}{v:.3f}")

    # ── 4. compound yield ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4.  COMPOUND YIELD")
    print("=" * 70)
    det = 0.79  # measured detection yield from the sanity run
    pose = len(g) / len(df)
    neut = 1 - rm.mean()
    print(f"  detection      {100*det:5.1f}%")
    print(f"  pose gate      {100*pose:5.1f}%")
    print(f"  neutrality     {100*neut:5.1f}%")
    print(f"  ------------------------")
    print(f"  end to end     {100*det*pose*neut:5.1f}%")
    per_cell = 6200 * det * pose * neut
    print(f"\n~{per_cell:,.0f} per region-sex cell (from ~6,200 raw)")
    print("floor is 200 —", "OK" if per_cell > 400 else "TOO THIN, loosen further")

    # ── 5. write config ─────────────────────────────────────────────────
    cfg = {
        "pose": {"yaw_max_deg": yaw_lim, "pitch_max_deg": pitch_lim,
                 "roll_max_deg": 25.0, "roll_autocorrect": True},
        # Realness is not calibrated from this dataset (every FairFace image is a
        # real photograph); carried here so re-running does not drop the key.
        "realness": {"min_p_photo": REALNESS_MIN_P_PHOTO},
        "neutrality": {
            "percentiles": PCTL,
            "global": glob,
            "per_region": {s: {k: float(v) for k, v in per[s].items()} for s in per},
            "messages": SIGNALS,
        },
        "provenance": {
            "source": str(SRC.relative_to(REPO)),
            "n_extracted": int(len(df)),
            "n_after_pose": int(len(g)),
            "global_spread_pts": round(float(g_spread), 2),
            "region_spread_pts": round(float(r_spread), 2),
        },
    }
    OUT.write_text(json.dumps(cfg, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())