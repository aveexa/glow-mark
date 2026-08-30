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

# Signals removed from the gate, with the measurement that removed them.
#
# Recorded here rather than recomputed: this script only sees FairFace, which has no
# relaxed-vs-active labels. The figures come from
# /tmp scratch analysis over datasets/devset (PASS set n=38 = relaxed closed mouths
# by construction, FAIL smiling/laughing n=6 = genuinely active mouths), reading the
# same frame the gate judges — pass 2, after roll autocorrect.
#
# The test each one failed: can a relaxed mouth read higher than an active one? For
# all three the answer is yes, so the signal is reading mouth morphology rather than
# mouth movement and no threshold on it can separate the two. This is the third such
# finding, after the brow-raise signals and the CLIP realness prompts — a gate signal
# is worth measuring for separation before it is worth calibrating.
DROPPED_SIGNALS = {
    "mouthPucker": {
        "auc_active_vs_relaxed": 0.254,
        "relaxed_max": 0.915,
        "active_max": 0.014,
        "why": (
            "Inverted, not mistuned. AUC 0.254 is below chance: relaxed mouths score "
            "HIGHER than active ones, and 71% of relaxed images exceed the active "
            "group's median. A calm closed mouth reading 0.915 pucker is measuring "
            "lip shape, not pursing. No threshold on an inverted signal works."
        ),
    },
    "mouthPressLeft": {
        "auc_active_vs_relaxed": 0.632,
        "relaxed_max": 0.434,
        "active_max": 0.165,
        "why": (
            "The relaxed maximum exceeds every active reading in the sample, so a calm "
            "mouth outscores a genuinely active one. Weak AUC and no usable tail."
        ),
    },
    "mouthPressRight": {
        "auc_active_vs_relaxed": 0.798,
        "relaxed_max": 0.408,
        "active_max": 0.208,
        "why": (
            "Same tail failure as the left: relaxed max 0.408 above active max 0.208. "
            "The higher AUC is driven by the middle of the distribution, which a gate "
            "never uses — a gate operates at the tail. A left/right asymmetry on a "
            "paired anatomical signal is itself grounds to distrust both."
        ),
    },
}

# Kept: mouthFunnel is the only mouth signal that separates in the right direction
# (AUC 0.961, relaxed max 0.092 below active max 0.095). jawOpen and mouthSmile*
# cover the mouth movements that actually displace the measured features.

# Signals whose cut comes from a separating gap rather than a percentile.
#
# A percentile of the reference population is the wrong estimator when that
# population contains the thing being gated against. FairFace is ~25% smiling
# (mouthSmileLeft p75 = 0.704), so its p97 lands at 0.95 — inside the range where
# real smiles live (0.823-0.985). That defines "neutral" as "smiling less than the
# top 3% of smilers", and it let a broad open-teeth smile through.
#
# Where labelled active and relaxed dev-set images separate with nothing in between,
# the cut goes in the middle of that gap instead. Measured on the roll-corrected
# frame — the one the gate judges — with relaxed = PASS set (n=38) and active = the
# expression category each signal is meant to catch.
#
# PROVISIONAL: the active sets are 3-6 images. The gaps are wide relative to their
# own scale and no relaxed face falls inside any of them, but these cuts should be
# revisited against a larger labelled set before being treated as settled.
GAP_CUTS = {
    "jawOpen":         {"cut": 0.2568, "gap": 0.471, "relaxed_max": 0.021, "active_min": 0.492,
                        "active_n": 3, "auc": 1.000, "was_percentile": 0.270},
    "mouthSmileLeft":  {"cut": 0.7913, "gap": 0.064, "relaxed_max": 0.759, "active_min": 0.823,
                        "active_n": 6, "auc": 1.000, "was_percentile": 0.952},
    "mouthSmileRight": {"cut": 0.8109, "gap": 0.100, "relaxed_max": 0.761, "active_min": 0.861,
                        "active_n": 6, "auc": 1.000, "was_percentile": 0.955},
    "eyeBlinkLeft":    {"cut": 0.4798, "gap": 0.401, "relaxed_max": 0.279, "active_min": 0.681,
                        "active_n": 3, "auc": 1.000, "was_percentile": 0.642},
    "eyeBlinkRight":   {"cut": 0.3904, "gap": 0.359, "relaxed_max": 0.211, "active_min": 0.570,
                        "active_n": 3, "auc": 1.000, "was_percentile": 0.620},
}

# Signals that keep a percentile cut because they do NOT separate: a relaxed face can
# read higher than a genuinely active one, so no single cut divides the two. Recorded
# rather than dropped — a non-separating signal still contributes nothing reliable,
# and this is the evidence for revisiting it.
NON_SEPARATING = {
    "mouthFunnel":      {"auc": 0.961, "relaxed_max": 0.092, "active_min": 0.029},
    "browInnerUp":      {"auc": 0.667, "relaxed_max": 0.482, "active_min": 0.002},
    "browDownLeft":     {"auc": 0.895, "relaxed_max": 0.724, "active_min": 0.206},
    "browDownRight":    {"auc": 0.939, "relaxed_max": 0.747, "active_min": 0.187},
    "browOuterUpLeft":  {"auc": 0.965, "relaxed_max": 0.714, "active_min": 0.469},
    "browOuterUpRight": {"auc": 0.456, "relaxed_max": 0.504, "active_min": 0.007},
    "eyeSquintLeft":    {"auc": 0.798, "relaxed_max": 0.695, "active_min": 0.339},
    "eyeSquintRight":   {"auc": 0.667, "relaxed_max": 0.690, "active_min": 0.083},
}

# Considered and rejected as a new gated signal. The lopsided mouth pull it registers
# is real expression, but it does not separate: a relaxed face reaches 0.567 while the
# least-active expression image sits at 0.014.
CANDIDATE_REJECTED = {
    "mouthStretchRight": {"auc": 0.895, "relaxed_max": 0.567, "active_min": 0.014},
}

# per-signal percentile: how much of the reference population each cut admits.
# jawOpen is stricter because an open mouth genuinely wrecks the jaw/lip features;
# the rest are looser because they mostly track morphology, not expression.
PCTL = {s: 0.98 for s in SIGNALS}
PCTL["jawOpen"] = 0.96
PCTL["mouthSmileLeft"] = 0.97
PCTL["mouthSmileRight"] = 0.97


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
    # A gap cut is fixed, so it enters both arms unchanged.
    for s, spec in GAP_CUTS.items():
        if s in glob:
            glob[s] = float(spec["cut"])
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
    for s, spec in GAP_CUTS.items():
        if s in per:
            per[s] = pd.Series(float(spec["cut"]), index=per[s].index)
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
    # One file, one owner per block. This script owns `pose` and `neutrality` only.
    # `realness` is owned by backend/scripts/calibrate_realness.py — it cannot be
    # calibrated from FairFace, which has no negative class — so it is read through
    # untouched rather than mirrored here. Same in reverse: the realness script
    # leaves pose and neutrality alone.
    cfg = json.loads(OUT.read_text()) if OUT.is_file() else {}
    preserved = sorted(k for k in cfg if k not in ("pose", "neutrality", "provenance"))

    cfg["pose"] = {"yaw_max_deg": yaw_lim, "pitch_max_deg": pitch_lim,
                   "roll_max_deg": 25.0, "roll_autocorrect": True}
    cfg["neutrality"] = {
        "percentiles": PCTL,
        "dropped_signals": DROPPED_SIGNALS,
        "gap_cuts": {s: float(v["cut"]) for s, v in GAP_CUTS.items()},
        "gap_cut_evidence": GAP_CUTS,
        "non_separating": NON_SEPARATING,
        "candidate_rejected": CANDIDATE_REJECTED,
        "global": glob,
        "per_region": {s: {k: float(v) for k, v in per[s].items()} for s in per},
        "messages": SIGNALS,
    }
    # Provenance is shared: update only our keys, leave other owners' entries alone.
    cfg.setdefault("provenance", {}).update({
        "source": str(SRC.relative_to(REPO)),
        "n_extracted": int(len(df)),
        "n_after_pose": int(len(g)),
        "global_spread_pts": round(float(g_spread), 2),
        "region_spread_pts": round(float(r_spread), 2),
    })
    OUT.write_text(json.dumps(cfg, indent=2))
    print(f"\nwrote {OUT}")
    print(f"  updated: pose, neutrality")
    print(f"  preserved (owned elsewhere): {', '.join(preserved) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())