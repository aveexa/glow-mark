"""Run experiments E1, E2, E3, E4a and E6 from artifacts already on disk.

    python backend/scripts/run_experiments.py

Reads:
    data/interim/diagnose_gates.csv           features + blendshapes + race/gender/pose
    data/processed/region_reference_stats.csv the 576-row reference table

Writes:
    data/processed/experiments/*.csv          one table per experiment
    data/processed/experiments/*.png          figures

E4b (validation against MEBeauty human ratings) needs that dataset and is not
covered here. E5 and E7 come from the gate calibration scripts.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SRC = REPO / "data" / "interim" / "diagnose_gates.csv"
REF = REPO / "data" / "processed" / "region_reference_stats.csv"
OUT = REPO / "data" / "processed" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

EXCLUDE = {"beauty_score_raw", "chin_length_ratio"}
GATE_BS = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker",
           "mouthPressLeft", "mouthPressRight", "mouthFunnel",
           "browInnerUp", "browDownLeft", "browDownRight",
           "browOuterUpLeft", "browOuterUpRight",
           "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight"]

INK, BLUE, RED, GREEN = "#18181b", "#2563eb", "#dc2626", "#16a34a"


def head(n, t):
    print("\n" + "=" * 72)
    print(f"{n}  {t}")
    print("=" * 72)


def load():
    df = pd.read_csv(SRC)
    ref = pd.read_csv(REF)
    feats = sorted(set(ref.feature.unique()) - EXCLUDE)
    feats = [f for f in feats if f in df.columns]
    return df, ref, feats


# ───────────────────────────────────────────────────────── E1
def e1(df, feats):
    head("E1", "Does expression distort the measurements?")
    bs = [c for c in GATE_BS if c in df.columns]
    energy = np.sqrt((df[bs] ** 2).sum(axis=1) / len(bs))
    df = df.assign(expression_energy=energy)

    rows = []
    for f in feats:
        r, p = stats.pearsonr(df.expression_energy, df[f])
        rows.append({"feature": f, "r": r, "abs_r": abs(r), "p_value": p})
    t = pd.DataFrame(rows).sort_values("abs_r", ascending=False)
    t.to_csv(OUT / "e1_expression_correlation.csv", index=False)

    print(f"{'feature':<26}{'r':>8}{'p':>12}")
    for _, x in t.iterrows():
        flag = "  <-- strong" if x.abs_r > 0.20 else ""
        print(f"{x.feature:<26}{x.r:>8.3f}{x.p_value:>12.2e}{flag}")
    n_strong = (t.abs_r > 0.20).sum()
    print(f"\n{n_strong} of {len(t)} features correlate |r|>0.20 with expression energy.")

    fig, ax = plt.subplots(figsize=(8, 6))
    tt = t.iloc[::-1]
    ax.barh(range(len(tt)), tt.r,
            color=[RED if abs(v) > .20 else "#cbd5e1" for v in tt.r])
    ax.set_yticks(range(len(tt))); ax.set_yticklabels(tt.feature, fontsize=8.5)
    ax.axvline(0, color=INK, lw=.8)
    for v in (-.2, .2):
        ax.axvline(v, color=RED, ls=":", lw=1)
    ax.set_xlabel("Pearson r vs expression energy")
    ax.set_title("E1 — Expression contaminates the measurements",
                 fontweight="bold", loc="left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT / "e1.png", dpi=180, facecolor="white"); plt.close()
    return t


# ───────────────────────────────────────────────────────── E2
def e2(df, feats):
    head("E2", "Do the measurements vary by region?")
    rows = []
    for f in feats:
        groups = [g[f].dropna().values for _, g in df.groupby("race") if len(g) > 20]
        if len(groups) < 2:
            continue
        F, p = stats.f_oneway(*groups)
        allv = np.concatenate(groups)
        ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups)
        eta2 = ss_b / ((allv - allv.mean()) ** 2).sum()
        med = df.groupby("race")[f].median()
        rows.append({"feature": f, "eta_squared": eta2, "F": F, "p_value": p,
                     "spread": med.max() - med.min(),
                     "highest": med.idxmax(), "lowest": med.idxmin()})
    t = pd.DataFrame(rows).sort_values("eta_squared", ascending=False)
    t.to_csv(OUT / "e2_region_effect_sizes.csv", index=False)

    print(f"{'feature':<26}{'eta2':>8}{'p':>11}   highest vs lowest")
    for _, x in t.iterrows():
        sz = "large" if x.eta_squared > .14 else "medium" if x.eta_squared > .06 else \
             "small" if x.eta_squared > .01 else "-"
        print(f"{x.feature:<26}{x.eta_squared:>8.3f}{x.p_value:>11.1e}   "
              f"{x.highest} vs {x.lowest}  ({sz})")
    print(f"\n{(t.eta_squared > .06).sum()} of {len(t)} features show a medium or "
          f"larger region effect. These are the ones region conditioning matters for.")
    return t


# ───────────────────────────────────────────────────────── E3
def e3(df, ref, feats):
    head("E3", "Does region conditioning change the classifications?  [HEADLINE]")
    g = ref[(ref.region == "global") & (ref.gender == "pooled")].set_index("feature")
    per = {r: ref[(ref.region == r) & (ref.gender == "pooled")].set_index("feature")
           for r in ref.region.unique() if r != "global"}

    def cls(v, lo, hi):
        return "low" if v < lo else "high" if v > hi else "ok"

    recs = []
    for _, row in df.iterrows():
        reg = row["race"]
        if reg not in per:
            continue
        for f in feats:
            if f not in g.index or f not in per[reg].index:
                continue
            v = row[f]
            a = cls(v, g.loc[f, "p20"], g.loc[f, "p80"])
            b = cls(v, per[reg].loc[f, "p20"], per[reg].loc[f, "p80"])
            recs.append({"race": reg, "feature": f, "global": a, "region": b,
                         "changed": a != b})
    d = pd.DataFrame(recs)
    d.to_csv(OUT / "e3_classification_changes.csv", index=False)

    by_r = d.groupby("race")["changed"].agg(n="size", changed="sum")
    by_r["pct"] = 100 * by_r.changed / by_r.n
    by_f = d.groupby("feature")["changed"].mean().mul(100).sort_values(ascending=False)

    print("By region:")
    print(by_r.to_string(float_format=lambda x: f"{x:8.1f}"))
    print(f"\nOverall: {100*d.changed.mean():.1f}% of classifications change.")
    print("\nTop 8 features by change rate:")
    for f, v in by_f.head(8).items():
        print(f"  {f:<28}{v:5.1f}%")

    fig, ax = plt.subplots(figsize=(8.6, 4))
    s = by_r.sort_values("pct")
    ax.barh(range(len(s)), s.pct, color=BLUE)
    for i, v in enumerate(s.pct):
        ax.text(v + .4, i, f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=10)
    ax.set_xlabel("classifications that change (%)")
    ax.set_title("E3 — Effect of region conditioning, by population",
                 fontweight="bold", loc="left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT / "e3.png", dpi=180, facecolor="white"); plt.close()
    return d


# ───────────────────────────────────────────────────────── E4a
def e4a(ref):
    head("E4a", "Is the beauty score biased by region?")
    b = ref[(ref.feature == "beauty_score_raw") & (ref.gender == "pooled")]
    b = b[b.region != "global"].set_index("region")[["mu", "sigma", "n"]]
    b = b.sort_values("mu", ascending=False)
    b["z_vs_global"] = (b.mu - b.mu.mean()) / b.mu.std()
    b.to_csv(OUT / "e4a_beauty_bias.csv")

    print(b.to_string(float_format=lambda x: f"{x:9.3f}"))
    spread = b.mu.max() - b.mu.min()
    print(f"\nSpread in mean raw score: {spread:.2f} points "
          f"({b.mu.idxmax()} highest, {b.mu.idxmin()} lowest)")
    print("After region z-normalisation every group centres on 0 by construction,")
    print("so this spread is exactly what the normalisation removes.")

    fig, ax = plt.subplots(figsize=(8.6, 4))
    ax.bar(range(len(b)), b.mu, yerr=b.sigma, color=RED, alpha=.85,
           capsize=4, ecolor="#94a3b8")
    ax.axhline(b.mu.mean(), color=INK, ls="--", lw=1.2, label="mean across regions")
    ax.set_xticks(range(len(b)))
    ax.set_xticklabels(b.index, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("mean raw beauty score")
    ax.set_title("E4a — Raw beauty score before normalisation",
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT / "e4a.png", dpi=180, facecolor="white"); plt.close()
    return b


# ───────────────────────────────────────────────────────── E6
def e6(ref, feats):
    head("E6", "Which measurements differ by sex?  (cost of pooling)")
    rows = []
    for f in feats:
        m = ref[(ref.feature == f) & (ref.gender == "male") & (ref.region != "global")]
        w = ref[(ref.feature == f) & (ref.gender == "female") & (ref.region != "global")]
        if m.empty or w.empty:
            continue
        mu_m, mu_f = m.mu.mean(), w.mu.mean()
        sd = np.sqrt((m.sigma.pow(2).mean() + w.sigma.pow(2).mean()) / 2)
        d = (mu_m - mu_f) / sd if sd > 0 else 0.0
        rows.append({"feature": f, "cohens_d": d, "abs_d": abs(d),
                     "mu_male": mu_m, "mu_female": mu_f})
    t = pd.DataFrame(rows).sort_values("abs_d", ascending=False)
    t.to_csv(OUT / "e6_sex_differences.csv", index=False)

    print(f"{'feature':<26}{'d':>8}   size")
    for _, x in t.iterrows():
        sz = "large" if x.abs_d > .8 else "medium" if x.abs_d > .5 else \
             "small" if x.abs_d > .2 else "-"
        print(f"{x.feature:<26}{x.cohens_d:>8.3f}   {sz}")
    print(f"\n{(t.abs_d > .5).sum()} of {len(t)} features show a medium or larger sex "
          f"difference.\nThese are what pooling averages away. We ship pooled by "
          f"choice; this is the measured cost.")
    return t


def main():
    df, ref, feats = load()
    print(f"loaded {len(df)} faces, {len(ref)} reference rows, {len(feats)} features")
    e1(df, feats)
    e2(df, feats)
    e3(df, ref, feats)
    e4a(ref)
    e6(ref, feats)
    print("\n" + "=" * 72)
    print(f"tables and figures written to {OUT}")
    print("Still outstanding: E4b (needs MEBeauty human ratings), E5 (dev set).")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())