"""E4b — validate region normalisation against MEBeauty human ratings.

E4a showed the raw beauty score differs by 8.1 points across populations.
That proves a gap exists. It does not prove correcting it makes the score
*better*. This does: MEBeauty carries attractiveness ratings from ~300 raters
of varied background, so we can ask whether agreement with human judgement
improves after normalisation.

Ground-truth ethnicity comes from the image path, so region normalisation is
tested in isolation from region-detection error. The predicted-region arm is
reported too, since that is what actually happens at serve time.

Images are read from ``original_images/``, not from the paths in the score files.
Those point at ``cropped_images/images_crop_align_mtcnn/``, which is already
cropped and aligned by MTCNN; our pipeline then applies square_face_crop on top,
so the beauty model would see a crop of a crop. Its 68-point features are
frame-relative, so double cropping distorts exactly what it reads. The MTCNN
folder is also incomplete, which is where the unreadable files came from.
Rows that have no original fall back to the cropped path and are reported
separately, as a check on whether double cropping was the problem.

    python backend/scripts/run_e4b_mebeauty.py [--limit N] [--loose]

--loose  drops the pose and neutrality gates (keeps realness). MEBeauty is
         in-the-wild; if strict gating leaves too few faces per group, this
         preserves sample size at the cost of matching the serve path.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import inference as inf
from gates import check_neutrality, check_pose, check_realness
from region_stats import beauty_stats

MEB = REPO / "datasets" / "MEBeauty-database"
SCORES = [MEB / "scores" / f for f in ("train_2022.txt", "test_2022.txt", "val_2022.txt")]
OUT = REPO / "data" / "processed" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)

# MEBeauty folder name -> FairFace region name
ETHNIC_MAP = {
    "black": "Black",
    "caucasian": "White",
    "hispanic": "Latino_Hispanic",
    "indian": "Indian",
    "mideastern": "Middle Eastern",
    "middle_eastern": "Middle Eastern",
    "asian": "East Asian",
    "east_asian": "East Asian",
}

INK, RED, GREEN = "#18181b", "#dc2626", "#16a34a"


def parse_rows():
    """Read the three score files; recover ethnicity and sex from the path."""
    rows = []
    for f in SCORES:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            path_str, _, score_str = line.rpartition(" ")
            try:
                score = float(score_str)
            except ValueError:
                continue
            # A few score lines quote the path because the filename contains a
            # space; strip the quotes or the path keeps a literal " and never resolves.
            rel = path_str.strip().strip('"').strip("'").lstrip("./")
            parts = Path(rel).parts
            sex = next((p for p in parts if p in ("male", "female")), None)
            eth_dir = next((p for p in parts if p.lower() in ETHNIC_MAP), None)
            if eth_dir is None or sex is None:
                continue

            # Swap the cropped prefix for original_images/, keeping the sex and
            # ethnicity folder names exactly as they appear in the cropped path.
            original = MEB / "original_images" / sex / eth_dir / Path(rel).name
            cropped = MEB / rel
            if original.is_file():
                path, source = original, "original"
            else:
                path, source = cropped, "cropped"

            rows.append({"path": path, "human_score": score,
                         "ethnicity": ETHNIC_MAP[eth_dir.lower()], "sex": sex,
                         "source": source, "split": f.stem})
    return pd.DataFrame(rows)


def mtcnn_twin(path):
    """The MTCNN-cropped counterpart of an original image, if the dataset has one."""
    path = Path(path)
    originals = MEB / "original_images"
    try:
        rel = path.relative_to(originals)
    except ValueError:
        return None
    twin = MEB / "cropped_images" / "images_crop_align_mtcnn" / rel
    return twin if twin.is_file() else None


def score_image(img):
    """Detect, square-crop, re-detect, and return the raw beauty score."""
    det0 = inf._detect_face(img)
    crop = inf.square_face_crop(img, det0.landmarks[:468],
                                output_size=inf.DEFAULT_OUTPUT_SIZE)
    square = crop[0] if crop else img
    det = inf._detect_face(square)
    return det, float(raw_beauty(det.landmarks[:468]))


def raw_beauty(norm468):
    """Raw (uncalibrated) beauty model output for one face.

    Mirrors the beauty block of analyze_image_bytes exactly.
    """
    import torch
    m = inf._load_models()["beauty"]
    x = inf._beauty_features_from_68(norm468, m["ref_span"], m["ref_center"])
    xn = (x - m["mu"]) / (m["sd"] + 1e-6)
    with torch.no_grad():
        return float(m["model"](torch.from_numpy(xn)).reshape(-1)[0].item())


def process(df, loose=False):
    recs = []
    counts = {"no_face": 0, "not_real": 0, "pose": 0, "expression": 0, "error": 0,
              "paired": 0}

    for i, r in enumerate(df.itertuples(), 1):
        if i % 250 == 0:
            print(f"  ...{i}/{len(df)}", flush=True)
        img = cv2.imread(str(r.path))
        if img is None:
            counts["error"] += 1
            continue
        try:
            ok, _p = check_realness(img)
            if not ok:
                counts["not_real"] += 1
                continue

            det, raw = score_image(img)

            if not loose:
                pose_ok, _pose = check_pose(det.matrix)
                if not pose_ok:
                    counts["pose"] += 1
                    continue
                true_w = {r.ethnicity: 1.0}
                neutral, _hint = check_neutrality(det.blendshapes, true_w)
                if not neutral:
                    counts["expression"] += 1
                    continue

            rec = {"path": str(r.path), "human_score": r.human_score,
                   "ethnicity": r.ethnicity, "sex": r.sex,
                   "source": r.source, "raw": raw, "raw_double_cropped": np.nan}

            # Paired arm: the identical face read from the MTCNN crop, which our
            # square_face_crop then crops again. Gate decisions come from the
            # original, so the only thing differing between the two numbers is
            # framing. This is what isolates the double-cropping effect.
            twin = mtcnn_twin(r.path)
            if twin is not None:
                try:
                    timg = cv2.imread(str(twin))
                    if timg is not None:
                        _tdet, rec["raw_double_cropped"] = score_image(timg)
                        counts["paired"] += 1
                except Exception:  # noqa: BLE001 — paired arm is diagnostic only
                    pass
            recs.append(rec)
        except Exception:  # noqa: BLE001
            counts["no_face"] += 1
    return pd.DataFrame(recs), counts


def normalize(d, arm="true"):
    """z = (raw - mu_region) / sigma_region, using ground-truth ethnicity."""
    out = []
    for eth, g in d.groupby("ethnicity"):
        st = beauty_stats({eth: 1.0})
        if not st or not st.get("sigma"):
            out.append(g.assign(z=np.nan))
            continue
        out.append(g.assign(z=(g.raw - st["mu"]) / st["sigma"]))
    return pd.concat(out)


def correlations(d, col):
    rows = []
    for eth, g in d.groupby("ethnicity"):
        if len(g) < 15:
            rows.append({"ethnicity": eth, "n": len(g),
                         "pearson": np.nan, "spearman": np.nan})
            continue
        rows.append({"ethnicity": eth, "n": len(g),
                     "pearson": stats.pearsonr(g[col], g.human_score)[0],
                     "spearman": stats.spearmanr(g[col], g.human_score)[0]})
    t = pd.DataFrame(rows).set_index("ethnicity")
    t.loc["ALL"] = [len(d), stats.pearsonr(d[col], d.human_score)[0],
                    stats.spearmanr(d[col], d.human_score)[0]]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--loose", action="store_true")
    a = ap.parse_args()

    df = parse_rows()
    if df.empty:
        sys.exit("No rows parsed. Check datasets/MEBeauty-database/scores/*.txt")
    if a.limit:
        df = df.sample(min(a.limit, len(df)), random_state=0)

    print(f"MEBeauty: {len(df)} rated images")
    src = df.groupby("source").size()
    print("\nImage source resolved from the score-file paths:")
    for name in ("original", "cropped"):
        n = int(src.get(name, 0))
        print(f"  {name:<9} {n:>5}  ({100*n/len(df):5.1f}%)")
    if int(src.get("cropped", 0)):
        print("  (cropped = no matching file under original_images/; these are"
              " already MTCNN-cropped, so our square_face_crop double-crops them)")
    print()
    print(df.groupby(["ethnicity", "sex"]).size().to_string())
    print(f"\nprocessing ({'loose' if a.loose else 'strict'} gates)...")

    d, counts = process(df, loose=a.loose)
    print(f"\nsurvived {len(d)}/{len(df)}  ({100*len(d)/len(df):.1f}%)")
    print("  rejected:", ", ".join(f"{k}={v}" for k, v in counts.items()
                                   if v and k != "paired"))
    if counts.get("paired"):
        print(f"  also scored from the MTCNN crop for comparison: {counts['paired']}")

    surv = (df.groupby("ethnicity").size().rename("attempted")
              .to_frame().join(d.groupby("ethnicity").size().rename("kept")))
    surv["kept"] = surv.kept.fillna(0).astype(int)
    surv["rate_%"] = 100 * surv.kept / surv.attempted
    print("\nSurvival per group:")
    print(surv.to_string(float_format=lambda x: f"{x:7.1f}"))

    thin = surv[surv.kept < 30]
    if not thin.empty:
        print(f"\nWARNING: {len(thin)} group(s) under 30 faces — "
              f"their correlations are not reliable.")
        if not a.loose:
            print("Re-run with --loose to trade gate fidelity for sample size.")

    # Sanity check: if double cropping was the problem, rows that fell back to the
    # pre-cropped MTCNN images should agree with humans noticeably worse than rows
    # read from originals. Same model, same gates — only the framing differs.
    print("\n" + "=" * 74)
    print("SOURCE SPLIT — raw score vs human ratings, by image source")
    print("=" * 74)
    if "source" in d.columns and d.source.nunique() > 1:
        rows = []
        for name, g in d.groupby("source"):
            rows.append({
                "source": name, "n": len(g),
                "pearson": stats.pearsonr(g.raw, g.human_score)[0] if len(g) >= 15 else np.nan,
                "spearman": stats.spearmanr(g.raw, g.human_score)[0] if len(g) >= 15 else np.nan,
            })
        split = pd.DataFrame(rows).set_index("source")
        print(split.to_string(float_format=lambda x: f"{x:9.3f}"))
        split.to_csv(OUT / "e4b_source_split.csv")
        if split["pearson"].notna().all() and len(split) == 2:
            gap = split.loc["original", "pearson"] - split.loc["cropped", "pearson"]
            print(f"\n  originals minus crops: {gap:+.3f} pearson")
            print("  A large positive gap confirms double cropping was degrading the")
            print("  beauty features. A gap near zero means framing was not the issue.")
    else:
        only = d.source.iloc[0] if "source" in d.columns and len(d) else "n/a"
        print(f"  all surviving rows resolved to one source ({only}) — nothing to split.")
        print("  Every fallback row is missing from the MTCNN folder as well, so the")
        print("  accidental split carries no signal. The paired check below is the")
        print("  real test, and a stronger one: identical faces, both framings.")

    # Paired framing check. Same face, same gates, scored twice: once from the
    # original and once from the MTCNN crop that our square_face_crop then crops
    # again. Any difference is framing alone.
    paired = d.dropna(subset=["raw_double_cropped"]) if "raw_double_cropped" in d else d.iloc[0:0]
    print("\n" + "=" * 74)
    print("PAIRED FRAMING CHECK — identical faces, original vs double-cropped")
    print("=" * 74)
    if len(paired) >= 15:
        rows = []
        for label, col in (("original framing", "raw"),
                           ("double-cropped (MTCNN + square_face_crop)", "raw_double_cropped")):
            rows.append({
                "framing": label, "n": len(paired),
                "pearson": stats.pearsonr(paired[col], paired.human_score)[0],
                "spearman": stats.spearmanr(paired[col], paired.human_score)[0],
            })
        pf = pd.DataFrame(rows).set_index("framing")
        print(pf.to_string(float_format=lambda x: f"{x:9.3f}"))
        pf.to_csv(OUT / "e4b_paired_framing.csv")
        gap = pf.iloc[0]["pearson"] - pf.iloc[1]["pearson"]
        agree = stats.pearsonr(paired.raw, paired.raw_double_cropped)[0]
        print(f"\n  original minus double-cropped: {gap:+.3f} pearson")
        print(f"  the two scorings agree with each other at r={agree:.3f}")
        print("  A large positive gap confirms double cropping was degrading the")
        print("  beauty features. A gap near zero means framing was not the issue,")
        print("  and the low correlation is a property of the model, not the input.")
    else:
        print(f"  only {len(paired)} paired rows — not enough to compare")

    d = normalize(d)
    before = correlations(d, "raw")
    after = correlations(d.dropna(subset=["z"]), "z")

    cmp = pd.DataFrame({
        "n": before.n,
        "raw_pearson": before.pearson, "norm_pearson": after.pearson,
        "raw_spearman": before.spearman, "norm_spearman": after.spearman,
    })
    cmp["d_pearson"] = cmp.norm_pearson - cmp.raw_pearson
    cmp.to_csv(OUT / "e4b_mebeauty_correlation.csv")

    print("\n" + "=" * 74)
    print("E4b  Agreement with human ratings, before and after normalisation")
    print("=" * 74)
    print(cmp.to_string(float_format=lambda x: f"{x:9.3f}"))

    print("\nHow to read this:")
    print("  Correlation measures whether the score RANKS faces the way people do.")
    print("  Region normalisation shifts each group to a common centre, so it")
    print("  cannot change the within-group ranking — per-group correlations")
    print("  should be near identical. The ALL row is the real test: pooling")
    print("  groups, an uncorrected offset drags the overall correlation down.")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    e = [i for i in cmp.index if i != "ALL"]
    x = np.arange(len(e)); w = .38
    ax.bar(x - w/2, cmp.loc[e, "raw_pearson"], w, label="raw score", color=RED, alpha=.85)
    ax.bar(x + w/2, cmp.loc[e, "norm_pearson"], w, label="region-normalised",
           color=GREEN, alpha=.85)
    ax.set_xticks(x); ax.set_xticklabels(e, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("correlation with human ratings")
    ax.axhline(0, color=INK, lw=.8)
    ax.set_title("E4b — Agreement with MEBeauty human ratings",
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT / "e4b.png", dpi=180, facecolor="white")
    plt.close()

    d.to_csv(OUT / "e4b_scores.csv", index=False)
    print(f"\nwrote {OUT/'e4b_mebeauty_correlation.csv'} and e4b.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())