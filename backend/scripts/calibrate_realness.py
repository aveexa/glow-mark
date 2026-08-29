"""Calibrate realness.min_p_photo from measured distributions.

The realness gate exists to reject non-photographic input that the landmark
detector will happily mesh. So the population that matters is not "all images" but
"images that reach the gate" — i.e. those where a face was detected and the pose
gate passed. Everything else is already rejected upstream by NO_FACE_DETECTED.

Reports p_photo percentiles for the positive class (FairFace val) and the negative
class (cartoons, anime, 3D renders, paintings, statues, dogs, cats, primates), then
picks a threshold in the gap if the two separate.

Writes the measured distributions into gate_config.json under
provenance.realness_calibration. With --set it also writes realness.min_p_photo,
choosing the highest cut that keeps false rejection of real photographs under
TARGET_FALSE_REJECT.

The classes do not separate cleanly, and that is a product decision rather than a
bug to tune away: photorealistic 3D-rendered faces score inside the real-photograph
distribution and pass the gate. That leak is accepted and documented; a 9%
false-reject rate on real users was not.

Run from:  repo root
    python backend/scripts/calibrate_realness.py <negatives_dir> [n_positive] [--set]
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import csv
import json

import cv2
import numpy as np

from face_normalize import DEFAULT_OUTPUT_SIZE, square_face_crop
from gates import GATE_CONFIG_PATH, check_pose, check_realness, load_gate_config
from inference import _detect_face, AnalyzeError

VAL = REPO / "datasets" / "FairFace" / "val"
TARGET_FALSE_REJECT = 0.02  # brief: under 2% false-reject on real photographs
PCTS_POS = (1, 5, 10, 25, 50)
PCTS_NEG = (50, 75, 90, 99, 100)

# Only the render category is split by source, and only because the whole finding
# rests on it: a leak that held for one generator would be a property of that
# generator rather than of the gate. Filename prefix -> generator.
RENDER_CATEGORY = "render3d"
RENDER_SOURCES = {"m": "unreal_metahuman"}          # prefix -> name
RENDER_DEFAULT_SOURCE = "ms_facesynthetics"          # bare numeric filenames

# Per-image scores are cached so the held-out split can be re-derived, and a
# different split re-evaluated, without re-running detection and CLIP over
# thousands of images.
SCORE_CACHE = REPO / "data" / "interim" / "realness_scores.csv"
HOLDOUT_FRACTION = 0.30
HOLDOUT_SEED = 20260829


def reaches_gate(img) -> bool:
    """True when a single face is detected and the pose gate passes — i.e. the gate runs."""
    try:
        det = _detect_face(img)
        cropped = square_face_crop(img, det.landmarks[:468], output_size=DEFAULT_OUTPUT_SIZE)
        if cropped is None:
            return False
        return check_pose(_detect_face(cropped[0]).matrix)[0]
    except (AnalyzeError, Exception):  # noqa: BLE001
        return False


_CACHE: dict[str, tuple[float, bool]] = {}
_MEASURED: list[tuple[str, float, bool]] = []


def load_cache() -> None:
    """Populate the in-memory score cache from a previous run, if present."""
    if not SCORE_CACHE.is_file():
        return
    with SCORE_CACHE.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            _CACHE[row["path"]] = (float(row["p_photo"]), row["reaches_gate"] == "1")
    print(f"  loaded {len(_CACHE)} cached scores from {SCORE_CACHE.relative_to(REPO)}")


def measure(paths):
    """Return (p_photo for all, p_photo for those that reach the gate)."""
    every, gated = [], []
    for p in paths:
        key = str(p)
        if key in _CACHE:
            p_photo, gate = _CACHE[key]
        else:
            img = cv2.imread(key)
            if img is None:
                continue
            _, p_photo = check_realness(img)
            gate = reaches_gate(img)
            _CACHE[key] = (p_photo, gate)
        _MEASURED.append((key, p_photo, gate))
        every.append(p_photo)
        if gate:
            gated.append(p_photo)
    return np.array(every), np.array(gated)


def write_cache() -> None:
    """Persist every score measured this run, so the split can be revisited."""
    SCORE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with SCORE_CACHE.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "p_photo", "reaches_gate"])
        for path, p_photo, gate in sorted(set(_MEASURED)):
            w.writerow([path, f"{p_photo:.6f}", int(gate)])


def holdout_check(pos_gated: np.ndarray) -> dict:
    """Refit the cut on 70% of gated positives and score it on the untouched 30%.

    The headline rate is true by construction — the threshold *is* that percentile of
    the sample it was fitted on. This is the only figure that says anything about
    unseen photographs.
    """
    rng = np.random.default_rng(HOLDOUT_SEED)
    idx = rng.permutation(pos_gated.size)
    n_hold = int(round(pos_gated.size * HOLDOUT_FRACTION))
    hold, fit = pos_gated[idx[:n_hold]], pos_gated[idx[n_hold:]]
    refit = float(np.percentile(fit, 100 * TARGET_FALSE_REJECT))
    return {
        "seed": HOLDOUT_SEED,
        "fit_n": int(fit.size),
        "holdout_n": int(hold.size),
        "threshold_fitted_on_fit_split": round(refit, 4),
        "false_reject_fitted": round(float((fit < refit).mean()), 4),
        "false_reject_holdout": round(float((hold < refit).mean()), 4),
    }


def line(label, arr, pcts):
    if arr.size == 0:
        return f"  {label:<26} (empty)"
    q = [np.percentile(arr, p) for p in pcts]
    return f"  {label:<26} n={arr.size:<5}" + "".join(f"{v:>9.3f}" for v in q)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write_threshold = "--set" in sys.argv
    neg_dir = Path(args[0])
    n_pos = int(args[1]) if len(args) > 1 else 600

    load_cache()
    print("measuring positives (FairFace val)...", flush=True)
    pos_paths = sorted(x for x in VAL.iterdir() if x.suffix.lower() == ".jpg")[:n_pos]
    pos_all, pos_gated = measure(pos_paths)

    print("measuring negatives...", flush=True)
    cats = sorted(d for d in neg_dir.iterdir() if d.is_dir())
    neg_by_cat = {}
    for d in cats:
        files = sorted(d.iterdir())
        neg_by_cat[d.name] = measure(files)
        # Where one category was drawn from two independent generators, split it:
        # a finding that only holds for one generator is not a finding.
        if d.name == RENDER_CATEGORY:
            groups = {}
            for f in files:
                groups.setdefault(
                    RENDER_SOURCES.get(f.stem[0], RENDER_DEFAULT_SOURCE), []
                ).append(f)
            if len(groups) > 1:
                for tag, sub in sorted(groups.items()):
                    neg_by_cat[f"{d.name}:{tag}"] = measure(sub)
    base_cats = [d.name for d in cats]
    neg_all = np.concatenate([neg_by_cat[k][0] for k in base_cats])
    neg_gated = np.concatenate([neg_by_cat[k][1] for k in base_cats if neg_by_cat[k][1].size])

    print("\n" + "=" * 78)
    print("POSITIVE CLASS — real photographs")
    print("=" * 78)
    print(f"  {'':<26}{'':<7}" + "".join(f"{'p'+str(p):>9}" for p in PCTS_POS))
    print(line("all images", pos_all, PCTS_POS))
    print(line("reaching the gate", pos_gated, PCTS_POS))

    print("\n" + "=" * 78)
    print("NEGATIVE CLASS — non-photographic and animal faces")
    print("=" * 78)
    print(f"  {'':<26}{'':<7}" + "".join(f"{'p'+str(p) if p < 100 else 'max':>9}" for p in PCTS_NEG))
    for name, (a, g) in neg_by_cat.items():
        print(line(name, a, PCTS_NEG))
    print(line("ALL negatives", neg_all, PCTS_NEG))
    print(line("negatives reaching gate", neg_gated, PCTS_NEG))

    # Only images the detector meshes ever reach the gate; the rest are already
    # NO_FACE_DETECTED. This column is what decides whether the gate can separate.
    print("\n  how many of each category actually reach the gate:")
    for name, (a, g) in neg_by_cat.items():
        share = 100 * g.size / a.size if a.size else 0.0
        detail = f"  p_photo median {np.median(g):.3f}  max {g.max():.3f}" if g.size else ""
        print(f"    {name:<10} {g.size:>3}/{a.size:<3} ({share:5.1f}%){detail}")

    other = [k for k in base_cats if k != RENDER_CATEGORY]
    other_all = np.concatenate([neg_by_cat[k][0] for k in other])
    other_gated = [neg_by_cat[k][1] for k in other if neg_by_cat[k][1].size]
    other_gated = np.concatenate(other_gated) if other_gated else np.array([])
    print()
    print(line("ALL non-render negatives", other_all, PCTS_NEG))
    print(line("  ... reaching the gate", other_gated, PCTS_NEG))

    print("\n" + "=" * 78)
    print("SEPARATION — the gate only ever sees images that reach it")
    print("=" * 78)
    cur = float(load_gate_config()["realness"]["min_p_photo"])
    print(f"  positives reaching gate : n={pos_gated.size}  min={pos_gated.min():.3f}")
    print(f"  negatives reaching gate : n={neg_gated.size}  max={neg_gated.max():.3f}")
    print(f"  current threshold {cur:.2f}: false-reject "
          f"{100*(pos_gated < cur).mean():.1f}%  |  leak {100*(neg_gated >= cur).mean():.1f}%")

    # Highest threshold meeting the false-reject target, then how much leaks in.
    target = float(np.percentile(pos_gated, 100 * TARGET_FALSE_REJECT))
    print(f"\n  threshold for {100*TARGET_FALSE_REJECT:.0f}% false-reject on gated positives: {target:.3f}")

    print(f"\n  leak at {target:.3f}, by group (of those reaching the gate):")
    for k in sorted(neg_by_cat):
        g = neg_by_cat[k][1]
        if g.size:
            print(f"    {k:<28} {int((g >= target).sum()):>3}/{g.size:<3} "
                  f"({100*(g >= target).mean():5.1f}%)")

    hc = holdout_check(pos_gated)
    print("\n" + "=" * 78)
    print("HELD-OUT CHECK — the fitted rate is true by construction; this is not")
    print("=" * 78)
    print(f"  split {100*(1-HOLDOUT_FRACTION):.0f}/{100*HOLDOUT_FRACTION:.0f} of gated positives, seed {HOLDOUT_SEED}")
    print(f"  refit on {hc['fit_n']} -> threshold {hc['threshold_fitted_on_fit_split']:.4f}")
    print(f"  false-reject, fitted split  ({hc['fit_n']:>4}): {100*hc['false_reject_fitted']:.2f}%")
    print(f"  false-reject, held-out      ({hc['holdout_n']:>4}): {100*hc['false_reject_holdout']:.2f}%")

    overlap = neg_gated[neg_gated >= target]
    if overlap.size == 0:
        lo, hi = neg_gated.max() if neg_gated.size else 0.0, pos_gated.min()
        print(f"  CLEAN SEPARATION: no gated negative reaches {target:.3f}")
        print(f"  gap on gated images: [{lo:.3f}, {hi:.3f}]")
    else:
        print(f"  OVERLAP: {overlap.size}/{neg_gated.size} gated negatives score >= {target:.3f}")
        for name, (_, g) in neg_by_cat.items():
            bad = g[g >= target] if g.size else np.array([])
            if bad.size:
                print(f"    {name:<12} {bad.size}/{g.size} at or above, max {bad.max():.3f}")

    def summarize(arr, pcts):
        return {f"p{p}": round(float(np.percentile(arr, p)), 4) for p in pcts} if arr.size else {}

    # One file, one owner per block. This script owns `realness` and the
    # `provenance.realness_calibration` entry only; `pose` and `neutrality` belong to
    # backend/scripts/calibrate_gates.py and are read through untouched.
    cfg = json.loads(GATE_CONFIG_PATH.read_text()) if GATE_CONFIG_PATH.is_file() else {}
    preserved = sorted(k for k in cfg if k not in ("realness", "provenance"))
    cfg.setdefault("realness", {})
    cfg.setdefault("provenance", {})["realness_calibration"] = {
        "note": (
            "The gate only sees images the landmark detector meshes, so "
            "'reaching_gate' is the population that matters; everything else is "
            "already rejected as NO_FACE_DETECTED."
        ),
        "known_limitation": (
            "Photorealistic 3D-rendered faces pass this gate. They score inside the "
            "real-photograph distribution and are the negative most likely to be "
            "meshed. Measured across two independent generators (Microsoft "
            "FaceSynthetics and Unreal MetaHuman), so this is a property of the CLIP "
            "prompt set, not of one generator. Accepted deliberately: the leak is one "
            "category, while the cut that would exclude it rejects nearly every real "
            "photograph. Closing it needs a second discriminator, not a threshold."
        ),
        "detector_provides_no_discrimination": (
            "The landmark detector is not a second line of defence here. Unreal "
            "MetaHuman renders reached the gate 30/30 — every one was meshed "
            "successfully — and Microsoft FaceSynthetics 16/30. For photorealistic "
            "renders the entire burden of rejection falls on CLIP. Contrast the "
            "classes the detector does filter on its own: statues, cats and dogs "
            "reached the gate 0/25, 0/6 and 0/6."
        ),
        "target_false_reject": TARGET_FALSE_REJECT,
        "positive_source": "datasets/FairFace/val",
        "negative_categories": {k: int(neg_by_cat[k][0].size) for k in base_cats},
        "negative_sources": {k: int(v[0].size) for k, v in neg_by_cat.items() if ":" in k},
        "render_leak_by_generator": {
            k.split(":", 1)[1]: {
                "reaching_gate": int(v[1].size), "of": int(v[0].size),
                "leak_at_threshold": int((v[1] >= target).sum()) if v[1].size else 0,
                "median": round(float(np.median(v[1])), 4) if v[1].size else None,
                "max": round(float(v[1].max()), 4) if v[1].size else None,
            }
            for k, v in neg_by_cat.items()
            if k.startswith(RENDER_CATEGORY + ":")
        },
        "non_render_negatives_reaching_gate": {
            "n": int(other_gated.size), **summarize(other_gated, PCTS_NEG)},
        "positive_all": {"n": int(pos_all.size), **summarize(pos_all, PCTS_POS)},
        "positive_reaching_gate": {"n": int(pos_gated.size), **summarize(pos_gated, PCTS_POS)},
        "negative_reaching_gate": {"n": int(neg_gated.size), **summarize(neg_gated, PCTS_NEG)},
        "negative_reaching_gate_by_category": {
            k: {"n": int(g.size), "of": int(a.size),
                "median": round(float(np.median(g)), 4), "max": round(float(g.max()), 4)}
            for k, (a, g) in neg_by_cat.items() if g.size
        },
        "previous_threshold": cur,
        "false_reject_at_previous": round(float((pos_gated < cur).mean()), 4),
        "leak_at_previous": round(float((neg_gated >= cur).mean()), 4),
        "false_reject_at_applied_fitted": round(float((pos_gated < target).mean()), 4),
        "holdout_validation": hc,
        "leak_at_applied": round(float((neg_gated >= target).mean()), 4),
        "leak_at_applied_excluding_renders": (
            round(float((other_gated >= target).mean()), 4) if other_gated.size else None),
        "overlap_count": int(overlap.size),
    }
    if write_threshold:
        cfg["realness"]["min_p_photo"] = round(target, 4)
        cfg["provenance"]["realness_calibration"]["applied_threshold"] = round(target, 4)
        print(f"\n  SET realness.min_p_photo {cur:.4f} -> {target:.4f}")
    else:
        cfg["provenance"]["realness_calibration"]["applied_threshold"] = cur
        print(f"\n  min_p_photo left at {cur:.4f} (pass --set to apply {target:.4f})")
    GATE_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"  recorded distributions in {GATE_CONFIG_PATH.relative_to(REPO)}")
    print(f"  preserved (owned elsewhere): {', '.join(preserved) or 'none'}")

    write_cache()
    print(f"  cached {len(set(_MEASURED))} per-image scores in "
          f"{SCORE_CACHE.relative_to(REPO)} (re-runs and alternate splits are cheap)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
