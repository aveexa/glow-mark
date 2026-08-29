"""Calibrate realness.min_p_photo from measured distributions.

The realness gate exists to reject non-photographic input that the landmark
detector will happily mesh. So the population that matters is not "all images" but
"images that reach the gate" — i.e. those where a face was detected and the pose
gate passed. Everything else is already rejected upstream by NO_FACE_DETECTED.

Reports p_photo percentiles for the positive class (FairFace val) and the negative
class (cartoons, anime, 3D renders, paintings, statues, dogs, cats, primates), then
picks a threshold in the gap if the two separate.

Writes the measured distributions into gate_config.json under
provenance.realness_calibration. It never writes realness.min_p_photo: if the two
classes overlap there is no defensible value to write, and if they separate the
choice is still a product call.

Run from:  repo root
    python backend/scripts/calibrate_realness.py <negatives_dir> [n_positive]
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

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


def measure(paths):
    """Return (p_photo for all, p_photo for those that reach the gate)."""
    every, gated = [], []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        _, p_photo = check_realness(img)
        every.append(p_photo)
        if reaches_gate(img):
            gated.append(p_photo)
    return np.array(every), np.array(gated)


def line(label, arr, pcts):
    if arr.size == 0:
        return f"  {label:<26} (empty)"
    q = [np.percentile(arr, p) for p in pcts]
    return f"  {label:<26} n={arr.size:<5}" + "".join(f"{v:>9.3f}" for v in q)


def main() -> int:
    neg_dir = Path(sys.argv[1])
    n_pos = int(sys.argv[2]) if len(sys.argv) > 2 else 600

    print("measuring positives (FairFace val)...", flush=True)
    pos_paths = sorted(x for x in VAL.iterdir() if x.suffix.lower() == ".jpg")[:n_pos]
    pos_all, pos_gated = measure(pos_paths)

    print("measuring negatives...", flush=True)
    cats = sorted(d for d in neg_dir.iterdir() if d.is_dir())
    neg_by_cat = {}
    for d in cats:
        neg_by_cat[d.name] = measure(sorted(d.iterdir()))
    neg_all = np.concatenate([v[0] for v in neg_by_cat.values()])
    neg_gated = np.concatenate([v[1] for v in neg_by_cat.values()])

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

    cfg = json.loads(GATE_CONFIG_PATH.read_text())
    cfg.setdefault("provenance", {})["realness_calibration"] = {
        "note": (
            "Measured, not applied. The gate only sees images the landmark detector "
            "meshes, so 'reaching_gate' is the population that matters. "
            "min_p_photo is unchanged pending a decision on the overlap."
        ),
        "positive_source": "datasets/FairFace/val",
        "negative_categories": {k: int(v[0].size) for k, v in neg_by_cat.items()},
        "positive_all": {"n": int(pos_all.size), **summarize(pos_all, PCTS_POS)},
        "positive_reaching_gate": {"n": int(pos_gated.size), **summarize(pos_gated, PCTS_POS)},
        "negative_reaching_gate": {"n": int(neg_gated.size), **summarize(neg_gated, PCTS_NEG)},
        "negative_reaching_gate_by_category": {
            k: {"n": int(g.size), "of": int(a.size),
                "median": round(float(np.median(g)), 4), "max": round(float(g.max()), 4)}
            for k, (a, g) in neg_by_cat.items() if g.size
        },
        "current_threshold": cur,
        "current_false_reject_rate": round(float((pos_gated < cur).mean()), 4),
        "current_leak_rate": round(float((neg_gated >= cur).mean()), 4),
        "threshold_for_2pct_false_reject": round(target, 4),
        "overlap_count": int(overlap.size),
    }
    GATE_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"\n  recorded distributions in {GATE_CONFIG_PATH.relative_to(REPO)} "
          f"(provenance only; min_p_photo left at {cur:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
