"""E5 — evaluate the gates against the hand-collected dev set.

Labels come from the folder (PASS / FAIL / EDGE) and the expected rejection
reason from the filename prefix, so no CSV is needed.

    python backend/scripts/run_e5_devset.py

Reports:
  * pass rate on PASS          <- the number that matters
  * rejection rate on FAIL, and whether the RIGHT gate fired
  * EDGE outcomes, reported without an expected value
  * every disagreement, by filename, so the photo can be inspected
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import cv2
import numpy as np
import pandas as pd

import inference as inf
from face_normalize import DEFAULT_OUTPUT_SIZE, square_face_crop
from gates import autocorrect_roll, check_neutrality, check_pose, check_realness
from region import predict_region_weights

ROOT = REPO / "datasets" / "devset" / "images"
OUT = REPO / "data" / "processed" / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"}

# filename prefix -> gate expected to reject it
EXPECTED = {
    "cartoon face": "not_real", "anime face": "not_real",
    "3D character render": "not_real", "oil portrait painting": "not_real",
    "marble bust statue": "not_real", "dog face": "not_real",
    "cat face": "not_real", "chimpanzee face": "not_real",
    "gorilla face": "not_real",
    "person smiling": "expression", "person laughing": "expression",
    "person raised eyebrows": "expression", "person eyes closed": "expression",
    "person squinting": "expression",
    "person head tilted": "pose", "person head turned": "pose",
    "person looking up": "pose", "person looking down": "pose",
    "two people": "no_face", "landscape no people": "no_face",
    "very blurry": "no_face",
}


def read_image(path: Path):
    """cv2 first; Pillow fallback for avif/webp cv2 may not decode."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image
        try:
            import pillow_avif  # noqa: F401
        except ImportError:
            pass
        with Image.open(path) as im:
            return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001
        return None


def expected_reason(name: str) -> str | None:
    low = name.lower()
    for prefix, reason in EXPECTED.items():
        if low.startswith(prefix.lower()):
            return reason
    return None


def evaluate(path: Path):
    """Return (verdict, gate, detail). verdict is 'pass' or 'fail'."""
    img = read_image(path)
    if img is None:
        return "error", "read", "could not decode"

    try:
        ok, p_photo = check_realness(img)
        if not ok:
            return "fail", "not_real", f"p_photo={p_photo:.3f}"
    except Exception as e:  # noqa: BLE001
        return "error", "realness", str(e)[:60]

    try:
        det0 = inf._detect_face(img)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        gate = "no_face"
        if "MULTIPLE" in msg.upper():
            gate = "no_face"          # two-people case lands here too
        return "fail", gate, msg[:60]

    crop = square_face_crop(img, det0.landmarks[:468],
                            output_size=DEFAULT_OUTPUT_SIZE)
    square = crop[0] if crop else img
    try:
        det = inf._detect_face(square)
    except Exception as e:  # noqa: BLE001
        return "fail", "no_face", str(e)[:60]

    pose_ok, pose = check_pose(det.matrix)
    if not pose_ok:
        y, p = pose.get("yaw_deg", 0), pose.get("pitch_deg", 0)
        return "fail", "pose", f"yaw={y:.1f} pitch={p:.1f}"

    # Serve rotates the crop upright and re-detects before reading blendshapes or
    # region weights. Without this the dev set scores a frame the pipeline never
    # judges, and its numbers do not describe the deployed gate.
    frame = square
    if pose.get("roll_correction_deg"):
        try:
            frame = autocorrect_roll(square, pose["roll_correction_deg"])
            det = inf._detect_face(frame)
        except Exception:  # noqa: BLE001 — an uncorrected face still scores
            frame = square

    try:
        weights = predict_region_weights(frame)
    except Exception:  # noqa: BLE001
        weights = None

    neutral, hint = check_neutrality(det.blendshapes, weights)
    if not neutral:
        return "fail", "expression", hint or ""
    return "pass", "-", ""


def main():
    if not ROOT.exists():
        sys.exit(f"Missing {ROOT}")

    rows = []
    for folder in ("PASS", "FAIL", "EDGE"):
        d = ROOT / folder
        if not d.exists():
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS)
        print(f"{folder}: {len(files)} images")
        for i, p in enumerate(files, 1):
            verdict, gate, detail = evaluate(p)
            rows.append({"folder": folder, "file": p.name,
                         "expected": "pass" if folder == "PASS" else
                                     ("fail" if folder == "FAIL" else "edge"),
                         "expected_gate": expected_reason(p.name),
                         "verdict": verdict, "gate": gate, "detail": detail})
            if i % 10 == 0:
                print(f"  ...{i}/{len(files)}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "e5_devset_results.csv", index=False)

    errs = d[d.verdict == "error"]
    if not errs.empty:
        print(f"\n{len(errs)} unreadable file(s):")
        for _, r in errs.iterrows():
            print(f"  {r.file}  ({r.detail})")
        print("  If these are .avif, run: pip install pillow-avif-plugin")
        d = d[d.verdict != "error"]

    # ── PASS ────────────────────────────────────────────────────────
    P = d[d.folder == "PASS"]
    n_pass = (P.verdict == "pass").sum()
    print("\n" + "=" * 68)
    print("PASS SET  — the number that matters")
    print("=" * 68)
    print(f"  {n_pass}/{len(P)} accepted  ({100*n_pass/max(len(P),1):.1f}%)")
    rej = P[P.verdict == "fail"]
    if not rej.empty:
        print("\n  wrongly rejected:")
        for g, grp in rej.groupby("gate"):
            print(f"    {g}: {len(grp)}")
            for _, r in grp.iterrows():
                print(f"      {r.file:<34}{r.detail}")
    # ethnicity from the filename stem, e.g. "East Asian_2.webp"
    P = P.assign(group=P.file.str.rsplit("_", n=1).str[0])
    if P.group.nunique() > 1:
        t = P.groupby("group")["verdict"].agg(n="size",
                                              passed=lambda s: (s == "pass").sum())
        t["rate_%"] = 100 * t.passed / t.n
        print("\n  by group:")
        print(t.to_string(float_format=lambda x: f"{x:7.1f}"))

    # ── FAIL ────────────────────────────────────────────────────────
    F = d[d.folder == "FAIL"]
    n_rej = (F.verdict == "fail").sum()
    right = F[(F.verdict == "fail") & (F.gate == F.expected_gate)]
    print("\n" + "=" * 68)
    print("FAIL SET  — rejected, and by the right gate?")
    print("=" * 68)
    print(f"  {n_rej}/{len(F)} rejected      ({100*n_rej/max(len(F),1):.1f}%)")
    print(f"  {len(right)}/{len(F)} by the expected gate "
          f"({100*len(right)/max(len(F),1):.1f}%)")

    print("\n  by expected category:")
    for cat, grp in F.groupby("expected_gate"):
        r = (grp.verdict == "fail").sum()
        c = ((grp.verdict == "fail") & (grp.gate == cat)).sum()
        print(f"    {str(cat):<12} {r}/{len(grp)} rejected, {c} by the right gate")

    leaked = F[F.verdict == "pass"]
    if not leaked.empty:
        print("\n  LEAKED (should have been rejected):")
        for _, r in leaked.iterrows():
            print(f"    {r.file}")
    wrong = F[(F.verdict == "fail") & (F.gate != F.expected_gate)]
    if not wrong.empty:
        print("\n  rejected by a different gate than expected:")
        for _, r in wrong.iterrows():
            print(f"    {r.file:<44}{r.expected_gate} -> {r.gate}")

    # ── EDGE ────────────────────────────────────────────────────────
    E = d[d.folder == "EDGE"]
    if not E.empty:
        print("\n" + "=" * 68)
        print("EDGE SET  — no expected value, reported for judgement")
        print("=" * 68)
        for _, r in E.sort_values("file").iterrows():
            mark = "PASS" if r.verdict == "pass" else f"FAIL ({r.gate})"
            print(f"  {r.file:<46}{mark:<18}{r.detail}")

    print(f"\nwrote {OUT/'e5_devset_results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())