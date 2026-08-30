"""Fail the build if anything the serve path resolves is missing from the image.

Every path is read from the module that actually uses it, never hardcoded here, so
this cannot drift from the code. Missing weights otherwise surface as a 500 on the
first real request, after the revision has already gone live.

Run from the image root:
    python backend/scripts/verify_deploy_paths.py
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))


def main() -> int:
    import gates
    import inference
    import region
    import region_stats
    import suggestion_serve

    required = {
        "face landmarker task": inference._TASK_PATH,
        "beauty checkpoint": inference._MODELS_DIR / "beauty_landmarks_best.pt",
        "gate config": gates.GATE_CONFIG_PATH,
        "region model": region.region_model_path(),
        "region reference stats": region_stats.REFERENCE_STATS_PATH,
        "suggestion ranker": suggestion_serve.DEFAULT_CKPT,
        "suggestion threshold rules": suggestion_serve.DEFAULT_RULES,
        "suggestion catalog": suggestion_serve.DEFAULT_CATALOG,
    }

    missing = []
    print("verifying serve-path artifacts")
    for label, path in required.items():
        path = Path(path)
        ok = path.is_file()
        size = f"{path.stat().st_size / 1e6:8.1f} MB" if ok else "   MISSING"
        print(f"  [{'OK' if ok else '!!'}] {size}  {label:<28} {path}")
        if not ok:
            missing.append(f"{label}: {path}")

    # CLIP is fetched from the HuggingFace hub, not the repo. If it did not get baked
    # in, every cold start pays a ~350 MB download before the first response.
    try:
        from gates import _clip_bundle
        _clip_bundle()
        print("  [OK]              CLIP ViT-B/32                 cached in image")
    except Exception as e:  # noqa: BLE001
        missing.append(f"CLIP ViT-B/32 not cached in image: {e}")
        print(f"  [!!]              CLIP ViT-B/32                 NOT CACHED ({e})")

    if missing:
        print("\nBUILD FAILED — missing serve-path artifacts:")
        for m in missing:
            print(f"  {m}")
        print("\nCheck .gcloudignore: gcloud falls back to .gitignore when it is absent,")
        print("and .gitignore excludes backend/models/*.pt.")
        return 1

    print("\nall serve-path artifacts present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
