"""Validation gates for the analyze pipeline: real photograph, head pose, neutral expression.

Pipeline positions 2, 3 and 5 (see inference.analyze_image_bytes). Region detection
sits at position 4, between pose and neutrality, because neutrality thresholds are
population-specific.

No threshold is a literal in this module. Every cut is read from
``data/interim/gate_config.json``, which backend/scripts/calibrate_gates.py produces.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import cv2
import numpy as np
import torch

from geometry import euler_from_matrix

_REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_CONFIG_PATH = _REPO_ROOT / "data" / "interim" / "gate_config.json"

# Selects which neutrality thresholds are used, and nothing else. Region detection
# runs on every request regardless of this value, because the feature norms behind
# the classes, the UI gauges and the beauty score all need the region mixture. The
# older name (NEUTRALITY_MODE) read like a switch that turned region handling off.
NEUTRALITY_THRESHOLD_MODE_ENV = "NEUTRALITY_THRESHOLD_MODE"
NEUTRALITY_THRESHOLD_MODE_REGION = "region"
NEUTRALITY_THRESHOLD_MODE_GLOBAL = "global"

# Zero-shot class set for the realness gate. The first group is the "this really is
# a photo of a person" mass; the rest are the ways v1 got fooled — drawings, renders,
# sculpture, and animal faces, all of which the landmark detector will happily mesh.
REALNESS_PASS_PROMPTS: Tuple[str, ...] = (
    "a photograph of a real person's face",
    "a portrait photo of a human",
)
REALNESS_REJECT_PROMPTS: Tuple[str, ...] = (
    "a cartoon drawing of a face",
    "an anime character face",
    "a 3D rendered character",
    "an oil painting of a person",
    "a marble statue face",
    "a photograph of a dog",
    "a photograph of a cat",
    "a photograph of a monkey or ape",
)
REALNESS_PROMPTS: Tuple[str, ...] = REALNESS_PASS_PROMPTS + REALNESS_REJECT_PROMPTS

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"


@lru_cache(maxsize=1)
def load_gate_config() -> Dict[str, Any]:
    """Load the calibrated gate thresholds once per process."""
    with GATE_CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def beauty_region_normalize() -> bool:
    """Whether the beauty score is z-normalised against the user's region.

    Off by default. E4b measured this against MEBeauty human ratings and it made
    agreement worse, not better — pooled Pearson 0.236 raw against 0.209
    normalised, bootstrap 95% CI [-0.038, -0.016]. The code path is kept because
    the finding is one dataset deep and the reference statistics still carry
    beauty_score_raw. This flag governs the beauty score only; region conditioning
    of the geometry features is a different mechanism and is always on.
    """
    return bool(load_gate_config().get("beauty", {}).get("region_normalize", False))


def neutrality_threshold_mode() -> str:
    """``region`` (default) or ``global``, from the NEUTRALITY_THRESHOLD_MODE env var."""
    mode = os.environ.get(
        NEUTRALITY_THRESHOLD_MODE_ENV, NEUTRALITY_THRESHOLD_MODE_REGION
    ).strip().lower()
    if mode in (NEUTRALITY_THRESHOLD_MODE_REGION, NEUTRALITY_THRESHOLD_MODE_GLOBAL):
        return mode
    return NEUTRALITY_THRESHOLD_MODE_REGION


# ---------------------------------------------------------------- realness (2)


@lru_cache(maxsize=1)
def _clip_bundle():
    """Helper: cached CLIP model, preprocess transform, and normalized prompt embeddings."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    with torch.no_grad():
        text_features = model.encode_text(tokenizer(list(REALNESS_PROMPTS)))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return model, preprocess, text_features


def check_realness(img_bgr: np.ndarray) -> Tuple[bool, float]:
    """Zero-shot CLIP: is this a photograph of a person rather than a drawing/render/animal?

    Returns ``(passed, p_photo)`` where ``p_photo`` is the softmax mass on the two
    human-photograph prompts.

    KNOWN LIMITATION — photorealistic 3D-rendered faces pass this gate. They score
    inside the real-photograph distribution (median ~0.70 against a real p25 of 0.76)
    and are also the negative the landmark detector is most likely to mesh. Measured
    across two independent generators, so it is a property of the prompt set rather
    than of one renderer. Accepted deliberately: no threshold excludes renders without
    rejecting nearly every real photograph, so the cut is set for user cost instead.
    Closing this needs a second discriminator, not a different number. Every other
    negative class measured — cartoons, anime, paintings, statues, animals, primates —
    is separated cleanly. See provenance.realness_calibration in gate_config.json.
    """
    from PIL import Image

    model, preprocess, text_features = _clip_bundle()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0)

    with torch.no_grad():
        image_features = model.encode_image(tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * image_features @ text_features.T
        probs = logits.softmax(dim=-1).cpu().numpy().reshape(-1)

    p_photo = float(probs[: len(REALNESS_PASS_PROMPTS)].sum())
    min_p_photo = float(load_gate_config()["realness"]["min_p_photo"])
    return p_photo >= min_p_photo, p_photo


# -------------------------------------------------------------------- pose (3)


def check_pose(matrix: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    """Head pose gate from the 4x4 facial transformation matrix.

    Yaw and pitch outside the calibrated frontal band fail. Roll does not fail while
    it is inside ``roll_max_deg`` and ``roll_autocorrect`` is on — instead the returned
    ``roll_correction_deg`` tells the caller how far to rotate the image back upright.
    """
    limits = load_gate_config()["pose"]
    yaw, pitch, roll = euler_from_matrix(matrix)

    # Roll beyond the limit fails whether or not autocorrect is on — past that
    # angle the rotation would be recovering a face the detector barely resolved.
    within_roll = abs(roll) <= float(limits["roll_max_deg"])
    autocorrect = bool(limits["roll_autocorrect"]) and within_roll

    passed = (
        abs(yaw) <= float(limits["yaw_max_deg"])
        and abs(pitch) <= float(limits["pitch_max_deg"])
        and within_roll
    )
    pose = {
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "roll_deg": roll,
        "roll_correction_deg": roll if autocorrect else 0.0,
    }
    return passed, pose


def autocorrect_roll(img_bgr: np.ndarray, roll_deg: float) -> np.ndarray:
    """Rotate the image about its centre so the head sits upright. Identity at 0°."""
    if not roll_deg:
        return img_bgr
    h, w = img_bgr.shape[:2]
    rotation = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -float(roll_deg), 1.0)
    return cv2.warpAffine(
        img_bgr, rotation, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# -------------------------------------------------------------- neutrality (5)


def neutrality_threshold(
    signal: str,
    region_weights: Mapping[str, float] | None = None,
) -> float:
    """Threshold for one blendshape: the global cut, or the region-weighted blend of them.

    Falls back to the global cut whenever region weights cannot be applied — global
    mode, no weights, signal absent from the per-region table, or degenerate weights.
    """
    neutrality = load_gate_config()["neutrality"]
    global_threshold = float(neutrality["global"][signal])

    if region_weights is None or neutrality_threshold_mode() == NEUTRALITY_THRESHOLD_MODE_GLOBAL:
        return global_threshold

    per_region = neutrality.get("per_region", {}).get(signal)
    if not per_region:
        return global_threshold

    # Renormalize over the regions the table actually covers; with a full softmax
    # over the 7 known regions this is exactly the sum in the brief.
    total_weight = sum(float(w) for r, w in region_weights.items() if r in per_region)
    if total_weight <= 0.0:
        return global_threshold
    blended = sum(float(w) * float(per_region[r]) for r, w in region_weights.items() if r in per_region)
    return blended / total_weight


def check_neutrality(
    blendshapes: Mapping[str, float],
    region_weights: Mapping[str, float] | None = None,
) -> Tuple[bool, str | None]:
    """Neutral-expression gate. Returns ``(passed, hint)``; hint names what to relax.

    Rejects on the first signal over its threshold, in config order.
    """
    neutrality = load_gate_config()["neutrality"]
    messages = neutrality.get("messages", {})

    for signal in neutrality["global"]:
        value = blendshapes.get(signal)
        if value is None:
            continue
        if float(value) > neutrality_threshold(signal, region_weights):
            return False, messages.get(signal)
    return True, None
