"""FairFace region inference — the 7-way mixture that conditions the norms.

Only the race head is read. The checkpoint (ResNet-34, 18 logits) also carries a
gender head and an age head; **sex is never inferred from a user photograph**, by
design, so those logits are sliced away and never returned.

The full 7-way softmax is returned. Never argmax it — downstream thresholds and
reference statistics are a weighted blend across regions, not a bucket a person
gets sorted into.

Output order is FairFace's own (see its predict.py race_pred mapping) and is
verified against ground-truth labels by backend/scripts/verify_region_order.py.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

_MODELS_DIR = Path(__file__).resolve().parent / "models"
# Lives with the other serve checkpoints rather than under datasets/, so serve does
# not depend on the dataset tree. Like beauty_landmarks_best.pt and the other .pt
# files it is gitignored (backend/.gitignore: models/*.pt) and supplied out of band.
_DEFAULT_MODEL_PATH = _MODELS_DIR / "res34_fair_align_multi_7_20190809.pt"
REGION_MODEL_PATH_ENV = "REGION_MODEL_PATH"

# FairFace's race head order. Getting this wrong silently mismatches every
# threshold, so it is asserted against the label CSV by the verify script.
REGION_NAMES: Tuple[str, ...] = (
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
)
_N_RACE = len(REGION_NAMES)
_N_LOGITS = 18  # 7 race + 2 gender + 9 age; only the first 7 are ever read.

# Display-only. The keys above are FairFace's and stay as they are — they index the
# threshold tables and the reference statistics. These labels are what a user sees,
# and they name a comparison group, never an identity: the pipeline compares
# measurements against a population, it does not decide what anyone is.
REGION_DISPLAY_LABELS: Dict[str, str] = {
    "White": "European",
    "Black": "African",
    "Latino_Hispanic": "Latino / Hispanic",
    "East Asian": "East Asian",
    "Southeast Asian": "Southeast Asian",
    "Indian": "South Asian",
    "Middle Eastern": "Middle Eastern",
}

# Sentinel for "do not condition on a region"; not a member of REGION_NAMES.
GLOBAL_REGION = "global"
GLOBAL_DISPLAY_LABEL = "All (global)"

# Below this the top region is not a confident enough single answer to name alone,
# so the label names the top two instead. Display wording only — no gate reads it.
REFERENCE_LABEL_MIN_TOP = 0.5

# FairFace's own preprocessing (predict.py): resize to 224, ImageNet normalization.
_INPUT_SIZE = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def region_model_path() -> Path:
    """Checkpoint location, overridable with REGION_MODEL_PATH."""
    override = os.environ.get(REGION_MODEL_PATH_ENV)
    return Path(override) if override else _DEFAULT_MODEL_PATH


@lru_cache(maxsize=1)
def _load_region_model() -> nn.Module:
    """Helper: cached ResNet-34 with the 18-logit FairFace head."""
    from torchvision import models

    path = region_model_path()
    if not path.is_file():
        raise FileNotFoundError(f"Region model not found: {path}")

    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, _N_LOGITS)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    model.eval()
    return model


def _preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """Helper: BGR image → normalized (1, 3, 224, 224) tensor."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def predict_region_weights(img_bgr: np.ndarray) -> Dict[str, float]:
    """Full 7-way region softmax as ``{region_name: weight}``. Weights sum to 1."""
    model = _load_region_model()
    with torch.no_grad():
        logits = model(_preprocess(img_bgr)).reshape(-1)[:_N_RACE]
        probs = torch.softmax(logits, dim=0).cpu().numpy()
    return {name: float(p) for name, p in zip(REGION_NAMES, probs)}


def display_label(region: str) -> str:
    """UI label for one region key, or the key itself if it is unknown."""
    if region == GLOBAL_REGION:
        return GLOBAL_DISPLAY_LABEL
    return REGION_DISPLAY_LABELS.get(region, region)


def region_choices() -> list[dict]:
    """The comparison groups a user may pick, in the model's own order, plus global."""
    return [{"value": r, "label": display_label(r)} for r in REGION_NAMES] + [
        {"value": GLOBAL_REGION, "label": GLOBAL_DISPLAY_LABEL}
    ]


def normalize_region_override(value: str | None) -> str | None:
    """Validate a user-supplied override against the known groups.

    Returns a region name, ``GLOBAL_REGION``, or None when the value is absent or
    unrecognised — an unrecognised override is ignored rather than rejected, so a
    stale client cannot break analysis.
    """
    if not value:
        return None
    candidate = str(value).strip()
    if candidate == GLOBAL_REGION:
        return GLOBAL_REGION
    for name in REGION_NAMES:
        if candidate.lower() == name.lower():
            return name
    return None


def reference_label(weights: Mapping[str, float] | None) -> str:
    """Human-readable name for the comparison group a mixture represents.

    Names the top group when it clearly dominates, otherwise names the top two, so
    the label does not assert a single answer the mixture does not support.
    """
    if not weights:
        return GLOBAL_DISPLAY_LABEL
    ranked = sorted(weights.items(), key=lambda kv: -float(kv[1]))
    if not ranked:
        return GLOBAL_DISPLAY_LABEL
    if float(ranked[0][1]) >= REFERENCE_LABEL_MIN_TOP or len(ranked) == 1:
        return display_label(ranked[0][0])
    return " / ".join(display_label(r) for r, _ in ranked[:2])
