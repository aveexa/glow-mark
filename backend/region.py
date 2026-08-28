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
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_PATH = _REPO_ROOT / "datasets" / "FairFace" / "res34_fair_align_multi_7_20190809.pt"
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
