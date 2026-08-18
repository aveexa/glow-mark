"""Suggestion ranker MLP — serve helpers (vendored for Cloud Run image)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from geometry_cols import FEATURE_COLS, FEATURE_CONTRACT_VERSION

CLASS_ORDER = ("low", "ok", "high")
IN_DIM = len(FEATURE_COLS) + len(FEATURE_COLS) * len(CLASS_ORDER)  # 96


class SuggestionRanker(nn.Module):
    def __init__(self, in_dim: int, n_labels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def y_one_hot(y_classes: Mapping[str, str]) -> np.ndarray:
    """72-d one-hot for 24 features × {low,ok,high}."""
    out = np.zeros(len(FEATURE_COLS) * 3, dtype=np.float32)
    for i, feat in enumerate(FEATURE_COLS):
        cls = str(y_classes.get(feat, y_classes.get(f"y_{feat}", "ok")))
        if cls not in CLASS_ORDER:
            cls = "ok"
        out[i * 3 + CLASS_ORDER.index(cls)] = 1.0
    return out


def encode_features(
    feats: Mapping[str, float],
    y_classes: Mapping[str, str],
    feat_mu: np.ndarray,
    feat_sd: np.ndarray,
) -> np.ndarray:
    """Return shape (1, 96) float32."""
    x = np.array([[float(feats[c]) for c in FEATURE_COLS]], dtype=np.float32)
    xn = (x - feat_mu.reshape(1, -1)) / (feat_sd.reshape(1, -1) + 1e-6)
    oh = y_one_hot(y_classes).reshape(1, -1)
    return np.concatenate([xn, oh], axis=1).astype(np.float32)


def decode_top_k(
    logits: np.ndarray,
    vocab: Sequence[str],
    k: int = 4,
) -> list[tuple[str, float]]:
    """Sigmoid probs → top-k (id, confidence)."""
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    order = np.argsort(-probs)[:k]
    return [(vocab[i], float(probs[i])) for i in order]


def build_mlp(in_dim: int, n_labels: int) -> SuggestionRanker:
    return SuggestionRanker(in_dim, n_labels)


def load_checkpoint(path: Path | str, map_location: str = "cpu") -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    ids = list(ckpt["suggestion_ids"])
    model = build_mlp(int(ckpt.get("in_dim", IN_DIM)), len(ids))
    model.load_state_dict(ckpt["state"])
    model.eval()
    return {
        "model": model,
        "feat_mu": np.asarray(ckpt["feat_mu"], dtype=np.float32).reshape(-1),
        "feat_sd": np.asarray(ckpt["feat_sd"], dtype=np.float32).reshape(-1),
        "suggestion_ids": ids,
        "feature_contract_version": ckpt.get("feature_contract_version", FEATURE_CONTRACT_VERSION),
        "use_class_onehots": bool(ckpt.get("use_class_onehots", True)),
        "in_dim": int(ckpt.get("in_dim", IN_DIM)),
    }
