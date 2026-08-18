"""Glow-Mark Feature prediction API — z-scored 24 → low/ok/high per geometry label."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from gcs_utils import resolve_artifact

app = FastAPI(title="Glow-Mark Feature API")


def _mlp_feature(in_dim: int, out_dim: int) -> nn.Module:
    # Matches state dict keys: 0.weight, 2.weight, 4.weight ...
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, out_dim),
    )


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    path = resolve_artifact(
        local_env="MODEL_PATH",
        gcs_env="MODEL_GCS_URI",
        default_dest=Path("/tmp/models/feature_geometry_model.pt"),
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    feat_cols: list[str] = list(ckpt["feat_cols"])
    label_cols: list[str] = list(ckpt["label_cols"])
    in_dim = len(feat_cols)
    out_dim = int(np.array(list(ckpt["state"].values())[-2]).shape[0])
    model = _mlp_feature(in_dim, out_dim)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return {
        "model": model,
        "in_dim": in_dim,
        "feat_cols": feat_cols,
        "label_cols": label_cols,
    }


class PredictIn(BaseModel):
    features: list[float] = Field(..., min_length=1)


@app.on_event("startup")
def _startup():
    load_bundle()


@app.get("/health")
def health():
    load_bundle()
    return {"ok": True, "model_loaded": True}


@app.post("/v1/feature/predict")
def predict(body: PredictIn):
    b = load_bundle()
    if len(body.features) != b["in_dim"]:
        raise HTTPException(400, f"Expected {b['in_dim']} features, got {len(body.features)}")

    x = np.asarray(body.features, dtype=np.float32).reshape(1, -1)
    with torch.no_grad():
        logits = b["model"](torch.from_numpy(x)).cpu().numpy().reshape(-1)

    label_cols: list[str] = b["label_cols"]
    if logits.shape[0] % len(label_cols) != 0:
        raise HTTPException(
            500,
            f"Feature output dim {logits.shape[0]} not compatible with {len(label_cols)} labels.",
        )

    classes_per = logits.shape[0] // len(label_cols)
    scores = logits.reshape(len(label_cols), classes_per)
    probs = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)

    class_names = (
        ["low", "ok", "high"] if classes_per == 3 else [f"class_{i}" for i in range(classes_per)]
    )

    recommendation_items = []
    recommendations = []
    for i, name in enumerate(label_cols):
        ci = int(np.argmax(probs[i]))
        conf = float(probs[i][ci])
        cls = class_names[ci]
        recommendation_items.append({"label": name, "class": cls, "confidence": round(conf, 4)})
        if cls != "ok":
            recommendations.append(f"{name}: {cls}")

    return {
        "recommendation_items": recommendation_items,
        "recommendations": recommendations,
    }
