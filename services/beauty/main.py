"""Glow-Mark Beauty prediction API — z-scored 136 → beauty score."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from gcs_utils import resolve_artifact

app = FastAPI(title="Glow-Mark Beauty API")


def _mlp_beauty(in_dim: int) -> nn.Module:
    # Matches state dict keys: 0.weight, 3.weight, 6.weight ...
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1),
    )


@lru_cache(maxsize=1)
def load_bundle():
    path = resolve_artifact(
        local_env="MODEL_PATH",
        gcs_env="MODEL_GCS_URI",
        default_dest=Path("/tmp/models/beauty_landmarks_best.pt"),
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    in_dim = int(ckpt["in_dim"])
    model = _mlp_beauty(in_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return {"model": model, "in_dim": in_dim}


class PredictIn(BaseModel):
    features: list[float] = Field(..., min_length=1)


@app.on_event("startup")
def _startup():
    load_bundle()


@app.get("/health")
def health():
    load_bundle()
    return {"ok": True, "model_loaded": True}


@app.post("/v1/beauty/predict")
def predict(body: PredictIn):
    b = load_bundle()
    if len(body.features) != b["in_dim"]:
        raise HTTPException(400, f"Expected {b['in_dim']} features, got {len(body.features)}")
    x = np.asarray(body.features, dtype=np.float32).reshape(1, -1)
    with torch.no_grad():
        raw = float(b["model"](torch.from_numpy(x)).reshape(-1)[0].item())
    return {"score": float(np.clip(raw, 0.0, 100.0)), "score_raw": raw}
