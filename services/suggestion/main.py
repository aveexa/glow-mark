"""Glow-Mark Suggestion prediction API — raw 24 feats → top-k catalog suggestions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from gcs_utils import resolve_artifact
from geometry_cols import FEATURE_COLS
from suggestion_serve import predict_suggestions

app = FastAPI(title="Glow-Mark Suggestion API")


@lru_cache(maxsize=1)
def load_paths() -> dict[str, str]:
    ckpt = resolve_artifact(
        local_env="MODEL_PATH",
        gcs_env="MODEL_GCS_URI",
        default_dest=Path("/tmp/models/suggestion_ranker.pt"),
    )
    catalog = resolve_artifact(
        local_env="CATALOG_PATH",
        gcs_env="CATALOG_GCS_URI",
        default_dest=Path("/tmp/models/suggestions.csv"),
    )
    rules = resolve_artifact(
        local_env="RULES_PATH",
        gcs_env="RULES_GCS_URI",
        default_dest=Path("/tmp/models/suggestion_mapping_rules.csv"),
    )
    return {"ckpt": str(ckpt), "catalog": str(catalog), "rules": str(rules)}


class PredictIn(BaseModel):
    features: dict[str, float]
    top_k: int = Field(default=4, ge=1, le=48)


@app.on_event("startup")
def _startup():
    load_paths()


@app.get("/health")
def health():
    load_paths()
    return {"ok": True, "model_loaded": True}


@app.post("/v1/suggestion/predict")
def predict(body: PredictIn) -> dict[str, Any]:
    try:
        missing = [c for c in FEATURE_COLS if c not in body.features]
        if missing:
            return {"suggestions": []}
        paths = load_paths()
        suggestions = predict_suggestions(
            body.features,
            top_k=body.top_k,
            ckpt_path=paths["ckpt"],
            catalog_path=paths["catalog"],
            rules_path=paths["rules"],
        )
        return {"suggestions": suggestions}
    except Exception:  # noqa: BLE001 — fail-soft per contract
        return {"suggestions": []}
