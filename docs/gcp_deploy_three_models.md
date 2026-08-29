# Glow-Mark — GCP Three-Model Deployment Guide

> [!WARNING]
> **SUPERSEDED — this guide deploys the v1 pipeline.**
>
> It instructs you to deploy `services/orchestrator/` as production `POST /analyze`
> and to "point the Next.js app at the orchestrator only". That orchestrator is a
> second, complete copy of the serve path that was never migrated to v2. It still
> builds and runs — its `requirements.txt` pins `mediapipe>=0.10.14,<0.10.30`, which
> still has the removed `mp.solutions` API — so following this guide produces a
> working service with **none** of the v2 validation gates and **no** region
> conditioning: no realness check, no neutrality check, no roll autocorrect, and a
> single global reference population fitted on 412 mostly white faces.
>
> `backend/` is the maintained serve path. Do not follow this guide until the
> orchestrator has been migrated or the guide has been rewritten against `backend/`.

**Audience:** Engineers deploying Glow-Mark ML serve to Google Cloud  
**Scope:** Put the three production MLPs on GCP as callable prediction APIs, plus an orchestrator that keeps the existing `POST /analyze` contract for Next.js  
**Platform (locked):** Cloud Storage (artifacts) + Cloud Run (CPU services)  
**Model I/O detail:** See [KNOWLEDGE_TRANSFER.md](../KNOWLEDGE_TRANSFER.md) §7 and [data/docs/model_contract_v1.md](../data/docs/model_contract_v1.md)

---

## Table of contents

1. [Architecture](#1-architecture)
2. [Prerequisites](#2-prerequisites)
3. [Artifacts to upload](#3-artifacts-to-upload)
4. [API contracts](#4-api-contracts)
5. [Recommended service layout](#5-recommended-service-layout)
6. [Implement the three prediction services](#6-implement-the-three-prediction-services)
7. [GCS setup](#7-gcs-setup)
8. [Build images and deploy model APIs](#8-build-images-and-deploy-model-apis)
9. [Health and smoke tests](#9-health-and-smoke-tests)
10. [Orchestrator deploy](#10-orchestrator-deploy)
11. [Frontend wiring](#11-frontend-wiring)
12. [Security](#12-security)
13. [Cost and operations](#13-cost-and-operations)
14. [Supervisor demo checklist](#14-supervisor-demo-checklist)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Architecture

Glow-Mark ships **three** neural nets. Deploy them as **three independent Cloud Run prediction APIs**. Keep MediaPipe + geometry extraction in a **fourth** Cloud Run orchestrator that owns `POST /analyze`.

```text
┌─────────────────────┐   multipart image    ┌──────────────────────────────┐
│ Next.js             │ ───────────────────► │ Orchestrator (Cloud Run)     │
│ NEXT_PUBLIC_        │ ◄─────────────────── │ Flask POST /analyze          │
│ BACKEND_URL         │   JSON result        │ MediaPipe + geometry.py      │
└─────────────────────┘                      └──────────────┬───────────────┘
                                                            │
              ┌─────────────────────────────┬───────────────┼────────────────┐
              ▼                             ▼               ▼                │
     Beauty API (CR)                 Feature API (CR)   Suggestion API (CR)     │
     POST /v1/beauty/predict         /v1/feature/…      /v1/suggestion/…        │
     136 → score                     24 → low/ok/high  24 → top-4 catalog    │
              ▲                             ▲               ▲                │
              └─────────────────────────────┴───────────────┘                │
                                    GCS bucket (*.pt, CSV)                   │
```

| Service | Role | Heavy deps |
|---------|------|------------|
| `glow-beauty-api` | Beauty MLP only | `torch`, `numpy` |
| `glow-feature-api` | Feature MLP only | `torch`, `numpy` |
| `glow-suggestion-api` | Ranker + catalog lookup | `torch`, `numpy` + CSV rules/catalog |
| `glow-analyze-api` | Image → landmarks → call 3 APIs | `mediapipe`, `opencv`, `flask` |

**Rules:**

- Do **not** put MediaPipe inside the three model APIs.
- Do **not** call model URLs from the browser; only the orchestrator calls them.
- Suggestion class one-hots at serve time come from **percentile thresholds** (`suggestion_mapping_rules.csv`), **not** from Feature MLP argmax.
- Beauty canonicalize-to-350 + z-score with beauty `mu`/`sd` happens in the **orchestrator** (or beauty service if you send raw 136 pre-z-score — this guide standardizes on **orchestrator builds z-scored 136** for beauty and **z-scored 24** for feature; suggestion receives **raw 24 floats** and encodes internally).

Vertex AI Prediction endpoints are a future upgrade path using the same containers; this guide uses **Cloud Run only**.

---

## 2. Prerequisites

### Accounts and tools

1. A GCP project with **billing enabled**
2. Local tools:
   - [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`, `gsutil`)
   - Docker Desktop (or Docker Engine)
   - Python 3.11+
3. The three `.pt` checkpoints present under `backend/models/` (they may be gitignored — confirm before upload)

### One-time project setup

Replace placeholders everywhere:

| Placeholder | Example |
|-------------|---------|
| `PROJECT_ID` | `glow-mark-prod` |
| `REGION` | `asia-south1` |
| `BUCKET` | `glow-mark-models-$PROJECT_ID` |

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="asia-south1"
export BUCKET="glow-mark-models-${PROJECT_ID}"

gcloud auth login
gcloud config set project "${PROJECT_ID}"
gcloud config set run/region "${REGION}"

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com
```

### IAM roles (deployer account)

Minimum for the person running deploy commands:

- `roles/run.admin`
- `roles/artifactregistry.admin`
- `roles/storage.admin`
- `roles/iam.serviceAccountUser`

Cloud Run default compute service account needs **read** on the models bucket (`roles/storage.objectViewer` on `BUCKET`).

```bash
export PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
export RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gsutil iam ch \
  "serviceAccount:${RUNTIME_SA}:objectViewer" \
  "gs://${BUCKET}"
```

(Run the IAM bind **after** the bucket exists — see §7.)

---

## 3. Artifacts to upload

| Local path | GCS object | Used by |
|------------|------------|---------|
| `backend/models/beauty_landmarks_best.pt` | `gs://$BUCKET/beauty/beauty_landmarks_best.pt` | Beauty API |
| `backend/models/feature_geometry_model.pt` | `gs://$BUCKET/feature/feature_geometry_model.pt` | Feature API |
| `backend/models/suggestion_ranker.pt` | `gs://$BUCKET/suggestion/suggestion_ranker.pt` | Suggestion API |
| `data/catalogs/suggestions.csv` | `gs://$BUCKET/suggestion/suggestions.csv` | Suggestion API |
| `data/processed/suggestion_mapping_rules.csv` | `gs://$BUCKET/suggestion/suggestion_mapping_rules.csv` | Suggestion API |

Checkpoint keys (do not rename inside the `.pt` files):

| Model | Weight key | Norm stats | Other |
|-------|------------|------------|-------|
| Beauty | `model_state` | `mu`, `sd`, `in_dim` | — |
| Feature | `state` | `mu`, `sd` | `feat_cols`, `label_cols` |
| Suggestion | `state` | `feat_mu`, `feat_sd` | `suggestion_ids`, `in_dim=96`, `use_class_onehots` |

Confirm files exist before upload:

```bash
ls -lh backend/models/beauty_landmarks_best.pt \
       backend/models/feature_geometry_model.pt \
       backend/models/suggestion_ranker.pt \
       data/catalogs/suggestions.csv \
       data/processed/suggestion_mapping_rules.csv
```

---

## 4. API contracts

### 4.1 Beauty — `POST /v1/beauty/predict`

**Request**

```json
{
  "features": [0.01, -0.2 /* … exactly 136 floats, already z-scored */]
}
```

**Response**

```json
{
  "score": 72.4,
  "score_raw": 72.41
}
```

- Input length must equal checkpoint `in_dim` (typically **136**).
- Service runs MLP forward only; `score = clip(score_raw, 0, 100)`.

### 4.2 Feature — `POST /v1/feature/predict`

**Request**

```json
{
  "features": [0.1, -0.3 /* … exactly 24 floats, z-scored, feat_cols order */]
}
```

**Response**

```json
{
  "recommendation_items": [
    {"label": "nose_width_ratio", "class": "high", "confidence": 0.91}
  ],
  "recommendations": ["nose_width_ratio: high"]
}
```

- 72 logits → reshape `(24, 3)` → softmax per row → `low` / `ok` / `high`.
- `recommendations` lists only non-`ok` labels (UI string list).

### 4.3 Suggestion — `POST /v1/suggestion/predict`

**Request**

```json
{
  "features": {
    "symmetry_error": 0.02,
    "face_aspect_ratio": 1.35
  },
  "top_k": 4
}
```

`features` must include all 24 keys from `FEATURE_COLS` in [`backend/geometry.py`](../backend/geometry.py). Values are **raw** geometry floats (not z-scored). The service applies p20/p80 thresholds, builds 96-d input, ranks, and looks up catalog text.

**Response**

```json
{
  "suggestions": [
    {"id": "makeup_contour_cheek", "text": "…approved catalog text…", "confidence": 0.83}
  ]
}
```

If checkpoint/catalog load fails, return `{"suggestions": []}` (fail-soft).

### 4.4 Health — all services

`GET /health` → `{"ok": true}` (and optionally `{"model_loaded": true}`).

### 4.5 Orchestrator — `POST /analyze`

Unchanged product contract (multipart `image` field). Response fields the UI already consumes: `score`, `metrics`, `landmarks`, `ratios`, `recommendations`, `recommendation_items`, plus `suggestions`.

---

## 5. Recommended service layout

Create this tree at the repo root (implement once; deploy many times):

```text
services/
  beauty/
    Dockerfile
    requirements.txt
    main.py
  feature/
    Dockerfile
    requirements.txt
    main.py
  suggestion/
    Dockerfile
    requirements.txt
    main.py
    geometry_cols.py          # FEATURE_COLS + threshold helpers (or copy modules)
  orchestrator/
    Dockerfile
    requirements.txt
    # Reuse / copy backend: app.py, inference.py, geometry.py, …
    # Change analyze path to HTTP-call the three URLs
```

Slim prediction `requirements.txt` (beauty / feature):

```text
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
numpy>=1.26.0
torch>=2.0.0
google-cloud-storage>=2.14.0
pydantic>=2.0.0
```

Suggestion adds nothing beyond that if CSVs are loaded from disk after GCS download.

Orchestrator `requirements.txt` mirrors [`backend/requirements.txt`](../backend/requirements.txt) core serve deps (`flask`, `flask-cors`, `mediapipe`, `opencv-python-headless`, `numpy`, `pillow`, `httpx` or `requests`) — **omit** jupyter / lightgbm / bakeoff extras.

---

## 6. Implement the three prediction services

Logic sources:

| Service | Extract from |
|---------|----------------|
| Beauty | [`backend/inference.py`](../backend/inference.py) — `_mlp_beauty`, beauty branch of `_load_models` |
| Feature | Same file — `_mlp_feature`, feature branch + softmax / argmax loop in `analyze_image_bytes` |
| Suggestion | [`backend/suggestion_serve.py`](../backend/suggestion_serve.py), [`backend/suggestion_model.py`](../backend/suggestion_model.py), [`backend/suggestion_rules.py`](../backend/suggestion_rules.py) |

### 6.1 Shared: download checkpoint from GCS at startup

```python
import os
from pathlib import Path

from google.cloud import storage


def download_gcs_uri(gs_uri: str, dest: Path) -> Path:
    """gs://bucket/path → local file. No-op if dest already exists."""
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {gs_uri}")
    _, _, rest = gs_uri.partition("gs://")
    bucket_name, _, blob_name = rest.partition("/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return dest
    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(dest))
    return dest
```

Env vars:

| Variable | Example |
|----------|---------|
| `MODEL_GCS_URI` | `gs://$BUCKET/beauty/beauty_landmarks_best.pt` |
| `PORT` | Cloud Run sets this (default `8080`) |

Suggestion also needs:

| Variable | Example |
|----------|---------|
| `CATALOG_GCS_URI` | `gs://$BUCKET/suggestion/suggestions.csv` |
| `RULES_GCS_URI` | `gs://$BUCKET/suggestion/suggestion_mapping_rules.csv` |

### 6.2 Beauty — `services/beauty/main.py` (skeleton)

```python
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# paste download_gcs_uri here

app = FastAPI(title="Glow-Mark Beauty API")


def _mlp_beauty(in_dim: int) -> nn.Module:
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
    uri = os.environ["MODEL_GCS_URI"]
    path = download_gcs_uri(uri, Path("/tmp/models/beauty_landmarks_best.pt"))
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
```

### 6.3 Feature — `services/feature/main.py` (skeleton)

```python
# Same FastAPI + GCS download pattern.
# Load feature_geometry_model.pt:
#   model = _mlp_feature(len(feat_cols), out_dim)
#   model.load_state_dict(ckpt["state"])
# Predict:
#   logits → reshape (n_labels, 3) → softmax → argmax
#   recommendation_items for all labels
#   recommendations = [f"{label}: {cls}" for non-ok]


def _mlp_feature(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, out_dim),
    )
```

Match the softmax / `class_names = ["low", "ok", "high"]` loop in `analyze_image_bytes` ([`backend/inference.py`](../backend/inference.py)).

### 6.4 Suggestion — `services/suggestion/main.py` (skeleton)

```python
# On startup download:
#   suggestion_ranker.pt, suggestions.csv, suggestion_mapping_rules.csv
# Reuse predict_suggestions(feats, top_k=…) from suggestion_serve.py
# Endpoint accepts dict of 24 raw floats + top_k (default 4)
# On any failure: return {"suggestions": []}
```

Do **not** accept Feature classes as the default path; keep `classes_from_thresholds`.

### 6.5 Dockerfile (each prediction service)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
# suggestion: also COPY geometry helpers / suggestion_*.py as needed

ENV PORT=8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
```

Use `opencv-python-headless` only on the **orchestrator**, not on prediction images.

---

## 7. GCS setup

```bash
# Create bucket (pick a globally unique name)
gsutil mb -l "${REGION}" -p "${PROJECT_ID}" "gs://${BUCKET}"

# Optional: uniform bucket-level access (recommended)
gsutil uniformbucketlevelaccess set on "gs://${BUCKET}"

# Upload artifacts
gsutil cp backend/models/beauty_landmarks_best.pt \
  "gs://${BUCKET}/beauty/beauty_landmarks_best.pt"

gsutil cp backend/models/feature_geometry_model.pt \
  "gs://${BUCKET}/feature/feature_geometry_model.pt"

gsutil cp backend/models/suggestion_ranker.pt \
  "gs://${BUCKET}/suggestion/suggestion_ranker.pt"

gsutil cp data/catalogs/suggestions.csv \
  "gs://${BUCKET}/suggestion/suggestions.csv"

gsutil cp data/processed/suggestion_mapping_rules.csv \
  "gs://${BUCKET}/suggestion/suggestion_mapping_rules.csv"

# Verify
gsutil ls -r "gs://${BUCKET}/**"

# Bind runtime SA read access (see §2)
gsutil iam ch \
  "serviceAccount:${RUNTIME_SA}:objectViewer" \
  "gs://${BUCKET}"
```

---

## 8. Build images and deploy model APIs

### 8.1 Artifact Registry repository

```bash
export AR_REPO="glow-mark"
export AR_HOST="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}"

gcloud artifacts repositories create "${AR_REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Glow-Mark serve images" \
  || true

gcloud auth configure-docker "${REGION}-docker.pkg.dev"
```

### 8.2 Build and push (from each service directory)

```bash
# Beauty
cd services/beauty
docker build -t "${AR_HOST}/beauty-api:v1" .
docker push "${AR_HOST}/beauty-api:v1"
cd ../..

# Feature
cd services/feature
docker build -t "${AR_HOST}/feature-api:v1" .
docker push "${AR_HOST}/feature-api:v1"
cd ../..

# Suggestion
cd services/suggestion
docker build -t "${AR_HOST}/suggestion-api:v1" .
docker push "${AR_HOST}/suggestion-api:v1"
cd ../..
```

Alternative without local Docker: `gcloud builds submit --tag "${AR_HOST}/beauty-api:v1" services/beauty`.

### 8.3 Deploy Cloud Run (three model APIs)

```bash
gcloud run deploy glow-beauty-api \
  --image "${AR_HOST}/beauty-api:v1" \
  --region "${REGION}" \
  --platform managed \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 20 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 60 \
  --set-env-vars "MODEL_GCS_URI=gs://${BUCKET}/beauty/beauty_landmarks_best.pt" \
  --allow-unauthenticated

gcloud run deploy glow-feature-api \
  --image "${AR_HOST}/feature-api:v1" \
  --region "${REGION}" \
  --platform managed \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 20 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 60 \
  --set-env-vars "MODEL_GCS_URI=gs://${BUCKET}/feature/feature_geometry_model.pt" \
  --allow-unauthenticated

gcloud run deploy glow-suggestion-api \
  --image "${AR_HOST}/suggestion-api:v1" \
  --region "${REGION}" \
  --platform managed \
  --memory 1Gi \
  --cpu 1 \
  --concurrency 20 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 60 \
  --set-env-vars "MODEL_GCS_URI=gs://${BUCKET}/suggestion/suggestion_ranker.pt,CATALOG_GCS_URI=gs://${BUCKET}/suggestion/suggestions.csv,RULES_GCS_URI=gs://${BUCKET}/suggestion/suggestion_mapping_rules.csv" \
  --allow-unauthenticated
```

Capture URLs:

```bash
export BEAUTY_URL="$(gcloud run services describe glow-beauty-api --region "${REGION}" --format='value(status.url)')"
export FEATURE_URL="$(gcloud run services describe glow-feature-api --region "${REGION}" --format='value(status.url)')"
export SUGGESTION_URL="$(gcloud run services describe glow-suggestion-api --region "${REGION}" --format='value(status.url)')"

echo "BEAUTY_URL=${BEAUTY_URL}"
echo "FEATURE_URL=${FEATURE_URL}"
echo "SUGGESTION_URL=${SUGGESTION_URL}"
```

**Sizing note:** These MLPs are tiny. **CPU only — no GPUs.** `1Gi` memory is comfortable for `torch` cold start; `512Mi` may work after trimming the image.

---

## 9. Health and smoke tests

```bash
curl -sS "${BEAUTY_URL}/health"
curl -sS "${FEATURE_URL}/health"
curl -sS "${SUGGESTION_URL}/health"
```

### Beauty (136 zeros is a valid shape check; score will be meaningless)

```bash
python3 - <<'PY'
import json, urllib.request, os
url = os.environ["BEAUTY_URL"] + "/v1/beauty/predict"
body = json.dumps({"features": [0.0] * 136}).encode()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(req).read().decode())
PY
```

### Feature (24 zeros)

```bash
python3 - <<'PY'
import json, urllib.request, os
url = os.environ["FEATURE_URL"] + "/v1/feature/predict"
body = json.dumps({"features": [0.0] * 24}).encode()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(req).read().decode())
PY
```

### Suggestion (all FEATURE_COLS present)

```bash
python3 - <<'PY'
import json, urllib.request, os
cols = [
  "symmetry_error","face_aspect_ratio","midface_length_ratio","lowerface_length_ratio",
  "jaw_width_ratio","jaw_angle_sharpness","chin_length_ratio","chin_width_ratio",
  "cheekbone_width_ratio","lower_cheek_ratio","eye_openness_ratio","eye_size_ratio",
  "eye_spacing_ratio","eye_tilt_deg","brow_height_ratio","brow_tilt_deg",
  "nose_width_ratio","nose_length_ratio","nose_tip_deviation_ratio","mouth_width_ratio",
  "mouth_corner_tilt_deg","lip_thickness_ratio","upper_lip_ratio","philtrum_ratio",
]
feats = {c: 0.0 for c in cols}
url = os.environ["SUGGESTION_URL"] + "/v1/suggestion/predict"
body = json.dumps({"features": feats, "top_k": 4}).encode()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
print(urllib.request.urlopen(req).read().decode())
PY
```

For a **realistic** smoke test: run local Flask `/analyze` once, log the z-scored beauty vector, z-scored feature vector, and raw `feats` dict, then replay those payloads against Cloud Run.

---

## 10. Orchestrator deploy

### 10.1 Code change intent

Keep [`backend/app.py`](../backend/app.py) as the HTTP surface. In `analyze_image_bytes` (or a thin wrapper):

1. MediaPipe + pose gate + `extract_geometry_features` — unchanged locally.
2. Build beauty 136 features, z-score with beauty `mu`/`sd` (still need beauty checkpoint **stats** locally, or call beauty API with features z-scored using stats downloaded once).
3. `POST {BEAUTY_URL}/v1/beauty/predict` with the 136-d vector.
4. Build feature 24-d z-scored vector → `POST {FEATURE_URL}/v1/feature/predict`.
5. `POST {SUGGESTION_URL}/v1/suggestion/predict` with raw `feats` dict; on error use `suggestions = []`.
6. Assemble the same JSON keys as today.

Minimal HTTP client pattern:

```python
import os
import httpx

BEAUTY_URL = os.environ["BEAUTY_URL"].rstrip("/")
FEATURE_URL = os.environ["FEATURE_URL"].rstrip("/")
SUGGESTION_URL = os.environ["SUGGESTION_URL"].rstrip("/")


def call_beauty(features_136: list[float]) -> dict:
    r = httpx.post(f"{BEAUTY_URL}/v1/beauty/predict", json={"features": features_136}, timeout=30.0)
    r.raise_for_status()
    return r.json()


def call_feature(features_24: list[float]) -> dict:
    r = httpx.post(f"{FEATURE_URL}/v1/feature/predict", json={"features": features_24}, timeout=30.0)
    r.raise_for_status()
    return r.json()


def call_suggestion(feats: dict, top_k: int = 4) -> list:
    try:
        r = httpx.post(
            f"{SUGGESTION_URL}/v1/suggestion/predict",
            json={"features": feats, "top_k": top_k},
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json().get("suggestions") or []
    except Exception:
        return []
```

**Stats note:** Beauty/Feature z-scoring today uses `mu`/`sd` inside the `.pt` files. Either:

- Keep loading checkpoints in the orchestrator **only for mu/sd/ref frame** and send z-scored vectors to the APIs, or  
- Move z-scoring into the prediction services and send raw features (then update §4 contracts).

This guide assumes **orchestrator owns z-score + canonicalize**; APIs receive model-ready vectors for beauty/feature.

### 10.2 Orchestrator Dockerfile sketch

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
# or COPY app entry that imports inference

ENV PORT=8080
WORKDIR /app/backend
CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 "app:create_app()"
```

(Adjust module path to match how you package `create_app`. For Flask factory + gunicorn you may need `app:create_app()` with `--factory` depending on gunicorn version.)

### 10.3 Deploy orchestrator

```bash
cd services/orchestrator   # or repo root if Dockerfile expects that
gcloud builds submit --tag "${AR_HOST}/analyze-api:v1" .

gcloud run deploy glow-analyze-api \
  --image "${AR_HOST}/analyze-api:v1" \
  --region "${REGION}" \
  --platform managed \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 10 \
  --min-instances 0 \
  --max-instances 5 \
  --timeout 120 \
  --set-env-vars "BEAUTY_URL=${BEAUTY_URL},FEATURE_URL=${FEATURE_URL},SUGGESTION_URL=${SUGGESTION_URL}" \
  --allow-unauthenticated

export ANALYZE_URL="$(gcloud run services describe glow-analyze-api --region "${REGION}" --format='value(status.url)')"
echo "ANALYZE_URL=${ANALYZE_URL}"

curl -sS "${ANALYZE_URL}/health"
```

End-to-end image test:

```bash
curl -sS -X POST "${ANALYZE_URL}/analyze" \
  -F "image=@/path/to/frontal_selfie.jpg;type=image/jpeg" | jq .
```

---

## 11. Frontend wiring

Point the Next.js app at the orchestrator only:

```bash
# .env.local (or hosting provider env)
NEXT_PUBLIC_BACKEND_URL=https://glow-analyze-api-xxxxx-xx.a.run.app
```

Do **not** set the frontend to beauty/feature/suggestion URLs.

Redeploy or restart the Next.js app so the env var is picked up. Existing client code already uses:

```ts
const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5001'
await fetch(`${backendUrl}/analyze`, { method: 'POST', body: form })
```

---

## 12. Security

### Demo / viva (current guide defaults)

`--allow-unauthenticated` on all four services is acceptable for a supervised demo if:

- You do not put PII in logs
- You understand anyone with the URL can call predict

### Production hardening (do this before a public launch)

1. **Remove public access on model APIs**

```bash
gcloud run services remove-iam-policy-binding glow-beauty-api \
  --region="${REGION}" \
  --member="allUsers" \
  --role="roles/run.invoker"

# repeat for glow-feature-api and glow-suggestion-api
```

2. **Grant only the orchestrator runtime SA** `roles/run.invoker` on each model service.
3. Orchestrator uses **authenticated** HTTP (Google ID token) via `google-auth` / `httpx` with audience = service URL.
4. Keep `/analyze` public **or** put it behind Firebase Auth / API Gateway — never expose raw model endpoints to the browser.
5. Optionally enable Cloud Armor / rate limits on the orchestrator.

---

## 13. Cost and operations

| Topic | Guidance |
|-------|----------|
| GPU | Not needed |
| Memory | Prediction: 1Gi; Orchestrator: 2Gi (MediaPipe) |
| Min instances | `0` to save cost; accept cold starts (~few seconds with torch) |
| Warm demo | Temporarily `--min-instances 1` on orchestrator + models before viva |
| Logs | Cloud Run → Logs Explorer; filter by service name |
| Rollback | Redeploy previous image tag (`:v1` → keep `:v0` around) |
| Model update | Upload new `.pt` to GCS same path, then **restart** Cloud Run revision (or rebuild if weights are baked into the image). Prefer GCS download at startup so weight swaps need only a new revision restart. |
| Monitoring | Alert on 5xx rate for `glow-analyze-api` |

Cold start tip: import `torch` at module level once; download weights in `startup` / `@lru_cache` so the first request after scale-from-zero pays the cost once per instance.

---

## 14. Supervisor demo checklist

Use this as a live walkthrough script (~10 minutes).

1. **GCP Console → Cloud Storage** — show `gs://$BUCKET` with three `.pt` files + CSVs.
2. **Cloud Run** — show four services: `glow-beauty-api`, `glow-feature-api`, `glow-suggestion-api`, `glow-analyze-api`.
3. **Postman / curl** — hit each model `/health`, then each `/v1/.../predict` with a real feature vector; show JSON.
4. **Architecture slide** — one sentence: “MediaPipe stays in the orchestrator; three MLPs are separate prediction APIs.”
5. **Product path** — open Next.js `/analyze`, upload a frontal selfie, show score + recommendations (+ suggestions if UI wired).
6. **Gotchas call-outs** (from KT):
   - Suggestion one-hots ≠ Feature argmax (percentile rules).
   - Beauty features are resolution-invariant (350 canvas + canonicalize).
   - Suggestion fail-soft returns `[]` without breaking analyze.
7. **Cost** — “CPU-only Cloud Run; no GPU.”

---

## 15. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Cloud Run start failed / permission on GCS | Runtime SA cannot read bucket | Grant `objectViewer` on bucket (§2 / §7) |
| `Expected 136 features` | Orchestrator sent wrong length | Confirm flatten of 68×2 after canonicalize |
| Feature all `ok` on real faces | Sent raw floats instead of z-scored | Apply feature `mu`/`sd` before call |
| Suggestion always `[]` | Missing ckpt/CSV or wrong env URIs | Check Cloud Run env vars + GCS objects; check logs |
| `NO_FACE` / pose errors | Client image, not GCP | Frontal selfie; yaw/pitch within ±25° |
| Timeout on `/analyze` | Cold start × 3 model calls | Raise timeout to 120s; set min-instances 1 for demo |
| OpenCV / MediaPipe import error in container | Missing system libs | Use `libgl1` / `libglib2.0-0` or headless wheels |
| Frontend CORS | Orchestrator missing CORS | Keep `flask-cors` on orchestrator as in `app.py` |
| Torch `weights_only` errors | Older/newer torch mismatch | Pin `torch` in requirements; use `weights_only=False` for these checkpoints |

---

## Appendix A — Environment variable cheat sheet

```bash
export PROJECT_ID="..."
export REGION="asia-south1"
export BUCKET="glow-mark-models-${PROJECT_ID}"
export AR_HOST="${REGION}-docker.pkg.dev/${PROJECT_ID}/glow-mark"

export BEAUTY_URL="https://...."
export FEATURE_URL="https://...."
export SUGGESTION_URL="https://...."
export ANALYZE_URL="https://...."

# Frontend
export NEXT_PUBLIC_BACKEND_URL="${ANALYZE_URL}"
```

---

## Appendix B — FEATURE_COLS order (suggestion + feature)

Must stay locked with [`backend/geometry.py`](../backend/geometry.py):

1. `symmetry_error`  
2. `face_aspect_ratio`  
3. `midface_length_ratio`  
4. `lowerface_length_ratio`  
5. `jaw_width_ratio`  
6. `jaw_angle_sharpness`  
7. `chin_length_ratio`  
8. `chin_width_ratio`  
9. `cheekbone_width_ratio`  
10. `lower_cheek_ratio`  
11. `eye_openness_ratio`  
12. `eye_size_ratio`  
13. `eye_spacing_ratio`  
14. `eye_tilt_deg`  
15. `brow_height_ratio`  
16. `brow_tilt_deg`  
17. `nose_width_ratio`  
18. `nose_length_ratio`  
19. `nose_tip_deviation_ratio`  
20. `mouth_width_ratio`  
21. `mouth_corner_tilt_deg`  
22. `lip_thickness_ratio`  
23. `upper_lip_ratio`  
24. `philtrum_ratio`

---

## Appendix C — Done criteria

You are done when:

1. Three prediction services respond on `/health` and `/v1/.../predict` from public HTTPS URLs (or authenticated URLs in prod).
2. Orchestrator `POST /analyze` returns the same JSON shape as local Flask.
3. Next.js `/analyze` works with `NEXT_PUBLIC_BACKEND_URL` pointing at `glow-analyze-api`.
4. You can explain to a supervisor: **GCS stores weights; Cloud Run serves three model APIs; orchestrator owns MediaPipe and product aggregation.**

---

*This document is the operational playbook for GCP serve of Glow-Mark’s three MLPs. Model training and feature math remain documented under `data/docs/` and `KNOWLEDGE_TRANSFER.md`.*
