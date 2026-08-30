# GlowMark v2 backend for Cloud Run.
#
# Build from the REPO ROOT, not from backend/ — the serve path reads data/ as well:
#   docker build -f backend/Dockerfile -t glowmark-backend .
#
# This is backend/. services/orchestrator/ is the superseded v1 pipeline and must
# not be deployed; see the note at the top of its app.py.
FROM python:3.12-slim

# 3.12, not 3.14: torch wheel coverage on 3.14 is thin and the image would build
# from source or fail outright.

# OpenCV needs these even for headless decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # CLIP is fetched through huggingface_hub; pin the cache so the copy baked in
    # at build time is the one found at runtime.
    HF_HOME=/opt/hf-cache \
    # Cloud Run is CPU-only. Without this, torch pulls its CUDA build (~2.5 GB of
    # nvidia wheels) that can never be used.
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install open_clip_torch gunicorn

# Application code and every artifact the serve path resolves. The two suggestion
# files under data/ are easy to miss: without them predict_suggestions fails soft
# and returns an empty list rather than erroring, so the omission is silent.
COPY backend/ ./backend/
COPY data/interim/gate_config.json              ./data/interim/gate_config.json
COPY data/processed/region_reference_stats.csv  ./data/processed/region_reference_stats.csv
COPY data/processed/suggestion_mapping_rules.csv ./data/processed/suggestion_mapping_rules.csv
COPY data/catalogs/suggestions.csv              ./data/catalogs/suggestions.csv

# Bake CLIP into the image. Left to runtime this is a ~350 MB download on every
# cold start, paid before the first response.
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')"

# Fail the build if anything the serve path resolves is absent. Reads each path
# from the module that uses it, so it cannot drift from the code.
RUN python backend/scripts/verify_deploy_paths.py

ENV PORT=8080
EXPOSE 8080

# 1 worker: each one loads torch, MediaPipe, CLIP and the ResNet separately, so a
# second worker doubles memory for no throughput on a single-CPU instance. Threads
# carry the concurrency. 120s covers a cold start warming every model.
CMD exec gunicorn --chdir /app/backend \
    --bind "0.0.0.0:${PORT}" \
    --workers 1 --threads 4 --timeout 120 \
    "app:create_app()"
