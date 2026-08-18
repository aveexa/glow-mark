"""Resolve local model paths or download from GCS."""

from __future__ import annotations

import os
from pathlib import Path


def download_gcs_uri(gs_uri: str, dest: Path) -> Path:
    """gs://bucket/path → local file. No-op if dest already exists."""
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {gs_uri}")
    _, _, rest = gs_uri.partition("gs://")
    bucket_name, _, blob_name = rest.partition("/")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return dest
    from google.cloud import storage

    client = storage.Client()
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(dest))
    return dest


def resolve_artifact(
    *,
    local_env: str,
    gcs_env: str,
    default_dest: Path,
) -> Path:
    """Prefer LOCAL path env, else download GCS URI to default_dest."""
    local = os.environ.get(local_env)
    if local:
        path = Path(local)
        if not path.is_file():
            raise FileNotFoundError(f"{local_env}={local} is not a file")
        return path
    uri = os.environ.get(gcs_env)
    if not uri:
        raise RuntimeError(f"Set {local_env} (local file) or {gcs_env} (gs://…)")
    return download_gcs_uri(uri, default_dest)
