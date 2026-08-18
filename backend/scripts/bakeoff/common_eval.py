"""Shared bakeoff helpers: data I/O, metrics, timing, metrics.json merge.

All bakeoff trainers should import from this module so accuracy and efficiency
metrics stay identical across algorithms.

Usage (from repo root, after sys.path includes backend/):

  from common_eval import load_rows, eval_scores, write_metrics_entry, ...
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

# bakeoff/ → scripts/ → backend/
_BAKEOFF_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _BAKEOFF_DIR.parent
BACKEND_DIR = _SCRIPTS_DIR.parent
REPO_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(_BAKEOFF_DIR) not in sys.path:
    sys.path.insert(0, str(_BAKEOFF_DIR))

from geometry import FEATURE_COLS  # noqa: E402
from suggestion_label_trust import row_trust_weights  # noqa: E402
from suggestion_model import (  # noqa: E402
    IN_DIM,
    encode_dataset_row,
    load_checkpoint,
    multilabel_target,
    parse_id_list,
)
from suggestion_rl import TOP_K, eval_ndcg_map, gold_relevance  # noqa: E402
from suggestion_rules import load_catalog  # noqa: E402

SEED = 42
DEFAULT_DATASET = REPO_ROOT / "data" / "processed" / "suggestion_dataset.parquet"
DEFAULT_CATALOG = REPO_ROOT / "data" / "catalogs" / "suggestions.csv"
DEFAULT_METRICS = REPO_ROOT / "data" / "processed" / "bakeoff" / "metrics.json"
DEFAULT_MODELS_DIR = REPO_ROOT / "backend" / "models" / "bakeoff"


def resolve_dataset(path: Path) -> Path:
    """Prefer parquet; fall back to sibling .csv if missing."""
    if path.is_file():
        return path
    csv_path = path.with_suffix(".csv")
    if csv_path.is_file():
        return csv_path
    raise FileNotFoundError(f"dataset not found: {path}")


def load_rows(path: Path) -> list[dict[str, str]]:
    """Load Dataset C rows from CSV or Parquet as list[dict[str, str]]."""
    path = resolve_dataset(Path(path))
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        return df.astype(object).where(pd.notnull(df), "").to_dict(orient="records")
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def priority_ids(row: Mapping[str, str]) -> list[str]:
    return parse_id_list(row.get("priority_order") or row.get("suggestion_ids") or "")


def fit_mu_sd(rows: Sequence[Mapping[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std on FEATURE_COLS (fit on train only)."""
    x = np.array([[float(r[c]) for c in FEATURE_COLS] for r in rows], dtype=np.float32)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    return mu.astype(np.float32), sd


def split_rows(
    rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Split by frozen `split` column → train / val / test."""
    train = [dict(r) for r in rows if r.get("split") == "train"]
    val = [dict(r) for r in rows if r.get("split") == "val"]
    test = [dict(r) for r in rows if r.get("split") == "test"]
    return train, val, test


def build_vocab(
    train_rows: Sequence[Mapping[str, str]],
    catalog_ids: Sequence[str] | Mapping[str, Any],
) -> list[str]:
    """Suggestion IDs with ≥1 train positive ∩ catalog."""
    catalog_set = set(catalog_ids.keys() if isinstance(catalog_ids, Mapping) else catalog_ids)
    counts: Counter[str] = Counter()
    for r in train_rows:
        for sid in priority_ids(r):
            if sid in catalog_set:
                counts[sid] += 1
    return sorted(counts.keys())


def build_xy(
    rows: Sequence[Mapping[str, str]],
    vocab: Sequence[str],
    feat_mu: np.ndarray,
    feat_sd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(N, 96) features + (N, K) multi-hot targets."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for r in rows:
        xs.append(encode_dataset_row(r, feat_mu, feat_sd)[0])
        ys.append(multilabel_target(priority_ids(r), vocab))
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def build_x_rel(
    rows: Sequence[Mapping[str, str]],
    vocab: Sequence[str],
    feat_mu: np.ndarray,
    feat_sd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(N, 96) features + (N, K) graded relevance (gains 3,2,1,1)."""
    xs: list[np.ndarray] = []
    rels: list[np.ndarray] = []
    for r in rows:
        xs.append(encode_dataset_row(r, feat_mu, feat_sd)[0])
        rels.append(gold_relevance(priority_ids(r), vocab).numpy())
    return np.stack(xs).astype(np.float32), np.stack(rels).astype(np.float32)


def eval_scores(
    scores: np.ndarray | torch.Tensor,
    rel: np.ndarray | torch.Tensor,
    k: int = TOP_K,
) -> tuple[float, float]:
    """Mean NDCG@k and MAP@k over rows. scores/rel shape (N, K)."""
    if isinstance(scores, np.ndarray):
        scores_t = torch.from_numpy(scores.astype(np.float32))
    else:
        scores_t = scores.detach().float().cpu()
    if isinstance(rel, np.ndarray):
        rel_t = torch.from_numpy(rel.astype(np.float32))
    else:
        rel_t = rel.detach().float().cpu()
    return eval_ndcg_map(scores_t, rel_t, k=k)


def time_train(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run fn() and return (result, wall-clock seconds)."""
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def time_infer_ms(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    warmup: int = 3,
) -> float:
    """Mean inference latency in ms per sample (batch forward)."""
    n = max(int(x.shape[0]), 1)
    for _ in range(max(warmup, 0)):
        predict_fn(x)
    t0 = time.perf_counter()
    predict_fn(x)
    elapsed = time.perf_counter() - t0
    return float(1000.0 * elapsed / n)


def count_params(model: Any) -> int:
    """Parameter count for torch nn.Module or sklearn-like estimators."""
    if isinstance(model, torch.nn.Module):
        return int(sum(p.numel() for p in model.parameters()))
    # sklearn / joblib: count coefficients when present
    coef = getattr(model, "coef_", None)
    if coef is not None:
        return int(np.asarray(coef).size)
    # LightGBM: sum num_leaves across trees (proxy for model capacity)
    booster = getattr(model, "booster_", None)
    if booster is not None:
        try:
            dump = booster.dump_model()
            trees = dump.get("tree_info") or []
            leaves = sum(int(t.get("num_leaves") or 0) for t in trees)
            return int(leaves) if leaves else int(booster.num_trees())
        except Exception:
            try:
                return int(booster.num_trees())
            except Exception:
                pass
    estimators = getattr(model, "estimators_", None)
    if estimators is not None:
        total = 0
        for est in np.asarray(estimators).ravel():
            total += count_params(est)
        return total
    return 0


def make_entry(
    *,
    tag: str,
    category: str,
    val_ndcg: float,
    val_map: float,
    test_ndcg: float | None,
    test_map: float | None,
    train_seconds: float,
    infer_ms_per_sample: float,
    n_params: int,
    seed: int = SEED,
    artifact: str = "",
    k: int = TOP_K,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one bakeoff metrics row (BakeoffEntry fields)."""
    entry: dict[str, Any] = {
        "tag": tag,
        "category": category,
        "k": k,
        f"val_ndcg@{k}": round(float(val_ndcg), 6),
        f"val_map@{k}": round(float(val_map), 6),
        "train_seconds": round(float(train_seconds), 4),
        "infer_ms_per_sample": round(float(infer_ms_per_sample), 6),
        "n_params": int(n_params),
        "seed": int(seed),
        "artifact": artifact,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    if test_ndcg is not None:
        entry[f"test_ndcg@{k}"] = round(float(test_ndcg), 6)
    if test_map is not None:
        entry[f"test_map@{k}"] = round(float(test_map), 6)
    if extra:
        entry.update(dict(extra))
    return entry


def write_metrics_entry(path: Path, entry: Mapping[str, Any]) -> None:
    """Merge-write entry into metrics.json under models[tag]."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"models": {}}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
                payload.setdefault("models", {})
        except json.JSONDecodeError:
            payload = {"models": {}}
    tag = str(entry.get("tag", "unknown"))
    payload["models"][tag] = dict(entry)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def add_bakeoff_args(parser: Any) -> None:
    """Attach shared CLI flags used by every bakeoff trainer."""
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out", type=Path, default=None, help="Artifact path (.pt / .joblib)")
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--k", type=int, default=TOP_K)


def prepare_bakeoff_data(
    dataset: Path,
    catalog: Path,
    *,
    seed: int = SEED,
    init_path: Path | None = None,
) -> dict[str, Any]:
    """Load Dataset C, build vocab / matrices shared by all bakeoff trainers.

    If init_path is set, reuse vocab + feat_mu/sd from that torch checkpoint
    (REINFORCE warm-start). Returns init_bundle when init_path is provided.
    """
    del seed  # seeding is caller's responsibility; kept for API symmetry
    rows = load_rows(dataset)
    catalog_map = load_catalog(catalog)
    train_rows, val_rows, test_rows = split_rows(rows)
    if len(train_rows) < 20:
        raise RuntimeError("train split too small")
    if not val_rows:
        val_rows = train_rows[-max(5, len(train_rows) // 10) :]

    init_bundle: dict[str, Any] | None = None
    if init_path is not None:
        if not Path(init_path).is_file():
            raise FileNotFoundError(f"--init not found: {init_path}")
        init_bundle = load_checkpoint(init_path)
        vocab = list(init_bundle["suggestion_ids"])
        feat_mu = init_bundle["feat_mu"]
        feat_sd = init_bundle["feat_sd"]
        missing = [s for s in vocab if s not in catalog_map]
        if missing:
            raise RuntimeError(f"init vocab IDs missing from catalog: {missing[:5]}")
    else:
        vocab = build_vocab(train_rows, catalog_map)
        if not vocab:
            raise RuntimeError("empty vocabulary")
        feat_mu, feat_sd = fit_mu_sd(train_rows)

    x_train, y_train = build_xy(train_rows, vocab, feat_mu, feat_sd)
    x_val, y_val = build_xy(val_rows, vocab, feat_mu, feat_sd)
    _, rel_train = build_x_rel(train_rows, vocab, feat_mu, feat_sd)
    x_val2, rel_val = build_x_rel(val_rows, vocab, feat_mu, feat_sd)
    assert np.allclose(x_val, x_val2)

    x_test = y_test = rel_test = None
    if test_rows:
        x_test, y_test = build_xy(test_rows, vocab, feat_mu, feat_sd)
        _, rel_test = build_x_rel(test_rows, vocab, feat_mu, feat_sd)

    w_train = row_trust_weights(train_rows)
    return {
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "vocab": vocab,
        "feat_mu": feat_mu,
        "feat_sd": feat_sd,
        "x_train": x_train,
        "y_train": y_train,
        "rel_train": rel_train,
        "x_val": x_val,
        "y_val": y_val,
        "rel_val": rel_val,
        "x_test": x_test,
        "y_test": y_test,
        "rel_test": rel_test,
        "w_train": w_train,
        "catalog": catalog_map,
        "init_bundle": init_bundle,
        "in_dim": IN_DIM,
    }


@torch.no_grad()
def torch_logits(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    """Batch forward → (N, K) float32 logits on CPU."""
    model.eval()
    out = model(torch.from_numpy(np.asarray(x, dtype=np.float32)))
    return out.detach().cpu().numpy().astype(np.float32)


def finalize_bakeoff(
    *,
    tag: str,
    category: str,
    scores_val: np.ndarray,
    rel_val: np.ndarray,
    scores_test: np.ndarray | None,
    rel_test: np.ndarray | None,
    train_seconds: float,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x_infer: np.ndarray,
    model: Any,
    metrics_out: Path,
    artifact: Path | str,
    seed: int = SEED,
    k: int = TOP_K,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Eval val/test, time inference, merge metrics.json, print summary."""
    val_ndcg, val_map = eval_scores(scores_val, rel_val, k=k)
    test_ndcg = test_map = None
    if scores_test is not None and rel_test is not None and len(scores_test):
        test_ndcg, test_map = eval_scores(scores_test, rel_test, k=k)
    infer_ms = time_infer_ms(predict_fn, x_infer)
    n_params = count_params(model)
    entry = make_entry(
        tag=tag,
        category=category,
        val_ndcg=val_ndcg,
        val_map=val_map,
        test_ndcg=test_ndcg,
        test_map=test_map,
        train_seconds=train_seconds,
        infer_ms_per_sample=infer_ms,
        n_params=n_params,
        seed=seed,
        artifact=str(artifact),
        k=k,
        extra=extra,
    )
    write_metrics_entry(metrics_out, entry)
    parts = [
        f"tag={tag}",
        f"val_ndcg@{k}={val_ndcg:.4f}",
        f"val_map@{k}={val_map:.4f}",
    ]
    if test_ndcg is not None and test_map is not None:
        parts += [f"test_ndcg@{k}={test_ndcg:.4f}", f"test_map@{k}={test_map:.4f}"]
    parts += [
        f"train_s={train_seconds:.2f}",
        f"infer_ms={infer_ms:.4f}",
        f"n_params={n_params}",
    ]
    print("  ".join(parts))
    print(f"Wrote metrics → {metrics_out}")
    return entry


def save_sklearn_bundle(
    path: Path | str,
    estimator: Any,
    feat_mu: np.ndarray,
    feat_sd: np.ndarray,
    suggestion_ids: Sequence[str],
) -> None:
    """Persist classical bakeoff model + encoding metadata via joblib."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": estimator,
            "feat_mu": np.asarray(feat_mu, dtype=np.float32),
            "feat_sd": np.asarray(feat_sd, dtype=np.float32),
            "suggestion_ids": list(suggestion_ids),
            "in_dim": IN_DIM,
        },
        path,
    )


def load_sklearn_bundle(path: Path | str) -> dict[str, Any]:
    """Load classical bakeoff joblib bundle."""
    import joblib

    bundle = joblib.load(path)
    return {
        "model": bundle["model"],
        "feat_mu": np.asarray(bundle["feat_mu"], dtype=np.float32).reshape(-1),
        "feat_sd": np.asarray(bundle["feat_sd"], dtype=np.float32).reshape(-1),
        "suggestion_ids": list(bundle["suggestion_ids"]),
        "in_dim": int(bundle.get("in_dim", IN_DIM)),
    }


def ovr_predict_scores(estimator: Any, x: np.ndarray) -> np.ndarray:
    """Stack OneVsRest / MultiOutput positive-class probabilities → (N, K)."""
    # OneVsRestClassifier.predict_proba → (N, K) when multilabel binary
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x)
        if isinstance(proba, list):
            cols = []
            for p in proba:
                p = np.asarray(p)
                if p.ndim == 2 and p.shape[1] >= 2:
                    cols.append(p[:, 1])
                elif p.ndim == 2 and p.shape[1] == 1:
                    cols.append(p[:, 0])
                else:
                    cols.append(p.reshape(-1))
            return np.stack(cols, axis=1).astype(np.float32)
        proba = np.asarray(proba, dtype=np.float32)
        if proba.ndim == 3:
            # (K, N, 2) style
            return proba[:, :, 1].T.astype(np.float32)
        return proba.astype(np.float32)
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(x), dtype=np.float32)
    return np.asarray(estimator.predict(x), dtype=np.float32)
