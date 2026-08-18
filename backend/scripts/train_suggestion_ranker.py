#!/usr/bin/env python3
"""Train suggestion_ranker.pt from Dataset C (Phase 10).

Usage (from repo root):

  python backend/scripts/run_dataset_quality_gates.py
  python backend/scripts/train_suggestion_ranker.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from geometry import FEATURE_COLS  # noqa: E402
from suggestion_model import (  # noqa: E402
    IN_DIM,
    build_mlp,
    encode_dataset_row,
    multilabel_target,
    parse_id_list,
    save_checkpoint,
)
from suggestion_label_trust import count_label_methods, row_trust_weights  # noqa: E402
from suggestion_rules import load_catalog  # noqa: E402


def _load_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        return df.astype(object).where(pd.notnull(df), "").to_dict(orient="records")
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def _fit_mu_sd(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(r[c]) for c in FEATURE_COLS] for r in rows], dtype=np.float32)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    return mu, sd


def _build_matrices(
    rows: list[dict[str, str]],
    vocab: list[str],
    feat_mu: np.ndarray,
    feat_sd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for r in rows:
        xs.append(encode_dataset_row(r, feat_mu, feat_sd)[0])
        ids = parse_id_list(r.get("priority_order") or r.get("suggestion_ids") or "")
        ys.append(multilabel_target(ids, vocab))
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


@torch.no_grad()
def _val_bce(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        xb, yb = batch[0], batch[1]
        logits = model(xb)
        # criterion may be reduction='none' — mean over labels then batch
        per = criterion(logits, yb)
        if per.ndim > 0:
            loss = per.mean()
        else:
            loss = per
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def _map_at_k(model: nn.Module, x: np.ndarray, y: np.ndarray, k: int = 4) -> float:
    model.eval()
    logits = model(torch.from_numpy(x)).cpu().numpy()
    scores = []
    for i in range(len(x)):
        probs = 1.0 / (1.0 + np.exp(-logits[i]))
        top = np.argsort(-probs)[:k]
        hits = y[i][top]
        if hits.sum() == 0:
            scores.append(0.0)
            continue
        precisions = []
        hit_count = 0
        for rank, h in enumerate(hits, start=1):
            if h > 0.5:
                hit_count += 1
                precisions.append(hit_count / rank)
        scores.append(float(np.mean(precisions)) if precisions else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train suggestion ranker MLP.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "suggestion_dataset.parquet",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=REPO_ROOT / "data" / "catalogs" / "suggestions.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "backend" / "models" / "suggestion_ranker.pt",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dataset.is_file():
        # fallback csv
        csv_path = args.dataset.with_suffix(".csv")
        if csv_path.is_file():
            args.dataset = csv_path
        else:
            print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
            return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows = _load_rows(args.dataset)
    catalog = load_catalog(args.catalog)
    train_rows = [r for r in rows if r.get("split") == "train"]
    val_rows = [r for r in rows if r.get("split") == "val"]
    test_rows = [r for r in rows if r.get("split") == "test"]
    if len(train_rows) < 20:
        print("ERROR: train split too small", file=sys.stderr)
        return 1
    if not val_rows:
        val_rows = train_rows[-max(5, len(train_rows) // 10) :]

    # Vocabulary = IDs with ≥1 train positive ∩ active catalog
    train_counts: Counter[str] = Counter()
    for r in train_rows:
        for sid in parse_id_list(r.get("priority_order") or r.get("suggestion_ids") or ""):
            if sid in catalog:
                train_counts[sid] += 1
    vocab = sorted(train_counts.keys())
    if not vocab:
        print("ERROR: empty vocabulary", file=sys.stderr)
        return 1
    unused = sorted(set(catalog.keys()) - set(vocab))
    print(f"Vocabulary K={len(vocab)} (unused catalog IDs={len(unused)})")
    print("Train ID counts (top 15):", train_counts.most_common(15))

    feat_mu, feat_sd = _fit_mu_sd(train_rows)
    x_train, y_train = _build_matrices(train_rows, vocab, feat_mu, feat_sd)
    x_val, y_val = _build_matrices(val_rows, vocab, feat_mu, feat_sd)
    w_train = row_trust_weights(train_rows)
    method_counts = count_label_methods(train_rows)
    print(f"Train label_method counts: {method_counts}")
    print(f"Train human_v1 rows: {method_counts.get('human_v1', 0)}  weight_sum={float(w_train.sum()):.1f}")
    assert x_train.shape[1] == IN_DIM

    # pos_weight = (N - pos) / pos
    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos
    pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0).astype(np.float32)

    device = torch.device("cpu")
    model = build_mlp(IN_DIM, len(vocab)).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(pos_weight),
        reduction="none",
    )
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(w_train),
    )
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for xb, yb, wb in train_loader:
            optim.zero_grad()
            logits = model(xb)
            per_elem = criterion(logits, yb)  # (B, K)
            per_sample = per_elem.mean(dim=1)  # (B,)
            loss = (per_sample * wb).sum() / wb.sum().clamp_min(1e-6)
            loss.backward()
            optim.step()
            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        train_loss = running / max(n, 1)
        val_loss = _val_bce(model, val_loader, criterion)
        map4 = _map_at_k(model, x_val, y_val, k=4)
        print(
            f"epoch {epoch:03d}  train_bce={train_loss:.4f}  val_bce={val_loss:.4f}  val_map@4={map4:.4f}"
        )
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if test_rows:
        x_test, y_test = _build_matrices(test_rows, vocab, feat_mu, feat_sd)
        test_map = _map_at_k(model, x_test, y_test, k=4)
        print(f"test_map@4={test_map:.4f}  n_test={len(test_rows)}")

    save_checkpoint(args.out, model, feat_mu, feat_sd, vocab)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
