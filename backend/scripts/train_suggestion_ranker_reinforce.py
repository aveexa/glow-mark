#!/usr/bin/env python3
"""Train suggestion ranker with Offline REINFORCE (NDCG@4).

Usage (from repo root):

  python backend/scripts/train_suggestion_ranker.py   # optional BCE warm-start
  python backend/scripts/train_suggestion_ranker_reinforce.py \\
      --init backend/models/suggestion_ranker.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
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
    load_checkpoint,
    parse_id_list,
    save_checkpoint,
)
from suggestion_rl import (  # noqa: E402
    TOP_K,
    batch_ndcg_rewards,
    eval_ndcg_map,
    gold_relevance,
    plackett_luce_sample,
    reinforce_loss,
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


def _priority_ids(row: dict[str, str]) -> list[str]:
    return parse_id_list(row.get("priority_order") or row.get("suggestion_ids") or "")


def _build_matrices(
    rows: list[dict[str, str]],
    vocab: list[str],
    feat_mu: np.ndarray,
    feat_sd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    rels = []
    for r in rows:
        xs.append(encode_dataset_row(r, feat_mu, feat_sd)[0])
        rels.append(gold_relevance(_priority_ids(r), vocab).numpy())
    return np.stack(xs).astype(np.float32), np.stack(rels).astype(np.float32)


@torch.no_grad()
def _eval_split(model: torch.nn.Module, x: np.ndarray, rel: np.ndarray, k: int) -> tuple[float, float]:
    model.eval()
    logits = model(torch.from_numpy(x))
    return eval_ndcg_map(logits, torch.from_numpy(rel), k=k)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train suggestion ranker with Offline REINFORCE.")
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
        default=REPO_ROOT / "backend" / "models" / "suggestion_ranker_rl.pt",
    )
    parser.add_argument(
        "--init",
        type=Path,
        default=None,
        help="Optional BCE / prior checkpoint to warm-start weights and vocab.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=TOP_K)
    args = parser.parse_args()

    if not args.dataset.is_file():
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

    init_bundle = None
    if args.init is not None:
        if not args.init.is_file():
            print(f"ERROR: --init not found: {args.init}", file=sys.stderr)
            return 1
        init_bundle = load_checkpoint(args.init)
        vocab = list(init_bundle["suggestion_ids"])
        feat_mu = init_bundle["feat_mu"]
        feat_sd = init_bundle["feat_sd"]
        # Require vocab IDs exist in catalog (active set)
        missing = [s for s in vocab if s not in catalog]
        if missing:
            print(f"ERROR: init vocab IDs missing from catalog: {missing[:5]}", file=sys.stderr)
            return 1
        print(f"Warm-start from {args.init}  K={len(vocab)}")
    else:
        train_counts: Counter[str] = Counter()
        for r in train_rows:
            for sid in _priority_ids(r):
                if sid in catalog:
                    train_counts[sid] += 1
        vocab = sorted(train_counts.keys())
        if not vocab:
            print("ERROR: empty vocabulary", file=sys.stderr)
            return 1
        feat_mu, feat_sd = _fit_mu_sd(train_rows)
        print(f"Vocabulary K={len(vocab)} (from train positives)")
        print("Train ID counts (top 15):", train_counts.most_common(15))

    x_train, rel_train = _build_matrices(train_rows, vocab, feat_mu, feat_sd)
    x_val, rel_val = _build_matrices(val_rows, vocab, feat_mu, feat_sd)
    w_train = row_trust_weights(train_rows)
    method_counts = count_label_methods(train_rows)
    print(f"Train label_method counts: {method_counts}")
    print(
        f"Train human_v1 rows: {method_counts.get('human_v1', 0)}  "
        f"weight_sum={float(w_train.sum()):.1f}"
    )
    assert x_train.shape[1] == IN_DIM

    device = torch.device("cpu")
    model = build_mlp(IN_DIM, len(vocab)).to(device)
    if init_bundle is not None:
        model.load_state_dict(init_bundle["model"].state_dict())
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(rel_train),
        torch.from_numpy(w_train),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    best_val = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_reward = 0.0
        n = 0
        for xb, relb, wb in train_loader:
            optim.zero_grad()
            logits = model(xb)
            indices, log_prob, entropy = plackett_luce_sample(logits, k=args.k)
            rewards = batch_ndcg_rewards(indices, relb, k=args.k)
            loss, stats = reinforce_loss(
                log_prob,
                rewards,
                entropy,
                entropy_coef=args.entropy_coef,
                sample_weights=wb,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            bs = xb.size(0)
            running_loss += stats["loss"] * bs
            running_reward += stats["reward_mean"] * bs
            n += bs

        train_loss = running_loss / max(n, 1)
        train_r = running_reward / max(n, 1)
        val_ndcg, val_map = _eval_split(model, x_val, rel_val, k=args.k)
        print(
            f"epoch {epoch:03d}  train_loss={train_loss:.4f}  train_R={train_r:.4f}  "
            f"val_ndcg@{args.k}={val_ndcg:.4f}  val_map@{args.k}={val_map:.4f}"
        )
        if val_ndcg > best_val + 1e-4:
            best_val = val_ndcg
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
        x_test, rel_test = _build_matrices(test_rows, vocab, feat_mu, feat_sd)
        test_ndcg, test_map = _eval_split(model, x_test, rel_test, k=args.k)
        print(
            f"test_ndcg@{args.k}={test_ndcg:.4f}  test_map@{args.k}={test_map:.4f}  "
            f"n_test={len(test_rows)}"
        )

    save_checkpoint(args.out, model, feat_mu, feat_sd, vocab)
    print(f"Wrote {args.out}  best_val_ndcg@{args.k}={best_val:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
