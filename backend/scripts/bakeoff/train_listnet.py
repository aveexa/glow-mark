#!/usr/bin/env python3
"""Bakeoff: ListNet (softmax CE vs soft target from graded relevance).

Category: listwise_supervised

Usage (from repo root):

  python backend/scripts/bakeoff/train_listnet.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

BAKEOFF_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BAKEOFF_DIR.parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(BAKEOFF_DIR) not in sys.path:
    sys.path.insert(0, str(BAKEOFF_DIR))

from common_eval import (  # noqa: E402
    DEFAULT_MODELS_DIR,
    add_bakeoff_args,
    eval_scores,
    finalize_bakeoff,
    prepare_bakeoff_data,
    time_train,
    torch_logits,
)
from suggestion_model import IN_DIM, build_mlp, save_checkpoint  # noqa: E402

TAG = "listnet"
CATEGORY = "listwise_supervised"


def listnet_loss(
    logits: torch.Tensor,
    rel: torch.Tensor,
    w: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    p = F.softmax(rel / temperature, dim=-1)
    log_q = F.log_softmax(logits, dim=-1)
    per = -(p * log_q).sum(dim=-1)
    return (per * w).sum() / w.sum().clamp_min(1e-6)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bakeoff train: ListNet listwise CE.")
    add_bakeoff_args(parser)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    if args.out is None:
        args.out = DEFAULT_MODELS_DIR / f"{TAG}.pt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = prepare_bakeoff_data(args.dataset, args.catalog, seed=args.seed)
    vocab = data["vocab"]
    feat_mu, feat_sd = data["feat_mu"], data["feat_sd"]
    x_train, rel_train = data["x_train"], data["rel_train"]
    x_val, rel_val = data["x_val"], data["rel_val"]
    w_train = data["w_train"]
    print(
        f"[{TAG}] category={CATEGORY}  K={len(vocab)}  "
        f"n_train={len(data['train_rows'])} n_val={len(data['val_rows'])} "
        f"n_test={len(data['test_rows'])}"
    )

    device = torch.device("cpu")
    model = build_mlp(IN_DIM, len(vocab)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(rel_train),
            torch.from_numpy(w_train),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    def train_fn() -> None:
        best_val = -1.0
        best_state = None
        bad_epochs = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            n = 0
            for xb, relb, wb in train_loader:
                optim.zero_grad()
                logits = model(xb)
                loss = listnet_loss(logits, relb, wb, args.temperature)
                loss.backward()
                optim.step()
                running += float(loss.item()) * xb.size(0)
                n += xb.size(0)
            train_loss = running / max(n, 1)
            scores = torch_logits(model, x_val)
            val_ndcg, val_map = eval_scores(scores, rel_val, k=args.k)
            print(
                f"epoch {epoch:03d}  train_listnet={train_loss:.4f}  "
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

    _, train_seconds = time_train(train_fn)

    scores_val = torch_logits(model, x_val)
    scores_test = None
    rel_test = None
    if data["x_test"] is not None:
        scores_test = torch_logits(model, data["x_test"])
        rel_test = data["rel_test"]

    save_checkpoint(args.out, model, feat_mu, feat_sd, vocab)
    print(f"Wrote {args.out}")

    finalize_bakeoff(
        tag=TAG,
        category=CATEGORY,
        scores_val=scores_val,
        rel_val=rel_val,
        scores_test=scores_test,
        rel_test=rel_test,
        train_seconds=train_seconds,
        predict_fn=lambda x: torch_logits(model, x),
        x_infer=x_val,
        model=model,
        metrics_out=args.metrics_out,
        artifact=args.out,
        seed=args.seed,
        k=args.k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
