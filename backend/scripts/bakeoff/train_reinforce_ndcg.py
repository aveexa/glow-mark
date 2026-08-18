#!/usr/bin/env python3
"""Bakeoff: Offline REINFORCE with NDCG@4 reward (listwise RL).

Category: listwise_rl

Usage (from repo root):

  python backend/scripts/bakeoff/train_reinforce_ndcg.py --init backend/models/bakeoff/bce.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
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
from suggestion_rl import (  # noqa: E402
    batch_ndcg_rewards,
    plackett_luce_sample,
    reinforce_loss,
)

TAG = "reinforce_ndcg"
CATEGORY = "listwise_rl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Bakeoff train: REINFORCE + NDCG@4.")
    add_bakeoff_args(parser)
    parser.add_argument(
        "--init",
        type=Path,
        default=DEFAULT_MODELS_DIR / "bce.pt",
        help="BCE checkpoint to warm-start (Stage B).",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    if args.out is None:
        args.out = DEFAULT_MODELS_DIR / f"{TAG}.pt"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    init_path = args.init if args.init.is_file() else None
    if args.init and not args.init.is_file():
        print(f"WARNING: --init not found ({args.init}); training from scratch", file=sys.stderr)

    data = prepare_bakeoff_data(
        args.dataset, args.catalog, seed=args.seed, init_path=init_path
    )
    vocab = data["vocab"]
    feat_mu, feat_sd = data["feat_mu"], data["feat_sd"]
    x_train, rel_train = data["x_train"], data["rel_train"]
    x_val, rel_val = data["x_val"], data["rel_val"]
    w_train = data["w_train"]
    print(
        f"[{TAG}] category={CATEGORY}  K={len(vocab)}  "
        f"n_train={len(data['train_rows'])} n_val={len(data['val_rows'])} "
        f"n_test={len(data['test_rows'])}  init={init_path}"
    )

    device = torch.device("cpu")
    model = build_mlp(IN_DIM, len(vocab)).to(device)
    if data["init_bundle"] is not None:
        model.load_state_dict(data["init_bundle"]["model"].state_dict())
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
            scores = torch_logits(model, x_val)
            val_ndcg, val_map = eval_scores(scores, rel_val, k=args.k)
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
