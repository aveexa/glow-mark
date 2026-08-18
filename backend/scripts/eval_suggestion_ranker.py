#!/usr/bin/env python3
"""Evaluate suggestion ranker checkpoint on Dataset C (NDCG@4 / MAP@4).

Writes/merges results into data/processed/ranker_bakeoff_metrics.json.

Usage (from repo root):

  python backend/scripts/eval_suggestion_ranker.py \\
    --ckpt backend/models/suggestion_ranker_bce_v1.pt --tag bce_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from suggestion_model import encode_dataset_row, load_checkpoint, parse_id_list  # noqa: E402
from suggestion_rl import TOP_K, eval_ndcg_map, gold_relevance  # noqa: E402


def _load_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        return df.astype(object).where(pd.notnull(df), "").to_dict(orient="records")
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: ("" if v is None else str(v)) for k, v in r.items()} for r in csv.DictReader(f)]


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
    parser = argparse.ArgumentParser(description="Eval suggestion ranker NDCG@4 / MAP@4.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tag", type=str, required=True, help="Bakeoff tag, e.g. bce_v1 / rl_v1")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "suggestion_dataset.parquet",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "ranker_bakeoff_metrics.json",
    )
    parser.add_argument("--k", type=int, default=TOP_K)
    args = parser.parse_args()

    if not args.ckpt.is_file():
        print(f"ERROR: checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 1

    if not args.dataset.is_file():
        csv_path = args.dataset.with_suffix(".csv")
        if csv_path.is_file():
            args.dataset = csv_path
        else:
            print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
            return 1

    bundle = load_checkpoint(args.ckpt)
    vocab = list(bundle["suggestion_ids"])
    feat_mu = bundle["feat_mu"]
    feat_sd = bundle["feat_sd"]
    model = bundle["model"]

    rows = _load_rows(args.dataset)
    val_rows = [r for r in rows if r.get("split") == "val"]
    test_rows = [r for r in rows if r.get("split") == "test"]
    if not val_rows:
        print("ERROR: empty val split", file=sys.stderr)
        return 1

    x_val, rel_val = _build_matrices(val_rows, vocab, feat_mu, feat_sd)
    val_ndcg, val_map = _eval_split(model, x_val, rel_val, k=args.k)

    entry: dict = {
        "tag": args.tag,
        "ckpt": str(args.ckpt),
        "k": args.k,
        "vocab_k": len(vocab),
        "n_val": len(val_rows),
        f"val_ndcg@{args.k}": round(val_ndcg, 6),
        f"val_map@{args.k}": round(val_map, 6),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    if test_rows:
        x_test, rel_test = _build_matrices(test_rows, vocab, feat_mu, feat_sd)
        test_ndcg, test_map = _eval_split(model, x_test, rel_test, k=args.k)
        entry["n_test"] = len(test_rows)
        entry[f"test_ndcg@{args.k}"] = round(test_ndcg, 6)
        entry[f"test_map@{args.k}"] = round(test_map, 6)
    else:
        test_ndcg = test_map = None

    print(
        f"tag={args.tag}  ckpt={args.ckpt.name}  K={len(vocab)}  "
        f"val_ndcg@{args.k}={val_ndcg:.4f}  val_map@{args.k}={val_map:.4f}"
        + (
            f"  test_ndcg@{args.k}={test_ndcg:.4f}  test_map@{args.k}={test_map:.4f}"
            if test_ndcg is not None
            else ""
        )
    )

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"models": {}}
    if args.metrics_out.is_file():
        try:
            payload = json.loads(args.metrics_out.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                payload = {"models": {}}
            payload.setdefault("models", {})
        except json.JSONDecodeError:
            payload = {"models": {}}

    payload["models"][args.tag] = entry
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    args.metrics_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
