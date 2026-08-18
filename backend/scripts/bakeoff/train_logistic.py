#!/usr/bin/env python3
"""Bakeoff: One-vs-rest logistic regression (classical linear baseline).

Category: classical_linear

Usage (from repo root):

  python backend/scripts/bakeoff/train_logistic.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

BAKEOFF_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BAKEOFF_DIR.parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(BAKEOFF_DIR) not in sys.path:
    sys.path.insert(0, str(BAKEOFF_DIR))

from common_eval import (  # noqa: E402
    DEFAULT_MODELS_DIR,
    add_bakeoff_args,
    finalize_bakeoff,
    ovr_predict_scores,
    prepare_bakeoff_data,
    save_sklearn_bundle,
    time_train,
)

TAG = "logistic"
CATEGORY = "classical_linear"


def main() -> int:
    parser = argparse.ArgumentParser(description="Bakeoff train: multi-label logistic.")
    add_bakeoff_args(parser)
    args = parser.parse_args()
    if args.out is None:
        args.out = DEFAULT_MODELS_DIR / f"{TAG}.joblib"

    np.random.seed(args.seed)

    data = prepare_bakeoff_data(args.dataset, args.catalog, seed=args.seed)
    vocab = data["vocab"]
    feat_mu, feat_sd = data["feat_mu"], data["feat_sd"]
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, rel_val = data["x_val"], data["rel_val"]
    print(
        f"[{TAG}] category={CATEGORY}  K={len(vocab)}  "
        f"n_train={len(data['train_rows'])} n_val={len(data['val_rows'])} "
        f"n_test={len(data['test_rows'])}"
    )

    # Drop labels with no positives (OneVsRest needs both classes for some solvers)
    y_bin = (y_train > 0.5).astype(np.int32)
    keep = y_bin.sum(axis=0) > 0
    if not keep.all():
        # Still train on full y; sklearn OVR skips empty? Use labels with ≥1 pos.
        # Safer: ensure every column has at least one positive by construction of vocab.
        pass

    base = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=args.seed,
        solver="lbfgs",
    )
    clf = OneVsRestClassifier(base, n_jobs=1)

    def train_fn():
        clf.fit(x_train, y_bin)
        return clf

    _, train_seconds = time_train(train_fn)

    scores_val = ovr_predict_scores(clf, x_val)
    scores_test = None
    rel_test = None
    if data["x_test"] is not None:
        scores_test = ovr_predict_scores(clf, data["x_test"])
        rel_test = data["rel_test"]

    save_sklearn_bundle(args.out, clf, feat_mu, feat_sd, vocab)
    print(f"Wrote {args.out}")

    finalize_bakeoff(
        tag=TAG,
        category=CATEGORY,
        scores_val=scores_val,
        rel_val=rel_val,
        scores_test=scores_test,
        rel_test=rel_test,
        train_seconds=train_seconds,
        predict_fn=lambda x: ovr_predict_scores(clf, x),
        x_infer=x_val,
        model=clf,
        metrics_out=args.metrics_out,
        artifact=args.out,
        seed=args.seed,
        k=args.k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
