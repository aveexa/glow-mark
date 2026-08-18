#!/usr/bin/env python3
"""Bakeoff: LightGBM multi-label / GBDT tabular baseline.

Category: classical_gbdt

Usage (from repo root):

  python backend/scripts/bakeoff/train_lightgbm.py

Note: macOS may need `brew install libomp` for lightgbm native lib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.multioutput import MultiOutputClassifier

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

TAG = "lightgbm"
CATEGORY = "classical_gbdt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Bakeoff train: LightGBM GBDT.")
    add_bakeoff_args(parser)
    args = parser.parse_args()
    if args.out is None:
        args.out = DEFAULT_MODELS_DIR / f"{TAG}.joblib"

    try:
        from lightgbm import LGBMClassifier
    except OSError as exc:
        print(
            "ERROR: lightgbm failed to load native library "
            "(macOS usually needs: brew install libomp).\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 1
    except ImportError as exc:
        print(f"ERROR: lightgbm not installed: {exc}", file=sys.stderr)
        return 1

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

    y_bin = (y_train > 0.5).astype(np.int32)
    base = LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.05,
        random_state=args.seed,
        verbosity=-1,
    )
    clf = MultiOutputClassifier(base, n_jobs=1)

    def train_fn():
        # Pass feature names to avoid sklearn/LGBM warning on predict
        import pandas as pd

        cols = [f"f{i}" for i in range(x_train.shape[1])]
        clf.fit(pd.DataFrame(x_train, columns=cols), y_bin)
        return clf

    _, train_seconds = time_train(train_fn)

    import pandas as pd

    cols = [f"f{i}" for i in range(x_val.shape[1])]

    def predict_scores(x: np.ndarray) -> np.ndarray:
        return ovr_predict_scores(clf, pd.DataFrame(x, columns=cols))

    scores_val = predict_scores(x_val)
    scores_test = None
    rel_test = None
    if data["x_test"] is not None:
        scores_test = predict_scores(data["x_test"])
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
        predict_fn=predict_scores,
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
