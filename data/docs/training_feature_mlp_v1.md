# Training: Feature MLP v1 (geometry recommendation classifier)

**Status:** Shipped checkpoint in use for `/analyze`  
**Checkpoint:** `backend/models/reco_geometry_model.pt`  
**Serve code:** `backend/inference.py` → `_mlp_feature` / feature branch of `analyze_image_bytes`  
**Feature contract:** [`feature_contract_v1.md`](feature_contract_v1.md) (`FEATURE_CONTRACT_VERSION = "v1"`)

This document describes the **algorithm**, **training pipeline**, and **reported validation metric** for the Feature MLP — the model that turns 24 facial geometry floats into per-feature `{low, ok, high}` classes.

---

## Purpose

The Feature MLP does **not** predict a beauty score and does **not** generate advice text.

It classifies each of 24 geometry channels as:

| Class | Meaning (relative to the training population) |
|-------|-----------------------------------------------|
| `low` | Feature value below the train-split 20th percentile |
| `ok` | Feature value between the 20th and 80th percentiles |
| `high` | Feature value above the train-split 80th percentile |

At serve time, non-`ok` classes become short strings in `recommendations` / structured rows in `recommendation_items`.

---

## Algorithm (for supervisors)

| Question | Answer |
|----------|--------|
| **What algorithm?** | **Multilayer Perceptron (MLP)** — a feedforward neural network |
| **Is it classification?** | **Yes** — multi-head multi-class classification |
| **Not used** | Random Forest, SVM, decision trees, CNNs on pixels, free-text LLMs |
| **Loss** | Categorical **Cross-Entropy** (one CE head per geometry feature, summed) |
| **Optimizer** | **Adam** |

In short: a small **MLP classifier** maps standardized geometry features to discrete `low` / `ok` / `high` labels.

---

## Problem formulation

| | |
|--|--|
| **Input `X`** | 24 geometry floats (`FEATURE_COLS` order), z-scored with train μ / σ |
| **Output `Y`** | 72 logits = **24 features × 3 classes** |
| **Decode** | Softmax **per feature** (shape `(24, 3)`), then argmax → class + confidence |

Architecture (must match serve):

```text
Linear(24 → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → 72)
```

Defined in code as `_mlp_feature` in `backend/inference.py`. No Dropout in the feature network (unlike the beauty MLP).

---

## Labeling method

Labels are **not** human beauty ratings. They are built automatically from the population distribution:

1. Split samples by stable hash of `sample_id` → train / val / test (80 / 10 / 10).
2. On the **train split only**, for each of the 24 features compute **p20** and **p80**.
3. Assign:

```text
value < p20  → low
p20 ≤ value ≤ p80 → ok
value > p80  → high
```

Label method string: `percentile_p20_p80`. Expected train class rates ≈ **20% / 60% / 20%**.

This is Dataset B (`geometry_dataset`) under the frozen schema in [`dataset_schema_v1.md`](dataset_schema_v1.md).

---

## Data and preprocess pipeline

```text
FFHQ face images
  → MediaPipe FaceMesh (468 landmarks, static_image_mode, refine_landmarks=False)
  → Pose gate (|yaw| ≤ 15°, |pitch| ≤ 15°)
  → Extract 24 geometry floats (feature contract v1)
  → Percentile labels (p20/p80 on train)
  → Train Feature MLP
  → Export reco_geometry_model.pt
```

| Item | Spec |
|------|------|
| **Source** | FFHQ (checkpoint meta: `source = "ffhq"`) |
| **Contract** | Feature contract **v1** (same formulas as `backend/geometry.py`) |
| **Training scripts** | Colab notebooks (preprocess + train); not checked into this repo yet |

Rejected faces (no face, multiple faces, non-frontal pose) are excluded before training.

---

## Training recipe

| Hyperparameter | Value used |
|----------------|------------|
| Standardization | Train-split μ / σ only (stored in checkpoint as `mu`, `sd`) |
| Loss | Sum of 24 `CrossEntropyLoss` terms (class-weighted for imbalance) |
| Optimizer | Adam, learning rate `1e-3`, weight decay `1e-4` |
| Batch size | 64 |
| Epochs | Up to 80 with early stopping |
| Early stop | Patience 12 on **validation macro-F1** |
| Class map | `low=0`, `ok=1`, `high=2` |

### Checkpoint keys (serve contract)

```text
feat_cols, label_cols, mu, sd, state,
feature_contract_version, n_classes, class_names,
best_val_macro_f1, source
```

Note: weights are under the key **`state`** (not `model_state`).

---

## Results

Metric stored in the shipped checkpoint (`backend/models/reco_geometry_model.pt`):

| Metric | Value | Notes |
|--------|-------|--------|
| **Best validation macro-F1** | **0.9347 (~93.5%)** | Early-stopping criterion; average F1 across `low` / `ok` / `high` over all label heads |
| Feature contract | `v1` | |
| Classes | 3 (`low`, `ok`, `high`) | |
| Source | `ffhq` | |

**How to describe this to a supervisor**

- Prefer: *“Validation macro-F1 ≈ 93.5% on the held-out FFHQ geometry split.”*
- Avoid claiming *“93.5% accuracy”* unless a separate overall accuracy number was recorded in Colab; the checkpoint stores **macro-F1**, not raw accuracy.
- Test-split accuracy / F1 were printed during Colab training but are **not** stored inside the `.pt` file.

Macro-F1 is appropriate here because `ok` is more frequent (~60%) than `low`/`high` (~20% each); it does not let the majority class dominate the score.

---

## How `/analyze` uses the model

1. Image → MediaPipe 468 → pose gate → 24 geometry floats.
2. Z-score with checkpoint `mu` / `sd`.
3. Feature MLP → 72 logits → per-feature softmax → argmax class.
4. API fields:
   - `recommendation_items`: all 24 `{label, class, confidence}`
   - `recommendations`: strings only where `class != "ok"`

---

## Distinction from the other two MLPs

| Model | Checkpoint | Algorithm / task | Output |
|-------|------------|------------------|--------|
| **Beauty MLP** | `beauty_landmarks_best.pt` | MLP **regression** from 68-pt landmarks | Scalar score 0–100 |
| **Feature MLP** (this doc) | `reco_geometry_model.pt` | MLP **classification** of 24 geometry channels | `low` / `ok` / `high` × 24 |
| **Suggestion ranker** | `suggestion_ranker.pt` | MLP multi-label ranking over catalog IDs | Top-k approved tip texts |

Feature MLP and the suggestion ranker both consume the same 24 geometry floats, but they are **independent** models. At serve time, suggestion class one-hots come from percentile thresholds in `suggestion_mapping_rules.csv`, not from Feature argmax.

---

## Related docs

| Doc | Topic |
|-----|--------|
| [`feature_contract_v1.md`](feature_contract_v1.md) | 24 feature formulas + pose gate |
| [`dataset_schema_v1.md`](dataset_schema_v1.md) | Dataset B column schema |
| [`create_dataset.md`](create_dataset.md) | End-to-end dataset build phases |
| [`model_contract_v1.md`](model_contract_v1.md) | Suggestion ranker contract (separate from Feature) |
| [`training_suggestion_ranker_v1.md`](training_suggestion_ranker_v1.md) | Suggestion ranker training |

---

## Summary for oral / slide use

1. **Algorithm:** Multilayer Perceptron (MLP).  
2. **Type:** Classification (`low` / `ok` / `high` per facial geometry feature).  
3. **Training data:** FFHQ faces → MediaPipe geometry → percentile labels.  
4. **Result:** Best validation **macro-F1 ≈ 93.5%** (checkpoint `best_val_macro_f1 = 0.9347`).
