# Suggestion ranker — Offline REINFORCE contract v1

**Status:** Locked (Phase 1)  
**Date:** 2026-07-15  
**Code:** [`backend/suggestion_rl.py`](../../backend/suggestion_rl.py), [`backend/scripts/train_suggestion_ranker_reinforce.py`](../../backend/scripts/train_suggestion_ranker_reinforce.py)  
**Depends on:** [`model_contract_v1.md`](model_contract_v1.md), [`feature_contract_v1.md`](feature_contract_v1.md), Dataset C schema

This document freezes the Offline REINFORCE ranking problem. Serve path and checkpoint keys stay identical to the BCE ranker; only the training objective changes.

---

## Problem formalization

| Piece | Locked definition |
|-------|-------------------|
| State `s` | 96-d vector: 24 train-standardized geometry floats + 72 class one-hots (`encode_dataset_row` / `encode_features`) |
| Policy `π_θ` | Suggestion MLP → logits over vocab `K` ([`suggestion_model.py`](../../backend/suggestion_model.py)) |
| Action `a` | Ordered list of `k = 4` suggestion IDs |
| Episode | One face → one ranking (1-step contextual bandit) |
| Primary reward `R` | **NDCG@4** vs gold `priority_order` |
| Secondary metric | **MAP@4** (monitor only; not the train loss) |
| Relevance gains | Graded: gold ranks 1..4 → `[3, 2, 1, 1]`; IDs not in gold → `0` |
| Policy sampling | **Plackett–Luce:** sequential softmax sample without replacement for 4 steps |
| Baseline `b` | Batch mean of rewards |
| Entropy | Bonus with default coefficient `0.01` |
| Eval / serve decode | Greedy top-4 by logit rank (same order as sigmoid `decode_top_k`; ranking is monotonic) |

---

## Vocabulary and gold labels

- **Vocab** = active catalog IDs (`suggestions.csv`) with ≥1 train-split positive (same rule as BCE trainer).
- **Gold list** = `priority_order`; if empty, fall back to `suggestion_ids` (pipe-separated).
- Inactive catalog rows never enter the vocabulary.

---

## Relevance and NDCG@4

For gold ordered IDs `(g_1, …, g_m)` truncate/pad at `k = 4`:

```text
rel[g_r] = RELEVANCE_GAINS[r-1]   for r = 1..min(m, 4)
rel[other] = 0
```

where `RELEVANCE_GAINS = (3, 2, 1, 1)`.

For a predicted ordered list `a = (a_1, …, a_k)`:

```text
DCG@k  = Σ_{t=1..k}  rel[a_t] / log2(t + 1)
IDCG@k = DCG@k of gold IDs sorted by relevance (ideal order)
NDCG@k = DCG@k / IDCG@k   (0 if IDCG@k = 0)
```

Reward for REINFORCE: `R = NDCG@4` (scalar in `[0, 1]`).

---

## Plackett–Luce policy

Given logits `z ∈ R^K`:

```text
log π(a|s) = Σ_{t=1..4} log softmax(z masked_t)[a_t]
```

At step `t`, previously chosen indices are masked with `-∞` (sample without replacement).

Entropy bonus uses the average per-step categorical entropy over the four steps (or equivalent batch mean used in code).

---

## Loss

```text
b = mean(R) over the minibatch
L = − mean( (R − b) · log π(a|s) ) − λ · mean(H)
```

with `λ = 0.01` by default. Gradient clip: `max_norm = 1.0`.

---

## Checkpoint and serve

Artifact keys stay exactly as BCE / [`model_contract_v1.md`](model_contract_v1.md):

```text
feat_mu, feat_sd, suggestion_ids, state,
feature_contract_version, use_class_onehots=true, in_dim=96
```

No new serve API fields. [`suggestion_serve.py`](../../backend/suggestion_serve.py) continues to load the same `.pt` and greedy `decode_top_k`.

Warm-start: load BCE `suggestion_ranker.pt` via `--init` when running REINFORCE fine-tune (Stage B). Do not replace production serve weights until val NDCG@4 beats the BCE baseline.

---

## Done criteria

Anyone can read this file and know state, action, reward gains, sampling, loss, and that the serve checkpoint schema is unchanged.
