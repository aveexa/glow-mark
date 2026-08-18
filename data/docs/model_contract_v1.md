# Suggestion model contract v1

**Status:** Locked (Phase 0)  
**Date:** 2026-07-14  
**Feature dependency:** [`feature_contract_v1.md`](feature_contract_v1.md) (`FEATURE_CONTRACT_VERSION = "v1"`)

This document freezes what the beauty-suggestion model learns and how it is served. Do not invent free-text advice at inference time.

---

## Task

Multi-label classification and ranking over a fixed catalog of `suggestion_id`s.

The model does **not** generate natural language. It ranks approved template IDs; the server looks up `approved_text` from the catalog.

---

## Input `X`

| Component | Shape / type | Notes |
|-----------|--------------|--------|
| Geometry floats | 24 | Order = `FEATURE_COLS` in `backend/geometry.py`, standardized with train μ/σ |
| Class one-hots | 72 | `24 × {low, ok, high}` — **included in v1** (`in_dim=96`) |

Float features must be extracted with the same module and contract version used at train time.

---

## Output `Y`

| Field | Type | Notes |
|-------|------|--------|
| Multi-hot over `K` IDs | `float[K]` | Positive = relevant suggestion |
| Serve top-`k` | `k ∈ {3,4,5}` | Default product `k = 4` |
| `priority_order` | ordered IDs | Multi-hot source in v1; ranking loss optional later |

---

## Text at serve

1. Model → logits over catalog IDs.  
2. Take top-`k` IDs (optionally reorder by learned rank).  
3. Lookup only from [`data/catalogs/suggestions.csv`](../catalogs/suggestions.csv).  
4. Never emit model-generated prose; never invent new IDs.

Inactive (`active=false`) catalog rows must not appear in the output vocabulary at train or serve.

---

## Checkpoint contract

Artifact: `backend/models/suggestion_ranker.pt`

| Key | Type | Meaning |
|-----|------|---------|
| `feat_mu` | array | Train-split mean over the 24 floats |
| `feat_sd` | array | Train-split std over the 24 floats |
| `suggestion_ids` | `list[str]` | Ordered vocabulary (length `K`) |
| `state` | `state_dict` | MLP weights |
| `feature_contract_version` | `str` | Must be `"v1"` for this contract |
| `use_class_onehots` | `bool` | `true` for v1 |
| `in_dim` | `int` | `96` |

Architecture: `Linear(96→128)→ReLU→Dropout(0.2)→Linear(128→128)→ReLU→Dropout(0.2)→Linear(128→K)`.

See [`training_suggestion_ranker_v1.md`](training_suggestion_ranker_v1.md).

---

## Loss

- Primary: `BCEWithLogitsLoss` on multi-hot ID targets (`pos_weight` for rare IDs).  
- Optional later: listwise / pairwise ranking loss on `priority_order`.

---

## Safety

Copy in the catalog must be **cosmetic / photographic / styling** only:

- Allowed: makeup, lighting, hair framing, pose retake, skincare-routine style tips that are non-clinical.
- Forbidden: medical diagnoses, surgical recommendations, guarantees of attractiveness, clinical procedure language.

Any new catalog text requires product/legal review before `active=true`.

---

## Serve path (target)

```text
image → MediaPipe 468 → pose gate → 24 geometry floats
      → (optional low/ok/high) → suggestion MLP → top-k IDs
      → catalog lookup → API suggestions[]
```

Beauty MLP and Feature MLP are separate checkpoints. This contract does not require them for Dataset C labeling (classes may come from percentile rules).

---

## Done criteria

Anyone can read this file and know exact `X`, `Y`, checkpoint keys, and the serve path without opening Python.
