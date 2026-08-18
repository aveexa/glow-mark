# Training: suggestion ranker v1

**Status:** Locked for Phase 10  
**Checkpoint:** `backend/models/suggestion_ranker.pt`  
**Data:** `data/processed/suggestion_dataset.parquet` (Dataset C, feature contract v1)

---

## What the model sees

| | |
|--|--|
| **Input** | 24 geometry floats (train μ/σ) **+** 72 one-hots (`y_*`: low/ok/high) → **96-d** |
| **Target** | Multi-hot over catalog IDs with ≥1 train positive |
| **Loss** | `BCEWithLogitsLoss` with `pos_weight` for rare IDs |
| **Architecture** | `Linear(96→128)→ReLU→Dropout→Linear(128→128)→ReLU→Dropout→Linear(128→K)` |
| **Serve** | top-4 IDs → approved text from `data/catalogs/suggestions.csv` → API `suggestions[]` |

At train time, `y_*` come from Dataset C (percentile labels).  
At serve time (no Feature MLP required), `y_*` are rebuilt with `data/processed/suggestion_mapping_rules.csv`.

---

## Commands

```bash
source backend/.venv/bin/activate
python backend/scripts/run_dataset_quality_gates.py   # hard FAIL blocks training readiness
python backend/scripts/train_suggestion_ranker.py
```

Quality gates often **WARN** that many IDs have &lt;50 positives on ~412 faces — expected. Expand extracts / human labels before production; training still uses class weights.

Manual spot-check: `data/labeling/spotcheck/spotcheck_100.csv`.

---

## Checkpoint keys

```text
feat_mu, feat_sd, suggestion_ids, state,
feature_contract_version, use_class_onehots=true, in_dim=96
```

---

## API field

When the checkpoint is present, `POST /analyze` includes:

```json
"suggestions": [{"id": "SUG_...", "text": "...", "confidence": 0.83}]
```

If missing, `suggestions` is `[]` and legacy `recommendations` / `recommendation_items` still return (beauty+feature path).

---

## Offline REINFORCE (ranking fine-tune)

Listwise training objective for the same MLP / checkpoint schema:

- Contract: [`suggestion_rl_contract_v1.md`](suggestion_rl_contract_v1.md) (NDCG@4 reward, Plackett–Luce, graded gains).
- Stage A (BCE): `backend/models/suggestion_ranker_bce_v1.pt` (rollback).
- Stage B (REINFORCE): `python backend/scripts/train_suggestion_ranker_reinforce.py --init …/suggestion_ranker_bce_v1.pt --out …/suggestion_ranker_rl_v1.pt`
- Bakeoff: `python backend/scripts/eval_suggestion_ranker.py --ckpt … --tag …` → [`../processed/ranker_bakeoff_metrics.json`](../processed/ranker_bakeoff_metrics.json)
- Ship rule: promote RL to `suggestion_ranker.pt` only if `val_ndcg@4` ≥ BCE and `test_ndcg@4` ≥ BCE − 0.02.

### Stage A/B bakeoff (locked seed 42)

| Tag | Artifact | val NDCG@4 | val MAP@4 | test NDCG@4 | test MAP@4 |
|-----|----------|------------|-----------|-------------|------------|
| `bce_v1` | `suggestion_ranker_bce_v1.pt` | 0.6991 | 0.8698 | 0.7359 | 0.9211 |
| `rl_v1` | `suggestion_ranker_rl_v1.pt` | **0.8066** | 0.9149 | **0.8187** | 0.9306 |

**Shipped:** `rl_v1` → production `backend/models/suggestion_ranker.pt` (BCE kept as rollback).

---

## Phase 5 evaluation (frozen choice)

Command:

```bash
python backend/scripts/report_suggestion_ranker_eval.py --smoke-analyze
```

Artifacts:

- Report: [`../processed/suggestion_ranker_phase5_report.md`](../processed/suggestion_ranker_phase5_report.md)
- Spot-check (25 test rows): [`../labeling/spotcheck/suggestion_ranker_spotcheck_25_test.csv`](../labeling/spotcheck/suggestion_ranker_spotcheck_25_test.csv)
- Metrics `phase5` block: [`../processed/ranker_bakeoff_metrics.json`](../processed/ranker_bakeoff_metrics.json)

Outcome (production `suggestion_ranker.pt`):

| Check | Result |
|-------|--------|
| val / test NDCG@4 | 0.8066 / 0.8187 |
| val / test MAP@4 | 0.9149 / 0.9306 |
| Ship gate reaffirm | **pass** (`rl_v1`) |
| Catalog coverage | 38 / 48 unique IDs; top-ID share 0.081 (no collapse) |
| Serve smoke | `fallback_predict_suggestions_missing_beauty_feature` (beauty/feature `.pt` absent; suggestions still catalog-valid) |

---

## Phase 6 — Export and integration (frozen)

| Item | Locked value |
|------|----------------|
| **Algorithm** | Offline REINFORCE (Plackett–Luce policy over top-4) |
| **Reward** | NDCG@4 with graded gains `[3, 2, 1, 1]` — [`suggestion_rl_contract_v1.md`](suggestion_rl_contract_v1.md) |
| **Pipeline** | Stage A BCE warm-start → Stage B REINFORCE fine-tune → bakeoff → ship |
| **Production** | `backend/models/suggestion_ranker.pt` (= `suggestion_ranker_rl_v1.pt`) |
| **Rollback** | `backend/models/suggestion_ranker_bce_v1.pt` |
| **Named RL** | `backend/models/suggestion_ranker_rl_v1.pt` |
| **API contract** | **Unchanged** — same checkpoint keys; same `suggestions[{id,text,confidence}]` |

Promote after bakeoff (only if RL wins ship gate):

```bash
cp backend/models/suggestion_ranker_rl_v1.pt backend/models/suggestion_ranker.pt
```

Serve loads `DEFAULT_CKPT = backend/models/suggestion_ranker.pt` ([`suggestion_serve.py`](../../backend/suggestion_serve.py)). Full `POST /analyze` still requires beauty/feature `.pt` files (separate from the suggestion ranker).

---

## Phase 7 — Recommended order of work

| Step | Work | Outcome | Status |
|------|------|---------|--------|
| 1 | Gates + freeze RL reward contract | Ready | **Done** ([`suggestion_rl_contract_v1.md`](suggestion_rl_contract_v1.md)) |
| 2 | NDCG + Plackett–Luce + REINFORCE | Train script | **Done** ([`suggestion_rl.py`](../../backend/suggestion_rl.py), [`train_suggestion_ranker_reinforce.py`](../../backend/scripts/train_suggestion_ranker_reinforce.py)) |
| 3 | BCE Stage A + baseline metrics | `*_bce_v1.pt` | **Done** |
| 4 | REINFORCE Stage B | `*_rl_v1.pt` | **Done** |
| 5 | Val/test compare + spot-check | Choose ship model | **Done** (Phase 5 report + spot-check CSV) |
| 6 | Deploy chosen `.pt` into serve | Live suggestions path | **Done** (production `.pt`; full `/analyze` still needs beauty/feature) |
| 7 | Human labels → retrain A+B | Higher reward quality | **Ready for annotator** (packets refreshed; Dataset C still `rules_v1` until merge) |

Optional next step uses existing labeling tooling documented in [`create_dataset.md`](create_dataset.md) (`labeling_app.py`, `merge_human_labels.py`), then re-run Stage A → Stage B → Phase 5 report. See **Phase 8** below.

---

## Phase 8 — Human labels (reward quality)

**Current state:** Dataset C is 100% `rules_v1`. That is valid for v1 — Offline REINFORCE learns to match the rules `priority_order` well.

**Why human labels:** Human `priority_order` is higher-trust reward for NDCG@4. After merge, Stage A/B train with sample weights:

| `label_method` | Train weight |
|----------------|--------------|
| `rules_v1` | 1.0 |
| `ensemble_v1` | 2.0 |
| `human_v1` | **3.0** |

Code: [`suggestion_label_trust.py`](../../backend/suggestion_label_trust.py) (wired into BCE + REINFORCE trainers).

### Calibration packets (refreshed)

```bash
python backend/scripts/export_labeling_packets.py --annotator-id ann_a --role primary --limit 50
python backend/scripts/export_labeling_packets.py --annotator-id ann_b --role secondary --secondary-frac 0.15
python backend/scripts/labeling_app.py --packet data/labeling/packets/ann_a_primary.jsonl
# open http://127.0.0.1:5055/
```

Guide: [`../catalogs/labeling_guide.md`](../catalogs/labeling_guide.md).

### After you finish labeling

```bash
python backend/scripts/merge_human_labels.py \
  --primary data/labeling/submissions/ann_a.csv \
  --primary-annotator ann_a
python backend/scripts/freeze_dataset_schema.py
python backend/scripts/validate_dataset_schema.py
python backend/scripts/run_dataset_quality_gates.py
python backend/scripts/train_suggestion_ranker.py
cp backend/models/suggestion_ranker.pt backend/models/suggestion_ranker_bce_v1.pt
python backend/scripts/train_suggestion_ranker_reinforce.py \
  --init backend/models/suggestion_ranker_bce_v1.pt \
  --out backend/models/suggestion_ranker_rl_v1.pt
# bakeoff + Phase 5 report, then promote if ship gate passes
python backend/scripts/report_suggestion_ranker_eval.py --smoke-analyze
```

Do **not** invent human ranks in code — only merge real UI submissions.
