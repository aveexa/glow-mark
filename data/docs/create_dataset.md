# Glow-Mark — Create Dataset & Suggestion Ranker (Phases 0–10)

**Purpose:** End-to-end record of how Dataset C (beauty suggestions) and the suggestion ranker were built, what each phase produced, and how to rebuild.

**Product goal:** Face image → MediaPipe geometry → multi-label suggestion MLP → top-k catalog IDs → approved text on `POST /analyze` as `suggestions[]`.

**Current scale:** ~5k LFW images under `data/raw/images/`; ~412 accepted after pose/face gates; Dataset B/C frozen at that size; `suggestion_ranker.pt` trained with class-weighted BCE.

**Related contracts (source of truth for details):**

| Doc | Topic |
|-----|--------|
| [feature_contract_v1.md](feature_contract_v1.md) | 24 geometry formulas, pose ±15°, freeze rule |
| [model_contract_v1.md](model_contract_v1.md) | Ranker I/O, checkpoint keys, safety |
| [dataset_schema_v1.md](dataset_schema_v1.md) | Dataset B/C column order (CSV + Parquet) |
| [training_suggestion_ranker_v1.md](training_suggestion_ranker_v1.md) | Train/serve recipe for the ranker |

---

## End-to-end pipeline

```mermaid
flowchart LR
  images[data/raw/images] --> extract[extract_geometry_batch]
  extract --> interim[interim geometry CSV]
  interim --> geom[geometry_dataset B]
  geom --> rules[rules mapper]
  rules --> sug[suggestion_dataset C]
  sug --> human[optional human labels]
  human --> freeze[freeze schema + Parquet]
  freeze --> gates[quality gates]
  gates --> train[train_suggestion_ranker]
  train --> ckpt[suggestion_ranker.pt]
  ckpt --> api["POST /analyze suggestions"]
```

---

## Phase 0 — Model contract

### Goal

Freeze what the suggestion model learns and serves **before** collecting/training data.

### What was done

- Locked task type: multi-label ranking over approved `suggestion_id`s (not free-text generation).
- Locked input/output, checkpoint keys, safety (cosmetic/styling only).
- Seeded catalog so IDs exist before labeling.

### Files created / modified

| Path | Purpose |
|------|---------|
| [data/docs/model_contract_v1.md](model_contract_v1.md) | Frozen model contract (X/Y, loss, serve, safety) |
| [data/catalogs/suggestions.csv](../catalogs/suggestions.csv) | Initial approved suggestion templates (later expanded in Phase 2) |

### How to re-run

Read-only contract — edit only with a version bump (v2) if changing I/O.

---

## Phase 1 — Feature contract freeze

### Goal

One shared geometry module for train and serve; fix dead formulas; reject non-frontal faces.

### What was done

- Extracted geometry from `inference.py` into `backend/geometry.py`.
- Fixed `jaw_width_ratio`, `chin_width_ratio`, `upper_lip_ratio`, `philtrum_ratio`.
- Added yaw/pitch estimate + ±15° frontal gate.
- Batch extract script for `data/raw/images/`.

### Files created / modified

| Path | Purpose |
|------|---------|
| [backend/geometry.py](../../backend/geometry.py) | `FEATURE_COLS`, extract, pose gate, contract version `v1` |
| [backend/inference.py](../../backend/inference.py) | Imports geometry; pose gate before scoring |
| [backend/scripts/extract_geometry_batch.py](../../backend/scripts/extract_geometry_batch.py) | MediaPipe → landmarks + 24 features + rejects log |
| [data/docs/feature_contract_v1.md](feature_contract_v1.md) | Documented formulas and freeze rule |
| `data/raw/images/` | Place for face images |
| `data/interim/landmarks_468/` | Per-sample `.npy` landmark caches |
| `data/interim/geometry_features/` | Per-sample JSON + combined CSV |
| `data/interim/rejects.csv` | Failed faces (pose / multi-face / no face) |

### How to re-run

```bash
source backend/.venv/bin/activate
python backend/scripts/extract_geometry_batch.py
```

---

## Phase 2 — Suggestion catalog v0

### Goal

~50 safe cosmetic templates covering actionable `feature × {low,high}` pairs + neutrals.

### What was done

- Expanded catalog with `forbidden` + `active` columns.
- Built trigger map for rules/annotators.
- Validator for IDs, banned clinical words, coverage matrix.

### Files created / modified

| Path | Purpose |
|------|---------|
| [data/catalogs/suggestions.csv](../catalogs/suggestions.csv) | ~50 `SUG_*` templates + approved text |
| [data/catalogs/suggestion_trigger_map.md](../catalogs/suggestion_trigger_map.md) | `feature + class → suggestion_id` cheat sheet |
| [backend/scripts/validate_suggestion_catalog.py](../../backend/scripts/validate_suggestion_catalog.py) | Catalog QA (exit non-zero on errors) |

### How to re-run

```bash
python backend/scripts/validate_suggestion_catalog.py
```

---

## Phase 3 — Folder layout + labeling guide

### Goal

Finish data layout and freeze Dataset B/C schema docs for later labeling.

### What was done

- Documented folder tree and schema column names.
- Wrote annotator SOP (keep/drop/reorder, safety, agreement targets).
- Noted that pre-v1 synthetic CSVs are not train-ready.

### Files created / modified

| Path | Purpose |
|------|---------|
| [data/catalogs/labeling_guide.md](../catalogs/labeling_guide.md) | Human labeling SOP (+ later UI runbook) |
| [data/docs/dataset_schema_v1.md](dataset_schema_v1.md) | Frozen B/C column contracts |
| [data/processed/README.md](../processed/README.md) | What each processed file is; rebuild commands |

### How to re-run

N/A (documentation). Follow labeling guide when running Phase 7 UI.

---

## Phase 4 — Collect / prepare face images

### Goal

Have a legal/usable pool of faces for landmark extraction.

### What was done

- Populated `data/raw/images/` with ~5k flat `lfw_#####.jpg` files.
- Extracted features with Phase 1 script; pose gate kept ~412 accepts (~4.2k angle rejects, ~386 multi-face).

### Files created / modified

| Path | Purpose |
|------|---------|
| `data/raw/images/lfw_*.jpg` | Source faces |
| `data/interim/geometry_features/geometry_features.csv` | Accepted feature rows (`feature_contract_version=v1`) |
| `data/interim/rejects.csv` | Rejection codes for failed images |

### How to re-run

Add images to `data/raw/images/`, then re-run Phase 1 extract. Loosening pose limits requires a feature-contract version bump.

---

## Phase 5 — Dataset B + percentile classes

### Goal

Build geometry dataset with train-only p20/p80 → `y_*` ∈ `{low,ok,high}`.

### What was done

- Archived synthetic pre-v1 CSVs.
- Hash-split `sample_id` → train/val/test (80/10/10).
- Fit thresholds on train; applied to all splits.
- Wrote Dataset B + mapping rules.

### Files created / modified

| Path | Purpose |
|------|---------|
| [backend/scripts/build_geometry_dataset.py](../../backend/scripts/build_geometry_dataset.py) | Split, thresholds, Dataset B writer |
| [data/processed/geometry_dataset.csv](../processed/geometry_dataset.csv) | Dataset B (later enriched/frozen in Phase 8) |
| [data/processed/suggestion_mapping_rules.csv](../processed/suggestion_mapping_rules.csv) | Per-feature p20/p80 thresholds |
| `data/processed/*.synthetic_pre_v1.csv` | Archived old synthetic drafts (do not train) |

### How to re-run

```bash
python backend/scripts/build_geometry_dataset.py
```

---

## Phase 6 — Rules-mapped Dataset C

### Goal

Weak labels: map non-`ok` classes → catalog IDs, top-k=4, `annotator_id=rules_v1`.

### What was done

- Shared catalog mapper module.
- Built Dataset C from Dataset B + catalog.
- Cap k=4; severity ranks `info` before `mild`.

### Files created / modified

| Path | Purpose |
|------|---------|
| [backend/suggestion_rules.py](../../backend/suggestion_rules.py) | `load_catalog`, `map_candidates` (shared) |
| [backend/scripts/build_suggestion_dataset_rules.py](../../backend/scripts/build_suggestion_dataset_rules.py) | Writes Dataset C (`rules_v1`) |
| [data/processed/suggestion_dataset.csv](../processed/suggestion_dataset.csv) | Dataset C core labels |

### How to re-run

```bash
python backend/scripts/build_suggestion_dataset_rules.py
```

---

## Phase 7 — Human refinement tooling

### Goal

Let annotators keep/drop/reorder candidates into `human_v1` labels; measure agreement.

### What was done

- Exported JSONL packets with up to 8 candidates.
- Local Flask UI on `127.0.0.1:5055` (image + features + candidates).
- Merge primary submissions into Dataset C; Jaccard agreement report.

### Files created / modified

| Path | Purpose |
|------|---------|
| [backend/scripts/export_labeling_packets.py](../../backend/scripts/export_labeling_packets.py) | Build `*_primary.jsonl` / `*_secondary.jsonl` |
| [backend/scripts/labeling_app.py](../../backend/scripts/labeling_app.py) | Local labeling UI |
| [backend/scripts/merge_human_labels.py](../../backend/scripts/merge_human_labels.py) | Overlay `human_v1` onto Dataset C |
| [backend/scripts/compute_label_agreement.py](../../backend/scripts/compute_label_agreement.py) | Primary vs secondary Jaccard |
| `data/labeling/packets/` | Per-annotator JSONL queues |
| `data/labeling/submissions/` | Human CSV submissions |
| `data/labeling/agreement/` | Agreement reports |
| [data/catalogs/labeling_guide.md](../catalogs/labeling_guide.md) | Updated with UI runbook |
| `data/processed/suggestion_dataset.rules_v1_backup.csv` | Backup before first human merge (if created) |

### How to re-run

```bash
python backend/scripts/export_labeling_packets.py --annotator-id ann_a --role primary
python backend/scripts/labeling_app.py --packet data/labeling/packets/ann_a_primary.jsonl
# open http://127.0.0.1:5055/
python backend/scripts/merge_human_labels.py --primary data/labeling/submissions/ann_a.csv --primary-annotator ann_a
python backend/scripts/compute_label_agreement.py \
  --primary data/labeling/submissions/ann_a.csv \
  --secondary data/labeling/submissions/ann_b.csv
```

---

## Phase 8 — Schema freeze + Parquet

### Goal

Freeze column order; add training enrich columns; export CSV + Parquet for B and C.

### What was done

- Added `num_non_ok_features`, `primary_feature`, `quality_score` (pose proxy).
- Rewrote CSV with locked column order; wrote sibling Parquet files.
- Schema validator for CSV/Parquet parity.

### Files created / modified

| Path | Purpose |
|------|---------|
| [backend/dataset_schema.py](../../backend/dataset_schema.py) | Frozen column lists + enrich helpers |
| [backend/scripts/freeze_dataset_schema.py](../../backend/scripts/freeze_dataset_schema.py) | Enrich + write CSV/Parquet |
| [backend/scripts/validate_dataset_schema.py](../../backend/scripts/validate_dataset_schema.py) | Assert frozen schema |
| [data/processed/geometry_dataset.parquet](../processed/geometry_dataset.parquet) | Dataset B Parquet |
| [data/processed/suggestion_dataset.parquet](../processed/suggestion_dataset.parquet) | Dataset C Parquet |
| [data/docs/dataset_schema_v1.md](dataset_schema_v1.md) | Marked frozen; documented enrich formulas |
| [backend/requirements.txt](../../backend/requirements.txt) | Added `pandas`, `pyarrow` |

### How to re-run

```bash
python backend/scripts/freeze_dataset_schema.py
python backend/scripts/validate_dataset_schema.py
```

---

## Phase 9 — Quality gates

### Goal

ML-readiness checks before training (schema, catalog IDs, leakage, balance, duplicates).

### What was done

- Hard FAIL vs WARN report (balance &lt;50 and missing identity map are WARNs at current scale).
- Exported 100-row spot-check sheet for manual review.
- Passed gates on current LFW Dataset C (0 fails, 2 warns expected).

### Files created / modified

| Path | Purpose |
|------|---------|
| [backend/scripts/run_dataset_quality_gates.py](../../backend/scripts/run_dataset_quality_gates.py) | Phase 9 gate runner |
| [data/processed/quality_report.json](../processed/quality_report.json) | Machine-readable gate result |
| [data/processed/quality_report.md](../processed/quality_report.md) | Human-readable gate result |
| `data/labeling/spotcheck/spotcheck_100.csv` | Manual sanity spot-check worksheet |

### How to re-run

```bash
python backend/scripts/run_dataset_quality_gates.py
```

---

## Phase 10 — Train + serve suggestion ranker

### Goal

Train catalog-constrained MLP on Dataset C; serve `suggestions[]` from checkpoint.

### What was done

- Model: 96-d input (24 standardized floats + 72 `y_*` one-hots) → K logits; BCE + `pos_weight`.
- Checkpoint: `backend/models/suggestion_ranker.pt`.
- Serve: thresholds rebuild `y_*` → ranker → top-4 → catalog text.
- Wired into `inference.py` (empty `suggestions` if `.pt` missing; no crash).

### Files created / modified

| Path | Purpose |
|------|---------|
| [backend/suggestion_model.py](../../backend/suggestion_model.py) | MLP + encode/decode/checkpoint helpers |
| [backend/scripts/train_suggestion_ranker.py](../../backend/scripts/train_suggestion_ranker.py) | Train loop + export `.pt` |
| [backend/suggestion_serve.py](../../backend/suggestion_serve.py) | Inference helper → `[{id,text,confidence}]` |
| [backend/inference.py](../../backend/inference.py) | Adds `suggestions` field on analyze |
| [backend/models/suggestion_ranker.pt](../../backend/models/suggestion_ranker.pt) | Trained weights (gitignored) |
| [backend/models/.gitkeep](../../backend/models/.gitkeep) | Keep models dir in git |
| [backend/.gitignore](../../backend/.gitignore) | Ignore `models/*.pt` |
| [data/docs/training_suggestion_ranker_v1.md](training_suggestion_ranker_v1.md) | Train/serve recipe |
| [data/docs/model_contract_v1.md](model_contract_v1.md) | Updated for `in_dim=96` |

### How to re-run

```bash
python backend/scripts/run_dataset_quality_gates.py
python backend/scripts/train_suggestion_ranker.py
# restart Flask: python backend/app.py
```

API field when checkpoint present:

```json
"suggestions": [{"id": "SUG_...", "text": "...", "confidence": 0.83}]
```

---

## Master file index

### Docs

| Path | Purpose |
|------|---------|
| [create_dataset.md](create_dataset.md) | This guide (all phases) |
| [feature_contract_v1.md](feature_contract_v1.md) | Geometry formulas + pose |
| [model_contract_v1.md](model_contract_v1.md) | Ranker contract |
| [dataset_schema_v1.md](dataset_schema_v1.md) | B/C columns |
| [training_suggestion_ranker_v1.md](training_suggestion_ranker_v1.md) | Train/serve details |

### Catalogs / labeling

| Path | Purpose |
|------|---------|
| `data/catalogs/suggestions.csv` | Approved suggestion texts |
| `data/catalogs/suggestion_trigger_map.md` | Rules trigger map |
| `data/catalogs/labeling_guide.md` | Annotator + UI SOP |
| `data/labeling/packets/` | JSONL labeling queues |
| `data/labeling/submissions/` | Human CSV outputs |
| `data/labeling/agreement/` | Double-label metrics |
| `data/labeling/spotcheck/` | Manual QA sheet |

### Processed data

| Path | Purpose |
|------|---------|
| `data/processed/geometry_dataset.csv` / `.parquet` | Dataset B |
| `data/processed/suggestion_dataset.csv` / `.parquet` | Dataset C (**main training set for suggestions**) |
| `data/processed/suggestion_mapping_rules.csv` | Train p20/p80 |
| `data/processed/quality_report.json` / `.md` | Gate results |
| `data/processed/*.synthetic_pre_v1.csv` | Archived junk for training |

### Backend modules

| Path | Purpose |
|------|---------|
| `backend/geometry.py` | Shared feature extract + pose |
| `backend/suggestion_rules.py` | Rules mapper |
| `backend/dataset_schema.py` | Frozen schema + enrich |
| `backend/suggestion_model.py` | Ranker MLP |
| `backend/suggestion_serve.py` | Serve path |
| `backend/inference.py` | Analyze API logic |
| `backend/app.py` | Flask entrypoint |

### Scripts

| Path | Purpose |
|------|---------|
| `extract_geometry_batch.py` | Phase 1/4 extract |
| `validate_suggestion_catalog.py` | Phase 2 catalog QA |
| `build_geometry_dataset.py` | Phase 5 Dataset B |
| `build_suggestion_dataset_rules.py` | Phase 6 Dataset C |
| `export_labeling_packets.py` | Phase 7 packets |
| `labeling_app.py` | Phase 7 UI |
| `merge_human_labels.py` | Phase 7 merge |
| `compute_label_agreement.py` | Phase 7 agreement |
| `freeze_dataset_schema.py` | Phase 8 freeze/export |
| `validate_dataset_schema.py` | Phase 8 schema QA |
| `run_dataset_quality_gates.py` | Phase 9 gates |
| `train_suggestion_ranker.py` | Phase 10 train |

### Models

| Path | Purpose |
|------|---------|
| `backend/models/suggestion_ranker.pt` | Suggestion MLP weights |
| `backend/models/beauty_landmarks_best.pt` | Beauty score (separate; may be missing) |
| `backend/models/feature_geometry_model.pt` | Feature classes (separate; may be missing) |

---

## Full rebuild sequence

```bash
source backend/.venv/bin/activate

# Features
python backend/scripts/extract_geometry_batch.py

# Dataset B + C
python backend/scripts/build_geometry_dataset.py
python backend/scripts/build_suggestion_dataset_rules.py

# Optional human labels
# python backend/scripts/export_labeling_packets.py --annotator-id ann_a --role primary
# python backend/scripts/labeling_app.py --packet data/labeling/packets/ann_a_primary.jsonl
# python backend/scripts/merge_human_labels.py --primary data/labeling/submissions/ann_a.csv

# Freeze + validate + gates + train
python backend/scripts/freeze_dataset_schema.py
python backend/scripts/validate_dataset_schema.py
python backend/scripts/validate_suggestion_catalog.py
python backend/scripts/run_dataset_quality_gates.py
python backend/scripts/train_suggestion_ranker.py

# Serve
python backend/app.py
```

---

## Known limitations

- **Yield:** Pose gate (±15°) rejects most LFW images → ~412 training rows.
- **Balance:** Many `suggestion_id`s have &lt;50 positives → quality WARN; training uses `pos_weight`.
- **Identity:** Flat `lfw_#####` names → no person-level leakage check (WARN only).
- **Labels:** Dataset C is mostly `rules_v1` until you complete human labeling (Phase 7).
- **Beauty / Feature MLPs:** Separate checkpoints for score / class labels; may still be missing. Suggestion path can run with percentile thresholds + `suggestion_ranker.pt` alone for `suggestions[]`, but full `/analyze` still expects beauty+feature today.
- **Quality score:** Pose proxy only (not blur/landmark confidence yet).
- **Spot-check:** `spotcheck_100.csv` must be reviewed by a human; not automated.

---

## Where is the “exact” suggestion dataset?

Primary artifact for ranking training:

- **CSV:** `data/processed/suggestion_dataset.csv`
- **Parquet:** `data/processed/suggestion_dataset.parquet`

Catalog text lookup:

- `data/catalogs/suggestions.csv`

Trained model:

- `backend/models/suggestion_ranker.pt`
