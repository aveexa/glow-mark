# `data/processed` — dataset artifacts

| File | Role | Train-ready? |
|------|------|----------------|
| `geometry_dataset.csv` / `.parquet` | **Dataset B (v1 frozen)** LFW accepted extracts + p20/p80 classes + enrich columns | Pipeline-ready |
| `suggestion_dataset.csv` / `.parquet` | **Dataset C (v1 frozen)** rules/human suggestion labels + enrich columns | Weak/human labels for ranker |
| `suggestion_mapping_rules.csv` | Train-split p20/p80 thresholds | Yes (paired with B) |
| `quality_report.json` / `.md` | Phase 9 ML-readiness gates | Pass before training |
| `ranker_bakeoff_metrics.json` | BCE vs RL val/test NDCG@4/MAP@4 + Phase 5 block | Post-train |
| `suggestion_ranker_phase5_report.md` | Phase 5 eval summary (coverage, confidence, smoke) | Post-train |
| `suggestion_dataset.rules_v1_backup.csv` | Pre-human-merge backup (if created) | Rules snapshot |
| `*.synthetic_pre_v1.csv` | Archived pre–feature-contract synthetics | **No** |
| `README.md` | This file | — |

Schemas: [`../docs/dataset_schema_v1.md`](../docs/dataset_schema_v1.md) (frozen).  
Catalog: [`../catalogs/suggestions.csv`](../catalogs/suggestions.csv).  
Ranker train/serve: [`../docs/training_suggestion_ranker_v1.md`](../docs/training_suggestion_ranker_v1.md).  
See also: [`../docs/create_dataset.md`](../docs/create_dataset.md) — full Phases 0–10 guide (what was built, files, rebuild).

## Rebuild

```bash
source backend/.venv/bin/activate
python backend/scripts/extract_geometry_batch.py
python backend/scripts/build_geometry_dataset.py
python backend/scripts/build_suggestion_dataset_rules.py
# optional human labels (Phase 8) — do not invent ranks:
#   export packets → labeling_app → merge_human_labels.py
#   human_v1 rows get train weight 3.0 vs rules_v1 weight 1.0
#   see ../docs/training_suggestion_ranker_v1.md Phase 8
python backend/scripts/freeze_dataset_schema.py
python backend/scripts/validate_dataset_schema.py
python backend/scripts/validate_suggestion_catalog.py
python backend/scripts/run_dataset_quality_gates.py
# Stage A (BCE)
python backend/scripts/train_suggestion_ranker.py
cp backend/models/suggestion_ranker.pt backend/models/suggestion_ranker_bce_v1.pt
# Stage B (Offline REINFORCE) + promote after bakeoff
python backend/scripts/train_suggestion_ranker_reinforce.py \
  --init backend/models/suggestion_ranker_bce_v1.pt \
  --out backend/models/suggestion_ranker_rl_v1.pt
python backend/scripts/eval_suggestion_ranker.py --ckpt backend/models/suggestion_ranker_bce_v1.pt --tag bce_v1
python backend/scripts/eval_suggestion_ranker.py --ckpt backend/models/suggestion_ranker_rl_v1.pt --tag rl_v1
# if ship gate passes:
cp backend/models/suggestion_ranker_rl_v1.pt backend/models/suggestion_ranker.pt
python backend/scripts/report_suggestion_ranker_eval.py --smoke-analyze
```

Notes:

- Percentiles fit on **train** only (`sha1(sample_id) % 100` → 80/10/10).
- After any Dataset C rebuild or human merge, re-run `freeze_dataset_schema.py` so CSV and Parquet stay aligned.
- Enrich columns: `num_non_ok_features`, `primary_feature`, `quality_score` (pose proxy).
- Quality gates may WARN on rare suggestion_ids (<50 positives) with only 412 faces — train with `pos_weight`; expand data later.
- Spot-check sheets: `data/labeling/spotcheck/spotcheck_100.csv` (dataset QA); `suggestion_ranker_spotcheck_25_test.csv` (Phase 5 ranker).
- Ranker rollback: `backend/models/suggestion_ranker_bce_v1.pt`.
- Phase 8: Dataset C is rules-only until human merge; calibration packets at `data/labeling/packets/ann_a_primary.jsonl` (50) / `ann_b_secondary.jsonl`.
