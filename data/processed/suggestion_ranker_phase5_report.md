# Suggestion ranker — Phase 5 evaluation

**Evaluated at:** `2026-07-15T07:31:17.339753+00:00`  
**Checkpoint:** `/Users/tharushasamarawickrama/Downloads/glow-mark/backend/models/suggestion_ranker.pt`  

## Ranking metrics (greedy top-4)

| Split | NDCG@4 | MAP@4 | n |
|-------|--------|-------|---|
| val | 0.8066 | 0.9149 | 48 |
| test | 0.8187 | 0.9306 | 38 |

## Ship gate (reaffirm vs bakeoff bce_v1 / rl_v1)

- Rule: `rl.val_ndcg>=bce.val_ndcg AND rl.test_ndcg>=bce.test_ndcg-0.02`
- Pass: **True** (winner `rl_v1`)
- BCE val/test NDCG@4: 0.6991 / 0.7359
- RL  val/test NDCG@4: 0.8066 / 0.8187

## Catalog ID coverage (val ∪ test predictions)

- Unique predicted IDs: **38** / vocab 48 (79.2%)
- Top ID: `SUG_SYM_HIGH_01` share **0.081**
- All predicted IDs in catalog: True

## Confidence (sigmoid on greedy top-4 slots)

- mean=0.8803  p50=0.9346  p90=0.9976  n=344
- histogram bins: ['[0,0.2)', '[0.2,0.4)', '[0.4,0.6)', '[0.6,0.8)', '[0.8,1]']
- histogram counts: [0, 1, 17, 63, 263]

## Spot-check

- Sheet: `/Users/tharushasamarawickrama/Downloads/glow-mark/data/labeling/spotcheck/suggestion_ranker_spotcheck_25_test.csv` (25 test rows, seed 42)

## Serve smoke

- status: `fallback_predict_suggestions_missing_beauty_reco`
- sample_id: `lfw_00288`
- suggestion_ids: `['SUG_FACE_ASPECT_HIGH_01', 'SUG_JAW_WIDE_01', 'SUG_JAW_ANGLE_LOW_01', 'SUG_MIDFACE_HIGH_01']`

## Warns

- none
