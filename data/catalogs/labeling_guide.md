# Labeling guide — Dataset C (suggestion ranking)

**Catalog:** [`suggestions.csv`](suggestions.csv) (v0)  
**Trigger map:** [`suggestion_trigger_map.md`](suggestion_trigger_map.md)  
**Model contract:** [`../docs/model_contract_v1.md`](../docs/model_contract_v1.md)  
**Feature contract:** [`../docs/feature_contract_v1.md`](../docs/feature_contract_v1.md)

This guide is for human annotators who refine rule-proposed beauty suggestions.  
You never write free-text advice. You only rank approved `suggestion_id`s.

---

## Current pool size

The feature-contract v1 extract currently has **~412** accepted faces.  
Label **all available** samples first (calibration). Scaling to 500–5k requires more accepted extracts later — do not invent IDs or lower safety to fill volume.

Identity note: flat `lfw_#####` files have no person map yet; keep the existing hash split. When an identity map exists, re-split by person (not by frame).

---

## Goal

For each face sample, produce an ordered list of up to **4** suggestion IDs that are:

1. Allowed by the catalog (`active=true`, `forbidden=false`)
2. Relevant to the measured geometry state
3. Useful and safe (cosmetic / photo / styling only)

---

## Launch the labeling UI

```bash
source backend/.venv/bin/activate

# Primary queue (calibration batch of 50, or omit --limit for all)
python backend/scripts/export_labeling_packets.py --annotator-id ann_a --role primary --limit 50

# Secondary double-label ~15% of train
python backend/scripts/export_labeling_packets.py --annotator-id ann_b --role secondary --secondary-frac 0.15

# Start UI (binds 127.0.0.1:5055 only)
python backend/scripts/labeling_app.py --packet data/labeling/packets/ann_a_primary.jsonl
```

Open http://127.0.0.1:5055/ in your browser.  
Submissions append to `data/labeling/submissions/{annotator_id}.csv`.

### Primary vs secondary

| Role | Who | Packet | `agreement_flag` |
|------|-----|--------|------------------|
| primary | Main annotator | `*_primary.jsonl` | `primary` |
| secondary | Second annotator (overlap) | `*_secondary.jsonl` (~15% train) | `secondary` |

After both finish:

```bash
python backend/scripts/merge_human_labels.py \
  --primary data/labeling/submissions/ann_a.csv \
  --primary-annotator ann_a

python backend/scripts/compute_label_agreement.py \
  --primary data/labeling/submissions/ann_a.csv \
  --secondary data/labeling/submissions/ann_b.csv
```

Target: mean Jaccard ≥ **0.6** on ID sets for double-labeled samples.

---

## What you see per sample (UI)

| Input | Meaning |
|-------|---------|
| Face image | Frontal photo (pose gate should already have passed) |
| 24 geometry floats | Feature-contract v1 values |
| `y_<feature>` classes | Each ∈ `{low, ok, high}` |
| Candidate IDs | Rules mapper up to 8 IDs (promote/drop/reorder) |

---

## UI actions

1. **Review** candidates against the image and classes.
2. **Keep / drop** each candidate (checkboxes).
3. **Reorder** kept IDs (Move up / Move down) — most important first.
4. Cap at **top-k = 4**. Prefer fewer if unsure.
5. Optionally **Add from catalog** (dropdown of active IDs only — no free text).
6. **Save** → next sample. Use **Skip** to leave unlabeled for later.

### Prefer capture / quality IDs when

- Lighting is uneven or harsh side shadow → `SUG_LIGHTING_01`
- Head appears turned / chin extreme → `SUG_POSE_RETAKE_01`
- `y_symmetry_error == high` → put `SUG_SYM_HIGH_01` near the front

Do not stack many conflicting shape tips on a poor capture; fix capture first.

---

## Hard rules

- Only IDs from [`suggestions.csv`](suggestions.csv) with `active=true` and `forbidden=false`.
- Never invent new IDs or rewrite `approved_text`.
- Never add medical, surgical, or guaranteed-outcome language.
- Prefer **one ID per feature** (do not select both “narrow” and “wide” for the same feature).
- Do not select an ID whose `trigger_class` contradicts the sample’s `y_<feature>` unless you are choosing a capture/neutral ID.

---

## Output fields (submission CSV)

| Field | Format | Example |
|-------|--------|---------|
| `sample_id` | string | `lfw_00042` |
| `annotator_id` | Stable handle | `ann_a` |
| `priority_order` | Pipe-separated ranked list | `SUG_LIGHTING_01\|SUG_NOSE_WIDE_01` |
| `suggestion_ids` | Same IDs as priority (pipe-separated) | same as priority |
| `agreement_flag` | `primary` \| `secondary` | `primary` |
| `notes` | Optional short note | |
| `labeled_at` | ISO timestamp | |

After merge into Dataset C:

| Field | Value |
|-------|--------|
| `label_method` | `human_v1` |
| `annotator_id` | Your handle |
| `suggestion_ids` / `priority_order` | Human ranking |

Rules-only rows stay `rules_v1` until labeled.

**Production ranker note:** v1 ships Offline REINFORCE (NDCG@4). After you merge `human_v1` rows, Stage A/B retrain with trust weights (`human_v1` = 3×, `rules_v1` = 1×). See [`../docs/training_suggestion_ranker_v1.md`](../docs/training_suggestion_ranker_v1.md) Phase 8.

---

## Quality / agreement

- Double-label **≥10–15%** of the train pool with a second annotator.
- Target **Jaccard ≥ 0.6** on ID sets.
- Escalate catalog gaps to product — do not invent wording.

---

## Examples

**Mostly ok classes**

- Keep: `SUG_OK_KEEP_01` (+ `SUG_OK_SKIN_PREP_01` if useful).

**`nose_width_ratio=high`, others ok**

- Priority: `SUG_NOSE_WIDE_01` then optional `SUG_OK_SKIN_PREP_01`.

**`symmetry_error=high` + `jaw_width_ratio=high` + harsh lighting in photo**

- Priority: `SUG_LIGHTING_01` → `SUG_SYM_HIGH_01` → `SUG_JAW_WIDE_01` (drop extra weak tips).
