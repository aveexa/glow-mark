# Dataset schema v1 (geometry + suggestions) — FROZEN

**Status:** Frozen (Phase 8)  
**Date:** 2026-07-14  
**Code:** [`backend/dataset_schema.py`](../../backend/dataset_schema.py)  
**Depends on:** [`feature_contract_v1.md`](feature_contract_v1.md), [`model_contract_v1.md`](model_contract_v1.md), [`../catalogs/suggestions.csv`](../catalogs/suggestions.csv)

Exports (same columns):

- `data/processed/geometry_dataset.csv` + `.parquet` (Dataset B)
- `data/processed/suggestion_dataset.csv` + `.parquet` (Dataset C)

Current processed LFW v1 builds are production-candidate pipeline artifacts.  
Archived synthetics live as `*.synthetic_pre_v1.csv` (do not train on those).

Rebuild after extract / rules / human merge:

```bash
python backend/scripts/freeze_dataset_schema.py
python backend/scripts/validate_dataset_schema.py
```

---

## FEATURE_COLS (24)

Order must match `backend/geometry.py`:

```text
symmetry_error, face_aspect_ratio, midface_length_ratio, lowerface_length_ratio,
jaw_width_ratio, jaw_angle_sharpness, chin_length_ratio, chin_width_ratio,
cheekbone_width_ratio, lower_cheek_ratio, eye_openness_ratio, eye_size_ratio,
eye_spacing_ratio, eye_tilt_deg, brow_height_ratio, brow_tilt_deg,
nose_width_ratio, nose_length_ratio, nose_tip_deviation_ratio, mouth_width_ratio,
mouth_corner_tilt_deg, lip_thickness_ratio, upper_lip_ratio, philtrum_ratio
```

Class columns: `y_<feature_name>` ∈ `{low, ok, high}`.

---

## Dataset B — `geometry_dataset` (frozen column order)

```text
sample_id, image_path, split,
FEATURE_COLS (24),
y_<FEATURE_COLS> (24),
label_method, consent_flag, source,
feature_contract_version, yaw_deg, pitch_deg,
num_non_ok_features, primary_feature, quality_score
```

| Column | Notes |
|--------|-------|
| `label_method` | Geometry class method, e.g. `percentile_p20_p80` |
| `feature_contract_version` | `v1` |
| `num_non_ok_features` | Count of `y_*` in `{low, high}` |
| `primary_feature` | Feature with max `|(x-μ)/σ|` using **train** μ/σ |
| `quality_score` | Pose proxy `[0,1]`: `exp(-(|yaw|+|pitch|)/30)` (interim until blur/confidence) |

---

## Dataset C — `suggestion_dataset` (frozen column order)

```text
sample_id, image_path, split,
FEATURE_COLS (24),
y_<FEATURE_COLS> (24),
suggestion_ids, priority_order, annotator_id, label_method,
consent_flag, source,
feature_contract_version, yaw_deg, pitch_deg,
num_non_ok_features, primary_feature, quality_score
```

| Column | Notes |
|--------|-------|
| `suggestion_ids` | Pipe-separated catalog IDs |
| `priority_order` | Ranked pipe-separated IDs (same set; ≤4) |
| `annotator_id` | `rules_v1` or human handle |
| `label_method` | `rules_v1` \| `human_v1` \| `ensemble_v1` |

Every ID must exist in the active catalog (`active=true`, `forbidden=false`).

---

## Derived-column formulas (locked)

| Column | Formula |
|--------|---------|
| `num_non_ok_features` | `sum(y_f in {low,high})` over 24 features |
| `primary_feature` | `argmax_f |(x_f - μ_train,f) / σ_train,f|` (σ=0 → z=0) |
| `quality_score` | `clip(exp(-(|yaw_deg|+|pitch_deg|)/30), 0, 1)` |

---

## Mapping thresholds — `suggestion_mapping_rules.csv`

Train-split `p20` / `p80` thresholds for feature-contract v1. Refit after a new extract.

---

## Freeze rule

- Do not rename the 24 features, `y_*` columns, or frozen column order without a new schema version.
- Catalog ID renames require a catalog bump + note in `data/processed/README.md`.
- Changing geometry formulas → feature contract `v2` + full re-extract.
