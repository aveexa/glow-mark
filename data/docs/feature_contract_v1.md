# Feature contract v1 (locked)

**Status:** Locked — do not edit formulas after labeling starts  
**Date:** 2026-07-14  
**Code:** [`backend/geometry.py`](../../backend/geometry.py) (`FEATURE_CONTRACT_VERSION = "v1"`)  
**Model contract:** [`model_contract_v1.md`](model_contract_v1.md)

Next breaking change must bump the version to `v2` and **re-extract all landmarks/features**.

---

## Normalization

- Landmarks: MediaPipe FaceMesh, 468 points, `static_image_mode=True`, `refine_landmarks=False` (same as [`backend/inference.py`](../../backend/inference.py) `_extract_landmarks_468`).
- Feature extraction uses **normalized** coordinates (`x,y` in image-relative space, plus `z` when available for pose).
- Pixel XY is used only for the separate beauty MLP (68-pt flatten), not for these 24 geometry floats.

---

## Pose gate

| Constant | Value |
|----------|-------|
| `POSE_YAW_MAX_DEG` | `25.0` |
| `POSE_PITCH_MAX_DEG` | `25.0` |
| `EXPECTED_NOSE_EYE_CHIN_FRAC` | `0.37` |

**Yaw (v1):** nose tip `4` offset from face midline `(234+454)/2`, divided by half face width, × 90°, optional soft blend with cheek `z` asymmetry `(z[234]-z[454])`.

**Pitch (v1):** nose tip `4` vs expected nose `y = eye_line + 0.37×(chin−eye_line)` (calibrated frontal tip), divided by half the eye→chin span, × 90°.

Reject with `FACE_TOO_ANGLED_OR_SMALL` when `|yaw| > 25` or `|pitch| > 25`.

**Serve-time note:** Pose limits and the nose-fraction constant were recalibrated to reduce false rejects. The 24 geometry feature formulas are unchanged, so `FEATURE_CONTRACT_VERSION` stays `v1` (no dataset re-extract / model retrain required for this change).

---

## Ordered feature list (`FEATURE_COLS`)

1. `symmetry_error`  
2. `face_aspect_ratio`  
3. `midface_length_ratio`  
4. `lowerface_length_ratio`  
5. `jaw_width_ratio`  
6. `jaw_angle_sharpness`  
7. `chin_length_ratio`  
8. `chin_width_ratio`  
9. `cheekbone_width_ratio`  
10. `lower_cheek_ratio`  
11. `eye_openness_ratio`  
12. `eye_size_ratio`  
13. `eye_spacing_ratio`  
14. `eye_tilt_deg`  
15. `brow_height_ratio`  
16. `brow_tilt_deg`  
17. `nose_width_ratio`  
18. `nose_length_ratio`  
19. `nose_tip_deviation_ratio`  
20. `mouth_width_ratio`  
21. `mouth_corner_tilt_deg`  
22. `lip_thickness_ratio`  
23. `upper_lip_ratio`  
24. `philtrum_ratio`  

Length must remain **24**.

---

## Per-feature formulas (v1)

Notation: `d(a,b)` = Euclidean distance; indices are MediaPipe Face Mesh.

| Feature | Formula / landmarks |
|---------|---------------------|
| `symmetry_error` | Mean \| (lx−c)+(rx−c) \| for pairs `(33,263),(133,362),(61,291),(234,454),(105,334),(10,10)`; `c` = midline x from `(234,454)` |
| `face_aspect_ratio` | `d(234,454) / d(10,152)` |
| `midface_length_ratio` | \|nose_tip.y − eye_line_y\| / face_height; eye_line from `(33,263)`; nose `4` |
| `lowerface_length_ratio` | \|chin.y − nose_tip.y\| / face_height; chin `152` |
| **`jaw_width_ratio`** | **`d(172,397) / face_width`** (mandibular / jaw-angle width; **v1 fix**) |
| `jaw_angle_sharpness` | `180 − angle_at(152; 234,454)` |
| `chin_length_ratio` | \|chin.y − nose_tip.y\| / face_height |
| **`chin_width_ratio`** | **`d(176,400) / face_width`** (narrower chin band; **v1 fix**, distinct from jaw) |
| `cheekbone_width_ratio` | `d(93,323) / face_width` |
| `lower_cheek_ratio` | `d(58,288) / face_width` |
| `eye_openness_ratio` | \|y(145)−y(159)\| / left eye width `d(33,133)` |
| `eye_size_ratio` | left eye width / face_width |
| `eye_spacing_ratio` | `d(133,362) / d(33,133)` |
| `eye_tilt_deg` | angle of vector `33→263` |
| `brow_height_ratio` | \|y(105)−y(33)\| / face_height |
| `brow_tilt_deg` | angle of vector `105→334` |
| `nose_width_ratio` | `d(131,360) / face_width` |
| `nose_length_ratio` | `d(6,4) / face_height` |
| `nose_tip_deviation_ratio` | \|nose_tip.x − midline\| / face_width |
| `mouth_width_ratio` | `d(61,291) / face_width` |
| `mouth_corner_tilt_deg` | angle of vector `61→291` |
| `lip_thickness_ratio` | \|y(14)−y(13)\| / face_height |
| **`upper_lip_ratio`** | **`d(0,13) / face_height`** (upper vermilion; **v1 fix**) |
| **`philtrum_ratio`** | **`d(2,0) / face_height`** (subnasale → cupid; **v1 fix**) |

Face width = `d(234,454)`; face height = `d(10,152)`.

---

## Dataset notes

- Existing synthetic CSVs under `data/processed/` (`geometry_dataset.csv`, `suggestion_dataset.csv`) were built with the **old** jaw/lip formulas. They are **not valid** for training under feature contract v1. Re-extract after placing real/licensed images in `data/raw/images/`.
- Use [`backend/scripts/extract_geometry_batch.py`](../../backend/scripts/extract_geometry_batch.py) for v1 extraction.
- Percentile class thresholds (`low`/`ok`/`high`) must be recomputed on the train split after re-extraction.

---

## Freeze rule

**v1 is locked.**  
Any formula, landmark index, or pose-limit change → `FEATURE_CONTRACT_VERSION = "v2"`, new docs, and full data re-extraction. Do not mix v1 and v2 rows in the same training set.
