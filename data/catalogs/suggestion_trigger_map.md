# Suggestion trigger map (catalog v0)

Maps geometry `feature` + class → approved `suggestion_id` from [`suggestions.csv`](suggestions.csv).  
Used by annotators and the future rules mapper. Model never invents IDs outside this catalog.

**Neutrals / capture (no feature trigger required)**

| Trigger | suggestion_id |
|---------|-----------------|
| all features `ok` / fallback | `SUG_OK_KEEP_01` |
| general capture consistency | `SUG_OK_SKIN_PREP_01` |
| uneven lighting suspected | `SUG_LIGHTING_01` |
| pose / centering doubtful | `SUG_POSE_RETAKE_01` |

**Feature × class**

| Feature | low | high |
|---------|-----|------|
| `symmetry_error` | — (omit; no useful styling action) | `SUG_SYM_HIGH_01` |
| `face_aspect_ratio` | `SUG_FACE_ASPECT_LOW_01` | `SUG_FACE_ASPECT_HIGH_01` |
| `midface_length_ratio` | `SUG_MIDFACE_LOW_01` | `SUG_MIDFACE_HIGH_01` |
| `lowerface_length_ratio` | `SUG_LOWERFACE_LOW_01` | `SUG_LOWERFACE_HIGH_01` |
| `jaw_width_ratio` | `SUG_JAW_NARROW_01` | `SUG_JAW_WIDE_01` |
| `jaw_angle_sharpness` | `SUG_JAW_ANGLE_LOW_01` | `SUG_JAW_ANGLE_HIGH_01` |
| `chin_length_ratio` | `SUG_CHIN_SHORT_01` | `SUG_CHIN_LONG_01` |
| `chin_width_ratio` | `SUG_CHIN_NARROW_01` | `SUG_CHIN_WIDE_01` |
| `cheekbone_width_ratio` | `SUG_CHEEK_NARROW_01` | `SUG_CHEEK_WIDE_01` |
| `lower_cheek_ratio` | `SUG_LOWER_CHEEK_LOW_01` | `SUG_LOWER_CHEEK_HIGH_01` |
| `eye_openness_ratio` | `SUG_EYE_OPEN_LOW_01` | `SUG_EYE_OPEN_HIGH_01` |
| `eye_size_ratio` | `SUG_EYE_SIZE_LOW_01` | `SUG_EYE_SIZE_HIGH_01` |
| `eye_spacing_ratio` | `SUG_EYE_CLOSE_01` | `SUG_EYE_WIDE_01` |
| `eye_tilt_deg` | `SUG_EYE_TILT_LOW_01` | `SUG_EYE_TILT_HIGH_01` |
| `brow_height_ratio` | `SUG_BROW_LOW_01` | `SUG_BROW_HIGH_01` |
| `brow_tilt_deg` | `SUG_BROW_TILT_LOW_01` | `SUG_BROW_TILT_HIGH_01` |
| `nose_width_ratio` | `SUG_NOSE_NARROW_01` | `SUG_NOSE_WIDE_01` |
| `nose_length_ratio` | `SUG_NOSE_SHORT_01` | `SUG_NOSE_LONG_01` |
| `nose_tip_deviation_ratio` | — (omit; prefer retake on high only) | `SUG_NOSE_DEV_HIGH_01` |
| `mouth_width_ratio` | `SUG_MOUTH_NARROW_01` | `SUG_MOUTH_WIDE_01` |
| `mouth_corner_tilt_deg` | `SUG_MOUTH_TILT_LOW_01` | `SUG_MOUTH_TILT_HIGH_01` |
| `lip_thickness_ratio` | `SUG_LIP_THIN_01` | `SUG_LIP_FULL_01` |
| `upper_lip_ratio` | `SUG_UPPER_LIP_LOW_01` | `SUG_UPPER_LIP_HIGH_01` |
| `philtrum_ratio` | `SUG_PHILTRUM_LOW_01` | `SUG_PHILTRUM_HIGH_01` |

**Priority hint for rules (before human re-rank)**

1. Capture issues: `SUG_POSE_RETAKE_01`, `SUG_LIGHTING_01`, `SUG_SYM_HIGH_01`
2. Non-`ok` feature triggers (max one ID per feature)
3. If empty: `SUG_OK_KEEP_01`, optionally `SUG_OK_SKIN_PREP_01`
4. Cap at top-`k` = 4 for product UI
