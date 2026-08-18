# Dataset C quality report

**Status:** PASS

## Fails
- (none)

## Warns
- identity_map_missing: flat lfw_##### IDs — cannot verify person-level leakage
- balance: 36 suggestion_ids have <50 positives (train with pos_weight; expand data later). Examples: [('SUG_PHILTRUM_LOW_01', 1), ('SUG_OK_KEEP_01', 1), ('SUG_OK_SKIN_PREP_01', 1), ('SUG_UPPER_LIP_LOW_01', 3), ('SUG_NOSE_LONG_01', 4), ('SUG_MOUTH_TILT_HIGH_01', 6), ('SUG_PHILTRUM_HIGH_01', 6), ('SUG_LIP_FULL_01', 7), ('SUG_LIP_THIN_01', 7), ('SUG_MOUTH_NARROW_01', 7)]

## Info
- validate_dataset_schema.py OK
- near-duplicate conflicting-label check OK
- spotcheck export: /Users/tharushasamarawickrama/Downloads/glow-mark/data/labeling/spotcheck/spotcheck_100.csv (100 rows)

## Summary
- rows: 412
- unique suggestion_ids: 48
- splits: {'test': 38, 'train': 326, 'val': 48}

