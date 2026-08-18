# Glow-Mark — AI Engineering Audit, Issues & Solutions Guide

**Audience:** AI engineers / tech leads rebuilding the ML stack  
**Scope:** `app/`, `backend/`, `components/`, `contexts/`, `hooks/`, `lib/`, `store/`  
**Date:** 2026-07-13

---

## 1. Executive verdict

Glow-Mark has a **clear intended ML architecture** (MediaPipe → beauty MLP → geometry recommendation MLP), but the product is **not shippable as an ML system today** because:

1. **`backend/models/` does not exist** — both required checkpoints are missing.
2. **There is no trainable suggestion / advice model** — natural-language “beauty suggestions” are **hardcoded if/else rules** in `lib/analysis/beauty-scorer.ts`.
3. **There is no dataset, no training code, and no label taxonomy** in the repo.
4. The app runs **two incompatible analysis pipelines** (`/analyze` → Flask ML vs `/dashboard` → client rules), so the same photo can produce different scores and advice.

This document lists every material issue found in a full codebase review, then gives a concrete rebuild plan: datasets (with sample rows), training recipes that match the existing checkpoint contracts, and product/architecture fixes.

---

## 2. Current architecture (what the code actually does)

```text
┌─────────────────┐     FormData image      ┌──────────────────────────────┐
│  /analyze page  │ ───────────────────────► │ Flask :5001 POST /analyze    │
└─────────────────┘                          │ MediaPipe FaceMesh (468)     │
                                             │  ├─ 68-pt XY → beauty MLP    │
                                             │  └─ 24 geometry feats → feature │
                                             └──────────────────────────────┘
                                                          ▲
                                                          │ BROKEN: missing .pt files

┌─────────────────┐     client MediaPipe     ┌──────────────────────────────┐
│  /dashboard     │ ───────────────────────► │ beauty-scorer.ts (rules)     │
└─────────────────┘                          │ hardcoded string suggestions │
                                             └──────────────────────────────┘

┌─────────────────┐
│ /api/analyze    │ ──► always HTTP 501 (dead stub)
└─────────────────┘
```

| Path | Landmarks | Beauty score | Suggestions | Status |
|------|-----------|--------------|-------------|--------|
| `/analyze` | Server MediaPipe | PyTorch MLP | Feature MLP labels (`feature: low\|high`) | **Broken — models missing** |
| `/dashboard` | Browser MediaPipe | Geometric heuristics | Hardcoded golden-ratio prose | Works offline |
| `POST /api/analyze` | — | — | — | Always 501 |

---

## 3. Issue register (prioritized)

### P0 — Blocks ML product

| ID | Issue | Where | Impact |
|----|--------|--------|--------|
| **P0-1** | No `backend/models/` directory | Repo root | Inference cannot start |
| **P0-2** | Missing `beauty_landmarks_best.pt` | Expected at `backend/models/beauty_landmarks_best.pt` | Beauty score path crashes on first `/analyze` |
| **P0-3** | Missing `reco_geometry_model.pt` | Expected at `backend/models/reco_geometry_model.pt` | Recommendation path crashes |
| **P0-4** | No suggestion / advice model | Product expectation vs code | Users get either terse labels (`nose_width_ratio: high`) or hardcoded surgical-sounding prose — neither is a trained advice model |
| **P0-5** | No datasets or training pipelines | Entire repo | Models cannot be recreated |

Checkpoint load (CWD-relative, fragile):

```python
# backend/inference.py — _load_models()
torch.load("backend/models/beauty_landmarks_best.pt", ...)
torch.load("backend/models/reco_geometry_model.pt", ...)
```

### P1 — Wrong or misleading ML / geometry

| ID | Issue | Where | Why it matters |
|----|--------|--------|----------------|
| **P1-1** | Dual scoring systems | `/analyze` vs `/dashboard` | Same face → different score & advice |
| **P1-2** | Suggestions hardcoded | `lib/analysis/beauty-scorer.ts` → `generateRecommendations()` | Not ML; not personalizable; not data-driven |
| **P1-3** | Backend feature is class labels only | `inference.py` builds `"label: low\|high"` | Not user-facing advice |
| **P1-4** | `jaw_width_ratio ≈ 1.0` always | `inference.py` (`face_width / face_width`) | Dead feature; any label on it is noise |
| **P1-5** | `upper_lip_ratio` == `philtrum_ratio` | Same formula twice | Duplicate / useless channel |
| **P1-6** | Lip ratio unit bug (client) | Measured `upper/lower ≈ 0.618` vs ideal `1.618` | Almost always flags lips incorrectly |
| **P1-7** | Jaw width bug (client) | Jaw = face left/right → ratio ~1 | Jaw recommendations nonsense |
| **P1-8** | Beauty features use raw pixel XY | 68 landmarks flattened to 136 | Resolution/crop sensitive unless training matched |
| **P1-9** | No face pose gate | Both pipelines | Angled faces still scored |
| **P1-10** | Fake “Insights” copy | `results-dashboard.tsx` | Always praises profile regardless of score |

### P2 — Engineering / product / security

| ID | Issue | Notes |
|----|--------|--------|
| **P2-1** | TypeScript contract drift | `app/analyze/page.tsx` imports `BackendAnalyzeResponse` / `recommendation_items`; `lib/types.ts` does not define them |
| **P2-2** | Next `/api/analyze` dead | Returns 501; README still describes mock API |
| **P2-3** | Windows-only npm backend scripts | `backend\\.venv\\Scripts\\python` breaks on macOS/Linux |
| **P2-4** | Model paths not `__file__`-relative | Fail if CWD ≠ repo root |
| **P2-5** | `/health` does not verify models | Green health while analyze will 500 |
| **P2-6** | Flask: open CORS, no auth, `debug=True`, bind `0.0.0.0` | Anyone can POST faces; abuse / privacy risk |
| **P2-7** | Privacy copy says “browser-only / never uploaded” | `/analyze` uploads image to Flask; Firestore persists scores |
| **P2-8** | Landmarks not saved to Firestore | History cannot restore mesh overlays |
| **P2-9** | No graceful fallback | When backend/models fail, client scorer is not used on `/analyze` |
| **P2-10** | README outdated | Documents mock API; omits Flask, Firebase, model setup |

---

## 4. What the existing checkpoint contracts expect

You must train **new** models that match what `inference.py` already loads. Do not invent a different schema without updating inference + frontend together.

### 4.1 Beauty model — `beauty_landmarks_best.pt`

| Field | Type | Meaning |
|-------|------|---------|
| `in_dim` | `int` | Must be **136** (= 68 landmarks × 2 XY) with current code |
| `mu`, `sd` | vectors length `in_dim` | Feature standardization |
| `model_state` | `state_dict` | Sequential: `Linear(136→256) → ReLU → Dropout(0.2) → Linear(256→256) → ReLU → Dropout(0.2) → Linear(256→1)` |

**Input construction (must match serve time):**

1. MediaPipe FaceMesh → 468 points  
2. Subsample with `MP_468_TO_68` in `inference.py`  
3. Flatten pixel XY → shape `(N, 136)`  
4. `(x - mu) / sd` → MLP → scalar clipped to `[0, 100]` in serve code  

**Strong recommendation before training:** switch training **and** inference to **face-crop + scale-normalized** coordinates (e.g. bbox → fixed 256×256, center/scale), then retrain. Raw pixels are a liability.

### 4.2 Geometry recommendation model — `reco_geometry_model.pt`

| Field | Type | Meaning |
|-------|------|---------|
| `feat_cols` | `list[str]` | Ordered feature names (expect the 24 keys below) |
| `label_cols` | `list[str]` | Ordered label names (typically same 24 names) |
| `mu`, `sd` | length = `len(feat_cols)` | Feature standardization |
| `state` | `state_dict` | Sequential: `Linear → ReLU → Linear → ReLU → Linear` → **72 outs** (= 24 × 3) |

**Classes:** `low`, `ok`, `high` per label (softmax over 3 logits each).

**Feature keys produced by `_extract_geometry_features`:**

```text
symmetry_error, face_aspect_ratio, midface_length_ratio, lowerface_length_ratio,
jaw_width_ratio, jaw_angle_sharpness, chin_length_ratio, chin_width_ratio,
cheekbone_width_ratio, lower_cheek_ratio, eye_openness_ratio, eye_size_ratio,
eye_spacing_ratio, eye_tilt_deg, brow_height_ratio, brow_tilt_deg,
nose_width_ratio, nose_length_ratio, nose_tip_deviation_ratio, mouth_width_ratio,
mouth_corner_tilt_deg, lip_thickness_ratio, upper_lip_ratio, philtrum_ratio
```

**Fix before collecting labels:** redefine `jaw_width_ratio` (true jaw landmarks, not face width) and split `upper_lip_ratio` vs `philtrum_ratio` (different landmark pairs). Retrain after the fix.

### 4.3 What is *not* covered by those two models

Neither checkpoint produces:

- Natural-language advice (“consider contouring to soften jaw width…”)  
- Procedure recommendations  
- Skincare suggestions  
- Personalized coaching  

That is a **third model** (or a retrieval + LLM layer) — see §6.

---

## 5. Solution plan — recreate the two missing models

### Phase 0 — Repo hygiene (1 day)

1. Create `backend/models/` and add `backend/models/.gitkeep`.  
2. Add to `.gitignore`: `backend/models/*.pt` (keep weights out of git; ship via release artifact / LFS / object storage).  
3. Change load paths to be relative to `__file__`:

```python
from pathlib import Path
MODELS_DIR = Path(__file__).resolve().parent / "models"
beauty_ckpt = torch.load(MODELS_DIR / "beauty_landmarks_best.pt", map_location="cpu", weights_only=False)
```

4. Make `/health` attempt a dry `_load_models()` (or check files exist) so ops know models are present.  
5. Fix `package.json` backend scripts for Unix: `backend/.venv/bin/python`.

### Phase 1 — Fix geometry bugs (before any labeling)

| Bug | Fix |
|-----|-----|
| `jaw_width_ratio` | Use true jaw landmarks (e.g. distance between jaw angles / face width) |
| Identical lip/philtrum | Philtrum = nose base → upper lip; upper lip = vermilion height / face height |
| Client lip 0.618 vs 1.618 | Compare like units (ratio vs ratio, or convert ideal to lower/upper) |
| Client jaw | Same landmark fix as backend |

Commit these formulas; freeze them as the **feature contract** for all datasets.

### Phase 2 — Build Dataset A: Beauty score regression

**Goal:** Train `beauty_landmarks_best.pt`.

#### Recommended data sources (legal / ethical)

You need **faces + continuous beauty-related targets**. Options ranked by practicality:

| Option | Pros | Cons |
|--------|------|------|
| **A. Public face datasets + synthetic / proxy labels** | Scalable | Proxy ≠ true aesthetics |
| **B. Crowd ratings on licensed faces** (your own collection) | Best product fit | Cost, ethics, IRB-like review |
| **C. Relative ranking (Bradley–Terry)** | More stable than absolute 0–100 | Needs pair UI |
| **D. Do not train absolute “beauty”** — train **geometry typicality / symmetry only** | Less harmful framing | Product copy must change |

**World-class recommendation:** Prefer **relative preference learning** or **multi-metric aesthetic attributes** (symmetry, proportion typicality, lighting quality) over a single “beauty” score. If the product must keep 0–100, treat it as **“alignment score to a declared geometric prior”**, not universal attractiveness — and update UI copy accordingly.

#### Dataset A schema (CSV / Parquet)

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Unique ID |
| `image_path` | string | Path to face image |
| `split` | string | `train` / `val` / `test` |
| `lm_x_00` … `lm_x_67`, `lm_y_00` … `lm_y_67` | float | 68-point subset after alignment (normalized preferred) |
| `beauty_score` | float | Target in `[0, 100]` (or `[0, 1]` then scale) |
| `rater_count` | int | Optional: number of raters |
| `score_std` | float | Optional: rater disagreement |
| `yaw_deg`, `pitch_deg` | float | Optional: filter pose |
| `consent_flag` | bool | Required for any proprietary faces |
| `source` | string | Dataset provenance |

#### Sample rows — Dataset A (Beauty) — 6 examples

> Values are **illustrative** (realistic ranges). In production, landmarks come from MediaPipe + `MP_468_TO_68` after your chosen normalization.

| sample_id | image_path | split | lm_x_00 | lm_y_00 | … | lm_x_67 | lm_y_67 | beauty_score | rater_count | score_std | yaw_deg | pitch_deg | consent_flag | source |
|-----------|------------|-------|---------|---------|---|---------|---------|--------------|-------------|-----------|---------|-----------|--------------|--------|
| bty_0001 | data/raw/faces/0001.jpg | train | 0.312 | 0.401 | … | 0.688 | 0.742 | 72.4 | 7 | 4.1 | 2.1 | -1.3 | true | proprietary_v1 |
| bty_0002 | data/raw/faces/0002.jpg | train | 0.298 | 0.388 | … | 0.701 | 0.755 | 58.0 | 5 | 6.8 | -4.0 | 3.2 | true | proprietary_v1 |
| bty_0003 | data/raw/faces/0003.jpg | train | 0.305 | 0.410 | … | 0.695 | 0.738 | 81.2 | 9 | 3.0 | 0.5 | 0.8 | true | proprietary_v1 |
| bty_0004 | data/raw/faces/0004.jpg | val | 0.320 | 0.395 | … | 0.680 | 0.760 | 64.5 | 6 | 5.2 | 1.8 | -2.0 | true | proprietary_v1 |
| bty_0005 | data/raw/faces/0005.jpg | val | 0.291 | 0.420 | … | 0.710 | 0.748 | 45.8 | 8 | 7.5 | 6.2 | 4.1 | true | proprietary_v1 |
| bty_0006 | data/raw/faces/0006.jpg | test | 0.308 | 0.405 | … | 0.692 | 0.745 | 76.9 | 7 | 3.6 | -1.1 | 1.0 | true | proprietary_v1 |

**Minimum viable size:** ~5k–20k faces with ≥3 raters each (or ≥50k ranked pairs). Below that, the MLP will overfit and look “confident but wrong.”

#### Training recipe (matches checkpoint)

```text
1. Extract MediaPipe 468 → map to 68 → normalize (crop/scale)
2. Flatten to 136-d vector
3. Fit mu/sd on train split only
4. Train MLP (same architecture as _mlp_beauty) with MSE or SmoothL1 on beauty_score
5. Early-stop on val MAE
6. Export:
   torch.save({
     "in_dim": 136,
     "mu": mu,
     "sd": sd,
     "model_state": model.state_dict(),
   }, "backend/models/beauty_landmarks_best.pt")
```

Suggested folder:

```text
backend/training/
  extract_landmarks.py
  build_beauty_dataset.py
  train_beauty.py
  export_beauty_ckpt.py
```

---

### Phase 3 — Build Dataset B: Geometry low / ok / high classifier

**Goal:** Train `reco_geometry_model.pt`.

This model does **not** invent advice text. It classifies each geometry channel as **below / within / above** an acceptable band relative to a population prior (or expert-defined bands).

#### Labeling strategies (pick one)

1. **Population percentiles (recommended bootstrap):**  
   On a large unlabeled face set, compute the 24 features. For each feature:  
   - `ok` = between P20–P80 (or P15–P85)  
   - `low` = below lower band  
   - `high` = above upper band  

2. **Expert bands:** Plastic-surgery / aesthetics literature ranges (document every threshold).  

3. **Hybrid:** Population bands + manual review on edge cases.

#### Dataset B schema

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Unique ID |
| `image_path` | string | Image path |
| `split` | train/val/test | |
| `symmetry_error` … `philtrum_ratio` | float | All 24 features |
| `y_symmetry_error` … `y_philtrum_ratio` | string | Each ∈ {`low`,`ok`,`high`} |
| `label_method` | string | `percentile_p20_p80` / `expert_v1` |
| `consent_flag` | bool | |

#### Sample rows — Dataset B (Recommendation geometry) — 6 examples

Abbreviated feature set shown for readability; real CSV must include **all 24 features + 24 labels**.

| sample_id | split | face_aspect_ratio | nose_width_ratio | eye_spacing_ratio | mouth_width_ratio | lip_thickness_ratio | symmetry_error | … | y_face_aspect_ratio | y_nose_width_ratio | y_eye_spacing_ratio | y_mouth_width_ratio | y_lip_thickness_ratio | y_symmetry_error | … | label_method |
|-----------|-------|-------------------|------------------|-------------------|-------------------|---------------------|----------------|---|---------------------|--------------------|---------------------|---------------------|-----------------------|------------------|---|--------------|
| geo_0001 | train | 0.78 | 0.24 | 0.92 | 0.48 | 0.031 | 0.012 | … | high | low | ok | ok | ok | ok | … | percentile_p20_p80 |
| geo_0002 | train | 0.65 | 0.31 | 1.22 | 0.56 | 0.028 | 0.041 | … | ok | high | high | high | ok | high | … | percentile_p20_p80 |
| geo_0003 | train | 0.61 | 0.27 | 1.01 | 0.50 | 0.035 | 0.009 | … | ok | ok | ok | ok | ok | ok | … | percentile_p20_p80 |
| geo_0004 | val | 0.72 | 0.29 | 0.81 | 0.42 | 0.022 | 0.018 | … | high | ok | low | low | low | ok | … | percentile_p20_p80 |
| geo_0005 | val | 0.58 | 0.22 | 1.05 | 0.51 | 0.040 | 0.055 | … | low | low | ok | ok | high | high | … | percentile_p20_p80 |
| geo_0006 | test | 0.66 | 0.275 | 0.98 | 0.49 | 0.033 | 0.011 | … | ok | ok | ok | ok | ok | ok | … | percentile_p20_p80 |

#### Training recipe (matches checkpoint)

```text
1. Extract 24 features with the FIXED formulas
2. One-hot / class index: low=0, ok=1, high=2 per label
3. Fit mu/sd on train features
4. Train MLP to 72 logits; loss = sum of 24 CrossEntropies (or MultiLabelCE)
5. Export:
   torch.save({
     "feat_cols": feat_cols,      # length 24
     "label_cols": label_cols,    # length 24
     "mu": mu,
     "sd": sd,
     "state": model.state_dict(),  # note: key is "state", not "model_state"
   }, "backend/models/reco_geometry_model.pt")
```

---

## 6. Solution plan — suggestion / advice model (the missing third model)

### Problem statement

Today:

- **Dashboard:** long hardcoded strings in `generateRecommendations()` (golden ratio thresholds).  
- **Backend:** `"nose_width_ratio: high"` — not product-ready advice.  

You want a **model (or structured generation system)** that turns geometry state → **actionable, safe, non-clinical suggestions**.

### Recommended architecture (pragmatic + world-class)

Do **not** start with an end-to-end LLM that invents medical advice. Use a **constrained pipeline**:

```text
Geometry features + feature classes (low/ok/high)
        │
        ▼
  Suggestion Ranking Model (or rules→ranker)
        │
        ▼
  Template / retrieval library (approved copy only)
        │
        ▼
  Optional LLM rewrite (style only, no new claims)
        │
        ▼
  Safety filter (no surgical guarantees, no medical claims)
```

### Phase 4 — Build Dataset C: Suggestion mapping

Each row maps a **geometry state** to **approved suggestion IDs** (multi-label).

#### Suggestion catalog (example entries you author once)

| suggestion_id | category | severity | approved_text |
|---------------|----------|----------|---------------|
| SUG_NOSE_WIDE_01 | nose | mild | “Nose width reads above your population band. Soft contouring along the nasal sidewalls can create a narrower visual line in photos.” |
| SUG_EYE_CLOSE_01 | eyes | mild | “Eye spacing reads slightly close. Highlighting the outer eye corners can balance the look in makeup applications.” |
| SUG_JAW_WIDE_01 | jaw | mild | “Jaw width sits above the typical band. Hairstyles with volume at the crown often balance a stronger jaw visually.” |
| SUG_SYM_HIGH_01 | symmetry | info | “Left–right alignment shows elevated asymmetry. Retake with face centered and even lighting before drawing conclusions.” |
| SUG_LIP_THIN_01 | lips | mild | “Lip thickness is below the typical band. Lip-focused makeup (liner + gloss) can enhance perceived fullness.” |
| SUG_OK_KEEP_01 | general | info | “Most measured proportions sit in a typical range. Maintain consistent lighting and a front-facing pose for tracking.” |

**Critical product rule:** Copy must be **cosmetic / photographic / styling**, not medical or surgical instructions — unless you have clinical partners and legal review.

#### Dataset C schema

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Links to Dataset B sample |
| `feature_vector_json` | string/json | 24 floats |
| `class_vector_json` | string/json | 24 classes |
| `suggestion_ids` | string | Pipe-separated IDs, e.g. `SUG_NOSE_WIDE_01\|SUG_SYM_HIGH_01` |
| `priority_order` | string | Ordered IDs for ranking |
| `annotator_id` | string | Who labeled |
| `split` | train/val/test | |

#### Sample rows — Dataset C (Suggestions) — 6 examples

| sample_id | nose_width_ratio | y_nose_width_ratio | eye_spacing_ratio | y_eye_spacing_ratio | mouth_width_ratio | y_mouth_width_ratio | symmetry_error | y_symmetry_error | suggestion_ids | priority_order | annotator_id | split |
|-----------|------------------|--------------------|-------------------|---------------------|-------------------|---------------------|----------------|------------------|----------------|----------------|--------------|-------|
| sug_0001 | 0.31 | high | 1.01 | ok | 0.50 | ok | 0.010 | ok | SUG_NOSE_WIDE_01 | SUG_NOSE_WIDE_01 | ann_a | train |
| sug_0002 | 0.27 | ok | 0.80 | low | 0.49 | ok | 0.012 | ok | SUG_EYE_CLOSE_01 | SUG_EYE_CLOSE_01 | ann_a | train |
| sug_0003 | 0.28 | ok | 0.99 | ok | 0.57 | high | 0.048 | high | SUG_MOUTH_WIDE_01\|SUG_SYM_HIGH_01 | SUG_SYM_HIGH_01\|SUG_MOUTH_WIDE_01 | ann_b | train |
| sug_0004 | 0.22 | low | 1.20 | high | 0.43 | low | 0.015 | ok | SUG_NOSE_NARROW_01\|SUG_EYE_WIDE_01\|SUG_MOUTH_NARROW_01 | SUG_EYE_WIDE_01\|SUG_NOSE_NARROW_01\|SUG_MOUTH_NARROW_01 | ann_b | val |
| sug_0005 | 0.275 | ok | 1.00 | ok | 0.50 | ok | 0.008 | ok | SUG_OK_KEEP_01 | SUG_OK_KEEP_01 | ann_a | val |
| sug_0006 | 0.30 | high | 0.95 | ok | 0.52 | ok | 0.022 | ok | SUG_NOSE_WIDE_01\|SUG_LIP_THIN_01 | SUG_NOSE_WIDE_01\|SUG_LIP_THIN_01 | ann_c | test |

#### How to create Dataset C without a large public “beauty advice” corpus

You will not find a clean public dataset for surgical/beauty suggestions. Build it yourself:

1. Freeze a **suggestion catalog** of 40–80 approved templates (spreadsheet).  
2. Write **deterministic mapping rules** first (e.g. `nose_width_ratio == high` → candidate `SUG_NOSE_WIDE_*`).  
3. Have 2–3 annotators **rank / select** top-k suggestions per face (or per class vector).  
4. Train a small **multi-label classifier** or **learning-to-rank** model on top of the 24 features + 24 class one-hots.  
5. At serve time: model ranks → fill templates with measured numbers → optional LLM polish under a strict system prompt.

**Bootstrap path (week 1):** ship **template mapping from feature classes** (no NN yet). That already replaces the messy golden-ratio prose and unifies `/analyze` + `/dashboard`. Then collect Dataset C while users run the product (with consent).

#### Minimal “suggestion model” checkpoint (optional future)

If you want a PyTorch artifact similar to the others:

```text
Input:  24 features (+ optional 24×3 class probs) → ~96-d
Output: logits over K suggestion IDs (multi-label BCE)
Save:   suggestion_ranker.pt with {feat_mu, feat_sd, suggestion_ids, state}
```

Wire a new field in the API, e.g. `suggestions: [{id, text, confidence}]`, and stop returning raw `"label: high"` strings to the UI.

---

## 7. How to create the datasets from scratch (practical pipeline)

```text
data/
  raw/images/                 # consented or licensed faces
  interim/landmarks_468/      # .npz per image
  interim/landmarks_68/       # mapped subset
  processed/
    beauty_dataset.parquet    # Dataset A
    geometry_dataset.parquet  # Dataset B
    suggestion_dataset.parquet# Dataset C
  catalogs/
    suggestions.csv           # approved copy
  docs/
    labeling_guide.md
    ethics_consent.md
```

### Step-by-step

1. **Acquire images legally**  
   - Prefer your own consented uploads (product opt-in).  
   - Or licensed research sets with redistribution rights for derived landmarks (check licenses carefully).  
   - Filter: single face, frontal (|yaw|<15°, |pitch|<15°), min resolution, no heavy occlusion.

2. **Run landmark extraction** (same MediaPipe settings as production: static image, refine_landmarks consistent with train).

3. **Build Dataset B automatically** via percentile bands on a large pool (no human beauty labels needed).

4. **Build Dataset A** via:  
   - Preference pairs UI, or  
   - Attribute scores (symmetry/proportion quality) averaged into a composite, or  
   - Explicit “alignment to geometric prior” score derived from Dataset B (count of `ok` labels → 0–100).  
   The last option is the **fastest ethically safer bootstrap**: beauty_score ≈ f(# of ok geometry channels, symmetry).

5. **Build Dataset C** via catalog + annotator ranking (or pure rules as v0).

6. **Train → export → place `.pt` files in `backend/models/` → integration test `/analyze`.**

### Bootstrap beauty score without subjective ratings

If you have no raters yet:

```text
beauty_score_proxy = 100 * mean( exp(-|z_f|) for f in feat_cols )
```

Train the beauty MLP to regress this proxy from 68-pt landmarks. It will be **correlated with geometry typicality**, not “true beauty,” but it unblocks the missing `.pt` and keeps the product honest if you rename the metric (e.g. “Geometric Balance Score”).

---

## 8. Unify the product (required after models exist)

| Action | Detail |
|--------|--------|
| Single pipeline | `/analyze` and `/dashboard` both call Flask (or both call one shared TS client that hits Flask) |
| Remove dual scores | Deprecate `calculateAestheticScore` for production scoring; keep as offline fallback only |
| Replace hardcoded suggestions | `generateRecommendations()` → template/ranker outputs |
| Fix types | Add `BackendAnalyzeResponse`, `recommendation_items`, `suggestions` to `lib/types.ts` |
| Render structured items | UI should show confidence + approved text, not only string list |
| Fix Insights | Drive copy from real metrics; delete always-positive filler |
| Auth on Flask | Verify Firebase ID token on `/analyze` |
| Docs / privacy | Rewrite README + privacy to match upload-to-backend + Firestore |

---

## 9. Evaluation checklist (do not ship without this)

### Beauty model

- [ ] Val MAE / RMSE on held-out faces  
- [ ] Calibration: predicted vs true by score buckets  
- [ ] Stability: same face, mild crop/jpeg → score Δ < threshold  
- [ ] Pose stress test: reject or down-weight high yaw/pitch  

### Feature model

- [ ] Per-label accuracy / F1 for `low|ok|high`  
- [ ] Confusion mostly adjacent (`low↔ok`, not `low↔high`)  
- [ ] Dead features removed (`jaw_width`, duplicate philtrum) before measuring  

### Suggestion system

- [ ] 100% of emitted text comes from approved catalog (or LLM rewrite of approved text)  
- [ ] Safety review: no medical guarantees, no demographic stereotyping  
- [ ] Human preference: ranked suggestions judged useful ≥ X%  

### Integration

- [ ] `/health` fails if `.pt` missing  
- [ ] `/analyze` e2e with sample images  
- [ ] Frontend types compile; dashboard history consistent  

---

## 10. Suggested 4-week execution plan

| Week | Deliverable |
|------|-------------|
| **1** | Fix geometry bugs; `__file__` model paths; health check; Unix scripts; create `backend/models/`; data folder layout; suggestion catalog v0 (40 templates) |
| **2** | Extract landmarks on 5k–20k faces; build Dataset B via percentiles; train + export `reco_geometry_model.pt`; wire template suggestions from classes |
| **3** | Build Dataset A (proxy or ratings); train + export `beauty_landmarks_best.pt`; unify `/analyze` + `/dashboard`; fix types/UI |
| **4** | Collect Dataset C rankings; train suggestion ranker (or ship rules v1); security (auth/CORS); rewrite README/privacy; evaluation report |

---

## 11. Immediate unblock (minimum to make `/analyze` run)

If you need a temporary local demo **before** real training:

1. Train tiny overfit models on 50–100 synthetic/proxy-labeled faces using the exact export schemas in §5.  
2. Place files at:

```text
backend/models/beauty_landmarks_best.pt
backend/models/reco_geometry_model.pt
```

3. Run Flask from repo root (or after `__file__` path fix).  
4. Call `POST /analyze` with a frontal face image.

**Do not** ship overfit demo weights as production quality.

---

## 12. Summary of solutions mapped to your identified issues

| You identified | Solution |
|----------------|----------|
| No `beauty_landmarks_best.pt` | Create Dataset A → train MLP → export checkpoint matching §4.1 |
| No `reco_geometry_model.pt` | Create Dataset B → train 24×3 classifier → export matching §4.2 |
| No `/backend/models` folder | Create it; gitignore `*.pt`; load via `__file__` |
| No model for beauty suggestions; hardcoded in `beauty-scorer.ts` | Build suggestion catalog + Dataset C → template mapper / ranker; retire hardcoded `generateRecommendations` for production |
| No suitable advice dataset | You must create it: features + classes → approved `suggestion_ids` (samples in §6) |

---

## 13. Key file reference

| Path | Role |
|------|------|
| `backend/inference.py` | MediaPipe + beauty/feature load & serve |
| `backend/app.py` | Flask `/analyze`, `/health` |
| `lib/analysis/beauty-scorer.ts` | Client heuristics + hardcoded suggestions |
| `app/analyze/page.tsx` | Production path → Flask |
| `app/dashboard/page.tsx` | Legacy client path |
| `app/api/analyze/route.ts` | Dead 501 stub |
| `lib/types.ts` | Incomplete API types |
| `components/results-dashboard.tsx` | Score / ratios / recs UI |
| `lib/firebase/analysis.ts` | Persistence without landmarks |

---

## 14. Ethics note (non-optional for a world-class team)

Attractiveness scoring systems can cause harm. Before marketing “beauty AI”:

- Frame scores as **geometric / photographic metrics**, not human worth.  
- Avoid demographic targeting or “ideal race/age” priors.  
- Prefer advice that is **styling / capture / makeup**, not medical.  
- Store consent, allow deletion, and be honest when images leave the device.

---

*This document is the engineering source of truth for rebuilding Glow-Mark’s ML stack from the current broken state (missing weights, dual pipelines, hardcoded suggestions) into a reproducible train → export → serve system with three datasets: beauty regression, geometry classification, and suggestion ranking.*
