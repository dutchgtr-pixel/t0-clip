# Fusion Store (Damage Fusion v2) — Full Documentation + Registry Certification

**Feature store name (reporting):** `fusion_store`  
**Primary modeling surface:** `ml.v_damage_fusion_features_v2_scored`  
**Certified entrypoint (registry):** `ml.fusion_feature_store_t0_v1_v`  
**Guarded consumption view (trainer should use):** `ml.v_damage_fusion_features_v2_scored_train_v`

This package documents, from first principles, how the Fusion Store is built, what every layer does, the complete SQL used to create it, and the exact steps used to certify it into your `audit.t0_cert_registry` with strict guard enforcement.

It is written so you can recreate the entire fusion layer on a fresh database, validate correctness, and re-run certification without reverse‑engineering your history.

---

## 1. Executive summary

The Fusion Store solves a specific modeling failure mode: “damage = 0” and “damage = unknown” are not the same, and treating them as the same destroys survival model performance.

You have three partially overlapping signals:

1. **Image-based damage signal** (good when present, but coverage varies; stock photos exist; quality varies)
2. **Text-based decision + meta flags** (good semantics, but can be inconsistent; some flags imply minimum severity)
3. **Battery health** (can be extracted from text and/or from screenshots)

The fusion layer provides:

- **Signal:** damage severity and battery estimate
- **Reliability:** evidence strength and image quality context
- **Missingness semantics:** explicit “why missing” codes so missingness is learnable

The core design decision is: models must be able to distinguish “clean with strong evidence” from “clean because we saw nothing.”

---

## 2. Objects and their roles

### 2.1 Core views (feature layer)

1) **Raw fusion + diagnostics**
- `ml.v_damage_fusion_features_v2`

2) **Model-safe wrapper**
- `ml.v_damage_fusion_features_v2_scored`

The wrapper adds “known/scored” semantics so missingness never silently masquerades as a clean phone.

### 2.2 Certification/guard surfaces (registry layer)

3) **Certified entrypoint**
- `ml.fusion_feature_store_t0_v1_v`
- Adds `edited_date` for dataset hashing by joining to `"iPhone".iphone_listings`

4) **Guarded consumption view**
- `ml.v_damage_fusion_features_v2_scored_train_v`
- Fails closed unless the entrypoint is registry-certified

---

## 3. Join key contract

All fusion outputs are keyed strictly on:

**(generation, listing_id)**

This is the join contract expected by your wider feature store and by the slow21 trainer.

---

## 4. Upstream sources

### 4.1 Listing source: `"iPhone".iphone_listings`

Required fields:
- `generation`, `listing_id`
- `spam` (used to exclude spam rows)
- `damage_ai_json` (JSON: decision, meta flags, battery_effective)

### 4.2 Image feature source: `ml.iphone_image_features_v1`

Used for:
- pipeline coverage (`feature_version`, `damage_done`)
- stock detection (`is_stock_photo`)
- quality (`photo_quality_level`, `background_clean_level`)
- damage (`visible_damage_level`, `damage_on_protector_only`)
- battery screenshot extraction (`battery_screenshot`, `battery_health_pct_img`)

---

## 5. Fusion view architecture (v2)

`ml.v_damage_fusion_features_v2` is constructed in layers:

### 5.1 `img_prof` — listing-level image profile aggregation

Outputs:
- Coverage:
  - `img_rows_v1`
  - `img_rows_damage_done`
- Stock vs real:
  - `img_stock_cnt`, `img_real_cnt`, `img_stock_ratio`
- Image reliability:
  - photo quality histogram: `img_pq0_cnt … img_pq4_cnt`, mean `img_pq_avg_real`
  - background histogram: `img_bg0_cnt … img_bg2_cnt`, mean `img_bg_avg_real`
- Evidence counts and maxima:
  - `img_real_q2_cnt`, `img_real_q2_vis_cnt`, `img_real_q1_vis_cnt`
  - `img_max_eff_q2`, `img_max_eff_q1`
  - `img_max_noprotector_q2`
  - evidence strength counts: `img_n_ge3_real_q2`, `img_n_ge8_real_q2`

Key gating idea: Q2 (“good quality”) requires `photo_quality_level >= 2`.

### 5.2 `batt_prof` — battery screenshot aggregation

Outputs:
- `batt_img_min`, `batt_img_max`, `batt_img_hits`
- `batt_img_spread = batt_img_max - batt_img_min`

Design choice: use the **minimum** screenshot battery as the most conservative estimate.

### 5.3 `base` — join listing + image profile + battery profile + text decision/meta

- Joins `"iPhone".iphone_listings` (spam filtered) to `img_prof` and `batt_prof`
- Parses numeric fields from JSON robustly using regex guards:
  - `batt_eff_text` from `damage_ai_json->>'battery_effective'`
  - `text_sev_decision`, `text_bin_decision` from `damage_ai_json->'decision'`
- Extracts meta flags (booleans) from `damage_ai_json->'decision'->'meta'`:
  - `txt_no_wear_global`, `txt_protector_only`
  - `txt_glass`, `txt_back_glass`, `txt_lens_glass`, `txt_panel_severe`
  - `txt_light_panel`, `txt_charging`, `txt_non_oem`, `txt_battery_clamp`

Also defines:
- `text_damage_scored` = decision present

### 5.4 `derived` — deterministic fusion logic

This is where the feature-store logic lives.

#### Battery fusion
- `batt_fused = COALESCE(batt_img_min, batt_eff_text)`
- `batt_source ∈ {both, img_only, text_only, none}`
- `batt_bucket` categorical bucket
- convenience flags: `batt_lt80`, `batt_ge90`

#### Image damage scoring source selection
You require **at least 2** usable images with a visible damage score:
- `img_damage_source_level`:
  - 2 = Q2 primary if `img_real_q2_vis_cnt >= 2`
  - 1 = Q1 fallback if `img_real_q1_vis_cnt >= 2`
  - 0 = not scored
- `img_damage_score`:
  - Q2 max if Q2 evidence sufficient
  - else Q1 max if Q1 evidence sufficient
  - else NULL

#### Missingness semantics: `img_score_reason_code`
Explains why image scoring is missing or degraded:
1. no image rows
2. pipeline never completed
3. all stock
4. no Q2-quality real photos
5. no Q2 damage values
6. insufficient Q2 evidence (<2)
7. Q2 insufficient but Q1 has >=2 (fallback)
0. scored normally

#### Text severity correction: `text_sev_corr`
Deterministic floors based on meta + battery:
- baseline = `COALESCE(text_sev_decision, 0)`
- floor ≥2 if:
  - battery known and <80, OR
  - `txt_battery_clamp` OR `txt_non_oem` OR `txt_charging`
- floor ≥3 if:
  - `txt_glass OR txt_back_glass OR txt_lens_glass OR txt_panel_severe`
- `text_sev_corr = GREATEST(baseline, floor2, floor3)`
Only increases severity; never lowers.

#### Final fused tier: `damage_fused_tier ∈ {0,1,2,3}`
Coarse, stable categorical summary:
- Tier 3 (structural) if image structural evidence strong OR corrected text ≥3
- Tier 2 if image score ≥3
- Tier 1 if image score =2 OR corrected text ≥2
- Tier 0 otherwise

---

## 6. v2_scored wrapper: modeling-safe semantics

`ml.v_damage_fusion_features_v2_scored` adds:

- `img_damage_scored = (img_damage_source_level >= 1)`
- `batt_scored = (batt_fused IS NOT NULL)`
- `damage_any_scored = img_scored OR text_scored OR batt_scored`

And “known-only” columns:
- `text_sev_raw_known` only when `text_damage_scored`
- `text_sev_corr_known` only when `text_damage_scored OR batt_scored`
- `damage_fused_tier_known` only when any evidence exists

This prevents “unknown” being treated as zero.

---

## 7. Integration into slow21 trainer

### 7.1 Join pattern

Join on `(generation,listing_id)`:

```sql
LEFT JOIN ml.v_damage_fusion_features_v2_scored_train_v df
  ON df.generation = base.generation
 AND df.listing_id    = base.listing_id
```

### 7.2 Minimum recommended features

- `fusion_img_damage_score`
- `fusion_img_damage_scored::int`
- `fusion_img_n_ge3_real_q2`
- `fusion_img_n_ge8_real_q2`
- `fusion_img_stock_ratio`
- `fusion_img_pq_avg_real`
- `fusion_img_bg_avg_real`
- `fusion_img_score_reason_code`
- `fusion_batt_fused`
- `fusion_batt_scored::int`
- `fusion_text_sev_corr_known`
- `fusion_text_damage_scored::int`
- `fusion_damage_fused_tier_known`
- `fusion_damage_any_scored::int`

### 7.3 Important modeling rule

Do **not** coalesce unknown to 0 without also including the scored flags and reason codes.

---

## 8. Registry certification (leak-proof + guard enforced)

### 8.1 What “certified” means here

The fusion store is certified when:

- closure contains **no time-of-query tokens** (no `CURRENT_DATE`, `now()`, etc.)
- fusion outputs contain **no timestamp columns**
- scored fusion view is **unique per (generation,listing_id)**
- viewdefs are baselined (Gate A)
- dataset hashes are baselined (Gate C)
- entrypoint is marked `CERTIFIED` in `audit.t0_cert_registry`
- `audit.require_certified_strict(entrypoint, max_age)` passes
- guarded consumption view returns rows (fails closed when uncertified)

### 8.2 Certified entrypoint and guarded view

- `ml.fusion_feature_store_t0_v1_v` is the entrypoint used for hashing and registry
- `ml.v_damage_fusion_features_v2_scored_train_v` is the view your trainer should read from

### 8.3 Drift policy

Fusion can drift due to retroactive enrichment/backfills. Certification treats dataset drift as **allowed**, but records it in `audit.t0_cert_registry.notes` and can optionally update baselines for drifted days.

---

## 9. Operational runbook insertion

Daily after upstream pipelines run:

```sql
CALL audit.run_t0_cert_fusion_store_v1(3, true);
SELECT audit.require_certified_strict('ml.fusion_feature_store_t0_v1_v', interval '24 hours');
```

Use `p_check_days=3` daily for speed; use 30 for weekly deeper checks.

---

## 10. Validation queries

### 10.1 Coverage / scoring rates

```sql
SELECT
  AVG((img_damage_scored)::int)::float8 AS pct_img_scored,
  AVG((text_damage_scored)::int)::float8 AS pct_text_scored,
  AVG((batt_scored)::int)::float8 AS pct_batt_scored,
  AVG((damage_any_scored)::int)::float8 AS pct_any_scored
FROM ml.v_damage_fusion_features_v2_scored;
```

### 10.2 Reason code distribution

```sql
SELECT img_score_reason_code, COUNT(*) AS n
FROM ml.v_damage_fusion_features_v2
GROUP BY 1
ORDER BY 1;
```

### 10.3 Duplicate key test

```sql
SELECT generation, listing_id, COUNT(*) AS n
FROM ml.v_damage_fusion_features_v2_scored
GROUP BY 1,2
HAVING COUNT(*) > 1
LIMIT 20;
```

---

## 11. Full SQL inventory

See `sql/` in this package:
- `01_create_fusion_views_v2.sql`
- `02_create_cert_entrypoint_and_guard.sql`
- `03_baseline_viewdefs.sql`
- `04_baseline_dataset_hashes_365.sql`
- `05_cert_functions.sql`
- `06_run_certification.sql`

Optional strict T0/SLA-gated materialization is included in `sql/optional/`.

---
