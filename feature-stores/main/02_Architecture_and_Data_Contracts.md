# Architecture and Data Contracts (T0 / Leakage-Safe)

This file describes the **logical architecture**, the **data contracts**, and the **invariants** enforced by the certification program.

---

## 1. High-level architecture (certified surface)

```
                         ┌─────────────────────────────────────────┐
                         │  TRAINING CODE (model pipeline)         │
                         │  SELECT ... FROM ml.survival_feature...  │
                         └─────────────────────────────────────────┘
                                            │
                                            │  (must fail closed if not certified)
                                            ▼
                         ┌─────────────────────────────────────────┐
                         │ ml.survival_feature_store_train_v        │
                         │ (guarded view; calls audit.cert_guard)   │
                         └─────────────────────────────────────────┘
                                            │
                                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                 CERTIFIED FEATURE STORE ENTRYPOINT                        │
│                   ml.survival_feature_store_t0_v1_v                       │
│  - joins ONLY T0-safe MVs/views                                            │
│  - contains canonical img_within_sla flag                                  │
│  - image/damage columns are NULL when img_within_sla=false                 │
└──────────────────────────────────────────────────────────────────────────┘
        │                     │                     │
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────────┐
│ TOM core (T0)     │  │ Image store (T0)  │  │ Damage store (T0)           │
│  - speed (as-of)  │  │ SLA gated + flag  │  │ SLA gated (inherits flag)   │
│  - AI enrich asof │  └──────────────────┘  └────────────────────────────┘
└──────────────────┘                 │
        │                             ▼
        ▼                    ┌────────────────────────────┐
┌──────────────────────┐     │ Device meta (T0)            │
│ Geo dim (pinned/T0)   │     │ derived from T0 base        │
└──────────────────────┘     └────────────────────────────┘
```

---

## 2. Data contracts (non-negotiable)

### 2.1 Primary key contract
All certified feature-store rows are keyed by:

- `generation` (model-generation/version key)
- `listing_id` (listing identifier)

**Invariant**: `(generation, listing_id)` is unique at the certified entrypoint for any `edited_date::date = t0_day`.

### 2.2 Anchor / T0 contract
Training features must be representable as **as-of `edited_date`** (“T0”):

- **Anchor column**: `edited_date` (timestamp; `edited_date::date` used in proofs)
- For speed features: `anchor_day = edited_date::date`
- For dynamic proofs: select a historical `t0_day` and run the same query later → hashes must match.

### 2.3 Event-time embargo contract (no future data)
Any feature derived from time-varying facts must enforce:

- `event_time < anchor_day`

Example:
- sales used to compute speed features must satisfy: `sold_day < anchor_day`
- the anchored speed MV exposes proof columns:
  - `min_sold_day_used`
  - `max_sold_day_used`

### 2.4 SLA gating contract (image/damage)
Image-derived features may be computed later than the listing edit time. To prevent “late-arriving features” from changing historical rows:

- `img_within_sla` is computed per row using pipeline completion timestamps (`*_done_at`) and optionally image asset timestamps (e.g., `iphone_image_assets.created_at`)
- If `img_within_sla=false`, then **every `img__*` and `dmg__*` output column must be NULL**

**Important: wildcard pitfall**
SQL `LIKE 'img__%'` treats `_` as a wildcard. Always use either:
- `kv.key ~ '^img__'` (regex), or
- `kv.key LIKE 'img\_\_%' ESCAPE '\'` (escaped LIKE)

### 2.5 “No labels inside feature store” contract
Labels are not features. Certified feature-store entrypoints must not depend on label MVs or training-only mixed views.

- Allowed: labels joined downstream in training code
- Not allowed: label dependencies inside `ml.survival_feature_store_t0_v1_v` closure

---

## 3. Objects and responsibilities (by layer)

### 3.1 Speed anchor (as-of)
- `ml.tom_speed_anchor_asof_v1_mv`
  - Key: `(anchor_day, generation, sbucket, ptv_bucket)`
  - Embargo: `sold_day < anchor_day`
  - Outputs: anchored speed rates + proof columns

### 3.2 TOM enriched (T0)
- `ml.tom_features_v1_enriched_speed_t0_v1_mv`
  - joins `ml.tom_speed_anchor_asof_v1_mv` at `anchor_day = edited_date::date`

- `ml.tom_features_v1_enriched_ai_clean_t0_v1_mv`
  - AI enrichment selected as-of `edited_date + 24h` via `LATERAL (...) ORDER BY updated_at DESC LIMIT 1`
  - includes audit fields: `ai_updated_at`, `ai_model`, `ai_version`

- `ml.tom_features_v1_enriched_ai_clean_t0_read_v`
  - adds pinned geo attributes for training convenience

### 3.3 Geo mapping pinned
- `ref.geo_mapping_pinned_super_metro_v4_v1`
- `ml.geo_dim_super_metro_v4_pinned_v1`
- `ml.geo_dim_super_metro_v4_t0_v1`

### 3.4 Image/damage/device stores (T0)
- `ml.iphone_image_features_unified_t0_v1_mv`
- `ml.v_damage_fusion_features_v2_scored_t0_v1_mv`
- `ml.iphone_device_meta_encoded_t0_v1_mv`

### 3.5 Certified feature store entrypoint
- `ml.survival_feature_store_t0_v1_v`
  - emits:
    - base features (`b.*`) from TOM T0 read view
    - `img_within_sla` canonical flag
    - `img__*`, `dmg__*`, `dev__*` prefixed feature columns

---

## 4. Certification invariants (what is actually asserted)
The certification function asserts (at minimum):

1. **Forbidden dependency closure** is empty (no labels/current/legacy/leaky objects)
2. **No time-of-query tokens** appear in viewdefs/matview defs in the closure
3. **Speed embargo** holds (no future sold_day relative to anchor_day)
4. **AI as-of window** holds (no ai_updated_at beyond edited_date + SLA window)
5. **Image/damage SLA gating** holds (no populated img__/dmg__ when img_within_sla=false)
6. **Dynamic hash baselines** match for pinned sample sets and multiple historical days
7. **Viewdef baselines** match for closure objects (drift detection)

---

## 5. Certification claims (how to defend them under scrutiny)
- The program provides both **static proofs** (closure + token bans + viewdef hashes) and a **behavioral proof** (dataset hash invariance).
- Any drift is either:
  - leakage (bad), or
  - an upstream backfill (acceptable only with explicit review + recertification).

