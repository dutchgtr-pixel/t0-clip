# Refresh Runbook (End-to-End) and Concurrent Refresh Strategy

This file is the operational “do this every day” runbook for the certified survival/TOM feature store.

---

## 1) The only supported operational pathway

### 1.1 Refresh → certify → guard
The supported workflow is:

1. Refresh materialized views in dependency order
2. Run certification assertions (and update registry)
3. Training reads through the guarded view; any stale/uncertified state fails closed.

### 1.2 Canonical procedure
```sql
CREATE OR REPLACE PROCEDURE audit.refresh_and_certify_survival_v1(p_raise boolean DEFAULT true)
LANGUAGE plpgsql
AS $$
BEGIN
  -- Refresh in dependency order (T0-safe objects only)
  REFRESH MATERIALIZED VIEW ml.tom_speed_anchor_asof_v1_mv;
  REFRESH MATERIALIZED VIEW ml.tom_features_v1_enriched_speed_t0_v1_mv;
  REFRESH MATERIALIZED VIEW ml.tom_features_v1_enriched_ai_clean_t0_v1_mv;

  REFRESH MATERIALIZED VIEW ml.iphone_image_features_unified_t0_v1_mv;
  REFRESH MATERIALIZED VIEW ml.v_damage_fusion_features_v2_scored_t0_v1_mv;
  REFRESH MATERIALIZED VIEW ml.iphone_device_meta_encoded_t0_v1_mv;

  -- Run certification assertions + write registry status
  CALL audit.run_t0_cert_survival_v1();

  -- Optionally fail hard if status != CERTIFIED or stale
  IF p_raise THEN
    PERFORM audit.require_certified('ml.survival_feature_store_t0_v1_v', interval '24 hours');
  END IF;
END $$;
```

Usage:
```sql
CALL audit.refresh_and_certify_survival_v1(true);
```

---

## 2) Concurrent refresh strategy (when you need low read disruption)

`REFRESH MATERIALIZED VIEW CONCURRENTLY` can reduce blocking, but has requirements:
- MV must have a **unique index** covering a set of columns that uniquely identify rows.
- You cannot run `REFRESH ... CONCURRENTLY` inside a transaction block, and it is typically orchestrated by an external job runner.

### 2.1 Minimum unique index set (already recommended in builds)
- `ml.tom_speed_anchor_asof_v1_mv`: `(anchor_day, generation, sbucket, ptv_bucket)`
- `ml.tom_features_v1_enriched_speed_t0_v1_mv`: `(listing_id)`
- `ml.tom_features_v1_enriched_ai_clean_t0_v1_mv`: `(listing_id)`
- `ml.iphone_image_features_unified_t0_v1_mv`: `(generation, listing_id)`
- `ml.v_damage_fusion_features_v2_scored_t0_v1_mv`: `(generation, listing_id)`
- `ml.iphone_device_meta_encoded_t0_v1_mv`: `(generation, listing_id)`

### 2.2 Suggested orchestration
If you need concurrency, run these as individual statements (outside a transaction):
```sql
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_speed_anchor_asof_v1_mv;
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_speed_t0_v1_mv;
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_ai_clean_t0_v1_mv;
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.iphone_image_features_unified_t0_v1_mv;
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.v_damage_fusion_features_v2_scored_t0_v1_mv;
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.iphone_device_meta_encoded_t0_v1_mv;

CALL audit.run_t0_cert_survival_v1();
SELECT audit.require_certified_strict('ml.survival_feature_store_t0_v1_v', interval '24 hours');
```

---

## 3) What to do when certification fails (incident playbook)

### 3.1 Failure classes
1. **Forbidden deps present**: Someone reintroduced a legacy/current/label dependency.
   - Action: fix dependency chain; refresh; recertify.
2. **Time-of-query tokens detected**: Someone added `now()`/`CURRENT_DATE`.
   - Action: refactor to anchor-based logic.
3. **SLA leakage**: `img__*` populated when `img_within_sla=false`.
   - Action: fix gating logic at the feature-store entrypoint (must be CASE-gated).
   - Also verify leakage queries with regex or escaped LIKE.
4. **Dataset hash drift**: baseline mismatch.
   - Action: determine cause:
     - leakage or code drift → fix and keep baseline
     - upstream backfill → accept only with explicit review; then update baseline and re-certify
5. **Viewdef drift**: closure definition hashes changed.
   - Action: treat as code change; re-run Gate A and update baselines only after review.

### 3.2 How to re-baseline safely (controlled change)
If you intentionally changed feature logic:
1. Commit the DDL change to version control
2. Recompute and store:
   - `audit.t0_viewdef_baseline` (closure)
   - `audit.t0_dataset_hash_baseline` (selected historical days)
3. Run `CALL audit.run_t0_cert_survival_v1();`

---

## 4) Legacy refresh script (verbatim; deprecated)

This is the previous refresh artifact you provided. It is preserved for history, but it is **not** the supported process going forward because it refreshes legacy (non T0-safe) objects.

-- A/B/C: core base + labels + training base
REFRESH MATERIALIZED VIEW ml.tom_features_v1_mv;
REFRESH MATERIALIZED VIEW ml.tom_labels_v1_mv;
REFRESH MATERIALIZED VIEW ml.tom_train_base_v1_mv;

-- D1/D2/D3: anchor/history layers
REFRESH MATERIALIZED VIEW ml.sold_durations_v1_mv;
REFRESH MATERIALIZED VIEW ml.ask_daily_median_v1;
REFRESH MATERIALIZED VIEW ml.sold_prices_v1_mv;

-- E: enriched
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_mv;

-- E-clean: spam-clean enriched
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_clean_mv;

-- D4: speed anchors (depends on D1 + E)
REFRESH MATERIALIZED VIEW ml.tom_speed_anchor_v1_mv;

-- E+speed: enriched + speed_* (depends on clean + D4)
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_speed_mv;

-- E-clean+AI: AI-augmented read layer (depends on E+speed)
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_ai_clean_mv;

-- === OB pipeline (depends on v1 base + geo + labels; and v1 enriched AI clean) ===
REFRESH MATERIALIZED VIEW ml.tom_ob_pool_v1_mv;
REFRESH MATERIALIZED VIEW ml.tom_ob_features_v1_mv;
REFRESH MATERIALIZED VIEW ml.tom_features_v2_enriched_ai_ob_clean_mv;

-- Stats for the planner
ANALYZE ml.tom_features_v1_enriched_mv;
ANALYZE ml.tom_features_v1_enriched_speed_mv;
ANALYZE ml.tom_features_v1_enriched_clean_mv;
ANALYZE ml.tom_features_v1_enriched_ai_clean_mv;

ANALYZE ml.sold_durations_v1_mv;
ANALYZE ml.sold_prices_v1_mv;
ANALYZE ml.tom_speed_anchor_v1_mv;

ANALYZE ml.tom_ob_pool_v1_mv;
ANALYZE ml.tom_ob_features_v1_mv;
ANALYZE ml.tom_features_v2_enriched_ai_ob_clean_mv;

ANALYZE ref.postal_code_to_super_metro;
ANALYZE ref.city_to_super_metro;
ANALYZE ref.geo_mapping_release;


-- 1) refresh socio MV first (dependency)
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v2_enriched_ai_ob_clean_socio_mv;
ANALYZE ml.tom_features_v2_enriched_ai_ob_clean_socio_mv;

-- 2) refresh socio+market MV next (depends on socio MV)
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v2_enriched_ai_ob_clean_socio_market_v1_mv;
ANALYZE ml.tom_features_v2_enriched_ai_ob_clean_socio_market_v1_mv;



