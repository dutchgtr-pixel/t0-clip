# 8. Validation queries

## 8.1 Column presence (affordability must exist)

```sql
SELECT attname
FROM pg_attribute
WHERE attrelid = 'ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv'::regclass
  AND attnum > 0 AND NOT attisdropped
  AND attname IN ('price_months_income','log_price_to_income_cap','aff_resid_in_seg','aff_decile_in_seg')
ORDER BY attname;
```

```sql
SELECT attname
FROM pg_attribute
WHERE attrelid = 'ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv'::regclass
  AND attnum > 0 AND NOT attisdropped
  AND attname IN ('price_months_income','log_price_to_income_cap','aff_resid_in_seg','aff_decile_in_seg')
ORDER BY attname;
```

## 8.2 Invariants

```sql
SELECT COUNT(*) AS n_bad_socio
FROM ml.tom_features_v2_enriched_ai_ob_clean_socio_t0_v1_mv
WHERE (postal_kommune_snapshot_date IS NOT NULL AND postal_kommune_snapshot_date > t0::date)
   OR (socio_snapshot_date IS NOT NULL AND socio_snapshot_date > t0::date);
```

```sql
SELECT COUNT(*) AS n_bad_market
FROM ml.tom_features_v2_enriched_ai_ob_clean_socio_market_t0_v1_mv
WHERE (comp_max_t0_super_metro IS NOT NULL AND comp_max_t0_super_metro >= t0)
   OR (comp_max_t0_kommune     IS NOT NULL AND comp_max_t0_kommune     >= t0);
```

## 8.3 Guard proof

```sql
SELECT audit.require_certified_strict('ml.socio_market_feature_store_t0_v1_v', interval '24 hours');
SELECT COUNT(*) FROM ml.socio_market_feature_store_train_v;
```

## 8.4 Trainer attachment (12 columns)

The trainer is expected to log:

`centrality_class, miss_kommune, miss_socio, price_months_income, log_price_to_income_cap,
 aff_resid_in_seg, aff_decile_in_seg, rel_log_price_to_comp_mean30_super,
 rel_log_price_to_comp_mean30_kommune, rel_price_best, rel_price_source, miss_rel_price`

