-- Selected historical implementation; see provenance.json and README.md.
-- Reference only: dependencies, current-state inputs and destructive rebuilds require review.
\timing on

WITH eval_rows AS (
    SELECT
        l.generation,
        l.listing_id,
        l.edited_date,
        (l.sold_date AT TIME ZONE 'Europe/Berlin')::date AS sold_day,
        (EXTRACT(EPOCH FROM (l.sold_date - l.edited_date)) / 3600.0)::numeric AS duration_h,
        (EXTRACT(EPOCH FROM (l.sold_date - l.edited_date)) / 3600.0 <= 72.0) AS fast72,
        pa.pa__anchor_n_90d,
        pa.pa__price_to_anchor_90d,
        pa.pa__prior_fast72_rate_90d,
        pa.pa__prior_avg_hours_90d,
        pa.pa__anchor_source_level_90d,
        CASE
            WHEN (l.sold_date AT TIME ZONE 'Europe/Berlin')::date BETWEEN DATE '2026-03-21' AND DATE '2026-03-31' THEN 'sval'
            WHEN (l.sold_date AT TIME ZONE 'Europe/Berlin')::date BETWEEN DATE '2026-04-01' AND DATE '2026-04-10' THEN 'eval'
            ELSE 'other'
        END AS split,
        CASE
            WHEN pa.pa__price_to_anchor_90d IS NULL THEN 'missing_anchor'
            WHEN pa.pa__price_to_anchor_90d < 0.85 THEN '<0.85'
            WHEN pa.pa__price_to_anchor_90d < 0.95 THEN '0.85-0.95'
            WHEN pa.pa__price_to_anchor_90d <= 1.05 THEN '0.95-1.05'
            WHEN pa.pa__price_to_anchor_90d <= 1.15 THEN '1.05-1.15'
            ELSE '>1.15'
        END AS price_anchor_bucket
    FROM "device".device_listings l
    LEFT JOIN ml.price_anchor_feature_store_t0_v1_mv pa
      ON pa.generation = l.generation
     AND pa.listing_id = l.listing_id
     AND pa.edited_date = l.edited_date
    WHERE l.spam IS NULL
      AND l.sold_date IS NOT NULL
      AND l.edited_date IS NOT NULL
      AND l.sold_date >= l.edited_date
      AND (l.sold_date AT TIME ZONE 'Europe/Berlin')::date BETWEEN DATE '2026-03-21' AND DATE '2026-04-10'
)
SELECT
    split,
    count(*) AS rows,
    count(*) FILTER (WHERE pa__anchor_n_90d >= 30) AS anchored_rows,
    round(100.0 * count(*) FILTER (WHERE pa__anchor_n_90d >= 30) / NULLIF(count(*), 0), 2) AS anchored_pct,
    round(corr(pa__price_to_anchor_90d, fast72::int)::numeric, 4) AS corr_price_to_fast72,
    round(corr(pa__price_to_anchor_90d, duration_h)::numeric, 4) AS corr_price_to_duration,
    round(corr(pa__prior_fast72_rate_90d, fast72::int)::numeric, 4) AS corr_prior_to_fast72,
    round(corr(pa__prior_avg_hours_90d, duration_h)::numeric, 4) AS corr_prior_to_duration
FROM eval_rows
WHERE split IN ('sval', 'eval')
GROUP BY split
ORDER BY split;

WITH eval_rows AS (
    SELECT
        (EXTRACT(EPOCH FROM (l.sold_date - l.edited_date)) / 3600.0)::numeric AS duration_h,
        (EXTRACT(EPOCH FROM (l.sold_date - l.edited_date)) / 3600.0 <= 72.0) AS fast72,
        CASE
            WHEN (l.sold_date AT TIME ZONE 'Europe/Berlin')::date BETWEEN DATE '2026-03-21' AND DATE '2026-03-31' THEN 'sval'
            WHEN (l.sold_date AT TIME ZONE 'Europe/Berlin')::date BETWEEN DATE '2026-04-01' AND DATE '2026-04-10' THEN 'eval'
            ELSE 'other'
        END AS split,
        CASE
            WHEN pa.pa__price_to_anchor_90d IS NULL THEN 'missing_anchor'
            WHEN pa.pa__price_to_anchor_90d < 0.85 THEN '<0.85'
            WHEN pa.pa__price_to_anchor_90d < 0.95 THEN '0.85-0.95'
            WHEN pa.pa__price_to_anchor_90d <= 1.05 THEN '0.95-1.05'
            WHEN pa.pa__price_to_anchor_90d <= 1.15 THEN '1.05-1.15'
            ELSE '>1.15'
        END AS price_anchor_bucket,
        pa.pa__anchor_n_90d,
        pa.pa__price_to_anchor_90d
    FROM "device".device_listings l
    LEFT JOIN ml.price_anchor_feature_store_t0_v1_mv pa
      ON pa.generation = l.generation
     AND pa.listing_id = l.listing_id
     AND pa.edited_date = l.edited_date
    WHERE l.spam IS NULL
      AND l.sold_date IS NOT NULL
      AND l.edited_date IS NOT NULL
      AND l.sold_date >= l.edited_date
      AND (l.sold_date AT TIME ZONE 'Europe/Berlin')::date BETWEEN DATE '2026-03-21' AND DATE '2026-04-10'
)
SELECT
    split,
    price_anchor_bucket,
    count(*) AS rows,
    count(*) FILTER (WHERE fast72) AS fast72_rows,
    round(avg(fast72::int)::numeric, 4) AS fast72_rate,
    round(percentile_cont(0.50) WITHIN GROUP (ORDER BY duration_h)::numeric, 1) AS median_h,
    round(avg(duration_h)::numeric, 1) AS avg_h,
    round(avg(pa__anchor_n_90d)::numeric, 1) AS avg_anchor_n_90d,
    round(percentile_cont(0.50) WITHIN GROUP (ORDER BY pa__price_to_anchor_90d)::numeric, 3) AS median_price_to_anchor_90d
FROM eval_rows
WHERE split IN ('sval', 'eval')
GROUP BY split, price_anchor_bucket
ORDER BY split,
    CASE price_anchor_bucket
        WHEN 'missing_anchor' THEN 0
        WHEN '<0.85' THEN 1
        WHEN '0.85-0.95' THEN 2
        WHEN '0.95-1.05' THEN 3
        WHEN '1.05-1.15' THEN 4
        ELSE 5
    END;

SELECT
    count(*) AS self_leak_rows
FROM ml.price_anchor_feature_store_t0_v1_mv pa
JOIN ml.price_anchor_history_t0_v1_mv h
  ON h.generation = pa.generation
 AND h.listing_id = pa.listing_id
 AND h.edited_date = pa.edited_date
WHERE h.sold_day < (pa.edited_date AT TIME ZONE 'Europe/Berlin')::date;

SELECT
    count(*) FILTER (WHERE k.source_level IS NOT NULL AND k.source_n_90d < 30) AS fallback_support_violations,
    count(*) FILTER (WHERE k.source_level IS NOT NULL AND k.source_n_90d <> COALESCE(m.n_90d, -1)) AS support_mismatch_rows,
    count(*) FILTER (WHERE k.source_level IS NOT NULL AND m.n_90d IS NULL) AS missing_metric_rows
FROM ml.price_anchor_target_keys_t0_v1_mv k
LEFT JOIN ml.price_anchor_key_metrics_t0_v1_mv m
  ON m.key_id = k.key_id;

WITH matched_history AS (
    SELECT k.key_id, k.edited_day, h.sold_day
    FROM ml.price_anchor_target_keys_t0_v1_mv k
    JOIN ml.price_anchor_history_t0_v1_mv h
      ON k.source_level = 1
     AND h.sold_day < k.edited_day
     AND h.sold_day >= k.edited_day - 90
     AND h.generation = k.generation
     AND h.model_key = k.model_key
     AND h.storage_bucket = k.storage_bucket
     AND h.condition_bucket = k.condition_bucket
     AND h.battery_bucket = k.battery_bucket
     AND h.damage_bucket = k.damage_bucket
    UNION ALL
    SELECT k.key_id, k.edited_day, h.sold_day
    FROM ml.price_anchor_target_keys_t0_v1_mv k
    JOIN ml.price_anchor_history_t0_v1_mv h
      ON k.source_level = 2
     AND h.sold_day < k.edited_day
     AND h.sold_day >= k.edited_day - 90
     AND h.generation = k.generation
     AND h.model_key = k.model_key
     AND h.storage_bucket = k.storage_bucket
     AND h.condition_bucket = k.condition_bucket
     AND h.damage_bucket = k.damage_bucket
    UNION ALL
    SELECT k.key_id, k.edited_day, h.sold_day
    FROM ml.price_anchor_target_keys_t0_v1_mv k
    JOIN ml.price_anchor_history_t0_v1_mv h
      ON k.source_level = 3
     AND h.sold_day < k.edited_day
     AND h.sold_day >= k.edited_day - 90
     AND h.generation = k.generation
     AND h.model_key = k.model_key
     AND h.storage_bucket = k.storage_bucket
     AND h.condition_bucket = k.condition_bucket
    UNION ALL
    SELECT k.key_id, k.edited_day, h.sold_day
    FROM ml.price_anchor_target_keys_t0_v1_mv k
    JOIN ml.price_anchor_history_t0_v1_mv h
      ON k.source_level = 4
     AND h.sold_day < k.edited_day
     AND h.sold_day >= k.edited_day - 90
     AND h.generation = k.generation
     AND h.model_key = k.model_key
     AND h.storage_bucket = k.storage_bucket
    UNION ALL
    SELECT k.key_id, k.edited_day, h.sold_day
    FROM ml.price_anchor_target_keys_t0_v1_mv k
    JOIN ml.price_anchor_history_t0_v1_mv h
      ON k.source_level = 5
     AND h.sold_day < k.edited_day
     AND h.sold_day >= k.edited_day - 90
     AND h.generation = k.generation
     AND h.model_key = k.model_key
    UNION ALL
    SELECT k.key_id, k.edited_day, h.sold_day
    FROM ml.price_anchor_target_keys_t0_v1_mv k
    JOIN ml.price_anchor_history_t0_v1_mv h
      ON k.source_level = 6
     AND h.sold_day < k.edited_day
     AND h.sold_day >= k.edited_day - 90
     AND h.generation = k.generation
)
SELECT
    count(*) AS matched_prior_rows,
    min(edited_day - sold_day) AS min_days_before_target,
    count(*) FILTER (WHERE sold_day >= edited_day) AS future_or_same_day_rows
FROM matched_history;

SELECT
    pa__anchor_source_level_90d,
    count(*) AS rows
FROM ml.price_anchor_feature_store_t0_v1_mv
GROUP BY pa__anchor_source_level_90d
ORDER BY pa__anchor_source_level_90d NULLS LAST;
