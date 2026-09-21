-- Selected historical implementation; see provenance.json and README.md.
-- Reference only: dependencies, current-state inputs and destructive rebuilds require review.
SET statement_timeout = 0;
SET work_mem = '512MB';
SET max_parallel_workers_per_gather = 0;

CREATE SCHEMA IF NOT EXISTS ml;

DROP MATERIALIZED VIEW IF EXISTS ml.price_anchor_feature_store_t0_v1_mv;
DROP MATERIALIZED VIEW IF EXISTS ml.price_anchor_key_metrics_t0_v1_mv;
DROP MATERIALIZED VIEW IF EXISTS ml.price_anchor_target_keys_t0_v1_mv;
DROP MATERIALIZED VIEW IF EXISTS ml.price_anchor_prior_daily_t0_v1_mv;
DROP MATERIALIZED VIEW IF EXISTS ml.price_anchor_history_t0_v1_mv;

CREATE MATERIALIZED VIEW ml.price_anchor_history_t0_v1_mv AS
SELECT
    l.generation,
    l.listing_id,
    l.edited_date,
    (l.edited_date AT TIME ZONE 'Europe/Berlin')::date AS edited_day,
    l.sold_date,
    (l.sold_date AT TIME ZONE 'Europe/Berlin')::date AS sold_day,
    lower(regexp_replace(coalesce(nullif(trim(l.model), ''), 'unknown'), '\s+', ' ', 'g')) AS model_key,
    CASE
        WHEN l.storage_gb IS NULL THEN 'stor_missing'
        WHEN l.storage_gb <= 128 THEN 'stor_le128'
        WHEN l.storage_gb <= 256 THEN 'stor_256'
        WHEN l.storage_gb <= 512 THEN 'stor_512'
        ELSE 'stor_1tbplus'
    END AS storage_bucket,
    CASE
        WHEN l.condition_score IS NULL THEN 'cond_missing'
        WHEN l.condition_score < 2 THEN 'cond_low'
        WHEN l.condition_score < 4 THEN 'cond_mid'
        ELSE 'cond_high'
    END AS condition_bucket,
    CASE
        WHEN l.battery_pct IS NULL THEN 'bat_missing'
        WHEN l.battery_pct < 80 THEN 'bat_lt80'
        WHEN l.battery_pct < 86 THEN 'bat_80_85'
        WHEN l.battery_pct < 91 THEN 'bat_86_90'
        ELSE 'bat_91plus'
    END AS battery_bucket,
    CASE
        WHEN k.max_visible_damage_level IS NULL THEN 'dmg_missing'
        WHEN k.max_visible_damage_level = 0 THEN 'dmg_0'
        WHEN k.max_visible_damage_level <= 5 THEN 'dmg_1_5'
        ELSE 'dmg_6plus'
    END AS damage_bucket,
    l.price::numeric AS ask_price,
    (EXTRACT(EPOCH FROM (l.sold_date - l.edited_date)) / 3600.0)::numeric AS duration_h,
    (EXTRACT(EPOCH FROM (l.sold_date - l.edited_date)) / 3600.0 <= 72.0)::int AS y_fast72,
    (EXTRACT(EPOCH FROM (l.sold_date - l.edited_date)) / 3600.0 > 72.0
        AND EXTRACT(EPOCH FROM (l.sold_date - l.edited_date)) / 3600.0 <= 168.0)::int AS y_3_7d,
    (EXTRACT(EPOCH FROM (l.sold_date - l.edited_date)) / 3600.0 > 168.0
        AND EXTRACT(EPOCH FROM (l.sold_date - l.edited_date)) / 3600.0 <= 240.0)::int AS y_7_10d,
    (EXTRACT(EPOCH FROM (l.sold_date - l.edited_date)) / 3600.0 > 240.0
        AND EXTRACT(EPOCH FROM (l.sold_date - l.edited_date)) / 3600.0 <= 504.0)::int AS y_10_21d,
    (EXTRACT(EPOCH FROM (l.sold_date - l.edited_date)) / 3600.0 > 504.0)::int AS y_gt21d
FROM "device".device_listings l
LEFT JOIN image_map.device_listing_image_k8_manifest_v1 k
  ON k.generation = l.generation
 AND k.listing_id = l.listing_id
WHERE l.spam IS NULL
  AND l.edited_date IS NOT NULL
  AND l.sold_date IS NOT NULL
  AND l.sold_date >= l.edited_date
  AND l.price IS NOT NULL
  AND l.price > 0;

CREATE INDEX IF NOT EXISTS price_anchor_history_t0_v1_l1_idx
    ON ml.price_anchor_history_t0_v1_mv (
        generation, model_key, storage_bucket, condition_bucket, battery_bucket, damage_bucket, sold_day
    );
CREATE INDEX IF NOT EXISTS price_anchor_history_t0_v1_l2_idx
    ON ml.price_anchor_history_t0_v1_mv (
        generation, model_key, storage_bucket, condition_bucket, damage_bucket, sold_day
    );
CREATE INDEX IF NOT EXISTS price_anchor_history_t0_v1_l3_idx
    ON ml.price_anchor_history_t0_v1_mv (
        generation, model_key, storage_bucket, condition_bucket, sold_day
    );
CREATE INDEX IF NOT EXISTS price_anchor_history_t0_v1_l4_idx
    ON ml.price_anchor_history_t0_v1_mv (
        generation, model_key, storage_bucket, sold_day
    );
CREATE INDEX IF NOT EXISTS price_anchor_history_t0_v1_l5_idx
    ON ml.price_anchor_history_t0_v1_mv (
        generation, model_key, sold_day
    );
CREATE INDEX IF NOT EXISTS price_anchor_history_t0_v1_l6_idx
    ON ml.price_anchor_history_t0_v1_mv (
        generation, sold_day
    );
CREATE INDEX IF NOT EXISTS price_anchor_history_t0_v1_identity_idx
    ON ml.price_anchor_history_t0_v1_mv (generation, listing_id, edited_date, sold_day);

ANALYZE ml.price_anchor_history_t0_v1_mv;

CREATE MATERIALIZED VIEW ml.price_anchor_prior_daily_t0_v1_mv AS
SELECT
    generation,
    model_key,
    storage_bucket,
    condition_bucket,
    battery_bucket,
    damage_bucket,
    sold_day,
    count(*)::int AS sold_n,
    sum(y_fast72)::int AS fast72_n,
    sum(y_3_7d)::int AS bucket_3_7d_n,
    sum(y_7_10d)::int AS bucket_7_10d_n,
    sum(y_10_21d)::int AS bucket_10_21d_n,
    sum(y_gt21d)::int AS bucket_gt21d_n,
    sum(duration_h)::numeric AS duration_hours_sum
FROM ml.price_anchor_history_t0_v1_mv
GROUP BY
    generation,
    model_key,
    storage_bucket,
    condition_bucket,
    battery_bucket,
    damage_bucket,
    sold_day;

CREATE INDEX IF NOT EXISTS price_anchor_prior_daily_t0_v1_l1_idx
    ON ml.price_anchor_prior_daily_t0_v1_mv (
        generation, model_key, storage_bucket, condition_bucket, battery_bucket, damage_bucket, sold_day
    );
CREATE INDEX IF NOT EXISTS price_anchor_prior_daily_t0_v1_l2_idx
    ON ml.price_anchor_prior_daily_t0_v1_mv (
        generation, model_key, storage_bucket, condition_bucket, damage_bucket, sold_day
    );
CREATE INDEX IF NOT EXISTS price_anchor_prior_daily_t0_v1_l3_idx
    ON ml.price_anchor_prior_daily_t0_v1_mv (
        generation, model_key, storage_bucket, condition_bucket, sold_day
    );
CREATE INDEX IF NOT EXISTS price_anchor_prior_daily_t0_v1_l4_idx
    ON ml.price_anchor_prior_daily_t0_v1_mv (
        generation, model_key, storage_bucket, sold_day
    );
CREATE INDEX IF NOT EXISTS price_anchor_prior_daily_t0_v1_l5_idx
    ON ml.price_anchor_prior_daily_t0_v1_mv (
        generation, model_key, sold_day
    );
CREATE INDEX IF NOT EXISTS price_anchor_prior_daily_t0_v1_l6_idx
    ON ml.price_anchor_prior_daily_t0_v1_mv (
        generation, sold_day
    );

ANALYZE ml.price_anchor_prior_daily_t0_v1_mv;

CREATE MATERIALIZED VIEW ml.price_anchor_target_keys_t0_v1_mv AS
WITH target_rows AS (
    SELECT
        l.generation,
        (l.edited_date AT TIME ZONE 'Europe/Berlin')::date AS edited_day,
        lower(regexp_replace(coalesce(nullif(trim(l.model), ''), 'unknown'), '\s+', ' ', 'g')) AS model_key,
        CASE
            WHEN l.storage_gb IS NULL THEN 'stor_missing'
            WHEN l.storage_gb <= 128 THEN 'stor_le128'
            WHEN l.storage_gb <= 256 THEN 'stor_256'
            WHEN l.storage_gb <= 512 THEN 'stor_512'
            ELSE 'stor_1tbplus'
        END AS storage_bucket,
        CASE
            WHEN l.condition_score IS NULL THEN 'cond_missing'
            WHEN l.condition_score < 2 THEN 'cond_low'
            WHEN l.condition_score < 4 THEN 'cond_mid'
            ELSE 'cond_high'
        END AS condition_bucket,
        CASE
            WHEN l.battery_pct IS NULL THEN 'bat_missing'
            WHEN l.battery_pct < 80 THEN 'bat_lt80'
            WHEN l.battery_pct < 86 THEN 'bat_80_85'
            WHEN l.battery_pct < 91 THEN 'bat_86_90'
            ELSE 'bat_91plus'
        END AS battery_bucket,
        CASE
            WHEN k.max_visible_damage_level IS NULL THEN 'dmg_missing'
            WHEN k.max_visible_damage_level = 0 THEN 'dmg_0'
            WHEN k.max_visible_damage_level <= 5 THEN 'dmg_1_5'
            ELSE 'dmg_6plus'
        END AS damage_bucket
    FROM "device".device_listings l
    LEFT JOIN image_map.device_listing_image_k8_manifest_v1 k
      ON k.generation = l.generation
     AND k.listing_id = l.listing_id
    WHERE l.spam IS NULL
      AND l.edited_date IS NOT NULL
), target_keys AS (
    SELECT
        row_number() OVER () AS key_id,
        d.*
    FROM (
        SELECT DISTINCT
            generation,
            edited_day,
            model_key,
            storage_bucket,
            condition_bucket,
            battery_bucket,
            damage_bucket
        FROM target_rows
    ) d
), key_counts AS (
    SELECT
        k.*,
        COALESCE(l1.n_90d, 0) AS l1_n_90d,
        COALESCE(l2.n_90d, 0) AS l2_n_90d,
        COALESCE(l3.n_90d, 0) AS l3_n_90d,
        COALESCE(l4.n_90d, 0) AS l4_n_90d,
        COALESCE(l5.n_90d, 0) AS l5_n_90d,
        COALESCE(l6.n_90d, 0) AS l6_n_90d
    FROM target_keys k
    LEFT JOIN LATERAL (
        SELECT sum(d.sold_n)::int AS n_90d
        FROM ml.price_anchor_prior_daily_t0_v1_mv d
        WHERE d.sold_day < k.edited_day
          AND d.sold_day >= k.edited_day - 90
          AND d.generation = k.generation
          AND d.model_key = k.model_key
          AND d.storage_bucket = k.storage_bucket
          AND d.condition_bucket = k.condition_bucket
          AND d.battery_bucket = k.battery_bucket
          AND d.damage_bucket = k.damage_bucket
    ) l1 ON true
    LEFT JOIN LATERAL (
        SELECT sum(d.sold_n)::int AS n_90d
        FROM ml.price_anchor_prior_daily_t0_v1_mv d
        WHERE d.sold_day < k.edited_day
          AND d.sold_day >= k.edited_day - 90
          AND d.generation = k.generation
          AND d.model_key = k.model_key
          AND d.storage_bucket = k.storage_bucket
          AND d.condition_bucket = k.condition_bucket
          AND d.damage_bucket = k.damage_bucket
    ) l2 ON true
    LEFT JOIN LATERAL (
        SELECT sum(d.sold_n)::int AS n_90d
        FROM ml.price_anchor_prior_daily_t0_v1_mv d
        WHERE d.sold_day < k.edited_day
          AND d.sold_day >= k.edited_day - 90
          AND d.generation = k.generation
          AND d.model_key = k.model_key
          AND d.storage_bucket = k.storage_bucket
          AND d.condition_bucket = k.condition_bucket
    ) l3 ON true
    LEFT JOIN LATERAL (
        SELECT sum(d.sold_n)::int AS n_90d
        FROM ml.price_anchor_prior_daily_t0_v1_mv d
        WHERE d.sold_day < k.edited_day
          AND d.sold_day >= k.edited_day - 90
          AND d.generation = k.generation
          AND d.model_key = k.model_key
          AND d.storage_bucket = k.storage_bucket
    ) l4 ON true
    LEFT JOIN LATERAL (
        SELECT sum(d.sold_n)::int AS n_90d
        FROM ml.price_anchor_prior_daily_t0_v1_mv d
        WHERE d.sold_day < k.edited_day
          AND d.sold_day >= k.edited_day - 90
          AND d.generation = k.generation
          AND d.model_key = k.model_key
    ) l5 ON true
    LEFT JOIN LATERAL (
        SELECT sum(d.sold_n)::int AS n_90d
        FROM ml.price_anchor_prior_daily_t0_v1_mv d
        WHERE d.sold_day < k.edited_day
          AND d.sold_day >= k.edited_day - 90
          AND d.generation = k.generation
    ) l6 ON true
)
SELECT
    *,
    CASE
        WHEN l1_n_90d >= 30 THEN 1
        WHEN l2_n_90d >= 30 THEN 2
        WHEN l3_n_90d >= 30 THEN 3
        WHEN l4_n_90d >= 30 THEN 4
        WHEN l5_n_90d >= 30 THEN 5
        WHEN l6_n_90d >= 30 THEN 6
        ELSE NULL
    END AS source_level,
    CASE
        WHEN l1_n_90d >= 30 THEN l1_n_90d
        WHEN l2_n_90d >= 30 THEN l2_n_90d
        WHEN l3_n_90d >= 30 THEN l3_n_90d
        WHEN l4_n_90d >= 30 THEN l4_n_90d
        WHEN l5_n_90d >= 30 THEN l5_n_90d
        WHEN l6_n_90d >= 30 THEN l6_n_90d
        ELSE 0
    END AS source_n_90d
FROM key_counts;

CREATE UNIQUE INDEX IF NOT EXISTS price_anchor_target_keys_t0_v1_key_idx
    ON ml.price_anchor_target_keys_t0_v1_mv (key_id);
CREATE INDEX IF NOT EXISTS price_anchor_target_keys_t0_v1_lookup_idx
    ON ml.price_anchor_target_keys_t0_v1_mv (
        generation, edited_day, model_key, storage_bucket, condition_bucket, battery_bucket, damage_bucket
    );
CREATE INDEX IF NOT EXISTS price_anchor_target_keys_t0_v1_source_idx
    ON ml.price_anchor_target_keys_t0_v1_mv (source_level, source_n_90d);

ANALYZE ml.price_anchor_target_keys_t0_v1_mv;

CREATE MATERIALIZED VIEW ml.price_anchor_key_metrics_t0_v1_mv AS
WITH matched_history AS (
    SELECT k.key_id, k.source_level, h.*, k.edited_day AS target_edited_day
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
    SELECT k.key_id, k.source_level, h.*, k.edited_day AS target_edited_day
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
    SELECT k.key_id, k.source_level, h.*, k.edited_day AS target_edited_day
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
    SELECT k.key_id, k.source_level, h.*, k.edited_day AS target_edited_day
    FROM ml.price_anchor_target_keys_t0_v1_mv k
    JOIN ml.price_anchor_history_t0_v1_mv h
      ON k.source_level = 4
     AND h.sold_day < k.edited_day
     AND h.sold_day >= k.edited_day - 90
     AND h.generation = k.generation
     AND h.model_key = k.model_key
     AND h.storage_bucket = k.storage_bucket
    UNION ALL
    SELECT k.key_id, k.source_level, h.*, k.edited_day AS target_edited_day
    FROM ml.price_anchor_target_keys_t0_v1_mv k
    JOIN ml.price_anchor_history_t0_v1_mv h
      ON k.source_level = 5
     AND h.sold_day < k.edited_day
     AND h.sold_day >= k.edited_day - 90
     AND h.generation = k.generation
     AND h.model_key = k.model_key
    UNION ALL
    SELECT k.key_id, k.source_level, h.*, k.edited_day AS target_edited_day
    FROM ml.price_anchor_target_keys_t0_v1_mv k
    JOIN ml.price_anchor_history_t0_v1_mv h
      ON k.source_level = 6
     AND h.sold_day < k.edited_day
     AND h.sold_day >= k.edited_day - 90
     AND h.generation = k.generation
)
SELECT
    key_id,
    source_level,
    count(*) FILTER (WHERE sold_day >= target_edited_day - 30)::int AS n_30d,
    count(*)::int AS n_90d,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY ask_price)
        FILTER (WHERE sold_day >= target_edited_day - 30)::numeric AS ask_median_30d,
    percentile_cont(0.50) WITHIN GROUP (ORDER BY ask_price)::numeric AS ask_median_90d,
    percentile_cont(0.25) WITHIN GROUP (ORDER BY ask_price)::numeric AS ask_p25_90d,
    percentile_cont(0.75) WITHIN GROUP (ORDER BY ask_price)::numeric AS ask_p75_90d,
    avg(y_fast72) FILTER (WHERE sold_day >= target_edited_day - 30)::numeric AS fast72_rate_30d,
    avg(y_fast72)::numeric AS fast72_rate_90d,
    avg(duration_h) FILTER (WHERE sold_day >= target_edited_day - 30)::numeric AS avg_hours_30d,
    avg(duration_h)::numeric AS avg_hours_90d,
    avg(y_fast72)::numeric AS bucket_le3d_rate_90d,
    avg(y_3_7d)::numeric AS bucket_3_7d_rate_90d,
    avg(y_7_10d)::numeric AS bucket_7_10d_rate_90d,
    avg(y_10_21d)::numeric AS bucket_10_21d_rate_90d,
    avg(y_gt21d)::numeric AS bucket_gt21d_rate_90d
FROM matched_history
GROUP BY key_id, source_level;

CREATE UNIQUE INDEX IF NOT EXISTS price_anchor_key_metrics_t0_v1_key_idx
    ON ml.price_anchor_key_metrics_t0_v1_mv (key_id);
CREATE INDEX IF NOT EXISTS price_anchor_key_metrics_t0_v1_source_idx
    ON ml.price_anchor_key_metrics_t0_v1_mv (source_level, n_90d);

ANALYZE ml.price_anchor_key_metrics_t0_v1_mv;

CREATE MATERIALIZED VIEW ml.price_anchor_feature_store_t0_v1_mv AS
WITH target_rows AS (
    SELECT
        l.generation,
        l.listing_id,
        l.edited_date,
        (l.edited_date AT TIME ZONE 'Europe/Berlin')::date AS edited_day,
        lower(regexp_replace(coalesce(nullif(trim(l.model), ''), 'unknown'), '\s+', ' ', 'g')) AS model_key,
        CASE
            WHEN l.storage_gb IS NULL THEN 'stor_missing'
            WHEN l.storage_gb <= 128 THEN 'stor_le128'
            WHEN l.storage_gb <= 256 THEN 'stor_256'
            WHEN l.storage_gb <= 512 THEN 'stor_512'
            ELSE 'stor_1tbplus'
        END AS storage_bucket,
        CASE
            WHEN l.condition_score IS NULL THEN 'cond_missing'
            WHEN l.condition_score < 2 THEN 'cond_low'
            WHEN l.condition_score < 4 THEN 'cond_mid'
            ELSE 'cond_high'
        END AS condition_bucket,
        CASE
            WHEN l.battery_pct IS NULL THEN 'bat_missing'
            WHEN l.battery_pct < 80 THEN 'bat_lt80'
            WHEN l.battery_pct < 86 THEN 'bat_80_85'
            WHEN l.battery_pct < 91 THEN 'bat_86_90'
            ELSE 'bat_91plus'
        END AS battery_bucket,
        CASE
            WHEN k.max_visible_damage_level IS NULL THEN 'dmg_missing'
            WHEN k.max_visible_damage_level = 0 THEN 'dmg_0'
            WHEN k.max_visible_damage_level <= 5 THEN 'dmg_1_5'
            ELSE 'dmg_6plus'
        END AS damage_bucket,
        NULLIF(l.price, 0)::numeric AS ask_price
    FROM "device".device_listings l
    LEFT JOIN image_map.device_listing_image_k8_manifest_v1 k
      ON k.generation = l.generation
     AND k.listing_id = l.listing_id
    WHERE l.spam IS NULL
      AND l.edited_date IS NOT NULL
)
SELECT
    t.generation,
    t.listing_id,
    t.edited_date,
    COALESCE(m.n_30d, 0)::int AS pa__anchor_n_30d,
    COALESCE(m.n_90d, 0)::int AS pa__anchor_n_90d,
    m.ask_median_30d::double precision AS pa__ask_median_30d,
    m.ask_median_90d::double precision AS pa__ask_median_90d,
    m.ask_p25_90d::double precision AS pa__ask_p25_90d,
    m.ask_p75_90d::double precision AS pa__ask_p75_90d,
    CASE WHEN t.ask_price IS NOT NULL AND m.ask_median_30d > 0 THEN (t.ask_price / m.ask_median_30d)::double precision ELSE NULL END AS pa__price_to_anchor_30d,
    CASE WHEN t.ask_price IS NOT NULL AND m.ask_median_90d > 0 THEN (t.ask_price / m.ask_median_90d)::double precision ELSE NULL END AS pa__price_to_anchor_90d,
    CASE WHEN t.ask_price IS NOT NULL AND t.ask_price > 0 AND m.ask_median_90d > 0 THEN ln((t.ask_price / m.ask_median_90d)::double precision) ELSE NULL END AS pa__log_price_to_anchor_90d,
    m.fast72_rate_30d::double precision AS pa__prior_fast72_rate_30d,
    m.fast72_rate_90d::double precision AS pa__prior_fast72_rate_90d,
    m.avg_hours_30d::double precision AS pa__prior_avg_hours_30d,
    m.avg_hours_90d::double precision AS pa__prior_avg_hours_90d,
    m.bucket_le3d_rate_90d::double precision AS pa__prior_bucket_le3d_rate_90d,
    m.bucket_3_7d_rate_90d::double precision AS pa__prior_bucket_3_7d_rate_90d,
    m.bucket_7_10d_rate_90d::double precision AS pa__prior_bucket_7_10d_rate_90d,
    m.bucket_10_21d_rate_90d::double precision AS pa__prior_bucket_10_21d_rate_90d,
    m.bucket_gt21d_rate_90d::double precision AS pa__prior_bucket_gt21d_rate_90d,
    k.source_level::int AS pa__anchor_source_level_90d,
    (COALESCE(m.n_30d, 0) = 0)::int AS pa__anchor_missing_30d,
    (COALESCE(m.n_90d, 0) = 0)::int AS pa__anchor_missing_90d
FROM target_rows t
JOIN ml.price_anchor_target_keys_t0_v1_mv k
  ON k.generation = t.generation
 AND k.edited_day = t.edited_day
 AND k.model_key = t.model_key
 AND k.storage_bucket = t.storage_bucket
 AND k.condition_bucket = t.condition_bucket
 AND k.battery_bucket = t.battery_bucket
 AND k.damage_bucket = t.damage_bucket
LEFT JOIN ml.price_anchor_key_metrics_t0_v1_mv m
  ON m.key_id = k.key_id;

CREATE UNIQUE INDEX IF NOT EXISTS price_anchor_feature_store_t0_v1_identity_idx
    ON ml.price_anchor_feature_store_t0_v1_mv (generation, listing_id, edited_date);
CREATE INDEX IF NOT EXISTS price_anchor_feature_store_t0_v1_source_idx
    ON ml.price_anchor_feature_store_t0_v1_mv (pa__anchor_source_level_90d, pa__anchor_n_90d);

ANALYZE ml.price_anchor_feature_store_t0_v1_mv;
