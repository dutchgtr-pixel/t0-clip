-- Public Release: T0 remediation / T0-safe objects

CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- used for sha256 proofs
CREATE SCHEMA IF NOT EXISTS audit;        -- used for certification artifacts


DO $$
DECLARE
  min_day date;
  max_day date;
  lookback_days int := 730;  -- tune as needed
  listing regclass;
BEGIN
  -- Locate iphone_listings robustly regardless of schema case
  SELECT c.oid::regclass INTO listing
  FROM pg_class c
  JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE c.relname='iphone_listings'
    AND c.relkind IN ('r','p','f')
  ORDER BY (n.nspname='iPhone') DESC, (n.nspname='iphone') DESC, n.nspname
  LIMIT 1;

  IF listing IS NULL THEN
    RAISE EXCEPTION 'Could not locate iphone_listings in pg_catalog';
  END IF;

  EXECUTE format(
    'SELECT min(edited_date::date), max(edited_date::date) FROM %s WHERE edited_date IS NOT NULL',
    listing
  ) INTO min_day, max_day;

  IF max_day IS NULL THEN
    RAISE EXCEPTION 'No edited_date found in %s', listing;
  END IF;

  min_day := GREATEST(min_day, (max_day - lookback_days));

  RAISE NOTICE 'Using listings table: %', listing::text;
  RAISE NOTICE 'anchor_day range: % -> %', min_day, max_day;

  DROP MATERIALIZED VIEW IF EXISTS ml.tom_speed_anchor_asof_v1_mv CASCADE;

  CREATE MATERIALIZED VIEW ml.tom_speed_anchor_asof_v1_mv AS
  WITH anchors AS (
    SELECT gs::date AS anchor_day
    FROM generate_series(min_day::date, max_day::date, interval '1 day') gs
  ),
  sold AS (
    SELECT
      f.generation,
      CASE
        WHEN f.storage_gb >= 900 THEN 1024
        WHEN f.storage_gb >= 500 THEN 512
        WHEN f.storage_gb >= 250 THEN 256
        WHEN f.storage_gb >= 120 THEN 128
        ELSE f.storage_gb
      END AS sbucket,
      CASE
        WHEN f.ptv_final < 0.50 THEN NULL::int
        WHEN f.ptv_final < 0.80 THEN 1
        WHEN f.ptv_final < 0.90 THEN 2
        WHEN f.ptv_final < 1.00 THEN 3
        WHEN f.ptv_final < 1.10 THEN 4
        WHEN f.ptv_final < 1.20 THEN 5
        WHEN f.ptv_final < 1.40 THEN 6
        ELSE 7
      END AS ptv_bucket,
      d.sold_day,
      d.duration_hours
    FROM ml.sold_durations_v1_mv d
    JOIN ml.tom_features_v1_enriched_mv f USING (listing_id)
    WHERE f.ptv_final IS NOT NULL
      AND f.ptv_final BETWEEN 0.50 AND 2.50
  ),
  recent AS (
    SELECT
      a.anchor_day,
      s.generation,
      s.sbucket,
      s.ptv_bucket,
      s.sold_day,
      s.duration_hours,

      -- Recency weight is anchored to anchor_day (NOT CURRENT_DATE)
      (0.5 ^ ((a.anchor_day - s.sold_day)::numeric / 90.0)) AS w,

      CASE WHEN s.duration_hours <=  24    THEN 1::numeric ELSE 0::numeric END AS is_fast24,
      CASE WHEN s.duration_hours <= 168    THEN 1::numeric ELSE 0::numeric END AS is_fast7,
      CASE WHEN s.duration_hours >  21*24 THEN 1::numeric ELSE 0::numeric END AS is_slow21
    FROM anchors a
    JOIN sold s
      ON s.ptv_bucket IS NOT NULL
     -- Embargo: never use sales on/after anchor_day
     AND s.sold_day < a.anchor_day
     AND s.sold_day >= (a.anchor_day - 90)
  ),
  agg AS (
    SELECT
      anchor_day,
      generation,
      sbucket,
      ptv_bucket,

      SUM(w) AS sum_w,
      SUM(w * is_fast7)  / NULLIF(SUM(w),0) AS speed_fast7_anchor,
      SUM(w * is_fast24) / NULLIF(SUM(w),0) AS speed_fast24_anchor,
      SUM(w * is_slow21) / NULLIF(SUM(w),0) AS speed_slow21_anchor,

      PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_hours) AS speed_median_hours_ptv,

      CASE WHEN SUM(w*w) > 0 THEN (SUM(w)*SUM(w)) / SUM(w*w) ELSE 0 END AS speed_n_eff_ptv,

      -- Proof columns
      MIN(sold_day) AS min_sold_day_used,
      MAX(sold_day) AS max_sold_day_used
    FROM recent
    GROUP BY anchor_day, generation, sbucket, ptv_bucket
  )
  SELECT * FROM agg;

  CREATE UNIQUE INDEX IF NOT EXISTS tom_speed_anchor_asof_v1_uq
    ON ml.tom_speed_anchor_asof_v1_mv (anchor_day, generation, sbucket, ptv_bucket);

  CREATE INDEX IF NOT EXISTS tom_speed_anchor_asof_v1_anchor_idx
    ON ml.tom_speed_anchor_asof_v1_mv (anchor_day);

  ANALYZE ml.tom_speed_anchor_asof_v1_mv;
END $$;


-- No future relative to anchor_day (must be 0)
SELECT COUNT(*) AS violations_future_sales
FROM ml.tom_speed_anchor_asof_v1_mv
WHERE max_sold_day_used >= anchor_day;

-- No time-of-query tokens in definition (must be false/false)
SELECT
  strpos(lower(pg_get_viewdef('ml.tom_speed_anchor_asof_v1_mv'::regclass,true)),'current_date')>0 AS has_current_date,
  strpos(lower(pg_get_viewdef('ml.tom_speed_anchor_asof_v1_mv'::regclass,true)),'now(')>0          AS has_now;


DROP MATERIALIZED VIEW IF EXISTS ml.tom_features_v1_enriched_speed_t0_v1_mv CASCADE;

CREATE MATERIALIZED VIEW ml.tom_features_v1_enriched_speed_t0_v1_mv AS
WITH f AS (
  SELECT
    f.*,

    CASE
      WHEN f.storage_gb >= 900 THEN 1024
      WHEN f.storage_gb >= 500 THEN 512
      WHEN f.storage_gb >= 250 THEN 256
      WHEN f.storage_gb >= 120 THEN 128
      ELSE f.storage_gb
    END AS sbucket,

    CASE
      WHEN f.ptv_final IS NULL OR f.ptv_final < 0.50 OR f.ptv_final > 2.50 THEN NULL::integer
      WHEN f.ptv_final < 0.80 THEN 1
      WHEN f.ptv_final < 0.90 THEN 2
      WHEN f.ptv_final < 1.00 THEN 3
      WHEN f.ptv_final < 1.10 THEN 4
      WHEN f.ptv_final < 1.20 THEN 5
      WHEN f.ptv_final < 1.40 THEN 6
      ELSE 7
    END AS ptv_bucket
  FROM ml.tom_features_v1_enriched_clean_mv f
)
SELECT
  f.*,
  s.speed_fast7_anchor,
  s.speed_fast24_anchor,
  s.speed_slow21_anchor,
  s.speed_median_hours_ptv,
  s.speed_n_eff_ptv
FROM f
LEFT JOIN ml.tom_speed_anchor_asof_v1_mv s
  ON s.anchor_day = f.edited_date::date
 AND s.generation = f.generation
 AND s.sbucket    = f.sbucket
 AND s.ptv_bucket = f.ptv_bucket;

CREATE UNIQUE INDEX IF NOT EXISTS tom_features_v1_enriched_speed_t0_v1_uq
  ON ml.tom_features_v1_enriched_speed_t0_v1_mv (listing_id);

ANALYZE ml.tom_features_v1_enriched_speed_t0_v1_mv;


DROP MATERIALIZED VIEW IF EXISTS ml.tom_features_v1_enriched_ai_clean_t0_v1_mv CASCADE;

CREATE MATERIALIZED VIEW ml.tom_features_v1_enriched_ai_clean_t0_v1_mv AS
SELECT
  base.*,

  ai.sale_mode            AS sale_mode_ai,
  ai.owner_type           AS owner_type_ai,
  ai.can_ship,
  ai.pickup_only,
  ai.repair_provider      AS repair_provider_ai,
  ai.vat_invoice,
  ai.first_owner,
  ai.used_with_case_claim,
  ai.negotiability_ai     AS negotiability_ai_enhanced,
  ai.urgency_ai           AS urgency_ai_enhanced,
  ai.lqs_textonly         AS lqs_textonly_ai,
  ai.opening_offer_nok    AS opening_offer_nok_ai,
  ai.storage_gb_fixed_ai  AS storage_gb_fixed_ai_enhanced,

  -- audit/proof fields (not meant to be model features)
  ai.updated_at           AS ai_updated_at,
  ai.model                AS ai_model,
  ai.version              AS ai_version

FROM ml.tom_features_v1_enriched_speed_t0_v1_mv base

LEFT JOIN LATERAL (
  SELECT *
  FROM "iPhone".iphone_ai_enrich ai
  WHERE ai.listing_id = base.listing_id
    AND ai.updated_at <= (base.edited_date + interval '24 hours')
  ORDER BY ai.updated_at DESC
  LIMIT 1
) ai ON true

WHERE ai.sale_mode IS NULL OR ai.sale_mode <> 'not_sold';

CREATE UNIQUE INDEX IF NOT EXISTS tom_features_v1_enriched_ai_clean_t0_v1_uq
  ON ml.tom_features_v1_enriched_ai_clean_t0_v1_mv (listing_id);

ANALYZE ml.tom_features_v1_enriched_ai_clean_t0_v1_mv;

-- AI window violations (should be 0 for SLA=24h)
SELECT COUNT(*) AS ai_window_violations
FROM ml.tom_features_v1_enriched_ai_clean_t0_v1_mv
WHERE ai_updated_at IS NOT NULL
  AND ai_updated_at > (edited_date + interval '24 hours');


DO $$
DECLARE rid bigint;
BEGIN
  SELECT release_id INTO rid FROM ref.geo_mapping_current;
  RAISE NOTICE 'Pinning geo mapping release_id = %', rid;

  DROP VIEW IF EXISTS ref.geo_mapping_pinned_super_metro_v4_v1;

  EXECUTE format(
    'CREATE VIEW ref.geo_mapping_pinned_super_metro_v4_v1 AS SELECT %s::bigint AS release_id',
    rid
  );
END $$;


CREATE OR REPLACE VIEW ml.geo_dim_super_metro_v4_pinned_v1 AS
WITH base AS (
  SELECT
    f.listing_id,
    ref.norm_postal_code(f.postal_code) AS postal_code_norm,
    ref.norm_city(f.location_city)      AS location_city_norm
  FROM ml.tom_features_v1_mv f
),
pin AS (
  SELECT release_id FROM ref.geo_mapping_pinned_super_metro_v4_v1
)
SELECT
  b.listing_id,
  pin.release_id AS geo_release_id,
  COALESCE(pc.region, cc.region, 'unknown'::text) AS region_geo,
  COALESCE(pc.pickup_metro_30_200, cc.pickup_metro_30_200,
           'other_'::text || lower(COALESCE(pc.region, cc.region, 'unknown'::text))) AS pickup_metro_30_200_geo,
  COALESCE(pc.super_metro_v4, cc.super_metro_v4,
           'other_'::text || lower(COALESCE(pc.region, cc.region, 'unknown'::text))) AS super_metro_v4_geo,
  CASE
    WHEN pc.postal_code IS NOT NULL THEN 'postal'::text
    WHEN cc.location_city_norm IS NOT NULL THEN 'city'::text
    WHEN b.postal_code_norm IS NULL AND b.location_city_norm IS NULL THEN 'missing_keys'::text
    ELSE 'unmapped'::text
  END AS geo_match_method
FROM base b
CROSS JOIN pin
LEFT JOIN ref.postal_code_to_super_metro pc
  ON pc.release_id = pin.release_id AND pc.postal_code = b.postal_code_norm
LEFT JOIN ref.city_to_super_metro cc
  ON cc.release_id = pin.release_id AND cc.location_city_norm = b.location_city_norm;


CREATE OR REPLACE VIEW ml.geo_dim_super_metro_v4_t0_v1 AS
SELECT * FROM ml.geo_dim_super_metro_v4_pinned_v1;


CREATE OR REPLACE VIEW ml.tom_features_v1_enriched_ai_clean_t0_read_v AS
SELECT
  f.*,
  g.geo_release_id,
  g.region_geo,
  g.pickup_metro_30_200_geo,
  g.super_metro_v4_geo,
  g.geo_match_method
FROM ml.tom_features_v1_enriched_ai_clean_t0_v1_mv f
LEFT JOIN ml.geo_dim_super_metro_v4_t0_v1 g
  ON g.listing_id = f.listing_id;


DO $$
DECLARE
  sla_text text := '24 hours';
  done_exprs text;
  asset_ts_col text;
  cols text;
BEGIN
  -- Build expressions for all *_done_at timestamp columns
  SELECT string_agg(
           format('COALESCE(f.%I::timestamptz, ''epoch''::timestamptz)', column_name),
           ', ' ORDER BY ordinal_position
         )
  INTO done_exprs
  FROM information_schema.columns
  WHERE table_schema='ml'
    AND table_name='iphone_image_features_v1'
    AND column_name LIKE '%\_done\_at' ESCAPE '\'
    AND data_type LIKE 'timestamp%';

  IF done_exprs IS NULL THEN
    RAISE EXCEPTION 'No *_done_at timestamp columns found on ml.iphone_image_features_v1';
  END IF;

  -- Choose a timestamp column from iphone_image_assets (prefer created_at/updated_at/scraped_at/etc.)
  SELECT c.column_name
  INTO asset_ts_col
  FROM information_schema.columns c
  WHERE c.table_schema='iPhone'
    AND c.table_name='iphone_image_assets'
    AND c.data_type LIKE 'timestamp%'
  ORDER BY
    CASE
      WHEN c.column_name IN ('created_at','updated_at','scraped_at','observed_at','ingested_at','captured_at') THEN 0
      ELSE 1
    END,
    c.ordinal_position
  LIMIT 1;

  -- Build gated select list for all unified image columns (except keys)
  SELECT string_agg(
           format(
             'CASE WHEN within_sla THEN u.%I ELSE NULL END AS %I',
             column_name, column_name
           ),
           E',\n      ' ORDER BY ordinal_position
         )
  INTO cols
  FROM information_schema.columns
  WHERE table_schema='ml'
    AND table_name='iphone_image_features_unified_v1'
    AND column_name NOT IN ('generation','listing_id');

  DROP MATERIALIZED VIEW IF EXISTS ml.iphone_image_features_unified_t0_v1_mv CASCADE;

  EXECUTE format($fmt$
    CREATE MATERIALIZED VIEW ml.iphone_image_features_unified_t0_v1_mv AS
    WITH base AS (
      SELECT generation, listing_id, edited_date
      FROM ml.tom_features_v1_enriched_ai_clean_t0_v1_mv
    ),
    status AS (
      SELECT
        generation,
        listing_id,
        GREATEST(%s) AS img_max_done_at
      FROM ml.iphone_image_features_v1 f
      GROUP BY generation, listing_id
    ),
    assets AS (
      SELECT
        generation,
        listing_id,
        MAX(%I)::timestamptz AS img_max_asset_at
      FROM "iPhone".iphone_image_assets
      GROUP BY generation, listing_id
    )
    SELECT
      b.generation,
      b.listing_id,
      b.edited_date,
      (
        COALESCE(s.img_max_done_at,  'epoch'::timestamptz) <= (b.edited_date::timestamptz + %L::interval)
        AND
        COALESCE(a.img_max_asset_at, 'epoch'::timestamptz) <= (b.edited_date::timestamptz + %L::interval)
      ) AS img_within_sla,
      %s
    FROM base b
    LEFT JOIN ml.iphone_image_features_unified_v1 u USING (generation, listing_id)
    LEFT JOIN status s USING (generation, listing_id)
    LEFT JOIN assets a USING (generation, listing_id)
    CROSS JOIN LATERAL (
      SELECT (
        COALESCE(s.img_max_done_at,  'epoch'::timestamptz) <= (b.edited_date::timestamptz + %L::interval)
        AND
        COALESCE(a.img_max_asset_at, 'epoch'::timestamptz) <= (b.edited_date::timestamptz + %L::interval)
      ) AS within_sla
    ) __g
  $fmt$, done_exprs, asset_ts_col, sla_text, sla_text, cols, sla_text, sla_text);

  CREATE UNIQUE INDEX IF NOT EXISTS iphone_image_features_unified_t0_v1_uq
    ON ml.iphone_image_features_unified_t0_v1_mv (generation, listing_id);

  ANALYZE ml.iphone_image_features_unified_t0_v1_mv;
END $$;


SELECT img_within_sla, COUNT(*) AS n
FROM ml.iphone_image_features_unified_t0_v1_mv
GROUP BY 1
ORDER BY 1;

SELECT
  strpos(lower(pg_get_viewdef('ml.iphone_image_features_unified_t0_v1_mv'::regclass,true)),'current_date')>0 AS has_current_date,
  strpos(lower(pg_get_viewdef('ml.iphone_image_features_unified_t0_v1_mv'::regclass,true)),'now(')>0          AS has_now;


DO $$
DECLARE
  cols text;
  ok_expr text := 'COALESCE(i.img_within_sla, false)';
BEGIN
  SELECT string_agg(
           format('CASE WHEN %s THEN f.%I ELSE NULL END AS %I', ok_expr, column_name, column_name),
           E',\n      ' ORDER BY ordinal_position
         )
  INTO cols
  FROM information_schema.columns
  WHERE table_schema='ml'
    AND table_name='v_damage_fusion_features_v2_scored'
    AND column_name NOT IN ('generation','listing_id');

  IF cols IS NULL THEN
    RAISE EXCEPTION 'Could not read columns for ml.v_damage_fusion_features_v2_scored';
  END IF;

  DROP MATERIALIZED VIEW IF EXISTS ml.v_damage_fusion_features_v2_scored_t0_v1_mv CASCADE;

  EXECUTE format($fmt$
    CREATE MATERIALIZED VIEW ml.v_damage_fusion_features_v2_scored_t0_v1_mv AS
    WITH base AS (
      SELECT generation, listing_id, edited_date
      FROM ml.tom_features_v1_enriched_ai_clean_t0_v1_mv
    )
    SELECT
      b.generation,
      b.listing_id,
      b.edited_date,
      %s AS img_within_sla,
      %s
    FROM base b
    LEFT JOIN ml.iphone_image_features_unified_t0_v1_mv i USING (generation, listing_id)
    LEFT JOIN ml.v_damage_fusion_features_v2_scored f USING (generation, listing_id)
  $fmt$, ok_expr, cols);

  CREATE UNIQUE INDEX IF NOT EXISTS damage_fusion_v2_scored_t0_v1_uq
    ON ml.v_damage_fusion_features_v2_scored_t0_v1_mv (generation, listing_id);

  ANALYZE ml.v_damage_fusion_features_v2_scored_t0_v1_mv;
END $$;


DO $$
DECLARE
  q text;
BEGIN
  q := pg_get_viewdef('ml.iphone_device_meta_encoded_v1'::regclass, true);

  q := regexp_replace(
        q,
        '\mml\.tom_features_v1_enriched_ai_clean_mv\M',
        'ml.tom_features_v1_enriched_ai_clean_t0_v1_mv',
        'g'
      );

  q := regexp_replace(
        q,
        '\mml\.iphone_image_features_unified_v1\M',
        'ml.iphone_image_features_unified_t0_v1_mv',
        'g'
      );

  EXECUTE 'CREATE OR REPLACE VIEW ml.iphone_device_meta_encoded_t0_v1 AS ' || q;
END $$;


DO $$
DECLARE
  cols text;
BEGIN
  SELECT string_agg(
           format('m.%I AS %I', column_name, column_name),
           E',\n  ' ORDER BY ordinal_position
         )
  INTO cols
  FROM information_schema.columns
  WHERE table_schema='ml'
    AND table_name='iphone_device_meta_encoded_t0_v1'
    AND column_name NOT IN ('generation','listing_id');

  IF cols IS NULL THEN
    RAISE EXCEPTION 'Could not introspect columns for ml.iphone_device_meta_encoded_t0_v1';
  END IF;

  DROP MATERIALIZED VIEW IF EXISTS ml.iphone_device_meta_encoded_t0_v1_mv CASCADE;

  EXECUTE format($fmt$
    CREATE MATERIALIZED VIEW ml.iphone_device_meta_encoded_t0_v1_mv AS
    WITH base AS (
      SELECT generation, listing_id, edited_date
      FROM ml.tom_features_v1_enriched_ai_clean_t0_v1_mv
    )
    SELECT
      b.generation,
      b.listing_id,
      b.edited_date,
      %s
    FROM base b
    LEFT JOIN ml.iphone_device_meta_encoded_t0_v1 m
      ON m.generation=b.generation AND m.listing_id=b.listing_id
  $fmt$, cols);

  CREATE UNIQUE INDEX IF NOT EXISTS iphone_device_meta_encoded_t0_v1_uq
    ON ml.iphone_device_meta_encoded_t0_v1_mv(generation,listing_id);

  ANALYZE ml.iphone_device_meta_encoded_t0_v1_mv;
END $$;


DO $$
DECLARE
  img_cols text;
  dmg_cols text;
  dev_cols text;
  sql text;
BEGIN
  -- Image cols (prefix img__) with explicit gating
  SELECT COALESCE(
    E',\n      ' || string_agg(
      format(
        'CASE WHEN COALESCE(i.img_within_sla,false) THEN i.%I ELSE NULL END AS img__%I',
        a.attname, a.attname
      ),
      E',\n      ' ORDER BY a.attnum
    ),
    ''
  )
  INTO img_cols
  FROM pg_attribute a
  WHERE a.attrelid = 'ml.iphone_image_features_unified_t0_v1_mv'::regclass
    AND a.attnum > 0 AND NOT a.attisdropped
    AND a.attname NOT IN ('generation','listing_id','edited_date','img_within_sla');

  -- Damage cols (prefix dmg__) gated off the same SLA flag
  SELECT COALESCE(
    E',\n      ' || string_agg(
      format(
        'CASE WHEN COALESCE(i.img_within_sla,false) THEN d.%I ELSE NULL END AS dmg__%I',
        a.attname, a.attname
      ),
      E',\n      ' ORDER BY a.attnum
    ),
    ''
  )
  INTO dmg_cols
  FROM pg_attribute a
  WHERE a.attrelid = 'ml.v_damage_fusion_features_v2_scored_t0_v1_mv'::regclass
    AND a.attnum > 0 AND NOT a.attisdropped
    AND a.attname NOT IN ('generation','listing_id','edited_date','img_within_sla');

  -- Device meta cols (prefix dev__) (no SLA gating)
  SELECT COALESCE(
    E',\n      ' || string_agg(
      format('m.%I AS dev__%I', a.attname, a.attname),
      E',\n      ' ORDER BY a.attnum
    ),
    ''
  )
  INTO dev_cols
  FROM pg_attribute a
  WHERE a.attrelid = 'ml.iphone_device_meta_encoded_t0_v1_mv'::regclass
    AND a.attnum > 0 AND NOT a.attisdropped
    AND a.attname NOT IN ('generation','listing_id','edited_date');

  sql := format($fmt$
    CREATE OR REPLACE VIEW ml.survival_feature_store_t0_v1_v AS
    SELECT
      b.*,
      COALESCE(i.img_within_sla,false) AS img_within_sla%s%s%s
    FROM ml.tom_features_v1_enriched_ai_clean_t0_read_v b
    LEFT JOIN ml.iphone_image_features_unified_t0_v1_mv i
      ON i.generation=b.generation AND i.listing_id=b.listing_id
    LEFT JOIN ml.v_damage_fusion_features_v2_scored_t0_v1_mv d
      ON d.generation=b.generation AND d.listing_id=b.listing_id
    LEFT JOIN ml.iphone_device_meta_encoded_t0_v1_mv m
      ON m.generation=b.generation AND m.listing_id=b.listing_id;
  $fmt$,
  img_cols,
  dmg_cols,
  dev_cols);

  EXECUTE sql;
END $$;


-- Image leakage check: use regex or escaped LIKE (NOT 'img__%')
SELECT COUNT(*) AS img_leak_rows
FROM ml.survival_feature_store_t0_v1_v s
WHERE NOT s.img_within_sla
  AND EXISTS (
    SELECT 1
    FROM jsonb_each(to_jsonb(s)) kv
    WHERE kv.key ~ '^img__'
      AND kv.value <> 'null'::jsonb
  );

SELECT COUNT(*) AS dmg_leak_rows
FROM ml.survival_feature_store_t0_v1_v s
WHERE NOT s.img_within_sla
  AND EXISTS (
    SELECT 1
    FROM jsonb_each(to_jsonb(s)) kv
    WHERE kv.key ~ '^dmg__'
      AND kv.value <> 'null'::jsonb
  );


COMMENT ON MATERIALIZED VIEW ml.tom_speed_anchor_v1_mv IS
'LEGACY / NOT T0 SAFE: contains time-of-query logic. Replaced by ml.tom_speed_anchor_asof_v1_mv. Do not use for training.';

COMMENT ON MATERIALIZED VIEW ml.tom_features_v1_enriched_speed_mv IS
'LEGACY / NOT T0 SAFE: depends on ml.tom_speed_anchor_v1_mv. Use ml.tom_features_v1_enriched_speed_t0_v1_mv instead.';

COMMENT ON MATERIALIZED VIEW ml.tom_features_v1_enriched_ai_clean_mv IS
'LEGACY / NOT T0 SAFE: superseded by ml.tom_features_v1_enriched_ai_clean_t0_v1_mv.';
