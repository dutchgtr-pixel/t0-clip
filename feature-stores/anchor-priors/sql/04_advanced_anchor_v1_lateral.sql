-- Advanced anchor v1 (SQL fragment; correlated LATERAL)
--
-- Public-sanitized version.
-- This file is a query fragment intended to be embedded into a larger dataset assembly query.
--
-- Conventions:
--   - `b` is the active/base listing row at T0
--   - `public.listings` is the SOLD history table (adapt as needed)
--   - `listing_id` is the primary key
--
-- Optional objects (remove joins if not applicable):
--   - public.listing_image_assets
--   - ml.listing_image_features_v1
--
-- NEW: Advanced anchor v1 @ t0 (price + duration + support + level)
      anc_smart.anchor_price_smart::numeric,
      anc_smart.anchor_tts_median_h::numeric,
      anc_smart.anchor_n_support::int,
      anc_smart.anchor_level_k::int,
      CASE WHEN anc_smart.anchor_price_smart > 0 THEN b.price::numeric / anc_smart.anchor_price_smart ELSE NULL END AS ptv_anchor_smart

    FROM base b
    LEFT JOIN y USING (listing_id)

    -- NEW: image counts per listing (no matview, treat "no assets" as NULL/missing)
    LEFT JOIN LATERAL (
      SELECT
        CASE
          WHEN COUNT(*) = 0 THEN NULL::int
          ELSE COUNT(*)
        END AS image_count,
        CASE
          WHEN COUNT(*) FILTER (
                 WHERE caption_text IS NOT NULL AND caption_text <> ''
               ) = 0
          THEN NULL::int
          ELSE COUNT(*) FILTER (
                 WHERE caption_text IS NOT NULL AND caption_text <> ''
               )
        END AS caption_count
      FROM public.listing_image_assets ia
      WHERE ia.generation = b.generation
        AND ia.listing_id    = b.listing_id
    ) img ON TRUE

    -- NEW: battery % from screenshots (vision model; NULL if no screenshot / no value)
    LEFT JOIN LATERAL (
      SELECT
        MAX(i.battery_health_pct_img) AS battery_pct_img
      FROM ml.listing_image_features_v1 i
      WHERE i.feature_version = 1
        AND i.battery_screenshot IS TRUE
        AND i.battery_health_pct_img IS NOT NULL
        AND i.generation = b.generation
        AND i.listing_id    = b.listing_id
    ) batt ON TRUE

    -- Old strict anchor logic (model×storage×CS×SEV, 30/60d blend, 5d embargo)
    LEFT JOIN LATERAL (
      WITH t0 AS (SELECT b.edited_date::date AS d),
      sold30 AS (
        SELECT
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY s.sold_price::numeric) AS med,
          COUNT(*)::int AS n
        FROM public.listings s, t0
        WHERE s.spam IS NULL
          AND s.status = 'sold'
          AND s.sold_price IS NOT NULL
          AND s.sold_date::date >= t0.d - INTERVAL '35 days'
          AND s.sold_date::date <  t0.d - INTERVAL '5 days'
          AND s.generation = b.generation
          AND LOWER(REGEXP_REPLACE(s.model,'\\s+',' ','g')) = b.model_norm
          AND (CASE
                 WHEN s.storage_gb >= 900 THEN 1024
                 WHEN s.storage_gb >= 500 THEN 512
                 WHEN s.storage_gb >= 250 THEN 256
                 WHEN s.storage_gb >= 120 THEN 128
                 ELSE s.storage_gb
               END) = b.sbucket
          AND (
                (COALESCE(s.condition_score,0) IN (0.7,0.9,1.0)
                 AND COALESCE(s.damage_severity_ai,0) = b.sev)
                OR (COALESCE(s.condition_score,0)=0.0
                    AND COALESCE(s.damage_severity_ai,0)=0 AND b.sev=0)
              )
      ),
      sold60 AS (
        SELECT
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY s.sold_price::numeric) AS med,
          COUNT(*)::int AS n
        FROM public.listings s, t0
        WHERE s.spam IS NULL
          AND s.status = 'sold'
          AND s.sold_price IS NOT NULL
          AND s.sold_date::date >= t0.d - INTERVAL '65 days'
          AND s.sold_date::date <  t0.d - INTERVAL '5 days'
          AND s.generation = b.generation
          AND LOWER(REGEXP_REPLACE(s.model,'\\s+',' ','g')) = b.model_norm
          AND (CASE
                 WHEN s.storage_gb >= 900 THEN 1024
                 WHEN s.storage_gb >= 500 THEN 512
                 WHEN s.storage_gb >= 250 THEN 256
                 WHEN s.storage_gb >= 120 THEN 128
                 ELSE s.storage_gb
               END) = b.sbucket
          AND (
                (COALESCE(s.condition_score,0) IN (0.7,0.9,1.0)
                 AND COALESCE(s.damage_severity_ai,0) = b.sev)
                OR (COALESCE(s.condition_score,0)=0.0
                    AND COALESCE(s.damage_severity_ai,0)=0 AND b.sev=0)
              )
      )
      SELECT
        (SELECT med FROM sold30) AS anchor_30d_t0,
        (SELECT n   FROM sold30) AS n30_t0,
        (SELECT med FROM sold60) AS anchor_60d_t0,
        (SELECT n   FROM sold60) AS n60_t0,
        CASE
          WHEN COALESCE((SELECT n FROM sold30),0) >= 10 THEN (SELECT med FROM sold30)
          WHEN COALESCE((SELECT n FROM sold30),0) + COALESCE((SELECT n FROM sold60),0) > 0 THEN
            (
              ((COALESCE((SELECT n FROM sold30),0) + 4) * COALESCE((SELECT med FROM sold30),0)) +
              ((COALESCE((SELECT n FROM sold60),0) + 2) * COALESCE((SELECT med FROM sold60),0))
            ) / NULLIF(((COALESCE((SELECT n FROM sold30),0) + 4)
                        + (COALESCE((SELECT n FROM sold60),0) + 2)), 0)
          ELSE NULL
        END AS anchor_blend_t0
    ) a ON TRUE

    -- NEW: Advanced anchor v1 @ t0 (365d lookback, 5d embargo), never mixing model/gen/storage.
    LEFT JOIN LATERAL (
      WITH t0 AS (SELECT b.edited_date::date AS d),

      -- Sold comps in the leak-safe window; battery comes from feature-store view to keep hierarchy consistent.
      sold_s AS (
        SELECT
          s.sold_price::numeric AS sold_price,
          EXTRACT(EPOCH FROM (s.sold_date - s.edited_date))/3600.0 AS duration_h,
          s.generation,
          LOWER(REGEXP_REPLACE(s.model,'\\s+',' ','g')) AS model_norm,
          CASE
            WHEN s.storage_gb >= 1900
                 AND s.generation = 17
                 AND LOWER(REGEXP_REPLACE(s.model,'\\s+',' ','g')) LIKE '%pro%'
              THEN 2048
            WHEN s.storage_gb >= 900 THEN 1024
            WHEN s.storage_gb >= 500 THEN  512
            WHEN s.storage_gb >= 250 THEN  256
            WHEN s.storage_gb >= 120 THEN  128
            ELSE NULL
          END AS sbucket,
          CASE
            WHEN COALESCE(s.condition_score,0) >= 0.99 THEN 1.0
            WHEN COALESCE(s.condition_score,0) >= 0.90 THEN 0.9
            WHEN COALESCE(s.condition_score,0) >= 0.70 THEN 0.7
            WHEN COALESCE(s.condition_score,0) >= 0.50 THEN 0.5
            WHEN COALESCE(s.condition_score,0) >= 0.20 THEN 0.2
            ELSE 0.0
          END AS cs_bucket,
          COALESCE(s.damage_severity_ai,0)::int AS sev,
          fv.battery_pct_effective AS batt_pct_eff,
          fv.seller_rating::double precision AS seller_rating,
          fv.review_count,
          fv.member_since_year
        FROM public.listings s
        JOIN ml.listing_features_v1_mv fv USING (listing_id)
        JOIN t0 ON TRUE
        WHERE s.spam IS NULL
          AND s.status = 'sold'
          AND s.sold_price IS NOT NULL
          AND s.sold_date::date >= t0.d - INTERVAL '365 days'
          AND s.sold_date::date <  t0.d - INTERVAL '5 days'
          AND s.edited_date IS NOT NULL
      ),

      -- Buckets: battery + trust
      s2 AS (
        SELECT
          sold_price, duration_h, generation, model_norm, sbucket, cs_bucket, sev,
          CASE
            WHEN batt_pct_eff IS NULL OR batt_pct_eff = 0 THEN 'B_MISSING'
            WHEN batt_pct_eff <  80 THEN 'B_LOW'
            WHEN batt_pct_eff <  85 THEN 'B80_84'
            WHEN batt_pct_eff <  90 THEN 'B85_89'
            WHEN batt_pct_eff <  95 THEN 'B90_94'
            ELSE 'B95_100'
          END AS battery_bucket,
          CASE
            WHEN seller_rating >= 9.7
             AND review_count >= 50
             AND member_since_year <= EXTRACT(YEAR FROM (SELECT d FROM t0)) - 3 THEN 'HIGH'
            WHEN seller_rating >= 9.0
             AND review_count >= 10 THEN 'MED'
            ELSE 'LOW'
          END AS trust_tier

        FROM sold_s
        WHERE sbucket IS NOT NULL
      ),

      -- Cohort keys for the active listing
      base_keys AS (
        SELECT
          b.generation AS generation,
          b.model_norm AS model_norm,
          CASE
            WHEN b.storage_gb >= 1900
                 AND b.generation = 17
                 AND b.model_norm LIKE '%pro%'
              THEN 2048
            WHEN b.storage_gb >= 900 THEN 1024
            WHEN b.storage_gb >= 500 THEN  512
            WHEN b.storage_gb >= 250 THEN  256
            WHEN b.storage_gb >= 120 THEN  128
            ELSE NULL
          END AS sbucket,
          CASE
            WHEN COALESCE(b.condition_score,0) >= 0.99 THEN 1.0
            WHEN COALESCE(b.condition_score,0) >= 0.90 THEN 0.9
            WHEN COALESCE(b.condition_score,0) >= 0.70 THEN 0.7
            WHEN COALESCE(b.condition_score,0) >= 0.50 THEN 0.5
            WHEN COALESCE(b.condition_score,0) >= 0.20 THEN 0.2
            ELSE 0.0
          END AS cs_bucket,
          COALESCE(b.sev,0)::int AS sev,
          CASE
            WHEN b.battery_pct_effective IS NULL OR b.battery_pct_effective = 0 THEN 'B_MISSING'
            WHEN b.battery_pct_effective <  80 THEN 'B_LOW'
            WHEN b.battery_pct_effective <  85 THEN 'B80_84'
            WHEN b.battery_pct_effective <  90 THEN 'B85_89'
            WHEN b.battery_pct_effective <  95 THEN 'B90_94'
            ELSE 'B95_100'
          END AS battery_bucket,
          CASE
            WHEN b.seller_rating >= 9.7
             AND b.review_count >= 50
             AND b.member_since_year <= EXTRACT(YEAR FROM (SELECT d FROM t0)) - 3 THEN 'HIGH'
            WHEN b.seller_rating >= 9.0
             AND b.review_count >= 10 THEN 'MED'
            ELSE 'LOW'
          END AS trust_tier
      ),

      -- lock model/gen/storage to the active listing exactly
      s3 AS (
        SELECT s2.*
        FROM s2
        JOIN base_keys k
          ON s2.generation=k.generation
         AND s2.model_norm=k.model_norm
         AND s2.sbucket=k.sbucket
      ),

      -- add pooled groups (battery groups, sev_bin, condition groups) for fallbacks
      s3b AS (
        SELECT
          s3.*,
          CASE
            WHEN battery_bucket IN ('B95_100','B90_94') THEN 'B90P'
            WHEN battery_bucket IN ('B85_89','B80_84') THEN 'B80_89'
            ELSE 'B_LOW_OR_MISS'
          END AS bat_grp,
          CASE WHEN sev=0 THEN 0 ELSE 1 END AS sev_bin,
          CASE
            WHEN cs_bucket IN (0.7,0.9,1.0) THEN 'GOODP'
            WHEN cs_bucket = 0.5          THEN 'MID'
            ELSE 'POOR_OR_UNK'
          END AS cs_grp
        FROM s3
      ),

      base_b AS (
        SELECT
          k.*,
          CASE
            WHEN k.battery_bucket IN ('B95_100','B90_94') THEN 'B90P'
            WHEN k.battery_bucket IN ('B85_89','B80_84') THEN 'B80_89'
            ELSE 'B_LOW_OR_MISS'
          END AS bat_grp,
          CASE WHEN k.sev=0 THEN 0 ELSE 1 END AS sev_bin,
          CASE
            WHEN k.cs_bucket IN (0.7,0.9,1.0) THEN 'GOODP'
            WHEN k.cs_bucket = 0.5          THEN 'MID'
            ELSE 'POOR_OR_UNK'
          END AS cs_grp
        FROM base_keys k
      ),

      -- L0: full cohort (trust + battery + sev + condition)
      lvl0 AS (
        SELECT
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sold_price) AS p_med,
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_h) AS t_med,
          COUNT(*)::int AS n
        FROM s3b, base_b
        WHERE s3b.cs_bucket = base_b.cs_bucket
          AND s3b.sev       = base_b.sev
          AND s3b.battery_bucket = base_b.battery_bucket
          AND s3b.trust_tier     = base_b.trust_tier
      ),

      -- L1: drop trust (pool trust tiers)
      lvl1 AS (
        SELECT
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sold_price) AS p_med,
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_h) AS t_med,
          COUNT(*)::int AS n
        FROM s3b, base_b
        WHERE s3b.cs_bucket = base_b.cs_bucket
          AND s3b.sev       = base_b.sev
          AND s3b.battery_bucket = base_b.battery_bucket
      ),

      -- L2: merge battery (B95_100+B90_94) and (B85_89+B80_84), keep B_LOW/MISS distinct
      lvl2 AS (
        SELECT
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sold_price) AS p_med,
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_h) AS t_med,
          COUNT(*)::int AS n
        FROM s3b, base_b
        WHERE s3b.cs_bucket = base_b.cs_bucket
          AND s3b.sev       = base_b.sev
          AND s3b.bat_grp   = base_b.bat_grp
      ),

      -- L3: pool condition (GOOD+ vs MID vs POOR/UNK) and bin damage (0 vs >0)
      lvl3 AS (
        SELECT
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sold_price) AS p_med,
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_h) AS t_med,
          COUNT(*)::int AS n
        FROM s3b, base_b
        WHERE s3b.cs_grp = base_b.cs_grp
          AND s3b.sev_bin= base_b.sev_bin
      ),

      -- L4 (last resort): keep damage bin only (0 vs >0), pool all else
      lvl4 AS (
        SELECT
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sold_price) AS p_med,
          PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_h) AS t_med,
          COUNT(*)::int AS n
        FROM s3b, base_b
        WHERE s3b.sev_bin= base_b.sev_bin
      ),

      candidates AS (
        SELECT 0 AS k, 'lvl0' AS lvl, p_med, t_med, n FROM lvl0
        UNION ALL SELECT 1, 'lvl1', p_med, t_med, n FROM lvl1
        UNION ALL SELECT 2, 'lvl2', p_med, t_med, n FROM lvl2
        UNION ALL SELECT 3, 'lvl3', p_med, t_med, n FROM lvl3
        UNION ALL SELECT 4, 'lvl4', p_med, t_med, n FROM lvl4
      )
      SELECT
        p_med AS anchor_price_smart,
        t_med AS anchor_tts_median_h,
        n     AS anchor_n_support,
        k     AS anchor_level_k
      FROM (
        SELECT c.*,
               ROW_NUMBER() OVER (
                 ORDER BY
                   CASE WHEN c.n >= 200 THEN 0 ELSE 1 END,
                   CASE WHEN c.n >= 200 THEN c.k ELSE -c.k END,
                   c.n DESC
               ) AS rnk
        FROM candidates c
      ) z
      WHERE rnk = 1
    ) anc_smart ON TRUE