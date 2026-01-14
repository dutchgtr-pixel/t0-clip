-- Strict price anchor (SQL fragment; correlated LATERAL)
--
-- Public-sanitized version.
-- This file is intended to be embedded into a larger SELECT over a base table/CTE aliased as `b`.
--
-- Required base-row columns (alias `b`):
--   - b.edited_date (timestamp/date; T0 reference)
--   - b.generation
--   - b.model_norm
--   - b.sbucket
--   - b.sev
--
-- Required SOLD history table:
--   - public.listings (adapt if your schema differs)
--
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