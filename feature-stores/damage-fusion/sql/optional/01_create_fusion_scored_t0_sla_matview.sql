-- ============================================================
-- OPTIONAL: Strict T0/SLA-gated fusion scored materialization
--
-- This creates: ml.v_damage_fusion_features_v2_scored_t0_v1_mv
-- It gates every fusion column behind img_within_sla so evidence is only
-- present when the image pipeline completed within SLA relative to edited_date.
--
-- Prerequisites (from your remediation framework):
--   - ml.iphone_image_features_unified_t0_v1_mv
--   - ml.tom_features_v1_enriched_ai_clean_t0_v1_mv
--   - ml.v_damage_fusion_features_v2_scored
-- ============================================================

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
