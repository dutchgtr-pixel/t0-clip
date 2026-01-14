-- Public Release: Refresh/runbook SQL snippets (optional)

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


CALL audit.refresh_and_certify_survival_v1(true);


REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_speed_anchor_asof_v1_mv;
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_speed_t0_v1_mv;
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.tom_features_v1_enriched_ai_clean_t0_v1_mv;
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.iphone_image_features_unified_t0_v1_mv;
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.v_damage_fusion_features_v2_scored_t0_v1_mv;
REFRESH MATERIALIZED VIEW CONCURRENTLY ml.iphone_device_meta_encoded_t0_v1_mv;

CALL audit.run_t0_cert_survival_v1();
SELECT audit.require_certified_strict('ml.survival_feature_store_t0_v1_v', interval '24 hours');
