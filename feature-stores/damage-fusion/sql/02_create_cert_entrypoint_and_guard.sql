-- ============================================================
-- Fusion Store — Certification entrypoint + guarded train view
-- ============================================================

-- Entry point for audit.dataset_sha256() requires edited_date.
CREATE OR REPLACE VIEW ml.fusion_feature_store_t0_v1_v AS
SELECT
  l.edited_date,
  f.*
FROM "iPhone".iphone_listings l
JOIN ml.v_damage_fusion_features_v2_scored f
  ON f.generation = l.generation
 AND f.listing_id    = l.listing_id
WHERE l.edited_date IS NOT NULL
  AND l.spam IS NULL;

-- Guarded consumption view (trainer should use this surface)
CREATE OR REPLACE VIEW ml.v_damage_fusion_features_v2_scored_train_v AS
SELECT f.*
FROM ml.v_damage_fusion_features_v2_scored f
CROSS JOIN LATERAL (
  SELECT audit.require_certified_strict('ml.fusion_feature_store_t0_v1_v', interval '24 hours')
) guard;
