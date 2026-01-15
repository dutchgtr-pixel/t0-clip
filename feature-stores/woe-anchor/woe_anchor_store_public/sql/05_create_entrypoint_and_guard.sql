-- 05_create_entrypoint_and_guard.sql
-- Creates certified entrypoint and fail-closed guarded consumer view.

CREATE OR REPLACE VIEW ml.woe_anchor_feature_store_t0_v1_v AS
SELECT
  s.t0 AS edited_date,
  s.generation,
  s.listing_id,
  s.model_key,
  s.woe_logit,
  s.woe_anchor_p_slow21
FROM ml.woe_anchor_scores_live_v1 s;

CREATE OR REPLACE VIEW ml.woe_anchor_scores_live_train_v AS
SELECT s.*
FROM ml.woe_anchor_scores_live_v1 s
CROSS JOIN LATERAL (
  SELECT audit.require_certified_strict('ml.woe_anchor_feature_store_t0_v1_v', interval '24 hours')
) guard;
