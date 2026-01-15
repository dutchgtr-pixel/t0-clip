-- 04_create_scoring_view.sql
-- Creates ml.woe_anchor_scores_live_v1 using active model_key + banding view + mapping table.

CREATE OR REPLACE VIEW ml.woe_anchor_scores_live_v1 AS
SELECT
  b.generation,
  b.listing_id,
  b.t0,
  b.model_key,

  (
    b.base_logit
    + COALESCE(wd.woe,0)
    + COALESCE(wt.woe,0)
    + COALESCE(ws.woe,0)
    + COALESCE(wsa.woe,0)
    + COALESCE(wc.woe,0)
    + COALESCE(wdm.woe,0)
    + COALESCE(wb.woe,0)
    + COALESCE(wp.woe,0)
    + COALESCE(wvd.woe,0)
    + COALESCE(wa.woe,0)
  )::float8 AS woe_logit,

  (1.0 / (1.0 + EXP(-(
    b.base_logit
    + COALESCE(wd.woe,0)
    + COALESCE(wt.woe,0)
    + COALESCE(ws.woe,0)
    + COALESCE(wsa.woe,0)
    + COALESCE(wc.woe,0)
    + COALESCE(wdm.woe,0)
    + COALESCE(wb.woe,0)
    + COALESCE(wp.woe,0)
    + COALESCE(wvd.woe,0)
    + COALESCE(wa.woe,0)
  ))))::float8 AS woe_anchor_p_slow21

FROM ml.v_woe_anchor_bands_live_v1 b
LEFT JOIN ml.woe_anchor_map_v1 wd
  ON wd.model_key=b.model_key AND wd.fold_id IS NULL AND wd.band_name='dsold' AND wd.band_value=b.dsold_band
LEFT JOIN ml.woe_anchor_map_v1 wt
  ON wt.model_key=b.model_key AND wt.fold_id IS NULL AND wt.band_name='trust' AND wt.band_value=b.trust_tier
LEFT JOIN ml.woe_anchor_map_v1 ws
  ON ws.model_key=b.model_key AND ws.fold_id IS NULL AND ws.band_name='ship' AND ws.band_value=b.ship_band
LEFT JOIN ml.woe_anchor_map_v1 wsa
  ON wsa.model_key=b.model_key AND wsa.fold_id IS NULL AND wsa.band_name='sale' AND wsa.band_value=b.sale_band
LEFT JOIN ml.woe_anchor_map_v1 wc
  ON wc.model_key=b.model_key AND wc.fold_id IS NULL AND wc.band_name='cond' AND wc.band_value=b.cond_band
LEFT JOIN ml.woe_anchor_map_v1 wdm
  ON wdm.model_key=b.model_key AND wdm.fold_id IS NULL AND wdm.band_name='dmg' AND wdm.band_value=b.dmg_ai_band
LEFT JOIN ml.woe_anchor_map_v1 wb
  ON wb.model_key=b.model_key AND wb.fold_id IS NULL AND wb.band_name='bat' AND wb.band_value=b.bat_band
LEFT JOIN ml.woe_anchor_map_v1 wp
  ON wp.model_key=b.model_key AND wp.fold_id IS NULL AND wp.band_name='present' AND wp.band_value=b.presentation_band
LEFT JOIN ml.woe_anchor_map_v1 wvd
  ON wvd.model_key=b.model_key AND wvd.fold_id IS NULL AND wvd.band_name='vdmg' AND wvd.band_value=b.vdmg_band
LEFT JOIN ml.woe_anchor_map_v1 wa
  ON wa.model_key=b.model_key AND wa.fold_id IS NULL AND wa.band_name='acc' AND wa.band_value=b.accessories_band;
