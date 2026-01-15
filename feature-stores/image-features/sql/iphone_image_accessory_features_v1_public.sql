DROP VIEW IF EXISTS ml.iphone_image_accessory_features_v1;

CREATE VIEW ml.iphone_image_accessory_features_v1 AS
WITH assets AS (
  SELECT
    generation,
    listing_id,
    COUNT(*)::bigint AS n_assets
  FROM "iPhone".iphone_image_assets
  GROUP BY 1,2
),

per_image AS (
  SELECT
    generation,
    listing_id,
    image_index,

    -- processing flag (used for filtering; NOT exposed as timestamp)
    accessories_done,

    -- accessory outputs
    has_box,
    box_state_level,

    has_charger,
    has_charger_brick,
    has_cable,

    has_earbuds,

    has_case,
    case_count,

    has_receipt,
    has_other_accessory

  FROM ml.iphone_image_features_v1
  WHERE feature_version = 1
),

per_listing AS (
  SELECT
    generation,
    listing_id,

    /* ---------- coverage / completeness ---------- */
    COUNT(*)::bigint AS n_feat_rows,

    COUNT(*) FILTER (WHERE accessories_done IS TRUE)::bigint AS n_acc_done_imgs,
    MIN(image_index) FILTER (WHERE accessories_done IS TRUE) AS min_acc_done_idx,
    MAX(image_index) FILTER (WHERE accessories_done IS TRUE) AS max_acc_done_idx,

    /* ---------- global “any output observed” (catches Gen13 done-but-all-null) ---------- */
    BOOL_OR(
      has_box IS NOT NULL OR box_state_level IS NOT NULL OR
      has_charger IS NOT NULL OR has_charger_brick IS NOT NULL OR has_cable IS NOT NULL OR
      has_earbuds IS NOT NULL OR
      has_case IS NOT NULL OR case_count IS NOT NULL OR
      has_receipt IS NOT NULL OR has_other_accessory IS NOT NULL
    ) FILTER (WHERE accessories_done IS TRUE) AS any_acc_output_observed,

    COUNT(*) FILTER (
      WHERE accessories_done IS TRUE AND (
        has_box IS NOT NULL OR box_state_level IS NOT NULL OR
        has_charger IS NOT NULL OR has_charger_brick IS NOT NULL OR has_cable IS NOT NULL OR
        has_earbuds IS NOT NULL OR
        has_case IS NOT NULL OR case_count IS NOT NULL OR
        has_receipt IS NOT NULL OR has_other_accessory IS NOT NULL
      )
    )::bigint AS n_any_acc_output_observed_imgs,

    /* ---------- BOX ---------- */
    BOOL_OR(has_box) FILTER (WHERE accessories_done IS TRUE) AS has_box_any,
    BOOL_OR(has_box IS NOT NULL) FILTER (WHERE accessories_done IS TRUE) AS has_box_observed_any,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_box IS NOT NULL)::bigint AS n_box_obs_imgs,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_box IS TRUE)::bigint AS n_box_true_imgs,

    MAX(box_state_level) FILTER (WHERE accessories_done IS TRUE) AS box_state_max,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND box_state_level IS NOT NULL)::bigint AS n_box_state_obs_imgs,

    /* ---------- CHARGER / CABLE / BRICK ---------- */
    BOOL_OR(has_charger) FILTER (WHERE accessories_done IS TRUE) AS has_charger_any,
    BOOL_OR(has_charger IS NOT NULL) FILTER (WHERE accessories_done IS TRUE) AS has_charger_observed_any,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_charger IS NOT NULL)::bigint AS n_charger_obs_imgs,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_charger IS TRUE)::bigint AS n_charger_true_imgs,

    BOOL_OR(has_cable) FILTER (WHERE accessories_done IS TRUE) AS has_cable_any,
    BOOL_OR(has_cable IS NOT NULL) FILTER (WHERE accessories_done IS TRUE) AS has_cable_observed_any,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_cable IS NOT NULL)::bigint AS n_cable_obs_imgs,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_cable IS TRUE)::bigint AS n_cable_true_imgs,

    BOOL_OR(has_charger_brick) FILTER (WHERE accessories_done IS TRUE) AS has_brick_any,
    BOOL_OR(has_charger_brick IS NOT NULL) FILTER (WHERE accessories_done IS TRUE) AS has_brick_observed_any,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_charger_brick IS NOT NULL)::bigint AS n_brick_obs_imgs,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_charger_brick IS TRUE)::bigint AS n_brick_true_imgs,

    /* ---------- EARBUDS ---------- */
    BOOL_OR(has_earbuds) FILTER (WHERE accessories_done IS TRUE) AS has_earbuds_any,
    BOOL_OR(has_earbuds IS NOT NULL) FILTER (WHERE accessories_done IS TRUE) AS has_earbuds_observed_any,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_earbuds IS NOT NULL)::bigint AS n_earbuds_obs_imgs,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_earbuds IS TRUE)::bigint AS n_earbuds_true_imgs,

    /* ---------- CASE ---------- */
    BOOL_OR(has_case) FILTER (WHERE accessories_done IS TRUE) AS has_case_any,
    BOOL_OR(has_case IS NOT NULL) FILTER (WHERE accessories_done IS TRUE) AS has_case_observed_any,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_case IS NOT NULL)::bigint AS n_case_obs_imgs,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_case IS TRUE)::bigint AS n_case_true_imgs,

    MAX(case_count) FILTER (WHERE accessories_done IS TRUE AND case_count IS NOT NULL) AS case_count_max,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY case_count)
      FILTER (WHERE accessories_done IS TRUE AND case_count IS NOT NULL) AS case_count_p90,
    BOOL_OR(case_count IS NOT NULL) FILTER (WHERE accessories_done IS TRUE) AS case_count_observed_any,

    /* ---------- RECEIPT ---------- */
    BOOL_OR(has_receipt) FILTER (WHERE accessories_done IS TRUE) AS has_receipt_any,
    BOOL_OR(has_receipt IS NOT NULL) FILTER (WHERE accessories_done IS TRUE) AS has_receipt_observed_any,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_receipt IS NOT NULL)::bigint AS n_receipt_obs_imgs,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_receipt IS TRUE)::bigint AS n_receipt_true_imgs,

    /* ---------- OTHER ACCESSORY ---------- */
    BOOL_OR(has_other_accessory) FILTER (WHERE accessories_done IS TRUE) AS has_other_accessory_any,
    BOOL_OR(has_other_accessory IS NOT NULL) FILTER (WHERE accessories_done IS TRUE) AS has_other_accessory_observed_any,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_other_accessory IS NOT NULL)::bigint AS n_other_obs_imgs,
    COUNT(*) FILTER (WHERE accessories_done IS TRUE AND has_other_accessory IS TRUE)::bigint AS n_other_true_imgs

  FROM per_image
  GROUP BY 1,2
),

final AS (
  SELECT
    a.generation,
    a.listing_id,

    /* ---------- coverage metadata (no timestamps) ---------- */
    a.n_assets,
    COALESCE(p.n_feat_rows,0) AS n_feat_rows,

    COALESCE(p.n_acc_done_imgs,0) AS n_acc_done_imgs,
    p.min_acc_done_idx,
    p.max_acc_done_idx,

    (COALESCE(p.n_feat_rows,0)::numeric / NULLIF(a.n_assets,0)) AS feat_row_coverage_assets,
    (COALESCE(p.n_acc_done_imgs,0)::numeric / NULLIF(a.n_assets,0)) AS acc_done_coverage_assets,

    (a.n_assets - COALESCE(p.n_acc_done_imgs,0)) AS undone_assets,
    (COALESCE(p.n_acc_done_imgs,0) < a.n_assets) AS is_incomplete,

    (a.n_assets > 8  AND COALESCE(p.max_acc_done_idx, -1) = 7)  AS hit_cap_8,
    (a.n_assets > 16 AND COALESCE(p.max_acc_done_idx, -1) = 15) AS hit_cap_16,

    (COALESCE(p.n_acc_done_imgs,0) > 0) AS accessories_processed,

    COALESCE(p.any_acc_output_observed, FALSE) AS any_acc_output_observed,
    COALESCE(p.n_any_acc_output_observed_imgs,0) AS n_any_acc_output_observed_imgs,

    ((COALESCE(p.n_acc_done_imgs,0) > 0) AND COALESCE(p.any_acc_output_observed,FALSE) = FALSE) AS acc_done_but_all_outputs_null,

    /* ---------- BOX (canonical) ---------- */
    p.box_state_max,
    (p.box_state_max IS NOT NULL) AS box_state_known,

    CASE
      WHEN p.box_state_max IS NOT NULL THEN (p.box_state_max >= 1)
      WHEN p.has_box_observed_any IS TRUE THEN p.has_box_any
      ELSE NULL
    END AS box_present,

    (COALESCE(p.has_box_observed_any,FALSE) OR (p.box_state_max IS NOT NULL)) AS box_known,
    COALESCE(p.n_box_obs_imgs,0) AS n_box_obs_imgs,
    COALESCE(p.n_box_true_imgs,0) AS n_box_true_imgs,
    COALESCE(p.n_box_state_obs_imgs,0) AS n_box_state_obs_imgs,

    /* ---------- CHARGER (canonical across charger/cable/brick) ---------- */
    CASE
      WHEN (COALESCE(p.has_charger_observed_any,FALSE)
         OR COALESCE(p.has_cable_observed_any,FALSE)
         OR COALESCE(p.has_brick_observed_any,FALSE))
      THEN (COALESCE(p.has_charger_any,FALSE)
         OR COALESCE(p.has_cable_any,FALSE)
         OR COALESCE(p.has_brick_any,FALSE))
      ELSE NULL
    END AS charger_present,

    (COALESCE(p.has_charger_observed_any,FALSE)
      OR COALESCE(p.has_cable_observed_any,FALSE)
      OR COALESCE(p.has_brick_observed_any,FALSE)) AS charger_known,

    CASE
      WHEN (COALESCE(p.has_charger_observed_any,FALSE)
         OR COALESCE(p.has_cable_observed_any,FALSE)
         OR COALESCE(p.has_brick_observed_any,FALSE)) IS NOT TRUE
      THEN NULL
      WHEN (COALESCE(p.has_charger_any,FALSE)
         OR COALESCE(p.has_cable_any,FALSE)
         OR COALESCE(p.has_brick_any,FALSE)) IS FALSE
      THEN 0
      WHEN COALESCE(p.has_brick_any,FALSE) AND COALESCE(p.has_cable_any,FALSE)
      THEN 2
      ELSE 1
    END AS charger_bundle_level,  -- 0 none, 1 partial (cable or brick or generic charger), 2 brick+cable

    /* expose the components too (with knownness via *_observed_any + evidence) */
    p.has_cable_any AS cable_present,
    p.has_cable_observed_any AS cable_known,
    COALESCE(p.n_cable_obs_imgs,0) AS n_cable_obs_imgs,
    COALESCE(p.n_cable_true_imgs,0) AS n_cable_true_imgs,

    p.has_brick_any AS brick_present,
    p.has_brick_observed_any AS brick_known,
    COALESCE(p.n_brick_obs_imgs,0) AS n_brick_obs_imgs,
    COALESCE(p.n_brick_true_imgs,0) AS n_brick_true_imgs,

    /* ---------- EARBUDS ---------- */
    CASE WHEN p.has_earbuds_observed_any IS TRUE THEN p.has_earbuds_any ELSE NULL END AS earbuds_present,
    COALESCE(p.has_earbuds_observed_any,FALSE) AS earbuds_known,
    COALESCE(p.n_earbuds_obs_imgs,0) AS n_earbuds_obs_imgs,
    COALESCE(p.n_earbuds_true_imgs,0) AS n_earbuds_true_imgs,

    /* ---------- CASE ---------- */
    CASE WHEN p.has_case_observed_any IS TRUE THEN p.has_case_any ELSE NULL END AS case_present,
    COALESCE(p.has_case_observed_any,FALSE) AS case_known,
    COALESCE(p.n_case_obs_imgs,0) AS n_case_obs_imgs,
    COALESCE(p.n_case_true_imgs,0) AS n_case_true_imgs,

    p.case_count_max,
    p.case_count_p90,
    COALESCE(p.case_count_observed_any,FALSE) AS case_count_known,

    /* ---------- RECEIPT ---------- */
    CASE WHEN p.has_receipt_observed_any IS TRUE THEN p.has_receipt_any ELSE NULL END AS receipt_present,
    COALESCE(p.has_receipt_observed_any,FALSE) AS receipt_known,
    COALESCE(p.n_receipt_obs_imgs,0) AS n_receipt_obs_imgs,
    COALESCE(p.n_receipt_true_imgs,0) AS n_receipt_true_imgs,

    /* ---------- OTHER ---------- */
    CASE WHEN p.has_other_accessory_observed_any IS TRUE THEN p.has_other_accessory_any ELSE NULL END AS other_accessory_present,
    COALESCE(p.has_other_accessory_observed_any,FALSE) AS other_accessory_known,
    COALESCE(p.n_other_obs_imgs,0) AS n_other_obs_imgs,
    COALESCE(p.n_other_true_imgs,0) AS n_other_true_imgs

  FROM assets a
  LEFT JOIN per_listing p
    ON p.generation=a.generation AND p.listing_id=a.listing_id
)

SELECT * FROM final;
