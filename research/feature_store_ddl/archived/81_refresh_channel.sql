-- Selected historical implementation; see provenance.json and README.md.
-- Reference only: dependencies, current-state inputs and destructive rebuilds require review.
\timing on

REFRESH MATERIALIZED VIEW ml.channel_listing_flags_t0_v1_mv;
REFRESH MATERIALIZED VIEW ml.channel_seller_prior_daily_t0_v1_mv;
REFRESH MATERIALIZED VIEW ml.channel_seller_priors_t0_v2_mv;

SELECT ml.refresh_power_seller_tracking_snapshot_v1() AS tracking_profiles_snapshotted;

SELECT
    count(*) AS channel_rows,
    sum(chan__is_bidding) AS bidding_rows,
    sum(chan__delivery_enabled) AS fiks_rows,
    sum(chan__is_professional) AS professional_rows,
    sum(chan__is_power_seller) AS power_seller_rows,
    count(*) FILTER (WHERE chan__prior_all_channel_n_30d > 0) AS rows_with_30d_prior,
    count(*) FILTER (WHERE chan__prior_all_channel_n_90d > 0) AS rows_with_90d_prior
FROM ml.channel_feature_store_t0_v1_v;
