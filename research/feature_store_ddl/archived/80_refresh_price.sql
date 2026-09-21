-- Selected historical implementation; see provenance.json and README.md.
-- Reference only: dependencies, current-state inputs and destructive rebuilds require review.
\timing on

SET statement_timeout = 0;
SET work_mem = '512MB';
SET max_parallel_workers_per_gather = 0;

REFRESH MATERIALIZED VIEW ml.price_anchor_history_t0_v1_mv;
REFRESH MATERIALIZED VIEW ml.price_anchor_prior_daily_t0_v1_mv;
REFRESH MATERIALIZED VIEW ml.price_anchor_target_keys_t0_v1_mv;
REFRESH MATERIALIZED VIEW ml.price_anchor_key_metrics_t0_v1_mv;
REFRESH MATERIALIZED VIEW ml.price_anchor_feature_store_t0_v1_mv;
ANALYZE ml.price_anchor_history_t0_v1_mv;
ANALYZE ml.price_anchor_prior_daily_t0_v1_mv;
ANALYZE ml.price_anchor_target_keys_t0_v1_mv;
ANALYZE ml.price_anchor_key_metrics_t0_v1_mv;
ANALYZE ml.price_anchor_feature_store_t0_v1_mv;

SELECT
    count(*) AS price_anchor_rows,
    count(*) FILTER (WHERE pa__anchor_n_30d > 0) AS rows_with_30d_anchor,
    count(*) FILTER (WHERE pa__anchor_n_90d > 0) AS rows_with_90d_anchor,
    round(100.0 * count(*) FILTER (WHERE pa__anchor_n_90d > 0) / NULLIF(count(*), 0), 2) AS pct_with_90d_anchor,
    count(*) FILTER (WHERE pa__price_to_anchor_90d IS NOT NULL) AS rows_with_price_to_anchor_90d
FROM ml.price_anchor_feature_store_t0_v1_mv;
