-- Selected historical implementation; see provenance.json and README.md.
-- Reference only: dependencies, current-state inputs and destructive rebuilds require review.
DROP VIEW IF EXISTS ml.channel_feature_store_t0_v1_v;
DROP MATERIALIZED VIEW IF EXISTS ml.channel_seller_priors_t0_v1_mv;
DROP MATERIALIZED VIEW IF EXISTS ml.channel_seller_priors_t0_v2_mv;
DROP MATERIALIZED VIEW IF EXISTS ml.channel_seller_prior_daily_t0_v1_mv;

CREATE MATERIALIZED VIEW ml.channel_seller_prior_daily_t0_v1_mv AS
SELECT
    f._sold_date::date AS sold_day,
    f.generation,
    f.chan__is_bidding,
    f.chan__delivery_enabled,
    f.chan__is_professional,
    f.chan__is_power_seller,
    count(*)::int AS sold_n,
    count(*) FILTER (WHERE f._duration_hours <= 72)::int AS fast72_n,
    sum(f._duration_hours)::numeric AS duration_hours_sum,
    avg(f._duration_hours)::numeric AS duration_hours_avg
FROM ml.channel_listing_flags_t0_v1_mv f
WHERE f._sold_date IS NOT NULL
  AND f._duration_hours > 0
GROUP BY
    f._sold_date::date,
    f.generation,
    f.chan__is_bidding,
    f.chan__delivery_enabled,
    f.chan__is_professional,
    f.chan__is_power_seller;

CREATE INDEX IF NOT EXISTS channel_seller_prior_daily_t0_v1_lookup_idx
    ON ml.channel_seller_prior_daily_t0_v1_mv (
        sold_day,
        generation,
        chan__is_bidding,
        chan__delivery_enabled,
        chan__is_professional,
        chan__is_power_seller
    );

CREATE INDEX IF NOT EXISTS channel_seller_prior_daily_t0_v1_channel_idx
    ON ml.channel_seller_prior_daily_t0_v1_mv (
        chan__is_bidding,
        chan__delivery_enabled,
        chan__is_professional,
        chan__is_power_seller,
        sold_day
    );

CREATE MATERIALIZED VIEW ml.channel_seller_priors_t0_v2_mv AS
SELECT
    b.generation,
    b.listing_id,
    b.edited_date,
    COALESCE(same_gen.n_30d, 0)::int AS chan__prior_gen_channel_n_30d,
    CASE WHEN COALESCE(same_gen.n_30d, 0) > 0 THEN same_gen.fast72_n_30d::numeric / same_gen.n_30d ELSE NULL END AS chan__prior_gen_channel_fast72_rate_30d,
    CASE WHEN COALESCE(same_gen.n_30d, 0) > 0 THEN same_gen.duration_sum_30d / same_gen.n_30d ELSE NULL END AS chan__prior_gen_channel_avg_hours_30d,
    COALESCE(same_gen.n_90d, 0)::int AS chan__prior_gen_channel_n_90d,
    CASE WHEN COALESCE(same_gen.n_90d, 0) > 0 THEN same_gen.fast72_n_90d::numeric / same_gen.n_90d ELSE NULL END AS chan__prior_gen_channel_fast72_rate_90d,
    CASE WHEN COALESCE(same_gen.n_90d, 0) > 0 THEN same_gen.duration_sum_90d / same_gen.n_90d ELSE NULL END AS chan__prior_gen_channel_avg_hours_90d,
    COALESCE(all_gen.n_30d, 0)::int AS chan__prior_all_channel_n_30d,
    CASE WHEN COALESCE(all_gen.n_30d, 0) > 0 THEN all_gen.fast72_n_30d::numeric / all_gen.n_30d ELSE NULL END AS chan__prior_all_channel_fast72_rate_30d,
    CASE WHEN COALESCE(all_gen.n_30d, 0) > 0 THEN all_gen.duration_sum_30d / all_gen.n_30d ELSE NULL END AS chan__prior_all_channel_avg_hours_30d,
    COALESCE(all_gen.n_90d, 0)::int AS chan__prior_all_channel_n_90d,
    CASE WHEN COALESCE(all_gen.n_90d, 0) > 0 THEN all_gen.fast72_n_90d::numeric / all_gen.n_90d ELSE NULL END AS chan__prior_all_channel_fast72_rate_90d,
    CASE WHEN COALESCE(all_gen.n_90d, 0) > 0 THEN all_gen.duration_sum_90d / all_gen.n_90d ELSE NULL END AS chan__prior_all_channel_avg_hours_90d,
    (COALESCE(same_gen.n_30d, 0) = 0)::int AS chan__prior_gen_channel_30d_missing,
    (COALESCE(same_gen.n_90d, 0) = 0)::int AS chan__prior_gen_channel_90d_missing,
    (COALESCE(all_gen.n_30d, 0) = 0)::int AS chan__prior_all_channel_30d_missing,
    (COALESCE(all_gen.n_90d, 0) = 0)::int AS chan__prior_all_channel_90d_missing
FROM ml.channel_listing_flags_t0_v1_mv b
LEFT JOIN LATERAL (
    SELECT
        sum(d.sold_n) FILTER (WHERE d.sold_day >= b.edited_date::date - 30) AS n_30d,
        sum(d.fast72_n) FILTER (WHERE d.sold_day >= b.edited_date::date - 30) AS fast72_n_30d,
        sum(d.duration_hours_sum) FILTER (WHERE d.sold_day >= b.edited_date::date - 30) AS duration_sum_30d,
        sum(d.sold_n) AS n_90d,
        sum(d.fast72_n) AS fast72_n_90d,
        sum(d.duration_hours_sum) AS duration_sum_90d
    FROM ml.channel_seller_prior_daily_t0_v1_mv d
    WHERE d.generation = b.generation
      AND d.sold_day < b.edited_date::date
      AND d.sold_day >= b.edited_date::date - 90
      AND d.chan__is_bidding = b.chan__is_bidding
      AND d.chan__delivery_enabled = b.chan__delivery_enabled
      AND d.chan__is_professional = b.chan__is_professional
      AND d.chan__is_power_seller = b.chan__is_power_seller
) same_gen ON true
LEFT JOIN LATERAL (
    SELECT
        sum(d.sold_n) FILTER (WHERE d.sold_day >= b.edited_date::date - 30) AS n_30d,
        sum(d.fast72_n) FILTER (WHERE d.sold_day >= b.edited_date::date - 30) AS fast72_n_30d,
        sum(d.duration_hours_sum) FILTER (WHERE d.sold_day >= b.edited_date::date - 30) AS duration_sum_30d,
        sum(d.sold_n) AS n_90d,
        sum(d.fast72_n) AS fast72_n_90d,
        sum(d.duration_hours_sum) AS duration_sum_90d
    FROM ml.channel_seller_prior_daily_t0_v1_mv d
    WHERE d.sold_day < b.edited_date::date
      AND d.sold_day >= b.edited_date::date - 90
      AND d.chan__is_bidding = b.chan__is_bidding
      AND d.chan__delivery_enabled = b.chan__delivery_enabled
      AND d.chan__is_professional = b.chan__is_professional
      AND d.chan__is_power_seller = b.chan__is_power_seller
) all_gen ON true;

CREATE INDEX IF NOT EXISTS channel_seller_priors_t0_v2_identity_idx
    ON ml.channel_seller_priors_t0_v2_mv (generation, listing_id, edited_date);

CREATE OR REPLACE VIEW ml.channel_feature_store_t0_v1_v AS
SELECT
    f.generation,
    f.listing_id,
    f.edited_date,
    f.chan__is_bidding,
    f.chan__has_bidding_evidence,
    f.chan__delivery_enabled,
    f.chan__is_professional,
    f.chan__seller_rating,
    f.chan__seller_rating_missing,
    f.chan__review_count_log1p,
    f.chan__review_count_missing,
    f.chan__member_since_year,
    f.chan__member_age_years,
    f.chan__member_since_missing,
    f.chan__is_power_seller,
    f.chan__power_seller_profile_idx,
    f.chan__power_seller_review_count_above_seed_max,
    f.chan__private_bidding,
    f.chan__private_delivery,
    f.chan__professional_delivery,
    p.chan__prior_gen_channel_n_30d,
    p.chan__prior_gen_channel_fast72_rate_30d,
    p.chan__prior_gen_channel_avg_hours_30d,
    p.chan__prior_gen_channel_n_90d,
    p.chan__prior_gen_channel_fast72_rate_90d,
    p.chan__prior_gen_channel_avg_hours_90d,
    p.chan__prior_all_channel_n_30d,
    p.chan__prior_all_channel_fast72_rate_30d,
    p.chan__prior_all_channel_avg_hours_30d,
    p.chan__prior_all_channel_n_90d,
    p.chan__prior_all_channel_fast72_rate_90d,
    p.chan__prior_all_channel_avg_hours_90d,
    p.chan__prior_gen_channel_30d_missing,
    p.chan__prior_gen_channel_90d_missing,
    p.chan__prior_all_channel_30d_missing,
    p.chan__prior_all_channel_90d_missing
FROM ml.channel_listing_flags_t0_v1_mv f
LEFT JOIN ml.channel_seller_priors_t0_v2_mv p
  ON p.generation = f.generation
 AND p.listing_id = f.listing_id
 AND p.edited_date IS NOT DISTINCT FROM f.edited_date;
