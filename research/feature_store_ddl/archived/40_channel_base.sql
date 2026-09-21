-- Selected historical implementation; see provenance.json and README.md.
-- Reference only: dependencies, current-state inputs and destructive rebuilds require review.
BEGIN;

CREATE SCHEMA IF NOT EXISTS ml;

CREATE TABLE IF NOT EXISTS ml.power_seller_fingerprint_v1 (
    profile_id text PRIMARY KEY,
    profile_label text NOT NULL,
    notes text,
    location_city text NOT NULL,
    location_city_display text,
    postal_code text NOT NULL,
    seller_rating_min numeric(4,2) NOT NULL,
    seller_rating_max numeric(4,2) NOT NULL,
    review_count_seed_min integer NOT NULL,
    review_count_seed_max integer NOT NULL,
    member_since_year smallint NOT NULL,
    allow_review_count_growth boolean NOT NULL DEFAULT true,
    active boolean NOT NULL DEFAULT true,
    profile_priority integer NOT NULL DEFAULT 1000,
    source text NOT NULL DEFAULT 'operator_supplied_reference',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS ml.power_seller_tracking_snapshot_v1 (
    snapshot_id bigserial PRIMARY KEY,
    snapshot_at timestamptz NOT NULL DEFAULT now(),
    profile_id text NOT NULL REFERENCES ml.power_seller_fingerprint_v1(profile_id),
    profile_label text NOT NULL,
    total_rows integer NOT NULL,
    live_rows integer NOT NULL,
    sold_rows integer NOT NULL,
    removed_rows integer NOT NULL,
    first_listing_at timestamptz,
    latest_listing_at timestamptz,
    min_review_count integer,
    max_review_count integer,
    latest_review_count integer,
    latest_seller_rating numeric,
    latest_member_since_year smallint,
    review_count_above_seed_max boolean NOT NULL,
    sample_seller_name text
);

COMMIT;

CREATE OR REPLACE VIEW ml.power_seller_listing_match_v1_v AS
SELECT
    l.generation,
    l.listing_id,
    l.edited_date,
    l.first_seen,
    l.last_seen,
    l.sold_date,
    lower(COALESCE(l.status, '')) AS status,
    NULLIF(trim(l.seller_name), '') AS seller_name,
    l.location_city,
    l.postal_code::text AS postal_code,
    l.seller_rating,
    l.review_count,
    l.member_since_year,
    p.profile_id,
    p.profile_label,
    p.profile_priority,
    p.review_count_seed_min,
    p.review_count_seed_max,
    (l.review_count > p.review_count_seed_max) AS review_count_above_seed_max
FROM "device".device_listings l
JOIN ml.power_seller_fingerprint_v1 p
  ON p.active
 AND lower(trim(COALESCE(l.location_city, ''))) = lower(trim(p.location_city))
 AND trim(COALESCE(l.postal_code::text, '')) = trim(p.postal_code)
 AND l.seller_rating >= p.seller_rating_min
 AND l.seller_rating <= p.seller_rating_max
 AND l.review_count >= p.review_count_seed_min
 AND (
        l.review_count <= p.review_count_seed_max
        OR p.allow_review_count_growth
     )
 AND l.member_since_year = p.member_since_year;

CREATE OR REPLACE VIEW ml.power_seller_listing_match_one_v1_v AS
SELECT *
FROM (
    SELECT
        m.*,
        row_number() OVER (
            PARTITION BY m.generation, m.listing_id, m.edited_date
            ORDER BY m.profile_priority ASC, m.review_count_seed_min DESC, m.profile_id ASC
        ) AS match_rank
    FROM ml.power_seller_listing_match_v1_v m
) ranked
WHERE match_rank = 1;

CREATE OR REPLACE FUNCTION ml.refresh_power_seller_tracking_snapshot_v1()
RETURNS integer
LANGUAGE plpgsql
AS $$
DECLARE
    inserted_rows integer;
BEGIN
    INSERT INTO ml.power_seller_tracking_snapshot_v1 (
        profile_id,
        profile_label,
        total_rows,
        live_rows,
        sold_rows,
        removed_rows,
        first_listing_at,
        latest_listing_at,
        min_review_count,
        max_review_count,
        latest_review_count,
        latest_seller_rating,
        latest_member_since_year,
        review_count_above_seed_max,
        sample_seller_name
    )
    WITH latest AS (
        SELECT DISTINCT ON (m.profile_id)
            m.profile_id,
            m.review_count,
            m.seller_rating,
            m.member_since_year,
            m.seller_name
        FROM ml.power_seller_listing_match_v1_v m
        ORDER BY m.profile_id, COALESCE(m.sold_date, m.edited_date, m.last_seen, m.first_seen) DESC NULLS LAST, m.listing_id DESC
    )
    SELECT
        p.profile_id,
        p.profile_label,
        count(m.*)::int AS total_rows,
        count(*) FILTER (WHERE m.status IN ('live', 'older21days'))::int AS live_rows,
        count(*) FILTER (WHERE m.status = 'sold')::int AS sold_rows,
        count(*) FILTER (WHERE m.status = 'removed')::int AS removed_rows,
        min(COALESCE(m.first_seen, m.edited_date)) AS first_listing_at,
        max(COALESCE(m.sold_date, m.edited_date, m.last_seen, m.first_seen)) AS latest_listing_at,
        min(m.review_count)::int AS min_review_count,
        max(m.review_count)::int AS max_review_count,
        latest.review_count AS latest_review_count,
        latest.seller_rating AS latest_seller_rating,
        latest.member_since_year AS latest_member_since_year,
        bool_or(COALESCE(m.review_count_above_seed_max, false)) AS review_count_above_seed_max,
        latest.seller_name AS sample_seller_name
    FROM ml.power_seller_fingerprint_v1 p
    LEFT JOIN ml.power_seller_listing_match_v1_v m
      ON m.profile_id = p.profile_id
    LEFT JOIN latest
      ON latest.profile_id = p.profile_id
    WHERE p.active
    GROUP BY
        p.profile_id,
        p.profile_label,
        latest.review_count,
        latest.seller_rating,
        latest.member_since_year,
        latest.seller_name;

    GET DIAGNOSTICS inserted_rows = ROW_COUNT;
    RETURN inserted_rows;
END;
$$;

DROP MATERIALIZED VIEW IF EXISTS ml.channel_seller_priors_t0_v1_mv CASCADE;
DROP MATERIALIZED VIEW IF EXISTS ml.channel_listing_flags_t0_v1_mv CASCADE;

CREATE MATERIALIZED VIEW ml.channel_listing_flags_t0_v1_mv AS
SELECT
    l.generation,
    l.listing_id,
    l.edited_date,
    l.first_seen,
    l.last_seen,
    lower(COALESCE(l.status, '')) AS _status,
    l.sold_date AS _sold_date,
    CASE
        WHEN l.sold_date IS NOT NULL AND l.edited_date IS NOT NULL
        THEN EXTRACT(EPOCH FROM (l.sold_date - l.edited_date)) / 3600.0
        ELSE NULL
    END AS _duration_hours,
    COALESCE(l.source_is_bidding, false)::int AS chan__is_bidding,
    (
        l.source_bidding_evidence IS NOT NULL
        AND l.source_bidding_evidence::text NOT IN ('null', '{}', '[]')
    )::int AS chan__has_bidding_evidence,
    COALESCE(l.delivery_enabled, false)::int AS chan__delivery_enabled,
    COALESCE(l.source_is_professional, false)::int AS chan__is_professional,
    l.seller_rating::numeric AS chan__seller_rating,
    (l.seller_rating IS NULL)::int AS chan__seller_rating_missing,
    CASE WHEN l.review_count IS NULL THEN NULL ELSE ln(1.0 + GREATEST(l.review_count, 0)) END AS chan__review_count_log1p,
    (l.review_count IS NULL)::int AS chan__review_count_missing,
    l.member_since_year::numeric AS chan__member_since_year,
    CASE
        WHEN l.member_since_year IS NULL OR l.edited_date IS NULL THEN NULL
        ELSE GREATEST(EXTRACT(YEAR FROM l.edited_date)::int - l.member_since_year::int, 0)
    END::numeric AS chan__member_age_years,
    (l.member_since_year IS NULL)::int AS chan__member_since_missing,
    (pm.profile_id IS NOT NULL)::int AS chan__is_power_seller,
    COALESCE(pm.profile_priority, 0)::int AS chan__power_seller_profile_idx,
    COALESCE(pm.review_count_above_seed_max, false)::int AS chan__power_seller_review_count_above_seed_max,
    (COALESCE(l.source_is_bidding, false) AND NOT COALESCE(l.source_is_professional, false))::int AS chan__private_bidding,
    (COALESCE(l.delivery_enabled, false) AND NOT COALESCE(l.source_is_professional, false))::int AS chan__private_delivery,
    (COALESCE(l.delivery_enabled, false) AND COALESCE(l.source_is_professional, false))::int AS chan__professional_delivery
FROM "device".device_listings l
LEFT JOIN ml.power_seller_listing_match_one_v1_v pm
  ON pm.generation = l.generation
 AND pm.listing_id = l.listing_id
 AND pm.edited_date IS NOT DISTINCT FROM l.edited_date
WHERE l.edited_date IS NOT NULL;

CREATE INDEX IF NOT EXISTS channel_listing_flags_t0_v1_identity_idx
    ON ml.channel_listing_flags_t0_v1_mv (generation, listing_id, edited_date);

CREATE INDEX IF NOT EXISTS channel_listing_flags_t0_v1_prior_idx
    ON ml.channel_listing_flags_t0_v1_mv (
        generation,
        _sold_date,
        chan__is_bidding,
        chan__delivery_enabled,
        chan__is_professional,
        chan__is_power_seller
    )
    WHERE _sold_date IS NOT NULL AND _duration_hours > 0;

CREATE MATERIALIZED VIEW ml.channel_seller_priors_t0_v1_mv AS
SELECT
    b.generation,
    b.listing_id,
    b.edited_date,
    COALESCE(same_gen.chan__prior_gen_channel_n_30d, 0)::int AS chan__prior_gen_channel_n_30d,
    same_gen.chan__prior_gen_channel_fast72_rate_30d,
    same_gen.chan__prior_gen_channel_median_hours_30d,
    same_gen.chan__prior_gen_channel_avg_hours_30d,
    COALESCE(same_gen.chan__prior_gen_channel_n_90d, 0)::int AS chan__prior_gen_channel_n_90d,
    same_gen.chan__prior_gen_channel_fast72_rate_90d,
    same_gen.chan__prior_gen_channel_median_hours_90d,
    same_gen.chan__prior_gen_channel_avg_hours_90d,
    COALESCE(all_gen.chan__prior_all_channel_n_30d, 0)::int AS chan__prior_all_channel_n_30d,
    all_gen.chan__prior_all_channel_fast72_rate_30d,
    all_gen.chan__prior_all_channel_median_hours_30d,
    all_gen.chan__prior_all_channel_avg_hours_30d,
    COALESCE(all_gen.chan__prior_all_channel_n_90d, 0)::int AS chan__prior_all_channel_n_90d,
    all_gen.chan__prior_all_channel_fast72_rate_90d,
    all_gen.chan__prior_all_channel_median_hours_90d,
    all_gen.chan__prior_all_channel_avg_hours_90d,
    (COALESCE(same_gen.chan__prior_gen_channel_n_30d, 0) = 0)::int AS chan__prior_gen_channel_30d_missing,
    (COALESCE(same_gen.chan__prior_gen_channel_n_90d, 0) = 0)::int AS chan__prior_gen_channel_90d_missing,
    (COALESCE(all_gen.chan__prior_all_channel_n_30d, 0) = 0)::int AS chan__prior_all_channel_30d_missing,
    (COALESCE(all_gen.chan__prior_all_channel_n_90d, 0) = 0)::int AS chan__prior_all_channel_90d_missing
FROM ml.channel_listing_flags_t0_v1_mv b
LEFT JOIN LATERAL (
    SELECT
        count(*) FILTER (WHERE s._sold_date >= b.edited_date - interval '30 days')::int AS chan__prior_gen_channel_n_30d,
        avg((s._duration_hours <= 72)::int::numeric) FILTER (WHERE s._sold_date >= b.edited_date - interval '30 days') AS chan__prior_gen_channel_fast72_rate_30d,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY s._duration_hours) FILTER (WHERE s._sold_date >= b.edited_date - interval '30 days') AS chan__prior_gen_channel_median_hours_30d,
        avg(s._duration_hours) FILTER (WHERE s._sold_date >= b.edited_date - interval '30 days') AS chan__prior_gen_channel_avg_hours_30d,
        count(*)::int AS chan__prior_gen_channel_n_90d,
        avg((s._duration_hours <= 72)::int::numeric) AS chan__prior_gen_channel_fast72_rate_90d,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY s._duration_hours) AS chan__prior_gen_channel_median_hours_90d,
        avg(s._duration_hours) AS chan__prior_gen_channel_avg_hours_90d
    FROM ml.channel_listing_flags_t0_v1_mv s
    WHERE s.generation = b.generation
      AND s._sold_date IS NOT NULL
      AND s._duration_hours > 0
      AND s._sold_date < b.edited_date
      AND s._sold_date >= b.edited_date - interval '90 days'
      AND NOT (s.generation = b.generation AND s.listing_id = b.listing_id)
      AND s.chan__is_bidding = b.chan__is_bidding
      AND s.chan__delivery_enabled = b.chan__delivery_enabled
      AND s.chan__is_professional = b.chan__is_professional
      AND s.chan__is_power_seller = b.chan__is_power_seller
) same_gen ON true
LEFT JOIN LATERAL (
    SELECT
        count(*) FILTER (WHERE s._sold_date >= b.edited_date - interval '30 days')::int AS chan__prior_all_channel_n_30d,
        avg((s._duration_hours <= 72)::int::numeric) FILTER (WHERE s._sold_date >= b.edited_date - interval '30 days') AS chan__prior_all_channel_fast72_rate_30d,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY s._duration_hours) FILTER (WHERE s._sold_date >= b.edited_date - interval '30 days') AS chan__prior_all_channel_median_hours_30d,
        avg(s._duration_hours) FILTER (WHERE s._sold_date >= b.edited_date - interval '30 days') AS chan__prior_all_channel_avg_hours_30d,
        count(*)::int AS chan__prior_all_channel_n_90d,
        avg((s._duration_hours <= 72)::int::numeric) AS chan__prior_all_channel_fast72_rate_90d,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY s._duration_hours) AS chan__prior_all_channel_median_hours_90d,
        avg(s._duration_hours) AS chan__prior_all_channel_avg_hours_90d
    FROM ml.channel_listing_flags_t0_v1_mv s
    WHERE s._sold_date IS NOT NULL
      AND s._duration_hours > 0
      AND s._sold_date < b.edited_date
      AND s._sold_date >= b.edited_date - interval '90 days'
      AND NOT (s.generation = b.generation AND s.listing_id = b.listing_id)
      AND s.chan__is_bidding = b.chan__is_bidding
      AND s.chan__delivery_enabled = b.chan__delivery_enabled
      AND s.chan__is_professional = b.chan__is_professional
      AND s.chan__is_power_seller = b.chan__is_power_seller
) all_gen ON true;

CREATE INDEX IF NOT EXISTS channel_seller_priors_t0_v1_identity_idx
    ON ml.channel_seller_priors_t0_v1_mv (generation, listing_id, edited_date);

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
    p.chan__prior_gen_channel_median_hours_30d,
    p.chan__prior_gen_channel_avg_hours_30d,
    p.chan__prior_gen_channel_n_90d,
    p.chan__prior_gen_channel_fast72_rate_90d,
    p.chan__prior_gen_channel_median_hours_90d,
    p.chan__prior_gen_channel_avg_hours_90d,
    p.chan__prior_all_channel_n_30d,
    p.chan__prior_all_channel_fast72_rate_30d,
    p.chan__prior_all_channel_median_hours_30d,
    p.chan__prior_all_channel_avg_hours_30d,
    p.chan__prior_all_channel_n_90d,
    p.chan__prior_all_channel_fast72_rate_90d,
    p.chan__prior_all_channel_median_hours_90d,
    p.chan__prior_all_channel_avg_hours_90d,
    p.chan__prior_gen_channel_30d_missing,
    p.chan__prior_gen_channel_90d_missing,
    p.chan__prior_all_channel_30d_missing,
    p.chan__prior_all_channel_90d_missing
FROM ml.channel_listing_flags_t0_v1_mv f
LEFT JOIN ml.channel_seller_priors_t0_v1_mv p
  ON p.generation = f.generation
 AND p.listing_id = f.listing_id
 AND p.edited_date IS NOT DISTINCT FROM f.edited_date;

SELECT ml.refresh_power_seller_tracking_snapshot_v1();
