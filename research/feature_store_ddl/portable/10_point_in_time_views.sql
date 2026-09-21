CREATE VIEW ref_feature.main_v AS
SELECT d.entity_id, d.t0, d.region_key, d.cohort_key,
       v.revision, v.content_digest, v.ask_price, v.condition_score,
       v.available_at AS main_available_at
FROM ref_input.decision d
LEFT JOIN LATERAL (
    SELECT v.* FROM ref_input.listing_version v
    WHERE v.entity_id = d.entity_id AND v.observed_at <= d.t0 AND v.available_at <= d.t0
    ORDER BY v.available_at DESC, v.revision DESC LIMIT 1
) v ON true;

CREATE VIEW ref_feature.image_report_v AS
SELECT d.entity_id, d.t0, s.slot, v.source_digest, v.encoder_digest,
       v.damage_score, v.image_vector, v.report_vector,
       (v.image_vector IS NOT NULL) AS image_available,
       (v.report_vector IS NOT NULL) AS report_available,
       v.available_at AS report_available_at
FROM ref_input.decision d CROSS JOIN generate_series(0, 7) s(slot)
LEFT JOIN LATERAL (
    SELECT v.* FROM ref_input.image_report_version v
    WHERE v.entity_id = d.entity_id AND v.slot = s.slot
      AND v.evidence_at <= d.t0 AND v.available_at <= d.t0
    ORDER BY v.available_at DESC, v.revision DESC LIMIT 1
) v ON true;

CREATE VIEW ref_feature.geo_v AS
SELECT d.entity_id, g.release_id, g.density, g.published_at
FROM ref_input.decision d
LEFT JOIN LATERAL (
    SELECT g.* FROM ref_input.geo_release g
    WHERE g.region_key = d.region_key AND g.valid_from <= d.t0 AND g.published_at <= d.t0
    ORDER BY g.published_at DESC, g.release_id DESC LIMIT 1
) g ON true;

CREATE VIEW ref_feature.labels_v AS
SELECT d.entity_id, d.t0, d.label_cutoff,
       CASE WHEN o.observed_until IS NULL THEN NULL ELSE
       GREATEST(EXTRACT(EPOCH FROM (LEAST(COALESCE(o.event_at, o.observed_until), d.label_cutoff) - d.t0)) / 3600, 0) END AS duration_hours,
       (o.observed_until IS NOT NULL) AS label_available,
       (o.event_at IS NOT NULL AND o.event_at <= d.label_cutoff AND o.event_at >= d.t0) AS event,
       (o.event_at = d.t0) AS zero_duration_event,
       (o.event_at < d.t0 OR o.observed_until < d.t0) AS invalid_origin
FROM ref_input.decision d
LEFT JOIN LATERAL (
    SELECT o.* FROM ref_input.outcome_version o
    WHERE o.entity_id = d.entity_id AND o.available_at <= d.label_cutoff
    ORDER BY o.available_at DESC, o.revision DESC LIMIT 1
) o ON true;
-- Labels are a separate surface: no outcome field enters feature assembly.
