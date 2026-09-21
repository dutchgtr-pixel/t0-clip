CREATE VIEW ref_feature.context_prior_v AS
SELECT d.entity_id, h.history_n, h.fast72_rate, h.anchor_price,
       h.latest_history_available_at
FROM ref_input.decision d
LEFT JOIN LATERAL (
    SELECT count(*)::integer AS history_n,
           avg((h.duration_hours <= 72)::integer) AS fast72_rate,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY h.ask_price) AS anchor_price,
           max(h.available_at) AS latest_history_available_at
    FROM ref_input.market_observation h
    WHERE h.cohort_key = d.cohort_key AND h.source_entity_id <> d.entity_id
      AND h.event_at < d.t0 AND h.available_at < d.t0
      AND h.event_at >= d.t0 - interval '90 days'
) h ON true;

CREATE VIEW ref_feature.learned_anchor_v AS
SELECT d.entity_id, a.artifact_id, a.log_odds
FROM ref_input.decision d
LEFT JOIN LATERAL (
    SELECT a.* FROM ref_input.fitted_anchor a
    WHERE a.cohort_key = d.cohort_key AND a.fitted_through < d.t0
      AND a.label_available_through < d.t0 AND a.published_at <= d.t0
      AND a.fold_policy = 'forward_disjoint'
    ORDER BY a.published_at DESC, a.artifact_id DESC LIMIT 1
) a ON true;
-- An out-of-fold token is not enough: adapters must prove entity-disjoint fit
-- membership. This portable forward-only query intentionally rejects that mode.

CREATE VIEW ref_feature.assembled_v AS
SELECT m.entity_id, m.t0, m.content_digest, m.ask_price, m.condition_score,
       g.release_id AS geo_release_id, g.density,
       h.history_n, h.fast72_rate, h.anchor_price, a.artifact_id, a.log_odds,
       CASE WHEN h.anchor_price > 0 THEN m.ask_price / h.anchor_price ELSE NULL END AS price_to_anchor,
       i.image_count, i.report_count, i.max_damage,
       (m.content_digest IS NOT NULL) AS main_ready
FROM ref_feature.main_v m
JOIN ref_feature.geo_v g USING (entity_id)
JOIN ref_feature.context_prior_v h USING (entity_id)
JOIN ref_feature.learned_anchor_v a USING (entity_id)
LEFT JOIN LATERAL (
    SELECT count(*) FILTER (WHERE image_available)::integer AS image_count,
           count(*) FILTER (WHERE report_available)::integer AS report_count,
           max(damage_score) AS max_damage
    FROM ref_feature.image_report_v i WHERE i.entity_id = m.entity_id
) i ON true;
