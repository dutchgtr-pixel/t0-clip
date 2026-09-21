-- Entirely artificial cases. No source records, paths or fitted values.
INSERT INTO ref_input.decision VALUES
('fixture-a', '2025-01-10 00:00Z', '2025-01-14 00:00Z', 'region-a', 'cohort-a');
INSERT INTO ref_input.listing_version VALUES
('fixture-a', 1, '2025-01-09 00:00Z', '2025-01-09 01:00Z', 'content-before', 100, 3),
('fixture-a', 2, '2025-01-09 00:00Z', '2025-01-11 01:00Z', 'content-late', 1, 5);
INSERT INTO ref_input.image_report_version VALUES
('fixture-a', 0, 1, '2025-01-09 00:00Z', '2025-01-09 02:00Z', 'image-a', 'encoder-a', 0.2, ARRAY[1,0,0,0], ARRAY[0,1,0,0]),
('fixture-a', 1, 1, '2025-01-09 00:00Z', '2025-01-11 02:00Z', 'image-late', 'encoder-a', 0.9, ARRAY[1,0,0,0], ARRAY[0,1,0,0]);
INSERT INTO ref_input.geo_release VALUES
('region-a', 'release-before', '2024-01-01Z', '2024-01-01Z', 10),
('region-a', 'release-late', '2024-01-01Z', '2025-01-11Z', 99);
INSERT INTO ref_input.market_observation VALUES
('history-ok', 'other-a', 'cohort-a', '2025-01-07Z', '2025-01-08Z', 80, 24),
('history-late', 'other-b', 'cohort-a', '2025-01-08Z', '2025-01-11Z', 1000, 200),
('history-self', 'fixture-a', 'cohort-a', '2025-01-07Z', '2025-01-08Z', 1000, 200);
INSERT INTO ref_input.outcome_version VALUES
('fixture-a', 1, '2025-01-13Z', '2025-01-13Z', '2025-01-12Z'),
('fixture-a', 2, '2025-01-13Z', '2025-01-15Z', '2025-01-11Z');
INSERT INTO ref_input.fitted_anchor VALUES
('anchor-before', 'cohort-a', '2025-01-01Z', '2025-01-05Z', '2025-01-06Z', 'forward_disjoint', 0.3),
('anchor-late-label', 'cohort-a', '2025-01-01Z', '2025-01-11Z', '2025-01-06Z', 'forward_disjoint', 10);

DO $$
BEGIN
    IF (SELECT ask_price FROM ref_feature.main_v) <> 100 THEN RAISE EXCEPTION 'Late listing revision leaked'; END IF;
    IF (SELECT count(*) FROM ref_feature.image_report_v) <> 8 THEN RAISE EXCEPTION 'Slot contract failed'; END IF;
    IF (SELECT sum(image_available::integer) FROM ref_feature.image_report_v) <> 1 THEN RAISE EXCEPTION 'Late image leaked'; END IF;
    IF (SELECT density FROM ref_feature.geo_v) <> 10 THEN RAISE EXCEPTION 'Late geographic release leaked'; END IF;
    IF (SELECT history_n FROM ref_feature.context_prior_v) <> 1 THEN RAISE EXCEPTION 'Historical availability or self exclusion failed'; END IF;
    IF (SELECT duration_hours FROM ref_feature.labels_v) <> 48 THEN RAISE EXCEPTION 'Late label revision leaked'; END IF;
    IF (SELECT log_odds FROM ref_feature.learned_anchor_v) <> 0.3 THEN RAISE EXCEPTION 'Future fitted labels leaked'; END IF;
END
$$;
CALL ref_audit.certify();
SELECT * FROM ref_feature.read_certified();
