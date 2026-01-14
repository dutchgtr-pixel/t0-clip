-- 10_pin_release.sql
-- Pin the current geo mapping release into a stable 1-row view (no DROP to avoid dependency cascades).

DO $$
DECLARE rid bigint;
BEGIN
  SELECT release_id INTO rid FROM ref.geo_mapping_current;
  IF rid IS NULL THEN
    RAISE EXCEPTION 'No current geo release_id (ref.geo_mapping_current is empty)';
  END IF;

  RAISE NOTICE 'Pinning geo mapping release_id = %', rid;

  EXECUTE format(
    'CREATE OR REPLACE VIEW ref.geo_mapping_pinned_super_metro_v4_v1 AS SELECT %s::bigint AS release_id',
    rid
  );
END $$;
