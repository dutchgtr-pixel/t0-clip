-- 06_switch_current_release.sql
-- Switch the active mapping release (atomic flip).
-- Replace :target_release_id with the release id you want to activate.

BEGIN;

UPDATE ref.geo_mapping_release
SET is_current = false
WHERE is_current = true;

UPDATE ref.geo_mapping_release
SET is_current = true
WHERE release_id = :target_release_id;

COMMIT;
