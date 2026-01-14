-- 02_create_current_views.sql
-- Views that expose only the mapping rows for the current release.

CREATE OR REPLACE VIEW ref.geo_mapping_current AS
SELECT release_id
FROM ref.geo_mapping_release
WHERE is_current
ORDER BY release_id DESC
LIMIT 1;

CREATE OR REPLACE VIEW ref.postal_code_to_super_metro_current AS
SELECT m.*
FROM ref.postal_code_to_super_metro m
JOIN ref.geo_mapping_current c ON c.release_id = m.release_id;

CREATE OR REPLACE VIEW ref.city_to_super_metro_current AS
SELECT m.*
FROM ref.city_to_super_metro m
JOIN ref.geo_mapping_current c ON c.release_id = m.release_id;
