-- 03_create_normalization_functions.sql
-- Canonical normalization helpers for join keys.

CREATE OR REPLACE FUNCTION ref.norm_city(x text)
RETURNS text
LANGUAGE sql
IMMUTABLE
AS $$
  SELECT NULLIF(lower(regexp_replace(btrim(x), '\s+', ' ', 'g')), '');
$$;

CREATE OR REPLACE FUNCTION ref.norm_postal_code(x text)
RETURNS text
LANGUAGE sql
IMMUTABLE
AS $$
  WITH cleaned AS (
    SELECT regexp_replace(btrim(coalesce(x,'')), '\D', '', 'g') AS pc
  )
  SELECT CASE WHEN pc ~ '^[0-9]{4}$' THEN pc ELSE NULL END
  FROM cleaned;
$$;
