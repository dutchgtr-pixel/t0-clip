-- Anchor priors verification helpers

-- 1) Strict anchor embargo proof (spot-check):
-- For a chosen listing_id, verify strict-anchor sold rows are strictly before t0-5d.
-- (Adjust listing_id and ensure you use the exact sold filters used in the trainer.)

-- 2) Advanced anchor support distribution:
-- In Python, after training dataset assembly, check:
--   df['anchor_level_k'].value_counts(dropna=False)
--   df['anchor_n_support'].describe()

-- 3) Speed anchor token scan (should not include CURRENT_DATE / now() in the certified closure):
-- Replace entrypoint with your certified features_view entrypoint if needed.
WITH RECURSIVE rels(oid) AS (
  SELECT 'ml.feature_store_t0_entrypoint_v'::regclass
  UNION
  SELECT c2.oid
  FROM rels r
  JOIN pg_rewrite w ON w.ev_class = r.oid
  JOIN pg_depend  d ON d.objid = w.oid
  JOIN pg_class   c2 ON c2.oid = d.refobjid
  WHERE c2.relkind IN ('v','m')
)
SELECT n.nspname||'.'||c.relname AS obj, c.relkind
FROM rels
JOIN pg_class c ON c.oid=rels.oid
JOIN pg_namespace n ON n.oid=c.relnamespace
WHERE lower(pg_get_viewdef(c.oid,true)) ~
  '\m(current_date|current_timestamp|now\(|clock_timestamp\(|transaction_timestamp\(|statement_timestamp\(|localtimestamp)\M'
ORDER BY 1;

