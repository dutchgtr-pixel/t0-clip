# Proof Queries (Copy/Paste)

## A) No duplicate keys
```sql
SELECT listing_id, COUNT(*) AS n
FROM ml.geo_dim_super_metro_v4_t0_v1
GROUP BY listing_id
HAVING COUNT(*) > 1
LIMIT 20;
```
Expected: 0 rows

## B) Token scan (no time-of-query tokens in closure)
```sql
WITH RECURSIVE rels(oid) AS (
  SELECT 'ml.geo_feature_store_t0_v1_v'::regclass
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
```
Expected: 0 rows

## C) Baseline counts
```sql
SELECT
  COUNT(*) AS baseline_days,
  MIN(t0_day) AS oldest_day,
  MAX(t0_day) AS newest_day
FROM audit.t0_dataset_hash_baseline
WHERE entrypoint='ml.geo_feature_store_t0_v1_v'
  AND sample_limit=2000;
```
Expected: 365 baseline days (or whatever policy you chose).

## D) Registry status
```sql
SELECT entrypoint, status, certified_at, dataset_days, notes
FROM audit.t0_cert_registry
WHERE entrypoint='ml.geo_feature_store_t0_v1_v';
```

## E) Guarded view works
```sql
SELECT * FROM ml.geo_dim_super_metro_v4_t0_train_v LIMIT 10;
```
