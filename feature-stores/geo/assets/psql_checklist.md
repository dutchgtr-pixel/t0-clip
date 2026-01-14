# Example: end-to-end mapping setup (psql)

This is a copy/paste checklist.

## 1) Connect to Postgres inside Docker

```powershell
docker exec -it postgres psql -U postgres -d scrapes
```

## 2) Apply SQL scripts

Inside psql:

```sql
\i sql/01_create_versioned_reference_tables.sql
\i sql/02_create_current_views.sql
\i sql/03_create_normalization_functions.sql
\i sql/04_create_geo_enriched_listings_view.sql
```

## 3) Load a mapping release from Windows host (Python)

In PowerShell (outside the container):

```powershell
$env:PG_DSN = "<REDACTED_PG_DSN>"

python .\load_geo_mapping_release.py `
  --label "super_metro_v4" `
  --notes "initial v4 mapping" `
  --postal_csv ".\postal_code_to_super_metro_v4.csv" `
  --city_csv ".\city_postal_codes_with_region_super_metro_v4.csv" `
  --set_current
```

## 4) Validate in psql

```sql
\i sql/05_post_load_qa_checks.sql
```

## 5) Spot check

```sql
SELECT listing_id, postal_code, location_city, super_metro_v4_geo
FROM ml.listings_geo_current
ORDER BY listing_id DESC
LIMIT 25;
```
