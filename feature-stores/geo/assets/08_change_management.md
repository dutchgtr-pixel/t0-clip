# Change management (how to evolve v4 safely)

This system is designed for frequent iteration without breaking production.

## What constitutes a “mapping change”

Any change to:
- postal_code → super_metro_v4 assignment
- city fallback → super_metro_v4 assignment
- region or pickup_metro labels

…should be shipped as a **new release** (new `release_id`).

Never edit mapping rows for an existing release in place, because:
- it breaks reproducibility
- it makes rollbacks impossible
- it complicates model debugging

## Release process (recommended)

1) Prepare new CSVs
- `postal_code_to_super_metro_v4.csv`
- `city_postal_codes_with_region_super_metro_v4.csv`

2) Run loader in dry run
- ensure row counts match expectation
- ensure no “bad postcodes” exist

3) Load as a new release (not current yet)
- run loader without `--set_current`

4) QA the release explicitly
- distribution checks
- unknown coverage check
- join coverage on recent listings

5) Flip pointer to make it current (atomic transaction)
- do it during a low-traffic period if needed

6) Refresh feature store MVs
- ensures model features now reference the new mapping

## Rollback

Rollback is always possible as long as you never delete mapping releases:

```sql
BEGIN;
UPDATE ref.geo_mapping_release SET is_current=false WHERE is_current=true;
UPDATE ref.geo_mapping_release SET is_current=true WHERE release_id = <previous_release_id>;
COMMIT;
```

Then:
- re-run your validation checks
- refresh the feature store if you need snapshot views to reflect the rollback

## Release metadata best practices

Use `label` and `notes` to encode:

- mapping version tag (super_metro_v4)
- source data (e.g., “derived from postal-code mapping v4 file on YYYY-MM-DD”)
- major change summary (e.g., “split Hallingdal vs Oslo inland; moved Hol to inland cluster”)

## Model governance

For every model artifact, store:

- training window
- feature store version (git hash)
- `geo_release_id` used during training
- dataset snapshot ids / checksums

This is what makes geo changes explainable rather than “mysterious drift”.

