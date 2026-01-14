# Geo Feature Store Public Release Sanitization Report

Generated: 2026-01-14 22:51:36Z (UTC)

## Summary of changes

This package was sanitized for public release to remove any platform-identifying terminology and any credential/login material, while preserving functional structure.

- Standardized the primary listing identifier field name to: `listing_id`
- Generalized internal example schema/table naming used in assets and documentation:
  - Raw listings table: `marketplace.listings_raw`
  - Geo-enriched listings view: `ml.listings_geo_current`
- Removed credential-like database connection strings from docs and scripts (replaced with `<REDACTED_PG_DSN>`).

## Files changed

- assets/00_all_geo_mapping_objects.sql
- assets/04_create_geo_enriched_listings_view.sql
- assets/05_post_load_qa_checks.sql
- assets/07_optional_indexes.sql
- assets/Documentation end to   end.txt
- assets/load_geo_mapping_release.py
- assets/psql_checklist.md
- docs/02_objects_and_contracts.md
- docs/03_t0_and_leakproof.md
- docs/04_certification.md
- proofs/proof_queries.md
- sql/11_create_pinned_geo_dim.sql
- sql/12_create_entrypoint_and_guard.sql
- sql/15_create_cert_assert_and_runner.sql

## Final scan

Verified across all files in this public-release package:

- No target platform name/domain terms remain.
- No database login strings, passwords, tokens, cookies, or API keys remain.
