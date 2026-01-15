# Public Release Redaction Report — Vision Image Feature Store
Generated: 2026-01-14T23:22:28Z
## Summary
This package contains sanitized SQL and documentation for a vision-oriented image feature store.
Redactions were performed to remove platform-identifying terms and normalize identifier naming.

## Key replacements
- Replaced the platform-specific listing identifier column name with `listing_id` across all SQL and docs.
- Removed platform brand references (replaced with the generic term `marketplace`).
- Renamed example index identifiers that previously included platform-specific substrings.
- Verified no secrets (API keys, passwords, DSNs) are embedded.

## Files processed
- `docs/Documentation_image_store_full_public.md` (from `Documentation____ image store full.txt`)
- `docs/Documentation_iphone_image_features_unified_v1_public.md` (from `Documentation____ iphone_image_features_unified_v1 — Unified Vision Feature Store.txt`)
- `docs/Documentation_vision_color_feature_store_public.md` (from `Documentation____ Vision Color Feature Store.txt`)
- `docs/Documentation_iphone_image_damage_features_v1_public.md` (from `Documentation____ml.iphone_image_damage_features_v1.txt`)
- `docs/Documentation_vision_accessory_feature_store_public.md` (from `Documentation____Vision Accessory Feature Store.txt`)
- `sql/iphone_image_damage_features_v1_public.sql` (from `FEATURE_STORE_IMAGE_DAMAGE_SQL.txt`)
- `sql/iphone_image_accessory_features_v1_public.sql` (from `SQ Drop and create ml iphone_image_accessory_features_v1.txt`)
- `sql/iphone_image_color_features_v1_public.sql` (from `SQL ml.iphone_image_color_features_v1.txt`)
- `sql/iphone_image_features_unified_v1_public.sql` (from `SQL ml.iphone_image_features_unified_v1.txt`)

## Scan results
- Forbidden platform strings: all zero occurrences.
  - platform string set: [platform brand terms + domain terms + parent company terms]
- Credential-like patterns: no matches for common API-key or credentialed-DSN formats.
