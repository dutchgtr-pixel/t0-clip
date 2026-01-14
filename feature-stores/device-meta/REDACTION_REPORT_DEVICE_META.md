# Redaction Report: Device Meta Feature Store (Public Release)

## Summary

The uploaded documentation and SQL were sanitized for safe public release while preserving the functional structure:

- No secrets or credentials are present.
- The listing identifier field is standardized to `listing_id`.
- Marketplace branding terms and platform-specific identifiers are removed.
- The SQL remains internally consistent and runnable as a generic template (assuming upstream source relations exist).

## Files produced

- `device_meta_feature_store_public.sql`
- `DEVICE_META_FEATURE_STORE_PUBLIC.md`

## Replacements performed

1) Identifier standardization
- Replaced platform-specific listing identifier references with `listing_id`.

2) Source relation genericization
- Replaced internal, project-specific source relation names with generic placeholders:
  - `feature_store.listing_device_features_clean_mv`
  - `feature_store.device_image_features_unified_v1`

3) Public-facing object naming
- The final encoded view name was genericized to `feature_store.device_meta_encoded_t0_v1`.

## New placeholders introduced

- Upstream source relations are treated as prerequisites and must be provided by the adopter:
  - `feature_store.listing_device_features_clean_mv`
  - `feature_store.device_image_features_unified_v1`

## Scan results (released artifacts)

- Forbidden marketplace brand terms: 0 occurrences.
- Credential-like patterns (DSNs with embedded passwords, tokens, cookies, bearer headers): 0 occurrences.

