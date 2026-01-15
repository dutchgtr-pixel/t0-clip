# Public Release Sanitization Report — Drift‑Tolerant Certification Package

## Scope
Sanitized the drift‑tolerant T0 certification package (SQL + docs) for safe public release.

## Changes applied
### 1) Removed platform‑specific branding from documentation
- Updated the top‑level README to use neutral, platform‑agnostic language.

### 2) Generalized platform‑specific identifier naming
- Renamed the listing identifier column used in certification checks to `listing_id`.

## Credential hygiene
- Confirmed there are no embedded authentication artifacts (no credential‑bearing DSN literals, no API keys, no access tokens).

## Final scan results
- Confirmed zero occurrences of prohibited platform brand/domain strings.
- Confirmed zero occurrences of credential‑bearing DSN literals.

## Files modified
- README.md
- sql/03_device_meta_drift_allowed_cert.sql

Generated on: 2026-01-14
