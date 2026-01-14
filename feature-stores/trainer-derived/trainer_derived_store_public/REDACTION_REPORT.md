# Redaction report — Trainer Derived Feature Store (Public Release)

## Objective
Sanitize the trainer-derived feature store SQL package for public release by removing platform-identifying strings and ensuring the package is generic and reusable.

## Replacements performed

### Identifier normalization
- Replaced a platform-specific primary key column name with the neutral key: `listing_id` (across SQL and documentation)

### Input surface name generalization
- Replaced internal anchor surface naming with: `ml.anchor_features_v1_mv`
- Replaced internal base feature surface naming with: `ml.market_feature_store_train_v`
- Replaced internal vision/image surface naming with: `ml.image_features_unified_v1_train_v`

### Example data sanitization
- Replaced any real-looking example identifiers with synthetic values (e.g., `listing_id=123456789`)
- Replaced concrete example timestamps with generic placeholders (e.g., `2026-01-01 00:00:00+00`)

## Forbidden-string scan
Confirmed the public package contains **zero occurrences** of platform brand/domain identifiers and internal system-specific naming.

## Notes
- No credentials or secrets were present in the source SQL files; none were added.
- The certification framework shipped here (`sql/00_audit_primitives_public.sql`) is a minimal public template meant to demonstrate the pattern. Production systems may use richer registry, alerting, and CI enforcement.

