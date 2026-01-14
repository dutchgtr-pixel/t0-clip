# Public Release Sanitization Notes

This package is a **sanitized public-release** variant of an internal T0 feature-store module.

## What was sanitized

- The internal identifier column name `<internal_id>` was renamed to **`listing_id`** everywhere (SQL + docs).
- All references to the token `platform` were removed.

## What is intentionally not included

- No marketplace or platform names.
- No URLs, selectors, or request patterns.
- No real listing IDs or example payloads.

## Compatibility notes

- The store expects your upstream listing feature surface to provide:
  - `listing_id`
  - `generation`
  - `edited_date` (T0 timestamp)
  - `price` and relevant textual/geo columns used by the joins and affordability features.
- Reference tables (`ref.*`) are treated as **replaceable adapters**. If you do not maintain history snapshots,
  adjust the as-of join logic accordingly.

## Audit / certification framework

This public release includes a minimal audit kernel in:

- `sql/00a_minimal_audit_framework.sql`

If you already have an audit framework in your database, you can skip this file and map the store to your existing
equivalents (same function names are not required, but the semantics should match).
