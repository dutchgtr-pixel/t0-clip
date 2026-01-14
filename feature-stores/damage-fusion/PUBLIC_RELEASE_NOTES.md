# Public release notes

This package contains the SQL and documentation for a **damage fusion** feature-store layer implemented as:

- `ml.v_damage_fusion_features_v2`
- `ml.v_damage_fusion_features_v2_scored`

## Identifier naming

The public release uses **`listing_id`** as the canonical listing identifier column name in all SQL and documentation.

If your internal schema uses a different identifier name, adapt the upstream table/view definitions accordingly (keep the fusion view outputs keyed on `(generation, listing_id)`).

## Security and privacy

- No credentials, secrets, or connection strings are included.
- Example identifiers in docs are illustrative/synthetic.

