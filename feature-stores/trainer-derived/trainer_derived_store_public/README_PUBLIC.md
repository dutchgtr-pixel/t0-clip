# Trainer Derived Feature Store — Public Release Package

This package provides a **platform-agnostic, T0-safe (“no-leak”)** derived feature store implemented entirely in **PostgreSQL**, plus a **certification and fail-closed guard** pattern.

It is designed for ML training and inference pipelines that require:

- deterministic features computed at an event time `t0`
- strict reproducibility across time (definition hashes + dataset hashes)
- operational controls that **block consumption** if drift is detected

The store is intentionally generic: it uses `listing_id` as the entity key and does not reference any specific marketplace, crawler, or proprietary source system.

---

## What’s included

### SQL modules

| File | Purpose |
|---|---|
| `sql/00_audit_primitives_public.sql` | Minimal certification framework: viewdef baselines, dataset hash baselines, certification registry, strict guard |
| `sql/01_example_input_surfaces_public.sql` | **Optional** synthetic inputs to run the package end-to-end |
| `sql/02_create_trainer_derived_store_public.sql` | Creates the derived store view + materialized view + certification entrypoint + guarded consumer |
| `sql/03_certify_trainer_derived_store_public.sql` | Captures baselines, rebaselines recent days, certifies, and enforces strict guard |
| `sql/05_feature_block_normalization_public.sql` | Optional metadata normalization for model feature importance reporting |
| `sql/07_validation_queries_public.sql` | Coverage checks, point-lookup plan check, and registry checks |

### Documentation

- `docs/00_overview.md` — what the store is and why it exists
- `docs/01_feature_spec.md` — feature-by-feature specification
- `docs/02_drift_policy.md` — drift-tolerant certification policy (last-N days)
- `docs/04_change_management.md` — how to evolve the store safely
- `docs/05_runbook.md` — operational workflow (daily/weekly)

---

## Key database objects (canonical names)

### Store objects (created by `sql/02_create_trainer_derived_store_public.sql`)
- Logical view: `ml.trainer_derived_features_v1`
- Materialized view: `ml.trainer_derived_features_v1_mv`
- Certification entrypoint: `ml.trainer_derived_feature_store_t0_v1_v`
- Guarded consumer view (fail-closed): `ml.trainer_derived_features_train_v`

### Audit / certification objects (created by `sql/00_audit_primitives_public.sql`)
- `audit.t0_viewdef_baseline`
- `audit.t0_dataset_hash_baseline`
- `audit.t0_cert_registry`
- `audit.capture_viewdef_baseline(p_entrypoint)`
- `audit.rebaseline_last_n_days(p_entrypoint, p_n, p_sample_limit)`
- `audit.run_t0_cert_trainer_derived_store_v1(p_check_days)`
- `audit.require_certified_strict(p_entrypoint, p_max_age)`

---

## Contract surface (training consumption)

`ml.trainer_derived_features_train_v` returns **one row per** `(generation, listing_id, t0)` and includes:

- calendar flags (`dow`, `is_weekend`)
- strict T0 trailing activity counts (`gen_30d_post_count`, `allgen_30d_post_count`)
- cross-modal consistency flags (example: battery fusion signals)
- deterministic rule / pattern features (examples: `rocket_clean`, `fast_pattern_v2`)

See `docs/01_feature_spec.md` for details.

---

## Quickstart (optional demo)

1) Install audit primitives:

```sql
\i sql/00_audit_primitives_public.sql
```

2) Create synthetic input surfaces (optional):

```sql
\i sql/01_example_input_surfaces_public.sql
```

3) Create the derived store:

```sql
\i sql/02_create_trainer_derived_store_public.sql
```

4) Capture baselines, rebaseline recent days, certify:

```sql
\i sql/03_certify_trainer_derived_store_public.sql
```

5) Validate:

```sql
\i sql/07_validation_queries_public.sql
```

---

## Public release notes

This package intentionally **does not** ship:
- any marketplace-specific connector logic
- any private schemas, credentials, or environment-specific configuration
- any real data or identifiers

All example identifiers are synthetic.

