# Drift-tolerant certification policy

## Why drift can exist even with T0 discipline
Even when features avoid target leakage and ban time-of-query tokens, *values can still drift* due to upstream backfills:

- late-arriving enrichments (e.g., image-derived attributes)
- corrected upstream fields (price, condition scores, etc.)
- late ingestion of historical rows

This drift is not leakage, but it is nondeterminism over time.

## Recommended policy
- Keep long baseline history for auditability (e.g., 365 days).
- Allow rebaselining only in the last **N days** (commonly N=10) to accommodate backfills.
- Certify using drift checks over the same last **N days** window.

## Implementation primitives (public)
- `audit.capture_viewdef_baseline(entrypoint)`
- `audit.rebaseline_last_n_days(entrypoint, n, sample_limit)`
- `audit.run_t0_cert_trainer_derived_store_v1(p_check_days)`
- `audit.require_certified_strict(entrypoint, max_age)`

