# Redaction report (public release)

This repository is a sanitized, platform-agnostic template derived from an internal codebase.

## Summary of changes

### Secrets / credentials removed
- Removed hardcoded database connection strings and any embedded credentials from:
  - container build configuration
  - job entrypoints
  - orchestration manifests (Airflow DAG)
- Introduced environment-variable-only configuration:
  - `PG_DSN` is never defaulted to a value in code, Dockerfiles, or docs.
  - `.env.example` contains placeholders only and leaves secrets blank.

### Target-platform identifiers and fingerprints removed
- Removed any site- or platform-identifying strings, URLs, endpoint paths, and parsing logic.
- Replaced any platform-specific ID fields with the generic `listing_id` concept.
- Implemented a platform adapter layer (`adapters/marketplace_adapter.go`) so core job logic is generic.

### Personal data removed
- Removed any real listing IDs, seller/user identifiers, and target-derived example outputs.
- Kept examples purely synthetic.

## Files added or replaced (public release)

### Code
- `cmd/ingest/main.go`
  - Generic ingest job (search → fetch → normalize → sink)
  - Uses `MarketplaceAdapter` (no target logic embedded)
- `cmd/auditor/main.go`
  - Generic auditor job (re-check / refresh bounded sample)
- `adapters/marketplace_adapter.go`
  - Adapter interface + mock adapter + placeholder HTTP JSON adapter

### Infra / orchestration
- `Dockerfile`
  - Builds `ingestd` and `auditd` with no secret defaults
- `docker/entrypoint.sh`
  - Job-mode dispatch: `ingest`, `audit`, `pipeline`
  - Optional multi-segment sweep via `SWEEP_QUERIES`
- `airflow/dags/marketplace_jobs.py`
  - Example Airflow DAG using `DockerOperator`
  - Uses an Airflow Connection for `PG_DSN` (no embedded secrets)

### Database
- `sql/schema.sql`
  - `marketplace_listings` normalized table
  - `marketplace_audit_runs` run metadata table

### Documentation
- `README.md`
  - Public-release overview and run instructions
- `.env.example`
  - Placeholder configuration (no secrets)

## Placeholders introduced
- Adapter configuration:
  - `MARKETPLACE_ADAPTER` = `mock` (offline) or `http-json`
  - `MARKETPLACE_BASE_URL` (required for `http-json`)
- Ingest tuning:
  - `SEARCH_QUERY`, `PAGES`, `WORKERS`, `SEARCH_WORKERS`, `REQUEST_RPS`, `MIN_PRICE`
- Audit tuning:
  - `AUDIT_SINCE_DAYS`, `AUDIT_LIMIT`, `AUDIT_WORKERS`, `AUDIT_RPS`, `AUDIT_DRY_RUN`
- Postgres:
  - `PG_DSN`, `PG_SCHEMA`, `PG_MAX_CONNS`, `PG_VIA_BOUNCER`, `PG_BATCH`

## Notes for private forks
Implement any target-specific connector behind the `MarketplaceAdapter` interface in a private repository. The public template is intentionally generic.
