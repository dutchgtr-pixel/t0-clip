# marketplace-ingest-template (public release)

This repository is a security-sanitized reference implementation of a **job-oriented data acquisition** pattern:

- A containerized **ingest job** (`ingestd`) that pulls listings via a pluggable adapter, normalizes them, and writes to a sink (Postgres or CSV).
- A companion **auditor job** (`auditd`) that re-checks a bounded sample of stored records and updates status / timestamps in Postgres.
- An example **Airflow DAG** showing how to schedule both jobs with `DockerOperator`.
- A minimal Postgres schema for the normalized listing table and audit run metadata.

This public release intentionally excludes any target-specific connector logic. The default adapter can run in **offline mock mode** to keep the project runnable without external dependencies.

## What is included

### Ingest job (`ingestd`)
- Search → fetch → normalize → sink pipeline.
- Bounded pressure controls:
  - configurable request rate limit (`REQUEST_RPS`)
  - adaptive concurrency gate (AIMD) based on observed latency / error rates
- Optional Postgres sink (`PG_DSN`) with idempotent inserts (`ON CONFLICT DO NOTHING`).
- Optional append-only CSV sink (`OUT_CSV`) with a sidecar ID index.
- Optional Prometheus-style `/metrics` and `/debug/pprof/*` endpoints (`METRICS_ADDR`).

### Auditor job (`auditd`)
- Selects a bounded set of stale records from Postgres (`AUDIT_SINCE_DAYS`, `AUDIT_LIMIT`).
- Re-fetches via the same adapter interface with bounded concurrency and optional RPS limit (`AUDIT_WORKERS`, `AUDIT_RPS`).
- Updates `status`, `last_seen`, and `last_fetched` (or runs in dry mode).
- Records run metadata in `marketplace_audit_runs`.

### Platform adapter layer
All external connectivity is behind a generic interface in `adapters/marketplace_adapter.go`. The public repo includes:
- `mock` adapter: generates synthetic listings (offline demo mode)
- `http-json` adapter: a placeholder JSON API client driven by `MARKETPLACE_BASE_URL`

## What is intentionally omitted
- Any target-specific connector logic (endpoints, selectors, parsing, headers, fingerprints).
- Any real credentials, tokens, cookies, DSNs, or private infrastructure identifiers.
- Any real listing/seller/location examples or logs derived from a production target.

If you need a real connector, implement it in a **private repository** and keep this repo as the generic job/infra pattern.

## Quick start (no Postgres, offline)

### Run ingest in mock + CSV mode
```bash
go run ./cmd/ingest \
  --out ./data/listings.csv \
  --pages 2
```

Or via Docker:
```bash
docker build -t marketplace-ingest:latest .
docker run --rm \
  -e MARKETPLACE_ADAPTER=mock \
  -e OUT_CSV=/data/listings.csv \
  -v "$PWD/data:/data" \
  marketplace-ingest:latest ingest
```

This will generate synthetic listing rows and write them to `./data/listings.csv`.

## Running with Postgres

1) Apply the schema:
```sql
-- see sql/schema.sql
```

2) Provide `PG_DSN` via your secret manager or runtime environment (do not commit it).

3) Run ingest with Postgres as the sink:
```bash
export PG_DSN="(set via secrets)"
export PG_SCHEMA="public"
export MARKETPLACE_ADAPTER="mock"   # or http-json with a real private endpoint
go run ./cmd/ingest
```

4) Run audit:
```bash
export PG_DSN="(set via secrets)"
go run ./cmd/auditor
```

## Entrypoint modes (Docker jobs)

The image ENTRYPOINT supports job-mode dispatch:

- `ingest` (default): run `ingestd`
- `audit`: run `auditd` (requires `PG_DSN`)
- `pipeline`: run ingest and then audit (requires `PG_DSN` for audit)

Example:
```bash
docker run --rm -e JOB=ingest marketplace-ingest:latest
docker run --rm -e JOB=audit  -e PG_DSN="(set via secrets)" marketplace-ingest:latest
```

To demonstrate a “multi-segment sweep” pattern (similar to running multiple partitions back-to-back), set:
- `SWEEP_QUERIES="query A|query B|query C"`
- `SLEEP_BETWEEN=10`

The entrypoint will run `ingestd` once per segment.

## Airflow DAG

See `airflow/dags/marketplace_jobs.py`.

Key points:
- No secrets are embedded in the DAG code.
- Store Postgres credentials in an Airflow Connection (example ID: `marketplace_postgres`).
- Configure non-secret parameters (base URL, query, tuning) via Airflow Variables.

## Configuration

A complete list of supported environment variables is in `.env.example`.

## Public release notes

GitHub Releases can attach “snapshot zips”, but they do not build or run containers. The correct pattern for a public artifact is:

- Commit code + Dockerfile + entrypoint + DAG in the repository.
- Tag a release and optionally attach a zip snapshot that matches the tag.

For the full redaction details and scan results, see `REDACTION_REPORT.md`.
