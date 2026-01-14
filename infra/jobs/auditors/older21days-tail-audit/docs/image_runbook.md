# Docker Image Runbook (Public Template)

This document describes how to build, run, and verify the **stale listing audit** container image.

The image contains:
- a single static Go binary
- a small shell entrypoint that runs the job in “one-shot” mode

---

## 1) Build

From the repo root:

```bash
docker build -f Dockerfile.audit_stale_listings -t marketplace-audit-stale:latest .
```

---

## 2) Run (one-shot job)

Provide runtime config via environment variables. Do not hardcode secrets in Compose files committed to git.

```bash
docker run --rm \
  -e PG_DSN="$PG_DSN" \
  -e PG_SCHEMA="marketplace" \
  -e ADAPTER_KIND="mock" \
  marketplace-audit-stale:latest
```

### Running multiple cohorts sequentially (optional)

If you use a cohort key (e.g., `generation`) and want one container run to process multiple cohorts:

```bash
docker run --rm \
  -e PG_DSN="$PG_DSN" \
  -e PG_SCHEMA="marketplace" \
  -e GENERATIONS="1,2,3" \
  marketplace-audit-stale:latest
```

---

## 3) Example docker-compose service (dev)

This is a **job-style** service (does not auto-start unless you run it):

```yaml
services:
  audit-stale-listings:
    build:
      context: .
      dockerfile: Dockerfile.audit_stale_listings
    image: marketplace-audit-stale:latest
    environment:
      PG_DSN: ${PG_DSN}
      PG_SCHEMA: ${PG_SCHEMA:-marketplace}
      ADAPTER_KIND: ${ADAPTER_KIND:-mock}
      GENERATION: ${GENERATION:-0}
    restart: "no"
    profiles: ["job"]
```

---

## 4) Airflow integration

The repo includes a sanitized DAG:

- `airflow/dags/audit_stale_listings_daily.py`

It uses `DockerOperator` to run this image as a one-shot job.

All secrets (including `PG_DSN` and any adapter auth header) must be injected via Airflow Variables/Connections or worker environment variables.

---

## 5) Verification

Use the SQL checks in `docs/runbook_audit_stale_listings.md` to verify:
- daily baselines exist
- per-listing history is sparse
- terminal flips are attributed to the audit job
- inactive state events are event-based
