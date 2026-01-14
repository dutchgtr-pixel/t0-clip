# etl/dags/status_sanity_audit_midnight.public.py
"""Public-release Airflow DAG (sanitized): status sanity audit (job-mode).

This DAG demonstrates a "job-mode" pattern:
- Airflow schedules a Docker container once per day at midnight.
- The container runs the audit and exits.
- No credentials are embedded; all config is supplied via environment variables and/or Airflow Connections.

Public-release constraints:
- No target-platform identifiers or scraping fingerprints.
- All marketplace-specific logic must be behind the adapter layer inside the container image.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator


DOCKER_NETWORK = os.getenv("DOCKER_NETWORK_MODE", "bridge")

DEFAULT_ARGS = dict(
    owner="airflow",
    depends_on_past=False,
    email_on_failure=False,
    email_on_retry=False,
    retries=0,
    execution_timeout=timedelta(minutes=int(os.getenv("JOB_TIMEOUT_MINUTES", "90"))),
)

with DAG(
    dag_id="status_sanity_audit_midnight",
    description="Daily status sanity audit (job mode) for marketplace listings",
    schedule=os.getenv("DAG_SCHEDULE_CRON", "0 0 * * *"),
    start_date=datetime(2025, 5, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["marketplace", "audit", "status"],
) as dag:

    run_status_sanity_audit = DockerOperator(
        task_id="run_status_sanity_audit",
        image=os.getenv("STATUS_SANITY_IMAGE", "status-sanity-audit:latest"),
        api_version="auto",
        auto_remove=True,
        docker_url=os.getenv("DOCKER_HOST", "unix://var/run/docker.sock"),
        network_mode=DOCKER_NETWORK,
        # Use image ENTRYPOINT (job-mode) by default.
        entrypoint=None,
        command=None,
        mount_tmp_dir=False,
        environment={
            # Database connection MUST be provided externally.
            "PG_DSN": os.getenv("PG_DSN", ""),
            "PG_SCHEMA": os.getenv("PG_SCHEMA", "marketplace"),

            # Adapter config (safe public defaults).
            "MARKETPLACE_ADAPTER": os.getenv("MARKETPLACE_ADAPTER", "mock"),
            "MARKETPLACE_BASE_URL": os.getenv("MARKETPLACE_BASE_URL", "https://example-marketplace.invalid"),
            # Secret at runtime; do not commit. Prefer an Airflow Connection / secret manager.
            "MARKETPLACE_AUTH_HEADER": os.getenv("MARKETPLACE_AUTH_HEADER", ""),

            # Job behavior (optional)
            "MODE": os.getenv("MODE", "run"),
            "SCOPE": os.getenv("SCOPE", "all"),
            "CADENCE_DAYS": os.getenv("CADENCE_DAYS", "30"),
            "WORKERS": os.getenv("WORKERS", "16"),
            "REQUEST_RPS": os.getenv("REQUEST_RPS", "3.0"),
            "REQUEST_RPS_MAX": os.getenv("REQUEST_RPS_MAX", "10.0"),
            "REQUEST_RPS_MIN": os.getenv("REQUEST_RPS_MIN", "0.25"),

            # Mutation guardrails (default: none)
            "APPLY_MODE": os.getenv("APPLY_MODE", "none"),  # none|safe|all
            "ALLOW_UNSELL": os.getenv("ALLOW_UNSELL", "false"),
            "ALLOW_REVIVE": os.getenv("ALLOW_REVIVE", "false"),

            # Optional dev logging
            "LOG_JSON": os.getenv("LOG_JSON", "false"),
            "LOG_URL": os.getenv("LOG_URL", "false"),
            "WRITE_EVENTS": os.getenv("WRITE_EVENTS", "true"),
        },
    )
