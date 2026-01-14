# airflow/dags/audit_stale_listings_daily.py
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

# NOTE:
# - This DAG is a public-release template.
# - Do not embed secrets (DSNs, passwords, tokens) in the DAG file.
# - Inject sensitive values via Airflow Variables/Connections or worker env vars.

DOCKER_NETWORK = os.getenv("DOCKER_NETWORK_MODE", "bridge")

DEFAULT_ARGS = dict(
    owner=os.getenv("AIRFLOW_OWNER", "airflow"),
    depends_on_past=False,
    email_on_failure=False,
    email_on_retry=False,
    retries=int(os.getenv("AUDIT_RETRIES", "0")),
    execution_timeout=timedelta(minutes=int(os.getenv("AUDIT_EXEC_TIMEOUT_MIN", "120"))),
)

with DAG(
    dag_id=os.getenv("AUDIT_DAG_ID", "audit_stale_listings_daily"),
    description="Daily audit of stale listings in Postgres (public template)",
    schedule=os.getenv("AUDIT_SCHEDULE", "45 23 * * *"),  # daily @ 23:45 UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["audit", "stale", "template"],
) as dag:
    run_audit = DockerOperator(
        task_id="run_audit_stale_listings",
        image=os.getenv("AUDIT_IMAGE", "marketplace-audit-stale:latest"),
        api_version="auto",
        auto_remove=True,
        docker_url=os.getenv("DOCKER_URL", "unix://var/run/docker.sock"),
        network_mode=DOCKER_NETWORK,
        entrypoint=None,  # use image ENTRYPOINT
        command=None,
        mount_tmp_dir=False,
        environment={
            # Postgres (inject at runtime; leave blank here)
            "PG_DSN": os.getenv("PG_DSN", ""),
            "PG_SCHEMA": os.getenv("PG_SCHEMA", "marketplace"),
            "GENERATION": os.getenv("GENERATION", "0"),

            # Selection
            "STALE_STATUS": os.getenv("STALE_STATUS", "older21days"),
            "AUDIT_DAYS": os.getenv("AUDIT_DAYS", "0"),
            "AUDIT_LIMIT": os.getenv("AUDIT_LIMIT", "0"),
            "AUDIT_ONLY_IDS": os.getenv("AUDIT_ONLY_IDS", ""),

            # Adapter
            "ADAPTER_KIND": os.getenv("ADAPTER_KIND", "mock"),
            "MARKETPLACE_BASE_URL": os.getenv("MARKETPLACE_BASE_URL", "https://marketplace.example"),
            "MARKETPLACE_AUTH_HEADER": os.getenv("MARKETPLACE_AUTH_HEADER", ""),

            # Concurrency + request bounding
            "WORKERS": os.getenv("WORKERS", "16"),
            "REQUEST_MIN_RPS": os.getenv("REQUEST_MIN_RPS", "2.0"),
            "REQUEST_MAX_RPS": os.getenv("REQUEST_MAX_RPS", "10.0"),
            "REQUEST_STEP_UP_RPS": os.getenv("REQUEST_STEP_UP_RPS", "0.5"),
            "REQUEST_DOWN_MULT": os.getenv("REQUEST_DOWN_MULT", "0.60"),
            "REQUEST_BURST_FACTOR": os.getenv("REQUEST_BURST_FACTOR", "2.0"),
            "REQUEST_JITTER_MS": os.getenv("REQUEST_JITTER_MS", "150"),
            "REQUEST_RETRY_MAX": os.getenv("REQUEST_RETRY_MAX", "3"),
            "REQUEST_FALLBACK_THROTTLE_MS": os.getenv("REQUEST_FALLBACK_THROTTLE_MS", "3000"),

            "VERBOSE": os.getenv("VERBOSE", "false"),
        },
    )
