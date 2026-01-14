from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

# Public-release DAG
# ------------------
# This DAG demonstrates a job-oriented pattern using DockerOperator.
#
# Secrets are not embedded in this file. Store Postgres credentials in an Airflow
# Connection (example ID: "marketplace_postgres") and reference it via Jinja.

DOCKER_NETWORK = os.getenv("DOCKER_NETWORK_MODE", "bridge")
IMAGE = os.getenv("MARKETPLACE_INGEST_IMAGE", "marketplace-ingest:latest")

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "execution_timeout": timedelta(minutes=30),
}

with DAG(
    dag_id="marketplace_ingest_and_audit",
    description="Run marketplace ingest and (optional) audit jobs on a schedule",
    schedule_interval="*/30 * * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["marketplace", "ingest", "audit"],
) as dag:
    ingest = DockerOperator(
        task_id="ingest_listings",
        image=IMAGE,
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        entrypoint=None,   # use image ENTRYPOINT
        command=None,      # entrypoint reads JOB env
        mount_tmp_dir=False,
        environment={
            "JOB": "ingest",

            # Adapter selection. In a real deployment, provide MARKETPLACE_BASE_URL and set
            # MARKETPLACE_ADAPTER=http-json (or a private adapter).
            "MARKETPLACE_ADAPTER": "{{ var.value.get('marketplace_adapter', 'http-json') }}",
            "MARKETPLACE_BASE_URL": "{{ var.value.get('marketplace_base_url', 'https://marketplace.example') }}",

            # Ingest settings (all strings because DockerOperator env is str->str)
            "SEARCH_QUERY": "{{ var.value.get('search_query', 'example query') }}",
            "PAGES": "{{ var.value.get('pages', '1') }}",
            "WORKERS": "{{ var.value.get('workers', '16') }}",
            "SEARCH_WORKERS": "{{ var.value.get('search_workers', '4') }}",
            "REQUEST_RPS": "{{ var.value.get('request_rps', '0') }}",

            # Sink: Postgres (credentials via Airflow Connection)
            "PG_DSN": "{{ conn.marketplace_postgres.get_uri() }}",
            "PG_SCHEMA": "{{ var.value.get('pg_schema', 'public') }}",
        },
    )

    audit = DockerOperator(
        task_id="audit_listings",
        image=IMAGE,
        api_version="auto",
        auto_remove=True,
        docker_url="unix://var/run/docker.sock",
        network_mode=DOCKER_NETWORK,
        entrypoint=None,
        command=None,
        mount_tmp_dir=False,
        environment={
            "JOB": "audit",

            "MARKETPLACE_ADAPTER": "{{ var.value.get('marketplace_adapter', 'http-json') }}",
            "MARKETPLACE_BASE_URL": "{{ var.value.get('marketplace_base_url', 'https://marketplace.example') }}",

            # Audit settings
            "AUDIT_SINCE_DAYS": "{{ var.value.get('audit_since_days', '7') }}",
            "AUDIT_LIMIT": "{{ var.value.get('audit_limit', '500') }}",
            "AUDIT_WORKERS": "{{ var.value.get('audit_workers', '8') }}",
            "AUDIT_RPS": "{{ var.value.get('audit_rps', '0') }}",
            "AUDIT_DRY_RUN": "{{ var.value.get('audit_dry_run', '0') }}",

            "PG_DSN": "{{ conn.marketplace_postgres.get_uri() }}",
            "PG_SCHEMA": "{{ var.value.get('pg_schema', 'public') }}",
        },
    )

    ingest >> audit
