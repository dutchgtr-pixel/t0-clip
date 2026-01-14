from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator


# Public-release DAG:
# - Runs a single container that sequentially executes the 4 scripts (via entrypoint.sh).
# - Schedules every 30 minutes.
# - NO secrets are embedded. Provide PG_DSN / OPENAI_API_KEY via Airflow Variables,
#   environment injection, or your secrets backend.

DOCKER_NETWORK = os.getenv("DOCKER_NETWORK_MODE", "bridge")
IMAGE_NAME = os.getenv("PIPELINE_IMAGE", "marketplace-image-pipeline:latest")

DEFAULT_ARGS = dict(
    owner="airflow",
    depends_on_past=False,
    email_on_failure=False,
    email_on_retry=False,
    retries=0,
    execution_timeout=timedelta(minutes=120),
)

def env(name: str, default: str = "") -> str:
    # Prefer task env, fall back to empty/default. In production, inject via Airflow Variables/Connections/Secrets.
    return os.getenv(name, default)

with DAG(
    dag_id="marketplace_image_pipeline_30m",
    description="Runs a 4-stage image enrichment pipeline (scrape -> accessories -> color -> damage) every 30 minutes.",
    schedule="*/30 * * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["pipeline", "images", "marketplace", "job"],
) as dag:

    run_pipeline = DockerOperator(
        task_id="run_image_pipeline",
        image=IMAGE_NAME,
        api_version="auto",
        auto_remove=True,
        docker_url=env("DOCKER_URL", "unix://var/run/docker.sock"),
        network_mode=DOCKER_NETWORK,
        mount_tmp_dir=False,
        environment={
            # Required
            "PG_DSN": env("PG_DSN", ""),
            "OPENAI_API_KEY": env("OPENAI_API_KEY", ""),

            # Optional overrides
            "IMAGE_ROOT_DIR": env("IMAGE_ROOT_DIR", "./listing_images"),
            "PLAYWRIGHT_BATCH_SIZE": env("PLAYWRIGHT_BATCH_SIZE", "50"),
            "PLAYWRIGHT_MAX_IMAGES": env("PLAYWRIGHT_MAX_IMAGES", "16"),
            "PIPELINE_LIMIT_LISTINGS": env("PIPELINE_LIMIT_LISTINGS", "5500"),

            "MODEL_ACCESSORIES": env("MODEL_ACCESSORIES", "gpt-5-nano-2025-08-07"),
            "MODEL_COLOR": env("MODEL_COLOR", "gpt-5-nano-2025-08-07"),
            "MODEL_DAMAGE": env("MODEL_DAMAGE", "gpt-5-mini"),

            "MAX_IMAGES_ACCESSORIES": env("MAX_IMAGES_ACCESSORIES", "10"),
            "MAX_IMAGES_COLOR": env("MAX_IMAGES_COLOR", "10"),
            "MAX_IMAGES_DAMAGE": env("MAX_IMAGES_DAMAGE", "8"),

            "ACCESSORIES_MAX_IMAGE_LONG_SIDE": env("ACCESSORIES_MAX_IMAGE_LONG_SIDE", "1024"),
            "ACCESSORIES_JPEG_QUALITY": env("ACCESSORIES_JPEG_QUALITY", "90"),
            "COLOR_MAX_IMAGE_LONG_SIDE": env("COLOR_MAX_IMAGE_LONG_SIDE", "640"),
            "COLOR_JPEG_QUALITY": env("COLOR_JPEG_QUALITY", "75"),
            "DAMAGE_MAX_IMAGE_LONG_SIDE": env("DAMAGE_MAX_IMAGE_LONG_SIDE", "1024"),
            "DAMAGE_JPEG_QUALITY": env("DAMAGE_JPEG_QUALITY", "90"),

            # Optional API base override (e.g., if using a compatible gateway)
            "OPENAI_API_BASE": env("OPENAI_API_BASE", "https://api.openai.com/v1"),
            "OPENAI_BASE_URL": env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        },
    )
