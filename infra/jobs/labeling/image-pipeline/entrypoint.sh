#!/usr/bin/env sh
set -euo pipefail

# Public-release entrypoint for a 4-stage, job-oriented image enrichment pipeline:
#   (1) scrape_images_playwright.py
#   (2) analyze_images_gpt5nano.py        (accessories + battery screenshot)
#   (3) images_iphone_color.py            (body color + model consistency)
#   (4) image_damage_analysis.py          (damage + photo quality)
#
# NO secrets are embedded. Provide all credentials via environment variables.

require_env() {
  name="$1"
  if [ -z "${!name:-}" ]; then
    echo "[FATAL] Missing required env var: ${name}" 1>&2
    exit 2
  fi
}

# Required for all stages
require_env PG_DSN

# Optional paths / tuning
: "${IMAGE_ROOT_DIR:=./listing_images}"
: "${PLAYWRIGHT_BATCH_SIZE:=50}"
: "${PLAYWRIGHT_MAX_IMAGES:=16}"
: "${PIPELINE_LIMIT_LISTINGS:=5500}"

# Model choices (override as needed)
: "${MODEL_ACCESSORIES:=gpt-5-nano-2025-08-07}"
: "${MODEL_COLOR:=gpt-5-nano-2025-08-07}"
: "${MODEL_DAMAGE:=gpt-5-mini}"

# Per-stage max images
: "${MAX_IMAGES_ACCESSORIES:=10}"
: "${MAX_IMAGES_COLOR:=10}"
: "${MAX_IMAGES_DAMAGE:=8}"

# Per-stage resize/quality (token-cost controls)
: "${ACCESSORIES_MAX_IMAGE_LONG_SIDE:=1024}"
: "${ACCESSORIES_JPEG_QUALITY:=90}"

: "${COLOR_MAX_IMAGE_LONG_SIDE:=640}"
: "${COLOR_JPEG_QUALITY:=75}"

: "${DAMAGE_MAX_IMAGE_LONG_SIDE:=1024}"
: "${DAMAGE_JPEG_QUALITY:=90}"

echo "[PIPELINE] starting"
echo "[PIPELINE] IMAGE_ROOT_DIR=${IMAGE_ROOT_DIR}"
echo "[PIPELINE] PLAYWRIGHT_BATCH_SIZE=${PLAYWRIGHT_BATCH_SIZE} PLAYWRIGHT_MAX_IMAGES=${PLAYWRIGHT_MAX_IMAGES}"
echo "[PIPELINE] LIMIT_LISTINGS=${PIPELINE_LIMIT_LISTINGS}"

# -----------------------------
# (1) Scrape/download images
# -----------------------------
echo "[STEP 1/4] scrape_images_playwright.py"
IMAGE_ROOT_DIR="${IMAGE_ROOT_DIR}" \
PLAYWRIGHT_BATCH_SIZE="${PLAYWRIGHT_BATCH_SIZE}" \
PLAYWRIGHT_MAX_IMAGES="${PLAYWRIGHT_MAX_IMAGES}" \
python -u /app/scrape_images_playwright.py

# -----------------------------
# (2) Accessories analysis (requires OPENAI_API_KEY)
# -----------------------------
echo "[STEP 2/4] analyze_images_gpt5nano.py (accessories)"
require_env OPENAI_API_KEY
OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}" \
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}" \
IMAGE_ROOT_DIR="${IMAGE_ROOT_DIR}" \
MAX_IMAGE_LONG_SIDE="${ACCESSORIES_MAX_IMAGE_LONG_SIDE}" \
JPEG_QUALITY="${ACCESSORIES_JPEG_QUALITY}" \
python -u /app/analyze_images_gpt5nano.py \
  --limit-listings "${PIPELINE_LIMIT_LISTINGS}" \
  --model "${MODEL_ACCESSORIES}" \
  --max-images-per-listing "${MAX_IMAGES_ACCESSORIES}"

# -----------------------------
# (3) Color analysis (requires OPENAI_API_KEY)
# -----------------------------
echo "[STEP 3/4] images_iphone_color.py (color)"
OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}" \
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}" \
IMAGE_ROOT_DIR="${IMAGE_ROOT_DIR}" \
MAX_IMAGE_LONG_SIDE="${COLOR_MAX_IMAGE_LONG_SIDE}" \
JPEG_QUALITY="${COLOR_JPEG_QUALITY}" \
python -u /app/images_iphone_color.py \
  --limit-listings "${PIPELINE_LIMIT_LISTINGS}" \
  --model "${MODEL_COLOR}" \
  --max-images-per-listing "${MAX_IMAGES_COLOR}"

# -----------------------------
# (4) Damage analysis (requires OPENAI_API_KEY)
# -----------------------------
echo "[STEP 4/4] image_damage_analysis.py (damage)"
OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}" \
IMAGE_ROOT_DIR="${IMAGE_ROOT_DIR}" \
MAX_IMAGE_LONG_SIDE="${DAMAGE_MAX_IMAGE_LONG_SIDE}" \
JPEG_QUALITY="${DAMAGE_JPEG_QUALITY}" \
python -u /app/image_damage_analysis.py \
  --limit-listings "${PIPELINE_LIMIT_LISTINGS}" \
  --model "${MODEL_DAMAGE}" \
  --max-images-per-listing "${MAX_IMAGES_DAMAGE}"

echo "[PIPELINE] done"
