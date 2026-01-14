# Public-release Dockerfile for the 4-stage image pipeline
# No secrets are embedded. Configure via environment variables at runtime.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for Playwright + basic build tools (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (use psycopg2-binary to avoid build toolchain)
RUN pip install --no-cache-dir \
      playwright==1.49.0 \
      requests==2.32.3 \
      psycopg2-binary==2.9.9 \
      pillow==10.4.0 \
      openai==1.40.6

# Install Chromium for Playwright
RUN python -m playwright install --with-deps chromium

# Copy pipeline scripts
COPY scrape_images_playwright.py /app/scrape_images_playwright.py
COPY analyze_images_gpt5nano.py /app/analyze_images_gpt5nano.py
COPY images_iphone_color.py /app/images_iphone_color.py
COPY image_damage_analysis.py /app/image_damage_analysis.py

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
