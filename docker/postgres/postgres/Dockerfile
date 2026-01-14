# ------------------------------------------------------------------
# Postgres 15 + pg_partman + pg_cron + pgvector (build from source)
# ------------------------------------------------------------------
FROM postgres:15-bullseye

USER root

RUN set -eux; \
    apt-get update -qq; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      wget gnupg \
      git \
      build-essential \
      postgresql-server-dev-15 \
      postgresql-15-cron \
      postgresql-15-partman \
      postgresql-contrib ; \
    \
    # Add PGDG repo (needed for partman & cron packages)
    echo "deb [signed-by=/usr/share/keyrings/postgresql.gpg] \
      http://apt.postgresql.org/pub/repos/apt bullseye-pgdg main" \
      > /etc/apt/sources.list.d/pgdg.list; \
    wget -qO- https://www.postgresql.org/media/keys/ACCC4CF8.asc \
      | gpg --dearmor -o /usr/share/keyrings/postgresql.gpg; \
    apt-get update -qq; \
    \
    # Build & install pgvector
    git clone --depth=1 https://github.com/pgvector/pgvector.git /tmp/pgvector; \
    cd /tmp/pgvector; \
    make && make install; \
    cd /; rm -rf /tmp/pgvector; \
    \
    # Clean up apt caches
    rm -rf /var/lib/apt/lists/*

# Enable pg_cron on startup
RUN echo "shared_preload_libraries = 'pg_cron'" \
       >> /usr/share/postgresql/postgresql.conf.sample; \
    echo "cron.database_name = 'scrapes'" \
       >> /usr/share/postgresql/postgresql.conf.sample

USER postgres












