#!/usr/bin/env sh
set -euo pipefail

# Public template entrypoint
#
# This entrypoint intentionally contains:
#   - no credentials
#   - no target-marketplace endpoints
#   - no platform-specific fingerprints
#
# Provide secrets at runtime (Docker secrets, CI/CD secret store, Airflow Connections, etc.)

: "${PG_DSN:?PG_DSN must be set at runtime (do not commit it to the repo)}"
: "${PG_SCHEMA:=marketplace}"

: "${MODE:=audit-stale}"

# Optional: comma-separated list of cohort keys to run sequentially.
# If unset, the job uses GENERATION (default may be 0).
if [ -n "${GENERATIONS:-}" ]; then
  for g in $(echo "${GENERATIONS}" | tr ',' ' '); do
    g="$(echo "$g" | tr -d ' ')"
    [ -z "$g" ] && continue
    echo "[entrypoint] running generation=${g}"
    GENERATION="$g" /usr/local/bin/audit_stale_listings --mode "${MODE}" "$@" || true
    if [ "${SLEEP_BETWEEN_RUNS:-0}" != "0" ]; then
      sleep "${SLEEP_BETWEEN_RUNS}"
    fi
  done
  exit 0
fi

exec /usr/local/bin/audit_stale_listings --mode "${MODE}" "$@"
