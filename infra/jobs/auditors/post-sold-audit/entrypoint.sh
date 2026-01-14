#!/usr/bin/env sh
set -euo pipefail

# Public template entrypoint: post-sold audit job (job-mode).
# - No credentials embedded.
# - No target-platform identifiers.
# - Fetching/parsing performed via the MarketplaceAdapter inside the binary.

: "${MODE:=run}"                 # init | seed | run | refresh-desc
: "${PG_DSN:=}"                  # required
: "${PG_SCHEMA:=marketplace}"
: "${LISTINGS_TABLE:=listings}"
: "${AUDIT_TABLE:=post_sold_audit}"

: "${MARKETPLACE_ADAPTER:=mock}" # mock | http-json
: "${MARKETPLACE_BASE_URL:=https://example-marketplace.invalid}"
: "${MARKETPLACE_AUTH_HEADER:=}" # optional secret (adapter=http-json)
: "${HTTP_USER_AGENT:=marketplace-audit-template/1.0}"

# Selection/snapshot policy
: "${MIN_AGE_DAYS:=7}"
: "${DAY_OFFSET:=7}"
: "${SEGMENTS:=}"                # optional CSV
: "${PER_SEGMENT_CAP:=0}"
: "${BATCH:=0}"
: "${ONLY_IDS:=}"                # optional CSV

# Runtime tuning
: "${WORKERS:=32}"
: "${REQUEST_TIMEOUT:=15s}"
: "${REQUEST_RPS:=3.0}"
: "${REQUEST_RPS_MAX:=12.0}"
: "${REQUEST_RPS_MIN:=0.7}"
: "${REQUEST_RPS_STEP:=0.5}"
: "${REQUEST_RPS_DOWN:=0.5}"
: "${REQUEST_COOL_OFF:=25s}"
: "${REQUEST_JITTER_MS:=120}"
: "${REQUEST_RETRY_MAX:=2}"
: "${REQUEST_BACKOFF_INITIAL:=2s}"
: "${REQUEST_BACKOFF_MAX:=20s}"
: "${MAX_CONNS_PER_HOST:=6}"

: "${DRY_RUN:=false}"

if [ -z "${PG_DSN}" ]; then
  echo "[post-sold-audit] ERROR: PG_DSN must be set" >&2
  exit 2
fi

export MODE PG_DSN PG_SCHEMA LISTINGS_TABLE AUDIT_TABLE \
  MARKETPLACE_ADAPTER MARKETPLACE_BASE_URL MARKETPLACE_AUTH_HEADER HTTP_USER_AGENT \
  MIN_AGE_DAYS DAY_OFFSET SEGMENTS PER_SEGMENT_CAP BATCH ONLY_IDS \
  WORKERS REQUEST_TIMEOUT REQUEST_RPS REQUEST_RPS_MAX REQUEST_RPS_MIN REQUEST_RPS_STEP REQUEST_RPS_DOWN REQUEST_COOL_OFF \
  REQUEST_JITTER_MS REQUEST_RETRY_MAX REQUEST_BACKOFF_INITIAL REQUEST_BACKOFF_MAX MAX_CONNS_PER_HOST \
  DRY_RUN

exec /usr/local/bin/post-sold-audit "$@"
