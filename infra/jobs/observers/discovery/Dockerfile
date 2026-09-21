# syntax=docker/dockerfile:1

# Public-release Dockerfile
# -------------------------
# Builds two job-mode binaries:
#   • ingestd  (search -> fetch -> normalize -> sink)
#   • auditd   (re-check / refresh a bounded sample in Postgres)
#
# No secrets are baked into the image. Configure everything via environment
# variables at runtime (see .env.example and README.md).

FROM golang:1.22-alpine AS build
WORKDIR /src
RUN apk add --no-cache ca-certificates git

# Prime the module cache (go.sum is optional)
COPY go.mod ./
RUN go mod download

# Bring in source
COPY . .

# Build static binaries
RUN CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o /out/ingestd ./cmd/ingest && \
    CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o /out/auditd  ./cmd/auditor

FROM alpine:3.19
RUN apk add --no-cache ca-certificates tzdata
WORKDIR /app

COPY --from=build /out/ingestd /usr/local/bin/ingestd
COPY --from=build /out/auditd  /usr/local/bin/auditd

COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# Safe defaults only (no secrets)
ENV JOB="ingest" \
    MARKETPLACE_ADAPTER="mock" \
    MARKETPLACE_BASE_URL="" \
    SEARCH_QUERY="example query" \
    OUT_CSV="/data/listings.csv" \
    PG_SCHEMA="public" \
    PAGES="1" \
    WORKERS="16" \
    SEARCH_WORKERS="4" \
    REQUEST_RPS="0" \
    MIN_PRICE="0" \
    AUDIT_SINCE_DAYS="7" \
    AUDIT_LIMIT="500" \
    AUDIT_WORKERS="8" \
    AUDIT_RPS="0"

ENTRYPOINT ["sh", "/entrypoint.sh"]
