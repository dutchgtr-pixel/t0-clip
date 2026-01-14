# -------- builder --------
FROM golang:1.25-alpine AS build
WORKDIR /src

RUN apk add --no-cache git ca-certificates

# Copy only the public-release source into an isolated module so the image can
# be built as a standalone job-template (no repo-wide go.mod required).
COPY status_sanity_audit_public.go ./main.go

RUN go mod init example.com/public-template/status-sanity-audit \
 && go mod tidy

RUN CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o /out/status-sanity-audit ./main.go

# -------- runtime --------
FROM alpine:3.20
WORKDIR /app

RUN apk add --no-cache ca-certificates tzdata \
 && addgroup -S app \
 && adduser -S app -G app \
 && mkdir -p /app/out \
 && chown -R app:app /app

COPY --from=build /out/status-sanity-audit /usr/local/bin/status-sanity-audit
COPY jobs/status_sanity_audit/entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh

USER app

# Safe defaults (override via Docker/Airflow env)
ENV MODE="run" \
    PG_SCHEMA="marketplace" \
    MARKETPLACE_ADAPTER="mock" \
    MARKETPLACE_BASE_URL="https://example-marketplace.invalid" \
    WRITE_EVENTS="true" \
    APPLY_MODE="none"

ENTRYPOINT ["/entrypoint.sh"]
