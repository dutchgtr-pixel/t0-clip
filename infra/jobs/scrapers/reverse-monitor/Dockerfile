# -------- builder --------
FROM golang:1.25-alpine AS build
WORKDIR /src

RUN apk add --no-cache git ca-certificates
COPY go.mod go.sum ./
RUN go mod download

# Bring in repository sources
COPY . .

# Build target can be overridden at build time:
#   docker build --build-arg GO_BUILD_TARGET=./cmd/survival -f Dockerfile.survival.public .
ARG GO_BUILD_TARGET=./
RUN CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o /usr/local/bin/survival ${GO_BUILD_TARGET}

# -------- runtime --------
FROM alpine:3.19
RUN apk add --no-cache ca-certificates tzdata
WORKDIR /app

COPY --from=build /usr/local/bin/survival /usr/local/bin/survival

# Entrypoint path can be overridden at build time:
#   docker build --build-arg ENTRYPOINT_PATH=docker/entrypoint_survival.sh -f Dockerfile.survival.public .
ARG ENTRYPOINT_PATH=entrypoint_survival.public.sh
COPY ${ENTRYPOINT_PATH} /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

# Safe defaults (override in docker-compose / Kubernetes manifests)
# NOTE: PG_DSN is intentionally not set here.
ENV MODE="diagnose" \
    MARKETPLACE_ADAPTER="mock" \
    MARKETPLACE_BASE_URL="https://marketplace.example" \
    HTTP_USER_AGENT="public-template/1.0" \
    PG_SCHEMA="public" \
    GEN_LIST="1" \
    SCAN_SINCE_DAYS="30" \
    MAX_PROBE="10000" \
    WORKERS="32" \
    REQUEST_RPS="12" \
    WRITE_DB="false" \
    WRITE_CSV="false" \
    VERBOSE="false"

ENTRYPOINT ["/entrypoint.sh"]
