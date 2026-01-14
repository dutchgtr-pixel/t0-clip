# -------- builder --------
FROM golang:1.23-alpine AS build
WORKDIR /src

RUN apk add --no-cache git ca-certificates

# Copy only go.mod first for better layer caching. (go.sum is intentionally not committed in this template.)
COPY go.mod ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" \
  -o /usr/local/bin/audit_stale_listings \
  .

# -------- runtime --------
FROM alpine:3.19
RUN apk add --no-cache ca-certificates tzdata
WORKDIR /app

COPY --from=build /usr/local/bin/audit_stale_listings /usr/local/bin/audit_stale_listings
COPY docker/entrypoint_audit_stale_listings.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
