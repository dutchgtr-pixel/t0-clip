//go:build !js
// +build !js

// Marketplace listing survival tool (public template)
//
// This program demonstrates a job-oriented pattern for:
//   - probing listing state (live/sold/removed) using a pluggable adapter
//   - persisting authoritative terminal events into a main table
//   - writing sparse price/status history (daily baseline + change/terminal events)
//   - tracking separate UI/state flags (e.g., inactive, bidding) without conflating them with main status
//
// IMPORTANT (public release posture)
//   - No target site identifiers are embedded.
//   - No credentials, tokens, cookies, or private headers are embedded.
//   - All marketplace-specific fetching/parsing must be implemented behind the MarketplaceAdapter interface.
//
// Database design assumptions (template)
//   - Main table:   <schema>.listings
//   - History table:<schema>.price_history
//   - State events: <schema>.inactive_state_events
//
// All runtime configuration is provided via environment variables and/or flags.

package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

/* ========================= Environment helpers ========================= */

func envString(key, def string) string {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	return v
}

func envInt(key string, def int) int {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}

func envBool(key string, def bool) bool {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	switch strings.ToLower(v) {
	case "1", "true", "t", "yes", "y", "on":
		return true
	case "0", "false", "f", "no", "n", "off":
		return false
	default:
		return def
	}
}

func envFloat(key string, def float64) float64 {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return def
	}
	return f
}

/* ========================= CLI & Config ========================= */

type config struct {
	mode string // reverse | repair | stale | diagnose | test | audit-status

	// Adapter
	adapter            string // mock | http-json
	marketplaceBaseURL string // used by http-json adapter
	userAgent          string // optional; used by http-json adapter

	// DB
	pgDSN      string
	pgSchema   string
	generation int

	// Outputs
	writeDB    bool
	writeCSV   bool
	freshCSV   bool
	out        string
	historyOut string

	// Reverse window
	scanSinceDays int
	maxProbe      int

	// Concurrency / HTTP
	workers int
	rps     int
	verbose bool

	// Retry / throttle
	throttleSleepMs int
	retryMax        int
	jitterMs        int

	// Dynamic limiter knobs
	minRPS      float64
	maxRPS      float64
	stepUpRPS   float64
	downMult    float64
	burstFactor float64

	// Mode-specific knobs
	testHours int
	testBand  int
	testLimit int

	repairFallbackDays int
	repairThresholdSec int
	repairLimit        int
	onlyIDs            string

	auditDays    int
	auditLimit   int
	auditOnlyIDs string
}

func parseFlags() config {
	var cfg config

	// Defaults from env (flags override)
	cfg.mode = envString("MODE", "test")

	cfg.adapter = envString("MARKETPLACE_ADAPTER", "mock")
	cfg.marketplaceBaseURL = envString("MARKETPLACE_BASE_URL", "https://marketplace.example")
	cfg.userAgent = envString("HTTP_USER_AGENT", "public-template/1.0")

	cfg.pgDSN = envString("PG_DSN", "")
	cfg.pgSchema = envString("PG_SCHEMA", "public")
	cfg.generation = envInt("GENERATION", 1)

	cfg.writeDB = envBool("WRITE_DB", false)
	cfg.writeCSV = envBool("WRITE_CSV", false)
	cfg.freshCSV = envBool("FRESH_CSV", false)
	cfg.out = envString("OUT", "")
	cfg.historyOut = envString("HISTORY_OUT", "")

	cfg.scanSinceDays = envInt("SCAN_SINCE_DAYS", 30)
	cfg.maxProbe = envInt("MAX_PROBE", 10000)

	cfg.workers = envInt("WORKERS", 32)
	cfg.rps = envInt("REQUEST_RPS", 12)
	cfg.verbose = envBool("VERBOSE", false)

	cfg.throttleSleepMs = envInt("THROTTLE_SLEEP_MS", 3000)
	cfg.retryMax = envInt("RETRY_MAX", 4)
	cfg.jitterMs = envInt("JITTER_MS", 150)

	cfg.minRPS = envFloat("MIN_RPS", 3.0)
	cfg.maxRPS = envFloat("MAX_RPS", 0.0) // 0 => use rps
	cfg.stepUpRPS = envFloat("STEP_UP_RPS", 0.5)
	cfg.downMult = envFloat("DOWN_MULT", 0.60)
	cfg.burstFactor = envFloat("BURST_FACTOR", 2.0)

	cfg.testHours = envInt("TEST_HOURS", 48)
	cfg.testBand = envInt("TEST_BAND", 2)
	cfg.testLimit = envInt("TEST_LIMIT", 30)

	cfg.repairFallbackDays = envInt("REPAIR_FALLBACK_DAYS", 60)
	cfg.repairThresholdSec = envInt("REPAIR_THRESHOLD_SEC", 300)
	cfg.repairLimit = envInt("REPAIR_LIMIT", 0)
	cfg.onlyIDs = envString("ONLY_IDS", "")

	cfg.auditDays = envInt("AUDIT_DAYS", 90)
	cfg.auditLimit = envInt("AUDIT_LIMIT", 0)
	cfg.auditOnlyIDs = envString("AUDIT_ONLY_IDS", "")

	// Flags (override env)
	flag.StringVar(&cfg.mode, "mode", cfg.mode, "Mode: reverse | repair | stale | diagnose | test | audit-status")

	flag.StringVar(&cfg.adapter, "adapter", cfg.adapter, "Marketplace adapter: mock | http-json")
	flag.StringVar(&cfg.marketplaceBaseURL, "marketplace-base-url", cfg.marketplaceBaseURL, "Base URL for http-json adapter")
	flag.StringVar(&cfg.userAgent, "user-agent", cfg.userAgent, "HTTP User-Agent (http-json adapter)")

	flag.StringVar(&cfg.pgDSN, "pg-dsn", cfg.pgDSN, "Postgres DSN (recommended via PG_DSN env var)")
	flag.StringVar(&cfg.pgSchema, "pg-schema", cfg.pgSchema, "Postgres schema")
	flag.IntVar(&cfg.generation, "generation", cfg.generation, "Generation / partition key")

	flag.BoolVar(&cfg.writeDB, "write-db", cfg.writeDB, "Write updates to DB (default false)")
	flag.BoolVar(&cfg.writeCSV, "write-csv", cfg.writeCSV, "Write CSV outputs")
	flag.BoolVar(&cfg.freshCSV, "fresh-csv", cfg.freshCSV, "Truncate CSV outputs before writing")
	flag.StringVar(&cfg.out, "out", cfg.out, "Output CSV path")
	flag.StringVar(&cfg.historyOut, "history-out", cfg.historyOut, "Price history CSV path")

	flag.IntVar(&cfg.scanSinceDays, "scan-since-days", cfg.scanSinceDays, "reverse: last_seen within N days (0=all)")
	flag.IntVar(&cfg.maxProbe, "max-probe", cfg.maxProbe, "reverse: cap candidates (0=all)")

	flag.IntVar(&cfg.workers, "workers", cfg.workers, "Concurrent workers")
	flag.IntVar(&cfg.rps, "rps", cfg.rps, "Initial/max RPS target")
	flag.BoolVar(&cfg.verbose, "verbose", cfg.verbose, "Verbose per-row logs")

	flag.IntVar(&cfg.throttleSleepMs, "throttle-sleep-ms", cfg.throttleSleepMs, "Fallback sleep on 429/403/408/5xx (ms) when Retry-After missing")
	flag.IntVar(&cfg.retryMax, "retry-max", cfg.retryMax, "Max retry attempts per request on throttle/5xx")
	flag.IntVar(&cfg.jitterMs, "jitter-ms", cfg.jitterMs, "Jitter added to waits (ms)")

	flag.Float64Var(&cfg.minRPS, "min-rps", cfg.minRPS, "Minimum RPS when throttled")
	flag.Float64Var(&cfg.maxRPS, "max-rps", cfg.maxRPS, "Hard ceiling RPS (0 => use --rps)")
	flag.Float64Var(&cfg.stepUpRPS, "step-up-rps", cfg.stepUpRPS, "Additive step-up RPS when stable (per second)")
	flag.Float64Var(&cfg.downMult, "down-mult", cfg.downMult, "Multiplicative decrease on throttle (0.0–1.0)")
	flag.Float64Var(&cfg.burstFactor, "burst-factor", cfg.burstFactor, "Burst capacity = burstFactor * current RPS")

	flag.IntVar(&cfg.testHours, "test-hours", cfg.testHours, "test: center at now()-H hours")
	flag.IntVar(&cfg.testBand, "test-band", cfg.testBand, "test: ± hours window around test-hours")
	flag.IntVar(&cfg.testLimit, "test-limit", cfg.testLimit, "test: number of rows")

	flag.IntVar(&cfg.repairFallbackDays, "repair-fallback-days", cfg.repairFallbackDays, "repair: look back N days")
	flag.IntVar(&cfg.repairThresholdSec, "repair-threshold-sec", cfg.repairThresholdSec, "repair: |observed_at - sold_date| ≤ N sec")
	flag.IntVar(&cfg.repairLimit, "repair-limit", cfg.repairLimit, "repair: LIMIT (0=all)")
	flag.StringVar(&cfg.onlyIDs, "only-ids", cfg.onlyIDs, "repair: only these listing IDs (csv)")

	flag.IntVar(&cfg.auditDays, "audit-days", cfg.auditDays, "audit-status: sold rows in last N days")
	flag.IntVar(&cfg.auditLimit, "audit-limit", cfg.auditLimit, "audit-status: LIMIT (0=all)")
	flag.StringVar(&cfg.auditOnlyIDs, "audit-only-ids", cfg.auditOnlyIDs, "audit-status: only these listing IDs (csv)")

	flag.Parse()

	// Derived / sanity
	if cfg.writeCSV && cfg.out == "" && cfg.historyOut == "" {
		fmt.Fprintln(os.Stderr, "[args] --write-csv set but no --out/--history-out")
		os.Exit(2)
	}
	if cfg.workers <= 0 {
		cfg.workers = 1
	}
	if cfg.rps <= 0 {
		cfg.rps = 1
	}
	if cfg.maxRPS <= 0 {
		cfg.maxRPS = float64(cfg.rps)
	}
	if cfg.minRPS <= 0.1 {
		cfg.minRPS = 0.1
	}
	if cfg.downMult <= 0.1 || cfg.downMult >= 1.0 {
		cfg.downMult = 0.6
	}
	if cfg.burstFactor < 1.0 {
		cfg.burstFactor = 1.0
	}

	cfg.adapter = strings.ToLower(strings.TrimSpace(cfg.adapter))
	cfg.mode = strings.ToLower(strings.TrimSpace(cfg.mode))

	return cfg
}

/* ========================= HTTP + throttling ========================= */

const (
	connectTimeout   = 4 * time.Second
	headerTimeout    = 15 * time.Second
	idleConnTimeout  = 90 * time.Second
	maxBodyBytes     = 8 << 20
	defaultHTTPBurst = 512
)

type HTTPClient struct{ Client *http.Client }

func newHTTPClient(maxPerHost int) *HTTPClient {
	tr := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		MaxConnsPerHost:       maxPerHost,
		MaxIdleConns:          defaultHTTPBurst,
		MaxIdleConnsPerHost:   maxPerHost,
		IdleConnTimeout:       idleConnTimeout,
		TLSHandshakeTimeout:   connectTimeout,
		ResponseHeaderTimeout: headerTimeout,
		ExpectContinueTimeout: 1 * time.Second,
	}
	return &HTTPClient{Client: &http.Client{Transport: tr}}
}

func setJSONHeaders(req *http.Request, userAgent string) {
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("Accept", "application/json")
}

type dynLimiter struct {
	mu sync.Mutex

	curRPS      float64
	minRPS      float64
	maxRPS      float64
	stepUpRPS   float64
	downMult    float64
	burstFactor float64

	tokens      float64
	lastRefill  time.Time
	lastPenalty time.Time

	cooldownUntil time.Time
	jitterMs      int
}

func newDynLimiter(minRPS, maxRPS, stepUp, downMult, burstFactor float64, jitterMs int) *dynLimiter {
	now := time.Now()
	if maxRPS < minRPS {
		maxRPS = minRPS
	}
	return &dynLimiter{
		curRPS:      minRPS,
		minRPS:      minRPS,
		maxRPS:      maxRPS,
		stepUpRPS:   stepUp,
		downMult:    downMult,
		burstFactor: burstFactor,
		tokens:      minRPS * burstFactor,
		lastRefill:  now,
		jitterMs:    jitterMs,
	}
}

func (d *dynLimiter) burstCap() float64 { return d.curRPS * d.burstFactor }

func (d *dynLimiter) refill(now time.Time) {
	elapsed := now.Sub(d.lastRefill).Seconds()
	if elapsed <= 0 {
		return
	}
	d.tokens = math.Min(d.burstCap(), d.tokens+elapsed*d.curRPS)
	d.lastRefill = now

	// passive additive increase when stable
	if now.Sub(d.lastPenalty) > 5*time.Second && d.curRPS < d.maxRPS {
		increment := d.stepUpRPS * elapsed
		d.curRPS = math.Min(d.maxRPS, d.curRPS+increment)
		d.tokens = math.Min(d.burstCap(), d.tokens)
	}
}

func (d *dynLimiter) Take(ctx context.Context) bool {
	for {
		d.mu.Lock()
		now := time.Now()

		if now.Before(d.cooldownUntil) {
			sleep := time.Until(d.cooldownUntil)
			if d.jitterMs > 0 {
				sleep += time.Duration(rand.Intn(d.jitterMs)) * time.Millisecond
			}
			d.mu.Unlock()
			select {
			case <-time.After(sleep):
			case <-ctx.Done():
				return false
			}
			continue
		}

		d.refill(now)
		if d.tokens >= 1.0 {
			d.tokens -= 1.0
			sleep := time.Duration(rand.Intn(d.jitterMs+1)) * time.Millisecond
			d.mu.Unlock()
			if sleep > 0 {
				select {
				case <-time.After(sleep):
				case <-ctx.Done():
					return false
				}
			}
			return true
		}

		need := 1.0 - d.tokens
		wait := time.Duration(need/d.curRPS*float64(time.Second)) + time.Duration(rand.Intn(d.jitterMs+1))*time.Millisecond
		d.mu.Unlock()

		select {
		case <-time.After(wait):
		case <-ctx.Done():
			return false
		}
	}
}

func parseRetryAfter(h http.Header) time.Duration {
	v := strings.TrimSpace(h.Get("Retry-After"))
	if v == "" {
		return 0
	}
	if n, err := strconv.Atoi(v); err == nil && n > 0 {
		return time.Duration(n) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		d := time.Until(t)
		if d > 0 {
			return d
		}
	}
	return 0
}

func (d *dynLimiter) Penalize(retryAfter time.Duration) {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.lastPenalty = time.Now()
	d.curRPS = math.Max(d.minRPS, d.curRPS*d.downMult)
	d.tokens = math.Min(d.tokens, d.burstCap())

	if retryAfter > 0 {
		if until := time.Now().Add(retryAfter); until.After(d.cooldownUntil) {
			d.cooldownUntil = until
		}
	}
}

func (d *dynLimiter) Reward() {
	// handled by refill()
}

func smartGET(
	ctx context.Context,
	hc *HTTPClient,
	url string,
	userAgent string,
	lim *dynLimiter,
	retryMax int,
	fallbackThrottle time.Duration,
	verbose bool,
) ([]byte, int, error) {
	var lastBody []byte
	var lastCode int

	for attempt := 0; attempt <= retryMax; attempt++ {
		if !lim.Take(ctx) {
			return nil, 0, ctx.Err()
		}
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		setJSONHeaders(req, userAgent)

		resp, err := hc.Client.Do(req)
		if err != nil {
			lastCode, lastBody = 0, nil
			lim.Penalize(500 * time.Millisecond)
			if attempt < retryMax {
				continue
			}
			return nil, 0, err
		}

		body, _ := io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
		resp.Body.Close()
		code := resp.StatusCode
		lastBody, lastCode = body, code

		switch {
		case code >= 200 && code < 300:
			lim.Reward()
			return body, code, nil
		case code == 404 || code == 410:
			return body, code, nil
		case code == 429 || code == 403 || code == 408 || (code >= 500 && code <= 599):
			ra := parseRetryAfter(resp.Header)
			if ra == 0 {
				ra = fallbackThrottle
			}
			lim.Penalize(ra)
			backoff := ra + time.Duration(attempt*attempt)*250*time.Millisecond + time.Duration(rand.Intn(151))*time.Millisecond
			if verbose {
				fmt.Printf("[throttle] http=%d retry_after=%s attempt=%d\n", code, ra, attempt)
			}
			if attempt < retryMax {
				select {
				case <-time.After(backoff):
					continue
				case <-ctx.Done():
					return body, code, ctx.Err()
				}
			}
			return body, code, nil
		default:
			return body, code, nil
		}
	}
	return lastBody, lastCode, nil
}

/* ========================= Marketplace adapter interface ========================= */

// SearchParams is intentionally minimal in the template.
type SearchParams struct {
	Query string
	Limit int
}

type ListingSummary struct {
	ListingID int64
	URL       string
}

type ListingSnapshot struct {
	ListingID   int64
	URL         string
	HTTPStatus  int
	ObservedAt  time.Time
	Status      string // live | sold | removed | unknown
	Price       int    // current/observed price (may be 0 if unknown)
	SoldPrice   int    // optional; if 0, Price may represent last visible price at sale time
	SoldAt      *time.Time
	Inactive    *bool // nil => not provided by adapter
	Bidding     *bool // nil => not provided by adapter
	Evidence    string
	RawMetadata json.RawMessage // optional passthrough for debugging; keep empty in public template
}

// MarketplaceAdapter abstracts all marketplace-specific logic.
// Public template adapters:
//   - mock: deterministic synthetic data, no network calls
//   - http-json: fetches JSON from MARKETPLACE_BASE_URL; expects a simple API shape
type MarketplaceAdapter interface {
	Name() string
	FetchListing(ctx context.Context, listingID int64, listingURL string) (ListingSnapshot, error)
	SearchListings(ctx context.Context, params SearchParams) ([]ListingSummary, error)
	ParsePayload(raw []byte, httpStatus int) (ListingSnapshot, error)
}

/* ========================= Adapter: mock (default) ========================= */

type MockAdapter struct{}

func (a *MockAdapter) Name() string { return "mock" }

func (a *MockAdapter) SearchListings(ctx context.Context, params SearchParams) ([]ListingSummary, error) {
	_ = ctx
	limit := params.Limit
	if limit <= 0 || limit > 25 {
		limit = 25
	}
	out := make([]ListingSummary, 0, limit)
	for i := 0; i < limit; i++ {
		id := int64(100000 + i)
		out = append(out, ListingSummary{
			ListingID: id,
			URL:       fmt.Sprintf("https://marketplace.example/listing/%d", id),
		})
	}
	return out, nil
}

func (a *MockAdapter) ParsePayload(raw []byte, httpStatus int) (ListingSnapshot, error) {
	_ = raw
	return ListingSnapshot{HTTPStatus: httpStatus, Status: "unknown", Evidence: "mock_parse_noop"}, nil
}

// FetchListing returns deterministic synthetic states derived from listingID.
// This enables end-to-end runs without external dependencies.
func (a *MockAdapter) FetchListing(ctx context.Context, listingID int64, listingURL string) (ListingSnapshot, error) {
	_ = ctx

	now := time.Now().UTC()

	// Stable pseudo-randomness from listingID
	h := sha256.Sum256([]byte(strconv.FormatInt(listingID, 10)))
	seed := int64(0)
	for i := 0; i < 8; i++ {
		seed = (seed << 8) | int64(h[i])
	}
	r := rand.New(rand.NewSource(seed))

	status := "live"
	httpStatus := 200

	switch {
	case listingID%17 == 0:
		status = "removed"
		httpStatus = 404
	case listingID%11 == 0:
		status = "sold"
		httpStatus = 200
	}

	price := 0
	if status == "live" {
		price = 1000 + int(listingID%5000)
	} else if status == "sold" {
		price = 900 + int(listingID%4500)
	}

	var soldAt *time.Time
	if status == "sold" {
		t := now.Add(-time.Duration(1+int(listingID%72)) * time.Hour)
		soldAt = &t
	}

	// Optional UI/state flags.
	var inactive *bool
	if r.Intn(10) == 0 {
		v := true
		inactive = &v
	} else if r.Intn(10) == 1 {
		v := false
		inactive = &v
	}

	var bidding *bool
	if r.Intn(15) == 0 {
		v := true
		bidding = &v
	} else if r.Intn(15) == 1 {
		v := false
		bidding = &v
	}

	return ListingSnapshot{
		ListingID:  listingID,
		URL:        listingURL,
		HTTPStatus: httpStatus,
		ObservedAt: now,
		Status:     status,
		Price:      price,
		SoldPrice:  0,
		SoldAt:     soldAt,
		Inactive:   inactive,
		Bidding:    bidding,
		Evidence:   "mock_deterministic",
	}, nil
}

/* ========================= Adapter: http-json (optional) ========================= */

type HTTPJSONAdapter struct {
	baseURL   string
	userAgent string
	hc        *HTTPClient
	lim       *dynLimiter
	cfg       config
}

func (a *HTTPJSONAdapter) Name() string { return "http-json" }

func (a *HTTPJSONAdapter) SearchListings(ctx context.Context, params SearchParams) ([]ListingSummary, error) {
	_ = ctx
	_ = params
	return nil, errors.New("http-json adapter: SearchListings not implemented in template")
}

type listingAPIResponse struct {
	ListingID int64  `json:"listing_id"`
	URL       string `json:"url,omitempty"`
	Status    string `json:"status"`
	Price     int    `json:"price,omitempty"`
	SoldPrice int    `json:"sold_price,omitempty"`
	SoldAt    string `json:"sold_at,omitempty"`

	IsInactive *bool `json:"is_inactive,omitempty"`
	IsBidding  *bool `json:"is_bidding,omitempty"`

	Evidence    string          `json:"evidence,omitempty"`
	RawMetadata json.RawMessage `json:"raw_metadata,omitempty"`
}

func (a *HTTPJSONAdapter) ParsePayload(raw []byte, httpStatus int) (ListingSnapshot, error) {
	if httpStatus == 404 || httpStatus == 410 {
		return ListingSnapshot{HTTPStatus: httpStatus, Status: "removed", Evidence: fmt.Sprintf("http_%d", httpStatus)}, nil
	}
	if httpStatus < 200 || httpStatus >= 300 {
		return ListingSnapshot{HTTPStatus: httpStatus, Status: "unknown", Evidence: fmt.Sprintf("http_%d", httpStatus)}, nil
	}

	var resp listingAPIResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return ListingSnapshot{HTTPStatus: httpStatus, Status: "unknown", Evidence: "json_unmarshal_failed"}, err
	}

	var soldAt *time.Time
	if strings.TrimSpace(resp.SoldAt) != "" {
		if t, err := time.Parse(time.RFC3339, strings.TrimSpace(resp.SoldAt)); err == nil {
			tt := t.UTC()
			soldAt = &tt
		}
	}

	return ListingSnapshot{
		ListingID:   resp.ListingID,
		URL:         resp.URL,
		HTTPStatus:  httpStatus,
		ObservedAt:  time.Now().UTC(),
		Status:      strings.ToLower(strings.TrimSpace(resp.Status)),
		Price:       resp.Price,
		SoldPrice:   resp.SoldPrice,
		SoldAt:      soldAt,
		Inactive:    resp.IsInactive,
		Bidding:     resp.IsBidding,
		Evidence:    strings.TrimSpace(resp.Evidence),
		RawMetadata: resp.RawMetadata,
	}, nil
}

func (a *HTTPJSONAdapter) FetchListing(ctx context.Context, listingID int64, listingURL string) (ListingSnapshot, error) {
	_ = listingURL // prefer stable ID endpoint in template

	base := strings.TrimRight(a.baseURL, "/")
	url := fmt.Sprintf("%s/api/listings/%d", base, listingID)

	body, status, err := smartGET(
		ctx,
		a.hc,
		url,
		a.userAgent,
		a.lim,
		a.cfg.retryMax,
		time.Duration(a.cfg.throttleSleepMs)*time.Millisecond,
		a.cfg.verbose,
	)
	if err != nil {
		return ListingSnapshot{ListingID: listingID, HTTPStatus: 0, Status: "unknown", Evidence: "request_error"}, err
	}

	snap, perr := a.ParsePayload(body, status)
	if snap.ListingID == 0 {
		snap.ListingID = listingID
	}
	if snap.URL == "" {
		snap.URL = url
	}
	snap.HTTPStatus = status
	if snap.ObservedAt.IsZero() {
		snap.ObservedAt = time.Now().UTC()
	}
	return snap, perr
}

func buildAdapter(cfg config, hc *HTTPClient, lim *dynLimiter) (MarketplaceAdapter, error) {
	switch cfg.adapter {
	case "mock":
		return &MockAdapter{}, nil
	case "http-json":
		if strings.TrimSpace(cfg.marketplaceBaseURL) == "" {
			return nil, errors.New("http-json adapter requires MARKETPLACE_BASE_URL / --marketplace-base-url")
		}
		return &HTTPJSONAdapter{
			baseURL:   cfg.marketplaceBaseURL,
			userAgent: cfg.userAgent,
			hc:        hc,
			lim:       lim,
			cfg:       cfg,
		}, nil
	default:
		return nil, fmt.Errorf("unknown adapter: %q (expected mock|http-json)", cfg.adapter)
	}
}

/* ========================= Decision model ========================= */

type Decision struct {
	Status string // live | sold | removed

	// Price is the observed price written to price_history for this scan.
	// For SOLD decisions, Price (if >0) is also written to listings.sold_price.
	//
	// HARD RULE (template): NEVER overwrite listings.price from probe results.
	// listings.price is assumed to be the first-seen price.
	Price       int
	PriceSource string
	SoldDate    time.Time
	Evidence    string

	// UI/state flags: nil => no update
	IsInactive         *bool
	InactiveEvidence   string
	InactiveMetaEdited *time.Time

	IsBidding       *bool
	BiddingEvidence string
}

func decisionFromSnapshot(s ListingSnapshot, lastPrice int) Decision {
	// Terminal removed (404/410)
	if s.HTTPStatus == 404 || s.HTTPStatus == 410 || s.Status == "removed" {
		price := lastPrice
		src := "db_fallback"
		if price <= 0 {
			src = "none"
		}
		return Decision{
			Status:      "removed",
			Price:       price,
			PriceSource: src,
			Evidence:    firstNonEmpty(s.Evidence, fmt.Sprintf("http_%d", s.HTTPStatus)),
			IsInactive:  s.Inactive,
			IsBidding:   s.Bidding,
		}
	}

	// Sold requires an authoritative timestamp if available.
	if s.Status == "sold" && s.SoldAt != nil && !s.SoldAt.IsZero() {
		price := s.SoldPrice
		src := "adapter_sold_price"
		if price <= 0 {
			price = s.Price
			src = "adapter_price_as_sold_price"
		}
		if price <= 0 && lastPrice > 0 {
			price = lastPrice
			src = "db_fallback"
		}
		return Decision{
			Status:           "sold",
			Price:            price,
			PriceSource:      src,
			SoldDate:         s.SoldAt.UTC(),
			Evidence:         firstNonEmpty(s.Evidence, "adapter_sold"),
			IsInactive:       s.Inactive,
			InactiveEvidence: firstNonEmpty(s.Evidence, "adapter"),
			IsBidding:        s.Bidding,
			BiddingEvidence:  firstNonEmpty(s.Evidence, "adapter"),
		}
	}

	// Live/unknown -> treat as live and keep last known price if missing.
	price := s.Price
	src := "adapter"
	if price <= 0 {
		price = lastPrice
		src = "db_fallback"
		if price <= 0 {
			src = "none"
		}
	}

	return Decision{
		Status:           "live",
		Price:            price,
		PriceSource:      src,
		Evidence:         firstNonEmpty(s.Evidence, "adapter_live"),
		IsInactive:       s.Inactive,
		InactiveEvidence: firstNonEmpty(s.Evidence, "adapter"),
		IsBidding:        s.Bidding,
		BiddingEvidence:  firstNonEmpty(s.Evidence, "adapter"),
	}
}

func firstNonEmpty(v ...string) string {
	for _, s := range v {
		if strings.TrimSpace(s) != "" {
			return s
		}
	}
	return ""
}

/* ========================= DB / CSV helpers ========================= */

func mustOpenPool(ctx context.Context, dsn string, maxConns int32) *pgxpool.Pool {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		fmt.Fprintln(os.Stderr, "pg-dsn parse:", err)
		os.Exit(2)
	}
	if maxConns <= 0 {
		maxConns = 4
	}
	cfg.MaxConns = maxConns

	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, "pg connect:", err)
		os.Exit(2)
	}
	return pool
}

func ensureCSVHeader(path string, header []string) error {
	if path == "" {
		return nil
	}
	needHeader := false
	if _, err := os.Stat(path); err != nil {
		needHeader = true
	} else {
		if fi, _ := os.Stat(path); fi != nil && fi.Size() == 0 {
			needHeader = true
		}
	}
	if !needHeader {
		return nil
	}

	_ = os.MkdirAll(filepath.Dir(path), 0755)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	// UTF-8 BOM for spreadsheet compatibility
	_, _ = f.Write([]byte{0xEF, 0xBB, 0xBF})

	w := csv.NewWriter(f)
	if err := w.Write(header); err != nil {
		return err
	}
	w.Flush()
	return w.Error()
}

func appendCSV(path string, rows [][]string) error {
	if path == "" || len(rows) == 0 {
		return nil
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	bufw := bufio.NewWriterSize(f, 1<<20)
	w := csv.NewWriter(bufw)
	for _, r := range rows {
		if err := w.Write(r); err != nil {
			return err
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		return err
	}
	if err := bufw.Flush(); err != nil {
		return err
	}
	return f.Sync()
}

// Postgres-ish timestamptz for CSV
const pgTS = "2006-01-02 15:04:05.000000-07"

/* ========================= Domain structs & loaders ========================= */

type Latest struct {
	ListingID int64
	URL       string
	Status    string

	Price int

	FirstSeen  *time.Time
	LastSeen   *time.Time
	EditedDate *time.Time

	Title     *string
	Desc      *string
	City      *string
	Postal    *string
	Storage   *string
	Condition *string
	Model     *string

	IsInactive bool
	IsBidding  bool
}

func loadLatestLive(ctx context.Context, pool *pgxpool.Pool, schema string, gen, sinceDays, maxRows int) ([]Latest, error) {
	args := []any{gen}
	idx := 2

	sb := strings.Builder{}
	sb.WriteString(fmt.Sprintf(
		`SELECT listing_id, url, status, COALESCE(price,0) AS price, first_seen, last_seen, edited_date,
		        title, description, location_city, postal_code, storage, condition, model,
		        COALESCE(is_inactive,false) AS is_inactive,
		        COALESCE(is_bidding,false)  AS is_bidding
		   FROM "%s".listings
		  WHERE generation=$1 AND status='live'`,
		schema,
	))

	if sinceDays > 0 {
		sb.WriteString(fmt.Sprintf(" AND last_seen >= now() - ($%d::int) * interval '1 day'", idx))
		args = append(args, sinceDays)
		idx++
	}

	sb.WriteString(" ORDER BY last_seen DESC")
	if maxRows > 0 {
		sb.WriteString(fmt.Sprintf(" LIMIT %d", maxRows))
	}

	rows, err := pool.Query(ctx, sb.String(), args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]Latest, 0, 8192)
	for rows.Next() {
		var l Latest
		if err := rows.Scan(
			&l.ListingID, &l.URL, &l.Status, &l.Price,
			&l.FirstSeen, &l.LastSeen, &l.EditedDate,
			&l.Title, &l.Desc, &l.City, &l.Postal,
			&l.Storage, &l.Condition, &l.Model,
			&l.IsInactive,
			&l.IsBidding,
		); err != nil {
			return nil, err
		}
		out = append(out, l)
	}
	return out, rows.Err()
}

/* ========================= DB writes ========================= */

func updateMain(ctx context.Context, pool *pgxpool.Pool, schema string, gen int, listingID int64, dec Decision) error {
	if pool == nil || dec.Status == "" || dec.Status == "live" {
		return nil
	}

	switch dec.Status {
	case "sold":
		if dec.SoldDate.IsZero() {
			return nil
		}
		if dec.Price > 0 {
			_, err := pool.Exec(ctx, fmt.Sprintf(
				`UPDATE "%s".listings
				    SET status='sold', sold_price=$1, sold_date=$2, last_seen=now()
				  WHERE generation=$3 AND listing_id=$4`,
				schema,
			), dec.Price, dec.SoldDate.UTC(), gen, listingID)
			return err
		}
		_, err := pool.Exec(ctx, fmt.Sprintf(
			`UPDATE "%s".listings
			    SET status='sold', sold_date=$1, last_seen=now()
			  WHERE generation=$2 AND listing_id=$3`,
			schema,
		), dec.SoldDate.UTC(), gen, listingID)
		return err

	case "removed":
		_, err := pool.Exec(ctx, fmt.Sprintf(
			`UPDATE "%s".listings
			    SET status='removed', last_seen=now()
			  WHERE generation=$1 AND listing_id=$2 AND status <> 'removed'`,
			schema,
		), gen, listingID)
		return err

	default:
		return nil
	}
}

// is_inactive is a UI/state flag. Persist it into dedicated columns.
// Safe to call even when status stays 'live'.
func updateInactive(ctx context.Context, pool *pgxpool.Pool, schema string, gen int, listingID int64, isInactive bool, evidence string, metaEdited *time.Time) error {
	if pool == nil {
		return nil
	}
	if isInactive {
		_, err := pool.Exec(ctx, fmt.Sprintf(
			`UPDATE "%s".listings
			    SET is_inactive=true,
			        inactive_observed_at=now(),
			        inactive_evidence=$1,
			        inactive_meta_edited_at = COALESCE($2, inactive_meta_edited_at)
			  WHERE generation=$3 AND listing_id=$4`,
			schema,
		), evidence, metaEdited, gen, listingID)
		return err
	}

	// Clear only if previously inactive.
	_, err := pool.Exec(ctx, fmt.Sprintf(
		`UPDATE "%s".listings
		    SET is_inactive=false,
		        inactive_observed_at=NULL,
		        inactive_evidence=NULL,
		        inactive_meta_edited_at=NULL
		  WHERE generation=$1 AND listing_id=$2 AND is_inactive=true`,
		schema,
	), gen, listingID)
	return err
}

// is_bidding is a UI/state flag. Persist it into dedicated columns.
// Safe to call even when status stays 'live'.
func updateBidding(ctx context.Context, pool *pgxpool.Pool, schema string, gen int, listingID int64, isBidding bool, evidenceJSON string) error {
	if pool == nil {
		return nil
	}

	if isBidding {
		_, err := pool.Exec(ctx, fmt.Sprintf(`
UPDATE "%s".listings
   SET is_bidding=true,
       bidding_evidence=$1::jsonb
 WHERE generation=$2 AND listing_id=$3
   AND (is_bidding IS DISTINCT FROM true OR bidding_evidence IS DISTINCT FROM $1::jsonb)`, schema),
			nullJSON(evidenceJSON), gen, listingID,
		)
		return err
	}

	// Clear only if previously bidding.
	_, err := pool.Exec(ctx, fmt.Sprintf(`
UPDATE "%s".listings
   SET is_bidding=false,
       bidding_evidence=NULL
 WHERE generation=$1 AND listing_id=$2 AND is_bidding=true`, schema),
		gen, listingID,
	)
	return err
}

func nullJSON(s string) string {
	if strings.TrimSpace(s) == "" {
		return "null"
	}
	return s
}

// Append-only inactive state history (minimal):
// - Insert TRUE when we first observe inactive=true (or when it flips false->true).
// - Insert FALSE only when it flips true->false (to close the inactive window).
// - DO NOT insert baseline inactive=false rows.
func logInactiveEventIfChanged(
	ctx context.Context,
	pool *pgxpool.Pool,
	schema string,
	gen int,
	listingID int64,
	isInactive bool,
	metaEdited *time.Time,
	observedBy string,
	mainStatus string,
	evidence string,
	httpStatus int,
) error {
	if pool == nil {
		return nil
	}

	// Get last recorded state (if any)
	var lastVal bool
	hasLast := false
	err := pool.QueryRow(ctx, fmt.Sprintf(
		`SELECT is_inactive
		   FROM "%s".inactive_state_events
		  WHERE generation=$1 AND listing_id=$2
		  ORDER BY observed_at DESC, event_id DESC
		  LIMIT 1`,
		schema,
	), gen, listingID).Scan(&lastVal)
	if err == nil {
		hasLast = true
	}

	// If we have a last state and it's the same, do nothing.
	if hasLast && lastVal == isInactive {
		return nil
	}
	// If we have NO last state and current isInactive=false, skip (don't log baseline active).
	if !hasLast && !isInactive {
		return nil
	}

	_, err = pool.Exec(ctx, fmt.Sprintf(
		`INSERT INTO "%s".inactive_state_events
		     (generation, listing_id, observed_at, is_inactive, meta_edited_at, observed_by, main_status, evidence, http_status)
		 VALUES ($1,$2,now(),$3,$4,$5,$6,$7,$8)`,
		schema,
	), gen, listingID, isInactive, metaEdited, observedBy, mainStatus, evidence, httpStatus)
	return err
}

// lastNonZeroHistoryPrice returns the most recent non-zero price we have in price_history.
func lastNonZeroHistoryPrice(ctx context.Context, pool *pgxpool.Pool, schema string, gen int, listingID int64) (int, error) {
	var p int
	err := pool.QueryRow(ctx, fmt.Sprintf(
		`SELECT price
		   FROM "%s".price_history
		  WHERE generation=$1 AND listing_id=$2 AND price > 0
		  ORDER BY observed_at DESC
		  LIMIT 1`,
		schema,
	), gen, listingID).Scan(&p)
	return p, err
}

// upsertHistory writes an event snapshot using an hour+price dedupe constraint.
// NOTE: The constraint name is a template placeholder. Adjust to your schema if needed.
func upsertHistory(ctx context.Context, pool *pgxpool.Pool, schema string, gen int, listingID int64, observedAt time.Time, price int, status string) error {
	if pool == nil {
		return nil
	}

	_, err := pool.Exec(ctx, fmt.Sprintf(
		`INSERT INTO "%s".price_history AS ph (generation, listing_id, observed_at, price, status, source)
		 VALUES ($1,$2,$3,$4,$5,'reverse')
		 ON CONFLICT ON CONSTRAINT price_history_hour_dedupe DO UPDATE SET
		   status = CASE
		     WHEN ph.status = 'sold' THEN 'sold'
		     WHEN EXCLUDED.status = 'sold' THEN 'sold'
		     WHEN ph.source <> 'reverse' THEN ph.status
		     WHEN EXCLUDED.status = 'removed' AND ph.status <> 'sold' THEN 'removed'
		     ELSE ph.status
		   END,
		   price = CASE
		     WHEN EXCLUDED.status = 'sold' AND EXCLUDED.price > 0 THEN EXCLUDED.price
		     WHEN ph.source <> 'reverse' THEN ph.price
		     WHEN EXCLUDED.status <> 'removed'
		          AND ph.status NOT IN ('sold','removed')
		          AND EXCLUDED.price > 0
		          AND EXCLUDED.price <> ph.price
		       THEN EXCLUDED.price
		     ELSE ph.price
		   END,
		   observed_at = CASE
		     WHEN EXCLUDED.status = 'sold' THEN EXCLUDED.observed_at
		     WHEN ph.status = 'sold' THEN ph.observed_at
		     WHEN ph.source <> 'reverse' THEN ph.observed_at
		     ELSE GREATEST(ph.observed_at, EXCLUDED.observed_at)
		   END,
		   source = CASE
		     WHEN ph.source <> 'reverse' THEN ph.source
		     ELSE EXCLUDED.source
		   END`,
		schema,
	), gen, listingID, observedAt.UTC(), price, status)
	return err
}

// upsertHistoryDailyBaseline writes a once-per-UTC-day baseline snapshot for LIVE listings.
// observedAt MUST be a fixed daily bucket time in UTC (e.g., 00:05:00).
func upsertHistoryDailyBaseline(ctx context.Context, pool *pgxpool.Pool, schema string, gen int, listingID int64, dayBucket time.Time, price int, status string) error {
	if pool == nil {
		return nil
	}

	_, err := pool.Exec(ctx, fmt.Sprintf(
		`INSERT INTO "%s".price_history AS ph (generation, listing_id, observed_at, price, status, source)
		 VALUES ($1,$2,$3,$4,$5,'reverse')
		 ON CONFLICT (generation, listing_id, observed_at) DO UPDATE SET
		   price = CASE
		     WHEN EXCLUDED.price > 0 THEN EXCLUDED.price
		     ELSE ph.price
		   END,
		   status = CASE
		     WHEN ph.status = 'sold' THEN 'sold'
		     WHEN EXCLUDED.status = 'sold' THEN 'sold'
		     WHEN EXCLUDED.status = 'removed' AND ph.status <> 'sold' THEN 'removed'
		     ELSE EXCLUDED.status
		   END,
		   source = CASE
		     WHEN ph.source <> 'reverse' THEN ph.source
		     ELSE EXCLUDED.source
		   END`,
		schema,
	), gen, listingID, dayBucket.UTC(), price, status)
	return err
}

/* ========================= Prefetch helpers for sparse history ========================= */

type histLatest struct {
	ObservedAt time.Time
	Price      int
	Status     string
}

func prefetchLatestHistory(ctx context.Context, pool *pgxpool.Pool, schema string, gen int, listingIDs []int64) (map[int64]histLatest, error) {
	out := make(map[int64]histLatest, len(listingIDs))
	if pool == nil || len(listingIDs) == 0 {
		return out, nil
	}

	rows, err := pool.Query(ctx, fmt.Sprintf(
		`SELECT DISTINCT ON (listing_id) listing_id, observed_at, price, status
		   FROM "%s".price_history
		  WHERE generation=$1 AND listing_id = ANY($2::bigint[])
		  ORDER BY listing_id, observed_at DESC`,
		schema,
	), gen, listingIDs)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var id int64
		var t time.Time
		var p int
		var st string
		if err := rows.Scan(&id, &t, &p, &st); err != nil {
			return nil, err
		}
		out[id] = histLatest{ObservedAt: t.UTC(), Price: p, Status: st}
	}
	return out, rows.Err()
}

func prefetchLastNonZeroPrice(ctx context.Context, pool *pgxpool.Pool, schema string, gen int, listingIDs []int64) (map[int64]int, error) {
	out := make(map[int64]int, len(listingIDs))
	if pool == nil || len(listingIDs) == 0 {
		return out, nil
	}

	rows, err := pool.Query(ctx, fmt.Sprintf(
		`SELECT DISTINCT ON (listing_id) listing_id, price
		   FROM "%s".price_history
		  WHERE generation=$1 AND listing_id = ANY($2::bigint[]) AND price > 0
		  ORDER BY listing_id, observed_at DESC`,
		schema,
	), gen, listingIDs)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var id int64
		var p int
		if err := rows.Scan(&id, &p); err != nil {
			return nil, err
		}
		out[id] = p
	}
	return out, rows.Err()
}

func prefetchBaselineExists(ctx context.Context, pool *pgxpool.Pool, schema string, gen int, listingIDs []int64, dayBucket time.Time) (map[int64]bool, error) {
	out := make(map[int64]bool, len(listingIDs))
	if pool == nil || len(listingIDs) == 0 {
		return out, nil
	}

	rows, err := pool.Query(ctx, fmt.Sprintf(
		`SELECT listing_id
		   FROM "%s".price_history
		  WHERE generation=$1 AND listing_id = ANY($2::bigint[]) AND observed_at=$3`,
		schema,
	), gen, listingIDs, dayBucket.UTC())
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		out[id] = true
	}
	return out, rows.Err()
}

func isUniqueViolationOnConstraint(err error, constraint string) bool {
	var pgErr *pgconn.PgError
	if err == nil || !errors.As(err, &pgErr) {
		return false
	}
	return pgErr.Code == "23505" && pgErr.ConstraintName == constraint
}

/* ========================= Reverse runner (sparse history) ========================= */

type historyAction struct {
	kind       string // baseline | event
	observedAt time.Time
	price      int
	status     string
	reason     string
}

// dailyBucketUTC returns date_trunc('day', t at time zone 'UTC') + 00:05:00.
func dailyBucketUTC(t time.Time) time.Time {
	tt := t.UTC()
	return time.Date(tt.Year(), tt.Month(), tt.Day(), 0, 5, 0, 0, time.UTC)
}

func shouldUsePriceForChangeDetection(price int, src string) bool {
	if price <= 0 {
		return false
	}
	ls := strings.ToLower(src)
	if ls == "" || ls == "none" {
		return false
	}
	// "db_" sources are fallbacks and not safe for "price changed" events.
	if strings.HasPrefix(ls, "db_") {
		return false
	}
	return true
}

func reverseOnce(
	ctx context.Context,
	cfg config,
	pool *pgxpool.Pool,
	adapter MarketplaceAdapter,
) (cands, histOK, histErr, liveCnt, soldCnt, removedCnt, inactiveTrue, inactiveFalse, inactiveSeen int, durSec float64) {
	start := time.Now()

	// CSV headers
	reverseCSVHeader := []string{
		"title", "price", "sold_price", "sold_date", "url", "description", "edited_date", "location_city",
		"postal_code", "last_fetched", "status", "first_seen", "last_seen", "listing_id",
		"storage", "condition", "model",
	}
	historyHeader := []string{"generation", "listing_id", "observed_at", "price", "status", "source", "kind", "reason"}

	// Fresh CSV prep
	if cfg.writeCSV && cfg.freshCSV && cfg.out != "" {
		_ = os.Remove(cfg.out)
	}
	if cfg.writeCSV && cfg.freshCSV && cfg.historyOut != "" {
		_ = os.Remove(cfg.historyOut)
	}
	if cfg.writeCSV && cfg.out != "" {
		_ = ensureCSVHeader(cfg.out, reverseCSVHeader)
	}
	if cfg.writeCSV && cfg.historyOut != "" {
		_ = ensureCSVHeader(cfg.historyOut, historyHeader)
	}

	// Load live candidates
	latests, err := loadLatestLive(ctx, pool, cfg.pgSchema, cfg.generation, cfg.scanSinceDays, cfg.maxProbe)
	if err != nil {
		fmt.Fprintln(os.Stderr, "loadLatestLive:", err)
		os.Exit(2)
	}
	cands = len(latests)
	if cands == 0 {
		return cands, 0, 0, 0, 0, 0, 0, 0, 0, time.Since(start).Seconds()
	}

	// Precompute day bucket & prefetch history state in batch (NO per-row history reads).
	runTS := time.Now().UTC()
	dayBucket := dailyBucketUTC(runTS)

	ids := make([]int64, 0, len(latests))
	for _, l := range latests {
		ids = append(ids, l.ListingID)
	}

	lastHist := map[int64]histLatest{}
	lastNonZero := map[int64]int{}
	baselineExists := map[int64]bool{}

	needHistoryState := cfg.writeDB || (cfg.writeCSV && cfg.historyOut != "")
	if needHistoryState {
		if m, err := prefetchLatestHistory(ctx, pool, cfg.pgSchema, cfg.generation, ids); err == nil {
			lastHist = m
		} else if cfg.verbose {
			fmt.Fprintf(os.Stderr, "[prefetch] latest history failed: %v\n", err)
		}
		if m, err := prefetchLastNonZeroPrice(ctx, pool, cfg.pgSchema, cfg.generation, ids); err == nil {
			lastNonZero = m
		} else if cfg.verbose {
			fmt.Fprintf(os.Stderr, "[prefetch] last nonzero failed: %v\n", err)
		}
		if m, err := prefetchBaselineExists(ctx, pool, cfg.pgSchema, cfg.generation, ids, dayBucket); err == nil {
			baselineExists = m
		} else if cfg.verbose {
			fmt.Fprintf(os.Stderr, "[prefetch] baseline exists failed: %v\n", err)
		}
	}

	type outRow struct {
		row      []string
		histRows [][]string
	}

	results := make(chan outRow, max(128, cfg.workers*4))

	var live32, sold32, removed32 int32
	var histOK32, histErr32 int32
	var inactiveTrue32, inactiveFalse32, inactiveSeen32 int32

	var wg sync.WaitGroup
	workers := max(1, min(cfg.workers, len(latests)))
	wg.Add(workers)

	// startup jitter so workers don't burst at t=0
	if cfg.jitterMs > 0 {
		time.Sleep(time.Duration(rand.Intn(cfg.jitterMs)) * time.Millisecond)
	}

	errTag := func(err error) string {
		if err == nil {
			return "none"
		}
		msg := err.Error()
		msg = strings.ReplaceAll(msg, "\n", " ")
		if len(msg) > 180 {
			msg = msg[:180] + "..."
		}
		return fmt.Sprintf("%T:%s", err, msg)
	}

	for w := 0; w < workers; w++ {
		go func(shard int) {
			defer wg.Done()

			for i := shard; i < len(latests); i += workers {
				l := latests[i]

				snap, err := adapter.FetchListing(ctx, l.ListingID, l.URL)
				if err != nil && cfg.verbose {
					fmt.Printf("[probe_err] id=%d adapter=%s err=%s\n", l.ListingID, adapter.Name(), errTag(err))
				}

				dec := decisionFromSnapshot(snap, l.Price)

				// Track inactive updates if provided
				if dec.IsInactive != nil {
					atomic.AddInt32(&inactiveSeen32, 1)
					if *dec.IsInactive {
						atomic.AddInt32(&inactiveTrue32, 1)
					} else {
						atomic.AddInt32(&inactiveFalse32, 1)
					}
				}

				switch dec.Status {
				case "live":
					atomic.AddInt32(&live32, 1)
				case "sold":
					atomic.AddInt32(&sold32, 1)
				case "removed":
					atomic.AddInt32(&removed32, 1)
				}

				scanTS := time.Now().UTC()

				// --- DB updates ---
				var errMain, errInact, errInactLog, errBid error
				if cfg.writeDB {
					if dec.IsInactive != nil {
						errInact = updateInactive(ctx, pool, cfg.pgSchema, cfg.generation, l.ListingID, *dec.IsInactive, dec.InactiveEvidence, dec.InactiveMetaEdited)
						errInactLog = logInactiveEventIfChanged(ctx, pool, cfg.pgSchema, cfg.generation, l.ListingID, *dec.IsInactive, dec.InactiveMetaEdited, "reverse", "live", dec.InactiveEvidence, snap.HTTPStatus)
					}
					if dec.IsBidding != nil {
						// evidenceJSON is intentionally empty in template unless your adapter provides one
						errBid = updateBidding(ctx, pool, cfg.pgSchema, cfg.generation, l.ListingID, *dec.IsBidding, dec.BiddingEvidence)
					}
					if dec.Status != "live" {
						errMain = updateMain(ctx, pool, cfg.pgSchema, cfg.generation, l.ListingID, dec)
					}
				}

				// --- Sparse price_history writes ---
				var histRows [][]string
				if needHistoryState {
					actions := make([]historyAction, 0, 2)

					hasBaseline := baselineExists[l.ListingID]

					lastP := 0
					if p, ok := lastNonZero[l.ListingID]; ok {
						lastP = p
					}
					if lastP == 0 {
						if lh, ok := lastHist[l.ListingID]; ok && lh.Price > 0 {
							lastP = lh.Price
						}
					}

					priceNow := dec.Price
					priceNowReliable := shouldUsePriceForChangeDetection(priceNow, dec.PriceSource)

					// Bid-mode listings may have no asking price; do not emit "price change" events in that case.
					effectiveBidding := l.IsBidding
					if dec.IsBidding != nil {
						effectiveBidding = *dec.IsBidding
					}
					if effectiveBidding {
						priceNowReliable = false
					}

					// baseline price: prefer reliable current extraction; else last nonzero; else (as last resort) main-table price.
					baselinePrice := 0
					if priceNowReliable {
						baselinePrice = priceNow
					} else if lastP > 0 {
						baselinePrice = lastP
					} else if l.Price > 0 {
						baselinePrice = l.Price
					}

					// Terminal events always write
					if dec.Status == "sold" {
						obs := dec.SoldDate.UTC()
						if obs.IsZero() {
							obs = scanTS
						}
						actions = append(actions, historyAction{
							kind:       "event",
							observedAt: obs,
							price:      dec.Price,
							status:     "sold",
							reason:     "terminal_sold",
						})
					} else if dec.Status == "removed" {
						actions = append(actions, historyAction{
							kind:       "event",
							observedAt: scanTS,
							price:      baselinePrice,
							status:     "removed",
							reason:     "terminal_removed",
						})
					} else {
						// LIVE: baseline once per day + change events only.

						if !hasBaseline {
							actions = append(actions, historyAction{
								kind:       "baseline",
								observedAt: dayBucket,
								price:      baselinePrice,
								status:     "live",
								reason:     "daily_baseline_missing",
							})
						}

						priceChanged := false
						if priceNowReliable {
							if lastP == 0 {
								priceChanged = true
							} else if lastP > 0 && priceNow != lastP {
								priceChanged = true
							}
						}

						inactiveFlipped := false
						if dec.IsInactive != nil && *dec.IsInactive != l.IsInactive {
							inactiveFlipped = true
						}

						if priceChanged {
							actions = append(actions, historyAction{
								kind:       "event",
								observedAt: scanTS,
								price:      priceNow,
								status:     "live",
								reason:     fmt.Sprintf("price_changed:%d->%d", lastP, priceNow),
							})
						} else if inactiveFlipped {
							eventPrice := baselinePrice
							if priceNowReliable {
								eventPrice = priceNow
							}
							actions = append(actions, historyAction{
								kind:       "event",
								observedAt: scanTS,
								price:      eventPrice,
								status:     "live",
								reason:     fmt.Sprintf("inactive_flip:%t->%t", l.IsInactive, *dec.IsInactive),
							})
						}
					}

					for _, act := range actions {
						if cfg.writeDB {
							var herr error
							if act.kind == "baseline" {
								herr = upsertHistoryDailyBaseline(ctx, pool, cfg.pgSchema, cfg.generation, l.ListingID, act.observedAt, act.price, act.status)
								// If baseline collides with hour_dedupe constraint (rare), treat as satisfied.
								if herr != nil && isUniqueViolationOnConstraint(herr, "price_history_hour_dedupe") {
									herr = nil
								}
							} else {
								herr = upsertHistory(ctx, pool, cfg.pgSchema, cfg.generation, l.ListingID, act.observedAt, act.price, act.status)
							}
							if herr != nil {
								atomic.AddInt32(&histErr32, 1)
								if cfg.verbose {
									fmt.Printf("[hist_err] id=%d kind=%s reason=%s err=%s\n", l.ListingID, act.kind, act.reason, errTag(herr))
								}
							} else {
								atomic.AddInt32(&histOK32, 1)
							}
						}

						if cfg.writeCSV && cfg.historyOut != "" {
							histRows = append(histRows, []string{
								strconv.Itoa(cfg.generation),
								strconv.FormatInt(l.ListingID, 10),
								act.observedAt.UTC().Format(pgTS),
								strconv.Itoa(act.price),
								act.status,
								"reverse",
								act.kind,
								act.reason,
							})
						}
					}
				}

				if cfg.verbose {
					inactStr := "na"
					if dec.IsInactive != nil {
						if *dec.IsInactive {
							inactStr = "true"
						} else {
							inactStr = "false"
						}
					}
					bidStr := "na"
					if dec.IsBidding != nil {
						if *dec.IsBidding {
							bidStr = "true"
						} else {
							bidStr = "false"
						}
					}
					fmt.Printf("[reverse] ts=%s gen=%d id=%d http=%d status=%s price=%d src=%s ev=%s inactive=%s bidding=%s adapter=%s write_db=%t main_err=%s inact_err=%s inactlog_err=%s bid_err=%s\n",
						scanTS.Format(time.RFC3339),
						cfg.generation,
						l.ListingID,
						snap.HTTPStatus,
						dec.Status,
						dec.Price,
						dec.PriceSource,
						dec.Evidence,
						inactStr,
						bidStr,
						adapter.Name(),
						cfg.writeDB,
						errTag(errMain),
						errTag(errInact),
						errTag(errInactLog),
						errTag(errBid),
					)
				}

				// Main CSV row
				var row []string
				if cfg.writeCSV && cfg.out != "" {
					soldStr := ""
					soldPriceStr := ""
					if dec.Status == "sold" && !dec.SoldDate.IsZero() {
						soldStr = dec.SoldDate.UTC().Format(pgTS)
					}
					if dec.Status == "sold" && dec.Price > 0 {
						soldPriceStr = strconv.Itoa(dec.Price)
					}
					row = []string{
						derefStr(l.Title),
						intToStr(dec.Price),
						soldPriceStr,
						soldStr,
						l.URL,
						derefStr(l.Desc),
						timeStr(l.EditedDate),
						derefStr(l.City),
						derefStr(l.Postal),
						scanTS.UTC().Format(pgTS),
						dec.Status,
						timeStr(l.FirstSeen),
						timeStr(l.LastSeen),
						strconv.FormatInt(l.ListingID, 10),
						derefStr(l.Storage),
						derefStr(l.Condition),
						derefStr(l.Model),
					}
				}

				results <- outRow{row: row, histRows: histRows}
			}
		}(w)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	var toWrite [][]string
	var toHist [][]string
	for r := range results {
		if len(r.row) > 0 {
			toWrite = append(toWrite, r.row)
		}
		if len(r.histRows) > 0 {
			toHist = append(toHist, r.histRows...)
		}
	}

	if cfg.writeCSV && cfg.out != "" && len(toWrite) > 0 {
		_ = appendCSV(cfg.out, toWrite)
		if cfg.verbose {
			fmt.Printf("[reverse] wrote %d rows to %s\n", len(toWrite), cfg.out)
		}
	}
	if cfg.writeCSV && cfg.historyOut != "" && len(toHist) > 0 {
		_ = appendCSV(cfg.historyOut, toHist)
		if cfg.verbose {
			fmt.Printf("[reverse] wrote %d history rows to %s\n", len(toHist), cfg.historyOut)
		}
	}

	histOK = int(atomic.LoadInt32(&histOK32))
	histErr = int(atomic.LoadInt32(&histErr32))
	liveCnt = int(atomic.LoadInt32(&live32))
	soldCnt = int(atomic.LoadInt32(&sold32))
	removedCnt = int(atomic.LoadInt32(&removed32))
	inactiveTrue = int(atomic.LoadInt32(&inactiveTrue32))
	inactiveFalse = int(atomic.LoadInt32(&inactiveFalse32))
	inactiveSeen = int(atomic.LoadInt32(&inactiveSeen32))
	durSec = time.Since(start).Seconds()

	return
}

/* ========================= Audit-status (template) ========================= */

type auditRow struct {
	ListingID  int64
	URL        string
	StatusDB   string
	SoldDateDB *time.Time
	Generation int
}

func loadAuditCandidates(ctx context.Context, pool *pgxpool.Pool, schema string, gen, days, limit int, ids []int64) ([]auditRow, error) {
	args := []any{gen, days}
	idx := 3

	sb := strings.Builder{}
	sb.WriteString(fmt.Sprintf(
		`SELECT listing_id, url, status, sold_date, generation
		   FROM "%s".listings
		  WHERE generation=$1 AND status='sold' AND sold_date >= now() - ($2::int) * interval '1 day'`,
		schema,
	))

	if len(ids) > 0 {
		sb.WriteString(fmt.Sprintf(" AND listing_id = ANY($%d::bigint[])", idx))
		args = append(args, ids)
		idx++
	}

	sb.WriteString(" ORDER BY sold_date DESC")
	if limit > 0 {
		sb.WriteString(fmt.Sprintf(" LIMIT %d", limit))
	}

	rows, err := pool.Query(ctx, sb.String(), args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]auditRow, 0, 4096)
	for rows.Next() {
		var r auditRow
		if err := rows.Scan(&r.ListingID, &r.URL, &r.StatusDB, &r.SoldDateDB, &r.Generation); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func runAuditStatus(ctx context.Context, cfg config, pool *pgxpool.Pool, adapter MarketplaceAdapter) error {
	if pool == nil {
		return errors.New("audit-status requires --pg-dsn / PG_DSN")
	}
	start := time.Now()

	if cfg.writeCSV && cfg.freshCSV && cfg.out != "" {
		_ = os.Remove(cfg.out)
	}
	if cfg.writeCSV && cfg.out != "" {
		_ = ensureCSVHeader(cfg.out, []string{
			"listing_id", "generation", "sold_date_db", "status_db", "inactive_detected", "decision", "http_status", "url", "evidence",
		})
	}

	var ids []int64
	if strings.TrimSpace(cfg.auditOnlyIDs) != "" {
		for _, t := range strings.Split(cfg.auditOnlyIDs, ",") {
			if v, err := strconv.ParseInt(strings.TrimSpace(t), 10, 64); err == nil {
				ids = append(ids, v)
			}
		}
	}

	cands, err := loadAuditCandidates(ctx, pool, cfg.pgSchema, cfg.generation, cfg.auditDays, cfg.auditLimit, ids)
	if err != nil {
		return err
	}
	if len(cands) == 0 {
		fmt.Println("audit-status: no candidates")
		return nil
	}

	type rec struct{ r []string }
	outc := make(chan rec, len(cands))

	inspected := int64(0)
	applied := int64(0)

	var wg sync.WaitGroup
	wk := max(1, min(cfg.workers, len(cands)))
	wg.Add(wk)

	for w := 0; w < wk; w++ {
		go func(shard int) {
			defer wg.Done()
			for i := shard; i < len(cands); i += wk {
				c := cands[i]
				snap, _ := adapter.FetchListing(ctx, c.ListingID, c.URL)

				atomic.AddInt64(&inspected, 1)

				decision := "keep"
				inactiveDetected := "unknown"

				if snap.Inactive != nil {
					if *snap.Inactive {
						inactiveDetected = "true"
						decision = "set_inactive"
						if cfg.writeDB {
							_ = updateInactive(ctx, pool, cfg.pgSchema, c.Generation, c.ListingID, true, snap.Evidence, nil)
							atomic.AddInt64(&applied, 1)
						}
					} else {
						inactiveDetected = "false"
						decision = "clear_inactive"
						if cfg.writeDB {
							_ = updateInactive(ctx, pool, cfg.pgSchema, c.Generation, c.ListingID, false, "", nil)
							atomic.AddInt64(&applied, 1)
						}
					}
				}

				if cfg.writeCSV && cfg.out != "" {
					soldStr := ""
					if c.SoldDateDB != nil {
						soldStr = c.SoldDateDB.UTC().Format(pgTS)
					}
					outc <- rec{r: []string{
						strconv.FormatInt(c.ListingID, 10),
						strconv.Itoa(c.Generation),
						soldStr,
						c.StatusDB,
						inactiveDetected,
						decision,
						strconv.Itoa(snap.HTTPStatus),
						c.URL,
						firstNonEmpty(snap.Evidence, "adapter"),
					}}
				}
			}
		}(w)
	}

	go func() {
		wg.Wait()
		close(outc)
	}()

	var rows [][]string
	for r := range outc {
		rows = append(rows, r.r)
	}
	if cfg.writeCSV && cfg.out != "" && len(rows) > 0 {
		_ = appendCSV(cfg.out, rows)
	}

	dur := time.Since(start).Seconds()
	fmt.Printf(
		"audit-status: gen=%d inspected=%d applied=%d dur=%.2fs adapter=%s\n",
		cfg.generation,
		inspected,
		applied,
		dur,
		adapter.Name(),
	)
	return nil
}

/* ========================= Test mode (template) ========================= */

type SoldTest struct {
	ListingID int64
	URL       string
	SoldDate  time.Time
}

func loadSoldForTest(ctx context.Context, pool *pgxpool.Pool, schema string, gen, hoursCenter, band, limit int) ([]SoldTest, error) {
	end := time.Now().Add(-time.Duration(max(0, hoursCenter-band)) * time.Hour)
	start := time.Now().Add(-time.Duration(hoursCenter+band) * time.Hour)
	if start.After(end) {
		start, end = end, start
	}

	q := fmt.Sprintf(
		`SELECT listing_id, url, sold_date
		   FROM "%s".listings
		  WHERE generation=$1 AND status='sold' AND sold_date BETWEEN $2 AND $3
		  ORDER BY sold_date DESC
		  LIMIT $4`,
		schema,
	)
	rows, err := pool.Query(ctx, q, gen, start.UTC(), end.UTC(), max(1, limit))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]SoldTest, 0, limit)
	for rows.Next() {
		var s SoldTest
		if err := rows.Scan(&s.ListingID, &s.URL, &s.SoldDate); err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

func runTest(ctx context.Context, cfg config, pool *pgxpool.Pool, adapter MarketplaceAdapter) error {
	if pool == nil {
		return errors.New("test mode requires --pg-dsn / PG_DSN")
	}

	if cfg.writeCSV && cfg.freshCSV && cfg.out != "" {
		_ = os.Remove(cfg.out)
	}
	if cfg.writeCSV && cfg.out != "" {
		_ = ensureCSVHeader(cfg.out, []string{
			"listing_id", "sold_date_db", "sold_date_adapter", "sold_price_adapter", "http_status", "url", "adapter",
		})
	}

	cands, err := loadSoldForTest(ctx, pool, cfg.pgSchema, cfg.generation, cfg.testHours, cfg.testBand, cfg.testLimit)
	if err != nil {
		return err
	}
	if len(cands) == 0 {
		fmt.Println("test: no candidates in this window")
		return nil
	}

	type rec struct{ r []string }
	outc := make(chan rec, len(cands))

	var wg sync.WaitGroup
	wk := max(1, min(cfg.workers, len(cands)))
	wg.Add(wk)

	for w := 0; w < wk; w++ {
		go func(shard int) {
			defer wg.Done()
			for i := shard; i < len(cands); i += wk {
				c := cands[i]
				snap, _ := adapter.FetchListing(ctx, c.ListingID, c.URL)

				adapterSoldStr := ""
				if snap.SoldAt != nil && !snap.SoldAt.IsZero() {
					adapterSoldStr = snap.SoldAt.UTC().Format(pgTS)
				}
				adapterPriceStr := ""
				if snap.SoldPrice > 0 {
					adapterPriceStr = strconv.Itoa(snap.SoldPrice)
				} else if snap.Status == "sold" && snap.Price > 0 {
					adapterPriceStr = strconv.Itoa(snap.Price)
				}

				if cfg.writeCSV && cfg.out != "" {
					outc <- rec{r: []string{
						strconv.FormatInt(c.ListingID, 10),
						c.SoldDate.UTC().Format(pgTS),
						adapterSoldStr,
						adapterPriceStr,
						strconv.Itoa(snap.HTTPStatus),
						c.URL,
						adapter.Name(),
					}}
				}
			}
		}(w)
	}

	go func() {
		wg.Wait()
		close(outc)
	}()

	var rows [][]string
	for r := range outc {
		rows = append(rows, r.r)
	}
	if cfg.writeCSV && cfg.out != "" && len(rows) > 0 {
		_ = appendCSV(cfg.out, rows)
		if cfg.verbose {
			fmt.Printf("[test] wrote %d rows to %s\n", len(rows), cfg.out)
		}
	}
	return nil
}

/* ========================= Repair mode (template) ========================= */

func runRepair(ctx context.Context, cfg config, pool *pgxpool.Pool, adapter MarketplaceAdapter) error {
	if pool == nil {
		return errors.New("repair requires --pg-dsn / PG_DSN")
	}

	if cfg.writeCSV && cfg.freshCSV && cfg.out != "" {
		_ = os.Remove(cfg.out)
	}
	if cfg.writeCSV && cfg.out != "" {
		_ = ensureCSVHeader(cfg.out, []string{"listing_id", "old_sold_date", "new_sold_date", "price", "url", "evidence", "http_status"})
	}

	var onlyIDs []int64
	if strings.TrimSpace(cfg.onlyIDs) != "" {
		for _, t := range strings.Split(cfg.onlyIDs, ",") {
			if v, err := strconv.ParseInt(strings.TrimSpace(t), 10, 64); err == nil {
				onlyIDs = append(onlyIDs, v)
			}
		}
	}

	args := []any{cfg.generation, cfg.repairFallbackDays, cfg.repairThresholdSec}
	argIdx := 4

	sb := strings.Builder{}
	sb.WriteString(fmt.Sprintf(
		`WITH base AS (
		   SELECT l.listing_id, l.url, l.sold_date
		     FROM "%s".listings l
		    WHERE l.generation=$1 AND l.status='sold'
		      AND l.sold_date >= now() - ($2::int) * interval '1 day'
		 ),
		 nearest AS (
		   SELECT b.listing_id, b.url, b.sold_date,
		          MIN(ABS(EXTRACT(EPOCH FROM (ph.observed_at - b.sold_date)))) AS delta_s
		     FROM base b
		     JOIN "%s".price_history ph
		       ON ph.listing_id=b.listing_id AND ph.generation=$1 AND ph.source='reverse' AND ph.status='sold'
		    GROUP BY 1,2,3
		 ),
		 todo AS (
		   SELECT n.listing_id, n.url, n.sold_date, n.delta_s
		     FROM nearest n
		    WHERE n.delta_s <= $3
		 )
		 SELECT listing_id, url, sold_date, delta_s
		   FROM todo`,
		cfg.pgSchema, cfg.pgSchema,
	))

	if len(onlyIDs) > 0 {
		sb.WriteString(fmt.Sprintf(" WHERE listing_id = ANY($%d::bigint[])", argIdx))
		args = append(args, onlyIDs)
		argIdx++
	}

	if cfg.repairLimit > 0 {
		sb.WriteString(fmt.Sprintf(" ORDER BY sold_date DESC LIMIT %d", cfg.repairLimit))
	} else {
		sb.WriteString(" ORDER BY sold_date DESC")
	}

	type cand struct {
		id      int64
		url     string
		oldSold *time.Time
		delta   *float64
	}

	rows, err := pool.Query(ctx, sb.String(), args...)
	if err != nil {
		return fmt.Errorf("repair query: %w", err)
	}
	defer rows.Close()

	var cands []cand
	for rows.Next() {
		var c cand
		if err := rows.Scan(&c.id, &c.url, &c.oldSold, &c.delta); err != nil {
			return fmt.Errorf("repair scan: %w", err)
		}
		cands = append(cands, c)
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("repair rows err: %w", err)
	}

	if len(cands) == 0 {
		fmt.Println("repair: no candidates")
		return nil
	}

	type rec struct{ r []string }
	outc := make(chan rec, len(cands))

	var fixed32 int32
	var wg sync.WaitGroup
	wk := max(1, min(cfg.workers, len(cands)))
	wg.Add(wk)

	for w := 0; w < wk; w++ {
		go func(shard int) {
			defer wg.Done()
			for i := shard; i < len(cands); i += wk {
				c := cands[i]
				snap, _ := adapter.FetchListing(ctx, c.id, c.url)

				if snap.SoldAt == nil || snap.SoldAt.IsZero() {
					continue
				}

				newDate := snap.SoldAt.UTC()

				price := snap.SoldPrice
				if price <= 0 {
					price = snap.Price
				}

				if cfg.writeDB {
					dec := Decision{Status: "sold", Price: price, SoldDate: newDate, Evidence: "repair_adapter"}
					_ = updateMain(ctx, pool, cfg.pgSchema, cfg.generation, c.id, dec)
					_ = upsertHistory(ctx, pool, cfg.pgSchema, cfg.generation, c.id, newDate, price, "sold")
				}

				if cfg.writeCSV && cfg.out != "" {
					oldStr := ""
					if c.oldSold != nil {
						oldStr = c.oldSold.UTC().Format(pgTS)
					}
					outc <- rec{r: []string{
						strconv.FormatInt(c.id, 10),
						oldStr,
						newDate.Format(pgTS),
						strconv.Itoa(price),
						c.url,
						firstNonEmpty(snap.Evidence, "repair_adapter"),
						strconv.Itoa(snap.HTTPStatus),
					}}
				}

				atomic.AddInt32(&fixed32, 1)
				if cfg.verbose {
					fmt.Printf("[repair] id=%d set sold_date=%s price=%d\n", c.id, newDate.Format(pgTS), price)
				}
			}
		}(w)
	}

	go func() {
		wg.Wait()
		close(outc)
	}()

	var rowsOut [][]string
	for r := range outc {
		rowsOut = append(rowsOut, r.r)
	}
	if cfg.writeCSV && cfg.out != "" && len(rowsOut) > 0 {
		_ = appendCSV(cfg.out, rowsOut)
		if cfg.verbose {
			fmt.Printf("[repair] wrote %d rows to %s\n", len(rowsOut), cfg.out)
		}
	}

	fmt.Printf("repair: gen=%d candidates=%d fixed=%d workers=%d rps_init=%d adapter=%s\n",
		cfg.generation, len(cands), int(atomic.LoadInt32(&fixed32)), cfg.workers, cfg.rps, adapter.Name(),
	)
	return nil
}

/* ========================= Stale & Diagnose ========================= */

func markStale(ctx context.Context, cfg config, pool *pgxpool.Pool) (int, error) {
	if pool == nil {
		return 0, errors.New("stale requires --pg-dsn / PG_DSN")
	}

	// Preview-only: DB trigger/scheduler typically owns "older than N days" transitions.
	var n int
	err := pool.QueryRow(ctx, fmt.Sprintf(
		`SELECT COUNT(*)
		   FROM "%s".listings l
		  WHERE l.generation=$1
		    AND l.status='live'
		    AND l.edited_date IS NOT NULL
		    AND l.edited_date <= now() - interval '21 days'`,
		cfg.pgSchema,
	), cfg.generation).Scan(&n)
	if err != nil {
		return 0, err
	}
	return n, nil
}

func diagnose(ctx context.Context, cfg config, pool *pgxpool.Pool) {
	if pool == nil {
		fmt.Println("diagnose: no DB")
		return
	}
	var total, sold, live, removed int
	_ = pool.QueryRow(ctx, fmt.Sprintf(`SELECT COUNT(*) FROM "%s".listings WHERE generation=$1`, cfg.pgSchema), cfg.generation).Scan(&total)
	_ = pool.QueryRow(ctx, fmt.Sprintf(`SELECT COUNT(*) FROM "%s".listings WHERE generation=$1 AND status='sold'`, cfg.pgSchema), cfg.generation).Scan(&sold)
	_ = pool.QueryRow(ctx, fmt.Sprintf(`SELECT COUNT(*) FROM "%s".listings WHERE generation=$1 AND status='live'`, cfg.pgSchema), cfg.generation).Scan(&live)
	_ = pool.QueryRow(ctx, fmt.Sprintf(`SELECT COUNT(*) FROM "%s".listings WHERE generation=$1 AND status='removed'`, cfg.pgSchema), cfg.generation).Scan(&removed)
	fmt.Printf("diagnose: gen=%d total=%d sold=%d live=%d removed=%d\n", cfg.generation, total, sold, live, removed)
}

/* ========================= Helpers ========================= */

func timeStr(t *time.Time) string {
	if t == nil {
		return ""
	}
	return t.UTC().Format(pgTS)
}

func derefStr(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func intToStr(v int) string {
	if v == 0 {
		return "0"
	}
	return strconv.Itoa(v)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

/* ========================= Main ========================= */

func main() {
	rand.Seed(time.Now().UnixNano())
	cfg := parseFlags()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// DB pool is optional for modes that don't need it, but reverse/repair/test/audit-status do.
	var pool *pgxpool.Pool
	if strings.TrimSpace(cfg.pgDSN) != "" {
		pool = mustOpenPool(ctx, cfg.pgDSN, 8)
		defer pool.Close()
	}

	// Rate limiter is owned by the adapter for networked adapters; mock ignores it.
	lim := newDynLimiter(cfg.minRPS, cfg.maxRPS, cfg.stepUpRPS, cfg.downMult, cfg.burstFactor, cfg.jitterMs)
	hc := newHTTPClient(max(8, min(cfg.workers, 256)))

	adapter, err := buildAdapter(cfg, hc, lim)
	if err != nil {
		fmt.Fprintln(os.Stderr, "adapter:", err)
		os.Exit(2)
	}

	switch cfg.mode {
	case "test":
		if err := runTest(ctx, cfg, pool, adapter); err != nil {
			fmt.Fprintln(os.Stderr, "test:", err)
			os.Exit(2)
		}

	case "reverse":
		if pool == nil {
			fmt.Fprintln(os.Stderr, "reverse requires --pg-dsn / PG_DSN")
			os.Exit(2)
		}
		cands, histOK, histErr, liveCnt, soldCnt, removedCnt, inactTrue, inactFalse, inactSeen, dur := reverseOnce(ctx, cfg, pool, adapter)
		fmt.Printf("reverse: gen=%d cands=%d history_ok=%d history_err=%d status{live=%d sold=%d removed=%d} inactive_true=%d inactive_false=%d inactive_seen=%d dur=%.2fs workers=%d rps_init=%d gomaxprocs=%d numcpu=%d adapter=%s\n",
			cfg.generation, cands, histOK, histErr,
			liveCnt, soldCnt, removedCnt,
			inactTrue, inactFalse, inactSeen,
			dur,
			cfg.workers, cfg.rps,
			runtime.GOMAXPROCS(0), runtime.NumCPU(),
			adapter.Name(),
		)

	case "repair":
		if err := runRepair(ctx, cfg, pool, adapter); err != nil {
			fmt.Fprintln(os.Stderr, "repair:", err)
			os.Exit(2)
		}

	case "audit-status":
		if err := runAuditStatus(ctx, cfg, pool, adapter); err != nil {
			fmt.Fprintln(os.Stderr, "audit-status:", err)
			os.Exit(2)
		}

	case "stale":
		n, err := markStale(ctx, cfg, pool)
		if err != nil {
			fmt.Fprintln(os.Stderr, "stale:", err)
			os.Exit(2)
		}
		fmt.Printf("stale: gen=%d eligible=%d (noop; DB marks older records)\n", cfg.generation, n)

	case "diagnose":
		diagnose(ctx, cfg, pool)

	default:
		fmt.Fprintln(os.Stderr, "unknown --mode")
		os.Exit(2)
	}
}

/* ========================= Optional helper ========================= */

func hashTitleDesc(title, desc string) string {
	h := sha256.Sum256([]byte(title + "\n" + desc))
	return fmt.Sprintf("%x", h[:])
}

/* ========================= pgx helpers (optional) ========================= */

// runInTx is a small helper you can use later if you decide to batch writes.
// Not used by default to keep the reverse loop straightforward.
func runInTx(ctx context.Context, pool *pgxpool.Pool, fn func(pgx.Tx) error) error {
	if pool == nil {
		return nil
	}
	tx, err := pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	if err := fn(tx); err != nil {
		return err
	}
	return tx.Commit(ctx)
}
