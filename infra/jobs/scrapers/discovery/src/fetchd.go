//go:build !js

// Marketplace ingest template (Go)
// --------------------------------
//
// This is a security-sanitized, public-release version of an internal ingest job.
// It demonstrates:
//   • Job-oriented ingestion (search -> detail -> normalize -> sink)
//   • Append-only CSV sink with a sidecar ID index and a cross-process lock
//   • Optional Postgres sink (single table; ON CONFLICT DO NOTHING)
//   • Adaptive concurrency (AIMD) driven by p95 latency and 429-rate SLO
//   • Optional bounded request rate (token bucket)
//   • Embedded /metrics (Prometheus exposition) and /debug/pprof/*
//
// Important public-release constraints:
//   • No platform identifiers, site-specific endpoints, HTML selectors, or parsing logic.
//   • All connector logic is behind MarketplaceAdapter.
//   • Default adapter is offline-safe mock mode.
//
// Configuration is primarily via environment variables (flags can override):
//   OUT_CSV, PG_DSN, PG_SCHEMA, MARKETPLACE_ADAPTER, MARKETPLACE_BASE_URL,
//   SEARCH_QUERY, PAGES, WORKERS, SEARCH_WORKERS, REQUEST_RPS, METRICS_ADDR, ...
package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"net/http/pprof"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// ───────── Adapter layer ─────────


// SearchParams describes a paginated marketplace search.
type SearchParams struct {
	Query    string
	Page     int
	MinPrice int
}

// ListingSummary is a lightweight representation returned from SearchListings.
type ListingSummary struct {
	ListingID string `json:"listing_id"`
	URL       string `json:"url,omitempty"`
	Title     string `json:"title,omitempty"`
	Price     int    `json:"price,omitempty"`
}

// ListingDetails is a normalized listing record returned from FetchListing / ParsePayload.
type ListingDetails struct {
	ListingID     string            `json:"listing_id"`
	Title         string            `json:"title"`
	Price         int               `json:"price"`
	URL           string            `json:"url"`
	Description   string            `json:"description,omitempty"`
	UpdatedAtISO  string            `json:"updated_at,omitempty"`
	LocationCity  string            `json:"location_city,omitempty"`
	PostalCode    string            `json:"postal_code,omitempty"`
	Attributes    map[string]string `json:"attributes,omitempty"` // safe, generic extension point
}

// FetchMeta provides request-level telemetry without leaking connector details.
type FetchMeta struct {
	StatusCode int
	Latency    time.Duration
}

// MarketplaceAdapter abstracts all marketplace-specific logic.
type MarketplaceAdapter interface {
	// SearchListings returns listing summaries for a query and page.
	SearchListings(ctx context.Context, params SearchParams) ([]ListingSummary, FetchMeta, error)

	// FetchListing fetches a single listing by ID (networked adapters) or synthesizes it (mock).
	FetchListing(ctx context.Context, listingID string) (ListingDetails, FetchMeta, error)

	// ParsePayload parses a raw listing payload into a normalized ListingDetails.
	ParsePayload(raw []byte) (ListingDetails, error)
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP JSON adapter (generic placeholder)
// ─────────────────────────────────────────────────────────────────────────────

// HTTPJSONAdapter expects a JSON API under MARKETPLACE_BASE_URL.
//
// Expected endpoints (placeholders, not target-specific):
//   GET {base}/api/search?q=...&page=...&min_price=...
//     -> either {"listings":[...]} or [...]
//   GET {base}/api/listings/{listing_id}
//     -> either {"listing":{...}} or {...}
//
// This is intentionally minimal; public releases should keep the adapter as
// a stub and implement any real connector in a private repository.
type HTTPJSONAdapter struct {
	baseURL   string
	client    *http.Client
	userAgent string
	timeout   time.Duration
}

type HTTPJSONAdapterOptions struct {
	BaseURL   string
	UserAgent string
	Timeout   time.Duration
}

func NewHTTPJSONAdapter(opts HTTPJSONAdapterOptions) (*HTTPJSONAdapter, error) {
	base := strings.TrimSpace(opts.BaseURL)
	if base == "" {
		return nil, errors.New("BaseURL is required")
	}
	if _, err := url.Parse(base); err != nil {
		return nil, fmt.Errorf("invalid BaseURL: %w", err)
	}
	to := opts.Timeout
	if to <= 0 {
		to = 20 * time.Second
	}
	ua := strings.TrimSpace(opts.UserAgent)
	if ua == "" {
		ua = "marketplace-ingest-template/1.0"
	}
	return &HTTPJSONAdapter{
		baseURL:   strings.TrimRight(base, "/"),
		client:    &http.Client{Timeout: to},
		userAgent: ua,
		timeout:   to,
	}, nil
}

func (a *HTTPJSONAdapter) SearchListings(ctx context.Context, params SearchParams) ([]ListingSummary, FetchMeta, error) {
	start := time.Now()
	u, err := url.Parse(a.baseURL + "/api/search")
	if err != nil {
		return nil, FetchMeta{StatusCode: 0, Latency: time.Since(start)}, err
	}
	q := u.Query()
	q.Set("q", strings.TrimSpace(params.Query))
	if params.Page > 0 {
		q.Set("page", strconv.Itoa(params.Page))
	}
	if params.MinPrice > 0 {
		q.Set("min_price", strconv.Itoa(params.MinPrice))
	}
	u.RawQuery = q.Encode()

	body, status, err := a.doGET(ctx, u.String())
	meta := FetchMeta{StatusCode: status, Latency: time.Since(start)}
	if err != nil {
		return nil, meta, err
	}

	// Accept both object-wrapped and bare-array payloads.
	var wrapped struct {
		Listings []ListingSummary `json:"listings"`
	}
	if err := json.Unmarshal(body, &wrapped); err == nil && len(wrapped.Listings) > 0 {
		return normalizeSummaries(wrapped.Listings), meta, nil
	}
	var arr []ListingSummary
	if err := json.Unmarshal(body, &arr); err != nil {
		return nil, meta, fmt.Errorf("search payload parse: %w", err)
	}
	return normalizeSummaries(arr), meta, nil
}

func (a *HTTPJSONAdapter) FetchListing(ctx context.Context, listingID string) (ListingDetails, FetchMeta, error) {
	start := time.Now()
	id := strings.TrimSpace(listingID)
	if id == "" {
		return ListingDetails{}, FetchMeta{StatusCode: 0, Latency: time.Since(start)}, errors.New("listingID is required")
	}

	u := a.baseURL + "/api/listings/" + url.PathEscape(id)
	body, status, err := a.doGET(ctx, u)
	meta := FetchMeta{StatusCode: status, Latency: time.Since(start)}
	if err != nil {
		return ListingDetails{}, meta, err
	}

	d, err := a.ParsePayload(body)
	if err != nil {
		return ListingDetails{}, meta, err
	}
	// Ensure ID is set even if the payload omitted it.
	if d.ListingID == "" {
		d.ListingID = id
	}
	// If URL is missing, synthesize a safe placeholder URL under baseURL.
	if d.URL == "" {
		d.URL = a.baseURL + "/listings/" + url.PathEscape(d.ListingID)
	}
	return d, meta, nil
}

func (a *HTTPJSONAdapter) ParsePayload(raw []byte) (ListingDetails, error) {
	// Accept both object-wrapped and bare-object payloads.
	var wrapped struct {
		Listing ListingDetails `json:"listing"`
	}
	if err := json.Unmarshal(raw, &wrapped); err == nil && wrapped.Listing.ListingID != "" {
		return normalizeDetails(wrapped.Listing), nil
	}
	var d ListingDetails
	if err := json.Unmarshal(raw, &d); err != nil {
		return ListingDetails{}, fmt.Errorf("detail payload parse: %w", err)
	}
	return normalizeDetails(d), nil
}

func (a *HTTPJSONAdapter) doGET(ctx context.Context, u string) ([]byte, int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", a.userAgent)

	resp, err := a.client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	status := resp.StatusCode
	b, _ := io.ReadAll(resp.Body)
	if status < 200 || status >= 300 {
		// Return body for debugging but do not attempt to interpret it here.
		return nil, status, fmt.Errorf("http status %d", status)
	}
	return b, status, nil
}

func normalizeSummaries(in []ListingSummary) []ListingSummary {
	out := make([]ListingSummary, 0, len(in))
	seen := make(map[string]struct{}, len(in))
	for _, it := range in {
		id := strings.TrimSpace(it.ListingID)
		if id == "" {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		it.ListingID = id
		it.Title = strings.TrimSpace(it.Title)
		it.URL = strings.TrimSpace(it.URL)
		out = append(out, it)
	}
	return out
}

func normalizeDetails(d ListingDetails) ListingDetails {
	d.ListingID = strings.TrimSpace(d.ListingID)
	d.Title = strings.TrimSpace(d.Title)
	d.URL = strings.TrimSpace(d.URL)
	d.Description = strings.TrimSpace(d.Description)
	d.UpdatedAtISO = strings.TrimSpace(d.UpdatedAtISO)
	d.LocationCity = strings.TrimSpace(d.LocationCity)
	d.PostalCode = strings.TrimSpace(d.PostalCode)
	if d.Attributes == nil {
		d.Attributes = map[string]string{}
	}
	return d
}

// ─────────────────────────────────────────────────────────────────────────────
// Mock adapter (offline-safe)
// ─────────────────────────────────────────────────────────────────────────────

// MockAdapter produces synthetic listings for demos and unit tests.
// It is deterministic within a single process and does not make network calls.
type MockAdapter struct {
	baseURL string
	seed    int64
}

type MockAdapterOptions struct {
	BaseURL string // used only to synthesize URLs; safe to set to an .invalid domain.
	Seed    int64  // optional; 0 uses current time
}

func NewMockAdapter(opts MockAdapterOptions) *MockAdapter {
	base := strings.TrimSpace(opts.BaseURL)
	if base == "" {
		base = "https://example-marketplace.invalid"
	}
	seed := opts.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	return &MockAdapter{
		baseURL: strings.TrimRight(base, "/"),
		seed:    seed,
	}
}

func (m *MockAdapter) SearchListings(ctx context.Context, params SearchParams) ([]ListingSummary, FetchMeta, error) {
	start := time.Now()
	_ = ctx // kept for signature parity; mock does not block on ctx.

	page := params.Page
	if page <= 0 {
		page = 1
	}
	q := strings.TrimSpace(params.Query)
	if q == "" {
		q = "example"
	}

	// Deterministic pseudo-random from query+page.
	h := fnv64(q + "|" + strconv.Itoa(page))
	r := rand.New(rand.NewSource(int64(h) ^ m.seed))

	n := 12 // per page
	out := make([]ListingSummary, 0, n)
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("%d%08d", page, i+1) // synthetic, non-traceable
		price := 1000 + (i * 25) + int(r.Int31n(50))
		if params.MinPrice > 0 && price < params.MinPrice {
			price = params.MinPrice
		}
		out = append(out, ListingSummary{
			ListingID: id,
			Title:     fmt.Sprintf("%s item %d", q, i+1),
			Price:     price,
			URL:       m.baseURL + "/listings/" + url.PathEscape(id),
		})
	}
	return out, FetchMeta{StatusCode: 200, Latency: time.Since(start)}, nil
}

func (m *MockAdapter) FetchListing(ctx context.Context, listingID string) (ListingDetails, FetchMeta, error) {
	start := time.Now()
	select {
	case <-ctx.Done():
		return ListingDetails{}, FetchMeta{StatusCode: 0, Latency: time.Since(start)}, ctx.Err()
	default:
	}

	id := strings.TrimSpace(listingID)
	if id == "" {
		return ListingDetails{}, FetchMeta{StatusCode: 0, Latency: time.Since(start)}, errors.New("listingID is required")
	}

	// Small synthetic latency to exercise metrics/tuning without network calls.
	time.Sleep(5 * time.Millisecond)

	d := ListingDetails{
		ListingID:    id,
		Title:        "Synthetic listing " + id,
		Price:        1250,
		URL:          m.baseURL + "/listings/" + url.PathEscape(id),
		Description:  "Synthetic description (public-release mock adapter).",
		UpdatedAtISO: time.Now().UTC().Format(time.RFC3339),
		LocationCity: "Example City",
		PostalCode:   "0000",
		Attributes: map[string]string{
			"attribute_text_1": "example",
			"attribute_text_2": "example",
			"category":         "example",
		},
	}
	return d, FetchMeta{StatusCode: 200, Latency: time.Since(start)}, nil
}

func (m *MockAdapter) ParsePayload(raw []byte) (ListingDetails, error) {
	// Mock adapter treats payload as ListingDetails JSON.
	var d ListingDetails
	if err := json.Unmarshal(raw, &d); err != nil {
		return ListingDetails{}, fmt.Errorf("mock payload parse: %w", err)
	}
	return normalizeDetails(d), nil
}

// fnv64 returns a simple 64-bit hash for deterministic mock data.
func fnv64(s string) uint64 {
	const (
		offset64 = 14695981039346656037
		prime64  = 1099511628211
	)
	var h uint64 = offset64
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= prime64
	}
	return h
}


// ───────── Defaults ─────────

const (
	CONNECT_TIMEOUT   = 3 * time.Second
	HEADER_TIMEOUT    = 12 * time.Second
	IDLE_CONN_TIMEOUT = 90 * time.Second

	LOCK_TTL_SECS = 600 // 10 minutes

	// Tuner defaults
	defaultSLOMult     = 1.35  // target: p95 <= baseline * sloMult
	defaultMax429Rate  = 0.01  // 1%
	defaultBaseGuessMs = 250.0 // baseline guess until we have samples
	defaultEvalEvery   = 1500 * time.Millisecond
	defaultP95WinSize  = 256 // samples in theFS = 256
)

// CSV schema (public-release, generic).
var CSV_COLS = []string{
	"title", "price", "url", "description", "updated_at",
	"location_city", "postal_code", "attribute_num_1", "last_fetched", "status",
	"first_seen", "last_seen", "listing_id", "attribute_text_1", "attribute_text_2", "category", "score", "price_per_unit",
}

// ───────── Globals for signals/lock ─────────

var (
	stopRequested int32 // atomic
	lockPath      string
	lockHeld      int32
)

// ───────── Config ─────────

type config struct {
	out       string
	pages     int
	workers   int
	minPrice  int
	rawPostal bool

	printNew bool
	jsonLogs bool

	metricsAddr string
	daemon      bool
	daemonMinSec int
	daemonMaxSec int

	searchWorkers int
	rps           float64
	sloMult       float64
	max429Rate    float64
	p95Window     int

	// Adapter
	adapter          string // mock|http-json
	marketplaceBase  string
	searchQuery      string

	// DB (optional)
	pgDSN        string
	pgSchema     string
	pgBatch      int
	pgMaxConns   int
	pgViaBouncer bool
}

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
	i, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return i
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

func parseFlags() config {
	var cfg config

	flag.StringVar(&cfg.out, "out", envString("OUT_CSV", ""), "Output CSV path (append-only). Env: OUT_CSV")
	flag.IntVar(&cfg.pages, "pages", envInt("PAGES", 1), "Search pages to process (0 = healthcheck). Env: PAGES")
	flag.IntVar(&cfg.workers, "workers", envInt("WORKERS", 64), "Max concurrent detail fetch slots (tuner adjusts). Env: WORKERS")
	flag.IntVar(&cfg.minPrice, "min-price", envInt("MIN_PRICE", 0), "Drop listings below this price (0 disables). Env: MIN_PRICE")
	flag.BoolVar(&cfg.rawPostal, "raw-postal", envBool("RAW_POSTAL", false), "Write postal_code as digits only (not Excel-safe). Env: RAW_POSTAL")

	flag.BoolVar(&cfg.printNew, "print-new", envBool("PRINT_NEW", false), "Print listing_id + title for newly inserted rows. Env: PRINT_NEW")
	flag.BoolVar(&cfg.jsonLogs, "json-logs", envBool("JSON_LOGS", false), "Emit a JSON summary line (keeps human summary too). Env: JSON_LOGS")

	flag.StringVar(&cfg.metricsAddr, "metrics", envString("METRICS_ADDR", ""), "Serve /metrics and /debug/pprof/* on this address, e.g. :6060. Env: METRICS_ADDR")
	flag.BoolVar(&cfg.daemon, "daemon", envBool("DAEMON", false), "Run forever: sleep between runs. Env: DAEMON")
	flag.IntVar(&cfg.daemonMinSec, "daemon-min-sec", envInt("DAEMON_MIN_SEC", 20), "Daemon: minimum seconds between runs. Env: DAEMON_MIN_SEC")
	flag.IntVar(&cfg.daemonMaxSec, "daemon-max-sec", envInt("DAEMON_MAX_SEC", 180), "Daemon: maximum seconds between runs. Env: DAEMON_MAX_SEC")

	flag.IntVar(&cfg.searchWorkers, "search-workers", envInt("SEARCH_WORKERS", 16), "Concurrent search page workers. Env: SEARCH_WORKERS")
	flag.Float64Var(&cfg.rps, "rps", envFloat("REQUEST_RPS", 0), "Token bucket for detail requests (tokens/sec). 0=unlimited. Env: REQUEST_RPS")
	flag.Float64Var(&cfg.sloMult, "slo-mult", envFloat("SLO_MULT", defaultSLOMult), "Adaptive concurrency target p95 multiplier vs baseline. Env: SLO_MULT")
	flag.Float64Var(&cfg.max429Rate, "max-429-rate", envFloat("MAX_429_RATE", defaultMax429Rate), "Adaptive concurrency max 429 ratio per window. Env: MAX_429_RATE")
	flag.IntVar(&cfg.p95Window, "p95-window", envInt("P95_WINDOW", defaultP95WinSize), "Window size for p95 computation. Env: P95_WINDOW")

	// Adapter config
	flag.StringVar(&cfg.adapter, "marketplace-adapter", envString("MARKETPLACE_ADAPTER", "mock"), "Adapter: mock|http-json. Env: MARKETPLACE_ADAPTER")
	flag.StringVar(&cfg.marketplaceBase, "marketplace-base-url", envString("MARKETPLACE_BASE_URL", "https://example-marketplace.invalid"), "Marketplace base URL (placeholder). Env: MARKETPLACE_BASE_URL")
	flag.StringVar(&cfg.searchQuery, "search-query", envString("SEARCH_QUERY", "example"), "Search query (adapter-defined). Env: SEARCH_QUERY")

	// DB config (optional)
	flag.StringVar(&cfg.pgDSN, "pg-dsn", envString("PG_DSN", ""), "Postgres DSN (enables DB mode). Env: PG_DSN")
	flag.StringVar(&cfg.pgSchema, "pg-schema", envString("PG_SCHEMA", "public"), "Target Postgres schema. Env: PG_SCHEMA")
	flag.IntVar(&cfg.pgBatch, "pg-batch", envInt("PG_BATCH", 200), "DB insert batch size. Env: PG_BATCH")
	flag.IntVar(&cfg.pgMaxConns, "pg-max-conns", envInt("PG_MAX_CONNS", 2), "DB max connections. Env: PG_MAX_CONNS")
	flag.BoolVar(&cfg.pgViaBouncer, "pg-via-bouncer", envBool("PG_VIA_BOUNCER", true), "Use simple protocol for PgBouncer txn pooling. Env: PG_VIA_BOUNCER")

	flag.Parse()

	if cfg.out == "" && cfg.pgDSN == "" {
		fmt.Fprintln(os.Stderr, "either --out (CSV) / OUT_CSV or --pg-dsn (DB) / PG_DSN is required")
		os.Exit(2)
	}
	if cfg.searchWorkers <= 0 {
		cfg.searchWorkers = 1
	}
	if cfg.workers <= 0 {
		cfg.workers = 1
	}
	if cfg.p95Window < 64 {
		cfg.p95Window = 64
	}
	if cfg.daemonMaxSec < cfg.daemonMinSec {
		cfg.daemonMaxSec = cfg.daemonMinSec
	}

	return cfg
}

// ───────── Metrics (Prometheus) ─────────

type Metrics struct {
	mu sync.Mutex

	// Adapter requests (search + detail)
	reqTotalByCode map[int]uint64
	req429Count    uint64
	reqCountWindow uint64
	latSamplesMs   []float64
	latIdx         int
	latCount       int
	p50ms          float64
	p95ms          float64
	baselineP95ms  float64 // learned after first stable window

	// Queues & workers
	queueDepth     int
	inflight       int
	tunerWindow    int
	rpsTokensAvail float64

	// Errors
	fetchErrors int
	parseErrors int

	start time.Time
}

func NewMetrics(win int) *Metrics {
	return &Metrics{
		reqTotalByCode: make(map[int]uint64, 8),
		latSamplesMs:   make([]float64, win),
		start:          time.Now(),
	}
}

func (m *Metrics) RecordRequest(code int, ms float64) {
	m.mu.Lock()
	m.reqTotalByCode[code]++
	if code == 429 {
		m.req429Count++
	}
	m.reqCountWindow++
	// record latency into ring
	m.latSamplesMs[m.latIdx] = ms
	m.latIdx = (m.latIdx + 1) % len(m.latSamplesMs)
	if m.latCount < len(m.latSamplesMs) {
		m.latCount++
	}
	m.mu.Unlock()
}

func (m *Metrics) SnapshotLatencies() (p50, p95 float64) {
	m.mu.Lock()
	n := m.latCount
	if n == 0 {
		m.mu.Unlock()
		return 0, 0
	}
	buf := make([]float64, n)
	copy(buf, m.latSamplesMs[:n])
	m.mu.Unlock()

	sort.Float64s(buf)
	p50 = quantile(buf, 0.50)
	p95 = quantile(buf, 0.95)
	m.mu.Lock()
	m.p50ms, m.p95ms = p50, p95
	m.mu.Unlock()
	return
}

func quantile(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := q * float64(len(sorted)-1)
	i := int(idx)
	if i >= len(sorted)-1 {
		return sorted[len(sorted)-1]
	}
	frac := idx - float64(i)
	return sorted[i]*(1-frac) + sorted[i+1]*frac
}

func (m *Metrics) RecordErrors(errFetch, errParse bool) {
	m.mu.Lock()
	if errFetch {
		m.fetchErrors++
	}
	if errParse {
		m.parseErrors++
	}
	m.mu.Unlock()
}

func (m *Metrics) SetQueueDepth(d int) {
	m.mu.Lock()
	m.queueDepth = d
	m.mu.Unlock()
}

func (m *Metrics) SetInflight(n int) {
	m.mu.Lock()
	m.inflight = n
	m.mu.Unlock()
}

func (m *Metrics) SetTunerWindow(n int) {
	m.mu.Lock()
	m.tunerWindow = n
	m.mu.Unlock()
}

func (m *Metrics) SetRPSTokensAvail(t float64) {
	m.mu.Lock()
	m.rpsTokensAvail = t
	m.mu.Unlock()
}

func (m *Metrics) Reset429Window() (ratio float64) {
	m.mu.Lock()
	ratio = 0
	if m.reqCountWindow > 0 {
		ratio = float64(m.req429Count) / float64(m.reqCountWindow)
	}
	m.req429Count = 0
	m.reqCountWindow = 0
	m.mu.Unlock()
	return
}

// ───────── Adaptive Concurrency Gate (AIMD) ─────────

type ConcurrencyGate struct {
	mu      sync.Mutex
	cond    *sync.Cond
	window  int
	current int
}

func NewConcurrencyGate(n int) *ConcurrencyGate {
	g := &ConcurrencyGate{window: max(1, n)}
	g.cond = sync.NewCond(&g.mu)
	return g
}

func (g *ConcurrencyGate) Acquire(ctx context.Context) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	for g.current >= g.window {
		if ctx.Err() != nil {
			return false
		}
		g.cond.Wait()
	}
	g.current++
	return true
}

func (g *ConcurrencyGate) Release() {
	g.mu.Lock()
	if g.current > 0 {
		g.current--
	}
	g.cond.Broadcast()
	g.mu.Unlock()
}

func (g *ConcurrencyGate) Inflight() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.current
}

func (g *ConcurrencyGate) Window() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.window
}

func (g *ConcurrencyGate) SetWindow(n int) {
	g.mu.Lock()
	if n < 1 {
		n = 1
	}
	g.window = n
	g.cond.Broadcast()
	g.mu.Unlock()
}

type AutoTuner struct {
	mu              sync.Mutex
	gate            *ConcurrencyGate
	minW            int
	maxW            int
	goodStreak      int
	badStreak       int
	sloMult         float64
	max429Rate      float64
	baselineP95ms   float64
	baselineSet     bool
	evalEvery       time.Duration
	metrics         *Metrics
	lastRecalc      time.Time
	last429Ratio    float64
	lastObservedP95 float64
}

func NewAutoTuner(g *ConcurrencyGate, minW, maxW int, sloMult, max429 float64, evalEvery time.Duration, m *Metrics) *AutoTuner {
	if minW <= 0 {
		minW = 1
	}
	if maxW < minW {
		maxW = minW
	}
	if evalEvery <= 0 {
		evalEvery = defaultEvalEvery
	}
	return &AutoTuner{
		gate:       g,
		minW:       minW,
		maxW:       maxW,
		sloMult:    sloMult,
		max429Rate: max429,
		evalEvery:  evalEvery,
		metrics:    m,
	}
}

func (t *AutoTuner) Recalc() {
	now := time.Now()
	t.mu.Lock()
	if !t.lastRecalc.IsZero() && now.Sub(t.lastRecalc) < t.evalEvery {
		t.mu.Unlock()
		return
	}
	t.lastRecalc = now
	t.mu.Unlock()

	_, p95 := t.metrics.SnapshotLatencies()
	r429 := t.metrics.Reset429Window()

	t.mu.Lock()
	defer t.mu.Unlock()
	t.lastObservedP95 = p95
	t.last429Ratio = r429

	if !t.baselineSet && p95 > 0 {
		t.baselineP95ms = p95
		t.baselineSet = true
	}
	base := t.baselineP95ms
	if base <= 0 {
		base = defaultBaseGuessMs
	}
	tooSlow := p95 > (base * t.sloMult)
	tooMany429 := r429 > t.max429Rate

	w := t.gate.Window()
	switch {
	case tooSlow || tooMany429:
		newW := int(float64(w) * 0.70)
		if newW < t.minW {
			newW = t.minW
		}
		t.gate.SetWindow(newW)
		t.metrics.SetTunerWindow(newW)
		t.badStreak++
		t.goodStreak = 0
	default:
		t.goodStreak++
		if p95 < (base*t.sloMult*0.75) && r429 < (t.max429Rate*0.5) {
			inc := w / 16
			if inc < 2 {
				inc = 2
			}
			newW := w + inc
			if newW > t.maxW {
				newW = t.maxW
			}
			if newW != w {
				t.gate.SetWindow(newW)
				t.metrics.SetTunerWindow(newW)
			}
		} else if t.goodStreak%2 == 0 {
			newW := w + 1
			if newW > t.maxW {
				newW = t.maxW
			}
			if newW != w {
				t.gate.SetWindow(newW)
				t.metrics.SetTunerWindow(newW)
			}
		}
	}
}

// ───────── Token bucket ─────────

type TokenBucket struct {
	mu           sync.Mutex
	capacity     float64
	tokens       float64
	refillPerSec float64
	last         time.Time
	jitterFrac   float64
}

func NewTokenBucket(rps float64, jitterFrac float64) *TokenBucket {
	if rps <= 0 {
		return nil
	}
	cap := math.Max(1, rps*2)
	return &TokenBucket{
		capacity:     cap,
		tokens:       cap,
		refillPerSec: rps,
		last:         time.Now(),
		jitterFrac:   jitterFrac,
	}
}

func (b *TokenBucket) Take(ctx context.Context, m *Metrics) bool {
	if b == nil {
		return true
	}
	for {
		b.mu.Lock()
		now := time.Now()
		elapsed := now.Sub(b.last).Seconds()
		if elapsed > 0 {
			b.tokens = math.Min(b.capacity, b.tokens+elapsed*b.refillPerSec)
			b.last = now
		}
		ok := false
		if b.tokens >= 1.0 {
			b.tokens -= 1.0
			ok = true
		}
		tok := b.tokens
		b.mu.Unlock()

		if m != nil {
			m.SetRPSTokensAvail(tok)
		}
		if ok {
			return true
		}
		toNext := time.Duration((1.0/b.refillPerSec)*float64(time.Second)) + jitterDuration(b.jitterFrac, 0.10)
		select {
		case <-ctx.Done():
			return false
		case <-time.After(toNext):
		}
	}
}

func jitterDuration(frac float64, base float64) time.Duration {
	if frac <= 0 {
		frac = base
	}
	j := 1 + ((rand.Float64()*2 - 1) * frac)
	return time.Duration(j * float64(30*time.Millisecond))
}

// ───────── Stats & 429 dampener ─────────

type Stats struct {
	mu             sync.Mutex
	FetchErrors    int
	ParseErrors    int
	BelowMin       int
	consecutive429 int
}

func (s *Stats) inc(field *int, n int) {
	s.mu.Lock()
	*field += n
	s.mu.Unlock()
}

func (s *Stats) record429() int {
	s.mu.Lock()
	s.consecutive429++
	v := s.consecutive429
	s.mu.Unlock()
	return v
}

func (s *Stats) reset429() {
	s.mu.Lock()
	s.consecutive429 = 0
	s.mu.Unlock()
}

type Cooloff struct {
	Threshold int
	Min       time.Duration
	Max       time.Duration
}

func maybeGlobalCooloff(stats *Stats, cool Cooloff) {
	count := stats.record429()
	if count >= cool.Threshold {
		d := cool.Min + time.Duration(rand.Float64()*float64(cool.Max-cool.Min))
		time.Sleep(d)
		stats.reset429()
	}
}

// ───────── Small utils ─────────

func nowISO() string { return time.Now().UTC().Format(time.RFC3339) }

func digitOnly(s string) string {
	var b strings.Builder
	for _, r := range s {
		if r >= '0' && r <= '9' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func fixPostalCode(pc string, excelSafe bool) string {
	d := digitOnly(pc)
	if d == "" {
		return ""
	}
	for len(d) < 4 {
		d = "0" + d
	}
	if excelSafe {
		return fmt.Sprintf("=\"%s\"", d)
	}
	return d
}

// ───────── CSV + .ids (append-only; fsync) ─────────

func ensureCSVHeader(path string) error {
	needHeader := false
	if _, err := os.Stat(path); err != nil {
		needHeader = true
	} else {
		fi, err := os.Stat(path)
		if err != nil || fi.Size() == 0 {
			needHeader = true
		}
	}
	if !needHeader {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(absPath(path)), 0755); err != nil && !errors.Is(err, os.ErrExist) {
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	// Write UTF-8 BOM for Excel friendliness
	if _, err := f.Write([]byte{0xEF, 0xBB, 0xBF}); err != nil {
		f.Close()
		return err
	}
	w := csv.NewWriter(f)
	if err := w.Write(CSV_COLS); err != nil {
		f.Close()
		return err
	}
	w.Flush()
	if err := w.Error(); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func absPath(p string) string {
	ap, err := filepath.Abs(p)
	if err != nil {
		return p
	}
	return ap
}

func readIDsSidecar(idsPath string) map[string]struct{} {
	out := make(map[string]struct{}, 65536)
	b, err := os.ReadFile(idsPath)
	if err != nil {
		return out
	}
	for _, line := range strings.Split(string(b), "\n") {
		s := strings.TrimSpace(line)
		if s != "" {
			out[s] = struct{}{}
		}
	}
	return out
}

func scanCSVForIDs(csvPath string) map[string]struct{} {
	out := make(map[string]struct{}, 65536)
	f, err := os.Open(csvPath)
	if err != nil {
		return out
	}
	defer f.Close()

	// skip BOM if present
	br := bufio.NewReader(f)
	first3, _ := br.Peek(3)
	if len(first3) == 3 && first3[0] == 0xEF && first3[1] == 0xBB && first3[2] == 0xBF {
		br.Discard(3)
	}
	r := csv.NewReader(br)

	header, err := r.Read()
	if err != nil {
		return out
	}
	idx := -1
	for i, h := range header {
		if h == "listing_id" {
			idx = i
			break
		}
	}
	if idx < 0 {
		return out
	}
	for {
		row, err := r.Read()
		if err != nil {
			break
		}
		if len(row) <= idx {
			continue
		}
		id := strings.TrimSpace(row[idx])
		if id != "" {
			out[id] = struct{}{}
		}
	}
	return out
}

func ensureIDsIndex(csvPath, idsPath string) map[string]struct{} {
	csvExists := fileExists(csvPath)
	idsExists := fileExists(idsPath)

	if idsExists && csvExists {
		csvInfo, _ := os.Stat(csvPath)
		idsInfo, _ := os.Stat(idsPath)
		if csvInfo != nil && idsInfo != nil && csvInfo.ModTime().After(idsInfo.ModTime()) {
			ids := scanCSVForIDs(csvPath)
			writeIDs(idsPath, ids)
			return ids
		}
	}

	if idsExists {
		return readIDsSidecar(idsPath)
	}

	var ids map[string]struct{}
	if csvExists {
		ids = scanCSVForIDs(csvPath)
	} else {
		ids = make(map[string]struct{})
	}
	writeIDs(idsPath, ids)
	return ids
}

func writeIDs(idsPath string, ids map[string]struct{}) {
	_ = os.MkdirAll(filepath.Dir(absPath(idsPath)), 0755)
	sl := make([]string, 0, len(ids))
	for id := range ids {
		sl = append(sl, id)
	}
	sort.Strings(sl)
	_ = os.WriteFile(idsPath, []byte(strings.Join(sl, "\n")+"\n"), 0644)
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

type Row struct {
	Title          string
	Price          int
	URL            string
	Description    string
	UpdatedAt      string
	LocationCity   string
	PostalCode     string
	AttributeNum1  int
	LastFetched    string
	Status         string
	FirstSeen      string
	LastSeen       string
	ListingID      string
	AttributeText1 string
	AttributeText2 string
	Category       string
	Score          string
	PricePerUnit   string
}

func appendRowsCSV(path string, rows []*Row) error {
	if len(rows) == 0 {
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
		rec := []string{
			r.Title,
			strconv.Itoa(r.Price),
			r.URL,
			r.Description,
			r.UpdatedAt,
			r.LocationCity,
			r.PostalCode,
			strconv.Itoa(r.AttributeNum1),
			r.LastFetched,
			r.Status,
			r.FirstSeen,
			r.LastSeen,
			r.ListingID,
			r.AttributeText1,
			r.AttributeText2,
			r.Category,
			r.Score,
			r.PricePerUnit,
		}
		if err := w.Write(rec); err != nil {
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

func appendIDs(idsPath string, ids []string) error {
	if len(ids) == 0 {
		return nil
	}
	f, err := os.OpenFile(idsPath, os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	bw := bufio.NewWriter(f)
	for _, id := range ids {
		bw.WriteString(id)
		bw.WriteByte('\n')
	}
	if err := bw.Flush(); err != nil {
		return err
	}
	return f.Sync()
}

// ───────── Lock file (with TTL & heartbeat) ─────────

func acquireLock(lockPath string, ttl time.Duration) bool {
	abspath := absPath(lockPath)
	for {
		f, err := os.OpenFile(abspath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
		if err == nil {
			_, _ = f.WriteString(fmt.Sprintf(`{"pid":%d,"time":%d}`+"\n", os.Getpid(), time.Now().Unix()))
			_ = f.Close()
			return true
		}
		fi, err := os.Stat(abspath)
		if err != nil {
			continue
		}
		age := time.Since(fi.ModTime())
		if age >= ttl {
			_ = os.Remove(abspath)
			continue
		}
		fmt.Println("another writer active; aborting")
		return false
	}
}

func releaseLock(lockPath string) {
	if lockPath == "" {
		return
	}
	_ = os.Remove(lockPath)
}

func lockHeartbeat(lockPath string, alive *int32) {
	t := time.NewTicker(60 * time.Second)
	defer t.Stop()
	for atomic.LoadInt32(alive) == 1 {
		<-t.C
		now := time.Now()
		_ = os.Chtimes(lockPath, now, now)
	}
}

// ───────── Embedded metrics server ─────────

func startMetrics(addr string, m *Metrics) {
	if addr == "" {
		return
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		m.mu.Lock()
		defer m.mu.Unlock()
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		fmt.Fprintf(w, "# HELP ingest_http_requests_total Total adapter requests\n")
		fmt.Fprintf(w, "# TYPE ingest_http_requests_total counter\n")
		for code, n := range m.reqTotalByCode {
			fmt.Fprintf(w, "ingest_http_requests_total{code=\"%d\"} %d\n", code, n)
		}
		fmt.Fprintf(w, "# HELP ingest_http_latency_ms_p50 50th percentile latency\n# TYPE ingest_http_latency_ms_p50 gauge\ningest_http_latency_ms_p50 %f\n", m.p50ms)
		fmt.Fprintf(w, "# HELP ingest_http_latency_ms_p95 95th percentile latency\n# TYPE ingest_http_latency_ms_p95 gauge\ningest_http_latency_ms_p95 %f\n", m.p95ms)
		fmt.Fprintf(w, "# HELP ingest_queue_depth Jobs waiting in queue\n# TYPE ingest_queue_depth gauge\ningest_queue_depth %d\n", m.queueDepth)
		fmt.Fprintf(w, "# HELP ingest_inflight Current in-flight fetches\n# TYPE ingest_inflight gauge\ningest_inflight %d\n", m.inflight)
		fmt.Fprintf(w, "# HELP ingest_tuner_window Current concurrency window\n# TYPE ingest_tuner_window gauge\ningest_tuner_window %d\n", m.tunerWindow)
		fmt.Fprintf(w, "# HELP ingest_rps_tokens_available Current RPS bucket tokens available\n# TYPE ingest_rps_tokens_available gauge\ningest_rps_tokens_available %f\n", m.rpsTokensAvail)
		fmt.Fprintf(w, "# HELP ingest_errors_total Fetch/parse errors\n# TYPE ingest_errors_total counter\ningest_errors_total{type=\"fetch\"} %d\n", m.fetchErrors)
		fmt.Fprintf(w, "ingest_errors_total{type=\"parse\"} %d\n", m.parseErrors)
	})

	// pprof
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("/debug/pprof/trace", pprof.Trace)

	go func() {
		_ = http.ListenAndServe(addr, mux)
	}()
}

// ───────── Direct-to-Postgres helpers ─────────

func mustOpenPool(ctx context.Context, dsn string, maxConns int, viaBouncer bool) *pgxpool.Pool {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		fmt.Fprintln(os.Stderr, "PG_DSN parse:", err)
		os.Exit(2)
	}
	if maxConns <= 0 {
		maxConns = 2
	}
	cfg.MaxConns = int32(maxConns)
	if viaBouncer {
		cfg.ConnConfig.DefaultQueryExecMode = pgx.QueryExecModeSimpleProtocol
	}
	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, "PG connect:", err)
		os.Exit(2)
	}
	return pool
}

func parseTimePtrRFC3339(s string) *time.Time {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return nil
	}
	return &t
}

func parseNullableFloat(s string) *float64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return nil
	}
	return &f
}

func parseNullableIntPtr(i int) *int {
	if i == 0 {
		return nil
	}
	return &i
}

func insertRowsDB(ctx context.Context, pool *pgxpool.Pool, schema string, rows []*Row, batch int) (int, error) {
	if len(rows) == 0 {
		return 0, nil
	}
	if batch <= 0 {
		batch = 200
	}
	total := 0
	table := fmt.Sprintf(`"%s".marketplace_listings`, schema)

	for i := 0; i < len(rows); i += batch {
		j := i + batch
		if j > len(rows) {
			j = len(rows)
		}
		b := &pgx.Batch{}
		count := 0
		for _, r := range rows[i:j] {
			if strings.TrimSpace(r.ListingID) == "" {
				continue
			}

			updated := parseTimePtrRFC3339(r.UpdatedAt)
			firstSeen := parseTimePtrRFC3339(r.FirstSeen)
			lastSeen := parseTimePtrRFC3339(r.LastSeen)
			lastFetched := parseTimePtrRFC3339(r.LastFetched)

			attrNum1 := parseNullableIntPtr(r.AttributeNum1)
			score := parseNullableFloat(r.Score)
			pricePerUnit := parseNullableFloat(r.PricePerUnit)

			postalDigits := digitOnly(r.PostalCode)

			b.Queue(
				`INSERT INTO `+table+`
				(listing_id, title, price, url, description, updated_at,
				 location_city, postal_code, attribute_num_1, attribute_text_1, attribute_text_2,
				 category, score, price_per_unit, status, first_seen, last_seen, last_fetched)
				VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)
				ON CONFLICT (listing_id) DO NOTHING`,
				r.ListingID, r.Title, r.Price, r.URL, r.Description, updated,
				r.LocationCity, postalDigits, attrNum1, r.AttributeText1, r.AttributeText2,
				r.Category, score, pricePerUnit, r.Status, firstSeen, lastSeen, lastFetched,
			)
			count++
		}
		br := pool.SendBatch(ctx, b)
		for k := 0; k < count; k++ {
			tag, err := br.Exec()
			if err != nil {
				_ = br.Close()
				return total, err
			}
			total += int(tag.RowsAffected())
		}
		if err := br.Close(); err != nil {
			return total, err
		}
	}
	return total, nil
}

// ───────── Orchestration (concurrent search → auto-tuned detail) ─────────

type detailJob struct {
	listingID string
}

type result struct{ row *Row }

func produceDetailJobs(
	ctx context.Context,
	cfg config,
	adapter MarketplaceAdapter,
	stats *Stats,
	cool Cooloff,
	ids map[string]struct{},
	m *Metrics,
) <-chan detailJob {

	out := make(chan detailJob, max(64, cfg.workers*2))

	go func() {
		defer close(out)
		if cfg.pages <= 0 {
			return
		}

		type pageOut struct {
			page int
			ids  []string
		}
		pageCh := make(chan pageOut, cfg.pages)
		var wg sync.WaitGroup
		sem := make(chan struct{}, cfg.searchWorkers)

		for p := 1; p <= cfg.pages && atomic.LoadInt32(&stopRequested) == 0; p++ {
			sem <- struct{}{}
			wg.Add(1)
			go func(page int) {
				defer wg.Done()
				defer func() { <-sem }()

				ctxPage, cancel := context.WithTimeout(ctx, 25*time.Second)
				defer cancel()

				summ, meta, err := adapter.SearchListings(ctxPage, SearchParams{
					Query:    cfg.searchQuery,
					Page:     page,
					MinPrice: cfg.minPrice,
				})
				m.RecordRequest(meta.StatusCode, float64(meta.Latency.Milliseconds()))
				if err != nil {
					if meta.StatusCode == 429 {
						time.Sleep(randFloat(1.0, 2.0))
						maybeGlobalCooloff(stats, cool)
					}
					stats.inc(&stats.FetchErrors, 1)
					m.RecordErrors(true, false)
					pageCh <- pageOut{page: page, ids: nil}
					return
				}
				stats.reset429()

				// Extract IDs
				idsOut := make([]string, 0, len(summ))
				for _, s := range summ {
					id := strings.TrimSpace(s.ListingID)
					if id == "" {
						continue
					}
					idsOut = append(idsOut, id)
				}
				pageCh <- pageOut{page: page, ids: idsOut}
			}(p)
		}

		go func() { wg.Wait(); close(pageCh) }()

		seenIDs := make(map[string]struct{}, 2048)
		for po := range pageCh {
			_ = po.page
			for _, id := range po.ids {
				if atomic.LoadInt32(&stopRequested) == 1 {
					return
				}
				if _, ok := ids[id]; ok {
					continue
				}
				if _, ok := seenIDs[id]; ok {
					continue
				}
				seenIDs[id] = struct{}{}
				select {
				case out <- detailJob{listingID: id}:
					m.SetQueueDepth(len(out))
				case <-time.After(2 * time.Second):
					return
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	return out
}

func consumeDetails(
	ctx context.Context,
	cfg config,
	adapter MarketplaceAdapter,
	stats *Stats,
	cool Cooloff,
	jobs <-chan detailJob,
	m *Metrics,
	tuner *AutoTuner,
	gate *ConcurrencyGate,
	bucket *TokenBucket,
) (fresh []*Row, candidates int, tried int) {

	var tried64 int64
	results := make(chan result, max(64, cfg.workers*2))
	var wg sync.WaitGroup

	for i := 0; i < cfg.workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for jb := range jobs {
				atomic.AddInt64(&tried64, 1)
				if atomic.LoadInt32(&stopRequested) == 1 {
					return
				}
				id := strings.TrimSpace(jb.listingID)
				if id == "" {
					continue
				}

				ctxReq, cancel := context.WithTimeout(ctx, 30*time.Second)
				if !gate.Acquire(ctxReq) {
					cancel()
					return
				}
				m.SetInflight(gate.Inflight())

				if !bucket.Take(ctxReq, m) {
					gate.Release()
					cancel()
					return
				}

				start := time.Now()
				detail, meta, err := adapter.FetchListing(ctxReq, id)
				latMs := float64(time.Since(start).Milliseconds())
				if meta.Latency > 0 {
					latMs = float64(meta.Latency.Milliseconds())
				}
				code := meta.StatusCode
				if code == 0 {
					// Adapters should provide status, but default to "unknown error".
					code = 520
				}
				m.RecordRequest(code, latMs)

				cancel()
				gate.Release()
				m.SetInflight(gate.Inflight())

				if err != nil {
					if meta.StatusCode == 429 {
						time.Sleep(randFloat(1.0, 2.0))
						maybeGlobalCooloff(stats, cool)
					}
					stats.inc(&stats.FetchErrors, 1)
					m.RecordErrors(true, false)
					continue
				}
				stats.reset429()

				row, why := normalizeToRow(detail, cfg)
				if row == nil {
					switch why {
					case "below_min_price":
						stats.inc(&stats.BelowMin, 1)
					default:
						stats.inc(&stats.ParseErrors, 1)
					}
					m.RecordErrors(false, true)
					continue
				}

				select {
				case results <- result{row: row}:
				case <-time.After(2 * time.Second):
				}
			}
		}()
	}

	// tuner loop
	go func() {
		t := time.NewTicker(defaultEvalEvery)
		defer t.Stop()
		for {
			select {
			case <-t.C:
				tuner.Recalc()
			case <-ctx.Done():
				return
			}
		}
	}()

	go func() {
		wg.Wait()
		close(results)
	}()

	for r := range results {
		if r.row != nil {
			candidates++
			fresh = append(fresh, r.row)
		}
	}
	return fresh, candidates, int(atomic.LoadInt64(&tried64))
}

func normalizeToRow(d ListingDetails, cfg config) (*Row, string) {
	id := strings.TrimSpace(d.ListingID)
	if id == "" {
		return nil, "missing_id"
	}
	price := d.Price
	if cfg.minPrice > 0 && price > 0 && price < cfg.minPrice {
		return nil, "below_min_price"
	}

	now := nowISO()
	attrText1 := ""
	attrText2 := ""
	category := ""
	score := ""
	pricePerUnit := ""

	if d.Attributes != nil {
		attrText1 = strings.TrimSpace(d.Attributes["attribute_text_1"])
		attrText2 = strings.TrimSpace(d.Attributes["attribute_text_2"])
		category = strings.TrimSpace(d.Attributes["category"])
		score = strings.TrimSpace(d.Attributes["score"])
		pricePerUnit = strings.TrimSpace(d.Attributes["price_per_unit"])
	}

	postal := fixPostalCode(d.PostalCode, (cfg.pgDSN == "" && !cfg.rawPostal))

	return &Row{
		Title:          strings.TrimSpace(d.Title),
		Price:          price,
		URL:            strings.TrimSpace(d.URL),
		Description:    strings.TrimSpace(d.Description),
		UpdatedAt:      strings.TrimSpace(d.UpdatedAtISO),
		LocationCity:   strings.TrimSpace(d.LocationCity),
		PostalCode:     postal,
		AttributeNum1:  0,
		LastFetched:    now,
		Status:         "live",
		FirstSeen:      now,
		LastSeen:       now,
		ListingID:      id,
		AttributeText1: attrText1,
		AttributeText2: attrText2,
		Category:       category,
		Score:          score,
		PricePerUnit:   pricePerUnit,
	}, "ok"
}

// ───────── Main scrape pass ─────────

type summary struct {
	Pages       int
	Candidates  int
	ParsedOK    int
	Inserted    int
	Errors      int
	DurationSec float64
}

func scrapeOnce(ctx context.Context, cfg config, adapter MarketplaceAdapter, m *Metrics) summary {
	start := time.Now()

	outCSV := cfg.out
	outIDs := outCSV + ".ids"
	lockPath = outCSV + ".lock"

	useCSV := (cfg.pgDSN == "")
	var ids map[string]struct{}
	var pool *pgxpool.Pool

	if useCSV {
		if err := ensureCSVHeader(outCSV); err != nil {
			fmt.Fprintln(os.Stderr, "csv header:", err)
			os.Exit(2)
		}
		ids = ensureIDsIndex(outCSV, outIDs)
		if cfg.pages == 0 {
			fmt.Println("healthcheck=ok")
			return summary{}
		}

		if !acquireLock(lockPath, time.Duration(LOCK_TTL_SECS)*time.Second) {
			os.Exit(1)
		}
		atomic.StoreInt32(&lockHeld, 1)
		defer func() { releaseLock(lockPath); atomic.StoreInt32(&lockHeld, 0) }()
		go lockHeartbeat(lockPath, &lockHeld)
	} else {
		ids = make(map[string]struct{})
		if cfg.pages == 0 {
			fmt.Println("healthcheck=ok")
			return summary{}
		}
		pool = mustOpenPool(ctx, cfg.pgDSN, cfg.pgMaxConns, cfg.pgViaBouncer)
		defer pool.Close()
	}

	stats := &Stats{}
	cool := Cooloff{
		Threshold: max(1, 10),
		Min:       5 * time.Second,
		Max:       10 * time.Second,
	}

	minW := max(4, cfg.workers/8)
	gate := NewConcurrencyGate(cfg.workers)
	m.SetTunerWindow(gate.Window())
	tuner := NewAutoTuner(gate, minW, cfg.workers, cfg.sloMult, cfg.max429Rate, defaultEvalEvery, m)

	var bucket *TokenBucket
	if cfg.rps > 0 {
		bucket = NewTokenBucket(cfg.rps, 0.10)
	}

	jobs := produceDetailJobs(ctx, cfg, adapter, stats, cool, ids, m)
	fresh, candidates, tried := consumeDetails(ctx, cfg, adapter, stats, cool, jobs, m, tuner, gate, bucket)
	parsedOK := len(fresh)

	toWrite := make([]*Row, 0, len(fresh))
	newIDs := make([]string, 0, len(fresh))
	seenNew := make(map[string]struct{}, len(fresh))

	for _, r := range fresh {
		if atomic.LoadInt32(&stopRequested) == 1 {
			break
		}
		id := strings.TrimSpace(r.ListingID)
		if id == "" {
			continue
		}
		if _, ok := ids[id]; ok {
			continue
		}
		if _, ok := seenNew[id]; ok {
			continue
		}
		toWrite = append(toWrite, r)
		newIDs = append(newIDs, id)
		seenNew[id] = struct{}{}
	}

	if atomic.LoadInt32(&stopRequested) == 1 {
		os.Exit(2)
	}

	inserted := 0
	if useCSV {
		if err := appendRowsCSV(cfg.out, toWrite); err != nil {
			fmt.Fprintln(os.Stderr, "csv append:", err)
			os.Exit(2)
		}
		if err := appendIDs(cfg.out+".ids", newIDs); err != nil {
			fmt.Fprintln(os.Stderr, "ids append:", err)
			os.Exit(2)
		}
		inserted = len(toWrite)
	} else {
		var err error
		inserted, err = insertRowsDB(ctx, pool, cfg.pgSchema, toWrite, cfg.pgBatch)
		if err != nil {
			fmt.Fprintln(os.Stderr, "db insert:", err)
			os.Exit(2)
		}
	}

	errorCount := func() int {
		stats.mu.Lock()
		defer stats.mu.Unlock()
		return stats.FetchErrors + stats.ParseErrors
	}()

	m.mu.Lock()
	http200 := int(m.reqTotalByCode[200])
	m.mu.Unlock()

	dur := time.Since(start).Seconds()

	fmt.Printf(
		"adapter=%s pages=%d details=%d parsed_ok=%d inserted=%d http_200=%d skipped_below_min=%d errors_fetch=%d errors_parse=%d duration=%0.2f\n",
		cfg.adapter, cfg.pages, tried, parsedOK, inserted, http200, stats.BelowMin, stats.FetchErrors, stats.ParseErrors, dur,
	)

	if cfg.jsonLogs {
		type js struct {
			Event       string  `json:"event"`
			Adapter     string  `json:"adapter"`
			Query       string  `json:"query"`
			Pages       int     `json:"pages"`
			Details     int     `json:"details"`
			Candidates  int     `json:"candidates"`
			ParsedOK    int     `json:"parsed_ok"`
			Inserted    int     `json:"inserted"`
			Errors      int     `json:"errors"`
			ErrorsFetch int     `json:"errors_fetch"`
			ErrorsParse int     `json:"errors_parse"`
			HTTP200     int     `json:"http_200"`
			DurationSec float64 `json:"duration_sec"`
			GoMaxProcs  int     `json:"gomaxprocs"`
			NumCPU      int     `json:"num_cpu"`
			TunerWindow int     `json:"tuner_window"`
			HTTPp95ms   float64 `json:"http_p95_ms"`
			HTTPp50ms   float64 `json:"http_p50_ms"`
			Daemon      bool    `json:"daemon"`
		}
		p50, p95 := m.SnapshotLatencies()
		j := js{
			Event:       "summary",
			Adapter:     cfg.adapter,
			Query:       cfg.searchQuery,
			Pages:       cfg.pages,
			Details:     tried,
			Candidates:  candidates,
			ParsedOK:    parsedOK,
			Inserted:    inserted,
			Errors:      errorCount,
			ErrorsFetch: stats.FetchErrors,
			ErrorsParse: stats.ParseErrors,
			HTTP200:     http200,
			DurationSec: round2(dur),
			GoMaxProcs:  runtime.GOMAXPROCS(0),
			NumCPU:      runtime.NumCPU(),
			TunerWindow: gate.Window(),
			HTTPp95ms:   p95,
			HTTPp50ms:   p50,
			Daemon:      cfg.daemon,
		}
		b, _ := json.Marshal(j)
		fmt.Println(string(b))
	}

	if cfg.printNew && len(newIDs) > 0 {
		for _, id := range newIDs {
			for _, r := range toWrite {
				if r.ListingID == id {
					fmt.Printf("%s :: %s\n", id, r.Title)
					break
				}
			}
		}
	}

	return summary{
		Pages:       cfg.pages,
		Candidates:  candidates,
		ParsedOK:    parsedOK,
		Inserted:    inserted,
		Errors:      errorCount,
		DurationSec: dur,
	}
}

func randFloat(min, max float64) time.Duration {
	sec := min + rand.Float64()*(max-min)
	return time.Duration(sec * float64(time.Second))
}

// ───────── helpers ─────────

func min(a, b int) int { if a < b { return a }; return b }
func max(a, b int) int { if a > b { return a }; return b }
func round2(x float64) float64 { return math.Round(x*100) / 100 }

// ───────── Adapter selection ─────────

func buildAdapter(cfg config) MarketplaceAdapter {
	switch strings.ToLower(strings.TrimSpace(cfg.adapter)) {
	case "http-json", "httpjson", "http":
		a, err := NewHTTPJSONAdapter(HTTPJSONAdapterOptions{
			BaseURL: cfg.marketplaceBase,
			// The user agent is intentionally generic and can be overridden by env vars in a private fork.
			UserAgent: envString("HTTP_USER_AGENT", "marketplace-ingest-template/1.0"),
			Timeout:   25 * time.Second,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, "adapter init failed; falling back to mock:", err)
			return NewMockAdapter(MockAdapterOptions{BaseURL: cfg.marketplaceBase})
		}
		return a
	default:
		return NewMockAdapter(MockAdapterOptions{BaseURL: cfg.marketplaceBase})
	}
}

// ───────── Main ─────────

func main() {
	rand.Seed(time.Now().UnixNano())
	cfg := parseFlags()
	adapter := buildAdapter(cfg)

	// Signals
	sigc := make(chan os.Signal, 2)
	signal.Notify(sigc, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigc
		atomic.StoreInt32(&stopRequested, 1)
	}()

	metrics := NewMetrics(cfg.p95Window)
	startMetrics(cfg.metricsAddr, metrics)

	if !cfg.daemon {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		_ = scrapeOnce(ctx, cfg, adapter, metrics)
		return
	}

	minSleep := time.Duration(max(1, cfg.daemonMinSec)) * time.Second
	maxSleep := time.Duration(max(cfg.daemonMinSec, cfg.daemonMaxSec)) * time.Second
	if maxSleep < minSleep {
		maxSleep = minSleep
	}

	for atomic.LoadInt32(&stopRequested) == 0 {
		ctx, cancel := context.WithCancel(context.Background())
		_ = scrapeOnce(ctx, cfg, adapter, metrics)
		cancel()

		span := maxSleep - minSleep
		j := time.Duration(rand.Int63n(int64(span)))
		sleep := minSleep + j
		select {
		case <-time.After(sleep):
		case <-sigc:
			atomic.StoreInt32(&stopRequested, 1)
		}
	}
}