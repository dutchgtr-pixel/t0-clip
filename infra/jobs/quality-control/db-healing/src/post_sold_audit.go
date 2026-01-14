// post_sold_audit_public.go
//
// Public-release template version of a post-sale snapshot auditor.
//
// This program:
//   - Selects sold listings from Postgres (schema/table names configurable).
//   - Enforces a minimum age (default 7 days) before taking a snapshot.
//   - Fetches listing payloads via a pluggable marketplace adapter (no site-specific scraping).
//   - Inserts one audit snapshot per listing_id/day_offset.
//
// Intentionally omitted from the public version:
//   - Any target-site identifiers, URLs, headers, cookies, or browser-fingerprinting logic.
//   - Any target-site HTML parsing (selectors, hydration keys, etc.).
//
// Adapters:
//   - mock: offline synthetic payloads (default; safe for local dev/tests).
//   - http-json: GET {MARKETPLACE_BASE_URL}/api/listings/{listing_id} (expects JSON).
//
// Build:
//   go build -trimpath -ldflags="-s -w" -o post_sold_audit ./post_sold_audit_public.go
package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"golang.org/x/time/rate"
)

/* =========================
   CLI / configuration
   ========================= */

var (
	mode = flag.String("mode", getenv("MODE", "run"), "init|seed|run|refresh-desc")

	dsn    = flag.String("dsn", getenv("PG_DSN", ""), "Postgres DSN (or set PG_DSN)")
	schema = flag.String("schema", getenv("PG_SCHEMA", "marketplace"), "Postgres schema name")

	listingsTable = flag.String("listings-table", getenv("LISTINGS_TABLE", "listings"), "Listings table name")
	auditTable    = flag.String("audit-table", getenv("AUDIT_TABLE", "post_sold_audit"), "Audit table name")

	// Runtime behavior
	workers      = flag.Int("workers", getenvInt("WORKERS", 32), "parallel workers")
	timeout      = flag.Duration("timeout", getenvDuration("REQUEST_TIMEOUT", 15*time.Second), "request timeout")
	dryRun       = flag.Bool("dry", getenvBool("DRY_RUN", false), "log only; do not insert/update")
	maxConnsHost = flag.Int("max-conns-per-host", getenvInt("MAX_CONNS_PER_HOST", 6), "HTTP MaxConnsPerHost (adapter=http-json)")

	// Selection
	segmentsCSV   = flag.String("segments", getenv("SEGMENTS", ""), "optional segment/group CSV filter (empty = all)")
	perSegCap     = flag.Int("per-segment-cap", getenvInt("PER_SEGMENT_CAP", 0), "per-segment cap of not-yet-audited rows (0=unlimited)")
	batchRows     = flag.Int("batch", getenvInt("BATCH", 0), "global batch cap across segments (0=unlimited)")
	onlyIDsCSV    = flag.String("only-ids", getenv("ONLY_IDS", ""), "only these listing_id values (CSV)")
	minAgeDays    = flag.Int("min-age-days", getenvInt("MIN_AGE_DAYS", 7), "minimum age (days) since sold_date before snapshot")
	dayOffset     = flag.Int("day-offset", getenvInt("DAY_OFFSET", 7), "stored day_offset value (commonly equals min-age-days)")

	// Adaptive RPS (AIMD)
	rpsStart       = flag.Float64("rps", getenvFloat("REQUEST_RPS", 3.0), "initial global requests/sec (adaptive)")
	rpsMax         = flag.Float64("rps-max", getenvFloat("REQUEST_RPS_MAX", 12.0), "max global requests/sec (adaptive ceiling)")
	rpsMin         = flag.Float64("rps-min", getenvFloat("REQUEST_RPS_MIN", 0.7), "min global requests/sec (adaptive floor)")
	incEveryOK     = flag.Int("inc-every-ok", getenvInt("REQUEST_OK_EVERY", 64), "additive increase after N OK responses")
	incStep        = flag.Float64("inc-step", getenvFloat("REQUEST_RPS_STEP", 0.5), "RPS additive step")
	mulOnThrottle  = flag.Float64("mul-on-throttle", getenvFloat("REQUEST_RPS_DOWN_MULT", 0.5), "RPS multiplier on throttle")
	globalCoolOff  = flag.Duration("cool-off", getenvDuration("REQUEST_COOL_OFF", 25*time.Second), "global cool-off after throttle")
	jitterMs       = flag.Int("jitter-ms", getenvInt("REQUEST_JITTER_MS", 120), "random jitter per request in ms (0=off)")
	retryMax       = flag.Int("retry-max", getenvInt("REQUEST_RETRY_MAX", 2), "max retries on 429/403/408/5xx")
	backoffInitial = flag.Duration("backoff-initial", getenvDuration("REQUEST_BACKOFF_INITIAL", 2*time.Second), "first backoff sleep")
	backoffMax     = flag.Duration("backoff-max", getenvDuration("REQUEST_BACKOFF_MAX", 20*time.Second), "max backoff sleep")

	// Adapter configuration (no target identifiers)
	adapterName = flag.String("adapter", getenv("MARKETPLACE_ADAPTER", "mock"), "marketplace adapter: mock|http-json")
	baseURL     = flag.String("base-url", getenv("MARKETPLACE_BASE_URL", "https://example-marketplace.invalid"), "adapter=http-json base URL")
	authHeader  = flag.String("auth-header", getenv("MARKETPLACE_AUTH_HEADER", ""), "optional Authorization header value (secret)")
	userAgent   = flag.String("http-ua", getenv("HTTP_USER_AGENT", "marketplace-audit-template/1.0"), "adapter=http-json User-Agent")

	// Refresh mode (description only)
	refreshThreshold    = flag.Int("refresh-threshold", getenvInt("REFRESH_THRESHOLD", 160), "only refresh rows with length(description_snapshot) <= N; 0 = ignore")
	refreshIDsCSV       = flag.String("refresh-ids", getenv("REFRESH_IDS", ""), "comma-separated listing_id values to refresh (overrides threshold)")
	refreshAll          = flag.Bool("refresh-all", getenvBool("REFRESH_ALL", false), "refresh all rows (ignores threshold)")
	refreshUpdateHash   = flag.Bool("refresh-update-hash", getenvBool("REFRESH_UPDATE_HASH", false), "also recompute title_desc_hash using existing title + new desc")
	refreshAllSnapshots = flag.Bool("refresh-all-snapshots", getenvBool("REFRESH_ALL_SNAPSHOTS", false), "update all snapshots per listing_id (default: only latest per listing_id)")
	refreshLimit        = flag.Int("refresh-limit", getenvInt("REFRESH_LIMIT", 0), "cap how many rows to refresh (0 = no cap)")
)

func nowts() string { return time.Now().Format("15:04:05.000") }
func pfx(listingID int64, seg int, day int16) string {
	return fmt.Sprintf("[%s] row listing_id=%d segment=%d day=%d", nowts(), listingID, seg, day)
}

/* =========================
   Adaptive gate (AIMD + cool-off)
   ========================= */

type adaptiveGate struct {
	mu        sync.Mutex
	lim       *rate.Limiter
	curr      rate.Limit
	min, max  rate.Limit
	incOK     int
	incStep   rate.Limit
	okCount   int
	coolUntil time.Time
}

func newAdaptiveGate(start, min, max, incStep float64, incEveryOK int) *adaptiveGate {
	if start < min {
		start = min
	}
	if incEveryOK <= 0 {
		incEveryOK = 1
	}
	ag := &adaptiveGate{
		lim:     rate.NewLimiter(rate.Limit(start), 1),
		curr:    rate.Limit(start),
		min:     rate.Limit(min),
		max:     rate.Limit(max),
		incOK:   incEveryOK,
		incStep: rate.Limit(incStep),
	}
	return ag
}

func (g *adaptiveGate) wait(ctx context.Context, jitter int) error {
	g.mu.Lock()
	cool := g.coolUntil
	lim := g.lim
	g.mu.Unlock()

	now := time.Now()
	if cool.After(now) {
		d := time.Until(cool)
		if d > 0 {
			time.Sleep(d + time.Duration(rand.Intn(100))*time.Millisecond)
		}
	}
	if jitter > 0 {
		time.Sleep(time.Duration(rand.Intn(jitter)) * time.Millisecond)
	}
	return lim.Wait(ctx)
}

func (g *adaptiveGate) onOK() {
	g.mu.Lock()
	g.okCount++
	if g.okCount >= g.incOK {
		g.okCount = 0
		newVal := g.curr + g.incStep
		if newVal > g.max {
			newVal = g.max
		}
		if newVal != g.curr {
			g.curr = newVal
			g.lim.SetLimit(g.curr)
		}
	}
	g.mu.Unlock()
}

func (g *adaptiveGate) onThrottle(mult float64, coolOff time.Duration) {
	if mult <= 0 || mult >= 1 {
		mult = 0.5
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	n := rate.Limit(float64(g.curr) * mult)
	if n < g.min {
		n = g.min
	}
	if n != g.curr {
		g.curr = n
		g.lim.SetLimit(g.curr)
	}
	g.okCount = 0
	g.coolUntil = time.Now().Add(coolOff)
}

/* =========================
   Marketplace adapter layer (required for public release)
   ========================= */

type ListingSnapshot struct {
	ListingID    int64          `json:"listing_id"`
	Status       string         `json:"status,omitempty"`
	Title        string         `json:"title,omitempty"`
	Description  string         `json:"description,omitempty"`
	Price        int            `json:"price,omitempty"`
	Currency     string         `json:"currency,omitempty"`
	Attributes   map[string]any `json:"attributes,omitempty"`
	ObservedAt   *time.Time     `json:"observed_at,omitempty"`
	Source       string         `json:"source,omitempty"`
	SourceDetail string         `json:"source_detail,omitempty"`
}

type MarketplaceAdapter interface {
	Name() string
	FetchListing(ctx context.Context, listingID int64) (raw []byte, httpStatus int, err error)
	ParsePayload(raw []byte) (ListingSnapshot, error)
	SearchListings(ctx context.Context, params map[string]string) ([]int64, error) // optional/stub
}

type mockAdapter struct{}

func (a *mockAdapter) Name() string { return "mock" }

func (a *mockAdapter) FetchListing(ctx context.Context, listingID int64) ([]byte, int, error) {
	// Offline synthetic payload; deterministic enough for tests.
	_ = ctx
	now := time.Now().UTC()
	price := 10000 + int(listingID%5000)
	status := "sold"
	payload := ListingSnapshot{
		ListingID:    listingID,
		Status:       status,
		Title:        fmt.Sprintf("Example Listing %d", listingID),
		Description:  "Synthetic description for public release (no real data).",
		Price:        price,
		Currency:     "USD",
		Attributes:   map[string]any{"synthetic": true},
		ObservedAt:   &now,
		Source:       "mock",
		SourceDetail: "generated",
	}
	b, _ := json.Marshal(payload)
	return b, http.StatusOK, nil
}

func (a *mockAdapter) ParsePayload(raw []byte) (ListingSnapshot, error) {
	var s ListingSnapshot
	if err := json.Unmarshal(raw, &s); err != nil {
		return ListingSnapshot{}, err
	}
	return s, nil
}

func (a *mockAdapter) SearchListings(ctx context.Context, params map[string]string) ([]int64, error) {
	// Not used by this job; included for interface completeness.
	_ = ctx
	_ = params
	return nil, nil
}

type httpJSONAdapter struct {
	baseURL    string
	authHeader string
	userAgent  string
	client     *http.Client
}

func (a *httpJSONAdapter) Name() string { return "http-json" }

func newHTTPJSONAdapter(timeout time.Duration, maxConnsPerHost int, baseURL, authHeader, userAgent string) (*httpJSONAdapter, error) {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if baseURL == "" {
		return nil, errors.New("MARKETPLACE_BASE_URL/base-url is required for adapter=http-json")
	}
	if userAgent == "" {
		userAgent = "marketplace-audit-template/1.0"
	}

	tr := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		MaxIdleConns:          256,
		MaxConnsPerHost:       maxConnsPerHost,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		DialContext:           (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
	}
	return &httpJSONAdapter{
		baseURL:    baseURL,
		authHeader: strings.TrimSpace(authHeader),
		userAgent:  userAgent,
		client:     &http.Client{Timeout: timeout, Transport: tr},
	}, nil
}

func (a *httpJSONAdapter) FetchListing(ctx context.Context, listingID int64) ([]byte, int, error) {
	u := fmt.Sprintf("%s/api/listings/%d", a.baseURL, listingID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", a.userAgent)
	if a.authHeader != "" {
		// Treat as secret; do not log.
		req.Header.Set("Authorization", a.authHeader)
	}

	resp, err := a.client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()

	b, rerr := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if rerr != nil {
		return nil, resp.StatusCode, rerr
	}
	return b, resp.StatusCode, nil
}

func (a *httpJSONAdapter) ParsePayload(raw []byte) (ListingSnapshot, error) {
	// Expected payload shape (example):
	// {
	//   "listing_id": 123,
	//   "status": "sold",
	//   "title": "Example",
	//   "description": "Example",
	//   "price": 12345,
	//   "currency": "USD",
	//   "attributes": {"key": "value"}
	// }
	var s ListingSnapshot
	if err := json.Unmarshal(raw, &s); err != nil {
		return ListingSnapshot{}, err
	}
	return s, nil
}

func (a *httpJSONAdapter) SearchListings(ctx context.Context, params map[string]string) ([]int64, error) {
	// Stub: implement in your private connector if you need it.
	_ = ctx
	_ = params
	return nil, nil
}

func buildAdapter() (MarketplaceAdapter, error) {
	switch strings.ToLower(strings.TrimSpace(*adapterName)) {
	case "mock", "":
		return &mockAdapter{}, nil
	case "http-json":
		return newHTTPJSONAdapter(*timeout, *maxConnsHost, *baseURL, *authHeader, *userAgent)
	default:
		return nil, fmt.Errorf("unknown adapter %q (expected mock|http-json)", *adapterName)
	}
}

/* =========================
   Fetcher (rate limiting + retries)
   ========================= */

type fetcher struct {
	adapter        MarketplaceAdapter
	gate           *adaptiveGate
	retryMax       int
	backoffInitial time.Duration
	backoffMax     time.Duration
	jitterMs       int
}

func newFetcher(adapter MarketplaceAdapter, gate *adaptiveGate, retryMax int, backoffInitial, backoffMax time.Duration, jitterMs int) *fetcher {
	return &fetcher{
		adapter:        adapter,
		gate:           gate,
		retryMax:       retryMax,
		backoffInitial: backoffInitial,
		backoffMax:     backoffMax,
		jitterMs:       jitterMs,
	}
}

func (f *fetcher) sleepBackoff(attempt int) {
	d := f.backoffInitial * time.Duration(1<<attempt)
	if d > f.backoffMax {
		d = f.backoffMax
	}
	time.Sleep(d + time.Duration(rand.Intn(250))*time.Millisecond)
}

func (f *fetcher) fetch(ctx context.Context, listingID int64) (raw []byte, httpStatus int, err error) {
	var lastErr error
	var status int
	for attempt := 0; attempt <= f.retryMax; attempt++ {
		if err := f.gate.wait(ctx, f.jitterMs); err != nil {
			return nil, 0, err
		}
		b, st, err := f.adapter.FetchListing(ctx, listingID)
		status = st
		if err != nil {
			lastErr = err
			// Treat transport errors as throttling signals.
			f.gate.onThrottle(*mulOnThrottle, *globalCoolOff)
			if attempt < f.retryMax {
				f.sleepBackoff(attempt)
				continue
			}
			return nil, status, err
		}

		// Backoff on common throttling/transient codes.
		if status == 429 || status == 403 || status == 408 || (status >= 500 && status <= 599) {
			lastErr = fmt.Errorf("http %d", status)
			f.gate.onThrottle(*mulOnThrottle, *globalCoolOff)
			if attempt < f.retryMax {
				f.sleepBackoff(attempt)
				continue
			}
			return b, status, lastErr
		}

		f.gate.onOK()
		return b, status, nil
	}
	return nil, status, lastErr
}

/* =========================
   DB selection + insert
   ========================= */

type candidate struct {
	ListingID    int64
	URL          string
	Segment      int
	SoldDateRef  *time.Time
	DayOffset    int16
}

func parseIntCSV(s string) []int {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
	for _, t := range parts {
		t = strings.TrimSpace(t)
		if t == "" {
			continue
		}
		n, err := strconv.Atoi(t)
		if err != nil {
			continue
		}
		out = append(out, n)
	}
	return out
}

func selectCandidates(ctx context.Context, db *pgxpool.Pool) ([]candidate, error) {
	if !isSafeIdent(*schema) || !isSafeIdent(*listingsTable) || !isSafeIdent(*auditTable) {
		return nil, fmt.Errorf("unsafe schema/table identifier(s)")
	}

	minAge := *minAgeDays
	if minAge < 0 {
		minAge = 0
	}
	day := *dayOffset
	if day < 0 {
		day = 0
	}

	idsCSV := strings.TrimSpace(*onlyIDsCSV)
	segs := parseIntCSV(*segmentsCSV)

	// Note: selection is intentionally generic. It expects a listings table with:
	//   listing_id bigint, url text, segment int, status text, sold_date timestamptz
	//
	// It enforces:
	//   - status='sold'
	//   - sold_date age >= min_age_days
	//   - not already audited for this day_offset
	var sql string
	var args []any

	if idsCSV != "" {
		sql = fmt.Sprintf(`
WITH base AS (
  SELECT l.listing_id, l.url, COALESCE(l.segment,0) AS segment, l.sold_date AS sold_date_ref,
         GREATEST(0, floor(EXTRACT(EPOCH FROM (now() - l.sold_date)) / 86400))::int AS age_days
  FROM "%[1]s".%[2]s l
  WHERE l.status='sold'
    AND l.sold_date IS NOT NULL
    AND l.listing_id = ANY(string_to_array($1, ',')::bigint[])
    AND COALESCE(l.url,'') <> ''
),
todo AS (
  SELECT b.listing_id, b.url, b.segment, b.sold_date_ref,
         $3::smallint AS day_offset
  FROM base b
  WHERE b.age_days >= $2
    AND NOT EXISTS (
      SELECT 1 FROM "%[1]s".%[4]s a
      WHERE a.listing_id = b.listing_id
        AND a.day_offset = $3
    )
)
SELECT listing_id, url, segment, sold_date_ref, day_offset
FROM todo
ORDER BY sold_date_ref DESC NULLS LAST, listing_id DESC
`, *schema, *listingsTable, *auditTable)
		if *batchRows > 0 {
			sql += fmt.Sprintf(" LIMIT %d", *batchRows)
		}
		args = []any{idsCSV, minAge, int16(day)}
	} else {
		sql = fmt.Sprintf(`
WITH base AS (
  SELECT l.listing_id, l.url, COALESCE(l.segment,0) AS segment, l.sold_date AS sold_date_ref,
         GREATEST(0, floor(EXTRACT(EPOCH FROM (now() - l.sold_date)) / 86400))::int AS age_days
  FROM "%[1]s".%[2]s l
  WHERE l.status='sold'
    AND l.sold_date IS NOT NULL
    AND COALESCE(l.url,'') <> ''
    AND ($1::int[] IS NULL OR l.segment = ANY($1))
),
todo AS (
  SELECT b.listing_id, b.url, b.segment, b.sold_date_ref,
         $3::smallint AS day_offset
  FROM base b
  WHERE b.age_days >= $2
    AND NOT EXISTS (
      SELECT 1 FROM "%[1]s".%[4]s a
      WHERE a.listing_id = b.listing_id
        AND a.day_offset = $3
    )
),
ranked AS (
  SELECT t.*,
         row_number() OVER (PARTITION BY t.segment
                            ORDER BY t.sold_date_ref DESC NULLS LAST, t.listing_id DESC) AS rn
  FROM todo t
)
SELECT listing_id, url, segment, sold_date_ref, day_offset
FROM ranked
WHERE ($4::int = 0 OR rn <= $4)
ORDER BY sold_date_ref DESC NULLS LAST, listing_id DESC
`, *schema, *listingsTable, *auditTable)
		if *batchRows > 0 {
			sql += fmt.Sprintf(" LIMIT %d", *batchRows)
		}
		var segArg any
		if len(segs) == 0 {
			segArg = nil
		} else {
			segArg = segs
		}
		args = []any{segArg, minAge, int16(day), *perSegCap}
	}

	rows, err := db.Query(ctx, sql, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []candidate
	for rows.Next() {
		var c candidate
		if err := rows.Scan(&c.ListingID, &c.URL, &c.Segment, &c.SoldDateRef, &c.DayOffset); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

type snapshotRow struct {
	Title       string
	Description string
	Price       int
	Currency    string
	Evidence    map[string]any
	ParseErr    string
}

func hashTitleDesc(title, desc string) string {
	h := sha256.Sum256([]byte(title + "\n" + desc))
	return fmt.Sprintf("%x", h[:])
}

func lastTitleHash(ctx context.Context, db *pgxpool.Pool, listingID int64) (string, error) {
	q := fmt.Sprintf(`
SELECT title_desc_hash
FROM "%s".%s
WHERE listing_id=$1
ORDER BY snapshot_at DESC
LIMIT 1`, *schema, *auditTable)
	row := db.QueryRow(ctx, q, listingID)
	var h *string
	if err := row.Scan(&h); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return "", nil
		}
		return "", err
	}
	if h == nil {
		return "", nil
	}
	return *h, nil
}

func jsonStringOrNull(v any) *string {
	if v == nil {
		return nil
	}
	b, _ := json.Marshal(v)
	s := strings.TrimSpace(string(b))
	if s == "" || s == "null" || s == "{}" {
		return nil
	}
	return &s
}

func nullStr(s string) *string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	v := s
	return &v
}

func nullIntPtr(v int) *int {
	if v == 0 {
		return nil
	}
	return &v
}

func insertSnapshot(ctx context.Context, db *pgxpool.Pool, c candidate, httpStatus int, raw []byte, snap snapshotRow) error {
	payloadOK := (httpStatus >= 200 && httpStatus < 300 && len(raw) > 0 && snap.ParseErr == "")

	var title, desc, hash *string
	if payloadOK {
		newHash := hashTitleDesc(snap.Title, snap.Description)
		save := true

		// Optional space optimization: only store title/desc if it changed since last snapshot.
		if last, err := lastTitleHash(ctx, db, c.ListingID); err == nil && last != "" && last == newHash {
			save = false
		}
		if save && strings.TrimSpace(snap.Title) != "" {
			h := newHash
			hash = &h
			t := snap.Title
			d := snap.Description
			title, desc = &t, &d
		}
	}

	diffJSON := jsonStringOrNull(map[string]any{
		"price":    snap.Price,
		"currency": snap.Currency,
	})
	evJSON := jsonStringOrNull(snap.Evidence)
	parseErr := nullStr(snap.ParseErr)

	if *dryRun {
		fmt.Printf("%s [dry-insert] status=%d payload_ok=%v title_len=%d desc_len=%d price=%d\n",
			pfx(c.ListingID, c.Segment, c.DayOffset), httpStatus, payloadOK, len(snap.Title), len(snap.Description), snap.Price)
		return nil
	}

	q := fmt.Sprintf(`
INSERT INTO "%[1]s".%[2]s
(listing_id, day_offset, snapshot_at, sold_date_ref, segment_ref,
 url, http_status, payload_ok,
 title_snapshot, description_snapshot, title_desc_hash,
 price_snapshot, currency_snapshot,
 evidence_json, parse_errors, diff_json)
VALUES
($1,$2,now(),$3,$4,
 $5,$6,$7,
 $8,$9,$10,
 $11,$12,
 CAST($13 AS jsonb), $14, CAST($15 AS jsonb))
ON CONFLICT (listing_id, day_offset) DO NOTHING`, *schema, *auditTable)

	_, err := db.Exec(ctx, q,
		c.ListingID, c.DayOffset, c.SoldDateRef, c.Segment,
		c.URL, httpStatus, payloadOK,
		title, desc, hash,
		nullIntPtr(snap.Price), nullStr(snap.Currency),
		evJSON, parseErr, diffJSON,
	)
	return err
}

/* =========================
   Refresh mode (description only; generic)
   ========================= */

type refreshCand struct {
	ListingID   int64
	Segment     int
	DayOffset   int16
	SnapshotAt  time.Time
	OldLen      int
	Title       *string
}

func loadRefreshList(ctx context.Context, db *pgxpool.Pool) ([]refreshCand, error) {
	if !isSafeIdent(*schema) || !isSafeIdent(*auditTable) {
		return nil, fmt.Errorf("unsafe schema/table identifier(s)")
	}

	ids := strings.TrimSpace(*refreshIDsCSV)
	limitClause := ""
	if *refreshLimit > 0 {
		limitClause = fmt.Sprintf(" LIMIT %d", *refreshLimit)
	}

	// We refresh either:
	//   - explicit listing IDs, or
	//   - rows below a description length threshold, or
	//   - all rows (refresh-all).
	//
	// For simplicity, we select the latest snapshot per listing_id unless refresh-all-snapshots is set.
	var q string
	var rows pgx.Rows
	var err error

	if ids != "" {
		q = fmt.Sprintf(`
WITH sel AS (
  SELECT a.listing_id, a.segment_ref, a.day_offset, a.snapshot_at,
         COALESCE(length(a.description_snapshot), 0) AS old_len,
         a.title_snapshot
  FROM "%[1]s".%[2]s a
  WHERE a.listing_id = ANY(string_to_array($1, ',')::bigint[])
),
latest AS (
  SELECT DISTINCT ON (listing_id) *
  FROM sel
  ORDER BY listing_id, snapshot_at DESC
)
SELECT listing_id, COALESCE(segment_ref,0), day_offset, snapshot_at, old_len, title_snapshot
FROM %[3]s
ORDER BY listing_id
%[4]s
`, *schema, *auditTable, ternary(*refreshAllSnapshots, "sel", "latest"), limitClause)
		rows, err = db.Query(ctx, q, ids)
	} else if *refreshAll {
		q = fmt.Sprintf(`
WITH sel AS (
  SELECT a.listing_id, a.segment_ref, a.day_offset, a.snapshot_at,
         COALESCE(length(a.description_snapshot), 0) AS old_len,
         a.title_snapshot
  FROM "%[1]s".%[2]s a
),
latest AS (
  SELECT DISTINCT ON (listing_id) *
  FROM sel
  ORDER BY listing_id, snapshot_at DESC
)
SELECT listing_id, COALESCE(segment_ref,0), day_offset, snapshot_at, old_len, title_snapshot
FROM %[3]s
ORDER BY listing_id
%[4]s
`, *schema, *auditTable, ternary(*refreshAllSnapshots, "sel", "latest"), limitClause)
		rows, err = db.Query(ctx, q)
	} else {
		th := *refreshThreshold
		if th < 0 {
			th = 0
		}
		q = fmt.Sprintf(`
WITH sel AS (
  SELECT a.listing_id, a.segment_ref, a.day_offset, a.snapshot_at,
         COALESCE(length(a.description_snapshot), 0) AS old_len,
         a.title_snapshot
  FROM "%[1]s".%[2]s a
  WHERE $1 = 0 OR COALESCE(length(a.description_snapshot), 0) <= $1
),
latest AS (
  SELECT DISTINCT ON (listing_id) *
  FROM sel
  ORDER BY listing_id, snapshot_at DESC
)
SELECT listing_id, COALESCE(segment_ref,0), day_offset, snapshot_at, old_len, title_snapshot
FROM %[3]s
ORDER BY listing_id
%[4]s
`, *schema, *auditTable, ternary(*refreshAllSnapshots, "sel", "latest"), limitClause)
		rows, err = db.Query(ctx, q, th)
	}

	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]refreshCand, 0, 1024)
	for rows.Next() {
		var c refreshCand
		if err := rows.Scan(&c.ListingID, &c.Segment, &c.DayOffset, &c.SnapshotAt, &c.OldLen, &c.Title); err != nil {
			return nil, err
		}
		out = append(out, c)
	}
	return out, rows.Err()
}

func refreshDescriptions(ctx context.Context, db *pgxpool.Pool, f *fetcher) error {
	list, err := loadRefreshList(ctx, db)
	if err != nil {
		return err
	}
	fmt.Printf("[%s] [refresh] rows=%d\n", nowts(), len(list))
	if len(list) == 0 {
		return nil
	}

	for _, c := range list {
		prefix := pfx(c.ListingID, c.Segment, c.DayOffset)
		raw, status, err := f.fetch(ctx, c.ListingID)
		if err != nil {
			fmt.Printf("%s [refresh] fetch err=%v status=%d\n", prefix, err, status)
			continue
		}
		snap, perr := f.adapter.ParsePayload(raw)
		if perr != nil {
			fmt.Printf("%s [refresh] parse err=%v\n", prefix, perr)
			continue
		}

		desc := strings.TrimSpace(snap.Description)
		if desc == "" {
			fmt.Printf("%s [refresh] empty description; skipping\n", prefix)
			continue
		}

		var newHash *string
		if *refreshUpdateHash {
			title := ""
			if c.Title != nil {
				title = *c.Title
			}
			h := hashTitleDesc(title, desc)
			newHash = &h
		}

		ev := map[string]any{
			"adapter":       f.adapter.Name(),
			"refresh_at":    time.Now().UTC().Format(time.RFC3339),
			"old_len":       c.OldLen,
			"new_len":       len(desc),
			"payload_bytes": len(raw),
		}
		evJSON := jsonStringOrNull(ev)

		if *dryRun {
			fmt.Printf("%s [dry-refresh] status=%d new_len=%d\n", prefix, status, len(desc))
			continue
		}

		q := fmt.Sprintf(`
UPDATE "%[1]s".%[2]s
SET description_snapshot=$1,
    evidence_json = COALESCE(evidence_json, '{}'::jsonb) || COALESCE($2::jsonb, '{}'::jsonb),
    title_desc_hash = COALESCE($3, title_desc_hash)
WHERE listing_id=$4 AND day_offset=$5`, *schema, *auditTable)
		_, uerr := db.Exec(ctx, q, desc, evJSON, newHash, c.ListingID, c.DayOffset)
		if uerr != nil {
			fmt.Printf("%s [refresh] update err=%v\n", prefix, uerr)
			continue
		}
		fmt.Printf("%s [refresh] updated desc_len=%d\n", prefix, len(desc))
	}
	return nil
}

/* =========================
   Runner
   ========================= */

func selectAndRun(ctx context.Context, pool *pgxpool.Pool, f *fetcher) error {
	cands, err := selectCandidates(ctx, pool)
	if err != nil {
		return err
	}
	fmt.Printf("[%s] [audit] candidates=%d adapter=%s\n", nowts(), len(cands), f.adapter.Name())
	if len(cands) == 0 {
		return nil
	}

	jobs := make(chan candidate)
	errc := make(chan error, 10000)
	var wg sync.WaitGroup

	start := time.Now()
	var processed int64
	var fail int64

	stop := make(chan struct{})
	go func() {
		tk := time.NewTicker(3 * time.Second)
		defer tk.Stop()
		for {
			select {
			case <-tk.C:
				p := atomic.LoadInt64(&processed)
				fc := atomic.LoadInt64(&fail)
				secs := time.Since(start).Seconds()
				rps := 0.0
				if secs > 0 {
					rps = float64(p) / secs
				}
				fmt.Printf("[%s] [audit] progress=%d/%d fail=%d rps=%.2f\n", nowts(), p, len(cands), fc, rps)
			case <-stop:
				return
			}
		}
	}()

	worker := func(id int) {
		defer wg.Done()
		for c := range jobs {
			prefix := pfx(c.ListingID, c.Segment, c.DayOffset)
			fmt.Printf("%s [start]\n", prefix)

			raw, status, err := f.fetch(ctx, c.ListingID)
			if err != nil {
				fmt.Printf("%s [fetch] status=%d err=%v\n", prefix, status, err)
				snap := snapshotRow{ParseErr: err.Error()}
				if e := insertSnapshot(ctx, pool, c, status, raw, snap); e != nil {
					fmt.Printf("%s [db] insert error: %v\n", prefix, e)
					select {
					case errc <- e:
					default:
					}
				}
				atomic.AddInt64(&processed, 1)
				atomic.AddInt64(&fail, 1)
				continue
			}

			snapObj, perr := f.adapter.ParsePayload(raw)
			sr := snapshotRow{
				Title:       strings.TrimSpace(snapObj.Title),
				Description: snapObj.Description,
				Price:       snapObj.Price,
				Currency:    strings.TrimSpace(snapObj.Currency),
				Evidence: map[string]any{
					"adapter":       f.adapter.Name(),
					"http_status":   status,
					"payload_bytes": len(raw),
					"status":        snapObj.Status,
					"source":        snapObj.Source,
					"source_detail": snapObj.SourceDetail,
				},
			}
			if snapObj.Attributes != nil {
				sr.Evidence["attributes"] = snapObj.Attributes
			}
			if snapObj.ObservedAt != nil {
				sr.Evidence["observed_at"] = snapObj.ObservedAt.UTC().Format(time.RFC3339)
			}
			if perr != nil {
				sr.ParseErr = perr.Error()
			}

			fmt.Printf("%s [parse] title_len=%d desc_len=%d price=%d currency=%q err=%q\n",
				prefix, len(sr.Title), len(sr.Description), sr.Price, sr.Currency, sr.ParseErr)

			if e := insertSnapshot(ctx, pool, c, status, raw, sr); e != nil {
				fmt.Printf("%s [db] insert error: %v\n", prefix, e)
				select {
				case errc <- e:
				default:
				}
				atomic.AddInt64(&fail, 1)
			} else {
				fmt.Printf("%s [db] inserted\n", prefix)
			}

			atomic.AddInt64(&processed, 1)
		}
	}

	w := *workers
	if w < 1 {
		w = 1
	}
	wg.Add(w)
	for i := 0; i < w; i++ {
		go worker(i + 1)
	}
	for _, c := range cands {
		jobs <- c
	}
	close(jobs)
	wg.Wait()

	close(stop)
	close(errc)

	failTotal := 0
	for range errc {
		failTotal++
	}

	ok := len(cands) - (int(atomic.LoadInt64(&fail)) + failTotal)
	fmt.Printf("[%s] [audit] done. ok=%d fail=%d\n", nowts(), ok, int(atomic.LoadInt64(&fail))+failTotal)
	return nil
}

/* =========================
   DDL: init + seed (synthetic)
   ========================= */

func ensureSchemaAndTables(ctx context.Context, pool *pgxpool.Pool) error {
	if !isSafeIdent(*schema) || !isSafeIdent(*listingsTable) || !isSafeIdent(*auditTable) {
		return fmt.Errorf("unsafe schema/table identifier(s)")
	}

	ddl := fmt.Sprintf(`
CREATE SCHEMA IF NOT EXISTS "%[1]s";

CREATE TABLE IF NOT EXISTS "%[1]s".%[2]s (
  listing_id bigint PRIMARY KEY,
  url text,
  segment int NOT NULL DEFAULT 0,
  status text NOT NULL DEFAULT 'unknown',
  sold_date timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS %[2]s_status_sold_idx
  ON "%[1]s".%[2]s (status, sold_date DESC);

CREATE TABLE IF NOT EXISTS "%[1]s".%[3]s (
  listing_id bigint NOT NULL,
  day_offset smallint NOT NULL,
  snapshot_at timestamptz NOT NULL DEFAULT now(),
  sold_date_ref timestamptz,
  segment_ref int,
  url text,

  http_status int,
  payload_ok boolean NOT NULL DEFAULT false,

  title_snapshot text,
  description_snapshot text,
  title_desc_hash text,

  price_snapshot int,
  currency_snapshot text,

  evidence_json jsonb,
  parse_errors text,
  diff_json jsonb,

  PRIMARY KEY (listing_id, day_offset)
);

CREATE INDEX IF NOT EXISTS %[3]s_snapshot_at_idx
  ON "%[1]s".%[3]s (snapshot_at DESC);
`, *schema, *listingsTable, *auditTable)

	_, err := pool.Exec(ctx, ddl)
	return err
}

func seedSyntheticListings(ctx context.Context, pool *pgxpool.Pool) error {
	// Safe synthetic seed data for local runs.
	// This does NOT represent any real marketplace content.
	if *dryRun {
		fmt.Printf("[%s] [seed] dry-run: not inserting\n", nowts())
		return nil
	}

	now := time.Now().UTC()
	soldAt := now.Add(-time.Duration(maxInt(8, *minAgeDays+1)) * 24 * time.Hour)

	q := fmt.Sprintf(`
INSERT INTO "%[1]s".%[2]s (listing_id, url, segment, status, sold_date)
VALUES ($1,$2,$3,'sold',$4)
ON CONFLICT (listing_id) DO NOTHING
`, *schema, *listingsTable)

	for i := int64(1); i <= 10; i++ {
		id := int64(1000) + i
		url := fmt.Sprintf("https://marketplace.example/listings/%d", id)
		seg := int(i%3) + 1
		if _, err := pool.Exec(ctx, q, id, url, seg, soldAt); err != nil {
			return err
		}
	}
	fmt.Printf("[%s] [seed] inserted up to 10 synthetic sold listings (if missing)\n", nowts())
	return nil
}

/* =========================
   Main
   ========================= */

func main() {
	rand.Seed(time.Now().UnixNano())
	flag.Parse()

	if strings.TrimSpace(*dsn) == "" {
		fmt.Fprintln(os.Stderr, "ERROR: --dsn (or PG_DSN) is required")
		os.Exit(2)
	}
	if !isSafeIdent(*schema) || !isSafeIdent(*listingsTable) || !isSafeIdent(*auditTable) {
		fmt.Fprintln(os.Stderr, "ERROR: unsafe schema/table name (must be [A-Za-z0-9_])")
		os.Exit(2)
	}

	ctx := context.Background()
	pool, err := pgxpool.New(ctx, *dsn)
	if err != nil {
		fmt.Fprintln(os.Stderr, "pg connect:", err)
		os.Exit(2)
	}
	defer pool.Close()

	// init / seed
	switch strings.ToLower(strings.TrimSpace(*mode)) {
	case "init":
		if err := ensureSchemaAndTables(ctx, pool); err != nil {
			fmt.Fprintln(os.Stderr, "init:", err)
			os.Exit(2)
		}
		fmt.Println("OK: schema/tables ensured.")
		return
	case "seed":
		if err := ensureSchemaAndTables(ctx, pool); err != nil {
			fmt.Fprintln(os.Stderr, "seed init:", err)
			os.Exit(2)
		}
		if err := seedSyntheticListings(ctx, pool); err != nil {
			fmt.Fprintln(os.Stderr, "seed:", err)
			os.Exit(2)
		}
		return
	}

	// build adapter + fetcher
	adapter, err := buildAdapter()
	if err != nil {
		fmt.Fprintln(os.Stderr, "adapter:", err)
		os.Exit(2)
	}

	gate := newAdaptiveGate(
		*rpsStart,
		math.Max(0.1, *rpsMin),
		math.Max(*rpsMin, *rpsMax),
		math.Max(0.1, *incStep),
		maxInt(1, *incEveryOK),
	)
	f := newFetcher(adapter, gate, *retryMax, *backoffInitial, *backoffMax, *jitterMs)

	// refresh-desc mode
	if strings.ToLower(strings.TrimSpace(*mode)) == "refresh-desc" {
		if err := refreshDescriptions(ctx, pool, f); err != nil {
			fmt.Fprintln(os.Stderr, "refresh:", err)
			os.Exit(2)
		}
		return
	}

	// normal run
	if err := ensureSchemaAndTables(ctx, pool); err != nil {
		fmt.Fprintln(os.Stderr, "ensure schema/tables:", err)
		os.Exit(2)
	}
	if err := selectAndRun(ctx, pool, f); err != nil {
		fmt.Fprintln(os.Stderr, "fatal:", err)
		os.Exit(2)
	}
}

/* =========================
   Helpers
   ========================= */

var safeIdentRE = regexp.MustCompile(`^[A-Za-z0-9_]+$`)

func isSafeIdent(s string) bool { return safeIdentRE.MatchString(s) }

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func ternary[T any](cond bool, a, b T) T {
	if cond {
		return a
	}
	return b
}

func getenv(key, def string) string {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	return v
}

func getenvInt(key string, def int) int {
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

func getenvBool(key string, def bool) bool {
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

func getenvFloat(key string, def float64) float64 {
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

func getenvDuration(key string, def time.Duration) time.Duration {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	if d, err := time.ParseDuration(v); err == nil {
		return d
	}
	// convenience: treat plain integers as seconds
	if n, err := strconv.Atoi(v); err == nil {
		return time.Duration(n) * time.Second
	}
	return def
}
