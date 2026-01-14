//go:build !js
// +build !js

// Package adapters contains a platform-agnostic "marketplace adapter" interface.
// The public repo intentionally ships with:
//   - an HTTP adapter that targets a placeholder JSON API, configured via env vars
//   - a mock adapter that returns synthetic data for local development and CI
//
// IMPORTANT: This package must not contain any target-marketplace identifiers, endpoints,
// selectors, or request fingerprints beyond generic HTTP client behavior.
package adapters

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

/* ========================= Public types ========================= */

type Listing struct {
	ListingID int64 `json:"listing_id"`

	// Status is the normalized listing lifecycle state as returned by the adapter.
	// Supported values: live | sold | removed | inactive | unknown
	Status string `json:"status"`

	// Price is the best-effort current asking price (for live/inactive),
	// or the observed terminal price (for sold), depending on the upstream system.
	Price int `json:"price,omitempty"`

	// SoldAt is the upstream terminal timestamp when status == sold (if available).
	SoldAt *time.Time `json:"sold_at,omitempty"`

	// SoldPrice is the explicit sold price (if the upstream provides it).
	SoldPrice *int `json:"sold_price,omitempty"`

	// IsInactive is an upstream UI/availability state flag, distinct from stale/age policies.
	// nil means "no signal provided".
	IsInactive *bool `json:"is_inactive,omitempty"`

	// IsBidding indicates "price not reliably set" / "bid-only" type flows.
	// nil means "no signal provided".
	IsBidding *bool `json:"is_bidding,omitempty"`

	// PriceSource is a short label describing how Price was derived (api|mock|unknown).
	PriceSource string `json:"price_source,omitempty"`

	// Evidence is a short, non-sensitive tag for observability/debugging.
	Evidence string `json:"evidence,omitempty"`
}

// MarketplaceAdapter is the required adapter interface for the public repo.
// All target-specific connector logic must live behind this interface.
type MarketplaceAdapter interface {
	FetchListing(ctx context.Context, listingID int64) (Listing, int, error)
	SearchListings(ctx context.Context, params map[string]string) ([]int64, int, error)
	ParsePayload(raw []byte) (Listing, error)
}

/* ========================= Config helpers ========================= */

type LimiterConfig struct {
	MinRPS      float64
	MaxRPS      float64
	StepUpRPS   float64
	DownMult    float64
	BurstFactor float64
	JitterMs    int
}

type HTTPAdapterConfig struct {
	BaseURL string

	// Optional: Authorization header value for the upstream API (inject at runtime).
	// Leave empty for public template use.
	AuthHeader string

	// User agent. If empty, a generic default is used.
	UserAgent string

	// Timeouts
	ConnectTimeout time.Duration
	HeaderTimeout  time.Duration
	IdleConnTimeout time.Duration
	RequestTimeout time.Duration

	// Concurrency/transport
	MaxConnsPerHost int

	// Retry & throttle
	RetryMax         int
	FallbackThrottle time.Duration

	// Limiter
	Limiter LimiterConfig
}

func DefaultHTTPAdapterConfigFromEnv() HTTPAdapterConfig {
	// NOTE: Keep defaults conservative and generic.
	base := strings.TrimSpace(os.Getenv("MARKETPLACE_BASE_URL"))
	if base == "" {
		base = "https://marketplace.example"
	}

	auth := strings.TrimSpace(os.Getenv("MARKETPLACE_AUTH_HEADER"))
	ua := strings.TrimSpace(os.Getenv("MARKETPLACE_HTTP_USER_AGENT"))

	// Parse RPS knobs (optional)
	parseFloat := func(key string, def float64) float64 {
		s := strings.TrimSpace(os.Getenv(key))
		if s == "" {
			return def
		}
		if v, err := strconv.ParseFloat(s, 64); err == nil {
			return v
		}
		return def
	}
	parseInt := func(key string, def int) int {
		s := strings.TrimSpace(os.Getenv(key))
		if s == "" {
			return def
		}
		if v, err := strconv.Atoi(s); err == nil {
			return v
		}
		return def
	}

	minRPS := parseFloat("REQUEST_MIN_RPS", 2.0)
	maxRPS := parseFloat("REQUEST_MAX_RPS", 10.0)
	stepUp := parseFloat("REQUEST_STEP_UP_RPS", 0.5)
	downMult := parseFloat("REQUEST_DOWN_MULT", 0.60)
	burst := parseFloat("REQUEST_BURST_FACTOR", 2.0)
	jitter := parseInt("REQUEST_JITTER_MS", 150)

	retryMax := parseInt("REQUEST_RETRY_MAX", 3)
	fallbackThrottleMs := parseInt("REQUEST_FALLBACK_THROTTLE_MS", 3000)

	return HTTPAdapterConfig{
		BaseURL:          base,
		AuthHeader:       auth,
		UserAgent:        ua,
		ConnectTimeout:   4 * time.Second,
		HeaderTimeout:    15 * time.Second,
		IdleConnTimeout:  90 * time.Second,
		RequestTimeout:   20 * time.Second,
		MaxConnsPerHost:  32,
		RetryMax:         retryMax,
		FallbackThrottle: time.Duration(fallbackThrottleMs) * time.Millisecond,
		Limiter: LimiterConfig{
			MinRPS:      minRPS,
			MaxRPS:      maxRPS,
			StepUpRPS:   stepUp,
			DownMult:    downMult,
			BurstFactor: burst,
			JitterMs:    jitter,
		},
	}
}

/* ========================= Dynamic AIMD limiter ========================= */

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

func newDynLimiter(cfg LimiterConfig) *dynLimiter {
	now := time.Now()
	minRPS := cfg.MinRPS
	maxRPS := cfg.MaxRPS
	if maxRPS < minRPS {
		maxRPS = minRPS
	}
	if minRPS <= 0.1 {
		minRPS = 0.1
	}
	down := cfg.DownMult
	if down <= 0.1 || down >= 1.0 {
		down = 0.6
	}
	burst := cfg.BurstFactor
	if burst < 1.0 {
		burst = 1.0
	}

	return &dynLimiter{
		curRPS:        minRPS,
		minRPS:        minRPS,
		maxRPS:        maxRPS,
		stepUpRPS:     cfg.StepUpRPS,
		downMult:      down,
		burstFactor:   burst,
		tokens:        minRPS * burst,
		lastRefill:    now,
		lastPenalty:   now,
		cooldownUntil: time.Time{},
		jitterMs:      cfg.JitterMs,
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

	// Passive additive increase when stable.
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

/* ========================= HTTP adapter ========================= */

type HTTPAdapter struct {
	cfg HTTPAdapterConfig

	client  *http.Client
	limiter *dynLimiter
}

func NewHTTPAdapter(cfg HTTPAdapterConfig) (*HTTPAdapter, error) {
	u, err := url.Parse(cfg.BaseURL)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("invalid base URL: %q", cfg.BaseURL)
	}

	if cfg.MaxConnsPerHost <= 0 {
		cfg.MaxConnsPerHost = 16
	}
	if cfg.RetryMax < 0 {
		cfg.RetryMax = 0
	}
	if cfg.FallbackThrottle <= 0 {
		cfg.FallbackThrottle = 3 * time.Second
	}
	if cfg.ConnectTimeout <= 0 {
		cfg.ConnectTimeout = 4 * time.Second
	}
	if cfg.HeaderTimeout <= 0 {
		cfg.HeaderTimeout = 15 * time.Second
	}
	if cfg.IdleConnTimeout <= 0 {
		cfg.IdleConnTimeout = 90 * time.Second
	}
	if cfg.RequestTimeout <= 0 {
		cfg.RequestTimeout = 20 * time.Second
	}

	tr := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		MaxConnsPerHost:       cfg.MaxConnsPerHost,
		MaxIdleConns:          256,
		MaxIdleConnsPerHost:   cfg.MaxConnsPerHost,
		IdleConnTimeout:       cfg.IdleConnTimeout,
		TLSHandshakeTimeout:   cfg.ConnectTimeout,
		ResponseHeaderTimeout: cfg.HeaderTimeout,
		ExpectContinueTimeout: 1 * time.Second,
	}

	hc := &http.Client{Transport: tr, Timeout: cfg.RequestTimeout}

	lim := newDynLimiter(cfg.Limiter)

	return &HTTPAdapter{
		cfg:     cfg,
		client:  hc,
		limiter: lim,
	}, nil
}

func (h *HTTPAdapter) FetchListing(ctx context.Context, listingID int64) (Listing, int, error) {
	// Placeholder endpoint contract:
	// GET {BASE_URL}/v1/listings/{listing_id}
	// Response: JSON matching Listing.
	u := strings.TrimRight(h.cfg.BaseURL, "/") + fmt.Sprintf("/v1/listings/%d", listingID)

	body, code, err := h.smartGET(ctx, u)
	if err != nil {
		return Listing{ListingID: listingID, Status: "unknown", Evidence: "http_error"}, code, err
	}

	// Map terminal HTTP status codes into normalized status.
	if code == http.StatusNotFound || code == http.StatusGone {
		return Listing{ListingID: listingID, Status: "removed", Evidence: fmt.Sprintf("http_%d", code)}, code, nil
	}
	if code < 200 || code >= 300 {
		return Listing{ListingID: listingID, Status: "unknown", Evidence: fmt.Sprintf("http_%d", code)}, code, nil
	}

	l, perr := h.ParsePayload(body)
	if perr != nil {
		return Listing{ListingID: listingID, Status: "unknown", Evidence: "parse_error"}, code, perr
	}
	if l.ListingID == 0 {
		l.ListingID = listingID
	}
	if l.Status == "" {
		l.Status = "unknown"
	}
	if l.PriceSource == "" {
		l.PriceSource = "api"
	}
	if l.Evidence == "" {
		l.Evidence = "api_ok"
	}
	return l, code, nil
}

func (h *HTTPAdapter) SearchListings(ctx context.Context, params map[string]string) ([]int64, int, error) {
	// Placeholder endpoint contract:
	// GET {BASE_URL}/v1/listings/search?...
	// Response: JSON: {"listing_ids":[1,2,3]}
	u, _ := url.Parse(strings.TrimRight(h.cfg.BaseURL, "/") + "/v1/listings/search")
	q := u.Query()
	for k, v := range params {
		q.Set(k, v)
	}
	u.RawQuery = q.Encode()

	body, code, err := h.smartGET(ctx, u.String())
	if err != nil {
		return nil, code, err
	}
	if code < 200 || code >= 300 {
		return nil, code, fmt.Errorf("search http=%d", code)
	}

	var resp struct {
		ListingIDs []int64 `json:"listing_ids"`
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, code, err
	}
	return resp.ListingIDs, code, nil
}

func (h *HTTPAdapter) ParsePayload(raw []byte) (Listing, error) {
	// Public repo default parsing expects JSON compatible with Listing.
	var l Listing
	if err := json.Unmarshal(raw, &l); err != nil {
		return Listing{}, err
	}

	// Normalize known statuses.
	switch strings.ToLower(strings.TrimSpace(l.Status)) {
	case "live":
		l.Status = "live"
	case "sold":
		l.Status = "sold"
	case "removed", "deleted":
		l.Status = "removed"
	case "inactive":
		l.Status = "inactive"
	case "unknown", "":
		l.Status = "unknown"
	default:
		// Keep unknown/other statuses as-is, but lower-case for consistency.
		l.Status = strings.ToLower(strings.TrimSpace(l.Status))
	}

	return l, nil
}

func (h *HTTPAdapter) setHeaders(req *http.Request) {
	ua := strings.TrimSpace(h.cfg.UserAgent)
	if ua == "" {
		ua = "marketplace-audit-template/1.0"
	}
	req.Header.Set("User-Agent", ua)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")

	if strings.TrimSpace(h.cfg.AuthHeader) != "" {
		req.Header.Set("Authorization", h.cfg.AuthHeader)
	}
}

func parseRetryAfter(hdr http.Header) time.Duration {
	v := strings.TrimSpace(hdr.Get("Retry-After"))
	if v == "" {
		return 0
	}
	if n, err := strconv.Atoi(v); err == nil && n > 0 {
		return time.Duration(n) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}

func (h *HTTPAdapter) smartGET(ctx context.Context, u string) ([]byte, int, error) {
	var lastBody []byte
	var lastCode int

	for attempt := 0; attempt <= h.cfg.RetryMax; attempt++ {
		if !h.limiter.Take(ctx) {
			return nil, 0, ctx.Err()
		}

		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
		h.setHeaders(req)

		resp, err := h.client.Do(req)
		if err != nil {
			lastCode, lastBody = 0, nil
			h.limiter.Penalize(500 * time.Millisecond)
			if attempt < h.cfg.RetryMax {
				continue
			}
			return nil, 0, err
		}

		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		resp.Body.Close()

		code := resp.StatusCode
		lastBody, lastCode = body, code

		switch {
		case code >= 200 && code < 300:
			h.limiter.Reward()
			return body, code, nil
		case code == 404 || code == 410:
			return body, code, nil
		case code == 429 || code == 408 || (code >= 500 && code <= 599):
			ra := parseRetryAfter(resp.Header)
			if ra == 0 {
				ra = h.cfg.FallbackThrottle
			}
			h.limiter.Penalize(ra)

			// Quadratic-ish backoff with small jitter.
			backoff := ra + time.Duration(attempt*attempt)*250*time.Millisecond + time.Duration(rand.Intn(151))*time.Millisecond
			if attempt < h.cfg.RetryMax {
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

/* ========================= Mock adapter (synthetic) ========================= */

type MockAdapter struct {
	// Optional: fixed reference time (useful for deterministic tests).
	Now func() time.Time
}

func NewMockAdapter() *MockAdapter {
	return &MockAdapter{
		Now: time.Now,
	}
}

func (m *MockAdapter) FetchListing(ctx context.Context, listingID int64) (Listing, int, error) {
	_ = ctx

	now := m.Now().UTC()

	// Deterministic pseudo-status:
	//  - 0 mod 10 => removed
	//  - 1 mod 10 => sold
	//  - otherwise live (some "inactive" and "bidding" flags)
	mod := listingID % 10

	switch mod {
	case 0:
		return Listing{
			ListingID:    listingID,
			Status:       "removed",
			Price:        0,
			PriceSource:  "mock",
			Evidence:     "mock_removed",
			IsInactive:   boolPtr(false),
			IsBidding:    boolPtr(false),
		}, 404, nil
	case 1:
		soldAt := now.Add(-24 * time.Hour)
		p := 1000 + int(listingID%250)*10
		return Listing{
			ListingID:    listingID,
			Status:       "sold",
			Price:        p,
			SoldAt:       &soldAt,
			SoldPrice:    intPtr(p),
			PriceSource:  "mock",
			Evidence:     "mock_sold",
			IsInactive:   boolPtr(false),
			IsBidding:    boolPtr(false),
		}, 200, nil
	default:
		price := 1000 + int(listingID%500)*5
		inactive := (mod == 2 || mod == 3)
		bidding := (mod == 4)

		st := "live"
		if inactive {
			st = "inactive"
		}

		return Listing{
			ListingID:    listingID,
			Status:       st,
			Price:        price,
			PriceSource:  "mock",
			Evidence:     "mock_live",
			IsInactive:   boolPtr(inactive),
			IsBidding:    boolPtr(bidding),
		}, 200, nil
	}
}

func (m *MockAdapter) SearchListings(ctx context.Context, params map[string]string) ([]int64, int, error) {
	_ = ctx
	_ = params

	// Minimal synthetic search: return a fixed small range.
	out := make([]int64, 0, 10)
	for i := int64(1); i <= 10; i++ {
		out = append(out, i)
	}
	return out, 200, nil
}

func (m *MockAdapter) ParsePayload(raw []byte) (Listing, error) {
	// Reuse the JSON contract so tests can share payload fixtures.
	var l Listing
	if err := json.Unmarshal(raw, &l); err != nil {
		return Listing{}, err
	}
	if l.Status == "" {
		l.Status = "unknown"
	}
	if l.PriceSource == "" {
		l.PriceSource = "mock"
	}
	if l.Evidence == "" {
		l.Evidence = "mock_payload"
	}
	return l, nil
}

/* ========================= Adapter selection ========================= */

var ErrUnknownAdapterKind = errors.New("unknown adapter kind")

func NewAdapter(kind string, httpCfg HTTPAdapterConfig) (MarketplaceAdapter, error) {
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case "", "mock":
		return NewMockAdapter(), nil
	case "http":
		return NewHTTPAdapter(httpCfg)
	default:
		return nil, ErrUnknownAdapterKind
	}
}

/* ========================= Small helpers ========================= */

func boolPtr(v bool) *bool { return &v }
func intPtr(v int) *int    { return &v }
