//go:build !js

// Package adapters contains pluggable marketplace connectors.
//
// Public release note:
// This package is intentionally generic. It does not contain any site-specific
// parsing rules, endpoint paths, headers, or fingerprints. The default
// implementation can operate in a fully offline mock mode for safe demos.
package adapters

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

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
