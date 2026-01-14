//go:build !js
// +build !js

// Marketplace stale-listing audit (public template)
//
// This binary is intended as a reusable reference implementation for:
//   - job-oriented auditing of "stale" listings already persisted in Postgres
//   - terminal state reconciliation (sold/removed) without any target-specific connector logic
//   - bounded-pressure outbound requests via an adapter layer (HTTP or mock)
//   - sparse history writes (daily baselines + change events) to avoid storage bloat
//
// Public-release posture
// ----------------------
// The adapter interface is the only place where external marketplace-specific logic is allowed.
// This repo ships with:
//   - a mock adapter (synthetic data) for safe local runs
//   - an HTTP adapter targeting a placeholder JSON API (configurable via env vars)
//
// No real marketplace identifiers, endpoints, selectors, or fingerprints are included.
package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
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

	"example.com/marketplace-audit-template/adapters"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
)

/* ========================= CLI & Config ========================= */

type config struct {
	mode string // audit-stale | diagnose

	// DB
	pgDSN      string
	pgSchema   string
	generation int

	// Selection
	staleStatus   string // default "older21days" (configurable)
	auditDays     int
	auditLimit    int
	auditOnlyIDs  string
	includeURLs   bool // default false to avoid accidental leakage in shared logs/CSVs

	// Adapter
	adapterKind        string // mock|http
	marketplaceBaseURL string // convenience override (else env)
	authHeader         string // convenience override (else env)

	// Outputs
	writeDB    bool
	writeCSV   bool
	freshCSV   bool
	out        string
	historyOut string

	// Concurrency
	workers int
	verbose bool
}

func envOrDefault(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

func envIntOrDefault(key string, def int) int {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func parseFlags() config {
	var cfg config

	flag.StringVar(&cfg.mode, "mode", envOrDefault("MODE", "audit-stale"), "Mode: audit-stale | diagnose")

	flag.StringVar(&cfg.pgDSN, "pg-dsn", envOrDefault("PG_DSN", ""), "Postgres DSN (or set PG_DSN)")
	flag.StringVar(&cfg.pgSchema, "pg-schema", envOrDefault("PG_SCHEMA", "marketplace"), "Postgres schema")
	flag.IntVar(&cfg.generation, "generation", envIntOrDefault("GENERATION", 0), "Logical generation / cohort key (integer)")

	flag.StringVar(&cfg.staleStatus, "stale-status", envOrDefault("STALE_STATUS", "older21days"), "Listings status value to audit (default: older21days)")
	flag.IntVar(&cfg.auditDays, "audit-days", envIntOrDefault("AUDIT_DAYS", 0), "Only rows with last_seen in last N days (0=all)")
	flag.IntVar(&cfg.auditLimit, "audit-limit", envIntOrDefault("AUDIT_LIMIT", 0), "LIMIT (0=all)")
	flag.StringVar(&cfg.auditOnlyIDs, "audit-only-ids", envOrDefault("AUDIT_ONLY_IDS", ""), "Only these listing IDs (csv)")
	flag.BoolVar(&cfg.includeURLs, "include-urls", false, "Include external_url in CSV/log outputs (default false)")

	flag.StringVar(&cfg.adapterKind, "adapter", envOrDefault("ADAPTER_KIND", "mock"), "Adapter kind: mock | http")
	flag.StringVar(&cfg.marketplaceBaseURL, "marketplace-base-url", envOrDefault("MARKETPLACE_BASE_URL", ""), "Marketplace base URL (optional override)")
	flag.StringVar(&cfg.authHeader, "auth-header", envOrDefault("MARKETPLACE_AUTH_HEADER", ""), "Authorization header value (optional override)")

	flag.BoolVar(&cfg.writeDB, "write-db", true, "Write updates to DB (default true)")
	flag.BoolVar(&cfg.writeCSV, "write-csv", false, "Write CSV outputs")
	flag.BoolVar(&cfg.freshCSV, "fresh-csv", false, "Truncate CSV outputs before writing")
	flag.StringVar(&cfg.out, "out", "", "Output CSV path")
	flag.StringVar(&cfg.historyOut, "history-out", "", "History CSV path")

	flag.IntVar(&cfg.workers, "workers", envIntOrDefault("WORKERS", 16), "Concurrent workers")
	flag.BoolVar(&cfg.verbose, "verbose", false, "Verbose per-row logs")

	flag.Parse()

	if cfg.workers <= 0 {
		cfg.workers = 1
	}
	if cfg.writeCSV && cfg.out == "" && cfg.historyOut == "" {
		fmt.Fprintln(os.Stderr, "[args] --write-csv set but no --out/--history-out")
		os.Exit(2)
	}
	return cfg
}

/* ========================= CSV helpers ========================= */

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

	// BOM for spreadsheet tooling
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

/* ========================= DB helpers ========================= */

func mustOpenPool(ctx context.Context, dsn string, maxConns int32) *pgxpool.Pool {
	if strings.TrimSpace(dsn) == "" {
		fmt.Fprintln(os.Stderr, "pg-dsn is required (set --pg-dsn or PG_DSN)")
		os.Exit(2)
	}

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

func isUniqueViolationOnConstraint(err error, constraint string) bool {
	var pgErr *pgconn.PgError
	if err == nil || !errors.As(err, &pgErr) {
		return false
	}
	return pgErr.Code == "23505" && pgErr.ConstraintName == constraint
}

/* ========================= Domain structs & loaders ========================= */

type ListingRow struct {
	ListingID   int64
	ExternalURL string
	Status      string

	Price int

	FirstSeen *time.Time
	LastSeen  *time.Time

	// Adapter-exposed state flags (persisted in main table).
	MarketplaceIsInactive bool
	MarketplaceIsBidding  bool
}

func parseIDsCSV(s string) []int64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	out := make([]int64, 0, 32)
	for _, t := range strings.Split(s, ",") {
		t = strings.TrimSpace(t)
		if t == "" {
			continue
		}
		if v, err := strconv.ParseInt(t, 10, 64); err == nil {
			out = append(out, v)
		}
	}
	return out
}

func loadStaleListings(ctx context.Context, pool *pgxpool.Pool, schema string, gen int, staleStatus string, auditDays, limit int, onlyIDs []int64) ([]ListingRow, error) {
	args := []any{gen, staleStatus}
	idx := 3

	sb := strings.Builder{}
	sb.WriteString(fmt.Sprintf(
		`SELECT listing_id,
		        COALESCE(external_url,'') AS external_url,
		        status,
		        COALESCE(price,0) AS price,
		        first_seen,
		        last_seen,
		        COALESCE(marketplace_is_inactive,false) AS marketplace_is_inactive,
		        COALESCE(marketplace_is_bidding,false)  AS marketplace_is_bidding
		   FROM "%s".listings
		  WHERE generation=$1 AND status=$2`,
		schema,
	))

	if auditDays > 0 {
		sb.WriteString(fmt.Sprintf(" AND last_seen >= now() - ($%d::int) * interval '1 day'", idx))
		args = append(args, auditDays)
		idx++
	}
	if len(onlyIDs) > 0 {
		sb.WriteString(fmt.Sprintf(" AND listing_id = ANY($%d::bigint[])", idx))
		args = append(args, onlyIDs)
		idx++
	}

	sb.WriteString(" ORDER BY last_seen DESC")
	if limit > 0 {
		sb.WriteString(fmt.Sprintf(" LIMIT %d", limit))
	}

	rows, err := pool.Query(ctx, sb.String(), args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := make([]ListingRow, 0, 4096)
	for rows.Next() {
		var l ListingRow
		if err := rows.Scan(
			&l.ListingID,
			&l.ExternalURL,
			&l.Status,
			&l.Price,
			&l.FirstSeen,
			&l.LastSeen,
			&l.MarketplaceIsInactive,
			&l.MarketplaceIsBidding,
		); err != nil {
			return nil, err
		}
		out = append(out, l)
	}
	return out, rows.Err()
}

/* ========================= Audit decisions ========================= */

const (
	// Guardrail to avoid polluting history with clearly-wrong prices.
	MIN_PRICE = 1
	MAX_PRICE = 100000000
)

func plausiblePrice(p int) bool { return p >= MIN_PRICE && p <= MAX_PRICE }

// dailyBucketUTC returns date_trunc('day', t) + 00:05:00 UTC.
func dailyBucketUTC(t time.Time) time.Time {
	tt := t.UTC()
	return time.Date(tt.Year(), tt.Month(), tt.Day(), 0, 5, 0, 0, time.UTC)
}

func shouldUsePriceForChangeDetection(price int, priceSource string) bool {
	if !plausiblePrice(price) {
		return false
	}
	ls := strings.ToLower(strings.TrimSpace(priceSource))
	if ls == "" || ls == "unknown" {
		return false
	}
	// Anything that starts with "db_" is a fallback and not safe for "price changed" events.
	if strings.HasPrefix(ls, "db_") {
		return false
	}
	return true
}

/* ========================= DB writes ========================= */

func updateMainStatus(ctx context.Context, pool *pgxpool.Pool, schema string, gen int, listingID int64, status string, soldPrice *int, soldAt *time.Time) error {
	if pool == nil {
		return nil
	}
	switch status {
	case "sold":
		if soldAt == nil || soldAt.IsZero() {
			return errors.New("sold requires soldAt")
		}
		if soldPrice != nil && *soldPrice > 0 {
			_, err := pool.Exec(ctx, fmt.Sprintf(
				`UPDATE "%s".listings
				 SET status='sold',
				     sold_price=$1,
				     sold_date=$2,
				     last_seen=now()
				 WHERE generation=$3 AND listing_id=$4`,
				schema,
			), *soldPrice, soldAt.UTC(), gen, listingID)
			return err
		}
		_, err := pool.Exec(ctx, fmt.Sprintf(
			`UPDATE "%s".listings
			 SET status='sold',
			     sold_date=$1,
			     last_seen=now()
			 WHERE generation=$2 AND listing_id=$3`,
			schema,
		), soldAt.UTC(), gen, listingID)
		return err

	case "removed":
		_, err := pool.Exec(ctx, fmt.Sprintf(
			`UPDATE "%s".listings
			 SET status='removed',
			     last_seen=now()
			 WHERE generation=$1 AND listing_id=$2 AND status <> 'removed'`,
			schema,
		), gen, listingID)
		return err

	default:
		return nil
	}
}

// Marketplace inactive is an upstream UI/state flag. Persist it into dedicated columns.
// Safe to call even when main status stays stale.
func updateMarketplaceInactive(ctx context.Context, pool *pgxpool.Pool, schema string, gen int, listingID int64, isInactive bool, evidence string, sourceTimestamp *time.Time) error {
	if pool == nil {
		return nil
	}

	if isInactive {
		_, err := pool.Exec(ctx, fmt.Sprintf(
			`UPDATE "%s".listings
			 SET marketplace_is_inactive=true,
			     marketplace_inactive_observed_at=now(),
			     marketplace_inactive_evidence=$1,
			     marketplace_inactive_source_timestamp = COALESCE($2, marketplace_inactive_source_timestamp)
			 WHERE generation=$3 AND listing_id=$4`,
			schema,
		), evidence, sourceTimestamp, gen, listingID)
		return err
	}

	// Clear only if previously inactive.
	_, err := pool.Exec(ctx, fmt.Sprintf(
		`UPDATE "%s".listings
		 SET marketplace_is_inactive=false,
		     marketplace_inactive_observed_at=NULL,
		     marketplace_inactive_evidence=NULL,
		     marketplace_inactive_source_timestamp=NULL
		 WHERE generation=$1 AND listing_id=$2 AND marketplace_is_inactive=true`,
		schema,
	), gen, listingID)
	return err
}

// Marketplace bidding is an upstream UI/state flag. Persist it into dedicated columns.
// Safe to call even when main status stays stale.
func updateMarketplaceBidding(ctx context.Context, pool *pgxpool.Pool, schema string, gen int, listingID int64, isBidding bool, evidence string) error {
	if pool == nil {
		return nil
	}

	if isBidding {
		_, err := pool.Exec(ctx, fmt.Sprintf(
			`UPDATE "%s".listings
			 SET marketplace_is_bidding=true,
			     marketplace_bidding_evidence=$1
			 WHERE generation=$2 AND listing_id=$3
			   AND (marketplace_is_bidding IS DISTINCT FROM true OR marketplace_bidding_evidence IS DISTINCT FROM $1)`,
			schema,
		), evidence, gen, listingID)
		return err
	}

	// Clear only if previously bidding.
	_, err := pool.Exec(ctx, fmt.Sprintf(
		`UPDATE "%s".listings
		 SET marketplace_is_bidding=false,
		     marketplace_bidding_evidence=NULL
		 WHERE generation=$1 AND listing_id=$2 AND marketplace_is_bidding=true`,
		schema,
	), gen, listingID)
	return err
}

// Append-only inactive state history:
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
	sourceTimestamp *time.Time,
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
		 FROM "%s".listing_inactive_state_events
		 WHERE generation=$1 AND listing_id=$2
		 ORDER BY observed_at DESC, event_id DESC
		 LIMIT 1`,
		schema,
	), gen, listingID).Scan(&lastVal)
	if err == nil {
		hasLast = true
	}

	if hasLast && lastVal == isInactive {
		return nil
	}
	if !hasLast && !isInactive {
		return nil
	}

	_, err = pool.Exec(ctx, fmt.Sprintf(
		`INSERT INTO "%s".listing_inactive_state_events
		 (generation, listing_id, observed_at, is_inactive, source_timestamp, observed_by, main_status, evidence, http_status)
		 VALUES ($1,$2,now(),$3,$4,$5,$6,$7,$8)`,
		schema,
	), gen, listingID, isInactive, sourceTimestamp, observedBy, mainStatus, evidence, httpStatus)
	return err
}

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

// upsertHistory writes an event snapshot using the hour+price dedupe constraint.
func upsertHistory(ctx context.Context, pool *pgxpool.Pool, schema string, gen int, listingID int64, observedAt time.Time, price int, status string, source string) error {
	if pool == nil {
		return nil
	}

	_, err := pool.Exec(ctx, fmt.Sprintf(
		`INSERT INTO "%s".price_history AS ph (generation, listing_id, observed_at, price, status, source)
		 VALUES ($1,$2,$3,$4,$5,$6)
		 ON CONFLICT ON CONSTRAINT price_history_hour_dedupe DO UPDATE SET
		   status = CASE
		     WHEN ph.status = 'sold' THEN 'sold'
		     WHEN EXCLUDED.status = 'sold' THEN 'sold'
		     WHEN ph.source <> $6 THEN ph.status
		     WHEN EXCLUDED.status = 'removed' AND ph.status <> 'sold' THEN 'removed'
		     WHEN EXCLUDED.status = $7 AND ph.status = 'live' THEN $7
		     ELSE ph.status
		   END,
		   price = CASE
		     WHEN EXCLUDED.status = 'sold' AND EXCLUDED.price > 0 THEN EXCLUDED.price
		     WHEN ph.source <> $6 THEN ph.price
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
		     WHEN ph.source <> $6 THEN ph.observed_at
		     WHEN EXCLUDED.status = 'removed' AND ph.status <> 'sold'
		       THEN GREATEST(ph.observed_at, EXCLUDED.observed_at)
		     WHEN ph.status = 'removed' AND EXCLUDED.status <> 'removed'
		       THEN ph.observed_at
		     ELSE GREATEST(ph.observed_at, EXCLUDED.observed_at)
		   END,
		   source = CASE
		     WHEN ph.source <> $6 THEN ph.source
		     ELSE EXCLUDED.source
		   END`,
		schema,
	), gen, listingID, observedAt.UTC(), price, status, source, "older21days")
	return err
}

// upsertHistoryDailyBaseline writes a once-per-UTC-day baseline snapshot for LIVE tail listings.
func upsertHistoryDailyBaseline(ctx context.Context, pool *pgxpool.Pool, schema string, gen int, listingID int64, dayBucket time.Time, price int, status string, source string) error {
	if pool == nil {
		return nil
	}

	_, err := pool.Exec(ctx, fmt.Sprintf(
		`INSERT INTO "%s".price_history AS ph (generation, listing_id, observed_at, price, status, source)
		 VALUES ($1,$2,$3,$4,$5,$6)
		 ON CONFLICT (generation, listing_id, observed_at) DO UPDATE SET
		   price = CASE
		     WHEN ph.source <> $6 THEN ph.price
		     WHEN EXCLUDED.price > 0 THEN EXCLUDED.price
		     ELSE ph.price
		   END,
		   status = CASE
		     WHEN ph.source <> $6 THEN ph.status
		     WHEN ph.status = 'sold' THEN 'sold'
		     WHEN EXCLUDED.status = 'sold' THEN 'sold'
		     WHEN EXCLUDED.status = 'removed' AND ph.status <> 'sold' THEN 'removed'
		     WHEN EXCLUDED.status = $7 AND ph.status = 'live' THEN $7
		     ELSE EXCLUDED.status
		   END,
		   source = CASE
		     WHEN ph.source <> $6 THEN ph.source
		     ELSE EXCLUDED.source
		   END`,
		schema,
	), gen, listingID, dayBucket.UTC(), price, status, source, "older21days")
	return err
}

/* ========================= Audit runner ========================= */

type auditResult struct {
	candidates int
	histOK     int
	histErr    int

	stateLive     int
	stateInactive int
	stateSold     int
	stateRemoved  int

	inactiveSeen  int
	inactiveTrue  int
	inactiveFalse int

	durationSec float64
}

func safeExternalURL(cfg config, listingID int64, externalURL string) string {
	if cfg.includeURLs && strings.TrimSpace(externalURL) != "" {
		return externalURL
	}
	// Safe placeholder for outputs.
	base := cfg.marketplaceBaseURL
	if strings.TrimSpace(base) == "" {
		base = envOrDefault("MARKETPLACE_BASE_URL", "https://marketplace.example")
	}
	return strings.TrimRight(base, "/") + fmt.Sprintf("/listing/%d", listingID)
}

func auditStaleOnce(ctx context.Context, cfg config, pool *pgxpool.Pool, adapter adapters.MarketplaceAdapter) (auditResult, error) {
	start := time.Now()

	if pool == nil {
		return auditResult{}, errors.New("audit requires DB connection")
	}

	only := parseIDsCSV(cfg.auditOnlyIDs)
	rows, err := loadStaleListings(ctx, pool, cfg.pgSchema, cfg.generation, cfg.staleStatus, cfg.auditDays, cfg.auditLimit, only)
	if err != nil {
		return auditResult{}, err
	}
	if len(rows) == 0 {
		return auditResult{candidates: 0, durationSec: time.Since(start).Seconds()}, nil
	}

	runTS := time.Now().UTC()
	dayBucket := dailyBucketUTC(runTS)
	historySource := "audit-stale"

	ids := make([]int64, 0, len(rows))
	for _, r := range rows {
		ids = append(ids, r.ListingID)
	}

	// Prefetch sparse-history state (no per-row history reads).
	lastHist, _ := prefetchLatestHistory(ctx, pool, cfg.pgSchema, cfg.generation, ids)
	lastNonZero, _ := prefetchLastNonZeroPrice(ctx, pool, cfg.pgSchema, cfg.generation, ids)
	baselineExists, _ := prefetchBaselineExists(ctx, pool, cfg.pgSchema, cfg.generation, ids, dayBucket)

	// CSV headers
	if cfg.writeCSV && cfg.freshCSV && cfg.out != "" {
		_ = os.Remove(cfg.out)
	}
	if cfg.writeCSV && cfg.freshCSV && cfg.historyOut != "" {
		_ = os.Remove(cfg.historyOut)
	}
	if cfg.writeCSV && cfg.out != "" {
		_ = ensureCSVHeader(cfg.out, []string{
			"listing_id", "observed_at", "status_detected", "status_db", "price", "price_source", "is_inactive", "is_bidding", "http_status", "evidence", "external_url",
		})
	}
	if cfg.writeCSV && cfg.historyOut != "" {
		_ = ensureCSVHeader(cfg.historyOut, []string{
			"generation", "listing_id", "observed_at", "price", "status", "source", "kind", "reason",
		})
	}

	type outRow struct {
		mainRow []string
		hRows   [][]string
	}

	results := make(chan outRow, max(128, cfg.workers*4))

	var histOK32, histErr32 int32
	var live32, inactive32, sold32, removed32 int32
	var inactSeen32, inactTrue32, inactFalse32 int32

	var wg sync.WaitGroup
	workers := max(1, min(cfg.workers, len(rows)))
	wg.Add(workers)

	for w := 0; w < workers; w++ {
		go func(shard int) {
			defer wg.Done()

			for i := shard; i < len(rows); i += workers {
				r := rows[i]
				scanTS := time.Now().UTC()

				// Adapter observation
				obs, httpStatus, obsErr := adapter.FetchListing(ctx, r.ListingID)
				if obsErr != nil {
					if cfg.verbose {
						fmt.Fprintf(os.Stderr, "[audit] listing_id=%d adapter_err=%v\n", r.ListingID, obsErr)
					}
					continue
				}

				// Effective flags: key-missing semantics (nil => keep previous DB state).
				effectiveInactive := r.MarketplaceIsInactive
				if obs.IsInactive != nil {
					effectiveInactive = *obs.IsInactive
					atomic.AddInt32(&inactSeen32, 1)
					if effectiveInactive {
						atomic.AddInt32(&inactTrue32, 1)
					} else {
						atomic.AddInt32(&inactFalse32, 1)
					}
				}
				effectiveBidding := r.MarketplaceIsBidding
				if obs.IsBidding != nil {
					effectiveBidding = *obs.IsBidding
				}

				// Persist adapter state flags (does not change main status).
				var errInact, errInactLog, errBid error
				if cfg.writeDB {
					if obs.IsInactive != nil {
						ev := strings.TrimSpace(obs.Evidence)
						if ev == "" {
							ev = "adapter_inactive_signal"
						}
						errInact = updateMarketplaceInactive(ctx, pool, cfg.pgSchema, cfg.generation, r.ListingID, effectiveInactive, ev, obs.SoldAt /* may be nil */)
						errInactLog = logInactiveEventIfChanged(ctx, pool, cfg.pgSchema, cfg.generation, r.ListingID, effectiveInactive, obs.SoldAt, historySource, cfg.staleStatus, ev, httpStatus)
					}
					if obs.IsBidding != nil {
						ev := strings.TrimSpace(obs.Evidence)
						if ev == "" {
							ev = "adapter_bidding_signal"
						}
						errBid = updateMarketplaceBidding(ctx, pool, cfg.pgSchema, cfg.generation, r.ListingID, effectiveBidding, ev)
					}
				}

				// Carry-forward last known non-zero price
				lastP := 0
				if p, ok := lastNonZero[r.ListingID]; ok && plausiblePrice(p) {
					lastP = p
				} else if lh, ok := lastHist[r.ListingID]; ok && plausiblePrice(lh.Price) {
					lastP = lh.Price
				} else if plausiblePrice(r.Price) {
					lastP = r.Price
				}

				// Determine action
				statusDetected := strings.ToLower(strings.TrimSpace(obs.Status))
				if statusDetected == "" {
					statusDetected = "unknown"
				}

				priceNow := obs.Price
				priceSource := strings.TrimSpace(obs.PriceSource)
				if priceSource == "" {
					priceSource = "unknown"
				}

				priceNowReliable := shouldUsePriceForChangeDetection(priceNow, priceSource)
				if effectiveBidding {
					priceNowReliable = false
				}

				var hRows [][]string

				// Terminal transitions (allowed): sold/removed
				var errMain error
				switch statusDetected {
				case "sold":
					atomic.AddInt32(&sold32, 1)

					// Choose sold timestamp
					soldAt := obs.SoldAt
					if soldAt == nil || soldAt.IsZero() {
						t := scanTS
						soldAt = &t
					}

					// Choose sold price (best effort).
					var soldPrice *int
					if obs.SoldPrice != nil && plausiblePrice(*obs.SoldPrice) {
						soldPrice = obs.SoldPrice
					} else if plausiblePrice(priceNow) {
						soldPrice = &priceNow
					} else if plausiblePrice(lastP) {
						soldPrice = &lastP
					}

					if cfg.writeDB {
						errMain = updateMainStatus(ctx, pool, cfg.pgSchema, cfg.generation, r.ListingID, "sold", soldPrice, soldAt)
						if herr := upsertHistory(ctx, pool, cfg.pgSchema, cfg.generation, r.ListingID, soldAt.UTC(), derefInt(soldPrice), "sold", historySource); herr != nil {
							atomic.AddInt32(&histErr32, 1)
						} else {
							atomic.AddInt32(&histOK32, 1)
						}
					}
					if cfg.writeCSV && cfg.historyOut != "" {
						hRows = append(hRows, []string{
							strconv.Itoa(cfg.generation),
							strconv.FormatInt(r.ListingID, 10),
							soldAt.UTC().Format(pgTS),
							strconv.Itoa(derefInt(soldPrice)),
							"sold",
							historySource,
							"event",
							"terminal_sold",
						})
					}

				case "removed":
					atomic.AddInt32(&removed32, 1)

					if cfg.writeDB {
						errMain = updateMainStatus(ctx, pool, cfg.pgSchema, cfg.generation, r.ListingID, "removed", nil, nil)
						if herr := upsertHistory(ctx, pool, cfg.pgSchema, cfg.generation, r.ListingID, scanTS, lastP, "removed", historySource); herr != nil {
							atomic.AddInt32(&histErr32, 1)
						} else {
							atomic.AddInt32(&histOK32, 1)
						}
					}
					if cfg.writeCSV && cfg.historyOut != "" {
						hRows = append(hRows, []string{
							strconv.Itoa(cfg.generation),
							strconv.FormatInt(r.ListingID, 10),
							scanTS.UTC().Format(pgTS),
							strconv.Itoa(lastP),
							"removed",
							historySource,
							"event",
							"terminal_removed",
						})
					}

				default:
					// Non-terminal tail state: never flip main status back to live.
					if effectiveInactive {
						atomic.AddInt32(&inactive32, 1)
					} else {
						atomic.AddInt32(&live32, 1)
					}

					// Tail-LIVE: once-per-UTC-day baseline (anti-bloat)
					if !effectiveInactive {
						if cfg.writeDB && !baselineExists[r.ListingID] && plausiblePrice(lastP) {
							herr := upsertHistoryDailyBaseline(ctx, pool, cfg.pgSchema, cfg.generation, r.ListingID, dayBucket, lastP, "live", historySource)
							if herr != nil {
								// If baseline collides with hour_dedupe unique constraint (rare),
								// treat as satisfied baseline.
								if !isUniqueViolationOnConstraint(herr, "price_history_hour_dedupe") {
									atomic.AddInt32(&histErr32, 1)
								}
							} else {
								atomic.AddInt32(&histOK32, 1)
								baselineExists[r.ListingID] = true
							}
						}
						if cfg.writeCSV && cfg.historyOut != "" && !baselineExists[r.ListingID] && plausiblePrice(lastP) {
							hRows = append(hRows, []string{
								strconv.Itoa(cfg.generation),
								strconv.FormatInt(r.ListingID, 10),
								dayBucket.UTC().Format(pgTS),
								strconv.Itoa(lastP),
								"live",
								historySource,
								"baseline",
								"daily_baseline_missing",
							})
						}
					}

					// Tail price-change events (only if reliable and changed)
					if cfg.writeDB && priceNowReliable && plausiblePrice(lastP) && priceNow != lastP {
						if herr := upsertHistory(ctx, pool, cfg.pgSchema, cfg.generation, r.ListingID, scanTS, priceNow, "live", historySource); herr != nil {
							atomic.AddInt32(&histErr32, 1)
						} else {
							atomic.AddInt32(&histOK32, 1)
						}
						if cfg.writeCSV && cfg.historyOut != "" {
							hRows = append(hRows, []string{
								strconv.Itoa(cfg.generation),
								strconv.FormatInt(r.ListingID, 10),
								scanTS.UTC().Format(pgTS),
								strconv.Itoa(priceNow),
								"live",
								historySource,
								"event",
								fmt.Sprintf("price_changed:%d->%d", lastP, priceNow),
							})
						}
					}
				}

				// Main CSV row (optional)
				var mainRow []string
				if cfg.writeCSV && cfg.out != "" {
					inactStr := ""
					if obs.IsInactive != nil {
						inactStr = strconv.FormatBool(*obs.IsInactive)
					}
					bidStr := ""
					if obs.IsBidding != nil {
						bidStr = strconv.FormatBool(*obs.IsBidding)
					}
					mainRow = []string{
						strconv.FormatInt(r.ListingID, 10),
						scanTS.UTC().Format(pgTS),
						statusDetected,
						r.Status,
						strconv.Itoa(priceNow),
						priceSource,
						inactStr,
						bidStr,
						strconv.Itoa(httpStatus),
						obs.Evidence,
						safeExternalURL(cfg, r.ListingID, r.ExternalURL),
					}
				}

				if cfg.verbose {
					fmt.Printf("[audit] ts=%s gen=%d listing_id=%d det=%s stale_status=%s price=%d price_src=%s inact=%t bid=%t http=%d main_err=%s inact_err=%s inactlog_err=%s bid_err=%s\n",
						scanTS.Format(time.RFC3339),
						cfg.generation,
						r.ListingID,
						statusDetected,
						cfg.staleStatus,
						priceNow,
						priceSource,
						effectiveInactive,
						effectiveBidding,
						httpStatus,
						errTag(errMain),
						errTag(errInact),
						errTag(errInactLog),
						errTag(errBid),
					)
				}

				results <- outRow{mainRow: mainRow, hRows: hRows}
			}
		}(w)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	var mainRows [][]string
	var histRows [][]string

	for r := range results {
		if len(r.mainRow) > 0 {
			mainRows = append(mainRows, r.mainRow)
		}
		if len(r.hRows) > 0 {
			histRows = append(histRows, r.hRows...)
		}
	}

	if cfg.writeCSV && cfg.out != "" && len(mainRows) > 0 {
		_ = appendCSV(cfg.out, mainRows)
	}
	if cfg.writeCSV && cfg.historyOut != "" && len(histRows) > 0 {
		_ = appendCSV(cfg.historyOut, histRows)
	}

	res := auditResult{
		candidates:     len(rows),
		histOK:         int(atomic.LoadInt32(&histOK32)),
		histErr:        int(atomic.LoadInt32(&histErr32)),
		stateLive:      int(atomic.LoadInt32(&live32)),
		stateInactive:  int(atomic.LoadInt32(&inactive32)),
		stateSold:      int(atomic.LoadInt32(&sold32)),
		stateRemoved:   int(atomic.LoadInt32(&removed32)),
		inactiveSeen:   int(atomic.LoadInt32(&inactSeen32)),
		inactiveTrue:   int(atomic.LoadInt32(&inactTrue32)),
		inactiveFalse:  int(atomic.LoadInt32(&inactFalse32)),
		durationSec:    time.Since(start).Seconds(),
	}
	return res, nil
}

/* ========================= Diagnose mode ========================= */

func diagnose(ctx context.Context, cfg config, pool *pgxpool.Pool) {
	if pool == nil {
		fmt.Println("diagnose: no DB")
		return
	}
	var total, sold, stale, removed int
	_ = pool.QueryRow(ctx, fmt.Sprintf(`SELECT COUNT(*) FROM "%s".listings WHERE generation=$1`, cfg.pgSchema), cfg.generation).Scan(&total)
	_ = pool.QueryRow(ctx, fmt.Sprintf(`SELECT COUNT(*) FROM "%s".listings WHERE generation=$1 AND status='sold'`, cfg.pgSchema), cfg.generation).Scan(&sold)
	_ = pool.QueryRow(ctx, fmt.Sprintf(`SELECT COUNT(*) FROM "%s".listings WHERE generation=$1 AND status=$2`, cfg.pgSchema), cfg.generation, cfg.staleStatus).Scan(&stale)
	_ = pool.QueryRow(ctx, fmt.Sprintf(`SELECT COUNT(*) FROM "%s".listings WHERE generation=$1 AND status='removed'`, cfg.pgSchema), cfg.generation).Scan(&removed)

	fmt.Printf("diagnose: gen=%d total=%d sold=%d stale=%d removed=%d\n", cfg.generation, total, sold, stale, removed)
}

/* ========================= Small helpers ========================= */

// Postgres-ish timestamptz for CSV
const pgTS = "2006-01-02 15:04:05.000000-07"

func derefInt(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

func errTag(err error) string {
	if err == nil {
		return "none"
	}
	msg := err.Error()
	msg = strings.ReplaceAll(msg, "\n", " ")
	if len(msg) > 160 {
		msg = msg[:160] + "..."
	}
	return fmt.Sprintf("%T:%s", err, msg)
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
	cfg := parseFlags()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	var pool *pgxpool.Pool
	if strings.TrimSpace(cfg.pgDSN) != "" {
		pool = mustOpenPool(ctx, cfg.pgDSN, int32(max(4, min(cfg.workers, 64))))
		defer pool.Close()
	}

	// Adapter config (HTTP adapter reads additional env vars; flags provide explicit overrides).
	httpCfg := adapters.DefaultHTTPAdapterConfigFromEnv()
	if strings.TrimSpace(cfg.marketplaceBaseURL) != "" {
		httpCfg.BaseURL = cfg.marketplaceBaseURL
	}
	if strings.TrimSpace(cfg.authHeader) != "" {
		httpCfg.AuthHeader = cfg.authHeader
	}

	adapter, err := adapters.NewAdapter(cfg.adapterKind, httpCfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, "adapter:", err)
		os.Exit(2)
	}

	switch strings.ToLower(strings.TrimSpace(cfg.mode)) {
	case "audit-stale", "audit-older": // keep "audit-older" as a compatibility alias
		if pool == nil {
			fmt.Fprintln(os.Stderr, "audit requires --pg-dsn (or PG_DSN)")
			os.Exit(2)
		}
		res, err := auditStaleOnce(ctx, cfg, pool, adapter)
		if err != nil {
			fmt.Fprintln(os.Stderr, "audit:", err)
			os.Exit(2)
		}
		fmt.Printf(
			"audit-stale: gen=%d candidates=%d history_ok=%d history_err=%d states{live=%d inactive=%d sold=%d removed=%d} inactive_seen=%d inactive_true=%d inactive_false=%d dur=%.2fs workers=%d gomaxprocs=%d numcpu=%d adapter=%s\n",
			cfg.generation,
			res.candidates,
			res.histOK,
			res.histErr,
			res.stateLive,
			res.stateInactive,
			res.stateSold,
			res.stateRemoved,
			res.inactiveSeen,
			res.inactiveTrue,
			res.inactiveFalse,
			res.durationSec,
			cfg.workers,
			runtime.GOMAXPROCS(0),
			runtime.NumCPU(),
			strings.ToLower(strings.TrimSpace(cfg.adapterKind)),
		)

	case "diagnose":
		diagnose(ctx, cfg, pool)

	default:
		fmt.Fprintln(os.Stderr, "unknown --mode (use: audit-stale | diagnose)")
		os.Exit(2)
	}
}

/* ========================= pgx helpers (optional) ========================= */

// runInTx is a small helper you can use later if you decide to batch writes.
// Not used by default to keep the audit loop straightforward.
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
