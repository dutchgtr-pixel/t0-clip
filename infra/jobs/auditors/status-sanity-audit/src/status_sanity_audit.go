package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/cookiejar"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

/*
status_sanity_audit_public.go

Purpose
-------
A single “buck stops here” status sanity auditor for generic marketplace listings.

What it does:
- Selects candidate rows from "<schema>".listings by scope/statuses/ids.
- Respects an audit cadence (default 30 days) so the same rows are not re-audited too often.
- Fetches listing payloads through a marketplace adapter layer, and derives:
    * status: sold / live / removed / inactive / unknown_*
    * sold_at, sold_price
    * live_price
    * is_inactive flag (when present in the payload)
- Writes a detailed per-listing audit event to a dedicated log table.
- Optionally applies SAFE corrections (sold/removal + inactive flag sync) with guardrails.

Public release notes
--------------------
This public template intentionally omits any target-site-specific scraping logic. All fetch + parsing is
abstracted behind a MarketplaceAdapter interface. The default adapter expects JSON responses from
a configurable base URL and parses only generic fields (status/price/sold_at/etc).

Build
-----
go build -trimpath -ldflags="-s -w" -o status_sanity_audit ./status_sanity_audit_public.go

Examples
--------
# 1) Create tables (audit tables + minimal example schema tables)
PG_DSN="..." ./status_sanity_audit --mode init --schema marketplace

# 2) Run the audit (reads MARKETPLACE_BASE_URL / MARKETPLACE_ADAPTER from env unless flags override)
PG_DSN="..." MARKETPLACE_ADAPTER=mock ./status_sanity_audit --mode run --scope all

# 3) Force-run audit on a specific set of listing IDs (ignore cadence)
./status_sanity_audit --mode run --only-ids 123456789,123456790 --force

# 4) Apply SAFE fixes (never "unsell" sold rows; never resurrect removed rows)
./status_sanity_audit --mode run --scope all --apply safe

Security posture
----------------
- No secrets are embedded in this file. Supply runtime configuration via environment variables.
- Do not commit PG_DSN or any authorization headers to version control. Use a secret manager,
  Docker secrets, or your orchestrator’s secret/connection mechanism.
*/

type cfg struct {
	mode   string
	dsn    string
	schema string

	// Marketplace adapter
	baseURL string // MARKETPLACE_BASE_URL
	adapter string // http-json|mock (MARKETPLACE_ADAPTER)

	// Selection
	scope        string   // sold|live|older21days|removed|all
	statusesCSV  string   // optional explicit list (overrides scope)
	statuses     []string // parsed
	gensCSV      string
	generations  []int
	onlyIDsCSV   string
	onlyIDs      []int64
	inactiveOnly bool
	limit        int

	// Ordering
	order string // candidate ordering (auto|first_seen_asc|sold_at_asc|...)

	// Cadence / scheduling
	cadenceDays int
	force       bool

	// HTTP
	workers          int
	timeout          time.Duration
	headFirst        bool
	userAgent        string
	acceptLanguage   string
	authHeader       string
	authHeaderFile   string
	maxBodyBytes     int64
	retryMax         int
	throttleSleep    time.Duration
	retryBackoffBase time.Duration
	jitterMax        time.Duration

	// Adaptive limiter
	rpsStart     float64
	rpsMax       float64
	rpsMin       float64
	rpsStepUp    float64
	rpsDownMult  float64
	burstFactor  float64
	okEvery      int64
	printLimiter bool

	// DB logging
	runID           string
	observedBy      string
	note            string
	dbBatch         int
	dbFlush         time.Duration
	outJSONL        string
	outCSV          string
	verbose         bool
	writeEvents     bool   // write audit run+events to DB
	logJSON         bool   // print one-line JSON log per row to stdout
	logURL          bool   // include url_used field in stdout JSON
	mismatchSummary bool   // print end-of-run mismatch summary
	applyChanges    bool   // alias: if set and apply=none, apply=safe
	apply           string // none|safe|all
	allowUnsell     bool
	allowRevive     bool
}

type candidate struct {
	Generation int
	ListingID  int64
	URLHint    string

	MainStatus      string
	MainFirstSeen   *time.Time
	MainEditedAt    *time.Time
	MainLastSeen    *time.Time
	MainSoldAt      *time.Time
	MainSoldPrice   *int
	MainLivePrice   *int
	MainIsInactive  *bool
	MainIsBidding   *bool
	LastAuditedAt   *time.Time
	FirstMarkedSold *time.Time
	MarkedSoldBy    *string
	SoldPriceAtMark *int
	PrevEventAt     *time.Time
	PrevStatus      *string
	PrevSource      *string
	PrevPrice       *int

	RowLoadedAt      time.Time
	RowLoadedAtEpoch int64
}

type detection struct {
	ObservedAt time.Time

	HTTPStatus int
	PayloadOK  bool

	DetectedStatus string // sold|live|inactive|removed|unknown_*
	PayloadSrc     string
	RawPayload     []byte

	// Inactivity flag handling (generic field: is_inactive)
	InactiveKeySeen bool
	IsInactive      *bool
	InactiveMetaTS  *time.Time

	// Sold info
	SoldAt    *time.Time
	SoldPrice *int
	SoldSrc   string
	SoldRaw   string

	// Live price
	LivePrice *int
	LiveSrc   string
	LiveRaw   string

	// Bidding (optional)
	BidKeySeen bool
	IsBidding  *bool
	BidRaw     string

	// Free-form evidence to persist
	Evidence map[string]any
	Err      string

	// URL used by the adapter (for logging only; may differ from DB URLHint)
	URLUsed string
}

type auditRow struct {
	RunID      string
	ObservedAt time.Time
	Generation int
	ListingID  int64
	URLUsed    string

	HTTPStatus int
	PayloadOK  bool

	DetectedStatus       string
	DetectedPrice        *int
	DetectedPriceSrc     string
	DetectedSoldAt       *time.Time
	DetectedIsInactive   *bool
	InactiveKeySeen      bool
	InactiveMetaEditedAt *time.Time
	PayloadSrc           string

	EvidenceJSON []byte

	MainStatus     string
	MainSoldAt     *time.Time
	MainSoldPrice  *int
	MainIsInactive *bool
	MainLastSeen   *time.Time

	FirstMarkedSoldAt *time.Time
	MarkedSoldBy      *string
	SoldPriceAtMark   *int
	PrevEventAt       *time.Time
	PrevStatus        *string
	PrevSource        *string
	PrevPrice         *int

	MismatchStatus   bool
	MismatchInactive bool

	SuggestedStatus     *string
	SuggestedIsInactive *bool
	SuggestedReason     *string

	Applied     bool
	AppliedAt   *time.Time
	ApplyAction *string
	ApplyError  *string

	ReviewAction *string
	ReviewNote   *string
	ReviewedAt   *time.Time
	ReviewedBy   *string
}

// mismatchMini is a small record used for end-of-run mismatch summaries.
type mismatchMini struct {
	ListingID          int64
	Generation         int
	MainStatus         string
	MainIsInactive     *bool
	DetectedStatus     string
	DetectedIsInactive *bool
	MismatchStatus     bool
	MismatchInactive   bool
}

func main() {
	c := parseFlags()

	ctx := context.Background()
	pool, err := pgxpool.New(ctx, c.dsn)
	if err != nil {
		fatalf("pgxpool.New: %v", err)
	}
	defer pool.Close()

	switch strings.ToLower(c.mode) {
	case "init":
		if err := ensureTables(ctx, pool, c.schema); err != nil {
			fatalf("init: %v", err)
		}
		fmt.Println("OK: tables ensured.")
		return
	case "run":
		if err := ensureTables(ctx, pool, c.schema); err != nil {
			fatalf("ensureTables: %v", err)
		}
		if err := runAudit(ctx, pool, c); err != nil {
			fatalf("run: %v", err)
		}
		return
	case "export":
		if err := exportLatestMismatches(ctx, pool, c); err != nil {
			fatalf("export: %v", err)
		}
		return
	default:
		fatalf("unknown --mode %q (expected init|run|export)", c.mode)
	}
}

func parseFlags() cfg {
	var c cfg

	flag.StringVar(&c.mode, "mode", "run", "init|run|export")
	flag.StringVar(&c.dsn, "dsn", "", "Postgres DSN (recommended via env PG_DSN)")
	flag.StringVar(&c.dsn, "db-url", "", "alias for --dsn")
	flag.StringVar(&c.schema, "schema", "", "Postgres schema (defaults to env PG_SCHEMA or \"marketplace\")")

	flag.StringVar(&c.baseURL, "marketplace-base-url", "", "Marketplace base URL (env: MARKETPLACE_BASE_URL). Empty + adapter=mock runs offline.")
	flag.StringVar(&c.baseURL, "base-url", "", "alias for --marketplace-base-url")
	flag.StringVar(&c.adapter, "adapter", "", "Marketplace adapter: http-json|mock (env: MARKETPLACE_ADAPTER)")

	flag.StringVar(&c.scope, "scope", "sold", "candidate scope: sold|live|older21days|removed|all")
	flag.StringVar(&c.statusesCSV, "statuses", "", "explicit statuses CSV (overrides --scope), e.g. sold,older21days")
	flag.StringVar(&c.gensCSV, "gens", "", "generation(s) CSV, e.g. 13 or 13,14,15 (empty = all gens)")
	flag.StringVar(&c.onlyIDsCSV, "only-ids", "", "only audit these listing IDs (CSV)")
	flag.BoolVar(&c.inactiveOnly, "inactive-only", false, "only audit rows currently marked listing_is_inactive=true")
	flag.IntVar(&c.limit, "limit", 0, "limit number of candidates (0 = no limit)")

	flag.StringVar(&c.order, "order", "auto", "candidate ordering: auto|first_seen_asc|first_seen_desc|sold_at_asc|sold_at_desc|last_seen_asc|last_seen_desc|listing_id_asc|listing_id_desc")

	flag.IntVar(&c.cadenceDays, "cadence-days", 30, "skip rows audited within this many days (0 = no cadence filter)")
	flag.BoolVar(&c.force, "force", false, "ignore cadence-days filter")

	flag.IntVar(&c.workers, "workers", 16, "number of concurrent workers")
	flag.DurationVar(&c.timeout, "timeout", 25*time.Second, "HTTP timeout per request")
	flag.BoolVar(&c.headFirst, "head-first", false, "do a HEAD preflight before GET (saves bandwidth on 404/410)")
	flag.StringVar(&c.userAgent, "ua", "marketplace-status-audit/1.0", "User-Agent")
	flag.StringVar(&c.acceptLanguage, "accept-language", "", "Optional Accept-Language (default: empty)")
	flag.StringVar(&c.authHeader, "auth-header", "", "Optional Authorization header value to send (secret; prefer env MARKETPLACE_AUTH_HEADER)")
	flag.StringVar(&c.authHeaderFile, "auth-header-file", "", "Path to file containing Authorization header value (secret)")

	flag.Int64Var(&c.maxBodyBytes, "max-body-bytes", 2<<20, "max response body bytes to read")
	flag.IntVar(&c.retryMax, "retry", 4, "max retries on throttling/network errors")
	flag.DurationVar(&c.throttleSleep, "throttle-sleep", 3*time.Second, "base sleep when throttled (will be jittered and backed off)")
	flag.DurationVar(&c.retryBackoffBase, "retry-backoff", 750*time.Millisecond, "base backoff for transient errors")
	flag.DurationVar(&c.jitterMax, "jitter", 750*time.Millisecond, "max random jitter to add to sleeps")

	flag.Float64Var(&c.rpsStart, "rps", 3.0, "starting requests-per-second")
	flag.Float64Var(&c.rpsMax, "rps-max", 10.0, "max requests-per-second")
	flag.Float64Var(&c.rpsMin, "rps-min", 0.25, "min requests-per-second after backoff")
	flag.Float64Var(&c.rpsStepUp, "rps-step", 0.25, "additive increase step when stable")
	flag.Float64Var(&c.rpsDownMult, "rps-down", 0.7, "multiplicative decrease on throttling (0<down<1)")
	flag.Float64Var(&c.burstFactor, "burst", 2.0, "token bucket burst factor")
	flag.Int64Var(&c.okEvery, "ok-every", 30, "increase rps every N OKs")
	flag.BoolVar(&c.printLimiter, "print-limiter", false, "periodically print limiter state")

	flag.StringVar(&c.runID, "run-id", "", "run UUID (optional; auto-generated if empty)")
	flag.StringVar(&c.observedBy, "observed-by", "status-sanity-audit", "label written into run metadata")
	flag.StringVar(&c.note, "note", "", "optional run note")

	flag.IntVar(&c.dbBatch, "db-batch", 200, "CopyFrom batch size")
	flag.DurationVar(&c.dbFlush, "db-flush", 2*time.Second, "flush interval for db batch writer")

	flag.StringVar(&c.outJSONL, "out-jsonl", "", "optional path to write JSONL of audit rows")
	flag.StringVar(&c.outCSV, "out-csv", "", "optional path to write CSV of audit rows")
	flag.BoolVar(&c.verbose, "v", false, "verbose logging")

	flag.BoolVar(&c.writeEvents, "write-events", true, "write audit runs+events to DB (status_sanity_runs/events). Set false for dev dry-runs")
	flag.BoolVar(&c.logJSON, "log-json", false, "print one-line JSON log per row to stdout (dev)")
	flag.BoolVar(&c.logURL, "log-url", false, "include url_used in --log-json output")
	flag.BoolVar(&c.mismatchSummary, "mismatch-summary", true, "print end-of-run mismatch summary (mismatched listing_id + main/detected statuses)")

	flag.StringVar(&c.apply, "apply", "none", "apply mode: none|safe|all")
	flag.BoolVar(&c.applyChanges, "apply-changes", false, "alias: if set and --apply is none, treat as --apply safe")
	flag.BoolVar(&c.allowUnsell, "allow-unsell", false, "ALLOW changing main status from sold -> live (dangerous; requires --apply all)")
	flag.BoolVar(&c.allowRevive, "allow-revive", false, "ALLOW changing main status from removed/older21days -> live (dangerous; requires --apply all)")

	flag.Parse()

	// Track which flags were explicitly set so environment variables can act as defaults.
	set := map[string]bool{}
	flag.Visit(func(f *flag.Flag) { set[f.Name] = true })
	// Treat alias flags as the primary flags for env-default precedence.
	if set["base-url"] {
		set["marketplace-base-url"] = true
	}
	if set["db-url"] {
		set["dsn"] = true
	}

	// Environment variable defaults (only applied when the corresponding flag was not set).
	// This keeps the public template convenient to run in containers/orchestrators without
	// committing any configuration files containing secrets.
	applyEnvString := func(flagName, envKey string, dst *string) {
		if set[flagName] {
			return
		}
		if v := strings.TrimSpace(os.Getenv(envKey)); v != "" {
			*dst = v
		}
	}
	applyEnvBool := func(flagName, envKey string, dst *bool) {
		if set[flagName] {
			return
		}
		if v := strings.TrimSpace(os.Getenv(envKey)); v != "" {
			b, err := strconv.ParseBool(v)
			if err != nil {
				fatalf("invalid %s=%q (expected boolean): %v", envKey, v, err)
			}
			*dst = b
		}
	}
	applyEnvInt := func(flagName, envKey string, dst *int) {
		if set[flagName] {
			return
		}
		if v := strings.TrimSpace(os.Getenv(envKey)); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil {
				fatalf("invalid %s=%q (expected int): %v", envKey, v, err)
			}
			*dst = n
		}
	}
	applyEnvInt64 := func(flagName, envKey string, dst *int64) {
		if set[flagName] {
			return
		}
		if v := strings.TrimSpace(os.Getenv(envKey)); v != "" {
			n, err := strconv.ParseInt(v, 10, 64)
			if err != nil {
				fatalf("invalid %s=%q (expected int64): %v", envKey, v, err)
			}
			*dst = n
		}
	}
	applyEnvFloat := func(flagName, envKey string, dst *float64) {
		if set[flagName] {
			return
		}
		if v := strings.TrimSpace(os.Getenv(envKey)); v != "" {
			n, err := strconv.ParseFloat(v, 64)
			if err != nil {
				fatalf("invalid %s=%q (expected float): %v", envKey, v, err)
			}
			*dst = n
		}
	}
	applyEnvDuration := func(flagName, envKey string, dst *time.Duration) {
		if set[flagName] {
			return
		}
		if v := strings.TrimSpace(os.Getenv(envKey)); v != "" {
			d, err := time.ParseDuration(v)
			if err != nil {
				fatalf("invalid %s=%q (expected duration, e.g. 10s, 2m): %v", envKey, v, err)
			}
			*dst = d
		}
	}

	applyEnvString("mode", "MODE", &c.mode)
	applyEnvString("schema", "PG_SCHEMA", &c.schema)
	applyEnvString("scope", "SCOPE", &c.scope)
	applyEnvString("statuses", "STATUSES", &c.statusesCSV)
	applyEnvString("gens", "GENS", &c.gensCSV)
	applyEnvString("only-ids", "ONLY_IDS", &c.onlyIDsCSV)
	applyEnvBool("inactive-only", "INACTIVE_ONLY", &c.inactiveOnly)
	applyEnvInt("limit", "LIMIT", &c.limit)
	applyEnvString("order", "ORDER", &c.order)
	applyEnvInt("cadence-days", "CADENCE_DAYS", &c.cadenceDays)
	applyEnvBool("force", "FORCE", &c.force)

	applyEnvString("marketplace-base-url", "MARKETPLACE_BASE_URL", &c.baseURL)
	applyEnvString("adapter", "MARKETPLACE_ADAPTER", &c.adapter)

	applyEnvInt("workers", "WORKERS", &c.workers)
	applyEnvDuration("timeout", "REQUEST_TIMEOUT", &c.timeout)
	applyEnvBool("head-first", "HEAD_FIRST", &c.headFirst)
	applyEnvString("ua", "USER_AGENT", &c.userAgent)
	applyEnvString("accept-language", "ACCEPT_LANGUAGE", &c.acceptLanguage)
	applyEnvString("auth-header", "MARKETPLACE_AUTH_HEADER", &c.authHeader)
	applyEnvString("auth-header-file", "MARKETPLACE_AUTH_HEADER_FILE", &c.authHeaderFile)

	applyEnvInt64("max-body-bytes", "MAX_BODY_BYTES", &c.maxBodyBytes)
	applyEnvInt("retry", "RETRY_MAX", &c.retryMax)
	applyEnvDuration("throttle-sleep", "THROTTLE_SLEEP", &c.throttleSleep)
	applyEnvDuration("retry-backoff", "RETRY_BACKOFF_BASE", &c.retryBackoffBase)
	applyEnvDuration("jitter", "JITTER_MAX", &c.jitterMax)

	applyEnvFloat("rps", "REQUEST_RPS", &c.rpsStart)
	applyEnvFloat("rps-max", "REQUEST_RPS_MAX", &c.rpsMax)
	applyEnvFloat("rps-min", "REQUEST_RPS_MIN", &c.rpsMin)
	applyEnvFloat("rps-step", "REQUEST_RPS_STEP", &c.rpsStepUp)
	applyEnvFloat("rps-down", "REQUEST_RPS_DOWN", &c.rpsDownMult)
	applyEnvFloat("burst", "REQUEST_BURST", &c.burstFactor)
	applyEnvInt64("ok-every", "REQUEST_OK_EVERY", &c.okEvery)
	applyEnvBool("print-limiter", "PRINT_LIMITER", &c.printLimiter)

	applyEnvString("run-id", "RUN_ID", &c.runID)
	applyEnvString("observed-by", "OBSERVED_BY", &c.observedBy)
	applyEnvString("note", "RUN_NOTE", &c.note)

	applyEnvInt("db-batch", "DB_BATCH", &c.dbBatch)
	applyEnvDuration("db-flush", "DB_FLUSH", &c.dbFlush)
	applyEnvString("out-jsonl", "OUT_JSONL", &c.outJSONL)
	applyEnvString("out-csv", "OUT_CSV", &c.outCSV)
	applyEnvBool("v", "VERBOSE", &c.verbose)

	applyEnvBool("write-events", "WRITE_EVENTS", &c.writeEvents)
	applyEnvBool("log-json", "LOG_JSON", &c.logJSON)
	applyEnvBool("log-url", "LOG_URL", &c.logURL)
	applyEnvBool("mismatch-summary", "MISMATCH_SUMMARY", &c.mismatchSummary)

	applyEnvString("apply", "APPLY_MODE", &c.apply)
	applyEnvBool("apply-changes", "APPLY_CHANGES", &c.applyChanges)
	applyEnvBool("allow-unsell", "ALLOW_UNSELL", &c.allowUnsell)
	applyEnvBool("allow-revive", "ALLOW_REVIVE", &c.allowRevive)

	// Schema env fallback
	if strings.TrimSpace(c.schema) == "" {
		c.schema = strings.TrimSpace(os.Getenv("PG_SCHEMA"))
	}
	if strings.TrimSpace(c.schema) == "" {
		c.schema = "marketplace"
	}

	// DSN resolution: allow env fallback to avoid accidental flags being consumed as the DSN.
	if strings.TrimSpace(c.dsn) == "" {
		c.dsn = strings.TrimSpace(os.Getenv("PG_DSN"))
	}
	if strings.TrimSpace(c.dsn) == "" {
		c.dsn = strings.TrimSpace(os.Getenv("DB_URL"))
	}
	if strings.TrimSpace(c.dsn) == "" {
		flag.Usage()
		fatalf("--dsn (or --db-url) is required (or set PG_DSN / DB_URL)")
	}
	if strings.HasPrefix(strings.TrimSpace(c.dsn), "-") {
		fatalf("dsn looks like a flag value: %q (check quoting / env var expansion)", c.dsn)
	}

	// Marketplace adapter env fallback.
	if strings.TrimSpace(c.baseURL) == "" {
		c.baseURL = strings.TrimSpace(os.Getenv("MARKETPLACE_BASE_URL"))
	}
	if strings.TrimSpace(c.adapter) == "" {
		c.adapter = strings.TrimSpace(os.Getenv("MARKETPLACE_ADAPTER"))
	}
	if strings.TrimSpace(c.adapter) == "" {
		if strings.TrimSpace(c.baseURL) == "" {
			c.adapter = "mock"
		} else {
			c.adapter = "http-json"
		}
	}
	c.adapter = strings.ToLower(strings.TrimSpace(c.adapter))
	switch c.adapter {
	case "http-json", "mock":
	default:
		fatalf("invalid --adapter %q (expected http-json|mock)", c.adapter)
	}

	if strings.TrimSpace(c.authHeader) == "" {
		c.authHeader = strings.TrimSpace(os.Getenv("MARKETPLACE_AUTH_HEADER"))
	}

	c.statuses = parseStatuses(c.scope, c.statusesCSV)
	c.generations = parseIntsCSV(c.gensCSV)
	c.onlyIDs = parseIDsCSV(c.onlyIDsCSV)

	// normalize ordering
	c.order = strings.ToLower(strings.TrimSpace(c.order))
	if c.order == "" {
		c.order = "auto"
	}

	if c.workers <= 0 {
		c.workers = 1
	}
	if c.runID == "" {
		c.runID = newUUIDv4()
	}
	c.apply = strings.ToLower(strings.TrimSpace(c.apply))
	if c.applyChanges && c.apply == "none" {
		// Developer-friendly alias: --apply-changes maps to safe mutations
		c.apply = "safe"
	}
	if c.apply != "none" && c.apply != "safe" && c.apply != "all" {
		fatalf("invalid --apply %q (expected none|safe|all)", c.apply)
	}
	if !c.writeEvents && c.apply != "none" {
		fatalf("refusing to --apply=%s when --write-events=false (must keep an audit trail)", c.apply)
	}
	if c.apply != "all" {
		// guardrails: cannot unsell/revive unless apply=all
		c.allowUnsell = false
		c.allowRevive = false
	}
	// default ordering selection
	if c.order == "auto" {
		if len(c.statuses) == 1 && c.statuses[0] == "sold" {
			c.order = "sold_at_asc"
		} else {
			c.order = "first_seen_asc"
		}
	}
	if !isAllowedOrder(c.order) {
		fatalf("invalid --order %q (expected auto|first_seen_asc|first_seen_desc|sold_at_asc|sold_at_desc|last_seen_asc|last_seen_desc|listing_id_asc|listing_id_desc)", c.order)
	}

	if c.dbBatch <= 0 {
		c.dbBatch = 200
	}
	if c.dbFlush <= 0 {
		c.dbFlush = 2 * time.Second
	}
	return c
}

func parseStatuses(scope, explicit string) []string {
	if strings.TrimSpace(explicit) != "" {
		out := parseStringsCSV(explicit)
		if len(out) == 0 {
			fatalf("--statuses was provided but parsed empty")
		}
		return out
	}
	switch strings.ToLower(strings.TrimSpace(scope)) {
	case "sold":
		return []string{"sold"}
	case "live":
		return []string{"live"}
	case "older21days":
		return []string{"older21days"}
	case "removed":
		return []string{"removed"}
	case "all":
		return []string{"live", "older21days", "sold", "removed"}
	default:
		fatalf("invalid --scope %q (expected sold|live|older21days|removed|all)", scope)
		return nil
	}
}

func parseStringsCSV(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	seen := map[string]bool{}
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		p = strings.ToLower(p)
		if !seen[p] {
			seen[p] = true
			out = append(out, p)
		}
	}
	return out
}

func parseIntsCSV(s string) []int {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil {
			fatalf("invalid int in CSV %q: %v", p, err)
		}
		out = append(out, n)
	}
	sort.Ints(out)
	return out
}

func parseIDsCSV(s string) []int64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]int64, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.ParseInt(p, 10, 64)
		if err != nil {
			fatalf("invalid id in CSV %q: %v", p, err)
		}
		out = append(out, n)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

func isAllowedOrder(s string) bool {
	switch s {
	case "auto",
		"first_seen_asc", "first_seen_desc",
		"sold_at_asc", "sold_at_desc",
		"last_seen_asc", "last_seen_desc",
		"listing_id_asc", "listing_id_desc":
		return true
	default:
		return false
	}
}

func orderBySQL(order string) string {
	switch order {
	case "first_seen_desc":
		return "ORDER BY l.first_seen DESC NULLS LAST, l.generation ASC, l.listing_id ASC"
	case "sold_at_asc":
		return "ORDER BY l.sold_at ASC NULLS LAST, l.first_seen ASC NULLS LAST, l.generation ASC, l.listing_id ASC"
	case "sold_at_desc":
		return "ORDER BY l.sold_at DESC NULLS LAST, l.first_seen DESC NULLS LAST, l.generation ASC, l.listing_id ASC"
	case "last_seen_asc":
		return "ORDER BY l.last_seen ASC NULLS LAST, l.first_seen ASC NULLS LAST, l.generation ASC, l.listing_id ASC"
	case "last_seen_desc":
		return "ORDER BY l.last_seen DESC NULLS LAST, l.first_seen DESC NULLS LAST, l.generation ASC, l.listing_id ASC"
	case "listing_id_desc":
		return "ORDER BY l.generation ASC, l.listing_id DESC"
	case "listing_id_asc":
		return "ORDER BY l.generation ASC, l.listing_id ASC"
	case "first_seen_asc":
		fallthrough
	default:
		return "ORDER BY l.first_seen ASC NULLS LAST, l.generation ASC, l.listing_id ASC"
	}
}

func newUUIDv4() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		// fallback: timestamp-based pseudo (still good enough for internal logs)
		now := time.Now().UnixNano()
		return fmt.Sprintf("00000000-0000-4000-8000-%012x", uint64(now))
	}
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	hexs := func(bs []byte) string {
		dst := make([]byte, len(bs)*2)
		hex.Encode(dst, bs)
		return string(dst)
	}
	return fmt.Sprintf("%s-%s-%s-%s-%s",
		hexs(b[0:4]),
		hexs(b[4:6]),
		hexs(b[6:8]),
		hexs(b[8:10]),
		hexs(b[10:16]),
	)
}

func runAudit(ctx context.Context, pool *pgxpool.Pool, c cfg) error {
	started := time.Now().UTC()

	if c.writeEvents {
		if err := insertRun(ctx, pool, c, started); err != nil {
			return fmt.Errorf("insertRun: %w", err)
		}
	}

	// output file writers (optional)
	var jsonlW *bufio.Writer
	var jsonlF *os.File
	if c.outJSONL != "" {
		f, err := os.Create(c.outJSONL)
		if err != nil {
			return fmt.Errorf("create out-jsonl: %w", err)
		}
		jsonlF = f
		jsonlW = bufio.NewWriterSize(f, 1<<20)
		defer func() {
			_ = jsonlW.Flush()
			_ = jsonlF.Close()
		}()
	}
	var csvW *csvWriter
	if c.outCSV != "" {
		w, err := newCSVWriter(c.outCSV)
		if err != nil {
			return err
		}
		csvW = w
		defer func() {
			_ = csvW.Close()
		}()
	}

	client, err := newHTTPClient(c.timeout)
	if err != nil {
		return err
	}
	authHdr, err := loadSecretValue(c.authHeader, c.authHeaderFile)
	if err != nil {
		return err
	}

	lim := newDynLimiter(c.rpsStart, c.rpsMax, c.rpsMin, c.rpsStepUp, c.rpsDownMult, c.burstFactor, c.okEvery)

	adapter := newMarketplaceAdapter(c, client, lim, authHdr)

	jobs := make(chan candidate, 512)
	results := make(chan auditRow, 512)

	var processed int64
	var mismatches int64
	var errorsCnt int64
	var appliedCnt int64
	var candidateCnt int64

	mismatchList := make([]mismatchMini, 0, 256)

	var logSeq int64
	var stdoutMu sync.Mutex

	// writer
	writerDone := make(chan struct{})
	go func() {
		defer close(writerDone)
		batch := make([]auditRow, 0, c.dbBatch)
		ticker := time.NewTicker(c.dbFlush)
		defer ticker.Stop()

		flush := func() {
			if len(batch) == 0 {
				return
			}
			if c.writeEvents {
				if err := copyAuditRows(ctx, pool, c.schema, batch); err != nil {
					atomic.AddInt64(&errorsCnt, int64(len(batch)))
					fmt.Fprintf(os.Stderr, "DB flush error (kept running): %v\n", err)
				}
			}
			if jsonlW != nil {
				for _, r := range batch {
					b, _ := json.Marshal(auditRowForExport(r))
					_, _ = jsonlW.Write(b)
					_, _ = jsonlW.WriteString("\n")
				}
				_ = jsonlW.Flush()
			}
			if csvW != nil {
				for _, r := range batch {
					_ = csvW.Write(r)
				}
				_ = csvW.Flush()
			}
			batch = batch[:0]
		}

		for {
			select {
			case r, ok := <-results:
				if !ok {
					flush()
					return
				}

				if c.mismatchSummary && (r.MismatchStatus || r.MismatchInactive) {
					mismatchList = append(mismatchList, mismatchMini{
						ListingID:          r.ListingID,
						Generation:         r.Generation,
						MainStatus:         r.MainStatus,
						MainIsInactive:     r.MainIsInactive,
						DetectedStatus:     r.DetectedStatus,
						DetectedIsInactive: r.DetectedIsInactive,
						MismatchStatus:     r.MismatchStatus,
						MismatchInactive:   r.MismatchInactive,
					})
				}

				batch = append(batch, r)
				if len(batch) >= c.dbBatch {
					flush()
				}
			case <-ticker.C:
				flush()
			}
		}
	}()

	// limiter printer
	stopPrint := make(chan struct{})
	if c.printLimiter {
		go func() {
			t := time.NewTicker(5 * time.Second)
			defer t.Stop()
			for {
				select {
				case <-stopPrint:
					return
				case <-t.C:
					rps, burst := lim.Snapshot()
					fmt.Printf("[limiter] rps=%.2f burst=%.0f processed=%d mismatches=%d errors=%d applied=%d\n",
						rps, burst,
						atomic.LoadInt64(&processed),
						atomic.LoadInt64(&mismatches),
						atomic.LoadInt64(&errorsCnt),
						atomic.LoadInt64(&appliedCnt),
					)
				}
			}
		}()
		defer close(stopPrint)
	}

	// workers
	var wg sync.WaitGroup
	for i := 0; i < c.workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for cand := range jobs {
				det := auditOne(ctx, adapter, c, cand)
				row := buildAuditRow(c.runID, cand, det)
				if row.MismatchStatus || row.MismatchInactive {
					atomic.AddInt64(&mismatches, 1)
				}
				if det.Err != "" {
					atomic.AddInt64(&errorsCnt, 1)
				}

				applied, applyAction, applyErr := maybeApply(ctx, pool, c, cand, det, row)
				if applied {
					atomic.AddInt64(&appliedCnt, 1)
					row.Applied = true
					now := time.Now().UTC()
					row.AppliedAt = &now
					row.ApplyAction = &applyAction
				}
				if applyErr != "" {
					row.ApplyError = &applyErr
				}

				if c.logJSON {
					seq := atomic.AddInt64(&logSeq, 1)
					line := buildStdoutJSONLine(c, seq, cand, det, row)
					stdoutMu.Lock()
					fmt.Println(line)
					stdoutMu.Unlock()
				}
				atomic.AddInt64(&processed, 1)
				results <- row
			}
		}(i)
	}

	// candidate feeder
	feedErr := make(chan error, 1)
	go func() {
		defer close(jobs)
		err := streamCandidates(ctx, pool, c, func(cand candidate) error {
			atomic.AddInt64(&candidateCnt, 1)
			jobs <- cand
			return nil
		})
		feedErr <- err
	}()

	wg.Wait()
	close(results)
	<-writerDone

	if err := <-feedErr; err != nil {
		if c.writeEvents {
			_ = finishRun(ctx, pool, c, started, time.Now().UTC(), int(candidateCnt), int(processed), int(mismatches), int(errorsCnt), int(appliedCnt))
		}
		return err
	}

	finished := time.Now().UTC()
	if c.writeEvents {
		if err := finishRun(ctx, pool, c, started, finished, int(candidateCnt), int(processed), int(mismatches), int(errorsCnt), int(appliedCnt)); err != nil {
			return err
		}
	}

	fmt.Printf("OK run_id=%s candidates=%d processed=%d mismatches=%d errors=%d applied=%d elapsed=%s go=%s os=%s/%s adapter=%s\n",
		c.runID,
		candidateCnt,
		processed,
		mismatches,
		errorsCnt,
		appliedCnt,
		finished.Sub(started).Truncate(time.Second),
		runtime.Version(),
		runtime.GOOS,
		runtime.GOARCH,
		adapter.Name(),
	)

	if c.mismatchSummary {
		fmt.Printf("MISMATCH SUMMARY: n=%d\n", len(mismatchList))
		if len(mismatchList) > 0 {
			ids := make([]string, 0, len(mismatchList))
			for _, mm := range mismatchList {
				mi := ""
				if mm.MainIsInactive != nil {
					mi = strconv.FormatBool(*mm.MainIsInactive)
				}
				di := ""
				if mm.DetectedIsInactive != nil {
					di = strconv.FormatBool(*mm.DetectedIsInactive)
				}
				fmt.Printf("listing_id: %d | gen: %d | main_status: %s | main_inactive: %s | detected_status: %s | detected_inactive: %s | mismatch_status: %t | mismatch_inactive: %t\n",
					mm.ListingID, mm.Generation, mm.MainStatus, mi, mm.DetectedStatus, di, mm.MismatchStatus, mm.MismatchInactive,
				)
				ids = append(ids, strconv.FormatInt(mm.ListingID, 10))
			}
			fmt.Printf("MISMATCH_IDS_CSV: %s\n", strings.Join(ids, ","))
		}
	}

	return nil
}

func auditRowForExport(r auditRow) map[string]any {
	m := map[string]any{
		"run_id":            r.RunID,
		"observed_at":       r.ObservedAt,
		"generation":        r.Generation,
		"listing_id":        r.ListingID,
		"url_used":          r.URLUsed,
		"http_status":       r.HTTPStatus,
		"payload_ok":        r.PayloadOK,
		"detected_status":   r.DetectedStatus,
		"main_status":       r.MainStatus,
		"mismatch_status":   r.MismatchStatus,
		"mismatch_inactive": r.MismatchInactive,
		"applied":           r.Applied,
		"payload_src":       r.PayloadSrc,
	}
	if r.DetectedPrice != nil {
		m["detected_price"] = *r.DetectedPrice
	}
	if r.DetectedPriceSrc != "" {
		m["detected_price_src"] = r.DetectedPriceSrc
	}
	if r.DetectedSoldAt != nil {
		m["detected_sold_at"] = *r.DetectedSoldAt
	}
	if r.DetectedIsInactive != nil {
		m["detected_is_inactive"] = *r.DetectedIsInactive
	}
	if r.SuggestedStatus != nil {
		m["suggested_status"] = *r.SuggestedStatus
	}
	if r.SuggestedIsInactive != nil {
		m["suggested_is_inactive"] = *r.SuggestedIsInactive
	}
	if r.SuggestedReason != nil {
		m["suggested_reason"] = *r.SuggestedReason
	}
	if r.ApplyAction != nil {
		m["apply_action"] = *r.ApplyAction
	}
	if r.ApplyError != nil {
		m["apply_error"] = *r.ApplyError
	}
	return m
}

type stdoutRowLog struct {
	ListingID  int64   `json:"listing_id"`
	Seq        int64   `json:"seq"`
	Generation int     `json:"generation"`
	URLUsed    *string `json:"url_used,omitempty"`

	MainStatus    string  `json:"main_status,omitempty"`
	MainSoldAt    *string `json:"main_sold_at,omitempty"`
	MainSoldPrice *int    `json:"main_sold_price,omitempty"`

	DetectedStatus string `json:"detected_status"`
	HTTPStatus     int    `json:"http_status"`
	PayloadOK      bool   `json:"payload_ok"`

	// inactive is a STATUS indicator (derived): detected_status == "inactive"
	Inactive bool `json:"inactive"`
	// detected_is_inactive is the payload is_inactive flag (nil if missing)
	DetectedIsInactive *bool `json:"detected_is_inactive,omitempty"`
	// main_is_inactive is the DB flag (nil if NULL)
	MainIsInactive *bool `json:"main_is_inactive,omitempty"`

	Price    *int   `json:"price,omitempty"`
	PriceSrc string `json:"price_src,omitempty"`

	DetectedSoldAt *string `json:"detected_sold_at,omitempty"`

	MismatchStatus   bool `json:"mismatch_status"`
	MismatchInactive bool `json:"mismatch_inactive"`

	SuggestedStatus   *string `json:"suggested_status,omitempty"`
	SuggestedInactive *bool   `json:"suggested_inactive,omitempty"`
	SuggestedReason   *string `json:"suggested_reason,omitempty"`

	Applied     bool    `json:"applied"`
	ApplyAction *string `json:"apply_action,omitempty"`
	ApplyError  *string `json:"apply_error,omitempty"`

	Err *string `json:"err,omitempty"`

	ObservedAt string `json:"observed_at"`

	PayloadSrc string `json:"payload_src,omitempty"`
}

func prettyJSONOneLine(b []byte) []byte {
	// Add a single space after ':' and ',' when not inside a string, for easier copy/paste in terminals.
	out := make([]byte, 0, len(b)+len(b)/10)
	inStr := false
	esc := false
	for i := 0; i < len(b); i++ {
		c := b[i]
		out = append(out, c)

		if inStr {
			if esc {
				esc = false
				continue
			}
			if c == '\\' {
				esc = true
				continue
			}
			if c == '"' {
				inStr = false
			}
			continue
		}

		if c == '"' {
			inStr = true
			continue
		}

		if c == ':' || c == ',' {
			if i+1 < len(b) && b[i+1] != ' ' {
				out = append(out, ' ')
			}
		}
	}
	return out
}

func buildStdoutJSONLine(c cfg, seq int64, cand candidate, det detection, row auditRow) string {
	var mainSoldAt *string
	if row.MainSoldAt != nil {
		s := row.MainSoldAt.UTC().Format(time.RFC3339)
		mainSoldAt = &s
	}
	var detectedSoldAt *string
	if row.DetectedSoldAt != nil {
		s := row.DetectedSoldAt.UTC().Format(time.RFC3339)
		detectedSoldAt = &s
	}

	var urlPtr *string
	if c.logURL && strings.TrimSpace(row.URLUsed) != "" {
		u := row.URLUsed
		urlPtr = &u
	}

	var errPtr *string
	if strings.TrimSpace(det.Err) != "" {
		e := det.Err
		errPtr = &e
	} else if row.ApplyError != nil && strings.TrimSpace(*row.ApplyError) != "" {
		e := *row.ApplyError
		errPtr = &e
	}

	out := stdoutRowLog{
		Seq:                seq,
		ObservedAt:         row.ObservedAt.UTC().Format(time.RFC3339),
		ListingID:          row.ListingID,
		Generation:         row.Generation,
		URLUsed:            urlPtr,
		MainStatus:         row.MainStatus,
		MainSoldAt:         mainSoldAt,
		MainSoldPrice:      row.MainSoldPrice,
		DetectedStatus:     row.DetectedStatus,
		HTTPStatus:         row.HTTPStatus,
		PayloadOK:          row.PayloadOK,
		Inactive:           (row.DetectedStatus == "inactive"),
		DetectedIsInactive: row.DetectedIsInactive,
		MainIsInactive:     row.MainIsInactive,
		Price:              row.DetectedPrice,
		PriceSrc:           row.DetectedPriceSrc,
		DetectedSoldAt:     detectedSoldAt,
		MismatchStatus:     row.MismatchStatus,
		MismatchInactive:   row.MismatchInactive,
		SuggestedStatus:    row.SuggestedStatus,
		SuggestedInactive:  row.SuggestedIsInactive,
		SuggestedReason:    row.SuggestedReason,
		Applied:            row.Applied,
		ApplyAction:        row.ApplyAction,
		ApplyError:         row.ApplyError,
		Err:                errPtr,
		PayloadSrc:         row.PayloadSrc,
	}
	b, _ := json.Marshal(out)
	b = prettyJSONOneLine(b)
	return string(b)
}

/* ===========================
   Marketplace adapter layer
   =========================== */

// MarketplaceAdapter is the generic interface that hides all platform-specific details.
// Required methods (conceptual):
//   - fetch_listing(listing_id)
//   - search_listings(params)
//   - parse_payload(raw)
type MarketplaceAdapter interface {
	Name() string
	FetchListing(ctx context.Context, listingID int64) (raw []byte, httpStatus int, urlUsed string, err error)
	SearchListings(ctx context.Context, params map[string]string) ([]int64, error)
	ParsePayload(raw []byte) (ParsedListing, error)
}

type ParsedListing struct {
	Status string // normalized: live|sold|removed|inactive|unknown

	LivePrice *int

	SoldAt    *time.Time
	SoldPrice *int

	InactiveKeySeen bool
	IsInactive      *bool
	InactiveMetaTS  *time.Time

	BidKeySeen bool
	IsBidding  *bool

	// Adapter-provided evidence (will be stored as JSON)
	Evidence map[string]any
}

type requestConfig struct {
	userAgent      string
	acceptLanguage string
	authHeader     string
}

func newMarketplaceAdapter(c cfg, client *http.Client, lim *dynLimiter, authHeader string) MarketplaceAdapter {
	rc := requestConfig{
		userAgent:      c.userAgent,
		acceptLanguage: c.acceptLanguage,
		authHeader:     authHeader,
	}

	switch c.adapter {
	case "mock":
		return &mockAdapter{rc: rc}
	case "http-json":
		return &httpJSONAdapter{
			baseURL:          strings.TrimRight(strings.TrimSpace(c.baseURL), "/"),
			client:           client,
			lim:              lim,
			rc:               rc,
			maxBodyBytes:     c.maxBodyBytes,
			retryMax:         c.retryMax,
			throttleSleep:    c.throttleSleep,
			retryBackoffBase: c.retryBackoffBase,
			jitterMax:        c.jitterMax,
			headFirst:        c.headFirst,
		}
	default:
		// parseFlags guarantees we never hit this.
		return &mockAdapter{rc: rc}
	}
}

// httpJSONAdapter is a safe default for public release: it expects a JSON payload.
// It does NOT parse HTML and does NOT embed any site-specific selectors or strings.
type httpJSONAdapter struct {
	baseURL string
	client  *http.Client
	lim     *dynLimiter
	rc      requestConfig

	maxBodyBytes     int64
	retryMax         int
	throttleSleep    time.Duration
	retryBackoffBase time.Duration
	jitterMax        time.Duration
	headFirst        bool
}

func (a *httpJSONAdapter) Name() string { return "http-json" }

func (a *httpJSONAdapter) listingURL(listingID int64) (string, error) {
	if strings.TrimSpace(a.baseURL) == "" {
		return "", errors.New("MARKETPLACE_BASE_URL is empty")
	}
	// Public placeholder endpoint shape (customize for your environment):
	//   GET {MARKETPLACE_BASE_URL}/api/listings/{listing_id}
	return fmt.Sprintf("%s/api/listings/%d", a.baseURL, listingID), nil
}

func (a *httpJSONAdapter) FetchListing(ctx context.Context, listingID int64) ([]byte, int, string, error) {
	url, err := a.listingURL(listingID)
	if err != nil {
		return nil, 0, "", err
	}

	if a.headFirst {
		code, herr := doRequest(ctx, a.client, a.lim, "HEAD", url, nil, a.rc)
		if herr == nil && (code == 404 || code == 410) {
			return nil, code, url, nil
		}
	}

	body, code, gerr := smartGET(ctx, a.client, a.lim, url, a.rc, a.maxBodyBytes, a.retryMax, a.throttleSleep, a.retryBackoffBase, a.jitterMax)
	return body, code, url, gerr
}

func (a *httpJSONAdapter) SearchListings(ctx context.Context, params map[string]string) ([]int64, error) {
	// Intentionally a stub in this auditor. Implement in your private connector if needed.
	_ = ctx
	_ = params
	return nil, errors.New("search_listings not implemented in public template")
}

func (a *httpJSONAdapter) ParsePayload(raw []byte) (ParsedListing, error) {
	return parseGenericJSONPayload(raw)
}

// mockAdapter is an offline adapter for smoke-testing the job wiring without any network calls.
// It produces deterministic synthetic payloads derived from listing_id.
type mockAdapter struct {
	rc requestConfig
}

func (m *mockAdapter) Name() string { return "mock" }

func (m *mockAdapter) FetchListing(ctx context.Context, listingID int64) ([]byte, int, string, error) {
	_ = ctx
	// Deterministic synthetic payload:
	// - Every 10th listing is "removed"
	// - Every 7th listing is "sold"
	// - Every 5th listing is "inactive"
	// - Otherwise "live"
	status := "live"
	switch {
	case listingID%10 == 0:
		status = "removed"
	case listingID%7 == 0:
		status = "sold"
	case listingID%5 == 0:
		status = "inactive"
	}

	price := int(1000 + (listingID % 2500))
	var soldAt *time.Time
	var soldPrice *int
	if status == "sold" {
		t := time.Now().UTC().Add(-time.Duration(listingID%86400) * time.Second).Truncate(time.Second)
		soldAt = &t
		sp := int(price - 50)
		soldPrice = &sp
	}

	isInactive := (status == "inactive")
	payload := map[string]any{
		"listing_id":  listingID,
		"status":      status,
		"price":       price,
		"is_inactive": isInactive,
	}
	if soldAt != nil {
		payload["sold_at"] = soldAt.Format(time.RFC3339)
	}
	if soldPrice != nil {
		payload["sold_price"] = *soldPrice
	}

	b, _ := json.Marshal(payload)
	// Use a placeholder URL string for logs only.
	urlUsed := fmt.Sprintf("mock://listing/%d", listingID)
	return b, 200, urlUsed, nil
}

func (m *mockAdapter) SearchListings(ctx context.Context, params map[string]string) ([]int64, error) {
	_ = ctx
	_ = params
	return nil, errors.New("search_listings not implemented in mock adapter")
}

func (m *mockAdapter) ParsePayload(raw []byte) (ParsedListing, error) {
	return parseGenericJSONPayload(raw)
}

func parseGenericJSONPayload(raw []byte) (ParsedListing, error) {
	if len(raw) == 0 {
		return ParsedListing{}, errors.New("empty payload")
	}
	if !json.Valid(raw) {
		return ParsedListing{}, errors.New("payload is not valid JSON")
	}
	var root map[string]any
	if err := json.Unmarshal(raw, &root); err != nil {
		return ParsedListing{}, err
	}

	out := ParsedListing{Evidence: map[string]any{}}

	// Copy a small subset of fields into evidence for audit/debugging.
	for _, k := range []string{"status", "state", "price", "sold_price", "sold_at", "is_inactive", "is_bidding"} {
		if v, ok := root[k]; ok {
			out.Evidence[k] = v
		}
	}

	// status/state
	status := ""
	if v, ok := root["status"].(string); ok {
		status = v
	} else if v, ok := root["state"].(string); ok {
		status = v
	}
	status = strings.ToLower(strings.TrimSpace(status))

	// is_inactive flag
	if v, ok := root["is_inactive"].(bool); ok {
		out.InactiveKeySeen = true
		out.IsInactive = boolPtr(v)
	}

	// is_bidding flag
	if v, ok := root["is_bidding"].(bool); ok {
		out.BidKeySeen = true
		out.IsBidding = boolPtr(v)
	} else if v, ok := root["is_auction"].(bool); ok {
		out.BidKeySeen = true
		out.IsBidding = boolPtr(v)
	}

	// price fields
	if v, ok := root["price"]; ok {
		if n, ok2 := anyToInt(v); ok2 && n > 0 {
			out.LivePrice = intPtr(n)
		}
	} else if v, ok := root["current_price"]; ok {
		if n, ok2 := anyToInt(v); ok2 && n > 0 {
			out.LivePrice = intPtr(n)
		}
	}

	// sold fields
	if v, ok := root["sold_price"]; ok {
		if n, ok2 := anyToInt(v); ok2 && n > 0 {
			out.SoldPrice = intPtr(n)
		}
	} else if v, ok := root["final_price"]; ok {
		if n, ok2 := anyToInt(v); ok2 && n > 0 {
			out.SoldPrice = intPtr(n)
		}
	}
	if s, ok := root["sold_at"].(string); ok {
		if t, err := parseTimeLoose(s); err == nil {
			out.SoldAt = &t
		}
	} else if s, ok := root["sold_date"].(string); ok {
		if t, err := parseTimeLoose(s); err == nil {
			out.SoldAt = &t
		}
	}

	// Normalize status.
	// Prefer explicit status; otherwise infer from sold fields / inactivity.
	switch status {
	case "live", "active", "available":
		out.Status = "live"
	case "inactive", "paused":
		out.Status = "inactive"
	case "sold", "closed":
		out.Status = "sold"
	case "removed", "deleted", "unavailable":
		out.Status = "removed"
	case "":
		// Inference
		if out.SoldAt != nil || out.SoldPrice != nil {
			out.Status = "sold"
		} else if out.InactiveKeySeen && out.IsInactive != nil && *out.IsInactive {
			out.Status = "inactive"
		} else {
			out.Status = "unknown_no_status"
		}
	default:
		out.Status = "unknown_" + status
	}

	// If inactive flag exists but status says live, keep status as live (mirrors original semantics).
	// Consumers can still inspect IsInactive.
	if out.Status == "live" && out.InactiveKeySeen && out.IsInactive != nil && *out.IsInactive {
		out.Status = "inactive"
	}

	return out, nil
}

func parseTimeLoose(s string) (time.Time, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, errors.New("empty time")
	}
	// Try RFC3339 first.
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t.UTC(), nil
	}
	// Accept a common date-only form.
	if t, err := time.Parse("2006-01-02", s); err == nil {
		return t.UTC(), nil
	}
	return time.Time{}, fmt.Errorf("unsupported time format: %q", s)
}

/* ===========================
   Audit execution
   =========================== */

func auditOne(ctx context.Context, adapter MarketplaceAdapter, c cfg, cand candidate) detection {
	obs := time.Now().UTC()
	det := detection{
		ObservedAt: obs,
		Evidence:   map[string]any{},
	}

	raw, code, urlUsed, err := adapter.FetchListing(ctx, cand.ListingID)
	det.URLUsed = urlUsed
	det.HTTPStatus = code
	det.RawPayload = raw
	det.PayloadOK = (code >= 200 && code < 300 && len(raw) > 0)

	if err != nil {
		det.Err = err.Error()
	}

	// HTTP-only classification before parsing.
	if code == 404 || code == 410 {
		det.DetectedStatus = "removed"
		det.PayloadSrc = adapter.Name()
		return det
	}
	if code < 200 || code >= 300 {
		det.DetectedStatus = fmt.Sprintf("unknown_http_%d", code)
		det.PayloadSrc = adapter.Name()
		return det
	}
	if len(raw) == 0 {
		det.DetectedStatus = "unknown_empty_payload"
		det.PayloadSrc = adapter.Name()
		return det
	}

	parsed, perr := adapter.ParsePayload(raw)
	det.PayloadSrc = adapter.Name()
	if perr != nil {
		det.DetectedStatus = "unknown_parse_error"
		det.Err = mergeErr(det.Err, perr.Error())
		return det
	}

	// Evidence is adapter-provided, but we persist it as JSON for traceability.
	for k, v := range parsed.Evidence {
		det.Evidence[k] = v
	}

	det.InactiveKeySeen = parsed.InactiveKeySeen
	det.IsInactive = parsed.IsInactive
	det.InactiveMetaTS = parsed.InactiveMetaTS

	if parsed.BidKeySeen {
		det.BidKeySeen = true
		det.IsBidding = parsed.IsBidding
		det.BidRaw = "payload"
	}

	switch parsed.Status {
	case "sold":
		det.DetectedStatus = "sold"
		det.SoldAt = parsed.SoldAt
		if parsed.SoldPrice != nil {
			det.SoldPrice = intPtr(*parsed.SoldPrice)
			det.SoldSrc = "payload"
			det.SoldRaw = fmt.Sprintf("%d", *parsed.SoldPrice)
		}
		// Fall back: if no sold price, reuse last live price as evidence only (non-authoritative).
		if det.SoldPrice == nil && cand.MainLivePrice != nil && *cand.MainLivePrice > 0 {
			det.SoldPrice = intPtr(*cand.MainLivePrice)
			det.SoldSrc = "last"
			det.SoldRaw = fmt.Sprintf("%d", *cand.MainLivePrice)
		}
		return det

	case "removed":
		det.DetectedStatus = "removed"
		return det

	case "inactive":
		det.DetectedStatus = "inactive"
		if parsed.LivePrice != nil && *parsed.LivePrice > 0 {
			det.LivePrice = intPtr(*parsed.LivePrice)
			det.LiveSrc = "payload"
			det.LiveRaw = fmt.Sprintf("%d", *parsed.LivePrice)
		}
		return det

	case "live":
		det.DetectedStatus = "live"
		if parsed.LivePrice != nil && *parsed.LivePrice > 0 {
			det.LivePrice = intPtr(*parsed.LivePrice)
			det.LiveSrc = "payload"
			det.LiveRaw = fmt.Sprintf("%d", *parsed.LivePrice)
		}
		return det

	default:
		det.DetectedStatus = parsed.Status
		if parsed.LivePrice != nil && *parsed.LivePrice > 0 {
			det.LivePrice = intPtr(*parsed.LivePrice)
			det.LiveSrc = "payload"
			det.LiveRaw = fmt.Sprintf("%d", *parsed.LivePrice)
		}
		return det
	}
}

func mergeErr(existing, add string) string {
	existing = strings.TrimSpace(existing)
	add = strings.TrimSpace(add)
	if add == "" {
		return existing
	}
	if existing == "" {
		return add
	}
	return existing + "; " + add
}

func buildAuditRow(runID string, cand candidate, det detection) auditRow {
	row := auditRow{
		RunID:      runID,
		ObservedAt: det.ObservedAt,
		Generation: cand.Generation,
		ListingID:  cand.ListingID,
		URLUsed:    det.URLUsed,

		HTTPStatus: det.HTTPStatus,
		PayloadOK:  det.PayloadOK,

		DetectedStatus: det.DetectedStatus,
		PayloadSrc:     det.PayloadSrc,

		MainStatus:     cand.MainStatus,
		MainSoldAt:     cand.MainSoldAt,
		MainSoldPrice:  cand.MainSoldPrice,
		MainIsInactive: cand.MainIsInactive,
		MainLastSeen:   cand.MainLastSeen,

		FirstMarkedSoldAt: cand.FirstMarkedSold,
		MarkedSoldBy:      cand.MarkedSoldBy,
		SoldPriceAtMark:   cand.SoldPriceAtMark,
		PrevEventAt:       cand.PrevEventAt,
		PrevStatus:        cand.PrevStatus,
		PrevSource:        cand.PrevSource,
		PrevPrice:         cand.PrevPrice,
	}

	switch det.DetectedStatus {
	case "sold":
		row.DetectedPrice = det.SoldPrice
		row.DetectedPriceSrc = det.SoldSrc
		row.DetectedSoldAt = det.SoldAt
	case "live", "inactive":
		row.DetectedPrice = det.LivePrice
		row.DetectedPriceSrc = det.LiveSrc
	}

	row.DetectedIsInactive = det.IsInactive
	row.InactiveKeySeen = det.InactiveKeySeen
	row.InactiveMetaEditedAt = det.InactiveMetaTS

	ev := map[string]any{
		"http_status": det.HTTPStatus,
		"payload_ok":  det.PayloadOK,
		"adapter":     det.PayloadSrc,
	}
	for k, v := range det.Evidence {
		ev[k] = v
	}
	if det.SoldRaw != "" {
		ev["sold_price_raw"] = det.SoldRaw
	}
	if det.LiveRaw != "" {
		ev["live_price_raw"] = det.LiveRaw
	}
	if det.Err != "" {
		ev["error"] = det.Err
	}
	if det.URLUsed != "" {
		ev["url_used"] = det.URLUsed
	}
	row.EvidenceJSON, _ = json.Marshal(ev)

	expectedMainStatus := expectedStatusForMain(det.DetectedStatus)
	row.MismatchStatus = (expectedMainStatus != "" && cand.MainStatus != expectedMainStatus)

	inactiveEligible := (expectedMainStatus == "live")
	if inactiveEligible && det.InactiveKeySeen && cand.MainIsInactive != nil && det.IsInactive != nil && (*cand.MainIsInactive != *det.IsInactive) {
		row.MismatchInactive = true
	}

	var sugStatus *string
	var sugInactive *bool
	var reason *string

	if expectedMainStatus != "" && cand.MainStatus != expectedMainStatus {
		s := expectedMainStatus
		sugStatus = &s
		r := fmt.Sprintf("status_mismatch main=%s detected=%s", cand.MainStatus, det.DetectedStatus)
		reason = &r
	}
	if inactiveEligible && det.InactiveKeySeen && det.IsInactive != nil {
		if cand.MainIsInactive == nil || *cand.MainIsInactive != *det.IsInactive {
			b := *det.IsInactive
			sugInactive = &b
			r := "inactive_flag_mismatch"
			reason = mergeReasons(reason, &r)
		}
	}

	if cand.MainStatus == "sold" && expectedMainStatus == "live" {
		r := "sold_row_appears_live_or_inactive (candidate for UNSOLD; manual review)"
		reason = mergeReasons(reason, &r)
	}
	if cand.MainStatus == "removed" && expectedMainStatus == "live" {
		r := "removed_row_appears_live_or_inactive (candidate for REVIVE; manual review)"
		reason = mergeReasons(reason, &r)
	}
	if cand.MainStatus == "older21days" && expectedMainStatus == "live" {
		r := "older21days_row_appears_live_or_inactive (candidate for REVIVE; manual review)"
		reason = mergeReasons(reason, &r)
	}

	row.SuggestedStatus = sugStatus
	row.SuggestedIsInactive = sugInactive
	row.SuggestedReason = reason

	return row
}

func mergeReasons(existing *string, add *string) *string {
	if add == nil {
		return existing
	}
	if existing == nil {
		return add
	}
	s := *existing + "; " + *add
	return &s
}

func expectedStatusForMain(detected string) string {
	switch detected {
	case "sold":
		return "sold"
	case "removed":
		return "removed"
	case "live", "inactive":
		return "live"
	default:
		return ""
	}
}

func maybeApply(ctx context.Context, pool *pgxpool.Pool, c cfg, cand candidate, det detection, row auditRow) (bool, string, string) {
	if c.apply == "none" {
		return false, "", ""
	}

	appliedAnything := false
	var actions []string

	if det.InactiveKeySeen && det.IsInactive != nil {
		want := *det.IsInactive
		cur := false
		if cand.MainIsInactive != nil {
			cur = *cand.MainIsInactive
		}
		if cur != want {
			evJSON := row.EvidenceJSON
			metaEdited := det.InactiveMetaTS
			if err := updateListingInactive(ctx, pool, c.schema, want, det.ObservedAt, metaEdited, evJSON, cand.Generation, cand.ListingID); err != nil {
				return false, "", fmt.Sprintf("updateListingInactive: %v", err)
			}
			appliedAnything = true
			actions = append(actions, fmt.Sprintf("sync_inactive=%t", want))
		}
	}

	expected := expectedStatusForMain(det.DetectedStatus)
	if expected == "" || cand.MainStatus == expected {
		if appliedAnything {
			return true, strings.Join(actions, ","), ""
		}
		return false, "", ""
	}

	if c.apply == "safe" {
		if expected == "sold" && (cand.MainStatus == "live" || cand.MainStatus == "older21days") {
			if err := updateMainSold(ctx, pool, c.schema, det.SoldAt, det.SoldPrice, det.ObservedAt, cand.Generation, cand.ListingID); err != nil {
				return false, "", fmt.Sprintf("updateMainSold: %v", err)
			}
			appliedAnything = true
			actions = append(actions, "set_status=sold")
		} else if expected == "removed" && det.DetectedStatus == "removed" && (cand.MainStatus == "live" || cand.MainStatus == "older21days") {
			if err := updateMainRemoved(ctx, pool, c.schema, det.ObservedAt, cand.Generation, cand.ListingID); err != nil {
				return false, "", fmt.Sprintf("updateMainRemoved: %v", err)
			}
			appliedAnything = true
			actions = append(actions, "set_status=removed")
		}
		if appliedAnything {
			return true, strings.Join(actions, ","), ""
		}
		return false, "", ""
	}

	if c.apply == "all" {
		switch {
		case expected == "sold" && (cand.MainStatus == "live" || cand.MainStatus == "older21days"):
			if err := updateMainSold(ctx, pool, c.schema, det.SoldAt, det.SoldPrice, det.ObservedAt, cand.Generation, cand.ListingID); err != nil {
				return false, "", fmt.Sprintf("updateMainSold: %v", err)
			}
			appliedAnything = true
			actions = append(actions, "set_status=sold")

		case expected == "removed" && det.DetectedStatus == "removed" && (cand.MainStatus == "live" || cand.MainStatus == "older21days"):
			if err := updateMainRemoved(ctx, pool, c.schema, det.ObservedAt, cand.Generation, cand.ListingID); err != nil {
				return false, "", fmt.Sprintf("updateMainRemoved: %v", err)
			}
			appliedAnything = true
			actions = append(actions, "set_status=removed")

		case expected == "live" && cand.MainStatus == "sold" && c.allowUnsell:
			if err := updateMainUnsellToLive(ctx, pool, c.schema, det.ObservedAt, cand.Generation, cand.ListingID); err != nil {
				return false, "", fmt.Sprintf("updateMainUnsellToLive: %v", err)
			}
			appliedAnything = true
			actions = append(actions, "unsell_to_live")

		case expected == "live" && (cand.MainStatus == "removed" || cand.MainStatus == "older21days") && c.allowRevive:
			if err := updateMainReviveToLive(ctx, pool, c.schema, det.ObservedAt, cand.Generation, cand.ListingID); err != nil {
				return false, "", fmt.Sprintf("updateMainReviveToLive: %v", err)
			}
			appliedAnything = true
			actions = append(actions, "revive_to_live")
		}
	}

	if appliedAnything {
		return true, strings.Join(actions, ","), ""
	}
	return false, "", ""
}

/* ===========================
   DB: schema + queries
   =========================== */

func ensureTables(ctx context.Context, pool *pgxpool.Pool, schema string) error {
	if !isSafeIdent(schema) {
		return fmt.Errorf("unsafe schema name %q", schema)
	}
	ddl := fmt.Sprintf(`
CREATE SCHEMA IF NOT EXISTS "%[1]s";

-- Minimal example "main" tables required by this auditor.
-- For public release, they are created if missing so the template is runnable.
CREATE TABLE IF NOT EXISTS "%[1]s".listings (
  generation int NOT NULL,
  listing_id bigint NOT NULL,
  url text,
  status text NOT NULL,
  first_seen timestamptz,
  edited_at timestamptz,
  last_seen timestamptz,
  sold_at timestamptz,
  sold_price int,
  price int,
  listing_is_inactive boolean,
  listing_is_bidding boolean,

  listing_inactive_observed_at timestamptz,
  listing_inactive_evidence jsonb,
  listing_inactive_meta_edited_at timestamptz,

  PRIMARY KEY (generation, listing_id)
);

CREATE TABLE IF NOT EXISTS "%[1]s".price_history (
  event_id bigserial PRIMARY KEY,
  observed_at timestamptz NOT NULL DEFAULT now(),
  generation int NOT NULL,
  listing_id bigint NOT NULL,
  status text NOT NULL,
  price int,
  source text
);

CREATE INDEX IF NOT EXISTS price_history_listing_idx
  ON "%[1]s".price_history (generation, listing_id, observed_at DESC);

-- Auditor tables
CREATE TABLE IF NOT EXISTS "%[1]s".status_sanity_runs (
  run_id uuid PRIMARY KEY,
  started_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz,
  observed_by text NOT NULL,
  note text,
  scope text NOT NULL,
  statuses text[] NOT NULL,
  generations int[] NULL,
  inactive_only boolean NOT NULL DEFAULT false,
  cadence_days int NOT NULL,
  force_run boolean NOT NULL DEFAULT false,
  apply_mode text NOT NULL DEFAULT 'none',
  allow_unsell boolean NOT NULL DEFAULT false,
  allow_revive boolean NOT NULL DEFAULT false,
  workers int NOT NULL,
  head_first boolean NOT NULL,
  rps_start double precision NOT NULL,
  rps_max double precision NOT NULL,
  adapter text NOT NULL,
  base_url text,
  config_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  candidate_count int NOT NULL DEFAULT 0,
  processed_count int NOT NULL DEFAULT 0,
  mismatch_count int NOT NULL DEFAULT 0,
  error_count int NOT NULL DEFAULT 0,
  applied_count int NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS "%[1]s".status_sanity_events (
  event_id bigserial PRIMARY KEY,
  run_id uuid NOT NULL REFERENCES "%[1]s".status_sanity_runs(run_id) ON DELETE CASCADE,
  observed_at timestamptz NOT NULL DEFAULT now(),
  generation int NOT NULL,
  listing_id bigint NOT NULL,
  url_used text,

  http_status int,
  payload_ok boolean,

  detected_status text NOT NULL,
  detected_price int,
  detected_price_src text,
  detected_sold_at timestamptz,
  detected_is_inactive boolean,
  inactive_key_seen boolean,
  inactive_meta_edited_at timestamptz,
  payload_src text,

  evidence_json jsonb,

  main_status text,
  main_sold_at timestamptz,
  main_sold_price int,
  main_is_inactive boolean,
  main_last_seen timestamptz,

  first_marked_sold_at timestamptz,
  marked_sold_by text,
  sold_price_at_mark int,
  prev_event_at timestamptz,
  prev_status text,
  prev_source text,
  prev_price int,

  mismatch_status boolean NOT NULL DEFAULT false,
  mismatch_inactive boolean NOT NULL DEFAULT false,

  suggested_status text,
  suggested_is_inactive boolean,
  suggested_reason text,

  applied boolean NOT NULL DEFAULT false,
  applied_at timestamptz,
  apply_action text,
  apply_error text,

  review_action text,
  review_note text,
  reviewed_at timestamptz,
  reviewed_by text,

  UNIQUE(run_id, generation, listing_id)
);

CREATE INDEX IF NOT EXISTS status_sanity_events_listing_idx
  ON "%[1]s".status_sanity_events (generation, listing_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS status_sanity_events_run_idx
  ON "%[1]s".status_sanity_events (run_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS status_sanity_events_mismatch_idx
  ON "%[1]s".status_sanity_events (observed_at DESC)
  WHERE mismatch_status OR mismatch_inactive;
`, schema)

	_, err := pool.Exec(ctx, ddl)
	return err
}

func insertRun(ctx context.Context, pool *pgxpool.Pool, c cfg, started time.Time) error {
	configJSON, _ := json.Marshal(map[string]any{
		"gens":           c.generations,
		"only_ids":       c.onlyIDs,
		"limit":          c.limit,
		"http_timeout":   c.timeout.String(),
		"head_first":     c.headFirst,
		"retry":          c.retryMax,
		"throttle_sleep": c.throttleSleep.String(),
		"rps_start":      c.rpsStart,
		"rps_max":        c.rpsMax,
		"rps_min":        c.rpsMin,
		"workers":        c.workers,
		"apply":          c.apply,
	})
	q := fmt.Sprintf(`
INSERT INTO "%s".status_sanity_runs (
  run_id, started_at, observed_by, note,
  scope, statuses, generations, inactive_only,
  cadence_days, force_run, apply_mode, allow_unsell, allow_revive,
  workers, head_first, rps_start, rps_max,
  adapter, base_url, config_json
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)
ON CONFLICT (run_id) DO NOTHING
`, c.schema)

	_, err := pool.Exec(ctx, q,
		c.runID, started, c.observedBy, nullIfEmpty(c.note),
		c.scope, c.statuses, intSliceOrNull(c.generations), c.inactiveOnly,
		c.cadenceDays, c.force, c.apply, c.allowUnsell, c.allowRevive,
		c.workers, c.headFirst, c.rpsStart, c.rpsMax,
		c.adapter, nullIfEmpty(c.baseURL), configJSON,
	)
	return err
}

func finishRun(ctx context.Context, pool *pgxpool.Pool, c cfg, started, finished time.Time, candidates, processed, mismatches, errorsCnt, applied int) error {
	q := fmt.Sprintf(`
UPDATE "%s".status_sanity_runs
SET finished_at=$2,
    candidate_count=$3,
    processed_count=$4,
    mismatch_count=$5,
    error_count=$6,
    applied_count=$7
WHERE run_id=$1
`, c.schema)
	_, err := pool.Exec(ctx, q, c.runID, finished, candidates, processed, mismatches, errorsCnt, applied)
	return err
}

func copyAuditRows(ctx context.Context, pool *pgxpool.Pool, schema string, rows []auditRow) error {
	if len(rows) == 0 {
		return nil
	}
	cols := []string{
		"run_id",
		"observed_at",
		"generation",
		"listing_id",
		"url_used",
		"http_status",
		"payload_ok",
		"detected_status",
		"detected_price",
		"detected_price_src",
		"detected_sold_at",
		"detected_is_inactive",
		"inactive_key_seen",
		"inactive_meta_edited_at",
		"payload_src",
		"evidence_json",
		"main_status",
		"main_sold_at",
		"main_sold_price",
		"main_is_inactive",
		"main_last_seen",
		"first_marked_sold_at",
		"marked_sold_by",
		"sold_price_at_mark",
		"prev_event_at",
		"prev_status",
		"prev_source",
		"prev_price",
		"mismatch_status",
		"mismatch_inactive",
		"suggested_status",
		"suggested_is_inactive",
		"suggested_reason",
		"applied",
		"applied_at",
		"apply_action",
		"apply_error",
		"review_action",
		"review_note",
		"reviewed_at",
		"reviewed_by",
	}

	values := make([][]any, 0, len(rows))
	for _, r := range rows {
		values = append(values, []any{
			r.RunID,
			r.ObservedAt,
			r.Generation,
			r.ListingID,
			nullIfEmpty(r.URLUsed),
			nullableInt(r.HTTPStatus),
			r.PayloadOK,
			r.DetectedStatus,
			r.DetectedPrice,
			nullIfEmpty(r.DetectedPriceSrc),
			r.DetectedSoldAt,
			r.DetectedIsInactive,
			r.InactiveKeySeen,
			r.InactiveMetaEditedAt,
			nullIfEmpty(r.PayloadSrc),
			json.RawMessage(r.EvidenceJSON),
			nullIfEmpty(r.MainStatus),
			r.MainSoldAt,
			r.MainSoldPrice,
			r.MainIsInactive,
			r.MainLastSeen,
			r.FirstMarkedSoldAt,
			r.MarkedSoldBy,
			r.SoldPriceAtMark,
			r.PrevEventAt,
			r.PrevStatus,
			r.PrevSource,
			r.PrevPrice,
			r.MismatchStatus,
			r.MismatchInactive,
			r.SuggestedStatus,
			r.SuggestedIsInactive,
			r.SuggestedReason,
			r.Applied,
			r.AppliedAt,
			r.ApplyAction,
			r.ApplyError,
			r.ReviewAction,
			r.ReviewNote,
			r.ReviewedAt,
			r.ReviewedBy,
		})
	}

	_, err := pool.CopyFrom(ctx,
		pgx.Identifier{schema, "status_sanity_events"},
		cols,
		pgx.CopyFromRows(values),
	)
	return err
}

func streamCandidates(ctx context.Context, pool *pgxpool.Pool, c cfg, yield func(candidate) error) error {
	where := []string{"l.status = ANY($1::text[])"}
	args := []any{c.statuses}
	argN := 2

	if len(c.generations) > 0 {
		where = append(where, fmt.Sprintf("l.generation = ANY($%d::int[])", argN))
		args = append(args, c.generations)
		argN++
	}
	if len(c.onlyIDs) > 0 {
		where = append(where, fmt.Sprintf("l.listing_id = ANY($%d::bigint[])", argN))
		args = append(args, c.onlyIDs)
		argN++
	}
	if c.inactiveOnly {
		where = append(where, "l.listing_is_inactive IS TRUE")
	}
	if c.cadenceDays > 0 && !c.force {
		where = append(where, fmt.Sprintf("(last_audit.last_audited_at IS NULL OR last_audit.last_audited_at < now() - make_interval(days => $%d))", argN))
		args = append(args, c.cadenceDays)
		argN++
	}

	limitSQL := ""
	if c.limit > 0 {
		limitSQL = fmt.Sprintf("LIMIT %d", c.limit)
	}

	q := fmt.Sprintf(`
SELECT
  l.generation,
  l.listing_id,
  l.url,
  l.status,
  l.first_seen,
  l.edited_at,
  l.last_seen,
  l.sold_at,
  l.sold_price,
  l.price,
  l.listing_is_inactive,
  l.listing_is_bidding,
  last_audit.last_audited_at,
  fs.first_marked_sold_at,
  fs.marked_sold_by,
  fs.sold_price_at_mark,
  prev.prev_event_at,
  prev.prev_status,
  prev.prev_source,
  prev.prev_price
FROM "%[1]s".listings l
LEFT JOIN LATERAL (
  SELECT max(observed_at) AS last_audited_at
  FROM "%[1]s".status_sanity_events e
  WHERE e.generation = l.generation AND e.listing_id = l.listing_id
) last_audit ON true
LEFT JOIN LATERAL (
  SELECT
    h.observed_at AS first_marked_sold_at,
    h.source      AS marked_sold_by,
    h.price       AS sold_price_at_mark
  FROM "%[1]s".price_history h
  WHERE h.generation = l.generation
    AND h.listing_id = l.listing_id
    AND h.status = 'sold'
  ORDER BY h.observed_at ASC
  LIMIT 1
) fs ON true
LEFT JOIN LATERAL (
  SELECT
    h.observed_at AS prev_event_at,
    h.status      AS prev_status,
    h.source      AS prev_source,
    h.price       AS prev_price
  FROM "%[1]s".price_history h
  WHERE h.generation = l.generation
    AND h.listing_id = l.listing_id
    AND (fs.first_marked_sold_at IS NULL OR h.observed_at < fs.first_marked_sold_at)
  ORDER BY h.observed_at DESC
  LIMIT 1
) prev ON true
WHERE %[2]s
%[4]s
%[3]s
`, c.schema, strings.Join(where, " AND "), limitSQL, orderBySQL(c.order))

	rows, err := pool.Query(ctx, q, args...)
	if err != nil {
		return fmt.Errorf("query candidates: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var cand candidate
		cand.RowLoadedAt = time.Now().UTC()
		cand.RowLoadedAtEpoch = cand.RowLoadedAt.Unix()

		var url *string
		var markedSoldBy *string
		var prevStatus *string
		var prevSource *string

		var soldPrice, livePrice *int
		var mainIsInactive *bool
		var mainIsBidding *bool

		if err := rows.Scan(
			&cand.Generation,
			&cand.ListingID,
			&url,
			&cand.MainStatus,
			&cand.MainFirstSeen,
			&cand.MainEditedAt,
			&cand.MainLastSeen,
			&cand.MainSoldAt,
			&soldPrice,
			&livePrice,
			&mainIsInactive,
			&mainIsBidding,
			&cand.LastAuditedAt,
			&cand.FirstMarkedSold,
			&markedSoldBy,
			&cand.SoldPriceAtMark,
			&cand.PrevEventAt,
			&prevStatus,
			&prevSource,
			&cand.PrevPrice,
		); err != nil {
			return fmt.Errorf("scan: %w", err)
		}
		if url != nil {
			cand.URLHint = *url
		}
		cand.MainSoldPrice = soldPrice
		cand.MainLivePrice = livePrice
		cand.MainIsInactive = mainIsInactive
		cand.MainIsBidding = mainIsBidding
		cand.MarkedSoldBy = markedSoldBy
		cand.PrevStatus = prevStatus
		cand.PrevSource = prevSource

		if err := yield(cand); err != nil {
			return err
		}
	}
	if rows.Err() != nil {
		return fmt.Errorf("rows: %w", rows.Err())
	}
	return nil
}

func updateMainSold(ctx context.Context, pool *pgxpool.Pool, schema string, soldAt *time.Time, soldPrice *int, observedAt time.Time, generation int, listingID int64) error {
	q := fmt.Sprintf(`UPDATE "%s".listings SET status='sold', sold_at=$1, sold_price=$2, last_seen=$3 WHERE generation=$4 AND listing_id=$5`, schema)
	_, err := pool.Exec(ctx, q, soldAt, soldPrice, observedAt, generation, listingID)
	return err
}

func updateMainRemoved(ctx context.Context, pool *pgxpool.Pool, schema string, observedAt time.Time, generation int, listingID int64) error {
	q := fmt.Sprintf(`UPDATE "%s".listings SET status='removed', last_seen=$1 WHERE generation=$2 AND listing_id=$3`, schema)
	_, err := pool.Exec(ctx, q, observedAt, generation, listingID)
	return err
}

func updateMainUnsellToLive(ctx context.Context, pool *pgxpool.Pool, schema string, observedAt time.Time, generation int, listingID int64) error {
	q := fmt.Sprintf(`UPDATE "%s".listings SET status='live', sold_at=NULL, sold_price=NULL, last_seen=$1 WHERE generation=$2 AND listing_id=$3`, schema)
	_, err := pool.Exec(ctx, q, observedAt, generation, listingID)
	return err
}

func updateMainReviveToLive(ctx context.Context, pool *pgxpool.Pool, schema string, observedAt time.Time, generation int, listingID int64) error {
	q := fmt.Sprintf(`UPDATE "%s".listings SET status='live', last_seen=$1 WHERE generation=$2 AND listing_id=$3`, schema)
	_, err := pool.Exec(ctx, q, observedAt, generation, listingID)
	return err
}

func updateListingInactive(ctx context.Context, pool *pgxpool.Pool, schema string, isInactive bool, observedAt time.Time, metaEditedAt *time.Time, evidenceJSON []byte, generation int, listingID int64) error {
	if isInactive {
		q := fmt.Sprintf(`
UPDATE "%s".listings
SET listing_is_inactive=true,
    listing_inactive_observed_at=$1,
    listing_inactive_evidence=$2,
    listing_inactive_meta_edited_at = COALESCE($3, listing_inactive_meta_edited_at)
WHERE generation=$4 AND listing_id=$5`, schema)
		_, err := pool.Exec(ctx, q, observedAt, json.RawMessage(evidenceJSON), metaEditedAt, generation, listingID)
		return err
	}

	// Clear only if previously inactive.
	q := fmt.Sprintf(`
UPDATE "%s".listings
SET listing_is_inactive=false,
    listing_inactive_observed_at=NULL,
    listing_inactive_evidence=NULL,
    listing_inactive_meta_edited_at=NULL
WHERE generation=$1 AND listing_id=$2 AND listing_is_inactive=true`, schema)

	_, err := pool.Exec(ctx, q, generation, listingID)
	return err
}

/* ===========================
   HTTP + adaptive limiter
   =========================== */

func newHTTPClient(timeout time.Duration) (*http.Client, error) {
	jar, err := cookiejar.New(nil)
	if err != nil {
		return nil, err
	}
	tr := &http.Transport{
		MaxIdleConns:        200,
		MaxIdleConnsPerHost: 50,
		IdleConnTimeout:     60 * time.Second,
		DisableCompression:  false,
	}
	return &http.Client{
		Timeout:   timeout,
		Transport: tr,
		Jar:       jar,
	}, nil
}

func loadSecretValue(explicit, filePath string) (string, error) {
	if strings.TrimSpace(explicit) != "" {
		return strings.TrimSpace(explicit), nil
	}
	if strings.TrimSpace(filePath) == "" {
		return "", nil
	}
	b, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}
	s := strings.TrimSpace(string(b))
	if s == "" {
		return "", nil
	}
	// First line only (avoid accidental multiline secrets).
	lines := strings.Split(s, "\n")
	return strings.TrimSpace(lines[0]), nil
}

func doRequest(ctx context.Context, client *http.Client, lim *dynLimiter, method, url string, body io.Reader, rc requestConfig) (int, error) {
	if err := lim.Wait(ctx); err != nil {
		return 0, err
	}
	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return 0, err
	}
	setHeaders(req, rc)
	resp, err := client.Do(req)
	if err != nil {
		lim.OnThrottle()
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == 429 || resp.StatusCode == 403 || resp.StatusCode >= 500 {
		lim.OnThrottle()
	} else {
		lim.OnOK()
	}
	return resp.StatusCode, nil
}

func smartGET(ctx context.Context, client *http.Client, lim *dynLimiter, url string, rc requestConfig, maxBodyBytes int64, retryMax int, throttleSleep, retryBackoffBase, jitterMax time.Duration) ([]byte, int, error) {
	var lastErr error
	var lastCode int
	for attempt := 0; attempt <= retryMax; attempt++ {
		if err := lim.Wait(ctx); err != nil {
			return nil, 0, err
		}
		req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
		if err != nil {
			return nil, 0, err
		}
		setHeaders(req, rc)

		resp, err := client.Do(req)
		if err != nil {
			lim.OnThrottle()
			lastErr = err
			sleepWithJitter(ctx, retryBackoffBase*time.Duration(attempt+1), jitterMax)
			continue
		}
		code := resp.StatusCode
		lastCode = code

		b, rerr := readAllLimit(resp.Body, maxBodyBytes)
		_ = resp.Body.Close()

		if rerr != nil {
			lim.OnThrottle()
			lastErr = rerr
			sleepWithJitter(ctx, retryBackoffBase*time.Duration(attempt+1), jitterMax)
			continue
		}

		if code == 429 || code == 403 || code >= 500 {
			lim.OnThrottle()
			lastErr = fmt.Errorf("throttled_http_%d", code)
			sleepWithJitter(ctx, throttleSleep*time.Duration(attempt+1), jitterMax)
			continue
		}

		lim.OnOK()
		return b, code, nil
	}
	if lastErr == nil {
		lastErr = errors.New("unknown_http_error")
	}
	return nil, lastCode, lastErr
}

func setHeaders(req *http.Request, rc requestConfig) {
	if strings.TrimSpace(rc.userAgent) != "" {
		req.Header.Set("User-Agent", rc.userAgent)
	}
	req.Header.Set("Accept", "application/json")
	if strings.TrimSpace(rc.acceptLanguage) != "" {
		req.Header.Set("Accept-Language", rc.acceptLanguage)
	}
	req.Header.Set("Cache-Control", "no-cache")
	req.Header.Set("Pragma", "no-cache")
	if strings.TrimSpace(rc.authHeader) != "" {
		// Secret at runtime; do not commit.
		req.Header.Set("Authorization", rc.authHeader)
	}
}

func readAllLimit(r io.Reader, limit int64) ([]byte, error) {
	if limit <= 0 {
		return io.ReadAll(r)
	}
	lr := io.LimitReader(r, limit)
	return io.ReadAll(lr)
}

func sleepWithJitter(ctx context.Context, base time.Duration, jitterMax time.Duration) {
	if base <= 0 {
		return
	}
	j := time.Duration(0)
	if jitterMax > 0 {
		var b [2]byte
		_, _ = rand.Read(b[:])
		n := int(b[0])<<8 | int(b[1])
		j = time.Duration(n) % jitterMax
	}
	d := base + j
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return
	case <-t.C:
		return
	}
}

type dynLimiter struct {
	mu    sync.Mutex
	rps   float64
	max   float64
	min   float64
	step  float64
	down  float64
	burst float64

	tokens float64
	last   time.Time

	okEvery int64
	okCnt   int64
}

func newDynLimiter(startRPS, maxRPS, minRPS, stepUpRPS, downMult, burstFactor float64, okEvery int64) *dynLimiter {
	if startRPS <= 0 {
		startRPS = 1
	}
	if maxRPS < startRPS {
		maxRPS = startRPS
	}
	if minRPS <= 0 {
		minRPS = 0.25
	}
	if stepUpRPS <= 0 {
		stepUpRPS = 0.25
	}
	if downMult <= 0 || downMult >= 1 {
		downMult = 0.7
	}
	if burstFactor <= 0 {
		burstFactor = 2.0
	}
	if okEvery <= 0 {
		okEvery = 30
	}
	now := time.Now()
	return &dynLimiter{
		rps:     startRPS,
		max:     maxRPS,
		min:     minRPS,
		step:    stepUpRPS,
		down:    downMult,
		burst:   math.Max(1.0, startRPS*burstFactor),
		tokens:  math.Max(1.0, startRPS*burstFactor),
		last:    now,
		okEvery: okEvery,
	}
}

func (l *dynLimiter) Snapshot() (rps float64, burst float64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.rps, l.burst
}

func (l *dynLimiter) Wait(ctx context.Context) error {
	for {
		l.mu.Lock()
		now := time.Now()
		elapsed := now.Sub(l.last).Seconds()
		if elapsed > 0 {
			l.tokens = math.Min(l.burst, l.tokens+elapsed*l.rps)
			l.last = now
		}
		if l.tokens >= 1.0 {
			l.tokens -= 1.0
			l.mu.Unlock()
			return nil
		}
		need := 1.0 - l.tokens
		sec := need / l.rps
		if l.rps <= 0 {
			sec = 1
		}
		wait := time.Duration(sec * float64(time.Second))
		l.mu.Unlock()

		t := time.NewTimer(wait)
		select {
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		case <-t.C:
		}
	}
}

func (l *dynLimiter) OnOK() {
	n := atomic.AddInt64(&l.okCnt, 1)
	if n%l.okEvery != 0 {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.rps = math.Min(l.max, l.rps+l.step)
	l.burst = math.Max(1.0, l.rps*2.0)
	l.tokens = math.Min(l.tokens, l.burst)
}

func (l *dynLimiter) OnThrottle() {
	atomic.StoreInt64(&l.okCnt, 0)
	l.mu.Lock()
	defer l.mu.Unlock()
	l.rps = math.Max(l.min, l.rps*l.down)
	l.burst = math.Max(1.0, l.rps*2.0)
	l.tokens = math.Min(l.tokens, l.burst)
}

/* ===========================
   Generic parsing helpers
   =========================== */

func anyToInt(v any) (int, bool) {
	switch t := v.(type) {
	case float64:
		return int(t), true
	case json.Number:
		i, _ := t.Int64()
		return int(i), true
	case string:
		// Handles "12 345" etc by stripping non-digits.
		digits := make([]rune, 0, len(t))
		for _, r := range t {
			if r >= '0' && r <= '9' {
				digits = append(digits, r)
			}
		}
		if len(digits) == 0 {
			return 0, false
		}
		n, err := strconv.Atoi(string(digits))
		if err != nil {
			return 0, false
		}
		return n, true
	case int:
		return t, true
	case int64:
		return int(t), true
	default:
		return 0, false
	}
}

/* ===========================
   Export helpers
   =========================== */

func exportLatestMismatches(ctx context.Context, pool *pgxpool.Pool, c cfg) error {
	q := fmt.Sprintf(`
WITH latest AS (
  SELECT DISTINCT ON (generation, listing_id)
    generation, listing_id, url_used, observed_at, detected_status, main_status,
    mismatch_status, mismatch_inactive, suggested_status, suggested_is_inactive, suggested_reason,
    http_status, payload_ok, run_id
  FROM "%s".status_sanity_events
  WHERE mismatch_status OR mismatch_inactive
  ORDER BY generation, listing_id, observed_at DESC
)
SELECT * FROM latest
ORDER BY observed_at DESC
LIMIT 200
`, c.schema)

	rows, err := pool.Query(ctx, q)
	if err != nil {
		return err
	}
	defer rows.Close()

	fmt.Printf("generation\tlisting_id\tobserved_at\thttp\tpayload_ok\tdetected\tmain\tmismatch_status\tmismatch_inactive\tsuggested_status\tsuggested_inactive\treason\trun_id\turl_used\n")
	for rows.Next() {
		var gen int
		var id int64
		var urlUsed *string
		var obs time.Time
		var detected, main string
		var ms, mi bool
		var sugStatus *string
		var sugInactive *bool
		var reason *string
		var httpStatus *int
		var payloadOK *bool
		var runID string
		if err := rows.Scan(&gen, &id, &urlUsed, &obs, &detected, &main, &ms, &mi, &sugStatus, &sugInactive, &reason, &httpStatus, &payloadOK, &runID); err != nil {
			return err
		}
		fmt.Printf("%d\t%d\t%s\t%v\t%v\t%s\t%s\t%t\t%t\t%v\t%v\t%v\t%s\t%s\n",
			gen, id, obs.Format(time.RFC3339), valOrNil(httpStatus), valOrNil(payloadOK),
			detected, main, ms, mi,
			valOrNil(sugStatus), valOrNil(sugInactive), valOrNil(reason), runID,
			valOrNil(urlUsed),
		)
	}
	return rows.Err()
}

/* ===========================
   CSV writer
   =========================== */

type csvWriter struct {
	f   *os.File
	w   *bufio.Writer
	hdr bool
}

func newCSVWriter(path string) (*csvWriter, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	return &csvWriter{f: f, w: bufio.NewWriterSize(f, 1<<20)}, nil
}

func (c *csvWriter) Close() error {
	if c == nil {
		return nil
	}
	_ = c.w.Flush()
	return c.f.Close()
}

func (c *csvWriter) Flush() error {
	if c == nil {
		return nil
	}
	return c.w.Flush()
}

func (c *csvWriter) Write(r auditRow) error {
	if c == nil {
		return nil
	}
	if !c.hdr {
		c.hdr = true
		_, _ = c.w.WriteString(strings.Join([]string{
			"run_id", "observed_at", "generation", "listing_id", "url_used",
			"http_status", "payload_ok", "detected_status", "detected_price", "detected_price_src", "detected_sold_at",
			"detected_is_inactive", "inactive_key_seen", "payload_src",
			"main_status", "main_sold_at", "main_sold_price", "main_is_inactive", "main_last_seen",
			"mismatch_status", "mismatch_inactive",
			"suggested_status", "suggested_is_inactive", "suggested_reason",
			"applied", "apply_action", "apply_error",
		}, ",") + "\n")
	}
	fields := []string{
		r.RunID,
		r.ObservedAt.Format(time.RFC3339),
		strconv.Itoa(r.Generation),
		strconv.FormatInt(r.ListingID, 10),
		csvEscape(r.URLUsed),
		intToStrPtr(nullableIntPtr(r.HTTPStatus)),
		strconv.FormatBool(r.PayloadOK),
		r.DetectedStatus,
		intToStrPtr(r.DetectedPrice),
		csvEscape(r.DetectedPriceSrc),
		timeToStrPtr(r.DetectedSoldAt),
		boolToStrPtr(r.DetectedIsInactive),
		strconv.FormatBool(r.InactiveKeySeen),
		csvEscape(r.PayloadSrc),
		r.MainStatus,
		timeToStrPtr(r.MainSoldAt),
		intToStrPtr(r.MainSoldPrice),
		boolToStrPtr(r.MainIsInactive),
		timeToStrPtr(r.MainLastSeen),
		strconv.FormatBool(r.MismatchStatus),
		strconv.FormatBool(r.MismatchInactive),
		strPtrToStr(r.SuggestedStatus),
		boolPtrToStr(r.SuggestedIsInactive),
		strPtrToStr(r.SuggestedReason),
		strconv.FormatBool(r.Applied),
		strPtrToStr(r.ApplyAction),
		strPtrToStr(r.ApplyError),
	}
	_, err := c.w.WriteString(strings.Join(fields, ",") + "\n")
	return err
}

func csvEscape(s string) string {
	if s == "" {
		return ""
	}
	if strings.ContainsAny(s, ",\"\n\r\t") {
		s = strings.ReplaceAll(s, "\"", "\"\"")
		return "\"" + s + "\""
	}
	return s
}

func strPtrToStr(s *string) string {
	if s == nil {
		return ""
	}
	return csvEscape(*s)
}
func boolPtrToStr(b *bool) string {
	if b == nil {
		return ""
	}
	if *b {
		return "true"
	}
	return "false"
}

// boolToStrPtr is a compatibility alias used by some CSV call sites.
// It matches boolPtrToStr semantics: empty string if nil.
func boolToStrPtr(b *bool) string {
	return boolPtrToStr(b)
}

func intToStrPtr(n *int) string {
	if n == nil {
		return ""
	}
	return strconv.Itoa(*n)
}
func timeToStrPtr(t *time.Time) string {
	if t == nil {
		return ""
	}
	return t.Format(time.RFC3339)
}

/* ===========================
   misc helpers
   =========================== */

func nullableInt(n int) *int    { return &n }
func nullableIntPtr(n int) *int { return &n }
func intPtr(n int) *int         { return &n }
func boolPtr(b bool) *bool      { return &b }
func nullIfEmpty(s string) *string {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	ss := s
	return &ss
}
func intSliceOrNull(s []int) any {
	if len(s) == 0 {
		return nil
	}
	return s
}
func isSafeIdent(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !(r == '_' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')) {
			return false
		}
	}
	return true
}
func valOrNil[T any](p *T) any {
	if p == nil {
		return nil
	}
	return *p
}
func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(2)
}
