# Redaction report (public release)

This report summarizes the sanitization work performed to prepare the provided build and documentation artifacts for safe public release.

## Scope (artifacts processed)

- `entrypoint_survival.sh` → `entrypoint_survival.public.sh` fileciteturn0file4  
- `Dockerfile.survival` → `Dockerfile.survival.public`  
- `Documentation inactive table.txt` → `Documentation_inactive_state_events.public.md` fileciteturn0file1  
- `Useful Sql query informative.txt` → `Useful_SQL_queries.public.sql` fileciteturn0file2  
- `Updated documentation extensive end-to-end technical.txt` → `RUNBOOK_survival_marketplace.public.md` fileciteturn0file3  
- `Trigger  Function  older21 days.txt` → `Trigger_older21days_proof_pack.public.md`  
- `.env.example` → `.env.example.public`  

(Additionally, the main Go program was previously sanitized to `survival_marketplace.go`.) fileciteturn0file5

## Summary of key changes

### 1) Secrets / credentials / sensitive access

- Removed hardcoded database connection strings that contained embedded credentials and internal hostnames.
- Replaced all database connectivity with environment-variable driven configuration (`PG_DSN`, `PG_SCHEMA`).
- Ensured runtime configuration is injectable via environment variables and does not rely on committed secrets.

### 2) Platform identifiers and unique fingerprints

- Removed platform branding/domain references and target-specific terminology from operational documentation.
- Standardized the primary identifier to `listing_id` in all public docs/SQL, matching the sanitized code.
- Replaced any platform-specific parsing explanations with a generic **adapter interface** model (mock + simple HTTP JSON adapter).

### 3) Personal data / privacy

- Removed production query outputs and real operational logs from the trigger proof document.
- Updated examples to use synthetic listing IDs (e.g., `listing_id=100011`) and reserved example URLs (e.g., `https://marketplace.example/...`).

## Placeholders introduced / enforced

The public artifacts now consistently rely on:

- `PG_DSN` (required for DB-writing modes)
- `PG_SCHEMA`
- `GENERATION` / `GEN_LIST`
- `MODE`
- `MARKETPLACE_ADAPTER` (`mock` | `http-json`)
- `MARKETPLACE_BASE_URL`
- `HTTP_USER_AGENT`
- `REQUEST_RPS`, `WORKERS`, and retry/throttle knobs

## File-by-file notes

### `Dockerfile.survival.public`
- Removed embedded DSN defaults from `ENV`.
- Added `GO_BUILD_TARGET` and `ENTRYPOINT_PATH` build args to keep the Docker build generic and repo-structure-agnostic.
- Set safe defaults that do not require a database secret at image build time.

### `entrypoint_survival.public.sh`
- Removed embedded DSN defaults and any generation hardcoding.
- Added `GEN_LIST` looping and safe defaults (`MODE=diagnose`) so the container can start without a database.
- All configuration is now environment-driven; DB usage is gated behind `WRITE_DB=true`.

### `Documentation_inactive_state_events.public.md`
- Rewrote the table documentation to be platform-agnostic.
- Aligned table/column names to the sanitized code: `inactive_state_events`, `listing_id`, `is_inactive`, and snapshot columns in `listings`.

### `Useful_SQL_queries.public.sql`
- Updated operational queries to use `public.listings`, `public.price_history`, and `public.inactive_state_events`.
- Removed references to platform-specific statuses; inactivity is queried via the dedicated flag/event log.

### `RUNBOOK_survival_marketplace.public.md`
- Replaced target-specific runbook content with a public “how it works” + “how to run with synthetic data” guide.
- Added a “Public release notes” section describing what is included vs intentionally omitted.

### `Trigger_older21days_proof_pack.public.md`
- Converted the proof pack into a generic template (queries + expected results), with no real outputs.

## Final scan results (public artifacts)

- Disallowed platform-identifying brand/domain terms: **0 matches** across the public artifacts and `survival_marketplace.go`.
- Credential-like DSN patterns (e.g., embedded `user:password@`): **0 matches**.
- No API keys, bearer tokens, cookie/session strings, or private headers are present in the public artifacts.

