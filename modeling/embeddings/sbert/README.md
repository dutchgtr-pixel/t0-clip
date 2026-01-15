# SBERT Vector Upsert (Title + Description + Captions) — Public Release

This utility builds **semantic text embeddings** for marketplace-style listings (title, description, and optional image captions), projects them to a fixed dimension via **PCA**, and **upserts** them into Postgres using **pgvector**.

It is designed for **continuous refresh**: if media/captions arrive after an embedding row was created, the script can refresh embeddings without getting stuck in reprocessing loops.

---

## What it does

For each `(listing_id, edited_at)` snapshot:

1. **Extract text** from the listings table:
   - `title`
   - `description`
   - aggregated captions from an image-assets table (`string_agg(...) ORDER BY image_index`)
2. **Merge** non-empty parts into one string.
3. Compute a stable input hash: `text_sha1 = sha1(merged_text)`.
4. Encode with **SBERT** (native embedding dimension depends on the model).
5. Project to **D ∈ {64, 128, 256}** using **IncrementalPCA**.
6. **Upsert** into `ml.sbert_vec{D}_v1` with:
   - primary key `(listing_id, edited_at, source, model_rev, pca_rev)`
   - fields `text_sha1`, `vec`, and `created_at`

---

## Output tables (created automatically)

When run with `--ensure-table`, the script creates:

- `ml.sbert_vec64_v1`  (`vec vector(64)`)
- `ml.sbert_vec128_v1` (`vec vector(128)`)
- `ml.sbert_vec256_v1` (`vec vector(256)`)

Each table schema is:

```sql
CREATE TABLE IF NOT EXISTS ml.sbert_vec{D}_v1 (
  listing_id     bigint      NOT NULL,
  edited_at      timestamptz NOT NULL,
  source         text        NOT NULL,
  model_rev      text        NOT NULL,
  pca_rev        text        NOT NULL,
  text_sha1      text        NOT NULL,
  vec            vector(D)   NOT NULL,
  created_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (listing_id, edited_at, source, model_rev, pca_rev)
);
```

The script also creates two pragmatic lookup indexes (not ANN):

```sql
CREATE INDEX IF NOT EXISTS sbert_vec{D}_v1_lookup
  ON ml.sbert_vec{D}_v1 (source, model_rev, pca_rev, edited_at DESC);

CREATE INDEX IF NOT EXISTS sbert_vec{D}_v1_listing_edit
  ON ml.sbert_vec{D}_v1 (listing_id, edited_at DESC);
```

---

## Input requirements (interface contract)

### Listings table (default: `public.listings`)
Must include these columns (names configurable via CLI):

- `listing_id` (bigint / int)
- `edited_at` (timestamptz) — the snapshot timestamp (t0)
- `title` (text, nullable)
- `description` (text, nullable)
- `media_scraped_at` (timestamptz, nullable) — optional but recommended for refresh logic
- `spam_flag` (nullable boolean or any nullable marker) — non-null indicates spam; default behavior excludes spam

### Image assets table (default: `public.listing_image_assets`)
Must include:

- `listing_id` (bigint / int)
- `caption_text` (text, nullable)
- `image_index` (int) — used only for ordering captions deterministically

If you do not have `image_index`, create a view that provides it (e.g., constant `0`).

---

## Installation

### Python dependencies
```bash
pip install numpy psycopg2-binary sentence-transformers scikit-learn
```

### Postgres dependencies
You need pgvector installed in the database. The script will run:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## Quickstart

### 1) Set your Postgres DSN
Use environment variables (recommended) rather than hardcoding DSNs.

**bash**
```bash
export PG_DSN="postgresql://USER:PASSWORD@HOST:5432/DBNAME"
```

**PowerShell**
```powershell
$env:PG_DSN = "postgresql://USER:PASSWORD@HOST:5432/DBNAME"
```

### 2) Fit PCA and write vectors (example: 64-D)
This fits PCA from a sample of the most recent rows and then backfills/upserts.

```bash
python sbert_vec_upsert_title_desc_cap_PUBLIC.py \
  --pg-dsn "$PG_DSN" \
  --ensure-table \
  --pca-dim 64 \
  --pca-path ./pca_models/pca_64.pkl \
  --fit-pca
```

### 3) Repeat for 128-D and 256-D
```bash
python sbert_vec_upsert_title_desc_cap_PUBLIC.py \
  --pg-dsn "$PG_DSN" \
  --ensure-table \
  --pca-dim 128 \
  --pca-path ./pca_models/pca_128.pkl \
  --fit-pca

python sbert_vec_upsert_title_desc_cap_PUBLIC.py \
  --pg-dsn "$PG_DSN" \
  --ensure-table \
  --pca-dim 256 \
  --pca-path ./pca_models/pca_256.pkl \
  --fit-pca
```

---

## Continuous refresh usage

On each run, the script selects candidates that are either:

- missing from the target table, or
- **stale** because `media_scraped_at > created_at` for the existing embedding row

If a row is selected only due to refresh timestamps but its **merged text hash is unchanged**, the default behavior is to **touch** `created_at` to stop the row being re-selected forever. This is the purpose of the robust “touch unchanged refresh rows” logic.

If you want to disable that behavior (not recommended):

```bash
python sbert_vec_upsert_title_desc_cap_PUBLIC.py \
  --pg-dsn "$PG_DSN" \
  --pca-dim 64 \
  --pca-path ./pca_models/pca_64.pkl \
  --no-touch-unchanged
```

---

## CLI reference

Run:
```bash
python sbert_vec_upsert_title_desc_cap_PUBLIC.py --help
```

Key options:

### Database / output
- `--pg-dsn`  
  Postgres DSN. Defaults to env `PG_DSN`.
- `--ensure-table`  
  Create `ml` schema + `ml.sbert_vec{D}_v1` + base indexes if missing.

### Embedding & versioning
- `--source` (default: `title_desc_cap`)  
  Label for the embedded text composition.
- `--model` (default: `bert-base-uncased`)  
  HuggingFace model name or path.
- `--model-rev` (default: `meanpool_v1`)  
  Freeform revision label recorded in the output table.

### PCA
- `--pca-dim` (default: 64; allowed: 64, 128, 256)  
  Output vector dimension and target table selector.
- `--pca-path`  
  Path to PCA pickle file. Required unless `--fit-pca` writes it.
- `--fit-pca`  
  Fit PCA from database texts and overwrite `--pca-path`, then proceed to backfill.
- `--fit-only`  
  Fit PCA and exit (do not backfill/upsert).
- `--pca-fit-limit` (default: 30000)  
  Max number of rows to sample for PCA fitting.

### Throughput controls
- `--db-batch` (default: 2000)  
  Candidate fetch batch size per DB round trip.
- `--enc-batch` (default: 64)  
  Encoder batch size (increase carefully if using GPU).
- `--max-rows` (default: 0 = unlimited)  
  Upper limit on rows processed in this run.
- `--sleep-s` (default: 0)  
  Optional sleep between DB batches.

### Device / tokenizer
- `--device` (auto|cpu|cuda)  
  `auto` attempts to use CUDA if available.
- `--max-seq-len` (default: 256)  
  Max sequence length for tokenization (affects speed and quality).

### Spam filtering
- `--include-spam`  
  Include spam rows too. Default behavior excludes rows where `spam_flag` is non-null.

If your dataset does not have a spam concept, set `SPAM_FLAG_COL` to empty in the environment or point it to a nullable column that is always NULL.

### Source schema customization
Tables and column names are configurable, but must be simple identifiers (or `schema.table` for tables).

- `--listings-table` (default: `public.listings`)
- `--image-assets-table` (default: `public.listing_image_assets`)
- `--listing-id-col` (default: `listing_id`)
- `--edited-at-col` (default: `edited_at`)
- `--title-col` (default: `title`)
- `--description-col` (default: `description`)
- `--media-scraped-at-col` (default: `media_scraped_at`)
- `--spam-flag-col` (default: `spam_flag`)
- `--caption-text-col` (default: `caption_text`)
- `--image-index-col` (default: `image_index`)

---

## Similarity search examples

### 1) Get the latest embedding for one listing_id
```sql
SELECT *
FROM ml.sbert_vec64_v1
WHERE listing_id = 123
ORDER BY edited_at DESC
LIMIT 1;
```

### 2) Nearest neighbors (cosine distance)
If you have a query vector `:q` (vector(64)):

```sql
SELECT listing_id, edited_at, (vec <=> :q) AS cosine_distance
FROM ml.sbert_vec64_v1
WHERE source='title_desc_cap' AND model_rev='meanpool_v1'
ORDER BY vec <=> :q
LIMIT 50;
```

### Optional: add an ANN index (pgvector)
For large tables, consider HNSW or IVFFlat depending on your pgvector version and workload. Example HNSW:

```sql
CREATE INDEX IF NOT EXISTS sbert_vec64_v1_hnsw_cos
ON ml.sbert_vec64_v1
USING hnsw (vec vector_cosine_ops);
```

---

## Operational notes

- Fit **separate PCA artifacts** for each target dimension (64/128/256).
- If you change the SBERT model or pooling strategy, treat that as a new `model_rev` and re-fit PCA.
- The output key includes `model_rev` and `pca_rev` to allow multiple embedding generations to coexist safely.

---

## Public release safety

This script is marketplace-agnostic:
- It uses `listing_id` (not platform-specific IDs).
- All sensitive connectivity is via `PG_DSN` environment variable.
- No selectors, URLs, cookies, or tokens are embedded.
