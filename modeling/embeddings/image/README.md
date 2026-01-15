# build_img_vec512_openclip_pg_public.py

This script builds **one 512‑D image embedding per marketplace listing** using an OpenCLIP image encoder, and writes the result into Postgres using **pgvector**.

It is designed as a **job-mode batch**: you run it periodically (or on demand), and it will skip work that has already been computed unless you force a rebuild.

---

## What the script does

At a high level, the script:

1. **Finds eligible listings** in the listings table:
   - requires `edited_date IS NOT NULL`
   - requires `url IS NOT NULL`
   - excludes rows marked as spam (`spam IS NULL`)
   - if not `--force`, it **skips listings already embedded** for the same `(source, model_rev, pca_rev)`.

2. **Reads image inventory** for each listing from the image assets table:
   - uses `storage_path` to map a DB row → a file on disk under `IMAGE_ROOT_DIR`.

3. **Selects up to Top‑K images per listing** (default `--topk 5`):
   - by default, excludes:
     - stock photos (`is_stock_photo = true`)
     - battery screenshots (`battery_screenshot = true`)
   - ranks candidates by a simple heuristic based on per-image quality flags:
     - `score = 10 * photo_quality_level + 3 * background_clean_level`
   - if exclusions remove all images, it **falls back** to the first available images.

4. **Computes OpenCLIP embeddings** in batches:
   - loads OpenCLIP model + preprocess transform
   - runs inference on GPU if available (default `--device cuda` when CUDA is present)
   - optional fp16 path (`--fp16` default on CUDA).

5. **Aggregates to a single listing vector**:
   - L2-normalizes each per-image vector
   - averages the vectors per listing
   - L2-normalizes the mean vector again (stable listing representation).

6. **Upserts** the listing-level vector into Postgres:
   - key: `(listing_id, edited_date, source, model_rev, pca_rev)`
   - payload: `vec (vector(512))`, `n_images`, `updated_at`.

7. **Benchmark mode**:
   - `--benchmark-images N` runs inference on only the first `N` selected images, prints throughput, and **never writes to DB**.

---

## Database objects expected

The script expects four logical tables:

### 1) Listings table (eligibility + edited_date)
Default (can be overridden via env vars):

- schema: `LISTINGS_SCHEMA=iPhone`
- table: `LISTINGS_TABLE=iphone_listings`

Required columns:
- `generation` (int)
- `listing_id` (bigint/int)
- `edited_date` (timestamp/timestamptz)
- `url` (text)
- `spam` (text nullable)

### 2) Image assets table (inventory of images)
Default:
- schema: `ASSETS_SCHEMA=iPhone`
- table: `ASSETS_TABLE=iphone_image_assets`

Required columns:
- `generation` (int)
- `listing_id` (bigint/int)
- `image_index` (int)
- `storage_path` (text)

### 3) Per-image features table (optional; used for selection only)
Default:
- schema: `ML_SCHEMA=ml`
- table: `IMAGE_FEATURES_TABLE=iphone_image_features_v1`

Used columns:
- `feature_version` (int)
- `generation`, `listing_id`, `image_index`
- `is_stock_photo` (bool)
- `battery_screenshot` (bool)
- `photo_quality_level` (smallint/int)
- `background_clean_level` (smallint/int)

If your schema names differ, override them with the env vars listed below.

### 4) Listing embeddings table (output; pgvector)
Default:
- schema: `ML_SCHEMA=ml`
- table: `VECTORS_TABLE=img_vec512_v1`

Required columns:
- `listing_id` (bigint/int)
- `edited_date` (timestamptz)
- `source` (text)
- `model_rev` (text)
- `pca_rev` (text)
- `vec` (vector(512))
- `n_images` (int)
- `updated_at` (timestamptz)

**Suggested DDL** (example, public-safe; adjust types as needed):

```sql
CREATE SCHEMA IF NOT EXISTS ml;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS ml.img_vec512_v1 (
  listing_id  bigint      NOT NULL,
  edited_date timestamptz NOT NULL,
  source      text        NOT NULL,
  model_rev   text        NOT NULL,
  pca_rev     text        NOT NULL DEFAULT '',
  vec         vector(512) NOT NULL,
  n_images    integer     NOT NULL,
  updated_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (listing_id, edited_date, source, model_rev, pca_rev)
);
```

---

## Runtime configuration

### Required env vars

- `PG_DSN`  
  Postgres connection string. **Required**.  
  This script does not provide a default and will fail fast if it is missing.

### Optional env vars

- `IMAGE_ROOT_DIR` (default: `./listing_images`)  
  Root folder on disk that contains image files. The script loads images from:
  `os.path.join(IMAGE_ROOT_DIR, storage_path)`.

- `IMAGE_FEATURE_VERSION` (default: `1`)  
  Used when joining to the per-image features table.

### Optional schema/table override env vars (validated identifiers)

These are useful if you want to use the script without editing SQL strings:

- `LISTINGS_SCHEMA` (default: `iPhone`)
- `LISTINGS_TABLE` (default: `iphone_listings`)
- `ASSETS_SCHEMA` (default: `iPhone`)
- `ASSETS_TABLE` (default: `iphone_image_assets`)
- `ML_SCHEMA` (default: `ml`)
- `IMAGE_FEATURES_TABLE` (default: `iphone_image_features_v1`)
- `VECTORS_TABLE` (default: `img_vec512_v1`)

**Note:** the script validates these values as SQL identifiers (letters/numbers/underscore only) to avoid SQL injection via env vars.

---

## CLI reference

Run:

```bash
python build_img_vec512_openclip_pg_public.py --help
```

### Job sizing / selection

- `--limit-listings INT` (default: `50000`)  
  Upper bound on the number of listings considered per run.

- `--topk INT` (default: `5`)  
  Images per listing to embed.

- `--force`  
  Re-embed and **overwrite** existing rows for the current `(source, model_rev, pca_rev)`.

### Image filtering

- `--exclude-stock` (default behavior)  
  Exclude stock photos when selecting Top‑K images.

- `--include-stock`  
  Include stock photos in candidate selection.

- `--exclude-screenshots` (default behavior)  
  Exclude screenshots (e.g., battery screenshots) when selecting Top‑K images.

- `--include-screenshots`  
  Include screenshots in candidate selection.

### Model identity / versioning

- `--model-name TEXT` (default: `ViT-B-32`)  
  OpenCLIP model architecture.

- `--pretrained TEXT` (default: `laion2b_s34b_b79k`)  
  OpenCLIP pretrained weight tag.

- `--source TEXT` (default: `listing_images_topk5_v1`)  
  A free-form label stored in the output table so you can track how embeddings were built.

- `--pca-rev TEXT` (default: empty)  
  Extra revision tag if you later add a PCA post-processing step. Stored as part of the uniqueness key.

### Performance / device

- `--batch-size INT` (default: `96`)  
  Inference batch size.

- `--num-workers INT` (default: `6`)  
  DataLoader workers for image decode / preprocessing.

- `--device TEXT` (default: `cuda` if available else `cpu`)  
  Torch device.

- `--fp16` / `--no-fp16`  
  Enable/disable fp16 model weights (default enabled on CUDA).

### Dry-run / benchmarking

- `--dry-run`  
  Runs everything but **skips DB writes**.

- `--benchmark-images INT`  
  Runs inference on the first N selected images and prints throughput; **never writes** to DB.

---

## Example runs (no secrets embedded)

### PowerShell

```powershell
# Inject PG_DSN from your secret storage (example: local file)
$env:PG_DSN = (Get-Content .\secrets\PG_DSN.txt -Raw)

# Point to your image folder
$env:IMAGE_ROOT_DIR = ".\listing_images"

# Benchmark 2k images without writing
python .\build_img_vec512_openclip_pg_public.py --benchmark-images 2000 --limit-listings 5000 --dry-run

# Real run
python .\build_img_vec512_openclip_pg_public.py --limit-listings 50000 --topk 5 --batch-size 96 --num-workers 6
```

### Bash

```bash
export PG_DSN="$(cat ./secrets/pg_dsn.txt)"
export IMAGE_ROOT_DIR="./listing_images"

# Real run
python ./build_img_vec512_openclip_pg_public.py --limit-listings 50000 --topk 5 --batch-size 96 --num-workers 6

# Force overwrite for this source/model identity
python ./build_img_vec512_openclip_pg_public.py --force --limit-listings 5000
```

---

## Operational notes

- **Disk I/O dominates** on many setups. If throughput is low, first check:
  - storage speed (SSD vs network drive)
  - number of DataLoader workers (`--num-workers`)
  - image decode cost (very large JPEGs)
  - GPU utilization

- **Idempotency**: the script is safe to run repeatedly; without `--force` it only computes missing embeddings for the current identity key.

- **Leakage**: the output table uses `(edited_date, source, model_rev, pca_rev)` so you can build as-of snapshots and avoid mixing embeddings across different listing edit states.

