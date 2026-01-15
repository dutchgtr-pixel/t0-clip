-- 00_create_tables.sql
-- Creates the WOE anchor artifact tables.
-- NOTE: fold_id is nullable by design. Uniqueness is enforced via partial unique indexes (see 01_nullable_fold_id_migration.sql).

CREATE SCHEMA IF NOT EXISTS ml;

CREATE TABLE IF NOT EXISTS ml.woe_anchor_model_registry_v1 (
  model_key text PRIMARY KEY,
  train_cutoff_ts timestamptz NOT NULL,
  half_life_days float8 NOT NULL,
  eps float8 NOT NULL,
  n_folds int NOT NULL,
  band_schema_version int NOT NULL,
  code_sha256 text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  is_active boolean NOT NULL DEFAULT false,

  -- fields added later (see 02_add_registry_fields.sql):
  base_rate float8,
  base_logit float8,
  dsold_t1 float8,
  dsold_t2 float8,
  dsold_t3 float8,
  dsold_t4 float8
);

CREATE TABLE IF NOT EXISTS ml.woe_anchor_cuts_v1 (
  model_key text NOT NULL,
  fold_id int NULL,              -- NULL = final mapping
  c1 float8,
  c2 float8,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ml.woe_anchor_map_v1 (
  model_key text NOT NULL,
  fold_id int NULL,              -- NULL = final mapping
  band_name text NOT NULL,
  band_value text NOT NULL,
  woe float8 NOT NULL,
  sum_w_pos float8,
  sum_w_neg float8,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ml.woe_anchor_scores_v1 (
  model_key text NOT NULL,
  generation int NOT NULL,
  listing_id bigint NOT NULL,
  t0 timestamptz NOT NULL,
  fold_id int NULL,
  is_oof boolean NOT NULL,
  woe_logit float8,
  woe_p float8,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (model_key, generation, listing_id, t0)
);
