-- 01_nullable_fold_id_migration.sql
-- Use this ONLY if you previously created PKs involving fold_id (forcing NOT NULL).
-- This migration makes fold_id nullable and enforces uniqueness via partial unique indexes.

BEGIN;

ALTER TABLE ml.woe_anchor_cuts_v1 DROP CONSTRAINT IF EXISTS woe_anchor_cuts_v1_pkey;
ALTER TABLE ml.woe_anchor_map_v1  DROP CONSTRAINT IF EXISTS woe_anchor_map_v1_pkey;

ALTER TABLE ml.woe_anchor_cuts_v1 ALTER COLUMN fold_id DROP NOT NULL;
ALTER TABLE ml.woe_anchor_map_v1  ALTER COLUMN fold_id DROP NOT NULL;

-- Final cuts uniqueness (fold_id IS NULL)
CREATE UNIQUE INDEX IF NOT EXISTS woe_anchor_cuts_v1_final_ux
  ON ml.woe_anchor_cuts_v1 (model_key)
  WHERE fold_id IS NULL;

-- Fold cuts uniqueness (fold_id IS NOT NULL)
CREATE UNIQUE INDEX IF NOT EXISTS woe_anchor_cuts_v1_fold_ux
  ON ml.woe_anchor_cuts_v1 (model_key, fold_id)
  WHERE fold_id IS NOT NULL;

-- Final map uniqueness
CREATE UNIQUE INDEX IF NOT EXISTS woe_anchor_map_v1_final_ux
  ON ml.woe_anchor_map_v1 (model_key, band_name, band_value)
  WHERE fold_id IS NULL;

-- Fold map uniqueness
CREATE UNIQUE INDEX IF NOT EXISTS woe_anchor_map_v1_fold_ux
  ON ml.woe_anchor_map_v1 (model_key, fold_id, band_name, band_value)
  WHERE fold_id IS NOT NULL;

-- Lookup indexes for scoring joins
CREATE INDEX IF NOT EXISTS woe_anchor_cuts_v1_lookup_ix
  ON ml.woe_anchor_cuts_v1 (model_key, fold_id);

CREATE INDEX IF NOT EXISTS woe_anchor_map_v1_lookup_ix
  ON ml.woe_anchor_map_v1 (model_key, fold_id, band_name, band_value);

COMMIT;
