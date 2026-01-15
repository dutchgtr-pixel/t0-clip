-- 02_add_registry_fields.sql
-- Adds required fields for SQL scoring/banding.

ALTER TABLE ml.woe_anchor_model_registry_v1
  ADD COLUMN IF NOT EXISTS base_rate float8,
  ADD COLUMN IF NOT EXISTS base_logit float8,
  ADD COLUMN IF NOT EXISTS dsold_t1 float8,
  ADD COLUMN IF NOT EXISTS dsold_t2 float8,
  ADD COLUMN IF NOT EXISTS dsold_t3 float8,
  ADD COLUMN IF NOT EXISTS dsold_t4 float8;
