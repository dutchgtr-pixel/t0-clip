-- 05_feature_block_normalization.sql
-- For historical runs where some features were labeled "trainer_derived" while others were "trainer_derived_store",
-- you can normalize the meta JSONB to a single bucket.

-- Example:
-- Update all rows in model_feature_importance_v1 for a model_key/scope/type where feature_block='trainer_derived'
-- to become feature_block='trainer_derived_store'.

UPDATE ml.model_feature_importance_v1
SET meta = jsonb_set(meta, '{feature_block}', to_jsonb('trainer_derived_store'::text), true)
WHERE model_key = :model_key
  AND scope = :scope
  AND importance_type = :importance_type
  AND meta->>'feature_block' = 'trainer_derived';

-- Validate:
SELECT meta->>'feature_block' AS feature_block, COUNT(*) AS n
FROM ml.model_feature_importance_v1
WHERE model_key = :model_key
  AND scope = :scope
  AND importance_type = :importance_type
GROUP BY 1
ORDER BY 1;
