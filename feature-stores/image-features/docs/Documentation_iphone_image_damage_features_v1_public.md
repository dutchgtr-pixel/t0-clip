Documentation: ml.iphone_image_damage_features_v1
Purpose

ml.iphone_image_damage_features_v1 provides listing-level image-derived damage and presentation features, aggregated from per-image LLM labels.

This view is designed to:

Use only content-derived fields for modeling (no pipeline timestamps).

Be robust to partial labeling (labels arrive over time).

Support joins into your TOM / survival model feature pipeline via (generation, listing_id).

Inputs and dependencies
1) Image inventory (Playwright output)

"iPhone".iphone_image_assets

One row per image per listing: (generation, listing_id, image_index)

Source of truth for how many images exist per listing.

Used in this view:

COUNT(*) AS n_assets

2) Per-image damage labels (LLM output)

ml.iphone_image_features_v1

One row per image per feature_version: (generation, listing_id, image_index, feature_version)

Contains the damage labeler outputs: visible_damage_level, photo_quality_level, background_clean_level, is_stock_photo, has_screen_protector, damage_on_protector_only, damage_summary_text, etc.

Used in this view:

All aggregations come from feature_version = 1.

Grain and keys

One row per listing that has at least one image in "iPhone".iphone_image_assets.

Join keys:

generation (integer)

listing_id (bigint)

Important modeling rule:

listing_id is join key only. Do not feed it into XGBoost as a feature.

Update semantics

This is a VIEW, not a materialized view.

No refresh job is needed for the view itself.

Whenever new labels are written into ml.iphone_image_features_v1, results are reflected immediately in this view.

Whenever Playwright adds/removes image asset rows in "iPhone".iphone_image_assets, n_assets changes immediately.

Output schema (grouped by function)
A) Keys

generation

listing_id

B) Image inventory and labeling coverage

n_assets: total images available for the listing (from assets)

n_feat_rows: feature rows found for listing in ml.iphone_image_features_v1 (any)

n_dmg_labeled: number of images with visible_damage_level IS NOT NULL

min_labeled_idx, max_labeled_idx: labeled index range (helps detect prefix-labeling behavior)

dmg_coverage_assets = n_dmg_labeled / n_assets

unlabeled_tail_assets = n_assets - n_dmg_labeled

is_incomplete = (n_dmg_labeled < n_assets)

hit_cap_8 = (max_labeled_idx = 7)

hit_cap_16 = (max_labeled_idx = 15)

Operational note:

These are not timestamps, but they are still “pipeline progress / evidence strength” signals. If you don’t want the model to learn pipeline rollout patterns, exclude them from X or gate scoring to “fully labeled enough” rows.

C) Photo quality / trust signals

Computed over labeled images:

photo_quality_avg, photo_quality_max

bg_clean_avg

has_stock_labeled, stock_share_labeled

has_screen_protector_any

protector_only_share_labeled

D) Damage severity aggregates

All labeled images

dmg_min_all, dmg_max_all, dmg_mean_all, dmg_sd_all, dmg_p90_all

Non-stock images

dmg_max_nostock

High-quality non-stock evidence (photo_quality_level >= 3 AND NOT stock)

n_hq_nostock

dmg_max_hq, dmg_p90_hq, dmg_mean_hq

High-quality “structural” evidence (photo_quality_level >= 3 AND NOT stock AND NOT protector-only)

n_hq_struct

dmg_max_hq_struct, dmg_p90_hq_struct

share_3plus_hq_struct, share_5plus_hq_struct, share_8plus_hq_struct

dmg_band_hq_struct:

0 = max ≤2

1 = max 3–4

2 = max 5–7

3 = max 8–10

E) Damage type flags (keyword-based)

Booleans indicating presence of certain damage types with severity ≥5:

has_display_fault_5plus

has_back_glass_struct_5plus

has_front_glass_struct_5plus

has_camera_lens_issue_5plus

Null semantics

For listings where labeling has not run yet:

n_assets will be populated

Most per-image-derived aggregates will be NULL

Coverage fields may be NULL or 0 depending on your SQL expressions (in your current view, many will be NULL because per_listing is missing; that’s expected)

Modeling implication:

You must decide whether to:

gate scoring to rows with evidence (n_hq_struct > 0 or dmg_max_hq_struct IS NOT NULL), or

allow missing and impute in the model consistently in train and serve.

Recommended usage patterns
A) Join into your TOM feature pipeline
SELECT
  f.*,
  d.dmg_max_hq_struct,
  d.dmg_band_hq_struct,
  d.share_5plus_hq_struct,
  d.has_display_fault_5plus,
  d.has_back_glass_struct_5plus
FROM ml.tom_features_v1_enriched_ai_clean_mv f
LEFT JOIN ml.iphone_image_damage_features_v1 d
  ON d.generation = f.generation
 AND d.listing_id    = f.listing_id;

B) Safe feature allowlist for XGBoost

Keep generation,listing_id for joins, but build X from an allowlist. For example:

Use as features:

quality/trust: photo_quality_avg, bg_clean_avg, stock_share_labeled, protector_only_share_labeled

severity: dmg_max_hq_struct, dmg_p90_hq_struct, share_5plus_hq_struct, dmg_band_hq_struct

type flags: has_*_5plus

Do not include:

listing_id

any future timestamps (none exist now)

(optional) coverage/progress columns if you want maximal leak safety during rollout.

Operational monitoring queries
Coverage by generation
WITH assets AS (
  SELECT generation, listing_id
  FROM "iPhone".iphone_image_assets
  GROUP BY 1,2
),
labeled AS (
  SELECT generation, listing_id
  FROM ml.iphone_image_features_v1
  WHERE feature_version=1 AND visible_damage_level IS NOT NULL
  GROUP BY 1,2
)
SELECT
  a.generation,
  COUNT(*) AS listings_with_assets,
  COUNT(*) FILTER (WHERE l.listing_id IS NOT NULL) AS listings_labeled,
  ROUND(100.0 * COUNT(*) FILTER (WHERE l.listing_id IS NOT NULL) / NULLIF(COUNT(*),0), 2) AS pct_labeled
FROM assets a
LEFT JOIN labeled l USING (generation, listing_id)
GROUP BY 1
ORDER BY 1;

Spot-check latest labeled listings (from base table)
SELECT
  generation, listing_id,
  MAX(damage_done_at) AS damage_done_at,
  COUNT(*) FILTER (WHERE visible_damage_level IS NOT NULL) AS labeled_imgs
FROM ml.iphone_image_features_v1
WHERE feature_version=1 AND damage_done_at IS NOT NULL
GROUP BY 1,2
ORDER BY damage_done_at DESC
LIMIT 20;

Performance and maintenance
Because it’s a view

No refresh needed.

If you start joining this view in large training queries frequently and it becomes slow, the next step is to create a materialized view and refresh it on a cadence (hourly/daily).

Indexes that typically help

If you notice slow group-bys or joins, add:

CREATE INDEX IF NOT EXISTS iphone_assets_gen_listing_idx
  ON "iPhone".iphone_image_assets (generation, listing_id);

CREATE INDEX IF NOT EXISTS imgfeat_v1_gen_listing_idx
  ON ml.iphone_image_features_v1 (feature_version, generation, listing_id);

CREATE INDEX IF NOT EXISTS imgfeat_v1_damage_gen_listing_idx
  ON ml.iphone_image_features_v1 (feature_version, generation, listing_id)
  WHERE visible_damage_level IS NOT NULL;

Versioning and change control

The view is tied to feature_version = 1 in ml.iphone_image_features_v1.

If you change the prompt/model in a way that changes semantics, bump feature_version and create a new view:

ml.iphone_image_damage_features_v2 filtering to feature_version=2.

Next feature stores

Now you can build, with the same pattern:

ml.iphone_image_accessory_features_v1 (battery screenshot + accessories + box state)

ml.iphone_image_color_features_v1 (body_color_key selection + confidence + from_case)
Then ml.iphone_image_features_unified_v1 as a join of the three by (generation, listing_id).

If you want, paste your accessory and color column inventories (or the scripts you’re using), and I’ll draft the two corresponding views in the same “leak-safe” style.