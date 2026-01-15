ml.iphone_image_features_unified_v1 — Unified Vision Feature Store (v1)
Purpose

ml.iphone_image_features_unified_v1 is the canonical listing-level “vision” feature store that unifies:

raw image inventory features (image_count, caption metrics),

generic per-image quality/meta features (photo quality, background cleanliness, stock-photo share, battery screenshot extraction), and

the three specialized vision feature stores:

ml.iphone_image_accessory_features_v1

ml.iphone_image_color_features_v1

ml.iphone_image_damage_features_v1 (+ one computed structural mean)

This view exists to make it easy and safe to ship a single join to your XGBoost / survival pipelines, while preserving:

explicit missingness (“unknown” vs “observed false” semantics)

pipeline coverage/health signals for QA and gating

leak-safe output (no timestamps in the unified store)

Grain and Keys

Grain: one row per listing with at least one image in "iPhone".iphone_image_assets.

Join keys:

generation (integer)

listing_id (bigint)

Notes:

listing_id is a join key only; do not feed it into ML features.

generation remains the stored generation partition key.
A separate generation_effective_for_modeling is provided for modeling logic when the corrected model implies a different generation digit.

Dependencies and Source Tables
1) "iPhone".iphone_image_assets (image inventory + captions)

Used to anchor the listing universe and compute:

image_count: total images per listing

caption_count_raw: count of images with non-empty caption_text

Why this matters: image count and caption density are strong seller-effort signals and are already used in your ML pipelines.

2) ml.iphone_image_features_v1 (per-image vision outputs, feature_version=1)

Used twice:

A) Generic per-listing aggregates (generic CTE)

photo_quality_avg, photo_quality_max

bg_clean_avg

has_stock_photo, stock_photo_share

battery screenshot extraction:

has_battery_screenshot

n_battery_screenshot_imgs

battery_pct_img (max extracted value among screenshot images)

B) Structural damage HQ mean (dmg_struct_mean CTE)

Your damage feature store does not include dmg_mean_hq_struct. Therefore, unified computes it directly from per-image rows using the same HQ+struct filters:

visible_damage_level IS NOT NULL

is_stock_photo IS FALSE

photo_quality_level >= 3

damage_on_protector_only IS NOT TRUE

Result is exported as ds.dmg_mean_hq_struct.

3) Feature-store views joined in

ml.iphone_image_accessory_features_v1

ml.iphone_image_color_features_v1

ml.iphone_image_damage_features_v1

Unified does not recompute those features; it just joins them on (generation, listing_id).

Refresh / Update Semantics

This is a VIEW, not a materialized view.

No refresh is needed.

Every query re-reads current base tables/views.

New listings/images ingested into iphone_image_assets appear automatically.

New labeling writes to ml.iphone_image_features_v1 (accessory/color/damage scripts) flow through automatically.

Row-count behavior:

increases when the number of distinct (generation,listing_id) in iphone_image_assets increases.

Output Schema (Conceptual Sections)

The schema is logically grouped below. For exact column names/types, use information_schema.columns.

A) Keys

generation

listing_id

B) Image Inventory and Caption Metrics

These come from iphone_image_assets.

image_count (bigint): total images for the listing

caption_count_raw (bigint): count of images with caption_text IS NOT NULL AND caption_text <> ''

caption_count (bigint or NULL): legacy-compatible semantics

NULL if caption_count_raw = 0

else equals caption_count_raw

has_any_caption (boolean): caption_count_raw > 0

caption_share (numeric): caption_count_raw / image_count

Why both caption_count and caption_count_raw:

caption_count preserves your existing “NULL means none” model semantics.

caption_count_raw is the correct numeric count for new modeling.

C) Generic Image Meta (from generic CTE)

These are listing-level summaries across all feature rows (not limited to processed subsets).

n_feat_rows_all (bigint): count of rows in ml.iphone_image_features_v1 for this listing

feat_row_coverage_assets (numeric): n_feat_rows_all / image_count

Quality:

photo_quality_avg (numeric)

photo_quality_max (smallint)

bg_clean_avg (numeric)

Stock photos:

has_stock_photo (boolean)

stock_photo_share (double precision in [0,1])

Battery screenshot extraction:

has_battery_screenshot (boolean)

n_battery_screenshot_imgs (bigint)

battery_pct_img (smallint, expected 0–100; NULL when no extraction)

D) Accessories Sub-store Fields (acc_* prefixed in unified)

Unified includes the most important outputs from ml.iphone_image_accessory_features_v1.

Coverage / pipeline:

acc_n_feat_rows

n_acc_done_imgs

acc_done_coverage_assets

acc_is_incomplete

acc_hit_cap_16

any_acc_output_observed

acc_done_but_all_outputs_null (key Gen13 anomaly indicator)

Canonical accessory signals (unknown vs observed false semantics preserved):

Box:

box_present, box_known, box_state_max, box_state_known

Charger bundle:

charger_present, charger_known, charger_bundle_level

components: cable_present, cable_known, brick_present, brick_known

Earbuds:

earbuds_present, earbuds_known

Case:

case_present, case_known, case_count_max, case_count_p90, case_count_known

Receipt:

receipt_present, receipt_known

Other:

other_accessory_present, other_accessory_known

E) Color Sub-store Fields (color_* prefixed in unified)

Unified includes the listing-level color consensus outputs from ml.iphone_image_color_features_v1.

Coverage / pipeline:

color_n_feat_rows

n_color_done_imgs

color_feat_row_coverage_assets

color_done_coverage_assets

color_is_incomplete

color_hit_cap_16

color_has_feature_rows

color_backlog_not_done

color_processed

Consensus / quality:

color_known

color_done_but_no_votes

color_conflict

n_color_votes

n_distinct_color_keys

Primary + secondary choice:

color_key_primary, color_name_primary, color_primary_vote_share

color_key_secondary, color_name_secondary, color_secondary_vote_share

margins: color_vote_margin_votes, color_vote_margin_share

confidence stats: color_conf_max, color_conf_avg, color_conf_p10/p50/p90

vote source shares: p_votes_from_case, p_votes_stock, p_votes_hq

strict primary: color_key_primary_strict

Convenience alias:

body_color_key_main = color_key_primary

Model correction signals from the color store:

model_effective, model_effective_type, model_effective_generation

generation_effective_for_modeling = COALESCE(model_effective_generation, generation)

model_fixed_any

model_fix_new_model

model_generation_mismatch (rare; do not rewrite generation keys)

is_spam_below13

F) Damage Sub-store Fields (dmg_* prefixed in unified)

Unified includes summary and HQ-structural damage signals from ml.iphone_image_damage_features_v1, plus one computed mean.

Coverage / pipeline:

dmg_n_feat_rows

n_dmg_labeled

dmg_coverage_assets

dmg_is_incomplete

dmg_hit_cap_16

Overall damage summary (all labeled images):

dmg_max (maps from dmg_max_all)

dmg_mean (maps from dmg_mean_all)

dmg_p90 (maps from dmg_p90_all)

Binary flags (NULL if no labeled images):

dmg_any_3plus, dmg_any_5plus, dmg_any_8plus

Protector handling:

has_screen_protector_any

protector_only_share_labeled

HQ structural subset:

n_hq_struct

dmg_max_hq_struct

dmg_p90_hq_struct

dmg_mean_hq_struct (computed in unified view via dmg_struct_mean)

share_3plus_hq_struct

share_5plus_hq_struct

share_8plus_hq_struct

dmg_band_hq_struct

Severe issue flags:

has_display_fault_5plus

has_back_glass_struct_5plus

has_front_glass_struct_5plus

has_camera_lens_issue_5plus

Leak-Safety Contract

The unified view contains no timestamp columns by design:

no *_done_at

no model_fix_at

no assets created_at/updated_at

Leakage considerations:

listing_id is a time-correlated identifier; retain only for joins.

coverage fields (*_coverage_assets, *_is_incomplete, cap flags) can correlate with pipeline rollout. Use them for QA/gating; include in ML only if you ensure identical train/serve pipeline behavior.

Recommended ML Feature Usage
High-signal modeling features (safe content signals)

Inventory:

image_count, caption_count_raw, caption_share, has_any_caption

Quality/meta:

photo_quality_avg, photo_quality_max, bg_clean_avg

has_stock_photo, stock_photo_share

Battery:

battery_pct_img (and optionally has_battery_screenshot)

Accessories:

box_present, charger_bundle_level, case_present, receipt_present, earbuds_present

Color:

color_key_primary, color_primary_vote_share, color_vote_margin_share

color_conflict, p_votes_from_case, p_votes_stock

Damage:

dmg_max, dmg_mean, dmg_any_5plus, dmg_any_8plus

HQ structural: dmg_max_hq_struct, dmg_mean_hq_struct, dmg_band_hq_struct

flags: has_display_fault_5plus, etc.

QA / gating (use for filtering, not necessarily in X)

acc_done_but_all_outputs_null

color_backlog_not_done, color_done_but_no_votes

model_generation_mismatch, is_spam_below13

per-pipeline coverage/cap flags

Never feed into models

listing_id

any timestamps (none exist here)

Validation and QA Checklist (Production Contract)
1) Column presence (image+caption+battery)

image_count, caption_count, caption_count_raw, has_any_caption, caption_share

battery_pct_img, has_battery_screenshot, n_battery_screenshot_imgs

2) Universe match

row count equals number of distinct listings in iphone_image_assets

3) No duplicates

COUNT(*) == COUNT(DISTINCT (generation,listing_id))

4) Inventory invariants

caption_count_raw <= image_count

caption_share in [0,1]

caption_count is NULL iff caption_count_raw=0

5) Cross-store reconciliation (sampling)

accessory/color/damage fields match their source stores for random samples

6) Range checks

battery_pct_img in [0,100]

stock share in [0,1]

photo quality in expected range

Known Limitations / Known Behaviors

Gen-dependent rollout: color and damage labeling may lag for newer generations until backlog completes.

Cross-generation model fixes: rare mismatches may occur; unified preserves stored generation key and exposes mismatch flags.

Coverage flags are operationally sensitive: may correlate with labeling rollout; treat carefully in modeling.

If you want, I can also generate a short “integration snippet” for your XGBoost training SQL showing the exact join and the recommended column selection list (including which unified columns to exclude explicitly).