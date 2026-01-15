ml.iphone_image_color_features_v1 — Vision Color Feature Store (v1)
Purpose

ml.iphone_image_color_features_v1 is a listing-level (one row per (generation, listing_id)) feature store that aggregates image-derived body color predictions from the vision/LLM color pipeline into stable, joinable features for:

TOM / survival modeling

resale pricing models

QA and monitoring of color labeling quality and coverage

multimodal “unified” feature store joins (damage + accessories + color)

This view is explicitly designed to:

avoid timestamps and other pipeline-time leakage (no *_at columns)

expose coverage, vote strength, conflict/ambiguity, and source-quality flags (case/stock/HQ)

integrate the LLM-driven model correction (“model_fix”) as an effective model label without exposing evidence/timestamps

Grain and Keys

Grain: one row per iPhone listing with images, defined by "iPhone".iphone_image_assets ((generation,listing_id) universe).

Join keys:

generation (integer)

listing_id (bigint)

Notes:

listing_id is a join key only. Do not feed it into ML models as a feature.

Source Tables and Dependencies
1) "iPhone".iphone_image_assets (asset inventory)

Used to anchor the listing universe and compute n_assets:

One row per image per listing: (generation,listing_id,image_index)

In the view: n_assets = COUNT(*) per listing

2) "iPhone".iphone_listings (listing metadata)

Used only for:

db_model (current model string stored in the DB)

db_spam (spam tag, e.g. 'below13')
These are used to compute effective model labels and spam flags.

3) ml.iphone_image_features_v1 (per-image vision outputs)

Filtered to:

feature_version = 1

Used per image:

color_done (boolean processed marker)

body_color_name, body_color_key, body_color_confidence, body_color_from_case

is_stock_photo, photo_quality_level, background_clean_level (may be NULL for some rows/gens)

model_fix_new_model, model_fix_reason (used for effective model integration; no timestamps used)

Update / Refresh Semantics

This is a VIEW (not materialized):

No refresh jobs are needed.

When new (generation,listing_id) appear in "iPhone".iphone_image_assets, the view gains rows.

When the color labeler updates ml.iphone_image_features_v1 (color_done, body_color_*, model_fix_*), the view reflects changes immediately.

Row count behavior:

increases only when the asset listing universe increases (new listings scraped with images).

Processing States: How to Interpret Listings

The view makes processing state explicit via:

has_feature_rows: there exist feature rows in ml.iphone_image_features_v1

backlog_not_done: feature rows exist but n_color_done_imgs = 0 (color pipeline not yet run/completed)

color_processed: n_color_done_imgs > 0

color_known: n_color_votes > 0 (at least one accepted color key vote)

done_but_no_votes: color processed but no votes survived validation / detection

This is critical because you have multiple pipeline regimes:

assets exist but features may lag

features exist but color not done yet

done-but-no-votes occurs at a low but non-zero rate

Output Schema and Definitions
A) Keys and Asset Counts

generation (int): stored listing generation (partition key).

listing_id (bigint): listing id.

n_assets (bigint): number of images available for the listing.

B) Coverage / Completeness (no timestamps)

n_feat_rows (bigint): number of feature rows present in ml.iphone_image_features_v1 for the listing.

n_color_done_imgs (bigint): number of images where color_done = TRUE.

min_color_done_idx, max_color_done_idx (smallint): index span of processed images.

Coverage ratios:

feat_row_coverage_assets (numeric): n_feat_rows / n_assets

color_done_coverage_assets (numeric): n_color_done_imgs / n_assets

Completeness:

undone_assets (bigint): n_assets - n_color_done_imgs

is_incomplete (boolean): whether processed images are fewer than available assets

hit_cap_16 (boolean): indicates truncation at 16 images (prefix ends at idx 15 and assets > 16)

Pipeline state:

has_feature_rows (boolean): n_feat_rows > 0

backlog_not_done (boolean): has_feature_rows AND n_color_done_imgs = 0

color_processed (boolean): n_color_done_imgs > 0

C) Vote Strength and Consistency

A vote is an image row where:

color_done = TRUE

body_color_key IS NOT NULL

These are “accepted” votes (the upstream script enforces allowed colors and nulls invalid ones).

n_color_votes (bigint): count of accepted votes.

n_distinct_color_keys (int): number of distinct accepted keys.

color_known (boolean): n_color_votes > 0

done_but_no_votes (boolean): processed but no accepted votes

color_conflict (boolean): n_color_votes > 0 AND n_distinct_color_keys >= 2

Confidence summaries over accepted votes:

color_conf_max (real/double): max confidence

color_conf_avg (real/double): average confidence

color_conf_p10, color_conf_p50, color_conf_p90 (double precision): confidence percentiles

D) Vote Source Quality Flags (case, stock, HQ)

These are computed only across accepted votes:

n_votes_from_case (bigint): votes where body_color_from_case = TRUE

p_votes_from_case (numeric): n_votes_from_case / n_color_votes (NULL if no votes)

n_votes_stock (bigint): votes where is_stock_photo = TRUE

p_votes_stock (numeric): n_votes_stock / n_color_votes (NULL if no votes)

n_votes_hq (bigint): votes where photo_quality_level >= 3

p_votes_hq (numeric): n_votes_hq / n_color_votes (NULL if no votes)

These features provide reliability signals:

case-present listings tend to increase case-derived votes and conflict risk

stock involvement tends to increase conflict risk

HQ votes are a proxy for visibility/clarity

E) Selected Colors (primary + runner-up)

The view chooses a primary and secondary color at the per-color aggregate level:

Per color key within a listing:

votes = number of images voting for the key

p_from_case = fraction of votes for that key that are from-case

max_conf, avg_conf = confidence summaries

first_vote_idx = earliest image index voting for the key

Ranking rule (best to worst):

higher votes

lower p_from_case (prefer body color over case-inferred)

higher max_conf

higher avg_conf

earlier first_vote_idx

deterministic body_color_key tie-break

Primary:

color_key_primary (text)

color_name_primary (text)

color_votes_primary (bigint)

color_primary_vote_share (numeric): color_votes_primary / n_color_votes

color_conf_primary_max, color_conf_primary_avg

color_primary_p_from_case

color_primary_first_vote_idx

Secondary (if present):

color_key_secondary, color_name_secondary

color_votes_secondary

color_secondary_vote_share

color_conf_secondary_max, color_conf_secondary_avg

color_secondary_p_from_case

Margins:

color_vote_margin_votes (bigint): primary_votes - secondary_votes

color_vote_margin_share (numeric): (primary_votes - secondary_votes) / n_color_votes

Strict primary (higher precision, potentially lower recall):

color_key_primary_strict (text or NULL)

Strict logic:

NULL if no votes

primary if only one key

primary if no runner-up

primary if primary vote share ≥ 0.67

else NULL (ambiguous)

Use cases:

color_key_primary for maximum coverage

color_key_primary_strict for high precision or anchor-building

F) Model Correction Integration (no timestamps/evidence JSON)

The view integrates model correction signals recorded by the color pipeline into an effective model string, used downstream for variant-aware logic and QA.

Inputs:

model_fix_new_model from features table (if any)

fallback to "iPhone".iphone_listings.model

Outputs:

model_fixed_any (boolean): whether any model fix exists for the listing

model_fix_new_model (text): corrected model when present

model_effective (text): COALESCE(model_fix_new_model, db_model)

Derived model type:

model_effective_type (text): one of {base, pro, pro_max, mini, plus} or NULL

computed by substring matching rules on model_effective

Derived generation digit from the model string:

model_effective_generation (int): extracted digit from model_effective (NULL if not parseable)

Mismatch flag:

model_generation_mismatch (boolean): TRUE when model_effective_generation exists and differs from stored generation

Spam:

is_spam_below13 (boolean): TRUE if either:

model_fix_reason='spam_below13' was observed, OR

iphone_listings.spam='below13'

Notes:

The view does not change the primary key generation. It only flags mismatches.

If mismatches are significant, resolve upstream with a migration job (do not fix via views).

Known Data Patterns (from your research)

Color evidence usually appears early (p90 first evidence index ≈ 2).

Conflict rate is non-trivial (≈ 4–6% of voted listings).

Cases and stock photos increase conflict risk; case-inferred votes matter.

Model correction occurs in ~1–2% of listings; most fixes are within-generation variant corrections, but rare cross-generation mismatches exist and are flagged.

Recommended ML Usage (Leak-Safe)
Allowed (high-signal) modeling fields

color_key_primary, color_primary_vote_share, color_vote_margin_share

color_conflict, n_distinct_color_keys

p_votes_from_case, p_votes_stock, p_votes_hq

color_key_primary_strict (optional: as high-precision version)

model_effective_type (optional: can be useful if downstream models depend on variant behavior)

Keep for filtering / QA (not necessarily for XGBoost features)

coverage fields: n_color_done_imgs, color_done_coverage_assets, hit_cap_16, is_incomplete

state flags: backlog_not_done, done_but_no_votes

mismatch flags: model_generation_mismatch, is_spam_below13

Never feed into ML features

listing_id (join key only)

any timestamps (none exist in this view by design)

QA / Validation Queries
1) No timestamps in view
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema='ml'
  AND table_name='iphone_image_color_features_v1'
  AND data_type LIKE 'timestamp%';

2) Row count matches listing universe with assets
WITH asset_listings AS (
  SELECT generation, listing_id
  FROM "iPhone".iphone_image_assets
  GROUP BY 1,2
)
SELECT
  (SELECT COUNT(*) FROM asset_listings) AS listings_with_assets,
  (SELECT COUNT(*) FROM ml.iphone_image_color_features_v1) AS rows_in_view;

3) Key uniqueness
SELECT
  COUNT(*) AS rows,
  COUNT(DISTINCT (generation, listing_id)) AS distinct_keys
FROM ml.iphone_image_color_features_v1;

4) Invariants
SELECT COUNT(*) AS bad
FROM ml.iphone_image_color_features_v1
WHERE n_color_done_imgs > n_assets;

SELECT COUNT(*) AS bad
FROM ml.iphone_image_color_features_v1
WHERE is_incomplete IS DISTINCT FROM (n_color_done_imgs < n_assets);

5) Prefix-contiguity (should be ~0 exceptions)
SELECT COUNT(*) AS non_prefix
FROM ml.iphone_image_color_features_v1
WHERE color_processed
  AND (min_color_done_idx <> 0 OR n_color_done_imgs <> (max_color_done_idx + 1));

6) Model generation mismatch
SELECT generation, model_effective_generation, COUNT(*) AS listings
FROM ml.iphone_image_color_features_v1
WHERE model_generation_mismatch
GROUP BY 1,2
ORDER BY 1,2;

Versioning and Governance

This view is bound to ml.iphone_image_features_v1.feature_version = 1.

If you update the color prompt/model/enforcement logic:

bump feature_version in the per-image table

create ml.iphone_image_color_features_v2 rather than mutating v1 semantics

Limitations / Caveats

Listings with feature rows but no color processing are flagged (backlog_not_done); they are not treated as “no color.”

Model correction can imply cross-generation mismatch; the view flags this but does not rewrite generation.

Confidence is used for descriptive stats and tie-breaking; it is not assumed to be calibrated probability.

The strict color output (color_key_primary_strict) trades recall for precision.





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





ml.iphone_image_accessory_features_v1 — Vision Accessory Feature Store (v1)
Purpose

ml.iphone_image_accessory_features_v1 is a listing-level (one row per (generation, listing_id)) feature store that aggregates image-derived accessory signals produced by the vision/LLM accessory labeler into stable features for:

survival/TOM modeling

pricing models

QA/monitoring of the vision labeling pipeline

downstream unified multimodal feature joins

This view is explicitly designed to:

represent unknown vs observed false (critical for Gen13 behavior)

avoid timestamps and other leakage-prone pipeline time columns

remain live-updating as new images and new per-image labels arrive

Grain and Keys

Grain: one row per iPhone listing with images ((generation, listing_id)) as observed in "iPhone".iphone_image_assets.

Primary keys (logical):

generation (integer)

listing_id (bigint)

Notes:

listing_id is retained for joins only. Do not feed it as a feature to ML models.

Source Tables and Dependencies
1) "iPhone".iphone_image_assets (asset inventory)

Used to determine the listing universe and image counts:

(generation, listing_id, image_index) one row per image per listing

n_assets = COUNT(*) per listing

2) ml.iphone_image_features_v1 (per-image LLM vision outputs)

Used for all accessory signals and pipeline markers, filtered to:

feature_version = 1

Accessory labeler outputs used:

accessories_done

has_box, box_state_level

has_charger, has_charger_brick, has_cable

has_earbuds

has_case, case_count

has_receipt

has_other_accessory

This view does not use timestamps like accessories_done_at (even if present in base tables).

Update / Refresh Semantics

This is a VIEW (not a materialized view).

No refresh job is required for the view.

As new rows arrive in "iPhone".iphone_image_assets:

new (generation, listing_id) will appear in the view

n_assets increases for existing listings with newly scraped images

As the accessory labeler writes into ml.iphone_image_features_v1:

aggregates update automatically on next query

Row count behavior:

Row count increases only when new listings are added to iphone_image_assets.

Row count does not increase when more images are added to an existing listing.

Design Principles
1) “Known vs Unknown” is explicit

Accessory booleans in the raw table may be NULL even when accessories_done = TRUE (especially in older Gen13 data). Therefore, the view exports:

*_present where possible (presence)

*_known flags (whether the signal is observed at all)

evidence counts (n_*_obs_imgs, n_*_true_imgs) to quantify reliability

2) Canonicalization rules

Some raw fields overlap or conflict. The view defines canonical semantics:

Box presence is determined primarily by box_state_level (listing-level max), with fallback to has_box only when box_state is missing.

Charger presence is derived from the union of charger/cable/brick signals, with a charger_bundle_level to encode completeness.

3) No timestamps / no pipeline time columns

The view contains no *_at timestamps. This reduces leakage and avoids train/serve shortcut learning.

Output Schema (Field Dictionary)
A) Keys

generation (int): listing generation key (partition key across your ecosystem).

listing_id (bigint): marketplace listing id (join key).

B) Coverage / Completeness

These describe what the pipeline actually processed and how complete the per-image feature table is relative to assets.

n_assets (bigint): number of image assets for this listing (from iphone_image_assets).

n_feat_rows (bigint): number of rows in ml.iphone_image_features_v1 for this listing (feature_version=1).

n_acc_done_imgs (bigint): number of images for which accessories_done = TRUE.

min_acc_done_idx (smallint): smallest image_index with accessories_done = TRUE (normally 0).

max_acc_done_idx (smallint): largest image_index with accessories_done = TRUE.

Coverage ratios:

feat_row_coverage_assets (numeric): n_feat_rows / n_assets

acc_done_coverage_assets (numeric): n_acc_done_imgs / n_assets

Completion diagnostics:

undone_assets (bigint): n_assets - n_acc_done_imgs

is_incomplete (boolean): whether accessory processing covered fewer images than available

hit_cap_8 (boolean): indicates likely truncation at 8 images (prefix ending at idx 7 when assets>8)

hit_cap_16 (boolean): indicates truncation at 16 images (prefix ending at idx 15 when assets>16)

Pipeline state:

accessories_processed (boolean): n_acc_done_imgs > 0

C) Output Observability (critical for Gen13)

any_acc_output_observed (boolean): TRUE if any accessory output field is non-null in any processed image.

n_any_acc_output_observed_imgs (bigint): number of processed images where at least one accessory output is non-null.

acc_done_but_all_outputs_null (boolean): TRUE if images were processed (n_acc_done_imgs>0) but no accessory outputs were observed (a known Gen13 failure mode).

D) Box

Raw+canonical fields:

box_state_max (smallint): max box_state_level across processed images.

box_state_known (boolean): whether box_state_max IS NOT NULL.

box_present (boolean or NULL):

if box_state_max is known: box_state_max >= 1

else if has_box was observed: has_box_any

else NULL (unknown)

box_known (boolean): TRUE if either has_box was observed or box_state_max exists

Evidence counts:

n_box_obs_imgs (bigint): processed images where has_box IS NOT NULL

n_box_true_imgs (bigint): processed images where has_box IS TRUE

n_box_state_obs_imgs (bigint): processed images where box_state_level IS NOT NULL

Notes:

box_state_level behaves like a structured classifier:

0 ~ no box

1/2 ~ box present with condition tier

Using box_state_max avoids Gen13 NULL regime issues.

E) Charger Bundle (charger / cable / brick)

Canonical “bundle” fields:

charger_present (boolean or NULL):

NULL if none of charger/cable/brick were observed at all

TRUE if any of those are TRUE

FALSE if observed but none present

charger_known (boolean): TRUE if any of charger/cable/brick were observed

charger_bundle_level (int or NULL):

NULL if not observed

0 = no charger-related items

1 = partial (any one of charger/cable/brick)

2 = both brick and cable present

Component fields (with knownness and evidence):

cable_present (boolean): raw has_cable_any (may be NULL if never observed; see cable_known)

cable_known (boolean): whether has_cable was observed at least once

n_cable_obs_imgs (bigint)

n_cable_true_imgs (bigint)

brick_present (boolean): raw has_brick_any

brick_known (boolean)

n_brick_obs_imgs (bigint)

n_brick_true_imgs (bigint)

F) Earbuds

earbuds_present (boolean or NULL): TRUE/FALSE if observed, else NULL

earbuds_known (boolean): observedness flag

n_earbuds_obs_imgs (bigint)

n_earbuds_true_imgs (bigint)

G) Case and Case Count

case_present (boolean or NULL): TRUE/FALSE if observed, else NULL

case_known (boolean)

n_case_obs_imgs (bigint)

n_case_true_imgs (bigint)

Case quantity:

case_count_max (smallint): max observed case_count across processed images

case_count_p90 (double precision): p90 of observed case_count values

case_count_known (boolean): observedness flag for case_count

H) Receipt

receipt_present (boolean or NULL): TRUE/FALSE if observed, else NULL

receipt_known (boolean)

n_receipt_obs_imgs (bigint)

n_receipt_true_imgs (bigint)

I) Other Accessory

other_accessory_present (boolean or NULL): TRUE/FALSE if observed, else NULL

other_accessory_known (boolean)

n_other_obs_imgs (bigint)

n_other_true_imgs (bigint)

Recommended Model Usage (Leak-Safe)

When building your ML matrix X, treat:

listing_id as a join key only.

prefer content signals (box_present, charger_bundle_level, etc.)

optional: include evidence counts (n_*_true_imgs) as confidence proxies

avoid using coverage/process fields as predictive features unless you train and serve under identical pipeline conditions:

n_acc_done_imgs, acc_done_coverage_assets, hit_cap_8, hit_cap_16, acc_done_but_all_outputs_null, etc.

If you want strict gating for training:

filter to any_acc_output_observed = TRUE (or acc_done_but_all_outputs_null = FALSE) for Gen13 robustness.

QA / Monitoring Queries
1) View row count should equal asset listing count
WITH asset_listings AS (
  SELECT generation, listing_id
  FROM "iPhone".iphone_image_assets
  GROUP BY 1,2
)
SELECT
  (SELECT COUNT(*) FROM asset_listings) AS listings_with_assets,
  (SELECT COUNT(*) FROM ml.iphone_image_accessory_features_v1) AS rows_in_view;

2) Detect “processed but outputs missing” (Gen13 anomaly)
SELECT generation,
       COUNT(*) AS listings,
       COUNT(*) FILTER (WHERE acc_done_but_all_outputs_null) AS bad,
       ROUND(100.0 * COUNT(*) FILTER (WHERE acc_done_but_all_outputs_null) / NULLIF(COUNT(*),0), 2) AS pct_bad
FROM ml.iphone_image_accessory_features_v1
GROUP BY 1
ORDER BY 1;

3) Coverage diagnostics
SELECT
  generation,
  AVG(acc_done_coverage_assets) AS avg_cov,
  AVG(is_incomplete::int) AS p_incomplete,
  AVG(hit_cap_8::int) AS p_cap8,
  AVG(hit_cap_16::int) AS p_cap16
FROM ml.iphone_image_accessory_features_v1
GROUP BY 1
ORDER BY 1;

Versioning and Governance

This store is tied to ml.iphone_image_features_v1.feature_version = 1.

If you change prompts/model logic in a way that changes semantics, you should:

write new outputs under feature_version = 2

create ml.iphone_image_accessory_features_v2 filtering to v2

Do not silently change semantics of v1.

Known Limitations (Documented)

Accessory processing is prefix-contiguous by image index (usually 0..K); late-image evidence may be missed if the labeler caps processed images.

Gen13 contains a legacy “processed but all outputs null” regime; this view flags it explicitly and preserves unknown vs false semantics.

This store does not include timestamps by design; pipeline observability should use base table timestamps, not the feature store.