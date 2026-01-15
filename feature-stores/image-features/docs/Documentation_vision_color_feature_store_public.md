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