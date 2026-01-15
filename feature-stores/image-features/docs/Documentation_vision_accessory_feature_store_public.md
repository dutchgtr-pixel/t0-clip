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