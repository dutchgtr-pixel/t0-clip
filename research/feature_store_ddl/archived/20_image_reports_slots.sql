-- Selected historical implementation; see provenance.json and README.md.
-- Reference only: dependencies, current-state inputs and destructive rebuilds require review.
BEGIN;

CREATE SCHEMA IF NOT EXISTS image_map;

CREATE TABLE IF NOT EXISTS image_map.device_image_labeler_report_en_v1 (
    generation integer NOT NULL,
    listing_id bigint NOT NULL,
    image_index smallint NOT NULL,
    feature_version smallint NOT NULL,
    mapping_version smallint NOT NULL,
    report_version smallint NOT NULL,
    primary_view_role text,
    secondary_view_role text,
    role_confidence real,
    phone_presence text,
    body_visibility text,
    damage_surface_primary text,
    damage_surface_confidence real,
    visible_damage_level smallint,
    image_damage_level smallint,
    photo_quality_level smallint,
    background_clean_level smallint,
    is_stock_photo boolean,
    has_screen_protector boolean,
    damage_on_protector_only boolean,
    battery_screenshot boolean,
    battery_health_pct_img smallint,
    battery_cycle_count_img integer,
    has_box boolean,
    has_charger_brick boolean,
    has_cable boolean,
    has_charger boolean,
    has_earbuds boolean,
    has_case boolean,
    has_screen_guard boolean,
    has_other_accessory boolean,
    has_receipt boolean,
    case_count smallint,
    body_color_name text,
    body_color_key text,
    body_color_confidence real,
    body_color_from_case boolean,
    box_state_level smallint,
    seal_status text,
    seal_confidence real,
    fresh_unit_visual_type text,
    fresh_unit_confidence real,
    box_face_view text,
    caption_present boolean,
    text_hint_used boolean,
    caption_hint_used boolean,
    needs_llm_gap_fill boolean,
    canonical_image_report_en text NOT NULL,
    report_metadata_json jsonb NOT NULL,
    built_at timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT device_image_labeler_report_en_v1_pk
        PRIMARY KEY (generation, listing_id, image_index, feature_version, mapping_version, report_version)
);

CREATE TABLE IF NOT EXISTS image_map.device_listing_image_k8_slots_v1 (
    generation integer NOT NULL,
    listing_id bigint NOT NULL,
    slot_version smallint NOT NULL,
    mapping_version smallint NOT NULL,
    report_version smallint NOT NULL,
    slot_index smallint NOT NULL,
    slot_role text NOT NULL,
    slot_selection_reason text NOT NULL,
    is_padding boolean NOT NULL,
    image_index smallint,
    feature_version smallint,
    primary_view_role text,
    secondary_view_role text,
    role_confidence real,
    phone_presence text,
    body_visibility text,
    damage_surface_primary text,
    damage_surface_confidence real,
    visible_damage_level smallint,
    photo_quality_level smallint,
    background_clean_level smallint,
    is_stock_photo boolean,
    battery_screenshot boolean,
    battery_health_pct_img smallint,
    has_box boolean,
    has_charger boolean,
    has_case boolean,
    has_receipt boolean,
    body_color_key text,
    body_color_confidence real,
    canonical_image_report_en text NOT NULL,
    slot_score numeric,
    slot_metadata_json jsonb NOT NULL,
    built_at timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT device_listing_image_k8_slots_v1_pk
        PRIMARY KEY (generation, listing_id, slot_version, slot_index),
    CONSTRAINT device_listing_image_k8_slots_v1_slot_idx_chk
        CHECK (slot_index BETWEEN 0 AND 7)
);

CREATE TABLE IF NOT EXISTS image_map.device_listing_image_k8_manifest_v1 (
    generation integer NOT NULL,
    listing_id bigint NOT NULL,
    slot_version smallint NOT NULL,
    mapping_version smallint NOT NULL,
    report_version smallint NOT NULL,
    real_slot_count smallint NOT NULL,
    padding_slot_count smallint NOT NULL,
    selected_image_indices smallint[] NOT NULL,
    selected_feature_versions smallint[] NOT NULL,
    has_front boolean NOT NULL,
    has_back boolean NOT NULL,
    has_side boolean NOT NULL,
    has_battery boolean NOT NULL,
    has_accessory boolean NOT NULL,
    has_damage boolean NOT NULL,
    has_nonstock boolean NOT NULL,
    stock_slot_count smallint NOT NULL,
    damage_slot_count smallint NOT NULL,
    max_visible_damage_level smallint,
    avg_photo_quality numeric,
    ready_for_k8_image_tower boolean NOT NULL,
    needs_additional_labeler boolean NOT NULL,
    canonical_listing_report_en text NOT NULL,
    manifest_json jsonb NOT NULL,
    built_at timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT device_listing_image_k8_manifest_v1_pk
        PRIMARY KEY (generation, listing_id, slot_version)
);

TRUNCATE image_map.device_listing_image_k8_manifest_v1;
TRUNCATE image_map.device_listing_image_k8_slots_v1;
TRUNCATE image_map.device_image_labeler_report_en_v1;

WITH src AS (
    SELECT
        m.generation,
        m.listing_id,
        m.image_index,
        m.feature_version,
        m.mapping_version,
        1::smallint AS report_version,
        m.primary_view_role,
        m.secondary_view_role,
        m.role_confidence,
        m.phone_presence,
        m.body_visibility,
        m.damage_surface_primary,
        m.damage_surface_confidence,
        f.visible_damage_level,
        f.image_damage_level,
        f.photo_quality_level,
        f.background_clean_level,
        f.is_stock_photo,
        f.has_screen_protector,
        f.damage_on_protector_only,
        f.battery_screenshot,
        f.battery_health_pct_img,
        f.battery_cycle_count_img,
        f.has_box,
        f.has_charger_brick,
        f.has_cable,
        f.has_charger,
        f.has_earbuds,
        f.has_case,
        f.has_screen_guard,
        f.has_other_accessory,
        f.has_receipt,
        f.case_count,
        f.body_color_name,
        f.body_color_key,
        f.body_color_confidence,
        f.body_color_from_case,
        f.box_state_level,
        f.seal_status,
        f.seal_confidence,
        f.fresh_unit_visual_type,
        f.fresh_unit_confidence,
        f.box_face_view,
        f.damage_summary_text,
        a.caption_text,
        m.text_hint_used,
        m.caption_hint_used,
        m.needs_llm_gap_fill,
        m.evidence_json AS role_evidence_json
    FROM image_map.device_image_role_mapping_v2_det m
    JOIN ml.device_image_features_v1 f
      ON f.generation = m.generation
     AND f.listing_id = m.listing_id
     AND f.image_index = m.image_index
     AND f.feature_version = m.feature_version
    LEFT JOIN "device".device_image_assets a
      ON a.generation = m.generation
     AND a.listing_id = m.listing_id
     AND a.image_index = m.image_index
    WHERE m.mapping_version = 2
),
normalized AS (
    SELECT
        *,
        CASE photo_quality_level
            WHEN 4 THEN 'excellent'
            WHEN 3 THEN 'good'
            WHEN 2 THEN 'usable'
            WHEN 1 THEN 'poor'
            WHEN 0 THEN 'very poor'
            ELSE 'unknown'
        END AS photo_quality_label_en,
        CASE background_clean_level
            WHEN 2 THEN 'clean'
            WHEN 1 THEN 'moderate'
            WHEN 0 THEN 'cluttered'
            ELSE 'unknown'
        END AS background_label_en,
        CASE box_state_level
            WHEN 2 THEN 'sealed or unopened'
            WHEN 1 THEN 'opened box'
            WHEN 0 THEN 'no visible box'
            ELSE 'unknown'
        END AS box_state_label_en,
        NULLIF(btrim(damage_summary_text), '') AS damage_summary_clean
    FROM src
)
INSERT INTO image_map.device_image_labeler_report_en_v1 (
    generation,
    listing_id,
    image_index,
    feature_version,
    mapping_version,
    report_version,
    primary_view_role,
    secondary_view_role,
    role_confidence,
    phone_presence,
    body_visibility,
    damage_surface_primary,
    damage_surface_confidence,
    visible_damage_level,
    image_damage_level,
    photo_quality_level,
    background_clean_level,
    is_stock_photo,
    has_screen_protector,
    damage_on_protector_only,
    battery_screenshot,
    battery_health_pct_img,
    battery_cycle_count_img,
    has_box,
    has_charger_brick,
    has_cable,
    has_charger,
    has_earbuds,
    has_case,
    has_screen_guard,
    has_other_accessory,
    has_receipt,
    case_count,
    body_color_name,
    body_color_key,
    body_color_confidence,
    body_color_from_case,
    box_state_level,
    seal_status,
    seal_confidence,
    fresh_unit_visual_type,
    fresh_unit_confidence,
    box_face_view,
    caption_present,
    text_hint_used,
    caption_hint_used,
    needs_llm_gap_fill,
    canonical_image_report_en,
    report_metadata_json,
    built_at
)
SELECT
    generation,
    listing_id,
    image_index,
    feature_version,
    mapping_version,
    report_version,
    primary_view_role,
    secondary_view_role,
    role_confidence,
    phone_presence,
    body_visibility,
    damage_surface_primary,
    damage_surface_confidence,
    visible_damage_level,
    image_damage_level,
    photo_quality_level,
    background_clean_level,
    is_stock_photo,
    has_screen_protector,
    damage_on_protector_only,
    battery_screenshot,
    battery_health_pct_img,
    battery_cycle_count_img,
    has_box,
    has_charger_brick,
    has_cable,
    has_charger,
    has_earbuds,
    has_case,
    has_screen_guard,
    has_other_accessory,
    has_receipt,
    case_count,
    body_color_name,
    body_color_key,
    body_color_confidence,
    body_color_from_case,
    box_state_level,
    seal_status,
    seal_confidence,
    fresh_unit_visual_type,
    fresh_unit_confidence,
    box_face_view,
    (caption_text IS NOT NULL AND btrim(caption_text) <> '') AS caption_present,
    text_hint_used,
    caption_hint_used,
    needs_llm_gap_fill,
    concat_ws(
        ' ',
        format('Image %s.', image_index),
        format('Primary view role: %s.', COALESCE(primary_view_role, 'unknown')),
        CASE WHEN secondary_view_role IS NOT NULL THEN format('Secondary view role: %s.', secondary_view_role) END,
        format('Phone presence: %s.', COALESCE(phone_presence, 'unknown')),
        format('Body visibility: %s.', COALESCE(body_visibility, 'unknown')),
        format('Damage surface: %s.', COALESCE(damage_surface_primary, 'unknown')),
        format('Visible damage level: %s out of 10.', COALESCE(visible_damage_level::text, 'unknown')),
        CASE WHEN damage_summary_clean IS NOT NULL THEN format('Visual damage summary: %s.', damage_summary_clean) END,
        format('Photo quality: %s.', photo_quality_label_en),
        format('Background: %s.', background_label_en),
        format('Stock photo: %s.', CASE WHEN COALESCE(is_stock_photo, false) THEN 'yes' ELSE 'no' END),
        format('Screen protector visible: %s.', CASE WHEN COALESCE(has_screen_protector, false) THEN 'yes' ELSE 'no' END),
        format('Damage only on protector: %s.', CASE WHEN COALESCE(damage_on_protector_only, false) THEN 'yes' ELSE 'no' END),
        format('Battery screenshot: %s.', CASE WHEN COALESCE(battery_screenshot, false) THEN 'yes' ELSE 'no' END),
        CASE WHEN battery_health_pct_img IS NOT NULL THEN format('Battery health visible in image: %s percent.', battery_health_pct_img) END,
        CASE WHEN battery_cycle_count_img IS NOT NULL THEN format('Battery cycle count visible in image: %s.', battery_cycle_count_img) END,
        format(
            'Accessories visible: box=%s, charger=%s, charger brick=%s, cable=%s, earbuds=%s, case=%s, screen guard=%s, other accessory=%s, receipt=%s.',
            CASE WHEN COALESCE(has_box, false) THEN 'yes' ELSE 'no' END,
            CASE WHEN COALESCE(has_charger, false) THEN 'yes' ELSE 'no' END,
            CASE WHEN COALESCE(has_charger_brick, false) THEN 'yes' ELSE 'no' END,
            CASE WHEN COALESCE(has_cable, false) THEN 'yes' ELSE 'no' END,
            CASE WHEN COALESCE(has_earbuds, false) THEN 'yes' ELSE 'no' END,
            CASE WHEN COALESCE(has_case, false) THEN 'yes' ELSE 'no' END,
            CASE WHEN COALESCE(has_screen_guard, false) THEN 'yes' ELSE 'no' END,
            CASE WHEN COALESCE(has_other_accessory, false) THEN 'yes' ELSE 'no' END,
            CASE WHEN COALESCE(has_receipt, false) THEN 'yes' ELSE 'no' END
        ),
        CASE WHEN case_count IS NOT NULL THEN format('Visible case count: %s.', case_count) END,
        CASE WHEN body_color_key IS NOT NULL THEN format('Body color: %s.', body_color_key) ELSE 'Body color: unknown.' END,
        CASE WHEN body_color_confidence IS NOT NULL THEN format('Body color confidence: %s.', round(body_color_confidence::numeric, 3)) END,
        CASE WHEN COALESCE(body_color_from_case, false) THEN 'Color may come from a case, not the phone body.' END,
        format('Box state: %s.', box_state_label_en),
        CASE WHEN seal_status IS NOT NULL THEN format('Seal status: %s.', seal_status) END,
        CASE WHEN fresh_unit_visual_type IS NOT NULL THEN format('Fresh unit visual type: %s.', fresh_unit_visual_type) END,
        CASE WHEN box_face_view IS NOT NULL THEN format('Box face view: %s.', box_face_view) END,
        format('Needs image-role gap fill: %s.', CASE WHEN COALESCE(needs_llm_gap_fill, false) THEN 'yes' ELSE 'no' END)
    ) AS canonical_image_report_en,
    jsonb_build_object(
        'contract', 'device_image_labeler_report_en_v1',
        'mapping_table', 'image_map.device_image_role_mapping_v2_det',
        'mapping_version', mapping_version,
        'report_version', report_version,
        'caption_present', caption_text IS NOT NULL AND btrim(caption_text) <> '',
        'text_hint_used', text_hint_used,
        'caption_hint_used', caption_hint_used,
        'role_evidence', role_evidence_json
    ) AS report_metadata_json,
    now() AS built_at
FROM normalized;

CREATE INDEX IF NOT EXISTS device_image_labeler_report_en_v1_listing_idx
    ON image_map.device_image_labeler_report_en_v1 (generation, listing_id);

CREATE INDEX IF NOT EXISTS device_image_labeler_report_en_v1_role_idx
    ON image_map.device_image_labeler_report_en_v1 (primary_view_role);

CREATE INDEX IF NOT EXISTS device_image_labeler_report_en_v1_damage_idx
    ON image_map.device_image_labeler_report_en_v1 (damage_surface_primary, visible_damage_level);

WITH base AS (
    SELECT
        r.*,
        CASE WHEN COALESCE(r.is_stock_photo, false) THEN 1 ELSE 0 END AS stock_penalty,
        COALESCE(r.photo_quality_level, 0)::numeric AS quality_score,
        COALESCE(r.visible_damage_level, 0)::numeric AS damage_level_score
    FROM image_map.device_image_labeler_report_en_v1 r
    WHERE r.mapping_version = 2
      AND r.report_version = 1
),
candidates AS (
    SELECT *, 'front'::text AS slot_role, 'best front or screen view'::text AS slot_selection_reason,
           10 AS role_priority,
           (COALESCE(role_confidence, 0)::numeric + quality_score * 0.03 - stock_penalty * 0.20) AS slot_score
    FROM base
    WHERE primary_view_role = 'front_screen'
       OR secondary_view_role = 'front_screen'
       OR (primary_view_role = 'mixed' AND body_visibility IN ('front', 'mixed'))

    UNION ALL
    SELECT *, 'back_camera', 'best back or camera view',
           20,
           (COALESCE(role_confidence, 0)::numeric + quality_score * 0.03 - stock_penalty * 0.20)
    FROM base
    WHERE primary_view_role = 'back_camera'
       OR secondary_view_role = 'back_camera'
       OR (primary_view_role = 'mixed' AND body_visibility IN ('back', 'mixed'))

    UNION ALL
    SELECT *, 'side_frame', 'best side or frame view',
           30,
           (COALESCE(role_confidence, 0)::numeric + quality_score * 0.03 - stock_penalty * 0.20)
    FROM base
    WHERE primary_view_role = 'side_frame'
       OR secondary_view_role = 'side_frame'

    UNION ALL
    SELECT *, 'damage', 'highest visible damage / most useful damage surface',
           40,
           (damage_level_score * 0.20 + COALESCE(damage_surface_confidence, 0)::numeric + quality_score * 0.02 - stock_penalty * 0.30)
    FROM base
    WHERE COALESCE(visible_damage_level, 0) > 0
       OR damage_surface_primary NOT IN ('none_visible', 'unknown')

    UNION ALL
    SELECT *, 'battery', 'battery screenshot or battery-health image',
           50,
           (COALESCE(role_confidence, 0)::numeric + CASE WHEN battery_health_pct_img IS NOT NULL THEN 0.20 ELSE 0 END + quality_score * 0.02)
    FROM base
    WHERE primary_view_role = 'battery_screenshot'
       OR secondary_view_role = 'battery_screenshot'
       OR COALESCE(battery_screenshot, false)

    UNION ALL
    SELECT *, 'accessory', 'box accessory receipt or packaging evidence',
           60,
           (COALESCE(role_confidence, 0)::numeric + quality_score * 0.02 - stock_penalty * 0.10)
    FROM base
    WHERE primary_view_role IN ('accessory_bundle', 'box_only', 'receipt_only')
       OR secondary_view_role IN ('accessory_bundle', 'box_only', 'receipt_only')
       OR COALESCE(has_box, false)
       OR COALESCE(has_charger, false)
       OR COALESCE(has_charger_brick, false)
       OR COALESCE(has_cable, false)
       OR COALESCE(has_case, false)
       OR COALESCE(has_receipt, false)

    UNION ALL
    SELECT *, 'extra', 'highest quality remaining non-duplicate image',
           70,
           (quality_score * 0.12 + COALESCE(role_confidence, 0)::numeric - stock_penalty * 0.35 + damage_level_score * 0.02)
    FROM base
),
role_ranked AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY generation, listing_id, slot_role
            ORDER BY
                slot_score DESC,
                COALESCE(photo_quality_level, 0) DESC,
                CASE WHEN COALESCE(is_stock_photo, false) THEN 1 ELSE 0 END ASC,
                image_index ASC
        ) AS group_rank
    FROM candidates
    WHERE slot_role <> 'extra'
),
role_best AS (
    SELECT *
    FROM role_ranked
    WHERE group_rank = 1
),
role_best_dedup AS (
    SELECT DISTINCT ON (generation, listing_id, image_index, feature_version)
        *
    FROM role_best
    ORDER BY
        generation,
        listing_id,
        image_index,
        feature_version,
        role_priority ASC,
        slot_score DESC
),
extra_ranked AS (
    SELECT
        c.*,
        row_number() OVER (
            PARTITION BY c.generation, c.listing_id
            ORDER BY
                c.slot_score DESC,
                COALESCE(c.photo_quality_level, 0) DESC,
                CASE WHEN COALESCE(c.is_stock_photo, false) THEN 1 ELSE 0 END ASC,
                c.image_index ASC
        ) AS group_rank
    FROM candidates c
    WHERE c.slot_role = 'extra'
      AND NOT EXISTS (
          SELECT 1
          FROM role_best_dedup rb
          WHERE rb.generation = c.generation
            AND rb.listing_id = c.listing_id
            AND rb.image_index = c.image_index
            AND rb.feature_version = c.feature_version
      )
),
extra_selected AS (
    SELECT *
    FROM extra_ranked
    WHERE group_rank <= 8
),
combined_candidates AS (
    SELECT * FROM role_best_dedup
    UNION ALL
    SELECT * FROM extra_selected
),
dedup_image AS (
    SELECT DISTINCT ON (generation, listing_id, image_index, feature_version)
        *
    FROM combined_candidates
    ORDER BY
        generation,
        listing_id,
        image_index,
        feature_version,
        role_priority ASC,
        slot_score DESC
),
ranked AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY generation, listing_id
            ORDER BY
                role_priority ASC,
                slot_score DESC,
                COALESCE(photo_quality_level, 0) DESC,
                CASE WHEN COALESCE(is_stock_photo, false) THEN 1 ELSE 0 END ASC,
                image_index ASC
        ) AS selected_rank
    FROM dedup_image
),
selected AS (
    SELECT *
    FROM ranked
    WHERE selected_rank <= 8
),
listings AS (
    SELECT DISTINCT generation, listing_id
    FROM image_map.device_image_labeler_report_en_v1
    WHERE mapping_version = 2
      AND report_version = 1
),
slot_grid AS (
    SELECT
        l.generation,
        l.listing_id,
        gs.slot_index
    FROM listings l
    CROSS JOIN generate_series(0, 7) AS gs(slot_index)
),
slot_rows AS (
    SELECT
        g.generation,
        g.listing_id,
        1::smallint AS slot_version,
        2::smallint AS mapping_version,
        1::smallint AS report_version,
        g.slot_index::smallint,
        COALESCE(s.slot_role, 'padding') AS slot_role,
        COALESCE(s.slot_selection_reason, 'empty padded slot') AS slot_selection_reason,
        (s.image_index IS NULL) AS is_padding,
        s.image_index,
        s.feature_version,
        s.primary_view_role,
        s.secondary_view_role,
        s.role_confidence,
        s.phone_presence,
        s.body_visibility,
        s.damage_surface_primary,
        s.damage_surface_confidence,
        s.visible_damage_level,
        s.photo_quality_level,
        s.background_clean_level,
        s.is_stock_photo,
        s.battery_screenshot,
        s.battery_health_pct_img,
        s.has_box,
        s.has_charger,
        s.has_case,
        s.has_receipt,
        s.body_color_key,
        s.body_color_confidence,
        CASE
            WHEN s.image_index IS NULL THEN format('Slot %s is padding. No image is selected for this slot.', g.slot_index)
            ELSE s.canonical_image_report_en
        END AS canonical_image_report_en,
        s.slot_score,
        CASE
            WHEN s.image_index IS NULL THEN jsonb_build_object('slot_role', 'padding', 'is_padding', true)
            ELSE jsonb_build_object(
                'slot_role', s.slot_role,
                'slot_selection_reason', s.slot_selection_reason,
                'selected_rank', s.selected_rank,
                'role_priority', s.role_priority,
                'slot_score', s.slot_score,
                'is_padding', false,
                'report_metadata', s.report_metadata_json
            )
        END AS slot_metadata_json,
        now() AS built_at
    FROM slot_grid g
    LEFT JOIN selected s
      ON s.generation = g.generation
     AND s.listing_id = g.listing_id
     AND s.selected_rank = g.slot_index + 1
)
INSERT INTO image_map.device_listing_image_k8_slots_v1 (
    generation,
    listing_id,
    slot_version,
    mapping_version,
    report_version,
    slot_index,
    slot_role,
    slot_selection_reason,
    is_padding,
    image_index,
    feature_version,
    primary_view_role,
    secondary_view_role,
    role_confidence,
    phone_presence,
    body_visibility,
    damage_surface_primary,
    damage_surface_confidence,
    visible_damage_level,
    photo_quality_level,
    background_clean_level,
    is_stock_photo,
    battery_screenshot,
    battery_health_pct_img,
    has_box,
    has_charger,
    has_case,
    has_receipt,
    body_color_key,
    body_color_confidence,
    canonical_image_report_en,
    slot_score,
    slot_metadata_json,
    built_at
)
SELECT
    generation,
    listing_id,
    slot_version,
    mapping_version,
    report_version,
    slot_index,
    slot_role,
    slot_selection_reason,
    is_padding,
    image_index,
    feature_version,
    primary_view_role,
    secondary_view_role,
    role_confidence,
    phone_presence,
    body_visibility,
    damage_surface_primary,
    damage_surface_confidence,
    visible_damage_level,
    photo_quality_level,
    background_clean_level,
    is_stock_photo,
    battery_screenshot,
    battery_health_pct_img,
    has_box,
    has_charger,
    has_case,
    has_receipt,
    body_color_key,
    body_color_confidence,
    canonical_image_report_en,
    slot_score,
    slot_metadata_json,
    built_at
FROM slot_rows;

CREATE INDEX IF NOT EXISTS device_listing_image_k8_slots_v1_image_idx
    ON image_map.device_listing_image_k8_slots_v1 (generation, listing_id, image_index)
    WHERE is_padding = false;

CREATE INDEX IF NOT EXISTS device_listing_image_k8_slots_v1_role_idx
    ON image_map.device_listing_image_k8_slots_v1 (slot_role);

WITH grouped AS (
    SELECT
        generation,
        listing_id,
        slot_version,
        max(mapping_version)::smallint AS mapping_version,
        max(report_version)::smallint AS report_version,
        COUNT(*) FILTER (WHERE is_padding = false)::smallint AS real_slot_count,
        COUNT(*) FILTER (WHERE is_padding = true)::smallint AS padding_slot_count,
        COALESCE(array_agg(image_index ORDER BY slot_index) FILTER (WHERE is_padding = false), ARRAY[]::smallint[]) AS selected_image_indices,
        COALESCE(array_agg(feature_version ORDER BY slot_index) FILTER (WHERE is_padding = false), ARRAY[]::smallint[]) AS selected_feature_versions,
        bool_or(slot_role = 'front') AS has_front,
        bool_or(slot_role = 'back_camera') AS has_back,
        bool_or(slot_role = 'side_frame') AS has_side,
        bool_or(slot_role = 'battery') AS has_battery,
        bool_or(slot_role = 'accessory') AS has_accessory,
        bool_or(slot_role = 'damage') OR bool_or(COALESCE(visible_damage_level, 0) > 0) AS has_damage,
        bool_or(COALESCE(is_stock_photo, false) = false AND is_padding = false) AS has_nonstock,
        COUNT(*) FILTER (WHERE COALESCE(is_stock_photo, false) = true AND is_padding = false)::smallint AS stock_slot_count,
        COUNT(*) FILTER (WHERE (slot_role = 'damage' OR COALESCE(visible_damage_level, 0) > 0) AND is_padding = false)::smallint AS damage_slot_count,
        max(visible_damage_level)::smallint AS max_visible_damage_level,
        round(avg(photo_quality_level) FILTER (WHERE is_padding = false)::numeric, 3) AS avg_photo_quality,
        string_agg(
            CASE
                WHEN is_padding THEN format('Slot %s: padding.', slot_index)
                ELSE format('Slot %s role %s image %s. %s', slot_index, slot_role, image_index, canonical_image_report_en)
            END,
            E'\n'
            ORDER BY slot_index
        ) AS canonical_listing_report_en,
        jsonb_agg(
            jsonb_build_object(
                'slot_index', slot_index,
                'slot_role', slot_role,
                'is_padding', is_padding,
                'image_index', image_index,
                'feature_version', feature_version,
                'primary_view_role', primary_view_role,
                'damage_surface_primary', damage_surface_primary,
                'visible_damage_level', visible_damage_level,
                'photo_quality_level', photo_quality_level,
                'is_stock_photo', is_stock_photo
            )
            ORDER BY slot_index
        ) AS slots_json
    FROM image_map.device_listing_image_k8_slots_v1
    WHERE slot_version = 1
    GROUP BY 1,2,3
)
INSERT INTO image_map.device_listing_image_k8_manifest_v1 (
    generation,
    listing_id,
    slot_version,
    mapping_version,
    report_version,
    real_slot_count,
    padding_slot_count,
    selected_image_indices,
    selected_feature_versions,
    has_front,
    has_back,
    has_side,
    has_battery,
    has_accessory,
    has_damage,
    has_nonstock,
    stock_slot_count,
    damage_slot_count,
    max_visible_damage_level,
    avg_photo_quality,
    ready_for_k8_image_tower,
    needs_additional_labeler,
    canonical_listing_report_en,
    manifest_json,
    built_at
)
SELECT
    generation,
    listing_id,
    slot_version,
    mapping_version,
    report_version,
    real_slot_count,
    padding_slot_count,
    selected_image_indices,
    selected_feature_versions,
    has_front,
    has_back,
    has_side,
    has_battery,
    has_accessory,
    has_damage,
    has_nonstock,
    stock_slot_count,
    damage_slot_count,
    max_visible_damage_level,
    avg_photo_quality,
    (real_slot_count > 0 AND has_nonstock AND (has_front OR has_back)) AS ready_for_k8_image_tower,
    (real_slot_count = 0 OR NOT (has_front OR has_back) OR NOT has_nonstock) AS needs_additional_labeler,
    canonical_listing_report_en,
    jsonb_build_object(
        'contract', 'device_listing_image_k8_manifest_v1',
        'slot_version', slot_version,
        'mapping_version', mapping_version,
        'report_version', report_version,
        'k', 8,
        'selection_policy', 'role-priority unique-image selection with padding and masks',
        'selected_image_indices', selected_image_indices,
        'slots', slots_json
    ) AS manifest_json,
    now() AS built_at
FROM grouped;

CREATE INDEX IF NOT EXISTS device_listing_image_k8_manifest_v1_ready_idx
    ON image_map.device_listing_image_k8_manifest_v1 (ready_for_k8_image_tower, needs_additional_labeler);

COMMIT;
