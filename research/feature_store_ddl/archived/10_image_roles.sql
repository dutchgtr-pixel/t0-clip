-- Selected historical implementation; see provenance.json and README.md.
-- Reference only: dependencies, current-state inputs and destructive rebuilds require review.
BEGIN;

CREATE SCHEMA IF NOT EXISTS image_map;

CREATE TABLE IF NOT EXISTS image_map.device_image_role_mapping_v2_det (
    generation integer NOT NULL,
    listing_id bigint NOT NULL,
    image_index smallint NOT NULL,
    feature_version smallint NOT NULL,
    mapping_version smallint NOT NULL,
    primary_view_role text,
    secondary_view_role text,
    role_confidence real,
    phone_presence text,
    body_visibility text,
    damage_surface_primary text,
    damage_surface_confidence real,
    color_evidence_source text,
    text_hint_used boolean,
    caption_hint_used boolean,
    needs_llm_gap_fill boolean,
    role_rule_code text,
    surface_rule_code text,
    evidence_json jsonb,
    built_at timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT device_image_role_mapping_v2_det_pk
        PRIMARY KEY (generation, listing_id, image_index, feature_version, mapping_version)
);

CREATE TABLE IF NOT EXISTS image_map.device_listing_best_image_slots_v2_det (
    generation integer NOT NULL,
    listing_id bigint NOT NULL,
    mapping_version smallint NOT NULL,
    selection_version smallint NOT NULL,
    best_front_image_index smallint,
    best_front_feature_version smallint,
    best_back_image_index smallint,
    best_back_feature_version smallint,
    best_battery_image_index smallint,
    best_battery_feature_version smallint,
    best_accessory_image_index smallint,
    best_accessory_feature_version smallint,
    best_damage_image_index smallint,
    best_damage_feature_version smallint,
    selected_slot_count smallint,
    selection_audit_json jsonb,
    built_at timestamp with time zone NOT NULL DEFAULT now(),
    CONSTRAINT device_listing_best_image_slots_v2_det_pk
        PRIMARY KEY (generation, listing_id, mapping_version, selection_version)
);

TRUNCATE image_map.device_listing_best_image_slots_v2_det;
TRUNCATE image_map.device_image_role_mapping_v2_det;

WITH src AS (
    SELECT
        f.*,
        COALESCE(a.caption_text, '') AS caption_text,
        COALESCE(l.title, '') AS title,
        COALESCE(l.description, '') AS description
    FROM ml.device_image_features_v1 f
    LEFT JOIN "device".device_image_assets a
        ON a.generation = f.generation
       AND a.listing_id = f.listing_id
       AND a.image_index = f.image_index
    LEFT JOIN "device".device_listings l
        ON l.generation = f.generation
       AND l.listing_id = f.listing_id
),
norm AS (
    SELECT
        *,
        lower(
            concat_ws(
                ' ',
                damage_summary_text,
                caption_text,
                coalesce(extra_json->>'damage_summary_text', '')
            )
        ) AS img_text,
        lower(concat_ws(' ', title, description)) AS listing_text
    FROM src
),
signals AS (
    SELECT
        *,
        (img_text ~* '(battery|batteri|kapasitet|maximum capacity|maksimal kapasitet|settings|innstillinger|battery health|batteritilstand)'
            OR COALESCE(battery_screenshot, false)) AS sig_battery,
        (img_text ~* '(receipt|kvittering|invoice|faktura)'
            OR lower(caption_text) ~* '(receipt|kvittering|invoice|faktura)'
            OR COALESCE(has_receipt, false)) AS sig_receipt,
        (img_text ~* '(box|boxed|eske|emballasje|packaging|package|unopened|sealed|seal|forseglet|uåpnet)'
            OR COALESCE(has_box, false)
            OR box_face_view IS NOT NULL
            OR seal_status IS NOT NULL
            OR fresh_unit_visual_type IS NOT NULL) AS sig_box,
        (img_text ~* '(charger|lader|charging|cable|kabel|adapter|brick|case|deksel|cover|tilbehør|accessor)'
            OR COALESCE(has_charger, false)
            OR COALESCE(has_charger_brick, false)
            OR COALESCE(has_cable, false)
            OR COALESCE(has_case, false)
            OR COALESCE(has_screen_guard, false)
            OR COALESCE(has_other_accessory, false)
            OR COALESCE(has_earbuds, false)) AS sig_accessory,
        (img_text ~* '(front|screen|display|skjerm|glass|protector|beskytter|panel|forside|face)') AS sig_front_img,
        (img_text ~* '(back|rear|camera|kamera|bakside|bak glass|back glass|camera area|kameralinse|linse)') AS sig_back_img,
        (img_text ~* '(side|edge|frame|corner|button|ramme|kant|sidekant|hjørne|knapp|chassis)') AS sig_side_img,
        (listing_text ~* '(screen|skjerm|display|front|forside)') AS sig_front_listing,
        (listing_text ~* '(back|bakside|kamera|camera)') AS sig_back_listing,
        (listing_text ~* '(side|edge|frame|ramme|kant|hjørne)') AS sig_side_listing,
        (img_text ~* '(no visible damage|no obvious|clean|ser fin|ingen synlig|uten skade|looks clean)') AS sig_none_visible,
        (img_text ~* '(dead pixel|pixel fault|oled|burn.?in|screen defect|display defect|grønn linje|green line|white line|black line|vertical line|horizontal line|black spot)') AS sig_screen_panel
    FROM norm
),
scores AS (
    SELECT
        *,
        CASE
            WHEN sig_battery THEN 1.00
            ELSE 0.00
        END::numeric AS battery_score,
        CASE
            WHEN sig_front_img THEN 0.86
            WHEN sig_front_listing THEN 0.35
            ELSE 0.00
        END::numeric AS front_score,
        CASE
            WHEN sig_back_img THEN 0.90
            WHEN body_color_key IS NOT NULL AND NOT sig_front_img AND NOT sig_side_img THEN 0.62
            WHEN sig_back_listing THEN 0.35
            ELSE 0.00
        END::numeric AS back_score,
        CASE
            WHEN sig_side_img THEN 0.88
            WHEN sig_side_listing THEN 0.35
            ELSE 0.00
        END::numeric AS side_score,
        CASE
            WHEN sig_receipt THEN 0.95
            ELSE 0.00
        END::numeric AS receipt_score,
        CASE
            WHEN sig_box AND NOT (sig_front_img OR sig_back_img OR sig_side_img) THEN 0.92
            WHEN sig_box THEN 0.42
            ELSE 0.00
        END::numeric AS box_score,
        CASE
            WHEN sig_accessory AND NOT (sig_front_img OR sig_back_img OR sig_side_img) THEN 0.86
            WHEN sig_accessory THEN 0.38
            ELSE 0.00
        END::numeric AS accessory_score
    FROM signals
),
role_scores AS (
    SELECT
        *,
        GREATEST(front_score, back_score, side_score) AS phone_score,
        (
            (CASE WHEN front_score >= 0.70 THEN 1 ELSE 0 END) +
            (CASE WHEN back_score >= 0.70 THEN 1 ELSE 0 END) +
            (CASE WHEN side_score >= 0.70 THEN 1 ELSE 0 END)
        ) AS strong_phone_roles
    FROM scores
),
role_candidates AS (
    SELECT
        r.*,
        CASE WHEN strong_phone_roles >= 2 THEN 0.89 ELSE 0.00 END::numeric AS mixed_score
    FROM role_scores r
),
ranked AS (
    SELECT
        rc.*,
        rr.role_ranked,
        top1.target AS primary_role_raw,
        top1.score AS primary_role_score,
        top2.target AS secondary_role_raw,
        top2.score AS secondary_role_score
    FROM role_candidates rc
    CROSS JOIN LATERAL (
        SELECT jsonb_agg(jsonb_build_object('target', target, 'score', round(score, 4)) ORDER BY score DESC, target) AS role_ranked
        FROM (
            VALUES
                ('battery_screenshot', battery_score),
                ('front_screen', front_score),
                ('back_camera', back_score),
                ('side_frame', side_score),
                ('box_only', box_score),
                ('accessory_bundle', accessory_score),
                ('receipt_only', receipt_score),
                ('mixed', mixed_score)
        ) v(target, score)
        WHERE score > 0
    ) rr
    CROSS JOIN LATERAL (
        SELECT target, score
        FROM (
            VALUES
                ('battery_screenshot', battery_score),
                ('front_screen', front_score),
                ('back_camera', back_score),
                ('side_frame', side_score),
                ('box_only', box_score),
                ('accessory_bundle', accessory_score),
                ('receipt_only', receipt_score),
                ('mixed', mixed_score)
        ) v(target, score)
        ORDER BY score DESC, target
        LIMIT 1
    ) top1
    LEFT JOIN LATERAL (
        SELECT target, score
        FROM (
            VALUES
                ('battery_screenshot', battery_score),
                ('front_screen', front_score),
                ('back_camera', back_score),
                ('side_frame', side_score),
                ('box_only', box_score),
                ('accessory_bundle', accessory_score),
                ('receipt_only', receipt_score),
                ('mixed', mixed_score)
        ) v(target, score)
        WHERE target <> top1.target AND score >= 0.35
        ORDER BY score DESC, target
        LIMIT 1
    ) top2 ON true
),
surface_scores AS (
    SELECT
        *,
        CASE
            WHEN COALESCE(visible_damage_level, 0) > 0 AND sig_screen_panel THEN 0.88
            ELSE 0.00
        END::numeric AS screen_panel_score,
        CASE
            WHEN COALESCE(visible_damage_level, 0) > 0 AND sig_front_img THEN 0.84
            ELSE 0.00
        END::numeric AS front_glass_score,
        CASE
            WHEN COALESCE(visible_damage_level, 0) > 0 AND img_text ~* '(back|rear|bakside|back glass|bak glass)' THEN 0.82
            ELSE 0.00
        END::numeric AS back_glass_score,
        CASE
            WHEN COALESCE(visible_damage_level, 0) > 0 AND sig_side_img THEN 0.82
            ELSE 0.00
        END::numeric AS frame_score,
        CASE
            WHEN COALESCE(visible_damage_level, 0) > 0 AND img_text ~* '(lens|linse|camera lens|kameralinse)' THEN 0.86
            ELSE 0.00
        END::numeric AS camera_lens_score,
        CASE
            WHEN COALESCE(visible_damage_level, 0) > 0 AND img_text ~* '(camera ring|kameraring|around camera|camera bump|camera frame|camera rim)' THEN 0.85
            ELSE 0.00
        END::numeric AS camera_ring_score,
        CASE
            WHEN COALESCE(visible_damage_level, 0) = 0 OR sig_none_visible THEN 0.60
            ELSE 0.00
        END::numeric AS none_visible_score,
        CASE
            WHEN COALESCE(visible_damage_level, 0) > 0
             AND NOT (sig_front_img OR sig_back_img OR sig_side_img OR sig_screen_panel)
            THEN 0.65
            WHEN damage_summary_text IS NULL OR btrim(damage_summary_text) = ''
            THEN 0.30
            ELSE 0.00
        END::numeric AS unknown_surface_score
    FROM ranked
),
surface_ranked AS (
    SELECT
        ss.*,
        sr.surface_ranked,
        s1.target AS primary_surface_raw,
        s1.score AS primary_surface_score
    FROM surface_scores ss
    CROSS JOIN LATERAL (
        SELECT jsonb_agg(jsonb_build_object('target', target, 'score', round(score, 4)) ORDER BY score DESC, target) AS surface_ranked
        FROM (
            VALUES
                ('screen_panel', screen_panel_score),
                ('front_glass', front_glass_score),
                ('back_glass', back_glass_score),
                ('frame', frame_score),
                ('camera_lens', camera_lens_score),
                ('camera_ring', camera_ring_score),
                ('none_visible', none_visible_score),
                ('unknown', unknown_surface_score)
        ) v(target, score)
        WHERE score > 0
    ) sr
    CROSS JOIN LATERAL (
        SELECT target, score
        FROM (
            VALUES
                ('screen_panel', screen_panel_score),
                ('front_glass', front_glass_score),
                ('back_glass', back_glass_score),
                ('frame', frame_score),
                ('camera_lens', camera_lens_score),
                ('camera_ring', camera_ring_score),
                ('none_visible', none_visible_score),
                ('unknown', unknown_surface_score)
        ) v(target, score)
        ORDER BY score DESC, target
        LIMIT 1
    ) s1
),
final_rows AS (
    SELECT
        generation,
        listing_id,
        image_index,
        feature_version,
        2::smallint AS mapping_version,
        CASE WHEN primary_role_score < 0.35 THEN 'unknown' ELSE primary_role_raw END AS primary_view_role,
        CASE
            WHEN secondary_role_score IS NULL THEN NULL
            WHEN secondary_role_score < 0.35 THEN NULL
            ELSE secondary_role_raw
        END AS secondary_view_role,
        CASE WHEN primary_role_score < 0.35 THEN 0.15 ELSE primary_role_score END::real AS role_confidence,
        CASE
            WHEN primary_role_score < 0.35 THEN 'unknown'
            WHEN primary_role_raw IN ('front_screen','back_camera','side_frame','mixed') THEN 'phone_visible'
            WHEN primary_role_raw = 'battery_screenshot' THEN 'screen_capture'
            WHEN primary_role_raw IN ('box_only','accessory_bundle','receipt_only') THEN 'non_phone_primary'
            ELSE 'unknown'
        END AS phone_presence,
        CASE
            WHEN primary_role_raw = 'front_screen' THEN 'front'
            WHEN primary_role_raw = 'back_camera' THEN 'back'
            WHEN primary_role_raw = 'side_frame' THEN 'side'
            WHEN primary_role_raw = 'mixed' THEN 'mixed'
            WHEN primary_role_raw = 'battery_screenshot' THEN 'screen'
            WHEN primary_role_raw IN ('box_only','accessory_bundle','receipt_only') THEN 'none'
            ELSE 'unknown'
        END AS body_visibility,
        COALESCE(primary_surface_raw, 'unknown') AS damage_surface_primary,
        COALESCE(primary_surface_score, 0.15)::real AS damage_surface_confidence,
        CASE
            WHEN body_color_key IS NULL THEN NULL
            WHEN COALESCE(body_color_from_case, false) THEN 'case_color_label'
            ELSE 'image_color_label'
        END AS color_evidence_source,
        (
            sig_front_listing OR sig_back_listing OR sig_side_listing OR
            listing_text ~* '(receipt|kvittering|invoice|faktura|charger|lader|kabel|case|deksel|box|eske)'
        ) AS text_hint_used,
        (
            caption_text IS NOT NULL AND btrim(caption_text) <> '' AND
            lower(caption_text) ~* '(front|screen|skjerm|back|bak|camera|kamera|side|frame|box|eske|battery|batteri|charger|lader|case|deksel)'
        ) AS caption_hint_used,
        (
            primary_role_score < 0.35 OR
            (COALESCE(visible_damage_level, 0) > 0 AND COALESCE(primary_surface_raw, 'unknown') = 'unknown')
        ) AS needs_llm_gap_fill,
        CASE
            WHEN primary_role_score < 0.35 THEN 'unknown_low_score'
            WHEN primary_role_raw = 'mixed' THEN 'mixed_multi_phone_view'
            WHEN primary_role_raw = 'battery_screenshot' THEN 'battery_screenshot_hard'
            WHEN primary_role_raw = 'box_only' AND box_score >= 0.90 THEN 'winner_box_only'
            WHEN primary_role_raw = 'accessory_bundle' THEN 'winner_accessory_bundle'
            WHEN primary_role_raw = 'receipt_only' THEN 'winner_receipt_only'
            ELSE 'winner_' || primary_role_raw
        END AS role_rule_code,
        CASE
            WHEN COALESCE(primary_surface_raw, 'unknown') = 'none_visible' THEN 'surface_winner_none_visible'
            WHEN COALESCE(primary_surface_raw, 'unknown') = 'unknown' THEN 'surface_unknown_or_tie'
            ELSE 'surface_winner_' || primary_surface_raw
        END AS surface_rule_code,
        jsonb_build_object(
            'mapper', 'deterministic_sql_v2',
            'role_ranked', COALESCE(role_ranked, '[]'::jsonb),
            'surface_ranked', COALESCE(surface_ranked, '[]'::jsonb),
            'source_fields_used', jsonb_build_object(
                'structured', true,
                'damage_summary', damage_summary_text IS NOT NULL AND btrim(damage_summary_text) <> '',
                'caption', caption_text IS NOT NULL AND btrim(caption_text) <> '',
                'title_description', listing_text IS NOT NULL AND btrim(listing_text) <> ''
            ),
            'source_flags', jsonb_build_object(
                'battery_screenshot', COALESCE(battery_screenshot, false),
                'has_box', COALESCE(has_box, false),
                'has_accessory_bundle', (
                    COALESCE(has_charger, false) OR COALESCE(has_charger_brick, false) OR
                    COALESCE(has_cable, false) OR COALESCE(has_case, false) OR
                    COALESCE(has_screen_guard, false) OR COALESCE(has_other_accessory, false) OR
                    COALESCE(has_earbuds, false)
                ),
                'has_receipt', COALESCE(has_receipt, false),
                'stock_photo', COALESCE(is_stock_photo, false),
                'damage_summary_present', damage_summary_text IS NOT NULL AND btrim(damage_summary_text) <> ''
            ),
            'scores', jsonb_build_object(
                'front', round(front_score, 4),
                'back', round(back_score, 4),
                'side', round(side_score, 4),
                'battery', round(battery_score, 4),
                'box', round(box_score, 4),
                'accessory', round(accessory_score, 4),
                'receipt', round(receipt_score, 4),
                'mixed', round(mixed_score, 4)
            )
        ) AS evidence_json,
        now() AS built_at
    FROM surface_ranked
)
INSERT INTO image_map.device_image_role_mapping_v2_det (
    generation,
    listing_id,
    image_index,
    feature_version,
    mapping_version,
    primary_view_role,
    secondary_view_role,
    role_confidence,
    phone_presence,
    body_visibility,
    damage_surface_primary,
    damage_surface_confidence,
    color_evidence_source,
    text_hint_used,
    caption_hint_used,
    needs_llm_gap_fill,
    role_rule_code,
    surface_rule_code,
    evidence_json,
    built_at
)
SELECT
    generation,
    listing_id,
    image_index,
    feature_version,
    mapping_version,
    primary_view_role,
    secondary_view_role,
    role_confidence,
    phone_presence,
    body_visibility,
    damage_surface_primary,
    damage_surface_confidence,
    color_evidence_source,
    text_hint_used,
    caption_hint_used,
    needs_llm_gap_fill,
    role_rule_code,
    surface_rule_code,
    evidence_json,
    built_at
FROM final_rows;

CREATE INDEX IF NOT EXISTS device_image_role_mapping_v2_det_role_idx
    ON image_map.device_image_role_mapping_v2_det (primary_view_role);

CREATE INDEX IF NOT EXISTS device_image_role_mapping_v2_det_surface_idx
    ON image_map.device_image_role_mapping_v2_det (damage_surface_primary);

CREATE INDEX IF NOT EXISTS device_image_role_mapping_v2_det_listing_idx
    ON image_map.device_image_role_mapping_v2_det (generation, listing_id);

WITH joined AS (
    SELECT
        m.*,
        f.photo_quality_level,
        f.visible_damage_level,
        f.is_stock_photo
    FROM image_map.device_image_role_mapping_v2_det m
    JOIN ml.device_image_features_v1 f
      ON f.generation = m.generation
     AND f.listing_id = m.listing_id
     AND f.image_index = m.image_index
     AND f.feature_version = m.feature_version
    WHERE m.mapping_version = 2
),
front_pick AS (
    SELECT DISTINCT ON (generation, listing_id)
        generation, listing_id, image_index, feature_version
    FROM joined
    WHERE primary_view_role = 'front_screen'
       OR secondary_view_role = 'front_screen'
       OR (primary_view_role = 'mixed' AND evidence_json->'scores'->>'front' IS NOT NULL)
    ORDER BY generation, listing_id,
        CASE WHEN COALESCE(is_stock_photo, false) THEN 1 ELSE 0 END,
        role_confidence DESC,
        COALESCE(photo_quality_level, 0) DESC,
        image_index ASC
),
back_pick AS (
    SELECT DISTINCT ON (generation, listing_id)
        generation, listing_id, image_index, feature_version
    FROM joined
    WHERE primary_view_role = 'back_camera'
       OR secondary_view_role = 'back_camera'
       OR (primary_view_role = 'mixed' AND evidence_json->'scores'->>'back' IS NOT NULL)
    ORDER BY generation, listing_id,
        CASE WHEN COALESCE(is_stock_photo, false) THEN 1 ELSE 0 END,
        role_confidence DESC,
        COALESCE(photo_quality_level, 0) DESC,
        image_index ASC
),
battery_pick AS (
    SELECT DISTINCT ON (generation, listing_id)
        generation, listing_id, image_index, feature_version
    FROM joined
    WHERE primary_view_role = 'battery_screenshot'
       OR secondary_view_role = 'battery_screenshot'
    ORDER BY generation, listing_id,
        role_confidence DESC,
        COALESCE(photo_quality_level, 0) DESC,
        image_index ASC
),
accessory_pick AS (
    SELECT DISTINCT ON (generation, listing_id)
        generation, listing_id, image_index, feature_version
    FROM joined
    WHERE primary_view_role IN ('accessory_bundle', 'box_only', 'receipt_only')
       OR secondary_view_role IN ('accessory_bundle', 'box_only', 'receipt_only')
    ORDER BY generation, listing_id,
        CASE primary_view_role
            WHEN 'accessory_bundle' THEN 0
            WHEN 'box_only' THEN 1
            WHEN 'receipt_only' THEN 2
            ELSE 3
        END,
        role_confidence DESC,
        COALESCE(photo_quality_level, 0) DESC,
        image_index ASC
),
damage_pick AS (
    SELECT DISTINCT ON (generation, listing_id)
        generation, listing_id, image_index, feature_version
    FROM joined
    WHERE COALESCE(visible_damage_level, 0) > 0
       OR damage_surface_primary NOT IN ('none_visible', 'unknown')
    ORDER BY generation, listing_id,
        COALESCE(visible_damage_level, 0) DESC,
        damage_surface_confidence DESC,
        COALESCE(photo_quality_level, 0) DESC,
        image_index ASC
),
listings AS (
    SELECT DISTINCT generation, listing_id
    FROM image_map.device_image_role_mapping_v2_det
    WHERE mapping_version = 2
)
INSERT INTO image_map.device_listing_best_image_slots_v2_det (
    generation,
    listing_id,
    mapping_version,
    selection_version,
    best_front_image_index,
    best_front_feature_version,
    best_back_image_index,
    best_back_feature_version,
    best_battery_image_index,
    best_battery_feature_version,
    best_accessory_image_index,
    best_accessory_feature_version,
    best_damage_image_index,
    best_damage_feature_version,
    selected_slot_count,
    selection_audit_json,
    built_at
)
SELECT
    l.generation,
    l.listing_id,
    2::smallint AS mapping_version,
    2::smallint AS selection_version,
    fp.image_index AS best_front_image_index,
    fp.feature_version AS best_front_feature_version,
    bp.image_index AS best_back_image_index,
    bp.feature_version AS best_back_feature_version,
    bat.image_index AS best_battery_image_index,
    bat.feature_version AS best_battery_feature_version,
    ap.image_index AS best_accessory_image_index,
    ap.feature_version AS best_accessory_feature_version,
    dp.image_index AS best_damage_image_index,
    dp.feature_version AS best_damage_feature_version,
    (
        (CASE WHEN fp.image_index IS NULL THEN 0 ELSE 1 END) +
        (CASE WHEN bp.image_index IS NULL THEN 0 ELSE 1 END) +
        (CASE WHEN bat.image_index IS NULL THEN 0 ELSE 1 END) +
        (CASE WHEN ap.image_index IS NULL THEN 0 ELSE 1 END) +
        (CASE WHEN dp.image_index IS NULL THEN 0 ELSE 1 END)
    )::smallint AS selected_slot_count,
    jsonb_build_object(
        'mapper', 'deterministic_sql_v2',
        'front', jsonb_build_object('image_index', fp.image_index, 'feature_version', fp.feature_version),
        'back', jsonb_build_object('image_index', bp.image_index, 'feature_version', bp.feature_version),
        'battery', jsonb_build_object('image_index', bat.image_index, 'feature_version', bat.feature_version),
        'accessory', jsonb_build_object('image_index', ap.image_index, 'feature_version', ap.feature_version),
        'damage', jsonb_build_object('image_index', dp.image_index, 'feature_version', dp.feature_version)
    ) AS selection_audit_json,
    now() AS built_at
FROM listings l
LEFT JOIN front_pick fp USING (generation, listing_id)
LEFT JOIN back_pick bp USING (generation, listing_id)
LEFT JOIN battery_pick bat USING (generation, listing_id)
LEFT JOIN accessory_pick ap USING (generation, listing_id)
LEFT JOIN damage_pick dp USING (generation, listing_id);

CREATE INDEX IF NOT EXISTS device_listing_best_image_slots_v2_det_slots_idx
    ON image_map.device_listing_best_image_slots_v2_det (selected_slot_count);

COMMIT;
