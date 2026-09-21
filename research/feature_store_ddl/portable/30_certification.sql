CREATE TABLE ref_audit.certificate (
    contract_id text PRIMARY KEY,
    certified_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    row_count bigint NOT NULL,
    content_digest text NOT NULL,
    definition_digest text NOT NULL
);
CREATE FUNCTION ref_audit.content_digest() RETURNS text LANGUAGE sql STABLE AS $$
    SELECT md5(COALESCE(string_agg(row_to_json(f)::text, E'\n' ORDER BY f.entity_id), ''))
    FROM ref_feature.assembled_v f
$$;
CREATE FUNCTION ref_audit.definition_digest() RETURNS text LANGUAGE sql STABLE AS $$
    SELECT md5(string_agg(pg_get_viewdef(c.oid, true), E'\n' ORDER BY c.relname))
    FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'ref_feature' AND c.relkind = 'v'
$$;
CREATE PROCEDURE ref_audit.certify() LANGUAGE plpgsql AS $$
BEGIN
    IF EXISTS (SELECT 1 FROM ref_feature.assembled_v WHERE NOT main_ready)
       OR EXISTS (SELECT 1 FROM ref_feature.labels_v WHERE invalid_origin)
    THEN RAISE EXCEPTION 'Temporal contract failed'; END IF;
    INSERT INTO ref_audit.certificate
    SELECT 'portable_v1', clock_timestamp(), clock_timestamp() + interval '1 hour',
           count(*), ref_audit.content_digest(), ref_audit.definition_digest()
    FROM ref_feature.assembled_v
    ON CONFLICT (contract_id) DO UPDATE SET
        certified_at = EXCLUDED.certified_at, expires_at = EXCLUDED.expires_at,
        row_count = EXCLUDED.row_count, content_digest = EXCLUDED.content_digest,
        definition_digest = EXCLUDED.definition_digest;
END
$$;
CREATE FUNCTION ref_audit.require_certified() RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM ref_audit.certificate c
        WHERE c.contract_id = 'portable_v1' AND c.expires_at > clock_timestamp()
          AND c.content_digest = ref_audit.content_digest()
          AND c.definition_digest = ref_audit.definition_digest()
          AND c.row_count = (SELECT count(*) FROM ref_feature.assembled_v)
    ) THEN RAISE EXCEPTION 'Missing, stale or changed temporal contract'; END IF;
END
$$;
CREATE FUNCTION ref_feature.read_certified() RETURNS SETOF ref_feature.assembled_v
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM ref_audit.require_certified();
    RETURN QUERY SELECT * FROM ref_feature.assembled_v;
END
$$;
-- These change-detection hashes are illustrative, not cryptographic attestation.
-- Certification does not prove the truth of supplied source availability times.
