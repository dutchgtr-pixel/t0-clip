-- 02_rebaseline_last_n_days.sql
-- UPSERT dataset hash baselines for most recent N t0 days.

CREATE OR REPLACE PROCEDURE audit.rebaseline_last_n_days(
  p_entrypoint regclass,
  p_n int DEFAULT 10,
  p_sample_limit int DEFAULT 2000
)
LANGUAGE plpgsql
AS $$
BEGIN
  EXECUTE format($SQL$
    WITH picked AS (
      SELECT t0_day
      FROM (
        SELECT DISTINCT edited_date::date AS t0_day
        FROM %s
        WHERE edited_date IS NOT NULL
      ) d
      ORDER BY t0_day DESC
      LIMIT %s
    )
    INSERT INTO audit.t0_dataset_hash_baseline(entrypoint, t0_day, sample_limit, dataset_sha256)
    SELECT
      %L,
      p.t0_day,
      %s,
      audit.dataset_sha256(%L::regclass, p.t0_day, %s)
    FROM picked p
    ON CONFLICT (entrypoint, t0_day, sample_limit)
    DO UPDATE SET dataset_sha256 = EXCLUDED.dataset_sha256,
                  computed_at    = now();
  $SQL$,
    p_entrypoint, p_n,
    p_entrypoint::text, p_sample_limit,
    p_entrypoint::text, p_sample_limit
  );
END $$;
