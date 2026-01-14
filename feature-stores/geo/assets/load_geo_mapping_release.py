#!/usr/bin/env python3
"""Load a versioned geo mapping release into Postgres.

This script is designed for your Phase-1 schema:
  - ref.geo_mapping_release
  - ref.postal_code_to_super_metro
  - ref.city_to_super_metro
  - ref.*_current views which select the row where geo_mapping_release.is_current = true

It creates a new release row, bulk loads the two CSV files into TEMP staging tables
using COPY ... FROM STDIN (fast, works even when Postgres runs in Docker),
then inserts into the versioned tables with the new release_id.

Safety properties:
  - Does NOT touch your raw listing tables.
  - Does NOT truncate mapping tables; it APPENDS a new release.
  - Rollback is possible by flipping is_current on ref.geo_mapping_release.

Usage (PowerShell example):
  $env:PG_DSN = "<REDACTED_PG_DSN>"
  python .\load_geo_mapping_release.py \
    --label super_metro_v4_2025_12_28 \
    --postal_csv .\postal_code_to_super_metro_v4.csv \
    --city_csv .\city_postal_codes_with_region_super_metro_v4.csv \
    --notes "super_metro v4 from research" \
    --make_current

"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, text


@dataclass
class LoadStats:
    release_id: int
    postal_rows: int
    city_rows: int


def _require_file(path_str: str) -> Path:
    p = Path(path_str)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return p


def _get_dsn(cli_val: str | None) -> str:
    dsn = cli_val or os.environ.get("PG_DSN")
    if not dsn:
        raise SystemExit(
            "Missing PG DSN. Provide --pg_dsn or set env var PG_DSN."
        )
    return dsn


def load_release(
    *,
    pg_dsn: str,
    label: str,
    postal_csv: Path,
    city_csv: Path,
    notes: str | None,
    source: str | None,
    make_current: bool,
    dry_run: bool,
) -> LoadStats:
    engine = create_engine(pg_dsn)

    # raw_connection() gives us a DBAPI connection (psycopg2 under the hood)
    # so we can use cursor.copy_expert("COPY ... FROM STDIN") efficiently.
    conn = engine.raw_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute("BEGIN;")

            # 1) Create release row and capture release_id
            cur.execute(
                """
                INSERT INTO ref.geo_mapping_release(label, notes, is_current)
                VALUES (%s, %s, false)
                RETURNING release_id;
                """,
                (label, notes),
            )
            release_id = int(cur.fetchone()[0])

            # 2) Create TEMP staging tables that match the CSV headers.
            #    (All columns as TEXT => no accidental numeric coercion; preserves leading zeros.)
            cur.execute(
                """
                CREATE TEMP TABLE tmp_postal_map (
                  postal_code text,
                  region text,
                  fylke text,
                  kommune text,
                  kommune_code text,
                  centrality_class text,
                  pop text,
                  lat text,
                  lon text,
                  super_metro_v4 text
                ) ON COMMIT DROP;
                """
            )
            cur.execute(
                """
                CREATE TEMP TABLE tmp_city_map (
                  location_city text,
                  region text,
                  pickup_metro_30_200 text,
                  oslo_subarea text,
                  postal_codes text,
                  super_metro_v4 text
                ) ON COMMIT DROP;
                """
            )

            # 3) Bulk COPY the CSV files into temp tables (client -> server via STDIN)
            with postal_csv.open("r", encoding="utf-8") as f:
                cur.copy_expert(
                    "COPY tmp_postal_map FROM STDIN WITH (FORMAT csv, HEADER true);",
                    f,
                )
            with city_csv.open("r", encoding="utf-8") as f:
                cur.copy_expert(
                    "COPY tmp_city_map FROM STDIN WITH (FORMAT csv, HEADER true);",
                    f,
                )

            # 4) QA checks BEFORE inserting into the real tables
            #    4a) Postal codes must be exactly 4 digits
            cur.execute(
                """
                SELECT COUNT(*)
                FROM tmp_postal_map
                WHERE postal_code IS NULL OR postal_code !~ '^[0-9]{4}$';
                """
            )
            bad_pc = int(cur.fetchone()[0])
            if bad_pc != 0:
                raise ValueError(
                    f"Postal CSV contains {bad_pc} invalid postal_code values (must be exactly 4 digits)."
                )

            #    4b) Required fields present
            cur.execute(
                """
                SELECT COUNT(*)
                FROM tmp_postal_map
                WHERE region IS NULL OR btrim(region) = '' OR super_metro_v4 IS NULL OR btrim(super_metro_v4) = '';
                """
            )
            bad_req_postal = int(cur.fetchone()[0])
            if bad_req_postal != 0:
                raise ValueError(
                    f"Postal CSV contains {bad_req_postal} rows with missing region/super_metro_v4."
                )

            cur.execute(
                """
                SELECT COUNT(*)
                FROM tmp_city_map
                WHERE location_city IS NULL OR btrim(location_city) = ''
                   OR region IS NULL OR btrim(region) = ''
                   OR super_metro_v4 IS NULL OR btrim(super_metro_v4) = '';
                """
            )
            bad_req_city = int(cur.fetchone()[0])
            if bad_req_city != 0:
                raise ValueError(
                    f"City CSV contains {bad_req_city} rows with missing location_city/region/super_metro_v4."
                )

            #    4c) Duplicates inside the CSVs
            cur.execute(
                """
                SELECT COUNT(*)
                FROM (
                  SELECT postal_code FROM tmp_postal_map GROUP BY 1 HAVING COUNT(*) > 1
                ) d;
                """
            )
            dup_pc = int(cur.fetchone()[0])
            if dup_pc != 0:
                raise ValueError(
                    f"Postal CSV contains {dup_pc} duplicated postal_code values. Deduplicate before loading."
                )

            cur.execute(
                """
                SELECT COUNT(*)
                FROM (
                  SELECT lower(regexp_replace(btrim(location_city), '\\s+', ' ', 'g')) AS k
                  FROM tmp_city_map
                  GROUP BY 1 HAVING COUNT(*) > 1
                ) d;
                """
            )
            dup_city = int(cur.fetchone()[0])
            if dup_city != 0:
                raise ValueError(
                    f"City CSV contains {dup_city} duplicated city keys after normalization. Deduplicate before loading."
                )

            # 5) Insert into versioned mapping tables
            #    We normalize the city key here (lower + trim + collapse whitespace).
            #    We also store the CSV basename in source unless overridden.
            postal_source = source or postal_csv.name
            city_source = source or city_csv.name

            cur.execute(
                """
                INSERT INTO ref.postal_code_to_super_metro(
                  release_id, postal_code, region, pickup_metro_30_200, super_metro_v4, source
                )
                SELECT
                  %s AS release_id,
                  postal_code,
                  region,
                  NULL::text AS pickup_metro_30_200,
                  super_metro_v4,
                  %s AS source
                FROM tmp_postal_map;
                """,
                (release_id, postal_source),
            )

            cur.execute(
                """
                INSERT INTO ref.city_to_super_metro(
                  release_id, location_city_norm, region, pickup_metro_30_200, super_metro_v4, source
                )
                SELECT
                  %s AS release_id,
                  lower(regexp_replace(btrim(location_city), '\\s+', ' ', 'g')) AS location_city_norm,
                  region,
                  pickup_metro_30_200,
                  super_metro_v4,
                  %s AS source
                FROM tmp_city_map;
                """,
                (release_id, city_source),
            )

            # 6) Make this release current (optional)
            if make_current:
                # set old -> false first (so the unique index isn't violated)
                cur.execute("UPDATE ref.geo_mapping_release SET is_current = false WHERE is_current = true;")
                cur.execute(
                    "UPDATE ref.geo_mapping_release SET is_current = true WHERE release_id = %s;",
                    (release_id,),
                )

            # 7) Count rows inserted (for logging)
            cur.execute(
                "SELECT COUNT(*) FROM ref.postal_code_to_super_metro WHERE release_id = %s;",
                (release_id,),
            )
            postal_rows = int(cur.fetchone()[0])

            cur.execute(
                "SELECT COUNT(*) FROM ref.city_to_super_metro WHERE release_id = %s;",
                (release_id,),
            )
            city_rows = int(cur.fetchone()[0])

            if dry_run:
                cur.execute("ROLLBACK;")
            else:
                cur.execute("COMMIT;")

            return LoadStats(release_id=release_id, postal_rows=postal_rows, city_rows=city_rows)

        finally:
            cur.close()
    finally:
        conn.close()
        engine.dispose()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg_dsn", default=None, help="SQLAlchemy DSN. Or set env PG_DSN.")
    ap.add_argument("--label", required=True, help="Release label (e.g., super_metro_v4_2025_12_28)")
    ap.add_argument("--notes", default=None, help="Optional release notes")
    ap.add_argument("--source", default=None, help="Optional source string stored in mapping tables")
    ap.add_argument("--postal_csv", required=True, help="Path to postal_code_to_super_metro_v4.csv")
    ap.add_argument("--city_csv", required=True, help="Path to city_postal_codes_with_region_super_metro_v4.csv")
    ap.add_argument("--make_current", action="store_true", help="Mark this release as current")
    ap.add_argument("--dry_run", action="store_true", help="Validate + load in a transaction, then ROLLBACK")

    args = ap.parse_args()

    pg_dsn = _get_dsn(args.pg_dsn)
    postal_csv = _require_file(args.postal_csv)
    city_csv = _require_file(args.city_csv)

    stats = load_release(
        pg_dsn=pg_dsn,
        label=args.label,
        postal_csv=postal_csv,
        city_csv=city_csv,
        notes=args.notes,
        source=args.source,
        make_current=args.make_current,
        dry_run=args.dry_run,
    )

    print("OK")
    print(f"  release_id: {stats.release_id}")
    print(f"  postal rows inserted: {stats.postal_rows}")
    print(f"  city rows inserted:   {stats.city_rows}")
    print(f"  current? {'yes' if args.make_current and not args.dry_run else 'no (dry_run or not requested)'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
