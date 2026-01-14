# Sources / references (external)

Accessed: **2025-12-28**

These are the primary external references used for this implementation and documentation.

## PostgreSQL documentation (official)

- COPY (bulk load): `https://www.postgresql.org/docs/current/sql-copy.html`
- Partial indexes (used for “only one current release” enforcement): `https://www.postgresql.org/docs/current/indexes-partial.html`
- CREATE VIEW: `https://www.postgresql.org/docs/current/sql-createview.html`
- REFRESH MATERIALIZED VIEW: `https://www.postgresql.org/docs/current/sql-refreshmaterializedview.html`
- BEGIN / transactions: `https://www.postgresql.org/docs/current/sql-begin.html`
- FROM clause + JOIN / table expressions (CROSS JOIN semantics): `https://www.postgresql.org/docs/current/queries-table-expressions.html`

## psycopg2 documentation (COPY via Python)

- Cursor copy methods (e.g., `copy_expert`): `https://www.psycopg.org/docs/cursor.html`

## Apache Airflow documentation (official)

- PostgresOperator / Postgres hook docs: `https://airflow.apache.org/docs/apache-airflow-providers-postgres/stable/operators/postgres_operator.html`
- Authoring & scheduling DAGs: `https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/`

## pg_cron documentation

- pg_cron GitHub repository (cron.schedule examples, install notes): `https://github.com/citusdata/pg_cron`

