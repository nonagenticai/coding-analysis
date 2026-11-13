"""structure db-tables never reports system catalogs as tables (#137).

Live, a py2rs workspace (Rust sqlx target + FastAPI source with schema-parity
helpers) reported pg_attribute / pg_class / pg_constraint / pg_index and a table
named ``tables`` next to the real notes / tags / _sqlx_migrations. ``tables`` was
``information_schema.tables`` with its schema split off before the table was keyed.
"""

from __future__ import annotations

import pytest

from processing.structure.db_tables import is_system_catalog, table_inventory

#: A FastAPI source repo's schema-parity helpers: introspection SQL that reads the
#: Postgres catalogs, next to the app's real DDL and DML.
_PARITY_PY = '''
from sqlalchemy import text

COLUMNS_SQL = """
    SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod)
      FROM pg_catalog.pg_class c
      JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0
      JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
     WHERE n.nspname = 'public' AND NOT a.attisdropped
"""


def columns(conn):
    return conn.execute(text(COLUMNS_SQL)).all()


def indexes(conn):
    return conn.execute(text("""
        SELECT i.relname, ix.indisunique
          FROM pg_index ix
          JOIN pg_class i ON i.oid = ix.indexrelid
    """)).all()


def constraints(conn):
    return conn.execute(text(
        "SELECT conname, contype FROM pg_constraint WHERE conrelid = 'notes'::regclass"
    )).all()


def table_names(conn):
    return conn.execute(text(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    )).scalars().all()


def column_types(conn):
    return conn.execute(text(
        'SELECT column_name FROM "information_schema"."columns" c '
        "JOIN information_schema.table_constraints tc ON tc.table_name = c.table_name"
    )).all()


def activity(conn):
    return conn.execute(text("SELECT count(*) FROM pg_stat_activity")).scalar()


def create_schema(conn):
    conn.execute(text("CREATE TABLE IF NOT EXISTS notes (id serial PRIMARY KEY, body text)"))
    conn.execute(text("CREATE TABLE IF NOT EXISTS tags (id serial PRIMARY KEY, name text)"))


def add_note(conn, body):
    conn.execute(text("INSERT INTO notes (body) VALUES (:b)"), {"b": body})
'''

_SQLX_MIGRATION = """
CREATE TABLE notes (id BIGSERIAL PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE tags (id BIGSERIAL PRIMARY KEY, name TEXT NOT NULL UNIQUE);
"""

_SQLX_RS = """
pub async fn list_notes(pool: &PgPool) -> sqlx::Result<Vec<Note>> {
    sqlx::query_as!(Note, "SELECT n.id, n.body FROM notes n JOIN tags t ON t.id = n.id")
        .fetch_all(pool)
        .await
}

pub async fn applied(pool: &PgPool) -> sqlx::Result<i64> {
    sqlx::query_scalar!("SELECT count(*) FROM _sqlx_migrations").fetch_one(pool).await
}

pub async fn has_table(pool: &PgPool) -> sqlx::Result<bool> {
    sqlx::query_scalar!(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'notes')"
    )
    .fetch_one(pool)
    .await
}
"""


def _rust_str(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def test_workspace_with_introspection_sql_reports_only_real_tables():
    inv = table_inventory(
        {
            "migrations/0001_init.sql": _SQLX_MIGRATION,
            "src/db.rs": _SQLX_RS,
            ".repos/svc/app/parity.py": _PARITY_PY,
        }
    )
    found = {t.table: t for t in inv.tables}
    assert sorted(found) == ["_sqlx_migrations", "notes", "tags"]
    # Positive control: the real tables keep every role from every language.
    assert sorted(r.via for r in found["notes"].defined_in) == ["migration", "sql"]
    assert {r.file for r in found["notes"].read_by} == {"src/db.rs"}
    assert [r.via for r in found["notes"].written_by] == ["sql"]
    assert [r.via for r in found["_sqlx_migrations"].read_by] == ["sqlx"]
    assert all(t.schemas == [] for t in inv.tables)


_CATALOG_SQL = [
    "SELECT relname FROM pg_catalog.pg_class",
    "SELECT * FROM pg_class c JOIN pg_attribute a ON a.attrelid = c.oid",
    "SELECT * FROM pg_type",
    "SELECT * FROM pg_stat_statements",
    "SELECT * FROM PG_CATALOG.PG_INDEX",
    'SELECT * FROM "pg_catalog"."pg_constraint"',
    "SELECT * FROM pg_toast.pg_toast_2619",
    "SELECT table_name FROM information_schema.tables",
    "SELECT * FROM INFORMATION_SCHEMA.TABLES",
    'SELECT * FROM "information_schema"."tables"',
    "SELECT * FROM information_schema.columns c JOIN information_schema.views v ON true",
]


@pytest.mark.parametrize("sql", _CATALOG_SQL)
def test_system_catalog_sql_yields_no_table(sql):
    py = table_inventory({"x.py": f"def f(conn):\n    conn.execute({sql!r})\n"})
    rust = table_inventory({"y.rs": f"fn f() {{ sqlx::query({_rust_str(sql)}); }}\n"})
    assert py.tables == [] and rust.tables == [], sql


def test_user_tables_named_like_catalog_parts_are_kept():
    """Only the catalog itself is dropped: a user table named ``tables``, a
    ``columns`` in a user schema, and a schema-qualified ``public.pg_x`` are real."""
    src = (
        "def f(conn):\n"
        '    conn.execute("SELECT * FROM tables")\n'
        '    conn.execute("SELECT * FROM app.columns")\n'
        '    conn.execute("CREATE TABLE public.pg_jobs (id int)")\n'
        '    conn.execute("SELECT * FROM pgjobs")\n'
    )
    assert [t.table for t in table_inventory({"x.py": src}).tables] == [
        "columns",
        "pg_jobs",
        "pgjobs",
        "tables",
    ]


@pytest.mark.parametrize(
    ("schema", "name", "expected"),
    [
        ("pg_catalog", "pg_class", True),
        ("information_schema", "tables", True),
        ("pg_toast", "pg_toast_1", True),
        ("pg_temp_3", "scratch", True),
        (None, "pg_attribute", True),
        (None, "tables", False),
        ("public", "pg_jobs", False),
        ("billing", "invoices", False),
        (None, "_sqlx_migrations", False),
    ],
)
def test_is_system_catalog(schema, name, expected):
    assert is_system_catalog(schema, name) is expected
