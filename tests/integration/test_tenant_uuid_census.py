"""The migration-0005 census guard, against a real PostgreSQL.

WHY THIS SUITE IS NOT MOCKED
----------------------------
Every claim it makes is a claim about postgres semantics — what ``::uuid``
accepts, that ``uuid`` folds case, that ``NULLS NOT DISTINCT`` collapses NULL
rows. A double would be asserting my own beliefs about those, which is exactly
the failure mode the census exists to prevent. It therefore lives in
``tests/integration`` (marked ``integration``, gated on ``TEST_DATABASE_URL``),
which is the pass CI runs with a real ``postgres:18`` — see ``ci-cd.yaml``'s
``integration_test_select``.

THE CONTROL COMES FIRST
-----------------------
``test_control_many_rows_on_one_legitimate_tenant_migrates_green`` is the load-
bearing test in this file, and it is a GREEN one. The obvious-looking way to
write the case-collision guard is::

    count(*) <> count(DISTINCT lower(tenant_id))      -- WRONG

which counts ROWS, so it refuses every healthy table where more than one row
shares a tenant — and in prod 450,279 ``nodes`` rows share exactly one. Under
that mutant the census raises on the control, the deploy bricks, and no RED test
in this file would have noticed, because they are all asserting that a raise
happens. The right predicate counts SPELLINGS::

    GROUP BY lower(tenant_id) HAVING count(DISTINCT tenant_id) > 1

Mutation-tested: flipping the predicate in ``case_only_collisions`` to the
``count(*)`` form turns ``test_control_many_rows_on_one_legitimate_tenant_
migrates_green`` RED and leaves every other test in this file green.

⚠️  The tenant literals seeded below are DELIBERATELY DIRTY. ``"0001"``,
``"tenant-A"`` and the mixed-case uuid are the specimens under test, not
oversights. Do not "normalise" them with a bulk substitution — that silently
turns this file into a suite that can no longer fail.
"""

from __future__ import annotations

import ast
import os
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa

pytestmark = pytest.mark.integration

_DSN = os.getenv("TEST_DATABASE_URL")
if not _DSN:
    pytest.skip("TEST_DATABASE_URL not set", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: A tenant that really is a uuid7 — the healthy case.
LEGIT_TENANT = "0195f48b-0000-7000-8000-000000000003"
#: A second healthy tenant, so "one tenant" is a property of the data and not of
#: the table only ever having held one value.
OTHER_TENANT = "019de842-c92a-7e37-8fdd-eb0d8c5396f9"


@pytest.fixture(scope="module")
def migration():
    """The real revision-0005 module, loaded through alembic's own script directory.

    Loaded via ``ScriptDirectory`` rather than by file path so the test also
    proves the revision is discoverable and correctly chained — a file that
    alembic cannot see would otherwise sail through this whole suite.
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    script = ScriptDirectory.from_config(cfg)
    revision = script.get_revision("0005")
    assert revision.down_revision == "0004", "0005 is not chained onto 0004"
    return revision.module


@pytest.fixture(scope="module")
def engine():
    url = _DSN.replace("postgresql://", "postgresql+psycopg://", 1)
    eng = sa.create_engine(url, poolclass=sa.pool.NullPool)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module", autouse=True)
def _migrated_db(engine):
    """Bring the schema to head so ``tenant_id_uuid_quarantine`` exists.

    0005 creates the quarantine table and only ``downgrade`` drops it, so at head
    the quarantine round-trip test can use the REAL table rather than a hand-typed
    copy of its DDL that could drift away from the migration.
    """
    os.environ["DATABASE_URL"] = _DSN
    os.environ["PGBOUNCER_URL"] = ""

    from alembic.config import Config

    from alembic import command

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    command.upgrade(cfg, "head")
    with engine.connect() as conn:
        assert conn.execute(
            sa.text("SELECT to_regclass('tenant_id_uuid_quarantine')")
        ).scalar_one(), "0005 did not leave the quarantine table behind"
    yield
    command.downgrade(cfg, "base")


@pytest.fixture
def conn(engine):
    """One transaction per test, rolled back — nothing this suite writes survives."""
    with engine.connect() as connection:
        trans = connection.begin()
        try:
            yield connection
        finally:
            trans.rollback()


@pytest.fixture
def scratch(conn, migration):
    """Build a ``nodes``-shaped table and hand back its :class:`TenantColumn`.

    Shaped like the real thing where it matters: ``tenant_id varchar(128)``
    NULLABLE, identity ``(code_graph_id, tenant_id, id)`` unique with
    ``NULLS NOT DISTINCT``. Uniquely named per test so a leaked transaction
    cannot make two tests share a table.
    """
    created: list[str] = []

    def _make(nullable: bool = True) -> object:
        name = f"census_scratch_{uuid.uuid4().hex[:12]}"
        null_sql = "NULL" if nullable else "NOT NULL"
        conn.execute(
            sa.text(
                f"CREATE TABLE {name} ("
                f"  code_graph_id varchar(255) NOT NULL,"
                f"  id varchar(512) NOT NULL,"
                f"  tenant_id varchar(128) {null_sql}"
                f")"
            )
        )
        conn.execute(
            sa.text(
                f"CREATE UNIQUE INDEX uq_{name} ON {name} "
                f"(code_graph_id, tenant_id, id) NULLS NOT DISTINCT"
            )
        )
        created.append(name)
        return migration.TenantColumn(
            table=name,
            column="tenant_id",
            varchar_length=128,
            nullable=nullable,
            identity_columns=("code_graph_id", "id"),
        )

    yield _make
    for name in created:
        conn.execute(sa.text(f"DROP TABLE IF EXISTS {name}"))


def _seed(conn, col, rows: list[tuple[str, str, str | None]]) -> None:
    conn.execute(
        sa.text(
            f"INSERT INTO {col.table} (code_graph_id, id, tenant_id) VALUES (:cg, :id, :tenant)"
        ),
        [{"cg": cg, "id": rid, "tenant": tenant} for cg, rid, tenant in rows],
    )


def _alter_to_uuid(conn, col) -> None:
    conn.execute(
        sa.text(f"ALTER TABLE {col.table} ALTER COLUMN tenant_id TYPE uuid USING tenant_id::uuid")
    )


# ---------------------------------------------------------------------------
# CONTROL — the green case the wrong predicate destroys
# ---------------------------------------------------------------------------


def test_control_many_rows_on_one_legitimate_tenant_migrates_green(conn, scratch, migration):
    """5,000 healthy rows sharing ONE uuid must pass the census and ALTER cleanly.

    This is the shape of the real ``nodes`` table (450,279 rows on one tenant) and
    it is the case a row-counting collision predicate refuses. If this test is red,
    the census cannot be shipped: it would brick the in-cluster migrate
    initContainer on data that has nothing wrong with it.
    """
    col = scratch()
    rows = [("cg-control", f"node-{i}", LEGIT_TENANT) for i in range(5_000)]
    rows += [("cg-control", f"other-{i}", OTHER_TENANT) for i in range(500)]
    rows += [("cg-control", f"legacy-{i}", None) for i in range(50)]
    _seed(conn, col, rows)

    assert migration.non_uuid_values(conn, col) == []
    assert migration.case_only_collisions(conn, col) == [], (
        "the collision guard flagged a table whose only 'collision' is that many "
        "rows share one tenant — it is counting rows, not spellings"
    )

    monkey = migration.TABLES
    try:
        migration.TABLES = (col,)
        migration.census(conn)  # must NOT raise
    finally:
        migration.TABLES = monkey

    _alter_to_uuid(conn, col)
    assert (
        conn.execute(
            sa.text(
                f"SELECT count(*) FROM {col.table} WHERE tenant_id = CAST(:t AS uuid)"  # noqa: S608
            ),
            {"t": LEGIT_TENANT},
        ).scalar_one()
        == 5_000
    )


def test_control_is_not_vacuous_the_row_counting_predicate_would_refuse_it(conn, scratch):
    """Prove the mutant this file guards against really does fire on the control.

    Without this, "the control passes" is compatible with the wrong predicate
    simply never having been reachable. Here the WRONG form is written out
    explicitly and measured: on the same data it reports a collision.
    """
    col = scratch()
    _seed(conn, col, [("cg-control", f"node-{i}", LEGIT_TENANT) for i in range(200)])

    wrong = conn.execute(
        sa.text(  # noqa: S608 - table name is generated by this test
            f"SELECT count(*) <> count(DISTINCT lower(tenant_id)) FROM {col.table} "
            f"WHERE tenant_id IS NOT NULL"
        )
    ).scalar_one()
    assert wrong is True, (
        "the count(*) form did NOT flag 200 healthy rows on one tenant — this "
        "control no longer distinguishes the right predicate from the wrong one"
    )


# ---------------------------------------------------------------------------
# RED cases
# ---------------------------------------------------------------------------


def test_case_only_spellings_of_one_uuid_are_refused(conn, scratch, migration):
    """Two spellings that differ only in case fold to ONE uuid — a silent collapse."""
    col = scratch()
    upper = LEGIT_TENANT.upper()
    assert upper != LEGIT_TENANT
    _seed(
        conn,
        col,
        [
            ("cg-a", "n1", LEGIT_TENANT),
            ("cg-a", "n2", upper),
            ("cg-a", "n3", OTHER_TENANT),
        ],
    )

    collisions = migration.case_only_collisions(conn, col)
    assert collisions == [(LEGIT_TENANT, 2)], collisions

    monkey = migration.TABLES
    try:
        migration.TABLES = (col,)
        with pytest.raises(RuntimeError) as excinfo:
            migration.census(conn)
    finally:
        migration.TABLES = monkey

    message = str(excinfo.value)
    assert LEGIT_TENANT in message
    assert "no table was altered" in message


def test_a_non_uuid_outside_the_allowlist_is_refused(conn, scratch, migration):
    """An unrecognised placeholder RAISES; it is never guessed at or quietly nulled.

    ``"tenant-A"`` is the shape this repo's own integration fixtures used before
    this change. It is not in ``QUARANTINE_VALUES`` and must not be.
    """
    col = scratch()
    assert "tenant-A" not in migration.QUARANTINE_VALUES
    _seed(conn, col, [("cg-a", "n1", LEGIT_TENANT), ("cg-a", "n2", "tenant-A")])

    assert migration.non_uuid_values(conn, col) == [("tenant-A", 1)]

    monkey = migration.TABLES
    try:
        migration.TABLES = (col,)
        with pytest.raises(RuntimeError) as excinfo:
            migration.census(conn)
    finally:
        migration.TABLES = monkey

    message = str(excinfo.value)
    assert "tenant-A" in message
    assert "1 row(s)" in message
    assert "will not guess an owner" in message


def test_the_allowlist_is_an_explicit_literal_set(migration):
    """The allowlist must stay a hand-written set, not a pattern or a computation.

    A pattern here ("anything short", "anything numeric") would silently absorb
    the NEXT placeholder without anyone reviewing it, which is the difference
    between a census and a rubber stamp.
    """
    assert migration.QUARANTINE_VALUES == frozenset({"0001"})


def test_a_non_nullable_column_is_not_quarantined_but_refused(conn, scratch, migration):
    """``bug_hunts.tenant_id`` is NOT NULL — a placeholder there has no honest home.

    Quarantine must decline (NULL is not available), which leaves the census to
    refuse. Silently inventing a uuid for it would be the worst of both.
    """
    col = scratch(nullable=False)
    _seed(conn, col, [("cg-a", "n1", "0001")])

    assert migration.quarantine(conn, col) == 0

    monkey = migration.TABLES
    try:
        migration.TABLES = (col,)
        with pytest.raises(RuntimeError, match="0001"):
            migration.census(conn)
    finally:
        migration.TABLES = monkey


# ---------------------------------------------------------------------------
# Quarantine round trip
# ---------------------------------------------------------------------------


def test_the_allowlisted_placeholder_survives_a_quarantine_round_trip(conn, scratch, migration):
    """``"0001"`` goes out to NULL, the ALTER succeeds, and downgrade puts it back.

    This is what makes ``downgrade()`` real rather than nominal: without the
    quarantine table the original value is simply gone and the reverse migration
    is a lie that happens to run.
    """
    col = scratch()
    _seed(
        conn,
        col,
        [
            ("accept-graph", "n1", "0001"),
            ("probe-pysrc", "n2", "0001"),
            ("accept-gdel", "n3", "0001"),
            ("cg-real", "n4", LEGIT_TENANT),
            ("cg-real", "n5", None),
        ],
    )

    moved = migration.quarantine(conn, col)
    assert moved == 3

    stored = conn.execute(
        sa.text(
            "SELECT row_identity->>'code_graph_id' AS cg, row_identity->>'id' AS id, "
            "old_tenant_id FROM tenant_id_uuid_quarantine WHERE table_name = :t "
            "ORDER BY 1"
        ),
        {"t": col.table},
    ).all()
    assert [(r.cg, r.id, r.old_tenant_id) for r in stored] == [
        ("accept-gdel", "n3", "0001"),
        ("accept-graph", "n1", "0001"),
        ("probe-pysrc", "n2", "0001"),
    ]

    # With the placeholder out of the way the census passes and the ALTER lands.
    monkey = migration.TABLES
    try:
        migration.TABLES = (col,)
        migration.census(conn)
    finally:
        migration.TABLES = monkey
    _alter_to_uuid(conn, col)

    # The downgrade half: back to varchar, then restore from quarantine.
    conn.execute(
        sa.text(
            f"ALTER TABLE {col.table} ALTER COLUMN tenant_id "
            f"TYPE VARCHAR({col.varchar_length}) USING tenant_id::text"
        )
    )
    restored = migration.restore_quarantined(conn, col)
    assert restored == 3

    final = conn.execute(
        sa.text(  # noqa: S608 - table name is generated by this test
            f"SELECT code_graph_id, id, tenant_id FROM {col.table} ORDER BY id"
        )
    ).all()
    assert [(r.code_graph_id, r.id, r.tenant_id) for r in final] == [
        ("accept-graph", "n1", "0001"),
        ("probe-pysrc", "n2", "0001"),
        ("accept-gdel", "n3", "0001"),
        ("cg-real", "n4", LEGIT_TENANT),
        ("cg-real", "n5", None),
    ]


def test_quarantine_refuses_when_nulling_would_collide(conn, scratch, migration):
    """Nulling must not merge a placeholder row into an existing NULL-tenant row.

    Under ``NULLS NOT DISTINCT`` two rows with the same ``(code_graph_id, id)``
    and a NULL tenant are ONE row. Rather than let that surface as a bare 23505
    from an index the reader has to go and look up, the migration checks first and
    names the rows. (Measured on prod this cannot happen — the three ``"0001"``
    graphs hold no other tenant — but a guard that is only correct on today's data
    is not a guard.)
    """
    col = scratch()
    _seed(conn, col, [("cg-x", "n1", None), ("cg-x", "n1", "0001")])

    with pytest.raises(RuntimeError, match="would collide with existing NULL-tenant rows"):
        migration.quarantine(conn, col)


def test_the_migration_names_its_tables_literally(migration):
    """The DDL must be driven by a hardcoded list, never by ``Base.metadata``.

    A metadata walk carries the POST-migration type, so it compares uuid against
    uuid and is invisible in prod while being fatal on a from-zero CI run
    (``operator does not exist: character varying = uuid``). This asserts the five
    real tables are named as literals and that the ORM is not consulted.
    """
    named = {(c.table, c.column) for c in migration.TABLES}
    assert named == {
        ("nodes", "tenant_id"),
        ("edges", "tenant_id"),
        ("dependency_report_snapshots", "tenant_id"),
        ("bug_hunts", "tenant_id"),
        ("bug_hunt_findings", "tenant_id"),
    }
    # Checked over the AST, not the raw text: the migration's own docstring
    # explains at length WHY it does not walk ``Base.metadata``, and a substring
    # search cannot tell that prose apart from a call. Prose must stay legal.
    tree = ast.parse(Path(migration.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not any(m and m.startswith("db") for m in imported), (
        f"revision 0005 imports the ORM package: {sorted(m for m in imported if m)}"
    )
    assert not any(isinstance(node, ast.Name) and node.id == "Base" for node in ast.walk(tree)), (
        "revision 0005 references Base — the table list must stay hardcoded"
    )
