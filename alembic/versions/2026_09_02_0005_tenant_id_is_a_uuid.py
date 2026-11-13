"""Make ``tenant_id`` a real PostgreSQL ``uuid`` instead of ``character varying``.

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-02

WHY
---
Tenant identity in this estate has been a **uuid7** since the 2026-04-29 Keycloak
migration (the old text label was demoted to a ``tenant_name`` attribute — see
``tests/unit/test_no_tenant_literal.py`` for the incident that made that explicit).
Every ``tenant_id`` column here was still ``varchar(128)``, which means the database
accepted — and did accept — values that are not tenants at all. A ``uuid`` column
cannot: the type is the guard, and it is enforced on every future write by postgres
rather than by whichever call site remembered to validate.

WHAT THE DATA LOOKED LIKE (measured on prod ``coding_analysis`` @ worker4, at 0004)
----------------------------------------------------------------------------------
``nodes`` 471,767 rows / ``edges`` 501,752 rows / ``dependency_report_snapshots``
397 rows; ``bug_hunts`` and ``bug_hunt_findings`` EMPTY. Of ~974k tenant-bearing
rows, **47** carried a value that is not a uuid, and every one of those 47 carried
the same literal: ``"0001"``.

THE ``"0001"`` DISPOSITION
--------------------------
Those 47 rows belong to exactly three ``code_graph_id`` values — ``accept-graph``,
``probe-pysrc``, ``accept-gdel`` — and those three graphs are **100% ``"0001"``**;
no real tenant shares them. Their content is synthetic acceptance-test material
(``tests/test_simple.py`` / ``class TestSimple``, ``src/sample/__init__.py`` /
``def main``, dependency snapshots naming packages ``sample`` and ``package-data``
with garbage versions), written 2026-08-02..08-04. None of those three graph names
appears anywhere in this repository: they come from an external smoke/acceptance
harness.

``"0001"`` is therefore a **test-harness placeholder, not a tenant.** It must NOT
be mapped onto ``n8n-dev`` or any other real tenant — inventing an owner for 47
rows is exactly the "vouched for a tenant nobody chose" failure this service
already has a regression test about.

Disposition: **quarantine to NULL.** ``tenant_id`` is NULLABLE on all three
affected tables and 8,854 rows are already NULL, so NULL is the honest
"unattributed" — strictly more visible than a plausible-but-wrong uuid, and it
keeps the in-cluster ``alembic upgrade head`` initContainer green. Nothing is
lost: the original ``(table, row identity, old value)`` triples are copied into
the revision-owned :data:`QUARANTINE_TABLE` first, and :func:`downgrade` restores
from it, so the round trip is real rather than nominal.

The allowlist is an **explicit literal** (:data:`QUARANTINE_VALUES`). Any other
non-uuid value RAISES. A census that quietly widened its own allowlist would be a
census that can never fail.

THE CENSUS
----------
Two guards run before any ``ALTER``:

1. **Non-uuid values.** ``pg_input_is_valid(value, 'uuid')`` (PG16+; we run PG18)
   is used deliberately instead of a hand-written regex, because it is *exactly*
   the predicate ``::uuid`` will apply — a regex is either stricter (bricks the
   deploy on a legal non-canonical spelling) or looser (lets the ALTER fail with
   a bare 22P02 that names no row).

2. **Case-only collisions.** ``uuid`` normalises to lower case, so ``…A1B2`` and
   ``…a1b2`` are two distinct ``varchar`` values that become ONE ``uuid`` — which
   is a silent uniqueness collapse under ``uq_node_identity`` /
   ``uq_edge_identity`` / ``uq_dep_report_identity``. The test is
   ``GROUP BY lower(v) HAVING count(DISTINCT v) > 1``. It counts **variants, not
   rows**: the ``count(*) <> count(DISTINCT lower(v))`` form that looks equivalent
   counts ROWS and would refuse every healthy table on earth — 450,279 ``nodes``
   rows here legitimately share one tenant. ``tests/integration/test_tenant_uuid_census.py``
   carries that exact case as a CONTROL.

LOCK NOTE
---------
Prod runs ``alembic upgrade head`` in ONE transaction through PgBouncer txn-mode
(see the migrate initContainer in
``gitops-production/clusters/strato/services/coding-analysis/deployment.yaml``),
so this is plain blocking DDL under ACCESS EXCLUSIVE — no ``CONCURRENTLY``.
``ALTER COLUMN … TYPE`` rewrites the table and REBUILDS every dependent index,
which here means three unique indexes (``uq_node_identity``, ``uq_edge_identity``,
``uq_dep_report_identity``) plus ``idx_node_cg_tenant`` / ``idx_edge_cg_tenant``.

MEASURED, not estimated. A throwaway PostgreSQL 18.4 database was seeded at prod
scale — 470,000 ``nodes`` / 500,000 ``edges`` / 393 snapshots, one dominant tenant
(448,357 node rows), a NULL slice, four minority tenants and the 47 ``"0001"``
rows, with all fifteen indexes present including the three unique ones — and this
revision was run against it (two runs, agreeing):

    nodes                        470,000 rows    2.81s / 2.71s
    edges                        500,000 rows    4.11s / 4.22s
    dependency_report_snapshots      393 rows    0.00s
    bug_hunts / bug_hunt_findings      0 rows    0.00s
    ----------------------------------------------------
    ALTER wall-clock total                       6.93s / 6.94s
    whole `alembic upgrade` process              7.91s / 7.83s

So the ACCESS EXCLUSIVE window is single-digit seconds on hardware comparable to
worker4: a deploy-time stall, not an outage. ``downgrade -1`` measures the same
(7.53s at that scale, restoring all 47 rows).
"""

from __future__ import annotations

import logging
import time
from typing import NamedTuple

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

#: Revision-owned table holding the pre-quarantine ``(table, identity, value)``
#: triples. Created by :func:`upgrade`, read and dropped by :func:`downgrade`.
QUARANTINE_TABLE = "tenant_id_uuid_quarantine"

#: The ONLY non-uuid ``tenant_id`` values this migration is allowed to move out of
#: the way. Anything else that fails the census raises. Keep this an explicit set
#: of literals — a pattern here would let the next placeholder in unnoticed.
QUARANTINE_VALUES: frozenset[str] = frozenset({"0001"})


class TenantColumn(NamedTuple):
    """One ``tenant_id`` column, named literally.

    The table list is HARDCODED rather than derived from ``Base.metadata``. A
    metadata walk has twice broken ``alembic upgrade head`` from zero elsewhere in
    this fleet with ``operator does not exist: character varying = uuid``: the ORM
    already carries the POST-migration type, so a metadata-driven migration
    compares the new type against the old column and is invisible in prod (where
    the tables predate the ORM change) while being fatal in CI (where they do not).
    """

    table: str
    column: str
    #: Length to restore in :func:`downgrade`. Matches migrations 0001/0003/0004.
    varchar_length: int
    #: Whether the column is NULLABLE. Only a nullable column can be quarantined
    #: to NULL; a non-nullable one has no honest resting place, so a placeholder
    #: there RAISES via the census instead.
    nullable: bool
    #: The non-tenant columns of the row's identity, used to (a) prove nulling a
    #: quarantined row cannot collide with an existing NULL-tenant row under the
    #: ``NULLS NOT DISTINCT`` unique indexes and (b) address the row on restore.
    identity_columns: tuple[str, ...]


TABLES: tuple[TenantColumn, ...] = (
    TenantColumn("nodes", "tenant_id", 128, True, ("code_graph_id", "id")),
    TenantColumn(
        "edges", "tenant_id", 128, True, ("code_graph_id", "source", "target", "relationship")
    ),
    TenantColumn("dependency_report_snapshots", "tenant_id", 128, True, ("code_graph_id",)),
    # NOT NULL (migration 0004) and empty in prod. Listed so the census still
    # covers them; quarantine deliberately skips them.
    TenantColumn("bug_hunts", "tenant_id", 128, False, ("id",)),
    TenantColumn("bug_hunt_findings", "tenant_id", 128, False, ("id",)),
)

# Every identifier interpolated below comes from the module-level TABLES tuple —
# literal, developer-authored names. No request or row data reaches SQL text.


def _jsonb_identity(col: TenantColumn, alias: str = "") -> str:
    """``jsonb_build_object('c1', c1, 'c2', c2, …)`` over the identity columns."""
    prefix = f"{alias}." if alias else ""
    pairs = ", ".join(f"'{c}', {prefix}{c}" for c in col.identity_columns)
    return f"jsonb_build_object({pairs})"


def row_counts(bind: Connection) -> dict[str, int]:
    """Row count per table carrying a ``tenant_id``, for before/after logging."""
    return {
        col.table: bind.execute(sa.text(f"SELECT count(*) FROM {col.table}")).scalar_one()
        for col in TABLES
    }


def non_uuid_values(bind: Connection, col: TenantColumn) -> list[tuple[str, int]]:
    """``(value, row count)`` for every value ``::uuid`` would reject."""
    rows = bind.execute(
        sa.text(
            f"SELECT {col.column}::text AS value, count(*) AS n FROM {col.table} "
            f"WHERE {col.column} IS NOT NULL "
            f"  AND NOT pg_input_is_valid({col.column}::text, 'uuid') "
            f"GROUP BY 1 ORDER BY 2 DESC, 1"
        )
    ).all()
    return [(r.value, r.n) for r in rows]


def case_only_collisions(bind: Connection, col: TenantColumn) -> list[tuple[str, int]]:
    """``(folded value, distinct spellings)`` for values that ``uuid`` would merge.

    ``count(DISTINCT value)``, NOT ``count(*)``: the question is how many spellings
    exist, not how many rows carry them. A healthy table has exactly one spelling
    per folded value however many millions of rows share it.
    """
    rows = bind.execute(
        sa.text(
            f"SELECT lower({col.column}::text) AS folded, "
            f"       count(DISTINCT {col.column}::text) AS variants "
            f"FROM {col.table} WHERE {col.column} IS NOT NULL "
            f"GROUP BY 1 HAVING count(DISTINCT {col.column}::text) > 1 "
            f"ORDER BY 2 DESC, 1"
        )
    ).all()
    return [(r.folded, r.variants) for r in rows]


def census(bind: Connection) -> None:
    """Refuse the migration unless every remaining value survives ``::uuid`` intact.

    Raises :class:`RuntimeError` naming the offending values and their row counts.
    Deliberately fails CLOSED: a wrong tenant on a row is a cross-tenant data
    defect, which is strictly worse than a deploy that stops and says why.
    """
    offences: list[str] = []

    for col in TABLES:
        for value, count in non_uuid_values(bind, col):
            offences.append(
                f"{col.table}.{col.column}: {count} row(s) hold {value!r}, which is "
                f"not a uuid. It is not in QUARANTINE_VALUES ({sorted(QUARANTINE_VALUES)}), "
                f"so this migration will not guess an owner for it."
            )

    for col in TABLES:
        for folded, variants in case_only_collisions(bind, col):
            offences.append(
                f"{col.table}.{col.column}: {variants} distinct spellings fold to the "
                f"single uuid {folded!r}. Casting to uuid would merge them into one "
                f"value and silently collapse rows that are distinct today under the "
                f"NULLS NOT DISTINCT unique index. Normalise the spellings first."
            )

    if offences:
        raise RuntimeError(
            "0005 tenant_id->uuid census FAILED; no table was altered:\n  - "
            + "\n  - ".join(offences)
        )

    logger.info("0005 census: every tenant_id value casts to uuid, no case-only collisions")


def quarantine(bind: Connection, col: TenantColumn) -> int:
    """Copy allowlisted placeholder rows into the quarantine table, then NULL them.

    Returns the number of rows moved. A non-nullable column is skipped (there is
    nowhere honest to put the value; the census will refuse it instead).
    """
    if not col.nullable or not QUARANTINE_VALUES:
        return 0

    values = sorted(QUARANTINE_VALUES)
    in_clause = sa.bindparam("values", expanding=True)

    # Nulling a row makes it indistinguishable, under NULLS NOT DISTINCT, from an
    # existing NULL-tenant row with the same identity. Prove that cannot happen
    # BEFORE writing, so the failure names the rows instead of surfacing as a bare
    # 23505 from an index the reader has to go and look up.
    predicate = " AND ".join(f"n.{c} = q.{c}" for c in col.identity_columns)
    clashes = bind.execute(
        sa.text(
            f"SELECT {_jsonb_identity(col, 'q')}::text AS identity FROM {col.table} q "
            f"WHERE q.{col.column} IN :values "
            f"  AND EXISTS (SELECT 1 FROM {col.table} n "
            f"              WHERE n.{col.column} IS NULL AND {predicate}) "
            f"LIMIT 5"
        ).bindparams(in_clause),
        {"values": values},
    ).all()
    if clashes:
        raise RuntimeError(
            f"0005 quarantine refused on {col.table}: nulling the placeholder rows "
            f"would collide with existing NULL-tenant rows sharing the same identity "
            f"{col.identity_columns} (first offenders: {[c.identity for c in clashes]}). "
            f"Resolve those rows by hand; this migration will not delete data."
        )

    bind.execute(
        sa.text(
            f"INSERT INTO {QUARANTINE_TABLE} (table_name, row_identity, old_tenant_id) "
            f"SELECT :table, {_jsonb_identity(col)}, {col.column}::text "
            f"FROM {col.table} WHERE {col.column} IN :values"
        ).bindparams(in_clause),
        {"table": col.table, "values": values},
    )
    result = bind.execute(
        sa.text(
            f"UPDATE {col.table} SET {col.column} = NULL WHERE {col.column} IN :values"
        ).bindparams(in_clause),
        {"values": values},
    )
    return int(result.rowcount)


def restore_quarantined(bind: Connection, col: TenantColumn) -> int:
    """Put the quarantined placeholder values back (``downgrade`` half)."""
    if not col.nullable:
        return 0
    predicate = " AND ".join(f"t.{c}::text = q.row_identity->>'{c}'" for c in col.identity_columns)
    result = bind.execute(
        sa.text(
            f"UPDATE {col.table} AS t SET {col.column} = q.old_tenant_id "
            f"FROM {QUARANTINE_TABLE} q "
            f"WHERE q.table_name = :table AND t.{col.column} IS NULL AND {predicate}"
        ),
        {"table": col.table},
    )
    return int(result.rowcount)


def upgrade() -> None:
    bind = op.get_bind()

    before = row_counts(bind)
    logger.info("0005 tenant_id->uuid: row counts BEFORE %s", before)

    op.create_table(
        QUARANTINE_TABLE,
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        sa.Column("table_name", sa.Text(), nullable=False),
        sa.Column("row_identity", JSONB(), nullable=False),
        sa.Column("old_tenant_id", sa.Text(), nullable=False),
        sa.Column(
            "quarantined_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        comment=(
            "Owned by alembic revision 0005. Holds tenant_id values that were not "
            "uuids and were moved to NULL so the column could become uuid. "
            "downgrade() restores from here."
        ),
    )
    op.create_index(
        f"idx_{QUARANTINE_TABLE}_table", QUARANTINE_TABLE, ["table_name", "old_tenant_id"]
    )

    moved_total = 0
    for col in TABLES:
        moved = quarantine(bind, col)
        moved_total += moved
        if moved:
            logger.info(
                "0005 quarantine: %s.%s — %s row(s) holding %s copied to %s and set NULL",
                col.table,
                col.column,
                moved,
                sorted(QUARANTINE_VALUES),
                QUARANTINE_TABLE,
            )
    logger.info("0005 quarantine: %s row(s) moved in total", moved_total)

    census(bind)

    # Timed per table: ALTER … TYPE rewrites the heap AND rebuilds every dependent
    # index, and this is the entire ACCESS EXCLUSIVE window. Logging it means the
    # next person sizing a lock window reads a measurement instead of guessing.
    altered_total = 0.0
    for col in TABLES:
        started = time.perf_counter()
        op.execute(
            f"ALTER TABLE {col.table} ALTER COLUMN {col.column} TYPE uuid USING {col.column}::uuid"
        )
        elapsed = time.perf_counter() - started
        altered_total += elapsed
        logger.info(
            "0005 altered %s.%s -> uuid in %.2fs (%s rows)",
            col.table,
            col.column,
            elapsed,
            before[col.table],
        )
    logger.info("0005 ALTER wall-clock total: %.2fs", altered_total)

    after = row_counts(bind)
    logger.info("0005 tenant_id->uuid: row counts AFTER %s", after)
    if before != after:
        raise RuntimeError(
            f"0005 changed row counts (before={before} after={after}); this migration "
            f"only retypes a column and must never add or remove rows."
        )


def downgrade() -> None:
    bind = op.get_bind()

    for col in TABLES:
        op.execute(
            f"ALTER TABLE {col.table} ALTER COLUMN {col.column} "
            f"TYPE VARCHAR({col.varchar_length}) USING {col.column}::text"
        )

    restored_total = 0
    for col in TABLES:
        restored = restore_quarantined(bind, col)
        restored_total += restored
        if restored:
            logger.info(
                "0005 downgrade: restored %s quarantined row(s) on %s.%s",
                restored,
                col.table,
                col.column,
            )
    logger.info("0005 downgrade: %s row(s) restored in total", restored_total)

    op.drop_index(f"idx_{QUARANTINE_TABLE}_table", table_name=QUARANTINE_TABLE)
    op.drop_table(QUARANTINE_TABLE)
