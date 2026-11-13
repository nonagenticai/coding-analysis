"""Attribute the untenanted ``nodes``/``edges`` rows their own graph can identify.

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-02

WHY
---
``tenant_id`` became a real ``uuid`` in 0005, but the type says nothing about the
rows that hold no value at all. Measured on prod ``coding_analysis`` @ worker4
immediately after 0005 landed:

    nodes                        471,946 rows      7,759 NULL
    edges                        501,965 rows      1,139 NULL
    dependency_report_snapshots      397 rows          3 NULL
    ------------------------------------------------------------
                                                  8,901 unattributed

A uuid column with NULL data passes every type-level check while remaining
unattributed, and in THIS service unattributed does not mean invisible — it means
the opposite. Three read sites are deliberately NULL-tolerant
(``tenant_id = %s OR tenant_id IS NULL``, added by PR #54 so the
``architecture_refactoring`` cycle-detection arc did not regress on legacy
graphs):

  * ``src/processing/architecture/graph_adapter.py`` ``_get_all_entities``
  * ``src/processing/architecture/graph_adapter.py`` ``_get_all_relationships``
  * ``src/services/code_graph/graph_builder.py`` ``graph_is_populated``

plus every ``code_graph_id``-only read in ``graphs_reader.py`` / ``graph_metrics.py``,
which does not mention ``tenant_id`` at all. So a NULL row is not hidden from
tenants; it is served to **every** tenant that clears the container-name ownership
check. NULL is currently behaving as a wildcard, which is precisely the shape this
estate has been burned by: "all tenants" living in the identity column as the
ABSENCE of an id rather than in an explicit scope column.

Reducing that population is therefore an isolation fix, not bookkeeping.

THE RULE THIS MIGRATION APPLIES
-------------------------------
A graph's owner is recorded on the graph itself. ``graphs_reader.initialize_graph``
writes one ``type='repository'`` anchor node per graph, ``id = 'repo_<cg>'``, behind
``tenant_id: str = Depends(require_tenant)`` — the ONE writer in this service that
cannot produce a NULL, because ``require_tenant`` raises 401 rather than returning
None. Every other writer either omits the column or binds a ``None`` default.

So: **the tenant on a graph's repository anchor is that graph's tenant**, and the
NULL rows in the same ``code_graph_id`` are that tenant's rows written by one of the
tenant-blind writers.

That is a claim about data, so it was tested against data rather than assumed. Of
the 22 repository anchors in prod, 17 sit on graphs that ALSO hold tenant-bearing
non-anchor rows — i.e. 17 cases where the rule can be checked against a value that
was written independently by a different code path:

    anchor tenant == bulk-row tenant     17 / 17     (0 counterexamples)

Two graphs carry the anchor and nothing else attributed:

    fr-test-verification   6,543 NULL nodes +  64 NULL edges  anchor 0195f48b-…-0001
    arch-proof4              133 NULL nodes + 117 NULL edges  anchor 0195f48b-…-0001

``arch-proof4``'s anchor was written 7 seconds before its NULL bulk rows;
``fr-test-verification``'s 78 minutes before. No second tenant has ever written to
either ``code_graph_id``.

WHAT IS DELIBERATELY LEFT ALONE
-------------------------------
2,044 rows have no anchor and are NOT touched:

  * 1,997 rows across nine ``tour-1803*`` graphs (1,059 nodes / 938 edges), written
    2026-06-22..06-23. No repository anchor, no tenant-bearing row in ANY table of
    this database, and no match in the ``coding`` database either. Their owner is
    not recoverable from the data.
  * 47 rows (``accept-graph``, ``probe-pysrc``, ``accept-gdel``) whose ``tenant_id``
    was the literal ``"0001"`` until 0005 quarantined it. 0005 refused to guess an
    owner for those on purpose; this revision does not overturn that decision. The
    originals remain in ``tenant_id_uuid_quarantine``.

Inventing a tenant for a row whose owner cannot be determined is worse than an
honest NULL, so those 2,044 stay NULL and are counted, named and logged instead.

WHY NO ``SET NOT NULL``
-----------------------
Because it cannot be honoured. ``NOT NULL`` on a column that 2,044 live rows cannot
populate would either fail the deploy or force a fabricated value into every one of
them. The constraint is only correct once the writers below are fixed AND the
residual rows are dispositioned; both are open decisions, recorded in
``tests/unit/test_tenant_writer_ratchet.py`` rather than pre-empted here.

WHY THE WRITERS ARE NOT FIXED IN THIS REVISION
----------------------------------------------
The rows keep arriving from writers that have no tenant to write. ``INSERT INTO
nodes`` in ``src/processing/task_result_processor.py`` and
``src/services/code_graph/graph_database_populator.py`` omit the column entirely;
``db/facade.py``, ``db/entity_mapper.py`` and
``services/incremental/incremental_update_processor.py`` all bind a ``tenant_id:
str | None = None`` default that ten of their eleven construction sites leave
unset. The reason is structural, not an oversight: those writers run on the task
path (``analyze_codebase`` / ``multi_language_analysis`` in
``src/processing/task_processors.py``), and the task payload carries **no tenant at
all** — ``grep -c tenant src/processing/task_processors.py`` is 0. Threading one in
is an interface change to the task system, which is an owner decision and not the
implicit reading of "backfill the untenanted rows". The ratchet test added
alongside this revision pins the offenders so a tenth tenant-blind writer cannot be
added quietly in the meantime.

REVERSIBILITY
-------------
Every row this revision changes is recorded in :data:`BACKFILL_TABLE` first, keyed
by its identity columns AND the tenant assigned, so :func:`downgrade` restores
exactly those rows to NULL and touches nothing an unrelated writer has since
attributed. The round trip is real, not nominal.

FAIL-CLOSED GUARDS
------------------
Both run before any ``UPDATE`` and raise rather than guess:

1. **Ambiguous anchor.** A graph whose anchors disagree on the tenant has no single
   answer. ``count(DISTINCT tenant_id) > 1`` over its ``type='repository'`` rows
   raises. (Zero such graphs in prod; the guard exists for the graph that gets one.)
2. **Identity collision.** Setting ``tenant_id`` moves a row within the
   ``NULLS NOT DISTINCT`` unique indexes ``uq_node_identity`` / ``uq_edge_identity``
   / ``uq_dep_report_identity``. If the destination identity is already occupied,
   the ``UPDATE`` would abort mid-migration with a bare 23505 naming no row, so the
   collision is detected first and named. (Zero such rows in prod, measured.)

MEASURED EFFECT (prod scale, at revision 0005)
----------------------------------------------
    nodes   6,676 of 7,759 NULL attributed   (1,083 remain: tour-1803*, accept-*)
    edges     181 of 1,139 NULL attributed   (  958 remain)
    dependency_report_snapshots  0 of 3      (    3 remain: the 0005 quarantine)
    -----------------------------------------------------------------------------
    6,857 of 8,901 attributed (77%);  2,044 remain honestly unattributed

The statements are two indexed ``UPDATE``s over a few thousand rows — no table
rewrite, no index rebuild, unlike 0005. Sub-second at this scale.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.engine import Connection

from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

#: Revision-owned table holding one ``(table, identity, assigned tenant)`` triple per
#: row this migration attributes. Created by :func:`upgrade`; read and dropped by
#: :func:`downgrade`. Separate from 0005's ``tenant_id_uuid_quarantine`` on purpose —
#: that table is 0005's record of a DIFFERENT decision and must not be co-mingled.
BACKFILL_TABLE = "tenant_anchor_backfill"

#: The node ``type`` that carries a graph's owning tenant. Written only by
#: ``graphs_reader.initialize_graph`` under ``Depends(require_tenant)``.
ANCHOR_TYPE = "repository"

#: The table the anchor lives in. Named once so the join below reads honestly:
#: ``edges`` and ``dependency_report_snapshots`` are attributed from ``nodes``.
ANCHOR_TABLE = "nodes"


class TenantColumn(NamedTuple):
    """One ``tenant_id`` column this revision may attribute.

    HARDCODED rather than derived from ``Base.metadata``, for the same reason 0005
    spelled out: a metadata walk carries the CURRENT ORM type and shape, so it is
    invisible against a prod table that predates the model and fatal on a from-zero
    CI run.
    """

    table: str
    #: The non-tenant columns of the row's identity — the unique index minus
    #: ``tenant_id``. Used to detect collisions and to address rows on restore.
    identity_columns: tuple[str, ...]
    #: The ``NULLS NOT DISTINCT`` unique index those columns belong to. Named
    #: literally rather than derived from ``table``: the snapshots index is
    #: ``uq_dep_report_identity``, which no rule over the table name produces.
    unique_index: str


TABLES: tuple[TenantColumn, ...] = (
    TenantColumn("nodes", ("code_graph_id", "id"), "uq_node_identity"),
    TenantColumn(
        "edges",
        ("code_graph_id", "source", "target", "relationship"),
        "uq_edge_identity",
    ),
    # Applies the same rule for uniformity; resolves 0 rows in prod because the
    # three NULL snapshots belong to graphs that have no repository anchor.
    TenantColumn("dependency_report_snapshots", ("code_graph_id",), "uq_dep_report_identity"),
)

# Every identifier interpolated into SQL below comes from the module-level TABLES
# tuple and the ANCHOR_* constants — literal, developer-authored names. No request
# data and no row data reaches SQL text.


def _jsonb_identity(col: TenantColumn, alias: str = "") -> str:
    """``jsonb_build_object('c1', c1, …)`` over ``col``'s identity columns."""
    prefix = f"{alias}." if alias else ""
    pairs = ", ".join(f"'{c}', {prefix}{c}" for c in col.identity_columns)
    return f"jsonb_build_object({pairs})"


def _anchor_cte() -> str:
    """One row per graph that has exactly one attributed repository anchor.

    ``count(DISTINCT tenant_id) = 1`` is the whole rule: a graph whose anchors
    disagree yields no row here and is additionally refused by
    :func:`ambiguous_anchors`. Aggregating at all is safe only because of that
    predicate — over one distinct value the aggregate IS that value.

    ``min(tenant_id::text)::uuid``, not ``min(tenant_id)``: PostgreSQL has **no
    ``min(uuid)``** (`function min(uuid) does not exist`, measured on PG18.6 by
    the first run of this revision). 0005 hit the identical wall in
    ``graphs_reader.list_graphs`` and fixed it the same way. Since 0005 made the
    column a real ``uuid``, the round trip through ``text`` is lossless: the type
    normalises the spelling on the way out, so ordering is over canonical forms.
    """
    return (
        f"SELECT code_graph_id, min(tenant_id::text)::uuid AS tenant_id "
        f"FROM {ANCHOR_TABLE} "
        f"WHERE type = '{ANCHOR_TYPE}' AND tenant_id IS NOT NULL "
        f"GROUP BY code_graph_id "
        f"HAVING count(DISTINCT tenant_id) = 1"
    )


def untenanted_counts(bind: Connection) -> dict[str, int]:
    """Rows still holding a NULL ``tenant_id``, per table."""
    return {
        col.table: bind.execute(
            sa.text(f"SELECT count(*) FROM {col.table} WHERE tenant_id IS NULL")
        ).scalar_one()
        for col in TABLES
    }


def ambiguous_anchors(bind: Connection) -> list[tuple[str, int]]:
    """``(code_graph_id, distinct anchor tenants)`` for graphs whose anchors disagree.

    A graph with two repository nodes naming two tenants has no single owner, so
    there is nothing to backfill FROM. Refuse rather than let ``min()`` pick.
    """
    rows = bind.execute(
        sa.text(
            f"SELECT code_graph_id, count(DISTINCT tenant_id) AS tenants "
            f"FROM {ANCHOR_TABLE} "
            f"WHERE type = '{ANCHOR_TYPE}' AND tenant_id IS NOT NULL "
            f"GROUP BY code_graph_id HAVING count(DISTINCT tenant_id) > 1 "
            f"ORDER BY 2 DESC, 1"
        )
    ).all()
    return [(r.code_graph_id, r.tenants) for r in rows]


def identity_collisions(bind: Connection, col: TenantColumn) -> list[str]:
    """Identities that already exist under the tenant this revision would assign.

    Attributing a row MOVES it inside ``uq_*_identity`` (``NULLS NOT DISTINCT``), so
    a row already sitting at the destination identity turns the ``UPDATE`` into a
    23505 that names nothing. Detect and name it here instead.
    """
    predicate = " AND ".join(f"occupied.{c} = target.{c}" for c in col.identity_columns)
    rows = bind.execute(
        sa.text(
            f"WITH anchor AS ({_anchor_cte()}) "
            f"SELECT {_jsonb_identity(col, 'target')}::text AS identity "
            f"FROM {col.table} target "
            f"JOIN anchor ON anchor.code_graph_id = target.code_graph_id "
            f"WHERE target.tenant_id IS NULL "
            f"  AND EXISTS (SELECT 1 FROM {col.table} occupied "
            f"              WHERE occupied.tenant_id = anchor.tenant_id AND {predicate}) "
            f"LIMIT 5"
        )
    ).all()
    return [r.identity for r in rows]


def guard(bind: Connection) -> None:
    """Refuse the migration unless every row it would touch has ONE safe answer.

    Fails CLOSED: a wrong tenant on a row is a cross-tenant data defect, which is
    strictly worse than a deploy that stops and says why.
    """
    offences: list[str] = []

    for code_graph_id, tenants in ambiguous_anchors(bind):
        offences.append(
            f"{ANCHOR_TABLE}: graph {code_graph_id!r} has {tenants} distinct "
            f"{ANCHOR_TYPE!r}-node tenants. A graph with more than one owner has no "
            f"single answer, so this migration will not pick one for it."
        )

    for col in TABLES:
        for identity in identity_collisions(bind, col):
            offences.append(
                f"{col.table}: attributing {identity} to its graph's anchor tenant "
                f"would collide with a row that already holds that identity under "
                f"{col.unique_index} (NULLS NOT DISTINCT)."
            )

    if offences:
        raise RuntimeError(
            "Refusing to attribute rows from the repository anchor:\n  - " + "\n  - ".join(offences)
        )


def upgrade() -> None:
    """Attribute every NULL-tenant row whose graph names exactly one owner."""
    bind = op.get_bind()

    before = untenanted_counts(bind)
    logger.info("0006 anchor backfill: untenanted before = %s", before)

    guard(bind)

    op.create_table(
        BACKFILL_TABLE,
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        sa.Column("table_name", sa.Text(), nullable=False),
        sa.Column("row_identity", JSONB(), nullable=False),
        sa.Column("assigned_tenant_id", UUID(as_uuid=False), nullable=False),
        sa.Column(
            "backfilled_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        f"idx_{BACKFILL_TABLE}_table", BACKFILL_TABLE, ["table_name", "assigned_tenant_id"]
    )

    attributed: dict[str, int] = {}
    for col in TABLES:
        # Record BEFORE the update, so downgrade can address exactly these rows.
        bind.execute(
            sa.text(
                f"WITH anchor AS ({_anchor_cte()}) "
                f"INSERT INTO {BACKFILL_TABLE} "
                f"       (table_name, row_identity, assigned_tenant_id) "
                f"SELECT :table, {_jsonb_identity(col, 'target')}, anchor.tenant_id "
                f"FROM {col.table} target "
                f"JOIN anchor ON anchor.code_graph_id = target.code_graph_id "
                f"WHERE target.tenant_id IS NULL"
            ),
            {"table": col.table},
        )
        result = bind.execute(
            sa.text(
                f"WITH anchor AS ({_anchor_cte()}) "
                f"UPDATE {col.table} AS target SET tenant_id = anchor.tenant_id "
                f"FROM anchor "
                f"WHERE anchor.code_graph_id = target.code_graph_id "
                f"  AND target.tenant_id IS NULL"
            )
        )
        attributed[col.table] = result.rowcount

    after = untenanted_counts(bind)
    residual = sum(after.values())
    logger.info(
        "0006 anchor backfill: attributed %s; untenanted after = %s (%d rows remain "
        "unattributed and stay NULL — no repository anchor names their owner, so "
        "this revision does not invent one, and tenant_id is NOT made NOT NULL)",
        attributed,
        after,
        residual,
    )

    # Named explicitly so a reader of the deploy log sees the residual is a KNOWN
    # population and not a silent partial success.
    for col in TABLES:
        if after[col.table]:
            graphs = bind.execute(
                sa.text(
                    f"SELECT code_graph_id, count(*) AS n FROM {col.table} "
                    f"WHERE tenant_id IS NULL GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 20"
                )
            ).all()
            logger.info(
                "0006 residual %s: %s",
                col.table,
                ", ".join(f"{g.code_graph_id}={g.n}" for g in graphs),
            )


def downgrade() -> None:
    """Restore exactly the rows :func:`upgrade` attributed back to NULL.

    Addressed by identity AND the tenant this revision assigned, so a row some other
    writer has attributed differently since is left alone.
    """
    bind = op.get_bind()

    restored: dict[str, int] = {}
    for col in TABLES:
        predicate = " AND ".join(
            f"target.{c} = (record.row_identity ->> '{c}')" for c in col.identity_columns
        )
        result = bind.execute(
            sa.text(
                f"UPDATE {col.table} AS target SET tenant_id = NULL "
                f"FROM {BACKFILL_TABLE} AS record "
                f"WHERE record.table_name = :table "
                f"  AND target.tenant_id = record.assigned_tenant_id::uuid "
                f"  AND {predicate}"
            ),
            {"table": col.table},
        )
        restored[col.table] = result.rowcount

    logger.info("0006 downgrade: restored to NULL = %s", restored)

    op.drop_index(f"idx_{BACKFILL_TABLE}_table", table_name=BACKFILL_TABLE)
    op.drop_table(BACKFILL_TABLE)
