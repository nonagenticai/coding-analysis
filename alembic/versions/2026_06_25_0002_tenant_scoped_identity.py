"""Make tenant_id part of the code-graph node/edge identity.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-25

Closes the WRITE half of the cross-tenant graph isolation defect (the READ half
shipped in PR #54). ``code_graph_id == container_name`` is a namespace SHARED
across tenants, so the old node PK ``(code_graph_id, id)`` and edge UNIQUE
``(code_graph_id, source, target, relationship)`` made two tenants with the same
container name collide on write — tenant-B's rows hit ``ON CONFLICT ... DO
NOTHING`` and were silently dropped.

Fix: replace those keys with unique indexes that carry ``tenant_id`` and use
``NULLS NOT DISTINCT`` (PG15+, we are on PG18). ``tenant_id`` stays NULLABLE so:
  * two NULL-tenant *legacy* rows for the same ``(cg, id)`` still collapse to one
    (idempotent re-ingest; the NULL-tolerant reads from PR #54 keep working, so
    the ``architecture_refactoring`` cycle-detection arc does not regress), and
  * two real tenants sharing a ``code_graph_id`` no longer collide.

The two edge foreign keys are DROPPED (not re-pointed): re-pointing to
``(cg, tenant_id, source/target)`` is MATCH SIMPLE (skips every NULL-tenant row)
and would FK-violate when a tenant writes an edge against a legacy NULL-tenant
node. The application already orders ``DELETE FROM edges`` before ``DELETE FROM
nodes`` and orphan-checks before inserting edges, so the FKs guarded nothing the
app does not already guarantee.

Lock note: prod runs ``alembic upgrade head`` in ONE transaction through
PgBouncer txn-mode, so ``CREATE INDEX CONCURRENTLY`` is impossible — this is
plain blocking DDL. The per-graph node/edge tables are small (thousands to low
millions of rows) so the ACCESS EXCLUSIVE window is sub-second to seconds. If a
future deployment has very large tables, run this out-of-band over a DIRECT
``postgres:5432`` session with ``CREATE UNIQUE INDEX CONCURRENTLY`` +
a fast transactional swap instead.
"""

import logging

import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")


def _log_table_sizes() -> None:
    """Log node/edge row counts so the blocking-DDL lock window is observable."""
    bind = op.get_bind()
    nodes = bind.execute(sa.text("SELECT count(*) FROM nodes")).scalar_one()
    edges = bind.execute(sa.text("SELECT count(*) FROM edges")).scalar_one()
    logger.info(
        "0002 tenant-identity: nodes=%s edges=%s — plain blocking DDL "
        "(ACCESS EXCLUSIVE). If these are very large, run out-of-band with "
        "CREATE UNIQUE INDEX CONCURRENTLY instead.",
        nodes,
        edges,
    )


def upgrade() -> None:
    _log_table_sizes()

    # 1. Drop the edge FKs first — they depend on the node (code_graph_id, id) PK.
    op.drop_constraint("fk_edge_source_node", "edges", type_="foreignkey")
    op.drop_constraint("fk_edge_target_node", "edges", type_="foreignkey")

    # 2. Node identity -> (code_graph_id, tenant_id, id), NULL tenants collapse.
    op.drop_constraint("nodes_pkey", "nodes", type_="primary")
    op.create_index(
        "uq_node_identity",
        "nodes",
        ["code_graph_id", "tenant_id", "id"],
        unique=True,
        postgresql_nulls_not_distinct=True,
    )

    # 3. Edge identity -> (code_graph_id, tenant_id, source, target, relationship).
    op.drop_constraint("uq_edge_cg_source_target_rel", "edges", type_="unique")
    op.create_index(
        "uq_edge_identity",
        "edges",
        ["code_graph_id", "tenant_id", "source", "target", "relationship"],
        unique=True,
        postgresql_nulls_not_distinct=True,
    )


def downgrade() -> None:
    # Best-effort reverse. Only safe BEFORE multi-tenant rows exist: recreating
    # the (code_graph_id, id) PK / the (cg, source, target, rel) UNIQUE will fail
    # on the very duplicate-across-tenant rows this migration was created to
    # allow. CI runs ``downgrade base`` on empty data; prod is forward-only.
    op.drop_index("uq_edge_identity", table_name="edges")
    op.create_unique_constraint(
        "uq_edge_cg_source_target_rel",
        "edges",
        ["code_graph_id", "source", "target", "relationship"],
    )

    op.drop_index("uq_node_identity", table_name="nodes")
    op.create_primary_key("nodes_pkey", "nodes", ["code_graph_id", "id"])

    op.create_foreign_key(
        "fk_edge_source_node",
        "edges",
        "nodes",
        ["code_graph_id", "source"],
        ["code_graph_id", "id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_edge_target_node",
        "edges",
        "nodes",
        ["code_graph_id", "target"],
        ["code_graph_id", "id"],
        ondelete="CASCADE",
    )
