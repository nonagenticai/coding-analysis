"""Persist dependency-report snapshots so the report degrades like the graph.

The fr "Dependency Report" is computed on-demand by coding-security, which execs
``find`` + ``cat`` inside the LIVE job container to read manifest files. When the
container Deployment is deprovisioned the report 404s — unlike the Architecture
graph, which reads persisted nodes/edges from THIS service's Postgres and
degrades gracefully (see ``graphs_reader.py``).

This table stores the last computed ``DependencyRiskReport`` JSON per
``(code_graph_id == container_name, tenant_id)``, written best-effort at
deep-analysis time (when the container is live, alongside ``populate_graph_database``),
so a reader endpoint can serve the stored snapshot after the container is gone.

Tenant-scoping mirrors ``nodes``/``edges`` (migration 0002): ``tenant_id`` stays
NULLABLE and the uniqueness lives in an index with ``NULLS NOT DISTINCT`` (PG15+,
we run PG18) so two NULL-tenant legacy writes for the same container collapse to
one idempotent row while two real tenants sharing a container name do not collide.

Lock note: prod runs ``alembic upgrade head`` in ONE transaction through PgBouncer
txn-mode, so this is plain blocking DDL. It creates a brand-new empty table, so the
ACCESS EXCLUSIVE window is sub-millisecond.
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dependency_report_snapshots",
        sa.Column("code_graph_id", sa.String(255), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=True),
        sa.Column("report", JSONB(), nullable=False),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "uq_dep_report_identity",
        "dependency_report_snapshots",
        ["code_graph_id", "tenant_id"],
        unique=True,
        postgresql_nulls_not_distinct=True,
    )


def downgrade() -> None:
    op.drop_index("uq_dep_report_identity", table_name="dependency_report_snapshots")
    op.drop_table("dependency_report_snapshots")
