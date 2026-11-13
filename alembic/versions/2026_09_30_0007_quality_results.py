"""Persist quality-analysis results per (tenant, container, commit).

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-30

WHY (coding-analysis#106)
-------------------------
fr's project Quality tab needs a STORED result. Before this table:

* ``GET /quality/{c}/latest|issues|summary|gate`` each ran a full
  ``analyzer.analyze()`` on every read, so viewing a tab re-executed every lint
  tool in the tenant's container and nothing was ever kept.
* ``POST /quality/analyze`` wrote only a Redis ``quality_task:{id}`` key (24h TTL).
* ``/{c}/trend`` read an aiosqlite cache that nothing wrote.

One row per analysis run. There is deliberately NO unique constraint on
``commit_sha``: re-analysing the same commit is a new row, and readers take the
newest. ``commit_sha`` is NULLABLE and holds a full 40-hex sha or nothing — it is
never substituted with a branch name, because a branch name is a moving pointer
and would make two different trees look like one measurement.

``issues`` is capped at write time (see ``services.quality_result_store``);
``issues_total`` / ``issues_truncated`` record what was dropped so a reader can
never mistake a truncated list for the whole result. ``issue_stats`` carries the
severity / per-file aggregates computed over the FULL issue list, so severity
counts and gate checks stay exact even when the list itself is truncated.

Lock note: brand-new empty table, so the ACCESS EXCLUSIVE window is trivial.
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "quality_results",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column("tenant_id", PG_UUID(as_uuid=False), nullable=False),
        sa.Column("container_name", sa.String(255), nullable=False),
        sa.Column("repo_url", sa.Text(), nullable=True),
        sa.Column("commit_sha", sa.String(40), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("overall_score", sa.Double(), nullable=True),
        sa.Column("grade", sa.String(16), nullable=True),
        sa.Column("language", sa.String(64), nullable=True),
        sa.Column("tools_attempted", JSONB(), server_default="[]", nullable=False),
        sa.Column("tools_succeeded", JSONB(), server_default="[]", nullable=False),
        sa.Column("tools_failed", JSONB(), server_default="[]", nullable=False),
        sa.Column("summary", JSONB(), server_default="{}", nullable=False),
        sa.Column("issue_stats", JSONB(), server_default="{}", nullable=False),
        sa.Column("issues", JSONB(), server_default="[]", nullable=False),
        sa.Column("issues_total", sa.Integer(), server_default="0", nullable=False),
        sa.Column("issues_truncated", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("recommendations", JSONB(), server_default="[]", nullable=False),
        sa.Column("execution_time", sa.Double(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "status IN ('complete', 'partial', 'inconclusive', 'failed')",
            name="ck_quality_results_status",
        ),
        sa.CheckConstraint(
            "commit_sha IS NULL OR commit_sha ~ '^[0-9a-f]{40}$'",
            name="ck_quality_results_commit_sha",
        ),
    )
    op.create_index(
        "idx_quality_results_tenant_container_created",
        "quality_results",
        ["tenant_id", "container_name", sa.text("created_at DESC")],
    )
    op.create_index(
        "idx_quality_results_tenant_repo_sha",
        "quality_results",
        ["tenant_id", "repo_url", "commit_sha"],
    )


def downgrade() -> None:
    op.drop_index("idx_quality_results_tenant_repo_sha", table_name="quality_results")
    op.drop_index("idx_quality_results_tenant_container_created", table_name="quality_results")
    op.drop_table("quality_results")
