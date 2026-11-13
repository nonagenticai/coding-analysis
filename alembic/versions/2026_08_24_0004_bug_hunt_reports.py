"""Persist asynchronous bug hunts and path-queryable findings.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-24

The report is deliberately normalized: hunt-level summaries live in
``bug_hunts`` and every detected bug lives in ``bug_hunt_findings``.  This
keeps a large report out of one cache value and lets the API query one file or
directory subtree without deserializing every finding.
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "bug_hunts",
        sa.Column("id", sa.String(36), primary_key=True, nullable=False),
        sa.Column("code_graph_id", sa.String(255), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), server_default="queued", nullable=False),
        sa.Column("request_payload", JSONB(), nullable=True),
        sa.Column("errors_found", sa.Integer(), server_default="0", nullable=False),
        sa.Column("fixes_generated", sa.Integer(), server_default="0", nullable=False),
        sa.Column("root_cause_analysis", sa.Text(), nullable=True),
        sa.Column("fix_summary", sa.Text(), nullable=True),
        sa.Column("estimated_effort", sa.Text(), nullable=True),
        sa.Column("risk_assessment", sa.Text(), nullable=True),
        sa.Column("analysis_time", sa.Double(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("worker_id", sa.String(64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index(
        "idx_bug_hunt_tenant_container_created",
        "bug_hunts",
        ["tenant_id", "code_graph_id", "created_at"],
    )
    op.create_index("idx_bug_hunt_status_created", "bug_hunts", ["status", "created_at"])
    op.create_index(
        "idx_bug_hunt_claim",
        "bug_hunts",
        ["status", "lease_expires_at", "created_at"],
    )

    op.create_table(
        "bug_hunt_findings",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        sa.Column(
            "hunt_id",
            sa.String(36),
            sa.ForeignKey("bug_hunts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("code_graph_id", sa.String(255), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("file_name", sa.Text(), nullable=False),
        sa.Column("line_number", sa.Integer(), server_default="0", nullable=False),
        sa.Column("severity", sa.String(32), server_default="medium", nullable=False),
        sa.Column("finding", JSONB(), nullable=False),
        sa.Column("suggested_fixes", JSONB(), server_default="[]", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index(
        "idx_bug_finding_hunt_path",
        "bug_hunt_findings",
        ["hunt_id", "tenant_id", "file_path"],
        postgresql_ops={"file_path": "text_pattern_ops"},
    )


def downgrade() -> None:
    op.drop_index("idx_bug_finding_hunt_path", table_name="bug_hunt_findings")
    op.drop_table("bug_hunt_findings")
    op.drop_index("idx_bug_hunt_claim", table_name="bug_hunts")
    op.drop_index("idx_bug_hunt_status_created", table_name="bug_hunts")
    op.drop_index("idx_bug_hunt_tenant_container_created", table_name="bug_hunts")
    op.drop_table("bug_hunts")
