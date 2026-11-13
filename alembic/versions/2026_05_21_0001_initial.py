"""Initial analysis-owned code-graph schema.

Revision ID: 0001
Revises: None
Create Date: 2026-05-21

Creates: nodes, edges, graph_versions, analysis_sessions — all scoped by
code_graph_id. Matches the ORM in src/db/models.py exactly.
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- nodes ----------------------------------------------------------------
    op.create_table(
        "nodes",
        sa.Column("code_graph_id", sa.String(255), primary_key=True, nullable=False),
        sa.Column("id", sa.String(512), primary_key=True, nullable=False),
        sa.Column("type", sa.String(64), nullable=False),
        sa.Column("properties", JSONB(), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("tenant_id", sa.String(128), nullable=True),
        sa.Column("language", sa.String(32), nullable=True, server_default="python"),
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("line_number", sa.Integer(), nullable=True),
        sa.Column("column_start", sa.Integer(), nullable=True),
        sa.Column("column_end", sa.Integer(), nullable=True),
        sa.Column("visibility", sa.String(32), nullable=True),
        sa.Column("is_static", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("is_abstract", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("is_final", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("is_async", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("signature", sa.Text(), nullable=True),
        sa.Column("docstring", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("idx_node_cg", "nodes", ["code_graph_id"])
    op.create_index("idx_node_cg_type", "nodes", ["code_graph_id", "type"])
    op.create_index("idx_node_cg_content_hash", "nodes", ["code_graph_id", "content_hash"])
    op.create_index("idx_node_cg_tenant", "nodes", ["code_graph_id", "tenant_id"])
    op.create_index("idx_node_cg_language", "nodes", ["code_graph_id", "language"])
    op.create_index("idx_node_cg_file_path", "nodes", ["code_graph_id", "file_path"])

    # -- edges ----------------------------------------------------------------
    op.create_table(
        "edges",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=True), primary_key=True),
        sa.Column("code_graph_id", sa.String(255), nullable=False),
        sa.Column("source", sa.String(512), nullable=False),
        sa.Column("target", sa.String(512), nullable=False),
        sa.Column("relationship", sa.String(64), nullable=False),
        sa.Column("properties", JSONB(), nullable=True),
        sa.Column("weight", sa.Double(), nullable=False, server_default="1.0"),
        sa.Column("tenant_id", sa.String(128), nullable=True),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column("location_file", sa.Text(), nullable=True),
        sa.Column("location_line", sa.Integer(), nullable=True),
        sa.Column("location_column", sa.Integer(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["code_graph_id", "source"],
            ["nodes.code_graph_id", "nodes.id"],
            ondelete="CASCADE",
            name="fk_edge_source_node",
        ),
        sa.ForeignKeyConstraint(
            ["code_graph_id", "target"],
            ["nodes.code_graph_id", "nodes.id"],
            ondelete="CASCADE",
            name="fk_edge_target_node",
        ),
        sa.UniqueConstraint(
            "code_graph_id",
            "source",
            "target",
            "relationship",
            name="uq_edge_cg_source_target_rel",
        ),
    )
    op.create_index("idx_edge_cg", "edges", ["code_graph_id"])
    op.create_index("idx_edge_cg_source", "edges", ["code_graph_id", "source"])
    op.create_index("idx_edge_cg_target", "edges", ["code_graph_id", "target"])
    op.create_index("idx_edge_cg_relationship", "edges", ["code_graph_id", "relationship"])
    op.create_index("idx_edge_cg_tenant", "edges", ["code_graph_id", "tenant_id"])

    # -- graph_versions -------------------------------------------------------
    op.create_table(
        "graph_versions",
        sa.Column("code_graph_id", sa.String(255), primary_key=True, nullable=False),
        sa.Column("id", sa.String(255), primary_key=True, nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index(
        "idx_graph_version_cg_created", "graph_versions", ["code_graph_id", "created_at"]
    )

    # -- analysis_sessions ----------------------------------------------------
    op.create_table(
        "analysis_sessions",
        sa.Column("code_graph_id", sa.String(255), primary_key=True, nullable=False),
        sa.Column("id", sa.String(255), primary_key=True, nullable=False),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("languages", JSONB(), nullable=True),
        sa.Column("files_processed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("files_failed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_entities", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_relationships", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(32), nullable=False, server_default="running"),
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_index("idx_session_cg_status", "analysis_sessions", ["code_graph_id", "status"])


def downgrade() -> None:
    op.drop_table("analysis_sessions")
    op.drop_table("graph_versions")
    op.drop_table("edges")
    op.drop_table("nodes")
