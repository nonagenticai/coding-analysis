"""Bug hunts record whether the workspace's working tree differed from HEAD.

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-04

WHY
---
The bug hunt now runs INSIDE the tenant workspace (``services.workspace_hunt``)
and scans the COMMITTED tree at HEAD, never the working tree, so ``commit_sha``
always names exactly the bytes parsed. A workspace with uncommitted edits is
still scanned at HEAD; ``worktree_dirty`` records that it was dirty, so such a
hunt is visibly a HEAD scan and not a scan of what is on disk.

NULLABLE with NO server default, on purpose: NULL = not measured (pre-0010 rows,
log-only hunts). A default of false would claim a clean tree nobody measured.

Lock note: ADD COLUMN without a default is metadata-only on PostgreSQL.
"""

import sqlalchemy as sa

from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("bug_hunts", sa.Column("worktree_dirty", sa.Boolean(), nullable=True))


def downgrade() -> None:
    op.drop_column("bug_hunts", "worktree_dirty")
