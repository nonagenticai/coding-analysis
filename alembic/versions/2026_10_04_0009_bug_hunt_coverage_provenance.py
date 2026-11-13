"""Bug hunts record coverage provenance: scanned/tracked counts, commit sha, scope.

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-04

WHY
---
fr's ``bug_report/v1`` (fr ``backend/src/lib/bug-report-hash.ts``) stores a hunt
only when ``hunt.scanned_files == hunt.tracked_source_files`` and the report
names a ``commit_sha``. The summary report is served from ``bug_hunts``, so the
three values measured during the hunt (``services.hunt_coverage``) must be
persisted with the job.

``excluded_dirs`` (text[]) is the DECLARED scope policy of the run: the
directory names excluded from BOTH counts. A narrowed scope is allowed only when
it is declared, so it is stored beside the counts it qualifies.

All four are NULLABLE with NO server default, on purpose: NULL is "not
measured" (pre-0009 rows, log-only hunts, no git, dirty tree), which a consumer
must refuse. A default of 0 would read as a measurement of an empty repository.

Lock note: ADD COLUMN without a default is metadata-only on PostgreSQL.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("bug_hunts", sa.Column("scanned_files", sa.Integer(), nullable=True))
    op.add_column("bug_hunts", sa.Column("tracked_source_files", sa.Integer(), nullable=True))
    op.add_column("bug_hunts", sa.Column("commit_sha", sa.String(40), nullable=True))
    op.add_column(
        "bug_hunts",
        sa.Column("excluded_dirs", postgresql.ARRAY(sa.Text()), nullable=True),
    )
    op.create_check_constraint(
        "ck_bug_hunts_scanned_files_nonneg", "bug_hunts", "scanned_files >= 0"
    )
    op.create_check_constraint(
        "ck_bug_hunts_tracked_source_files_nonneg", "bug_hunts", "tracked_source_files >= 0"
    )
    op.create_check_constraint(
        "ck_bug_hunts_commit_sha_full_hex",
        "bug_hunts",
        "commit_sha IS NULL OR commit_sha ~ '^[0-9a-f]{40}$'",
    )


def downgrade() -> None:
    op.drop_constraint("ck_bug_hunts_commit_sha_full_hex", "bug_hunts", type_="check")
    op.drop_constraint("ck_bug_hunts_tracked_source_files_nonneg", "bug_hunts", type_="check")
    op.drop_constraint("ck_bug_hunts_scanned_files_nonneg", "bug_hunts", type_="check")
    op.drop_column("bug_hunts", "excluded_dirs")
    op.drop_column("bug_hunts", "commit_sha")
    op.drop_column("bug_hunts", "tracked_source_files")
    op.drop_column("bug_hunts", "scanned_files")
