"""Bug-hunt finding severity defaults to ``unrated``, not ``medium``.

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-04

WHY (coding-analysis#112)
-------------------------
The bug hunter has nothing to rate a finding from: ``ParsedError`` carries no
severity. Its endpoint emitted a literal ``"medium"``, the store fell back to
``"medium"``, and this column's server default was ``"medium"``, so every row
read ``medium``. It was a constant presented as data. Any filter or count over
it would look like it worked while distinguishing nothing.

This revision:

* sets the server default to ``unrated``, so a row written without a severity
  says so;
* rewrites existing ``medium`` rows to ``unrated``. Every such row came from the
  constant, because no code path has ever computed a bug-hunt severity, so the
  rewrite restores what was really known rather than discarding information.
  Measured on prod 2026-10-04 (worker4 `coding_analysis`, alembic 0007): 0 rows.

Downgrade restores the ``medium`` default only. It does NOT rewrite ``unrated``
back to ``medium``, because that would re-introduce the false value.

Lock note: ALTER COLUMN ... SET DEFAULT is metadata-only. The UPDATE touches
only rows still holding the constant.
"""

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("bug_hunt_findings", "severity", server_default="unrated")
    op.execute("UPDATE bug_hunt_findings SET severity = 'unrated' WHERE severity = 'medium'")


def downgrade() -> None:
    op.alter_column("bug_hunt_findings", "severity", server_default="medium")
