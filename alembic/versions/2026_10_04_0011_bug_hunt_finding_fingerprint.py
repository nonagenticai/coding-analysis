"""Finding fingerprint; hunt agent_sha and excluded_entry_kinds.

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-04

WHY
---
A finding is rechecked by re-running the detector on its file at the hunt's
``commit_sha`` (``POST /api/bug-hunter/{c}/hunts/{h}/findings/{f}/recheck``).
"Is it still there?" needs an identity that does not move when unrelated lines
move, and ``bug_hunt_findings.id`` is a row id, not an identity. The agent now
emits ``fingerprint`` (``hunt_agent.hunt.finding_fingerprint``: error type +
path + whitespace-normalised flagged line + same-anchor ordinal; no line
number) and it is stored here.

``bug_hunts.agent_sha`` records which detector produced the findings. A recheck
by a different agent reproducing nothing would not be a refutation, so the
recheck refuses on a mismatch.

``bug_hunts.excluded_entry_kinds`` (text[], same type as 0009's
``excluded_dirs``) is the declared tree-entry scope of the run: entry kinds out
of BOTH counts. fr's bug_report/v1 REQUIRES it alongside ``excluded_dirs`` and
``worktree_dirty``. The CHECK pins every element to the closed enum
(``non_utf8_path`` / ``submodule`` / ``symlink``), mirroring the service's
fail-closed validation in ``services.workspace_hunt``.

All NULLABLE with NO default and NO backfill: NULL = not measured (log-analysis
findings, rows written before 0011). Prod had 0 bug_hunts / 0 findings rows when
this was written, and a fingerprint cannot be computed from a stored row (the
source never leaves the workspace). The CHECKs pin the sha256 hex shape.

Lock note: ADD COLUMN without a default is metadata-only on PostgreSQL; the
CHECKs are added NOT VALID and then validated (an empty or all-NULL column
validates without a long lock).
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None

#: NULL = not measured; otherwise every element is one of the closed enum. ``<@``
#: (contained-by) on an empty array is true, and an array with a NULL element is
#: refused by the ``array_position`` arm.
EXCLUDED_ENTRY_KINDS_CHECK = (
    "excluded_entry_kinds IS NULL OR ("
    "excluded_entry_kinds <@ ARRAY['non_utf8_path','submodule','symlink']::text[] "
    "AND array_position(excluded_entry_kinds, NULL) IS NULL)"
)


def upgrade() -> None:
    op.add_column("bug_hunt_findings", sa.Column("fingerprint", sa.String(64), nullable=True))
    op.add_column("bug_hunts", sa.Column("agent_sha", sa.String(64), nullable=True))
    op.execute(
        "ALTER TABLE bug_hunt_findings ADD CONSTRAINT ck_bug_hunt_findings_fingerprint_hex "
        "CHECK (fingerprint IS NULL OR fingerprint ~ '^[0-9a-f]{64}$') NOT VALID"
    )
    op.execute(
        "ALTER TABLE bug_hunt_findings VALIDATE CONSTRAINT ck_bug_hunt_findings_fingerprint_hex"
    )
    op.execute(
        "ALTER TABLE bug_hunts ADD CONSTRAINT ck_bug_hunts_agent_sha_hex "
        "CHECK (agent_sha IS NULL OR agent_sha ~ '^[0-9a-f]{64}$') NOT VALID"
    )
    op.execute("ALTER TABLE bug_hunts VALIDATE CONSTRAINT ck_bug_hunts_agent_sha_hex")
    op.add_column(
        "bug_hunts",
        sa.Column("excluded_entry_kinds", postgresql.ARRAY(sa.Text()), nullable=True),
    )
    op.execute(
        "ALTER TABLE bug_hunts ADD CONSTRAINT ck_bug_hunts_excluded_entry_kinds_enum "
        f"CHECK ({EXCLUDED_ENTRY_KINDS_CHECK}) NOT VALID"
    )
    op.execute("ALTER TABLE bug_hunts VALIDATE CONSTRAINT ck_bug_hunts_excluded_entry_kinds_enum")


def downgrade() -> None:
    op.drop_constraint("ck_bug_hunts_excluded_entry_kinds_enum", "bug_hunts", type_="check")
    op.drop_column("bug_hunts", "excluded_entry_kinds")
    op.drop_constraint("ck_bug_hunts_agent_sha_hex", "bug_hunts", type_="check")
    op.drop_constraint("ck_bug_hunt_findings_fingerprint_hex", "bug_hunt_findings", type_="check")
    op.drop_column("bug_hunts", "agent_sha")
    op.drop_column("bug_hunt_findings", "fingerprint")
