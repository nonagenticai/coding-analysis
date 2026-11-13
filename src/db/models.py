"""SQLAlchemy ORM models for the analysis-owned code-graph schema.

These models are the schema source of truth consumed by Alembic. The
analysis service writes/reads the code graph through raw psycopg at runtime
(see ``db.facade`` / ``db.entity_mapper``); the ORM is *not* used on the
request path. Keeping the models here lets Alembic autogenerate/diff the
PostgreSQL schema while the runtime stays on a single psycopg pool.

Differences from the legacy aiosqlite schema (``coding_client.db``):
- A ``code_graph_id`` discriminator column scopes every row to one graph, so
  many graphs share one PostgreSQL database instead of one file per graph.
- Row identity carries ``tenant_id``: nodes are unique on
  ``(code_graph_id, tenant_id, id)`` and edges on
  ``(code_graph_id, tenant_id, source, target, relationship)`` via
  ``NULLS NOT DISTINCT`` unique indexes (migration 0002), so two tenants sharing a
  ``code_graph_id`` (== container name) no longer collide on write while legacy
  NULL-tenant rows still collapse and stay readable. The DB has NO node PRIMARY
  KEY — identity is the unique index. ``Node`` keeps a mapper-level ``primary_key``
  below only because SQLAlchemy requires one; it is not on the runtime path and is
  the one intentional ORM-vs-DB divergence (autogenerate would show a PK diff; CI
  gates on ``alembic upgrade head``, not autogenerate). The edge foreign keys were
  dropped in 0002 — the application orders edge-before-node deletes and
  orphan-checks before inserting edges, so they guarded nothing the app does not.
- SQLite type mappings: TEXT-JSON -> JSONB, REAL -> DOUBLE PRECISION,
  integer 0/1 flags -> SMALLINT, AUTOINCREMENT -> IDENTITY,
  CURRENT_TIMESTAMP -> ``now()`` TIMESTAMPTZ.

Only tables the service actually uses are modelled: code graph tables plus
the persisted Bug Hunter job/report tables and the stored quality results. Audit/FTS/schema_version/language
stats/metrics tables from the legacy schema are intentionally omitted (unused,
or owned by other modules migrated later).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    Identity,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: ``tenant_id`` is a native PostgreSQL ``uuid`` (migration 0005) — tenant identity
#: has been a uuid7 since the 2026-04-29 Keycloak migration, so the column type is
#: now the guard rather than whichever call site remembered to validate.
#:
#: ``as_uuid=False`` keeps the Python-side value a ``str``, matching the
#: ``Mapped[str]`` annotations below and every ``tenant_id: str`` signature in this
#: service. ``as_uuid=True`` hands back ``uuid.UUID``, which compares False against
#: the string it prints as and is not JSON-serialisable; that shape has frozen
#: outboxes elsewhere in this fleet. Do not flip it.
#:
#: ⚠️ Be clear about its REACH: this setting governs the SQLAlchemy path only, and
#: the SQLAlchemy path here is Alembic plus ``tests/unit/test_db_package.py`` — the
#: request path never touches the ORM (see this module's docstring; runtime goes
#: through raw psycopg via ``db.facade`` / ``db.entity_mapper``). psycopg3 adapts a
#: ``uuid`` column to ``uuid.UUID`` on its own, whatever the ORM declares. So the
#: runtime guarantee is NOT this line — it is that every raw query which SELECTs or
#: RETURNS the column does so as ``tenant_id::text``. Add that cast to any new one:
#: ``src/services/bug_hunt_report_store.py``, ``src/processing/task_utilities.py``
#: and ``src/app/api/endpoints/graphs_reader.py`` are the existing examples.
TenantId = PG_UUID(as_uuid=False)


class Base(DeclarativeBase):
    """Shared declarative base for all analysis code-graph models."""


class Node(Base):
    """A code entity (function, class, module, ...) within one code graph."""

    __tablename__ = "nodes"

    code_graph_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    id: Mapped[str] = mapped_column(String(512), primary_key=True)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    properties: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tenant_id: Mapped[str | None] = mapped_column(TenantId, nullable=True)

    # Multi-language columns (legacy "migrate_to_multi_language" set).
    language: Mapped[str | None] = mapped_column(String(32), nullable=True, server_default="python")
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    line_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    visibility: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_static: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default="0")
    is_abstract: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default="0")
    is_final: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default="0")
    is_async: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default="0")
    signature: Mapped[str | None] = mapped_column(Text, nullable=True)
    docstring: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        # Tenant-scoped identity (migration 0002). NULLS NOT DISTINCT collapses
        # legacy NULL-tenant rows while keeping distinct tenants distinct. This is
        # the real DB identity; the mapper-level primary_key above is logical only.
        Index(
            "uq_node_identity",
            "code_graph_id",
            "tenant_id",
            "id",
            unique=True,
            postgresql_nulls_not_distinct=True,
        ),
        Index("idx_node_cg", "code_graph_id"),
        Index("idx_node_cg_type", "code_graph_id", "type"),
        Index("idx_node_cg_content_hash", "code_graph_id", "content_hash"),
        Index("idx_node_cg_tenant", "code_graph_id", "tenant_id"),
        Index("idx_node_cg_language", "code_graph_id", "language"),
        Index("idx_node_cg_file_path", "code_graph_id", "file_path"),
    )


class Edge(Base):
    """A relationship between two code entities within one code graph."""

    __tablename__ = "edges"

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    code_graph_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(512), nullable=False)
    target: Mapped[str] = mapped_column(String(512), nullable=False)
    relationship_type: Mapped[str] = mapped_column("relationship", String(64), nullable=False)
    properties: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    weight: Mapped[float] = mapped_column(Double, nullable=False, server_default="1.0")
    tenant_id: Mapped[str | None] = mapped_column(TenantId, nullable=True)

    # Multi-language relationship columns (legacy edge augmentation set).
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    location_file: Mapped[str | None] = mapped_column(Text, nullable=True)
    location_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    location_column: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        # Tenant-scoped edge identity (migration 0002). The composite FKs into
        # nodes were dropped: re-pointing them through nullable tenant_id is
        # MATCH SIMPLE (skips NULL rows) and would FK-violate a tenant edge over a
        # legacy NULL node; the app already orders edge-before-node deletes and
        # orphan-checks before insert.
        Index(
            "uq_edge_identity",
            "code_graph_id",
            "tenant_id",
            "source",
            "target",
            "relationship",
            unique=True,
            postgresql_nulls_not_distinct=True,
        ),
        Index("idx_edge_cg", "code_graph_id"),
        Index("idx_edge_cg_source", "code_graph_id", "source"),
        Index("idx_edge_cg_target", "code_graph_id", "target"),
        Index("idx_edge_cg_relationship", "code_graph_id", "relationship"),
        Index("idx_edge_cg_tenant", "code_graph_id", "tenant_id"),
    )


class GraphVersion(Base):
    """A named version checkpoint of a code graph (incremental-update subsystem)."""

    __tablename__ = "graph_versions"

    code_graph_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("idx_graph_version_cg_created", "code_graph_id", "created_at"),)


class AnalysisSession(Base):
    """Tracks one multi-language analysis run for a code graph."""

    __tablename__ = "analysis_sessions"

    code_graph_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    languages: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    files_processed: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    files_failed: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    total_entities: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    total_relationships: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="running")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("idx_session_cg_status", "code_graph_id", "status"),)


class BugHunt(Base):
    """One asynchronous, tenant-scoped bug hunting job."""

    __tablename__ = "bug_hunts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    code_graph_id: Mapped[str] = mapped_column(String(255), nullable=False)
    tenant_id: Mapped[str] = mapped_column(TenantId, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="queued")
    request_payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    errors_found: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    fixes_generated: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    root_cause_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    fix_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    estimated_effort: Mapped[str | None] = mapped_column(Text, nullable=True)
    risk_assessment: Mapped[str | None] = mapped_column(Text, nullable=True)
    analysis_time: Mapped[float | None] = mapped_column(Double, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Coverage provenance (migration 0009). NULL = not measured; no default.
    scanned_files: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tracked_source_files: Mapped[int | None] = mapped_column(Integer, nullable=True)
    commit_sha: Mapped[str | None] = mapped_column(String(40), nullable=True)
    # Declared hunt scope: directory names excluded from both counts. NULL = no scan.
    excluded_dirs: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    # The hunt scans HEAD; true = the working tree had uncommitted changes (0010).
    worktree_dirty: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # sha256(MANIFEST.sha256) of the agent that ran the hunt (0011). NULL = no scan.
    agent_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Declared entry-kind scope (0011): non_utf8_path / submodule / symlink. NULL = no scan.
    excluded_entry_kinds: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)

    __table_args__ = (
        CheckConstraint("scanned_files >= 0", name="ck_bug_hunts_scanned_files_nonneg"),
        CheckConstraint(
            "agent_sha IS NULL OR agent_sha ~ '^[0-9a-f]{64}$'",
            name="ck_bug_hunts_agent_sha_hex",
        ),
        CheckConstraint(
            "excluded_entry_kinds IS NULL OR ("
            "excluded_entry_kinds <@ ARRAY['non_utf8_path','submodule','symlink']::text[] "
            "AND array_position(excluded_entry_kinds, NULL) IS NULL)",
            name="ck_bug_hunts_excluded_entry_kinds_enum",
        ),
        CheckConstraint(
            "tracked_source_files >= 0", name="ck_bug_hunts_tracked_source_files_nonneg"
        ),
        CheckConstraint(
            "commit_sha IS NULL OR commit_sha ~ '^[0-9a-f]{40}$'",
            name="ck_bug_hunts_commit_sha_full_hex",
        ),
        Index("idx_bug_hunt_tenant_container_created", "tenant_id", "code_graph_id", "created_at"),
        Index("idx_bug_hunt_status_created", "status", "created_at"),
        Index("idx_bug_hunt_claim", "status", "lease_expires_at", "created_at"),
    )


class BugHuntFinding(Base):
    """A single bug finding stored separately for path-scoped report queries."""

    __tablename__ = "bug_hunt_findings"

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    hunt_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("bug_hunts.id", ondelete="CASCADE"), nullable=False
    )
    code_graph_id: Mapped[str] = mapped_column(String(255), nullable=False)
    tenant_id: Mapped[str] = mapped_column(TenantId, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    line_number: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    severity: Mapped[str] = mapped_column(String(32), nullable=False, server_default="unrated")
    finding: Mapped[dict] = mapped_column(JSONB, nullable=False)
    suggested_fixes: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")
    # The hunt agent's content fingerprint (0011); the finding's identity for a
    # recheck. NULL = not fingerprinted (log-analysis findings, pre-0011 rows).
    fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "fingerprint IS NULL OR fingerprint ~ '^[0-9a-f]{64}$'",
            name="ck_bug_hunt_findings_fingerprint_hex",
        ),
        Index(
            "idx_bug_finding_hunt_path",
            "hunt_id",
            "tenant_id",
            "file_path",
            postgresql_ops={"file_path": "text_pattern_ops"},
        ),
    )


class QualityResult(Base):
    """One stored quality-analysis run for a (tenant, container, commit).

    Written by ``services.quality_result_store`` after ``POST /quality/analyze``
    and ``POST /quality/analyze-sync``; read by the ``GET /quality/{c}/…`` routes,
    which never re-run the analyzer (migration 0007).
    """

    __tablename__ = "quality_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(TenantId, nullable=False)
    container_name: Mapped[str] = mapped_column(String(255), nullable=False)
    repo_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    commit_sha: Mapped[str | None] = mapped_column(String(40), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    overall_score: Mapped[float | None] = mapped_column(Double, nullable=True)
    grade: Mapped[str | None] = mapped_column(String(16), nullable=True)
    language: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tools_attempted: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")
    tools_succeeded: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")
    tools_failed: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")
    summary: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    issue_stats: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")
    issues: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")
    issues_total: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    issues_truncated: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    recommendations: Mapped[list] = mapped_column(JSONB, nullable=False, server_default="[]")
    execution_time: Mapped[float | None] = mapped_column(Double, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "status IN ('complete', 'partial', 'inconclusive', 'failed')",
            name="ck_quality_results_status",
        ),
        CheckConstraint(
            "commit_sha IS NULL OR commit_sha ~ '^[0-9a-f]{40}$'",
            name="ck_quality_results_commit_sha",
        ),
        Index("idx_quality_results_tenant_repo_sha", "tenant_id", "repo_url", "commit_sha"),
    )


# Declared outside the class body because it needs the mapped column's ``.desc()``
# (readers take the newest row per (tenant, container)).
Index(
    "idx_quality_results_tenant_container_created",
    QualityResult.tenant_id,
    QualityResult.container_name,
    QualityResult.created_at.desc(),
)
