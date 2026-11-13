"""Integration tests for src/db against a real PostgreSQL.

Exercises the actual migration apply, the facade batch round-trip, the entity
mapper, tenant-scoped identity (migration 0002), and cross-graph
(code_graph_id) isolation.

⚠️ Every tenant literal here must be a real uuid. ``tenant_id`` is a native
``uuid`` column since migration 0005, so the old ``"t1"`` / ``"tenant-A"`` seeds
now raise ``22P02 invalid input syntax for type uuid`` at INSERT time rather
than storing a wrong-but-plausible value. Use the ``TENANT_*`` constants below;
they are uuid7-shaped and ordered so ``ORDER BY tenant_id`` is still meaningful.

Skipped unless ``TEST_DATABASE_URL`` points at a reachable postgres (the CI
``postgres:18`` service provides it). Marked ``integration`` so it is excluded
from the default ``-m 'not integration'`` run.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration

_DSN = os.getenv("TEST_DATABASE_URL")
if not _DSN:
    pytest.skip("TEST_DATABASE_URL not set", allow_module_level=True)

# Tenant uuids. Chosen so TENANT_A < TENANT_B under uuid ordering, which
# ``test_same_container_distinct_tenants_do_not_collide`` relies on.
TENANT_1 = "0195f48b-0000-7000-8000-000000000011"
TENANT_9 = "0195f48b-0000-7000-8000-000000000099"
TENANT_A = "0195f48b-0000-7000-8000-00000000000a"
TENANT_B = "0195f48b-0000-7000-8000-00000000000b"
TENANT_T = "0195f48b-0000-7000-8000-0000000000ee"


@pytest.fixture(scope="module", autouse=True)
def _migrated_db():
    """Point settings/session at the test DB and run alembic upgrade head."""
    os.environ["DATABASE_URL"] = _DSN
    os.environ["PGBOUNCER_URL"] = ""  # local PG: no pgbouncer, use direct URL

    from app.core.config.core_config_settings import get_settings

    get_settings.cache_clear()

    from alembic.config import Config

    from alembic import command

    repo_root = Path(__file__).resolve().parents[2]
    cfg = Config(str(repo_root / "alembic.ini"))
    command.upgrade(cfg, "head")
    yield
    command.downgrade(cfg, "base")


@pytest.fixture(autouse=True)
async def _clean_tables():
    """TRUNCATE between tests (facade/mapper COMMIT, so rollback fixtures won't work)."""
    from db import session

    async with session.get_raw_connection() as conn:
        await conn.execute(
            "TRUNCATE bug_hunt_findings, bug_hunts, edges, nodes, analysis_sessions, "
            "graph_versions RESTART IDENTITY CASCADE"
        )
        await conn.commit()
    yield


def _entity(entity_id: str, name: str):
    return SimpleNamespace(
        id=entity_id,
        type=SimpleNamespace(value="function"),
        language="python",
        file_path="src/x.py",
        line_number=10,
        column_start=0,
        column_end=4,
        visibility="public",
        is_static=True,
        is_abstract=False,
        is_final=False,
        is_async=False,
        signature=f"def {name}()",
        docstring="doc",
        properties={"content_hash": "h-" + entity_id, "name": name},
        metadata={},
        return_type=None,
        parameter_types=[],
        created_at=dt.datetime(2026, 5, 21, tzinfo=dt.UTC),
        updated_at=dt.datetime(2026, 5, 21, tzinfo=dt.UTC),
    )


async def _count(code_graph_id: str, table: str) -> int:
    from db import session

    async with session.get_raw_connection() as conn:
        cur = await conn.execute(
            f"SELECT count(*) FROM {table} WHERE code_graph_id = %s", (code_graph_id,)
        )
        row = await cur.fetchone()
        return row[0]


async def test_facade_batch_roundtrip():
    from db.facade import GraphDatabaseConnection

    db = GraphDatabaseConnection("graph-a", tenant_id=TENANT_1)
    await db.connect()
    try:
        db.add_node("n1", "function", {"name": "f"})
        db.add_node("n2", "function", {"name": "g"})
        db.add_edge("n1", "n2", "calls", {"line": 5})
        await db.flush_pending_operations()
    finally:
        await db.close()

    assert await _count("graph-a", "nodes") == 2
    assert await _count("graph-a", "edges") == 1


async def test_facade_upsert_overwrites():
    from db.facade import GraphDatabaseConnection

    db = GraphDatabaseConnection("graph-a")
    await db.connect()
    try:
        db.add_node("n1", "function", {"name": "first"})
        await db.flush_pending_operations()
        db.add_node("n1", "class", {"name": "second"})
        await db.flush_pending_operations()
    finally:
        await db.close()

    from db import session

    async with session.get_raw_connection() as conn:
        cur = await conn.execute(
            "SELECT type, properties->>'name' FROM nodes WHERE code_graph_id=%s AND id=%s",
            ("graph-a", "n1"),
        )
        row = await cur.fetchone()
    assert row == ("class", "second")  # ON CONFLICT overwrote
    assert await _count("graph-a", "nodes") == 1


async def test_entity_mapper_store_entity_columns():
    from db import session
    from db.entity_mapper import MultiLanguageEntityMapper

    async with session.get_raw_connection() as conn:
        mapper = MultiLanguageEntityMapper(conn, "graph-b", tenant_id=TENANT_9)
        await mapper.store_entity(_entity("ent-1", "do_work"))
        await conn.commit()
        cur = await conn.execute(
            "SELECT language, is_static, is_async, content_hash, tenant_id::text, "
            "properties->'properties'->>'name' FROM nodes WHERE code_graph_id=%s AND id=%s",
            ("graph-b", "ent-1"),
        )
        row = await cur.fetchone()
    assert row == ("python", 1, 0, "h-ent-1", TENANT_9, "do_work")


async def test_multi_graph_isolation_no_leak():
    from db import session
    from db.entity_mapper import MultiLanguageEntityMapper

    # Same node id "shared" in two graphs — composite PK keeps them distinct.
    async with session.get_raw_connection() as conn:
        await MultiLanguageEntityMapper(conn, "graph-a").store_entity(_entity("shared", "a_name"))
        await MultiLanguageEntityMapper(conn, "graph-b").store_entity(_entity("shared", "b_name"))
        await conn.commit()

    assert await _count("graph-a", "nodes") == 1
    assert await _count("graph-b", "nodes") == 1

    async with session.get_raw_connection() as conn:
        cur = await conn.execute(
            "SELECT properties->'properties'->>'name' FROM nodes WHERE code_graph_id=%s AND id=%s",
            ("graph-a", "shared"),
        )
        assert (await cur.fetchone())[0] == "a_name"  # graph-a never sees graph-b's row


async def test_orphan_edge_persists_after_fk_drop():
    """Migration 0002 dropped the composite edge FKs (they were MATCH SIMPLE and
    FK-violated tenant edges over legacy NULL nodes). Orphan prevention now lives
    in the application writers (``analysis_utils`` orphan-checks before insert),
    not the DB — so a raw facade edge insert against missing nodes now persists.
    """
    from db.facade import GraphDatabaseConnection

    db = GraphDatabaseConnection("graph-c")
    await db.connect()
    try:
        db.add_edge("missing-src", "missing-tgt", "calls")
        await db.flush_pending_operations()  # no FK -> no exception
    finally:
        await db.close()

    assert await _count("graph-c", "edges") == 1


async def test_same_container_distinct_tenants_do_not_collide():
    """The fix: two tenants sharing a code_graph_id + node id both persist."""
    from db import session
    from db.entity_mapper import MultiLanguageEntityMapper

    async with session.get_raw_connection() as conn:
        await MultiLanguageEntityMapper(conn, "shared-cg", tenant_id=TENANT_A).store_entity(
            _entity("shared", "a_name")
        )
        await MultiLanguageEntityMapper(conn, "shared-cg", tenant_id=TENANT_B).store_entity(
            _entity("shared", "b_name")
        )
        await conn.commit()

    assert await _count("shared-cg", "nodes") == 2  # both tenants kept, no DO-NOTHING drop
    async with session.get_raw_connection() as conn:
        cur = await conn.execute(
            "SELECT tenant_id::text, properties->'properties'->>'name' FROM nodes "
            "WHERE code_graph_id=%s ORDER BY tenant_id",
            ("shared-cg",),
        )
        rows = await cur.fetchall()
    assert rows == [(TENANT_A, "a_name"), (TENANT_B, "b_name")]


async def test_null_tenant_rows_still_collapse():
    """Legacy idempotence: two NULL-tenant inserts for the same (cg,id) collapse
    to one row under NULLS NOT DISTINCT (no tenant supplied -> NULL)."""
    from db.facade import GraphDatabaseConnection

    db = GraphDatabaseConnection("legacy-cg")  # tenant_id defaults to None
    await db.connect()
    try:
        db.add_node("n1", "function", {"name": "first"})
        await db.flush_pending_operations()
        db.add_node("n1", "class", {"name": "second"})  # same (cg, NULL, id) -> upsert
        await db.flush_pending_operations()
    finally:
        await db.close()

    assert await _count("legacy-cg", "nodes") == 1
    from db import session

    async with session.get_raw_connection() as c:
        cur = await c.execute(
            "SELECT tenant_id::text, type FROM nodes WHERE code_graph_id=%s", ("legacy-cg",)
        )
        assert await cur.fetchone() == (None, "class")


async def test_tenant_write_over_legacy_null_graph_does_not_throw():
    """A tenant-attributed write against a code_graph_id that already holds legacy
    NULL-tenant rows must not FK-violate or error (FKs dropped); it adds the
    tenant's own distinct rows alongside the NULL ones."""
    from db import session
    from db.entity_mapper import MultiLanguageEntityMapper

    # Pre-existing legacy NULL-tenant node + edge for "graph-leg".
    async with session.get_raw_connection() as conn:
        await conn.execute(
            "INSERT INTO nodes (code_graph_id, id, type) VALUES (%s,%s,%s),(%s,%s,%s)",
            ("graph-leg", "src", "function", "graph-leg", "dst", "function"),
        )
        await conn.execute(
            "INSERT INTO edges (code_graph_id, source, target, relationship) VALUES (%s,%s,%s,%s)",
            ("graph-leg", "src", "dst", "calls"),
        )
        await conn.commit()

    # Tenant-T writes an entity into the same code_graph_id — no throw.
    async with session.get_raw_connection() as conn:
        mapper = MultiLanguageEntityMapper(conn, "graph-leg", tenant_id=TENANT_T)
        await mapper.store_entity(_entity("src", "t_src"))
        await mapper.store_relationship(
            SimpleNamespace(
                source_id="src",
                target_id="dst",
                type=SimpleNamespace(value="calls"),
                weight=1.0,
                context=None,
                location=None,
                properties={},
                created_at=dt.datetime(2026, 5, 21, tzinfo=dt.UTC),
            )
        )
        await conn.commit()

    # 2 legacy NULL nodes + 1 tenant-T node ("src" duplicated under tenant-T).
    assert await _count("graph-leg", "nodes") == 3
    async with session.get_raw_connection() as conn:
        cur = await conn.execute(
            "SELECT count(*) FROM nodes WHERE code_graph_id=%s AND tenant_id IS NULL",
            ("graph-leg",),
        )
        assert (await cur.fetchone())[0] == 2
        cur = await conn.execute(
            "SELECT count(*) FROM nodes WHERE code_graph_id=%s AND tenant_id=%s",
            ("graph-leg", TENANT_T),
        )
        assert (await cur.fetchone())[0] == 1


async def test_bug_hunt_report_store_roundtrip_and_recursive_path_query():
    """Large reports are normalized and every read remains tenant-scoped."""
    from models.analysis_models import BugHuntingResponse
    from services.bug_hunt_report_store import BugHuntReportStore

    store = BugHuntReportStore()
    await store.create_hunt(
        hunt_id="hunt-db-1",
        container_name="shared-repo",
        tenant_id=TENANT_A,
        request_payload={"error_logs": ""},
    )
    claimed = await store.claim_next_hunt(worker_id="integration-worker")
    assert claimed is not None and claimed["hunt_id"] == "hunt-db-1"

    def error(error_id: str, file_path: str, line_number: int) -> dict:
        return {
            "error_id": error_id,
            "error_type": "syntax_error",
            "error_message": "syntax failure",
            "stack_trace": "",
            "file_path": file_path,
            "line_number": line_number,
            "function_name": "",
            "error_category": "syntax_error",
            "severity": "medium",
            "timestamp": "now",
            "frequency": 1,
            "impact_score": 0.9,
            "suggested_fix": "",
            "related_errors": [],
        }

    report = BugHuntingResponse(
        errors_found=3,
        fixes_generated=0,
        root_cause_analysis="parser failures",
        fix_summary="repair syntax",
        estimated_effort="small",
        risk_assessment="low",
        errors=[
            error("e1", "/repo/src/api/routes.py", 10),
            error("e2", "/repo/src/api/internal/auth.py", 20),
            error("e3", "/repo/tests/test_routes.py", 30),
        ],
        resolves=[],
        pull_request=None,
        analysis_time=1.0,
    )
    await store.complete_hunt(
        hunt_id="hunt-db-1",
        container_name="shared-repo",
        tenant_id=TENANT_A,
        report=report,
        codebase_path="/repo",
        worker_id="integration-worker",
    )

    job = await store.get_hunt(
        hunt_id="hunt-db-1", container_name="shared-repo", tenant_id=TENANT_A
    )
    assert job is not None and job["status"] == "completed"
    assert (
        await store.get_hunt(hunt_id="hunt-db-1", container_name="shared-repo", tenant_id=TENANT_B)
        is None
    )

    total, matched_files, findings = await store.get_findings_for_path(
        hunt_id="hunt-db-1",
        container_name="shared-repo",
        tenant_id=TENANT_A,
        path="src/api",
        limit=100,
        offset=0,
    )
    assert total == 2
    assert matched_files == 2
    assert {item["file_path"] for item in findings} == {
        "src/api/routes.py",
        "src/api/internal/auth.py",
    }


async def test_bug_hunt_coverage_provenance_persists_and_null_stays_null():
    """scanned/tracked/commit_sha round-trip through migration 0009 columns.

    An unmeasured hunt must read back NULL (never 0), because fr's
    bug_report/v1 must refuse it rather than see a measured empty repository.
    """
    from models.analysis_models import BugHuntingResponse
    from services.bug_hunt_report_store import BugHuntReportStore

    store = BugHuntReportStore()
    sha = "ab" * 20

    async def run(hunt_id: str, **coverage) -> dict:
        await store.create_hunt(
            hunt_id=hunt_id,
            container_name="cov-repo",
            tenant_id=TENANT_A,
            request_payload={"error_logs": ""},
        )
        claimed = await store.claim_next_hunt(worker_id="cov-worker")
        assert claimed is not None and claimed["hunt_id"] == hunt_id
        assert claimed["scanned_files"] is None  # nothing measured before the run
        await store.complete_hunt(
            hunt_id=hunt_id,
            container_name="cov-repo",
            tenant_id=TENANT_A,
            report=BugHuntingResponse(
                errors_found=0,
                fixes_generated=0,
                root_cause_analysis="none",
                fix_summary="",
                estimated_effort="",
                risk_assessment="",
                errors=[],
                resolves=[],
                pull_request=None,
                analysis_time=0.5,
                **coverage,
            ),
            codebase_path="/repo",
            worker_id="cov-worker",
        )
        job = await store.get_hunt(hunt_id=hunt_id, container_name="cov-repo", tenant_id=TENANT_A)
        assert job is not None
        return job

    measured = await run(
        "hunt-cov-1",
        scanned_files=11,
        tracked_source_files=12,
        commit_sha=sha,
        worktree_dirty=True,
        excluded_dirs=["build", "vendor"],
        excluded_entry_kinds=["non_utf8_path", "submodule", "symlink"],
    )
    assert measured["excluded_entry_kinds"] == ["non_utf8_path", "submodule", "symlink"]
    assert (measured["scanned_files"], measured["tracked_source_files"]) == (11, 12)
    assert measured["commit_sha"] == sha
    assert measured["worktree_dirty"] is True
    assert measured["excluded_dirs"] == ["build", "vendor"]

    unmeasured = await run("hunt-cov-2")
    assert unmeasured["scanned_files"] is None
    assert unmeasured["tracked_source_files"] is None
    assert unmeasured["commit_sha"] is None
    assert unmeasured["worktree_dirty"] is None
    assert unmeasured["excluded_dirs"] is None
    assert unmeasured["excluded_entry_kinds"] is None


async def test_finding_fingerprint_and_agent_sha_persist_and_get_finding_is_tenant_scoped():
    """0011: fingerprint per finding, agent_sha per hunt; get_finding never crosses tenants."""
    from models.analysis_models import BugHuntingResponse
    from services.bug_hunt_report_store import BugHuntReportStore

    store = BugHuntReportStore()
    fingerprint, agent_sha = "a1" * 32, "b2" * 32
    await store.create_hunt(
        hunt_id="hunt-fp-1",
        container_name="fp-repo",
        tenant_id=TENANT_A,
        request_payload={"error_logs": ""},
    )
    claimed = await store.claim_next_hunt(worker_id="fp-worker")
    assert claimed is not None and claimed["agent_sha"] is None
    error = {
        "error_id": "err_0",
        "error_type": "syntax_error",
        "error_message": "bad",
        "stack_trace": "",
        "file_path": "src/bad.py",
        "line_number": 3,
        "function_name": "",
        "error_category": "syntax_error",
        "severity": "unrated",
        "timestamp": "now",
        "frequency": 1,
        "impact_score": 0.9,
        "suggested_fix": "",
        "related_errors": [],
    }
    await store.complete_hunt(
        hunt_id="hunt-fp-1",
        container_name="fp-repo",
        tenant_id=TENANT_A,
        report=BugHuntingResponse(
            errors_found=2,
            fixes_generated=0,
            root_cause_analysis="r",
            fix_summary="",
            estimated_effort="",
            risk_assessment="",
            errors=[{**error, "fingerprint": fingerprint}, {**error, "line_number": 9}],
            resolves=[],
            pull_request=None,
            analysis_time=0.1,
            commit_sha="cd" * 20,
            scanned_files=1,
            tracked_source_files=1,
            agent_sha=agent_sha,
        ),
        codebase_path="/workspace",
        worker_id="fp-worker",
    )

    job = await store.get_hunt(hunt_id="hunt-fp-1", container_name="fp-repo", tenant_id=TENANT_A)
    assert job is not None and job["agent_sha"] == agent_sha
    _total, _files, rows = await store.get_findings_for_path(
        hunt_id="hunt-fp-1",
        container_name="fp-repo",
        tenant_id=TENANT_A,
        path="",
        limit=10,
        offset=0,
    )
    by_line = {row["line_number"]: row for row in rows}
    assert by_line[3]["fingerprint"] == fingerprint
    assert by_line[9]["fingerprint"] is None  # absent stays NULL, never invented

    finding_id = by_line[3]["finding_id"]
    mine = await store.get_finding(
        hunt_id="hunt-fp-1", container_name="fp-repo", tenant_id=TENANT_A, finding_id=finding_id
    )
    assert mine is not None and mine["fingerprint"] == fingerprint
    theirs = await store.get_finding(
        hunt_id="hunt-fp-1", container_name="fp-repo", tenant_id=TENANT_B, finding_id=finding_id
    )
    assert theirs is None
    other_hunt = await store.get_finding(
        hunt_id="hunt-other", container_name="fp-repo", tenant_id=TENANT_A, finding_id=finding_id
    )
    assert other_hunt is None


async def test_the_database_refuses_a_malformed_fingerprint_or_agent_sha():
    import psycopg

    from db import session

    async with session.get_raw_connection() as conn:
        await conn.execute(
            "INSERT INTO bug_hunts (id, code_graph_id, tenant_id, status) "
            "VALUES ('hunt-ck', 'ck-repo', %s, 'completed')",
            (TENANT_A,),
        )
        await conn.commit()
        with pytest.raises(psycopg.errors.CheckViolation):
            await conn.execute("UPDATE bug_hunts SET agent_sha = 'XYZ' WHERE id = 'hunt-ck'")
        await conn.rollback()
        for bad in (["symlink", "hardlink"], ["symlink", None]):
            with pytest.raises(psycopg.errors.CheckViolation):
                await conn.execute(
                    "UPDATE bug_hunts SET excluded_entry_kinds = %s WHERE id = 'hunt-ck'", (bad,)
                )
            await conn.rollback()
        await conn.execute(
            "UPDATE bug_hunts SET excluded_entry_kinds = %s WHERE id = 'hunt-ck'",
            (["non_utf8_path", "submodule", "symlink"],),
        )
        await conn.commit()
        with pytest.raises(psycopg.errors.CheckViolation):
            await conn.execute(
                "INSERT INTO bug_hunt_findings (hunt_id, code_graph_id, tenant_id, file_path, "
                "file_name, finding, fingerprint) "
                "VALUES ('hunt-ck', 'ck-repo', %s, 'a.py', 'a.py', '{}', 'not-hex')",
                (TENANT_A,),
            )
        await conn.rollback()


async def _seed_graph(code_graph_id, nodes, edges):
    """Seed a graph through the facade batch API (nodes before edges)."""
    from db.facade import GraphDatabaseConnection

    db = GraphDatabaseConnection(code_graph_id)
    await db.connect()
    try:
        for node_id, node_type, props in nodes:
            db.add_node(node_id, node_type, props)
        for source, target, rel in edges:
            db.add_edge(source, target, rel)
        await db.flush_pending_operations()
    finally:
        await db.close()


async def test_application_read_paths_do_not_leak_across_graphs():
    """The real guard for the ~141 hand-threaded ``WHERE code_graph_id = %s`` filters.

    Two graphs share a node id ("shared"). We drive the *application* read code
    (not the facade) scoped to graph-b and assert graph-a's rows never surface.
    If any read path dropped its ``code_graph_id`` predicate, graph-a's data
    would leak in and these assertions would fail.
    """
    from db.facade import GraphDatabaseConnection
    from services.code_graph.structure_storage import CodeStructureStorage
    from services.code_graph_metrics import CodeGraphMetricsService

    # graph-a: 3 nodes / 2 edges, including an edge INTO the shared node.
    await _seed_graph(
        "graph-a",
        nodes=[
            ("a1", "function", {"name": "a_one"}),
            ("a2", "class", {"name": "a_two"}),
            ("shared", "function", {"name": "a_shared"}),
        ],
        edges=[("a1", "a2", "calls"), ("a2", "shared", "calls")],
    )
    # graph-b: 2 nodes / 1 edge; same "shared" id but distinct properties.
    await _seed_graph(
        "graph-b",
        nodes=[
            ("b1", "function", {"name": "b_one"}),
            ("shared", "function", {"name": "b_shared"}),
        ],
        edges=[("b1", "shared", "calls")],
    )

    db_b = GraphDatabaseConnection("graph-b")
    await db_b.connect()
    try:
        # --- structure_storage read path ---
        storage = CodeStructureStorage(db_b)
        entities = await storage.get_stored_entities()
        ids = {e["id"] for e in entities}
        assert ids == {"b1", "shared"}  # graph-a's a1/a2 never appear
        shared = next(e for e in entities if e["id"] == "shared")
        assert shared["properties"]["name"] == "b_shared"  # graph-b's row, not graph-a's

        # entity_type filter is also scoped
        funcs = await storage.get_stored_entities(entity_type="function")
        assert {e["id"] for e in funcs} == {"b1", "shared"}

        # relationships of "shared": only graph-b's b1->shared, not graph-a's a2->shared
        rels = await storage.get_entity_relationships("shared")
        assert rels == [{"source": "b1", "target": "shared", "relationship": "calls"}]

        # --- code_graph_metrics read path ---
        metrics = await CodeGraphMetricsService(db_b).get_graph_metrics()
        assert metrics.node_count == 2  # graph-b only (graph-a has 3)
        assert metrics.edge_count == 1  # graph-b only (graph-a has 2)
    finally:
        await db_b.close()


# ---------------------------------------------------------------------------
# tenant_id is a native uuid (migration 0005)
# ---------------------------------------------------------------------------


async def test_tenant_id_is_a_native_uuid_column_on_every_table():
    """The type IS the guard, so assert the type — not just that writes succeed.

    A migration that silently no-ops (wrong table name, a metadata walk that
    compared uuid against uuid) leaves every other test in this file green,
    because a varchar column happily stores uuid strings.
    """
    from db import session

    async with session.get_raw_connection() as conn:
        cur = await conn.execute(
            "SELECT table_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE column_name = 'tenant_id' AND table_schema = current_schema() "
            "ORDER BY table_name"
        )
        rows = await cur.fetchall()

    assert rows == [
        ("bug_hunt_findings", "uuid", "NO"),
        ("bug_hunts", "uuid", "NO"),
        ("dependency_report_snapshots", "uuid", "YES"),
        ("edges", "uuid", "YES"),
        ("nodes", "uuid", "YES"),
        ("quality_results", "uuid", "NO"),
    ], rows


async def test_the_database_refuses_a_tenant_that_is_not_a_uuid():
    """``"tenant-A"`` used to be storable. It is now 22P02 at the door.

    This is the whole point of the change: a label that is not a tenant can no
    longer be attributed to one by a call site that forgot to validate.
    """
    import psycopg

    from db import session

    async with session.get_raw_connection() as conn:
        with pytest.raises(psycopg.errors.InvalidTextRepresentation):
            await conn.execute(
                "INSERT INTO nodes (code_graph_id, id, type, tenant_id) VALUES (%s,%s,%s,%s)",
                ("cg-reject", "n1", "function", "tenant-A"),
            )
        await conn.rollback()


async def test_an_uncast_read_returns_a_uuid_object_which_is_why_reads_cast():
    """psycopg3 adapts a ``uuid`` column to ``uuid.UUID``, whatever the ORM says.

    ``db.models.TenantId`` is ``PG_UUID(as_uuid=False)``, but the ORM is not on
    this service's request path — runtime is raw psycopg. So the ``str``
    guarantee every consumer relies on comes from the ``tenant_id::text`` casts in
    ``bug_hunt_report_store`` / ``task_utilities`` / ``graphs_reader``, not from
    the ORM. If this test ever goes green on the uncast branch, those casts have
    become redundant; while it is red-shaped as written, dropping one of them
    ships ``uuid.UUID`` into an X-Tenant-Id header and a ``!=`` against a
    container label.
    """
    import uuid as uuid_mod

    from db import session

    async with session.get_raw_connection() as conn:
        await conn.execute(
            "INSERT INTO nodes (code_graph_id, id, type, tenant_id) VALUES (%s,%s,%s,%s)",
            ("cg-cast", "n1", "function", TENANT_1),
        )
        await conn.commit()

        cur = await conn.execute(
            "SELECT tenant_id, tenant_id::text FROM nodes WHERE code_graph_id=%s", ("cg-cast",)
        )
        raw, cast = await cur.fetchone()

    assert isinstance(raw, uuid_mod.UUID), (
        f"expected uuid.UUID from an uncast read, got {type(raw)}"
    )
    assert raw != TENANT_1, "a UUID compares False against the string it prints as"
    assert cast == TENANT_1
    assert isinstance(cast, str)
