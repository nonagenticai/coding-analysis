"""Unit tests for services.quality_result_store (no live database, #106).

The pool/connection is faked; these pin the SQL shape (tenant scoping, newest
first), the row shaping (status, sentinel nulling, issue cap, aggregates) and
the commit-sha provenance rules. The real round-trip against migration 0007 is
tests/integration/test_quality_result_store_integration.py.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services import quality_result_store as qrs
from services.quality_result_store import (
    MAX_STORED_ISSUES,
    QualityResultStore,
    build_row,
    derive_status,
    resolve_commit_sha,
    sanitize_repo_url,
    validate_commit_sha,
)
from tests.unit.quality_store_fakes import make_issue, make_report

TENANT = "0195f48b-0000-7000-8000-00000000000a"
SHA = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret


def _fake_connection(fetchone=None, fetchall=None):
    """A fake pooled connection that records the pool timeout and txn nesting.

    ``conn.events`` is the ordered log: ``("acquire", timeout)``, ``"begin"``,
    ``("execute", sql)``, ``"end"`` — so a test can prove every statement ran
    INSIDE the transaction the ``SET LOCAL`` was issued in.
    """
    cursor = MagicMock()
    cursor.fetchone = AsyncMock(return_value=fetchone)
    cursor.fetchall = AsyncMock(return_value=fetchall or [])
    conn = MagicMock()
    conn.events = []

    async def _execute(sql, params=None):
        conn.events.append(("execute", sql))
        return cursor

    conn.execute = AsyncMock(side_effect=_execute)
    conn.commit = AsyncMock()

    @asynccontextmanager
    async def _transaction():
        conn.events.append("begin")
        yield
        conn.events.append("end")

    conn.transaction = _transaction

    @asynccontextmanager
    async def _get(timeout=None):
        conn.events.append(("acquire", timeout))
        yield conn

    return conn, _get


def _executed_sql(conn) -> list[str]:
    return [
        " ".join(e[1].split()) for e in conn.events if isinstance(e, tuple) and e[0] == "execute"
    ]


# ---------------------------------------------------------------------------
# status / row shaping
# ---------------------------------------------------------------------------


class TestDeriveStatus:
    def test_none_is_failed(self):
        assert derive_status(None) == "failed"

    def test_inconclusive(self):
        assert derive_status(make_report(inconclusive=True, tools_succeeded=[])) == "inconclusive"

    def test_partial(self):
        report = make_report(
            partial=True, tools_attempted=["a", "b"], tools_succeeded=["a"], tools_failed=["b"]
        )
        assert derive_status(report) == "partial"

    def test_complete(self):
        assert derive_status(make_report()) == "complete"

    def test_every_derived_status_is_allowed_by_the_check_constraint(self):
        assert {
            derive_status(None),
            derive_status(make_report()),
            derive_status(make_report(inconclusive=True)),
            derive_status(make_report(partial=True)),
        } == qrs.QUALITY_STATUSES


class TestBuildRow:
    def test_failed_row_has_no_score_or_grade(self):
        row = build_row(None, tenant_id=TENANT, container_name="c1", repo_url=None, commit_sha=SHA)
        assert row["status"] == "failed"
        assert row["overall_score"] is None
        assert row["grade"] is None
        assert row["error_message"] == "Analysis produced no results"
        assert row["commit_sha"] == SHA

    def test_inconclusive_sentinel_is_stored_as_null(self):
        row = build_row(
            make_report(inconclusive=True, tools_succeeded=[], tools_failed=["ruff"]),
            tenant_id=TENANT,
            container_name="c1",
            repo_url=None,
            commit_sha=None,
        )
        assert row["status"] == "inconclusive"
        assert row["overall_score"] is None  # never the -1.0 sentinel
        assert row["grade"] == "N/A"

    def test_issues_capped_and_aggregates_use_the_full_list(self):
        n = MAX_STORED_ISSUES + 25
        issues = [make_issue("High", f"dir/f{i % 4}.py", line=i) for i in range(n)]
        row = build_row(
            make_report(issues=issues),
            tenant_id=TENANT,
            container_name="c1",
            repo_url=None,
            commit_sha=None,
        )
        assert len(row["issues"]) == MAX_STORED_ISSUES
        assert row["issues_total"] == n
        assert row["issues_truncated"] is True
        assert row["issue_stats"]["severity_counts"] == {"high": n}
        assert row["issue_stats"]["files_with_issues"] == 4
        assert sum(row["issue_stats"]["file_counts"].values()) == n

    def test_not_truncated_below_cap(self):
        row = build_row(
            make_report(issues=[make_issue()]),
            tenant_id=TENANT,
            container_name="c1",
            repo_url=None,
            commit_sha=None,
        )
        assert row["issues_truncated"] is False
        assert row["issues_total"] == 1
        assert row["issues"][0]["rule_id"] == "R001"

    def test_branch_name_is_never_stored_as_the_commit(self):
        row = build_row(
            make_report(), tenant_id=TENANT, container_name="c1", repo_url=None, commit_sha="main"
        )
        assert row["commit_sha"] is None


class TestProvenanceHelpers:
    @pytest.mark.parametrize(
        "value",
        [SHA, f"  {SHA}\n"],
    )
    def test_valid_sha(self, value):
        assert validate_commit_sha(value) == SHA

    @pytest.mark.parametrize(
        "value",
        ["main", "HEAD", SHA[:7], SHA.upper(), SHA + "0", "", None, 123, "g" * 40],
    )
    def test_invalid_sha_is_none(self, value):
        assert validate_commit_sha(value) is None

    def test_repo_url_token_is_stripped(self):
        tokenised = (
            "https://x-access-token:ghs_secret@github.com/o/r.git"  # pragma: allowlist secret
        )
        assert sanitize_repo_url(tokenised) == "https://github.com/o/r.git"

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_empty_repo_url_is_none(self, value):
        assert sanitize_repo_url(value) is None


class TestResolveCommitSha:
    @staticmethod
    def _client(head="SHA", status="", head_rc=0, status_rc=0):
        """A coding-containers fake that enforces the real exec allowlist rule
        that bit production: the first argv element must be an allowlisted
        binary — `sh` is refused with a 400, which surfaces as an exception."""

        async def _execute(**kw):
            cmd = kw["command"]
            if cmd[0] != "git":
                raise RuntimeError(f"400 Command '{cmd[0]}' not allowed")
            if "rev-parse" in cmd:
                return {"exit_code": head_rc, "stdout": (SHA if head == "SHA" else head) + "\n"}
            if "status" in cmd:
                return {"exit_code": status_rc, "stdout": status}
            raise AssertionError(f"unexpected git call {cmd}")

        client = MagicMock()
        client.execute = AsyncMock(side_effect=_execute)
        return client

    @pytest.mark.asyncio
    async def test_clean_tree_resolves_head_as_the_tenant_with_plain_git(self):
        client = self._client()
        with patch.object(qrs, "get_coding_containers_client", return_value=client):
            assert await resolve_commit_sha("c1", TENANT) == SHA
        calls = [c.kwargs for c in client.execute.await_args_list]
        assert len(calls) == 2
        for kw in calls:
            assert kw["command"][0] == "git"  # never a shell: the allowlist refuses it
            assert kw["container_name"] == "c1"
            assert kw["tenant_id"] == TENANT
            assert kw["workdir"] == "/workspace"
        assert calls[0]["command"][-2:] == ["rev-parse", "HEAD"]
        assert calls[1]["command"][-2:] == ["status", "--porcelain"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_lines",
        [" M src/app.py\n", "?? src/new_module.py\n", "A  staged.py\n"],
    )
    async def test_dirty_tree_is_unknown_commit(self, status_lines):
        """Edited-but-uncommitted code is NOT HEAD: never key it to the sha."""
        with patch.object(
            qrs, "get_coding_containers_client", return_value=self._client(status=status_lines)
        ):
            assert await resolve_commit_sha("c1", TENANT) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kw",
        [
            {"head_rc": 128},  # not a git repository
            {"head": "main"},  # not a sha
            {"status_rc": 1},  # cleanliness unknown -> commit unknown
        ],
    )
    async def test_unusable_output_is_none(self, kw):
        with patch.object(qrs, "get_coding_containers_client", return_value=self._client(**kw)):
            assert await resolve_commit_sha("c1", TENANT) is None

    @pytest.mark.asyncio
    async def test_slow_container_is_cut_off_at_the_timeout(self):
        import asyncio
        import time

        async def _hang(**kwargs):
            await asyncio.sleep(60)

        client = MagicMock()
        client.execute = AsyncMock(side_effect=_hang)
        with (
            patch.object(qrs, "get_coding_containers_client", return_value=client),
            patch.object(qrs, "COMMIT_SHA_TIMEOUT_S", 0.05),
        ):
            started = time.monotonic()
            assert await resolve_commit_sha("c1", TENANT) is None
        assert time.monotonic() - started < 2

    def test_default_timeout_is_five_seconds(self):
        assert qrs.COMMIT_SHA_TIMEOUT_S == 5.0

    @pytest.mark.asyncio
    async def test_exec_exception_is_none_not_raised(self):
        client = MagicMock()
        client.execute = AsyncMock(side_effect=OSError("connection refused"))
        with patch.object(qrs, "get_coding_containers_client", return_value=client):
            assert await resolve_commit_sha("c1", TENANT) is None


# ---------------------------------------------------------------------------
# SQL shape
# ---------------------------------------------------------------------------


def _db_row(**over):
    base = {
        "id": "r1",
        "tenant_id": TENANT,
        "container_name": "c1",
        "repo_url": None,
        "commit_sha": SHA,
        "status": "complete",
        "overall_score": 91.0,
        "grade": "A",
        "language": "python",
        "tools_attempted": ["ruff"],
        "tools_succeeded": ["ruff"],
        "tools_failed": [],
        "summary": {},
        "issue_stats": {},
        "issues": [],
        "issues_total": 0,
        "issues_truncated": False,
        "recommendations": [],
        "execution_time": 1.0,
        "error_message": None,
        "created_at": datetime(2026, 9, 30, tzinfo=UTC),
    }
    base.update(over)
    return tuple(base[name] for name in qrs._COLUMNS)


class TestStoreSql:
    @pytest.mark.asyncio
    async def test_save_inserts_one_row_and_commits(self):
        conn, get = _fake_connection()
        with patch.object(qrs, "get_raw_connection", get):
            result_id = await QualityResultStore().save(
                make_report(issues=[make_issue()]),
                tenant_id=TENANT,
                container_name="c1",
                repo_url="https://github.com/o/r",
                commit_sha=SHA,
            )
        sql, params = conn.execute.await_args.args
        assert " ".join(sql.split()).startswith("INSERT INTO quality_results (")
        columns = [c.strip() for c in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
        # The static column list, the placeholder count and the param tuple agree.
        assert tuple(columns) == qrs._INSERT_COLUMNS
        assert sql.count("%s") == len(columns) == len(params)
        assert "created_at" not in columns  # server default now()
        values = dict(zip(columns, params, strict=True))
        assert values["id"] == result_id
        assert values["tenant_id"] == TENANT
        assert values["commit_sha"] == SHA
        assert values["status"] == "complete"
        assert values["issues_total"] == 1
        # The insert commits via the transaction block, not a separate commit().
        assert conn.events[0] == ("acquire", qrs.STORE_POOL_TIMEOUT_S)
        assert conn.events[1] == "begin"
        assert conn.events[-1] == "end"

    @pytest.mark.asyncio
    async def test_save_none_inserts_a_failed_row(self):
        conn, get = _fake_connection()
        with patch.object(qrs, "get_raw_connection", get):
            await QualityResultStore().save(
                None, tenant_id=TENANT, container_name="c1", repo_url=None, commit_sha=None
            )
        sql, params = conn.execute.await_args.args
        columns = [c.strip() for c in sql.split("(", 1)[1].split(")", 1)[0].split(",")]
        values = dict(zip(columns, params, strict=True))
        assert values["status"] == "failed"
        assert values["overall_score"] is None

    @pytest.mark.asyncio
    async def test_latest_is_tenant_scoped_and_newest_first(self):
        conn, get = _fake_connection(fetchone=_db_row())
        with patch.object(qrs, "get_raw_connection", get):
            row = await QualityResultStore().latest(TENANT, "c1")
        sql, params = conn.execute.await_args.args
        normalized = " ".join(sql.split())
        assert "FROM quality_results" in normalized
        assert "WHERE tenant_id = %s AND container_name = %s" in normalized
        assert "ORDER BY created_at DESC" in normalized
        assert "LIMIT 1" in normalized
        assert "tenant_id::text" in normalized
        assert params == (TENANT, "c1")
        assert row["id"] == "r1"
        assert row["commit_sha"] == SHA

    @pytest.mark.asyncio
    async def test_save_bounds_statements_with_set_local_never_a_session_set(self):
        """pgbouncer transaction pooling: a session-level SET would leak to the
        next client on that backend. Only SET LOCAL, first, inside the txn."""
        conn, get = _fake_connection()
        with patch.object(qrs, "get_raw_connection", get):
            await QualityResultStore().save(
                make_report(), tenant_id=TENANT, container_name="c1", repo_url=None, commit_sha=SHA
            )
        sqls = _executed_sql(conn)
        assert sqls[0] == "SET LOCAL statement_timeout = '5s'"
        assert sqls[1].startswith("INSERT INTO quality_results")
        for sql in sqls:
            upper = sql.upper()
            assert not (upper.startswith("SET ") and not upper.startswith("SET LOCAL ")), sql
        # Both statements ran inside the one transaction.
        begin, end = conn.events.index("begin"), conn.events.index("end")
        exec_positions = [
            i for i, e in enumerate(conn.events) if isinstance(e, tuple) and e[0] == "execute"
        ]
        assert all(begin < i < end for i in exec_positions)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["latest", "latest_for_sha", "history"])
    async def test_reads_are_bounded_too(self, method):
        conn, get = _fake_connection(fetchone=None)
        args = (TENANT, "c1", SHA) if method == "latest_for_sha" else (TENANT, "c1")
        with patch.object(qrs, "get_raw_connection", get):
            await getattr(QualityResultStore(), method)(*args)
        assert conn.events[0] == ("acquire", qrs.STORE_POOL_TIMEOUT_S)
        assert _executed_sql(conn)[0] == "SET LOCAL statement_timeout = '5s'"

    @pytest.mark.asyncio
    async def test_latest_for_sha_is_tenant_and_commit_scoped(self):
        conn, get = _fake_connection(fetchone=_db_row())
        with patch.object(qrs, "get_raw_connection", get):
            row = await QualityResultStore().latest_for_sha(TENANT, "c1", SHA)
        sql, params = conn.execute.await_args.args
        normalized = " ".join(sql.split())
        assert "WHERE tenant_id = %s AND container_name = %s AND commit_sha = %s" in normalized
        assert "ORDER BY created_at DESC" in normalized
        assert "LIMIT 1" in normalized
        assert params == (TENANT, "c1", SHA)
        assert row["commit_sha"] == SHA

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["latest", "latest_for_sha", "history"])
    async def test_non_uuid_tenant_is_refused_before_any_sql(self, method):
        conn, get = _fake_connection()
        args = ("not-a-uuid", "c1", SHA) if method == "latest_for_sha" else ("not-a-uuid", "c1")
        with patch.object(qrs, "get_raw_connection", get), pytest.raises(qrs.InvalidTenantIdError):
            await getattr(QualityResultStore(), method)(*args)
        assert conn.events == []

    @pytest.mark.asyncio
    async def test_latest_none_when_nothing_stored(self):
        _, get = _fake_connection(fetchone=None)
        with patch.object(qrs, "get_raw_connection", get):
            assert await QualityResultStore().latest(TENANT, "c1") is None

    @pytest.mark.asyncio
    async def test_history_is_tenant_scoped_limited_and_windowed(self):
        since = datetime(2026, 9, 1, tzinfo=UTC)
        conn, get = _fake_connection(fetchall=[_db_row(id="r2"), _db_row(id="r1")])
        with patch.object(qrs, "get_raw_connection", get):
            rows = await QualityResultStore().history(TENANT, "c1", limit=7, since=since)
        sql, params = conn.execute.await_args.args
        normalized = " ".join(sql.split())
        assert "WHERE tenant_id = %s AND container_name = %s" in normalized
        assert "ORDER BY created_at DESC" in normalized
        assert params == (TENANT, "c1", since, since, 7)
        assert [r["id"] for r in rows] == ["r2", "r1"]
