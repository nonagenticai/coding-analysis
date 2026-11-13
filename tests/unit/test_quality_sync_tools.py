"""``POST /quality/analyze-sync`` honours ``tools``.

Measured live: ``tools`` was accepted on the request model and never passed on,
so ``["mypy"]`` (removed) came back 200 ``success: true`` having run the default
set, and ``["ruff", "ty"]`` silently ran bandit + radon as well. These tests pin
that (a) unknown names and ``[]`` are a 422 on ``tools`` that runs nothing, (b) a
non-empty list runs exactly those tools and the report carries nothing else,
(c) omitting ``tools`` runs the unchanged default set, and (d) a subset run is
not stored, so it cannot become the container's gate verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.endpoints.quality import QualityAnalysisSyncRequest, run_quality_analysis_sync
from models.architecture_models import ArchitectureQualityIssue
from processing.architecture.architecture_pattern_analysis import enabled_tools_by_language
from processing.architecture.quality_analyzer import (
    ArchitectureQualityAnalyzer,
    ToolsNotApplicableError,
)
from tests.unit.quality_store_fakes import FakeQualityResultStore

_VALIDATE_CONTAINER = "app.api.endpoints.quality.validate_and_get_container_for_tenant"
_RESOLVE_SHA = "app.api.endpoints.quality.resolve_commit_sha"
_ANALYZER_CLIENT = "processing.architecture.quality_analyzer.get_coding_containers_client"
_TOOL_CLIENT = "processing.architecture.tool_execution._base.get_coding_containers_client"
_TENANT = "0195f48b-0000-7000-8000-00000000000b"

_PY_DEFAULT = ["ruff", "ty", "bandit", "radon", "ruff-format"]
_REMOVED = ["mypy", "pylint", "pyright", "flake8", "isort", "black", "autoflake"]


def _tool_of(command: list[str]) -> str | None:
    """Registry key of a tool argv the analyzer sent (None for find etc.)."""
    head = command[0]
    if head == "ruff":
        return "ruff-format" if command[1] == "format" else "ruff"
    if head in ("ty", "bandit", "radon"):
        return head
    return None


class _FakeWorkspace:
    """In-memory ``/workspace`` with two Python files; every tool finds one issue
    per file it can, so a tool that ran is visible in the report."""

    files = ("/workspace/app/a.py", "/workspace/app/b.py")

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    async def execute(self, container_name: str, command: list[str], **_: Any) -> dict[str, Any]:
        self.commands.append(list(command))
        tool = _tool_of(command)
        if command[0] == "find":
            return {"exit_code": 0, "stdout": "\n".join(self.files), "stderr": ""}
        if tool == "ruff":
            rows = [f"{f}:1:1: F401 [*] `os` imported but unused" for f in self.files]
            return {"exit_code": 1, "stdout": "\n".join(rows), "stderr": ""}
        if tool == "ty":
            return {"exit_code": 0, "stdout": "[]", "stderr": ""}
        if tool == "bandit":
            return {"exit_code": 0, "stdout": '{"results": [], "errors": []}', "stderr": ""}
        if tool == "radon":
            return {"exit_code": 0, "stdout": "{}", "stderr": ""}
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    async def read_file(self, container_name: str, file_path: str, **_: Any) -> str:
        return "import os\n"

    def tools_run(self) -> set[str]:
        return {t for c in self.commands if (t := _tool_of(c)) is not None}


def _perf_issue() -> ArchitectureQualityIssue:
    return ArchitectureQualityIssue(
        id="perf-1",
        file_path="/workspace/app/a.py",
        line_number=1,
        severity="low",
        category="performance",
        message="loop",
        rule_id="PERF001",
        rule_name="loop",
        tool="performance_analyzer",
    )


def _analyzer() -> ArchitectureQualityAnalyzer:
    """The REAL analyzer + registry; only the perf scan is stubbed so its issue
    is a deterministic marker of whether it ran."""
    analyzer = ArchitectureQualityAnalyzer()
    analyzer.performance_analyzer.analyze_performance_patterns = MagicMock(  # type: ignore[method-assign]
        return_value=[_perf_issue()]
    )
    return analyzer


def _container() -> MagicMock:
    c = MagicMock()
    c.codebase_path = "/workspace"
    c.repo_url = "https://github.com/acme/repo"
    return c


def _client(analyzer: Any, store: FakeQualityResultStore):
    from coding_client.api.exception_handlers import register_exception_handlers
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.endpoints.quality import get_quality_analyzer, router
    from auth.tenant import require_tenant
    from services.quality_result_store import get_quality_result_store

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router, prefix="/quality")
    app.dependency_overrides[require_tenant] = lambda: _TENANT
    app.dependency_overrides[get_quality_analyzer] = lambda: analyzer
    app.dependency_overrides[get_quality_result_store] = lambda: store
    return TestClient(app)


def _post(body: dict[str, Any], ws: _FakeWorkspace, store: FakeQualityResultStore) -> Any:
    with (
        patch(_VALIDATE_CONTAINER, new_callable=AsyncMock, return_value=_container()),
        patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=None),
        patch(_ANALYZER_CLIENT, return_value=ws),
        patch(_TOOL_CLIENT, return_value=ws),
    ):
        return _client(_analyzer(), store).post(
            "/quality/analyze-sync", json={"container_name": "ws-1", **body}
        )


def test_registry_is_the_accepted_python_set():
    """The accepted names ARE the registry; none of the removed tools is in it."""
    assert enabled_tools_by_language()["python"] == _PY_DEFAULT
    assert not set(_REMOVED) & {t for ts in enabled_tools_by_language().values() for t in ts}


# ---------------------------------------------------------------------------
# Endpoint over HTTP, real analyzer
# ---------------------------------------------------------------------------


class TestRejectedTools:
    @pytest.mark.parametrize("name", _REMOVED)
    def test_removed_tool_is_422_and_runs_nothing(self, name: str):
        ws, store = _FakeWorkspace(), FakeQualityResultStore()
        r = _post({"tools": [name]}, ws, store)
        assert r.status_code == 422, r.text
        err = r.json()["error"]
        assert err["details"]["field"] == "tools"
        assert err["details"]["unknown"] == [name]
        assert set(_PY_DEFAULT) <= set(err["details"]["accepted"])
        assert name in err["message"]
        assert ws.commands == []
        assert store.save_calls == []

    def test_mixed_known_and_unknown_names_only_the_unknown(self):
        ws, store = _FakeWorkspace(), FakeQualityResultStore()
        r = _post({"tools": ["ruff", "mypy", "pylint"], "language": "python"}, ws, store)
        assert r.status_code == 422, r.text
        details = r.json()["error"]["details"]
        assert details["unknown"] == ["mypy", "pylint"]
        # Pre-validation is against EVERY language's registry, listed per language.
        assert details["accepted_by_language"] == enabled_tools_by_language()
        assert set(_PY_DEFAULT) <= set(details["accepted"])
        assert ws.commands == []

    def test_empty_list_is_422_not_a_clean_run(self):
        ws, store = _FakeWorkspace(), FakeQualityResultStore()
        r = _post({"tools": []}, ws, store)
        assert r.status_code == 422, r.text
        assert r.json()["error"]["details"]["field"] == "tools"
        assert ws.commands == []
        assert store.save_calls == []

    def test_pinned_language_tool_not_in_detected_language_is_422_after_detection(self):
        """``eslint`` passes pre-validation (a known name, and the caller pinned
        javascript) but the workspace is Python: 422 from detection, no tool run."""
        ws, store = _FakeWorkspace(), FakeQualityResultStore()
        r = _post({"tools": ["eslint"], "language": "javascript"}, ws, store)
        assert r.status_code == 422, r.text
        details = r.json()["error"]["details"]
        assert details["unknown"] == ["eslint"]
        assert details["language"] == "python"
        assert any(c[0] == "find" for c in ws.commands)  # got past pre-validation
        assert ws.tools_run() == set()
        assert store.save_calls == []

    def test_tool_of_another_language_is_422_after_detection(self):
        """``eslint`` is a known name, but the workspace is Python: refuse, never
        run nothing and report success."""
        ws, store = _FakeWorkspace(), FakeQualityResultStore()
        r = _post({"tools": ["ruff", "eslint"]}, ws, store)
        assert r.status_code == 422, r.text
        details = r.json()["error"]["details"]
        assert details["unknown"] == ["eslint"]
        assert details["language"] == "python"
        assert details["accepted"] == _PY_DEFAULT
        assert ws.tools_run() == set()
        assert store.save_calls == []


class TestRestrictedRun:
    def test_ruff_and_ty_run_only_ruff_and_ty(self):
        ws, store = _FakeWorkspace(), FakeQualityResultStore()
        r = _post({"tools": ["ruff", "ty"]}, ws, store)
        assert r.status_code == 200, r.text
        body = r.json()
        assert ws.tools_run() == {"ruff", "ty"}
        assert set(body["tools_attempted"]) == {"ruff", "ty"}
        assert set(body["tools_results"]) == {"ruff", "ty"}
        assert {i["tool"] for i in body["issues"]} == {"ruff"}
        assert body["total_issues"] == 2  # one ruff issue per file; no perf finding
        assert body["tools_requested"] == ["ruff", "ty"]
        assert body["success"] is True
        # A subset of the default tools is not the container's full result.
        assert body["persisted"] is False
        assert store.save_calls == []

    def test_single_tool_duplicates_collapse(self):
        ws, store = _FakeWorkspace(), FakeQualityResultStore()
        r = _post({"tools": ["ty", "ty"]}, ws, store)
        assert r.status_code == 200, r.text
        assert ws.tools_run() == {"ty"}
        assert r.json()["tools_attempted"] == ["ty"]
        assert r.json()["issues"] == []

    def test_tools_and_paths_compose(self):
        ws, store = _FakeWorkspace(), FakeQualityResultStore()
        r = _post({"tools": ["ruff"], "paths": ["app/a.py"]}, ws, store)
        assert r.status_code == 200, r.text
        body = r.json()
        assert ws.tools_run() == {"ruff"}
        assert body["scope"] == ["app/a.py"]
        ruff = next(c for c in ws.commands if _tool_of(c) == "ruff")
        assert "/workspace/app/a.py" in ruff and "/workspace/app/b.py" not in ruff
        assert {i["file_path"] for i in body["issues"]} == {"/workspace/app/a.py"}

    def test_full_default_set_named_explicitly_is_not_stored(self):
        """Naming every default tool still skips the performance scan, so the
        row would not match the default rows /gate and /history grade."""
        ws, store = _FakeWorkspace(), FakeQualityResultStore()
        r = _post({"tools": list(reversed(_PY_DEFAULT))}, ws, store)
        assert r.status_code == 200, r.text
        assert ws.tools_run() == set(_PY_DEFAULT)
        assert "performance_analyzer" not in {i["tool"] for i in r.json()["issues"]}
        assert r.json()["persisted"] is False
        assert store.save_calls == []

    def test_pinned_language_does_not_gate_names_the_detected_language_runs(self):
        """``language`` is a hint; the run uses the DETECTED language. ``ruff``
        on a Python repo must run even when the caller pinned javascript."""
        ws, store = _FakeWorkspace(), FakeQualityResultStore()
        r = _post({"tools": ["ruff"], "language": "javascript"}, ws, store)
        assert r.status_code == 200, r.text
        assert ws.tools_run() == {"ruff"}
        assert r.json()["tools_attempted"] == ["ruff"]


class TestRequestLanguage:
    """``language`` is honoured only as ``"rust"``; any other value is echoed and
    ignored exactly as on main: the detected language's tools run and the
    whole-workspace run is stored."""

    @pytest.mark.parametrize("language", ["kotlin", "ruby", "python", "javascript"])
    def test_non_rust_language_runs_detected_tools_and_is_stored(self, language: str):
        ws, store = _FakeWorkspace(), FakeQualityResultStore()
        r = _post({"language": language}, ws, store)
        assert r.status_code == 200, r.text
        body = r.json()
        assert ws.tools_run() == set(_PY_DEFAULT)
        assert body["language"] == language  # echoed, as on main
        assert body["persisted"] is True
        assert len(store.save_calls) == 1

    def test_rust_overrides_detection_and_is_stored(self):
        ws, store = _FakeWorkspace(), FakeQualityResultStore()
        r = _post({"language": "rust"}, ws, store)
        assert r.status_code == 200, r.text
        assert ws.tools_run() == set()  # no Python tool ran on the Python files
        assert {c[0] for c in ws.commands} >= {"cargo"}
        assert r.json()["persisted"] is True
        assert len(store.save_calls) == 1


class TestOmittedTools:
    def test_omitted_runs_the_unchanged_default_set_and_stores(self):
        ws, store = _FakeWorkspace(), FakeQualityResultStore()
        r = _post({}, ws, store)
        assert r.status_code == 200, r.text
        body = r.json()
        assert ws.tools_run() == set(_PY_DEFAULT)
        assert set(body["tools_attempted"]) == set(_PY_DEFAULT)
        # The built-in performance scan still contributes to a default run.
        assert "performance_analyzer" in {i["tool"] for i in body["issues"]}
        assert body["tools_requested"] is None
        assert body["persisted"] is True
        assert len(store.save_calls) == 1


# ---------------------------------------------------------------------------
# Analyzer directly
# ---------------------------------------------------------------------------


class TestAnalyzerTools:
    async def _analyze(self, ws: _FakeWorkspace, **kwargs: Any) -> Any:
        with patch(_ANALYZER_CLIENT, return_value=ws), patch(_TOOL_CLIENT, return_value=ws):
            return await _analyzer().analyze(
                Path("/workspace"), "ws-1", tenant_id=_TENANT, **kwargs
            )

    @pytest.mark.asyncio
    async def test_restricted_run_excludes_unrequested_tools_and_perf(self):
        ws = _FakeWorkspace()
        report = await self._analyze(ws, tools=["bandit"])
        assert ws.tools_run() == {"bandit"}
        assert report.tools_attempted == ["bandit"]
        assert report.issues == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tools", [[], ["mypy"], ["eslint"]])
    async def test_unrunnable_selection_raises_not_none(self, tools: list[str]):
        ws = _FakeWorkspace()
        with pytest.raises(ToolsNotApplicableError) as exc:
            await self._analyze(ws, tools=tools)
        assert exc.value.language == "python"
        assert exc.value.accepted == _PY_DEFAULT
        assert ws.tools_run() == set()

    @pytest.mark.asyncio
    async def test_none_is_the_default_set(self):
        ws = _FakeWorkspace()
        report = await self._analyze(ws)
        assert ws.tools_run() == set(_PY_DEFAULT)
        assert "performance_analyzer" in {i.tool for i in report.issues}


@pytest.mark.asyncio
async def test_handler_threads_tools_to_the_analyzer_only_when_given():
    """Existing call shape is unchanged when ``tools`` is omitted."""
    from tests.unit.quality_store_fakes import make_report

    for tools, expected in ((None, None), (["ruff", "ruff"], ["ruff"])):
        analyzer = AsyncMock()
        analyzer.analyze = AsyncMock(return_value=make_report())
        with (
            patch(_VALIDATE_CONTAINER, new_callable=AsyncMock, return_value=_container()),
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=None),
        ):
            await run_quality_analysis_sync(
                request=QualityAnalysisSyncRequest(container_name="ws-1", tools=tools),
                analyzer=analyzer,
                tenant_id=_TENANT,
                store=FakeQualityResultStore(),
            )
        kwargs = analyzer.analyze.await_args.kwargs
        assert kwargs.get("tools") == expected
        assert ("tools" in kwargs) is (tools is not None)


# ---------------------------------------------------------------------------
# What the API advertises == what analyze-sync accepts
# ---------------------------------------------------------------------------


class TestAdvertisedTools:
    def test_python_tools_endpoint_is_the_accepted_set(self):
        """``/python/tools`` lists exactly the names a ``language: python``
        analyze-sync accepts: each one is accepted, and the accepted list that a
        rejection reports is the advertised list."""
        client = _client(AsyncMock(), FakeQualityResultStore())
        r = client.get("/quality/python/tools")
        assert r.status_code == 200, r.text
        tools = r.json()["tools"]
        names = [t["name"] for t in tools]
        assert names == _PY_DEFAULT
        assert not set(names) & set(_REMOVED)
        assert all(t["category"] and t["description"] for t in tools)

        ws, store = _FakeWorkspace(), FakeQualityResultStore()
        rejected = _post({"tools": ["mypy"], "language": "python"}, ws, store)
        assert rejected.status_code == 422
        assert rejected.json()["error"]["details"]["accepted_by_language"]["python"] == names
        accepted = _post({"tools": names, "language": "python"}, ws, store)
        assert accepted.status_code == 200, accepted.text
        assert set(accepted.json()["tools_attempted"]) == set(names)

    def test_info_lists_the_registry_for_every_language(self):
        client = _client(AsyncMock(), FakeQualityResultStore())
        info = client.get("/quality/info").json()["available_tools"]
        assert info == enabled_tools_by_language()
        assert info["python"] == _PY_DEFAULT
        for lang, names in info.items():
            r = client.get(f"/quality/{lang}/tools")
            assert [t["name"] for t in r.json()["tools"]] == names, lang

    def test_languages_endpoint_is_the_registry(self):
        client = _client(AsyncMock(), FakeQualityResultStore())
        langs = client.get("/quality/languages").json()["languages"]
        assert {lang["name"]: lang["tools"] for lang in langs} == enabled_tools_by_language()
        by_name = {lang["name"]: lang for lang in langs}
        assert by_name["python"]["extensions"] == [".py"]
        assert "prettier" in by_name["javascript"]["tools"]
        assert all(" " not in t for lang in langs for t in lang["tools"])  # no "go vet"
        exts = client.get("/quality/extensions").json()["extensions"]
        assert {".py", ".js", ".ts", ".go", ".rs"} <= set(exts)

    def test_unknown_language_is_still_404(self):
        client = _client(AsyncMock(), FakeQualityResultStore())
        assert client.get("/quality/brainfuck/tools").status_code == 404
