"""py2rust #6-lite: a multi-repo workspace is partitioned by repository.

Layout: the Rust target is the primary repo at the workspace root, each source
repo is a checkout at ``.repos/<owner>__<repo>``. Every workspace walk (a)
EXCLUDES ``.repos/`` by default, (b) takes a ``repo`` selector to analyse one
source checkout instead, and (c) attributes each file-level result to its repo.

Everything below runs the real commands (``find``, ``ruff``, ``sh``) on a local
copy of ``tests/fixtures/py2rust_workspace`` mapped onto ``/workspace``
(:class:`_LocalExec`), and asserts on what comes back.
"""

from __future__ import annotations

import asyncio
import importlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from processing.architecture.container_fs import materialize_from_container
from processing.workspace_repos import (
    is_repos_path,
    normalize_repo,
    repo_of_path,
    repo_root,
    repos_exclude_globs,
)
from tests.unit.quality_store_fakes import FakeQualityResultStore
from tests.unit.test_py2rust_rust_quality import _LocalExec

_TENANT = "0195f48b-0000-7000-8000-00000000000a"
_SRC = ".repos/acme__legacy"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "py2rust_workspace"
SVC_A = ".repos/acme__svc_a"
SVC_A_PY = {
    "alembic/versions/0001_init.py",
    "app/__init__.py",
    "app/api.py",
    "app/client.py",
    "app/cyc_a.py",
    "app/cyc_b.py",
    "app/main.py",
    "app/models.py",
    "app/plugins.py",
    "app/repo.py",
    "app/tasks.py",
    "app/worker.py",
}


@pytest.fixture
def ws(tmp_path: Path) -> _LocalExec:
    root = tmp_path / "ws"
    shutil.copytree(FIXTURE, root)
    return _LocalExec(root, tmp_path / "target")


def _tree(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


# ---------------------------------------------------------------------------
# The selector
# ---------------------------------------------------------------------------


class TestSelector:
    @pytest.mark.parametrize("value", [None, "", "primary", "/", "."])
    def test_primary(self, value):
        assert normalize_repo(value) == "primary"

    @pytest.mark.parametrize(
        "value", ["acme__legacy", ".repos/acme__legacy", "/.repos/acme__legacy/"]
    )
    def test_source_repo(self, value):
        assert normalize_repo(value) == _SRC

    @pytest.mark.parametrize(
        "value", ["..", ".repos/..", ".repos/a/b", "a/b", ".hidden", "../x", "a..b", ".repos/"]
    )
    def test_rejected(self, value):
        with pytest.raises(ValueError):
            normalize_repo(value)

    def test_roots_and_attribution(self):
        assert repo_root("/workspace", None) == "/workspace"
        assert repo_root("/workspace/", _SRC) == f"/workspace/{_SRC}"
        assert repos_exclude_globs("/workspace") == ("/workspace/.repos", "/workspace/.repos/*")
        assert repo_of_path("src/a.rs") == "primary"
        assert repo_of_path(f"./{_SRC}/x.py") == _SRC
        assert repo_of_path("x.py", _SRC) == _SRC
        assert is_repos_path(".repos/x/y") and not is_repos_path("repos/x")


# ---------------------------------------------------------------------------
# The shared find-based mirror
# ---------------------------------------------------------------------------


class TestMaterialize:
    async def test_primary_excludes_repos(self, ws):
        out = await materialize_from_container(ws, "ws", "/workspace", ("*.py", "*.rs"))
        assert out is not None
        tree = _tree(out)
        assert "src/main.rs" in tree and not any(is_repos_path(p) for p in tree)

    async def test_negative_control_exclusion_off_lumps_them_in(self, ws):
        out = await materialize_from_container(
            ws, "ws", "/workspace", ("*.py", "*.rs"), exclude_repos=False
        )
        assert out is not None and f"{SVC_A}/app/main.py" in _tree(out)

    async def test_source_repo_root(self, ws):
        out = await materialize_from_container(ws, "ws", f"/workspace/{SVC_A}", ("*.py",))
        assert out is not None and _tree(out) == SVC_A_PY

    async def test_explicit_scope_into_repos_is_honoured(self, ws):
        out = await materialize_from_container(
            ws, "ws", "/workspace", ("*.py",), scope_paths=[f"{SVC_A}/app/cyc_a.py"]
        )
        assert out is not None and _tree(out) == {f"{SVC_A}/app/cyc_a.py"}


# ---------------------------------------------------------------------------
# Quality: analyzer (real ruff) + POST /quality/analyze-sync
# ---------------------------------------------------------------------------

_PATCH_A = "processing.architecture.quality_analyzer.get_coding_containers_client"
_PATCH_T = "processing.architecture.tool_execution._base.get_coding_containers_client"


async def _analyze(ws: _LocalExec, **kwargs: Any) -> Any:
    from processing.architecture.quality_analyzer import ArchitectureQualityAnalyzer

    with patch(_PATCH_A, return_value=ws), patch(_PATCH_T, return_value=ws):
        return await ArchitectureQualityAnalyzer().analyze(
            Path("/workspace"), "ws", _TENANT, tools=["ruff"], **kwargs
        )


def _ruff(report: Any) -> set[tuple[str, str, str]]:
    return {(Path(i.file_path).name, i.rule_id, i.repo) for i in report.issues if i.tool == "ruff"}


class TestQualityRepoSelector:
    async def test_source_repo_run(self, ws):
        report = await _analyze(ws, repo=SVC_A)
        assert ("cyc_a.py", "F401", SVC_A) in _ruff(report)
        assert {i.repo for i in report.issues} == {SVC_A}

    async def test_primary_scope_into_repos_is_attributed(self, ws):
        report = await _analyze(ws, scope_paths=[f"{SVC_A}/app/cyc_a.py"])
        assert _ruff(report) == {("cyc_a.py", "F401", SVC_A)}

    def test_route_repo_is_threaded_echoed_and_not_stored(self, ws):
        from coding_client.api.exception_handlers import register_exception_handlers

        from app.api.endpoints.quality import get_quality_analyzer, router
        from auth.tenant import require_tenant
        from processing.architecture.quality_analyzer import ArchitectureQualityAnalyzer
        from services.quality_result_store import get_quality_result_store

        store = FakeQualityResultStore()
        app = FastAPI()
        register_exception_handlers(app)
        app.include_router(router, prefix="/quality")
        app.dependency_overrides[require_tenant] = lambda: _TENANT
        app.dependency_overrides[get_quality_analyzer] = ArchitectureQualityAnalyzer
        app.dependency_overrides[get_quality_result_store] = lambda: store
        container = SimpleNamespace(codebase_path="/workspace", repo_url=None)
        with (
            patch(
                "app.api.endpoints.quality.validate_and_get_container_for_tenant",
                new_callable=AsyncMock,
                return_value=container,
            ),
            patch("app.api.endpoints.quality.resolve_commit_sha", new_callable=AsyncMock),
            patch(_PATCH_A, return_value=ws),
            patch(_PATCH_T, return_value=ws),
        ):
            client = TestClient(app)
            r = client.post(
                "/quality/analyze-sync", json={"container_name": "ws", "repo": "acme__svc_a"}
            )
            bad = client.post(
                "/quality/analyze-sync", json={"container_name": "ws", "repo": "../x"}
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["repo"] == SVC_A
        assert any(i["rule_id"] == "F401" for i in body["issues"])
        assert {i["repo"] for i in body["issues"]} == {SVC_A}
        assert body["persisted"] is False and store.save_calls == []
        assert bad.status_code == 422


# ---------------------------------------------------------------------------
# Contracts: POST /api/analysis/contracts/{call-sites,routes}
# ---------------------------------------------------------------------------


@pytest.fixture
def contracts(monkeypatch, ws):
    sys.modules.pop("app.api.endpoints.contract_analysis", None)
    module = importlib.import_module("app.api.endpoints.contract_analysis")
    import app.api.utils.container_validation as cv
    from auth.tenant import require_tenant

    monkeypatch.setattr(
        cv,
        "validate_and_get_container_for_tenant",
        AsyncMock(return_value=SimpleNamespace(codebase_path="/workspace")),
    )
    monkeypatch.setattr(module, "get_coding_containers_client", lambda: ws)
    app = FastAPI()
    app.include_router(module.router, prefix="/c")
    app.dependency_overrides[require_tenant] = lambda: "t"
    return TestClient(app)


def _route_files(body: dict) -> set[tuple[str, str]]:
    return {(r["file"], r["repo"]) for r in body["routes"]}


class TestContractsRepoSelector:
    def test_primary_excludes_repos(self, contracts):
        body = contracts.post("/c/routes", json={"container_name": "ws"}).json()
        assert body["repo"] == "primary"
        assert _route_files(body) and {repo for _, repo in _route_files(body)} == {"primary"}
        assert all(f.startswith("src/") for f, _ in _route_files(body))

    def test_source_repo(self, contracts):
        body = contracts.post("/c/routes", json={"container_name": "ws", "repo": SVC_A}).json()
        assert body["repo"] == SVC_A
        assert ("app/api.py", SVC_A) in _route_files(body)
        assert {repo for _, repo in _route_files(body)} == {SVC_A}

    def test_explicit_paths_into_repos_are_attributed(self, contracts):
        body = contracts.post(
            "/c/routes", json={"container_name": "ws", "paths": ["src", f"{SVC_A}/app"]}
        ).json()
        assert {repo for _, repo in _route_files(body)} == {"primary", SVC_A}
        assert (f"{SVC_A}/app/api.py", SVC_A) in _route_files(body)

    def test_repo_with_inline_sources_is_422(self, contracts):
        r = contracts.post(
            "/c/call-sites", json={"sources": [{"path": "a.py", "source": "x"}], "repo": _SRC}
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Every other workspace walker
# ---------------------------------------------------------------------------


async def _sh(ws: _LocalExec, command: str) -> dict[str, Any]:
    """``execute_in_container`` (a shell command string) run for real on the copy."""
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        ws._map(command),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    return {
        "success": proc.returncode == 0,
        "output": out.decode().replace(str(ws.root), "/workspace"),
        "exit_code": proc.returncode,
    }


class TestOtherWalkers:
    async def test_workspace_files_repo_selector(self, ws):
        from app.api.utils import workspace_files

        with patch.object(workspace_files, "get_coding_containers_client", return_value=ws):
            primary = await workspace_files.materialize_workspace(
                "ws", tenant_id="t", name_patterns=("*.py", "*.rs")
            )
            source = await workspace_files.materialize_workspace(
                "ws", tenant_id="t", name_patterns=("*.py",), repo=SVC_A
            )
        try:
            assert "src/main.rs" in _tree(primary)
            assert not any(is_repos_path(p) for p in _tree(primary))
            assert _tree(source) == SVC_A_PY
        finally:
            workspace_files.remove_materialized(primary)
            workspace_files.remove_materialized(source)

    def test_static_semantic_similarity_complexity_batch_accept_repo(self):
        from app.api.endpoints.complexity import BatchComplexityRequest, ComplexityAnalysisRequest
        from app.api.endpoints.semantic_analysis import SemanticAnalysisRequest
        from app.api.endpoints.similarity_modules.similarity_config import (
            CloneDetectionRequest,
            CrossLanguagePatternRequest,
            SimilarityAnalysisRequest,
        )
        from app.api.models.analysis_api_models import BatchAnalysisRequest
        from models.static_analysis_models import (
            CallGraphRequest,
            DocumentationRequest,
            SignatureAnalysisRequest,
        )

        for model in (
            BatchComplexityRequest,
            ComplexityAnalysisRequest,
            SemanticAnalysisRequest,
            SimilarityAnalysisRequest,
            CloneDetectionRequest,
            CrossLanguagePatternRequest,
            BatchAnalysisRequest,
            CallGraphRequest,
            DocumentationRequest,
            SignatureAnalysisRequest,
        ):
            assert model.model_fields["repo"].default is None, model
            with pytest.raises(ValidationError):
                model.model_validate({"container_name": "w", "repo": "../x"})

    async def test_complexity_lists_a_source_repo_root(self, ws):
        """The old unanchored ``*/.*`` exclusion matched every file of a
        ``.repos/<name>`` root, so a source repo listed nothing."""
        from app.api.endpoints import complexity

        svc = SimpleNamespace(
            exec_in_container=lambda name, argv, **kw: ws.execute(name, argv, **kw)
        )
        src = await complexity._list_container_code_files(
            svc, "ws", f"/workspace/{SVC_A}", tenant_id="t"
        )
        primary = await complexity._list_container_code_files(
            svc, "ws", "/workspace", tenant_id="t"
        )
        assert {p.removeprefix(f"/workspace/{SVC_A}/") for p in src} == SVC_A_PY
        assert "/workspace/src/main.rs" in primary
        assert not any("/.repos/" in p for p in primary)

    async def test_scanner_discovery_excludes_repos(self, ws):
        from app.api.endpoints.repo_scanner_modules import file_discovery

        async def run(container_name, command, **_):
            return await _sh(ws, command)

        with patch.object(file_discovery, "execute_in_container", side_effect=run):
            py = await file_discovery.discover_python_files("ws", tenant_id="t")
            src = await file_discovery.discover_source_files("ws", tenant_id="t")
        assert py["python_files"] == []  # the primary is Rust; its Python is under .repos/
        assert "/workspace/src/main.rs" in src["source_files"]
        assert not any("/.repos/" in p for p in src["source_files"])

    async def test_manifest_discovery_excludes_repos(self, ws):
        from app.api.endpoints.repo_scanner_modules import multi_language_scan

        async def run(container_name, command, **_):
            return await _sh(ws, command)

        with patch.object(multi_language_scan, "execute_in_container", side_effect=run):
            found = await multi_language_scan._discover_manifest_files("ws", tenant_id="t")
        assert found == ["/workspace/Cargo.toml"]
