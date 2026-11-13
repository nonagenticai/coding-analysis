"""Path-scoped ``POST /quality/analyze-sync`` (``paths``).

coding-testing's quality-scan / quality-check need a scoped run: on a workspace
holding a large clone under ``.repos/`` an unscoped call exceeded 60 s and
analysed files nobody asked about. These tests pin that a scope (a) is applied
at enumeration time, so out-of-scope files are never fetched, (b) narrows every
path-taking tool and skips the ones that cannot be narrowed, (c) reports only
in-scope issues, (d) is a 422 when it matches nothing, and (e) is never stored,
so it cannot replace the container's full-run ``latest`` used by the gate.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.api.endpoints.quality import (
    MAX_SCOPE_PATHS,
    QualityAnalysisSyncRequest,
    run_quality_analysis_sync,
)
from processing.architecture.quality_analyzer import (
    ArchitectureQualityAnalyzer,
    ScopeMatchesNoFilesError,
)
from processing.architecture.tool_execution import ToolExecution
from processing.architecture.tool_execution._base import (
    COMMA_AFTER_FLAG,
    SCOPE_MODES,
    SEPARATE,
    UNSCOPABLE,
)
from tests.unit.quality_store_fakes import FakeQualityResultStore, make_report

_VALIDATE_CONTAINER = "app.api.endpoints.quality.validate_and_get_container_for_tenant"
_RESOLVE_SHA = "app.api.endpoints.quality.resolve_commit_sha"
_ANALYZER_CLIENT = "processing.architecture.quality_analyzer.get_coding_containers_client"
_TOOL_CLIENT = "processing.architecture.tool_execution._base.get_coding_containers_client"
_TENANT = "0195f48b-0000-7000-8000-00000000000a"

_WORKSPACE_FILES = (
    "app/a.py",
    "app/b.py",
    "pkg/x.py",
    "pkg/sub/y.py",
    ".repos/big/huge.py",
)


class _FakeWorkspace:
    """A coding-containers client over an in-memory ``/workspace``.

    ``find`` honours its start points; ``ruff`` reports one violation per
    ``.py`` file under its target arguments. Every argv and read is recorded.
    """

    def __init__(self, files: tuple[str, ...] = _WORKSPACE_FILES) -> None:
        self.files = [f"/workspace/{f}" for f in files]
        self.commands: list[list[str]] = []
        self.reads: list[str] = []

    @staticmethod
    def _under(path: str, roots: list[str]) -> bool:
        return any(path == r or path.startswith(r.rstrip("/") + "/") for r in roots)

    async def execute(self, container_name: str, command: list[str], **_: Any) -> dict[str, Any]:
        self.commands.append(list(command))
        if command[0] == "find":
            roots = command[1 : command.index("-type")]
            # ``-not -path <glob>``: find's -path is fnmatch over the whole path.
            excludes = [command[i + 2] for i, a in enumerate(command) if a == "-not"]
            hits = [
                f
                for f in self.files
                if self._under(f, roots) and not any(fnmatch.fnmatchcase(f, g) for g in excludes)
            ]
            return {"exit_code": 0, "stdout": "\n".join(hits), "stderr": ""}
        if command[0] == "ruff":
            targets = command[2 : command.index("--output-format=concise")]
            rows = [
                f"{f}:1:1: F401 [*] unused import" for f in self.files if self._under(f, targets)
            ]
            return {"exit_code": 1 if rows else 0, "stdout": "\n".join(rows), "stderr": ""}
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    async def read_file(self, container_name: str, file_path: str, **_: Any) -> str:
        self.reads.append(file_path)
        return "import os\n"


def _analyzer(tools: list[str]) -> ArchitectureQualityAnalyzer:
    analyzer = ArchitectureQualityAnalyzer()
    pa = analyzer.pattern_analyzer
    pa.get_available_tools = MagicMock(return_value=tools)  # type: ignore[method-assign]
    pa.is_tool_enabled = MagicMock(return_value=True)  # type: ignore[method-assign]
    pa.get_tool_config = MagicMock(return_value={"command": []})  # type: ignore[method-assign]
    return analyzer


async def _run(ws: _FakeWorkspace, scope: list[str] | None, tools: list[str] | None = None) -> Any:
    analyzer = _analyzer(tools or ["ruff"])
    with patch(_ANALYZER_CLIENT, return_value=ws), patch(_TOOL_CLIENT, return_value=ws):
        kwargs: dict[str, Any] = {"tenant_id": _TENANT}
        if scope is not None:
            kwargs["scope_paths"] = scope
        return await analyzer.analyze(Path("/workspace"), "ws-1", **kwargs)


def _issue_files(report: Any) -> set[str]:
    return {i.file_path for i in report.issues if i.tool == "ruff"}


# ---------------------------------------------------------------------------
# Analyzer: the scope reaches enumeration, the tools and the issue list
# ---------------------------------------------------------------------------


class TestScopedAnalyzer:
    @pytest.mark.asyncio
    async def test_scoped_to_one_file_reports_only_that_file(self):
        ws = _FakeWorkspace()
        report = await _run(ws, ["app/a.py"])
        assert report is not None
        assert _issue_files(report) == {"/workspace/app/a.py"}
        # Enumeration started at the file, so nothing else was ever fetched.
        assert ws.reads == ["/workspace/app/a.py"]
        ruff = next(c for c in ws.commands if c[0] == "ruff")
        assert ruff == [
            "ruff",
            "check",
            "/workspace/app/a.py",
            "--output-format=concise",
            "--extend-select",
            "I,PL",
        ]

    @pytest.mark.asyncio
    async def test_scoped_to_a_directory(self):
        ws = _FakeWorkspace()
        report = await _run(ws, ["pkg"])
        assert _issue_files(report) == {"/workspace/pkg/x.py", "/workspace/pkg/sub/y.py"}
        assert sorted(ws.reads) == ["/workspace/pkg/sub/y.py", "/workspace/pkg/x.py"]
        assert not any(".repos" in r for r in ws.reads)

    @pytest.mark.asyncio
    async def test_scope_matching_nothing_raises(self):
        ws = _FakeWorkspace()
        with pytest.raises(ScopeMatchesNoFilesError):
            await _run(ws, ["does/not/exist.py"])
        assert not any(c[0] == "ruff" for c in ws.commands)

    @pytest.mark.asyncio
    async def test_one_unmatched_path_among_real_ones_raises_naming_it(self):
        """Each path is checked: a typo next to a real file is not a narrower run."""
        ws = _FakeWorkspace()
        with pytest.raises(ScopeMatchesNoFilesError) as exc:
            await _run(ws, ["app/a.py", "app/typo.py"])
        assert exc.value.unmatched == ["app/typo.py"]
        assert ws.reads == []
        assert [c[0] for c in ws.commands] == ["find", "find"]

    @pytest.mark.asyncio
    async def test_all_paths_matched_is_unchanged(self):
        ws = _FakeWorkspace()
        report = await _run(ws, ["app/a.py", "pkg"])
        assert _issue_files(report) == {
            "/workspace/app/a.py",
            "/workspace/pkg/x.py",
            "/workspace/pkg/sub/y.py",
        }

    @pytest.mark.asyncio
    async def test_out_of_scope_issue_from_a_tool_is_dropped(self):
        """A tool may report on files it pulled in (imports); the scope wins."""
        ws = _FakeWorkspace()

        async def leaky(container_name: str, command: list[str], **_: Any) -> dict[str, Any]:
            if command[0] == "ruff":
                ws.commands.append(list(command))
                leaked = "".join(
                    f"{f}:1:1: F401 m\n" for f in ("app/a.py", "./app/b.py", "/workspace/pkg/x.py")
                )
                return {"exit_code": 1, "stdout": leaked, "stderr": ""}
            return await _FakeWorkspace.execute(ws, container_name, command)

        ws.execute = leaky  # type: ignore[method-assign]
        report = await _run(ws, ["app/a.py"])
        assert _issue_files(report) == {"app/a.py"}

    @pytest.mark.asyncio
    async def test_unscopable_tool_is_skipped_not_run_project_wide(self):
        ws = _FakeWorkspace()
        report = await _run(ws, ["app/a.py"], tools=["ruff", "tsc"])
        assert not any(c[0] == "tsc" for c in ws.commands)
        assert report.tools_attempted == ["ruff"]

    @pytest.mark.asyncio
    async def test_unscoped_run_is_unchanged(self):
        """Unchanged except for py2rust #6: the source checkouts under
        ``.repos/`` are not part of the primary repo -- never fetched, and the
        workspace-wide ruff's findings on them are dropped."""
        ws = _FakeWorkspace()
        report = await _run(ws, None, tools=["ruff", "tsc"])
        find = next(c for c in ws.commands if c[0] == "find")
        assert find[1:3] == ["/workspace", "-type"]
        assert "/workspace/.repos/*" in find
        assert "/workspace/.repos/big/huge.py" not in ws.reads
        assert len(ws.reads) == len(_WORKSPACE_FILES) - 1
        assert _issue_files(report) == set(ws.files) - {"/workspace/.repos/big/huge.py"}
        assert {i.repo for i in report.issues} == {"primary"}
        assert any(c[0] == "tsc" for c in ws.commands)
        assert report.tools_attempted == ["ruff", "tsc"]

    @pytest.mark.asyncio
    async def test_scope_does_not_leak_into_the_next_run(self):
        """The tool executor is reused across requests; scope is per-run."""
        ws = _FakeWorkspace()
        analyzer = _analyzer(["ruff"])
        with patch(_ANALYZER_CLIENT, return_value=ws), patch(_TOOL_CLIENT, return_value=ws):
            await analyzer.analyze(Path("/workspace"), "ws-1", _TENANT, scope_paths=["app/a.py"])
            report = await analyzer.analyze(Path("/workspace"), "ws-1", _TENANT)
        assert _issue_files(report) == set(ws.files) - {"/workspace/.repos/big/huge.py"}


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


class TestPathsValidation:
    @pytest.mark.parametrize(
        "bad",
        ["../etc/passwd", "app/../../x", "/etc/passwd", "C:/x", "a\\b", ".", "./", "", "a\x00b"],
    )
    def test_rejected(self, bad):
        with pytest.raises(ValidationError):
            QualityAnalysisSyncRequest(container_name="ws-1", paths=[bad])

    def test_too_many_entries(self):
        with pytest.raises(ValidationError):
            QualityAnalysisSyncRequest(
                container_name="ws-1", paths=[f"f{i}.py" for i in range(MAX_SCOPE_PATHS + 1)]
            )

    def test_empty_list_rejected(self):
        with pytest.raises(ValidationError):
            QualityAnalysisSyncRequest(container_name="ws-1", paths=[])

    def test_paths_require_a_container(self):
        with pytest.raises(ValidationError):
            QualityAnalysisSyncRequest(code_content="x = 1\n", paths=["a.py"])

    def test_normalised_and_deduplicated(self):
        req = QualityAnalysisSyncRequest(
            container_name="ws-1", paths=["./app/a.py", "app//a.py", "pkg/"]
        )
        assert req.paths == ["app/a.py", "pkg"]


# ---------------------------------------------------------------------------
# Endpoint: 422s, scope echo, and the store decision
# ---------------------------------------------------------------------------


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


class TestScopedEndpoint:
    def test_traversal_is_422_over_http(self):
        analyzer = AsyncMock()
        r = _client(analyzer, FakeQualityResultStore()).post(
            "/quality/analyze-sync", json={"container_name": "ws-1", "paths": ["../secret"]}
        )
        assert r.status_code == 422, r.text
        analyzer.analyze.assert_not_called()

    def test_no_match_is_422_not_an_empty_report(self):
        # Raise the class the ENDPOINT module holds: other suites reload the
        # analyzer module, which would otherwise make this a different class.
        from app.api.endpoints import quality as quality_endpoint

        analyzer = AsyncMock()
        analyzer.analyze = AsyncMock(side_effect=quality_endpoint.ScopeMatchesNoFilesError("none"))
        store = FakeQualityResultStore()
        with (
            patch(_VALIDATE_CONTAINER, new_callable=AsyncMock, return_value=_container()),
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=None),
        ):
            r = _client(analyzer, store).post(
                "/quality/analyze-sync", json={"container_name": "ws-1", "paths": ["nope.py"]}
            )
        assert r.status_code == 422, r.text
        assert "paths match no files" in r.text
        assert store.save_calls == []

    @pytest.mark.parametrize("tools", [["radon"], ["ruff", "ty", "bandit", "radon", "ruff-format"]])
    def test_one_missing_path_is_422_naming_it_over_http(self, tools: list[str]):
        """Real analyzer: radon prints ``{}``/exit 0 for a missing file, so only a
        per-path check stops ``[real, typo]`` from returning a 200 success."""
        ws = _FakeWorkspace()
        store = FakeQualityResultStore()
        with (
            patch(_VALIDATE_CONTAINER, new_callable=AsyncMock, return_value=_container()),
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=None),
            patch(_ANALYZER_CLIENT, return_value=ws),
            patch(_TOOL_CLIENT, return_value=ws),
        ):
            r = _client(_analyzer(tools), store).post(
                "/quality/analyze-sync",
                json={"container_name": "ws-1", "paths": ["app/a.py", "app/typo.py"]},
            )
        assert r.status_code == 422, r.text
        err = r.json()["error"]
        assert err["message"] == "paths match no files: ['app/typo.py']"
        assert err["details"]["unmatched"] == ["app/typo.py"]
        assert [c[0] for c in ws.commands] == ["find", "find"]
        assert ws.reads == []
        assert store.save_calls == []

    @pytest.mark.asyncio
    async def test_scoped_run_threads_scope_and_is_not_stored(self):
        store = FakeQualityResultStore()
        full = make_report()
        await store.save(
            full, tenant_id=_TENANT, container_name="ws-1", repo_url=None, commit_sha=None
        )
        full_latest = await store.latest(_TENANT, "ws-1")

        analyzer = AsyncMock()
        analyzer.analyze = AsyncMock(return_value=make_report())
        with (
            patch(_VALIDATE_CONTAINER, new_callable=AsyncMock, return_value=_container()),
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=None),
        ):
            result = await run_quality_analysis_sync(
                request=QualityAnalysisSyncRequest(container_name="ws-1", paths=["app/a.py"]),
                analyzer=analyzer,
                tenant_id=_TENANT,
                store=store,
            )
        assert analyzer.analyze.await_args.kwargs["scope_paths"] == ["app/a.py"]
        assert result["scope"] == ["app/a.py"]
        assert result["persisted"] is False
        assert result["result_id"] is None
        assert len(store.save_calls) == 1  # only the seeded full run
        assert await store.latest(_TENANT, "ws-1") == full_latest

    @pytest.mark.asyncio
    async def test_unscoped_container_run_is_still_stored(self):
        store = FakeQualityResultStore()
        analyzer = AsyncMock()
        analyzer.analyze = AsyncMock(return_value=make_report())
        with (
            patch(_VALIDATE_CONTAINER, new_callable=AsyncMock, return_value=_container()),
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=None),
        ):
            result = await run_quality_analysis_sync(
                request=QualityAnalysisSyncRequest(container_name="ws-1"),
                analyzer=analyzer,
                tenant_id=_TENANT,
                store=store,
            )
        assert "scope_paths" not in analyzer.analyze.await_args.kwargs
        assert result["scope"] is None
        assert result["persisted"] is True
        assert len(store.save_calls) == 1

    def test_sync_request_reuses_the_shared_repo_selector(self):
        from processing.workspace_repos import RepoSelector

        assert issubclass(QualityAnalysisSyncRequest, RepoSelector)
        assert "_validate_repo" not in vars(QualityAnalysisSyncRequest)
        assert QualityAnalysisSyncRequest(container_name="w", repo="primary").repo == "primary"
        assert (
            QualityAnalysisSyncRequest(container_name="w", repo="acme__x").repo == ".repos/acme__x"
        )


# ---------------------------------------------------------------------------
# Multi-path argv: declared per tool, never inferred from the argv shape
# ---------------------------------------------------------------------------

_REPO = Path("/workspace")
_T = str(_REPO)
# The real argv each path-taking tool is built with (target = str(repo_path)).
_ARGV: dict[str, list[str]] = {
    "ruff": ["ruff", "check", _T, "--output-format=concise", "--extend-select", "I,PL"],
    "ruff-format": ["ruff", "format", "--check", "--output-format=concise", _T],
    "ty": ["ty", "check", "--output-format", "gitlab", _T],
    "bandit": ["bandit", "-r", _T, "-f", "json"],
    "radon": ["radon", "cc", _T, "-j", "-n", "C"],
    "eslint": ["eslint", _T, "-f", "json"],
    "prettier": ["prettier", "--check", _T],
    "jshint": ["jshint", _T, "--reporter=json"],
    "checkstyle": ["checkstyle", "-c", "/google_checks.xml", _T],
    "pmd": ["pmd", "check", "-d", _T, "-R", "rulesets/java/quickstart.xml", "-f", "json"],
    "cppcheck": ["cppcheck", "--enable=all", "--template='{file}:{line}:{severity}:{message}'", _T],
    "cpplint": ["cpplint", "--recursive", _T],
    "semgrep": ["semgrep", "--config=auto", "--json", _T],
}
_SCOPES = {
    2: ["app/a.py", "pkg"],
    3: ["app/a.py", "pkg", "pkg/sub/y.py"],
}


def _scoped(tool: str, scope: list[str] | None) -> list[str] | None:
    te = ToolExecution()
    te._scope_paths = scope
    return te._scope_command(tool, list(_ARGV.get(tool, [tool, _T])), _REPO)


class TestMultiPathArgv:
    def test_every_argv_tool_is_declared(self):
        assert set(_ARGV) <= set(SCOPE_MODES)
        assert {m for t, m in SCOPE_MODES.items() if t in _ARGV} <= {SEPARATE, COMMA_AFTER_FLAG}

    @pytest.mark.parametrize("n", [2, 3])
    @pytest.mark.parametrize("tool", sorted(t for t in _ARGV if SCOPE_MODES[t] == SEPARATE))
    def test_separate_tools_get_each_path_as_its_own_arg(self, tool: str, n: int):
        scope = _SCOPES[n]
        expected_paths = [f"/workspace/{p}" for p in scope]
        argv = _ARGV[tool]
        i = argv.index(_T)
        assert _scoped(tool, scope) == argv[:i] + expected_paths + argv[i + 1 :]
        assert not any("," in a for a in _scoped(tool, scope) or [] if a.startswith("/workspace"))

    @pytest.mark.parametrize("n", [2, 3])
    def test_pmd_gets_one_comma_joined_dir_value(self, n: int):
        scope = _SCOPES[n]
        out = _scoped("pmd", scope)
        assert out is not None
        assert out[out.index("-d") + 1] == ",".join(f"/workspace/{p}" for p in scope)
        assert len(out) == len(_ARGV["pmd"])

    @pytest.mark.parametrize(
        ("tool", "argv_bit"),
        [
            ("bandit", ["-r", "/workspace/app/a.py", "/workspace/pkg", "-f"]),
            ("ruff-format", ["--output-format=concise", "/workspace/app/a.py", "/workspace/pkg"]),
            ("cppcheck", ["/workspace/app/a.py", "/workspace/pkg"]),
            ("cpplint", ["--recursive", "/workspace/app/a.py", "/workspace/pkg"]),
        ],
    )
    def test_flag_before_target_is_not_a_comma_list(self, tool: str, argv_bit: list[str]):
        """The b4eb09f regression: a flag before the target is not PMD's -d."""
        out = _scoped(tool, _SCOPES[2])
        assert out is not None
        j = out.index(argv_bit[0])
        assert out[j : j + len(argv_bit)] == argv_bit

    @pytest.mark.parametrize("tool", sorted(_ARGV))
    def test_single_path_is_unchanged(self, tool: str):
        argv = _ARGV[tool]
        assert _scoped(tool, ["app/a.py"]) == [
            "/workspace/app/a.py" if a == _T else a for a in argv
        ]

    @pytest.mark.parametrize("tool", sorted(_ARGV))
    def test_unscoped_argv_is_untouched(self, tool: str):
        assert _scoped(tool, None) == _ARGV[tool]

    @pytest.mark.parametrize(
        "tool",
        sorted([t for t, m in SCOPE_MODES.items() if m == UNSCOPABLE] + ["flake8-undeclared"]),
    )
    def test_unscopable_or_undeclared_tool_is_skipped(self, tool: str):
        """Even with the target in argv: no declaration means no guessing."""
        assert _scoped(tool, _SCOPES[2]) is None


# ---------------------------------------------------------------------------
# A scoped tool that did not find its target is a no-verdict, not a pass
# ---------------------------------------------------------------------------

_MISSING_OUTPUT: dict[str, dict[str, Any]] = {
    "ruff": {
        # REAL ruff 0.17.3 concise output for a missing target (tests/fixtures/ruff).
        "exit_code": 1,
        "stdout": "/workspace/nope.py:1:1: E902 No such file or directory (os error 2)\nFound 1 error.\n",
        "stderr": "",
    },
    "bandit": {
        "exit_code": 0,
        "stdout": json.dumps(
            {
                "errors": [
                    {
                        "filename": "/workspace/nope.py",
                        "reason": "[Errno 2] No such file or directory: '/workspace/nope.py'",
                    }
                ],
                "results": [],
            }
        ),
        "stderr": "",
    },
    "ty": {
        # REAL ty 0.0.92 output for a missing target (tests/fixtures/ty).
        "exit_code": 2,
        "stdout": json.dumps(
            [
                {
                    "check_name": "io",
                    "description": "io: `/workspace/nope.py`: No such file or directory (os error 2)",
                    "severity": "major",
                    "fingerprint": "482aa4d4dbea8467",
                    "location": {
                        "path": "",
                        "positions": {
                            "begin": {"line": 1, "column": 1},
                            "end": {"line": 1, "column": 1},
                        },
                    },
                }
            ]
        ),
        "stderr": "WARN No python files found under the given path(s)\n",
    },
    "eslint": {
        "exit_code": 2,
        "stdout": "",
        "stderr": 'Oops! Something went wrong! :(\nNo files matching the pattern "/workspace/nope.js" were found.\n',
    },
    "prettier": {
        "exit_code": 2,
        "stdout": "Checking formatting...\n",
        "stderr": '[error] No files matching the pattern were found: "/workspace/nope.js".\n',
    },
    "cppcheck": {
        "exit_code": 1,
        "stdout": "cppcheck: error: could not find or open any of the paths given.\n",
        "stderr": "",
    },
    "cpplint": {
        "exit_code": 1,
        "stdout": "",
        "stderr": "Skipping input '/workspace/nope.cc': Can't open for reading\n",
    },
    "radon": {
        "exit_code": 0,
        "stdout": json.dumps(
            {"/workspace/nope.py": {"error": "[Errno 2] No such file or directory"}}
        ),
        "stderr": "",
    },
}


class _CannedWorkspace(_FakeWorkspace):
    def __init__(self, outputs: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self.outputs = outputs

    async def execute(self, container_name: str, command: list[str], **kw: Any) -> dict[str, Any]:
        if command[0] in self.outputs:
            self.commands.append(list(command))
            return self.outputs[command[0]]
        return await super().execute(container_name, command, **kw)


class TestMissingTargetIsInconclusive:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool", sorted(_MISSING_OUTPUT))
    async def test_missing_target_is_not_a_zero_issue_success(self, tool: str):
        te = ToolExecution()
        ws = _CannedWorkspace(_MISSING_OUTPUT)
        with patch(_TOOL_CLIENT, return_value=ws):
            issues, result = await te.run_tool(
                tool, [], _REPO, "python", "ws-1", _TENANT, scope_paths=["app/a.py", "nope.py"]
            )
        assert result.get("success") is False
        assert "not analysed" in (result.get("error") or "")
        assert issues == []

    @pytest.mark.asyncio
    async def test_only_tool_missing_its_target_makes_the_run_inconclusive(self):
        ws = _CannedWorkspace({"bandit": _MISSING_OUTPUT["bandit"]})
        report = await _run(ws, ["app/a.py"], tools=["bandit"])
        assert report.tools_attempted == ["bandit"]
        assert report.tools_succeeded == []
        assert report.inconclusive is True

    @pytest.mark.asyncio
    async def test_ruff_format_normal_summary_is_not_a_missing_target(self):
        """'0 files would be left unchanged' is a formatter's ordinary summary."""
        ws = _CannedWorkspace(
            {
                "ruff": {
                    "exit_code": 1,
                    "stdout": "/workspace/app/a.py:1:1: unformatted: File would be reformatted\n"
                    "1 file would be reformatted, 0 files would be left unchanged.\n",
                    "stderr": "1 file would be reformatted, 0 files would be left unchanged.\n",
                }
            }
        )
        te = ToolExecution()
        with patch(_TOOL_CLIENT, return_value=ws):
            issues, result = await te.run_tool(
                "ruff-format", [], _REPO, "python", "ws-1", _TENANT, scope_paths=["app/a.py", "pkg"]
            )
        assert result.get("success") is True
        assert [i.rule_id for i in issues] == ["unformatted"]

    @pytest.mark.asyncio
    async def test_unscoped_run_is_not_reinterpreted(self):
        te = ToolExecution()
        ws = _CannedWorkspace({"bandit": _MISSING_OUTPUT["bandit"]})
        with patch(_TOOL_CLIENT, return_value=ws):
            _, result = await te.run_tool("bandit", [], _REPO, "python", "ws-1", _TENANT)
        assert result.get("success") is True
