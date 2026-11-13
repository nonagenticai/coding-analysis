"""Unit tests for app.api.endpoints.analysis module.

The analysis.py module is a router aggregator that includes sub-routers:
- analysis_endpoints
- batch_analysis
- capabilities
- multi_language_analysis
"""

from __future__ import annotations

import pathlib
import sys
import types as _types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# sys.modules surgery — must run before importing the real endpoint module.
#
# conftest.py pre-registers stub versions of all app.api.endpoints.* and the
# app package hierarchy.  Those stubs must be evicted before we can import
# the real module.
#
# Broken import chains in analysis sub-modules:
#   analysis_endpoints.py  -> processing.analyzers.deep
#                          -> services.github_provisioning_service
#     (services/__init__.py pulls in coding_client which has broken imports)
#   batch_analysis.py      -> coding_client.utils.async_timeout (missing)
#                          -> app.core.dependencies
#                             -> processing.multi_language_factory  (BROKEN ..core)
#   capabilities.py        -> (clean)
#   multi_language_analysis.py
#                          -> processing.multi_language_factory (BROKEN ..core)
#                          -> processing.languages.base (works standalone)
#   core_analysis.py       -> processing.analyzers (BROKEN ..core.file_service)
# ---------------------------------------------------------------------------

# 1. Evict only the endpoint stubs we need to replace (NOT app.core — other tests need it).
for _key in list(sys.modules):
    if _key in ("app.api.endpoints.analysis",):
        del sys.modules[_key]

# Ensure app packages exist with real __path__ for submodule resolution.
_src_dir = pathlib.Path(__file__).resolve().parents[2] / "src"
for _pkg, _subpath in (
    ("app", "app"),
    ("app.api", "app/api"),
    ("app.api.endpoints", "app/api/endpoints"),
    ("app.core", "app/core"),
):
    if _pkg not in sys.modules:
        _m = _types.ModuleType(_pkg)
        _m.__path__ = [str(_src_dir / _subpath)]
        _m.__package__ = _pkg
        sys.modules[_pkg] = _m
    elif not getattr(sys.modules[_pkg], "__path__", None):
        sys.modules[_pkg].__path__ = [str(_src_dir / _subpath)]

# 2. Stub 'services' as a fake package so services/__init__.py never runs.
#    (A fake package with a real __path__, so submodules still import.)
if "services" not in sys.modules:
    _fake_services = _types.ModuleType("services")
    _fake_services.__path__ = [
        str(pathlib.Path(__file__).resolve().parents[2].joinpath("src", "services"))
    ]
    _fake_services.__package__ = "services"
    sys.modules["services"] = _fake_services

# Stub specific services sub-modules needed by analysis sub-routers.
_gh_stub = _types.ModuleType("services.github_provisioning_service")
_gh_stub.get_github_provisioning_service = MagicMock(return_value=MagicMock())
sys.modules.setdefault("services.github_provisioning_service", _gh_stub)

# 3. Stub processing.multi_language_factory and processing.analyzers
#    (both have broken relative imports when loaded as top-level packages).
#    Force-set (not setdefault) so we own all required attributes regardless
#    of what a previously run test file may have installed.
if "processing.multi_language_factory" not in sys.modules:
    _mlf_stub = _types.ModuleType("processing.multi_language_factory")
    _mlf_stub.MultiLanguageProcessorFactory = MagicMock
    _mlf_stub.get_processor_for_language = MagicMock(return_value=None)
    _mlf_stub.get_supported_languages = MagicMock(return_value=[])
    sys.modules["processing.multi_language_factory"] = _mlf_stub

_an_pkg = _types.ModuleType("processing.analyzers")
_an_pkg.__path__ = []
sys.modules.setdefault("processing.analyzers", _an_pkg)

_pa_stub = _types.ModuleType("processing.analyzers.python_analyzer")
_pa_stub.PythonCodeProcessor = MagicMock
sys.modules.setdefault("processing.analyzers.python_analyzer", _pa_stub)

# Stub processing.analyzers.deep (analysis_modules now imports DeepCodeAnalyzer
# from there) so the empty-path processing.analyzers stub above doesn't block it.
_deep_stub = _types.ModuleType("processing.analyzers.deep")
_deep_stub.DeepCodeAnalyzer = MagicMock
sys.modules.setdefault("processing.analyzers.deep", _deep_stub)

# 5. Stub app.core.dependencies to avoid it re-importing the broken factory.
if "app.core.dependencies" not in sys.modules:
    _dep_stub = _types.ModuleType("app.core.dependencies")
    _dep_stub.get_multi_language_factory = MagicMock(return_value=MagicMock())
    sys.modules["app.core.dependencies"] = _dep_stub

# 6. Now the real import succeeds.
from fastapi import APIRouter  # noqa: E402


class TestAnalysisRouter:
    """Tests for the analysis aggregation router."""

    def test_router_is_api_router_instance(self):
        from app.api.endpoints.analysis import router

        assert isinstance(router, APIRouter)

    def test_router_has_routes(self):
        from app.api.endpoints.analysis import router

        assert len(router.routes) > 0

    def test_router_exported_in_all(self):
        import app.api.endpoints.analysis as mod

        if hasattr(mod, "__all__"):
            assert "router" in mod.__all__
        else:
            # Module loaded as stub in full-suite — verify router exists
            assert hasattr(mod, "router")


# ---------------------------------------------------------------------------
# Tests merged from test_endpoint_analysis_coverage.py
# ---------------------------------------------------------------------------

from app.api.endpoints.analysis_modules.analysis_utils import (  # noqa: E402
    detect_language_from_code,
    detect_language_from_file_path,
    find_code_files,
    get_extension_for_language,
)
from app.api.endpoints.analysis_modules.batch_analysis import (  # noqa: E402
    aggregate_analysis_results,
    process_files_concurrent,
)


class TestAnalysisRouterCoverage:
    """Verify the top-level analysis router includes sub-routers."""

    def test_capabilities_sub_router_has_capabilities_route(self):
        from app.api.endpoints.analysis_modules.capabilities import router

        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("capabilities" in p for p in paths)

    def test_batch_sub_router_has_batch_route(self):
        from app.api.endpoints.analysis_modules.batch_analysis import router

        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("batch" in p for p in paths)

    def test_capabilities_sub_router_has_status_route(self):
        from app.api.endpoints.analysis_modules.capabilities import router

        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert any("status" in p for p in paths)


class TestCapabilitiesEndpoint:
    """Test the GET /capabilities endpoint handler directly."""

    @pytest.mark.asyncio
    async def test_returns_capabilities_response(self):
        from app.api.endpoints.analysis_modules.capabilities import (
            get_analysis_capabilities,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "test-user"
        mock_user.tenant_id = "test-tenant"

        result = await get_analysis_capabilities(user=mock_user)
        assert result.message == "Analysis capabilities retrieved"
        assert result.capabilities is not None

    @pytest.mark.asyncio
    async def test_capabilities_has_expected_languages(self):
        from app.api.endpoints.analysis_modules.capabilities import (
            get_analysis_capabilities,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "test-user"
        result = await get_analysis_capabilities(user=mock_user)
        langs = result.capabilities.supported_languages
        assert "python" in langs
        assert "javascript" in langs
        # py2rust #13: no /analyze runner has Rust-specific logic, so it is not
        # claimed (it used to be, with go/cpp/csharp/c).
        assert "rust" not in langs

    @pytest.mark.asyncio
    async def test_capabilities_has_expected_features(self):
        from app.api.endpoints.analysis_modules.capabilities import (
            get_analysis_capabilities,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "test-user"
        result = await get_analysis_capabilities(user=mock_user)
        features = result.capabilities.features
        assert features["syntax_analysis"] is True
        assert features["security_scanning"] is True

    @pytest.mark.asyncio
    async def test_capabilities_has_expected_types(self):
        from app.api.endpoints.analysis_modules.capabilities import (
            get_analysis_capabilities,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "test-user"
        result = await get_analysis_capabilities(user=mock_user)
        types = [str(t) for t in result.capabilities.supported_types]
        assert "basic" in types
        assert "security" in types
        assert "deep" in types


class TestDetectLanguageFromCode:
    @pytest.mark.parametrize(
        ("code", "expected_lang"),
        [
            ("def foo():\n    pass", "python"),
            ("import os\nprint(os.getcwd())", "python"),
            ("class Foo:\n    pass", "python"),
            ("function greet() { console.log('hi'); }", "javascript"),
            ("const x = 42; console.log(x);", "javascript"),
            ("interface Foo { bar: string; }", "typescript"),
            ("let x: number = 5;", "typescript"),
            ("System.out.println(42);", "java"),
            ('#include <stdio.h>\nint main() { printf("hi"); }', "c"),
            ("#include <iostream>\nstd::cout << 1;", "cpp"),
            ('package main\nfunc main() { fmt.Println("hi") }', "go"),
            ('fn main() { println!("hello"); }', "rust"),
        ],
    )
    def test_detects_language(self, code: str, expected_lang: str):
        assert detect_language_from_code(code) == expected_lang

    def test_returns_none_for_unknown(self):
        assert detect_language_from_code("just some text no keywords") is None


class TestDetectLanguageFromFilePath:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("main.py", "python"),
            ("index.js", "javascript"),
            ("app.ts", "typescript"),
            ("Main.java", "java"),
            ("lib.rs", "rust"),
            ("main.go", "go"),
            ("file.cpp", "cpp"),
            ("file.c", "c"),
            ("file.cs", "csharp"),
            ("script.sh", "shell"),
            ("unknown.xyz", None),
        ],
    )
    def test_detects_from_extension(self, path: str, expected: str | None):
        assert detect_language_from_file_path(path) == expected


class TestGetExtensionForLanguage:
    @pytest.mark.parametrize(
        ("language", "expected"),
        [
            ("python", ".py"),
            ("javascript", ".js"),
            ("typescript", ".ts"),
            ("java", ".java"),
            ("go", ".go"),
            ("rust", ".rs"),
            ("unknown_lang", ".txt"),
        ],
    )
    def test_extension_mapping(self, language: str, expected: str):
        assert get_extension_for_language(language) == expected


class TestFindCodeFilesCoverage:
    """Tests use a 'project/' subdirectory inside tmp_path to avoid
    find_code_files skipping paths containing 'tmp'.
    """

    @pytest.fixture
    def project_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "project"
        d.mkdir()
        return d

    def test_finds_python_files(self, project_dir: Path):
        (project_dir / "main.py").write_text("x = 1\n")
        (project_dir / "lib.py").write_text("y = 2\n")
        result = find_code_files(project_dir)
        assert len(result) == 2
        names = {f.name for f in result}
        assert "main.py" in names
        assert "lib.py" in names

    def test_skips_gitdir(self, project_dir: Path):
        git_dir = project_dir / ".git"
        git_dir.mkdir()
        (git_dir / "config.py").write_text("x = 1\n")
        (project_dir / "app.py").write_text("y = 2\n")
        result = find_code_files(project_dir)
        assert len(result) == 1
        assert result[0].name == "app.py"

    def test_skips_node_modules(self, project_dir: Path):
        nm = project_dir / "node_modules"
        nm.mkdir()
        (nm / "dep.js").write_text("module.exports = {};")
        (project_dir / "index.js").write_text("console.log('hi');")
        result = find_code_files(project_dir)
        assert len(result) == 1

    def test_handles_multiple_extensions(self, project_dir: Path):
        (project_dir / "app.py").write_text("# python\n")
        (project_dir / "index.js").write_text("// js\n")
        (project_dir / "main.go").write_text("package main\n")
        (project_dir / "readme.txt").write_text("not code\n")
        result = find_code_files(project_dir)
        names = {f.name for f in result}
        assert "app.py" in names
        assert "index.js" in names
        assert "main.go" in names
        assert "readme.txt" not in names

    def test_returns_sorted_results(self, project_dir: Path):
        (project_dir / "z.py").write_text("z = 1\n")
        (project_dir / "a.py").write_text("a = 1\n")
        result = find_code_files(project_dir)
        assert result[0].name == "a.py"
        assert result[1].name == "z.py"

    def test_empty_dir_returns_empty(self, project_dir: Path):
        result = find_code_files(project_dir)
        assert result == []


class TestAggregateAnalysisResults:
    def test_empty_results_returns_empty_dict(self):
        assert aggregate_analysis_results([]) == {}

    def test_all_failed_returns_empty_dict(self):
        results = [{"success": False, "error": "fail"}]
        assert aggregate_analysis_results(results) == {}

    def test_aggregates_successful_results(self):
        results = [
            {
                "success": True,
                "entities_count": 10,
                "relationships_count": 5,
                "lines_of_code": 100,
                "complexity_score": 3.0,
                "execution_time": 1.0,
                "language": "python",
                "parser_used": "ast",
            },
            {
                "success": True,
                "entities_count": 20,
                "relationships_count": 10,
                "lines_of_code": 200,
                "complexity_score": 5.0,
                "execution_time": 2.0,
                "language": "python",
                "parser_used": "ast",
            },
        ]
        agg = aggregate_analysis_results(results)
        assert agg["summary"]["total_entities"] == 30
        assert agg["summary"]["total_relationships"] == 15
        assert agg["summary"]["total_lines_of_code"] == 300
        assert agg["summary"]["files_processed"] == 2
        assert agg["summary"]["average_complexity"] == 4.0

    def test_language_distribution(self):
        results = [
            {
                "success": True,
                "entities_count": 5,
                "relationships_count": 2,
                "lines_of_code": 50,
                "complexity_score": 2.0,
                "execution_time": 0.5,
                "language": "python",
                "parser_used": "ast",
            },
            {
                "success": True,
                "entities_count": 3,
                "relationships_count": 1,
                "lines_of_code": 30,
                "complexity_score": 1.0,
                "execution_time": 0.3,
                "language": "javascript",
                "parser_used": "tree-sitter",
            },
        ]
        agg = aggregate_analysis_results(results)
        assert "python" in agg["language_distribution"]
        assert "javascript" in agg["language_distribution"]
        assert agg["language_distribution"]["python"]["files"] == 1


class TestProcessFilesConcurrent:
    @pytest.mark.asyncio
    async def test_handles_missing_file_gracefully(self):
        import asyncio

        tasks = [
            {"index": 0, "path": "/nonexistent/file.py", "timeout": 1},
        ]
        semaphore = asyncio.Semaphore(5)
        results = await process_files_concurrent(tasks, semaphore)

        assert len(results) == 1
        assert results[0]["success"] is False
        assert results[0]["index"] == 0
        assert "not found" in results[0].get("error", "").lower() or results[0]["success"] is False


# ---------------------------------------------------------------------------
# Capabilities endpoint — error handler coverage (line 201-202)
# ---------------------------------------------------------------------------


class TestCapabilitiesErrorHandler:
    """Tests for error handling in get_analysis_capabilities."""

    @pytest.mark.asyncio
    async def test_runtime_error_raises_500(self):
        from unittest.mock import patch

        from app.api.endpoints.analysis_modules.capabilities import (
            get_analysis_capabilities,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "test-user"

        with (
            patch(
                "app.api.endpoints.analysis_modules.capabilities.AnalysisCapabilities",
                side_effect=RuntimeError("broken factory"),
            ),
            pytest.raises(Exception) as exc_info,
        ):
            await get_analysis_capabilities(user=mock_user)
        assert exc_info.value.status_code == 500
        assert "Failed to get capabilities" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_import_error_raises_500(self):
        from unittest.mock import patch

        from app.api.endpoints.analysis_modules.capabilities import (
            get_analysis_capabilities,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "test-user"

        with (
            patch(
                "app.api.endpoints.analysis_modules.capabilities.AnalysisCapabilities",
                side_effect=ImportError("missing module"),
            ),
            pytest.raises(Exception) as exc_info,
        ):
            await get_analysis_capabilities(user=mock_user)
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_value_error_raises_500(self):
        from unittest.mock import patch

        from app.api.endpoints.analysis_modules.capabilities import (
            get_analysis_capabilities,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "test-user"

        with (
            patch(
                "app.api.endpoints.analysis_modules.capabilities.AnalysisCapabilities",
                side_effect=ValueError("invalid config"),
            ),
            pytest.raises(Exception) as exc_info,
        ):
            await get_analysis_capabilities(user=mock_user)
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# Analysis Status endpoint (lines 403-427)
# ---------------------------------------------------------------------------


class TestAnalysisStatusEndpoint:
    """Tests for get_analysis_status endpoint in capabilities.py."""

    def _stub_analysis_status_module(self, mock_service):
        """Ensure services.analysis_status is stubbed with the given mock_service."""
        import sys
        import types as _t

        if "services.analysis_status" not in sys.modules:
            stub = _t.ModuleType("services.analysis_status")
            sys.modules["services.analysis_status"] = stub
        sys.modules["services.analysis_status"].get_analysis_status_service = MagicMock(
            return_value=mock_service
        )

    @pytest.mark.asyncio
    async def test_status_found_returns_response(self):
        from app.api.endpoints.analysis_modules.capabilities import (
            get_analysis_status,
        )
        from app.api.models.analysis_api_models import AnalysisStatusInfo

        mock_user = MagicMock()
        mock_user.preferred_username = "test-user"

        mock_status_info = AnalysisStatusInfo(
            analysis_id="test-id-123",
            status="running",
            progress=50.0,
        )

        mock_service = MagicMock()
        mock_service.get_status = AsyncMock(return_value=mock_status_info)

        self._stub_analysis_status_module(mock_service)

        result = await get_analysis_status(analysis_id="test-id-123", user=mock_user)

        assert result.message == "Status for analysis 'test-id-123'"
        assert result.request_id == "test-id-123"
        assert result.analysis_status.progress == 50.0

    @pytest.mark.asyncio
    async def test_status_not_found_raises_404(self):
        from fastapi import HTTPException

        from app.api.endpoints.analysis_modules.capabilities import (
            get_analysis_status,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "test-user"

        mock_service = MagicMock()
        mock_service.get_status = AsyncMock(return_value=None)

        self._stub_analysis_status_module(mock_service)

        with pytest.raises(HTTPException) as exc_info:
            await get_analysis_status(analysis_id="missing-id", user=mock_user)
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_status_exception_raises_404(self):
        """Any non-HTTP exception in get_analysis_status results in 404."""
        from fastapi import HTTPException

        from app.api.endpoints.analysis_modules.capabilities import (
            get_analysis_status,
        )

        mock_user = MagicMock()
        mock_user.preferred_username = "test-user"

        mock_service = MagicMock()
        mock_service.get_status = AsyncMock(side_effect=RuntimeError("redis down"))

        self._stub_analysis_status_module(mock_service)

        with pytest.raises(HTTPException) as exc_info:
            await get_analysis_status(analysis_id="fail-id", user=mock_user)
        assert exc_info.value.status_code == 404
