"""Unit tests for app.api.endpoints.complexity module."""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# sys.modules surgery — must run before importing the real endpoint module.
#
# conftest.py pre-registers stub versions of all app.api.endpoints.* and the
# app package hierarchy so that `import main` works without the full tree.
# Those stubs must be removed and replaced before we can import the real module.
#
# Broken import chain we must break:
#   complexity.py
#     -> app.core.dependencies          -> processing.multi_language_factory
#        -> from ..core.file_service    (BROKEN: relative import beyond top-level)
#     -> processing.code_document_processor
#        -> from ..core.file_service    (BROKEN)
# ---------------------------------------------------------------------------

# 1. Remove conftest stubs for the modules we need to load for real.
for _key in list(sys.modules):
    if _key in (
        "app",
        "app.api",
        "app.api.endpoints",
        "app.api.endpoints.complexity",
        "app.core",
        "app.core.dependencies",
        "app.core.file_service",
        "app.core.security_config",
    ):
        del sys.modules[_key]

# 2. Stub the specific broken leaf modules before they are imported.
_mlf_stub = _types.ModuleType("processing.multi_language_factory")
_mlf_stub.MultiLanguageProcessorFactory = MagicMock
sys.modules.setdefault("processing.multi_language_factory", _mlf_stub)

_cdp_stub = _types.ModuleType("processing.code_document_processor")
_cdp_stub.CodeDocumentProcessor = MagicMock
sys.modules.setdefault("processing.code_document_processor", _cdp_stub)

# Stub app.core.dependencies so it won't re-import the broken factory.
if "app.core.dependencies" not in sys.modules:
    _dep_stub = _types.ModuleType("app.core.dependencies")
    _dep_stub.get_multi_language_factory = MagicMock(return_value=MagicMock())
    sys.modules["app.core.dependencies"] = _dep_stub

# 3. Now import the real module — the real 'app' package loads from src/ on sys.path.
from app.api.endpoints.complexity import (  # noqa: E402
    ComplexityAnalysisRequest,
    ComplexityCapabilitiesResponse,
    ComplexityMetricsResponse,
    FileComplexityResponse,
    _find_code_files,
    _initialize_batch_summary,
    _update_quality_distribution,
    _validate_file_path,
)

# ---------------------------------------------------------------------------
# _find_code_files
# ---------------------------------------------------------------------------


class TestFindCodeFiles:
    """Tests for _find_code_files helper."""

    def test_finds_py_files(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("print('hi')")
        (tmp_path / "utils.py").write_text("x = 1")
        result = _find_code_files(tmp_path)
        assert len(result) == 2
        names = {p.name for p in result}
        assert names == {"main.py", "utils.py"}

    def test_finds_multiple_extensions(self, tmp_path: Path):
        (tmp_path / "app.py").write_text("")
        (tmp_path / "index.js").write_text("")
        (tmp_path / "main.go").write_text("")
        (tmp_path / "lib.rs").write_text("")
        (tmp_path / "App.java").write_text("")
        result = _find_code_files(tmp_path)
        assert len(result) == 5

    def test_ignores_non_code_files(self, tmp_path: Path):
        (tmp_path / "readme.md").write_text("")
        (tmp_path / "data.json").write_text("")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        result = _find_code_files(tmp_path)
        assert len(result) == 0

    def test_ignores_hidden_directories(self, tmp_path: Path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("")
        (tmp_path / "visible.py").write_text("")
        result = _find_code_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "visible.py"

    def test_recurses_into_subdirectories(self, tmp_path: Path):
        sub = tmp_path / "sub" / "deep"
        sub.mkdir(parents=True)
        (sub / "deep.py").write_text("")
        (tmp_path / "top.py").write_text("")
        result = _find_code_files(tmp_path)
        assert len(result) == 2

    def test_returns_sorted(self, tmp_path: Path):
        (tmp_path / "z.py").write_text("")
        (tmp_path / "a.py").write_text("")
        result = _find_code_files(tmp_path)
        assert result[0].name == "a.py"
        assert result[1].name == "z.py"

    def test_empty_directory(self, tmp_path: Path):
        result = _find_code_files(tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# _validate_file_path
# ---------------------------------------------------------------------------


class TestValidateFilePath:
    """Tests for _validate_file_path helper."""

    def test_valid_path(self, tmp_path: Path):
        f = tmp_path / "test.py"
        f.write_text("x = 1")
        result = _validate_file_path(str(f))
        assert result == f

    def test_path_too_long(self):
        long_path = "/" + "a" * 300
        with pytest.raises(HTTPException) as exc_info:
            _validate_file_path(long_path)
        assert exc_info.value.status_code == 400

    def test_nonexistent_file(self, tmp_path: Path):
        missing = tmp_path / "no_such_file.py"
        with pytest.raises(HTTPException) as exc_info:
            _validate_file_path(str(missing))
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# _initialize_batch_summary
# ---------------------------------------------------------------------------


class TestInitializeBatchSummary:
    """Tests for _initialize_batch_summary helper."""

    def test_structure(self):
        summary = _initialize_batch_summary(10)
        assert summary["total_files"] == 10
        assert summary["analyzed_files"] == 0
        assert summary["failed_files"] == 0
        assert summary["below_threshold"] == 0
        assert summary["average_quality"] == 0.0
        assert summary["total_loc"] == 0
        assert summary["total_entities"] == 0
        assert summary["total_quality"] == 0.0
        assert isinstance(summary["languages"], set)
        assert len(summary["languages"]) == 0
        assert isinstance(summary["all_hotspots"], list)
        assert len(summary["all_hotspots"]) == 0

    def test_quality_distribution_keys(self):
        summary = _initialize_batch_summary(5)
        dist = summary["quality_distribution"]
        assert dist == {"excellent": 0, "good": 0, "fair": 0, "poor": 0}

    def test_different_file_counts(self):
        for count in (0, 1, 100):
            summary = _initialize_batch_summary(count)
            assert summary["total_files"] == count


# ---------------------------------------------------------------------------
# _update_quality_distribution
# ---------------------------------------------------------------------------


class TestUpdateQualityDistribution:
    """Tests for _update_quality_distribution helper."""

    def test_excellent_score(self):
        dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        _update_quality_distribution(dist, 95.0)
        assert dist["excellent"] == 1

    def test_boundary_90_is_excellent(self):
        dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        _update_quality_distribution(dist, 90.0)
        assert dist["excellent"] == 1

    def test_good_score(self):
        dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        _update_quality_distribution(dist, 85.0)
        assert dist["good"] == 1

    def test_boundary_70_is_good(self):
        dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        _update_quality_distribution(dist, 70.0)
        assert dist["good"] == 1

    def test_fair_score(self):
        dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        _update_quality_distribution(dist, 60.0)
        assert dist["fair"] == 1

    def test_boundary_50_is_fair(self):
        dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        _update_quality_distribution(dist, 50.0)
        assert dist["fair"] == 1

    def test_poor_score(self):
        dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        _update_quality_distribution(dist, 30.0)
        assert dist["poor"] == 1

    def test_boundary_49_is_poor(self):
        dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        _update_quality_distribution(dist, 49.9)
        assert dist["poor"] == 1

    def test_zero_is_poor(self):
        dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        _update_quality_distribution(dist, 0.0)
        assert dist["poor"] == 1

    def test_accumulates(self):
        dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        _update_quality_distribution(dist, 95.0)
        _update_quality_distribution(dist, 92.0)
        _update_quality_distribution(dist, 50.0)
        assert dist["excellent"] == 2
        assert dist["fair"] == 1


# ---------------------------------------------------------------------------
# GET /capabilities endpoint
# ---------------------------------------------------------------------------


class TestGetComplexityCapabilities:
    """Tests for get_complexity_capabilities endpoint."""

    @patch("app.api.endpoints.complexity.get_complexity_service")
    async def test_capabilities_returns_required_fields(self, mock_get_svc):
        mock_svc = MagicMock()
        mock_svc.get_supported_languages.return_value = ["python", "java", "go"]
        mock_get_svc.return_value = mock_svc

        from app.api.endpoints.complexity import get_complexity_capabilities

        result = await get_complexity_capabilities(tenant_id="test-tenant")
        assert isinstance(result, ComplexityCapabilitiesResponse)
        assert "python" in result.supported_languages
        assert len(result.complexity_levels) > 0
        assert "cyclomatic_complexity" in result.available_metrics
        assert "maintainability_index" in result.available_metrics
        assert isinstance(result.language_specific_features, dict)
        assert "java" in result.language_specific_features

    @patch("app.api.endpoints.complexity.get_complexity_service")
    async def test_capabilities_includes_all_metric_names(self, mock_get_svc):
        mock_svc = MagicMock()
        mock_svc.get_supported_languages.return_value = ["python"]
        mock_get_svc.return_value = mock_svc

        from app.api.endpoints.complexity import get_complexity_capabilities

        result = await get_complexity_capabilities(tenant_id="test-tenant")
        expected_metrics = {
            "cyclomatic_complexity",
            "cognitive_complexity",
            "lines_of_code",
            "nesting_depth",
            "parameter_count",
            "dependency_count",
            "halstead_difficulty",
            "maintainability_index",
            "coupling_factor",
        }
        assert set(result.available_metrics) == expected_metrics


# ---------------------------------------------------------------------------
# Pydantic model validation
# ---------------------------------------------------------------------------


class TestPydanticModels:
    """Basic model validation tests."""

    def test_complexity_analysis_request_required_fields(self):
        req = ComplexityAnalysisRequest(container_name="test-container")
        assert req.container_name == "test-container"
        assert req.file_paths is None
        assert req.language is None
        assert req.include_suggestions is True

    def test_complexity_metrics_response_creation(self):
        resp = ComplexityMetricsResponse(
            cyclomatic_complexity=5.0,
            cognitive_complexity=3.0,
            lines_of_code=20,
            nesting_depth=2,
            parameter_count=3,
            dependency_count=1,
            halstead_difficulty=2.5,
            maintainability_index=80.0,
            coupling_factor=0.3,
            complexity_level="moderate",
            risk_factors=["nesting"],
            suggestions=["refactor"],
            language="python",
            entity_type="function",
            entity_name="do_work",
        )
        assert resp.cyclomatic_complexity == 5.0
        assert resp.language == "python"

    def test_file_complexity_response_creation(self):
        resp = FileComplexityResponse(
            file_path="/app/test.py",
            language="python",
            total_loc=100,
            average_complexity=5.0,
            max_complexity=12.0,
            total_entities=3,
            complex_entities=1,
            entity_metrics=[],
            hotspots=["function main"],
            refactoring_suggestions=["split file"],
            quality_score=75.0,
        )
        assert resp.file_path == "/app/test.py"
        assert resp.quality_score == 75.0


# ---------------------------------------------------------------------------
# Tests merged from test_complexity_endpoints_coverage.py
# ---------------------------------------------------------------------------

_RESOLVE_FILES = "app.api.endpoints.complexity._resolve_container_files"
_GET_FACTORY = "app.api.endpoints.complexity.get_multi_language_factory"
_GET_COMPLEXITY_SVC = "app.api.endpoints.complexity.get_complexity_service"
_GET_PROC = "app.api.endpoints.complexity._get_processor_for_analysis"
_PARSE_CONTENT = "app.api.endpoints.complexity._parse_file_content"
_GET_CM = "app.api.endpoints.complexity.get_container_manager"
_DOCKER = "app.api.endpoints.complexity.DockerService"
_TENANT = "0195f48b-d000-7000-8000-00000000000a"


def _fake_docker(files: dict[str, str], find_list: list[str] | None = None):
    """Mock DockerService: `cat` returns content for known files; `find` lists find_list."""

    async def _exec(container_name, command, timeout=30, tenant_id=None):
        if command and command[0] == "find":
            return {
                "exit_code": 0,
                "stdout": "\n".join(find_list if find_list is not None else list(files)),
            }
        if command and command[0] == "cat":
            path = command[1]
            if path in files:
                return {"exit_code": 0, "stdout": files[path]}
            return {"exit_code": 1, "stdout": "", "stderr": "No such file"}
        return {"exit_code": 0, "stdout": ""}

    inst = MagicMock()
    inst.exec_in_container = AsyncMock(side_effect=_exec)
    return inst


def _make_mock_entity_metric():
    m = MagicMock()
    m.cyclomatic_complexity = 5.0
    m.cognitive_complexity = 3.0
    m.lines_of_code = 20
    m.nesting_depth = 2
    m.parameter_count = 3
    m.dependency_count = 1
    m.halstead_difficulty = 2.5
    m.maintainability_index = 80.0
    m.coupling_factor = 0.3
    m.complexity_level = MagicMock()
    m.complexity_level.value = "moderate"
    m.risk_factors = ["nesting"]
    m.suggestions = ["refactor"]
    m.language = "python"
    m.entity_type = "function"
    m.entity_name = "do_work"
    return m


def _make_mock_complexity_report(quality_score=85.0, entity_count=1):
    report = MagicMock()
    report.file_path = "/app/test.py"
    report.language = "python"
    report.total_loc = 100
    report.average_complexity = 5.0
    report.max_complexity = 12.0
    report.total_entities = entity_count
    report.complex_entities = 1
    report.entity_metrics = [_make_mock_entity_metric() for _ in range(entity_count)]
    report.hotspots = ["main()"]
    report.refactoring_suggestions = ["split function"]
    report.quality_score = quality_score
    return report


class TestResolveContainerFiles:
    @pytest.mark.asyncio
    async def test_container_not_found_via_value_error(self):
        from app.api.endpoints.complexity import _resolve_container_files

        mock_mgr = MagicMock()
        mock_mgr.get_container = AsyncMock(side_effect=ValueError("not found"))
        with (
            patch(_GET_CM, return_value=mock_mgr),
            pytest.raises(HTTPException) as exc_info,
        ):
            await _resolve_container_files("missing", tenant_id=_TENANT)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_container_none_raises_404(self):
        from app.api.endpoints.complexity import _resolve_container_files

        mock_mgr = MagicMock()
        mock_mgr.get_container = AsyncMock(return_value=None)
        with (
            patch(_GET_CM, return_value=mock_mgr),
            pytest.raises(HTTPException) as exc_info,
        ):
            await _resolve_container_files("missing", tenant_id=_TENANT)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_no_readable_files_raises_404(self):
        # container resolves, but exec finds/reads nothing -> 404 (not a local-FS check)
        from app.api.endpoints.complexity import _resolve_container_files

        mock_container = MagicMock()
        mock_container.codebase_path = "/workspace"
        mock_mgr = MagicMock()
        mock_mgr.get_container = AsyncMock(return_value=mock_container)
        with (
            patch(_GET_CM, return_value=mock_mgr),
            patch(_DOCKER, return_value=_fake_docker(files={}, find_list=[])),
            pytest.raises(HTTPException) as exc_info,
        ):
            await _resolve_container_files("test", tenant_id=_TENANT)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_specific_file_paths_resolved(self):
        # main.py is readable in the container; missing.py is not -> 1 resolved
        from app.api.endpoints.complexity import _resolve_container_files

        mock_container = MagicMock()
        mock_container.codebase_path = "/workspace"
        mock_mgr = MagicMock()
        mock_mgr.get_container = AsyncMock(return_value=mock_container)
        docker = _fake_docker({"/workspace/main.py": "x=1"})
        with patch(_GET_CM, return_value=mock_mgr), patch(_DOCKER, return_value=docker):
            result = await _resolve_container_files(
                "test", ["main.py", "missing.py"], tenant_id=_TENANT
            )
        assert len(result) == 1
        assert result[0].name == "main.py"
        assert result[0].read_text() == "x=1"

    @pytest.mark.asyncio
    async def test_no_valid_files_raises_404(self):
        from app.api.endpoints.complexity import _resolve_container_files

        mock_container = MagicMock()
        mock_container.codebase_path = "/workspace"
        mock_mgr = MagicMock()
        mock_mgr.get_container = AsyncMock(return_value=mock_container)
        with (
            patch(_GET_CM, return_value=mock_mgr),
            patch(_DOCKER, return_value=_fake_docker(files={})),
            pytest.raises(HTTPException) as exc_info,
        ):
            await _resolve_container_files("test", ["nonexistent.py"], tenant_id=_TENANT)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_absolute_path_under_workspace_is_accepted(self):
        from app.api.endpoints.complexity import _resolve_container_files

        mock_container = MagicMock()
        mock_container.codebase_path = "/workspace"
        mock_mgr = MagicMock()
        mock_mgr.get_container = AsyncMock(return_value=mock_container)
        docker = _fake_docker({"/workspace/pkg/abs.py": "z=3"})
        with patch(_GET_CM, return_value=mock_mgr), patch(_DOCKER, return_value=docker):
            result = await _resolve_container_files(
                "test", ["/workspace/pkg/abs.py"], tenant_id=_TENANT
            )
        assert len(result) == 1
        assert result[0].parts[-2:] == ("pkg", "abs.py")

    @pytest.mark.parametrize(
        "bad_path", ["/abs/abs.py", "../../etc/passwd", "src/../../x.py", "/workspace/../etc/x"]
    )
    @pytest.mark.asyncio
    async def test_unconfined_paths_are_refused(self, bad_path):
        # Absolute paths outside the workspace and any `..` are 400, before any
        # exec -- the local copy is written at local_root / rel, so an unconfined
        # rel would also escape the temp dir on this pod.
        from app.api.endpoints.complexity import _resolve_container_files

        mock_container = MagicMock()
        mock_container.codebase_path = "/workspace"
        mock_mgr = MagicMock()
        mock_mgr.get_container = AsyncMock(return_value=mock_container)
        docker = _fake_docker({"/etc/passwd": "root:x:0:0"})
        with (
            patch(_GET_CM, return_value=mock_mgr),
            patch(_DOCKER, return_value=docker),
            pytest.raises(HTTPException) as exc_info,
        ):
            await _resolve_container_files("test", [bad_path], tenant_id=_TENANT)
        assert exc_info.value.status_code == 400
        docker.exec_in_container.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tenant_forwarded_to_lookup_and_every_exec(self):
        # coding-containers' lookup and exec are tenant-scoped; an untenanted
        # call is 404'd for every customer workspace (coding-analysis#91 class).
        from app.api.endpoints.complexity import _resolve_container_files

        mock_container = MagicMock()
        mock_container.codebase_path = "/workspace"
        mock_mgr = MagicMock()
        mock_mgr.get_container = AsyncMock(return_value=mock_container)
        docker = _fake_docker({"/workspace/main.py": "x=1"}, find_list=["/workspace/main.py"])
        with patch(_GET_CM, return_value=mock_mgr), patch(_DOCKER, return_value=docker):
            await _resolve_container_files("test", tenant_id=_TENANT)
            await _resolve_container_files("test", ["main.py"], tenant_id=_TENANT)
        mock_mgr.get_container.assert_awaited_with("test", _TENANT)
        assert docker.exec_in_container.await_count == 3  # find + cat, then cat
        for call in docker.exec_in_container.await_args_list:
            assert call.kwargs["tenant_id"] == _TENANT

    @pytest.mark.asyncio
    async def test_auto_discovers_code_files(self):
        # `find` lists only main.py (readme.md is not a code extension) -> 1 resolved
        from app.api.endpoints.complexity import _resolve_container_files

        mock_container = MagicMock()
        mock_container.codebase_path = "/workspace"
        mock_mgr = MagicMock()
        mock_mgr.get_container = AsyncMock(return_value=mock_container)
        docker = _fake_docker({"/workspace/main.py": "x=1"}, find_list=["/workspace/main.py"])
        with patch(_GET_CM, return_value=mock_mgr), patch(_DOCKER, return_value=docker):
            result = await _resolve_container_files("test", tenant_id=_TENANT)
        assert len(result) == 1
        assert result[0].name == "main.py"


class TestGetProcessorForAnalysis:
    @pytest.mark.asyncio
    async def test_no_processor_raises_422(self):
        from app.api.endpoints.complexity import _get_processor_for_analysis

        mock_factory = MagicMock()
        mock_factory.get_processor_for_file = AsyncMock(return_value=None)
        with pytest.raises(HTTPException) as exc_info:
            await _get_processor_for_analysis(mock_factory, Path("/test.py"))
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_with_language_hint(self):
        from app.api.endpoints.complexity import _get_processor_for_analysis

        mock_proc = MagicMock()
        mock_factory = MagicMock()
        mock_factory.get_processor = AsyncMock(return_value=mock_proc)

        result = await _get_processor_for_analysis(mock_factory, Path("/test.py"), "python")
        assert result is mock_proc
        mock_factory.get_processor.assert_awaited_once_with("python")


class TestParseFileContent:
    @pytest.mark.asyncio
    async def test_no_parse_method_raises_422(self):
        from app.api.endpoints.complexity import _parse_file_content

        processor = MagicMock(spec=[])
        with pytest.raises(HTTPException) as exc_info:
            await _parse_file_content(processor, Path("/test.py"), "x=1")
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_parse_failure_raises_422(self):
        from app.api.endpoints.complexity import _parse_file_content

        parse_result = MagicMock()
        parse_result.success = False
        parse_result.errors = ["Syntax error at line 1"]
        processor = MagicMock()
        processor.parse_file_with_metrics = AsyncMock(return_value=parse_result)

        with pytest.raises(HTTPException) as exc_info:
            await _parse_file_content(processor, Path("/test.py"), "x=")
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_parse_file_fallback(self):
        from app.api.endpoints.complexity import _parse_file_content

        parse_result = MagicMock()
        parse_result.success = True
        processor = MagicMock(spec=["parse_file"])
        processor.parse_file = AsyncMock(return_value=parse_result)

        result = await _parse_file_content(processor, Path("/test.py"), "x=1")
        assert result is parse_result


class TestAnalyzeFileComplexity:
    @pytest.mark.asyncio
    @patch(_GET_COMPLEXITY_SVC)
    @patch(_GET_FACTORY)
    @patch(_RESOLVE_FILES, new_callable=AsyncMock)
    async def test_success(self, mock_resolve, mock_factory, mock_svc):
        from app.api.endpoints.complexity import (
            ComplexityAnalysisRequest,
            analyze_file_complexity,
        )

        mock_resolve.return_value = [Path("/tmp/test.py")]

        mock_proc = MagicMock()
        parse_result = MagicMock()
        parse_result.success = True
        mock_proc.parse_file_with_metrics = AsyncMock(return_value=parse_result)
        mock_factory_inst = MagicMock()
        mock_factory_inst.get_processor_for_file = AsyncMock(return_value=mock_proc)
        mock_factory.return_value = mock_factory_inst

        report = _make_mock_complexity_report()
        mock_svc_inst = MagicMock()
        mock_svc_inst.analyze_file_complexity = MagicMock(return_value=report)
        mock_svc.return_value = mock_svc_inst

        with (
            patch("builtins.open", MagicMock()),
            patch.object(Path, "read_text", return_value="x = 1"),
        ):
            req = ComplexityAnalysisRequest(container_name="test")
            result = await analyze_file_complexity(request=req, tenant_id="test-tenant")

        assert result.result.file_path == "/app/test.py"
        assert result.result.quality_score == 85.0

    @pytest.mark.asyncio
    @patch(_RESOLVE_FILES, new_callable=AsyncMock)
    async def test_no_files_raises_404(self, mock_resolve):
        from app.api.endpoints.complexity import (
            ComplexityAnalysisRequest,
            analyze_file_complexity,
        )

        mock_resolve.return_value = []

        with pytest.raises(HTTPException) as exc_info:
            req = ComplexityAnalysisRequest(container_name="test")
            await analyze_file_complexity(request=req, tenant_id="test-tenant")
        assert exc_info.value.status_code == 404
        # The caller's tenant reaches the resolver (and from there every exec).
        assert mock_resolve.await_args.kwargs["tenant_id"] == "test-tenant"

    @pytest.mark.asyncio
    @patch(_RESOLVE_FILES, new_callable=AsyncMock)
    async def test_file_not_found_raises_404(self, mock_resolve):
        from app.api.endpoints.complexity import (
            ComplexityAnalysisRequest,
            analyze_file_complexity,
        )

        mock_resolve.side_effect = FileNotFoundError("gone")

        with pytest.raises(HTTPException) as exc_info:
            req = ComplexityAnalysisRequest(container_name="test")
            await analyze_file_complexity(request=req, tenant_id="test-tenant")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch(_RESOLVE_FILES, new_callable=AsyncMock)
    async def test_generic_error_raises_500(self, mock_resolve):
        from app.api.endpoints.complexity import (
            ComplexityAnalysisRequest,
            analyze_file_complexity,
        )

        mock_resolve.side_effect = RuntimeError("internal error")

        with pytest.raises(HTTPException) as exc_info:
            req = ComplexityAnalysisRequest(container_name="test")
            await analyze_file_complexity(request=req, tenant_id="test-tenant")
        assert exc_info.value.status_code == 500


class TestAnalyzeBatchComplexity:
    @pytest.mark.asyncio
    @patch("app.api.endpoints.complexity._process_single_file_for_batch", new_callable=AsyncMock)
    @patch("app.api.endpoints.complexity.FileService")
    @patch(_GET_COMPLEXITY_SVC)
    @patch(_GET_FACTORY)
    @patch(_RESOLVE_FILES, new_callable=AsyncMock)
    async def test_batch_success(
        self, mock_resolve, mock_factory, mock_svc, mock_file_svc, mock_process
    ):
        from app.api.endpoints.complexity import (
            BatchComplexityRequest,
            FileComplexityResponse,
            analyze_batch_complexity,
        )

        mock_resolve.return_value = [Path("/tmp/a.py"), Path("/tmp/b.py")]

        file_resp = FileComplexityResponse(
            file_path="/tmp/a.py",
            language="python",
            total_loc=50,
            average_complexity=3.0,
            max_complexity=5.0,
            total_entities=2,
            complex_entities=0,
            entity_metrics=[],
            hotspots=["func1"],
            refactoring_suggestions=[],
            quality_score=92.0,
        )
        mock_process.side_effect = [file_resp, None]

        req = BatchComplexityRequest(container_name="test")
        result = await analyze_batch_complexity(request=req, tenant_id="test-tenant")

        assert len(result.results) == 1
        assert result.summary["analyzed_files"] == 1
        assert result.summary["failed_files"] == 1
        assert result.summary["average_quality"] == 92.0
        assert mock_resolve.await_args.kwargs["tenant_id"] == "test-tenant"

    @pytest.mark.asyncio
    @patch(_RESOLVE_FILES, new_callable=AsyncMock)
    async def test_no_files_raises_404(self, mock_resolve):
        from app.api.endpoints.complexity import (
            BatchComplexityRequest,
            analyze_batch_complexity,
        )

        mock_resolve.return_value = []
        with pytest.raises(HTTPException) as exc_info:
            req = BatchComplexityRequest(container_name="test")
            await analyze_batch_complexity(request=req, tenant_id="test-tenant")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    @patch(_RESOLVE_FILES, new_callable=AsyncMock)
    async def test_generic_error_raises_500(self, mock_resolve):
        from app.api.endpoints.complexity import (
            BatchComplexityRequest,
            analyze_batch_complexity,
        )

        mock_resolve.side_effect = RuntimeError("internal")
        with pytest.raises(HTTPException) as exc_info:
            req = BatchComplexityRequest(container_name="test")
            await analyze_batch_complexity(request=req, tenant_id="test-tenant")
        assert exc_info.value.status_code == 500


class TestGetComplexityCapabilitiesError:
    @pytest.mark.asyncio
    @patch(_GET_COMPLEXITY_SVC)
    async def test_error_raises_500(self, mock_svc):
        from app.api.endpoints.complexity import get_complexity_capabilities

        mock_svc.side_effect = RuntimeError("service error")
        with pytest.raises(HTTPException) as exc_info:
            await get_complexity_capabilities(tenant_id="test-tenant")
        assert exc_info.value.status_code == 500


class TestUpdateBatchStatistics:
    def test_below_threshold(self):
        from app.api.endpoints.complexity import (
            FileComplexityResponse,
            _initialize_batch_summary,
            _update_batch_statistics,
        )

        summary = _initialize_batch_summary(5)
        resp = FileComplexityResponse(
            file_path="test.py",
            language="python",
            total_loc=100,
            average_complexity=3.0,
            max_complexity=5.0,
            total_entities=2,
            complex_entities=0,
            entity_metrics=[],
            hotspots=["fn1"],
            refactoring_suggestions=[],
            quality_score=40.0,
        )
        _update_batch_statistics(resp, 70.0, summary)
        assert summary["below_threshold"] == 1
        assert summary["analyzed_files"] == 1
        assert summary["total_loc"] == 100
        assert "python" in summary["languages"]
        assert summary["quality_distribution"]["poor"] == 1

    def test_above_threshold(self):
        from app.api.endpoints.complexity import (
            FileComplexityResponse,
            _initialize_batch_summary,
            _update_batch_statistics,
        )

        summary = _initialize_batch_summary(5)
        resp = FileComplexityResponse(
            file_path="test.py",
            language="java",
            total_loc=200,
            average_complexity=3.0,
            max_complexity=5.0,
            total_entities=5,
            complex_entities=1,
            entity_metrics=[],
            hotspots=[],
            refactoring_suggestions=[],
            quality_score=95.0,
        )
        _update_batch_statistics(resp, 70.0, summary)
        assert summary["below_threshold"] == 0
        assert summary["quality_distribution"]["excellent"] == 1

    def test_hotspots_collected(self):
        from app.api.endpoints.complexity import (
            FileComplexityResponse,
            _initialize_batch_summary,
            _update_batch_statistics,
        )

        summary = _initialize_batch_summary(5)
        fr = FileComplexityResponse(
            file_path="/tmp/test.py",
            language="python",
            total_loc=100,
            average_complexity=4.0,
            max_complexity=8.0,
            total_entities=5,
            complex_entities=1,
            entity_metrics=[],
            hotspots=["line 42: complex", "line 99: nesting"],
            refactoring_suggestions=[],
            quality_score=80.0,
        )
        _update_batch_statistics(fr, 70.0, summary)
        assert len(summary["all_hotspots"]) == 2


class TestConvertHelpers:
    def test_convert_metrics_to_response(self):
        from app.api.endpoints.complexity import _convert_metrics_to_response

        metrics = _make_mock_entity_metric()
        result = _convert_metrics_to_response(metrics, include_suggestions=True)
        assert result.cyclomatic_complexity == 5.0
        assert result.suggestions == ["refactor"]

    def test_convert_metrics_without_suggestions(self):
        from app.api.endpoints.complexity import _convert_metrics_to_response

        metrics = _make_mock_entity_metric()
        result = _convert_metrics_to_response(metrics, include_suggestions=False)
        assert result.suggestions == []

    def test_create_file_complexity_response(self):
        from app.api.endpoints.complexity import (
            _convert_metrics_to_response,
            _create_file_complexity_response,
        )

        report = _make_mock_complexity_report()
        entity_metrics = [_convert_metrics_to_response(m) for m in report.entity_metrics]
        result = _create_file_complexity_response(report, entity_metrics)
        assert result.file_path == "/app/test.py"
        assert result.quality_score == 85.0

    def test_create_file_complexity_response_no_suggestions(self):
        from app.api.endpoints.complexity import (
            _convert_metrics_to_response,
            _create_file_complexity_response,
        )

        report = _make_mock_complexity_report()
        entity_metrics = [_convert_metrics_to_response(m, False) for m in report.entity_metrics]
        result = _create_file_complexity_response(report, entity_metrics, include_suggestions=False)
        assert result.refactoring_suggestions == []


# ---------------------------------------------------------------------------
# Tests merged from test_complexity_helpers_coverage.py
# ---------------------------------------------------------------------------


class TestConvertMetricsToResponse:
    """Tests for _convert_metrics_to_response."""

    def _make_metrics(self, **overrides):
        defaults = dict(
            cyclomatic_complexity=5.0,
            cognitive_complexity=3.0,
            lines_of_code=100,
            nesting_depth=2,
            parameter_count=3,
            dependency_count=5,
            halstead_difficulty=10.0,
            maintainability_index=80.0,
            coupling_factor=0.3,
            complexity_level=MagicMock(value="moderate"),
            risk_factors=["high nesting"],
            suggestions=["reduce nesting"],
            language="python",
            entity_type="function",
            entity_name="my_func",
        )
        defaults.update(overrides)
        mock = MagicMock()
        for k, v in defaults.items():
            setattr(mock, k, v)
        return mock

    def test_basic_conversion(self):
        from app.api.endpoints.complexity import _convert_metrics_to_response

        metrics = self._make_metrics()
        result = _convert_metrics_to_response(metrics)
        assert isinstance(result, ComplexityMetricsResponse)
        assert result.cyclomatic_complexity == 5.0
        assert result.entity_name == "my_func"
        assert result.suggestions == ["reduce nesting"]

    def test_without_suggestions(self):
        from app.api.endpoints.complexity import _convert_metrics_to_response

        metrics = self._make_metrics()
        result = _convert_metrics_to_response(metrics, include_suggestions=False)
        assert result.suggestions == []

    def test_with_suggestions(self):
        from app.api.endpoints.complexity import _convert_metrics_to_response

        metrics = self._make_metrics(suggestions=["refactor", "split"])
        result = _convert_metrics_to_response(metrics, include_suggestions=True)
        assert len(result.suggestions) == 2

    def test_all_fields_mapped(self):
        from app.api.endpoints.complexity import _convert_metrics_to_response

        metrics = self._make_metrics(
            cognitive_complexity=7.0,
            halstead_difficulty=15.0,
            coupling_factor=0.5,
        )
        result = _convert_metrics_to_response(metrics)
        assert result.cognitive_complexity == 7.0
        assert result.halstead_difficulty == 15.0
        assert result.coupling_factor == 0.5


class TestCreateFileComplexityResponse:
    """Tests for _create_file_complexity_response."""

    def _make_report(self, **overrides):
        defaults = dict(
            file_path="/tmp/test.py",
            language="python",
            total_loc=200,
            average_complexity=4.5,
            max_complexity=12.0,
            total_entities=10,
            complex_entities=3,
            hotspots=["func_a line 42"],
            refactoring_suggestions=["reduce complexity of func_a"],
            quality_score=75.0,
        )
        defaults.update(overrides)
        mock = MagicMock()
        for k, v in defaults.items():
            setattr(mock, k, v)
        return mock

    def test_basic_response(self):
        from app.api.endpoints.complexity import _create_file_complexity_response

        report = self._make_report()
        result = _create_file_complexity_response(report, [])
        assert isinstance(result, FileComplexityResponse)
        assert result.file_path == "/tmp/test.py"
        assert result.quality_score == 75.0
        assert result.language == "python"

    def test_without_suggestions(self):
        from app.api.endpoints.complexity import _create_file_complexity_response

        report = self._make_report(refactoring_suggestions=["a", "b"])
        result = _create_file_complexity_response(report, [], include_suggestions=False)
        assert result.refactoring_suggestions == []

    def test_with_entity_metrics(self):
        from app.api.endpoints.complexity import _create_file_complexity_response

        report = self._make_report()
        entity = ComplexityMetricsResponse(
            cyclomatic_complexity=5.0,
            cognitive_complexity=3.0,
            lines_of_code=50,
            nesting_depth=2,
            parameter_count=2,
            dependency_count=1,
            halstead_difficulty=8.0,
            maintainability_index=85.0,
            coupling_factor=0.2,
            complexity_level="low",
            risk_factors=[],
            suggestions=[],
            language="python",
            entity_type="function",
            entity_name="helper",
        )
        result = _create_file_complexity_response(report, [entity])
        assert len(result.entity_metrics) == 1
        assert result.entity_metrics[0].entity_name == "helper"


class TestComplexityModels:
    """Tests for Pydantic model classes from helpers coverage."""

    def test_complexity_analysis_request(self):
        req = ComplexityAnalysisRequest(container_name="test")
        assert req.container_name == "test"
        assert req.include_suggestions is True
        assert req.file_paths is None
        assert req.language is None

    def test_batch_complexity_request(self):
        from app.api.endpoints.complexity import BatchComplexityRequest

        req = BatchComplexityRequest(container_name="test", quality_threshold=80.0)
        assert req.quality_threshold == 80.0
        assert req.include_file_suggestions is True

    def test_complexity_capabilities_response(self):
        resp = ComplexityCapabilitiesResponse(
            supported_languages=["python", "java"],
            complexity_levels=["low", "medium", "high"],
            available_metrics=["cyclomatic"],
            language_specific_features={"python": ["decorators"]},
        )
        assert "python" in resp.supported_languages

    def test_find_code_files_finds_java(self, tmp_path):
        (tmp_path / "App.java").write_text("class App {}")
        (tmp_path / "lib.rs").write_text("fn main() {}")
        (tmp_path / "data.txt").write_text("data")
        result = _find_code_files(tmp_path)
        exts = {f.suffix for f in result}
        assert ".java" in exts
        assert ".rs" in exts
        assert ".txt" not in exts

    def test_find_code_files_skips_hidden(self, tmp_path):
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.py").write_text("pass")
        result = _find_code_files(tmp_path)
        assert len(result) == 0
