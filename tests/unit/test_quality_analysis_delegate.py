"""Unit tests for services.quality_analysis_delegate module."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.testing_models import (
    QualityIssue,
    QualityRequest,
    QualityResult,
    QualityScanRequest,
)
from services.quality_analysis_delegate import (
    ContainerAnalysisResults,
    ContainerScanContext,
    QualityAnalysisDelegate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_quality_result(**overrides) -> QualityResult:
    """Create a QualityResult with sensible defaults, overriding any field."""
    defaults = {
        "analysis_id": "test-id",
        "success": True,
        "overall_score": 95.0,
        "total_issues": 0,
        "issues": [],
        "tools_results": {},
        "execution_time": 0.1,
        "created_at": datetime.now(UTC),
        "language": "python",
        "error": None,
    }
    defaults.update(overrides)
    return QualityResult(**defaults)


def _make_quality_request(**overrides) -> QualityRequest:
    defaults = {
        "container_name": "test-container",
        "code": "print('hello')",
        "language": "python",
        "tools": ["ruff"],
    }
    defaults.update(overrides)
    return QualityRequest(**defaults)


def _make_scan_request(**overrides) -> QualityScanRequest:
    defaults = {
        "container_name": "test-container",
        "scan_path": "/app/data",
        "recursive": True,
        "include_patterns": ["*.py"],
        "exclude_patterns": ["__pycache__"],
        "max_files": 100,
        "language": "python",
        "tools": ["ruff"],
        "parallel_analysis": False,
    }
    defaults.update(overrides)
    return QualityScanRequest(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_docker_service():
    ds = AsyncMock()
    ds.is_docker_available = MagicMock(return_value=True)
    ds.exec_in_container = AsyncMock(return_value={"exit_code": 0, "stdout": "test", "stderr": ""})
    ds.list_containers = AsyncMock(return_value=[])
    return ds


@pytest.fixture
def mock_quality_runner():
    qr = AsyncMock()
    qr.run_quality_analysis = AsyncMock(return_value=_make_quality_result())
    qr.run_single_tool = AsyncMock(
        return_value={"success": True, "issues": [], "execution_time": 0.1}
    )
    return qr


@pytest.fixture
def mock_tool_registry():
    tr = MagicMock()
    tr.list_available_tools = MagicMock(return_value=["ruff", "ty"])
    tr.get_tool_info = MagicMock(return_value={"name": "ruff", "category": "linter"})
    tr.list_supported_languages = MagicMock(return_value=["python", "javascript"])
    return tr


@pytest.fixture
def mock_cache_manager():
    cm = AsyncMock()
    cm.get = AsyncMock(return_value=None)
    cm.set = AsyncMock()
    return cm


@pytest.fixture
def delegate(mock_docker_service, mock_quality_runner, mock_tool_registry, mock_cache_manager):
    return QualityAnalysisDelegate(
        docker_service=mock_docker_service,
        quality_runner=mock_quality_runner,
        tool_registry=mock_tool_registry,
        cache_manager=mock_cache_manager,
    )


@pytest.fixture
def delegate_no_cache(mock_docker_service, mock_quality_runner, mock_tool_registry):
    return QualityAnalysisDelegate(
        docker_service=mock_docker_service,
        quality_runner=mock_quality_runner,
        tool_registry=mock_tool_registry,
        cache_manager=None,
    )


# ===========================================================================
# ContainerScanContext tests
# ===========================================================================


class TestContainerScanContext:
    def test_init_with_files(self):
        ctx = ContainerScanContext(
            files_to_scan=["/a.py", "/b.py"],
            files_by_language={"python": ["/a.py", "/b.py"]},
        )
        assert ctx.has_files is True
        assert ctx.total_files == 2
        assert ctx.language_count == 1

    def test_init_empty(self):
        ctx = ContainerScanContext(files_to_scan=[], files_by_language={})
        assert ctx.has_files is False
        assert ctx.total_files == 0
        assert ctx.language_count == 0


# ===========================================================================
# ContainerAnalysisResults tests
# ===========================================================================


class TestContainerAnalysisResults:
    def test_init(self):
        issues = [{"severity": "error", "message": "bad"}]
        results = ContainerAnalysisResults(
            all_issues=issues,
            tools_results={"ruff": {"success": True}},
            files_by_language={"python": ["/a.py"]},
        )
        assert results.total_issues == 1
        assert results.all_issues == issues


# ===========================================================================
# Cache methods
# ===========================================================================


class TestCacheMethods:
    def test_generate_cache_key_deterministic(self, delegate):
        req = _make_quality_request(code="x = 1", language="python", tools=["ruff"])
        key1 = delegate.generate_cache_key(req)
        key2 = delegate.generate_cache_key(req)
        assert key1 == key2
        assert key1.startswith("quality_")

    def test_generate_cache_key_different_for_different_input(self, delegate):
        req1 = _make_quality_request(code="x = 1", tools=["ruff"])
        req2 = _make_quality_request(code="y = 2", tools=["ruff"])
        assert delegate.generate_cache_key(req1) != delegate.generate_cache_key(req2)

    def test_generate_cache_key_sorts_tools(self, delegate):
        req1 = _make_quality_request(tools=["ty", "ruff"])
        req2 = _make_quality_request(tools=["ruff", "ty"])
        assert delegate.generate_cache_key(req1) == delegate.generate_cache_key(req2)

    async def test_get_cached_result_hit(self, delegate, mock_cache_manager):
        cached = _make_quality_result()
        mock_cache_manager.get = AsyncMock(return_value=cached.model_dump())
        result = await delegate.get_cached_result("key123")
        assert result is not None
        assert result.success is True

    async def test_get_cached_result_miss(self, delegate, mock_cache_manager):
        mock_cache_manager.get = AsyncMock(return_value=None)
        result = await delegate.get_cached_result("key123")
        assert result is None

    async def test_get_cached_result_no_cache_manager(self, delegate_no_cache):
        result = await delegate_no_cache.get_cached_result("key123")
        assert result is None

    async def test_get_cached_result_error(self, delegate, mock_cache_manager):
        mock_cache_manager.get = AsyncMock(side_effect=RuntimeError("redis down"))
        result = await delegate.get_cached_result("key123")
        assert result is None

    async def test_get_cached_result_deserialization_error(self, delegate, mock_cache_manager):
        # Return data that won't deserialize into QualityResult
        mock_cache_manager.get = AsyncMock(return_value={"bad": "data"})
        result = await delegate.get_cached_result("key123")
        assert result is None

    async def test_cache_result_stores(self, delegate, mock_cache_manager):
        result = _make_quality_result()
        await delegate.cache_result("key123", result)
        mock_cache_manager.set.assert_awaited_once()

    async def test_cache_result_no_cache_manager(self, delegate_no_cache):
        result = _make_quality_result()
        await delegate_no_cache.cache_result("key123", result)
        # Should not raise

    async def test_cache_result_error_suppressed(self, delegate, mock_cache_manager):
        mock_cache_manager.set = AsyncMock(side_effect=RuntimeError("redis down"))
        result = _make_quality_result()
        await delegate.cache_result("key123", result)
        # Should not raise


# ===========================================================================
# Tool Registry Methods
# ===========================================================================


class TestToolRegistryMethods:
    def test_get_supported_tools(self, delegate, mock_tool_registry):
        tools = delegate.get_supported_tools("python")
        mock_tool_registry.list_available_tools.assert_called_once_with("python")
        assert tools == ["ruff", "ty"]

    def test_get_tool_info(self, delegate, mock_tool_registry):
        info = delegate.get_tool_info("ruff", "python")
        mock_tool_registry.get_tool_info.assert_called_once_with("python", "ruff")
        assert info == {"name": "ruff", "category": "linter"}

    def test_get_supported_languages(self, delegate, mock_tool_registry):
        langs = delegate.get_supported_languages()
        mock_tool_registry.list_supported_languages.assert_called_once()
        assert langs == ["python", "javascript"]


# ===========================================================================
# analyze_quality
# ===========================================================================


class TestAnalyzeQuality:
    async def test_success(self, delegate, mock_quality_runner):
        req = _make_quality_request()
        result = await delegate.analyze_quality(req)
        assert result.success is True
        mock_quality_runner.run_quality_analysis.assert_awaited_once_with(req)

    async def test_failed_result_enhances_error(self, delegate, mock_quality_runner):
        failed = _make_quality_result(
            success=False,
            error="Analysis failed without specific error details",
            tools_results={
                "ruff": {
                    "success": False,
                    "error": "ruff not found",
                    "issues": [],
                    "execution_time": 0.0,
                }
            },
        )
        mock_quality_runner.run_quality_analysis = AsyncMock(return_value=failed)
        req = _make_quality_request()
        result = await delegate.analyze_quality(req)
        assert result.success is False
        assert "ruff" in result.error

    async def test_exception_returns_error_result(self, delegate, mock_quality_runner):
        mock_quality_runner.run_quality_analysis = AsyncMock(side_effect=TimeoutError("timed out"))
        req = _make_quality_request()
        result = await delegate.analyze_quality(req)
        assert result.success is False
        assert "TimeoutError" in result.error

    async def test_runtime_error_returns_error_result(self, delegate, mock_quality_runner):
        mock_quality_runner.run_quality_analysis = AsyncMock(side_effect=RuntimeError("boom"))
        req = _make_quality_request()
        result = await delegate.analyze_quality(req)
        assert result.success is False
        assert "RuntimeError" in result.error


# ===========================================================================
# _enhance_error_details
# ===========================================================================


class TestEnhanceErrorDetails:
    def test_with_specific_tool_errors(self, delegate):
        result = _make_quality_result(
            success=False,
            error="Analysis failed without specific error details",
            tools_results={
                "ruff": {
                    "success": False,
                    "error": "module not found",
                    "issues": [],
                    "execution_time": 0.5,
                },
                "ty": {
                    "success": True,
                    "error": "",
                    "issues": [{"severity": "warning"}],
                    "execution_time": 1.0,
                },
            },
        )
        delegate._enhance_error_details(result)
        assert "ruff: module not found" in result.error

    def test_tool_failed_without_error_message(self, delegate):
        result = _make_quality_result(
            success=False,
            error="Analysis failed without specific error details",
            tools_results={
                "ruff": {
                    "success": False,
                    "error": "",
                    "issues": [],
                    "execution_time": 0.5,
                },
            },
        )
        delegate._enhance_error_details(result)
        assert "failed without error details" in result.error

    def test_no_specific_errors_with_tool_diagnostics(self, delegate):
        result = _make_quality_result(
            success=False,
            error="Analysis failed without specific error details",
            tools_results={
                "ruff": {
                    "success": True,
                    "error": "",
                    "issues": [],
                    "execution_time": 0.5,
                },
            },
        )
        delegate._enhance_error_details(result)
        assert "tools may not be installed" in result.error

    def test_no_tools_results(self, delegate):
        result = _make_quality_result(
            success=False,
            error="Analysis failed without specific error details",
            tools_results={},
        )
        delegate._enhance_error_details(result)
        assert "no tool results available" in result.error

    def test_preserves_existing_specific_error(self, delegate):
        result = _make_quality_result(
            success=False,
            error="Custom error message",
            tools_results={
                "ruff": {
                    "success": False,
                    "error": "tool crashed",
                    "issues": [],
                    "execution_time": 0.1,
                },
            },
        )
        delegate._enhance_error_details(result)
        assert "Custom error message" in result.error
        assert "ruff: tool crashed" in result.error


# ===========================================================================
# run_single_tool
# ===========================================================================


class TestRunSingleTool:
    async def test_success(self, delegate, mock_quality_runner):
        result = await delegate.run_single_tool("ruff", "python", "x = 1")
        assert result["success"] is True
        mock_quality_runner.run_single_tool.assert_awaited_once_with("ruff", "python", "x = 1")

    async def test_exception(self, delegate, mock_quality_runner):
        mock_quality_runner.run_single_tool = AsyncMock(side_effect=RuntimeError("crash"))
        result = await delegate.run_single_tool("ruff", "python", "x = 1")
        assert result["success"] is False
        assert "crash" in result["error"]


# ===========================================================================
# _validate_container_access
# ===========================================================================


class TestValidateContainerAccess:
    async def test_docker_not_available(self, delegate, mock_docker_service):
        mock_docker_service.is_docker_available = MagicMock(return_value=False)
        result = await delegate._validate_container_access("container1")
        assert result is False

    async def test_exec_succeeds(self, delegate, mock_docker_service):
        mock_docker_service.exec_in_container = AsyncMock(return_value={"exit_code": 0})
        result = await delegate._validate_container_access("container1")
        assert result is True

    async def test_exec_fails_but_list_finds_container(self, delegate, mock_docker_service):
        mock_docker_service.exec_in_container = AsyncMock(side_effect=RuntimeError("exec failed"))
        container_obj = MagicMock()
        container_obj.container_name = "container1"
        mock_docker_service.list_containers = AsyncMock(return_value=[container_obj])
        result = await delegate._validate_container_access("container1")
        assert result is True

    async def test_exec_fails_and_list_empty(self, delegate, mock_docker_service):
        mock_docker_service.exec_in_container = AsyncMock(side_effect=RuntimeError("exec failed"))
        mock_docker_service.list_containers = AsyncMock(return_value=[])
        result = await delegate._validate_container_access("container1")
        assert result is False

    async def test_exec_nonzero_exit_and_list_fails(self, delegate, mock_docker_service):
        mock_docker_service.exec_in_container = AsyncMock(return_value={"exit_code": 1})
        mock_docker_service.list_containers = AsyncMock(side_effect=RuntimeError("list failed"))
        result = await delegate._validate_container_access("container1")
        assert result is False

    async def test_outer_exception(self, delegate, mock_docker_service):
        mock_docker_service.is_docker_available = MagicMock(
            side_effect=RuntimeError("docker crash")
        )
        result = await delegate._validate_container_access("container1")
        assert result is False


# ===========================================================================
# _matches_pattern
# ===========================================================================


class TestMatchesPattern:
    def test_full_path_match(self, delegate):
        assert delegate._matches_pattern("/app/data/test.py", "*.py") is True

    def test_basename_match(self, delegate):
        assert delegate._matches_pattern("/app/data/test.py", "test.py") is True

    def test_directory_part_match(self, delegate):
        assert delegate._matches_pattern("/app/__pycache__/test.pyc", "__pycache__") is True

    def test_no_match(self, delegate):
        assert delegate._matches_pattern("/app/data/test.py", "*.js") is False


# ===========================================================================
# _group_files_by_language
# ===========================================================================


class TestGroupFilesByLanguage:
    def test_python_files(self, delegate):
        files = ["/a.py", "/b.pyw", "/c.pyi"]
        result = delegate._group_files_by_language(files)
        assert "python" in result
        assert len(result["python"]) == 3

    def test_javascript_files(self, delegate):
        files = ["/a.js", "/b.jsx", "/c.mjs"]
        result = delegate._group_files_by_language(files)
        assert "javascript" in result
        assert len(result["javascript"]) == 3

    def test_typescript_files(self, delegate):
        files = ["/a.ts", "/b.tsx", "/c.d.ts"]
        result = delegate._group_files_by_language(files)
        assert "typescript" in result
        assert len(result["typescript"]) == 3

    def test_mixed_languages(self, delegate):
        files = ["/a.py", "/b.js", "/c.ts"]
        result = delegate._group_files_by_language(files)
        assert len(result) == 3

    def test_unknown_extension_skipped(self, delegate):
        files = ["/a.rb", "/b.go", "/c.py"]
        result = delegate._group_files_by_language(files)
        assert "python" in result
        assert len(result) == 1


# ===========================================================================
# _discover_container_files
# ===========================================================================


class TestDiscoverContainerFiles:
    async def test_basic_discovery(self, delegate, mock_docker_service):
        mock_docker_service.exec_in_container = AsyncMock(
            return_value={
                "exit_code": 0,
                "stdout": "/app/data/a.py\n/app/data/b.py\n",
            }
        )
        files = await delegate._discover_container_files(
            "container1", "/app/data", ["*.py"], [], True, 100
        )
        assert len(files) == 2

    async def test_exclude_patterns(self, delegate, mock_docker_service):
        mock_docker_service.exec_in_container = AsyncMock(
            return_value={
                "exit_code": 0,
                "stdout": "/app/data/a.py\n/app/__pycache__/c.pyc\n",
            }
        )
        files = await delegate._discover_container_files(
            "container1", "/app/data", [], ["__pycache__"], True, 100
        )
        assert len(files) == 1
        assert "/app/data/a.py" in files

    async def test_max_files_limit(self, delegate, mock_docker_service):
        stdout = "\n".join([f"/app/file{i}.py" for i in range(20)])
        mock_docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": stdout}
        )
        files = await delegate._discover_container_files("container1", "/app", [], [], True, 5)
        assert len(files) == 5

    async def test_nonzero_exit_returns_empty(self, delegate, mock_docker_service):
        mock_docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 1, "stderr": "not found"}
        )
        files = await delegate._discover_container_files("container1", "/app", [], [], True, 100)
        assert files == []

    async def test_exception_returns_empty(self, delegate, mock_docker_service):
        mock_docker_service.exec_in_container = AsyncMock(side_effect=RuntimeError("exec failed"))
        files = await delegate._discover_container_files("container1", "/app", [], [], True, 100)
        assert files == []

    async def test_non_recursive(self, delegate, mock_docker_service):
        mock_docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "/app/a.py\n"}
        )
        await delegate._discover_container_files("container1", "/app", [], [], False, 100)
        call_args = mock_docker_service.exec_in_container.call_args
        cmd = call_args[0][1]
        assert "-maxdepth" in cmd
        assert "1" in cmd


# ===========================================================================
# _calculate_line_offsets
# ===========================================================================


class TestCalculateLineOffsets:
    def test_basic(self, delegate):
        files = ["/a.py", "/b.py"]
        contents = {"/a.py": "line1\nline2\nline3", "/b.py": "x\ny"}
        offsets = delegate._calculate_line_offsets(files, contents)
        assert offsets["/a.py"] == 1
        # a.py has 3 lines (\n count = 2), +2 = 4, so b.py starts at line 5
        assert offsets["/b.py"] == 5

    def test_empty_content(self, delegate):
        files = ["/a.py", "/b.py"]
        contents = {"/a.py": "", "/b.py": "x"}
        offsets = delegate._calculate_line_offsets(files, contents)
        assert offsets["/a.py"] == 1
        assert offsets["/b.py"] == 3  # 0 newlines + 2 = 2, starts at 3


# ===========================================================================
# _determine_issue_file / _adjust_line_number
# ===========================================================================


class TestDetermineIssueFile:
    def test_basic(self, delegate):
        offsets = {"/a.py": 1, "/b.py": 10}
        files = ["/a.py", "/b.py"]
        issue = MagicMock()
        issue.line_number = 5
        assert delegate._determine_issue_file(issue, offsets, files) == "/a.py"

    def test_second_file(self, delegate):
        offsets = {"/a.py": 1, "/b.py": 10}
        files = ["/a.py", "/b.py"]
        issue = MagicMock()
        issue.line_number = 12
        assert delegate._determine_issue_file(issue, offsets, files) == "/b.py"

    def test_empty_files(self, delegate):
        offsets = {}
        files = []
        issue = MagicMock()
        issue.line_number = 1
        assert delegate._determine_issue_file(issue, offsets, files) == "unknown"


class TestAdjustLineNumber:
    def test_basic(self, delegate):
        offsets = {"/a.py": 1, "/b.py": 10}
        issue = MagicMock()
        issue.line_number = 12
        result = delegate._adjust_line_number(issue, offsets, "/b.py")
        assert result == 2  # 12 - 10

    def test_none_line(self, delegate):
        offsets = {"/a.py": 1}
        issue = MagicMock()
        issue.line_number = None
        result = delegate._adjust_line_number(issue, offsets, "/a.py")
        assert result is None

    def test_non_int_line(self, delegate):
        offsets = {"/a.py": 1}
        issue = MagicMock()
        issue.line_number = "not a number"
        result = delegate._adjust_line_number(issue, offsets, "/a.py")
        assert result is None

    def test_min_1(self, delegate):
        offsets = {"/a.py": 10}
        issue = MagicMock()
        issue.line_number = 5  # 5 - 10 = -5, should clamp to 1
        result = delegate._adjust_line_number(issue, offsets, "/a.py")
        assert result == 1


# ===========================================================================
# _determine_tool_issue_file / _adjust_tool_issue_line
# ===========================================================================


class TestDetermineToolIssueFile:
    def test_basic(self, delegate):
        offsets = {"/a.py": 1, "/b.py": 10}
        files = ["/a.py", "/b.py"]
        issue = {"line": 12}
        assert delegate._determine_tool_issue_file(issue, offsets, files) == "/b.py"

    def test_empty_files(self, delegate):
        assert delegate._determine_tool_issue_file({"line": 1}, {}, []) == "unknown"


class TestAdjustToolIssueLine:
    def test_basic(self, delegate):
        offsets = {"/b.py": 10}
        issue = {"line": 15}
        result = delegate._adjust_tool_issue_line(issue, offsets, "/b.py")
        assert result == 5

    def test_none_line(self, delegate):
        result = delegate._adjust_tool_issue_line({"line": None}, {"/a.py": 1}, "/a.py")
        assert result is None

    def test_non_int_line(self, delegate):
        result = delegate._adjust_tool_issue_line({"line": "x"}, {"/a.py": 1}, "/a.py")
        assert result is None

    def test_clamp_to_1(self, delegate):
        result = delegate._adjust_tool_issue_line({"line": 2}, {"/a.py": 10}, "/a.py")
        assert result == 1


# ===========================================================================
# _merge_tools_results / _merge_language_tools_results
# ===========================================================================


class TestMergeToolsResults:
    def test_merge_new_tool(self, delegate):
        target = {}
        source = {"ruff": {"success": True, "issues": [{"msg": "a"}], "execution_time": 1.0}}
        delegate._merge_tools_results(target, source)
        assert "ruff" in target
        assert len(target["ruff"]["issues"]) == 1

    def test_merge_existing_tool(self, delegate):
        target = {"ruff": {"success": True, "issues": [{"msg": "a"}], "execution_time": 1.0}}
        source = {"ruff": {"success": True, "issues": [{"msg": "b"}], "execution_time": 2.0}}
        delegate._merge_tools_results(target, source)
        assert len(target["ruff"]["issues"]) == 2
        assert target["ruff"]["execution_time"] == 3.0

    def test_merge_sets_success_false_on_failure(self, delegate):
        target = {"ruff": {"success": True, "issues": [], "execution_time": 1.0}}
        source = {"ruff": {"success": False, "issues": [], "execution_time": 0.5}}
        delegate._merge_tools_results(target, source)
        assert target["ruff"]["success"] is False


class TestMergeLanguageToolsResults:
    def test_merge(self, delegate):
        target = {}
        source = {"ty": {"success": True, "issues": [{"msg": "x"}], "execution_time": 1.0}}
        delegate._merge_language_tools_results(target, source)
        assert "ty" in target
        assert len(target["ty"]["issues"]) == 1


# ===========================================================================
# _calculate_container_quality_score
# ===========================================================================


class TestCalculateContainerQualityScore:
    def test_no_issues_no_tools(self, delegate):
        assert delegate._calculate_container_quality_score([], {}) == 90.0

    def test_critical_deduction(self, delegate):
        issues = [{"severity": "critical"}, {"severity": "critical"}]
        tools = {"ruff": {"success": True}}
        score = delegate._calculate_container_quality_score(issues, tools)
        assert score == 80.0  # 100 - 2*10 = 80, tool rate 1/1

    def test_mixed_severity(self, delegate):
        issues = [
            {"severity": "critical"},
            {"severity": "error"},
            {"severity": "warning"},
            {"severity": "info"},
        ]
        tools = {"ruff": {"success": True}}
        # 100 - 10 - 5 - 2 - 0.5 = 82.5
        score = delegate._calculate_container_quality_score(issues, tools)
        assert score == 82.5

    def test_tool_success_rate_scaling(self, delegate):
        issues = []
        tools = {"ruff": {"success": True}, "ty": {"success": False}}
        # 100 * 0.5 = 50
        score = delegate._calculate_container_quality_score(issues, tools)
        assert score == 50.0

    def test_score_clamped_to_zero(self, delegate):
        issues = [{"severity": "critical"}] * 15  # 15 * 10 = 150 deduction
        tools = {"ruff": {"success": True}}
        score = delegate._calculate_container_quality_score(issues, tools)
        assert score == 0.0


# ===========================================================================
# _convert_to_quality_issues
# ===========================================================================


class TestConvertToQualityIssues:
    def test_basic_conversion(self, delegate):
        issues = [
            {
                "tool": "ruff",
                "severity": "error",
                "message": "bad code",
                "file_path": "/a.py",
                "line_number": 10,
                "column_number": 5,
                "rule_id": "E001",
                "suggested_fix": "fix it",
                "fix_confidence": 0.9,
            }
        ]
        result = delegate._convert_to_quality_issues(issues)
        assert len(result) == 1
        assert isinstance(result[0], QualityIssue)
        assert result[0].tool == "ruff"

    def test_defaults_for_missing_fields(self, delegate):
        issues = [{}]
        result = delegate._convert_to_quality_issues(issues)
        assert len(result) == 1
        assert result[0].tool == "unknown"
        assert result[0].severity == "info"

    def test_invalid_issue_skipped(self, delegate):
        issues = [
            {"tool": "ok", "severity": "info", "message": "m", "file_path": "f"},
            {
                "tool": "bad",
                "severity": "info",
                "message": "m",
                "file_path": "f",
                "fix_confidence": 999.0,
            },
        ]
        result = delegate._convert_to_quality_issues(issues)
        assert len(result) == 1


# ===========================================================================
# _create_empty_quality_result
# ===========================================================================


class TestCreateEmptyQualityResult:
    def test_returns_success(self, delegate):
        import time

        result = delegate._create_empty_quality_result("id1", time.time())
        assert result.success is True
        assert result.overall_score == 90.0
        assert result.total_issues == 0
        assert result.language == "unknown"


# ===========================================================================
# _detect_container_language
# ===========================================================================


class TestDetectContainerLanguage:
    def test_python_dominant(self, delegate):
        files = ["/a.py", "/b.py", "/c.js"]
        assert delegate._detect_container_language(files) == "python"

    def test_javascript_dominant(self, delegate):
        files = ["/a.js", "/b.jsx", "/c.py"]
        assert delegate._detect_container_language(files) == "javascript"

    def test_unknown_defaults_python(self, delegate):
        files = ["/a.rb", "/b.go"]
        assert delegate._detect_container_language(files) == "python"

    def test_empty_defaults_python(self, delegate):
        assert delegate._detect_container_language([]) == "python"


# ===========================================================================
# _get_compatible_files_for_tool
# ===========================================================================


class TestGetCompatibleFilesForTool:
    def test_python_filter(self, delegate):
        files = ["/a.py", "/b.js", "/c.pyw", "/d.ts"]
        result = delegate._get_compatible_files_for_tool("ruff", "python", files)
        assert len(result) == 2
        assert "/a.py" in result
        assert "/c.pyw" in result

    def test_unknown_language_returns_all(self, delegate):
        files = ["/a.py", "/b.js"]
        result = delegate._get_compatible_files_for_tool("some_tool", "ruby", files)
        assert len(result) == 2

    def test_typescript_includes_dts(self, delegate):
        files = ["/a.ts", "/b.d.ts", "/c.tsx", "/d.py"]
        result = delegate._get_compatible_files_for_tool("tsc", "typescript", files)
        assert len(result) == 3


# ===========================================================================
# _determine_primary_language
# ===========================================================================


class TestDeterminePrimaryLanguage:
    def test_empty(self, delegate):
        assert delegate._determine_primary_language({}) == "unknown"

    def test_single(self, delegate):
        assert delegate._determine_primary_language({"python": ["/a.py"]}) == "python"

    def test_mixed(self, delegate):
        result = delegate._determine_primary_language(
            {"python": ["/a.py", "/b.py"], "javascript": ["/c.js"]}
        )
        assert "mixed" in result
        assert "python" in result


# ===========================================================================
# _validate_quality_thresholds
# ===========================================================================


class TestValidateQualityThresholds:
    def test_fail_on_critical_raises(self, delegate):
        req = _make_scan_request(fail_on_critical=True)
        issues = [{"severity": "critical"}]
        with pytest.raises(ValueError, match="Critical issues found"):
            delegate._validate_quality_thresholds(req, issues, 50.0)

    def test_no_critical_no_raise(self, delegate):
        req = _make_scan_request(fail_on_critical=True)
        issues = [{"severity": "warning"}]
        delegate._validate_quality_thresholds(req, issues, 50.0)  # should not raise

    def test_min_score_logs_warning_but_no_raise(self, delegate):
        req = _make_scan_request(min_quality_score=80.0)
        delegate._validate_quality_thresholds(req, [], 50.0)  # should not raise


# ===========================================================================
# _create_combined_analysis_content
# ===========================================================================


class TestCreateCombinedAnalysisContent:
    def test_python(self, delegate):
        files = ["/a.py", "/b.py"]
        contents = {"/a.py": "x = 1", "/b.py": "y = 2"}
        result = delegate._create_combined_analysis_content(files, contents, "python")
        assert "# FILE: /a.py" in result
        assert "# FILE: /b.py" in result
        assert "x = 1" in result

    def test_javascript(self, delegate):
        files = ["/a.js"]
        contents = {"/a.js": "let x = 1;"}
        result = delegate._create_combined_analysis_content(files, contents, "javascript")
        assert "// FILE: /a.js" in result

    def test_unknown_language_uses_hash(self, delegate):
        files = ["/a.rb"]
        contents = {"/a.rb": "puts 'hi'"}
        result = delegate._create_combined_analysis_content(files, contents, "ruby")
        assert "# FILE: /a.rb" in result


# ===========================================================================
# _process_file_analysis_result
# ===========================================================================


class TestProcessFileAnalysisResult:
    def test_extracts_issues(self, delegate):
        issue = QualityIssue(
            tool="ruff",
            severity="error",
            message="bad",
            file_path="/old.py",
            line_number=5,
            column_number=2,
            rule_id="E001",
            suggested_fix="fix it",
        )
        result = _make_quality_result(
            issues=[issue],
            tools_results={
                "ruff": {
                    "success": True,
                    "issues": [
                        {
                            "severity": "warning",
                            "message": "tool issue",
                            "line": 3,
                            "column": 1,
                            "rule_id": "W001",
                        }
                    ],
                    "execution_time": 0.5,
                }
            },
        )
        issues, tools_results = delegate._process_file_analysis_result(
            result, "/target.py", "python"
        )
        # Should have 2 issues total: 1 from result.issues + 1 from tools_results
        assert len(issues) == 2
        # First issue from result.issues should have file_path overridden
        assert issues[0]["file_path"] == "/target.py"
        assert "ruff" in tools_results


# ===========================================================================
# _log_file_analysis_failure / _log_grouped_analysis_failure
# ===========================================================================


class TestLogFailures:
    def test_log_file_failure_with_tool_errors(self, delegate):
        result = _make_quality_result(
            success=False,
            error="",
            tools_results={
                "ruff": {"success": False, "error": "crashed"},
            },
        )
        # Should not raise
        delegate._log_file_analysis_failure("/a.py", result)

    def test_log_file_failure_without_tool_errors(self, delegate):
        result = _make_quality_result(success=False, error="general error")
        delegate._log_file_analysis_failure("/a.py", result)

    def test_log_file_failure_empty_error(self, delegate):
        result = _make_quality_result(success=False, error="")
        delegate._log_file_analysis_failure("/a.py", result)

    def test_log_grouped_failure_with_tool_errors(self, delegate):
        result = _make_quality_result(
            success=False,
            tools_results={"ty": {"success": False, "error": "type error"}},
        )
        delegate._log_grouped_analysis_failure(result)

    def test_log_grouped_failure_empty(self, delegate):
        result = _make_quality_result(success=False, error="")
        delegate._log_grouped_analysis_failure(result)


# ===========================================================================
# analyze_container_quality (integration-style with mocks)
# ===========================================================================


class TestAnalyzeContainerQuality:
    async def test_container_not_accessible(self, delegate, mock_docker_service):
        mock_docker_service.is_docker_available = MagicMock(return_value=False)
        req = _make_scan_request()
        result = await delegate.analyze_container_quality(req)
        assert result.success is False
        assert "not found or not accessible" in result.error

    async def test_no_files_found(self, delegate, mock_docker_service):
        # Container accessible, but no files found
        mock_docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": ""}
        )
        req = _make_scan_request()
        result = await delegate.analyze_container_quality(req)
        assert result.success is True
        assert result.total_issues == 0
        assert result.overall_score == 90.0

    async def test_successful_scan(self, delegate, mock_docker_service, mock_quality_runner):
        # First call: validation exec (echo test) -> success
        # Second call: find files -> returns files
        # Third+ calls: cat file -> returns content
        call_count = 0

        async def exec_side_effect(container, cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if cmd[0] == "echo":
                return {"exit_code": 0, "stdout": "test", "stderr": ""}
            if cmd[0] == "find":
                return {"exit_code": 0, "stdout": "/app/data/a.py\n"}
            if cmd[0] == "cat":
                return {"exit_code": 0, "stdout": "x = 1\n"}
            return {"exit_code": 0, "stdout": "", "stderr": ""}

        mock_docker_service.exec_in_container = AsyncMock(side_effect=exec_side_effect)
        mock_quality_runner.run_quality_analysis = AsyncMock(
            return_value=_make_quality_result(total_issues=1, issues=[], tools_results={})
        )

        req = _make_scan_request(parallel_analysis=False)
        result = await delegate.analyze_container_quality(req)
        assert result.success is True


# ===========================================================================
# _process_files_sequential
# ===========================================================================


class TestProcessFilesSequential:
    async def test_basic(self, delegate, mock_docker_service, mock_quality_runner):
        async def exec_side_effect(container, cmd, **kwargs):
            if cmd[0] == "cat":
                return {"exit_code": 0, "stdout": "x = 1"}
            return {"exit_code": 0, "stdout": "", "stderr": ""}

        mock_docker_service.exec_in_container = AsyncMock(side_effect=exec_side_effect)
        mock_quality_runner.run_quality_analysis = AsyncMock(return_value=_make_quality_result())

        req = _make_scan_request()
        issues, tools_results = await delegate._process_files_sequential(
            req, "analysis-1", ["/a.py"], "python", ["ruff"]
        )
        assert isinstance(issues, list)
        assert isinstance(tools_results, dict)
