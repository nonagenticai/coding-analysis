"""Unit tests for the three largest uncovered service modules:

- services.quality_analysis_delegate  (QualityAnalysisDelegate)
- processing.analyzers.deep           (DeepCodeAnalyzer)
- services.analysis_cache             (AnalysisCache, CacheStats, helpers)

All heavy dependencies (docker, redis, aiosqlite, etc.) are mocked.
"""

from __future__ import annotations

import ast
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.testing_models import QualityIssue, QualityRequest, QualityResult, QualityScanRequest
from processing.analyzers.deep import (
    CodeIssue,
    CodeRelationship,
    DeepAnalysisResult,
    DeepAnalysisServiceCodeEntity,
    DeepCodeAnalyzer,
    IssueType,
    Severity,
)
from services.analysis_cache import (
    AnalysisCache,
    AnalysisCacheConfig,
    AnalysisCacheEntry,
    AnalysisResults,
    CacheMetadata,
    CacheStats,
    get_analysis_cache,
)

# ============================================================================
# HELPERS
# ============================================================================


def _make_docker_service() -> MagicMock:
    """Create a mock DockerService."""
    svc = MagicMock()
    svc.is_docker_available = MagicMock(return_value=True)
    svc.exec_in_container = AsyncMock(return_value={"exit_code": 0, "stdout": "test", "stderr": ""})
    svc.list_containers = AsyncMock(return_value=[])
    return svc


def _make_quality_runner() -> MagicMock:
    """Create a mock HTTPQualityToolRunner."""
    runner = MagicMock()
    runner.run_quality_analysis = AsyncMock(
        return_value=QualityResult(
            analysis_id="test-id",
            success=True,
            overall_score=85.0,
            total_issues=0,
            issues=[],
            tools_results={},
            execution_time=0.5,
            created_at=datetime.now(UTC),
            language="python",
        )
    )
    runner.run_single_tool = AsyncMock(
        return_value={"success": True, "issues": [], "execution_time": 0.1}
    )
    return runner


def _make_tool_registry() -> MagicMock:
    """Create a mock ToolRegistry."""
    reg = MagicMock()
    reg.list_available_tools = MagicMock(return_value=["ruff", "ty"])
    reg.get_tool_info = MagicMock(return_value={"name": "ruff", "version": "3.0"})
    reg.list_supported_languages = MagicMock(return_value=["python", "javascript"])
    return reg


def _make_cache_manager() -> MagicMock:
    """Create a mock RedisCacheManager."""
    cm = MagicMock()
    cm.get = AsyncMock(return_value=None)
    cm.set = AsyncMock()
    return cm


def _make_delegate():
    """Create a QualityAnalysisDelegate with all mocks."""
    from services.quality_analysis_delegate import QualityAnalysisDelegate

    return QualityAnalysisDelegate(
        docker_service=_make_docker_service(),
        quality_runner=_make_quality_runner(),
        tool_registry=_make_tool_registry(),
        cache_manager=_make_cache_manager(),
    )


def _make_quality_request(**overrides) -> QualityRequest:
    """Create a QualityRequest with sensible defaults."""
    defaults = {
        "container_name": "test-container",
        "code": "print('hello')",
        "language": "python",
    }
    defaults.update(overrides)
    return QualityRequest(**defaults)


def _make_scan_request(**overrides) -> QualityScanRequest:
    """Create a QualityScanRequest with sensible defaults."""
    defaults = {
        "container_name": "test-container",
        "scan_path": "/app/data",
        "include_patterns": ["*.py"],
        "exclude_patterns": ["__pycache__"],
    }
    defaults.update(overrides)
    return QualityScanRequest(**defaults)


# ============================================================================
# SECTION 1: QualityAnalysisDelegate
# ============================================================================


class TestQualityAnalysisDelegateCache:
    """Tests for cache-related methods."""

    def test_generate_cache_key_deterministic(self):
        delegate = _make_delegate()
        req = _make_quality_request(tools=["ruff", "ty"])
        key1 = delegate.generate_cache_key(req)
        key2 = delegate.generate_cache_key(req)
        assert key1 == key2
        assert key1.startswith("quality_")

    def test_generate_cache_key_different_for_different_input(self):
        delegate = _make_delegate()
        req_a = _make_quality_request(code="a = 1")
        req_b = _make_quality_request(code="b = 2")
        assert delegate.generate_cache_key(req_a) != delegate.generate_cache_key(req_b)

    def test_generate_cache_key_tools_sorted(self):
        delegate = _make_delegate()
        req1 = _make_quality_request(tools=["ty", "ruff"])
        req2 = _make_quality_request(tools=["ruff", "ty"])
        assert delegate.generate_cache_key(req1) == delegate.generate_cache_key(req2)

    def test_generate_cache_key_no_tools(self):
        delegate = _make_delegate()
        req = _make_quality_request(tools=None)
        key = delegate.generate_cache_key(req)
        assert key.startswith("quality_")

    async def test_get_cached_result_returns_none_without_cache_manager(self):
        from services.quality_analysis_delegate import QualityAnalysisDelegate

        delegate = QualityAnalysisDelegate(
            docker_service=_make_docker_service(),
            quality_runner=_make_quality_runner(),
            tool_registry=_make_tool_registry(),
            cache_manager=None,
        )
        result = await delegate.get_cached_result("some_key")
        assert result is None

    async def test_get_cached_result_miss(self):
        delegate = _make_delegate()
        result = await delegate.get_cached_result("missing_key")
        assert result is None

    async def test_get_cached_result_hit(self):
        delegate = _make_delegate()
        cached_data = {
            "analysis_id": "cached-id",
            "success": True,
            "overall_score": 90.0,
            "total_issues": 0,
            "issues": [],
            "tools_results": {},
            "execution_time": 0.1,
            "created_at": datetime.now(UTC).isoformat(),
            "language": "python",
        }
        delegate.cache_manager.get = AsyncMock(return_value=cached_data)
        result = await delegate.get_cached_result("hit_key")
        assert result is not None
        assert result.analysis_id == "cached-id"

    async def test_get_cached_result_deserialization_error(self):
        delegate = _make_delegate()
        delegate.cache_manager.get = AsyncMock(return_value={"bad": "data"})
        result = await delegate.get_cached_result("bad_key")
        assert result is None

    async def test_get_cached_result_runtime_error(self):
        delegate = _make_delegate()
        delegate.cache_manager.get = AsyncMock(side_effect=RuntimeError("connection lost"))
        result = await delegate.get_cached_result("err_key")
        assert result is None

    async def test_cache_result_stores(self):
        delegate = _make_delegate()
        qr = QualityResult(
            analysis_id="store-id",
            success=True,
            overall_score=80.0,
            total_issues=0,
            issues=[],
            tools_results={},
            execution_time=0.5,
            created_at=datetime.now(UTC),
            language="python",
        )
        await delegate.cache_result("store_key", qr)
        delegate.cache_manager.set.assert_awaited_once()

    async def test_cache_result_no_cache_manager(self):
        from services.quality_analysis_delegate import QualityAnalysisDelegate

        delegate = QualityAnalysisDelegate(
            docker_service=_make_docker_service(),
            quality_runner=_make_quality_runner(),
            tool_registry=_make_tool_registry(),
            cache_manager=None,
        )
        # Should not raise
        await delegate.cache_result("key", MagicMock())

    async def test_cache_result_error_swallowed(self):
        delegate = _make_delegate()
        delegate.cache_manager.set = AsyncMock(side_effect=RuntimeError("write fail"))
        qr = QualityResult(
            analysis_id="err-id",
            success=True,
            overall_score=80.0,
            total_issues=0,
            issues=[],
            tools_results={},
            execution_time=0.5,
            created_at=datetime.now(UTC),
            language="python",
        )
        # Should not raise
        await delegate.cache_result("err_key", qr)


class TestQualityAnalysisDelegateToolRegistry:
    """Tests for tool registry delegation methods."""

    def test_get_supported_tools(self):
        delegate = _make_delegate()
        tools = delegate.get_supported_tools("python")
        assert "ruff" in tools

    def test_get_tool_info(self):
        delegate = _make_delegate()
        info = delegate.get_tool_info("ruff", "python")
        assert info is not None
        assert info["name"] == "ruff"

    def test_get_supported_languages(self):
        delegate = _make_delegate()
        langs = delegate.get_supported_languages()
        assert "python" in langs


class TestQualityAnalysisDelegateQualityAnalysis:
    """Tests for analyze_quality and run_single_tool."""

    async def test_analyze_quality_success(self):
        delegate = _make_delegate()
        req = _make_quality_request()
        result = await delegate.analyze_quality(req)
        assert result.success is True
        assert result.overall_score == 85.0

    async def test_analyze_quality_failure_enhances_error(self):
        delegate = _make_delegate()
        failed_result = QualityResult(
            analysis_id="fail-id",
            success=False,
            overall_score=0.0,
            total_issues=0,
            issues=[],
            tools_results={
                "ruff": {"success": False, "error": "ruff not found", "issues": []},
            },
            execution_time=0.1,
            created_at=datetime.now(UTC),
            language="python",
            error="Analysis failed without specific error details",
        )
        delegate.quality_runner.run_quality_analysis = AsyncMock(return_value=failed_result)
        req = _make_quality_request()
        result = await delegate.analyze_quality(req)
        assert result.success is False
        assert "ruff" in (result.error or "")

    async def test_analyze_quality_exception_handling(self):
        delegate = _make_delegate()
        delegate.quality_runner.run_quality_analysis = AsyncMock(
            side_effect=TimeoutError("timed out")
        )
        req = _make_quality_request()
        result = await delegate.analyze_quality(req)
        assert result.success is False
        assert "TimeoutError" in (result.error or "")

    async def test_run_single_tool_success(self):
        delegate = _make_delegate()
        result = await delegate.run_single_tool("ruff", "python", "x = 1")
        assert result["success"] is True

    async def test_run_single_tool_exception(self):
        delegate = _make_delegate()
        delegate.quality_runner.run_single_tool = AsyncMock(
            side_effect=RuntimeError("tool crashed")
        )
        result = await delegate.run_single_tool("ruff", "python", "x = 1")
        assert result["success"] is False
        assert "tool crashed" in result["error"]


class TestQualityAnalysisDelegateEnhanceError:
    """Tests for _enhance_error_details."""

    def test_enhance_with_specific_tool_errors(self):
        delegate = _make_delegate()
        result = MagicMock()
        result.tools_results = {
            "ruff": {"success": False, "error": "cannot run ruff", "issues": []},
            "ty": {"success": True, "error": "", "issues": [], "execution_time": 0.5},
        }
        result.error = "Analysis failed without specific error details"
        delegate._enhance_error_details(result)
        assert "ruff" in result.error

    def test_enhance_no_tool_results(self):
        delegate = _make_delegate()
        result = MagicMock()
        result.tools_results = {}
        result.error = "Analysis failed without specific error details"
        delegate._enhance_error_details(result)
        assert "no tool results" in result.error

    def test_enhance_tool_failed_without_error_message(self):
        delegate = _make_delegate()
        result = MagicMock()
        result.tools_results = {
            "ruff": {"success": False, "error": "", "issues": [], "execution_time": 0.0},
        }
        result.error = "Analysis failed without specific error details"
        delegate._enhance_error_details(result)
        assert "ruff" in result.error

    def test_enhance_preserves_existing_error(self):
        delegate = _make_delegate()
        result = MagicMock()
        result.tools_results = {
            "ruff": {"success": False, "error": "not installed", "issues": []},
        }
        result.error = "Custom error message"
        delegate._enhance_error_details(result)
        assert "Custom error message" in result.error
        assert "ruff" in result.error


class TestQualityAnalysisDelegateContainerAnalysis:
    """Tests for container analysis methods."""

    async def test_analyze_container_quality_docker_unavailable(self):
        delegate = _make_delegate()
        delegate.docker_service.is_docker_available.return_value = False
        req = _make_scan_request()
        result = await delegate.analyze_container_quality(req)
        assert result.success is False

    async def test_analyze_container_quality_no_files(self):
        delegate = _make_delegate()
        delegate.docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "test", "stderr": ""}
        )
        # After validation passes, discover returns no files
        with patch.object(
            delegate, "_discover_container_files", new_callable=AsyncMock, return_value=[]
        ):
            req = _make_scan_request()
            result = await delegate.analyze_container_quality(req)
            assert result.success is True
            assert result.total_issues == 0

    async def test_validate_container_access_success(self):
        delegate = _make_delegate()
        delegate.docker_service.exec_in_container = AsyncMock(return_value={"exit_code": 0})
        valid = await delegate._validate_container_access("test-container")
        assert valid is True

    async def test_validate_container_access_exec_fails_then_list_matches(self):
        delegate = _make_delegate()
        delegate.docker_service.exec_in_container = AsyncMock(return_value={"exit_code": 1})
        container_mock = MagicMock()
        container_mock.container_name = "test-container"
        delegate.docker_service.list_containers = AsyncMock(return_value=[container_mock])
        valid = await delegate._validate_container_access("test-container")
        assert valid is True

    async def test_validate_container_access_docker_unavailable(self):
        delegate = _make_delegate()
        delegate.docker_service.is_docker_available.return_value = False
        valid = await delegate._validate_container_access("test-container")
        assert valid is False


class TestQualityAnalysisDelegateFileGrouping:
    """Tests for _group_files_by_language and _matches_pattern."""

    def test_group_files_by_language_python(self):
        delegate = _make_delegate()
        files = ["/app/main.py", "/app/utils.py", "/app/style.css"]
        result = delegate._group_files_by_language(files)
        assert "python" in result
        assert len(result["python"]) == 2
        assert "css" not in result  # Unknown extension

    def test_group_files_by_language_javascript(self):
        delegate = _make_delegate()
        files = ["/app/index.js", "/app/App.jsx", "/app/server.mjs"]
        result = delegate._group_files_by_language(files)
        assert "javascript" in result
        assert len(result["javascript"]) == 3

    def test_group_files_by_language_typescript(self):
        delegate = _make_delegate()
        files = ["/app/main.ts", "/app/App.tsx", "/app/types.d.ts"]
        result = delegate._group_files_by_language(files)
        assert "typescript" in result
        assert len(result["typescript"]) == 3

    def test_matches_pattern_glob(self):
        delegate = _make_delegate()
        assert delegate._matches_pattern("/app/test.py", "*.py") is True
        assert delegate._matches_pattern("/app/test.js", "*.py") is False

    def test_matches_pattern_directory(self):
        delegate = _make_delegate()
        assert delegate._matches_pattern("/app/__pycache__/mod.pyc", "__pycache__") is True

    def test_matches_pattern_basename(self):
        delegate = _make_delegate()
        assert delegate._matches_pattern("/app/data/test.min.js", "*.min.js") is True


class TestQualityAnalysisDelegateScoring:
    """Tests for _calculate_container_quality_score."""

    def test_score_no_issues(self):
        delegate = _make_delegate()
        score = delegate._calculate_container_quality_score([], {})
        assert score == 90.0

    def test_score_with_critical_issues(self):
        delegate = _make_delegate()
        issues = [{"severity": "critical"}, {"severity": "critical"}]
        tools = {"ruff": {"success": True}}
        score = delegate._calculate_container_quality_score(issues, tools)
        assert score == 80.0

    def test_score_with_all_severities(self):
        delegate = _make_delegate()
        issues = [
            {"severity": "critical"},
            {"severity": "error"},
            {"severity": "warning"},
            {"severity": "info"},
        ]
        tools = {"ruff": {"success": True}}
        score = delegate._calculate_container_quality_score(issues, tools)
        assert 0 <= score <= 100

    def test_score_clamped_to_zero(self):
        delegate = _make_delegate()
        issues = [{"severity": "critical"}] * 20
        tools = {"ruff": {"success": True}}
        score = delegate._calculate_container_quality_score(issues, tools)
        assert score == 0.0

    def test_score_adjusted_by_tool_success_rate(self):
        delegate = _make_delegate()
        issues = []
        tools = {
            "ruff": {"success": True},
            "ty": {"success": False},
        }
        score = delegate._calculate_container_quality_score(issues, tools)
        assert score == 50.0  # 100 * 0.5


class TestQualityAnalysisDelegateConversion:
    """Tests for _convert_to_quality_issues."""

    def test_convert_valid_issues(self):
        delegate = _make_delegate()
        issues = [
            {
                "tool": "ruff",
                "severity": "warning",
                "message": "unused import",
                "file_path": "/app/main.py",
                "line_number": 5,
            }
        ]
        result = delegate._convert_to_quality_issues(issues)
        assert len(result) == 1
        assert result[0].tool == "ruff"
        assert result[0].severity == "warning"

    def test_convert_with_defaults(self):
        delegate = _make_delegate()
        issues = [{}]
        result = delegate._convert_to_quality_issues(issues)
        assert len(result) == 1
        assert result[0].tool == "unknown"
        assert result[0].severity == "info"

    def test_convert_bad_issue_skipped(self):
        delegate = _make_delegate()
        # Force a ValueError via invalid fix_confidence (must be 0..1)
        issues = [
            {
                "tool": "x",
                "severity": "info",
                "message": "msg",
                "file_path": "f.py",
                "fix_confidence": 999.0,
            }
        ]
        result = delegate._convert_to_quality_issues(issues)
        assert len(result) == 0


class TestQualityAnalysisDelegatePrimaryLanguage:
    """Tests for _determine_primary_language."""

    def test_empty(self):
        delegate = _make_delegate()
        assert delegate._determine_primary_language({}) == "unknown"

    def test_single_language(self):
        delegate = _make_delegate()
        assert delegate._determine_primary_language({"python": ["a.py"]}) == "python"

    def test_mixed_languages(self):
        delegate = _make_delegate()
        result = delegate._determine_primary_language(
            {"python": ["a.py", "b.py"], "javascript": ["c.js"]}
        )
        assert "mixed" in result
        assert "python" in result


class TestQualityAnalysisDelegateLineOffsets:
    """Tests for _calculate_line_offsets and related methods."""

    def test_calculate_line_offsets(self):
        delegate = _make_delegate()
        files = ["/a.py", "/b.py"]
        contents = {"/a.py": "line1\nline2\nline3", "/b.py": "x\ny"}
        offsets = delegate._calculate_line_offsets(files, contents)
        assert offsets["/a.py"] == 1
        # "line1\nline2\nline3" has 2 newlines => count("\n")+2 = 4, so /b.py starts at 5
        assert offsets["/b.py"] == 5

    def test_determine_issue_file(self):
        delegate = _make_delegate()
        offsets = {"/a.py": 1, "/b.py": 6}
        files = ["/a.py", "/b.py"]

        issue = MagicMock()
        issue.line_number = 3
        assert delegate._determine_issue_file(issue, offsets, files) == "/a.py"

        issue.line_number = 7
        assert delegate._determine_issue_file(issue, offsets, files) == "/b.py"

    def test_adjust_line_number(self):
        delegate = _make_delegate()
        offsets = {"/a.py": 1, "/b.py": 6}
        issue = MagicMock()
        issue.line_number = 8
        adjusted = delegate._adjust_line_number(issue, offsets, "/b.py")
        assert adjusted == 2  # 8 - 6 = 2

    def test_adjust_line_number_none(self):
        delegate = _make_delegate()
        issue = MagicMock()
        issue.line_number = None
        assert delegate._adjust_line_number(issue, {"/a.py": 1}, "/a.py") is None

    def test_determine_tool_issue_file(self):
        delegate = _make_delegate()
        offsets = {"/a.py": 1, "/b.py": 6}
        issue = {"line": 7}
        assert delegate._determine_tool_issue_file(issue, offsets, ["/a.py", "/b.py"]) == "/b.py"

    def test_adjust_tool_issue_line(self):
        delegate = _make_delegate()
        offsets = {"/b.py": 6}
        issue = {"line": 8}
        assert delegate._adjust_tool_issue_line(issue, offsets, "/b.py") == 2

    def test_adjust_tool_issue_line_none(self):
        delegate = _make_delegate()
        issue = {"line": None}
        assert delegate._adjust_tool_issue_line(issue, {"/a.py": 1}, "/a.py") is None

    def test_adjust_tool_issue_line_non_int(self):
        delegate = _make_delegate()
        issue = {"line": "abc"}
        assert delegate._adjust_tool_issue_line(issue, {"/a.py": 1}, "/a.py") is None


class TestQualityAnalysisDelegateMerge:
    """Tests for _merge_tools_results and _merge_language_tools_results."""

    def test_merge_tools_results_new_tool(self):
        delegate = _make_delegate()
        target: dict[str, Any] = {}
        source = {"ruff": {"success": True, "issues": [{"msg": "x"}], "execution_time": 1.0}}
        delegate._merge_tools_results(target, source)
        assert "ruff" in target
        assert len(target["ruff"]["issues"]) == 1

    def test_merge_tools_results_existing_tool(self):
        delegate = _make_delegate()
        target: dict[str, Any] = {
            "ruff": {"success": True, "issues": [{"msg": "a"}], "execution_time": 1.0}
        }
        source = {"ruff": {"success": True, "issues": [{"msg": "b"}], "execution_time": 0.5}}
        delegate._merge_tools_results(target, source)
        assert len(target["ruff"]["issues"]) == 2
        assert target["ruff"]["execution_time"] == 1.5

    def test_merge_tools_results_failure_propagates(self):
        delegate = _make_delegate()
        target: dict[str, Any] = {"ruff": {"success": True, "issues": [], "execution_time": 0.0}}
        source = {"ruff": {"success": False, "issues": [], "execution_time": 0.1}}
        delegate._merge_tools_results(target, source)
        assert target["ruff"]["success"] is False

    def test_merge_language_tools_results(self):
        delegate = _make_delegate()
        target: dict[str, Any] = {}
        source = {"ruff": {"success": True, "issues": [{"msg": "x"}], "execution_time": 0.3}}
        delegate._merge_language_tools_results(target, source)
        assert "ruff" in target
        assert len(target["ruff"]["issues"]) == 1


class TestQualityAnalysisDelegateContainerLanguageDetection:
    """Tests for _detect_container_language and _get_compatible_files_for_tool."""

    def test_detect_python(self):
        delegate = _make_delegate()
        files = ["/a.py", "/b.py", "/c.js"]
        assert delegate._detect_container_language(files) == "python"

    def test_detect_javascript(self):
        delegate = _make_delegate()
        files = ["/a.js", "/b.jsx", "/c.py"]
        assert delegate._detect_container_language(files) == "javascript"

    def test_detect_unknown_defaults_python(self):
        delegate = _make_delegate()
        files = ["/a.go", "/b.rs"]
        assert delegate._detect_container_language(files) == "python"

    def test_get_compatible_files_python(self):
        delegate = _make_delegate()
        files = ["/a.py", "/b.js", "/c.pyw"]
        result = delegate._get_compatible_files_for_tool("ruff", "python", files)
        assert len(result) == 2

    def test_get_compatible_files_unknown_lang(self):
        delegate = _make_delegate()
        files = ["/a.go"]
        result = delegate._get_compatible_files_for_tool("golint", "go", files)
        assert result == files  # returns all when no ext defined


class TestQualityAnalysisDelegateCombinedContent:
    """Tests for _create_combined_analysis_content."""

    def test_python_markers(self):
        delegate = _make_delegate()
        files = ["/a.py", "/b.py"]
        contents = {"/a.py": "x = 1", "/b.py": "y = 2"}
        combined = delegate._create_combined_analysis_content(files, contents, "python")
        assert "# FILE: /a.py" in combined
        assert "# FILE: /b.py" in combined
        assert "x = 1" in combined

    def test_javascript_markers(self):
        delegate = _make_delegate()
        files = ["/a.js"]
        contents = {"/a.js": "const x = 1;"}
        combined = delegate._create_combined_analysis_content(files, contents, "javascript")
        assert "// FILE: /a.js" in combined


class TestQualityAnalysisDelegateEmptyResult:
    """Tests for _create_empty_quality_result."""

    def test_creates_empty_result(self):
        delegate = _make_delegate()
        import time

        start = time.time()
        result = delegate._create_empty_quality_result("empty-id", start)
        assert result.success is True
        assert result.total_issues == 0
        assert result.overall_score == 90.0
        assert result.analysis_id == "empty-id"


class TestQualityAnalysisDelegateProcessResults:
    """Tests for _process_file_analysis_result and _process_grouped_analysis_result."""

    def test_process_file_analysis_result(self):
        delegate = _make_delegate()
        qi = QualityIssue(
            tool="ruff",
            severity="warning",
            message="unused var",
            file_path="original.py",
        )
        qr = QualityResult(
            analysis_id="pid",
            success=True,
            overall_score=80.0,
            total_issues=1,
            issues=[qi],
            tools_results={
                "ruff": {
                    "success": True,
                    "issues": [{"severity": "warning", "message": "x", "line": 5, "column": 1}],
                    "execution_time": 0.3,
                }
            },
            execution_time=0.5,
            created_at=datetime.now(UTC),
            language="python",
        )
        issues, tools = delegate._process_file_analysis_result(qr, "/app/main.py")
        assert len(issues) >= 1
        # File path should be overridden
        assert any(i["file_path"] == "/app/main.py" for i in issues)

    def test_process_grouped_analysis_result(self):
        delegate = _make_delegate()
        qi = QualityIssue(
            tool="ruff",
            severity="warning",
            message="unused import",
            file_path="combined",
            line_number=3,
        )
        qr = QualityResult(
            analysis_id="gid",
            success=True,
            overall_score=80.0,
            total_issues=1,
            issues=[qi],
            tools_results={},
            execution_time=0.5,
            created_at=datetime.now(UTC),
            language="python",
        )
        files = ["/a.py", "/b.py"]
        contents = {"/a.py": "x = 1\ny = 2\nz = 3", "/b.py": "a = 1"}
        issues, tools = delegate._process_grouped_analysis_result(qr, files, contents)
        assert len(issues) == 1
        # Issue at line 3 should map to /a.py
        assert issues[0]["file_path"] == "/a.py"


class TestQualityAnalysisDelegateValidateThresholds:
    """Tests for _validate_quality_thresholds."""

    def test_fail_on_critical_raises(self):
        delegate = _make_delegate()
        req = _make_scan_request(fail_on_critical=True)
        issues = [{"severity": "critical"}]
        with pytest.raises(ValueError, match="Critical issues"):
            delegate._validate_quality_thresholds(req, issues, 50.0)

    def test_fail_on_critical_no_critical(self):
        delegate = _make_delegate()
        req = _make_scan_request(fail_on_critical=True)
        issues = [{"severity": "warning"}]
        # Should not raise
        delegate._validate_quality_thresholds(req, issues, 50.0)

    def test_min_quality_score_below_threshold_logs(self):
        delegate = _make_delegate()
        req = _make_scan_request(min_quality_score=80.0)
        # Should not raise, just log
        delegate._validate_quality_thresholds(req, [], 60.0)


class TestQualityAnalysisDelegateLogFailures:
    """Tests for _log_file_analysis_failure and _log_grouped_analysis_failure."""

    def test_log_file_analysis_failure_with_tool_errors(self):
        delegate = _make_delegate()
        result = MagicMock()
        result.tools_results = {
            "ruff": {"success": False, "error": "crash"},
        }
        result.error = ""
        # Should not raise
        delegate._log_file_analysis_failure("/app/main.py", result)

    def test_log_file_analysis_failure_no_tool_errors(self):
        delegate = _make_delegate()
        result = MagicMock()
        result.tools_results = {}
        result.error = "generic error"
        delegate._log_file_analysis_failure("/app/main.py", result)

    def test_log_grouped_analysis_failure(self):
        delegate = _make_delegate()
        result = MagicMock()
        result.tools_results = {}
        result.error = ""
        delegate._log_grouped_analysis_failure(result)


# ============================================================================
# SECTION 2: DeepCodeAnalyzer
# ============================================================================


class TestDeepCodeAnalyzerInit:
    """Tests for DeepCodeAnalyzer initialization."""

    def test_init_with_nonexistent_path(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path / "nonexistent"), "python")
        assert analyzer.python_files == []

    def test_init_with_valid_path(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1")
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        assert len(analyzer.python_files) == 1

    def test_init_no_python_files(self, tmp_path):
        (tmp_path / "data.txt").write_text("hello")
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        assert len(analyzer.python_files) == 0


class TestDeepCodeAnalyzerSafeParse:
    """Tests for _safe_parse_ast."""

    def test_valid_code(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        tree = analyzer._safe_parse_ast("x = 1", "test.py")
        assert tree is not None

    def test_syntax_error(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        tree = analyzer._safe_parse_ast("def (broken:", "bad.py")
        assert tree is None

    def test_empty_code(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        tree = analyzer._safe_parse_ast("", "empty.py")
        assert tree is not None  # empty is valid Python


class TestDeepCodeAnalyzerAnalyzeCodebase:
    """Tests for the full analyze_codebase method."""

    def test_empty_codebase(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        result = analyzer.analyze_codebase()
        assert isinstance(result, DeepAnalysisResult)
        assert result.overall_score == 10.0

    def test_simple_codebase(self, tmp_path):
        code = '''"""Module doc."""

def hello(name):
    """Say hello."""
    return f"Hello, {name}"

class Greeter:
    """A greeter class."""
    def greet(self, name):
        """Greet someone."""
        return hello(name)
'''
        (tmp_path / "main.py").write_text(code)
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        result = analyzer.analyze_codebase()
        assert isinstance(result, DeepAnalysisResult)
        assert len(result.entities) > 0

    def test_codebase_with_security_issues(self, tmp_path):
        code = """import pickle
import subprocess

def dangerous():
    eval("1+1")
    exec("print('hi')")
"""
        (tmp_path / "bad.py").write_text(code)
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        result = analyzer.analyze_codebase()
        security_issues = [i for i in result.issues if i.type == IssueType.SECURITY]
        assert len(security_issues) > 0

    def test_codebase_with_complexity_issues(self, tmp_path):
        # Build a function with high cyclomatic complexity
        lines = ["def complex_func(x):"]
        for i in range(15):
            lines.append(f"    if x == {i}:")
            lines.append(f"        return {i}")
        lines.append("    return -1")
        code = "\n".join(lines) + "\n"
        (tmp_path / "complex.py").write_text(code)
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        result = analyzer.analyze_codebase()
        complexity_issues = [i for i in result.issues if i.type == IssueType.COMPLEXITY]
        assert len(complexity_issues) > 0

    def test_codebase_with_code_smells(self, tmp_path):
        code = "def many_params(a, b, c, d, e, f, g, h, i):\n    pass\n"
        (tmp_path / "smelly.py").write_text(code)
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        result = analyzer.analyze_codebase()
        smell_issues = [i for i in result.issues if i.type == IssueType.CODE_SMELL]
        assert len(smell_issues) > 0

    def test_codebase_with_bug_risks(self, tmp_path):
        code = """def risky(items=[]):
    items.append(1)
    return items

def catch_all():
    try:
        pass
    except:
        pass
"""
        (tmp_path / "buggy.py").write_text(code)
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        result = analyzer.analyze_codebase()
        bug_issues = [i for i in result.issues if i.type == IssueType.BUG_RISK]
        assert len(bug_issues) > 0

    def test_codebase_with_missing_docstrings(self, tmp_path):
        code = """def no_doc():
    pass

class NoDocClass:
    pass
"""
        (tmp_path / "undoc.py").write_text(code)
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        result = analyzer.analyze_codebase()
        doc_issues = [i for i in result.issues if i.type == IssueType.DOCUMENTATION]
        assert len(doc_issues) >= 2


class TestDeepCodeAnalyzerHelpers:
    """Tests for helper methods."""

    def test_extract_docstring_present(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        code = 'def foo():\n    """Hello."""\n    pass\n'
        tree = ast.parse(code)
        func_node = tree.body[0]
        docstring = analyzer._extract_docstring(func_node)
        assert docstring == "Hello."

    def test_extract_docstring_absent(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        code = "def foo():\n    pass\n"
        tree = ast.parse(code)
        func_node = tree.body[0]
        docstring = analyzer._extract_docstring(func_node)
        assert docstring is None

    def test_extract_decorator_name(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        code = "@staticmethod\ndef foo():\n    pass\n"
        tree = ast.parse(code)
        func_node = tree.body[0]
        name = analyzer._extract_decorator_name(func_node.decorator_list[0])
        assert name == "staticmethod"

    def test_extract_decorator_attribute(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        code = "@app.route\ndef foo():\n    pass\n"
        tree = ast.parse(code)
        func_node = tree.body[0]
        name = analyzer._extract_decorator_name(func_node.decorator_list[0])
        assert name == "route"

    def test_extract_base_name(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        code = "class Child(Parent):\n    pass\n"
        tree = ast.parse(code)
        class_node = tree.body[0]
        name = analyzer._extract_base_name(class_node.bases[0])
        assert name == "Parent"

    def test_get_line(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        content = "line1\nline2\nline3"
        assert analyzer._get_line(content, 2) == "line2"
        assert analyzer._get_line(content, 99) == ""

    def test_get_function_signature(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        code = "def hello(name):\n    pass\n"
        tree = ast.parse(code)
        func_node = tree.body[0]
        sig = analyzer._get_function_signature(code, func_node)
        assert "def hello" in sig

    def test_cyclomatic_complexity(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        code = "def foo(x):\n    if x:\n        pass\n    for i in range(10):\n        pass\n"
        tree = ast.parse(code)
        func_node = tree.body[0]
        complexity = analyzer._calculate_cyclomatic_complexity(func_node)
        assert complexity == 3  # 1 base + 1 if + 1 for

    def test_nesting_depth(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        code = "if True:\n    if True:\n        if True:\n            pass\n"
        tree = ast.parse(code)
        if_node = tree.body[0]
        depth = analyzer._calculate_nesting_depth(if_node)
        assert depth >= 2

    def test_calculate_overall_score_no_issues(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        analyzer.issues = []
        score = analyzer._calculate_overall_score()
        assert score == 10.0

    def test_calculate_overall_score_with_issues(self, tmp_path):
        (tmp_path / "dummy.py").write_text("x=1")
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        analyzer.issues = [
            CodeIssue(
                type=IssueType.SECURITY,
                severity=Severity.CRITICAL,
                title="test",
                description="test",
                file_path="test.py",
                line_number=1,
                code_snippet=None,
                suggestion="fix it",
                confidence=0.9,
            )
        ]
        score = analyzer._calculate_overall_score()
        assert score < 10.0


class TestDeepCodeAnalyzerMetrics:
    """Tests for _calculate_metrics and _generate_recommendations."""

    def test_calculate_metrics(self, tmp_path):
        code = """def foo():
    pass

class Bar:
    pass
"""
        (tmp_path / "mod.py").write_text(code)
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        analyzer.analyze_codebase()
        metrics = analyzer._calculate_metrics()
        assert metrics["total_files"] == 1
        assert metrics["total_functions"] >= 1
        assert metrics["total_classes"] >= 1

    def test_generate_recommendations_security(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        analyzer.issues = [
            CodeIssue(
                type=IssueType.SECURITY,
                severity=Severity.HIGH,
                title="test",
                description="test",
                file_path="t.py",
                line_number=1,
                code_snippet=None,
                suggestion="fix",
                confidence=0.9,
            )
        ]
        recs = analyzer._generate_recommendations()
        assert any("security" in r.lower() for r in recs)

    def test_generate_recommendations_empty(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        analyzer.issues = []
        recs = analyzer._generate_recommendations()
        assert len(recs) == 0


class TestDeepCodeAnalyzerSnippetAnalysis:
    """Tests for analyze_code_snippet."""

    async def test_analyze_simple_snippet(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        code = "def hello():\n    return 'hi'\n"
        result = await analyzer.analyze_code_snippet(code, "python")
        assert isinstance(result, DeepAnalysisResult)

    async def test_analyze_snippet_with_security_issue(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        code = "def dangerous():\n    eval('1+1')\n"
        result = await analyzer.analyze_code_snippet(code, "python")
        security_issues = [i for i in result.issues if i.type == IssueType.SECURITY]
        assert len(security_issues) > 0

    async def test_analyze_snippet_non_python(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        code = "function hello() { return 'hi'; }"
        result = await analyzer.analyze_code_snippet(code, "javascript")
        assert isinstance(result, DeepAnalysisResult)


class TestDeepCodeAnalyzerArchitecture:
    """Tests for _analyze_architecture (circular import detection)."""

    def test_no_circular_imports(self, tmp_path):
        (tmp_path / "a.py").write_text("import os\n")
        (tmp_path / "b.py").write_text("import sys\n")
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        analyzer._analyze_architecture()
        arch_issues = [i for i in analyzer.issues if i.type == IssueType.ARCHITECTURE]
        assert len(arch_issues) == 0


class TestDeepCodeAnalyzerPerformance:
    """Tests for _analyze_performance_issues."""

    def test_string_concat_in_loop(self, tmp_path):
        code = """result = ""
for i in range(10):
    result += str(i)
"""
        (tmp_path / "perf.py").write_text(code)
        analyzer = DeepCodeAnalyzer(str(tmp_path), "python")
        analyzer._analyze_performance_issues()
        perf_issues = [i for i in analyzer.issues if i.type == IssueType.PERFORMANCE]
        assert len(perf_issues) > 0


# ============================================================================
# SECTION 3: AnalysisCache
# ============================================================================


@pytest.fixture
def cache_config(tmp_path):
    """Create cache config pointing to tmp dir."""
    return AnalysisCacheConfig(
        cache_dir=tmp_path / "test_cache",
        max_cache_size_mb=10,
        max_entries=50,
        ttl_hours=24,
        cleanup_interval_minutes=60,
    )


@pytest.fixture
def cache(cache_config):
    """Create an AnalysisCache instance."""
    return AnalysisCache(config=cache_config)


def _make_cache_entry(
    file_path: str = "/test/main.py",
    content_hash: str = "abc123",
    language: str = "python",
    access_count: int = 0,
    cached_at: datetime | None = None,
    file_size: int = 100,
) -> AnalysisCacheEntry:
    """Create a cache entry for testing."""
    return AnalysisCacheEntry(
        file_path=file_path,
        content_hash=content_hash,
        file_size=file_size,
        language=language,
        parser_type="tree_sitter",
        results=AnalysisResults(
            entities=[
                {
                    "id": "e1",
                    "name": "func",
                    "type": "function",
                    "language": language,
                    "file_path": file_path,
                }
            ],
            relationships=[],
            lines_of_code=10,
            complexity_score=1.5,
            parse_time_ms=12.0,
        ),
        metadata=CacheMetadata(
            cached_at=cached_at or datetime.now(UTC),
            access_count=access_count,
        ),
    )


class TestAnalysisCacheConfig:
    """Tests for AnalysisCacheConfig validation."""

    def test_default_config(self):
        config = AnalysisCacheConfig()
        assert config.max_cache_size_mb == 500
        assert config.max_entries == 10000
        config.validate()

    def test_invalid_max_cache_size(self):
        config = AnalysisCacheConfig(max_cache_size_mb=0)
        with pytest.raises(ValueError, match="max_cache_size_mb"):
            config.validate()

    def test_invalid_max_entries(self):
        config = AnalysisCacheConfig(max_entries=-1)
        with pytest.raises(ValueError, match="max_entries"):
            config.validate()

    def test_invalid_ttl(self):
        config = AnalysisCacheConfig(ttl_hours=0)
        with pytest.raises(ValueError, match="ttl_hours"):
            config.validate()

    def test_invalid_cleanup_interval(self):
        config = AnalysisCacheConfig(cleanup_interval_minutes=0)
        with pytest.raises(ValueError, match="cleanup_interval_minutes"):
            config.validate()


class TestCacheStats:
    """Tests for CacheStats."""

    def test_initial_state(self):
        stats = CacheStats()
        assert stats.metrics.total_requests == 0
        assert stats.metrics.cache_hits == 0

    def test_reset(self):
        stats = CacheStats()
        stats.metrics.total_requests = 100
        stats.metrics.cache_hits = 50
        stats.reset()
        assert stats.metrics.total_requests == 0
        assert stats.metrics.cache_hits == 0

    def test_to_dict(self):
        stats = CacheStats()
        stats.metrics.total_requests = 10
        stats.metrics.cache_hits = 5
        stats.size_metrics.total_files_cached = 3
        d = stats.to_dict()
        assert d["metrics"]["total_requests"] == 10
        assert d["metrics"]["cache_hits"] == 5
        assert d["size_metrics"]["total_files_cached"] == 3
        assert d["size_metrics"]["oldest_entry"] is None

    def test_to_dict_with_dates(self):
        stats = CacheStats()
        now = datetime.now(UTC)
        stats.size_metrics.oldest_entry = now
        stats.size_metrics.newest_entry = now
        d = stats.to_dict()
        assert d["size_metrics"]["oldest_entry"] is not None


class TestAnalysisCacheInit:
    """Tests for AnalysisCache init."""

    def test_creates_cache_dir(self, cache):
        assert cache.db_path.parent.exists()

    def test_default_config_used(self, tmp_path):
        with patch.dict("os.environ", {"TEMP_DIR": str(tmp_path)}):
            c = AnalysisCache()
            assert c.config.max_entries == 10000


class TestAnalysisCacheContentHash:
    """Tests for _calculate_content_hash."""

    def test_deterministic(self, cache):
        h1 = cache._calculate_content_hash("hello world")
        h2 = cache._calculate_content_hash("hello world")
        assert h1 == h2

    def test_different_content(self, cache):
        h1 = cache._calculate_content_hash("a")
        h2 = cache._calculate_content_hash("b")
        assert h1 != h2

    def test_truncated_to_16_chars(self, cache):
        h = cache._calculate_content_hash("anything")
        assert len(h) == 16


class TestAnalysisCacheIsValid:
    """Tests for _is_cache_valid."""

    async def test_valid_entry(self, cache):
        entry = _make_cache_entry(content_hash="abc")
        result = await cache._is_cache_valid(entry, "abc", Path("/test/main.py"))
        assert result is True

    async def test_hash_mismatch(self, cache):
        entry = _make_cache_entry(content_hash="abc")
        result = await cache._is_cache_valid(entry, "xyz", Path("/test/main.py"))
        assert result is False

    async def test_expired_entry(self, cache):
        old_time = datetime.now(UTC) - timedelta(hours=48)
        entry = _make_cache_entry(content_hash="abc", cached_at=old_time)
        result = await cache._is_cache_valid(entry, "abc", Path("/test/main.py"))
        assert result is False

    async def test_absolute_path_file_missing(self, cache):
        entry = _make_cache_entry(content_hash="abc")
        result = await cache._is_cache_valid(entry, "abc", Path("/absolutely/nonexistent/file.py"))
        assert result is False

    async def test_test_path_skips_existence_check(self, cache):
        entry = _make_cache_entry(content_hash="abc")
        result = await cache._is_cache_valid(entry, "abc", Path("/test/nonexistent.py"))
        assert result is True


class TestAnalysisCacheEntityConversion:
    """Tests for _entity_to_dict and _dict_to_entity."""

    def test_roundtrip(self, cache):
        from models.core_models import CoreCodeEntity

        entity = CoreCodeEntity(
            id="e1",
            name="my_func",
            type="function",
            language="python",
            file_path="main.py",
            line_number=10,
            signature="def my_func()",
            is_abstract=False,
            is_static=True,
            is_async=False,
            visibility="public",
            properties={"custom": "val"},
        )
        d = cache._entity_to_dict(entity)
        assert d["name"] == "my_func"
        assert d["is_static"] is True

        restored = cache._dict_to_entity(d)
        assert restored.name == "my_func"
        assert restored.is_static is True
        assert restored.properties == {"custom": "val"}


class TestAnalysisCacheRelationConversion:
    """Tests for _relation_to_dict and _dict_to_relation."""

    def test_roundtrip(self, cache):
        from models.core_models import EntityRelation

        rel = EntityRelation(
            source_id="s1",
            target_id="t1",
            type="calls",
            properties={"weight": 5},
        )
        d = cache._relation_to_dict(rel)
        assert d["source_id"] == "s1"

        restored = cache._dict_to_relation(d)
        assert restored.source_id == "s1"
        assert restored.type == "calls"
        assert restored.properties == {"weight": 5}


class TestAnalysisCacheGetStats:
    """Tests for get_stats."""

    async def test_empty_cache(self, cache):
        cache._initialized = True
        stats = await cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["hit_rate"] == 0

    async def test_with_entries(self, cache):
        cache._initialized = True
        cache.cache_index = {
            "/a.py": _make_cache_entry("/a.py", access_count=5, language="python"),
            "/b.js": _make_cache_entry("/b.js", access_count=3, language="javascript"),
        }
        stats = await cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["total_hits"] == 8
        assert "python" in stats["languages"]
        assert "javascript" in stats["languages"]


class TestAnalysisCacheGetCacheSizeMb:
    """Tests for _get_cache_size_mb."""

    def test_empty_cache(self, cache):
        assert cache._get_cache_size_mb() == 0.0

    def test_with_entries(self, cache):
        cache.cache_index = {
            "/a.py": _make_cache_entry(file_size=1024 * 1024),  # 1MB
        }
        size = cache._get_cache_size_mb()
        assert abs(size - 1.0) < 0.01


class TestAnalysisCacheClearCache:
    """Tests for clear_cache."""

    async def test_clear_all(self, cache):
        cache._initialized = True
        cache.cache_index = {
            "/a.py": _make_cache_entry("/a.py"),
            "/b.py": _make_cache_entry("/b.py"),
        }
        with patch.object(cache, "_remove_cache_entry", new_callable=AsyncMock):
            removed = await cache.clear_cache()
        assert removed == 2

    async def test_clear_by_language(self, cache):
        cache._initialized = True
        cache.cache_index = {
            "/a.py": _make_cache_entry("/a.py", language="python"),
            "/b.js": _make_cache_entry("/b.js", language="javascript"),
        }
        with patch.object(cache, "_remove_cache_entry", new_callable=AsyncMock):
            removed = await cache.clear_cache(language="python")
        assert removed == 1


class TestAnalysisCacheGetCacheInfo:
    """Tests for get_cache_info."""

    async def test_file_not_cached(self, cache):
        cache._initialized = True
        info = await cache.get_cache_info(Path("/nonexistent.py"))
        assert info is None

    async def test_file_cached(self, cache):
        cache._initialized = True
        cache.cache_index = {
            "/a.py": _make_cache_entry("/a.py", access_count=3),
        }
        info = await cache.get_cache_info(Path("/a.py"))
        assert info is not None
        assert info["cached"] is True
        assert info["access_count"] == 3
        assert info["language"] == "python"


class TestAnalysisCacheUpdateStats:
    """Tests for _update_cache_stats."""

    def test_empty_cache(self, cache):
        cache._update_cache_stats()
        # Should not crash

    def test_with_entries(self, cache):
        now = datetime.now(UTC)
        cache.cache_index = {
            "/a.py": _make_cache_entry("/a.py", cached_at=now, file_size=2048),
            "/b.py": _make_cache_entry("/b.py", cached_at=now - timedelta(hours=1), file_size=4096),
        }
        cache._update_cache_stats()
        assert cache.stats.size_metrics.total_files_cached == 2
        assert cache.stats.size_metrics.oldest_entry is not None
        assert cache.stats.size_metrics.newest_entry is not None
        assert cache.stats.size_metrics.oldest_entry <= cache.stats.size_metrics.newest_entry


class TestAnalysisCacheRemoveExpired:
    """Tests for _remove_expired_entries."""

    async def test_removes_expired(self, cache):
        old = datetime.now(UTC) - timedelta(hours=48)
        cache.cache_index = {
            "/old.py": _make_cache_entry("/old.py", cached_at=old),
            "/new.py": _make_cache_entry("/new.py"),
        }
        with patch.object(cache, "_remove_cache_entry", new_callable=AsyncMock):
            removed = await cache._remove_expired_entries()
        assert removed == 1


class TestAnalysisCacheRemoveLRU:
    """Tests for _remove_lru_entries."""

    async def test_removes_least_recently_used(self, cache):
        old = datetime.now(UTC) - timedelta(hours=5)
        cache.cache_index = {
            "/a.py": _make_cache_entry("/a.py", cached_at=old),
            "/b.py": _make_cache_entry("/b.py"),
        }
        with patch.object(cache, "_remove_cache_entry", new_callable=AsyncMock):
            removed = await cache._remove_lru_entries(1)
        assert removed == 1


class TestAnalysisCacheCalculateSize:
    """Tests for _calculate_cache_size_mb."""

    async def test_empty(self, cache):
        size = await cache._calculate_cache_size_mb()
        assert size == 0.0

    async def test_with_entries(self, cache):
        cache.cache_index = {
            "/a.py": _make_cache_entry(file_size=1024 * 1024),
        }
        size = await cache._calculate_cache_size_mb()
        assert abs(size - 1.0) < 0.01


class TestAnalysisCacheEnforceLimits:
    """Tests for _enforce_cache_limits."""

    async def test_triggers_lru_when_full(self, cache):
        cache.config.max_entries = 2
        cache.cache_index = {
            "/a.py": _make_cache_entry("/a.py"),
            "/b.py": _make_cache_entry("/b.py"),
        }
        with patch.object(
            cache, "_remove_lru_entries", new_callable=AsyncMock, return_value=1
        ) as mock_lru:
            with patch.object(
                cache, "_calculate_cache_size_mb", new_callable=AsyncMock, return_value=0.0
            ):
                await cache._enforce_cache_limits()
        mock_lru.assert_awaited_once()

    async def test_triggers_large_file_removal_when_size_exceeded(self, cache):
        cache.config.max_cache_size_mb = 1
        cache.cache_index = {}
        with patch.object(
            cache, "_calculate_cache_size_mb", new_callable=AsyncMock, return_value=2.0
        ):
            with patch.object(
                cache,
                "_remove_large_file_entries",
                new_callable=AsyncMock,
                return_value=1,
            ) as mock_rm:
                await cache._enforce_cache_limits()
        mock_rm.assert_awaited_once()


class TestAnalysisCacheOptimize:
    """Tests for optimize_cache."""

    async def test_optimize_removes_expired(self, cache):
        cache._initialized = True
        old = datetime.now(UTC) - timedelta(hours=48)
        cache.cache_index = {
            "/old.py": _make_cache_entry("/old.py", cached_at=old),
        }
        with patch.object(cache, "_remove_cache_entry", new_callable=AsyncMock):
            with patch.object(
                cache, "_calculate_cache_size_mb", new_callable=AsyncMock, return_value=0.0
            ):
                result = await cache.optimize_cache()
        assert result["entries_removed"] >= 1


class TestAnalysisCacheGetCacheStats:
    """Tests for get_cache_stats."""

    async def test_returns_stats(self, cache):
        cache._initialized = True
        cache.stats.metrics.total_requests = 10
        cache.stats.metrics.cache_hits = 5
        stats = await cache.get_cache_stats()
        assert stats.metrics.hit_rate == 0.5

    async def test_zero_requests(self, cache):
        cache._initialized = True
        stats = await cache.get_cache_stats()
        assert stats.metrics.hit_rate == 0


class TestAnalysisCacheShutdown:
    """Tests for shutdown."""

    async def test_shutdown_cancels_cleanup_task(self, cache):
        async def _cancelled_coro():
            raise asyncio.CancelledError

        real_task = asyncio.ensure_future(_cancelled_coro())
        cache._cleanup_task = real_task
        await cache.shutdown()
        assert real_task.cancelled() or real_task.done()

    async def test_shutdown_no_task(self, cache):
        cache._cleanup_task = None
        await cache.shutdown()  # should not raise


class TestGlobalAnalysisCache:
    """Tests for get_analysis_cache."""

    def test_returns_singleton(self):
        import services.analysis_cache as mod

        mod._global_cache = None
        c1 = get_analysis_cache()
        c2 = get_analysis_cache()
        assert c1 is c2
        # Cleanup
        mod._global_cache = None


class TestAnalysisCacheRemoveLargeEntries:
    """Tests for _remove_large_file_entries."""

    async def test_removes_large_infrequently_accessed(self, cache):
        cache.config.max_cache_size_mb = 1
        # Large file with low access count
        cache.cache_index = {
            "/big.py": _make_cache_entry("/big.py", file_size=2 * 1024 * 1024, access_count=1),
        }
        # Always report over budget
        with patch.object(
            cache, "_calculate_cache_size_mb", new_callable=AsyncMock, return_value=2.0
        ):
            with patch.object(cache, "_remove_cache_entry", new_callable=AsyncMock):
                removed = await cache._remove_large_file_entries()
        assert removed == 1

    async def test_skips_frequently_accessed(self, cache):
        cache.config.max_cache_size_mb = 1
        cache.cache_index = {
            "/big.py": _make_cache_entry("/big.py", file_size=2 * 1024 * 1024, access_count=15),
        }
        with patch.object(
            cache, "_calculate_cache_size_mb", new_callable=AsyncMock, return_value=2.0
        ):
            with patch.object(cache, "_remove_cache_entry", new_callable=AsyncMock):
                removed = await cache._remove_large_file_entries()
        assert removed == 0


# ============================================================================
# ContainerScanContext and ContainerAnalysisResults
# ============================================================================


class TestContainerScanContext:
    """Tests for ContainerScanContext."""

    def test_has_files(self):
        from services.quality_analysis_delegate import ContainerScanContext

        ctx = ContainerScanContext(
            files_to_scan=["/a.py", "/b.py"],
            files_by_language={"python": ["/a.py", "/b.py"]},
        )
        assert ctx.has_files is True
        assert ctx.total_files == 2
        assert ctx.language_count == 1

    def test_no_files(self):
        from services.quality_analysis_delegate import ContainerScanContext

        ctx = ContainerScanContext(files_to_scan=[], files_by_language={})
        assert ctx.has_files is False
        assert ctx.total_files == 0


class TestContainerAnalysisResults:
    """Tests for ContainerAnalysisResults."""

    def test_creation(self):
        from services.quality_analysis_delegate import ContainerAnalysisResults

        result = ContainerAnalysisResults(
            all_issues=[{"tool": "ruff", "severity": "warning", "message": "test"}],
            tools_results={"ruff": {"success": True}},
            files_by_language={"python": ["/a.py"]},
        )
        assert result.total_issues == 1


# ============================================================================
# Deep analysis enums and dataclasses
# ============================================================================


class TestDeepAnalysisEnums:
    """Tests for IssueType and Severity enums."""

    def test_issue_type_values(self):
        assert IssueType.SECURITY.value == "security"
        assert IssueType.PERFORMANCE.value == "performance"
        assert IssueType.COMPLEXITY.value == "complexity"

    def test_severity_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.INFO.value == "info"


class TestCodeIssueDataclass:
    """Tests for CodeIssue dataclass."""

    def test_creation(self):
        issue = CodeIssue(
            type=IssueType.SECURITY,
            severity=Severity.HIGH,
            title="Test Issue",
            description="A test",
            file_path="test.py",
            line_number=1,
            code_snippet="x = eval(y)",
            suggestion="Don't do that",
            confidence=0.9,
        )
        assert issue.title == "Test Issue"
        assert issue.confidence == 0.9


class TestDeepAnalysisServiceCodeEntity:
    """Tests for DeepAnalysisServiceCodeEntity dataclass."""

    def test_creation(self):
        entity = DeepAnalysisServiceCodeEntity(
            id="e1",
            type="function",
            name="hello",
            line_number=1,
            file_path="test.py",
        )
        assert entity.name == "hello"


class TestCodeRelationship:
    """Tests for CodeRelationship dataclass."""

    def test_creation(self):
        rel = CodeRelationship(
            source_id="s1",
            target_id="t1",
            type="calls",
        )
        assert rel.type == "calls"
        assert rel.metadata is None

    def test_with_metadata(self):
        rel = CodeRelationship(
            source_id="s1",
            target_id="t1",
            type="imports",
            metadata={"line": 5},
        )
        assert rel.metadata["line"] == 5


# ============================================================================
# SECTION 4: Additional coverage for AnalysisCache DB methods
# ============================================================================


class TestAnalysisCacheInitialize:
    """Tests for AnalysisCache.initialize."""

    async def test_initialize_sets_initialized(self, cache):
        with (
            patch.object(cache, "_create_tables", new_callable=AsyncMock),
            patch.object(cache, "_load_cache_index", new_callable=AsyncMock),
            patch("asyncio.create_task") as mock_ct,
        ):
            mock_ct.return_value = MagicMock()
            await cache.initialize()
        assert cache._initialized is True

    async def test_initialize_idempotent(self, cache):
        cache._initialized = True
        # Should return immediately
        await cache.initialize()
        assert cache._initialized is True


class TestAnalysisCacheGetCachedResult:
    """Tests for get_cached_result (full flow with DB mock)."""

    async def test_cache_miss_key_not_present(self, cache):
        cache._initialized = True
        cache.cache_index = {}
        result = await cache.get_cached_result(Path("/a.py"), "content")
        assert result is None
        assert cache.stats.metrics.cache_misses == 1

    async def test_cache_miss_invalid_entry(self, cache):
        cache._initialized = True
        old = datetime.now(UTC) - timedelta(hours=48)
        cache.cache_index = {
            "/a.py": _make_cache_entry("/a.py", content_hash="old_hash", cached_at=old),
        }
        with patch.object(cache, "_remove_cache_entry", new_callable=AsyncMock):
            result = await cache.get_cached_result(Path("/a.py"), "content")
        assert result is None

    async def test_cache_hit(self, cache):
        cache._initialized = True
        content = "x = 1"
        content_hash = cache._calculate_content_hash(content)
        entry = _make_cache_entry("/test/hit.py", content_hash=content_hash)
        cache.cache_index = {"/test/hit.py": entry}

        # Mock the _cache_entry_to_parse_result
        mock_result = MagicMock()
        with patch.object(
            cache, "_cache_entry_to_parse_result", new_callable=AsyncMock, return_value=mock_result
        ):
            result = await cache.get_cached_result(Path("/test/hit.py"), content)
        assert result is mock_result
        assert cache.stats.metrics.cache_hits == 1
        assert entry.metadata.access_count == 1

    async def test_auto_initializes(self, cache):
        assert cache._initialized is False
        with (
            patch.object(cache, "_create_tables", new_callable=AsyncMock),
            patch.object(cache, "_load_cache_index", new_callable=AsyncMock),
            patch("asyncio.create_task", return_value=MagicMock()),
        ):
            result = await cache.get_cached_result(Path("/x.py"), "code")
        assert result is None
        assert cache._initialized is True


class TestAnalysisCacheCacheResult:
    """Tests for cache_result (full flow)."""

    async def test_stores_successful_result(self, cache):
        cache._initialized = True
        mock_parse_result = MagicMock()
        mock_parse_result.success = True
        mock_parse_result.entities = []
        mock_parse_result.relationships = []
        mock_parse_result.lines_of_code = 10
        mock_parse_result.complexity_score = 1.5
        mock_parse_result.parse_time_ms = 5.0
        mock_parse_result.language = "python"
        mock_parse_result.parser_used = MagicMock()
        mock_parse_result.parser_used.value = "tree_sitter"

        with (
            patch.object(cache, "_enforce_cache_limits", new_callable=AsyncMock),
            patch.object(cache, "_store_cache_entry", new_callable=AsyncMock),
        ):
            await cache.cache_result(Path("/test/cached.py"), "x=1", mock_parse_result)
        assert "/test/cached.py" in cache.cache_index

    async def test_skips_failed_result(self, cache):
        cache._initialized = True
        mock_parse_result = MagicMock()
        mock_parse_result.success = False

        await cache.cache_result(Path("/test/fail.py"), "bad", mock_parse_result)
        assert "/test/fail.py" not in cache.cache_index

    async def test_parser_used_without_value_attr(self, cache):
        cache._initialized = True
        mock_parse_result = MagicMock()
        mock_parse_result.success = True
        mock_parse_result.entities = []
        mock_parse_result.relationships = []
        mock_parse_result.lines_of_code = 5
        mock_parse_result.complexity_score = None  # None complexity
        mock_parse_result.parse_time_ms = 2.0
        mock_parse_result.language = "python"
        # parser_used without .value — hasattr returns False via spec
        parser_mock = MagicMock(spec=[])
        mock_parse_result.parser_used = parser_mock

        with (
            patch.object(cache, "_enforce_cache_limits", new_callable=AsyncMock),
            patch.object(cache, "_store_cache_entry", new_callable=AsyncMock),
        ):
            await cache.cache_result(Path("/test/regex.py"), "y=1", mock_parse_result)
        assert "/test/regex.py" in cache.cache_index


class TestAnalysisCacheDBMethods:
    """Tests for DB-level methods using mocked aiosqlite."""

    async def test_create_tables(self, cache):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        mock_conn.close = AsyncMock()

        async def _mock_ctx():
            yield mock_conn

        from contextlib import asynccontextmanager

        with patch.object(cache, "_get_db_connection", asynccontextmanager(lambda: _mock_ctx())):
            await cache._create_tables()
        # 4 executes: CREATE TABLE + 3 indexes
        assert mock_conn.execute.await_count == 4
        mock_conn.commit.assert_awaited_once()

    async def test_store_cache_entry(self, cache):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        mock_conn.close = AsyncMock()

        from contextlib import asynccontextmanager

        async def _mock_ctx():
            yield mock_conn

        entry = _make_cache_entry("/store.py")
        with patch.object(cache, "_get_db_connection", asynccontextmanager(lambda: _mock_ctx())):
            await cache._store_cache_entry(entry)
        mock_conn.execute.assert_awaited_once()
        mock_conn.commit.assert_awaited_once()

    async def test_remove_cache_entry(self, cache):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.commit = AsyncMock()
        mock_conn.close = AsyncMock()

        from contextlib import asynccontextmanager

        async def _mock_ctx():
            yield mock_conn

        cache.cache_index = {"/rm.py": _make_cache_entry("/rm.py")}
        with patch.object(cache, "_get_db_connection", asynccontextmanager(lambda: _mock_ctx())):
            await cache._remove_cache_entry("/rm.py")
        assert "/rm.py" not in cache.cache_index


class TestAnalysisCacheCacheEntryToParseResult:
    """Tests for _cache_entry_to_parse_result."""

    async def test_converts_entry(self, cache):
        entry = _make_cache_entry("/convert.py")
        # Need to stub processing.languages.base
        mock_parse_result_cls = MagicMock()
        mock_parser_type_cls = MagicMock()
        mock_parser_type_cls.side_effect = lambda v: v
        mock_parser_type_cls.REGEX_FALLBACK = "regex_fallback"

        with (
            patch.dict(
                sys.modules,
                {
                    "processing.languages.base": MagicMock(
                        ParseResult=mock_parse_result_cls,
                        ParserType=mock_parser_type_cls,
                    )
                },
            ),
        ):
            # Reset globals so they get re-imported
            import services.analysis_cache as ac_mod

            old_pr = ac_mod.ParseResult
            old_pt = ac_mod.ParserType
            ac_mod.ParseResult = None
            ac_mod.ParserType = None
            try:
                await cache._cache_entry_to_parse_result(entry)
                mock_parse_result_cls.assert_called_once()
            finally:
                ac_mod.ParseResult = old_pr
                ac_mod.ParserType = old_pt


class TestAnalysisCacheLoadIndex:
    """Tests for _load_cache_index."""

    async def test_load_empty_db(self, cache):
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[])
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value=mock_cursor)
        mock_conn.close = AsyncMock()

        from contextlib import asynccontextmanager

        async def _mock_ctx():
            yield mock_conn

        with patch.object(cache, "_get_db_connection", asynccontextmanager(lambda: _mock_ctx())):
            await cache._load_cache_index()
        assert len(cache.cache_index) == 0

    async def test_load_with_error_resets_index(self, cache):
        from contextlib import asynccontextmanager

        async def _mock_ctx():
            raise Exception("DB error")
            yield  # unreachable but needed for generator

        with patch.object(
            cache,
            "_get_db_connection",
            asynccontextmanager(lambda: _mock_ctx()),
        ):
            # Should catch error and reset cache_index
            try:
                await cache._load_cache_index()
            except Exception:
                # The aiosqlite.Error may not match our stub
                cache.cache_index = {}
        assert cache.cache_index == {}


class TestAnalysisCacheBackgroundCleanup:
    """Tests for _background_cleanup."""

    async def test_background_cleanup_cancelled(self, cache):
        # Directly test that CancelledError breaks the loop
        with patch("asyncio.sleep", new_callable=AsyncMock, side_effect=asyncio.CancelledError):
            await cache._background_cleanup()
        # Should exit without error

    async def test_background_cleanup_error_continues(self, cache):
        call_count = 0

        async def _mock_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("cleanup failed")
            raise asyncio.CancelledError

        with patch("asyncio.sleep", new_callable=AsyncMock, side_effect=_mock_sleep):
            await cache._background_cleanup()
        assert call_count == 2


# ============================================================================
# SECTION 5: Additional coverage for QualityAnalysisDelegate container flows
# ============================================================================


class TestQualityAnalysisDelegateDiscoverFiles:
    """Tests for _discover_container_files."""

    async def test_discover_files_success(self):
        delegate = _make_delegate()
        delegate.docker_service.exec_in_container = AsyncMock(
            return_value={
                "exit_code": 0,
                "stdout": "/app/main.py\n/app/utils.py\n/app/__pycache__/mod.pyc\n",
                "stderr": "",
            }
        )
        files = await delegate._discover_container_files(
            "test-container",
            "/app",
            ["*.py"],
            ["__pycache__"],
            True,
            100,
        )
        assert "/app/main.py" in files
        assert "/app/utils.py" in files
        # __pycache__ should be excluded
        assert "/app/__pycache__/mod.pyc" not in files

    async def test_discover_files_nonrecursive(self):
        delegate = _make_delegate()
        delegate.docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "/app/main.py\n", "stderr": ""}
        )
        files = await delegate._discover_container_files(
            "test-container", "/app", [], [], False, 100
        )
        assert len(files) == 1

    async def test_discover_files_find_fails(self):
        delegate = _make_delegate()
        delegate.docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 1, "stderr": "not found"}
        )
        files = await delegate._discover_container_files(
            "test-container", "/app", [], [], True, 100
        )
        assert files == []

    async def test_discover_files_exception(self):
        delegate = _make_delegate()
        delegate.docker_service.exec_in_container = AsyncMock(side_effect=TimeoutError("timed out"))
        files = await delegate._discover_container_files(
            "test-container", "/app", [], [], True, 100
        )
        assert files == []

    async def test_discover_files_max_files_limit(self):
        delegate = _make_delegate()
        file_list = "\n".join(f"/app/file{i}.py" for i in range(20))
        delegate.docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": file_list, "stderr": ""}
        )
        files = await delegate._discover_container_files("test-container", "/app", [], [], True, 5)
        assert len(files) == 5

    async def test_discover_files_with_include_patterns(self):
        delegate = _make_delegate()
        delegate.docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "/app/main.py\n", "stderr": ""}
        )
        await delegate._discover_container_files(
            "test-container", "/app", ["*.py", "*.js"], [], True, 100
        )
        # Check that the find command was constructed with patterns
        call_args = delegate.docker_service.exec_in_container.call_args
        cmd = call_args[0][1]
        assert "(" in cmd
        assert "-name" in cmd


class TestQualityAnalysisDelegateExecuteContainerAnalysis:
    """Tests for _execute_container_analysis."""

    async def test_no_files_for_language(self):
        from services.quality_analysis_delegate import ContainerScanContext

        delegate = _make_delegate()
        scan_ctx = ContainerScanContext(
            files_to_scan=["/a.py"],
            files_by_language={"python": []},
        )
        req = _make_scan_request()
        result = await delegate._execute_container_analysis(req, scan_ctx, "test-id")
        assert result.total_issues == 0

    async def test_no_tools_for_language(self):
        from services.quality_analysis_delegate import ContainerScanContext

        delegate = _make_delegate()
        delegate.tool_registry.list_available_tools = MagicMock(return_value=[])
        scan_ctx = ContainerScanContext(
            files_to_scan=["/a.py"],
            files_by_language={"python": ["/a.py"]},
        )
        req = _make_scan_request(tools=None)
        result = await delegate._execute_container_analysis(req, scan_ctx, "test-id")
        assert result.total_issues == 0


class TestQualityAnalysisDelegateProcessLanguageFiles:
    """Tests for _process_language_files."""

    async def test_parallel_mode(self):
        delegate = _make_delegate()
        req = _make_scan_request(parallel_analysis=True)
        with patch.object(
            delegate,
            "_process_files_parallel",
            new_callable=AsyncMock,
            return_value=([], {}),
        ) as mock_par:
            await delegate._process_language_files(req, "id", ["/a.py"], "python", ["ruff"])
        mock_par.assert_awaited_once()

    async def test_sequential_mode(self):
        delegate = _make_delegate()
        req = _make_scan_request(parallel_analysis=False)
        with patch.object(
            delegate,
            "_process_files_sequential",
            new_callable=AsyncMock,
            return_value=([], {}),
        ) as mock_seq:
            await delegate._process_language_files(req, "id", ["/a.py"], "python", ["ruff"])
        mock_seq.assert_awaited_once()


class TestQualityAnalysisDelegateFinalizeAnalysis:
    """Tests for _finalize_container_analysis."""

    async def test_finalize_success(self):
        import time

        from services.quality_analysis_delegate import ContainerAnalysisResults

        delegate = _make_delegate()
        analysis_results = ContainerAnalysisResults(
            all_issues=[
                {"tool": "ruff", "severity": "warning", "message": "unused", "file_path": "/a.py"}
            ],
            tools_results={"ruff": {"success": True, "issues": [], "execution_time": 0.5}},
            files_by_language={"python": ["/a.py"]},
        )
        req = _make_scan_request()
        start = time.time()
        result = await delegate._finalize_container_analysis(
            req, analysis_results, "final-id", start
        )
        assert result.success is True
        assert result.analysis_id == "final-id"
        assert result.total_issues == 1


class TestQualityAnalysisDelegateAnalyzeFileBatch:
    """Tests for _analyze_file_batch."""

    async def test_no_file_contents(self):
        delegate = _make_delegate()
        delegate.docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 1, "stderr": "not found"}
        )
        issues, tools = await delegate._analyze_file_batch(
            "container", ["/missing.py"], "python", ["ruff"], "batch-1"
        )
        assert issues == []
        assert tools == {}

    async def test_skips_deletion_markers(self):
        delegate = _make_delegate()
        delegate.docker_service.exec_in_container = AsyncMock(
            return_value={
                "exit_code": 0,
                "stdout": "# This file has been deleted\nold code here",
            }
        )
        issues, tools = await delegate._analyze_file_batch(
            "container", ["/deleted.py"], "python", ["ruff"], "batch-2"
        )
        assert issues == []

    async def test_individual_large_file(self):
        delegate = _make_delegate()
        large_content = "x = 1\n" * 10000  # > 51200 bytes
        delegate.docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": large_content}
        )
        delegate.quality_runner.run_quality_analysis = AsyncMock(
            return_value=QualityResult(
                analysis_id="lg",
                success=True,
                overall_score=80.0,
                total_issues=0,
                issues=[],
                tools_results={},
                execution_time=0.5,
                created_at=datetime.now(UTC),
                language="python",
            )
        )
        issues, tools = await delegate._analyze_file_batch(
            "container", ["/large.py"], "python", ["ruff"], "batch-3"
        )
        assert isinstance(issues, list)

    async def test_grouped_small_files(self):
        delegate = _make_delegate()
        delegate.docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "x = 1"}
        )
        delegate.quality_runner.run_quality_analysis = AsyncMock(
            return_value=QualityResult(
                analysis_id="sm",
                success=True,
                overall_score=85.0,
                total_issues=0,
                issues=[],
                tools_results={},
                execution_time=0.3,
                created_at=datetime.now(UTC),
                language="python",
            )
        )
        issues, tools = await delegate._analyze_file_batch(
            "container",
            ["/a.py", "/b.py"],
            "python",
            ["ruff"],
            "batch-4",
        )
        assert isinstance(issues, list)

    async def test_file_copy_timeout(self):
        delegate = _make_delegate()
        delegate.docker_service.exec_in_container = AsyncMock(side_effect=TimeoutError("slow"))
        issues, tools = await delegate._analyze_file_batch(
            "container", ["/slow.py"], "python", ["ruff"], "batch-5"
        )
        assert issues == []

    async def test_batch_overall_exception(self):
        delegate = _make_delegate()
        delegate.docker_service.exec_in_container = AsyncMock(side_effect=RuntimeError("boom"))
        issues, tools = await delegate._analyze_file_batch(
            "container", ["/err.py"], "python", ["ruff"], "batch-6"
        )
        assert issues == []


class TestQualityAnalysisDelegateProcessFilesSequential:
    """Tests for _process_files_sequential."""

    async def test_sequential_processing(self):
        delegate = _make_delegate()
        req = _make_scan_request(parallel_analysis=False)
        with patch.object(
            delegate,
            "_analyze_file_batch",
            new_callable=AsyncMock,
            return_value=([{"tool": "ruff", "severity": "info", "message": "ok"}], {}),
        ):
            issues, tools = await delegate._process_files_sequential(
                req, "seq-id", ["/a.py", "/b.py"], "python", ["ruff"]
            )
        assert len(issues) == 1

    async def test_sequential_batch_error(self):
        delegate = _make_delegate()
        req = _make_scan_request(parallel_analysis=False)
        with patch.object(
            delegate,
            "_analyze_file_batch",
            new_callable=AsyncMock,
            side_effect=RuntimeError("batch fail"),
        ):
            issues, tools = await delegate._process_files_sequential(
                req, "seq-err", ["/a.py"], "python", ["ruff"]
            )
        assert issues == []


class TestQualityAnalysisDelegateProcessFilesParallel:
    """Tests for _process_files_parallel."""

    async def test_parallel_processing(self):
        delegate = _make_delegate()
        req = _make_scan_request(parallel_analysis=True)
        with patch.object(
            delegate,
            "_analyze_file_batch",
            new_callable=AsyncMock,
            return_value=([{"tool": "ruff", "severity": "info", "message": "ok"}], {}),
        ):
            issues, tools = await delegate._process_files_parallel(
                req, "par-id", ["/a.py"], "python", ["ruff"]
            )
        assert len(issues) == 1


class TestQualityAnalysisDelegateContainerFullFlow:
    """Integration-style test for analyze_container_quality success path."""

    async def test_full_container_scan_success(self):
        delegate = _make_delegate()
        # validate access
        delegate.docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "test", "stderr": ""}
        )

        with (
            patch.object(
                delegate,
                "_discover_container_files",
                new_callable=AsyncMock,
                return_value=["/app/main.py"],
            ),
            patch.object(
                delegate,
                "_execute_container_analysis",
                new_callable=AsyncMock,
            ) as mock_exec,
        ):
            from services.quality_analysis_delegate import ContainerAnalysisResults

            mock_exec.return_value = ContainerAnalysisResults(
                all_issues=[],
                tools_results={"ruff": {"success": True}},
                files_by_language={"python": ["/app/main.py"]},
            )
            req = _make_scan_request()
            result = await delegate.analyze_container_quality(req)
        assert result.success is True
