"""Unit tests for processing.quality_analysis.quality_runner module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.testing_models import QualityIssue, QualityRequest
from processing.quality_analysis.quality_runner import QualityToolRunner
from processing.quality_analysis.tool_definitions import QualityTool, ToolCategory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_docker_service():
    ds = AsyncMock()
    ds.create_test_container = AsyncMock(return_value="container-id-1")
    ds.start_container = AsyncMock()
    ds.exec_in_container = AsyncMock(return_value={"exit_code": 0, "stdout": "", "stderr": ""})
    ds.create_file_in_container = AsyncMock(return_value=True)
    ds.stop_container = AsyncMock()
    ds.remove_container_by_id = AsyncMock()
    return ds


@pytest.fixture
def runner(mock_docker_service):
    with patch("processing.quality_analysis.quality_runner.get_tool_registry") as mock_reg:
        registry = MagicMock()
        registry.list_available_tools.return_value = ["ruff", "ty"]
        registry.get_tool.return_value = QualityTool(
            name="ruff",
            category=ToolCategory.LINTER,
            command=["ruff", "--output-format=json"],
            output_format="json",
            description="Python linter",
            timeout=30,
        )
        registry.get_installation_commands.return_value = ["uv pip install --system ruff ty"]
        mock_reg.return_value = registry
        r = QualityToolRunner(mock_docker_service)
    return r


@pytest.fixture
def quality_request():
    return QualityRequest(
        container_name="test-container",
        code="def hello(): pass",
        language="python",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolOrchestrationOverLanguageToolMatrix:
    """test_tool_orchestration_over_language_tool_matrix"""

    def test_detect_language_from_request(self, runner):
        from models.testing_models import QualityRequest

        req = QualityRequest(container_name="c", language="javascript")
        lang = runner._detect_language(req)
        assert lang == "javascript"

    def test_detect_language_from_extension(self, runner):
        from models.testing_models import QualityRequest

        req = QualityRequest(container_name="c", code_path="/app/main.ts")
        lang = runner._detect_language(req)
        assert lang == "typescript"

    def test_select_tools_filters_requested(self, runner):
        runner.tool_registry.list_available_tools.return_value = ["ruff", "ty", "ruff-format"]
        selected = runner._select_tools("python", ["ruff", "ruff-format"])
        assert "ruff" in selected
        assert "ruff-format" in selected
        assert "ty" not in selected

    def test_select_tools_returns_all_when_none_requested(self, runner):
        runner.tool_registry.list_available_tools.return_value = ["ruff", "ty"]
        selected = runner._select_tools("python", None)
        assert selected == ["ruff", "ty"]


class TestPartialToolFailure:
    """test_partial_tool_failure"""

    async def test_run_quality_analysis_no_tools_raises(self, runner, quality_request):
        """When no tools are available, the error is returned in QualityResult."""
        runner.tool_registry.list_available_tools.return_value = []
        result = await runner.run_quality_analysis(quality_request)
        assert result.error is not None
        assert result.overall_score == 0.0

    async def test_single_tool_exception_returns_error_result(self, runner, mock_docker_service):
        """A failing single tool raises ValueError."""
        runner.tool_registry.get_tool.return_value = None
        with pytest.raises(ValueError, match="not available"):
            await runner.run_single_tool("nonexistent", "python", "x = 1")


class TestTimeoutHandling:
    """test_timeout_handling"""

    async def test_execute_tools_parallel_timeout(
        self, runner, quality_request, mock_docker_service
    ):
        """Timed-out tools produce error results."""
        tool = QualityTool(
            name="slow",
            category=ToolCategory.LINTER,
            command=["slow-tool"],
            output_format="text",
            description="A slow tool",
            timeout=30,
        )
        runner.tool_registry.get_tool.return_value = tool

        # Make exec_in_container raise TimeoutError
        mock_docker_service.exec_in_container = AsyncMock(side_effect=TimeoutError("timed out"))

        result = await runner._execute_single_tool("cid", tool, "main.py")
        assert result["success"] is False


class TestMergedReportShape:
    """test_merged_report_shape"""

    def test_calculate_quality_score_no_issues(self, runner):
        score = runner._calculate_quality_score([], {})
        assert score == 10.0

    def test_calculate_quality_score_with_issues(self, runner):
        issues = [
            QualityIssue(
                tool="ruff", severity="error", message="bad", file_path="x.py", line_number=1
            ),
            QualityIssue(
                tool="ruff", severity="warning", message="warn", file_path="x.py", line_number=2
            ),
        ]
        score = runner._calculate_quality_score(issues, {})
        assert 0.0 <= score <= 10.0
        assert score < 10.0  # should be penalized

    def test_map_ruff_severity(self, runner):
        assert runner._map_ruff_severity("F841") == "error"
        assert runner._map_ruff_severity("W291") == "warning"
        assert runner._map_ruff_severity("unformatted") == "info"

    def test_get_file_extension(self, runner):
        assert runner._get_file_extension("python") == ".py"
        assert runner._get_file_extension("javascript") == ".js"
        assert runner._get_file_extension("ruby") == ".py"  # default

    def test_get_available_tools(self, runner):
        runner.tool_registry.list_available_tools.return_value = ["a", "b"]
        tools = runner.get_available_tools("python")
        assert tools == ["a", "b"]

    def test_calculate_quality_score_exact_values(self, runner):
        """Verify exact scoring calculation: 10 - (weighted_sum * 0.2)."""
        issues = [
            QualityIssue(
                tool="ruff", severity="critical", message="crit", file_path="x.py", line_number=1
            ),
        ]
        score = runner._calculate_quality_score(issues, {})
        # critical weight = 3.0, score = 10 - 3.0*0.2 = 9.4
        assert score == 9.4

    def test_calculate_quality_score_info_penalty(self, runner):
        """Info severity has weight 0.3."""
        issues = [
            QualityIssue(
                tool="ruff", severity="info", message="info", file_path="x.py", line_number=1
            ),
        ]
        score = runner._calculate_quality_score(issues, {})
        # info weight = 0.3, score = 10 - 0.3*0.2 = 9.94
        assert score == 9.94

    def test_calculate_quality_score_floors_at_zero(self, runner):
        """Score should never go below 0.0."""
        issues = [
            QualityIssue(
                tool="ruff",
                severity="critical",
                message=f"issue-{i}",
                file_path="x.py",
                line_number=i,
            )
            for i in range(100)
        ]
        score = runner._calculate_quality_score(issues, {})
        assert score == 0.0

    def test_map_ruff_severity_pylint_family(self, runner):
        assert runner._map_ruff_severity("PLE1142") == "error"
        assert runner._map_ruff_severity("PLW0120") == "warning"
        assert runner._map_ruff_severity("PLC0414") == "info"
        assert runner._map_ruff_severity("PLR1714") == "info"

    def test_get_file_extension_typescript(self, runner):
        assert runner._get_file_extension("typescript") == ".ts"

    def test_get_target_filename_with_code_path(self, runner):
        req = QualityRequest(container_name="c", code_path="/some/dir/myfile.py")
        assert runner._get_target_filename(req, "python") == "myfile.py"

    def test_get_target_filename_without_code_path(self, runner):
        req = QualityRequest(container_name="c", code="x=1", language="javascript")
        assert runner._get_target_filename(req, "javascript") == "main.js"

    def test_get_tool_info(self, runner):
        runner.tool_registry.get_tool_info.return_value = {"name": "ruff", "version": "3.0"}
        info = runner.get_tool_info("ruff", "python")
        assert info["name"] == "ruff"
        assert info["version"] == "3.0"

    def test_initialize_container_images(self, runner):
        images = runner._initialize_container_images()
        assert images["python"] == "code_graph_helper-api:latest"
        assert images["javascript"] == "node:18-slim"
        assert images["typescript"] == "node:18-slim"


class TestQualityRunnerContainerOps:
    """Tests for container operations and tool execution paths."""

    async def test_cleanup_container_success(self, runner, mock_docker_service):
        await runner._cleanup_container("cid-1")
        mock_docker_service.stop_container.assert_awaited_once_with("cid-1")
        mock_docker_service.remove_container_by_id.assert_awaited_once_with("cid-1")

    async def test_cleanup_container_failure_does_not_raise(self, runner, mock_docker_service):
        mock_docker_service.stop_container = AsyncMock(side_effect=OSError("stop failed"))
        await runner._cleanup_container("cid-fail")  # Should not raise

    async def test_ensure_tools_installed_exception_does_not_raise(
        self, runner, mock_docker_service
    ):
        mock_docker_service.exec_in_container = AsyncMock(
            side_effect=TimeoutError("install timed out")
        )
        runner.tool_registry.get_installation_commands.return_value = [
            "uv pip install --system ruff"
        ]
        await runner._ensure_tools_installed("cid", "python", ["ruff"])  # Should not raise

    async def test_ensure_tools_installed_nonzero_exit_continues(self, runner, mock_docker_service):
        """If first install returns non-zero, the runner logs and continues."""
        mock_docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 1, "stderr": "failed to install"}
        )
        runner.tool_registry.get_installation_commands.return_value = [
            "uv pip install --system ruff",
            "uv pip install --system ty",
        ]
        await runner._ensure_tools_installed("cid", "python", ["ruff"])
        # Both commands should have been tried (first failed, second attempted)
        assert mock_docker_service.exec_in_container.await_count == 2

    async def test_execute_tools_parallel_with_tool_failure(
        self, runner, quality_request, mock_docker_service
    ):
        """Tool that raises an exception in parallel execution."""
        tool = QualityTool(
            name="failing",
            category=ToolCategory.LINTER,
            command=["failing-tool"],
            output_format="text",
            description="Failing tool",
            timeout=30,
        )
        runner.tool_registry.get_tool.return_value = tool
        mock_docker_service.exec_in_container = AsyncMock(side_effect=RuntimeError("exec failed"))

        results = await runner._execute_tools_parallel(
            "cid", ["failing"], "python", quality_request
        )
        assert "failing" in results
        assert results["failing"]["success"] is False

    async def test_parse_tool_output_exception_creates_error_issue(self, runner):
        """When parse output raises, an error issue is created."""
        tool = QualityTool(
            name="badtool",
            category=ToolCategory.LINTER,
            command=["badtool"],
            output_format="json",
            description="Bad tool",
            timeout=30,
        )
        # Force json.loads to fail by passing invalid data via _parse_json_output
        # Then call the outer _parse_tool_output which catches the exception
        with patch.object(runner, "_parse_json_output", side_effect=TypeError("nope")):
            issues = await runner._parse_tool_output(tool, "data", "", "test.py")
        assert len(issues) == 1
        assert issues[0].rule_id == "parse_error"
        assert issues[0].severity == "error"
        assert "Failed to parse" in issues[0].message

    async def test_parse_diff_output_nonempty(self, runner):
        tool = QualityTool(
            name="diffcheck",
            category=ToolCategory.FORMATTER,
            command=["diffcheck"],
            output_format="dif",
            description="Diff checker",
            timeout=30,
        )
        issues = await runner._parse_diff_output(tool, "--- a/f.py\n+++ b/f.py\n", "f.py")
        assert len(issues) == 1
        assert issues[0].rule_id == "format_dif"
        assert issues[0].severity == "info"

    async def test_parse_diff_output_empty(self, runner):
        tool = QualityTool(
            name="diffcheck",
            category=ToolCategory.FORMATTER,
            command=["diffcheck"],
            output_format="dif",
            description="Diff checker",
            timeout=30,
        )
        issues = await runner._parse_diff_output(tool, "   ", "f.py")
        assert issues == []

    async def test_run_single_tool_success(self, runner, mock_docker_service):
        """Full single tool execution flow."""
        tool = QualityTool(
            name="ruff",
            category=ToolCategory.LINTER,
            command=["ruff", "--output-format=json"],
            output_format="json",
            description="Linter",
            timeout=30,
        )
        runner.tool_registry.get_tool.return_value = tool
        mock_docker_service.exec_in_container = AsyncMock(
            return_value={"exit_code": 0, "stdout": "[]", "stderr": ""}
        )
        result = await runner.run_single_tool("ruff", "python", "x = 1")
        assert result["success"] is True

    async def test_run_single_tool_file_creation_failure(self, runner, mock_docker_service):
        """When file creation fails, should raise RuntimeError."""
        tool = QualityTool(
            name="ruff",
            category=ToolCategory.LINTER,
            command=["ruff"],
            output_format="json",
            description="Linter",
            timeout=30,
        )
        runner.tool_registry.get_tool.return_value = tool
        mock_docker_service.create_file_in_container = AsyncMock(return_value=False)
        with pytest.raises(RuntimeError, match="Failed to create code file"):
            await runner.run_single_tool("ruff", "python", "x = 1")

    async def test_detect_language_from_py_extension(self, runner):
        req = QualityRequest(container_name="c", code_path="/app/main.py")
        assert runner._detect_language(req) == "python"

    async def test_detect_language_from_jsx_extension(self, runner):
        req = QualityRequest(container_name="c", code_path="/app/comp.jsx")
        assert runner._detect_language(req) == "javascript"

    async def test_detect_language_from_tsx_extension(self, runner):
        req = QualityRequest(container_name="c", code_path="/app/comp.tsx")
        assert runner._detect_language(req) == "typescript"

    async def test_detect_language_unknown_extension_defaults_python(self, runner):
        req = QualityRequest(container_name="c", code_path="/app/script.rb")
        assert runner._detect_language(req) == "python"

    async def test_detect_language_no_language_no_path(self, runner):
        req = QualityRequest(container_name="c")
        assert runner._detect_language(req) == "python"


class TestCreateQualityToolRunnerFactory:
    """Test the factory function."""

    def test_create_quality_tool_runner(self):
        from processing.quality_analysis.quality_runner import create_quality_tool_runner

        mock_docker = MagicMock()
        runner = create_quality_tool_runner(mock_docker)
        assert isinstance(runner, QualityToolRunner)
        assert runner.docker_service is mock_docker
