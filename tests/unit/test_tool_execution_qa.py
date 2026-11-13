"""Tests for processing.quality_analysis.tool_execution module - covers ToolExecutor and ToolExecutionError."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from processing.quality_analysis.tool_definitions import QualityTool, ToolCategory
from processing.quality_analysis.tool_execution import ToolExecutionError, ToolExecutor


def _make_tool(
    name: str = "ruff",
    command: list[str] | None = None,
    output_format: str = "json",
    timeout: int | None = None,
) -> QualityTool:
    if command is None:
        command = ["ruff", "check", "."]
    kwargs = {}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return QualityTool(
        name=name,
        category=ToolCategory.LINTER,
        command=command,
        output_format=output_format,
        description="test tool",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# ToolExecutionError
# ---------------------------------------------------------------------------


class TestToolExecutionError:
    def test_attributes_stored(self):
        err = ToolExecutionError("ty", 2, "type error found")
        assert err.tool_name == "ty"
        assert err.exit_code == 2
        assert err.stderr == "type error found"
        assert "ty" in str(err)
        assert "2" in str(err)

    def test_is_exception(self):
        err = ToolExecutionError("ruff", 1, "failed")
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# ToolExecutor.execute_tool
# ---------------------------------------------------------------------------


class TestToolExecutorExecuteTool:
    def test_success(self):
        executor = ToolExecutor()
        tool = _make_tool()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"ok": true}'
        mock_result.stderr = ""

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run", return_value=mock_result
        ) as mock_run:
            success, stdout, stderr = executor.execute_tool(tool, "/tmp/code")
            assert success is True
            assert stdout == '{"ok": true}'
            assert stderr == ""
            mock_run.assert_called_once()

    def test_nonzero_return_code(self):
        executor = ToolExecutor()
        tool = _make_tool()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "issues found"
        mock_result.stderr = "warning"

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run", return_value=mock_result
        ):
            success, stdout, stderr = executor.execute_tool(tool, "/tmp/code")
            assert success is False
            assert stdout == "issues found"
            assert stderr == "warning"

    def test_timeout_raises_tool_execution_error(self):
        executor = ToolExecutor()
        tool = _make_tool(timeout=5)

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["ruff"], timeout=5),
        ):
            with pytest.raises(ToolExecutionError) as exc_info:
                executor.execute_tool(tool, "/tmp/code")
            assert exc_info.value.exit_code == -1
            assert "timed out" in exc_info.value.stderr

    def test_file_not_found_raises_tool_execution_error(self):
        executor = ToolExecutor()
        tool = _make_tool(command=["nonexistent_tool"])

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run",
            side_effect=FileNotFoundError("not found"),
        ):
            with pytest.raises(ToolExecutionError) as exc_info:
                executor.execute_tool(tool, "/tmp/code")
            assert exc_info.value.exit_code == -2
            assert "not found" in exc_info.value.stderr.lower()

    def test_generic_exception_raises_tool_execution_error(self):
        executor = ToolExecutor()
        tool = _make_tool()

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run",
            side_effect=RuntimeError("something broke"),
        ):
            with pytest.raises(ToolExecutionError) as exc_info:
                executor.execute_tool(tool, "/tmp/code")
            assert exc_info.value.exit_code == -3

    def test_uses_tool_timeout_over_default(self):
        executor = ToolExecutor()
        tool = _make_tool(timeout=42)
        mock_result = MagicMock(returncode=0, stdout="", stderr="")

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run", return_value=mock_result
        ) as mock_run:
            executor.execute_tool(tool, "/tmp/code")
            assert mock_run.call_args[1]["timeout"] == 42

    def test_uses_explicit_timeout(self):
        executor = ToolExecutor()
        tool = _make_tool(timeout=42)
        mock_result = MagicMock(returncode=0, stdout="", stderr="")

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run", return_value=mock_result
        ) as mock_run:
            executor.execute_tool(tool, "/tmp/code", timeout=99)
            assert mock_run.call_args[1]["timeout"] == 99

    def test_uses_tool_default_timeout(self):
        executor = ToolExecutor()
        tool = _make_tool()  # timeout defaults to 30 in QualityTool
        mock_result = MagicMock(returncode=0, stdout="", stderr="")

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run", return_value=mock_result
        ) as mock_run:
            executor.execute_tool(tool, "/tmp/code")
            assert mock_run.call_args[1]["timeout"] == 30


# ---------------------------------------------------------------------------
# ToolExecutor.execute_tool_safe
# ---------------------------------------------------------------------------


class TestToolExecutorExecuteToolSafe:
    def test_success_returns_structured_result(self):
        executor = ToolExecutor()
        tool = _make_tool()
        mock_result = MagicMock(returncode=0, stdout="output", stderr="")

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run", return_value=mock_result
        ):
            result = executor.execute_tool_safe(tool, "/tmp/code")
            assert result["tool_name"] == "ruff"
            assert result["success"] is True
            assert result["output"] == "output"
            assert result["exit_code"] == 0

    def test_failure_returns_structured_result(self):
        executor = ToolExecutor()
        tool = _make_tool()

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["ruff"], timeout=5),
        ):
            result = executor.execute_tool_safe(tool, "/tmp/code")
            assert result["success"] is False
            assert result["exit_code"] == -1
            assert result["output"] == ""


# ---------------------------------------------------------------------------
# ToolExecutor._prepare_command and _tool_needs_target_path
# ---------------------------------------------------------------------------


class TestToolExecutorPrepareCommand:
    def test_appends_target_path_for_unknown_tool(self):
        executor = ToolExecutor()
        tool = _make_tool(command=["custom_tool", "--check"])
        command = executor._prepare_command(tool, "/my/project")
        assert command == ["custom_tool", "--check", "/my/project"]

    def test_does_not_append_for_black(self):
        executor = ToolExecutor()
        tool = _make_tool(command=["ruff-format", "--check", "."])
        command = executor._prepare_command(tool, "/my/project")
        # ruff-format is in no_path_tools AND "." already in command
        assert "/my/project" not in command

    def test_does_not_append_if_path_already_in_command(self):
        executor = ToolExecutor()
        tool = _make_tool(command=["custom_tool", "/my/project"])
        command = executor._prepare_command(tool, "/my/project")
        assert command.count("/my/project") == 1

    def test_does_not_append_for_pylint(self):
        executor = ToolExecutor()
        tool = _make_tool(command=["ruff", "src/"])
        assert executor._tool_needs_target_path(tool, "/some/path") is False

    def test_does_not_append_for_eslint(self):
        executor = ToolExecutor()
        tool = _make_tool(command=["eslint", "--fix"])
        assert executor._tool_needs_target_path(tool, "/some/path") is False

    def test_does_not_append_for_cargo(self):
        executor = ToolExecutor()
        tool = _make_tool(command=["cargo", "clippy"])
        assert executor._tool_needs_target_path(tool, "/some/path") is False


# ---------------------------------------------------------------------------
# ToolExecutor.validate_tool_availability
# ---------------------------------------------------------------------------


class TestToolExecutorValidateAvailability:
    def test_empty_command_returns_false(self):
        executor = ToolExecutor()
        tool = _make_tool(command=[])
        assert executor.validate_tool_availability(tool) is False

    def test_available_tool_returns_true(self):
        executor = ToolExecutor()
        tool = _make_tool(command=["python3"])

        with patch("processing.quality_analysis.tool_execution.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert executor.validate_tool_availability(tool) is True

    def test_unavailable_tool_returns_false(self):
        executor = ToolExecutor()
        tool = _make_tool(command=["nonexistent_xyz_tool"])

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert executor.validate_tool_availability(tool) is False

    def test_all_args_timeout_returns_false(self):
        executor = ToolExecutor()
        tool = _make_tool(command=["slow_tool"])

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["slow_tool"], timeout=10),
        ):
            assert executor.validate_tool_availability(tool) is False

    def test_generic_exception_returns_false(self):
        executor = ToolExecutor()
        tool = _make_tool(command=["broken"])

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run",
            side_effect=RuntimeError("boom"),
        ):
            assert executor.validate_tool_availability(tool) is False


# ---------------------------------------------------------------------------
# ToolExecutor.get_tool_version
# ---------------------------------------------------------------------------


class TestToolExecutorGetToolVersion:
    def test_empty_command_returns_none(self):
        executor = ToolExecutor()
        tool = _make_tool()
        tool.command = []
        assert executor.get_tool_version(tool) is None

    def test_success_returns_version(self):
        executor = ToolExecutor()
        tool = _make_tool(command=["ruff"])
        mock_result = MagicMock(returncode=0, stdout="ruff 0.3.0\n")

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run", return_value=mock_result
        ):
            version = executor.get_tool_version(tool)
            assert version == "ruff 0.3.0"

    def test_failure_returns_none(self):
        executor = ToolExecutor()
        tool = _make_tool(command=["ruff"])
        mock_result = MagicMock(returncode=1, stdout="")

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run", return_value=mock_result
        ):
            assert executor.get_tool_version(tool) is None

    def test_exception_returns_none(self):
        executor = ToolExecutor()
        tool = _make_tool(command=["ruff"])

        with patch(
            "processing.quality_analysis.tool_execution.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            assert executor.get_tool_version(tool) is None
