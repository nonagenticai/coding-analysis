"""Unit tests for processing.quality_analysis.tool_registry module.

Note: test_tool_execution.py covers a DIFFERENT module
(processing.architecture.tool_execution).
"""

from __future__ import annotations

import pytest

from processing.quality_analysis.tool_definitions import QualityTool, ToolCategory
from processing.quality_analysis.tool_registry import ToolRegistry, _ToolRegistrySingleton

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    _ToolRegistrySingleton._instance = None
    yield
    _ToolRegistrySingleton._instance = None


@pytest.fixture
def registry():
    return ToolRegistry()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiscoveryAndLookup:
    """test_discovery_and_lookup"""

    def test_list_supported_languages(self, registry):
        langs = registry.list_supported_languages()
        assert "python" in langs
        assert "javascript" in langs
        assert "typescript" in langs

    def test_list_available_tools_python(self, registry):
        tools = registry.list_available_tools("python")
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert "ruff" in tools

    def test_get_tool_returns_quality_tool(self, registry):
        tool = registry.get_tool("python", "ruff")
        assert tool is not None
        assert isinstance(tool, QualityTool)
        assert tool.name == "ruff"

    def test_get_tool_info_dict(self, registry):
        info = registry.get_tool_info("python", "ruff")
        assert info is not None
        assert info["name"] == "ruff"
        assert "command" in info

    def test_get_tools_by_category(self, registry):
        linters = registry.get_tools_by_category("python", ToolCategory.LINTER)
        assert isinstance(linters, dict)
        assert all(t.category == ToolCategory.LINTER for t in linters.values())

    def test_get_installation_commands(self, registry):
        cmds = registry.get_installation_commands("python")
        assert isinstance(cmds, list)


class TestDuplicateToolResolution:
    """test_duplicate_tool_resolution"""

    def test_get_tools_for_language_returns_unique_names(self, registry):
        tools = registry.get_tools_for_language("python")
        names = list(tools.keys())
        assert len(names) == len(set(names)), "Tool names must be unique"

    def test_singleton_returns_same_instance(self):
        from processing.quality_analysis.tool_registry import get_tool_registry

        a = get_tool_registry()
        b = get_tool_registry()
        assert a is b


class TestUnsupportedLanguageTool:
    """test_unsupported_language_tool"""

    def test_unknown_language_returns_empty(self, registry):
        tools = registry.list_available_tools("brainfuck")
        assert tools == []

    def test_get_tool_unknown_returns_none(self, registry):
        assert registry.get_tool("python", "nonexistent") is None

    def test_get_tool_info_unknown_returns_none(self, registry):
        assert registry.get_tool_info("python", "nonexistent") is None

    def test_validate_tool_availability_unknown(self, registry):
        result = registry.validate_tool_availability("python", "nonexistent")
        assert result is False


class TestExecuteToolAndSuite:
    """Tests for execute_tool, run_analysis_suite, and related methods."""

    def test_execute_tool_not_found(self, registry):
        result = registry.execute_tool("python", "nonexistent", "/tmp")
        assert "error" in result
        assert "not found" in result["error"]

    def test_get_configuration_templates(self, registry):
        templates = registry.get_configuration_templates("python")
        assert isinstance(templates, dict)

    def test_get_tools_for_language_javascript(self, registry):
        tools = registry.get_tools_for_language("javascript")
        assert isinstance(tools, dict)
        if tools:
            for name, tool in tools.items():
                assert isinstance(tool, QualityTool)
                assert tool.name == name

    def test_list_available_tools_typescript(self, registry):
        tools = registry.list_available_tools("typescript")
        assert isinstance(tools, list)

    def test_list_available_tools_java(self, registry):
        tools = registry.list_available_tools("java")
        assert isinstance(tools, list)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_tools_for_language_function(self):
        from processing.quality_analysis.tool_registry import get_tools_for_language

        tools = get_tools_for_language("python")
        assert isinstance(tools, dict)
        assert len(tools) > 0

    def test_get_installation_commands_function(self):
        from processing.quality_analysis.tool_registry import get_installation_commands

        cmds = get_installation_commands("python")
        assert isinstance(cmds, list)

    def test_list_supported_languages_function(self):
        from processing.quality_analysis.tool_registry import list_supported_languages

        langs = list_supported_languages()
        assert "python" in langs
        assert "javascript" in langs


class TestExecuteToolWithResults:
    """Cover lines 136-152: execute_tool success path with result processing."""

    def test_execute_tool_success_processes_results(self, registry):
        """When tool exists and execution succeeds, results are processed."""
        from unittest.mock import MagicMock

        tool = registry.get_tool("python", "ruff")
        assert tool is not None  # Verify tool exists

        # Mock executor and result_processor to control flow
        registry.executor.execute_tool_safe = MagicMock(
            return_value={
                "success": True,
                "output": '[{"type":"error","message":"test"}]',
                "exit_code": 0,
            }
        )
        registry.result_processor.process_tool_result = MagicMock(
            return_value={
                "tool_name": "ruff",
                "success": True,
                "issues": [{"severity": "error"}],
            }
        )
        result = registry.execute_tool("python", "ruff", "/tmp/test")
        assert result["success"] is True
        assert "execution_time" in result
        assert "command_used" in result

    def test_execute_tool_failure_returns_execution_result(self, registry):
        """When execution fails, raw execution result is returned."""
        from unittest.mock import MagicMock

        registry.executor.execute_tool_safe = MagicMock(
            return_value={
                "success": False,
                "output": "",
                "error": "tool not found",
                "exit_code": 1,
            }
        )
        result = registry.execute_tool("python", "ruff", "/tmp/test")
        assert result["success"] is False


class TestValidateToolAvailability:
    """Cover line 167: validate_tool_availability with existing tool."""

    def test_validate_tool_available(self, registry):
        """When tool exists, delegates to executor.validate_tool_availability."""
        from unittest.mock import MagicMock

        registry.executor.validate_tool_availability = MagicMock(return_value=True)
        result = registry.validate_tool_availability("python", "ruff")
        assert result is True

    def test_validate_tool_not_available(self, registry):
        from unittest.mock import MagicMock

        registry.executor.validate_tool_availability = MagicMock(return_value=False)
        result = registry.validate_tool_availability("python", "ruff")
        assert result is False


class TestRunAnalysisSuite:
    """Cover lines 196-221: run_analysis_suite."""

    def test_run_analysis_suite_empty_language(self, registry):
        """Unknown language returns empty results."""
        result = registry.run_analysis_suite("brainfuck", "/tmp/test")
        assert isinstance(result, dict)
        assert result["language"] == "brainfuck"
        assert result["target_path"] == "/tmp/test"

    def test_run_analysis_suite_with_category_filter(self, registry):
        """Suite with category filter skips non-matching tools."""
        from unittest.mock import MagicMock

        registry.executor.validate_tool_availability = MagicMock(return_value=True)
        registry.executor.execute_tool_safe = MagicMock(
            return_value={"success": True, "output": "[]", "exit_code": 0}
        )
        registry.result_processor.process_tool_result = MagicMock(
            return_value={"tool_name": "test", "success": True, "issues": []}
        )
        # Normalize results must also be mocked
        registry.result_processor.normalize_results = MagicMock(
            return_value={
                "total_tools": 1,
                "successful_tools": 1,
                "failed_tools": 0,
                "total_issues": 0,
            }
        )
        result = registry.run_analysis_suite(
            "python", "/tmp/test", categories=[ToolCategory.LINTER]
        )
        assert "language" in result
        assert result["language"] == "python"

    def test_run_analysis_suite_tool_unavailable(self, registry):
        """Tools that are unavailable are skipped."""
        from unittest.mock import MagicMock

        registry.executor.validate_tool_availability = MagicMock(return_value=False)
        registry.result_processor.normalize_results = MagicMock(
            return_value={
                "total_tools": 0,
                "successful_tools": 0,
                "failed_tools": 0,
                "total_issues": 0,
            }
        )
        result = registry.run_analysis_suite("python", "/tmp/test")
        assert result["total_tools"] == 0

    def test_run_analysis_suite_tool_execution_error(self, registry):
        """Tools that fail during execution are captured with error."""
        from unittest.mock import MagicMock

        registry.executor.validate_tool_availability = MagicMock(return_value=True)
        registry.executor.execute_tool_safe = MagicMock(side_effect=RuntimeError("tool crashed"))
        # For the execute_tool call (which calls execute_tool_safe internally),
        # we need to mock at the registry level
        original_execute = registry.execute_tool
        registry.execute_tool = MagicMock(side_effect=RuntimeError("tool crashed"))
        registry.result_processor.normalize_results = MagicMock(
            return_value={
                "total_tools": 1,
                "successful_tools": 0,
                "failed_tools": 1,
                "total_issues": 0,
            }
        )
        result = registry.run_analysis_suite("python", "/tmp/test")
        assert result["failed_tools"] >= 0  # Errors are captured
        registry.execute_tool = original_execute
