"""Unit tests for processing.architecture.tool_execution module."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Pre-seed ComplexityAnalysis as a module-level attribute so unittest.mock.patch
# can find and replace it when tests patch
# "processing.architecture.tool_execution.ComplexityAnalysis".
import processing.architecture.tool_execution as _te_mod
from processing.architecture.tool_execution import ToolExecution

if not hasattr(_te_mod, "ComplexityAnalysis"):
    _te_mod.ComplexityAnalysis = MagicMock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def executor():
    """Create a fresh ToolExecution instance."""
    return ToolExecution(container_name="test-worker")


@pytest.fixture
def mock_client():
    """Mock the coding-containers client returned by get_coding_containers_client."""
    client = AsyncMock()
    client.execute = AsyncMock(
        return_value={
            "exit_code": 0,
            "stdout": "some output",
            "stderr": "",
            "duration": 0.5,
        }
    )
    return client


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestToolExecutionInit:
    def test_default_container_name(self):
        te = ToolExecution()
        assert te._container_name == ""

    def test_custom_container_name(self, executor):
        assert executor._container_name == "test-worker"


# ---------------------------------------------------------------------------
# _execute_tool_in_container
# ---------------------------------------------------------------------------


class TestExecuteToolInContainer:
    async def test_success_returns_empty_issues_and_result(self, executor, mock_client):
        """Successful container execution returns empty issues list and metadata."""
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._execute_tool_in_container(
                tool_name="ruff",
                command=["ruff", "check", "."],
                repo_path=Path("/repo"),
                timeout=30,
            )

        assert isinstance(issues, list)
        assert len(issues) == 0
        assert result["success"] is True
        assert result["exit_code"] == 0
        assert result["stdout"] == "some output"
        assert result["stderr"] == ""
        assert result["duration"] == 0.5

        mock_client.execute.assert_awaited_once_with(
            container_name="test-worker",
            command=["ruff", "check", "."],
            workdir="/repo",
            timeout=30,
            # The container is tenant-scoped: the exec presents the owning
            # tenant, and an absent one stays absent rather than defaulted.
            tenant_id=None,
        )

    async def test_negative_exit_code_returns_error(self, executor, mock_client):
        """When the container returns a negative exit_code, result should carry the error."""
        mock_client.execute.return_value = {
            "exit_code": -1,
            "stdout": "",
            "stderr": "Container crashed",
            "duration": 0.0,
        }
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._execute_tool_in_container(
                "mytool", ["mytool"], Path("/repo")
            )

        assert issues == []
        assert result["success"] is False
        assert "error" in result
        assert result["error"] == "Container crashed"

    async def test_nonzero_positive_exit_code_still_success(self, executor, mock_client):
        """Tools like linters return exit_code=1 when they find issues -- this is still 'success'."""
        mock_client.execute.return_value = {
            "exit_code": 1,
            "stdout": "found issues",
            "stderr": "",
            "duration": 1.0,
        }
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._execute_tool_in_container(
                "ruff", ["ruff", "check", "."], Path("/repo")
            )

        assert result["success"] is True
        assert result["exit_code"] == 1

    async def test_exception_from_client_propagates(self, executor, mock_client):
        """If the client raises an exception, it should propagate up."""
        mock_client.execute.side_effect = ConnectionError("Connection refused")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            with pytest.raises(ConnectionError, match="Connection refused"):
                await executor._execute_tool_in_container(
                    "ruff", ["ruff", "check", "."], Path("/repo")
                )

    async def test_default_timeout(self, executor, mock_client):
        """Default timeout should be 60 seconds."""
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            await executor._execute_tool_in_container("tool", ["tool"], Path("/repo"))

        mock_client.execute.assert_awaited_once_with(
            container_name="test-worker",
            command=["tool"],
            workdir="/repo",
            timeout=60,
            tenant_id=None,
        )


# ---------------------------------------------------------------------------
# _parse_line_issues
# ---------------------------------------------------------------------------


class TestParseLineIssues:
    def test_standard_file_line_message_format(self):
        """Parse typical 'file:line: message' output."""
        stdout = "src/main.py:10: unused import os\nsrc/util.py:25: variable unused"
        issues = ToolExecution._parse_line_issues(
            stdout, tool="ruff", category="style", severity="medium"
        )
        assert len(issues) == 2

        assert issues[0].file_path == "src/main.py"
        assert issues[0].line_number == 10
        assert issues[0].message == "unused import os"
        assert issues[0].tool == "ruff"
        assert issues[0].category == "style"
        assert issues[0].severity == "medium"
        assert issues[0].id == "ruff-line-0"
        assert issues[0].rule_id == "ruff-issue"

        assert issues[1].file_path == "src/util.py"
        assert issues[1].line_number == 25
        assert issues[1].id == "ruff-line-1"

    def test_file_line_col_message_format(self):
        """Parse 'file:line:col: message' output."""
        stdout = "file.py:10:5: E123 some error"
        issues = ToolExecution._parse_line_issues(
            stdout, tool="ruff", category="lint", severity="high"
        )
        assert len(issues) == 1
        assert issues[0].file_path == "file.py"
        assert issues[0].line_number == 10
        # Column is in parts[2], message is parts[-1] (parts[3])
        assert "E123 some error" in issues[0].message

    def test_empty_output_returns_no_issues(self):
        issues = ToolExecution._parse_line_issues("", tool="t", category="c")
        assert issues == []

    def test_blank_lines_are_skipped(self):
        stdout = "\n  \nfile.py:1: msg\n\n"
        issues = ToolExecution._parse_line_issues(stdout, tool="t", category="c")
        assert len(issues) == 1

    def test_line_without_colon_is_skipped(self):
        stdout = "no colon here\nfile.py:5: valid"
        issues = ToolExecution._parse_line_issues(stdout, tool="t", category="c")
        assert len(issues) == 1
        assert issues[0].file_path == "file.py"

    def test_non_numeric_line_number_defaults_to_1(self):
        stdout = "file.py:abc: some message"
        issues = ToolExecution._parse_line_issues(stdout, tool="t", category="c")
        assert len(issues) == 1
        assert issues[0].line_number == 1

    def test_column_number_is_always_none(self):
        stdout = "file.py:10:5: msg"
        issues = ToolExecution._parse_line_issues(stdout, tool="t", category="c")
        assert issues[0].column_number is None

    def test_custom_severity(self):
        stdout = "a.py:1: x"
        issues = ToolExecution._parse_line_issues(
            stdout, tool="t", category="c", severity="critical"
        )
        assert issues[0].severity == "critical"

    def test_default_severity_is_medium(self):
        stdout = "a.py:1: x"
        issues = ToolExecution._parse_line_issues(stdout, tool="t", category="c")
        assert issues[0].severity == "medium"


# ---------------------------------------------------------------------------
# run_tool dispatcher
# ---------------------------------------------------------------------------


class TestRunTool:
    async def test_dispatches_ruff_to_container_method(self, executor):
        """run_tool('ruff', ...) should call _run_ruff (container exec)."""
        with patch.object(
            executor, "_run_ruff", new_callable=AsyncMock, return_value=([], {"success": True})
        ) as mock_ruff:
            issues, result = await executor.run_tool(
                "ruff", ["ruff", "check", "."], Path("/repo"), "python"
            )

        mock_ruff.assert_awaited_once_with(Path("/repo"))
        assert result["success"] is True

    async def test_dispatches_ty_to_container_method(self, executor):
        with patch.object(
            executor, "_run_ty", new_callable=AsyncMock, return_value=([], {"success": True})
        ) as mock_ty:
            await executor.run_tool("ty", [], Path("/repo"), "python")
        mock_ty.assert_awaited_once_with(Path("/repo"))

    async def test_dispatches_bandit_to_container_method(self, executor):
        with patch.object(
            executor, "_run_bandit", new_callable=AsyncMock, return_value=([], {"success": True})
        ) as mock_bandit:
            await executor.run_tool("bandit", [], Path("/repo"), "python")
        mock_bandit.assert_awaited_once_with(Path("/repo"))

    async def test_dispatches_radon_to_container_method(self, executor):
        with patch.object(
            executor, "_run_radon", new_callable=AsyncMock, return_value=([], {"success": True})
        ) as mock_radon:
            await executor.run_tool("radon", [], Path("/repo"), "python")
        mock_radon.assert_awaited_once_with(Path("/repo"))

    async def test_run_tool_sets_container_name(self, executor):
        """A request-scoped container_name overrides the instance default."""
        with patch.object(
            executor, "_run_ruff", new_callable=AsyncMock, return_value=([], {"success": True})
        ):
            await executor.run_tool(
                "ruff", [], Path("/repo"), "python", container_name="codegraph-abc"
            )
        assert executor._container_name == "codegraph-abc"

    async def test_dispatches_ruff_format_to_container_method(self, executor):
        with patch.object(
            executor,
            "_run_ruff_format",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as mock_fmt:
            await executor.run_tool("ruff-format", [], Path("/repo"), "python")
        mock_fmt.assert_awaited_once_with(Path("/repo"))

    async def test_dispatches_eslint_to_container_method(self, executor):
        with patch.object(
            executor, "_run_eslint", new_callable=AsyncMock, return_value=([], {"success": True})
        ) as mock_eslint:
            await executor.run_tool("eslint", [], Path("/repo"), "javascript")
        mock_eslint.assert_awaited_once_with(Path("/repo"), "javascript")

    async def test_dispatches_tsc_to_container_method(self, executor):
        with patch.object(
            executor,
            "_run_typescript_compiler",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as mock_tsc:
            await executor.run_tool("tsc", [], Path("/repo"), "typescript")
        mock_tsc.assert_awaited_once_with(Path("/repo"))

    async def test_dispatches_checkstyle(self, executor):
        with patch.object(
            executor,
            "_run_checkstyle",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool("checkstyle", [], Path("/repo"), "java")
        m.assert_awaited_once()

    async def test_dispatches_clippy(self, executor):
        with patch.object(
            executor, "_run_clippy", new_callable=AsyncMock, return_value=([], {"success": True})
        ) as m:
            await executor.run_tool("clippy", [], Path("/repo"), "rust")
        m.assert_awaited_once()

    async def test_dispatches_golint(self, executor):
        with patch.object(
            executor, "_run_golint", new_callable=AsyncMock, return_value=([], {"success": True})
        ) as m:
            await executor.run_tool("golint", [], Path("/repo"), "go")
        m.assert_awaited_once()

    async def test_dispatches_cppcheck(self, executor):
        with patch.object(
            executor, "_run_cppcheck", new_callable=AsyncMock, return_value=([], {"success": True})
        ) as m:
            await executor.run_tool("cppcheck", [], Path("/repo"), "cpp")
        m.assert_awaited_once()

    async def test_dispatches_semgrep(self, executor):
        with patch.object(
            executor, "_run_semgrep", new_callable=AsyncMock, return_value=([], {"success": True})
        ) as m:
            await executor.run_tool("semgrep", [], Path("/repo"), "python")
        m.assert_awaited_once_with(Path("/repo"), "python")

    async def test_dispatches_dotnet_format(self, executor):
        with patch.object(
            executor,
            "_run_dotnet_format",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool("dotnet-format", [], Path("/repo"), "csharp")
        m.assert_awaited_once()

    async def test_unknown_tool_returns_empty(self, executor):
        """Unknown tool name should return empty issues with no error."""
        issues, result = await executor.run_tool("nonexistent-tool", [], Path("/repo"), "python")
        assert issues == []
        assert result["success"] is False

    async def test_subprocess_error_caught(self, executor):
        """SubprocessError from a local tool should be caught and stored in result."""
        with patch.object(executor, "_run_ruff", side_effect=subprocess.SubprocessError("boom")):
            issues, result = await executor.run_tool("ruff", [], Path("/repo"), "python")
        assert issues == []
        assert result["error"] == "boom"

    async def test_os_error_caught(self, executor):
        with patch.object(executor, "_run_ruff", side_effect=OSError("disk error")):
            issues, result = await executor.run_tool("ruff", [], Path("/repo"), "python")
        assert result["error"] == "disk error"

    async def test_json_decode_error_caught(self, executor):
        with patch.object(executor, "_run_ruff", side_effect=json.JSONDecodeError("bad", "", 0)):
            issues, result = await executor.run_tool("ruff", [], Path("/repo"), "python")
        assert "error" in result
        assert result["error"] is not None


# ---------------------------------------------------------------------------
# _ruff_severity_to_standard
# ---------------------------------------------------------------------------


class TestRuffSeverityToStandard:
    def test_f_codes_are_critical(self, executor):
        assert executor._ruff_severity_to_standard("F401") == "critical"
        assert executor._ruff_severity_to_standard("F811") == "critical"

    def test_e_codes_are_high(self, executor):
        assert executor._ruff_severity_to_standard("E501") == "high"
        assert executor._ruff_severity_to_standard("E123") == "high"

    def test_w_codes_are_medium(self, executor):
        assert executor._ruff_severity_to_standard("W291") == "medium"
        assert executor._ruff_severity_to_standard("W605") == "medium"

    def test_other_codes_are_low(self, executor):
        assert executor._ruff_severity_to_standard("I001") == "low"
        assert executor._ruff_severity_to_standard("N801") == "low"
        assert executor._ruff_severity_to_standard("UP006") == "low"
        assert executor._ruff_severity_to_standard("") == "low"


# ---------------------------------------------------------------------------
# Local tool: _run_ruff
# ---------------------------------------------------------------------------


@patch("processing.architecture.tool_execution._base.get_coding_containers_client")
class TestRunRuffContainer:
    async def test_ruff_parses_concise_output(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        ruff_output = (
            "src/app.py:3:1: F401 [*] os imported but unused\n"
            "src/util.py:10:80: E501 line too long\n"
            "Found 2 errors.\n"
        )
        mock_client.execute.return_value = {
            "exit_code": 1,
            "stdout": ruff_output,
            "stderr": "",
            "duration": 0.1,
        }

        issues, result = await executor._run_ruff(Path("/repo"))

        assert result["success"] is True
        assert len(issues) == 2

        assert issues[0].tool == "ruff"
        assert issues[0].severity == "critical"  # F401 -> critical
        assert issues[0].file_path == "src/app.py"
        assert issues[0].line_number == 3
        assert issues[0].column_number == 1
        assert issues[0].message == "os imported but unused"

        assert issues[1].severity == "high"  # E501 -> high
        assert issues[1].file_path == "src/util.py"

    async def test_ruff_clean_run(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        mock_client.execute.return_value = {
            "exit_code": 0,
            "stdout": "[]",
            "stderr": "",
            "duration": 0.1,
        }

        issues, result = await executor._run_ruff(Path("/repo"))

        assert issues == []
        # Tool ran cleanly: success means "ran", not "found nothing to parse".
        assert result["success"] is True

    async def test_ruff_invalid_json(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        mock_client.execute.return_value = {
            "exit_code": 1,
            "stdout": "not valid json",
            "stderr": "",
            "duration": 0.1,
        }

        issues, result = await executor._run_ruff(Path("/repo"))

        assert issues == []
        # exit 1 ("violations found") with nothing parseable is a FAILED run,
        # never a clean one.
        assert result["success"] is False
        assert result["error"].startswith("ruff exited 1")

    async def test_ruff_exec_failed(self, mock_get_client, executor, mock_client):
        # exit_code -1 == the exec call failed (400 whitelist reject / missing container).
        mock_get_client.return_value = mock_client
        mock_client.execute.return_value = {
            "exit_code": -1,
            "stdout": "",
            "stderr": "boom",
            "duration": 0.0,
        }

        issues, result = await executor._run_ruff(Path("/repo"))

        assert issues == []
        assert result["success"] is False

    async def test_ruff_tool_not_installed(self, mock_get_client, executor, mock_client):
        # 127 == command not found inside the container → not a run.
        mock_get_client.return_value = mock_client
        mock_client.execute.return_value = {
            "exit_code": 127,
            "stdout": "",
            "stderr": "ruff: not found",
            "duration": 0.0,
        }

        issues, result = await executor._run_ruff(Path("/repo"))

        assert issues == []
        assert result["success"] is False

    async def test_ruff_command_args(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        mock_client.execute.return_value = {
            "exit_code": 0,
            "stdout": "[]",
            "stderr": "",
            "duration": 0.1,
        }

        await executor._run_ruff(Path("/my/repo"))

        call = mock_client.execute.call_args
        assert call.kwargs["command"] == [
            "ruff",
            "check",
            "/my/repo",
            "--output-format=concise",
            "--extend-select",
            "I,PL",
        ]
        assert call.kwargs["timeout"] == 60
        assert call.kwargs["container_name"] == "test-worker"


# ---------------------------------------------------------------------------
# Local tool: _run_bandit
# ---------------------------------------------------------------------------


@patch("processing.architecture.tool_execution._base.get_coding_containers_client")
class TestRunBanditContainer:
    async def test_bandit_parses_json_output(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        bandit_output = json.dumps(
            {
                "results": [
                    {
                        "test_id": "B101",
                        "test_name": "assert_used",
                        "issue_severity": "LOW",
                        "issue_text": "Use of assert detected",
                        "filename": "tests/test_app.py",
                        "line_number": 5,
                        "col_offset": 0,
                        "more_info": "https://bandit.readthedocs.io/B101",
                    }
                ]
            }
        )
        mock_client.execute.return_value = {
            "exit_code": 1,
            "stdout": bandit_output,
            "stderr": "",
            "duration": 0.1,
        }

        issues, result = await executor._run_bandit(Path("/repo"))

        assert result["success"] is True
        assert len(issues) == 1
        assert issues[0].tool == "bandit"
        assert issues[0].severity == "low"
        assert issues[0].category == "security"
        assert issues[0].rule_id == "B101"
        assert issues[0].file_path == "tests/test_app.py"
        assert issues[0].line_number == 5

    async def test_bandit_command_args(self, mock_get_client, executor, mock_client):
        cmd = await self._bandit_cmd(mock_get_client, mock_client, executor, {})
        assert cmd == ["bandit", "-r", "/repo", "-f", "json"]

    @staticmethod
    def _exec_with_files(files):
        """execute() stub: ``cat <path>`` serves ``files``; bandit returns no results."""

        async def fake(**kw):
            cmd = kw["command"]
            if cmd[0] == "cat":
                if cmd[1] in files:
                    return {"exit_code": 0, "stdout": files[cmd[1]], "stderr": ""}
                return {"exit_code": 1, "stdout": "", "stderr": "No such file"}
            return {"exit_code": 0, "stdout": '{"results": []}', "stderr": ""}

        return fake

    async def _bandit_cmd(self, mock_get_client, mock_client, executor, files):
        mock_get_client.return_value = mock_client
        mock_client.execute.side_effect = self._exec_with_files(files)
        await executor._run_bandit(Path("/repo"))
        return mock_client.execute.call_args.kwargs["command"]

    async def test_bandit_uses_pyproject_tool_bandit(self, mock_get_client, executor, mock_client):
        files = {
            "/repo/pyproject.toml": '[tool.bandit]\nexclude_dirs = ["tests"]\nskips = ["B101"]\n'
        }
        cmd = await self._bandit_cmd(mock_get_client, mock_client, executor, files)
        assert cmd == ["bandit", "-c", "/repo/pyproject.toml", "-r", "/repo", "-f", "json"]

    async def test_bandit_skips_c_without_tool_bandit_section(
        self, mock_get_client, executor, mock_client
    ):
        files = {"/repo/pyproject.toml": "[tool.ruff]\nline-length = 100\n"}
        cmd = await self._bandit_cmd(mock_get_client, mock_client, executor, files)
        assert cmd == ["bandit", "-r", "/repo", "-f", "json"]

    async def test_bandit_invalid_pyproject_falls_back(
        self, mock_get_client, executor, mock_client
    ):
        files = {"/repo/pyproject.toml": "[tool.bandit\n"}
        cmd = await self._bandit_cmd(mock_get_client, mock_client, executor, files)
        assert cmd == ["bandit", "-r", "/repo", "-f", "json"]

    async def test_bandit_uses_dot_bandit_ini(self, mock_get_client, executor, mock_client):
        files = {"/repo/.bandit": "[bandit]\nskips = B101\n"}
        cmd = await self._bandit_cmd(mock_get_client, mock_client, executor, files)
        assert cmd == ["bandit", "--ini", "/repo/.bandit", "-r", "/repo", "-f", "json"]

    async def test_bandit_pyproject_section_wins_over_ini(
        self, mock_get_client, executor, mock_client
    ):
        files = {
            "/repo/pyproject.toml": "[tool.bandit]\nskips = ['B101']\n",
            "/repo/.bandit": "[bandit]\nskips = B101\n",
        }
        cmd = await self._bandit_cmd(mock_get_client, mock_client, executor, files)
        assert cmd[1:3] == ["-c", "/repo/pyproject.toml"]

    async def test_bandit_probe_exception_falls_back(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        calls = []

        async def fake(**kw):
            calls.append(kw["command"])
            if kw["command"][0] == "cat":
                raise RuntimeError("boom")
            return {"exit_code": 0, "stdout": '{"results": []}', "stderr": ""}

        mock_client.execute.side_effect = fake
        issues, result = await executor._run_bandit(Path("/repo"))
        assert result["success"] is True
        assert calls[-1] == ["bandit", "-r", "/repo", "-f", "json"]

    async def test_bandit_scoped_run_keeps_config(self, mock_get_client, executor, mock_client):
        executor._scope_paths = ["src/a.py"]
        files = {"/repo/pyproject.toml": "[tool.bandit]\nskips = ['B101']\n"}
        cmd = await self._bandit_cmd(mock_get_client, mock_client, executor, files)
        assert cmd == ["bandit", "-c", "/repo/pyproject.toml", "-r", "/repo/src/a.py", "-f", "json"]


# ---------------------------------------------------------------------------
# Container tool: _run_ty
# ---------------------------------------------------------------------------


@patch("processing.architecture.tool_execution._base.get_coding_containers_client")
class TestRunTyContainer:
    async def test_ty_exec_failed(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        mock_client.execute.return_value = {
            "exit_code": -1,
            "stdout": "",
            "stderr": "boom",
            "duration": 0.0,
        }

        issues, result = await executor._run_ty(Path("/repo"))

        assert issues == []
        assert result["success"] is False

    async def test_ty_not_installed(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        mock_client.execute.return_value = {
            "exit_code": 127,
            "stdout": "",
            "stderr": "ty: not found",
            "duration": 0.0,
        }

        issues, result = await executor._run_ty(Path("/repo"))

        assert issues == []
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Container tools call _execute_tool_in_container
# ---------------------------------------------------------------------------


class TestContainerToolMethods:
    async def test_ty_calls_execute_tool_in_container(self, executor, mock_client):
        mock_client.execute.return_value = {
            "exit_code": 0,
            "stdout": "[]",
            "stderr": "",
            "duration": 1.0,
        }
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_ty(Path("/repo"))

        assert result["success"] is True
        assert issues == []
        mock_client.execute.assert_awaited_once()
        call_kwargs = mock_client.execute.call_args
        assert call_kwargs.kwargs["command"] == [
            "ty",
            "check",
            "--output-format",
            "gitlab",
            "/repo",
        ]
        assert call_kwargs.kwargs["timeout"] == 120

    async def test_eslint_calls_container(self, executor, mock_client):
        mock_client.execute.return_value = {
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "duration": 0.5,
        }
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            await executor._run_eslint(Path("/repo"), "javascript")

        call_kwargs = mock_client.execute.call_args.kwargs
        assert call_kwargs["command"] == ["eslint", "/repo", "-f", "json"]

    async def test_eslint_parses_json_output(self, executor, mock_client):
        eslint_json = json.dumps(
            [
                {
                    "filePath": "src/index.js",
                    "messages": [
                        {
                            "severity": 2,
                            "ruleId": "no-unused-vars",
                            "message": "'x' is defined but never used",
                            "line": 5,
                            "column": 10,
                        },
                        {
                            "severity": 1,
                            "ruleId": "no-console",
                            "message": "Unexpected console statement",
                            "line": 12,
                            "column": 1,
                        },
                    ],
                }
            ]
        )
        mock_client.execute.return_value = {
            "exit_code": 0,
            "stdout": eslint_json,
            "stderr": "",
            "duration": 1.0,
        }
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_eslint(Path("/repo"), "javascript")

        assert len(issues) == 2
        assert issues[0].severity == "high"  # severity 2
        assert issues[0].rule_id == "no-unused-vars"
        assert issues[1].severity == "medium"  # severity 1
        assert issues[1].rule_id == "no-console"

    async def test_golint_calls_container_and_parses_line_output(self, executor, mock_client):
        mock_client.execute.return_value = {
            "exit_code": 0,
            "stdout": "main.go:10:5: exported function Foo should have comment",
            "stderr": "",
            "duration": 0.3,
        }
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_golint(Path("/repo"))

        assert result["success"] is True
        assert len(issues) == 1
        assert issues[0].tool == "golint"
        assert issues[0].file_path == "main.go"

    async def test_ruff_format_reports_formatting_issues(self, executor, mock_client):
        mock_client.execute.return_value = {
            "exit_code": 1,
            "stdout": "src/app.py:1:1: unformatted: File would be reformatted\n"
            "src/util.py:1:1: unformatted: File would be reformatted\n"
            "2 files would be reformatted\n",
            "stderr": "",
            "duration": 0.5,
        }
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_ruff_format(Path("/repo"))

        assert len(issues) == 2
        assert issues[0].tool == "ruff-format"
        assert issues[0].severity == "low"
        assert "src/app.py" in issues[0].message

    async def test_tsc_parses_typescript_errors(self, executor, mock_client):
        mock_client.execute.return_value = {
            "exit_code": 1,
            "stdout": "src/foo.ts(10,5): error TS2322: Type 'string' is not assignable to type 'number'\n",
            "stderr": "",
            "duration": 2.0,
        }
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_typescript_compiler(Path("/repo"))

        assert len(issues) == 1
        assert issues[0].tool == "tsc"
        assert issues[0].file_path == "src/foo.ts"
        assert issues[0].line_number == 10
        assert issues[0].severity == "high"
        assert issues[0].rule_id == "TS2322"

    async def test_semgrep_parses_json(self, executor, mock_client):
        semgrep_json = json.dumps(
            {
                "results": [
                    {
                        "check_id": "python.lang.security.audit.exec-detected",
                        "path": "app.py",
                        "start": {"line": 20, "col": 1},
                        "extra": {
                            "severity": "ERROR",
                            "message": "Detected use of exec()",
                            "metadata": {"source": "https://semgrep.dev/r/rule"},
                        },
                    }
                ]
            }
        )
        mock_client.execute.return_value = {
            "exit_code": 0,
            "stdout": semgrep_json,
            "stderr": "",
            "duration": 5.0,
        }
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_semgrep(Path("/repo"), "python")

        assert len(issues) == 1
        assert issues[0].severity == "high"
        assert issues[0].tool == "semgrep"
        assert issues[0].category == "security"

    async def test_checkstyle_parses_error_lines(self, executor, mock_client):
        mock_client.execute.return_value = {
            "exit_code": 0,
            "stdout": "[ERROR] App.java:15:5: Missing Javadoc comment [JavadocMethod]\n[WARN] App.java:20: Line is too long [LineLength]\n",
            "stderr": "",
            "duration": 1.0,
        }
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_checkstyle(Path("/repo"))

        assert len(issues) == 2
        assert issues[0].severity == "high"
        assert issues[0].rule_id == "JavadocMethod"
        assert issues[1].severity == "medium"
        assert issues[1].rule_id == "LineLength"

    async def test_container_tool_with_no_stdout(self, executor, mock_client):
        """When container returns no stdout, no issues should be parsed."""
        mock_client.execute.return_value = {
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "duration": 0.1,
        }
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_eslint(Path("/repo"), "javascript")

        assert issues == []
        assert result["success"] is True
