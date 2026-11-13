"""Unit tests for processing.architecture.tool_execution — comprehensive coverage.

Covers every public/private method in ToolExecution that the existing
test_tool_execution.py does not (or only minimally) test:
  - _run_ty (GitLab JSON report + exit contract)
  - _run_ruff (fix suggestions, PL* severities), _run_ruff_format
  - _run_bandit (edge cases)
  - _run_prettier, _run_jshint
  - _run_pmd, _run_spotbugs
  - _run_go_vet, _run_golangci_lint, _run_gosec
  - _run_clippy, _run_rustfmt, _run_cargo_audit
  - _run_cppcheck, _run_clang_tidy, _run_cpplint
  - _run_dotnet_format, _run_roslynator
  - _run_semgrep edge cases
  - run_tool dispatcher edge cases
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from processing.architecture.tool_execution import ToolExecution
from processing.architecture.tool_execution._rust import CARGO_AUDIT_SEVERITY


@pytest.fixture()
def executor():
    return ToolExecution(container_name="test-worker")


@pytest.fixture()
def mock_client():
    client = AsyncMock()
    client.execute = AsyncMock(
        return_value={
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "duration": 0.5,
        }
    )
    return client


def _container_response(stdout="", stderr="", exit_code=0, duration=0.5):
    return {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration": duration,
    }


# ===================================================================
# _run_ty — GitLab JSON report
# ===================================================================


def _ty_row(rule, line, severity="major", path="app.py"):
    return {
        "check_name": rule,
        "description": f"{rule}: message for {rule}",
        "severity": severity,
        "fingerprint": "abc",
        "location": {
            "path": path,
            "positions": {"begin": {"line": line, "column": 5}, "end": {"line": line, "column": 9}},
        },
    }


@patch("processing.architecture.tool_execution._base.get_coding_containers_client")
class TestRunTyGitlabOutput:
    async def test_ty_parses_gitlab_output(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        stdout = json.dumps(
            [
                _ty_row("invalid-assignment", 10),
                _ty_row("possibly-unresolved-reference", 20, "minor"),
            ]
        )
        mock_client.execute.return_value = _container_response(stdout=stdout, exit_code=1)

        issues, result = await executor._run_ty(Path("/repo"))

        assert result["success"] is True
        assert [(i.rule_id, i.severity, i.line_number) for i in issues] == [
            ("invalid-assignment", "high", 10),
            ("possibly-unresolved-reference", "medium", 20),
        ]
        assert issues[0].message == "message for invalid-assignment"
        assert issues[0].file_path == "app.py"
        assert issues[0].column_number == 5
        assert {i.tool for i in issues} == {"ty"}

    async def test_ty_bad_config_no_json_is_a_failure(self, mock_get_client, executor, mock_client):
        """ty exits 2 with NO JSON on an unparseable config: a tool failure."""
        mock_get_client.return_value = mock_client
        mock_client.execute.return_value = _container_response(
            stderr="ty failed\n  Cause: /repo/pyproject.toml is not a valid `pyproject.toml`\n",
            exit_code=2,
        )

        issues, result = await executor._run_ty(Path("/repo"))

        assert issues == []
        assert result["success"] is False
        assert "not a valid `pyproject.toml`" in result["error"]


# ===================================================================
# _run_ruff edge cases
# ===================================================================


@patch("processing.architecture.tool_execution._base.get_coding_containers_client")
class TestRunRuffEdgeCases:
    async def test_ruff_fixable_row_gets_fix_suggestion(
        self, mock_get_client, executor, mock_client
    ):
        mock_get_client.return_value = mock_client
        ruff_output = "app.py:1:1: F401 [*] os imported but unused\nFound 1 error.\n"
        mock_client.execute.return_value = _container_response(stdout=ruff_output, exit_code=1)

        issues, result = await executor._run_ruff(Path("/repo"))

        assert result["success"] is True
        assert len(issues) == 1
        assert issues[0].resolve_suggestions == "Auto-fixable: ruff check --fix --select F401"
        assert issues[0].rule_name == "F401"
        assert issues[0].message == "os imported but unused"

    async def test_ruff_without_resolve(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        ruff_output = "a.py:1:80: E501 line too long\n"
        mock_client.execute.return_value = _container_response(stdout=ruff_output, exit_code=1)

        issues, result = await executor._run_ruff(Path("/repo"))

        assert issues[0].resolve_suggestions is None


# ===================================================================
# _run_bandit edge cases
# ===================================================================


@patch("processing.architecture.tool_execution._base.get_coding_containers_client")
class TestRunBanditEdgeCases:
    async def test_bandit_exec_failed(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        mock_client.execute.return_value = _container_response(stderr="boom", exit_code=-1)
        issues, result = await executor._run_bandit(Path("/repo"))
        assert issues == []
        assert result["success"] is False

    async def test_bandit_not_installed(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        mock_client.execute.return_value = _container_response(
            stderr="bandit: not found", exit_code=127
        )
        issues, result = await executor._run_bandit(Path("/repo"))
        assert issues == []
        assert result["success"] is False

    async def test_bandit_invalid_json(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        mock_client.execute.return_value = _container_response(stdout="not json", exit_code=0)
        issues, result = await executor._run_bandit(Path("/repo"))
        assert issues == []
        # exec ran (success True); unparseable output simply yields no issues.
        assert result["success"] is True

    async def test_bandit_empty_output(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        mock_client.execute.return_value = _container_response(stdout="", exit_code=0)
        issues, result = await executor._run_bandit(Path("/repo"))
        assert issues == []
        assert result["success"] is True


# ===================================================================
# _run_radon
# ===================================================================


@patch("processing.architecture.tool_execution._base.get_coding_containers_client")
class TestRunRadon:
    async def test_radon_parses_high_complexity(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        radon_output = json.dumps(
            {
                "app.py": [
                    {"name": "big_func", "complexity": 25, "lineno": 4, "col_offset": 0},
                    {"name": "mid_func", "complexity": 12, "lineno": 40, "col_offset": 0},
                    {"name": "ok_func", "complexity": 3, "lineno": 80, "col_offset": 0},
                ]
            }
        )
        mock_client.execute.return_value = _container_response(stdout=radon_output)

        issues, result = await executor._run_radon(Path("/repo"))

        assert result["success"] is True
        # Only complexity >= 10 is reported (25 -> high, 12 -> medium; 3 dropped).
        assert len(issues) == 2
        assert issues[0].tool == "radon"
        assert issues[0].severity == "high"
        assert issues[0].file_path == "app.py"
        assert issues[0].line_number == 4
        assert issues[1].severity == "medium"

    async def test_radon_command_args(self, mock_get_client, executor, mock_client):
        mock_get_client.return_value = mock_client
        mock_client.execute.return_value = _container_response(stdout="{}")

        await executor._run_radon(Path("/repo"))

        call = mock_client.execute.call_args
        assert call.kwargs["command"] == ["radon", "cc", "/repo", "-j", "-n", "C"]


# ===================================================================
# _run_ruff severities for the pylint (PL*) family, and _run_ruff_format
# ===================================================================


def _ruff_rows(*codes):
    return "".join(f"app.py:1:1: {c} {c}\n" for c in codes)


class TestRuffPylintSeverities:
    @pytest.mark.parametrize(
        ("code", "severity"),
        [
            ("PLC0415", "low"),
            ("PLR0913", "low"),
            ("PLW0602", "medium"),
            ("PLE0101", "critical"),
            ("I001", "low"),
            ("E501", "high"),
            ("W291", "medium"),
            ("F401", "critical"),
        ],
    )
    async def test_ruff_code_severity(self, executor, mock_client, code, severity):
        mock_client.execute.return_value = _container_response(stdout=_ruff_rows(code), exit_code=1)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_ruff(Path("/repo"))
        assert [(i.rule_id, i.severity) for i in issues] == [(code, severity)]

    async def test_ruff_exit_1_unparseable_is_a_failure(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="not json", exit_code=1)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_ruff(Path("/repo"))
        assert issues == []
        assert result["success"] is False


class TestRunRuffFormatEdgeCases:
    async def test_ruff_format_unformatted_rows(self, executor, mock_client):
        stdout = "/repo/app.py:3:1: unformatted: File would be reformatted\n1 file would be reformatted\n"
        mock_client.execute.return_value = _container_response(stdout=stdout, exit_code=1)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_ruff_format(Path("/repo"))
        assert len(issues) == 1
        assert issues[0].tool == "ruff-format"
        assert issues[0].rule_id == "unformatted"
        assert issues[0].file_path == "/repo/app.py"
        argv = mock_client.execute.call_args.kwargs["command"]
        assert argv == ["ruff", "format", "--check", "--output-format=concise", "/repo"]

    async def test_ruff_format_clean(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="[]", exit_code=0)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_ruff_format(Path("/repo"))
        assert issues == []
        assert result["success"] is True

    async def test_ruff_format_not_success(self, executor, mock_client):
        mock_client.execute.return_value = {"exit_code": -1, "stderr": "Container failed"}
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_ruff_format(Path("/repo"))
        assert issues == []
        assert result["success"] is False


# ===================================================================
# _run_prettier
# ===================================================================


class TestRunPrettier:
    async def test_prettier_finds_unformatted_files(self, executor, mock_client):
        stdout = (
            "Checking formatting...\nsrc/index.js\nsrc/util.ts\nAll matched files use Prettier\n"
        )
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_prettier(Path("/repo"), "javascript")
        # Lines starting with "Checking" or "All" are skipped
        assert len(issues) == 2
        assert issues[0].file_path == "src/index.js"
        assert issues[0].tool == "prettier"
        assert issues[0].severity == "low"
        assert issues[1].file_path == "src/util.ts"

    async def test_prettier_empty_output(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_prettier(Path("/repo"), "javascript")
        assert issues == []

    async def test_prettier_all_formatted(self, executor, mock_client):
        stdout = "Checking formatting...\nAll matched files use Prettier code style!\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_prettier(Path("/repo"), "typescript")
        assert issues == []


# ===================================================================
# _run_jshint
# ===================================================================


class TestRunJshint:
    async def test_jshint_parses_list_format(self, executor, mock_client):
        data = [
            {
                "file": "app.js",
                "error": {
                    "code": "W033",
                    "reason": "Missing semicolon",
                    "line": 5,
                    "character": 10,
                },
            }
        ]
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_jshint(Path("/repo"))
        assert len(issues) == 1
        assert issues[0].severity == "medium"  # W-code -> medium
        assert issues[0].rule_id == "W033"
        assert issues[0].file_path == "app.js"

    async def test_jshint_parses_dict_result_format(self, executor, mock_client):
        data = {
            "result": [
                {
                    "file": "app.js",
                    "error": {
                        "code": "E041",
                        "reason": "Unexpected token",
                        "line": 3,
                        "character": 1,
                    },
                }
            ]
        }
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_jshint(Path("/repo"))
        assert len(issues) == 1
        assert issues[0].severity == "high"  # E-code -> high

    async def test_jshint_invalid_json(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="bad json")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_jshint(Path("/repo"))
        assert issues == []

    async def test_jshint_empty_stdout(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_jshint(Path("/repo"))
        assert issues == []


# ===================================================================
# _run_typescript_compiler edge cases
# ===================================================================


class TestRunTypescriptCompilerEdgeCases:
    async def test_tsc_no_error_prefix(self, executor, mock_client):
        """Line with paren format but no 'error' prefix."""
        stdout = "src/a.ts(5,1): warning some warning\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_typescript_compiler(Path("/repo"))
        assert len(issues) == 1
        assert issues[0].rule_id == "tsc-error"  # no "error " prefix

    async def test_tsc_invalid_line_number(self, executor, mock_client):
        stdout = "src/a.ts(abc,1): error TS0001: bad\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_typescript_compiler(Path("/repo"))
        assert len(issues) == 1
        assert issues[0].line_number == 1

    async def test_tsc_invalid_col_number(self, executor, mock_client):
        stdout = "src/a.ts(5,abc): error TS0001: bad\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_typescript_compiler(Path("/repo"))
        assert len(issues) == 1
        assert issues[0].column_number is None

    async def test_tsc_single_loc_no_comma(self, executor, mock_client):
        stdout = "src/a.ts(5): error TS1234: some error\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_typescript_compiler(Path("/repo"))
        assert len(issues) == 1
        assert issues[0].column_number is None

    async def test_tsc_skips_lines_without_paren_format(self, executor, mock_client):
        stdout = "some random noise\n\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_typescript_compiler(Path("/repo"))
        assert issues == []


# ===================================================================
# _run_checkstyle edge cases
# ===================================================================


class TestRunCheckstyleEdgeCases:
    async def test_checkstyle_skips_non_error_warn_lines(self, executor, mock_client):
        stdout = "[INFO] Starting audit\n[ERROR] File.java:10: msg [Rule]\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_checkstyle(Path("/repo"))
        assert len(issues) == 1

    async def test_checkstyle_no_bracket_rule(self, executor, mock_client):
        stdout = "[ERROR] File.java:5: Missing space\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_checkstyle(Path("/repo"))
        assert len(issues) == 1
        assert issues[0].rule_id == "checkstyle-issue"  # no bracket -> default

    async def test_checkstyle_non_numeric_line(self, executor, mock_client):
        stdout = "[ERROR] File.java:abc: msg [Rule]\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_checkstyle(Path("/repo"))
        assert len(issues) == 1
        assert issues[0].line_number == 1


# ===================================================================
# _run_pmd
# ===================================================================


class TestRunPmd:
    async def test_pmd_parses_json(self, executor, mock_client):
        data = {
            "files": [
                {
                    "filename": "App.java",
                    "violations": [
                        {
                            "rule": "UnusedImports",
                            "ruleset": "Best Practices",
                            "description": "Unused import",
                            "priority": 1,
                            "beginline": 3,
                            "begincolumn": 1,
                            "externalInfoUrl": "https://pmd.github.io",
                        },
                        {
                            "rule": "AvoidStarImport",
                            "ruleset": "Best Practices",
                            "description": "Star import",
                            "priority": 4,
                            "beginline": 5,
                            "begincolumn": None,
                        },
                    ],
                }
            ]
        }
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_pmd(Path("/repo"))

        assert len(issues) == 2
        assert issues[0].severity == "critical"  # priority 1
        assert issues[0].rule_id == "UnusedImports"
        assert issues[0].tool == "pmd"
        assert issues[1].severity == "low"  # priority 4

    async def test_pmd_priority_2_is_high(self, executor, mock_client):
        data = {
            "files": [
                {
                    "filename": "A.java",
                    "violations": [
                        {
                            "rule": "X",
                            "ruleset": "R",
                            "description": "d",
                            "priority": 2,
                            "beginline": 1,
                        }
                    ],
                }
            ]
        }
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_pmd(Path("/repo"))
        assert issues[0].severity == "high"

    async def test_pmd_priority_3_is_medium(self, executor, mock_client):
        data = {
            "files": [
                {
                    "filename": "A.java",
                    "violations": [
                        {
                            "rule": "X",
                            "ruleset": "R",
                            "description": "d",
                            "priority": 3,
                            "beginline": 1,
                        }
                    ],
                }
            ]
        }
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_pmd(Path("/repo"))
        assert issues[0].severity == "medium"

    async def test_pmd_invalid_json(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="broken")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_pmd(Path("/repo"))
        assert issues == []

    async def test_pmd_empty_stdout(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_pmd(Path("/repo"))
        assert issues == []


# ===================================================================
# _run_spotbugs
# ===================================================================


class TestRunSpotbugs:
    async def test_spotbugs_parses_xml(self, executor, mock_client):
        xml_output = """<?xml version="1.0"?>
<BugCollection>
  <BugInstance type="NP_NULL_ON_SOME_PATH" priority="1" category="CORRECTNESS">
    <ShortMessage>Null pointer dereference</ShortMessage>
    <SourceLine sourcepath="App.java" start="10"/>
  </BugInstance>
  <BugInstance type="DM_DEFAULT_ENCODING" priority="2" category="I18N">
    <ShortMessage>Reliance on default encoding</ShortMessage>
    <SourceLine sourcepath="Util.java" start="20"/>
  </BugInstance>
  <BugInstance type="ST_WRITE_TO_STATIC_FROM_INSTANCE_METHOD" priority="3" category="STYLE">
    <SourceLine sourcepath="Helper.java" start="bad_number"/>
  </BugInstance>
</BugCollection>"""
        mock_client.execute.return_value = _container_response(stdout=xml_output)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_spotbugs(Path("/repo"))

        assert len(issues) == 3
        assert issues[0].severity == "high"  # priority 1
        assert issues[0].rule_id == "NP_NULL_ON_SOME_PATH"
        assert issues[0].file_path == "App.java"
        assert issues[0].line_number == 10

        assert issues[1].severity == "medium"  # priority 2
        assert issues[1].file_path == "Util.java"

        assert issues[2].severity == "low"  # priority 3
        assert issues[2].line_number == 1  # bad number -> default 1

    async def test_spotbugs_no_short_message(self, executor, mock_client):
        xml_output = """<?xml version="1.0"?>
<BugCollection>
  <BugInstance type="NP_NULL" priority="1" category="CORRECTNESS">
    <SourceLine sourcepath="A.java" start="1"/>
  </BugInstance>
</BugCollection>"""
        mock_client.execute.return_value = _container_response(stdout=xml_output)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_spotbugs(Path("/repo"))
        assert issues[0].message == "NP_NULL"  # falls back to bug_type

    async def test_spotbugs_no_source_line(self, executor, mock_client):
        xml_output = """<?xml version="1.0"?>
<BugCollection>
  <BugInstance type="X" priority="2" category="C">
    <ShortMessage>Msg</ShortMessage>
  </BugInstance>
</BugCollection>"""
        mock_client.execute.return_value = _container_response(stdout=xml_output)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_spotbugs(Path("/repo"))
        assert issues[0].file_path == ""
        assert issues[0].line_number == 1

    async def test_spotbugs_invalid_xml(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="not xml")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_spotbugs(Path("/repo"))
        assert issues == []

    async def test_spotbugs_empty_stdout(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_spotbugs(Path("/repo"))
        assert issues == []


# ===================================================================
# _run_go_vet
# ===================================================================


class TestRunGoVet:
    async def test_go_vet_parses_stderr(self, executor, mock_client):
        stderr = "main.go:15: unreachable code\n"
        mock_client.execute.return_value = _container_response(stderr=stderr)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_go_vet(Path("/repo"))
        assert len(issues) == 1
        assert issues[0].tool == "go-vet"
        assert issues[0].category == "bug-risk"
        assert issues[0].severity == "high"

    async def test_go_vet_empty_stderr(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stderr="")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_go_vet(Path("/repo"))
        assert issues == []

    async def test_go_vet_not_success(self, executor, mock_client):
        mock_client.execute.return_value = {
            "exit_code": -1,
            "stderr": "container died",
        }
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_go_vet(Path("/repo"))
        assert issues == []


# ===================================================================
# _run_golangci_lint
# ===================================================================


class TestRunGolangciLint:
    async def test_golangci_lint_parses_json(self, executor, mock_client):
        data = {
            "Issues": [
                {
                    "FromLinter": "errcheck",
                    "Text": "Error return value not checked",
                    "Severity": "error",
                    "Pos": {"Filename": "main.go", "Line": 10, "Column": 5},
                },
                {
                    "FromLinter": "golint",
                    "Text": "exported function should have comment",
                    "Severity": "warning",
                    "Pos": {"Filename": "lib.go", "Line": 20, "Column": None},
                },
                {
                    "FromLinter": "stylecheck",
                    "Text": "comment on exported",
                    "Severity": "info",
                    "Pos": {"Filename": "util.go", "Line": 5},
                },
            ]
        }
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_golangci_lint(Path("/repo"))

        assert len(issues) == 3
        assert issues[0].severity == "high"
        assert issues[0].tool == "golangci-lint"
        assert issues[1].severity == "medium"
        assert issues[2].severity == "low"

    async def test_golangci_lint_unknown_severity(self, executor, mock_client):
        data = {
            "Issues": [
                {
                    "FromLinter": "x",
                    "Text": "msg",
                    "Severity": "unknown_level",
                    "Pos": {"Filename": "a.go", "Line": 1},
                }
            ]
        }
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_golangci_lint(Path("/repo"))
        assert issues[0].severity == "medium"  # default

    async def test_golangci_lint_invalid_json(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="bad")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_golangci_lint(Path("/repo"))
        assert issues == []


# ===================================================================
# _run_gosec
# ===================================================================


class TestRunGosec:
    async def test_gosec_parses_json(self, executor, mock_client):
        data = {
            "Issues": [
                {
                    "severity": "HIGH",
                    "rule_id": "G101",
                    "details": "Potential hardcoded credentials",
                    "file": "config.go",
                    "line": "15",
                },
                {
                    "severity": "LOW",
                    "rule_id": "G104",
                    "details": "Unhandled error",
                    "file": "main.go",
                    "line": "10-15",
                },
            ]
        }
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_gosec(Path("/repo"))

        assert len(issues) == 2
        assert issues[0].severity == "high"
        assert issues[0].tool == "gosec"
        assert issues[0].category == "security"
        assert issues[0].line_number == 15

        assert issues[1].severity == "low"
        assert issues[1].line_number == 10  # "10-15" -> int("10")

    async def test_gosec_unknown_severity(self, executor, mock_client):
        data = {
            "Issues": [
                {
                    "severity": "UNKNOWN",
                    "rule_id": "G999",
                    "details": "msg",
                    "file": "a.go",
                    "line": "1",
                }
            ]
        }
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_gosec(Path("/repo"))
        assert issues[0].severity == "medium"

    async def test_gosec_bad_line_number(self, executor, mock_client):
        data = {
            "Issues": [
                {
                    "severity": "MEDIUM",
                    "rule_id": "G101",
                    "details": "msg",
                    "file": "a.go",
                    "line": "abc",
                }
            ]
        }
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_gosec(Path("/repo"))
        assert issues[0].line_number == 1

    async def test_gosec_invalid_json(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="bad")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_gosec(Path("/repo"))
        assert issues == []


# ===================================================================
# _run_clippy
# ===================================================================


class TestRunClippy:
    async def test_clippy_parses_json_lines(self, executor, mock_client):
        lines = [
            json.dumps(
                {
                    "reason": "compiler-message",
                    "message": {
                        "level": "warning",
                        "message": "unused variable",
                        "code": {"code": "unused_variables"},
                        "spans": [
                            {
                                "file_name": "src/main.rs",
                                "line_start": 10,
                                "column_start": 5,
                            }
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "reason": "compiler-message",
                    "message": {
                        "level": "error",
                        "message": "type mismatch",
                        "code": {"code": "E0308"},
                        "spans": [
                            {
                                "file_name": "src/lib.rs",
                                "line_start": 20,
                                "column_start": 1,
                            }
                        ],
                    },
                }
            ),
            json.dumps({"reason": "build-finished", "success": True}),
        ]
        stdout = "\n".join(lines)
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_clippy(Path("/repo"))

        assert len(issues) == 2
        assert issues[0].severity == "medium"  # warning
        assert issues[0].rule_id == "unused_variables"
        assert issues[0].file_path == "src/main.rs"
        assert issues[0].tool == "clippy"

        assert issues[1].severity == "high"  # error
        assert issues[1].rule_id == "E0308"

    async def test_clippy_no_spans(self, executor, mock_client):
        """A span-less message ("aborting due to ...", "N warnings emitted") is a
        summary, not a finding: it names no file, so it is not reported."""
        line = json.dumps(
            {
                "reason": "compiler-message",
                "message": {
                    "level": "note",
                    "message": "some note",
                    "code": None,
                    "spans": [],
                },
            }
        )
        mock_client.execute.return_value = _container_response(stdout=line)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_clippy(Path("/repo"))
        assert issues == []

    async def test_clippy_code_is_not_dict(self, executor, mock_client):
        """When code is a string instead of dict."""
        line = json.dumps(
            {
                "reason": "compiler-message",
                "message": {
                    "level": "help",
                    "message": "help msg",
                    "code": "some_code",
                    "spans": [{"file_name": "src/a.rs", "line_start": 3}],
                },
            }
        )
        mock_client.execute.return_value = _container_response(stdout=line)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_clippy(Path("/repo"))
        assert issues[0].rule_id == "rustc"  # not a dict: no lint code
        assert issues[0].severity == "low"  # help

    async def test_clippy_invalid_json_line_skipped(self, executor, mock_client):
        stdout = "not json\n" + json.dumps(
            {
                "reason": "compiler-message",
                "message": {
                    "level": "warning",
                    "message": "msg",
                    "code": {"code": "W1"},
                    "spans": [{"file_name": "src/a.rs", "line_start": 3}],
                },
            }
        )
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_clippy(Path("/repo"))
        assert len(issues) == 1

    async def test_clippy_unknown_severity(self, executor, mock_client):
        line = json.dumps(
            {
                "reason": "compiler-message",
                "message": {
                    "level": "unknown_level",
                    "message": "msg",
                    "code": {"code": "X"},
                    "spans": [{"file_name": "src/a.rs", "line_start": 3}],
                },
            }
        )
        mock_client.execute.return_value = _container_response(stdout=line)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_clippy(Path("/repo"))
        assert issues[0].severity == "medium"  # default


# ===================================================================
# _run_rustfmt
# ===================================================================


class TestRunRustfmt:
    async def test_rustfmt_finds_diffs(self, executor, mock_client):
        stdout = "Diff in src/main.rs:\nDiff in src/lib.rs:\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_rustfmt(Path("/repo"))
        assert len(issues) == 2
        assert issues[0].file_path == "src/main.rs"
        assert issues[0].tool == "rustfmt"
        assert issues[0].severity == "low"
        assert issues[1].file_path == "src/lib.rs"

    async def test_rustfmt_no_diffs(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_rustfmt(Path("/repo"))
        assert issues == []

    async def test_rustfmt_skips_non_diff_lines(self, executor, mock_client):
        stdout = "some noise\nDiff in src/main.rs:\nmore noise\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_rustfmt(Path("/repo"))
        assert len(issues) == 1


# ===================================================================
# _run_cargo_audit
# ===================================================================


class TestRunCargoAudit:
    async def test_cargo_audit_real_report(self, executor, mock_client):
        """A real ``cargo audit --json`` report (tests/fixtures/rust_quality/audit,
        regenerated by its generate.sh): one finding per vulnerable package, all at
        the one fixed severity."""
        report = (
            Path(__file__).resolve().parents[1] / "fixtures/rust_quality/audit/cargo-audit.json"
        ).read_text()
        mock_client.execute.return_value = _container_response(stdout=report, exit_code=1)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_cargo_audit(Path("/repo"))

        assert [(i.rule_id, i.file_path, i.tool, i.category) for i in issues] == [
            ("RUSTSEC-2021-0003", "Cargo.lock", "cargo-audit", "security")
        ]
        assert issues[0].message.startswith("smallvec 1.6.0: ")
        assert {i.severity for i in issues} == {CARGO_AUDIT_SEVERITY}

    async def test_cargo_audit_invalid_json(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="bad")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_cargo_audit(Path("/repo"))
        assert issues == []

    async def test_cargo_audit_empty_vulnerabilities(self, executor, mock_client):
        data = {"vulnerabilities": {"list": []}}
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_cargo_audit(Path("/repo"))
        assert issues == []


# ===================================================================
# _run_cppcheck
# ===================================================================


class TestRunCppcheck:
    async def test_cppcheck_parses_stderr(self, executor, mock_client):
        output = (
            "'file.cpp:10:error:null pointer dereference'\n"
            "'util.cpp:20:warning:unused variable'\n"
            "'helper.cpp:5:style:redundant condition'\n"
            "'perf.cpp:30:performance:inefficient operation'\n"
            "'port.cpp:1:portability:platform specific'\n"
            "'info.cpp:1:information:info message'\n"
        )
        mock_client.execute.return_value = _container_response(stderr=output)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_cppcheck(Path("/repo"))

        assert len(issues) == 6
        assert issues[0].severity == "high"  # error
        assert issues[0].category == "bug-risk"
        assert issues[1].severity == "medium"  # warning
        assert issues[2].severity == "low"  # style
        assert issues[3].severity == "medium"  # performance
        assert issues[4].severity == "low"  # portability
        assert issues[5].severity == "low"  # information

    async def test_cppcheck_unknown_severity(self, executor, mock_client):
        output = "'file.cpp:1:unknown_sev:msg'\n"
        mock_client.execute.return_value = _container_response(stderr=output)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_cppcheck(Path("/repo"))
        assert issues[0].severity == "medium"  # default

    async def test_cppcheck_non_numeric_line(self, executor, mock_client):
        output = "'file.cpp:abc:error:msg'\n"
        mock_client.execute.return_value = _container_response(stderr=output)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_cppcheck(Path("/repo"))
        assert issues[0].line_number == 1

    async def test_cppcheck_skips_short_lines(self, executor, mock_client):
        output = "noise\nab\n'file.cpp:1:error:msg'\n"
        mock_client.execute.return_value = _container_response(stderr=output)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_cppcheck(Path("/repo"))
        assert len(issues) == 1

    async def test_cppcheck_uses_stdout_fallback(self, executor, mock_client):
        """If stderr is empty, cppcheck falls back to stdout."""
        output = "'file.cpp:1:error:msg'\n"
        mock_client.execute.return_value = _container_response(stdout=output, stderr="")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_cppcheck(Path("/repo"))
        assert len(issues) == 1

    async def test_cppcheck_not_success(self, executor, mock_client):
        mock_client.execute.return_value = {
            "exit_code": -1,
            "stderr": "fail",
        }
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_cppcheck(Path("/repo"))
        assert issues == []


# ===================================================================
# _run_clang_tidy
# ===================================================================


class TestRunClangTidy:
    async def test_clang_tidy_parses_output(self, executor, mock_client):
        stdout = (
            "file.cpp:10:5: error: use of undeclared identifier [undeclared-id]\n"
            "file.cpp:20:3: warning: unused variable [unused-var]\n"
            "file.cpp:30:1: note: declared here [note-check]\n"
        )
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_clang_tidy(Path("/repo"))

        assert len(issues) == 3
        assert issues[0].severity == "high"  # error
        assert issues[0].rule_id == "undeclared-id"
        assert issues[0].line_number == 10
        assert issues[0].column_number == 5

        assert issues[1].severity == "medium"  # warning
        assert issues[1].rule_id == "unused-var"

        assert issues[2].severity == "low"  # note

    async def test_clang_tidy_no_bracket_rule(self, executor, mock_client):
        stdout = "file.cpp:1:1: error: some message without bracket\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_clang_tidy(Path("/repo"))
        assert len(issues) == 1
        assert issues[0].rule_id == "clang-tidy-issue"

    async def test_clang_tidy_invalid_line_skipped(self, executor, mock_client):
        stdout = "file.cpp:abc:1: error: msg\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_clang_tidy(Path("/repo"))
        assert issues == []  # skipped because line number invalid

    async def test_clang_tidy_invalid_col(self, executor, mock_client):
        stdout = "file.cpp:10:abc: error: msg\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_clang_tidy(Path("/repo"))
        assert issues[0].column_number is None

    async def test_clang_tidy_empty_stdout(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_clang_tidy(Path("/repo"))
        assert issues == []

    async def test_clang_tidy_too_few_parts(self, executor, mock_client):
        stdout = "short:line\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_clang_tidy(Path("/repo"))
        assert issues == []


# ===================================================================
# _run_cpplint
# ===================================================================


class TestRunCpplint:
    async def test_cpplint_parses_stderr(self, executor, mock_client):
        stderr = (
            "file.cpp:10:  Missing space [whitespace/braces] [5]\n"
            "file.cpp:20:  Tab found [whitespace/tab] [1]\n"
            "Done processing file.cpp\n"
            "Total errors found: 2\n"
        )
        mock_client.execute.return_value = _container_response(stderr=stderr)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_cpplint(Path("/repo"))

        assert len(issues) == 2
        assert issues[0].rule_id == "whitespace/braces"
        assert issues[0].tool == "cpplint"
        assert issues[0].line_number == 10
        assert issues[1].rule_id == "whitespace/tab"

    async def test_cpplint_invalid_line_number_skipped(self, executor, mock_client):
        stderr = "file.cpp:abc:  msg [cat] [1]\n"
        mock_client.execute.return_value = _container_response(stderr=stderr)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_cpplint(Path("/repo"))
        assert issues == []

    async def test_cpplint_uses_stdout_fallback(self, executor, mock_client):
        stdout = "file.cpp:5:  msg [cat] [1]\n"
        mock_client.execute.return_value = _container_response(stdout=stdout, stderr="")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_cpplint(Path("/repo"))
        assert len(issues) == 1

    async def test_cpplint_no_bracket_category(self, executor, mock_client):
        stderr = "file.cpp:5:  plain message no brackets\n"
        mock_client.execute.return_value = _container_response(stderr=stderr)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_cpplint(Path("/repo"))
        assert len(issues) == 1
        assert issues[0].rule_id == "cpplint-issue"


# ===================================================================
# _run_dotnet_format
# ===================================================================


class TestRunDotnetFormat:
    async def test_dotnet_format_finds_cs_files(self, executor, mock_client):
        stdout = "Program.cs\nHelper.csproj\n"
        mock_client.execute.return_value = _container_response(stdout=stdout, exit_code=1)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_dotnet_format(Path("/repo"))
        assert len(issues) == 2
        assert issues[0].file_path == "Program.cs"
        assert issues[0].tool == "dotnet-format"
        assert issues[1].file_path == "Helper.csproj"

    async def test_dotnet_format_cs_with_paren(self, executor, mock_client):
        stdout = "Program.cs(10,5): warning\n"
        mock_client.execute.return_value = _container_response(stdout=stdout, exit_code=1)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_dotnet_format(Path("/repo"))
        assert len(issues) == 1
        assert issues[0].file_path == "Program.cs"

    async def test_dotnet_format_exit_code_zero(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="all good", exit_code=0)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_dotnet_format(Path("/repo"))
        assert issues == []

    async def test_dotnet_format_skips_non_cs_lines(self, executor, mock_client):
        stdout = "some random text\nno match here\nProgram.cs\n"
        mock_client.execute.return_value = _container_response(stdout=stdout, exit_code=1)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_dotnet_format(Path("/repo"))
        assert len(issues) == 1


# ===================================================================
# _run_roslynator
# ===================================================================


class TestRunRoslynator:
    async def test_roslynator_parses_output(self, executor, mock_client):
        stdout = (
            "Program.cs(10,5): warning RCS1001: Add braces\n"
            "Helper.cs(20,1): error RCS2002: Remove empty statement\n"
            "Util.cs(5,3): info RCS3003: Simplify expression\n"
        )
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_roslynator(Path("/repo"))

        assert len(issues) == 3
        assert issues[0].severity == "medium"  # warning
        assert issues[0].rule_id == "RCS1001"
        assert issues[0].file_path == "Program.cs"
        assert issues[0].line_number == 10
        assert issues[0].column_number == 5
        assert issues[0].tool == "roslynator"

        assert issues[1].severity == "high"  # error
        assert issues[1].rule_id == "RCS2002"

        assert issues[2].severity == "low"  # info

    async def test_roslynator_no_severity_prefix(self, executor, mock_client):
        stdout = "File.cs(1,1): some message without severity prefix\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_roslynator(Path("/repo"))
        assert len(issues) == 1
        assert issues[0].severity == "medium"  # default
        assert issues[0].rule_id == "roslynator-issue"

    async def test_roslynator_invalid_line_col(self, executor, mock_client):
        stdout = "File.cs(abc,def): warning RCS1: msg\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_roslynator(Path("/repo"))
        assert issues[0].line_number == 1
        assert issues[0].column_number is None

    async def test_roslynator_single_loc(self, executor, mock_client):
        stdout = "File.cs(10): warning RCS1: msg\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_roslynator(Path("/repo"))
        assert issues[0].line_number == 10
        assert issues[0].column_number is None

    async def test_roslynator_empty_stdout(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_roslynator(Path("/repo"))
        assert issues == []

    async def test_roslynator_skips_unparseable_lines(self, executor, mock_client):
        stdout = "some random noise\n"
        mock_client.execute.return_value = _container_response(stdout=stdout)
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_roslynator(Path("/repo"))
        assert issues == []


# ===================================================================
# _run_semgrep edge cases
# ===================================================================


class TestRunSemgrepEdgeCases:
    async def test_semgrep_warning_severity(self, executor, mock_client):
        data = {
            "results": [
                {
                    "check_id": "rule1",
                    "path": "app.py",
                    "start": {"line": 1, "col": 1},
                    "extra": {"severity": "WARNING", "message": "warn msg"},
                }
            ]
        }
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_semgrep(Path("/repo"), "python")
        assert issues[0].severity == "medium"

    async def test_semgrep_info_severity(self, executor, mock_client):
        data = {
            "results": [
                {
                    "check_id": "rule1",
                    "path": "app.py",
                    "start": {"line": 1, "col": 1},
                    "extra": {"severity": "INFO", "message": "info msg"},
                }
            ]
        }
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_semgrep(Path("/repo"), "python")
        assert issues[0].severity == "low"

    async def test_semgrep_unknown_severity(self, executor, mock_client):
        data = {
            "results": [
                {
                    "check_id": "r",
                    "path": "a.py",
                    "start": {"line": 1},
                    "extra": {"severity": "UNKNOWN", "message": "m"},
                }
            ]
        }
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_semgrep(Path("/repo"), "python")
        assert issues[0].severity == "medium"  # default

    async def test_semgrep_invalid_json(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="bad")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_semgrep(Path("/repo"), "python")
        assert issues == []

    async def test_semgrep_empty_results(self, executor, mock_client):
        data = {"results": []}
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_semgrep(Path("/repo"), "python")
        assert issues == []


# ===================================================================
# run_tool dispatcher — additional tools
# ===================================================================


class TestRunToolAdditionalDispatchers:
    @pytest.mark.parametrize(
        ("tool", "method"), [("ruff-format", "_run_ruff_format"), ("ty", "_run_ty")]
    )
    async def test_dispatches_astral_tools(self, executor, tool, method):
        with patch.object(
            executor,
            method,
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool(tool, [], Path("/repo"), "python")
        m.assert_awaited_once_with(Path("/repo"))

    @pytest.mark.parametrize("tool", ["mypy", "pylint", "black", "flake8", "isort", "autoflake"])
    async def test_removed_tool_names_are_unknown(self, executor, tool):
        issues, result = await executor.run_tool(tool, [], Path("/repo"), "python")
        assert issues == []
        assert result.get("success") is not True

    async def test_dispatches_prettier(self, executor):
        with patch.object(
            executor,
            "_run_prettier",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool("prettier", [], Path("/repo"), "javascript")
        m.assert_awaited_once_with(Path("/repo"), "javascript")

    async def test_dispatches_jshint(self, executor):
        with patch.object(
            executor,
            "_run_jshint",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool("jshint", [], Path("/repo"), "javascript")
        m.assert_awaited_once_with(Path("/repo"))

    async def test_dispatches_pmd(self, executor):
        with patch.object(
            executor,
            "_run_pmd",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool("pmd", [], Path("/repo"), "java")
        m.assert_awaited_once_with(Path("/repo"))

    async def test_dispatches_spotbugs(self, executor):
        with patch.object(
            executor,
            "_run_spotbugs",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool("spotbugs", [], Path("/repo"), "java")
        m.assert_awaited_once_with(Path("/repo"))

    async def test_dispatches_go_vet(self, executor):
        with patch.object(
            executor,
            "_run_go_vet",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool("go-vet", [], Path("/repo"), "go")
        m.assert_awaited_once_with(Path("/repo"))

    async def test_dispatches_golangci_lint(self, executor):
        with patch.object(
            executor,
            "_run_golangci_lint",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool("golangci-lint", [], Path("/repo"), "go")
        m.assert_awaited_once_with(Path("/repo"))

    async def test_dispatches_gosec(self, executor):
        with patch.object(
            executor,
            "_run_gosec",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool("gosec", [], Path("/repo"), "go")
        m.assert_awaited_once_with(Path("/repo"))

    async def test_dispatches_rustfmt(self, executor):
        with patch.object(
            executor,
            "_run_rustfmt",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool("rustfmt", [], Path("/repo"), "rust")
        m.assert_awaited_once_with(Path("/repo"))

    async def test_dispatches_cargo_audit(self, executor):
        with patch.object(
            executor,
            "_run_cargo_audit",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool("cargo-audit", [], Path("/repo"), "rust")
        m.assert_awaited_once_with(Path("/repo"))

    async def test_dispatches_clang_tidy(self, executor):
        with patch.object(
            executor,
            "_run_clang_tidy",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool("clang-tidy", [], Path("/repo"), "cpp")
        m.assert_awaited_once_with(Path("/repo"))

    async def test_dispatches_cpplint(self, executor):
        with patch.object(
            executor,
            "_run_cpplint",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool("cpplint", [], Path("/repo"), "cpp")
        m.assert_awaited_once_with(Path("/repo"))

    async def test_dispatches_roslynator(self, executor):
        with patch.object(
            executor,
            "_run_roslynator",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool("roslynator", [], Path("/repo"), "csharp")
        m.assert_awaited_once_with(Path("/repo"))

    async def test_value_error_caught(self, executor):
        with patch.object(executor, "_run_ruff", side_effect=ValueError("bad value")):
            issues, result = await executor.run_tool("ruff", [], Path("/repo"), "python")
        assert result["error"] == "bad value"

    async def test_dispatches_golint_from_run_tool(self, executor):
        with patch.object(
            executor,
            "_run_golint",
            new_callable=AsyncMock,
            return_value=([], {"success": True}),
        ) as m:
            await executor.run_tool("golint", [], Path("/repo"), "go")
        m.assert_awaited_once()


# ===================================================================
# _run_golint — additional edge cases
# ===================================================================


class TestRunGolintEdgeCases:
    async def test_golint_empty_stdout(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_golint(Path("/repo"))
        assert issues == []

    async def test_golint_not_success(self, executor, mock_client):
        mock_client.execute.return_value = {
            "exit_code": -1,
            "stderr": "fail",
        }
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor._run_golint(Path("/repo"))
        assert issues == []


# ===================================================================
# _run_eslint — additional edge cases
# ===================================================================


class TestRunEslintEdgeCases:
    async def test_eslint_missing_rule_id(self, executor, mock_client):
        data = [
            {
                "filePath": "a.js",
                "messages": [
                    {
                        "severity": 2,
                        "ruleId": None,
                        "message": "Parse error",
                        "line": 1,
                    }
                ],
            }
        ]
        mock_client.execute.return_value = _container_response(stdout=json.dumps(data))
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_eslint(Path("/repo"), "javascript")
        assert issues[0].rule_id == "eslint-error"

    async def test_eslint_invalid_json(self, executor, mock_client):
        mock_client.execute.return_value = _container_response(stdout="bad")
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, _ = await executor._run_eslint(Path("/repo"), "javascript")
        assert issues == []


# ===================================================================
# Tenant threading into the container exec
# ===================================================================

TENANT = "0195f48b-d000-7000-8000-000000000003"


class TestTenantThreadedIntoContainerExec:
    """The exec must be made as the tenant that owns the workspace.

    With no ``X-Tenant-Id``, coding-containers resolves the caller as
    ``svc-coding-analysis``'s own service account and its BOLA guard answers
    404 "Container '<name>' not found" in ~1.2 ms — so all six Python tools
    "failed" in 25 ms total and the report came back inconclusive.
    """

    async def test_run_tool_execs_as_the_tenant_that_owns_the_container(self, mock_client):
        executor = ToolExecution()
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            await executor.run_tool(
                "ruff",
                [],
                Path("/workspace"),
                "python",
                container_name="ws-1",
                tenant_id=TENANT,
            )
        assert mock_client.execute.await_args.kwargs.get("tenant_id") == TENANT, (
            "the container exec carried NO tenant: coding-containers resolves it "
            "as the service account and its BOLA guard 404s the customer's workspace"
        )

    async def test_absent_tenant_stays_absent(self, mock_client):
        """A missing tenant is never defaulted to a literal."""
        executor = ToolExecution()
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            await executor.run_tool("ruff", [], Path("/workspace"), "python", container_name="ws-1")
        assert mock_client.execute.await_args.kwargs.get("tenant_id") is None

    async def test_switching_container_does_not_carry_the_previous_tenant(self, mock_client):
        """Container and tenant are rebound together, so no tenant outlives its run.

        The executor instance is reused across every tool in a run; a tenant
        left behind from an earlier container would assert tenant A's identity
        against tenant B's workspace.
        """
        executor = ToolExecution()
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            await executor.run_tool(
                "ruff", [], Path("/workspace"), "python", container_name="ws-1", tenant_id=TENANT
            )
            await executor.run_tool("ruff", [], Path("/workspace"), "python", container_name="ws-2")
        assert mock_client.execute.await_args.kwargs.get("tenant_id") is None, (
            "the previous request's tenant was asserted against a different container"
        )

    async def test_a_failed_exec_is_still_reported_as_a_failure(self, mock_client):
        """Threading the tenant must not launder a real error into a pass.

        A 404/-1 exec still yields success=False with the stderr preserved, so
        the analyzer keeps abstaining (inconclusive) rather than scoring a
        vacuous 100/A on zero measurements.
        """
        mock_client.execute.return_value = _container_response(
            exit_code=-1, stderr="Container 'ws-1' not found"
        )
        executor = ToolExecution()
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            issues, result = await executor.run_tool(
                "ruff",
                [],
                Path("/workspace"),
                "python",
                container_name="ws-1",
                tenant_id=TENANT,
            )
        assert result["success"] is False, "a failed exec was reported as a success"
        assert "not found" in result["error"]
        assert issues == []

    async def test_tool_not_executable_in_container_is_a_failure(self, mock_client):
        """127 (command not found) is 'the tool did not run', not a clean pass."""
        mock_client.execute.return_value = _container_response(
            exit_code=127, stderr="ruff: not found"
        )
        executor = ToolExecution()
        with patch(
            "processing.architecture.tool_execution._base.get_coding_containers_client",
            return_value=mock_client,
        ):
            _, result = await executor.run_tool(
                "ruff",
                [],
                Path("/workspace"),
                "python",
                container_name="ws-1",
                tenant_id=TENANT,
            )
        assert result["success"] is False
