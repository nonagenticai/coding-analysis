"""ty / ruff product runners: real rule codes, and a crash is a FAILURE.

The quality runners execute against USER code. A finding's identity downstream
is keyed on ``tool + rule_id + file + unit``, so each finding must carry the
tool's own rule code (ty's ``check_name``, ruff's ``code``). And a run that did
not measure the code (exit 2: bad config / argv / missing target, a panic,
unparseable output) must come back ``success=False`` with no findings -- never
as a clean, zero-finding run (the invariant mypy got in PR #102).

Fixtures in ``tests/fixtures/{ty,ruff}/`` are REAL output of ty 0.0.92 and
ruff 0.17.3 (the pinned versions), captured by
``tests/fixtures/capture_astral_runs.py.txt`` from temp dirs holding only the
recorded ``files``, with the runner's own argv.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from processing.architecture.tool_execution import ToolExecution
from processing.quality_analysis.quality_runner import QualityToolRunner
from processing.quality_analysis.tool_definitions import QualityTool, ToolCategory
from processing.utils.ruff_output import parse_ruff_concise, parse_ruff_json, ruff_failure
from processing.utils.ty_output import parse_ty_gitlab, ty_failure
from services.quality.tool_integration import ToolExecutor as ServiceToolExecutor

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
TY = json.loads((FIXTURES / "ty" / "ty_runs.json").read_text())
RUFF = json.loads((FIXTURES / "ruff" / "ruff_runs.json").read_text())

# What the captured ty run reports, by (line, rule).
TY_EXPECTED = [
    (2, "unresolved-import"),
    (6, "invalid-return-type"),
    (9, "invalid-argument-type"),
    (10, "unresolved-attribute"),
]
RUFF_EXPECTED = [(1, "I001"), (1, "F401"), (5, "F841")]
TY_CRASHES = ["bad_config", "missing_path", "invalid_flag"]
PATCH_CLIENT = "processing.architecture.tool_execution._base.get_coding_containers_client"


def _container(run: dict) -> dict:
    return {
        "exit_code": run["exit_code"],
        "stdout": run["stdout"],
        "stderr": run["stderr"],
        "duration": 0.1,
    }


def _client(run: dict) -> AsyncMock:
    client = AsyncMock()
    client.execute = AsyncMock(return_value=_container(run))
    return client


@pytest.fixture()
def executor():
    return ToolExecution(container_name="test-worker")


# --------------------------------------------------------------------------- fixtures


def test_ty_fixtures_are_the_real_captured_runs():
    assert {k: v["exit_code"] for k, v in TY.items() if k != "_about"} == {
        "diagnostics": 1,
        "clean": 0,
        "syntax_error": 1,
        "bad_config": 2,
        "missing_path": 2,
        "invalid_flag": 2,
    }
    assert "ty 0.0.92" in TY["_about"]
    assert "is not a valid `pyproject.toml`" in TY["bad_config"]["stderr"]
    assert "unexpected argument '--no-such-flag'" in TY["invalid_flag"]["stderr"]
    # ty exit 2 on a missing target still prints a JSON row -- not a finding.
    assert [r["check_name"] for r in json.loads(TY["missing_path"]["stdout"])] == ["io"]


def test_ruff_fixtures_are_the_real_captured_runs():
    assert {k: v["exit_code"] for k, v in RUFF.items() if k != "_about"} == {
        "check_diagnostics": 1,
        "check_clean": 0,
        "check_bad_config": 2,
        "format_unformatted": 1,
        "format_clean": 0,
        "format_syntax_error": 2,
        "concise_check_diagnostics": 1,
        "concise_check_clean": 0,
        "concise_check_missing": 1,
        "concise_check_bad_config": 2,
        "concise_format_unformatted": 1,
        "concise_format_clean": 0,
        "concise_format_syntax_error": 2,
        "concise_format_missing": 2,
    }
    assert "ruff 0.17.3" in RUFF["_about"]


# The container-exec path parses CONCISE output (JSON overflows the 1 MiB cap).
CONCISE_EXPECTED = [
    ("broken.py", 2, "invalid-syntax", False),
    ("sample.py", 1, "I001", True),
    ("sample.py", 1, "F401", True),
    ("sample.py", 5, "F841", False),
]


def test_parse_ruff_concise_real_check_output():
    diags = parse_ruff_concise(RUFF["concise_check_diagnostics"]["stdout"])
    assert [(d.file, d.line, d.code, d.fixable) for d in diags] == CONCISE_EXPECTED
    assert diags[2].message == "`sys` imported but unused"
    assert diags[2].column == 8
    # Summary lines ("Found 4 errors.", "[*] 2 fixable ...") are not rows.
    assert parse_ruff_concise(RUFF["concise_check_clean"]["stdout"]) == []


def test_parse_ruff_concise_real_format_output():
    diags = parse_ruff_concise(RUFF["concise_format_unformatted"]["stdout"])
    assert [(d.file, d.line, d.code) for d in diags] == [("sample.py", 3, "unformatted")]
    assert parse_ruff_concise(RUFF["concise_format_clean"]["stdout"]) == []


def test_parse_ruff_concise_path_with_spaces_and_colons_in_message():
    diags = parse_ruff_concise("sp ace/a b.py:1:8: F401 [*] `os` imported: but unused\n")
    assert [(d.file, d.line, d.column, d.code, d.message) for d in diags] == [
        ("sp ace/a b.py", 1, 8, "F401", "`os` imported: but unused")
    ]


# --------------------------------------------------------------------------- parsers


def test_parse_ty_gitlab_carries_ty_rule_names():
    diags = parse_ty_gitlab(TY["diagnostics"]["stdout"])
    assert diags is not None
    assert [(d.line, d.rule) for d in diags] == TY_EXPECTED
    first = diags[0]
    assert first.file == "sample.py"
    assert first.column == 8
    assert first.severity == "high"  # gitlab "major" = ty error
    # The "<rule>: " prefix of description is stripped from the message.
    assert first.message == "Cannot resolve imported module `nonexistent_mod`"
    assert ty_failure(1, diags) is None


def test_parse_ty_gitlab_clean_is_empty_not_none():
    assert parse_ty_gitlab(TY["clean"]["stdout"]) == []
    assert ty_failure(0, []) is None


def test_parse_ty_gitlab_unparseable_is_none():
    assert parse_ty_gitlab("") is None
    assert parse_ty_gitlab("ty failed") is None
    assert parse_ty_gitlab('{"not": "a list"}') is None


@pytest.mark.parametrize("name", TY_CRASHES)
def test_ty_failure_on_every_exit_2(name):
    run = TY[name]
    reason = ty_failure(
        run["exit_code"], parse_ty_gitlab(run["stdout"]), run["stdout"], run["stderr"]
    )
    assert reason is not None
    assert reason.startswith("ty exited 2")


@pytest.mark.parametrize("exit_code", [101, 137, -9])
def test_ty_failure_on_panic_or_kill_even_with_rows(exit_code):
    # Synthetic exit codes over the real diagnostics: 101 = Rust panic,
    # 137 = SIGKILL/OOM. A run killed mid-check is unfinished.
    stdout = TY["diagnostics"]["stdout"]
    assert ty_failure(exit_code, parse_ty_gitlab(stdout), stdout) is not None


def test_ty_exit_1_with_nothing_parseable_fails():
    assert ty_failure(1, None, "garbage").startswith("ty exited 1")
    assert ty_failure(1, []) is not None


def test_parse_ruff_json_carries_ruff_codes():
    diags = parse_ruff_json(RUFF["check_diagnostics"]["stdout"])
    assert diags is not None
    assert [(d.line, d.code) for d in diags] == RUFF_EXPECTED
    assert diags[0].file == "/repo/sample.py"
    assert diags[0].fixable is True
    assert diags[0].url == "https://docs.astral.sh/ruff/rules/unsorted-imports"
    assert ruff_failure(1, diags) is None


def test_ruff_bad_config_is_a_failure():
    run = RUFF["check_bad_config"]
    reason = ruff_failure(
        run["exit_code"], parse_ruff_json(run["stdout"]), run["stdout"], run["stderr"]
    )
    assert reason is not None and "Failed to parse" in reason


def test_ruff_format_exit_2_with_rows_is_a_failure():
    run = RUFF["format_syntax_error"]
    diags = parse_ruff_json(run["stdout"])
    assert diags and diags[0].code == "invalid-syntax"
    assert ruff_failure(run["exit_code"], diags, run["stdout"], tool="ruff format") is not None


# --------------------------------------------------------------------------- container runners


@patch(PATCH_CLIENT)
async def test_run_ty_findings_carry_rule_codes(mock_get_client, executor):
    mock_get_client.return_value = _client(TY["diagnostics"])

    issues, result = await executor._run_ty(Path("/repo"))

    assert result["success"] is True
    assert [(i.line_number, i.rule_id) for i in issues] == TY_EXPECTED
    assert {i.tool for i in issues} == {"ty"}
    assert {i.category for i in issues} == {"type-safety"}
    assert all(i.file_path == "sample.py" for i in issues)
    assert issues[1].external_url.endswith("#invalid-return-type")
    argv = mock_get_client.return_value.execute.call_args.kwargs["command"]
    assert argv == ["ty", "check", "--output-format", "gitlab", "/repo"]


@patch(PATCH_CLIENT)
async def test_run_ty_syntax_error_is_a_finding(mock_get_client, executor):
    """ty reports a syntax error as an invalid-syntax diagnostic, exit 1."""
    mock_get_client.return_value = _client(TY["syntax_error"])

    issues, result = await executor._run_ty(Path("/repo"))

    assert result["success"] is True
    assert {i.rule_id for i in issues} == {"invalid-syntax"}


@pytest.mark.parametrize("name", TY_CRASHES)
@patch(PATCH_CLIENT)
async def test_run_ty_crash_is_a_tool_failure_not_a_clean_run(mock_get_client, executor, name):
    mock_get_client.return_value = _client(TY[name])

    issues, result = await executor._run_ty(Path("/repo"))

    # success=False is what the analyzer turns into tools_failed=["ty"].
    assert result["success"] is False
    assert result["exit_code"] == 2
    assert result["error"].startswith("ty exited 2")
    assert issues == []


@patch(PATCH_CLIENT)
async def test_run_ty_failure_carries_tys_own_words(mock_get_client, executor):
    mock_get_client.return_value = _client(TY["bad_config"])

    _, result = await executor._run_ty(Path("/repo"))

    assert "is not a valid `pyproject.toml`" in result["error"]


@patch(PATCH_CLIENT)
async def test_run_ty_clean_exit_0_stays_clean(mock_get_client, executor):
    mock_get_client.return_value = _client(TY["clean"])

    issues, result = await executor._run_ty(Path("/repo"))

    assert result["success"] is True
    assert "error" not in result
    assert issues == []


@patch(PATCH_CLIENT)
async def test_run_tool_dispatch_keeps_the_ty_failure(mock_get_client, executor):
    mock_get_client.return_value = _client(TY["invalid_flag"])

    issues, result = await executor.run_tool(
        "ty", [], Path("/repo"), "python", container_name="c1", tenant_id="t1"
    )

    assert result["success"] is False
    assert issues == []


@patch(PATCH_CLIENT)
async def test_run_ruff_findings_carry_ruff_codes(mock_get_client, executor):
    mock_get_client.return_value = _client(RUFF["concise_check_diagnostics"])

    issues, result = await executor._run_ruff(Path("/repo"))

    assert result["success"] is True
    assert [(i.file_path, i.line_number, i.rule_id) for i in issues] == [
        (f, ln, c) for f, ln, c, _ in CONCISE_EXPECTED
    ]
    assert {i.tool for i in issues} == {"ruff"}
    assert issues[0].severity == "critical"  # invalid-syntax
    assert issues[2].severity == "critical"  # F401
    assert issues[2].resolve_suggestions == "Auto-fixable: ruff check --fix --select F401"
    assert issues[3].resolve_suggestions is None  # F841: no [*] marker
    argv = mock_get_client.return_value.execute.call_args.kwargs["command"]
    assert argv == [
        "ruff",
        "check",
        "/repo",
        "--output-format=concise",
        "--extend-select",
        "I,PL",
    ]


@patch(PATCH_CLIENT)
async def test_run_ruff_bad_config_is_a_tool_failure(mock_get_client, executor):
    mock_get_client.return_value = _client(RUFF["concise_check_bad_config"])

    issues, result = await executor._run_ruff(Path("/repo"))

    assert result["success"] is False
    assert result["error"].startswith("ruff exited 2")
    assert issues == []


@patch(PATCH_CLIENT)
async def test_run_ruff_format_reports_unformatted_files(mock_get_client, executor):
    mock_get_client.return_value = _client(RUFF["concise_format_unformatted"])

    issues, result = await executor.run_tool("ruff-format", [], Path("/repo"), "python")

    assert result["success"] is True
    assert [(i.file_path, i.rule_id, i.tool) for i in issues] == [
        ("sample.py", "unformatted", "ruff-format")
    ]
    argv = mock_get_client.return_value.execute.call_args.kwargs["command"]
    assert argv == ["ruff", "format", "--check", "--output-format=concise", "/repo"]


@patch(PATCH_CLIENT)
async def test_run_ruff_format_syntax_error_is_a_tool_failure(mock_get_client, executor):
    mock_get_client.return_value = _client(RUFF["concise_format_syntax_error"])

    issues, result = await executor._run_ruff_format(Path("/repo"))

    assert result["success"] is False
    assert issues == []


@pytest.mark.parametrize(
    ("tool", "fixture"),
    [("ruff", "concise_check_missing"), ("ruff-format", "concise_format_missing")],
)
@patch(PATCH_CLIENT)
async def test_scoped_ruff_missing_target_is_not_analysed(mock_get_client, executor, tool, fixture):
    mock_get_client.return_value = _client(RUFF[fixture])

    issues, result = await executor.run_tool(
        tool, [], Path("/repo"), "python", scope_paths=["nope.py"]
    )

    assert result["success"] is False
    assert "E902/io" in result["error"]
    assert issues == []


# --------------------------------------------------------------------------- truncated exec output

ONE_MIB = 1024 * 1024


def _truncated_resp(stdout: str, exit_code: int = 1) -> dict:
    # Shape of coding-containers' exec response when a stream hit its 1 MiB cap.
    return {
        "exit_code": exit_code,
        "stdout": stdout[:ONE_MIB],
        "stderr": "",
        "duration": 0.1,
        "truncated": True,
        "bytes_omitted": max(len(stdout) - ONE_MIB, 0),
    }


@pytest.mark.parametrize("tool", ["ruff", "ruff-format", "ty", "bandit", "radon"])
@patch(PATCH_CLIENT)
async def test_truncated_exec_output_is_a_tool_failure(mock_get_client, executor, tool):
    """Partial output must never be parsed as a complete report."""
    # >1 MiB of perfectly parseable concise rows: without the truncated check
    # the first 1 MiB would come back as a "complete" success.
    row = "pkg/module.py:1:1: E501 Line too long (120 > 88)\n"
    big = row * (ONE_MIB // len(row) + 500)
    assert len(big) > ONE_MIB
    client = AsyncMock()
    client.execute = AsyncMock(return_value=_truncated_resp(big))
    mock_get_client.return_value = client

    issues, result = await executor.run_tool(tool, [], Path("/repo"), "python")

    assert result["success"] is False
    assert "output truncated" in result["error"]
    assert result["truncated"] is True
    assert issues == []


@patch(PATCH_CLIENT)
async def test_untruncated_flag_false_is_a_normal_run(mock_get_client, executor):
    resp = _container(RUFF["concise_check_diagnostics"]) | {"truncated": False, "bytes_omitted": 0}
    client = AsyncMock()
    client.execute = AsyncMock(return_value=resp)
    mock_get_client.return_value = client

    issues, result = await executor._run_ruff(Path("/repo"))

    assert result["success"] is True
    assert len(issues) == len(CONCISE_EXPECTED)


@patch(PATCH_CLIENT)
async def test_scoped_ty_missing_target_is_not_analysed(mock_get_client, executor):
    """A path-scoped ty run whose target is missing is caught before parsing."""
    mock_get_client.return_value = _client(TY["missing_path"])

    issues, result = await executor.run_tool(
        "ty", [], Path("/repo"), "python", scope_paths=["does_not_exist"]
    )

    assert result["success"] is False
    assert "ty io" in result["error"]
    assert issues == []


# --------------------------------------------------------------------------- QualityToolRunner


def _qa_tool(name: str, command: list[str]) -> QualityTool:
    return QualityTool(
        name=name,
        category=ToolCategory.TYPE_CHECKER if name == "ty" else ToolCategory.LINTER,
        command=command,
        output_format="json",
        description=name,
        timeout=30,
    )


async def _qa_run(tool: QualityTool, run: dict) -> dict:
    docker = MagicMock()
    docker.exec_in_container = AsyncMock(return_value=_container(run))
    return await QualityToolRunner(docker)._execute_single_tool("cid", tool, "sample.py")


async def test_quality_runner_ty_reports_rule_codes():
    out = await _qa_run(
        _qa_tool("ty", ["ty", "check", "--output-format", "gitlab"]), TY["diagnostics"]
    )

    assert out["success"] is True
    assert [(i.line_number, i.rule_id) for i in out["issues"]] == TY_EXPECTED
    assert {i.severity for i in out["issues"]} == {"error"}


@pytest.mark.parametrize("name", TY_CRASHES)
async def test_quality_runner_ty_crash_is_a_failure(name):
    out = await _qa_run(_qa_tool("ty", ["ty", "check", "--output-format", "gitlab"]), TY[name])

    assert out["success"] is False
    assert out["issues"] == []
    assert out["error"].startswith("ty exited 2")


async def test_quality_runner_ruff_reports_codes():
    out = await _qa_run(
        _qa_tool("ruff", ["ruff", "check", "--output-format", "json"]), RUFF["check_diagnostics"]
    )

    assert out["success"] is True
    assert [(i.line_number, i.rule_id) for i in out["issues"]] == RUFF_EXPECTED
    assert out["issues"][1].severity == "error"  # F401


# --------------------------------------------------------------------------- services ToolExecutor


def test_service_executor_ty_parse_and_success():
    ex = ServiceToolExecutor()
    run = TY["diagnostics"]
    issues = ex._parse_ty(run["stdout"])
    assert [(i.line, i.rule_id) for i in issues] == TY_EXPECTED
    assert ex._check_success("ty", run["exit_code"], run["stdout"], run["stderr"]) is True


@pytest.mark.parametrize("name", TY_CRASHES)
def test_service_executor_ty_crash_is_not_success(name):
    run = TY[name]
    assert (
        ServiceToolExecutor()._check_success("ty", run["exit_code"], run["stdout"], run["stderr"])
        is False
    )


def test_service_executor_ruff_parse():
    issues = ServiceToolExecutor()._parse_ruff("ruff", RUFF["check_diagnostics"]["stdout"])
    assert [(i.line, i.rule_id) for i in issues] == RUFF_EXPECTED
    assert issues[1].severity == "error"


@pytest.mark.parametrize("name", ["eslint", "ruff", "ty"])
async def test_quality_runner_truncated_output_is_a_failure(name):
    """Same rule on the QualityToolRunner exec path: partial is never complete."""
    run = {"exit_code": 0, "stdout": "[]", "stderr": ""}
    docker = MagicMock()
    docker.exec_in_container = AsyncMock(
        return_value=_container(run) | {"truncated": True, "bytes_omitted": 4096}
    )
    out = await QualityToolRunner(docker)._execute_single_tool(
        "cid", _qa_tool(name, [name]), "sample.py"
    )
    assert out["success"] is False
    assert "output truncated" in out["error"]
    assert "4096 bytes omitted" in out["error"]
    assert out["issues"] == []
