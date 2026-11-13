"""Regression tests for robust quality-tool output parsing.

A tool can emit valid JSON that is NOT the expected dict/list — e.g. a bare
string error message. Previously the parsers called ``.get()`` on it and raised
``'str' object has no attribute 'get'``, which aborted the *entire* quality
analysis. The parsers must now skip cleanly and return no issues.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from processing.architecture.tool_execution import ToolExecution
from processing.architecture.tool_execution._base import BaseToolExecution


@pytest.fixture
def executor():
    return ToolExecution(container_name="test-worker")


def _result(stdout: str) -> dict:
    return {"success": True, "stdout": stdout, "stderr": "", "exit_code": 0}


# Valid JSON values that are NOT the dict/list each parser expects.
MALFORMED = ['"some tool error message"', "42", "true", "null"]


@pytest.mark.parametrize("stdout", MALFORMED)
async def test_safe_json_rejects_unexpected_shapes(stdout):
    assert BaseToolExecution._safe_json(stdout, dict) is None
    assert BaseToolExecution._safe_json(stdout, list) is None
    assert BaseToolExecution._safe_json("not json at all", dict) is None


def test_safe_json_accepts_expected_shapes():
    assert BaseToolExecution._safe_json("{}", dict) == {}
    assert BaseToolExecution._safe_json("[]", list) == []
    assert BaseToolExecution._safe_json('{"a": 1}', (dict, list)) == {"a": 1}


@pytest.mark.parametrize("stdout", [*MALFORMED, "[]", "{}", '["x"]', '{"k": "v"}'])
async def test_parsers_do_not_crash_on_unexpected_shape(executor, monkeypatch, stdout):
    """Feed every JSON parser malformed-but-valid JSON via a stubbed container
    exec; none may raise, and none should fabricate issues."""
    monkeypatch.setattr(
        executor,
        "_execute_tool_in_container",
        AsyncMock(return_value=([], _result(stdout))),
    )
    repo = Path("/workspace")

    no_lang = [
        executor._run_ruff,
        executor._run_bandit,
        executor._run_radon,
        executor._run_ruff_format,
        executor._run_ty,
        executor._run_jshint,
        executor._run_pmd,
    ]
    for runner in no_lang:
        issues, _result_out = await runner(repo)
        assert issues == []

    for runner in (executor._run_eslint, executor._run_semgrep):
        issues, _result_out = await runner(repo, "python")
        assert issues == []
