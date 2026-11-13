"""Unit tests for BugHunter service.

Tests BugHunter with REAL internal services (ErrorParser, FixGenerator, BugAnalyzer)
instead of mocking them, so the tests actually prove the orchestration pipeline works.
Only GitHubIntegration and CodebaseScanner are mocked (they need external resources).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from models.bug_models import BugAnalysisResult, CodeFix, ErrorType, ParsedError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bug_hunter(tmp_path):
    """Create a BugHunter with real internal services.

    Only CodebaseScanner (needs filesystem) and GitHubIntegration (needs token)
    are mocked. ErrorParser, FixGenerator, and BugAnalyzer are REAL.

    """
    with patch("services.bug_hunter.CodebaseScanner") as mock_scanner_cls:
        mock_scanner = MagicMock()
        mock_scanner.is_deep_analysis_result = MagicMock(return_value=False)
        mock_scanner_cls.return_value = mock_scanner

        from services.bug_hunter import BugHunter

        hunter = BugHunter(str(tmp_path), github_token="fake-token")
        yield hunter


# ---------------------------------------------------------------------------
# analyze_error_logs: end-to-end with real parsing
# ---------------------------------------------------------------------------


def test_analyze_python_traceback_end_to_end(bug_hunter):
    """Full pipeline: real traceback -> ErrorParser -> FixGenerator -> BugAnalyzer."""
    traceback_log = (
        "Traceback (most recent call last):\n"
        '  File "app/main.py", line 3, in <module>\n'
        "    import requests\n"
        "ModuleNotFoundError: No module named 'requests'\n"
    )
    result = bug_hunter.analyze_error_logs(traceback_log)

    assert isinstance(result, BugAnalysisResult)
    # ErrorParser should have parsed exactly 1 error
    assert len(result.errors) == 1
    assert result.errors[0].error_type == ErrorType.IMPORT_ERROR
    assert result.errors[0].file_path == "app/main.py"
    assert result.errors[0].line_number == 3
    assert "requests" in result.errors[0].message

    # FixGenerator should have produced fixes for the import error
    assert len(result.resolves) >= 1
    req_fixes = [f for f in result.resolves if "requirements" in f.file_path]
    assert len(req_fixes) >= 1
    assert "requests" in req_fixes[0].fixed_code

    # BugAnalyzer should have produced a real root cause analysis
    assert "Root Cause Analysis" in result.root_cause_analysis
    assert "1" in result.root_cause_analysis  # total errors count
    assert result.estimated_effort != ""
    assert result.risk_assessment != ""


def test_analyze_key_error_traceback(bug_hunter):
    """KeyError traceback should produce .get() fix suggestions."""
    traceback_log = (
        "Traceback (most recent call last):\n"
        '  File "data.py", line 10, in process\n'
        "    val = d['missing']\n"
        "KeyError: 'missing'\n"
    )
    result = bug_hunter.analyze_error_logs(traceback_log)

    assert len(result.errors) == 1
    assert result.errors[0].error_type == ErrorType.KEY_ERROR
    assert result.errors[0].file_path == "data.py"
    assert result.errors[0].line_number == 10

    # FixGenerator should suggest .get()
    assert len(result.resolves) >= 1
    assert any(".get(" in f.fixed_code for f in result.resolves)


def test_analyze_attribute_error_suggests_correction(bug_hunter):
    """AttributeError with typo should suggest corrected attribute name."""
    traceback_log = (
        "Traceback (most recent call last):\n"
        '  File "app.py", line 7, in run\n'
        "    s.spllit(',')\n"
        "AttributeError: 'str' object has no attribute 'spllit'\n"
    )
    result = bug_hunter.analyze_error_logs(traceback_log)

    assert len(result.errors) == 1
    assert result.errors[0].error_type == ErrorType.ATTRIBUTE_ERROR

    # FixGenerator should suggest 'split' as correction
    assert len(result.resolves) >= 1
    assert any("split" in f.fixed_code for f in result.resolves)


def test_analyze_multiple_tracebacks(bug_hunter):
    """Multiple tracebacks should all be parsed and have fixes generated."""
    logs = (
        "Traceback (most recent call last):\n"
        '  File "a.py", line 1, in f\n'
        "    int('abc')\n"
        "ValueError: invalid literal for int() with base 10: 'abc'\n"
        "\n"
        "Traceback (most recent call last):\n"
        '  File "b.py", line 2, in g\n'
        "    return 1 + 'two'\n"
        "TypeError: unsupported operand type(s) for +: 'int' and 'str'\n"
    )
    result = bug_hunter.analyze_error_logs(logs)

    assert len(result.errors) == 2
    error_types = {e.error_type for e in result.errors}
    assert ErrorType.VALUE_ERROR in error_types
    assert ErrorType.TYPE_ERROR in error_types

    # Each error should have at least one fix
    assert len(result.resolves) >= 2

    # Root cause analysis should mention both error types
    assert "Root Cause Analysis" in result.root_cause_analysis
    assert "2" in result.root_cause_analysis  # total errors


def test_analyze_name_error_suggests_import(bug_hunter):
    """NameError for a known module should suggest an import statement."""
    traceback_log = (
        "Traceback (most recent call last):\n"
        '  File "script.py", line 5, in main\n'
        "    path = os.path.join('/tmp', name)\n"
        "NameError: name 'os' is not defined\n"
    )
    result = bug_hunter.analyze_error_logs(traceback_log)

    assert len(result.errors) == 1
    assert result.errors[0].error_type == ErrorType.NAME_ERROR

    # Should suggest 'import os'
    import_fixes = [f for f in result.resolves if "import os" in f.fixed_code]
    assert len(import_fixes) >= 1


def test_analyze_syntax_error_eof(bug_hunter):
    """SyntaxError with EOF should suggest checking unclosed brackets."""
    traceback_log = (
        "Traceback (most recent call last):\n"
        '  File "bad.py", line 1\n'
        "    def foo(\n"
        "          ^\n"
        "SyntaxError: unexpected EOF while parsing\n"
    )
    result = bug_hunter.analyze_error_logs(traceback_log)

    assert len(result.errors) == 1
    assert result.errors[0].error_type == ErrorType.SYNTAX_ERROR

    # Fix should mention unclosed brackets
    assert len(result.resolves) >= 1
    assert any(
        "bracket" in f.explanation.lower() or "unclosed" in f.explanation.lower()
        for f in result.resolves
    )


# ---------------------------------------------------------------------------
# analyze_error_logs: empty/whitespace input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("logs", ["", "   \n\t\n  "])
def test_analyze_without_logs_refuses_instead_of_scanning_this_pod(bug_hunter, logs):
    """No logs = a codebase scan, which never runs in this pod (no checkout).

    It runs inside the workspace (services.workspace_hunt). Raising here is what
    keeps a stray local call from ever reporting a clean repository again.
    """
    with pytest.raises(ValueError, match="runs in the workspace"):
        bug_hunter.analyze_error_logs(logs)


# ---------------------------------------------------------------------------
# analyze_error_logs: deep analysis input
# ---------------------------------------------------------------------------


def test_analyze_deep_analysis_json(bug_hunter):
    """Deep analysis JSON input should be routed to parse_deep_analysis_results."""
    import json

    deep_json = json.dumps({"issues": [{"type": "syntax", "message": "bad syntax"}]})
    error = ParsedError(
        error_type=ErrorType.SYNTAX_ERROR,
        message="bad syntax",
        file_path="test.py",
        line_number=1,
        function_name=None,
        stack_trace=[],
        confidence=0.9,
    )
    bug_hunter.codebase_scanner.is_deep_analysis_result = MagicMock(return_value=True)
    bug_hunter.codebase_scanner.parse_deep_analysis_results = MagicMock(return_value=[error])

    result = bug_hunter.analyze_error_logs(deep_json)

    bug_hunter.codebase_scanner.is_deep_analysis_result.assert_called_once_with(deep_json)
    bug_hunter.codebase_scanner.parse_deep_analysis_results.assert_called_once_with(deep_json)
    assert isinstance(result, BugAnalysisResult)
    assert len(result.errors) == 1
    assert result.errors[0].error_type == ErrorType.SYNTAX_ERROR
    # Real FixGenerator should produce a fix for the syntax error
    assert len(result.resolves) >= 1


# ---------------------------------------------------------------------------
# analyze_error_logs: error handling
# ---------------------------------------------------------------------------


def test_analyze_error_logs_parser_failure_propagates(bug_hunter):
    """A ValueError in error_parser must PROPAGATE, not become an empty result.

    Regression: this method used to swallow the exception and return
    ``BugAnalysisResult(errors=[], root_cause_analysis="Analysis failed: …")``.
    Because it returned instead of raising, ``_run_claimed_hunt`` took its
    success path and persisted the job as ``completed / errors_found=0 /
    error_message=NULL`` — identical to a clean repository.
    """
    bug_hunter.error_parser.parse_error_logs = MagicMock(side_effect=ValueError("parse failed"))

    with pytest.raises(ValueError, match="parse failed"):
        bug_hunter.analyze_error_logs("some error log content")


# ---------------------------------------------------------------------------
# create_pull_request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_pull_request_delegates_to_github(bug_hunter):
    """create_pull_request should delegate to GitHubIntegration."""
    fix = CodeFix(
        file_path="test.py",
        line_number=1,
        original_code="x",
        fixed_code="y",
        explanation="fix issue",
        confidence=0.8,
    )
    expected_pr = {"number": 42, "url": "https://github.com/owner/repo/pull/42"}
    bug_hunter.github_integration.create_pull_request = MagicMock(return_value=expected_pr)

    result = await bug_hunter.create_pull_request(
        repo_owner="owner",
        repo_name="repo",
        resolves=[fix],
        branch_name="fix-bugs",
    )

    bug_hunter.github_integration.create_pull_request.assert_called_once()
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_create_pull_request_empty_resolves(bug_hunter):
    """create_pull_request with empty resolves should still work."""
    expected_pr = {"success": True}
    bug_hunter.github_integration.create_pull_request = MagicMock(return_value=expected_pr)

    result = await bug_hunter.create_pull_request(
        repo_owner="owner",
        repo_name="repo",
        resolves=[],
    )

    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_create_pull_request_failure(bug_hunter):
    """create_pull_request with OSError should return error dict."""
    bug_hunter.github_integration.create_pull_request = MagicMock(
        side_effect=OSError("network error")
    )

    result = await bug_hunter.create_pull_request(
        repo_owner="owner",
        repo_name="repo",
        resolves=[
            CodeFix(
                file_path="test.py",
                line_number=1,
                original_code="x",
                fixed_code="y",
                explanation="fix",
                confidence=0.8,
            )
        ],
    )

    assert result["success"] is False
    assert "network error" in result["error"]


# ---------------------------------------------------------------------------
# get_analysis_summary: uses real BugAnalysisResult data
# ---------------------------------------------------------------------------


def test_get_analysis_summary_with_real_pipeline_output(bug_hunter):
    """get_analysis_summary should work on real pipeline output."""
    traceback_log = (
        "Traceback (most recent call last):\n"
        '  File "data.py", line 10, in process\n'
        "    val = d['missing']\n"
        "KeyError: 'missing'\n"
    )
    analysis = bug_hunter.analyze_error_logs(traceback_log)
    summary = bug_hunter.get_analysis_summary(analysis)

    assert summary["total_errors"] == 1
    assert summary["total_fixes"] >= 1
    assert "key_error" in summary["error_types"]
    assert summary["average_fix_confidence"] > 0
    assert len(summary["files_affected"]) >= 1
    assert summary["root_cause_summary"] != ""
    assert "estimated_effort" in summary
    assert "risk_level" in summary


def test_get_analysis_summary_empty(bug_hunter):
    """get_analysis_summary with no errors/fixes returns zeros."""
    result = BugAnalysisResult(
        errors=[],
        resolves=[],
        root_cause_analysis="No issues",
        fix_summary="No fixes",
        estimated_effort="Low",
        risk_assessment="Low",
    )

    summary = bug_hunter.get_analysis_summary(result)

    assert summary["total_errors"] == 0
    assert summary["total_fixes"] == 0
    assert summary["average_fix_confidence"] == 0.0
    assert summary["files_affected"] == []


def test_get_analysis_summary_long_root_cause(bug_hunter):
    """Root cause longer than 200 chars gets truncated with ellipsis."""
    long_analysis = "x" * 250
    result = BugAnalysisResult(
        errors=[],
        resolves=[],
        root_cause_analysis=long_analysis,
        fix_summary="summary",
        estimated_effort="Low",
        risk_assessment="Low",
    )

    summary = bug_hunter.get_analysis_summary(result)

    assert summary["root_cause_summary"].endswith("...")
    assert len(summary["root_cause_summary"]) <= 203  # 200 + len("...")


# ---------------------------------------------------------------------------
# JavaScript error through the full pipeline
# ---------------------------------------------------------------------------


def test_analyze_javascript_error_end_to_end(bug_hunter):
    """JavaScript TypeError should be parsed and generate fixes."""
    js_log = (
        "TypeError: Cannot read properties of undefined (reading 'map')\n"
        "    at processData (/app/src/index.js:42:10)\n"
        "    at main (/app/src/index.js:10:5)\n"
    )
    result = bug_hunter.analyze_error_logs(js_log)

    assert isinstance(result, BugAnalysisResult)
    assert len(result.errors) == 1
    assert result.errors[0].error_type == ErrorType.TYPE_ERROR
    assert result.errors[0].file_path == "/app/src/index.js"
    assert result.errors[0].line_number == 42
    assert len(result.resolves) >= 1


# ---------------------------------------------------------------------------
# Go panic through the full pipeline
# ---------------------------------------------------------------------------


def test_analyze_go_panic_end_to_end(bug_hunter):
    """Go panic should be parsed and generate fixes."""
    go_log = (
        "panic: runtime error: index out of range [5] with length 3\n"
        "\n"
        "goroutine 1 [running]:\n"
        "main.process(0x1234)\n"
        "\t/app/main.go:42 +0x80\n"
    )
    result = bug_hunter.analyze_error_logs(go_log)

    assert isinstance(result, BugAnalysisResult)
    assert len(result.errors) == 1
    assert result.errors[0].error_type == ErrorType.RUNTIME_ERROR
    assert "index out of range" in result.errors[0].message
    assert result.errors[0].file_path == "/app/main.go"
    assert len(result.resolves) >= 1


# ---------------------------------------------------------------------------
# Bug Hunter ENDPOINT tests (app.api.endpoints.bug_hunter)
# Covering uncovered lines: 252-260, 411-419, 610-612, 632-634, 706-717, 747
# ---------------------------------------------------------------------------

from unittest.mock import AsyncMock

from app.api.endpoints.bug_hunter import (
    get_bug_hunter_status,
    hunt_bugs,
    start_bug_hunt,
)


@pytest.fixture
def mock_container_manager():
    """Mock container manager for endpoint tests."""
    manager = AsyncMock()
    return manager


@pytest.mark.asyncio
async def test_get_bug_hunter_status_success():
    """Successful status retrieval (lines 215-248)."""
    mock_mgr = AsyncMock()
    mock_mgr.get_container = AsyncMock(return_value=MagicMock(container_name="test-ctr"))
    with patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=mock_mgr):
        response = await get_bug_hunter_status("test-ctr", tenant_id="test-tenant")
        assert response.container_name == "test-ctr"
        assert response.analysis_status is not None


@pytest.mark.asyncio
async def test_get_bug_hunter_status_file_not_found():
    """FileNotFoundError during status retrieval (lines 252-253)."""
    from fastapi import HTTPException

    mock_mgr = AsyncMock()
    mock_mgr.get_container = AsyncMock(side_effect=FileNotFoundError("container missing"))
    with patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=mock_mgr):
        with pytest.raises(HTTPException) as exc_info:
            await get_bug_hunter_status("missing-ctr", tenant_id="test-tenant")
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_bug_hunter_status_generic_exception_not_found_text():
    """Generic exception with 'not found' text returns 404 (lines 257-258)."""
    from fastapi import HTTPException

    mock_mgr = AsyncMock()
    mock_mgr.get_container = AsyncMock(side_effect=Exception("container not found in cluster"))
    with patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=mock_mgr):
        with pytest.raises(HTTPException) as exc_info:
            await get_bug_hunter_status("bad-ctr", tenant_id="test-tenant")
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_bug_hunter_status_generic_exception_500():
    """Generic exception returns 500 (lines 259-260)."""
    from fastapi import HTTPException

    mock_mgr = AsyncMock()
    mock_mgr.get_container = AsyncMock(side_effect=Exception("database connection failed"))
    with patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=mock_mgr):
        with pytest.raises(HTTPException) as exc_info:
            await get_bug_hunter_status("err-ctr", tenant_id="test-tenant")
        assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_start_bug_hunt_success():
    """Successful start hunt (lines 378-407)."""
    mock_mgr = AsyncMock()
    mock_mgr.get_container = AsyncMock(return_value=MagicMock(container_name="test-ctr"))
    with patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=mock_mgr):
        response = await start_bug_hunt("test-ctr", request=None, tenant_id="test-tenant")
        assert "started" in response.message.lower()


@pytest.mark.asyncio
async def test_start_bug_hunt_file_not_found():
    """FileNotFoundError during start hunt (lines 411-412)."""
    from fastapi import HTTPException

    mock_mgr = AsyncMock()
    mock_mgr.get_container = AsyncMock(side_effect=FileNotFoundError("container missing"))
    with patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=mock_mgr):
        with pytest.raises(HTTPException) as exc_info:
            await start_bug_hunt("missing-ctr", request=None, tenant_id="test-tenant")
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_start_bug_hunt_generic_not_found_exception():
    """Generic exception with 'not found' text (lines 416-417)."""
    from fastapi import HTTPException

    mock_mgr = AsyncMock()
    mock_mgr.get_container = AsyncMock(side_effect=Exception("resource not found"))
    with patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=mock_mgr):
        with pytest.raises(HTTPException) as exc_info:
            await start_bug_hunt("bad-ctr", request=None, tenant_id="test-tenant")
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_start_bug_hunt_generic_exception_500():
    """Generic exception returns 500 (lines 418-419)."""
    from fastapi import HTTPException

    mock_mgr = AsyncMock()
    mock_mgr.get_container = AsyncMock(side_effect=Exception("random failure"))
    with patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=mock_mgr):
        with pytest.raises(HTTPException) as exc_info:
            await start_bug_hunt("err-ctr", request=None, tenant_id="test-tenant")
        assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_hunt_bugs_container_validation_unexpected_error():
    """Unexpected error during container validation (lines 610-612)."""
    from fastapi import HTTPException

    mock_mgr = AsyncMock()
    mock_mgr.get_container = AsyncMock(side_effect=TypeError("unexpected"))
    mock_request = MagicMock()
    mock_request.error_logs = ""
    mock_request.error_logs_base64 = None
    mock_request.context = None

    with patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=mock_mgr):
        with pytest.raises(HTTPException) as exc_info:
            await hunt_bugs(
                "test-ctr",
                request=mock_request,
                bug_analyzer=MagicMock(),
                tenant_id="test-tenant",
            )
        assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_hunt_bugs_base64_error_logs():
    """Base64 error logs decoding (lines 626-634)."""
    import base64

    mock_mgr = AsyncMock()
    mock_container = MagicMock()
    mock_container.work_dir = "/tmp/test"
    mock_mgr.get_container = AsyncMock(return_value=mock_container)

    error_text = "Traceback (most recent call last):\nValueError: bad\n"
    encoded = base64.b64encode(error_text.encode()).decode()

    mock_request = MagicMock()
    mock_request.error_logs = None
    mock_request.error_logs_base64 = encoded
    mock_request.context = None
    mock_request.create_pr = False
    mock_request.github_repo = None
    mock_request.GITHUB_ACCESS_TOKEN = None

    mock_result = MagicMock()
    mock_result.errors = []
    mock_result.resolves = []
    mock_result.root_cause_analysis = "No issues"
    mock_result.fix_summary = "None"
    mock_result.estimated_effort = "Low"
    mock_result.risk_assessment = "Low"
    # BugAnalysisResult coverage fields: unmeasured (a bare MagicMock attr is not an int).
    mock_result.scanned_files = None
    mock_result.tracked_source_files = None
    mock_result.commit_sha = None

    with (
        patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=mock_mgr),
        patch("app.api.endpoints.bug_hunter.BugHunter") as mock_bh_cls,
    ):
        mock_bh_cls.return_value.analyze_error_logs.return_value = mock_result
        response = await hunt_bugs(
            "test-ctr",
            request=mock_request,
            bug_analyzer=MagicMock(),
            tenant_id="test-tenant",
        )
        assert response.errors_found == 0


@pytest.mark.asyncio
async def test_hunt_bugs_base64_decode_failure():
    """Invalid base64 falls back to plain text (lines 632-634)."""
    mock_mgr = AsyncMock()
    mock_container = MagicMock()
    mock_container.work_dir = "/tmp/test"
    mock_mgr.get_container = AsyncMock(return_value=mock_container)

    mock_request = MagicMock()
    mock_request.error_logs = None
    mock_request.error_logs_base64 = "not-valid-base64!!!"
    mock_request.context = None
    mock_request.create_pr = False
    mock_request.github_repo = None
    mock_request.GITHUB_ACCESS_TOKEN = None

    mock_result = MagicMock()
    mock_result.errors = []
    mock_result.resolves = []
    mock_result.root_cause_analysis = "No issues"
    mock_result.fix_summary = "None"
    mock_result.estimated_effort = "Low"
    mock_result.risk_assessment = "Low"
    # BugAnalysisResult coverage fields: unmeasured (a bare MagicMock attr is not an int).
    mock_result.scanned_files = None
    mock_result.tracked_source_files = None
    mock_result.commit_sha = None

    with (
        patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=mock_mgr),
        patch("app.api.endpoints.bug_hunter.BugHunter") as mock_bh_cls,
    ):
        mock_bh_cls.return_value.analyze_error_logs.return_value = mock_result
        response = await hunt_bugs(
            "test-ctr",
            request=mock_request,
            bug_analyzer=MagicMock(),
            tenant_id="test-tenant",
        )
        assert response.errors_found == 0


@pytest.mark.asyncio
async def test_hunt_bugs_with_pr_creation_failure():
    """PR creation failure path (lines 706-717)."""
    mock_mgr = AsyncMock()
    mock_container = MagicMock()
    mock_container.work_dir = "/tmp/test"
    mock_mgr.get_container = AsyncMock(return_value=mock_container)

    mock_request = MagicMock()
    mock_request.error_logs = "some error"
    mock_request.error_logs_base64 = None
    mock_request.context = None
    mock_request.create_pr = True
    mock_request.github_repo = "owner/repo"
    mock_request.GITHUB_ACCESS_TOKEN = "fake-token"

    mock_result = MagicMock()
    mock_result.errors = []
    mock_result.resolves = []
    mock_result.root_cause_analysis = "No issues"
    mock_result.fix_summary = "None"
    mock_result.estimated_effort = "Low"
    mock_result.risk_assessment = "Low"
    # BugAnalysisResult coverage fields: unmeasured (a bare MagicMock attr is not an int).
    mock_result.scanned_files = None
    mock_result.tracked_source_files = None
    mock_result.commit_sha = None

    with (
        patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=mock_mgr),
        patch("app.api.endpoints.bug_hunter.BugHunter") as mock_bh_cls,
    ):
        mock_bh = MagicMock()
        mock_bh.analyze_error_logs.return_value = mock_result
        # PR creation fails - covers the except path on line 716-717
        mock_bh.create_pull_request = AsyncMock(side_effect=Exception("PR creation failed"))
        mock_bh_cls.return_value = mock_bh

        response = await hunt_bugs(
            "test-ctr",
            request=mock_request,
            bug_analyzer=MagicMock(),
            tenant_id="test-tenant",
        )
        assert response.errors_found == 0
        assert response.pull_request is None
        mock_bh.create_pull_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_hunt_bugs_unexpected_analysis_failure():
    """Unexpected exception during analysis returns 500 (lines 748-752)."""
    from fastapi import HTTPException

    mock_mgr = AsyncMock()
    mock_container = MagicMock()
    mock_container.work_dir = "/tmp/test"
    mock_mgr.get_container = AsyncMock(return_value=mock_container)

    mock_request = MagicMock()
    mock_request.error_logs = "some error"
    mock_request.error_logs_base64 = None
    mock_request.context = None
    mock_request.create_pr = False

    with (
        patch("app.api.endpoints.bug_hunter.get_container_manager", return_value=mock_mgr),
        patch(
            "app.api.endpoints.bug_hunter.BugHunter",
            side_effect=Exception("unexpected crash"),
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await hunt_bugs(
                "test-ctr",
                request=mock_request,
                bug_analyzer=MagicMock(),
                tenant_id="test-tenant",
            )
        assert exc_info.value.status_code == 500
