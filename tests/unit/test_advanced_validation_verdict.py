"""Tests that `/api/analysis/static/advanced-validation` can return a FAILING verdict.

Before the fix this endpoint was structurally incapable of failing: the taint
result was a literal ``security_issues: []``, formal verification was hard-coded
``status: "passed"`` with coverage 0.92, and the per-file safety score was the
constant 0.8. ``blocking_issues`` derived from the always-empty taint result
alone, so it was always ``[]``.

The load-bearing test here is `TestVerdictIsNotConstant` — the negative control.
It asserts that a defective change and a benign change produce DIFFERENT
verdicts. A regression to any constant verdict, constant score or empty issue
list reds that test, however plausible the constant looks in isolation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks

from app.api.endpoints.static_analysis import (
    _identify_blocking_issues,
    _run_taint_analysis,
    blocking_severities_for,
    validate_code_changes,
)
from models.static_analysis_models import (
    ANALYZER_NOT_COMPUTED,
    AdvancedValidationRequest,
    AdvancedValidationResponse,
    CodeChange,
    StaticAnalysisChangeType,
    ValidationVerdict,
)

_VALIDATE = "app.api.endpoints.static_analysis.validate_and_get_container_for_tenant"
_SYMBOLIC = "app.api.endpoints.static_analysis._run_symbolic_execution"
_TAINT = "app.api.endpoints.static_analysis._run_taint_analysis"
_FORMAL = "app.api.endpoints.static_analysis._run_formal_verification"

# A genuine command-injection defect: unvalidated stdin straight into a shell.
DEFECTIVE_CODE = "import os\n\n\ndef handle() -> None:\n    os.system(input())\n"

# The same shape of change with nothing dangerous in it.
BENIGN_CODE = "def add(a: int, b: int) -> int:\n    return a + b\n"

# CrossHair is not a dependency of this service, so symbolic execution abstains
# in every environment the suite runs in. The suites below request only the two
# analyzers that genuinely run, except where the abstention is the subject.
_REAL_ANALYZERS = ["taint_analysis", "formal_verification"]


def _request(code: str, *, validation_types: list[str] | None = None, **change_kwargs):
    """Build an AdvancedValidationRequest carrying one modified Python file."""
    payload = {
        "file_path": "handler.py",
        "modified_code": code,
        "change_type": "MODIFIED",
    }
    payload.update(change_kwargs)
    return AdvancedValidationRequest(
        container_name="proj",
        code_changes=[CodeChange(**payload)],
        validation_types=validation_types or _REAL_ANALYZERS,
        security_level="medium",
    )


async def _validate(request) -> AdvancedValidationResponse:
    """Run the endpoint with the container lookup stubbed out."""
    with patch(_VALIDATE) as mock_validate:
        mock_validate.return_value = MagicMock(codebase_path="/app/data/repo")
        return await validate_code_changes(
            request=request,
            background_tasks=BackgroundTasks(),
            redis_client=None,
            tenant_id="t",
        )


# ---------------------------------------------------------------------------
# A real defect must fail
# ---------------------------------------------------------------------------


class TestDefectiveChangeFails:
    """A change carrying a genuine security defect must not pass."""

    async def test_command_injection_produces_fail_verdict(self):
        result = await _validate(_request(DEFECTIVE_CODE))

        assert result.verdict is ValidationVerdict.FAIL
        assert result.blocking_issues, "a command-injection defect must block"

    async def test_issue_is_named_and_located(self):
        result = await _validate(_request(DEFECTIVE_CODE))

        blocking = " ".join(result.blocking_issues)
        assert "Command Injection" in blocking
        # Located: the change's own path, and the line the call is on.
        assert "handler.py:5" in blocking

        issues = result.validation_results[0].taint_analysis.security_issues
        assert [i.severity for i in issues] == ["high"]
        assert issues[0].line == 5
        assert issues[0].code_snippet == "os.system(input())"
        assert issues[0].recommendation

    async def test_eval_of_untrusted_input_is_critical(self):
        result = await _validate(_request("def run(x):\n    return eval(x)\n"))

        assert result.verdict is ValidationVerdict.FAIL
        severities = {
            i.severity for i in result.validation_results[0].taint_analysis.security_issues
        }
        assert "critical" in severities


# ---------------------------------------------------------------------------
# A benign change must pass
# ---------------------------------------------------------------------------


class TestBenignChangePasses:
    """A change with nothing wrong with it must still come back as a pass."""

    async def test_benign_change_passes(self):
        result = await _validate(_request(BENIGN_CODE))

        assert result.verdict is ValidationVerdict.PASS
        assert result.blocking_issues == []

    async def test_benign_change_reports_which_analyzers_ran(self):
        result = await _validate(_request(BENIGN_CODE))

        assert sorted(result.analyzers_ran) == ["formal_verification", "taint_analysis"]
        assert result.overall_safety_score is not None

    async def test_formal_verification_status_is_earned_not_defaulted(self):
        """The old code hard-coded status 'passed' with coverage 0.92."""
        formal = (await _validate(_request(BENIGN_CODE))).validation_results[0].formal_verification

        assert formal.status == "passed"
        assert formal.violations == []
        # The real verifier reports the properties it actually checked.
        assert "none_checking" in formal.properties_verified
        assert formal.coverage != 0.92


# ---------------------------------------------------------------------------
# ⭐ Negative control — the test that would have caught the original bug
# ---------------------------------------------------------------------------


class TestVerdictIsNotConstant:
    """The defective and benign changes must not receive the same answer.

    This is the regression guard. Under the old implementation both of these
    returned verdict-equivalent responses: safety score 0.8, no blocking issues,
    formal verification 'passed'. Any future regression to a constant reds this.
    """

    async def test_defective_and_benign_verdicts_differ(self):
        defective = await _validate(_request(DEFECTIVE_CODE))
        benign = await _validate(_request(BENIGN_CODE))

        assert defective.verdict != benign.verdict
        assert (defective.verdict, benign.verdict) == (
            ValidationVerdict.FAIL,
            ValidationVerdict.PASS,
        )

    async def test_defective_and_benign_scores_differ(self):
        defective = await _validate(_request(DEFECTIVE_CODE))
        benign = await _validate(_request(BENIGN_CODE))

        assert defective.overall_safety_score != benign.overall_safety_score
        assert defective.overall_safety_score < benign.overall_safety_score

    async def test_defective_and_benign_blocking_issues_differ(self):
        defective = await _validate(_request(DEFECTIVE_CODE))
        benign = await _validate(_request(BENIGN_CODE))

        assert defective.blocking_issues != benign.blocking_issues
        assert benign.blocking_issues == []

    async def test_no_hardcoded_safety_score(self):
        """0.8 was the old constant; nothing may return it for every input."""
        scores = {
            (await _validate(_request(code))).overall_safety_score
            for code in (DEFECTIVE_CODE, BENIGN_CODE, "def f(x):\n    return exec(x)\n")
        }
        assert len(scores) > 1


# ---------------------------------------------------------------------------
# Abstention — no analyzer ran
# ---------------------------------------------------------------------------


class TestAbstentionIsNotAPass:
    """A request nobody could verify must say so, not quietly pass."""

    async def test_all_analyzers_unavailable_abstains(self):
        unavailable = {
            "status": ANALYZER_NOT_COMPUTED,
            "reason": "tool is not installed in this deployment",
            "security_issues": [],
        }
        with (
            patch(_TAINT, return_value=dict(unavailable)),
            patch(_FORMAL, new=AsyncMock(return_value=dict(unavailable))),
            patch(_SYMBOLIC, new=AsyncMock(return_value=dict(unavailable))),
        ):
            result = await _validate(
                _request(
                    BENIGN_CODE,
                    validation_types=[
                        "taint_analysis",
                        "formal_verification",
                        "symbolic_execution",
                    ],
                )
            )

        assert result.verdict is ValidationVerdict.ABSTAIN
        assert result.verdict is not ValidationVerdict.PASS
        assert result.analyzers_ran == []
        # No score may be reported when nothing was measured.
        assert result.overall_safety_score is None
        assert result.validation_results[0].safety_score is None

    async def test_abstention_names_every_analyzer_and_its_reason(self):
        unavailable = {
            "status": ANALYZER_NOT_COMPUTED,
            "reason": "tool is not installed in this deployment",
            "security_issues": [],
        }
        with (
            patch(_TAINT, return_value=dict(unavailable)),
            patch(_FORMAL, new=AsyncMock(return_value=dict(unavailable))),
        ):
            result = await _validate(_request(BENIGN_CODE))

        named = {a.analyzer for a in result.analyzers_not_computed}
        assert named == {"taint_analysis", "formal_verification"}
        assert all("not installed" in a.reason for a in result.analyzers_not_computed)
        assert any("No verdict" in r for r in result.recommendations)

    async def test_a_failed_analyzer_does_not_count_as_a_pass(self):
        """An analyzer that ran and errored produced no verdict either."""
        errored = {"status": "failed", "reason": "boom", "security_issues": []}
        with (
            patch(_TAINT, return_value=dict(errored)),
            patch(_FORMAL, new=AsyncMock(return_value=dict(errored))),
        ):
            result = await _validate(_request(BENIGN_CODE))

        assert result.verdict is ValidationVerdict.ABSTAIN
        assert {a.status for a in result.analyzers_not_computed} == {"failed"}

    async def test_unparseable_code_is_not_reported_as_clean(self):
        """A file that does not parse has not been checked, so it cannot pass."""
        result = await _validate(
            _request("def broken(:\n    pass\n", validation_types=["taint_analysis"])
        )

        assert result.verdict is ValidationVerdict.ABSTAIN
        assert result.validation_results[0].taint_analysis.status == "failed"
        assert "does not parse" in result.validation_results[0].taint_analysis.reason

    def test_taint_analysis_on_a_missing_file_reports_failure(self):
        taint = _run_taint_analysis("/nonexistent/path/to/file.py", "file.py")

        assert taint["status"] == "failed"
        assert taint["security_issues"] == []


# ---------------------------------------------------------------------------
# Blocking issues come from every analyzer that ran
# ---------------------------------------------------------------------------


class TestBlockingIssuesSpanAllAnalyzers:
    """`blocking_issues` used to derive from the (always empty) taint result alone."""

    def test_symbolic_assertion_violation_blocks(self):
        issues = _identify_blocking_issues(
            [
                {
                    "file_path": "m.py",
                    "symbolic_execution": {
                        "status": "completed",
                        "assertion_violations": ["x may be negative"],
                    },
                }
            ]
        )
        assert issues == ["m.py: assertion violation: x may be negative"]

    def test_formal_violation_blocks_at_its_severity(self):
        results = [
            {
                "file_path": "m.py",
                "formal_verification": {
                    "status": "violations_found",
                    "violations": [
                        {
                            "type": "unchecked_index",
                            "message": "bad",
                            "severity": "high",
                            "line": 3,
                        },
                        {
                            "type": "missing_annotation",
                            "message": "x",
                            "severity": "low",
                            "line": 1,
                        },
                    ],
                },
            }
        ]
        issues = _identify_blocking_issues(results)
        assert len(issues) == 1
        assert "unchecked_index" in issues[0]
        assert "missing_annotation" not in issues[0]

    def test_an_analyzer_that_abstained_contributes_nothing(self):
        issues = _identify_blocking_issues(
            [
                {
                    "file_path": "m.py",
                    "symbolic_execution": {
                        "status": ANALYZER_NOT_COMPUTED,
                        "assertion_violations": ["stale finding"],
                    },
                    "formal_verification": {
                        "status": ANALYZER_NOT_COMPUTED,
                        "violations": [{"type": "t", "message": "m", "severity": "critical"}],
                    },
                }
            ]
        )
        assert issues == []

    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            ("low", {"critical"}),
            ("medium", {"critical", "high"}),
            ("high", {"critical", "high", "medium"}),
            ("unrecognised", {"critical", "high"}),
        ],
    )
    def test_security_level_selects_blocking_severities(self, level, expected):
        assert set(blocking_severities_for(level)) == expected


# ---------------------------------------------------------------------------
# The conveyor's code-change shape
# ---------------------------------------------------------------------------


class TestConveyorCodeChangeShape:
    """The producer emits `{file_path, content, change_type: "deleted"}`, lowercase."""

    def test_lowercase_change_type_is_accepted(self):
        change = CodeChange.model_validate(
            {"file_path": "a.py", "content": BENIGN_CODE, "change_type": "modified"}
        )
        assert change.change_type is StaticAnalysisChangeType.MODIFIED
        assert change.modified_code == BENIGN_CODE

    def test_content_is_accepted_as_modified_code(self):
        change = CodeChange.model_validate(
            {"file_path": "a.py", "content": "x = 1\n", "change_type": "ADDED"}
        )
        assert change.modified_code == "x = 1\n"

    def test_deleted_entry_with_no_content_is_accepted(self):
        """A DELETED file legitimately has no body — it must not 422."""
        change = CodeChange.model_validate({"file_path": "gone.py", "change_type": "deleted"})
        assert change.change_type is StaticAnalysisChangeType.DELETED
        assert change.modified_code == ""

    def test_deleted_entry_with_explicit_null_content_is_accepted(self):
        change = CodeChange.model_validate(
            {"file_path": "gone.py", "content": None, "change_type": "deleted"}
        )
        assert change.modified_code == ""

    def test_non_deleted_entry_without_a_body_is_rejected(self):
        with pytest.raises(ValueError, match="required"):
            CodeChange.model_validate({"file_path": "a.py", "change_type": "modified"})

    async def test_full_conveyor_request_validates_and_still_fails_on_a_defect(self):
        request = AdvancedValidationRequest.model_validate(
            {
                "container_name": "proj",
                "code_changes": [
                    {"file_path": "gone.py", "content": None, "change_type": "deleted"},
                    {
                        "file_path": "handler.py",
                        "content": DEFECTIVE_CODE,
                        "change_type": "modified",
                    },
                ],
                "validation_types": _REAL_ANALYZERS,
            }
        )
        result = await _validate(request)

        assert result.verdict is ValidationVerdict.FAIL
        assert any("Command Injection" in issue for issue in result.blocking_issues)

    async def test_deleted_file_abstains_rather_than_scoring_its_empty_body(self):
        request = AdvancedValidationRequest.model_validate(
            {
                "container_name": "proj",
                "code_changes": [
                    {"file_path": "gone.py", "content": None, "change_type": "deleted"}
                ],
                "validation_types": _REAL_ANALYZERS,
            }
        )
        result = await _validate(request)

        assert result.verdict is ValidationVerdict.ABSTAIN
        assert result.overall_safety_score is None
        assert all("deleted" in a.reason for a in result.analyzers_not_computed)
