"""Keep-out: a bug-hunt finding's severity is ``unrated`` unless it was COMPUTED.

coding-analysis#112. The bug hunter has nothing to rate a finding from
(``ParsedError`` carries no severity), yet it emitted ``"medium"`` and the store
fell back to ``"medium"``, so every persisted row read ``medium``. That was a
constant presented as data. These tests pin both halves:

* the PRODUCER emits ``unrated``;
* the STORE persists a severity only when the finding names a computed
  ``severity_basis``. Anything else, including absent, a bare "medium" or a
  basis that is blank, is stored as ``unrated``.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from app.api.endpoints import bug_hunter as bug_hunter_endpoint
from models.analysis_models import UNRATED_SEVERITY, BugHuntingResponse
from services.bug_hunt_report_store import build_finding_rows, persisted_severity


def test_unrated_is_the_literal_unrated():
    assert UNRATED_SEVERITY == "unrated"


@pytest.mark.parametrize(
    "finding",
    [
        {},
        {"severity": None},
        {"severity": ""},
        {"severity": "medium"},  # the old constant
        {"severity": "high"},  # a value with no basis is not a rating
        {"severity": "high", "severity_basis": ""},
        {"severity": "high", "severity_basis": "   "},
        {"severity": "high", "severity_basis": None},
        {"severity": "", "severity_basis": "error_type_map:v1"},
    ],
)
def test_store_never_persists_an_unbased_severity(finding):
    assert persisted_severity(finding) == UNRATED_SEVERITY


def test_store_keeps_a_computed_severity():
    # Positive control: the keep-out must not swallow a REAL rating.
    finding = {"severity": "blocking", "severity_basis": "error_type_map:v1"}
    assert persisted_severity(finding) == "blocking"


def test_store_has_no_medium_fallback():
    # The function's CODE, not its docstring (which explains the old fallback).
    fn = ast.parse(textwrap.dedent(inspect.getsource(persisted_severity))).body[0]
    body = [
        n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    ]
    literals = {n.value for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.Constant)}
    assert "medium" not in literals


def test_producer_emits_unrated_not_a_literal_severity():
    # The ErrorDetailDict built from each ParsedError must take its severity from
    # UNRATED_SEVERITY, never from a string literal.
    src = inspect.getsource(bug_hunter_endpoint)
    assert '"severity": UNRATED_SEVERITY' in src
    for literal in ("medium", "high", "low", "critical"):
        assert f'"severity": "{literal}"' not in src, literal


def _error(i: int, **extra: object) -> dict[str, object]:
    return {
        "error_id": f"err_{i}",
        "error_type": "syntax_error",
        "error_message": "bad syntax",
        "stack_trace": "",
        "file_path": f"/repo/src/m{i}.py",
        "line_number": i + 1,
        "function_name": "",
        "error_category": "syntax_error",
        "timestamp": "t",
        "frequency": 1,
        "impact_score": 0.9,
        "suggested_fix": "",
        "related_errors": [],
        **extra,
    }


def test_build_finding_rows_persists_unrated_unless_computed():
    # Pins the STORE call site, not just the helper: restoring the old
    # 'or "medium"' at build_finding_rows must turn this red.
    report = BugHuntingResponse(
        errors_found=4,
        fixes_generated=0,
        root_cause_analysis="root",
        fix_summary="fix",
        estimated_effort="small",
        risk_assessment="low",
        errors=[
            _error(0, severity=""),  # empty (the model requires the key)
            _error(1, severity="medium"),  # the old constant
            _error(2, severity="high"),  # a value without a basis
            _error(3, severity="blocking", severity_basis="error_type_map:v1"),
        ],
        resolves=[],
        pull_request=None,
        analysis_time=0.1,
    )

    rows = build_finding_rows(report, codebase_path="/repo")

    assert [r["severity"] for r in rows] == ["unrated", "unrated", "unrated", "blocking"]
