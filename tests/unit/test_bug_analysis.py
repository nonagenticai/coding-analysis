"""Unit tests for BugAnalyzer service."""

from __future__ import annotations

import pytest

from models.bug_models import BugAnalysisResult, CodeFix, ErrorType, ParsedError


def _make_error(
    error_type: ErrorType = ErrorType.SYNTAX_ERROR,
    message: str = "test error",
    file_path: str = "test.py",
    confidence: float = 0.9,
) -> ParsedError:
    return ParsedError(
        error_type=error_type,
        message=message,
        file_path=file_path,
        line_number=1,
        function_name=None,
        stack_trace=[],
        confidence=confidence,
    )


def _make_fix(
    file_path: str = "test.py",
    confidence: float = 0.8,
    fixed_code: str = "y = 1",
    explanation: str = "fix issue",
) -> CodeFix:
    return CodeFix(
        file_path=file_path,
        line_number=1,
        original_code="x",
        fixed_code=fixed_code,
        explanation=explanation,
        confidence=confidence,
    )


@pytest.fixture
def analyzer():
    from services.bug_analysis import BugAnalyzer

    return BugAnalyzer()


def test_analyze_root_causes_with_errors(analyzer):
    """analyze_root_causes returns markdown with 'Root Cause Analysis'."""
    errors = [_make_error()]
    fixes = [_make_fix()]

    result = analyzer.analyze_root_causes(errors, fixes)

    assert "Root Cause Analysis" in result
    assert "1" in result  # 1 error


def test_analyze_root_causes_multiple_types(analyzer):
    """Multiple error types appear in the distribution section."""
    errors = [
        _make_error(ErrorType.SYNTAX_ERROR),
        _make_error(ErrorType.IMPORT_ERROR),
        _make_error(ErrorType.NAME_ERROR),
    ]
    fixes = []

    result = analyzer.analyze_root_causes(errors, fixes)

    assert "Root Cause Analysis" in result
    assert "Syntax Error" in result or "syntax_error" in result.lower()


def test_analyze_root_causes_empty(analyzer):
    """Empty error list returns 'No errors found'."""
    result = analyzer.analyze_root_causes([], [])

    assert "No errors found" in result


def test_analyze_root_causes_file_patterns(analyzer):
    """Multiple errors in same file appear as most problematic file."""
    errors = [
        _make_error(file_path="bad_file.py"),
        _make_error(file_path="bad_file.py"),
        _make_error(file_path="other.py"),
    ]

    result = analyzer.analyze_root_causes(errors, [])

    assert "bad_file.py" in result


def test_analyze_root_causes_confidence_distribution(analyzer):
    """Fix confidence distribution is included in root cause analysis."""
    fixes = [
        _make_fix(confidence=0.9),  # high
        _make_fix(confidence=0.6),  # medium
        _make_fix(confidence=0.3),  # low
    ]

    result = analyzer.analyze_root_causes([_make_error()], fixes)

    assert "High confidence" in result
    assert "Medium confidence" in result
    assert "Low confidence" in result


def test_generate_fix_summary_with_fixes(analyzer):
    """generate_fix_summary returns markdown with 'Resolve Summary'."""
    fixes = [_make_fix(confidence=0.9), _make_fix(confidence=0.6)]

    result = analyzer.generate_fix_summary(fixes)

    assert "Resolve Summary" in result
    assert "test.py" in result


def test_generate_fix_summary_empty(analyzer):
    """Empty fix list returns 'No resolves generated'."""
    result = analyzer.generate_fix_summary([])

    assert "No resolves generated" in result


def test_generate_fix_summary_high_confidence_shown(analyzer):
    """High-confidence fixes are highlighted in summary."""
    fixes = [_make_fix(confidence=0.95, explanation="critical fix")]

    result = analyzer.generate_fix_summary(fixes)

    assert "High-Confidence" in result or "high-confidence" in result.lower()
    assert "critical fix" in result


def test_estimate_effort_empty(analyzer):
    """Empty fix list returns 'No effort required'."""
    result = analyzer.estimate_effort([])

    assert "No effort required" in result


def test_estimate_effort_with_fixes(analyzer):
    """estimate_effort returns markdown with effort breakdown."""
    fixes = [
        _make_fix(confidence=0.9, fixed_code="x = 1"),  # simple: high conf + short code
        _make_fix(confidence=0.6, fixed_code="x = 1"),  # moderate
        _make_fix(confidence=0.3, fixed_code="x = 1"),  # complex
    ]

    result = analyzer.estimate_effort(fixes)

    assert "Effort Estimation" in result
    assert "Estimated Total Time" in result


def test_estimate_effort_categorizes_correctly(analyzer):
    """High confidence + short code -> basic; low confidence -> complex."""
    simple_fix = _make_fix(confidence=0.9, fixed_code="x")  # < 100 chars, high conf
    complex_fix = _make_fix(confidence=0.2, fixed_code="y")  # low confidence

    result = analyzer.estimate_effort([simple_fix, complex_fix])

    assert "Basic" in result or "basic" in result.lower()
    assert "Complex" in result or "complex" in result.lower()


def test_assess_risks_empty(analyzer):
    """Empty fix list returns 'No risks'."""
    result = analyzer.assess_risks([])

    assert "No risks" in result


def test_assess_risks_with_fixes(analyzer):
    """assess_risks returns markdown with risk distribution."""
    fixes = [_make_fix(), _make_fix(confidence=0.3)]

    result = analyzer.assess_risks(fixes)

    assert "Risk Assessment" in result
    assert "Risk Distribution" in result


def test_assess_risks_high_risk_low_confidence(analyzer):
    """Low confidence fix gets elevated risk score."""
    # Low confidence (< 0.5) adds 3 to risk score -> high risk threshold 5
    low_conf_fix = _make_fix(confidence=0.1, fixed_code="x" * 300)  # also > 200 chars -> +2 more

    result = analyzer.assess_risks([low_conf_fix])

    assert "High Risk" in result or "high risk" in result.lower()


def test_assess_risks_core_file_increases_risk(analyzer):
    """Core files (main, config, __init__) increase risk score."""
    core_fix = _make_fix(file_path="main.py", confidence=0.4)  # low conf + core path

    result = analyzer.assess_risks([core_fix])

    assert "Risk Assessment" in result


def test_analyze_bugs_and_fixes_returns_result(analyzer):
    """analyze_bugs_and_fixes returns complete BugAnalysisResult."""
    errors = [_make_error()]
    fixes = [_make_fix()]

    result = analyzer.analyze_bugs_and_fixes(errors, fixes)

    assert isinstance(result, BugAnalysisResult)
    assert result.errors == errors
    assert result.resolves == fixes
    assert "Root Cause Analysis" in result.root_cause_analysis
    assert "Resolve Summary" in result.fix_summary
    assert "Effort Estimation" in result.estimated_effort
    assert "Risk Assessment" in result.risk_assessment


def test_analyze_bugs_and_fixes_empty(analyzer):
    """analyze_bugs_and_fixes with empty inputs returns valid result."""
    result = analyzer.analyze_bugs_and_fixes([], [])

    assert isinstance(result, BugAnalysisResult)
    assert result.errors == []
    assert result.resolves == []


@pytest.mark.asyncio
async def test_analyze_container_valid(analyzer):
    """analyze_container with valid inputs returns dict with expected keys."""
    result = await analyzer.analyze_container("my-container", "Error: something went wrong")

    assert isinstance(result, dict)
    assert "errors" in result
    assert "resolves" in result
    assert "container_context" in result
    assert "file_mappings" in result
    assert "analysis" in result
    assert result["container_context"]["container_name"] == "my-container"


@pytest.mark.asyncio
async def test_analyze_container_invalid_name(analyzer):
    """analyze_container with empty container_name raises ValueError."""
    with pytest.raises(ValueError):
        await analyzer.analyze_container("", "some logs")


@pytest.mark.asyncio
async def test_analyze_container_invalid_logs(analyzer):
    """analyze_container with empty logs raises ValueError."""
    with pytest.raises(ValueError):
        await analyzer.analyze_container("my-container", "")


@pytest.mark.asyncio
async def test_generate_fixes_empty(analyzer):
    """generate_fixes with empty list returns empty list."""
    result = await analyzer.generate_fixes([])

    assert result == []


@pytest.mark.asyncio
async def test_generate_fixes_with_errors(analyzer):
    """generate_fixes returns CodeFix list for each valid ParsedError."""
    errors = [
        _make_error(ErrorType.SYNTAX_ERROR),
        _make_error(ErrorType.IMPORT_ERROR),
        _make_error(ErrorType.NAME_ERROR),
    ]

    result = await analyzer.generate_fixes(errors)

    assert isinstance(result, list)
    assert all(isinstance(f, CodeFix) for f in result)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_generate_fixes_invalid_input(analyzer):
    """generate_fixes with non-list raises ValueError."""
    with pytest.raises(ValueError):
        await analyzer.generate_fixes("not a list")


@pytest.mark.asyncio
async def test_generate_fixes_skips_non_parsed_error(analyzer):
    """generate_fixes skips items that are not ParsedError instances."""
    mixed = [_make_error(), "not an error", None]

    result = await analyzer.generate_fixes(mixed)

    # Only the first ParsedError should produce a fix
    assert isinstance(result, list)
    assert len(result) == 1
