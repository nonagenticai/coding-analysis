"""Unit tests for app.api.endpoints.quality module.

Focuses on the module-level helper functions (_calculate_grade, _build_summary,
_generate_recommendations) and basic model/endpoint existence checks.
"""

from __future__ import annotations

from app.api.endpoints.quality import (
    _build_summary,
    _calculate_grade,
    _generate_recommendations,
    get_quality_analyzer,
    get_supported_languages,
    router,
)
from models.architecture_models import ArchitectureQualityIssue


def _make_issue(severity: str = "low") -> ArchitectureQualityIssue:
    """Create a minimal ArchitectureQualityIssue for testing."""
    return ArchitectureQualityIssue(
        id="test-1",
        severity=severity,
        message="test issue",
        rule_id="R001",
        rule_name="test-rule",
        file_path="test.py",
        line_number=1,
        tool="test",
        category="test",
    )


# ---------------------------------------------------------------------------
# _calculate_grade
# ---------------------------------------------------------------------------


class TestCalculateGrade:
    """Tests for the _calculate_grade helper function."""

    def test_grade_a(self):
        assert _calculate_grade(95.0) == "A"
        assert _calculate_grade(90.0) == "A"
        assert _calculate_grade(100.0) == "A"

    def test_grade_b(self):
        assert _calculate_grade(85.0) == "B"
        assert _calculate_grade(80.0) == "B"
        assert _calculate_grade(89.9) == "B"

    def test_grade_c(self):
        assert _calculate_grade(75.0) == "C"
        assert _calculate_grade(70.0) == "C"
        assert _calculate_grade(79.9) == "C"

    def test_grade_d(self):
        assert _calculate_grade(65.0) == "D"
        assert _calculate_grade(60.0) == "D"
        assert _calculate_grade(69.9) == "D"

    def test_grade_f(self):
        assert _calculate_grade(59.9) == "F"
        assert _calculate_grade(0.0) == "F"
        assert _calculate_grade(50.0) == "F"


# ---------------------------------------------------------------------------
# _build_summary
# ---------------------------------------------------------------------------


class TestBuildSummary:
    """Tests for the _build_summary helper function."""

    def test_empty_issues(self):
        result = _build_summary([])
        assert result["total_issues"] == 0

    def test_counts_by_severity(self):
        issue_high = _make_issue("high")
        issue_low = _make_issue("low")
        issue_critical = _make_issue("critical")
        result = _build_summary([issue_high, issue_low, issue_critical])
        assert result["total_issues"] == 3
        assert result["high"] == 1
        assert result["low"] == 1
        assert result["critical"] == 1

    def test_multiple_same_severity(self):
        issues = [_make_issue("high") for _ in range(5)]
        result = _build_summary(issues)
        assert result["total_issues"] == 5
        assert result["high"] == 5

    def test_unknown_severity_fallback(self):
        class NoSeverity:
            pass

        issue = NoSeverity()
        result = _build_summary([issue])  # type: ignore[invalid-argument-type]
        assert result["total_issues"] == 1
        assert result.get("unknown", 0) == 1


# ---------------------------------------------------------------------------
# _generate_recommendations
# ---------------------------------------------------------------------------


class TestGenerateRecommendations:
    """Tests for the _generate_recommendations helper function."""

    def test_no_issues(self):
        result = _generate_recommendations([])
        assert len(result) == 1
        assert "Great job" in result[0]

    def test_with_issues_includes_count(self):
        issues = [_make_issue("low") for _ in range(3)]
        result = _generate_recommendations(issues)
        assert any("3" in rec for rec in result)

    def test_critical_issues_get_prioritized(self):
        issues = [
            _make_issue("critical"),
            _make_issue("low"),
        ]
        result = _generate_recommendations(issues)
        assert any("critical" in rec.lower() for rec in result)

    def test_high_issues_mentioned(self):
        issues = [
            _make_issue("high"),
            _make_issue("high"),
            _make_issue("low"),
        ]
        result = _generate_recommendations(issues)
        assert any("high" in rec.lower() for rec in result)

    def test_no_critical_or_high_skips_those_messages(self):
        issues = [_make_issue("low"), _make_issue("moderate")]
        result = _generate_recommendations(issues)
        assert len(result) == 1
        assert "2 quality issues" in result[0]


# ---------------------------------------------------------------------------
# get_quality_analyzer dependency
# ---------------------------------------------------------------------------


class TestGetQualityAnalyzer:
    """Tests for the get_quality_analyzer dependency function."""

    def test_returns_quality_analyzer_instance(self):
        analyzer = get_quality_analyzer()
        assert analyzer is not None
        assert hasattr(analyzer, "analyze") or hasattr(analyzer, "analyze_quality")


# ---------------------------------------------------------------------------
# Router existence
# ---------------------------------------------------------------------------


class TestQualityRouter:
    """Tests for the quality router configuration."""

    def test_router_exists(self):
        from fastapi import APIRouter

        assert isinstance(router, APIRouter)

    def test_router_has_routes(self):
        assert len(router.routes) > 0

    def test_languages_endpoint_registered(self):
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/languages" in paths


# ---------------------------------------------------------------------------
# get_supported_languages endpoint
# ---------------------------------------------------------------------------


class TestGetSupportedLanguages:
    """Tests for the get_supported_languages endpoint."""

    async def test_returns_languages_dict(self):
        result = await get_supported_languages()
        assert "languages" in result
        assert isinstance(result["languages"], list)
        assert len(result["languages"]) > 0

    async def test_python_in_languages(self):
        result = await get_supported_languages()
        names = [lang["name"] for lang in result["languages"]]
        assert "python" in names

    async def test_each_language_has_tools(self):
        result = await get_supported_languages()
        for lang in result["languages"]:
            assert "tools" in lang
            assert isinstance(lang["tools"], list)
            assert len(lang["tools"]) > 0

    async def test_each_language_has_extensions(self):
        result = await get_supported_languages()
        for lang in result["languages"]:
            assert "extensions" in lang
            assert isinstance(lang["extensions"], list)
            assert len(lang["extensions"]) > 0
