"""Unit tests for services.quality_metrics_analyzer module."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from models.testing_models import QualityIssue, QualityResult
from services.quality_metrics_analyzer import (
    QualityAnalysisResult,
    QualityCategory,
    QualityIssueAnalysis,
    QualityMetric,
    QualityMetricsAnalyzer,
    QualityPriority,
    QualityRecommendation,
    QualityTrend,
    create_quality_metrics_analyzer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_issue(
    *,
    tool: str = "ruff",
    severity: str = "warning",
    message: str = "some issue",
    file_path: str = "src/foo.py",
    fix_confidence: float | None = None,
) -> QualityIssue:
    return QualityIssue(
        tool=tool,
        severity=severity,
        message=message,
        file_path=file_path,
        fix_confidence=fix_confidence,
    )


def _make_quality_result(
    *,
    overall_score: float = 85.0,
    issues: list[QualityIssue] | None = None,
    language: str = "python",
    created_at: datetime | None = None,
) -> QualityResult:
    issues = issues or []
    return QualityResult(
        analysis_id="qr-1",
        success=True,
        overall_score=overall_score,
        total_issues=len(issues),
        issues=issues,
        execution_time=0.5,
        created_at=created_at or datetime.now(UTC),
        language=language,
    )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_quality_category_values(self):
        assert QualityCategory.MAINTAINABILITY == "maintainability"
        assert QualityCategory.SECURITY == "security"
        assert QualityCategory.PERFORMANCE == "performance"
        assert QualityCategory.COMPLEXITY == "complexity"
        assert QualityCategory.DOCUMENTATION == "documentation"
        assert QualityCategory.COVERAGE == "coverage"
        assert QualityCategory.STYLE == "style"
        assert QualityCategory.RELIABILITY == "reliability"

    def test_quality_priority_values(self):
        assert QualityPriority.CRITICAL == "critical"
        assert QualityPriority.HIGH == "high"
        assert QualityPriority.MEDIUM == "medium"
        assert QualityPriority.LOW == "low"
        assert QualityPriority.INFO == "info"

    def test_quality_trend_values(self):
        assert QualityTrend.IMPROVING == "improving"
        assert QualityTrend.DEGRADING == "degrading"
        assert QualityTrend.STABLE == "stable"
        assert QualityTrend.VOLATILE == "volatile"


# ---------------------------------------------------------------------------
# QualityMetric model tests
# ---------------------------------------------------------------------------


class TestQualityMetric:
    def test_weighted_score(self):
        metric = QualityMetric(
            metric_id="m1",
            metric_name="Test",
            category=QualityCategory.MAINTAINABILITY,
            current_value=80.0,
            target_value=90.0,
            score=80.0,
            weight=0.5,
            trend=QualityTrend.STABLE,
            trend_confidence=0.8,
            description="desc",
            unit="score",
        )
        assert metric.weighted_score == pytest.approx(40.0)

    def test_improvement_needed(self):
        metric = QualityMetric(
            metric_id="m1",
            metric_name="Test",
            category=QualityCategory.MAINTAINABILITY,
            current_value=70.0,
            target_value=100.0,
            score=70.0,
            weight=0.5,
            trend=QualityTrend.STABLE,
            trend_confidence=0.8,
            description="desc",
            unit="score",
        )
        # (100 - 70) / 100 * 100 = 30.0
        assert metric.improvement_needed == pytest.approx(30.0)

    def test_improvement_needed_target_zero(self):
        metric = QualityMetric(
            metric_id="m1",
            metric_name="Test",
            category=QualityCategory.MAINTAINABILITY,
            current_value=5.0,
            target_value=0.0,
            score=70.0,
            weight=0.5,
            trend=QualityTrend.STABLE,
            trend_confidence=0.8,
            description="desc",
            unit="issues",
        )
        assert metric.improvement_needed == 0.0

    def test_improvement_needed_above_target(self):
        metric = QualityMetric(
            metric_id="m1",
            metric_name="Test",
            category=QualityCategory.MAINTAINABILITY,
            current_value=100.0,
            target_value=90.0,
            score=100.0,
            weight=0.5,
            trend=QualityTrend.IMPROVING,
            trend_confidence=0.9,
            description="desc",
            unit="score",
        )
        # (90 - 100) / 90 * 100 < 0 => max(0, ...) = 0
        assert metric.improvement_needed == 0.0


# ---------------------------------------------------------------------------
# QualityIssueAnalysis model tests
# ---------------------------------------------------------------------------


class TestQualityIssueAnalysis:
    def test_severity_score_no_issues(self):
        analysis = QualityIssueAnalysis(
            total_issues=0,
            issues_by_severity={},
            technical_debt_hours=0.0,
            debt_ratio=0.0,
        )
        assert analysis.severity_score == pytest.approx(100.0)

    def test_severity_score_with_issues(self):
        analysis = QualityIssueAnalysis(
            total_issues=3,
            issues_by_severity={"critical": 1, "medium": 2},
            technical_debt_hours=6.0,
            debt_ratio=0.03,
        )
        # weighted = 1*10 + 2*2 = 14 => max(0, 100-min(100,14)) = 86
        assert analysis.severity_score == pytest.approx(86.0)

    def test_severity_score_overflow(self):
        analysis = QualityIssueAnalysis(
            total_issues=50,
            issues_by_severity={"critical": 20},
            technical_debt_hours=80.0,
            debt_ratio=0.5,
        )
        # weighted = 20*10 = 200 => max(0, 100-min(100,200)) = 0
        assert analysis.severity_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# QualityRecommendation model tests
# ---------------------------------------------------------------------------


class TestQualityRecommendation:
    def test_impact_score_critical(self):
        rec = QualityRecommendation(
            recommendation_id="r1",
            title="Fix critical",
            description="desc",
            category=QualityCategory.RELIABILITY,
            priority=QualityPriority.CRITICAL,
            expected_improvement=20.0,
            confidence=1.0,
            effort_estimate="high",
        )
        # base=10, (10 + 20/10) * 1.0 = 12.0
        assert rec.impact_score == pytest.approx(12.0)

    def test_impact_score_low_confidence(self):
        rec = QualityRecommendation(
            recommendation_id="r2",
            title="Maybe do this",
            description="desc",
            category=QualityCategory.STYLE,
            priority=QualityPriority.LOW,
            expected_improvement=10.0,
            confidence=0.5,
            effort_estimate="low",
        )
        # base=4, (4 + 10/10) * 0.5 = 2.5
        assert rec.impact_score == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# QualityAnalysisResult model tests
# ---------------------------------------------------------------------------


class TestQualityAnalysisResult:
    def _make_result(
        self, score: float, blockers: list[str] | None = None
    ) -> QualityAnalysisResult:
        return QualityAnalysisResult(
            analysis_id="a1",
            total_files_analyzed=1,
            total_tests_analyzed=0,
            overall_quality_score=score,
            issue_analysis=QualityIssueAnalysis(
                total_issues=0,
                technical_debt_hours=0.0,
                debt_ratio=0.0,
            ),
            quality_trend=QualityTrend.STABLE,
            blockers=blockers or [],
        )

    @pytest.mark.parametrize(
        "score,grade",
        [
            (95.0, "A"),
            (90.0, "A"),
            (85.0, "B"),
            (80.0, "B"),
            (75.0, "C"),
            (70.0, "C"),
            (65.0, "D"),
            (60.0, "D"),
            (55.0, "F"),
            (0.0, "F"),
        ],
    )
    def test_quality_grade(self, score, grade):
        result = self._make_result(score)
        assert result.quality_grade == grade

    def test_has_blockers_false(self):
        result = self._make_result(90.0)
        assert result.has_blockers is False

    def test_has_blockers_true(self):
        result = self._make_result(50.0, blockers=["critical issue"])
        assert result.has_blockers is True

    def test_serialize_datetime(self):
        result = self._make_result(80.0)
        data = result.model_dump()
        assert isinstance(data["analyzed_at"], str)

    def test_serialize_datetime_none(self):
        result = self._make_result(80.0)
        # Call the serializer directly with None
        assert result.serialize_datetime(None) is None


# ---------------------------------------------------------------------------
# QualityMetricsAnalyzer tests
# ---------------------------------------------------------------------------


class TestQualityMetricsAnalyzer:
    def test_init_default_config(self):
        analyzer = QualityMetricsAnalyzer()
        assert analyzer.config["min_quality_score"] == 70.0
        assert "metric_weights" in analyzer.config
        assert "severity_weights" in analyzer.config
        assert "debt_estimation" in analyzer.config

    def test_init_custom_config(self):
        custom = {
            "min_quality_score": 90.0,
            "metric_weights": {},
            "severity_weights": {},
            "debt_estimation": {},
        }
        analyzer = QualityMetricsAnalyzer(config=custom)
        assert analyzer.config["min_quality_score"] == 90.0

    def test_factory_function(self):
        analyzer = create_quality_metrics_analyzer()
        assert isinstance(analyzer, QualityMetricsAnalyzer)

    def test_factory_function_with_config(self):
        config = {"min_quality_score": 50.0}
        analyzer = create_quality_metrics_analyzer(config=config)
        assert analyzer.config["min_quality_score"] == 50.0


# ---------------------------------------------------------------------------
# _categorize_issue
# ---------------------------------------------------------------------------


class TestCategorizeIssue:
    def setup_method(self):
        self.analyzer = QualityMetricsAnalyzer()

    def test_security_keyword_in_message(self):
        issue = _make_issue(message="SQL injection vulnerability detected")
        assert self.analyzer._categorize_issue(issue) == QualityCategory.SECURITY

    def test_security_tool_bandit(self):
        issue = _make_issue(tool="bandit", message="something")
        assert self.analyzer._categorize_issue(issue) == QualityCategory.SECURITY

    def test_security_auth_keyword(self):
        issue = _make_issue(message="auth bypass found")
        assert self.analyzer._categorize_issue(issue) == QualityCategory.SECURITY

    def test_performance_keyword(self):
        issue = _make_issue(message="slow database query detected")
        assert self.analyzer._categorize_issue(issue) == QualityCategory.PERFORMANCE

    def test_performance_optimization(self):
        issue = _make_issue(message="optimization needed")
        assert self.analyzer._categorize_issue(issue) == QualityCategory.PERFORMANCE

    def test_complexity_keyword(self):
        issue = _make_issue(message="cyclomatic complexity too high")
        assert self.analyzer._categorize_issue(issue) == QualityCategory.COMPLEXITY

    def test_documentation_keyword(self):
        issue = _make_issue(message="missing docstring for function")
        assert self.analyzer._categorize_issue(issue) == QualityCategory.DOCUMENTATION

    def test_style_keyword(self):
        issue = _make_issue(message="whitespace error")
        assert self.analyzer._categorize_issue(issue) == QualityCategory.STYLE

    def test_style_tool_black(self):
        issue = _make_issue(tool="ruff-format", message="reformatted")
        assert self.analyzer._categorize_issue(issue) == QualityCategory.STYLE

    def test_style_tool_eslint(self):
        issue = _make_issue(tool="eslint", message="some rule")
        assert self.analyzer._categorize_issue(issue) == QualityCategory.STYLE

    def test_style_tool_prettier(self):
        issue = _make_issue(tool="prettier", message="format issue")
        assert self.analyzer._categorize_issue(issue) == QualityCategory.STYLE

    def test_reliability_keyword(self):
        issue = _make_issue(message="null pointer error detected")
        assert self.analyzer._categorize_issue(issue) == QualityCategory.RELIABILITY

    def test_default_maintainability(self):
        issue = _make_issue(message="refactor this code block")
        assert self.analyzer._categorize_issue(issue) == QualityCategory.MAINTAINABILITY


# ---------------------------------------------------------------------------
# _calculate_technical_debt
# ---------------------------------------------------------------------------


class TestCalculateTechnicalDebt:
    def setup_method(self):
        self.analyzer = QualityMetricsAnalyzer()

    def test_no_issues(self):
        assert self.analyzer._calculate_technical_debt([]) == 0.0

    def test_single_critical_default_config(self):
        # Default config has keys like "critical_hours", not "critical",
        # so severity "critical" falls through to default 1.0
        issues = [_make_issue(severity="critical")]
        assert self.analyzer._calculate_technical_debt(issues) == pytest.approx(1.0)

    def test_with_matching_config_keys(self):
        # If config keys match severity names directly, lookup works
        analyzer = QualityMetricsAnalyzer(
            config={
                **self.analyzer.config,
                "debt_estimation": {
                    "critical": 4.0,
                    "high": 2.0,
                    "medium": 1.0,
                    "low": 0.5,
                    "info": 0.1,
                },
            }
        )
        issues = [
            _make_issue(severity="critical"),
            _make_issue(severity="high"),
            _make_issue(severity="medium"),
            _make_issue(severity="low"),
            _make_issue(severity="info"),
        ]
        # 4.0 + 2.0 + 1.0 + 0.5 + 0.1 = 7.6
        assert analyzer._calculate_technical_debt(issues) == pytest.approx(7.6)

    def test_unknown_severity_defaults(self):
        issues = [_make_issue(severity="unknown")]
        # unknown not in debt_estimation => default 1.0
        assert self.analyzer._calculate_technical_debt(issues) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _calculate_percentile_rank
# ---------------------------------------------------------------------------


class TestCalculatePercentileRank:
    def setup_method(self):
        self.analyzer = QualityMetricsAnalyzer()

    @pytest.mark.parametrize(
        "score,expected",
        [
            (95.0, 95),
            (90.0, 95),
            (85.0, 80),
            (80.0, 80),
            (75.0, 60),
            (70.0, 60),
            (65.0, 40),
            (60.0, 40),
            (55.0, 20),
            (0.0, 20),
        ],
    )
    def test_percentile(self, score, expected):
        assert self.analyzer._calculate_percentile_rank(score) == expected


# ---------------------------------------------------------------------------
# _analyze_issues (async)
# ---------------------------------------------------------------------------


class TestAnalyzeIssues:
    def setup_method(self):
        self.analyzer = QualityMetricsAnalyzer()

    async def test_empty_issues(self):
        result = await self.analyzer._analyze_issues([])
        assert result.total_issues == 0
        assert result.issues_by_severity == {}
        assert result.technical_debt_hours == 0.0
        assert result.debt_ratio == 0.0
        assert result.critical_issues == []
        assert result.high_priority_issues == []
        assert result.auto_fixable_issues == []
        assert result.manual_fix_issues == []

    async def test_categorizes_by_severity(self):
        issues = [
            _make_issue(severity="critical"),
            _make_issue(severity="critical"),
            _make_issue(severity="high"),
            _make_issue(severity="low"),
        ]
        result = await self.analyzer._analyze_issues(issues)
        assert result.issues_by_severity["critical"] == 2
        assert result.issues_by_severity["high"] == 1
        assert result.issues_by_severity["low"] == 1

    async def test_critical_and_high_priority_lists(self):
        issues = [
            _make_issue(severity="critical", message="crit1"),
            _make_issue(severity="high", message="high1"),
            _make_issue(severity="medium", message="med1"),
        ]
        result = await self.analyzer._analyze_issues(issues)
        assert len(result.critical_issues) == 1
        assert len(result.high_priority_issues) == 1
        assert result.critical_issues[0].message == "crit1"
        assert result.high_priority_issues[0].message == "high1"

    async def test_auto_fixable_issues(self):
        issues = [
            _make_issue(fix_confidence=0.9, message="auto-fix"),
            _make_issue(fix_confidence=0.5, message="manual"),
            _make_issue(fix_confidence=None, message="no confidence"),
        ]
        result = await self.analyzer._analyze_issues(issues)
        assert len(result.auto_fixable_issues) == 1
        assert result.auto_fixable_issues[0].message == "auto-fix"
        assert len(result.manual_fix_issues) == 2

    async def test_debt_ratio_capped(self):
        # 150 issues => debt_ratio = min(1.0, 150/100) = 1.0
        issues = [_make_issue() for _ in range(150)]
        result = await self.analyzer._analyze_issues(issues)
        assert result.debt_ratio == 1.0

    async def test_issues_by_tool(self):
        issues = [
            _make_issue(tool="ruff"),
            _make_issue(tool="ruff"),
            _make_issue(tool="ty"),
        ]
        result = await self.analyzer._analyze_issues(issues)
        assert result.issues_by_tool["ruff"] == 2
        assert result.issues_by_tool["ty"] == 1

    async def test_issues_by_category(self):
        issues = [
            _make_issue(message="SQL injection found"),  # security
            _make_issue(message="slow query"),  # performance
            _make_issue(message="refactor needed"),  # maintainability
        ]
        result = await self.analyzer._analyze_issues(issues)
        assert (
            QualityCategory.SECURITY in result.issues_by_category
            or "security" in result.issues_by_category
        )


# ---------------------------------------------------------------------------
# _calculate_quality_metrics (async)
# ---------------------------------------------------------------------------


class TestCalculateQualityMetrics:
    def setup_method(self):
        self.analyzer = QualityMetricsAnalyzer()

    async def test_returns_five_metrics(self):
        qr = _make_quality_result(overall_score=80.0)
        issue_analysis = QualityIssueAnalysis(
            total_issues=0,
            technical_debt_hours=0.0,
            debt_ratio=0.0,
        )
        metrics = await self.analyzer._calculate_quality_metrics([qr], issue_analysis)
        assert len(metrics) == 5
        ids = {m.metric_id for m in metrics}
        assert ids == {
            "overall_score",
            "issue_density",
            "technical_debt_ratio",
            "security_score",
            "maintainability_score",
        }

    async def test_overall_score_metric_value(self):
        qr1 = _make_quality_result(overall_score=80.0)
        qr2 = _make_quality_result(overall_score=60.0)
        issue_analysis = QualityIssueAnalysis(
            total_issues=0,
            technical_debt_hours=0.0,
            debt_ratio=0.0,
        )
        metrics = await self.analyzer._calculate_quality_metrics([qr1, qr2], issue_analysis)
        overall = next(m for m in metrics if m.metric_id == "overall_score")
        assert overall.current_value == pytest.approx(70.0)

    async def test_issue_density_zero_issues(self):
        qr = _make_quality_result()
        issue_analysis = QualityIssueAnalysis(
            total_issues=0,
            technical_debt_hours=0.0,
            debt_ratio=0.0,
        )
        metrics = await self.analyzer._calculate_quality_metrics([qr], issue_analysis)
        density = next(m for m in metrics if m.metric_id == "issue_density")
        assert density.current_value == 0.0
        assert density.score == pytest.approx(100.0)

    async def test_security_score_no_issues(self):
        qr = _make_quality_result()
        issue_analysis = QualityIssueAnalysis(
            total_issues=0,
            technical_debt_hours=0.0,
            debt_ratio=0.0,
        )
        metrics = await self.analyzer._calculate_quality_metrics([qr], issue_analysis)
        sec = next(m for m in metrics if m.metric_id == "security_score")
        assert sec.score == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# _calculate_overall_score (async)
# ---------------------------------------------------------------------------


class TestCalculateOverallScore:
    def setup_method(self):
        self.analyzer = QualityMetricsAnalyzer()

    async def test_empty_metrics(self):
        issue_analysis = QualityIssueAnalysis(
            total_issues=0,
            technical_debt_hours=0.0,
            debt_ratio=0.0,
        )
        score = await self.analyzer._calculate_overall_score([], issue_analysis)
        assert score == 0.0

    async def test_with_metrics_no_issues(self):
        metrics = [
            QualityMetric(
                metric_id="m1",
                metric_name="Test",
                category=QualityCategory.MAINTAINABILITY,
                current_value=80.0,
                target_value=90.0,
                score=80.0,
                weight=1.0,
                trend=QualityTrend.STABLE,
                trend_confidence=0.8,
                description="desc",
                unit="score",
            )
        ]
        issue_analysis = QualityIssueAnalysis(
            total_issues=0,
            technical_debt_hours=0.0,
            debt_ratio=0.0,
        )
        score = await self.analyzer._calculate_overall_score(metrics, issue_analysis)
        assert score == pytest.approx(80.0)

    async def test_penalties_for_critical_issues(self):
        metrics = [
            QualityMetric(
                metric_id="m1",
                metric_name="Test",
                category=QualityCategory.MAINTAINABILITY,
                current_value=80.0,
                target_value=90.0,
                score=80.0,
                weight=1.0,
                trend=QualityTrend.STABLE,
                trend_confidence=0.8,
                description="desc",
                unit="score",
            )
        ]
        crit_issue = _make_issue(severity="critical")
        high_issue = _make_issue(severity="high")
        issue_analysis = QualityIssueAnalysis(
            total_issues=2,
            issues_by_severity={"critical": 1, "high": 1},
            critical_issues=[crit_issue],
            high_priority_issues=[high_issue],
            technical_debt_hours=6.0,
            debt_ratio=0.02,
        )
        score = await self.analyzer._calculate_overall_score(metrics, issue_analysis)
        # 80 - 10 (crit) - 5 (high) = 65
        assert score == pytest.approx(65.0)

    async def test_score_clamped_to_zero(self):
        metrics = [
            QualityMetric(
                metric_id="m1",
                metric_name="Test",
                category=QualityCategory.MAINTAINABILITY,
                current_value=10.0,
                target_value=90.0,
                score=10.0,
                weight=1.0,
                trend=QualityTrend.STABLE,
                trend_confidence=0.8,
                description="desc",
                unit="score",
            )
        ]
        crit_issues = [_make_issue(severity="critical") for _ in range(5)]
        issue_analysis = QualityIssueAnalysis(
            total_issues=5,
            issues_by_severity={"critical": 5},
            critical_issues=crit_issues,
            technical_debt_hours=20.0,
            debt_ratio=0.05,
        )
        score = await self.analyzer._calculate_overall_score(metrics, issue_analysis)
        # 10 - 50 = -40, clamped to 0
        assert score == 0.0


# ---------------------------------------------------------------------------
# _analyze_trends (async)
# ---------------------------------------------------------------------------


class TestAnalyzeTrends:
    def setup_method(self):
        self.analyzer = QualityMetricsAnalyzer()

    async def test_single_result_stable(self):
        qr = _make_quality_result()
        trend, analysis = await self.analyzer._analyze_trends([qr])
        assert trend == QualityTrend.STABLE
        assert analysis == {}

    async def test_insufficient_data(self):
        """Two results but all in 'recent' window => insufficient data."""
        qr1 = _make_quality_result(overall_score=70.0, created_at=datetime(2025, 1, 1, tzinfo=UTC))
        qr2 = _make_quality_result(overall_score=80.0, created_at=datetime(2025, 1, 2, tzinfo=UTC))
        trend, analysis = await self.analyzer._analyze_trends([qr1, qr2])
        # With 2 results, recent_scores takes last min(5, 2)=2, earlier_scores is empty
        assert trend == QualityTrend.STABLE
        assert analysis.get("reason") == "insufficient_data"

    async def test_improving_trend(self):
        # Need >5 results so earlier_scores is non-empty
        results = []
        for i in range(10):
            results.append(
                _make_quality_result(
                    overall_score=50.0 + i * 5,  # 50, 55, 60, ... 95
                    created_at=datetime(2025, 1, i + 1, tzinfo=UTC),
                )
            )
        trend, analysis = await self.analyzer._analyze_trends(results)
        assert trend == QualityTrend.IMPROVING
        assert analysis["score_change"] > 0

    async def test_degrading_trend(self):
        results = []
        for i in range(10):
            results.append(
                _make_quality_result(
                    overall_score=95.0 - i * 5,  # 95, 90, 85, ... 50
                    created_at=datetime(2025, 1, i + 1, tzinfo=UTC),
                )
            )
        trend, analysis = await self.analyzer._analyze_trends(results)
        assert trend == QualityTrend.DEGRADING
        assert analysis["score_change"] < 0

    async def test_stable_trend(self):
        results = []
        for i in range(10):
            results.append(
                _make_quality_result(
                    overall_score=75.0 + (i % 2) * 0.5,  # tiny oscillation
                    created_at=datetime(2025, 1, i + 1, tzinfo=UTC),
                )
            )
        trend, analysis = await self.analyzer._analyze_trends(results)
        assert trend == QualityTrend.STABLE


# ---------------------------------------------------------------------------
# _check_quality_gates (async)
# ---------------------------------------------------------------------------


class TestCheckQualityGates:
    def setup_method(self):
        self.analyzer = QualityMetricsAnalyzer()

    async def test_all_gates_pass(self):
        issue_analysis = QualityIssueAnalysis(
            total_issues=0,
            technical_debt_hours=0.0,
            debt_ratio=0.01,
        )
        gates = await self.analyzer._check_quality_gates(90.0, issue_analysis)
        assert gates["min_quality_score"] is True
        assert gates["no_critical_issues"] is True
        assert gates["limited_high_issues"] is True
        assert gates["technical_debt_ratio"] is True
        assert gates["no_security_issues"] is True

    async def test_quality_score_gate_fails(self):
        issue_analysis = QualityIssueAnalysis(
            total_issues=0,
            technical_debt_hours=0.0,
            debt_ratio=0.01,
        )
        gates = await self.analyzer._check_quality_gates(50.0, issue_analysis)
        assert gates["min_quality_score"] is False

    async def test_critical_issues_gate_fails(self):
        crit = _make_issue(severity="critical")
        issue_analysis = QualityIssueAnalysis(
            total_issues=1,
            issues_by_severity={"critical": 1},
            critical_issues=[crit],
            technical_debt_hours=4.0,
            debt_ratio=0.01,
        )
        gates = await self.analyzer._check_quality_gates(90.0, issue_analysis)
        assert gates["no_critical_issues"] is False

    async def test_high_issues_gate_fails(self):
        highs = [_make_issue(severity="high") for _ in range(10)]
        issue_analysis = QualityIssueAnalysis(
            total_issues=10,
            issues_by_severity={"high": 10},
            high_priority_issues=highs,
            technical_debt_hours=20.0,
            debt_ratio=0.1,
        )
        gates = await self.analyzer._check_quality_gates(90.0, issue_analysis)
        assert gates["limited_high_issues"] is False

    async def test_debt_ratio_gate_fails(self):
        issue_analysis = QualityIssueAnalysis(
            total_issues=50,
            technical_debt_hours=50.0,
            debt_ratio=0.5,
        )
        gates = await self.analyzer._check_quality_gates(90.0, issue_analysis)
        assert gates["technical_debt_ratio"] is False

    async def test_security_gate_fails(self):
        sec_issue = _make_issue(severity="critical", message="SQL injection vulnerability")
        issue_analysis = QualityIssueAnalysis(
            total_issues=1,
            issues_by_severity={"critical": 1},
            critical_issues=[sec_issue],
            technical_debt_hours=4.0,
            debt_ratio=0.01,
        )
        gates = await self.analyzer._check_quality_gates(90.0, issue_analysis)
        assert gates["no_security_issues"] is False


# ---------------------------------------------------------------------------
# _identify_blockers (async)
# ---------------------------------------------------------------------------


class TestIdentifyBlockers:
    def setup_method(self):
        self.analyzer = QualityMetricsAnalyzer()

    async def test_no_blockers(self):
        issue_analysis = QualityIssueAnalysis(
            total_issues=0,
            technical_debt_hours=0.0,
            debt_ratio=0.0,
        )
        gates = {"min_quality_score": True, "no_critical_issues": True}
        blockers = await self.analyzer._identify_blockers(issue_analysis, gates)
        assert blockers == []

    async def test_critical_issues_blocker(self):
        crit = _make_issue(severity="critical")
        issue_analysis = QualityIssueAnalysis(
            total_issues=1,
            issues_by_severity={"critical": 1},
            critical_issues=[crit],
            technical_debt_hours=4.0,
            debt_ratio=0.01,
        )
        gates = {"no_critical_issues": False}
        blockers = await self.analyzer._identify_blockers(issue_analysis, gates)
        assert any("critical" in b.lower() for b in blockers)

    async def test_security_blocker(self):
        sec_issue = _make_issue(severity="critical", message="SQL injection vulnerability")
        issue_analysis = QualityIssueAnalysis(
            total_issues=1,
            issues_by_severity={"critical": 1},
            critical_issues=[sec_issue],
            technical_debt_hours=4.0,
            debt_ratio=0.01,
        )
        gates = {"no_security_issues": False}
        blockers = await self.analyzer._identify_blockers(issue_analysis, gates)
        assert any("security" in b.lower() for b in blockers)

    async def test_failed_gates_blocker(self):
        issue_analysis = QualityIssueAnalysis(
            total_issues=0,
            technical_debt_hours=0.0,
            debt_ratio=0.0,
        )
        gates = {"min_quality_score": False, "technical_debt_ratio": False}
        blockers = await self.analyzer._identify_blockers(issue_analysis, gates)
        assert any("quality gates failed" in b.lower() for b in blockers)


# ---------------------------------------------------------------------------
# _compare_with_benchmarks (async)
# ---------------------------------------------------------------------------


class TestCompareWithBenchmarks:
    def setup_method(self):
        self.analyzer = QualityMetricsAnalyzer()

    async def test_above_all_benchmarks(self):
        issue_analysis = QualityIssueAnalysis(
            total_issues=0,
            technical_debt_hours=0.0,
            debt_ratio=0.0,
        )
        result = await self.analyzer._compare_with_benchmarks(95.0, [], issue_analysis)
        assert result["overall_score_vs_industry"] == pytest.approx(20.0)
        assert result["overall_score_vs_best_practice"] == pytest.approx(10.0)
        assert result["overall_score_vs_enterprise"] == pytest.approx(15.0)
        assert result["improvement_to_industry"] == 0
        assert result["improvement_to_best_practice"] == 0
        assert result["percentile_rank"] == 95

    async def test_below_all_benchmarks(self):
        issue_analysis = QualityIssueAnalysis(
            total_issues=0,
            technical_debt_hours=0.0,
            debt_ratio=0.0,
        )
        result = await self.analyzer._compare_with_benchmarks(50.0, [], issue_analysis)
        assert result["overall_score_vs_industry"] == pytest.approx(-25.0)
        assert result["improvement_to_industry"] == pytest.approx(25.0)
        assert result["improvement_to_best_practice"] == pytest.approx(35.0)
        assert result["percentile_rank"] == 20


# ---------------------------------------------------------------------------
# _generate_recommendations (async)
# ---------------------------------------------------------------------------


class TestGenerateRecommendations:
    def setup_method(self):
        self.analyzer = QualityMetricsAnalyzer()

    async def test_no_recommendations_clean(self):
        issue_analysis = QualityIssueAnalysis(
            total_issues=0,
            technical_debt_hours=0.0,
            debt_ratio=0.0,
        )
        recs = await self.analyzer._generate_recommendations([], issue_analysis, [])
        assert recs == []

    async def test_critical_issues_recommendation(self):
        crit = _make_issue(severity="critical", message="bad thing")
        issue_analysis = QualityIssueAnalysis(
            total_issues=1,
            issues_by_severity={"critical": 1},
            critical_issues=[crit],
            technical_debt_hours=4.0,
            debt_ratio=0.01,
        )
        recs = await self.analyzer._generate_recommendations([], issue_analysis, [])
        assert any(r.recommendation_id == "critical_issues" for r in recs)

    async def test_auto_fixable_recommendation(self):
        auto = _make_issue(fix_confidence=0.9, message="auto fixable")
        issue_analysis = QualityIssueAnalysis(
            total_issues=1,
            auto_fixable_issues=[auto],
            technical_debt_hours=0.5,
            debt_ratio=0.01,
        )
        recs = await self.analyzer._generate_recommendations([], issue_analysis, [])
        assert any(r.recommendation_id == "auto_fixable_issues" for r in recs)

    async def test_technical_debt_recommendation(self):
        issue_analysis = QualityIssueAnalysis(
            total_issues=50,
            technical_debt_hours=50.0,
            debt_ratio=0.5,  # way above 0.05 threshold
        )
        recs = await self.analyzer._generate_recommendations([], issue_analysis, [])
        assert any(r.recommendation_id == "technical_debt" for r in recs)

    async def test_security_recommendation(self):
        sec = _make_issue(severity="critical", message="SQL injection vulnerability")
        issue_analysis = QualityIssueAnalysis(
            total_issues=1,
            issues_by_severity={"critical": 1},
            critical_issues=[sec],
            technical_debt_hours=4.0,
            debt_ratio=0.01,
        )
        recs = await self.analyzer._generate_recommendations([], issue_analysis, [])
        assert any(r.recommendation_id == "security_issues" for r in recs)

    async def test_documentation_recommendation(self):
        doc = _make_issue(severity="high", message="missing docstring for method")
        issue_analysis = QualityIssueAnalysis(
            total_issues=1,
            issues_by_severity={"high": 1},
            high_priority_issues=[doc],
            technical_debt_hours=2.0,
            debt_ratio=0.01,
        )
        recs = await self.analyzer._generate_recommendations([], issue_analysis, [])
        assert any(r.recommendation_id == "documentation" for r in recs)

    async def test_recommendations_sorted_by_impact(self):
        crit = _make_issue(severity="critical", message="bad")
        auto = _make_issue(fix_confidence=0.9, message="auto")
        issue_analysis = QualityIssueAnalysis(
            total_issues=2,
            issues_by_severity={"critical": 1},
            critical_issues=[crit],
            auto_fixable_issues=[auto],
            technical_debt_hours=4.5,
            debt_ratio=0.5,
        )
        recs = await self.analyzer._generate_recommendations([], issue_analysis, [])
        # Should be sorted by impact_score descending
        scores = [r.impact_score for r in recs]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# analyze_quality_metrics (full integration-style async test)
# ---------------------------------------------------------------------------


class TestAnalyzeQualityMetricsFull:
    def setup_method(self):
        self.analyzer = QualityMetricsAnalyzer()

    async def test_basic_clean_analysis(self):
        qr = _make_quality_result(overall_score=90.0, language="python")
        result = await self.analyzer.analyze_quality_metrics([qr])
        assert isinstance(result, QualityAnalysisResult)
        assert result.total_files_analyzed == 1
        assert result.total_tests_analyzed == 0
        assert "python" in result.languages_analyzed
        assert result.overall_quality_score >= 0
        assert result.quality_grade in ("A", "B", "C", "D", "F")
        assert isinstance(result.quality_metrics, list)
        assert isinstance(result.issue_analysis, QualityIssueAnalysis)
        assert isinstance(result.recommendations, list)
        assert isinstance(result.quality_gates_passed, dict)

    async def test_analysis_with_issues(self):
        issues = [
            _make_issue(severity="critical", message="SQL injection vulnerability"),
            _make_issue(severity="high", message="error handling missing"),
            _make_issue(severity="medium", message="complexity too high"),
            _make_issue(severity="low", fix_confidence=0.95, message="style issue"),
        ]
        qr = _make_quality_result(overall_score=60.0, issues=issues)
        result = await self.analyzer.analyze_quality_metrics([qr])
        assert result.issue_analysis.total_issues == 4
        assert len(result.blockers) > 0  # should have blockers due to critical
        assert result.has_blockers is True

    async def test_analysis_with_test_results(self):
        from models.testing_models import TestResult, TestSecurityLevel

        qr = _make_quality_result(overall_score=85.0)
        tr = TestResult(
            test_id="t1",
            success=True,
            stdout="OK",
            stderr="",
            exit_code=0,
            execution_time=1.0,
            created_at=datetime.now(UTC),
            language="python",
            security_level=TestSecurityLevel.MEDIUM,
        )
        result = await self.analyzer.analyze_quality_metrics([qr], test_results=[tr])
        assert result.total_tests_analyzed == 1

    async def test_analysis_multiple_languages(self):
        qr1 = _make_quality_result(overall_score=85.0, language="python")
        qr2 = _make_quality_result(overall_score=80.0, language="javascript")
        result = await self.analyzer.analyze_quality_metrics([qr1, qr2])
        assert "python" in result.languages_analyzed
        assert "javascript" in result.languages_analyzed
        assert result.total_files_analyzed == 2

    async def test_analysis_benchmark_present(self):
        qr = _make_quality_result(overall_score=75.0)
        result = await self.analyzer.analyze_quality_metrics([qr])
        assert "overall_score_vs_industry" in result.benchmark_comparison
        assert "percentile_rank" in result.benchmark_comparison
