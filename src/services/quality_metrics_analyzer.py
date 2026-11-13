"""Quality Metrics Analyzer for AI-powered test insights.

This module provides comprehensive quality analysis for test code and test results,
identifying quality issues, technical debt, and improvement opportunities.
It integrates with quality analysis tools and provides intelligent recommendations
for improving test and code quality.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid7

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer

from models.testing_models import QualityIssue, QualityResult, TestResult

logger = logging.getLogger(__name__)


class QualityCategory(StrEnum):
    """Categories of quality metrics."""

    MAINTAINABILITY = "maintainability"
    RELIABILITY = "reliability"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    COMPLEXITY = "complexity"
    COVERAGE = "coverage"
    STYLE = "style"


class QualityPriority(StrEnum):
    """Priority levels for quality issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class QualityTrend(StrEnum):
    """Quality trend directions."""

    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"


class QualityMetric(BaseModel):
    """Individual quality metric."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Metric identification
    metric_id: str = Field(description="Unique metric identifier")
    metric_name: str = Field(description="Human-readable metric name")
    category: QualityCategory = Field(description="Quality category")

    # Metric values
    current_value: float = Field(description="Current metric value")
    target_value: float = Field(description="Target metric value")
    baseline_value: float | None = Field(None, description="Baseline value for comparison")

    # Scoring
    score: float = Field(ge=0.0, le=100.0, description="Quality score (0-100)")
    weight: float = Field(ge=0.0, le=1.0, description="Weight in overall quality calculation")

    # Trend analysis
    trend: QualityTrend = Field(description="Trend direction")
    trend_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in trend analysis")

    # Supporting data
    description: str = Field(description="Metric description")
    unit: str = Field(description="Unit of measurement")

    @computed_field
    @property
    def weighted_score(self) -> float:
        """Calculate weighted score."""
        return self.score * self.weight

    @computed_field
    @property
    def improvement_needed(self) -> float:
        """Calculate improvement needed to reach target."""
        if self.target_value == 0:
            return 0.0
        return max(0.0, (self.target_value - self.current_value) / self.target_value * 100)


class QualityIssueAnalysis(BaseModel):
    """Analysis of quality issues."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Issue categorization
    total_issues: int = Field(description="Total number of issues")
    issues_by_severity: dict[str, int] = Field(
        default_factory=dict, description="Issues grouped by severity"
    )
    issues_by_category: dict[QualityCategory, int] = Field(
        default_factory=dict, description="Issues grouped by category"
    )
    issues_by_tool: dict[str, int] = Field(
        default_factory=dict, description="Issues grouped by tool"
    )

    # Issue trends
    new_issues: int = Field(default=0, description="New issues since last analysis")
    resolved_issues: int = Field(default=0, description="Resolved issues since last analysis")
    recurring_issues: int = Field(default=0, description="Recurring issues")

    # Priority analysis
    critical_issues: list[QualityIssue] = Field(
        default_factory=list,
        description="Critical issues requiring immediate attention",
    )
    high_priority_issues: list[QualityIssue] = Field(
        default_factory=list, description="High priority issues"
    )

    # Fixability analysis
    auto_fixable_issues: list[QualityIssue] = Field(
        default_factory=list, description="Issues that can be automatically resolved"
    )
    manual_fix_issues: list[QualityIssue] = Field(
        default_factory=list, description="Issues requiring manual intervention"
    )

    # Technical debt estimation
    technical_debt_hours: float = Field(description="Estimated technical debt in hours")
    debt_ratio: float = Field(ge=0.0, le=1.0, description="Technical debt ratio")

    @computed_field
    @property
    def severity_score(self) -> float:
        """Calculate severity-weighted score."""
        weights = {
            "critical": 10.0,
            "high": 5.0,
            "medium": 2.0,
            "low": 1.0,
            "info": 0.5,
        }

        total_weighted = sum(
            count * weights.get(severity, 1.0)
            for severity, count in self.issues_by_severity.items()
        )

        return max(0.0, 100.0 - min(100.0, total_weighted))


class QualityRecommendation(BaseModel):
    """Quality improvement recommendation."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Recommendation details
    recommendation_id: str = Field(description="Unique recommendation ID")
    title: str = Field(description="Recommendation title")
    description: str = Field(description="Detailed description")
    category: QualityCategory = Field(description="Quality category")
    priority: QualityPriority = Field(description="Priority level")

    # Impact assessment
    expected_improvement: float = Field(
        ge=0.0, le=100.0, description="Expected quality improvement"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in recommendation")
    effort_estimate: str = Field(description="Effort estimate (low, medium, high)")

    # Implementation details
    implementation_steps: list[str] = Field(
        default_factory=list, description="Step-by-step implementation guide"
    )
    tools_required: list[str] = Field(
        default_factory=list, description="Tools required for implementation"
    )
    prerequisites: list[str] = Field(
        default_factory=list, description="Prerequisites for implementation"
    )

    # Risk assessment
    risks: list[str] = Field(
        default_factory=list, description="Potential risks and mitigation strategies"
    )
    benefits: list[str] = Field(default_factory=list, description="Expected benefits")

    # Supporting evidence
    supporting_issues: list[str] = Field(
        default_factory=list, description="Issues supporting this recommendation"
    )
    supporting_evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Supporting evidence and data for this recommendation",
    )
    affected_components: list[str] = Field(
        default_factory=list, description="Components affected by this recommendation"
    )

    @computed_field
    @property
    def impact_score(self) -> float:
        """Calculate impact score."""
        priority_weights = {
            QualityPriority.CRITICAL: 10.0,
            QualityPriority.HIGH: 8.0,
            QualityPriority.MEDIUM: 6.0,
            QualityPriority.LOW: 4.0,
            QualityPriority.INFO: 2.0,
        }

        base_score = priority_weights[self.priority]
        return (base_score + self.expected_improvement / 10) * self.confidence


class QualityAnalysisResult(BaseModel):
    """Result of comprehensive quality analysis."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    # Analysis metadata
    analysis_id: str = Field(description="Unique analysis ID")
    analyzed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When analysis was performed",
    )

    # Input data summary
    total_files_analyzed: int = Field(description="Total files analyzed")
    total_tests_analyzed: int = Field(description="Total tests analyzed")
    languages_analyzed: set[str] = Field(
        default_factory=set, description="Programming languages analyzed"
    )

    # Overall quality metrics
    overall_quality_score: float = Field(ge=0.0, le=100.0, description="Overall quality score")
    quality_metrics: list[QualityMetric] = Field(
        default_factory=list, description="Detailed quality metrics"
    )

    # Issue analysis
    issue_analysis: QualityIssueAnalysis = Field(description="Analysis of quality issues")

    # Quality trends
    quality_trend: QualityTrend = Field(description="Overall quality trend")
    trend_analysis: dict[str, Any] = Field(
        default_factory=dict, description="Detailed trend analysis"
    )

    # Recommendations
    recommendations: list[QualityRecommendation] = Field(
        default_factory=list, description="Quality improvement recommendations"
    )

    # Quality gates
    quality_gates_passed: dict[str, bool] = Field(
        default_factory=dict, description="Quality gates status"
    )
    blockers: list[str] = Field(
        default_factory=list, description="Quality blockers preventing release"
    )

    # Comparative analysis
    benchmark_comparison: dict[str, Any] = Field(
        default_factory=dict, description="Comparison with benchmarks"
    )

    @field_serializer("analyzed_at")
    def serialize_datetime(self, v: datetime | None) -> str | None:
        if v is None:
            return None
        return v.isoformat()

    @computed_field
    @property
    def quality_grade(self) -> str:
        """Calculate quality grade."""
        if self.overall_quality_score >= 90:
            return "A"
        elif self.overall_quality_score >= 80:
            return "B"
        elif self.overall_quality_score >= 70:
            return "C"
        elif self.overall_quality_score >= 60:
            return "D"
        else:
            return "F"

    @computed_field
    @property
    def has_blockers(self) -> bool:
        """Check if there are quality blockers."""
        return len(self.blockers) > 0


class QualityMetricsAnalyzer:
    """Comprehensive quality metrics analyzer."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the quality metrics analyzer."""
        self.config = config or self._get_default_config()
        logger.info("Quality Metrics Analyzer initialized")

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            # Quality thresholds
            "min_quality_score": 70.0,
            "critical_issue_threshold": 0,  # No critical issues allowed
            "high_issue_threshold": 5,  # Max 5 high-severity issues
            "debt_ratio_threshold": 0.05,  # 5% technical debt ratio
            # Metric weights
            "metric_weights": {
                "maintainability": 0.25,
                "reliability": 0.20,
                "security": 0.20,
                "performance": 0.15,
                "documentation": 0.10,
                "complexity": 0.10,
            },
            # Issue severity weights
            "severity_weights": {
                "critical": 10.0,
                "high": 5.0,
                "medium": 2.0,
                "low": 1.0,
                "info": 0.5,
            },
            # Technical debt estimation
            "debt_estimation": {
                "critical_hours": 4.0,
                "high_hours": 2.0,
                "medium_hours": 1.0,
                "low_hours": 0.5,
                "info_hours": 0.1,
            },
        }

    async def analyze_quality_metrics(
        self,
        quality_results: list[QualityResult],
        test_results: list[TestResult] | None = None,
    ) -> QualityAnalysisResult:
        """Perform comprehensive quality metrics analysis."""
        analysis_id = str(uuid7())

        logger.info(
            f"Starting quality analysis {analysis_id} for {len(quality_results)} quality results"
        )

        # Extract all issues
        all_issues = []
        for qr in quality_results:
            all_issues.extend(qr.issues)

        # Analyze issues
        issue_analysis = await self._analyze_issues(all_issues)

        # Calculate quality metrics
        quality_metrics = await self._calculate_quality_metrics(quality_results, issue_analysis)

        # Calculate overall quality score
        overall_score = await self._calculate_overall_score(quality_metrics, issue_analysis)

        # Analyze trends
        quality_trend, trend_analysis = await self._analyze_trends(quality_results)

        # Generate recommendations
        recommendations = await self._generate_recommendations(
            quality_metrics, issue_analysis, quality_results
        )

        # Check quality gates
        quality_gates = await self._check_quality_gates(overall_score, issue_analysis)

        # Identify blockers
        blockers = await self._identify_blockers(issue_analysis, quality_gates)

        # Benchmark comparison
        benchmark_comparison = await self._compare_with_benchmarks(
            overall_score, quality_metrics, issue_analysis
        )

        # Get analyzed languages
        languages = set()
        for qr in quality_results:
            languages.add(qr.language)

        # Create result
        result = QualityAnalysisResult(
            analysis_id=analysis_id,
            total_files_analyzed=len(quality_results),
            total_tests_analyzed=len(test_results) if test_results else 0,
            languages_analyzed=languages,
            overall_quality_score=overall_score,
            quality_metrics=quality_metrics,
            issue_analysis=issue_analysis,
            quality_trend=quality_trend,
            trend_analysis=trend_analysis,
            recommendations=recommendations,
            quality_gates_passed=quality_gates,
            blockers=blockers,
            benchmark_comparison=benchmark_comparison,
        )

        logger.info(f"Quality analysis {analysis_id} completed")

        return result

    async def _analyze_issues(self, issues: list[QualityIssue]) -> QualityIssueAnalysis:
        """Analyze quality issues comprehensively."""

        # Categorize issues
        issues_by_severity: dict[str, int] = defaultdict(int)
        issues_by_category: dict[str, int] = defaultdict(int)
        issues_by_tool: dict[str, int] = defaultdict(int)

        critical_issues = []
        high_priority_issues = []
        auto_fixable_issues = []
        manual_fix_issues = []

        for issue in issues:
            # By severity
            issues_by_severity[issue.severity] += 1

            # By tool
            issues_by_tool[issue.tool] += 1

            # By category (derive from tool and message)
            category = self._categorize_issue(issue)
            issues_by_category[category] += 1

            # Priority classification
            if issue.severity == "critical":
                critical_issues.append(issue)
            elif issue.severity == "high":
                high_priority_issues.append(issue)

            # Fixability analysis
            if issue.fix_confidence and issue.fix_confidence > 0.8:
                auto_fixable_issues.append(issue)
            else:
                manual_fix_issues.append(issue)

        # Calculate technical debt
        technical_debt_hours = self._calculate_technical_debt(issues)

        # Calculate debt ratio (simplified)
        total_issues = len(issues)
        debt_ratio = min(1.0, total_issues / 100.0)  # Simplified calculation

        return QualityIssueAnalysis(
            total_issues=total_issues,
            issues_by_severity=dict(issues_by_severity),
            issues_by_category=dict(issues_by_category),  # type: ignore[invalid-argument-type]  # ty: ignore[invalid-argument-type]
            issues_by_tool=dict(issues_by_tool),
            critical_issues=critical_issues,
            high_priority_issues=high_priority_issues,
            auto_fixable_issues=auto_fixable_issues,
            manual_fix_issues=manual_fix_issues,
            technical_debt_hours=technical_debt_hours,
            debt_ratio=debt_ratio,
        )

    def _categorize_issue(self, issue: QualityIssue) -> QualityCategory:
        """Categorize an issue based on its properties."""

        message_lower = issue.message.lower()
        tool_lower = issue.tool.lower()

        # Security issues
        if (
            any(
                keyword in message_lower
                for keyword in [
                    "security",
                    "vulnerability",
                    "injection",
                    "xss",
                    "csrf",
                    "auth",
                ]
            )
            or "bandit" in tool_lower
        ):
            return QualityCategory.SECURITY

        # Performance issues
        if any(
            keyword in message_lower
            for keyword in ["performance", "slow", "inefficient", "optimization"]
        ):
            return QualityCategory.PERFORMANCE

        # Complexity issues
        if any(
            keyword in message_lower
            for keyword in ["complexity", "complex", "cognitive", "cyclomatic"]
        ):
            return QualityCategory.COMPLEXITY

        # Documentation issues
        if any(
            keyword in message_lower
            for keyword in ["docstring", "documentation", "comment", "missing doc"]
        ):
            return QualityCategory.DOCUMENTATION

        # Style issues
        if any(
            keyword in message_lower
            for keyword in ["style", "format", "whitespace", "naming", "convention"]
        ) or tool_lower in ["ruff-format", "prettier", "eslint"]:
            return QualityCategory.STYLE

        # Reliability issues
        if any(
            keyword in message_lower
            for keyword in ["error", "exception", "null", "undefined", "assertion"]
        ):
            return QualityCategory.RELIABILITY

        # Default to maintainability
        return QualityCategory.MAINTAINABILITY

    def _calculate_technical_debt(self, issues: list[QualityIssue]) -> float:
        """Calculate technical debt in hours."""

        total_hours = 0.0
        debt_hours = self.config["debt_estimation"]

        for issue in issues:
            severity = issue.severity.lower()
            hours = debt_hours.get(severity, 1.0)
            total_hours += hours

        return total_hours

    async def _calculate_quality_metrics(
        self, quality_results: list[QualityResult], issue_analysis: QualityIssueAnalysis
    ) -> list[QualityMetric]:
        """Calculate comprehensive quality metrics."""

        metrics = []

        # Overall quality score metric
        avg_quality_score = sum(qr.overall_score for qr in quality_results) / len(quality_results)
        metrics.append(
            QualityMetric(
                metric_id="overall_score",
                metric_name="Overall Quality Score",
                category=QualityCategory.MAINTAINABILITY,
                current_value=avg_quality_score,
                target_value=90.0,
                score=avg_quality_score,
                weight=0.3,
                trend=QualityTrend.STABLE,
                trend_confidence=0.8,
                description="Average quality score across all analyses",
                unit="score",
            )
        )

        # Issue density metric
        total_issues = issue_analysis.total_issues
        issue_density = total_issues / len(quality_results) if quality_results else 0
        issue_density_score = max(0.0, 100.0 - issue_density * 2)  # 2 points per issue

        metrics.append(
            QualityMetric(
                metric_id="issue_density",
                metric_name="Issue Density",
                category=QualityCategory.RELIABILITY,
                current_value=issue_density,
                target_value=5.0,  # Target: max 5 issues per file
                score=issue_density_score,
                weight=0.2,
                trend=QualityTrend.STABLE,
                trend_confidence=0.7,
                description="Average number of issues per analyzed file",
                unit="issues/file",
            )
        )

        # Technical debt ratio
        debt_ratio = issue_analysis.debt_ratio
        debt_score = max(0.0, 100.0 - debt_ratio * 100)

        metrics.append(
            QualityMetric(
                metric_id="technical_debt_ratio",
                metric_name="Technical Debt Ratio",
                category=QualityCategory.MAINTAINABILITY,
                current_value=debt_ratio,
                target_value=0.05,  # Target: 5% debt ratio
                score=debt_score,
                weight=0.15,
                trend=QualityTrend.STABLE,
                trend_confidence=0.8,
                description="Ratio of technical debt to total codebase",
                unit="ratio",
            )
        )

        # Security score
        security_issues = sum(
            1
            for issue in issue_analysis.critical_issues + issue_analysis.high_priority_issues
            if self._categorize_issue(issue) == QualityCategory.SECURITY
        )
        security_score = max(0.0, 100.0 - security_issues * 20)  # 20 points per security issue

        metrics.append(
            QualityMetric(
                metric_id="security_score",
                metric_name="Security Score",
                category=QualityCategory.SECURITY,
                current_value=security_issues,
                target_value=0.0,
                score=security_score,
                weight=0.2,
                trend=QualityTrend.STABLE,
                trend_confidence=0.9,
                description="Security-related issues score",
                unit="issues",
            )
        )

        # Maintainability score
        maintainability_issues = sum(
            1
            for issue in issue_analysis.critical_issues + issue_analysis.high_priority_issues
            if self._categorize_issue(issue) == QualityCategory.MAINTAINABILITY
        )
        maintainability_score = max(0.0, 100.0 - maintainability_issues * 5)

        metrics.append(
            QualityMetric(
                metric_id="maintainability_score",
                metric_name="Maintainability Score",
                category=QualityCategory.MAINTAINABILITY,
                current_value=maintainability_issues,
                target_value=0.0,
                score=maintainability_score,
                weight=0.15,
                trend=QualityTrend.STABLE,
                trend_confidence=0.7,
                description="Maintainability-related issues score",
                unit="issues",
            )
        )

        return metrics

    async def _calculate_overall_score(
        self, metrics: list[QualityMetric], issue_analysis: QualityIssueAnalysis
    ) -> float:
        """Calculate overall quality score."""

        if not metrics:
            return 0.0

        # Calculate weighted average of metric scores
        total_weighted_score = sum(metric.weighted_score for metric in metrics)
        total_weight = sum(metric.weight for metric in metrics)

        base_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

        # Apply penalties for critical issues
        critical_penalty = len(issue_analysis.critical_issues) * 10
        high_penalty = len(issue_analysis.high_priority_issues) * 5

        final_score = base_score - critical_penalty - high_penalty

        return max(0.0, min(100.0, final_score))

    async def _analyze_trends(
        self, quality_results: list[QualityResult]
    ) -> tuple[QualityTrend, dict[str, Any]]:
        """Analyze quality trends."""

        if len(quality_results) < 2:
            return QualityTrend.STABLE, {}

        # Sort by creation time
        sorted_results = sorted(quality_results, key=lambda r: r.created_at)

        # Calculate trend in overall scores
        scores = [r.overall_score for r in sorted_results]

        # Simple trend analysis
        recent_scores = scores[-min(5, len(scores)) :]  # Last 5 results
        earlier_scores = scores[: -len(recent_scores)] if len(scores) > len(recent_scores) else []

        if not earlier_scores:
            return QualityTrend.STABLE, {"reason": "insufficient_data"}

        recent_avg = sum(recent_scores) / len(recent_scores)
        earlier_avg = sum(earlier_scores) / len(earlier_scores)

        change = recent_avg - earlier_avg

        if abs(change) < 2.0:  # Less than 2 points change
            trend = QualityTrend.STABLE
        elif change > 0:
            trend = QualityTrend.IMPROVING
        else:
            trend = QualityTrend.DEGRADING

        trend_analysis = {
            "score_change": change,
            "recent_average": recent_avg,
            "earlier_average": earlier_avg,
            "samples_analyzed": len(scores),
            "trend_strength": min(1.0, abs(change) / 10.0),
        }

        return trend, trend_analysis

    async def _generate_recommendations(
        self,
        metrics: list[QualityMetric],
        issue_analysis: QualityIssueAnalysis,
        quality_results: list[QualityResult],
    ) -> list[QualityRecommendation]:
        """Generate quality improvement recommendations."""

        recommendations = []

        # Critical issues recommendation
        if issue_analysis.critical_issues:
            recommendations.append(
                QualityRecommendation(
                    recommendation_id="critical_issues",
                    title="Address Critical Quality Issues",
                    description=f"Found {len(issue_analysis.critical_issues)} critical issues requiring immediate attention",
                    category=QualityCategory.RELIABILITY,
                    priority=QualityPriority.CRITICAL,
                    expected_improvement=20.0,
                    confidence=0.95,
                    effort_estimate="high",
                    implementation_steps=[
                        "Review all critical issues identified",
                        "Prioritize based on security and reliability impact",
                        "Implement fixes for each critical issue",
                        "Add validation to prevent recurrence",
                    ],
                    tools_required=["Static analysis tools", "Code review tools"],
                    prerequisites=[
                        "Development environment access",
                        "Code review process",
                    ],
                    risks=[
                        "Potential for introducing regressions",
                        "Time investment required",
                    ],
                    benefits=[
                        "Reduced security vulnerabilities",
                        "Improved reliability",
                        "Better maintainability",
                    ],
                    supporting_issues=[
                        issue.message for issue in issue_analysis.critical_issues[:5]
                    ],
                    affected_components=["core_functionality", "security_layer"],
                )
            )

        # Auto-fixable issues recommendation
        if issue_analysis.auto_fixable_issues:
            recommendations.append(
                QualityRecommendation(
                    recommendation_id="auto_fixable_issues",
                    title="Apply Automated Fixes",
                    description=f"Found {len(issue_analysis.auto_fixable_issues)} issues that can be automatically resolved",
                    category=QualityCategory.STYLE,
                    priority=QualityPriority.MEDIUM,
                    expected_improvement=15.0,
                    confidence=0.9,
                    effort_estimate="low",
                    implementation_steps=[
                        "Run automated formatters (ruff format, prettier, etc.)",
                        "Apply automated linting fixes",
                        "Review and commit automated changes",
                        "Update pre-commit hooks to prevent recurrence",
                    ],
                    tools_required=[
                        "Automated formatters",
                        "Linting tools",
                        "Pre-commit hooks",
                    ],
                    prerequisites=["CI/CD access", "Code formatting standards"],
                    risks=["Minor formatting conflicts", "Need for code review"],
                    benefits=[
                        "Consistent code style",
                        "Reduced manual effort",
                        "Improved readability",
                    ],
                    supporting_issues=[
                        issue.message for issue in issue_analysis.auto_fixable_issues[:5]
                    ],
                    affected_components=["code_style", "formatting"],
                )
            )

        # Technical debt recommendation
        if issue_analysis.debt_ratio > self.config["debt_ratio_threshold"]:
            recommendations.append(
                QualityRecommendation(
                    recommendation_id="technical_debt",
                    title="Reduce Technical Debt",
                    description=f"Technical debt ratio is {issue_analysis.debt_ratio:.1%}, exceeding {self.config['debt_ratio_threshold']:.1%} threshold",
                    category=QualityCategory.MAINTAINABILITY,
                    priority=QualityPriority.HIGH,
                    expected_improvement=25.0,
                    confidence=0.8,
                    effort_estimate="high",
                    implementation_steps=[
                        "Conduct technical debt audit",
                        "Prioritize debt items by impact and effort",
                        "Allocate dedicated refactoring time",
                        "Implement systematic debt reduction plan",
                    ],
                    tools_required=[
                        "Code analysis tools",
                        "Refactoring tools",
                        "Project management tools",
                    ],
                    prerequisites=[
                        "Team commitment",
                        "Management support",
                        "Clear refactoring guidelines",
                    ],
                    risks=[
                        "Temporary reduction in feature velocity",
                        "Potential for introducing bugs",
                    ],
                    benefits=[
                        "Improved maintainability",
                        "Faster development",
                        "Better code quality",
                    ],
                    supporting_issues=[],
                    affected_components=[
                        "architecture",
                        "code_structure",
                        "maintainability",
                    ],
                )
            )

        # Security issues recommendation
        security_issues = [
            issue
            for issue in issue_analysis.critical_issues + issue_analysis.high_priority_issues
            if self._categorize_issue(issue) == QualityCategory.SECURITY
        ]

        if security_issues:
            recommendations.append(
                QualityRecommendation(
                    recommendation_id="security_issues",
                    title="Address Security Vulnerabilities",
                    description=f"Found {len(security_issues)} security-related issues",
                    category=QualityCategory.SECURITY,
                    priority=QualityPriority.CRITICAL,
                    expected_improvement=30.0,
                    confidence=0.95,
                    effort_estimate="high",
                    implementation_steps=[
                        "Review all security issues immediately",
                        "Implement security fixes",
                        "Add security testing",
                        "Establish security review process",
                    ],
                    tools_required=[
                        "Security scanning tools",
                        "Penetration testing tools",
                    ],
                    prerequisites=["Security expertise", "Security review process"],
                    risks=["Potential security breaches", "Compliance issues"],
                    benefits=[
                        "Improved security posture",
                        "Reduced vulnerability risk",
                        "Better compliance",
                    ],
                    supporting_issues=[issue.message for issue in security_issues[:5]],
                    affected_components=[
                        "security_layer",
                        "authentication",
                        "data_protection",
                    ],
                )
            )

        # Documentation improvement
        doc_issues = [
            issue
            for issue in issue_analysis.critical_issues + issue_analysis.high_priority_issues
            if self._categorize_issue(issue) == QualityCategory.DOCUMENTATION
        ]

        if doc_issues:
            recommendations.append(
                QualityRecommendation(
                    recommendation_id="documentation",
                    title="Improve Code Documentation",
                    description=f"Found {len(doc_issues)} documentation-related issues",
                    category=QualityCategory.DOCUMENTATION,
                    priority=QualityPriority.MEDIUM,
                    expected_improvement=10.0,
                    confidence=0.8,
                    effort_estimate="medium",
                    implementation_steps=[
                        "Review and add missing docstrings",
                        "Improve code comments",
                        "Add API documentation",
                        "Establish documentation standards",
                    ],
                    tools_required=["Documentation generators", "Code review tools"],
                    prerequisites=["Documentation standards", "Team training"],
                    risks=[
                        "Additional maintenance overhead",
                        "Potential for outdated docs",
                    ],
                    benefits=[
                        "Better code understanding",
                        "Easier maintenance",
                        "Improved collaboration",
                    ],
                    supporting_issues=[issue.message for issue in doc_issues[:3]],
                    affected_components=["documentation", "code_comments", "api_docs"],
                )
            )

        # Sort recommendations by impact score
        recommendations.sort(key=lambda r: r.impact_score, reverse=True)

        return recommendations

    async def _check_quality_gates(
        self, overall_score: float, issue_analysis: QualityIssueAnalysis
    ) -> dict[str, bool]:
        """Check quality gates."""

        gates = {}

        # Minimum quality score gate
        gates["min_quality_score"] = overall_score >= self.config["min_quality_score"]

        # Critical issues gate
        gates["no_critical_issues"] = (
            len(issue_analysis.critical_issues) <= self.config["critical_issue_threshold"]
        )

        # High severity issues gate
        gates["limited_high_issues"] = (
            len(issue_analysis.high_priority_issues) <= self.config["high_issue_threshold"]
        )

        # Technical debt gate
        gates["technical_debt_ratio"] = (
            issue_analysis.debt_ratio <= self.config["debt_ratio_threshold"]
        )

        # Security gate
        security_issues = sum(
            1
            for issue in issue_analysis.critical_issues + issue_analysis.high_priority_issues
            if self._categorize_issue(issue) == QualityCategory.SECURITY
        )
        gates["no_security_issues"] = security_issues == 0

        return gates

    async def _identify_blockers(
        self, issue_analysis: QualityIssueAnalysis, quality_gates: dict[str, bool]
    ) -> list[str]:
        """Identify quality blockers."""

        blockers = []

        # Critical issues are always blockers
        if issue_analysis.critical_issues:
            blockers.append(
                f"{len(issue_analysis.critical_issues)} critical issues must be resolved"
            )

        # Security issues are blockers
        security_issues = sum(
            1
            for issue in issue_analysis.critical_issues + issue_analysis.high_priority_issues
            if self._categorize_issue(issue) == QualityCategory.SECURITY
        )
        if security_issues > 0:
            blockers.append(f"{security_issues} security issues must be addressed")

        # Failed quality gates
        failed_gates = [gate for gate, passed in quality_gates.items() if not passed]
        if failed_gates:
            blockers.append(f"Quality gates failed: {', '.join(failed_gates)}")

        return blockers

    async def _compare_with_benchmarks(
        self,
        overall_score: float,
        metrics: list[QualityMetric],
        issue_analysis: QualityIssueAnalysis,
    ) -> dict[str, Any]:
        """Compare with industry benchmarks."""

        benchmarks = {
            "industry_average_score": 75.0,
            "best_practice_score": 85.0,
            "enterprise_standard": 80.0,
        }

        comparison = {
            "overall_score_vs_industry": overall_score - benchmarks["industry_average_score"],
            "overall_score_vs_best_practice": overall_score - benchmarks["best_practice_score"],
            "overall_score_vs_enterprise": overall_score - benchmarks["enterprise_standard"],
            "percentile_rank": self._calculate_percentile_rank(overall_score),
            "improvement_to_industry": max(0, benchmarks["industry_average_score"] - overall_score),
            "improvement_to_best_practice": max(
                0, benchmarks["best_practice_score"] - overall_score
            ),
        }

        return comparison

    def _calculate_percentile_rank(self, score: float) -> int:
        """Calculate percentile rank based on score."""
        if score >= 90:
            return 95
        elif score >= 80:
            return 80
        elif score >= 70:
            return 60
        elif score >= 60:
            return 40
        else:
            return 20


# Factory function
def create_quality_metrics_analyzer(
    config: dict[str, Any] | None = None,
) -> QualityMetricsAnalyzer:
    """Create a quality metrics analyzer instance."""
    return QualityMetricsAnalyzer(config=config)
