"""Safety score calculator for symbolic analysis."""

from __future__ import annotations

from models.static_analysis_models import StaticAnalysisValidationResult


class SafetyCalculator:
    """Calculates safety scores for validation results."""

    def calculate_score(self, validation_result: StaticAnalysisValidationResult) -> float:
        """Calculate safety score for validation result."""
        score = 1.0

        # Reduce score for symbolic execution issues
        if validation_result.symbolic_execution:
            violations = len(validation_result.symbolic_execution.assertion_violations)
            if violations > 0:
                score -= min(0.3, violations * 0.1)

        # Reduce score for security issues
        if validation_result.taint_analysis:
            security_issues = validation_result.taint_analysis.security_issues
            for issue in security_issues:
                if issue.severity == "critical":
                    score -= 0.4
                elif issue.severity == "high":
                    score -= 0.2
                elif issue.severity == "medium":
                    score -= 0.1

        # Bonus for formal verification
        if validation_result.formal_verification:
            if validation_result.formal_verification.status == "passed":
                score += 0.1

        return max(0.0, min(1.0, score))

    def check_blocking_issues(self, validation_result: StaticAnalysisValidationResult) -> list[str]:
        """Check for blocking issues that prevent deployment."""
        blocking = []

        # Check for critical security issues
        if validation_result.taint_analysis:
            for issue in validation_result.taint_analysis.security_issues:
                if issue.severity in ["critical", "high"]:
                    blocking.append(f"{validation_result.file_path}: {issue.type}")

        # Check for assertion violations
        if validation_result.symbolic_execution:
            violations = validation_result.symbolic_execution.assertion_violations
            if violations:
                blocking.append(f"{validation_result.file_path}: assertion violations found")

        return blocking
