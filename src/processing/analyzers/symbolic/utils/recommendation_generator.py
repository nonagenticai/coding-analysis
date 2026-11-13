"""Recommendation generator for symbolic analysis."""

from __future__ import annotations

from models.static_analysis_models import StaticAnalysisValidationResult


class SymbolicRecommendationGenerator:
    """Generates recommendations based on validation results."""

    def get_security_recommendation(self, issue_type: str) -> str:
        """Get security recommendation for issue type."""
        recommendations = {
            "taint_flow": "Sanitize user input before processing",
            "injection": "Use parameterized queries and input validation",
            "xss": "Escape output and validate input",
            "csrf": "Implement CSRF tokens",
            "sql": "Use parameterized queries to prevent SQL injection",
            "command": "Use subprocess with shell=False and validate input",
            "path": "Validate and sanitize file paths to prevent traversal",
            "default": "Review code for security best practices",
        }

        return recommendations.get(issue_type.lower(), recommendations["default"])

    def generate_recommendations(
        self,
        validation_results: list[StaticAnalysisValidationResult],
        overall_safety_score: float,
        blocking_issues: list[str],
    ) -> list[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if blocking_issues:
            recommendations.append("Address critical security issues before deployment")

        if overall_safety_score < 0.7:
            recommendations.append("Increase test coverage for modified functions")
            recommendations.append("Add input validation and error handling")

        if overall_safety_score < 0.5:
            recommendations.append("Consider code review before proceeding")

        # Check for common patterns
        has_taint_issues = any(
            result.taint_analysis and result.taint_analysis.security_issues
            for result in validation_results
            if isinstance(result, StaticAnalysisValidationResult)
        )

        if has_taint_issues:
            recommendations.append("Implement input sanitization and validation")

        # Check for formal verification failures
        has_verification_failures = any(
            result.formal_verification and result.formal_verification.status != "passed"
            for result in validation_results
        )

        if has_verification_failures:
            recommendations.append("Review formal verification violations")

        if not recommendations:
            recommendations.append("Code changes appear safe for deployment")

        return recommendations
