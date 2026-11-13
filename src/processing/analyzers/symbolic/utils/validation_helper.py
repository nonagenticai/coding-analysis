"""Validation helper utilities for symbolic analysis."""

from __future__ import annotations

import tempfile
from typing import IO, Any

from models.static_analysis_models import (
    FormalVerificationResult,
    SecurityIssue,
    StaticAnalysisValidationResult,
    SymbolicExecutionResult,
    TaintAnalysisResult,
    ValidationType,
)


class ValidationHelper:
    """Helper methods for validation operations."""

    def create_validation_result(
        self,
        file_path: str,
        validation_types: list[ValidationType],
        symbolic_result: dict[str, Any],
        taint_result: dict[str, Any],
        formal_result: dict[str, Any],
    ) -> StaticAnalysisValidationResult:
        """Create a validation result from analysis outputs."""
        # Parse symbolic execution result
        symbolic_execution = None
        if ValidationType.SYMBOLIC_EXECUTION in validation_types:
            symbolic_execution = SymbolicExecutionResult(
                status=symbolic_result.get("status", "unknown"),
                edge_cases_found=symbolic_result.get("edge_cases", []),
                assertion_violations=symbolic_result.get("violations", []),
            )

        # Parse taint analysis result
        taint_analysis = None
        if ValidationType.TAINT_ANALYSIS in validation_types:
            security_issues = []
            if taint_result.get("issues"):
                for issue in taint_result["issues"]:
                    security_issues.append(
                        SecurityIssue(
                            type=issue.get("type", "unknown"),
                            severity=issue.get("severity", "low"),
                            location=f"line {issue.get('line', 0)}, column {issue.get('column', 0)}",
                            flow_path=issue.get("flow_path", []),
                            recommendation=issue.get("recommendation", ""),
                        )
                    )

            taint_analysis = TaintAnalysisResult(
                status=taint_result.get("status", "unknown"),
                security_issues=security_issues,
            )

        # Parse formal verification result
        formal_verification = None
        if ValidationType.FORMAL_VERIFICATION in validation_types:
            formal_verification = FormalVerificationResult(
                status=formal_result.get("status", "unknown"),
                properties_verified=formal_result.get("properties", []),
                coverage=formal_result.get("coverage", 0.0),
            )

        return StaticAnalysisValidationResult(
            file_path=file_path,
            symbolic_execution=symbolic_execution,
            taint_analysis=taint_analysis,
            formal_verification=formal_verification,
        )

    def create_temp_file(self, content: str, suffix: str = ".py") -> IO[str]:
        """Create a temporary file with content."""
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        )
        temp_file.write(content)
        temp_file.flush()
        return temp_file
