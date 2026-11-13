"""
Core Analysis Functions

Implementation of all analysis types: basic, enhanced, AST, deep, security, and performance.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from processing.analyzers.deep import DeepCodeAnalyzer
from processing.analyzers.deep import analyzer as deep_analyzer_module
from processing.analyzers.python_analyzer import (
    PythonCodeProcessor as PythonAnalyzer,
)
from services.code_analysis import ast_validation_service
from services.code_analysis.ast_validation_service import ASTValidationService

from ...models.analysis_api_models import (
    AnalysisIssue,
    AnalysisMetrics,
    AnalysisPhase,
    AnalysisRequest,
    AnalysisStatus,
    ApiAnalysisResult,
)
from ...models.analysis_api_models import PydanticAIAnalysisType as AnalysisType

logger = logging.getLogger(__name__)


def _contains(needle: str) -> Callable[[str], int | None]:
    """Line check: 1-based column of *needle*, None when absent."""
    return lambda line: line.find(needle) + 1 if needle in line else None


def _unclosed_print(line: str) -> int | None:
    stripped = line.strip()
    return 1 if stripped.startswith("print(") and not stripped.endswith(")") else None


_CONSOLE_LOG = (
    _contains("console.log"),
    "quality",
    "low",
    "Console.log statement found - consider removing for production",
    "no-console",
)
#: THE language-specific ``basic_analysis`` rules: language -> (line check,
#: issue type, severity, message, rule id). Every language also gets the
#: language-agnostic line checks.
_BASIC_RULES: dict[str, tuple[Callable[[str], int | None], str, str, str, str]] = {
    "python": (
        _unclosed_print,
        "syntax",
        "medium",
        "Possible syntax error in print statement",
        "syntax-check",
    ),
    "javascript": _CONSOLE_LOG,
    "typescript": _CONSOLE_LOG,
    "java": (
        _contains("System.out.println"),
        "quality",
        "low",
        "System.out.println found - consider using proper logging",
        "no-println",
    ),
}
#: Languages ``basic_analysis`` has language-specific rules for.
BASIC_RULE_LANGUAGES: tuple[str, ...] = tuple(_BASIC_RULES)
#: ``enhanced_analysis`` runs ``PythonCodeProcessor``: Python only.
AUGMENTED_LANGUAGES: tuple[str, ...] = ("python",)


async def basic_analysis(
    request: AnalysisRequest, analysis_id: str, start_time: float
) -> ApiAnalysisResult:
    """Perform basic static analysis with multi-language support."""
    if not request.code:
        raise ValueError("Code content is required for basic analysis")

    issues = []
    lines = request.code.split("\n")
    lines_of_code = len([line for line in lines if line.strip()])

    # Language-agnostic basic checks
    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()

        # Check for TODO/FIXME comments (all languages)
        if any(marker in line_stripped.upper() for marker in ["TODO:", "FIXME:", "HACK:", "XXX:"]):
            issues.append(
                AnalysisIssue(
                    type="documentation",
                    severity="low",
                    message="Found unresolved TODO/FIXME comment",
                    line=i,
                    column=1,
                    file="input",
                    rule="todo-comment",
                )
            )

        # Check for very long lines (all languages)
        if len(line) > 120:
            issues.append(
                AnalysisIssue(
                    type="style",
                    severity="low",
                    message=f"Line exceeds 120 characters ({len(line)} chars)",
                    line=i,
                    column=121,
                    file="input",
                    rule="line-length",
                )
            )

        # Check for trailing whitespace
        if line.rstrip() != line:
            issues.append(
                AnalysisIssue(
                    type="style",
                    severity="low",
                    message="Trailing whitespace detected",
                    line=i,
                    column=len(line.rstrip()) + 1,
                    file="input",
                    rule="trailing-whitespace",
                )
            )

    # Language-specific basic checks
    language = request.language or "unknown"
    rule = _BASIC_RULES.get(language)
    if rule is not None:
        check, kind, severity, message, rule_id = rule
        for i, line in enumerate(lines, 1):
            column = check(line)
            if column is not None:
                issues.append(
                    AnalysisIssue(
                        type=kind,
                        severity=severity,
                        message=message,
                        line=i,
                        column=column,
                        file="input",
                        rule=rule_id,
                    )
                )

    # Calculate basic complexity estimate based on control flow keywords
    control_flow_keywords = {
        "python": ["if", "elif", "else", "for", "while", "try", "except", "finally"],
        "javascript": ["if", "else", "for", "while", "switch", "case", "try", "catch"],
        "typescript": ["if", "else", "for", "while", "switch", "case", "try", "catch"],
        "java": ["if", "else", "for", "while", "switch", "case", "try", "catch"],
        "csharp": [
            "if",
            "else",
            "for",
            "foreach",
            "while",
            "switch",
            "case",
            "try",
            "catch",
        ],
        "go": ["if", "else", "for", "switch", "case", "select"],
        "rust": ["if", "else", "match", "loop", "while", "for"],
        "cpp": ["if", "else", "for", "while", "switch", "case", "try", "catch"],
    }

    keywords = control_flow_keywords.get(language, control_flow_keywords["python"])
    complexity_count = 0
    for line in lines:
        line_lower = line.lower()
        for keyword in keywords:
            if f" {keyword} " in f" {line_lower} " or line_lower.startswith(f"{keyword} "):
                complexity_count += 1

    # Estimate complexity score (1 + number of control flow statements)
    estimated_complexity = 1 + complexity_count

    metrics = AnalysisMetrics(
        lines_of_code=lines_of_code,
        complexity=float(estimated_complexity),
        maintainability=max(0, 100 - (estimated_complexity * 2)),  # Simple maintainability estimate
        test_coverage=None,
        technical_debt=None,
    )

    return ApiAnalysisResult(
        analysis_type=request.analysis_type,
        success=True,
        issues=issues,
        metrics=metrics,
        suggestions=["Consider more detailed analysis for comprehensive insights"],
        execution_time=time.time() - start_time,
        metadata={
            "analysis_id": analysis_id,
            "analysis_scope": "basic",
            "language": language,
        },
    )


async def enhanced_analysis(
    request: AnalysisRequest, analysis_id: str, start_time: float
) -> ApiAnalysisResult:
    """Perform comprehensive AST-based analysis."""
    from services.analysis_status import AnalysisStatusService

    status_service = AnalysisStatusService()

    try:
        # Update status - initializing
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.RUNNING,
            progress=20.0,
            phase=AnalysisPhase.INITIALIZING,
        )

        if not request.code:
            raise ValueError("Code content is required for augmented analysis")

        # Initialize analyzer
        analyzer = PythonAnalyzer()
        issues: list[Any] = []
        suggestions: list[Any] = []

        # Update status - parsing
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.RUNNING,
            progress=30.0,
            phase=AnalysisPhase.PARSING,
        )

        # Perform analysis on code snippet
        result = await analyzer.analyze_snippet(request.code, "analysis_input")

        # Update status - extracting
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.RUNNING,
            progress=50.0,
            phase=AnalysisPhase.EXTRACTING,
        )

        # Process analysis results
        metrics = AnalysisMetrics(
            lines_of_code=0,
            complexity=0.0,
            maintainability=0.0,
            test_coverage=None,
            technical_debt=None,
        )

        if result.success and result.complexity_metrics:
            complexity = result.complexity_metrics
            metrics.lines_of_code = complexity.lines_of_code
            metrics.complexity = complexity.cyclomatic_complexity

            # Calculate maintainability index
            if complexity.lines_of_code > 0:
                # Basic maintainability calculation based on complexity and documentation
                doc_ratio = (
                    complexity.docstring_coverage / 100.0 if complexity.docstring_coverage else 0.0
                )
                complexity_factor = max(0, 100 - complexity.cyclomatic_complexity * 5)
                metrics.maintainability = min(100.0, complexity_factor * 0.7 + doc_ratio * 30)

            # Convert technical debt from complexity
            if complexity.cyclomatic_complexity > 10:
                metrics.technical_debt = (complexity.cyclomatic_complexity - 10) * 0.5

        # Update status - analyzing
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.RUNNING,
            progress=70.0,
            phase=AnalysisPhase.ANALYZING,
        )

        # Convert entities to issues
        if result.entities:
            entity_counts: dict[str, Any] = {}
            for entity in result.entities:
                entity_type = entity.type
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1

                # Generate suggestions based on entity analysis
                if entity.type == "function" and entity.metadata:
                    complexity = entity.metadata.get("complexity", 0)
                    if complexity > 10:
                        issues.append(
                            AnalysisIssue(
                                type="complexity",
                                severity="high",
                                message=(
                                    f"Function '{entity.name}' has high complexity ({complexity})"
                                ),
                                line=entity.line_number,
                                column=1,
                                file="analysis_input",
                                rule="complexity-check",
                            )
                        )
                        suggestions.append(
                            f"Consider refactoring function '{entity.name}' to reduce complexity"
                        )
                elif entity.type == "class" and not entity.docstring:
                    issues.append(
                        AnalysisIssue(
                            type="documentation",
                            severity="medium",
                            message=f"Class '{entity.name}' missing docstring",
                            line=entity.line_number,
                            column=1,
                            file="analysis_input",
                            rule="docstring-check",
                        )
                    )
                    suggestions.append(f"Add docstring to class '{entity.name}'")

        # Convert security patterns to issues
        if result.security_patterns:
            for pattern in result.security_patterns:
                severity_map = {
                    "high": "high",
                    "medium": "medium",
                    "low": "low",
                }
                issues.append(
                    AnalysisIssue(
                        type="security",
                        severity=severity_map.get(pattern.severity, "medium"),
                        message=pattern.description,
                        line=pattern.line_number,
                        column=1,
                        file="analysis_input",
                        rule=pattern.type,
                    )
                )
                if pattern.metadata and pattern.metadata.get("suggestion"):
                    suggestions.append(pattern.metadata["suggestion"])

        # Update status - generating report
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.RUNNING,
            progress=90.0,
            phase=AnalysisPhase.GENERATING_REPORT,
        )

        # Add general suggestions
        if request.language == "python":
            if metrics.lines_of_code > 0:
                if metrics.complexity > 5:
                    suggestions.append("Consider breaking down complex functions into smaller ones")
                if metrics.maintainability < 70:
                    suggestions.append("Update code documentation and reduce complexity")
                if not any("docstring" in s for s in suggestions):
                    suggestions.append("Add comprehensive docstrings to update maintainability")

        return ApiAnalysisResult(
            analysis_type=request.analysis_type,
            success=True,
            issues=issues,
            metrics=metrics,
            suggestions=suggestions,
            execution_time=time.time() - start_time,
            metadata={
                "analysis_id": analysis_id,
                "entity_count": len(result.entities) if result.entities else 0,
                "relationship_count": (len(result.relationships) if result.relationships else 0),
                "security_pattern_count": (
                    len(result.security_patterns) if result.security_patterns else 0
                ),
            },
        )

    except (RuntimeError, ValueError, ImportError, OSError) as e:
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.FAILED,
            details={"error": str(e), "phase": "comprehensive_analysis"},
        )
        # Fallback to basic analysis
        return await basic_analysis(request, analysis_id, start_time)


def _issue_text(text: str) -> str:
    """Make free text safe for ``AnalysisIssue`` (its fields refuse ``<>&``)."""
    return text.replace("&", " and ").replace("<", "(").replace(">", ")")


def _syntax_issues(ast_result: Any) -> list[AnalysisIssue]:
    """``syntax_errors`` of a failed parse as structured ``AnalysisIssue`` rows."""
    if ast_result.is_valid:
        return []
    details = list(getattr(ast_result, "syntax_error_details", None) or [])
    if details:
        return [
            AnalysisIssue(
                type="syntax",
                severity="critical",
                message=_issue_text(d.message or "invalid syntax"),
                line=d.line,
                column=max(1, d.column),
                file="analysis_input",
                rule="syntax-error",
            )
            for d in details
        ]
    # No location: the parse was not attempted or crashed (e.g. unsupported
    # language). Still a finding -- never a silent pass.
    return [
        AnalysisIssue(
            type="syntax",
            severity="high",
            message=_issue_text(message),
            line=0,
            column=1,
            file="analysis_input",
            rule="ast-parse-failed",
        )
        for message in (ast_result.syntax_errors or ["AST parse failed"])
    ]


async def ast_analysis(
    request: AnalysisRequest, analysis_id: str, start_time: float
) -> ApiAnalysisResult:
    """Perform AST analysis."""
    from services.analysis_status import AnalysisStatusService

    status_service = AnalysisStatusService()

    try:
        # Update status - initializing AST analysis
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.RUNNING,
            progress=25.0,
            phase=AnalysisPhase.INITIALIZING,
        )

        if not request.code:
            raise ValueError("Code content is required for AST analysis")

        # Initialize AST validation service
        ast_service = ASTValidationService()
        issues: list[Any] = []
        suggestions: list[Any] = []

        # Update status - parsing AST
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.RUNNING,
            progress=40.0,
            phase=AnalysisPhase.PARSING,
        )

        # Perform AST analysis
        ast_result = await ast_service.analyze_proposed_code_ast(
            request.code, "analysis_input", request.language or "python"
        )

        # Update status - extracting AST elements
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.RUNNING,
            progress=60.0,
            phase=AnalysisPhase.EXTRACTING,
        )

        # Process AST analysis results
        metrics = AnalysisMetrics(
            lines_of_code=ast_result.line_count,
            complexity=ast_result.complexity_score,
            maintainability=0.0,
            test_coverage=None,
            technical_debt=None,
        )

        # Calculate maintainability based on AST structure
        if ast_result.line_count > 0:
            # Basic maintainability based on complexity and function count
            function_ratio = len(ast_result.functions) / max(1, ast_result.line_count / 10)
            complexity_penalty = min(50, ast_result.complexity_score * 5)
            metrics.maintainability = max(0, 100 - complexity_penalty + function_ratio * 10)

        # Convert AST issues
        for ast_issue in ast_result.issues:
            issues.append(
                AnalysisIssue(
                    type="structure",
                    severity="medium",
                    message=ast_issue.description,
                    line=ast_issue.line_number,
                    column=1,
                    file="analysis_input",
                    rule="ast-validation",
                )
            )

        # A parse failure is a FINDING, not a pass. ``ASTValidationService``
        # catches the SyntaxError into ``syntax_errors`` / ``is_valid=False``;
        # this used to map only ``ast_result.issues`` (always empty on a failed
        # parse), so syntactically invalid code came back ``success: true``
        # with zero issues.
        issues.extend(_syntax_issues(ast_result))

        # Add AST-specific suggestions
        if ast_result.classes and not ast_result.functions:
            suggestions.append("Consider adding standalone functions for better modularity")
        if ast_result.complexity_score > 15:
            suggestions.append("High complexity detected - consider refactoring complex functions")

        return ApiAnalysisResult(
            analysis_type=request.analysis_type,
            success=ast_result.is_valid,
            issues=issues,
            metrics=metrics,
            suggestions=suggestions,
            execution_time=time.time() - start_time,
            metadata={
                "analysis_id": analysis_id,
                "ast_valid": ast_result.is_valid,
                "ast_function_count": len(ast_result.functions),
                "ast_class_count": len(ast_result.classes),
                "ast_import_count": len(ast_result.imports),
            },
        )

    except Exception as e:
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.FAILED,
            details={"error": str(e), "phase": "ast_analysis"},
        )
        # Fallback to basic analysis
        return await basic_analysis(request, analysis_id, start_time)


async def deep_analysis(
    request: AnalysisRequest, analysis_id: str, start_time: float
) -> ApiAnalysisResult:
    """Perform deep code analysis."""
    from services.analysis_status import AnalysisStatusService

    status_service = AnalysisStatusService()
    temp_dir = None  # Initialize temp_dir to None for proper cleanup

    try:
        # Update status - initializing deep analysis
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.RUNNING,
            progress=15.0,
            phase=AnalysisPhase.INITIALIZING,
        )

        if not request.code:
            raise ValueError("Code content is required for deep analysis")

        # Initialize deep analyzer with a temporary directory
        import shutil
        import tempfile

        temp_dir = tempfile.mkdtemp(prefix="deep_analysis_")
        deep_analyzer = DeepCodeAnalyzer(temp_dir, language=request.language or "python")

        # Update status - performing deep analysis
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.RUNNING,
            progress=40.0,
            phase=AnalysisPhase.ANALYZING,
        )

        # Perform deep analysis
        deep_result = await deep_analyzer.analyze_code_snippet(
            request.code, request.language or "python"
        )

        # Update status - processing results
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.RUNNING,
            progress=80.0,
            phase=AnalysisPhase.GENERATING_REPORT,
        )

        # Process deep analysis results
        issues = []
        suggestions = []

        # Convert deep analysis findings to issues
        if hasattr(deep_result, "quality_issues"):
            for quality_issue in deep_result.quality_issues:  # type: ignore[not-iterable]  # ty: ignore[not-iterable]
                issues.append(
                    AnalysisIssue(
                        type="quality",
                        severity=quality_issue.severity,
                        message=quality_issue.description,
                        line=quality_issue.line_number,
                        column=1,
                        file="analysis_input",
                        rule="deep-quality-check",
                    )
                )

        if hasattr(deep_result, "suggestions"):
            suggestions.extend(deep_result.suggestions)  # type: ignore[invalid-argument-type]  # ty: ignore[invalid-argument-type]

        # Create metrics from deep analysis
        metrics = AnalysisMetrics(
            lines_of_code=getattr(deep_result, "lines_of_code", 0),
            complexity=getattr(deep_result, "complexity_score", 0.0),
            maintainability=getattr(deep_result, "maintainability_score", 0.0),
            test_coverage=getattr(deep_result, "test_coverage", None),
            technical_debt=getattr(deep_result, "technical_debt_score", None),
        )

        # Clean up temporary directory
        if temp_dir is not None:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass  # Ignore cleanup errors

        return ApiAnalysisResult(
            analysis_type=request.analysis_type,
            success=True,
            issues=issues,
            metrics=metrics,
            suggestions=suggestions,
            execution_time=time.time() - start_time,
            metadata={
                "analysis_id": analysis_id,
                "deep_analysis_version": "v2",
                "analysis_depth": "comprehensive",
            },
        )

    except Exception as e:
        # Clean up temporary directory on error
        if temp_dir is not None:
            try:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass  # Ignore cleanup errors

        await status_service.update_status(
            analysis_id,
            AnalysisStatus.FAILED,
            details={"error": str(e), "phase": "deep_analysis"},
        )
        # Fallback to enhanced analysis
        return await enhanced_analysis(request, analysis_id, start_time)


async def security_analysis(
    request: AnalysisRequest, analysis_id: str, start_time: float
) -> ApiAnalysisResult:
    """Perform security vulnerability analysis."""
    from services.analysis_status import AnalysisStatusService

    status_service = AnalysisStatusService()

    try:
        # Update status - initializing security analysis
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.RUNNING,
            progress=20.0,
            phase=AnalysisPhase.INITIALIZING,
        )

        if not request.code:
            raise ValueError("Code content is required for security analysis")

        # Update status - scanning for vulnerabilities
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.RUNNING,
            progress=50.0,
            phase=AnalysisPhase.ANALYZING,
        )

        issues = []
        suggestions = []

        # Basic security pattern detection
        lines = request.code.split("\n")
        for i, line in enumerate(lines, 1):
            line_lower = line.lower()

            # SQL injection patterns
            if any(
                pattern in line_lower for pattern in ["select * from", "drop table", "delete from"]
            ):
                if "format(" in line_lower or "%s" in line or "+" in line:
                    issues.append(
                        AnalysisIssue(
                            type="security",
                            severity="high",
                            message="Potential SQL injection vulnerability detected",
                            line=i,
                            column=1,
                            file="input",
                            rule="sql-injection",
                        )
                    )
                    suggestions.append("Use parameterized queries to prevent SQL injection")

            # XSS patterns
            if "eval(" in line_lower or "exec(" in line_lower:
                issues.append(
                    AnalysisIssue(
                        type="security",
                        severity="high",
                        message="Use of eval/exec can lead to code injection",
                        line=i,
                        column=1,
                        file="input",
                        rule="code-injection",
                    )
                )
                suggestions.append("Avoid using eval() or exec() with user input")

            # Hardcoded secrets
            if any(pattern in line_lower for pattern in ["password=", "api_key=", "secret="]):
                if '"' in line or "'" in line:
                    issues.append(
                        AnalysisIssue(
                            type="security",
                            severity="medium",
                            message="Possible hardcoded credential detected",
                            line=i,
                            column=1,
                            file="input",
                            rule="hardcoded-secret",
                        )
                    )
                    suggestions.append("Use environment variables or secure credential storage")

        # Calculate basic security metrics
        security_score = max(0, 100 - len([i for i in issues if i.type == "security"]) * 10)

        metrics = AnalysisMetrics(
            lines_of_code=len([line for line in lines if line.strip()]),
            complexity=0.0,
            maintainability=0.0,
            test_coverage=None,
            technical_debt=None,
        )

        return ApiAnalysisResult(
            analysis_type=request.analysis_type,
            success=True,
            issues=issues,
            metrics=metrics,
            suggestions=suggestions,
            execution_time=time.time() - start_time,
            metadata={
                "analysis_id": analysis_id,
                "security_score": security_score,
                "vulnerabilities_found": len([i for i in issues if i.type == "security"]),
            },
        )

    except Exception as e:
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.FAILED,
            details={"error": str(e), "phase": "security_analysis"},
        )
        # Fallback to basic analysis
        return await basic_analysis(request, analysis_id, start_time)


async def performance_analysis(
    request: AnalysisRequest, analysis_id: str, start_time: float
) -> ApiAnalysisResult:
    """Perform performance bottleneck analysis."""
    from services.analysis_status import AnalysisStatusService

    status_service = AnalysisStatusService()

    try:
        # Update status - initializing performance analysis
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.RUNNING,
            progress=25.0,
            phase=AnalysisPhase.INITIALIZING,
        )

        if not request.code:
            raise ValueError("Code content is required for performance analysis")

        # Update status - analyzing performance patterns
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.RUNNING,
            progress=60.0,
            phase=AnalysisPhase.ANALYZING,
        )

        issues = []
        suggestions = []

        # Basic performance pattern detection
        lines = request.code.split("\n")
        for i, line in enumerate(lines, 1):
            line_lower = line.lower().strip()

            # Nested loops
            if "for" in line_lower and any("for" in ln.lower() for ln in lines[i : i + 5]):
                issues.append(
                    AnalysisIssue(
                        type="performance",
                        severity="medium",
                        message="Nested loops detected - potential O(n²) complexity",
                        line=i,
                        column=1,
                        file="input",
                        rule="nested-loops",
                    )
                )
                suggestions.append(
                    "Consider optimizing nested loops or using more efficient algorithms"
                )

            # Inefficient string concatenation
            if "+=" in line and any(str_type in line for str_type in ["str", '"', "'"]):
                issues.append(
                    AnalysisIssue(
                        type="performance",
                        severity="low",
                        message="String concatenation in loop may be inefficient",
                        line=i,
                        column=1,
                        file="input",
                        rule="string-concat",
                    )
                )
                suggestions.append("Use list.join() or f-strings for efficient string building")

            # Synchronous I/O in async context
            if "async def" in request.code and any(
                pattern in line_lower for pattern in ["open(", "requests.get", "time.sleep"]
            ):
                issues.append(
                    AnalysisIssue(
                        type="performance",
                        severity="medium",
                        message="Synchronous operation in async function",
                        line=i,
                        column=1,
                        file="input",
                        rule="sync-in-async",
                    )
                )
                suggestions.append("Use async/await versions of I/O operations")

        # Performance metrics
        loop_count = sum(1 for line in lines if "for" in line.lower() or "while" in line.lower())
        performance_score = max(0, 100 - len(issues) * 15)

        metrics = AnalysisMetrics(
            lines_of_code=len([line for line in lines if line.strip()]),
            complexity=0.0,
            maintainability=0.0,
            test_coverage=None,
            technical_debt=None,
        )

        return ApiAnalysisResult(
            analysis_type=request.analysis_type,
            success=True,
            issues=issues,
            metrics=metrics,
            suggestions=suggestions,
            execution_time=time.time() - start_time,
            metadata={
                "analysis_id": analysis_id,
                "performance_score": performance_score,
                "loop_count": loop_count,
                "bottlenecks_found": len(issues),
            },
        )

    except Exception as e:
        await status_service.update_status(
            analysis_id,
            AnalysisStatus.FAILED,
            details={"error": str(e), "phase": "performance_analysis"},
        )
        # Fallback to basic analysis
        return await basic_analysis(request, analysis_id, start_time)


#: THE ``POST /api/analysis/analyze`` (inline code) dispatch table.
#: ``analysis_endpoints.analyze_standalone_code`` routes through it and
#: ``/api/analysis/capabilities`` is computed from it.
ANALYSIS_RUNNERS: dict[AnalysisType, Any] = {
    AnalysisType.BASIC: basic_analysis,
    AnalysisType.AUGMENTED: enhanced_analysis,
    AnalysisType.AST: ast_analysis,
    AnalysisType.DEEP: deep_analysis,
    AnalysisType.SECURITY: security_analysis,
    AnalysisType.PERFORMANCE: performance_analysis,
}

#: Per analysis type, which optional result parts its runner fills in
#: (``complexity_metrics``: ``metrics.complexity`` > 0; ``suggestions``: a
#: non-empty ``suggestions`` list) for an ordinary Python snippet. The
#: capabilities ``features`` are computed from this table, and a unit test runs
#: every runner to keep it true.
ANALYSIS_OUTPUTS: dict[AnalysisType, frozenset[str]] = {
    AnalysisType.BASIC: frozenset({"complexity_metrics", "suggestions"}),
    AnalysisType.AUGMENTED: frozenset({"complexity_metrics", "suggestions"}),
    AnalysisType.AST: frozenset({"complexity_metrics"}),
    AnalysisType.DEEP: frozenset(),
    AnalysisType.SECURITY: frozenset(),
    AnalysisType.PERFORMANCE: frozenset(),
}

#: Per analysis type, the languages it has language-SPECIFIC logic for, each
#: taken from the constant that implementation itself checks. ``()`` = the type
#: is language-agnostic text heuristics only (it accepts any language and
#: understands none).
ANALYSIS_LANGUAGES: dict[AnalysisType, tuple[str, ...]] = {
    AnalysisType.BASIC: BASIC_RULE_LANGUAGES,
    AnalysisType.AUGMENTED: AUGMENTED_LANGUAGES,
    AnalysisType.AST: ast_validation_service.AST_LANGUAGES,
    AnalysisType.DEEP: deep_analyzer_module.SNIPPET_LANGUAGES,
    AnalysisType.SECURITY: (),
    AnalysisType.PERFORMANCE: (),
}
