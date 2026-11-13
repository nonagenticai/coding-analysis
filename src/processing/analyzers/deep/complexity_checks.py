"""Cyclomatic complexity + long-function checks for the deep code analyzer."""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING

from .base import (
    CodeIssue,
    IssueType,
    Severity,
    calculate_cyclomatic_complexity,
    get_function_signature,
    safe_parse_ast,
)

if TYPE_CHECKING:
    from .analyzer import DeepCodeAnalyzer

logger = logging.getLogger(__name__)


def analyze(analyzer: DeepCodeAnalyzer) -> None:
    """Analyze code complexity."""
    logger.debug("Analyzing code complexity")
    for file_path in analyzer.python_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = safe_parse_ast(content, str(file_path))
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = calculate_cyclomatic_complexity(node)
                    if complexity > 10:
                        severity = Severity.HIGH if complexity > 20 else Severity.MEDIUM
                        analyzer.issues.append(
                            CodeIssue(
                                type=IssueType.COMPLEXITY,
                                severity=severity,
                                title=f"High cyclomatic complexity: {complexity}",
                                description=f"Function '{node.name}' has high complexity ({complexity})",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                code_snippet=get_function_signature(content, node),
                                suggestion="Consider breaking this function into smaller functions",
                                confidence=0.9,
                            )
                        )
                    func_lines = (
                        (node.end_lineno or node.lineno) - node.lineno
                        if hasattr(node, "end_lineno")
                        else 0
                    )
                    if func_lines > 50:
                        analyzer.issues.append(
                            CodeIssue(
                                type=IssueType.MAINTAINABILITY,
                                severity=Severity.MEDIUM,
                                title=f"Long function: {func_lines} lines",
                                description=f"Function '{node.name}' is very long",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                code_snippet=get_function_signature(content, node),
                                suggestion="Consider breaking this function into smaller functions",
                                confidence=0.8,
                            )
                        )
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Error analyzing complexity in {file_path}: {e}")
