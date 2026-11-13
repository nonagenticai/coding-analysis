"""Performance issue analysis for the deep code analyzer."""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING

from .base import (
    CodeIssue,
    IssueType,
    Severity,
    get_line,
    safe_parse_ast,
)

if TYPE_CHECKING:
    from .analyzer import DeepCodeAnalyzer

logger = logging.getLogger(__name__)


def analyze(analyzer: DeepCodeAnalyzer) -> None:
    """Analyze for performance problems."""
    logger.debug("Analyzing performance issues")
    for file_path in analyzer.python_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = safe_parse_ast(content, str(file_path))
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    for child in ast.walk(node):
                        if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                            if isinstance(child.target, ast.Name):
                                analyzer.issues.append(
                                    CodeIssue(
                                        type=IssueType.PERFORMANCE,
                                        severity=Severity.MEDIUM,
                                        title="String concatenation in loop",
                                        description="String concatenation in loops can be inefficient",
                                        file_path=str(file_path),
                                        line_number=child.lineno,
                                        code_snippet=get_line(content, child.lineno),
                                        suggestion="Use ''.join() or list comprehension instead",
                                        confidence=0.8,
                                    )
                                )
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "*":
                        analyzer.issues.append(
                            CodeIssue(
                                type=IssueType.PERFORMANCE,
                                severity=Severity.LOW,
                                title="Wildcard import",
                                description="Wildcard imports can slow startup and pollute namespace",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                code_snippet=get_line(content, node.lineno),
                                suggestion="Import specific names instead of using wildcard",
                                confidence=0.6,
                            )
                        )
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Error analyzing performance in {file_path}: {e}")
