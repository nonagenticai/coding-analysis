"""Documentation analysis (missing docstrings) for the deep code analyzer."""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING

from .base import (
    CodeIssue,
    IssueType,
    Severity,
    get_function_signature,
    safe_parse_ast,
)

if TYPE_CHECKING:
    from .analyzer import DeepCodeAnalyzer

logger = logging.getLogger(__name__)


def analyze(analyzer: DeepCodeAnalyzer) -> None:
    """Analyze documentation quality."""
    logger.debug("Analyzing documentation")
    for file_path in analyzer.python_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = safe_parse_ast(content, str(file_path))
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        severity = (
                            Severity.MEDIUM if isinstance(node, ast.ClassDef) else Severity.LOW
                        )
                        analyzer.issues.append(
                            CodeIssue(
                                type=IssueType.DOCUMENTATION,
                                severity=severity,
                                title=f"Missing docstring for {type(node).__name__.lower()}",
                                description=f"{type(node).__name__} '{node.name}' has no docstring",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                code_snippet=(
                                    get_function_signature(content, node)
                                    if isinstance(node, ast.FunctionDef)
                                    else ""
                                ),
                                suggestion="Add a descriptive docstring",
                                confidence=1.0,
                            )
                        )
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Error analyzing documentation in {file_path}: {e}")
