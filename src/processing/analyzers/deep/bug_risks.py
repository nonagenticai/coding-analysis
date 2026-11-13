"""Bug-risk analysis (bare except + mutable default args) for the deep code analyzer."""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING

from .base import (
    CodeIssue,
    IssueType,
    Severity,
    get_function_signature,
    get_line,
    safe_parse_ast,
)

if TYPE_CHECKING:
    from .analyzer import DeepCodeAnalyzer

logger = logging.getLogger(__name__)


def analyze(analyzer: DeepCodeAnalyzer) -> None:
    """Analyze for potential bug risks."""
    logger.debug("Analyzing bug risks")
    for file_path in analyzer.python_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = safe_parse_ast(content, str(file_path))
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler):
                    if node.type is None:
                        analyzer.issues.append(
                            CodeIssue(
                                type=IssueType.BUG_RISK,
                                severity=Severity.MEDIUM,
                                title="Bare except clause",
                                description="Bare except clauses can hide bugs",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                code_snippet=get_line(content, node.lineno),
                                suggestion="Specify exception types or use 'except Exception:'",
                                confidence=0.9,
                            )
                        )
                elif isinstance(node, ast.FunctionDef):
                    for default in node.args.defaults:
                        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                            analyzer.issues.append(
                                CodeIssue(
                                    type=IssueType.BUG_RISK,
                                    severity=Severity.HIGH,
                                    title="Mutable default argument",
                                    description="Mutable default arguments can cause unexpected behavior",
                                    file_path=str(file_path),
                                    line_number=node.lineno,
                                    code_snippet=get_function_signature(content, node),
                                    suggestion="Use None as default and create the mutable object inside the function",
                                    confidence=0.95,
                                )
                            )
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Error analyzing bug risks in {file_path}: {e}")
