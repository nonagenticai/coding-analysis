"""Code-smell analysis (parameter count + nesting depth) for the deep code analyzer."""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING

from .base import (
    CodeIssue,
    IssueType,
    Severity,
    calculate_nesting_depth,
    get_function_signature,
    get_line,
    safe_parse_ast,
)

if TYPE_CHECKING:
    from .analyzer import DeepCodeAnalyzer

logger = logging.getLogger(__name__)


def analyze(analyzer: DeepCodeAnalyzer) -> None:
    """Analyze for code smells."""
    logger.debug("Analyzing code smells")
    for file_path in analyzer.python_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = safe_parse_ast(content, str(file_path))
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    param_count = len(node.args.args)
                    if param_count > 7:
                        analyzer.issues.append(
                            CodeIssue(
                                type=IssueType.CODE_SMELL,
                                severity=Severity.MEDIUM,
                                title=f"Too many parameters: {param_count}",
                                description=f"Function '{node.name}' has too many parameters",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                code_snippet=get_function_signature(content, node),
                                suggestion="Consider using a configuration object or breaking the function",
                                confidence=0.8,
                            )
                        )
                elif isinstance(node, (ast.If, ast.For, ast.While)):
                    depth = calculate_nesting_depth(node)
                    if depth > 4:
                        analyzer.issues.append(
                            CodeIssue(
                                type=IssueType.CODE_SMELL,
                                severity=Severity.MEDIUM,
                                title=f"Deep nesting: {depth} levels",
                                description="Deeply nested code is hard to understand",
                                file_path=str(file_path),
                                line_number=node.lineno,
                                code_snippet=get_line(content, node.lineno),
                                suggestion="Consider extracting nested logic into functions",
                                confidence=0.7,
                            )
                        )
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Error analyzing code smells in {file_path}: {e}")
