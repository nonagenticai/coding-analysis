"""Architecture issue analysis (circular import detection) for the deep code analyzer."""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING, Any

from .base import (
    CodeIssue,
    IssueType,
    Severity,
    safe_parse_ast,
)

if TYPE_CHECKING:
    from .analyzer import DeepCodeAnalyzer

logger = logging.getLogger(__name__)


def analyze(analyzer: DeepCodeAnalyzer) -> None:
    """Analyze architectural issues."""
    logger.debug("Analyzing architecture")
    import_graph: dict[str, Any] = {}
    for file_path in analyzer.python_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = safe_parse_ast(content, str(file_path))
            if tree is None:
                continue
            imports: list[Any] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif node.module:
                        imports.append(node.module)
            import_graph[str(file_path)] = imports
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Error analyzing imports in {file_path}: {e}")
    for file_path_str, imports in import_graph.items():
        for imp in imports:
            if imp in import_graph and file_path_str in import_graph[imp]:
                analyzer.issues.append(
                    CodeIssue(
                        type=IssueType.ARCHITECTURE,
                        severity=Severity.HIGH,
                        title="Potential circular import",
                        description=f"Circular import detected between {file_path_str} and {imp}",
                        file_path=file_path_str,
                        line_number=None,
                        code_snippet=None,
                        suggestion="Restructure imports to avoid circular dependencies",
                        confidence=0.6,
                    )
                )
