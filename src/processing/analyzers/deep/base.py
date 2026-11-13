"""Shared primitives for the deep code analyzer package.

Contains issue enums, dataclasses, and AST helper functions used by every
sub-analyzer in this package.
"""

from __future__ import annotations

import ast
import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic.dataclasses import dataclass

from models.core_models import CoreCodeEntity, EntityRelation

logger = logging.getLogger(__name__)


class IssueCollector(Protocol):
    """The surface the per-concern sub-analyzers in this package actually need.

    ``security.analyze`` and its siblings were annotated as taking a whole
    ``DeepCodeAnalyzer``, but they only read ``python_files`` and append to
    ``issues``. Naming that surface lets the rules be reused over a single file
    — as the advanced-validation endpoint does — without constructing a full
    codebase analyzer, and without a cast that would silence a real mismatch.

    ``DeepCodeAnalyzer`` satisfies this structurally; nothing about it changes.
    """

    python_files: list[Path]
    issues: list[CodeIssue]


class IssueType(Enum):
    """Enumeration of different types of code issues."""

    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    COMPLEXITY = "complexity"
    CODE_SMELL = "code_smell"
    BUG_RISK = "bug_risk"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"


class Severity(Enum):
    """Enumeration of severity levels for code issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class CodeIssue:
    """Represents a code issue detected during analysis."""

    type: IssueType
    severity: Severity
    title: str
    description: str
    file_path: str
    line_number: int | None
    code_snippet: str | None
    suggestion: str
    confidence: float


@dataclass
class DeepAnalysisServiceCodeEntity:
    """Represents a code entity like function, class, etc."""

    id: str
    type: str
    name: str
    line_number: int
    file_path: str
    docstring: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class CodeRelationship:
    """Represents a relationship between code entities."""

    source_id: str
    target_id: str
    type: str
    metadata: dict[str, Any] | None = None


@dataclass
class DeepAnalysisResult:
    """Stores results of a deep code analysis, including issues and metrics.

    ``relationships`` accepts both the Python-path ``CodeRelationship`` dataclass
    and the multi-language ``EntityRelation`` model — the graph writer reads the
    common ``source_id``/``target_id``/``type`` attributes from either.
    """

    overall_score: float
    issues: list[CodeIssue]
    metrics: dict[str, Any]
    recommendations: list[str]
    entities: list[CoreCodeEntity]
    relationships: list[CodeRelationship | EntityRelation]


def safe_parse_ast(content: str, filename: str = "<unknown>") -> ast.AST | None:
    """Parse Python code into AST while suppressing SyntaxWarnings about escape sequences."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            return ast.parse(content, filename=filename)
    except SyntaxError as e:
        logger.debug(f"Syntax error parsing {filename}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Error parsing {filename}: {e}")
        return None


def extract_docstring(node: ast.AST) -> str | None:
    """Extract docstring from AST node."""
    body = getattr(node, "body", None)
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        # ast.Constant is used in Python 3.8+ for string literals
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def extract_decorator_name(decorator: ast.AST) -> str:
    """Extract decorator name from AST node."""
    if isinstance(decorator, ast.Name):
        return decorator.id
    elif isinstance(decorator, ast.Attribute):
        return f"{decorator.attr}"
    else:
        return str(decorator)


def extract_base_name(base: ast.AST) -> str:
    """Extract base class name from AST node."""
    if isinstance(base, ast.Name):
        return base.id
    elif isinstance(base, ast.Attribute):
        return f"{base.attr}"
    else:
        return str(base)


def get_line(content: str, line_no: int) -> str:
    """Get specific line from content."""
    lines = content.splitlines()
    if 0 <= line_no - 1 < len(lines):
        return lines[line_no - 1].strip()
    return ""


def get_function_signature(content: str, node: ast.FunctionDef) -> str:
    """Get function signature."""
    lines = content.splitlines()
    if 0 <= node.lineno - 1 < len(lines):
        return lines[node.lineno - 1].strip()
    return f"def {node.name}(...):"


def calculate_cyclomatic_complexity(node: ast.FunctionDef) -> int:
    """Calculate cyclomatic complexity of a function."""
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
    return complexity


def calculate_nesting_depth(node: ast.AST) -> int:
    """Calculate maximum nesting depth."""
    max_depth = 0

    def _get_depth(n: ast.AST, current_depth: int = 0) -> None:
        nonlocal max_depth
        max_depth = max(max_depth, current_depth)
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                _get_depth(child, current_depth + 1)
            else:
                _get_depth(child, current_depth)

    _get_depth(node)
    return max_depth


__all__ = [
    "CodeIssue",
    "CodeRelationship",
    "DeepAnalysisResult",
    "DeepAnalysisServiceCodeEntity",
    "IssueCollector",
    "IssueType",
    "Severity",
    "calculate_cyclomatic_complexity",
    "calculate_nesting_depth",
    "extract_base_name",
    "extract_decorator_name",
    "extract_docstring",
    "get_function_signature",
    "get_line",
    "safe_parse_ast",
]
