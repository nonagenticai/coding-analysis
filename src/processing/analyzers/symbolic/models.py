"""Data models for symbolic analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SecurityIssue:
    """Security issue found during symbolic/taint analysis."""

    type: str
    severity: str
    message: str
    file_path: str
    line_number: int = 0
    column_number: int = 0
    code_snippet: str = ""
    recommendation: str = ""
    cwe_id: str | None = None


@dataclass
class SymbolicAnalysisResult:
    """Complete result of symbolic analysis (e.g. Pysa)."""

    file_path: str
    security_issues: list[SecurityIssue] = field(default_factory=list)
    taint_flows: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
