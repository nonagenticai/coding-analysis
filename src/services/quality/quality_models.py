#!/usr/bin/env python3
"""Quality models module."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QualityServiceRequest(BaseModel):
    """Request model for quality analysis."""

    code_path: str | None = None
    code_content: str | None = None
    language: str | None = None
    tools: list[str] = Field(default_factory=list, description="Quality tools to run")
    fix_issues: bool = Field(default=False, description="Attempt to resolve issues automatically")
    custom_rules: dict[str, Any] | None = Field(default=None, description="Custom rules")


class QualityServiceIssue(BaseModel):
    """Model for individual quality issues found by tools."""

    tool: str
    severity: str
    message: str
    line: int | None = None
    column: int | None = None
    rule_id: str | None = None
    fixable: bool = False


class AppToolResult(BaseModel):
    """Model for individual tool execution results."""

    tool: str
    success: bool
    issues: list[QualityServiceIssue] = Field(default_factory=list)
    execution_time: float
    output: str = ""
    error: str = ""


class QualityResult(BaseModel):
    """Model for complete quality analysis results."""

    analysis_id: str
    success: bool
    language: str
    tools_results: dict[str, AppToolResult] = Field(default_factory=dict)
    overall_score: float
    total_issues: int
    execution_time: float
    created_at: str
    error: str | None = None  # Add error field for failed analysis
