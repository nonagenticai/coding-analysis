"""Validation base models stub.

TODO: Implement validation base models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ValidationIssue(BaseModel):
    """Validation issue."""

    type: str = ""
    message: str = ""
    severity: str = "warning"
    location: dict[str, Any] = {}


class ValidationMetrics(BaseModel):
    """Validation metrics."""

    total_checks: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0


class BaseValidationRequest(BaseModel):
    """Base validation request."""

    container_id: str | None = None


class BaseValidationResponse(BaseModel):
    """Base validation response."""

    valid: bool = False
    issues: list[ValidationIssue] = []
    metrics: ValidationMetrics = ValidationMetrics()
