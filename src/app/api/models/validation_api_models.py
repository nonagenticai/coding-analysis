"""Validation API models stub.

TODO: Implement validation API models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ValidateFileReferenceRequest(BaseModel):
    """Validate file reference request."""

    file_path: str
    container_id: str | None = None


class ValidateFileReferenceResponse(BaseModel):
    """Validate file reference response."""

    valid: bool = False
    issues: list[dict[str, Any]] = []


class ValidateFunctionReferenceRequest(BaseModel):
    """Validate function reference request."""

    function_name: str
    file_path: str | None = None


class ValidateFunctionReferenceResponse(BaseModel):
    """Validate function reference response."""

    valid: bool = False
    issues: list[dict[str, Any]] = []


class ValidateClassReferenceRequest(BaseModel):
    """Validate class reference request."""

    class_name: str
    file_path: str | None = None


class ValidateClassReferenceResponse(BaseModel):
    """Validate class reference response."""

    valid: bool = False
    issues: list[dict[str, Any]] = []


class ValidateCodeSnippetRequest(BaseModel):
    """Validate code snippet request."""

    code: str
    language: str = "python"


class ValidateCodeSnippetResponse(BaseModel):
    """Validate code snippet response."""

    valid: bool = False
    issues: list[dict[str, Any]] = []


class CheckConsistencyResponse(BaseModel):
    """Check consistency response."""

    consistent: bool = False
    issues: list[dict[str, Any]] = []


class CheckFunctionConsistencyResponse(BaseModel):
    """Check function consistency response."""

    consistent: bool = False
    issues: list[dict[str, Any]] = []


class CheckClassConsistencyResponse(BaseModel):
    """Check class consistency response."""

    consistent: bool = False
    issues: list[dict[str, Any]] = []
