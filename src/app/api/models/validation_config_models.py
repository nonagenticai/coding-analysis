"""Validation config models stub.

TODO: Implement validation configuration models.
"""

from __future__ import annotations

from pydantic import BaseModel


class ValidationConfig(BaseModel):
    """Validation configuration."""

    strict_mode: bool = False
    max_issues: int = 100
    include_warnings: bool = True
    include_suggestions: bool = False
