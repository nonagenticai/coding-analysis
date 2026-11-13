"""Package manager models for dependency analysis."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExternalPackage(BaseModel):
    """Represents an external package dependency."""

    name: str = Field(..., description="Package name")
    version: str = Field(default="*", description="Version constraint or installed version")
    license: str = Field(default="Unknown", description="Package license")
    ecosystem: str = Field(
        default="unknown", description="Package ecosystem (pypi, npm, cargo, etc.)"
    )
    description: str | None = Field(default=None, description="Package description")
    current_version: str | None = Field(default=None, description="Currently installed version")
    latest_version: str | None = Field(default=None, description="Latest available version")
    severity: str | None = Field(default=None, description="Vulnerability severity if applicable")
