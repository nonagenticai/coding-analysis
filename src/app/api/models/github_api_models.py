"""GitHub API models stub.

TODO: Implement GitHub-related API models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ForkRequest(BaseModel):
    """Fork request."""

    repo_url: str
    organization: str | None = None


class ForkResponse(BaseModel):
    """Fork response."""

    success: bool = False
    fork_url: str | None = None


class CreatePRRequest(BaseModel):
    """Create PR request."""

    repo_owner: str
    repo_name: str
    title: str
    body: str = ""
    head_branch: str = "main"
    base_branch: str = "main"


class CreatePRResponse(BaseModel):
    """Create PR response."""

    success: bool = False
    pr_url: str | None = None
    pr_number: int | None = None


class RepositoryInfo(BaseModel):
    """Repository info."""

    owner: str = ""
    name: str = ""
    full_name: str = ""
    default_branch: str = "main"
    metadata: dict[str, Any] = {}
