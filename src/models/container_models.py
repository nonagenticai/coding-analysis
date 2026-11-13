"""Module for defining container models used in the application."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContainerStatus(StrEnum):
    """Container status enumeration"""

    CREATING = "creating"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    REMOVING = "removing"


class AnalysisConfig(BaseModel):
    """Configuration for code analysis during provisioning."""

    include_tests: bool | None = Field(
        None,
        description="Whether to include test files in code graph analysis",
    )
    language_hint: str | None = Field(
        None,
        description="Primary programming language hint for better analysis",
    )
    exclude_patterns: list[str] | None = Field(
        None, description="Additional patterns to exclude from analysis"
    )
    include_patterns: list[str] | None = Field(
        None, description="Patterns to include despite other exclusions"
    )


class GitHubRepoRequest(BaseModel):
    """Request model for GitHub repository operations."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    repo_url: str = Field(
        ...,
        min_length=1,
        description="GitHub repository URL",
        pattern=r"^https://github\.com/[A-Za-z0-9\-_.]+/[A-Za-z0-9\-_.]+(?:/pull/\d+)?(?:\.git)?/?$",
        json_schema_extra={"example": "https://github.com/georgiedekker/enterprise_mcp_server"},
    )
    branch: str = Field(
        "main",
        min_length=1,
        description="Git branch to clone",
        json_schema_extra={"example": "main"},
    )
    container_name: str | None = Field(
        None,
        description="Optional container name",
        pattern=r"^[a-zA-Z][a-zA-Z0-9\-_]{0,62}$",
        json_schema_extra={"example": "enterprise-mcp-server"},
    )
    description: str | None = Field(None, description="Optional description", max_length=500)
    access_token: str | None = Field(
        None,
        description="GitHub Personal Access Token for private repositories",
        json_schema_extra={"example": "<github-personal-access-token>"},
    )
    user_id: str | None = Field(
        None,
        description="User ID for multi-account workflow (defaults to anonymous if not provided)",
        pattern=r"^[a-zA-Z0-9\-_\.@]+$",
        max_length=128,
        json_schema_extra={"example": "user-123"},
    )
    force_rebuild: bool | None = Field(
        False, description="Force recreation if container already exists"
    )
    analysis_config: AnalysisConfig | None = Field(
        None, description="Configuration for code analysis during provisioning"
    )
    pr_number: int | None = Field(
        None, description="Pull request number to provision", ge=1, le=999999
    )
    provision_type: Literal["branch", "pull_request"] = Field(
        "branch", description="Type of provisioning"
    )
    secret_config_id: str | None = Field(
        None,
        description="Client ID for retrieving environment variables from secrets store",
        pattern=r"^[a-zA-Z0-9\-_\.]+$",
        max_length=128,
        json_schema_extra={"example": "repo-myorg-myproject"},
    )

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, v):
        """Validate repo URL is not empty and follows GitHub URL format."""
        if not v or not v.strip():
            raise ValueError("Repository URL cannot be empty")

        v = v.strip()

        # Basic URL format validation for GitHub URLs

        # Check if it's a PR URL
        pr_url_pattern = re.compile(
            r"^https://github\.com/[A-Za-z0-9\-_.]+/[A-Za-z0-9\-_.]+/pull/\d+(?:/.*)?$"
        )

        if pr_url_pattern.match(v):
            # This is a PR URL - we'll handle parsing in the model validator
            return v

        # Standard repo URL pattern
        github_url_pattern = re.compile(
            r"^https://github\.com/[A-Za-z0-9\-_.]+/[A-Za-z0-9\-_.]+(?:\.git)?/?$"
        )

        # Additional check for obviously invalid URLs like single characters
        if len(v) < 10 or not v.startswith("https://"):
            raise ValueError(
                "Repository URL must be a valid GitHub URL (e.g., https://github.com/owner/repo)"
            )

        if not github_url_pattern.match(v):
            raise ValueError(
                "Repository URL must be a valid GitHub URL format (e.g., https://github.com/owner/repo)"
            )

        return v

    @field_validator("branch")
    @classmethod
    def validate_branch(cls, v):
        """Validate branch is not empty."""
        if not v or not v.strip():
            raise ValueError("Branch name cannot be empty")
        return v.strip()

    @field_validator("pr_number")
    @classmethod
    def validate_pr_number(cls, v):
        """Validate PR number if provided."""
        if v is not None and v <= 0:
            raise ValueError("PR number must be positive if provided")
        return v

    @model_validator(mode="after")
    def parse_pr_url(self):
        """Parse PR URL and extract repository URL and PR number if applicable."""

        # Check if repo_url is a PR URL
        pr_url_pattern = re.compile(
            r"^https://github\.com/([A-Za-z0-9\-_.]+)/([A-Za-z0-9\-_.]+)/pull/(\d+)(?:/.*)?$"
        )

        match = pr_url_pattern.match(self.repo_url)
        if match:
            # Extract components from PR URL
            owner = match.group(1)
            repo = match.group(2)
            pr_number = int(match.group(3))

            # Update fields
            self.repo_url = f"https://github.com/{owner}/{repo}"
            self.pr_number = pr_number
            self.provision_type = "pull_request"

        return self


class ContainerInfo(BaseModel):
    """Information model for container instances."""

    container_name: (
        str  # Primary identifier - same as Docker container name, used to access code graphs
    )
    api_endpoint: str = Field(
        ...,
        description="Deprecated alias for the public/gateway-facing endpoint",
    )
    internal_endpoint: str = Field(
        default="",
        description="Direct runtime-reachable endpoint used for internal probes and exec routing",
    )
    public_endpoint: str = Field(
        default="",
        description="Client-facing or gateway-facing endpoint for API responses",
    )
    code_graph_id: str = ""  # Code graph identifier for this container
    codebase_path: str
    repo_url: str
    branch: str
    status: ContainerStatus
    description: str | None = None
    pr_number: int | None = None
    provision_type: str = "branch"
    github_access_token: str | None = Field(
        None,
        description="GitHub access token for this container's repository operations (deprecated - use user_token_id)",
    )
    user_token_id: str | None = Field(
        None,
        description="Reference to encrypted user GitHub token in SecureTokenManager",
    )
    user_id: str | None = Field(
        None,
        description="User ID who provisioned this container",
    )
    original_repo_owner: str | None = Field(
        None,
        description="Original repository owner (for creating upstream PRs)",
    )
    original_repo_name: str | None = Field(
        None,
        description="Original repository name (for creating upstream PRs)",
    )
    system_fork_owner: str | None = Field(
        None,
        description="System account owner where fork resides (e.g., 'nonagenticai')",
    )
    graph_initialized: bool = Field(
        False,
        description="Whether the code graph has been initialized for this container",
    )
    tenant_id: str | None = Field(
        None,
        description="Tenant identifier for multi-tenant data isolation",
    )

    @model_validator(mode="after")
    def sync_endpoint_fields(self):
        """Keep legacy api_endpoint aligned with explicit internal/public endpoints."""
        if not self.public_endpoint:
            self.public_endpoint = self.api_endpoint
        if not self.api_endpoint:
            self.api_endpoint = self.public_endpoint or self.internal_endpoint
        return self


class ContainerSpec(BaseModel):
    """Specification model for container setup."""

    name: str = Field(..., description="Container name")
    image: str = Field(..., description="Docker image to use")
    port: int = Field(..., description="Host port to bind to")
    command: str | None = Field(None, description="Command to run in container")
    working_dir: str | None = Field(None, description="Working directory")
    environment: dict[str, Any] = Field(default_factory=dict, description="Environment variables")
    volumes: dict[str, Any] = Field(default_factory=dict, description="Volume mounts")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """Validate container name is not empty."""
        if not v or not v.strip():
            raise ValueError("Container name cannot be empty")
        return v.strip()
