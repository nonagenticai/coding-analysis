"""GitHub integration service for creating pull requests via the GitHub API.

Uses httpx to interact with the GitHub REST API for PR creation.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"


class GitHubIntegration:
    """Integrates with GitHub for PR creation and code management."""

    def __init__(
        self,
        github_token: str | None = None,
        codebase_path: str | None = None,
    ) -> None:
        self.github_token = github_token
        self.codebase_path = codebase_path

    def create_pull_request(
        self,
        repo_owner: str,
        repo_name: str,
        title: str,
        message: str,
        head_branch: str,
        base_branch: str,
        resolves: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a pull request with bug fixes.

        Args:
            repo_owner: GitHub repository owner.
            repo_name: GitHub repository name.
            title: PR title.
            message: PR description body text.
            head_branch: Source branch name.
            base_branch: Target branch name.
            resolves: Optional list of fix dicts to include in the PR body.

        Returns:
            Dict with 'success', 'pr_url', 'pr_number' on success,
            or 'success' False with 'error' on failure.
        """
        if not self.github_token:
            return {"success": False, "error": "No GitHub token configured"}

        body_text = self._build_body(message, resolves)

        url = f"{GITHUB_API_BASE}/repos/{repo_owner}/{repo_name}/pulls"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        payload = {
            "title": title,
            "body": body_text,
            "head": head_branch,
            "base": base_branch,
        }

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            return {
                "success": True,
                "pr_url": data.get("html_url", ""),
                "pr_number": data.get("number"),
            }
        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text
            logger.error(
                "GitHub API error %s: %s",
                exc.response.status_code,
                error_body,
            )
            return {
                "success": False,
                "error": f"GitHub API error {exc.response.status_code}: {error_body}",
                "status_code": exc.response.status_code,
            }
        except httpx.RequestError as exc:
            logger.error("Request to GitHub failed: %s", exc)
            return {
                "success": False,
                "error": f"Request failed: {exc}",
            }

    @staticmethod
    def _build_body(
        message: str,
        resolves: list[dict[str, Any]] | None,
    ) -> str:
        """Build the PR body text from the message and optional fixes list."""
        parts = [message]

        if resolves:
            parts.append("\n\n## Fixes\n")
            for i, fix in enumerate(resolves, 1):
                file_path = fix.get("file_path", "unknown")
                explanation = fix.get("explanation", "No description")
                line = fix.get("line_number")
                loc = f" (line {line})" if line else ""
                parts.append(f"**{i}.** `{file_path}`{loc} - {explanation}")

        return "\n".join(parts)
