"""Unit tests for GitHubIntegration service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from services.github_integration_service import GitHubIntegration


@pytest.fixture
def integration() -> GitHubIntegration:
    return GitHubIntegration(github_token="ghp_test123", codebase_path="/tmp/test")


@pytest.fixture
def no_token_integration() -> GitHubIntegration:
    return GitHubIntegration(github_token=None, codebase_path="/tmp/test")


# ---------------------------------------------------------------------------
# No token
# ---------------------------------------------------------------------------


class TestNoToken:
    def test_returns_failure_without_token(self, no_token_integration: GitHubIntegration):
        result = no_token_integration.create_pull_request(
            repo_owner="owner",
            repo_name="repo",
            title="Fix bugs",
            message="Some fixes",
            head_branch="fix-branch",
            base_branch="main",
        )
        assert result["success"] is False
        assert "token" in result["error"].lower()

    def test_empty_string_token(self):
        integration = GitHubIntegration(github_token="", codebase_path="/tmp")
        result = integration.create_pull_request(
            repo_owner="owner",
            repo_name="repo",
            title="Fix",
            message="Fix",
            head_branch="fix",
            base_branch="main",
        )
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Successful PR creation
# ---------------------------------------------------------------------------


class TestSuccessfulPR:
    @patch("services.github_integration_service.httpx.post")
    def test_creates_pr_successfully(self, mock_post: MagicMock, integration: GitHubIntegration):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "html_url": "https://github.com/owner/repo/pull/42",
            "number": 42,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = integration.create_pull_request(
            repo_owner="owner",
            repo_name="repo",
            title="Fix 3 bugs",
            message="Automated PR",
            head_branch="bug-fixes",
            base_branch="main",
        )

        assert result["success"] is True
        assert result["pr_url"] == "https://github.com/owner/repo/pull/42"
        assert result["pr_number"] == 42

        # Verify the API was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "repos/owner/repo/pulls" in call_args[0][0]
        assert call_args[1]["headers"]["Authorization"] == "token ghp_test123"
        assert call_args[1]["json"]["title"] == "Fix 3 bugs"
        assert call_args[1]["json"]["head"] == "bug-fixes"
        assert call_args[1]["json"]["base"] == "main"

    @patch("services.github_integration_service.httpx.post")
    def test_includes_resolves_in_body(self, mock_post: MagicMock, integration: GitHubIntegration):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "html_url": "https://github.com/owner/repo/pull/1",
            "number": 1,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        resolves = [
            {
                "file_path": "src/app.py",
                "line_number": 10,
                "explanation": "Fix import error",
            },
            {
                "file_path": "src/util.py",
                "line_number": None,
                "explanation": "Fix key error",
            },
        ]

        integration.create_pull_request(
            repo_owner="owner",
            repo_name="repo",
            title="Fixes",
            message="Bug fixes",
            head_branch="fix",
            base_branch="main",
            resolves=resolves,
        )

        body = mock_post.call_args[1]["json"]["body"]
        assert "src/app.py" in body
        assert "line 10" in body
        assert "Fix import error" in body
        assert "src/util.py" in body
        assert "Fix key error" in body

    @patch("services.github_integration_service.httpx.post")
    def test_no_resolves(self, mock_post: MagicMock, integration: GitHubIntegration):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "html_url": "https://github.com/owner/repo/pull/5",
            "number": 5,
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = integration.create_pull_request(
            repo_owner="owner",
            repo_name="repo",
            title="Fixes",
            message="Just a message",
            head_branch="fix",
            base_branch="main",
            resolves=None,
        )

        assert result["success"] is True
        body = mock_post.call_args[1]["json"]["body"]
        assert "Fixes" not in body or "Just a message" in body


# ---------------------------------------------------------------------------
# API errors
# ---------------------------------------------------------------------------


class TestAPIErrors:
    @patch("services.github_integration_service.httpx.post")
    def test_handles_422_validation_error(
        self, mock_post: MagicMock, integration: GitHubIntegration
    ):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 422
        mock_response.text = '{"message": "Validation Failed"}'
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "422 Client Error",
            request=MagicMock(),
            response=mock_response,
        )
        mock_post.return_value = mock_response

        result = integration.create_pull_request(
            repo_owner="owner",
            repo_name="repo",
            title="Fix",
            message="Fix",
            head_branch="fix",
            base_branch="main",
        )

        assert result["success"] is False
        assert "422" in result["error"]
        assert result.get("status_code") == 422

    @patch("services.github_integration_service.httpx.post")
    def test_handles_401_unauthorized(self, mock_post: MagicMock, integration: GitHubIntegration):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.text = '{"message": "Bad credentials"}'
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401 Client Error",
            request=MagicMock(),
            response=mock_response,
        )
        mock_post.return_value = mock_response

        result = integration.create_pull_request(
            repo_owner="owner",
            repo_name="repo",
            title="Fix",
            message="Fix",
            head_branch="fix",
            base_branch="main",
        )

        assert result["success"] is False
        assert "401" in result["error"]

    @patch("services.github_integration_service.httpx.post")
    def test_handles_network_error(self, mock_post: MagicMock, integration: GitHubIntegration):
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        result = integration.create_pull_request(
            repo_owner="owner",
            repo_name="repo",
            title="Fix",
            message="Fix",
            head_branch="fix",
            base_branch="main",
        )

        assert result["success"] is False
        assert "Connection refused" in result["error"] or "Request failed" in result["error"]


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


class TestBuildBody:
    def test_message_only(self):
        body = GitHubIntegration._build_body("Hello world", None)
        assert body == "Hello world"

    def test_with_resolves(self):
        resolves = [
            {"file_path": "a.py", "line_number": 5, "explanation": "Fix A"},
            {"file_path": "b.py", "explanation": "Fix B"},
        ]
        body = GitHubIntegration._build_body("PR body", resolves)
        assert "PR body" in body
        assert "## Fixes" in body
        assert "`a.py`" in body
        assert "(line 5)" in body
        assert "Fix A" in body
        assert "`b.py`" in body
        assert "Fix B" in body

    def test_empty_resolves_list(self):
        body = GitHubIntegration._build_body("Message", [])
        # Empty list is falsy, should not add fixes section
        assert "## Fixes" not in body


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_stores_token_and_path(self):
        gi = GitHubIntegration(github_token="tok", codebase_path="/code")
        assert gi.github_token == "tok"
        assert gi.codebase_path == "/code"

    def test_defaults_to_none(self):
        gi = GitHubIntegration()
        assert gi.github_token is None
        assert gi.codebase_path is None
