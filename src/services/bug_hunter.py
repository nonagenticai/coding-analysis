"""
Bug Hunter Service

Clean orchestrator that coordinates bug analysis and resolving using focused services.
Eliminates the previous 1,550+ line bloated implementation.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from models.bug_models import BugAnalysisResult, CodeFix
from services.bug_analysis import BugAnalyzer
from services.codebase_scanner import CodebaseScanner
from services.error_parser import ErrorParser
from services.fix_generator import FixGenerator
from services.github_integration_service import GitHubIntegration

logger = logging.getLogger(__name__)


class BugHunter:
    """Clean orchestrator for bug analysis and resolving."""

    def __init__(self, codebase_path: str, github_token: str | None = None):
        """Initialize the BugHunter service.

        Args:
            codebase_path (str): Codebase path
            github_token (Optional[str]): Github token
        """
        self.codebase_path = Path(codebase_path)
        self.github_token = github_token

        # Initialize focused services
        self.error_parser = ErrorParser(str(codebase_path))
        self.fix_generator = FixGenerator(str(codebase_path))
        self.bug_analyzer = BugAnalyzer()
        self.codebase_scanner = CodebaseScanner(str(codebase_path))
        self.github_integration = GitHubIntegration(
            github_token=github_token,
            codebase_path=str(codebase_path),
        )

    def analyze_error_logs(self, error_logs: str, context: dict | None = None) -> BugAnalysisResult:
        """Analyze error logs and generate comprehensive bug analysis.

        Raises:
            ValueError: No error logs were given (a codebase scan is not done
                here), or error-log / deep-analysis parsing failed.
            OSError: The codebase could not be read.

        ⛔ Do NOT re-introduce a blanket ``except`` that returns an empty
        ``BugAnalysisResult``. This method used to catch
        ``(ValueError, FileNotFoundError, OSError)`` and RETURN
        ``BugAnalysisResult(errors=[], ..., root_cause_analysis="Analysis
        failed: …")``. Because it returned rather than raised, the caller chain
        ``_run_claimed_hunt -> hunt_bugs -> analyze_error_logs`` took its
        SUCCESS path and persisted the job via ``store.complete_hunt``, so a
        crashed hunt read ``status=completed, errors_found=0,
        error_message=NULL`` — byte-identical to a hunt over a clean
        repository. No consumer polling ``GET …/hunts/{id}/status`` could tell
        the two apart. Propagating the failure is what reaches the existing
        ``except Exception -> store.fail_hunt(...)`` handler in
        ``_run_claimed_hunt``, the only code that writes ``status=failed`` with
        a non-null ``error_message``.
        """
        logger.info("Starting bug analysis from error logs")

        if not error_logs or not error_logs.strip():
            # A codebase scan never runs in this pod: it has no checkout of the
            # tenant's repository. It runs inside the workspace via the hunt
            # agent (services.workspace_hunt), and hunt_bugs routes it there.
            raise ValueError(
                "No error logs to analyse; a codebase scan runs in the workspace "
                "(services.workspace_hunt), not here"
            )
        if self.codebase_scanner.is_deep_analysis_result(error_logs):
            logger.info("Processing deep analysis results as error input")
            errors = self.codebase_scanner.parse_deep_analysis_results(error_logs)
        else:
            # Parse traditional error logs
            errors = self.error_parser.parse_error_logs(error_logs)

        # Generate resolves for each error
        resolves: list[Any] = []
        for error in errors:
            error_fixes = self.fix_generator.generate_fixes_for_error(error)
            resolves.extend(error_fixes)

        # Perform comprehensive analysis
        analysis_result = self.bug_analyzer.analyze_bugs_and_fixes(errors, resolves)

        logger.info(f"Analysis complete: {len(errors)} errors, {len(resolves)} resolves")
        return analysis_result

    async def create_pull_request(
        self,
        repo_owner: str,
        repo_name: str,
        resolves: list[CodeFix],
        branch_name: str | None = None,
    ) -> dict[str, Any]:
        """Create a pull request with bug resolutiones."""
        logger.info(f"Creating PR for {len(resolves)} resolves")

        try:
            # Convert CodeFix objects to dictionaries
            fixes_dict_list = [asdict(resolve) for resolve in resolves] if resolves else None

            result = self.github_integration.create_pull_request(
                repo_owner=repo_owner,
                repo_name=repo_name,
                title=f"Resolve {len(resolves)} bugs",
                message=f"Automatically generated PR with {len(resolves)} bug resolutiones",
                head_branch=branch_name or "bug-resolves",
                base_branch="main",
                resolves=fixes_dict_list,
            )
            return result if isinstance(result, dict) else {"success": True, "data": result}
        except (ValueError, FileNotFoundError, OSError) as e:
            logger.error(f"Failed to create PR: {e}")
            return {"success": False, "error": str(e)}

    def get_analysis_summary(self, result: BugAnalysisResult) -> dict[str, Any]:
        """Get a summary of the analysis results."""
        return {
            "total_errors": len(result.errors),
            "total_fixes": len(result.resolves),
            "error_types": list(set(error.error_type.value for error in result.errors)),
            "average_fix_confidence": (
                sum(resolve.confidence for resolve in result.resolves) / len(result.resolves)
                if result.resolves
                else 0.0
            ),
            "files_affected": list(
                set(resolve.file_path for resolve in result.resolves if resolve.file_path)
            ),
            "root_cause_summary": (
                result.root_cause_analysis[:200] + "..."
                if len(result.root_cause_analysis) > 200
                else result.root_cause_analysis
            ),
            "estimated_effort": result.estimated_effort,
            "risk_level": result.risk_assessment,
        }
