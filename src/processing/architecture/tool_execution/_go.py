"""Go tool runners (golint, go vet, golangci-lint, gosec)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from models.architecture_models import ArchitectureQualityIssue

logger = logging.getLogger(__name__)


class GoToolsMixin:
    """Mixin providing Go static-analysis runners."""

    _execute_tool_in_container: Any  # provided by BaseToolExecution
    _parse_line_issues: Any  # provided by BaseToolExecution

    async def _run_golint(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run golint via container."""
        command = ["golint", "./..."]
        issues, result = await self._execute_tool_in_container(
            "golint", command, repo_path, timeout=60
        )

        if result.get("success") and result.get("stdout"):
            issues.extend(
                self._parse_line_issues(
                    result["stdout"], tool="golint", category="style", severity="medium"
                )
            )

        return issues, result

    async def _run_go_vet(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run go vet via container."""
        command = ["go", "vet", "./..."]
        issues, result = await self._execute_tool_in_container(
            "go-vet", command, repo_path, timeout=60
        )

        if result.get("success"):
            # go vet prints to stderr
            stderr = result.get("stderr", "")
            if stderr:
                issues.extend(
                    self._parse_line_issues(
                        stderr, tool="go-vet", category="bug-risk", severity="high"
                    )
                )

        return issues, result

    async def _run_golangci_lint(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run golangci-lint via container."""
        command = ["golangci-lint", "run", "--out-format=json", "./..."]
        issues, result = await self._execute_tool_in_container(
            "golangci-lint", command, repo_path, timeout=120
        )

        if result.get("success") and result.get("stdout"):
            try:
                data = json.loads(result["stdout"])
                for issue_data in data.get("Issues", []):
                    pos = issue_data.get("Pos", {})
                    severity_map = {
                        "error": "high",
                        "warning": "medium",
                        "info": "low",
                    }
                    severity = severity_map.get(issue_data.get("Severity", "warning"), "medium")

                    issues.append(
                        ArchitectureQualityIssue(
                            id=f"golangci-{issue_data.get('FromLinter', 'unknown')}-{len(issues)}",
                            severity=severity,
                            message=issue_data.get("Text", ""),
                            rule_id=issue_data.get("FromLinter", ""),
                            rule_name=issue_data.get("FromLinter", ""),
                            file_path=pos.get("Filename", ""),
                            line_number=pos.get("Line", 1),
                            column_number=pos.get("Column"),
                            tool="golangci-lint",
                            category="style",
                            description=issue_data.get("Text", ""),
                        )
                    )
            except json.JSONDecodeError:
                logger.warning("Failed to parse golangci-lint JSON output")

        return issues, result

    async def _run_gosec(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run gosec via container."""
        command = ["gosec", "-fmt=json", "./..."]
        issues, result = await self._execute_tool_in_container(
            "gosec", command, repo_path, timeout=120
        )

        if result.get("success") and result.get("stdout"):
            try:
                data = json.loads(result["stdout"])
                for issue_data in data.get("Issues", []):
                    severity_str = issue_data.get("severity", "MEDIUM").upper()
                    severity_map = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}
                    severity = severity_map.get(severity_str, "medium")

                    line_number = 1
                    line_str = issue_data.get("line", "1")
                    # gosec may report ranges like "10-15"
                    try:
                        line_number = int(line_str.split("-")[0])
                    except (ValueError, AttributeError):  # fmt: skip
                        line_number = 1

                    issues.append(
                        ArchitectureQualityIssue(
                            id=f"gosec-{issue_data.get('rule_id', 'unknown')}-{len(issues)}",
                            severity=severity,
                            message=issue_data.get("details", ""),
                            rule_id=issue_data.get("rule_id", ""),
                            rule_name=issue_data.get("rule_id", ""),
                            file_path=issue_data.get("file", ""),
                            line_number=line_number,
                            column_number=None,
                            tool="gosec",
                            category="security",
                            description=issue_data.get("details", ""),
                        )
                    )
            except json.JSONDecodeError:
                logger.warning("Failed to parse gosec JSON output")

        return issues, result
