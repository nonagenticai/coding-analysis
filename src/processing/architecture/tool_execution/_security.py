"""Multi-language security tool runners (semgrep)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from models.architecture_models import ArchitectureQualityIssue

logger = logging.getLogger(__name__)


class SecurityToolsMixin:
    """Mixin providing multi-language security runners."""

    _execute_tool_in_container: Any  # provided by BaseToolExecution

    async def _run_semgrep(
        self, repo_path: Path, language: str
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run Semgrep via container."""
        command = ["semgrep", "--config=auto", "--json", str(repo_path)]
        issues, result = await self._execute_tool_in_container(
            "semgrep", command, repo_path, timeout=180
        )

        if result.get("success") and result.get("stdout"):
            data = self._safe_json(result["stdout"], dict)
            if data is not None:
                for finding in data.get("results", []):
                    if not isinstance(finding, dict):
                        continue
                    severity_map = {
                        "ERROR": "high",
                        "WARNING": "medium",
                        "INFO": "low",
                    }
                    severity = severity_map.get(
                        finding.get("extra", {}).get("severity", "WARNING"),
                        "medium",
                    )

                    issues.append(
                        ArchitectureQualityIssue(
                            id=f"semgrep-{finding.get('check_id', 'unknown')}-{len(issues)}",
                            severity=severity,
                            message=finding.get("extra", {}).get("message", ""),
                            rule_id=finding.get("check_id", ""),
                            rule_name=finding.get("check_id", ""),
                            file_path=finding.get("path", ""),
                            line_number=finding.get("start", {}).get("line", 1),
                            column_number=finding.get("start", {}).get("col"),
                            tool="semgrep",
                            category="security",
                            description=finding.get("extra", {}).get("message", ""),
                            external_url=finding.get("extra", {}).get("metadata", {}).get("source"),
                        )
                    )

        return issues, result
