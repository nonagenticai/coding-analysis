"""C# tool runners (dotnet-format, roslynator)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from models.architecture_models import ArchitectureQualityIssue

logger = logging.getLogger(__name__)


class CSharpToolsMixin:
    """Mixin providing C# static-analysis runners."""

    _execute_tool_in_container: Any  # provided by BaseToolExecution

    async def _run_dotnet_format(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run dotnet format check via container."""
        command = ["dotnet", "format", "--verify-no-changes", str(repo_path)]
        issues, result = await self._execute_tool_in_container(
            "dotnet-format", command, repo_path, timeout=120
        )

        if result.get("success") and result.get("exit_code", 0) != 0:
            # dotnet format returns non-zero if files would change
            output = result.get("stdout", "") + "\n" + result.get("stderr", "")
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Look for file paths in output
                if line.endswith(".cs") or line.endswith(".csproj") or ".cs(" in line:
                    file_path = line.split("(")[0].strip() if "(" in line else line
                    issues.append(
                        ArchitectureQualityIssue(
                            id=f"dotnet-format-{len(issues)}",
                            severity="low",
                            message=f"File needs formatting: {file_path}",
                            rule_id="dotnet-format",
                            rule_name="dotnet format",
                            file_path=file_path,
                            line_number=1,
                            column_number=None,
                            tool="dotnet-format",
                            category="formatting",
                            description=f"File would be reformatted by dotnet format: {file_path}",
                        )
                    )

        return issues, result

    async def _run_roslynator(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run Roslynator via container."""
        command = ["roslynator", "analyze", str(repo_path)]
        issues, result = await self._execute_tool_in_container(
            "roslynator", command, repo_path, timeout=180
        )

        if result.get("success") and result.get("stdout"):
            # Roslynator output: file(line,col): severity ID: message
            for line in result["stdout"].splitlines():
                line = line.strip()
                if not line:
                    continue
                paren_idx = line.find("(")
                colon_after_paren = line.find("): ")
                if paren_idx > 0 and colon_after_paren > paren_idx:
                    file_path = line[:paren_idx]
                    loc = line[paren_idx + 1 : colon_after_paren]
                    rest = line[colon_after_paren + 3 :]

                    loc_parts = loc.split(",")
                    try:
                        line_number = int(loc_parts[0])
                    except (ValueError, IndexError):  # fmt: skip
                        line_number = 1
                    try:
                        col_number = int(loc_parts[1]) if len(loc_parts) > 1 else None
                    except ValueError:
                        col_number = None

                    # Parse "severity RCSxxxx: message"
                    severity = "medium"
                    rule_id = "roslynator-issue"
                    message = rest
                    if (
                        rest.startswith("error")
                        or rest.startswith("warning")
                        or rest.startswith("info")
                    ):
                        sev_end = rest.find(" ")
                        sev_str = rest[:sev_end]
                        rest = rest[sev_end + 1 :]
                        severity_map = {"error": "high", "warning": "medium", "info": "low"}
                        severity = severity_map.get(sev_str, "medium")

                        id_end = rest.find(":")
                        if id_end > 0:
                            rule_id = rest[:id_end].strip()
                            message = rest[id_end + 1 :].strip()

                    issues.append(
                        ArchitectureQualityIssue(
                            id=f"roslynator-{rule_id}-{len(issues)}",
                            severity=severity,
                            message=message,
                            rule_id=rule_id,
                            rule_name="Roslynator diagnostic",
                            file_path=file_path,
                            line_number=line_number,
                            column_number=col_number,
                            tool="roslynator",
                            category="style",
                            description=message,
                        )
                    )

        return issues, result
