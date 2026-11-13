"""C/C++ tool runners (cppcheck, clang-tidy, cpplint)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from models.architecture_models import ArchitectureQualityIssue

logger = logging.getLogger(__name__)


class CppToolsMixin:
    """Mixin providing C/C++ static-analysis runners."""

    _execute_tool_in_container: Any  # provided by BaseToolExecution

    async def _run_cppcheck(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run cppcheck via container."""
        command = [
            "cppcheck",
            "--enable=all",
            "--template='{file}:{line}:{severity}:{message}'",
            str(repo_path),
        ]
        issues, result = await self._execute_tool_in_container(
            "cppcheck", command, repo_path, timeout=120
        )

        if result.get("success"):
            # cppcheck writes to stderr by default
            output = result.get("stderr", "") or result.get("stdout", "")
            for line in output.splitlines():
                line = line.strip().strip("'")
                if not line:
                    continue
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    file_path = parts[0].strip()
                    try:
                        line_number = int(parts[1].strip())
                    except ValueError:
                        line_number = 1
                    cppcheck_severity = parts[2].strip()
                    message = parts[3].strip()

                    severity_map = {
                        "error": "high",
                        "warning": "medium",
                        "style": "low",
                        "performance": "medium",
                        "portability": "low",
                        "information": "low",
                    }
                    severity = severity_map.get(cppcheck_severity, "medium")

                    issues.append(
                        ArchitectureQualityIssue(
                            id=f"cppcheck-{cppcheck_severity}-{len(issues)}",
                            severity=severity,
                            message=message,
                            rule_id=f"cppcheck-{cppcheck_severity}",
                            rule_name=f"cppcheck {cppcheck_severity}",
                            file_path=file_path,
                            line_number=line_number,
                            column_number=None,
                            tool="cppcheck",
                            category="bug-risk" if cppcheck_severity == "error" else "style",
                            description=message,
                        )
                    )

        return issues, result

    async def _run_clang_tidy(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run clang-tidy via container."""
        command = ["clang-tidy", str(repo_path)]
        issues, result = await self._execute_tool_in_container(
            "clang-tidy", command, repo_path, timeout=120
        )

        if result.get("success") and result.get("stdout"):
            # clang-tidy output: file:line:col: severity: message [check-name]
            for line in result["stdout"].splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(":", 4)
                if len(parts) >= 5:
                    file_path = parts[0].strip()
                    try:
                        line_number = int(parts[1].strip())
                    except ValueError:
                        continue
                    try:
                        col_number = int(parts[2].strip())
                    except ValueError:
                        col_number = None

                    rest = parts[3].strip() + ":" + parts[4] if len(parts) > 4 else parts[3].strip()
                    severity = "medium"
                    if rest.startswith("error"):
                        severity = "high"
                    elif rest.startswith("warning"):
                        severity = "medium"
                    elif rest.startswith("note"):
                        severity = "low"

                    message = rest.split(":", 1)[-1].strip() if ":" in rest else rest

                    # Extract check name from [check-name]
                    rule_id = "clang-tidy-issue"
                    if message.endswith("]"):
                        bracket_idx = message.rfind("[")
                        if bracket_idx > 0:
                            rule_id = message[bracket_idx + 1 : -1]
                            message = message[:bracket_idx].strip()

                    issues.append(
                        ArchitectureQualityIssue(
                            id=f"clang-tidy-{rule_id}-{len(issues)}",
                            severity=severity,
                            message=message,
                            rule_id=rule_id,
                            rule_name="clang-tidy check",
                            file_path=file_path,
                            line_number=line_number,
                            column_number=col_number,
                            tool="clang-tidy",
                            category="bug-risk",
                            description=message,
                        )
                    )

        return issues, result

    async def _run_cpplint(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run cpplint via container."""
        command = ["cpplint", "--recursive", str(repo_path)]
        issues, result = await self._execute_tool_in_container(
            "cpplint", command, repo_path, timeout=60
        )

        if result.get("success"):
            # cpplint writes to stderr
            output = result.get("stderr", "") or result.get("stdout", "")
            for line in output.splitlines():
                line = line.strip()
                if (
                    not line
                    or line.startswith("Done processing")
                    or line.startswith("Total errors")
                ):
                    continue
                # Format: file:line: message [category/subcategory] [confidence]
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    file_path = parts[0].strip()
                    try:
                        line_number = int(parts[1].strip())
                    except ValueError:
                        continue
                    message = parts[2].strip()

                    # Extract category from [category] and confidence from [N]
                    rule_id = "cpplint-issue"
                    bracket_parts = message.split("[")
                    if len(bracket_parts) >= 2:
                        rule_id = bracket_parts[1].strip().rstrip("]").strip()
                        message = bracket_parts[0].strip()

                    issues.append(
                        ArchitectureQualityIssue(
                            id=f"cpplint-{rule_id}-{len(issues)}",
                            severity="medium",
                            message=message,
                            rule_id=rule_id,
                            rule_name="cpplint check",
                            file_path=file_path,
                            line_number=line_number,
                            column_number=None,
                            tool="cpplint",
                            category="style",
                            description=message,
                        )
                    )

        return issues, result
