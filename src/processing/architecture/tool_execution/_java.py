"""Java tool runners (checkstyle, pmd, spotbugs)."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from models.architecture_models import ArchitectureQualityIssue

logger = logging.getLogger(__name__)


class JavaToolsMixin:
    """Mixin providing Java static-analysis runners."""

    _execute_tool_in_container: Any  # provided by BaseToolExecution

    async def _run_checkstyle(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run Checkstyle via container."""
        command = ["checkstyle", "-c", "/google_checks.xml", str(repo_path)]
        issues, result = await self._execute_tool_in_container(
            "checkstyle", command, repo_path, timeout=120
        )

        if result.get("success") and result.get("stdout"):
            # Checkstyle output: [ERROR] file:line:col: message [RuleName]
            for line in result["stdout"].splitlines():
                line = line.strip()
                if not line:
                    continue
                # Strip severity prefix
                severity = "medium"
                if line.startswith("[ERROR]"):
                    severity = "high"
                    line = line[7:].strip()
                elif line.startswith("[WARN]"):
                    severity = "medium"
                    line = line[6:].strip()
                else:
                    continue

                parts = line.split(":", 3)
                if len(parts) >= 3:
                    file_path = parts[0].strip()
                    try:
                        line_number = int(parts[1].strip())
                    except ValueError:
                        line_number = 1
                    message = parts[-1].strip()

                    # Extract rule name from [RuleName] at end
                    rule_name = "checkstyle-issue"
                    if message.endswith("]"):
                        bracket_idx = message.rfind("[")
                        if bracket_idx > 0:
                            rule_name = message[bracket_idx + 1 : -1]
                            message = message[:bracket_idx].strip()

                    issues.append(
                        ArchitectureQualityIssue(
                            id=f"checkstyle-{rule_name}-{len(issues)}",
                            severity=severity,
                            message=message,
                            rule_id=rule_name,
                            rule_name=rule_name,
                            file_path=file_path,
                            line_number=line_number,
                            column_number=None,
                            tool="checkstyle",
                            category="style",
                            description=message,
                        )
                    )

        return issues, result

    async def _run_pmd(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run PMD via container."""
        command = [
            "pmd",
            "check",
            "-d",
            str(repo_path),
            "-R",
            "rulesets/java/quickstart.xml",
            "-f",
            "json",
        ]
        issues, result = await self._execute_tool_in_container(
            "pmd", command, repo_path, timeout=120
        )

        if result.get("success") and result.get("stdout"):
            data = self._safe_json(result["stdout"], dict)
            if data is not None:
                for violation in data.get("files", []):
                    if not isinstance(violation, dict):
                        continue
                    file_path = violation.get("filename", "")
                    for v in violation.get("violations", []):
                        if not isinstance(v, dict):
                            continue
                        priority = v.get("priority", 3)
                        if priority <= 1:
                            severity = "critical"
                        elif priority <= 2:
                            severity = "high"
                        elif priority <= 3:
                            severity = "medium"
                        else:
                            severity = "low"

                        issues.append(
                            ArchitectureQualityIssue(
                                id=f"pmd-{v.get('rule', 'unknown')}-{len(issues)}",
                                severity=severity,
                                message=v.get("description", ""),
                                rule_id=v.get("rule", ""),
                                rule_name=v.get("ruleset", ""),
                                file_path=file_path,
                                line_number=v.get("beginline", 1),
                                column_number=v.get("begincolumn"),
                                tool="pmd",
                                category="style",
                                description=v.get("description", ""),
                                external_url=v.get("externalInfoUrl"),
                            )
                        )

        return issues, result

    async def _run_spotbugs(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run SpotBugs via container."""
        command = ["spotbugs", "-textui", "-xml", str(repo_path)]
        issues, result = await self._execute_tool_in_container(
            "spotbugs", command, repo_path, timeout=180
        )

        if result.get("success") and result.get("stdout"):
            # SpotBugs outputs XML; parse line-based summary from stderr as fallback
            stdout = result["stdout"]
            try:
                root = ET.fromstring(stdout)  # nosec B314
                for bug in root.iter("BugInstance"):
                    bug_type = bug.get("type", "unknown")
                    priority = int(bug.get("priority", "3"))
                    if priority <= 1:
                        severity = "high"
                    elif priority <= 2:
                        severity = "medium"
                    else:
                        severity = "low"

                    message = ""
                    msg_elem = bug.find("ShortMessage")
                    if msg_elem is not None and msg_elem.text:
                        message = msg_elem.text

                    file_path = ""
                    line_number = 1
                    source = bug.find("SourceLine")
                    if source is not None:
                        file_path = source.get("sourcepath", "")
                        try:
                            line_number = int(source.get("start", "1"))
                        except ValueError:
                            line_number = 1

                    issues.append(
                        ArchitectureQualityIssue(
                            id=f"spotbugs-{bug_type}-{len(issues)}",
                            severity=severity,
                            message=message or bug_type,
                            rule_id=bug_type,
                            rule_name=bug.get("category", "bug"),
                            file_path=file_path,
                            line_number=line_number,
                            column_number=None,
                            tool="spotbugs",
                            category="bug-risk",
                            description=message or bug_type,
                        )
                    )
            except ET.ParseError:
                logger.warning("Failed to parse SpotBugs XML output")

        return issues, result
