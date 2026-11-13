"""JavaScript / TypeScript tool runners (eslint, tsc, prettier, jshint)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from models.architecture_models import ArchitectureQualityIssue

logger = logging.getLogger(__name__)


class JavaScriptToolsMixin:
    """Mixin providing JavaScript/TypeScript static-analysis runners."""

    _execute_tool_in_container: Any  # provided by BaseToolExecution

    async def _run_eslint(
        self, repo_path: Path, language: str
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run ESLint via container."""
        command = ["eslint", str(repo_path), "-f", "json"]
        issues, result = await self._execute_tool_in_container(
            "eslint", command, repo_path, timeout=120
        )

        if result.get("success") and result.get("stdout"):
            file_results = self._safe_json(result["stdout"], list)
            if file_results is not None:
                for file_result in file_results:
                    if not isinstance(file_result, dict):
                        continue
                    file_path = file_result.get("filePath", "")
                    for msg in file_result.get("messages", []):
                        if not isinstance(msg, dict):
                            continue
                        severity = "high" if msg.get("severity", 1) == 2 else "medium"
                        issues.append(
                            ArchitectureQualityIssue(
                                id=f"eslint-{msg.get('ruleId', 'unknown')}-{len(issues)}",
                                severity=severity,
                                message=msg.get("message", ""),
                                rule_id=msg.get("ruleId", "") or "eslint-error",
                                rule_name=msg.get("ruleId", "") or "ESLint error",
                                file_path=file_path,
                                line_number=msg.get("line", 1),
                                column_number=msg.get("column"),
                                tool="eslint",
                                category="style",
                                description=msg.get("message", ""),
                            )
                        )

        return issues, result

    async def _run_typescript_compiler(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run TypeScript compiler via container."""
        command = ["tsc", "--noEmit", "--pretty", "false"]
        issues, result = await self._execute_tool_in_container(
            "tsc", command, repo_path, timeout=120
        )

        if result.get("success") and result.get("stdout"):
            # tsc output: file(line,col): error TSxxxx: message
            for line in result["stdout"].splitlines():
                line = line.strip()
                if not line:
                    continue
                # Parse: src/foo.ts(10,5): error TS2322: Type ... is not assignable
                paren_idx = line.find("(")
                colon_idx = line.find("): ")
                if paren_idx > 0 and colon_idx > paren_idx:
                    file_path = line[:paren_idx]
                    loc = line[paren_idx + 1 : colon_idx]
                    rest = line[colon_idx + 3 :]
                    parts = loc.split(",")
                    try:
                        line_number = int(parts[0])
                    except (ValueError, IndexError):  # fmt: skip
                        line_number = 1
                    try:
                        col_number = int(parts[1]) if len(parts) > 1 else None
                    except ValueError:
                        col_number = None

                    # Extract TS error code
                    rule_id = "tsc-error"
                    if rest.startswith("error "):
                        ts_code_end = rest.find(":")
                        if ts_code_end > 0:
                            rule_id = rest[6:ts_code_end].strip()
                            rest = rest[ts_code_end + 1 :].strip()

                    issues.append(
                        ArchitectureQualityIssue(
                            id=f"tsc-{rule_id}-{len(issues)}",
                            severity="high",
                            message=rest,
                            rule_id=rule_id,
                            rule_name="TypeScript Error",
                            file_path=file_path,
                            line_number=line_number,
                            column_number=col_number,
                            tool="tsc",
                            category="type-safety",
                            description=rest,
                        )
                    )

        return issues, result

    async def _run_prettier(
        self, repo_path: Path, language: str
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run Prettier check via container."""
        command = ["prettier", "--check", str(repo_path)]
        issues, result = await self._execute_tool_in_container(
            "prettier", command, repo_path, timeout=60
        )

        if result.get("success") and result.get("stdout"):
            for line in result["stdout"].splitlines():
                line = line.strip()
                # prettier --check prints file paths that need formatting
                if line and not line.startswith("Checking") and not line.startswith("All"):
                    issues.append(
                        ArchitectureQualityIssue(
                            id=f"prettier-format-{len(issues)}",
                            severity="low",
                            message=f"File needs formatting: {line}",
                            rule_id="prettier-format",
                            rule_name="Prettier formatting",
                            file_path=line,
                            line_number=1,
                            column_number=None,
                            tool="prettier",
                            category="formatting",
                            description=f"File would be reformatted by prettier: {line}",
                        )
                    )

        return issues, result

    async def _run_jshint(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run JSHint via container."""
        command = ["jshint", str(repo_path), "--reporter=json"]
        issues, result = await self._execute_tool_in_container(
            "jshint", command, repo_path, timeout=60
        )

        if result.get("success") and result.get("stdout"):
            data = self._safe_json(result["stdout"], (dict, list))
            if data is not None:
                # jshint JSON reporter returns {"result": [...]} or a list
                results = data if isinstance(data, list) else data.get("result", [])
                for entry in results:
                    if not isinstance(entry, dict):
                        continue
                    err = entry.get("error", entry)
                    if not isinstance(err, dict):
                        continue
                    issues.append(
                        ArchitectureQualityIssue(
                            id=f"jshint-{err.get('code', 'unknown')}-{len(issues)}",
                            severity="medium" if err.get("code", "").startswith("W") else "high",
                            message=err.get("reason", ""),
                            rule_id=err.get("code", "jshint-error"),
                            rule_name="JSHint issue",
                            file_path=entry.get("file", ""),
                            line_number=err.get("line", 1),
                            column_number=err.get("character"),
                            tool="jshint",
                            category="style",
                            description=err.get("reason", ""),
                        )
                    )

        return issues, result
