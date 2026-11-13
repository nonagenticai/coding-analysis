"""
Quality tool orchestration for container-based code analysis.
Coordinates execution of quality tools like linters, formatters, and analyzers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid7

from models.testing_models import (
    QualityIssue,
    QualityRequest,
    QualityResult,
)
from processing.utils.ruff_output import parse_ruff_json, ruff_failure
from processing.utils.ty_output import parse_ty_gitlab, ty_failure
from services.docker_service import DockerService

from .tool_registry import QualityTool, get_tool_registry

logger = logging.getLogger(__name__)


class QualityToolRunner:
    """
    Orchestrates quality tool execution in containerized environments.
    Manages tool selection, execution, and result aggregation.
    """

    def __init__(self, docker_service: DockerService):
        self.docker_service = docker_service
        self.tool_registry = get_tool_registry()
        self.container_images = self._initialize_container_images()

    async def run_quality_analysis(self, request: QualityRequest) -> QualityResult:
        """Execute comprehensive quality analysis using available tools."""
        analysis_id = str(uuid7())
        start_time = datetime.now(UTC)

        try:
            logger.info("Starting quality analysis %s", analysis_id)

            # Validate and prepare request
            language = self._detect_language(request)
            tools_to_run = self._select_tools(language, request.tools)

            if not tools_to_run:
                raise ValueError(f"No quality tools available for language: {language}")

            # Create container for analysis
            docker_id = await self._create_quality_container(language, request)

            try:
                # Prepare code files in container
                await self._prepare_code_files(docker_id, request, language)

                # Install tools if needed
                await self._ensure_tools_installed(docker_id, language, tools_to_run)

                # Run tools in parallel
                tool_results = await self._execute_tools_parallel(
                    docker_id, tools_to_run, language, request
                )

                # Aggregate results
                all_issues = []
                for result in tool_results.values():
                    if result.get("success") and result.get("issues"):
                        all_issues.extend(result["issues"])

                # Calculate quality metrics
                quality_score = self._calculate_quality_score(all_issues, tool_results)

                execution_time = (datetime.now(UTC) - start_time).total_seconds()

                return QualityResult(
                    analysis_id=analysis_id,
                    overall_score=quality_score,
                    total_issues=len(all_issues),
                    issues=all_issues,
                    tools_results=tool_results,
                    execution_time=execution_time,
                    created_at=start_time,
                    language=language,
                    error=None,
                )

            finally:
                # Always cleanup container
                await self._cleanup_container(docker_id)

        except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Quality analysis %s failed: %s", analysis_id, e)
            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            return QualityResult(
                analysis_id=analysis_id,
                success=False,
                overall_score=0.0,
                total_issues=0,
                issues=[],
                tools_results={},
                execution_time=execution_time,
                created_at=start_time,
                language=request.language or "unknown",
                error=str(e),
            )

    async def run_single_tool(self, tool_name: str, language: str, code: str) -> dict[str, Any]:
        """Run a single quality tool and return results."""
        tool = self.tool_registry.get_tool(language, tool_name)

        if not tool:
            raise ValueError(f"Tool {tool_name} not available for {language}")

        docker_id = await self._create_quality_container(language)

        try:
            # Prepare code file
            filename = f"code{self._get_file_extension(language)}"
            success = await self.docker_service.create_file_in_container(
                docker_id, f"/app/quality/{filename}", code
            )

            if not success:
                raise RuntimeError("Failed to create code file in container")

            # Install tool
            await self._ensure_tools_installed(docker_id, language, [tool_name])

            # Execute tool
            result = await self._execute_single_tool(docker_id, tool, filename)

            return result

        finally:
            await self._cleanup_container(docker_id)

    def get_available_tools(self, language: str) -> list[str]:
        """Get list of available quality tools for a language."""
        return self.tool_registry.list_available_tools(language)

    def get_tool_info(self, tool_name: str, language: str) -> dict[str, Any] | None:
        """Get information about a specific tool."""
        return self.tool_registry.get_tool_info(language, tool_name)

    async def _create_quality_container(
        self, language: str, request: QualityRequest | None = None
    ) -> str:
        """Create container with quality tools for specified language."""
        image = self.container_images.get(language, "code_graph_helper-api:latest")

        # Use request for future container customization if needed
        _ = request  # Parameter reserved for future use
        container_config = {
            "image": image,
            "working_dir": "/app/quality",
            "mem_limit": "512m",
            "cpu_quota": 50000,  # 0.5 CPU
            "network_disabled": True,
            "read_only": False,  # Quality tools may need to write temp files
            "user": "testuser",
            "security_opt": ["no-new-privileges:true"],
            "tmpfs": {"/tmp": "noexec,nosuid,size=200m"},  # nosec B108
            "labels": {"service": "quality_analysis", "language": language},
            "language": language,
        }

        docker_id = await self.docker_service.create_test_container(container_config)
        await self.docker_service.start_container(docker_id)

        # Create quality directory
        await self.docker_service.exec_in_container(
            docker_id, ["mkdir", "-p", "/app/quality"], timeout=10
        )

        logger.debug("Created quality container %s for %s", docker_id, language)
        return docker_id

    async def _prepare_code_files(
        self, docker_id: str, request: QualityRequest, language: str
    ) -> None:
        """Prepare code files in container for analysis."""
        files_created = 0

        # Handle direct code content
        if request.code:
            extension = self._get_file_extension(language)
            success = await self.docker_service.create_file_in_container(
                docker_id, f"/app/quality/main{extension}", request.code
            )
            if success:
                files_created += 1

        # Handle file path
        if request.code_path:
            try:
                code_content = Path(request.code_path).read_text(encoding="utf-8")
                filename = Path(request.code_path).name
                success = await self.docker_service.create_file_in_container(
                    docker_id, f"/app/quality/{filename}", code_content
                )
                if success:
                    files_created += 1
            except (OSError, ValueError) as e:
                logger.warning("Failed to read code file %s: %s", request.code_path, e)

        if files_created == 0:
            raise ValueError("No code files prepared for analysis")

        logger.debug("Prepared %d code files in container %s", files_created, docker_id)

    async def _ensure_tools_installed(
        self, docker_id: str, language: str, tool_names: list[str]
    ) -> None:
        """Ensure required tools are installed in container."""
        try:
            # Get installation commands for the language
            _ = tool_names  # Parameter reserved for future tool-specific installation
            install_commands = self.tool_registry.get_installation_commands(language)

            for cmd in install_commands:
                # Split command string into list
                cmd_parts = cmd.split()
                result = await self.docker_service.exec_in_container(
                    docker_id, cmd_parts, timeout=120
                )

                if result["exit_code"] != 0:
                    logger.warning("Tool installation warning: %s", result["stderr"])
                else:
                    logger.debug("Installed tools for %s", language)
                    break  # Success, don't try other commands

        except (OSError, RuntimeError, TimeoutError) as e:
            logger.warning("Failed to install tools for %s: %s", language, e)
            # Continue anyway - tools might already be installed

    async def _execute_tools_parallel(
        self,
        docker_id: str,
        tool_names: list[str],
        language: str,
        request: QualityRequest,
    ) -> dict[str, dict[str, Any]]:
        """Execute multiple quality tools in parallel."""
        tasks = []

        for tool_name in tool_names:
            tool = self.tool_registry.get_tool(language, tool_name)
            if not tool:
                continue

            filename = self._get_target_filename(request, language)

            task = self._execute_single_tool(docker_id, tool, filename)
            tasks.append((tool_name, task))

        results = {}

        # Execute tools with timeout
        for tool_name, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=90)
                results[tool_name] = result
            except TimeoutError:
                logger.warning("Tool %s timed out", tool_name)
                results[tool_name] = {
                    "success": False,
                    "error": "Tool execution timed out",
                    "issues": [],
                    "execution_time": 90.0,
                }
            except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("Tool %s failed: %s", tool_name, e)
                results[tool_name] = {
                    "success": False,
                    "error": str(e),
                    "issues": [],
                    "execution_time": 0.0,
                }

        return results

    async def _execute_single_tool(
        self, docker_id: str, tool: QualityTool, filename: str
    ) -> dict[str, Any]:
        """Execute a single quality tool and parse results."""
        start_time = datetime.now(UTC)

        try:
            # Build command - add filename to the tool's base command
            command = tool.command + [filename]

            # Execute tool
            result = await self.docker_service.exec_in_container(
                docker_id, command, timeout=tool.timeout
            )

            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            # Parse output
            issues = await self._parse_tool_output(
                tool,
                result.get("stdout", ""),
                result.get("stderr", ""),
                filename,
            )

            # Many tools return non-zero exit codes when they find issues
            # Consider it successful if we got output to parse or exit code is 0
            success = (
                result.get("exit_code", 1) == 0 or len(issues) > 0 or bool(result.get("stdout"))
            )
            # The astral tools have an exact exit contract: anything but
            # "0 with a report" / "1 with parsed rows" did NOT measure the code
            # (exit 2 = bad config/argv/missing target, 101 = panic). That is a
            # tool FAILURE, never a clean run -- and its rows are not findings.
            failure = self._astral_failure(tool, result)
            if result.get("truncated"):
                # coding-containers caps each exec stream (1 MiB): a truncated
                # report is partial and must never be read as complete.
                failure = (
                    f"{tool.name} output truncated by coding-containers "
                    f"({result.get('bytes_omitted', 'unknown')} bytes omitted); report is incomplete"
                )
            if failure is not None:
                logger.warning("%s did not produce a usable result: %s", tool.name, failure)
                return {
                    "success": False,
                    "error": failure,
                    "issues": [],
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                    "execution_time": execution_time,
                    "exit_code": result.get("exit_code", 1),
                }

            return {
                "success": success,
                "issues": issues,
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "execution_time": execution_time,
                "exit_code": result.get("exit_code", 1),
            }

        except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            execution_time = (datetime.now(UTC) - start_time).total_seconds()
            logger.error("Tool %s execution failed: %s", tool.name, e)

            return {
                "success": False,
                "error": str(e),
                "issues": [],
                "execution_time": execution_time,
            }

    @staticmethod
    def _astral_failure(tool: QualityTool, result: dict[str, Any]) -> str | None:
        """Exit-contract verdict for ruff / ruff-format / ty (None = measured)."""
        stdout = result.get("stdout", "") or ""
        stderr = result.get("stderr", "") or ""
        exit_code = result.get("exit_code", 1)
        if tool.name == "ty":
            return ty_failure(exit_code, parse_ty_gitlab(stdout), stdout, stderr)
        if tool.name in ("ruff", "ruff-format"):
            return ruff_failure(exit_code, parse_ruff_json(stdout), stdout, stderr, tool=tool.name)
        return None

    async def _parse_tool_output(
        self, tool: QualityTool, stdout: str, stderr: str, filename: str
    ) -> list[QualityIssue]:
        """Parse tool output into standardized quality issues."""
        issues = []

        try:
            if tool.output_format == "json":
                issues = await self._parse_json_output(tool, stdout, filename)
            elif tool.output_format == "text":
                issues = await self._parse_text_output(tool, stdout, stderr, filename)
            elif tool.output_format == "dif":
                issues = await self._parse_diff_output(tool, stdout, filename)
            else:
                logger.warning("Unknown output format for %s: %s", tool.name, tool.output_format)

        except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Failed to parse output for %s: %s", tool.name, e)
            # Create error issue
            issues.append(
                QualityIssue(
                    tool=tool.name,
                    severity="error",
                    message=f"Failed to parse {tool.name} output: {str(e)}",
                    file_path=filename,
                    line_number=None,
                    column_number=None,
                    rule_id="parse_error",
                    suggested_fix=None,
                    fix_confidence=None,
                )
            )

        return issues

    async def _parse_json_output(
        self, tool: QualityTool, output: str, filename: str
    ) -> list[QualityIssue]:
        """Parse JSON format output."""
        issues: list[QualityIssue] = []

        if not output.strip():
            return issues

        try:
            data = json.loads(output)

            if tool.name in ("ruff", "ruff-format"):
                # rule_id is ruff's own code (F401, I001, PLR0913, unformatted ...)
                for diag in parse_ruff_json(output) or []:
                    issues.append(
                        QualityIssue(
                            tool=tool.name,
                            severity=self._map_ruff_severity(diag.code),
                            message=diag.message,
                            file_path=filename,
                            line_number=diag.line,
                            column_number=diag.column,
                            rule_id=diag.code,
                            suggested_fix=(
                                f"ruff check --fix --select {diag.code}"
                                if diag.fixable and tool.name == "ruff"
                                else ("ruff format" if tool.name == "ruff-format" else None)
                            ),
                            fix_confidence=None,
                        )
                    )

            elif tool.name == "ty":
                # rule_id is ty's rule name (invalid-argument-type ...).
                for ty_diag in parse_ty_gitlab(output) or []:
                    issues.append(
                        QualityIssue(
                            tool=tool.name,
                            severity="warning" if ty_diag.raw_severity == "minor" else "error",
                            message=ty_diag.message,
                            file_path=filename,
                            line_number=ty_diag.line,
                            column_number=ty_diag.column,
                            rule_id=ty_diag.rule,
                            suggested_fix=None,
                            fix_confidence=None,
                        )
                    )

            elif tool.name == "eslint":
                for file_result in data:
                    for message in file_result.get("messages", []):
                        issues.append(
                            QualityIssue(
                                tool=tool.name,
                                severity=("error" if message.get("severity") == 2 else "warning"),
                                message=message.get("message", ""),
                                file_path=filename,
                                line_number=message.get("line"),
                                column_number=message.get("column"),
                                rule_id=message.get("ruleId"),
                                suggested_fix=None,
                                fix_confidence=None,
                            )
                        )

            elif tool.name == "bandit":
                for result in data.get("results", []):
                    issues.append(
                        QualityIssue(
                            tool=tool.name,
                            severity=result.get("issue_severity", "medium").lower(),
                            message=result.get("issue_text", ""),
                            file_path=filename,
                            line_number=result.get("line_number"),
                            column_number=None,
                            rule_id=result.get("test_id"),
                            suggested_fix=None,
                            fix_confidence=None,
                        )
                    )

        except json.JSONDecodeError as e:
            logger.error("Invalid JSON from %s: %s", tool.name, e)

        return issues

    async def _parse_text_output(
        self, tool: QualityTool, stdout: str, stderr: str, filename: str
    ) -> list[QualityIssue]:
        """Parse text format output."""
        issues = []

        if tool.name == "prettier":
            # Formatters typically indicate issues through non-zero exit codes
            if stdout or stderr:
                issues.append(
                    QualityIssue(
                        tool=tool.name,
                        severity="warning",
                        message=f"Code formatting issues found by {tool.name}",
                        file_path=filename,
                        line_number=None,
                        column_number=None,
                        rule_id="formatting",
                        suggested_fix=None,
                        fix_confidence=None,
                    )
                )

        return issues

    async def _parse_diff_output(
        self, tool: QualityTool, output: str, filename: str
    ) -> list[QualityIssue]:
        """Parse diff format output."""
        issues = []

        if output.strip():
            issues.append(
                QualityIssue(
                    tool=tool.name,
                    severity="info",
                    message=f"Formatting changes suggested by {tool.name}",
                    file_path=filename,
                    line_number=None,
                    column_number=None,
                    rule_id="format_dif",
                    suggested_fix=None,
                    fix_confidence=None,
                )
            )

        return issues

    async def _cleanup_container(self, docker_id: str) -> None:
        """Cleanup quality analysis container."""
        try:
            await self.docker_service.stop_container(docker_id)
            await self.docker_service.remove_container_by_id(docker_id)
            logger.debug("Cleaned up quality container %s", docker_id)
        except (OSError, RuntimeError) as e:
            logger.warning("Failed to cleanup container %s: %s", docker_id, e)

    def _detect_language(self, request: QualityRequest) -> str:
        """Detect programming language from request."""
        if request.language:
            return request.language.lower()

        if request.code_path:
            extension = Path(request.code_path).suffix.lower()
            extension_map = {
                ".py": "python",
                ".js": "javascript",
                ".jsx": "javascript",
                ".ts": "typescript",
                ".tsx": "typescript",
            }
            return extension_map.get(extension, "python")

        return "python"

    def _select_tools(self, language: str, requested_tools: list[str] | None) -> list[str]:
        """Select quality tools to run based on language and request."""
        available_tools = self.tool_registry.list_available_tools(language)

        if requested_tools:
            # Filter to only requested tools that are available
            return [tool for tool in requested_tools if tool in available_tools]

        return available_tools

    def _get_target_filename(self, request: QualityRequest, language: str) -> str:
        """Get the target filename for analysis."""
        if request.code_path:
            return Path(request.code_path).name

        extension = self._get_file_extension(language)
        return f"main{extension}"

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language."""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
        }
        return extensions.get(language, ".py")

    def _calculate_quality_score(
        self, issues: list[QualityIssue], tool_results: dict[str, Any]
    ) -> float:
        """Calculate overall quality score based on issues found."""
        if not issues:
            return 10.0

        # tool_results parameter reserved for future scoring enhancements
        _ = tool_results
        # Weight issues by severity
        severity_weights = {
            "critical": 3.0,
            "error": 2.5,
            "warning": 1.0,
            "info": 0.3,
        }

        total_weight = sum(severity_weights.get(issue.severity.lower(), 1.0) for issue in issues)

        # Calculate score (10 - weighted penalty, minimum 0)
        score = max(0.0, 10.0 - (total_weight * 0.2))
        return round(score, 2)

    @staticmethod
    def _map_ruff_severity(code: str) -> str:
        """error/warning/info for a ruff code (pyflakes + pylint-error = error)."""
        if code.startswith(("F", "E9", "PLE")) or code == "invalid-syntax":
            return "error"
        if code.startswith(("E", "W", "PLW")):
            return "warning"
        return "info"

    def _initialize_container_images(self) -> dict[str, str]:
        """Initialize container image mapping for languages."""
        return {
            "python": "code_graph_helper-api:latest",
            "javascript": "node:18-slim",
            "typescript": "node:18-slim",
        }


# Factory function for dependency injection
def create_quality_tool_runner(
    docker_service: DockerService,
) -> QualityToolRunner:
    """Factory function to create QualityToolRunner."""
    return QualityToolRunner(docker_service)
