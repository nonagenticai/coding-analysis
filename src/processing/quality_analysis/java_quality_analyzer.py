"""
Java Quality Analyzer
Comprehensive quality analysis for Java projects using Checkstyle, PMD, SpotBugs, and other tools.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid7

import yaml

from models.testing_models import QualityIssue, QualityResult
from services.docker_service import DockerService

from .tool_registry import QualityTool, get_tool_registry

logger = logging.getLogger(__name__)


class JavaQualityAnalyzer:
    """
    Comprehensive Java quality analyzer supporting multiple tools and configurations.
    """

    def __init__(self, docker_service: DockerService):
        self.docker_service = docker_service
        self.tool_registry = get_tool_registry()
        self.config = self._load_java_quality_config()

    def _load_java_quality_config(self) -> dict[str, Any]:
        """Load Java quality configuration from YAML file."""
        config_path = (
            Path(__file__).parent.parent.parent.parent
            / "config"
            / "quality"
            / "java_quality_configs.yml"
        )
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                result = yaml.safe_load(f)
                return result if result is not None else {}
        return {}

    async def analyze_java_project(
        self,
        project_path: str,
        profile: str = "standard",
        tools: list[str] | None = None,
    ) -> QualityResult:
        """
        Analyze a Java project using specified quality tools.

        Args:
            project_path: Path to the Java project
            profile: Quality profile to use (strict, standard, basic, security_focused)
            tools: Specific tools to run (overrides profile)

        Returns:
            QualityResult with aggregated findings
        """
        logger.info(
            "Starting Java quality analysis for %s with profile '%s'",
            project_path,
            profile,
        )

        # Determine which tools to run
        tools_to_run = tools or self._get_tools_for_profile(profile)

        # Initialize container if needed
        container_id = await self._setup_java_container(project_path)

        try:
            # Run each tool
            all_issues = []
            tool_results = {}

            for tool_name in tools_to_run:
                try:
                    logger.info("Running %s analysis...", tool_name)
                    issues = await self._run_java_tool(container_id, tool_name, project_path)
                    all_issues.extend(issues)
                    tool_results[tool_name] = {
                        "status": "success",
                        "issues_count": len(issues),
                        "issues": issues,
                    }
                    logger.info("%s completed with %d issues", tool_name, len(issues))
                except (FileNotFoundError, PermissionError, OSError) as e:
                    logger.error("Error running %s: %s", tool_name, e)
                    tool_results[tool_name] = {
                        "status": "error",
                        "error": str(e),
                        "issues_count": 0,
                    }

            # Generate comprehensive report
            return self._create_quality_result(all_issues, tool_results, profile, project_path)

        finally:
            # Cleanup container
            if container_id:
                await self.docker_service.remove_container_by_id(container_id)

    def _get_tools_for_profile(self, profile: str) -> list[str]:
        """Get tools for specified quality profile."""
        profiles = self.config.get("quality_profiles", {})
        profile_config = profiles.get(profile, profiles.get("standard", {}))
        tools = profile_config.get("tools", ["checkstyle", "pmd", "spotbugs"])
        return tools if isinstance(tools, list) else ["checkstyle", "pmd", "spotbugs"]

    async def _setup_java_container(self, project_path: str) -> str:
        """Setup Java container with quality tools installed."""
        container_config = {
            "image": "openjdk:17-jdk-slim",
            "working_dir": "/workspace",
            "environment": {
                "JAVA_HOME": "/usr/local/openjdk-17",
                "JAVA_OPTS": "-Xmx2g -Xms512m",
                "MAVEN_OPTS": "-Xmx1g",
                "GRADLE_OPTS": "-Xmx1g",
            },
            "volumes": {project_path: "/workspace"},
            "command": ["sleep", "3600"],  # Keep container alive
        }

        container_id = await self.docker_service.create_test_container(container_config)
        await self.docker_service.start_container(container_id)

        # Install Java quality tools
        await self._install_java_tools(container_id)

        return container_id

    async def _install_java_tools(self, container_id: str):
        """Install Java quality tools in the container."""
        installation_commands = self.tool_registry.get_installation_commands("java")

        for command in installation_commands:
            try:
                result = await self.docker_service.exec_in_container(
                    container_id,
                    command.split() if isinstance(command, str) else command,
                    timeout=600,
                )
                if result["exit_code"] != 0:
                    logger.warning("Installation command failed: %s", command)
                    logger.warning("Error: %s", result["stderr"])
            except (FileNotFoundError, PermissionError, OSError) as e:
                logger.error("Error installing tools: %s", e)

    async def _run_java_tool(
        self, container_id: str, tool_name: str, project_path: str
    ) -> list[QualityIssue]:
        """Run a specific Java quality tool."""
        tool = self.tool_registry.get_tool("java", tool_name)
        if not tool:
            logger.warning("Tool %s not found in registry", tool_name)
            return []

        # Adjust command for the specific project structure
        command = self._prepare_tool_command(tool, project_path)

        try:
            result = await self.docker_service.exec_in_container(
                container_id, command, timeout=tool.timeout
            )

            # Parse tool output based on format
            return self._parse_tool_output(tool, result["stdout"], result["stderr"])

        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error("Error executing %s: %s", tool_name, e)
            return []

    def _prepare_tool_command(self, tool: QualityTool, _project_path: str) -> list[str]:
        """Prepare tool command for execution."""
        command = tool.command.copy()

        # Add project-specific arguments
        if tool.name == "checkstyle":
            command.extend(["/workspace"])  # Add source directory
        elif tool.name == "pmd":
            # PMD command needs adjustment for directory
            command = [cmd.replace(".", "/workspace") for cmd in command]
        elif tool.name == "spotbugs":
            # SpotBugs needs compiled classes
            command.extend(["/workspace/target/classes"])
        elif tool.name == "google-java-format":
            command.extend(["/workspace/**/*.java"])

        return command

    def _parse_tool_output(self, tool: QualityTool, stdout: str, stderr: str) -> list[QualityIssue]:
        """Parse tool output into QualityIssue objects."""
        issues = []

        try:
            if tool.output_format == "xml":
                issues = self._parse_xml_output(tool, stdout)
            elif tool.output_format == "json":
                issues = self._parse_json_output(tool, stdout)
            else:
                issues = self._parse_text_output(tool, stdout)

        except (ET.ParseError, ValueError) as e:
            logger.error("Error parsing %s output: %s", tool.name, e)
            # Fall back to text parsing
            issues = self._parse_text_output(tool, stdout + stderr)

        return issues

    def _parse_xml_output(self, tool: QualityTool, output: str) -> list[QualityIssue]:
        """Parse XML output from tools like Checkstyle, PMD, SpotBugs."""
        issues = []

        try:
            root = ET.fromstring(output)  # nosec B314

            if tool.name == "checkstyle":
                issues = self._parse_checkstyle_xml(root)
            elif tool.name == "pmd":
                issues = self._parse_pmd_xml(root)
            elif tool.name == "spotbugs":
                issues = self._parse_spotbugs_xml(root)

        except ET.ParseError as e:
            logger.error("XML parsing error for %s: %s", tool.name, e)

        return issues

    def _parse_checkstyle_xml(self, root: ET.Element) -> list[QualityIssue]:
        """Parse Checkstyle XML output."""
        issues = []

        for file_elem in root.findall("file"):
            file_path = file_elem.get("name", "")

            for error_elem in file_elem.findall("error"):
                issue = QualityIssue(
                    tool="checkstyle",
                    severity=error_elem.get("severity", "warning"),
                    message=error_elem.get("message", ""),
                    file_path=file_path,
                    line_number=int(error_elem.get("line", 0)),
                    column_number=int(error_elem.get("column", 0)),
                    rule_id=error_elem.get("source", ""),
                    suggested_fix=None,
                    fix_confidence=None,
                )
                issues.append(issue)

        return issues

    def _parse_pmd_xml(self, root: ET.Element) -> list[QualityIssue]:
        """Parse PMD XML output."""
        issues = []

        for file_elem in root.findall("file"):
            file_path = file_elem.get("name", "")

            for violation_elem in file_elem.findall("violation"):
                issue = QualityIssue(
                    tool="pmd",
                    severity=violation_elem.get("priority", "3"),
                    message=violation_elem.text or "",
                    file_path=file_path,
                    line_number=int(violation_elem.get("beginline", 0)),
                    column_number=int(violation_elem.get("begincolumn", 0)),
                    rule_id=violation_elem.get("rule", ""),
                    suggested_fix=None,
                    fix_confidence=None,
                )
                issues.append(issue)

        return issues

    def _parse_spotbugs_xml(self, root: ET.Element) -> list[QualityIssue]:
        """Parse SpotBugs XML output."""
        issues = []

        for bug_instance in root.findall(".//BugInstance"):
            # Get source line info
            source_line = bug_instance.find("SourceLine")
            file_path = source_line.get("sourcepath", "") if source_line is not None else ""
            line_number = int(source_line.get("start", 0)) if source_line is not None else 0

            # Get bug pattern info
            bug_pattern = bug_instance.find("ShortMessage")
            message = bug_pattern.text if bug_pattern is not None else ""

            issue = QualityIssue(
                tool="spotbugs",
                severity=bug_instance.get("priority", "Medium"),
                message=message or "SpotBugs issue detected",
                file_path=file_path,
                line_number=line_number,
                column_number=0,
                rule_id=bug_instance.get("type", ""),
                suggested_fix=None,
                fix_confidence=None,
            )
            issues.append(issue)

        return issues

    def _parse_json_output(self, tool: QualityTool, output: str) -> list[QualityIssue]:
        """Parse JSON output from tools."""
        issues: list[QualityIssue] = []

        try:
            json.loads(output)
            # JSON parsing logic would be tool-specific
            # This is a placeholder for tools that output JSON

        except json.JSONDecodeError as e:
            logger.error("JSON parsing error for %s: %s", tool.name, e)

        return issues

    def _parse_text_output(self, tool: QualityTool, output: str) -> list[QualityIssue]:
        """Parse text output from tools."""
        issues = []

        # Basic text parsing - extract file paths, line numbers, and messages
        lines = output.split("\n")

        for line in lines:
            if not line.strip():
                continue

            # Try to extract file:line:message pattern
            parts = line.split(":")
            if len(parts) >= 3:
                try:
                    file_path = parts[0]
                    line_number = int(parts[1]) if parts[1].isdigit() else 0
                    message = ":".join(parts[2:]).strip()

                    issue = QualityIssue(
                        tool=tool.name,
                        severity="warning",
                        message=message,
                        file_path=file_path,
                        line_number=line_number,
                        column_number=0,
                        rule_id="",
                        suggested_fix=None,
                        fix_confidence=None,
                    )
                    issues.append(issue)

                except ValueError:
                    continue

        return issues

    def _create_quality_result(
        self,
        all_issues: list[QualityIssue],
        tool_results: dict[str, Any],
        _profile: str,
        _project_path: str,
    ) -> QualityResult:
        """Create comprehensive quality result."""

        self._calculate_issue_metrics(all_issues)  # Calculate metrics for potential future use
        total_issues = len(all_issues)
        overall_score = max(0.0, 100.0 - (total_issues * 2.0))  # Basic scoring

        return QualityResult(
            analysis_id=str(uuid7()),
            success=True,
            overall_score=overall_score,
            total_issues=total_issues,
            issues=all_issues,
            tools_results=tool_results,
            execution_time=0.0,  # Would need to track actual time
            created_at=datetime.now(UTC),
            language="java",
            error=None,
        )

    def _calculate_issue_metrics(self, all_issues: list[QualityIssue]) -> dict[str, dict[str, int]]:
        """Calculate issue metrics by severity, tool, and category."""
        metrics: dict[str, dict[str, int]] = {
            "by_severity": {},
            "by_tool": {},
            "by_category": {},
        }

        for issue in all_issues:
            # Count by severity
            metrics["by_severity"][issue.severity] = (
                metrics["by_severity"].get(issue.severity, 0) + 1
            )

            # Count by tool
            metrics["by_tool"][issue.tool] = metrics["by_tool"].get(issue.tool, 0) + 1

            # Count by category (using tool as category)
            metrics["by_category"][issue.tool] = metrics["by_category"].get(issue.tool, 0) + 1

        return metrics

    async def generate_quality_report(
        self, quality_result: QualityResult, output_format: str = "html"
    ) -> str:
        """Generate a formatted quality report."""

        if output_format == "html":
            return self._generate_html_report(quality_result)
        if output_format == "json":
            return json.dumps(quality_result.model_dump(), indent=2)
        if output_format == "markdown":
            return self._generate_markdown_report(quality_result)
        return str(quality_result)

    def _generate_html_report(self, result: QualityResult) -> str:
        """Generate HTML quality report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Java Quality Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .issues {{ margin-top: 20px; }}
                .issue {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
                .error {{ border-color: #d32f2f; }}
                .warning {{ border-color: #f57c00; }}
                .info {{ border-color: #1976d2; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>Java Quality Analysis Report</h1>

            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Overall Score:</strong> {result.overall_score:.1f}%</p>
                <p><strong>Total Issues:</strong> {result.total_issues}</p>
                <p><strong>Language:</strong> {result.language}</p>

                <h3>Issues by Severity</h3>
                <table>
                    <tr><th>Severity</th><th>Count</th></tr>
        """

        # Group issues by severity for display
        issues_by_severity: dict[str, int] = {}
        for issue in result.issues:
            issues_by_severity[issue.severity] = issues_by_severity.get(issue.severity, 0) + 1

        for severity, count in issues_by_severity.items():
            html += f"<tr><td>{severity}</td><td>{count}</td></tr>"

        html += """
                </table>

                <h3>Issues by Tool</h3>
                <table>
                    <tr><th>Tool</th><th>Count</th></tr>
        """

        # Group issues by tool for display
        issues_by_tool: dict[str, int] = {}
        for issue in result.issues:
            issues_by_tool[issue.tool] = issues_by_tool.get(issue.tool, 0) + 1

        for tool, count in issues_by_tool.items():
            html += f"<tr><td>{tool}</td><td>{count}</td></tr>"

        html += """
                </table>
            </div>

            <div class="issues">
                <h2>Detailed Issues</h2>
        """

        for issue in result.issues[:50]:  # Limit to first 50 issues
            issue.severity.lower()
            html += """
                <div class="issue {css_class}">
                    <strong>[{issue.tool}] {issue.severity.upper()}</strong><br>
                    <strong>File:</strong> {issue.file_path}:{issue.line_number}<br>
                    <strong>Rule:</strong> {issue.rule_id}<br>
                    <strong>Message:</strong> {issue.message}
                </div>
            """

        html += """
            </div>
        </body>
        </html>
        """

        return html

    def _generate_markdown_report(self, result: QualityResult) -> str:
        """Generate Markdown quality report."""
        md = """# Java Quality Analysis Report

## Summary
- **Overall Score:** {result.overall_score:.1f}%
- **Total Issues:** {result.total_issues}
- **Language:** {result.language}

## Issues by Severity
| Severity | Count |
|----------|-------|
"""

        # Group issues by severity for markdown
        issues_by_severity: dict[str, int] = {}
        for issue in result.issues:
            issues_by_severity[issue.severity] = issues_by_severity.get(issue.severity, 0) + 1

        for severity, count in issues_by_severity.items():
            md += f"| {severity} | {count} |\n"

        md += "\n## Issues by Tool\n| Tool | Count |\n|------|-------|\n"

        # Group issues by tool for markdown
        issues_by_tool: dict[str, int] = {}
        for issue in result.issues:
            issues_by_tool[issue.tool] = issues_by_tool.get(issue.tool, 0) + 1

        for tool, count in issues_by_tool.items():
            md += f"| {tool} | {count} |\n"

        md += "\n## Detailed Issues\n\n"

        for issue in result.issues[:20]:  # Limit to first 20
            md += f"""### Issue: {issue.severity.upper()}
- **Tool:** {issue.tool}
- **File:** {issue.file_path}:{issue.line_number}
- **Rule:** {issue.rule_id}
- **Message:** {issue.message}

"""

        return md
