"""Pysa (Python Static Analyzer) integration for symbolic analysis."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..exceptions import PysaAnalysisError
from ..models import SecurityIssue, SymbolicAnalysisResult

logger = logging.getLogger(__name__)


class PysaAnalyzer:
    """Python Static Analyzer (Pysa) integration for taint analysis and security issues."""

    def __init__(self, tool_registry: Any = None, project_root: Path | None = None) -> None:
        """Initialize Pysa analyzer.

        Args:
            tool_registry: Optional tool registry for tool management
            project_root: Root directory of the project to analyze (defaults to cwd)
        """
        self.tool_registry = tool_registry
        self.project_root = project_root or Path.cwd()
        self.pysa_config_path = self.project_root / ".pyre_configuration"
        self._setup_pysa_config()

    def _setup_pysa_config(self) -> None:
        """Setup Pysa configuration if not present."""
        if not self.pysa_config_path.exists():
            default_config = {
                "source_directories": ["src"],
                "taint_models_path": ["stubs/taint"],
                "search_path": ["stubs"],
                "strict": False,
            }
            self.pysa_config_path.write_text(json.dumps(default_config, indent=2))
            logger.info(f"Created default Pysa configuration at {self.pysa_config_path}")

    def analyze(self, file_path: Path) -> SymbolicAnalysisResult:
        """Run Pysa analysis on a Python file.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            Analysis results containing security issues and taint flows
        """
        if not file_path.suffix == ".py":
            logger.warning(f"Pysa can only analyze Python files, skipping {file_path}")
            return self._create_empty_result(file_path)

        try:
            # Run Pysa analysis
            result = subprocess.run(
                ["pyre", "analyze", "--no-verify", str(file_path)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                logger.error(f"Pysa analysis failed: {result.stderr}")
                return self._create_empty_result(file_path)

            # Parse Pysa output
            issues = self._parse_pysa_output(result.stdout, file_path)
            taint_flows = self._extract_taint_flows(result.stdout)

            return SymbolicAnalysisResult(
                file_path=str(file_path),
                security_issues=issues,
                taint_flows=taint_flows,
                metrics={
                    "total_issues": len(issues),
                    "critical_issues": sum(1 for i in issues if i.severity == "critical"),
                    "high_issues": sum(1 for i in issues if i.severity == "high"),
                    "medium_issues": sum(1 for i in issues if i.severity == "medium"),
                    "low_issues": sum(1 for i in issues if i.severity == "low"),
                    "taint_sources": len(set(tf.get("source", "") for tf in taint_flows)),
                    "taint_sinks": len(set(tf.get("sink", "") for tf in taint_flows)),
                },
            )

        except subprocess.TimeoutExpired:
            logger.error(f"Pysa analysis timed out for {file_path}")
            return self._create_empty_result(file_path)
        except Exception as e:
            logger.error(f"Error running Pysa analysis: {e}")
            return self._create_empty_result(file_path)

    async def analyze_legacy(self, file_path: str) -> dict[str, Any]:
        """Run Pysa taint analysis."""
        try:
            tool = self.tool_registry.get_tool("python", "pysa")
            if not tool:
                return {
                    "status": "tool_not_found",
                    "error": "Pysa not available",
                }

            # Pysa requires a project directory, so create a temporary project
            temp_dir = tempfile.mkdtemp()
            try:
                # Copy file to temp directory
                temp_file = os.path.join(temp_dir, "target.py")
                with (
                    open(file_path, encoding="utf-8") as src,
                    open(temp_file, "w", encoding="utf-8") as dst,
                ):
                    dst.write(src.read())

                # Create minimal Pyre configuration
                pyre_config = {
                    "source_directories": ["."],
                    "taint_models_path": [],
                }

                config_path = os.path.join(temp_dir, ".pyre_configuration")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(pyre_config, f)

                # Run Pysa
                cmd = tool.command.copy()
                result = subprocess.run(
                    cmd,
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=tool.timeout,
                    check=False,
                )

                # Parse Pysa output
                if result.stdout:
                    try:
                        pysa_output = json.loads(result.stdout)
                        return self._parse_output(pysa_output)
                    except json.JSONDecodeError:
                        return {
                            "status": "completed",
                            "issues": [],
                            "raw_output": result.stdout,
                        }
                else:
                    return {
                        "status": "no_output",
                        "issues": [],
                        "stderr": result.stderr,
                    }

            finally:
                # Cleanup temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)

        except subprocess.TimeoutExpired as e:
            logger.warning("Pysa analysis timeout for %s", file_path)
            raise PysaAnalysisError("Pysa analysis timeout") from e
        except subprocess.CalledProcessError as e:
            logger.error("Pysa process error: %s", e)
            raise PysaAnalysisError(f"Pysa process error: {e}") from e
        except OSError as e:
            logger.error("IO error running Pysa: %s", e)
            raise PysaAnalysisError(f"IO error: {e}") from e

    def _parse_output(self, pysa_data: dict[str, Any]) -> dict[str, Any]:
        """Parse Pysa output for security issues."""
        parsed_issues = []
        safety_score = 100

        if "errors" in pysa_data:
            for error in pysa_data["errors"]:
                issue = self._parse_security_issue(error)
                parsed_issues.append(issue)
                # Reduce safety score based on severity
                if issue.get("severity") == "high":
                    safety_score -= 20
                elif issue.get("severity") == "medium":
                    safety_score -= 10
                else:
                    safety_score -= 5

        return {
            "status": "completed",
            "issues": parsed_issues,
            "safety_score": max(0, safety_score),
            "recommendations": self._generate_recommendations(parsed_issues),
        }

    def _parse_security_issue(self, error: dict[str, Any]) -> dict[str, Any]:
        """Parse individual security issue from Pysa error."""
        issue_type = error.get("name", "Unknown")
        line = error.get("line", 0)
        column = error.get("column", 0)
        description = error.get("description", "")
        severity = self._determine_severity(issue_type)

        return {
            "type": issue_type,
            "severity": severity,
            "line": line,
            "column": column,
            "message": description,
            "recommendation": self._get_recommendation(issue_type),
        }

    def _determine_severity(self, issue_type: str) -> str:
        """Determine severity based on issue type."""
        high_severity = ["SQL", "Command", "XSS", "RCE"]
        medium_severity = ["Taint", "Path", "SSRF"]

        if any(s in issue_type for s in high_severity):
            return "high"
        elif any(s in issue_type for s in medium_severity):
            return "medium"
        return "low"

    def _get_recommendation(self, issue_type: str) -> str:
        """Get security recommendation for issue type."""
        recommendations = {
            "SQL": "Use parameterized queries to prevent SQL injection",
            "Command": "Use subprocess with shell=False and validate input",
            "XSS": "Sanitize user input before rendering in HTML",
            "Path": "Validate and sanitize file paths to prevent traversal",
            "Taint": "Track and validate data flow from untrusted sources",
        }

        for key, rec in recommendations.items():
            if key in issue_type:
                return rec

        return "Review and validate data flow for security implications"

    def _generate_recommendations(self, issues: list) -> list:
        """Generate recommendations based on found issues."""
        recommendations = []
        issue_types = {issue["type"] for issue in issues}

        for issue_type in issue_types:
            rec = self._get_recommendation(issue_type)
            if rec not in recommendations:
                recommendations.append(rec)

        return recommendations

    def _parse_pysa_output(self, output: str, file_path: Path) -> list[SecurityIssue]:
        """Parse Pysa output to extract security issues.

        Args:
            output: Raw output from Pysa
            file_path: Path to the analyzed file

        Returns:
            List of security issues found
        """
        issues = []

        try:
            # Pysa output is typically in JSON format
            data = json.loads(output)

            for error in data.get("errors", []):
                if error.get("path", "").endswith(file_path.name):
                    issue = SecurityIssue(
                        type=self._map_issue_type(error.get("code", "")),
                        severity=self._map_severity(error.get("level", "warning")),
                        message=error.get("description", ""),
                        file_path=str(file_path),
                        line_number=error.get("line", 0),
                        column_number=error.get("column", 0),
                        code_snippet=error.get("code_snippet", ""),
                        recommendation=self._get_recommendation_enhanced(error.get("code", "")),
                        cwe_id=self._map_cwe_id(error.get("code", "")),
                    )
                    issues.append(issue)

        except json.JSONDecodeError:
            # Fallback to line-based parsing if JSON parsing fails
            lines = output.strip().split("\n")
            for line in lines:
                if file_path.name in line and ":" in line:
                    issue = self._parse_issue_line(line, file_path)
                    if issue:
                        issues.append(issue)

        return issues

    def _extract_taint_flows(self, output: str) -> list[dict[str, Any]]:
        """Extract taint flow information from Pysa output.

        Args:
            output: Raw output from Pysa

        Returns:
            List of taint flows
        """
        taint_flows = []

        try:
            data = json.loads(output)

            for flow in data.get("taint_flows", []):
                taint_flow = {
                    "source": flow.get("source", {}).get("name", "unknown"),
                    "source_location": {
                        "file": flow.get("source", {}).get("location", {}).get("file", ""),
                        "line": flow.get("source", {}).get("location", {}).get("line", 0),
                    },
                    "sink": flow.get("sink", {}).get("name", "unknown"),
                    "sink_location": {
                        "file": flow.get("sink", {}).get("location", {}).get("file", ""),
                        "line": flow.get("sink", {}).get("location", {}).get("line", 0),
                    },
                    "path": flow.get("path", []),
                    "issue_type": flow.get("issue_type", ""),
                }
                taint_flows.append(taint_flow)

        except json.JSONDecodeError:
            logger.warning("Could not parse taint flows from Pysa output")

        return taint_flows

    def _parse_issue_line(self, line: str, file_path: Path) -> SecurityIssue | None:
        """Parse a single issue line from Pysa output.

        Args:
            line: Line containing issue information
            file_path: Path to the analyzed file

        Returns:
            SecurityIssue if successfully parsed, None otherwise
        """
        # Example format: "file.py:10:5: Issue: SQL injection vulnerability"
        parts = line.split(":", 3)
        if len(parts) >= 4:
            try:
                line_num = int(parts[1])
                col_num = int(parts[2])
                message = parts[3].strip()

                return SecurityIssue(
                    type=self._extract_issue_type(message),
                    severity="high",  # Default severity
                    message=message,
                    file_path=str(file_path),
                    line_number=line_num,
                    column_number=col_num,
                    code_snippet="",
                    recommendation=self._get_recommendation_from_message(message),
                    cwe_id=self._extract_cwe_from_message(message),
                )
            except (ValueError, IndexError):  # fmt: skip
                pass

        return None

    def _map_issue_type(self, code: str) -> str:
        """Map Pysa issue code to issue type.

        Args:
            code: Pysa issue code

        Returns:
            Human-readable issue type
        """
        issue_map = {
            "5001": "SQL Injection",
            "5002": "Command Injection",
            "5003": "Path Traversal",
            "5004": "Cross-Site Scripting (XSS)",
            "5005": "Remote Code Execution",
            "5006": "Server-Side Request Forgery (SSRF)",
            "5007": "XML External Entity (XXE)",
            "5008": "Insecure Deserialization",
            "5009": "LDAP Injection",
            "5010": "NoSQL Injection",
        }
        return issue_map.get(code, f"Security Issue ({code})")

    def _map_severity(self, level: str) -> str:
        """Map Pysa severity level to standard severity.

        Args:
            level: Pysa severity level

        Returns:
            Standard severity level
        """
        severity_map = {
            "error": "critical",
            "warning": "high",
            "advice": "medium",
            "info": "low",
        }
        return severity_map.get(level.lower(), "medium")

    def _map_cwe_id(self, code: str) -> str | None:
        """Map Pysa issue code to CWE ID.

        Args:
            code: Pysa issue code

        Returns:
            CWE ID if found, None otherwise
        """
        cwe_map = {
            "5001": "CWE-89",  # SQL Injection
            "5002": "CWE-78",  # OS Command Injection
            "5003": "CWE-22",  # Path Traversal
            "5004": "CWE-79",  # Cross-site Scripting
            "5005": "CWE-94",  # Code Injection
            "5006": "CWE-918",  # Server-Side Request Forgery
            "5007": "CWE-611",  # XML External Entity
            "5008": "CWE-502",  # Deserialization of Untrusted Data
            "5009": "CWE-90",  # LDAP Injection
            "5010": "CWE-943",  # NoSQL Injection
        }
        return cwe_map.get(code)

    def _get_recommendation_enhanced(self, code: str) -> str:
        """Get recommendation for fixing the issue.

        Args:
            code: Pysa issue code

        Returns:
            Recommendation text
        """
        recommendations = {
            "5001": "Use parameterized queries or prepared statements to prevent SQL injection.",
            "5002": "Validate and sanitize user input. Use subprocess with shell=False.",
            "5003": "Validate file paths and use os.path.join() to prevent directory traversal.",
            "5004": "Escape user input before rendering in HTML. Use template auto-escaping.",
            "5005": "Never execute user-controlled input as code. Use safe alternatives.",
            "5006": "Validate and whitelist URLs. Implement SSRF protection.",
            "5007": "Disable XML external entity processing. Use defusedxml library.",
            "5008": "Avoid deserializing untrusted data. Use safe serialization formats.",
            "5009": "Escape special characters in LDAP queries. Use parameterized LDAP queries.",
            "5010": "Sanitize user input in NoSQL queries. Use query builders.",
        }
        return recommendations.get(
            code, "Review and fix the security issue according to best practices."
        )

    def _extract_issue_type(self, message: str) -> str:
        """Extract issue type from message text.

        Args:
            message: Issue message

        Returns:
            Issue type
        """
        message_lower = message.lower()

        if "nosql injection" in message_lower:
            return "NoSQL Injection"
        elif "sql injection" in message_lower:
            return "SQL Injection"
        elif "command injection" in message_lower:
            return "Command Injection"
        elif "path traversal" in message_lower or "directory traversal" in message_lower:
            return "Path Traversal"
        elif "xss" in message_lower or "cross-site scripting" in message_lower:
            return "Cross-Site Scripting (XSS)"
        elif "remote code execution" in message_lower or "rce" in message_lower:
            return "Remote Code Execution"
        elif "ssrf" in message_lower or "server-side request forgery" in message_lower:
            return "Server-Side Request Forgery (SSRF)"
        elif "xxe" in message_lower or "xml external entity" in message_lower:
            return "XML External Entity (XXE)"
        elif "deserialization" in message_lower:
            return "Insecure Deserialization"
        elif "ldap injection" in message_lower:
            return "LDAP Injection"
        else:
            return "Security Issue"

    def _get_recommendation_from_message(self, message: str) -> str:
        """Get recommendation based on message content.

        Args:
            message: Issue message

        Returns:
            Recommendation text
        """
        message_lower = message.lower()

        if "sql" in message_lower:
            return "Use parameterized queries or prepared statements."
        elif "command" in message_lower:
            return "Validate input and use safe command execution methods."
        elif "path" in message_lower or "directory" in message_lower:
            return "Validate file paths and prevent directory traversal."
        elif "xss" in message_lower or "script" in message_lower:
            return "Escape user input in HTML contexts."
        else:
            return "Review and fix the security issue."

    def _extract_cwe_from_message(self, message: str) -> str | None:
        """Extract CWE ID from message if present.

        Args:
            message: Issue message

        Returns:
            CWE ID if found, None otherwise
        """
        import re

        # Look for CWE-XXX pattern in message
        match = re.search(r"CWE-(\d+)", message)
        if match:
            return f"CWE-{match.group(1)}"

        return None

    def _create_empty_result(self, file_path: Path) -> SymbolicAnalysisResult:
        """Create empty result for when analysis fails or is skipped.

        Args:
            file_path: Path to the file

        Returns:
            Empty analysis result
        """
        return SymbolicAnalysisResult(
            file_path=str(file_path),
            security_issues=[],
            taint_flows=[],
            metrics={
                "total_issues": 0,
                "critical_issues": 0,
                "high_issues": 0,
                "medium_issues": 0,
                "low_issues": 0,
                "taint_sources": 0,
                "taint_sinks": 0,
            },
        )

    def get_custom_taint_models(self) -> list[dict[str, Any]]:
        """Get custom taint models for the project.

        Returns:
            List of custom taint model definitions
        """
        return [
            {
                "sources": [
                    {"name": "flask.request.args"},
                    {"name": "flask.request.form"},
                    {"name": "flask.request.json"},
                    {"name": "django.http.HttpRequest.GET"},
                    {"name": "django.http.HttpRequest.POST"},
                ],
                "sinks": [
                    {"name": "subprocess.call"},
                    {"name": "subprocess.run"},
                    {"name": "os.system"},
                    {"name": "eval"},
                    {"name": "exec"},
                ],
                "rules": [
                    {
                        "name": "User input to command execution",
                        "sources": ["flask.request.*", "django.http.HttpRequest.*"],
                        "sinks": ["subprocess.*", "os.system"],
                        "message": "User input flows to command execution",
                    }
                ],
            }
        ]

    def generate_taint_config(self, output_path: Path) -> None:
        """Generate Pysa taint configuration for the project.

        Args:
            output_path: Path to write the taint configuration
        """
        taint_config = {
            "sources": self._get_project_sources(),
            "sinks": self._get_project_sinks(),
            "features": self._get_project_features(),
            "rules": self._get_project_rules(),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(taint_config, indent=2))
        logger.info(f"Generated Pysa taint configuration at {output_path}")

    def _get_project_sources(self) -> list[dict[str, Any]]:
        """Get taint sources for the project.

        Returns:
            List of taint source definitions
        """
        return [
            {"name": "UserControlled", "comment": "Input from users"},
            {"name": "RemoteData", "comment": "Data from external sources"},
            {"name": "FileSystem", "comment": "Data from file system"},
            {"name": "Database", "comment": "Data from database"},
        ]

    def _get_project_sinks(self) -> list[dict[str, Any]]:
        """Get taint sinks for the project.

        Returns:
            List of taint sink definitions
        """
        return [
            {"name": "SQL", "comment": "SQL query execution"},
            {"name": "Shell", "comment": "Shell command execution"},
            {"name": "FileSystem", "comment": "File system operations"},
            {"name": "Network", "comment": "Network operations"},
            {"name": "Eval", "comment": "Code evaluation"},
        ]

    def _get_project_features(self) -> list[dict[str, Any]]:
        """Get taint features for the project.

        Returns:
            List of taint feature definitions
        """
        return [
            {"name": "Sanitized", "comment": "Data has been sanitized"},
            {"name": "Validated", "comment": "Data has been validated"},
            {"name": "Encoded", "comment": "Data has been encoded"},
        ]

    def _get_project_rules(self) -> list[dict[str, Any]]:
        """Get taint rules for the project.

        Returns:
            List of taint rule definitions
        """
        return [
            {
                "name": "SQL Injection",
                "code": 5001,
                "sources": ["UserControlled"],
                "sinks": ["SQL"],
                "message_format": "User input flows to SQL query",
            },
            {
                "name": "Command Injection",
                "code": 5002,
                "sources": ["UserControlled"],
                "sinks": ["Shell"],
                "message_format": "User input flows to shell command",
            },
            {
                "name": "Path Traversal",
                "code": 5003,
                "sources": ["UserControlled"],
                "sinks": ["FileSystem"],
                "message_format": "User input flows to file system operation",
            },
        ]
