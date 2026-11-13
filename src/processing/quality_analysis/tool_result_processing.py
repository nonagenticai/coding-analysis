"""
Tool result processing and formatting.
Handles parsing, formatting, and normalization of tool outputs.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from typing import Any

from processing.utils.ty_output import parse_ty_rows

from .tool_definitions import QualityTool

logger = logging.getLogger(__name__)


class ToolResultProcessor:
    """Processes and formats results from quality analysis tools."""

    def __init__(self):
        self.supported_formats = ["json", "xml", "text", "html", "yaml"]

    def process_tool_result(
        self, tool: QualityTool, raw_output: str, exit_code: int = 0
    ) -> dict[str, Any]:
        """Process raw tool output into structured format.

        Args:
            tool: The QualityTool that generated the output
            raw_output: Raw output from the tool
            exit_code: Tool exit code

        Returns:
            Structured result dictionary
        """
        processed_result = {
            "tool_name": tool.name,
            "tool_category": tool.category.value,
            "exit_code": exit_code,
            "success": exit_code == 0,
            "raw_output": raw_output,
            "processed_output": None,
            "issues": [],
            "metrics": {},
            "summary": {},
        }

        try:
            if tool.output_format.lower() == "json":
                processed_result["processed_output"] = self._process_json_output(raw_output)
            elif tool.output_format.lower() == "xml":
                processed_result["processed_output"] = self._process_xml_output(raw_output)
            elif tool.output_format.lower() in ["text", "txt"]:
                processed_result["processed_output"] = self._process_text_output(raw_output)
            else:
                processed_result["processed_output"] = raw_output

            # Extract issues and metrics based on tool type
            processed_result["issues"] = self._extract_issues(
                tool, processed_result["processed_output"]
            )
            processed_result["metrics"] = self._extract_metrics(
                tool, processed_result["processed_output"]
            )
            processed_result["summary"] = self._generate_summary(processed_result)

        except Exception as e:
            logger.error(f"Error processing {tool.name} output: {str(e)}")
            processed_result["error"] = str(e)

        return processed_result

    def _process_json_output(self, raw_output: str) -> dict[str, Any] | list[Any]:
        """Process JSON output from tools."""
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON output: {str(e)}")
            return {"parse_error": str(e), "raw": raw_output}

    def _process_xml_output(self, raw_output: str) -> dict[str, Any] | str:
        """Process XML output from tools."""
        try:
            root = ET.fromstring(raw_output)  # nosec B314
            return self._xml_to_dict(root)
        except ET.ParseError as e:
            logger.warning(f"Failed to parse XML output: {str(e)}")
            return {"parse_error": str(e), "raw": raw_output}

    def _process_text_output(self, raw_output: str) -> list[str]:
        """Process text output from tools."""
        return raw_output.strip().split("\n") if raw_output.strip() else []

    def _xml_to_dict(self, element: ET.Element) -> dict[str, Any] | str:
        """Convert XML element to dictionary."""
        result = {}

        # Add attributes
        if element.attrib:
            result.update(element.attrib)

        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:
                return element.text.strip()
            result["text"] = element.text.strip()

        # Add child elements
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data

        return result

    def _extract_issues(self, tool: QualityTool, processed_output: Any) -> list[dict[str, Any]]:
        """Extract issues/violations from processed output."""
        issues = []

        try:
            if tool.name in ("ruff", "ruff-format") and isinstance(processed_output, list):
                for issue in processed_output:
                    if isinstance(issue, dict):
                        location = issue.get("location") or {}
                        issues.append(
                            {
                                "file": issue.get("filename", ""),
                                "line": location.get("row", 0),
                                "column": location.get("column", 0),
                                "severity": issue.get("severity", ""),
                                "message": issue.get("message", ""),
                                "rule": issue.get("code", ""),
                            }
                        )

            elif tool.name == "ty" and isinstance(processed_output, list):
                for diag in parse_ty_rows(processed_output) or []:
                    issues.append(
                        {
                            "file": diag.file,
                            "line": diag.line,
                            "column": diag.column or 0,
                            "severity": diag.raw_severity,
                            "message": diag.message,
                            "rule": diag.rule,
                        }
                    )

            elif tool.name == "eslint" and isinstance(processed_output, list):
                for file_result in processed_output:
                    if isinstance(file_result, dict) and "messages" in file_result:
                        for message in file_result["messages"]:
                            issues.append(
                                {
                                    "file": file_result.get("filePath", ""),
                                    "line": message.get("line", 0),
                                    "column": message.get("column", 0),
                                    "severity": message.get("severity", 1),
                                    "message": message.get("message", ""),
                                    "rule": message.get("ruleId", ""),
                                }
                            )

            # Add more tool-specific parsing as needed

        except Exception as e:
            logger.warning(f"Failed to extract issues from {tool.name}: {str(e)}")

        return issues

    def _extract_metrics(self, tool: QualityTool, processed_output: Any) -> dict[str, Any]:
        """Extract metrics from processed output."""
        metrics = {}

        try:
            if tool.category.value == "complexity":
                if tool.name == "radon" and isinstance(processed_output, dict):
                    # Extract complexity metrics from radon
                    total_complexity = 0
                    file_count = 0

                    for file_path, file_data in processed_output.items():
                        if isinstance(file_data, list):
                            for item in file_data:
                                if isinstance(item, dict) and "complexity" in item:
                                    total_complexity += item["complexity"]
                            file_count += 1

                    metrics = {
                        "total_complexity": total_complexity,
                        "average_complexity": total_complexity / max(file_count, 1),
                        "file_count": file_count,
                    }

            elif tool.category.value == "security":
                if isinstance(processed_output, dict) and "results" in processed_output:
                    issues = processed_output["results"]
                    metrics = {
                        "total_issues": len(issues),
                        "high_severity": len(
                            [i for i in issues if i.get("issue_severity") == "HIGH"]
                        ),
                        "medium_severity": len(
                            [i for i in issues if i.get("issue_severity") == "MEDIUM"]
                        ),
                        "low_severity": len(
                            [i for i in issues if i.get("issue_severity") == "LOW"]
                        ),
                    }

        except Exception as e:
            logger.warning(f"Failed to extract metrics from {tool.name}: {str(e)}")

        return metrics

    def _generate_summary(self, result: dict[str, Any]) -> dict[str, Any]:
        """Generate summary information from processed result."""
        summary = {
            "tool_name": result["tool_name"],
            "success": result["success"],
            "issue_count": len(result["issues"]),
            "has_metrics": bool(result["metrics"]),
        }

        if result["issues"]:
            # Count issues by severity
            severity_counts = {}
            for issue in result["issues"]:
                severity = str(issue.get("severity", "unknown"))
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            summary["severity_breakdown"] = severity_counts

        return summary

    def normalize_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Normalize results from multiple tools into a unified format.

        Args:
            results: List of tool results

        Returns:
            Normalized results summary
        """
        normalized = {
            "total_tools": len(results),
            "successful_tools": len([r for r in results if r["success"]]),
            "failed_tools": len([r for r in results if not r["success"]]),
            "total_issues": sum(len(r["issues"]) for r in results),
            "issues_by_category": {},
            "issues_by_severity": {},
            "tool_summaries": [],
            "overall_score": 0.0,
        }

        # Aggregate issues by category and severity
        for result in results:
            category = result["tool_category"]
            if category not in normalized["issues_by_category"]:
                normalized["issues_by_category"][category] = 0
            normalized["issues_by_category"][category] += len(result["issues"])

            # Add tool summary
            normalized["tool_summaries"].append(result["summary"])

            # Aggregate severity counts
            for issue in result["issues"]:
                severity = str(issue.get("severity", "unknown"))
                if severity not in normalized["issues_by_severity"]:
                    normalized["issues_by_severity"][severity] = 0
                normalized["issues_by_severity"][severity] += 1

        # Calculate overall score (simple implementation)
        if normalized["total_tools"] > 0:
            success_rate = normalized["successful_tools"] / normalized["total_tools"]
            # Score decreases with more issues, but success rate is primary factor
            issue_penalty = min(normalized["total_issues"] * 0.01, 0.5)
            normalized["overall_score"] = max(0.0, success_rate - issue_penalty)

        return normalized

    def format_for_output(self, result: dict[str, Any], format_type: str = "json") -> str:
        """Format result for different output types.

        Args:
            result: Processed result dictionary
            format_type: Output format (json, yaml, text)

        Returns:
            Formatted result string
        """
        if format_type.lower() == "json":
            return json.dumps(result, indent=2)
        elif format_type.lower() == "text":
            return self._format_text_report(result)
        elif format_type.lower() == "yaml":
            try:
                import yaml

                return yaml.dump(result, default_flow_style=False)
            except ImportError:
                logger.warning("YAML library not available, falling back to JSON")
                return json.dumps(result, indent=2)
        else:
            return str(result)

    def _format_text_report(self, result: dict[str, Any]) -> str:
        """Format result as human-readable text report."""
        lines = []
        lines.append(f"Tool: {result['tool_name']}")
        lines.append(f"Category: {result['tool_category']}")
        lines.append(f"Success: {result['success']}")
        lines.append(f"Issues Found: {len(result['issues'])}")

        if result["issues"]:
            lines.append("\nIssues:")
            for i, issue in enumerate(result["issues"][:10], 1):  # Limit to first 10
                lines.append(
                    f"  {i}. {issue.get('file', 'unknown')}:{issue.get('line', 0)} - {issue.get('message', 'No message')}"
                )

            if len(result["issues"]) > 10:
                lines.append(f"  ... and {len(result['issues']) - 10} more issues")

        if result["metrics"]:
            lines.append(f"\nMetrics: {result['metrics']}")

        return "\n".join(lines)
