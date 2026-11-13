"""Module for complexity analysis and calculations."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from models.architecture_models import ArchitectureQualityIssue

logger = logging.getLogger(__name__)


class ComplexityAnalysis:
    """Handler for complexity analysis and calculations."""

    def __init__(self) -> None:
        """Initialize the complexity analyzer."""
        self.complexity_thresholds = {
            "low": 5,
            "medium": 10,
            "high": 15,
            "critical": 20,
        }

    def run_radon_complexity(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run Radon complexity analyzer."""
        issues: list[ArchitectureQualityIssue] = []
        result: dict[str, Any] = {"success": False}

        try:
            cmd = [
                "radon",
                "cc",
                str(repo_path),
                "-j",
                "-n",
                "C",
            ]  # Only report complexity >= C
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)

            if proc.stdout:
                try:
                    report = json.loads(proc.stdout)
                    for file_path, functions in report.items():
                        for func_data in functions:
                            if func_data["complexity"] >= 10:  # Only report high complexity
                                issue = ArchitectureQualityIssue(
                                    id=f"radon-cc-{len(issues)}",
                                    severity=(
                                        "high" if func_data["complexity"] >= 20 else "medium"
                                    ),
                                    message=f"Function '{func_data['name']}' has complexity {func_data['complexity']}",
                                    rule_id="complex-function",
                                    rule_name="High Complexity",
                                    file_path=file_path,
                                    line_number=func_data["lineno"],
                                    column_number=func_data.get("col_offset", 1),
                                    tool="radon",
                                    category="maintainability",
                                    description=f"Cyclomatic complexity of {func_data['complexity']} is too high",
                                    resolve_suggestions="Consider breaking this function into smaller, more focused functions",
                                )
                                issues.append(issue)
                    result["success"] = True
                except json.JSONDecodeError:
                    logger.error("Failed to parse Radon output")

        except subprocess.TimeoutExpired:
            logger.error("Radon timed out")
        except FileNotFoundError:
            logger.warning("Radon not found, skipping")

        return issues, result

    def calculate_cyclomatic_complexity(
        self, code_content: str, language: str = "python"
    ) -> dict[str, Any]:
        """Calculate cyclomatic complexity metrics for code content."""
        complexity_data = {
            "total_functions": 0,
            "avg_complexity": 0.0,
            "max_complexity": 0,
            "high_complexity_functions": 0,
            "complexity_distribution": {
                "low": 0,  # 1-5
                "medium": 0,  # 6-10
                "high": 0,  # 11-15
                "critical": 0,  # 16+
            },
        }

        if language == "python":
            complexity_data = self._analyze_python_complexity(code_content)
        elif language in ["javascript", "typescript"]:
            complexity_data = self._analyze_js_complexity(code_content)

        return complexity_data

    def _analyze_python_complexity(self, code_content: str) -> dict[str, Any]:
        """Analyze Python code complexity using simple heuristics."""
        lines = code_content.split("\n")
        functions = []
        current_function = None
        current_complexity = 1  # Base complexity

        for line in lines:
            stripped = line.strip()

            # Function definition
            if stripped.startswith("def ") or stripped.startswith("async def "):
                if current_function is not None:
                    functions.append({"name": current_function, "complexity": current_complexity})
                current_function = (
                    stripped.split("(")[0].replace("def ", "").replace("async def ", "")
                )
                current_complexity = 1

            # Complexity adding statements
            elif current_function is not None:
                if any(
                    keyword in stripped
                    for keyword in [
                        "if ",
                        "elif ",
                        "while ",
                        "for ",
                        "except ",
                        "and ",
                        "or ",
                    ]
                ):
                    current_complexity += 1

        # Add last function
        if current_function is not None:
            functions.append({"name": current_function, "complexity": current_complexity})

        return self._calculate_complexity_metrics(functions)

    def _analyze_js_complexity(self, code_content: str) -> dict[str, Any]:
        """Analyze JavaScript/TypeScript code complexity using simple heuristics."""
        lines = code_content.split("\n")
        functions = []
        current_function = None
        current_complexity = 1
        brace_level = 0

        for line in lines:
            stripped = line.strip()

            # Function definitions (simplified)
            if (
                "function " in stripped
                or "=>" in stripped
                or stripped.startswith("const ")
                and "=>" in stripped
            ):
                if current_function is not None:
                    functions.append({"name": current_function, "complexity": current_complexity})
                current_function = self._extract_js_function_name(stripped)
                current_complexity = 1
                brace_level = 0

            # Track braces for scope
            brace_level += stripped.count("{") - stripped.count("}")

            # Complexity adding statements
            if current_function is not None and brace_level > 0:
                if any(
                    keyword in stripped
                    for keyword in [
                        "if ",
                        "else if ",
                        "while ",
                        "for ",
                        "switch ",
                        "catch ",
                        "&&",
                        "||",
                    ]
                ):
                    current_complexity += 1

        # Add last function
        if current_function is not None:
            functions.append({"name": current_function, "complexity": current_complexity})

        return self._calculate_complexity_metrics(functions)

    def _extract_js_function_name(self, line: str) -> str:
        """Extract function name from JavaScript function declaration."""
        if "function " in line:
            return line.split("function ")[1].split("(")[0].strip()
        elif "const " in line and "=>" in line:
            return line.split("const ")[1].split("=")[0].strip()
        elif "=>" in line:
            return "anonymous"
        return "unknown"

    def _calculate_complexity_metrics(self, functions: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate aggregate complexity metrics from function data."""
        if not functions:
            return {
                "total_functions": 0,
                "avg_complexity": 0.0,
                "max_complexity": 0,
                "high_complexity_functions": 0,
                "complexity_distribution": {
                    "low": 0,
                    "medium": 0,
                    "high": 0,
                    "critical": 0,
                },
            }

        complexities = [f["complexity"] for f in functions]
        total_complexity = sum(complexities)

        distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        high_complexity_count = 0

        for complexity in complexities:
            if complexity <= 5:
                distribution["low"] += 1
            elif complexity <= 10:
                distribution["medium"] += 1
            elif complexity <= 15:
                distribution["high"] += 1
                high_complexity_count += 1
            else:
                distribution["critical"] += 1
                high_complexity_count += 1

        return {
            "total_functions": len(functions),
            "avg_complexity": round(total_complexity / len(functions), 2),
            "max_complexity": max(complexities),
            "high_complexity_functions": high_complexity_count,
            "complexity_distribution": distribution,
            "functions": functions,
        }

    def analyze_nesting_depth(self, code_content: str, language: str = "python") -> dict[str, Any]:
        """Analyze nesting depth in code."""
        if language == "python":
            return self._analyze_python_nesting(code_content)
        elif language in ["javascript", "typescript"]:
            return self._analyze_js_nesting(code_content)
        return {"max_depth": 0, "avg_depth": 0.0, "deep_nesting_lines": []}

    def _analyze_python_nesting(self, code_content: str) -> dict[str, Any]:
        """Analyze Python code nesting depth."""
        lines = code_content.split("\n")
        depths = []
        deep_nesting_lines = []

        for i, line in enumerate(lines, 1):
            if line.strip():  # Skip empty lines
                # Calculate indentation depth (assuming 4 spaces per level)
                leading_spaces = len(line) - len(line.lstrip())
                depth = leading_spaces // 4
                depths.append(depth)

                if depth > 4:  # Flag deep nesting
                    deep_nesting_lines.append(
                        {
                            "line": i,
                            "depth": depth,
                            "content": (
                                line.strip()[:50] + "..."
                                if len(line.strip()) > 50
                                else line.strip()
                            ),
                        }
                    )

        if not depths:
            return {"max_depth": 0, "avg_depth": 0.0, "deep_nesting_lines": []}

        return {
            "max_depth": max(depths),
            "avg_depth": round(sum(depths) / len(depths), 2),
            "deep_nesting_lines": deep_nesting_lines,
        }

    def _analyze_js_nesting(self, code_content: str) -> dict[str, Any]:
        """Analyze JavaScript/TypeScript code nesting depth."""
        lines = code_content.split("\n")
        current_depth = 0
        max_depth = 0
        depths = []
        deep_nesting_lines = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped:
                # Count opening and closing braces
                opening_braces = stripped.count("{")
                closing_braces = stripped.count("}")

                # Update depth
                current_depth += opening_braces - closing_braces
                current_depth = max(0, current_depth)  # Prevent negative depth

                depths.append(current_depth)
                max_depth = max(max_depth, current_depth)

                if current_depth > 4:  # Flag deep nesting
                    deep_nesting_lines.append(
                        {
                            "line": i,
                            "depth": current_depth,
                            "content": (stripped[:50] + "..." if len(stripped) > 50 else stripped),
                        }
                    )

        return {
            "max_depth": max_depth,
            "avg_depth": round(sum(depths) / len(depths), 2) if depths else 0.0,
            "deep_nesting_lines": deep_nesting_lines,
        }

    def calculate_maintainability_index(
        self, complexity_metrics: dict[str, Any], lines_of_code: int
    ) -> float:
        """Calculate maintainability index based on complexity and LOC."""
        if lines_of_code == 0:
            return 100.0

        avg_complexity = complexity_metrics.get("avg_complexity", 1)
        max_complexity = complexity_metrics.get("max_complexity", 1)

        # Simplified maintainability index calculation
        # Higher is better (0-100 scale)
        complexity_penalty = min(50, avg_complexity * 2)
        max_complexity_penalty = min(30, max_complexity)
        loc_penalty = min(20, lines_of_code / 100)

        maintainability = 100 - complexity_penalty - max_complexity_penalty - loc_penalty
        return max(0.0, round(maintainability, 1))

    def get_complexity_recommendations(self, complexity_metrics: dict[str, Any]) -> list[str]:
        """Generate recommendations based on complexity analysis."""
        recommendations = []

        high_complexity_count = complexity_metrics.get("high_complexity_functions", 0)
        max_complexity = complexity_metrics.get("max_complexity", 0)
        avg_complexity = complexity_metrics.get("avg_complexity", 0)

        if max_complexity > 20:
            recommendations.append(
                f"Refactor functions with complexity > 20 (found {high_complexity_count} functions)"
            )

        if avg_complexity > 8:
            recommendations.append(
                "Consider breaking down complex functions to improve maintainability"
            )

        if high_complexity_count > 5:
            recommendations.append(
                "High number of complex functions detected - consider architectural refactoring"
            )

        distribution = complexity_metrics.get("complexity_distribution", {})
        critical_count = distribution.get("critical", 0)
        if critical_count > 0:
            recommendations.append(
                f"Address {critical_count} critically complex functions immediately"
            )

        return recommendations
