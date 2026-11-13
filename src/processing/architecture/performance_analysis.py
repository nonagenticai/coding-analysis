"""Module for performance analysis and optimization recommendations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from models.architecture_models import ArchitectureQualityIssue

logger = logging.getLogger(__name__)


class PerformanceAnalysis:
    """Handler for performance analysis and optimization recommendations."""

    def __init__(self) -> None:
        """Initialize the performance analyzer."""
        self.performance_patterns = {
            "python": {
                "anti_patterns": [
                    r"for.*in.*range.*len",  # Use enumerate instead
                    r"\.append.*in.*for",  # List comprehension might be better
                    r"global ",  # Global variables
                    r"eval\(",  # Eval usage
                    r"exec\(",  # Exec usage
                ],
                "good_patterns": [
                    r"with open\(",  # Context managers
                    r"enumerate\(",  # Proper iteration
                    r"list\(.*\)",  # List comprehensions
                    r"cache",  # Caching
                ],
            },
            "javascript": {
                "anti_patterns": [
                    r"for.*in.*length",  # Use for...of or forEach
                    r"document\.getElementById",  # Use modern selectors
                    r"new Date\(\).*getTime",  # Performance.now() is better
                    r"setTimeout.*0",  # Unnecessary timeout
                ],
                "good_patterns": [
                    r"requestAnimationFrame",
                    r"addEventListener",
                    r"querySelector",
                    r"const.*=.*await",
                ],
            },
        }

    def analyze_performance_patterns(
        self, repo_path: Path, language: str
    ) -> list[ArchitectureQualityIssue]:
        """Analyze code for performance anti-patterns."""
        issues: list[ArchitectureQualityIssue] = []

        if language not in self.performance_patterns:
            return issues

        # Get file extensions for the language
        extensions = self._get_file_extensions(language)

        for ext in extensions:
            for file_path in repo_path.rglob(f"*{ext}"):
                try:
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        file_issues = self._analyze_file_performance(
                            content, str(file_path), language
                        )
                        issues.extend(file_issues)
                except (OSError, UnicodeDecodeError):  # fmt: skip
                    continue

        return issues

    def _analyze_file_performance(
        self, content: str, file_path: str, language: str
    ) -> list[ArchitectureQualityIssue]:
        """Analyze a single file for performance issues."""
        issues: list[ArchitectureQualityIssue] = []
        lines = content.split("\n")

        anti_patterns = self.performance_patterns[language]["anti_patterns"]

        for line_num, line in enumerate(lines, 1):
            for pattern in anti_patterns:
                if self._matches_pattern(line, pattern):
                    issue = ArchitectureQualityIssue(
                        id=f"perf-{len(issues)}-{line_num}",
                        severity="medium",
                        message=f"Performance anti-pattern detected: {pattern}",
                        rule_id=f"perf-{pattern[:20]}",
                        rule_name="Performance Anti-pattern",
                        file_path=file_path,
                        line_number=line_num,
                        column_number=1,
                        tool="performance_analyzer",
                        category="performance",
                        description=f"Line contains performance anti-pattern: {line.strip()}",
                        resolve_suggestions=self._get_performance_suggestion(pattern, language),
                    )
                    issues.append(issue)

        return issues

    def _matches_pattern(self, line: str, pattern: str) -> bool:
        """Check if a line matches a performance pattern."""
        import re

        try:
            return bool(re.search(pattern, line, re.IGNORECASE))
        except re.error:
            # Fallback to simple string matching
            return pattern.lower() in line.lower()

    def _get_performance_suggestion(self, pattern: str, language: str) -> str:
        """Get performance improvement suggestion for a pattern."""
        suggestions = {
            "python": {
                r"for.*in.*range.*len": "Use enumerate() instead of range(len())",
                r"\.append.*in.*for": "Consider using list comprehension",
                r"global ": "Avoid global variables, use function parameters",
                r"eval\(": "Avoid eval(), use safer alternatives",
                r"exec\(": "Avoid exec(), use safer alternatives",
            },
            "javascript": {
                r"for.*in.*length": "Use for...of loop or forEach method",
                r"document\.getElementById": "Use querySelector for modern DOM access",
                r"new Date\(\).*getTime": "Use performance.now() for timing",
                r"setTimeout.*0": "Use requestAnimationFrame or remove unnecessary timeout",
            },
        }

        return suggestions.get(language, {}).get(pattern, "Review for performance optimization")

    def _get_file_extensions(self, language: str) -> list[str]:
        """Get file extensions for a language."""
        extensions_map = {
            "python": [".py"],
            "javascript": [".js", ".jsx"],
            "typescript": [".ts", ".tsx"],
            "java": [".java"],
            "go": [".go"],
            "rust": [".rs"],
            "cpp": [".cpp", ".cc", ".cxx"],
            "csharp": [".cs"],
        }
        return extensions_map.get(language, [])

    def analyze_memory_usage_patterns(self, repo_path: Path, language: str) -> dict[str, Any]:
        """Analyze potential memory usage issues."""
        potential_memory_leaks: int = 0
        large_data_structures: int = 0
        inefficient_loops: int = 0
        recommendations: list[str] = []

        extensions = self._get_file_extensions(language)

        for ext in extensions:
            for file_path in repo_path.rglob(f"*{ext}"):
                try:
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        file_analysis = self._analyze_file_memory_patterns(content, language)

                        potential_memory_leaks += file_analysis["memory_leaks"]
                        large_data_structures += file_analysis["large_structures"]
                        inefficient_loops += file_analysis["inefficient_loops"]

                except (OSError, UnicodeDecodeError):  # fmt: skip
                    continue

        # Generate recommendations
        if potential_memory_leaks > 0:
            recommendations.append(f"Found {potential_memory_leaks} potential memory leak patterns")

        if large_data_structures > 0:
            recommendations.append(f"Found {large_data_structures} large data structure patterns")

        if inefficient_loops > 0:
            recommendations.append(f"Found {inefficient_loops} inefficient loop patterns")

        return {
            "potential_memory_leaks": potential_memory_leaks,
            "large_data_structures": large_data_structures,
            "inefficient_loops": inefficient_loops,
            "recommendations": recommendations,
        }

    def _analyze_file_memory_patterns(self, content: str, language: str) -> dict[str, int]:
        """Analyze memory patterns in a single file."""
        patterns = {"memory_leaks": 0, "large_structures": 0, "inefficient_loops": 0}

        lines = content.split("\n")

        if language == "python":
            for line in lines:
                line_lower = line.lower().strip()

                # Check for potential memory leaks
                if "global " in line_lower or "globals()" in line_lower:
                    patterns["memory_leaks"] += 1

                # Check for large data structures
                if any(
                    pattern in line_lower
                    for pattern in ["range(", "list(range(", "* 1000", "* 10000"]
                ):
                    patterns["large_structures"] += 1

                # Check for inefficient loops
                if "for " in line_lower and "range(len(" in line_lower:
                    patterns["inefficient_loops"] += 1

        elif language in ["javascript", "typescript"]:
            for line in lines:
                line_lower = line.lower().strip()

                # Check for potential memory leaks
                if any(
                    pattern in line_lower
                    for pattern in ["setinterval", "settimeout", "addeventlistener"]
                ):
                    if "clear" not in line_lower and "remove" not in line_lower:
                        patterns["memory_leaks"] += 1

                # Check for large arrays/objects
                if "new Array(" in line or "Array(" in line:
                    patterns["large_structures"] += 1

                # Check for inefficient loops
                if "for (" in line_lower and ".length" in line_lower:
                    patterns["inefficient_loops"] += 1

        return patterns

    def run_profiling_analysis(self, repo_path: Path, language: str) -> dict[str, Any]:
        """Run basic profiling analysis for the codebase."""
        total_files: int = 0
        total_lines: int = 0
        avg_file_size: float = 0.0
        large_files: list[dict[str, Any]] = []
        complexity_hotspots: list[dict[str, Any]] = []
        recommendations: list[str] = []

        extensions = self._get_file_extensions(language)
        file_sizes: list[int] = []

        for ext in extensions:
            for file_path in repo_path.rglob(f"*{ext}"):
                try:
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                        line_count = len(lines)

                        total_files += 1
                        total_lines += line_count
                        file_sizes.append(line_count)

                        # Flag large files
                        if line_count > 500:
                            large_files.append({"path": str(file_path), "lines": line_count})

                        # Basic complexity analysis
                        if language == "python":
                            complexity_score = self._calculate_basic_complexity(lines, "python")
                        elif language in ["javascript", "typescript"]:
                            complexity_score = self._calculate_basic_complexity(lines, "javascript")
                        else:
                            complexity_score = 0

                        if complexity_score > 50:
                            complexity_hotspots.append(
                                {
                                    "path": str(file_path),
                                    "complexity_score": complexity_score,
                                }
                            )

                except (OSError, UnicodeDecodeError):  # fmt: skip
                    continue

        # Calculate averages
        if file_sizes:
            avg_file_size = sum(file_sizes) / len(file_sizes)

        # Generate recommendations
        if len(large_files) > 0:
            recommendations.append(
                f"Consider splitting {len(large_files)} large files (>500 lines)"
            )

        if len(complexity_hotspots) > 0:
            recommendations.append(f"Review {len(complexity_hotspots)} complexity hotspots")

        if avg_file_size > 200:
            recommendations.append("Average file size is large - consider modular design")

        return {
            "total_files": total_files,
            "total_lines": total_lines,
            "avg_file_size": avg_file_size,
            "large_files": large_files,
            "complexity_hotspots": complexity_hotspots,
            "recommendations": recommendations,
        }

    def _calculate_basic_complexity(self, lines: list[str], language: str) -> int:
        """Calculate basic complexity score for a file."""
        complexity = 0

        for line in lines:
            line_stripped = line.strip().lower()

            if language == "python":
                # Count control structures
                if any(
                    keyword in line_stripped
                    for keyword in [
                        "if ",
                        "elif ",
                        "else:",
                        "for ",
                        "while ",
                        "except ",
                        "try:",
                    ]
                ):
                    complexity += 1
                # Count function definitions
                if line_stripped.startswith("def ") or line_stripped.startswith("async def "):
                    complexity += 2
                # Count class definitions
                if line_stripped.startswith("class "):
                    complexity += 3

            elif language == "javascript":
                # Count control structures
                if any(
                    keyword in line_stripped
                    for keyword in [
                        "if (",
                        "else",
                        "for (",
                        "while (",
                        "switch (",
                        "catch (",
                    ]
                ):
                    complexity += 1
                # Count function definitions
                if "function " in line_stripped or "=>" in line_stripped:
                    complexity += 2
                # Count class definitions
                if line_stripped.startswith("class "):
                    complexity += 3

        return complexity

    def get_performance_recommendations(self, analysis_results: dict[str, Any]) -> list[str]:
        """Generate comprehensive performance recommendations."""
        recommendations = []

        # Memory usage recommendations
        memory_analysis = analysis_results.get("memory_analysis", {})
        if memory_analysis.get("potential_memory_leaks", 0) > 0:
            recommendations.append(
                "Review code for potential memory leaks and ensure proper cleanup"
            )

        # File size recommendations
        profiling = analysis_results.get("profiling_analysis", {})
        large_files = profiling.get("large_files", [])
        if len(large_files) > 3:
            recommendations.append(
                "Consider breaking down large files into smaller, focused modules"
            )

        # Complexity recommendations
        hotspots = profiling.get("complexity_hotspots", [])
        if len(hotspots) > 0:
            recommendations.append(
                "Refactor complexity hotspots to improve maintainability and performance"
            )

        # General performance recommendations
        recommendations.extend(
            [
                "Use appropriate data structures for your use case",
                "Consider caching for expensive operations",
                "Profile critical paths in production environments",
                "Implement lazy loading where appropriate",
                "Monitor memory usage in production",
            ]
        )

        return recommendations
