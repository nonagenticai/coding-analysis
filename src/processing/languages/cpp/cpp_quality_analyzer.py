"""
C++ code quality analysis module.

This module handles code quality metrics, complexity analysis,
and best practices checking for C++ code.
"""

from __future__ import annotations

import re
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    CoreComplexityMetrics,
    EntityType,
)


class CppQualityAnalyzer:
    """Code quality analyzer for C++ code."""

    def __init__(self):
        """Initialize the quality analyzer."""
        self.quality_cache: dict[str, Any] = {}

    def calculate_complexity_metrics(
        self, content: str, entities: list[CoreCodeEntity]
    ) -> CoreComplexityMetrics:
        """Calculate complexity metrics for C++ code."""
        lines = content.splitlines()

        # Basic metrics
        total_lines = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = self._count_comment_lines(content)
        code_lines = total_lines - blank_lines - comment_lines

        # Cyclomatic complexity
        cyclomatic_complexity = self._calculate_cyclomatic_complexity(content)

        # Cognitive complexity
        cognitive_complexity = self._calculate_cognitive_complexity(content)

        # Halstead metrics
        halstead_metrics = self._calculate_halstead_metrics(content)

        # Maintainability index
        maintainability_index = self._calculate_maintainability_index(
            cyclomatic_complexity, code_lines, halstead_metrics
        )

        return CoreComplexityMetrics(
            cyclomatic_complexity=cyclomatic_complexity,
            cognitive_complexity=cognitive_complexity,
            lines_of_code=code_lines,
            maintainability_index=maintainability_index,
            halstead_volume=halstead_metrics.get("volume", 0),
            halstead_difficulty=halstead_metrics.get("difficulty", 0),
            halstead_effort=halstead_metrics.get("effort", 0),
        )

    def _count_comment_lines(self, content: str) -> int:
        """Count comment lines in C++ code."""
        comment_count = 0
        in_multiline_comment = False

        lines = content.splitlines()
        for line in lines:
            stripped = line.strip()

            # Handle multi-line comments
            if "/*" in stripped:
                in_multiline_comment = True
                if "*/" in stripped:
                    in_multiline_comment = False
                comment_count += 1
                continue

            if in_multiline_comment:
                if "*/" in stripped:
                    in_multiline_comment = False
                comment_count += 1
                continue

            # Single line comments
            if stripped.startswith("//"):
                comment_count += 1

        return comment_count

    def _calculate_cyclomatic_complexity(self, content: str) -> int:
        """Calculate cyclomatic complexity for C++ code."""
        complexity = 1  # Base complexity

        # Decision points that increase complexity
        decision_patterns = [
            r"\bif\s*\(",
            r"\belse\s+if\s*\(",
            r"\bwhile\s*\(",
            r"\bfor\s*\(",
            r"\bdo\s+",
            r"\bswitch\s*\(",
            r"\bcase\s+",
            r"\bcatch\s*\(",
            r"\b\?\s*",  # Ternary operator
            r"\b&&\b",
            r"\b\|\|\b",
        ]

        for pattern in decision_patterns:
            complexity += len(re.findall(pattern, content, re.IGNORECASE))

        return complexity

    def _calculate_cognitive_complexity(self, content: str) -> int:
        """Calculate cognitive complexity for C++ code."""
        complexity = 0

        # Patterns that increase cognitive complexity
        patterns = {
            r"\bif\s*\(": 1,
            r"\belse\s+if\s*\(": 1,
            r"\belse\b": 1,
            r"\bwhile\s*\(": 1,
            r"\bfor\s*\(": 1,
            r"\bdo\s+": 1,
            r"\bswitch\s*\(": 1,
            r"\bcatch\s*\(": 1,
            r"\b\?\s*": 1,  # Ternary operator
            r"\b&&\b": 1,
            r"\b\|\|\b": 1,
        }

        # Simple approximation - in reality, you'd need to parse the AST
        # to accurately track nesting levels
        for pattern, increment in patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            complexity += len(matches) * increment

        # Add nesting penalty (simplified)
        brace_depth = 0
        for char in content:
            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth > 0:
                    complexity += 1

        return complexity

    def _calculate_halstead_metrics(self, content: str) -> dict[str, float]:
        """Calculate Halstead complexity metrics."""
        # C++ operators
        operators = [
            "+",
            "-",
            "*",
            "/",
            "%",
            "=",
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
            "&&",
            "||",
            "!",
            "&",
            "|",
            "^",
            "~",
            "<<",
            ">>",
            "++",
            "--",
            "+=",
            "-=",
            "*=",
            "/=",
            "%=",
            "&=",
            "|=",
            "^=",
            "<<=",
            ">>=",
            "->",
            ".",
            "::",
            "?",
            ":",
            ",",
            ";",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "new",
            "delete",
            "sizeof",
            "typeof",
            "static_cast",
            "dynamic_cast",
            "const_cast",
            "reinterpret_cast",
        ]

        # Count operators and operands
        operator_counts = {}
        operand_counts = {}

        # Simple operator counting (would need more sophisticated parsing for accuracy)
        for op in operators:
            count = len(re.findall(re.escape(op), content))
            if count > 0:
                operator_counts[op] = count

        # Operand counting (simplified - identifiers and literals)
        operand_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b|\b\d+(?:\.\d+)?\b|"[^"]*"|\'[^\']*\''
        operands = re.findall(operand_pattern, content)
        for operand in operands:
            operand_counts[operand] = operand_counts.get(operand, 0) + 1

        # Calculate Halstead metrics
        n1 = len(operator_counts)  # Number of distinct operators
        n2 = len(operand_counts)  # Number of distinct operands
        N1 = sum(operator_counts.values())  # Total number of operators  # noqa: N806
        N2 = sum(operand_counts.values())  # Total number of operands  # noqa: N806

        vocabulary = n1 + n2
        length = N1 + N2

        if vocabulary > 0 and length > 0:
            volume = length * (vocabulary.bit_length() if vocabulary > 0 else 0)
            difficulty = (n1 / 2) * (N2 / n2) if n1 > 0 and n2 > 0 else 0
            effort = difficulty * volume
        else:
            volume = difficulty = effort = 0

        return {
            "vocabulary": vocabulary,
            "length": length,
            "volume": volume,
            "difficulty": difficulty,
            "effort": effort,
        }

    def _calculate_maintainability_index(
        self,
        cyclomatic_complexity: int,
        lines_of_code: int,
        halstead_metrics: dict[str, float],
    ) -> float:
        """Calculate maintainability index."""
        if lines_of_code == 0:
            return 0

        halstead_volume = halstead_metrics.get("volume", 0)

        # Simplified maintainability index calculation
        mi = (
            171
            - 5.2 * (halstead_volume**0.23)
            - 0.23 * cyclomatic_complexity
            - 16.2 * (lines_of_code**0.5)
        )

        # Normalize to 0-100 range
        return max(0, min(100, mi))

    def analyze_code_smells(
        self, content: str, entities: list[CoreCodeEntity]
    ) -> dict[str, list[dict[str, Any]]]:
        """Analyze code smells in C++ code."""
        smells = {
            "long_methods": [],
            "large_classes": [],
            "too_many_parameters": [],
            "deep_nesting": [],
            "duplicate_code": [],
            "dead_code": [],
            "magic_numbers": [],
            "long_parameter_lists": [],
        }

        # Analyze long methods
        functions = [e for e in entities if e.type == EntityType.FUNCTION]
        for func in functions:
            if func.signature:
                func_lines = self._estimate_function_length(content, func)
                if func_lines > 50:  # Threshold for long method
                    smells["long_methods"].append(
                        {
                            "entity": func.name,
                            "lines": func_lines,
                            "location": f"{func.file_path}:{func.line_number}",
                        }
                    )

        # Analyze large classes
        classes = [e for e in entities if e.type == EntityType.CLASS]
        for cls in classes:
            class_methods = [
                f
                for f in functions
                if f.line_number and cls.line_number and f.line_number > cls.line_number
            ]
            if len(class_methods) > 20:  # Threshold for large class
                smells["large_classes"].append(
                    {
                        "entity": cls.name,
                        "methods": len(class_methods),
                        "location": f"{cls.file_path}:{cls.line_number}",
                    }
                )

        # Analyze function parameters
        for func in functions:
            if func.signature:
                param_count = self._count_parameters(func.signature)
                if param_count > 5:  # Threshold for too many parameters
                    smells["too_many_parameters"].append(
                        {
                            "entity": func.name,
                            "parameters": param_count,
                            "location": f"{func.file_path}:{func.line_number}",
                        }
                    )

        # Analyze deep nesting
        deep_nesting_locations = self._find_deep_nesting(content)
        smells["deep_nesting"] = deep_nesting_locations

        # Analyze magic numbers
        magic_numbers = self._find_magic_numbers(content)
        smells["magic_numbers"] = magic_numbers

        return smells

    def _estimate_function_length(self, content: str, function: CoreCodeEntity) -> int:
        """Estimate the length of a function in lines."""
        if not function.line_number:
            return 0

        lines = content.splitlines()
        start_line = function.line_number - 1

        if start_line >= len(lines):
            return 0

        # Find the opening brace
        brace_count = 0
        func_lines = 0
        in_function = False

        for i in range(start_line, len(lines)):
            line = lines[i]
            func_lines += 1

            if "{" in line:
                brace_count += line.count("{")
                in_function = True

            if "}" in line:
                brace_count -= line.count("}")
                if in_function and brace_count == 0:
                    break

        return func_lines

    def _count_parameters(self, signature: str) -> int:
        """Count the number of parameters in a function signature."""
        # Extract parameter list from signature
        match = re.search(r"\(([^)]*)\)", signature)
        if not match:
            return 0

        param_str = match.group(1).strip()
        if not param_str:
            return 0

        # Simple parameter counting (split by comma, accounting for templates)
        params = []
        depth = 0
        current_param = ""

        for char in param_str:
            if char in "<(":
                depth += 1
            elif char in ">)":
                depth -= 1
            elif char == "," and depth == 0:
                if current_param.strip():
                    params.append(current_param.strip())
                current_param = ""
                continue

            current_param += char

        if current_param.strip():
            params.append(current_param.strip())

        return len(params)

    def _find_deep_nesting(self, content: str, threshold: int = 4) -> list[dict[str, Any]]:
        """Find locations with deep nesting."""
        deep_nesting = []
        lines = content.splitlines()

        for i, line in enumerate(lines):
            # Count indentation level (simplified)
            indent_level = len(line) - len(line.lstrip())
            spaces_per_level = 4  # Assume 4 spaces per indentation level
            nesting_level = indent_level // spaces_per_level

            if nesting_level > threshold:
                deep_nesting.append(
                    {
                        "line": i + 1,
                        "nesting_level": nesting_level,
                        "code": line.strip(),
                    }
                )

        return deep_nesting

    def _find_magic_numbers(self, content: str) -> list[dict[str, Any]]:
        """Find magic numbers in the code."""
        magic_numbers = []
        lines = content.splitlines()

        # Common magic numbers to ignore
        ignore_numbers = {"0", "1", "2", "10", "100", "1000"}

        number_pattern = r"\b(\d+(?:\.\d+)?)\b"

        for i, line in enumerate(lines):
            # Skip comments and string literals
            if "//" in line or "/*" in line or '"' in line or "'" in line:
                continue

            for match in re.finditer(number_pattern, line):
                number = match.group(1)
                if number not in ignore_numbers:
                    magic_numbers.append(
                        {
                            "line": i + 1,
                            "number": number,
                            "context": line.strip(),
                        }
                    )

        return magic_numbers

    def analyze_design_patterns(
        self, entities: list[CoreCodeEntity]
    ) -> dict[str, list[dict[str, Any]]]:
        """Analyze common design patterns in C++ code."""
        patterns = {
            "singleton": [],
            "factory": [],
            "observer": [],
            "strategy": [],
            "decorator": [],
            "adapter": [],
        }

        classes = [e for e in entities if e.type == EntityType.CLASS]

        # Detect Singleton pattern (simplified)
        for cls in classes:
            if cls.name.lower().endswith("singleton") or "singleton" in cls.name.lower():
                patterns["singleton"].append(
                    {
                        "class": cls.name,
                        "location": f"{cls.file_path}:{cls.line_number}",
                        "confidence": "medium",
                    }
                )

        # Detect Factory pattern (simplified)
        for cls in classes:
            if any(word in cls.name.lower() for word in ["factory", "creator", "builder"]):
                patterns["factory"].append(
                    {
                        "class": cls.name,
                        "location": f"{cls.file_path}:{cls.line_number}",
                        "confidence": "low",
                    }
                )

        # More sophisticated pattern detection would require deeper analysis
        # of class relationships and method signatures

        return patterns

    def calculate_maintainability_score(
        self, content: str, entities: list[CoreCodeEntity]
    ) -> dict[str, Any]:
        """Calculate an overall maintainability score."""
        complexity_metrics = self.calculate_complexity_metrics(content, entities)
        code_smells = self.analyze_code_smells(content, entities)

        # Calculate penalty scores for different aspects
        complexity_penalty = min(50, complexity_metrics.cyclomatic_complexity * 2)
        smell_penalty = sum(len(smells) for smells in code_smells.values()) * 2

        # Calculate base score
        base_score = 100
        final_score = max(0, base_score - complexity_penalty - smell_penalty)

        return {
            "score": final_score,
            "grade": self._score_to_grade(final_score),
            "complexity_metrics": {
                "cyclomatic_complexity": complexity_metrics.cyclomatic_complexity,
                "cognitive_complexity": complexity_metrics.cognitive_complexity,
                "maintainability_index": complexity_metrics.maintainability_index,
            },
            "code_smells_summary": {
                smell_type: len(smells) for smell_type, smells in code_smells.items()
            },
            "recommendations": self._generate_recommendations(complexity_metrics, code_smells),
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _generate_recommendations(
        self,
        complexity_metrics: CoreComplexityMetrics,
        code_smells: dict[str, list[dict[str, Any]]],
    ) -> list[str]:
        """Generate maintainability recommendations."""
        recommendations = []

        if complexity_metrics.cyclomatic_complexity > 15:
            recommendations.append(
                "Consider breaking down complex functions to reduce cyclomatic complexity"
            )

        if complexity_metrics.cognitive_complexity > 20:
            recommendations.append("Simplify control flow to reduce cognitive complexity")

        if len(code_smells["long_methods"]) > 0:
            recommendations.append(f"Refactor {len(code_smells['long_methods'])} long methods")

        if len(code_smells["large_classes"]) > 0:
            recommendations.append(
                f"Consider splitting {len(code_smells['large_classes'])} large classes"
            )

        if len(code_smells["too_many_parameters"]) > 0:
            recommendations.append("Reduce parameter counts in functions")

        if len(code_smells["magic_numbers"]) > 5:
            recommendations.append("Replace magic numbers with named constants")

        if not recommendations:
            recommendations.append("Code quality looks good! Keep up the good work.")

        return recommendations
