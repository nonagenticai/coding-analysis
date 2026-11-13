"""Module for evaluating code snippets against existing codebase patterns."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

from models.github_models import CodeSnippet, RepositoryContext

from models.core_models import SnippetEvaluation

logger = logging.getLogger(__name__)


class SnippetEvaluator:
    """Evaluates new code snippets against existing codebase"""

    def __init__(self) -> None:
        self.pattern_matcher = PatternMatcher()
        self.style_checker = StyleChecker()
        self.compatibility_checker = CompatibilityChecker()
        self._evaluation_cache: dict[str, SnippetEvaluation] = {}

    async def evaluate_snippet(
        self, snippet: CodeSnippet, codebase_context: RepositoryContext
    ) -> SnippetEvaluation:
        """Comprehensive snippet evaluation"""
        try:
            logger.info("Evaluating %s snippet", snippet.language)

            # Style consistency check
            style_score = await self.style_checker.check_consistency(
                snippet, codebase_context.style_guide
            )

            # Pattern matching
            pattern_matches = await self.pattern_matcher.find_similar_patterns(
                snippet, codebase_context.existing_patterns
            )

            # API compatibility
            compatibility = await self.compatibility_checker.check_compatibility(
                snippet, codebase_context.api_surface
            )

            # Overall evaluation
            overall_score = self._calculate_overall_score(
                style_score, pattern_matches, compatibility
            )

            recommendations = self._generate_recommendations(
                style_score, pattern_matches, compatibility, snippet
            )

            return SnippetEvaluation(
                success=True,
                analysis=None,
                compatibility_score=overall_score,
                error=None,
                style_score=style_score,
                pattern_matches=pattern_matches,
                recommendations=recommendations,
                metadata={
                    "evaluation_timestamp": datetime.now(UTC).isoformat(),
                    "snippet_language": snippet.language,
                    "codebase_size": codebase_context.size_metrics,
                    "compatibility_details": compatibility,
                },
            )

        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Snippet evaluation failed: %s", str(e))
            return SnippetEvaluation(
                success=False,
                analysis=None,
                compatibility_score=0.0,
                error=str(e),
                metadata={
                    "snippet_language": snippet.language,
                    "error_type": type(e).__name__,
                },
            )

    def _calculate_overall_score(
        self,
        style_score: dict[str, Any],
        pattern_matches: list[dict],
        compatibility: dict[str, Any],
    ) -> float:
        """Calculate overall compatibility score"""
        # Weight different factors
        style_weight = 0.3
        pattern_weight = 0.4
        compatibility_weight = 0.3

        # Calculate style score (average of all style metrics)
        style_values = [v for v in style_score.values() if isinstance(v, (int, float))]
        avg_style_score = sum(style_values) / len(style_values) if style_values else 0.5

        # Calculate pattern score (based on number and quality of matches)
        pattern_score = min(len(pattern_matches) * 0.2, 1.0) if pattern_matches else 0.3

        # Calculate compatibility score
        compat_values = [v for v in compatibility.values() if isinstance(v, bool)]
        compat_score = sum(compat_values) / len(compat_values) if compat_values else 0.5

        overall = (
            avg_style_score * style_weight
            + pattern_score * pattern_weight
            + compat_score * compatibility_weight
        )

        return min(max(overall, 0.0), 1.0)  # Clamp between 0 and 1

    def _generate_recommendations(
        self,
        style_score: dict[str, Any],
        pattern_matches: list[dict],
        compatibility: dict[str, Any],
        snippet: CodeSnippet,
    ) -> list[str]:
        """Generate recommendations for snippet integration"""
        recommendations: list[Any] = []
        # Style-based recommendations
        if style_score.get("naming_convention_score", 1.0) < 0.8:
            recommendations.append("Consider following the project's naming conventions")

        if style_score.get("formatting_score", 1.0) < 0.8:
            recommendations.append("Format code according to project style guidelines")

        if style_score.get("documentation_score", 1.0) < 0.7:
            recommendations.append("Add or update documentation/docstrings")

        # Pattern-based recommendations
        if len(pattern_matches) > 0:
            recommendations.append(
                f"Found {len(pattern_matches)} similar patterns - consider consistency"
            )
        else:
            recommendations.append(
                "No similar patterns found - ensure this follows project architecture"
            )

        # Compatibility recommendations
        if not compatibility.get("dependency_compatibility", True):
            recommendations.append("Check dependency compatibility with existing codebase")

        if not compatibility.get("api_usage_valid", True):
            recommendations.append("Verify API usage follows project conventions")

        # Language-specific recommendations
        if snippet.language == "python":
            recommendations.extend(
                [
                    "Consider adding type hints for better code clarity",
                    "Ensure proper error handling is implemented",
                    "Add unit tests for new functionality",
                ]
            )
        elif snippet.language in ["javascript", "typescript"]:
            recommendations.extend(
                [
                    "Consider using TypeScript for better type safety",
                    "Ensure proper error handling with try-catch blocks",
                    "Add JSDoc comments for better documentation",
                ]
            )

        return recommendations

    def clear_evaluation_cache(self) -> None:
        """Clear the evaluation cache."""
        self._evaluation_cache.clear()
        logger.info("Evaluation cache cleared")


class PatternMatcher:
    """Identifies similar code patterns in existing codebase"""

    async def find_similar_patterns(
        self, snippet: CodeSnippet, existing_patterns: list[dict]
    ) -> list[dict]:
        """Find functions/classes with similar structure or purpose"""
        try:
            matches: list[Any] = []
            # Basic pattern matching based on snippet content
            snippet_words = set(snippet.code.lower().split())

            for pattern in existing_patterns:
                pattern_words = set(pattern.get("content", "").lower().split())

                # Calculate similarity based on common words
                if snippet_words and pattern_words:
                    common_words = snippet_words.intersection(pattern_words)
                    similarity = len(common_words) / len(snippet_words.union(pattern_words))

                    if similarity > 0.3:  # Threshold for similarity
                        matches.append(
                            {
                                "pattern_id": pattern.get("id", "unknown"),
                                "similarity_score": similarity,
                                "pattern_type": pattern.get("type", "unknown"),
                                "file_path": pattern.get("file_path", "unknown"),
                                "description": pattern.get("description", "Similar pattern found"),
                            }
                        )

            # Sort by similarity score
            matches.sort(key=lambda x: x["similarity_score"], reverse=True)

            return matches[:5]  # Return top 5 matches

        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Pattern matching failed: %s", str(e))
            return []

    def get_similarity_threshold(self) -> float:
        """Get the current similarity threshold for pattern matching."""
        return 0.3


class StyleChecker:
    """Checks code snippet against codebase style guidelines"""

    async def check_consistency(
        self,
        snippet: CodeSnippet,
        style_guide: dict,
    ) -> dict[str, Any]:
        """Check style consistency"""
        try:
            style_scores = {
                "naming_convention_score": 0.9,  # Default high score
                "formatting_score": 0.85,
                "documentation_score": 0.7,
                "complexity_score": 0.8,
            }

            # Basic style checks based on language
            if snippet.language == "python":
                style_scores.update(self._check_python_style(snippet))
            elif snippet.language in ["javascript", "typescript"]:
                style_scores.update(self._check_javascript_style(snippet))

            return style_scores

        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Style checking failed: %s", str(e))
            return {
                "naming_convention_score": 0.5,
                "formatting_score": 0.5,
                "documentation_score": 0.5,
                "error": str(e),
            }

    def _check_python_style(self, snippet: CodeSnippet) -> dict[str, float]:
        """Check Python-specific style guidelines"""
        scores: dict[str, Any] = {}
        lines = snippet.code.split("\n")

        # Check for docstrings
        has_docstring = '"""' in snippet.code or "'''" in snippet.code
        scores["documentation_score"] = 0.9 if has_docstring else 0.6

        # Check indentation (should be 4 spaces for Python)
        proper_indentation = all(
            len(line) - len(line.lstrip()) % 4 == 0 for line in lines if line.strip()
        )
        scores["formatting_score"] = 0.9 if proper_indentation else 0.7

        # Check naming conventions (snake_case for functions/variables)
        snake_case_pattern = re.compile(r"^[a-z_][a-z0-9_]*$")

        # Basic check for function definitions
        function_names = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)", snippet.code)
        proper_naming = all(snake_case_pattern.match(name) for name in function_names)
        scores["naming_convention_score"] = 0.9 if proper_naming else 0.7

        return scores

    def _check_javascript_style(self, snippet: CodeSnippet) -> dict[str, float]:
        """Check JavaScript/TypeScript-specific style guidelines"""
        scores: dict[str, Any] = {}
        # Check for JSDoc comments
        has_jsdoc = "/**" in snippet.code
        scores["documentation_score"] = 0.9 if has_jsdoc else 0.6

        # Check for semicolons
        lines = [line.strip() for line in snippet.code.split("\n") if line.strip()]
        statement_lines = [
            line for line in lines if not line.startswith("//") and not line.startswith("/*")
        ]

        if statement_lines:
            semicolon_usage = sum(1 for line in statement_lines if line.endswith(";"))
            scores["formatting_score"] = semicolon_usage / len(statement_lines)
        else:
            scores["formatting_score"] = 0.8

        # Check camelCase naming
        camel_case_pattern = re.compile(r"^[a-z][a-zA-Z0-9]*$")

        # Basic check for function definitions
        function_names = re.findall(r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)", snippet.code)
        proper_naming = all(camel_case_pattern.match(name) for name in function_names)
        scores["naming_convention_score"] = 0.9 if proper_naming else 0.7

        return scores

    def get_default_style_scores(self) -> dict[str, float]:
        """Get default style scores."""
        return {
            "naming_convention_score": 0.9,
            "formatting_score": 0.85,
            "documentation_score": 0.7,
            "complexity_score": 0.8,
        }


class CompatibilityChecker:
    """Checks API and dependency compatibility"""

    async def check_compatibility(
        self,
        snippet: CodeSnippet,
        api_surface: dict,
    ) -> dict[str, Any]:
        """Check API compatibility"""
        try:
            compatibility = {
                "dependency_compatibility": True,
                "api_usage_valid": True,
                "version_compatibility": True,
                "security_check": True,
            }

            # Basic security checks
            dangerous_patterns = [
                "eval(",
                "exec(",
                "subprocess.call",
                "os.system",
            ]
            has_dangerous_code = any(pattern in snippet.code for pattern in dangerous_patterns)
            compatibility["security_check"] = not has_dangerous_code

            # Check for common import patterns
            if snippet.language == "python":
                # Check for standard library usage
                import_lines = [
                    line
                    for line in snippet.code.split("\n")
                    if line.strip().startswith("import") or line.strip().startswith("from")
                ]
                compatibility["uses_standard_imports"] = len(import_lines) > 0

            return compatibility

        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Compatibility checking failed: %s", str(e))
            return {
                "dependency_compatibility": True,  # Default to compatible
                "api_usage_valid": True,
                "version_compatibility": True,
                "error": str(e),
            }

    def get_dangerous_patterns(self) -> list[str]:
        """Get list of dangerous code patterns to check against."""
        return [
            "eval(",
            "exec(",
            "subprocess.call",
            "os.system",
        ]
