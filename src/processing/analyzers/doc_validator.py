"""
Documentation Validation and Quality Module
Handles documentation coverage calculation, quality metrics, and validation.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class DocumentationValidator:
    """Handles documentation validation, quality assessment, and coverage calculation."""

    def __init__(self):
        """Initialize the documentation validator."""
        self.quality_thresholds = {
            "excellent": 0.9,
            "good": 0.7,
            "fair": 0.5,
            "poor": 0.3,
        }

    def calculate_overall_documentation_coverage(
        self, documentation_data: dict[str, Any], api_docs: dict[str, Any]
    ) -> float:
        """Calculate overall documentation coverage."""
        try:
            total_items = 0
            documented_items = 0

            # Count documentation files
            total_items += len(documentation_data)
            documented_items += len(
                [doc for doc in documentation_data.values() if self._has_content(doc)]
            )

            # Count API documentation
            for api_doc in api_docs.values():
                functions = api_doc.get("functions", [])
                classes = api_doc.get("classes", [])

                # Count functions
                total_items += len(functions)
                documented_items += len([func for func in functions if func.get("docstring")])

                # Count classes and their methods
                for cls in classes:
                    total_items += 1  # Class itself
                    documented_items += 1 if cls.get("docstring") else 0

                    methods = cls.get("methods", [])
                    total_items += len(methods)
                    documented_items += len(
                        [method for method in methods if method.get("docstring")]
                    )

            return documented_items / total_items if total_items > 0 else 0.0

        except (ValueError, TypeError, ZeroDivisionError):  # fmt: skip
            return 0.5  # Default coverage

    def assess_documentation_quality(
        self, documentation_sections: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess the quality of documentation sections."""
        quality_report = {
            "overall_score": 0.0,
            "section_scores": {},
            "recommendations": [],
            "quality_level": "poor",
        }

        try:
            section_scores = []

            for section_name, section_data in documentation_sections.items():
                score = self._assess_section_quality(section_data)
                quality_report["section_scores"][section_name] = score
                section_scores.append(score)

            # Calculate overall score
            if section_scores:
                quality_report["overall_score"] = sum(section_scores) / len(section_scores)

            # Determine quality level
            quality_report["quality_level"] = self._get_quality_level(
                quality_report["overall_score"]
            )

            # Generate recommendations
            quality_report["recommendations"] = self._generate_recommendations(
                quality_report["section_scores"]
            )

        except (ValueError, TypeError) as e:
            logger.warning("Error assessing documentation quality: %s", e)

        return quality_report

    def validate_documentation_completeness(
        self, repository_path: str, documentation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate documentation completeness against project structure."""
        validation_report = {
            "missing_docs": [],
            "incomplete_sections": [],
            "coverage_gaps": [],
            "completeness_score": 0.0,
        }

        try:
            # Check for essential documentation files
            essential_docs = ["README", "CHANGELOG", "CONTRIBUTING", "LICENSE"]
            found_docs = set()

            for doc_path in documentation_data.get("documentation_files", {}):
                doc_name = doc_path.upper().split(".")[0]
                if doc_name in essential_docs:
                    found_docs.add(doc_name)

            missing_essential = set(essential_docs) - found_docs
            validation_report["missing_docs"] = list(missing_essential)

            # Check API documentation completeness
            api_docs = documentation_data.get("api_documentation", {})
            incomplete_api = []

            for file_path, api_data in api_docs.items():
                functions = api_data.get("functions", [])
                undocumented_functions = [
                    func["name"] for func in functions if not func.get("docstring")
                ]
                if undocumented_functions:
                    incomplete_api.append(
                        {
                            "file": file_path,
                            "undocumented_functions": undocumented_functions,
                        }
                    )

            validation_report["incomplete_sections"] = incomplete_api

            # Calculate completeness score
            total_checks = len(essential_docs) + len(api_docs)
            passed_checks = len(found_docs) + len(api_docs) - len(incomplete_api)
            validation_report["completeness_score"] = (
                passed_checks / total_checks if total_checks > 0 else 0.0
            )

        except (ValueError, TypeError) as e:
            logger.warning("Error validating documentation completeness: %s", e)

        return validation_report

    def analyze_documentation_consistency(
        self, documentation_sections: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze consistency across documentation sections."""
        consistency_report = {
            "style_consistency": 0.0,
            "terminology_consistency": 0.0,
            "format_consistency": 0.0,
            "inconsistencies": [],
            "overall_consistency": 0.0,
        }

        try:
            # Analyze style consistency
            style_scores = []
            format_scores = []
            all_content = []

            for section_name, section_data in documentation_sections.items():
                content = self._extract_content(section_data)
                if content:
                    all_content.append((section_name, content))

                    # Check style consistency
                    style_score = self._check_style_consistency(content)
                    style_scores.append(style_score)

                    # Check format consistency
                    format_score = self._check_format_consistency(content)
                    format_scores.append(format_score)

            # Calculate averages
            consistency_report["style_consistency"] = (
                sum(style_scores) / len(style_scores) if style_scores else 0.0
            )
            consistency_report["format_consistency"] = (
                sum(format_scores) / len(format_scores) if format_scores else 0.0
            )

            # Analyze terminology consistency
            terminology_score, inconsistencies = self._check_terminology_consistency(all_content)
            consistency_report["terminology_consistency"] = terminology_score
            consistency_report["inconsistencies"] = inconsistencies

            # Calculate overall consistency
            consistency_report["overall_consistency"] = (
                consistency_report["style_consistency"]
                + consistency_report["terminology_consistency"]
                + consistency_report["format_consistency"]
            ) / 3

        except (ValueError, TypeError) as e:
            logger.warning("Error analyzing documentation consistency: %s", e)

        return consistency_report

    def generate_quality_metrics(self, documentation_data: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive quality metrics for documentation."""
        metrics = {
            "readability": {},
            "structure": {},
            "completeness": {},
            "maintainability": {},
            "overall_score": 0.0,
        }

        try:
            # Readability metrics
            metrics["readability"] = self._calculate_readability_metrics(documentation_data)

            # Structure metrics
            metrics["structure"] = self._calculate_structure_metrics(documentation_data)

            # Completeness metrics
            metrics["completeness"] = self._calculate_completeness_metrics(documentation_data)

            # Maintainability metrics
            metrics["maintainability"] = self._calculate_maintainability_metrics(documentation_data)

            # Overall score
            category_scores = [
                metrics["readability"].get("score", 0.0),
                metrics["structure"].get("score", 0.0),
                metrics["completeness"].get("score", 0.0),
                metrics["maintainability"].get("score", 0.0),
            ]
            metrics["overall_score"] = sum(category_scores) / len(category_scores)

        except (ValueError, TypeError) as e:
            logger.warning("Error generating quality metrics: %s", e)

        return metrics

    def _assess_section_quality(self, section_data: Any) -> float:
        """Assess the quality of a single documentation section."""
        try:
            if isinstance(section_data, dict):
                if "content" in section_data:
                    content = section_data["content"]
                    return self._assess_content_quality(content)
                elif "coverage" in section_data:
                    return section_data["coverage"]
                else:
                    # Assess based on data completeness
                    non_empty_fields = sum(1 for value in section_data.values() if value)
                    total_fields = len(section_data)
                    return non_empty_fields / total_fields if total_fields > 0 else 0.0
            elif isinstance(section_data, str):
                return self._assess_content_quality(section_data)
            else:
                return 0.5  # Default score for unknown types

        except (ValueError, TypeError):  # fmt: skip
            return 0.0

    def _assess_content_quality(self, content: str) -> float:
        """Assess the quality of text content."""
        if not content or not content.strip():
            return 0.0

        score = 0.0
        max_score = 1.0

        # Length check
        word_count = len(content.split())
        if word_count > 10:
            score += 0.2
        if word_count > 50:
            score += 0.2

        # Structure check
        if self._has_headers(content):
            score += 0.2

        # Code examples check
        if self._has_code_examples(content):
            score += 0.2

        # Grammar and style check (basic)
        if self._basic_grammar_check(content):
            score += 0.2

        return min(score, max_score)

    def _has_content(self, doc_data: Any) -> bool:
        """Check if documentation data has meaningful content."""
        if isinstance(doc_data, dict):
            return bool(
                doc_data.get("content") or doc_data.get("functions") or doc_data.get("classes")
            )
        elif isinstance(doc_data, str):
            return bool(doc_data.strip())
        return False

    def _extract_content(self, section_data: Any) -> str:
        """Extract text content from section data."""
        if isinstance(section_data, dict):
            return section_data.get("content", "")
        elif isinstance(section_data, str):
            return section_data
        return ""

    def _get_quality_level(self, score: float) -> str:
        """Get quality level based on score."""
        for level, threshold in self.quality_thresholds.items():
            if score >= threshold:
                return level
        return "poor"

    def _generate_recommendations(self, section_scores: dict[str, float]) -> list[str]:
        """Generate recommendations based on section scores."""
        recommendations = []

        for section_name, score in section_scores.items():
            if score < 0.3:
                recommendations.append(
                    f"Section '{section_name}' needs significant improvement (score: {score:.2f})"
                )
            elif score < 0.7:
                recommendations.append(
                    f"Section '{section_name}' could benefit from enhancement (score: {score:.2f})"
                )

        if not recommendations:
            recommendations.append("Documentation quality is good across all sections")

        return recommendations

    def _has_headers(self, content: str) -> bool:
        """Check if content has proper headers."""
        header_patterns = [r"^#+\s", r"^=+$", r"^-+$"]
        for pattern in header_patterns:
            if re.search(pattern, content, re.MULTILINE):
                return True
        return False

    def _has_code_examples(self, content: str) -> bool:
        """Check if content has code examples."""
        code_patterns = [r"```", r"`[^`]+`", r"    \w+"]
        for pattern in code_patterns:
            if re.search(pattern, content):
                return True
        return False

    def _basic_grammar_check(self, content: str) -> bool:
        """Basic grammar and style check."""
        # Check for proper sentence endings
        sentences = re.split(r"[.!?]+", content)
        if len(sentences) < 2:
            return False

        # Check for reasonable sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        return 5 <= avg_sentence_length <= 30

    def _check_style_consistency(self, content: str) -> float:
        """Check style consistency within content."""
        # Basic style checks
        score = 0.0

        # Header consistency
        headers = re.findall(r"^(#+)\s", content, re.MULTILINE)
        if headers and len(set(headers)) <= 3:  # Reasonable header variety
            score += 0.5

        # Code block consistency
        code_blocks = re.findall(r"```(\w*)", content)
        if not code_blocks or len(set(code_blocks)) <= 2:  # Consistent code languages
            score += 0.5

        return score

    def _check_format_consistency(self, content: str) -> float:
        """Check format consistency within content."""
        score = 1.0  # Start with perfect score

        # Check for mixed list styles
        bullet_styles = re.findall(r"^(\s*[-*+])", content, re.MULTILINE)
        if len(set(bullet_styles)) > 1:
            score -= 0.3

        # Check for consistent indentation
        indented_lines = re.findall(r"^(\s+)", content, re.MULTILINE)
        if indented_lines:
            indent_lengths = [len(indent) for indent in indented_lines]
            if len(set(indent_lengths)) > 3:  # Too many different indentation levels
                score -= 0.3

        return max(0.0, score)

    def _check_terminology_consistency(
        self, all_content: list[tuple[str, str]]
    ) -> tuple[float, list[str]]:
        """Check terminology consistency across all content."""
        # Extract technical terms
        all_terms = set()
        section_terms = {}

        for section_name, content in all_content:
            # Extract potential technical terms (capitalized words, underscored words)
            terms = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b|\b\w+_\w+\b", content)
            section_terms[section_name] = set(terms)
            all_terms.update(terms)

        # Find inconsistencies
        inconsistencies = []
        if len(all_terms) > 0:
            # Check for similar terms that might be inconsistent
            term_list = list(all_terms)
            for i, term1 in enumerate(term_list):
                for term2 in term_list[i + 1 :]:
                    if self._are_similar_terms(term1, term2):
                        inconsistencies.append(f"Similar terms found: '{term1}' and '{term2}'")

        # Calculate consistency score
        consistency_score = 1.0 - (len(inconsistencies) / max(len(all_terms), 1))
        return max(0.0, consistency_score), inconsistencies

    def _are_similar_terms(self, term1: str, term2: str) -> bool:
        """Check if two terms are similar and might be inconsistent."""
        # Simple similarity check
        if abs(len(term1) - len(term2)) > 3:
            return False

        # Check for common patterns of inconsistency
        similarity_patterns = [
            (r"(\w+)Service", r"\1Manager"),
            (r"(\w+)Handler", r"\1Processor"),
            (r"get(\w+)", r"fetch\1"),
        ]

        for pattern1, pattern2 in similarity_patterns:
            if re.match(pattern1, term1) and re.match(pattern2, term2):
                return True
            if re.match(pattern2, term1) and re.match(pattern1, term2):
                return True

        return False

    def _calculate_readability_metrics(self, documentation_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate readability metrics."""
        metrics = {"score": 0.0, "word_count": 0, "avg_sentence_length": 0.0}

        try:
            all_text = ""
            for data in documentation_data.values():
                content = self._extract_content(data)
                all_text += " " + content

            if all_text.strip():
                words = all_text.split()
                sentences = re.split(r"[.!?]+", all_text)

                metrics["word_count"] = len(words)
                metrics["avg_sentence_length"] = len(words) / len(sentences) if sentences else 0

                # Simple readability score
                if 10 <= metrics["avg_sentence_length"] <= 20:
                    metrics["score"] = 1.0
                elif 5 <= metrics["avg_sentence_length"] <= 25:
                    metrics["score"] = 0.7
                else:
                    metrics["score"] = 0.4

        except (ValueError, TypeError):  # fmt: skip
            pass

        return metrics

    def _calculate_structure_metrics(self, documentation_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate structure metrics."""
        metrics = {"score": 0.0, "sections": 0, "hierarchy_depth": 0}

        try:
            metrics["sections"] = len(documentation_data)

            # Check for hierarchical structure
            for data in documentation_data.values():
                content = self._extract_content(data)
                headers = re.findall(r"^(#+)", content, re.MULTILINE)
                if headers:
                    max_depth = max(len(h) for h in headers)
                    metrics["hierarchy_depth"] = max(metrics["hierarchy_depth"], max_depth)

            # Score based on structure
            if metrics["sections"] >= 3 and metrics["hierarchy_depth"] >= 2:
                metrics["score"] = 1.0
            elif metrics["sections"] >= 2:
                metrics["score"] = 0.7
            else:
                metrics["score"] = 0.4

        except (ValueError, TypeError):  # fmt: skip
            pass

        return metrics

    def _calculate_completeness_metrics(self, documentation_data: dict[str, Any]) -> dict[str, Any]:
        """Calculate completeness metrics."""
        metrics = {"score": 0.0, "coverage": 0.0}

        try:
            # Basic completeness check
            required_sections = ["architecture", "api_reference", "overview"]
            found_sections = sum(
                1
                for section in required_sections
                if any(section in key.lower() for key in documentation_data.keys())
            )

            metrics["coverage"] = found_sections / len(required_sections)
            metrics["score"] = metrics["coverage"]

        except (ValueError, TypeError):  # fmt: skip
            pass

        return metrics

    def _calculate_maintainability_metrics(
        self, documentation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate maintainability metrics."""
        metrics = {"score": 0.0, "outdated_indicators": 0}

        try:
            # Look for indicators of maintainability
            maintainability_score = 0.0
            total_checks = 0

            for data in documentation_data.values():
                content = self._extract_content(data)
                total_checks += 1

                # Check for recent updates (basic heuristic)
                if any(
                    keyword in content.lower() for keyword in ["updated", "version", "changelog"]
                ):
                    maintainability_score += 1

            metrics["score"] = maintainability_score / total_checks if total_checks > 0 else 0.0

        except (ValueError, TypeError):  # fmt: skip
            pass

        return metrics
