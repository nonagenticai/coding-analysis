"""Code processor module for orchestrating language-specific analyzers.

This module provides a unified interface for analyzing code repositories,
individual files, and code snippets across multiple programming languages.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from models.github_models import CodeSnippet, RepositoryContext

from models.core_models import (
    CoreAnalysisResult,
    RepositoryAnalysis,
    SnippetEvaluation,
)

from ..analyzers.base_analyzer import BaseCodeProcessor
from ..analyzers.javascript_analyzer import JSAnalyzerProcessor
from ..analyzers.python_analyzer import PythonCodeProcessor
from ..analyzers.typescript_analyzer import APITypeScriptProcessor

logger = logging.getLogger(__name__)


class CodeProcessor:
    """Main processor that orchestrates different analyzers

    Attributes:
        extension_mapping (Dict): Instance attribute extension_mapping."""

    def __init__(self) -> None:
        self.analyzers: dict[str, BaseCodeProcessor] = {
            "python": PythonCodeProcessor(),
            "javascript": JSAnalyzerProcessor(),
            "typescript": APITypeScriptProcessor(),
        }

        # File extension mappings
        self.extension_mapping = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",  # Now uses dedicated TypeScript analyzer
            ".tsx": "typescript",
        }

    async def process_repository(self, repo_path: str) -> RepositoryAnalysis:
        """Process entire repository

        Args:
            repo_path (str): The repo_path parameter.

        Returns:
            RepositoryAnalysis: The repositoryanalysis result.

        Example:
            >>> await process_repository("repo_path_value")"""
        try:
            logger.info("Starting repository analysis for: %s", repo_path)

            # Get all code files
            code_files = self._get_code_files(repo_path)
            logger.info("Found %d code files to analyze", len(code_files))

            if not code_files:
                return RepositoryAnalysis(
                    success=True,
                    file_results=[],
                    failed_files=[],
                    summary={
                        "total_files": 0,
                        "message": "No code files found",
                    },
                    metadata={"repo_path": repo_path},
                )

            # Process files in parallel (with limit to avoid overwhelming system)
            semaphore = asyncio.Semaphore(5)  # Limit concurrent processing

            async def process_file(file_path: str) -> CoreAnalysisResult:
                """Async Function process_file.

                Args:
                    file_path (str): The file_path parameter.

                Returns:
                    CoreAnalysisResult: The coreanalysisresult result."""
                async with semaphore:
                    return await self.analyze_file(file_path)

            # Create tasks for all files
            tasks = [process_file(file_path) for file_path in code_files]

            # Wait for all tasks to complete
            file_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and create proper results
            successful_results: list[Any] = []
            failed_results: list[Any] = []
            for i, result in enumerate(file_results):
                if isinstance(result, Exception):
                    failed_results.append(
                        {
                            "file_path": code_files[i],
                            "error": str(result),
                            "error_type": type(result).__name__,
                        }
                    )
                    logger.error("Failed to analyze %s: %s", code_files[i], result)
                else:
                    successful_results.append(result)

            # Create summary
            summary = self._create_repository_summary(successful_results, failed_results)

            logger.info(
                "Repository analysis complete. Success: %d, Failed: %d",
                len(successful_results),
                len(failed_results),
            )

            return RepositoryAnalysis(
                success=True,
                file_results=successful_results,
                failed_files=failed_results,
                summary=summary,
                metadata={
                    "repo_path": repo_path,
                    "total_files": len(code_files),
                    "successful_files": len(successful_results),
                    "failed_files": len(failed_results),
                },
            )

        except (OSError, ValueError) as e:
            logger.error("Repository analysis failed for %s: %s", repo_path, e)
            return RepositoryAnalysis(
                success=False,
                file_results=[],
                failed_files=[],
                summary={},
                metadata={"repo_path": repo_path, "error": str(e)},
            )

    async def analyze_file(self, file_path: str) -> CoreAnalysisResult:
        """Analyze a single file

        Args:
            file_path (str): The file_path parameter.

        Returns:
            CoreAnalysisResult: The coreanalysisresult result.

        Example:
            >>> await analyze_file("file_path_value")"""
        try:
            language = self._detect_language(file_path)

            if language not in self.analyzers:
                return CoreAnalysisResult(
                    success=False,
                    language=language,
                    file_path=file_path,
                    error=f"Unsupported language: {language}",
                    complexity_metrics=None,
                    parse_time_ms=0.0,
                    total_lines=0,
                    metadata={
                        "file_path": file_path,
                        "detected_language": language,
                    },
                )

            analyzer = self.analyzers[language]
            result = await analyzer.analyze_file(file_path)

            # Ensure metadata includes file path
            if "file_path" not in result.metadata:
                result.metadata["file_path"] = file_path

            return result

        except (OSError, ValueError) as e:
            logger.error("File analysis failed for %s: %s", file_path, e)
            return CoreAnalysisResult(
                success=False,
                language="unknown",
                file_path=file_path,
                error=str(e),
                complexity_metrics=None,
                parse_time_ms=0.0,
                total_lines=0,
                metadata={
                    "file_path": file_path,
                    "error_type": type(e).__name__,
                },
            )

    async def evaluate_snippet(
        self, snippet: CodeSnippet, context: RepositoryContext | None = None
    ) -> SnippetEvaluation:
        """Evaluate how a code snippet fits with existing codebase

        Args:
            snippet (CodeSnippet): The snippet parameter.
            context (Optional[RepositoryContext], optional): The context parameter. Defaults to None.

        Returns:
            SnippetEvaluation: The snippetevaluation result.

        Example:
            >>> await evaluate_snippet("snippet_value", "context_value")"""
        try:
            language = snippet.language

            if language not in self.analyzers:
                return SnippetEvaluation(
                    success=False,
                    error=f"Unsupported language: {language}",
                    analysis=None,
                    compatibility_score=0.0,
                    recommendations=[],
                    metadata={"snippet_language": language},
                )

            analyzer = self.analyzers[language]

            # Analyze snippet
            snippet_analysis = await analyzer.analyze_snippet(snippet.code, snippet.file_path)

            if not snippet_analysis.success:
                return SnippetEvaluation(
                    success=False,
                    error=f"Snippet analysis failed: {snippet_analysis.error}",
                    analysis=snippet_analysis,
                    compatibility_score=0.0,
                    recommendations=[],
                    metadata={"snippet_language": language},
                )

            # Calculate compatibility score if context provided
            compatibility_score = 0.85  # Default placeholder
            recommendations: list[Any] = []
            if context:
                compatibility_score = await self._calculate_compatibility(snippet_analysis, context)
                recommendations = await self._generate_recommendations(snippet_analysis, context)
            else:
                recommendations = [
                    "Consider adding type hints for better code clarity",
                    "Ensure function has appropriate docstring",
                    "Add unit tests for new functionality",
                ]

            return SnippetEvaluation(
                success=True,
                analysis=snippet_analysis,
                compatibility_score=compatibility_score,
                recommendations=recommendations,
                metadata={
                    "snippet_language": language,
                    "has_context": context is not None,
                },
                error=None,
            )

        except (ValueError, AttributeError) as e:
            logger.error("Snippet evaluation failed: %s", e)
            return SnippetEvaluation(
                success=False,
                error=str(e),
                analysis=None,
                compatibility_score=0.0,
                recommendations=[],
                metadata={"snippet_language": (snippet.language if snippet else "unknown")},
            )

    def _get_code_files(self, directory: str) -> list[str]:
        """Get all code files in directory"""
        code_files: list[str] = []
        directory_path = Path(directory)

        if not directory_path.exists():
            logger.warning("Directory does not exist: %s", directory)
            return code_files

        # Skip common non-code directories
        skip_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            ".pytest_cache",
            ".mypy_cache",
            "dist",
            "build",
            ".tox",
            ".coverage",
            "htmlcov",
            ".hypothesis",
            "eggs",
            "*.egg-info",
        }

        for root, dirs, files in os.walk(directory):
            # Filter out directories to skip
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]

            for file in files:
                file_path = os.path.join(root, file)
                if self._is_code_file(file_path):
                    code_files.append(file_path)

        return sorted(code_files)  # Sort for consistent ordering

    def _is_code_file(self, file_path: str) -> bool:
        """Check if file is a code file we can analyze"""
        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.extension_mapping

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        _, ext = os.path.splitext(file_path)
        return self.extension_mapping.get(ext.lower(), "unknown")

    def _create_repository_summary(
        self,
        successful_results: list[CoreAnalysisResult],
        failed_results: list[dict],
    ) -> dict[str, Any]:
        """Create summary of repository analysis"""
        total_entities = sum(len(result.entities) for result in successful_results)
        total_relationships = sum(len(result.relationships) for result in successful_results)
        total_dependencies = sum(len(result.dependencies) for result in successful_results)

        # Language breakdown
        language_counts: dict[str, int] = {}
        for result in successful_results:
            lang = result.metadata.get("language", "unknown")
            language_counts[lang] = language_counts.get(lang, 0) + 1

        # Calculate success rate
        total_files = len(successful_results) + len(failed_results)
        success_rate = len(successful_results) / total_files if total_files > 0 else 0

        return {
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "total_dependencies": total_dependencies,
            "language_breakdown": language_counts,
            "analysis_success_rate": success_rate,
            "successful_files": len(successful_results),
            "failed_files": len(failed_results),
            "total_files_analyzed": total_files,
        }

    async def _calculate_compatibility(
        self, snippet_analysis: CoreAnalysisResult, context: RepositoryContext
    ) -> float:
        """Calculate how compatible snippet is with existing codebase"""
        base_score = 0.8

        # Adjust score based on analysis results
        if snippet_analysis.entities:
            # Bonus for having entities (structured code)
            base_score += 0.1

        if snippet_analysis.dependencies:
            # Check if dependencies are compatible with context
            dependency_score = await self._analyze_dependency_compatibility(
                snippet_analysis.dependencies, context
            )
            base_score += dependency_score * 0.1

        # Check naming convention compatibility
        naming_score = self._analyze_naming_conventions(snippet_analysis, context)
        base_score += naming_score * 0.05

        # Check import pattern consistency
        import_score = self._analyze_import_patterns(snippet_analysis, context)
        base_score += import_score * 0.05

        return min(base_score, 1.0)

    async def _analyze_dependency_compatibility(
        self, snippet_dependencies: list, context: RepositoryContext
    ) -> float:
        """
        Analyze compatibility of snippet dependencies with existing codebase.

        Args:
            snippet_dependencies: Dependencies from the snippet
            context: Repository context with existing dependency information

        Returns:
            Compatibility score between 0.0 and 1.0
        """
        if not snippet_dependencies:
            return 1.0  # No dependencies means no conflicts

        # Extract dependencies from context patterns or return neutral score
        context_dependencies = (
            context.existing_patterns if hasattr(context, "existing_patterns") else []
        )

        if not context_dependencies:
            # No context dependencies to compare against
            return 0.8  # Neutral score

        # Get context dependency names
        context_dep_names = self._extract_context_dependency_names(context_dependencies)

        # Analyze each dependency
        dep_stats = self._analyze_dependencies(
            snippet_dependencies, context_dependencies, context_dep_names
        )

        # Calculate final score
        return self._calculate_dependency_score(dep_stats, len(snippet_dependencies))

    def _extract_context_dependency_names(self, context_dependencies: list) -> set:
        """Extract dependency names from context dependencies."""
        context_dep_names = set()
        for dep in context_dependencies:
            if isinstance(dep, dict) and "name" in dep:
                context_dep_names.add(dep["name"])
            elif isinstance(dep, str):
                name = self._extract_package_name(dep)
                context_dep_names.add(name)
        return context_dep_names

    def _extract_package_name(self, dep_string: str) -> str:
        """Extract package name from dependency string."""
        return (
            dep_string.split("==")[0]
            .split(">=")[0]
            .split("<=")[0]
            .split(">")[0]
            .split("<")[0]
            .strip()
        )

    def _analyze_dependencies(
        self,
        snippet_dependencies: list,
        context_dependencies: list,
        context_dep_names: set,
    ) -> dict[str, int]:
        """Analyze snippet dependencies against context."""
        dep_stats = {"compatible": 0, "conflicting": 0, "new": 0}

        for dep in snippet_dependencies:
            # Handle both string and DependencyInfoModel types
            if hasattr(dep, "module_name"):  # DependencyInfoModel object
                dep_name = dep.module_name
                dep_string = dep.module_name  # No version info in DependencyInfoModel
            else:  # String
                dep_string = dep
                dep_name = self._extract_package_name(dep)

            if dep_name in context_dep_names:
                # Check version compatibility
                context_dep_strings = self._convert_context_deps_to_strings(context_dependencies)
                if self._check_version_compatibility(dep_string, context_dep_strings):
                    dep_stats["compatible"] += 1
                else:
                    dep_stats["conflicting"] += 1
            else:
                dep_stats["new"] += 1

        return dep_stats

    def _convert_context_deps_to_strings(self, context_dependencies: list) -> list[str]:
        """Convert context dependencies to string format."""
        context_dep_strings = []
        for context_dep in context_dependencies:
            if isinstance(context_dep, dict) and "name" in context_dep:
                # Convert dict to string format
                version = context_dep.get("version", "")
                if version:
                    context_dep_strings.append(f"{context_dep['name']}=={version}")
                else:
                    context_dep_strings.append(context_dep["name"])
            elif isinstance(context_dep, str):
                context_dep_strings.append(context_dep)
        return context_dep_strings

    def _calculate_dependency_score(self, dep_stats: dict[str, int], total_deps: int) -> float:
        """Calculate dependency compatibility score."""
        if dep_stats["conflicting"] > 0:
            # Penalize version conflicts heavily
            return max(0.0, 0.5 - (dep_stats["conflicting"] / total_deps) * 0.5)
        if dep_stats["compatible"] > 0:
            # Reward compatible dependencies
            return min(1.0, 0.8 + (dep_stats["compatible"] / total_deps) * 0.2)
        # All new dependencies - neutral score
        return 0.7

    def _check_version_compatibility(self, snippet_dep: str, context_deps: list[str]) -> bool:
        """
        Check if snippet dependency version is compatible with context versions.

        Args:
            snippet_dep: Dependency from snippet (e.g., "package>=1.0.0")
            context_deps: List of context dependencies

        Returns:
            True if compatible, False if conflict detected
        """
        # Extract package name from snippet dependency
        dep_name = (
            snippet_dep.split("==")[0]
            .split(">=")[0]
            .split("<=")[0]
            .split(">")[0]
            .split("<")[0]
            .strip()
        )

        # Find matching dependency in context
        for context_dep in context_deps:
            context_name = (
                context_dep.split("==")[0]
                .split(">=")[0]
                .split("<=")[0]
                .split(">")[0]
                .split("<")[0]
                .strip()
            )

            if dep_name == context_name:
                # Basic compatibility check - exact matches are compatible
                if snippet_dep == context_dep:
                    return True

                # Extract version constraints for more sophisticated checking
                snippet_version = self._extract_version_info(snippet_dep)
                context_version = self._extract_version_info(context_dep)

                # If either has no version constraint, assume compatible
                if not snippet_version["version"] or not context_version["version"]:
                    return True

                # Check for obvious conflicts (different exact versions)
                if (
                    snippet_version["operator"] == "=="
                    and context_version["operator"] == "=="
                    and snippet_version["version"] != context_version["version"]
                ):
                    return False

                # For other cases, assume compatible (more sophisticated logic could be added)
                return True

        return True  # Not found in context, so no conflict

    def _extract_version_info(self, dependency: str) -> dict[str, str | None]:
        """
        Extract version information from a dependency string.

        Args:
            dependency: Dependency string (e.g., "package>=1.0.0")

        Returns:
            Dictionary with 'operator' and 'version' keys
        """

        # Match version patterns
        version_pattern = r"([><=!~]+)([0-9]+(?:\.[0-9]+)*(?:\.[a-zA-Z0-9]+)*)"
        match = re.search(version_pattern, dependency)

        if match:
            return {"operator": match.group(1), "version": match.group(2)}

        return {"operator": None, "version": None}

    def _analyze_naming_conventions(
        self, snippet_analysis: CoreAnalysisResult, context: RepositoryContext
    ) -> float:
        """
        Analyze naming convention compatibility between snippet and context.

        Args:
            snippet_analysis: Analysis result from snippet
            context: Repository context

        Returns:
            Naming compatibility score between 0.0 and 1.0
        """
        # Extract naming patterns from style guide
        naming_patterns = context.style_guide if hasattr(context, "style_guide") else {}

        if not snippet_analysis.entities or not naming_patterns:
            return 0.8  # Neutral score if no data to compare

        compatible_patterns = 0
        total_entities = len(snippet_analysis.entities)

        for entity in snippet_analysis.entities:
            entity_name = entity.name
            entity_type = entity.type.value

            # Check against context naming patterns
            if entity_type in naming_patterns:
                expected_pattern = naming_patterns[entity_type]
                if self._matches_naming_pattern(entity_name, expected_pattern):
                    compatible_patterns += 1

        return compatible_patterns / total_entities if total_entities > 0 else 0.8

    def _matches_naming_pattern(self, name: str, pattern: str) -> bool:
        """
        Check if a name matches the expected naming pattern.

        Args:
            name: Entity name to check
            pattern: Expected naming pattern (e.g., "snake_case", "camelCase")

        Returns:
            True if name matches pattern, False otherwise
        """

        if pattern == "snake_case":
            return bool(re.match(r"^[a-z_][a-z0-9_]*$", name))
        if pattern == "camelCase":
            return bool(re.match(r"^[a-z][a-zA-Z0-9]*$", name))
        if pattern == "PascalCase":
            return bool(re.match(r"^[A-Z][a-zA-Z0-9]*$", name))
        if pattern == "UPPER_CASE":
            return bool(re.match(r"^[A-Z_][A-Z0-9_]*$", name))
        # Unknown pattern, assume compatible
        return True

    def _analyze_import_patterns(
        self, snippet_analysis: CoreAnalysisResult, context: RepositoryContext
    ) -> float:
        """
        Analyze import pattern compatibility between snippet and context.

        Args:
            snippet_analysis: Analysis result from snippet
            context: Repository context

        Returns:
            Import pattern compatibility score between 0.0 and 1.0
        """
        # Extract import patterns from style guide
        import_patterns = (
            context.style_guide.get("import_patterns", [])
            if hasattr(context, "style_guide")
            else []
        )

        if not snippet_analysis.dependencies or not import_patterns:
            return 0.8  # Neutral score if no data to compare

        compatible_imports = 0
        total_imports = len(snippet_analysis.dependencies)

        for dependency in snippet_analysis.dependencies:
            # Check if import follows expected patterns
            for pattern in import_patterns:
                if self._matches_import_pattern(dependency, pattern):
                    compatible_imports += 1
                    break

        return compatible_imports / total_imports if total_imports > 0 else 0.8

    def _matches_import_pattern(self, dependency, pattern: str) -> bool:
        """
        Check if an import matches the expected pattern.

        Args:
            dependency: Import dependency to check (string or DependencyInfoModel)
            pattern: Expected import pattern

        Returns:
            True if import matches pattern, False otherwise
        """
        # Extract module name from dependency
        if hasattr(dependency, "module_name"):
            dep_string = dependency.module_name
        else:
            dep_string = str(dependency)

        # Basic pattern matching - could be extended with more sophisticated logic
        if pattern == "relative_imports":
            return bool(dep_string.startswith("."))
        if pattern == "absolute_imports":
            return bool(not dep_string.startswith("."))
        if pattern == "package_imports":
            return bool("." not in dep_string or not dep_string.startswith("."))
        # Unknown pattern, assume compatible
        return True

    async def _generate_recommendations(
        self,
        snippet_analysis: CoreAnalysisResult,
        context: RepositoryContext,
    ) -> list[str]:
        """Generate recommendations for snippet integration"""
        recommendations: list[Any] = []
        # Check for missing docstrings
        has_functions = any(entity.type.value == "function" for entity in snippet_analysis.entities)
        if has_functions:
            recommendations.append("Ensure all functions have appropriate docstrings")

        # Check for type hints
        if snippet_analysis.metadata.get("language") == "python":
            recommendations.append("Consider adding type hints for better code clarity")

        # General recommendations
        recommendations.extend(
            [
                "Add unit tests for new functionality",
                "Follow existing code style conventions",
                "Consider impact on existing dependencies",
            ]
        )

        return recommendations
