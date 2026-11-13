"""File analyzer for language-agnostic file analysis.

This module provides a general-purpose file analyzer that can delegate
to appropriate language-specific analyzers based on file extensions.
It's designed to work with the incremental update system.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from models.core_models import (
    CoreAnalysisResult,
)
from processing.analyzers.analyzer_factory import AnalyzerFactory
from processing.analyzers.base_analyzer import BaseCodeProcessor

logger = logging.getLogger(__name__)

# Supported programming languages
SupportedLanguage = Literal[
    "python", "javascript", "typescript", "java", "go", "rust", "cpp", "csharp"
]

# File extension mapping type
ExtensionMap = dict[str, SupportedLanguage]


class FileAnalyzer(BaseCodeProcessor):
    """General-purpose file analyzer that delegates to language-specific analyzers.

    This analyzer acts as a facade that automatically selects and uses the
    appropriate language-specific analyzer based on file extensions. It provides
    a unified interface for file analysis across multiple programming languages.
    """

    def __init__(self) -> None:
        """Initialize the file analyzer with analyzer factory.

        Creates an instance of FileAnalyzer with an empty analyzer cache
        and initializes the analyzer factory for language-specific analyzers.
        """
        logger.info("Initializing FileAnalyzer with multi-language support")
        self.analyzer_factory: AnalyzerFactory = AnalyzerFactory()
        self._analyzer_cache: dict[str, BaseCodeProcessor] = {}
        self._extension_map: ExtensionMap = self._create_extension_map()

    async def analyze_file(self, file_path: str) -> CoreAnalysisResult:
        """Analyze a single file using the appropriate language analyzer.

        Args:
            file_path: Path to the file to analyze

        Returns:
            CoreAnalysisResult containing entities, relationships, and metrics

        Raises:
            ValueError: If no suitable analyzer is found for the file type
        """
        try:
            # Get the appropriate analyzer for this file
            analyzer = await self._get_analyzer_for_file(file_path)
            if not analyzer:
                logger.warning(f"No analyzer found for file: {file_path}")
                return CoreAnalysisResult(
                    success=False,
                    language="unknown",
                    file_path=file_path,
                    parse_time_ms=0.0,
                    total_lines=0,
                    entities=[],
                    relationships=[],
                    dependencies=[],
                    complexity_metrics=None,
                    security_patterns=[],
                    error=f"No analyzer available for file type: {Path(file_path).suffix}",
                    metadata={},
                )

            # Delegate to the language-specific analyzer
            return await analyzer.analyze_file(file_path)

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return CoreAnalysisResult(
                success=False,
                language="unknown",
                file_path=file_path,
                parse_time_ms=0.0,
                total_lines=0,
                entities=[],
                relationships=[],
                dependencies=[],
                complexity_metrics=None,
                security_patterns=[],
                error=str(e),
                metadata={},
            )

    async def analyze_directory(self, directory_path: str) -> CoreAnalysisResult:
        """Analyze all supported files in a directory.

        Args:
            directory_path: Path to the directory to analyze

        Returns:
            Aggregated CoreAnalysisResult for all files in the directory
        """
        all_entities = []
        all_relationships = []
        all_dependencies = []
        all_security_patterns = []
        file_count = 0
        error_count = 0
        languages_found = set()

        directory = Path(directory_path)

        # Find all supported files
        for file_path in directory.rglob("*"):
            if file_path.is_file() and self.can_analyze_file(str(file_path)):
                result = await self.analyze_file(str(file_path))

                if result.success:
                    all_entities.extend(result.entities)
                    all_relationships.extend(result.relationships)
                    all_dependencies.extend(result.dependencies)
                    all_security_patterns.extend(result.security_patterns)
                    languages_found.add(result.language)
                    file_count += 1
                else:
                    error_count += 1

        return CoreAnalysisResult(
            success=error_count == 0,
            language="multi",
            file_path=directory_path,
            parse_time_ms=0.0,
            total_lines=0,  # Would need to aggregate from all file results
            entities=all_entities,
            relationships=all_relationships,
            dependencies=all_dependencies,
            complexity_metrics=None,  # Aggregated metrics would go here
            security_patterns=all_security_patterns,
            error=f"{error_count} files failed to analyze" if error_count > 0 else None,
            metadata={
                "directory": directory_path,
                "files_analyzed": file_count,
                "error_count": error_count,
                "languages": list(languages_found),
            },
        )

    def get_supported_extensions(self) -> list[str]:
        """Return list of all supported file extensions.

        Returns:
            Combined list of extensions from all available analyzers
        """
        extensions: set[str] = set()

        # Get extensions from all registered analyzers
        supported_languages: list[SupportedLanguage] = [
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
            "rust",
            "cpp",
            "csharp",
        ]
        for language in supported_languages:
            try:
                analyzer = self.analyzer_factory.get_analyzer(language)
                if analyzer:
                    extensions.update(analyzer.get_supported_extensions())
            except Exception as e:
                logger.debug(f"Analyzer for {language} not available: {e}")
                continue

        return sorted(list(extensions))

    async def analyze_snippet(
        self, code_snippet: str, file_path: str | None = None
    ) -> CoreAnalysisResult:
        """Analyze a code snippet.

        Args:
            code_snippet: The code to analyze
            file_path: Optional file path to determine language

        Returns:
            CoreAnalysisResult for the code snippet
        """
        # Try to determine language from file path
        if file_path:
            analyzer = await self._get_analyzer_for_file(file_path)
            if analyzer:
                return await analyzer.analyze_snippet(code_snippet, file_path)

        # Try each analyzer until one succeeds
        priority_languages: list[str] = ["python", "javascript", "typescript"]
        for language in priority_languages:
            try:
                analyzer = self.analyzer_factory.get_analyzer(language)
                if analyzer:
                    result = await analyzer.analyze_snippet(code_snippet, file_path)
                    if result.success:
                        logger.debug(f"Successfully analyzed snippet as {language}")
                        return result
            except Exception as e:
                logger.debug(f"Failed to analyze snippet as {language}: {e}")
                continue

        return CoreAnalysisResult(
            success=False,
            language="unknown",
            file_path=file_path or "<snippet>",
            parse_time_ms=0.0,
            total_lines=len(code_snippet.splitlines()),
            entities=[],
            relationships=[],
            dependencies=[],
            complexity_metrics=None,
            security_patterns=[],
            error="Could not determine language for snippet",
            metadata={},
        )

    def get_language_name(self) -> str:
        """Return the analyzer type name.

        Returns:
            'file' as this is a general file analyzer
        """
        return "file"

    def can_analyze_file(self, file_path: str) -> bool:
        """Check if the file can be analyzed by any available analyzer.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file extension is supported by any analyzer
        """
        extension = Path(file_path).suffix.lower()
        supported_extensions = self.get_supported_extensions()
        return extension in supported_extensions

    async def _get_analyzer_for_file(self, file_path: str) -> BaseCodeProcessor | None:
        """Get the appropriate analyzer for a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            The appropriate language analyzer or None
        """
        # Check cache first
        extension = Path(file_path).suffix.lower()
        if extension in self._analyzer_cache:
            return self._analyzer_cache[extension]

        # Use the pre-built extension map
        language = self._extension_map.get(extension)
        if language:
            try:
                analyzer = self.analyzer_factory.get_analyzer(language)
                if analyzer:
                    self._analyzer_cache[extension] = analyzer
                    return analyzer
            except Exception as e:
                logger.debug(f"Could not get {language} analyzer: {e}")

        return None

    def _create_extension_map(self) -> ExtensionMap:
        """Create a mapping of file extensions to languages.

        Returns:
            Dictionary mapping file extensions to supported languages
        """
        return {
            # Python extensions
            ".py": "python",
            ".pyx": "python",
            ".pyi": "python",
            # JavaScript extensions
            ".js": "javascript",
            ".jsx": "javascript",
            ".mjs": "javascript",
            # TypeScript extensions
            ".ts": "typescript",
            ".tsx": "typescript",
            # Java extensions
            ".java": "java",
            # Go extensions
            ".go": "go",
            # Rust extensions
            ".rs": "rust",
            # C++ extensions
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".hpp": "cpp",
            ".h": "cpp",
            # C# extensions
            ".cs": "csharp",
        }
