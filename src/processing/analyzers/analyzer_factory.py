"""
Analyzer Factory for creating appropriate code analyzers.

Provides a centralized way to create and manage code analyzers
for different programming languages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.file_service import FileService

from .base_analyzer import BaseCodeProcessor
from .javascript_analyzer import JSAnalyzerProcessor as JavaScriptAnalyzer
from .python_analyzer import PythonCodeProcessor as PythonAnalyzer
from .typescript_analyzer import APITypeScriptProcessor as TypeScriptAnalyzer

logger = logging.getLogger(__name__)


class AnalyzerFactory:
    """Factory for creating and managing code analyzers."""

    def __init__(self, file_service: FileService | None = None):
        """Initialize the analyzer factory."""
        self.file_service = file_service
        self._analyzers: dict[str, BaseCodeProcessor] = {}
        self._language_mappings: dict[str, type[BaseCodeProcessor]] = {
            "python": PythonAnalyzer,
            "javascript": JavaScriptAnalyzer,
            "typescript": TypeScriptAnalyzer,
        }
        self._extension_mappings: dict[str, str] = {
            ".py": "python",
            ".pyw": "python",
            ".pyx": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".mjs": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".d.ts": "typescript",
        }

    def get_analyzer(self, language: str) -> BaseCodeProcessor | None:
        """Get an analyzer instance for the specified language."""
        try:
            if language not in self._analyzers:
                analyzer_class = self._language_mappings.get(language)
                if analyzer_class:
                    self._analyzers[language] = analyzer_class()
                    logger.debug("Created %s analyzer", language)
                else:
                    logger.warning("No analyzer available for language: %s", language)
                    return None

            return self._analyzers[language]

        except (ImportError, AttributeError, TypeError) as e:
            logger.error("Error creating analyzer for %s: %s", language, e)
            return None

    def get_analyzer_for_file(self, file_path: str) -> BaseCodeProcessor | None:
        """Get the appropriate analyzer for a file based on its extension."""
        try:
            path = Path(file_path)

            # Handle .d.ts files specifically
            if file_path.endswith(".d.ts"):
                return self.get_analyzer("typescript")

            extension = path.suffix.lower()
            language = self._extension_mappings.get(extension)

            if language:
                return self.get_analyzer(language)

            logger.debug("No analyzer available for file extension: %s", extension)
            return None

        except (OSError, ValueError, AttributeError) as e:
            logger.error("Error determining analyzer for file %s: %s", file_path, e)
            return None

    def get_supported_languages(self) -> list[str]:
        """Get list of supported programming languages."""
        return list(self._language_mappings.keys())

    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions."""
        return list(self._extension_mappings.keys())

    def is_supported_file(self, file_path: str) -> bool:
        """Check if a file is supported by any analyzer."""
        try:
            if file_path.endswith(".d.ts"):
                return True

            path = Path(file_path)
            extension = path.suffix.lower()
            return extension in self._extension_mappings

        except (OSError, ValueError):  # fmt: skip
            return False

    def get_language_for_file(self, file_path: str) -> str | None:
        """Get the programming language for a file based on its extension."""
        try:
            if file_path.endswith(".d.ts"):
                return "typescript"

            path = Path(file_path)
            extension = path.suffix.lower()
            return self._extension_mappings.get(extension)

        except (OSError, ValueError):  # fmt: skip
            return None

    def register_analyzer(
        self,
        language: str,
        analyzer_class: type[BaseCodeProcessor],
        extensions: list[str],
    ) -> None:
        """Register a new analyzer for a language."""
        try:
            self._language_mappings[language] = analyzer_class

            # Update extension mappings
            for ext in extensions:
                self._extension_mappings[ext.lower()] = language

            # Clear cached analyzer if it exists
            if language in self._analyzers:
                del self._analyzers[language]

            logger.info("Registered analyzer for %s with extensions: %s", language, extensions)

        except (TypeError, AttributeError, ValueError) as e:
            logger.error("Error registering analyzer for %s: %s", language, e)

    def get_analyzer_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all available analyzers."""
        info = {}

        for language, analyzer_class in self._language_mappings.items():
            try:
                # Get supported extensions for this language
                extensions = [
                    ext for ext, lang in self._extension_mappings.items() if lang == language
                ]

                info[language] = {
                    "class": analyzer_class.__name__,
                    "extensions": extensions,
                    "cached": language in self._analyzers,
                }

            except (AttributeError, TypeError) as e:
                logger.error("Error getting info for %s analyzer: %s", language, e)
                info[language] = {"error": str(e)}

        return info

    async def initialize_all_analyzers(self) -> None:
        """Initialize all available analyzers."""
        for language in self._language_mappings:
            try:
                analyzer = self.get_analyzer(language)
                if analyzer and hasattr(analyzer, "initialize"):
                    await analyzer.initialize()  # type: ignore[call-non-callable]  # ty: ignore[call-non-callable]
                    logger.debug("Initialized %s analyzer", language)
            except (AttributeError, ImportError, RuntimeError) as e:
                logger.error("Error initializing %s analyzer: %s", language, e)

    def clear_cache(self) -> None:
        """Clear all cached analyzer instances."""
        self._analyzers.clear()
        logger.debug("Cleared analyzer cache")


# Global factory instance
_default_factory: AnalyzerFactory | None = None


def get_analyzer_factory(
    file_service: FileService | None = None,
) -> AnalyzerFactory:
    """Get the default analyzer factory instance."""
    global _default_factory

    if _default_factory is None or (file_service and _default_factory.file_service != file_service):
        _default_factory = AnalyzerFactory(file_service)

    return _default_factory


def get_analyzer_for_file(
    file_path: str, file_service: FileService | None = None
) -> BaseCodeProcessor | None:
    """Convenience function to get analyzer for a file."""
    factory = get_analyzer_factory(file_service)
    return factory.get_analyzer_for_file(file_path)


def get_analyzer(
    language: str, file_service: FileService | None = None
) -> BaseCodeProcessor | None:
    """Convenience function to get analyzer for a language."""
    factory = get_analyzer_factory(file_service)
    return factory.get_analyzer(language)
