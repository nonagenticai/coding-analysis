"""
Processor factory for creating language-specific code processors with proper dependency injection.
"""

from __future__ import annotations

import logging
import os

from app.core.file_service import FileService

from .code_document_processor import CodeDocumentProcessor
from .javascript_processor import LegacyJavaScriptProcessor
from .python_processor import CorePythonProcessor

logger = logging.getLogger(__name__)


class ProcessorFactory:
    """Factory for creating code processors with proper dependency injection."""

    def __init__(
        self,
        file_service: FileService,
    ):
        """Initialize the processor factory.

        Args:
            file_service: File service for file operations
        """
        self.file_service = file_service

        # Registry of supported processors
        self._processors: dict[str, type[CodeDocumentProcessor]] = {
            "python": CorePythonProcessor,
            "javascript": LegacyJavaScriptProcessor,
            "typescript": LegacyJavaScriptProcessor,  # TypeScript uses the same processor
        }

        # File extension to language mapping
        self._extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
        }

    def get_processor(self, language: str) -> CodeDocumentProcessor | None:
        """Get a processor for the specified language.

        Args:
            language: Programming language name

        Returns:
            Configured processor instance or None if not supported
        """
        processor_class = self._processors.get(language.lower())
        if not processor_class:
            logger.warning("No processor available for language: %s", language)
            return None

        try:
            # Create processor with dependency injection
            processor = processor_class(file_service=self.file_service)
            logger.debug(
                "Created %s processor: %s",
                language,
                processor.__class__.__name__,
            )
            return processor
        except Exception as e:
            logger.error("Failed to create processor for %s: %s", language, e)
            return None

    def get_processor_for_file(self, file_path: str) -> CodeDocumentProcessor | None:
        """Get a processor based on file extension.

        Args:
            file_path: Path to the file

        Returns:
            Configured processor instance or None if not supported
        """
        _, ext = os.path.splitext(file_path.lower())

        language = self._extension_map.get(ext)
        if not language:
            logger.debug("No processor available for file extension: %s", ext)
            return None

        return self.get_processor(language)

    def get_supported_languages(self) -> list[str]:
        """Get list of supported programming languages.

        Returns:
            List of supported language names
        """
        return list(self._processors.keys())

    def get_supported_extensions(self) -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported file extensions
        """
        return list(self._extension_map.keys())

    def get_extension_language_map(self) -> dict[str, str]:
        """Get mapping of file extensions to languages.

        Returns:
            Dictionary mapping extensions to language names
        """
        return self._extension_map.copy()

    def is_supported_file(self, file_path: str) -> bool:
        """Check if a file is supported by any processor.

        Args:
            file_path: Path to the file

        Returns:
            True if file is supported, False otherwise
        """
        _, ext = os.path.splitext(file_path.lower())
        return ext in self._extension_map


# Global factory instance
_processor_factory: ProcessorFactory | None = None


def get_processor_factory(
    file_service: FileService | None = None,
) -> ProcessorFactory:
    """Get the global processor factory instance.

    Args:
        file_service: File service for dependency injection

    Returns:
        ProcessorFactory instance
    """
    global _processor_factory

    if _processor_factory is None:
        if file_service is None:
            # Create default dependencies if not provided
            file_service = FileService()

        _processor_factory = ProcessorFactory(file_service)
        logger.info("Created global processor factory")

    return _processor_factory
