"""Abstract base class for language-specific code document processors.

This module provides the base class for implementing language-specific
code document processors that extract entities and relationships from
source code files.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from app.core.file_service import FileService

logger = logging.getLogger(__name__)


class CodeDocumentProcessor(ABC):
    """Abstract base class for language-specific code document processors."""

    def __init__(self, file_service: FileService | None = None) -> None:
        """Initialize the code document processor.

        Args:
            file_service: Optional file service for file operations
        """
        self.file_service = file_service
        self.language = "unknown"

    @abstractmethod
    async def process_document(self, file_path: str, code: str | None = None) -> dict[str, Any]:
        """Process a code document to extract entities and relationships.

        Args:
            file_path: Path to the file to process
            code: Optional code content if already loaded

        Returns:
            Dictionary with extracted entities and relationships
        """

    @abstractmethod
    def get_supported_extensions(self) -> list[str]:
        """Get list of file extensions supported by this processor.

        Returns:
            List of file extensions (e.g., ['.py', '.pyi'])
        """

    def _get_module_name(self, file_path: str) -> str:
        """Extract module name from file path.

        Args:
            file_path: Path to the file

        Returns:
            Module name derived from file path
        """
        basename = os.path.basename(file_path)
        module_name, _ = os.path.splitext(basename)
        return module_name
