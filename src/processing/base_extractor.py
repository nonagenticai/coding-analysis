"""Base classes for language-agnostic code entity and relation extraction.

This module provides abstract base classes that define the common interface
for extracting entities and relationships from parsed code ASTs across different
programming languages.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)  # vulture: ignore


class BaseEntityExtractor(ABC):
    """Abstract base class for language-specific entity extractors."""

    @abstractmethod
    async def extract_entities(
        self, ast: Any, file_path: str, module_name: str
    ) -> list[dict[str, Any]]:
        """Extract code entities from parsed AST.

        Args:
            ast: Parsed AST or structured representation
            file_path: Path to the source file
            module_name: Name of the module being processed

        Returns:
            List of extracted entity dictionaries
        """


class BaseRelationExtractor(ABC):
    """Abstract base class for language-specific relation extractors."""

    @abstractmethod
    async def extract_all_relations(
        self, ast: Any, file_path: str, module_name: str
    ) -> list[dict[str, Any]]:
        """Extract all relationships between entities from parsed AST.

        Args:
            ast: Parsed AST or structured representation
            file_path: Path to the source file
            module_name: Name of the module being processed

        Returns:
            List of relationship dictionaries
        """


# Backward compatibility aliases
EntityExtractor = BaseEntityExtractor
RelationExtractor = BaseRelationExtractor

# Export all classes
__all__ = [
    "BaseEntityExtractor",
    "BaseRelationExtractor",
    "EntityExtractor",
    "RelationExtractor",
]
