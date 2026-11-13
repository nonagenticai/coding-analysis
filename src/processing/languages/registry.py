"""Registry for managing language configurations and processors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from .base import LanguageConfig, LanguageProcessor

SupportedLanguage = Literal[
    "python", "javascript", "typescript", "java", "go", "rust", "cpp", "csharp"
]

logger = logging.getLogger(__name__)


class LanguageRegistry:
    """Central registry for all supported programming languages."""

    def __init__(self) -> None:
        self._processors: dict[str, type[LanguageProcessor]] = {}
        self._configs: dict[str, LanguageConfig] = {}
        self._extension_map: dict[str, str] = {}  # .py -> python
        self._initialized_processors: dict[str, LanguageProcessor] = {}

    def register_language(
        self, config: LanguageConfig, processor_class: type[LanguageProcessor]
    ) -> None:
        """Register a new language and its processor.

        Args:
            config (LanguageConfig): The config parameter.
            processor_class (Type[LanguageProcessor]): The processor_class parameter.

        Returns:
            None: The none result.

        Raises:
            ValueError: Raised under certain conditions.

        Example:
            >>> register_language("config_value", "processor_class_value")"""
        language_name = config.name

        # Validate processor class
        if not issubclass(processor_class, LanguageProcessor):
            raise ValueError(f"Processor for {language_name} must inherit from LanguageProcessor")

        # Register language
        self._configs[language_name] = config
        self._processors[language_name] = processor_class

        # Map file extensions
        for ext in config.extensions:
            if ext in self._extension_map:
                logger.warning(
                    f"Extension {ext} already mapped to {self._extension_map[ext]}, overriding with {language_name}"
                )
            self._extension_map[ext] = language_name

        logger.info(f"Registered language: {config.display_name} ({language_name})")

    async def get_processor(self, language: SupportedLanguage) -> LanguageProcessor:
        """Get an initialized processor for a language.

        Args:
            language (SupportedLanguage): The language parameter.

        Returns:
            LanguageProcessor: The languageprocessor result.

        Raises:
            ValueError: Raised under certain conditions.

        Example:
            >>> await get_processor("language_value")"""
        if language not in self._processors:
            raise ValueError(f"Language '{language}' not registered")

        # Return cached processor if available
        if language in self._initialized_processors:
            return self._initialized_processors[language]

        # Create and initialize new processor
        processor_class = self._processors[language]
        config = self._configs[language]
        processor = processor_class(config)

        await processor.initialize()
        self._initialized_processors[language] = processor

        return processor

    def detect_language(self, file_path: Path) -> str | None:
        """Detect language from file extension.

        Args:
            file_path (Path): The file_path parameter.

        Returns:
            Optional[str]: The optional[str] result."""
        suffix = file_path.suffix.lower()
        return self._extension_map.get(suffix)

    def get_supported_languages(self) -> list[str]:
        """Get list of all supported languages.

        Returns:
            List[str]: The list[str] result."""
        return list(self._configs.keys())

    def get_supported_extensions(self) -> set[str]:
        """Get all supported file extensions.

        Returns:
            Set[str]: The set[str] result."""
        return set(self._extension_map.keys())

    def get_language_config(self, language: str) -> LanguageConfig:
        """Get configuration for a specific language.

        Args:
            language (str): The language parameter.

        Returns:
            LanguageConfig: The languageconfig result.

        Raises:
            ValueError: Raised under certain conditions."""
        if language not in self._configs:
            raise ValueError(f"Language '{language}' not registered")
        return self._configs[language]

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported.

        Args:
            language (str): The language parameter.

        Returns:
            bool: The bool result."""
        return language in self._configs

    def is_file_supported(self, file_path: Path) -> bool:
        """Check if a file can be processed based on its extension.

        Args:
            file_path (Path): The file_path parameter.

        Returns:
            bool: The bool result."""
        return self.detect_language(file_path) is not None


# Global registry instance
language_registry = LanguageRegistry()


def register_language(
    config: LanguageConfig,
    processor_class: type[LanguageProcessor] | None = None,
):
    """Decorator for registering language processors.

    Args:
        config (LanguageConfig): The config parameter.
        processor_class (Optional[Type[LanguageProcessor]], optional): The processor_class parameter. Defaults to None.

    Returns:
        Any: The any result."""

    def decorator(cls):
        """Function decorator.

        Returns:
            Any: The any result."""
        language_registry.register_language(config, cls)
        return cls

    return decorator(processor_class) if processor_class else decorator
