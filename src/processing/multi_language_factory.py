"""Multi-language processor factory for code analysis across different programming languages.

This module provides a factory pattern implementation for creating language-specific
code processors. It bridges the gap between legacy processors (Python, JavaScript)
and the new multi-language architecture, supporting automatic language detection
and processor instantiation.

The factory maintains backward compatibility while integrating with the new
language registry system that supports additional languages like Java, Rust,
Go, C#, and C++.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.file_service import FileService

# Import existing processors for backward compatibility
from .code_document_processor import CodeDocumentProcessor
from .javascript_processor import LegacyJavaScriptProcessor as JavaScriptProcessor
from .languages.base import LanguageProcessor

# Import new multi-language components
from .languages.registry import language_registry
from .python_processor import CorePythonProcessor

# Import new language processors


logger = logging.getLogger(__name__)


class CodebaseProcessorSet:
    """A processor set that handles comprehensive codebase analysis using the multi-language factory."""

    def __init__(self, factory: MultiLanguageProcessorFactory):
        """Initialize the processor set with a multi-language factory.

        Args:
            factory: The multi-language processor factory
        """
        self.factory = factory

    async def analyze_codebase_async(
        self,
        codebase_id: str,
        codebase_path: str,
        include_tests: bool = False,
        exclude_patterns: list[str] | None = None,
        include_patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Analyze a codebase and create a comprehensive code graph.

        Args:
            codebase_id: Unique identifier for the codebase
            codebase_path: Path to the codebase directory
            include_tests: Whether to include test files in analysis
            exclude_patterns: Patterns to exclude from analysis
            include_patterns: Patterns to force include in analysis

        Returns:
            Dict containing analysis results
        """
        try:
            from pathlib import Path

            from db.facade import GraphDatabaseConnection
            from services.code_graph.multi_language_graph_service import (
                MultiLanguageGraphService,
            )

            db_connection = GraphDatabaseConnection(codebase_id)

            # Initialize the multi-language graph service
            service = MultiLanguageGraphService(db_connection)
            await service.initialize()

            # Analyze the codebase directory
            codebase_path_obj = Path(codebase_path)

            if not codebase_path_obj.exists():
                return {
                    "status": "failed",
                    "error": f"Codebase path does not exist: {codebase_path}",
                    "code_graph_id": codebase_id,
                    "nodes_created": 0,
                    "edges_created": 0,
                }

            # Use the multi-language graph service to analyze the directory
            result = await service.analyze_directory(codebase_path_obj)

            # Close the database connection
            await db_connection.close()

            return {
                "status": "success",
                "code_graph_id": codebase_id,
                "nodes_created": result.get("total_entities", 0),
                "edges_created": result.get("total_relationships", 0),
                "files_processed": result.get("files_processed", 0),
                "languages_detected": result.get("languages_detected", []),
                "session_id": result.get("session_id"),
            }

        except Exception as e:
            logger.error(f"Error during codebase analysis: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "code_graph_id": codebase_id,
                "nodes_created": 0,
                "edges_created": 0,
            }

    async def create_minimal_graph(
        self,
        codebase_id: str,
        codebase_path: str,
    ) -> dict[str, Any]:
        """Create a minimal graph with basic file structure information.

        Args:
            codebase_id: Unique identifier for the codebase
            codebase_path: Path to the codebase directory

        Returns:
            Dict containing minimal graph creation results
        """
        try:
            from pathlib import Path

            from db.entity_mapper import MultiLanguageEntityMapper
            from db.facade import GraphDatabaseConnection
            from models.core_models import CoreCodeEntity, EntityType

            db_connection = GraphDatabaseConnection(codebase_id)

            # Initialize the database connection
            await db_connection.connect()
            conn = db_connection.get_connection()
            entity_mapper = MultiLanguageEntityMapper(conn, codebase_id)

            codebase_path_obj = Path(codebase_path)

            if not codebase_path_obj.exists():
                await db_connection.close()
                return {
                    "status": "failed",
                    "error": f"Codebase path does not exist: {codebase_path}",
                    "code_graph_id": codebase_id,
                    "nodes_created": 0,
                    "edges_created": 0,
                }

            # Create minimal file entities for supported files
            nodes_created = 0
            supported_extensions = self.factory.get_supported_extensions()

            for file_path in codebase_path_obj.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    # Create a minimal file entity
                    file_entity = CoreCodeEntity(
                        id=f"file:{file_path.relative_to(codebase_path_obj)}",
                        name=file_path.name,
                        type=EntityType.FILE,
                        language="unknown",  # Will be detected if needed
                        file_path=str(file_path),
                        line_number=1,
                        end_line_number=None,
                        column_start=0,
                        column_end=0,
                        visibility="public",
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={
                            "file_size": file_path.stat().st_size,
                            "file_extension": file_path.suffix,
                            "minimal_graph": True,
                        },
                    )

                    # Store the entity
                    await entity_mapper.store_entity(file_entity)
                    nodes_created += 1

            # Commit the transaction
            await conn.commit()
            await db_connection.close()

            return {
                "status": "success",
                "code_graph_id": codebase_id,
                "nodes_created": nodes_created,
                "edges_created": 0,  # Minimal graph doesn't create relationships
                "files_processed": nodes_created,
            }

        except Exception as e:
            logger.error(f"Error during minimal graph creation: {e}", exc_info=True)
            if "db_connection" in locals():
                try:
                    await db_connection.close()
                except Exception:
                    pass
            return {
                "status": "failed",
                "error": str(e),
                "code_graph_id": codebase_id,
                "nodes_created": 0,
                "edges_created": 0,
            }


class MultiLanguageProcessorFactory:
    """Factory for creating both legacy and new multi-language code processors."""

    def __init__(self, file_service: FileService | None = None):
        """Initialize the multi-language processor factory.

        Args:
            file_service: File service for file operations. If *None*,
                a default :class:`FileService` instance is created.
        """
        self.file_service = file_service if file_service is not None else FileService()

        # Legacy processors for backward compatibility
        self._legacy_processors: dict[str, type] = {
            "python": CorePythonProcessor,
            "javascript": JavaScriptProcessor,
            "typescript": JavaScriptProcessor,
        }

        # Initialize language processors (this will auto-register them via decorators)
        self._initialize_language_processors()

    def create_processor_set(self) -> CodebaseProcessorSet:
        """Create a processor set for codebase analysis.

        Returns:
            CodebaseProcessorSet: A processor set that can handle full codebase analysis
        """
        return CodebaseProcessorSet(self)

    def _initialize_language_processors(self):
        """Initialize and register all language processors."""
        try:
            # Import all language processors to trigger registration
            # This ensures all @register_language decorators are executed
            from .languages.cpp.cpp_processor import CppProcessor  # noqa: F401
            from .languages.csharp.csharp_processor import CSharpProcessor  # noqa: F401
            from .languages.go.go_processor import GoProcessor  # noqa: F401
            from .languages.java.java_processor import JavaProcessor  # noqa: F401
            from .languages.javascript.javascript_processor import JSLanguageProcessor  # noqa: F401
            from .languages.python.python_processor import PyLanguageProcessor  # noqa: F401
            from .languages.rust.rust_processor import RustProcessor  # noqa: F401
            from .languages.typescript.typescript_processor import TSLanguageProcessor  # noqa: F401

            logger.info("Registered languages: %s", language_registry.get_supported_languages())
            logger.info("Supported extensions: %s", language_registry.get_supported_extensions())

        except ImportError as e:
            logger.warning("Some language processors could not be imported: %s", e)

    async def get_processor(
        self, language: str
    ) -> LanguageProcessor | CodeDocumentProcessor | None:
        """Get a processor for the specified language.

        Args:
            language: Programming language name

        Returns:
            Configured processor instance or None if not supported
        """
        language = language.lower()

        # First, try to get from new language registry
        if language_registry.is_language_supported(language):
            try:
                processor: (
                    LanguageProcessor | CodeDocumentProcessor | None
                ) = await language_registry.get_processor(language)  # type: ignore[invalid-argument-type]  # ty: ignore[invalid-argument-type]
                logger.debug(
                    "Created new %s processor: %s",
                    language,
                    processor.__class__.__name__,
                )
                return processor
            except Exception as e:
                # Catch all exceptions as processor creation can fail for various reasons
                import traceback

                logger.error("Failed to create new processor for %s: %s", language, e)
                logger.error("Traceback: %s", traceback.format_exc())

        # Fall back to legacy processors
        processor_class = self._legacy_processors.get(language)
        if processor_class:
            try:
                legacy_processor: LanguageProcessor | CodeDocumentProcessor = processor_class(
                    file_service=self.file_service
                )
                logger.debug(
                    "Created legacy %s processor: %s",
                    language,
                    legacy_processor.__class__.__name__,
                )
                return legacy_processor
            except Exception as e:
                # Catch all exceptions as processor creation can fail for various reasons
                logger.error("Failed to create legacy processor for %s: %s", language, e)

        logger.warning("No processor available for language: %s", language)
        return None

    async def get_processor_for_file(
        self, file_path: str | Path
    ) -> LanguageProcessor | CodeDocumentProcessor | None:
        """Get a processor based on file extension.

        Args:
            file_path: Path to the file

        Returns:
            Configured processor instance or None if not supported
        """
        file_path = Path(file_path)

        # First, try new language registry
        language = language_registry.detect_language(file_path)
        if language:
            return await self.get_processor(language)

        # Fall back to legacy extension mapping
        legacy_extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
        }

        ext = file_path.suffix.lower()
        language = legacy_extension_map.get(ext)
        if language:
            return await self.get_processor(language)

        logger.debug("No processor available for file extension: %s", ext)
        return None

    def get_supported_languages(self) -> list[str]:
        """Get list of all supported programming languages.

        Returns:
            List of supported language names
        """
        # Combine new and legacy languages
        new_languages = set(language_registry.get_supported_languages())
        legacy_languages = set(self._legacy_processors.keys())
        return list(new_languages | legacy_languages)

    def get_supported_extensions(self) -> list[str]:
        """Get list of all supported file extensions.

        Returns:
            List of supported file extensions
        """
        # Combine new and legacy extensions
        new_extensions = language_registry.get_supported_extensions()
        legacy_extensions = {".py", ".js", ".jsx", ".ts", ".tsx"}
        return list(new_extensions | legacy_extensions)

    def is_supported_file(self, file_path: str | Path) -> bool:
        """Check if a file is supported by any available processor.

        Args:
            file_path: Path to the file to check

        Returns:
            True if the file is supported, False otherwise
        """
        file_path = Path(file_path)

        # Check if the file has a supported extension
        ext = file_path.suffix.lower()
        supported_extensions = self.get_supported_extensions()

        # Also check if language registry can detect the language
        language = language_registry.detect_language(file_path)

        return ext in supported_extensions or language is not None

    def get_language_info(self, language: str) -> dict[str, Any] | None:
        """Get information about a specific language.

        Args:
            language: Programming language name

        Returns:
            Dictionary with language information or None if not supported
        """
        language = language.lower()

        # Check if language is supported in the new registry
        if language_registry.is_language_supported(language):
            try:
                config = language_registry.get_language_config(language)
                return {
                    "name": config.name,
                    "display_name": config.display_name,
                    "extensions": config.extensions,
                    "parser_type": config.parser_type.value,
                    "features": config.features.model_dump(),
                }
            except Exception as e:
                logger.error("Failed to get language info for %s: %s", language, e)

        # For legacy languages, provide basic info
        if language in self._legacy_processors:
            legacy_info = {
                "python": {
                    "name": "python",
                    "display_name": "Python",
                    "extensions": [".py"],
                    "parser_type": "ast",
                    "features": {
                        "syntax_parsing": True,
                        "semantic_analysis": True,
                        "type_inference": True,
                        "dependency_tracking": True,
                    },
                },
                "javascript": {
                    "name": "javascript",
                    "display_name": "JavaScript",
                    "extensions": [".js", ".jsx"],
                    "parser_type": "treesitter",
                    "features": {
                        "syntax_parsing": True,
                        "semantic_analysis": True,
                        "type_inference": False,
                        "dependency_tracking": True,
                    },
                },
                "typescript": {
                    "name": "typescript",
                    "display_name": "TypeScript",
                    "extensions": [".ts", ".tsx"],
                    "parser_type": "treesitter",
                    "features": {
                        "syntax_parsing": True,
                        "semantic_analysis": True,
                        "type_inference": True,
                        "dependency_tracking": True,
                    },
                },
            }
            return legacy_info.get(language)

        return None


# Global factory instance
_multi_language_factory: MultiLanguageProcessorFactory | None = None


async def get_multi_language_factory(
    file_service: FileService | None = None,
) -> MultiLanguageProcessorFactory:
    """Get the global multi-language processor factory instance.

    Args:
        file_service: File service for dependency injection

    Returns:
        MultiLanguageProcessorFactory instance
    """
    # Using module-level variable instead of global statement
    if _multi_language_factory is None:
        if file_service is None:
            # Create default dependencies if not provided
            file_service = FileService()

        globals()["_multi_language_factory"] = MultiLanguageProcessorFactory(file_service)
        logger.info("Created global multi-language processor factory")

    # ty doesn't understand globals() assignment, so we need to assert
    assert _multi_language_factory is not None
    return _multi_language_factory


# Convenience functions for easy integration
async def get_processor_for_file(
    file_path: str | Path,
) -> LanguageProcessor | CodeDocumentProcessor | None:
    """Get a processor for a specific file based on its extension.

    Args:
        file_path: Path to the file for which to get a processor

    Returns:
        Optional[Union[LanguageProcessor, CodeDocumentProcessor]]: Processor instance
            appropriate for the file's language, or None if unsupported
    """
    factory = await get_multi_language_factory()
    return await factory.get_processor_for_file(file_path)


async def get_processor_for_language(
    language: str,
) -> LanguageProcessor | CodeDocumentProcessor | None:
    """Get a processor for a specific programming language.

    Args:
        language: Name of the programming language (e.g., 'python', 'javascript')

    Returns:
        Optional[Union[LanguageProcessor, CodeDocumentProcessor]]: Processor instance
            for the specified language, or None if unsupported
    """
    factory = await get_multi_language_factory()
    return await factory.get_processor(language)


def get_supported_languages() -> list[str]:
    """Get all supported programming languages (synchronous version).

    This is a synchronous version that returns known supported languages
    without initializing any processors, making it safe to call during
    startup or in synchronous contexts.

    Returns:
        List[str]: List of all supported programming language names
    """
    # This is a sync version that doesn't initialize processors
    # It returns the known languages without creating processors
    new_languages = {
        "java",
        "rust",
        "go",
        "csharp",
        "cpp",
        "c",
    }  # Known new languages
    legacy_languages = {"python", "javascript", "typescript"}
    return list(new_languages | legacy_languages)


# Backward compatibility alias
MultiLanguageFactory = MultiLanguageProcessorFactory
