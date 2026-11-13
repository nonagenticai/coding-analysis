"""JavaScript and TypeScript code processor.

Extracts entities and relationships from JavaScript and TypeScript code.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.core.file_service import FileService
from processing.code_document_processor import CodeDocumentProcessor

from .javascript.javascript_entity_extractor import JavaScriptEntityExtractor
from .javascript.javascript_parser import JavaScriptParser
from .javascript.javascript_relation_extractor import (
    JavaScriptRelationExtractor,
)

logger = logging.getLogger(__name__)


class LegacyJavaScriptProcessor(CodeDocumentProcessor):
    """Legacy processor for JavaScript and TypeScript code documents using ESLint parser."""

    def __init__(self, file_service: FileService):
        """Initialize the JavaScript processor with proper dependency injection.

        Args:
            file_service: Service for file system operations

        Raises:
            ValueError: If file_service is None
        """
        super().__init__(file_service)
        # Ensure file_service is available (it should never be None for this processor)
        if self.file_service is None:
            raise ValueError("FileService is required for LegacyJavaScriptProcessor")
        self.language = "javascript"

        # Initialize specialized components
        self.parser = JavaScriptParser()
        self.entity_extractor = JavaScriptEntityExtractor()
        self.relation_extractor = JavaScriptRelationExtractor()

    async def process_document(self, file_path: str, code: str | None = None) -> dict[str, Any]:
        """Process a JavaScript/TypeScript document.

        Extracts entities and relationships from the document.

        Args:
            file_path: Path to the file to process
            code: Optional code content if already loaded

        Returns:
            Dictionary with extracted entities and relationships
        """
        try:
            if code is None:
                if self.file_service is None:
                    raise ValueError("FileService is required to read file content")
                code = self.file_service.read_file(file_path)

            # Determine language based on extension
            self.language = self._determine_language(file_path)

            # Parse the code using the specialized parser
            ast = await self.parser.parse_code(
                code,
                file_path,
                {"is_typescript": self.language == "typescript"},
            )

            if not ast:
                logger.error("Failed to parse %s", file_path)
                return self._create_error_result(file_path, "Failed to parse file")

            # Extract entities and relationships using specialized extractors
            module_name = self._get_module_name(file_path)

            entities = await self.entity_extractor.extract_entities(ast, file_path, module_name)
            relations = await self.relation_extractor.extract_all_relations(
                ast, file_path, module_name
            )

            return {
                "file_path": file_path,
                "language": self.language,
                "entities": entities,
                # Use 'relationships' key for consistency with PythonProcessor
                "relationships": relations,
            }

        except (ValueError, OSError, RuntimeError) as e:
            logger.error(
                "Error processing %s file %s: %s",
                self.language,
                file_path,
                e,
                exc_info=True,
            )
            return self._create_error_result(file_path, str(e))

    def _determine_language(self, file_path: str) -> str:
        """Determine if this is JavaScript or TypeScript based on file extension.

        Args:
            file_path: Path to the file being analyzed

        Returns:
            str: Either 'javascript' or 'typescript' based on file extension
        """
        return "typescript" if file_path.lower().endswith((".ts", ".tsx")) else "javascript"

    def _create_error_result(self, file_path: str, error_message: str) -> dict[str, Any]:
        """Create a standardized error result.

        Args:
            file_path: Path to the file that caused the error
            error_message: Description of the error that occurred

        Returns:
            Dict[str, Any]: Error result containing file path, language, error message,
                          and empty entities/relationships lists
        """
        return {
            "file_path": file_path,
            "language": self.language,
            "error": error_message,
            "entities": [],
            "relationships": [],  # Use 'relationships' key for consistency
        }

    async def analyze_directory(
        self, directory_path: str, recursive: bool = True
    ) -> dict[str, Any]:
        """Analyze all JavaScript/TypeScript files in a directory.

        Args:
            directory_path: Path to the directory to analyze
            recursive: Whether to recursively analyze subdirectories

        Returns:
            Dictionary containing combined entities and relationships
        """
        if not os.path.isdir(directory_path):
            logger.warning("Invalid directory path: %s", directory_path)
            return {"entities": [], "relations": []}

        all_entities: list[Any] = []
        all_relations: list[Any] = []
        processed_files = set()

        # Walk through the directory
        for root, _, files in os.walk(directory_path):
            # Skip if not recursive and not in the root directory
            if not recursive and root != directory_path:
                continue

            # Process JavaScript/TypeScript files
            for file in files:
                if file.lower().endswith((".js", ".jsx", ".ts", ".tsx")):
                    file_path = os.path.join(root, file)

                    # Skip if already processed
                    if file_path in processed_files:
                        continue

                    processed_files.add(file_path)

                    # Analyze the file
                    result = await self.process_document(file_path)

                    # Get entities and convert Pydantic models to dictionaries if needed
                    entities = result.get("entities", [])
                    if entities and hasattr(entities[0], "dict"):
                        entities = [e.model_dump() for e in entities]
                    all_entities.extend(entities)

                    # Get relationships and convert Pydantic models to dicts if needed
                    relationships = result.get("relationships", [])
                    if relationships and hasattr(relationships[0], "dict"):
                        relationships = [r.model_dump() for r in relationships]
                    all_relations.extend(relationships)

        return {
            "entities": all_entities,
            "relationships": all_relations,  # Use 'relationships' key for consistency
            "directory": directory_path,
        }

    def get_supported_extensions(self) -> list[str]:
        """Get list of file extensions supported by this processor.

        Returns:
            List[str]: List of supported file extensions including .js, .jsx, .ts, .tsx
        """
        return [".js", ".jsx", ".ts", ".tsx"]
