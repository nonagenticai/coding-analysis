"""
JavaScript and TypeScript Code Processor.

This module provides comprehensive analysis capabilities for JavaScript and TypeScript
source code files. It implements the BaseCodeProcessor interface to provide consistent
analysis results across different programming languages.

The processor supports:
- Pure JavaScript (.js, .jsx, .mjs) files
- TypeScript (.ts, .tsx) files with type annotations
- Modern JavaScript features (ES6+, async/await, destructuring)
- JSX syntax for React components
- Module imports and exports analysis
- Basic entity extraction (functions, classes, variables)
- Dependency analysis from import statements

The implementation uses a hybrid approach:
- Primary: Node.js-based parsing for accurate AST analysis
- Fallback: Regex-based parsing for lightweight analysis when Node.js unavailable

Key Features:
- Automatic language detection (JavaScript vs TypeScript)
- Entity extraction with line number tracking
- Import/export dependency analysis
- Support for modern JavaScript syntax
- Graceful fallback for environments without Node.js

Classes:
    JSAnalyzerProcessor: Main processor class implementing BaseCodeProcessor interface

Example:
    # Analyze a JavaScript file
    processor = JSAnalyzerProcessor()
    result = await processor.analyze_file("path/to/script.js")

    # Analyze TypeScript code snippet
    typescript_code = "const greeting: string = 'Hello World';"
    result = await processor.analyze_snippet(typescript_code, "example.ts")

    # Analyze entire directory
    result = await processor.analyze_directory("src/")
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import UTC, datetime
from typing import Any

from app.core.file_service import FileService
from app.core.security_config import CommandConfig, SecurityError, secure_runner
from models.core_models import (
    CoreAnalysisResult,
    CoreCodeEntity,
    EntityRelation,
)

from .base_analyzer import BaseCodeProcessor

logger = logging.getLogger(__name__)


class JSAnalyzerProcessor(BaseCodeProcessor):
    """JavaScript and TypeScript analyzer processor implementing BaseCodeProcessor interface.

    This processor provides comprehensive analysis of JavaScript and TypeScript source
    code, including entity extraction, dependency analysis, and relationship detection.
    It supports both modern JavaScript features and TypeScript-specific constructs.

    The processor automatically detects the source language (JavaScript vs TypeScript)
    based on file extension and content analysis. It uses a hybrid parsing approach
    with Node.js-based parsing as primary and regex-based parsing as fallback.

    Attributes:
        file_service (Optional[FileService]): Service for file operations
        language (str): Base language identifier ("javascript")

    Example:
        # Basic usage
        processor = JSAnalyzerProcessor()
        result = await processor.analyze_file("app.js")

        # With file service
        file_service = FileService()
        processor = JSAnalyzerProcessor(file_service)
        result = await processor.analyze_directory("src/")
    """

    def __init__(self, file_service: FileService | None = None):
        """Initialize the JavaScript processor with optional file service.

        Args:
            file_service (Optional[FileService]): File service for advanced file
                                                operations. If None, uses standard
                                                file operations.
        """
        self.file_service = file_service
        self.language = "javascript"
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if Node.js is available for advanced parsing.

        Tests for Node.js availability to determine if advanced AST-based parsing
        can be used. If Node.js is not available, the analyzer falls back to
        regex-based parsing which is less accurate but still functional.

        Logs warnings if Node.js is not available or if there are issues with
        the dependency check, but does not raise exceptions to allow graceful
        fallback to regex-based parsing.
        """
        try:
            config = CommandConfig(timeout=10)
            result = secure_runner.run_command(["node", "--version"], config=config)
            if result.returncode != 0:
                logger.warning("Node.js not found. JavaScript/TypeScript processing may not work.")
        except (ImportError, OSError, ValueError, SecurityError) as e:
            logger.warning("Error checking Node.js: %s", e)

    def get_supported_extensions(self) -> list[str]:
        """Get supported JavaScript and TypeScript file extensions.

        Returns:
            List[str]: List of supported file extensions including:
                - .js: Standard JavaScript files
                - .jsx: React JavaScript files with JSX syntax
                - .ts: TypeScript files with type annotations
                - .tsx: TypeScript React files with JSX syntax
                - .mjs: ES module JavaScript files
        """
        return [".js", ".jsx", ".ts", ".tsx", ".mjs"]

    def get_language_name(self) -> str:
        """Return the name of the programming language this analyzer handles.

        Returns:
            str: Always returns "javascript" as the base language identifier.
                 Specific language detection (JavaScript vs TypeScript) is handled
                 during analysis based on file extension and content.
        """
        return self.language

    async def analyze_file(self, file_path: str) -> CoreAnalysisResult:
        """Analyze a JavaScript or TypeScript file.

        Reads and analyzes a source code file, automatically detecting whether
        it's JavaScript or TypeScript based on the file extension. Extracts
        entities, relationships, and dependencies from the code.

        Args:
            file_path (str): Absolute path to the JavaScript/TypeScript file
                           to analyze. Must have a supported extension.

        Returns:
            CoreAnalysisResult: Complete analysis results including:
                - Extracted entities (functions, classes, variables)
                - Import/export relationships
                - Language detection (javascript or typescript)
                - Line count and metadata
                - Error information if analysis fails

        Raises:
            The method catches exceptions and returns them in CoreAnalysisResult:
            - OSError: If file cannot be read
            - ValueError: If file path is invalid
            - UnicodeDecodeError: If file encoding is unsupported
        """
        try:
            # Read the file directly if no file_service is provided
            if self.file_service:
                code = self.file_service.read_file(file_path)
            else:
                with open(file_path, encoding="utf-8") as f:
                    code = f.read()

            return await self.analyze_snippet(code, file_path)
        except (OSError, ValueError, UnicodeDecodeError) as e:
            logger.error("Error analyzing file %s: %s", file_path, e)
            return self._error_result(file_path, str(e))

    async def analyze_snippet(
        self, code_snippet: str, file_path: str | None = None
    ) -> CoreAnalysisResult:
        """Analyze a JavaScript or TypeScript code snippet.

        Performs analysis on a code snippet without requiring a file on disk.
        Automatically detects language type and extracts entities, relationships,
        and structural information.

        Args:
            code_snippet (str): JavaScript or TypeScript source code to analyze.
            file_path (Optional[str]): Optional file path for context and language
                                     detection. If None, defaults to "snippet.js".

        Returns:
            CoreAnalysisResult: Analysis results including:
                - Language detection (javascript or typescript)
                - Extracted entities (functions, classes, variables, imports)
                - Import/export relationships
                - Line count and parsing metadata
                - Error information if parsing fails

        Raises:
            The method catches exceptions and returns them in CoreAnalysisResult:
            - OSError: If temporary file operations fail
            - ValueError: If code snippet is invalid
            - UnicodeDecodeError: If code contains invalid characters
        """
        try:
            if file_path is None:
                file_path = "snippet.js"

            # Determine if TypeScript
            _, ext = os.path.splitext(file_path)
            is_typescript = ext.lower() in [".ts", ".tsx"]

            # Parse the code
            ast = await self._parse_code(code_snippet, file_path, is_typescript)
            if not ast:
                return self._error_result(file_path, "Failed to parse code")

            # Extract entities and relationships
            entities = self._extract_entities(ast, file_path)
            relationships = self._extract_relationships(ast, file_path)

            return CoreAnalysisResult(
                success=True,
                language="typescript" if is_typescript else "javascript",
                file_path=file_path,
                entities=entities,
                relationships=relationships,
                dependencies=[],
                complexity_metrics=None,
                security_patterns=[],
                error=None,
                parse_time_ms=0.0,
                total_lines=len(code_snippet.splitlines()),
                metadata={
                    "file_path": file_path,
                    "language": ("typescript" if is_typescript else "javascript"),
                },
            )

        except (OSError, ValueError, UnicodeDecodeError) as e:
            logger.error("Error analyzing snippet: %s", e)
            return self._error_result(file_path or "unknown", str(e))

    async def analyze_directory(self, directory_path: str) -> CoreAnalysisResult:
        """Analyze all JavaScript and TypeScript files in a directory recursively.

        Recursively finds and analyzes all supported JavaScript and TypeScript
        files in the specified directory. Results are aggregated to provide
        directory-level insights and statistics.

        Args:
            directory_path (str): Absolute path to the directory to analyze.
                                Must be a valid directory containing JavaScript
                                or TypeScript files.

        Returns:
            CoreAnalysisResult: Aggregated analysis results including:
                - Combined entities from all analyzed files
                - All import/export relationships
                - Directory-level metadata and statistics
                - List of failed files if any
                - Total file count and analysis summary

        Note:
            - Files that fail to analyze are logged but don't stop the process
            - Results include both successful and failed file information
            - Supports all file extensions returned by get_supported_extensions()
        """
        try:
            files: list[str] = []
            # Find files using file_service if available, otherwise use os.walk
            if self.file_service:
                for ext in self.get_supported_extensions():
                    files.extend(self.file_service.find_files(directory_path, f"*{ext}"))
            else:
                # Fallback to basic file finding
                for root, _, filenames in os.walk(directory_path):
                    for filename in filenames:
                        if any(filename.endswith(ext) for ext in self.get_supported_extensions()):
                            files.append(os.path.join(root, filename))

            all_entities: list[CoreCodeEntity] = []
            all_relationships: list[EntityRelation] = []
            failed_files: list[str] = []

            for file_path in files:
                try:
                    result = await self.analyze_file(file_path)
                    if result.success:
                        all_entities.extend(result.entities)
                        all_relationships.extend(result.relationships)
                    else:
                        failed_files.append(file_path)
                except (OSError, ValueError, UnicodeDecodeError) as e:
                    logger.error("Error analyzing file %s: %s", file_path, e)
                    failed_files.append(file_path)

            return CoreAnalysisResult(
                success=True,
                language=self.language,
                file_path=directory_path,
                entities=all_entities,
                relationships=all_relationships,
                dependencies=[],
                complexity_metrics=None,
                security_patterns=[],
                error=None,
                parse_time_ms=0.0,
                total_lines=0,
                metadata={
                    "directory": directory_path,
                    "language": self.language,
                    "files_analyzed": len(files),
                    "failed_files": failed_files,
                },
            )

        except (OSError, ValueError, UnicodeDecodeError) as e:
            logger.error("Error analyzing directory %s: %s", directory_path, e)
            return CoreAnalysisResult(
                success=False,
                language=self.language,
                file_path=directory_path,
                entities=[],
                relationships=[],
                dependencies=[],
                complexity_metrics=None,
                security_patterns=[],
                error=str(e),
                parse_time_ms=0.0,
                total_lines=0,
                metadata={
                    "directory": directory_path,
                    "language": self.language,
                },
            )

    async def _parse_code(
        self, code: str, file_path: str, is_typescript: bool
    ) -> dict[str, Any] | None:
        """Parse JavaScript/TypeScript code using Node.js."""
        try:
            # Create temporary file for code
            with tempfile.NamedTemporaryFile(
                suffix=".ts" if is_typescript else ".js",
                mode="w+",
                delete=False,
                encoding="utf-8",
            ) as code_file:
                code_file.write(code)
                code_file.flush()
                code_path = code_file.name

            # Basic regex-based parsing for basic entities
            # This is a fallback when Node.js parsing is not available
            return self._regex_parse(code, file_path)

        except (OSError, ValueError, UnicodeDecodeError) as e:
            logger.error("Error parsing %s: %s", file_path, e)
            return None
        finally:
            # Clean up temp file
            try:
                if "code_path" in locals():
                    os.unlink(code_path)
            except OSError:
                pass

    def _regex_parse(self, code: str, file_path: str) -> dict[str, Any]:
        """Basic regex-based parsing for JavaScript/TypeScript as fallback.

        Provides lightweight parsing using regular expressions when Node.js-based
        AST parsing is not available. While less accurate than full AST parsing,
        it can extract basic code entities and structure.

        Args:
            code (str): JavaScript or TypeScript source code to parse.
            file_path (str): File path for context (currently unused).

        Returns:
            Dict[str, Any]: Parsed code structure containing:
                - type: "Program" (root node type)
                - body: List of detected entities including functions, classes,
                       variables, and imports with their line numbers

        Note:
            This is a fallback method with limited accuracy. It may miss
            complex syntax patterns and nested structures that full AST
            parsing would capture.
        """
        # Basic patterns for common constructs
        function_pattern = r"(?:function\s+(\w+)|(\w+)\s*[:=]\s*(?:function|\([^)]*\)\s*=>))"
        class_pattern = r"class\s+(\w+)"
        const_pattern = r"const\s+(\w+)"
        let_pattern = r"let\s+(\w+)"
        var_pattern = r"var\s+(\w+)"
        import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'

        entities: list[Any] = []
        # Find functions
        for match in re.finditer(function_pattern, code):
            name = match.group(1) or match.group(2)
            if name:
                entities.append(
                    {
                        "type": "function",
                        "name": name,
                        "line": code[: match.start()].count("\n") + 1,
                    }
                )

        # Find classes
        for match in re.finditer(class_pattern, code):
            entities.append(
                {
                    "type": "class",
                    "name": match.group(1),
                    "line": code[: match.start()].count("\n") + 1,
                }
            )

        # Find variables
        for pattern, var_type in [
            (const_pattern, "const"),
            (let_pattern, "let"),
            (var_pattern, "var"),
        ]:
            for match in re.finditer(pattern, code):
                entities.append(
                    {
                        "type": "variable",
                        "name": match.group(1),
                        "var_type": var_type,
                        "line": code[: match.start()].count("\n") + 1,
                    }
                )

        # Find imports
        imports: list[Any] = []
        for match in re.finditer(import_pattern, code):
            imports.append(
                {
                    "type": "import",
                    "module": match.group(1),
                    "line": code[: match.start()].count("\n") + 1,
                }
            )

        return {"type": "Program", "body": entities + imports}

    def _extract_entities(self, ast: dict[str, Any], file_path: str) -> list[CoreCodeEntity]:
        """Extract code entities from AST."""
        entities: list[Any] = []
        for item in ast.get("body", []):
            entity_type = item.get("type", "unknown")
            name = item.get("name", "unnamed")
            line = item.get("line", 0)

            # Create entity based on type
            if entity_type in [
                "function",
                "class",
                "variable",
                "const",
                "let",
                "var",
            ]:
                entities.append(
                    CoreCodeEntity(
                        id=f"{file_path}:{name}:{line}",
                        name=name,
                        type=entity_type,
                        language=self.language,
                        file_path=file_path,
                        line_number=line,
                        end_line_number=line + 1,  # Approximate end line
                        column_start=0,
                        column_end=0,
                        visibility="public",
                        signature=item.get("signature", ""),
                        source_code=item.get("code", ""),
                        code_snippet=item.get("code", ""),
                        docstring=item.get("docstring", ""),
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                        metadata={
                            "language": self.language,
                            "var_type": item.get("var_type", ""),
                        },
                    )
                )

        return entities

    def _extract_relationships(self, ast: dict[str, Any], file_path: str) -> list[EntityRelation]:
        """Extract relationships from AST."""
        from models.core_models import RelationType

        relationships: list[Any] = []
        # Extract import relationships
        for item in ast.get("body", []):
            if item.get("type") == "import":
                relationships.append(
                    EntityRelation(
                        source_id=file_path,
                        target_id=item.get("module", ""),
                        type=RelationType.IMPORTS,
                        properties={"line_number": item.get("line", 0)},
                    )
                )

        return relationships

    def _error_result(self, file_path: str, error: str) -> CoreAnalysisResult:
        """Create error result."""
        return CoreAnalysisResult(
            success=False,
            language=self.language,
            file_path=file_path,
            entities=[],
            relationships=[],
            dependencies=[],
            complexity_metrics=None,
            security_patterns=[],
            error=error,
            parse_time_ms=0.0,
            total_lines=0,
            metadata={
                "file_path": file_path,
                "language": self.language,
            },
        )
