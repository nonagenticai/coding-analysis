"""Python code analysis processor.

This module provides the CorePythonProcessor class for analyzing Python source code
files and extracting entities, relationships, and other structural information.
"""

from __future__ import annotations

import ast
import logging
import os
from typing import Any

from app.core.file_service import FileService

from .code_document_processor import CodeDocumentProcessor
from .python.python_entity_extractor import PythonEntityExtractor
from .python.python_parser import PythonParser
from .python.python_relation_extractor import PythonRelationExtractor

logger = logging.getLogger(__name__)


class CorePythonProcessor(CodeDocumentProcessor):
    """Processor for Python code analysis.

    This class handles parsing and analyzing Python files to extract
    entities, relationships, and other code structure information.
    """

    def __init__(self, file_service: FileService):
        """Initialize the Python processor with proper dependency injection.

        Args:
            file_service: Service for file system operations

        Raises:
            ValueError: If file_service is None
        """
        super().__init__(file_service)
        # Ensure file_service is available (it should never be None for this processor)
        if self.file_service is None:
            raise ValueError("FileService is required for CorePythonProcessor")
        self.language = "python"
        self.processed_files: set[str] = set()

        # Initialize specialized components
        self.parser = PythonParser()
        self.entity_extractor = PythonEntityExtractor()
        self.relation_extractor = PythonRelationExtractor()

    async def process_document(self, file_path: str, code: str | None = None) -> dict[str, Any]:
        """Process a Python file and extract code structure information.

        Args:
            file_path: Path to the Python file to analyze
            code: Optional code content if already loaded

        Returns:
            Dictionary containing extracted entities and relationships
        """
        if not os.path.exists(file_path) or not file_path.endswith(".py"):
            logger.warning("Invalid Python file path: %s", file_path)
            return {"entities": [], "relationships": []}

        try:
            if code is None:
                if self.file_service is None:
                    raise ValueError("FileService is required to read file content")
                code = self.file_service.read_file(file_path)

            # Parse the AST
            ast_tree = await self.parser.parse_code(code, file_path)

            if not ast_tree:
                logger.error("Failed to parse %s", file_path)
                return {"entities": [], "relationships": []}

            # Extract entities and relationships using specialized extractors
            module_name = self._get_module_name(file_path)

            entities = await self.entity_extractor.extract_entities(
                ast_tree, file_path, module_name
            )
            relationships = await self.relation_extractor.extract_all_relations(
                ast_tree, file_path, module_name
            )

            # Mark file as processed
            self.processed_files.add(file_path)

            return {
                "file_path": file_path,
                "language": self.language,
                "entities": entities,
                "relationships": relationships,
            }

        except (OSError, ValueError, AttributeError, TypeError) as e:
            logger.error(
                "Error analyzing Python file %s: %s",
                file_path,
                str(e),
                exc_info=True,
            )
            return {
                "file_path": file_path,
                "language": self.language,
                "error": str(e),
                "entities": [],
                "relationships": [],
            }
        except Exception as e:
            logger.error(
                "Unexpected error analyzing Python file %s: %s",
                file_path,
                str(e),
                exc_info=True,
            )
            return {
                "file_path": file_path,
                "language": self.language,
                "error": str(e),
                "entities": [],
                "relationships": [],
            }

    async def analyze_directory(
        self, directory_path: str, recursive: bool = True
    ) -> dict[str, Any]:
        """Analyze all Python files in a directory.

        Args:
            directory_path: Path to the directory to analyze
            recursive: Whether to recursively analyze subdirectories

        Returns:
            Dictionary containing combined entities and relationships
        """
        if not os.path.isdir(directory_path):
            logger.warning("Invalid directory path: %s", directory_path)
            return {"entities": [], "relationships": []}

        all_entities: list[Any] = []
        all_relationships: list[Any] = []
        # Walk through the directory
        for root, _, files in os.walk(directory_path):
            # Skip if not recursive and not in the root directory
            if not recursive and root != directory_path:
                continue

            # Process Python files
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)

                    # Skip if already processed
                    if file_path in self.processed_files:
                        continue

                    # Analyze the file
                    result = await self.process_document(file_path)

                    all_entities.extend(result.get("entities", []))
                    all_relationships.extend(result.get("relationships", []))

        return {
            "entities": all_entities,
            "relationships": all_relationships,
            "directory": directory_path,
        }

    async def _extract_basic_relationships(
        self, ast_tree: ast.AST, file_path: str
    ) -> list[dict[str, Any]]:
        """Extract basic relationships between Python entities from AST.

        Args:
            ast_tree: Parsed AST of the Python file
            file_path: Path to the Python file being analyzed

        Returns:
            List of dictionaries containing:
            - source: Source entity ID
            - target: Target entity ID
            - type: Relationship type (inheritance, composition, etc.)
            - properties: Additional relationship metadata
        """
        relationships: list[dict[str, Any]] = []

        for node in ast.walk(ast_tree):
            if not isinstance(node, ast.ClassDef):
                continue

            class_name = node.name

            # --- Inheritance ---
            for base in node.bases:
                target: str | None = None
                if isinstance(base, ast.Name):
                    target = base.id
                elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                    target = f"{base.value.id}.{base.attr}"
                if target is not None:
                    relationships.append(
                        {
                            "source": class_name,
                            "target": target,
                            "type": "inheritance",
                            "properties": {"file": file_path, "line": node.lineno},
                        }
                    )

            # --- Composition and Uses ---
            # Track unique uses targets per class to avoid duplicates
            seen_uses: set[str] = set()

            for item in node.body:
                # Composition via ast.Assign: e.g. self.x = SomeClass()
                if isinstance(item, ast.Assign):
                    if isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Name):
                        func_name = item.value.func.id
                        # Determine the attribute name from the first target
                        attr_name: str | None = None
                        if item.targets and isinstance(item.targets[0], ast.Attribute):
                            attr_name = item.targets[0].attr
                        elif item.targets and isinstance(item.targets[0], ast.Name):
                            attr_name = item.targets[0].id
                        relationships.append(
                            {
                                "source": class_name,
                                "target": func_name,
                                "type": "composition",
                                "properties": {
                                    "file": file_path,
                                    "line": item.lineno,
                                    "attribute": attr_name,
                                },
                            }
                        )

                # Composition via ast.AnnAssign: e.g. self.x: Type = SomeClass()
                if isinstance(item, ast.AnnAssign):
                    if (
                        item.value is not None
                        and isinstance(item.value, ast.Call)
                        and isinstance(item.value.func, ast.Name)
                    ):
                        func_name = item.value.func.id
                        attr_name = None
                        if isinstance(item.target, ast.Attribute):
                            attr_name = item.target.attr
                        elif isinstance(item.target, ast.Name):
                            attr_name = item.target.id
                        relationships.append(
                            {
                                "source": class_name,
                                "target": func_name,
                                "type": "composition",
                                "properties": {
                                    "file": file_path,
                                    "line": item.lineno,
                                    "attribute": attr_name,
                                },
                            }
                        )

                # Uses (attribute access) inside class methods
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for sub_node in ast.walk(item):
                        if isinstance(sub_node, ast.Attribute):
                            # Skip self.x references
                            if isinstance(sub_node.value, ast.Name) and sub_node.value.id == "self":
                                continue
                            # Resolve the type reference
                            attr_type: str | None = None
                            if isinstance(sub_node.value, ast.Name):
                                attr_type = sub_node.value.id
                            elif isinstance(sub_node.value, ast.Attribute):
                                # Nested attribute: try to resolve the chain
                                parts: list[str] = [sub_node.attr]
                                current = sub_node.value
                                while isinstance(current, ast.Attribute):
                                    parts.append(current.attr)
                                    current = current.value
                                if isinstance(current, ast.Name) and current.id != "self":
                                    parts.append(current.id)
                                    attr_type = ".".join(reversed(parts[1:]))
                                else:
                                    attr_type = None
                            if attr_type is not None and attr_type not in seen_uses:
                                seen_uses.add(attr_type)
                                relationships.append(
                                    {
                                        "source": class_name,
                                        "target": attr_type,
                                        "type": "uses",
                                        "properties": {"file": file_path, "line": sub_node.lineno},
                                    }
                                )

        return relationships

    async def _extract_function_calls(
        self, ast_tree: ast.AST, file_path: str
    ) -> list[dict[str, Any]]:
        """Extract function call relationships from Python AST.

        Args:
            ast_tree: Parsed AST of the Python file
            file_path: Path to the Python file being analyzed

        Returns:
            List of dictionaries containing:
            - caller: Calling function/method ID
            - callee: Called function ID
            - location: Source location of call
            - properties: Additional call metadata
        """
        calls: list[dict[str, Any]] = []

        # Annotate every node with a _parent reference
        for node in ast.walk(ast_tree):
            for child in ast.iter_child_nodes(node):
                child._parent = node

        def _resolve_value(node: ast.expr) -> str:
            """Resolve an AST expression to a dotted name string."""
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                parent_str = _resolve_value(node.value)
                return f"{parent_str}.{node.attr}"
            return "<unknown>"

        def _find_enclosing_function(node: ast.AST) -> str:
            """Walk up _parent chain to find the enclosing function/method."""
            current = node
            enclosing_func: str | None = None
            enclosing_class: str | None = None
            while hasattr(current, "_parent"):
                current = current._parent
                if (
                    isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and enclosing_func is None
                ):
                    enclosing_func = current.name
                elif isinstance(current, ast.ClassDef) and enclosing_class is None:
                    enclosing_class = current.name
            if enclosing_func is not None:
                if enclosing_class is not None:
                    return f"{enclosing_class}.{enclosing_func}"
                return enclosing_func
            return "<module>"

        for node in ast.walk(ast_tree):
            if not isinstance(node, ast.Call):
                continue

            # Determine the callee
            callee_id: str
            if isinstance(node.func, ast.Name):
                callee_id = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee_id = _resolve_value(node.func)
            else:
                callee_id = "<unknown>"

            caller_id = _find_enclosing_function(node)

            calls.append(
                {
                    "caller": caller_id,
                    "callee": callee_id,
                    "location": {
                        "file": file_path,
                        "line": node.lineno,
                        "col": node.col_offset,
                    },
                    "properties": {
                        "arg_count": len(node.args) + len(node.keywords),
                    },
                }
            )

        return calls

    async def _extract_imports(self, ast_tree: ast.AST, file_path: str) -> list[dict[str, Any]]:
        """Extract import relationships from Python AST.

        Args:
            ast_tree: Parsed AST of the Python file
            file_path: Path to the Python file being analyzed

        Returns:
            List of dictionaries containing:
            - source: Importing module path
            - target: Imported module/entity
            - type: Import type (absolute, relative, etc.)
            - properties: Additional import metadata
        """
        imports: list[dict[str, Any]] = []

        for node in ast.walk(ast_tree):
            # Handle: import X, import X as Y, import X, Y
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "source": file_path,
                            "target": alias.name,
                            "type": "import",
                            "properties": {
                                "alias": alias.asname,
                                "line": node.lineno,
                                "import_type": "absolute",
                            },
                        }
                    )

            # Handle: from X import Y, from . import Y, from ..X import Y
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level = node.level or 0
                import_type = "relative" if level > 0 else "absolute"
                relative_prefix = "." * level

                for alias in node.names:
                    # Star import: from X import *
                    if alias.name == "*":
                        if module:
                            target = f"{relative_prefix}{module}.*"
                        else:
                            target = f"{relative_prefix}*"
                    elif module:
                        target = f"{relative_prefix}{module}.{alias.name}"
                    else:
                        target = f"{relative_prefix}{alias.name}"

                    imports.append(
                        {
                            "source": file_path,
                            "target": target,
                            "type": "import_from",
                            "properties": {
                                "module": module,
                                "alias": alias.asname,
                                "line": node.lineno,
                                "import_type": import_type,
                                "level": level,
                            },
                        }
                    )

        return imports

    def get_supported_extensions(self) -> list[str]:
        """Get list of file extensions supported by this processor.

        Returns:
            List[str]: List of supported file extensions (.py, .pyi)
        """
        return [".py", ".pyi"]

    async def parse_file_with_metrics(self, file_path: str, content: str | None = None):
        """Parse a Python file and return metrics and analysis results.

        This method provides the interface expected by the multi-language analysis system.

        Args:
            file_path: Path to the Python file to analyze
            content: Optional file content (if None, will read from file)

        Returns:
            ParseResult object with success status, entities, relationships, and metrics
        """
        from .languages.base import ParseResult, ParserType

        try:
            # Use the existing process_document method
            result = await self.process_document(file_path, content)

            # Convert to ParseResult object
            return ParseResult(
                success=True,
                file_path=file_path,
                language="python",
                parser_used=ParserType.AST_NATIVE,
                ast=result.get("ast"),  # Include the AST if available
                entities=result.get("entities", []),
                relationships=result.get("relationships", []),
                lines_of_code=result.get("metrics", {}).get("lines_of_code", 0),
                complexity_score=result.get("metrics", {}).get("complexity", 0.0),
                parse_time_ms=0.0,  # Could track this if needed
                errors=[],
                warnings=[],
                properties=result.get("metrics", {}),
            )
        except (OSError, SyntaxError, ValueError, TypeError, AttributeError) as e:
            logger.error("Error parsing Python file %s: %s", file_path, e)
            return ParseResult(
                success=False,
                file_path=file_path,
                language="python",
                parser_used=ParserType.AST_NATIVE,
                ast=None,  # No AST available on error
                entities=[],
                relationships=[],
                lines_of_code=0,
                complexity_score=0.0,
                parse_time_ms=0.0,
                errors=[str(e)],
                warnings=[],
                properties={},
            )

    async def parse_file(self, file_path: str, content: str | None = None) -> dict[str, Any]:
        """Parse a Python file (simplified interface).

        This method provides backward compatibility for code expecting the parse_file interface.

        Args:
            file_path: Path to the Python file to analyze
            content: Optional file content (if None, will read from file)

        Returns:
            Dict containing analysis results
        """
        parse_result = await self.parse_file_with_metrics(file_path, content)
        # Convert ParseResult object back to dictionary for backward compatibility
        return {
            "entities": parse_result.entities,
            "relationships": parse_result.relationships,
            "metrics": {
                "lines_of_code": parse_result.lines_of_code,
                "complexity": parse_result.complexity_score,
                **parse_result.properties,
            },
            "file_info": {"file_path": parse_result.file_path},
            "language": parse_result.language,
            "success": parse_result.success,
            "errors": parse_result.errors,
        }
