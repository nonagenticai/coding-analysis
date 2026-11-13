"""Module for processing Java code and extracting relevant information."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    EntityRelation,
    EntityType,
    RelationType,
    SourceLocation,
)

from ..base import LanguageProcessor, ParseResult, ParserType
from ..registry import register_language
from .java_config import JAVA_CONFIG

# Import Tree-sitter components if available
try:
    from tree_sitter import Node, Tree

    from processing.parsers.tree_sitter_base import BaseTreeSitterProcessor

    # Create JavaTreeSitterProcessor that properly inherits from BaseTreeSitterProcessor
    class JavaTreeSitterProcessor(BaseTreeSitterProcessor):
        """Java Tree-sitter processor that inherits from BaseTreeSitterProcessor."""

        pass

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

    # Create a dummy JavaTreeSitterProcessor that inherits from LanguageProcessor
    class JavaTreeSitterProcessor(LanguageProcessor):
        """Dummy JavaTreeSitterProcessor for when tree-sitter is not available."""

        async def _initialize_impl(self) -> None:
            pass

        async def parse_file(self, file_path: Path, content: str) -> ParseResult:
            """Async Function parse_file.

            Args:
                file_path (Path): The file_path parameter.
                content (str): The content parameter.

            Returns:
                ParseResult: The parseresult result.

            Raises:
                NotImplementedError: Raised under certain conditions."""
            """Async Method parse_file.

            Args:
                file_path (Path): The file_path parameter.
                content (str): The content parameter.

            Returns:
                ParseResult: The parseresult result.

            Raises:
                NotImplementedError: Raised under certain conditions."""
            # This should be overridden by subclasses
            raise NotImplementedError("Tree-sitter not available")

        async def extract_entities(self, ast: Any, file_path: Path) -> list[CoreCodeEntity]:
            """Async Function extract_entities.

            Args:
                ast (Any): The ast parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[CoreCodeEntity]: The list[corecodeentity] result."""
            """Async Method extract_entities.

            Args:
                ast (Any): The ast parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[CoreCodeEntity]: The list[corecodeentity] result."""
            return []

        async def extract_relationships(
            self, ast: Any, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Async Function extract_relationships.

            Args:
                ast (Any): The ast parameter.
                entities (List[CoreCodeEntity]): The entities parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[EntityRelation]: The list[entityrelation] result.

            Example:
                >>> await extract_relationships("ast_value", "entities_value", "file_path_value")
            """
            """Async Method extract_relationships.

            Args:
                ast (Any): The ast parameter.
                entities (List[CoreCodeEntity]): The entities parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[EntityRelation]: The list[entityrelation] result.

            Example:
                >>> await extract_relationships("ast_value", "entities_value", "file_path_value")"""
            return []

    # Tree/Node types not available - use Any in method signatures


@register_language(JAVA_CONFIG)
class JavaProcessor(JavaTreeSitterProcessor):  # type: ignore[unsupported-base]
    """Java language processor using Tree-sitter."""

    # Java-specific node types
    CLASS_NODE_TYPES = {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "annotation_type_declaration",
    }
    METHOD_NODE_TYPES = {"method_declaration", "constructor_declaration"}
    FIELD_NODE_TYPES = {"field_declaration", "variable_declarator"}
    IMPORT_NODE_TYPES = {"import_declaration"}

    def __init__(self, config=None):
        super().__init__(config or JAVA_CONFIG)

    async def _initialize_impl(self) -> None:
        """Initialize Java processor."""
        if TREE_SITTER_AVAILABLE:
            await super()._initialize_impl()
        else:
            # Fallback initialization without Tree-sitter
            pass

    async def parse_file(self, file_path: Path, content: str) -> ParseResult:
        """Parse Java file.

        Args:
            file_path (Path): The file_path parameter.
            content (str): The content parameter.

        Returns:
            ParseResult: The parseresult result.

        Example:
            >>> await parse_file("file_path_value", "content_value")"""
        if TREE_SITTER_AVAILABLE and self.parser:
            result = await super().parse_file(file_path, content)
            return (
                result
                if hasattr(result, "success")
                else ParseResult(
                    success=False,
                    file_path=str(file_path),
                    language="java",
                    parser_used=ParserType.TREE_SITTER,
                    ast=None,
                    entities=[],
                    relationships=[],
                    errors=["Invalid result from parent parser"],
                )
            )
        else:
            # Fallback to regex-based parsing
            return await self._regex_parse(file_path, content)

    async def _regex_parse(self, file_path: Path, content: str) -> ParseResult:
        """Fallback regex-based parsing for Java."""
        entities = []
        relationships: list[EntityRelation] = []

        # Extract packages
        package_match = re.search(r"package\s+([\w.]+);", content)
        if package_match:
            package_name = package_match.group(1)
            entities.append(
                CoreCodeEntity(
                    id=f"package:{package_name}",
                    name=package_name,
                    type=EntityType.PACKAGE,
                    language="java",
                    file_path=str(file_path),
                    line_number=content[: package_match.start()].count("\n") + 1,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility=None,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                )
            )

        # Extract imports
        import_pattern = r"import\s+(static\s+)?([\w.*]+);"
        for match in re.finditer(import_pattern, content):
            is_static = match.group(1) is not None
            import_path = match.group(2)
            entities.append(
                CoreCodeEntity(
                    id=f"import:{import_path}:{file_path}",
                    name=import_path,
                    type=EntityType.IMPORT,
                    language="java",
                    file_path=str(file_path),
                    line_number=content[: match.start()].count("\n") + 1,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility=None,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={
                        "is_static": is_static,
                        "is_wildcard": import_path.endswith("*"),
                    },
                )
            )

        # Extract classes
        class_pattern = r"(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)"
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            entities.append(
                CoreCodeEntity(
                    id=f"class:{class_name}:{file_path}",
                    name=class_name,
                    type=EntityType.CLASS,
                    language="java",
                    file_path=str(file_path),
                    line_number=content[: match.start()].count("\n") + 1,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility=None,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                )
            )

        # Extract methods
        method_pattern = r"(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{"
        for match in re.finditer(method_pattern, content):
            method_name = match.group(1)
            # Skip constructors and common keywords
            if method_name not in [
                "class",
                "interface",
                "enum",
                "i",
                "for",
                "while",
            ]:
                entities.append(
                    CoreCodeEntity(
                        id=f"method:{method_name}:{file_path}:{match.start()}",
                        name=method_name,
                        type=EntityType.FUNCTION,
                        language="java",
                        file_path=str(file_path),
                        line_number=content[: match.start()].count("\n") + 1,
                        end_line_number=None,
                        column_start=None,
                        column_end=None,
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                    )
                )

        return ParseResult(
            success=True,
            file_path=str(file_path),
            language="java",
            parser_used=ParserType.REGEX_FALLBACK,
            ast=None,
            entities=entities,
            relationships=relationships,
            lines_of_code=len(content.splitlines()),
        )

    # Tree-sitter specific methods (only available when Tree-sitter is installed)
    if TREE_SITTER_AVAILABLE:

        async def extract_entities(self, ast: Any, file_path: Path) -> list[CoreCodeEntity]:
            """Extract Java entities from Tree-sitter AST.

            Args:
                ast (Tree): The ast parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[CoreCodeEntity]: The list[corecodeentity] result.

            Example:
                >>> await extract_entities("ast_value", "file_path_value")"""
            entities = []
            if ast is None:
                return entities
            content = ast.root_node.text

            # Extract package
            package_entity = self._extract_package(ast, file_path, content)
            if package_entity:
                entities.append(package_entity)

            # Extract imports
            import_entities = self._extract_imports(ast, file_path, content)
            entities.extend(import_entities)

            # Extract classes, interfaces, enums
            class_entities = self._extract_classes(ast, file_path, content)
            entities.extend(class_entities)

            # Extract methods and constructors
            method_entities = self._extract_methods(ast, file_path, content)
            entities.extend(method_entities)

            # Extract fields
            field_entities = self._extract_fields(ast, file_path, content)
            entities.extend(field_entities)

            return entities

        async def extract_relationships(
            self, ast: Any, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Extract relationships between Java entities.

            Args:
                ast (Tree): The ast parameter.
                entities (List[CoreCodeEntity]): The entities parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[EntityRelation]: The list[entityrelation] result.

            Example:
                >>> await extract_relationships("ast_value", "entities_value", "file_path_value")
            """
            relationships = []
            if ast is None:
                return relationships

            # Extract inheritance relationships
            inheritance_rels = self._extract_inheritance_relationships(ast, entities, file_path)
            relationships.extend(inheritance_rels)

            # Extract method call relationships
            call_rels = self._extract_call_relationships(ast, entities, file_path)
            relationships.extend(call_rels)

            # Extract import relationships
            import_rels = self._extract_import_relationships(ast, entities, file_path)
            relationships.extend(import_rels)

            return relationships

        def _extract_package(
            self, tree: Tree, file_path: Path, content: bytes
        ) -> CoreCodeEntity | None:
            """Extract package declaration."""
            package_nodes = self._find_nodes_by_type(tree, {"package_declaration"})
            if not package_nodes:
                return None

            package_node = package_nodes[0]
            package_name_node = self._get_child_by_type(package_node, "scoped_identifier")
            if not package_name_node:
                package_name_node = self._get_child_by_type(package_node, "identifier")

            if not package_name_node:
                return None

            package_name = self._get_node_text(package_name_node, content)

            return CoreCodeEntity(
                id=f"package:{package_name}",
                name=package_name,
                type=EntityType.PACKAGE,
                language="java",
                file_path=str(file_path),
                line_number=package_node.start_point[0] + 1,
                end_line_number=package_node.end_point[0] + 1,
                column_start=package_node.start_point[1],
                column_end=package_node.end_point[1],
                visibility=None,
                code_snippet=None,
                docstring=None,
                signature=None,
                source_code=None,
                created_at=None,
                updated_at=None,
                properties={
                    "package_name": package_name,
                    "file_path": str(file_path),
                },
            )

        def _extract_imports(
            self, tree: Tree, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract import declarations."""
            entities = []
            import_nodes = self._find_nodes_by_type(tree, self.IMPORT_NODE_TYPES)

            for import_node in import_nodes:
                # Get the import path
                import_path_node = None
                for child in import_node.children:
                    if child.type in {"scoped_identifier", "identifier"}:
                        import_path_node = child
                        break

                if not import_path_node:
                    continue

                import_path = self._get_node_text(import_path_node, content)
                is_static = "static" in self._get_node_text(import_node, content)
                is_wildcard = import_path.endswith("*")

                entity = CoreCodeEntity(
                    id=f"import:{import_path}:{file_path}",
                    name=import_path,
                    type=EntityType.IMPORT,
                    language="java",
                    file_path=str(file_path),
                    line_number=import_node.start_point[0] + 1,
                    end_line_number=import_node.end_point[0] + 1,
                    column_start=import_node.start_point[1],
                    column_end=import_node.end_point[1],
                    visibility=None,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={
                        "import_path": import_path,
                        "is_static": is_static,
                        "is_wildcard": is_wildcard,
                        "file_path": str(file_path),
                    },
                )
                entities.append(entity)

            return entities

        def _extract_classes(
            self, tree: Tree, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract class, interface, and enum declarations."""
            entities = []
            class_nodes = self._find_nodes_by_type(tree, self.CLASS_NODE_TYPES)

            for class_node in class_nodes:
                class_name_node = self._get_child_by_type(class_node, "identifier")
                if not class_name_node:
                    continue

                class_name = self._get_node_text(class_name_node, content)

                # Determine entity type
                if class_node.type == "class_declaration":
                    entity_type = EntityType.CLASS
                elif class_node.type == "interface_declaration":
                    entity_type = EntityType.INTERFACE
                elif class_node.type == "enum_declaration":
                    entity_type = EntityType.ENUM
                else:
                    entity_type = EntityType.CLASS  # Default

                # Extract modifiers
                modifiers = self._extract_modifiers(class_node, content)

                entity = CoreCodeEntity(
                    id=f"class:{class_name}:{file_path}",
                    name=class_name,
                    type=entity_type,
                    language="java",
                    file_path=str(file_path),
                    line_number=class_node.start_point[0] + 1,
                    end_line_number=class_node.end_point[0] + 1,
                    column_start=class_node.start_point[1],
                    column_end=class_node.end_point[1],
                    visibility=modifiers.get("visibility", "public"),
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    is_static=modifiers.get("static", False),
                    is_abstract=modifiers.get("abstract", False),
                    is_final=modifiers.get("final", False),
                    properties={
                        "modifiers": modifiers,
                        "file_path": str(file_path),
                    },
                )
                entities.append(entity)

            return entities

        def _extract_methods(
            self, tree: Tree, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract method and constructor declarations."""
            entities = []
            method_nodes = self._find_nodes_by_type(tree, self.METHOD_NODE_TYPES)

            for method_node in method_nodes:
                method_name_node = self._get_child_by_type(method_node, "identifier")
                if not method_name_node:
                    continue

                method_name = self._get_node_text(method_name_node, content)

                # Extract modifiers
                modifiers = self._extract_modifiers(method_node, content)

                entity = CoreCodeEntity(
                    id=f"method:{method_name}:Point(row={method_node.start_point[0]}, column={method_node.start_point[1]}):{file_path}",
                    name=method_name,
                    type=EntityType.FUNCTION,
                    language="java",
                    file_path=str(file_path),
                    line_number=method_node.start_point[0] + 1,
                    end_line_number=method_node.end_point[0] + 1,
                    column_start=method_node.start_point[1],
                    column_end=method_node.end_point[1],
                    visibility=modifiers.get("visibility", "public"),
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    is_static=modifiers.get("static", False),
                    is_abstract=modifiers.get("abstract", False),
                    is_final=modifiers.get("final", False),
                    properties={
                        "modifiers": modifiers,
                        "is_constructor": method_node.type == "constructor_declaration",
                        "file_path": str(file_path),
                    },
                )
                entities.append(entity)

            return entities

        def _extract_fields(
            self, tree: Tree, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract field declarations."""
            entities = []
            field_nodes = self._find_nodes_by_type(tree, {"field_declaration"})

            for field_node in field_nodes:
                # Handle field declaration with potentially multiple variables
                declarators = self._get_children_by_type(field_node, "variable_declarator")
                for declarator in declarators:
                    identifier_node = self._get_child_by_type(declarator, "identifier")
                    if not identifier_node:
                        continue

                    field_name = self._get_node_text(identifier_node, content)

                    # Extract modifiers
                    modifiers = self._extract_modifiers(field_node, content)

                    entity = CoreCodeEntity(
                        id=f"field:{field_name}:Point(row={declarator.start_point[0]}, column={declarator.start_point[1]}):{file_path}",
                        name=field_name,
                        type=EntityType.VARIABLE,
                        language="java",
                        file_path=str(file_path),
                        line_number=declarator.start_point[0] + 1,
                        end_line_number=declarator.end_point[0] + 1,
                        column_start=declarator.start_point[1],
                        column_end=declarator.end_point[1],
                        visibility=modifiers.get("visibility", "private"),
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        is_static=modifiers.get("static", False),
                        is_final=modifiers.get("final", False),
                        properties={
                            "modifiers": modifiers,
                            "is_constant": modifiers.get("static", False)
                            and modifiers.get("final", False),
                            "file_path": str(file_path),
                        },
                    )
                    entities.append(entity)

            return entities

        def _extract_modifiers(self, node: Node, content: bytes) -> dict[str, Any]:
            """Extract Java modifiers from a node."""
            modifiers: dict[str, Any] = {}
            text = self._get_node_text(node, content).lower()

            # Visibility modifiers
            if "public" in text:
                modifiers["visibility"] = "public"
            elif "private" in text:
                modifiers["visibility"] = "private"
            elif "protected" in text:
                modifiers["visibility"] = "protected"
            else:
                # Default package visibility
                modifiers["visibility"] = "package"

            # Other modifiers
            modifiers["static"] = "static" in text
            modifiers["final"] = "final" in text
            modifiers["abstract"] = "abstract" in text
            modifiers["synchronized"] = "synchronized" in text
            modifiers["native"] = "native" in text
            modifiers["transient"] = "transient" in text
            modifiers["volatile"] = "volatile" in text

            return modifiers

        def _extract_inheritance_relationships(
            self, tree: Tree, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Extract inheritance and implementation relationships."""
            relationships = []
            class_nodes = self._find_nodes_by_type(tree, self.CLASS_NODE_TYPES)

            for class_node in class_nodes:
                class_entity = self._find_entity_by_location(entities, class_node)
                if not class_entity:
                    continue

                # Extract extends clause
                extends_clause = self._get_child_by_type(class_node, "superclass")
                if extends_clause:
                    parent_type_node = self._get_child_by_type(extends_clause, "type_identifier")
                    if parent_type_node:
                        parent_name = self._get_node_text(parent_type_node, tree.root_node.text)

                        relationship = EntityRelation(
                            source_id=class_entity.id,
                            target_id=f"class:{parent_name}:{file_path}",
                            type=RelationType.INHERITS,
                            location=self._node_to_location(extends_clause, file_path),
                        )
                        relationships.append(relationship)

                # Extract implements clause
                implements_clause = self._get_child_by_type(class_node, "super_interfaces")
                if implements_clause:
                    for interface_node in implements_clause.children:
                        if interface_node.type == "type_identifier":
                            interface_name = self._get_node_text(
                                interface_node, tree.root_node.text
                            )

                            relationship = EntityRelation(
                                source_id=class_entity.id,
                                target_id=f"interface:{interface_name}:{file_path}",
                                type=RelationType.IMPLEMENTS,
                                location=self._node_to_location(implements_clause, file_path),
                            )
                            relationships.append(relationship)

            return relationships

        def _extract_call_relationships(
            self, tree: Tree, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Extract method call relationships."""
            relationships = []
            method_invocation_nodes = self._find_nodes_by_type(tree, {"method_invocation"})

            for call_node in method_invocation_nodes:
                # Find the containing method
                containing_method = self._find_containing_method(
                    call_node, entities, self.METHOD_NODE_TYPES
                )
                if not containing_method:
                    continue

                # Extract method name
                method_name_node = self._get_child_by_type(call_node, "identifier")
                if not method_name_node:
                    continue

                method_name = self._get_node_text(method_name_node, tree.root_node.text)

                # Create relationship
                relationship = EntityRelation(
                    source_id=containing_method.id,
                    target_id=f"method:{method_name}:unknown",
                    type=RelationType.CALLS,
                    location=self._node_to_location(call_node, file_path),
                    properties={"method_name": method_name},
                )
                relationships.append(relationship)

            return relationships

        def _extract_import_relationships(
            self, tree: Tree, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Extract import relationships."""
            relationships = []

            # Find file entity (or create one)
            file_entity = next((e for e in entities if e.type == EntityType.FILE), None)
            if not file_entity:
                # Create a file entity if not exists
                file_entity = CoreCodeEntity(
                    id=f"file:{file_path}",
                    name=file_path.name,
                    type=EntityType.FILE,
                    language="java",
                    file_path=str(file_path),
                    line_number=1,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility=None,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                )

            # Create relationships from file to imports
            import_entities = [e for e in entities if e.type == EntityType.IMPORT]
            for import_entity in import_entities:
                relationship = EntityRelation(
                    source_id=file_entity.id,
                    target_id=import_entity.id,
                    type=RelationType.IMPORTS,
                    location=SourceLocation(
                        file_path=str(file_path),
                        line_start=import_entity.line_number,
                        line_end=import_entity.line_number,
                        column_start=import_entity.column_start or 0,
                        column_end=import_entity.column_end or 0,
                    ),
                )
                relationships.append(relationship)

            return relationships

    else:
        # Fallback implementations when Tree-sitter is not available
        async def extract_entities(self, ast: Any, file_path: Path) -> list[CoreCodeEntity]:
            """Fallback entity extraction.

            Args:
                ast (Any): The ast parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[CoreCodeEntity]: The list[corecodeentity] result."""
            return []

        async def extract_relationships(
            self, ast: Any, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Fallback relationship extraction.

            Args:
                ast (Any): The ast parameter.
                entities (List[CoreCodeEntity]): The entities parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[EntityRelation]: The list[entityrelation] result.

            Example:
                >>> await extract_relationships("ast_value", "entities_value", "file_path_value")
            """
            return []
