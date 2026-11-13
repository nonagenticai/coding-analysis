"""
C# language processor using Tree-sitter with regex fallback.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    CoreComplexityMetrics,
    EntityRelation,
    EntityType,
    RelationType,
)

from ..base import LanguageProcessor, ParseResult, ParserType
from ..registry import register_language
from .csharp_config import CSHARP_CONFIG

# Import Tree-sitter components if available
try:
    from tree_sitter import Node, Tree  # noqa: F401

    from processing.parsers.tree_sitter_base import BaseTreeSitterProcessor

    # Create CSharpTreeSitterProcessor that properly inherits from BaseTreeSitterProcessor
    class CSharpTreeSitterProcessor(BaseTreeSitterProcessor):
        """C# Tree-sitter processor that inherits from BaseTreeSitterProcessor."""

        pass

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

    # Create a dummy CSharpTreeSitterProcessor that inherits from LanguageProcessor
    class CSharpTreeSitterProcessor(LanguageProcessor):
        """Dummy CSharpTreeSitterProcessor for when tree-sitter is not available."""

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


@register_language(CSHARP_CONFIG)
class CSharpProcessor(CSharpTreeSitterProcessor):  # type: ignore[unsupported-base]
    """C# language processor using Tree-sitter."""

    # C#-specific node types for Tree-sitter
    NAMESPACE_NODE_TYPES = {"namespace_declaration"}
    USING_NODE_TYPES = {"using_directive"}
    CLASS_NODE_TYPES = {"class_declaration"}
    INTERFACE_NODE_TYPES = {"interface_declaration"}
    STRUCT_NODE_TYPES = {"struct_declaration"}
    ENUM_NODE_TYPES = {"enum_declaration"}
    METHOD_NODE_TYPES = {
        "method_declaration",
        "constructor_declaration",
        "destructor_declaration",
    }
    PROPERTY_NODE_TYPES = {"property_declaration"}
    FIELD_NODE_TYPES = {"field_declaration"}
    EVENT_NODE_TYPES = {"event_declaration"}
    DELEGATE_NODE_TYPES = {"delegate_declaration"}

    def __init__(self, config=None):
        super().__init__(config or CSHARP_CONFIG)

    async def _initialize_impl(self) -> None:
        """Initialize C# processor."""
        if TREE_SITTER_AVAILABLE:
            await super()._initialize_impl()
        else:
            # Fallback initialization without Tree-sitter
            pass

    async def parse_file(self, file_path: Path, content: str) -> ParseResult:
        """Parse C# file.

        Args:
            file_path (Path): The file_path parameter.
            content (str): The content parameter.

        Returns:
            ParseResult: The parseresult result."""
        if TREE_SITTER_AVAILABLE and self.parser:
            result = await super().parse_file(file_path, content)
            return result
        else:
            # Fallback to regex-based parsing
            return await self._regex_parse(file_path, content)

    async def _regex_parse(self, file_path: Path, content: str) -> ParseResult:
        """Fallback regex-based parsing for C#."""
        entities = []
        relationships: list[EntityRelation] = []

        # Extract namespace declarations
        namespace_pattern = r"namespace\s+([A-Za-z_][A-Za-z0-9_.]*)"
        for match in re.finditer(namespace_pattern, content):
            namespace_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"namespace:{namespace_name}:{file_path}",
                    name=namespace_name,
                    type=EntityType.NAMESPACE,
                    language="csharp",
                    file_path=str(file_path),
                    line_number=line_num,
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
                    properties={"namespace_name": namespace_name},
                )
            )

        # Extract using statements
        using_pattern = r"using\s+(?:static\s+)?([A-Za-z_][A-Za-z0-9_.]*)\s*;"
        for match in re.finditer(using_pattern, content):
            using_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"using:{using_name}:{file_path}",
                    name=using_name,
                    type=EntityType.IMPORT,
                    language="csharp",
                    file_path=str(file_path),
                    line_number=line_num,
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
                    properties={"using_directive": using_name},
                )
            )

        # Extract class declarations
        class_pattern = r"(?:(public|private|protected|internal)\s+)?(?:(static|abstract|sealed)\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)"
        for match in re.finditer(class_pattern, content):
            visibility = match.group(1) or "internal"
            modifiers = match.group(2) or ""
            class_name = match.group(3)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"class:{class_name}:{file_path}",
                    name=class_name,
                    type=EntityType.CLASS,
                    language="csharp",
                    file_path=str(file_path),
                    line_number=line_num,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility=visibility,
                    is_static="static" in modifiers,
                    is_abstract="abstract" in modifiers,
                    is_final="sealed" in modifiers,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={
                        "modifiers": modifiers.split() if modifiers else [],
                        "is_sealed": "sealed" in modifiers,
                    },
                )
            )

        # Extract interface declarations
        interface_pattern = (
            r"(?:(public|private|protected|internal)\s+)?interface\s+([A-Za-z_][A-Za-z0-9_]*)"
        )
        for match in re.finditer(interface_pattern, content):
            visibility = match.group(1) or "internal"
            interface_name = match.group(2)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"interface:{interface_name}:{file_path}",
                    name=interface_name,
                    type=EntityType.INTERFACE,
                    language="csharp",
                    file_path=str(file_path),
                    line_number=line_num,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility=visibility,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={"interface_type": True},
                )
            )

        # Extract struct declarations
        struct_pattern = r"(?:(public|private|protected|internal)\s+)?(?:(readonly)\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)"
        for match in re.finditer(struct_pattern, content):
            visibility = match.group(1) or "internal"
            modifiers = match.group(2) or ""
            struct_name = match.group(3)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"struct:{struct_name}:{file_path}",
                    name=struct_name,
                    type=EntityType.STRUCT,
                    language="csharp",
                    file_path=str(file_path),
                    line_number=line_num,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility=visibility,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={
                        "is_readonly": "readonly" in modifiers,
                        "value_type": True,
                    },
                )
            )

        # Extract enum declarations
        enum_pattern = r"(?:(public|private|protected|internal)\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)"
        for match in re.finditer(enum_pattern, content):
            visibility = match.group(1) or "internal"
            enum_name = match.group(2)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"enum:{enum_name}:{file_path}",
                    name=enum_name,
                    type=EntityType.ENUM,
                    language="csharp",
                    file_path=str(file_path),
                    line_number=line_num,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility=visibility,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={"enum_type": True},
                )
            )

        # Extract method declarations
        method_pattern = r"(?:(public|private|protected|internal)\s+)?(?:(static|virtual|override|abstract|async)\s+)?(?:(\w+(?:<[^>]+>)?)\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)"
        for match in re.finditer(method_pattern, content):
            visibility = match.group(1) or "private"
            modifiers = match.group(2) or ""
            return_type = match.group(3) or "void"
            method_name = match.group(4)
            line_num = content[: match.start()].count("\n") + 1

            # Skip property getters/setters and common keywords
            if method_name.lower() in [
                "get",
                "set",
                "i",
                "for",
                "while",
                "switch",
                "using",
                "var",
                "new",
            ]:
                continue

            entities.append(
                CoreCodeEntity(
                    id=f"method:{method_name}:{file_path}:{line_num}",
                    name=method_name,
                    type=EntityType.FUNCTION,
                    language="csharp",
                    file_path=str(file_path),
                    line_number=line_num,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility=visibility,
                    is_static="static" in modifiers,
                    is_abstract="abstract" in modifiers,
                    is_async="async" in modifiers,
                    code_snippet=None,
                    docstring=None,
                    signature=match.group(0).strip(),
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={
                        "return_type": return_type,
                        "modifiers": modifiers.split() if modifiers else [],
                        "is_virtual": "virtual" in modifiers,
                        "is_override": "override" in modifiers,
                    },
                )
            )

        # Extract property declarations
        property_pattern = r"(?:(public|private|protected|internal)\s+)?(?:(static|virtual|override|abstract)\s+)?(\w+(?:<[^>]+>)?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{"
        for match in re.finditer(property_pattern, content):
            visibility = match.group(1) or "private"
            modifiers = match.group(2) or ""
            prop_type = match.group(3)
            prop_name = match.group(4)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"property:{prop_name}:{file_path}:{line_num}",
                    name=prop_name,
                    type=EntityType.VARIABLE,  # Properties are a form of variable access
                    language="csharp",
                    file_path=str(file_path),
                    line_number=line_num,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility=visibility,
                    is_static="static" in modifiers,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={
                        "property_type": prop_type,
                        "is_property": True,
                        "modifiers": modifiers.split() if modifiers else [],
                    },
                )
            )

        # Extract field declarations
        field_pattern = r"(?:(public|private|protected|internal)\s+)?(?:(static|readonly|const)\s+)?(\w+(?:<[^>]+>)?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*[;=]"
        for match in re.finditer(field_pattern, content):
            visibility = match.group(1) or "private"
            modifiers = match.group(2) or ""
            field_type = match.group(3)
            field_name = match.group(4)
            line_num = content[: match.start()].count("\n") + 1

            # Determine entity type
            if "const" in modifiers:
                entity_type = EntityType.CONSTANT
            else:
                entity_type = EntityType.VARIABLE

            entities.append(
                CoreCodeEntity(
                    id=f"field:{field_name}:{file_path}:{line_num}",
                    name=field_name,
                    type=entity_type,
                    language="csharp",
                    file_path=str(file_path),
                    line_number=line_num,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility=visibility,
                    is_static="static" in modifiers,
                    is_final="readonly" in modifiers or "const" in modifiers,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={
                        "field_type": field_type,
                        "is_readonly": "readonly" in modifiers,
                        "is_const": "const" in modifiers,
                        "modifiers": modifiers.split() if modifiers else [],
                    },
                )
            )

        # Basic complexity metrics
        complexity = CoreComplexityMetrics(
            lines_of_code=len(content.splitlines()),
            logical_lines=len(
                [
                    line
                    for line in content.splitlines()
                    if line.strip() and not line.strip().startswith("//")
                ]
            ),
            blank_lines=len([line for line in content.splitlines() if not line.strip()]),
            comment_lines=len(
                [
                    line
                    for line in content.splitlines()
                    if line.strip().startswith("//") or "/*" in line
                ]
            ),
            function_count=len([e for e in entities if e.type == EntityType.FUNCTION]),
            class_count=len([e for e in entities if e.type == EntityType.CLASS]),
            docstring_coverage=0.0,
        )

        return ParseResult(
            language="csharp",
            file_path=str(file_path),
            parser_used=ParserType.REGEX_FALLBACK,
            success=True,
            entities=entities,
            relationships=relationships,
            lines_of_code=len(content.splitlines()),
            complexity_score=complexity.cyclomatic_complexity,
            parse_time_ms=0.0,
            ast=None,
        )

    # Tree-sitter implementation methods (when available)
    async def _extract_tree_sitter_entities(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract entities using Tree-sitter (when available)."""
        if not TREE_SITTER_AVAILABLE:
            return []

        entities = []

        # Extract namespaces
        entities.extend(await self._extract_namespaces(tree, file_path, content))

        # Extract using directives
        entities.extend(await self._extract_using_directives(tree, file_path, content))

        # Extract classes
        entities.extend(await self._extract_classes(tree, file_path, content))

        # Extract interfaces
        entities.extend(await self._extract_interfaces(tree, file_path, content))

        # Extract structs
        entities.extend(await self._extract_structs(tree, file_path, content))

        # Extract enums
        entities.extend(await self._extract_enums(tree, file_path, content))

        # Extract methods
        entities.extend(await self._extract_methods(tree, file_path, content))

        # Extract properties and fields
        entities.extend(await self._extract_properties(tree, file_path, content))
        entities.extend(await self._extract_fields(tree, file_path, content))

        return entities

    async def _extract_namespaces(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract namespace declarations."""
        entities = []
        namespace_nodes = self._find_nodes_by_type(tree, self.NAMESPACE_NODE_TYPES)

        for node in namespace_nodes:
            namespace_text = self._get_node_text(node, content)
            name_match = re.search(r"namespace\s+([A-Za-z_][A-Za-z0-9_.]*)", namespace_text)
            if name_match:
                namespace_name = name_match.group(1)

                entities.append(
                    CoreCodeEntity(
                        id=f"namespace:{namespace_name}:{file_path}",
                        name=namespace_name,
                        type=EntityType.NAMESPACE,
                        language="csharp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"namespace_name": namespace_name},
                    )
                )

        return entities

    async def _extract_using_directives(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract using directives."""
        entities = []
        using_nodes = self._find_nodes_by_type(tree, self.USING_NODE_TYPES)

        for node in using_nodes:
            using_text = self._get_node_text(node, content)
            name_match = re.search(r"using\s+(?:static\s+)?([A-Za-z_][A-Za-z0-9_.]*)", using_text)
            if name_match:
                using_name = name_match.group(1)

                entities.append(
                    CoreCodeEntity(
                        id=f"using:{using_name}:{file_path}",
                        name=using_name,
                        type=EntityType.IMPORT,
                        language="csharp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"using_directive": using_name},
                    )
                )

        return entities

    async def _extract_classes(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract class declarations."""
        entities = []
        class_nodes = self._find_nodes_by_type(tree, self.CLASS_NODE_TYPES)

        for node in class_nodes:
            class_text = self._get_node_text(node, content)

            # Extract class name and modifiers
            class_match = re.search(
                r"(?:(public|private|protected|internal)\s+)?(?:(static|abstract|sealed)\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)",
                class_text,
            )
            if class_match:
                visibility = class_match.group(1) or "internal"
                modifiers = class_match.group(2) or ""
                class_name = class_match.group(3)

                entities.append(
                    CoreCodeEntity(
                        id=f"class:{class_name}:{file_path}",
                        name=class_name,
                        type=EntityType.CLASS,
                        language="csharp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=visibility,
                        is_static="static" in modifiers,
                        is_abstract="abstract" in modifiers,
                        is_final="sealed" in modifiers,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={
                            "modifiers": (modifiers.split() if modifiers else []),
                            "is_sealed": "sealed" in modifiers,
                        },
                    )
                )

        return entities

    async def _extract_interfaces(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract interface declarations."""
        entities = []
        interface_nodes = self._find_nodes_by_type(tree, self.INTERFACE_NODE_TYPES)

        for node in interface_nodes:
            interface_text = self._get_node_text(node, content)

            interface_match = re.search(
                r"(?:(public|private|protected|internal)\s+)?interface\s+([A-Za-z_][A-Za-z0-9_]*)",
                interface_text,
            )
            if interface_match:
                visibility = interface_match.group(1) or "internal"
                interface_name = interface_match.group(2)

                entities.append(
                    CoreCodeEntity(
                        id=f"interface:{interface_name}:{file_path}",
                        name=interface_name,
                        type=EntityType.INTERFACE,
                        language="csharp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=visibility,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"interface_type": True},
                    )
                )

        return entities

    async def _extract_structs(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract struct declarations."""
        entities = []
        struct_nodes = self._find_nodes_by_type(tree, self.STRUCT_NODE_TYPES)

        for node in struct_nodes:
            struct_text = self._get_node_text(node, content)

            struct_match = re.search(
                r"(?:(public|private|protected|internal)\s+)?(?:(readonly)\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)",
                struct_text,
            )
            if struct_match:
                visibility = struct_match.group(1) or "internal"
                modifiers = struct_match.group(2) or ""
                struct_name = struct_match.group(3)

                entities.append(
                    CoreCodeEntity(
                        id=f"struct:{struct_name}:{file_path}",
                        name=struct_name,
                        type=EntityType.STRUCT,
                        language="csharp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=visibility,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={
                            "is_readonly": "readonly" in modifiers,
                            "value_type": True,
                        },
                    )
                )

        return entities

    async def _extract_enums(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract enum declarations."""
        entities = []
        enum_nodes = self._find_nodes_by_type(tree, self.ENUM_NODE_TYPES)

        for node in enum_nodes:
            enum_text = self._get_node_text(node, content)

            enum_match = re.search(
                r"(?:(public|private|protected|internal)\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)",
                enum_text,
            )
            if enum_match:
                visibility = enum_match.group(1) or "internal"
                enum_name = enum_match.group(2)

                entities.append(
                    CoreCodeEntity(
                        id=f"enum:{enum_name}:{file_path}",
                        name=enum_name,
                        type=EntityType.ENUM,
                        language="csharp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=visibility,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"enum_type": True},
                    )
                )

        return entities

    async def _extract_methods(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract method declarations."""
        entities = []
        method_nodes = self._find_nodes_by_type(tree, self.METHOD_NODE_TYPES)

        for node in method_nodes:
            method_text = self._get_node_text(node, content)

            method_match = re.search(
                r"(?:(public|private|protected|internal)\s+)?(?:(static|virtual|override|abstract|async)\s+)?(?:(\w+(?:<[^>]+>)?)\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                method_text,
            )
            if method_match:
                visibility = method_match.group(1) or "private"
                modifiers = method_match.group(2) or ""
                return_type = method_match.group(3) or "void"
                method_name = method_match.group(4)

                entities.append(
                    CoreCodeEntity(
                        id=f"method:{method_name}:{file_path}:{node.start_point[0] + 1}",
                        name=method_name,
                        type=EntityType.FUNCTION,
                        language="csharp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=visibility,
                        is_static="static" in modifiers,
                        is_abstract="abstract" in modifiers,
                        is_async="async" in modifiers,
                        code_snippet=None,
                        docstring=None,
                        signature=method_text.split("\n")[0].strip(),
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={
                            "return_type": return_type,
                            "modifiers": (modifiers.split() if modifiers else []),
                            "is_virtual": "virtual" in modifiers,
                            "is_override": "override" in modifiers,
                        },
                    )
                )

        return entities

    async def _extract_properties(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract property declarations."""
        entities = []
        property_nodes = self._find_nodes_by_type(tree, self.PROPERTY_NODE_TYPES)

        for node in property_nodes:
            prop_text = self._get_node_text(node, content)

            prop_match = re.search(
                r"(?:(public|private|protected|internal)\s+)?(?:(static|virtual|override|abstract)\s+)?(\w+(?:<[^>]+>)?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{",
                prop_text,
            )
            if prop_match:
                visibility = prop_match.group(1) or "private"
                modifiers = prop_match.group(2) or ""
                prop_type = prop_match.group(3)
                prop_name = prop_match.group(4)

                entities.append(
                    CoreCodeEntity(
                        id=f"property:{prop_name}:{file_path}:{node.start_point[0] + 1}",
                        name=prop_name,
                        type=EntityType.VARIABLE,
                        language="csharp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=visibility,
                        is_static="static" in modifiers,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={
                            "property_type": prop_type,
                            "is_property": True,
                            "modifiers": (modifiers.split() if modifiers else []),
                        },
                    )
                )

        return entities

    async def _extract_fields(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract field declarations."""
        entities = []
        field_nodes = self._find_nodes_by_type(tree, self.FIELD_NODE_TYPES)

        for node in field_nodes:
            field_text = self._get_node_text(node, content)

            field_match = re.search(
                r"(?:(public|private|protected|internal)\s+)?(?:(static|readonly|const)\s+)?(\w+(?:<[^>]+>)?)\s+([A-Za-z_][A-Za-z0-9_]*)",
                field_text,
            )
            if field_match:
                visibility = field_match.group(1) or "private"
                modifiers = field_match.group(2) or ""
                field_type = field_match.group(3)
                field_name = field_match.group(4)

                # Determine entity type
                if "const" in modifiers:
                    entity_type = EntityType.CONSTANT
                else:
                    entity_type = EntityType.VARIABLE

                entities.append(
                    CoreCodeEntity(
                        id=f"field:{field_name}:{file_path}:{node.start_point[0] + 1}",
                        name=field_name,
                        type=entity_type,
                        language="csharp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=visibility,
                        is_static="static" in modifiers,
                        is_final="readonly" in modifiers or "const" in modifiers,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={
                            "field_type": field_type,
                            "is_readonly": "readonly" in modifiers,
                            "is_const": "const" in modifiers,
                            "modifiers": (modifiers.split() if modifiers else []),
                        },
                    )
                )

        return entities

    # Required abstract methods from LanguageProcessor
    async def extract_entities(self, ast: Any, file_path: Path) -> list[CoreCodeEntity]:
        """Extract code entities from AST (Tree-sitter tree).

        Args:
            ast (Any): The ast parameter.
            file_path (Path): The file_path parameter.

        Returns:
            List[CoreCodeEntity]: The list[corecodeentity] result."""
        if TREE_SITTER_AVAILABLE and isinstance(ast, Tree):
            content_bytes = file_path.read_bytes()
            return await self._extract_tree_sitter_entities(ast, file_path, content_bytes)
        else:
            # Fallback to regex parsing
            try:
                content_str = file_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, FileNotFoundError):
                return []
            result = await self._regex_parse(file_path, content_str)
            return result.entities

    async def extract_relationships(
        self, ast: Any, entities: list[CoreCodeEntity], file_path: Path
    ) -> list[EntityRelation]:
        """Extract relationships between entities.

        Args:
            ast (Any): The ast parameter.
            entities (List[CoreCodeEntity]): The entities parameter.
            file_path (Path): The file_path parameter.

        Returns:
            List[EntityRelation]: The list[entityrelation] result.

        Example:
            >>> await extract_relationships("ast_value", "entities_value", "file_path_value")
        """
        relationships: list[EntityRelation] = []

        # Basic relationships based on entity structure
        namespace_entities = [e for e in entities if e.type == EntityType.NAMESPACE]

        # Namespace contains other entities
        for entity in entities:
            if entity.type != EntityType.NAMESPACE and namespace_entities:
                namespace = namespace_entities[0]  # C# files typically have one namespace
                relationships.append(
                    EntityRelation(
                        source_id=namespace.id,
                        target_id=entity.id,
                        type=RelationType.CONTAINS,
                        properties={"relationship_context": "namespace_member"},
                    )
                )

        # Class/Interface contains methods and properties
        container_entities = [
            e
            for e in entities
            if e.type in [EntityType.CLASS, EntityType.INTERFACE, EntityType.STRUCT]
        ]
        member_entities = [
            e for e in entities if e.type in [EntityType.FUNCTION, EntityType.VARIABLE]
        ]

        for container in container_entities:
            for member in member_entities:
                # Basic heuristic: if member is defined after container in same file
                if member.line_number > container.line_number:
                    relationships.append(
                        EntityRelation(
                            source_id=container.id,
                            target_id=member.id,
                            type=RelationType.DEFINES,
                            properties={"relationship_context": "member_definition"},
                        )
                    )

        return relationships
