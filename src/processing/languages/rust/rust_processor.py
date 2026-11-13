"""Rust language processor module using Tree-sitter for parsing Rust code.

This module provides a comprehensive Rust language processor that can extract
entities (structs, enums, traits, functions, etc.) and their relationships
from Rust source code. It supports both Tree-sitter AST parsing and regex
fallback for when Tree-sitter is not available.
"""

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
from .rust_config import RUST_CONFIG

# Import Tree-sitter components if available
try:
    from tree_sitter import Node, Tree

    from processing.parsers.tree_sitter_base import BaseTreeSitterProcessor

    # Create RustTreeSitterProcessor that properly inherits from BaseTreeSitterProcessor
    class RustTreeSitterProcessor(BaseTreeSitterProcessor):
        """Rust Tree-sitter processor that inherits from BaseTreeSitterProcessor."""

        pass

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

    # Create a dummy RustTreeSitterProcessor that inherits from LanguageProcessor
    class RustTreeSitterProcessor(LanguageProcessor):
        """Dummy RustTreeSitterProcessor for when tree-sitter is not available."""

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


@register_language(RUST_CONFIG)
class RustProcessor(RustTreeSitterProcessor):  # type: ignore[unsupported-base]
    """Rust language processor using Tree-sitter."""

    # Rust-specific node types
    STRUCT_NODE_TYPES = {"struct_item"}
    ENUM_NODE_TYPES = {"enum_item"}
    TRAIT_NODE_TYPES = {"trait_item"}
    IMPL_NODE_TYPES = {"impl_item"}
    FUNCTION_NODE_TYPES = {"function_item"}
    MOD_NODE_TYPES = {"mod_item"}
    USE_NODE_TYPES = {"use_declaration"}
    CONST_NODE_TYPES = {"const_item"}
    STATIC_NODE_TYPES = {"static_item"}
    TYPE_NODE_TYPES = {"type_item"}
    MACRO_NODE_TYPES = {"macro_definition", "macro_invocation"}

    def __init__(self, config=None):
        super().__init__(config or RUST_CONFIG)

    async def _initialize_impl(self) -> None:
        """Initialize Rust processor."""
        if TREE_SITTER_AVAILABLE:
            await super()._initialize_impl()
        else:
            # Fallback initialization without Tree-sitter
            pass

    async def parse_file(self, file_path: Path, content: str) -> ParseResult:
        """Parse Rust file.

        Args:
            file_path (Path): The file_path parameter.
            content (str): The content parameter.

        Returns:
            ParseResult: The parseresult result."""
        if TREE_SITTER_AVAILABLE and self.parser:
            result = await super().parse_file(file_path, content)
            return result
        # Fallback to regex-based parsing
        return await self._regex_parse(file_path, content)

    async def _regex_parse(self, file_path: Path, content: str) -> ParseResult:
        """Fallback regex-based parsing for Rust."""
        entities = []
        relationships: list[EntityRelation] = []

        # Extract different entity types
        entities.extend(self._extract_modules_regex(content, file_path))
        entities.extend(self._extract_use_statements_regex(content, file_path))
        entities.extend(self._extract_structs_regex(content, file_path))
        entities.extend(self._extract_enums_regex(content, file_path))
        entities.extend(self._extract_traits_regex(content, file_path))
        entities.extend(self._extract_functions_regex(content, file_path))
        entities.extend(self._extract_impl_blocks_regex(content, file_path))
        entities.extend(self._extract_macros_regex(content, file_path))

        return ParseResult(
            success=True,
            file_path=str(file_path),
            language="rust",
            parser_used=ParserType.REGEX_FALLBACK,
            entities=entities,
            relationships=relationships,
            lines_of_code=len(content.splitlines()),
            complexity_score=1.0,
            parse_time_ms=0.0,
            ast=None,
        )

    def _extract_modules_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract module declarations using regex."""
        entities = []
        mod_pattern = r"(?:pub\s+)?mod\s+(\w+)"
        for match in re.finditer(mod_pattern, content):
            mod_name = match.group(1)
            entities.append(
                CoreCodeEntity(
                    id=f"mod:{mod_name}:{file_path}",
                    name=mod_name,
                    type=EntityType.MODULE,
                    language="rust",
                    file_path=str(file_path),
                    line_number=content[: match.start()].count("\n") + 1,
                    end_line_number=None,
                    column_start=0,
                    column_end=0,
                    visibility=("public" if "pub" in match.group(0) else "private"),
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={"is_public": "pub" in match.group(0)},
                )
            )
        return entities

    def _extract_use_statements_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract use statements using regex."""
        entities = []
        use_pattern = r"use\s+([^;]+);"
        for match in re.finditer(use_pattern, content):
            use_path = match.group(1).strip()
            entities.append(
                CoreCodeEntity(
                    id=f"use:{use_path}:{file_path}",
                    name=use_path,
                    type=EntityType.IMPORT,
                    language="rust",
                    file_path=str(file_path),
                    line_number=content[: match.start()].count("\n") + 1,
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
                    properties={"use_path": use_path},
                )
            )
        return entities

    def _extract_structs_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract struct definitions using regex."""
        entities = []
        struct_pattern = r"(?:pub\s+)?struct\s+(\w+)"
        for match in re.finditer(struct_pattern, content):
            struct_name = match.group(1)
            entities.append(
                CoreCodeEntity(
                    id=f"struct:{struct_name}:{file_path}",
                    name=struct_name,
                    type=EntityType.STRUCT,
                    language="rust",
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
        return entities

    def _extract_enums_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract enum definitions using regex."""
        entities = []
        enum_pattern = r"(?:pub\s+)?enum\s+(\w+)"
        for match in re.finditer(enum_pattern, content):
            enum_name = match.group(1)
            entities.append(
                CoreCodeEntity(
                    id=f"enum:{enum_name}:{file_path}",
                    name=enum_name,
                    type=EntityType.ENUM,
                    language="rust",
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
        return entities

    def _extract_traits_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract trait definitions using regex."""
        entities = []
        trait_pattern = r"(?:pub\s+)?trait\s+(\w+)"
        for match in re.finditer(trait_pattern, content):
            trait_name = match.group(1)
            entities.append(
                CoreCodeEntity(
                    id=f"trait:{trait_name}:{file_path}",
                    name=trait_name,
                    type=EntityType.TRAIT,
                    language="rust",
                    file_path=str(file_path),
                    line_number=content[: match.start()].count("\n") + 1,
                    end_line_number=None,
                    column_start=0,
                    column_end=0,
                    visibility=("public" if "pub" in match.group(0) else "private"),
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                )
            )
        return entities

    def _extract_functions_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract function definitions using regex."""
        entities = []
        fn_pattern = r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\("
        for match in re.finditer(fn_pattern, content):
            fn_name = match.group(1)
            # Skip common keywords
            if fn_name not in ["main", "new", "default"]:
                entities.append(
                    CoreCodeEntity(
                        id=f"fn:{fn_name}:{file_path}:{match.start()}",
                        name=fn_name,
                        type=EntityType.FUNCTION,
                        language="rust",
                        file_path=str(file_path),
                        line_number=content[: match.start()].count("\n") + 1,
                        end_line_number=None,
                        column_start=0,
                        column_end=0,
                        visibility=("public" if "pub" in match.group(0) else "private"),
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        is_async="async" in match.group(0),
                    )
                )
        return entities

    def _extract_impl_blocks_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract impl blocks using regex."""
        entities = []
        impl_pattern = r"impl(?:\s*<[^>]*>)?\s+(?:(\w+)\s+for\s+)?(\w+)"
        for match in re.finditer(impl_pattern, content):
            trait_name = match.group(1)  # Could be None for inherent impl
            type_name = match.group(2)

            if trait_name:
                # Trait implementation
                entities.append(
                    CoreCodeEntity(
                        id=f"impl:{trait_name}:for:{type_name}:{file_path}:{match.start()}",
                        name=f"{trait_name} for {type_name}",
                        type=EntityType.CLASS,  # Using class to represent impl blocks
                        language="rust",
                        file_path=str(file_path),
                        line_number=content[: match.start()].count("\n") + 1,
                        end_line_number=None,
                        column_start=0,
                        column_end=0,
                        visibility="private",
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={
                            "impl_type": "trait",
                            "trait_name": trait_name,
                            "type_name": type_name,
                        },
                    )
                )
            else:
                # Inherent implementation
                entities.append(
                    CoreCodeEntity(
                        id=f"impl:{type_name}:{file_path}:{match.start()}",
                        name=f"impl {type_name}",
                        type=EntityType.CLASS,
                        language="rust",
                        file_path=str(file_path),
                        line_number=content[: match.start()].count("\n") + 1,
                        end_line_number=None,
                        column_start=0,
                        column_end=0,
                        visibility="private",
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={
                            "impl_type": "inherent",
                            "type_name": type_name,
                        },
                    )
                )
        return entities

    def _extract_macros_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract macro definitions using regex."""
        entities = []
        macro_pattern = r"macro_rules!\s+(\w+)"
        for match in re.finditer(macro_pattern, content):
            macro_name = match.group(1)
            entities.append(
                CoreCodeEntity(
                    id=f"macro:{macro_name}:{file_path}",
                    name=macro_name,
                    type=EntityType.MACRO,
                    language="rust",
                    file_path=str(file_path),
                    line_number=content[: match.start()].count("\n") + 1,
                    end_line_number=None,
                    column_start=0,
                    column_end=0,
                    visibility="private",
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                )
            )
        return entities

    # Tree-sitter specific methods (only available when Tree-sitter is installed)
    if TREE_SITTER_AVAILABLE:

        async def extract_entities(self, tree: Tree, file_path: Path) -> list[CoreCodeEntity]:
            """Extract Rust entities from Tree-sitter AST.

            Args:
                tree (Tree): The tree parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[CoreCodeEntity]: The list[corecodeentity] result.

            Example:
                >>> await extract_entities("tree_value", "file_path_value")"""
            entities = []
            # tree-sitter exposes source bytes on the root node, not the Tree.
            content = tree.root_node.text or b""

            # Extract modules
            mod_entities = self._extract_modules(tree, file_path, content)
            entities.extend(mod_entities)

            # Extract use declarations
            use_entities = self._extract_use_declarations(tree, file_path, content)
            entities.extend(use_entities)

            # Extract structs
            struct_entities = self._extract_structs(tree, file_path, content)
            entities.extend(struct_entities)

            # Extract enums
            enum_entities = self._extract_enums(tree, file_path, content)
            entities.extend(enum_entities)

            # Extract traits
            trait_entities = self._extract_traits(tree, file_path, content)
            entities.extend(trait_entities)

            # Extract functions
            function_entities = self._extract_functions(tree, file_path, content)
            entities.extend(function_entities)

            # Extract impl blocks
            impl_entities = self._extract_impl_blocks(tree, file_path, content)
            entities.extend(impl_entities)

            # Extract constants and statics
            const_entities = self._extract_constants(tree, file_path, content)
            entities.extend(const_entities)

            return entities

        async def extract_relationships(
            self, tree: Tree, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Extract relationships between Rust entities.

            Args:
                tree (Tree): The tree parameter.
                entities (List[CoreCodeEntity]): The entities parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[EntityRelation]: The list[entityrelation] result.

            Example:
                >>> await extract_relationships("tree_value", "entities_value", "file_path_value")
            """
            relationships: list[EntityRelation] = []

            # Extract trait implementations
            impl_rels = self._extract_trait_implementations(tree, entities, file_path)
            relationships.extend(impl_rels)

            # Extract function calls
            call_rels = self._extract_function_calls(tree, entities, file_path)
            relationships.extend(call_rels)

            # Extract use relationships
            use_rels = self._extract_use_relationships(tree, entities, file_path)
            relationships.extend(use_rels)

            return relationships

        def _extract_modules(
            self, tree: Tree, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract module declarations."""
            entities = []
            mod_nodes = self._find_nodes_by_type(tree, self.MOD_NODE_TYPES)

            for mod_node in mod_nodes:
                name_node = self._get_child_by_type(mod_node, "identifier")
                if not name_node:
                    continue

                mod_name = self._get_node_text(name_node, content)
                visibility = self._extract_visibility(mod_node, content)

                entity = CoreCodeEntity(
                    id=f"mod:{mod_name}:{file_path}",
                    name=mod_name,
                    type=EntityType.MODULE,
                    language="rust",
                    file_path=str(file_path),
                    line_number=mod_node.start_point[0] + 1,
                    end_line_number=None,
                    column_start=mod_node.start_point[1],
                    column_end=mod_node.end_point[1],
                    visibility=visibility,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={"node_type": "module"},
                )
                entities.append(entity)

            return entities

        def _extract_use_declarations(
            self, tree: Tree, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract use declarations (imports)."""
            entities = []
            use_nodes = self._find_nodes_by_type(tree, self.USE_NODE_TYPES)

            for use_node in use_nodes:
                # Get the use path
                use_path = (
                    self._get_node_text(use_node, content)
                    .replace("use ", "")
                    .replace(";", "")
                    .strip()
                )

                entity = CoreCodeEntity(
                    id=f"use:{use_path}:{file_path}",
                    name=use_path,
                    type=EntityType.IMPORT,
                    language="rust",
                    file_path=str(file_path),
                    line_number=use_node.start_point[0] + 1,
                    end_line_number=None,
                    column_start=use_node.start_point[1],
                    column_end=use_node.end_point[1],
                    visibility="private",
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={"use_path": use_path},
                )
                entities.append(entity)

            return entities

        def _extract_structs(
            self, tree: Tree, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract struct definitions."""
            entities = []
            struct_nodes = self._find_nodes_by_type(tree, self.STRUCT_NODE_TYPES)

            for struct_node in struct_nodes:
                name_node = self._get_child_by_type(struct_node, "type_identifier")
                if not name_node:
                    continue

                struct_name = self._get_node_text(name_node, content)
                visibility = self._extract_visibility(struct_node, content)

                entity = CoreCodeEntity(
                    id=f"struct:{struct_name}:{file_path}",
                    name=struct_name,
                    type=EntityType.STRUCT,
                    language="rust",
                    file_path=str(file_path),
                    line_number=struct_node.start_point[0] + 1,
                    end_line_number=None,
                    column_start=struct_node.start_point[1],
                    column_end=struct_node.end_point[1],
                    visibility=visibility,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={"node_type": "struct"},
                )
                entities.append(entity)

            return entities

        def _extract_enums(
            self, tree: Tree, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract enum definitions."""
            entities = []
            enum_nodes = self._find_nodes_by_type(tree, self.ENUM_NODE_TYPES)

            for enum_node in enum_nodes:
                name_node = self._get_child_by_type(enum_node, "type_identifier")
                if not name_node:
                    continue

                enum_name = self._get_node_text(name_node, content)
                visibility = self._extract_visibility(enum_node, content)

                entity = CoreCodeEntity(
                    id=f"enum:{enum_name}:{file_path}",
                    name=enum_name,
                    type=EntityType.ENUM,
                    language="rust",
                    file_path=str(file_path),
                    line_number=enum_node.start_point[0] + 1,
                    end_line_number=None,
                    column_start=enum_node.start_point[1],
                    column_end=enum_node.end_point[1],
                    visibility=visibility,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={"node_type": "enum"},
                )
                entities.append(entity)

            return entities

        def _extract_traits(
            self, tree: Tree, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract trait definitions."""
            entities = []
            trait_nodes = self._find_nodes_by_type(tree, self.TRAIT_NODE_TYPES)

            for trait_node in trait_nodes:
                name_node = self._get_child_by_type(trait_node, "type_identifier")
                if not name_node:
                    continue

                trait_name = self._get_node_text(name_node, content)
                visibility = self._extract_visibility(trait_node, content)

                entity = CoreCodeEntity(
                    id=f"trait:{trait_name}:{file_path}",
                    name=trait_name,
                    type=EntityType.TRAIT,
                    language="rust",
                    file_path=str(file_path),
                    line_number=trait_node.start_point[0] + 1,
                    end_line_number=None,
                    column_start=trait_node.start_point[1],
                    column_end=trait_node.end_point[1],
                    visibility=visibility,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={"node_type": "trait"},
                )
                entities.append(entity)

            return entities

        def _extract_functions(
            self, tree: Tree, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract function definitions."""
            entities = []
            function_nodes = self._find_nodes_by_type(tree, self.FUNCTION_NODE_TYPES)

            for fn_node in function_nodes:
                name_node = self._get_child_by_type(fn_node, "identifier")
                if not name_node:
                    continue

                fn_name = self._get_node_text(name_node, content)
                visibility = self._extract_visibility(fn_node, content)
                is_async = "async" in self._get_node_text(fn_node, content)

                entity = CoreCodeEntity(
                    id=f"fn:{fn_name}:Point(row={fn_node.start_point[0]}, column={fn_node.start_point[1]}):{file_path}",
                    name=fn_name,
                    type=EntityType.FUNCTION,
                    language="rust",
                    file_path=str(file_path),
                    line_number=fn_node.start_point[0] + 1,
                    end_line_number=None,
                    column_start=fn_node.start_point[1],
                    column_end=fn_node.end_point[1],
                    visibility=visibility,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    is_async=is_async,
                    properties={"node_type": "function"},
                )
                entities.append(entity)

            return entities

        def _extract_impl_blocks(
            self, tree: Tree, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract impl blocks."""
            entities = []
            impl_nodes = self._find_nodes_by_type(tree, self.IMPL_NODE_TYPES)

            for impl_node in impl_nodes:
                impl_text = self._get_node_text(impl_node, content)

                # Try to determine if it's a trait impl or inherent impl
                if " for " in impl_text:
                    # Trait implementation
                    parts = impl_text.split(" for ")
                    if len(parts) >= 2:
                        trait_part = parts[0].replace("impl", "").strip()
                        type_part = parts[1].split("{")[0].strip()

                        entity = CoreCodeEntity(
                            id=(
                                f"impl:{trait_part}:for:{type_part}:{file_path}:Point(row={impl_node.start_point[0]}, column={impl_node.start_point[1]})"
                            ),
                            name=f"{trait_part} for {type_part}",
                            type=EntityType.CLASS,  # Using class to represent impl
                            language="rust",
                            file_path=str(file_path),
                            line_number=impl_node.start_point[0] + 1,
                            end_line_number=None,
                            column_start=impl_node.start_point[1],
                            column_end=impl_node.end_point[1],
                            visibility=None,
                            code_snippet=None,
                            docstring=None,
                            signature=None,
                            source_code=None,
                            created_at=None,
                            updated_at=None,
                            properties={
                                "impl_type": "trait",
                                "trait_name": trait_part,
                                "type_name": type_part,
                            },
                        )
                        entities.append(entity)
                else:
                    # Inherent implementation
                    type_part = impl_text.replace("impl", "").split("{")[0].strip()

                    entity = CoreCodeEntity(
                        id=f"impl:{type_part}:{file_path}:Point(row={impl_node.start_point[0]}, column={impl_node.start_point[1]})",
                        name=f"impl {type_part}",
                        type=EntityType.CLASS,
                        language="rust",
                        file_path=str(file_path),
                        line_number=impl_node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=impl_node.start_point[1],
                        column_end=impl_node.end_point[1],
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={
                            "impl_type": "inherent",
                            "type_name": type_part,
                        },
                    )
                    entities.append(entity)

            return entities

        def _extract_constants(
            self, tree: Tree, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract const and static declarations."""
            entities = []

            # Extract const items
            const_nodes = self._find_nodes_by_type(tree, self.CONST_NODE_TYPES)
            for const_node in const_nodes:
                name_node = self._get_child_by_type(const_node, "identifier")
                if name_node:
                    const_name = self._get_node_text(name_node, content)
                    visibility = self._extract_visibility(const_node, content)

                    entity = CoreCodeEntity(
                        id=f"const:{const_name}:{file_path}",
                        name=const_name,
                        type=EntityType.CONSTANT,
                        language="rust",
                        file_path=str(file_path),
                        line_number=const_node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=const_node.start_point[1],
                        column_end=const_node.end_point[1],
                        visibility=visibility,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"node_type": "const"},
                    )
                    entities.append(entity)

            # Extract static items
            static_nodes = self._find_nodes_by_type(tree, self.STATIC_NODE_TYPES)
            for static_node in static_nodes:
                name_node = self._get_child_by_type(static_node, "identifier")
                if name_node:
                    static_name = self._get_node_text(name_node, content)
                    visibility = self._extract_visibility(static_node, content)

                    entity = CoreCodeEntity(
                        id=f"static:{static_name}:{file_path}",
                        name=static_name,
                        type=EntityType.VARIABLE,
                        language="rust",
                        file_path=str(file_path),
                        line_number=static_node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=static_node.start_point[1],
                        column_end=static_node.end_point[1],
                        visibility=visibility,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        is_static=True,
                        properties={"node_type": "static"},
                    )
                    entities.append(entity)

            return entities

        def _extract_visibility(self, node: Node, content: bytes) -> str:
            """Extract visibility modifier from a node."""
            node_text = self._get_node_text(node, content)
            if node_text.startswith("pub"):
                return "public"
            return "private"

        def _extract_trait_implementations(
            self, tree: Tree, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Extract trait implementation relationships."""
            relationships: list[EntityRelation] = []
            impl_nodes = self._find_nodes_by_type(tree, self.IMPL_NODE_TYPES)

            for impl_node in impl_nodes:
                impl_text = self._get_node_text(impl_node, tree.root_node.text or b"")

                if " for " in impl_text:
                    # This is a trait implementation
                    impl_entity = self._find_entity_by_location(entities, impl_node)
                    if impl_entity and "trait_name" in impl_entity.properties:
                        trait_name = impl_entity.properties["trait_name"]
                        type_name = impl_entity.properties["type_name"]

                        # Create relationship from type to trait
                        relationship = EntityRelation(
                            # Could also be enum, etc.
                            source_id=f"struct:{type_name}:{file_path}",
                            target_id=f"trait:{trait_name}:{file_path}",
                            type=RelationType.IMPLEMENTS,
                            location=self._node_to_location(impl_node, file_path),
                        )
                        relationships.append(relationship)

            return relationships

        def _extract_function_calls(
            self, tree: Tree, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Extract function call relationships."""
            relationships: list[EntityRelation] = []
            call_nodes = self._find_nodes_by_type(tree, {"call_expression"})

            for call_node in call_nodes:
                # Find the containing function
                containing_function = self._find_containing_method(
                    call_node, entities, self.FUNCTION_NODE_TYPES
                )
                if not containing_function:
                    continue

                # Extract function name
                function_node = self._get_child_by_type(call_node, "identifier")
                if not function_node:
                    function_node = self._get_child_by_type(call_node, "field_expression")

                if function_node:
                    function_name = self._get_node_text(function_node, tree.root_node.text or b"")

                    # Create relationship
                    relationship = EntityRelation(
                        source_id=containing_function.id,
                        target_id=f"fn:{function_name}:unknown",
                        type=RelationType.CALLS,
                        location=self._node_to_location(call_node, file_path),
                        properties={"function_name": function_name},
                    )
                    relationships.append(relationship)

            return relationships

        def _extract_use_relationships(
            self, tree: Tree, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Extract use/import relationships."""
            relationships: list[EntityRelation] = []

            # Find file entity (or create one)
            file_entity = next((e for e in entities if e.type == EntityType.FILE), None)
            if not file_entity:
                file_entity = CoreCodeEntity(
                    id=f"file:{file_path}",
                    name=file_path.name,
                    type=EntityType.FILE,
                    language="rust",
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

            # Create relationships from file to use declarations
            use_entities = [e for e in entities if e.type == EntityType.IMPORT]
            for use_entity in use_entities:
                relationship = EntityRelation(
                    source_id=file_entity.id,
                    target_id=use_entity.id,
                    type=RelationType.IMPORTS,
                    location=SourceLocation(
                        file_path=str(file_path),
                        line_start=use_entity.line_number,
                        line_end=use_entity.line_number,
                        column_start=use_entity.column_start or 0,
                        column_end=use_entity.column_end or 0,
                    ),
                )
                relationships.append(relationship)

            return relationships

    else:
        # Fallback implementations when Tree-sitter is not available
        async def extract_entities(self, tree: Any, file_path: Path) -> list[CoreCodeEntity]:
            """Fallback entity extraction.

            Args:
                tree (Any): The tree parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[CoreCodeEntity]: The list[corecodeentity] result."""
            return []

        async def extract_relationships(
            self, tree: Any, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Fallback relationship extraction.

            Args:
                tree (Any): The tree parameter.
                entities (List[CoreCodeEntity]): The entities parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[EntityRelation]: The list[entityrelation] result.

            Example:
                >>> await extract_relationships("tree_value", "entities_value", "file_path_value")
            """
            return []
