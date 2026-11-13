"""Rust syntax parsing and analysis module.

This module handles the core syntax parsing functionality for Rust code,
including both Tree-sitter AST parsing and regex fallback parsing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    EntityType,
)

# Import Tree-sitter components if available
try:
    from tree_sitter import Node, Tree  # noqa: F401

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    # No type aliases needed - we use Any for method signatures


class RustSyntaxParser:
    """Handles Rust syntax parsing using Tree-sitter and regex fallback."""

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

    def __init__(self, tree_sitter_processor=None):
        """Initialize the syntax parser.

        Args:
            tree_sitter_processor: Optional tree-sitter processor instance
        """
        self.tree_sitter_processor = tree_sitter_processor

    # Regex-based parsing methods
    def extract_modules_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract module declarations using regex."""
        entities: list[CoreCodeEntity] = []
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

    def extract_structs_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract struct definitions using regex."""
        entities: list[CoreCodeEntity] = []
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

    def extract_enums_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract enum definitions using regex."""
        entities: list[CoreCodeEntity] = []
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

    def extract_traits_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract trait definitions using regex."""
        entities: list[CoreCodeEntity] = []
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

    def extract_functions_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract function definitions using regex."""
        entities: list[CoreCodeEntity] = []
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

    def extract_impl_blocks_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract impl blocks using regex."""
        entities: list[CoreCodeEntity] = []
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

    def extract_macros_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract macro definitions using regex."""
        entities: list[CoreCodeEntity] = []
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

        def extract_modules_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract module declarations using Tree-sitter."""
            entities: list[CoreCodeEntity] = []
            if not self.tree_sitter_processor:
                return entities

            mod_nodes = self.tree_sitter_processor._find_nodes_by_type(tree, self.MOD_NODE_TYPES)

            for mod_node in mod_nodes:
                name_node = self.tree_sitter_processor._get_child_by_type(mod_node, "identifier")
                if not name_node:
                    continue

                mod_name = self.tree_sitter_processor._get_node_text(name_node, content)
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

        def extract_structs_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract struct definitions using Tree-sitter."""
            entities: list[CoreCodeEntity] = []
            if not self.tree_sitter_processor:
                return entities

            struct_nodes = self.tree_sitter_processor._find_nodes_by_type(
                tree, self.STRUCT_NODE_TYPES
            )

            for struct_node in struct_nodes:
                name_node = self.tree_sitter_processor._get_child_by_type(
                    struct_node, "type_identifier"
                )
                if not name_node:
                    continue

                struct_name = self.tree_sitter_processor._get_node_text(name_node, content)
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

        def extract_enums_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract enum definitions using Tree-sitter."""
            entities: list[CoreCodeEntity] = []
            if not self.tree_sitter_processor:
                return entities

            enum_nodes = self.tree_sitter_processor._find_nodes_by_type(tree, self.ENUM_NODE_TYPES)

            for enum_node in enum_nodes:
                name_node = self.tree_sitter_processor._get_child_by_type(
                    enum_node, "type_identifier"
                )
                if not name_node:
                    continue

                enum_name = self.tree_sitter_processor._get_node_text(name_node, content)
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

        def extract_traits_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract trait definitions using Tree-sitter."""
            entities: list[CoreCodeEntity] = []
            if not self.tree_sitter_processor:
                return entities

            trait_nodes = self.tree_sitter_processor._find_nodes_by_type(
                tree, self.TRAIT_NODE_TYPES
            )

            for trait_node in trait_nodes:
                name_node = self.tree_sitter_processor._get_child_by_type(
                    trait_node, "type_identifier"
                )
                if not name_node:
                    continue

                trait_name = self.tree_sitter_processor._get_node_text(name_node, content)
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

        def extract_functions_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract function definitions using Tree-sitter."""
            entities: list[CoreCodeEntity] = []
            if not self.tree_sitter_processor:
                return entities

            function_nodes = self.tree_sitter_processor._find_nodes_by_type(
                tree, self.FUNCTION_NODE_TYPES
            )

            for fn_node in function_nodes:
                name_node = self.tree_sitter_processor._get_child_by_type(fn_node, "identifier")
                if not name_node:
                    continue

                fn_name = self.tree_sitter_processor._get_node_text(name_node, content)
                visibility = self._extract_visibility(fn_node, content)
                is_async = "async" in self.tree_sitter_processor._get_node_text(fn_node, content)

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

        def extract_impl_blocks_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract impl blocks using Tree-sitter."""
            entities: list[CoreCodeEntity] = []
            if not self.tree_sitter_processor:
                return entities

            impl_nodes = self.tree_sitter_processor._find_nodes_by_type(tree, self.IMPL_NODE_TYPES)

            for impl_node in impl_nodes:
                impl_text = self.tree_sitter_processor._get_node_text(impl_node, content)

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

        def extract_constants_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract const and static declarations using Tree-sitter."""
            entities: list[CoreCodeEntity] = []
            if not self.tree_sitter_processor:
                return entities

            # Extract const items
            const_nodes = self.tree_sitter_processor._find_nodes_by_type(
                tree, self.CONST_NODE_TYPES
            )
            for const_node in const_nodes:
                name_node = self.tree_sitter_processor._get_child_by_type(const_node, "identifier")
                if name_node:
                    const_name = self.tree_sitter_processor._get_node_text(name_node, content)
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
            static_nodes = self.tree_sitter_processor._find_nodes_by_type(
                tree, self.STATIC_NODE_TYPES
            )
            for static_node in static_nodes:
                name_node = self.tree_sitter_processor._get_child_by_type(static_node, "identifier")
                if name_node:
                    static_name = self.tree_sitter_processor._get_node_text(name_node, content)
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

        def _extract_visibility(self, node: Any, content: bytes) -> str:
            """Extract visibility modifier from a node."""
            if not self.tree_sitter_processor:
                return "private"

            node_text = self.tree_sitter_processor._get_node_text(node, content)
            if node_text.startswith("pub"):
                return "public"
            return "private"

    else:
        # Fallback methods when Tree-sitter is not available
        def extract_modules_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Fallback module extraction."""
            return []

        def extract_structs_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Fallback struct extraction."""
            return []

        def extract_enums_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Fallback enum extraction."""
            return []

        def extract_traits_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Fallback trait extraction."""
            return []

        def extract_functions_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Fallback function extraction."""
            return []

        def extract_impl_blocks_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Fallback impl block extraction."""
            return []

        def extract_constants_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Fallback constants extraction."""
            return []

        def _extract_visibility(self, node: Any, content: bytes) -> str:
            """Fallback visibility extraction."""
            return "private"
