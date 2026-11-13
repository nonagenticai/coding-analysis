"""
Go syntax parsing and AST analysis module.

Provides Tree-sitter AST parsing and regex fallback parsing for Go source code.
Handles extraction of all major Go language constructs including functions,
structs, interfaces, types, constants, and variables.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    EntityType,
)

try:
    from tree_sitter import Node, Tree

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    # Tree/Node types not available - use Any in method signatures

logger = logging.getLogger(__name__)


class GoSyntaxParser:
    """Go syntax parser using Tree-sitter with regex fallback."""

    # Go-specific node types for Tree-sitter
    FUNCTION_NODE_TYPES = {"function_declaration", "method_declaration"}
    STRUCT_NODE_TYPES = {"type_declaration"}
    INTERFACE_NODE_TYPES = {"interface_type"}
    VAR_NODE_TYPES = {"var_declaration", "short_var_declaration"}
    CONST_NODE_TYPES = {"const_declaration"}
    TYPE_NODE_TYPES = {"type_alias", "type_declaration"}

    def __init__(self):
        """Initialize Go syntax parser."""
        self.parser = None

    def set_parser(self, parser: Any) -> None:
        """Set the Tree-sitter parser instance.

        Args:
            parser: Tree-sitter parser instance
        """
        self.parser = parser

    def _find_nodes_by_type(self, tree: Tree | Any, node_types: set) -> list[Node]:
        """Find all nodes of specified types in the AST.

        Args:
            tree: Tree-sitter AST tree
            node_types: Set of node type names to find

        Returns:
            List of matching nodes
        """
        if not TREE_SITTER_AVAILABLE:
            return []

        nodes = []

        def traverse(node: Node) -> None:
            if node.type in node_types:
                nodes.append(node)
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return nodes

    def _get_node_text(self, node: Node, content: bytes) -> str | bytes:
        """Extract text content from a Tree-sitter node.

        Args:
            node: Tree-sitter node
            content: Source code bytes

        Returns:
            Text content of the node
        """
        if not TREE_SITTER_AVAILABLE:
            return ""
        return content[node.start_byte : node.end_byte]

    async def extract_functions_and_methods_regex(
        self, content: str, file_path: Path
    ) -> list[CoreCodeEntity]:
        """Extract functions and methods using regex patterns.

        Args:
            content: Source code content
            file_path: Path to the source file

        Returns:
            List of function and method entities
        """
        entities = []

        # Extract functions and methods
        func_pattern = r"func\s+(?:\((\w+)\s+[*]?(\w+)\)\s+)?(\w+)\s*\(([^)]*)\)\s*(?:\(([^)]*)\)|([^{]*))?\s*{"
        for match in re.finditer(func_pattern, content):
            receiver_name = match.group(1)
            receiver_type = match.group(2)
            func_name = match.group(3)
            params = match.group(4) or ""
            return_multi = match.group(5)  # Multiple returns
            return_single = match.group(6)  # Single return
            returns = return_multi or return_single or ""
            line_num = content[: match.start()].count("\n") + 1

            # Determine if it's a method or function
            if receiver_name:
                entity_type = EntityType.FUNCTION  # Methods are functions in Go
                full_name = f"{receiver_type}.{func_name}"
                signature = f"({receiver_name} {receiver_type}) {func_name}({params})"
            else:
                entity_type = EntityType.FUNCTION
                full_name = func_name
                signature = f"{func_name}({params})"

            if returns:
                signature += f" {returns.strip()}"

            entities.append(
                CoreCodeEntity(
                    id=f"function:{full_name}:{file_path}:{line_num}",
                    name=func_name,
                    type=entity_type,
                    language="go",
                    file_path=str(file_path),
                    line_number=line_num,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility="public" if func_name[0].isupper() else "private",
                    code_snippet=None,
                    docstring=None,
                    signature=signature,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={
                        "receiver_type": receiver_type,
                        "receiver_name": receiver_name,
                        "is_method": bool(receiver_name),
                        "returns": returns.strip() if returns else "void",
                    },
                )
            )

        return entities

    async def extract_structs_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract struct definitions using regex patterns.

        Args:
            content: Source code content
            file_path: Path to the source file

        Returns:
            List of struct entities
        """
        entities = []

        # Extract structs
        struct_pattern = r"type\s+(\w+)\s+struct\s*{"
        for match in re.finditer(struct_pattern, content):
            struct_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"struct:{struct_name}:{file_path}",
                    name=struct_name,
                    type=EntityType.CLASS,  # Structs are similar to classes
                    language="go",
                    file_path=str(file_path),
                    line_number=line_num,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility="public" if struct_name[0].isupper() else "private",
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={"struct_type": True},
                )
            )

        return entities

    async def extract_interfaces_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract interface definitions using regex patterns.

        Args:
            content: Source code content
            file_path: Path to the source file

        Returns:
            List of interface entities
        """
        entities = []

        # Extract interfaces
        interface_pattern = r"type\s+(\w+)\s+interface\s*{"
        for match in re.finditer(interface_pattern, content):
            interface_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"interface:{interface_name}:{file_path}",
                    name=interface_name,
                    type=EntityType.INTERFACE,
                    language="go",
                    file_path=str(file_path),
                    line_number=line_num,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility="public" if interface_name[0].isupper() else "private",
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

    async def extract_types_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract type aliases using regex patterns.

        Args:
            content: Source code content
            file_path: Path to the source file

        Returns:
            List of type alias entities
        """
        entities = []

        # Extract type aliases
        type_alias_pattern = r"type\s+(\w+)\s+(?!struct|interface)(\w+)"
        for match in re.finditer(type_alias_pattern, content):
            type_name = match.group(1)
            base_type = match.group(2)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"type:{type_name}:{file_path}",
                    name=type_name,
                    type=EntityType.TYPE_ALIAS,
                    language="go",
                    file_path=str(file_path),
                    line_number=line_num,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility="public" if type_name[0].isupper() else "private",
                    code_snippet=None,
                    docstring=None,
                    signature=f"type {type_name} {base_type}",
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={"base_type": base_type},
                )
            )

        return entities

    async def extract_constants_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract constant definitions using regex patterns.

        Args:
            content: Source code content
            file_path: Path to the source file

        Returns:
            List of constant entities
        """
        entities = []

        # Extract constants
        const_pattern = r"const\s+(?:\(([^)]+)\)|(\w+)\s*(?:=|\s))"
        for match in re.finditer(const_pattern, content):
            if match.group(1):  # Const block
                const_block = match.group(1)
                block_start_line = content[: match.start()].count("\n") + 1
                for const_match in re.finditer(r"(\w+)\s*(?:=\s*([^\n]+))?", const_block):
                    const_name = const_match.group(1)
                    const_value = const_match.group(2) or ""
                    line_offset = const_block[: const_match.start()].count("\n")
                    line_num = block_start_line + line_offset

                    entities.append(
                        CoreCodeEntity(
                            id=f"const:{const_name}:{file_path}:{line_num}",
                            name=const_name,
                            type=EntityType.CONSTANT,
                            language="go",
                            file_path=str(file_path),
                            line_number=line_num,
                            end_line_number=None,
                            column_start=None,
                            column_end=None,
                            visibility=("public" if const_name[0].isupper() else "private"),
                            code_snippet=None,
                            docstring=None,
                            signature=None,
                            source_code=None,
                            created_at=None,
                            updated_at=None,
                            properties={"value": const_value.strip()},
                        )
                    )
            else:  # Single const
                const_name = match.group(2)
                if const_name:
                    line_num = content[: match.start()].count("\n") + 1
                    entities.append(
                        CoreCodeEntity(
                            id=f"const:{const_name}:{file_path}:{line_num}",
                            name=const_name,
                            type=EntityType.CONSTANT,
                            language="go",
                            file_path=str(file_path),
                            line_number=line_num,
                            end_line_number=None,
                            column_start=None,
                            column_end=None,
                            visibility=("public" if const_name[0].isupper() else "private"),
                            code_snippet=None,
                            docstring=None,
                            signature=None,
                            source_code=None,
                            created_at=None,
                            updated_at=None,
                            properties={},
                        )
                    )

        return entities

    async def extract_variables_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract variable declarations using regex patterns.

        Args:
            content: Source code content
            file_path: Path to the source file

        Returns:
            List of variable entities
        """
        entities = []

        # Extract variables
        var_pattern = r"var\s+(?:\(([^)]+)\)|(\w+)\s+([^=\n]+))"
        for match in re.finditer(var_pattern, content):
            if match.group(1):  # Var block
                var_block = match.group(1)
                block_start_line = content[: match.start()].count("\n") + 1
                for var_match in re.finditer(r"(\w+)\s+([^\n=]+)", var_block):
                    var_name = var_match.group(1)
                    var_type = var_match.group(2).strip()
                    line_offset = var_block[: var_match.start()].count("\n")
                    line_num = block_start_line + line_offset

                    entities.append(
                        CoreCodeEntity(
                            id=f"var:{var_name}:{file_path}:{line_num}",
                            name=var_name,
                            type=EntityType.VARIABLE,
                            language="go",
                            file_path=str(file_path),
                            line_number=line_num,
                            end_line_number=None,
                            column_start=None,
                            column_end=None,
                            visibility="public" if var_name[0].isupper() else "private",
                            code_snippet=None,
                            docstring=None,
                            signature=None,
                            source_code=None,
                            created_at=None,
                            updated_at=None,
                            properties={"var_type": var_type},
                        )
                    )
            else:  # Single var
                var_name = match.group(2)
                var_type = match.group(3)
                if var_name and var_type:
                    line_num = content[: match.start()].count("\n") + 1
                    entities.append(
                        CoreCodeEntity(
                            id=f"var:{var_name}:{file_path}:{line_num}",
                            name=var_name,
                            type=EntityType.VARIABLE,
                            language="go",
                            file_path=str(file_path),
                            line_number=line_num,
                            end_line_number=None,
                            column_start=None,
                            column_end=None,
                            visibility="public" if var_name[0].isupper() else "private",
                            code_snippet=None,
                            docstring=None,
                            signature=None,
                            source_code=None,
                            created_at=None,
                            updated_at=None,
                            properties={"var_type": var_type.strip()},
                        )
                    )

        return entities

    async def extract_entities_tree_sitter(
        self, tree: Tree | Any, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract Go entities using Tree-sitter AST parsing.

        Provides more accurate parsing than regex fallback by using
        the Go grammar to identify language constructs.

        Args:
            tree: Tree-sitter parsed AST tree
            file_path: Source file path for entity identification
            content: Raw source code bytes for text extraction

        Returns:
            List of CoreCodeEntity objects extracted from the AST

        Raises:
            ValueError: If Tree-sitter parsing fails or content is invalid
        """
        logger.debug("Extracting entities using Tree-sitter for: %s", file_path)
        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter not available for Go parsing")
            return []

        try:
            entities = []
            logger.debug(
                "Processing Tree-sitter AST with %d nodes",
                len(list(tree.root_node.children)),
            )
        except Exception as e:
            logger.error("Error processing Tree-sitter AST for %s: %s", file_path, e)
            raise ValueError(f"Tree-sitter processing failed for {file_path}: {e}") from e

        # Extract functions and methods
        function_nodes = self._find_nodes_by_type(tree, self.FUNCTION_NODE_TYPES)
        for node in function_nodes:
            func_text = self._get_node_text(node, content)
            if isinstance(func_text, bytes):
                func_text = func_text.decode("utf-8")

            # Parse function signature
            func_match = re.search(
                r"func\s+(?:\((\w+)\s+[*]?(\w+)\)\s+)?(\w+)\s*\(([^)]*)\)\s*(?:\(([^)]*)\)|([^{]*))?",
                func_text,
            )
            if func_match:
                receiver_name = func_match.group(1)
                receiver_type = func_match.group(2)
                func_name = func_match.group(3)
                params = func_match.group(4) or ""
                return_multi = func_match.group(5)
                return_single = func_match.group(6)
                returns = return_multi or return_single or ""

                # Build signature
                if receiver_name:
                    full_name = f"{receiver_type}.{func_name}"
                    signature = f"({receiver_name} {receiver_type}) {func_name}({params})"
                else:
                    full_name = func_name
                    signature = f"{func_name}({params})"

                if returns:
                    signature += f" {returns.strip()}"

                entities.append(
                    CoreCodeEntity(
                        id=f"function:{full_name}:{file_path}:{node.start_point[0] + 1}",
                        name=func_name,
                        type=EntityType.FUNCTION,
                        language="go",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=node.end_point[0] + 1,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility="public" if func_name[0].isupper() else "private",
                        code_snippet=None,
                        docstring=None,
                        signature=signature,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={
                            "receiver_type": receiver_type,
                            "receiver_name": receiver_name,
                            "is_method": bool(receiver_name),
                            "returns": returns.strip() if returns else "void",
                        },
                    )
                )

        # Extract structs
        struct_nodes = self._find_nodes_by_type(tree, self.STRUCT_NODE_TYPES)
        for node in struct_nodes:
            struct_text = self._get_node_text(node, content)
            if isinstance(struct_text, bytes):
                struct_text = struct_text.decode("utf-8")

            struct_match = re.search(r"type\s+(\w+)\s+struct", struct_text)
            if struct_match:
                struct_name = struct_match.group(1)
                entities.append(
                    CoreCodeEntity(
                        id=f"struct:{struct_name}:{file_path}",
                        name=struct_name,
                        type=EntityType.CLASS,  # Structs are similar to classes
                        language="go",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=node.end_point[0] + 1,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility="public" if struct_name[0].isupper() else "private",
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"struct_type": True},
                    )
                )

        # Extract interfaces
        interface_nodes = self._find_nodes_by_type(tree, self.INTERFACE_NODE_TYPES)
        for node in interface_nodes:
            interface_text = self._get_node_text(node, content)
            if isinstance(interface_text, bytes):
                interface_text = interface_text.decode("utf-8")

            # Get parent node to find type name
            parent = node.parent
            if parent and parent.type == "type_declaration":
                parent_text = self._get_node_text(parent, content)
                if isinstance(parent_text, bytes):
                    parent_text = parent_text.decode("utf-8")

                interface_match = re.search(r"type\s+(\w+)\s+interface", parent_text)
                if interface_match:
                    interface_name = interface_match.group(1)
                    entities.append(
                        CoreCodeEntity(
                            id=f"interface:{interface_name}:{file_path}",
                            name=interface_name,
                            type=EntityType.INTERFACE,
                            language="go",
                            file_path=str(file_path),
                            line_number=parent.start_point[0] + 1,
                            end_line_number=parent.end_point[0] + 1,
                            column_start=parent.start_point[1],
                            column_end=parent.end_point[1],
                            visibility=("public" if interface_name[0].isupper() else "private"),
                            code_snippet=None,
                            docstring=None,
                            signature=None,
                            source_code=None,
                            created_at=None,
                            updated_at=None,
                            properties={"interface_type": True},
                        )
                    )

        # Extract constants
        const_nodes = self._find_nodes_by_type(tree, self.CONST_NODE_TYPES)
        for node in const_nodes:
            const_text = self._get_node_text(node, content)
            if isinstance(const_text, bytes):
                const_text = const_text.decode("utf-8")

            # Extract constant names
            const_matches = re.finditer(r"(\w+)\s*(?:=\s*[^\n]+)?", const_text)
            for const_match in const_matches:
                const_name = const_match.group(1)
                if const_name != "const":  # Skip the const keyword itself
                    entities.append(
                        CoreCodeEntity(
                            id=f"const:{const_name}:{file_path}:{node.start_point[0] + 1}",
                            name=const_name,
                            type=EntityType.CONSTANT,
                            language="go",
                            file_path=str(file_path),
                            line_number=node.start_point[0] + 1,
                            end_line_number=None,
                            column_start=node.start_point[1],
                            column_end=node.end_point[1],
                            visibility=("public" if const_name[0].isupper() else "private"),
                            code_snippet=None,
                            docstring=None,
                            signature=None,
                            source_code=None,
                            created_at=None,
                            updated_at=None,
                            properties={},
                        )
                    )

        # Extract variables
        var_nodes = self._find_nodes_by_type(tree, self.VAR_NODE_TYPES)
        for node in var_nodes:
            var_text = self._get_node_text(node, content)
            if isinstance(var_text, bytes):
                var_text = var_text.decode("utf-8")

            # Extract variable names
            if node.type == "var_declaration":
                # Strip leading "var" keyword before matching names
                stripped_text = re.sub(r"^\s*var\s+", "", var_text)
                var_matches = re.finditer(r"(\w+)\s+([^=\n,]+)", stripped_text)
                for var_match in var_matches:
                    var_name = var_match.group(1)
                    if var_name != "var":  # Skip the var keyword itself
                        var_type = var_match.group(2).strip()
                        entities.append(
                            CoreCodeEntity(
                                id=f"var:{var_name}:{file_path}:{node.start_point[0] + 1}",
                                name=var_name,
                                type=EntityType.VARIABLE,
                                language="go",
                                file_path=str(file_path),
                                line_number=node.start_point[0] + 1,
                                end_line_number=None,
                                column_start=node.start_point[1],
                                column_end=node.end_point[1],
                                visibility=("public" if var_name[0].isupper() else "private"),
                                code_snippet=None,
                                docstring=None,
                                signature=None,
                                source_code=None,
                                created_at=None,
                                updated_at=None,
                                properties={"var_type": var_type},
                            )
                        )
            elif node.type == "short_var_declaration":
                # Handle := syntax
                short_var_match = re.search(r"(\w+)\s*:=", var_text)
                if short_var_match:
                    var_name = short_var_match.group(1)
                    entities.append(
                        CoreCodeEntity(
                            id=f"var:{var_name}:{file_path}:{node.start_point[0] + 1}",
                            name=var_name,
                            type=EntityType.VARIABLE,
                            language="go",
                            file_path=str(file_path),
                            line_number=node.start_point[0] + 1,
                            end_line_number=None,
                            column_start=node.start_point[1],
                            column_end=node.end_point[1],
                            visibility="public" if var_name[0].isupper() else "private",
                            code_snippet=None,
                            docstring=None,
                            signature=None,
                            source_code=None,
                            created_at=None,
                            updated_at=None,
                            properties={"short_declaration": True},
                        )
                    )

        # Extract type aliases and definitions
        type_nodes = self._find_nodes_by_type(tree, self.TYPE_NODE_TYPES)
        for node in type_nodes:
            type_text = self._get_node_text(node, content)
            if isinstance(type_text, bytes):
                type_text = type_text.decode("utf-8")

            # Skip structs and interfaces (already handled)
            if "struct" not in type_text and "interface" not in type_text:
                type_match = re.search(r"type\s+(\w+)\s+([^\n]+)", type_text)
                if type_match:
                    type_name = type_match.group(1)
                    base_type = type_match.group(2).strip()
                    entities.append(
                        CoreCodeEntity(
                            id=f"type:{type_name}:{file_path}",
                            name=type_name,
                            type=EntityType.TYPE_ALIAS,
                            language="go",
                            file_path=str(file_path),
                            line_number=node.start_point[0] + 1,
                            end_line_number=None,
                            column_start=node.start_point[1],
                            column_end=node.end_point[1],
                            visibility=("public" if type_name[0].isupper() else "private"),
                            code_snippet=None,
                            docstring=None,
                            signature=f"type {type_name} {base_type}",
                            source_code=None,
                            created_at=None,
                            updated_at=None,
                            properties={"base_type": base_type},
                        )
                    )

        return entities
