"""
Go dependency analysis module.

Handles import analysis, dependency extraction, and relationship building
for Go source code. Provides functionality for analyzing Go import statements
and building dependency graphs.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    EntityRelation,
    EntityType,
    RelationType,
)

try:
    from tree_sitter import Node, Tree

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    # Tree/Node types not available - use Any in method signatures

logger = logging.getLogger(__name__)


class GoDependencyAnalyzer:
    """Go dependency analysis and import handling functionality."""

    # Go-specific node types for Tree-sitter
    IMPORT_NODE_TYPES = {"import_declaration", "import_spec"}

    def __init__(self):
        """Initialize Go dependency analyzer."""
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

    async def extract_imports_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract import statements using regex patterns.

        Args:
            content: Source code content
            file_path: Path to the source file

        Returns:
            List of import entities
        """
        entities = []

        # Extract imports
        import_pattern = r'import\s+(?:\(([^)]+)\)|"([^"]+)")'
        for match in re.finditer(import_pattern, content):
            if match.group(1):  # Multiple imports
                imports_text = match.group(1)
                for imp_match in re.finditer(r'"([^"]+)"', imports_text):
                    import_path = imp_match.group(1)
                    line_num = content[: imp_match.start()].count("\n") + 1
                    entities.append(
                        CoreCodeEntity(
                            id=f"import:{import_path}:{file_path}:{line_num}",
                            name=import_path,
                            type=EntityType.IMPORT,
                            language="go",
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
                            properties={"import_path": import_path},
                        )
                    )
            else:  # Single import
                import_path = match.group(2)
                line_num = content[: match.start()].count("\n") + 1
                entities.append(
                    CoreCodeEntity(
                        id=f"import:{import_path}:{file_path}:{line_num}",
                        name=import_path,
                        type=EntityType.IMPORT,
                        language="go",
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
                        properties={"import_path": import_path},
                    )
                )

        return entities

    async def extract_imports_tree_sitter(
        self, tree: Tree | Any, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract import statements using Tree-sitter AST parsing.

        Args:
            tree: Tree-sitter parsed AST tree
            file_path: Source file path for entity identification
            content: Raw source code bytes for text extraction

        Returns:
            List of import entities

        Raises:
            ValueError: If Tree-sitter parsing fails or content is invalid
        """
        logger.debug("Extracting imports using Tree-sitter for: %s", file_path)
        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter not available for Go import parsing")
            return []

        try:
            entities = []
            logger.debug("Processing Tree-sitter AST for import declarations")
        except Exception as e:
            logger.error("Error processing Tree-sitter AST for imports in %s: %s", file_path, e)
            raise ValueError(f"Tree-sitter import processing failed for {file_path}: {e}") from e

        # Extract imports
        import_nodes = self._find_nodes_by_type(tree, self.IMPORT_NODE_TYPES)
        for node in import_nodes:
            import_text = self._get_node_text(node, content)
            if isinstance(import_text, bytes):
                import_text = import_text.decode("utf-8")

            # Extract import path from quotes
            import_match = re.search(r'"([^"]+)"', import_text)
            if import_match:
                import_path = import_match.group(1)
                entities.append(
                    CoreCodeEntity(
                        id=f"import:{import_path}:{file_path}:{node.start_point[0] + 1}",
                        name=import_path,
                        type=EntityType.IMPORT,
                        language="go",
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
                        properties={"import_path": import_path},
                    )
                )

        return entities

    async def build_import_relationships(
        self, entities: list[CoreCodeEntity], file_path: Path
    ) -> list[EntityRelation]:
        """Build relationships between packages and their imports.

        Args:
            entities: List of extracted entities
            file_path: Source file path for context

        Returns:
            List of import relationships
        """
        logger.debug(
            "Building import relationships for %d entities in: %s",
            len(entities),
            file_path,
        )
        relationships: list[EntityRelation] = []

        # Get package and import entities
        package_entities = [e for e in entities if e.type == EntityType.PACKAGE]
        import_entities = [e for e in entities if e.type == EntityType.IMPORT]

        # Import relationships
        for import_entity in import_entities:
            # Import is used by the package
            if package_entities:
                relationships.append(
                    EntityRelation(
                        source_id=package_entities[0].id,
                        target_id=import_entity.id,
                        type=RelationType.IMPORTS,
                        properties={"relationship_context": "package_import"},
                    )
                )

        return relationships

    async def build_method_relationships(
        self, entities: list[CoreCodeEntity], file_path: Path
    ) -> list[EntityRelation]:
        """Build relationships between types and their methods.

        Args:
            entities: List of extracted entities
            file_path: Source file path for context

        Returns:
            List of method definition relationships
        """
        logger.debug(
            "Building method relationships for %d entities in: %s",
            len(entities),
            file_path,
        )
        relationships: list[EntityRelation] = []

        # Methods belong to their receiver types
        struct_and_interface_entities = [
            e for e in entities if e.type in [EntityType.CLASS, EntityType.INTERFACE]
        ]
        method_entities = [
            e for e in entities if e.type == EntityType.FUNCTION and e.properties.get("is_method")
        ]

        for method in method_entities:
            receiver_type = method.properties.get("receiver_type")
            if receiver_type:
                for struct_or_interface in struct_and_interface_entities:
                    if struct_or_interface.name == receiver_type:
                        relationships.append(
                            EntityRelation(
                                source_id=struct_or_interface.id,
                                target_id=method.id,
                                type=RelationType.DEFINES,
                                properties={"relationship_context": "method_definition"},
                            )
                        )

        return relationships

    async def build_package_relationships(
        self, entities: list[CoreCodeEntity], file_path: Path
    ) -> list[EntityRelation]:
        """Build relationships between packages and contained entities.

        Args:
            entities: List of extracted entities
            file_path: Source file path for context

        Returns:
            List of containment relationships
        """
        logger.debug(
            "Building package relationships for %d entities in: %s",
            len(entities),
            file_path,
        )
        relationships: list[EntityRelation] = []

        # Get package entities
        package_entities = [e for e in entities if e.type == EntityType.PACKAGE]

        # Package contains other entities
        for entity in entities:
            if entity.type != EntityType.PACKAGE and package_entities:
                package = package_entities[0]  # Go files have one package
                relationships.append(
                    EntityRelation(
                        source_id=package.id,
                        target_id=entity.id,
                        type=RelationType.CONTAINS,
                        properties={"relationship_context": "package_member"},
                    )
                )

        return relationships

    def classify_import_type(self, import_path: str) -> str:
        """Classify the type of import (standard library, external, local).

        Args:
            import_path: The import path string

        Returns:
            Import classification: 'standard', 'external', or 'local'
        """
        # Standard library packages (common ones)
        standard_packages = {
            "fmt",
            "os",
            "io",
            "net",
            "http",
            "json",
            "time",
            "strings",
            "strconv",
            "context",
            "sync",
            "log",
            "errors",
            "sort",
            "math",
            "crypto",
            "encoding",
            "database",
            "html",
            "text",
            "regexp",
            "path",
            "filepath",
            "bufio",
            "bytes",
            "reflect",
            "runtime",
            "syscall",
            "unsafe",
            "archive",
            "compress",
            "container",
            "go",
            "hash",
            "image",
            "index",
            "mime",
            "plugin",
            "testing",
            "unicode",
        }

        # Check if it's a standard library package
        if "/" not in import_path:
            # Single component, likely standard library
            if import_path in standard_packages:
                return "standard"
        else:
            # Check if it starts with standard library paths
            parts = import_path.split("/")
            if parts[0] in standard_packages:
                return "standard"

        # Check for local imports (relative paths or no domain)
        if import_path.startswith("./") or import_path.startswith("../"):
            return "local"

        # Check if it contains a domain (external package)
        if "." in import_path.split("/")[0]:
            return "external"

        # Default to local for ambiguous cases
        return "local"

    def extract_import_alias(self, import_text: str) -> str | None:
        """Extract import alias if present.

        Args:
            import_text: The full import statement text

        Returns:
            Import alias if present, None otherwise
        """
        # Look for alias pattern: alias "package"
        alias_pattern = r'(\w+)\s+"[^"]+"'
        alias_match = re.search(alias_pattern, import_text)

        if alias_match:
            return alias_match.group(1)

        return None

    def is_dot_import(self, import_text: str) -> bool:
        """Check if this is a dot import (imports into current namespace).

        Args:
            import_text: The full import statement text

        Returns:
            True if this is a dot import
        """
        return re.search(r'\.\s+"[^"]+"', import_text) is not None

    def is_blank_import(self, import_text: str) -> bool:
        """Check if this is a blank import (for side effects only).

        Args:
            import_text: The full import statement text

        Returns:
            True if this is a blank import
        """
        return re.search(r'_\s+"[^"]+"', import_text) is not None

    def get_package_name_from_import(self, import_path: str) -> str:
        """Get the package name from an import path.

        Args:
            import_path: The import path string

        Returns:
            The package name (last component of path)
        """
        # Handle special cases
        if import_path.endswith("/"):
            import_path = import_path[:-1]

        # Get the last component
        if "/" in import_path:
            return import_path.split("/")[-1]

        return import_path

    def analyze_import_dependencies(self, entities: list[CoreCodeEntity]) -> dict:
        """Analyze import dependencies and provide statistics.

        Args:
            entities: List of entities to analyze

        Returns:
            Dictionary with dependency analysis results
        """
        import_entities = [e for e in entities if e.type == EntityType.IMPORT]

        # Classify imports
        standard_imports = []
        external_imports = []
        local_imports = []

        for import_entity in import_entities:
            import_path = import_entity.properties.get("import_path", "")
            import_type = self.classify_import_type(import_path)

            if import_type == "standard":
                standard_imports.append(import_path)
            elif import_type == "external":
                external_imports.append(import_path)
            else:
                local_imports.append(import_path)

        return {
            "total_imports": len(import_entities),
            "standard_library": {
                "count": len(standard_imports),
                "imports": standard_imports,
            },
            "external_packages": {
                "count": len(external_imports),
                "imports": external_imports,
            },
            "local_packages": {"count": len(local_imports), "imports": local_imports},
            "import_complexity": len(set(standard_imports + external_imports + local_imports)),
        }
