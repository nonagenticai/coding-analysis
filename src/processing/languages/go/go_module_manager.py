"""
Go module and package management module.

Handles Go package declarations, module management, and package-level operations.
Provides functionality for extracting and managing Go package information.
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


class GoModuleManager:
    """Go module and package management functionality."""

    # Go-specific node types for Tree-sitter
    PACKAGE_NODE_TYPES = {"package_clause"}

    def __init__(self):
        """Initialize Go module manager."""
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

    async def extract_package_regex(self, content: str, file_path: Path) -> CoreCodeEntity | None:
        """Extract package declaration using regex patterns.

        Args:
            content: Source code content
            file_path: Path to the source file

        Returns:
            Package entity if found, None otherwise
        """
        # Extract package declaration
        package_pattern = r"package\s+(\w+)"
        package_match = re.search(package_pattern, content)

        if package_match:
            package_name = package_match.group(1)
            return CoreCodeEntity(
                id=f"package:{package_name}:{file_path}",
                name=package_name,
                type=EntityType.PACKAGE,
                language="go",
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
                properties={"package_name": package_name},
            )

        return None

    async def extract_package_tree_sitter(
        self, tree: Tree | Any, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract package declarations using Tree-sitter AST parsing.

        Args:
            tree: Tree-sitter parsed AST tree
            file_path: Source file path for entity identification
            content: Raw source code bytes for text extraction

        Returns:
            List of package entities (typically one per Go file)

        Raises:
            ValueError: If Tree-sitter parsing fails or content is invalid
        """
        logger.debug("Extracting packages using Tree-sitter for: %s", file_path)
        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter not available for Go package parsing")
            return []

        try:
            entities = []
            logger.debug("Processing Tree-sitter AST for package declarations")
        except Exception as e:
            logger.error("Error processing Tree-sitter AST for packages in %s: %s", file_path, e)
            raise ValueError(f"Tree-sitter package processing failed for {file_path}: {e}") from e

        # Extract packages
        package_nodes = self._find_nodes_by_type(tree, self.PACKAGE_NODE_TYPES)
        node: Node  # Type annotation to show Node is used
        for node in package_nodes:
            node_text = self._get_node_text(node, content)
            if isinstance(node_text, bytes):
                package_name = node_text.decode("utf-8").split()[-1]
            else:
                package_name = node_text.split()[-1]
            entities.append(
                CoreCodeEntity(
                    id=f"package:{package_name}:{file_path}",
                    name=package_name,
                    type=EntityType.PACKAGE,
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
                    properties={"package_name": package_name},
                )
            )

        return entities

    def get_package_name_from_content(self, content: str) -> str | None:
        """Extract package name from Go source content.

        Args:
            content: Go source code content

        Returns:
            Package name if found, None otherwise
        """
        package_pattern = r"package\s+(\w+)"
        package_match = re.search(package_pattern, content)

        if package_match:
            return package_match.group(1)

        return None

    def get_package_name_from_path(self, file_path: Path) -> str | None:
        """Infer package name from file path following Go conventions.

        Args:
            file_path: Path to the Go source file

        Returns:
            Inferred package name based on directory structure
        """
        # In Go, package name typically matches the directory name
        # unless it's main package
        parent_dir = file_path.parent.name

        # Special cases
        if parent_dir in ["cmd", "main"]:
            return "main"

        # Use directory name as package name
        return parent_dir if parent_dir else None

    def is_main_package(self, content: str) -> bool:
        """Check if the Go file belongs to the main package.

        Args:
            content: Go source code content

        Returns:
            True if this is a main package file
        """
        package_name = self.get_package_name_from_content(content)
        return package_name == "main"

    def is_test_file(self, file_path: Path) -> bool:
        """Check if the Go file is a test file.

        Args:
            file_path: Path to the Go source file

        Returns:
            True if this is a test file (_test.go)
        """
        return file_path.name.endswith("_test.go")

    def get_module_info_from_go_mod(self, project_root: Path) -> dict | None:
        """Extract module information from go.mod file.

        Args:
            project_root: Root directory to search for go.mod

        Returns:
            Dictionary with module information if go.mod exists
        """
        go_mod_path = project_root / "go.mod"

        if not go_mod_path.exists():
            # Try to find go.mod in parent directories
            current = project_root
            while current.parent != current:
                go_mod_path = current / "go.mod"
                if go_mod_path.exists():
                    break
                current = current.parent
            else:
                return None

        try:
            content = go_mod_path.read_text(encoding="utf-8")

            # Extract module name
            module_pattern = r"module\s+([^\s\n]+)"
            module_match = re.search(module_pattern, content)

            # Extract Go version
            go_version_pattern = r"go\s+(\d+\.\d+(?:\.\d+)?)"
            go_version_match = re.search(go_version_pattern, content)

            # Extract dependencies
            require_pattern = r"require\s*\(\s*([^)]+)\s*\)"
            require_match = re.search(require_pattern, content, re.DOTALL)

            dependencies = []
            if require_match:
                require_block = require_match.group(1)
                dep_pattern = r"([^\s]+)\s+([^\s\n]+)"
                for dep_match in re.finditer(dep_pattern, require_block):
                    dependencies.append({"name": dep_match.group(1), "version": dep_match.group(2)})

            return {
                "module_name": module_match.group(1) if module_match else None,
                "go_version": go_version_match.group(1) if go_version_match else None,
                "dependencies": dependencies,
                "go_mod_path": str(go_mod_path),
            }

        except OSError as e:
            logger.warning("Failed to read go.mod file %s: %s", go_mod_path, e)
            return None

    def get_package_dependencies(self, content: str) -> list[str]:
        """Extract package dependencies from import statements.

        Args:
            content: Go source code content

        Returns:
            List of imported package paths
        """
        dependencies = []

        # Extract imports
        import_pattern = r'import\s+(?:\(([^)]+)\)|"([^"]+)")'
        for match in re.finditer(import_pattern, content):
            if match.group(1):  # Multiple imports
                imports_text = match.group(1)
                for imp_match in re.finditer(r'"([^"]+)"', imports_text):
                    dependencies.append(imp_match.group(1))
            else:  # Single import
                dependencies.append(match.group(2))

        return dependencies

    def classify_package_type(self, file_path: Path, content: str) -> str:
        """Classify the type of Go package.

        Args:
            file_path: Path to the Go source file
            content: Go source code content

        Returns:
            Package type classification
        """
        package_name = self.get_package_name_from_content(content)

        if package_name == "main":
            # Check if it has main function for executable
            if re.search(r"func\s+main\s*\(\s*\)", content):
                return "executable"
            else:
                return "main_package"
        elif self.is_test_file(file_path):
            return "test"
        elif package_name and package_name.endswith("_test"):
            return "external_test"
        else:
            return "library"

    def get_package_visibility_exports(self, content: str) -> list[str]:
        """Get list of exported (public) identifiers from package.

        Args:
            content: Go source code content

        Returns:
            List of exported function, type, variable, and constant names
        """
        exports = []

        # Exported functions (start with uppercase)
        func_pattern = r"func\s+(?:\([^)]+\)\s+)?([A-Z]\w*)\s*\("
        for match in re.finditer(func_pattern, content):
            exports.append(match.group(1))

        # Exported types
        type_pattern = r"type\s+([A-Z]\w*)\s+"
        for match in re.finditer(type_pattern, content):
            exports.append(match.group(1))

        # Exported variables
        var_pattern = r"var\s+([A-Z]\w*)\s+"
        for match in re.finditer(var_pattern, content):
            exports.append(match.group(1))

        # Exported constants
        const_pattern = r"const\s+([A-Z]\w*)\s+"
        for match in re.finditer(const_pattern, content):
            exports.append(match.group(1))

        # Also check const blocks
        const_block_pattern = r"const\s*\(\s*([^)]+)\s*\)"
        for match in re.finditer(const_block_pattern, content):
            block = match.group(1)
            for const_match in re.finditer(r"([A-Z]\w*)\s*[=\s]", block):
                exports.append(const_match.group(1))

        return list(set(exports))  # Remove duplicates
