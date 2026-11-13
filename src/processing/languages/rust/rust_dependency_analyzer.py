"""Rust dependency analysis module.

This module handles analysis of Rust dependencies, use statements,
and import relationships within Rust projects.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypedDict


class UseInfo(TypedDict):
    """Type definition for parsed use statement information."""

    raw: str
    is_external: bool
    is_std: bool
    is_relative: bool
    is_glob: bool
    is_aliased: bool
    base_path: str
    imported_items: list[str]
    alias: str | None


class DependencyAnalysis(TypedDict):
    """Type definition for dependency analysis results."""

    total_imports: int
    external_crates: list[UseInfo]
    std_imports: list[UseInfo]
    relative_imports: list[UseInfo]
    glob_imports: list[UseInfo]
    aliased_imports: list[UseInfo]


from models.core_models import (
    CoreCodeEntity,
    EntityRelation,
    EntityType,
    RelationType,
    SourceLocation,
)

# Import Tree-sitter components if available
try:
    from tree_sitter import Node, Tree  # noqa: F401

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    # Tree/Node types not available - use Any in method signatures


class RustDependencyAnalyzer:
    """Handles Rust dependency and import analysis."""

    # Use statement node types
    USE_NODE_TYPES = {"use_declaration"}

    def __init__(self, tree_sitter_processor=None):
        """Initialize the dependency analyzer.

        Args:
            tree_sitter_processor: Optional tree-sitter processor instance
        """
        self.tree_sitter_processor = tree_sitter_processor

    def extract_use_statements_regex(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
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

    def parse_use_statement(self, use_statement: str) -> UseInfo:
        """Parse a use statement into its components.

        Args:
            use_statement: The use statement string

        Returns:
            UseInfo with parsed components
        """
        use_info: UseInfo = {
            "raw": use_statement,
            "is_external": False,
            "is_std": False,
            "is_relative": False,
            "is_glob": False,
            "is_aliased": False,
            "base_path": "",
            "imported_items": [],
            "alias": None,
        }

        # Remove 'use' keyword and semicolon
        path = use_statement.replace("use", "").replace(";", "").strip()

        # Check for glob imports
        if path.endswith("*"):
            use_info["is_glob"] = True
            path = path.rstrip("*").rstrip("::").strip()

        # Check for aliases
        if " as " in path:
            use_info["is_aliased"] = True
            parts = path.split(" as ")
            path = parts[0].strip()
            use_info["alias"] = parts[1].strip()

        # Check for group imports: use std::{io, fs};
        if "{" in path and "}" in path:
            base_path = path.split("{")[0].rstrip("::")
            items_str = path.split("{")[1].split("}")[0]
            items = [item.strip() for item in items_str.split(",")]
            use_info["base_path"] = base_path
            use_info["imported_items"] = items
        else:
            use_info["base_path"] = path
            use_info["imported_items"] = [path.split("::")[-1]] if "::" in path else [path]

        # Determine import type
        if use_info["base_path"].startswith("std::"):
            use_info["is_std"] = True
        elif use_info["base_path"].startswith("crate::"):
            use_info["is_relative"] = True
        elif use_info["base_path"].startswith("super::") or use_info["base_path"].startswith(
            "self::"
        ):
            use_info["is_relative"] = True
        elif "::" in use_info["base_path"] and not use_info["base_path"].startswith("crate::"):
            use_info["is_external"] = True

        return use_info

    def extract_external_dependencies(self, entities: list[CoreCodeEntity]) -> list[str]:
        """Extract external dependency names from use statements.

        Args:
            entities: List of entities containing use statements

        Returns:
            List of external dependency names
        """
        external_deps = set()

        for entity in entities:
            if entity.type == EntityType.IMPORT and "use_path" in entity.properties:
                use_path = entity.properties["use_path"]
                use_info = self.parse_use_statement(f"use {use_path};")

                if use_info["is_external"]:
                    # Extract the crate name (first component)
                    crate_name = use_info["base_path"].split("::")[0]
                    external_deps.add(crate_name)

        return list(external_deps)

    def extract_internal_dependencies(self, entities: list[CoreCodeEntity]) -> list[dict]:
        """Extract internal module dependencies from use statements.

        Args:
            entities: List of entities containing use statements

        Returns:
            List of internal dependency information
        """
        internal_deps = []

        for entity in entities:
            if entity.type == EntityType.IMPORT and "use_path" in entity.properties:
                use_path = entity.properties["use_path"]
                use_info = self.parse_use_statement(f"use {use_path};")

                if use_info["is_relative"] or (
                    not use_info["is_external"] and not use_info["is_std"]
                ):
                    internal_deps.append(
                        {
                            "entity_id": entity.id,
                            "path": use_info["base_path"],
                            "items": use_info["imported_items"],
                            "is_glob": use_info["is_glob"],
                            "alias": use_info["alias"],
                            "line_number": entity.line_number,
                        }
                    )

        return internal_deps

    # Tree-sitter specific methods
    if TREE_SITTER_AVAILABLE:

        def extract_use_declarations_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Extract use declarations (imports) using Tree-sitter."""
            entities: list[CoreCodeEntity] = []
            if not self.tree_sitter_processor:
                return entities

            use_nodes = self.tree_sitter_processor._find_nodes_by_type(tree, self.USE_NODE_TYPES)

            for use_node in use_nodes:
                # Get the use path
                use_path = (
                    self.tree_sitter_processor._get_node_text(use_node, content)
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

        def extract_use_relationships_tree_sitter(
            self, tree: Any, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Extract use/import relationships using Tree-sitter."""
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

        def extract_function_calls_tree_sitter(
            self, tree: Any, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Extract function call relationships using Tree-sitter."""
            relationships: list[EntityRelation] = []
            if not self.tree_sitter_processor:
                return relationships

            call_nodes = self.tree_sitter_processor._find_nodes_by_type(tree, {"call_expression"})

            for call_node in call_nodes:
                # Find the containing function
                containing_function = self._find_containing_function(call_node, entities)
                if not containing_function:
                    continue

                # Extract function name
                function_node = self.tree_sitter_processor._get_child_by_type(
                    call_node, "identifier"
                )
                if not function_node:
                    function_node = self.tree_sitter_processor._get_child_by_type(
                        call_node, "field_expression"
                    )

                if function_node:
                    function_name = self.tree_sitter_processor._get_node_text(
                        function_node, tree.text
                    )

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

        def _find_containing_function(
            self, node: Any, entities: list[CoreCodeEntity]
        ) -> CoreCodeEntity | None:
            """Find the function that contains a given node."""
            if not self.tree_sitter_processor:
                return None

            current = node.parent
            while current:
                if current.type in {"function_item", "impl_item"}:
                    # Find entity that matches this node
                    for entity in entities:
                        if (
                            entity.type == EntityType.FUNCTION
                            and entity.line_number == current.start_point[0] + 1
                        ):
                            return entity
                current = current.parent
            return None

        def _node_to_location(self, node: Any, file_path: Path) -> SourceLocation:
            """Convert a Tree-sitter node to a SourceLocation."""
            return SourceLocation(
                file_path=str(file_path),
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                column_start=node.start_point[1],
                column_end=node.end_point[1],
            )

    else:
        # Fallback methods when Tree-sitter is not available
        def extract_use_declarations_tree_sitter(
            self, tree: Any, file_path: Path, content: bytes
        ) -> list[CoreCodeEntity]:
            """Fallback use declarations extraction."""
            return []

        def extract_use_relationships_tree_sitter(
            self, tree: Any, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Fallback use relationships extraction."""
            return []

        def extract_function_calls_tree_sitter(
            self, tree: Any, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Fallback function calls extraction."""
            return []

        def _find_containing_function(
            self, node: Any, entities: list[CoreCodeEntity]
        ) -> CoreCodeEntity | None:
            """Fallback containing function finder."""
            return None

        def _node_to_location(self, node: Any, file_path: Path) -> SourceLocation:
            """Fallback node to location converter."""
            return SourceLocation(
                file_path=str(file_path),
                line_start=1,
                line_end=1,
                column_start=0,
                column_end=0,
            )

    def analyze_dependency_graph(self, entities: list[CoreCodeEntity]) -> DependencyAnalysis:
        """Analyze the dependency graph for the current file.

        Args:
            entities: List of entities from the file

        Returns:
            DependencyAnalysis containing dependency graph analysis
        """
        use_entities = [e for e in entities if e.type == EntityType.IMPORT]

        analysis: DependencyAnalysis = {
            "total_imports": len(use_entities),
            "external_crates": [],
            "std_imports": [],
            "relative_imports": [],
            "glob_imports": [],
            "aliased_imports": [],
        }

        for entity in use_entities:
            if "use_path" in entity.properties:
                use_path = entity.properties["use_path"]
                use_info = self.parse_use_statement(f"use {use_path};")

                if use_info["is_std"]:
                    analysis["std_imports"].append(use_info)
                elif use_info["is_external"]:
                    analysis["external_crates"].append(use_info)
                elif use_info["is_relative"]:
                    analysis["relative_imports"].append(use_info)

                if use_info["is_glob"]:
                    analysis["glob_imports"].append(use_info)

                if use_info["is_aliased"]:
                    analysis["aliased_imports"].append(use_info)

        return analysis

    def get_dependency_complexity_score(self, entities: list[CoreCodeEntity]) -> float:
        """Calculate a complexity score based on dependencies.

        Args:
            entities: List of entities from the file

        Returns:
            Complexity score (higher = more complex)
        """
        analysis = self.analyze_dependency_graph(entities)

        # Base score
        score = 0.0

        # Add points for different types of imports
        score += len(analysis["external_crates"]) * 0.3  # External deps add complexity
        score += len(analysis["std_imports"]) * 0.1  # Std imports are simpler
        score += len(analysis["relative_imports"]) * 0.2  # Module deps are moderate
        score += len(analysis["glob_imports"]) * 0.5  # Glob imports add significant complexity
        score += len(analysis["aliased_imports"]) * 0.3  # Aliases add some complexity

        return score

    def suggest_import_optimizations(self, entities: list[CoreCodeEntity]) -> list[dict]:
        """Suggest optimizations for import statements.

        Args:
            entities: List of entities from the file

        Returns:
            List of optimization suggestions
        """
        suggestions = []
        analysis = self.analyze_dependency_graph(entities)

        # Check for too many glob imports
        if len(analysis["glob_imports"]) > 3:
            suggestions.append(
                {
                    "type": "reduce_glob_imports",
                    "message": f"Consider reducing glob imports ({len(analysis['glob_imports'])} found). Import specific items instead.",
                    "severity": "warning",
                }
            )

        # Check for unused aliases (would need more sophisticated analysis)
        if len(analysis["aliased_imports"]) > 5:
            suggestions.append(
                {
                    "type": "review_aliases",
                    "message": f"Many aliased imports found ({len(analysis['aliased_imports'])}). Ensure all aliases are necessary.",
                    "severity": "info",
                }
            )

        # Check for too many external dependencies
        if len(analysis["external_crates"]) > 10:
            suggestions.append(
                {
                    "type": "reduce_external_deps",
                    "message": f"High number of external crates imported ({len(analysis['external_crates'])}). Consider reducing dependencies.",
                    "severity": "warning",
                }
            )

        return suggestions
