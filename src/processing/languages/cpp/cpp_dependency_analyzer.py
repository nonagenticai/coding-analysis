"""
C++ dependency analysis module.

This module handles analysis of dependencies in C++ projects, including
include dependencies, library dependencies, and symbol resolution.
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
)


class CppDependencyAnalyzer:
    """Analyzer for C++ dependencies and relationships."""

    def __init__(self):
        """Initialize the dependency analyzer."""
        self.include_cache: dict[str, set[str]] = {}
        self.symbol_cache: dict[str, list[CoreCodeEntity]] = {}

    def analyze_include_dependencies(self, file_path: Path, content: str) -> list[str]:
        """Analyze #include dependencies in a C++ file."""
        includes = []

        # Match both angle bracket and quote includes
        include_patterns = [
            r"#include\s*<([^>]+)>",  # System includes
            r'#include\s*"([^"]+)"',  # Local includes
        ]

        for pattern in include_patterns:
            for match in re.finditer(pattern, content):
                include_path = match.group(1)
                includes.append(include_path)

        return includes

    def resolve_include_paths(
        self,
        file_path: Path,
        includes: list[str],
        include_directories: list[str] | None = None,
    ) -> dict[str, Path | None]:
        """Resolve include paths to actual file locations."""
        resolved = {}
        base_dir = file_path.parent

        include_dirs = include_directories or []
        # Add standard system directories
        standard_dirs = [
            "/usr/include",
            "/usr/local/include",
            "/opt/local/include",
        ]
        include_dirs.extend(standard_dirs)

        for include in includes:
            resolved_path = None

            # Try relative to current file
            candidate = base_dir / include
            if candidate.exists():
                resolved_path = candidate
            else:
                # Try include directories
                for inc_dir in include_dirs:
                    candidate = Path(inc_dir) / include
                    if candidate.exists():
                        resolved_path = candidate
                        break

            resolved[include] = resolved_path

        return resolved

    def analyze_symbol_dependencies(self, entities: list[CoreCodeEntity]) -> list[EntityRelation]:
        """Analyze symbol dependencies between entities."""
        relationships = []

        # Group entities by type for easier analysis
        namespaces = [e for e in entities if e.type == EntityType.NAMESPACE]
        classes = [e for e in entities if e.type == EntityType.CLASS]
        structs = [e for e in entities if e.type == EntityType.STRUCT]
        functions = [e for e in entities if e.type == EntityType.FUNCTION]
        variables = [e for e in entities if e.type == EntityType.VARIABLE]

        # Namespace contains other entities
        for entity in entities:
            if entity.type != EntityType.NAMESPACE and namespaces:
                # Simple heuristic: assign to first namespace found
                namespace = namespaces[0]
                relationships.append(
                    EntityRelation(
                        source_id=namespace.id,
                        target_id=entity.id,
                        type=RelationType.CONTAINS,
                        properties={"relationship_context": "namespace_member"},
                    )
                )

        # Class/Struct contains methods and member variables
        containers = classes + structs
        members = functions + variables

        for container in containers:
            for member in members:
                # Heuristic: if member is defined after container in same file
                if (
                    member.line_number
                    and container.line_number
                    and member.line_number > container.line_number
                    and member.file_path == container.file_path
                ):
                    relationships.append(
                        EntityRelation(
                            source_id=container.id,
                            target_id=member.id,
                            type=RelationType.DEFINES,
                            properties={"relationship_context": "member_definition"},
                        )
                    )

        # Inheritance relationships (basic detection)
        relationships.extend(self._analyze_inheritance_relationships(classes))

        # Function call relationships
        relationships.extend(self._analyze_function_calls(entities))

        return relationships

    def _analyze_inheritance_relationships(
        self, classes: list[CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Analyze inheritance relationships between classes."""
        relationships = []

        for class_entity in classes:
            inheritance_info = class_entity.properties.get("inheritance", [])
            for parent_spec in inheritance_info:
                # Parse inheritance specification (e.g., "public BaseClass")
                parts = parent_spec.strip().split()
                if len(parts) >= 2:
                    access_level = parts[0]  # public, private, protected
                    parent_name = parts[1]

                    # Find parent class entity
                    parent_entity = None
                    for candidate in classes:
                        if candidate.name == parent_name:
                            parent_entity = candidate
                            break

                    if parent_entity:
                        relationships.append(
                            EntityRelation(
                                source_id=class_entity.id,
                                target_id=parent_entity.id,
                                type=RelationType.INHERITS,
                                properties={
                                    "access_level": access_level,
                                    "relationship_context": "inheritance",
                                },
                            )
                        )

        return relationships

    def _analyze_function_calls(self, entities: list[CoreCodeEntity]) -> list[EntityRelation]:
        """Analyze function call relationships."""
        relationships = []
        functions = [e for e in entities if e.type == EntityType.FUNCTION]

        # This is a simplified analysis - in practice, you'd need to parse
        # function bodies to detect calls
        for func in functions:
            signature = func.properties.get("signature", "")
            # Look for function calls in the signature (very basic)
            for other_func in functions:
                if other_func != func and other_func.name in signature:
                    relationships.append(
                        EntityRelation(
                            source_id=func.id,
                            target_id=other_func.id,
                            type=RelationType.CALLS,
                            properties={"relationship_context": "function_call"},
                        )
                    )

        return relationships

    def analyze_library_dependencies(self, project_root: Path) -> dict[str, Any]:
        """Analyze library dependencies in the project."""
        dependencies: dict[str, Any] = {
            "system_libraries": set(),
            "third_party_libraries": set(),
            "local_libraries": set(),
            "package_managers": [],
        }

        # Look for common package manager files
        package_files = {
            "conanfile.txt": "conan",
            "conanfile.py": "conan",
            "vcpkg.json": "vcpkg",
            "Pipfile": "pip",  # If using Python tools
        }

        for file_name, manager in package_files.items():
            if (project_root / file_name).exists():
                dependencies["package_managers"].append(manager)

        # Analyze CMake dependencies if CMake files exist
        cmake_deps = self._analyze_cmake_dependencies(project_root)
        dependencies.update(cmake_deps)

        # Analyze pkg-config dependencies
        pkg_config_deps = self._analyze_pkg_config_dependencies(project_root)
        dependencies["system_libraries"].update(pkg_config_deps)

        # Convert sets to lists for JSON serialization
        dependencies["system_libraries"] = list(dependencies["system_libraries"])
        dependencies["third_party_libraries"] = list(dependencies["third_party_libraries"])
        dependencies["local_libraries"] = list(dependencies["local_libraries"])

        return dependencies

    def _analyze_cmake_dependencies(self, project_root: Path) -> dict[str, set[str]]:
        """Analyze dependencies from CMake files."""
        dependencies: dict[str, set[str]] = {
            "system_libraries": set(),
            "third_party_libraries": set(),
            "local_libraries": set(),
        }

        # Find CMake files
        cmake_files = list(project_root.rglob("CMakeLists.txt"))
        cmake_files.extend(list(project_root.rglob("*.cmake")))

        for cmake_file in cmake_files:
            try:
                content = cmake_file.read_text(encoding="utf-8", errors="ignore")

                # Extract find_package dependencies
                find_package_pattern = r"find_package\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)"
                for match in re.finditer(find_package_pattern, content, re.IGNORECASE):
                    package_name = match.group(1)
                    dependencies["third_party_libraries"].add(package_name)

                # Extract target_link_libraries
                link_pattern = r"target_link_libraries\s*\([^)]+\s+([^)]+)\)"
                for match in re.finditer(link_pattern, content, re.IGNORECASE | re.MULTILINE):
                    libraries = match.group(1).split()
                    for lib in libraries:
                        lib = lib.strip()
                        if lib.startswith("-l"):
                            # System library (e.g., -lpthread)
                            dependencies["system_libraries"].add(lib[2:])
                        elif "::" in lib:
                            # Modern CMake target (e.g., Boost::filesystem)
                            dependencies["third_party_libraries"].add(lib)
                        elif lib and not lib.startswith("$"):
                            # Local or system library
                            dependencies["local_libraries"].add(lib)

            except Exception:
                continue

        return dependencies

    def _analyze_pkg_config_dependencies(self, project_root: Path) -> set[str]:
        """Analyze pkg-config dependencies."""
        dependencies: set[str] = set()

        # Look for .pc files or references to pkg-config
        pc_files = list(project_root.rglob("*.pc"))
        pc_in_files = list(project_root.rglob("*.pc.in"))

        for pc_file in pc_files + pc_in_files:
            try:
                content = pc_file.read_text(encoding="utf-8", errors="ignore")

                # Extract Requires field
                requires_pattern = r"Requires:\s*([^\n]+)"
                for match in re.finditer(requires_pattern, content):
                    deps = match.group(1).split()
                    dependencies.update(deps)

            except Exception:
                continue

        return dependencies

    def analyze_header_dependencies(self, file_path: Path, content: str) -> dict[str, Any]:
        """Analyze header file dependencies and structure."""
        analysis: dict[str, Any] = {
            "includes": [],
            "forward_declarations": [],
            "header_guards": None,
            "pragma_once": False,
            "conditional_compilation": [],
        }

        # Extract includes
        analysis["includes"] = self.analyze_include_dependencies(file_path, content)

        # Check for header guards
        guard_pattern = r"#ifndef\s+([A-Za-z_][A-Za-z0-9_]*)"
        guard_match = re.search(guard_pattern, content)
        if guard_match:
            analysis["header_guards"] = guard_match.group(1)

        # Check for pragma once
        if "#pragma once" in content:
            analysis["pragma_once"] = True

        # Extract forward declarations
        forward_decl_patterns = [
            r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*;",
            r"struct\s+([A-Za-z_][A-Za-z0-9_]*)\s*;",
            r"enum\s+(?:class\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*;",
        ]

        for pattern in forward_decl_patterns:
            for match in re.finditer(pattern, content):
                analysis["forward_declarations"].append(match.group(1))

        # Extract conditional compilation blocks
        ifdef_pattern = r"#ifdef\s+([A-Za-z_][A-Za-z0-9_]*)"
        ifndef_pattern = r"#ifndef\s+([A-Za-z_][A-Za-z0-9_]*)"

        for match in re.finditer(ifdef_pattern, content):
            analysis["conditional_compilation"].append(
                {
                    "type": "ifdef",
                    "condition": match.group(1),
                }
            )

        for match in re.finditer(ifndef_pattern, content):
            analysis["conditional_compilation"].append(
                {
                    "type": "ifndef",
                    "condition": match.group(1),
                }
            )

        return analysis

    def create_dependency_graph(
        self, entities: list[CoreCodeEntity], relationships: list[EntityRelation]
    ) -> dict[str, Any]:
        """Create a dependency graph representation."""
        graph: dict[str, Any] = {
            "nodes": [],
            "edges": [],
            "statistics": {
                "total_nodes": len(entities),
                "total_edges": len(relationships),
                "node_types": {},
                "relationship_types": {},
            },
        }

        # Add nodes
        for entity in entities:
            graph["nodes"].append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type.value,
                    "file_path": entity.file_path,
                    "line_number": entity.line_number,
                    "properties": entity.properties,
                }
            )

            # Update statistics
            node_type = entity.type.value
            graph["statistics"]["node_types"][node_type] = (
                graph["statistics"]["node_types"].get(node_type, 0) + 1
            )

        # Add edges
        for relationship in relationships:
            graph["edges"].append(
                {
                    "source": relationship.source_id,
                    "target": relationship.target_id,
                    "type": relationship.type.value,
                    "properties": relationship.properties,
                }
            )

            # Update statistics
            rel_type = relationship.type.value
            graph["statistics"]["relationship_types"][rel_type] = (
                graph["statistics"]["relationship_types"].get(rel_type, 0) + 1
            )

        return graph

    def find_circular_dependencies(self, relationships: list[EntityRelation]) -> list[list[str]]:
        """Find circular dependencies in the relationship graph."""
        # Build adjacency list
        graph: dict[str, list[str]] = {}
        for rel in relationships:
            if rel.source_id not in graph:
                graph[rel.source_id] = []
            graph[rel.source_id].append(rel.target_id)

        # Find cycles using DFS
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node, path):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                dfs(neighbor, path[:])

            rec_stack.remove(node)
            path.pop()

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def analyze_coupling_metrics(
        self, entities: list[CoreCodeEntity], relationships: list[EntityRelation]
    ) -> dict[str, Any]:
        """Analyze coupling metrics for the codebase."""
        metrics: dict[str, Any] = {
            "afferent_coupling": {},  # Incoming dependencies
            "efferent_coupling": {},  # Outgoing dependencies
            "instability": {},  # Efferent / (Afferent + Efferent)
            "total_coupling": 0,
        }

        # Initialize metrics for all entities
        for entity in entities:
            metrics["afferent_coupling"][entity.id] = 0
            metrics["efferent_coupling"][entity.id] = 0

        # Count relationships
        for rel in relationships:
            if rel.target_id in metrics["afferent_coupling"]:
                metrics["afferent_coupling"][rel.target_id] += 1
            if rel.source_id in metrics["efferent_coupling"]:
                metrics["efferent_coupling"][rel.source_id] += 1

        # Calculate instability
        for entity_id in metrics["afferent_coupling"]:
            afferent = metrics["afferent_coupling"][entity_id]
            efferent = metrics["efferent_coupling"][entity_id]
            total = afferent + efferent

            if total > 0:
                metrics["instability"][entity_id] = efferent / total
            else:
                metrics["instability"][entity_id] = 0

        metrics["total_coupling"] = len(relationships)

        return metrics
