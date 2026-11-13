"""Helper functions for component extraction.

This module contains utility functions and helper methods for component extraction
to reduce the complexity of the main component extractor class.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx: Any = None

from models.architecture_models import Component


class ComponentExtractionHelpers:
    """Helper class for component extraction utility functions."""

    @staticmethod
    def get_annotation_patterns() -> list[tuple[str, str]]:
        """Get the list of annotation patterns for component detection."""
        return [
            # Python decorators
            (r'@component\s*\(\s*["\'](\w+)["\']\s*\)', "python"),
            (r'@module\s*\(\s*["\'](\w+)["\']\s*\)', "python"),
            (r'@service\s*\(\s*["\'](\w+)["\']\s*\)', "python"),
            # JavaScript/TypeScript decorators
            (
                r'@Component\s*\(\s*\{[^}]*name\s*:\s*["\'](\w+)["\']',
                "javascript",
            ),
            (
                r'@Module\s*\(\s*\{[^}]*name\s*:\s*["\'](\w+)["\']',
                "javascript",
            ),
            (
                r'@Injectable\s*\(\s*\{[^}]*providedIn\s*:\s*["\'](\w+)["\']',
                "javascript",
            ),
            # Comment-based annotations
            (r"#\s*component:\s*(\w+)", "any"),
            (r"//\s*component:\s*(\w+)", "any"),
            (r"/\*\s*component:\s*(\w+)\s*\*/", "any"),
            # Java annotations
            (r'@Component\s*\(\s*"(\w+)"\s*\)', "java"),
            (r'@Service\s*\(\s*"(\w+)"\s*\)', "java"),
            (r'@Repository\s*\(\s*"(\w+)"\s*\)', "java"),
        ]

    @staticmethod
    def get_node_code(node_data: dict[str, Any]) -> str:
        """Extract code content from node data."""
        # Try different possible keys for code content
        for key in ["code", "content", "source", "text", "body"]:
            if key in node_data and isinstance(node_data[key], str):
                return str(node_data[key])

        # Check nested properties
        if "properties" in node_data and isinstance(node_data["properties"], dict):
            for key in ["code", "content", "source"]:
                if key in node_data["properties"]:
                    return str(node_data["properties"][key])

        return ""

    @staticmethod
    def matches_language(file_path: str, language: str) -> bool:
        """Check if file path matches the specified language."""
        ext = Path(file_path).suffix.lower()
        language_extensions = {
            "python": {".py"},
            "javascript": {".js", ".jsx", ".ts", ".tsx"},
            "java": {".java"},
            "csharp": {".cs"},
            "go": {".go"},
            "rust": {".rs"},
        }

        return ext in language_extensions.get(language, set())

    @staticmethod
    def extract_component_annotation(code_text: str) -> str | None:
        """Extract component name from code annotations or decorators.

        Args:
            code_text: Source code to analyze for annotations

        Returns:
            Component name if found, None otherwise
        """
        if not code_text:
            return None

        # Pattern matching for common component annotations across languages
        patterns = [
            # Python style
            r'@component\s*[\(\[\{]?[\'"]([\w-]+)[\'"]',  # @component('name')
            # @Component(name='name')
            r'@Component\s*\([^)]*name\s*=\s*[\'"]([\w-]+)[\'"]',
            r"#\s*Component:\s*([\w-]+)",  # # Component: name
            # JavaScript/TypeScript style
            # @Component({selector: 'name'})
            r'@Component\s*\([^)]*selector\s*:\s*[\'"]([\w-]+)[\'"]',
            # Java/Kotlin style
            # @Component(name = "name")
            r'@Component\s*\([^)]*name\s*=\s*[\'"]([\w-]+)[\'"]',
            # Generic doc comment style
            r"[/\*]\*\s*@component\s+([\w-]+)",  # /** @component name */
        ]

        # Check each pattern
        for pattern in patterns:
            try:
                matches = re.findall(pattern, code_text, re.IGNORECASE)
                if matches:
                    return str(matches[0])
            except (re.error, IndexError):
                continue

        return None

    @staticmethod
    def scan_file_annotations(
        graph: Any, code_extractor: Callable[[dict[str, Any]], str]
    ) -> dict[str, set[str]]:
        """Scan for component annotations in file comments."""
        component_files: dict[str, set[str]] = {}

        # Group nodes by file
        nodes_by_file: dict[str, list[dict[str, Any]]] = {}
        for _, node_data in graph.nodes(data=True):
            file_path = node_data.get("file_path", "")
            if file_path:
                if file_path not in nodes_by_file:
                    nodes_by_file[file_path] = []
                nodes_by_file[file_path].append(node_data)

        # Process each file
        for file_path, nodes in nodes_by_file.items():
            # For each file, check if any node has a component annotation
            for node_data in nodes:
                component_name = ComponentExtractionHelpers.extract_component_annotation(
                    code_extractor(node_data)
                )
                if component_name:
                    if component_name not in component_files:
                        component_files[component_name] = set()
                    component_files[component_name].add(file_path)
                    break  # One component per file is enough

        return component_files

    @staticmethod
    def create_annotation_components(
        component_files: dict[str, set[str]],
    ) -> list[Component]:
        """Create components from annotation file mappings."""
        components = []

        for component_name, files in component_files.items():
            if files:  # Only create component if it has files
                component = Component(
                    id=f"annotation-{component_name}",
                    name=component_name,
                    files=sorted(files),
                    dependencies=[],
                    layer=None,
                )
                components.append(component)

        return components

    @staticmethod
    def create_build_components(
        dir_to_files: dict[str, set[str]], name_generator: Callable[[str], str]
    ) -> list[Component]:
        """Create components from build directory file mappings."""
        components = []
        for build_dir, files in dir_to_files.items():
            if files:
                component_id = f"component_{Path(build_dir).name}"
                component = Component(
                    id=component_id,
                    name=name_generator(component_id),
                    files=sorted(files),
                    dependencies=[],  # Will be resolved later
                    layer=None,
                )
                components.append(component)
        return components
