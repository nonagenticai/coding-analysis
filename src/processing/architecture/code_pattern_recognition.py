"""Code pattern recognition utilities for pattern detection.

This module provides the CodePatternRecognizer class which handles common pattern
recognition tasks including regex compilation, node matching, and code extraction
from graph nodes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from re import Pattern

import networkx as nx

from models.architecture_models import DetectedPattern


class CodePatternRecognizer:
    """Utility class for recognizing patterns in code graphs."""

    def __init__(self) -> None:
        """Initialize the pattern recognizer."""
        self._compiled_patterns: dict[str, Pattern[str]] = {}

    def get_compiled_pattern(self, pattern_str: str) -> Pattern[str]:
        """Get a compiled regex pattern, caching for performance.

        Args:
            pattern_str: The regex pattern to compile

        Returns:
            Compiled regex pattern
        """
        if pattern_str not in self._compiled_patterns:
            self._compiled_patterns[pattern_str] = re.compile(pattern_str, re.IGNORECASE)
        return self._compiled_patterns[pattern_str]

    def node_matches_pattern(self, node_data: dict, pattern: Pattern[str]) -> bool:
        """Check if a node matches a pattern.

        Args:
            node_data: Node data to check
            pattern: Compiled regex pattern

        Returns:
            True if node matches pattern, False otherwise
        """
        code = self.get_node_code(node_data)
        return bool(pattern.search(code))

    def get_node_code(self, node_data: dict) -> str:
        """Extract code content from node data.

        Args:
            node_data: Node data to extract code from

        Returns:
            Code content as a string
        """
        # Try different fields where code might be stored
        code_fields = ["code", "docstring", "properties"]

        for field in code_fields:
            if field in node_data:
                value = node_data[field]
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    # Look for code in nested properties
                    for key, val in value.items():
                        if "code" in key.lower() and isinstance(val, str):
                            return val

        # Fallback to node ID if no code found
        return str(node_data.get("id", ""))

    def detect_custom_patterns(
        self, graph: nx.DiGraph, custom_patterns: Sequence[dict]
    ) -> list[DetectedPattern]:
        """Detect patterns defined in configuration.

        Args:
            graph: The code graph to analyze for patterns
            custom_patterns: List of custom pattern configurations

        Returns:
            List of detected custom patterns
        """
        detected = []

        for pattern_config in custom_patterns:
            pattern_name = pattern_config.get("name", "Unknown")
            pattern_match = pattern_config.get("match", "")

            if not pattern_match:
                continue

            regex = self.get_compiled_pattern(pattern_match)

            # Search through all nodes
            for node_id, node_data in graph.nodes(data=True):
                if self.node_matches_pattern(node_data, regex):
                    detected.append(
                        DetectedPattern(
                            pattern_type=pattern_name,
                            confidence=0.9,
                            components=[node_id],
                            description=f"Custom pattern '{pattern_name}' detected",
                            name=pattern_name,
                            location=node_id,
                        )
                    )

        return detected

    def find_nodes_by_pattern(self, graph: nx.DiGraph, pattern_str: str) -> list[tuple[str, dict]]:
        """Find all nodes that match a given pattern.

        Args:
            graph: The code graph to search
            pattern_str: The regex pattern to match against

        Returns:
            List of (node_id, node_data) tuples that match the pattern
        """
        pattern = self.get_compiled_pattern(pattern_str)
        matching_nodes = []

        for node_id, node_data in graph.nodes(data=True):
            if self.node_matches_pattern(node_data, pattern):
                matching_nodes.append((node_id, node_data))

        return matching_nodes

    def get_node_type(self, node_data: dict) -> str:
        """Get the type of a node from its data.

        Args:
            node_data: Node data to extract type from

        Returns:
            Node type as a string, or 'unknown' if not found
        """
        return node_data.get("type", "unknown").lower()

    def has_edge_type(self, graph: nx.DiGraph, node_id: str, edge_type: str) -> bool:
        """Check if a node has any edges of a specific type.

        Args:
            graph: The code graph to check
            node_id: The node to check edges for
            edge_type: The type of edge to look for

        Returns:
            True if node has edges of the specified type, False otherwise
        """
        # Check outgoing edges
        for _, _, edge_data in graph.out_edges(node_id, data=True):
            if edge_data.get("type") == edge_type:
                return True

        # Check incoming edges
        for _, _, edge_data in graph.in_edges(node_id, data=True):
            if edge_data.get("type") == edge_type:
                return True

        return False

    def get_connected_nodes_by_edge_type(
        self, graph: nx.DiGraph, node_id: str, edge_type: str
    ) -> list[str]:
        """Get all nodes connected to a given node by a specific edge type.

        Args:
            graph: The code graph to search
            node_id: The starting node
            edge_type: The type of edge to follow

        Returns:
            List of connected node IDs
        """
        connected = []

        # Check outgoing edges
        for _, target_id, edge_data in graph.out_edges(node_id, data=True):
            if edge_data.get("type") == edge_type:
                connected.append(target_id)

        # Check incoming edges
        for source_id, _, edge_data in graph.in_edges(node_id, data=True):
            if edge_data.get("type") == edge_type:
                connected.append(source_id)

        return connected

    def extract_class_info(self, node_data: dict) -> dict:
        """Extract class-specific information from node data.

        Args:
            node_data: Node data to extract class info from

        Returns:
            Dictionary with class information
        """
        code = self.get_node_code(node_data)
        info = {
            "is_class": False,
            "class_name": "",
            "methods": [],
            "inherits_from": [],
            "implements": [],
        }

        # Check if this is a class
        class_pattern = self.get_compiled_pattern(r"class\s+(\w+)")
        class_match = class_pattern.search(code)
        if class_match:
            info["is_class"] = True
            info["class_name"] = class_match.group(1)

        # Extract method names
        method_pattern = self.get_compiled_pattern(r"def\s+(\w+)\s*\(")
        method_matches = method_pattern.findall(code)
        info["methods"] = method_matches

        # Extract inheritance info
        inheritance_pattern = self.get_compiled_pattern(r"class\s+\w+\s*\(\s*([^)]+)\s*\)")
        inheritance_match = inheritance_pattern.search(code)
        if inheritance_match:
            inherited_classes = [cls.strip() for cls in inheritance_match.group(1).split(",")]
            info["inherits_from"] = inherited_classes

        return info

    def calculate_code_complexity(self, node_data: dict) -> int:
        """Calculate a simple complexity metric for node code.

        Args:
            node_data: Node data to analyze

        Returns:
            Complexity score (higher = more complex)
        """
        code = self.get_node_code(node_data)

        # Simple complexity indicators
        complexity = 0

        # Count control structures
        control_patterns = [
            r"\bif\b",
            r"\belse\b",
            r"\belif\b",
            r"\bfor\b",
            r"\bwhile\b",
            r"\btry\b",
            r"\bexcept\b",
            r"\bfinally\b",
            r"\bswitch\b",
            r"\bcase\b",
        ]

        for pattern_str in control_patterns:
            pattern = self.get_compiled_pattern(pattern_str)
            complexity += len(pattern.findall(code))

        # Count function/method definitions
        func_pattern = self.get_compiled_pattern(r"\b(def|function)\s+\w+")
        complexity += len(func_pattern.findall(code))

        # Count class definitions
        class_pattern = self.get_compiled_pattern(r"\bclass\s+\w+")
        complexity += len(class_pattern.findall(code)) * 2  # Classes are more complex

        return complexity

    def is_test_code(self, node_data: dict) -> bool:
        """Determine if node contains test code.

        Args:
            node_data: Node data to check

        Returns:
            True if this appears to be test code, False otherwise
        """
        code = self.get_node_code(node_data).lower()
        node_id = str(node_data.get("id", "")).lower()

        # Test indicators
        test_indicators = [
            "test_",
            "_test",
            "test.py",
            "spec.js",
            ".test.",
            ".spec.",
            "unittest",
            "pytest",
            "jest",
            "mocha",
            "describe(",
            "it(",
            "test(",
            "assert",
            "expect(",
            "should",
            "mock",
        ]

        return any(indicator in code or indicator in node_id for indicator in test_indicators)

    def is_configuration_code(self, node_data: dict) -> bool:
        """Determine if node contains configuration code.

        Args:
            node_data: Node data to check

        Returns:
            True if this appears to be configuration code, False otherwise
        """
        code = self.get_node_code(node_data).lower()
        node_id = str(node_data.get("id", "")).lower()

        # Configuration indicators
        config_indicators = [
            "config",
            "settings",
            ".env",
            ".yml",
            ".yaml",
            ".json",
            ".ini",
            "docker",
            "compose",
            "makefile",
            "requirements.txt",
            "package.json",
            "pom.xml",
            "build.gradle",
            ".toml",
        ]

        return any(indicator in code or indicator in node_id for indicator in config_indicators)
