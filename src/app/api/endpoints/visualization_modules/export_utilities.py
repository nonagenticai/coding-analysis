"""
Export Utilities

Common utilities and helper functions for visualization export operations.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ExportUtilities:
    """Common utilities for visualization export."""

    @staticmethod
    def get_language_color(language: str, index: int = 0) -> str:
        """Get color for a programming language."""
        color_map = {
            "java": "#ff6b6b",
            "python": "#4ecdc4",
            "typescript": "#45b7d1",
            "javascript": "#f9ca24",
            "rust": "#fd79a8",
            "go": "#00b894",
            "csharp": "#6c5ce7",
            "cpp": "#fd79a8",
            "c": "#e17055",
        }
        return color_map.get(language.lower(), f"#{hash(language) % 0xFFFFFF:06x}")

    @staticmethod
    def get_cytoscape_style() -> list[dict[str, Any]]:
        """Get default Cytoscape.js styling."""
        return [
            {
                "selector": "node",
                "style": {
                    "content": "data(label)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "background-color": "#666",
                    "color": "white",
                    "font-size": "12px",
                    "width": "40px",
                    "height": "40px",
                },
            },
            {
                "selector": "edge",
                "style": {
                    "width": 2,
                    "line-color": "#ccc",
                    "target-arrow-color": "#ccc",
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                },
            },
            {"selector": ".node-java", "style": {"background-color": "#ff6b6b"}},
            {"selector": ".node-python", "style": {"background-color": "#4ecdc4"}},
            {"selector": ".node-typescript", "style": {"background-color": "#45b7d1"}},
        ]

    @staticmethod
    def get_node_attributes(node: Any, color: str) -> str:
        """Get Graphviz node attributes."""
        attrs = [f'fillcolor="{color}"']

        if hasattr(node, "complexity") and node.complexity > 10:
            attrs.append("penwidth=3")

        if hasattr(node, "shape"):
            attrs.append(f"shape={node.shape}")

        return f", {', '.join(attrs)}" if attrs else ""

    @staticmethod
    def get_edge_attributes(edge: Any) -> str:
        """Get Graphviz edge attributes."""
        attrs = []

        if hasattr(edge, "color") and edge.color:
            attrs.append(f'color="{edge.color}"')

        if hasattr(edge, "weight") and edge.weight > 1:
            attrs.append(f"penwidth={min(edge.weight, 5)}")

        if hasattr(edge, "style") and edge.style == "dashed":
            attrs.append("style=dashed")

        return f" [{', '.join(attrs)}]" if attrs else ""

    @staticmethod
    def sanitize_mermaid_id(node_id: str) -> str:
        """Sanitize node ID for Mermaid compatibility."""
        # Replace invalid characters for Mermaid
        return node_id.replace(".", "_").replace("/", "_").replace("-", "_")

    @staticmethod
    def get_mermaid_styling(nodes: list[Any]) -> list[str]:
        """Get Mermaid styling lines."""
        styling = []

        # Group nodes by language for styling
        languages = set(node.language for node in nodes)
        for i, language in enumerate(languages):
            color = ExportUtilities.get_language_color(language, i)
            class_def = f"  classDef {language}Class fill:{color},stroke:#333,stroke-width:2px"
            styling.append(class_def)

        # Apply classes to nodes
        for node in nodes:
            node_id = ExportUtilities.sanitize_mermaid_id(node.id)
            styling.append(f"  class {node_id} {node.language}Class")

        return styling

    @staticmethod
    def convert_nodes_to_dict(nodes: list[Any]) -> list[dict[str, Any]]:
        """Convert nodes to dictionary format."""
        json_nodes = []
        for node in nodes:
            node_dict = {
                "id": node.id,
                "label": node.label,
                "type": node.type,
                "language": node.language,
                "size": getattr(node, "size", 1),
                "complexity": getattr(node, "complexity", 0),
                "importance": getattr(node, "importance", 0),
                "file_path": getattr(node, "file_path", None),
                "group": getattr(node, "group", None),
                "color": getattr(node, "color", None),
                "shape": getattr(node, "shape", "box"),
                "properties": getattr(node, "properties", {}),
            }
            json_nodes.append(node_dict)
        return json_nodes

    @staticmethod
    def convert_edges_to_dict(edges: list[Any]) -> list[dict[str, Any]]:
        """Convert edges to dictionary format."""
        json_edges = []
        for edge in edges:
            edge_dict = {
                "id": edge.id,
                "source": edge.source,
                "target": edge.target,
                "type": edge.type,
                "label": getattr(edge, "label", None),
                "weight": getattr(edge, "weight", 1),
                "color": getattr(edge, "color", None),
                "style": getattr(edge, "style", "solid"),
                "properties": getattr(edge, "properties", {}),
            }
            json_edges.append(edge_dict)
        return json_edges

    def to_dot(self, nodes: list[Any], edges: list[Any], title: str = "Graph") -> str:
        """Export graph to Graphviz DOT format."""
        lines = [f'digraph "{title}" {{', "  rankdir=LR;", "  node [shape=box];"]
        for node in nodes:
            node_id = (
                getattr(node, "id", str(node)) if not isinstance(node, dict) else node.get("id", "")
            )
            label = (
                getattr(node, "label", node_id)
                if not isinstance(node, dict)
                else node.get("label", node_id)
            )
            lines.append(f'  "{node_id}" [label="{label}"];')
        for edge in edges:
            source = (
                getattr(edge, "source", "")
                if not isinstance(edge, dict)
                else edge.get("source", "")
            )
            target = (
                getattr(edge, "target", "")
                if not isinstance(edge, dict)
                else edge.get("target", "")
            )
            lines.append(f'  "{source}" -> "{target}";')
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self, nodes: list[Any], edges: list[Any]) -> str:
        """Export graph to Mermaid format."""
        lines = ["graph LR"]
        for node in nodes:
            node_id = (
                getattr(node, "id", str(node)) if not isinstance(node, dict) else node.get("id", "")
            )
            label = (
                getattr(node, "label", node_id)
                if not isinstance(node, dict)
                else node.get("label", node_id)
            )
            safe_id = self.sanitize_mermaid_id(str(node_id))
            lines.append(f"  {safe_id}[{label}]")
        for edge in edges:
            source = (
                getattr(edge, "source", "")
                if not isinstance(edge, dict)
                else edge.get("source", "")
            )
            target = (
                getattr(edge, "target", "")
                if not isinstance(edge, dict)
                else edge.get("target", "")
            )
            safe_source = self.sanitize_mermaid_id(str(source))
            safe_target = self.sanitize_mermaid_id(str(target))
            lines.append(f"  {safe_source} --> {safe_target}")
        return "\n".join(lines)

    def to_graphml(self, nodes: list[Any], edges: list[Any], title: str = "Graph") -> str:
        """Export graph to GraphML format."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphstudio.org/xmlns">',
            f'  <graph id="{title}" edgedefault="directed">',
        ]
        for node in nodes:
            node_id = (
                getattr(node, "id", str(node)) if not isinstance(node, dict) else node.get("id", "")
            )
            lines.append(f'    <node id="{node_id}"/>')
        for i, edge in enumerate(edges):
            source = (
                getattr(edge, "source", "")
                if not isinstance(edge, dict)
                else edge.get("source", "")
            )
            target = (
                getattr(edge, "target", "")
                if not isinstance(edge, dict)
                else edge.get("target", "")
            )
            lines.append(f'    <edge id="e{i}" source="{source}" target="{target}"/>')
        lines.extend(["  </graph>", "</graphml>"])
        return "\n".join(lines)


def export_to_format(
    visualization_data: dict[str, Any],
    format: str,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Export visualization to specified format.

    Args:
        visualization_data: Visualization data to export
        format: Target export format
        options: Optional export options

    Returns:
        Exported data in specified format
    """
    options = options or {}
    exporter = ExportUtilities()

    if format == "json":
        return visualization_data
    elif format == "dot":
        return {
            "format": "dot",
            "content": exporter.to_dot(
                visualization_data.get("nodes", []),
                visualization_data.get("edges", []),
                visualization_data.get("title", "Graph"),
            ),
        }
    elif format == "mermaid":
        return {
            "format": "mermaid",
            "content": exporter.to_mermaid(
                visualization_data.get("nodes", []), visualization_data.get("edges", [])
            ),
        }
    elif format == "graphml":
        return {
            "format": "graphml",
            "content": exporter.to_graphml(
                visualization_data.get("nodes", []),
                visualization_data.get("edges", []),
                visualization_data.get("title", "Graph"),
            ),
        }
    else:
        # For image formats (png, svg, pdf), return format info
        # Actual rendering would be done by frontend or separate service
        return {
            "format": format,
            "message": f"Export to {format} requires client-side rendering",
            "data": visualization_data,
        }


def get_supported_formats() -> list[str]:
    """
    Get list of supported export formats.

    Returns:
        List of supported format strings
    """
    return ["json", "dot", "mermaid", "graphml", "png", "svg", "pdf"]


def validate_export_format(format: str) -> bool:
    """
    Validate if export format is supported.

    Args:
        format: Format to validate

    Returns:
        True if format is supported, False otherwise
    """
    return format.lower() in get_supported_formats()
