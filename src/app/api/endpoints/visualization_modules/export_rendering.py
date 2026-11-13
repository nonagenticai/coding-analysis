"""
Export and Rendering Functions

Export utilities and rendering functions for different visualization formats.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter

from .export_utilities import ExportUtilities

logger = logging.getLogger(__name__)

router = APIRouter()


class VisualizationExporter:
    """Handles export of visualizations to different formats."""

    def __init__(self):
        """Initialize the visualization exporter."""
        self.supported_formats = [
            "cytoscape",
            "d3",
            "graphviz",
            "mermaid",
            "json",
            "svg",
            "png",
        ]
        self.utils = ExportUtilities()

    async def export_to_cytoscape(
        self,
        nodes: list[Any],
        edges: list[Any],
        options: dict[str, Any] | None = None,
    ) -> str:
        """
        Export visualization to Cytoscape.js format.

        Args:
            nodes: List of visualization nodes
            edges: List of visualization edges
            options: Optional export configuration

        Returns:
            Cytoscape.js JSON string
        """
        try:
            options = options or {}

            # Convert nodes to Cytoscape format
            cytoscape_nodes = []
            for node in nodes:
                cyto_node = {
                    "data": {
                        "id": node.id,
                        "label": node.label,
                        "type": node.type,
                        "language": node.language,
                        "complexity": getattr(node, "complexity", 0),
                        "importance": getattr(node, "importance", 0),
                    },
                    "position": options.get("positions", {}).get(node.id, {"x": 0, "y": 0}),
                    "classes": f"node-{node.language} node-{node.type}",
                }

                # Add styling
                if hasattr(node, "color") and node.color:
                    cyto_node["style"] = {"background-color": node.color}
                if hasattr(node, "size"):
                    style_dict: dict[str, Any] = cyto_node.get("style", {})  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
                    style_dict["width"] = node.size * 20
                    style_dict["height"] = node.size * 20
                    cyto_node["style"] = style_dict

                cytoscape_nodes.append(cyto_node)

            # Convert edges to Cytoscape format
            cytoscape_edges = []
            for edge in edges:
                cyto_edge = {
                    "data": {
                        "id": edge.id,
                        "source": edge.source,
                        "target": edge.target,
                        "type": edge.type,
                        "weight": getattr(edge, "weight", 1),
                    },
                    "classes": f"edge-{edge.type}",
                }

                # Add styling
                if hasattr(edge, "color") and edge.color:
                    cyto_edge["style"] = {"line-color": edge.color}
                if hasattr(edge, "style"):
                    edge_style_dict: dict[str, Any] = cyto_edge.get("style", {})  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
                    edge_style_dict["line-style"] = edge.style
                    cyto_edge["style"] = edge_style_dict

                cytoscape_edges.append(cyto_edge)

            # Create Cytoscape.js structure
            cytoscape_data = {
                "elements": {
                    "nodes": cytoscape_nodes,
                    "edges": cytoscape_edges,
                },
                "style": self.utils.get_cytoscape_style(),
                "layout": options.get(
                    "layout",
                    {
                        "name": "force-directed",
                        "animate": True,
                        "fit": True,
                    },
                ),
            }

            return json.dumps(cytoscape_data, indent=2)

        except Exception as e:
            logger.error("Error exporting to Cytoscape format: %s", str(e))
            raise

    async def export_to_d3(
        self,
        nodes: list[Any],
        edges: list[Any],
        options: dict[str, Any] | None = None,
    ) -> str:
        """
        Export visualization to D3.js format.

        Args:
            nodes: List of visualization nodes
            edges: List of visualization edges
            options: Optional export configuration

        Returns:
            D3.js JSON string
        """
        try:
            options = options or {}

            # Convert nodes to D3 format
            d3_nodes = []
            for node in nodes:
                d3_node = {
                    "id": node.id,
                    "name": node.label,
                    "type": node.type,
                    "language": node.language,
                    "group": getattr(node, "group", node.language),
                    "size": getattr(node, "size", 1),
                    "complexity": getattr(node, "complexity", 0),
                    "importance": getattr(node, "importance", 0),
                    "color": getattr(node, "color", None),
                    "file_path": getattr(node, "file_path", None),
                }
                d3_nodes.append(d3_node)

            # Convert edges to D3 format (links)
            d3_links = []
            for edge in edges:
                d3_link = {
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "weight": getattr(edge, "weight", 1),
                    "value": getattr(edge, "weight", 1),  # D3 uses 'value' for link strength
                    "color": getattr(edge, "color", None),
                    "style": getattr(edge, "style", "solid"),
                }
                d3_links.append(d3_link)

            # Create D3.js structure
            d3_data = {
                "nodes": d3_nodes,
                "links": d3_links,
                "metadata": {
                    "node_count": len(d3_nodes),
                    "link_count": len(d3_links),
                    "languages": list(set(node["language"] for node in d3_nodes)),
                    "layout_config": options.get(
                        "layout",
                        {
                            "force": {
                                "charge": -300,
                                "link_distance": 50,
                                "gravity": 0.1,
                            }
                        },
                    ),
                },
            }

            return json.dumps(d3_data, indent=2)

        except Exception as e:
            logger.error("Error exporting to D3 format: %s", str(e))
            raise

    async def export_to_graphviz(
        self,
        nodes: list[Any],
        edges: list[Any],
        options: dict[str, Any] | None = None,
    ) -> str:
        """
        Export visualization to Graphviz DOT format.

        Args:
            nodes: List of visualization nodes
            edges: List of visualization edges
            options: Optional export configuration

        Returns:
            Graphviz DOT string
        """
        try:
            options = options or {}
            graph_type = options.get("graph_type", "digraph")
            graph_name = options.get("graph_name", "dependency_graph")

            dot_lines = [f"{graph_type} {graph_name} {{"]

            # Add graph attributes
            dot_lines.append("  rankdir=TB;")
            dot_lines.append("  node [shape=box, style=filled];")
            dot_lines.append("  edge [arrowhead=normal];")
            dot_lines.append("")

            # Add language-based subgraphs if requested
            if options.get("group_by_language", True):
                languages = set(node.language for node in nodes)
                for i, language in enumerate(languages):
                    color = self.utils.get_language_color(language, i)
                    dot_lines.append(f"  subgraph cluster_{language} {{")
                    dot_lines.append(f'    label="{language.upper()}";')
                    dot_lines.append(f'    color="{color}";')
                    dot_lines.append("    style=filled;")
                    dot_lines.append(f'    fillcolor="{color}20";')

                    # Add nodes for this language
                    for node in nodes:
                        if node.language == language:
                            node_attrs = self.utils.get_node_attributes(node, color)
                            dot_lines.append(f'    "{node.id}" [label="{node.label}"{node_attrs}];')

                    dot_lines.append("  }")
                    dot_lines.append("")
            else:
                # Add nodes without grouping
                for node in nodes:
                    color = self.utils.get_language_color(node.language)
                    node_attrs = self.utils.get_node_attributes(node, color)
                    dot_lines.append(f'  "{node.id}" [label="{node.label}"{node_attrs}];')

            # Add edges
            dot_lines.append("")
            for edge in edges:
                edge_attrs = self.utils.get_edge_attributes(edge)
                if graph_type == "digraph":
                    dot_lines.append(f'  "{edge.source}" -> "{edge.target}"{edge_attrs};')
                else:
                    dot_lines.append(f'  "{edge.source}" -- "{edge.target}"{edge_attrs};')

            dot_lines.append("}")

            return "\n".join(dot_lines)

        except Exception as e:
            logger.error("Error exporting to Graphviz format: %s", str(e))
            raise

    async def export_to_mermaid(
        self,
        nodes: list[Any],
        edges: list[Any],
        options: dict[str, Any] | None = None,
    ) -> str:
        """
        Export visualization to Mermaid diagram format.

        Args:
            nodes: List of visualization nodes
            edges: List of visualization edges
            options: Optional export configuration

        Returns:
            Mermaid diagram string
        """
        try:
            options = options or {}
            diagram_type = options.get("diagram_type", "graph")
            direction = options.get("direction", "TD")  # TD, LR, BT, RL

            mermaid_lines = [f"{diagram_type} {direction}"]

            # Add nodes with styling
            for node in nodes:
                node_id = self.utils.sanitize_mermaid_id(node.id)
                node_label = node.label.replace('"', "'")

                # Choose node shape based on type
                if node.type in ["class", "interface"]:
                    shape_start, shape_end = "[", "]"
                elif node.type in ["function", "method"]:
                    shape_start, shape_end = "(", ")"
                elif node.type in ["module", "package"]:
                    shape_start, shape_end = "[[", "]]"
                else:
                    shape_start, shape_end = "[", "]"

                mermaid_lines.append(f'  {node_id}{shape_start}"{node_label}"{shape_end}')

            # Add edges
            for edge in edges:
                source_id = self.utils.sanitize_mermaid_id(edge.source)
                target_id = self.utils.sanitize_mermaid_id(edge.target)

                # Choose arrow style based on edge type
                if edge.type == "inheritance":
                    arrow = "-->"
                elif edge.type == "dependency":
                    arrow = "-->"
                elif edge.type == "call":
                    arrow = "-.->"
                else:
                    arrow = "-->"

                # Add edge label if available
                if hasattr(edge, "label") and edge.label:
                    label = edge.label.replace('"', "'")
                    mermaid_lines.append(f"  {source_id} {arrow}|{label}| {target_id}")
                else:
                    mermaid_lines.append(f"  {source_id} {arrow} {target_id}")

            # Add styling
            if options.get("add_styling", True):
                mermaid_lines.extend(self.utils.get_mermaid_styling(nodes))

            return "\n".join(mermaid_lines)

        except Exception as e:
            logger.error("Error exporting to Mermaid format: %s", str(e))
            raise

    async def export_to_json(
        self,
        nodes: list[Any],
        edges: list[Any],
        metadata: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """
        Export visualization to generic JSON format.

        Args:
            nodes: List of visualization nodes
            edges: List of visualization edges
            metadata: Optional metadata to include
            options: Optional export configuration

        Returns:
            JSON string
        """
        try:
            # Convert nodes and edges to dict format
            json_nodes = self.utils.convert_nodes_to_dict(nodes)
            json_edges = self.utils.convert_edges_to_dict(edges)

            # Create JSON structure
            json_data = {
                "nodes": json_nodes,
                "edges": json_edges,
                "metadata": metadata or {},
                "export_options": options or {},
                "statistics": {
                    "node_count": len(json_nodes),
                    "edge_count": len(json_edges),
                    "languages": list(set(node["language"] for node in json_nodes)),
                    "node_types": list(set(node["type"] for node in json_nodes)),
                    "edge_types": list(set(edge["type"] for edge in json_edges)),
                },
            }

            return json.dumps(json_data, indent=2, default=str)

        except Exception as e:
            logger.error("Error exporting to JSON format: %s", str(e))
            raise
