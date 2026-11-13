"""Graph API models stub.

TODO: Implement graph-related API models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class GraphNode(BaseModel):
    """Graph node."""

    id: str
    label: str = ""
    type: str = ""
    properties: dict[str, Any] = {}


class GraphEdge(BaseModel):
    """Graph edge."""

    source: str
    target: str
    type: str = ""
    properties: dict[str, Any] = {}


class GraphInfo(BaseModel):
    """Graph information."""

    id: str = ""
    name: str = ""
    node_count: int = 0
    edge_count: int = 0


class GraphQuery(BaseModel):
    """Graph query."""

    query: str
    parameters: dict[str, Any] = {}


class GraphListResponse(BaseModel):
    """Graph list response."""

    graphs: list[GraphInfo] = []


class GraphDetailResponse(BaseModel):
    """Graph detail response."""

    graph: GraphInfo | None = None
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []


class NodesResponse(BaseModel):
    """Nodes response."""

    nodes: list[GraphNode] = []
    count: int = 0


class EdgesResponse(BaseModel):
    """Edges response."""

    edges: list[GraphEdge] = []
    count: int = 0


class GraphQueryResponse(BaseModel):
    """Graph query response."""

    results: list[dict[str, Any]] = []
    count: int = 0


class GraphDeletionResponse(BaseModel):
    """Graph deletion response."""

    deleted: bool = False
    graph_id: str = ""
