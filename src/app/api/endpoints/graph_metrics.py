"""Graph-theoretic metrics over the analysis-owned Postgres code-graph store.

This compute used to live in coding-gateway's ``CodeGraphMetricsService``, which
read a per-container SQLite file. coding-analysis already owns the postgres-backed
``CodeGraphMetricsService`` (``services.code_graph_metrics``) — these endpoints
simply expose it over HTTP so the gateway can proxy here instead of opening a
database. Everything is scoped by ``code_graph_id`` (== container name) inside the
service, so one container's metrics can never include another's rows.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, Field

from auth.tenant import require_tenant
from db.facade import GraphDatabaseConnection
from models.core_models import OverallGraphMetrics
from services.code_graph_metrics import CodeGraphMetricsService

from ..utils.container_validation import validate_and_get_container_for_tenant

logger = logging.getLogger(__name__)

router = APIRouter()

_NAME_PATTERN = r"^[A-Za-z0-9._-]+$"


async def _require_graph_or_404(conn: Any, container_name: str) -> None:
    """Raise a stable 404 if the container's graph has no rows yet."""
    cur = await conn.execute(
        "SELECT 1 FROM nodes WHERE code_graph_id = %s LIMIT 1", (container_name,)
    )
    if await cur.fetchone() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Graph database for container '{container_name}' not found",
        )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class NodeCentrality(BaseModel):
    degree_centrality: float = 0.0
    in_degree: int = 0
    out_degree: int = 0
    total_degree: int = 0


class CentralityMetrics(BaseModel):
    container_name: str
    entity_type: str
    metrics: dict[str, NodeCentrality] = Field(default_factory=dict)


class EntityNetworkMetrics(BaseModel):
    container_name: str
    entity_name: str
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    pagerank: float = 0.0
    network_importance: float = 0.0
    in_degree: int = 0
    out_degree: int = 0
    total_degree: int = 0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/{container_name}/metrics",
    response_model=OverallGraphMetrics,
    operation_id="analysis_graph_metrics",
    summary="Overall graph metrics",
)
async def get_graph_metrics(
    container_name: str = Path(..., pattern=_NAME_PATTERN),
    tenant_id: str = Depends(require_tenant),
) -> OverallGraphMetrics:
    """Node/edge counts, type breakdowns, density and average degree for a graph."""
    await validate_and_get_container_for_tenant(container_name, tenant_id)
    db = GraphDatabaseConnection(container_name)
    await db.connect()
    try:
        await _require_graph_or_404(db.get_connection(), container_name)
        return await CodeGraphMetricsService(db).get_graph_metrics()
    finally:
        await db.close()


@router.get(
    "/{container_name}/metrics/centrality",
    response_model=CentralityMetrics,
    operation_id="analysis_graph_centrality",
    summary="Centrality metrics for all nodes of a type",
)
async def get_centrality_metrics(
    container_name: str = Path(..., pattern=_NAME_PATTERN),
    entity_type: str = Query(..., description="Node type to compute centrality for"),
    tenant_id: str = Depends(require_tenant),
) -> CentralityMetrics:
    """Degree-centrality metrics for every node of ``entity_type`` in the graph."""
    await validate_and_get_container_for_tenant(container_name, tenant_id)
    db = GraphDatabaseConnection(container_name)
    await db.connect()
    try:
        await _require_graph_or_404(db.get_connection(), container_name)
        raw = await CodeGraphMetricsService(db).calculate_centrality_metrics(entity_type)
    finally:
        await db.close()
    return CentralityMetrics(
        container_name=container_name,
        entity_type=entity_type,
        metrics={
            node_id: NodeCentrality(
                degree_centrality=values.get("degree_centrality", 0.0),
                in_degree=int(values.get("in_degree", 0)),
                out_degree=int(values.get("out_degree", 0)),
                total_degree=int(values.get("total_degree", 0)),
            )
            for node_id, values in raw.items()
        },
    )


@router.get(
    "/{container_name}/metrics/entity/{entity_name}",
    response_model=EntityNetworkMetrics,
    operation_id="analysis_graph_entity_metrics",
    summary="Network metrics for a single entity",
)
async def get_entity_network_metrics(
    container_name: str = Path(..., pattern=_NAME_PATTERN),
    entity_name: str = Path(...),
    tenant_id: str = Depends(require_tenant),
) -> EntityNetworkMetrics:
    """Degree/betweenness/closeness/pagerank + network importance for one entity."""
    await validate_and_get_container_for_tenant(container_name, tenant_id)
    db = GraphDatabaseConnection(container_name)
    await db.connect()
    try:
        await _require_graph_or_404(db.get_connection(), container_name)
        raw = await CodeGraphMetricsService(db).get_function_network_metrics(entity_name)
    finally:
        await db.close()
    return EntityNetworkMetrics(
        container_name=container_name,
        entity_name=entity_name,
        degree_centrality=raw.get("degree_centrality", 0.0),
        betweenness_centrality=raw.get("betweenness_centrality", 0.0),
        closeness_centrality=raw.get("closeness_centrality", 0.0),
        pagerank=raw.get("pagerank", 0.0),
        network_importance=raw.get("network_importance", 0.0),
        in_degree=raw.get("in_degree", 0),
        out_degree=raw.get("out_degree", 0),
        total_degree=raw.get("total_degree", 0),
    )
