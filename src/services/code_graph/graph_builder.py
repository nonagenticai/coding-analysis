"""Reusable, idempotent code-graph build (ingestion) entry point.

This is the single source of truth for "populate the code graph for a
container". Both triggers call it:

* the eager, provision-time build (coding-containers lifecycle, via the
  ``/api/analysis/analyze`` HTTP surface), and
* the synchronous self-heal in the architecture endpoints (in-process).

It delegates to the working deep-analysis path (``analyze_code``), which
resolves source files (shared-volume candidates or a remote ``find``/``cat``
exec fallback against the container's ``/workspace``), parses them with
``DeepCodeAnalyzer``, and persists nodes/edges via ``populate_graph_database``.

Historically the "build the graph" step existed but its trigger was orphaned
across two refactors (the SQLite→Postgres graph-management migration reduced
``initialize`` to a bare repo node, and the monolith→microservices split
dropped the container-lifecycle trigger). This module reconnects it.
"""

from __future__ import annotations

import logging
from typing import Any

from db.session import get_raw_connection

logger = logging.getLogger(__name__)


async def graph_is_populated(container_name: str, tenant_id: str) -> bool:
    """Return True if the graph already holds at least one parsed node for ``tenant_id``.

    The ``initialize`` lifecycle op inserts a single ``repository`` root node;
    a *populated* graph additionally has parsed entity nodes (functions,
    classes, files, …). So "populated" == any node whose type is not
    ``repository``.

    Scoped to ``tenant_id`` (NULL-tolerant for legacy rows) so a different
    tenant's graph under a colliding ``container_name`` is not mistaken for this
    tenant's — otherwise this tenant's build would be wrongly skipped.
    """
    async with get_raw_connection() as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) FROM nodes "
            "WHERE code_graph_id = %s AND (tenant_id = %s OR tenant_id IS NULL) "
            "AND type != 'repository'",
            (container_name, tenant_id),
        )
        row = await cur.fetchone()
        return bool(row and int(row[0]) > 0)


async def _clear_parsed_nodes(container_name: str, tenant_id: str) -> None:
    """Drop edges and parsed nodes for a rebuild, preserving the repo root node.

    Strictly scoped to ``tenant_id``: a destructive DELETE must only ever touch
    the caller's own attributed rows — never another tenant's, and never the
    unattributed legacy (NULL) rows that may belong to someone else.
    """
    async with get_raw_connection() as conn:
        await conn.execute(
            "DELETE FROM edges WHERE code_graph_id = %s AND tenant_id = %s",
            (container_name, tenant_id),
        )
        await conn.execute(
            "DELETE FROM nodes WHERE code_graph_id = %s AND tenant_id = %s AND type != 'repository'",
            (container_name, tenant_id),
        )
        await conn.commit()


async def build_code_graph(
    container_name: str,
    tenant_id: str,
    *,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Idempotently build the code graph for ``container_name``.

    Returns ``{"nodes": int, "edges": int, "skipped": bool}``.

    Idempotent: if the graph is already populated and ``force_rebuild`` is
    False, this is a cheap no-op (one ``COUNT(*)``). With ``force_rebuild`` the
    existing parsed nodes/edges are cleared first so stale structure cannot
    survive a re-analysis (``populate_graph_database`` uses
    ``ON CONFLICT DO NOTHING`` and would otherwise keep stale rows).

    Raises only on genuine errors from the underlying analysis path; callers
    that must stay 500-free (e.g. the architecture self-heal) wrap the call.
    """
    if not force_rebuild and await graph_is_populated(container_name, tenant_id):
        logger.info("Code graph for %s already populated; skipping build", container_name)
        return {"nodes": 0, "edges": 0, "skipped": True}

    if force_rebuild:
        await _clear_parsed_nodes(container_name, tenant_id)

    # Lazy imports avoid an endpoint<->service import cycle (the endpoint module
    # imports many services; importing it at module load would be circular).
    from app.api.endpoints.analysis_modules.analysis_endpoints import analyze_code
    from app.api.models.analysis_api_models import AnalysisRequest, PydanticAIAnalysisType

    request = AnalysisRequest(
        analysis_type=PydanticAIAnalysisType.DEEP,
        container_name=container_name,
        language="python",
    )

    # Call the endpoint function directly. ``tenant_id`` is passed explicitly,
    # so FastAPI's Depends(require_tenant) is bypassed (it only runs under the
    # request lifecycle); the deep path resolves the container, source files,
    # parses, and persists the graph.
    response = await analyze_code(request, tenant_id=tenant_id)

    meta = (getattr(response.result, "metadata", None) or {}) if response else {}
    nodes = int(meta.get("graph_nodes", 0) or 0)
    edges = int(meta.get("graph_edges", 0) or 0)
    logger.info(
        "Built code graph for %s: %d nodes, %d edges (files_analyzed=%s)",
        container_name,
        nodes,
        edges,
        meta.get("files_analyzed"),
    )
    return {"nodes": nodes, "edges": edges, "skipped": False}
