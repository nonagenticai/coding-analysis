"""HTTP access to the populated graph databases in analysis-owned Postgres.

The writer side of the graph (``populate_graph_database`` in ``analysis_utils``)
has always lived here in coding-analysis; the reader side used to live in
coding-gateway, which opened a SQLite file directly. That only worked when
both services shared a volume — with gateway and analysis landing on different
nodes it produced silent 404s ("Database not found") on Architecture queries.

This module exposes both the reader and the lifecycle-management operations
(list / initialize / delete) over HTTP so they run in the same pod that owns
the Postgres store. coding-gateway's ``/api/graphs/*`` handlers forward here
instead of touching SQLite themselves.

Read endpoints are paginated where relevant and tolerate databases that do not
exist yet (404 with a stable shape, not a crash). Writes operate by
``code_graph_id`` (== container name) and commit explicitly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from auth.tenant import require_tenant
from db.session import get_raw_connection

from ..utils.container_validation import validate_and_get_container_for_tenant

logger = logging.getLogger(__name__)

router = APIRouter()

_NAME_PATTERN = r"^[A-Za-z0-9._-]+$"

# A container's code graph lives in the shared PostgreSQL database scoped by
# ``code_graph_id`` == the container name (the writer, ``populate_graph_database``
# in ``analysis_utils``, uses the same key).


async def _require_graph_or_404(conn, container_name: str) -> None:
    """Raise a stable 404 if the container's graph has no rows yet."""
    cur = await conn.execute(
        "SELECT 1 FROM nodes WHERE code_graph_id = %s LIMIT 1", (container_name,)
    )
    if await cur.fetchone() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Graph database for container '{container_name}' not found",
        )


def _decode_properties(raw: Any) -> dict[str, Any]:
    """Best-effort decode of the properties TEXT column into a dict."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
        if isinstance(decoded, dict):
            return decoded
        return {"value": decoded}
    return {"value": raw}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    id: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class GraphEdge(BaseModel):
    source: str
    target: str
    relationship: str
    properties: dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0

    model_config = ConfigDict(extra="allow")


class GraphMetadata(BaseModel):
    container_name: str
    exists: bool
    node_count: int
    edge_count: int


class AugmentedGraph(BaseModel):
    container_name: str
    node_count: int
    edge_count: int
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class NodesPage(BaseModel):
    container_name: str
    total: int
    limit: int
    offset: int
    nodes: list[GraphNode]


class EdgesPage(BaseModel):
    container_name: str
    total: int
    limit: int
    offset: int
    edges: list[GraphEdge]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/{container_name}",
    response_model=GraphMetadata,
    operation_id="analysis_get_graph_metadata",
    summary="Graph database metadata",
)
async def get_graph_metadata(
    container_name: str = Path(..., pattern=_NAME_PATTERN),
    tenant_id: str = Depends(require_tenant),
) -> GraphMetadata:
    """Return node/edge counts for a container's graph database.

    Returns 404 if the database does not exist, but with a shape that makes
    callers' JSON parsing straightforward.
    """
    await validate_and_get_container_for_tenant(container_name, tenant_id)
    async with get_raw_connection() as conn:
        await _require_graph_or_404(conn, container_name)
        cur = await conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE code_graph_id = %s", (container_name,)
        )
        row = await cur.fetchone()
        node_count = int(row[0]) if row else 0
        cur = await conn.execute(
            "SELECT COUNT(*) FROM edges WHERE code_graph_id = %s", (container_name,)
        )
        row = await cur.fetchone()
        edge_count = int(row[0]) if row else 0
    return GraphMetadata(
        container_name=container_name,
        exists=True,
        node_count=node_count,
        edge_count=edge_count,
    )


@router.get(
    "/{container_name}/nodes",
    response_model=NodesPage,
    operation_id="analysis_list_graph_nodes",
    summary="List graph nodes",
)
async def list_graph_nodes(
    container_name: str = Path(..., pattern=_NAME_PATTERN),
    node_type: str | None = Query(None, description="Filter by node type"),
    name_pattern: str | None = Query(
        None,
        description=(
            "Substring match against properties.name (case-insensitive). "
            "Uses SQLite LIKE with escaped %/_ so the literal is matched "
            "verbatim, no wildcard injection."
        ),
    ),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    tenant_id: str = Depends(require_tenant),
) -> NodesPage:
    """List nodes with optional type and name filters.

    ``name_pattern`` matches ``properties->>'name'`` case-insensitively via
    ``LIKE %s ESCAPE '\\\\'``. The handler escapes ``%``, ``_``, and
    backslashes in the caller-supplied literal so users can't slip wildcards
    into the pattern; only the outer ``%...%`` wrap is under our control.
    """
    clauses: list[str] = ["code_graph_id = %s"]
    params: list[Any] = [container_name]
    if node_type:
        clauses.append("type = %s")
        params.append(node_type)
    if name_pattern:
        # Escape LIKE wildcards + the escape char itself so the user's literal
        # can't act as a pattern.
        escaped = name_pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        clauses.append("LOWER(COALESCE(properties->>'name', '')) LIKE %s ESCAPE '\\'")
        params.append(f"%{escaped.lower()}%")

    # Build the WHERE out of literal-only clauses; user input stays bound.
    where = f"WHERE {' AND '.join(clauses)}"

    await validate_and_get_container_for_tenant(container_name, tenant_id)
    async with get_raw_connection() as conn:
        await _require_graph_or_404(conn, container_name)
        cur = await conn.execute(
            f"SELECT COUNT(*) FROM nodes {where}",  # nosec B608  # ty: ignore[invalid-argument-type]
            params,
        )
        row = await cur.fetchone()
        total = int(row[0]) if row else 0
        cur = await conn.execute(
            f"SELECT id, type, properties FROM nodes {where} "  # nosec B608  # ty: ignore[invalid-argument-type]
            "ORDER BY id LIMIT %s OFFSET %s",
            [*params, limit, offset],
        )
        rows = await cur.fetchall()

    nodes = [
        GraphNode(id=str(r[0]), type=str(r[1]), properties=_decode_properties(r[2])) for r in rows
    ]
    return NodesPage(
        container_name=container_name,
        total=total,
        limit=limit,
        offset=offset,
        nodes=nodes,
    )


@router.get(
    "/{container_name}/edges",
    response_model=EdgesPage,
    operation_id="analysis_list_graph_edges",
    summary="List graph edges",
)
async def list_graph_edges(
    container_name: str = Path(..., pattern=_NAME_PATTERN),
    source: str | None = Query(None, description="Filter by source node id"),
    target: str | None = Query(None, description="Filter by target node id"),
    relationship: str | None = Query(None, description="Filter by relationship"),
    limit: int = Query(1000, ge=1, le=10000),
    offset: int = Query(0, ge=0),
    tenant_id: str = Depends(require_tenant),
) -> EdgesPage:
    await validate_and_get_container_for_tenant(container_name, tenant_id)
    clauses: list[str] = ["code_graph_id = %s"]
    params: list[Any] = [container_name]
    if source:
        clauses.append("source = %s")
        params.append(source)
    if target:
        clauses.append("target = %s")
        params.append(target)
    if relationship:
        clauses.append("relationship = %s")
        params.append(relationship)
    # ``where`` is built exclusively from string literals owned by this
    # function; the user-provided values live in ``params`` and are bound
    # positionally. No interpolated value reaches the SQL text.
    where = f"WHERE {' AND '.join(clauses)}"

    async with get_raw_connection() as conn:
        await _require_graph_or_404(conn, container_name)
        cur = await conn.execute(
            f"SELECT COUNT(*) FROM edges {where}",  # nosec B608  # ty: ignore[invalid-argument-type]
            params,
        )
        row = await cur.fetchone()
        total = int(row[0]) if row else 0
        cur = await conn.execute(
            f"SELECT source, target, relationship, properties, weight FROM edges {where} "  # nosec B608  # ty: ignore[invalid-argument-type]
            "ORDER BY id LIMIT %s OFFSET %s",
            [*params, limit, offset],
        )
        rows = await cur.fetchall()

    edges = [
        GraphEdge(
            source=str(r[0]),
            target=str(r[1]),
            relationship=str(r[2]),
            properties=_decode_properties(r[3]),
            weight=float(r[4]) if r[4] is not None else 1.0,
        )
        for r in rows
    ]
    return EdgesPage(
        container_name=container_name,
        total=total,
        limit=limit,
        offset=offset,
        edges=edges,
    )


@router.get(
    "/{container_name}/augmented",
    response_model=AugmentedGraph,
    operation_id="analysis_get_augmented_graph",
    summary="Full graph as nodes + edges in one payload",
)
async def get_augmented_graph(
    container_name: str = Path(..., pattern=_NAME_PATTERN),
    node_limit: int = Query(2000, ge=1, le=20000),
    edge_limit: int = Query(5000, ge=1, le=50000),
    tenant_id: str = Depends(require_tenant),
) -> AugmentedGraph:
    """Return the complete graph in one round-trip.

    This is the primary consumer endpoint — coding-gateway's
    ``/api/graphs/{c}/augmented`` forwards straight here, which in turn
    feeds fr-frontend's Architecture tab via ``adaptAugmentedGraph`` in
    fr-backend. ``node_limit``/``edge_limit`` exist so a pathologically
    large graph can't blow up the response; callers needing exhaustive
    data should paginate via ``/nodes`` and ``/edges`` instead.
    """
    await validate_and_get_container_for_tenant(container_name, tenant_id)
    async with get_raw_connection() as conn:
        await _require_graph_or_404(conn, container_name)
        cur = await conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE code_graph_id = %s", (container_name,)
        )
        row = await cur.fetchone()
        node_count = int(row[0]) if row else 0
        cur = await conn.execute(
            "SELECT COUNT(*) FROM edges WHERE code_graph_id = %s", (container_name,)
        )
        row = await cur.fetchone()
        edge_count = int(row[0]) if row else 0

        cur = await conn.execute(
            "SELECT id, type, properties FROM nodes WHERE code_graph_id = %s ORDER BY id LIMIT %s",
            (container_name, node_limit),
        )
        node_rows = await cur.fetchall()
        cur = await conn.execute(
            "SELECT source, target, relationship, properties, weight FROM edges "
            "WHERE code_graph_id = %s ORDER BY id LIMIT %s",
            (container_name, edge_limit),
        )
        edge_rows = await cur.fetchall()

    nodes = [
        GraphNode(id=str(r[0]), type=str(r[1]), properties=_decode_properties(r[2]))
        for r in node_rows
    ]
    edges = [
        GraphEdge(
            source=str(r[0]),
            target=str(r[1]),
            relationship=str(r[2]),
            properties=_decode_properties(r[3]),
            weight=float(r[4]) if r[4] is not None else 1.0,
        )
        for r in edge_rows
    ]
    return AugmentedGraph(
        container_name=container_name,
        node_count=node_count,
        edge_count=edge_count,
        nodes=nodes,
        edges=edges,
    )


# ---------------------------------------------------------------------------
# Management (list / initialize / delete)
# ---------------------------------------------------------------------------
#
# These are the lifecycle operations coding-gateway used to perform against its
# own ephemeral SQLite files (initialize / delete / orphan-scan). They now live
# here so the single Postgres store is the only writer. Like the readers, they
# key on ``code_graph_id`` == container name and use ``require_tenant`` for
# authz only (graphs are single-tenant in practice).


class GraphSummary(BaseModel):
    container_name: str
    node_count: int
    edge_count: int
    tenant_id: str | None = None


class GraphList(BaseModel):
    graphs: list[GraphSummary]
    total: int


class GraphDeleted(BaseModel):
    container_name: str
    deleted: bool
    nodes_deleted: int
    edges_deleted: int


class GraphInitialized(BaseModel):
    container_name: str
    created: bool
    repo_node_id: str


class GraphInitRequest(BaseModel):
    codebase_path: str | None = None
    language: str = "python"
    force_rebuild: bool = False

    model_config = ConfigDict(extra="ignore")


@router.get(
    "",
    response_model=GraphList,
    operation_id="analysis_list_graphs",
    summary="List all graph databases",
)
@router.get("/", response_model=GraphList, include_in_schema=False)
async def list_graphs(tenant_id: str = Depends(require_tenant)) -> GraphList:
    """List every graph (distinct ``code_graph_id``) with node/edge counts.

    Scoped to the caller's tenant: only graphs whose rows are attributed to
    ``tenant_id`` are listed, so one tenant cannot enumerate another's graph
    names. (Legacy rows written before tenant attribution carry a NULL
    ``tenant_id`` and are not listed until re-analysed; individual graphs remain
    reachable via the per-container endpoints, which authorise by container
    ownership.)
    """
    async with get_raw_connection() as conn:
        cur = await conn.execute(
            # tenant_id is a native uuid (migration 0005). Two consequences here:
            # postgres has NO min(uuid) aggregate (measured on PG18: "function
            # min(uuid) does not exist"), and `tenant_id != ''` is 22P02 rather
            # than False because '' is not a uuid — this one statement carried
            # both. Aggregating over ::text fixes the first; the second arm is
            # simply gone, since a uuid column cannot hold an empty string. The
            # outer WHERE already pins every row to the caller's own tenant, so
            # MIN is a formality that keeps the aggregate well-formed.
            "SELECT code_graph_id, COUNT(*), MIN(tenant_id::text) "
            "FROM nodes WHERE tenant_id = %s GROUP BY code_graph_id ORDER BY code_graph_id",
            (tenant_id,),
        )
        node_rows = await cur.fetchall()
        cur = await conn.execute("SELECT code_graph_id, COUNT(*) FROM edges GROUP BY code_graph_id")
        edge_rows = await cur.fetchall()

    edge_counts = {str(r[0]): int(r[1]) for r in edge_rows}
    graphs = [
        GraphSummary(
            container_name=str(r[0]),
            node_count=int(r[1]),
            edge_count=edge_counts.get(str(r[0]), 0),
            tenant_id=str(r[2]) if r[2] else None,
        )
        for r in node_rows
    ]
    return GraphList(graphs=graphs, total=len(graphs))


@router.post(
    "/{container_name}",
    response_model=GraphInitialized,
    status_code=status.HTTP_201_CREATED,
    operation_id="analysis_initialize_graph",
    summary="Initialize a graph database",
)
async def initialize_graph(
    container_name: str = Path(..., pattern=_NAME_PATTERN),
    body: GraphInitRequest = Body(default_factory=GraphInitRequest),
    tenant_id: str = Depends(require_tenant),
) -> GraphInitialized:
    """Create a minimal graph (a single ``repository`` node) for a container.

    Mirrors the gateway's old ``initialize`` behaviour: 409 if a graph already
    exists unless ``force_rebuild`` is set, in which case the existing graph is
    cleared first. Idempotent at the repo-node level via ``ON CONFLICT``.
    """
    await validate_and_get_container_for_tenant(container_name, tenant_id)
    repo_node_id = f"repo_{container_name}"
    async with get_raw_connection() as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE code_graph_id = %s", (container_name,)
        )
        row = await cur.fetchone()
        existing = int(row[0]) if row else 0

        if existing > 0 and not body.force_rebuild:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Code graph for container '{container_name}' already exists. "
                    "Use force_rebuild=true or delete the existing graph first."
                ),
            )
        if existing > 0 and body.force_rebuild:
            await conn.execute("DELETE FROM edges WHERE code_graph_id = %s", (container_name,))
            await conn.execute("DELETE FROM nodes WHERE code_graph_id = %s", (container_name,))

        properties = {
            "container_name": container_name,
            "codebase_path": body.codebase_path or f"/data/{container_name}",
            "name": container_name,
            "language": body.language,
        }
        await conn.execute(
            "INSERT INTO nodes (code_graph_id, id, type, properties, tenant_id) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (code_graph_id, tenant_id, id) DO NOTHING",
            (container_name, repo_node_id, "repository", Jsonb(properties), tenant_id),
        )
        await conn.commit()

    return GraphInitialized(
        container_name=container_name,
        created=existing == 0,
        repo_node_id=repo_node_id,
    )


@router.delete(
    "/{container_name}",
    response_model=GraphDeleted,
    operation_id="analysis_delete_graph",
    summary="Delete a graph database",
)
async def delete_graph(
    container_name: str = Path(..., pattern=_NAME_PATTERN),
    tenant_id: str = Depends(require_tenant),
) -> GraphDeleted:
    """Delete a container's graph (all edges then nodes). 404 if it has no rows."""
    await validate_and_get_container_for_tenant(container_name, tenant_id)
    async with get_raw_connection() as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE code_graph_id = %s", (container_name,)
        )
        row = await cur.fetchone()
        node_count = int(row[0]) if row else 0
        cur = await conn.execute(
            "SELECT COUNT(*) FROM edges WHERE code_graph_id = %s", (container_name,)
        )
        row = await cur.fetchone()
        edge_count = int(row[0]) if row else 0

        if node_count == 0 and edge_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Graph database for container '{container_name}' not found",
            )

        # Edges carry an FK to nodes(code_graph_id, source/target), so delete them first.
        await conn.execute("DELETE FROM edges WHERE code_graph_id = %s", (container_name,))
        await conn.execute("DELETE FROM nodes WHERE code_graph_id = %s", (container_name,))
        await conn.commit()

    return GraphDeleted(
        container_name=container_name,
        deleted=True,
        nodes_deleted=node_count,
        edges_deleted=edge_count,
    )
