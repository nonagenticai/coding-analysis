"""Adapter for integrating graph structures with multi-language processing services."""

from __future__ import annotations

import contextlib
from typing import Any

# Try to import networkx, but make it optional
try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    # NetworkX integration is optional - application will work without graph analysis
    nx: Any = None

from services.code_graph.multi_language_graph_service import (
    MultiLanguageGraphService,
)


class GraphAdapter:
    """Adapter between MultiLanguageGraphService and NetworkX graphs for architecture analysis."""

    def __init__(self, graph_service: MultiLanguageGraphService) -> None:
        self._service = graph_service

    async def load_graph(self, codebase_id: str, tenant_id: str) -> nx.DiGraph:
        """Load a NetworkX DiGraph from the graph service for the given codebase.

        Scoped to ``tenant_id``: the architecture read path reaches this after the
        container-ownership boundary guard, but ``code_graph_id == container_name``
        is a namespace SHARED across tenants — without a tenant predicate two
        tenants using the same container name would read each other's graph. Reads
        are NULL-tolerant (``tenant_id = %s OR tenant_id IS NULL``) so legacy rows
        written before tenant attribution (idx_node_cg_tenant / idx_edge_cg_tenant)
        still load and cycle detection does not regress.
        """
        if not HAS_NETWORKX:
            raise RuntimeError("NetworkX not available - graph analysis disabled")

        if not self._service.entity_mapper:
            raise RuntimeError("Graph service not initialized")

        # Create a new directed graph
        graph: nx.DiGraph = nx.DiGraph()

        # Get all entities from the database
        entities = await self._get_all_entities(codebase_id, tenant_id)

        # Add nodes to the graph
        for entity in entities:
            graph.add_node(
                entity["id"],
                type=entity["type"],
                name=entity.get("name"),
                language=entity["language"],
                # Default to "" (not None) so the extractors' Path(file_path)
                # calls never hit Path(None); nodes without a path (e.g. the repo
                # root) simply don't match any file/folder grouping.
                file_path=entity["file_path"] or "",
                line_number=entity.get("line_number", 0),
                properties=entity.get("properties", {}),
                code=entity.get("code", ""),
            )

        # Get all relationships from the database
        relationships = await self._get_all_relationships(codebase_id, tenant_id)

        # Add edges to the graph
        for rel in relationships:
            if graph.has_node(rel["source"]) and graph.has_node(rel["target"]):
                graph.add_edge(
                    rel["source"],
                    rel["target"],
                    relationship=rel["relationship"],
                    properties=rel.get("properties", {}),
                )

        return graph

    async def _get_all_entities(self, codebase_id: str, tenant_id: str) -> list[dict[str, Any]]:
        """Get all entities for a codebase from the database, scoped to ``tenant_id``.

        Runs through ``db.run_read`` so a connection Postgres/PgBouncer has closed
        under the long-lived singleton is replaced (and the read retried once)
        instead of 500-ing every request until the pod restarts.
        """
        return await self._service.db.run_read(
            lambda conn: self._query_entities(conn, codebase_id, tenant_id)
        )

    async def _query_entities(
        self, conn: Any, codebase_id: str, tenant_id: str
    ) -> list[dict[str, Any]]:
        try:
            # Rows are persisted under `code_graph_id = container_name` (see
            # graph_database_populator / entity_mapper), and the `codebase_id`
            # argument IS that container name. Scope by it directly. The old
            # `properties->>'codebase_id'` sub-filter was dead — that key is never
            # written, so it excluded every row whenever codebase_id was non-NULL.
            # Additionally scope by `tenant_id` so a colliding container name in a
            # different tenant can't leak its graph; NULL-tolerant so pre-attribution
            # (legacy) rows still load (no cycle-detection regression).
            cursor = await conn.execute(
                """
                SELECT id, type, language, file_path, line_number, properties, docstring
                FROM nodes
                WHERE code_graph_id = %s AND (tenant_id = %s OR tenant_id IS NULL)
                ORDER BY language, type, line_number
                """,
                (codebase_id, tenant_id),
            )

            entities = []
            async for row in cursor:
                # name/file_path/line_number/language are written into the
                # ``properties`` JSON by the ingestion writers
                # (populate_graph_database, entity_mapper) and the dedicated
                # columns are left NULL. Coalesce column -> properties so the
                # architecture extractors (which group by file_path) see the real
                # values instead of None. Without this, a fully-parsed graph still
                # yields 0 components.
                props = row[5] or {}
                file_path = row[3] if row[3] is not None else props.get("file_path")
                line_number = row[4] if row[4] is not None else props.get("line_number", 0)
                language = row[2] if row[2] is not None else props.get("language", "unknown")
                entities.append(
                    {
                        "id": row[0],
                        "type": row[1],
                        "name": props.get("name"),
                        "language": language,
                        "file_path": file_path,
                        "line_number": line_number,
                        "properties": props,
                        "code": row[6] or props.get("code", ""),
                    }
                )

            return entities
        finally:
            # Always end the transaction this method opened. A bare SELECT (with
            # autocommit off) leaves the connection idle-in-transaction; the
            # long-lived pooled connection — held by a per-code_graph singleton —
            # is then terminated by Postgres's idle-in-transaction timeout and
            # reused closed on the next request ("current transaction is aborted" /
            # "connection is closed" → 500). Rolling back is correct on both paths:
            # this method only runs SELECTs (nothing to commit) and rollback also
            # clears an aborted txn after a failed query. Ending the read txn here
            # also means later external I/O in process() (the osv.dev dependency
            # scan) never runs while a graph-DB transaction is held open.
            with contextlib.suppress(Exception):
                await conn.rollback()

    async def _get_all_relationships(
        self, codebase_id: str, tenant_id: str
    ) -> list[dict[str, Any]]:
        """Get all relationships for a codebase from the database, scoped to ``tenant_id``."""
        return await self._service.db.run_read(
            lambda conn: self._query_relationships(conn, codebase_id, tenant_id)
        )

    async def _query_relationships(
        self, conn: Any, codebase_id: str, tenant_id: str
    ) -> list[dict[str, Any]]:
        try:
            # See `_get_all_entities`: scope by the per-codebase `code_graph_id`
            # (== the container name passed as codebase_id) AND by `tenant_id`
            # (NULL-tolerant for legacy rows) so a colliding container name in a
            # different tenant cannot leak its edges. Roll back on error so a failed
            # query cannot poison the shared pooled connection. The edge JOINs
            # already pin both endpoints to the same code_graph_id.
            cursor = await conn.execute(
                """
                SELECT e.source, e.target, e.relationship, e.properties
                FROM edges e
                JOIN nodes n1 ON e.code_graph_id = n1.code_graph_id AND e.source = n1.id
                JOIN nodes n2 ON e.code_graph_id = n2.code_graph_id AND e.target = n2.id
                WHERE e.code_graph_id = %s AND (e.tenant_id = %s OR e.tenant_id IS NULL)
                """,
                (codebase_id, tenant_id),
            )

            relationships = []
            async for row in cursor:
                relationships.append(
                    {
                        "source": row[0],
                        "target": row[1],
                        "relationship": row[2],
                        "properties": row[3] or {},
                    }
                )

            return relationships
        finally:
            # End the read-only transaction so the pooled connection is never left
            # idle-in-transaction (see _get_all_entities for the full rationale).
            with contextlib.suppress(Exception):
                await conn.rollback()

    def get_service(self) -> MultiLanguageGraphService:
        """Get the underlying graph service."""
        return self._service
