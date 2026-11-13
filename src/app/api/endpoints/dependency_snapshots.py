"""Reader for stored dependency-report snapshots (the graph-style fallback).

The fr "Dependency Report" is normally computed live by coding-security, which
execs into the running job container. When that container Deployment is gone the
live path 404s. This endpoint serves the last snapshot persisted at analysis time
(see ``services/code_graph/dependency_snapshot.py`` and migration ``0003``), so fr
can degrade to stored data exactly like the Architecture graph does.

Authorization is by ``require_tenant`` + a strict ``tenant_id`` row match ONLY —
deliberately NOT ``validate_and_get_container_for_tenant`` (which hits the
coding-containers registry): the whole point of this reader is to work AFTER the
container is deprovisioned, when that validation would itself 404.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, ConfigDict, Field

from auth.tenant import require_tenant
from db.session import get_raw_connection

logger = logging.getLogger(__name__)

router = APIRouter()

_NAME_PATTERN = r"^[A-Za-z0-9._-]+$"


class DependencySnapshot(BaseModel):
    """A stored dependency-report snapshot for a container."""

    container_name: str
    computed_at: datetime
    report: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


@router.get(
    "/{container_name}",
    response_model=DependencySnapshot,
    operation_id="analysis_get_dependency_snapshot",
    summary="Stored dependency report snapshot",
)
async def get_dependency_snapshot(
    container_name: str = Path(..., pattern=_NAME_PATTERN),
    tenant_id: str = Depends(require_tenant),
) -> DependencySnapshot:
    """Return the last persisted dependency report for a container.

    Returns a stable 404 (not a crash) when no snapshot exists for the caller's
    tenant — mirroring ``graphs_reader``'s missing-graph shape.
    """
    async with get_raw_connection() as conn:
        cur = await conn.execute(
            "SELECT report, computed_at FROM dependency_report_snapshots "
            "WHERE code_graph_id = %s AND tenant_id = %s",
            (container_name, tenant_id),
        )
        row = await cur.fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No stored dependency report for container '{container_name}'",
        )

    report, computed_at = row
    return DependencySnapshot(
        container_name=container_name,
        computed_at=computed_at,
        report=report if isinstance(report, dict) else {},
    )
