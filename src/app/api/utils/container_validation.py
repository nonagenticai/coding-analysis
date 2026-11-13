"""Container validation utilities.

Validates container access with tenant isolation using DockerService.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from fastapi import HTTPException

logger = logging.getLogger(__name__)


async def validate_and_get_container_for_tenant(
    container_name: str,
    tenant_id: str | None = None,
    **kwargs: Any,
) -> SimpleNamespace:
    """Validate container access for a tenant.

    Args:
        container_name: Name of the container to validate.
        tenant_id: Optional tenant ID for isolation.

    Returns:
        Container info as SimpleNamespace with ``validated=True`` added.

    Raises:
        HTTPException: If container is not found (404) or access is denied (403).
    """
    from services.docker_service import get_docker_service

    docker_service = get_docker_service()

    container = await docker_service.find_container_by_name(container_name, tenant_id)
    if container is None:
        raise HTTPException(status_code=404, detail=f"Container '{container_name}' not found")

    if tenant_id is not None:
        container_tenant = container.get("tenant_id") or container.get("labels", {}).get(
            "tenant_id"
        )
        if container_tenant is not None and container_tenant != tenant_id:
            raise HTTPException(
                status_code=403, detail="Access denied: container does not belong to tenant"
            )

    return SimpleNamespace(**{**container, "validated": True})
