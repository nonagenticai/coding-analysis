"""Container API models stub.

TODO: Implement container-related API models.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class ContainerOperation(StrEnum):
    """Container operations."""

    START = "start"
    STOP = "stop"
    RESTART = "restart"


class ContainerOperationRequest(BaseModel):
    """Container operation request."""

    container_id: str
    operation: ContainerOperation


class ContainerOperationResponse(BaseModel):
    """Container operation response."""

    success: bool = False
    message: str = ""


class ContainerListResponse(BaseModel):
    """Container list response."""

    containers: list[dict[str, Any]] = []
    count: int = 0


class ContainerDetailResponse(BaseModel):
    """Container detail response."""

    container_id: str = ""
    name: str = ""
    status: str = ""
    details: dict[str, Any] = {}


class ContainerStatusResponse(BaseModel):
    """Container status response."""

    container_id: str = ""
    status: str = ""
    healthy: bool = False
