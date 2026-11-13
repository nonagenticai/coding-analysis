"""Tenant middleware + dependencies — mostly re-exported from coding_client SDK.

The canonical implementation lives in ``coding_client.auth.tenant`` so all
coding-* services share identical tenant-extraction behaviour. The previous
local copy was a near byte-clone of the SDK module; the only divergence (a
local ``_settings()`` helper reading ``LICENSE_INTROSPECT_URL`` from pydantic
settings) is removed because pydantic settings reads the same env var at
startup that the SDK reads at call time — semantics are equivalent.

Only ``create_tenant_context`` / ``require_tenant_context`` remain local
because they return ``app.core.tenant_context.TenantContext`` (a service-
specific object with ``.metadata`` and friends), NOT the dict shape the SDK
returns. Callers and tests depend on the object shape, so we keep the
construction here while reusing the SDK's tenant-id extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from coding_client.auth.models import User
from coding_client.auth.tenant import (
    DEFAULT_TENANT_ID,
    LICENSE_INTROSPECT_URL,
    TENANT_ID_HEADER,
    TRUSTED_NETWORKS,
    TenantMiddleware,
    _introspect_license,
    _is_trusted_network,
    _license_cache,
    check_tenant_access,
    get_current_user_with_tenant,
    get_tenant_id,
    optional_tenant,
    require_same_tenant,
    require_tenant,
)
from fastapi import HTTPException, Request, status

if TYPE_CHECKING:
    from app.core.tenant_context import TenantContext


def create_tenant_context(request: Request) -> TenantContext:
    """Create a :class:`app.core.tenant_context.TenantContext` from the request.

    This returns the service-local ``TenantContext`` object (with ``.metadata``
    and ``user_id``/``is_admin`` kwargs), NOT the SDK's dict shape. Callers
    and tests rely on this object shape.

    Tenant-id extraction reuses the SDK helper :func:`get_tenant_id`.
    """
    # Runtime import to avoid circular dependency
    from app.core.tenant_context import TenantContext

    tenant_id = get_tenant_id(request)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant context required. Ensure X-Tenant-Id header is set.",
        )

    # Get user info if available
    user: User | None = getattr(request.state, "user", None)

    if user:
        return TenantContext(
            tenant_id=tenant_id,
            user_id=user.sub,
            is_admin=user.is_admin(),
            allow_legacy_access=True,
            metadata={"preferred_username": user.preferred_username},
        )
    return TenantContext(
        tenant_id=tenant_id,
        user_id=None,
        is_admin=False,
        allow_legacy_access=True,
    )


def require_tenant_context(request: Request) -> TenantContext:
    """FastAPI dependency that provides the service-local ``TenantContext``."""
    return create_tenant_context(request)


__all__ = [
    "DEFAULT_TENANT_ID",
    "LICENSE_INTROSPECT_URL",
    "TENANT_ID_HEADER",
    "TRUSTED_NETWORKS",
    "TenantMiddleware",
    "User",
    "_introspect_license",
    "_is_trusted_network",
    "_license_cache",
    "check_tenant_access",
    "create_tenant_context",
    "get_current_user_with_tenant",
    "get_tenant_id",
    "optional_tenant",
    "require_same_tenant",
    "require_tenant",
    "require_tenant_context",
]
