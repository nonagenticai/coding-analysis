"""Authentication and authorization module.

User and get_current_user are re-exported from coding_client.auth.* (the
local models.py + dependencies.py were byte-identical duplicates and have
been removed).
"""

from coding_client.auth.dependencies import get_current_user
from coding_client.auth.models import User

from .middleware import AuthMiddleware
from .rbac import require_access_coding, require_role
from .tenant import (
    TenantMiddleware,
    check_tenant_access,
    get_current_user_with_tenant,
    require_tenant,
)

__all__ = [
    "User",
    "TenantMiddleware",
    "require_tenant",
    "get_current_user_with_tenant",
    "check_tenant_access",
    "get_current_user",
    "AuthMiddleware",
    "require_role",
    "require_access_coding",
]
