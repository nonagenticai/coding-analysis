"""Role-based access control helpers."""

from coding_client.auth.dependencies import get_current_user
from coding_client.auth.models import User
from fastapi import Depends, HTTPException


def require_role(role: str):
    """Require user has specific role."""

    async def checker(user: User = Depends(get_current_user)):
        if not user.has_realm_role(role):
            raise HTTPException(status_code=403, detail=f"Role '{role}' required")
        return user

    return checker


# Backwards compatibility - was require_access_coding in rbac_coding.py
require_access_coding = require_role("coding-access")
