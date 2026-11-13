"""Backwards compatibility - re-exports from rbac module."""

from .rbac import require_access_coding, require_role

__all__ = ["require_access_coding", "require_role"]
