"""Tests for small uncovered modules.

Covers real behaviour of:
- app/core/tenant_context.py (TenantContext)
- auth/rbac.py (require_role)
- auth/models.py (User)
- processing/semantic_analysis/__init__.py (exports)
- app/api/endpoints/similarity.py (router composition)
- app/api/endpoints/repo_scanner.py (router)
- app/api/endpoints/analysis.py (router)
"""

from __future__ import annotations

import pytest
from coding_client.auth.models import User
from fastapi import HTTPException

from app.core.tenant_context import TenantContext

# ---------------------------------------------------------------------------
# TenantContext
# ---------------------------------------------------------------------------


class TestTenantContext:
    """Test TenantContext init, repr, and metadata."""

    def test_init_minimal(self):
        ctx = TenantContext("tenant-1")
        assert ctx.tenant_id == "tenant-1"
        assert ctx.metadata == {}

    def test_init_with_metadata(self):
        ctx = TenantContext("tenant-1", plan="pro", region="us")
        assert ctx.metadata == {"plan": "pro", "region": "us"}

    def test_repr_contains_tenant_id(self):
        ctx = TenantContext("abc")
        r = repr(ctx)
        assert "TenantContext" in r
        assert "abc" in r

    @pytest.mark.parametrize(
        "tenant_id",
        ["t1", "org-123", "user@domain"],
    )
    def test_various_tenant_ids(self, tenant_id: str):
        ctx = TenantContext(tenant_id)
        assert ctx.tenant_id == tenant_id


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------


class TestUserModel:
    """Test auth.models.User."""

    def test_minimal_creation(self):
        user = User(sub="user-1", preferred_username="alice")
        assert user.sub == "user-1"
        assert user.preferred_username == "alice"
        # tenant_id should default to sub
        assert user.tenant_id == "user-1"

    def test_tenant_id_from_organization(self):
        user = User(
            sub="user-1",
            preferred_username="alice",
            organization_id="org-99",
        )
        assert user.tenant_id == "org-99"

    def test_explicit_tenant_id_preserved(self):
        user = User(
            sub="user-1",
            preferred_username="alice",
            tenant_id="custom-tenant",
        )
        assert user.tenant_id == "custom-tenant"

    def test_effective_tenant_id(self):
        user = User(sub="u1", preferred_username="bob", tenant_id="t1")
        assert user.effective_tenant_id == "t1"

    def test_effective_tenant_id_fallback(self):
        user = User(sub="u1", preferred_username="bob")
        assert user.effective_tenant_id == "u1"

    def test_has_realm_role(self):
        user = User(
            sub="u1",
            preferred_username="alice",
            realm_access={"roles": ["admin", "user"]},
        )
        assert user.has_realm_role("admin") is True
        assert user.has_realm_role("superadmin") is False

    def test_has_client_role(self):
        user = User(
            sub="u1",
            preferred_username="alice",
            resource_access={"my-client": {"roles": ["editor"]}},
        )
        assert user.has_client_role("my-client", "editor") is True
        assert user.has_client_role("my-client", "admin") is False
        assert user.has_client_role("other", "editor") is False

    def test_get_all_roles(self):
        user = User(
            sub="u1",
            preferred_username="alice",
            realm_access={"roles": ["admin"]},
            resource_access={"client-a": {"roles": ["editor"]}, "client-b": {"roles": ["viewer"]}},
        )
        roles = user.get_all_roles()
        assert "admin" in roles
        assert "editor" in roles
        assert "viewer" in roles

    def test_is_admin(self):
        admin = User(
            sub="u1",
            preferred_username="admin",
            realm_access={"roles": ["admin"]},
        )
        assert admin.is_admin() is True

        platform_admin = User(
            sub="u2",
            preferred_username="padmin",
            realm_access={"roles": ["platform-admin"]},
        )
        assert platform_admin.is_admin() is True

        regular = User(sub="u3", preferred_username="user")
        assert regular.is_admin() is False


# ---------------------------------------------------------------------------
# auth/rbac.py - require_role
# ---------------------------------------------------------------------------


class TestRbacRequireRole:
    """Test require_role dependency factory."""

    def test_returns_callable(self):
        from auth.rbac import require_role

        checker = require_role("admin")
        assert callable(checker)

    @pytest.mark.asyncio
    async def test_allows_user_with_role(self):
        from auth.rbac import require_role

        checker = require_role("admin")
        user = User(
            sub="u1",
            preferred_username="alice",
            realm_access={"roles": ["admin"]},
        )
        result = await checker(user=user)
        assert result is user

    @pytest.mark.asyncio
    async def test_rejects_user_without_role(self):
        from auth.rbac import require_role

        checker = require_role("admin")
        user = User(
            sub="u1",
            preferred_username="bob",
            realm_access={"roles": ["user"]},
        )
        with pytest.raises(HTTPException) as exc_info:
            await checker(user=user)
        assert exc_info.value.status_code == 403
        assert "admin" in exc_info.value.detail

    def test_require_access_coding_exists(self):
        from auth.rbac import require_access_coding

        assert callable(require_access_coding)


# ---------------------------------------------------------------------------
# Semantic analysis package exports
# ---------------------------------------------------------------------------


class TestSemanticAnalysisPackage:
    """Test that processing.semantic_analysis exports are valid classes/functions.

    The __init__.py uses relative imports that require the full package path in
    sys.modules, so we parse the source AST to verify exports and use the live
    module when available.
    """

    def _get_init_source(self):
        """Read the raw source of semantic_analysis/__init__.py."""
        from pathlib import Path

        return (
            Path(__file__).resolve().parents[2]
            / "src"
            / "processing"
            / "semantic_analysis"
            / "__init__.py"
        ).read_text()

    def _get_module(self):
        """Try to get the live module, return None if stubbed."""
        import sys
        from unittest.mock import MagicMock

        mod = sys.modules.get("processing.semantic_analysis")
        if mod is None or isinstance(mod, MagicMock):
            return None
        if not hasattr(mod, "__all__"):
            return None
        return mod

    def test_init_declares_all(self):
        """The __init__.py source should declare __all__."""
        source = self._get_init_source()
        assert "__all__" in source

    def test_key_exports_declared(self):
        """Key classes and factory functions should be declared in __all__."""
        import ast

        source = self._get_init_source()
        tree = ast.parse(source)

        # Find the __all__ assignment and extract its values
        all_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant):
                                    all_names.add(elt.value)

        expected = {
            "SemanticAnalyzer",
            "ResultAggregator",
            "SymbolTableBuilder",
            "TypeSystemAnalyzer",
            "SemanticRuleValidator",
            "create_semantic_analyzer",
        }
        assert expected.issubset(all_names), f"Missing: {expected - all_names}"

    def test_init_has_import_statements(self):
        """The __init__.py should import from submodules."""
        import ast

        source = self._get_init_source()
        tree = ast.parse(source)
        import_froms = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        # Should have several relative imports
        assert len(import_froms) >= 5, f"Expected at least 5 from-imports, got {len(import_froms)}"

    def test_live_module_if_available(self):
        """If the live module is available (not stubbed), verify exports are real classes."""
        mod = self._get_module()
        if mod is None:
            pytest.skip("semantic_analysis module is stubbed by another test")
        for name in ["SemanticAnalyzer", "ResultAggregator", "SymbolTableBuilder"]:
            cls = getattr(mod, name)
            assert isinstance(cls, type), f"{name} should be a class"

    def test_all_count(self):
        """__all__ should export at least 20 names (classes, enums, factories)."""
        import ast

        source = self._get_init_source()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            assert len(node.value.elts) >= 20, (
                                "Expected at least 20 exports in __all__"
                            )


# ---------------------------------------------------------------------------
# Endpoint routers (verify structure via actual imports)
# Uses direct attribute checks that work even when stubs are present
# ---------------------------------------------------------------------------


class TestSimilarityEndpoint:
    """Test similarity router structure."""

    def test_router_has_routes(self):
        from app.api.endpoints.similarity import router

        assert len(router.routes) > 0

    def test_router_is_api_router(self):
        """Router should be a FastAPI APIRouter instance."""
        from fastapi import APIRouter

        from app.api.endpoints.similarity import router

        assert isinstance(router, APIRouter)

    def test_router_route_paths_are_strings(self):
        """All route paths should be non-empty strings."""
        from app.api.endpoints.similarity import router

        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert all(isinstance(p, str) and len(p) > 0 for p in paths)


class TestRepoScannerEndpoint:
    """Test repo_scanner router structure."""

    def test_router_has_routes(self):
        from app.api.endpoints.repo_scanner import router

        assert len(router.routes) > 0

    def test_module_version_in_source(self):
        """The repo_scanner module source should define __version__ = '2.1.0'."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "app"
            / "api"
            / "endpoints"
            / "repo_scanner.py"
        ).read_text()
        assert '__version__ = "2.1.0"' in source

    def test_router_is_api_router(self):
        """Router should be a FastAPI APIRouter instance."""
        from fastapi import APIRouter

        from app.api.endpoints.repo_scanner import router

        assert isinstance(router, APIRouter)


class TestAnalysisEndpoint:
    """Test analysis router structure."""

    def test_router_has_routes(self):
        from app.api.endpoints.analysis import router

        assert len(router.routes) > 0

    def test_router_is_api_router(self):
        """Router should be a FastAPI APIRouter instance."""
        from fastapi import APIRouter

        from app.api.endpoints.analysis import router

        assert isinstance(router, APIRouter)

    def test_analysis_source_defines_all(self):
        """Analysis module source should declare __all__ with 'router'."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "app"
            / "api"
            / "endpoints"
            / "analysis.py"
        ).read_text()
        assert '"router"' in source or "'router'" in source
        assert "__all__" in source

    def test_analysis_source_includes_subrouters(self):
        """Analysis module source should compose sub-routers."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "app"
            / "api"
            / "endpoints"
            / "analysis.py"
        ).read_text()
        # Verify that the source includes the expected sub-routers
        assert "analysis_router" in source
        assert "batch_router" in source
        assert "capabilities_router" in source
        assert "multi_language_router" in source
