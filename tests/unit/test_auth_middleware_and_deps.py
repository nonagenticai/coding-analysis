"""Tests for auth/middleware.py and auth/dependencies.py.

Covers auth/middleware.py lines 31% uncovered, auth/dependencies.py lines 25% uncovered.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coding_client.auth.dependencies import get_current_user, require_any_role, require_role
from coding_client.auth.models import User
from fastapi import HTTPException

from auth.middleware import (
    PUBLIC_PATHS,
    TRUSTED_NETWORKS,
    AuthMiddleware,
    _is_trusted_network,
)

# ---------------------------------------------------------------------------
# _is_trusted_network
# ---------------------------------------------------------------------------


class TestIsTrustedNetwork:
    """No host is trusted on network position alone by default.

    These read the module's real default (TRUSTED_CIDRS is unset in the test
    env, as it is in the production configmap) and never monkeypatch
    TRUSTED_NETWORKS, so they assert the shipped behaviour rather than a value
    the test itself installed.
    """

    def test_default_config_trusts_no_network(self):
        assert TRUSTED_NETWORKS == []

    @pytest.mark.parametrize(
        "host",
        [
            "10.42.1.5",  # cluster pod CIDR — the address family that leaked
            "10.0.0.1",
            "192.168.1.100",
            "172.16.0.1",
            "127.0.0.1",
            "::1",
            "8.8.8.8",
        ],
    )
    def test_no_host_is_trusted_by_default(self, host):
        assert _is_trusted_network(host) is False

    def test_none_untrusted(self):
        assert _is_trusted_network(None) is False

    def test_invalid_ip_untrusted(self):
        assert _is_trusted_network("not-an-ip") is False

    def test_empty_string_untrusted(self):
        assert _is_trusted_network("") is False

    def test_explicit_config_is_honoured_and_stays_narrow(self):
        """An operator-set TRUSTED_CIDRS still works, and only for what it names.

        Reaches into ``coding_client.auth._trusted_networks`` rather than
        reloading this module. Since the fork was replaced by a re-export of the
        SDK, ``TRUSTED_CIDRS`` is parsed at import of the SDK's module, so
        reloading the shim would re-bind the same already-computed list and the
        test would pass vacuously — it would assert nothing about the env at all.
        ``is_trusted_network`` reads ``TRUSTED_NETWORKS`` from its own module at
        CALL time precisely so this substitution works; this is the same seam the
        SDK's own suite uses.
        """
        from coding_client.auth import _trusted_networks as tn_mod

        original = tn_mod.TRUSTED_NETWORKS
        with patch.dict(os.environ, {"TRUSTED_CIDRS": "127.0.0.1/32"}):
            tn_mod.TRUSTED_NETWORKS = tn_mod.load_trusted_networks()
            try:
                # Non-vacuity: the reconfiguration really took effect.
                assert tn_mod.TRUSTED_NETWORKS != original
                assert _is_trusted_network("127.0.0.1") is True
                assert _is_trusted_network("10.42.1.5") is False
                assert _is_trusted_network("192.168.1.100") is False
            finally:
                tn_mod.TRUSTED_NETWORKS = original
        # And it is genuinely restored, so no later test inherits a trusted CIDR.
        assert _is_trusted_network("127.0.0.1") is False


# ---------------------------------------------------------------------------
# PUBLIC_PATHS
# ---------------------------------------------------------------------------


class TestPublicPaths:
    def test_health_is_public(self):
        assert "/health" in PUBLIC_PATHS

    def test_docs_is_public(self):
        assert "/docs" in PUBLIC_PATHS

    def test_root_is_public(self):
        assert "/" in PUBLIC_PATHS

    def test_api_is_not_public(self):
        assert "/api/analysis" not in PUBLIC_PATHS


# ---------------------------------------------------------------------------
# AuthMiddleware.dispatch
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    @pytest.mark.asyncio
    async def test_public_path_passes_through(self):
        middleware = AuthMiddleware(app=MagicMock())
        request = MagicMock()
        request.url.path = "/health"
        call_next = AsyncMock(return_value="response")

        await middleware.dispatch(request, call_next)
        call_next.assert_awaited_once_with(request)

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"ENVIRONMENT": "development", "DEV_API_KEY": "test-key"})
    async def test_dev_api_key_auth(self):
        # The dev key proves nothing about WHICH tenant; since 1.9.0 it only
        # authenticates when a DEFAULT_TENANT_ID is configured to grant. The SDK
        # reads that as a module attribute (captured at import), so patch it there.
        import coding_client.auth._default_tenant as dt

        middleware = AuthMiddleware(app=MagicMock())
        request = MagicMock()
        request.url.path = "/api/analysis"
        request.headers = {"authorization": "Bearer test-key"}
        call_next = AsyncMock(return_value="response")

        with patch.object(dt, "DEFAULT_TENANT_ID", "dev-tenant"):
            await middleware.dispatch(request, call_next)
        call_next.assert_awaited_once()
        assert request.state.user.preferred_username == "dev-user"

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"ENVIRONMENT": "development", "DEV_API_KEY": "test-key"})
    async def test_dev_api_key_without_default_tenant_is_rejected(self):
        """Control for the above: the same valid dev key with no DEFAULT_TENANT_ID
        configured does NOT authenticate — it must not mint a made-up tenant."""
        middleware = AuthMiddleware(app=MagicMock())
        request = self._anonymous_request("8.8.8.8")
        request.headers = MagicMock()
        request.headers.get = lambda k, d="": {"authorization": "Bearer test-key"}.get(k, d)
        call_next = AsyncMock()

        result = await middleware.dispatch(request, call_next)
        assert result.status_code == 401
        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_istio_headers_create_user(self):
        middleware = AuthMiddleware(app=MagicMock())
        request = MagicMock()
        request.url.path = "/api/analysis"
        request.headers = {
            "x-user-id": "user-123",
            "x-tenant-id": "tenant-456",
            "x-user-name": "testuser",
            "x-user-email": "test@example.com",
            "x-organization-id": "org-1",
            "x-user-roles": "admin,user",
            "authorization": "",
        }
        call_next = AsyncMock(return_value="response")

        await middleware.dispatch(request, call_next)
        user = request.state.user
        assert user.sub == "user-123"
        assert user.tenant_id == "tenant-456"
        assert user.preferred_username == "testuser"

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"TRUST_TENANT_HEADER": "false"})
    async def test_forged_headers_rejected_when_gate_off(self):
        """The security contract this PR ships: with TRUST_TENANT_HEADER=false a
        forged x-tenant-id/x-user-roles carries NO identity, so a credential-less
        caller is 401'd instead of admitted as admin of the named tenant.

        Reproduced live 2026-08-03: a coding-testing pod set these headers and
        read a fabricated tenant off coding-analysis (HTTP 200). This asserts the
        gate closes that.
        """
        middleware = AuthMiddleware(app=MagicMock())
        request = MagicMock()
        request.url.path = "/api/analysis"
        request.method = "GET"
        _headers = {
            "x-user-id": "attacker",
            "x-tenant-id": "victim-tenant",
            "x-user-roles": "admin,platform-admin",
            "authorization": "",
        }
        mock_headers = MagicMock()
        mock_headers.get = lambda key, default="": _headers.get(key, default)
        request.headers = mock_headers
        request.client = MagicMock()
        request.client.host = "10.42.1.5"  # a pod IP — the real attacker position
        call_next = AsyncMock()

        result = await middleware.dispatch(request, call_next)
        assert result.status_code == 401
        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_forged_headers_honoured_when_gate_default(self):
        """False-positive control: unset TRUST_TENANT_HEADER defaults True, so the
        same forged request is still admitted today (parity preserved until the
        gitops flip). Guards against the gate silently defaulting closed."""
        middleware = AuthMiddleware(app=MagicMock())
        request = MagicMock()
        request.url.path = "/api/analysis"
        request.method = "GET"
        _headers = {
            "x-user-id": "attacker",
            "x-tenant-id": "victim-tenant",
            "x-user-roles": "admin",
            "authorization": "",
        }
        mock_headers = MagicMock()
        mock_headers.get = lambda key, default="": _headers.get(key, default)
        request.headers = mock_headers
        request.client = MagicMock()
        request.client.host = "10.42.1.5"
        call_next = AsyncMock(return_value="response")

        await middleware.dispatch(request, call_next)
        assert request.state.user.tenant_id == "victim-tenant"

    @staticmethod
    def _anonymous_request(client_host: str):
        request = MagicMock()
        request.url.path = "/api/analysis"
        request.method = "GET"
        _headers = {
            "x-user-id": None,
            "x-tenant-id": None,
            "x-user-name": "anonymous",
            "x-user-email": None,
            "x-organization-id": None,
            "x-user-roles": "",
            "authorization": "",
        }
        mock_headers = MagicMock()
        mock_headers.get = lambda key, default="": _headers.get(key, default)
        request.headers = mock_headers
        request.client = MagicMock()
        request.client.host = client_host
        return request

    @pytest.mark.asyncio
    @pytest.mark.parametrize("client_host", ["10.42.1.5", "192.168.1.100", "172.16.0.1"])
    async def test_private_source_ip_alone_is_rejected(self, client_host):
        """A credential-free request is 401 no matter how private its source IP is.

        10.42.1.5 is the cluster pod CIDR: under the previous default this
        returned 200 with a full DEFAULT_TENANT_ID context to any pod in any
        namespace.
        """
        middleware = AuthMiddleware(app=MagicMock())
        request = self._anonymous_request(client_host)
        call_next = AsyncMock()

        result = await middleware.dispatch(request, call_next)
        assert result.status_code == 401
        call_next.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_trusted_network_fallback_when_explicitly_configured(self):
        """The fallback still works — but only for a CIDR an operator named.

        Patches the SDK's own module attributes rather than reloading this one.
        Now that the fork is a re-export, ``importlib.reload(auth.middleware)``
        re-binds names to the SDK's ALREADY-computed ``TRUSTED_NETWORKS`` and
        changes nothing — the dispatch consults
        ``coding_client.auth._trusted_networks.TRUSTED_NETWORKS`` at call time.
        Reloading here would have made this test assert the 401 path forever
        while claiming to prove the grant path.
        """
        import coding_client.auth._default_tenant as dt
        from coding_client.auth import _trusted_networks as tn_mod

        original = tn_mod.TRUSTED_NETWORKS
        with (
            patch.dict(os.environ, {"TRUSTED_CIDRS": "192.168.0.0/16"}),
            patch.object(dt, "DEFAULT_TENANT_ID", "tenant-configured"),
        ):
            tn_mod.TRUSTED_NETWORKS = tn_mod.load_trusted_networks()
            try:
                assert tn_mod.TRUSTED_NETWORKS != original  # non-vacuity
                middleware = AuthMiddleware(app=MagicMock())
                request = self._anonymous_request("192.168.1.100")
                call_next = AsyncMock(return_value="response")

                await middleware.dispatch(request, call_next)
                # Both a named CIDR AND a configured tenant are required to mint.
                assert request.state.user.preferred_username == "internal"
                assert request.state.user.tenant_id == "tenant-configured"
            finally:
                tn_mod.TRUSTED_NETWORKS = original

    @pytest.mark.asyncio
    async def test_untrusted_network_returns_401(self):
        middleware = AuthMiddleware(app=MagicMock())
        request = MagicMock()
        request.url.path = "/api/analysis"
        _headers = {
            "x-user-id": None,
            "x-tenant-id": None,
            "x-user-roles": "",
            "authorization": "",
        }
        mock_headers = MagicMock()
        mock_headers.get = lambda key, default="": _headers.get(key, default)
        request.headers = mock_headers
        request.client = MagicMock()
        request.client.host = "8.8.8.8"
        request.method = "GET"
        call_next = AsyncMock()

        result = await middleware.dispatch(request, call_next)
        assert result.status_code == 401


# ---------------------------------------------------------------------------
# get_current_user
# ---------------------------------------------------------------------------


class TestGetCurrentUser:
    def test_returns_user_from_state(self):
        user = User(sub="u1", preferred_username="alice")
        request = MagicMock()
        request.state.user = user
        result = get_current_user(request)
        assert result.sub == "u1"

    def test_no_user_raises_401(self):
        request = MagicMock()
        # Simulate request.state having no 'user' attribute
        del request.state.user
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(request)
        assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# require_role
# ---------------------------------------------------------------------------


class TestRequireRole:
    def test_has_role_succeeds(self):
        user = User(sub="u1", preferred_username="alice", realm_access={"roles": ["admin"]})
        request = MagicMock()
        request.state.user = user
        checker = require_role("admin")
        result = checker(request)
        assert result.sub == "u1"

    def test_missing_role_raises_403(self):
        user = User(sub="u1", preferred_username="alice", realm_access={"roles": ["user"]})
        request = MagicMock()
        request.state.user = user
        checker = require_role("admin")
        with pytest.raises(HTTPException) as exc_info:
            checker(request)
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# require_any_role
# ---------------------------------------------------------------------------


class TestRequireAnyRole:
    def test_has_one_of_roles_succeeds(self):
        user = User(sub="u1", preferred_username="alice", realm_access={"roles": ["editor"]})
        request = MagicMock()
        request.state.user = user
        checker = require_any_role("editor", "admin")
        result = checker(request)
        assert result.sub == "u1"

    def test_missing_all_roles_raises_403(self):
        user = User(sub="u1", preferred_username="alice", realm_access={"roles": ["viewer"]})
        request = MagicMock()
        request.state.user = user
        checker = require_any_role("editor", "admin")
        with pytest.raises(HTTPException) as exc_info:
            checker(request)
        assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------


class TestUserModel:
    def test_effective_tenant_id_from_tenant(self):
        user = User(sub="u1", preferred_username="alice", tenant_id="t1")
        assert user.effective_tenant_id == "t1"

    def test_effective_tenant_id_fallback_to_sub(self):
        user = User(sub="u1", preferred_username="alice")
        # model_post_init sets tenant_id = sub
        assert user.effective_tenant_id == "u1"

    def test_is_admin_with_admin_role(self):
        user = User(sub="u1", preferred_username="alice", realm_access={"roles": ["admin"]})
        assert user.is_admin() is True

    def test_is_admin_with_platform_admin(self):
        user = User(
            sub="u1", preferred_username="alice", realm_access={"roles": ["platform-admin"]}
        )
        assert user.is_admin() is True

    def test_is_not_admin(self):
        user = User(sub="u1", preferred_username="alice", realm_access={"roles": ["user"]})
        assert user.is_admin() is False

    def test_has_client_role(self):
        user = User(
            sub="u1",
            preferred_username="alice",
            resource_access={"my-app": {"roles": ["read"]}},
        )
        assert user.has_client_role("my-app", "read") is True
        assert user.has_client_role("my-app", "write") is False
        assert user.has_client_role("other-app", "read") is False

    def test_get_all_roles(self):
        user = User(
            sub="u1",
            preferred_username="alice",
            realm_access={"roles": ["admin"]},
            resource_access={"app1": {"roles": ["read", "write"]}},
        )
        roles = user.get_all_roles()
        assert "admin" in roles
        assert "read" in roles
        assert "write" in roles

    def test_model_post_init_uses_org_id(self):
        user = User(sub="u1", preferred_username="alice", organization_id="org-1")
        assert user.tenant_id == "org-1"


# ---------------------------------------------------------------------------
# Tests merged from test_tenant_coverage.py
# ---------------------------------------------------------------------------

from auth.tenant import (  # noqa: E402
    TenantMiddleware,
    _introspect_license,
    check_tenant_access,
    get_current_user_with_tenant,
    get_tenant_id,
    require_same_tenant,
    require_tenant,
)
from auth.tenant import (
    _is_trusted_network as _tenant_is_trusted_network,
)


class TestTenantIsTrustedNetwork:
    """Tests for auth.tenant._is_trusted_network (the coding_client SDK shim).

    Since coding-client 1.7.0 (#33) TRUSTED_CIDRS is **empty by default** — no
    source IP is trusted on network position alone, because 10.0.0.0/8 contains
    the pod CIDR 10.42.0.0/16. coding-analysis additionally pins TRUSTED_CIDRS=""
    in its production configmap. So these formerly-True cases (loopback + the
    RFC1918 ranges) are now False by default; the network bypass is only live for
    a CIDR an operator explicitly names — see
    test_explicitly_configured_cidr_is_trusted for that positive control.
    """

    def test_localhost_ipv4(self):
        assert _tenant_is_trusted_network("127.0.0.1") is False

    def test_localhost_ipv6(self):
        assert _tenant_is_trusted_network("::1") is False

    def test_private_192(self):
        assert _tenant_is_trusted_network("192.168.1.100") is False

    def test_private_10(self):
        # 10.0.0.0/8 ⊇ the pod CIDR — the address family that leaked the graphs.
        assert _tenant_is_trusted_network("10.0.0.1") is False

    def test_private_172(self):
        assert _tenant_is_trusted_network("172.16.0.1") is False

    def test_explicitly_configured_cidr_is_trusted(self):
        """False-positive control: the mechanism still works when configured.

        Proves the by-default False assertions above are the empty default, not
        a dead function that can never return True.
        """
        import ipaddress

        import coding_client.auth._trusted_networks as tn

        with patch.object(tn, "TRUSTED_NETWORKS", [ipaddress.ip_network("127.0.0.1/32")]):
            assert _tenant_is_trusted_network("127.0.0.1") is True
            assert _tenant_is_trusted_network("10.0.0.1") is False

    def test_public_ip(self):
        assert _tenant_is_trusted_network("8.8.8.8") is False

    def test_none(self):
        assert _tenant_is_trusted_network(None) is False

    def test_empty_string(self):
        assert _tenant_is_trusted_network("") is False

    def test_invalid_ip(self):
        assert _tenant_is_trusted_network("not-an-ip") is False


class TestCheckTenantAccess:
    def test_admin_can_access_any(self):
        user = MagicMock(spec=User)
        user.is_admin.return_value = True
        assert check_tenant_access(user, "any-tenant") is True

    def test_user_own_tenant(self):
        user = MagicMock(spec=User)
        user.is_admin.return_value = False
        user.effective_tenant_id = "tenant-a"
        assert check_tenant_access(user, "tenant-a") is True

    def test_user_different_tenant(self):
        user = MagicMock(spec=User)
        user.is_admin.return_value = False
        user.effective_tenant_id = "tenant-a"
        assert check_tenant_access(user, "tenant-b") is False


class TestRequireSameTenant:
    def test_returns_callable(self):
        validator = require_same_tenant("tenant-a")
        assert callable(validator)


class TestGetTenantId:
    def test_with_tenant_id(self):
        request = MagicMock()
        request.state.tenant_id = "my-tenant"
        assert get_tenant_id(request) == "my-tenant"

    def test_without_tenant_id(self):
        request = MagicMock(spec=[])
        request.state = MagicMock(spec=[])
        assert get_tenant_id(request) is None


class TestGetCurrentUserWithTenant:
    def test_with_existing_user(self):
        user = MagicMock(spec=User)
        user.tenant_id = "tenant-a"
        request = MagicMock()
        request.state.user = user
        result = get_current_user_with_tenant(request)
        assert result is user

    def test_skip_auth_no_longer_bypasses_anything(self):
        """SKIP_AUTH was deleted from coding-client in 1.16.0.

        Inverted rather than deleted: the old assertion — SKIP_AUTH=true yields
        a User whose tenant is "test-tenant" — WAS the hatch. Without a test in
        its place, nothing here would notice it coming back through a pin
        downgrade.
        """
        request = MagicMock()
        request.state = MagicMock(spec=[])
        request.headers = {}
        request.client.host = "10.42.0.1"
        with patch.dict(os.environ, {"SKIP_AUTH": "true"}), pytest.raises(HTTPException) as exc:
            get_current_user_with_tenant(request)
        assert exc.value.status_code == 401

    def test_trusted_network_fallback_closed_by_default(self):
        """With TRUSTED_CIDRS empty (the 1.7.0+ default AND coding-analysis's
        production configmap), a header-less caller from 127.0.0.1 no longer
        falls into the priority-4 bypass — it reaches a 401 like any other
        credential-less request. Previously this returned DEFAULT_TENANT_ID.
        """
        request = MagicMock()
        request.state = MagicMock(spec=[])
        request.headers = {}
        request.client.host = "127.0.0.1"
        with patch.dict(os.environ, {"SKIP_AUTH": "false"}):
            with pytest.raises(HTTPException) as exc:
                get_current_user_with_tenant(request)
            assert exc.value.status_code == 401

    def test_trusted_network_fallback_when_configured(self):
        """False-positive control: with an operator-named CIDR AND a configured
        DEFAULT_TENANT_ID, the bypass still grants that tenant — proving the
        401 above is the empty default, not a broken path.
        """
        import ipaddress

        import coding_client.auth._default_tenant as dt
        import coding_client.auth._trusted_networks as tn

        request = MagicMock()
        request.state = MagicMock(spec=[])
        request.headers = {}
        request.client.host = "127.0.0.1"
        with (
            patch.dict(os.environ, {"SKIP_AUTH": "false"}),
            patch.object(dt, "DEFAULT_TENANT_ID", "tenant-configured"),
            patch.object(tn, "TRUSTED_NETWORKS", [ipaddress.ip_network("127.0.0.1/32")]),
        ):
            result = get_current_user_with_tenant(request)
            assert result.tenant_id == "tenant-configured"

    def test_untrusted_no_user_raises(self):
        request = MagicMock()
        request.state = MagicMock(spec=[])
        # Empty headers + untrusted client → SDK falls through every priority to 401.
        request.headers = {}
        request.client.host = "8.8.8.8"
        with patch.dict(os.environ, {"SKIP_AUTH": "false"}):
            with pytest.raises(HTTPException) as exc:
                get_current_user_with_tenant(request)
            assert exc.value.status_code == 401

    def test_user_without_tenant_falls_back_to_sub(self):
        user = User(sub="user-sub", preferred_username="test")
        user.tenant_id = None
        request = MagicMock()
        request.state.user = user
        result = get_current_user_with_tenant(request)
        assert result.tenant_id == "user-sub"


class TestRequireTenantCoverage:
    async def test_skip_auth_no_longer_bypasses_anything(self):
        """Companion to TestGetCurrentUserWithTenant's inverted case, for the
        other dependency the hatch used to short-circuit."""
        request = MagicMock()
        request.state = MagicMock(spec=[])
        request.headers = {}
        request.client.host = "10.42.0.1"
        with patch.dict(os.environ, {"SKIP_AUTH": "true"}), pytest.raises(HTTPException) as exc:
            await require_tenant(request)
        assert exc.value.status_code == 401

    async def test_from_request_state(self):
        request = MagicMock()
        request.state.tenant_id = "my-tenant"
        request.headers = {}
        with patch.dict(os.environ, {"SKIP_AUTH": "false"}):
            result = await require_tenant(request)
            assert result == "my-tenant"

    async def test_from_bearer_token(self):
        request = MagicMock()
        request.state = MagicMock(spec=[])
        request.headers = {"Authorization": "Bearer valid-token"}
        request.client.host = "8.8.8.8"
        with (
            patch.dict(os.environ, {"SKIP_AUTH": "false"}),
            patch(
                "coding_client.auth.tenant._introspect_license", new_callable=AsyncMock
            ) as mock_introspect,
        ):
            # coding-client 1.5.7: resolve the tenant from `tenant_id` (workspace),
            # not `subject_id` (the user). Provide both; assert tenant, not user.
            mock_introspect.return_value = {
                "active": True,
                "tenant_id": "license-tenant",
                "subject_id": "the-user",
            }
            result = await require_tenant(request)
            assert result == "license-tenant"
            assert result != "the-user"

    async def test_from_user_state_fallback(self):
        user = MagicMock(spec=User)
        user.effective_tenant_id = "user-tenant"
        request = MagicMock()
        request.state = MagicMock()
        request.state.tenant_id = None
        request.state.user = user
        request.headers = {}
        request.client.host = "8.8.8.8"
        with patch.dict(os.environ, {"SKIP_AUTH": "false"}):
            with patch("coding_client.auth.tenant.get_tenant_id", return_value=None):
                result = await require_tenant(request)
                assert result == "user-tenant"

    async def test_trusted_network_fallback_closed_by_default(self):
        """Empty TRUSTED_CIDRS (1.7.0+ default + coding-analysis's configmap):
        127.0.0.1 with no headers now 401s instead of resolving DEFAULT_TENANT_ID."""
        request = MagicMock()
        request.state = MagicMock(spec=[])
        request.headers = {}
        request.client.host = "127.0.0.1"
        with patch.dict(os.environ, {"SKIP_AUTH": "false"}):
            with patch("coding_client.auth.tenant.get_tenant_id", return_value=None):
                with pytest.raises(HTTPException) as exc:
                    await require_tenant(request)
                assert exc.value.status_code == 401

    async def test_trusted_network_fallback_when_configured(self):
        """False-positive control: configured CIDR + DEFAULT_TENANT_ID still grants."""
        import ipaddress

        import coding_client.auth._default_tenant as dt
        import coding_client.auth._trusted_networks as tn

        request = MagicMock()
        request.state = MagicMock(spec=[])
        request.headers = {}
        request.client.host = "127.0.0.1"
        with (
            patch.dict(os.environ, {"SKIP_AUTH": "false"}),
            patch.object(dt, "DEFAULT_TENANT_ID", "tenant-configured"),
            patch("coding_client.auth.tenant.get_tenant_id", return_value=None),
            patch.object(tn, "TRUSTED_NETWORKS", [ipaddress.ip_network("127.0.0.1/32")]),
        ):
            result = await require_tenant(request)
            assert result == "tenant-configured"

    async def test_no_tenant_raises_401(self):
        request = MagicMock()
        request.state = MagicMock(spec=[])
        request.headers = {}
        request.client.host = "8.8.8.8"
        with patch.dict(os.environ, {"SKIP_AUTH": "false"}):
            with patch("coding_client.auth.tenant.get_tenant_id", return_value=None):
                with pytest.raises(HTTPException) as exc:
                    await require_tenant(request)
                assert exc.value.status_code == 401


class TestIntrospectLicense:
    async def test_cached_result(self):
        import hashlib

        from auth.tenant import _license_cache

        token = "cached-token"
        cache_key = f"lic:{hashlib.sha256(token.encode()).hexdigest()[:16]}"
        _license_cache[cache_key] = {"active": True, "subject_id": "cached-tenant"}
        try:
            result = await _introspect_license(token)
            assert result["subject_id"] == "cached-tenant"
        finally:
            _license_cache.pop(cache_key, None)

    async def test_network_error(self):
        with (
            patch.dict(
                os.environ,
                {"LICENSE_INTROSPECT_URL": "http://stub-introspect:8544/license/introspect"},
            ),
            patch("coding_client.auth.tenant.httpx.AsyncClient") as mock_client,
        ):
            mock_instance = AsyncMock()
            mock_instance.__aenter__ = AsyncMock(side_effect=Exception("connection error"))
            mock_client.return_value = mock_instance
            result = await _introspect_license("invalid-token")
            assert result is None


class TestTenantMiddleware:
    async def test_with_user_and_header(self):
        middleware = TenantMiddleware(app=MagicMock())
        request = MagicMock()
        headers_mock = MagicMock()
        headers_mock.get = MagicMock(return_value="header-tenant")
        request.headers = headers_mock
        user = MagicMock()
        user.effective_tenant_id = "header-tenant"
        user.preferred_username = "testuser"
        request.state.user = user
        call_next = AsyncMock(return_value=MagicMock())
        await middleware.dispatch(request, call_next)
        assert user.tenant_id == "header-tenant"

    async def test_with_user_no_header(self):
        middleware = TenantMiddleware(app=MagicMock())
        request = MagicMock()
        headers_mock = MagicMock()
        headers_mock.get = MagicMock(return_value=None)
        request.headers = headers_mock
        user = MagicMock(spec=User)
        user.tenant_id = None
        user.sub = "user-sub"
        user.effective_tenant_id = "user-sub"
        request.state.user = user
        call_next = AsyncMock(return_value=MagicMock())
        await middleware.dispatch(request, call_next)
        assert user.tenant_id == "user-sub"

    async def test_no_user_with_header(self):
        middleware = TenantMiddleware(app=MagicMock())
        request = MagicMock()
        headers_mock = MagicMock()
        headers_mock.get = MagicMock(return_value="header-tenant")
        request.headers = headers_mock
        request.state = MagicMock(spec=[])
        call_next = AsyncMock(return_value=MagicMock())
        await middleware.dispatch(request, call_next)
        assert request.state.tenant_id == "header-tenant"


# ---------------------------------------------------------------------------
# Uncovered lines in auth/tenant.py: 83-88, 292-301, 354-376, 403
# ---------------------------------------------------------------------------

from auth.tenant import (
    create_tenant_context,
    require_tenant_context,
)


class TestIntrospectLicenseSuccess:
    """Test successful license introspection with HTTP (lines 83-88)."""

    async def test_successful_introspection_caches_result(self):
        """Successful introspection returns and caches data (lines 84-88)."""
        from auth.tenant import _license_cache

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": True, "subject_id": "new-tenant"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.dict(
                os.environ,
                {"LICENSE_INTROSPECT_URL": "http://stub-introspect:8544/license/introspect"},
            ),
            patch("coding_client.auth.tenant.httpx.AsyncClient", return_value=mock_client),
        ):
            result = await _introspect_license("new-valid-token-xyz")
            assert result["active"] is True
            assert result["subject_id"] == "new-tenant"

        # Clean up cache
        import hashlib

        cache_key = f"lic:{hashlib.sha256(b'new-valid-token-xyz').hexdigest()[:16]}"
        _license_cache.pop(cache_key, None)

    async def test_introspection_inactive_token(self):
        """Inactive token returns data but does not cache (lines 86-88)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"active": False, "subject_id": "expired-tenant"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.dict(
                os.environ,
                {"LICENSE_INTROSPECT_URL": "http://stub-introspect:8544/license/introspect"},
            ),
            patch("coding_client.auth.tenant.httpx.AsyncClient", return_value=mock_client),
        ):
            result = await _introspect_license("inactive-token-abc")
            assert result["active"] is False


class TestRequireSameTenantValidator:
    """Test the validator function returned by require_same_tenant (lines 292-301)."""

    def test_admin_bypasses_tenant_check(self):
        """Admin users can access any tenant's resources (lines 292-293)."""
        user = User(
            sub="admin-user",
            preferred_username="admin",
            realm_access={"roles": ["admin"]},
            tenant_id="admin-tenant",
        )
        request = MagicMock()
        request.state.user = user
        validator_fn = require_same_tenant("other-tenant")
        # Call the validator with user directly (simulating dependency injection)
        result = validator_fn(user)
        assert result is user

    def test_same_tenant_passes(self):
        """User with matching tenant passes (line 296-301)."""
        user = User(
            sub="regular-user",
            preferred_username="bob",
            tenant_id="tenant-a",
        )
        request = MagicMock()
        request.state.user = user
        validator_fn = require_same_tenant("tenant-a")
        result = validator_fn(user)
        assert result is user

    def test_different_tenant_raises_403(self):
        """User with different tenant raises 403 (lines 297-300)."""
        user = User(
            sub="regular-user",
            preferred_username="bob",
            tenant_id="tenant-a",
        )
        validator_fn = require_same_tenant("tenant-b")
        with pytest.raises(HTTPException) as exc:
            validator_fn(user)
        assert exc.value.status_code == 403
        assert "different tenant" in exc.value.detail


class TestCreateTenantContext:
    """Test create_tenant_context (lines 354-376)."""

    def test_with_tenant_and_user(self):
        """Creates TenantContext with user info (lines 366-373)."""
        user = User(
            sub="user-1",
            preferred_username="alice",
            tenant_id="tenant-1",
            realm_access={"roles": ["admin"]},
        )
        request = MagicMock()
        request.state.tenant_id = "tenant-1"
        request.state.user = user
        ctx = create_tenant_context(request)
        assert ctx.tenant_id == "tenant-1"
        # TenantContext stores extra kwargs in metadata
        assert ctx.metadata.get("user_id") == "user-1" or hasattr(ctx, "user_id")

    def test_with_tenant_no_user(self):
        """Creates TenantContext without user (lines 374-381)."""
        request = MagicMock()
        request.state.tenant_id = "tenant-2"
        request.state.user = None
        ctx = create_tenant_context(request)
        assert ctx.tenant_id == "tenant-2"

    def test_no_tenant_raises_401(self):
        """Missing tenant raises 401 (lines 357-361)."""
        request = MagicMock(spec=[])
        request.state = MagicMock(spec=[])
        with pytest.raises(HTTPException) as exc:
            create_tenant_context(request)
        assert exc.value.status_code == 401


class TestRequireTenantContext:
    """Test require_tenant_context (line 403)."""

    def test_delegates_to_create_tenant_context(self):
        """require_tenant_context calls create_tenant_context (line 403)."""
        user = User(
            sub="user-1",
            preferred_username="alice",
            tenant_id="tenant-1",
        )
        request = MagicMock()
        request.state.tenant_id = "tenant-1"
        request.state.user = user
        ctx = require_tenant_context(request)
        assert ctx.tenant_id == "tenant-1"


# ---------------------------------------------------------------------------
# Licence-Bearer authentication — the capability the de-fork exists to gain
# ---------------------------------------------------------------------------


class TestLicenceBearerAuthentication:
    """A valid licence Bearer must authenticate.

    This is the whole point of replacing the local ``AuthMiddleware`` fork with
    ``coding_client.auth.middleware``. The fork had NO introspection: its only
    ``Authorization`` read was a static dev/test API-key comparison gated on
    ``ENVIRONMENT in ("development","testing")``, dead in production. So the
    fork's only working production credential was a bare, forgeable
    ``X-Tenant-Id`` — which is why ``TRUST_TENANT_HEADER=false`` could not be
    set on this service without 401ing every request.

    ⚠️ These are the mutation guard for the de-fork. Restore the fork (or pin
    coding-client back below 1.10.0) and they fail BY NAME rather than leaving
    a green suite over a service that silently lost its Bearer path.

    Patches ``coding_client.auth.tenant.httpx.AsyncClient`` rather than
    ``_introspect_license`` so the real introspection function, its parsing and
    its caches are exercised — a stub of the function under test would prove
    nothing about it.
    """

    @staticmethod
    def _bearer_request(token: str):
        request = MagicMock()
        request.url.path = "/api/analysis"
        request.method = "GET"
        _headers = {
            "authorization": f"Bearer {token}",
            "x-user-id": None,
            "x-tenant-id": None,
            "x-user-name": "anonymous",
            "x-user-email": None,
            "x-organization-id": None,
            "x-user-roles": "",
        }
        request.headers.get = lambda k, d=None: _headers.get(k.lower(), d)
        request.client.host = "10.42.9.9"  # untrusted; only the Bearer can grant
        return request

    @staticmethod
    def _introspection(payload: dict, status_code: int = 200):
        response = MagicMock()
        response.status_code = status_code
        response.json = MagicMock(return_value=payload)
        client = AsyncMock()
        client.post = AsyncMock(return_value=response)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return patch("coding_client.auth.tenant.httpx.AsyncClient", return_value=ctx), client

    @pytest.mark.asyncio
    async def test_valid_licence_bearer_authenticates(self):
        from coding_client.auth.tenant import clear_license_caches

        clear_license_caches()
        introspect, http = self._introspection(
            {"active": True, "tenant_id": "tenant-from-licence", "subject_id": "subject-1"}
        )
        with (
            patch.dict(
                os.environ, {"LICENSE_INTROSPECT_URL": "http://licences.invalid/introspect"}
            ),
            introspect,
        ):
            middleware = AuthMiddleware(app=MagicMock())
            request = self._bearer_request("a-real-licence")
            call_next = AsyncMock(return_value="response")

            await middleware.dispatch(request, call_next)

        call_next.assert_awaited()  # not a 401
        assert request.state.user.tenant_id == "tenant-from-licence"
        # The tenant came off the wire, not off a header or a default.
        http.post.assert_awaited_once()
        clear_license_caches()

    @pytest.mark.asyncio
    async def test_inactive_licence_is_rejected(self):
        """Non-vacuity: the Bearer arm can say no, so the test above means something."""
        from coding_client.auth.tenant import clear_license_caches

        clear_license_caches()
        introspect, _ = self._introspection({"active": False, "tenant_id": "tenant-from-licence"})
        with (
            patch.dict(
                os.environ, {"LICENSE_INTROSPECT_URL": "http://licences.invalid/introspect"}
            ),
            introspect,
        ):
            middleware = AuthMiddleware(app=MagicMock())
            request = self._bearer_request("a-revoked-licence")
            call_next = AsyncMock()

            result = await middleware.dispatch(request, call_next)

        assert result.status_code == 401
        call_next.assert_not_awaited()
        clear_license_caches()

    @pytest.mark.asyncio
    async def test_licence_bearer_works_with_header_trust_disabled(self):
        """The combination this whole campaign is aiming at.

        With ``TRUST_TENANT_HEADER=false`` the forged-header arm is closed AND
        the licence Bearer still authenticates — a hardening, not the
        401-everything switch the flag was on coding-client <=1.9.0.
        """
        from coding_client.auth.tenant import clear_license_caches

        clear_license_caches()
        introspect, _ = self._introspection(
            {"active": True, "tenant_id": "tenant-from-licence", "subject_id": "subject-1"}
        )
        with (
            patch.dict(
                os.environ,
                {
                    "LICENSE_INTROSPECT_URL": "http://licences.invalid/introspect",
                    "TRUST_TENANT_HEADER": "false",
                },
            ),
            introspect,
        ):
            middleware = AuthMiddleware(app=MagicMock())
            request = self._bearer_request("a-real-licence")
            call_next = AsyncMock(return_value="response")

            await middleware.dispatch(request, call_next)

        call_next.assert_awaited()
        assert request.state.user.tenant_id == "tenant-from-licence"
        clear_license_caches()
