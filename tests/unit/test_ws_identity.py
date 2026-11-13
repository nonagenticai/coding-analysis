"""The quality WS tenant must be a resolved identity, not a header anyone can set.

`AuthMiddleware` is a `BaseHTTPMiddleware`; Starlette never invokes `dispatch`
on a `websocket` scope. So no middleware and no config flip can protect
`/api/quality/ws/{container_name}` — it is this service's only WebSocket route
(census over `src/`: 469 .py files, one `@router.websocket` decorator, one
`websocket.headers` read), and once the HTTP paths stopped trusting a bare
`X-Tenant-Id` (`TRUST_TENANT_HEADER=false`) it was among the last places in the
fleet where any pod could assert any tenant and be believed.

Every test here fails if the guard is removed and the raw header read comes
back.

The uncredentialed residual is CLOSED as of coding-client 1.15.0. It used to be
pinned here as a deliberate gap; that gap is what made a credential-less
handshake carrying only a forged `X-Tenant-Id` be believed, and this route
answered `connection_established` for a container the caller held no licence
for. `TestTheUncredentialedHandshakeIsClosed` below is the same class, inverted:
every case that used to prove the residual was served now proves it is refused.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coding_client.auth import _default_tenant as dt_mod
from coding_client.auth import _onbehalf as ob_mod
from coding_client.auth import _trusted_networks as tn_mod
from coding_client.auth.tenant import clear_license_caches
from fastapi import WebSocketDisconnect
from starlette.datastructures import Headers

from app.api.endpoints.quality import quality_websocket

# svc-coding-gateway — already in this service's CODING_ONBEHALF_SUBJECTS, so
# the allowlisted case exercised here is the production one.
GATEWAY_SUBJECT = "019ffa19-db80-700e-9840-8c17e23b35d5"
STRANGER_SUBJECT = "019ff971-7cbf-7505-8bad-1f86ab493f6e"
SERVICE_TENANT = "service-own-tenant-0001"
CUSTOMER_TENANT = "customer-tenant-0042"
VICTIM_TENANT = "victim-tenant-0099"
CONTAINER = "victim-box"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Production posture: no header trust, no trusted network, no default."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("SKIP_AUTH", raising=False)
    monkeypatch.delenv("CODING_ONBEHALF_SUBJECTS", raising=False)
    monkeypatch.setenv("TRUST_TENANT_HEADER", "false")
    monkeypatch.setenv("LICENSE_INTROSPECT_URL", "http://test-licenses/license/introspect")
    monkeypatch.setattr(tn_mod, "TRUSTED_NETWORKS", [])
    monkeypatch.setattr(dt_mod, "DEFAULT_TENANT_ID", None)
    ob_mod._warned_synthetic.clear()
    clear_license_caches()
    yield
    clear_license_caches()


def _ws(headers: dict[str, str] | None = None):
    """A WebSocket double exposing what the handler and resolver actually read.

    ``headers`` is Starlette's real :class:`Headers`, not a dict: it is
    case-insensitive, and the handler reads ``X-Tenant-Id`` while the SDK reads
    ``authorization`` / ``x-tenant-id``. A plain dict is *narrower than the real
    object* and would make half these assertions pass for the wrong reason.
    """
    ws = AsyncMock()
    ws.headers = Headers(headers or {})
    ws.url.path = f"/api/quality/ws/{CONTAINER}"
    ws.query_params = {}
    ws.client = MagicMock(host="10.42.7.13", port=51234)
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    ws.send_json = AsyncMock()
    # End the main loop immediately so the handler returns instead of hanging.
    ws.receive_json = AsyncMock(side_effect=WebSocketDisconnect())
    return ws


def _introspection(payload: dict | None):
    """Patch the licence introspection call the SDK makes."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    client = AsyncMock()
    client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return patch("coding_client.auth.tenant.httpx.AsyncClient", return_value=client)


def _licence(subject: str = GATEWAY_SUBJECT) -> dict:
    return {"active": True, "tenant_id": SERVICE_TENANT, "subject_id": subject}


def _container_ok():
    return patch(
        "app.api.endpoints.quality.validate_and_get_container_for_tenant",
        new=AsyncMock(return_value=MagicMock(codebase_path="/repo")),
    )


def _close_kwargs(ws):
    assert ws.close.await_args is not None, "the handler never closed the socket"
    return ws.close.await_args.kwargs


class TestCredentialledCallersCannotForge:
    """The hole this closes: a credential no longer buys an arbitrary tenant."""

    async def test_a_non_allowlisted_credential_asserting_a_tenant_is_closed_1008(
        self, monkeypatch
    ):
        monkeypatch.setenv("CODING_ONBEHALF_SUBJECTS", GATEWAY_SUBJECT)
        ws = _ws({"Authorization": "Bearer other", "X-Tenant-Id": VICTIM_TENANT})
        with (
            _introspection(_licence(subject=STRANGER_SUBJECT)),
            _container_ok() as validate,
        ):
            await quality_websocket(ws, CONTAINER)

        assert _close_kwargs(ws)["code"] == 1008
        # Never served: the container was never looked up for the asserted tenant.
        validate.assert_not_awaited()
        ws.send_json.assert_not_awaited()

    async def test_the_close_reason_never_echoes_the_asserted_tenant(self, monkeypatch):
        """The close frame is visible to whoever sent the guess."""
        monkeypatch.setenv("CODING_ONBEHALF_SUBJECTS", GATEWAY_SUBJECT)
        ws = _ws({"Authorization": "Bearer other", "X-Tenant-Id": VICTIM_TENANT})
        with _introspection(_licence(subject=STRANGER_SUBJECT)), _container_ok():
            await quality_websocket(ws, CONTAINER)

        reason = _close_kwargs(ws)["reason"]
        assert VICTIM_TENANT not in reason
        # Must be the AUTHORISATION refusal, not the generic missing-tenant close.
        # A handler that drops the refusal branch still reaches 1008 by accident
        # (a refusal carries no tenant, so it falls into the `not tenant_id`
        # arm) — asserting only the code would pass against exactly the bug this
        # test exists to catch.
        assert "Tenant context required" not in reason
        assert "may not act for the tenant it asserted" in reason

    async def test_an_empty_allowlist_refuses_a_differing_assertion(self):
        """Unset vouches for nobody — fail-closed on install."""
        ws = _ws({"Authorization": "Bearer gw", "X-Tenant-Id": VICTIM_TENANT})
        with _introspection(_licence()), _container_ok() as validate:
            await quality_websocket(ws, CONTAINER)

        assert _close_kwargs(ws)["code"] == 1008
        validate.assert_not_awaited()

    async def test_an_allowlisted_proxy_is_served_as_the_customer_tenant(self, monkeypatch):
        """The gateway's production shape, and the green half of the guard.

        The vouched call must resolve to the CUSTOMER tenant — not the
        gateway's own, which would scope every customer's quality stream to the
        service account.
        """
        monkeypatch.setenv("CODING_ONBEHALF_SUBJECTS", GATEWAY_SUBJECT)
        ws = _ws({"Authorization": "Bearer gw", "X-Tenant-Id": CUSTOMER_TENANT})
        with _introspection(_licence()), _container_ok() as validate:
            await quality_websocket(ws, CONTAINER)

        validate.assert_awaited_once_with(CONTAINER, CUSTOMER_TENANT)
        ws.send_json.assert_awaited()
        for call in ws.close.await_args_list:
            assert call.kwargs.get("code") != 1008

    async def test_no_assertion_is_served_as_the_licence_tenant(self):
        ws = _ws({"Authorization": "Bearer gw"})
        with _introspection(_licence()), _container_ok() as validate:
            await quality_websocket(ws, CONTAINER)

        validate.assert_awaited_once_with(CONTAINER, SERVICE_TENANT)

    async def test_an_agreeing_header_is_not_an_assertion(self):
        """coding-gateway re-injecting the tenant it read from this same token."""
        ws = _ws({"Authorization": "Bearer gw", "X-Tenant-Id": SERVICE_TENANT})
        with _introspection(_licence()), _container_ok() as validate:
            await quality_websocket(ws, CONTAINER)

        validate.assert_awaited_once_with(CONTAINER, SERVICE_TENANT)
        for call in ws.close.await_args_list:
            assert call.kwargs.get("code") != 1008


class TestTheUncredentialedHandshakeIsClosed:
    """Closed as of coding-client 1.15.0 — this class used to pin it open.

    The justification held while it held: a browser cannot set `Authorization`
    on a WS handshake, so a browser-originated socket is credentialed only if a
    server-side proxy does it, and fr's quality proxy sent none. Refusing before
    it was converted would have broken the live quality panel, not hardened it.

    The WARNING was the instrument for finding that proxy, and it has reported:
    over ~18h of retention on THIS service it fired ZERO times, and 5 times on
    coding-realtime — every one from peer=127.0.0.1, a verification probe. No
    real caller ever used it. fr now presents svc-fr-backend's licence on both
    of its WS proxies (driven live: 101 with auth, 403 without).
    """

    async def test_an_uncredentialed_caller_is_refused(self):
        """The forgery itself: a header, and nothing at all behind it.

        ⚠️ The REASON is asserted, not just the code. Because the raw-header
        fallback is gone, a handler that lost the `should_close` guard still
        reaches 1008 — via the `not tenant_id` arm, since a NO_CREDENTIAL
        identity carries no tenant. Asserting only the code would pass against
        exactly the regression this test exists to catch. The identity's own
        reason ("no verifiable credential on the handshake") is what proves the
        refusal happened at the guard.
        """
        ws = _ws({"X-Tenant-Id": CUSTOMER_TENANT})
        with _container_ok() as validate:
            await quality_websocket(ws, CONTAINER)

        assert _close_kwargs(ws)["code"] == 1008
        assert "no verifiable credential" in _close_kwargs(ws)["reason"]
        # Never served: the container was never looked up for the claimed tenant.
        validate.assert_not_awaited()
        ws.send_json.assert_not_awaited()

    async def test_the_refusal_never_echoes_the_asserted_tenant(self):
        """The close frame goes back to whoever sent the guess."""
        ws = _ws({"X-Tenant-Id": CUSTOMER_TENANT})
        with _container_ok():
            await quality_websocket(ws, CONTAINER)

        assert CUSTOMER_TENANT not in _close_kwargs(ws)["reason"]

    async def test_the_refusal_warns_and_names_the_peer(self, caplog):
        """Refusing silently would hide a caller that legitimately broke.

        The WARNING moved rather than disappeared: it used to say "accepted",
        it now says "Refused", and it still names the peer so an unconverted
        caller is identifiable from the log alone.
        """
        ws = _ws({"X-Tenant-Id": CUSTOMER_TENANT})
        with caplog.at_level(logging.WARNING), _container_ok():
            await quality_websocket(ws, CONTAINER)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "an uncredentialed refusal must be logged, not silent"
        rendered = " ".join(r.getMessage() for r in warnings)
        assert "Refused an uncredentialed quality WebSocket handshake" in rendered
        assert "10.42.7.13" in rendered
        assert "51234" in rendered

    async def test_a_credentialled_caller_does_not_trip_the_refusal_warning(self, caplog):
        """The warning must name the callers still to convert — not every caller."""
        ws = _ws({"Authorization": "Bearer gw"})
        with caplog.at_level(logging.WARNING), _introspection(_licence()), _container_ok():
            await quality_websocket(ws, CONTAINER)

        rendered = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        assert "Refused an uncredentialed quality WebSocket" not in rendered

    async def test_no_credential_and_no_header_is_closed_1008(self):
        """Never invent a tenant."""
        ws = _ws({})
        with _container_ok() as validate:
            await quality_websocket(ws, CONTAINER)

        assert _close_kwargs(ws)["code"] == 1008
        assert "no verifiable credential" in _close_kwargs(ws)["reason"]
        validate.assert_not_awaited()

    async def test_an_inactive_licence_is_refused(self):
        """A revoked token is not a credential, and now buys nothing."""
        ws = _ws({"Authorization": "Bearer revoked", "X-Tenant-Id": CUSTOMER_TENANT})
        with _introspection({"active": False, "reason": "token_invalid"}), _container_ok() as v:
            await quality_websocket(ws, CONTAINER)

        assert _close_kwargs(ws)["code"] == 1008
        assert "no verifiable credential" in _close_kwargs(ws)["reason"]
        v.assert_not_awaited()

    async def test_the_close_still_happens_after_accept(self):
        """The refusal must reach the client as a real 1008 Close frame.

        `websocket.close(code=…)` before `accept()` makes Starlette/uvicorn
        reject the upgrade with HTTP 403, and the client then sees a generic
        1006 abnormal-close instead of the policy violation. Closing the
        residual must not regress that ordering.
        """
        ws = _ws({"X-Tenant-Id": CUSTOMER_TENANT})
        with _container_ok():
            await quality_websocket(ws, CONTAINER)

        ws.accept.assert_awaited_once()
        assert _close_kwargs(ws)["code"] == 1008
        assert "no verifiable credential" in _close_kwargs(ws)["reason"]

    async def test_the_decision_is_taken_from_should_close_not_from_the_outcome(self):
        """A handler that re-derives `outcome is Outcome.REFUSED` bypasses the fix.

        This is the mutation that reopens the hole while every other test here
        still passes, so it is pinned directly: an identity whose `should_close`
        is True must be closed even when its outcome is NOT REFUSED.
        """
        from coding_client.auth.websocket import Outcome, WebSocketIdentity

        identity = WebSocketIdentity(
            outcome=Outcome.NO_CREDENTIAL, reason="no verifiable credential on the handshake"
        )
        assert identity.outcome is not Outcome.REFUSED
        assert identity.should_close is True

        ws = _ws({"X-Tenant-Id": CUSTOMER_TENANT})
        with (
            patch(
                "app.api.endpoints.quality.resolve_websocket_identity",
                new=AsyncMock(return_value=identity),
            ),
            _container_ok() as validate,
        ):
            await quality_websocket(ws, CONTAINER)

        assert _close_kwargs(ws)["code"] == 1008
        # See test_an_uncredentialed_caller_is_refused: the code alone does not
        # discriminate, because the `not tenant_id` arm also closes 1008.
        assert "no verifiable credential" in _close_kwargs(ws)["reason"]
        validate.assert_not_awaited()


class TestTheQueryStringBackdoorStaysClosed:
    """The 2026-07-29 finding. The edge filter cannot strip a query string."""

    async def test_a_query_param_tenant_is_ignored(self):
        ws = _ws({})
        ws.query_params = {"tenant_id": VICTIM_TENANT}
        with _container_ok() as validate:
            await quality_websocket(ws, CONTAINER)

        assert _close_kwargs(ws)["code"] == 1008
        assert "no verifiable credential" in _close_kwargs(ws)["reason"]
        validate.assert_not_awaited()

    async def test_a_query_param_cannot_override_a_licence(self):
        """Non-vacuity: since 1.15.0 every uncredentialed handshake closes, so
        "it closed" no longer proves the query string was ignored. This is the
        served half — a real licence resolves, and ?tenant_id= is still nowhere.
        """
        ws = _ws({"Authorization": "Bearer gw"})
        ws.query_params = {"tenant_id": VICTIM_TENANT}
        with _introspection(_licence()), _container_ok() as validate:
            await quality_websocket(ws, CONTAINER)

        validate.assert_awaited_once_with(CONTAINER, SERVICE_TENANT)

    def test_no_source_file_reads_tenant_id_from_query_params(self):
        from pathlib import Path

        src = Path(__file__).resolve().parents[2] / "src"
        py_files = list(src.rglob("*.py"))
        assert py_files, "positive control: the census must actually see source files"
        offenders = [
            str(p.relative_to(src))
            for p in py_files
            if 'query_params.get("tenant_id")' in p.read_text()
        ]
        assert offenders == [], (
            f"tenant_id read from query params in: {offenders}. The edge filter "
            f"strips headers but cannot strip a query string."
        )
