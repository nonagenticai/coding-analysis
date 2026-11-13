"""The pool must VALIDATE a connection before handing it out.

THE INCIDENT (measured 2026-09-01). The ``pgbouncer`` Deployment rolled at 09:22:50Z — new
ConfigMap, new ReplicaSet, both pods replaced. This service's pod was **2d20h old with 0
restarts**, so every connection in its ``AsyncConnectionPool`` still pointed at pgbouncer pods
that no longer existed. The pool had no reason to discard them, so it served them indefinitely::

    first call after the roll : consuming input failed: server closed the connection unexpectedly
    every call after that     : the connection is closed      (~35 ms, BEFORE any query ran)

Architecture endpoints returned **6x 500 / 0x 200 for three hours** and did not self-heal; only a
pod restart cleared it. Crucially there is **no backend death in the postgres log** — nothing
crashed, the proxy simply moved — so the usual "look for a Postgres restart" diagnosis finds
nothing and the outage reads as unexplained.

``check=AsyncConnectionPool.check_connection`` makes the pool probe a connection and discard a
dead one, turning a proxy roll into one wasted probe instead of an outage.
"""

from __future__ import annotations

import inspect

import psycopg_pool
import pytest

from src.db import session as session_mod


@pytest.mark.asyncio
async def test_pool_is_constructed_with_a_liveness_check(monkeypatch):
    """RED against a pool built without ``check=`` — the shape that caused the outage."""
    captured: dict = {}

    class _FakePool:
        # The source references ``AsyncConnectionPool.check_connection`` at construction time, so
        # the double must carry it or the patch itself raises AttributeError and the test fails
        # for a reason that has nothing to do with the defect.
        check_connection = staticmethod(lambda conn: None)

        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def open(self, wait: bool = False):  # noqa: D401 - test double
            return None

    monkeypatch.setattr(session_mod, "_pool", None, raising=False)
    monkeypatch.setattr(session_mod, "AsyncConnectionPool", _FakePool)
    monkeypatch.setattr(session_mod, "_resolve_dsn", lambda: "postgresql://u@h/db")

    await session_mod.get_pool()

    assert "check" in captured, (
        "the pool was built with NO check= — it will serve a socket whose far end is gone, "
        "forever. This is the 2026-09-01 pgbouncer-roll outage."
    )
    assert captured["check"] is not None, "check= must not be disabled"


def test_check_connection_is_the_real_psycopg_api():
    """Guard the fix against being 'satisfied' by an attribute that does not exist upstream."""
    assert hasattr(psycopg_pool.AsyncConnectionPool, "check_connection")
    assert "check" in inspect.signature(psycopg_pool.AsyncConnectionPool.__init__).parameters


def test_prepared_statements_stay_disabled_for_pgbouncer_txn_mode():
    """NEGATIVE CONTROL: the new kwarg must not have displaced the existing pgbouncer safeguard."""
    src = inspect.getsource(session_mod)
    assert "prepare_threshold = None" in src
    assert "configure=_configure_connection" in src
