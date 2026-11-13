"""Best-effort persistence of the coding-security dependency report.

Written at deep-analysis time (when the job container is still live), so the fr
"Dependency Report" can degrade to a stored snapshot after the container is
deprovisioned — mirroring how the Architecture graph reads persisted nodes/edges
from this service's Postgres. See migration ``0003`` and the reader in
``app/api/endpoints/dependency_snapshots.py``.

The dependency report itself is computed by coding-security (it execs into the
live container to read manifest files); we simply fetch that report while the
container is up and stash the JSON keyed by ``(code_graph_id == container_name,
tenant_id)``.
"""

from __future__ import annotations

import logging
import os

import httpx
from coding_client.auth.outbound import tenant_headers
from psycopg.types.json import Jsonb

from config.settings import get_settings
from db.session import get_raw_connection

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(30.0, connect=5.0)

_UPSERT_SQL = (
    "INSERT INTO dependency_report_snapshots "
    "(code_graph_id, tenant_id, report, computed_at) "
    "VALUES (%s, %s, %s, now()) "
    "ON CONFLICT (code_graph_id, tenant_id) DO UPDATE "
    "SET report = EXCLUDED.report, computed_at = EXCLUDED.computed_at"
)


async def store_dependency_report_snapshot(container_name: str, tenant_id: str | None) -> bool:
    """Fetch the coding-security dependency report and upsert it as a snapshot.

    Best-effort: any failure (security unreachable, non-200, bad JSON, DB error)
    is logged and swallowed — this MUST NOT fail the analysis that calls it.
    Returns ``True`` only when a snapshot row was written.

    Requires a tenant. This previously sent **no headers at all** when
    ``tenant_id`` was falsy, which reached coding-security only through that
    service's trusted-network fallback (any RFC1918 source IP, i.e. the whole pod
    CIDR) and then wrote the row under a NULL tenant. Note this one must NOT get a
    ``DEFAULT_TENANT_ID`` fallback the way the fire-and-forget kickoffs do: the
    result is upserted keyed by ``(code_graph_id, tenant_id)``, so guessing a
    tenant files another tenant's dependency report under it. Unknown tenant ⇒
    skip, which is exactly what "best-effort" already licenses.
    """
    if not tenant_id:
        logger.warning(
            "No tenant in scope; skipping dependency snapshot for %s "
            "(a snapshot is keyed by tenant — writing one under a guessed tenant "
            "would file it against the wrong one)",
            container_name,
        )
        return False

    base_url = get_settings().CODING_SECURITY_URL.rstrip("/")
    if not base_url:
        logger.warning(
            "CODING_SECURITY_URL unset; skipping dependency snapshot for %s",
            container_name,
        )
        return False

    # The tenant is the CUSTOMER's and is REQUIRED here (the caller above
    # refuses a falsy one) — the result is upserted keyed by
    # (code_graph_id, tenant_id), so a guessed tenant files another tenant's
    # dependency report under it. The service license proves WHO is asking;
    # coding-security honours the asserted tenant only when svc-coding-analysis
    # is in its CODING_ONBEHALF_SUBJECTS. This call is best-effort background
    # work with no inbound request to borrow a credential from.
    headers = tenant_headers(
        tenant_id,
        operation="coding-security dependency report",
        service_token=os.getenv("CODING_SERVICE_TOKEN") or None,
    )
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{base_url}/api/dependencies/{container_name}/report",
                headers=headers,
            )
    except httpx.HTTPError as exc:
        logger.warning("Dependency snapshot fetch failed for %s: %s", container_name, exc)
        return False

    if resp.status_code != 200:
        logger.info(
            "Dependency snapshot skipped for %s: security report HTTP %d",
            container_name,
            resp.status_code,
        )
        return False

    try:
        report = resp.json()
    except ValueError as exc:
        logger.warning("Dependency snapshot for %s: non-JSON report body: %s", container_name, exc)
        return False

    try:
        async with get_raw_connection() as conn:
            await conn.execute(_UPSERT_SQL, (container_name, tenant_id, Jsonb(report)))
            await conn.commit()
    except Exception as exc:  # noqa: BLE001 - best-effort persistence, never propagate
        logger.warning("Dependency snapshot DB upsert failed for %s: %s", container_name, exc)
        return False

    logger.info("Stored dependency report snapshot for %s (tenant=%s)", container_name, tenant_id)
    return True
