"""The /api/analysis/query router is gone, not merely refusing.

It published get_analysis_summary and query_analysis_results, which always
answered 501 ("analysis query has no result store"). Dead operations in the
published API also become dead operations in the generated n8n node, so they
were removed. These tests pin the removal against the real app.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

REMOVED_OPERATION_IDS = {"get_analysis_summary", "query_analysis_results"}
REMOVED_PATHS = (
    "/api/analysis/query/proj/summary",
    "/api/analysis/query/proj/query",
)


@pytest.fixture
def _transport():
    from main import app

    return ASGITransport(app=app)


@pytest.mark.asyncio
async def test_removed_ops_absent_from_openapi(_transport):
    async with AsyncClient(transport=_transport, base_url="http://test") as client:
        resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    spec = resp.json()
    paths = spec["paths"]
    # Positive control: the spec is the real one, not an empty document.
    assert any(p.startswith("/api/analysis/") for p in paths)
    assert not [p for p in paths if p.startswith("/api/analysis/query")]
    op_ids = {op.get("operationId") for item in paths.values() for op in item.values()}
    assert not (op_ids & REMOVED_OPERATION_IDS)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", REMOVED_PATHS)
async def test_removed_routes_answer_404(_transport, path):
    async with AsyncClient(transport=_transport, base_url="http://test") as client:
        resp = await client.get(path)
    assert resp.status_code == 404
