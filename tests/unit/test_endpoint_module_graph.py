"""POST /api/analysis/structure/module-graph over HTTP: the response carries ``slices``.

The route, workspace read, isolated extractor and response model are real. Only the
two calls that reach the coding-containers SERVICE are replaced: the tenant/container
lookup, and the copy of the workspace files out of the container (which here copies a
real Python tree from disk instead).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

TENANT = "tenant-a"
URL = "/api/analysis/structure/module-graph"

#: a -> b -> c, d <-> e (a cycle), e -> c.
TREE = {
    "a.py": "import b\n\n\ndef run():\n    return b.go()\n",
    "b.py": "from c import C\n\n\ndef go():\n    return C\n",
    "c.py": "C = 1\n",
    "d.py": "import e\n\n\ndef dd():\n    return e.ee()\n",
    "e.py": "import c\n\n\ndef ee():\n    import d\n\n    return c.C, d\n",
}


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app.api.endpoints.structure as module
    import app.api.utils.container_validation as cv
    from auth.tenant import require_tenant

    for rel, text in TREE.items():
        (tmp_path / rel).write_text(text)

    async def lookup(container_name, tenant_id):
        assert (container_name, tenant_id) == ("ws-1", TENANT)
        return SimpleNamespace(codebase_path="/workspace")

    async def copy_out(_client, container_name, base, patterns, *_a, **_kw):
        assert (container_name, base) == ("ws-1", "/workspace")
        dest = Path(tempfile.mkdtemp(prefix="structure-test-"))
        shutil.copytree(tmp_path, dest, dirs_exist_ok=True)
        return dest

    monkeypatch.setattr(cv, "validate_and_get_container_for_tenant", lookup)
    monkeypatch.setattr(module, "materialize_from_container", copy_out)
    monkeypatch.setattr(module, "get_coding_containers_client", lambda: None)
    app = FastAPI()
    app.include_router(module.router, prefix="/api/analysis/structure")
    app.dependency_overrides[require_tenant] = lambda: TENANT
    return TestClient(app)


def test_module_graph_http_response_has_slices(client):
    resp = client.post(URL, json={"container_name": "ws-1", "language": "python"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["files_read"] == 5
    assert body["sccs"] == [["d", "e"]]
    assert body["order"] == [["c"], ["b", "d", "e"], ["a"]]
    assert body["slices"] == [
        {"id": "s0", "layer": 0, "modules": ["c"], "cyclic": False, "depends_on": [], "loc": 1},
        {"id": "s1", "layer": 1, "modules": ["b"], "cyclic": False, "depends_on": ["s0"], "loc": 3},
        {
            "id": "s2",
            "layer": 1,
            "modules": ["d", "e"],
            "cyclic": True,
            "depends_on": ["s0"],
            "loc": 7,
        },
        {"id": "s3", "layer": 2, "modules": ["a"], "cyclic": False, "depends_on": ["s1"], "loc": 3},
    ]
