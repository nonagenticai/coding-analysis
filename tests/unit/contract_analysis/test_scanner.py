from processing.contract_analysis.scanner import ContractScanner

MOCK_CLIENT = """
class FooClient(BaseServiceClient):
    async def create(self):
        return await self.submit_job("/api/foo", {"name": "test"})
    async def get(self, id):
        return await self._request("GET", f"/api/foo/{id}")
"""

MOCK_SCHEMA = """
from pydantic import BaseModel
class FooRequest(BaseModel):
    name: str
    description: str
class FooResponse(BaseModel):
    id: str
    name: str
"""

MOCK_ROUTES = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")
@router.post("/foo")
async def create(body: FooRequest): pass
@router.get("/foo/{id}")
async def get(id: str): pass
"""


def test_scanner_finds_violation():
    scanner = ContractScanner()
    result = scanner.scan_sources(
        name="FooClient -> foo_service",
        client_source=MOCK_CLIENT,
        schema_sources=[MOCK_SCHEMA],
        route_sources=[MOCK_ROUTES],
    )
    assert result.passed is False
    assert len(result.violations) >= 1
    msgs = [v.message for v in result.violations]
    assert any("description" in m for m in msgs)


def test_scanner_clean_contract():
    clean_client = """
class BarClient(BaseServiceClient):
    async def create(self):
        return await self.submit_job("/api/bar", {"name": "test"})
"""
    clean_schema = """
from pydantic import BaseModel
class BarRequest(BaseModel):
    name: str
"""
    clean_routes = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")
@router.post("/bar")
async def create(body: BarRequest): pass
"""
    scanner = ContractScanner()
    result = scanner.scan_sources(
        name="BarClient -> bar_service",
        client_source=clean_client,
        schema_sources=[clean_schema],
        route_sources=[clean_routes],
    )
    assert result.passed is True
    assert len(result.violations) == 0


def test_scanner_enriches_violations():
    """Verify route_path violations have enrichment fields after scan."""
    client_bad_path = """
class BazClient(BaseServiceClient):
    async def create(self):
        return await self._request("POST", "/api/baz/generate")
"""
    schema = """
from pydantic import BaseModel
class BazRequest(BaseModel):
    name: str
"""
    routes = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")
@router.post("/baz/export/generate")
async def generate(body: BazRequest): pass
"""
    scanner = ContractScanner()
    result = scanner.scan_sources(
        name="BazClient -> baz_service",
        client_source=client_bad_path,
        schema_sources=[schema],
        route_sources=[routes],
    )
    route_violations = [v for v in result.violations if v.check == "route_path"]
    assert len(route_violations) >= 1
    for v in route_violations:
        assert v.root_cause != ""
        assert v.suggestion != ""


def test_cross_file_schema_inheritance():
    """Schemas split across files resolve inheritance correctly."""
    client = """
class FooClient(BaseServiceClient):
    async def create(self, data):
        return await self.submit_job("/api/devices", {"ip": "1.2.3.4"})
"""
    schema_base = """
from pydantic import BaseModel

class DeviceBase(BaseModel):
    name: str
"""
    schema_child = """
from pydantic import BaseModel

class Device(DeviceBase):
    ip: str
"""
    routes = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")
@router.post("/devices")
async def create(body: Device): pass
"""
    scanner = ContractScanner()
    result = scanner.scan_sources(
        name="cross-file",
        client_source=client,
        schema_sources=[schema_base, schema_child],
        route_sources=[routes],
    )
    # Client sends only "ip" but Device (via DeviceBase) also requires "name"
    msgs = [v.message for v in result.violations]
    assert any("name" in m for m in msgs)


def test_per_file_route_prefix():
    """Per-file prefix list applies different prefixes to each route source."""
    client = """
class MultiClient(BaseServiceClient):
    async def get_public(self):
        return await self._request("GET", "/api/items")
    async def get_internal(self):
        return await self._request("GET", "/_internal/v1/items")
"""
    schema = """
from pydantic import BaseModel
class ItemResponse(BaseModel):
    id: str
"""
    routes_public = """
from fastapi import APIRouter
router = APIRouter(prefix="/items")
@router.get("")
async def list_items() -> list: pass
"""
    routes_internal = """
from fastapi import APIRouter
router = APIRouter(prefix="/items")
@router.get("")
async def list_items_internal() -> list: pass
"""
    scanner = ContractScanner()
    result = scanner.scan_sources(
        name="test",
        client_source=client,
        schema_sources=[schema],
        route_sources=[routes_public, routes_internal],
        route_prefix=["/api", "/_internal/v1"],
    )
    route_violations = [v for v in result.violations if v.check == "route_path"]
    assert len(route_violations) == 0


def test_scanner_syntax_error_in_route_source():
    """Route source with real syntax error triggers FAIL via fallback sources."""
    client = """
class FooClient(BaseServiceClient):
    async def create(self):
        return await self.submit_job("/api/foo", {"name": "test"})
"""
    schema = """
from pydantic import BaseModel
class FooRequest(BaseModel):
    name: str
"""
    bad_routes = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")
def broken(
@router.post("/foo")
async def create(body: FooRequest): pass
"""
    scanner = ContractScanner()
    result = scanner.scan_sources(
        name="syntax-test",
        client_source=client,
        schema_sources=[schema],
        route_sources=[bad_routes],
    )
    syntax_violations = [v for v in result.violations if v.check == "python3_syntax"]
    assert len(syntax_violations) == 1
    assert result.passed is False
    assert result.stats["syntax_errors"] == 1


def test_scanner_pep758_in_route_source_tolerated():
    """PEP 758 except in route source is tolerated (Python 3.14 valid)."""
    client = """
class FooClient(BaseServiceClient):
    async def create(self):
        return await self.submit_job("/api/foo", {"name": "test"})
"""
    schema = """
from pydantic import BaseModel
class FooRequest(BaseModel):
    name: str
"""
    routes_with_pep758 = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")
try:
    pass
except ValueError, TypeError:
    pass
@router.post("/foo")
async def create(body: FooRequest): pass
"""
    scanner = ContractScanner()
    result = scanner.scan_sources(
        name="pep758-route-test",
        client_source=client,
        schema_sources=[schema],
        route_sources=[routes_with_pep758],
    )
    syntax_violations = [v for v in result.violations if v.check == "python3_syntax"]
    assert len(syntax_violations) == 0
    assert result.stats["syntax_errors"] == 0


def test_scanner_pep758_syntax_not_flagged():
    """PEP 758 unparenthesized except (valid Python 3.14) should not be flagged."""
    client = """
class FooClient(BaseServiceClient):
    async def create(self):
        return await self.submit_job("/api/foo", {"name": "test"})
"""
    schema = """
from pydantic import BaseModel
class FooRequest(BaseModel):
    name: str
"""
    routes = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")
@router.post("/foo")
async def create(body: FooRequest): pass
"""
    scanner = ContractScanner()
    result = scanner.scan_sources(
        name="pep758-test",
        client_source=client,
        schema_sources=[schema],
        route_sources=[routes],
        service_sources={
            "src/utils/helpers.py": """
import json

def parse_config(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError, ValueError, TypeError:
        return {}
""",
        },
    )
    syntax_violations = [v for v in result.violations if v.check == "python3_syntax"]
    assert len(syntax_violations) == 0


def test_scanner_syntax_error_in_service_sources():
    """Explicit service_sources with syntax error triggers FAIL."""
    client = """
class FooClient(BaseServiceClient):
    async def create(self):
        return await self.submit_job("/api/foo", {"name": "test"})
"""
    schema = """
from pydantic import BaseModel
class FooRequest(BaseModel):
    name: str
"""
    routes = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")
@router.post("/foo")
async def create(body: FooRequest): pass
"""
    scanner = ContractScanner()
    result = scanner.scan_sources(
        name="svc-sources-test",
        client_source=client,
        schema_sources=[schema],
        route_sources=[routes],
        service_sources={
            "src/utils/helpers.py": "def foo(\n",
            "src/api/routes.py": routes,
        },
    )
    syntax_violations = [v for v in result.violations if v.check == "python3_syntax"]
    assert len(syntax_violations) == 1
    assert "helpers.py" in syntax_violations[0].message
    assert result.stats["syntax_errors"] == 1
