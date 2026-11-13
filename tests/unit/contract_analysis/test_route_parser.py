from processing.contract_analysis.parsers.route_parser import parse_routes

SAMPLE_ROUTES = """
from fastapi import APIRouter, Request
from src.api.schemas import AnalyzeRequest

router = APIRouter(prefix="/api", tags=["analyze"])

@router.post("/analyze", status_code=202)
async def start_analysis(body: AnalyzeRequest, request: Request):
    pass

@router.get("/analyze/{job_id}")
async def get_analysis(job_id: str, request: Request):
    pass

@router.delete("/analyze/{job_id}")
async def cancel_analysis(job_id: str, request: Request):
    pass
"""


def test_parse_routes():
    result = parse_routes(SAMPLE_ROUTES)
    assert len(result.routes) == 3


def test_route_methods():
    result = parse_routes(SAMPLE_ROUTES)
    methods = sorted([r.method for r in result.routes])
    assert methods == ["DELETE", "GET", "POST"]


def test_route_paths_include_prefix():
    result = parse_routes(SAMPLE_ROUTES)
    paths = [r.path for r in result.routes]
    assert "/api/analyze" in paths
    assert "/api/analyze/{job_id}" in paths


def test_request_model_detected():
    result = parse_routes(SAMPLE_ROUTES)
    post_route = [r for r in result.routes if r.method == "POST"][0]
    assert post_route.request_model == "AnalyzeRequest"


def test_no_request_model_for_get():
    result = parse_routes(SAMPLE_ROUTES)
    get_route = [r for r in result.routes if r.method == "GET"][0]
    assert get_route.request_model is None


def test_empty_source():
    result = parse_routes("")
    assert len(result.routes) == 0


def test_return_annotation_response_model():
    """Detect -> ResponseModel return annotations as response_model."""
    source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")

@router.post("/foo")
async def create(body: FooRequest) -> FooResponse:
    pass

@router.get("/foo/{id}")
async def get(id: str) -> FooResponse:
    pass
"""
    result = parse_routes(source)
    assert len(result.routes) == 2
    post_route = [r for r in result.routes if r.method == "POST"][0]
    assert post_route.response_model == "FooResponse"
    get_route = [r for r in result.routes if r.method == "GET"][0]
    assert get_route.response_model == "FooResponse"


def test_return_annotation_none():
    """Functions without return annotations have no response_model."""
    source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")

@router.post("/bar")
async def create(body: BarRequest):
    pass
"""
    result = parse_routes(source)
    assert result.routes[0].response_model is None


def test_return_annotation_builtin_ignored():
    """Return annotations of dict, list, etc. are NOT response models."""
    source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")

@router.get("/status")
async def status() -> dict:
    pass
"""
    result = parse_routes(source)
    assert result.routes[0].response_model is None


def test_empty_path_string():
    """Routes with empty path string should use prefix only."""
    source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api/redesign")

@router.post("")
async def start(body: RedesignRequest) -> RedesignResponse:
    pass

@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> JobStatus:
    pass
"""
    result = parse_routes(source)
    assert len(result.routes) == 2
    paths = [r.path for r in result.routes]
    assert "/api/redesign" in paths
    assert "/api/redesign/jobs/{job_id}" in paths


def test_empty_path_with_prefix():
    """Empty path with prefix should resolve to just the prefix."""
    source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api/compile")

@router.post("")
async def start_compile(body: CompileRequest) -> CompileResponse:
    pass
"""
    result = parse_routes(source)
    assert len(result.routes) == 1
    assert result.routes[0].path == "/api/compile"
    assert result.routes[0].method == "POST"


def test_external_prefix_override():
    """External prefix is used when file has no APIRouter prefix."""
    source = """
from fastapi import APIRouter
router = APIRouter()

@router.post("/generate")
async def generate(body: GenRequest) -> GenResponse:
    pass

@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> JobStatus:
    pass
"""
    result = parse_routes(source, prefix="/api/videogen")
    assert len(result.routes) == 2
    paths = sorted([r.path for r in result.routes])
    assert paths == ["/api/videogen/generate", "/api/videogen/jobs/{job_id}"]


def test_external_prefix_composes_with_local():
    """External prefix composes with local prefix (FastAPI include_router behavior)."""
    source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api/local")

@router.get("/status")
async def status() -> dict:
    pass
"""
    result = parse_routes(source, prefix="/api/external")
    assert len(result.routes) == 1
    # Prefixes compose: external + local
    assert result.routes[0].path == "/api/external/api/local/status"


def test_prefix_composition_stacks():
    """Aggregator prefix + local prefix compose (FastAPI include_router behavior)."""
    source = """
from fastapi import APIRouter
router = APIRouter(prefix="/presets")

@router.get("")
async def list_presets() -> list: pass

@router.get("/{preset_id}")
async def get_preset(preset_id: str) -> dict: pass
"""
    result = parse_routes(source, prefix="/api/concierge")
    paths = sorted(r.path for r in result.routes)
    assert paths == ["/api/concierge/presets", "/api/concierge/presets/{preset_id}"]


def test_multi_router_per_file():
    """Multiple routers in one file get correct per-router prefixes."""
    source = """
from fastapi import APIRouter

client_router = APIRouter(prefix="/api/client-portal")
owner_router = APIRouter(prefix="/api/projects")

@client_router.get("/dashboard")
async def client_dashboard() -> dict: pass

@owner_router.get("/list")
async def project_list() -> dict: pass
"""
    result = parse_routes(source)
    assert len(result.routes) == 2
    paths = sorted(r.path for r in result.routes)
    assert paths == ["/api/client-portal/dashboard", "/api/projects/list"]


def test_status_code_extraction():
    """Extract status_code from decorator kwargs."""
    source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")

@router.post("/analyze", status_code=202)
async def start(body: AnalyzeRequest): pass

@router.get("/status")
async def status() -> dict: pass
"""
    result = parse_routes(source)
    post_route = [r for r in result.routes if r.method == "POST"][0]
    assert post_route.status_code == 202
    get_route = [r for r in result.routes if r.method == "GET"][0]
    assert get_route.status_code == 200


def test_query_param_extraction():
    """Extract Query() parameters from function signature."""
    source = """
from fastapi import APIRouter, Query
router = APIRouter(prefix="/api")

@router.get("/items")
async def list_items(page: int = Query(...), limit: int = Query(default=20)):
    pass
"""
    result = parse_routes(source)
    assert len(result.routes) == 1
    route = result.routes[0]
    qp_names = {qp.name for qp in route.query_params}
    assert "page" in qp_names
    assert "limit" in qp_names
    qp_by_name = {qp.name: qp for qp in route.query_params}
    assert qp_by_name["page"].required is True
    assert qp_by_name["limit"].required is False


def test_fastapi_app_routes_do_not_inherit_the_only_apirouter_prefix():
    from processing.contract_analysis.parsers.route_parser import parse_routes

    src = (
        "from fastapi import APIRouter, FastAPI\n"
        "app = FastAPI()\n"
        'api_router = APIRouter(prefix="/api")\n'
        '@app.get("/health")\n'
        "async def health(): ...\n"
        '@api_router.get("/x")\n'
        "async def x(): ...\n"
    )
    routes = parse_routes(src).routes
    assert [(r.path, r.line, r.handler, r.router) for r in routes] == [
        ("/health", 4, "health", "app"),
        ("/api/x", 6, "x", "api_router"),
    ]
