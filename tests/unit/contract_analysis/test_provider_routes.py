"""Workspace provider routes: FastAPI include_router mounts + TypeScript routes."""

from __future__ import annotations

from processing.contract_analysis.provider_routes import extract_provider_routes

MAIN = """\
from fastapi import APIRouter, FastAPI
from app.api.endpoints.contracts import router as contracts_router
from app.api.endpoints import analysis
from .api import items

app = FastAPI()
api_router = APIRouter(prefix="/api")

def _include():
    api_router.include_router(contracts_router, prefix="/analysis/contracts")
    api_router.include_router(analysis.router, prefix="/analysis")
    api_router.include_router(items.router)
    app.include_router(api_router)

@app.get("/health")
async def health(): ...
"""

CONTRACTS = """\
from fastapi import APIRouter
router = APIRouter()

@router.post("/scan")
async def scan_contracts(body): ...

@router.post("/call-sites")
async def scan_call_sites(body): ...
"""

ANALYSIS = """\
from fastapi import APIRouter
from .modules import symbols_router
router = APIRouter()
router.include_router(symbols_router)

@router.get("/status/{job_id}/")
async def status(job_id: str): ...
"""

MODULES_INIT = "from .symbols import router as symbols_router\n"

SYMBOLS = """\
from fastapi import APIRouter
router = APIRouter()

@router.post("/symbols")
async def extract_symbols(body): ...
"""

ITEMS = """\
from fastapi import APIRouter
router = APIRouter(prefix="/items")

@router.delete("/{item_id}")
async def delete_item(item_id: str): ...
"""


def _keys(result):
    return sorted((r.method, r.path, r.handler, r.mount_prefix) for r in result.routes)


def test_fastapi_include_router_prefixes_across_files():
    result = extract_provider_routes(
        {
            "src/main.py": MAIN,
            "src/app/api/endpoints/contracts.py": CONTRACTS,
            "src/app/api/endpoints/analysis.py": ANALYSIS,
            "src/app/api/endpoints/modules/__init__.py": MODULES_INIT,
            "src/app/api/endpoints/modules/symbols.py": SYMBOLS,
            "src/api/items.py": ITEMS,
        }
    )
    assert _keys(result) == sorted(
        [
            ("GET", "/health", "health", None),
            ("POST", "/api/analysis/contracts/scan", "scan_contracts", "/api/analysis/contracts"),
            (
                "POST",
                "/api/analysis/contracts/call-sites",
                "scan_call_sites",
                "/api/analysis/contracts",
            ),
            ("GET", "/api/analysis/status/{param}", "status", "/api/analysis"),
            ("POST", "/api/analysis/symbols", "extract_symbols", "/api/analysis"),
            ("DELETE", "/api/items/{param}", "delete_item", "/api"),
        ]
    )
    assert result.unresolved == []
    route = next(r for r in result.routes if r.handler == "scan_contracts")
    assert (route.file, route.line, route.framework) == (
        "src/app/api/endpoints/contracts.py",
        4,
        "fastapi",
    )


def test_dynamic_prefix_unknown_target_and_orphan_router_are_unresolved():
    main = """\
from fastapi import FastAPI
from app.a import router as a_router
from external_pkg import router as ext_router
app = FastAPI()
app.include_router(a_router, prefix=settings.PREFIX)
app.include_router(ext_router, prefix="/ext")
"""
    a = 'from fastapi import APIRouter\nrouter = APIRouter()\n@router.get("/a")\nasync def a(): ...\n'
    orphan = 'from fastapi import APIRouter\nrouter = APIRouter()\n@router.get("/o")\nasync def o(): ...\n'
    result = extract_provider_routes({"main.py": main, "app/a.py": a, "app/orphan.py": orphan})
    assert result.routes == []
    assert sorted((u.file, u.reason) for u in result.unresolved) == [
        ("app/a.py", "included under a dynamic prefix"),
        ("app/orphan.py", "APIRouter that no scanned file includes"),
        ("main.py", "included router is not a router in the scanned files"),
    ]


def test_single_file_without_includes_keeps_its_own_prefix():
    result = extract_provider_routes({"items.py": ITEMS})
    assert _keys(result) == [("DELETE", "/items/{param}", "delete_item", None)]


def test_python_and_typescript_together():
    ts = "const router = new Router();\nrouter.add('GET', '/api/projects/:id', handleGet);\n"
    result = extract_provider_routes({"items.py": ITEMS, "src/server.ts": ts})
    assert [(r.file, r.method, r.path, r.framework, r.handler) for r in result.routes] == [
        ("items.py", "DELETE", "/items/{param}", "fastapi", "delete_item"),
        ("src/server.ts", "GET", "/api/projects/{param}", "method-first", "handleGet"),
    ]


# --------------------------------------------------------------------------- Flask prefixes

_FLASK_BP = (
    "from flask import Blueprint\n"
    "bp = Blueprint('items', __name__{own})\n"
    "@bp.route('/x')\ndef x():\n    return ''\n"
)


def _flask(own: str, register: str) -> list[str]:
    app = (
        "from flask import Flask\nfrom app.items import bp\n"
        f"app = Flask(__name__)\napp.register_blueprint(bp{register})\n"
    )
    files = {"app/main.py": app, "app/items.py": _FLASK_BP.format(own=own)}
    result = extract_provider_routes(files)
    assert result.unresolved == []
    return [r.path for r in result.routes]


def test_flask_blueprint_own_prefix_only():
    assert _flask(", url_prefix='/bp'", "") == ["/bp/x"]


def test_flask_register_prefix_only():
    assert _flask("", ", url_prefix='/reg'") == ["/reg/x"]


def test_flask_register_prefix_replaces_blueprint_prefix():
    # Real Flask: the register_blueprint url_prefix REPLACES the blueprint's own.
    assert _flask(", url_prefix='/bp'", ", url_prefix='/reg'") == ["/reg/x"]


def test_flask_unregistered_blueprint_keeps_its_own_prefix():
    result = extract_provider_routes({"app/items.py": _FLASK_BP.format(own=", url_prefix='/bp'")})
    assert [(r.path, r.framework) for r in result.routes] == [("/bp/x", "flask")]


def test_flask_nested_blueprints_append_child_to_parent_effective_prefix():
    parent = (
        "from flask import Blueprint\nfrom app.child import child\n"
        "parent = Blueprint('parent', __name__, url_prefix='/parent')\n"
        "parent.register_blueprint(child)\n"
        "parent.register_blueprint(child, url_prefix='/again', name='again')\n"
        "@parent.route('/p')\ndef p():\n    return ''\n"
    )
    child = (
        "from flask import Blueprint\nchild = Blueprint('child', __name__, url_prefix='/child')\n"
        "@child.route('/c')\ndef c():\n    return ''\n"
    )
    app = (
        "from flask import Flask\nfrom app.parent import parent\n"
        "app = Flask(__name__)\napp.register_blueprint(parent, url_prefix='/api')\n"
    )
    result = extract_provider_routes(
        {"app/main.py": app, "app/parent.py": parent, "app/child.py": child}
    )
    assert result.unresolved == []
    assert sorted(r.path for r in result.routes) == [
        "/api/again/c",
        "/api/child/c",
        "/api/p",
    ]


def test_flask_unregistered_blueprint_is_unresolved_when_others_are_registered():
    app = (
        "from flask import Flask\nfrom app.items import bp\n"
        "app = Flask(__name__)\napp.register_blueprint(bp)\n"
    )
    orphan = "from flask import Blueprint\nlost = Blueprint('lost', __name__)\n"
    orphan += "@lost.route('/o')\ndef o():\n    return ''\n"
    result = extract_provider_routes(
        {"app/main.py": app, "app/items.py": _FLASK_BP.format(own=""), "app/lost.py": orphan}
    )
    assert [(u.file, u.reason) for u in result.unresolved] == [
        ("app/lost.py", "Blueprint that no scanned file registers")
    ]


def test_fastapi_include_prefix_concatenates_with_router_prefix():
    # FastAPI, unlike Flask, concatenates include_router(prefix) + APIRouter(prefix),
    # and a nested router adds the parent's prefixes in front.
    child = (
        "from fastapi import APIRouter\nrouter = APIRouter(prefix='/child')\n"
        "@router.get('/c')\nasync def c(): ...\n"
    )
    parent = (
        "from fastapi import APIRouter\nfrom app.child import router as child_router\n"
        "router = APIRouter(prefix='/parent')\n"
        "router.include_router(child_router, prefix='/inc')\n"
        "@router.get('/p')\nasync def p(): ...\n"
    )
    main = (
        "from fastapi import FastAPI\nfrom app.parent import router\n"
        "app = FastAPI()\napp.include_router(router, prefix='/api')\n"
    )
    result = extract_provider_routes(
        {"app/main.py": main, "app/parent.py": parent, "app/child.py": child}
    )
    assert result.unresolved == []
    assert sorted((r.path, r.framework) for r in result.routes) == [
        ("/api/parent/inc/child/c", "fastapi"),
        ("/api/parent/p", "fastapi"),
    ]
