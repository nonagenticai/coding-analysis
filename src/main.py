"""FastAPI application entry point for coding-analysis."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import redis.asyncio as aioredis
from coding_client.api.logging_setup import configure_logging
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings

configure_logging()

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Redis connection (module-level, initialised in lifespan)
# ---------------------------------------------------------------------------
_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Return the shared Redis connection (available after startup)."""
    if _redis is None:
        raise RuntimeError("Redis is not initialised -- called before lifespan startup?")
    return _redis


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup / shutdown events."""
    global _redis  # noqa: PLW0603
    bug_hunt_workers = []

    logger.info(
        "Starting %s v%s (%s) on port %s",
        settings.SERVICE_NAME,
        settings.VERSION,
        settings.ENVIRONMENT,
        settings.PORT,
    )

    # ── Database ──────────────────────────────────────────────────────
    # coding-analysis owns its graph state in the analysis-owned Postgres
    # schema; the request-path pool is opened lazily by ``db.session`` at
    # first use, not pinned in the lifespan. The previous SDK
    # ``init_postgres_pool`` call required a settings-provider hook this
    # service never wires, so every startup logged "Settings provider not
    # configured". Removed rather than papering over with a no-op hook.

    # ── Redis ─────────────────────────────────────────────────────────
    try:
        _redis = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
        )
        await _redis.ping()  # type: ignore[invalid-await]
        logger.info("Redis connected: %s", settings.REDIS_URL)
    except Exception as exc:
        logger.warning("Redis connection failed (service will run without cache): %s", exc)
        _redis = None

    # ── Contract extraction workers ───────────────────────────────────
    # Fail the deploy, not every /contracts request, when the extraction
    # forkserver cannot start (see processing/contract_analysis/isolation.py).
    from processing.contract_analysis import isolation

    await isolation.self_check()

    # ── Durable Bug Hunter workers ────────────────────────────────────
    # Jobs are claimed from PostgreSQL with expiring leases. A rolling restart
    # therefore reclaims interrupted hunts instead of leaving them stuck.
    if settings.DATABASE_URL or settings.PGBOUNCER_URL:
        from app.api.endpoints.bug_hunter import start_bug_hunt_workers

        bug_hunt_workers = start_bug_hunt_workers()
        logger.info("Started %d Bug Hunter workers", len(bug_hunt_workers))
    else:
        logger.warning("Bug Hunter workers disabled: no PostgreSQL URL configured")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────
    if bug_hunt_workers:
        from app.api.endpoints.bug_hunter import stop_bug_hunt_workers

        await stop_bug_hunt_workers(bug_hunt_workers)
    if _redis is not None:
        await _redis.aclose()
        logger.info("Redis connection closed")
    logger.info("Shutting down %s", settings.SERVICE_NAME)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.SERVICE_NAME,
    description="""\
Multi-language code analysis engine for the Code Graph Helper platform.
Operates on codebases inside provisioned Docker containers (via **coding-containers**).
Supports **9 languages** via tree-sitter: Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, C#.

### Authentication

Istio-forwarded headers (`X-User-Id`, `X-Tenant-Id`, `X-User-Roles`). License
token introspection as fallback (`Authorization: Bearer <token>`). Public
paths: `/`, `/health`, `/docs`, `/redoc`.

No source IP is trusted on network position alone. The trusted-network bypass
is opt-in via `TRUSTED_CIDRS` and defaults to empty; it must never be widened
to cover the cluster pod CIDR.

### Endpoint groups

| Prefix | Tag | Routes |
|--------|-----|--------|
| `/api/analysis` | Analysis | `/analyze`, `/batch`, `/capabilities`, `/status/{id}`, `/multi-language` |
| `/api/analysis/semantic` | Semantic | `/analyze-codebase`, `/capabilities`, `/patterns` |
| `/api/analysis/static` | Static | `/call-graph`, `/signatures`, `/advanced-validation`, `/documentation` |
| `/api/analysis/complexity` | Complexity | `/analyze-file`, `/analyze-batch`, `/capabilities` |
| `/api/analysis/architecture` | Architecture | `/components/{c}`, `/dependencies/{c}`, `/layers/{c}`, `/metrics/{c}`, `/patterns/{c}`, `/analysis/{c}`, `/analyze/{c}`, `/graph/{c}` |
| `/api/analysis/similarity` | Similarity | `/analyze`, `/detect-clones`, `/cross-language-patterns`, `/capabilities` |
| `/api/analysis/scan` | Repo Scanner | `/scan` |
| `/api/analysis/contracts` | Contracts | `/scan`, `/call-sites`, `/routes`, `/capabilities` |
| `/api/bug-hunter` | Bug Hunter | `/{c}/hunt`, `/{c}/hunts/{id}/status`, `/{c}/hunts/{id}/report`, `/{c}/hunts/{id}/findings?path=...`, `POST /{c}/hunts/{id}/findings/{fid}/recheck` |
| `/api/quality` | Quality | `/languages`, `/{lang}/tools`, `/analyze`, `/results/{id}`, `/{c}/latest`, `/{c}/issues`, `/{c}/summary`, `/{c}/gate`, `/{c}/trend`, `/cache/stats`, `/cache/{c}`, `/cache`, `/{c}/cached` |
| `/api/quality/dashboard` | Dashboard | `/container/{c}`, `/overview`, `/container/{c}/trends`, `/overview/{id}`, `/trends/{id}`, `/issues/{id}`, `/tools/usage`, `/summary`, `/projects` |
""",
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,  # type: ignore[invalid-argument-type]
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request logging middleware ─────────────────────────────────────────
from coding_client.api.request_logging import HealthCheckFilter, RequestLoggingMiddleware

logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())
app.add_middleware(RequestLoggingMiddleware)  # type: ignore[invalid-argument-type]


# ── Auth middleware (from coding-client) ──────────────────────────────
def install_auth_middleware(target_app: FastAPI) -> None:
    """Attach the auth + tenant middleware, or refuse to start.

    Fails CLOSED. This used to log a warning and then serve every route
    unauthenticated -- the loudest possible security failure reduced to a line
    nobody reads. That is a live risk, not a hypothetical: ``auth.tenant``
    re-exports ``TRUSTED_NETWORKS`` from coding_client, so a coding-client
    version that moves that name takes this branch and silently disables auth
    for the entire service.
    """
    try:
        from auth.middleware import AuthMiddleware
        from auth.tenant import TenantMiddleware
    except ImportError as exc:
        raise RuntimeError(
            "ENABLE_AUTH is set but the auth middleware could not be imported; "
            "refusing to start unauthenticated"
        ) from exc

    target_app.add_middleware(TenantMiddleware)  # type: ignore[invalid-argument-type]
    target_app.add_middleware(AuthMiddleware)  # type: ignore[invalid-argument-type]
    logger.info("Auth + Tenant middleware enabled")


if settings.ENABLE_AUTH:
    install_auth_middleware(app)

# --- Router registration ---
api_router = APIRouter(prefix="/api")


def _include_routers() -> None:
    """Lazy-import and register all analysis routers."""
    from app.api.endpoints.analysis import router as analysis_router
    from app.api.endpoints.architecture import router as architecture_router
    from app.api.endpoints.bug_hunter import router as bug_hunter_router
    from app.api.endpoints.complexity import router as complexity_router
    from app.api.endpoints.contract_analysis import router as contract_analysis_router
    from app.api.endpoints.dependency_snapshots import router as dependency_snapshots_router
    from app.api.endpoints.graph_metrics import router as graph_metrics_router
    from app.api.endpoints.graphs_reader import router as graphs_reader_router
    from app.api.endpoints.quality import router as quality_router
    from app.api.endpoints.repo_scanner import router as repo_scanner_router
    from app.api.endpoints.semantic_analysis import router as semantic_analysis_router
    from app.api.endpoints.similarity import router as similarity_router
    from app.api.endpoints.static_analysis import router as static_analysis_router
    from app.api.endpoints.structure import router as structure_router
    from app.api.endpoints.visualization import router as visualization_router

    api_router.include_router(analysis_router, prefix="/analysis", tags=["Analysis"])
    api_router.include_router(
        semantic_analysis_router, prefix="/analysis/semantic", tags=["Analysis - Semantic"]
    )
    api_router.include_router(
        static_analysis_router, prefix="/analysis/static", tags=["Analysis - Static"]
    )
    api_router.include_router(
        complexity_router, prefix="/analysis/complexity", tags=["Analysis - Complexity"]
    )
    api_router.include_router(
        architecture_router, prefix="/analysis/architecture", tags=["Analysis - Architecture"]
    )
    api_router.include_router(
        similarity_router, prefix="/analysis/similarity", tags=["Analysis - Similarity"]
    )
    api_router.include_router(
        repo_scanner_router, prefix="/analysis/scan", tags=["Analysis - Repository Scan"]
    )
    api_router.include_router(bug_hunter_router, prefix="/bug-hunter", tags=["Bug Hunter"])
    api_router.include_router(quality_router, prefix="/quality", tags=["Quality"])
    api_router.include_router(
        contract_analysis_router, prefix="/analysis/contracts", tags=["Analysis - Contracts"]
    )
    api_router.include_router(
        structure_router, prefix="/analysis/structure", tags=["Analysis - Structure"]
    )
    api_router.include_router(
        visualization_router, prefix="/analysis/visualization", tags=["Analysis - Visualization"]
    )
    api_router.include_router(
        graphs_reader_router, prefix="/analysis/graphs", tags=["Analysis - Graph Reader"]
    )
    api_router.include_router(
        graph_metrics_router, prefix="/analysis/graphs", tags=["Analysis - Graph Metrics"]
    )
    api_router.include_router(
        dependency_snapshots_router,
        prefix="/analysis/dependency-report",
        tags=["Analysis - Dependency Report"],
    )

    app.include_router(api_router)


_include_routers()

# ── OpenAPI security configuration ────────────────────────────────────
from coding_client.api.openapi_config import configure_openapi

configure_openapi(
    app,
    protected_prefixes={"/api/analysis", "/api/bug-hunter", "/api/quality"},
    public_endpoints={"/", "/health", "/docs", "/redoc", "/openapi.json"},
)


@app.get("/health", tags=["Health"], operation_id="health_check")
async def health():
    """Liveness / readiness probe."""
    redis_ok = False
    if _redis is not None:
        try:
            await _redis.ping()  # type: ignore[invalid-await]
            redis_ok = True
        except Exception:
            pass

    return {
        "status": "ok" if redis_ok else "degraded",
        "service": settings.SERVICE_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "redis": "connected" if redis_ok else "disconnected",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/", tags=["Root"], operation_id="root")
async def root():
    return {
        "service": settings.SERVICE_NAME,
        "version": settings.VERSION,
        "docs": "/docs" if settings.docs_enabled else None,
        "health": "/health",
    }


# ── Exception handlers (from coding-client KAG API standard) ──────────
from coding_client.api.exception_handlers import register_exception_handlers

register_exception_handlers(app, debug=getattr(settings, "DEBUG", False))

# Starlette 1.0 splits HTTPException into separate starlette/fastapi classes.
# register_exception_handlers only handles FastAPI's HTTPException, so route-level
# 404s (raised as starlette.exceptions.HTTPException) bypass it.  Re-register
# the same handler for the starlette base class.
from starlette.exceptions import HTTPException as _StarletteHTTPException

if _StarletteHTTPException not in app.exception_handlers or app.exception_handlers[
    _StarletteHTTPException
].__module__.startswith("fastapi"):
    from fastapi.exceptions import HTTPException as _FastAPIHTTPException

    _fastapi_handler = app.exception_handlers.get(_FastAPIHTTPException)
    if _fastapi_handler is not None:
        app.add_exception_handler(_StarletteHTTPException, _fastapi_handler)
