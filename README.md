# coding-analysis

Multi-language code analysis engine for the Code Graph Helper platform. Performs static analysis, complexity metrics, architecture extraction, semantic cross-language relationship detection, quality scanning with language-specific tools, similarity/clone detection, bug hunting with auto-fix generation, and API contract validation -- all operating on codebases inside provisioned Docker containers.

## What It Does

The Code Graph Helper system provisions Docker containers that hold cloned repositories. This service is the analysis engine that inspects code inside those containers. It answers questions like: What are the architectural components? Where are the complexity hotspots? Are there duplicated functions across languages? What security vulnerabilities exist? It produces structured data consumed by the frontend dashboard and by n8n automation workflows.

Supports 9 languages via tree-sitter: Python, JavaScript, TypeScript, Java, Go, Rust, C, C++, C#.

## Authentication

Istio-forwarded headers with trusted-network bypass. No Keycloak dependency at the application level.

**Auth middleware** (`src/auth/middleware.py`):
1. **Public paths** (no auth): `/`, `/health`, `/docs`, `/redoc`, `/openapi.json`, `/metrics`
2. **Dev/test mode** (`ENVIRONMENT=development|testing`): Bearer token matching `DEV_API_KEY` or `TEST_API_KEY` env vars
3. **Istio headers**: `X-User-Id`, `X-Tenant-Id`, `X-User-Roles`, `X-User-Name`, `X-User-Email`, `X-Organization-Id`
4. **Trusted network bypass**: RFC1918 + loopback IPs get default tenant context (`DEFAULT_TENANT_ID`)
5. **Untrusted network without headers**: 401

**Tenant middleware** (`src/auth/tenant.py`):
- Extracts `X-Tenant-Id` header (set by Istio after license validation)
- Fallback: license token introspection via `POST http://licenses.licenses.svc.cluster.local:8544/license/introspect`
- Fallback: user state from auth middleware
- Fallback: trusted network bypass with `DEFAULT_TENANT_ID`

## API Routes

All routes are prefixed with `/api`. Port 8688.

### Health & Root

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Liveness/readiness probe (includes Redis status) |

### Analysis (Unified) -- `/api/analysis`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze` | Unified code analysis (types: basic, augmented, ast, deep, security, performance) |
| POST | `/batch` | Batch analysis of multiple files in a container |
| GET | `/capabilities` | Supported analysis types, languages, features |
| GET | `/status/{analysis_id}` | Track long-running analysis progress |
| POST | `/multi-language` | Multi-language analysis across 9 languages |

### Semantic Analysis -- `/api/analysis/semantic`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze-codebase` | Cross-language relationship detection (FFI, API, data flow, type system) |
| GET | `/capabilities` | Supported relationship types, languages, features |
| GET | `/patterns` | Browse detection patterns (filter: `pattern_type` query param) |

### Static Analysis -- `/api/analysis/static`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/call-graph` | Generate call graphs (JSON, DOT, SVG) |
| POST | `/signatures` | Extract function signatures with type info |
| POST | `/advanced-validation` | Symbolic execution, taint analysis, formal verification |
| POST | `/documentation` | Generate architecture docs, API refs, change summaries |

### Complexity Analysis -- `/api/analysis/complexity`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze-file` | Single file complexity (cyclomatic, cognitive, Halstead, maintainability) |
| POST | `/analyze-batch` | Batch complexity analysis with quality threshold flagging |
| GET | `/capabilities` | Supported languages, metrics, language-specific features |

### Architecture Analysis -- `/api/analysis/architecture`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/components/{container_name}` | List architectural components |
| GET | `/dependencies/{container_name}` | Component dependency mappings |
| GET | `/layers/{container_name}` | Layer assignments (presentation, business, data) |
| GET | `/metrics/{container_name}` | Instability, abstractness, distance from main sequence |
| GET | `/patterns/{container_name}` | Detected design patterns (singleton, factory, observer) |
| GET | `/analysis/{container_name}` | Full architecture analysis result |
| POST | `/analyze/{container_name}` | Trigger architecture analysis (optional `force_rebuild`) |
| GET | `/graph/{container_name}` | D3.js force-directed graph data |

### Similarity Detection -- `/api/analysis/similarity`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze` | Similarity analysis (scopes: file, function, class, block) |
| POST | `/detect-clones` | Clone detection (exact, near, structural, semantic) |
| POST | `/cross-language-patterns` | Cross-language pattern matching |
| GET | `/capabilities` | Supported similarity types and algorithms |

### Repository Scanner -- `/api/analysis/scan`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/scan` | Scan container repo (actions: `dependencies`, `imports`, `validation`) |

### Structure -- `/api/analysis/structure`

Computed facts of one workspace repo (`repo`) or all of them (`all_repos`), for porting N Python repos to one Rust repo.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/module-graph` | Internal module graph (Python or Rust) with SCCs and leaves-first porting layers |
| POST | `/api-surface` | Public API surface with signatures; macro-generated Rust items listed as unresolved |
| POST | `/tables` | Database tables defined / altered / read / written (SQLAlchemy, Alembic, raw SQL, sqlx, `*.sql`) |
| POST | `/entrypoints` | Non-HTTP entrypoints: lifespan/startup tasks, workers, consumers, schedulers, CLIs, websockets, SSE |
| POST | `/service-graph` | Cross-repo HTTP call graph |

### Contract Analysis -- `/api/analysis/contracts`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/scan` | Scan contract pairs for API violations (routes, methods, fields, types); route sources may be Python (FastAPI) or TypeScript/JavaScript (method-first `router.add`, Express/Fastify/Hono, `url.pathname` dispatch, cross-file `use()` mounts) |
| POST | `/call-sites` | Consumer side: outbound HTTP/event call sites in a workspace or inline sources (Python, JS, TS, Svelte, Vue), targets normalised to join `/scan`'s routes |
| POST | `/routes` | Provider side, listed: HTTP routes of a workspace or inline sources (FastAPI incl. cross-file `include_router` prefixes; Flask incl. `register_blueprint` prefixes, which replace the blueprint's own; axum; TS/JS routers and mounts), paths normalised like `/call-sites` targets; the endpoint description documents the join rule |
| GET | `/capabilities` | Available contract checks |

`/call-sites`, `/routes` and `/scan`'s TypeScript routes parse tenant source in a separate worker process per request, SIGKILLed at `CONTRACT_EXTRACTION_DEADLINE_SECONDS` (default 60; `/call-sites` then returns the files it finished with `truncation_reason: deadline`, `/routes` returns 422). Workers are capped by `CONTRACT_EXTRACTION_MEMORY_MB` (RLIMIT_AS, default 768) and `CONTRACT_EXTRACTION_MAX_WORKERS` (default 2), at most `CONTRACT_EXTRACTION_MAX_PER_TENANT` (default 1) of them for one tenant. A job waits for a worker at most `CONTRACT_EXTRACTION_QUEUE_WAIT_SECONDS` (default 10) and is never started with less than `CONTRACT_EXTRACTION_MIN_RUN_SECONDS` (default 5, at most half the deadline) left; at most `CONTRACT_EXTRACTION_MAX_PENDING` (default 4) extraction requests are in flight (running + waiting), checked before any file is read. Otherwise the request is answered `503` with `Retry-After` (busy: nothing was extracted). Coverage is exact: every loaded file is counted in `files_scanned` (extracted without error), or listed in `files_unparsed` (attempted; reason in `files_unparsed_reasons`, including the file in progress at a deadline) or `files_not_reached` (never attempted: after a deadline or the `max_call_sites` early stop); the lists are capped at 100, so use `files_unparsed_count` / `files_not_reached_count`. A client that disconnects does not stop its worker: it runs until it finishes or the deadline kills it. Any entrypoint that runs extraction must guard `if __name__ == "__main__":` (the forkserver preloads `__main__`); startup runs a worker self-check and fails if no worker can start. Inputs: at most 1 MiB per file (larger workspace files are listed in `files_skipped_large`; larger inline files are rejected 422) and 20 MiB per request (workspace reads stop with `truncation_reason: max_bytes`; inline bodies over it are rejected 422). The limits are echoed in each response's `limits`.

### Bug Hunter -- `/api/bug-hunter`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/{container_name}/hunt` | Queue a hunt and return `202` with a `hunt_id` and result links |
| GET | `/{container_name}/hunts/{hunt_id}/status` | Poll the queued/running/completed/failed job |
| GET | `/{container_name}/hunts/{hunt_id}/report` | Paginated summary of flagged file paths and file names |
| GET | `/{container_name}/hunts/{hunt_id}/findings?path=src/api` | Detailed bugs for a file or directory subtree |

An empty JSON body scans the repository. The findings `path` is relative to
the repository root; directory matches include all nested subdirectories.
PostgreSQL-backed workers use expiring leases, so interrupted jobs can be
reclaimed by another service instance after a restart.

### Quality Analysis -- `/api/quality`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/languages` | Supported languages and tools |
| GET | `/{language}/tools` | Tools for a specific language |
| POST | `/analyze` | Trigger async quality scan (returns task ID) |
| GET | `/results/{task_id}` | Get scan results by task ID |
| GET | `/{container_name}/latest` | Latest quality results for container |
| GET | `/{container_name}/issues` | Quality issues (filter: severity, category, fixable; paginated) |
| GET | `/{container_name}/summary` | Quality scores, grades, issue counts |
| GET | `/{container_name}/gate` | Quality gate pass/fail evaluation |
| GET | `/{container_name}/trend` | Quality score trend over time |
| GET | `/cache/stats` | Cache statistics |
| DELETE | `/cache/{container_name}` | Clear cache for container |
| DELETE | `/cache` | Clear all quality cache |
| GET | `/{container_name}/cached` | Get cached quality results |

### Quality Dashboard -- `/api/quality/dashboard`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/container/{container_name}` | Comprehensive quality dashboard for container |
| GET | `/overview` | Multi-container quality overview with rankings |
| GET | `/container/{container_name}/trends` | Quality trends over time (`days` param, default 30) |
| GET | `/overview/{code_graph_id}` | Legacy: project quality overview by graph ID |
| GET | `/trends/{code_graph_id}` | Legacy: quality trends by graph ID |
| GET | `/issues/{code_graph_id}` | Legacy: quality issues by graph ID (filter: language, severity, category) |
| GET | `/tools/usage` | Tool usage statistics (filter: language) |
| GET | `/summary` | Global dashboard summary |
| GET | `/projects` | All projects quality status (filter: `status_filter`) |

## Configuration

Environment variables (all optional, with defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_NAME` | `coding-analysis` | Service identity |
| `VERSION` | `1.0.0` | Reported version |
| `ENVIRONMENT` | `production` | `production` or `development` (enables docs in dev) |
| `PORT` | `8688` | Listen port |
| `DATABASE_URL` | none | PostgreSQL connection string (via `coding-client` pool) |
| `PGBOUNCER_URL` | none | PgBouncer connection string |
| `REDIS_URL` | `redis://code-graph-redis:6379/0` | Redis for caching and status tracking |
| `ENABLE_AUTH` | `true` | Enable auth + tenant middleware |
| `DEFAULT_TENANT_ID` | *(unset)* | Fallback tenant for the trusted-network / dev-key bypass. Must be a tenant **uuid**: `tenant_id` is a native `uuid` column (migration 0005), so a label like the old `n8n-dev` default is rejected by postgres. Unset = the bypass grants nothing. |
| `ENABLE_DOCS` | `false` | Force-enable `/docs` and `/redoc` in production |
| `LOG_LEVEL` | `INFO` | Python log level |
| `LOG_FORMAT` | `json` | `json` or `text` |
| `CORS_ORIGINS` | `["*"]` | Allowed CORS origins (via Istio, not Caddy) |
| `OTLP_ENDPOINT` | none | OpenTelemetry collector endpoint |
| `OTLP_INSECURE` | `true` | Use insecure OTLP connection |
| `OTLP_SERVICE_NAME` | `coding-analysis` | OTLP service name |
| `CODING_CONTAINERS_URL` | `http://coding-containers:8691` | URL of coding-containers service |

## Service Dependencies

- **coding-containers** (port 8691) -- manages Docker containers holding cloned repos. This service executes analysis inside those containers via httpx client (`src/clients/coding_containers.py`).
- **PostgreSQL** (`postgres:5432`) -- stores code graph data, quality results, project metadata (via `coding-client` connection pool).
- **PgBouncer** (`pgbouncer:6432`) -- connection pooler for PostgreSQL.
- **Redis** -- caching analysis results, tracking async analysis status, storing temporary state.
- **licenses** (`licenses.licenses.svc.cluster.local:8544`) -- license token introspection for tenant resolution fallback.
- **coding-client** (Python package) -- shared library providing database connection pooling, auth middleware, and common utilities.
- **coding-gateway** (port 8687) -- API gateway that routes external requests to this service.

## Project Structure

```
src/
  main.py                    FastAPI app, lifespan, router registration
  config/
    settings.py              Pydantic Settings (env vars)
  auth/
    middleware.py             Istio header auth + trusted network bypass
    tenant.py                Tenant isolation + license introspection
    models.py                User model
    rbac.py                  Role-based access control
    dependencies.py          FastAPI auth dependencies
  app/
    api/
      endpoints/             FastAPI routers (12 router modules)
        analysis.py          Unified analysis (delegates to analysis_modules/)
        semantic_analysis.py Cross-language semantic analysis
        static_analysis.py   Call graph, signatures, validation, docs
        complexity.py        Cyclomatic/cognitive complexity
        architecture.py      Component, layer, pattern, metric extraction
        similarity.py        Clone detection (delegates to similarity_modules/)
        repo_scanner.py      Dependency/import/validation scanning
        contract_analysis.py API contract validation
        bug_hunter.py        Error parsing, root-cause, fix generation
        quality.py           Multi-language quality scanning
      models/                Pydantic request/response models
      responses.py           Standard error/success response helpers
      utils/                 Container validation helpers
    core/                    Config, container manager, Redis service, dependencies
  clients/
    coding_containers.py     httpx client for coding-containers service
  models/                    Domain models (bugs, architecture, contracts, etc.)
  processing/                Analysis engines
    analyzers/               Complexity, call graph, symbolic analyzers
    architecture/            Architecture extraction and graph adapters
    ast_analysis/            AST-based class/function/API surface analysis
    batch_processing/        Directory scanning, parallel execution
    contract_analysis/       Client-service contract scanning
    languages/               Language-specific configs (Go, Java, Python, Rust, C++, C#, JS)
    parsers/                 Tree-sitter parser wrappers
    quality_analysis/        Tool definitions, trend analysis, fix formatting
    semantic_analysis/       Cross-language relationship detection
    structure/               Module graph, API surface, tables, entrypoints, service graph
    test_execution/          Test result processing, hypothesis management
  scoring/                   Risk calculation, recommendation generation
  services/                  Business logic orchestration
    bug_analysis.py          Bug analysis service
    bug_hunter.py            Bug hunting orchestrator
    complexity_analyzer.py   Complexity analysis service
    quality/                 Quality analysis service
    code_graph/              Graph service and examples
    incremental/             Incremental analysis support
```

## Development

```bash
uv sync                                          # install dependencies
ENVIRONMENT=development uv run uvicorn main:app --app-dir src --port 8688 --reload
uv run pytest                                    # run tests
uv run ruff check src/                           # lint
uv run ruff format src/                          # format
uv run ty check src/                             # type check
```

Requires Python 3.14+. Dev tools: **ruff** (lint + format), **ty** (type checking).

## CI/CD

GitHub Actions workflow: `ruff + ty → build amd64 → smoke test → push`

Image pushed to `192.168.1.114:5000/coding-analysis:sha-<commit>`, promoted to `:latest` on success. Flux auto-deploys to K3s.
