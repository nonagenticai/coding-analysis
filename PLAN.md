# coding-analysis — Remaining Work

Port: 8688 | 372 files | ~130K lines extracted

---

## P0 — Blocking Startup ✅

### 1. ✅ Lifespan initialization (`src/main.py`)
- Pydantic Settings in `src/config/settings.py`
- DB init via `coding_client.db.postgres_connection.init_postgres_pool()`
- Redis via `redis.asyncio` with ping check + graceful degradation
- Cleanup on shutdown

### 2. ✅ Auth middleware (`src/main.py`)
- `AuthMiddleware` + `TenantMiddleware` from coding-client
- Conditional on `settings.ENABLE_AUTH`
- ImportError fallback for running without coding-client

---

## P1 — Core Functionality Gaps

### 3. ✅ Tool runners → HTTP to coding-containers
- HTTP client in `src/clients/coding_containers.py` (singleton, httpx)
- Shared `_execute_tool_in_container()` helper in `tool_execution.py`
- All 22 tool runners implemented with per-tool output parsing
- `run_tool()` made async; callers in quality_analyzer.py, quality.py, architecture_processor.py updated

### 4. ✅ Language processor tree-sitter stubs
Tree-sitter processor classes implemented for all 6 languages (JS, Go, Java, Rust, C++, C#).

### 5. ✅ Python processor stubs (`src/processing/python_processor.py`)
- `_extract_basic_relationships()` — inheritance, composition, uses detection via AST
- `_extract_function_calls()` — caller/callee resolution with parent tracking
- `_extract_imports()` — absolute/relative import handling incl. star imports

---

## P2 — Production Readiness

### 6. ✅ Alembic migrations — removed
No ORM models exist in this service; alembic scaffold deleted.

### 7. ✅ Test coverage (`tests/`)
16 test files, 449 tests passing:
- Infrastructure: settings, containers client, container manager, main app
- Processing: python parser, entity extractor, relation extractor, python processor
- Services: complexity analyzer, analysis aggregator, quality analyzer, tool execution
- Endpoints: complexity, quality, bug hunter, analysis

### 8. ✅ Container manager HTTP client (`src/app/core/container_manager.py`)
- Delegates to `CodingContainersClient` from `src/clients/coding_containers.py`
- `get_container_files()` and `read_container_file()` implemented

---

## P3 — Cleanup

### 9. ✅ Verify deleted files
Confirmed none exist: `temp_analysis.py`, `config_legacy.py`, `analysis_error_handling_patch.py`.

---

## P4 — Service Implementations ✅

### 10. ✅ Service stubs implemented (8 services)
- `docker_service.py` — wraps CodingContainersClient for container lifecycle
- `error_parser.py` — parses Python/JS/Go error logs into ParsedError
- `fix_generator.py` — generates CodeFix suggestions per error type
- `github_integration_service.py` — GitHub API client for PR creation
- `github_provisioning_service.py` — repo cloning via containers client
- `dependency_parser.py` — parses requirements.txt/pyproject.toml
- `semantic_analysis_service.py` — AST-based code entity extraction
- `similarity_detector.py` — hash/structural/token similarity detection

### 11. ✅ GitHub Actions CI workflow
- `.github/workflows/coding-analysis-ci.yaml` — lint, test, build, push, auto-format

---

## DB Tables Owned
Node, Edge, Codebase, ArchitectureComponent, ArchitectureMetric, DetectedPattern, QualityResult, AnalysisSession

## Dependencies
- coding-client (shared models, auth, DB)
- coding-containers:8691 (tool execution via HTTP)
- tree-sitter + all language grammars
- scikit-learn, numpy, networkx
