# Operation ID Plan: coding-analysis

## Files to Edit

- `src/main.py` (app-level routes: `/health`, `/`)
- `src/app/api/endpoints/analysis_modules/analysis_endpoints.py` (prefix `/api/analysis`)
- `src/app/api/endpoints/analysis_modules/batch_analysis.py` (prefix `/api/analysis`)
- `src/app/api/endpoints/analysis_modules/capabilities.py` (prefix `/api/analysis`)
- `src/app/api/endpoints/analysis_modules/multi_language_analysis.py` (prefix `/api/analysis`)
- `src/app/api/endpoints/semantic_analysis.py` (prefix `/api/analysis/semantic`)
- `src/app/api/endpoints/static_analysis.py` (prefix `/api/analysis/static`)
- `src/app/api/endpoints/complexity.py` (prefix `/api/analysis/complexity`)
- `src/app/api/endpoints/architecture.py` (prefix `/api/analysis/architecture`)
- `src/app/api/endpoints/similarity_modules/similarity_endpoints.py` (prefix `/api/analysis/similarity`)
- `src/app/api/endpoints/repo_scanner_modules/scanner_endpoints.py` (prefix `/api/analysis/scan`)
- `src/app/api/endpoints/bug_hunter.py` (prefix `/api/bug-hunter`)
- `src/app/api/endpoints/quality_dashboard.py` (prefix `/api/quality/dashboard`)
- `src/app/api/endpoints/quality.py` (prefix `/api/quality`)
- `src/app/api/endpoints/contract_analysis.py` (prefix `/api/analysis/contracts`)
- `src/app/api/endpoints/visualization_modules/graph_visualization.py` (included via visualization router -- NOT mounted in main.py)
- `src/app/api/endpoints/visualization_modules/configuration.py` (included via visualization router -- NOT mounted in main.py)

**Note:** The visualization router is defined but NOT included in `main.py` `_include_routers()`. Routes from `graph_visualization.py` and `configuration.py` are therefore not reachable unless the visualization router is mounted elsewhere. They are still documented here for completeness.

## Mapping Table

### main.py (app-level routes, no prefix)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/health` | `health` | `health_check` |
| GET | `/` | `root` | `root` |

### analysis_modules/analysis_endpoints.py (prefix `/api/analysis`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/analysis/analyze` | `analyze_code` | `analyze_code` |

### analysis_modules/batch_analysis.py (prefix `/api/analysis`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/analysis/batch` | `analyze_files_batch` | `analyze_files_batch` |

### analysis_modules/capabilities.py (prefix `/api/analysis`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/analysis/capabilities` | `get_analysis_capabilities` | `get_analysis_capabilities` |
| GET | `/api/analysis/status/{analysis_id}` | `get_analysis_status` | `get_analysis_status` |

### analysis_modules/multi_language_analysis.py (prefix `/api/analysis`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/analysis/multi-language` | `analyze_multi_language_code` | `analyze_multi_language` |

### semantic_analysis.py (prefix `/api/analysis/semantic`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/analysis/semantic/analyze-codebase` | `analyze_codebase_semantics` | `analyze_codebase_semantics` |
| GET | `/api/analysis/semantic/capabilities` | `get_semantic_capabilities` | `get_semantic_capabilities` |
| GET | `/api/analysis/semantic/patterns` | `get_semantic_patterns` | `get_semantic_patterns` |

### static_analysis.py (prefix `/api/analysis/static`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/analysis/static/call-graph` | `generate_call_graph` | `generate_call_graph` |
| POST | `/api/analysis/static/signatures` | `analyze_signatures` | `analyze_signatures` |
| POST | `/api/analysis/static/advanced-validation` | `validate_code_changes` | `validate_code_changes` |
| POST | `/api/analysis/static/documentation` | `generate_documentation` | `generate_documentation` |

### complexity.py (prefix `/api/analysis/complexity`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/analysis/complexity/analyze-file` | `analyze_file_complexity` | `analyze_file_complexity` |
| POST | `/api/analysis/complexity/analyze-batch` | `analyze_batch_complexity` | `analyze_batch_complexity` |
| GET | `/api/analysis/complexity/capabilities` | `get_complexity_capabilities` | `get_complexity_capabilities` |

### architecture.py (prefix `/api/analysis/architecture`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/analysis/architecture/components/{container_name}` | `list_components` | `list_architecture_components` |
| GET | `/api/analysis/architecture/dependencies/{container_name}` | `get_dependencies` | `get_architecture_dependencies` |
| GET | `/api/analysis/architecture/layers/{container_name}` | `get_layers` | `get_architecture_layers` |
| GET | `/api/analysis/architecture/metrics/{container_name}` | `get_metrics` | `get_architecture_metrics` |
| GET | `/api/analysis/architecture/patterns/{container_name}` | `get_patterns` | `get_architecture_patterns` |
| GET | `/api/analysis/architecture/analysis/{container_name}` | `get_full_analysis` | `get_architecture_full_analysis` |
| POST | `/api/analysis/architecture/analyze/{container_name}` | `trigger_analysis` | `trigger_architecture_analysis` |
| GET | `/api/analysis/architecture/graph/{container_name}` | `get_architecture_graph` | `get_architecture_graph` |

### similarity_modules/similarity_endpoints.py (prefix `/api/analysis/similarity`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/analysis/similarity/analyze` | `analyze_similarity` | `analyze_similarity` |
| POST | `/api/analysis/similarity/detect-clones` | `detect_clones` | `detect_clones` |
| POST | `/api/analysis/similarity/cross-language-patterns` | `find_cross_language_patterns` | `find_cross_language_patterns` |
| GET | `/api/analysis/similarity/capabilities` | `get_similarity_capabilities` | `get_similarity_capabilities` |

### repo_scanner_modules/scanner_endpoints.py (prefix `/api/analysis/scan`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/analysis/scan/scan` | `scan_repository` | `scan_repository` |

### bug_hunter.py (prefix `/api/bug-hunter`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/bug-hunter/{container_name}/status` | `get_bug_hunter_status` | `get_bug_hunter_status` |
| POST | `/api/bug-hunter/{container_name}/start-hunt` | `start_bug_hunt` | `start_bug_hunt` |
| POST | `/api/bug-hunter/{container_name}/hunt` | `hunt_bugs` | `hunt_bugs` |

### contract_analysis.py (prefix `/api/analysis/contracts`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | `/api/analysis/contracts/scan` | `scan_contracts` | `scan_contracts` |
| GET | `/api/analysis/contracts/capabilities` | `scan_capabilities` | `get_contract_capabilities` |

### quality_dashboard.py (prefix `/api/quality/dashboard`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/quality/dashboard/container/{container_name}` | `get_container_quality_dashboard` | `get_container_quality_dashboard` |
| GET | `/api/quality/dashboard/overview` | `get_multi_container_quality_overview` | `get_quality_overview` |
| GET | `/api/quality/dashboard/container/{container_name}/trends` | `get_container_quality_trends` | `get_container_quality_trends` |
| GET | `/api/quality/dashboard/overview/{code_graph_id}` | `get_project_quality_overview` | `get_project_quality_overview` |
| GET | `/api/quality/dashboard/trends/{code_graph_id}` | `get_quality_trends` | `get_quality_trends` |
| GET | `/api/quality/dashboard/issues/{code_graph_id}` | `get_quality_issues` | `get_dashboard_quality_issues` |
| GET | `/api/quality/dashboard/tools/usage` | `get_tool_usage_statistics` | `get_tool_usage_statistics` |
| GET | `/api/quality/dashboard/summary` | `get_dashboard_summary` | `get_dashboard_summary` |
| GET | `/api/quality/dashboard/projects` | `get_projects_quality_status` | `get_projects_quality_status` |

### quality.py (prefix `/api/quality`)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | `/api/quality/languages` | `get_supported_languages` | `list_quality_languages` |
| GET | `/api/quality/extensions` | `get_supported_extensions` | `list_quality_extensions` |
| GET | `/api/quality/{language}/tools` | `get_language_tools` | `get_quality_language_tools` |
| POST | `/api/quality/analyze` | `trigger_quality_analysis` | `trigger_quality_analysis` |
| POST | `/api/quality/analyze-sync` | `run_quality_analysis_sync` | `run_quality_analysis_sync` |
| GET | `/api/quality/health` | `quality_health` | `quality_health_check` |
| GET | `/api/quality/info` | `quality_info` | `get_quality_info` |
| GET | `/api/quality/results/{task_id}` | `get_analysis_results` | `get_quality_results` |
| GET | `/api/quality/{container_name}/latest` | `get_latest_quality_results` | `get_latest_quality_results` |
| GET | `/api/quality/{container_name}/issues` | `get_quality_issues` | `get_quality_issues` |
| GET | `/api/quality/{container_name}/summary` | `get_quality_summary` | `get_quality_summary` |
| GET | `/api/quality/{container_name}/gate` | `evaluate_quality_gate` | `evaluate_quality_gate` |
| GET | `/api/quality/{container_name}/trend` | `get_quality_trend` | `get_quality_trend` |
| GET | `/api/quality/cache/stats` | `get_cache_statistics` | `get_quality_cache_stats` |
| DELETE | `/api/quality/cache/{container_name}` | `clear_codebase_cache` | `clear_quality_container_cache` |
| DELETE | `/api/quality/cache` | `clear_all_cache` | `clear_quality_all_cache` |
| GET | `/api/quality/{container_name}/cached` | `check_cached_result` | `check_quality_cached_result` |

### visualization_modules/graph_visualization.py (NOT mounted in main.py)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| POST | (unmounted) `/create-graph` | `create_dependency_visualization` | `create_dependency_visualization` |

### visualization_modules/configuration.py (NOT mounted in main.py)

| Method | Effective Path | Function | operation_id |
|--------|---------------|----------|--------------|
| GET | (unmounted) `/capabilities` | `get_visualization_capabilities` | `get_visualization_capabilities` |
| GET | (unmounted) `/examples` | `get_visualization_examples` | `get_visualization_examples` |

## Total: 65 mounted routes + 3 unmounted visualization routes = 68 routes

**Note:** `quality.py` also contains a class-internal `@router.post("/analyze")` at line 290 inside a `try` block that registers `analyze_code`. This appears to be a dynamically registered endpoint that wraps quality analysis within a class init. It conflicts with the `trigger_quality_analysis` POST on `/analyze` at line 463. Verify whether this duplicate registration is intentional before assigning operation_ids.

## Execution Steps

```bash
cd /Users/mini1/Movies/code_graph_helper/coding-analysis && git pull --rebase
```

1. Add `operation_id=` to every `@router.get/post/put/delete/patch(` decorator and `@app.get(` decorator in all files listed above
2. Validate uniqueness -- ensure no duplicate `operation_id` values across the entire app
3. Run pre-commit checks:
   ```bash
   pre-commit run --all-files
   ```
4. Run tests:
   ```bash
   pytest
   ```
5. Commit and push:
   ```bash
   git add -A && git commit -m "Add explicit operation_id to all route decorators" && git push
   ```
