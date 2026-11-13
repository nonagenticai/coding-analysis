# Contract Scanner Backport: Upgrade coding-analysis to Standalone Parity

## Context

`coding-analysis` has contract scanning at `POST /api/analysis/contracts/scan` but its implementation lags behind the standalone `contract_scanner_service`. The standalone has 4 features that coding-analysis is missing. This spec covers backporting those features to bring coding-analysis to full parity.

## Scope

4 targeted improvements, no architectural changes. The scanner orchestrator, enrichment, flow checks, API schemas, and all other parsers/checks are already identical between both codebases.

## Changes

### 1. Type Checking (checks/type_check.py)

**Current state**: 17-line stub that returns `[]`.

**Target state**: Full 3-phase type validation (253 lines):

- **Phase 1 - Payload model comparison**: When a client uses `payload.model_dump()` with a known Pydantic model, compare its fields against service request models. Flag type mismatches (e.g., client sends `str` for a field that expects `int`).
- **Phase 2 - Response type conflict detection**: When multiple response models define the same field with incompatible types, warn if the client reads that field (ambiguous type resolution).
- **Phase 3 - Heuristic field name validation**: Common field names (`count`, `page`, `limit`, `offset`, `size`, `timeout`, `enabled`, `active`) checked against expected types. A field named `count` with type `str` produces a warning.

Support functions: `_COMPATIBLE_GROUPS` (type compatibility groups), `_normalize_type`, `_base_type`, `_types_compatible`, `_find_request_models`, `_find_response_models`, `_build_field_map`, `_NAME_TYPE_HINTS`.

**Import adjustments**: `from src.checks.types` becomes `from .types`; `from src.parsers.types` becomes `from ..parsers.types`.

**Integration**: scanner.py already calls `check_types()` at line 92. Replacing the stub enables type checking automatically.

### 2. Wrapper Client Detection (parsers/client_parser.py)

**Current state**: Detects `self.get()`, `self.post()`, `self._request()`, `self.submit_job()` patterns. Does NOT detect factory-created wrapper clients.

**Target state**: Additionally detect:

- **Factory pattern**: `client = _get_http_client()` or `client = await _get_client()` followed by `client.scan(json={...})` is recognized as a POST call. Without `json=` kwarg, inferred as GET.
- **Pseudo-path**: Wrapper calls get path `@wrapper/method_name` since the actual path is determined at runtime inside the wrapper.

New constants/functions:
- `_CLIENT_FACTORY_PATTERNS`: `{"_get_http_client", "_get_client"}`
- `_find_wrapper_client_vars(node)`: Finds variables assigned from factory functions
- `_walk_for_wrapper_client_vars(node, result)`: Recursive walker
- `_walk_for_wrapper_method_calls(node, caller, func_body, wrapper_vars, calls, func_node)`: Detects method calls on wrapper variables

### 3. Proxy Forwarding Detection (parsers/client_parser.py)

**Current state**: No proxy pattern detection.

**Target state**: Detect KAG gateway-style proxy forwarding:
```python
proxy.forward_request(path="/api/artifacts", method="POST", data={"key": "val"})
```

New constants/functions:
- `_PROXY_FORWARD_METHODS`: `{"forward_request", "forward_request_raw"}`
- `_extract_kwarg_string(call_node, kwarg_name)`: Extract string value of a named kwarg
- `_walk_for_proxy_calls(node, caller, func_body, calls)`: Detects proxy forwarding patterns
- `_extract_data_payload_fields(call_node, func_body)`: Extract payload from `data=` kwarg
- `_has_kwarg(call_node, kwarg_name)`: Generic kwarg presence check

### 4. Wildcard Path Matching (checks/route_check.py)

**Current state**: Exact normalized path matching only. F-string interpolated paths like `/api/{param}/health` that don't exactly match any route produce false-positive FAIL violations.

**Target state**:
- `_wildcard_match(client_path, route_map)`: Converts `{param}` segments to regex, matches against concrete route keys
- `@wrapper/` path skip: Wrapper calls with pseudo-paths are excluded from route checking
- Fallback chain: try exact match first, then wildcard match, then fail

## Coupling

Changes 2+3 (wrapper/proxy detection) and Change 4 (wildcard matching + @wrapper skip) are coupled. Wrapper detection produces `@wrapper/method_name` pseudo-paths. The route check must skip these to avoid false positives. **Deploy together.**

## Files Modified

| File | Action |
|------|--------|
| `src/processing/contract_analysis/checks/type_check.py` | Replace stub with full implementation |
| `src/processing/contract_analysis/parsers/client_parser.py` | Add wrapper + proxy detection functions; modify `_find_calls_in_function()` |
| `src/processing/contract_analysis/checks/route_check.py` | Add `import re`, `_wildcard_match()`, `@wrapper/` skip |

## Files NOT Modified

- `scanner.py` - already calls `check_types()`, no changes needed
- `parsers/types.py` - already has `ModelField`, `PydanticModel`
- `checks/types.py` - already has `ViolationSeverity`, all `RootCause` variants
- `enrichment/` - already identical to standalone
- `flow/` - already identical to standalone
- API schemas - no request/response model changes

## Testing

Fresh tests following coding-analysis conventions (bare functions, `from processing.contract_analysis.X import Y` imports):

| File | Tests |
|------|-------|
| `tests/unit/contract_analysis/test_type_check.py` | Expand from 2 to 16 tests covering all 3 phases + helpers |
| `tests/unit/contract_analysis/test_wrapper_client_detection.py` | 6 new tests (factory detection, method inference, payload extraction) |
| `tests/unit/contract_analysis/test_proxy_forwarding.py` | 6 new tests (forward_request, data extraction, dedup) |
| `tests/unit/contract_analysis/test_route_check.py` | 5 additional tests (wildcard match, @wrapper skip, f-string paths) |

## Verification

```bash
cd /Users/mini1/Movies/coding/coding-analysis
uv run ruff check src/processing/contract_analysis/
uv run ruff format --check src/processing/contract_analysis/
uv run ty check
uv run pytest tests/unit/contract_analysis/ -v
uv run pytest -v  # full suite
```
