"""Comprehensive tests for contract_analysis parsers: client, schema, route.

Targets ~80%+ coverage of:
- src/processing/contract_analysis/parsers/client_parser.py
- src/processing/contract_analysis/parsers/schema_parser.py
- src/processing/contract_analysis/parsers/route_parser.py
"""

from __future__ import annotations

import pytest

from processing.contract_analysis.parsers.client_parser import (
    _detect_model_dump_var,
    _extract_dict_keys,
    _extract_json_payload_fields,
    _extract_query_params,
    _extract_string_value,
    _find_dict_literal_keys,
    _find_param_type,
    _get_positional_args,
    _has_json_kwarg,
    _node_contains,
    _node_text,
    _parse_client_call,
    _parse_explicit_http_call,
    _parse_generic_http_call,
    _parse_httpx_call,
    _resolve_payload_model,
    _strip_base_url,
    parse_client,
)
from processing.contract_analysis.parsers.route_parser import (
    _extract_query_params as route_extract_query_params,
)
from processing.contract_analysis.parsers.route_parser import (
    _extract_request_model,
    _extract_response_model,
    _extract_return_annotation,
    _extract_status_code,
    _extract_string_content,
    _find_all_router_prefixes,
    parse_routes,
)
from processing.contract_analysis.parsers.schema_parser import (
    _check_annotated_field,
    _extract_annotated_inner_type,
    _extract_fields_from_body,
    _extract_type_annotation,
    _get_base_classes,
    _has_model_validator,
    _is_field_required,
    _unwrap_optional,
    parse_schemas,
    resolve_inheritance,
)
from processing.contract_analysis.parsers.types import (
    ModelField,
    PydanticModel,
    ServiceSchema,
)

# ─── Helpers ────────────────────────────────────────────────────────────


def _parse_tree(source: str):
    """Parse Python source and return the root node."""
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    lang = Language(tspython.language())
    parser = Parser(lang)
    tree = parser.parse(source.encode())
    return tree.root_node


def _find_node_type(root, type_name: str):
    """Find first node of given type via DFS."""
    if root.type == type_name:
        return root
    for child in root.children:
        result = _find_node_type(child, type_name)
        if result is not None:
            return result
    return None


def _find_all_nodes(root, type_name: str):
    """Find all nodes of given type via DFS."""
    results = []
    if root.type == type_name:
        results.append(root)
    for child in root.children:
        results.extend(_find_all_nodes(child, type_name))
    return results


# ═══════════════════════════════════════════════════════════════════════
# CLIENT PARSER TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestNodeText:
    def test_extracts_text(self):
        root = _parse_tree("x = 1")
        ident = _find_node_type(root, "identifier")
        assert _node_text(ident) == "x"

    def test_empty_text_returns_empty(self):
        """When node.text is None, returns empty string."""

        class FakeNode:
            text = None

        assert _node_text(FakeNode()) == ""


class TestExtractStringValue:
    def test_simple_string(self):
        root = _parse_tree('x = "hello"')
        string_node = _find_node_type(root, "string")
        assert _extract_string_value(string_node) == "hello"

    def test_fstring_with_interpolation(self):
        root = _parse_tree('x = f"/api/{item_id}"')
        string_node = _find_node_type(root, "string")
        result = _extract_string_value(string_node)
        assert result is not None
        assert "/api/" in result
        assert "{item_id}" in result

    def test_non_string_node_returns_none(self):
        root = _parse_tree("x = 42")
        int_node = _find_node_type(root, "integer")
        assert _extract_string_value(int_node) is None

    def test_empty_string_returns_none(self):
        root = _parse_tree('x = ""')
        string_node = _find_node_type(root, "string")
        result = _extract_string_value(string_node)
        # Empty string: no string_content child, not fstring -> returns None
        assert result is None


class TestGetPositionalArgs:
    def test_with_positional_and_keyword(self):
        root = _parse_tree('foo("/path", key="val")')
        call_node = _find_node_type(root, "call")
        args = _get_positional_args(call_node)
        assert len(args) == 1

    def test_no_args(self):
        root = _parse_tree("foo()")
        call_node = _find_node_type(root, "call")
        args = _get_positional_args(call_node)
        assert len(args) == 0

    def test_multiple_positional(self):
        root = _parse_tree('foo("a", "b", "c")')
        call_node = _find_node_type(root, "call")
        args = _get_positional_args(call_node)
        assert len(args) == 3

    def test_dictionary_splat_excluded(self):
        root = _parse_tree('foo("/path", **kwargs)')
        call_node = _find_node_type(root, "call")
        args = _get_positional_args(call_node)
        assert len(args) == 1


class TestFindDictLiteralKeys:
    def test_finds_keys_from_dict_assignment(self):
        source = """
def foo():
    payload = {"name": "x", "age": 1}
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        keys = _find_dict_literal_keys(body, "payload")
        assert "name" in keys
        assert "age" in keys

    def test_finds_subscript_assignment_keys(self):
        source = """
def foo():
    payload = {}
    payload["name"] = "x"
    payload["age"] = 1
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        keys = _find_dict_literal_keys(body, "payload")
        assert "name" in keys
        assert "age" in keys

    def test_finds_keys_inside_if_blocks(self):
        source = """
def foo():
    payload = {"base": 1}
    if True:
        payload["extra"] = 2
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        keys = _find_dict_literal_keys(body, "payload")
        assert "base" in keys
        assert "extra" in keys

    def test_no_keys_for_unknown_var(self):
        source = """
def foo():
    payload = {"name": "x"}
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        keys = _find_dict_literal_keys(body, "other_var")
        assert keys == []


class TestStripBaseUrl:
    def test_path_starting_with_slash(self):
        assert _strip_base_url("/api/foo") == "/api/foo"

    def test_path_with_base_url_prefix(self):
        result = _strip_base_url("{self.base_url}/api/foo")
        assert result == "/api/foo"

    def test_path_with_internal_prefix(self):
        result = _strip_base_url("{self.base_url}/_internal/v1/test")
        assert result == "/_internal/v1/test"

    def test_path_without_api_prefix(self):
        result = _strip_base_url("http://localhost/other")
        assert result == "http://localhost/other"


class TestHasJsonKwarg:
    def test_with_json_kwarg(self):
        root = _parse_tree('foo("/path", json=data)')
        call = _find_node_type(root, "call")
        assert _has_json_kwarg(call) is True

    def test_without_json_kwarg(self):
        root = _parse_tree('foo("/path", data=data)')
        call = _find_node_type(root, "call")
        assert _has_json_kwarg(call) is False

    def test_no_args_at_all(self):
        root = _parse_tree("foo()")
        call = _find_node_type(root, "call")
        assert _has_json_kwarg(call) is False


class TestExtractDictKeys:
    def test_basic_dict(self):
        root = _parse_tree('x = {"a": 1, "b": 2}')
        dict_node = _find_node_type(root, "dictionary")
        keys = _extract_dict_keys(dict_node)
        assert keys == ["a", "b"]

    def test_empty_dict(self):
        root = _parse_tree("x = {}")
        dict_node = _find_node_type(root, "dictionary")
        keys = _extract_dict_keys(dict_node)
        assert keys == []


class TestExtractJsonPayloadFields:
    def test_inline_dict(self):
        source = """
def foo():
    bar(json={"name": "x", "val": 1})
"""
        root = _parse_tree(source)
        func_body = _find_node_type(root, "function_definition").child_by_field_name("body")
        call = _find_node_type(root, "call")
        fields = _extract_json_payload_fields(call, func_body)
        assert "name" in fields
        assert "val" in fields

    def test_variable_reference(self):
        source = """
def foo():
    data = {"key1": "a", "key2": "b"}
    bar(json=data)
"""
        root = _parse_tree(source)
        func_body = _find_node_type(root, "function_definition").child_by_field_name("body")
        calls = _find_all_nodes(root, "call")
        bar_call = [c for c in calls if "bar" in _node_text(c)][0]
        fields = _extract_json_payload_fields(bar_call, func_body)
        assert "key1" in fields
        assert "key2" in fields

    def test_no_json_kwarg(self):
        source = """
def foo():
    bar(data={"key": "val"})
"""
        root = _parse_tree(source)
        func_body = _find_node_type(root, "function_definition").child_by_field_name("body")
        call = _find_node_type(root, "call")
        fields = _extract_json_payload_fields(call, func_body)
        assert fields == []

    def test_no_arguments(self):
        source = """
def foo():
    bar()
"""
        root = _parse_tree(source)
        func_body = _find_node_type(root, "function_definition").child_by_field_name("body")
        call = _find_node_type(root, "call")
        fields = _extract_json_payload_fields(call, func_body)
        assert fields == []


class TestExtractQueryParams:
    def test_from_url_query_string(self):
        source = """
def foo():
    bar("/api/items?page=1&limit=20")
"""
        root = _parse_tree(source)
        func_body = _find_node_type(root, "function_definition").child_by_field_name("body")
        call = _find_node_type(root, "call")
        params = _extract_query_params(call, "/api/items?page=1&limit=20", func_body)
        assert "page" in params
        assert "limit" in params

    def test_from_params_kwarg_dict(self):
        source = """
def foo():
    bar("/api/items", params={"page": 1, "limit": 20})
"""
        root = _parse_tree(source)
        func_body = _find_node_type(root, "function_definition").child_by_field_name("body")
        call = _find_node_type(root, "call")
        params = _extract_query_params(call, "/api/items", func_body)
        assert "page" in params
        assert "limit" in params

    def test_from_params_kwarg_variable(self):
        source = """
def foo():
    qp = {"sort": "asc", "filter": "active"}
    bar("/api/items", params=qp)
"""
        root = _parse_tree(source)
        func_body = _find_node_type(root, "function_definition").child_by_field_name("body")
        calls = _find_all_nodes(root, "call")
        bar_call = [c for c in calls if "bar" in _node_text(c)][0]
        params = _extract_query_params(bar_call, "/api/items", func_body)
        assert "sort" in params
        assert "filter" in params

    def test_no_query_params(self):
        source = """
def foo():
    bar("/api/items")
"""
        root = _parse_tree(source)
        func_body = _find_node_type(root, "function_definition").child_by_field_name("body")
        call = _find_node_type(root, "call")
        params = _extract_query_params(call, "/api/items", func_body)
        assert params == []


class TestDetectModelDumpVar:
    def test_model_dump_pattern(self):
        root = _parse_tree("bar(json=payload.model_dump())")
        # The outer call is bar(...), need to find the right one
        calls = _find_all_nodes(root, "call")
        bar_call = calls[0]  # bar() is the outer call
        result = _detect_model_dump_var(bar_call)
        assert result == "payload"

    def test_dict_method_pattern(self):
        root = _parse_tree("bar(json=payload.dict())")
        calls = _find_all_nodes(root, "call")
        bar_call = calls[0]
        result = _detect_model_dump_var(bar_call)
        assert result == "payload"

    def test_no_model_dump(self):
        root = _parse_tree('bar(json={"key": "val"})')
        call = _find_node_type(root, "call")
        result = _detect_model_dump_var(call)
        assert result is None

    def test_no_json_kwarg(self):
        root = _parse_tree('bar(data="hello")')
        call = _find_node_type(root, "call")
        result = _detect_model_dump_var(call)
        assert result is None

    def test_no_args(self):
        root = _parse_tree("bar()")
        call = _find_node_type(root, "call")
        result = _detect_model_dump_var(call)
        assert result is None


class TestFindParamType:
    def test_typed_parameter(self):
        source = """
def create(self, payload: CreateRequest):
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = _find_param_type(func, "payload")
        assert result == "CreateRequest"

    def test_no_annotation(self):
        source = """
def create(self, payload):
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = _find_param_type(func, "payload")
        assert result is None

    def test_param_not_found(self):
        source = """
def create(self, name: str):
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = _find_param_type(func, "payload")
        assert result is None

    def test_typed_default_parameter(self):
        source = """
def create(self, payload: CreateRequest = None):
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = _find_param_type(func, "payload")
        assert result == "CreateRequest"


class TestResolvePayloadModel:
    def test_resolves_from_model_dump(self):
        source = """
def create(self, payload: CreateRequest):
    self._client.post("/api/foo", json=payload.model_dump())
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        calls = _find_all_nodes(root, "call")
        # Find the post() call
        post_call = [
            c
            for c in calls
            if "post" in _node_text(c) and "model_dump" not in _node_text(c).split("(")[0]
        ][0]
        result = _resolve_payload_model(post_call, func)
        assert result == "CreateRequest"

    def test_no_model_dump_returns_none(self):
        source = """
def create(self, data: dict):
    self._client.post("/api/foo", json=data)
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        calls = _find_all_nodes(root, "call")
        post_call = [c for c in calls if "post" in _node_text(c)][0]
        result = _resolve_payload_model(post_call, func)
        assert result is None


class TestNodeContains:
    def test_same_node(self):
        root = _parse_tree("x = 1")
        assert _node_contains(root, root) is True

    def test_child_contained(self):
        root = _parse_tree("x = 1")
        ident = _find_node_type(root, "identifier")
        assert _node_contains(root, ident) is True

    def test_unrelated_node_not_contained(self):
        root1 = _parse_tree("x = 1")
        root2 = _parse_tree("y = 2")
        ident2 = _find_node_type(root2, "identifier")
        assert _node_contains(root1, ident2) is False


class TestParseClientIntegration:
    """Integration tests for parse_client top-level function."""

    def test_empty_source(self):
        result = parse_client("")
        assert result.calls == []
        assert result.parse_errors == []

    def test_whitespace_only(self):
        result = parse_client("   \n\n  ")
        assert result.calls == []

    def test_submit_job_with_dict_literal_payload(self):
        source = """
class MyClient(BaseServiceClient):
    async def create_job(self, name):
        return await self.submit_job("/api/jobs", {"name": name, "type": "fast"})
"""
        result = parse_client(source)
        calls = [c for c in result.calls if c.method == "POST"]
        assert len(calls) == 1
        assert calls[0].path == "/api/jobs"
        assert "name" in calls[0].payload_fields
        assert "type" in calls[0].payload_fields

    def test_request_method_with_no_path(self):
        """_request with too few positional args produces no call."""
        source = """
class MyClient(BaseServiceClient):
    async def bad_call(self):
        return await self._request("GET")
"""
        result = parse_client(source)
        # Should not crash and should yield no valid calls
        request_calls = [c for c in result.calls if c.caller == "bad_call"]
        assert len(request_calls) == 0

    def test_poll_job_path_has_param_suffix(self):
        source = """
class MyClient(BaseServiceClient):
    async def wait(self, jid):
        return await self.poll_job("/api/tasks", jid, timeout=60)
"""
        result = parse_client(source)
        calls = [c for c in result.calls if c.method == "GET"]
        assert len(calls) == 1
        assert calls[0].path == "/api/tasks/{param}"

    def test_httpx_async_client_multiple_methods(self):
        source = """
class MyClient:
    async def do_stuff(self):
        async with httpx.AsyncClient() as client:
            await client.get("/api/health")
            await client.post("/api/data", json={"key": "val"})
            await client.put("/api/data/1", json={"key": "updated"})
            await client.delete("/api/data/1")
            await client.patch("/api/data/1", json={"flag": True})
"""
        result = parse_client(source)
        methods = sorted(c.method for c in result.calls)
        assert "DELETE" in methods
        assert "GET" in methods
        assert "PATCH" in methods
        assert "POST" in methods
        assert "PUT" in methods

    def test_generic_http_call_detection(self):
        """Detect obj.get/post patterns on non-httpx variables."""
        source = """
class MyClient:
    async def fetch(self):
        resp = await self._client.get("/api/resource")
        return resp.json()
"""
        result = parse_client(source)
        assert len(result.calls) >= 1
        assert any(c.path == "/api/resource" and c.method == "GET" for c in result.calls)

    def test_explicit_stream_call(self):
        """obj.stream("POST", "/path") pattern."""
        source = """
class MyClient:
    async def stream_data(self, sid):
        async with self._client.stream("POST", f"/api/stream/{sid}") as resp:
            pass
"""
        result = parse_client(source)
        post_calls = [c for c in result.calls if c.method == "POST"]
        assert len(post_calls) >= 1
        assert any("/api/stream/" in c.path for c in post_calls)

    def test_explicit_request_call(self):
        """obj.request("GET", "/path") pattern."""
        source = """
class MyClient:
    async def get_data(self):
        resp = await self._client.request("GET", "/api/stuff")
        return resp.json()
"""
        result = parse_client(source)
        get_calls = [c for c in result.calls if c.method == "GET"]
        assert len(get_calls) >= 1
        assert any(c.path == "/api/stuff" for c in get_calls)

    def test_no_false_positive_non_path_string(self):
        """obj.get("non_path_string") should be ignored (no / prefix)."""
        source = """
class MyService:
    async def get_val(self):
        return self.cache.get("some_key")
"""
        result = parse_client(source)
        assert len(result.calls) == 0

    def test_dedup_between_self_and_generic(self):
        """When self.submit_job detects a call AND generic catches it, no dupes."""
        source = """
class MyClient(BaseServiceClient):
    async def create(self, name):
        payload = {"name": name}
        return await self.submit_job("/api/things", payload)
"""
        result = parse_client(source)
        post_calls = [c for c in result.calls if c.method == "POST" and c.path == "/api/things"]
        assert len(post_calls) == 1

    def test_caller_field_populated(self):
        source = """
class MyClient(BaseServiceClient):
    async def my_method(self):
        return await self._request("GET", "/api/items")
"""
        result = parse_client(source)
        assert result.calls[0].caller == "my_method"

    def test_response_reads_via_get_and_subscript(self):
        source = """
class MyClient(BaseServiceClient):
    async def fetch(self, jid):
        data = await self._request("GET", f"/api/jobs/{jid}")
        status = data.get("status")
        name = data["name"]
        count = data.get("count", 0)
        return status, name, count
"""
        result = parse_client(source)
        assert len(result.calls) == 1
        reads = result.calls[0].response_reads
        assert "status" in reads
        assert "name" in reads
        assert "count" in reads

    def test_response_reads_no_duplicate(self):
        """Same field accessed twice should only appear once."""
        source = """
class MyClient(BaseServiceClient):
    async def fetch(self, jid):
        data = await self._request("GET", f"/api/jobs/{jid}")
        x = data.get("status")
        y = data["status"]
        return x, y
"""
        result = parse_client(source)
        assert result.calls[0].response_reads.count("status") == 1

    def test_submit_job_no_positional_args(self):
        """submit_job() with no args produces no call."""
        source = """
class MyClient(BaseServiceClient):
    async def bad(self):
        return await self.submit_job()
"""
        result = parse_client(source)
        assert len(result.calls) == 0

    def test_payload_opaque_when_submit_job_var_not_dict(self):
        """submit_job with a non-traceable variable payload is opaque."""
        source = """
class MyClient(BaseServiceClient):
    async def create(self, data):
        return await self.submit_job("/api/foo", data)
"""
        result = parse_client(source)
        call = result.calls[0]
        assert call.payload_opaque is True
        assert call.payload_fields == []

    def test_query_params_from_url_cleaned(self):
        """Query params in URL are extracted and path is cleaned."""
        source = """
class MyClient:
    async def list_items(self, page):
        return await self._client.get(f"/api/items?page={page}&limit=20")
"""
        result = parse_client(source)
        get_calls = [c for c in result.calls if c.method == "GET"]
        assert len(get_calls) >= 1
        call = get_calls[0]
        assert "?" not in call.path  # path should be cleaned
        assert "page" in call.query_params
        assert "limit" in call.query_params

    def test_httpx_no_path_returns_none(self):
        """httpx call without positional args yields nothing."""
        source = """
class MyClient:
    async def bad(self):
        async with httpx.AsyncClient() as client:
            await client.get()
"""
        result = parse_client(source)
        assert len(result.calls) == 0

    def test_function_without_body(self):
        """Function with just a pass and no calls."""
        source = """
class MyClient(BaseServiceClient):
    async def noop(self):
        pass
"""
        result = parse_client(source)
        assert len(result.calls) == 0

    def test_non_class_function(self):
        """Top-level function with self.submit_job still detected."""
        source = """
async def standalone(self):
    return await self._request("GET", "/api/standalone")
"""
        result = parse_client(source)
        assert len(result.calls) == 1
        assert result.calls[0].path == "/api/standalone"


class TestParseClientCallHelperMethod:
    """Tests for _parse_client_call with various method types."""

    def test_unknown_method_returns_none(self):
        source = """
def foo():
    self.unknown_method("/api/foo")
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        calls = _find_all_nodes(root, "call")
        for call in calls:
            result = _parse_client_call(call, "unknown_method", "foo", body)
            assert result is None


class TestParseGenericHttpCall:
    def test_non_path_string_ignored(self):
        """Non-slash-starting paths are rejected."""
        source = """
def foo():
    self.client.get("not_a_path")
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        call = _find_node_type(root, "call")
        result = _parse_generic_http_call(call, "get", "foo", body)
        assert result is None

    def test_no_positional_args(self):
        source = """
def foo():
    self.client.get()
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        call = _find_node_type(root, "call")
        result = _parse_generic_http_call(call, "get", "foo", body)
        assert result is None

    def test_with_query_params_in_path(self):
        source = """
def foo():
    self.client.get("/api/items?sort=asc")
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        call = _find_node_type(root, "call")
        result = _parse_generic_http_call(call, "get", "foo", body)
        assert result is not None
        assert result.path == "/api/items"
        assert "sort" in result.query_params

    def test_with_json_and_model_dump(self):
        source = """
def foo(self, payload: MyModel):
    self.client.post("/api/items", json=payload.model_dump())
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        call = _find_all_nodes(root, "call")[0]
        result = _parse_generic_http_call(call, "post", "foo", body, func)
        assert result is not None
        assert result.payload_model == "MyModel"


class TestParseExplicitHttpCall:
    def test_stream_pattern(self):
        source = """
def foo():
    self.client.stream("POST", "/api/stream")
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        call = _find_node_type(root, "call")
        result = _parse_explicit_http_call(call, "foo", body)
        assert result is not None
        assert result.method == "POST"
        assert result.path == "/api/stream"

    def test_too_few_args(self):
        source = """
def foo():
    self.client.stream("POST")
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        call = _find_node_type(root, "call")
        result = _parse_explicit_http_call(call, "foo", body)
        assert result is None

    def test_non_path_string_rejected(self):
        source = """
def foo():
    self.client.stream("POST", "not_a_path")
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        call = _find_node_type(root, "call")
        result = _parse_explicit_http_call(call, "foo", body)
        assert result is None

    def test_non_string_method_rejected(self):
        source = """
def foo():
    self.client.stream(method_var, "/api/path")
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        call = _find_node_type(root, "call")
        result = _parse_explicit_http_call(call, "foo", body)
        assert result is None

    def test_with_query_params(self):
        source = """
def foo():
    self.client.request("GET", "/api/items?page=1")
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        call = _find_node_type(root, "call")
        result = _parse_explicit_http_call(call, "foo", body)
        assert result is not None
        assert result.path == "/api/items"
        assert "page" in result.query_params

    def test_with_base_url_prefix(self):
        source = """
def foo():
    self.client.request("GET", f"{self.base_url}/api/items")
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        call = _find_node_type(root, "call")
        result = _parse_explicit_http_call(call, "foo", body)
        assert result is not None
        assert result.path == "/api/items"


class TestParseHttpxCall:
    def test_basic_get(self):
        source = """
def foo():
    client.get("/api/items")
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        call = _find_node_type(root, "call")
        result = _parse_httpx_call(call, "get", "foo", body)
        assert result is not None
        assert result.method == "GET"
        assert result.path == "/api/items"

    def test_no_positional_returns_none(self):
        source = """
def foo():
    client.get()
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        call = _find_node_type(root, "call")
        result = _parse_httpx_call(call, "get", "foo", body)
        assert result is None

    def test_non_string_path_returns_none(self):
        source = """
def foo():
    client.get(url_var)
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        body = func.child_by_field_name("body")
        call = _find_node_type(root, "call")
        result = _parse_httpx_call(call, "get", "foo", body)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# SCHEMA PARSER TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestSchemaParserNodeText:
    def test_basic(self):
        from processing.contract_analysis.parsers.schema_parser import _node_text as sp_node_text

        root = _parse_tree("x = 1")
        ident = _find_node_type(root, "identifier")
        assert sp_node_text(ident) == "x"


class TestUnwrapOptional:
    def test_optional_str(self):
        source = """
from pydantic import BaseModel
class M(BaseModel):
    x: Optional[str] = None
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        # Find the type node
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        for sub in child.children:
                            if sub.type == "type":
                                result = _unwrap_optional(sub)
                                assert result == "str"
                                return
        pytest.fail("Did not find type node")

    def test_non_optional_returns_none(self):
        source = """
from pydantic import BaseModel
class M(BaseModel):
    x: str
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        for sub in child.children:
                            if sub.type == "type":
                                result = _unwrap_optional(sub)
                                assert result is None
                                return
        pytest.fail("Did not find type node")


class TestExtractTypeAnnotation:
    def test_simple_type(self):
        source = """
from pydantic import BaseModel
class M(BaseModel):
    x: int
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        for sub in child.children:
                            if sub.type == "type":
                                result = _extract_type_annotation(sub)
                                assert result == "int"
                                return
        pytest.fail("Did not find type node")

    def test_optional_normalized(self):
        source = """
from pydantic import BaseModel
class M(BaseModel):
    x: Optional[str] = None
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        for sub in child.children:
                            if sub.type == "type":
                                result = _extract_type_annotation(sub)
                                assert result == "str | None"
                                return
        pytest.fail("Did not find type node")


class TestIsFieldRequired:
    def test_no_equals_is_required(self):
        source = """
from pydantic import BaseModel
class M(BaseModel):
    x: str
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        req, default = _is_field_required(child)
                        assert req is True
                        assert default is None
                        return
        pytest.fail("Did not find assignment node")

    def test_with_default_value(self):
        source = """
from pydantic import BaseModel
class M(BaseModel):
    x: str = "hello"
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        req, default = _is_field_required(child)
                        assert req is False
                        assert default is not None
                        return
        pytest.fail("Did not find assignment node")

    def test_field_ellipsis_is_required(self):
        source = """
from pydantic import BaseModel, Field
class M(BaseModel):
    x: str = Field(...)
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        req, default = _is_field_required(child)
                        assert req is True
                        assert default is None
                        return
        pytest.fail("Did not find assignment node")

    def test_field_with_default_kwarg(self):
        source = """
from pydantic import BaseModel, Field
class M(BaseModel):
    x: str = Field(default="hello")
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        req, default = _is_field_required(child)
                        assert req is False
                        assert default is not None
                        return
        pytest.fail("Did not find assignment node")

    def test_field_with_default_factory(self):
        source = """
from pydantic import BaseModel, Field
class M(BaseModel):
    x: list = Field(default_factory=list)
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        req, default = _is_field_required(child)
                        assert req is False
                        return
        pytest.fail("Did not find assignment node")

    def test_field_with_only_keyword_args_no_default(self):
        source = """
from pydantic import BaseModel, Field
class M(BaseModel):
    x: str = Field(description="Name")
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        req, default = _is_field_required(child)
                        assert req is True
                        assert default is None
                        return
        pytest.fail("Did not find assignment node")

    def test_field_none_positional_default(self):
        source = """
from pydantic import BaseModel, Field
class M(BaseModel):
    x: str | None = Field(None, description="Optional")
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        req, default = _is_field_required(child)
                        assert req is False
                        return
        pytest.fail("Did not find assignment node")

    def test_field_no_args_is_required(self):
        source = """
from pydantic import BaseModel, Field
class M(BaseModel):
    x: str = Field()
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        req, default = _is_field_required(child)
                        assert req is True
                        return
        pytest.fail("Did not find assignment node")


class TestCheckAnnotatedField:
    def test_annotated_with_field_ellipsis(self):
        source = """
from pydantic import BaseModel, Field
from typing import Annotated
class M(BaseModel):
    x: Annotated[int, Field(...)]
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        for sub in child.children:
                            if sub.type == "type":
                                result = _check_annotated_field(sub)
                                assert result is not None
                                req, default = result
                                assert req is True
                                assert default is None
                                return
        pytest.fail("Did not find type node")

    def test_annotated_with_field_default(self):
        source = """
from pydantic import BaseModel, Field
from typing import Annotated
class M(BaseModel):
    x: Annotated[str, Field(default="hello")]
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        for sub in child.children:
                            if sub.type == "type":
                                result = _check_annotated_field(sub)
                                assert result is not None
                                req, default = result
                                assert req is False
                                return
        pytest.fail("Did not find type node")

    def test_non_annotated_returns_none(self):
        source = """
from pydantic import BaseModel
class M(BaseModel):
    x: str
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        for sub in child.children:
                            if sub.type == "type":
                                result = _check_annotated_field(sub)
                                assert result is None
                                return
        pytest.fail("Did not find type node")

    def test_annotated_field_no_args(self):
        source = """
from pydantic import BaseModel, Field
from typing import Annotated
class M(BaseModel):
    x: Annotated[str, Field()]
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        for sub in child.children:
                            if sub.type == "type":
                                result = _check_annotated_field(sub)
                                assert result is not None
                                req, default = result
                                assert req is True
                                return
        pytest.fail("Did not find type node")

    def test_annotated_field_positional_default(self):
        source = """
from pydantic import BaseModel, Field
from typing import Annotated
class M(BaseModel):
    x: Annotated[str | None, Field(None)]
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        for sub in child.children:
                            if sub.type == "type":
                                result = _check_annotated_field(sub)
                                assert result is not None
                                req, default = result
                                assert req is False
                                return
        pytest.fail("Did not find type node")


class TestExtractAnnotatedInnerType:
    def test_annotated_str(self):
        source = """
from pydantic import BaseModel, Field
from typing import Annotated
class M(BaseModel):
    x: Annotated[str, Field(...)]
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        for sub in child.children:
                            if sub.type == "type":
                                result = _extract_annotated_inner_type(sub)
                                assert result == "str"
                                return
        pytest.fail("Did not find type node")

    def test_non_annotated_returns_full_text(self):
        source = """
from pydantic import BaseModel
class M(BaseModel):
    x: int
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        body = class_node.child_by_field_name("body")
        for stmt in body.children:
            if stmt.type == "expression_statement":
                for child in stmt.children:
                    if child.type == "assignment":
                        for sub in child.children:
                            if sub.type == "type":
                                result = _extract_annotated_inner_type(sub)
                                assert result == "int"
                                return
        pytest.fail("Did not find type node")


class TestExtractFieldsFromBody:
    def test_multiple_fields(self):
        source = """
from pydantic import BaseModel
class M(BaseModel):
    name: str
    age: int = 0
    active: bool
"""
        root = _parse_tree(source)
        body = _find_node_type(root, "class_definition").child_by_field_name("body")
        fields = _extract_fields_from_body(body)
        assert len(fields) == 3
        field_map = {f.name: f for f in fields}
        assert field_map["name"].required is True
        assert field_map["age"].required is False
        assert field_map["active"].required is True

    def test_non_assignment_statements_skipped(self):
        source = """
from pydantic import BaseModel
class M(BaseModel):
    name: str

    def method(self):
        pass
"""
        root = _parse_tree(source)
        body = _find_node_type(root, "class_definition").child_by_field_name("body")
        fields = _extract_fields_from_body(body)
        assert len(fields) == 1
        assert fields[0].name == "name"


class TestHasModelValidator:
    def test_with_model_validator(self):
        source = """
from pydantic import BaseModel, model_validator
class M(BaseModel):
    x: str

    @model_validator(mode="after")
    def check(self):
        return self
"""
        root = _parse_tree(source)
        body = _find_node_type(root, "class_definition").child_by_field_name("body")
        assert _has_model_validator(body) is True

    def test_without_model_validator(self):
        source = """
from pydantic import BaseModel
class M(BaseModel):
    x: str
"""
        root = _parse_tree(source)
        body = _find_node_type(root, "class_definition").child_by_field_name("body")
        assert _has_model_validator(body) is False

    def test_with_other_decorator(self):
        source = """
from pydantic import BaseModel
class M(BaseModel):
    x: str

    @property
    def computed(self):
        return self.x
"""
        root = _parse_tree(source)
        body = _find_node_type(root, "class_definition").child_by_field_name("body")
        assert _has_model_validator(body) is False


class TestGetBaseClasses:
    def test_single_base(self):
        source = """
class M(BaseModel):
    pass
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        bases = _get_base_classes(class_node)
        assert bases == ["BaseModel"]

    def test_multiple_bases(self):
        source = """
class M(Base, Mixin):
    pass
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        bases = _get_base_classes(class_node)
        assert "Base" in bases
        assert "Mixin" in bases

    def test_no_bases(self):
        source = """
class M:
    pass
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        bases = _get_base_classes(class_node)
        assert bases == []

    def test_attribute_base(self):
        source = """
class M(module.BaseModel):
    pass
"""
        root = _parse_tree(source)
        class_node = _find_node_type(root, "class_definition")
        bases = _get_base_classes(class_node)
        assert len(bases) == 1
        assert "module.BaseModel" in bases[0]


class TestParseSchemasIntegration:
    def test_empty_source(self):
        result = parse_schemas("")
        assert len(result.models) == 0

    def test_whitespace_only(self):
        result = parse_schemas("   \n\n  ")
        assert len(result.models) == 0

    def test_simple_model(self):
        source = """
from pydantic import BaseModel
class Item(BaseModel):
    name: str
    price: float
"""
        result = parse_schemas(source)
        assert "Item" in result.models
        model = result.models["Item"]
        assert len(model.fields) == 2

    def test_non_basemodel_class_ignored(self):
        source = """
class NotAModel:
    x: str
"""
        result = parse_schemas(source)
        assert len(result.models) == 0

    def test_transitive_inheritance(self):
        source = """
from pydantic import BaseModel
class Base(BaseModel):
    id: str

class Middle(Base):
    name: str

class Child(Middle):
    extra: int
"""
        result = parse_schemas(source)
        assert "Child" in result.models
        child = result.models["Child"]
        field_names = [f.name for f in child.fields]
        assert "id" in field_names
        assert "name" in field_names
        assert "extra" in field_names

    def test_resolve_false(self):
        source = """
from pydantic import BaseModel
class Base(BaseModel):
    id: str

class Child(Base):
    name: str
"""
        result = parse_schemas(source, resolve=False)
        child = result.models["Child"]
        field_names = [f.name for f in child.fields]
        assert "id" not in field_names
        assert "name" in field_names

    def test_class_without_body(self):
        """Classes with only pass should be handled."""
        source = """
from pydantic import BaseModel
class Empty(BaseModel):
    pass
"""
        result = parse_schemas(source)
        assert "Empty" in result.models
        assert len(result.models["Empty"].fields) == 0

    def test_multiple_models_same_file(self):
        source = """
from pydantic import BaseModel, Field
class Request(BaseModel):
    data: str
    config: dict = Field(default_factory=dict)

class Response(BaseModel):
    result: str
    error: str | None = None
"""
        result = parse_schemas(source)
        assert len(result.models) == 2
        req = result.models["Request"]
        resp = result.models["Response"]
        assert len(req.fields) == 2
        assert len(resp.fields) == 2

    def test_annotated_inner_type_with_optional(self):
        source = """
from pydantic import BaseModel, Field
from typing import Annotated, Optional
class M(BaseModel):
    x: Annotated[Optional[str], Field(default=None)]
"""
        result = parse_schemas(source)
        model = result.models["M"]
        assert model.fields[0].type_annotation == "str | None"
        assert model.fields[0].required is False


class TestResolveInheritance:
    def test_parent_fields_prepended(self):
        schema = ServiceSchema()
        schema.models["Base"] = PydanticModel(
            name="Base",
            bases=["BaseModel"],
            fields=[ModelField(name="id", type_annotation="str", required=True)],
        )
        schema.models["Child"] = PydanticModel(
            name="Child",
            bases=["Base"],
            fields=[ModelField(name="name", type_annotation="str", required=True)],
        )
        resolve_inheritance(schema)
        child = schema.models["Child"]
        assert [f.name for f in child.fields] == ["id", "name"]

    def test_override_field_not_duplicated(self):
        schema = ServiceSchema()
        schema.models["Base"] = PydanticModel(
            name="Base",
            bases=["BaseModel"],
            fields=[
                ModelField(name="id", type_annotation="str", required=True),
                ModelField(name="name", type_annotation="str", required=True),
            ],
        )
        schema.models["Child"] = PydanticModel(
            name="Child",
            bases=["Base"],
            fields=[
                ModelField(name="name", type_annotation="str", required=False, default='"default"')
            ],
        )
        resolve_inheritance(schema)
        child = schema.models["Child"]
        names = [f.name for f in child.fields]
        assert names.count("name") == 1
        # The child's version should win (required=False)
        name_field = [f for f in child.fields if f.name == "name"][0]
        assert name_field.required is False

    def test_unknown_base_ignored(self):
        schema = ServiceSchema()
        schema.models["Child"] = PydanticModel(
            name="Child",
            bases=["UnknownBase"],
            fields=[ModelField(name="x", type_annotation="int", required=True)],
        )
        resolve_inheritance(schema)
        assert [f.name for f in schema.models["Child"].fields] == ["x"]

    def test_basemodel_base_skipped(self):
        schema = ServiceSchema()
        schema.models["Simple"] = PydanticModel(
            name="Simple",
            bases=["BaseModel"],
            fields=[ModelField(name="x", type_annotation="str", required=True)],
        )
        resolve_inheritance(schema)
        assert [f.name for f in schema.models["Simple"].fields] == ["x"]


# ═══════════════════════════════════════════════════════════════════════
# ROUTE PARSER TESTS
# ═══════════════════════════════════════════════════════════════════════


class TestExtractStringContent:
    def test_simple_string(self):
        root = _parse_tree('x = "/api/foo"')
        string_node = _find_node_type(root, "string")
        assert _extract_string_content(string_node) == "/api/foo"

    def test_empty_string(self):
        root = _parse_tree('x = ""')
        string_node = _find_node_type(root, "string")
        assert _extract_string_content(string_node) == ""

    def test_non_string_returns_none(self):
        root = _parse_tree("x = 42")
        int_node = _find_node_type(root, "integer")
        assert _extract_string_content(int_node) is None


class TestFindAllRouterPrefixes:
    def test_single_router(self):
        source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")
"""
        root = _parse_tree(source)
        result = _find_all_router_prefixes(root)
        assert result == {"router": "/api"}

    def test_multiple_routers(self):
        source = """
from fastapi import APIRouter
r1 = APIRouter(prefix="/api/v1")
r2 = APIRouter(prefix="/api/v2")
"""
        root = _parse_tree(source)
        result = _find_all_router_prefixes(root)
        assert result == {"r1": "/api/v1", "r2": "/api/v2"}

    def test_router_without_prefix(self):
        source = """
from fastapi import APIRouter
router = APIRouter()
"""
        root = _parse_tree(source)
        result = _find_all_router_prefixes(root)
        assert result == {"router": ""}

    def test_no_router(self):
        source = "x = 1"
        root = _parse_tree(source)
        result = _find_all_router_prefixes(root)
        assert result == {}


class TestExtractRequestModel:
    def test_custom_type_detected(self):
        source = """
async def create(body: CreateRequest, request: Request):
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = _extract_request_model(func)
        assert result == "CreateRequest"

    def test_builtin_types_ignored(self):
        source = """
async def get_item(item_id: str, count: int):
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = _extract_request_model(func)
        assert result is None

    def test_no_params(self):
        source = """
async def health():
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = _extract_request_model(func)
        assert result is None


class TestExtractResponseModel:
    def test_response_model_kwarg(self):
        source = """
@router.post("/foo", response_model=FooResponse)
async def create():
    pass
"""
        root = _parse_tree(source)
        dec = _find_node_type(root, "decorator")
        result = _extract_response_model(dec)
        assert result == "FooResponse"

    def test_no_response_model(self):
        source = """
@router.get("/foo")
async def get_foo():
    pass
"""
        root = _parse_tree(source)
        dec = _find_node_type(root, "decorator")
        result = _extract_response_model(dec)
        assert result is None


class TestExtractReturnAnnotation:
    def test_custom_return_type(self):
        source = """
async def create() -> FooResponse:
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = _extract_return_annotation(func)
        assert result == "FooResponse"

    def test_builtin_return_type_ignored(self):
        source = """
async def get() -> dict:
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = _extract_return_annotation(func)
        assert result is None

    def test_no_return_type(self):
        source = """
async def get():
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = _extract_return_annotation(func)
        assert result is None


class TestExtractStatusCode:
    def test_status_code_found(self):
        source = """
@router.post("/foo", status_code=201)
async def create():
    pass
"""
        root = _parse_tree(source)
        dec = _find_node_type(root, "decorator")
        assert _extract_status_code(dec) == 201

    def test_no_status_code_defaults_200(self):
        source = """
@router.get("/foo")
async def get():
    pass
"""
        root = _parse_tree(source)
        dec = _find_node_type(root, "decorator")
        assert _extract_status_code(dec) == 200

    def test_invalid_status_code_value(self):
        source = """
@router.get("/foo", status_code=status.HTTP_200_OK)
async def get():
    pass
"""
        root = _parse_tree(source)
        dec = _find_node_type(root, "decorator")
        # Can't parse non-integer, falls back to 200
        assert _extract_status_code(dec) == 200


class TestRouteExtractQueryParams:
    def test_query_required(self):
        source = """
@router.get("/items")
async def list_items(page: int = Query(...)):
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = route_extract_query_params(func, "/items")
        assert len(result) == 1
        assert result[0].name == "page"
        assert result[0].required is True

    def test_query_optional_with_default(self):
        source = """
@router.get("/items")
async def list_items(limit: int = Query(default=20)):
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = route_extract_query_params(func, "/items")
        assert len(result) == 1
        assert result[0].name == "limit"
        assert result[0].required is False

    def test_query_optional_with_positional_default(self):
        source = """
@router.get("/items")
async def list_items(limit: int = Query(20)):
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = route_extract_query_params(func, "/items")
        assert len(result) == 1
        assert result[0].name == "limit"
        assert result[0].required is False

    def test_path_param_excluded(self):
        source = """
@router.get("/items/{item_id}")
async def get_item(item_id: str = Query(...)):
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = route_extract_query_params(func, "/items/{item_id}")
        assert len(result) == 0

    def test_self_and_request_excluded(self):
        source = """
@router.get("/items")
async def list_items(self, request: Request, page: int = Query(...)):
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = route_extract_query_params(func, "/items")
        qp_names = [qp.name for qp in result]
        assert "self" not in qp_names
        assert "request" not in qp_names
        assert "page" in qp_names

    def test_implicit_query_param_simple_type_in_builtins(self):
        """str/int/float/bool are in BUILTIN_TYPES so they don't match the
        'not in BUILTIN_TYPES and in _SIMPLE_TYPES' condition -- no implicit
        query param is emitted for them."""
        source = """
@router.get("/items")
async def list_items(sort: str):
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = route_extract_query_params(func, "/items")
        # str is in BUILTIN_TYPES, so the implicit query param branch is dead
        assert len(result) == 0

    def test_no_params(self):
        source = """
@router.get("/health")
async def health():
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = route_extract_query_params(func, "/health")
        assert len(result) == 0

    def test_body_param_excluded(self):
        source = """
@router.post("/items")
async def create(body: ItemCreate, limit: int = Query(default=10)):
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = route_extract_query_params(func, "/items")
        qp_names = [qp.name for qp in result]
        assert "body" not in qp_names
        assert "limit" in qp_names

    def test_query_with_default_factory(self):
        source = """
@router.get("/items")
async def list_items(tags: str = Query(default_factory=list)):
    pass
"""
        root = _parse_tree(source)
        func = _find_node_type(root, "function_definition")
        result = route_extract_query_params(func, "/items")
        assert len(result) == 1
        assert result[0].required is False


class TestParseRoutesIntegration:
    def test_empty_source(self):
        result = parse_routes("")
        assert len(result.routes) == 0

    def test_whitespace_only(self):
        result = parse_routes("   \n\n  ")
        assert len(result.routes) == 0

    def test_basic_crud_routes(self):
        source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")

@router.post("/items", status_code=201)
async def create(body: ItemCreate) -> ItemResponse:
    pass

@router.get("/items/{item_id}")
async def get_item(item_id: str) -> ItemResponse:
    pass

@router.put("/items/{item_id}")
async def update(item_id: str, body: ItemUpdate) -> ItemResponse:
    pass

@router.delete("/items/{item_id}")
async def delete(item_id: str):
    pass
"""
        result = parse_routes(source)
        assert len(result.routes) == 4
        methods = sorted(r.method for r in result.routes)
        assert methods == ["DELETE", "GET", "POST", "PUT"]

    def test_prefix_applied(self):
        source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api/v1")

@router.get("/items")
async def list_items():
    pass
"""
        result = parse_routes(source)
        assert result.routes[0].path == "/api/v1/items"

    def test_external_prefix(self):
        source = """
from fastapi import APIRouter
router = APIRouter()

@router.get("/items")
async def list_items():
    pass
"""
        result = parse_routes(source, prefix="/api/v2")
        assert result.routes[0].path == "/api/v2/items"

    def test_multi_router_per_file(self):
        source = """
from fastapi import APIRouter
public = APIRouter(prefix="/api/public")
admin = APIRouter(prefix="/api/admin")

@public.get("/items")
async def list_items():
    pass

@admin.get("/users")
async def list_users():
    pass
"""
        result = parse_routes(source)
        paths = sorted(r.path for r in result.routes)
        assert paths == ["/api/admin/users", "/api/public/items"]

    def test_non_http_method_ignored(self):
        source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")

@router.websocket("/ws")
async def websocket_endpoint():
    pass
"""
        result = parse_routes(source)
        assert len(result.routes) == 0

    def test_no_decorator_ignored(self):
        source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")

async def not_a_route():
    pass
"""
        result = parse_routes(source)
        assert len(result.routes) == 0

    def test_response_model_from_decorator(self):
        source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")

@router.post("/items", response_model=ItemResponse)
async def create(body: ItemCreate):
    pass
"""
        result = parse_routes(source)
        assert result.routes[0].response_model == "ItemResponse"

    def test_response_model_from_return_annotation(self):
        source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")

@router.get("/items")
async def list_items() -> ItemList:
    pass
"""
        result = parse_routes(source)
        assert result.routes[0].response_model == "ItemList"

    def test_decorator_response_model_takes_precedence(self):
        source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")

@router.get("/items", response_model=ItemListResponse)
async def list_items() -> dict:
    pass
"""
        result = parse_routes(source)
        assert result.routes[0].response_model == "ItemListResponse"

    def test_all_http_methods(self):
        source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")

@router.get("/a")
async def a(): pass

@router.post("/b")
async def b(): pass

@router.put("/c")
async def c(): pass

@router.delete("/d")
async def d(): pass

@router.patch("/e")
async def e(): pass

@router.head("/f")
async def f(): pass

@router.options("/g")
async def g(): pass

@router.trace("/h")
async def h(): pass
"""
        result = parse_routes(source)
        methods = sorted(r.method for r in result.routes)
        assert methods == ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT", "TRACE"]

    def test_empty_path_with_prefix(self):
        source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api/compile")

@router.post("")
async def start(): pass
"""
        result = parse_routes(source)
        assert result.routes[0].path == "/api/compile"

    def test_external_prefix_with_local_prefix(self):
        source = """
from fastapi import APIRouter
router = APIRouter(prefix="/local")

@router.get("/items")
async def list_items(): pass
"""
        result = parse_routes(source, prefix="/external")
        assert result.routes[0].path == "/external/local/items"

    def test_query_params_attached_to_route(self):
        source = """
from fastapi import APIRouter, Query
router = APIRouter(prefix="/api")

@router.get("/items")
async def list_items(page: int = Query(...), limit: int = Query(default=10)):
    pass
"""
        result = parse_routes(source)
        route = result.routes[0]
        qp_names = {qp.name for qp in route.query_params}
        assert "page" in qp_names
        assert "limit" in qp_names

    def test_decorator_without_call_ignored(self):
        """@some_decorator without () is not a route."""
        source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")

@some_decorator
async def not_route():
    pass
"""
        result = parse_routes(source)
        assert len(result.routes) == 0

    def test_decorator_not_attribute_ignored(self):
        """@standalone_func("/path") is not a route."""
        source = """
from fastapi import APIRouter
router = APIRouter(prefix="/api")

@standalone_func("/path")
async def not_route():
    pass
"""
        result = parse_routes(source)
        assert len(result.routes) == 0


# ===========================================================================
# flow_check tests (lines 122-139, 145, 151-153)
# ===========================================================================

from processing.contract_analysis.checks.flow_check import (
    _looks_like_enum,
    _looks_like_model_name,
    _types_compatible,
    check_flow,
)
from processing.contract_analysis.checks.types import RootCause, ViolationSeverity
from processing.contract_analysis.flow.registry import FlowEdge


class TestFlowCheck:
    """Test check_flow and type compatibility functions."""

    def test_check_flow_model_not_found(self):
        """Missing model in schema (lines 21-30)."""
        edge = FlowEdge(
            producer_model="ModelA",
            consumer_model="ModelB",
            label="test",
        )
        schema_a = ServiceSchema(models={})
        schema_b = ServiceSchema(models={})
        violations = check_flow(edge, schema_a, schema_b)
        assert len(violations) == 1
        assert violations[0].severity == ViolationSeverity.WARN

    def test_check_flow_required_field_not_forwarded(self):
        """Required field not in forwarded_fields (lines 37-51)."""
        edge = FlowEdge(
            producer_model="ModelA",
            consumer_model="ModelB",
            label="test",
            forwarded_fields={"name"},
        )
        model_a = PydanticModel(
            name="ModelA",
            fields=[ModelField(name="name", type_annotation="str", required=True)],
        )
        model_b = PydanticModel(
            name="ModelB",
            fields=[
                ModelField(name="name", type_annotation="str", required=True),
                ModelField(name="email", type_annotation="str", required=True),
            ],
        )
        schema_a = ServiceSchema(models={"ModelA": model_a})
        schema_b = ServiceSchema(models={"ModelB": model_b})
        violations = check_flow(edge, schema_a, schema_b)
        # email is required but not forwarded
        assert any(v.root_cause == RootCause.FLOW_REQUIRED_DROPPED for v in violations)

    def test_check_flow_field_missing_in_producer(self):
        """Consumer requires field not present in producer (lines 53-71)."""
        edge = FlowEdge(
            producer_model="ModelA",
            consumer_model="ModelB",
            label="test",
        )
        model_a = PydanticModel(
            name="ModelA",
            fields=[ModelField(name="name", type_annotation="str", required=True)],
        )
        model_b = PydanticModel(
            name="ModelB",
            fields=[
                ModelField(name="name", type_annotation="str", required=True),
                ModelField(name="age", type_annotation="int", required=True),
            ],
        )
        schema_a = ServiceSchema(models={"ModelA": model_a})
        schema_b = ServiceSchema(models={"ModelB": model_b})
        violations = check_flow(edge, schema_a, schema_b)
        assert any(v.root_cause == RootCause.FLOW_FIELD_MISSING for v in violations)

    def test_check_flow_type_mismatch(self):
        """Type mismatch between producer and consumer (lines 76-89)."""
        edge = FlowEdge(
            producer_model="ModelA",
            consumer_model="ModelB",
            label="test",
        )
        model_a = PydanticModel(
            name="ModelA",
            fields=[ModelField(name="val", type_annotation="str", required=True)],
        )
        model_b = PydanticModel(
            name="ModelB",
            fields=[ModelField(name="val", type_annotation="list[str]", required=True)],
        )
        schema_a = ServiceSchema(models={"ModelA": model_a})
        schema_b = ServiceSchema(models={"ModelB": model_b})
        violations = check_flow(edge, schema_a, schema_b)
        assert any(v.root_cause == RootCause.FLOW_TYPE_MISMATCH for v in violations)

    def test_check_flow_type_warn_enum_to_str(self):
        """Enum to str produces WARN, not FAIL (lines 90-102)."""
        edge = FlowEdge(
            producer_model="ModelA",
            consumer_model="ModelB",
            label="test",
        )
        model_a = PydanticModel(
            name="ModelA",
            fields=[ModelField(name="status", type_annotation="TaskStatus", required=True)],
        )
        model_b = PydanticModel(
            name="ModelB",
            fields=[ModelField(name="status", type_annotation="str", required=True)],
        )
        schema_a = ServiceSchema(models={"ModelA": model_a})
        schema_b = ServiceSchema(models={"ModelB": model_b})
        violations = check_flow(edge, schema_a, schema_b)
        # enum -> str is a WARN, not FAIL
        warn_violations = [v for v in violations if v.severity == ViolationSeverity.WARN]
        assert len(warn_violations) >= 1


class TestTypesCompatible:
    """Test _types_compatible function (lines 107-139, 145, 151-153)."""

    def test_same_types(self):
        compat, reason = _types_compatible("str", "str")
        assert compat is True
        assert reason == ""

    def test_enum_to_str(self):
        compat, reason = _types_compatible("TaskType", "str")
        assert compat is True
        assert "enum" in reason

    def test_str_to_list_str_incompatible(self):
        compat, reason = _types_compatible("str", "list[str]")
        assert compat is False

    def test_model_to_dict(self):
        compat, reason = _types_compatible("UserModel", "dict")
        assert compat is True
        assert "untyped dict" in reason

    def test_list_model_to_list_dict(self):
        compat, reason = _types_compatible("list[UserModel]", "list[dict]")
        assert compat is True

    def test_list_inner_type_recursive(self):
        compat, reason = _types_compatible("list[str]", "list[str]")
        assert compat is True

    def test_optional_handling(self):
        compat, reason = _types_compatible("str | None", "str")
        assert compat is True

    def test_optional_bracket_handling(self):
        compat, reason = _types_compatible("Optional[str]", "str")
        assert compat is True

    def test_different_model_names(self):
        compat, reason = _types_compatible("UserModel", "AccountModel")
        assert compat is False
        assert "different model types" in reason

    def test_incompatible_types(self):
        compat, reason = _types_compatible("int", "str")
        assert compat is False


class TestLooksLikeEnum:
    """Test _looks_like_enum (lines 142-146)."""

    def test_enum_suffix(self):
        assert _looks_like_enum("TaskType") is True
        assert _looks_like_enum("UserStatus") is True
        assert _looks_like_enum("OrderKind") is True

    def test_not_enum(self):
        assert _looks_like_enum("str") is False
        assert _looks_like_enum("") is False
        assert _looks_like_enum("lowercase") is False


class TestLooksLikeModelName:
    """Test _looks_like_model_name (lines 149-153)."""

    def test_model_name(self):
        assert _looks_like_model_name("UserModel") is True
        assert _looks_like_model_name("Account") is True

    def test_not_model(self):
        assert _looks_like_model_name("str") is False
        assert _looks_like_model_name("") is False
        assert _looks_like_model_name("Optional") is False
        assert _looks_like_model_name("None") is False
        assert _looks_like_model_name("Any") is False


# ===========================================================================
# syntax_check tests (lines 19, 46-50)
# ===========================================================================

from processing.contract_analysis.checks.syntax_check import (
    _fix_pep758,
    check_syntax,
)


class TestSyntaxCheck:
    """Test check_syntax and PEP 758 tolerance."""

    def test_valid_python(self):
        violations = check_syntax({"test.py": "x = 1\nprint(x)\n"})
        assert violations == []

    def test_syntax_error(self):
        violations = check_syntax({"bad.py": "def f(\n"})
        assert len(violations) == 1
        assert violations[0].severity == ViolationSeverity.FAIL
        assert "bad.py" in violations[0].message

    def test_pep758_bare_except_comma_accepted(self):
        """PEP 758 unparenthesized multi-except is tolerated (lines 44-48)."""
        code = "try:\n    x = 1\nexcept ValueError, TypeError:\n    pass\n"
        violations = check_syntax({"pep758.py": code})
        # Should not report this as a syntax error (PEP 758)
        assert len(violations) == 0

    def test_pep758_fix_still_broken(self):
        """After PEP 758 fix, if still broken, report that error (lines 49-50)."""
        # This code has a PEP 758 pattern but also additional syntax errors
        code = "try:\n    x = 1\nexcept ValueError, TypeError:\n    def broken(\n"
        violations = check_syntax({"bad_pep758.py": code})
        assert len(violations) == 1
        assert "bad_pep758.py" in violations[0].message

    def test_multiple_files(self):
        sources = {
            "good.py": "x = 1\n",
            "bad.py": "def (\n",
        }
        violations = check_syntax(sources)
        assert len(violations) == 1
        assert "bad.py" in violations[0].message


class TestFixPep758:
    """Test _fix_pep758 helper (line 19)."""

    def test_parenthesizes_bare_except(self):
        code = "except ValueError, TypeError:"
        result = _fix_pep758(code)
        assert "(ValueError, TypeError)" in result

    def test_already_parenthesized(self):
        code = "except (ValueError, TypeError):"
        result = _fix_pep758(code)
        assert result == code

    def test_single_except_unchanged(self):
        code = "except ValueError:"
        result = _fix_pep758(code)
        assert result == code
