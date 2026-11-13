"""Parse HTTP client calls from Python source using tree-sitter."""

from __future__ import annotations

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from .types import ClientCall, ClientContract

PY_LANGUAGE = Language(tspython.language())

# Method patterns: maps client helper method names to HTTP methods
# submit_job -> POST, poll_job -> GET, _request -> explicit method arg
HELPER_METHODS = {
    "submit_job": "POST",
    "poll_job": "GET",
}

EXPLICIT_METHOD_HELPERS = {"_request"}

HTTPX_METHODS = {"get", "post", "put", "delete", "patch", "head", "options"}

# Methods that take (method_str, path, ...) instead of (path, ...)
HTTPX_EXPLICIT_METHODS = {"stream", "request"}

# Factory function patterns that return HTTP client wrappers.
# Variable assigned from these functions is tracked for wrapper method calls.
_CLIENT_FACTORY_PATTERNS = {"_get_http_client", "_get_client"}


def _make_parser() -> Parser:
    return Parser(PY_LANGUAGE)


def _node_text(node) -> str:
    """Extract text from a tree-sitter node."""
    return node.text.decode() if node.text else ""


def _extract_string_value(node) -> str | None:
    """Extract the string value from a string node, handling f-strings."""
    if node.type != "string":
        return None
    # Reconstruct the string content, preserving f-string interpolations
    parts = []
    is_fstring = False
    for child in node.children:
        if child.type == "string_start":
            text = _node_text(child)
            if text.startswith("f") or text.startswith("F"):
                is_fstring = True
        elif child.type == "string_content":
            parts.append(_node_text(child))
        elif child.type == "interpolation":
            # Keep the {expression} as-is for path templates
            parts.append(_node_text(child))

    if not parts and not is_fstring:
        return None
    return "".join(parts) if parts else None


def _get_positional_args(call_node) -> list:
    """Get positional argument nodes from a call node."""
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return []
    positional = []
    for child in args.children:
        if child.type in ("(", ")", ",", "keyword_argument", "dictionary_splat"):
            continue
        positional.append(child)
    return positional


def _find_dict_literal_keys(func_body, var_name: str) -> list[str]:
    """Find keys from a dict literal assigned to var_name in the function body.

    Also finds payload["key"] = ... patterns.
    """
    keys = []
    for stmt in func_body.children:
        if stmt.type == "expression_statement":
            for child in stmt.children:
                if child.type == "assignment":
                    # Check for `var_name = {...}`
                    lhs = child.children[0]
                    if lhs.type == "identifier" and _node_text(lhs) == var_name:
                        # Find the dict literal on the right side
                        for sub in child.children:
                            if sub.type == "dictionary":
                                for pair in sub.children:
                                    if pair.type == "pair":
                                        key_node = pair.children[0]
                                        key_text = _extract_string_value(key_node)
                                        if key_text:
                                            keys.append(key_text)
                    # Check for `var_name["key"] = ...`
                    if lhs.type == "subscript":
                        obj = lhs.children[0] if lhs.children else None
                        if obj and obj.type == "identifier" and _node_text(obj) == var_name:
                            # Extract the subscript key
                            for sub in lhs.children:
                                if sub.type == "string":
                                    key_text = _extract_string_value(sub)
                                    if key_text:
                                        keys.append(key_text)

        # Also check inside if-blocks for payload["key"] = ...
        if stmt.type == "if_statement":
            for child in stmt.children:
                if child.type == "block":
                    sub_keys = _find_dict_literal_keys(child, var_name)
                    keys.extend(sub_keys)

    return keys


def _find_wrapper_client_vars(node) -> set[str]:
    """Find variable names assigned from client factory functions.

    Detects patterns like:
        client = _get_http_client()
        client = _get_client()

    These factory functions return ServiceClient wrappers whose methods
    (e.g. .scan(), .broadcast()) internally make HTTP calls.
    """
    result: set[str] = set()
    _walk_for_wrapper_client_vars(node, result)
    return result


def _walk_for_wrapper_client_vars(node, result: set[str]) -> None:
    """Recursively walk looking for assignments from client factory functions."""
    if node.type == "expression_statement":
        for child in node.children:
            if child.type == "assignment":
                lhs = child.child_by_field_name("left")
                rhs = child.child_by_field_name("right")
                if lhs is None or rhs is None:
                    continue
                if lhs.type != "identifier":
                    continue
                # Check if RHS is a call to a factory function (with or without await)
                call_node = rhs
                if rhs.type == "await":
                    for sub in rhs.children:
                        if sub.type == "call":
                            call_node = sub
                            break
                if call_node.type == "call":
                    func = call_node.child_by_field_name("function")
                    if func is not None and func.type == "identifier":
                        func_name = _node_text(func)
                        if func_name in _CLIENT_FACTORY_PATTERNS:
                            result.add(_node_text(lhs))

    for child in node.children:
        _walk_for_wrapper_client_vars(child, result)


def _walk_for_wrapper_method_calls(
    node,
    caller: str,
    func_body,
    wrapper_vars: set[str],
    calls: list[ClientCall],
    func_node=None,
) -> None:
    """Detect method calls on wrapper client variables.

    Catches patterns like:
        client = _get_http_client()
        await client.scan(json={"code": code, "language": language})
        await client.broadcast(json={"event_type": event_type})

    When json= is present, infers POST. Otherwise infers GET.
    The wrapper method name is stored in the path as a hint for reporting.
    """
    if node.type == "call":
        func = node.child_by_field_name("function")
        if func is not None and func.type == "attribute":
            obj = func.child_by_field_name("object")
            attr = func.child_by_field_name("attribute")
            if obj is not None and attr is not None and obj.type == "identifier":
                var_name = _node_text(obj)
                method_name = _node_text(attr)
                if var_name in wrapper_vars and method_name not in HTTPX_METHODS:
                    has_json = _has_json_kwarg(node)
                    payload_fields = _extract_json_payload_fields(node, func_body)
                    http_method = "POST" if has_json else "GET"
                    calls.append(
                        ClientCall(
                            method=http_method,
                            path=f"@wrapper/{method_name}",
                            payload_fields=payload_fields,
                            payload_opaque=has_json and not payload_fields,
                            caller=caller,
                        )
                    )

    for child in node.children:
        _walk_for_wrapper_method_calls(child, caller, func_body, wrapper_vars, calls, func_node)


def _find_calls_in_function(func_node) -> list[ClientCall]:
    """Find HTTP client calls within a function definition."""
    func_name_node = func_node.child_by_field_name("name")
    caller = _node_text(func_name_node) if func_name_node else ""
    body = func_node.child_by_field_name("body")
    if body is None:
        return []

    # Collect call nodes alongside their ClientCall objects
    call_pairs: list[tuple[object, ClientCall]] = []
    _walk_for_calls_with_nodes(body, caller, body, call_pairs)

    # Also detect httpx direct calls
    httpx_vars = _find_httpx_client_vars(body)
    httpx_calls: list[ClientCall] = []
    if httpx_vars:
        _walk_for_httpx_calls(body, caller, body, httpx_vars, httpx_calls, func_node)

    calls = [pair[1] for pair in call_pairs] + httpx_calls

    # Collect already-detected call node IDs to avoid double-counting
    skip_nodes = {id(pair[0]) for pair in call_pairs}

    # Generic fallback: any obj.HTTP_METHOD("/path") not already detected
    generic_calls: list[ClientCall] = []
    _walk_for_generic_http_calls(body, caller, body, generic_calls, skip_nodes, func_node)

    # Deduplicate by (method, path)
    existing = {(c.method, c.path) for c in calls}
    for gc in generic_calls:
        if (gc.method, gc.path) not in existing:
            calls.append(gc)

    # Wrapper client detection: variables from _get_http_client() etc.
    wrapper_vars = _find_wrapper_client_vars(body)
    if wrapper_vars:
        wrapper_calls: list[ClientCall] = []
        _walk_for_wrapper_method_calls(body, caller, body, wrapper_vars, wrapper_calls, func_node)
        existing = {(c.method, c.path) for c in calls}
        for wc in wrapper_calls:
            if (wc.method, wc.path) not in existing:
                calls.append(wc)

    # Proxy forwarding detection: proxy.forward_request(path=..., method=...)
    proxy_calls: list[ClientCall] = []
    _walk_for_proxy_calls(body, caller, body, proxy_calls)
    existing = {(c.method, c.path) for c in calls}
    for pc in proxy_calls:
        if (pc.method, pc.path) not in existing:
            calls.append(pc)

    # Build mapping: variable name -> ClientCall for assigned results
    var_to_call = _map_vars_to_calls(body, call_pairs)

    # Extract response reads from variable usage
    if var_to_call:
        _extract_response_reads(body, var_to_call)

    return calls


def _walk_for_calls_with_nodes(
    node, caller: str, func_body, call_pairs: list[tuple[object, ClientCall]]
) -> None:
    """Recursively walk nodes, collecting (call_node, ClientCall) pairs."""
    if node.type == "call":
        func = node.child_by_field_name("function")
        if func and func.type == "attribute":
            children = list(func.children)
            idents = [c for c in children if c.type == "identifier"]
            if len(idents) >= 2 and _node_text(idents[0]) == "self":
                method_name = _node_text(idents[-1])
                call_result = _parse_client_call(node, method_name, caller, func_body)
                if call_result is not None:
                    call_pairs.append((node, call_result))

    for child in node.children:
        _walk_for_calls_with_nodes(child, caller, func_body, call_pairs)


def _node_contains(parent, target) -> bool:
    """Check if target node is a descendant of parent node.

    Uses tree-sitter node IDs for comparison since different Python wrapper
    objects may represent the same underlying node.
    """
    if parent.id == target.id:
        return True
    for child in parent.children:
        if _node_contains(child, target):
            return True
    return False


def _map_vars_to_calls(
    func_body, call_pairs: list[tuple[object, ClientCall]]
) -> dict[str, ClientCall]:
    """Build a mapping from variable names to ClientCall for assigned results.

    Detects patterns like:
        result = await self._request(...)
        data = await self.submit_job(...)
    """
    var_to_call: dict[str, ClientCall] = {}

    for stmt in func_body.children:
        if stmt.type == "expression_statement":
            for child in stmt.children:
                if child.type == "assignment":
                    lhs = child.child_by_field_name("left")
                    rhs = child.child_by_field_name("right")
                    if lhs is None or rhs is None:
                        continue
                    if lhs.type != "identifier":
                        continue
                    var_name = _node_text(lhs)
                    # Check if the RHS contains one of our detected call nodes
                    for call_node, client_call in call_pairs:
                        if _node_contains(rhs, call_node):
                            var_to_call[var_name] = client_call
                            break

    return var_to_call


def _extract_response_reads(func_body, var_to_call: dict[str, ClientCall]) -> None:
    """Scan the function body for field accesses on tracked response variables.

    Detects:
        - var.get("field", ...) -> response_reads += ["field"]
        - var["field"] -> response_reads += ["field"]
    """
    _walk_for_response_reads(func_body, var_to_call)


def _walk_for_response_reads(node, var_to_call: dict[str, ClientCall]) -> None:
    """Recursively walk looking for .get() calls and subscript accesses."""
    if node.type == "call":
        # Check for var.get("field", ...)
        func = node.child_by_field_name("function")
        if func and func.type == "attribute":
            obj_node = func.child_by_field_name("object")
            attr_node = func.child_by_field_name("attribute")
            if (
                obj_node
                and attr_node
                and obj_node.type == "identifier"
                and _node_text(attr_node) == "get"
            ):
                var_name = _node_text(obj_node)
                if var_name in var_to_call:
                    # Extract first string argument
                    positional = _get_positional_args(node)
                    if positional:
                        field_name = _extract_string_value(positional[0])
                        if field_name:
                            client_call = var_to_call[var_name]
                            if field_name not in client_call.response_reads:
                                client_call.response_reads.append(field_name)

    if node.type == "subscript":
        # Check for var["field"]
        obj_node = node.child_by_field_name("value")
        subscript_node = node.child_by_field_name("subscript")
        if obj_node and subscript_node and obj_node.type == "identifier":
            var_name = _node_text(obj_node)
            if var_name in var_to_call:
                field_name = _extract_string_value(subscript_node)
                if field_name:
                    client_call = var_to_call[var_name]
                    if field_name not in client_call.response_reads:
                        client_call.response_reads.append(field_name)

    for child in node.children:
        _walk_for_response_reads(child, var_to_call)


def _find_httpx_client_vars(node) -> set[str]:
    """Find variable names bound as httpx AsyncClient context managers."""
    result: set[str] = set()
    _walk_for_httpx_vars(node, result)
    return result


def _walk_for_httpx_vars(node, result: set[str]) -> None:
    """Recursively walk looking for with_statement nodes binding httpx clients."""
    if node.type == "with_statement":
        for child in node.children:
            if child.type == "with_clause":
                for with_item in child.children:
                    if with_item.type == "with_item":
                        _extract_httpx_var_from_with_item(with_item, result)

    for child in node.children:
        _walk_for_httpx_vars(child, result)


def _extract_httpx_var_from_with_item(with_item, result: set[str]) -> None:
    """Extract httpx client var from a with_item containing an as_pattern."""
    as_pattern = with_item.child_by_field_name("value")
    if as_pattern is None or as_pattern.type != "as_pattern":
        return

    call_node = None
    target_node = None
    for child in as_pattern.children:
        if child.type == "call":
            call_node = child
        elif child.type == "as_pattern_target":
            target_node = child

    if call_node is None or target_node is None:
        return

    func = call_node.child_by_field_name("function")
    if func is None or func.type != "attribute":
        return

    obj = func.child_by_field_name("object")
    attr = func.child_by_field_name("attribute")
    if obj is None or attr is None:
        return

    if _node_text(obj) == "httpx" and _node_text(attr) == "AsyncClient":
        for child in target_node.children:
            if child.type == "identifier":
                result.add(_node_text(child))
                return


def _walk_for_httpx_calls(
    node,
    caller: str,
    func_body,
    httpx_vars: set[str],
    calls: list[ClientCall],
    func_node=None,
) -> None:
    """Recursively walk looking for HTTP method calls on httpx client variables."""
    if node.type == "call":
        func = node.child_by_field_name("function")
        if func is not None and func.type == "attribute":
            obj = func.child_by_field_name("object")
            attr = func.child_by_field_name("attribute")
            if obj is not None and attr is not None and obj.type == "identifier":
                var_name = _node_text(obj)
                method_name = _node_text(attr)
                if var_name in httpx_vars and method_name in HTTPX_METHODS:
                    call = _parse_httpx_call(node, method_name, caller, func_body, func_node)
                    if call is not None:
                        calls.append(call)

    for child in node.children:
        _walk_for_httpx_calls(child, caller, func_body, httpx_vars, calls, func_node)


def _parse_httpx_call(
    call_node, method_name: str, caller: str, func_body, func_node=None
) -> ClientCall | None:
    """Parse a single httpx method call into a ClientCall."""
    positional = _get_positional_args(call_node)
    if not positional:
        return None

    path = _extract_string_value(positional[0])
    if path is None:
        return None

    payload_fields = _extract_json_payload_fields(call_node, func_body)
    has_json = _has_json_kwarg(call_node)
    payload_model = _resolve_payload_model(call_node, func_node) if func_node else None
    query_params = _extract_query_params(call_node, path, func_body)

    clean_path = path.split("?", 1)[0] if "?" in path else path

    return ClientCall(
        method=method_name.upper(),
        path=clean_path,
        payload_fields=payload_fields,
        payload_opaque=has_json and not payload_fields,
        caller=caller,
        payload_model=payload_model,
        query_params=query_params,
    )


def _extract_kwarg_payload_fields(call_node, func_body, kwarg_name: str) -> list[str]:
    """Extract payload field names from a keyword argument (json=, data=, etc.)."""
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return []

    for child in args.children:
        if child.type == "keyword_argument":
            key_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if key_node is None or value_node is None:
                continue
            if _node_text(key_node) != kwarg_name:
                continue

            if value_node.type == "dictionary":
                return _extract_dict_keys(value_node)

            if value_node.type == "identifier":
                var_name = _node_text(value_node)
                return _find_dict_literal_keys(func_body, var_name)

    return []


def _extract_json_payload_fields(call_node, func_body) -> list[str]:
    """Extract payload field names from json= keyword argument."""
    return _extract_kwarg_payload_fields(call_node, func_body, "json")


def _extract_dict_keys(dict_node) -> list[str]:
    """Extract string keys from a dictionary literal node."""
    keys: list[str] = []
    for child in dict_node.children:
        if child.type == "pair":
            key_node = child.children[0] if child.children else None
            if key_node is not None:
                key_text = _extract_string_value(key_node)
                if key_text is not None:
                    keys.append(key_text)
    return keys


def _strip_base_url(path: str) -> str:
    """Strip {self.base_url} or similar interpolation prefix from paths."""
    if not path.startswith("/"):
        for prefix in ("/api/", "/_internal/"):
            if prefix in path:
                return path[path.index(prefix) :]
    return path


def _extract_query_params(call_node, path: str, func_body) -> list[str]:
    """Extract query parameter names from URL string and params= kwarg."""
    result: list[str] = []
    # Check for ?key=val in path
    if "?" in path:
        query_part = path.split("?", 1)[1]
        for param in query_part.split("&"):
            name = param.split("=", 1)[0].strip()
            if name:
                result.append(name)
    # Check for params= kwarg
    args = call_node.child_by_field_name("arguments")
    if args:
        for child in args.children:
            if child.type == "keyword_argument":
                key_node = child.child_by_field_name("name")
                value_node = child.child_by_field_name("value")
                if key_node is None or value_node is None:
                    continue
                if _node_text(key_node) != "params":
                    continue
                if value_node.type == "dictionary":
                    result.extend(_extract_dict_keys(value_node))
                elif value_node.type == "identifier":
                    var_name = _node_text(value_node)
                    result.extend(_find_dict_literal_keys(func_body, var_name))
    return result


def _has_json_kwarg(call_node) -> bool:
    """Check if the call has a json= keyword argument."""
    return _has_kwarg(call_node, "json")


def _detect_model_dump_var(call_node) -> str | None:
    """Check if json= kwarg value is a .model_dump() or .dict() call.

    Returns the variable name if found, e.g. 'payload' from json=payload.model_dump().
    """
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return None
    for child in args.children:
        if child.type == "keyword_argument":
            key_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if key_node is None or value_node is None:
                continue
            if _node_text(key_node) != "json":
                continue
            # Check if value is a call to .model_dump() or .dict()
            if value_node.type == "call":
                func = value_node.child_by_field_name("function")
                if func and func.type == "attribute":
                    attr = func.child_by_field_name("attribute")
                    obj = func.child_by_field_name("object")
                    if attr and obj and _node_text(attr) in ("model_dump", "dict"):
                        return _node_text(obj)
            # Also handle await: json=await payload.model_dump()
            if value_node.type == "await":
                for sub in value_node.children:
                    if sub.type == "call":
                        func = sub.child_by_field_name("function")
                        if func and func.type == "attribute":
                            attr = func.child_by_field_name("attribute")
                            obj = func.child_by_field_name("object")
                            if attr and obj and _node_text(attr) in ("model_dump", "dict"):
                                return _node_text(obj)
    return None


def _find_param_type(func_node, var_name: str) -> str | None:
    """Find the type annotation for a parameter in a function definition."""
    params = func_node.child_by_field_name("parameters")
    if params is None:
        return None
    for child in params.children:
        if child.type in ("typed_parameter", "typed_default_parameter"):
            name_node = None
            type_node = None
            for sub in child.children:
                if sub.type == "identifier" and name_node is None:
                    name_node = sub
                elif sub.type == "type":
                    type_node = sub
            if name_node and type_node and _node_text(name_node) == var_name:
                return _node_text(type_node)
    return None


def _resolve_payload_model(call_node, func_node) -> str | None:
    """Try to resolve payload_model from json=X.model_dump() pattern."""
    dump_var = _detect_model_dump_var(call_node)
    if dump_var is None:
        return None
    return _find_param_type(func_node, dump_var)


def _parse_generic_http_call(
    call_node, method_name: str, caller: str, func_body, func_node=None
) -> ClientCall | None:
    """Parse obj.get("/path"), obj.post("/path", json=data), etc."""
    positional = _get_positional_args(call_node)
    if not positional:
        return None
    path = _extract_string_value(positional[0])
    if path is None:
        return None
    path = _strip_base_url(path)
    if not path.startswith("/"):
        return None  # Not an HTTP path -- skip (avoids dict.get("key") false positives)

    payload_fields = _extract_json_payload_fields(call_node, func_body)
    has_json_arg = _has_json_kwarg(call_node)
    payload_model = _resolve_payload_model(call_node, func_node) if func_node else None
    query_params = _extract_query_params(call_node, path, func_body)

    # Strip query string from path for route matching
    clean_path = path.split("?", 1)[0] if "?" in path else path

    return ClientCall(
        method=method_name.upper(),
        path=clean_path,
        payload_fields=payload_fields,
        payload_opaque=has_json_arg and not payload_fields,
        caller=caller,
        payload_model=payload_model,
        query_params=query_params,
    )


def _parse_explicit_http_call(
    call_node, caller: str, func_body, func_node=None
) -> ClientCall | None:
    """Parse obj.stream("POST", "/path", ...) or obj.request("GET", "/path", ...)."""
    positional = _get_positional_args(call_node)
    if len(positional) < 2:
        return None
    method_text = _extract_string_value(positional[0])
    if method_text is None:
        return None
    path = _extract_string_value(positional[1])
    if path is None:
        return None
    path = _strip_base_url(path)
    if not path.startswith("/"):
        return None

    payload_fields = _extract_json_payload_fields(call_node, func_body)
    has_json_arg = _has_json_kwarg(call_node)
    payload_model = _resolve_payload_model(call_node, func_node) if func_node else None
    query_params = _extract_query_params(call_node, path, func_body)

    clean_path = path.split("?", 1)[0] if "?" in path else path

    return ClientCall(
        method=method_text.upper(),
        path=clean_path,
        payload_fields=payload_fields,
        payload_opaque=has_json_arg and not payload_fields,
        caller=caller,
        payload_model=payload_model,
        query_params=query_params,
    )


def _walk_for_generic_http_calls(
    node,
    caller: str,
    func_body,
    calls: list[ClientCall],
    skip_nodes: set[int],
    func_node=None,
) -> None:
    """Detect any obj.HTTP_METHOD("/path", ...) call.

    Catches stored httpx client patterns:
    - self._client.get("/api/...")
    - client.post("/api/...", json=data)
    - self._client.stream("POST", "/api/...")
    """
    if node.type == "call" and id(node) not in skip_nodes:
        func = node.child_by_field_name("function")
        if func and func.type == "attribute":
            method_node = func.child_by_field_name("attribute")
            if method_node:
                method_name = _node_text(method_node)
                if method_name in HTTPX_METHODS:
                    call = _parse_generic_http_call(node, method_name, caller, func_body, func_node)
                    if call is not None:
                        calls.append(call)
                elif method_name in HTTPX_EXPLICIT_METHODS:
                    call = _parse_explicit_http_call(node, caller, func_body, func_node)
                    if call is not None:
                        calls.append(call)

    for child in node.children:
        _walk_for_generic_http_calls(child, caller, func_body, calls, skip_nodes, func_node)


def _parse_client_call(call_node, method_name: str, caller: str, func_body) -> ClientCall | None:
    """Parse a single client call node into a ClientCall."""
    positional = _get_positional_args(call_node)

    if method_name in HELPER_METHODS:
        # submit_job(path, payload) or poll_job(path, job_id, ...)
        http_method = HELPER_METHODS[method_name]
        if not positional:
            return None

        path = _extract_string_value(positional[0])
        if path is None:
            return None

        # poll_job appends /{job_id} at runtime
        if method_name == "poll_job":
            path = path.rstrip("/") + "/{param}"

        payload_fields: list[str] = []
        if method_name == "submit_job" and len(positional) >= 2:
            payload_arg = positional[1]
            if payload_arg.type == "identifier":
                # Trace the variable back to its dict literal
                var_name = _node_text(payload_arg)
                payload_fields = _find_dict_literal_keys(func_body, var_name)
            elif payload_arg.type == "dictionary":
                for pair in payload_arg.children:
                    if pair.type == "pair":
                        key_node = pair.children[0]
                        key_text = _extract_string_value(key_node)
                        if key_text:
                            payload_fields.append(key_text)

        has_payload_arg = method_name == "submit_job" and len(positional) >= 2

        return ClientCall(
            method=http_method,
            path=path,
            payload_fields=payload_fields,
            payload_opaque=has_payload_arg and not payload_fields,
            caller=caller,
        )

    if method_name in EXPLICIT_METHOD_HELPERS:
        # _request("METHOD", "path") or _request("METHOD", f"path/{var}")
        if len(positional) < 2:
            return None

        method_arg = positional[0]
        path_arg = positional[1]

        method_text = _extract_string_value(method_arg)
        if method_text is None:
            return None

        path = _extract_string_value(path_arg)
        if path is None:
            return None

        payload_fields = _extract_json_payload_fields(call_node, func_body)
        has_json = _has_json_kwarg(call_node)

        return ClientCall(
            method=method_text.upper(),
            path=path,
            payload_fields=payload_fields,
            payload_opaque=has_json and not payload_fields,
            caller=caller,
        )

    return None


# Methods on a proxy object that forward HTTP requests (KAG gateway pattern).
_PROXY_FORWARD_METHODS = {"forward_request", "forward_request_raw"}


def _extract_kwarg_string(call_node, kwarg_name: str) -> str | None:
    """Extract the string value of a keyword argument from a call node."""
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return None
    for child in args.children:
        if child.type == "keyword_argument":
            key_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if key_node is None or value_node is None:
                continue
            if _node_text(key_node) == kwarg_name:
                return _extract_string_value(value_node)
    return None


def _walk_for_proxy_calls(
    node,
    caller: str,
    func_body,
    calls: list[ClientCall],
) -> None:
    """Detect proxy.forward_request(path=..., method=...) calls.

    The KAG gateway uses a proxy object to forward requests:
        await proxy.forward_request(
            service_url=settings.SOME_URL,
            path="/api/endpoint",
            method="POST",
            ...
        )

    This walker extracts the path and method keyword arguments.
    """
    if node.type == "call":
        func = node.child_by_field_name("function")
        if func is not None and func.type == "attribute":
            attr = func.child_by_field_name("attribute")
            if attr is not None and _node_text(attr) in _PROXY_FORWARD_METHODS:
                path = _extract_kwarg_string(node, "path")
                method = _extract_kwarg_string(node, "method")
                if path is not None and method is not None:
                    clean_path = path.split("?", 1)[0] if "?" in path else path
                    # Extract data= payload fields if present
                    data_fields = _extract_data_payload_fields(node, func_body)
                    has_data = _has_kwarg(node, "data")
                    calls.append(
                        ClientCall(
                            method=method.upper(),
                            path=clean_path,
                            payload_fields=data_fields,
                            payload_opaque=has_data and not data_fields,
                            caller=caller,
                        )
                    )

    for child in node.children:
        _walk_for_proxy_calls(child, caller, func_body, calls)


def _extract_data_payload_fields(call_node, func_body) -> list[str]:
    """Extract payload field names from data= keyword argument."""
    return _extract_kwarg_payload_fields(call_node, func_body, "data")


def _has_kwarg(call_node, kwarg_name: str) -> bool:
    """Check if a call has a specific keyword argument."""
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return False
    for child in args.children:
        if child.type == "keyword_argument":
            key = child.child_by_field_name("name")
            if key and _node_text(key) == kwarg_name:
                return True
    return False


def parse_client(source: str) -> ClientContract:
    """Parse all HTTP client calls from Python source code.

    Detects patterns like:
    - self.submit_job("/api/path", payload) -> POST
    - self._request("GET", "/api/path") -> explicit method
    - self.poll_job("/api/path", job_id) -> GET
    - proxy.forward_request(path="/api/path", method="POST") -> proxy forwarding

    Args:
        source: Python source code string.

    Returns:
        ClientContract with parsed calls and any errors.
    """
    if not source.strip():
        return ClientContract()

    parser = _make_parser()
    tree = parser.parse(source.encode())
    root = tree.root_node

    result = ClientContract()

    # Walk through all function definitions (including nested in classes)
    _find_functions(root, result)

    return result


def _find_functions(node, result: ClientContract) -> None:
    """Recursively find function definitions and extract client calls."""
    if node.type == "function_definition":
        calls = _find_calls_in_function(node)
        result.calls.extend(calls)
        return  # Don't recurse into nested functions

    for child in node.children:
        _find_functions(child, result)
