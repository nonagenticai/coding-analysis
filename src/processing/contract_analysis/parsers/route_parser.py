"""Parse FastAPI and Flask route definitions from Python source using tree-sitter."""

from __future__ import annotations

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from .types import QueryParam, RouteDefinition, ServiceRoutes

PY_LANGUAGE = Language(tspython.language())

# HTTP methods that FastAPI routers support
HTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options", "trace"}

# Built-in types and framework types that are NOT request models
BUILTIN_TYPES = {
    "str",
    "int",
    "float",
    "bool",
    "dict",
    "list",
    "tuple",
    "set",
    "bytes",
    "None",
    "Request",
    "Response",
    "BackgroundTasks",
    "HTTPException",
    "Depends",
}


def _make_parser() -> Parser:
    return Parser(PY_LANGUAGE)


def _node_text(node) -> str:
    """Extract text from a tree-sitter node."""
    return node.text.decode() if node.text else ""


def _extract_string_content(node) -> str | None:
    """Extract string content from a string node (without quotes)."""
    if node.type != "string":
        return None
    for child in node.children:
        if child.type == "string_content":
            return _node_text(child)
    # Empty string "" has string_start + string_end but no string_content
    has_string_start = any(c.type == "string_start" for c in node.children)
    if has_string_start:
        return ""
    return None


def _find_all_router_prefixes(root) -> dict[str, str]:
    """Find all APIRouter(prefix=...) assignments and return {var_name: prefix}."""
    result: dict[str, str] = {}
    for node in root.children:
        if node.type != "expression_statement":
            continue
        for child in node.children:
            if child.type != "assignment":
                continue
            # Get the variable name on the left
            lhs = child.child_by_field_name("left")
            if lhs is None or lhs.type != "identifier":
                continue
            var_name = _node_text(lhs)
            # Look for call to APIRouter on the right
            for sub in child.children:
                if sub.type == "call":
                    func = sub.child_by_field_name("function")
                    if func and _node_text(func).split(".")[-1] in ("APIRouter", "Blueprint"):
                        prefix = ""
                        args = sub.child_by_field_name("arguments")
                        if args:
                            for arg in args.children:
                                if arg.type == "keyword_argument":
                                    key = arg.children[0]
                                    if (
                                        _node_text(key) in ("prefix", "url_prefix")
                                        and len(arg.children) >= 3
                                    ):
                                        val = arg.children[2]
                                        content = _extract_string_content(val)
                                        if content is not None:
                                            prefix = content
                        result[var_name] = prefix
    return result


def _find_app_vars(root) -> set[str]:
    """Module-level names assigned from ``FastAPI(...)``."""
    names: set[str] = set()
    for node in root.children:
        if node.type != "expression_statement":
            continue
        for child in node.children:
            if child.type != "assignment":
                continue
            lhs = child.child_by_field_name("left")
            rhs = child.child_by_field_name("right")
            if lhs is None or rhs is None or lhs.type != "identifier" or rhs.type != "call":
                continue
            func = rhs.child_by_field_name("function")
            if func is not None and _node_text(func).split(".")[-1] in ("FastAPI", "Flask"):
                names.add(_node_text(lhs))
    return names


def _extract_request_model(func_node) -> str | None:
    """Extract request model type from function parameters.

    Looks for typed parameters where the type is a custom class (not builtin).
    """
    params = func_node.child_by_field_name("parameters")
    if params is None:
        return None

    for child in params.children:
        if child.type == "typed_parameter":
            # Get the type annotation
            type_node = None
            for sub in child.children:
                if sub.type == "type":
                    type_node = sub
                    break
            if type_node is None:
                continue
            type_text = _node_text(type_node)
            if type_text not in BUILTIN_TYPES:
                return type_text
    return None


def _extract_response_model(decorator_node) -> str | None:
    """Extract response_model from decorator keyword arguments."""
    for child in decorator_node.children:
        if child.type == "call":
            args = child.child_by_field_name("arguments")
            if args:
                for arg in args.children:
                    if arg.type == "keyword_argument":
                        key = arg.children[0]
                        if _node_text(key) == "response_model" and len(arg.children) >= 3:
                            return _node_text(arg.children[2])
    return None


def _extract_return_annotation(func_node) -> str | None:
    """Extract response model from function return type annotation.

    Detects `async def foo(...) -> ResponseModel:` patterns.
    Returns None for builtin types (dict, list, str, etc.).
    """
    return_type = func_node.child_by_field_name("return_type")
    if return_type is None:
        return None
    type_text = _node_text(return_type)
    if type_text in BUILTIN_TYPES:
        return None
    return type_text


# Simple types that indicate query parameters (not path params, not body models)
_SIMPLE_TYPES = {"str", "int", "float", "bool"}


def _extract_query_params(func_node, path: str) -> list[QueryParam]:
    """Extract Query(...) parameters from function signature.

    Detects:
    - param: str = Query(...) -- required query param
    - param: str = Query(default="x") -- optional query param
    - param: str (simple type, not in path template) -- implicit query param
    """
    params_node = func_node.child_by_field_name("parameters")
    if params_node is None:
        return []

    # Collect path parameter names from the path template
    import re

    path_params = set(re.findall(r"\{(\w+)\}", path))

    result: list[QueryParam] = []
    for child in params_node.children:
        if child.type == "typed_default_parameter":
            # param: type = Query(...)
            name_node = None
            type_node = None
            default_node = None
            for sub in child.children:
                if sub.type == "identifier" and name_node is None:
                    name_node = sub
                elif sub.type == "type":
                    type_node = sub
                elif sub.type == "call" or (
                    sub.type not in (":", "=", "identifier", "type") and default_node is None
                ):
                    default_node = sub
            if name_node is None:
                continue
            param_name = _node_text(name_node)
            if param_name in ("self", "request", "body") or param_name in path_params:
                continue
            # Check if default is a Query() call
            if default_node and default_node.type == "call":
                func = default_node.child_by_field_name("function")
                if func and _node_text(func) == "Query":
                    required = True
                    args = default_node.child_by_field_name("arguments")
                    if args:
                        for arg in args.children:
                            if arg.type == "ellipsis":
                                required = True
                                break
                            if arg.type == "keyword_argument":
                                key = arg.children[0]
                                if _node_text(key) in ("default", "default_factory"):
                                    required = False
                                    break
                            if arg.type not in ("(", ")", ",", "keyword_argument"):
                                required = False
                                break
                    type_ann = _node_text(type_node) if type_node else "str"
                    result.append(
                        QueryParam(name=param_name, required=required, type_annotation=type_ann)
                    )
                    continue
            # Not a Query() default -- skip (could be Depends, Body, etc.)
        elif child.type == "typed_parameter":
            # param: type (no default)
            name_node = None
            type_node = None
            for sub in child.children:
                if sub.type == "identifier" and name_node is None:
                    name_node = sub
                elif sub.type == "type":
                    type_node = sub
            if name_node is None or type_node is None:
                continue
            param_name = _node_text(name_node)
            type_text = _node_text(type_node)
            if param_name in ("self", "request", "body") or param_name in path_params:
                continue
            if type_text not in BUILTIN_TYPES and type_text in _SIMPLE_TYPES:
                # Simple-typed param not in path -> implicit query param
                result.append(QueryParam(name=param_name, required=True, type_annotation=type_text))

    return result


def _extract_status_code(decorator_node) -> int:
    """Extract status_code=N from decorator keyword arguments."""
    for child in decorator_node.children:
        if child.type == "call":
            args = child.child_by_field_name("arguments")
            if args:
                for arg in args.children:
                    if arg.type == "keyword_argument":
                        key = arg.children[0]
                        if _node_text(key) == "status_code" and len(arg.children) >= 3:
                            try:
                                return int(_node_text(arg.children[2]))
                            except ValueError:
                                pass
    return 200


def _flask_methods(call_node) -> list[str] | None:
    """``methods=[...]`` of a Flask ``route`` decorator: GET when absent, None when dynamic."""
    args = call_node.child_by_field_name("arguments")
    for arg in args.children if args is not None else []:
        if arg.type != "keyword_argument" or _node_text(arg.children[0]) != "methods":
            continue
        value = arg.children[2] if len(arg.children) >= 3 else None
        if value is None or value.type not in ("list", "tuple", "set"):
            return None
        methods = []
        for item in value.named_children:
            text = _extract_string_content(item) if item.type == "string" else None
            if text is None or text.lower() not in HTTP_METHODS:
                return None
            methods.append(text.lower())
        return methods
    return ["get"]


def parse_routes(source: str, *, prefix: str = "") -> ServiceRoutes:
    """Parse all FastAPI route definitions from Python source code.

    Args:
        source: Python source code string.
        prefix: Optional external prefix to prepend when the file has no
            local ``APIRouter(prefix=...)`` defined. Useful for sub-module
            route files that inherit their prefix from a parent
            ``__init__.py``.

    Returns:
        ServiceRoutes with parsed routes and any errors.
    """
    if not source.strip():
        return ServiceRoutes()

    parser = _make_parser()
    tree = parser.parse(source.encode())
    root = tree.root_node

    result = ServiceRoutes()
    router_prefixes = _find_all_router_prefixes(root)
    app_vars = _find_app_vars(root)

    # Backward compat: when only one router exists, use its prefix as file_prefix
    if len(router_prefixes) == 1:
        file_prefix = next(iter(router_prefixes.values()))
    elif not router_prefixes:
        file_prefix = ""
    else:
        file_prefix = None  # multi-router mode: resolve per-decorator

    for node in root.children:
        if node.type != "decorated_definition":
            continue

        # Find decorator(s)
        func_node = None
        decorators = []
        for child in node.children:
            if child.type == "decorator":
                decorators.append(child)
            elif child.type == "function_definition":
                func_node = child

        if func_node is None:
            continue

        for dec in decorators:
            # Look for @router.method("path") pattern
            call_node = None
            for child in dec.children:
                if child.type == "call":
                    call_node = child
                    break

            if call_node is None:
                continue

            # Get the attribute (router.post, router.get, etc.)
            func = call_node.child_by_field_name("function")
            if func is None or func.type != "attribute":
                continue

            # Extract router variable and method name from the attribute
            attr_children = [c for c in func.children if c.type == "identifier"]
            if len(attr_children) < 2:
                continue
            router_var = _node_text(attr_children[0])
            method_name = _node_text(attr_children[-1])

            if method_name == "route":  # Flask: @app.route("/x", methods=["GET", "POST"])
                methods = _flask_methods(call_node)
                if methods is None:
                    args = call_node.child_by_field_name("arguments")
                    first = next(
                        (
                            a
                            for a in (args.children if args is not None else [])
                            if a.type == "string"
                        ),
                        None,
                    )
                    result.unresolved.append(
                        (
                            dec.start_point[0] + 1,
                            _extract_string_content(first) if first is not None else None,
                            _node_text(call_node)[:200],
                            "route methods are not a literal list of HTTP methods",
                        )
                    )
                    continue
            elif method_name in HTTP_METHODS:
                methods = [method_name]
            else:
                continue

            # Determine prefix for this route
            if router_var in app_vars:
                # A FastAPI() app never carries an APIRouter's prefix.
                route_prefix = prefix
            elif file_prefix is not None:
                # Single router mode
                route_prefix = prefix + file_prefix
            else:
                # Multi-router mode: look up this router's prefix
                route_prefix = prefix + router_prefixes.get(router_var, "")

            # Extract path from first string argument
            args = call_node.child_by_field_name("arguments")
            if args is None:
                continue

            path = None
            for arg in args.children:
                if arg.type == "string":
                    path = _extract_string_content(arg)
                    break

            if path is None:
                continue

            full_path = route_prefix + path

            # Extract request model and response model
            request_model = _extract_request_model(func_node)
            response_model = _extract_response_model(dec) or _extract_return_annotation(func_node)
            query_params = _extract_query_params(func_node, full_path)
            status_code = _extract_status_code(dec)

            for method in methods:
                result.routes.append(
                    RouteDefinition(
                        method=method.upper(),
                        path=full_path,
                        request_model=request_model,
                        response_model=response_model,
                        query_params=query_params,
                        status_code=status_code,
                        line=dec.start_point[0] + 1,
                        handler=_node_text(func_node.child_by_field_name("name")),
                        router=router_var,
                        rule=path,
                    )
                )

    return result
