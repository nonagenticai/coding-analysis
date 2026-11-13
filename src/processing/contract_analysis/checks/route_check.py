"""Check that client endpoint paths and HTTP methods match service routes."""

from __future__ import annotations

import re

from ..parsers.types import ClientCall, RouteDefinition
from .types import Violation, ViolationSeverity, normalize_path


def _wildcard_match(client_path: str, route_map: dict[str, list[tuple[str, str]]]) -> str | None:
    """Match a client path with {param} wildcards against concrete routes.

    Handles f-string interpolation patterns like /api/{service}/health where
    the client constructs the path at runtime from a variable, and the server
    has concrete routes like /api/layout/health, /api/tables/health, etc.
    """
    if "{param}" not in client_path:
        return None
    # Build a regex from the client path, replacing {param} with a segment matcher
    pattern = re.escape(client_path).replace(re.escape("{param}"), r"[^/]+")
    pattern = f"^{pattern}$"
    for route_key in route_map:
        if re.match(pattern, route_key):
            return route_key
    return None


def check_routes(calls: list[ClientCall], routes: list[RouteDefinition]) -> list[Violation]:
    violations: list[Violation] = []
    # Store (method, original_path) tuples for better messaging
    route_map: dict[str, list[tuple[str, str]]] = {}
    for r in routes:
        key = normalize_path(r.path)
        route_map.setdefault(key, []).append((r.method.upper(), r.path))

    for call in calls:
        # Skip wrapper method calls -- these use @wrapper/method_name pseudo-paths
        # which don't map to real HTTP paths.
        if call.path.startswith("@wrapper/"):
            continue

        norm_path = normalize_path(call.path)
        # Try exact match first, then wildcard match for f-string paths
        if norm_path not in route_map:
            matched_key = _wildcard_match(norm_path, route_map)
            if matched_key is not None:
                norm_path = matched_key
        if norm_path not in route_map:
            violations.append(
                Violation(
                    severity=ViolationSeverity.FAIL,
                    check="route_path",
                    message=f"Client calls {call.method} {call.path} but no matching route exists",
                    client_location=call.caller,
                )
            )
        else:
            available_methods = [method for method, _ in route_map[norm_path]]
            if call.method.upper() not in available_methods:
                # Include original paths for better diagnostics
                original_paths = sorted({orig for _, orig in route_map[norm_path]})
                available_str = ", ".join(available_methods)
                route_str = ", ".join(original_paths)
                violations.append(
                    Violation(
                        severity=ViolationSeverity.FAIL,
                        check="http_method",
                        message=(
                            f"Client uses {call.method} {call.path} "
                            f"but service only defines [{available_str}] "
                            f"for {route_str}"
                        ),
                        client_location=call.caller,
                    )
                )
    return violations
