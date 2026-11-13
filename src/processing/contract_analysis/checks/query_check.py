"""Check that client sends required query parameters."""

from __future__ import annotations

from ..parsers.types import ClientCall, RouteDefinition
from .types import Violation, ViolationSeverity, normalize_path


def check_query_params(
    calls: list[ClientCall],
    routes: list[RouteDefinition],
) -> list[Violation]:
    violations: list[Violation] = []

    route_query_map: dict[tuple[str, str], list[str]] = {}
    for r in routes:
        required_params = [qp.name for qp in r.query_params if qp.required]
        if required_params:
            key = (normalize_path(r.path), r.method.upper())
            route_query_map[key] = required_params

    for call in calls:
        norm_path = normalize_path(call.path)
        key = (norm_path, call.method.upper())
        required = route_query_map.get(key)
        if not required:
            continue

        sent = set(call.query_params)
        for param_name in required:
            if param_name not in sent:
                violations.append(
                    Violation(
                        severity=ViolationSeverity.FAIL,
                        check="query_param",
                        message=(
                            f"Client does not send required query parameter '{param_name}' "
                            f"for {call.method} {call.path}"
                        ),
                        client_location=call.caller,
                    )
                )
    return violations
