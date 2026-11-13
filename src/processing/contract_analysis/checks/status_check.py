"""Check for non-200 status codes on routes (WARN only, informational)."""

from __future__ import annotations

from ..parsers.types import ClientCall, RouteDefinition
from .types import Violation, ViolationSeverity, normalize_path

# Standard REST status codes that don't need warnings
_EXPECTED_REST_CODES = {201, 202, 204}


def check_status_codes(
    calls: list[ClientCall],
    routes: list[RouteDefinition],
) -> list[Violation]:
    violations: list[Violation] = []

    route_status_map: dict[tuple[str, str], int] = {}
    for r in routes:
        if r.status_code != 200 and r.status_code not in _EXPECTED_REST_CODES:
            key = (normalize_path(r.path), r.method.upper())
            route_status_map[key] = r.status_code

    for call in calls:
        norm_path = normalize_path(call.path)
        key = (norm_path, call.method.upper())
        status = route_status_map.get(key)
        if status is not None:
            violations.append(
                Violation(
                    severity=ViolationSeverity.WARN,
                    check="status_code",
                    message=(
                        f"Route {call.method} {call.path} returns status {status}, "
                        f"not 200 -- client should handle this"
                    ),
                    client_location=call.caller,
                )
            )
    return violations
