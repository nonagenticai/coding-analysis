"""Check that client sends all required request fields."""

from __future__ import annotations

from ..parsers.types import ClientCall, RouteDefinition, ServiceSchema
from .types import Violation, ViolationSeverity, normalize_path


def check_request_fields(
    calls: list[ClientCall],
    routes: list[RouteDefinition],
    schema: ServiceSchema,
) -> list[Violation]:
    violations: list[Violation] = []

    route_model_map: dict[tuple[str, str], str] = {}
    for r in routes:
        if r.request_model:
            key = (normalize_path(r.path), r.method.upper())
            route_model_map[key] = r.request_model

    for call in calls:
        norm_path = normalize_path(call.path)
        key = (norm_path, call.method.upper())
        model_name = route_model_map.get(key)
        if not model_name:
            continue

        # Skip field check when payload exists but can't be statically analyzed
        if call.payload_opaque and not call.payload_model:
            continue

        model = schema.models.get(model_name)
        if not model:
            continue

        if call.payload_model:
            pm = schema.models.get(call.payload_model)
            sent_fields = {f.name for f in pm.fields} if pm else set()
        else:
            sent_fields = set(call.payload_fields)
        for f in model.fields:
            if f.required and f.name not in sent_fields:
                violations.append(
                    Violation(
                        severity=ViolationSeverity.FAIL,
                        check="request_field",
                        message=(
                            f"Client does not send required field '{f.name}' "
                            f"expected by {model_name}"
                        ),
                        client_location=call.caller,
                        service_location=model_name,
                    )
                )
    return violations
