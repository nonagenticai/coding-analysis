"""Check that client response reads match service response models."""

from __future__ import annotations

from ..parsers.types import ClientCall, RouteDefinition, ServiceSchema
from .types import Violation, ViolationSeverity, normalize_path


def check_response_fields(
    calls: list[ClientCall],
    routes: list[RouteDefinition],
    schema: ServiceSchema,
) -> list[Violation]:
    violations: list[Violation] = []

    # Build route -> response model mapping
    route_response_map: dict[tuple[str, str], str] = {}
    for r in routes:
        if r.response_model:
            key = (normalize_path(r.path), r.method.upper())
            route_response_map[key] = r.response_model

    # Pooled fallback: all response model fields
    all_response_fields: set[str] = set()
    for model in schema.models.values():
        if "response" in model.name.lower():
            for f in model.fields:
                all_response_fields.add(f.name)
    if not all_response_fields:
        for model in schema.models.values():
            for f in model.fields:
                all_response_fields.add(f.name)

    for call in calls:
        if not call.response_reads:
            continue

        norm_path = normalize_path(call.path)
        key = (norm_path, call.method.upper())
        response_model_name = route_response_map.get(key)

        if response_model_name:
            # Route-specific check
            response_model = schema.models.get(response_model_name)
            if response_model:
                model_fields = {f.name for f in response_model.fields}
                for field_name in call.response_reads:
                    if field_name not in model_fields:
                        violations.append(
                            Violation(
                                severity=ViolationSeverity.FAIL,
                                check="response_field",
                                message=(
                                    f"Client reads response field '{field_name}' "
                                    f"but {response_model_name} does not define it"
                                ),
                                client_location=call.caller,
                                service_location=response_model_name,
                            )
                        )
        else:
            # Fallback: pooled check with WARN severity
            for field_name in call.response_reads:
                if field_name not in all_response_fields:
                    violations.append(
                        Violation(
                            severity=ViolationSeverity.WARN,
                            check="response_field",
                            message=(
                                f"Client reads response field '{field_name}' "
                                f"but no response model defines it"
                            ),
                            client_location=call.caller,
                        )
                    )
    return violations
