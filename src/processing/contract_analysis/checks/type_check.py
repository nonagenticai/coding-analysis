"""Type compatibility checker (WARN level only).

Detects type mismatches between client payload models and service schema models.
When a client uses ``payload.model_dump()`` with a known Pydantic model, this checker
compares the field types of the client model against the service request model. If the
same field name has incompatible type annotations, a WARN violation is emitted.

For plain dict payloads (payload_fields without payload_model), type checking is not
possible since we only know the field *names*, not their types.
"""

from __future__ import annotations

from ..parsers.types import ClientCall, ModelField, PydanticModel, ServiceSchema
from .types import Violation, ViolationSeverity

# Groups of types considered compatible with each other.
# If both the client type and the service type belong to the same group, no warning.
_COMPATIBLE_GROUPS: list[set[str]] = [
    {"str", "Optional[str]", "str | None"},
    {"int", "Optional[int]", "int | None", "float", "Optional[float]", "float | None"},
    {"bool", "Optional[bool]", "bool | None"},
    {"list", "List", "Sequence"},
    {"dict", "Dict", "Mapping"},
    {"Any"},
]


def _normalize_type(annotation: str) -> str:
    """Strip whitespace and outer Optional wrapper for comparison."""
    t = annotation.strip()
    # Normalise Optional[X] -> X | None
    if t.startswith("Optional[") and t.endswith("]"):
        inner = t[len("Optional[") : -1]
        t = f"{inner} | None"
    return t


def _base_type(annotation: str) -> str:
    """Extract the outermost type name for group lookup.

    ``list[str]`` -> ``list``, ``dict[str, int]`` -> ``dict``, ``str`` -> ``str``.
    Handles ``X | None`` by stripping the None alternative.
    """
    t = _normalize_type(annotation)
    # Strip "| None"
    parts = [p.strip() for p in t.split("|")]
    non_none = [p for p in parts if p != "None"]
    t = non_none[0] if non_none else "None"
    # Strip generic params
    bracket = t.find("[")
    if bracket != -1:
        t = t[:bracket]
    return t


def _types_compatible(client_type: str, service_type: str) -> bool:
    """Return True when two type annotations are considered compatible."""
    ct = _normalize_type(client_type)
    st = _normalize_type(service_type)

    # Exact match after normalisation
    if ct == st:
        return True

    # ``Any`` is compatible with everything
    if "Any" in (ct, st):
        return True

    cb = _base_type(client_type)
    sb = _base_type(service_type)

    # Same base type (e.g. both ``list`` even if generics differ)
    if cb == sb:
        return True

    # Check compatibility groups
    for group in _COMPATIBLE_GROUPS:
        if cb in group and sb in group:
            return True

    return False


def _find_request_models(schema: ServiceSchema) -> list[PydanticModel]:
    """Return models that look like request schemas (name heuristic)."""
    result: list[PydanticModel] = []
    for model in schema.models.values():
        lower = model.name.lower()
        if "request" in lower or "create" in lower or "update" in lower or "input" in lower:
            result.append(model)
    return result


def _find_response_models(schema: ServiceSchema) -> list[PydanticModel]:
    """Return models that look like response schemas (name heuristic)."""
    result: list[PydanticModel] = []
    for model in schema.models.values():
        lower = model.name.lower()
        if "response" in lower or "result" in lower or "output" in lower:
            result.append(model)
    return result


def _build_field_map(model: PydanticModel) -> dict[str, ModelField]:
    """Build name -> ModelField lookup for a model."""
    return {f.name: f for f in model.fields}


# Heuristic: common field names mapped to expected type annotations.
# Used in Phase 3 to flag suspicious types (e.g. "count" as str).
_NAME_TYPE_HINTS: dict[str, set[str]] = {
    "count": {"int", "float", "Optional[int]", "int | None"},
    "page": {"int", "Optional[int]", "int | None"},
    "limit": {"int", "Optional[int]", "int | None"},
    "offset": {"int", "Optional[int]", "int | None"},
    "size": {"int", "float", "Optional[int]", "int | None"},
    "timeout": {
        "int",
        "float",
        "Optional[float]",
        "float | None",
        "Optional[int]",
        "int | None",
    },
    "enabled": {"bool", "Optional[bool]", "bool | None"},
    "active": {"bool", "Optional[bool]", "bool | None"},
    "is_valid": {"bool", "Optional[bool]", "bool | None"},
    "verbose": {"bool", "Optional[bool]", "bool | None"},
}


def check_types(
    calls: list[ClientCall],
    schema: ServiceSchema,
) -> list[Violation]:
    """Check for type mismatches between client models and service schemas.

    This produces WARN-level violations only, because static type inference
    from source code is inherently limited.
    """
    violations: list[Violation] = []

    if not schema.models:
        return violations

    # -- 1. payload_model checks -------------------------------------------
    # When the client uses json=payload.model_dump() and we know the model
    # type, compare its fields against all request-like service models.
    request_models = _find_request_models(schema)

    for call in calls:
        if not call.payload_model:
            continue
        client_model = schema.models.get(call.payload_model)
        if client_model is None:
            continue

        client_fields = _build_field_map(client_model)

        for svc_model in request_models:
            if svc_model.name == client_model.name:
                continue  # same model, skip self-comparison
            svc_fields = _build_field_map(svc_model)
            for fname, cf in client_fields.items():
                sf = svc_fields.get(fname)
                if sf is None:
                    continue
                if not _types_compatible(cf.type_annotation, sf.type_annotation):
                    violations.append(
                        Violation(
                            severity=ViolationSeverity.WARN,
                            check="type_mismatch",
                            message=(
                                f"Field '{fname}' type mismatch: client model "
                                f"{client_model.name} has '{cf.type_annotation}' "
                                f"but service model {svc_model.name} expects "
                                f"'{sf.type_annotation}'"
                            ),
                            client_location=call.caller,
                            service_location=svc_model.name,
                        )
                    )

    # -- 2. response_reads field type cross-check --------------------------
    # When two response models define the same field name with different types,
    # emit a warning if the client reads that field.
    response_models = _find_response_models(schema)
    # Fallback: if no response models found by name, use all models
    if not response_models:
        response_models = list(schema.models.values())

    # Build field_name -> set of (model_name, type_annotation)
    field_types: dict[str, list[tuple[str, str]]] = {}
    for model in response_models:
        for f in model.fields:
            field_types.setdefault(f.name, []).append((model.name, f.type_annotation))

    for call in calls:
        for field_name in call.response_reads:
            entries = field_types.get(field_name)
            if entries is None or len(entries) < 2:
                continue
            # Check if any pair is incompatible
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    m1, t1 = entries[i]
                    m2, t2 = entries[j]
                    if not _types_compatible(t1, t2):
                        violations.append(
                            Violation(
                                severity=ViolationSeverity.WARN,
                                check="type_mismatch",
                                message=(
                                    f"Response field '{field_name}' has conflicting types: "
                                    f"{m1} defines '{t1}' but {m2} defines '{t2}'"
                                ),
                                client_location=call.caller,
                            )
                        )

    # -- 3. Payload field names vs request model type hints ----------------
    # When client sends payload_fields (dict literal keys), check if a
    # matching field in any request model has an obviously wrong type
    # for a "common name" heuristic (e.g. field named "count" with type
    # "str" is suspicious when service expects "int").
    for call in calls:
        if not call.payload_fields or call.payload_model:
            continue  # payload_model already handled above

        for svc_model in request_models:
            svc_fields = _build_field_map(svc_model)
            for fname in call.payload_fields:
                sf = svc_fields.get(fname)
                if sf is None:
                    continue
                expected_types = _NAME_TYPE_HINTS.get(fname)
                if expected_types is None:
                    continue
                base = _base_type(sf.type_annotation)
                if base not in expected_types and sf.type_annotation not in expected_types:
                    violations.append(
                        Violation(
                            severity=ViolationSeverity.WARN,
                            check="type_mismatch",
                            message=(
                                f"Field '{fname}' in {svc_model.name} has type "
                                f"'{sf.type_annotation}' which seems wrong for a "
                                f"field named '{fname}' (expected numeric/bool type)"
                            ),
                            client_location=call.caller,
                            service_location=svc_model.name,
                        )
                    )

    return violations
