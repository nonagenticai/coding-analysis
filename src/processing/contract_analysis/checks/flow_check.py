"""Cross-service data-flow field compatibility check."""

from __future__ import annotations

from ..flow.registry import FlowEdge
from ..parsers.types import ServiceSchema
from .types import RootCause, Violation, ViolationSeverity


def check_flow(
    edge: FlowEdge,
    producer_schema: ServiceSchema,
    consumer_schema: ServiceSchema,
) -> list[Violation]:
    """Compare producer model fields with consumer model fields."""
    violations: list[Violation] = []

    producer_model = producer_schema.models.get(edge.producer_model)
    consumer_model = consumer_schema.models.get(edge.consumer_model)

    if producer_model is None or consumer_model is None:
        missing = edge.producer_model if producer_model is None else edge.consumer_model
        violations.append(
            Violation(
                severity=ViolationSeverity.WARN,
                check="flow",
                message=f"[{edge.label}] Model '{missing}' not found in schema",
            )
        )
        return violations

    producer_fields = {f.name: f for f in producer_model.fields}
    consumer_fields = {f.name: f for f in consumer_model.fields}

    for name, cf in consumer_fields.items():
        # If forwarded_fields specified, skip fields not forwarded
        if edge.forwarded_fields is not None and name not in edge.forwarded_fields:
            if cf.required:
                violations.append(
                    Violation(
                        severity=ViolationSeverity.FAIL,
                        check="flow",
                        message=(
                            f"[{edge.label}] Required field '{name}' in "
                            f"{edge.consumer_model} is not forwarded by handler"
                        ),
                        root_cause=RootCause.FLOW_REQUIRED_DROPPED,
                        suggestion=f"Forward '{name}' or make it optional in {edge.consumer_model}",
                    )
                )
            continue

        if name not in producer_fields:
            severity = ViolationSeverity.FAIL if cf.required else ViolationSeverity.WARN
            violations.append(
                Violation(
                    severity=severity,
                    check="flow",
                    message=(
                        f"[{edge.label}] {'Required' if cf.required else 'Optional'} "
                        f"field '{name}' in {edge.consumer_model} has no equivalent "
                        f"in {edge.producer_model}"
                    ),
                    root_cause=RootCause.FLOW_FIELD_MISSING,
                    suggestion=(
                        f"Add '{name}' to {edge.producer_model} or "
                        f"make it optional in {edge.consumer_model}"
                    ),
                )
            )
            continue

        # Type compatibility
        pf = producer_fields[name]
        compat, reason = _types_compatible(pf.type_annotation, cf.type_annotation)
        if not compat:
            violations.append(
                Violation(
                    severity=ViolationSeverity.FAIL,
                    check="flow_type",
                    message=(
                        f"[{edge.label}] Field '{name}': producer has "
                        f"'{pf.type_annotation}' but consumer expects "
                        f"'{cf.type_annotation}' -- {reason}"
                    ),
                    root_cause=RootCause.FLOW_TYPE_MISMATCH,
                    suggestion=f"Reconcile type of '{name}' between models",
                )
            )
        elif reason:
            violations.append(
                Violation(
                    severity=ViolationSeverity.WARN,
                    check="flow_type",
                    message=(
                        f"[{edge.label}] Field '{name}': producer has "
                        f"'{pf.type_annotation}', consumer expects "
                        f"'{cf.type_annotation}' -- {reason}"
                    ),
                    root_cause=RootCause.FLOW_TYPE_MISMATCH,
                )
            )

    return violations


def _types_compatible(producer: str, consumer: str) -> tuple[bool, str]:
    """Check type compatibility. Returns (is_compatible, reason).

    Empty reason = fully compatible. Non-empty reason with True = WARN.
    """
    p, c = producer.strip(), consumer.strip()
    if p == c:
        return True, ""
    # StrEnum / enum -> str: serializes fine
    if c == "str" and _looks_like_enum(p):
        return True, f"enum '{p}' serializes as str but loses type safety"
    # str -> list[str]: structural mismatch
    if c.startswith("list[") and not p.startswith("list["):
        return False, f"'{p}' cannot be coerced to '{c}'"
    # Nested model -> dict: loses validation
    if c == "dict" and _looks_like_model_name(p):
        return True, f"typed model '{p}' passed as untyped dict"
    if c.startswith("list[dict]") and p.startswith("list[") and _looks_like_model_name(p[5:-1]):
        return True, f"typed list '{p}' passed as list[dict]"
    # list[X] vs list[Y]: compare inner types
    if p.startswith("list[") and c.startswith("list["):
        return _types_compatible(p[5:-1], c[5:-1])
    # Optional handling
    if "| None" in p:
        bare_p = p.replace(" | None", "").strip()
        return _types_compatible(bare_p, c)
    if p.startswith("Optional["):
        bare_p = p[9:-1]
        return _types_compatible(bare_p, c)
    # Different model names
    if _looks_like_model_name(p) and _looks_like_model_name(c) and p != c:
        return False, f"different model types: '{p}' vs '{c}'"
    return False, f"type '{p}' is not compatible with '{c}'"


def _looks_like_enum(annotation: str) -> bool:
    """Heuristic: names ending in Type/Status/Kind are likely enums."""
    if not annotation or not annotation[0].isupper():
        return False
    return any(annotation.endswith(suffix) for suffix in ("Type", "Status", "Kind", "Enum"))


def _looks_like_model_name(annotation: str) -> bool:
    """Heuristic: PascalCase names that aren't builtins are likely models."""
    if not annotation or not annotation[0].isupper():
        return False
    return annotation not in ("Optional", "None", "Any")
