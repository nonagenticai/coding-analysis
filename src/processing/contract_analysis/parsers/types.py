"""Shared data types for parser outputs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryParam:
    """A query parameter on a route or client call."""

    name: str
    required: bool = True
    type_annotation: str = "str"


@dataclass
class ModelField:
    """A field in a Pydantic model."""

    name: str
    type_annotation: str
    required: bool
    default: str | None = None


@dataclass
class PydanticModel:
    """A parsed Pydantic BaseModel class."""

    name: str
    bases: list[str] = field(default_factory=list)
    fields: list[ModelField] = field(default_factory=list)
    has_model_validator: bool = False


@dataclass
class ServiceSchema:
    """All Pydantic models from a service's schemas.py."""

    models: dict[str, PydanticModel] = field(default_factory=dict)
    parse_errors: list[str] = field(default_factory=list)


@dataclass
class RouteDefinition:
    """A parsed FastAPI or Flask route decorator."""

    method: str
    path: str
    request_model: str | None = None
    response_model: str | None = None
    query_params: list[QueryParam] = field(default_factory=list)
    status_code: int = 200
    line: int = 0  # 1-based line of the decorator/registration; 0 when unknown
    handler: str = ""  # handler function name
    router: str = ""  # router variable the route is registered on
    rule: str = ""  # the decorator's own path, before any router prefix


@dataclass
class ServiceRoutes:
    """All routes from a service's routes.py files."""

    routes: list[RouteDefinition] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    #: Route decorators whose methods cannot be read statically:
    #: ``(line, path or None, expression, reason)``.
    unresolved: list[tuple[int, str | None, str, str]] = field(default_factory=list)


@dataclass
class ClientCall:
    """A parsed HTTP call from a client file."""

    method: str
    path: str
    payload_fields: list[str] = field(default_factory=list)
    response_reads: list[str] = field(default_factory=list)
    caller: str = ""
    payload_opaque: bool = False  # True when json= arg exists but fields can't be extracted
    payload_model: str | None = None  # Model name when payload is e.g. payload.model_dump()
    query_params: list[str] = field(default_factory=list)


@dataclass
class ClientContract:
    """All HTTP calls from a client file."""

    calls: list[ClientCall] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
