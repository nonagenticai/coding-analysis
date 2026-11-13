"""Violation types for contract checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


def normalize_path(path: str) -> str:
    """Normalize path parameters to {param} and strip trailing slashes."""
    # ``[^{}]`` not ``[^}]``: the latter is quadratic on a long run of ``{``.
    return re.sub(r"\{[^{}]+\}", "{param}", path.rstrip("/"))


class ViolationSeverity(StrEnum):
    FAIL = "FAIL"
    WARN = "WARN"


class RootCause(StrEnum):
    MISSING_PREFIX = "missing_prefix"
    WRONG_METHOD = "wrong_method"
    PATH_RESTRUCTURED = "path_restructured"
    MISSING_ENDPOINT = "missing_endpoint"
    INTERNAL_PATH_CONVENTION = "internal_path_convention"
    MISSING_FIELD = "missing_field"
    UNKNOWN_RESPONSE_FIELD = "unknown_response_field"
    STALE_EXTERNAL_ROUTE = "stale_external_route"
    VERSION_MISMATCH = "version_mismatch"
    EXTRA_PATH_SEGMENT = "extra_path_segment"
    MISSING_QUERY_PARAM = "missing_query_param"
    STATUS_CODE_MISMATCH = "status_code_mismatch"
    SYNTAX_ERROR = "syntax_error"
    FLOW_FIELD_MISSING = "flow_field_missing"
    FLOW_TYPE_MISMATCH = "flow_type_mismatch"
    FLOW_REQUIRED_DROPPED = "flow_required_dropped"


@dataclass
class Violation:
    severity: ViolationSeverity
    check: str
    message: str
    client_location: str = ""
    service_location: str = ""
    closest_match: str | None = None
    root_cause: str = ""
    suggestion: str = ""
