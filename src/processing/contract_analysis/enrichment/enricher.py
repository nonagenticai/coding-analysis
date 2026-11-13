"""Post-process violations with closest-match routes, root cause classification, and fix suggestions."""

from __future__ import annotations

import re

from ..checks.types import RootCause, Violation, normalize_path
from ..parsers.types import RouteDefinition
from .path_similarity import _normalize_and_split, segment_overlap_score

MATCH_THRESHOLD = 0.40


def enrich_violations(
    violations: list[Violation],
    service_routes: list[RouteDefinition],
) -> list[Violation]:
    """Enrich violations with closest_match, root_cause, and suggestion."""
    route_map: dict[str, list[str]] = {}
    for r in service_routes:
        key = normalize_path(r.path)
        route_map.setdefault(key, []).append(r.method.upper())

    route_paths = list(route_map.keys())

    for v in violations:
        if v.check == "route_path":
            client_path = _extract_client_path(v.message)
            closest, score = _find_closest_route(client_path, route_paths)
            v.closest_match = closest
            v.root_cause = _classify_route_path(client_path, closest, score)
        elif v.check == "http_method":
            v.closest_match = _extract_client_path(v.message)
            v.root_cause = RootCause.WRONG_METHOD
        elif v.check == "request_field":
            v.root_cause = RootCause.MISSING_FIELD
        elif v.check == "response_field":
            v.root_cause = RootCause.UNKNOWN_RESPONSE_FIELD
        elif v.check == "query_param":
            v.root_cause = RootCause.MISSING_QUERY_PARAM
        elif v.check == "status_code":
            v.root_cause = RootCause.STATUS_CODE_MISMATCH
        elif v.check == "python3_syntax":
            v.root_cause = RootCause.SYNTAX_ERROR

        v.suggestion = _generate_suggestion(v)

    return violations


def _extract_client_path(message: str) -> str:
    """Extract client path from violation message."""
    m = re.search(r"Client (?:calls|uses) \w+ (\S+)", message)
    return m.group(1) if m else ""


def _find_closest_route(client_path: str, route_paths: list[str]) -> tuple[str | None, float]:
    """Find the service route with highest segment overlap."""
    best_path: str | None = None
    best_score = 0.0
    for service_path in route_paths:
        score = segment_overlap_score(client_path, service_path)
        if score > best_score:
            best_score = score
            best_path = service_path
    if best_score < MATCH_THRESHOLD:
        return None, 0.0
    return best_path, best_score


def _has_version_mismatch(client_path: str, closest: str) -> bool:
    """Check if paths differ by API version prefix (e.g., /v1/ vs /v2/)."""
    version_re = re.compile(r"/v(\d+)/")
    client_versions = version_re.findall(client_path)
    closest_versions = version_re.findall(closest)
    if client_versions and closest_versions:
        return client_versions[0] != closest_versions[0]
    return False


def _is_extra_path_segment(client_segs: list[str], match_segs: list[str]) -> bool:
    """Check if paths differ by exactly 1 segment and shorter is subsequence of longer."""
    if abs(len(client_segs) - len(match_segs)) != 1:
        return False
    shorter, longer = (
        (client_segs, match_segs)
        if len(client_segs) < len(match_segs)
        else (match_segs, client_segs)
    )
    # Check if shorter is a subsequence of longer
    j = 0
    for seg in longer:
        if j < len(shorter) and seg == shorter[j]:
            j += 1
    return j == len(shorter)


def _classify_route_path(client_path: str, closest: str | None, score: float) -> str:
    """Classify the root cause of a route_path violation."""
    if _is_internal_convention(client_path, closest):
        return RootCause.INTERNAL_PATH_CONVENTION

    if closest is None:
        # No match at all -- check if it looks like a well-formed API path
        if re.match(r"^/api/", client_path):
            return RootCause.STALE_EXTERNAL_ROUTE
        return RootCause.MISSING_ENDPOINT

    # Version mismatch check (before other structural checks)
    if _has_version_mismatch(client_path, closest):
        return RootCause.VERSION_MISMATCH

    client_segs = _normalize_and_split(client_path)
    match_segs = _normalize_and_split(closest)

    # Check suffix relationship (missing prefix)
    if len(client_segs) < len(match_segs):
        if match_segs[-len(client_segs) :] == client_segs:
            return RootCause.MISSING_PREFIX
    if len(match_segs) < len(client_segs):
        if client_segs[-len(match_segs) :] == match_segs:
            return RootCause.MISSING_PREFIX

    # Extra/missing path segment
    if _is_extra_path_segment(client_segs, match_segs):
        return RootCause.EXTRA_PATH_SEGMENT

    # Same terminal segment -> path was restructured
    if client_segs and match_segs and client_segs[-1] == match_segs[-1]:
        return RootCause.PATH_RESTRUCTURED

    if score >= MATCH_THRESHOLD:
        return RootCause.PATH_RESTRUCTURED

    return RootCause.MISSING_ENDPOINT


def _is_internal_convention(client_path: str, closest: str | None) -> bool:
    """Detect /api/v1/internal/... vs /_internal/v1/... mismatch."""
    if closest is None:
        return False
    client_internal = "/internal/" in client_path or client_path.startswith("/_internal")
    match_internal = "/internal/" in closest or closest.startswith("/_internal")
    return client_internal and match_internal


def _generate_suggestion(violation: Violation) -> str:
    """Generate a fix suggestion based on root cause and violation context."""
    root_cause = violation.root_cause
    closest_match = violation.closest_match
    message = violation.message

    if root_cause == RootCause.WRONG_METHOD:
        # Extract available methods from message
        m = re.search(r"only defines \[([^\]]+)\]", message)
        if m:
            return f"Change HTTP method to one of [{m.group(1)}]"
        return "Change HTTP method to match service definition"

    if root_cause == RootCause.MISSING_FIELD:
        # Extract field name and model from message
        m = re.search(r"field '(\w+)' expected by (\w+)", message)
        if m:
            return (
                f"Add required field '{m.group(1)}' "
                f"(expected by {m.group(2)}) to the request payload"
            )
        return "Add the missing field to the client's request payload"

    if root_cause == RootCause.VERSION_MISMATCH:
        return f"API version mismatch -- update client path to '{closest_match}'"

    if root_cause == RootCause.EXTRA_PATH_SEGMENT:
        return f"Path has extra/missing segment -- closest is '{closest_match}'"

    if root_cause == RootCause.STALE_EXTERNAL_ROUTE:
        return "Route may have been removed -- verify with service maintainer"

    # Fallback to template-based suggestions
    templates: dict[str, str] = {
        RootCause.MISSING_PREFIX: "Update client path to '{closest_match}'",
        RootCause.PATH_RESTRUCTURED: "Update client path -- closest service route is '{closest_match}'",
        RootCause.MISSING_ENDPOINT: "Endpoint not found in service -- remove client call or add the endpoint",
        RootCause.INTERNAL_PATH_CONVENTION: "Fix internal path convention -- service uses '{closest_match}'",
        RootCause.UNKNOWN_RESPONSE_FIELD: "Check field name -- service schema may have changed",
        RootCause.MISSING_QUERY_PARAM: "Add the missing query parameter to the client request",
        RootCause.STATUS_CODE_MISMATCH: "Route returns non-200 status -- ensure client handles this correctly",
        RootCause.SYNTAX_ERROR: "Fix Python 3 syntax error -- module cannot be imported at runtime",
    }
    template = templates.get(root_cause, "Review violation details")
    return template.format(closest_match=closest_match or "N/A")
