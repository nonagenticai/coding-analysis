from processing.contract_analysis.checks.status_check import check_status_codes
from processing.contract_analysis.parsers.types import ClientCall, RouteDefinition


def test_status_200_no_warning():
    calls = [ClientCall(method="POST", path="/api/foo", caller="create()")]
    routes = [RouteDefinition(method="POST", path="/api/foo", status_code=200)]
    violations = check_status_codes(calls, routes)
    assert len(violations) == 0


def test_standard_rest_codes_not_flagged():
    """201, 202, 204 are standard REST — no warnings for job-based APIs."""
    calls = [
        ClientCall(method="POST", path="/api/foo", caller="client"),
        ClientCall(method="DELETE", path="/api/foo/{id}", caller="client"),
    ]
    routes = [
        RouteDefinition(method="POST", path="/api/foo", status_code=202),
        RouteDefinition(method="DELETE", path="/api/foo/{id}", status_code=204),
    ]
    violations = check_status_codes(calls, routes)
    assert len(violations) == 0


def test_unexpected_status_code_flagged():
    """Non-standard status codes (e.g., 301, 409) should still warn."""
    calls = [ClientCall(method="GET", path="/api/foo", caller="client")]
    routes = [RouteDefinition(method="GET", path="/api/foo", status_code=301)]
    violations = check_status_codes(calls, routes)
    assert len(violations) == 1


def test_status_code_default_200():
    """Routes without explicit status_code default to 200 (no warning)."""
    calls = [ClientCall(method="GET", path="/api/foo", caller="get()")]
    routes = [RouteDefinition(method="GET", path="/api/foo")]
    violations = check_status_codes(calls, routes)
    assert len(violations) == 0
