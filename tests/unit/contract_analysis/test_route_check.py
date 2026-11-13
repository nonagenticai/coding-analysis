from processing.contract_analysis.checks.route_check import check_routes
from processing.contract_analysis.checks.types import ViolationSeverity
from processing.contract_analysis.parsers.types import ClientCall, RouteDefinition


def test_matching_routes_no_violations():
    calls = [ClientCall(method="POST", path="/api/analyze", caller="start_analysis()")]
    routes = [RouteDefinition(method="POST", path="/api/analyze")]
    violations = check_routes(calls, routes)
    assert len(violations) == 0


def test_missing_route():
    calls = [ClientCall(method="POST", path="/api/missing", caller="do_thing()")]
    routes = [RouteDefinition(method="POST", path="/api/analyze")]
    violations = check_routes(calls, routes)
    assert len(violations) == 1
    assert violations[0].severity == ViolationSeverity.FAIL
    assert violations[0].check == "route_path"
    assert "/api/missing" in violations[0].message


def test_method_mismatch():
    calls = [ClientCall(method="POST", path="/api/analyze", caller="start_analysis()")]
    routes = [RouteDefinition(method="GET", path="/api/analyze")]
    violations = check_routes(calls, routes)
    assert len(violations) == 1
    assert violations[0].check == "http_method"


def test_path_param_normalization():
    calls = [ClientCall(method="GET", path="/api/analyze/{job_id}", caller="get_analysis()")]
    routes = [RouteDefinition(method="GET", path="/api/analyze/{job_id}")]
    violations = check_routes(calls, routes)
    assert len(violations) == 0


def test_method_mismatch_shows_original_paths():
    """Method mismatch message includes original route paths with param names."""
    calls = [ClientCall(method="DELETE", path="/users/{id}", caller="delete()")]
    routes = [
        RouteDefinition(method="GET", path="/users/{user_id}"),
        RouteDefinition(method="POST", path="/users/{username}"),
    ]
    violations = check_routes(calls, routes)
    assert len(violations) == 1
    msg = violations[0].message
    assert "DELETE" in msg
    assert "[GET, POST]" in msg
    # Original paths with real param names should be in the message
    assert "/users/{user_id}" in msg or "/users/{username}" in msg


def test_wildcard_path_matches_concrete_route():
    """Client f-string path /api/{param}/health matches route /api/layout/health."""
    calls = [ClientCall(method="GET", path="/api/{param}/health", caller="check()")]
    routes = [RouteDefinition(method="GET", path="/api/layout/health")]
    violations = check_routes(calls, routes)
    assert len(violations) == 0


def test_wildcard_no_match_still_fails():
    """Wildcard path that doesn't match any route produces violation."""
    calls = [ClientCall(method="GET", path="/api/{param}/missing", caller="check()")]
    routes = [RouteDefinition(method="GET", path="/api/layout/health")]
    violations = check_routes(calls, routes)
    assert len(violations) == 1
    assert violations[0].check == "route_path"


def test_exact_match_preferred_over_wildcard():
    """When exact match exists, wildcard is not needed."""
    calls = [ClientCall(method="GET", path="/api/items/{param}", caller="get()")]
    routes = [RouteDefinition(method="GET", path="/api/items/{id}")]
    violations = check_routes(calls, routes)
    assert len(violations) == 0


def test_wrapper_path_skipped():
    """@wrapper/scan paths produce no violation."""
    calls = [ClientCall(method="POST", path="@wrapper/scan", caller="scan()")]
    routes = [RouteDefinition(method="GET", path="/api/health")]
    violations = check_routes(calls, routes)
    assert len(violations) == 0


def test_fstring_path_with_multiple_params():
    """/api/{param}/items/{param} matches /api/users/items/123."""
    calls = [ClientCall(method="GET", path="/api/{param}/items/{param}", caller="get()")]
    routes = [RouteDefinition(method="GET", path="/api/users/items/123")]
    violations = check_routes(calls, routes)
    assert len(violations) == 0
