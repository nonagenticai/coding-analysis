from processing.contract_analysis.checks.query_check import check_query_params
from processing.contract_analysis.checks.types import ViolationSeverity
from processing.contract_analysis.parsers.types import ClientCall, QueryParam, RouteDefinition


def test_no_query_params_no_violations():
    calls = [ClientCall(method="GET", path="/api/items", caller="get()")]
    routes = [RouteDefinition(method="GET", path="/api/items")]
    violations = check_query_params(calls, routes)
    assert len(violations) == 0


def test_required_query_param_missing():
    calls = [ClientCall(method="GET", path="/api/items", caller="get()")]
    routes = [
        RouteDefinition(
            method="GET",
            path="/api/items",
            query_params=[QueryParam(name="page", required=True)],
        )
    ]
    violations = check_query_params(calls, routes)
    assert len(violations) == 1
    assert violations[0].severity == ViolationSeverity.FAIL
    assert "page" in violations[0].message


def test_required_query_param_sent():
    calls = [ClientCall(method="GET", path="/api/items", caller="get()", query_params=["page"])]
    routes = [
        RouteDefinition(
            method="GET",
            path="/api/items",
            query_params=[QueryParam(name="page", required=True)],
        )
    ]
    violations = check_query_params(calls, routes)
    assert len(violations) == 0


def test_optional_query_param_not_flagged():
    calls = [ClientCall(method="GET", path="/api/items", caller="get()")]
    routes = [
        RouteDefinition(
            method="GET",
            path="/api/items",
            query_params=[QueryParam(name="page", required=False)],
        )
    ]
    violations = check_query_params(calls, routes)
    assert len(violations) == 0
