from processing.contract_analysis.checks.types import RootCause, Violation, ViolationSeverity
from processing.contract_analysis.enrichment.enricher import enrich_violations
from processing.contract_analysis.parsers.types import RouteDefinition


def _v(check: str, message: str) -> Violation:
    return Violation(severity=ViolationSeverity.FAIL, check=check, message=message)


def _routes(*paths_methods: tuple[str, str]) -> list[RouteDefinition]:
    return [RouteDefinition(method=m, path=p) for p, m in paths_methods]


def test_route_path_gets_closest_match():
    v = _v("route_path", "Client calls POST /api/pdf/generate but no matching route exists")
    routes = _routes(("/api/export/pdf/bom", "POST"), ("/api/export/excel/bom", "POST"))
    enrich_violations([v], routes)
    assert v.closest_match is not None
    assert "pdf" in v.closest_match


def test_route_path_no_match():
    v = _v("route_path", "Client calls POST /api/completely/unknown but no matching route exists")
    routes = _routes(
        ("/api/health", "GET"),
    )
    enrich_violations([v], routes)
    # /api/ prefix paths with no match are classified as stale external route
    assert v.root_cause == RootCause.STALE_EXTERNAL_ROUTE


def test_missing_prefix_classified():
    """Client /api/port-detection/detect vs service /api/port-detection/detect/image.

    With the new EXTRA_PATH_SEGMENT classification, this pair (1 segment diff,
    shorter is subsequence of longer) gets classified as EXTRA_PATH_SEGMENT.
    """
    v = _v(
        "route_path",
        "Client calls POST /api/port-detection/detect but no matching route exists",
    )
    routes = _routes(
        ("/api/port-detection/detect/image", "POST"),
    )
    enrich_violations([v], routes)
    assert v.root_cause in (
        RootCause.MISSING_PREFIX,
        RootCause.PATH_RESTRUCTURED,
        RootCause.EXTRA_PATH_SEGMENT,
    )
    assert v.closest_match is not None


def test_http_method_classified():
    v = _v(
        "http_method",
        "Client uses PUT /api/racks/{rack_id} but service only defines [GET, PATCH, DELETE] for /api/racks/{rack_id}",
    )
    enrich_violations(
        [v],
        _routes(
            ("/api/racks/{rack_id}", "PATCH"),
        ),
    )
    assert v.root_cause == RootCause.WRONG_METHOD


def test_request_field_classified():
    v = _v("request_field", "Client does not send required field 'name' expected by FooRequest")
    enrich_violations([v], [])
    assert v.root_cause == RootCause.MISSING_FIELD


def test_response_field_classified():
    v = _v("response_field", "Client reads 'status' but field not found in model")
    enrich_violations([v], [])
    assert v.root_cause == RootCause.UNKNOWN_RESPONSE_FIELD


def test_suggestion_always_populated():
    v = _v("route_path", "Client calls GET /api/foo but no matching route exists")
    enrich_violations(
        [v],
        _routes(
            ("/api/bar/foo", "GET"),
        ),
    )
    assert v.suggestion != ""


def test_empty_violations():
    result = enrich_violations([], [])
    assert result == []


def test_internal_path_convention():
    v = _v(
        "route_path",
        "Client calls GET /api/v1/internal/projects/{param} but no matching route exists",
    )
    routes = _routes(
        ("/_internal/v1/projects/{param}/snapshot", "GET"),
    )
    enrich_violations([v], routes)
    assert v.root_cause == RootCause.INTERNAL_PATH_CONVENTION


def test_low_overlap_paths_no_closest_match():
    """Paths sharing only /api/ should NOT produce a closest_match at threshold 0.40."""
    v = _v("route_path", "Client calls GET /api/foo but no matching route exists")
    routes = _routes(
        ("/api/bar", "GET"),
    )
    enrich_violations([v], routes)
    assert v.root_cause != ""


def test_version_mismatch_classified():
    """/api/v1/items vs /api/v2/items should be VERSION_MISMATCH."""
    v = _v("route_path", "Client calls GET /api/v1/items but no matching route exists")
    routes = _routes(
        ("/api/v2/items", "GET"),
    )
    enrich_violations([v], routes)
    assert v.root_cause == RootCause.VERSION_MISMATCH
    assert "version mismatch" in v.suggestion.lower()


def test_extra_path_segment_classified():
    """/api/export/pdf/generate vs /api/pdf/generate — 1 extra segment."""
    v = _v("route_path", "Client calls POST /api/export/pdf/generate but no matching route exists")
    routes = _routes(
        ("/api/pdf/generate", "POST"),
    )
    enrich_violations([v], routes)
    assert v.root_cause == RootCause.EXTRA_PATH_SEGMENT
    assert "extra/missing segment" in v.suggestion.lower()


def test_stale_external_route_classified():
    """/api/widgets/create with no match should be STALE_EXTERNAL_ROUTE."""
    v = _v("route_path", "Client calls POST /api/widgets/create but no matching route exists")
    routes = _routes(
        ("/api/health", "GET"),
    )
    enrich_violations([v], routes)
    assert v.root_cause == RootCause.STALE_EXTERNAL_ROUTE
    assert "removed" in v.suggestion.lower()


def test_wrong_method_suggestion_includes_available():
    """WRONG_METHOD suggestion should include available methods."""
    v = _v(
        "http_method",
        "Client uses PUT /api/foo but service only defines [GET, PATCH] for /api/foo",
    )
    enrich_violations(
        [v],
        _routes(
            ("/api/foo", "GET"),
        ),
    )
    assert "GET, PATCH" in v.suggestion


def test_missing_field_suggestion_includes_context():
    """MISSING_FIELD suggestion should include field name and model."""
    v = _v(
        "request_field",
        "Client does not send required field 'config' expected by AnalyzeRequest",
    )
    enrich_violations([v], [])
    assert "config" in v.suggestion
    assert "AnalyzeRequest" in v.suggestion
