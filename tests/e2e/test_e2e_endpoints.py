"""End-to-end tests for coding-analysis service.

These tests hit a live container at localhost:8688 (started via Docker).
They verify response codes, shapes, and key invariants.
"""

from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Health & root
# ---------------------------------------------------------------------------


class TestHealthAndRoot:
    def test_health_returns_200(self, client: httpx.Client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["service"] == "coding-analysis"
        assert data["version"] == "1.0.0"
        assert data["status"] in ("ok", "degraded")

    def test_root_returns_200(self, client: httpx.Client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["service"] == "coding-analysis"
        assert data["health"] == "/health"


# ---------------------------------------------------------------------------
# Analysis capabilities
# ---------------------------------------------------------------------------


class TestAnalysisCapabilities:
    def test_analysis_capabilities(self, client: httpx.Client):
        r = client.get("/api/analysis/capabilities")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        caps = data["capabilities"]
        assert "supported_types" in caps
        assert "supported_languages" in caps
        assert len(caps["supported_languages"]) > 0

    def test_semantic_capabilities(self, client: httpx.Client):
        r = client.get("/api/analysis/semantic/capabilities")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert len(data["semantic_relationship_types"]) > 0
        assert len(data["supported_languages"]) > 0
        assert "python" in data["supported_languages"]

    def test_semantic_patterns(self, client: httpx.Client):
        r = client.get("/api/analysis/semantic/patterns")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert len(data["cross_language_patterns"]) > 0
        assert len(data["architectural_patterns"]) > 0

    def test_complexity_capabilities(self, client: httpx.Client):
        r = client.get("/api/analysis/complexity/capabilities")
        assert r.status_code == 200
        data = r.json()
        assert "supported_languages" in data
        assert "complexity_levels" in data
        assert len(data["supported_languages"]) > 0

    def test_similarity_capabilities(self, client: httpx.Client):
        r = client.get("/api/analysis/similarity/capabilities")
        assert r.status_code == 200
        data = r.json()
        assert "similarity_types" in data
        assert len(data["similarity_types"]) > 0
        assert len(data["supported_languages"]) > 0

    def test_contract_capabilities(self, client: httpx.Client):
        r = client.get("/api/analysis/contracts/capabilities")
        assert r.status_code == 200
        data = r.json()
        assert "checks" in data
        assert len(data["checks"]) > 0


# ---------------------------------------------------------------------------
# Quality endpoints
# ---------------------------------------------------------------------------


class TestQualityEndpoints:
    def test_quality_languages(self, client: httpx.Client):
        r = client.get("/api/quality/languages")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        assert "languages" in data or isinstance(data, list)

    def test_quality_tools_for_python(self, client: httpx.Client):
        r = client.get("/api/quality/python/tools")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, (dict, list))


# ---------------------------------------------------------------------------
# POST endpoints (require container_name that won't exist, expect 4xx/5xx)
# ---------------------------------------------------------------------------


class TestAnalysisPostEndpoints:
    """Test POST endpoints with validation — no real container is available."""

    def test_analyze_missing_fields(self, client: httpx.Client):
        """POST /api/analysis/analyze without required fields should fail with 422."""
        r = client.post("/api/analysis/analyze", json={})
        assert r.status_code in (400, 422)

    def test_batch_analyze_missing_fields(self, client: httpx.Client):
        r = client.post("/api/analysis/batch", json={})
        assert r.status_code in (400, 422)

    def test_complexity_analyze_file_missing_fields(self, client: httpx.Client):
        r = client.post("/api/analysis/complexity/analyze-file", json={})
        assert r.status_code in (400, 422)

    def test_complexity_analyze_batch_missing_fields(self, client: httpx.Client):
        r = client.post("/api/analysis/complexity/analyze-batch", json={})
        assert r.status_code in (400, 422)

    def test_similarity_analyze_missing_fields(self, client: httpx.Client):
        r = client.post("/api/analysis/similarity/analyze", json={})
        assert r.status_code in (400, 422)

    def test_similarity_detect_clones_missing_fields(self, client: httpx.Client):
        r = client.post("/api/analysis/similarity/detect-clones", json={})
        # Can be 400 (no file_paths and no container_name) or 422 (validation)
        assert r.status_code in (400, 422)

    def test_contract_scan_missing_fields(self, client: httpx.Client):
        r = client.post("/api/analysis/contracts/scan", json={})
        assert r.status_code in (400, 422)

    def test_static_call_graph_missing_fields(self, client: httpx.Client):
        r = client.post("/api/analysis/static/call-graph", json={})
        assert r.status_code in (400, 422)

    def test_static_signatures_missing_fields(self, client: httpx.Client):
        r = client.post("/api/analysis/static/signatures", json={})
        assert r.status_code in (400, 422)

    def test_quality_analyze_missing_fields(self, client: httpx.Client):
        r = client.post("/api/quality/analyze", json={})
        assert r.status_code in (400, 422)

    def test_scan_missing_fields(self, client: httpx.Client):
        r = client.post("/api/analysis/scan/scan", json={})
        assert r.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Container-scoped GET endpoints (expect 404 or error for nonexistent containers)
# ---------------------------------------------------------------------------


class TestContainerScopedEndpoints:
    """GET endpoints that reference {container_name} — expect 404/500 for fake containers."""

    def test_architecture_graph_nonexistent(self, client: httpx.Client):
        r = client.get("/api/analysis/architecture/graph/nonexistent-container")
        assert r.status_code in (404, 500)

    def test_architecture_metrics_nonexistent(self, client: httpx.Client):
        r = client.get("/api/analysis/architecture/metrics/nonexistent-container")
        assert r.status_code in (404, 500)

    def test_architecture_patterns_nonexistent(self, client: httpx.Client):
        r = client.get("/api/analysis/architecture/patterns/nonexistent-container")
        assert r.status_code in (404, 500)

    def test_architecture_dependencies_nonexistent(self, client: httpx.Client):
        r = client.get("/api/analysis/architecture/dependencies/nonexistent-container")
        assert r.status_code in (404, 500)

    def test_architecture_components_nonexistent(self, client: httpx.Client):
        r = client.get("/api/analysis/architecture/components/nonexistent-container")
        assert r.status_code in (404, 500)

    def test_architecture_layers_nonexistent(self, client: httpx.Client):
        r = client.get("/api/analysis/architecture/layers/nonexistent-container")
        assert r.status_code in (404, 500)

    # Quality reads are served from stored results scoped by tenant_id (#106),
    # not from a live container lookup: an unknown container is "not_analyzed"
    # (200), never a fabricated score. 500 only if the store is unreachable.
    def test_quality_summary_nonexistent(self, client: httpx.Client):
        r = client.get("/api/quality/nonexistent-container/summary")
        assert r.status_code in (200, 500)
        if r.status_code == 200:
            assert r.json()["status"] == "not_analyzed"

    def test_quality_issues_nonexistent(self, client: httpx.Client):
        r = client.get("/api/quality/nonexistent-container/issues")
        assert r.status_code in (200, 500)
        if r.status_code == 200:
            assert r.json()["status"] == "not_analyzed"

    def test_quality_trend_nonexistent(self, client: httpx.Client):
        r = client.get("/api/quality/nonexistent-container/trend")
        assert r.status_code in (200, 500)
        if r.status_code == 200:
            assert r.json()["trend"] == []

    def test_bug_hunter_status_nonexistent(self, client: httpx.Client):
        r = client.get("/api/bug-hunter/nonexistent-container/status")
        assert r.status_code in (404, 500)

    def test_removed_query_summary_route_is_absent(self, client: httpx.Client):
        # The /api/analysis/query router (always 501, no result store) was removed.
        r = client.get("/api/analysis/query/nonexistent-container/summary")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# OpenAPI schema validation
# ---------------------------------------------------------------------------


class TestOpenAPISchema:
    def test_openapi_json_available(self, client: httpx.Client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert schema["info"]["title"] == "coding-analysis"
        assert schema["info"]["version"] == "1.0.0"

    def test_all_routes_documented(self, client: httpx.Client):
        r = client.get("/openapi.json")
        schema = r.json()
        paths = list(schema["paths"].keys())
        # Expect at least 50 documented endpoints
        assert len(paths) >= 50

    def test_health_in_schema(self, client: httpx.Client):
        r = client.get("/openapi.json")
        schema = r.json()
        assert "/health" in schema["paths"]


# ---------------------------------------------------------------------------
# Semantic analysis POST
# ---------------------------------------------------------------------------


class TestSemanticAnalysisPost:
    def test_analyze_codebase_missing_container(self, client: httpx.Client):
        r = client.post(
            "/api/analysis/semantic/analyze-codebase",
            json={"container_name": "nonexistent"},
        )
        # Should fail with 404 (container not found) or 500
        assert r.status_code in (404, 500)


# ---------------------------------------------------------------------------
# Multi-language analysis
# ---------------------------------------------------------------------------


class TestMultiLanguageAnalysis:
    def test_multi_language_missing_fields(self, client: httpx.Client):
        r = client.post("/api/analysis/multi-language", json={})
        assert r.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Static analysis POST
# ---------------------------------------------------------------------------


class TestStaticAnalysisPost:
    def test_documentation_generation_missing_fields(self, client: httpx.Client):
        r = client.post("/api/analysis/static/documentation", json={})
        assert r.status_code in (400, 422)

    def test_advanced_validation_missing_fields(self, client: httpx.Client):
        r = client.post("/api/analysis/static/advanced-validation", json={})
        assert r.status_code in (400, 422)
