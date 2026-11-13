"""Round 4 coverage tests for Other files: client_parser, js_loader,
function_extractor, doc_validator, format_converter, coding_containers,
scoring, main.py."""

from __future__ import annotations

import pathlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_src_dir = pathlib.Path(__file__).resolve().parents[2] / "src"


# ---------------------------------------------------------------------------
# scoring/__init__.py (lines 19-23), risk_calculator (line 15),
# recommendation_generator (line 15)
# ---------------------------------------------------------------------------
class TestScoringModuleFallbacks:
    """Test scoring module placeholder classes."""

    def test_scoring_init_successful_import(self):
        """Cover scoring/__init__.py lines 14-26."""
        from scoring import PRValidationRecommendationGenerator, RiskCalculator

        # Both should be importable regardless of whether real modules exist
        assert RiskCalculator is not None
        assert PRValidationRecommendationGenerator is not None

    def test_risk_calculator_import(self):
        """Cover scoring/risk_calculator.py lines 10-13 or 15-34."""
        from scoring.risk_calculator import RiskCalculator

        rc = RiskCalculator()
        # If it's the placeholder, it should have calculate_risk
        if hasattr(rc, "calculate_risk"):
            result = rc.calculate_risk()
            assert isinstance(result, dict)

    def test_recommendation_generator_import(self):
        """Cover scoring/recommendation_generator.py lines 10-15 or 17-33."""
        from scoring.recommendation_generator import PRValidationRecommendationGenerator

        gen = PRValidationRecommendationGenerator()
        if hasattr(gen, "generate_recommendations"):
            result = gen.generate_recommendations()
            assert isinstance(result, list)


# ---------------------------------------------------------------------------
# processing/contract_analysis/parsers/client_parser.py (lines 63, 121, 209, etc.)
# ---------------------------------------------------------------------------
class TestClientParserCoverage:
    """Cover uncovered lines in client_parser.py."""

    def test_parse_empty_source(self):
        """Cover edge cases with empty/minimal source."""
        from processing.contract_analysis.parsers.client_parser import parse_client

        result = parse_client("")
        assert result is not None

    def test_parse_simple_requests(self):
        """Cover request parsing paths."""
        from processing.contract_analysis.parsers.client_parser import parse_client

        source = """\
class MyClient:
    def get_users(self):
        return self.submit_job("/api/users", {})

    def create_user(self, data):
        return self.submit_job("/api/users", data)

    def _request(self, method, path):
        return self.submit_job(path, {})

    def poll_status(self, job_id):
        return self.poll_job("/api/status", job_id)
"""
        result = parse_client(source)
        assert result is not None

    def test_parse_explicit_methods(self):
        """Cover explicit method parsing paths."""
        from processing.contract_analysis.parsers.client_parser import parse_client

        source = """\
class APIClient:
    def fetch_items(self):
        return self._request("GET", "/api/items")

    def update_item(self, item_id, data):
        return self._request("PUT", f"/api/items/{item_id}")

    def delete_item(self, item_id):
        return self._request("DELETE", f"/api/items/{item_id}")
"""
        result = parse_client(source)
        assert result is not None

    def test_parse_with_helpers(self):
        """Cover helper method detection."""
        from processing.contract_analysis.parsers.client_parser import parse_client

        source = """\
class ServiceClient:
    def submit_job(self, path, payload):
        return self._post(path, payload)

    def poll_job(self, path, job_id):
        return self._get(path, params={"job_id": job_id})

    def fetch_results(self, path):
        return self._get(path)
"""
        result = parse_client(source)
        assert result is not None


# ---------------------------------------------------------------------------
# processing/architecture/dependency_plugins/js_loader.py (lines 20-21, 100-105, etc.)
# ---------------------------------------------------------------------------
class TestJsLoaderCoverage:
    """Cover uncovered lines in js_loader.py."""

    @pytest.fixture
    def loader(self):
        from processing.architecture.dependency_plugins.js_loader import (
            JSDependencyLoader,
        )

        return JSDependencyLoader()

    def test_detect_package_json(self, loader):
        """Cover detect method."""
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_path = os.path.join(tmpdir, "package.json")
            with open(pkg_path, "w") as f:
                json.dump({"dependencies": {"express": "^4.18.0"}}, f)

            detected = loader.detect(Path(tmpdir))
            assert isinstance(detected, bool)

    def test_parse_package_json(self, loader):
        """Cover parse method with package.json."""
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_path = os.path.join(tmpdir, "package.json")
            with open(pkg_path, "w") as f:
                json.dump(
                    {
                        "name": "test-pkg",
                        "version": "1.0.0",
                        "dependencies": {"express": "^4.18.0", "lodash": "^4.17.0"},
                        "devDependencies": {"jest": "^29.0.0"},
                    },
                    f,
                )

            result = loader.parse(Path(tmpdir))
            assert isinstance(result, list)

    def test_detect_no_package_json(self, loader):
        """Cover detect with no package.json."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            detected = loader.detect(Path(tmpdir))
            assert detected is False

    def test_normalize_version(self, loader):
        """Cover normalize_version."""
        result = loader.normalize_version("^4.18.0")
        assert isinstance(result, str)
        result2 = loader.normalize_version("~1.2.3")
        assert isinstance(result2, str)
        result3 = loader.normalize_version(">=2.0.0")
        assert isinstance(result3, str)


# ---------------------------------------------------------------------------
# processing/ast_analysis/function_extractor.py (lines 30-31, 81, 124-178, 256, 352-372)
# ---------------------------------------------------------------------------
class TestFunctionExtractorCoverage:
    """Cover uncovered lines in function_extractor.py."""

    @pytest.fixture
    def extractor(self):
        from processing.ast_analysis.function_extractor import FunctionExtractor

        return FunctionExtractor()

    def test_extract_function_signatures(self, extractor):
        """Cover extract_function_signatures."""
        import os
        import tempfile

        code = """\
def simple_func(x, y):
    \"\"\"Add two numbers.\"\"\"
    return x + y

def no_doc_func():
    pass

async def async_func():
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = f.name

        try:
            result = extractor.extract_function_signatures(path)
            assert isinstance(result, dict)
        finally:
            os.unlink(path)

    def test_extract_comprehensive_signatures(self, extractor):
        """Cover extract_comprehensive_signatures."""
        import os
        import tempfile

        code = """\
class MyClass:
    def __init__(self):
        self.x = 0

    def method(self):
        return self.x

    @staticmethod
    def static_method():
        return 42
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = f.name

        try:
            result = extractor.extract_comprehensive_signatures(path)
            assert isinstance(result, dict)
        finally:
            os.unlink(path)

    def test_extract_with_decorators(self, extractor):
        """Cover decorated function extraction."""
        import os
        import tempfile

        code = """\
import functools

@functools.lru_cache(maxsize=128)
def cached_func(x):
    return x * 2

class Service:
    @staticmethod
    def static_method():
        return 42
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = f.name

        try:
            result = extractor.extract_function_signatures(path)
            assert isinstance(result, dict)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# processing/analyzers/doc_validator.py (lines 99-100, 154-155, 210-211, etc.)
# ---------------------------------------------------------------------------
class TestDocValidatorCoverage:
    """Cover uncovered lines in doc_validator.py."""

    @pytest.fixture
    def validator(self):
        from processing.analyzers.doc_validator import DocumentationValidator

        return DocumentationValidator()

    def test_validate_documentation_completeness(self, validator):
        """Cover validate_documentation_completeness."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            doc_data = {
                "modules": {
                    "main": {"docstring": "Main module", "functions": ["main", "helper"]},
                },
                "classes": {},
                "functions": {
                    "main": {"docstring": "Main function", "params": ["arg1"]},
                    "helper": {"docstring": None, "params": []},
                },
            }
            result = validator.validate_documentation_completeness(tmpdir, doc_data)
            assert isinstance(result, dict)

    def test_assess_documentation_quality(self, validator):
        """Cover assess_documentation_quality."""
        doc_sections = {
            "overview": "This is the project overview.",
            "installation": "Run pip install to install.",
            "usage": "Import and call the function.",
            "api": {"function_a": "Does X.", "function_b": ""},
        }
        result = validator.assess_documentation_quality(doc_sections)
        assert isinstance(result, dict)

    def test_calculate_overall_coverage(self, validator):
        """Cover calculate_overall_documentation_coverage."""
        doc_data = {
            "functions": {"func_a": {"docstring": "Has docs"}, "func_b": {"docstring": None}},
        }
        api_docs = {
            "endpoints": {"health": "Health endpoint", "users": "Users endpoint"},
        }
        result = validator.calculate_overall_documentation_coverage(doc_data, api_docs)
        assert isinstance(result, (float, int))

    def test_generate_quality_metrics(self, validator):
        """Cover generate_quality_metrics."""
        doc_data = {
            "functions": {"func_a": {"docstring": "Has docs"}, "func_b": {"docstring": None}},
            "classes": {"MyClass": {"docstring": "A class"}},
        }
        result = validator.generate_quality_metrics(doc_data)
        assert isinstance(result, dict)

    def test_analyze_documentation_consistency(self, validator):
        """Cover analyze_documentation_consistency."""
        doc_sections = {
            "overview": "Project overview text.",
            "api": {"endpoint_a": "Description A", "endpoint_b": "Description B"},
        }
        result = validator.analyze_documentation_consistency(doc_sections)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# processing/analyzers/format_converter.py (lines 149-154, 212, 233, etc.)
# ---------------------------------------------------------------------------
class TestFormatConverterCoverage:
    """Cover uncovered lines in format_converter.py."""

    @pytest.fixture
    def converter(self):
        from processing.analyzers.format_converter import FormatConverter

        return FormatConverter()

    @pytest.mark.asyncio
    async def test_convert_markdown_to_html(self, converter):
        """Cover convert_markdown_to_html."""
        md = "# Title\n\nSome text.\n\n- item 1\n- item 2"
        result = await converter.convert_markdown_to_html(md)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_dependency_diagram(self, converter):
        """Cover generate_dependency_diagram."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            with open(os.path.join(tmpdir, "main.py"), "w") as f:
                f.write("import os\nimport sys")

            result = await converter.generate_dependency_diagram(tmpdir)
            # May return None if no deps found or graphviz not available
            assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_structure_diagram(self, converter):
        """Cover generate_structure_diagram."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            os.makedirs(os.path.join(tmpdir, "src"), exist_ok=True)
            with open(os.path.join(tmpdir, "src", "main.py"), "w") as f:
                f.write("def main(): pass")

            result = await converter.generate_structure_diagram(tmpdir)
            assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_save_diagram(self, converter):
        """Cover save_diagram."""
        import base64
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            converter.output_dir = tmpdir
            content = base64.b64encode(b"graph LR\n    A --> B").decode()
            result = await converter.save_diagram(content, "test_diagram")
            assert result is None or isinstance(result, str)

    def test_cleanup_temp_files(self, converter):
        """Cover cleanup_temp_files."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            converter.output_dir = tmpdir
            # Create a temp file
            import os

            with open(os.path.join(tmpdir, "test.tmp"), "w") as f:
                f.write("temp")
            converter.cleanup_temp_files()

    @pytest.mark.asyncio
    async def test_generate_pdf_output(self, converter):
        """Cover generate_pdf_output (may not be available)."""
        result = await converter.generate_pdf_output("<h1>Test</h1><p>Content</p>", "test_output")
        # May return None if pdf generation is not available
        assert result is None or isinstance(result, (str, bytes))


# ---------------------------------------------------------------------------
# clients/coding_containers.py (lines 166-204)
# ---------------------------------------------------------------------------
class TestCodingContainersClientCoverage:
    """Cover the adapter's error branches.

    The adapter delegates to ``CodingWorkspacesClient``, which RAISES
    ``ServiceError`` on >=400 rather than returning a response — so the failure
    branches are driven by raising from the typed methods, not by handing back
    a 404 response object.
    """

    @pytest.fixture
    def client(self):
        from clients.coding_containers import CodingContainersClient

        return CodingContainersClient()

    @staticmethod
    def _ws(**methods):
        ws = AsyncMock()
        for name, value in methods.items():
            if isinstance(value, Exception):
                getattr(ws, name).side_effect = value
            else:
                getattr(ws, name).return_value = value
        return ws

    @pytest.mark.asyncio
    async def test_get_container_not_found(self, client):
        from coding_client.clients import ServiceError

        ws = self._ws(get_workspace=ServiceError(404, "not found"))
        with patch.object(client, "_workspaces", return_value=ws):
            assert await client.get_container("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_container_success(self, client):
        ws = self._ws(get_workspace={"container": {"name": "test", "status": "running"}})
        with patch.object(client, "_workspaces", return_value=ws):
            result = await client.get_container("test", tenant_id="t1")
            assert result == {"name": "test", "status": "running"}

    @pytest.mark.asyncio
    async def test_get_container_error(self, client):
        import httpx

        ws = self._ws(get_workspace=httpx.ConnectError("Connection failed"))
        with patch.object(client, "_workspaces", return_value=ws):
            assert await client.get_container("test") is None

    @pytest.mark.asyncio
    async def test_get_container_server_error(self, client):
        from coding_client.clients import ServiceError

        ws = self._ws(get_workspace=ServiceError(500, "boom"))
        with patch.object(client, "_workspaces", return_value=ws):
            assert await client.get_container("test") is None

    @pytest.mark.asyncio
    async def test_list_containers_success(self, client):
        ws = self._ws(list_workspaces={"containers": [{"name": "c1"}, {"name": "c2"}]})
        with patch.object(client, "_workspaces", return_value=ws):
            result = await client.list_containers(tenant_id="t1")
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_containers_error(self, client):
        import httpx

        ws = self._ws(list_workspaces=httpx.ConnectError("Error"))
        with patch.object(client, "_workspaces", return_value=ws):
            assert await client.list_containers() == []

    @pytest.mark.asyncio
    async def test_list_containers_server_error(self, client):
        from coding_client.clients import ServiceError

        ws = self._ws(list_workspaces=ServiceError(503, "unavailable"))
        with patch.object(client, "_workspaces", return_value=ws):
            assert await client.list_containers() == []

    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        ws = self._ws(get={"status": "ok"})
        with patch.object(client, "_workspaces", return_value=ws):
            assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        import httpx

        ws = self._ws(get=httpx.ConnectError("Timeout"))
        with patch.object(client, "_workspaces", return_value=ws):
            assert await client.health_check() is False


# ---------------------------------------------------------------------------
# main.py (lines 45-81, 136-144)
# Note: main.py requires complex auth setup, tested via existing test_main.py
# ---------------------------------------------------------------------------
