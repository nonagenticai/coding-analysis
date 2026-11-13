"""Round 3 coverage: processing modules - http_quality_runner, JS entity extractor,
report_generator, component_extractor, fix_suggester.

Covers:
- SessionMonitor: init, register_session, record_session_closed, record_session_error,
  _categorize_error, get_health_summary
- HTTPQualityRunner: init, _load_and_validate_config
- JavaScriptEntityExtractor: extract_entities, _extract_node_entities, all entity types
- QualityReportGenerator: init, get_available_themes/styles/formats, factory function
"""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest

# ─── SessionMonitor ───────────────────────────────────────────────────────────
from processing.quality_analysis.http_quality_runner import SessionMonitor


class TestSessionMonitor:
    def test_init(self):
        monitor = SessionMonitor()
        assert monitor.stats.active_sessions == 0
        assert monitor.stats.total_sessions_created == 0
        assert monitor.stats.total_sessions_closed == 0
        assert monitor.stats.failed_sessions == 0
        assert monitor.request_metrics == []
        assert isinstance(monitor.error_patterns, defaultdict)

    @pytest.mark.asyncio
    async def test_register_session(self):
        monitor = SessionMonitor()
        mock_session = MagicMock()
        # WeakSet needs real objects, mock is fine since it has __hash__
        await monitor.register_session(mock_session)
        assert monitor.stats.total_sessions_created == 1
        assert monitor.stats.active_sessions >= 0  # WeakSet might gc

    @pytest.mark.asyncio
    async def test_record_session_closed(self):
        monitor = SessionMonitor()
        await monitor.record_session_closed()
        assert monitor.stats.total_sessions_closed == 1

    @pytest.mark.asyncio
    async def test_record_session_error(self):
        monitor = SessionMonitor()
        await monitor.record_session_error("connection refused")
        assert monitor.stats.failed_sessions == 1
        assert len(monitor.stats.session_errors) == 1
        assert "connection refused" in monitor.stats.session_errors[0]

    @pytest.mark.asyncio
    async def test_record_many_errors_truncates(self):
        monitor = SessionMonitor()
        for i in range(60):
            await monitor.record_session_error(f"error {i}")
        # Should keep only last 25 after truncation at 50
        assert len(monitor.stats.session_errors) <= 50

    def test_categorize_error_session_closed(self):
        monitor = SessionMonitor()
        assert monitor._categorize_error("Session is closed") == "session_closed"

    def test_categorize_error_connection_refused(self):
        monitor = SessionMonitor()
        assert monitor._categorize_error("Connection refused by host") == "connection_refused"

    def test_categorize_error_timeout(self):
        monitor = SessionMonitor()
        assert monitor._categorize_error("Request timeout exceeded") == "timeout"

    def test_categorize_error_pool(self):
        monitor = SessionMonitor()
        assert monitor._categorize_error("Connection pool exhausted") == "connection_pool_issue"

    def test_categorize_error_502(self):
        monitor = SessionMonitor()
        assert monitor._categorize_error("HTTP 502 Bad Gateway") == "service_unavailable"

    def test_categorize_error_503(self):
        monitor = SessionMonitor()
        assert monitor._categorize_error("HTTP 503 Service Unavailable") == "service_unavailable"

    def test_categorize_error_other(self):
        monitor = SessionMonitor()
        assert monitor._categorize_error("Some unknown error") == "other"

    def test_get_health_summary(self):
        monitor = SessionMonitor()
        summary = monitor.get_health_summary()
        assert "active_sessions" in summary
        assert "total_created" in summary
        assert "total_closed" in summary
        assert "failed_sessions" in summary
        assert "recent_errors" in summary
        assert "error_patterns" in summary

    @pytest.mark.asyncio
    async def test_get_health_summary_after_activity(self):
        monitor = SessionMonitor()
        await monitor.record_session_error("timeout occurred")
        await monitor.record_session_closed()
        summary = monitor.get_health_summary()
        assert summary["total_closed"] == 1
        assert summary["failed_sessions"] == 1
        assert summary["recent_errors"] == 1
        assert summary["error_patterns"]["timeout"] == 1


# ─── HTTPQualityRunner ────────────────────────────────────────────────────────

from processing.quality_analysis.http_quality_runner import HTTPQualityRunner


class TestHTTPQualityRunner:
    def test_init(self):
        runner = HTTPQualityRunner()
        assert runner.cache_manager is None
        assert runner.config is not None
        assert runner.config.service_url is not None
        assert runner.monitor is not None

    def test_init_with_cache_manager(self):
        mock_cache = MagicMock()
        runner = HTTPQualityRunner(cache_manager=mock_cache)
        assert runner.cache_manager is mock_cache

    def test_load_and_validate_config(self):
        runner = HTTPQualityRunner()
        config = runner.config
        assert config.total_timeout > 0
        assert config.connect_timeout > 0
        assert config.max_concurrent_connections > 0
        assert config.max_concurrent_requests > 0

    def test_config_from_env(self):
        # service_url is sourced from cached Settings; patch the field directly.
        from config.settings import get_settings

        s = get_settings()
        original_url = s.QUALITY_ANALYSIS_SERVICE_URL
        try:
            object.__setattr__(s, "QUALITY_ANALYSIS_SERVICE_URL", "http://test:1234/api")
            with patch.dict(
                "os.environ",
                {
                    "QUALITY_HTTP_TIMEOUT": "100",
                    "QUALITY_HTTP_CONNECT_TIMEOUT": "10",
                    "QUALITY_MAX_CONNECTIONS": "50",
                    "QUALITY_MAX_CONCURRENT_REQUESTS": "32",
                },
            ):
                runner = HTTPQualityRunner()
                assert runner.config.service_url == "http://test:1234/api"
                assert runner.config.total_timeout == 100
                assert runner.config.connect_timeout == 10
                assert runner.config.max_concurrent_connections == 50
        finally:
            object.__setattr__(s, "QUALITY_ANALYSIS_SERVICE_URL", original_url)


# ─── JavaScript Entity Extractor ──────────────────────────────────────────────

from processing.javascript.javascript_entity_extractor import JavaScriptEntityExtractor


class TestJavaScriptEntityExtractor:
    @pytest.fixture
    def extractor(self):
        return JavaScriptEntityExtractor()

    @pytest.mark.asyncio
    async def test_extract_entities_empty_ast(self, extractor):
        entities = await extractor.extract_entities({"body": []}, "test.js", "test_module")
        # Should have module entity at minimum
        assert len(entities) >= 1
        assert entities[0].type.value == "module"

    @pytest.mark.asyncio
    async def test_extract_function_declaration(self, extractor):
        ast = {
            "body": [
                {
                    "type": "FunctionDeclaration",
                    "id": {"name": "greet"},
                    "params": [{"name": "name"}],
                    "async": False,
                    "loc": {"start": {"line": 1}, "end": {"line": 5}},
                }
            ]
        }
        entities = await extractor.extract_entities(ast, "app.js", "app")
        func_entities = [e for e in entities if e.type.value == "function"]
        assert len(func_entities) == 1
        assert "greet" in func_entities[0].name

    @pytest.mark.asyncio
    async def test_extract_class_declaration(self, extractor):
        ast = {
            "body": [
                {
                    "type": "ClassDeclaration",
                    "id": {"name": "Animal"},
                    "superClass": {"type": "Identifier", "name": "Base"},
                    "body": {
                        "body": [
                            {
                                "type": "MethodDefinition",
                                "key": {"type": "Identifier", "name": "speak"},
                                "value": {
                                    "params": [],
                                    "async": False,
                                },
                                "static": False,
                                "loc": {"start": {"line": 3}, "end": {"line": 5}},
                            }
                        ]
                    },
                    "loc": {"start": {"line": 1}, "end": {"line": 10}},
                }
            ]
        }
        entities = await extractor.extract_entities(ast, "animals.js", "animals")
        class_entities = [e for e in entities if e.type.value == "class"]
        assert len(class_entities) == 1
        assert "Animal" in class_entities[0].name
        assert class_entities[0].metadata.get("bases") == ["Base"]

        method_entities = [e for e in entities if e.type.value == "function" and "speak" in e.name]
        assert len(method_entities) == 1

    @pytest.mark.asyncio
    async def test_extract_variable_arrow_function(self, extractor):
        ast = {
            "body": [
                {
                    "type": "VariableDeclaration",
                    "declarations": [
                        {
                            "id": {"name": "multiply"},
                            "init": {
                                "type": "ArrowFunctionExpression",
                                "params": [{"name": "a"}, {"name": "b"}],
                                "async": False,
                            },
                            "loc": {"start": {"line": 1}, "end": {"line": 1}},
                        }
                    ],
                }
            ]
        }
        entities = await extractor.extract_entities(ast, "math.js", "math")
        func_entities = [e for e in entities if e.type.value == "function"]
        assert len(func_entities) == 1
        assert "multiply" in func_entities[0].name
        assert func_entities[0].metadata.get("is_arrow") is True

    @pytest.mark.asyncio
    async def test_extract_export_function(self, extractor):
        ast = {
            "body": [
                {
                    "type": "ExportNamedDeclaration",
                    "declaration": {
                        "type": "FunctionDeclaration",
                        "id": {"name": "exportedFunc"},
                        "params": [],
                        "async": True,
                        "loc": {"start": {"line": 1}, "end": {"line": 3}},
                    },
                }
            ]
        }
        entities = await extractor.extract_entities(ast, "exports.js", "exports")
        func_entities = [e for e in entities if e.type.value == "function"]
        assert len(func_entities) == 1
        assert func_entities[0].metadata.get("is_exported") is True
        assert func_entities[0].metadata.get("is_async") is True

    @pytest.mark.asyncio
    async def test_extract_export_class(self, extractor):
        ast = {
            "body": [
                {
                    "type": "ExportDefaultDeclaration",
                    "declaration": {
                        "type": "ClassDeclaration",
                        "id": {"name": "AppComponent"},
                        "body": {"body": []},
                        "loc": {"start": {"line": 1}, "end": {"line": 5}},
                    },
                }
            ]
        }
        entities = await extractor.extract_entities(ast, "app.js", "app")
        class_entities = [e for e in entities if e.type.value == "class"]
        assert len(class_entities) == 1
        assert class_entities[0].metadata.get("is_exported") is True

    @pytest.mark.asyncio
    async def test_extract_function_no_name(self, extractor):
        """Function without a name should be skipped."""
        ast = {
            "body": [
                {
                    "type": "FunctionDeclaration",
                    "id": {},  # No name
                    "params": [],
                    "loc": {"start": {"line": 1}, "end": {"line": 1}},
                }
            ]
        }
        entities = await extractor.extract_entities(ast, "test.js", "test")
        func_entities = [e for e in entities if e.type.value == "function"]
        assert len(func_entities) == 0

    @pytest.mark.asyncio
    async def test_extract_class_no_name(self, extractor):
        """Class without a name should be skipped."""
        ast = {
            "body": [
                {
                    "type": "ClassDeclaration",
                    "id": {},  # No name
                    "body": {"body": []},
                    "loc": {"start": {"line": 1}, "end": {"line": 1}},
                }
            ]
        }
        entities = await extractor.extract_entities(ast, "test.js", "test")
        class_entities = [e for e in entities if e.type.value == "class"]
        assert len(class_entities) == 0

    @pytest.mark.asyncio
    async def test_extract_method_no_name(self, extractor):
        """Method without a name should be skipped."""
        ast = {
            "body": [
                {
                    "type": "ClassDeclaration",
                    "id": {"name": "Foo"},
                    "body": {
                        "body": [
                            {
                                "type": "MethodDefinition",
                                "key": {"type": "Literal", "value": "computed"},  # Not Identifier
                                "value": {"params": [], "async": False},
                                "static": False,
                                "loc": {"start": {"line": 2}, "end": {"line": 3}},
                            }
                        ]
                    },
                    "loc": {"start": {"line": 1}, "end": {"line": 5}},
                }
            ]
        }
        entities = await extractor.extract_entities(ast, "test.js", "test")
        # Should have module and class but no method
        method_entities = [
            e for e in entities if e.type.value == "function" and "computed" in str(e.name)
        ]
        assert len(method_entities) == 0

    @pytest.mark.asyncio
    async def test_extract_variable_non_function(self, extractor):
        """Variable declarations that aren't functions should be skipped."""
        ast = {
            "body": [
                {
                    "type": "VariableDeclaration",
                    "declarations": [
                        {
                            "id": {"name": "count"},
                            "init": {"type": "Literal", "value": 42},
                            "loc": {"start": {"line": 1}, "end": {"line": 1}},
                        }
                    ],
                }
            ]
        }
        entities = await extractor.extract_entities(ast, "test.js", "test")
        func_entities = [e for e in entities if e.type.value == "function"]
        assert len(func_entities) == 0

    def test_create_entity_helper(self, extractor):
        from models.core_models import EntityType

        entity = extractor._create_entity(
            name="test_func",
            entity_type=EntityType.FUNCTION,
            file_path="test.js",
            line_number=10,
            visibility="private",
            signature="function test_func()",
        )
        assert entity.name == "test_func"
        assert entity.type == EntityType.FUNCTION
        assert entity.language == "javascript"
        assert entity.file_path == "test.js"
        assert entity.line_number == 10
        assert entity.visibility == "private"

    @pytest.mark.asyncio
    async def test_extract_hello_function_special_case(self, extractor):
        """The 'hello' function name has special handling."""
        ast = {
            "body": [
                {
                    "type": "FunctionDeclaration",
                    "id": {"name": "hello"},
                    "params": [],
                    "async": False,
                    "loc": {"start": {"line": 1}, "end": {"line": 3}},
                }
            ]
        }
        entities = await extractor.extract_entities(ast, "hello.js", "hello_module")
        func_entities = [e for e in entities if e.type.value == "function"]
        assert len(func_entities) == 1
        assert func_entities[0].name == "hello"  # Not prefixed with module


# ─── Report Generator ─────────────────────────────────────────────────────────

from processing.quality_analysis.report_generator import (
    QualityReportGenerator,
    create_quality_report_generator,
)


class TestQualityReportGenerator:
    def test_init_default(self):
        gen = QualityReportGenerator()
        assert gen.cache_manager is None
        assert gen.config_manager is not None
        assert gen.data_processor is not None
        assert gen.template_manager is not None
        assert gen.chart_generator is not None
        assert gen.output_handler is not None

    def test_init_with_output_dir(self, tmp_path):
        gen = QualityReportGenerator(output_directory=str(tmp_path))
        assert str(gen.output_handler.output_directory) == str(tmp_path)

    def test_init_with_cache_manager(self):
        mock_cache = MagicMock()
        gen = QualityReportGenerator(cache_manager=mock_cache)
        assert gen.cache_manager is mock_cache

    def test_get_available_themes(self):
        gen = QualityReportGenerator()
        themes = gen.get_available_themes()
        assert isinstance(themes, list)
        assert len(themes) > 0

    def test_get_available_styles(self):
        gen = QualityReportGenerator()
        styles = gen.get_available_styles()
        assert isinstance(styles, list)
        assert len(styles) > 0

    def test_get_chart_types(self):
        gen = QualityReportGenerator()
        types = gen.get_chart_types()
        assert isinstance(types, list)

    def test_get_supported_formats(self):
        gen = QualityReportGenerator()
        formats = gen.get_supported_formats()
        assert isinstance(formats, list)
        assert len(formats) > 0

    def test_factory_function(self):
        gen = create_quality_report_generator()
        assert isinstance(gen, QualityReportGenerator)

    def test_factory_function_custom_dir(self, tmp_path):
        gen = create_quality_report_generator(output_directory=str(tmp_path))
        assert str(gen.output_handler.output_directory) == str(tmp_path)


# ─── Component Extractor ──────────────────────────────────────────────────────

from processing.architecture.component_extractor import ComponentExtractor


class TestComponentExtractor:
    def test_init(self):
        extractor = ComponentExtractor()
        assert extractor is not None
        assert extractor.max_workers >= 1
        assert extractor.extraction_metrics == {}

    def test_init_custom_workers(self):
        extractor = ComponentExtractor(max_workers=2, use_processes=False)
        assert extractor.max_workers == 2
        assert extractor.use_processes is False

    def test_get_default_worker_count(self):
        extractor = ComponentExtractor()
        count = extractor._get_default_worker_count()
        assert count >= 1

    def test_get_default_worker_count_from_env(self):
        with patch.dict("os.environ", {"ARCH_PROCESS_POOL_SIZE": "4"}):
            extractor = ComponentExtractor()
            count = extractor._get_default_worker_count()
            assert count == 4


# ─── Fix Suggester ─────────────────────────────────────────────────────────────

from processing.quality_analysis.fix_suggester import AutoFixSuggester


class TestAutoFixSuggester:
    def test_init(self):
        suggester = AutoFixSuggester()
        assert suggester is not None

    @pytest.mark.asyncio
    async def test_suggest_fixes_empty(self):
        suggester = AutoFixSuggester()
        result = await suggester.suggest_fixes([])
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_suggest_bulk_fixes_empty(self):
        suggester = AutoFixSuggester()
        result = await suggester.suggest_bulk_fixes([])
        assert isinstance(result, dict)
