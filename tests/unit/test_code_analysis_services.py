"""Unit tests for services.code_analysis modules.

Covers:
- call_graph_service.CallGraphService
- function_analysis_service.FunctionAnalysisService
- consistency_analysis_service.ConsistencyAnalysisService
- pattern_search_service.PatternSearchService
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

import models.static_analysis_models as _sam

if not hasattr(_sam, "CallGraphNodeLocal"):

    class _CallGraphNodeLocal(BaseModel):
        id: str = Field(default="")
        name: str = Field(default="")
        type: str = Field(default="function")
        depth: int = Field(default=0)
        properties: dict = Field(default_factory=dict)

    class _CallGraphEdgeLocal(BaseModel):
        source: str = Field(default="")
        target: str = Field(default="")
        type: str = Field(default="calls")
        properties: dict = Field(default_factory=dict)

    _sam.CallGraphNodeLocal = _CallGraphNodeLocal
    _sam.CallGraphEdgeLocal = _CallGraphEdgeLocal

from services.code_analysis.call_graph_service import CallGraph, CallGraphService
from services.code_analysis.function_analysis_service import (
    FunctionAnalysisService,
    FunctionCallRelation,
    FunctionMetrics,
)


@pytest.fixture
def mock_analyzer():
    analyzer = MagicMock()
    analyzer.analyze_call_graph = MagicMock(
        return_value={
            "nodes": [
                {"id": "fn1", "name": "main", "type": "function", "depth": 0},
                {"id": "fn2", "name": "helper", "type": "function", "depth": 1},
            ],
            "edges": [
                {"source": "fn1", "target": "fn2", "type": "calls"},
            ],
        }
    )
    analyzer.find_function_calls = MagicMock(return_value=[{"target": "helper", "line": 10}])
    analyzer.find_class_references = MagicMock(return_value=[])
    analyzer.find_import_usage = MagicMock(return_value=[])
    analyzer.analyze_dependency_graph = MagicMock(return_value={"dependencies": []})
    analyzer.extract_call_relationships_from_string = MagicMock(
        return_value=[
            {
                "source": "main",
                "target": "helper",
                "line_number": 5,
                "source_type": "function",
                "target_type": "function",
            }
        ]
    )
    analyzer.calculate_function_metrics = MagicMock(
        return_value={
            "outgoing_calls": 3,
            "incoming_calls": 1,
            "unique_callees": 2,
            "unique_callers": 1,
            "cyclomatic_complexity": 5,
            "call_based_importance": 0.7,
        }
    )
    return analyzer


@pytest.fixture
def mock_metrics_service():
    svc = MagicMock()
    svc.get_metrics = MagicMock(return_value={})
    return svc


@pytest.fixture
def call_graph_service(mock_analyzer):
    return CallGraphService(mock_analyzer)


@pytest.fixture
def function_analysis_service(mock_analyzer, mock_metrics_service):
    return FunctionAnalysisService(mock_analyzer, mock_metrics_service)


class TestCallGraphService:
    """test_call_graph_service"""

    async def test_build_call_graph(self, call_graph_service):
        result = await call_graph_service.build_call_graph("main", depth=2)
        assert isinstance(result, CallGraph)
        assert result.root == "main"
        assert result.depth == 2
        assert result.direction == "outgoing"
        assert result.nodes == []
        assert result.edges == []

    async def test_build_call_graph_error_returns_empty(self, call_graph_service):
        call_graph_service.analyzer.analyze_call_graph = MagicMock(
            side_effect=KeyError("not found")
        )
        result = await call_graph_service.build_call_graph("missing", depth=3, direction="incoming")
        assert isinstance(result, CallGraph)
        assert result.root == "missing"
        assert result.depth == 3
        assert result.direction == "incoming"
        assert result.nodes == []
        assert result.edges == []

    async def test_get_function_calls(self, call_graph_service):
        result = await call_graph_service.get_function_calls("cg1", "main")
        assert result["function_name"] == "main"
        assert result["calls"] == [{"target": "helper", "line": 10}]
        assert result["total_calls"] == 1

    async def test_get_function_calls_error(self, call_graph_service):
        call_graph_service.analyzer.find_function_calls = MagicMock(side_effect=ValueError("error"))
        result = await call_graph_service.get_function_calls("cg1", "main")
        assert result["error"] == "error"
        assert result["calls"] == []

    async def test_find_class_references(self, call_graph_service):
        result = await call_graph_service.find_class_references("cg1", "MyClass")
        assert result["class_name"] == "MyClass"
        assert result["references"] == []
        assert result["total_references"] == 0

    async def test_find_import_usage(self, call_graph_service):
        result = await call_graph_service.find_import_usage("cg1", "os")
        assert result["module_name"] == "os"
        assert result["usage"] == []
        assert result["total_usage"] == 0

    async def test_get_dependency_graph(self, call_graph_service):
        result = await call_graph_service.get_dependency_graph("cg1", "node1")
        assert result == {"dependencies": []}
        call_graph_service.analyzer.analyze_dependency_graph.assert_called_once_with("node1", 3)


class TestFunctionAnalysisService:
    """test_function_analysis_service"""

    async def test_extract_function_calls(self, function_analysis_service):
        code = "def main(): helper()"
        result = await function_analysis_service.extract_function_calls(code, "test.py")
        assert len(result) == 1
        rel = result[0]
        assert isinstance(rel, FunctionCallRelation)
        assert rel.source == "main"
        assert rel.target == "helper"
        assert rel.line_number == 5
        assert rel.file_path == "test.py"
        assert rel.source_type == "function"
        assert rel.target_type == "function"

    async def test_extract_function_calls_error(self, function_analysis_service):
        function_analysis_service.analyzer.extract_call_relationships_from_string = MagicMock(
            side_effect=RuntimeError("parse error")
        )
        result = await function_analysis_service.extract_function_calls("broken code")
        assert result == []

    async def test_get_function_metrics(self, function_analysis_service):
        result = await function_analysis_service.get_function_metrics("main")
        assert isinstance(result, FunctionMetrics)
        assert result.function_name == "main"
        assert result.outgoing_calls == 3
        assert result.incoming_calls == 1
        assert result.unique_callees == 2
        assert result.unique_callers == 1
        assert result.cyclomatic_complexity == 5
        assert result.call_based_importance == 0.7
        assert result.error is None

    async def test_get_function_metrics_error(self, function_analysis_service):
        function_analysis_service.analyzer.calculate_function_metrics = MagicMock(
            side_effect=RuntimeError("metrics error")
        )
        result = await function_analysis_service.get_function_metrics("broken")
        assert isinstance(result, FunctionMetrics)
        assert result.function_name == "broken"
        assert result.outgoing_calls == 0
        assert result.incoming_calls == 0
        assert result.unique_callees == 0
        assert result.unique_callers == 0
        assert result.cyclomatic_complexity == 0
        assert result.call_based_importance == 0.0
        assert result.error == "metrics error"


class TestConsistencyAnalysisService:
    """test_consistency_analysis_service"""

    async def test_analyze_code_consistency(self):
        import services.code_analysis.consistency_analysis_service as cas_mod

        with patch.object(cas_mod, "get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            from services.code_analysis.consistency_analysis_service import (
                ConsistencyAnalysisResult,
                ConsistencyAnalysisService,
            )

            svc = ConsistencyAnalysisService()

            ast_result = {
                "functions": [
                    {"name": "myFunction", "docstring": None},
                    {"name": "good_func", "docstring": "Does stuff"},
                ],
                "classes": [
                    {"name": "my_class", "docstring": None},
                ],
                "variables": [],
            }

            result = await svc.analyze_code_consistency("def myFunction(): pass", ast_result)
            assert isinstance(result, ConsistencyAnalysisResult)
            assert 0.0 <= result.consistency_score <= 1.0
            assert result.consistency_score < 1.0
            assert any("myFunction" in issue for issue in result.naming_issues)
            assert any("my_class" in issue for issue in result.naming_issues)
            assert any("my_class" in issue for issue in result.documentation_issues)
            assert any("myFunction" in issue for issue in result.documentation_issues)
            assert len(result.suggestions) >= 2

    async def test_analyze_code_consistency_empty_ast(self):
        import services.code_analysis.consistency_analysis_service as cas_mod

        with patch.object(cas_mod, "get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            from services.code_analysis.consistency_analysis_service import (
                ConsistencyAnalysisService,
            )

            svc = ConsistencyAnalysisService()
            result = await svc.analyze_code_consistency("", {})
            assert result.consistency_score == 1.0
            assert result.naming_issues == []
            assert result.documentation_issues == []
            assert result.style_issues == []
            assert result.pattern_violations == []
            assert result.suggestions == []

    async def test_close(self):
        import services.code_analysis.consistency_analysis_service as cas_mod

        with patch.object(cas_mod, "get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            from services.code_analysis.consistency_analysis_service import (
                ConsistencyAnalysisService,
            )

            svc = ConsistencyAnalysisService()
            mock_conn = MagicMock()
            mock_conn.conn = True
            mock_conn.close = AsyncMock()
            svc.db_connection = mock_conn
            await svc.close()
            mock_conn.close.assert_called_once()
            assert svc.db_connection is None

    async def test_get_db_connection(self):
        import services.code_analysis.consistency_analysis_service as cas_mod

        with patch.object(cas_mod, "get_settings") as mock_settings:
            # get_code_graph_db_path() (sqlite file resolution) no longer exists;
            # the constructor only needs a Settings stand-in.
            mock_settings.return_value = MagicMock()
            from services.code_analysis.consistency_analysis_service import (
                ConsistencyAnalysisService,
            )

            svc = ConsistencyAnalysisService()
            mock_conn = MagicMock()
            mock_conn.connect = AsyncMock()
            with patch.object(
                cas_mod, "GraphDatabaseConnection", return_value=mock_conn
            ) as mock_gdc:
                result = await svc._get_db_connection("test-graph")
            assert result is mock_conn
            # postgres pool: connection is scoped by code_graph_id, not a sqlite
            # file path resolved from settings.get_code_graph_db_path(...).
            mock_gdc.assert_called_once_with("test-graph")
            mock_conn.connect.assert_called_once()
            assert svc.db_connection is mock_conn

    async def test_analyze_code_consistency_with_issues(self):
        import services.code_analysis.consistency_analysis_service as cas_mod

        with patch.object(cas_mod, "get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            from services.code_analysis.consistency_analysis_service import (
                ConsistencyAnalysisService,
            )

            svc = ConsistencyAnalysisService()
            ast_result = {
                "functions": [
                    {"name": "BadName", "args": ["a", "b", "c", "d", "e", "f"]},
                    {"name": "_private_func", "args": []},
                    {"name": "good_func", "docstring": "OK", "args": [], "returns": "int"},
                ],
                "classes": [
                    {"name": "bad_class"},
                    {"name": "GoodClass", "docstring": "A good class with enough documentation"},
                ],
                "variables": [
                    {"name": "badVar"},
                    {"name": "good_var"},
                    {"name": "CONSTANT"},
                ],
            }
            code = "x = 1   \n" + "a" * 100 + "\n\n\n\npass\n"
            result = await svc.analyze_code_consistency(code, ast_result)

            assert result.consistency_score < 1.0
            assert any("BadName" in i and "snake_case" in i for i in result.naming_issues)
            assert any("bad_class" in i and "PascalCase" in i for i in result.naming_issues)
            assert any("badVar" in i for i in result.naming_issues)
            assert any("trailing whitespace" in i for i in result.style_issues)
            assert any("exceeds" in i for i in result.style_issues)
            assert any("blank lines" in i for i in result.style_issues)
            assert any(
                "BadName" in i and "too many parameters" in i for i in result.pattern_violations
            )
            assert any("BadName" in i and "return type" in i for i in result.pattern_violations)
            assert len(result.suggestions) == 4

    async def test_analyze_code_consistency_error_returns_default(self):
        import services.code_analysis.consistency_analysis_service as cas_mod

        with patch.object(cas_mod, "get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            from services.code_analysis.consistency_analysis_service import (
                ConsistencyAnalysisService,
            )

            svc = ConsistencyAnalysisService()
            with patch.object(svc, "_check_naming_conventions", side_effect=Exception("boom")):
                result = await svc.analyze_code_consistency("code", {})
            assert result.consistency_score == 0.5
            assert result.naming_issues == []
            assert result.documentation_issues == []
            assert result.style_issues == []
            assert result.pattern_violations == []
            assert result.suggestions == []

    async def test_check_documentation_short_docstrings(self):
        import services.code_analysis.consistency_analysis_service as cas_mod

        with patch.object(cas_mod, "get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            from services.code_analysis.consistency_analysis_service import (
                ConsistencyAnalysisService,
            )

            svc = ConsistencyAnalysisService()
            ast_result = {
                "functions": [
                    {"name": "public_func", "docstring": "Short"},
                ],
                "classes": [
                    {"name": "MyClass"},
                    {"name": "ShortDoc", "docstring": "Tiny"},
                ],
            }
            issues = await svc._check_documentation_style(ast_result)
            assert any("public_func" in i and "short docstring" in i for i in issues)
            assert any("MyClass" in i and "missing docstring" in i for i in issues)
            assert any("ShortDoc" in i and "short docstring" in i for i in issues)
            assert len(issues) == 3

    async def test_check_naming_variable_issues(self):
        import services.code_analysis.consistency_analysis_service as cas_mod

        with patch.object(cas_mod, "get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            from services.code_analysis.consistency_analysis_service import (
                ConsistencyAnalysisService,
            )

            svc = ConsistencyAnalysisService()
            ast_result = {
                "functions": [],
                "classes": [],
                "variables": [
                    {"name": "camelCase"},
                    {"name": "good_var"},
                    {"name": "CONSTANT"},
                ],
            }
            issues = await svc._check_naming_conventions(ast_result)
            assert any("camelCase" in i and "snake_case" in i for i in issues)
            flagged_names = [i for i in issues if "good_var" in i]
            assert flagged_names == []
            constant_issues = [i for i in issues if "CONSTANT" in i]
            assert constant_issues == []

    def test_is_snake_case_empty(self):
        import services.code_analysis.consistency_analysis_service as cas_mod

        with patch.object(cas_mod, "get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            from services.code_analysis.consistency_analysis_service import (
                ConsistencyAnalysisService,
            )

            svc = ConsistencyAnalysisService()
            assert svc._is_snake_case("") is False
            assert svc._is_snake_case("__") is True
            assert svc._is_snake_case("hello_world") is True
            assert svc._is_snake_case("CamelCase") is False
            assert svc._is_snake_case("__init__") is True

    def test_is_pascal_case_empty(self):
        import services.code_analysis.consistency_analysis_service as cas_mod

        with patch.object(cas_mod, "get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            from services.code_analysis.consistency_analysis_service import (
                ConsistencyAnalysisService,
            )

            svc = ConsistencyAnalysisService()
            assert svc._is_pascal_case("") is False
            assert svc._is_pascal_case("__") is True
            assert svc._is_pascal_case("MyClass") is True
            assert svc._is_pascal_case("snake_case") is False
            assert svc._is_pascal_case("HTTPClient") is True

    def test_check_code_style_multiple_blank_lines(self):
        import services.code_analysis.consistency_analysis_service as cas_mod

        with patch.object(cas_mod, "get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            from services.code_analysis.consistency_analysis_service import (
                ConsistencyAnalysisService,
            )

            svc = ConsistencyAnalysisService()
            code = "x = 1\n\n\n\ny = 2"
            issues = svc._check_code_style(code)
            assert any("blank lines" in i for i in issues)
            blank_issues = [i for i in issues if "blank lines" in i]
            assert len(blank_issues) == 1

    def test_generate_suggestions_all_types(self):
        import services.code_analysis.consistency_analysis_service as cas_mod

        with patch.object(cas_mod, "get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            from services.code_analysis.consistency_analysis_service import (
                ConsistencyAnalysisService,
            )

            svc = ConsistencyAnalysisService()
            suggestions = svc._generate_consistency_suggestions(
                ["naming issue"],
                ["doc issue"],
                ["style issue"],
                ["pattern issue"],
            )
            assert len(suggestions) == 4
            assert "PEP 8" in suggestions[0]
            assert "docstrings" in suggestions[1]
            assert "ruff format" in suggestions[2]
            assert "refactoring" in suggestions[3].lower()


class TestPatternSearchService:
    """test_pattern_search_service"""

    def test_init(self):
        from services.code_analysis.pattern_search_service import PatternSearchService

        analyzer = MagicMock()
        db_conn = MagicMock()
        processor_factory = MagicMock()

        svc = PatternSearchService(analyzer, db_conn, processor_factory)
        assert svc.analyzer is analyzer
        assert svc.db_connection is db_conn
        assert svc.processor_factory is processor_factory

    async def test_search_patterns_delegates_to_db(self):
        from models.code_graph_models import PatternSearchResponse
        from services.code_analysis.pattern_search_service import PatternSearchService

        analyzer = MagicMock()
        db_conn = MagicMock()
        db_conn.conn = None
        db_conn.connect = AsyncMock()

        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[])
        db_conn.connect = AsyncMock(side_effect=lambda: setattr(db_conn, "conn", MagicMock()))

        async def fake_execute(query):
            return mock_cursor

        db_conn.conn = None
        processor_factory = MagicMock()

        svc = PatternSearchService(analyzer, db_conn, processor_factory)

        request = MagicMock()
        request.pattern = "*.py"
        request.pattern_type = "glob"
        request.max_results = 10

        result = await svc.search_patterns("cg1", request)
        assert isinstance(result, PatternSearchResponse)
        assert result.code_graph_id == "cg1"
        assert result.matches == [] or result.error is not None
        assert result.count == 0 or result.error is not None
