"""Unit tests for import_validation_service, function_validation_service, and complexity_analyzer.

Covers:
- ImportValidationService: validate_imports, is_stdlib_module, _is_third_party_available,
  _is_likely_third_party, _check_import_in_graph
- FunctionValidationService: validate_function_calls, is_builtin_function,
  _check_function_in_graph_direct
- ComplexityAnalysisService: analyze_file_complexity, _calculate_quality_score,
  _generate_file_suggestions, get_supported_languages, get_complexity_service
- All per-language analyzers: analyze_entity_complexity, classify, suggestions, factors
"""

from __future__ import annotations

import types as _types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.core_models import CoreCodeEntity, EntityType
from processing.analyzers.complexity_analyzers import (
    ComplexityAnalysisService,
    ComplexityLevel,
    CppComplexityAnalyzer,
    CSharpComplexityAnalyzer,
    FileComplexityReport,
    GoComplexityAnalyzer,
    JavaComplexityAnalyzer,
    JavaScriptComplexityAnalyzer,
    PythonComplexityAnalyzer,
    RustComplexityAnalyzer,
    ServiceComplexityMetrics,
    TypeScriptComplexityAnalyzer,
    get_complexity_service,
)
from services.code_analysis.function_validation_service import (
    FunctionValidationResult,
    FunctionValidationService,
)
from services.code_analysis.import_validation_service import (
    ImportValidationResult,
    ImportValidationService,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_entity(
    name: str = "my_func",
    entity_type: str = "function",
    language: str = "python",
    signature: str | None = None,
    properties: dict | None = None,
    is_async: bool = False,
) -> CoreCodeEntity:
    return CoreCodeEntity(
        id=f"ent-{name}",
        name=name,
        type=EntityType(entity_type),
        language=language,
        file_path="test.py",
        line_number=1,
        signature=signature,
        properties=properties or {},
        is_async=is_async,
    )


def _make_relation(source_id: str, target_id: str, rel_type: str):
    """Create a relation-like object.

    Uses a simple namespace since the complexity analyzers only access
    source_id, target_id, and type as plain strings, and ParseResult
    validation would reject non-enum relation types.
    """
    rel = _types.SimpleNamespace()
    rel.source_id = source_id
    rel.target_id = target_id
    rel.type = rel_type
    return rel


def _make_parse_result(
    language: str = "python",
    entities: list[CoreCodeEntity] | None = None,
    relationships: list | None = None,
    lines_of_code: int = 50,
):
    """Create a parse-result-like object.

    Uses SimpleNamespace to avoid Pydantic validation of relationship types,
    since the complexity analyzers use non-enum string types like
    'trait_bound', 'template_specialization', 'uses', etc.
    """
    pr = _types.SimpleNamespace()
    pr.file_path = "test.py"
    pr.language = language
    pr.entities = entities or []
    pr.relationships = relationships or []
    pr.lines_of_code = lines_of_code
    return pr


# ===================================================================
# ImportValidationService
# ===================================================================
class TestImportValidationResult:
    def test_model_fields(self):
        r = ImportValidationResult(
            valid_imports=[],
            invalid_imports=[],
            missing_dependencies=[],
            stdlib_imports=[],
            third_party_imports=[],
            local_imports=[],
            suggestions=[],
        )
        assert r.valid_imports == []
        assert r.suggestions == []


class TestImportValidationService:
    @patch("services.code_analysis.import_validation_service.DependencyParser")
    def setup_method(self, _method, mock_dep_parser_cls):
        # The service no longer reads settings/CODE_GRAPH_DATABASE_DIR: imports
        # are checked against the shared postgres pool via get_raw_connection().
        self.mock_dep_parser = MagicMock()
        mock_dep_parser_cls.return_value = self.mock_dep_parser
        self.service = ImportValidationService(project_root="/tmp/project")

    def test_is_stdlib_module_os(self):
        assert self.service.is_stdlib_module("os") is True

    def test_is_stdlib_module_json_path(self):
        assert self.service.is_stdlib_module("json.decoder") is True

    def test_is_stdlib_module_relative(self):
        assert self.service.is_stdlib_module(".local_mod") is False

    def test_is_stdlib_module_unknown(self):
        assert self.service.is_stdlib_module("flask") is False

    def test_is_likely_third_party_true(self):
        assert self.service._is_likely_third_party("requests") is True
        assert self.service._is_likely_third_party("numpy.core") is True

    def test_is_likely_third_party_false(self):
        assert self.service._is_likely_third_party("my_custom_module") is False

    @pytest.mark.asyncio
    async def test_is_third_party_available_direct(self):
        result = await self.service._is_third_party_available("requests", {"requests": {}})
        assert result is True

    @pytest.mark.asyncio
    async def test_is_third_party_available_mapped(self):
        # cv2 -> opencv-python
        result = await self.service._is_third_party_available("cv2", {"opencv-python": {}})
        assert result is True

    @pytest.mark.asyncio
    async def test_is_third_party_available_prefix_match(self):
        result = await self.service._is_third_party_available("mylib.sub", {"mylib": {}})
        assert result is True

    @pytest.mark.asyncio
    async def test_is_third_party_available_false(self):
        result = await self.service._is_third_party_available("nonexistent", {"other": {}})
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_imports_stdlib(self):
        imports = [{"module": "os", "type": "import"}]
        result = await self.service.validate_imports("graph1", imports)
        assert isinstance(result, ImportValidationResult)
        assert "os" in result.stdlib_imports
        assert len(result.valid_imports) == 1

    @pytest.mark.asyncio
    async def test_validate_imports_empty_module_skipped(self):
        imports = [{"module": "", "type": "import"}]
        result = await self.service.validate_imports("graph1", imports)
        assert len(result.valid_imports) == 0
        assert len(result.invalid_imports) == 0

    @pytest.mark.asyncio
    async def test_validate_imports_local_in_graph(self):
        with patch.object(
            self.service, "_check_import_in_graph", new_callable=AsyncMock, return_value=True
        ):
            imports = [{"module": "my_local_mod", "type": "import"}]
            result = await self.service.validate_imports("graph1", imports)
            assert "my_local_mod" in result.local_imports
            assert len(result.valid_imports) == 1

    @pytest.mark.asyncio
    async def test_validate_imports_third_party_available(self):
        self.mock_dep_parser.get_available_packages = AsyncMock(
            return_value={"requests": {"version": "2.0"}}
        )
        with patch.object(
            self.service, "_check_import_in_graph", new_callable=AsyncMock, return_value=False
        ):
            imports = [{"module": "requests", "type": "import"}]
            result = await self.service.validate_imports("graph1", imports)
            assert "requests" in result.third_party_imports

    @pytest.mark.asyncio
    async def test_validate_imports_likely_third_party_missing_with_package_info(self):
        self.mock_dep_parser.get_available_packages = AsyncMock(return_value={})
        self.mock_dep_parser.get_package_info.return_value = {"name": "requests"}
        with (
            patch.object(
                self.service, "_check_import_in_graph", new_callable=AsyncMock, return_value=False
            ),
            patch.object(
                self.service,
                "_is_third_party_available",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            imports = [{"module": "requests", "type": "import"}]
            result = await self.service.validate_imports("graph1", imports)
            assert "requests" in result.missing_dependencies
            assert any("may not be installed" in s for s in result.suggestions)

    @pytest.mark.asyncio
    async def test_validate_imports_likely_third_party_missing_with_similar(self):
        self.mock_dep_parser.get_available_packages = AsyncMock(return_value={})
        self.mock_dep_parser.get_package_info.return_value = None
        self.mock_dep_parser.get_similar_packages.return_value = ["requests2"]
        with (
            patch.object(
                self.service, "_check_import_in_graph", new_callable=AsyncMock, return_value=False
            ),
            patch.object(
                self.service,
                "_is_third_party_available",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            imports = [{"module": "requests", "type": "import"}]
            result = await self.service.validate_imports("graph1", imports)
            assert any("Did you mean" in s for s in result.suggestions)

    @pytest.mark.asyncio
    async def test_validate_imports_likely_third_party_missing_no_similar(self):
        self.mock_dep_parser.get_available_packages = AsyncMock(return_value={})
        self.mock_dep_parser.get_package_info.return_value = None
        self.mock_dep_parser.get_similar_packages.return_value = []
        with (
            patch.object(
                self.service, "_check_import_in_graph", new_callable=AsyncMock, return_value=False
            ),
            patch.object(
                self.service,
                "_is_third_party_available",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            imports = [{"module": "requests", "type": "import"}]
            result = await self.service.validate_imports("graph1", imports)
            assert any("check spelling" in s for s in result.suggestions)

    @pytest.mark.asyncio
    async def test_validate_imports_unknown_not_third_party_with_similar(self):
        self.mock_dep_parser.get_available_packages = AsyncMock(return_value={})
        self.mock_dep_parser.get_similar_packages.return_value = ["my_mod_v2"]
        with (
            patch.object(
                self.service, "_check_import_in_graph", new_callable=AsyncMock, return_value=False
            ),
            patch.object(
                self.service,
                "_is_third_party_available",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            imports = [{"module": "my_mod", "type": "import"}]
            result = await self.service.validate_imports("graph1", imports)
            assert "my_mod" in result.missing_dependencies
            assert any("Did you mean" in s for s in result.suggestions)

    @pytest.mark.asyncio
    async def test_validate_imports_unknown_not_third_party_no_similar(self):
        self.mock_dep_parser.get_available_packages = AsyncMock(return_value={})
        self.mock_dep_parser.get_similar_packages.return_value = []
        with (
            patch.object(
                self.service, "_check_import_in_graph", new_callable=AsyncMock, return_value=False
            ),
            patch.object(
                self.service,
                "_is_third_party_available",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            imports = [{"module": "my_mod", "type": "import"}]
            result = await self.service.validate_imports("graph1", imports)
            assert any("Consider adding" in s for s in result.suggestions)

    @pytest.mark.asyncio
    async def test_validate_imports_exception_handling(self):
        with patch.object(
            self.service,
            "_check_import_in_graph",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            # is_stdlib_module returns False for 'my_weird', so it'll hit _check_import_in_graph
            imports = [{"module": "my_weird_thing_xyz", "type": "import"}]
            result = await self.service.validate_imports("graph1", imports)
            assert len(result.invalid_imports) == 1

    # _check_import_in_graph now queries the shared postgres pool
    # (get_raw_connection) scoped by code_graph_id, returning count > 0.
    _GET_CONN = "services.code_analysis.import_validation_service.get_raw_connection"

    @pytest.mark.asyncio
    async def test_check_import_in_graph_found(self, make_pg_conn):
        conn = make_pg_conn(routes={"COUNT(*) FROM nodes": [(1,)]})
        with patch(self._GET_CONN, return_value=conn):
            result = await self.service._check_import_in_graph("graph1", "my_mod")
        assert result is True
        # the lookup is scoped to the code_graph_id and bound, not interpolated
        sql, params = conn.executed[0]
        assert "code_graph_id = %s" in " ".join(sql.split())
        assert params == ("graph1", "my_mod")

    @pytest.mark.asyncio
    async def test_check_import_in_graph_not_found(self, make_pg_conn):
        conn = make_pg_conn(routes={"COUNT(*) FROM nodes": [(0,)]})
        with patch(self._GET_CONN, return_value=conn):
            result = await self.service._check_import_in_graph("graph1", "missing")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_import_in_graph_psycopg_error(self):
        import psycopg

        with patch(self._GET_CONN, side_effect=psycopg.Error("no db")):
            result = await self.service._check_import_in_graph("graph1", "mod")
        assert result is False

    @pytest.mark.asyncio
    async def test_check_import_in_graph_generic_error(self):
        # Any non-psycopg error is caught by the outer except and returns False.
        with patch(self._GET_CONN, side_effect=RuntimeError("boom")):
            result = await self.service._check_import_in_graph("graph1", "mod")
        assert result is False


# ===================================================================
# FunctionValidationService
# ===================================================================
class TestFunctionValidationResult:
    def test_model_fields(self):
        r = FunctionValidationResult(
            valid_functions=[],
            invalid_functions=[],
            builtin_functions=[],
            external_functions=[],
            local_functions=[],
            suggestions=[],
        )
        assert r.valid_functions == []


class TestFunctionValidationService:
    def setup_method(self):
        self.service = FunctionValidationService()

    def test_is_builtin_function_print(self):
        assert self.service.is_builtin_function("print") is True

    def test_is_builtin_function_len(self):
        assert self.service.is_builtin_function("len") is True

    def test_is_builtin_function_custom(self):
        assert self.service.is_builtin_function("my_function") is False

    def test_builtin_set_contains_expected(self):
        builtins = self.service._get_builtin_functions()
        for name in ("abs", "all", "any", "dict", "int", "str", "zip", "__import__"):
            assert name in builtins

    @pytest.mark.asyncio
    async def test_validate_function_calls_builtin(self):
        db_conn = MagicMock()
        calls = [{"name": "print", "args": []}]
        result = await self.service.validate_function_calls("g1", calls, db_conn)
        assert isinstance(result, FunctionValidationResult)
        assert "print" in result.builtin_functions
        assert len(result.valid_functions) == 1

    @pytest.mark.asyncio
    async def test_validate_function_calls_empty_name_skipped(self):
        db_conn = MagicMock()
        calls = [{"name": "", "args": []}]
        result = await self.service.validate_function_calls("g1", calls, db_conn)
        assert len(result.valid_functions) == 0

    @pytest.mark.asyncio
    async def test_validate_function_calls_local(self):
        db_conn = MagicMock()
        with patch.object(
            self.service,
            "_check_function_in_graph_direct",
            new_callable=AsyncMock,
            return_value=True,
        ):
            calls = [{"name": "my_func", "args": []}]
            result = await self.service.validate_function_calls("g1", calls, db_conn)
            assert "my_func" in result.local_functions

    @pytest.mark.asyncio
    async def test_validate_function_calls_external(self):
        db_conn = MagicMock()
        with patch.object(
            self.service,
            "_check_function_in_graph_direct",
            new_callable=AsyncMock,
            return_value=False,
        ):
            calls = [{"name": "third_party_fn", "args": []}]
            result = await self.service.validate_function_calls("g1", calls, db_conn)
            assert "third_party_fn" in result.external_functions

    @pytest.mark.asyncio
    async def test_validate_function_calls_exception(self):
        db_conn = MagicMock()
        with patch.object(
            self.service,
            "_check_function_in_graph_direct",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db error"),
        ):
            calls = [{"name": "bad_func", "args": []}]
            result = await self.service.validate_function_calls("g1", calls, db_conn)
            assert len(result.invalid_functions) == 1

    @pytest.mark.asyncio
    async def test_check_function_in_graph_direct_found(self):
        db_conn = MagicMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone.return_value = (3,)
        db_conn.conn = AsyncMock()
        db_conn.conn.execute = AsyncMock(return_value=mock_cursor)
        result = await self.service._check_function_in_graph_direct("my_func", db_conn)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_function_in_graph_direct_not_found(self):
        db_conn = MagicMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone.return_value = (0,)
        db_conn.conn = AsyncMock()
        db_conn.conn.execute = AsyncMock(return_value=mock_cursor)
        result = await self.service._check_function_in_graph_direct("missing", db_conn)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_function_in_graph_direct_no_conn(self):
        db_conn = MagicMock()
        db_conn.conn = None
        result = await self.service._check_function_in_graph_direct("fn", db_conn)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_function_in_graph_direct_error(self):
        db_conn = MagicMock()
        db_conn.conn = AsyncMock()
        db_conn.conn.execute = AsyncMock(side_effect=RuntimeError("boom"))
        result = await self.service._check_function_in_graph_direct("fn", db_conn)
        assert result is False

    @pytest.mark.asyncio
    async def test_check_function_in_graph_direct_none_result(self):
        db_conn = MagicMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone.return_value = None
        db_conn.conn = AsyncMock()
        db_conn.conn.execute = AsyncMock(return_value=mock_cursor)
        result = await self.service._check_function_in_graph_direct("fn", db_conn)
        assert result is False


# ===================================================================
# ComplexityAnalysisService + all language analyzers
# ===================================================================
class TestComplexityAnalysisService:
    def setup_method(self):
        self.service = ComplexityAnalysisService()

    def test_get_supported_languages(self):
        langs = self.service.get_supported_languages()
        assert set(langs) == {
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
            "rust",
            "cpp",
            "csharp",
        }

    def test_analyze_file_complexity_empty(self):
        pr = _make_parse_result(entities=[], lines_of_code=10)
        report = self.service.analyze_file_complexity(pr)
        assert isinstance(report, FileComplexityReport)
        assert report.total_entities == 0
        assert report.quality_score == 100.0

    def test_analyze_file_complexity_with_entities(self):
        entities = [
            _make_entity("simple_fn", "function", "python", signature="def simple_fn()"),
        ]
        pr = _make_parse_result(entities=entities, lines_of_code=20)
        report = self.service.analyze_file_complexity(pr)
        assert report.total_entities == 1
        # No source content passed: entities are not computed, never 1.0.
        assert report.entities_not_computed == report.total_entities
        assert report.max_complexity == 0.0

    def test_analyze_file_complexity_complex_entities(self):
        entities = [
            _make_entity(
                "big_fn",
                "function",
                "python",
                signature="if else elif while for switch case try catch if else elif while for switch case try catch if else elif while for",
            ),
        ]
        pr = _make_parse_result(entities=entities, lines_of_code=200)
        report = self.service.analyze_file_complexity(pr)
        # No source content passed: entities are not computed, never 1.0.
        assert report.entities_not_computed == report.total_entities
        assert report.max_complexity == 0.0

    def test_analyze_file_complexity_unsupported_language(self):
        entities = [_make_entity("fn", "function", "haskell")]
        pr = _make_parse_result(language="haskell", entities=entities)
        report = self.service.analyze_file_complexity(pr)
        # Falls back to Java analyzer
        assert report.language == "haskell"

    def test_calculate_quality_score_no_entities(self):
        report = FileComplexityReport(file_path="a.py", language="python", total_loc=10)
        report.total_entities = 0
        score = self.service._calculate_quality_score(report)
        assert score == 100.0

    def test_calculate_quality_score_high_complexity(self):
        report = FileComplexityReport(
            file_path="a.py", language="python", total_loc=800, average_complexity=25.0
        )
        report.total_entities = 10
        report.complex_entities = 8
        score = self.service._calculate_quality_score(report)
        assert 0.0 <= score <= 100.0
        assert score < 80.0

    def test_generate_file_suggestions_large_file(self):
        report = FileComplexityReport(file_path="a.py", language="python", total_loc=1500)
        report.total_entities = 2
        report.complex_entities = 0
        report.average_complexity = 3.0
        report.hotspots = []
        suggestions = self.service._generate_file_suggestions(report)
        assert any("Large file" in s for s in suggestions)

    def test_generate_file_suggestions_high_ratio(self):
        report = FileComplexityReport(file_path="a.py", language="python", total_loc=100)
        report.total_entities = 10
        report.complex_entities = 5
        report.average_complexity = 3.0
        report.hotspots = []
        suggestions = self.service._generate_file_suggestions(report)
        assert any("High ratio" in s for s in suggestions)

    def test_generate_file_suggestions_high_avg(self):
        report = FileComplexityReport(file_path="a.py", language="python", total_loc=100)
        report.total_entities = 5
        report.complex_entities = 0
        report.average_complexity = 20.0
        report.hotspots = []
        suggestions = self.service._generate_file_suggestions(report)
        assert any("average complexity" in s for s in suggestions)

    def test_generate_file_suggestions_many_hotspots(self):
        report = FileComplexityReport(file_path="a.py", language="python", total_loc=100)
        report.total_entities = 10
        report.complex_entities = 0
        report.average_complexity = 3.0
        report.hotspots = [f"hotspot_{i}" for i in range(7)]
        suggestions = self.service._generate_file_suggestions(report)
        assert any("hotspots" in s for s in suggestions)


class TestGetComplexityService:
    def test_get_complexity_service_singleton(self):
        from processing.analyzers.complexity_analyzers import service as ca_mod

        ca_mod._complexity_service = None
        svc1 = get_complexity_service()
        svc2 = get_complexity_service()
        assert svc1 is svc2
        ca_mod._complexity_service = None  # reset


# ===================================================================
# Java analyzer
# ===================================================================
class TestJavaComplexityAnalyzer:
    def setup_method(self):
        self.analyzer = JavaComplexityAnalyzer()

    def test_get_language_specific_factors(self):
        factors = self.analyzer.get_language_specific_factors()
        assert "inheritance_penalty" in factors
        assert "interface_bonus" in factors

    def test_analyze_class_entity(self):
        entity = _make_entity("MyClass", "class", "java")
        rel = _make_relation("ent-MyClass", "ent-other", "imports")
        pr = _make_parse_result("java", [entity], [rel])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.language == "java"
        assert metrics.entity_name == "MyClass"
        assert metrics.coupling_factor >= 0.0

    def test_analyze_method_entity(self):
        entity = _make_entity("doWork", "method", "java", signature="void doWork(int x, String y)")
        pr = _make_parse_result("java", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.parameter_count == 2

    def test_classify_complexity_thresholds(self):
        assert self.analyzer._classify_complexity(3) == ComplexityLevel.TRIVIAL
        assert self.analyzer._classify_complexity(7) == ComplexityLevel.BASIC
        assert self.analyzer._classify_complexity(15) == ComplexityLevel.MODERATE
        assert self.analyzer._classify_complexity(30) == ComplexityLevel.COMPLEX
        assert self.analyzer._classify_complexity(50) == ComplexityLevel.VERY_COMPLEX

    def test_maintainability_zero_loc(self):
        metrics = ServiceComplexityMetrics(lines_of_code=0)
        result = self.analyzer._calculate_maintainability_index(metrics)
        assert result == 100.0

    def test_suggestions_high_complexity(self):
        entity = _make_entity("big", "method", "java")
        metrics = ServiceComplexityMetrics(
            cyclomatic_complexity=15, parameter_count=7, nesting_depth=5, coupling_factor=0.9
        )
        suggestions = self.analyzer._generate_java_suggestions(metrics, entity)
        assert any("smaller methods" in s for s in suggestions)
        assert any("parameter object" in s for s in suggestions)
        assert any("nesting depth" in s for s in suggestions)
        assert any("coupling" in s.lower() for s in suggestions)

    def test_suggestions_large_class(self):
        entity = _make_entity("HugeClass", "class", "java")
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=60)
        suggestions = self.analyzer._generate_java_suggestions(metrics, entity)
        assert any("Large class" in s for s in suggestions)

    def test_count_parameters_empty(self):
        entity = _make_entity("fn", "method", "java", signature="void fn()")
        assert self.analyzer._count_parameters(entity) == 0


# ===================================================================
# Python analyzer
# ===================================================================
class TestPythonComplexityAnalyzer:
    def setup_method(self):
        self.analyzer = PythonComplexityAnalyzer()

    def test_get_language_specific_factors(self):
        factors = self.analyzer.get_language_specific_factors()
        assert "decorator_penalty" in factors
        assert "async_penalty" in factors

    def test_analyze_function_basic(self):
        entity = _make_entity("do_stuff", "function", "python", signature="def do_stuff(a, b)")
        pr = _make_parse_result("python", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.language == "python"
        assert metrics.parameter_count == 2

    def test_analyze_function_async(self):
        entity = _make_entity(
            "async_fn", "function", "python", signature="async def async_fn()", is_async=True
        )
        pr = _make_parse_result("python", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "async" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_decorators(self):
        entity = _make_entity(
            "dec_fn",
            "function",
            "python",
            signature="def dec_fn()",
            properties={"decorators": ["staticmethod", "cache", "retry"]},
        )
        pr = _make_parse_result("python", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "decorators" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_class_coupling(self):
        entity = _make_entity("MyClass", "class", "python")
        rels = [
            _make_relation("ent-MyClass", "ent-base", "inherits"),
            _make_relation("ent-MyClass", "ent-mod", "imports"),
            _make_relation("ent-MyClass", "ent-other", "uses"),
        ]
        pr = _make_parse_result("python", [entity], rels)
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.coupling_factor > 0.0

    def test_count_parameters_filters_self(self):
        entity = _make_entity("meth", "method", "python", signature="def meth(self, x, y)")
        assert self.analyzer._count_parameters(entity) == 2

    def test_count_parameters_empty(self):
        entity = _make_entity("fn", "function", "python", signature="def fn()")
        assert self.analyzer._count_parameters(entity) == 0

    def test_classify_complexity(self):
        assert self.analyzer._classify_complexity(3) == ComplexityLevel.TRIVIAL
        assert self.analyzer._classify_complexity(8) == ComplexityLevel.BASIC
        assert self.analyzer._classify_complexity(15) == ComplexityLevel.MODERATE
        assert self.analyzer._classify_complexity(30) == ComplexityLevel.COMPLEX
        assert self.analyzer._classify_complexity(50) == ComplexityLevel.VERY_COMPLEX

    def test_maintainability_magic_method(self):
        entity = _make_entity("__init__", "method", "python", signature="def __init__(self)")
        pr = _make_parse_result("python", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        # Should produce maintainability
        assert metrics.maintainability_index >= 0.0

    def test_suggestions_python(self):
        entity = _make_entity(
            "fn",
            "function",
            "python",
            properties={
                "decorators": ["a", "b", "c", "d"],
                "has_comprehension": True,
            },
        )
        metrics = ServiceComplexityMetrics(
            cyclomatic_complexity=20, parameter_count=7, coupling_factor=0.9
        )
        suggestions = self.analyzer._generate_python_suggestions(metrics, entity)
        assert any("smaller" in s for s in suggestions)
        assert any("kwargs" in s for s in suggestions)
        assert any("decorator stacking" in s.lower() for s in suggestions)
        assert any("comprehension" in s.lower() for s in suggestions)

    def test_suggestions_no_type_hints(self):
        # properties must be truthy (non-empty) for the check to fire
        entity = _make_entity("fn", "function", "python", properties={"some_key": True})
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=2, parameter_count=1)
        suggestions = self.analyzer._generate_python_suggestions(metrics, entity)
        assert any("type hints" in s for s in suggestions)

    def test_suggestions_magic_method(self):
        entity = _make_entity("__repr__", "function", "python")
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=2, parameter_count=0)
        suggestions = self.analyzer._generate_python_suggestions(metrics, entity)
        assert any("Magic methods" in s for s in suggestions)

    def test_analyze_with_metaclass(self):
        entity = _make_entity("mc_fn", "function", "python", properties={"has_metaclass": True})
        pr = _make_parse_result("python", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "metaclass" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_with_generator(self):
        entity = _make_entity("gen_fn", "function", "python", properties={"is_generator": True})
        pr = _make_parse_result("python", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "generator" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_maintainability_with_type_hints(self):
        entity = _make_entity("typed_fn", "function", "python", properties={"has_type_hints": True})
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=1, parameter_count=0)
        mi = self.analyzer._calculate_python_maintainability(metrics, entity)
        # Type hints give +5 bonus
        assert mi > 100.0 or mi == 100.0  # clamped to 100


# ===================================================================
# Rust analyzer
# ===================================================================
class TestRustComplexityAnalyzer:
    def setup_method(self):
        self.analyzer = RustComplexityAnalyzer()

    def test_get_language_specific_factors(self):
        factors = self.analyzer.get_language_specific_factors()
        assert "lifetime_penalty" in factors
        assert "unsafe_penalty" in factors

    def test_analyze_function(self):
        entity = _make_entity("do_work", "function", "rust", signature="fn do_work(x: i32, y: i32)")
        pr = _make_parse_result("rust", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.language == "rust"
        assert metrics.parameter_count == 2

    def test_analyze_async_function(self):
        entity = _make_entity(
            "async_fn", "function", "rust", signature="async fn async_fn()", is_async=True
        )
        pr = _make_parse_result("rust", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "async" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_struct_coupling(self):
        entity = _make_entity("MyStruct", "struct", "rust")
        rels = [
            _make_relation("ent-MyStruct", "ent-trait1", "implements"),
            _make_relation("ent-MyStruct", "ent-trait2", "trait_bound"),
        ]
        pr = _make_parse_result("rust", [entity], rels)
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.coupling_factor > 0.0

    def test_count_parameters_filters_self(self):
        entity = _make_entity("meth", "function", "rust", signature="fn meth(&self, x: i32)")
        assert self.analyzer._count_parameters(entity) == 1

    def test_classify_complexity(self):
        assert self.analyzer._classify_complexity(3) == ComplexityLevel.TRIVIAL
        assert self.analyzer._classify_complexity(6) == ComplexityLevel.BASIC
        assert self.analyzer._classify_complexity(12) == ComplexityLevel.MODERATE
        assert self.analyzer._classify_complexity(20) == ComplexityLevel.COMPLEX
        assert self.analyzer._classify_complexity(40) == ComplexityLevel.VERY_COMPLEX

    def test_maintainability_unsafe(self):
        entity = _make_entity("fn", "function", "rust", properties={"has_unsafe": True})
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=1)
        mi = self.analyzer._calculate_rust_maintainability(metrics, entity)
        assert mi < 100.0

    def test_maintainability_macro(self):
        entity = _make_entity("fn", "function", "rust", properties={"macro_usage": 2})
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=1)
        mi = self.analyzer._calculate_rust_maintainability(metrics, entity)
        assert mi < 100.0

    def test_suggestions_rust(self):
        entity = _make_entity("fn", "function", "rust", properties={"has_unsafe": True})
        metrics = ServiceComplexityMetrics(
            cyclomatic_complexity=10, parameter_count=6, coupling_factor=0.9
        )
        suggestions = self.analyzer._generate_rust_suggestions(metrics, entity)
        assert any("smaller" in s for s in suggestions)
        assert any("struct" in s for s in suggestions)
        assert any("unsafe" in s for s in suggestions)
        assert any("coupling" in s.lower() for s in suggestions)


# ===================================================================
# Go analyzer
# ===================================================================
class TestGoComplexityAnalyzer:
    def setup_method(self):
        self.analyzer = GoComplexityAnalyzer()

    def test_get_language_specific_factors(self):
        factors = self.analyzer.get_language_specific_factors()
        assert "goroutine_penalty" in factors
        assert "channel_complexity" in factors

    def test_analyze_function(self):
        entity = _make_entity("doWork", "function", "go", signature="func doWork(x int, y string)")
        pr = _make_parse_result("go", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.language == "go"
        assert metrics.parameter_count == 2

    def test_analyze_struct_coupling(self):
        entity = _make_entity("MyStruct", "struct", "go")
        rels = [_make_relation("ent-MyStruct", "ent-iface", "implements")]
        pr = _make_parse_result("go", [entity], rels)
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.coupling_factor > 0.0

    def test_classify_complexity(self):
        assert self.analyzer._classify_complexity(3) == ComplexityLevel.TRIVIAL
        assert self.analyzer._classify_complexity(8) == ComplexityLevel.BASIC
        assert self.analyzer._classify_complexity(12) == ComplexityLevel.MODERATE
        assert self.analyzer._classify_complexity(20) == ComplexityLevel.COMPLEX
        assert self.analyzer._classify_complexity(30) == ComplexityLevel.VERY_COMPLEX

    def test_maintainability(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=10, parameter_count=5)
        mi = self.analyzer._calculate_go_maintainability(metrics)
        assert 0.0 <= mi <= 100.0

    def test_suggestions_goroutine(self):
        entity = _make_entity("fn", "function", "go", signature="func fn() goroutine")
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=15, parameter_count=7)
        suggestions = self.analyzer._generate_go_suggestions(metrics, entity)
        assert any("splitting" in s for s in suggestions)
        assert any("struct" in s for s in suggestions)
        assert any("goroutine" in s for s in suggestions)


# ===================================================================
# TypeScript analyzer
# ===================================================================
class TestTypeScriptComplexityAnalyzer:
    def setup_method(self):
        self.analyzer = TypeScriptComplexityAnalyzer()

    def test_get_language_specific_factors(self):
        factors = self.analyzer.get_language_specific_factors()
        assert "type_system_penalty" in factors

    def test_analyze_interface(self):
        entity = _make_entity("IMyInterface", "interface", "typescript")
        rels = [_make_relation("ent-IMyInterface", "ent-other", "extends")]
        pr = _make_parse_result("typescript", [entity], rels)
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.coupling_factor > 0.0

    def test_analyze_class_with_decorators(self):
        entity = _make_entity(
            "MyClass",
            "class",
            "typescript",
            properties={"decorators": ["Injectable", "Component"]},
        )
        pr = _make_parse_result("typescript", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "decorators" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_generics(self):
        entity = _make_entity(
            "fn",
            "function",
            "typescript",
            properties={"generic_parameters": ["T", "U"], "parameters": "x: T, y: U"},
        )
        pr = _make_parse_result("typescript", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.parameter_count == 2

    def test_analyze_conditional_type(self):
        entity = _make_entity(
            "MyType",
            "type_alias",
            "typescript",
            properties={"is_conditional_type": True},
        )
        pr = _make_parse_result("typescript", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "conditional_type" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_mapped_type(self):
        entity = _make_entity(
            "Mapped", "type_alias", "typescript", properties={"is_mapped_type": True}
        )
        pr = _make_parse_result("typescript", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "mapped_type" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_classify_complexity(self):
        assert self.analyzer._classify_complexity(3) == ComplexityLevel.TRIVIAL
        assert self.analyzer._classify_complexity(9) == ComplexityLevel.BASIC
        assert self.analyzer._classify_complexity(20) == ComplexityLevel.MODERATE
        assert self.analyzer._classify_complexity(40) == ComplexityLevel.COMPLEX
        assert self.analyzer._classify_complexity(60) == ComplexityLevel.VERY_COMPLEX

    def test_count_typescript_parameters_from_properties(self):
        entity = _make_entity("fn", "function", "typescript", properties={"parameters": "a, b, c"})
        assert self.analyzer._count_typescript_parameters(entity) == 3

    def test_count_typescript_parameters_empty(self):
        entity = _make_entity("fn", "function", "typescript", properties={})
        assert self.analyzer._count_typescript_parameters(entity) == 0

    def test_maintainability_exported_generics(self):
        entity = _make_entity(
            "fn",
            "function",
            "typescript",
            properties={
                "is_exported": True,
                "generic_parameters": ["T", "U", "V"],
                "decorators": ["a", "b"],
            },
        )
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=5, parameter_count=3)
        mi = self.analyzer._calculate_typescript_maintainability(metrics, entity)
        assert 0.0 <= mi <= 100.0

    def test_suggestions_typescript(self):
        entity = _make_entity(
            "fn",
            "function",
            "typescript",
            properties={
                "generic_parameters": ["T", "U", "V", "W"],
                "is_conditional_type": True,
                "is_mapped_type": True,
                "decorators": ["a", "b", "c"],
            },
        )
        metrics = ServiceComplexityMetrics(
            cyclomatic_complexity=15, parameter_count=6, coupling_factor=0.9
        )
        suggestions = self.analyzer._generate_typescript_suggestions(metrics, entity)
        assert any("smaller" in s for s in suggestions)
        assert any("options" in s for s in suggestions)
        assert any("generic" in s.lower() for s in suggestions)
        assert any("conditional" in s.lower() for s in suggestions)
        assert any("mapped" in s.lower() for s in suggestions)
        assert any("coupling" in s.lower() for s in suggestions)

    def test_suggestions_decorators(self):
        entity = _make_entity(
            "fn",
            "function",
            "typescript",
            properties={"decorators": ["a", "b", "c"]},
        )
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=2, parameter_count=1)
        suggestions = self.analyzer._generate_typescript_suggestions(metrics, entity)
        assert any("decorator" in s.lower() for s in suggestions)


# ===================================================================
# JavaScript analyzer
# ===================================================================
class TestJavaScriptComplexityAnalyzer:
    def setup_method(self):
        self.analyzer = JavaScriptComplexityAnalyzer()

    def test_get_language_specific_factors(self):
        factors = self.analyzer.get_language_specific_factors()
        assert "callback_nesting_penalty" in factors

    def test_analyze_function_callbacks(self):
        entity = _make_entity(
            "fn",
            "function",
            "javascript",
            signature="function fn(cb)",
            properties={"has_callbacks": True},
        )
        pr = _make_parse_result("javascript", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "callbacks" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_promises(self):
        entity = _make_entity(
            "fn",
            "function",
            "javascript",
            properties={"has_promises": True},
        )
        pr = _make_parse_result("javascript", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "promises" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_closure_depth(self):
        entity = _make_entity("fn", "function", "javascript", properties={"closure_depth": 3})
        pr = _make_parse_result("javascript", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "closures" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_iife(self):
        entity = _make_entity("fn", "function", "javascript", properties={"is_iife": True})
        pr = _make_parse_result("javascript", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "iife" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_async_function(self):
        entity = _make_entity(
            "fn", "function", "javascript", signature="async function fn()", is_async=True
        )
        pr = _make_parse_result("javascript", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "async" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_arrow_function(self):
        entity = _make_entity("fn", "arrow_function", "javascript", signature="(x) => x")
        pr = _make_parse_result("javascript", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.entity_type == "arrow_function"

    def test_analyze_class_coupling(self):
        entity = _make_entity("MyClass", "class", "javascript")
        rels = [
            _make_relation("ent-MyClass", "ent-base", "extends"),
            _make_relation("ent-MyClass", "ent-mod", "imports"),
            _make_relation("ent-MyClass", "ent-other", "uses"),
        ]
        pr = _make_parse_result("javascript", [entity], rels)
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.coupling_factor > 0.0

    def test_count_parameters_filters_rest(self):
        entity = _make_entity("fn", "function", "javascript", signature="function fn(a, b, ...)")
        assert self.analyzer._count_parameters(entity) == 2

    def test_classify_complexity(self):
        assert self.analyzer._classify_complexity(3) == ComplexityLevel.TRIVIAL
        assert self.analyzer._classify_complexity(8) == ComplexityLevel.BASIC
        assert self.analyzer._classify_complexity(15) == ComplexityLevel.MODERATE
        assert self.analyzer._classify_complexity(30) == ComplexityLevel.COMPLEX
        assert self.analyzer._classify_complexity(40) == ComplexityLevel.VERY_COMPLEX

    def test_maintainability_callback_hell(self):
        entity = _make_entity(
            "fn",
            "function",
            "javascript",
            properties={"has_callbacks": True, "closure_depth": 4, "modifies_prototype": True},
        )
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=20, parameter_count=3)
        mi = self.analyzer._calculate_javascript_maintainability(metrics, entity)
        assert 0.0 <= mi <= 100.0

    def test_suggestions_javascript(self):
        entity = _make_entity(
            "fn",
            "function",
            "javascript",
            properties={
                "has_callbacks": True,
                "has_promises": True,
                "closure_depth": 4,
                "modifies_prototype": True,
            },
        )
        metrics = ServiceComplexityMetrics(
            cyclomatic_complexity=15, parameter_count=6, coupling_factor=0.9
        )
        suggestions = self.analyzer._generate_javascript_suggestions(metrics, entity)
        assert any("single-purpose" in s for s in suggestions)
        assert any("options" in s for s in suggestions)
        assert any("async/await" in s for s in suggestions)
        assert any("closure" in s.lower() for s in suggestions)
        assert any("prototype" in s.lower() for s in suggestions)
        assert any("coupling" in s.lower() for s in suggestions)


# ===================================================================
# C++ analyzer
# ===================================================================
class TestCppComplexityAnalyzer:
    def setup_method(self):
        self.analyzer = CppComplexityAnalyzer()

    def test_get_language_specific_factors(self):
        factors = self.analyzer.get_language_specific_factors()
        assert "template_penalty" in factors
        assert "raw_pointer_penalty" in factors

    def test_analyze_function_template(self):
        entity = _make_entity(
            "fn",
            "function",
            "cpp",
            signature="void fn(int x)",
            properties={"template_parameters": ["T", "U"]},
        )
        pr = _make_parse_result("cpp", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.parameter_count == 1

    def test_analyze_function_operator_overload(self):
        entity = _make_entity(
            "op",
            "function",
            "cpp",
            properties={"is_operator_overload": True},
        )
        pr = _make_parse_result("cpp", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "operator_overload" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_resources_exceptions_move_pointers(self):
        entity = _make_entity(
            "fn",
            "function",
            "cpp",
            properties={
                "manages_resources": True,
                "has_exceptions": True,
                "has_move_semantics": True,
                "uses_raw_pointers": True,
            },
        )
        pr = _make_parse_result("cpp", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "manages_resources" in metrics.risk_factors
        assert "exception_handling" in metrics.risk_factors
        assert "move_semantics" in metrics.risk_factors
        assert "raw_pointers" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_class_multiple_inheritance(self):
        entity = _make_entity(
            "MyClass",
            "class",
            "cpp",
            properties={"inheritance_count": 3, "has_virtual_functions": True},
        )
        rels = [_make_relation("ent-MyClass", "ent-base", "inherits")]
        pr = _make_parse_result("cpp", [entity], rels)
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.coupling_factor > 0.0

    def test_count_parameters_void(self):
        entity = _make_entity("fn", "function", "cpp", signature="void fn(void)")
        assert self.analyzer._count_parameters(entity) == 0

    def test_classify_complexity(self):
        assert self.analyzer._classify_complexity(3) == ComplexityLevel.TRIVIAL
        assert self.analyzer._classify_complexity(6) == ComplexityLevel.BASIC
        assert self.analyzer._classify_complexity(12) == ComplexityLevel.MODERATE
        assert self.analyzer._classify_complexity(25) == ComplexityLevel.COMPLEX
        assert self.analyzer._classify_complexity(35) == ComplexityLevel.VERY_COMPLEX

    def test_maintainability_smart_pointers_const(self):
        entity = _make_entity(
            "fn",
            "function",
            "cpp",
            properties={
                "template_parameters": ["T", "U", "V"],
                "inheritance_count": 3,
                "uses_raw_pointers": True,
                "uses_smart_pointers": True,
                "const_correct": True,
            },
        )
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=5, parameter_count=3)
        mi = self.analyzer._calculate_cpp_maintainability(metrics, entity)
        assert 0.0 <= mi <= 100.0

    def test_suggestions_cpp(self):
        entity = _make_entity(
            "fn",
            "function",
            "cpp",
            properties={
                "template_parameters": ["T", "U", "V"],
                "inheritance_count": 2,
                "uses_raw_pointers": True,
                "is_operator_overload": True,
                "has_exceptions": True,
            },
        )
        metrics = ServiceComplexityMetrics(
            cyclomatic_complexity=10, parameter_count=6, coupling_factor=0.9
        )
        suggestions = self.analyzer._generate_cpp_suggestions(metrics, entity)
        assert any("smaller" in s for s in suggestions)
        assert any("parameter struct" in s for s in suggestions)
        assert any("template" in s.lower() for s in suggestions)
        assert any("inheritance" in s.lower() for s in suggestions)
        assert any("smart pointer" in s.lower() for s in suggestions)
        assert any("operator" in s.lower() for s in suggestions)
        assert any("exception" in s.lower() for s in suggestions)

    def test_suggestions_rule_of_five(self):
        entity = _make_entity(
            "ResourceClass",
            "class",
            "cpp",
            properties={"manages_resources": True},
        )
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=2, coupling_factor=0.1)
        suggestions = self.analyzer._generate_cpp_suggestions(metrics, entity)
        assert any("Rule of Five" in s for s in suggestions)

    def test_class_coupling_various_types(self):
        entity = _make_entity("C", "class", "cpp")
        rels = [
            _make_relation("ent-C", "ent-b", "inherits"),
            _make_relation("ent-C", "ent-t", "template_specialization"),
            _make_relation("ent-C", "ent-i", "includes"),
            _make_relation("ent-C", "ent-o", "uses"),
        ]
        pr = _make_parse_result("cpp", [entity], rels)
        coupling = self.analyzer._analyze_class_coupling(entity, pr)
        assert coupling > 0.0


# ===================================================================
# C# analyzer
# ===================================================================
class TestCSharpComplexityAnalyzer:
    def setup_method(self):
        self.analyzer = CSharpComplexityAnalyzer()

    def test_get_language_specific_factors(self):
        factors = self.analyzer.get_language_specific_factors()
        assert "linq_penalty" in factors
        assert "reflection_penalty" in factors

    def test_analyze_method_linq(self):
        entity = _make_entity(
            "DoWork",
            "method",
            "csharp",
            signature="void DoWork(int x)",
            properties={"has_linq": True, "linq_depth": 2},
        )
        pr = _make_parse_result("csharp", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.parameter_count == 1

    def test_analyze_method_delegates_reflection(self):
        entity = _make_entity(
            "Process",
            "method",
            "csharp",
            properties={"has_delegates": True, "uses_reflection": True},
            is_async=True,
        )
        pr = _make_parse_result("csharp", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "async" in metrics.risk_factors
        assert "delegates" in metrics.risk_factors
        assert "reflection" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_extension_method(self):
        entity = _make_entity(
            "Ext",
            "method",
            "csharp",
            properties={"is_extension_method": True},
        )
        pr = _make_parse_result("csharp", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "extension_method" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_class_generics_attributes(self):
        entity = _make_entity(
            "MyClass",
            "class",
            "csharp",
            properties={
                "generic_parameters": ["T", "U"],
                "attributes": ["Serializable", "DataContract", "Injectable"],
            },
        )
        rels = [
            _make_relation("ent-MyClass", "ent-base", "inherits"),
            _make_relation("ent-MyClass", "ent-iface", "implements"),
            _make_relation("ent-MyClass", "ent-dep", "uses"),
            _make_relation("ent-MyClass", "ent-other", "references"),
        ]
        pr = _make_parse_result("csharp", [entity], rels)
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.coupling_factor > 0.0

    def test_analyze_property(self):
        entity = _make_entity(
            "Name",
            "property",
            "csharp",
            properties={"has_getter": True, "has_setter": True},
        )
        pr = _make_parse_result("csharp", [entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "property_getter_setter" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_count_parameters_filters_this(self):
        entity = _make_entity("Ext", "method", "csharp", signature="void Ext(this string s, int x)")
        assert self.analyzer._count_parameters(entity) == 1

    def test_classify_complexity(self):
        assert self.analyzer._classify_complexity(3) == ComplexityLevel.TRIVIAL
        assert self.analyzer._classify_complexity(8) == ComplexityLevel.BASIC
        assert self.analyzer._classify_complexity(15) == ComplexityLevel.MODERATE
        assert self.analyzer._classify_complexity(30) == ComplexityLevel.COMPLEX
        assert self.analyzer._classify_complexity(50) == ComplexityLevel.VERY_COMPLEX

    def test_maintainability_csharp(self):
        entity = _make_entity(
            "fn",
            "method",
            "csharp",
            properties={
                "has_linq": True,
                "linq_depth": 4,
                "uses_reflection": True,
                "has_delegates": True,
                "delegate_count": 3,
                "is_auto_property": True,
                "implements_interfaces": True,
            },
            is_async=True,
        )
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=15, parameter_count=3)
        mi = self.analyzer._calculate_csharp_maintainability(metrics, entity)
        assert 0.0 <= mi <= 100.0

    def test_maintainability_auto_property(self):
        entity = _make_entity(
            "Name",
            "property",
            "csharp",
            properties={"is_auto_property": True},
        )
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=1, parameter_count=0)
        mi = self.analyzer._calculate_csharp_maintainability(metrics, entity)
        assert mi >= 95.0  # very high

    def test_maintainability_class_implements_interfaces(self):
        entity = _make_entity(
            "Cls",
            "class",
            "csharp",
            properties={"implements_interfaces": True},
        )
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=1, parameter_count=0)
        mi = self.analyzer._calculate_csharp_maintainability(metrics, entity)
        assert mi >= 95.0

    def test_suggestions_csharp(self):
        entity = _make_entity(
            "fn",
            "method",
            "csharp",
            properties={
                "has_linq": True,
                "linq_depth": 4,
                "uses_reflection": True,
                "has_delegates": True,
                "delegate_count": 3,
                "generic_parameters": ["T", "U", "V"],
                "attributes": ["a", "b", "c", "d"],
            },
            is_async=True,
        )
        metrics = ServiceComplexityMetrics(
            cyclomatic_complexity=15, parameter_count=7, coupling_factor=0.9
        )
        suggestions = self.analyzer._generate_csharp_suggestions(metrics, entity)
        assert any("smaller" in s for s in suggestions)
        assert any("parameter object" in s for s in suggestions)
        assert any("linq" in s.lower() for s in suggestions)
        assert any("async" in s.lower() for s in suggestions)
        assert any("reflection" in s.lower() for s in suggestions)
        assert any("delegate" in s.lower() for s in suggestions)
        assert any("generic" in s.lower() for s in suggestions)
        assert any("attribute" in s.lower() for s in suggestions)
        assert any("coupling" in s.lower() for s in suggestions)

    def test_suggestions_property_complex(self):
        # properties must be non-empty (truthy) for the inner check to fire
        entity = _make_entity(
            "Prop",
            "property",
            "csharp",
            properties={"some_key": True},
        )
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=5)
        suggestions = self.analyzer._generate_csharp_suggestions(metrics, entity)
        assert any("property" in s.lower() for s in suggestions)


# ===================================================================
# Base analyzer methods
# ===================================================================
class TestLanguageComplexityAnalyzerBase:
    """Test the base class methods via a concrete subclass."""

    def setup_method(self):
        self.analyzer = PythonComplexityAnalyzer()
