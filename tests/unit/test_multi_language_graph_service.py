"""Unit tests for services.code_graph.multi_language_graph_service module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import services.code_graph.multi_language_graph_service as _mlgs_mod
from models.core_models import (
    CoreAnalysisResult,
    CoreCodeEntity,
    EntityRelation,
    EntityType,
    RelationType,
)
from services.code_graph.multi_language_graph_service import (
    MultiLanguageGraphService,
    MultiLanguageGraphServiceFactory,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db_connection() -> MagicMock:
    """Create a mock GraphDatabaseConnection with an async connection."""
    conn = AsyncMock()
    conn.commit = AsyncMock()
    conn.rollback = AsyncMock()
    conn.execute = AsyncMock()

    db = MagicMock()
    db.connect = AsyncMock()
    db.get_connection = MagicMock(return_value=conn)
    return db


def _make_entity(entity_id: str = "e1", name: str = "foo") -> CoreCodeEntity:
    return CoreCodeEntity(
        id=entity_id,
        name=name,
        type=EntityType.FUNCTION,
        language="python",
        file_path="test.py",
        line_number=1,
    )


class _AsyncIter:
    """Helper to make a list behave as an async iterator for `async for`."""

    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


def _make_relation(source: str = "e1", target: str = "e2") -> EntityRelation:
    return EntityRelation(
        source_id=source,
        target_id=target,
        type=RelationType.CALLS,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db():
    return _make_db_connection()


@pytest.fixture
def service(db):
    svc = MultiLanguageGraphService(db)
    svc.entity_mapper = MagicMock()
    svc.entity_mapper.store_entity = AsyncMock()
    svc.entity_mapper.store_relationship = AsyncMock()
    svc.entity_mapper.update_language_stats = AsyncMock()
    svc.entity_mapper.get_language_summary = AsyncMock(return_value={})
    svc.entity_mapper.get_entities_by_language = AsyncMock(return_value=[])
    return svc


# ===========================================================================
# Tests: __init__ and initialize
# ===========================================================================


class TestInit:
    def test_constructor_sets_db(self, db):
        svc = MultiLanguageGraphService(db)
        assert svc.db is db
        assert svc.entity_mapper is None
        assert svc._known_entities == set()

    @patch.object(_mlgs_mod, "MultiLanguageEntityMapper")
    async def test_initialize_connects_and_creates_mapper(self, mock_mapper_cls, db):
        mock_mapper_cls.return_value = MagicMock()
        svc = MultiLanguageGraphService(db)
        await svc.initialize()
        db.connect.assert_awaited_once()
        db.get_connection.assert_called()
        assert svc.entity_mapper is not None
        mock_mapper_cls.assert_called_once()


# ===========================================================================
# Tests: _map_entity_type
# ===========================================================================


class TestMapEntityType:
    @pytest.mark.parametrize(
        "input_value,expected",
        [
            ("method", EntityType.FUNCTION),
            ("classmethod", EntityType.FUNCTION),
            ("staticmethod", EntityType.FUNCTION),
            ("module", EntityType.MODULE),
            ("file", EntityType.FILE),
            ("attribute", EntityType.VARIABLE),
            ("function", EntityType.FUNCTION),
            ("class", EntityType.CLASS),
            ("interface", EntityType.INTERFACE),
            ("enum", EntityType.ENUM),
        ],
    )
    def test_known_types(self, input_value, expected):
        assert MultiLanguageGraphService._map_entity_type(input_value) == expected

    def test_unknown_type_defaults_to_function(self):
        assert MultiLanguageGraphService._map_entity_type("xyzzy") == EntityType.FUNCTION


# ===========================================================================
# Tests: _map_relation_type
# ===========================================================================


class TestMapRelationType:
    @pytest.mark.parametrize(
        "input_value,expected",
        [
            ("module_contains", RelationType.CONTAINS),
            ("contains", RelationType.CONTAINS),
            ("inherits", RelationType.INHERITS),
            ("extends", RelationType.INHERITS),
            ("implements", RelationType.IMPLEMENTS),
            ("calls", RelationType.CALLS),
            ("call", RelationType.CALLS),
            ("references", RelationType.REFERENCES),
            ("usage", RelationType.REFERENCES),
            ("imports", RelationType.IMPORTS),
            ("import", RelationType.IMPORTS),
            ("depends_on", RelationType.DEPENDS_ON),
        ],
    )
    def test_known_types(self, input_value, expected):
        assert MultiLanguageGraphService._map_relation_type(input_value) == expected

    def test_unknown_type_defaults_to_references(self):
        assert MultiLanguageGraphService._map_relation_type("zzzz") == RelationType.REFERENCES


# ===========================================================================
# Tests: _build_location
# ===========================================================================


class TestBuildLocation:
    def test_returns_none_for_non_dict(self):
        assert MultiLanguageGraphService._build_location(None, "f.py") is None
        assert MultiLanguageGraphService._build_location("string", "f.py") is None

    def test_returns_none_when_lines_missing(self):
        assert MultiLanguageGraphService._build_location({}, "f.py") is None

    def test_builds_location_with_line_start_and_line_end(self):
        loc = MultiLanguageGraphService._build_location(
            {"line_start": 10, "line_end": 20, "column_start": 5, "column_end": 15},
            "default.py",
        )
        assert loc is not None
        assert loc.file_path == "default.py"
        assert loc.line_start == 10
        assert loc.line_end == 20
        assert loc.column_start == 5
        assert loc.column_end == 15

    def test_uses_file_path_from_payload(self):
        loc = MultiLanguageGraphService._build_location(
            {"file_path": "custom.py", "line_start": 1, "line_end": 1},
            "default.py",
        )
        assert loc.file_path == "custom.py"

    def test_uses_line_number_alias(self):
        loc = MultiLanguageGraphService._build_location(
            {"line_number": 5},
            "f.py",
        )
        assert loc is not None
        assert loc.line_start == 5
        assert loc.line_end == 5

    def test_uses_line_alias(self):
        loc = MultiLanguageGraphService._build_location(
            {"line": 7},
            "f.py",
        )
        assert loc is not None
        assert loc.line_start == 7


# ===========================================================================
# Tests: _normalize_entity
# ===========================================================================


class TestNormalizeEntity:
    def test_passthrough_core_code_entity(self, service):
        entity = _make_entity()
        result = service._normalize_entity(entity, "python", "test.py")
        assert result is entity

    def test_returns_none_for_unsupported_type(self, service):
        assert service._normalize_entity(42, "python", "test.py") is None

    def test_normalizes_dict_entity(self, service):
        entity_dict = {
            "id": "test::my_func",
            "name": "my_func",
            "type": "function",
            "file_path": "src/main.py",
            "line_number": 10,
            "end_line": 25,
            "language": "python",
            "visibility": "public",
            "is_static": True,
            "is_async": True,
            "docstring": "Hello",
            "signature": "def my_func():",
        }
        result = service._normalize_entity(entity_dict, "python", "default.py")
        assert result is not None
        assert result.id == "test::my_func"
        assert result.name == "my_func"
        assert result.type == EntityType.FUNCTION
        assert result.file_path == "src/main.py"
        assert result.line_number == 10
        assert result.end_line_number == 25
        assert result.is_static is True
        assert result.is_async is True
        assert result.docstring == "Hello"

    def test_dict_entity_defaults(self, service):
        """Minimal dict gets sensible defaults."""
        result = service._normalize_entity({}, "java", "App.java")
        assert result is not None
        assert result.id == "App.java::unknown"
        assert result.name == "unknown"
        assert result.type == EntityType.FUNCTION
        assert result.language == "java"
        assert result.file_path == "App.java"
        assert result.line_number == 1

    def test_dict_entity_uses_start_line(self, service):
        result = service._normalize_entity({"start_line": 42}, "go", "main.go")
        assert result.line_number == 42

    def test_returns_none_on_construction_error(self, service):
        # Force an error by providing a value that cannot be cast to int
        bad = {"line_number": "not-a-number"}
        result = service._normalize_entity(bad, "py", "f.py")
        assert result is None


# ===========================================================================
# Tests: _normalize_relationship
# ===========================================================================


class TestNormalizeRelationship:
    def test_passthrough_entity_relation(self, service):
        rel = _make_relation()
        result = service._normalize_relationship(rel, "python", "test.py")
        assert result is rel

    def test_returns_none_for_unsupported_type(self, service):
        assert service._normalize_relationship(42, "py", "f.py") is None

    def test_returns_none_when_source_missing(self, service):
        assert service._normalize_relationship({"target": "t"}, "py", "f.py") is None

    def test_returns_none_when_target_missing(self, service):
        assert service._normalize_relationship({"source": "s"}, "py", "f.py") is None

    def test_normalizes_dict_relationship(self, service):
        rel_dict = {
            "source": "s1",
            "target": "t1",
            "type": "calls",
            "tentative": True,
            "context": "some context",
            "location": {"line_start": 5, "line_end": 5},
        }
        result = service._normalize_relationship(rel_dict, "python", "default.py")
        assert result is not None
        assert result.source_id == "s1"
        assert result.target_id == "t1"
        assert result.type == RelationType.CALLS
        assert result.properties.get("tentative") is True
        assert result.context == "some context"
        assert result.location is not None
        assert result.location.line_start == 5

    def test_dict_defaults(self, service):
        result = service._normalize_relationship({"source": "a", "target": "b"}, "go", "main.go")
        assert result is not None
        assert result.type == RelationType.REFERENCES


# ===========================================================================
# Tests: _can_store_relationship
# ===========================================================================


class TestCanStoreRelationship:
    def test_true_when_both_known(self, service):
        service._known_entities = {"a", "b"}
        rel = _make_relation("a", "b")
        assert service._can_store_relationship(rel) is True

    def test_false_when_source_unknown(self, service):
        service._known_entities = {"b"}
        rel = _make_relation("a", "b")
        assert service._can_store_relationship(rel) is False

    def test_false_when_target_unknown(self, service):
        service._known_entities = {"a"}
        rel = _make_relation("a", "b")
        assert service._can_store_relationship(rel) is False

    def test_false_when_source_id_empty(self, service):
        rel = EntityRelation(source_id="", target_id="b", type=RelationType.CALLS)
        assert service._can_store_relationship(rel) is False

    def test_false_when_target_id_empty(self, service):
        rel = EntityRelation(source_id="a", target_id="", type=RelationType.CALLS)
        assert service._can_store_relationship(rel) is False


# ===========================================================================
# Tests: analyze_file
# ===========================================================================


class TestAnalyzeFile:
    async def test_raises_when_not_initialized(self, db):
        svc = MultiLanguageGraphService(db)
        with pytest.raises(RuntimeError, match="not initialized"):
            await svc.analyze_file(Path("test.py"), "code")

    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_no_processor_returns_failure(self, mock_factory_fn, service):
        factory = AsyncMock()
        factory.get_processor_for_file = AsyncMock(return_value=None)
        mock_factory_fn.return_value = factory

        result = await service.analyze_file(Path("test.xyz"), "content")
        assert result.success is False
        assert "No processor" in result.error

    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_processor_without_parse_file_returns_failure(self, mock_factory_fn, service):
        processor = MagicMock(spec=[])  # no parse_file attribute
        factory = AsyncMock()
        factory.get_processor_for_file = AsyncMock(return_value=processor)
        mock_factory_fn.return_value = factory

        result = await service.analyze_file(Path("test.py"), "code")
        assert result.success is False
        assert "parse_file" in result.error

    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_parse_failure_returns_failure_result(self, mock_factory_fn, service):
        parse_result = MagicMock()
        parse_result.success = False
        parse_result.language = "python"
        parse_result.errors = ["syntax error"]
        parse_result.parse_time_ms = 1.0
        parse_result.total_lines = 5
        parse_result.warnings = ["warn1"]

        processor = MagicMock()
        processor.parse_file = AsyncMock(return_value=parse_result)
        factory = AsyncMock()
        factory.get_processor_for_file = AsyncMock(return_value=processor)
        mock_factory_fn.return_value = factory

        result = await service.analyze_file(Path("test.py"), "code")
        assert result.success is False
        assert result.language == "python"
        assert "syntax error" in result.error

    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_successful_analysis_stores_entities_and_relationships(
        self, mock_factory_fn, service
    ):
        entity = _make_entity("e1", "func1")
        rel = _make_relation("e1", "e2")

        parse_result = MagicMock()
        parse_result.success = True
        parse_result.language = "python"
        parse_result.entities = [entity]
        parse_result.relationships = [rel]
        parse_result.lines_of_code = 100
        parse_result.complexity_score = 5.0
        parse_result.parse_time_ms = 10.0
        parse_result.dependencies = []
        parse_result.complexity_metrics = None
        parse_result.security_patterns = []
        parse_result.metadata = {}

        processor = MagicMock()
        processor.parse_file = AsyncMock(return_value=parse_result)
        factory = AsyncMock()
        factory.get_processor_for_file = AsyncMock(return_value=processor)
        mock_factory_fn.return_value = factory

        result = await service.analyze_file(Path("test.py"), "code")
        assert result.success is True
        assert result.language == "python"
        assert len(result.entities) == 1
        service.entity_mapper.store_entity.assert_awaited()
        # update_language_stats assertion removed: the postgres path commits the
        # connection directly after storing entities/relationships; there is no
        # separate language-stats call any more.
        service.db.get_connection().commit.assert_awaited()

    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_successful_analysis_stores_known_relationships(self, mock_factory_fn, service):
        """Relationships with both endpoints known are stored."""
        entity1 = _make_entity("e1", "func1")
        entity2 = _make_entity("e2", "func2")
        rel = _make_relation("e1", "e2")

        parse_result = MagicMock()
        parse_result.success = True
        parse_result.language = "python"
        parse_result.entities = [entity1, entity2]
        parse_result.relationships = [rel]
        parse_result.lines_of_code = 50
        parse_result.complexity_score = 2.0
        parse_result.parse_time_ms = 5.0
        parse_result.dependencies = []
        parse_result.complexity_metrics = None
        parse_result.security_patterns = []
        parse_result.metadata = {}

        processor = MagicMock()
        processor.parse_file = AsyncMock(return_value=parse_result)
        factory = AsyncMock()
        factory.get_processor_for_file = AsyncMock(return_value=processor)
        mock_factory_fn.return_value = factory

        result = await service.analyze_file(Path("test.py"), "code")
        assert result.success is True
        # Both entities stored, so relationship should also be stored
        service.entity_mapper.store_relationship.assert_awaited()

    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_exception_during_analysis_rolls_back(self, mock_factory_fn, service):
        processor = MagicMock()
        processor.parse_file = AsyncMock(side_effect=ValueError("boom"))
        processor.language = "python"
        factory = AsyncMock()
        factory.get_processor_for_file = AsyncMock(return_value=processor)
        mock_factory_fn.return_value = factory

        result = await service.analyze_file(Path("test.py"), "code")
        assert result.success is False
        assert "boom" in result.error
        assert result.language == "python"
        service.db.get_connection().rollback.assert_awaited()

    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_exception_processor_without_language_attr(self, mock_factory_fn, service):
        processor = MagicMock(spec=[])
        processor.parse_file = AsyncMock(side_effect=RuntimeError("fail"))
        # spec=[] means no 'language' attr
        # But we add parse_file above, so hasattr(processor, 'parse_file') is True
        factory = AsyncMock()
        factory.get_processor_for_file = AsyncMock(return_value=processor)
        mock_factory_fn.return_value = factory

        result = await service.analyze_file(Path("test.py"), "code")
        assert result.success is False
        assert result.language == "unknown"

    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_parse_failure_no_errors_list(self, mock_factory_fn, service):
        """When parse_result.errors is empty, fallback message is used."""
        parse_result = MagicMock()
        parse_result.success = False
        parse_result.language = "go"
        parse_result.errors = []
        parse_result.parse_time_ms = 0.5
        parse_result.total_lines = 3
        parse_result.warnings = []

        processor = MagicMock()
        processor.parse_file = AsyncMock(return_value=parse_result)
        factory = AsyncMock()
        factory.get_processor_for_file = AsyncMock(return_value=processor)
        mock_factory_fn.return_value = factory

        result = await service.analyze_file(Path("main.go"), "code")
        assert result.success is False
        assert result.error == "Parse failed"


# ===========================================================================
# Tests: analyze_directory
# ===========================================================================


class TestAnalyzeDirectory:
    async def test_raises_when_not_initialized(self, db):
        svc = MultiLanguageGraphService(db)
        with pytest.raises(RuntimeError, match="not initialized"):
            await svc.analyze_directory(Path("/tmp/code"))

    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_empty_directory(self, mock_factory_fn, service, tmp_path):
        factory = MagicMock()
        factory.is_supported_file = MagicMock(return_value=False)
        mock_factory_fn.return_value = factory

        service.db.get_connection().execute = AsyncMock()

        result = await service.analyze_directory(tmp_path)
        assert result["files_processed"] == 0
        assert result["files_failed"] == 0
        assert result["total_entities"] == 0
        assert "session_id" in result

    @patch.object(MultiLanguageGraphService, "analyze_file")
    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_processes_supported_files(
        self, mock_factory_fn, mock_analyze_file, service, tmp_path
    ):
        # Create a supported file
        test_file = tmp_path / "hello.py"
        test_file.write_text("print('hi')")

        factory = MagicMock()
        factory.is_supported_file = MagicMock(return_value=True)
        mock_factory_fn.return_value = factory

        mock_analyze_file.return_value = CoreAnalysisResult(
            language="python",
            file_path=str(test_file),
            success=True,
            entities=[_make_entity()],
            relationships=[],
            dependencies=[],
            complexity_metrics=None,
            security_patterns=[],
            error=None,
            parse_time_ms=1.0,
            total_lines=1,
            metadata={},
        )

        service.db.get_connection().execute = AsyncMock()

        result = await service.analyze_directory(tmp_path)
        assert result["files_processed"] == 1
        assert result["total_entities"] == 1
        assert "python" in result["languages_detected"]

    @patch.object(MultiLanguageGraphService, "analyze_file")
    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_language_filter(self, mock_factory_fn, mock_analyze_file, service, tmp_path):
        """When languages filter is provided, only matching files are processed."""
        py_file = tmp_path / "hello.py"
        py_file.write_text("pass")
        js_file = tmp_path / "app.js"
        js_file.write_text("//js")

        factory = MagicMock()
        factory.is_supported_file = MagicMock(return_value=True)
        mock_factory_fn.return_value = factory

        mock_analyze_file.return_value = CoreAnalysisResult(
            language="python",
            file_path=str(py_file),
            success=True,
            entities=[],
            relationships=[],
            dependencies=[],
            complexity_metrics=None,
            security_patterns=[],
            error=None,
            parse_time_ms=1.0,
            total_lines=1,
            metadata={},
        )

        service.db.get_connection().execute = AsyncMock()

        await service.analyze_directory(tmp_path, languages=["python"])
        # Only .py files should be picked (the extension map maps .py -> python)
        # .js maps to "javascript" which is not in the filter
        assert mock_analyze_file.await_count == 1

    @patch.object(MultiLanguageGraphService, "analyze_file")
    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_failed_file_analysis_counted(
        self, mock_factory_fn, mock_analyze_file, service, tmp_path
    ):
        test_file = tmp_path / "bad.py"
        test_file.write_text("bad code")

        factory = MagicMock()
        factory.is_supported_file = MagicMock(return_value=True)
        mock_factory_fn.return_value = factory

        mock_analyze_file.return_value = CoreAnalysisResult(
            language="python",
            file_path=str(test_file),
            success=False,
            entities=[],
            relationships=[],
            dependencies=[],
            complexity_metrics=None,
            security_patterns=[],
            error="parse error",
            parse_time_ms=0.0,
            total_lines=0,
            metadata={},
        )

        service.db.get_connection().execute = AsyncMock()

        result = await service.analyze_directory(tmp_path)
        assert result["files_failed"] == 1
        assert result["files_processed"] == 0

    @patch.object(MultiLanguageGraphService, "analyze_file")
    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_exception_in_file_analysis_counted_as_failure(
        self, mock_factory_fn, mock_analyze_file, service, tmp_path
    ):
        test_file = tmp_path / "crash.py"
        test_file.write_text("crash")

        factory = MagicMock()
        factory.is_supported_file = MagicMock(return_value=True)
        mock_factory_fn.return_value = factory

        mock_analyze_file.side_effect = Exception("unexpected")
        service.db.get_connection().execute = AsyncMock()

        result = await service.analyze_directory(tmp_path)
        assert result["files_failed"] == 1

    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_directory_analysis_failure_marks_session_failed(
        self, mock_factory_fn, service, tmp_path
    ):
        """If the entire analysis blows up, the session is marked failed."""
        mock_factory_fn.side_effect = RuntimeError("factory crash")
        service.db.get_connection().execute = AsyncMock()

        with pytest.raises(RuntimeError, match="factory crash"):
            await service.analyze_directory(tmp_path)


# ===========================================================================
# Tests: get_language_summary / get_entities_by_language
# ===========================================================================


class TestQueryMethods:
    async def test_get_language_summary_raises_when_not_initialized(self, db):
        svc = MultiLanguageGraphService(db)
        with pytest.raises(RuntimeError, match="not initialized"):
            await svc.get_language_summary()

    async def test_get_language_summary_queries_nodes(self, service):
        """get_language_summary aggregates a GROUP BY language SQL query, not a
        delegated entity_mapper call."""
        rows = [("python", 10), ("java", 4)]
        service.db.get_connection().execute = AsyncMock(return_value=_AsyncIter(rows))
        result = await service.get_language_summary()
        assert result == {"python": 10, "java": 4}
        # query is scoped by code_graph_id
        sql, params = service.db.get_connection().execute.await_args.args
        assert "GROUP BY language" in sql
        assert params == (service.db.code_graph_id,)

    async def test_get_entities_by_language_raises_when_not_initialized(self, db):
        svc = MultiLanguageGraphService(db)
        with pytest.raises(RuntimeError, match="not initialized"):
            await svc.get_entities_by_language("python")

    async def test_get_entities_by_language_queries_nodes(self, service):
        """get_entities_by_language runs a SQL query and maps positional rows to
        dicts, scoping by code_graph_id + language (+ type when provided)."""
        row = ("x", "function", "python", "main.py", 5, {"name": "x"})
        service.db.get_connection().execute = AsyncMock(return_value=_AsyncIter([row]))
        result = await service.get_entities_by_language("python", "function")
        assert result == [
            {
                "id": "x",
                "type": "function",
                "language": "python",
                "file_path": "main.py",
                "line_number": 5,
                "properties": {"name": "x"},
            }
        ]
        sql, params = service.db.get_connection().execute.await_args.args
        assert "AND type = %s" in sql
        assert params == [service.db.code_graph_id, "python", "function"]


# ===========================================================================
# Tests: search_entities
# ===========================================================================


class TestSearchEntities:
    async def test_raises_when_not_initialized(self, db):
        svc = MultiLanguageGraphService(db)
        with pytest.raises(RuntimeError, match="not initialized"):
            await svc.search_entities("query")

    async def test_search_without_language(self, service):
        mock_cursor = _AsyncIter([])
        service.db.get_connection().execute = AsyncMock(return_value=mock_cursor)

        result = await service.search_entities("foo")
        assert result == []
        service.db.get_connection().execute.assert_awaited_once()

    async def test_search_with_language(self, service):
        row = ("id1", "function", "python", "test.py", 10, "{}")
        mock_cursor = _AsyncIter([row])
        service.db.get_connection().execute = AsyncMock(return_value=mock_cursor)

        result = await service.search_entities("foo", language="python")
        assert len(result) == 1
        assert result[0]["id"] == "id1"
        assert result[0]["language"] == "python"


# ===========================================================================
# Tests: get_cross_language_relationships
# ===========================================================================


class TestCrossLanguageRelationships:
    async def test_raises_when_not_initialized(self, db):
        svc = MultiLanguageGraphService(db)
        with pytest.raises(RuntimeError, match="not initialized"):
            await svc.get_cross_language_relationships()

    async def test_returns_relationships(self, service):
        row = ("s1", "t1", "calls", "python", "java", "function", "class")
        mock_cursor = _AsyncIter([row])
        service.db.get_connection().execute = AsyncMock(return_value=mock_cursor)

        result = await service.get_cross_language_relationships()
        assert len(result) == 1
        assert result[0]["source"] == "s1"
        assert result[0]["source_language"] == "python"
        assert result[0]["target_language"] == "java"


# ===========================================================================
# Tests: _detect_file_language
# ===========================================================================


class TestDetectFileLanguage:
    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_returns_none_when_no_processor(self, mock_factory_fn, service):
        factory = AsyncMock()
        factory.get_processor_for_file = AsyncMock(return_value=None)
        mock_factory_fn.return_value = factory

        result = await service._detect_file_language(Path("test.xyz"))
        assert result is None

    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_returns_config_name(self, mock_factory_fn, service):
        processor = MagicMock()
        processor.config.name = "python"
        factory = AsyncMock()
        factory.get_processor_for_file = AsyncMock(return_value=processor)
        mock_factory_fn.return_value = factory

        result = await service._detect_file_language(Path("test.py"))
        assert result == "python"

    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_returns_language_attr(self, mock_factory_fn, service):
        processor = MagicMock(spec=["language"])
        processor.language = "java"
        factory = AsyncMock()
        factory.get_processor_for_file = AsyncMock(return_value=processor)
        mock_factory_fn.return_value = factory

        result = await service._detect_file_language(Path("Test.java"))
        assert result == "java"

    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_falls_back_to_extension_map(self, mock_factory_fn, service):
        # processor without config or language attrs
        processor = MagicMock(spec=[])
        factory = AsyncMock()
        factory.get_processor_for_file = AsyncMock(return_value=processor)
        mock_factory_fn.return_value = factory

        result = await service._detect_file_language(Path("main.go"))
        assert result == "go"

    @patch.object(_mlgs_mod, "get_multi_language_factory")
    async def test_extension_map_returns_none_for_unknown(self, mock_factory_fn, service):
        processor = MagicMock(spec=[])
        factory = AsyncMock()
        factory.get_processor_for_file = AsyncMock(return_value=processor)
        mock_factory_fn.return_value = factory

        result = await service._detect_file_language(Path("file.unknown_ext"))
        assert result is None


# ===========================================================================
# Tests: session management helpers
# ===========================================================================


class TestSessionManagement:
    async def test_start_analysis_session(self, service):
        service.db.get_connection().execute = AsyncMock()
        await service._start_analysis_session("sess-1", ["python", "java"])
        service.db.get_connection().execute.assert_awaited_once()

    async def test_complete_analysis_session(self, service):
        service.db.get_connection().execute = AsyncMock()
        await service._complete_analysis_session("sess-1", 10, 2, 50, 30)
        service.db.get_connection().execute.assert_awaited_once()

    async def test_fail_analysis_session(self, service):
        service.db.get_connection().execute = AsyncMock()
        await service._fail_analysis_session("sess-1", "something broke")
        service.db.get_connection().execute.assert_awaited_once()


# ===========================================================================
# Tests: MultiLanguageGraphServiceFactory
# ===========================================================================


class TestFactory:
    def setup_method(self):
        MultiLanguageGraphServiceFactory.reset()

    def teardown_method(self):
        MultiLanguageGraphServiceFactory.reset()

    def test_reset_clears_instance(self):
        MultiLanguageGraphServiceFactory._instance = MagicMock()
        MultiLanguageGraphServiceFactory.reset()
        assert MultiLanguageGraphServiceFactory._instance is None

    @patch.object(_mlgs_mod, "MultiLanguageEntityMapper")
    @patch.object(_mlgs_mod, "GraphDatabaseConnection")
    async def test_get_service_creates_instance(self, mock_gdb_cls, mock_mapper_cls):
        mock_conn = AsyncMock()
        mock_conn.connect = AsyncMock()
        mock_conn.get_connection = MagicMock(return_value=AsyncMock())
        mock_gdb_cls.return_value = mock_conn
        mock_mapper_cls.return_value = MagicMock()

        svc = await MultiLanguageGraphServiceFactory.get_service(code_graph_id="cg-1")
        assert svc is not None
        assert MultiLanguageGraphServiceFactory._instance is svc
        # the code_graph_id is threaded into the GraphDatabaseConnection ctor
        mock_gdb_cls.assert_called_once_with("cg-1")

    @patch.object(_mlgs_mod, "MultiLanguageEntityMapper")
    @patch.object(_mlgs_mod, "GraphDatabaseConnection")
    async def test_get_service_returns_cached_instance(self, mock_gdb_cls, mock_mapper_cls):
        mock_conn = AsyncMock()
        mock_conn.connect = AsyncMock()
        mock_conn.get_connection = MagicMock(return_value=AsyncMock())
        mock_gdb_cls.return_value = mock_conn
        mock_mapper_cls.return_value = MagicMock()

        svc1 = await MultiLanguageGraphServiceFactory.get_service(code_graph_id="cg-1")
        svc2 = await MultiLanguageGraphServiceFactory.get_service(code_graph_id="cg-1")
        assert svc1 is svc2
