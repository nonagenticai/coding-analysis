"""Unit tests for code_graph services: structure_storage and validation_utilities.

Covers:
- CodeStructureStorage (structure_storage.py)
- ValidationCache, ASTUtils, TextAnalysisUtils, GraphUtils,
  HashUtils, ValidationScoreCalculator (validation_utilities.py)
"""

from __future__ import annotations

import ast
import time
from unittest.mock import AsyncMock, MagicMock, patch

from services.code_graph.validation_utilities import (
    ASTUtils,
    GraphUtils,
    HashUtils,
    TextAnalysisUtils,
    ValidationCache,
    ValidationScoreCalculator,
)

# ============================================================================
# Helper: build a mock settings object for CodeStructureStorage
# ============================================================================


def _make_mock_settings():
    s = MagicMock()
    s.include_tests_in_graphs = False
    s.ignore_patterns = ["__pycache__", "*.pyc", ".git"]
    return s


def _make_storage(custom_settings=None):
    """Create a CodeStructureStorage with mocked dependencies."""
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.get_connection.return_value = mock_conn

    with patch(
        "services.code_graph.structure_storage.get_settings", return_value=_make_mock_settings()
    ):
        from services.code_graph.structure_storage import CodeStructureStorage

        storage = CodeStructureStorage(mock_db, custom_settings=custom_settings)
    return storage, mock_db, mock_conn


# ============================================================================
# Tests for ValidationCache
# ============================================================================


class TestValidationCache:
    def test_init_defaults(self):
        cache = ValidationCache()
        assert cache.max_size == 1000
        assert cache.default_ttl == 3600

    def test_init_custom(self):
        cache = ValidationCache(max_size=10, default_ttl=60)
        assert cache.max_size == 10
        assert cache.default_ttl == 60

    def test_get_miss(self):
        cache = ValidationCache()
        assert cache.get("nonexistent") is None

    def test_set_and_get(self):
        cache = ValidationCache()
        cache.set("key1", {"result": True})
        assert cache.get("key1") == {"result": True}

    def test_set_with_custom_ttl(self):
        cache = ValidationCache()
        cache.set("key1", "value", ttl=10)
        assert cache.get("key1") == "value"

    def test_expired_entry_returns_none(self):
        cache = ValidationCache()
        cache.set("key1", "value", ttl=0)
        # Force expiry by manipulating the entry
        cache._cache["key1"]["expires"] = time.time() - 1
        assert cache.get("key1") is None
        assert "key1" not in cache._cache

    def test_clear(self):
        cache = ValidationCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_eviction_on_max_size(self):
        cache = ValidationCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        # Manually set last_accessed so 'b' is oldest
        cache._cache["a"]["last_accessed"] = time.time() + 100
        cache._cache["b"]["last_accessed"] = time.time() - 100
        cache.set("c", 3)
        # 'b' should have been evicted (oldest accessed)
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3

    def test_evict_oldest_empty_cache(self):
        cache = ValidationCache()
        cache._evict_oldest()  # should not raise


# ============================================================================
# Tests for ASTUtils
# ============================================================================


class TestASTUtils:
    def test_safe_parse_valid(self):
        tree = ASTUtils.safe_parse("x = 1")
        assert tree is not None
        assert isinstance(tree, ast.AST)

    def test_safe_parse_empty(self):
        assert ASTUtils.safe_parse("") is None
        assert ASTUtils.safe_parse("   ") is None

    def test_safe_parse_syntax_error(self):
        assert ASTUtils.safe_parse("def (") is None

    def test_safe_parse_none(self):
        assert ASTUtils.safe_parse(None) is None

    def test_extract_functions(self):
        tree = ASTUtils.safe_parse("def foo(): pass\ndef bar(): pass")
        funcs = ASTUtils.extract_functions(tree)
        assert len(funcs) == 2
        assert funcs[0].name == "foo"
        assert funcs[1].name == "bar"

    def test_extract_functions_empty(self):
        assert ASTUtils.extract_functions(None) == []

    def test_extract_classes(self):
        tree = ASTUtils.safe_parse("class Foo: pass\nclass Bar: pass")
        classes = ASTUtils.extract_classes(tree)
        assert len(classes) == 2
        names = {c.name for c in classes}
        assert names == {"Foo", "Bar"}

    def test_extract_classes_empty(self):
        assert ASTUtils.extract_classes(None) == []

    def test_extract_imports(self):
        code = "import os\nimport sys\nfrom pathlib import Path\nfrom os.path import join"
        tree = ASTUtils.safe_parse(code)
        result = ASTUtils.extract_imports(tree)
        assert "os" in result["imports"]
        assert "sys" in result["imports"]
        assert "pathlib.Path" in result["from_imports"]
        assert "os.path.join" in result["from_imports"]

    def test_extract_imports_empty(self):
        result = ASTUtils.extract_imports(None)
        assert result == {"imports": [], "from_imports": []}

    def test_get_function_info_basic(self):
        code = '''
def greet(name: str) -> str:
    """Say hello."""
    return "Hello " + name
'''
        tree = ASTUtils.safe_parse(code)
        funcs = ASTUtils.extract_functions(tree)
        info = ASTUtils.get_function_info(funcs[0])
        assert info["name"] == "greet"
        assert info["docstring"] == "Say hello."
        assert info["return_annotation"] == "str"
        assert info["has_return_statements"] is True
        assert info["is_async"] is False
        assert len(info["parameters"]) == 1
        assert info["parameters"][0]["name"] == "name"

    def test_get_function_info_complexity(self):
        code = """
def complex_func(x):
    for i in range(x):
        if i > 5:
            try:
                pass
            except Exception:
                pass
    while True:
        break
"""
        tree = ASTUtils.safe_parse(code)
        funcs = ASTUtils.extract_functions(tree)
        info = ASTUtils.get_function_info(funcs[0])
        ci = info["complexity_indicators"]
        assert ci["loops"] >= 2  # for + while
        assert ci["conditions"] >= 1  # if
        assert ci["try_except"] >= 1  # try

    def test_get_function_info_recursive(self):
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
        tree = ASTUtils.safe_parse(code)
        funcs = ASTUtils.extract_functions(tree)
        info = ASTUtils.get_function_info(funcs[0])
        assert info["complexity_indicators"]["recursive_calls"] >= 1

    def test_get_function_info_decorators(self):
        code = """
@staticmethod
def my_static():
    pass
"""
        tree = ASTUtils.safe_parse(code)
        funcs = ASTUtils.extract_functions(tree)
        info = ASTUtils.get_function_info(funcs[0])
        assert "staticmethod" in info["decorators"]

    def test_get_class_info_basic(self):
        code = '''
class MyClass:
    """A test class."""
    def __init__(self):
        self.name = "test"
        self.value = 42

    def method(self):
        pass

    @property
    def prop(self):
        return self.name

    @staticmethod
    def static_m():
        pass

    @classmethod
    def cls_m(cls):
        pass
'''
        tree = ASTUtils.safe_parse(code)
        classes = ASTUtils.extract_classes(tree)
        info = ASTUtils.get_class_info(classes[0])
        assert info["name"] == "MyClass"
        assert info["docstring"] == "A test class."
        assert "name" in info["attributes"]
        assert "value" in info["attributes"]
        method_names = [m["name"] for m in info["methods"]]
        assert "__init__" in method_names
        assert "method" in method_names
        # Check property detection
        prop_method = next(m for m in info["methods"] if m["name"] == "prop")
        assert prop_method["is_property"] is True
        # Check static
        static_method = next(m for m in info["methods"] if m["name"] == "static_m")
        assert static_method["is_static"] is True
        # Check classmethod
        cls_method = next(m for m in info["methods"] if m["name"] == "cls_m")
        assert cls_method["is_class_method"] is True

    def test_get_class_info_with_bases(self):
        code = """
class Child(Parent):
    pass
"""
        tree = ASTUtils.safe_parse(code)
        classes = ASTUtils.extract_classes(tree)
        info = ASTUtils.get_class_info(classes[0])
        assert "Parent" in info["base_classes"]

    def test_get_class_info_no_docstring(self):
        code = """
class Empty:
    pass
"""
        tree = ASTUtils.safe_parse(code)
        classes = ASTUtils.extract_classes(tree)
        info = ASTUtils.get_class_info(classes[0])
        assert info["docstring"] is None

    def test_get_class_info_private_method(self):
        code = """
class Foo:
    def _private(self):
        pass
    def public(self):
        pass
"""
        tree = ASTUtils.safe_parse(code)
        classes = ASTUtils.extract_classes(tree)
        info = ASTUtils.get_class_info(classes[0])
        priv = next(m for m in info["methods"] if m["name"] == "_private")
        pub = next(m for m in info["methods"] if m["name"] == "public")
        assert priv["is_private"] is True
        assert pub["is_private"] is False


# ============================================================================
# Tests for TextAnalysisUtils
# ============================================================================


class TestTextAnalysisUtils:
    def test_extract_keywords_empty(self):
        assert TextAnalysisUtils.extract_keywords("") == set()
        assert TextAnalysisUtils.extract_keywords(None) == set()

    def test_extract_keywords_filters_stop_words(self):
        keywords = TextAnalysisUtils.extract_keywords("the quick brown fox is a test")
        assert "the" not in keywords
        assert "is" not in keywords
        assert "quick" in keywords
        assert "brown" in keywords
        assert "fox" in keywords
        assert "test" in keywords

    def test_extract_keywords_filters_short_words(self):
        keywords = TextAnalysisUtils.extract_keywords("I am ok no")
        # all words are <= 2 chars or stop words
        assert len(keywords) == 0

    def test_calculate_text_similarity_identical(self):
        score = TextAnalysisUtils.calculate_text_similarity(
            "python code analysis", "python code analysis"
        )
        assert score == 1.0

    def test_calculate_text_similarity_no_overlap(self):
        score = TextAnalysisUtils.calculate_text_similarity("apples oranges", "bananas grapes")
        assert score == 0.0

    def test_calculate_text_similarity_partial(self):
        score = TextAnalysisUtils.calculate_text_similarity(
            "python code analysis", "python code testing"
        )
        assert 0.0 < score < 1.0

    def test_calculate_text_similarity_empty(self):
        assert TextAnalysisUtils.calculate_text_similarity("", "test") == 0.0
        assert TextAnalysisUtils.calculate_text_similarity("test", "") == 0.0

    def test_detect_mentioned_entities(self):
        text = "This module uses CodeAnalyzer and DataParser for processing"
        entities = ["CodeAnalyzer", "DataParser", "NetworkClient"]
        mentioned = TextAnalysisUtils.detect_mentioned_entities(text, entities)
        assert "CodeAnalyzer" in mentioned
        assert "DataParser" in mentioned
        assert "NetworkClient" not in mentioned

    def test_detect_mentioned_entities_empty(self):
        assert TextAnalysisUtils.detect_mentioned_entities("", ["foo"]) == []
        assert TextAnalysisUtils.detect_mentioned_entities("foo", []) == []
        assert TextAnalysisUtils.detect_mentioned_entities(None, None) == []

    def test_extract_code_patterns_empty(self):
        result = TextAnalysisUtils.extract_code_patterns("")
        assert result["function_names"] == []
        assert result["class_names"] == []

    def test_extract_code_patterns_empty_none(self):
        result = TextAnalysisUtils.extract_code_patterns(None)
        assert result["function_names"] == []

    def test_extract_code_patterns_functions(self):
        text = "Call process_data() and validate_input() to handle the request"
        result = TextAnalysisUtils.extract_code_patterns(text)
        assert "process_data" in result["function_names"]
        assert "validate_input" in result["function_names"]

    def test_extract_code_patterns_classes(self):
        text = "Use MyClass and DataProcessor for the task"
        result = TextAnalysisUtils.extract_code_patterns(text)
        assert "MyClass" in result["class_names"]
        assert "DataProcessor" in result["class_names"]

    def test_extract_code_patterns_technologies(self):
        text = "Connect to the database and run a SQL query"
        result = TextAnalysisUtils.extract_code_patterns(text)
        assert "database" in result["technologies"]

    def test_extract_code_patterns_filters_common_words(self):
        text = "The API endpoint handles HTTP requests"
        result = TextAnalysisUtils.extract_code_patterns(text)
        # "The" should be filtered out
        assert "The" not in result["class_names"]


# ============================================================================
# Tests for GraphUtils
# ============================================================================


class TestGraphUtils:
    def test_create_dependency_graph_empty(self):
        graph = GraphUtils.create_dependency_graph({})
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_create_dependency_graph_basic(self):
        modules = {
            "main": "import utils\nfrom helpers import helper_func",
            "utils": "x = 1",
            "helpers": "def helper_func(): pass",
        }
        graph = GraphUtils.create_dependency_graph(modules)
        assert "main" in graph.nodes
        assert "utils" in graph.nodes
        assert "helpers" in graph.nodes
        assert graph.has_edge("main", "utils")
        assert graph.has_edge("main", "helpers")

    def test_create_dependency_graph_no_cross_deps(self):
        modules = {
            "a": "x = 1",
            "b": "y = 2",
        }
        graph = GraphUtils.create_dependency_graph(modules)
        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 0

    def test_create_dependency_graph_syntax_error(self):
        modules = {
            "good": "x = 1",
            "bad": "def (",  # syntax error
        }
        graph = GraphUtils.create_dependency_graph(modules)
        # "good" should still be in the graph; "bad" should be skipped
        assert "good" in graph.nodes

    def test_analyze_graph_metrics_empty(self):
        import networkx as nx

        graph = nx.DiGraph()
        metrics = GraphUtils.analyze_graph_metrics(graph)
        assert metrics["node_count"] == 0
        assert metrics["edge_count"] == 0
        assert metrics["density"] == 0.0
        assert metrics["is_connected"] is False

    def test_analyze_graph_metrics_basic(self):
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        metrics = GraphUtils.analyze_graph_metrics(graph)
        assert metrics["node_count"] == 3
        assert metrics["edge_count"] == 2
        assert metrics["is_connected"] is True
        assert metrics["has_cycles"] is False

    def test_analyze_graph_metrics_with_cycle(self):
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_edge("a", "b")
        graph.add_edge("b", "a")
        metrics = GraphUtils.analyze_graph_metrics(graph)
        assert metrics["has_cycles"] is True
        assert len(metrics["cycles"]) > 0

    def test_analyze_graph_metrics_centrality(self):
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_edge("a", "hub")
        graph.add_edge("b", "hub")
        graph.add_edge("c", "hub")
        metrics = GraphUtils.analyze_graph_metrics(graph)
        assert metrics["most_depended_on"] == "hub"


# ============================================================================
# Tests for HashUtils
# ============================================================================


class TestHashUtils:
    def test_hash_code(self):
        h = HashUtils.hash_code("print('hello')")
        assert isinstance(h, str)
        assert len(h) == 32  # md5 hex digest

    def test_hash_code_deterministic(self):
        assert HashUtils.hash_code("x") == HashUtils.hash_code("x")

    def test_hash_code_different_inputs(self):
        assert HashUtils.hash_code("a") != HashUtils.hash_code("b")

    def test_hash_dict(self):
        h = HashUtils.hash_dict({"key": "value", "num": 42})
        assert isinstance(h, str)
        assert len(h) == 32

    def test_hash_dict_deterministic(self):
        d = {"b": 2, "a": 1}
        assert HashUtils.hash_dict(d) == HashUtils.hash_dict(d)

    def test_hash_dict_order_independent(self):
        # sort_keys=True means order shouldn't matter
        assert HashUtils.hash_dict({"a": 1, "b": 2}) == HashUtils.hash_dict({"b": 2, "a": 1})

    def test_create_cache_key_strings(self):
        key = HashUtils.create_cache_key("prefix", "suffix")
        assert key == "prefix:suffix"

    def test_create_cache_key_with_dict(self):
        key = HashUtils.create_cache_key("prefix", {"a": 1})
        parts = key.split(":")
        assert parts[0] == "prefix"
        assert len(parts[1]) == 32  # md5 hash of the dict

    def test_create_cache_key_with_other_types(self):
        key = HashUtils.create_cache_key("prefix", 42)
        assert "prefix" in key


# ============================================================================
# Tests for ValidationScoreCalculator
# ============================================================================


class TestValidationScoreCalculator:
    def test_init(self):
        calc = ValidationScoreCalculator()
        assert "critical" in calc.severity_weights
        assert "public_api" in calc.context_weights

    def test_consistency_score_no_issues(self):
        calc = ValidationScoreCalculator()
        assert calc.calculate_consistency_score([]) == 1.0

    def test_consistency_score_single_critical(self):
        calc = ValidationScoreCalculator()
        issues = [{"severity": "critical"}]
        score = calc.calculate_consistency_score(issues)
        assert 0.0 < score < 1.0

    def test_consistency_score_multiple_issues(self):
        calc = ValidationScoreCalculator()
        issues = [
            {"severity": "critical"},
            {"severity": "high"},
            {"severity": "medium"},
            {"severity": "low"},
        ]
        score = calc.calculate_consistency_score(issues)
        assert 0.0 < score < 1.0

    def test_consistency_score_public_api_context(self):
        calc = ValidationScoreCalculator()
        issues = [{"severity": "medium"}]
        internal_score = calc.calculate_consistency_score(issues, context="internal")
        public_score = calc.calculate_consistency_score(issues, context="public_api")
        # public_api has higher weight, so score should be lower
        assert public_score < internal_score

    def test_consistency_score_capped_at_zero(self):
        calc = ValidationScoreCalculator()
        # Many critical issues should drive score to 0, not negative
        issues = [{"severity": "critical"} for _ in range(100)]
        score = calc.calculate_consistency_score(issues)
        assert score == 0.0

    def test_consistency_score_unknown_severity(self):
        calc = ValidationScoreCalculator()
        issues = [{"severity": "unknown_level"}]
        score = calc.calculate_consistency_score(issues)
        # Should use default weight of 0.5
        assert 0.0 < score < 1.0

    def test_consistency_score_missing_severity(self):
        calc = ValidationScoreCalculator()
        issues = [{}]  # no severity key
        score = calc.calculate_consistency_score(issues)
        assert 0.0 < score < 1.0

    def test_quality_score_empty_metrics(self):
        calc = ValidationScoreCalculator()
        score = calc.calculate_quality_score({})
        assert score == 0.5  # default when no scores

    def test_quality_score_consistency_only(self):
        calc = ValidationScoreCalculator()
        score = calc.calculate_quality_score({"consistency_score": 0.8})
        assert score == 0.8

    def test_quality_score_structural_high_complexity(self):
        calc = ValidationScoreCalculator()
        metrics = {"structural_metrics": {"complexity": 15, "documentation_coverage": 1.0}}
        score = calc.calculate_quality_score(metrics)
        assert score == 0.8  # 1.0 - 0.2 for high complexity

    def test_quality_score_structural_low_docs(self):
        calc = ValidationScoreCalculator()
        metrics = {"structural_metrics": {"complexity": 5, "documentation_coverage": 0.3}}
        score = calc.calculate_quality_score(metrics)
        assert score == 0.7  # 1.0 - 0.3 for low docs

    def test_quality_score_graph_with_cycles(self):
        calc = ValidationScoreCalculator()
        metrics = {"graph_metrics": {"has_cycles": True, "connectivity_score": 1.0}}
        score = calc.calculate_quality_score(metrics)
        assert score == 0.6  # 1.0 - 0.4 for cycles

    def test_quality_score_graph_poor_connectivity(self):
        calc = ValidationScoreCalculator()
        metrics = {"graph_metrics": {"has_cycles": False, "connectivity_score": 0.5}}
        score = calc.calculate_quality_score(metrics)
        assert score == 0.8  # 1.0 - 0.2 for poor connectivity

    def test_quality_score_combined(self):
        calc = ValidationScoreCalculator()
        metrics = {
            "consistency_score": 0.9,
            "structural_metrics": {"complexity": 5, "documentation_coverage": 1.0},
            "graph_metrics": {"has_cycles": False, "connectivity_score": 0.9},
        }
        score = calc.calculate_quality_score(metrics)
        # Average of 0.9, 1.0, 1.0
        expected = (0.9 + 1.0 + 1.0) / 3
        assert abs(score - expected) < 0.01


# ============================================================================
# Tests for CodeStructureStorage
# ============================================================================


class TestCodeStructureStorage:
    def test_init_default_settings(self):
        storage, _, _ = _make_storage()
        assert storage.analysis_settings["include_tests_in_graphs"] is False
        assert "__pycache__" in storage.analysis_settings["ignore_patterns"]
        assert storage.analysis_settings["include_patterns"] == []

    def test_init_custom_settings_include_tests(self):
        storage, _, _ = _make_storage(custom_settings={"include_tests": True})
        assert storage.analysis_settings["include_tests_in_graphs"] is True

    def test_init_custom_settings_exclude_patterns_list(self):
        storage, _, _ = _make_storage(custom_settings={"exclude_patterns": ["*.log", "build/"]})
        assert "*.log" in storage.analysis_settings["ignore_patterns"]
        assert "build/" in storage.analysis_settings["ignore_patterns"]

    def test_init_custom_settings_exclude_patterns_string(self):
        storage, _, _ = _make_storage(custom_settings={"exclude_patterns": "vendor/"})
        assert "vendor/" in storage.analysis_settings["ignore_patterns"]

    def test_init_custom_settings_include_patterns(self):
        storage, _, _ = _make_storage(custom_settings={"include_patterns": ["important_*.py"]})
        assert "important_*.py" in storage.analysis_settings["include_patterns"]

    def test_is_test_file(self):
        storage, _, _ = _make_storage()
        assert storage._is_test_file("test_foo.py") is True
        assert storage._is_test_file("foo_test.py") is True
        assert storage._is_test_file("conftest.py") is True
        assert storage._is_test_file("/project/tests/something.py") is True
        assert storage._is_test_file("/project/test/something.py") is True
        assert storage._is_test_file("/project/testing/something.py") is True
        assert storage._is_test_file("/project/src/utils.py") is False

    def test_is_test_related_pattern(self):
        storage, _, _ = _make_storage()
        assert storage._is_test_related_pattern("test*") is True
        assert storage._is_test_related_pattern("*test*") is True
        assert storage._is_test_related_pattern("test_*.py") is True
        assert storage._is_test_related_pattern("conftest.py") is True
        assert storage._is_test_related_pattern("*.pyc") is False

    def test_matches_pattern_filename(self):
        storage, _, _ = _make_storage()
        assert storage._matches_pattern("src/utils.pyc", "utils.pyc", "*.pyc") is True
        assert storage._matches_pattern("src/utils.py", "utils.py", "*.pyc") is False

    def test_matches_pattern_relative_path(self):
        storage, _, _ = _make_storage()
        assert storage._matches_pattern("__pycache__/foo.py", "foo.py", "__pycache__") is True

    def test_matches_pattern_path_parts(self):
        storage, _, _ = _make_storage()
        assert storage._matches_pattern("vendor/lib/foo.py", "foo.py", "vendor") is True

    def test_should_exclude_path_ignored_pattern(self):
        storage, _, _ = _make_storage()
        # __pycache__ is in ignore_patterns
        result = storage._should_exclude_path("/base/__pycache__/foo.py", "/base")
        assert result is True

    def test_should_exclude_path_normal_file(self):
        storage, _, _ = _make_storage()
        result = storage._should_exclude_path("/base/src/utils.py", "/base")
        assert result is False

    def test_should_exclude_path_test_file_tests_disabled(self):
        storage, _, _ = _make_storage()
        storage.analysis_settings["include_tests_in_graphs"] = False
        result = storage._should_exclude_path("/base/tests/test_foo.py", "/base")
        assert result is True

    def test_should_exclude_path_test_file_tests_enabled(self):
        storage, _, _ = _make_storage()
        storage.analysis_settings["include_tests_in_graphs"] = True
        result = storage._should_exclude_path("/base/tests/test_foo.py", "/base")
        assert result is False

    def test_should_exclude_path_include_pattern_overrides(self):
        storage, _, _ = _make_storage()
        storage.analysis_settings["include_patterns"] = ["important_test.py"]
        # Even though tests are disabled, include_pattern should override
        result = storage._should_exclude_path("/base/important_test.py", "/base")
        assert result is False

    async def test_get_stored_entities_all(self):
        storage, _, mock_conn = _make_storage()
        mock_cursor = AsyncMock()
        # JSONB properties come back already decoded to dicts.
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                ("id1", "Module", {"name": "foo"}),
                ("id2", "class", {"name": "Bar"}),
            ]
        )
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        entities = await storage.get_stored_entities()
        assert len(entities) == 2
        assert entities[0]["id"] == "id1"
        assert entities[0]["type"] == "Module"
        assert entities[0]["properties"]["name"] == "foo"

    async def test_get_stored_entities_by_type(self):
        storage, _, mock_conn = _make_storage()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                ("id1", "Module", {"name": "foo"}),
            ]
        )
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        entities = await storage.get_stored_entities(entity_type="Module")
        assert len(entities) == 1
        mock_conn.execute.assert_called_once()
        # postgres parameterised query scoped by code_graph_id + type
        sql, params = mock_conn.execute.call_args[0]
        assert "code_graph_id = %s AND type = %s" in sql
        assert params == (storage.db.code_graph_id, "Module")

    async def test_get_stored_entities_empty_properties(self):
        storage, _, mock_conn = _make_storage()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                ("id1", "Module", None),
            ]
        )
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        entities = await storage.get_stored_entities()
        assert entities[0]["properties"] == {}

    async def test_get_entity_relationships(self):
        storage, _, mock_conn = _make_storage()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                ("src1", "tgt1", "calls"),
                ("src2", "id1", "imports"),
            ]
        )
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        rels = await storage.get_entity_relationships("id1")
        assert len(rels) == 2
        assert rels[0]["source"] == "src1"
        assert rels[0]["target"] == "tgt1"
        assert rels[0]["relationship"] == "calls"

    async def test_store_code_structure_no_tables(self):
        storage, _, mock_conn = _make_storage()
        # Source probes to_regclass(...) IS NOT NULL and reads the single
        # (nodes_exists, edges_exists) tuple via fetchone(); missing -> (False, False).
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(False, False))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await storage.store_code_structure("/some/path")
        assert "error" in result

    async def test_store_code_structure_db_connection_failure(self):
        storage, _, mock_conn = _make_storage()
        mock_conn.execute = AsyncMock(side_effect=Exception("connection failed"))

        result = await storage.store_code_structure("/some/path")
        assert "error" in result

    async def test_store_code_structure_empty_dir(self):
        """Test store_code_structure with a directory containing no Python files."""
        storage, _, mock_conn = _make_storage()

        # Mock table check to return nodes and edges tables
        table_cursor = AsyncMock()
        table_cursor.fetchall = AsyncMock(return_value=[("nodes",), ("edges",)])

        # Mock commit and count queries
        count_cursor = AsyncMock()
        count_cursor.fetchone = AsyncMock(return_value=(0,))

        mock_conn.execute = AsyncMock(
            side_effect=[
                table_cursor,  # table check
                count_cursor,  # commit
                count_cursor,  # node count
                count_cursor,  # edge count (fetchone)
            ]
        )
        mock_conn.commit = AsyncMock()

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await storage.store_code_structure(tmpdir)
            assert result["files_analyzed"] == 0

    async def test_merge_custom_settings_none_include_tests(self):
        """Verify that include_tests=None doesn't override."""
        storage, _, _ = _make_storage(custom_settings={"include_tests": None})
        assert storage.analysis_settings["include_tests_in_graphs"] is False
