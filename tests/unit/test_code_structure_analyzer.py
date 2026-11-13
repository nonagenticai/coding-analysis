"""Unit tests for CodeStructureAnalyzer service."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def analyzer():
    from services.code_structure_analyzer import CodeStructureAnalyzer

    return CodeStructureAnalyzer(graph_manager=None)


@pytest.fixture
def analyzer_with_graph():
    """CodeStructureAnalyzer with a mock graph manager."""
    from services.code_structure_analyzer import CodeStructureAnalyzer

    graph_manager = MagicMock()
    return CodeStructureAnalyzer(graph_manager=graph_manager), graph_manager


# ---------------------------------------------------------------------------
# find_pattern_in_file (regex fallback path for non-python files)
# ---------------------------------------------------------------------------


def test_find_pattern_in_file_regex_match(tmp_path, analyzer):
    """find_pattern_in_file with regex pattern finds text matches."""
    py_file = tmp_path / "sample.txt"
    py_file.write_text("hello world\nfoo bar\nhello again\n")

    results = analyzer.find_pattern_in_file(str(py_file), r"hello", language="text")

    assert len(results) == 2
    for r in results:
        assert r["match_text"] == "hello"
        assert "line" in r
        assert "column" in r
        assert "context" in r
        assert r["method"] == "regex"


def test_find_pattern_in_file_regex_no_match(tmp_path, analyzer):
    """find_pattern_in_file returns empty list when pattern not found."""
    py_file = tmp_path / "sample.txt"
    py_file.write_text("nothing interesting here\n")

    results = analyzer.find_pattern_in_file(str(py_file), r"xyz_not_here", language="text")

    assert results == []


def test_find_pattern_in_file_nonexistent_returns_empty(analyzer):
    """find_pattern_in_file with nonexistent file returns empty list (no crash)."""
    results = analyzer.find_pattern_in_file("/does/not/exist/file.txt", r"pattern", language="text")

    assert results == []


def test_find_pattern_in_file_detects_language_from_extension(tmp_path, analyzer):
    """find_pattern_in_file detects language from file extension when not specified."""
    js_file = tmp_path / "script.js"
    js_file.write_text("const x = 1;\nconst y = 2;\n")

    # Should use regex for .js
    results = analyzer.find_pattern_in_file(str(js_file), r"const")

    assert isinstance(results, list)


def test_find_pattern_in_file_python_falls_back_to_regex_on_bad_xpath(tmp_path, analyzer):
    """Python file with an unparseable XPath pattern falls back to regex."""
    py_file = tmp_path / "code.py"
    py_file.write_text("x = 1\ny = 2\n")

    # If pyastgrep fails with this pattern, it should fall back to regex
    results = analyzer.find_pattern_in_file(str(py_file), r"x", language="python")

    assert isinstance(results, list)


def test_find_pattern_in_file_multiline_regex(tmp_path, analyzer):
    """find_pattern_in_file matches multiple occurrences per line."""
    txt_file = tmp_path / "multi.txt"
    txt_file.write_text("aaa bbb aaa\n")

    results = analyzer.find_pattern_in_file(str(txt_file), r"aaa", language="text")

    assert len(results) == 2  # two occurrences on the same line


# ---------------------------------------------------------------------------
# extract_function_call_relationships
# ---------------------------------------------------------------------------


def test_extract_function_call_relationships_basic(tmp_path, analyzer):
    """extract_function_call_relationships finds caller->callee from real file."""
    py_file = tmp_path / "calls.py"
    py_file.write_text("def foo():\n    bar()\n\ndef bar():\n    pass\n")

    rels = analyzer.extract_function_call_relationships(str(py_file))

    assert isinstance(rels, list)
    # Should find foo -> bar
    sources = [r["source"] for r in rels]
    targets = [r["target"] for r in rels]
    assert "foo" in sources
    assert "bar" in targets


def test_extract_function_call_relationships_class_methods(tmp_path, analyzer):
    """extract_function_call_relationships handles class method calls."""
    py_file = tmp_path / "methods.py"
    py_file.write_text(
        "class MyClass:\n"
        "    def method_a(self):\n"
        "        self.method_b()\n"
        "\n"
        "    def method_b(self):\n"
        "        pass\n"
    )

    rels = analyzer.extract_function_call_relationships(str(py_file))

    assert isinstance(rels, list)
    assert len(rels) >= 1
    for r in rels:
        assert "source" in r
        assert "target" in r
        assert "line_number" in r
        assert "file_path" in r
        assert r["file_path"] == str(py_file)


def test_extract_function_call_relationships_empty_file(tmp_path, analyzer):
    """extract_function_call_relationships returns empty list for empty file."""
    py_file = tmp_path / "empty.py"
    py_file.write_text("")

    rels = analyzer.extract_function_call_relationships(str(py_file))

    assert rels == []


def test_extract_function_call_relationships_nonexistent_file(analyzer):
    """extract_function_call_relationships returns empty list for missing file."""
    rels = analyzer.extract_function_call_relationships("/no/such/file.py")

    assert rels == []


def test_extract_function_call_relationships_syntax_error_file(tmp_path, analyzer):
    """extract_function_call_relationships returns empty list for syntax-broken file."""
    py_file = tmp_path / "broken.py"
    py_file.write_text("def bad(\n    # unclosed\n")

    rels = analyzer.extract_function_call_relationships(str(py_file))

    assert rels == []


# ---------------------------------------------------------------------------
# extract_call_relationships_from_string
# ---------------------------------------------------------------------------


def test_extract_call_relationships_from_string_basic(analyzer):
    """extract_call_relationships_from_string extracts caller->callee pairs."""
    code = "def outer():\n    inner()\n\ndef inner():\n    pass\n"

    rels = analyzer.extract_call_relationships_from_string(code)

    assert isinstance(rels, list)
    assert len(rels) >= 1
    sources = [r["source"] for r in rels]
    targets = [r["target"] for r in rels]
    assert "outer" in sources
    assert "inner" in targets


def test_extract_call_relationships_from_string_chained_calls(analyzer):
    """extract_call_relationships_from_string handles chained method calls."""
    code = "def process():\n    result = obj.method_a()\n    obj.method_b(result)\n"

    rels = analyzer.extract_call_relationships_from_string(code)

    assert isinstance(rels, list)
    targets = [r["target"] for r in rels]
    assert any("method_a" in t or "method_b" in t for t in targets)


def test_extract_call_relationships_from_string_empty(analyzer):
    """extract_call_relationships_from_string returns empty list for empty code."""
    rels = analyzer.extract_call_relationships_from_string("")

    assert rels == []


def test_extract_call_relationships_from_string_syntax_error(analyzer):
    """extract_call_relationships_from_string returns empty list for syntax error."""
    rels = analyzer.extract_call_relationships_from_string("def bad(\n    # unclosed\n")

    assert rels == []


def test_extract_call_relationships_from_string_file_path_hint(analyzer):
    """extract_call_relationships_from_string uses provided file_path_hint."""
    code = "def a():\n    b()\n\ndef b():\n    pass\n"

    rels = analyzer.extract_call_relationships_from_string(code, file_path_hint="custom.py")

    for r in rels:
        assert r["file_path"] == "custom.py"


def test_extract_call_relationships_from_string_no_functions(analyzer):
    """Code with no function definitions produces no relationships."""
    code = "x = 1\ny = x + 2\n"

    rels = analyzer.extract_call_relationships_from_string(code)

    assert rels == []


# ---------------------------------------------------------------------------
# calculate_function_metrics (no graph_manager)
# ---------------------------------------------------------------------------


def test_calculate_function_metrics_no_graph(analyzer):
    """calculate_function_metrics returns error dict when no graph_manager."""
    result = analyzer.calculate_function_metrics("my_function")

    assert "error" in result
    assert result["outgoing_calls"] == 0
    assert result["incoming_calls"] == 0
    assert result["unique_callees"] == 0
    assert result["unique_callers"] == 0


def test_calculate_function_metrics_with_graph_not_found(analyzer_with_graph):
    """calculate_function_metrics returns error when function not in graph."""
    analyzer, graph_manager = analyzer_with_graph
    graph_manager.get_nodes_by_property.return_value = []

    result = analyzer.calculate_function_metrics("nonexistent_func")

    assert "error" in result
    assert "not found" in result["error"]


def test_calculate_function_metrics_with_graph_success(analyzer_with_graph):
    """calculate_function_metrics returns correct metrics when function found."""
    analyzer, graph_manager = analyzer_with_graph

    function_node = {"id": "func-001", "properties": {"name": "my_func", "complexity": 3}}
    outgoing_edges = [
        {"source": "func-001", "target": "func-002", "relationship": "calls", "properties": {}},
        {"source": "func-001", "target": "func-003", "relationship": "calls", "properties": {}},
    ]
    incoming_edges = [
        {"source": "func-004", "target": "func-001", "relationship": "calls", "properties": {}},
    ]

    graph_manager.get_nodes_by_property.return_value = [function_node]
    graph_manager.get_edges.side_effect = lambda **kwargs: (
        outgoing_edges if kwargs.get("source_id") == "func-001" else incoming_edges
    )

    result = analyzer.calculate_function_metrics("my_func")

    assert result["function_name"] == "my_func"
    assert result["outgoing_calls"] == 2
    assert result["incoming_calls"] == 1
    assert result["unique_callees"] == 2
    assert result["unique_callers"] == 1
    assert result["cyclomatic_complexity"] == 3


# ---------------------------------------------------------------------------
# find_function_calls and find_class_references (no graph)
# ---------------------------------------------------------------------------


def test_find_function_calls_no_graph(analyzer):
    """find_function_calls returns empty list when no graph manager."""
    result = analyzer.find_function_calls("some_function")

    assert result == []


def test_find_class_references_no_graph(analyzer):
    """find_class_references returns empty list when no graph manager."""
    result = analyzer.find_class_references("SomeClass")

    assert result == []


# ---------------------------------------------------------------------------
# analyze_dependency_graph (no graph)
# ---------------------------------------------------------------------------


def test_analyze_dependency_graph_no_graph(analyzer):
    """analyze_dependency_graph returns error structure when no graph_manager."""
    result = analyzer.analyze_dependency_graph("some-node-id")

    assert "error" in result
    assert result["nodes"] == []
    assert result["edges"] == []


def test_analyze_dependency_graph_with_graph(analyzer_with_graph):
    """analyze_dependency_graph traverses graph nodes."""
    analyzer, graph_manager = analyzer_with_graph

    node = {
        "id": "node-001",
        "type": "function",
        "properties": {"name": "my_func"},
    }
    graph_manager.get_node.return_value = node
    graph_manager.get_edges.return_value = []  # no outgoing edges

    result = analyzer.analyze_dependency_graph("node-001")

    assert isinstance(result, dict)
    assert len(result["nodes"]) >= 1
    assert result["nodes"][0]["id"] == "node-001"
