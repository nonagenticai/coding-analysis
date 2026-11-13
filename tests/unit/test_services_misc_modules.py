"""Unit tests for code_structure_analyzer, bug_analysis,
similarity_detector, and semantic_analysis_service -- targeting uncovered lines.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import services.code_structure_analyzer as _csa_mod
import services.semantic_analysis_service as _sas_mod
from models.bug_models import CodeFix, ErrorType, ParsedError
from models.container_models import ContainerInfo, ContainerSpec, ContainerStatus, GitHubRepoRequest
from services.bug_analysis import BugAnalyzer
from services.code_structure_analyzer import (
    CodeStructureAnalyzer,
    FunctionCallVisitor,
)
from services.semantic_analysis_service import (
    SemanticAnalysisService,
    _call_name,
    _decorator_name,
    _name_of,
    _parse_generic,
    _parse_python,
    get_semantic_analysis_service,
)
from services.similarity_detector import (
    SimilarityAnalysisResult,
    SimilarityDetector,
    SimilarityMatch,
    SimilarityScope,
    SimilarityType,
    _ast_node_types,
    _detect_language,
    _file_hash,
    _read_file,
    _sequence_ratio,
    _tokenize,
    get_similarity_detector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_error(
    error_type: ErrorType = ErrorType.SYNTAX_ERROR,
    message: str = "test error",
    file_path: str = "test.py",
    confidence: float = 0.9,
    line_number: int = 1,
) -> ParsedError:
    return ParsedError(
        error_type=error_type,
        message=message,
        file_path=file_path,
        line_number=line_number,
        function_name=None,
        stack_trace=[],
        confidence=confidence,
    )


def _make_fix(
    file_path: str = "test.py",
    confidence: float = 0.8,
    fixed_code: str = "y = 1",
    explanation: str = "fix issue",
) -> CodeFix:
    return CodeFix(
        file_path=file_path,
        line_number=1,
        original_code="x",
        fixed_code=fixed_code,
        explanation=explanation,
        confidence=confidence,
    )


def _make_container_info(
    name: str = "test-container",
    codebase_path: str = "/tmp/test-codebase",
    status: ContainerStatus = ContainerStatus.RUNNING,
) -> ContainerInfo:
    return ContainerInfo(
        container_name=name,
        api_endpoint="http://localhost:8080",
        codebase_path=codebase_path,
        repo_url="https://github.com/test/repo",
        branch="main",
        status=status,
    )


# ============================================================================
# 1. CodeStructureAnalyzer  (uncovered paths)
# ============================================================================


class TestFunctionCallVisitor:
    """Cover FunctionCallVisitor AST traversal branches."""

    def test_visit_class_method_call(self):
        code = "class A:\n    def m(self):\n        self.helper()\n"
        tree = ast.parse(code)
        v = FunctionCallVisitor()
        v.visit(tree)
        assert len(v.calls) >= 1
        callee_names = [c[1] for c in v.calls]
        assert any("helper" in n for n in callee_names)

    def test_visit_nested_functions(self):
        code = "def outer():\n    def inner():\n        print('hi')\n    inner()\n"
        tree = ast.parse(code)
        v = FunctionCallVisitor()
        v.visit(tree)
        sources = [c[0] for c in v.calls if c[0] is not None]
        assert any("outer" in s[0] for s in sources)

    def test_visit_call_outside_function(self):
        code = "print('hello')\n"
        tree = ast.parse(code)
        v = FunctionCallVisitor()
        v.visit(tree)
        # Calls outside functions are ignored
        assert len(v.calls) == 0

    def test_get_attribute_path_complex(self):
        code = "def f():\n    a.b.c()\n"
        tree = ast.parse(code)
        v = FunctionCallVisitor()
        v.visit(tree)
        callee_names = [c[1] for c in v.calls]
        assert any("a.b.c" in n for n in callee_names)

    def test_call_graph_structure(self):
        code = "def foo():\n    bar()\n    bar()\n    baz()\n"
        tree = ast.parse(code)
        v = FunctionCallVisitor()
        v.visit(tree)
        assert ("foo", "function") in v.call_graph
        assert "bar" in v.call_graph[("foo", "function")]
        assert len(v.call_graph[("foo", "function")]["bar"]) == 2


class TestCodeStructureAnalyzerPyastgrep:
    """Cover pyastgrep code paths via mocking."""

    def test_find_pattern_python_pyastgrep_success(self, tmp_path):
        """When pyastgrep returns a match with position, we get pyastgrep result."""
        py_file = tmp_path / "code.py"
        py_file.write_text("x = 1\ny = 2\nz = 3\n")

        match_obj = MagicMock()
        match_obj.position = MagicMock()
        match_obj.position.lineno = 1
        match_obj.position.col_offset = 0
        match_obj.path = str(py_file)
        match_obj.file_lines = ["x = 1", "y = 2", "z = 3"]
        match_obj.ast_node = ast.parse("x = 1").body[0]

        analyzer = CodeStructureAnalyzer()
        with patch.object(
            _csa_mod,
            "search_python_files",
            return_value=[match_obj],
        ):
            results = analyzer.find_pattern_in_file(str(py_file), ".//Name", language="python")
        assert len(results) == 1
        assert results[0]["method"] == "pyastgrep"
        assert results[0]["line"] == 1

    def test_find_pattern_python_pyastgrep_missing_path(self, tmp_path):
        """MissingPath from pyastgrep is logged, not returned."""
        from pyastgrep.search import MissingPath

        py_file = tmp_path / "code.py"
        py_file.write_text("x = 1\n")

        missing = MissingPath(Path("/not/there"))

        analyzer = CodeStructureAnalyzer()
        with patch.object(
            _csa_mod,
            "search_python_files",
            return_value=[missing],
        ):
            results = analyzer.find_pattern_in_file(str(py_file), ".//Name", language="python")
        assert results == []

    def test_find_pattern_python_pyastgrep_non_element(self, tmp_path):
        """NonElementReturned from pyastgrep is logged as debug."""
        from pyastgrep.search import NonElementReturned

        py_file = tmp_path / "code.py"
        py_file.write_text("x = 1\n")

        obj = NonElementReturned(MagicMock(), MagicMock())

        analyzer = CodeStructureAnalyzer()
        with patch.object(
            _csa_mod,
            "search_python_files",
            return_value=[obj],
        ):
            results = analyzer.find_pattern_in_file(str(py_file), ".//Name", language="python")
        assert results == []

    def test_find_pattern_python_pyastgrep_file_finished(self, tmp_path):
        """FileFinished from pyastgrep is silently ignored."""
        from pyastgrep.search import FileFinished

        py_file = tmp_path / "code.py"
        py_file.write_text("x = 1\n")

        obj = FileFinished(Path(py_file))

        analyzer = CodeStructureAnalyzer()
        with patch.object(
            _csa_mod,
            "search_python_files",
            return_value=[obj],
        ):
            results = analyzer.find_pattern_in_file(str(py_file), ".//Name", language="python")
        assert results == []

    def test_find_pattern_python_pyastgrep_unexpected_type(self, tmp_path):
        """Unexpected object type from pyastgrep is logged as warning."""
        py_file = tmp_path / "code.py"
        py_file.write_text("x = 1\n")

        analyzer = CodeStructureAnalyzer()
        with patch.object(
            _csa_mod,
            "search_python_files",
            return_value=["unexpected_string"],
        ):
            results = analyzer.find_pattern_in_file(str(py_file), ".//Name", language="python")
        assert results == []

    def test_find_pattern_python_pyastgrep_os_error_fallback(self, tmp_path):
        """OSError during pyastgrep falls back to regex."""
        py_file = tmp_path / "code.py"
        py_file.write_text("hello world\n")

        analyzer = CodeStructureAnalyzer()
        with patch.object(
            _csa_mod,
            "search_python_files",
            side_effect=OSError("disk error"),
        ):
            results = analyzer.find_pattern_in_file(str(py_file), "hello", language="python")
        assert len(results) == 1
        assert results[0]["method"] == "regex"

    def test_extract_match_text_unparse_failure(self):
        """When ast.unparse fails, fallback to line content extraction."""
        analyzer = CodeStructureAnalyzer()
        match_obj = MagicMock()
        match_obj.position = MagicMock()
        match_obj.position.col_offset = 0
        match_obj.file_lines = ["x = 1"]

        with patch("ast.unparse", side_effect=TypeError("cannot unparse")):
            result = analyzer._extract_match_text(match_obj, 1)
        assert result == "x = 1"

    def test_fallback_match_text_multiline(self):
        """Fallback extraction for multi-line AST node appends '...'."""
        analyzer = CodeStructureAnalyzer()
        match_obj = MagicMock()
        match_obj.position = MagicMock()
        match_obj.position.col_offset = 0
        match_obj.file_lines = ["def foo():", "    pass"]
        match_obj.ast_node = MagicMock()
        match_obj.ast_node.end_lineno = 2
        match_obj.ast_node.end_col_offset = 8

        result = analyzer._fallback_match_text_extraction(match_obj, 1)
        assert result.endswith("...")

    def test_fallback_match_text_single_line(self):
        """Fallback extraction for single-line AST node returns slice."""
        analyzer = CodeStructureAnalyzer()
        match_obj = MagicMock()
        match_obj.position = MagicMock()
        match_obj.position.col_offset = 4
        match_obj.file_lines = ["    x = 1"]
        match_obj.ast_node = MagicMock()
        match_obj.ast_node.end_lineno = 1
        match_obj.ast_node.end_col_offset = 9

        result = analyzer._fallback_match_text_extraction(match_obj, 1)
        assert result == "x = 1"


class TestCodeStructureAnalyzerGraphPaths:
    """Cover graph_manager-dependent code paths."""

    def test_analyze_dependency_graph_with_edges_and_traversal(self):
        graph_manager = MagicMock()
        node_a = {
            "id": "a",
            "type": "module",
            "properties": {"name": "mod_a"},
        }
        node_b = {
            "id": "b",
            "type": "function",
            "properties": {"name": "func_b"},
        }
        graph_manager.get_node.side_effect = lambda nid: (
            node_a if nid == "a" else node_b if nid == "b" else None
        )
        graph_manager.get_edges.side_effect = lambda **kw: (
            [
                {
                    "source": "a",
                    "target": "b",
                    "relationship": "calls",
                    "properties": {},
                }
            ]
            if kw.get("source_id") == "a"
            else []
        )

        analyzer = CodeStructureAnalyzer(graph_manager=graph_manager)
        result = analyzer.analyze_dependency_graph("a", max_depth=2)

        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1

    def test_analyze_dependency_graph_skips_defined_in_edges(self):
        graph_manager = MagicMock()
        node = {"id": "a", "type": "module", "properties": {"name": "mod"}}
        graph_manager.get_node.return_value = node
        graph_manager.get_edges.return_value = [
            {
                "source": "a",
                "target": "b",
                "relationship": "defined_in",
                "properties": {},
            },
            {
                "source": "a",
                "target": "c",
                "relationship": "contains",
                "properties": {},
            },
        ]

        analyzer = CodeStructureAnalyzer(graph_manager=graph_manager)
        result = analyzer.analyze_dependency_graph("a", max_depth=1)

        assert len(result["edges"]) == 0

    def test_analyze_dependency_graph_max_depth_stops(self):
        graph_manager = MagicMock()
        node = {"id": "a", "type": "func", "properties": {"name": "f"}}
        graph_manager.get_node.return_value = node
        graph_manager.get_edges.return_value = [
            {
                "source": "a",
                "target": "a2",
                "relationship": "calls",
                "properties": {},
            }
        ]

        analyzer = CodeStructureAnalyzer(graph_manager=graph_manager)
        result = analyzer.analyze_dependency_graph("a", max_depth=0)

        # At depth 0, the node is added, edges are listed, but target is NOT
        # traversed because depth+1 > max_depth.
        assert len(result["nodes"]) == 1
        assert len(result["edges"]) == 1  # edge recorded but target not visited

    def test_analyze_dependency_graph_none_node(self):
        graph_manager = MagicMock()
        graph_manager.get_node.return_value = None

        analyzer = CodeStructureAnalyzer(graph_manager=graph_manager)
        result = analyzer.analyze_dependency_graph("missing")

        assert result["nodes"] == []

    def test_analyze_call_graph_empty_function_name(self):
        analyzer = CodeStructureAnalyzer()
        result = analyzer.analyze_call_graph("")
        assert "error" in result

    def test_analyze_call_graph_invalid_direction(self):
        analyzer = CodeStructureAnalyzer()
        result = analyzer.analyze_call_graph("foo", direction="sideways")
        assert "error" in result

    def test_analyze_call_graph_no_graph_manager(self):
        analyzer = CodeStructureAnalyzer()
        result = analyzer.analyze_call_graph("foo")
        assert "error" in result

    def test_analyze_call_graph_function_not_found(self):
        gm = MagicMock()
        gm.get_nodes_by_property.return_value = []
        analyzer = CodeStructureAnalyzer(graph_manager=gm)
        result = analyzer.analyze_call_graph("missing_func")
        assert "not found" in result["error"]

    def test_analyze_call_graph_outgoing(self):
        gm = MagicMock()
        func_node = {"id": "f1", "properties": {"name": "foo"}, "type": "function"}
        callee_node = {"id": "f2", "properties": {"name": "bar"}, "type": "function"}
        gm.get_nodes_by_property.return_value = [func_node]
        gm.get_node.side_effect = lambda nid: func_node if nid == "f1" else callee_node

        gm.get_edges.side_effect = lambda **kw: (
            [{"source": "f1", "target": "f2", "relationship": "calls", "properties": {}}]
            if kw.get("source_id") == "f1" and kw.get("relationship") == "calls"
            else []
        )

        analyzer = CodeStructureAnalyzer(graph_manager=gm)
        result = analyzer.analyze_call_graph("foo", max_depth=1, direction="outgoing")

        assert result["root"] == "foo"
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        assert result["direction"] == "outgoing"

    def test_analyze_call_graph_incoming(self):
        gm = MagicMock()
        func_node = {"id": "f1", "properties": {"name": "foo"}, "type": "function"}
        caller_node = {"id": "f2", "properties": {"name": "bar"}, "type": "function"}
        gm.get_nodes_by_property.return_value = [func_node]
        gm.get_node.side_effect = lambda nid: func_node if nid == "f1" else caller_node

        gm.get_edges.side_effect = lambda **kw: (
            [{"source": "f2", "target": "f1", "relationship": "calls", "properties": {}}]
            if kw.get("target_id") == "f1" and kw.get("relationship") == "calls"
            else []
        )

        analyzer = CodeStructureAnalyzer(graph_manager=gm)
        result = analyzer.analyze_call_graph("foo", max_depth=1, direction="incoming")

        assert result["root"] == "foo"
        assert len(result["nodes"]) == 2
        assert result["direction"] == "incoming"

    def test_analyze_call_graph_max_depth_reached(self):
        gm = MagicMock()
        func_node = {"id": "f1", "properties": {"name": "foo"}, "type": "function"}
        gm.get_nodes_by_property.return_value = [func_node]
        gm.get_node.return_value = func_node
        gm.get_edges.return_value = [
            {"source": "f1", "target": "f2", "relationship": "calls", "properties": {}}
        ]

        analyzer = CodeStructureAnalyzer(graph_manager=gm)
        result = analyzer.analyze_call_graph("foo", max_depth=0, direction="outgoing")

        # At max_depth=0, node added but no edges followed
        assert len(result["nodes"]) == 1

    def test_extract_function_calls_delegates_to_string(self):
        analyzer = CodeStructureAnalyzer()
        code = "def a():\n    b()\n"
        result = analyzer.extract_function_calls(code, "test.py")
        assert isinstance(result, list)

    def test_extract_call_relationships_non_printable_chars(self, tmp_path):
        analyzer = CodeStructureAnalyzer()
        py_file = tmp_path / "bin.py"
        py_file.write_bytes(b"\x01\x02\x03def foo(): pass\n")
        rels = analyzer.extract_function_call_relationships(str(py_file))
        assert rels == []

    def test_extract_call_relationships_from_string_non_printable(self):
        analyzer = CodeStructureAnalyzer()
        code = "\x01\x02def foo(): pass\n"
        rels = analyzer.extract_call_relationships_from_string(code)
        assert rels == []

    def test_calculate_function_metrics_exception(self):
        gm = MagicMock()
        func_node = {"id": "f1", "properties": {"name": "foo"}}
        gm.get_nodes_by_property.return_value = [func_node]
        gm.get_edges.side_effect = KeyError("bad key")

        analyzer = CodeStructureAnalyzer(graph_manager=gm)
        result = analyzer.calculate_function_metrics("foo")
        assert "error" in result
        assert result["outgoing_calls"] == 0

    def test_find_import_usage_no_graph(self):
        analyzer = CodeStructureAnalyzer()
        result = analyzer.find_import_usage("os")
        assert result == []


# ============================================================================
# 2. BugAnalyzer  (uncovered paths)
# ============================================================================


class TestBugAnalyzerUncovered:
    """Cover branches not hit by existing test_bug_analysis.py."""

    @pytest.fixture
    def analyzer(self):
        return BugAnalyzer()

    def test_analyze_root_causes_many_error_types(self, analyzer):
        """More than 5 error types triggers 'systemic issues' assessment."""
        types = [
            ErrorType.SYNTAX_ERROR,
            ErrorType.IMPORT_ERROR,
            ErrorType.NAME_ERROR,
            ErrorType.TYPE_ERROR,
            ErrorType.ATTRIBUTE_ERROR,
            ErrorType.INDEX_ERROR,
        ]
        errors = [_make_error(error_type=t) for t in types]
        result = analyzer.analyze_root_causes(errors, [])
        assert "systemic" in result.lower() or "multiple" in result.lower()

    def test_analyze_root_causes_single_error_type(self, analyzer):
        """Single error type triggers 'systematically addressed' assessment."""
        errors = [_make_error(ErrorType.SYNTAX_ERROR) for _ in range(3)]
        result = analyzer.analyze_root_causes(errors, [])
        assert "systematically" in result.lower()

    def test_analyze_root_causes_mixed_types(self, analyzer):
        """2-5 error types triggers 'isolated issues' assessment."""
        errors = [
            _make_error(ErrorType.SYNTAX_ERROR),
            _make_error(ErrorType.IMPORT_ERROR),
        ]
        result = analyzer.analyze_root_causes(errors, [])
        assert "isolated" in result.lower() or "individually" in result.lower()

    def test_analyze_root_causes_invalid_type_raises(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.analyze_root_causes("not a list", [])

    def test_generate_fix_summary_invalid_type_raises(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.generate_fix_summary("not a list")

    def test_generate_fix_summary_medium_confidence(self, analyzer):
        """Medium confidence fixes appear in summary."""
        fixes = [_make_fix(confidence=0.65, explanation="medium fix")]
        result = analyzer.generate_fix_summary(fixes)
        assert "Medium-Confidence" in result or "medium fix" in result

    def test_generate_fix_summary_low_confidence(self, analyzer):
        """Low confidence fixes section appears."""
        fixes = [_make_fix(confidence=0.3, explanation="low fix")]
        result = analyzer.generate_fix_summary(fixes)
        assert "Low-Confidence" in result or "manual review" in result.lower()

    def test_generate_fix_summary_attribute_error_on_resolve(self, analyzer):
        """Malformed resolve object causes re-raise from outer handler."""
        bad = MagicMock(spec=[])  # no .file_path, .confidence, etc.
        with pytest.raises(AttributeError):
            analyzer.generate_fix_summary([bad])

    def test_estimate_effort_invalid_type_raises(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.estimate_effort("not a list")

    def test_estimate_effort_categorization(self, analyzer):
        """Verify time calculation categories."""
        # Simple: confidence > 0.8 AND code < 100 chars
        simple = _make_fix(confidence=0.9, fixed_code="x = 1")
        # Moderate: confidence > 0.5
        moderate = _make_fix(confidence=0.6, fixed_code="x" * 200)
        # Complex: confidence < 0.5
        cpx = _make_fix(confidence=0.2, fixed_code="x")

        result = analyzer.estimate_effort([simple, moderate, cpx])
        assert "Estimated Total Time" in result

    def test_estimate_effort_attribute_error_defaults_complex(self, analyzer):
        """Malformed resolve defaults to complex category (AttributeError caught)."""
        bad = MagicMock(spec=[])
        result = analyzer.estimate_effort([bad])
        assert "Effort Estimation" in result
        # The bad resolve should be counted as complex
        assert "Complex" in result or "complex" in result.lower()

    def test_assess_risks_invalid_type_raises(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.assess_risks("not a list")

    def test_assess_risks_security_keyword(self, analyzer):
        """Security-related explanation increases risk score."""
        fix = _make_fix(
            confidence=0.3,
            explanation="Fix security authentication bypass",
            fixed_code="x" * 300,
        )
        result = analyzer.assess_risks([fix])
        assert "High Risk" in result or "High-Risk" in result

    def test_assess_risks_attribute_error_defaults_medium(self, analyzer):
        """Malformed resolve gets categorized as medium, then output fails on .file_path."""
        bad = MagicMock(spec=[])
        with pytest.raises(AttributeError):
            analyzer.assess_risks([bad])

    def test_assess_risks_no_high_risk(self, analyzer):
        """Low-risk fix doesn't show high risk sections."""
        fix = _make_fix(confidence=0.95, fixed_code="x", explanation="trivial fix")
        fix.file_path = "utils.py"
        result = analyzer.assess_risks([fix])
        assert "Low Risk" in result

    async def test_analyze_container_none_name(self, analyzer):
        with pytest.raises(ValueError):
            await analyzer.analyze_container(None, "logs")

    async def test_analyze_container_non_string_logs(self, analyzer):
        with pytest.raises(ValueError):
            await analyzer.analyze_container("container", 123)

    async def test_generate_fixes_unmapped_error_type(self, analyzer):
        """Error types not in template map get generic fix."""
        error = _make_error(error_type=ErrorType.RUNTIME_ERROR, message="runtime boom")
        fixes = await analyzer.generate_fixes([error])
        assert len(fixes) == 1
        assert fixes[0].confidence == 0.5

    async def test_generate_fixes_known_types(self, analyzer):
        """Known error types get template-based fixes with correct confidence."""
        syntax_err = _make_error(error_type=ErrorType.SYNTAX_ERROR)
        import_err = _make_error(error_type=ErrorType.IMPORT_ERROR)
        name_err = _make_error(error_type=ErrorType.NAME_ERROR)

        fixes = await analyzer.generate_fixes([syntax_err, import_err, name_err])
        assert len(fixes) == 3
        confidences = {f.confidence for f in fixes}
        assert 0.7 in confidences  # syntax
        assert 0.8 in confidences  # import
        assert 0.6 in confidences  # name

    async def test_generate_fix_for_error_exception(self, analyzer):
        """_generate_fix_for_error returns None on exception."""
        bad_error = MagicMock()
        bad_error.error_type = MagicMock()
        bad_error.error_type.value = property(lambda s: (_ for _ in ()).throw(RuntimeError))
        result = analyzer._generate_fix_for_error(bad_error)
        # Should return None or a CodeFix (handles exception)
        assert result is None or isinstance(result, CodeFix)

    def test_analyze_root_causes_error_without_file_path(self, analyzer):
        """Error with None file_path doesn't crash file pattern analysis."""
        error = _make_error(file_path=None)
        result = analyzer.analyze_root_causes([error], [])
        assert "Root Cause Analysis" in result


# ============================================================================
# 4. SimilarityDetector  (uncovered paths)
# ============================================================================


class TestSimilarityHelpers:
    def test_detect_language_python(self):
        assert _detect_language("foo.py") == "python"

    def test_detect_language_js(self):
        assert _detect_language("app.js") == "javascript"

    def test_detect_language_unknown(self):
        assert _detect_language("data.xyz") == "unknown"

    def test_detect_language_various(self):
        assert _detect_language("a.ts") == "typescript"
        assert _detect_language("a.java") == "java"
        assert _detect_language("a.rs") == "rust"
        assert _detect_language("a.go") == "go"
        assert _detect_language("a.cs") == "csharp"
        assert _detect_language("a.cpp") == "cpp"
        assert _detect_language("a.c") == "c"
        assert _detect_language("a.rb") == "ruby"
        assert _detect_language("a.swift") == "swift"
        assert _detect_language("a.kt") == "kotlin"

    def test_file_hash_deterministic(self):
        h1 = _file_hash("hello")
        h2 = _file_hash("hello")
        assert h1 == h2

    def test_file_hash_different(self):
        assert _file_hash("a") != _file_hash("b")

    def test_ast_node_types_valid(self):
        nodes = _ast_node_types("x = 1")
        assert "Module" in nodes

    def test_ast_node_types_syntax_error(self):
        assert _ast_node_types("def (") == []

    def test_tokenize_basic(self):
        tokens = _tokenize("x = 1 + 2")
        assert "x" in tokens
        assert "1" in tokens
        assert "2" in tokens

    def test_tokenize_empty(self):
        assert _tokenize("") == []

    def test_sequence_ratio_identical(self):
        assert _sequence_ratio(["a", "b"], ["a", "b"]) == 1.0

    def test_sequence_ratio_different(self):
        assert _sequence_ratio(["a"], ["z"]) < 1.0

    def test_read_file_missing(self):
        assert _read_file("/no/such/path/file.txt") is None

    def test_read_file_valid(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert _read_file(str(f)) == "hello"


class TestSimilarityDetectorEnrich:
    def test_enrich_match_python_files(self):
        match = SimilarityMatch(
            source_file="a.py",
            target_file="b.py",
            source_range=(1, 10),
            target_range=(1, 20),
            similarity_score=0.85,
            similarity_type=SimilarityType.STRUCTURAL,
        )
        enriched = SimilarityDetector._enrich_match(match, SimilarityScope.FILE)
        assert enriched.score == 0.85
        assert enriched.confidence == 0.85
        assert enriched.source_language == "python"
        assert enriched.target_language == "python"
        assert enriched.is_cross_language is False
        assert enriched.scope == SimilarityScope.FILE
        assert "structural" in enriched.summary.lower()

    def test_enrich_match_cross_language(self):
        match = SimilarityMatch(
            source_file="a.py",
            target_file="b.js",
            similarity_score=0.7,
            similarity_type=SimilarityType.SEMANTIC,
        )
        enriched = SimilarityDetector._enrich_match(match)
        assert enriched.is_cross_language is True
        assert enriched.source_language == "python"
        assert enriched.target_language == "javascript"

    def test_enrich_match_unknown_language(self):
        match = SimilarityMatch(
            source_file="a.xyz",
            target_file="b.xyz",
            similarity_score=0.9,
        )
        enriched = SimilarityDetector._enrich_match(match)
        assert enriched.is_cross_language is False  # both unknown


class TestSimilarityDetectorBuildRichResult:
    def test_build_rich_result_empty(self):
        raw = SimilarityAnalysisResult(
            matches=[],
            total_comparisons=0,
            analysis_time_ms=10.0,
        )
        result = SimilarityDetector._build_rich_result(raw, [])
        assert result.total_files_analyzed == 0
        assert result.average_similarity_score == 0.0
        assert result.highest_similarity_match is None

    def test_build_rich_result_with_matches(self):
        m1 = SimilarityMatch(
            source_file="a.py",
            target_file="b.py",
            similarity_score=0.96,
            score=0.96,
            similarity_type=SimilarityType.EXACT,
            is_cross_language=False,
        )
        m2 = SimilarityMatch(
            source_file="c.py",
            target_file="d.js",
            similarity_score=0.7,
            score=0.7,
            similarity_type=SimilarityType.STRUCTURAL,
            is_cross_language=True,
            source_language="python",
            target_language="javascript",
        )
        raw = SimilarityAnalysisResult(
            matches=[m1, m2],
            total_comparisons=3,
            analysis_time_ms=50.0,
        )
        result = SimilarityDetector._build_rich_result(raw, ["a.py", "b.py", "c.py", "d.js"])
        assert result.total_files_analyzed == 4
        assert result.total_matches_found == 2
        assert result.highest_similarity_match is not None
        assert result.highest_similarity_match.score == 0.96
        assert len(result.cross_language_matches) == 1
        assert result.analysis_time_seconds == 0.05
        assert len(result.matches_by_score["exact_duplicates"]) == 1


class TestSimilarityDetectorHighLevel:
    @pytest.fixture
    def detector(self):
        return SimilarityDetector()

    async def test_analyze_similarity(self, detector, tmp_path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        code = "def foo(): return 1\n"
        f1.write_text(code)
        f2.write_text(code)

        result = await detector.analyze_similarity(
            [str(f1), str(f2)],
            scope=SimilarityScope.FILE,
            min_similarity_score=0.5,
        )
        assert isinstance(result, SimilarityAnalysisResult)
        assert result.total_files_analyzed == 2
        assert result.total_matches_found >= 1

    async def test_analyze_similarity_cross_language(self, detector, tmp_path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("x = 1\n")
        f2.write_text("x = 1\n")

        result = await detector.analyze_similarity(
            [str(f1), str(f2)],
            detect_cross_language=True,
            min_similarity_score=0.5,
        )
        assert isinstance(result, SimilarityAnalysisResult)

    async def test_analyze_similarity_max_files(self, detector, tmp_path):
        for i in range(5):
            (tmp_path / f"f{i}.py").write_text("pass\n")
        files = [str(tmp_path / f"f{i}.py") for i in range(5)]

        result = await detector.analyze_similarity(files, max_files=2)
        assert result.total_files_analyzed == 2

    async def test_detect_code_clones(self, detector, tmp_path):
        code = "for i in range(10):\n    print(i)\n"
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text(code)
        f2.write_text(code)

        clones = await detector.detect_code_clones([str(f1), str(f2)], clone_threshold=0.5)
        assert isinstance(clones, list)
        assert len(clones) >= 1

    async def test_find_cross_language_patterns(self, detector, tmp_path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.js"
        f1.write_text("def foo(): return 1\n")
        f2.write_text("function foo() { return 1; }\n")

        patterns = await detector.find_cross_language_patterns(
            [str(f1), str(f2)], pattern_threshold=0.1
        )
        assert isinstance(patterns, list)
        # Only cross-language matches returned
        for m in patterns:
            assert m.is_cross_language

    async def test_compare_token_both_empty(self, detector):
        score = detector._compare("", "", SimilarityType.TOKEN)
        assert score == 1.0

    async def test_compare_token_one_empty(self, detector):
        score = detector._compare("hello world", "", SimilarityType.TOKEN)
        assert score == 0.0

    async def test_compare_structural_both_unparseable(self, detector):
        score = detector._compare(
            "not valid python {{{", "also not valid {{", SimilarityType.STRUCTURAL
        )
        assert score == 1.0

    async def test_compare_structural_one_unparseable(self, detector):
        score = detector._compare("x = 1", "not valid python {{{", SimilarityType.STRUCTURAL)
        assert score == 0.0

    def test_singleton_accessor(self):
        import services.similarity_detector as mod

        mod._detector = None
        d1 = get_similarity_detector()
        d2 = get_similarity_detector()
        assert d1 is d2
        mod._detector = None  # cleanup


# ============================================================================
# 5. SemanticAnalysisService  (uncovered paths)
# ============================================================================


class TestSemanticInternalHelpers:
    def test_parse_python_decorators(self):
        code = "@decorator\ndef foo(): pass\n"
        result = _parse_python(code)
        funcs = [e for e in result["entities"] if e["type"] == "function"]
        assert len(funcs) == 1
        assert "decorator" in funcs[0]["decorators"]

    def test_parse_python_attribute_decorator(self):
        code = "@module.decorator\ndef foo(): pass\n"
        result = _parse_python(code)
        funcs = [e for e in result["entities"] if e["type"] == "function"]
        assert "module.decorator" in funcs[0]["decorators"]

    def test_parse_python_call_decorator(self):
        code = "@decorator(arg)\ndef foo(): pass\n"
        result = _parse_python(code)
        funcs = [e for e in result["entities"] if e["type"] == "function"]
        assert "decorator" in funcs[0]["decorators"]

    def test_parse_python_import_from_no_module(self):
        code = "from . import something\n"
        result = _parse_python(code)
        imports = [e for e in result["entities"] if e["type"] == "import"]
        assert len(imports) == 1

    def test_parse_python_class_no_bases(self):
        code = "class Standalone: pass\n"
        result = _parse_python(code)
        classes = [e for e in result["entities"] if e["type"] == "class"]
        assert len(classes) == 1
        assert classes[0]["name"] == "Standalone"

    def test_parse_python_attribute_call(self):
        code = "def f():\n    obj.method()\n"
        result = _parse_python(code)
        calls = [r for r in result["relationships"] if r["type"] == "calls"]
        assert any(r["target"] == "method" for r in calls)

    def test_parse_generic_function(self):
        code = "function greet(name) {\n    return name;\n}\n"
        result = _parse_generic(code, "javascript")
        funcs = [e for e in result["entities"] if e["type"] == "function"]
        assert len(funcs) == 1
        assert funcs[0]["name"] == "greet"

    def test_parse_generic_class_with_extends(self):
        code = "class Dog extends Animal {\n}\n"
        result = _parse_generic(code, "javascript")
        classes = [e for e in result["entities"] if e["type"] == "class"]
        assert len(classes) == 1
        inherits = [r for r in result["relationships"] if r["type"] == "inherits"]
        assert len(inherits) == 1
        assert inherits[0]["target"] == "Animal"

    def test_parse_generic_import(self):
        code = "import os\n"
        result = _parse_generic(code, "generic")
        imports = [e for e in result["entities"] if e["type"] == "import"]
        assert len(imports) == 1

    def test_parse_generic_rust_function(self):
        code = "pub fn calculate(x: i32) -> i32 {\n    x + 1\n}\n"
        result = _parse_generic(code, "rust")
        funcs = [e for e in result["entities"] if e["type"] == "function"]
        assert len(funcs) == 1
        assert funcs[0]["name"] == "calculate"

    def test_name_of_attribute(self):
        code = "a.b.c"
        tree = ast.parse(code, mode="eval")
        result = _name_of(tree.body)
        assert result == "a.b.c"

    def test_name_of_non_name(self):
        # Constant node
        tree = ast.parse("42", mode="eval")
        result = _name_of(tree.body)
        assert result == ""

    def test_call_name_attribute(self):
        code = "obj.method()"
        tree = ast.parse(code)
        call_node = tree.body[0].value
        result = _call_name(call_node)
        assert result == "method"

    def test_call_name_non_name(self):
        code = "(lambda: None)()"
        tree = ast.parse(code)
        call_node = tree.body[0].value
        result = _call_name(call_node)
        assert result == ""

    def test_decorator_name_non_standard(self):
        # Pass a constant node
        node = ast.Constant(value=42)
        result = _decorator_name(node)
        assert result == ""


class TestSemanticAnalysisServiceUncovered:
    @pytest.fixture
    def svc(self):
        return SemanticAnalysisService()

    async def test_analyze_generic_language(self, svc):
        code = "function hello() {\n    console.log('hi');\n}\n"
        result = await svc.analyze(code, "javascript")
        assert result["language"] == "javascript"
        funcs = [e for e in result["entities"] if e["type"] == "function"]
        assert len(funcs) >= 1

    async def test_analyze_unexpected_exception(self, svc):
        with patch.object(
            _sas_mod,
            "_parse_python",
            side_effect=RuntimeError("boom"),
        ):
            result = await svc.analyze("x = 1", "python")
        assert len(result["errors"]) > 0
        assert "RuntimeError" in result["errors"][0]

    async def test_analyze_deduplicates_relationships(self, svc):
        code = "def f():\n    g()\n    g()\n"
        result = await svc.analyze(code, "python")
        calls = [r for r in result["relationships"] if r["type"] == "calls"]
        # Deduplicated: only one (f, g, calls) triple
        assert len(calls) == 1

    async def test_analyze_codebase_semantics(self, svc, tmp_path):
        (tmp_path / "main.py").write_text("def main():\n    helper()\n")
        (tmp_path / "util.py").write_text("def helper(): pass\n")

        result = await svc.analyze_codebase_semantics(
            [tmp_path / "main.py", tmp_path / "util.py"],
            analysis_id="test-001",
        )
        assert result["analysis_summary"]["analysis_id"] == "test-001"
        assert result["analysis_summary"]["total_files"] == 2
        assert result["analysis_summary"]["total_entities"] > 0
        # Entities have file_path set
        for e in result["entities"]:
            assert "file_path" in e

    async def test_analyze_codebase_semantics_with_errors(self, svc, tmp_path):
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(:\n")

        result = await svc.analyze_codebase_semantics([bad_file], analysis_id="err-001")
        assert len(result["analysis_summary"]["errors"]) > 0

    async def test_analyze_codebase_semantics_non_python(self, svc, tmp_path):
        js_file = tmp_path / "app.js"
        js_file.write_text("function hello() {}\n")

        result = await svc.analyze_codebase_semantics([js_file], analysis_id="js-001")
        assert "javascript" in result["analysis_summary"]["languages"]

    async def test_analyze_codebase_semantics_unreadable_file(self, svc, tmp_path):
        missing = tmp_path / "missing.py"
        result = await svc.analyze_codebase_semantics([missing], analysis_id="miss-001")
        assert len(result["analysis_summary"]["errors"]) > 0

    async def test_get_semantic_insights_basic(self, svc, tmp_path):
        (tmp_path / "main.py").write_text(
            "def a():\n    b()\n    c()\n    d()\n\ndef b(): pass\ndef c(): pass\ndef d(): pass\n"
        )

        analysis = await svc.analyze_codebase_semantics(
            [tmp_path / "main.py"], analysis_id="insight-001"
        )
        insights = await svc.get_semantic_insights(analysis)

        assert "language_distribution" in insights
        assert "complexity_hotspots" in insights
        assert "coupling_analysis" in insights

    async def test_get_semantic_insights_empty(self, svc):
        analysis = {
            "entities": [],
            "basic_relationships": [],
            "analysis_summary": {"languages": {}},
        }
        insights = await svc.get_semantic_insights(analysis)
        assert insights["complexity_hotspots"] == []
        assert insights["language_distribution"] == {}

    async def test_get_semantic_insights_hotspot_detection(self, svc):
        """Entity with >= 3 relationships qualifies as hotspot."""
        entities = [
            {"name": "hub", "type": "function", "file_path": "hub.py"},
        ]
        relationships = [
            {"source": "hub", "target": "a", "type": "calls"},
            {"source": "hub", "target": "b", "type": "calls"},
            {"source": "hub", "target": "c", "type": "calls"},
        ]
        analysis = {
            "entities": entities,
            "basic_relationships": relationships,
            "analysis_summary": {"languages": {"python": 1}},
        }
        insights = await svc.get_semantic_insights(analysis)
        hotspots = insights["complexity_hotspots"]
        assert len(hotspots) >= 1
        assert hotspots[0]["entity_name"] == "hub"
        assert hotspots[0]["relationship_count"] == 3

    def test_singleton_accessor(self):
        import services.semantic_analysis_service as mod

        mod._semantic_analysis_service = None
        s1 = get_semantic_analysis_service()
        s2 = get_semantic_analysis_service()
        assert s1 is s2
        mod._semantic_analysis_service = None


# ---------------------------------------------------------------------------
# GitHubRepoRequest validators
# ---------------------------------------------------------------------------


class TestGitHubRepoRequestValidators:
    def test_valid_repo_url(self):
        req = GitHubRepoRequest(repo_url="https://github.com/owner/repo", branch="main")
        assert req.repo_url == "https://github.com/owner/repo"

    def test_repo_url_empty_raises(self):
        with pytest.raises(ValueError):
            GitHubRepoRequest(repo_url="", branch="main")

    def test_repo_url_pr_format(self):
        req = GitHubRepoRequest(repo_url="https://github.com/owner/repo/pull/42", branch="main")
        assert req.repo_url == "https://github.com/owner/repo"
        assert req.pr_number == 42
        assert req.provision_type == "pull_request"

    def test_validate_branch_empty_raises(self):
        with pytest.raises(ValueError):
            GitHubRepoRequest(repo_url="https://github.com/owner/repo", branch="")

    def test_validate_branch_strips(self):
        req = GitHubRepoRequest(repo_url="https://github.com/owner/repo", branch="  develop  ")
        assert req.branch == "develop"

    def test_validate_pr_number_zero_raises(self):
        with pytest.raises(ValueError):
            GitHubRepoRequest(repo_url="https://github.com/owner/repo", branch="main", pr_number=0)

    def test_validate_pr_number_positive(self):
        req = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo", branch="main", pr_number=5
        )
        assert req.pr_number == 5

    def test_non_pr_url_no_parse(self):
        req = GitHubRepoRequest(repo_url="https://github.com/owner/repo", branch="main")
        assert req.pr_number is None
        assert req.provision_type == "branch"


# ---------------------------------------------------------------------------
# ContainerInfo sync_endpoint_fields / ContainerSpec.validate_name
# ---------------------------------------------------------------------------

_CI_BASE = {
    "container_name": "test",
    "codebase_path": "/app/repo",
    "repo_url": "https://github.com/owner/repo",
    "branch": "main",
    "status": "running",
}


class TestContainerInfoSyncEndpoints:
    def test_public_endpoint_synced_from_api(self):
        info = ContainerInfo(
            **_CI_BASE, api_endpoint="http://api:8080", internal_endpoint="http://internal:8080"
        )
        assert info.public_endpoint == "http://api:8080"

    def test_api_endpoint_synced_from_public(self):
        info = ContainerInfo(
            **_CI_BASE,
            api_endpoint="",
            public_endpoint="http://public:8080",
            internal_endpoint="http://internal:8080",
        )
        assert info.api_endpoint == "http://public:8080"

    def test_api_endpoint_synced_from_internal(self):
        info = ContainerInfo(**_CI_BASE, api_endpoint="", internal_endpoint="http://internal:8080")
        assert info.api_endpoint == "http://internal:8080"


class TestContainerSpecValidateName:
    def test_valid_name(self):
        spec = ContainerSpec(name="my-container", image="python:3.12", port=8080)
        assert spec.name == "my-container"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            ContainerSpec(name="", image="python:3.12", port=8080)

    def test_whitespace_only_name_raises(self):
        with pytest.raises(ValueError):
            ContainerSpec(name="   ", image="python:3.12", port=8080)

    def test_name_stripped(self):
        spec = ContainerSpec(name="  my-app  ", image="python:3.12", port=8080)
        assert spec.name == "my-app"
