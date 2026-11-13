"""Unit tests for processing misc modules:

- codebase_analysis_processor.py
- parallel_processor.py
- multi_language_analysis_processor.py
- task_execution_engine.py
- processors/snippet_evaluator.py
- extractors/dependency_analyzer.py
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Guard against earlier tests (outside our scope) that replace
# sys.modules["processing"] with a MagicMock.
# ---------------------------------------------------------------------------
_src_dir = Path(__file__).resolve().parents[2] / "src"
_src_processing = str(_src_dir / "processing")


def _repair_processing_module():
    """Ensure sys.modules['processing'] is a real package pointing at src/."""
    import importlib

    proc = sys.modules.get("processing")
    if proc is None or isinstance(proc, MagicMock) or not getattr(proc, "__path__", None):
        proc = types.ModuleType("processing")
        proc.__path__ = [_src_processing]
        proc.__package__ = "processing"
        sys.modules["processing"] = proc
    elif _src_processing not in proc.__path__:
        proc.__path__.insert(0, _src_processing)

    # Also purge cached submodules that may be MagicMock
    for key in list(sys.modules):
        if key.startswith("processing.") and isinstance(sys.modules[key], MagicMock):
            del sys.modules[key]

    # Pre-import the submodules that tests use via patch().
    # Also re-import if the cached module is a MagicMock or missing the right attrs.
    proc_mod = sys.modules.get("processing")
    for submod in ("codebase_analysis_processor", "multi_language_analysis_processor"):
        fqn = f"processing.{submod}"
        existing = sys.modules.get(fqn)
        if existing is not None and not isinstance(existing, MagicMock):
            # Module already loaded, just ensure it's set as an attribute
            if proc_mod is not None:
                setattr(proc_mod, submod, existing)
        else:
            # Remove stale entry so importlib re-imports from disk
            sys.modules.pop(fqn, None)
            try:
                importlib.import_module(fqn)
            except Exception:
                # Last resort: force-import from file
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    fqn, str(_src_dir / "processing" / f"{submod}.py")
                )
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[fqn] = mod
                    try:
                        spec.loader.exec_module(mod)
                        if proc_mod is not None:
                            setattr(proc_mod, submod, mod)
                    except Exception:
                        pass


_repair_processing_module()

# ---------------------------------------------------------------------------
# Stub modules that truly do not exist locally.
# coding_client.* stubs are installed by conftest.py.
# app.core.file_service and processing.processor_factory are real modules.
# ---------------------------------------------------------------------------

# models.github_models (not a local file — stub CodeSnippet & RepositoryContext)


class _CodeSnippet:
    def __init__(self, code="", language="python", **kw):
        self.code = code
        self.language = language
        for k, v in kw.items():
            setattr(self, k, v)


class _RepositoryContext:
    def __init__(self, **kw):
        self.style_guide = kw.get("style_guide", {})
        self.existing_patterns = kw.get("existing_patterns", [])
        self.api_surface = kw.get("api_surface", {})
        self.size_metrics = kw.get("size_metrics", {})
        for k, v in kw.items():
            setattr(self, k, v)


if "models.github_models" not in sys.modules or isinstance(
    sys.modules.get("models.github_models"), MagicMock
):
    _gh_mod = types.ModuleType("models.github_models")
    _gh_mod.CodeSnippet = _CodeSnippet
    _gh_mod.RepositoryContext = _RepositoryContext
    sys.modules["models.github_models"] = _gh_mod

# services.multi_language_metrics — real module now exists, clean up any stale MagicMock
for _key in list(sys.modules):
    if _key == "services.multi_language_metrics" and isinstance(sys.modules[_key], MagicMock):
        del sys.modules[_key]


# ===================================================================
# IMPORTS under test (after stubs are in place)
# ===================================================================

from processing.extractors.dependency_analyzer import CodeDependencyExtractor
from processing.processors.snippet_evaluator import (
    CompatibilityChecker,
    PatternMatcher,
    SnippetEvaluator,
    StyleChecker,
)
from processing.task_execution_engine import (
    BaseTaskProcessor,
    CentralityCalculationProcessor,
    _build_incoming_links,
    _calculate_outgoing_counts,
    _calculate_pagerank_iteration,
    calculate_degree_centrality,
    calculate_pagerank,
)

# ===================================================================
# task_execution_engine tests
# ===================================================================


class TestBaseTaskProcessor:
    @pytest.mark.asyncio
    async def test_process_raises_not_implemented(self):
        p = BaseTaskProcessor()
        with pytest.raises(NotImplementedError):
            await p.process({})

    def test_get_processor_type(self):
        p = BaseTaskProcessor()
        assert p.get_processor_type() == "BaseTaskProcessor"


class TestCalculateDegreeCentrality:
    @pytest.mark.asyncio
    async def test_empty_nodes(self):
        conn = AsyncMock()
        cursor = AsyncMock()
        cursor.fetchall = AsyncMock(return_value=[])
        conn.execute = AsyncMock(return_value=cursor)
        result = await calculate_degree_centrality(conn, "g1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_single_node_no_edges(self):
        conn = AsyncMock()
        # First call returns node list
        node_cursor = AsyncMock()
        node_cursor.fetchall = AsyncMock(return_value=[("node1",)])

        # Out-degree and in-degree cursors
        zero_cursor = AsyncMock()
        zero_cursor.fetchone = AsyncMock(return_value=(0,))

        conn.execute = AsyncMock(side_effect=[node_cursor, zero_cursor, zero_cursor])
        result = await calculate_degree_centrality(conn, "g1")
        assert result == {"node1": 0.0}

    @pytest.mark.asyncio
    async def test_with_node_type_filter(self):
        conn = AsyncMock()
        node_cursor = AsyncMock()
        node_cursor.fetchall = AsyncMock(return_value=[("n1",)])
        out_cursor = AsyncMock()
        out_cursor.fetchone = AsyncMock(return_value=(3,))
        in_cursor = AsyncMock()
        in_cursor.fetchone = AsyncMock(return_value=(2,))
        conn.execute = AsyncMock(side_effect=[node_cursor, out_cursor, in_cursor])

        result = await calculate_degree_centrality(conn, "g1", node_type="function")
        assert result == {"n1": 1.0}
        # Verify filter query was used
        first_call = conn.execute.call_args_list[0]
        assert "type = %s" in first_call.args[0]

    @pytest.mark.asyncio
    async def test_normalization_two_nodes(self):
        conn = AsyncMock()
        node_cursor = AsyncMock()
        node_cursor.fetchall = AsyncMock(return_value=[("a",), ("b",)])
        # a: out=4, in=0 => 4
        # b: out=0, in=2 => 2
        cursors = [
            node_cursor,
            AsyncMock(fetchone=AsyncMock(return_value=(4,))),  # a out
            AsyncMock(fetchone=AsyncMock(return_value=(0,))),  # a in
            AsyncMock(fetchone=AsyncMock(return_value=(0,))),  # b out
            AsyncMock(fetchone=AsyncMock(return_value=(2,))),  # b in
        ]
        conn.execute = AsyncMock(side_effect=cursors)
        result = await calculate_degree_centrality(conn, "g1")
        assert result["a"] == 1.0
        assert result["b"] == 0.5


class TestCalculateOutgoingCounts:
    @pytest.mark.asyncio
    async def test_counts(self):
        conn = AsyncMock()
        c1 = AsyncMock(fetchone=AsyncMock(return_value=(5,)))
        c2 = AsyncMock(fetchone=AsyncMock(return_value=(0,)))
        conn.execute = AsyncMock(side_effect=[c1, c2])
        result = await _calculate_outgoing_counts(conn, "g1", ["x", "y"])
        assert result == [5, 0]

    @pytest.mark.asyncio
    async def test_none_row(self):
        conn = AsyncMock()
        c1 = AsyncMock(fetchone=AsyncMock(return_value=None))
        conn.execute = AsyncMock(return_value=c1)
        result = await _calculate_outgoing_counts(conn, "g1", ["x"])
        assert result == [0]


class TestBuildIncomingLinks:
    @pytest.mark.asyncio
    async def test_builds_adjacency(self):
        conn = AsyncMock()
        # For node "a" (index 0), targets: ["b"]
        # For node "b" (index 1), targets: []
        cursor_a = AsyncMock()
        cursor_a.__aiter__ = lambda self: self
        cursor_a.__anext__ = AsyncMock(side_effect=[("b",), StopAsyncIteration])
        cursor_b = AsyncMock()
        cursor_b.__aiter__ = lambda self: self
        cursor_b.__anext__ = AsyncMock(side_effect=[StopAsyncIteration])

        conn.execute = AsyncMock(side_effect=[cursor_a, cursor_b])
        nodes = ["a", "b"]
        node_to_index = {"a": 0, "b": 1}
        result = await _build_incoming_links(conn, "g1", nodes, node_to_index)
        # b (index 1) has incoming from a (index 0)
        assert result[1] == [0]
        assert result[0] == []


class TestCalculatePageRankIteration:
    def test_single_iteration(self):
        # 2 nodes, a->b
        scores = [0.5, 0.5]
        incoming = [[], [0]]  # b gets from a
        outgoing = [1, 0]
        damping = 0.85
        n = 2
        new_scores = _calculate_pagerank_iteration(scores, incoming, outgoing, damping, n)
        assert len(new_scores) == 2
        # node b should have higher score than base
        assert new_scores[1] > new_scores[0]


class TestCalculatePageRank:
    @pytest.mark.asyncio
    async def test_empty_graph(self):
        conn = AsyncMock()
        cursor = AsyncMock()
        cursor.fetchall = AsyncMock(return_value=[])
        conn.execute = AsyncMock(return_value=cursor)
        result = await calculate_pagerank(conn, "g1")
        assert result == {}

    @pytest.mark.asyncio
    async def test_with_node_type(self):
        conn = AsyncMock()
        # nodes query
        node_cursor = AsyncMock()
        node_cursor.fetchall = AsyncMock(return_value=[("x",)])
        # outgoing counts for x
        out_cursor = AsyncMock()
        out_cursor.fetchone = AsyncMock(return_value=(0,))
        # incoming links for x
        inc_cursor = AsyncMock()
        inc_cursor.__aiter__ = lambda self: self
        inc_cursor.__anext__ = AsyncMock(side_effect=[StopAsyncIteration])

        conn.execute = AsyncMock(side_effect=[node_cursor, out_cursor, inc_cursor])
        result = await calculate_pagerank(conn, "g1", node_type="class")
        assert "x" in result


class TestCentralityCalculationProcessor:
    def test_get_supported_metrics(self):
        p = CentralityCalculationProcessor()
        assert "degree_centrality" in p.get_supported_metrics()
        assert "pagerank" in p.get_supported_metrics()

    def test_get_processor_type(self):
        p = CentralityCalculationProcessor()
        assert p.get_processor_type() == "CentralityCalculationProcessor"

    @pytest.mark.asyncio
    async def test_process_success(self):
        p = CentralityCalculationProcessor()
        mock_conn = AsyncMock()
        mock_db_conn = MagicMock()
        mock_db_conn.connect = AsyncMock()
        mock_db_conn.get_connection = MagicMock(return_value=mock_conn)
        mock_db_conn.close = AsyncMock()

        mock_settings = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
        with (
            patch(
                "processing.task_execution_engine.settings",
                mock_settings,
            ),
            patch(
                "processing.task_execution_engine.GraphDatabaseConnection",
                return_value=mock_db_conn,
            ),
            patch(
                "processing.task_execution_engine.calculate_degree_centrality",
                new_callable=AsyncMock,
                return_value={"n1": 1.0},
            ),
            patch(
                "processing.task_execution_engine.calculate_pagerank",
                new_callable=AsyncMock,
                return_value={"n1": 0.5},
            ),
        ):
            result = await p.process({"code_graph_id": "test"})
        assert result["success"] is True
        assert len(result["top_central_nodes"]) == 1
        assert len(result["top_pagerank_nodes"]) == 1

    @pytest.mark.asyncio
    async def test_process_db_file_error(self):
        p = CentralityCalculationProcessor()
        mock_db_conn = MagicMock()
        mock_db_conn.connect = AsyncMock(side_effect=FileNotFoundError("db missing"))
        mock_db_conn.close = AsyncMock()
        mock_settings = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")

        with (
            patch("processing.task_execution_engine.settings", mock_settings),
            patch(
                "processing.task_execution_engine.GraphDatabaseConnection",
                return_value=mock_db_conn,
            ),
        ):
            result = await p.process({"code_graph_id": "bad"})
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_process_runtime_error(self):
        p = CentralityCalculationProcessor()
        mock_db_conn = MagicMock()
        mock_db_conn.connect = AsyncMock(side_effect=RuntimeError("oom"))
        mock_db_conn.close = AsyncMock()
        mock_settings = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
        with (
            patch("processing.task_execution_engine.settings", mock_settings),
            patch(
                "processing.task_execution_engine.GraphDatabaseConnection",
                return_value=mock_db_conn,
            ),
        ):
            result = await p.process({})
        assert result["success"] is False
        assert "Runtime/data error" in result["error"]


# ===================================================================
# snippet_evaluator tests
# ===================================================================


class TestPatternMatcher:
    @pytest.mark.asyncio
    async def test_find_similar_no_patterns(self):
        pm = PatternMatcher()
        snippet = _CodeSnippet(code="def foo(): pass")
        result = await pm.find_similar_patterns(snippet, [])
        assert result == []

    @pytest.mark.asyncio
    async def test_find_similar_above_threshold(self):
        pm = PatternMatcher()
        snippet = _CodeSnippet(code="def foo bar baz qux hello world")
        patterns = [
            {
                "id": "p1",
                "content": "def foo bar baz qux hello world extra",
                "type": "func",
                "file_path": "a.py",
                "description": "d",
            },
        ]
        result = await pm.find_similar_patterns(snippet, patterns)
        assert len(result) >= 1
        assert result[0]["pattern_id"] == "p1"

    @pytest.mark.asyncio
    async def test_find_similar_below_threshold(self):
        pm = PatternMatcher()
        snippet = _CodeSnippet(code="alpha beta gamma")
        patterns = [
            {"id": "p1", "content": "completely different words here xyz"},
        ]
        result = await pm.find_similar_patterns(snippet, patterns)
        assert result == []

    @pytest.mark.asyncio
    async def test_find_similar_caps_top5(self):
        pm = PatternMatcher()
        # All have 100% similarity to snippet
        snippet = _CodeSnippet(code="word")
        patterns = [{"id": f"p{i}", "content": "word"} for i in range(10)]
        result = await pm.find_similar_patterns(snippet, patterns)
        assert len(result) <= 5

    def test_get_similarity_threshold(self):
        pm = PatternMatcher()
        assert pm.get_similarity_threshold() == 0.3

    @pytest.mark.asyncio
    async def test_error_handling(self):
        pm = PatternMatcher()
        # Pass None as snippet to trigger AttributeError
        result = await pm.find_similar_patterns(None, [])
        assert result == []


class TestStyleChecker:
    @pytest.mark.asyncio
    async def test_check_python_with_docstring(self):
        sc = StyleChecker()
        snippet = _CodeSnippet(
            code='def foo():\n    """docstring"""\n    pass',
            language="python",
        )
        result = await sc.check_consistency(snippet, {})
        assert result["documentation_score"] == 0.9

    @pytest.mark.asyncio
    async def test_check_python_without_docstring(self):
        sc = StyleChecker()
        snippet = _CodeSnippet(code="def foo():\n    pass", language="python")
        result = await sc.check_consistency(snippet, {})
        assert result["documentation_score"] == 0.6

    @pytest.mark.asyncio
    async def test_check_javascript_with_jsdoc(self):
        sc = StyleChecker()
        snippet = _CodeSnippet(
            code="/** docs */\nfunction myFunc() { return 1; }",
            language="javascript",
        )
        result = await sc.check_consistency(snippet, {})
        assert result["documentation_score"] == 0.9

    @pytest.mark.asyncio
    async def test_check_javascript_semicolons(self):
        sc = StyleChecker()
        snippet = _CodeSnippet(
            code="const x = 1;\nconst y = 2;",
            language="javascript",
        )
        result = await sc.check_consistency(snippet, {})
        assert result["formatting_score"] == 1.0

    @pytest.mark.asyncio
    async def test_check_unknown_language(self):
        sc = StyleChecker()
        snippet = _CodeSnippet(code="some code", language="ruby")
        result = await sc.check_consistency(snippet, {})
        # Should return defaults
        assert "naming_convention_score" in result

    @pytest.mark.asyncio
    async def test_error_returns_defaults(self):
        sc = StyleChecker()
        result = await sc.check_consistency(None, {})
        assert result["naming_convention_score"] == 0.5

    def test_get_default_style_scores(self):
        sc = StyleChecker()
        defaults = sc.get_default_style_scores()
        assert defaults["naming_convention_score"] == 0.9
        assert defaults["complexity_score"] == 0.8

    @pytest.mark.asyncio
    async def test_python_naming_snake_case(self):
        sc = StyleChecker()
        snippet = _CodeSnippet(
            code="def my_function():\n    pass",
            language="python",
        )
        result = await sc.check_consistency(snippet, {})
        assert result["naming_convention_score"] == 0.9

    @pytest.mark.asyncio
    async def test_javascript_camelcase_naming(self):
        sc = StyleChecker()
        snippet = _CodeSnippet(
            code="function myFunc() {};",
            language="typescript",
        )
        result = await sc.check_consistency(snippet, {})
        assert result["naming_convention_score"] == 0.9


class TestCompatibilityChecker:
    @pytest.mark.asyncio
    async def test_safe_code(self):
        cc = CompatibilityChecker()
        snippet = _CodeSnippet(code="x = 1 + 2", language="python")
        result = await cc.check_compatibility(snippet, {})
        assert result["security_check"] is True

    @pytest.mark.asyncio
    async def test_dangerous_eval(self):
        cc = CompatibilityChecker()
        snippet = _CodeSnippet(code="result = eval(user_input)", language="python")
        result = await cc.check_compatibility(snippet, {})
        assert result["security_check"] is False

    @pytest.mark.asyncio
    async def test_python_imports_detected(self):
        cc = CompatibilityChecker()
        snippet = _CodeSnippet(
            code="import os\nfrom pathlib import Path",
            language="python",
        )
        result = await cc.check_compatibility(snippet, {})
        assert result["uses_standard_imports"] is True

    @pytest.mark.asyncio
    async def test_non_python_no_import_key(self):
        cc = CompatibilityChecker()
        snippet = _CodeSnippet(code="var x = 1;", language="javascript")
        result = await cc.check_compatibility(snippet, {})
        assert "uses_standard_imports" not in result

    @pytest.mark.asyncio
    async def test_error_returns_defaults(self):
        cc = CompatibilityChecker()
        result = await cc.check_compatibility(None, {})
        assert result["dependency_compatibility"] is True

    def test_get_dangerous_patterns(self):
        cc = CompatibilityChecker()
        patterns = cc.get_dangerous_patterns()
        assert "eval(" in patterns
        assert "os.system" in patterns


class TestSnippetEvaluator:
    @pytest.mark.asyncio
    async def test_evaluate_snippet_success(self):
        evaluator = SnippetEvaluator()
        snippet = _CodeSnippet(
            code="def my_func():\n    pass",
            language="python",
        )
        context = _RepositoryContext(
            style_guide={},
            existing_patterns=[],
            api_surface={},
            size_metrics={"files": 10},
        )
        result = await evaluator.evaluate_snippet(snippet, context)
        assert result.success is True
        assert 0.0 <= result.compatibility_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_snippet_with_patterns(self):
        evaluator = SnippetEvaluator()
        snippet = _CodeSnippet(
            code="def foo bar baz qux hello world",
            language="python",
        )
        context = _RepositoryContext(
            style_guide={},
            existing_patterns=[
                {"id": "p1", "content": "def foo bar baz qux hello world extra", "type": "func"},
            ],
            api_surface={},
            size_metrics={},
        )
        result = await evaluator.evaluate_snippet(snippet, context)
        assert result.success is True
        assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_evaluate_snippet_error(self):
        evaluator = SnippetEvaluator()
        # Make style_checker raise by passing None context
        evaluator.style_checker.check_consistency = AsyncMock(side_effect=ValueError("bad"))
        snippet = _CodeSnippet(code="x=1", language="python")
        context = _RepositoryContext()
        result = await evaluator.evaluate_snippet(snippet, context)
        assert result.success is False
        assert result.error == "bad"

    def test_calculate_overall_score_all_zeros(self):
        evaluator = SnippetEvaluator()
        score = evaluator._calculate_overall_score({}, [], {})
        # style defaults 0.5, pattern defaults 0.3, compat defaults 0.5
        assert 0.0 <= score <= 1.0

    def test_calculate_overall_score_with_values(self):
        evaluator = SnippetEvaluator()
        style = {"naming": 0.9, "format": 0.8}
        patterns = [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}, {"x": 5}, {"x": 6}]
        compat = {"dep": True, "api": True}
        score = evaluator._calculate_overall_score(style, patterns, compat)
        assert 0.0 <= score <= 1.0

    def test_generate_recommendations_low_scores(self):
        evaluator = SnippetEvaluator()
        style = {
            "naming_convention_score": 0.5,
            "formatting_score": 0.5,
            "documentation_score": 0.3,
        }
        patterns = []
        compat = {"dependency_compatibility": False, "api_usage_valid": False}
        snippet = _CodeSnippet(code="x=1", language="python")
        recs = evaluator._generate_recommendations(style, patterns, compat, snippet)
        assert any("naming" in r.lower() for r in recs)
        assert any("format" in r.lower() or "style" in r.lower() for r in recs)
        assert any("documentation" in r.lower() or "docstring" in r.lower() for r in recs)
        assert any("dependency" in r.lower() for r in recs)
        assert any("api" in r.lower() for r in recs)

    def test_generate_recommendations_javascript(self):
        evaluator = SnippetEvaluator()
        snippet = _CodeSnippet(code="var x", language="javascript")
        recs = evaluator._generate_recommendations({}, [], {}, snippet)
        assert any("TypeScript" in r for r in recs)

    def test_generate_recommendations_with_patterns(self):
        evaluator = SnippetEvaluator()
        snippet = _CodeSnippet(code="x=1", language="ruby")
        recs = evaluator._generate_recommendations({}, [{"pattern": 1}], {}, snippet)
        assert any("similar patterns" in r.lower() for r in recs)

    def test_clear_evaluation_cache(self):
        evaluator = SnippetEvaluator()
        evaluator._evaluation_cache["key"] = "value"
        evaluator.clear_evaluation_cache()
        assert len(evaluator._evaluation_cache) == 0


# ===================================================================
# dependency_analyzer tests
# ===================================================================


class TestCodeDependencyExtractor:
    def setup_method(self):
        self.extractor = CodeDependencyExtractor()

    @pytest.mark.asyncio
    async def test_analyze_imports_simple(self):
        code = "import os\nimport sys"
        tree = ast.parse(code)
        deps = await self.extractor.analyze_imports(tree, "test.py")
        assert len(deps) == 2
        names = {d.module_name for d in deps}
        assert "os" in names
        assert "sys" in names
        assert all(d.import_type == "import" for d in deps)

    @pytest.mark.asyncio
    async def test_analyze_from_import(self):
        code = "from pathlib import Path"
        tree = ast.parse(code)
        deps = await self.extractor.analyze_imports(tree, "test.py")
        assert len(deps) == 1
        assert deps[0].module_name == "pathlib"
        assert deps[0].import_type == "from_import"
        assert deps[0].metadata["imported_name"] == "Path"

    @pytest.mark.asyncio
    async def test_analyze_star_import(self):
        code = "from os.path import *"
        tree = ast.parse(code)
        deps = await self.extractor.analyze_imports(tree, "test.py")
        assert len(deps) == 1
        assert deps[0].import_type == "from_import_star"
        assert deps[0].metadata["is_star_import"] is True

    @pytest.mark.asyncio
    async def test_analyze_aliased_import(self):
        code = "import numpy as np"
        tree = ast.parse(code)
        deps = await self.extractor.analyze_imports(tree, "test.py")
        assert deps[0].alias == "np"
        assert deps[0].metadata["is_aliased"] is True

    @pytest.mark.asyncio
    async def test_analyze_from_import_alias(self):
        code = "from collections import OrderedDict as OD"
        tree = ast.parse(code)
        deps = await self.extractor.analyze_imports(tree, "test.py")
        assert deps[0].alias == "OD"

    def test_classify_stdlib(self):
        assert self.extractor._classify_dependency("os") == "stdlib"
        assert self.extractor._classify_dependency("sys") == "stdlib"
        assert self.extractor._classify_dependency("json") == "stdlib"

    def test_classify_third_party(self):
        assert self.extractor._classify_dependency("numpy") == "third_party"
        assert self.extractor._classify_dependency("flask") == "third_party"
        assert self.extractor._classify_dependency("requests") == "third_party"

    def test_classify_local(self):
        assert self.extractor._classify_dependency("") == "local"
        assert self.extractor._classify_dependency(".foo") == "local"

    def test_classify_dunder_prefix(self):
        assert self.extractor._classify_dependency("__future__") == "stdlib"
        assert self.extractor._classify_dependency("_thread") == "stdlib"

    def test_classify_short_unknown(self):
        # Short names without dots that aren't recognized
        result = self.extractor._classify_dependency("mymodule")
        assert result == "local"

    def test_classify_long_dotted_unknown(self):
        result = self.extractor._classify_dependency("some.very.deep.module.path.here")
        assert result == "third_party"


class TestDependencyGraph:
    def setup_method(self):
        self.extractor = CodeDependencyExtractor()

    @pytest.mark.asyncio
    async def test_analyze_dependency_graph_empty(self):
        result = await self.extractor.analyze_dependency_graph([])
        assert result["total_dependencies"] == 0
        assert result["by_type"] == {}
        assert result["by_category"] == {}

    @pytest.mark.asyncio
    async def test_analyze_dependency_graph_with_deps(self):
        code = "import os\nimport requests\nfrom . import local_mod"
        tree = ast.parse(code)
        deps = await self.extractor.analyze_imports(tree, "test.py")
        result = await self.extractor.analyze_dependency_graph(deps)
        assert result["total_dependencies"] == 3
        assert len(result["external_dependencies"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_dependency_graph_with_tree(self):
        code = "import os\nx = os.path.join('a', 'b')"
        tree = ast.parse(code)
        deps = await self.extractor.analyze_imports(tree, "test.py")
        result = await self.extractor.analyze_dependency_graph(deps, tree=tree)
        # os is used so no unused imports
        assert not any(
            i["type"] == "unused_import" and i.get("module") == "os"
            for i in result["unused_imports"]
        )

    @pytest.mark.asyncio
    async def test_star_import_issue_detected(self):
        code = "from os.path import *"
        tree = ast.parse(code)
        deps = await self.extractor.analyze_imports(tree, "test.py")
        result = await self.extractor.analyze_dependency_graph(deps)
        star_issues = [i for i in result["potential_issues"] if i["type"] == "star_import"]
        assert len(star_issues) == 1

    @pytest.mark.asyncio
    async def test_deprecated_module_detected(self):
        code = "import imp"
        tree = ast.parse(code)
        deps = await self.extractor.analyze_imports(tree, "test.py")
        result = await self.extractor.analyze_dependency_graph(deps)
        dep_issues = [i for i in result["potential_issues"] if i["type"] == "deprecated_module"]
        assert len(dep_issues) == 1

    @pytest.mark.asyncio
    async def test_security_sensitive_import(self):
        # "system" is not in the security_sensitive set {"eval","exec","subprocess","os"}
        # so verify with "exec" instead
        code2 = "from builtins import exec"
        tree2 = ast.parse(code2)
        deps2 = await self.extractor.analyze_imports(tree2, "test.py")
        result2 = await self.extractor.analyze_dependency_graph(deps2)
        sec_issues = [i for i in result2["potential_issues"] if i["type"] == "security_sensitive"]
        assert len(sec_issues) == 1


class TestUnusedImports:
    def setup_method(self):
        self.extractor = CodeDependencyExtractor()

    @pytest.mark.asyncio
    async def test_detect_unused_import(self):
        code = "import os\nx = 1"
        tree = ast.parse(code)
        deps = await self.extractor.analyze_imports(tree, "test.py")
        unused = await self.extractor.detect_unused_imports(tree, deps)
        assert len(unused) == 1
        assert unused[0]["type"] == "unused_import"
        assert "os" in unused[0]["message"]

    @pytest.mark.asyncio
    async def test_used_import_not_flagged(self):
        code = "import os\npath = os.getcwd()"
        tree = ast.parse(code)
        deps = await self.extractor.analyze_imports(tree, "test.py")
        unused = await self.extractor.detect_unused_imports(tree, deps)
        assert len(unused) == 0

    @pytest.mark.asyncio
    async def test_star_import_skipped(self):
        code = "from os import *\npath = getcwd()"
        tree = ast.parse(code)
        deps = await self.extractor.analyze_imports(tree, "test.py")
        unused = await self.extractor.detect_unused_imports(tree, deps)
        # Star imports are skipped in tracking
        assert len(unused) == 0

    @pytest.mark.asyncio
    async def test_aliased_import_used(self):
        code = "import numpy as np\nx = np.array([1])"
        tree = ast.parse(code)
        deps = await self.extractor.analyze_imports(tree, "test.py")
        unused = await self.extractor.detect_unused_imports(tree, deps)
        assert len(unused) == 0

    @pytest.mark.asyncio
    async def test_aliased_import_unused(self):
        code = "import numpy as np\nx = 1"
        tree = ast.parse(code)
        deps = await self.extractor.analyze_imports(tree, "test.py")
        unused = await self.extractor.detect_unused_imports(tree, deps)
        assert len(unused) == 1
        assert "np" in unused[0]["message"]


class TestFindUsedSymbols:
    def setup_method(self):
        self.extractor = CodeDependencyExtractor()

    def test_name_usage(self):
        code = "x = os.path.join('a')"
        tree = ast.parse(code)
        symbols = self.extractor._find_used_symbols(tree)
        assert "os" in symbols

    def test_function_def_body_visited(self):
        code = "def foo():\n    return os.getcwd()"
        tree = ast.parse(code)
        symbols = self.extractor._find_used_symbols(tree)
        assert "os" in symbols

    def test_class_def_base_visited(self):
        code = "class Foo(Base):\n    pass"
        tree = ast.parse(code)
        symbols = self.extractor._find_used_symbols(tree)
        assert "Base" in symbols

    def test_async_function_body_visited(self):
        code = "async def foo():\n    await bar()"
        tree = ast.parse(code)
        symbols = self.extractor._find_used_symbols(tree)
        assert "bar" in symbols

    def test_import_statements_skipped(self):
        code = "import os\nfrom sys import path"
        tree = ast.parse(code)
        symbols = self.extractor._find_used_symbols(tree)
        # Import nodes are skipped, so os and path shouldn't appear as "used"
        assert "os" not in symbols


class TestCircularDependencies:
    def setup_method(self):
        self.extractor = CodeDependencyExtractor()

    @pytest.mark.asyncio
    async def test_no_cycle(self):
        deps = {"a": ["b"], "b": ["c"], "c": []}
        result = await self.extractor.detect_circular_dependencies("a", deps)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_simple_cycle(self):
        deps = {"a": ["b"], "b": ["a"]}
        result = await self.extractor.detect_circular_dependencies("a", deps)
        assert len(result) >= 1
        assert result[0]["type"] == "circular_dependency"

    @pytest.mark.asyncio
    async def test_indirect_cycle(self):
        deps = {"a": ["b"], "b": ["c"], "c": ["a"]}
        result = await self.extractor.detect_circular_dependencies("a", deps)
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_no_deps(self):
        result = await self.extractor.detect_circular_dependencies("a", {})
        assert result == []


class TestDependencySummary:
    def setup_method(self):
        self.extractor = CodeDependencyExtractor()

    @pytest.mark.asyncio
    async def test_empty_summary(self):
        result = await self.extractor.get_dependency_summary([])
        assert result["total_count"] == 0
        assert result["stdlib_count"] == 0
        assert result["has_star_imports"] is False

    @pytest.mark.asyncio
    async def test_summary_with_deps(self):
        code = "import os\nimport requests\nfrom . import local"
        tree = ast.parse(code)
        deps = await self.extractor.analyze_imports(tree, "test.py")
        result = await self.extractor.get_dependency_summary(deps)
        assert result["total_count"] == 3
        assert result["stdlib_count"] >= 1
        assert result["unique_modules"] >= 2


class TestBuildProjectDependencyGraph:
    def setup_method(self):
        self.extractor = CodeDependencyExtractor()

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_path):
        result = await self.extractor.build_project_dependency_graph(str(tmp_path))
        assert result == {}

    @pytest.mark.asyncio
    async def test_single_file_no_local_deps(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_text("import os\nx = 1\n")
        result = await self.extractor.build_project_dependency_graph(str(tmp_path))
        assert "main.py" in result

    @pytest.mark.asyncio
    async def test_syntax_error_file_skipped(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def foo(\n")  # syntax error
        result = await self.extractor.build_project_dependency_graph(str(tmp_path))
        # File with syntax error is skipped
        assert "bad.py" not in result

    @pytest.mark.asyncio
    async def test_local_dependency_resolved(self, tmp_path):
        (tmp_path / "helper.py").write_text("x = 1\n")
        (tmp_path / "main.py").write_text("import helper\nhelper.x\n")
        result = await self.extractor.build_project_dependency_graph(str(tmp_path))
        assert "main.py" in result


# ===================================================================
# codebase_analysis_processor tests
# ===================================================================


class TestCodebaseAnalysisProcessor:
    @pytest.fixture(autouse=True)
    def _fix_processing_module(self):
        _repair_processing_module()
        yield
        _repair_processing_module()

    @pytest.mark.asyncio
    async def test_process_invalid_params(self):
        with (
            patch("processing.codebase_analysis_processor.get_settings") as mock_gs,
            patch("processing.codebase_analysis_processor.TaskResultProcessor"),
            patch("processing.codebase_analysis_processor.FileProcessingResultHandler"),
            patch("processing.codebase_analysis_processor.TaskErrorHandler"),
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.codebase_analysis_processor import CodebaseAnalysisProcessor

            proc = CodebaseAnalysisProcessor()
            result = await proc.process({})  # missing codebase_path
            assert result["success"] is False
            assert "errors" in result

    @pytest.mark.asyncio
    async def test_process_invalid_filesystem(self):
        with (
            patch("processing.codebase_analysis_processor.get_settings") as mock_gs,
            patch("processing.codebase_analysis_processor.TaskResultProcessor"),
            patch("processing.codebase_analysis_processor.FileProcessingResultHandler"),
            patch("processing.codebase_analysis_processor.TaskErrorHandler"),
            patch("processing.codebase_analysis_processor.FileSystemValidator") as mock_fsv,
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            mock_fsv.validate_codebase_path.return_value = {
                "is_valid": False,
                "error": "not found",
            }
            from processing.codebase_analysis_processor import CodebaseAnalysisProcessor

            proc = CodebaseAnalysisProcessor()
            result = await proc.process({"codebase_path": "/nonexistent", "code_graph_id": "test"})
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_get_supported_file_types(self):
        with (
            patch("processing.codebase_analysis_processor.get_settings") as mock_gs,
            patch("processing.codebase_analysis_processor.TaskResultProcessor"),
            patch("processing.codebase_analysis_processor.FileProcessingResultHandler"),
            patch("processing.codebase_analysis_processor.TaskErrorHandler"),
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.codebase_analysis_processor import CodebaseAnalysisProcessor

            proc = CodebaseAnalysisProcessor()
            types = proc.get_supported_file_types()
            assert isinstance(types, list)
            assert ".py" in types

    @pytest.mark.asyncio
    async def test_process_single_file_no_processor(self):
        with (
            patch("processing.codebase_analysis_processor.get_settings") as mock_gs,
            patch("processing.codebase_analysis_processor.TaskResultProcessor"),
            patch("processing.codebase_analysis_processor.FileProcessingResultHandler"),
            patch("processing.codebase_analysis_processor.TaskErrorHandler"),
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.codebase_analysis_processor import CodebaseAnalysisProcessor

            proc = CodebaseAnalysisProcessor()
            factory = MagicMock()
            factory.get_processor.return_value = None
            ctx = {
                "factory": factory,
                "file_path": "/test.py",
                "language": "python",
                "conn": AsyncMock(),
                "code_graph_id": "test",
            }
            result = await proc._process_single_file(ctx)
            assert result is False

    @pytest.mark.asyncio
    async def test_process_single_file_success(self):
        with (
            patch("processing.codebase_analysis_processor.get_settings") as mock_gs,
            patch("processing.codebase_analysis_processor.TaskResultProcessor"),
            patch("processing.codebase_analysis_processor.FileProcessingResultHandler"),
            patch("processing.codebase_analysis_processor.TaskErrorHandler"),
            patch("processing.codebase_analysis_processor.ProcessingUtils") as mock_pu,
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.codebase_analysis_processor import CodebaseAnalysisProcessor

            proc = CodebaseAnalysisProcessor()
            proc.file_result_handler = AsyncMock()
            factory = MagicMock()
            factory.get_processor.return_value = MagicMock()
            mock_pu.process_file_safely = AsyncMock(return_value={"success": True, "result": {}})
            ctx = {
                "factory": factory,
                "file_path": "/test.py",
                "language": "python",
                "conn": AsyncMock(),
                "code_graph_id": "test",
            }
            result = await proc._process_single_file(ctx)
            assert result is True

    @pytest.mark.asyncio
    async def test_process_single_file_processing_failure(self):
        with (
            patch("processing.codebase_analysis_processor.get_settings") as mock_gs,
            patch("processing.codebase_analysis_processor.TaskResultProcessor"),
            patch("processing.codebase_analysis_processor.FileProcessingResultHandler"),
            patch("processing.codebase_analysis_processor.TaskErrorHandler"),
            patch("processing.codebase_analysis_processor.ProcessingUtils") as mock_pu,
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.codebase_analysis_processor import CodebaseAnalysisProcessor

            proc = CodebaseAnalysisProcessor()
            factory = MagicMock()
            factory.get_processor.return_value = MagicMock()
            mock_pu.process_file_safely = AsyncMock(return_value={"success": False})
            ctx = {
                "factory": factory,
                "file_path": "/test.py",
                "language": "python",
                "conn": AsyncMock(),
                "code_graph_id": "test",
            }
            result = await proc._process_single_file(ctx)
            assert result is False

    @pytest.mark.asyncio
    async def test_process_single_file_exception(self):
        with (
            patch("processing.codebase_analysis_processor.get_settings") as mock_gs,
            patch("processing.codebase_analysis_processor.TaskResultProcessor"),
            patch("processing.codebase_analysis_processor.FileProcessingResultHandler"),
            patch("processing.codebase_analysis_processor.TaskErrorHandler"),
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.codebase_analysis_processor import CodebaseAnalysisProcessor

            proc = CodebaseAnalysisProcessor()
            proc.error_handler = MagicMock()
            factory = MagicMock()
            factory.get_processor.side_effect = RuntimeError("boom")
            ctx = {
                "factory": factory,
                "file_path": "/test.py",
                "language": "python",
                "conn": AsyncMock(),
                "code_graph_id": "test",
            }
            result = await proc._process_single_file(ctx)
            assert result is False

    @pytest.mark.asyncio
    async def test_process_full_success(self):
        mock_settings = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
        with (
            patch("processing.codebase_analysis_processor.settings", mock_settings),
            patch("processing.codebase_analysis_processor.TaskResultProcessor") as mock_trp,
            patch("processing.codebase_analysis_processor.FileProcessingResultHandler"),
            patch("processing.codebase_analysis_processor.TaskErrorHandler"),
            patch("processing.codebase_analysis_processor.TaskParameterValidator") as mock_tpv,
            patch("processing.codebase_analysis_processor.FileSystemValidator") as mock_fsv,
            patch("processing.codebase_analysis_processor.GraphDatabaseConnection") as mock_gdc,
            patch("processing.codebase_analysis_processor.PerformanceUtils") as mock_perf,
            patch("processing.codebase_analysis_processor.LanguageDetectionUtils") as mock_ldu,
            patch("processing.codebase_analysis_processor.os") as mock_os,
        ):
            mock_tpv.validate_codebase_analysis_params.return_value = {
                "is_valid": True,
                "validated_params": {
                    "codebase_path": "/code",
                    "code_graph_id": "g1",
                    "scan_depth": 1,
                },
            }
            mock_fsv.validate_codebase_path.return_value = {"is_valid": True}

            # postgres cutover: facade exposes a synchronous get_connection() that
            # returns the raw psycopg AsyncConnection after connect().
            mock_db = MagicMock()
            mock_db.connect = AsyncMock()
            mock_db.get_connection.return_value = AsyncMock()
            mock_db.close = AsyncMock()
            mock_gdc.return_value = mock_db

            timer = MagicMock()
            timer.__enter__ = MagicMock(return_value=timer)
            timer.__exit__ = MagicMock(return_value=False)
            timer.start_time = 100.0
            mock_perf.create_timer.return_value = timer

            mock_ldu.initialize_graph_factory = AsyncMock(return_value=MagicMock())
            mock_os.walk.return_value = []
            mock_os.path.join = lambda *a: "/".join(a)

            mock_rp = MagicMock()
            mock_rp.build_analysis_result = AsyncMock(
                return_value={"success": True, "processed_files": 0}
            )
            mock_trp.return_value = mock_rp

            from processing.codebase_analysis_processor import CodebaseAnalysisProcessor

            proc = CodebaseAnalysisProcessor()
            proc.result_processor = mock_rp
            result = await proc.process({"codebase_path": "/code", "code_graph_id": "g1"})
            assert result["success"] is True


# ===================================================================
# multi_language_analysis_processor tests
# ===================================================================


class TestMultiLanguageAnalysisProcessor:
    @pytest.fixture(autouse=True)
    def _fix_processing_module(self):
        _repair_processing_module()
        yield
        _repair_processing_module()

    @pytest.mark.asyncio
    async def test_process_invalid_params(self):
        with (
            patch("processing.multi_language_analysis_processor.get_settings") as mock_gs,
            patch("processing.multi_language_analysis_processor.TaskResultProcessor"),
            patch("processing.multi_language_analysis_processor.AnalysisMetadataManager"),
            patch("processing.multi_language_analysis_processor.TaskErrorHandler"),
            patch(
                "processing.multi_language_analysis_processor.TaskParameterValidator"
            ) as mock_tpv,
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            mock_tpv.validate_multi_language_analysis_params.return_value = {
                "is_valid": False,
                "errors": ["missing params"],
            }
            from processing.multi_language_analysis_processor import MultiLanguageAnalysisProcessor

            proc = MultiLanguageAnalysisProcessor()
            result = await proc.process({})
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_process_invalid_filesystem(self):
        with (
            patch("processing.multi_language_analysis_processor.get_settings") as mock_gs,
            patch("processing.multi_language_analysis_processor.TaskResultProcessor"),
            patch("processing.multi_language_analysis_processor.AnalysisMetadataManager"),
            patch("processing.multi_language_analysis_processor.TaskErrorHandler"),
            patch(
                "processing.multi_language_analysis_processor.TaskParameterValidator"
            ) as mock_tpv,
            patch("processing.multi_language_analysis_processor.FileSystemValidator") as mock_fsv,
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            mock_tpv.validate_multi_language_analysis_params.return_value = {
                "is_valid": True,
                "validated_params": {
                    "code_graph_id": "g1",
                    "codebase_path": "/bad",
                },
            }
            mock_fsv.validate_codebase_path.return_value = {
                "is_valid": False,
                "error": "not found",
            }
            from processing.multi_language_analysis_processor import MultiLanguageAnalysisProcessor

            proc = MultiLanguageAnalysisProcessor()
            result = await proc.process({"codebase_path": "/bad", "code_graph_id": "g1"})
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_get_supported_languages(self):
        with (
            patch("processing.multi_language_analysis_processor.get_settings") as mock_gs,
            patch("processing.multi_language_analysis_processor.TaskResultProcessor"),
            patch("processing.multi_language_analysis_processor.AnalysisMetadataManager"),
            patch("processing.multi_language_analysis_processor.TaskErrorHandler"),
            patch(
                "processing.multi_language_analysis_processor.LanguageDetectionUtils"
            ) as mock_ldu,
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            mock_ldu.get_supported_languages.return_value = ["python", "javascript"]
            from processing.multi_language_analysis_processor import MultiLanguageAnalysisProcessor

            proc = MultiLanguageAnalysisProcessor()
            langs = proc.get_supported_languages()
            assert "python" in langs

    @pytest.mark.asyncio
    async def test_process_languages_no_processor(self):
        with (
            patch("processing.multi_language_analysis_processor.get_settings") as mock_gs,
            patch("processing.multi_language_analysis_processor.TaskResultProcessor"),
            patch("processing.multi_language_analysis_processor.AnalysisMetadataManager"),
            patch("processing.multi_language_analysis_processor.TaskErrorHandler"),
            patch("processing.multi_language_analysis_processor.ProcessingUtils") as mock_pu,
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.multi_language_analysis_processor import MultiLanguageAnalysisProcessor

            proc = MultiLanguageAnalysisProcessor()
            factory = AsyncMock()
            factory.get_processor = AsyncMock(return_value=None)
            mock_pu.create_processing_context.return_value = {
                "language_stats": {"python": 5},
                "supported_files": [],
                "factory": factory,
                "conn": AsyncMock(),
                "code_graph_id": "g1",
            }
            ctx = mock_pu.create_processing_context.return_value
            results = await proc._process_languages(ctx)
            assert results["python"]["error"] == "No processor available"

    @pytest.mark.asyncio
    async def test_process_languages_exception(self):
        with (
            patch("processing.multi_language_analysis_processor.get_settings") as mock_gs,
            patch("processing.multi_language_analysis_processor.TaskResultProcessor"),
            patch("processing.multi_language_analysis_processor.AnalysisMetadataManager"),
            patch("processing.multi_language_analysis_processor.TaskErrorHandler"),
            patch("processing.multi_language_analysis_processor.ProcessingUtils") as mock_pu,
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.multi_language_analysis_processor import MultiLanguageAnalysisProcessor

            proc = MultiLanguageAnalysisProcessor()
            proc.error_handler = MagicMock()
            factory = AsyncMock()
            factory.get_processor = AsyncMock(side_effect=RuntimeError("fail"))
            mock_pu.create_processing_context.return_value = {
                "language_stats": {"rust": 3},
                "supported_files": [],
                "factory": factory,
                "conn": AsyncMock(),
                "code_graph_id": "g1",
            }
            ctx = mock_pu.create_processing_context.return_value
            results = await proc._process_languages(ctx)
            assert "error" in results["rust"]
            assert results["rust"]["total_files"] == 3

    @pytest.mark.asyncio
    async def test_analyze_codebase_nonexistent_path(self):
        with (
            patch("processing.multi_language_analysis_processor.get_settings") as mock_gs,
            patch("processing.multi_language_analysis_processor.TaskResultProcessor"),
            patch("processing.multi_language_analysis_processor.AnalysisMetadataManager"),
            patch("processing.multi_language_analysis_processor.TaskErrorHandler"),
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.multi_language_analysis_processor import MultiLanguageAnalysisProcessor

            proc = MultiLanguageAnalysisProcessor()
            with pytest.raises(FileNotFoundError):
                await proc._analyze_codebase(
                    "/definitely/not/a/real/path/abc123",
                    MagicMock(),
                    AsyncMock(),
                    "g1",
                )


# ===================================================================
# parallel_processor tests
# ===================================================================


class TestProcessingMetrics:
    def test_initial_state(self):
        from processing.parallel_processor import ProcessingMetrics

        m = ProcessingMetrics()
        assert m.start_time is None
        assert m.end_time is None
        assert m.total_files == 0
        assert m.processed_files == 0
        assert m.failed_files == 0

    def test_start_end(self):
        from processing.parallel_processor import ProcessingMetrics

        m = ProcessingMetrics()
        m.start()
        assert m.start_time is not None
        m.end()
        assert m.end_time is not None
        assert m.end_time >= m.start_time

    def test_add_file_time_success(self):
        from processing.parallel_processor import ProcessingMetrics

        m = ProcessingMetrics()
        m.add_file_time("a.py", 0.5, True)
        assert m.processed_files == 1
        assert m.failed_files == 0
        assert len(m.processing_times) == 1

    def test_add_file_time_failure(self):
        from processing.parallel_processor import ProcessingMetrics

        m = ProcessingMetrics()
        m.add_file_time("b.py", 1.0, False)
        assert m.processed_files == 1
        assert m.failed_files == 1

    def test_get_report_empty(self):
        from processing.parallel_processor import ProcessingMetrics

        m = ProcessingMetrics()
        report = m.get_report()
        assert report["total_files"] == 0
        assert report["processed_files"] == 0
        assert report["success_rate"] == 0.0
        assert report["total_duration"] == 0.0
        assert report["avg_file_time"] == 0.0
        assert report["slowest_files"] == []

    def test_get_report_with_data(self):
        from processing.parallel_processor import ProcessingMetrics

        m = ProcessingMetrics()
        m.start()
        m.total_files = 3
        m.add_file_time("a.py", 0.1, True)
        m.add_file_time("b.py", 0.5, True)
        m.add_file_time("c.py", 0.3, False)
        m.end()

        report = m.get_report()
        assert report["total_files"] == 3
        assert report["processed_files"] == 3
        assert report["failed_files"] == 1
        assert report["total_duration"] > 0
        assert report["avg_file_time"] > 0
        # Success rate: (3 - 1) / 3 * 100 = 66.67
        assert abs(report["success_rate"] - 66.666) < 1.0
        # Slowest files sorted descending by duration
        assert report["slowest_files"][0]["file"] == "b.py"


class TestParallelProcessor:
    @pytest.mark.asyncio
    async def test_find_files_empty_dir(self, tmp_path):
        with (
            patch(
                "processing.parallel_processor.get_settings",
                return_value=MagicMock(),
            ),
            patch("processing.parallel_processor.FileService"),
            patch("processing.parallel_processor.ProcessorFactory") as mock_pf,
        ):
            mock_pf_instance = MagicMock()
            mock_pf_instance.get_extension_language_map.return_value = {".py": "python"}
            mock_pf.return_value = mock_pf_instance
            from processing.parallel_processor import ParallelProcessor

            pp = ParallelProcessor(max_workers=2)
            pp.processor_factory = mock_pf_instance
            files = await pp._find_files(str(tmp_path))
            assert files == []

    @pytest.mark.asyncio
    async def test_find_files_with_matching(self, tmp_path):
        (tmp_path / "test.py").write_text("x=1")
        (tmp_path / "readme.txt").write_text("hi")
        with (
            patch(
                "processing.parallel_processor.get_settings",
                return_value=MagicMock(),
            ),
            patch("processing.parallel_processor.FileService"),
            patch("processing.parallel_processor.ProcessorFactory") as mock_pf,
        ):
            mock_pf_instance = MagicMock()
            mock_pf_instance.get_extension_language_map.return_value = {".py": "python"}
            mock_pf.return_value = mock_pf_instance
            from processing.parallel_processor import ParallelProcessor

            pp = ParallelProcessor(max_workers=2)
            pp.processor_factory = mock_pf_instance
            files = await pp._find_files(str(tmp_path))
            assert len(files) == 1
            assert files[0].endswith("test.py")

    @pytest.mark.asyncio
    async def test_find_files_language_filter(self, tmp_path):
        (tmp_path / "test.py").write_text("x=1")
        (tmp_path / "test.js").write_text("x=1")
        with (
            patch(
                "processing.parallel_processor.get_settings",
                return_value=MagicMock(),
            ),
            patch("processing.parallel_processor.FileService"),
            patch("processing.parallel_processor.ProcessorFactory") as mock_pf,
        ):
            mock_pf_instance = MagicMock()
            mock_pf_instance.get_extension_language_map.return_value = {
                ".py": "python",
                ".js": "javascript",
            }
            mock_pf.return_value = mock_pf_instance
            from processing.parallel_processor import ParallelProcessor

            pp = ParallelProcessor(max_workers=2)
            pp.processor_factory = mock_pf_instance
            files = await pp._find_files(str(tmp_path), languages=["python"])
            assert len(files) == 1
            assert files[0].endswith("test.py")

    @pytest.mark.asyncio
    async def test_process_file_with_metrics_success(self):
        with (
            patch(
                "processing.parallel_processor.get_settings",
                return_value=MagicMock(),
            ),
            patch("processing.parallel_processor.FileService"),
            patch("processing.parallel_processor.ProcessorFactory") as mock_pf,
        ):
            mock_pf_instance = MagicMock()
            mock_processor = AsyncMock()
            mock_processor.process_document = AsyncMock(
                return_value={"entities": [], "relations": []}
            )
            mock_pf_instance.get_processor_for_file.return_value = mock_processor
            mock_pf.return_value = mock_pf_instance
            from processing.parallel_processor import ParallelProcessor

            pp = ParallelProcessor(max_workers=2)
            pp.processor_factory = mock_pf_instance
            result = await pp._process_file_with_metrics("/test.py")
            assert result is not None
            assert pp.metrics.processed_files == 1

    @pytest.mark.asyncio
    async def test_process_file_with_metrics_no_processor(self):
        with (
            patch(
                "processing.parallel_processor.get_settings",
                return_value=MagicMock(),
            ),
            patch("processing.parallel_processor.FileService"),
            patch("processing.parallel_processor.ProcessorFactory") as mock_pf,
        ):
            mock_pf_instance = MagicMock()
            mock_pf_instance.get_processor_for_file.return_value = None
            mock_pf.return_value = mock_pf_instance
            from processing.parallel_processor import ParallelProcessor

            pp = ParallelProcessor(max_workers=2)
            pp.processor_factory = mock_pf_instance
            result = await pp._process_file_with_metrics("/test.py")
            assert result is None

    @pytest.mark.asyncio
    async def test_process_file_with_metrics_error(self):
        with (
            patch(
                "processing.parallel_processor.get_settings",
                return_value=MagicMock(),
            ),
            patch("processing.parallel_processor.FileService"),
            patch("processing.parallel_processor.ProcessorFactory") as mock_pf,
        ):
            mock_pf_instance = MagicMock()
            mock_pf_instance.get_processor_for_file.side_effect = RuntimeError("err")
            mock_pf.return_value = mock_pf_instance
            from processing.parallel_processor import ParallelProcessor

            pp = ParallelProcessor(max_workers=2)
            pp.processor_factory = mock_pf_instance
            result = await pp._process_file_with_metrics("/test.py")
            assert result is not None
            assert "error" in result
            assert pp.metrics.failed_files == 1

    @pytest.mark.asyncio
    async def test_process_directory(self, tmp_path):
        (tmp_path / "test.py").write_text("x=1")
        with (
            patch(
                "processing.parallel_processor.get_settings",
                return_value=MagicMock(),
            ),
            patch("processing.parallel_processor.FileService"),
            patch("processing.parallel_processor.ProcessorFactory") as mock_pf,
        ):
            mock_pf_instance = MagicMock()
            mock_pf_instance.get_extension_language_map.return_value = {".py": "python"}
            mock_processor = AsyncMock()
            mock_processor.process_document = AsyncMock(
                return_value={"entities": [], "relations": []}
            )
            mock_pf_instance.get_processor_for_file.return_value = mock_processor
            mock_pf.return_value = mock_pf_instance
            from processing.parallel_processor import ParallelProcessor

            pp = ParallelProcessor(max_workers=2)
            pp.processor_factory = mock_pf_instance
            results, metrics = await pp.process_directory(str(tmp_path))
            assert len(results) == 1
            assert metrics.total_files == 1
