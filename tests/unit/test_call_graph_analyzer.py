"""Unit tests for processing.analyzers.call_graph_analyzer module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.static_analysis_models import CallGraphMetrics, CallGraphResponse, OutputFormat
from processing.analyzers.call_graph_analyzer import (
    CallGraphConfig,
    CallGraphGenerator,
)

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from processing.quality_analysis.tool_definitions import QualityTool, ToolCategory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str = "pycg", command: list[str] | None = None) -> QualityTool:
    """Create a minimal QualityTool for testing."""
    return QualityTool(
        name=name,
        category=ToolCategory.LINTER,
        command=command or [name],
        output_format="json",
        description=f"{name} tool",
        timeout=30,
    )


def _make_generator(config: CallGraphConfig | None = None) -> CallGraphGenerator:
    """Create a CallGraphGenerator with a mocked tool_registry."""
    with patch("processing.analyzers.call_graph_analyzer.get_tool_registry") as mock_get_reg:
        mock_registry = MagicMock()
        mock_get_reg.return_value = mock_registry
        gen = CallGraphGenerator(config)
    # Expose the mock registry so tests can configure it
    gen.tool_registry = mock_registry
    return gen


# ---------------------------------------------------------------------------
# CallGraphConfig tests
# ---------------------------------------------------------------------------


class TestCallGraphConfig:
    def test_default_config(self):
        cfg = CallGraphConfig()
        assert cfg.output_format == OutputFormat.JSON
        assert cfg.include_external is False
        assert cfg.max_depth == 10
        assert cfg.analysis_scope == ["."]

    def test_custom_config(self):
        cfg = CallGraphConfig(
            output_format=OutputFormat.SVG,
            include_external=True,
            max_depth=5,
            analysis_scope=["lib/"],
        )
        assert cfg.output_format == OutputFormat.SVG
        assert cfg.include_external is True
        assert cfg.max_depth == 5
        assert cfg.analysis_scope == ["lib/"]

    def test_none_scope_gets_default(self):
        cfg = CallGraphConfig(analysis_scope=None)
        assert cfg.analysis_scope == ["."]


# ---------------------------------------------------------------------------
# CallGraphGenerator basic method tests
# ---------------------------------------------------------------------------


class TestCallGraphGeneratorBasics:
    def test_get_supported_extensions(self):
        gen = _make_generator()
        assert gen.get_supported_extensions() == [".py"]

    def test_get_language_name(self):
        gen = _make_generator()
        assert gen.get_language_name() == "python"

    def test_default_config_created(self):
        gen = _make_generator()
        assert isinstance(gen.config, CallGraphConfig)

    def test_custom_config_used(self):
        cfg = CallGraphConfig(max_depth=3)
        gen = _make_generator(cfg)
        assert gen.config.max_depth == 3

    def test_supported_tools(self):
        gen = _make_generator()
        assert gen.supported_tools == ["pyan3", "code2flow", "pycg"]


# ---------------------------------------------------------------------------
# _parse_dot_output tests
# ---------------------------------------------------------------------------


class TestParseDotOutput:
    def test_parse_nodes(self):
        gen = _make_generator()
        dot = """digraph G {
  "mod.func" [label="func"];
  "mod2.helper" [label="helper"];
}"""
        result = gen._parse_dot_output(dot, "test_tool")
        assert result["tool"] == "test_tool"
        assert len(result["nodes"]) == 2
        assert result["nodes"][0]["function"] == "func"
        assert result["nodes"][0]["module"] == "mod"

    def test_parse_edges_with_attributes(self):
        gen = _make_generator()
        # Parser uses split("[") to extract "to" node, so edges need [attrs]
        dot = '"a.foo" -> "b.bar" [style=solid];\n"b.bar" -> "c.baz" [style=dashed];'
        result = gen._parse_dot_output(dot, "pyan3")
        assert len(result["edges"]) == 2
        assert result["edges"][0]["from"] == "a.foo"
        assert result["edges"][0]["to"] == "b.bar"
        assert result["edges"][0]["call_type"] == "direct"

    def test_parse_edges_without_attributes(self):
        gen = _make_generator()
        # Without [...] the "to" value may keep trailing chars
        dot = '"x" -> "y";'
        result = gen._parse_dot_output(dot, "pyan3")
        assert len(result["edges"]) == 1
        assert result["edges"][0]["from"] == "x"
        # Parser does split("[")[0].strip().strip('"'), semicolon remains
        assert result["edges"][0]["to"] == 'y";'

    def test_parse_empty(self):
        gen = _make_generator()
        result = gen._parse_dot_output("", "tool")
        assert result["nodes"] == []
        assert result["edges"] == []

    def test_digraph_not_treated_as_node(self):
        gen = _make_generator()
        dot = "digraph [style=filled];"
        result = gen._parse_dot_output(dot, "t")
        # "digraph" should be filtered out
        assert all(n["function"] != "digraph" for n in result["nodes"])

    def test_node_without_dot_gets_empty_module(self):
        gen = _make_generator()
        dot = '"simple_func" [label="simple"];'
        result = gen._parse_dot_output(dot, "t")
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["function"] == "simple_func"
        assert result["nodes"][0]["module"] == ""


# ---------------------------------------------------------------------------
# _parse_pyan3_output tests
# ---------------------------------------------------------------------------


class TestParsePyan3Output:
    def test_delegates_to_parse_dot(self):
        gen = _make_generator()
        dot = '"a.x" -> "b.y";'
        result = gen._parse_pyan3_output(dot)
        assert result["tool"] == "pyan3"
        assert len(result["edges"]) == 1


# ---------------------------------------------------------------------------
# _parse_pycg_output tests
# ---------------------------------------------------------------------------


class TestParsePycgOutput:
    def test_basic_parsing(self):
        gen = _make_generator()
        pycg_data = {
            "module.main": ["module.helper", "module.utils"],
            "module.helper": [],
        }
        result = gen._parse_pycg_output(pycg_data)
        assert result["tool"] == "pycg"
        assert len(result["edges"]) == 2
        assert any(
            e["from"] == "module.main" and e["to"] == "module.helper" for e in result["edges"]
        )

    def test_empty_data(self):
        gen = _make_generator()
        result = gen._parse_pycg_output({})
        assert result["nodes"] == []
        assert result["edges"] == []

    def test_non_dict_data(self):
        gen = _make_generator()
        # Passing a list should result in empty output (isinstance check fails)
        result = gen._parse_pycg_output([])
        assert result["nodes"] == []
        assert result["edges"] == []

    def test_nodes_extracted_from_callers_and_callees(self):
        gen = _make_generator()
        pycg_data = {"a.foo": ["b.bar"]}
        result = gen._parse_pycg_output(pycg_data)
        functions = [n["function"] for n in result["nodes"]]
        assert "foo" in functions
        assert "bar" in functions

    def test_caller_without_dot(self):
        gen = _make_generator()
        pycg_data = {"main": ["helper"]}
        result = gen._parse_pycg_output(pycg_data)
        assert result["nodes"][0]["function"] == "main"
        assert result["nodes"][0]["module"] == ""


# ---------------------------------------------------------------------------
# _merge_call_graph_results tests
# ---------------------------------------------------------------------------


class TestMergeCallGraphResults:
    def test_merge_unique_nodes(self):
        gen = _make_generator()
        r1 = {"nodes": [{"module": "a", "function": "f1"}], "edges": []}
        r2 = {"nodes": [{"module": "b", "function": "f2"}], "edges": []}
        r3 = {"nodes": [], "edges": []}
        merged = gen._merge_call_graph_results(r1, r2, r3)
        assert len(merged["nodes"]) == 2
        assert "pyan3" in merged["tools_used"]

    def test_merge_duplicate_nodes_updates(self):
        gen = _make_generator()
        r1 = {"nodes": [{"module": "a", "function": "f1", "complexity": 1}], "edges": []}
        r2 = {"nodes": [{"module": "a", "function": "f1", "complexity": 5}], "edges": []}
        r3 = {"nodes": [], "edges": []}
        merged = gen._merge_call_graph_results(r1, r2, r3)
        assert len(merged["nodes"]) == 1
        assert merged["nodes"][0]["complexity"] == 5

    def test_merge_unique_edges(self):
        gen = _make_generator()
        r1 = {"nodes": [], "edges": [{"from": "a", "to": "b"}]}
        r2 = {"nodes": [], "edges": [{"from": "c", "to": "d"}]}
        r3 = {"nodes": [], "edges": []}
        merged = gen._merge_call_graph_results(r1, r2, r3)
        assert len(merged["edges"]) == 2

    def test_merge_duplicate_edges_deduplicated(self):
        gen = _make_generator()
        edge = {"from": "a", "to": "b"}
        r1 = {"nodes": [], "edges": [edge]}
        r2 = {"nodes": [], "edges": [edge.copy()]}
        r3 = {"nodes": [], "edges": []}
        merged = gen._merge_call_graph_results(r1, r2, r3)
        assert len(merged["edges"]) == 1

    def test_merge_has_timestamp(self):
        gen = _make_generator()
        r = {"nodes": [], "edges": []}
        merged = gen._merge_call_graph_results(r, r, r)
        assert "merge_timestamp" in merged


# ---------------------------------------------------------------------------
# _convert_to_dot tests
# ---------------------------------------------------------------------------


class TestConvertToDot:
    def test_basic_dot_output(self):
        gen = _make_generator()
        graph = {
            "nodes": [
                {"function": "main", "module": "app"},
                {"function": "helper", "module": ""},
            ],
            "edges": [{"from": "app.main", "to": "helper"}],
        }
        dot = gen._convert_to_dot(graph)
        assert "digraph CallGraph" in dot
        assert "rankdir=LR" in dot
        assert '"app.main"' in dot
        assert '"helper"' in dot
        assert '"app.main" -> "helper"' in dot

    def test_empty_graph(self):
        gen = _make_generator()
        dot = gen._convert_to_dot({"nodes": [], "edges": []})
        assert "digraph CallGraph" in dot
        assert "->" not in dot


# ---------------------------------------------------------------------------
# _identify_critical_paths tests
# ---------------------------------------------------------------------------


class TestIdentifyCriticalPaths:
    def test_empty_edges_returns_empty(self):
        gen = _make_generator()
        result = gen._identify_critical_paths({"edges": []})
        assert result == []

    def test_finds_long_paths(self):
        gen = _make_generator()
        # Build a chain: a -> b -> c -> d -> e
        edges = [
            {"from": "a", "to": "b"},
            {"from": "b", "to": "c"},
            {"from": "c", "to": "d"},
            {"from": "d", "to": "e"},
        ]
        graph = {"edges": edges}
        paths = gen._identify_critical_paths(graph)
        # Should find paths with length > 3
        assert len(paths) > 0
        assert all(" -> " in p for p in paths)

    def test_limited_to_10_paths(self):
        gen = _make_generator()
        # Create a wide graph that generates many paths
        edges = []
        for i in range(10):
            for j in range(10):
                edges.append({"from": f"n{i}", "to": f"n{j}"})
        graph = {"edges": edges}
        paths = gen._identify_critical_paths(graph)
        assert len(paths) <= 10


# ---------------------------------------------------------------------------
# _find_orphaned_functions tests
# ---------------------------------------------------------------------------


class TestFindOrphanedFunctions:
    def test_no_edges_all_orphaned(self):
        gen = _make_generator()
        graph = {
            "nodes": [
                {"function": "f1", "module": "m"},
                {"function": "f2", "module": ""},
            ],
            "edges": [],
        }
        orphaned = gen._find_orphaned_functions(graph)
        assert "m.f1" in orphaned
        assert "f2" in orphaned

    def test_called_functions_not_orphaned(self):
        gen = _make_generator()
        graph = {
            "nodes": [
                {"function": "caller", "module": "a"},
                {"function": "callee", "module": "b"},
            ],
            "edges": [{"from": "a.caller", "to": "b.callee"}],
        }
        orphaned = gen._find_orphaned_functions(graph)
        assert "b.callee" not in orphaned
        # caller itself is not called by anything, so it should be orphaned
        assert "a.caller" in orphaned

    def test_empty_graph(self):
        gen = _make_generator()
        graph = {"nodes": [], "edges": []}
        assert gen._find_orphaned_functions(graph) == []


# ---------------------------------------------------------------------------
# _calculate_call_graph_metrics tests
# ---------------------------------------------------------------------------


class TestCalculateCallGraphMetrics:
    def test_empty_graph(self):
        gen = _make_generator()
        metrics = gen._calculate_call_graph_metrics({"nodes": [], "edges": []})
        assert isinstance(metrics, CallGraphMetrics)
        assert metrics.total_functions == 0
        assert metrics.max_call_depth == 0
        assert metrics.cyclic_dependencies == []

    def test_with_nodes_and_edges(self):
        gen = _make_generator()
        graph = {
            "nodes": [{"function": "a"}, {"function": "b"}, {"function": "c"}],
            "edges": [
                {"from": "a", "to": "b"},
                {"from": "b", "to": "c"},
            ],
        }
        metrics = gen._calculate_call_graph_metrics(graph)
        assert metrics.total_functions == 3
        assert metrics.max_call_depth > 0


# ---------------------------------------------------------------------------
# _detect_cycles tests
# ---------------------------------------------------------------------------


class TestDetectCycles:
    def test_no_cycles(self):
        gen = _make_generator()
        adj = {"a": ["b"], "b": ["c"], "c": []}
        cycles = gen._detect_cycles(adj)
        assert cycles == []

    def test_simple_cycle(self):
        gen = _make_generator()
        adj = {"a": ["b"], "b": ["a"]}
        cycles = gen._detect_cycles(adj)
        assert len(cycles) >= 1
        # The cycle should mention both a and b
        assert any("a" in c and "b" in c for c in cycles)

    def test_self_loop(self):
        gen = _make_generator()
        adj = {"a": ["a"]}
        cycles = gen._detect_cycles(adj)
        assert len(cycles) >= 1

    def test_empty_adj_list(self):
        gen = _make_generator()
        assert gen._detect_cycles({}) == []


# ---------------------------------------------------------------------------
# _calculate_call_depth tests
# ---------------------------------------------------------------------------


class TestCalculateCallDepth:
    def test_linear_chain(self):
        gen = _make_generator()
        adj = {"a": ["b"], "b": ["c"], "c": []}
        depth = gen._calculate_call_depth(adj, "a")
        assert depth == 2

    def test_single_node(self):
        gen = _make_generator()
        adj = {"a": []}
        depth = gen._calculate_call_depth(adj, "a")
        assert depth == 0

    def test_handles_cycle(self):
        gen = _make_generator()
        adj = {"a": ["b"], "b": ["a"]}
        depth = gen._calculate_call_depth(adj, "a")
        # Should not infinite loop; depth should be reasonable
        assert depth >= 1


# ---------------------------------------------------------------------------
# _analyze_call_patterns tests
# ---------------------------------------------------------------------------


class TestAnalyzeCallPatterns:
    def test_empty_graph(self):
        gen = _make_generator()
        patterns = gen._analyze_call_patterns({"nodes": [], "edges": []})
        assert patterns["recursive_functions"] == []
        assert patterns["high_fan_out"] == []
        assert patterns["high_fan_in"] == []
        assert patterns["leaf_functions"] == []
        assert patterns["entry_points"] == []

    def test_recursive_function_detected(self):
        gen = _make_generator()
        graph = {
            "nodes": ["a"],
            "edges": [{"source": "a", "target": "a"}],
        }
        patterns = gen._analyze_call_patterns(graph)
        assert "a" in patterns["recursive_functions"]

    def test_high_fan_out(self):
        gen = _make_generator()
        edges = [{"source": "hub", "target": f"t{i}"} for i in range(7)]
        nodes = ["hub"] + [f"t{i}" for i in range(7)]
        patterns = gen._analyze_call_patterns({"nodes": nodes, "edges": edges})
        fan_out_names = [e["function"] for e in patterns["high_fan_out"]]
        assert "hub" in fan_out_names

    def test_leaf_functions(self):
        gen = _make_generator()
        graph = {
            "nodes": ["a", "b"],
            "edges": [{"source": "a", "target": "b"}],
        }
        patterns = gen._analyze_call_patterns(graph)
        assert "b" in patterns["leaf_functions"]

    def test_entry_points(self):
        gen = _make_generator()
        graph = {
            "nodes": ["a", "b"],
            "edges": [{"source": "a", "target": "b"}],
        }
        patterns = gen._analyze_call_patterns(graph)
        assert "a" in patterns["entry_points"]


# ---------------------------------------------------------------------------
# _find_python_files tests
# ---------------------------------------------------------------------------


class TestFindPythonFiles:
    def test_finds_py_files(self, tmp_path):
        gen = _make_generator()
        (tmp_path / "a.py").write_text("pass")
        (tmp_path / "b.txt").write_text("text")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").write_text("pass")
        files = gen._find_python_files(str(tmp_path))
        assert len(files) == 2
        assert all(f.endswith(".py") for f in files)

    def test_empty_dir(self, tmp_path):
        gen = _make_generator()
        assert gen._find_python_files(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# _find_python_entry_points tests
# ---------------------------------------------------------------------------


class TestFindPythonEntryPoints:
    def test_finds_main_files(self, tmp_path):
        gen = _make_generator()
        (tmp_path / "main.py").write_text("pass")
        (tmp_path / "app.py").write_text("pass")
        eps = gen._find_python_entry_points(str(tmp_path))
        assert any("main.py" in ep for ep in eps)
        assert any("app.py" in ep for ep in eps)

    def test_finds_init_files(self, tmp_path):
        gen = _make_generator()
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("pass")
        eps = gen._find_python_entry_points(str(tmp_path))
        assert any("__init__.py" in ep for ep in eps)

    def test_limited_to_10(self, tmp_path):
        gen = _make_generator()
        for i in range(15):
            pkg = tmp_path / f"pkg{i}"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("pass")
        eps = gen._find_python_entry_points(str(tmp_path))
        assert len(eps) <= 10


# ---------------------------------------------------------------------------
# _get_timestamp tests
# ---------------------------------------------------------------------------


class TestGetTimestamp:
    def test_returns_iso_string(self):
        gen = _make_generator()
        ts = gen._get_timestamp()
        assert isinstance(ts, str)
        assert "T" in ts  # ISO format has T separator


# ---------------------------------------------------------------------------
# analyze_file tests
# ---------------------------------------------------------------------------


class TestAnalyzeFile:
    async def test_success(self):
        gen = _make_generator()
        mock_data = {"nodes": [{"function": "f"}], "edges": []}
        with patch.object(
            gen, "generate_call_graph_for_file", new_callable=AsyncMock, return_value=mock_data
        ):
            result = await gen.analyze_file("/tmp/test.py")
        assert result.success is True
        assert result.language == "python"
        assert result.metadata["call_graph"] == mock_data
        assert result.metadata["analyzer"] == "call_graph"

    async def test_error_returns_failure(self):
        gen = _make_generator()
        with patch.object(
            gen,
            "generate_call_graph_for_file",
            new_callable=AsyncMock,
            side_effect=OSError("read error"),
        ):
            result = await gen.analyze_file("/tmp/bad.py")
        assert result.success is False
        assert "read error" in result.error


# ---------------------------------------------------------------------------
# analyze_directory tests
# ---------------------------------------------------------------------------


class TestAnalyzeDirectory:
    async def test_success(self):
        gen = _make_generator()
        mock_data = {"nodes": [], "edges": [], "tools_used": []}
        with patch.object(
            gen,
            "generate_call_graph_for_directory",
            new_callable=AsyncMock,
            return_value=mock_data,
        ):
            result = await gen.analyze_directory("/tmp/project")
        assert result.success is True
        assert result.metadata["call_graph"] == mock_data

    async def test_error_returns_failure(self):
        gen = _make_generator()
        with patch.object(
            gen,
            "generate_call_graph_for_directory",
            new_callable=AsyncMock,
            side_effect=OSError("dir error"),
        ):
            result = await gen.analyze_directory("/tmp/bad")
        assert result.success is False
        assert "dir error" in result.error


# ---------------------------------------------------------------------------
# analyze_snippet tests
# ---------------------------------------------------------------------------


class TestAnalyzeSnippet:
    async def test_success(self):
        gen = _make_generator()
        mock_data = {"nodes": [], "edges": []}
        with patch.object(
            gen, "generate_call_graph_for_file", new_callable=AsyncMock, return_value=mock_data
        ):
            result = await gen.analyze_snippet("def foo(): pass", "/orig.py")
        assert result.success is True
        assert result.metadata["snippet"] is True
        assert result.metadata["original_file"] == "/orig.py"

    async def test_no_file_path(self):
        gen = _make_generator()
        mock_data = {"nodes": [], "edges": []}
        with patch.object(
            gen, "generate_call_graph_for_file", new_callable=AsyncMock, return_value=mock_data
        ):
            result = await gen.analyze_snippet("def bar(): pass")
        assert result.success is True

    async def test_error_returns_failure(self):
        gen = _make_generator()
        with patch.object(
            gen,
            "generate_call_graph_for_file",
            new_callable=AsyncMock,
            side_effect=OSError("snippet error"),
        ):
            result = await gen.analyze_snippet("def bad(): pass", "/f.py")
        assert result.success is False
        assert "snippet error" in result.error


# ---------------------------------------------------------------------------
# _run_pycg_on_file tests
# ---------------------------------------------------------------------------


class TestRunPycgOnFile:
    async def test_tool_not_found(self):
        gen = _make_generator()
        gen.tool_registry.get_tool.return_value = None
        result = await gen._run_pycg_on_file("/tmp/test.py")
        assert result["error"] == "tool_not_found"

    async def test_success(self):
        gen = _make_generator()
        tool = _make_tool("pycg", ["pycg"])
        gen.tool_registry.get_tool.return_value = tool
        pycg_json = json.dumps({"main": ["helper"]})
        with patch("processing.analyzers.call_graph_analyzer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=pycg_json, stderr="")
            result = await gen._run_pycg_on_file("/tmp/test.py")
        assert result["tool"] == "pycg"
        assert len(result["edges"]) == 1

    async def test_non_zero_exit(self):
        gen = _make_generator()
        tool = _make_tool("pycg", ["pycg"])
        gen.tool_registry.get_tool.return_value = tool
        with patch("processing.analyzers.call_graph_analyzer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="fail")
            result = await gen._run_pycg_on_file("/tmp/test.py")
        assert result["error"] == "fail"

    async def test_json_parse_error(self):
        gen = _make_generator()
        tool = _make_tool("pycg", ["pycg"])
        gen.tool_registry.get_tool.return_value = tool
        with patch("processing.analyzers.call_graph_analyzer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="not json", stderr="")
            result = await gen._run_pycg_on_file("/tmp/test.py")
        assert result["error"] == "json_parse_error"

    async def test_oserror(self):
        gen = _make_generator()
        tool = _make_tool("pycg", ["pycg"])
        gen.tool_registry.get_tool.return_value = tool
        with patch(
            "processing.analyzers.call_graph_analyzer.subprocess.run",
            side_effect=OSError("no such file"),
        ):
            result = await gen._run_pycg_on_file("/tmp/test.py")
        assert "no such file" in result["error"]


# ---------------------------------------------------------------------------
# _run_pyan3_analysis tests
# ---------------------------------------------------------------------------


class TestRunPyan3Analysis:
    async def test_tool_not_found(self):
        gen = _make_generator()
        gen.tool_registry.get_tool.return_value = None
        cfg = CallGraphConfig(analysis_scope=["src/"])
        result = await gen._run_pyan3_analysis("/repo", cfg)
        assert result["status"] == "tool_not_found"
        assert result["tool"] == "pyan3"

    async def test_no_python_files(self, tmp_path):
        gen = _make_generator()
        tool = _make_tool("pyan3", ["pyan3", "--dot"])
        gen.tool_registry.get_tool.return_value = tool
        src = tmp_path / "src"
        src.mkdir()
        cfg = CallGraphConfig(analysis_scope=["src/"])
        result = await gen._run_pyan3_analysis(str(tmp_path), cfg)
        assert result["status"] == "no_files"

    async def test_success(self, tmp_path):
        gen = _make_generator()
        tool = _make_tool("pyan3", ["pyan3", "--dot"])
        gen.tool_registry.get_tool.return_value = tool
        src = tmp_path / "src"
        src.mkdir()
        (src / "mod.py").write_text("def f(): pass")
        cfg = CallGraphConfig(analysis_scope=["src/"])

        dot_output = '"mod.f" [label="f"];\n"mod.f" -> "mod.g";'
        with patch("processing.analyzers.call_graph_analyzer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=dot_output, stderr="")
            result = await gen._run_pyan3_analysis(str(tmp_path), cfg)

        assert len(result["nodes"]) >= 1
        assert len(result["edges"]) >= 1

    async def test_failure(self, tmp_path):
        gen = _make_generator()
        tool = _make_tool("pyan3", ["pyan3"])
        gen.tool_registry.get_tool.return_value = tool
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("pass")
        cfg = CallGraphConfig(analysis_scope=["src/"])

        with patch("processing.analyzers.call_graph_analyzer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
            result = await gen._run_pyan3_analysis(str(tmp_path), cfg)

        assert result["error"] == "error"

    async def test_oserror(self, tmp_path):
        gen = _make_generator()
        tool = _make_tool("pyan3", ["pyan3"])
        gen.tool_registry.get_tool.return_value = tool
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("pass")
        cfg = CallGraphConfig(analysis_scope=["src/"])

        with patch(
            "processing.analyzers.call_graph_analyzer.subprocess.run",
            side_effect=OSError("cmd not found"),
        ):
            result = await gen._run_pyan3_analysis(str(tmp_path), cfg)

        assert "cmd not found" in result["error"]


# ---------------------------------------------------------------------------
# _run_pycg_analysis tests
# ---------------------------------------------------------------------------


class TestRunPycgAnalysis:
    async def test_tool_not_found(self):
        gen = _make_generator()
        gen.tool_registry.get_tool.return_value = None
        cfg = CallGraphConfig()
        result = await gen._run_pycg_analysis("/repo", cfg)
        assert result["status"] == "tool_not_found"

    async def test_no_entry_points(self, tmp_path):
        gen = _make_generator()
        tool = _make_tool("pycg", ["pycg"])
        gen.tool_registry.get_tool.return_value = tool
        src = tmp_path / "src"
        src.mkdir()
        # No main.py, app.py, __init__.py etc.
        (src / "util.py").write_text("pass")
        cfg = CallGraphConfig(analysis_scope=["src/"])
        result = await gen._run_pycg_analysis(str(tmp_path), cfg)
        assert result["status"] == "no_entry_points"

    async def test_success(self, tmp_path):
        gen = _make_generator()
        tool = _make_tool("pycg", ["pycg"])
        gen.tool_registry.get_tool.return_value = tool
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("pass")
        cfg = CallGraphConfig(analysis_scope=["src/"])
        pycg_json = json.dumps({"main.run": ["main.helper"]})

        with patch("processing.analyzers.call_graph_analyzer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=pycg_json, stderr="")
            result = await gen._run_pycg_analysis(str(tmp_path), cfg)

        assert result["tool"] == "pycg"
        assert len(result["edges"]) == 1

    async def test_json_parse_error(self, tmp_path):
        gen = _make_generator()
        tool = _make_tool("pycg", ["pycg"])
        gen.tool_registry.get_tool.return_value = tool
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("pass")
        cfg = CallGraphConfig(analysis_scope=["src/"])

        with patch("processing.analyzers.call_graph_analyzer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="bad json", stderr="")
            result = await gen._run_pycg_analysis(str(tmp_path), cfg)

        assert result["error"] == "json_parse_error"

    async def test_non_zero_exit(self, tmp_path):
        gen = _make_generator()
        tool = _make_tool("pycg", ["pycg"])
        gen.tool_registry.get_tool.return_value = tool
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("pass")
        cfg = CallGraphConfig(analysis_scope=["src/"])

        with patch("processing.analyzers.call_graph_analyzer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="pycg fail")
            result = await gen._run_pycg_analysis(str(tmp_path), cfg)

        assert result["error"] == "pycg fail"


# ---------------------------------------------------------------------------
# _run_code2flow_analysis tests
# ---------------------------------------------------------------------------


class TestRunCode2flowAnalysis:
    async def test_tool_not_found(self):
        gen = _make_generator()
        gen.tool_registry.get_tool.return_value = None
        cfg = CallGraphConfig()
        result = await gen._run_code2flow_analysis("/repo", cfg)
        assert result["status"] == "tool_not_found"
        assert result["tool"] == "code2flow"

    async def test_success(self, tmp_path):
        gen = _make_generator()
        tool = _make_tool("code2flow", ["code2flow"])
        gen.tool_registry.get_tool.return_value = tool
        src = tmp_path / "src"
        src.mkdir()
        cfg = CallGraphConfig(analysis_scope=["src/"])

        dot_content = '"a.x" -> "b.y";'

        with (
            patch("processing.analyzers.call_graph_analyzer.subprocess.run") as mock_run,
            patch(
                "processing.analyzers.call_graph_analyzer.os.path.exists",
                return_value=True,
            ),
            patch(
                "builtins.open",
                MagicMock(
                    return_value=MagicMock(
                        __enter__=MagicMock(
                            return_value=MagicMock(read=MagicMock(return_value=dot_content))
                        ),
                        __exit__=MagicMock(return_value=False),
                    )
                ),
            ),
            patch("processing.analyzers.call_graph_analyzer.os.unlink"),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = await gen._run_code2flow_analysis(str(tmp_path), cfg)

        assert result["tool"] == "code2flow"

    async def test_oserror(self):
        gen = _make_generator()
        tool = _make_tool("code2flow", ["code2flow"])
        gen.tool_registry.get_tool.return_value = tool
        cfg = CallGraphConfig(analysis_scope=[])

        with patch(
            "processing.analyzers.call_graph_analyzer.subprocess.run",
            side_effect=OSError("missing"),
        ):
            result = await gen._run_code2flow_analysis("/repo", cfg)

        assert "missing" in result["error"]


# ---------------------------------------------------------------------------
# _generate_visual_output tests
# ---------------------------------------------------------------------------


class TestGenerateVisualOutput:
    async def test_dot_format(self):
        gen = _make_generator()
        graph = {"nodes": [], "edges": []}
        result = await gen._generate_visual_output(graph, OutputFormat.DOT)
        assert result is not None
        # Should be base64 encoded
        import base64

        decoded = base64.b64decode(result).decode()
        assert "digraph" in decoded

    async def test_svg_format_success(self):
        gen = _make_generator()
        graph = {"nodes": [], "edges": []}
        with patch("processing.analyzers.call_graph_analyzer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="<svg></svg>", stderr="")
            result = await gen._generate_visual_output(graph, OutputFormat.SVG)
        assert result is not None

    async def test_svg_format_failure(self):
        gen = _make_generator()
        graph = {"nodes": [], "edges": []}
        with patch("processing.analyzers.call_graph_analyzer.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="graphviz error")
            result = await gen._generate_visual_output(graph, OutputFormat.SVG)
        assert result is None

    async def test_json_format_returns_none(self):
        gen = _make_generator()
        graph = {"nodes": [], "edges": []}
        result = await gen._generate_visual_output(graph, OutputFormat.JSON)
        assert result is None

    async def test_oserror_returns_none(self):
        gen = _make_generator()
        graph = {"nodes": [], "edges": []}
        with patch(
            "processing.analyzers.call_graph_analyzer.subprocess.run",
            side_effect=OSError("no graphviz"),
        ):
            result = await gen._generate_visual_output(graph, OutputFormat.SVG)
        assert result is None


# ---------------------------------------------------------------------------
# generate_comprehensive_call_graph tests
# ---------------------------------------------------------------------------


class TestGenerateComprehensiveCallGraph:
    # The registry is a MagicMock (no real executable), so pyan3/code2flow/pycg
    # are reported unavailable and only the in-process ast backend runs. The
    # old versions of these tests mocked the three tool runners against a
    # non-existent "/repo" and accepted an empty graph as success.

    @staticmethod
    def _repo(tmp_path):
        (tmp_path / "mod.py").write_text("def a():\n    b()\n\n\ndef b():\n    pass\n")
        return str(tmp_path)

    async def test_success_json_format(self, tmp_path):
        gen = _make_generator(CallGraphConfig(analysis_scope=["."]))
        response = await gen.generate_comprehensive_call_graph(self._repo(tmp_path))

        assert isinstance(response, CallGraphResponse)
        assert response.analysis_id.startswith("cg_")
        assert response.status == "computed"
        assert response.tools_used == ["python-ast"]
        assert response.tools_unavailable == ["pyan3", "code2flow", "pycg"]
        assert isinstance(response.metrics, CallGraphMetrics)
        assert response.metrics.total_functions == 2
        assert response.visual_output is None  # JSON format, no visual

    async def test_success_svg_format(self, tmp_path):
        gen = _make_generator(CallGraphConfig(output_format=OutputFormat.SVG, analysis_scope=["."]))
        with patch.object(
            gen, "_generate_visual_output", new_callable=AsyncMock, return_value="base64svg"
        ):
            response = await gen.generate_comprehensive_call_graph(self._repo(tmp_path))

        assert response.visual_output == "base64svg"

    async def test_error_returns_error_response(self, tmp_path):
        gen = _make_generator(CallGraphConfig(analysis_scope=["."]))

        with patch.object(gen, "_build_ast_call_graph", side_effect=OSError("analysis failed")):
            response = await gen.generate_comprehensive_call_graph(self._repo(tmp_path))

        assert response.status == "not_computed"
        assert "analysis failed" in (response.not_computed_reason or "")
        assert "error" in response.call_graph
        assert response.metrics.total_functions == 0


# ---------------------------------------------------------------------------
# generate_call_graph_for_file tests
# ---------------------------------------------------------------------------


class TestGenerateCallGraphForFile:
    async def test_success(self):
        gen = _make_generator()
        mock_data = {"nodes": [{"function": "f"}], "edges": []}
        with patch.object(gen, "_run_pycg_on_file", new_callable=AsyncMock, return_value=mock_data):
            result = await gen.generate_call_graph_for_file("/tmp/test.py")
        assert result == mock_data

    async def test_error(self):
        gen = _make_generator()
        with patch.object(
            gen,
            "_run_pycg_on_file",
            new_callable=AsyncMock,
            side_effect=OSError("fail"),
        ):
            result = await gen.generate_call_graph_for_file("/tmp/test.py")
        assert "error" in result


# ---------------------------------------------------------------------------
# generate_call_graph_for_directory tests
# ---------------------------------------------------------------------------


class TestGenerateCallGraphForDirectory:
    async def test_success(self):
        gen = _make_generator()
        mock_response = CallGraphResponse(
            analysis_id="cg_test",
            call_graph={"nodes": [], "edges": []},
            visual_output=None,
            critical_paths=[],
            orphaned_functions=[],
            metrics=CallGraphMetrics(total_functions=0, max_call_depth=0, cyclic_dependencies=[]),
        )
        with patch.object(
            gen,
            "generate_comprehensive_call_graph",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await gen.generate_call_graph_for_directory("/tmp/project")
        assert result == {"nodes": [], "edges": []}

    async def test_error(self):
        gen = _make_generator()
        with patch.object(
            gen,
            "generate_comprehensive_call_graph",
            new_callable=AsyncMock,
            side_effect=OSError("dir error"),
        ):
            result = await gen.generate_call_graph_for_directory("/tmp/project")
        assert "error" in result


# ---------------------------------------------------------------------------
# CallGraphService tests (services/code_analysis/call_graph_service.py)
# Covers lines: 139, 155-158, 171-174, 182-184
# ---------------------------------------------------------------------------

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

from services.code_analysis.call_graph_service import CallGraphService


class TestCallGraphServiceDeep:
    """Deep coverage tests for CallGraphService."""

    @pytest.fixture
    def mock_analyzer(self):
        return MagicMock()

    @pytest.fixture
    def service(self, mock_analyzer):
        return CallGraphService(mock_analyzer)

    async def test_get_function_calls_returns_dict(self, service, mock_analyzer):
        """When analyzer returns dict, it's returned directly (line 139)."""
        mock_analyzer.find_function_calls.return_value = {
            "function_name": "test",
            "calls": ["a", "b"],
            "custom_key": True,
        }
        result = await service.get_function_calls("cg1", "test")
        assert result["custom_key"] is True

    async def test_find_class_references_returns_dict(self, service, mock_analyzer):
        """When analyzer returns dict, it's returned directly (line 155)."""
        mock_analyzer.find_class_references.return_value = {
            "class_name": "Foo",
            "references": ["ref1"],
            "extra": True,
        }
        result = await service.find_class_references("cg1", "Foo")
        assert result["extra"] is True

    async def test_find_class_references_error(self, service, mock_analyzer):
        """find_class_references error path (lines 156-158)."""
        mock_analyzer.find_class_references.side_effect = TypeError("type error")
        result = await service.find_class_references("cg1", "Missing")
        assert "error" in result
        assert result["references"] == []

    async def test_find_import_usage_returns_dict(self, service, mock_analyzer):
        """When analyzer returns dict, it's returned directly (line 171)."""
        mock_analyzer.find_import_usage.return_value = {
            "module_name": "os",
            "usage": ["u1"],
            "custom": True,
        }
        result = await service.find_import_usage("cg1", "os")
        assert result["custom"] is True

    async def test_find_import_usage_error(self, service, mock_analyzer):
        """find_import_usage error path (lines 172-174)."""
        mock_analyzer.find_import_usage.side_effect = AttributeError("no attr")
        result = await service.find_import_usage("cg1", "os")
        assert "error" in result
        assert result["usage"] == []

    async def test_get_dependency_graph_error(self, service, mock_analyzer):
        """get_dependency_graph error path (lines 182-184)."""
        mock_analyzer.analyze_dependency_graph.side_effect = KeyError("not found")
        result = await service.get_dependency_graph("cg1", "node1")
        assert "error" in result
        assert result["dependencies"] == []

    async def test_build_call_graph_empty_analyzer_output(self, service, mock_analyzer):
        """build_call_graph with empty graph data."""
        mock_analyzer.analyze_call_graph.return_value = {
            "nodes": [],
            "edges": [],
        }
        result = await service.build_call_graph("main", depth=3, direction="incoming")
        assert result.root == "main"
        assert len(result.nodes) == 0
        assert len(result.edges) == 0
        assert result.depth == 3
        assert result.direction == "incoming"

    async def test_find_import_usage_list_return(self, service, mock_analyzer):
        """find_import_usage wraps list return (lines 165-170)."""
        mock_analyzer.find_import_usage.return_value = ["usage1", "usage2"]
        result = await service.find_import_usage("cg1", "os")
        assert result["module_name"] == "os"
        assert result["total_usage"] == 2
