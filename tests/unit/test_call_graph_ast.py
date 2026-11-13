"""The in-process ``ast`` call-graph backend and honest status reporting.

Regression for the live defect where /static/call-graph returned 200 with an
empty graph and tools_used=[pyan3, code2flow, pycg] although none of those
tools was installed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from processing.analyzers.call_graph_analyzer import CallGraphConfig, CallGraphGenerator
from processing.analyzers.call_graph_metrics import analyse_graph
from processing.analyzers.python_call_graph import build_python_call_graph

MOD_A = """\
from pkg.b import g
import pkg.b as bmod


def f():
    g()
    bmod.h()
    nope()
    print("builtins are not unresolved")


class C:
    def m(self):
        self.n()
        self.ghost()

    def n(self):
        return helper()


def helper():
    return C()


def main():
    f()
"""

MOD_B = """\
def g():
    return h()


def h():
    return g()
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "a.py").write_text(MOD_A)
    (pkg / "b.py").write_text(MOD_B)
    return tmp_path


def _edges(graph) -> set[tuple[str, str]]:
    return {(e["from"], e["to"]) for e in graph.edges}


class TestAstBackend:
    def test_import_and_self_edges(self, repo: Path):
        graph = build_python_call_graph(str(repo), [str(repo)])
        edges = _edges(graph)
        assert ("pkg.a.f", "pkg.b.g") in edges  # from x import y
        assert ("pkg.a.f", "pkg.b.h") in edges  # import x as y; y.h()
        assert ("pkg.a.C.m", "pkg.a.C.n") in edges  # self.method
        assert ("pkg.a.C.n", "pkg.a.helper") in edges  # same module
        assert ("pkg.a.main", "pkg.a.f") in edges

    def test_unresolved_calls_are_counted_not_edged(self, repo: Path):
        graph = build_python_call_graph(str(repo), [str(repo)])
        targets = {to for _, to in _edges(graph)}
        assert "nope" not in targets
        assert not any("ghost" in t for t in targets)
        # nope() + self.ghost(); print() is a builtin and not counted
        assert graph.unresolved_calls == 2

    def test_nodes_are_qualified_with_file_and_line(self, repo: Path):
        graph = build_python_call_graph(str(repo), [str(repo)])
        by_id = {n["id"]: n for n in graph.nodes}
        assert by_id["pkg.a.C.m"]["type"] == "method"
        assert by_id["pkg.a.C.m"]["file"] == "pkg/a.py"
        assert by_id["pkg.a.f"]["line_number"] == 5
        assert by_id["pkg.a.C.m"]["module"] + "." + by_id["pkg.a.C.m"]["function"] == "pkg.a.C.m"

    def test_include_external_lists_unresolved_as_external_nodes(self, repo: Path):
        graph = build_python_call_graph(str(repo), [str(repo)], include_external=True)
        assert any(n["id"] == "nope" for n in graph.external_nodes)
        assert ("pkg.a.f", "nope") in _edges(graph)


class TestGenerator:
    async def test_computed_response_with_metrics(self, repo: Path):
        gen = CallGraphGenerator(CallGraphConfig(analysis_scope=["."]))
        resp = await gen.generate_comprehensive_call_graph(str(repo))
        assert resp.status == "computed"
        assert resp.tools_used[0] == "python-ast"
        # Only tools that are actually installed may be reported as used.
        for tool in ("pyan3", "code2flow", "pycg"):
            assert (tool in resp.tools_used) != (tool in resp.tools_unavailable)
        assert resp.metrics.total_functions == 7
        assert resp.metrics.unresolved_calls == 2
        assert resp.metrics.total_edges == len(resp.call_graph["edges"])
        assert "pkg.b.g -> pkg.b.h -> pkg.b.g" in resp.metrics.cyclic_dependencies
        # main -> f -> {g, h}: the g<->h cycle is condensed to one step
        assert resp.metrics.max_call_depth == 2
        assert "pkg.a.main" not in resp.orphaned_functions  # entry point
        assert "pkg.a.C.m" in resp.orphaned_functions  # nobody calls m
        assert "pkg.a.main -> pkg.a.f -> pkg.b.g" in resp.critical_paths

    async def test_no_python_is_not_computed(self, tmp_path: Path):
        (tmp_path / "README.md").write_text("no code here")
        gen = CallGraphGenerator(CallGraphConfig(analysis_scope=["."]))
        with patch.object(gen, "_available_tool", return_value=None):
            resp = await gen.generate_comprehensive_call_graph(str(tmp_path))
        assert resp.status == "not_computed"
        assert "no Python files" in (resp.not_computed_reason or "")
        assert resp.tools_used == []
        assert resp.tools_unavailable == ["pyan3", "code2flow", "pycg"]

    async def test_missing_scope_is_not_computed(self, repo: Path):
        gen = CallGraphGenerator(CallGraphConfig(analysis_scope=["src/"]))
        with patch.object(gen, "_available_tool", return_value=None):
            resp = await gen.generate_comprehensive_call_graph(str(repo))
        assert resp.status == "not_computed"
        assert "src/" in (resp.not_computed_reason or "")

    async def test_unavailable_tools_are_never_run(self, repo: Path):
        gen = CallGraphGenerator(CallGraphConfig(analysis_scope=["."]))
        runner = AsyncMock(return_value={"nodes": [], "edges": []})
        with (
            patch.object(gen, "_available_tool", return_value=None),
            patch.object(gen, "_run_pyan3_analysis", runner),
        ):
            resp = await gen.generate_comprehensive_call_graph(str(repo))
        runner.assert_not_called()
        assert resp.tools_used == ["python-ast"]


class TestGraphMetrics:
    def test_depth_is_capped_and_cycles_found(self):
        nodes = [f"m.f{i}" for i in range(6)]
        edges = [(nodes[i], nodes[i + 1]) for i in range(5)] + [("m.f5", "m.f4")]
        result = analyse_graph(nodes, edges, max_depth=2)
        assert result.max_depth == 2
        assert result.cycles == ["m.f4 -> m.f5 -> m.f4"]
        assert result.orphaned == ["m.f0"]


# --- analysis_scope confinement (cross-tenant read) ---------------------------


@pytest.fixture
def two_tenants(tmp_path: Path) -> tuple[Path, Path]:
    """Two sibling materialised workspaces, as /tmp holds them in the pod."""
    tenant_a = tmp_path / "ws-tenantA"
    tenant_b = tmp_path / "ws-tenantB"
    (tenant_a / "pkg").mkdir(parents=True)
    (tenant_a / "pkg" / "a.py").write_text("def a_only():\n    return 1\n")
    tenant_b.mkdir()
    (tenant_b / "secret.py").write_text("def tenant_b_secret():\n    return 2\n")
    return tenant_a, tenant_b


class TestScopeConfinement:
    @pytest.mark.parametrize(
        "scope",
        ["../ws-tenantB", "/", "pkg/../../ws-tenantB", "/etc", "pkg/\x00x"],
    )
    async def test_escaping_scope_is_refused_and_nothing_is_read(
        self, two_tenants: tuple[Path, Path], scope: str
    ):
        from processing.analyzers.call_graph_analyzer import InvalidAnalysisScopeError

        tenant_a, _ = two_tenants
        gen = CallGraphGenerator(CallGraphConfig(analysis_scope=[scope]))
        build = patch(
            "processing.analyzers.call_graph_analyzer.build_python_call_graph",
            side_effect=AssertionError("must not read an unconfined scope"),
        )
        with (
            patch.object(gen, "_available_tool", return_value=None),
            build,
            pytest.raises(InvalidAnalysisScopeError) as exc_info,
        ):
            await gen.generate_comprehensive_call_graph(str(tenant_a))
        assert exc_info.value.rejected == [scope]

    async def test_symlink_out_of_the_repo_is_refused(self, two_tenants: tuple[Path, Path]):
        from processing.analyzers.call_graph_analyzer import InvalidAnalysisScopeError

        tenant_a, tenant_b = two_tenants
        (tenant_a / "escape").symlink_to(tenant_b, target_is_directory=True)
        gen = CallGraphGenerator(CallGraphConfig(analysis_scope=["escape"]))
        with (
            patch.object(gen, "_available_tool", return_value=None),
            pytest.raises(InvalidAnalysisScopeError),
        ):
            await gen.generate_comprehensive_call_graph(str(tenant_a))

    async def test_confined_scope_still_analysed(self, two_tenants: tuple[Path, Path]):
        """Negative control: an in-repo scope works and never sees the sibling."""
        tenant_a, _ = two_tenants
        gen = CallGraphGenerator(CallGraphConfig(analysis_scope=["pkg", "./pkg/../pkg"]))
        with patch.object(gen, "_available_tool", return_value=None):
            resp = await gen.generate_comprehensive_call_graph(str(tenant_a))
        assert resp.status == "computed"
        ids = {n["id"] for n in resp.call_graph["nodes"]}
        assert any(i.endswith("a_only") for i in ids)
        assert not any("tenant_b_secret" in i for i in ids)

    async def test_endpoint_maps_escaping_scope_to_400(self, two_tenants: tuple[Path, Path]):
        from fastapi import BackgroundTasks, HTTPException

        from app.api.endpoints import static_analysis
        from models.static_analysis_models import CallGraphRequest

        tenant_a, _ = two_tenants
        with (
            patch.object(
                static_analysis,
                "_materialize_workspace_for",
                AsyncMock(return_value=tenant_a),
            ),
            patch.object(static_analysis, "remove_materialized"),
            patch.object(CallGraphGenerator, "_available_tool", return_value=None),
            pytest.raises(HTTPException) as exc_info,
        ):
            await static_analysis.generate_call_graph(
                request=CallGraphRequest(container_name="ws", analysis_scope=["../ws-tenantB"]),
                background_tasks=BackgroundTasks(),
                redis_client=None,
                tenant_id="0195f48b-d000-7000-8000-00000000000a",
            )
        assert exc_info.value.status_code == 400


# --- parse failures are not an empty graph -----------------------------------


class TestParseFailures:
    async def test_every_file_unparseable_is_not_computed(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("def broken(:\n")
        (tmp_path / "b.py").write_text("class (:\n")
        gen = CallGraphGenerator(CallGraphConfig(analysis_scope=["."]))
        with patch.object(gen, "_available_tool", return_value=None):
            resp = await gen.generate_comprehensive_call_graph(str(tmp_path))
        assert resp.status == "not_computed"
        assert "could be parsed" in (resp.not_computed_reason or "")
        assert resp.metrics.parse_errors == 2

    async def test_some_files_unparseable_is_computed_with_count(self, tmp_path: Path):
        """Negative control: one good file keeps it computed, the failure is counted."""
        (tmp_path / "a.py").write_text("def broken(:\n")
        (tmp_path / "ok.py").write_text("def fine():\n    return 1\n")
        gen = CallGraphGenerator(CallGraphConfig(analysis_scope=["."]))
        with patch.object(gen, "_available_tool", return_value=None):
            resp = await gen.generate_comprehensive_call_graph(str(tmp_path))
        assert resp.status == "computed"
        assert resp.metrics.parse_errors == 1
        assert resp.metrics.total_functions == 1
