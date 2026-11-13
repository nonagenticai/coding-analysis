"""
Call Graph Generator for AI Agent Static Analysis Augmentation
Generates comprehensive call graphs using multiple tools (Pyan3, Code2flow, PyCG).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from models.core_models import (
    CoreAnalysisResult,
)
from models.static_analysis_models import (
    CallGraphMetrics,
    CallGraphResponse,
    OutputFormat,
)

from ..quality_analysis.tool_registry import get_tool_registry
from .base_analyzer import BaseCodeProcessor
from .call_graph_metrics import analyse_graph
from .python_call_graph import PythonCallGraph, build_python_call_graph, find_python_files

logger = logging.getLogger(__name__)


class InvalidAnalysisScopeError(ValueError):
    """An ``analysis_scope`` entry that is absolute or resolves outside the repository.

    The repository is a tenant's materialised workspace; a scope that escapes it
    (``../ws-other-tenant``, ``/``, a symlink pointing out) would make the
    analysers read another tenant's tree or this pod's filesystem.
    """

    def __init__(self, rejected: list[str]) -> None:
        super().__init__(f"analysis_scope paths escape the repository: {rejected}")
        self.rejected = rejected


def confine_scopes(repository_path: str, scopes: list[str] | None) -> list[str]:
    """Return *scopes* as confined absolute paths under *repository_path*, or raise.

    Rejects absolute scopes, NUL, and any scope whose resolved realpath (symlinks
    followed) is not the repository root's realpath or below it. Scopes that do
    not exist are returned as-is (confined lexically); callers treat them as
    missing.
    """
    repo_real = os.path.realpath(repository_path)
    confined: list[str] = []
    rejected: list[str] = []
    for scope in scopes or ["."]:
        if not isinstance(scope, str) or "\x00" in scope or os.path.isabs(scope):
            rejected.append(str(scope))
            continue
        resolved = os.path.realpath(os.path.join(repo_real, scope))
        if resolved != repo_real and not resolved.startswith(repo_real + os.sep):
            rejected.append(scope)
            continue
        confined.append(resolved)
    if rejected:
        raise InvalidAnalysisScopeError(rejected)
    return confined


@dataclass
class CallGraphConfig:
    """Configuration for call graph analysis."""

    output_format: OutputFormat = OutputFormat.JSON
    include_external: bool = False
    max_depth: int = 10
    analysis_scope: list[str] | None = None

    def __post_init__(self) -> None:
        if self.analysis_scope is None:
            self.analysis_scope = ["."]


class CallGraphGenerator(BaseCodeProcessor):
    """Generator for creating comprehensive call graphs using multiple tools."""

    def __init__(self, config: CallGraphConfig | None = None):
        """Initialize call graph generator with configuration."""
        self.config = config or CallGraphConfig()
        self.tool_registry = get_tool_registry()
        self.supported_tools = ["pyan3", "code2flow", "pycg"]

    def get_supported_extensions(self) -> list[str]:
        """Return supported file extensions."""
        return [".py"]  # Focus on Python for initial implementation

    def get_language_name(self) -> str:
        """Return the programming language name."""
        return "python"

    async def analyze_file(self, file_path: str) -> CoreAnalysisResult:
        """Analyze a single file for call graph generation."""
        try:
            call_graph_data = await self.generate_call_graph_for_file(file_path)

            return CoreAnalysisResult(
                success=True,
                language="python",
                file_path=file_path,
                entities=[],  # Call graphs focus on relationshipships
                relationships=[],
                complexity_metrics=None,
                error=None,
                parse_time_ms=0.0,
                total_lines=0,
                metadata={
                    "call_graph": call_graph_data,
                    "analyzer": "call_graph",
                    "file_path": file_path,
                },
            )
        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.exception("Error analyzing file %s: %s", file_path, e)
            return CoreAnalysisResult(
                success=False,
                language="python",
                file_path=file_path,
                entities=[],
                relationships=[],
                complexity_metrics=None,
                error=str(e),
                parse_time_ms=0.0,
                total_lines=0,
                metadata={"error": str(e), "file_path": file_path},
            )

    async def analyze_directory(self, directory_path: str) -> CoreAnalysisResult:
        """Analyze directory to generate comprehensive call graph."""
        try:
            call_graph_data = await self.generate_call_graph_for_directory(directory_path)

            return CoreAnalysisResult(
                success=True,
                language="python",
                file_path=directory_path,
                entities=[],
                relationships=[],
                complexity_metrics=None,
                error=None,
                parse_time_ms=0.0,
                total_lines=0,
                metadata={
                    "call_graph": call_graph_data,
                    "analyzer": "call_graph",
                    "directory_path": directory_path,
                },
            )
        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.exception("Error analyzing directory %s: %s", directory_path, e)
            return CoreAnalysisResult(
                success=False,
                language="python",
                file_path=directory_path,
                entities=[],
                relationships=[],
                complexity_metrics=None,
                error=str(e),
                parse_time_ms=0.0,
                total_lines=0,
                metadata={"error": str(e), "directory_path": directory_path},
            )

    async def analyze_snippet(
        self, code_snippet: str, file_path: str | None = None
    ) -> CoreAnalysisResult:
        """Analyze code snippet for call relationshipships."""
        try:
            # Create temporary file for snippet analysis
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                tmp_file.write(code_snippet)
                tmp_file_path = tmp_file.name

            try:
                call_graph_data = await self.generate_call_graph_for_file(tmp_file_path)

                return CoreAnalysisResult(
                    success=True,
                    language="python",
                    file_path=file_path or tmp_file_path,
                    entities=[],
                    relationships=[],
                    complexity_metrics=None,
                    error=None,
                    parse_time_ms=0.0,
                    total_lines=0,
                    metadata={
                        "call_graph": call_graph_data,
                        "analyzer": "call_graph",
                        "snippet": True,
                        "original_file": file_path,
                    },
                )
            finally:
                os.unlink(tmp_file_path)

        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.exception("Error analyzing code snippet: %s", e)
            return CoreAnalysisResult(
                success=False,
                language="python",
                file_path=file_path or "",
                entities=[],
                relationships=[],
                complexity_metrics=None,
                error=str(e),
                parse_time_ms=0.0,
                total_lines=0,
                metadata={"error": str(e), "snippet": True},
            )

    async def generate_comprehensive_call_graph(
        self, repository_path: str, config: CallGraphConfig | None = None
    ) -> CallGraphResponse:
        """
        Generate the call graph for a repository.

        The primary backend is the in-process ``ast`` builder
        (``python_call_graph``), which needs no external tool. Pyan3/Code2flow/
        PyCG run only when installed; ``tools_used`` lists only backends that
        actually ran and ``tools_unavailable`` the ones that are not installed.
        When nothing could analyse the repository the response is
        ``status="not_computed"`` with a reason - never an empty "success".

        Args:
            repository_path: Path to the repository root
            config: Optional configuration override

        Returns:
            CallGraphResponse with complete analysis results
        """
        analysis_config = config or self.config
        analysis_id = f"cg_{hash(repository_path) % 1000000:06d}"

        try:
            roots, missing_scopes = self._scope_roots(repository_path, analysis_config)
            python_files = find_python_files(roots)

            ast_graph: PythonCallGraph | None = None
            ast_parse_failures: list[str] = []
            if python_files:
                ast_graph = self._build_ast_call_graph(repository_path, roots, analysis_config)
                if ast_graph.files_analyzed == 0:
                    # Every file failed to parse: an empty graph here is not
                    # evidence of an empty call graph, so the AST backend did not run.
                    ast_parse_failures = list(ast_graph.parse_errors)
                    ast_graph = None

            tool_graphs, tools_unavailable, tool_errors = await self._run_external_tools(
                repository_path, analysis_config
            )
            tools_used = (["python-ast"] if ast_graph is not None else []) + list(tool_graphs)

            if not tools_used:
                if missing_scopes and not roots:
                    reason = f"analysis_scope paths not found in repository: {missing_scopes}"
                elif ast_parse_failures:
                    reason = (
                        f"none of the {len(python_files)} Python files could be parsed "
                        f"({len(ast_parse_failures)} parse errors)"
                    )
                elif not python_files:
                    reason = "no Python files in analysis_scope and no external tool ran"
                else:
                    reason = "no call-graph backend ran"
                response = self._not_computed_response(
                    analysis_id, reason, tools_unavailable, tool_errors
                )
                if ast_parse_failures:
                    response.call_graph["parse_errors"] = ast_parse_failures
                    response.metrics.parse_errors = len(ast_parse_failures)
                return response

            if ast_graph is not None:
                nodes, edges = ast_graph.nodes, ast_graph.edges
                unresolved = ast_graph.unresolved_calls
            else:
                merged = self._merge_call_graph_results(
                    *(tool_graphs.get(t, {}) for t in ("pyan3", "code2flow", "pycg"))
                )
                nodes, edges, unresolved = merged["nodes"], merged["edges"], 0

            merged_graph: dict[str, Any] = {
                "nodes": nodes,
                "edges": edges,
                "tools_used": tools_used,
                "tools_unavailable": tools_unavailable,
                "merge_timestamp": self._get_timestamp(),
            }
            if ast_graph is not None:
                merged_graph["files_analyzed"] = ast_graph.files_analyzed
                merged_graph["parse_errors"] = ast_graph.parse_errors
                if analysis_config.include_external:
                    merged_graph["external_nodes"] = ast_graph.external_nodes
            if tool_graphs and ast_graph is not None:
                merged_graph["tool_graphs"] = tool_graphs
            if tool_errors:
                merged_graph["tool_errors"] = tool_errors
            if missing_scopes:
                merged_graph["missing_scopes"] = missing_scopes

            # Generate visual output if requested
            visual_output = None
            if analysis_config.output_format in [
                OutputFormat.SVG,
                OutputFormat.DOT,
            ]:
                visual_output = await self._generate_visual_output(
                    merged_graph, analysis_config.output_format
                )

            analysis = analyse_graph(
                [n.get("id") or f"{n.get('module', '')}.{n.get('function', '')}" for n in nodes],
                [
                    (e.get("from", ""), e.get("to", ""))
                    for e in edges
                    if e.get("call_type", "direct") != "external"
                ],
                analysis_config.max_depth,
            )

            return CallGraphResponse(
                analysis_id=analysis_id,
                status="computed",
                tools_used=tools_used,
                tools_unavailable=tools_unavailable,
                call_graph=merged_graph,
                visual_output=visual_output,
                critical_paths=analysis.critical_paths,
                orphaned_functions=analysis.orphaned,
                metrics=CallGraphMetrics(
                    total_functions=len(nodes),
                    max_call_depth=analysis.max_depth,
                    cyclic_dependencies=analysis.cycles,
                    total_edges=analysis.internal_edges,
                    unresolved_calls=unresolved,
                    parse_errors=(
                        len(ast_graph.parse_errors)
                        if ast_graph is not None
                        else len(ast_parse_failures)
                    ),
                ),
            )

        except (
            OSError,
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
        ) as e:
            logger.exception("Error generating comprehensive call graph: %s", e)
            response = self._not_computed_response(
                analysis_id, f"call graph generation failed: {e}", [], {}
            )
            response.call_graph["error"] = str(e)
            return response

    def _scope_roots(
        self, repository_path: str, config: CallGraphConfig
    ) -> tuple[list[str], list[str]]:
        """Existing analysis_scope paths (absolute) and the ones that are missing."""
        roots: list[str] = []
        missing: list[str] = []
        scopes = config.analysis_scope or ["."]
        for scope, scope_path in zip(scopes, confine_scopes(repository_path, scopes), strict=True):
            if os.path.exists(scope_path):
                roots.append(scope_path)
            else:
                missing.append(scope)
        return roots, missing

    def _build_ast_call_graph(
        self, repository_path: str, roots: list[str], config: CallGraphConfig
    ) -> PythonCallGraph:
        return build_python_call_graph(
            repository_path, roots, include_external=config.include_external
        )

    def _available_tool(self, name: str) -> Any | None:
        """The registry entry for *name* when its executable is on PATH."""
        tool = self.tool_registry.get_tool("python", name)
        command = getattr(tool, "command", None) if tool else None
        if not isinstance(command, list) or not command or not isinstance(command[0], str):
            return None
        return tool if shutil.which(command[0]) else None

    async def _run_external_tools(
        self, repository_path: str, config: CallGraphConfig
    ) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, str]]:
        """Run the installed external tools: (graphs that ran, unavailable, errors)."""
        runners = {
            "pyan3": self._run_pyan3_analysis,
            "code2flow": self._run_code2flow_analysis,
            "pycg": self._run_pycg_analysis,
        }
        ran: dict[str, dict[str, Any]] = {}
        unavailable: list[str] = []
        errors: dict[str, str] = {}
        for name, runner in runners.items():
            if self._available_tool(name) is None:
                unavailable.append(name)
                continue
            result = await runner(repository_path, config)
            if result.get("error") or result.get("status"):
                errors[name] = str(result.get("error") or result.get("status"))
                continue
            ran[name] = result
        return ran, unavailable, errors

    def _not_computed_response(
        self,
        analysis_id: str,
        reason: str,
        tools_unavailable: list[str],
        tool_errors: dict[str, str],
    ) -> CallGraphResponse:
        call_graph: dict[str, Any] = {"nodes": [], "edges": [], "tools_used": []}
        if tool_errors:
            call_graph["tool_errors"] = tool_errors
        return CallGraphResponse(
            analysis_id=analysis_id,
            status="not_computed",
            not_computed_reason=reason,
            tools_used=[],
            tools_unavailable=tools_unavailable,
            call_graph=call_graph,
            visual_output=None,
            critical_paths=[],
            orphaned_functions=[],
            metrics=CallGraphMetrics(total_functions=0, max_call_depth=0, cyclic_dependencies=[]),
        )

    async def generate_call_graph_for_file(self, file_path: str) -> dict[str, Any]:
        """Generate call graph for a single file."""
        try:
            # Use PyCG for precise single-file analysis
            result = await self._run_pycg_on_file(file_path)
            return result
        except (
            OSError,
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
        ) as e:
            logger.exception("Error generating call graph for file %s: %s", file_path, e)
            return {"error": str(e), "file_path": file_path}

    async def generate_call_graph_for_directory(self, directory_path: str) -> dict[str, Any]:
        """Generate call graph for entire directory."""
        try:
            config = CallGraphConfig(analysis_scope=["."])
            result = await self.generate_comprehensive_call_graph(directory_path, config)
            return result.call_graph
        except (
            OSError,
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
        ) as e:
            logger.exception(
                "Error generating call graph for directory %s: %s",
                directory_path,
                e,
            )
            return {"error": str(e), "directory_path": directory_path}

    async def _run_pyan3_analysis(
        self, repository_path: str, config: CallGraphConfig
    ) -> dict[str, Any]:
        """Run Pyan3 call graph analysis."""
        try:
            tool = self.tool_registry.get_tool("python", "pyan3")
            if not tool:
                logger.warning("Pyan3 tool not found in registry")
                return {
                    "nodes": [],
                    "edges": [],
                    "tool": "pyan3",
                    "status": "tool_not_found",
                }

            # Build command for Pyan3
            cmd = tool.command.copy()

            # Add repository files to analyze
            python_files = []
            for scope in config.analysis_scope or []:
                scope_path = os.path.join(repository_path, scope)
                if os.path.exists(scope_path):
                    python_files.extend(self._find_python_files(scope_path))

            if not python_files:
                return {
                    "nodes": [],
                    "edges": [],
                    "tool": "pyan3",
                    "status": "no_files",
                }

            # Run Pyan3
            cmd.extend(python_files)
            result = subprocess.run(
                cmd,
                cwd=repository_path,
                capture_output=True,
                text=True,
                timeout=tool.timeout,
                check=False,
            )

            if result.returncode == 0:
                # Parse Pyan3 DOT output
                return self._parse_pyan3_output(result.stdout)

            logger.warning("Pyan3 failed: %s", result.stderr)
            return {
                "nodes": [],
                "edges": [],
                "tool": "pyan3",
                "error": result.stderr,
            }

        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.exception("Error running Pyan3 analysis: %s", e)
            return {"nodes": [], "edges": [], "tool": "pyan3", "error": str(e)}

    async def _run_code2flow_analysis(
        self, repository_path: str, config: CallGraphConfig
    ) -> dict[str, Any]:
        """Run Code2flow call graph analysis."""
        try:
            tool = self.tool_registry.get_tool("python", "code2flow")
            if not tool:
                logger.warning("Code2flow tool not found in registry")
                return {
                    "nodes": [],
                    "edges": [],
                    "tool": "code2flow",
                    "status": "tool_not_found",
                }

            # Build command for Code2flow
            cmd = tool.command.copy()

            # Add output format
            with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as tmp_file:
                output_file = tmp_file.name

            # Add analysis scope if specified
            target_paths = []
            for scope in config.analysis_scope or ["."]:
                scope_path = os.path.join(repository_path, scope)
                if os.path.exists(scope_path):
                    target_paths.append(scope_path)

            target_path = target_paths[0] if target_paths else repository_path
            cmd.extend([target_path, "--output", output_file])

            # Run Code2flow
            result = subprocess.run(
                cmd,
                cwd=repository_path,
                capture_output=True,
                text=True,
                timeout=tool.timeout,
                check=False,
            )

            try:
                if result.returncode == 0 and os.path.exists(output_file):
                    with open(output_file, encoding="utf-8") as f:
                        dot_content = f.read()
                    return self._parse_dot_output(dot_content, "code2flow")

                logger.warning("Code2flow failed: %s", result.stderr)
                return {
                    "nodes": [],
                    "edges": [],
                    "tool": "code2flow",
                    "error": result.stderr,
                }
            finally:
                if os.path.exists(output_file):
                    os.unlink(output_file)

        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.exception("Error running Code2flow analysis: %s", e)
            return {
                "nodes": [],
                "edges": [],
                "tool": "code2flow",
                "error": str(e),
            }

    async def _run_pycg_analysis(
        self, repository_path: str, config: CallGraphConfig
    ) -> dict[str, Any]:
        """Run PyCG call graph analysis."""
        try:
            tool = self.tool_registry.get_tool("python", "pycg")
            if not tool:
                logger.warning("PyCG tool not found in registry")
                return {
                    "nodes": [],
                    "edges": [],
                    "tool": "pycg",
                    "status": "tool_not_found",
                }

            # Build command for PyCG
            cmd = tool.command.copy()

            # Add repository path and entry points
            entry_points = []
            for scope in config.analysis_scope or []:
                scope_path = os.path.join(repository_path, scope)
                if os.path.exists(scope_path):
                    entry_points.extend(self._find_python_entry_points(scope_path))

            if not entry_points:
                return {
                    "nodes": [],
                    "edges": [],
                    "tool": "pycg",
                    "status": "no_entry_points",
                }

            cmd.extend(entry_points)

            # Run PyCG
            result = subprocess.run(
                cmd,
                cwd=repository_path,
                capture_output=True,
                text=True,
                timeout=tool.timeout,
                check=False,
            )

            if result.returncode == 0:
                try:
                    pycg_data = json.loads(result.stdout)
                    return self._parse_pycg_output(pycg_data)
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse PyCG JSON output: %s", e)
                    return {
                        "nodes": [],
                        "edges": [],
                        "tool": "pycg",
                        "error": "json_parse_error",
                    }

            logger.warning("PyCG failed: %s", result.stderr)
            return {
                "nodes": [],
                "edges": [],
                "tool": "pycg",
                "error": result.stderr,
            }

        except (
            OSError,
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
        ) as e:
            logger.exception("Error running PyCG analysis: %s", e)
            return {"nodes": [], "edges": [], "tool": "pycg", "error": str(e)}

    async def _run_pycg_on_file(self, file_path: str) -> dict[str, Any]:
        """Run PyCG analysis on a single file."""
        try:
            tool = self.tool_registry.get_tool("python", "pycg")
            if not tool:
                logger.warning("PyCG tool not found in registry")
                return {"nodes": [], "edges": [], "error": "tool_not_found"}

            cmd = tool.command.copy()
            cmd.append(file_path)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=tool.timeout,
                check=False,
            )

            if result.returncode == 0:
                try:
                    pycg_data = json.loads(result.stdout)
                    return self._parse_pycg_output(pycg_data)
                except json.JSONDecodeError:
                    return {
                        "nodes": [],
                        "edges": [],
                        "error": "json_parse_error",
                    }

            return {"nodes": [], "edges": [], "error": result.stderr}

        except (
            OSError,
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
        ) as e:
            logger.exception("Error running PyCG on file %s: %s", file_path, e)
            return {"nodes": [], "edges": [], "error": str(e)}

    def _merge_call_graph_results(
        self,
        pyan3_result: dict[str, Any],
        code2flow_result: dict[str, Any],
        pycg_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge call graph results from multiple tools."""
        merged_nodes = {}
        merged_edges = []

        # Merge nodes from all tools
        for result in [pyan3_result, code2flow_result, pycg_result]:
            if "nodes" in result:
                for node in result["nodes"]:
                    node_key = f"{node.get('module', '')}.{node.get('function', '')}"
                    if node_key not in merged_nodes:
                        merged_nodes[node_key] = node
                    else:
                        # Merge additional information
                        merged_nodes[node_key].update(node)

        # Merge edges from all tools
        edge_set = set()
        for result in [pyan3_result, code2flow_result, pycg_result]:
            if "edges" in result:
                for edge in result["edges"]:
                    edge_key = (edge.get("from"), edge.get("to"))
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        merged_edges.append(edge)

        return {
            "nodes": list(merged_nodes.values()),
            "edges": merged_edges,
            "tools_used": ["pyan3", "code2flow", "pycg"],
            "merge_timestamp": self._get_timestamp(),
        }

    def _parse_pyan3_output(self, dot_content: str) -> dict[str, Any]:
        """Parse Pyan3 DOT output into structured format."""
        return self._parse_dot_output(dot_content, "pyan3")

    def _parse_dot_output(self, dot_content: str, tool_name: str) -> dict[str, Any]:
        """Parse DOT format output into structured nodes and edges."""
        nodes = []
        edges = []

        lines = dot_content.split("\n")
        for line in lines:
            line = line.strip()

            # Parse node definitions
            if " [" in line and "->" not in line:
                # Extract node name and attributes
                parts = line.split(" [")
                if len(parts) >= 2:
                    node_name = parts[0].strip().strip('"')
                    if node_name and node_name != "digraph":
                        nodes.append(
                            {
                                "function": (
                                    node_name.split(".")[-1] if "." in node_name else node_name
                                ),
                                "module": (
                                    ".".join(node_name.split(".")[:-1]) if "." in node_name else ""
                                ),
                                "complexity": 1,  # Default complexity
                            }
                        )

            # Parse edge definitions
            elif "->" in line:
                parts = line.split("->")
                if len(parts) >= 2:
                    from_node = parts[0].strip().strip('"')
                    to_part = parts[1].split("[")[0].strip().strip('"')

                    if from_node and to_part:
                        edges.append(
                            {
                                "from": from_node,
                                "to": to_part,
                                "call_type": "direct",
                            }
                        )

        return {"nodes": nodes, "edges": edges, "tool": tool_name}

    def _parse_pycg_output(self, pycg_data: dict[str, Any]) -> dict[str, Any]:
        """Parse PyCG JSON output into structured format."""
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        # Extract nodes and edges from PyCG format
        if isinstance(pycg_data, dict):
            for caller, callees in pycg_data.items():
                # Add caller as node
                if caller not in [node.get("function") for node in nodes]:
                    nodes.append(
                        {
                            "function": (caller.split(".")[-1] if "." in caller else caller),
                            "module": (".".join(caller.split(".")[:-1]) if "." in caller else ""),
                            "complexity": 1,
                        }
                    )

                # Add callees and edges
                if isinstance(callees, list):
                    for callee in callees:
                        # Add callee as node
                        if callee not in [node.get("function") for node in nodes]:
                            nodes.append(
                                {
                                    "function": (
                                        callee.split(".")[-1] if "." in callee else callee
                                    ),
                                    "module": (
                                        ".".join(callee.split(".")[:-1]) if "." in callee else ""
                                    ),
                                    "complexity": 1,
                                }
                            )

                        # Add edge
                        edges.append(
                            {
                                "from": caller,
                                "to": callee,
                                "call_type": "direct",
                            }
                        )

        return {"nodes": nodes, "edges": edges, "tool": "pycg"}

    def _find_python_files(self, directory: str) -> list[str]:
        """Find all Python files in directory."""
        python_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files

    def _find_python_entry_points(self, directory: str) -> list[str]:
        """Find Python entry points (main files, __init__.py, etc.)."""
        entry_points = []

        # Look for main files
        main_files = ["main.py", "__main__.py", "app.py", "run.py"]
        for main_file in main_files:
            main_path = os.path.join(directory, main_file)
            if os.path.exists(main_path):
                entry_points.append(main_path)

        # Look for __init__.py files
        for root, _, files in os.walk(directory):
            if "__init__.py" in files:
                entry_points.append(os.path.join(root, "__init__.py"))

        return entry_points[:10]  # Limit to prevent excessive analysis

    async def _generate_visual_output(
        self, call_graph: dict[str, Any], output_format: OutputFormat
    ) -> str | None:
        """Generate visual representation of call graph."""
        try:
            # Convert to DOT format
            dot_content = self._convert_to_dot(call_graph)

            if output_format == OutputFormat.DOT:
                return base64.b64encode(dot_content.encode()).decode()

            if output_format == OutputFormat.SVG:
                # Use Graphviz to generate SVG
                result = subprocess.run(
                    ["dot", "-Tsvg"],
                    input=dot_content,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )

                if result.returncode == 0:
                    return base64.b64encode(result.stdout.encode()).decode()

                logger.warning("Graphviz failed: %s", result.stderr)
                return None

            return None

        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.exception("Error generating visual output: %s", e)
            return None

    def _convert_to_dot(self, call_graph: dict[str, Any]) -> str:
        """Convert call graph to DOT format."""
        dot_lines = ["digraph CallGraph {", "  rankdir=LR;"]

        # Add nodes
        nodes = call_graph.get("nodes", [])
        for node in nodes:
            function_name = node.get("function", "unknown")
            module_name = node.get("module", "")
            label = f"{module_name}.{function_name}" if module_name else function_name
            dot_lines.append(f'  "{label}" [label="{label}"];')

        # Add edges
        edges = call_graph.get("edges", [])
        for edge in edges:
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            if from_node and to_node:
                dot_lines.append(f'  "{from_node}" -> "{to_node}";')

        dot_lines.append("}")
        return "\n".join(dot_lines)

    def _identify_critical_paths(self, call_graph: dict[str, Any]) -> list[str]:
        """Identify critical execution paths in call graph."""
        critical_paths: list[str] = []

        # Basic heuristic: find longest paths
        edges = call_graph.get("edges", [])
        if not edges:
            return critical_paths

        # Build adjacency list
        adj_list: dict[str, list[str]] = {}
        for edge in edges:
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            if from_node not in adj_list:
                adj_list[from_node] = []
            adj_list[from_node].append(to_node)

        # Find paths using DFS (streamlined)
        def dfs(node: str, path: list[str], visited: set[str]) -> None:
            if len(path) > 3:  # Limit path length
                critical_paths.append(" -> ".join(path))
                return

            if node in adj_list:
                for neighbor in adj_list[node]:
                    if neighbor not in visited:
                        dfs(neighbor, path + [neighbor], visited | {neighbor})

        # Start DFS from each node
        for node in list(adj_list.keys())[:5]:  # Limit starting nodes
            dfs(node, [node], {node})

        return critical_paths[:10]  # Return top 10 critical paths

    def _find_orphaned_functions(self, call_graph: dict[str, Any]) -> list[str]:
        """Find functions with no callers."""
        orphaned = []

        edges = call_graph.get("edges", [])
        nodes = call_graph.get("nodes", [])

        # Get all called functions
        called_functions = set()
        for edge in edges:
            called_functions.add(edge.get("to", ""))

        # Find functions that are never called
        for node in nodes:
            function_name = node.get("function", "")
            module = node.get("module", "")
            full_name = f"{module}.{function_name}" if module else function_name

            if full_name not in called_functions and function_name not in called_functions:
                orphaned.append(full_name)

        return orphaned

    def _calculate_call_graph_metrics(self, call_graph: dict[str, Any]) -> CallGraphMetrics:
        """Calculate metrics about the call graph."""
        nodes = call_graph.get("nodes", [])
        edges = call_graph.get("edges", [])

        total_functions = len(nodes)

        # Calculate max call depth (streamlined)
        max_depth = 0
        adj_list: dict[str, list[str]] = {}
        if edges:
            # Build adjacency list and calculate depths
            adj_list: dict[str, list[str]] = {}
            for edge in edges:
                from_node = edge.get("from", "")
                if from_node not in adj_list:
                    adj_list[from_node] = []
                adj_list[from_node].append(edge.get("to", ""))

            # Basic depth calculation
            max_depth = min(len(adj_list), 10)  # Cap at reasonable depth

        # Detect cyclic dependencies using DFS
        cyclic_dependencies = self._detect_cycles(adj_list)

        return CallGraphMetrics(
            total_functions=total_functions,
            max_call_depth=max_depth,
            cyclic_dependencies=cyclic_dependencies,
        )

    def _detect_cycles(self, adj_list: dict[str, list[str]]) -> list[str]:
        """Detect cycles in the call graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: list[str]) -> bool:
            """DFS helper to detect cycles."""
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycle_str = " -> ".join(cycle)
                if cycle_str not in cycles:
                    cycles.append(cycle_str)
                return True

            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            # Visit all neighbors
            for neighbor in adj_list.get(node, []):
                if dfs(neighbor, path):
                    pass  # Continue to find all cycles

            rec_stack.remove(node)
            path.pop()
            return False

        # Check all nodes for cycles
        for node in adj_list:
            if node not in visited:
                dfs(node, [])

        return cycles

    def _calculate_call_depth(self, adj_list: dict[str, list[str]], start_node: str) -> int:
        """Calculate maximum call depth from a starting node."""
        visited = set()

        def dfs_depth(node: str, current_depth: int) -> int:
            if node in visited:
                return current_depth  # Avoid infinite recursion

            visited.add(node)
            local_max = current_depth

            for neighbor in adj_list.get(node, []):
                depth = dfs_depth(neighbor, current_depth + 1)
                local_max = max(local_max, depth)

            visited.remove(node)
            return local_max

        return dfs_depth(start_node, 0)

    def _analyze_call_patterns(self, graph_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze common call patterns in the graph."""
        patterns: dict[str, Any] = {
            "recursive_functions": [],
            "high_fan_out": [],  # Functions that call many others
            "high_fan_in": [],  # Functions called by many others
            "leaf_functions": [],  # Functions that don't call others
            "entry_points": [],  # Functions not called by others
        }

        # Build adjacency lists for analysis
        calls_out: dict[str, list[str]] = {}  # node -> [nodes it calls]
        calls_in: dict[str, list[str]] = {}  # node -> [nodes that call it]

        for edge in graph_data.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")

            if source and target:
                if source not in calls_out:
                    calls_out[source] = []
                calls_out[source].append(target)

                if target not in calls_in:
                    calls_in[target] = []
                calls_in[target].append(source)

        all_nodes = set(graph_data.get("nodes", []))

        for node in all_nodes:
            node_id = node if isinstance(node, str) else node.get("id", "")

            # Check for recursive functions (calls itself)
            if node_id in calls_out.get(node_id, []):
                patterns["recursive_functions"].append(node_id)

            # High fan-out (calls many functions)
            out_degree = len(calls_out.get(node_id, []))
            if out_degree > 5:  # Threshold for "many"
                patterns["high_fan_out"].append({"function": node_id, "calls_count": out_degree})

            # High fan-in (called by many functions)
            in_degree = len(calls_in.get(node_id, []))
            if in_degree > 5:  # Threshold for "many"
                patterns["high_fan_in"].append({"function": node_id, "called_by_count": in_degree})

            # Leaf functions (don't call others)
            if out_degree == 0:
                patterns["leaf_functions"].append(node_id)

            # Entry points (not called by others)
            if in_degree == 0:
                patterns["entry_points"].append(node_id)

        return patterns

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now(UTC).isoformat()
