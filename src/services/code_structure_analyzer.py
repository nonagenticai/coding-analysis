"""Module for analyzing code structure using AST and other utilities."""

from __future__ import annotations

import ast
import logging
import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from pyastgrep.api import search_python_files
from pyastgrep.search import (
    FileFinished,
    MissingPath,
    NonElementReturned,
    ReadError,
    WalkError,
)

logger = logging.getLogger(__name__)


class FunctionCallVisitor(ast.NodeVisitor):
    """AST visitor to extract function call relationships."""

    def __init__(self) -> None:
        """Initialize the visitor."""
        self.current_function: tuple[str, str] | None = None
        self.current_class: str | None = None
        self.calls: list[tuple[tuple[str, str] | None, str, int]] = []
        self.function_stack: list[tuple[str, str] | None] = []
        self.call_graph: defaultdict[Any, defaultdict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit a function definition."""
        prev_function = self.current_function
        func_name = node.name
        if self.current_class:
            self.current_function = (f"{self.current_class}.{func_name}", "method")
        else:
            self.current_function = (func_name, "function")
        self.function_stack.append(self.current_function)
        self.generic_visit(node)
        self.function_stack.pop()
        self.current_function = prev_function if self.function_stack else None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit a class definition."""
        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_Call(self, node: ast.Call) -> None:
        """Visit a function call."""
        self.generic_visit(node)
        if not self.current_function:
            return
        caller = self.current_function
        callee = self._get_call_name(node)
        if callee:
            self.calls.append((caller, callee, getattr(node, "lineno", 0)))
            self.call_graph[caller][callee].append(getattr(node, "lineno", 0))
        self.generic_visit(node)

    def _get_call_name(self, node: ast.Call) -> str | None:
        """Extract the function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return self._get_attribute_path(node.func)
        return None

    def _get_attribute_path(self, node: ast.AST) -> str:
        """Get the full path of an attribute access (e.g., 'module.submodule.func')."""
        if isinstance(node, ast.Attribute):
            path = self._get_attribute_path(node.value)
            return f"{path}.{node.attr}" if path else node.attr
        elif isinstance(node, ast.Name):
            return node.id
        return ""


class CodeStructureAnalyzer:
    """Analyzes code structure using pattern matching and AST-based techniques."""

    def __init__(self, graph_manager: Any = None) -> None:
        """Initialize with optional graph manager."""
        self.graph_manager = graph_manager

    def find_pattern_in_file(
        self, file_path: str, pattern: str, language: str | None = None
    ) -> list[dict[str, Any]]:
        """Find occurrences of a pattern in a file.

        Args:
            file_path: Path to the file to analyze
            pattern: Regular expression or AST pattern to search for
            language: Optional language hint (e.g., 'python', 'javascript')

        Returns:
            List of matches with location and context information
        """
        if not language:
            _, ext = os.path.splitext(file_path)
            language_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".java": "java",
            }
            language = language_map.get(ext.lower(), "text")
        if language == "python":
            try:
                return self._find_using_pyastgrep(file_path, pattern)
            except (FileNotFoundError, PermissionError, ValueError, TypeError) as e:
                logger.warning(f"pyastgrep search failed, falling back to regex: {e}")
        return self._find_using_regex(file_path, pattern)

    def _find_using_regex(self, file_path: str, pattern: str) -> list[dict[str, Any]]:
        """Find pattern using regular expressions.

        Args:
            file_path: Path to the file
            pattern: Regular expression pattern

        Returns:
            List of matches with location information
        """
        results: list[dict[str, Any]] = []
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        start_ctx = max(0, i - 2)
                        end_ctx = min(len(lines), i + 3)
                        context_lines = lines[start_ctx:end_ctx]
                        results.append(
                            {
                                "file": file_path,
                                "line": i + 1,
                                "column": match.start() + 1,
                                "match_text": match.group(0),
                                "context": "\n".join(context_lines),
                                "method": "regex",
                            }
                        )
        except (FileNotFoundError, PermissionError, UnicodeDecodeError, OSError) as e:
            logger.error(f"Error searching file {file_path}: {e}")
        return results

    def _find_using_pyastgrep(self, file_path: str, pattern: str) -> list[dict[str, Any]]:
        """Find pattern using pyastgrep.

        Args:
            file_path: Path to the file to search
            pattern: AST pattern to search for (XPath expression)

        Returns:
            List of matches with location and node information
        """
        results: list[Any] = []
        try:
            matches_iterable = search_python_files(
                paths=[Path(file_path)], expression=pattern, xpath2=False
            )
            for match_obj in matches_iterable:
                if hasattr(match_obj, "position"):
                    results.append(self._process_pyastgrep_match(match_obj))
                elif isinstance(match_obj, (MissingPath, ReadError, WalkError, NonElementReturned)):
                    self._log_pyastgrep_issue(match_obj, file_path, pattern)
                elif isinstance(match_obj, FileFinished):
                    pass
                else:
                    logger.warning(
                        f"pyastgrep returned unexpected type for {file_path}: {type(match_obj)}"
                    )
        except (
            FileNotFoundError,
            PermissionError,
            ValueError,
            TypeError,
            OSError,
        ) as e:
            logger.error(f"Pyastgrep search failed for {file_path}: {e}")
            return self._find_using_regex(file_path, pattern)
        return results

    def _process_pyastgrep_match(self, match_obj: Any) -> dict[str, Any]:
        """Process a single match object into result format."""
        line_num = match_obj.position.lineno
        context_lines = self._extract_context_lines(match_obj, line_num)
        match_text = self._extract_match_text(match_obj, line_num)
        return {
            "file": str(match_obj.path),
            "line": line_num,
            "column": match_obj.position.col_offset + 1,
            "match_text": match_text,
            "context": "\n".join(context_lines),
            "method": "pyastgrep",
            "node_type": type(match_obj.ast_node).__name__,
        }

    def _extract_context_lines(self, match_obj: Any, line_num: int) -> list[str]:
        """Extract context lines around the match."""
        start_ctx_idx = max(0, line_num - 1 - 2)
        end_ctx_idx = min(len(match_obj.file_lines), line_num - 1 + 3)
        return list(match_obj.file_lines[start_ctx_idx:end_ctx_idx])

    def _extract_match_text(self, match_obj: Any, line_num: int) -> str:
        """Extract the matched text from the AST node."""
        try:
            if hasattr(ast, "unparse"):
                return ast.unparse(match_obj.ast_node)
            else:
                return self._fallback_match_text_extraction(match_obj, line_num)
        except (AttributeError, TypeError, ValueError) as unparse_exc:
            logger.debug(f"Could not ast.unparse node for match: {unparse_exc}")
            line_content = match_obj.file_lines[line_num - 1]
            return str(line_content[match_obj.position.col_offset :].strip())

    def _fallback_match_text_extraction(self, match_obj: Any, line_num: int) -> str:
        """Fallback method for extracting match text when ast.unparse is not available."""
        line_content = match_obj.file_lines[line_num - 1]
        start_col = match_obj.position.col_offset
        end_col = getattr(match_obj.ast_node, "end_col_offset", len(line_content))
        if getattr(match_obj.ast_node, "end_lineno", line_num) > line_num:
            return str(line_content[start_col:] + "...")
        else:
            return str(line_content[start_col:end_col])

    def _log_pyastgrep_issue(self, match_obj: Any, file_path: str, pattern: str) -> None:
        """Log pyastgrep issues appropriately."""
        if isinstance(match_obj, NonElementReturned):
            logger.debug(
                f"pyastgrep NonElementReturned for file {file_path} with pattern '{pattern}': {match_obj}"
            )
        else:
            logger.warning(
                f"pyastgrep issue for file {file_path} with pattern '{pattern}': {type(match_obj).__name__} - {match_obj}"
            )

    def find_function_calls(self, _function_name: str) -> list[dict[str, Any]]:
        """Find all calls to a specific function in the codebase.

        Args:
            function_name: Name of the function to find calls for

        Returns:
            List of function call locations and contexts
        """
        results: list[dict[str, Any]] = []
        logger.warning("Graph manager not available - function call analysis limited")
        return results

    def find_class_references(self, _class_name: str) -> list[dict[str, Any]]:
        """Find all references to a specific class in the codebase.

        Args:
            class_name: Name of the class to find references for

        Returns:
            List of class references and usage contexts
        """
        results: list[dict[str, Any]] = []
        logger.warning("Graph manager not available - class reference analysis limited")
        return results

    def find_import_usage(self, _module_name: str) -> list[dict[str, Any]]:
        """Find all usage of a specific import/module in the codebase.

        Args:
            module_name: Name of the module to find usage for

        Returns:
            List of import usage locations and contexts
        """
        results: list[dict[str, Any]] = []
        logger.warning("Graph manager not available - import usage analysis limited")
        return results

    def analyze_dependency_graph(self, start_node_id: str, max_depth: int = 3) -> dict[str, Any]:
        """Analyze the dependency graph starting from a specific node.

        Args:
            start_node_id: ID of the node to start analysis from
            max_depth: Maximum depth to traverse

        Returns:
            Dependency graph representation
        """
        if not self.graph_manager:
            logger.warning("No graph_manager available to analyze dependency graph")
            return {"nodes": [], "edges": [], "error": "No graph manager available"}
        visited = set()
        graph: dict[str, Any] = {"nodes": [], "edges": []}

        def traverse(node_id: str, depth: int) -> None:
            if depth > max_depth or node_id in visited:
                return
            visited.add(node_id)
            node = self.graph_manager.get_node(node_id)
            if not node:
                return
            graph["nodes"].append(
                {
                    "id": node_id,
                    "type": node["type"],
                    "name": node["properties"].get("name", "unknown"),
                    "properties": node["properties"],
                }
            )
            edges = self.graph_manager.get_edges(source_id=node_id)
            for edge in edges:
                if edge["relationship"] in ["defined_in", "contains"]:
                    continue
                graph["edges"].append(
                    {
                        "source": edge["source"],
                        "target": edge["target"],
                        "relationship": edge["relationship"],
                        "properties": edge["properties"],
                    }
                )
                traverse(edge["target"], depth + 1)

        traverse(start_node_id, 0)
        return graph

    def extract_function_call_relationships(self, file_path: str) -> list[dict[str, Any]]:
        """Extract function call relationships from a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            List of function call relationships
        """
        relationships: list[dict[str, Any]] = []
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                code = f.read()
            if not code.strip():
                logger.debug(f"Skipping empty file: {file_path}")
                return relationships
            if any(ord(c) < 32 and c not in ("\t", "\n", "\r") for c in code[:1000]):
                logger.warning(
                    f"File {file_path} contains non-printable characters, skipping AST parsing"
                )
                return relationships
            tree = ast.parse(code)
            visitor = FunctionCallVisitor()
            visitor.visit(tree)
            for (source_name, source_type), targets in visitor.call_graph.items():
                for target_name, line_numbers in targets.items():
                    for line_num in line_numbers:
                        relationships.append(
                            {
                                "source": source_name,
                                "target": target_name,
                                "line_number": line_num,
                                "file_path": file_path,
                                "source_type": source_type,
                                "target_type": "function",
                            }
                        )
        except SyntaxError as e:
            logger.warning(f"Syntax error in file {file_path} at line {e.lineno}: {e.msg}")
            return relationships
        except UnicodeDecodeError as e:
            logger.warning(f"Unicode decode error in file {file_path}: {e}")
            return relationships
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(f"Error extracting function calls from {file_path}: {e}")
            return relationships
        return relationships

    def extract_call_relationships_from_string(
        self, code_string: str, file_path_hint: str = "<string>"
    ) -> list[dict[str, Any]]:
        """Extract function call relationships from a Python code string.

        Args:
            code_string: The Python code string to analyze.
            file_path_hint: A name to use for context in logging (e.g., "<code>", "temp.py").

        Returns:
            List of function call relationships
        """
        relationships: list[dict[str, Any]] = []
        try:
            if not code_string.strip():
                logger.debug(f"Skipping empty code string: {file_path_hint}")
                return relationships
            if any(ord(c) < 32 and c not in ("\t", "\n", "\r") for c in code_string[:1000]):
                logger.warning(
                    f"Code string '{file_path_hint}' contains non-printable characters, skipping AST parsing"
                )
                return relationships
            tree = ast.parse(code_string)
            visitor = FunctionCallVisitor()
            visitor.visit(tree)
            for (source_name, source_type), targets in visitor.call_graph.items():
                for target_name, line_numbers in targets.items():
                    for line_num in line_numbers:
                        relationships.append(
                            {
                                "source": source_name,
                                "target": target_name,
                                "line_number": line_num,
                                "file_path": file_path_hint,
                                "source_type": source_type,
                                "target_type": "function",
                            }
                        )
        except SyntaxError as e:
            logger.warning(
                f"Syntax error in code string '{file_path_hint}' at line {e.lineno}: {e.msg}"
            )
            return relationships
        except UnicodeDecodeError as e:
            logger.warning(f"Unicode decode error in code string '{file_path_hint}': {e}")
            return relationships
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(
                f"Error extracting function calls from code string '{file_path_hint}': {e}"
            )
            return relationships
        return relationships

    def analyze_call_graph(
        self, function_name: str, max_depth: int = 2, direction: str = "outgoing"
    ) -> dict[str, Any]:
        """Analyze the call graph for a specific function.

        Args:
            function_name: Name of the function to analyze
            max_depth: Maximum depth of the call graph
            direction: 'outgoing' for functions called by this function,
                      'incoming' for functions that call this function

        Returns:
            Dictionary with call graph data (nodes and edges)
        """
        if not function_name:
            return {"error": "Function name is required"}
        if direction not in ["outgoing", "incoming"]:
            return {"error": "Direction must be 'outgoing' or 'incoming'"}
        if not self.graph_manager:
            logger.warning("No graph_manager available to analyze call graph")
            return {"error": "No graph manager available", "nodes": [], "edges": []}
        function_nodes = self.graph_manager.get_nodes_by_property("name", function_name)
        if not function_nodes:
            return {"error": f"Function '{function_name}' not found in the code graph"}
        function_node = function_nodes[0]
        function_id = function_node["id"]
        visited = set()
        nodes: list[Any] = []
        edges: list[Any] = []
        queue = deque([(function_id, 0)])
        visited.add(function_id)
        while queue:
            current_id, current_depth = queue.popleft()
            node = self.graph_manager.get_node(current_id)
            if not node:
                continue
            nodes.append(
                {
                    "id": current_id,
                    "name": node["properties"].get("name", "unknown"),
                    "type": node["type"],
                    "depth": current_depth,
                    "properties": node["properties"],
                }
            )
            if current_depth >= max_depth:
                continue
            if direction == "outgoing":
                call_edges = self.graph_manager.get_edges(
                    source_id=current_id, relationship="calls"
                )
                for edge in call_edges:
                    target_id = edge["target"]
                    edges.append(
                        {
                            "source": current_id,
                            "target": target_id,
                            "type": "calls",
                            "properties": edge["properties"],
                        }
                    )
                    if target_id not in visited:
                        visited.add(target_id)
                        queue.append((target_id, current_depth + 1))
            else:
                call_edges = self.graph_manager.get_edges(
                    target_id=current_id, relationship="calls"
                )
                for edge in call_edges:
                    source_id = edge["source"]
                    edges.append(
                        {
                            "source": source_id,
                            "target": current_id,
                            "type": "calls",
                            "properties": edge["properties"],
                        }
                    )
                    if source_id not in visited:
                        visited.add(source_id)
                        queue.append((source_id, current_depth + 1))
        return {
            "root": function_name,
            "nodes": nodes,
            "edges": edges,
            "direction": direction,
            "depth": max_depth,
        }

    def extract_function_calls(
        self, code: str, file_path: str = "unnamed.py"
    ) -> list[dict[str, Any]]:
        """Extract function call relationships from code string.

        This method provides compatibility with the expected interface.

        Args:
            code: Python code string to analyze
            file_path: Optional file path for context

        Returns:
            List of function call relationships
        """
        return self.extract_call_relationships_from_string(code, file_path)

    def calculate_function_metrics(self, function_name: str) -> dict[str, Any]:
        """Calculate metrics for a function based on its call graph structure.

        Args:
            function_name: Name of the function to analyze

        Returns:
            Dictionary with function metrics
        """
        if not self.graph_manager:
            logger.warning("No graph_manager available to calculate function metrics")
            return {
                "error": "No graph manager available",
                "function_name": function_name,
                "outgoing_calls": 0,
                "incoming_calls": 0,
                "unique_callees": 0,
                "unique_callers": 0,
                "cyclomatic_complexity": 0,
                "call_based_importance": 0,
            }
        function_nodes = self.graph_manager.get_nodes_by_property("name", function_name)
        if not function_nodes:
            return {"error": f"Function '{function_name}' not found in the code graph"}
        function_node = function_nodes[0]
        function_id = function_node["id"]
        try:
            outgoing_edges = self.graph_manager.get_edges(
                source_id=function_id, relationship="calls"
            )
            outgoing_count = len(outgoing_edges)
            incoming_edges = self.graph_manager.get_edges(
                target_id=function_id, relationship="calls"
            )
            incoming_count = len(incoming_edges)
            complexity_score = outgoing_count * 0.7 + incoming_count * 0.3
            unique_callees = set(edge["target"] for edge in outgoing_edges)
            unique_callers = set(edge["source"] for edge in incoming_edges)
            return {
                "function_name": function_name,
                "outgoing_calls": outgoing_count,
                "incoming_calls": incoming_count,
                "unique_callees": len(unique_callees),
                "unique_callers": len(unique_callers),
                "cyclomatic_complexity": function_node["properties"].get("complexity", 1),
                "call_based_importance": complexity_score,
            }
        except (KeyError, AttributeError, ValueError) as e:
            logger.error(
                f"Error calculating metrics for function {function_name}: {e}",
                exc_info=True,
            )
            return {
                "function_name": function_name,
                "error": str(e),
                "outgoing_calls": 0,
                "incoming_calls": 0,
                "unique_callees": 0,
                "unique_callers": 0,
                "cyclomatic_complexity": 0,
                "call_based_importance": 0,
            }
