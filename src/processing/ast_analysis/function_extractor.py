"""Function extractor module for analyzing Python AST and extracting function signatures.

This module provides comprehensive function signature extraction capabilities including:
- Basic function signature extraction
- Type annotation processing
- Function relationship mapping
- Complexity metrics calculation
- External dependency tracking
- AST visualization for debugging
"""

from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import astor  # For converting AST nodes back to source code
import astpretty  # For readable AST output
from coding_client.utils.ast_debug import ASTDebugMixin, log_ast_for_development

# Conditional import to avoid circular dependencies
try:
    from .directory_tree_analyzer import DirectoryTreeAnalyzer
except ImportError:
    DirectoryTreeAnalyzer: Any = None

logger = logging.getLogger(__name__)


class FunctionExtractor(ASTDebugMixin):
    """Extracts function signatures and details from Python files.

    Provides comprehensive AI-focused analysis including type annotations,
    complexity metrics, and function relationships.
    """

    def __init__(
        self,
        enable_type_inference: bool = True,
        analyze_usage_patterns: bool = True,
    ) -> None:
        """Initialize the function extractor with AST debugging capabilities."""
        super().__init__()
        self.debug_mode = logger.isEnabledFor(logging.DEBUG)
        self.enable_type_inference = enable_type_inference
        self.analyze_usage_patterns = analyze_usage_patterns
        self.function_calls_map: dict[str, set[str]] = {}  # Maps function to called functions
        # Maps a callee's SHORT name to its callers (keyed like function_calls_map).
        self.called_by_map: dict[str, set[str]] = {}
        # Real call sites per callee short name (source text, line, argument shape).
        self.call_sites: dict[str, list[dict[str, Any]]] = {}
        # Import bindings of the current file: local name -> dotted module path.
        self.import_bindings: dict[str, str] = {}
        self.method_nodes: set[int] = set()
        self._source: str = ""

    def extract_function_signatures(self, file_path: str) -> dict:
        """
        Extract precise function signatures from Python files.

        Returns:
            {
                "function_name": {
                    "inputs": ["param1: str", "param2: Optional[int]"],
                    "output": "Dict[str, Any]",
                    "file": "path/to/file.py",
                    "line": 42,
                    "docstring": "Function docstring if available"
                }
            }
        """
        with open(file_path, encoding="utf-8") as f:
            code = f.read()

        try:
            # Use Python's built-in AST module for parsing
            tree = ast.parse(code)

            # Debug AST if in debug mode
            if self.debug_mode:
                log_ast_for_development(tree, f"Parsed AST for {file_path}", logger_name=__name__)  # type: ignore[unknown-argument]  # ty: ignore[unknown-argument]

            signatures: dict[str, Any] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Debug individual function nodes if needed
                    if self.debug_mode:
                        self.debug_ast_node(node, f"Function: {node.name} in {file_path}")  # type: ignore[invalid-argument-type]  # ty: ignore[invalid-argument-type]

                    # Extract function signature details
                    signature = {
                        "inputs": self._extract_parameters(node),
                        "output": self._extract_return_type(node),
                        "file": file_path,
                        "line": node.lineno,
                        "docstring": ast.get_docstring(node),
                    }
                    signatures[node.name] = signature

            return signatures

        except SyntaxError as e:
            # Use syntax error handling from mixin
            debug_info = self.syntax_error_handling(code, file_path, raise_original=False)

            logger.error(
                "Failed to parse %s - Syntax error at line %s: %s",
                file_path,
                e.lineno,
                e.msg,
            )

            # Return error information with context
            return {
                "error": "Syntax error in file",
                "details": {
                    "message": str(e),
                    "line": e.lineno,
                    "offset": e.offset,
                    "file": file_path,
                },
                "debug_info": debug_info,
            }
        except OSError as e:
            logger.exception("File access error processing %s: %s", file_path, e)
            return {"error": f"File access error: {str(e)}", "file": file_path}
        except Exception as e:
            logger.exception("Unexpected error processing %s: %s", file_path, e)
            return {"error": f"Unexpected error: {str(e)}", "file": file_path}

    def _extract_parameters(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """Extract parameters with type annotations if available."""
        try:
            params: list[Any] = []
            for arg in node.args.args:
                if arg.annotation:
                    # Use astor to convert annotation AST to code string
                    annotation = astor.to_source(arg.annotation).strip()
                    params.append(f"{arg.arg}: {annotation}")
                else:
                    params.append(arg.arg)

            # Debug parameter extraction if in debug mode
            if self.debug_mode and params:
                logger.debug("Extracted parameters for %s: %s", node.name, params)

            return params

        except (AttributeError, TypeError, ValueError) as e:
            logger.warning("Error extracting parameters for function %s: %s", node.name, e)
            # Fallback to basic parameter names
            return [arg.arg for arg in node.args.args]

    def _extract_return_type(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
        """Extract return type annotation if available."""
        try:
            if node.returns:
                # Use astor to convert return annotation AST to code string
                return_type = astor.to_source(node.returns).strip()

                # Debug return type extraction if in debug mode
                if self.debug_mode:
                    logger.debug(
                        "Extracted return type for %s: %s",
                        node.name,
                        return_type,
                    )

                return return_type
            return None

        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(
                "Error extracting return type for function %s: %s",
                node.name,
                e,
            )
            return None

    def extract_with_ast_visualization(self, file_path: str) -> dict[str, Any]:
        """
        Extract function signatures with full AST visualization for debugging.

        This method is specifically designed for development and debugging
        purposes, providing detailed AST output using astpretty.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            Dictionary containing signatures and AST debug information
        """
        with open(file_path, encoding="utf-8") as f:
            code = f.read()

        try:
            tree = ast.parse(code)

            # Always provide AST visualization for this method
            logger.debug("=" * 80)
            logger.debug("AST VISUALIZATION FOR: %s", file_path)
            logger.debug("=" * 80)
            logger.debug("Full AST:\n%s", astpretty.pformat(tree))
            logger.debug("=" * 80)

            # Extract function signatures
            signatures: dict[str, Any] = {}
            function_nodes: list[Any] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_nodes.append(node)

                    # Log individual function AST
                    logger.debug("Function: %s", node.name)
                    logger.debug("-" * 40)
                    logger.debug("Function AST:\n%s", astpretty.pformat(node))

                    signature = {
                        "inputs": self._extract_parameters(node),
                        "output": self._extract_return_type(node),
                        "file": file_path,
                        "line": node.lineno,
                        "docstring": ast.get_docstring(node),
                        "ast_info": {
                            "node_type": type(node).__name__,
                            "lineno": node.lineno,
                            "end_lineno": getattr(node, "end_lineno", None),
                            "formatted_ast": astpretty.pformat(node),
                        },
                    }
                    signatures[node.name] = signature

            return {
                "signatures": signatures,
                "function_count": len(function_nodes),
                "ast_debug": {
                    "full_ast": astpretty.pformat(tree),
                    "file_path": file_path,
                    "parse_successful": True,
                },
            }

        except SyntaxError as e:
            # Error visualization
            debug_info = self.syntax_error_handling(code, file_path, raise_original=False)

            logger.error("=" * 80)
            logger.error("SYNTAX ERROR IN: %s", file_path)
            logger.error("=" * 80)
            logger.error("Error: %s", e.msg)
            logger.error("Line: %s, Column: %s", e.lineno, e.offset)
            logger.error("Context:")
            if debug_info is not None:
                logger.error("%s", debug_info["context"])
            else:
                logger.error("No debug context available")
            logger.error("=" * 80)

            return {
                "signatures": {},
                "function_count": 0,
                "error": "Syntax error in file",
                "ast_debug": debug_info,
            }

    def extract_comprehensive_signatures(
        self, file_path: str, repository_path: str | None = None
    ) -> dict[str, Any]:
        """
        Extract comprehensive function signatures with AI-focused analysis.

        This method implements the comprehensive signature analysis as specified in the PRD:
        - Type inference integration
        - Usage pattern analysis
        - Function relationship mapping
        - Complexity metrics
        - External dependency tracking

        Args:
            file_path: Path to the Python file to analyze
            repository_path: Optional path to repository root for cross-file analysis

        Returns:
            Dict containing comprehensive function analysis as per PRD specification
        """
        # Only process Python files
        if not file_path.endswith(".py"):
            return {
                "functions": [],
                "external_dependencies": [],
                "total_functions": 0,
                "typed_functions": 0,
                "function_complexity_distribution": {},
                "error": (
                    f"File {file_path} is not a Python file - FunctionExtractor only supports .py files"
                ),
            }

        with open(file_path, encoding="utf-8") as f:
            code = f.read()

        try:
            tree = ast.parse(code)

            # Build function relationship maps
            self._source = code
            self._build_function_relationships(tree)
            self.import_bindings = self._import_bindings(tree)
            self.top_level_nodes = {id(stmt) for stmt in tree.body}
            self.method_nodes = {
                id(item)
                for cls in ast.walk(tree)
                if isinstance(cls, ast.ClassDef)
                for item in cls.body
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
            }

            functions_data = []
            external_dependencies = set()
            total_functions = 0
            typed_functions = 0

            for node in ast.walk(tree):
                # async defs are functions too: skipping them dropped every FastAPI handler.
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    total_functions += 1

                    # Extract detailed function information
                    function_data = self._extract_comprehensive_function_data(
                        node, file_path, repository_path
                    )

                    # Check if function has type annotations
                    if function_data.get("return_type") or any(
                        param.get("type") and ":" in param.get("type")
                        for param in function_data.get("parameters", [])
                    ):
                        typed_functions += 1

                    # Collect external dependencies
                    external_dependencies.update(function_data.get("external_dependencies", []))

                    functions_data.append(function_data)

            patch_names, unknown_patches = self._patch_targets(tree)

            # Calculate type coverage
            type_coverage = typed_functions / total_functions if total_functions > 0 else 0.0

            return {
                "analysis_id": f"sig_{hash(file_path) % 1000000:06d}",
                "functions": functions_data,
                "type_coverage": round(type_coverage, 2),
                "external_dependencies": sorted(external_dependencies),
                "imports": dict(self.import_bindings),
                "call_sites": {k: list(v) for k, v in self.call_sites.items()},
                "rebound_names": sorted(self._rebound_names(tree)),
                "attribute_stores": sorted(patch_names),
                "unknown_patch_targets": unknown_patches,
                "metadata": {
                    "file_path": file_path,
                    "total_functions": total_functions,
                    "typed_functions": typed_functions,
                    "analysis_timestamp": self._get_timestamp(),
                    "type_inference_enabled": self.enable_type_inference,
                    "usage_patterns_analyzed": self.analyze_usage_patterns,
                },
            }

        except OSError as e:
            logger.exception(
                "File access error in comprehensive signature extraction for %s: %s",
                file_path,
                e,
            )
            return {
                "analysis_id": f"err_{hash(file_path) % 1000000:06d}",
                "error": str(e),
                "functions": [],
                "type_coverage": 0.0,
                "external_dependencies": [],
                "metadata": {"file_path": file_path, "error": True},
            }
        except Exception as e:
            logger.exception(
                "Error in comprehensive signature extraction for %s: %s",
                file_path,
                e,
            )
            return {
                "analysis_id": f"err_{hash(file_path) % 1000000:06d}",
                "error": str(e),
                "functions": [],
                "type_coverage": 0.0,
                "external_dependencies": [],
                "metadata": {"file_path": file_path, "error": True},
            }

    def _extract_comprehensive_function_data(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        repository_path: str | None = None,
    ) -> dict[str, Any]:
        """Extract comprehensive function data as per PRD specification."""
        module_name = self._get_module_name(file_path, repository_path)
        function_name = node.name
        full_function_name = f"{module_name}.{function_name}"

        # Extract detailed parameters and return type
        parameters = self._extract_detailed_parameters(node)
        return_type = self._extract_return_type(node)

        # Build function signature string
        signature = self._build_function_signature(function_name, parameters, return_type)

        # Extract function relationships and metadata
        function_relationships = self._extract_function_relationships(node, full_function_name)

        return {
            "name": function_name,
            "module": module_name,
            "signature": signature,
            "parameters": parameters,
            "return_type": return_type,
            "docstring": ast.get_docstring(node),
            "usage_examples": function_relationships["usage_examples"],
            "called_by": function_relationships["called_by"],
            "calls": function_relationships["calls"],
            "complexity_metrics": self._calculate_complexity_metrics(node),
            "external_dependencies": self._extract_external_dependencies(node),
            "line_number": node.lineno,
            "end_line_number": getattr(node, "end_lineno", None),
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "is_method": id(node) in self.method_nodes,
            # A direct child of the module body: not nested, not under if/try/with/...
            "is_top_level": id(node) in getattr(self, "top_level_nodes", set()),
            "decorators": [self._ast_to_string(dec) for dec in node.decorator_list],
        }

    def _extract_detailed_parameters(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[dict[str, Any]]:
        """Every parameter, in declaration order, with its binding ``kind``.

        ``kind`` is one of positional_only, positional_or_keyword, var_positional,
        keyword_only, var_keyword. Positional-only and keyword-only parameters
        used to be dropped, so ``def kw(a, *, b)`` read as ``def kw(a)``.
        """
        parameters: list[dict[str, Any]] = []
        args = node.args

        def _param(arg: ast.arg, kind: str, default: ast.expr | None) -> dict[str, Any]:
            info: dict[str, Any] = {
                "name": arg.arg,
                "type": (self._ast_to_string(arg.annotation) if arg.annotation else None),
                "required": default is None,
                "description": self._extract_param_description(node, arg.arg),
                "kind": kind,
            }
            if default is not None:
                info["default"] = ast.unparse(default)
            return info

        # Defaults apply to the LAST N of posonlyargs + args taken together.
        positional = [(a, "positional_only") for a in args.posonlyargs]
        positional += [(a, "positional_or_keyword") for a in args.args]
        defaults_offset = len(positional) - len(args.defaults)
        for i, (arg, kind) in enumerate(positional):
            default = args.defaults[i - defaults_offset] if i >= defaults_offset else None
            parameters.append(_param(arg, kind, default))

        if args.vararg:
            parameters.append(
                {
                    "name": f"*{args.vararg.arg}",
                    "type": (
                        self._ast_to_string(args.vararg.annotation)
                        if args.vararg.annotation
                        else None
                    ),
                    "required": False,
                    "description": "Variable length argument list",
                    "kind": "var_positional",
                }
            )

        for arg, kw_default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
            parameters.append(_param(arg, "keyword_only", kw_default))

        if args.kwarg:
            parameters.append(
                {
                    "name": f"**{args.kwarg.arg}",
                    "type": (
                        self._ast_to_string(args.kwarg.annotation)
                        if args.kwarg.annotation
                        else None
                    ),
                    "required": False,
                    "description": "Arbitrary keyword arguments",
                    "kind": "var_keyword",
                }
            )

        return parameters

    def _build_function_relationships(self, tree: ast.AST) -> None:
        """Build caller/callee maps and real call sites, keyed by SHORT function name.

        The maps used to be read back with the module-qualified name, so ``calls``
        and ``called_by`` were empty for every function ever analysed.
        """
        self.function_calls_map.clear()
        self.called_by_map.clear()
        self.call_sites = {}

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            function_name = node.name
            called_functions: set[str] = set()
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                if isinstance(child.func, ast.Name):
                    callee, short = child.func.id, child.func.id
                elif isinstance(child.func, ast.Attribute):
                    callee, short = self._ast_to_string(child.func), child.func.attr
                else:
                    continue
                called_functions.add(callee)
                self.called_by_map.setdefault(short, set()).add(function_name)
                starred = any(isinstance(a, ast.Starred) for a in child.args)
                self.call_sites.setdefault(short, []).append(
                    {
                        "caller": function_name,
                        "callee": callee,
                        "line": child.lineno,
                        "plain_name": isinstance(child.func, ast.Name),
                        "positional": None if starred else len(child.args),
                        "keywords": [kw.arg for kw in child.keywords],
                        "source": self._segment(child),
                    }
                )
            self.function_calls_map[function_name] = called_functions

    def _segment(self, node: ast.AST) -> str:
        """Source text of *node* in the file being analysed ("" when unavailable)."""
        if not self._source:
            return ""
        try:
            return (ast.get_source_segment(self._source, node) or "")[:200]
        except (IndexError, ValueError):
            return ""

    @staticmethod
    def _import_bindings(tree: ast.AST) -> dict[str, str]:
        """Local name -> dotted module path bound by this file's imports."""
        bindings: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        bindings[alias.asname] = alias.name
                    else:
                        root = alias.name.split(".")[0]
                        bindings[root] = root
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for alias in node.names:
                    if alias.name != "*":
                        bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        return bindings

    @staticmethod
    def _rebound_names(tree: ast.Module) -> set[str]:
        """Names this file binds by anything OTHER than a module-level ``def``.

        A plain-name call to one of these may not reach the module-level function
        of that name (an assignment, lambda, nested def, parameter, loop/with/
        except target, class attribute, relative or duplicate import can shadow
        it), so arity is not judged for them. ``"*"`` marks a star import or a
        ``globals()[...]`` write, after which no name in the file is certain.
        ``global x`` inside a function counts, as does the ``x = ...`` it permits.
        """
        module_level = {id(stmt) for stmt in tree.body}
        rebound: set[str] = set()
        import_counts: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
                rebound.add(node.id)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if id(node) not in module_level:
                    rebound.add(node.name)
            elif isinstance(node, ast.ClassDef):
                rebound.add(node.name)
            elif isinstance(node, ast.arg):
                rebound.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                rebound.add(node.name)
            elif isinstance(node, ast.Global | ast.Nonlocal):
                rebound.update(node.names)
            elif isinstance(node, ast.MatchAs | ast.MatchStar) and node.name:
                rebound.add(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                rebound.add(node.rest)
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Store | ast.Del)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in ("globals", "vars", "locals")
            ):
                rebound.add("*")  # globals()[...] = ...: any name may be rebound
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    import_counts[local] = import_counts.get(local, 0) + 1
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        rebound.add("*")
                        continue
                    local = alias.asname or alias.name
                    if node.level or not node.module:  # relative: target unresolved
                        rebound.add(local)
                    import_counts[local] = import_counts.get(local, 0) + 1
        rebound.update(name for name, count in import_counts.items() if count > 1)
        if FunctionExtractor._is_dynamic_module(tree):
            rebound.add("*")
        return rebound

    # Builtins that can rebind or create module names at runtime.
    _DYNAMIC_CALLS = frozenset(
        {
            "setattr",
            "delattr",
            "exec",
            "eval",
            "compile",
            "__import__",
            "import_module",
            "globals",
            "vars",
            "locals",
        }
    )
    # Function attributes whose write changes how a call binds its arguments.
    _SIGNATURE_ATTRS = frozenset(
        {"__defaults__", "__kwdefaults__", "__code__", "__signature__", "__wrapped__"}
    )

    @staticmethod
    def _is_dynamic_module(tree: ast.Module) -> bool:
        """True when this module may (re)bind names in ways the AST cannot follow.

        Deliberately broad: any call to setattr/delattr/exec/eval/compile/
        __import__/import_module/globals/vars/locals, any reference to
        ``sys.modules``, or any store/del of a signature-bearing function
        attribute (``__defaults__``, ``__code__``, ...). A missed arity mismatch
        is acceptable; a false one is not.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else None
                )
                if name in FunctionExtractor._DYNAMIC_CALLS:
                    return True
            elif isinstance(node, ast.Attribute):
                if node.attr == "modules" and (
                    isinstance(node.value, ast.Name) and node.value.id == "sys"
                ):
                    return True
                if (
                    isinstance(node.ctx, ast.Store | ast.Del)
                    and node.attr in FunctionExtractor._SIGNATURE_ATTRS
                ):
                    return True
            elif isinstance(node, ast.ImportFrom) and node.module == "sys":
                if any(alias.name in ("modules", "*") for alias in node.names):
                    return True
        return False

    # Function attributes whose write changes how a call binds (incl. ``__call__``).
    _PATCHED_SIG_ATTRS = _SIGNATURE_ATTRS | {"__call__"}

    @staticmethod
    def _patch_targets(tree: ast.AST) -> tuple[set[str], list[str]]:
        """Names this file may rebind on ANOTHER module, plus unknown module writes.

        Returns ``(names, unknown)``. ``names`` collects every ``X.<name>``
        store/del (``A.f = ...``), the function behind a signature-attribute
        write (``pkg.a.f.__defaults__ = ...`` -> ``f``), constant targets of
        ``setattr``/``delattr`` (incl. ``monkeypatch.setattr``), constant keys
        written through ``X.__dict__[...]`` / ``vars(X)[...]`` / ``globals()[...]``
        and their ``.update(k=...)``, and the last segment of a dotted
        ``mock.patch('a.b.f')`` / ``monkeypatch.setattr('a.b.f', ...)`` target.
        ``unknown`` describes writes whose target name cannot be read from the
        AST on a module-like object; any such write anywhere disables arity
        judging for the whole analysed set. Plain objects (``self``) do not.
        """
        names: set[str] = set()
        unknown: list[str] = []
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.update(a.asname or a.name for a in node.names if a.name != "*")

        def call_name(node: ast.AST) -> str | None:
            if not isinstance(node, ast.Call):
                return None
            func = node.func
            if isinstance(func, ast.Name):
                return func.id
            if isinstance(func, ast.Attribute):
                return func.attr
            return None

        def const_str(node: ast.AST | None) -> str | None:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            return None

        def module_like(node: ast.AST) -> bool:
            while isinstance(node, ast.Attribute):
                node = node.value
            if isinstance(node, ast.Name):
                return node.id in imported
            if isinstance(node, ast.Subscript):  # sys.modules[...]
                value = node.value
                return isinstance(value, ast.Attribute) and value.attr == "modules"
            name = call_name(node)
            if name in ("globals", "import_module", "__import__", "reload"):
                return True
            if name == "vars" and isinstance(node, ast.Call):
                return not node.args or module_like(node.args[0])
            return False

        # Names assigned a module-like value (``m = importlib.import_module(x)``).
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and module_like(node.value):
                imported.update(t.id for t in node.targets if isinstance(t, ast.Name))

        def namespace_owner(node: ast.AST) -> ast.AST | None:
            """The object whose namespace ``node`` is (``X.__dict__``/``vars(X)``/``globals()``)."""
            if isinstance(node, ast.Attribute) and node.attr == "__dict__":
                return node.value
            name = call_name(node)
            if name == "globals":
                return node
            if name == "vars" and isinstance(node, ast.Call):
                return node.args[0] if node.args else node
            return None

        def last_segment(node: ast.AST) -> str | None:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                return node.attr
            if call_name(node) == "getattr" and isinstance(node, ast.Call) and len(node.args) > 1:
                return const_str(node.args[1])
            return None

        def where(node: ast.AST) -> str:
            return f"line {getattr(node, 'lineno', '?')}: {ast.unparse(node)[:80]}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store | ast.Del):
                names.add(node.attr)
                if node.attr in FunctionExtractor._PATCHED_SIG_ATTRS:
                    target = last_segment(node.value)
                    if target:
                        names.add(target)
                    elif module_like(node.value) or (
                        isinstance(node.value, ast.Call)
                        and node.value.args
                        and module_like(node.value.args[0])
                    ):
                        unknown.append(where(node))
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store | ast.Del):
                owner = namespace_owner(node.value)
                is_sys_modules = isinstance(node.value, ast.Attribute) and (
                    node.value.attr == "modules"
                )
                if owner is not None:
                    key = const_str(node.slice)
                    if key is not None:
                        names.add(key)
                    elif owner is node.value or module_like(owner):
                        unknown.append(where(node))
                elif is_sys_modules:
                    unknown.append(where(node))  # replaces a whole module
            elif isinstance(node, ast.Call):
                name = call_name(node)
                args = node.args
                if name in ("setattr", "delattr"):
                    first = const_str(args[0]) if args else None
                    if first is not None:  # monkeypatch.setattr('pkg.a.f', ...)
                        names.add(first.rsplit(".", 1)[-1])
                    elif len(args) > 1:
                        key = const_str(args[1])
                        if key is not None:
                            names.add(key)
                        elif module_like(args[0]):
                            unknown.append(where(node))
                elif name == "patch":  # mock.patch('pkg.a.f', ...)
                    target = const_str(args[0]) if args else None
                    if target is not None:
                        names.add(target.rsplit(".", 1)[-1])
                elif name == "object" and len(args) > 1:  # mock.patch.object(X, 'f')
                    key = const_str(args[1])
                    if key is not None:
                        names.add(key)
                elif name == "update" and isinstance(node.func, ast.Attribute):
                    owner = namespace_owner(node.func.value)
                    if owner is None:
                        continue
                    opaque = False
                    for kw in node.keywords:
                        if kw.arg is None:
                            opaque = True
                        else:
                            names.add(kw.arg)
                    for arg in args:
                        keys = (
                            [const_str(k) if k is not None else None for k in arg.keys]
                            if (isinstance(arg, ast.Dict))
                            else [None]
                        )
                        if None in keys:
                            opaque = True
                        else:
                            names.update(k for k in keys if k is not None)
                    if opaque and (owner is node.func.value or module_like(owner)):
                        unknown.append(where(node))
        return names, unknown

    def _extract_usage_examples(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, full_function_name: str
    ) -> list[str]:
        """Usage examples that EXIST: docstring examples and real call sites.

        Nothing is synthesised. This used to invent ``name(param, ...)`` from the
        parameter list, which reads like evidence of a call nobody wrote.
        """
        examples = []

        docstring = ast.get_docstring(node)
        if docstring and ("example" in docstring.lower() or "usage" in docstring.lower()):
            for line in docstring.split("\n"):
                if full_function_name.split(".")[-1] in line and ("(" in line and ")" in line):
                    examples.append(line.strip())

        if self.analyze_usage_patterns:
            for site in self.call_sites.get(node.name, []):
                if site["plain_name"] and site["source"]:
                    examples.append(site["source"])

        return list(dict.fromkeys(examples))[:3]

    def _calculate_complexity_metrics(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> dict[str, int]:
        """Calculate complexity metrics for function."""
        cyclomatic_complexity = 1  # Base complexity
        cognitive_complexity = 0

        for child in ast.walk(node):
            # Cyclomatic complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                cyclomatic_complexity += 1
            elif isinstance(child, ast.BoolOp):
                cyclomatic_complexity += len(child.values) - 1

            # Cognitive complexity (streamlined)
            if isinstance(child, (ast.If, ast.While, ast.For)):
                cognitive_complexity += 1
            elif isinstance(child, (ast.Try, ast.ExceptHandler)):
                cognitive_complexity += 1

        return {
            "cyclomatic": cyclomatic_complexity,
            "cognitive": cognitive_complexity,
        }

    def _extract_external_dependencies(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[str]:
        """Imported modules (and names from them) this function references.

        Only names bound by an ``import`` count. It used to list every attribute
        chain (``item.strip``, ``self.x``), i.e. method calls, not dependencies.
        Whether a module is in-repo or third-party is decided by the caller,
        which sees the whole tree.
        """
        dependencies: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in self.import_bindings:
                dependencies.add(self.import_bindings[child.id])
        return sorted(dependencies)

    def _build_function_signature(
        self,
        function_name: str,
        parameters: list[dict[str, Any]],
        return_type: str | None,
    ) -> str:
        """Build function signature string from components.

        Renders the ``/`` and ``*`` markers from each parameter's ``kind`` so a
        positional-only or keyword-only parameter is not shown as an ordinary one.
        """
        param_strs = []
        kinds = [param.get("kind") for param in parameters]
        has_var_positional = "var_positional" in kinds
        for idx, param in enumerate(parameters):
            kind = kinds[idx]
            if kind == "keyword_only" and not has_var_positional:
                if idx == 0 or kinds[idx - 1] != "keyword_only":
                    param_strs.append("*")
            if param.get("type"):
                param_str = f"{param['name']}: {param['type']}"
                if param.get("default"):
                    param_str += f" = {param['default']}"
            else:
                param_str = param["name"]
                if param.get("default"):
                    param_str += f"={param['default']}"
            param_strs.append(param_str)
            next_kind = kinds[idx + 1] if idx + 1 < len(kinds) else None
            if kind == "positional_only" and next_kind != "positional_only":
                param_strs.append("/")

        signature = f"def {function_name}({', '.join(param_strs)})"
        if return_type:
            signature += f" -> {return_type}"
        return signature

    def _extract_function_relationships(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, full_function_name: str
    ) -> dict[str, list[str]]:
        """Extract function relationships and usage examples."""
        return {
            "usage_examples": self._extract_usage_examples(node, full_function_name),
            "called_by": sorted(self.called_by_map.get(node.name, set())),
            "calls": sorted(self.function_calls_map.get(node.name, set())),
        }

    def _extract_param_description(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, param_name: str
    ) -> str | None:
        """Extract parameter description from docstring."""
        docstring = ast.get_docstring(node)
        if not docstring:
            return None

        # Basic pattern matching for parameter descriptions
        lines = docstring.split("\n")
        for line in lines:
            if param_name in line and ":" in line:
                return line.split(":", 1)[1].strip()

        return None

    def _get_module_name(self, file_path: str, repository_path: str | None = None) -> str:
        """Get module name from file path."""
        if repository_path:
            relative_path = os.path.relpath(file_path, repository_path)
        else:
            relative_path = file_path

        # Convert path to module name
        module_parts = Path(relative_path).with_suffix("").parts
        if module_parts[0] == "src":
            module_parts = module_parts[1:]

        return ".".join(module_parts)

    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string representation."""
        try:
            return astor.to_source(node).strip()
        except (AttributeError, TypeError, ValueError):  # fmt: skip
            return str(type(node).__name__)

    def _get_timestamp(self) -> str:
        """Get current timestamp for analysis metadata."""
        return datetime.now(UTC).isoformat()

    def run_type_inference(self, file_path: str) -> dict[str, Any]:
        """Run the ty type checker on file (GitLab JSON diagnostics)."""
        if not self.enable_type_inference:
            return {"type_inference": "disabled"}

        try:
            result = subprocess.run(
                ["ty", "check", "--output-format", "gitlab", file_path],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            if result.stdout:
                try:
                    return {"type_inference": json.loads(result.stdout)}
                except json.JSONDecodeError:
                    return {"type_inference": result.stdout}

            return {"type_inference": "no_output"}

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning("Type inference failed for %s: %s", file_path, e)
            return {"type_inference": f"error: {str(e)}"}

    def analyze_with_directory_context(
        self, file_path: str, project_root: str | None = None
    ) -> dict[str, Any]:
        """
        Extract function signatures with directory tree context.

        Args:
            file_path: Path to the Python file to analyze
            project_root: Root path of the project for directory analysis

        Returns:
            Dictionary containing function analysis with directory context
        """
        try:
            # Get standard function analysis
            function_analysis = self.extract_comprehensive_signatures(file_path, project_root)

            if project_root:
                # Add directory tree context
                if DirectoryTreeAnalyzer is None:
                    raise ImportError("DirectoryTreeAnalyzer not available")

                tree_analyzer = DirectoryTreeAnalyzer()
                directory_analysis = tree_analyzer.analyze_directory(Path(project_root))

                # Find the file's position in the directory tree
                relative_path = os.path.relpath(file_path, project_root)
                file_context = self._find_file_in_tree(directory_analysis["tree"], relative_path)

                function_analysis["directory_context"] = {
                    "relative_path": relative_path,
                    "file_context": file_context,
                    "project_statistics": directory_analysis["statistics"],
                    "language_distribution": directory_analysis["language_distribution"],
                }

            return function_analysis

        except OSError as e:
            logger.error("File access error analyzing with directory context: %s", e)
            return {
                "error": str(e),
                "file_path": file_path,
                "functions": [],
                "directory_context": None,
            }
        except Exception as e:
            logger.error("Error analyzing with directory context: %s", e)
            return {
                "error": str(e),
                "file_path": file_path,
                "functions": [],
                "directory_context": None,
            }

    def _find_file_in_tree(self, tree: dict[str, Any], target_path: str) -> dict[str, Any] | None:
        """Find a file's context within the directory tree."""
        path_parts = Path(target_path).parts
        current = tree

        for part in path_parts[:-1]:  # Navigate to parent directory
            if current.get("type") == "directory" and "children" in current:
                for child in current["children"]:
                    if child.get("name") == part and child.get("type") == "directory":
                        current = child
                        break
                else:
                    return None
            else:
                return None

        # Find the target file
        if current.get("type") == "directory" and "children" in current:
            for child in current["children"]:
                if child.get("name") == path_parts[-1] and child.get("type") == "file":
                    return child

        return None
