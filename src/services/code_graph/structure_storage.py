"""Module for storing and managing code structure information in a database."""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import logging
import os
from typing import Any

import astor
from psycopg.types.json import Jsonb

from app.core.config.core_config_settings import get_settings
from db.facade import GraphDatabaseConnection

logger = logging.getLogger(__name__)


class CodeStructureStorage:
    """Stores code structure information in SQLite database for analysis and querying."""

    def __init__(
        self,
        db_connection: GraphDatabaseConnection,
        custom_settings: dict[str, Any] | None = None,
    ):
        """Initialize with direct database connection and optional custom settings."""
        self.db = db_connection
        self.conn = db_connection.get_connection()
        self.settings = get_settings()
        self.analysis_settings: dict[str, Any]
        if custom_settings:
            self.analysis_settings = self._merge_custom_settings(custom_settings)
        else:
            self.analysis_settings = {
                "include_tests_in_graphs": self.settings.include_tests_in_graphs,
                "ignore_patterns": list(self.settings.ignore_patterns),
                "include_patterns": [],
            }

    def _merge_custom_settings(self, custom_settings: dict[str, Any]) -> dict[str, Any]:
        """Merge custom settings with default settings for this analysis."""
        analysis_settings: dict[str, Any] = {
            "include_tests_in_graphs": self.settings.include_tests_in_graphs,
            "ignore_patterns": list(self.settings.ignore_patterns),
            "include_patterns": [],
        }
        if "include_tests" in custom_settings and custom_settings["include_tests"] is not None:
            analysis_settings["include_tests_in_graphs"] = custom_settings["include_tests"]
        if "exclude_patterns" in custom_settings and custom_settings["exclude_patterns"]:
            exclude_patterns = custom_settings["exclude_patterns"]
            ignore_patterns = analysis_settings["ignore_patterns"]
            if not isinstance(ignore_patterns, list):
                analysis_settings["ignore_patterns"] = (
                    list(ignore_patterns) if ignore_patterns else []
                )
                ignore_patterns = analysis_settings["ignore_patterns"]
            if isinstance(exclude_patterns, list):
                ignore_patterns.extend(exclude_patterns)
            else:
                ignore_patterns.append(exclude_patterns)
        if "include_patterns" in custom_settings and custom_settings["include_patterns"]:
            analysis_settings["include_patterns"] = custom_settings["include_patterns"]
        return analysis_settings

    def _should_exclude_path(self, path: str, base_path: str) -> bool:
        """
        Determine if a path should be excluded from analysis based on ignore patterns.

        Args:
            path: Full path to check
            base_path: Base directory being analyzed

        Returns:
            True if path should be excluded
        """
        relative_path = os.path.relpath(path, base_path)
        filename = os.path.basename(path)
        for pattern in list(self.analysis_settings["include_patterns"]):
            if fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(filename, pattern):
                logger.debug(f"Including {relative_path} - matches include pattern: {pattern}")
                return False
        is_test_file = self._is_test_file(path)
        if is_test_file:
            if self.analysis_settings["include_tests_in_graphs"]:
                logger.debug(f"Including test file: {relative_path} (tests enabled)")
                ignore_patterns = self.analysis_settings["ignore_patterns"]
                patterns_list: list[str] = (
                    [str(p) for p in ignore_patterns] if isinstance(ignore_patterns, list) else []
                )
                non_test_patterns = [
                    p for p in patterns_list if not self._is_test_related_pattern(p)
                ]
                for pattern in non_test_patterns:
                    if self._matches_pattern(relative_path, filename, pattern):
                        logger.debug(
                            f"Excluding test file {relative_path} - matches non-test pattern: {pattern}"
                        )
                        return True
                return False
            else:
                logger.debug(f"Excluding test file: {relative_path} (tests disabled)")
                return True
        for pattern in list(self.analysis_settings["ignore_patterns"]):
            if self._matches_pattern(relative_path, filename, pattern):
                logger.debug(f"Excluding {relative_path} - matches pattern: {pattern}")
                return True
        return False

    def _is_test_related_pattern(self, pattern: str) -> bool:
        """Check if an ignore pattern is test-related."""
        test_patterns = ["test*", "*test*", "*_test.py", "test_*.py", "conftest.py"]
        return pattern in test_patterns or "test" in pattern.lower()

    def _matches_pattern(self, relative_path: str, filename: str, pattern: str) -> bool:
        """Check if a path matches a given pattern."""
        if fnmatch.fnmatch(relative_path, pattern):
            return True
        if fnmatch.fnmatch(filename, pattern):
            return True
        path_parts = relative_path.split(os.sep)
        for part in path_parts:
            if fnmatch.fnmatch(part, pattern):
                return True
        return False

    def _is_test_file(self, path: str) -> bool:
        """Detect if a file is likely a test file using heuristics."""
        filename = os.path.basename(path).lower()
        relative_path = path.lower()
        test_indicators = [
            filename.startswith("test_"),
            filename.endswith("_test.py"),
            filename == "conftest.py",
            "/test/" in relative_path,
            "/tests/" in relative_path,
            "/testing/" in relative_path,
            relative_path.startswith("test"),
        ]
        return any(test_indicators)

    async def store_code_structure(self, codebase_path: str) -> dict:
        """
        Analyze a codebase and store its structure in the database using direct SQL.
        Leverages Python's AST module for parsing and direct SQL for storage.
        """
        logger.info(f"Starting code structure analysis for path: {codebase_path}")
        try:
            cursor = await self.conn.execute(
                "SELECT to_regclass('nodes') IS NOT NULL, to_regclass('edges') IS NOT NULL"
            )
            row = await cursor.fetchone()
            if not row or not row[0] or not row[1]:
                logger.error("Critical: nodes or edges tables not found in database!")
                return {"error": "Database tables not initialized"}
        except Exception as e:
            logger.error(f"Failed to verify database connection: {e}")
            return {"error": f"Database connection failed: {e}"}
        logger.info(f"Walking directory tree for: {codebase_path}")
        logger.info(f"Include tests setting: {self.analysis_settings['include_tests_in_graphs']}")
        logger.info(f"Active ignore patterns: {self.analysis_settings['ignore_patterns']}")
        if self.analysis_settings["include_patterns"]:
            logger.info(f"Active include patterns: {self.analysis_settings['include_patterns']}")
        python_files: list[Any] = []
        excluded_files = []
        for root, dirs, files in os.walk(codebase_path):
            dirs[:] = [
                d
                for d in dirs
                if not self._should_exclude_path(os.path.join(root, d), codebase_path)
            ]
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    if self._should_exclude_path(file_path, codebase_path):
                        excluded_files.append(file_path)
                        continue
                    python_files.append(file_path)
        logger.info(
            f"Found {len(python_files)} Python files to analyze (excluded {len(excluded_files)} files)"
        )
        stats: dict[str, Any] = {
            "modules": 0,
            "classes": 0,
            "functions": 0,
            "imports": 0,
            "calls": 0,
            "detailed_nodes_added": 0,
            "edges_added": 0,
            "entities_created": 0,
            "relationships_created": 0,
            "files_analyzed": len(python_files),
            "files_excluded": len(excluded_files),
            "include_tests": self.analysis_settings["include_tests_in_graphs"],
        }
        if excluded_files and logger.isEnabledFor(logging.DEBUG):
            logger.debug("Sample of excluded files:")
            for excluded in excluded_files[:5]:
                logger.debug(f"  - {os.path.relpath(excluded, codebase_path)}")
            if len(excluded_files) > 5:
                logger.debug(f"  ... and {len(excluded_files) - 5} more excluded files")
        processed_files = 0
        for file_path in python_files:
            try:
                logger.debug(
                    f"Processing file ({processed_files + 1}/{len(python_files)}): {file_path}"
                )
                stats_update = await self._analyze_file(file_path, codebase_path)
                for key, value in stats_update.items():
                    if key in stats:
                        stats[key] += value
                    else:
                        logger.debug(
                            f"Unexpected stats key '{key}' with value {value} from file {file_path}"
                        )
                        stats[key] = stats.get(key, 0) + value
                processed_files += 1
                if processed_files % 10 == 0:
                    logger.info(
                        f"Progress: {processed_files}/{len(python_files)} files processed. Stats so far: {stats['entities_created']} entities, {stats['relationships_created']} relationships"
                    )
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                continue
        try:
            logger.info("Committing final transaction...")
            await self.conn.commit()
            cursor = await self.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE code_graph_id = %s",
                (self.db.code_graph_id,),
            )
            node_result = await cursor.fetchone()
            final_node_count = node_result[0] if node_result else 0
            cursor = await self.conn.execute(
                "SELECT COUNT(*) FROM edges WHERE code_graph_id = %s",
                (self.db.code_graph_id,),
            )
            edge_result = await cursor.fetchone()
            final_edge_count = edge_result[0] if edge_result else 0
            logger.info(
                f"Final verification - Database contains: {final_node_count} nodes, {final_edge_count} edges"
            )
            if final_node_count == 0:
                logger.warning("Warning: No nodes were stored in the database!")
            if final_edge_count == 0:
                logger.warning("Warning: No edges were stored in the database!")
            stats["final_node_count"] = final_node_count
            stats["final_edge_count"] = final_edge_count
        except Exception as e:
            logger.error(f"Failed to commit or verify final data: {e}")
        logger.info(f"Code structure analysis completed. Final stats: {stats}")
        return stats

    async def _analyze_file(self, file_path: str, codebase_path: str) -> dict[str, int]:
        """Process a single Python file and store its structure using direct SQL."""
        stats = {
            "modules": 0,
            "classes": 0,
            "functions": 0,
            "entities_created": 0,
            "relationships_created": 0,
        }
        try:
            logger.debug(f"Reading file: {file_path}")
            with open(file_path, encoding="utf-8") as f:
                code = f.read()
            logger.debug(f"Parsing AST for: {file_path}")
            tree = ast.parse(code)
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            relative_path = os.path.relpath(file_path, codebase_path)
            module_id = f"module_{hashlib.md5(relative_path.encode()).hexdigest()[:12]}"  # nosec B324
            module_properties = {
                "name": module_name,
                "path": relative_path,
                "docstring": ast.get_docstring(tree) or "",
                "file_size": (os.path.getsize(file_path) if os.path.exists(file_path) else 0),
            }
            logger.debug(f"Creating module node: {module_id} for file: {relative_path}")
            try:
                await self.conn.execute(
                    """INSERT INTO nodes (code_graph_id, id, type, properties, content_hash,
                                          tenant_id)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       ON CONFLICT (code_graph_id, tenant_id, id) DO UPDATE SET
                           type = EXCLUDED.type,
                           properties = EXCLUDED.properties,
                           content_hash = EXCLUDED.content_hash,
                           updated_at = now()""",
                    (
                        self.db.code_graph_id,
                        module_id,
                        "Module",
                        Jsonb(module_properties),
                        hashlib.md5(code.encode()).hexdigest(),  # nosec B324
                        self.db.tenant_id,
                    ),
                )
                logger.debug(f"Successfully inserted module node: {module_id}")
                stats["modules"] += 1
                stats["entities_created"] += 1
            except Exception as e:
                logger.error(f"Failed to insert module node {module_id}: {e}")
                return stats

            class ASTNodeVisitor(ast.NodeVisitor):
                """AST visitor for extracting code structure elements.

                This visitor traverses Python AST nodes to extract classes, functions,
                imports, and function calls. It creates graph nodes and relationships
                for each discovered code element.

                Attributes:
                    db: Database connection for storing nodes and edges
                    file_id: Unique identifier for the current file
                    relative_path: Relative path of the file being analyzed
                    current_class_id: ID of the class currently being visited
                    current_function_id: ID of the function currently being visited
                    class_count: Number of classes found
                    function_count: Number of functions found
                    import_count: Number of imports found
                    call_count: Number of function calls found
                    detailed_nodes_added: Total detailed nodes added
                    edges_added: Total edges added
                """

                def __init__(
                    self,
                    db_conn: GraphDatabaseConnection,
                    file_id: str,
                    relative_path_str: str,
                ) -> None:
                    self.db = db_conn
                    self.file_id = file_id
                    self.relative_path = relative_path_str
                    self.current_class_id: str | None = None
                    self.current_function_id: str | None = None
                    self.class_count = 0
                    self.function_count = 0
                    self.import_count = 0
                    self.call_count = 0
                    self.detailed_nodes_added = 0
                    self.edges_added = 0

                def _get_node_source_code(self, node: ast.AST) -> str:
                    try:
                        source_code = astor.to_source(node)
                        return str(source_code).strip()
                    except Exception:
                        return f"<{node.__class__.__name__}>"

                def visit_ClassDef(self, node: ast.ClassDef) -> None:
                    """Visit a class definition node in the AST."""
                    self.class_count += 1
                    class_id = f"class_{hashlib.md5(f'{self.relative_path}_{node.name}_{node.lineno}'.encode()).hexdigest()[:12]}"  # nosec B324
                    properties = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "bases": [astor.to_source(base).strip() for base in node.bases],
                        "line_number": node.lineno,
                        "file_path": self.relative_path,
                    }
                    self.db.add_node(class_id, "class", properties)
                    self.detailed_nodes_added += 1
                    for base_node in node.bases:
                        try:
                            if isinstance(base_node, ast.Name):
                                base_name = base_node.id
                            elif isinstance(base_node, ast.Attribute):
                                if hasattr(base_node.value, "id"):
                                    base_name = f"{base_node.value.id}.{base_node.attr}"
                                else:
                                    base_name = self._get_node_source_code(base_node)
                            else:
                                base_name = self._get_node_source_code(base_node)
                            target_class_id = (
                                f"external_class_{hashlib.md5(base_name.encode()).hexdigest()[:12]}"  # nosec B324
                            )
                            self.db.add_node(
                                target_class_id,
                                "external_class",
                                {"name": base_name, "external": True},
                            )
                            self.db.add_edge(class_id, target_class_id, "inherits")
                            self.edges_added += 1
                        except Exception as e:
                            logger.debug(f"Failed to process inheritance for {class_id}: {e}")
                            continue
                    self.current_class_id = class_id
                    self.generic_visit(node)
                    self.current_class_id = None

                def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                    """Visit a function definition node in the AST."""
                    self.function_count += 1
                    func_id = f"func_{hashlib.md5(f'{self.relative_path}_{node.name}_{node.lineno}'.encode()).hexdigest()[:12]}"  # nosec B324
                    properties = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "parameters": [arg.arg for arg in node.args.args],
                        "return_type": (
                            astor.to_source(node.returns).strip() if node.returns else ""
                        ),
                        "line_number": node.lineno,
                        "file_path": self.relative_path,
                        "complexity": self._calculate_complexity(node),
                    }
                    self.db.add_node(func_id, "function", properties)
                    self.detailed_nodes_added += 1
                    if node.returns:
                        return_type_str = self._get_annotation_str(node.returns)
                        if return_type_str:
                            pass
                    parent_id = self.current_class_id if self.current_class_id else self.file_id
                    relationship_type = (
                        "defines_method" if self.current_class_id else "defines_function"
                    )
                    self.db.add_edge(parent_id, func_id, relationship_type)
                    self.edges_added += 1
                    self.current_function_id = func_id
                    self.generic_visit(node)
                    self.current_function_id = None

                def visit_Call(self, node: ast.Call) -> None:
                    """Visit a function call node in the AST."""
                    self.call_count += 1
                    caller_id = (
                        self.current_function_id if self.current_function_id else self.file_id
                    )
                    try:
                        if isinstance(node.func, ast.Name):
                            callee_name = node.func.id
                        elif isinstance(node.func, ast.Attribute):
                            if hasattr(node.func.value, "id"):
                                callee_name = f"{node.func.value.id}.{node.func.attr}"
                            else:
                                callee_name = self._get_node_source_code(node.func)
                        else:
                            callee_name = self._get_node_source_code(node.func)
                        target_id = (
                            f"external_func_{hashlib.md5(callee_name.encode()).hexdigest()[:12]}"  # nosec B324
                        )
                        self.db.add_node(
                            target_id,
                            "external_function",
                            {"name": callee_name, "external": True},
                        )
                        self.db.add_edge(caller_id, target_id, "calls")
                        self.edges_added += 1
                    except Exception as e:
                        logger.debug(
                            f"Failed to process function call in {self.relative_path}: {e}"
                        )
                    self.generic_visit(node)

                def visit_Import(self, node: ast.Import) -> None:
                    """Visit an import statement node in the AST."""
                    self.import_count += 1
                    for alias in node.names:
                        import_id = f"import_{hashlib.md5(f'{self.relative_path}_{alias.name}'.encode()).hexdigest()[:12]}"  # nosec B324
                        properties = {
                            "name": alias.name,
                            "alias": alias.asname,
                            "file_path": self.relative_path,
                            "import_type": "direct",
                        }
                        self.db.add_node(import_id, "import", properties)
                        self.detailed_nodes_added += 1
                        self.db.add_edge(self.file_id, import_id, "has_import")
                        self.edges_added += 1
                    self.generic_visit(node)

                def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                    """Visit an import-from statement node in the AST."""
                    self.import_count += 1
                    for alias in node.names:
                        import_id = f"import_from_{hashlib.md5(f'{self.relative_path}_{node.module}_{alias.name}'.encode()).hexdigest()[:12]}"  # nosec B324
                        properties = {
                            "name": alias.name,
                            "alias": alias.asname,
                            "module": node.module,
                            "file_path": self.relative_path,
                            "import_type": "from",
                        }
                        self.db.add_node(import_id, "import_from", properties)
                        self.detailed_nodes_added += 1
                        self.db.add_edge(self.file_id, import_id, "has_import_from")
                        self.edges_added += 1
                    self.generic_visit(node)

                def _get_annotation_str(self, annotation: ast.AST) -> str:
                    if isinstance(annotation, ast.Name):
                        return annotation.id
                    elif isinstance(annotation, ast.Attribute):
                        value_str = getattr(annotation.value, "id", str(annotation.value))
                        return f"{value_str}.{annotation.attr}"
                    else:
                        return self._get_node_source_code(annotation)

                def _calculate_complexity(self, node: ast.FunctionDef) -> int:
                    complexity = 1
                    for subnode in ast.walk(node):
                        if isinstance(
                            subnode,
                            (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler),
                        ):
                            complexity += 1
                        elif isinstance(subnode, ast.BoolOp):
                            complexity += len(subnode.values) - 1
                    return complexity

            visitor = ASTNodeVisitor(self.db, module_id, relative_path)
            try:
                visitor.visit(tree)
                await self.db.flush_pending_operations()
            except Exception as visitor_error:
                logger.warning(f"AST visitor failed for {file_path}: {visitor_error}")
            return {
                "modules": stats["modules"],
                "classes": getattr(visitor, "class_count", 0),
                "functions": getattr(visitor, "function_count", 0),
                "imports": getattr(visitor, "import_count", 0),
                "calls": getattr(visitor, "call_count", 0),
                "detailed_nodes_added": getattr(visitor, "detailed_nodes_added", 0),
                "edges_added": getattr(visitor, "edges_added", 0),
            }
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Error processing file {file_path}: {str(e)}")
            return stats
        except Exception as e:
            logger.error(f"Unexpected error processing file {file_path}: {str(e)}")
            return stats

    async def get_stored_entities(self, entity_type: str | None = None) -> list[dict[str, Any]]:
        """Retrieve stored entities from the database."""
        if entity_type:
            cursor = await self.conn.execute(
                "SELECT id, type, properties FROM nodes WHERE code_graph_id = %s AND type = %s",
                (self.db.code_graph_id, entity_type),
            )
        else:
            cursor = await self.conn.execute(
                "SELECT id, type, properties FROM nodes WHERE code_graph_id = %s",
                (self.db.code_graph_id,),
            )
        entities: list[Any] = []
        for row in await cursor.fetchall():
            entity = {
                "id": row[0],
                "type": row[1],
                "properties": row[2] or {},
            }
            entities.append(entity)
        return entities

    async def get_entity_relationships(self, entity_id: str) -> list[dict[str, Any]]:
        """Get all relationships for a specific entity."""
        cursor = await self.conn.execute(
            "SELECT source, target, relationship\n               FROM edges\n               WHERE code_graph_id = %s AND (source = %s OR target = %s)",
            (self.db.code_graph_id, entity_id, entity_id),
        )
        relationships: list[Any] = []
        for row in await cursor.fetchall():
            relationships.append({"source": row[0], "target": row[1], "relationship": row[2]})
        return relationships
