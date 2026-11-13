"""Base class for Tree-sitter based language processors.

This module provides a foundation for implementing language-specific parsers
using the Tree-sitter parsing library. It includes grammar management,
AST traversal utilities, and a consistent interface for entity extraction.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    EntityRelation,
    SourceLocation,
)

from ..languages.base import (
    LanguageConfig,
    LanguageProcessor,
    ParseResult,
    ParserType,
)

logger = logging.getLogger(__name__)

try:
    from tree_sitter import Language, Node, Parser, Tree

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("Tree-sitter not available. Install with: pip install tree-sitter")
    # Define dummy types for IDE support when tree-sitter isn't available
    Node: Any = Any
    Tree: Any = Any


class BaseTreeSitterProcessor(LanguageProcessor, ABC):
    """Base class for Tree-sitter based language processors."""

    # Common Tree-sitter grammar repositories
    GRAMMAR_REPOSITORIES = {
        "python": "https://github.com/tree-sitter/tree-sitter-python",
        "javascript": "https://github.com/tree-sitter/tree-sitter-javascript",
        "typescript": "https://github.com/tree-sitter/tree-sitter-typescript",
        "go": "https://github.com/tree-sitter/tree-sitter-go",
        "java": "https://github.com/tree-sitter/tree-sitter-java",
        "rust": "https://github.com/tree-sitter/tree-sitter-rust",
        "cpp": "https://github.com/tree-sitter/tree-sitter-cpp",
        "c": "https://github.com/tree-sitter/tree-sitter-c",
        "c_sharp": "https://github.com/tree-sitter/tree-sitter-c-sharp",
    }

    def __init__(self, config: LanguageConfig):
        super().__init__(config)
        self.parser: Parser | None = None
        self.language: Language | None = None
        self._grammar_path: Path | None = None

    async def _initialize_impl(self) -> None:
        """Initialize Tree-sitter parser and grammar."""
        if not TREE_SITTER_AVAILABLE:
            raise RuntimeError(
                "Tree-sitter is not available. Install with: pip install tree-sitter"
            )

        if not self.config.tree_sitter_grammar:
            raise ValueError(f"No Tree-sitter grammar specified for {self.config.name}")

        # Initialize parser with tree-sitter 0.25.0 API
        try:
            # Import the language-specific module
            language_module_name = f"tree_sitter_{self.config.tree_sitter_grammar}"
            language_module = importlib.import_module(language_module_name)

            # Get the language from the module and create Language object
            self.language = Language(language_module.language())
            self.parser = Parser(self.language)

        except ImportError as ie:
            # Language module not available
            logger.error(f"Language module {language_module_name} not found: {ie}")
            logger.error(
                f"Please install it with: pip install {language_module_name.replace('_', '-')}"
            )
            raise RuntimeError(
                f"Failed to initialize tree-sitter parser for {self.config.name}: "
                f"Language module {language_module_name} not installed. "
                f"Install with: pip install {language_module_name.replace('_', '-')}"
            )
        except Exception as e:
            logger.error("Failed to initialize tree-sitter parser: %s", e)
            raise RuntimeError(
                f"Failed to initialize tree-sitter parser for {self.config.name}: {e}"
            )

        logger.info("Initialized Tree-sitter parser for %s", self.config.display_name)

    async def _ensure_grammar_available(self) -> None:
        """Ensure Tree-sitter grammar is available (download/compile if needed)."""
        grammar_dir = Path("~/.tree-sitter/grammars").expanduser()
        grammar_dir.mkdir(parents=True, exist_ok=True)

        grammar_name = self.config.tree_sitter_grammar
        if grammar_name is None:
            raise ValueError(f"Tree-sitter grammar not configured for {self.config.name}")

        grammar_file = grammar_dir / f"{grammar_name}.so"

        if not grammar_file.exists():
            await self._download_and_compile_grammar(grammar_dir, grammar_name)

        self._grammar_path = grammar_file

    async def _download_and_compile_grammar(self, grammar_dir: Path, grammar_name: str) -> None:
        """Download and compile Tree-sitter grammar."""

        try:
            if grammar_name not in self.GRAMMAR_REPOSITORIES:
                logger.warning("No repository URL configured for grammar: %s", grammar_name)
                return

            repo_url = self.GRAMMAR_REPOSITORIES[grammar_name]

            # Create temporary directory for compilation
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Clone the grammar repository
                logger.info("Downloading grammar %s from %s", grammar_name, repo_url)
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        repo_url,
                        str(temp_path / grammar_name),
                    ],
                    check=True,
                    capture_output=True,
                )

                grammar_path = temp_path / grammar_name

                # Special handling for TypeScript which has subdirectories
                if grammar_name == "typescript":
                    # TypeScript repo contains typescript and tsx subdirectories
                    ts_path = grammar_path / "typescript"
                    if ts_path.exists():
                        grammar_path = ts_path
                        logger.info("Using TypeScript subdirectory for grammar compilation")

                # Try to compile using tree-sitter CLI if available
                try:
                    subprocess.run(
                        ["tree-sitter", "build-wasm", str(grammar_path)],
                        check=True,
                        capture_output=True,
                        cwd=str(grammar_path),
                    )

                    # Copy compiled grammar to grammar directory
                    compiled_file = grammar_path / f"tree-sitter-{grammar_name}.wasm"
                    if compiled_file.exists():
                        target_file = grammar_dir / f"{grammar_name}.wasm"
                        shutil.copy2(compiled_file, target_file)
                        logger.info("Successfully compiled grammar: %s", target_file)
                        return

                except subprocess.CalledProcessError:
                    logger.warning("tree-sitter CLI not available, trying manual compilation")

                # Fallback: try to compile manually using gcc if available
                try:
                    src_dir = grammar_path / "src"
                    if src_dir.exists():
                        subprocess.run(
                            [
                                "gcc",
                                "-fPIC",
                                "-shared",
                                "-o",
                                str(grammar_dir / f"{grammar_name}.so"),
                                str(src_dir / "parser.c"),
                                "-I",
                                str(src_dir),
                            ],
                            check=True,
                            capture_output=True,
                        )
                        logger.info("Successfully compiled grammar using gcc: %s", grammar_name)
                        return

                except subprocess.CalledProcessError as e:
                    logger.warning("Manual compilation failed: %s", e)

                logger.warning("Could not compile grammar %s", grammar_name)

        except (OSError, subprocess.CalledProcessError) as e:
            logger.error("Error downloading/compiling grammar %s: %s", grammar_name, e)
            # Don't raise - allow fallback to regex parsing

    async def parse_file(self, file_path: Path, content: str) -> ParseResult:
        """Parse file using Tree-sitter.

        Args:
            file_path (Path): The file_path parameter.
            content (str): The content parameter.

        Returns:
            ParseResult: The parseresult result.

        Raises:
            RuntimeError: Raised under certain conditions.

        Example:
            >>> await parse_file("file_path_value", "content_value")"""
        if not self.parser:
            raise RuntimeError("Parser not initialized")

        start_time = asyncio.get_event_loop().time()

        try:
            # Parse content
            tree = self.parser.parse(content.encode("utf-8"))

            # Extract entities and relationships
            entities = await self.extract_entities(tree, file_path)
            relationships = await self.extract_relationships(tree, entities, file_path)

            # Calculate metrics
            complexity = await self.calculate_complexity(tree)
            parse_time = (asyncio.get_event_loop().time() - start_time) * 1000

            return ParseResult(
                success=True,
                file_path=str(file_path),
                language=self.config.name,
                parser_used=ParserType.TREE_SITTER,
                ast=tree,
                entities=entities,
                relationships=relationships,
                lines_of_code=len(content.splitlines()),
                complexity_score=complexity,
                parse_time_ms=parse_time,
            )

        except (OSError, UnicodeDecodeError, RuntimeError, TypeError) as e:
            parse_time = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.error("Failed to parse %s with Tree-sitter: %s", file_path, e)

            return ParseResult(
                success=False,
                file_path=str(file_path),
                language=self.config.name,
                parser_used=ParserType.TREE_SITTER,
                parse_time_ms=parse_time,
                ast=None,
                errors=[str(e)],
            )

    @abstractmethod
    async def extract_entities(
        self,
        ast: Tree,
        file_path: Path,  # noqa: ARG002
    ) -> list[CoreCodeEntity]:
        """Extract entities from Tree-sitter AST.

        Args:
            ast (Tree): The ast parameter.
            file_path (Path): The file_path parameter.

        Returns:
            List[CoreCodeEntity]: The list[corecodeentity] result."""

    @abstractmethod
    async def extract_relationships(
        self,
        ast: Tree,
        entities: list[CoreCodeEntity],
        file_path: Path,  # noqa: ARG002
    ) -> list[EntityRelation]:
        """Extract relationships from Tree-sitter AST.

        Args:
            ast (Tree): The ast parameter.
            entities (List[CoreCodeEntity]): The entities parameter.
            file_path (Path): The file_path parameter.

        Returns:
            List[EntityRelation]: The list[entityrelation] result.

        Example:
            >>> await extract_relationships("ast_value", "entities_value", "file_path_value")
        """

    def _node_to_location(self, node: Node, file_path: Path) -> SourceLocation:
        """Convert Tree-sitter node to SourceLocation."""
        return SourceLocation(
            file_path=str(file_path),
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            column_start=node.start_point[1],
            column_end=node.end_point[1],
        )

    def _get_node_text(self, node: Node, content: bytes) -> str:
        """Extract text content from Tree-sitter node."""
        return content[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

    def _find_nodes_by_type(self, tree: Tree, node_types: set[str]) -> list[Node]:
        """Find all nodes of specified types in the tree."""
        nodes = []

        def visit_node(node: Node):
            """Function visit_node.

            Args:
                node (Node): The node parameter."""
            if node.type in node_types:
                nodes.append(node)
            for child in node.children:
                visit_node(child)

        visit_node(tree.root_node)
        return nodes

    def _get_child_by_type(self, node: Node, child_type: str) -> Node | None:
        """Get first child node of specified type."""
        for child in node.children:
            if child.type == child_type:
                return child
        return None

    def _get_children_by_type(self, node: Node, child_type: str) -> list[Node]:
        """Get all child nodes of specified type."""
        return [child for child in node.children if child.type == child_type]

    def _find_entity_by_location(
        self, entities: list[CoreCodeEntity], node: Node
    ) -> CoreCodeEntity | None:
        """Find entity that corresponds to a Tree-sitter node."""
        node_line = node.start_point[0] + 1
        node_col = node.start_point[1]

        for entity in entities:
            if entity.line_number == node_line and entity.column_start == node_col:
                return entity

        return None

    def _find_containing_method(
        self,
        node: Node,
        entities: list[CoreCodeEntity],
        method_node_types: set[str],
    ) -> CoreCodeEntity | None:
        """Find the method entity that contains the given node."""
        # Walk up the tree to find a method declaration
        current = node.parent
        while current:
            if current.type in method_node_types:
                return self._find_entity_by_location(entities, current)
            current = current.parent

        return None

    async def calculate_complexity(self, ast: Any) -> float:
        """Calculate cyclomatic complexity of the parsed code.

        Args:
            ast: The parsed tree (Tree-sitter Tree object).

        Returns:
            float: The float result.

        Example:
            >>> await calculate_complexity("tree_value")"""
        # Basic complexity calculation based on control flow nodes
        complexity_nodes = {
            "if_statement",
            "while_statement",
            "for_statement",
            "switch_statement",
            "case_statement",
            "catch_clause",
            "conditional_expression",
            "try_statement",
            "do_statement",
            "break_statement",
            "continue_statement",
            "return_statement",
            "throw_statement",
            "assert_statement",
            "lambda_expression",
            "method_declaration",
            "function_definition",
            "class_declaration",
            "interface_declaration",
        }

        complexity = 1.0  # Base complexity

        def visit_node(node: Node):
            """Function visit_node.

            Args:
                node (Node): The node parameter."""
            nonlocal complexity
            if node.type in complexity_nodes:
                complexity += 1.0
            for child in node.children:
                visit_node(child)

        tree: Tree = ast
        visit_node(tree.root_node)
        return complexity
