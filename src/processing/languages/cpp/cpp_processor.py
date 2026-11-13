"""
C/C++ language processor using Tree-sitter with regex fallback.

This refactored version uses modular components for different aspects
of C++ code analysis and processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    CoreComplexityMetrics,
    EntityRelation,
)
from processing.languages.cpp.cpp_cmake_manager import CppCMakeManager
from processing.languages.cpp.cpp_dependency_analyzer import (
    CppDependencyAnalyzer,
)
from processing.languages.cpp.cpp_quality_analyzer import CppQualityAnalyzer
from processing.languages.cpp.cpp_syntax_parser import CppSyntaxParser
from processing.languages.cpp.cpp_test_executor import CppTestExecutor

from ..base import LanguageProcessor, ParseResult, ParserType
from ..registry import register_language
from .cpp_config import CPP_CONFIG

# Import Tree-sitter components if available
try:
    from tree_sitter import Node, Tree  # noqa: F401

    from processing.parsers.tree_sitter_base import BaseTreeSitterProcessor

    # Create CppTreeSitterProcessor that properly inherits from BaseTreeSitterProcessor
    class CppTreeSitterProcessor(BaseTreeSitterProcessor):
        """C++ Tree-sitter processor that inherits from BaseTreeSitterProcessor."""

        pass

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

    # Create a dummy CppTreeSitterProcessor that inherits from LanguageProcessor
    class CppTreeSitterProcessor(LanguageProcessor):
        """Dummy CppTreeSitterProcessor for when tree-sitter is not available."""

        async def _initialize_impl(self) -> None:
            pass

        async def parse_file(self, file_path: Path, content: str) -> ParseResult:
            """Parse file fallback when tree-sitter is not available."""
            raise NotImplementedError("Tree-sitter not available")

        async def extract_entities(self, ast: Any, file_path: Path) -> list[CoreCodeEntity]:
            """Extract entities fallback when tree-sitter is not available."""
            return []

        async def extract_relationships(
            self, ast: Any, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Extract relationships fallback when tree-sitter is not available."""
            return []

    # Tree/Node types not available - use Any in method signatures


@register_language(CPP_CONFIG)
class CppProcessor(CppTreeSitterProcessor):  # type: ignore[unsupported-base]
    """C/C++ language processor using modular components."""

    def __init__(self, config=None):
        """Initialize the C++ processor with all specialized components."""
        super().__init__(config or CPP_CONFIG)

        # Initialize specialized components
        self.syntax_parser = CppSyntaxParser()
        self.dependency_analyzer = CppDependencyAnalyzer()
        self.quality_analyzer = CppQualityAnalyzer()
        self.cmake_manager = CppCMakeManager()
        self.test_executor = CppTestExecutor()

    async def _initialize_impl(self) -> None:
        """Initialize C++ processor."""
        if TREE_SITTER_AVAILABLE:
            await super()._initialize_impl()
        else:
            # Fallback initialization without Tree-sitter
            pass

    async def parse_file(self, file_path: Path, content: str) -> ParseResult:
        """Parse C++ file using Tree-sitter or regex fallback.

        Args:
            file_path (Path): The file path to parse.
            content (str): The file content.

        Returns:
            ParseResult: The parsing result with entities and relationships.
        """
        if TREE_SITTER_AVAILABLE and self.parser:
            # Use Tree-sitter parsing
            result = await super().parse_file(file_path, content)
            return (
                result
                if hasattr(result, "success")
                else ParseResult(
                    success=False,
                    file_path=str(file_path),
                    language="cpp",
                    parser_used=ParserType.TREE_SITTER,
                    ast=None,
                    entities=[],
                    relationships=[],
                    errors=["Invalid result from parent parser"],
                )
            )
        else:
            # Use regex-based parsing
            return await self._regex_parse(file_path, content)

    async def _regex_parse(self, file_path: Path, content: str) -> ParseResult:
        """Fallback regex-based parsing for C++."""
        # Use the syntax parser for entity extraction
        entities = await self.syntax_parser.parse_with_regex(file_path, content)

        # Use the dependency analyzer for relationships
        relationships = self.dependency_analyzer.analyze_symbol_dependencies(entities)

        # Calculate complexity metrics using the quality analyzer
        complexity = self.quality_analyzer.calculate_complexity_metrics(content, entities)

        # Determine if this is C or C++
        is_cpp = self.syntax_parser.is_cpp_file(file_path, content)
        lang_name = "cpp" if is_cpp else "c"

        return ParseResult(
            language=lang_name,
            file_path=str(file_path),
            parser_used=ParserType.REGEX_FALLBACK,
            success=True,
            ast=None,
            entities=entities,
            relationships=relationships,
            lines_of_code=len(content.splitlines()),
            complexity_score=complexity.cyclomatic_complexity,
            parse_time_ms=0.0,
        )

    async def extract_entities(self, ast: Any, file_path: Path) -> list[CoreCodeEntity]:
        """Extract code entities from AST (Tree-sitter tree).

        Args:
            ast (Any): The AST (Tree-sitter tree).
            file_path (Path): The file path.

        Returns:
            List[CoreCodeEntity]: The extracted entities.
        """
        if TREE_SITTER_AVAILABLE and isinstance(ast, Tree):
            # Use Tree-sitter parsing
            content_bytes = file_path.read_bytes()
            return await self.syntax_parser.parse_with_tree_sitter(ast, file_path, content_bytes)
        else:
            # Fallback to regex parsing
            content_str = file_path.read_text(encoding="utf-8", errors="ignore")
            return await self.syntax_parser.parse_with_regex(file_path, content_str)

    async def extract_relationships(
        self, ast: Any, entities: list[CoreCodeEntity], file_path: Path
    ) -> list[EntityRelation]:
        """Extract relationships between entities.

        Args:
            ast (Any): The AST.
            entities (List[CoreCodeEntity]): The entities.
            file_path (Path): The file path.

        Returns:
            List[EntityRelation]: The extracted relationships.
        """
        return self.dependency_analyzer.analyze_symbol_dependencies(entities)

    # Additional methods that leverage the specialized components

    def analyze_project_structure(self, project_root: Path) -> dict:
        """Analyze the overall project structure including CMake files."""
        return self.cmake_manager.analyze_project_structure(project_root)

    def analyze_dependencies(self, project_root: Path) -> dict:
        """Analyze project dependencies including libraries and packages."""
        return self.dependency_analyzer.analyze_library_dependencies(project_root)

    def analyze_code_quality(
        self, file_path: Path, content: str, entities: list[CoreCodeEntity]
    ) -> dict:
        """Analyze code quality metrics and detect code smells."""
        return self.quality_analyzer.calculate_maintainability_score(content, entities)

    def discover_tests(self, project_root: Path) -> dict:
        """Discover test files and executables in the project."""
        test_files = self.test_executor.discover_test_files(project_root)
        test_executables = self.test_executor.discover_test_executables(project_root)

        return {
            "test_files": {
                framework: [str(f) for f in files] for framework, files in test_files.items()
            },
            "test_executables": [str(exe) for exe in test_executables],
        }

    def extract_cmake_entities(self, cmake_file: Path) -> list[CoreCodeEntity]:
        """Extract entities from CMake files."""
        return self.cmake_manager.extract_cmake_entities(cmake_file)

    def extract_test_entities(self, test_file: Path, framework: str) -> list[CoreCodeEntity]:
        """Extract test entities from test files."""
        return self.test_executor.extract_test_entities(test_file, framework)

    def analyze_header_dependencies(self, file_path: Path, content: str) -> dict:
        """Analyze header file dependencies and structure."""
        return self.dependency_analyzer.analyze_header_dependencies(file_path, content)

    def detect_build_system(self, project_root: Path) -> dict:
        """Detect the build system used in the project."""
        return self.cmake_manager.detect_build_system(project_root)

    def analyze_code_smells(self, content: str, entities: list[CoreCodeEntity]) -> dict:
        """Analyze code smells in the source code."""
        return self.quality_analyzer.analyze_code_smells(content, entities)

    def find_circular_dependencies(self, relationships: list[EntityRelation]) -> list[list[str]]:
        """Find circular dependencies in the codebase."""
        return self.dependency_analyzer.find_circular_dependencies(relationships)

    def calculate_coupling_metrics(
        self, entities: list[CoreCodeEntity], relationships: list[EntityRelation]
    ) -> dict:
        """Calculate coupling metrics for the codebase."""
        return self.dependency_analyzer.analyze_coupling_metrics(entities, relationships)

    def create_dependency_graph(
        self, entities: list[CoreCodeEntity], relationships: list[EntityRelation]
    ) -> dict:
        """Create a dependency graph representation."""
        return self.dependency_analyzer.create_dependency_graph(entities, relationships)

    def run_tests(self, test_executable: Path, output_format: str = "xml") -> dict:
        """Run a test executable and return results."""
        return self.test_executor.run_tests(test_executable, output_format)

    def generate_test_report(self, test_results: list[dict]) -> dict:
        """Generate a comprehensive test report."""
        return self.test_executor.generate_test_report(test_results)

    # Legacy compatibility methods

    def calculate_basic_complexity_metrics(
        self,
        content: str,
        entities: list[CoreCodeEntity],
        single_line_comment_prefix: str = "//",
        multi_line_comment_marker: str = "/*",
    ) -> CoreComplexityMetrics:
        """Calculate basic complexity metrics (legacy compatibility)."""
        return self.quality_analyzer.calculate_complexity_metrics(content, entities)
