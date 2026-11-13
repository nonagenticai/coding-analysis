"""
Go language processor using Tree-sitter with regex fallback.

Coordinates Go language analysis using specialized modules:
- GoSyntaxParser: AST parsing and syntax analysis
- GoModuleManager: Package and module management
- GoDependencyAnalyzer: Import and dependency analysis
- GoQualityAnalyzer: Code quality and complexity metrics
- GoTestExecutor: Test execution and results analysis

Supports both Tree-sitter AST parsing and regex-based fallback parsing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    CoreComplexityMetrics,
    EntityRelation,
)
from processing.languages.go.go_dependency_analyzer import GoDependencyAnalyzer
from processing.languages.go.go_module_manager import GoModuleManager
from processing.languages.go.go_quality_analyzer import GoQualityAnalyzer

# Import Go-specific modules
from processing.languages.go.go_syntax_parser import GoSyntaxParser
from processing.languages.go.go_test_executor import GoTestExecutor

from ..base import LanguageProcessor, ParseResult, ParserType
from ..registry import register_language
from .go_config import GO_CONFIG

logger = logging.getLogger(__name__)

# Import Tree-sitter components and processor
from processing.parsers.tree_sitter_base import BaseTreeSitterProcessor

try:
    from tree_sitter import Node, Tree  # noqa: F401

    TREE_SITTER_AVAILABLE = True

    # Create GoTreeSitterProcessor that properly inherits from BaseTreeSitterProcessor
    class GoTreeSitterProcessor(BaseTreeSitterProcessor):
        """Go Tree-sitter processor that inherits from BaseTreeSitterProcessor."""

        pass

except ImportError:
    TREE_SITTER_AVAILABLE = False
    # Tree/Node types not available - use Any in method signatures

    # Create a dummy GoTreeSitterProcessor that inherits from LanguageProcessor
    class GoTreeSitterProcessor(LanguageProcessor):
        """Dummy GoTreeSitterProcessor for when tree-sitter is not available."""

        async def _initialize_impl(self) -> None:
            pass

        async def parse_file(self, file_path: Path, content: str) -> ParseResult:
            """Parse file using fallback method when tree-sitter unavailable."""
            return ParseResult(
                success=True,
                file_path=str(file_path),
                language="go",
                parser_used=ParserType.REGEX_FALLBACK,
                entities=[],
                relationships=[],
                lines_of_code=len(content.splitlines()),
            )


@register_language(GO_CONFIG)
class GoProcessor(GoTreeSitterProcessor):  # type: ignore[unsupported-base]
    """Go language processor using Tree-sitter."""

    # Go-specific node types for Tree-sitter
    PACKAGE_NODE_TYPES = {"package_clause"}
    IMPORT_NODE_TYPES = {"import_declaration", "import_spec"}
    FUNCTION_NODE_TYPES = {"function_declaration", "method_declaration"}
    STRUCT_NODE_TYPES = {"type_declaration"}
    INTERFACE_NODE_TYPES = {"interface_type"}
    VAR_NODE_TYPES = {"var_declaration", "short_var_declaration"}
    CONST_NODE_TYPES = {"const_declaration"}
    TYPE_NODE_TYPES = {"type_alias", "type_declaration"}

    def __init__(self, config: Any | None = None) -> None:
        """Initialize Go processor with configuration.

        Args:
            config: Optional language configuration. Defaults to GO_CONFIG if not provided.
        """
        super().__init__(config or GO_CONFIG)

        # Initialize specialized modules
        self.syntax_parser = GoSyntaxParser()
        self.module_manager = GoModuleManager()
        self.dependency_analyzer = GoDependencyAnalyzer()
        self.quality_analyzer = GoQualityAnalyzer()
        self.test_executor = GoTestExecutor()

        logger.debug("Go processor initialized with config: %s", config or GO_CONFIG)

    async def _initialize_impl(self) -> None:
        """Initialize Go processor with Tree-sitter support if available.

        Sets up Tree-sitter parser if available, otherwise uses regex fallback.
        """
        if TREE_SITTER_AVAILABLE:
            logger.debug("Initializing Go processor with Tree-sitter support")
            await super()._initialize_impl()

            # Share parser instance with specialized modules
            if self.parser:
                self.syntax_parser.set_parser(self.parser)
                self.module_manager.set_parser(self.parser)
                self.dependency_analyzer.set_parser(self.parser)
        else:
            logger.debug("Tree-sitter not available, using regex fallback for Go parsing")

    async def parse_file(self, file_path: Path, content: str) -> ParseResult:
        """Parse Go source file using Tree-sitter or regex fallback.

        Args:
            file_path: Path to the Go source file
            content: Source code content to parse

        Returns:
            ParseResult containing extracted entities, relationships, and metadata

        Raises:
            ValueError: If content is invalid or parsing fails
        """
        logger.debug("Parsing Go file: %s (%d chars)", file_path, len(content))

        if TREE_SITTER_AVAILABLE and self.parser:
            logger.debug("Using Tree-sitter parser for Go file: %s", file_path)
            result = await super().parse_file(file_path, content)
            return result

        logger.debug("Using regex fallback parser for Go file: %s", file_path)
        return await self._regex_parse(file_path, content)

    async def _regex_parse(self, file_path: Path, content: str) -> ParseResult:
        """Parse Go source using regex patterns as fallback.

        Uses specialized modules to extract Go language constructs:
        - GoModuleManager: Package declarations
        - GoDependencyAnalyzer: Import statements
        - GoSyntaxParser: Functions, structs, interfaces, types, constants, variables
        - GoQualityAnalyzer: Complexity and quality metrics

        Args:
            file_path: Path to the Go source file
            content: Source code content to analyze

        Returns:
            ParseResult with extracted entities and relationships

        Raises:
            ValueError: If content contains invalid syntax patterns
        """
        logger.debug("Starting regex-based Go parsing for: %s", file_path)
        entities = []
        relationships: list[EntityRelation] = []
        errors: list[str] = []
        warnings: list[str] = []
        package_name = None

        try:
            # Extract package declaration
            package_entity = await self.module_manager.extract_package_regex(content, file_path)
            if package_entity:
                entities.append(package_entity)
                package_name = package_entity.name
            else:
                package_name = None

            # Extract imports
            import_entities = await self.dependency_analyzer.extract_imports_regex(
                content, file_path
            )
            entities.extend(import_entities)

            # Extract syntax entities (functions, structs, interfaces, etc.)
            func_entities = await self.syntax_parser.extract_functions_and_methods_regex(
                content, file_path
            )
            entities.extend(func_entities)

            struct_entities = await self.syntax_parser.extract_structs_regex(content, file_path)
            entities.extend(struct_entities)

            interface_entities = await self.syntax_parser.extract_interfaces_regex(
                content, file_path
            )
            entities.extend(interface_entities)

            type_entities = await self.syntax_parser.extract_types_regex(content, file_path)
            entities.extend(type_entities)

            const_entities = await self.syntax_parser.extract_constants_regex(content, file_path)
            entities.extend(const_entities)

            var_entities = await self.syntax_parser.extract_variables_regex(content, file_path)
            entities.extend(var_entities)

            # Build relationships using dependency analyzer
            package_relationships = await self.dependency_analyzer.build_package_relationships(
                entities, file_path
            )
            relationships.extend(package_relationships)

            import_relationships = await self.dependency_analyzer.build_import_relationships(
                entities, file_path
            )
            relationships.extend(import_relationships)

            method_relationships = await self.dependency_analyzer.build_method_relationships(
                entities, file_path
            )
            relationships.extend(method_relationships)

            # Calculate complexity metrics using quality analyzer
            complexity = await self.quality_analyzer.calculate_complexity_metrics(content, entities)

        except Exception as e:
            logger.error("Error during regex parsing of %s: %s", file_path, e)
            errors.append(f"Parsing error: {e}")
            # Return partial results if available
            complexity = CoreComplexityMetrics(
                lines_of_code=len(content.splitlines()),
                logical_lines=0,
                function_count=0,
                class_count=0,
                comment_lines=0,
                blank_lines=0,
                docstring_coverage=0.0,
            )

        logger.info(
            "Go parsing completed for %s: %d entities, %d relationships, %d lines",
            file_path,
            len(entities),
            len(relationships),
            len(content.splitlines()),
        )

        return ParseResult(
            ast=None,
            language="go",
            file_path=str(file_path),
            parser_used=ParserType.REGEX_FALLBACK,
            success=len(errors) == 0,
            entities=entities,
            relationships=relationships,
            lines_of_code=len(content.splitlines()),
            complexity_score=complexity.cyclomatic_complexity,
            parse_time_ms=0.0,
            errors=errors,
            warnings=warnings,
            properties={
                "package_name": package_name,
                "entity_types": list(set(e.type.value for e in entities)),
                "has_methods": any(e.properties.get("is_method") for e in entities),
            },
        )

    # Required abstract methods from TreeSitterProcessor
    async def extract_entities(self, ast: Tree | Any, file_path: Path) -> list[CoreCodeEntity]:
        """Extract code entities from AST or fallback to regex parsing.

        Args:
            ast: Tree-sitter AST tree or None for regex fallback
            file_path: Path to the source file being processed

        Returns:
            List of extracted CoreCodeEntity objects representing Go constructs

        Raises:
            ValueError: If file cannot be read or parsed
        """
        logger.debug("Extracting entities from Go file: %s", file_path)
        try:
            if TREE_SITTER_AVAILABLE and isinstance(ast, Tree):
                content = file_path.read_bytes()

                # Use specialized modules for Tree-sitter parsing
                entities = []

                # Extract packages
                package_entities = await self.module_manager.extract_package_tree_sitter(
                    ast, file_path, content
                )
                entities.extend(package_entities)

                # Extract imports
                import_entities = await self.dependency_analyzer.extract_imports_tree_sitter(
                    ast, file_path, content
                )
                entities.extend(import_entities)

                # Extract syntax entities
                syntax_entities = await self.syntax_parser.extract_entities_tree_sitter(
                    ast, file_path, content
                )
                entities.extend(syntax_entities)

                return entities

            # Fallback to regex parsing
            content_str = file_path.read_text(encoding="utf-8", errors="ignore")
            result = await self._regex_parse(file_path, content_str)
            return result.entities
        except OSError as e:
            logger.error("Failed to read Go file %s: %s", file_path, e)
            raise ValueError(f"Cannot read Go file {file_path}: {e}") from e
        except Exception as e:
            logger.error("Unexpected error extracting entities from %s: %s", file_path, e)
            raise ValueError(f"Failed to extract entities from {file_path}: {e}") from e

    async def extract_relationships(
        self, ast: Tree | Any, entities: list[CoreCodeEntity], file_path: Path
    ) -> list[EntityRelation]:
        """Extract semantic relationships between Go entities.

        Identifies relationships such as:
        - Package contains entities
        - Struct/interface defines methods
        - Import dependencies

        Args:
            ast: Tree-sitter AST tree (unused in current implementation)
            entities: List of extracted entities to analyze
            file_path: Source file path for context

        Returns:
            List of EntityRelation objects representing semantic connections
        """
        logger.debug("Extracting relationships for %d entities in: %s", len(entities), file_path)
        relationships: list[EntityRelation] = []

        # Use dependency analyzer to build all relationships
        package_relationships = await self.dependency_analyzer.build_package_relationships(
            entities, file_path
        )
        relationships.extend(package_relationships)

        import_relationships = await self.dependency_analyzer.build_import_relationships(
            entities, file_path
        )
        relationships.extend(import_relationships)

        method_relationships = await self.dependency_analyzer.build_method_relationships(
            entities, file_path
        )
        relationships.extend(method_relationships)

        return relationships

    # Additional convenience methods for accessing specialized functionality
    async def analyze_code_quality(
        self, content: str, entities: list[CoreCodeEntity], file_path: Path
    ) -> dict:
        """Analyze code quality using the quality analyzer.

        Args:
            content: Source code content
            entities: List of extracted entities
            file_path: Path to the source file

        Returns:
            Code quality analysis results
        """
        return await self.quality_analyzer.analyze_code_quality(content, entities, file_path)

    async def analyze_dependencies(self, entities: list[CoreCodeEntity]) -> dict:
        """Analyze import dependencies using the dependency analyzer.

        Args:
            entities: List of extracted entities

        Returns:
            Dependency analysis results
        """
        return self.dependency_analyzer.analyze_import_dependencies(entities)

    def get_module_info(self, project_root: Path) -> dict | None:
        """Get Go module information from go.mod file.

        Args:
            project_root: Root directory to search for go.mod

        Returns:
            Module information if available
        """
        return self.module_manager.get_module_info_from_go_mod(project_root)

    async def run_tests(self, package_path: Path, **kwargs) -> dict:
        """Run Go tests for a package using the test executor.

        Args:
            package_path: Path to the Go package
            **kwargs: Additional test execution options

        Returns:
            Test execution results
        """
        test_suite = await self.test_executor.run_tests(package_path, **kwargs)
        return {
            "test_suite": test_suite,
            "statistics": self.test_executor.get_test_statistics([test_suite]),
        }
