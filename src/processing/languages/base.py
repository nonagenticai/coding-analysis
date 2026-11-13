"""Base module for language processing."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Import CoreComplexityMetrics for type hint
from models.core_models import CoreComplexityMetrics


class ParserType(StrEnum):
    """Supported parser types for different languages."""

    AST_NATIVE = "ast_native"  # Language's built-in AST (Python ast module)
    TREE_SITTER = "tree_sitter"  # Tree-sitter grammar-based parsing
    # External command-line tools (go tool, javac)
    EXTERNAL_TOOL = "external_tool"
    LANGUAGE_SERVER = "language_server"  # Language Server Protocol integration
    REGEX_FALLBACK = "regex_fallback"  # Basic regex-based parsing


class LanguageFeatures(BaseModel):
    """Feature capabilities of a programming language."""

    model_config = ConfigDict(frozen=True)

    # Object-oriented features
    classes: bool = False
    inheritance: bool = False
    interfaces: bool = False
    abstract_classes: bool = False
    multiple_inheritance: bool = False

    # Functional features
    functions: bool = True
    lambdas: bool = False
    closures: bool = False
    higher_order_functions: bool = False

    # Type system features
    static_typing: bool = False
    generics: bool = False
    type_inference: bool = False
    nullable_types: bool = False

    # Module system features
    modules: bool = True
    packages: bool = False
    namespaces: bool = False
    imports: bool = True

    # Advanced features
    async_support: bool = False
    coroutines: bool = False
    annotations: bool = False
    decorators: bool = False
    macros: bool = False

    # Memory management
    garbage_collected: bool = True
    manual_memory: bool = False
    reference_counting: bool = False


class LanguageConfig(BaseModel):
    """Configuration for a specific programming language."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Language name (lowercase)")
    display_name: str = Field(..., description="Human-readable language name")
    extensions: list[str] = Field(..., description="File extensions (.py, .java, etc.)")
    parser_type: ParserType = Field(..., description="Primary parser type")
    fallback_parsers: list[ParserType] = Field(default_factory=list)

    features: LanguageFeatures = Field(..., description="Language capabilities")

    # Parser-specific configuration
    tree_sitter_grammar: str | None = Field(None, description="Tree-sitter grammar name")
    external_command: str | None = Field(None, description="External parser command")
    language_server_command: str | None = Field(None, description="LSP server command")

    # File processing
    ignore_patterns: list[str] = Field(default_factory=list, description="Glob patterns to ignore")
    test_patterns: list[str] = Field(default_factory=list, description="Test file patterns")

    # Performance tuning
    max_file_size_mb: int = Field(default=10, description="Maximum file size to process")
    timeout_seconds: int = Field(default=30, description="Parser timeout")

    # Dependencies
    required_packages: list[str] = Field(default_factory=list, description="Python packages needed")
    optional_packages: list[str] = Field(
        default_factory=list, description="Optional augmentation packages"
    )


class ParseResult(BaseModel):
    """Result of parsing a source code file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool
    file_path: str
    language: str
    parser_used: ParserType

    # Parsed content
    ast: Any | None = Field(None, description="Abstract syntax tree")
    entities: list[CoreCodeEntity] = Field(default_factory=list)
    relationships: list[EntityRelation] = Field(default_factory=list)

    # Metadata
    lines_of_code: int = 0
    complexity_score: float = 0.0
    parse_time_ms: float = 0.0

    # Error handling
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # Language-specific properties
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Language-specific metadata"
    )


class LanguageProcessor(ABC):
    """Abstract base class for language-specific code processors."""

    def __init__(self, config: LanguageConfig):
        self.config = config
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the processor (download grammars, check dependencies, etc.)."""
        if not self._initialized:
            await self._initialize_impl()
            self._initialized = True

    @abstractmethod
    async def _initialize_impl(self) -> None:
        """Language-specific initialization logic."""

    @abstractmethod
    async def parse_file(self, file_path: Path, content: str) -> ParseResult:
        """Parse a source code file and extract structure."""

    async def parse_file_with_metrics(
        self, file_path: Path, content: str, analysis_id: str | None = None
    ) -> ParseResult:
        """Parse a file with automatic metrics collection and caching."""
        try:
            # Use lazy import to avoid circular dependency
            from services.analysis_cache import get_analysis_cache
            from services.multi_language_metrics import (
                ParserStrategy,
                track_language_operation,
            )

            async with track_language_operation(
                self.config.name, file_path, analysis_id
            ) as metrics:
                time.perf_counter()

                # Try to get cached result first using lazy import
                cache = None
                cached_result = None
                try:
                    cache = get_analysis_cache()
                    cached_result = await cache.get_cached_result(file_path, content)
                except ImportError:
                    # Cache not available, proceed without caching
                    pass

                if cached_result:
                    # Cache hit - use cached result
                    metrics.parse_time_ms = 0.0  # No parsing time
                    metrics.entities_extracted = len(cached_result.entities)
                    metrics.relationships_extracted = len(cached_result.relationships)
                    metrics.lines_of_code = cached_result.lines_of_code
                    metrics.errors_count = len(cached_result.errors)
                    metrics.parser_used = ParserStrategy.CACHED
                    metrics.cache_hit = True

                    return cached_result

                # Cache miss - parse the file
                result = await self.parse_file(file_path, content)

                # Cache successful results if cache is available
                if result.success and cache is not None:
                    try:
                        await cache.cache_result(file_path, content, result)
                    except (ImportError, AttributeError):  # fmt: skip
                        # Cache not available, skip caching
                        pass

                # Update metrics from result
                metrics.parse_time_ms = result.parse_time_ms
                metrics.entities_extracted = len(result.entities)
                metrics.relationships_extracted = len(result.relationships)
                metrics.lines_of_code = result.lines_of_code
                metrics.errors_count = len(result.errors)
                metrics.cache_hit = False

                # Map parser type to strategy
                if result.parser_used == ParserType.TREE_SITTER:
                    metrics.parser_used = ParserStrategy.TREE_SITTER
                elif result.parser_used == ParserType.REGEX_FALLBACK:
                    metrics.parser_used = ParserStrategy.REGEX_FALLBACK
                elif result.parser_used == ParserType.EXTERNAL_TOOL:
                    metrics.parser_used = ParserStrategy.EXTERNAL_TOOL
                else:
                    metrics.parser_used = ParserStrategy.LEGACY

                return result

        except ImportError:
            # Metrics not available, fall back to normal parsing
            return await self.parse_file(file_path, content)

    @abstractmethod
    async def extract_entities(self, ast: Any, file_path: Path) -> list[CoreCodeEntity]:
        """Extract code entities (classes, functions, etc.) from AST."""

    @abstractmethod
    async def extract_relationships(
        self, ast: Any, entities: list[CoreCodeEntity], file_path: Path
    ) -> list[EntityRelation]:
        """Extract relationships between entities."""

    async def calculate_complexity(self, ast: Any) -> float:
        """Calculate code complexity score (override for language-specific metrics)."""
        return 1.0

    def calculate_basic_complexity_metrics(
        self,
        content: str,
        entities: list[CoreCodeEntity],
        single_line_comment_prefix: str = "//",
        multi_line_comment_marker: str = "/*",
    ) -> CoreComplexityMetrics:
        """Calculate basic complexity metrics common to many languages.

        Args:
            content: Source code content
            entities: List of extracted entities
            single_line_comment_prefix: Prefix for single-line comments (default: "//")
            multi_line_comment_marker: Marker for multi-line comments (default: "/*")

        Returns:
            CoreComplexityMetrics object with basic metrics
        """
        from models.core_models import CoreComplexityMetrics, EntityType

        lines = content.splitlines()
        return CoreComplexityMetrics(
            lines_of_code=len(lines),
            logical_lines=len(
                [
                    line
                    for line in lines
                    if line.strip() and not line.strip().startswith(single_line_comment_prefix)
                ]
            ),
            function_count=len([e for e in entities if e.type == EntityType.FUNCTION]),
            class_count=len([e for e in entities if e.type == EntityType.CLASS]),
            comment_lines=len(
                [
                    line
                    for line in lines
                    if line.strip().startswith(single_line_comment_prefix)
                    or multi_line_comment_marker in line
                ]
            ),
            blank_lines=len([line for line in lines if not line.strip()]),
            docstring_coverage=0.0,
        )

    async def validate_syntax(self, content: str) -> list[str]:
        """Validate syntax and return error messages."""
        return []

    def supports_feature(self, feature: str) -> bool:
        """Check if language supports a specific feature."""
        return getattr(self.config.features, feature, False)


# Forward reference resolution
from models.core_models import CoreCodeEntity, EntityRelation

ParseResult.model_rebuild()
