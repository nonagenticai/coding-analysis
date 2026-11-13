"""Core models for unified multi-language code analysis and entity representation.

This module provides the foundational data structures for code analysis, entities,
and relationships across all supported programming languages. It serves as the
central schema definition for the code graph helper system, supporting both
static analysis and dynamic runtime data.

The core models enable:
- Universal entity representation across programming languages
- Type-safe relationship modeling between code elements
- Comprehensive analysis result aggregation
- Multi-language code complexity measurement
- Security pattern detection and reporting

Key Model Categories:
    Entity Models: CoreCodeEntity, EntityRelation for representing code structure
    Analysis Models: CoreAnalysisResult, CoreFileAnalysisResult for analysis outcomes
    Metrics Models: CoreComplexityMetrics, CodeSecurityPattern for code quality assessment
    Request/Response Models: CodebaseAnalysisRequest, AnalysisResponse for API operations

Language Support:
    - Python: Full AST analysis with type hints and decorators
    - JavaScript/TypeScript: ES6+ features and type annotations
    - Java: Classes, interfaces, and annotations
    - C#: Properties, LINQ, and async/await patterns
    - Go: Goroutines, channels, and interfaces
    - Rust: Traits, lifetimes, and ownership patterns
    - C++: Templates, namespaces, and STL containers

Example:
    ```python
    from core_models import CoreCodeEntity, EntityType, RelationType, EntityRelation

    # Create a function entity
    function = CoreCodeEntity(
        id="func_123",
        name="process_data",
        type=EntityType.FUNCTION,
        language="python",
        file_path="src/processor.py",
        line_number=42,
        signature="def process_data(data: List[Dict]) -> Dict:",
        return_type=TypeInfo(name="Dict", module="typing"),
        visibility="public"
    )

    # Create a relationship
    relation = EntityRelation(
        source_id="module_main",
        target_id="func_123",
        type=RelationType.CALLS,
        weight=0.8
    )
    ```
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict  # PEP 728 support for extra_items

# Configure logging
logger = logging.getLogger(__name__)


# TypedDict classes for structured type safety
class CodeEntityProperties(TypedDict, total=False):
    """Type-safe structure for CoreCodeEntity properties with common language fields."""

    # Core properties
    decorators: list[str]
    annotations: list[str]
    access_modifiers: list[str]
    language_specific: dict[str, Any]
    is_constructor: bool
    is_destructor: bool
    is_property: bool
    is_method: bool
    is_field: bool
    access_level: str
    modifiers: list[str] | dict[str, Any]  # Java uses dict, others use list
    attributes: dict[str, Any]

    # TypeScript/JavaScript specific
    generic_parameters: list[str]
    extends: str
    implements: list[str]
    is_exported: bool
    type_kind: str
    type_definition: str
    is_union_type: bool
    is_conditional_type: bool
    is_mapped_type: bool
    is_const_enum: bool
    parameters: list[dict[str, Any]]
    return_type: str
    is_arrow_function: bool
    module_path: str
    import_path: str
    type_safety_impact: dict[str, Any]

    # Rust specific
    is_public: bool
    use_path: str
    impl_type: str
    trait_name: str
    type_name: str
    node_type: str

    # Python specific
    import_type: str
    alias: str
    module: str
    name: str
    bases: list[str]
    arguments: list[str]

    # Java specific
    is_static: bool
    is_wildcard: bool
    package_name: str
    file_path: str
    is_constant: bool

    # C# specific
    namespace_name: str
    using_directive: str
    is_sealed: bool
    interface_type: bool
    is_readonly: bool
    value_type: bool
    enum_type: bool
    is_virtual: bool
    is_override: bool
    property_type: str
    field_type: str
    is_const: bool

    # Multi-language factory
    file_size: int
    file_extension: str
    minimal_graph: bool
    detected_version: str


class CodeEntityMetadata(TypedDict, total=False):
    """Type-safe structure for CoreCodeEntity metadata."""

    source_hash: str
    last_modified: str
    analysis_version: str
    confidence_score: float
    tags: list[str]
    category: str
    complexity_score: float
    maintainability_index: float
    test_coverage: float
    documentation_coverage: float
    legacy_fields: dict[str, Any]

    # Additional analyzer fields
    language: str
    var_type: str


class RelationProperties(TypedDict, total=False):
    """Type-safe structure for EntityRelation properties."""

    weight: float
    confidence: float
    context: str
    frequency: int
    strength: float
    call_count: int
    import_style: str
    access_pattern: str
    dependency_type: str
    metadata: dict[str, Any]

    # TypeScript analyzer specific
    coupling_strength: str
    is_type_relationship: bool


class RelationMetadata(TypedDict, total=False):
    """Type-safe structure for EntityRelation metadata."""

    source_location: str
    target_location: str
    analysis_confidence: float
    detected_by: str
    verification_status: str
    last_verified: str
    flags: list[str]
    tags: list[str]
    legacy_fields: dict[str, Any]


class DependencyMetadata(TypedDict, total=False):
    """Type-safe structure for DependencyInfoModel metadata."""

    version: str
    resolved_path: str
    package_manager: str
    dependency_scope: str
    is_dev_dependency: bool
    is_optional: bool
    resolution_method: str
    confidence_score: float
    flags: list[str]

    # TypeScript analyzer specific
    full_path: str


class SecurityMetadata(TypedDict, total=False):
    """Type-safe structure for CodeSecurityPattern metadata."""

    rule_id: str
    confidence: float
    impact: str
    remediation: str
    references: list[str]
    tags: list[str]
    custom_fields: dict[str, Any]

    # Analyzer specific
    match_text: str


class AnalysisMetadata(TypedDict, total=False):
    """Type-safe structure for CoreAnalysisResult metadata."""

    analyzer_version: str
    analysis_mode: str
    configuration: dict[str, Any]
    performance_metrics: dict[str, Any]
    quality_metrics: dict[str, Any]
    tags: list[str]
    flags: list[str]
    custom_fields: dict[str, Any]

    # Common analyzer fields
    file_path: str
    directory_path: str
    directory: str
    snippet: bool
    error: str
    error_type: str
    error_timestamp: str
    repo_path: str
    total_files: int
    successful_files: int
    failed_files: list[str]
    files_analyzed: int
    error_count: int
    languages: list[str]
    language: str
    detected_language: str
    snippet_language: str
    has_context: bool

    # Symbolic analyzer fields
    symbolic_analysis: dict[str, Any]
    analyzer: str
    documentation: dict[str, Any]
    original_file: str

    # TypeScript analyzer fields
    parser_used: str
    typescript_version: str
    type_safety: dict[str, Any]
    complexity: dict[str, Any]
    typescript_specific: dict[str, Any]
    analysis_timestamp: str


class RepositorySummary(TypedDict, total=False):
    """Type-safe structure for repository analysis summary."""

    total_files: int
    files_by_language: dict[str, int]
    lines_of_code: int
    complexity_distribution: dict[str, int]
    security_findings: dict[str, int]
    quality_score: float
    maintainability_index: float
    test_coverage: float
    documentation_coverage: float
    dependency_count: int
    external_dependencies: int
    analysis_duration_seconds: float
    timestamp: str

    # Code processor fields
    message: str


class AnalysisPatterns(TypedDict, total=False):
    """Type-safe structure for analysis patterns."""

    pattern_type: str
    pattern_name: str
    confidence: float
    frequency: int
    instances: list[dict[str, str | int | bool | float]]
    locations: list[str]
    description: str
    category: str
    impact: str
    recommendations: list[str]
    tags: list[str]


class CoreArchitectureAnalysis(TypedDict, total=False):
    """Type-safe structure for architecture analysis results."""

    layers: list[dict[str, str | int | bool | float]]
    components: list[dict[str, str | int | bool | float]]
    dependencies: list[dict[str, str | int | bool | float]]
    patterns: list[dict[str, str | int | bool | float]]
    metrics: dict[str, str | int | float]
    quality_indicators: dict[str, str | int | float]
    recommendations: list[str]
    analysis_timestamp: str
    # Extended fields used by analysis_orchestrator
    architectural_style: str
    total_modules: int
    total_components: int
    coupling_score: float
    cohesion_score: float
    complexity_score: float
    patterns_detected: list[str]
    dependencies_count: int
    layer_violations: int


class ImpactAnalysis(TypedDict, total=False):
    """Type-safe structure for impact analysis results."""

    affected_components: list[str]
    risk_level: str
    change_propagation: dict[str, str | int | bool | float]
    test_impact: dict[str, str | int | bool | float]
    performance_impact: dict[str, str | int | bool | float]
    recommendations: list[str]
    confidence_score: float
    analysis_timestamp: str
    # Extended fields used by analysis_orchestrator
    affected_files: int
    breaking_changes: list[str]
    change_complexity: float
    dependencies_affected: list[str]


class StyleScore(TypedDict, total=False):
    """Type-safe structure for style evaluation scores."""

    overall_score: float
    naming_conventions: float
    code_organization: float
    documentation_quality: float
    consistency_score: float
    best_practices: float
    maintainability: float
    readability: float
    detailed_metrics: dict[str, str | int | float]


class ProcessingTimeBreakdown(TypedDict, total=False):
    """Type-safe structure for processing time breakdown."""

    total_time_seconds: float
    parsing_time_seconds: float
    analysis_time_seconds: float
    relationship_extraction_seconds: float
    metrics_calculation_seconds: float
    post_processing_seconds: float
    file_io_seconds: float
    memory_usage_mb: float
    cpu_usage_percent: float
    concurrent_workers: int
    files_per_second: float


# Enums
class EntityType(StrEnum):
    """Universal entity types across all programming languages."""

    # Core entities (universal)
    MODULE = "module"
    FILE = "file"
    FUNCTION = "function"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"

    # Object-oriented entities
    CLASS = "class"
    INTERFACE = "interface"
    TRAIT = "trait"
    ENUM = "enum"
    STRUCT = "struct"
    UNION = "union"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    DESTRUCTOR = "destructor"
    PROPERTY = "property"

    # Functional entities
    CLOSURE = "closure"
    LAMBDA = "lambda"
    GENERATOR = "generator"
    ARROW_FUNCTION = "arrow_function"

    # Language-specific entities
    NAMESPACE = "namespace"
    PACKAGE = "package"
    ANNOTATION = "annotation"
    DECORATOR = "decorator"
    MACRO = "macro"
    GENERIC = "generic"
    TYPE_ALIAS = "type_alias"

    # Documentation entities
    COMMENT = "comment"
    DOCSTRING = "docstring"
    TODO = "todo"


class RelationType(StrEnum):
    """Universal relationship types between entities."""

    # Structural relationships
    CONTAINS = "contains"  # Module contains class
    DEFINES = "defines"  # Class defines method
    DECLARES = "declares"  # Function declares variable

    # Call relationships
    CALLS = "calls"  # Function calls another function
    INVOKES = "invokes"  # Class instantiation
    REFERENCES = "references"  # Variable reference

    # Inheritance relationships
    INHERITS = "inherits"  # Class inheritance
    IMPLEMENTS = "implements"  # Interface implementation
    EXTENDS = "extends"  # Trait/mixin extension

    # Import/dependency relationships
    IMPORTS = "imports"  # Module imports
    DEPENDS_ON = "depends_on"  # Dependency relationship
    EXPORTS = "exports"  # Module exports

    # Type relationships
    INSTANTIATES = "instantiates"  # Variable instantiates type
    CASTS_TO = "casts_to"  # Type casting
    GENERIC_OF = "generic_of"  # Generic type parameter

    # Documentation relationships
    DOCUMENTS = "documents"  # Comment/docstring relationship
    ANNOTATES = "annotates"  # Annotation relationship

    # Legacy compatibility
    INHERITANCE = "inherits"  # Maps to INHERITS
    COMPOSITION = "contains"  # Maps to CONTAINS
    DEPENDENCY = "depends_on"  # Maps to DEPENDS_ON
    CALL = "calls"  # Maps to CALLS
    IMPORT = "imports"  # Maps to IMPORTS
    USAGE = "references"  # Maps to REFERENCES


# Core Entity Models
class SourceLocation(BaseModel):
    """Precise location information within source code."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    line_start: int = Field(..., ge=1)
    line_end: int = Field(..., ge=1)
    column_start: int = Field(..., ge=0)
    column_end: int = Field(..., ge=0)


class TypeInfo(BaseModel):
    """Type information for entities."""

    model_config = ConfigDict(frozen=True)

    name: str
    module: str | None = None
    is_generic: bool = False
    generic_params: list[str] = Field(default_factory=list)
    is_nullable: bool = False
    is_array: bool = False
    array_dimensions: int = 0


class CoreCodeEntity(BaseModel):
    """
    Universal code entity representation for multi-language support.

    Combines the best features from both previous implementations with language-agnostic design.
    """

    id: str = Field(..., description="Unique identifier for the entity")
    name: str = Field(..., description="Entity name")
    type: EntityType = Field(..., description="Type of code entity")
    language: str = Field(..., description="Programming language")

    # Location information (augmented for multi-language)
    file_path: str = Field(..., description="File path where entity is defined")
    line_number: int = Field(..., description="Starting line number")
    end_line_number: int | None = Field(None, description="Ending line number")
    column_start: int | None = Field(None, description="Starting column")
    column_end: int | None = Field(None, description="Ending column")

    # Type information
    return_type: TypeInfo | None = None
    parameter_types: list[TypeInfo] = Field(default_factory=list)

    # Modifiers and attributes (language-agnostic)
    visibility: str | None = Field(None, description="public, private, protected, internal")
    is_static: bool = False
    is_abstract: bool = False
    is_final: bool = False
    is_async: bool = False
    is_generic: bool = False

    # Content and documentation
    code_snippet: str | None = Field(None, description="Code snippet")
    docstring: str | None = Field(None, description="Documentation string")
    signature: str | None = Field(None, description="Function/method signature")
    source_code: str | None = Field(None, description="Full source code")

    # Language-specific properties (Dict for Pydantic compatibility, matches CodeEntityProperties structure)
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Language-specific metadata"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (legacy compatibility)",
    )

    # Timestamps
    created_at: datetime | None = Field(None, description="Creation timestamp")
    updated_at: datetime | None = Field(None, description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)

    def model_post_init(self, __context: Any) -> None:
        """Log entity creation for debugging purposes."""
        logger.debug(
            f"Created CoreCodeEntity: {self.name} ({self.type.value}) in {self.file_path}:{self.line_number}"
        )


class EntityRelation(BaseModel):
    """
    Universal relationship model between two code entities for multi-language support.

    Combines features from both previous implementations with backward compatibility.
    """

    # Core relationship
    source_id: str = Field(..., description="ID of the source entity")
    target_id: str = Field(..., description="ID of the target entity")
    type: RelationType = Field(..., description="Type of relationship")

    # Relationship metadata
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship strength")
    properties: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Context information
    location: SourceLocation | None = None
    context: str | None = None  # Code snippet showing relationship

    # Legacy compatibility fields
    id: str | None = Field(None, description="Unique identifier for the relationship")
    relationship_type: str | None = Field(None, description="Backward compatibility alias")
    line_number: int | None = Field(None, description="Line number where relationship occurs")
    detail: str | None = Field(None, description="Additional relationship details")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (legacy)"
    )
    created_at: datetime | None = Field(None, description="Creation timestamp")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    def __init__(self, **data: Any) -> None:
        # Handle backward compatibility for relationship_type field
        if "relationship_type" in data and "type" not in data:
            data["type"] = data["relationship_type"]
        elif "type" in data and isinstance(data["type"], str):
            # Convert string to enum if needed
            try:
                data["type"] = RelationType(data["type"])
            except ValueError:
                # If conversion fails, keep as string (will be validated by Pydantic)
                pass

        # Generate ID if not provided
        if "id" not in data or data["id"] is None:
            source_id = data.get("source_id", "")
            target_id = data.get("target_id", "")
            rel_type = data.get("type", "")
            if hasattr(rel_type, "value"):
                rel_type = rel_type.value
            data["id"] = f"{source_id}-{rel_type}-{target_id}"

        super().__init__(**data)

        # Log relationship creation for debugging
        logger.debug(
            f"Created EntityRelation: {self.source_id} -> {self.target_id} ({self.type.value})"
        )


# Analysis Support Models
class DependencyInfoModel(BaseModel):
    """Information about code dependencies."""

    module_name: str = Field(..., description="Name of the dependency module")
    import_type: str = Field(..., description="Type of import ('import' or 'from_import')")
    alias: str | None = Field(None, description="Import alias if used")
    line_number: int = Field(0, description="Line number of import")
    is_external: bool = Field(False, description="Whether dependency is external")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def model_post_init(self, __context: Any) -> None:
        """Log dependency creation for debugging purposes."""
        logger.debug(
            f"Created DependencyInfoModel: {self.module_name} ({self.import_type}) - external: {self.is_external}"
        )


class CoreComplexityMetrics(BaseModel):
    """Code complexity measurements for multi-language support."""

    model_config = ConfigDict(frozen=True)

    cyclomatic_complexity: float = 0.0
    cognitive_complexity: float = 0.0
    lines_of_code: int = 0
    logical_lines: int = 0
    parameters_count: int = 0
    returns_count: int = 0
    branches_count: int = 0
    nested_depth: int = 0

    # Extended metrics
    maintainability_index: float = Field(0.0, description="Maintainability index score")
    halstead_volume: float = Field(0.0, description="Halstead volume metric")
    halstead_difficulty: float = Field(0.0, description="Halstead difficulty metric")
    halstead_effort: float = Field(0.0, description="Halstead effort metric")

    # Legacy compatibility
    function_count: int = Field(0, description="Number of functions")
    class_count: int = Field(0, description="Number of classes")
    comment_lines: int = Field(0, description="Lines of comments")
    blank_lines: int = Field(0, description="Blank lines")
    docstring_coverage: float = Field(0.0, description="Docstring coverage percentage")


class CodeSecurityPattern(BaseModel):
    """Security pattern detection result from code analysis."""

    type: str = Field(..., description="Type of security pattern")
    severity: str = Field(..., description="Severity level (low, medium, high, critical)")
    line_number: int = Field(..., description="Line number where pattern was found")
    description: str = Field(..., description="Description of the pattern")
    cwe_id: str | None = Field(None, description="Common Weakness Enumeration ID")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def model_post_init(self, __context: Any) -> None:
        """Log security pattern detection for debugging purposes."""
        logger.warning(
            f"SecurityPattern detected: {self.type} (severity: {self.severity}) at line {self.line_number}"
        )


# Analysis Result Models
class CoreAnalysisResult(BaseModel):
    """Result of code analysis operation."""

    success: bool = Field(..., description="Whether analysis was successful")
    language: str = Field(..., description="Programming language")
    file_path: str = Field(..., description="Path to analyzed file")
    entities: list[CoreCodeEntity] = Field(default_factory=list, description="Discovered entities")
    relationships: list[EntityRelation] = Field(
        default_factory=list, description="Entity relationships"
    )
    dependencies: list[DependencyInfoModel] = Field(
        default_factory=list, description="Dependencies found"
    )
    complexity_metrics: CoreComplexityMetrics | None = Field(None, description="Complexity metrics")
    security_patterns: list[CodeSecurityPattern] = Field(
        default_factory=list, description="Security patterns"
    )
    error: str | None = Field(None, description="Error message if analysis failed")
    errors: list[str] = Field(default_factory=list, description="Error messages")
    warnings: list[str] = Field(default_factory=list, description="Warning messages")
    parse_time_ms: float = Field(0.0, description="Parse time in milliseconds")
    total_lines: int = Field(0, description="Total lines of code")
    timestamp: datetime | None = Field(None, description="Analysis timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def model_post_init(self, __context: Any) -> None:
        """Log analysis result creation for debugging purposes."""
        status = "successful" if self.success else "failed"
        logger.info(
            f"CoreAnalysisResult created: {status} for {self.file_path} ({self.language}) - "
            f"entities: {len(self.entities)}, relationships: {len(self.relationships)}"
        )


class CoreFileAnalysisResult(BaseModel):
    """Analysis result for a single file."""

    file_path: str = Field(..., description="Path to analyzed file")
    language: str = Field(..., description="Programming language")
    analysis_result: CoreAnalysisResult = Field(..., description="Analysis results")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Analysis timestamp",
    )


class RepositoryAnalysis(BaseModel):
    """Analysis results for an entire repository."""

    success: bool = Field(..., description="Whether repository analysis was successful")
    file_results: list[CoreAnalysisResult] = Field(
        default_factory=list, description="File analysis results"
    )
    failed_files: list[dict[str, str]] = Field(
        default_factory=list, description="Files that failed analysis"
    )
    summary: dict[str, Any] = Field(default_factory=dict, description="Analysis summary")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def model_post_init(self, __context: Any) -> None:
        """Log repository analysis completion for debugging purposes."""
        status = "successful" if self.success else "failed"
        total_files = len(self.file_results)
        failed_count = len(self.failed_files)
        logger.info(
            f"RepositoryAnalysis completed: {status} - {total_files} files processed, {failed_count} failed"
        )


class FullAnalysis(BaseModel):
    """Complete analysis results including architecture analysis."""

    success: bool = Field(..., description="Whether full analysis was successful")
    file_results: list[CoreFileAnalysisResult] = Field(
        default_factory=list, description="File analysis results"
    )
    summary: dict[str, Any] = Field(default_factory=dict, description="Analysis summary")
    patterns: list[dict[str, Any]] = Field(default_factory=list, description="Detected patterns")
    architecture_analysis: dict[str, Any] = Field(
        default_factory=dict, description="Architecture analysis"
    )
    error: str | None = Field(None, description="Error message if analysis failed")

    def model_post_init(self, __context: Any) -> None:
        """Log full analysis completion for debugging purposes."""
        status = "successful" if self.success else "failed"
        pattern_count = len(self.patterns)
        logger.info(
            f"FullAnalysis completed: {status} - {len(self.file_results)} files, {pattern_count} patterns detected"
        )


class IncrementalAnalysis(BaseModel):
    """Results of incremental code analysis."""

    success: bool = Field(..., description="Whether incremental analysis was successful")
    changed_files: list[CoreFileAnalysisResult] = Field(
        default_factory=list, description="Changed file results"
    )
    impact_analysis: dict[str, Any] = Field(default_factory=dict, description="Impact analysis")
    error: str | None = Field(None, description="Error message if analysis failed")

    def model_post_init(self, __context: Any) -> None:
        """Log incremental analysis completion for debugging purposes."""
        status = "successful" if self.success else "failed"
        changed_count = len(self.changed_files)
        logger.info(
            f"IncrementalAnalysis completed: {status} - {changed_count} changed files analyzed"
        )


class SnippetEvaluation(BaseModel):
    """Evaluation results for a code snippet."""

    success: bool = Field(..., description="Whether snippet evaluation was successful")
    analysis: CoreAnalysisResult | None = Field(None, description="Analysis results")
    compatibility_score: float = Field(0.0, description="Compatibility score")
    style_score: dict[str, Any] = Field(default_factory=dict, description="Style evaluation")
    pattern_matches: list[dict[str, Any]] = Field(
        default_factory=list, description="Pattern matches"
    )
    recommendations: list[str] = Field(default_factory=list, description="Recommendations")
    error: str | None = Field(None, description="Error message if evaluation failed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def model_post_init(self, __context: Any) -> None:
        """Log snippet evaluation completion for debugging purposes."""
        status = "successful" if self.success else "failed"
        logger.info(
            f"SnippetEvaluation completed: {status} - compatibility: {self.compatibility_score:.2f}"
        )


# Graph and Metrics Models
class GraphMetrics(BaseModel):
    """Metrics for an entity in the code graph."""

    entity_id: str = Field(..., description="Entity identifier")
    entity_name: str = Field(..., description="Entity name")
    degree_centrality: float = Field(0.0, description="Degree centrality metric")
    betweenness_centrality: float = Field(0.0, description="Betweenness centrality metric")
    closeness_centrality: float = Field(0.0, description="Closeness centrality metric")
    pagerank: float = Field(0.0, description="PageRank metric")
    importance_score: float = Field(0.0, description="Combined importance score")


class OverallGraphMetrics(BaseModel):
    """Overall statistics for the entire code graph."""

    node_count: int = Field(0, description="Total number of nodes")
    edge_count: int = Field(0, description="Total number of edges")
    density: float = Field(0.0, description="Graph density")
    average_degree: float = Field(0.0, description="Average degree")
    node_types: dict[str, int] = Field(default_factory=dict, description="Node count by type")
    relationship_types: dict[str, int] = Field(
        default_factory=dict, description="Edge count by type"
    )
    calculation_time_seconds: float = Field(0.0, description="Time taken for calculation")


# Request Models
class CodebaseAnalysisRequest(BaseModel):
    """Request model for codebase analysis operations."""

    directory: str = Field(..., description="Directory to analyze")
    languages: list[str] = Field(["python"], description="Programming languages to analyze")
    max_workers: int | None = Field(None, description="Maximum number of worker processes")
    include_tests: bool = Field(False, description="Whether to include test files")
    ignore_patterns: list[str] = Field(default_factory=list, description="File patterns to ignore")


# Response Models
class AnalysisResponse(BaseModel):
    """Response model for codebase analysis."""

    status: str = Field(..., description="Analysis status")
    directory: str = Field(..., description="Analyzed directory")
    files_processed: int = Field(..., description="Number of files processed")
    files_failed: int = Field(..., description="Number of files that failed")
    total_entities: int = Field(..., description="Total entities found")
    total_relations: int = Field(..., description="Total relationships found")
    processing_time: ProcessingTimeBreakdown = Field(..., description="Processing time breakdown")
    languages: list[str] = Field(..., description="Languages analyzed")

    def model_post_init(self, __context: Any) -> None:
        """Log analysis response creation for debugging purposes."""
        success_rate = (
            (self.files_processed - self.files_failed) / self.files_processed * 100
            if self.files_processed > 0
            else 0
        )
        logger.info(
            f"AnalysisResponse created: {self.status} - {self.files_processed} files processed "
            f"({success_rate:.1f}% success rate), {self.total_entities} entities, "
            f"{self.total_relations} relationships"
        )
