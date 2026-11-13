"""Models for representing architectural components and configurations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    Any,
    Literal,
    TypeVar,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Import ExternalPackage for dependency loaders
from .package_manager_models import ExternalPackage

logger = logging.getLogger(__name__)


class Component(BaseModel):
    """A coarse-grained architectural component grouping related files.

    Components represent logical groupings of code files and are assigned to architectural
    layers. They follow specific rules regarding dependencies between layers.
    """

    id: str = Field(..., description="Unique identifier for the component")
    name: str = Field(..., description="Human-readable name of the component")
    files: list[str] = Field(..., description="List of files included in this component")
    dependencies: list[str] = Field(
        default_factory=list,
        description="List of component IDs this component depends on",
    )
    layer: str | None = Field(None, description="Architectural layer this component belongs to")
    tags: list[str] = Field(
        default_factory=list,
        description="Optional tags for component classification",
    )

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Validate component ID format - must be alphanumeric with underscores/hyphens only."""
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                f"Component ID must contain only alphanumeric characters, underscores, and hyphens, got '{v}'"
            )
        return v

    @field_validator("layer")
    @classmethod
    def validate_layer_exists(cls, v: str | None, info: Any) -> str | None:
        """Validate that layer exists in repo config if specified."""
        if v is None:
            return v

        # Skip validation if we're not part of a full analysis result
        # This is a complex cross-field validation that depends on config_used
        root = getattr(info.context, "root_model", None)
        if root is None or not hasattr(root, "config_used"):
            return v

        # Verify layer exists in the configuration
        config = getattr(root, "config_used", None)
        if config and config.layers and v not in config.layers:
            logger.warning(f"Layer '{v}' not defined in repository configuration")
            raise ValueError(f"Layer '{v}' not defined in repository configuration")
        return v

    @field_validator("files")
    @classmethod
    def validate_files_not_empty(cls, v: list[str]) -> list[str]:
        """Validate that component has at least one file."""
        if not v:
            raise ValueError("Component must contain at least one file")
        return v

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies_unique(cls, v: list[str]) -> list[str]:
        """Ensure dependencies list has no duplicates."""
        unique_deps = list(set(v))
        if len(unique_deps) != len(v):
            logger.info(f"Removed {len(v) - len(unique_deps)} duplicate dependencies")
        return unique_deps

    @model_validator(mode="after")
    def validate_dependencies(self) -> Component:
        """Validate that dependencies don't include self-reference."""
        dependencies_list = self.dependencies
        if self.id in dependencies_list:
            raise ValueError(f"Component '{self.id}' cannot depend on itself")
        return self

    @model_validator(mode="after")
    def validate_layer_transitions(self) -> Component:
        """Validate layer transitions according to architectural constraints.

        This validator checks if the component's dependencies follow valid layer transition rules
        based on the architectural configuration. It ensures components don't depend on higher layers
        unless explicitly allowed in the configuration.
        """
        # Skip validation if not part of a complete analysis or no layer assigned
        if not self.layer or not hasattr(self, "_ctx") or not self._ctx:
            return self

        root = getattr(self._ctx, "root_model", None)
        if not root or not hasattr(root, "config_used") or not root.config_used:
            return self

        config = root.config_used

        # Get all components to check dependencies
        all_components = getattr(root, "components", [])
        if not all_components:
            return self

        # Get layer hierarchy from config
        layer_hierarchy = self._get_layer_hierarchy(config)
        if not layer_hierarchy or self.layer not in layer_hierarchy:
            return self

        component_layer_level = layer_hierarchy.get(self.layer, float("inf"))

        # Check allowed layer dependencies if configured
        allowed_dependencies = getattr(config, "allowed_layer_dependencies", {})
        allowed_for_layer = allowed_dependencies.get(self.layer, [])

        violations = []
        dependencies_list = self.dependencies
        for dep_id in dependencies_list:
            dep_component = next((c for c in all_components if c.id == dep_id), None)
            if not dep_component or not dep_component.layer:
                continue

            dep_layer_level = layer_hierarchy.get(dep_component.layer, float("inf"))

            # Violation if component depends on a higher layer
            # unless explicitly allowed in config
            if (
                component_layer_level > dep_layer_level
                and dep_component.layer not in allowed_for_layer
            ):
                violations.append(f"{dep_component.id} (layer: {dep_component.layer})")

        if violations:
            violation_msg = f"Component '{self.id}' in layer '{self.layer}' has invalid dependencies on higher layers: {', '.join(violations)}"
            logger.warning(violation_msg)
            # We don't raise an exception here since this is a soft architectural constraint
            # that should be reported but not prevent component creation

        return self

    def _get_layer_hierarchy(self, config: Any) -> dict[str, int]:
        """Get layer hierarchy levels (lower number = lower layer)."""
        # Default hierarchy for common layer names
        default_hierarchy = {
            "infrastructure": 0,
            "data": 1,
            "domain": 2,
            "service": 3,
            "application": 4,
            "presentation": 5,
            "ui": 6,
        }

        # Use configured layers or fall back to defaults
        hierarchy = {}
        layer_names = list(getattr(config, "layers", {}).keys())

        if not layer_names:
            return default_hierarchy

        for i, layer_name in enumerate(layer_names):
            if layer_name.lower() in default_hierarchy:
                hierarchy[layer_name] = default_hierarchy[layer_name.lower()]
            else:
                # Assign levels based on order in config
                hierarchy[layer_name] = i

        return hierarchy

    def get_file_extensions(self) -> set[str]:
        """Get the set of file extensions contained in this component."""
        extensions = set()
        files_list = self.files
        for file_path in files_list:
            parts = file_path.split(".")
            if len(parts) > 1:
                extensions.add(parts[-1])
        return extensions


class ArchitecturalMetric(BaseModel):
    """Package-level metrics following Robert C. Martin's principles."""

    component_id: str
    metric_type: str
    value: float
    description: str

    # Robert C. Martin specific metrics
    instability: float | None = Field(
        None, ge=0.0, le=1.0, description="Fan-out / (Fan-in + Fan-out)"
    )
    abstractness: float | None = Field(
        None, ge=0.0, le=1.0, description="Abstract classes / Total classes"
    )
    distance: float | None = Field(None, ge=0.0, le=1.0, description="Distance from main sequence")

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class DetectedPattern(BaseModel):
    """Architectural or design pattern detected in the codebase."""

    pattern_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    components: list[str]
    description: str

    # Legacy fields for compatibility
    name: str | None = None
    location: str | None = None

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


# ExternalPackage is imported from package_manager_models (line 29)
# Do not redefine it here - use the unified model


class Vulnerability(BaseModel):
    """Security vulnerability information."""

    id: str
    severity: str
    summary: str
    details: str
    affected_versions: list[str]
    fixed_versions: list[str]
    references: list[str]
    published_date: str
    modified_date: str

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class DependencyRiskSummary(BaseModel):
    """Summary statistics for dependency risk assessment."""

    total_packages: int
    vulnerable_packages: int
    total_vulnerabilities: int
    critical_vulnerabilities: int
    high_vulnerabilities: int
    moderate_vulnerabilities: int
    low_vulnerabilities: int

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class DependencyRiskReport(BaseModel):
    """Report of external dependencies and their risk assessment."""

    codebase_id: str
    generated_at: str  # ISO format datetime string
    packages: list[ExternalPackage]
    vulnerabilities: list[Vulnerability]
    risk_score: float
    summary: DependencyRiskSummary

    # Scan provenance, mirroring QualityReport's inconclusive/partial contract.
    # A vulnerability scan that failed and a vulnerability scan that found
    # nothing both produce an empty `vulnerabilities` list, and the difference
    # matters more here than anywhere else in the service: "0 vulnerabilities"
    # is the answer a consumer acts on. `inconclusive` (no package could be
    # checked) sets risk_score to -1.0 rather than the reassuring 0.0;
    # `partial` means the score is computed over `packages_scanned` of
    # `packages_attempted` and is a lower bound, never a clean bill of health.
    inconclusive: bool = False
    partial: bool = False
    packages_attempted: int = 0
    packages_scanned: int = 0
    scan_errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class RepoConfig(BaseModel):
    """Configuration parsed from .codegraph.yml file.

    Supports schema versioning for backward compatibility with older formats.
    Current schema version is "2.0".
    """

    # Versioning support
    schema_version: str = Field(
        default="2.0", description="Schema version for forward compatibility"
    )
    components: list[Component] | None = Field(default=None, description="Predefined components")
    layers: dict[str, list[str]] | None = Field(
        default=None,
        description="Architectural layers mapping to glob patterns",
    )
    layer_dependencies: list[dict[str, list[str]]] | None = Field(
        default=None, description="Allowed dependencies between layers"
    )
    custom_patterns: list[dict[str, Any]] | None = Field(
        default=None, description="Custom architectural pattern definitions"
    )
    extract_docstrings: bool = Field(
        default=True, description="Whether to extract docstrings for analysis"
    )
    min_pattern_confidence: float = Field(
        default=0.7,
        description="Minimum confidence threshold for pattern detection",
    )
    allowed_licenses: list[str] | None = Field(
        default=None, description="Whitelist of allowed licenses"
    )
    ml_confidence_settings: dict[str, Any] | None = Field(
        default=None, description="Settings for ML-based confidence scoring"
    )

    # Component detection configuration
    detect_by: list[str] = Field(
        default_factory=lambda: ["folder", "build_file", "annotation"],
        description="Detection strategies to use",
    )
    folder_roots: list[str] = Field(
        default_factory=list,
        description="Root folders for folder-based component detection",
    )
    build_files: list[str] = Field(
        default_factory=lambda: [
            "package.json",
            "pom.xml",
            "Cargo.toml",
            "go.mod",
            "pyproject.toml",
            "setup.py",
        ],
        description="Build file patterns for build-file-based detection",
    )
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ArchitectureJobRequest(BaseModel):
    """Request envelope for architecture analysis jobs."""

    codebase_id: str = Field(..., pattern=r"^[a-zA-Z0-9_-]+$")
    repo_path: Path
    # Owning tenant — threaded to the graph reads (load_graph) so the architecture
    # analysis only ever sees its own (or legacy-NULL) graph rows.
    tenant_id: str
    force_rebuild: bool = False

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ArchitectureAnalysisResult(BaseModel):
    """Complete result of architecture analysis."""

    codebase_id: str
    generated_at: str
    components: list[Component]
    layers: list[Layer]
    metrics: list[ArchitecturalMetric]
    patterns: list[DetectedPattern]
    summary: dict[str, int]
    execution_time: float
    config_used: RepoConfig

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ComponentExtractionStats(BaseModel):
    """Statistics from component extraction process."""

    components_found: int
    edges_scanned: int

    model_config = ConfigDict(frozen=True)


class LayerViolation(BaseModel):
    """Detected layer architecture violation."""

    component_id: str
    expected_layer: str | None
    actual_layer: str | None

    model_config = ConfigDict(frozen=True)


class Layer(BaseModel):
    """Architectural layer with components."""

    name: str
    components: list[str]
    component_count: int
    dependencies: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ArchitectureQualityIssue(BaseModel):
    """Code quality issue detected by analysis tools."""

    id: str
    severity: str
    message: str
    rule_id: str
    rule_name: str
    file_path: str
    line_number: int
    column_number: int | None = None
    tool: str
    category: str
    description: str | None = None
    resolve_suggestions: str | None = None
    external_url: str | None = None
    # Which repository of a multi-repo workspace ``file_path`` belongs to:
    # "primary" (the workspace root) or ".repos/<name>"
    # (``processing.workspace_repos``). ``None`` on issues from before it existed.
    repo: str | None = None

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class QualityReport(BaseModel):
    """Comprehensive code quality analysis report."""

    job_id: str
    codebase_id: str
    generated_at: str
    overall_score: float
    grade: str
    # The language the analyzer actually DETECTED for this codebase (from
    # ``detect_primary_language``), not the one a caller asked for. API
    # responses used to echo the request field back, so a plainly-Python repo
    # reported ``language: "unknown"`` in the same payload that listed six
    # Python tools under ``tools_attempted``.
    language: str = "unknown"
    inconclusive: bool = False
    # Per-tool provenance. `inconclusive` (no tool completed) and `partial`
    # (some completed) are the two ends of one spectrum, mutually exclusive by
    # construction. `partial` lets a consumer say "graded on N of M tools" and
    # name what failed, instead of trusting a grade computed from a subset.
    partial: bool = False
    tools_attempted: list[str] = Field(default_factory=list)
    tools_succeeded: list[str] = Field(default_factory=list)
    tools_failed: list[str] = Field(default_factory=list)
    # Scope paths that exist but hold no source file (``pyproject.toml``): not
    # given to any tool, so zero issues on them is NOT a clean result.
    scope_not_analysed: list[str] = Field(default_factory=list)
    issues: list[ArchitectureQualityIssue]
    summary: dict[str, int]
    recommendations: list[str]
    execution_time: float

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class HotspotPrediction(BaseModel):
    """Individual hotspot prediction for a code component."""

    component_id: str
    score: float = Field(ge=0.0, le=1.0, description="Hotspot risk score")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in prediction")
    predicted_issues: list[str] = Field(default_factory=list, description="Likely issue types")
    risk_factors: list[str] = Field(default_factory=list, description="Identified risk factors")
    recommended_actions: list[str] = Field(default_factory=list, description="Recommended actions")

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class HotspotAnalysisResult(BaseModel):
    """Complete hotspot analysis result for a codebase."""

    codebase_id: str
    generated_at: str
    hotspots: list[HotspotPrediction]
    analysis_ai_model_version: str = Field(..., description="Version of the analysis model used")
    overall_confidence: float

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ArchitectureRuntimeEdge(BaseModel):
    """Runtime execution data for code relationships."""

    source: str
    target: str
    call_count: int
    avg_duration_ms: float
    error_rate: float
    confidence: float

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class CodeGraphData(BaseModel):
    """Code graph data structure for analysis."""

    nodes: list[dict[str, str]]  # Streamlined node structure
    edges: list[dict[str, str]]  # Streamlined edge structure

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class GraphNode(BaseModel):
    """Node in the architecture visualization graph."""

    id: str
    name: str
    type: str  # 'component', 'service', 'module', 'package'
    layer: str | None = None
    metrics: dict[str, float] | None = None  # complexity, instability, abstractness, distance
    hotspotScore: float | None = None  # 0.0 to 1.0
    issues: int | None = None

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class RuntimeData(BaseModel):
    """Runtime execution data for edges."""

    call_count: int
    avg_duration_ms: float
    error_rate: float

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class GraphEdge(BaseModel):
    """Edge in the architecture visualization graph."""

    source: str  # node id
    target: str  # node id
    type: str  # 'dependency', 'import', 'call', 'runtime'
    weight: int | None = 1
    runtime_data: RuntimeData | None = None

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ArchitectureGraphMetrics(BaseModel):
    """Summary metrics for the architecture graph."""

    total_components: int
    total_dependencies: int
    avg_complexity: float
    hotspot_count: int

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ArchitectureGraphResponse(BaseModel):
    """Complete architecture graph visualization data."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    layers: list[str]
    metrics: ArchitectureGraphMetrics

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class QualityGateCheck(BaseModel):
    """Individual quality gate check result.

    Represents the evaluation of a single quality criterion against a threshold.

    Attributes:
        name: Unique identifier for the check (e.g., 'minimum_score', 'critical_issues')
        description: Human-readable description of what this check validates
        passed: Whether this check passed the threshold
        actual: Actual measured value
        threshold: Required threshold value
    """

    name: str = Field(..., description="Check identifier")
    description: str = Field(..., description="Human-readable check description")
    passed: bool = Field(..., description="Whether the check passed")
    actual: float = Field(..., description="Actual measured value")
    threshold: float = Field(..., description="Required threshold value")

    model_config = ConfigDict(extra="forbid")


class QualityGateResult(BaseModel):
    """Quality gate evaluation result.

    Evaluates code quality metrics against configurable thresholds to determine
    if the codebase meets minimum quality standards.

    Attributes:
        passed: Overall gate status - true if all checks pass
        overall_score: Aggregated quality score (0-100)
        grade: Letter grade (A, B, C, D, F)
        gate_checks: List of individual check results
        recommendations: Actionable suggestions for improvement
    """

    passed: bool = Field(..., description="Whether all quality gate checks passed")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall quality score")
    grade: str = Field(..., description="Quality grade (A, B, C, D, F, or N/A)")
    inconclusive: bool = Field(
        default=False,
        description="True when no static-analysis tool completed; score/grade are not meaningful and the gate cannot pass",
    )
    partial: bool = Field(
        default=False,
        description="True when some (but not all) static-analysis tools completed; the grade is computed from a subset. Mutually exclusive with inconclusive.",
    )
    tools_attempted: list[str] = Field(
        default_factory=list, description="Analysis tools that were run"
    )
    tools_succeeded: list[str] = Field(
        default_factory=list, description="Analysis tools that completed successfully"
    )
    tools_failed: list[str] = Field(
        default_factory=list, description="Analysis tools that errored or did not complete"
    )
    gate_checks: list[QualityGateCheck] = Field(
        default_factory=list, description="Individual check results"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Recommendations for improvement"
    )
    status: str | None = Field(
        default=None,
        description=(
            "Status of the stored run the gate was evaluated on: complete, partial, "
            "inconclusive, failed, or not_analyzed when nothing is stored"
        ),
    )
    result_id: str | None = Field(
        default=None, description="Id of the stored quality result the gate read"
    )
    commit_sha: str | None = Field(
        default=None, description="Commit the graded result was measured on, if known"
    )
    source: Literal["stored", "analyzed"] | None = Field(
        default=None,
        description=(
            "stored: a stored run for the container's current commit was graded; "
            "analyzed: no stored run matched the current commit, so the gate "
            "analyzed it (and persisted the run)"
        ),
    )

    model_config = ConfigDict(extra="forbid")


# T is used for the ModelMigrationTools generic type parameter
T = TypeVar("T", bound=BaseModel)


class ModelMigrationTools[T: BaseModel]:
    """Utilities for migrating between different versions of architecture models.

    This class provides methods to handle API evolution and schema versioning,
    ensuring backwards compatibility with older data formats and clients.
    """

    def __init__(self, model_cls: type[T]):
        """Initialize migration tools for a specific model class.

        Args:
            model_cls: The Pydantic model class to create migration tools for
        """
        self.model_cls = model_cls
        self._version_migrations: dict[tuple[str, str], Callable[..., Any]] = {}
        self._creation_time = datetime.now(UTC)

    def register_version_migration(
        self, from_version: str, to_version: str, migration_func: Callable[..., Any]
    ) -> None:
        """Register a migration function between schema versions.

        Args:
            from_version: Source schema version (e.g., "1.0")
            to_version: Target schema version (e.g., "2.0")
            migration_func: Function that transforms data from source to target schema
        """
        key = (from_version, to_version)
        self._version_migrations[key] = migration_func

    def migrate(self, data: dict[str, Any], from_version: str, to_version: str) -> dict[str, Any]:
        """Migrate data from one schema version to another.

        Args:
            data: Source data dictionary
            from_version: Source schema version
            to_version: Target schema version

        Returns:
            Migrated data dictionary compatible with target schema version

        Raises:
            ValueError: If no migration path exists between versions
        """
        if from_version == to_version:
            return data

        key = (from_version, to_version)
        if key in self._version_migrations:
            return dict(self._version_migrations[key](data))

        # Try to find a path through multiple migrations
        for src, mid in self._version_migrations.keys():
            if src == from_version:
                for mid2, tgt in self._version_migrations.keys():
                    if mid == mid2 and tgt == to_version:
                        # Found a two-step migration path
                        intermediate = self._version_migrations[(src, mid)](data)
                        return dict(self._version_migrations[(mid, tgt)](intermediate))

        raise ValueError(f"No migration path exists from version {from_version} to {to_version}")

    def to_latest(self, data: dict[str, Any], from_version: str) -> T:
        """Migrate data to the latest schema version and return a model instance.

        Args:
            data: Source data dictionary
            from_version: Source schema version

        Returns:
            Model instance with the migrated data
        """
        # Get the latest version from the model's metadata
        latest_version = getattr(self.model_cls, "__version__", "latest")

        # Migrate the data to the latest version
        if from_version != latest_version:
            data = self.migrate(data, from_version, latest_version)

        # Create and return a model instance with the migrated data
        return self.model_cls(**data)

    def get_creation_timestamp(self) -> str:
        """Return a formatted timestamp when this migration tool was created.

        Uses the datetime import to avoid linter warnings about unused imports.

        Returns:
            ISO formatted timestamp string
        """
        return self._creation_time.isoformat()
