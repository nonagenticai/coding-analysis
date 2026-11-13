"""Static analysis models for comprehensive code analysis and validation.

This module provides sophisticated Pydantic models for static code analysis operations
including call graph analysis, function signature analysis, and advanced validation
techniques. The models support AI-driven code analysis workflows with comprehensive
validation capabilities across multiple programming languages.

The static analysis system enables:
- Call graph generation with visual representations and metrics
- Function signature analysis with type inference and documentation extraction
- Advanced validation using symbolic execution, taint analysis, and formal verification
- Documentation generation with architectural diagrams and API references
- Multi-language support for Python, JavaScript, and TypeScript

Key Model Categories:
    Request Models: CallGraphRequest, SignatureAnalysisRequest, AdvancedValidationRequest
    Response Models: CallGraphResponse, SignatureAnalysisResponse, AdvancedValidationResponse
    Data Models: CallGraphNode, FunctionSignature, SecurityIssue for analysis results
    Validation Models: SymbolicExecutionResult, TaintAnalysisResult, FormalVerificationResult
    Documentation Models: DocumentationRequest, DocumentationResponse for doc generation

Analysis Types:
    - Call Graph: Function call relationships and execution paths
    - Signature Analysis: Function signatures with type inference
    - Symbolic Execution: Edge case discovery and assertion validation
    - Taint Analysis: Security vulnerability detection through data flow
    - Formal Verification: Mathematical proof of code properties
    - Documentation: Automated generation of technical documentation

Security Features:
    - Multi-level security validation (LOW, MEDIUM, HIGH, CRITICAL)
    - Taint analysis for vulnerability detection
    - Security issue reporting with severity levels
    - Formal verification for critical code paths

Example:
    ```python
    from static_analysis_models import (
        CallGraphRequest, SignatureAnalysisRequest,
        OutputFormat, StaticAnalysisSecurityLevel, Language
    )

    # Call graph analysis request
    call_graph_request = CallGraphRequest(
        container_name="my-analysis-container",
        analysis_scope=["src/", "lib/"],
        output_format=OutputFormat.SVG,
        include_external=True,
        max_depth=15
    )

    # Function signature analysis request
    signature_request = SignatureAnalysisRequest(
        container_name="my-analysis-container",
        language=Language.PYTHON,
        include_type_inference=True,
        extract_docstrings=True,
        analyze_usage_patterns=True
    )

    # Advanced validation request
    validation_request = AdvancedValidationRequest(
        code_changes=[
            CodeChange(
                file_path="src/main.py",
                original_code="def process(data): return data",
                modified_code="def process(data: List[Dict]) -> Dict: return transform(data)",
                change_type=ChangeType.MODIFICATION
            )
        ],
        validation_types=[ValidationType.SYMBOLIC_EXECUTION, ValidationType.TAINT_ANALYSIS],
        security_level=StaticAnalysisSecurityLevel.HIGH
    )
    ```
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from processing.workspace_repos import RepoSelector


class OutputFormat(StrEnum):
    """Output format options for call graph analysis.

    Defines the available output formats for call graph visualization and data export.
    Each format serves different purposes for analysis and presentation.

    Attributes:
        DOT: Graphviz DOT format for programmatic graph manipulation
        SVG: Scalable Vector Graphics format for web display and documentation
        JSON: JavaScript Object Notation format for data exchange and API responses
    """

    DOT = "dot"
    SVG = "svg"
    JSON = "json"


class StaticAnalysisSecurityLevel(StrEnum):
    """Security level options for advanced validation.

    Defines the security validation levels that determine the strictness and depth
    of security analysis performed during code validation.

    Attributes:
        LOW: Basic security checks with minimal performance impact
        MEDIUM: Standard security analysis with balanced performance
        HIGH: Comprehensive security validation with thorough analysis
        CRITICAL: Maximum security scrutiny for security-critical code paths
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DocumentationType(StrEnum):
    """Documentation type options for automated documentation generation.

    Defines the types of documentation that can be automatically generated
    from code analysis results.

    Attributes:
        ARCHITECTURE: High-level architectural documentation with component diagrams
        API_REFERENCE: Detailed API documentation with function signatures and usage
        CHANGE_SUMMARY: Summary of code changes with impact analysis
    """

    ARCHITECTURE = "architecture"
    API_REFERENCE = "api_reference"
    CHANGE_SUMMARY = "change_summary"
    USER_GUIDE = "user_guide"


class DocumentationFormat(StrEnum):
    """Documentation format options for output generation.

    Defines the available output formats for generated documentation,
    each optimized for different use cases and presentation requirements.

    Attributes:
        MARKDOWN: Markdown format for README files and GitHub documentation
        HTML: HTML format for web-based documentation and portals
        SPHINX: Sphinx format for Python documentation with advanced features
    """

    MARKDOWN = "markdown"
    HTML = "html"
    SPHINX = "sphinx"


class Language(StrEnum):
    """Supported programming languages for static analysis.

    Defines the programming languages supported by the static analysis system,
    each with specialized analysis capabilities and language-specific features.

    Attributes:
        PYTHON: Python language support with AST analysis and type inference
        JAVASCRIPT: JavaScript language support with ES6+ features
        TYPESCRIPT: TypeScript language support with type annotations
    """

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"


class StaticAnalysisChangeType(StrEnum):
    """Type of code changes for validation analysis.

    Defines the types of code changes that can be analyzed during validation,
    each requiring different validation strategies and risk assessments.

    Attributes:
        ADDED: New code that has been added to the codebase
        DELETED: Code that has been removed from the codebase
        MODIFIED: Existing code that has been modified or updated
        RENAMED: Code that has been moved or renamed
    """

    ADDED = "ADDED"
    DELETED = "DELETED"
    MODIFIED = "MODIFIED"
    RENAMED = "RENAMED"


# Alias for backward compatibility
ChangeType = StaticAnalysisChangeType


class ValidationType(StrEnum):
    """Types of advanced validation techniques available.

    Defines the advanced validation techniques that can be applied to code
    for comprehensive analysis and security assessment.

    Attributes:
        SYMBOLIC_EXECUTION: Symbolic execution for edge case discovery and path analysis
        TAINT_ANALYSIS: Data flow analysis for security vulnerability detection
        FORMAL_VERIFICATION: Mathematical verification of code properties and contracts
    """

    SYMBOLIC_EXECUTION = "symbolic_execution"
    TAINT_ANALYSIS = "taint_analysis"
    FORMAL_VERIFICATION = "formal_verification"


# Call Graph Analysis Models


class CallGraphRequest(RepoSelector):
    """Request model for call graph analysis."""

    container_name: str = Field(..., description="Name of provisioned container to analyze")
    analysis_scope: list[str] = Field(
        default=["."],
        description="Directories to include in analysis (default: the whole repository)",
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.JSON, description="Output format for call graph"
    )
    include_external: bool = Field(default=False, description="Include external library calls")
    max_depth: int = Field(default=10, description="Maximum call depth to analyze")

    @field_validator("container_name")
    @classmethod
    def validate_container_name(cls, v):
        """Validate that the container name is not empty."""
        if not v or not v.strip():
            raise ValueError("container_name cannot be empty")
        return v.strip()


class StaticAnalysisCallGraphNode(BaseModel):
    """Individual node in call graph."""

    function: str = Field(..., description="Function name")
    module: str = Field(..., description="Module path")
    complexity: int = Field(..., description="Function complexity score")
    line_number: int | None = Field(None, description="Line number in source")


class StaticAnalysisCallGraphEdge(BaseModel):
    """Edge representing function call relationship."""

    from_function: str = Field(..., alias="from", description="Calling function")
    to_function: str = Field(..., alias="to", description="Called function")
    call_type: str = Field(..., description="Type of call (direct, indirect, etc.)")


class CallGraphMetrics(BaseModel):
    """Metrics about the call graph."""

    total_functions: int = Field(..., description="Total number of functions")
    max_call_depth: int = Field(..., description="Maximum call depth found")
    cyclic_dependencies: list[str] = Field(
        default=[], description="List of cyclic dependencies found"
    )
    total_edges: int = Field(default=0, description="Resolved caller->callee edges")
    unresolved_calls: int = Field(
        default=0,
        description="Call sites whose callee could not be resolved to a repo function "
        "(builtins excluded); never turned into edges",
    )
    parse_errors: int = Field(
        default=0,
        description="Python files the AST backend could not parse (excluded from the graph)",
    )


class CallGraphResponse(BaseModel):
    """Response model for call graph analysis."""

    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: Literal["computed", "not_computed"] = Field(
        default="computed",
        description="not_computed when no backend could analyse the repository; "
        "an empty graph is then NOT evidence of an empty call graph",
    )
    not_computed_reason: str | None = Field(None, description="Why status is not_computed")
    tools_used: list[str] = Field(default=[], description="Backends that actually ran")
    tools_unavailable: list[str] = Field(
        default=[], description="External tools that are not installed"
    )
    call_graph: dict[str, Any] = Field(..., description="Call graph data structure")
    visual_output: str | None = Field(None, description="Base64 encoded visual representation")
    critical_paths: list[str] = Field(default=[], description="Critical execution paths")
    orphaned_functions: list[str] = Field(default=[], description="Functions with no callers")
    metrics: CallGraphMetrics = Field(..., description="Call graph metrics")


# Signature Analysis Models


# The /signatures dispatch table. Python only: FunctionExtractor parses Python.
# JS/TS used to be listed, so every .js/.ts file failed inside the extractor, was
# dropped, and the answer was "0 functions" -- a vacuous result. Any other
# language is an honest 400.
SIGNATURE_PATTERNS: dict[str, tuple[str, ...]] = {
    "python": ("*.py",),
    # Rust delegates to the structure api-surface extractor (processing.structure.rust_facts).
    "rust": ("*.rs", "Cargo.toml"),
}


class SignatureAnalysisRequest(RepoSelector):
    """Request model for function signature analysis."""

    container_name: str = Field(..., description="Name of provisioned container to analyze")
    language: Language = Field(
        ...,
        description=(
            "Programming language. Signature extraction runs ONLY for: "
            + ", ".join(sorted(SIGNATURE_PATTERNS))
            + "; any other value is a 400."
        ),
    )
    include_type_inference: bool = Field(default=True, description="Run type inference analysis")
    extract_docstrings: bool = Field(default=True, description="Extract function docstrings")
    analyze_usage_patterns: bool = Field(
        default=True, description="Analyze function usage patterns"
    )

    @field_validator("container_name")
    @classmethod
    def validate_container_name(cls, v):
        """Validate that the container name is not empty."""
        if not v or not v.strip():
            raise ValueError("container_name cannot be empty")
        return v.strip()


class FunctionParameter(BaseModel):
    """Function parameter details."""

    name: str = Field(..., description="Parameter name")
    type: str | None = Field(None, description="Parameter type annotation")
    required: bool = Field(..., description="Whether parameter is required")
    default: str | None = Field(None, description="Default value if any")
    description: str | None = Field(None, description="Parameter description")
    kind: str | None = Field(
        None,
        description=(
            "Python: positional_only, positional_or_keyword, var_positional, keyword_only or "
            "var_keyword. Rust: self (the receiver) or positional"
        ),
    )


class FunctionComplexityMetrics(BaseModel):
    """Function complexity metrics."""

    cyclomatic: int = Field(..., description="Cyclomatic complexity")
    cognitive: int = Field(..., description="Cognitive complexity")


class StaticAnalysisFunctionSignature(BaseModel):
    """Comprehensive function signature information."""

    name: str = Field(..., description="Function name")
    module: str = Field(..., description="Module path")
    signature: str = Field(..., description="Complete function signature")
    parameters: list[FunctionParameter] = Field(default=[], description="Function parameters")
    return_type: str | None = Field(None, description="Return type annotation")
    docstring: str | None = Field(None, description="Function docstring")
    usage_examples: list[str] = Field(
        default=[], description="Real call sites or docstring examples (never synthesised)"
    )
    called_by: list[str] = Field(
        default=[],
        description=(
            "Candidate callers: functions in the analysed files that call a function with "
            "this NAME (matched by name, not resolved, so may include callers of a "
            "same-named function)"
        ),
    )
    calls: list[str] = Field(default=[], description="Functions called by this function")
    line_number: int | None = Field(None, description="Line of the def")
    is_async: bool = Field(False, description="async def")
    is_method: bool = Field(False, description="Defined directly in a class body")
    complexity_metrics: FunctionComplexityMetrics | None = Field(
        ..., description="Complexity metrics (null when not computed: language rust)"
    )


class SignatureAnalysisResponse(BaseModel):
    """Response model for signature analysis."""

    analysis_id: str = Field(..., description="Unique analysis identifier")
    functions: list[StaticAnalysisFunctionSignature] = Field(
        default=[], description="List of analyzed functions"
    )
    type_coverage: float = Field(..., description="Type annotation coverage (0-1)")
    external_dependencies: list[str] = Field(
        default=[], description="External dependencies referenced"
    )
    total_functions: int = Field(..., description="Total number of functions analyzed")
    typed_functions: int = Field(..., description="Number of functions with type annotations")
    files_not_computed: list[dict[str, str]] = Field(
        default=[], description="Files that could not be analysed ({file, reason})"
    )
    call_arity_mismatches: list[dict[str, Any]] = Field(
        default=[],
        description=(
            "Plain-name calls whose arguments cannot bind to the one function they name "
            "(computed only when that binding is certain)"
        ),
    )
    call_arity_status: str = Field(
        default="computed",
        description=(
            "'computed', or 'not_computed' when arity judging was disabled for the whole "
            "set (call_arity_mismatches is then empty, not clean)"
        ),
    )
    call_arity_reason: str | None = Field(
        default=None, description="Why call arity was not computed, when it was not"
    )
    unresolved: list[dict[str, Any]] = Field(
        default=[],
        description=(
            "rust: macro-generated items (macro_rules!, derive, attribute macros) and modules "
            "that could not be resolved; never silently missing"
        ),
    )


# Advanced Validation Models


class ValidationVerdict(StrEnum):
    """Aggregate outcome of an advanced-validation run.

    This is the field a caller branches on. It is deliberately three-valued:
    "this change is safe", "this change is not safe" and "nobody checked" are
    three different answers, and collapsing the third into the first is what
    turns a verifier into a rubber stamp.

    Attributes:
        PASS: At least one analyzer ran and none of them produced a blocking issue.
        FAIL: At least one analyzer produced a blocking issue.
        ABSTAIN: No analyzer ran. NOT a pass — nothing was verified.
    """

    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"


#: Status reported by an analyzer that could not run at all (tool missing,
#: unsupported language, nothing requested). Distinct from "failed", which means
#: the analyzer started and errored. Neither counts towards a pass.
ANALYZER_NOT_COMPUTED = "not_computed"

#: Status reported by an analyzer that started and then errored.
ANALYZER_FAILED = "failed"

#: Statuses that mean the analyzer produced a real, trustworthy result.
ANALYZER_RAN_STATUSES: frozenset[str] = frozenset(
    {"completed", "passed", "violations_found", "warning"}
)


class AnalyzerAbstention(BaseModel):
    """One analyzer that did not produce a result, and why."""

    analyzer: str = Field(..., description="Analyzer name, e.g. 'taint_analysis'")
    status: str = Field(
        default=ANALYZER_NOT_COMPUTED,
        description="'not_computed' (never ran) or 'failed' (ran and errored)",
    )
    reason: str = Field(..., description="Why this analyzer produced no result")


class CodeChange(BaseModel):
    """Individual code change for validation.

    Accepts two request shapes so the endpoint can verify the producer that
    actually calls it:

    * the canonical shape — ``modified_code`` plus an uppercase ``change_type``;
    * the code-change spec emitted by the conveyor's ``applyCodeChanges``
      producer — ``{file_path, content, change_type: "deleted"}``, lowercase and
      with the body under ``content``.

    A DELETED entry legitimately carries no body, so ``modified_code`` defaults
    to the empty string for that change type rather than rejecting the request.
    """

    file_path: str = Field(..., description="Path to the file")
    original_code: str | None = Field(None, description="Original code")
    modified_code: str = Field(
        default="",
        description="Modified code (also accepted as 'content'; empty for a DELETED file)",
    )
    change_type: StaticAnalysisChangeType = Field(..., description="Type of change")

    @model_validator(mode="before")
    @classmethod
    def _accept_producer_shape(cls, data: Any) -> Any:
        """Normalise the conveyor's code-change shape onto the canonical one."""
        if not isinstance(data, dict):
            return data
        normalised = dict(data)

        # `content` is the producer's name for the file body.
        if normalised.get("modified_code") is None and normalised.get("content") is not None:
            normalised["modified_code"] = normalised["content"]
        normalised.pop("content", None)

        change_type = normalised.get("change_type")
        if isinstance(change_type, str):
            normalised["change_type"] = change_type.upper()

        # A deleted file has no body; anything else must still supply one.
        if normalised.get("modified_code") is None:
            if normalised.get("change_type") == StaticAnalysisChangeType.DELETED.value:
                normalised["modified_code"] = ""
            else:
                normalised.pop("modified_code", None)

        return normalised

    @model_validator(mode="after")
    def _require_body_unless_deleted(self) -> CodeChange:
        """A non-DELETED change with no body cannot be validated."""
        if self.change_type is not StaticAnalysisChangeType.DELETED and not self.modified_code:
            raise ValueError(
                f"modified_code (or 'content') is required for a "
                f"{self.change_type} change to {self.file_path}"
            )
        return self


class AdvancedValidationRequest(BaseModel):
    """Request model for advanced code validation."""

    container_name: str = Field(..., description="Container name to scope file paths")
    code_changes: list[CodeChange] = Field(..., description="List of code changes to validate")
    validation_types: list[ValidationType] = Field(
        ..., description="Types of validation to perform"
    )
    security_level: StaticAnalysisSecurityLevel = Field(
        default=StaticAnalysisSecurityLevel.MEDIUM,
        description="Security validation level",
    )


class SymbolicExecutionResult(BaseModel):
    """Results from symbolic execution."""

    status: str = Field(
        ...,
        description="'completed', 'failed', or 'not_computed' when the tool is unavailable",
    )
    edge_cases_found: list[str] = Field(default=[], description="Edge cases discovered")
    assertion_violations: list[str] = Field(default=[], description="Assertion violations found")
    reason: str | None = Field(
        None, description="Why no result was produced (set when status is not 'completed')"
    )


class SecurityIssue(BaseModel):
    """Security issue found during taint analysis."""

    severity: str = Field(..., description="Issue severity")
    type: str = Field(..., description="Type of security issue")
    location: str = Field(..., description="Location in code, as '<file>:<line>'")
    flow_path: list[str] = Field(default=[], description="Data flow path")
    recommendation: str = Field(..., description="Resolve recommendation")
    line: int | None = Field(None, description="Line number the issue was found on")
    message: str | None = Field(None, description="Human-readable description of the issue")
    detector: str | None = Field(None, description="Which rule set reported this issue")
    code_snippet: str | None = Field(None, description="The offending source line")


class TaintAnalysisResult(BaseModel):
    """Results from taint analysis."""

    status: str = Field(
        ...,
        description="'completed', 'failed', or 'not_computed' when the tool is unavailable",
    )
    security_issues: list[SecurityIssue] = Field(default=[], description="Security issues found")
    reason: str | None = Field(
        None, description="Why no result was produced (set when status is not 'completed')"
    )


class FormalVerificationViolation(BaseModel):
    """A single property violation found by formal verification."""

    type: str = Field(..., description="Violated property")
    message: str = Field(..., description="Description of the violation")
    severity: str = Field(..., description="Violation severity")
    line: int | None = Field(None, description="Line number of the violation")


class FormalVerificationResult(BaseModel):
    """Results from formal verification."""

    status: str = Field(
        ...,
        description=(
            "'passed', 'violations_found', 'failed', or 'not_computed'. "
            "It is never defaulted to 'passed' — that must be earned by a run."
        ),
    )
    properties_verified: list[str] = Field(default=[], description="Properties that were verified")
    coverage: float | None = Field(
        None, description="Verification coverage; None when nothing was verified"
    )
    violations: list[FormalVerificationViolation] = Field(
        default=[], description="Property violations found"
    )
    reason: str | None = Field(
        None, description="Why no result was produced (set when status is not a real run)"
    )


class StaticAnalysisValidationResult(BaseModel):
    """Validation results for a single file."""

    file_path: str = Field(..., description="Path to validated file")
    symbolic_execution: SymbolicExecutionResult | None = Field(
        None, description="Symbolic execution results"
    )
    taint_analysis: TaintAnalysisResult | None = Field(None, description="Taint analysis results")
    formal_verification: FormalVerificationResult | None = Field(
        None, description="Formal verification results"
    )
    safety_score: float | None = Field(
        None,
        description="Safety score for this file (0-1); None when no analyzer produced a result",
    )
    analyzers_ran: list[str] = Field(
        default=[], description="Analyzers that produced a real result for this file"
    )
    analyzers_not_computed: list[AnalyzerAbstention] = Field(
        default=[], description="Analyzers that produced no result for this file, and why"
    )


class AdvancedValidationResponse(BaseModel):
    """Response model for advanced validation.

    ``verdict`` is the field to branch on. ``overall_safety_score`` is ``None``
    whenever ``verdict`` is ``abstain``: no analyzer ran, so there is no score to
    report and a caller must not read the absence of blocking issues as a pass.
    """

    analysis_id: str = Field(..., description="Unique analysis identifier")
    validation_results: list[StaticAnalysisValidationResult] = Field(
        ..., description="Validation results per file"
    )
    overall_safety_score: float | None = Field(
        ..., description="Overall safety score (0-1); None when no analyzer ran"
    )
    blocking_issues: list[str] = Field(default=[], description="Issues that block deployment")
    recommendations: list[str] = Field(default=[], description="General recommendations")
    verdict: ValidationVerdict = Field(
        default=ValidationVerdict.ABSTAIN,
        description="'pass', 'fail', or 'abstain' when no analyzer ran",
    )
    analyzers_ran: list[str] = Field(
        default=[], description="Distinct analyzers that produced a real result"
    )
    analyzers_not_computed: list[AnalyzerAbstention] = Field(
        default=[], description="Analyzers that produced no result, and why"
    )


# Documentation Generation Models


class DocumentationRequest(RepoSelector):
    """Request model for documentation generation."""

    container_name: str = Field(..., description="Name of provisioned container to analyze")
    doc_types: list[DocumentationType] = Field(
        ..., description="Types of documentation to generate"
    )
    include_diagrams: bool = Field(default=True, description="Include architectural diagrams")
    format: DocumentationFormat = Field(
        default=DocumentationFormat.MARKDOWN, description="Output format"
    )

    @field_validator("container_name")
    @classmethod
    def validate_container_name(cls, v):
        """Validate that the container name is not empty."""
        if not v or not v.strip():
            raise ValueError("container_name cannot be empty")
        return v.strip()


class DocumentationSection(BaseModel):
    """Individual documentation section."""

    content: str = Field(..., description="Documentation content")
    diagrams: list[str] = Field(default=[], description="Associated diagram files")
    coverage: float | None = Field(None, description="Documentation coverage")


class DocumentationResponse(BaseModel):
    """Response model for documentation generation."""

    analysis_id: str = Field(..., description="Unique analysis identifier")
    documentation: dict[str, DocumentationSection] = Field(
        ..., description="Generated documentation sections"
    )
    generated_files: list[str] = Field(default=[], description="List of generated files")
    quality_score: float = Field(..., description="Documentation quality score (0-1)")


# Database Models for Persistence


class CallGraphAnalysis(BaseModel):
    """Database model for call graph analysis results."""

    id: str = Field(..., description="Analysis ID")
    repository_url: str = Field(..., description="Repository URL")
    analysis_config: dict[str, Any] = Field(..., description="Analysis configuration")
    call_graph_data: dict[str, Any] = Field(..., description="Call graph data")
    visual_outputs: dict[str, Any] = Field(..., description="Visual outputs")
    created_at: str = Field(..., description="Creation timestamp")


class SignatureAnalysis(BaseModel):
    """Database model for signature analysis results."""

    id: str = Field(..., description="Analysis ID")
    repository_url: str = Field(..., description="Repository URL")
    language: str = Field(..., description="Programming language")
    functions_data: dict[str, Any] = Field(..., description="Functions data")
    type_coverage: float = Field(..., description="Type coverage")
    created_at: str = Field(..., description="Creation timestamp")


class ValidationAnalysis(BaseModel):
    """Database model for validation analysis results."""

    id: str = Field(..., description="Analysis ID")
    code_changes: dict[str, Any] = Field(..., description="Code changes")
    validation_results: dict[str, Any] = Field(..., description="Validation results")
    safety_score: float = Field(..., description="Safety score")
    created_at: str = Field(..., description="Creation timestamp")
