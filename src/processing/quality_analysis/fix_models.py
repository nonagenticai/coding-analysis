"""
Models for code fix suggestions and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class FixStrategy(StrEnum):
    """Types of resolve strategies available."""

    AUTOMATIC = "automatic"  # Can be safely applied automatically
    SUGGESTED = "suggested"  # Suggested but requires review
    MANUAL = "manual"  # Requires manual intervention
    EXPERIMENTAL = "experimental"  # Experimental resolve, use with caution


class FixCategory(StrEnum):
    """Categories of resolves."""

    FORMATTING = "formatting"
    IMPORT_ORGANIZATION = "import_organization"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TYPE_ANNOTATION = "type_annotation"
    CODE_STYLE = "code_style"
    BUG_FIX = "bug_fix"
    REFACTORING = "refactoring"


class QualityAnalysisResolveSuggestions(BaseModel):
    """A suggested resolve for a quality issue."""

    # Resolve identification
    fix_id: str = Field(description="Unique identifier for this resolve")
    issue_id: str | None = Field(None, description="Related issue ID")

    # Resolve metadata
    strategy: FixStrategy = Field(description="Resolve application strategy")
    category: FixCategory = Field(description="Category of resolve")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in resolve (0-1)")
    priority: int = Field(ge=1, le=10, description="Resolve priority (1=highest)")

    # Issue context
    tool: str = Field(description="Tool that identified the issue")
    rule_id: str | None = Field(None, description="Rule ID that triggered")
    severity: str = Field(description="Issue severity")

    # Resolve details
    title: str = Field(description="Short description of resolve")
    description: str = Field(description="Detailed resolve description")
    file_path: str = Field(description="File path where resolve applies")
    line_number: int | None = Field(None, description="Line number for resolve")

    # Code changes
    original_code: str | None = Field(None, description="Original code snippet")
    fixed_code: str | None = Field(None, description="Resolved code snippet")
    diff: str | None = Field(None, description="Unified diff of changes")

    # Metadata
    requires_review: bool = Field(default=True, description="Whether resolve needs review")
    breaking_change: bool = Field(
        default=False, description="Whether resolve might break functionality"
    )
    side_effects: list[str] = Field(default=[], description="Potential side effects")
    dependencies: list[str] = Field(default=[], description="Required dependencies or tools")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:  # noqa: N805
        """Ensure confidence is between 0 and 1."""
        return max(0.0, min(1.0, v))


class BugFixValidationResult(BaseModel):
    """Result of resolve validation."""

    validation_id: str = Field(description="Unique validation ID")
    fix_id: str = Field(description="ID of validated resolve")

    # Validation results
    is_valid: bool = Field(description="Whether resolve is valid")
    is_safe: bool = Field(description="Whether resolve is safe to apply")
    syntax_valid: bool = Field(description="Whether resulting code has valid syntax")

    # Analysis results
    issues_fixed: int = Field(ge=0, description="Number of issues resolved")
    new_issues: int = Field(ge=0, description="Number of new issues introduced")
    quality_score_change: float = Field(description="Change in quality score")

    # Validation details
    validation_errors: list[str] = Field(default=[], description="Validation errors found")
    warnings: list[str] = Field(default=[], description="Validation warnings")
    execution_time: float = Field(ge=0.0, description="Validation execution time")

    # Test results
    tests_passed: bool = Field(default=True, description="Whether tests still pass")
    test_output: str | None = Field(None, description="Test execution output")


class FixApplicationResult(BaseModel):
    """Result of applying resolves."""

    application_id: str = Field(description="Unique application ID")
    fixes_applied: list[str] = Field(description="List of resolve IDs that were applied")
    fixes_failed: list[str] = Field(description="List of resolve IDs that failed")

    # Results summary
    total_fixes_attempted: int = Field(ge=0, description="Total resolves attempted")
    successful_applications: int = Field(ge=0, description="Number of successful applications")
    failed_applications: int = Field(ge=0, description="Number of failed applications")

    # Quality impact
    initial_quality_score: float = Field(description="Quality score before resolves")
    final_quality_score: float = Field(description="Quality score after resolves")
    quality_improvement: float = Field(description="Quality score update")

    # File changes
    files_modified: list[str] = Field(default=[], description="List of modified files")
    backup_created: bool = Field(default=False, description="Whether backup was created")

    # Metadata
    execution_time: float = Field(ge=0.0, description="Total execution time")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When resolves were applied",
    )
    errors: list[str] = Field(default=[], description="Application errors")


@dataclass
class FixRule:
    """Rule for generating resolves for specific issue patterns."""

    rule_id: str
    tool: str
    rule_pattern: str  # Regex pattern to match rule IDs
    message_pattern: str | None = None  # Pattern to match messages
    category: FixCategory = FixCategory.CODE_STYLE
    strategy: FixStrategy = FixStrategy.SUGGESTED
    confidence: float = 0.7
    fix_template: str | None = None
    requires_review: bool = True
