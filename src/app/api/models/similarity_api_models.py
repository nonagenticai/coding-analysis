"""
Similarity Analysis API Response Models

This module provides response models for similarity detection endpoints,
including clone detection and cross-language pattern analysis.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .common_api_models import BaseResponse, ResponseStatus


class CloneDetectionResponse(BaseResponse):
    """Response from clone detection operation.

    Attributes:
        status: Response status (success/error)
        total_clones: Total number of code clones detected
        files_analyzed: Number of files analyzed for clones
        clone_threshold: Similarity threshold used for detection
        clones: List of detected clone pairs with details
        summary: Statistical summary of clone detection
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    total_clones: int = Field(..., description="Total number of clones detected")
    files_analyzed: int = Field(..., description="Number of files analyzed")
    clone_threshold: float = Field(..., description="Similarity threshold used")
    clones: list[dict[str, Any]] = Field(
        default_factory=list, description="List of detected clone pairs"
    )
    summary: dict[str, Any] = Field(
        default_factory=dict, description="Statistical summary of clones"
    )


class CrossLanguagePatternResponse(BaseResponse):
    """Response from cross-language pattern detection operation.

    Attributes:
        status: Response status (success/error)
        total_patterns: Total number of cross-language patterns found
        files_analyzed: Number of files analyzed
        pattern_threshold: Similarity threshold used for pattern detection
        patterns: List of detected cross-language patterns
        language_pairs: Count of patterns by language pair
        summary: Statistical summary of pattern detection
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    total_patterns: int = Field(..., description="Total patterns detected")
    files_analyzed: int = Field(..., description="Number of files analyzed")
    pattern_threshold: float = Field(..., description="Pattern threshold used")
    patterns: list[dict[str, Any]] = Field(
        default_factory=list, description="List of detected cross-language patterns"
    )
    language_pairs: dict[str, int] = Field(
        default_factory=dict, description="Pattern count by language pair"
    )
    summary: dict[str, Any] = Field(
        default_factory=dict, description="Statistical summary of patterns"
    )


class SimilarityCapabilitiesResponse(BaseResponse):
    """Response containing similarity detection system capabilities.

    Attributes:
        status: Response status (success/error)
        similarity_types: Available types of similarity detection
        analysis_scopes: Supported analysis granularity levels
        algorithms: Similarity calculation methods available
        supported_languages: Programming languages with similarity support
        thresholds: Default and recommended similarity thresholds
        performance_limits: System limitations and constraints
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    similarity_types: list[str] = Field(
        default_factory=list, description="Available similarity detection types"
    )
    analysis_scopes: list[str] = Field(
        default_factory=list, description="Supported analysis granularity levels"
    )
    algorithms: list[str] = Field(
        default_factory=list, description="Similarity calculation methods"
    )
    supported_languages: list[str] = Field(
        default_factory=list, description="Languages with similarity support"
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict, description="Default and recommended thresholds"
    )
    performance_limits: dict[str, Any] = Field(
        default_factory=dict, description="System limitations and constraints"
    )
