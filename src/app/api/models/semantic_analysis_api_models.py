"""
Semantic Analysis API Response Models

This module provides response models for semantic analysis endpoints,
including capabilities and pattern information.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from .common_api_models import BaseResponse, ResponseStatus


class SemanticCapabilitiesResponse(BaseResponse):
    """Response containing semantic analysis capabilities.

    Attributes:
        status: Response status (success/error)
        semantic_relationship_types: Available relationship types
        cross_language_patterns: Supported cross-language patterns
        analysis_features: Analysis capabilities and features
        supported_languages: Languages with semantic support
        performance_characteristics: Performance metrics and limits
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    semantic_relationship_types: list[dict[str, str]] = Field(
        default_factory=list,
        description="Available relationship types with descriptions",
    )
    cross_language_patterns: list[str] = Field(
        default_factory=list, description="Supported cross-language patterns"
    )
    analysis_features: dict[str, bool] = Field(
        default_factory=dict, description="Analysis capabilities"
    )
    supported_languages: list[str] = Field(
        default_factory=list, description="Languages with semantic support"
    )
    performance_characteristics: dict[str, str] = Field(
        default_factory=dict, description="Performance metrics and limits"
    )


class SemanticPatternsResponse(BaseResponse):
    """Response containing semantic analysis patterns information.

    Attributes:
        status: Response status (success/error)
        cross_language_patterns: Cross-language detection patterns
        architectural_patterns: Architectural pattern detection info
        type_system_patterns: Type system analysis patterns
    """

    status: ResponseStatus = ResponseStatus.SUCCESS
    cross_language_patterns: list[dict[str, Any]] = Field(
        default_factory=list, description="Cross-language detection patterns"
    )
    architectural_patterns: list[dict[str, Any]] = Field(
        default_factory=list, description="Architectural pattern detection"
    )
    type_system_patterns: list[dict[str, Any]] = Field(
        default_factory=list, description="Type system analysis patterns"
    )
