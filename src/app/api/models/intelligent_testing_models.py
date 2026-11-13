"""Intelligent testing API models stub.

TODO: Implement intelligent testing API models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ContainerScopedTestPatternRequest(BaseModel):
    """Container-scoped test pattern request."""

    container_id: str
    patterns: list[str] = []


class ContainerScopedTestRecommendationRequest(BaseModel):
    """Container-scoped test recommendation request."""

    container_id: str
    file_paths: list[str] = []


class ContainerScopedSmartTestSelectionRequest(BaseModel):
    """Container-scoped smart test selection request."""

    container_id: str
    changed_files: list[str] = []


class TestPatternAnalysisResponse(BaseModel):
    """Test pattern analysis response."""

    patterns: list[dict[str, Any]] = []


class TestRecommendationResponse(BaseModel):
    """Test recommendation response."""

    recommendations: list[dict[str, Any]] = []


class SmartTestSelectionResponse(BaseModel):
    """Smart test selection response."""

    selected_tests: list[str] = []
    reason: str = ""


class TestImpactAnalysisRequest(BaseModel):
    """Test impact analysis request."""

    changed_files: list[str] = []


class TestImpactAnalysisResponse(BaseModel):
    """Test impact analysis response."""

    impacted_tests: list[str] = []
    impact_score: float = 0.0


class TestPrioritizationRequest(BaseModel):
    """Test prioritization request."""

    tests: list[str] = []


class TestPrioritizationResponse(BaseModel):
    """Test prioritization response."""

    prioritized_tests: list[dict[str, Any]] = []
