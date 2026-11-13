"""Similarity module."""

from __future__ import annotations

"""
Code Similarity Detection API Endpoints

Modular API endpoints for detecting code similarities, clones, and cross-language patterns.
This module serves as the main entry point and delegates to focused sub-modules.

The original 1039-line file has been split into focused modules:
- similarity_endpoints.py: Main API endpoints
- similarity_algorithms.py: Code comparison algorithms
- similarity_metrics.py: Similarity metrics and scoring
- similarity_processing.py: Result processing and formatting
- similarity_config.py: Configuration and utilities

Each module is under 500 lines and maintains the original functionality.
"""

import logging

from fastapi import APIRouter

from .similarity_modules import router as similarity_router

logger = logging.getLogger(__name__)

# Main router that includes all similarity endpoints
router = APIRouter()

# Include the modular similarity router
router.include_router(similarity_router)

# Export key components for backward compatibility
from .similarity_modules import (
    CodeFeatureExtractor,
    SimilarityAlgorithms,
    SimilarityConfig,
    SimilarityMetrics,
    SimilarityProcessor,
    SimilarityScoring,
    SimilarityThresholds,
)

__all__ = [
    "router",
    "SimilarityConfig",
    "SimilarityProcessor",
    "SimilarityMetrics",
    "SimilarityScoring",
    "SimilarityThresholds",
    "CodeFeatureExtractor",
    "SimilarityAlgorithms",
]
