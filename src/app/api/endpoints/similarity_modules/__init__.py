"""Similarity detection modules package.

This package contains modular components for code similarity detection.
"""

from .similarity_algorithms import (
    CodeFeatureExtractor,
    SimilarityAlgorithms,
    calculate_similarity,
    detect_code_clones,
    find_similar_functions,
)
from .similarity_config import SimilarityConfig, get_similarity_config
from .similarity_endpoints import router
from .similarity_endpoints import router as similarity_router
from .similarity_metrics import (
    SimilarityMetric,
    SimilarityMetrics,
    SimilarityScoring,
    SimilarityThresholds,
    calculate_metrics,
    get_metric_weights,
)
from .similarity_processing import (
    SimilarityProcessor,
    batch_process_similarities,
    process_similarity_request,
)

__all__ = [
    # Router
    "router",
    "similarity_router",
    # Algorithms
    "calculate_similarity",
    "find_similar_functions",
    "detect_code_clones",
    "CodeFeatureExtractor",
    "SimilarityAlgorithms",
    # Metrics
    "SimilarityMetric",
    "calculate_metrics",
    "get_metric_weights",
    "SimilarityMetrics",
    "SimilarityScoring",
    "SimilarityThresholds",
    # Processing
    "process_similarity_request",
    "batch_process_similarities",
    "SimilarityProcessor",
    # Configuration
    "SimilarityConfig",
    "get_similarity_config",
]
