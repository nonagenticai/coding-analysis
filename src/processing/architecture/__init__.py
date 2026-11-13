"""Architecture processing package.

This package contains modules for processing and analyzing code architecture patterns,
dependencies, and architectural metrics.
"""

from __future__ import annotations

# Main processor - Entry point for architecture analysis
from .architecture_processor import ArchitectureProcessor
from .circular_detection import CircularDependencyDetector

# Component extraction
from .component_extractor import ComponentExtractor
from .component_extractor_helpers import ComponentExtractionHelpers
from .component_extractor_parallel import ComponentExtractorParallel
from .dependency_analysis import DependencyAnalyzer

# Core analyzers - Primary analysis components
from .dependency_analyzer import ArchitectureDependencyAnalyzer

# Dependency analysis components
from .dependency_extraction import DependencyExtractor
from .dependency_graph import DependencyGraphBuilder

# Plugin interfaces
from .dependency_plugins import AbstractDepLoader, BaseDepLoader
from .dependency_visualization import DependencyVulnerabilityScanner

# Graph integration
from .graph_adapter import GraphAdapter

# Hotspot analysis (re-export from submodule)
from .hotspot import (
    BottleneckDetector,
    BottleneckMetrics,
    ChangeMetrics,
    ComplexityAnalyzer,
    GitAnalyzer,
    HotspotAnalyzer,
    HotspotDetector,
    HotspotReport,
    HotspotReporter,
)
from .layer_analyzer import LayerAnalyzer
from .metrics_calculator import MetricsCalculator
from .ml_confidence_calculator import MLConfidenceCalculator
from .pattern_detector import ArchitecturePatternDetector

# Metrics and performance tracking
from .performance_metrics_tracker import (
    ArchitectureMetricTimeSeries,
    MetricTimeSeries,
    TrendType,
)
from .performance_optimizer import PerformanceOptimizer
from .quality_analyzer import ArchitectureQualityAnalyzer

# Runtime analysis
from .runtime_trace_ingestor import RuntimeSpan, RuntimeTraceIngestor
from .time_series_metrics import TimeSeriesMetricsCalculator

__all__ = [
    # Main processor
    "ArchitectureProcessor",
    # Core analyzers
    "ArchitectureDependencyAnalyzer",
    "ArchitecturePatternDetector",
    "ArchitectureQualityAnalyzer",
    "LayerAnalyzer",
    "MetricsCalculator",
    # Component extraction
    "ComponentExtractor",
    "ComponentExtractorParallel",
    "ComponentExtractionHelpers",
    # Graph integration
    "GraphAdapter",
    # Runtime analysis
    "RuntimeTraceIngestor",
    "RuntimeSpan",
    # Dependency analysis
    "DependencyExtractor",
    "DependencyGraphBuilder",
    "DependencyAnalyzer",
    "CircularDependencyDetector",
    "DependencyVulnerabilityScanner",
    # Metrics and performance
    "ArchitectureMetricTimeSeries",
    "MetricTimeSeries",
    "TrendType",
    "PerformanceOptimizer",
    "TimeSeriesMetricsCalculator",
    "MLConfidenceCalculator",
    # Hotspot analysis
    "HotspotAnalyzer",
    "GitAnalyzer",
    "ChangeMetrics",
    "ComplexityAnalyzer",
    "HotspotDetector",
    "BottleneckDetector",
    "BottleneckMetrics",
    "HotspotReporter",
    "HotspotReport",
    # Plugin interfaces
    "BaseDepLoader",
    "AbstractDepLoader",
]
