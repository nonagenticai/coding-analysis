"""
ML-based hotspot detection for architecture analysis.

This module provides comprehensive hotspot detection capabilities including
change frequency analysis, complexity analysis, bottleneck detection, and
reporting. The modular design allows for focused analysis of different aspects
of code hotspots and technical debt.
"""

from .bottleneck_detector import BottleneckDetector, BottleneckMetrics
from .change_frequency_analyzer import ChangeMetrics, GitAnalyzer
from .complexity_analyzer import ComplexityAnalyzer
from .detection_algorithms import HotspotDetector
from .hotspot_analyzer import HotspotAnalyzer
from .hotspot_reporter import HotspotReport, HotspotReporter

__all__ = [
    "HotspotAnalyzer",
    "GitAnalyzer",
    "ChangeMetrics",
    "ComplexityAnalyzer",
    "HotspotDetector",
    "BottleneckDetector",
    "BottleneckMetrics",
    "HotspotReporter",
    "HotspotReport",
]
