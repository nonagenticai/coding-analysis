"""Pattern reporting utilities for statistics, filtering, and analysis.

This module provides the PatternReportingUtils class which handles pattern
statistics generation, filtering, and reporting functionality.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from models.architecture_models import DetectedPattern

logger = logging.getLogger(__name__)


class PatternReportingUtils:
    """Utility class for pattern reporting and analysis."""

    def __init__(self) -> None:
        """Initialize the pattern reporting utilities."""
        pass

    def get_pattern_statistics(self, patterns: list[DetectedPattern]) -> dict[str, int]:
        """Get statistics about detected patterns.

        Args:
            patterns: List of detected patterns

        Returns:
            Dictionary with pattern names as keys and their counts as values
        """
        stats: dict[str, int] = {}

        for pattern in patterns:
            pattern_name = pattern.name
            if pattern_name and pattern_name not in stats:
                stats[pattern_name] = 0
            if pattern_name:
                stats[pattern_name] += 1

        return stats

    def filter_patterns_by_confidence(
        self, patterns: list[DetectedPattern], min_confidence: float = 0.7
    ) -> list[DetectedPattern]:
        """Filter patterns by minimum confidence threshold.

        Args:
            patterns: List of detected patterns
            min_confidence: Minimum confidence threshold (default: 0.7)

        Returns:
            List of patterns with confidence above the threshold
        """
        return [pattern for pattern in patterns if pattern.confidence >= min_confidence]

    def filter_patterns_by_type(
        self, patterns: list[DetectedPattern], pattern_types: list[str]
    ) -> list[DetectedPattern]:
        """Filter patterns by their types.

        Args:
            patterns: List of detected patterns
            pattern_types: List of pattern types to include

        Returns:
            List of patterns matching the specified types
        """
        return [pattern for pattern in patterns if pattern.pattern_type in pattern_types]

    def group_patterns_by_type(
        self, patterns: list[DetectedPattern]
    ) -> dict[str, list[DetectedPattern]]:
        """Group patterns by their types.

        Args:
            patterns: List of detected patterns

        Returns:
            Dictionary with pattern types as keys and lists of patterns as values
        """
        grouped: dict[str, list[DetectedPattern]] = defaultdict(list)

        for pattern in patterns:
            grouped[pattern.pattern_type].append(pattern)

        return dict(grouped)

    def get_patterns_by_location(
        self, patterns: list[DetectedPattern]
    ) -> dict[str, list[DetectedPattern]]:
        """Group patterns by their location.

        Args:
            patterns: List of detected patterns

        Returns:
            Dictionary with locations as keys and lists of patterns as values
        """
        by_location: dict[str, list[DetectedPattern]] = defaultdict(list)

        for pattern in patterns:
            if pattern.location:
                by_location[pattern.location].append(pattern)

        return dict(by_location)

    def get_confidence_distribution(self, patterns: list[DetectedPattern]) -> dict[str, int]:
        """Get distribution of confidence scores.

        Args:
            patterns: List of detected patterns

        Returns:
            Dictionary with confidence ranges as keys and counts as values
        """
        distribution = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }

        for pattern in patterns:
            confidence = pattern.confidence
            if confidence < 0.2:
                distribution["0.0-0.2"] += 1
            elif confidence < 0.4:
                distribution["0.2-0.4"] += 1
            elif confidence < 0.6:
                distribution["0.4-0.6"] += 1
            elif confidence < 0.8:
                distribution["0.6-0.8"] += 1
            else:
                distribution["0.8-1.0"] += 1

        return distribution

    def get_pattern_component_stats(self, patterns: list[DetectedPattern]) -> dict[str, Any]:
        """Get statistics about pattern components.

        Args:
            patterns: List of detected patterns

        Returns:
            Dictionary with component statistics
        """
        stats = {
            "total_patterns": len(patterns),
            "total_components": 0,
            "avg_components_per_pattern": 0.0,
            "max_components": 0,
            "min_components": float("inf"),
            "patterns_with_single_component": 0,
            "patterns_with_multiple_components": 0,
        }

        if not patterns:
            return stats

        component_counts = []
        for pattern in patterns:
            component_count = len(pattern.components)
            component_counts.append(component_count)
            stats["total_components"] += component_count

            if component_count == 1:
                stats["patterns_with_single_component"] += 1
            else:
                stats["patterns_with_multiple_components"] += 1

        stats["avg_components_per_pattern"] = stats["total_components"] / len(patterns)
        stats["max_components"] = max(component_counts)
        stats["min_components"] = min(component_counts)

        return stats

    def get_top_patterns_by_confidence(
        self, patterns: list[DetectedPattern], limit: int = 10
    ) -> list[DetectedPattern]:
        """Get top patterns sorted by confidence.

        Args:
            patterns: List of detected patterns
            limit: Maximum number of patterns to return

        Returns:
            List of patterns sorted by confidence (highest first)
        """
        sorted_patterns = sorted(patterns, key=lambda p: p.confidence, reverse=True)
        return sorted_patterns[:limit]

    def get_patterns_summary(self, patterns: list[DetectedPattern]) -> dict[str, Any]:
        """Get comprehensive summary of detected patterns.

        Args:
            patterns: List of detected patterns

        Returns:
            Dictionary with comprehensive pattern summary
        """
        summary = {
            "overview": {
                "total_patterns": len(patterns),
                "unique_types": len(set(p.pattern_type for p in patterns)),
                "avg_confidence": 0.0,
                "high_confidence_patterns": 0,
            },
            "statistics": self.get_pattern_statistics(patterns),
            "confidence_distribution": self.get_confidence_distribution(patterns),
            "component_stats": self.get_pattern_component_stats(patterns),
            "by_type": {},
            "top_patterns": [],
        }

        if patterns:
            # Calculate average confidence
            total_confidence = sum(p.confidence for p in patterns)
            summary["overview"]["avg_confidence"] = total_confidence / len(patterns)

            # Count high confidence patterns (>= 0.8)
            summary["overview"]["high_confidence_patterns"] = len(
                [p for p in patterns if p.confidence >= 0.8]
            )

            # Group by type with summary stats
            grouped = self.group_patterns_by_type(patterns)
            for pattern_type, type_patterns in grouped.items():
                summary["by_type"][pattern_type] = {
                    "count": len(type_patterns),
                    "avg_confidence": sum(p.confidence for p in type_patterns) / len(type_patterns),
                    "max_confidence": max(p.confidence for p in type_patterns),
                    "min_confidence": min(p.confidence for p in type_patterns),
                }

            # Get top patterns
            summary["top_patterns"] = [
                {
                    "type": p.pattern_type,
                    "name": p.name,
                    "confidence": p.confidence,
                    "location": p.location,
                    "component_count": len(p.components),
                }
                for p in self.get_top_patterns_by_confidence(patterns, 5)
            ]

        return summary

    def generate_pattern_report(self, patterns: list[DetectedPattern]) -> str:
        """Generate a formatted text report of detected patterns.

        Args:
            patterns: List of detected patterns

        Returns:
            Formatted text report
        """
        summary = self.get_patterns_summary(patterns)

        report_lines = [
            "Pattern Detection Report",
            "=" * 50,
            "",
            f"Total Patterns Detected: {summary['overview']['total_patterns']}",
            f"Unique Pattern Types: {summary['overview']['unique_types']}",
            f"Average Confidence: {summary['overview']['avg_confidence']:.2f}",
            f"High Confidence Patterns (≥0.8): {summary['overview']['high_confidence_patterns']}",
            "",
            "Pattern Type Distribution:",
            "-" * 30,
        ]

        # Add pattern type statistics
        for pattern_type, count in summary["statistics"].items():
            report_lines.append(f"  {pattern_type}: {count}")

        report_lines.extend(
            [
                "",
                "Confidence Distribution:",
                "-" * 25,
            ]
        )

        # Add confidence distribution
        for range_str, count in summary["confidence_distribution"].items():
            report_lines.append(f"  {range_str}: {count}")

        report_lines.extend(
            [
                "",
                "Top Patterns by Confidence:",
                "-" * 30,
            ]
        )

        # Add top patterns
        for i, pattern_info in enumerate(summary["top_patterns"], 1):
            report_lines.append(
                f"  {i}. {pattern_info['type']} - {pattern_info['confidence']:.2f} "
                f"({pattern_info['component_count']} components)"
            )

        report_lines.extend(
            [
                "",
                "Component Statistics:",
                "-" * 25,
                f"  Total Components: {summary['component_stats']['total_components']}",
                f"  Avg Components/Pattern: {summary['component_stats']['avg_components_per_pattern']:.1f}",
                f"  Max Components: {summary['component_stats']['max_components']}",
                f"  Min Components: {summary['component_stats']['min_components']}",
                f"  Single Component Patterns: {summary['component_stats']['patterns_with_single_component']}",
                f"  Multi Component Patterns: {summary['component_stats']['patterns_with_multiple_components']}",
            ]
        )

        return "\n".join(report_lines)

    def export_patterns_to_dict(self, patterns: list[DetectedPattern]) -> list[dict]:
        """Export patterns to a list of dictionaries for serialization.

        Args:
            patterns: List of detected patterns

        Returns:
            List of pattern dictionaries
        """
        return [
            {
                "pattern_type": pattern.pattern_type,
                "name": pattern.name,
                "confidence": pattern.confidence,
                "components": pattern.components,
                "description": pattern.description,
                "location": pattern.location,
            }
            for pattern in patterns
        ]

    def find_pattern_overlaps(self, patterns: list[DetectedPattern]) -> list[dict]:
        """Find patterns that share components (potential overlaps).

        Args:
            patterns: List of detected patterns

        Returns:
            List of dictionaries describing pattern overlaps
        """
        overlaps = []

        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i + 1 :], i + 1):
                # Find common components
                common_components = set(pattern1.components) & set(pattern2.components)

                if common_components:
                    overlap_ratio = len(common_components) / max(
                        len(pattern1.components), len(pattern2.components)
                    )

                    overlaps.append(
                        {
                            "pattern1": {
                                "type": pattern1.pattern_type,
                                "name": pattern1.name,
                                "confidence": pattern1.confidence,
                            },
                            "pattern2": {
                                "type": pattern2.pattern_type,
                                "name": pattern2.name,
                                "confidence": pattern2.confidence,
                            },
                            "common_components": list(common_components),
                            "overlap_ratio": overlap_ratio,
                        }
                    )

        return overlaps
