"""Module for hotspot reporting and visualization functionality."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from models.architecture_models import HotspotPrediction

logger = logging.getLogger(__name__)


class HotspotReport:
    """Represents a comprehensive hotspot analysis report."""

    def __init__(self, codebase_id: str) -> None:
        """Initialize a hotspot report.

        Args:
            codebase_id: Unique identifier for the analyzed codebase
        """
        self.codebase_id = codebase_id
        self.timestamp = datetime.now()
        self.hotspots: list[HotspotPrediction] = []
        self.bottlenecks: list[dict[str, Any]] = []
        self.summary_stats: dict[str, Any] = {}
        self.analysis_metadata: dict[str, Any] = {}

    def add_hotspots(self, hotspots: list[HotspotPrediction]) -> None:
        """Add hotspot predictions to the report.

        Args:
            hotspots: List of hotspot predictions to include
        """
        self.hotspots = hotspots

    def add_bottlenecks(self, bottlenecks: list[dict[str, Any]]) -> None:
        """Add bottleneck analysis results to the report.

        Args:
            bottlenecks: List of bottleneck analysis results
        """
        self.bottlenecks = bottlenecks

    def set_summary_stats(self, stats: dict[str, Any]) -> None:
        """Set summary statistics for the report.

        Args:
            stats: Dictionary containing summary statistics
        """
        self.summary_stats = stats

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """Set analysis metadata for the report.

        Args:
            metadata: Dictionary containing analysis metadata
        """
        self.analysis_metadata = metadata

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a dictionary representation.

        Returns:
            Dictionary containing all report data
        """
        return {
            "codebase_id": self.codebase_id,
            "timestamp": self.timestamp.isoformat(),
            "hotspots": [
                {
                    "component_id": h.component_id,
                    "score": h.score,
                    "confidence": h.confidence,
                    "predicted_issues": h.predicted_issues,
                    "risk_factors": h.risk_factors,
                    "recommended_actions": h.recommended_actions,
                }
                for h in self.hotspots
            ],
            "bottlenecks": self.bottlenecks,
            "summary_stats": self.summary_stats,
            "analysis_metadata": self.analysis_metadata,
        }


class HotspotReporter:
    """Generates reports and visualizations for hotspot analysis results.

    This class provides functionality to create comprehensive reports of hotspot
    analysis results, including summary statistics, detailed findings, and
    actionable recommendations. It supports multiple output formats including
    JSON, text, and structured data for visualization tools.
    """

    def __init__(self) -> None:
        """Initialize the hotspot reporter."""
        pass

    def generate_report(
        self,
        codebase_id: str,
        hotspots: list[HotspotPrediction],
        bottlenecks: list[dict[str, Any]] | None = None,
        analysis_metadata: dict[str, Any] | None = None,
    ) -> HotspotReport:
        """Generate a comprehensive hotspot analysis report.

        Args:
            codebase_id: Unique identifier for the analyzed codebase
            hotspots: List of hotspot predictions
            bottlenecks: Optional list of bottleneck analysis results
            analysis_metadata: Optional metadata about the analysis

        Returns:
            HotspotReport object containing all analysis results
        """
        logger.info(f"Generating hotspot report for codebase: {codebase_id}")

        report = HotspotReport(codebase_id)
        report.add_hotspots(hotspots)

        if bottlenecks:
            report.add_bottlenecks(bottlenecks)

        # Generate summary statistics
        summary_stats = self._calculate_summary_stats(hotspots, bottlenecks or [])
        report.set_summary_stats(summary_stats)

        # Set metadata
        if analysis_metadata:
            report.set_metadata(analysis_metadata)

        logger.info(f"Generated report with {len(hotspots)} hotspots")
        return report

    def _calculate_summary_stats(
        self,
        hotspots: list[HotspotPrediction],
        bottlenecks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate summary statistics for the analysis.

        Args:
            hotspots: List of hotspot predictions
            bottlenecks: List of bottleneck analysis results

        Returns:
            Dictionary containing summary statistics
        """
        stats = {
            "total_hotspots": len(hotspots),
            "total_bottlenecks": len(bottlenecks),
            "high_risk_hotspots": len([h for h in hotspots if h.score >= 0.8]),
            "medium_risk_hotspots": len([h for h in hotspots if 0.5 <= h.score < 0.8]),
            "average_hotspot_score": 0.0,
            "max_hotspot_score": 0.0,
            "issue_type_distribution": {},
            "risk_factor_distribution": {},
        }

        if hotspots:
            scores = [h.score for h in hotspots]
            stats["average_hotspot_score"] = sum(scores) / len(scores)
            stats["max_hotspot_score"] = max(scores)

            # Analyze issue type distribution
            issue_counts = {}
            for hotspot in hotspots:
                for issue in hotspot.predicted_issues:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
            stats["issue_type_distribution"] = issue_counts

            # Analyze risk factor distribution
            risk_counts = {}
            for hotspot in hotspots:
                for risk in hotspot.risk_factors:
                    # Extract risk type from the risk factor string
                    risk_type = risk.split("(")[0].strip()
                    risk_counts[risk_type] = risk_counts.get(risk_type, 0) + 1
            stats["risk_factor_distribution"] = risk_counts

        return stats

    def generate_text_report(self, report: HotspotReport) -> str:
        """Generate a human-readable text report.

        Args:
            report: HotspotReport to convert to text

        Returns:
            Formatted text report
        """
        lines = []
        lines.append("HOTSPOT ANALYSIS REPORT")
        lines.append("=" * 50)
        lines.append(f"Codebase: {report.codebase_id}")
        lines.append(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary section
        lines.append("SUMMARY")
        lines.append("-" * 20)
        stats = report.summary_stats
        lines.append(f"Total Hotspots: {stats.get('total_hotspots', 0)}")
        lines.append(f"High Risk (≥0.8): {stats.get('high_risk_hotspots', 0)}")
        lines.append(f"Medium Risk (0.5-0.8): {stats.get('medium_risk_hotspots', 0)}")
        lines.append(f"Average Score: {stats.get('average_hotspot_score', 0):.3f}")
        lines.append(f"Max Score: {stats.get('max_hotspot_score', 0):.3f}")
        lines.append("")

        # Issue type distribution
        if stats.get("issue_type_distribution"):
            lines.append("ISSUE TYPE DISTRIBUTION")
            lines.append("-" * 30)
            for issue_type, count in stats["issue_type_distribution"].items():
                lines.append(f"  {issue_type}: {count}")
            lines.append("")

        # Top hotspots
        lines.append("TOP HOTSPOTS")
        lines.append("-" * 20)
        top_hotspots = sorted(report.hotspots, key=lambda h: h.score, reverse=True)[:10]

        for i, hotspot in enumerate(top_hotspots, 1):
            lines.append(f"{i}. Component: {hotspot.component_id}")
            lines.append(f"   Score: {hotspot.score:.3f} (Confidence: {hotspot.confidence:.3f})")

            if hotspot.predicted_issues:
                lines.append(f"   Issues: {', '.join(hotspot.predicted_issues)}")

            if hotspot.risk_factors:
                lines.append("   Risk Factors:")
                for risk in hotspot.risk_factors:
                    lines.append(f"     - {risk}")

            if hotspot.recommended_actions:
                lines.append("   Recommendations:")
                for action in hotspot.recommended_actions:
                    lines.append(f"     - {action}")
            lines.append("")

        # Bottlenecks section
        if report.bottlenecks:
            lines.append("PERFORMANCE BOTTLENECKS")
            lines.append("-" * 30)
            top_bottlenecks = sorted(
                report.bottlenecks, key=lambda b: b.get("score", 0), reverse=True
            )[:5]

            for i, bottleneck in enumerate(top_bottlenecks, 1):
                lines.append(f"{i}. Component: {bottleneck.get('component_id', 'Unknown')}")
                lines.append(f"   Score: {bottleneck.get('score', 0):.3f}")

                if bottleneck.get("risk_factors"):
                    lines.append(f"   Risk Factors: {', '.join(bottleneck['risk_factors'])}")

                if bottleneck.get("recommendations"):
                    lines.append("   Recommendations:")
                    for rec in bottleneck["recommendations"]:
                        lines.append(f"     - {rec}")
                lines.append("")

        return "\n".join(lines)

    def save_report_json(self, report: HotspotReport, output_path: Path) -> None:
        """Save the report as a JSON file.

        Args:
            report: HotspotReport to save
            output_path: Path where to save the JSON file
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON report to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")

    def save_report_text(self, report: HotspotReport, output_path: Path) -> None:
        """Save the report as a text file.

        Args:
            report: HotspotReport to save
            output_path: Path where to save the text file
        """
        try:
            text_report = self.generate_text_report(report)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text_report)
            logger.info(f"Saved text report to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save text report: {e}")

    def generate_visualization_data(self, report: HotspotReport) -> dict[str, Any]:
        """Generate data structures suitable for visualization tools.

        Args:
            report: HotspotReport to convert for visualization

        Returns:
            Dictionary containing visualization-ready data
        """
        viz_data = {
            "hotspot_scores": [
                {
                    "component": h.component_id,
                    "score": h.score,
                    "confidence": h.confidence,
                }
                for h in report.hotspots
            ],
            "score_distribution": self._calculate_score_distribution(report.hotspots),
            "issue_type_counts": report.summary_stats.get("issue_type_distribution", {}),
            "risk_factor_counts": report.summary_stats.get("risk_factor_distribution", {}),
            "recommendation_frequency": self._calculate_recommendation_frequency(report.hotspots),
        }

        if report.bottlenecks:
            viz_data["bottleneck_scores"] = [
                {
                    "component": b.get("component_id", "Unknown"),
                    "score": b.get("score", 0),
                    "metrics": b.get("metrics", {}),
                }
                for b in report.bottlenecks
            ]

        return viz_data

    def _calculate_score_distribution(self, hotspots: list[HotspotPrediction]) -> dict[str, int]:
        """Calculate distribution of hotspot scores in ranges.

        Args:
            hotspots: List of hotspot predictions

        Returns:
            Dictionary mapping score ranges to counts
        """
        distribution = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }

        for hotspot in hotspots:
            score = hotspot.score
            if score < 0.2:
                distribution["0.0-0.2"] += 1
            elif score < 0.4:
                distribution["0.2-0.4"] += 1
            elif score < 0.6:
                distribution["0.4-0.6"] += 1
            elif score < 0.8:
                distribution["0.6-0.8"] += 1
            else:
                distribution["0.8-1.0"] += 1

        return distribution

    def _calculate_recommendation_frequency(
        self, hotspots: list[HotspotPrediction]
    ) -> dict[str, int]:
        """Calculate frequency of different recommendation types.

        Args:
            hotspots: List of hotspot predictions

        Returns:
            Dictionary mapping recommendation types to frequencies
        """
        recommendation_counts = {}

        for hotspot in hotspots:
            for recommendation in hotspot.recommended_actions:
                # Normalize recommendation text for counting
                normalized = recommendation.lower()

                if "refactor" in normalized:
                    rec_type = "Refactoring"
                elif "test" in normalized:
                    rec_type = "Testing"
                elif "document" in normalized:
                    rec_type = "Documentation"
                elif "split" in normalized or "decompose" in normalized:
                    rec_type = "Decomposition"
                elif "monitor" in normalized:
                    rec_type = "Monitoring"
                elif "architect" in normalized:
                    rec_type = "Architecture"
                else:
                    rec_type = "Other"

                recommendation_counts[rec_type] = recommendation_counts.get(rec_type, 0) + 1

        return recommendation_counts

    def create_dashboard_summary(self, report: HotspotReport) -> dict[str, Any]:
        """Create a summary suitable for dashboard display.

        Args:
            report: HotspotReport to summarize

        Returns:
            Dictionary containing dashboard-ready summary data
        """
        stats = report.summary_stats

        # Determine overall health score
        total_files = stats.get("total_hotspots", 0) + 100  # Assume baseline of files
        high_risk_ratio = stats.get("high_risk_hotspots", 0) / max(1, total_files)
        health_score = max(0, 100 - (high_risk_ratio * 100))

        # Risk level determination
        if high_risk_ratio > 0.1:
            risk_level = "HIGH"
        elif high_risk_ratio > 0.05:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "codebase_id": report.codebase_id,
            "timestamp": report.timestamp.isoformat(),
            "health_score": round(health_score, 1),
            "risk_level": risk_level,
            "total_hotspots": stats.get("total_hotspots", 0),
            "high_risk_count": stats.get("high_risk_hotspots", 0),
            "medium_risk_count": stats.get("medium_risk_hotspots", 0),
            "average_score": round(stats.get("average_hotspot_score", 0), 3),
            "top_issues": list(stats.get("issue_type_distribution", {}).keys())[:3],
            "bottleneck_count": len(report.bottlenecks),
            "needs_attention": stats.get("high_risk_hotspots", 0) > 0,
        }
