"""Module for aggregating metrics and generating comprehensive reports."""

from __future__ import annotations

import datetime
import logging
from collections import defaultdict
from typing import Any

import networkx as nx

from models.architecture_models import Component

from .performance_metrics_tracker import (
    ArchitectureMetricTimeSeries,
    MetricTimeSeries,
    TrendType,
)

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """Aggregates metrics and generates comprehensive reports across components and time."""

    def compare_metrics_across_commits(
        self,
        component_id: str,
        commit_data: list[tuple[str, datetime.datetime, list[Component], nx.DiGraph]],
        calculate_component_metrics_func,
        calculate_weighted_quality_score_func,
    ) -> dict[str, MetricTimeSeries]:
        """Compare metrics for a component across multiple git commits.

        Args:
            component_id: ID of the component to analyze
            commit_data: List of tuples (commit_id, timestamp, components, graph) for each commit
            calculate_component_metrics_func: Function to calculate component metrics
            calculate_weighted_quality_score_func: Function to calculate weighted quality score

        Returns:
            Dictionary of metric name to MetricTimeSeries objects
        """
        logger.info(
            "Comparing metrics for component '%s' across %d commits",
            component_id,
            len(commit_data),
        )

        # Initialize time series for each metric type
        time_series = {
            "instability": ArchitectureMetricTimeSeries("instability", component_id),
            "abstractness": ArchitectureMetricTimeSeries("abstractness", component_id),
            "distance": ArchitectureMetricTimeSeries("distance", component_id),
            "lcom4": ArchitectureMetricTimeSeries("lcom4", component_id),
            "tcc": ArchitectureMetricTimeSeries("tcc", component_id),
            "weighted_quality": ArchitectureMetricTimeSeries("weighted_quality", component_id),
        }

        for commit_id, timestamp, components, graph in commit_data:
            # Find the component in this commit
            component = next((c for c in components if c.id == component_id), None)
            if not component:
                logger.warning(
                    "Component '%s' not found in commit %s",
                    component_id,
                    commit_id[:8],
                )
                continue

            # Calculate metrics for this commit
            metrics_dict = calculate_component_metrics_func([component], graph)
            if component_id not in metrics_dict:
                logger.warning(
                    "Failed to calculate metrics for component '%s' in commit %s",
                    component_id,
                    commit_id[:8],
                )
                continue

            metrics = metrics_dict[component_id]

            # Add data points to time series
            for metric_name, time_series_obj in time_series.items():
                if metric_name in metrics:
                    value = metrics[metric_name]
                elif metric_name == "weighted_quality":
                    # Calculate weighted quality for just this component
                    value = calculate_weighted_quality_score_func({component_id: metrics})
                else:
                    continue

                time_series_obj.add_data_point(
                    timestamp=timestamp,
                    commit_id=commit_id,
                    value=value,
                    raw_data=metrics,
                )

        return time_series

    def get_architecture_evolution_report(
        self, component_id: str, time_series_data: dict[str, MetricTimeSeries]
    ) -> dict[str, Any]:
        """Generate a comprehensive report on architectural evolution for a component.

        Args:
            component_id: ID of the component being analyzed
            time_series_data: Dictionary of metric name to MetricTimeSeries objects

        Returns:
            Dictionary with architecture evolution analysis
        """
        if not time_series_data:
            return {
                "component_id": component_id,
                "metrics_analyzed": 0,
                "commits_analyzed": 0,
                "timespan_days": 0.0,
                "trends": {},
                "has_degradation": False,
                "overall_quality_trend": "unknown",
            }

        # Get trend analysis for each metric
        trends = {}
        degradation_found = False
        degrading_metrics = []
        improving_metrics = []
        sample_timespan = 0.0
        commits_analyzed = 0

        for metric_name, time_series in time_series_data.items():
            # Get trend analysis
            trend_analysis = time_series.get_trend_analysis()
            trends[metric_name] = {
                "trend": trend_analysis.trend.value,
                "confidence": trend_analysis.confidence,
                "percent_change": trend_analysis.percent_change,
                "rate_of_change": trend_analysis.rate_of_change,
                "data_points": trend_analysis.data_points,
                "forecast": trend_analysis.forecast_value,
                "forecast_confidence": trend_analysis.forecast_confidence,
            }

            # Track degrading metrics
            if trend_analysis.trend == TrendType.DEGRADING and trend_analysis.confidence > 0.6:
                degradation_found = True
                degrading_metrics.append(metric_name)
            elif trend_analysis.trend == TrendType.IMPROVING and trend_analysis.confidence > 0.6:
                improving_metrics.append(metric_name)

            # Update analysis metadata
            sample_timespan = max(sample_timespan, trend_analysis.timespan_days)
            commits_analyzed = max(commits_analyzed, trend_analysis.data_points)

        # Determine overall quality trend
        overall_quality_trend = "unknown"
        if "weighted_quality" in time_series_data:
            quality_trend = time_series_data["weighted_quality"].get_trend_analysis().trend
            overall_quality_trend = quality_trend.value
        elif len(degrading_metrics) > len(improving_metrics):
            overall_quality_trend = "degrading"
        elif len(improving_metrics) > len(degrading_metrics):
            overall_quality_trend = "updating"
        elif len(improving_metrics) == 0 and len(degrading_metrics) == 0:
            overall_quality_trend = "stable"
        else:
            # Same number of updating and degrading metrics
            overall_quality_trend = "mixed"

        # Generate summary report
        return {
            "component_id": component_id,
            "metrics_analyzed": len(time_series_data),
            "commits_analyzed": commits_analyzed,
            "timespan_days": sample_timespan,
            "start_date": (
                min(
                    ts.data_points[0].timestamp.isoformat()
                    for ts in time_series_data.values()
                    if ts.data_points
                )
                if any(ts.data_points for ts in time_series_data.values())
                else None
            ),
            "end_date": (
                max(
                    ts.data_points[-1].timestamp.isoformat()
                    for ts in time_series_data.values()
                    if ts.data_points
                )
                if any(ts.data_points for ts in time_series_data.values())
                else None
            ),
            "trends": trends,
            "has_degradation": degradation_found,
            "degrading_metrics": degrading_metrics,
            "improving_metrics": improving_metrics,
            "overall_quality_trend": overall_quality_trend,
        }

    def get_historical_data_for_component(
        self,
        component_id: str,
        git_history_data: dict[str, Any],
        metric_name: str | None = None,
    ) -> dict[str, Any]:
        """Get historical data for visualization of component metrics over time.

        Args:
            component_id: ID of the component to analyze
            git_history_data: Dictionary with git history data from compare_metrics_across_commits
            metric_name: Optional specific metric to retrieve data for

        Returns:
            Dictionary with formatted data for visualization
        """
        result: dict[str, Any] = {"component_id": component_id, "metrics": {}}

        # Filter to specific metric if requested
        metrics_to_process = [metric_name] if metric_name else list(git_history_data.keys())

        for m_name in metrics_to_process:
            if m_name in git_history_data:
                # Get visualization data for this metric
                viz_data = git_history_data[m_name].get_visualization_data()
                result["metrics"][m_name] = viz_data

        return result

    def detect_architectural_decay(
        self,
        component_ids: list[str],
        time_series_collection: dict[str, dict[str, MetricTimeSeries]],
    ) -> dict[str, Any]:
        """Detect architectural decay across multiple components over time.

        This method analyzes trends across components to identify systematic decay patterns
        in the architecture that may indicate technical debt accumulation.

        Args:
            component_ids: List of component IDs to analyze
            time_series_collection: Dict mapping component IDs to their metric time series data

        Returns:
            Dictionary with decay analysis results
        """
        decay_results = {
            "components_analyzed": len(component_ids),
            "components_with_decay": [],
            "metrics_with_decay": defaultdict(int),
            "system_decay_score": 0.0,
            "recommended_focus_areas": [],
            "decay_by_component": {},
        }

        # Count components with decay signs
        decay_count = 0
        max_timespan = 0.0

        # Analyze each component
        for component_id in component_ids:
            if component_id not in time_series_collection:
                continue

            # Get evolution report for this component
            component_report = self.get_architecture_evolution_report(
                component_id, time_series_collection[component_id]
            )

            # Extract key insights from the report
            has_decay = component_report["has_degradation"]
            degrading_metrics = component_report["degrading_metrics"]
            timespan = component_report["timespan_days"]
            max_timespan = max(max_timespan, timespan)

            # Track decay metrics across all components
            for metric in degrading_metrics:
                if isinstance(decay_results["metrics_with_decay"], defaultdict):
                    decay_results["metrics_with_decay"][metric] += 1

            # Add component to decay list if it shows decay signs
            if has_decay:
                decay_count += 1
                if isinstance(decay_results["components_with_decay"], list):
                    decay_results["components_with_decay"].append(component_id)

            # Store component-specific decay information
            if isinstance(decay_results["decay_by_component"], dict):
                decay_results["decay_by_component"][component_id] = {
                    "has_decay": has_decay,
                    "degrading_metrics": degrading_metrics,
                    "overall_trend": component_report["overall_quality_trend"],
                    "decay_severity": (
                        len(degrading_metrics) / len(component_report["trends"])
                        if component_report["trends"]
                        else 0.0
                    ),
                }

        # Calculate system-wide decay score (0 to 1, higher means more decay)
        if component_ids:
            decay_results["system_decay_score"] = decay_count / len(component_ids)

        # Determine most common decay metrics across components
        metrics_with_decay = decay_results["metrics_with_decay"]
        if hasattr(metrics_with_decay, "items"):
            common_decay_metrics = sorted(
                metrics_with_decay.items(), key=lambda x: x[1], reverse=True
            )
        else:
            common_decay_metrics = []

        # Generate recommended focus areas
        recommendations = []
        for metric, count in common_decay_metrics:
            if count >= len(component_ids) * 0.2:  # Affecting at least 20% of components
                if metric == "distance":
                    recommendations.append(
                        "Main Sequence Alignment: Components are drifting from the optimal balance of abstractness and instability"
                    )
                elif metric == "instability":
                    recommendations.append(
                        "Dependency Management: Components are accumulating too many outgoing dependencies"
                    )
                elif metric == "abstractness":
                    recommendations.append(
                        "Interface Design: Update abstract component definitions and interfaces"
                    )
                elif metric == "lcom4":
                    recommendations.append(
                        "Class Cohesion: Classes have fragmented responsibilities and poor cohesion"
                    )
                elif metric == "tcc":
                    recommendations.append(
                        "Method Cohesion: Update method-to-field relationships and encapsulation"
                    )
                elif metric == "weighted_quality":
                    recommendations.append(
                        "Overall Architecture: General architectural quality is declining"
                    )

        decay_results["recommended_focus_areas"] = recommendations
        decay_results["analysis_timespan_days"] = max_timespan

        return decay_results

    def generate_system_metrics_summary(
        self,
        components: list[Component],
        component_metrics: dict[str, dict[str, float]],
        calculate_weighted_quality_score_func,
    ) -> dict[str, float]:
        """Generate a summary of all architectural metrics across the system.

        Args:
            components: List of components to analyze
            component_metrics: Dictionary of component metrics
            calculate_weighted_quality_score_func: Function to calculate weighted quality score

        Returns:
            Dictionary of summarized metric values
        """
        if not components:
            return {
                "total_components": 0,
                "avg_instability": 0.0,
                "avg_abstractness": 0.0,
                "avg_distance": 0.0,
                "max_distance": 0.0,
                "components_in_main_sequence": 0,
                "avg_cohesion_lcom4": 0.0,
                "avg_cohesion_tcc": 0.0,
            }

        instabilities = [m["instability"] for m in component_metrics.values()]
        abstractnesses = [m["abstractness"] for m in component_metrics.values()]
        distances = [m["distance"] for m in component_metrics.values()]

        # Get cohesion metrics if available
        lcom4_values = [
            m.get("lcom4", 0.0) for m in component_metrics.values() if m.get("lcom4") is not None
        ]
        tcc_values = [
            m.get("tcc", 0.0) for m in component_metrics.values() if m.get("tcc") is not None
        ]

        # Count components close to main sequence (distance < 0.1)
        components_in_main_sequence = sum(1 for d in distances if d < 0.1)

        return {
            "total_components": len(components),
            "avg_instability": (sum(instabilities) / len(instabilities) if instabilities else 0.0),
            "avg_abstractness": (
                sum(abstractnesses) / len(abstractnesses) if abstractnesses else 0.0
            ),
            "avg_distance": (sum(distances) / len(distances) if distances else 0.0),
            "max_distance": max(distances) if distances else 0.0,
            "components_in_main_sequence": components_in_main_sequence,
            "avg_cohesion_lcom4": (sum(lcom4_values) / len(lcom4_values) if lcom4_values else 0.0),
            "avg_cohesion_tcc": (sum(tcc_values) / len(tcc_values) if tcc_values else 0.0),
            "weighted_quality_score": calculate_weighted_quality_score_func(component_metrics),
        }

    def generate_component_evolution_dashboard(
        self,
        component_ids: list[str],
        time_series_collection: dict[str, dict[str, MetricTimeSeries]],
    ) -> dict[str, Any]:
        """Generate a dashboard-ready summary of component evolution data.

        Args:
            component_ids: List of component IDs to analyze
            time_series_collection: Dict mapping component IDs to their metric time series data

        Returns:
            Dictionary with dashboard data including trends, alerts, and summaries
        """
        dashboard_data = {
            "summary": {
                "total_components": len(component_ids),
                "components_with_data": 0,
                "alert_count": 0,
                "stable_components": 0,
                "improving_components": 0,
                "degrading_components": 0,
            },
            "components": {},
            "system_alerts": [],
            "trending_metrics": defaultdict(int),
        }

        for component_id in component_ids:
            if component_id not in time_series_collection:
                continue

            dashboard_data["summary"]["components_with_data"] += 1

            # Get evolution report for this component
            evolution_report = self.get_architecture_evolution_report(
                component_id, time_series_collection[component_id]
            )

            # Classify component trend
            overall_trend = evolution_report["overall_quality_trend"]
            if overall_trend == "stable":
                dashboard_data["summary"]["stable_components"] += 1
            elif overall_trend in ["updating", "improving"]:
                dashboard_data["summary"]["improving_components"] += 1
            elif overall_trend == "degrading":
                dashboard_data["summary"]["degrading_components"] += 1

            # Generate alerts for concerning trends
            alerts = []
            if evolution_report["has_degradation"]:
                alerts.append(
                    {
                        "severity": "warning",
                        "message": f"Component {component_id} shows degradation in {len(evolution_report['degrading_metrics'])} metrics",
                        "metrics": evolution_report["degrading_metrics"],
                    }
                )
                dashboard_data["summary"]["alert_count"] += 1

            # Track trending metrics across system
            for metric in evolution_report["degrading_metrics"]:
                dashboard_data["trending_metrics"][f"{metric}_degrading"] += 1
            for metric in evolution_report["improving_metrics"]:
                dashboard_data["trending_metrics"][f"{metric}_improving"] += 1

            # Store component data
            dashboard_data["components"][component_id] = {
                "overall_trend": overall_trend,
                "metrics_count": evolution_report["metrics_analyzed"],
                "timespan_days": evolution_report["timespan_days"],
                "alerts": alerts,
                "degrading_metrics": evolution_report["degrading_metrics"],
                "improving_metrics": evolution_report["improving_metrics"],
            }

        # Generate system-level alerts
        if (
            dashboard_data["summary"]["degrading_components"]
            > dashboard_data["summary"]["improving_components"]
        ):
            dashboard_data["system_alerts"].append(
                {
                    "severity": "warning",
                    "message": f"System shows more degrading components ({dashboard_data['summary']['degrading_components']}) than improving ones ({dashboard_data['summary']['improving_components']})",
                }
            )

        # Convert trending metrics to sorted list
        dashboard_data["trending_metrics"] = dict(
            sorted(
                dashboard_data["trending_metrics"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        return dashboard_data
