"""
Report Chart Generator - Generate chart and visualization data for quality reports.
Provides chart generation, data visualization configurations, and chart rendering utilities.
"""

from __future__ import annotations

import logging
from typing import Any

from models.testing_models import QualityResult

from .report_data_processor import ReportDataProcessor

logger = logging.getLogger(__name__)


class ReportChartGenerator:
    """
    Generates chart and visualization data for quality reports.
    Handles chart configuration, data preparation, and visualization settings.
    """

    def __init__(self, data_processor: ReportDataProcessor | None = None):
        """
        Initialize the chart generator.

        Args:
            data_processor: Optional data processor instance
        """
        self.data_processor = data_processor or ReportDataProcessor()
        logger.info("ReportChartGenerator initialized")

    async def generate_severity_chart(self, results: list[QualityResult]) -> dict[str, Any]:
        """
        Generate severity distribution chart data.

        Args:
            results: List of QualityResult objects

        Returns:
            Chart configuration dictionary
        """
        severity_counts = {"critical": 0, "error": 0, "warning": 0, "info": 0}

        for result in results:
            for issue in result.issues:
                severity = issue.severity.lower()
                if severity in severity_counts:
                    severity_counts[severity] += 1

        return {
            "type": "pie",
            "title": "Issue Distribution by Severity",
            "data": {
                "labels": list(severity_counts.keys()),
                "values": list(severity_counts.values()),
                "colors": ["#e74c3c", "#e67e22", "#f39c12", "#3498db"],
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "legend": {
                    "position": "bottom",
                },
            },
        }

    async def generate_score_distribution_chart(
        self, results: list[QualityResult]
    ) -> dict[str, Any]:
        """
        Generate quality score distribution chart.

        Args:
            results: List of QualityResult objects

        Returns:
            Chart configuration dictionary
        """
        scores = [r.overall_score for r in results]

        # Create score ranges
        ranges = {
            "90-100": 0,
            "80-89": 0,
            "70-79": 0,
            "60-69": 0,
            "Below 60": 0,
        }

        for score in scores:
            if score >= 90:
                ranges["90-100"] += 1
            elif score >= 80:
                ranges["80-89"] += 1
            elif score >= 70:
                ranges["70-79"] += 1
            elif score >= 60:
                ranges["60-69"] += 1
            else:
                ranges["Below 60"] += 1

        return {
            "type": "bar",
            "title": "Quality Score Distribution",
            "data": {
                "labels": list(ranges.keys()),
                "datasets": [
                    {
                        "label": "Number of Files",
                        "data": list(ranges.values()),
                        "backgroundColor": [
                            "#27ae60",
                            "#2ecc71",
                            "#f39c12",
                            "#e67e22",
                            "#e74c3c",
                        ],
                    }
                ],
            },
            "options": {
                "responsive": True,
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "Number of Files",
                        },
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Quality Score Range",
                        },
                    },
                },
            },
        }

    async def generate_tools_performance_chart(
        self, results: list[QualityResult]
    ) -> dict[str, Any]:
        """
        Generate tools performance chart.

        Args:
            results: List of QualityResult objects

        Returns:
            Chart configuration dictionary
        """
        tool_performance = await self.data_processor.generate_tool_performance(results)

        tools = list(tool_performance.keys())
        success_rates = [tool_performance[tool]["success_rate"] for tool in tools]

        return {
            "type": "bar",
            "title": "Tool Success Rates",
            "data": {
                "labels": tools,
                "datasets": [
                    {
                        "label": "Success Rate (%)",
                        "data": success_rates,
                        "backgroundColor": "#3498db",
                        "borderColor": "#2980b9",
                        "borderWidth": 1,
                    }
                ],
            },
            "options": {
                "responsive": True,
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 100,
                        "title": {
                            "display": True,
                            "text": "Success Rate (%)",
                        },
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Tools",
                        },
                    },
                },
            },
        }

    async def generate_language_breakdown_chart(
        self, results: list[QualityResult]
    ) -> dict[str, Any]:
        """
        Generate language breakdown chart.

        Args:
            results: List of QualityResult objects

        Returns:
            Chart configuration dictionary
        """
        language_counts: dict[str, int] = {}
        for result in results:
            lang = result.language
            language_counts[lang] = language_counts.get(lang, 0) + 1

        return {
            "type": "pie",
            "title": "Analysis by Language",
            "data": {
                "labels": list(language_counts.keys()),
                "datasets": [
                    {
                        "data": list(language_counts.values()),
                        "backgroundColor": [
                            "#3498db",
                            "#e74c3c",
                            "#2ecc71",
                            "#f39c12",
                            "#9b59b6",
                            "#1abc9c",
                        ],
                    }
                ],
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {
                        "position": "bottom",
                    },
                },
            },
        }

    async def generate_trends_chart(
        self,
        results: list[QualityResult],
        historical_data: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate trends chart data.

        Args:
            results: List of QualityResult objects
            historical_data: Optional historical analysis data for trends

        Returns:
            Chart configuration dictionary
        """
        # Use historical data if provided, otherwise use current results
        if historical_data:
            # Historical trend analysis
            timestamps = [entry.get("timestamp", "") for entry in historical_data]
            scores = [entry.get("overall_score", 0) for entry in historical_data]
        else:
            # Basic trend based on creation time
            sorted_results = sorted(results, key=lambda r: r.created_at)
            timestamps = [r.created_at.isoformat() for r in sorted_results]
            scores = [r.overall_score for r in sorted_results]

        return {
            "type": "line",
            "title": "Quality Score Trend",
            "data": {
                "labels": timestamps,
                "datasets": [
                    {
                        "label": "Quality Score",
                        "data": scores,
                        "borderColor": "#3498db",
                        "backgroundColor": "rgba(52, 152, 219, 0.1)",
                        "fill": True,
                        "tension": 0.4,
                    }
                ],
            },
            "options": {
                "responsive": True,
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "max": 100,
                        "title": {
                            "display": True,
                            "text": "Quality Score",
                        },
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Time",
                        },
                    },
                },
                "plugins": {
                    "legend": {
                        "display": False,
                    },
                },
            },
        }

    async def generate_issue_trends_chart(self, results: list[QualityResult]) -> dict[str, Any]:
        """
        Generate issue count trends chart.

        Args:
            results: List of QualityResult objects

        Returns:
            Chart configuration dictionary
        """
        sorted_results = sorted(results, key=lambda r: r.created_at)

        timestamps = [r.created_at.isoformat() for r in sorted_results]
        critical_counts = []
        error_counts = []
        warning_counts = []

        for result in sorted_results:
            critical = len([i for i in result.issues if i.severity.lower() == "critical"])
            error = len([i for i in result.issues if i.severity.lower() == "error"])
            warning = len([i for i in result.issues if i.severity.lower() == "warning"])

            critical_counts.append(critical)
            error_counts.append(error)
            warning_counts.append(warning)

        return {
            "type": "line",
            "title": "Issue Count Trends by Severity",
            "data": {
                "labels": timestamps,
                "datasets": [
                    {
                        "label": "Critical",
                        "data": critical_counts,
                        "borderColor": "#e74c3c",
                        "backgroundColor": "rgba(231, 76, 60, 0.1)",
                        "fill": False,
                    },
                    {
                        "label": "Error",
                        "data": error_counts,
                        "borderColor": "#e67e22",
                        "backgroundColor": "rgba(230, 126, 34, 0.1)",
                        "fill": False,
                    },
                    {
                        "label": "Warning",
                        "data": warning_counts,
                        "borderColor": "#f39c12",
                        "backgroundColor": "rgba(243, 156, 18, 0.1)",
                        "fill": False,
                    },
                ],
            },
            "options": {
                "responsive": True,
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "Issue Count",
                        },
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Time",
                        },
                    },
                },
                "plugins": {
                    "legend": {
                        "position": "top",
                    },
                },
            },
        }

    async def generate_execution_time_chart(self, results: list[QualityResult]) -> dict[str, Any]:
        """
        Generate execution time chart by tool.

        Args:
            results: List of QualityResult objects

        Returns:
            Chart configuration dictionary
        """
        tool_times: dict[str, list[float]] = {}

        for result in results:
            for tool_name, tool_result in result.tools_results.items():
                execution_time = tool_result.get("execution_time", 0.0)
                if tool_name not in tool_times:
                    tool_times[tool_name] = []
                tool_times[tool_name].append(execution_time)

        # Calculate average execution times
        tools = list(tool_times.keys())
        avg_times = [sum(tool_times[tool]) / len(tool_times[tool]) for tool in tools]

        return {
            "type": "bar",
            "title": "Average Execution Time by Tool",
            "data": {
                "labels": tools,
                "datasets": [
                    {
                        "label": "Average Time (seconds)",
                        "data": avg_times,
                        "backgroundColor": "#9b59b6",
                        "borderColor": "#8e44ad",
                        "borderWidth": 1,
                    }
                ],
            },
            "options": {
                "responsive": True,
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {
                            "display": True,
                            "text": "Time (seconds)",
                        },
                    },
                    "x": {
                        "title": {
                            "display": True,
                            "text": "Tools",
                        },
                    },
                },
            },
        }

    async def generate_dashboard_charts(
        self,
        results: list[QualityResult],
        historical_data: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate all charts for dashboard display.

        Args:
            results: List of QualityResult objects
            historical_data: Optional historical data for trends

        Returns:
            Dictionary containing all chart configurations
        """
        charts = {
            "severity_distribution": await self.generate_severity_chart(results),
            "quality_score_distribution": await self.generate_score_distribution_chart(results),
            "tools_performance": await self.generate_tools_performance_chart(results),
            "language_breakdown": await self.generate_language_breakdown_chart(results),
            "quality_trends": await self.generate_trends_chart(results, historical_data),
            "issue_trends": await self.generate_issue_trends_chart(results),
            "execution_time": await self.generate_execution_time_chart(results),
        }

        return charts

    def get_available_chart_types(self) -> list[str]:
        """
        Get list of available chart types.

        Returns:
            List of chart type names
        """
        return [
            "severity_distribution",
            "quality_score_distribution",
            "tools_performance",
            "language_breakdown",
            "quality_trends",
            "issue_trends",
            "execution_time",
        ]

    def get_chart_color_scheme(self, chart_type: str) -> list[str]:
        """
        Get color scheme for specific chart type.

        Args:
            chart_type: Type of chart

        Returns:
            List of color codes
        """
        color_schemes = {
            "severity": ["#e74c3c", "#e67e22", "#f39c12", "#3498db"],
            "quality": ["#27ae60", "#2ecc71", "#f39c12", "#e67e22", "#e74c3c"],
            "default": [
                "#3498db",
                "#e74c3c",
                "#2ecc71",
                "#f39c12",
                "#9b59b6",
                "#1abc9c",
            ],
        }

        if chart_type in ["severity_distribution"]:
            return color_schemes["severity"]
        elif chart_type in ["quality_score_distribution"]:
            return color_schemes["quality"]
        else:
            return color_schemes["default"]


def create_chart_generator(
    data_processor: ReportDataProcessor | None = None,
) -> ReportChartGenerator:
    """Factory function to create ReportChartGenerator."""
    return ReportChartGenerator(data_processor)
