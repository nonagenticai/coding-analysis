"""Time series metrics and cross-language correlation analysis for architecture processing."""

from __future__ import annotations

import json
import logging
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureMetricTimeSeries:
    """Time series data for a specific metric."""

    metric_name: str
    timestamps: list[datetime] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_measurement(
        self,
        timestamp: datetime,
        value: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a new measurement to the time series."""
        self.timestamps.append(timestamp)
        self.values.append(value)
        if metadata:
            self.metadata[timestamp.isoformat()] = metadata

    def get_trend(self, window_size: int = 5) -> str:
        """Calculate trend over the specified window size."""
        if len(self.values) < window_size:
            return "insufficient_data"

        recent_values = self.values[-window_size:]
        older_values = (
            self.values[-(window_size * 2) : -window_size]
            if len(self.values) >= window_size * 2
            else []
        )

        if not older_values:
            return "insufficient_data"

        recent_avg = statistics.mean(recent_values)
        older_avg = statistics.mean(older_values)

        if recent_avg > older_avg * 1.1:
            return "increasing"
        if recent_avg < older_avg * 0.9:
            return "decreasing"
        return "stable"

    def get_statistics(self) -> dict[str, float]:
        """Get statistical summary of the time series."""
        if not self.values:
            return {}

        return {
            "mean": statistics.mean(self.values),
            "median": statistics.median(self.values),
            "std_dev": (statistics.stdev(self.values) if len(self.values) > 1 else 0),
            "min": min(self.values),
            "max": max(self.values),
            "range": max(self.values) - min(self.values),
            "latest": self.values[-1],
            "count": len(self.values),
        }


@dataclass
class CrossLanguageCorrelation:
    """Cross-language correlation data."""

    language_pair: tuple[str, str]
    correlation_coefficient: float
    shared_patterns: list[str] = field(default_factory=list)
    dependency_overlap: float = 0.0
    architectural_similarity: float = 0.0
    quality_correlation: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "language_pair": self.language_pair,
            "correlation_coefficient": self.correlation_coefficient,
            "shared_patterns": self.shared_patterns,
            "dependency_overlap": self.dependency_overlap,
            "architectural_similarity": self.architectural_similarity,
            "quality_correlation": self.quality_correlation,
        }


class TimeSeriesMetricsCalculator:
    """Calculator for time-series metrics and historical analysis."""

    def __init__(self, db_path: Path | None = None):
        """Initialize the time series metrics calculator.

        Args:
            db_path: Path to SQLite database for storing metrics history
        """
        self.db_path = db_path or Path("metrics_history.db")
        self.scaler = StandardScaler()
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize SQLite database for metrics storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Create tables for metrics history
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metrics_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        codebase_id TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        value REAL NOT NULL,
                        commit_hash TEXT,
                        metadata TEXT,
                        UNIQUE(codebase_id, metric_name, timestamp)
                    )
                """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS language_correlations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        codebase_id TEXT NOT NULL,
                        language_1 TEXT NOT NULL,
                        language_2 TEXT NOT NULL,
                        correlation_coefficient REAL NOT NULL,
                        shared_patterns TEXT,
                        dependency_overlap REAL,
                        architectural_similarity REAL,
                        quality_correlation REAL,
                        timestamp DATETIME NOT NULL,
                        UNIQUE(codebase_id, language_1, language_2, timestamp)
                    )
                """
                )

                # Create indexes for performance
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_metrics_codebase_time
                    ON metrics_history(codebase_id, timestamp DESC)
                """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_correlations_codebase_time
                    ON language_correlations(codebase_id, timestamp DESC)
                """
                )

                conn.commit()
                logger.info("Metrics database initialized successfully")

        except sqlite3.Error as e:
            logger.error("Failed to initialize metrics database: %s", e)
            raise

    def store_metrics(
        self,
        codebase_id: str,
        metrics: dict[str, float],
        commit_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store metrics in the time series database.

        Args:
            codebase_id: Identifier for the codebase
            metrics: Dictionary of metric names to values
            commit_hash: Git commit hash for this measurement
            metadata: Additional metadata for the measurement
        """
        timestamp = datetime.now(UTC)
        metadata_json = json.dumps(metadata) if metadata else None

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for metric_name, value in metrics.items():
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO metrics_history
                        (codebase_id, metric_name, timestamp, value, commit_hash, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            codebase_id,
                            metric_name,
                            timestamp.isoformat(),
                            value,
                            commit_hash,
                            metadata_json,
                        ),
                    )

                conn.commit()
                logger.info("Stored %d metrics for codebase %s", len(metrics), codebase_id)

        except sqlite3.Error as e:
            logger.error("Failed to store metrics: %s", e)
            raise

    def get_metrics_time_series(
        self, codebase_id: str, metric_name: str, days_back: int = 30
    ) -> ArchitectureMetricTimeSeries:
        """Retrieve time series data for a specific metric.

        Args:
            codebase_id: Identifier for the codebase
            metric_name: Name of the metric to retrieve
            days_back: Number of days to look back

        Returns:
            MetricTimeSeries object with historical data
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=days_back)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT timestamp, value, metadata
                    FROM metrics_history
                    WHERE codebase_id = ? AND metric_name = ? AND timestamp >= ?
                    ORDER BY timestamp ASC
                """,
                    (codebase_id, metric_name, cutoff_date.isoformat()),
                )

                rows = cursor.fetchall()

                time_series = ArchitectureMetricTimeSeries(metric_name=metric_name)

                for row in rows:
                    timestamp = datetime.fromisoformat(row[0])
                    value = row[1]
                    metadata = json.loads(row[2]) if row[2] else {}

                    time_series.add_measurement(timestamp, value, metadata)

                logger.info(
                    "Retrieved %d measurements for %s",
                    len(time_series.values),
                    metric_name,
                )
                return time_series

        except sqlite3.Error as e:
            logger.error("Failed to retrieve time series data: %s", e)
            return ArchitectureMetricTimeSeries(metric_name=metric_name)

    def compare_metrics_across_commits(
        self, codebase_id: str, commit_hashes: list[str]
    ) -> dict[str, list[float]]:
        """Compare metrics across different commits.

        Args:
            codebase_id: Identifier for the codebase
            commit_hashes: List of commit hashes to compare

        Returns:
            Dictionary mapping metric names to lists of values across commits
        """
        metrics_comparison: dict[str, list[float]] = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get all metrics for the specified commits
                placeholders = ",".join(["?"] * len(commit_hashes))
                query = f"""
                    SELECT metric_name, commit_hash, value
                    FROM metrics_history
                    WHERE codebase_id = ? AND commit_hash IN ({placeholders})
                    ORDER BY metric_name, timestamp ASC
                """  # nosec B608 -- placeholders are ? marks, not user input
                cursor.execute(query, [codebase_id] + commit_hashes)

                rows = cursor.fetchall()

                # Group by metric name
                for metric_name, _, value in rows:
                    if metric_name not in metrics_comparison:
                        metrics_comparison[metric_name] = []
                    metrics_comparison[metric_name].append(value)

                logger.info(
                    "Compared %d metrics across %d commits",
                    len(metrics_comparison),
                    len(commit_hashes),
                )
                return metrics_comparison

        except sqlite3.Error as e:
            logger.error("Failed to compare metrics across commits: %s", e)
            return {}

    def calculate_metric_trends(
        self, codebase_id: str, days_back: int = 30
    ) -> dict[str, dict[str, Any]]:
        """Calculate trends for all metrics in a codebase.

        Args:
            codebase_id: Identifier for the codebase
            days_back: Number of days to analyze

        Returns:
            Dictionary mapping metric names to trend analysis
        """
        trends = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get all metric names for the codebase
                cursor.execute(
                    """
                    SELECT DISTINCT metric_name
                    FROM metrics_history
                    WHERE codebase_id = ?
                """,
                    (codebase_id,),
                )

                metric_names = [row[0] for row in cursor.fetchall()]

                # Calculate trends for each metric
                for metric_name in metric_names:
                    time_series = self.get_metrics_time_series(codebase_id, metric_name, days_back)

                    if time_series.values:
                        trends[metric_name] = {
                            "trend": time_series.get_trend(),
                            "statistics": time_series.get_statistics(),
                            "data_points": len(time_series.values),
                        }

                logger.info("Calculated trends for %d metrics", len(trends))
                return trends

        except sqlite3.Error as e:
            logger.error("Failed to calculate metric trends: %s", e)
            return {}


class CrossLanguageCorrelationAnalyzer:
    """Analyzer for cross-language correlations in polyglot codebases."""

    def __init__(self, db_path: Path | None = None):
        """Initialize the cross-language correlation analyzer.

        Args:
            db_path: Path to SQLite database for storing correlation data
        """
        self.db_path = db_path or Path("metrics_history.db")
        self.scaler = StandardScaler()

    def analyze_cross_language_correlation(
        self,
        codebase_id: str,
        language_metrics: dict[str, dict[str, float]],
        language_patterns: dict[str, list[str]],
        language_dependencies: dict[str, list[str]],
    ) -> list[CrossLanguageCorrelation]:
        """Analyze correlations between different programming languages.

        Args:
            codebase_id: Identifier for the codebase
            language_metrics: Dictionary mapping languages to their metrics
            language_patterns: Dictionary mapping languages to detected patterns
            language_dependencies: Dictionary mapping languages to their dependencies

        Returns:
            List of cross-language correlations
        """
        correlations = []
        languages = list(language_metrics.keys())

        # Compare each pair of languages
        for i, lang1 in enumerate(languages):
            for j in range(i + 1, len(languages)):
                lang2 = languages[j]

                language_data = {
                    "metrics": language_metrics,
                    "patterns": language_patterns,
                    "dependencies": language_dependencies,
                }
                correlation = self._calculate_language_correlation(
                    lang1,
                    lang2,
                    language_data,
                )

                if correlation:
                    correlations.append(correlation)

        # Store correlations in database
        self._store_correlations(codebase_id, correlations)

        logger.info("Analyzed %d cross-language correlations", len(correlations))
        return correlations

    def _calculate_language_correlation(
        self,
        lang1: str,
        lang2: str,
        language_data: dict[str, Any],
    ) -> CrossLanguageCorrelation | None:
        """Calculate correlation between two specific languages."""
        try:
            # Extract metrics for both languages
            metrics_result = self._extract_language_metrics(lang1, lang2, language_data)
            if not metrics_result:
                return None

            metrics1, metrics2 = metrics_result
            correlation_coeff = self._calculate_metrics_correlation(metrics1, metrics2)

            # Calculate pattern and dependency similarities
            patterns_result = self._calculate_pattern_similarity(lang1, lang2, language_data)
            dependency_overlap = self._calculate_dependency_overlap(lang1, lang2, language_data)

            return CrossLanguageCorrelation(
                language_pair=(lang1, lang2),
                correlation_coefficient=correlation_coeff,
                shared_patterns=patterns_result["shared_patterns"],
                dependency_overlap=dependency_overlap,
                architectural_similarity=patterns_result["architectural_similarity"],
                quality_correlation=0.0,  # Placeholder - would need quality metrics
            )

        except ValueError as e:
            logger.error("Failed to calculate correlation between %s and %s: %s", lang1, lang2, e)
            return None

    def _extract_language_metrics(
        self, lang1: str, lang2: str, language_data: dict[str, Any]
    ) -> tuple[dict[str, float], dict[str, float]] | None:
        """Extract metrics for two languages."""
        language_metrics = language_data.get("metrics", {})
        metrics1 = language_metrics.get(lang1, {})
        metrics2 = language_metrics.get(lang2, {})

        if not metrics1 or not metrics2:
            return None
        return metrics1, metrics2

    def _calculate_pattern_similarity(
        self, lang1: str, lang2: str, language_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate pattern similarity between languages."""
        language_patterns = language_data.get("patterns", {})
        patterns1 = set(language_patterns.get(lang1, []))
        patterns2 = set(language_patterns.get(lang2, []))

        shared_patterns = list(patterns1.intersection(patterns2))
        architectural_similarity = self._calculate_architectural_similarity(patterns1, patterns2)

        return {
            "shared_patterns": shared_patterns,
            "architectural_similarity": architectural_similarity,
        }

    def _calculate_dependency_overlap(
        self, lang1: str, lang2: str, language_data: dict[str, Any]
    ) -> float:
        """Calculate dependency overlap between languages."""
        language_dependencies = language_data.get("dependencies", {})
        deps1 = set(language_dependencies.get(lang1, []))
        deps2 = set(language_dependencies.get(lang2, []))

        return len(deps1.intersection(deps2)) / max(len(deps1.union(deps2)), 1)

    def _calculate_metrics_correlation(
        self, metrics1: dict[str, float], metrics2: dict[str, float]
    ) -> float:
        """Calculate correlation coefficient between two sets of metrics."""
        try:
            # Find common metrics
            common_metrics = set(metrics1.keys()).intersection(set(metrics2.keys()))

            if len(common_metrics) < 2:
                return 0.0

            # Extract values for common metrics
            values1 = [metrics1[metric] for metric in common_metrics]
            values2 = [metrics2[metric] for metric in common_metrics]

            # Calculate correlation using numpy
            correlation_matrix = np.corrcoef(values1, values2)
            correlation_coeff = correlation_matrix[0, 1]

            # Handle NaN values
            if np.isnan(correlation_coeff):
                return 0.0

            return float(correlation_coeff)

        except (ValueError, TypeError) as e:
            logger.error("Failed to calculate metrics correlation: %s", e)
            return 0.0

    def _calculate_architectural_similarity(self, patterns1: set, patterns2: set) -> float:
        """Calculate architectural similarity based on shared patterns."""
        if not patterns1 and not patterns2:
            return 1.0

        if not patterns1 or not patterns2:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(patterns1.intersection(patterns2))
        union = len(patterns1.union(patterns2))

        return intersection / max(union, 1)

    def _store_correlations(
        self, codebase_id: str, correlations: list[CrossLanguageCorrelation]
    ) -> None:
        """Store correlations in the database."""
        timestamp = datetime.now(UTC)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for correlation in correlations:
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO language_correlations
                        (codebase_id, language_1, language_2, correlation_coefficient,
                         shared_patterns, dependency_overlap, architectural_similarity,
                         quality_correlation, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            codebase_id,
                            correlation.language_pair[0],
                            correlation.language_pair[1],
                            correlation.correlation_coefficient,
                            json.dumps(correlation.shared_patterns),
                            correlation.dependency_overlap,
                            correlation.architectural_similarity,
                            correlation.quality_correlation,
                            timestamp.isoformat(),
                        ),
                    )

                conn.commit()
                logger.info("Stored %d cross-language correlations", len(correlations))

        except sqlite3.Error as e:
            logger.error("Failed to store correlations: %s", e)
            raise

    def get_language_correlations(
        self, codebase_id: str, days_back: int = 30
    ) -> list[CrossLanguageCorrelation]:
        """Retrieve stored cross-language correlations.

        Args:
            codebase_id: Identifier for the codebase
            days_back: Number of days to look back

        Returns:
            List of cross-language correlations
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=days_back)
        correlations = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT language_1, language_2, correlation_coefficient,
                           shared_patterns, dependency_overlap, architectural_similarity,
                           quality_correlation
                    FROM language_correlations
                    WHERE codebase_id = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """,
                    (codebase_id, cutoff_date.isoformat()),
                )

                rows = cursor.fetchall()

                for row in rows:
                    correlation = CrossLanguageCorrelation(
                        language_pair=(row[0], row[1]),
                        correlation_coefficient=row[2],
                        shared_patterns=json.loads(row[3]) if row[3] else [],
                        dependency_overlap=row[4],
                        architectural_similarity=row[5],
                        quality_correlation=row[6],
                    )
                    correlations.append(correlation)

                logger.info("Retrieved %d cross-language correlations", len(correlations))
                return correlations

        except sqlite3.Error as e:
            logger.error("Failed to retrieve correlations: %s", e)
            return []

    def generate_correlation_report(self, codebase_id: str) -> dict[str, Any]:
        """Generate a comprehensive correlation report.

        Args:
            codebase_id: Identifier for the codebase

        Returns:
            Dictionary containing correlation analysis report
        """
        correlations = self.get_language_correlations(codebase_id)

        if not correlations:
            return {"error": "No correlation data available"}

        # Calculate summary statistics
        correlation_values = [c.correlation_coefficient for c in correlations]
        dependency_overlaps = [c.dependency_overlap for c in correlations]
        architectural_similarities = [c.architectural_similarity for c in correlations]

        # Find strongest correlations
        strongest_correlation = max(correlations, key=lambda x: abs(x.correlation_coefficient))
        weakest_correlation = min(correlations, key=lambda x: abs(x.correlation_coefficient))

        # Identify common patterns across languages
        all_patterns = set()
        for correlation in correlations:
            all_patterns.update(correlation.shared_patterns)

        report = {
            "summary": {
                "total_language_pairs": len(correlations),
                "average_correlation": statistics.mean(correlation_values),
                "average_dependency_overlap": statistics.mean(dependency_overlaps),
                "average_architectural_similarity": statistics.mean(architectural_similarities),
            },
            "strongest_correlation": strongest_correlation.to_dict(),
            "weakest_correlation": weakest_correlation.to_dict(),
            "common_patterns": list(all_patterns),
            "correlations": [c.to_dict() for c in correlations],
        }

        logger.info("Generated correlation report for %d language pairs", len(correlations))
        return report
