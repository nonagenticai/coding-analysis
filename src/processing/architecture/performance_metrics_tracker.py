"""Module for tracking performance metrics over time with trend analysis."""

from __future__ import annotations

import dataclasses
import datetime
import enum
import logging
import statistics
from statistics import StatisticsError
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class TrendType(enum.Enum):
    """Enumeration of possible metric trends."""

    IMPROVING = "updating"  # Metrics are getting better over time
    STABLE = "stable"  # Metrics remain roughly the same
    DEGRADING = "degrading"  # Metrics are getting worse over time
    FLUCTUATING = "fluctuating"  # Metrics show high variability with no clear trend
    UNKNOWN = "unknown"  # Not enough data points to determine a trend


class MetricDataPoint(BaseModel):
    """A single data point in a metric time series."""

    timestamp: datetime.datetime = Field(..., description="When this metric was recorded")
    commit_id: str = Field(..., description="Git commit hash when this metric was measured")
    value: float = Field(..., description="The metric value")
    raw_data: dict[str, Any] | None = Field(
        default=None, description="Raw data used for calculation"
    )

    model_config = ConfigDict(extra="forbid")


class MetricTrendAnalysis(BaseModel):
    """Analysis of a metric's trend over time."""

    metric_name: str = Field(..., description="Name of the analyzed metric")
    component_id: str = Field(..., description="Component this analysis belongs to")
    trend: TrendType = Field(..., description="The detected trend type")
    data_points: int = Field(..., description="Number of data points in the analysis")
    timespan_days: float = Field(..., description="Time period covered by the analysis in days")
    start_value: float = Field(..., description="First value in the time series")
    end_value: float = Field(..., description="Last value in the time series")
    min_value: float = Field(..., description="Minimum value in the time series")
    max_value: float = Field(..., description="Maximum value in the time series")
    percent_change: float = Field(..., description="Percentage change from start to end")
    rate_of_change: float = Field(..., description="Average daily change rate")
    confidence: float = Field(..., description="Confidence score for the trend (0-1)")
    forecast_value: float | None = Field(None, description="Predicted value for next period")
    forecast_confidence: float | None = Field(None, description="Confidence of the forecast (0-1)")
    anomalies: list[datetime.datetime] = Field(
        default_factory=list, description="Timestamps of anomalous values"
    )

    model_config = ConfigDict(extra="forbid")


@dataclasses.dataclass
class ArchitectureMetricTimeSeries:
    """Time series data for a specific metric with analysis capabilities.

    This class stores historical metric values and provides methods for
    trend analysis, forecasting, and visualization preparation.
    """

    metric_name: str
    component_id: str
    data_points: list[MetricDataPoint] = dataclasses.field(default_factory=list)

    def add_data_point(
        self,
        timestamp: datetime.datetime,
        commit_id: str,
        value: float,
        raw_data: dict[str, Any] | None = None,
    ) -> None:
        """Add a new data point to the time series.

        Args:
            timestamp: When the metric was recorded
            commit_id: Git commit hash when this metric was measured
            value: The metric value
            raw_data: Optional raw data used for calculation
        """
        data_point = MetricDataPoint(
            timestamp=timestamp,
            commit_id=commit_id,
            value=value,
            raw_data=raw_data,
        )
        self.data_points.append(data_point)
        # Sort by timestamp to ensure chronological order
        self.data_points.sort(key=lambda dp: dp.timestamp)

    def get_trend_analysis(self) -> MetricTrendAnalysis:
        """Analyze the trend of this metric over time.

        Returns:
            A MetricTrendAnalysis object with trend information
        """
        if len(self.data_points) < 2:
            first_point = self.data_points[0] if self.data_points else None
            return MetricTrendAnalysis(
                metric_name=self.metric_name,
                component_id=self.component_id,
                trend=TrendType.UNKNOWN,
                data_points=len(self.data_points),
                timespan_days=0.0,
                start_value=first_point.value if first_point else 0.0,
                end_value=first_point.value if first_point else 0.0,
                min_value=first_point.value if first_point else 0.0,
                max_value=first_point.value if first_point else 0.0,
                percent_change=0.0,
                rate_of_change=0.0,
                confidence=0.0,
                forecast_value=None,
                forecast_confidence=None,
            )

        # Extract basic statistics
        values = [dp.value for dp in self.data_points]
        timestamps = [dp.timestamp for dp in self.data_points]

        start_value = values[0]
        end_value = values[-1]
        min_value = min(values)
        max_value = max(values)

        # Calculate time span in days
        timespan = timestamps[-1] - timestamps[0]
        timespan_days = timespan.total_seconds() / (60 * 60 * 24)

        # Calculate percent change and rate of change
        if start_value != 0:
            percent_change = ((end_value - start_value) / abs(start_value)) * 100.0
        else:
            percent_change = 0.0 if end_value == 0 else 100.0

        # Daily rate of change
        if timespan_days > 0:
            rate_of_change = (end_value - start_value) / timespan_days
        else:
            rate_of_change = 0.0

        # Detect the trend type using linear regression and variability
        trend, confidence, anomalies = self._detect_trend(values, timestamps)

        # Basic forecasting for next period
        forecast_value, forecast_confidence = self._forecast_next_value(values, timestamps)

        return MetricTrendAnalysis(
            metric_name=self.metric_name,
            component_id=self.component_id,
            trend=trend,
            data_points=len(self.data_points),
            timespan_days=timespan_days,
            start_value=start_value,
            end_value=end_value,
            min_value=min_value,
            max_value=max_value,
            percent_change=percent_change,
            rate_of_change=rate_of_change,
            confidence=confidence,
            forecast_value=forecast_value,
            forecast_confidence=forecast_confidence,
            anomalies=[timestamps[i] for i in anomalies],
        )

    def _detect_trend(
        self, values: list[float], timestamps: list[datetime.datetime]
    ) -> tuple[TrendType, float, list[int]]:
        """Detect the trend type in the time series data.

        Args:
            values: List of metric values
            timestamps: List of corresponding timestamps

        Returns:
            Tuple of (trend_type, confidence, anomaly_indices)
        """
        if len(values) < 3:
            return TrendType.UNKNOWN, 0.0, []

        # Convert timestamps to relative days for numerical analysis
        start_ts = timestamps[0]
        days = [(ts - start_ts).total_seconds() / (60 * 60 * 24) for ts in timestamps]

        # If the time span is too small, consider it stable
        if max(days) - min(days) < 1.0:  # Less than one day
            return TrendType.STABLE, 1.0, []

        try:
            # Calculate linear regression slope (trend direction)
            n = len(values)
            sum_x = sum(days)
            sum_y = sum(values)
            sum_x_squared = sum(x * x for x in days)
            sum_xy = sum(x * y for x, y in zip(days, values))

            # Linear regression slope
            try:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
            except ZeroDivisionError:
                return TrendType.STABLE, 0.5, []

            # Calculate coefficient of variation to detect stability/fluctuation
            mean_value = statistics.mean(values) if values else 0
            # Avoid division by zero
            if mean_value == 0:
                cv = 0.0
            else:
                std_dev = statistics.stdev(values) if len(values) > 1 else 0
                cv = std_dev / abs(mean_value)  # Coefficient of variation

            # Calculate R-squared to determine confidence
            # Calculate mean of y
            y_mean = sum_y / n
            # Calculate total sum of squares
            ss_total = sum((y - y_mean) ** 2 for y in values)
            # Calculate regression sum of squares
            y_pred = [slope * x + (sum_y - slope * sum_x) / n for x in days]
            ss_regression = sum((y_pred_i - y_mean) ** 2 for y_pred_i in y_pred)

            # Calculate R-squared (coefficient of determination)
            if ss_total == 0:
                # Perfect prediction (all values are identical)
                r_squared = 1.0
            else:
                r_squared = ss_regression / ss_total

            # Detect anomalies (points far from the regression line)
            anomalies = []
            for i, (actual, predicted) in enumerate(zip(values, y_pred)):
                # If the value is significantly different from predicted
                if abs(actual - predicted) > 2 * statistics.stdev(values) if len(values) > 1 else 0:
                    anomalies.append(i)

            # Determine trend type based on slope and variability
            # Slope significance threshold (adjusted by time span)
            slope_threshold = 0.01 / max(1.0, max(days) - min(days))

            if cv > 0.25:  # High variability
                if abs(slope) > slope_threshold:
                    # There is a trend despite the variability
                    trend = TrendType.IMPROVING if slope < 0 else TrendType.DEGRADING
                    confidence = max(
                        0.5, min(0.7, r_squared)
                    )  # Limited confidence due to variability
                else:
                    trend = TrendType.FLUCTUATING
                    confidence = min(0.9, max(0.6, 1.0 - r_squared))
            else:  # Low variability
                if abs(slope) > slope_threshold:
                    # Clear trend
                    trend = TrendType.IMPROVING if slope < 0 else TrendType.DEGRADING
                    confidence = max(0.7, min(0.95, r_squared))
                else:
                    # Stable
                    trend = TrendType.STABLE
                    confidence = max(0.8, min(0.95, 1.0 - abs(slope) / slope_threshold))

            return trend, confidence, anomalies

        except (ValueError, ArithmeticError, TypeError, StatisticsError) as e:
            logger.warning(f"Error detecting trend: {e}")
            return TrendType.UNKNOWN, 0.0, []

    def _forecast_next_value(
        self, values: list[float], timestamps: list[datetime.datetime]
    ) -> tuple[float | None, float]:
        """Forecast the next value in the time series.

        Uses basic linear regression for forecasting.

        Args:
            values: List of metric values
            timestamps: List of corresponding timestamps

        Returns:
            Tuple of (forecasted_value, confidence)
        """
        if len(values) < 3:
            return None, 0.0

        try:
            # Convert timestamps to days for numerical analysis
            start_ts = timestamps[0]
            days = [(ts - start_ts).total_seconds() / (60 * 60 * 24) for ts in timestamps]

            # Calculate linear regression
            n = len(values)
            sum_x = sum(days)
            sum_y = sum(values)
            sum_x_squared = sum(x * x for x in days)
            sum_xy = sum(x * y for x, y in zip(days, values))

            # Calculate slope and intercept
            try:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
                intercept = (sum_y - slope * sum_x) / n
            except ZeroDivisionError:
                return None, 0.0

            # Forecast for next period (assuming next measurement would be in the same interval)
            avg_interval = (days[-1] - days[0]) / (n - 1) if n > 1 else 1.0
            next_x = days[-1] + avg_interval
            forecast = slope * next_x + intercept

            # Calculate prediction interval for confidence
            residuals = [y - (slope * x + intercept) for x, y in zip(days, values)]
            residual_std = statistics.stdev(residuals) if len(residuals) > 1 else abs(slope)

            # Basic confidence metric based on residual standard error relative to the mean
            mean_value = statistics.mean(values) if values else forecast
            confidence = 1.0 - min(
                1.0,
                (residual_std / abs(mean_value)) if mean_value != 0 else 0.5,
            )

            return forecast, max(0.1, min(0.95, confidence))

        except (ValueError, ArithmeticError, TypeError, StatisticsError) as e:
            logger.warning(f"Error forecasting next value: {e}")
            return None, 0.0

    def get_visualization_data(self) -> dict[str, Any]:
        """Prepare data for visualization.

        Returns:
            Dictionary with processed data ready for charting/visualization
        """
        if not self.data_points:
            return {
                "metric": self.metric_name,
                "component": self.component_id,
                "timestamps": [],
                "values": [],
                "commits": [],
                "trend": "unknown",
                "forecast": None,
            }

        # Get trend analysis
        trend_analysis = self.get_trend_analysis()

        # Format data for visualization
        result: dict[str, Any] = {
            "metric": self.metric_name,
            "component": self.component_id,
            "timestamps": [dp.timestamp.isoformat() for dp in self.data_points],
            "values": [dp.value for dp in self.data_points],
            "commits": [dp.commit_id for dp in self.data_points],
            "trend": trend_analysis.trend.value,
            "trend_confidence": trend_analysis.confidence,
            "anomalies": [
                i
                for i, ts in enumerate(self.data_points)
                if ts.timestamp in trend_analysis.anomalies
            ],
            "statistics": {
                "min": trend_analysis.min_value,
                "max": trend_analysis.max_value,
                "start": trend_analysis.start_value,
                "end": trend_analysis.end_value,
                "percent_change": trend_analysis.percent_change,
            },
        }

        # Add forecast data if available
        if trend_analysis.forecast_value is not None:
            # Calculate next timestamp based on average interval
            timestamps = [dp.timestamp for dp in self.data_points]
            if len(timestamps) >= 2:
                avg_interval = (timestamps[-1] - timestamps[0]).total_seconds() / (
                    len(timestamps) - 1
                )
                next_timestamp = timestamps[-1] + datetime.timedelta(seconds=avg_interval)

                result["forecast"] = {
                    "timestamp": next_timestamp.isoformat(),
                    "value": trend_analysis.forecast_value,
                    "confidence": trend_analysis.forecast_confidence,
                }

        return result


# Type alias for compatibility
MetricTimeSeries = ArchitectureMetricTimeSeries
