"""
Report Configuration - Configuration and styling management for quality reports.
Provides settings, themes, styling configurations, and report customization options.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class QualityMetrics(BaseModel):
    """Aggregated quality metrics for reporting."""

    total_files_analyzed: int = Field(ge=0, description="Total files analyzed")
    total_issues: int = Field(ge=0, description="Total issues found")
    overall_score: float = Field(ge=0.0, le=100.0, description="Overall quality score")

    # Issues by severity
    critical_issues: int = Field(ge=0, description="Number of critical issues")
    error_issues: int = Field(ge=0, description="Number of error issues")
    warning_issues: int = Field(ge=0, description="Number of warning issues")
    info_issues: int = Field(ge=0, description="Number of info issues")

    # Tools statistics
    tools_used: list[str] = Field(default=[], description="List of tools used")
    successful_tools: int = Field(ge=0, description="Number of successful tool runs")
    failed_tools: int = Field(ge=0, description="Number of failed tool runs")

    # Performance metrics
    total_execution_time: float = Field(ge=0.0, description="Total execution time")
    average_execution_time: float = Field(ge=0.0, description="Average execution time per tool")

    # Language distribution
    languages_analyzed: dict[str, int] = Field(default={}, description="Files by language")


class ReportSettings(BaseModel):
    """Settings for report generation."""

    include_fix_suggestions: bool = Field(default=True, description="Include fix suggestions")
    max_issues_per_file: int | None = Field(default=100, description="Max issues to show per file")
    severity_filter: list[str] = Field(
        default=["critical", "error", "warning", "info"],
        description="Severity levels to include",
    )

    @field_validator("severity_filter")
    @classmethod
    def validate_severity_filter(cls, v: list[str]) -> list[str]:  # noqa: N805
        """Validate severity levels."""
        valid_severities = {"critical", "error", "warning", "info"}
        for severity in v:
            if severity.lower() not in valid_severities:
                raise ValueError(f"Invalid severity level: {severity}")
        return [s.lower() for s in v]


class ReportTheme(BaseModel):
    """Theme configuration for reports."""

    name: str = Field(description="Theme name")
    primary_color: str = Field(default="#3498db", description="Primary color")
    secondary_color: str = Field(default="#2c3e50", description="Secondary color")
    success_color: str = Field(default="#27ae60", description="Success color")
    warning_color: str = Field(default="#f39c12", description="Warning color")
    danger_color: str = Field(default="#e74c3c", description="Danger color")
    info_color: str = Field(default="#3498db", description="Info color")

    background_color: str = Field(default="#f8f9fa", description="Background color")
    text_color: str = Field(default="#2c3e50", description="Text color")
    border_color: str = Field(default="#bdc3c7", description="Border color")

    font_family: str = Field(
        default="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif",
        description="Font family",
    )
    font_size_base: str = Field(default="14px", description="Base font size")

    @field_validator(
        "primary_color",
        "secondary_color",
        "success_color",
        "warning_color",
        "danger_color",
        "info_color",
        "background_color",
        "text_color",
        "border_color",
    )
    @classmethod
    def validate_color(cls, v: str) -> str:  # noqa: N805
        """Validate color format."""
        if not v.startswith("#") or len(v) != 7:
            raise ValueError("Color must be in hex format (#RRGGBB)")
        return v


class StyleConfiguration(BaseModel):
    """Style configuration for report components."""

    # Layout settings
    container_max_width: str = Field(default="1200px", description="Container max width")
    container_padding: str = Field(default="30px", description="Container padding")
    border_radius: str = Field(default="8px", description="Border radius")
    box_shadow: str = Field(default="0 2px 10px rgba(0,0,0,0.1)", description="Box shadow")

    # Grid settings
    metrics_grid_columns: str = Field(
        default="repeat(auto-fit, minmax(200px, 1fr))",
        description="Metrics grid columns",
    )
    grid_gap: str = Field(default="20px", description="Grid gap")

    # Card settings
    card_padding: str = Field(default="20px", description="Card padding")
    card_border_width: str = Field(default="4px", description="Card border width")

    # Typography
    heading_font_weight: str = Field(default="bold", description="Heading font weight")
    metric_value_font_size: str = Field(default="2em", description="Metric value font size")

    # Issue styling
    issue_item_padding: str = Field(default="15px", description="Issue item padding")
    issue_border_width: str = Field(default="3px", description="Issue border width")

    # Table styling
    table_padding: str = Field(default="12px", description="Table cell padding")
    table_border: str = Field(default="1px solid #ddd", description="Table border")


@dataclass
class ReportContext:
    """Context data for report generation."""

    metrics: QualityMetrics
    settings: ReportSettings
    theme: ReportTheme | None = None
    style: StyleConfiguration | None = None
    generation_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    report_id: str = field(default_factory=lambda: f"report_{int(datetime.now(UTC).timestamp())}")
    title: str = "Quality Analysis Report"
    description: str = "Comprehensive code quality analysis results"


class ReportConfigurationManager:
    """
    Manages report configuration, themes, and styling.
    Provides centralized configuration management for quality reports.
    """

    def __init__(self):
        """Initialize the configuration manager."""
        self.themes: dict[str, ReportTheme] = {}
        self.styles: dict[str, StyleConfiguration] = {}
        self._load_default_themes()
        self._load_default_styles()
        logger.info("ReportConfigurationManager initialized")

    def _load_default_themes(self) -> None:
        """Load default themes."""
        # Default theme
        self.themes["default"] = ReportTheme(
            name="default",
            primary_color="#3498db",
            secondary_color="#2c3e50",
            success_color="#27ae60",
            warning_color="#f39c12",
            danger_color="#e74c3c",
            info_color="#3498db",
        )

        # Dark theme
        self.themes["dark"] = ReportTheme(
            name="dark",
            primary_color="#3498db",
            secondary_color="#ecf0f1",
            success_color="#2ecc71",
            warning_color="#f1c40f",
            danger_color="#e74c3c",
            info_color="#3498db",
            background_color="#2c3e50",
            text_color="#ecf0f1",
            border_color="#34495e",
        )

        # Corporate theme
        self.themes["corporate"] = ReportTheme(
            name="corporate",
            primary_color="#1f4e79",
            secondary_color="#2c3e50",
            success_color="#28a745",
            warning_color="#ffc107",
            danger_color="#dc3545",
            info_color="#17a2b8",
            background_color="#ffffff",
            text_color="#343a40",
            border_color="#dee2e6",
        )

    def _load_default_styles(self) -> None:
        """Load default style configurations."""
        # Compact style
        self.styles["compact"] = StyleConfiguration(
            container_max_width="900px",
            container_padding="20px",
            metrics_grid_columns="repeat(auto-fit, minmax(150px, 1fr))",
            card_padding="15px",
            metric_value_font_size="1.5em",
        )

        # Detailed style
        self.styles["detailed"] = StyleConfiguration(
            container_max_width="1400px",
            container_padding="40px",
            metrics_grid_columns="repeat(auto-fit, minmax(250px, 1fr))",
            card_padding="25px",
            metric_value_font_size="2.5em",
        )

        # Minimal style
        self.styles["minimal"] = StyleConfiguration(
            container_padding="15px",
            border_radius="4px",
            box_shadow="0 1px 3px rgba(0,0,0,0.1)",
            card_padding="10px",
            grid_gap="10px",
        )

    def get_theme(self, theme_name: str) -> ReportTheme | None:
        """
        Get theme by name.

        Args:
            theme_name: Name of the theme

        Returns:
            ReportTheme object or None if not found
        """
        return self.themes.get(theme_name)

    def get_style(self, style_name: str) -> StyleConfiguration | None:
        """
        Get style configuration by name.

        Args:
            style_name: Name of the style

        Returns:
            StyleConfiguration object or None if not found
        """
        return self.styles.get(style_name)

    def register_theme(self, theme: ReportTheme) -> None:
        """
        Register a new theme.

        Args:
            theme: ReportTheme object to register
        """
        self.themes[theme.name] = theme
        logger.info("Registered theme: %s", theme.name)

    def register_style(self, name: str, style: StyleConfiguration) -> None:
        """
        Register a new style configuration.

        Args:
            name: Style name
            style: StyleConfiguration object to register
        """
        self.styles[name] = style
        logger.info("Registered style: %s", name)

    def get_available_themes(self) -> list[str]:
        """Get list of available theme names."""
        return list(self.themes.keys())

    def get_available_styles(self) -> list[str]:
        """Get list of available style names."""
        return list(self.styles.keys())

    def create_context(
        self,
        metrics: QualityMetrics,
        settings: ReportSettings | None = None,
        theme_name: str = "default",
        style_name: str = "default",
        title: str = "Quality Analysis Report",
        description: str = "Comprehensive code quality analysis results",
    ) -> ReportContext:
        """
        Create a report context with specified configuration.

        Args:
            metrics: Quality metrics
            settings: Report settings
            theme_name: Theme name to use
            style_name: Style name to use
            title: Report title
            description: Report description

        Returns:
            ReportContext object
        """
        theme = self.get_theme(theme_name) or self.themes["default"]
        style = self.get_style(style_name) or StyleConfiguration()
        settings = settings or ReportSettings()

        return ReportContext(
            metrics=metrics,
            settings=settings,
            theme=theme,
            style=style,
            title=title,
            description=description,
        )

    def generate_css_from_theme(self, theme: ReportTheme, style: StyleConfiguration) -> str:
        """
        Generate CSS styles from theme and style configuration.

        Args:
            theme: ReportTheme object
            style: StyleConfiguration object

        Returns:
            CSS string
        """
        css = f"""
        body {{
            font-family: {theme.font_family};
            font-size: {theme.font_size_base};
            margin: 40px;
            background-color: {theme.background_color};
            color: {theme.text_color};
        }}
        .container {{
            max-width: {style.container_max_width};
            margin: 0 auto;
            background: white;
            padding: {style.container_padding};
            border-radius: {style.border_radius};
            box-shadow: {style.box_shadow};
        }}
        h1, h2, h3 {{
            color: {theme.secondary_color};
            font-weight: {style.heading_font_weight};
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: {style.metrics_grid_columns};
            gap: {style.grid_gap};
            margin: 30px 0;
        }}
        .metric-card {{
            background: {theme.background_color};
            padding: {style.card_padding};
            border-radius: {style.border_radius};
            border-left: {style.card_border_width} solid {theme.primary_color};
        }}
        .metric-value {{
            font-size: {style.metric_value_font_size};
            font-weight: bold;
            color: {theme.secondary_color};
        }}
        .metric-label {{
            color: {theme.text_color};
            margin-top: 5px;
        }}
        .severity-critical {{
            color: {theme.danger_color};
            font-weight: bold;
        }}
        .severity-error {{
            color: #e67e22;
            font-weight: bold;
        }}
        .severity-warning {{
            color: {theme.warning_color};
        }}
        .severity-info {{
            color: {theme.info_color};
        }}
        .issue-item {{
            margin: 10px 0;
            padding: {style.issue_item_padding};
            background: {theme.background_color};
            border-radius: {style.border_radius};
            border-left: {style.issue_border_width} solid {theme.border_color};
        }}
        .file-path {{
            font-family: monospace;
            background: #ecf0f1;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        .tool-badge {{
            background: {theme.primary_color};
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
        }}
        .timestamp {{
            color: {theme.text_color};
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: {style.table_padding};
            text-align: left;
            border-bottom: {style.table_border};
        }}
        th {{
            background-color: {theme.background_color};
            font-weight: bold;
        }}
        .score-excellent {{
            color: {theme.success_color};
            font-weight: bold;
        }}
        .score-good {{
            color: {theme.warning_color};
        }}
        .score-poor {{
            color: {theme.danger_color};
            font-weight: bold;
        }}
        """
        return css

    def export_configuration(self, theme_name: str, style_name: str) -> dict[str, Any]:
        """
        Export configuration as dictionary.

        Args:
            theme_name: Theme name
            style_name: Style name

        Returns:
            Configuration dictionary
        """
        theme = self.get_theme(theme_name)
        style = self.get_style(style_name)

        return {
            "theme": theme.model_dump() if theme else None,
            "style": style.model_dump() if style else None,
            "css": (self.generate_css_from_theme(theme, style) if theme and style else None),
        }


def create_configuration_manager() -> ReportConfigurationManager:
    """Factory function to create ReportConfigurationManager."""
    return ReportConfigurationManager()


def create_default_settings() -> ReportSettings:
    """Create default report settings."""
    return ReportSettings()


def create_default_context(metrics: QualityMetrics) -> ReportContext:
    """Create default report context with provided metrics."""
    return ReportContext(
        metrics=metrics,
        settings=create_default_settings(),
    )
