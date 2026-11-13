"""
Report Template Manager - Handle HTML templates and rendering for quality reports.
Provides template management, HTML generation, and styling for quality analysis reports.
"""

from __future__ import annotations

import logging
from typing import Any

from models.testing_models import QualityResult

from .report_configuration import ReportContext

try:
    from jinja2 import BaseLoader, Environment, select_autoescape

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

logger = logging.getLogger(__name__)


class ReportTemplateManager:
    """
    Manages HTML templates and rendering for quality reports.
    Handles template loading, rendering, and HTML generation.
    """

    def __init__(self):
        """Initialize the template manager with Jinja2 environment."""
        if HAS_JINJA2:
            self.jinja_env = Environment(
                loader=BaseLoader(), autoescape=select_autoescape(["html", "xml"])
            )
        else:
            self.jinja_env = None
        logger.info("ReportTemplateManager initialized")

    async def render_html_template(
        self, results: list[QualityResult], context: ReportContext
    ) -> str:
        """
        Render HTML template with quality data.

        Args:
            results: List of QualityResult objects
            context: Report context with metrics and settings

        Returns:
            Rendered HTML content as string
        """
        try:
            if not HAS_JINJA2:
                return self._render_basic_html(results, context)

            template_content = self._get_html_template()
            assert self.jinja_env is not None, "Jinja2 environment not available"
            template = self.jinja_env.from_string(template_content)
            return template.render(results=results, context=context)
        except Exception as e:
            logger.error("Failed to render HTML template: %s", e)
            raise

    def _get_html_template(self) -> str:
        """Get the main HTML template for quality reports."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ context.title }}</title>
    <style>
        {{ self._get_css_styles() }}
    </style>
</head>
<body>
    <div class="container">
        {{ self._render_header() }}
        {{ self._render_metrics_overview() }}
        {{ self._render_severity_distribution() }}
        {{ self._render_analysis_results() }}
        {{ self._render_footer() }}
    </div>
</body>
</html>
        """

    def _get_css_styles(self) -> str:
        """Get CSS styles for the HTML report."""
        return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            margin: 40px;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            color: #7f8c8d;
            margin-top: 5px;
        }
        .severity-critical {
            color: #e74c3c;
            font-weight: bold;
        }
        .severity-error {
            color: #e67e22;
            font-weight: bold;
        }
        .severity-warning {
            color: #f39c12;
        }
        .severity-info {
            color: #3498db;
        }
        .issue-item {
            margin: 10px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
            border-left: 3px solid #bdc3c7;
        }
        .file-path {
            font-family: monospace;
            background: #ecf0f1;
            padding: 2px 6px;
            border-radius: 3px;
        }
        .tool-badge {
            background: #3498db;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
        }
        .timestamp {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .score-excellent {
            color: #27ae60;
            font-weight: bold;
        }
        .score-good {
            color: #f39c12;
        }
        .score-poor {
            color: #e74c3c;
            font-weight: bold;
        }
        """

    def _render_header(self) -> str:
        """Render the report header template."""
        return """
        <header>
            <h1>{{ context.title }}</h1>
            <p>{{ context.description }}</p>
            <p class="timestamp">Generated: {{ context.generation_time.strftime('%Y-%m-%d %H:%M:%S UTC') }}</p>
        </header>
        """

    def _render_metrics_overview(self) -> str:
        """Render the metrics overview template."""
        return """
        <section>
            <h2>Quality Metrics Overview</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value {% if context.metrics.overall_score >= 80 %}score-excellent{% elif context.metrics.overall_score >= 60 %}score-good{% else %}score-poor{% endif %}">
                        {{ "%.1f" | format(context.metrics.overall_score) }}
                    </div>
                    <div class="metric-label">Overall Quality Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ context.metrics.total_issues }}</div>
                    <div class="metric-label">Total Issues Found</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ context.metrics.total_files_analyzed }}</div>
                    <div class="metric-label">Files Analyzed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ context.metrics.tools_used | length }}</div>
                    <div class="metric-label">Tools Used</div>
                </div>
            </div>
        </section>
        """

    def _render_severity_distribution(self) -> str:
        """Render the severity distribution table template."""
        return """
        <section>
            <h2>Issue Distribution by Severity</h2>
            <table>
                <tr>
                    <th>Severity</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                <tr>
                    <td class="severity-critical">Critical</td>
                    <td>{{ context.metrics.critical_issues }}</td>
                    <td>{{ "%.1f" | format((context.metrics.critical_issues / context.metrics.total_issues * 100) if context.metrics.total_issues > 0 else 0) }}%</td>
                </tr>
                <tr>
                    <td class="severity-error">Error</td>
                    <td>{{ context.metrics.error_issues }}</td>
                    <td>{{ "%.1f" | format((context.metrics.error_issues / context.metrics.total_issues * 100) if context.metrics.total_issues > 0 else 0) }}%</td>
                </tr>
                <tr>
                    <td class="severity-warning">Warning</td>
                    <td>{{ context.metrics.warning_issues }}</td>
                    <td>{{ "%.1f" | format((context.metrics.warning_issues / context.metrics.total_issues * 100) if context.metrics.total_issues > 0 else 0) }}%</td>
                </tr>
                <tr>
                    <td class="severity-info">Info</td>
                    <td>{{ context.metrics.info_issues }}</td>
                    <td>{{ "%.1f" | format((context.metrics.info_issues / context.metrics.total_issues * 100) if context.metrics.total_issues > 0 else 0) }}%</td>
                </tr>
            </table>
        </section>
        """

    def _render_analysis_results(self) -> str:
        """Render the analysis results template."""
        return """
        {% for result in results %}
        <section>
            <h2>Analysis: {{ result.analysis_id }}</h2>
            <p><strong>Language:</strong> {{ result.language }}</p>
            <p><strong>Success:</strong> {{ "✅" if result.success else "❌" }}</p>
            <p><strong>Score:</strong> <span class="{% if result.overall_score >= 80 %}score-excellent{% elif result.overall_score >= 60 %}score-good{% else %}score-poor{% endif %}">{{ "%.1f" | format(result.overall_score) }}</span></p>
            <p><strong>Execution Time:</strong> {{ "%.2f" | format(result.execution_time) }}s</p>

            {% if result.issues %}
            <h3>Issues Found ({{ result.issues | length }})</h3>
            {% for issue in result.issues[:context.settings.max_issues_per_file] %}
            <div class="issue-item">
                <div><span class="tool-badge">{{ issue.tool }}</span> <span class="severity-{{ issue.severity }}">{{ issue.severity.upper() }}</span></div>
                <div><strong>{{ issue.message }}</strong></div>
                <div>File: <span class="file-path">{{ issue.file_path }}</span>
                {% if issue.line_number %} Line: {{ issue.line_number }}{% endif %}
                {% if issue.rule_id %} Rule: {{ issue.rule_id }}{% endif %}
                </div>
                {% if issue.suggested_fix and context.settings.include_fix_suggestions %}
                <div><strong>Suggested Fix:</strong> {{ issue.suggested_fix }}</div>
                {% endif %}
            </div>
            {% endfor %}
            {% if result.issues | length > context.settings.max_issues_per_file %}
            <p><em>... and {{ result.issues | length - context.settings.max_issues_per_file }} more issues</em></p>
            {% endif %}
            {% else %}
            <p>✅ No issues found!</p>
            {% endif %}
        </section>
        {% endfor %}
        """

    def _render_footer(self) -> str:
        """Render the report footer template."""
        return """
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d;">
            <p>Report generated by Code Quality Analysis System v1.0.0</p>
        </footer>
        """

    def _render_basic_html(self, results: list[QualityResult], context: ReportContext) -> str:
        """
        Render basic HTML without Jinja2 (fallback).

        Args:
            results: List of QualityResult objects
            context: Report context with metrics and settings

        Returns:
            Basic HTML content
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            f"<title>{context.title}</title>",
            "<style>body{font-family:Arial,sans-serif;margin:40px;}</style>",
            "</head>",
            "<body>",
            f"<h1>{context.title}</h1>",
            f"<p>{context.description}</p>",
            f"<p>Generated: {context.generation_time}</p>",
            "<h2>Quality Metrics</h2>",
            f"<p>Overall Score: {context.metrics.overall_score:.1f}</p>",
            f"<p>Total Issues: {context.metrics.total_issues}</p>",
            f"<p>Files Analyzed: {context.metrics.total_files_analyzed}</p>",
        ]

        for result in results:
            html_parts.extend(
                [
                    f"<h3>Analysis: {result.analysis_id}</h3>",
                    f"<p>Language: {result.language}</p>",
                    f"<p>Success: {'✅' if result.success else '❌'}</p>",
                    f"<p>Score: {result.overall_score:.1f}</p>",
                    f"<p>Issues: {len(result.issues)}</p>",
                ]
            )

        html_parts.extend(["</body>", "</html>"])

        return "\n".join(html_parts)

    async def render_custom_template(self, template_content: str, **kwargs: Any) -> str:
        """
        Render a custom template with provided content.

        Args:
            template_content: Custom template content
            **kwargs: Template variables

        Returns:
            Rendered template content
        """
        try:
            if not HAS_JINJA2:
                logger.warning("Jinja2 not available, returning template content as-is")
                return template_content

            assert self.jinja_env is not None, "Jinja2 environment not available"
            template = self.jinja_env.from_string(template_content)
            return template.render(**kwargs)
        except Exception as e:
            logger.error("Failed to render custom template: %s", e)
            raise

    def get_available_templates(self) -> list[str]:
        """Get list of available template types."""
        return [
            "html_report",
            "metrics_overview",
            "severity_distribution",
            "analysis_results",
            "custom",
        ]


def create_template_manager() -> ReportTemplateManager:
    """Factory function to create ReportTemplateManager."""
    return ReportTemplateManager()
