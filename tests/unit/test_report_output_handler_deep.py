"""Unit tests for report_output_handler.py — ReportOutputHandler and create_output_handler.

Covers:
- __init__ with defaults and custom args
- export_html_report (single result, list, custom filename, extension handling, error path)
- export_json_summary (with/without severity filter, custom filename, error path)
- export_csv_report (normal flow, CSV escaping, custom filename, error path)
- export_dashboard_data (normal, empty results, custom filename, success rate edge cases)
- export_comparative_report (normal, custom filename, error path)
- _escape_csv_value (None, plain, commas, quotes, newlines)
- _generate_quality_alerts (critical, low score, tool failures, no alerts)
- _generate_comparative_recommendations (regression, improvement, issue increase, stable)
- get_supported_formats
- set_output_directory
- create_output_handler factory
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.testing_models import QualityResult
from processing.quality_analysis.report_configuration import QualityMetrics, ReportSettings
from processing.quality_analysis.report_output_handler import (
    ReportOutputHandler,
    create_output_handler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime.now(UTC)


def _make_metrics(**overrides) -> QualityMetrics:
    defaults = dict(
        total_files_analyzed=5,
        total_issues=10,
        overall_score=85.0,
        critical_issues=1,
        error_issues=2,
        warning_issues=4,
        info_issues=3,
        tools_used=["ruff", "ty"],
        successful_tools=2,
        failed_tools=0,
        total_execution_time=3.5,
        average_execution_time=1.75,
        languages_analyzed={"python": 5},
    )
    defaults.update(overrides)
    return QualityMetrics(**defaults)


def _make_result(**overrides) -> QualityResult:
    defaults = dict(
        analysis_id="test-analysis-001",
        success=True,
        overall_score=85.0,
        total_issues=0,
        issues=[],
        tools_results={"ruff": {"status": "success"}},
        execution_time=2.5,
        created_at=NOW,
        language="python",
        error=None,
    )
    defaults.update(overrides)
    return QualityResult(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output(tmp_path):
    """Return a tmp directory string for report output."""
    return str(tmp_path / "reports")


@pytest.fixture
def mock_template_manager():
    mgr = MagicMock()
    mgr.render_html_template = AsyncMock(return_value="<html>report</html>")
    return mgr


@pytest.fixture
def mock_data_processor():
    dp = MagicMock()
    dp.calculate_metrics = AsyncMock(return_value=_make_metrics())
    dp.generate_issue_summary = AsyncMock(return_value={"total": 10, "by_severity": {}})
    dp.generate_tool_performance = AsyncMock(return_value={"ruff": {"success": True}})
    dp.generate_recommendations = AsyncMock(return_value=["Fix critical issues"])
    dp.filter_and_format_issues = AsyncMock(
        return_value=[
            {
                "tool": "ruff",
                "severity": "warning",
                "file_path": "main.py",
                "line_number": 10,
                "column_number": 5,
                "rule_id": "E501",
                "message": "Line too long",
                "suggested_fix": "Split the line",
            }
        ]
    )
    dp.calculate_comparative_metrics = AsyncMock(
        return_value={"score_change": -3.0, "issue_change_percentage": 5.0}
    )
    dp.identify_improvements = AsyncMock(return_value=["Better naming"])
    dp.identify_regressions = AsyncMock(return_value=["More warnings"])
    return dp


@pytest.fixture
def handler(tmp_output, mock_template_manager, mock_data_processor):
    return ReportOutputHandler(
        output_directory=tmp_output,
        template_manager=mock_template_manager,
        data_processor=mock_data_processor,
    )


# ===========================================================================
# __init__
# ===========================================================================


class TestInit:
    def test_default_output_directory(self, mock_template_manager, mock_data_processor):
        h = ReportOutputHandler(
            template_manager=mock_template_manager,
            data_processor=mock_data_processor,
        )
        assert h.output_directory == Path("/var/tmp/quality_reports")

    def test_custom_output_directory(self, tmp_output, mock_template_manager, mock_data_processor):
        h = ReportOutputHandler(
            output_directory=tmp_output,
            template_manager=mock_template_manager,
            data_processor=mock_data_processor,
        )
        assert h.output_directory == Path(tmp_output)
        assert h.output_directory.exists()

    def test_creates_missing_directory(self, tmp_path, mock_template_manager, mock_data_processor):
        deep = str(tmp_path / "a" / "b" / "c")
        ReportOutputHandler(
            output_directory=deep,
            template_manager=mock_template_manager,
            data_processor=mock_data_processor,
        )
        assert Path(deep).exists()


# ===========================================================================
# export_html_report
# ===========================================================================


class TestExportHtmlReport:
    async def test_single_result(self, handler):
        result = _make_result()
        path = await handler.export_html_report(result)
        assert path.endswith(".html")
        assert Path(path).exists()
        assert Path(path).read_text() == "<html>report</html>"

    async def test_list_of_results(self, handler):
        results = [_make_result(analysis_id="a1"), _make_result(analysis_id="a2")]
        path = await handler.export_html_report(results)
        assert Path(path).exists()

    async def test_custom_filename_with_extension(self, handler):
        path = await handler.export_html_report(_make_result(), filename="my_report.html")
        assert Path(path).name == "my_report.html"

    async def test_custom_filename_without_extension(self, handler):
        path = await handler.export_html_report(_make_result(), filename="my_report")
        assert Path(path).name == "my_report.html"

    async def test_with_settings(self, handler):
        settings = ReportSettings(include_fix_suggestions=False, severity_filter=["critical"])
        path = await handler.export_html_report(_make_result(), settings=settings)
        assert Path(path).exists()

    async def test_error_propagation(self, handler):
        handler.data_processor.calculate_metrics = AsyncMock(side_effect=RuntimeError("calc fail"))
        with pytest.raises(RuntimeError, match="calc fail"):
            await handler.export_html_report(_make_result())


# ===========================================================================
# export_json_summary
# ===========================================================================


class TestExportJsonSummary:
    async def test_basic_export(self, handler):
        path = await handler.export_json_summary(_make_result())
        assert path.endswith(".json")
        data = json.loads(Path(path).read_text())
        assert "report_metadata" in data
        assert "quality_metrics" in data
        assert "analysis_results" in data
        assert "issue_summary" in data
        assert "tool_performance" in data
        assert "recommendations" in data

    async def test_analysis_result_fields(self, handler):
        r = _make_result(analysis_id="xyz", language="go", overall_score=92.0)
        path = await handler.export_json_summary(r)
        data = json.loads(Path(path).read_text())
        ar = data["analysis_results"][0]
        assert ar["analysis_id"] == "xyz"
        assert ar["language"] == "go"
        assert ar["overall_score"] == 92.0
        assert ar["tools_executed"] == ["ruff"]

    async def test_severity_filter_adds_issues(self, handler):
        settings = ReportSettings(severity_filter=["critical", "error"])
        path = await handler.export_json_summary(_make_result(), settings=settings)
        data = json.loads(Path(path).read_text())
        ar = data["analysis_results"][0]
        assert "issues" in ar

    async def test_empty_severity_filter_no_issues_key(self, handler):
        settings = ReportSettings(severity_filter=[])
        path = await handler.export_json_summary(_make_result(), settings=settings)
        data = json.loads(Path(path).read_text())
        ar = data["analysis_results"][0]
        assert "issues" not in ar

    async def test_custom_filename(self, handler):
        path = await handler.export_json_summary(_make_result(), filename="summary")
        assert Path(path).name == "summary.json"

    async def test_custom_filename_with_extension(self, handler):
        path = await handler.export_json_summary(_make_result(), filename="summary.json")
        assert Path(path).name == "summary.json"

    async def test_multiple_results(self, handler):
        results = [_make_result(analysis_id="a1"), _make_result(analysis_id="a2")]
        path = await handler.export_json_summary(results)
        data = json.loads(Path(path).read_text())
        assert len(data["analysis_results"]) == 2

    async def test_error_propagation(self, handler):
        handler.data_processor.calculate_metrics = AsyncMock(side_effect=ValueError("metrics fail"))
        with pytest.raises(ValueError, match="metrics fail"):
            await handler.export_json_summary(_make_result())


# ===========================================================================
# export_csv_report
# ===========================================================================


class TestExportCsvReport:
    async def test_basic_export(self, handler):
        path = await handler.export_csv_report(_make_result())
        assert path.endswith(".csv")
        content = Path(path).read_text()
        lines = content.strip().split("\n")
        assert lines[0].startswith("Analysis ID,Language,Tool")
        # 1 header + 1 data line from mock
        assert len(lines) == 2

    async def test_custom_filename_without_extension(self, handler):
        path = await handler.export_csv_report(_make_result(), filename="issues")
        assert Path(path).name == "issues.csv"

    async def test_custom_filename_with_extension(self, handler):
        path = await handler.export_csv_report(_make_result(), filename="issues.csv")
        assert Path(path).name == "issues.csv"

    async def test_auto_generated_filename(self, handler):
        path = await handler.export_csv_report(_make_result())
        name = Path(path).name
        assert name.startswith("quality_report_")
        assert name.endswith(".csv")

    async def test_multiple_results(self, handler):
        results = [_make_result(analysis_id="a1"), _make_result(analysis_id="a2")]
        path = await handler.export_csv_report(results)
        content = Path(path).read_text()
        lines = content.strip().split("\n")
        # 1 header + 2 data lines (one per result, each with one issue from mock)
        assert len(lines) == 3

    async def test_error_propagation(self, handler):
        handler.data_processor.filter_and_format_issues = AsyncMock(
            side_effect=RuntimeError("filter fail")
        )
        with pytest.raises(RuntimeError, match="filter fail"):
            await handler.export_csv_report(_make_result())


# ===========================================================================
# _escape_csv_value
# ===========================================================================


class TestEscapeCsvValue:
    def test_none(self, handler):
        assert handler._escape_csv_value(None) == ""

    def test_plain_string(self, handler):
        assert handler._escape_csv_value("hello") == "hello"

    def test_with_comma(self, handler):
        assert handler._escape_csv_value("a,b") == '"a,b"'

    def test_with_double_quotes(self, handler):
        assert handler._escape_csv_value('say "hi"') == '"say ""hi"""'

    def test_with_newline(self, handler):
        assert handler._escape_csv_value("line1\nline2") == '"line1\nline2"'

    def test_integer_input(self, handler):
        assert handler._escape_csv_value(42) == "42"

    def test_empty_string(self, handler):
        assert handler._escape_csv_value("") == ""

    def test_with_comma_and_quotes(self, handler):
        val = handler._escape_csv_value('a,"b"')
        assert val == '"a,""b"""'


# ===========================================================================
# export_dashboard_data
# ===========================================================================


class TestExportDashboardData:
    async def test_basic_export(self, handler):
        results = [_make_result()]
        path = await handler.export_dashboard_data(results)
        assert path.endswith(".json")
        data = json.loads(Path(path).read_text())
        assert "dashboard_metadata" in data
        assert "key_metrics" in data
        assert "quality_alerts" in data
        assert data["dashboard_metadata"]["total_analyses"] == 1

    async def test_empty_results_raises(self, handler):
        with pytest.raises(ValueError, match="No quality results"):
            await handler.export_dashboard_data([])

    async def test_custom_filename(self, handler):
        path = await handler.export_dashboard_data([_make_result()], filename="dash")
        assert Path(path).name == "dash.json"

    async def test_custom_filename_with_extension(self, handler):
        path = await handler.export_dashboard_data([_make_result()], filename="dash.json")
        assert Path(path).name == "dash.json"

    async def test_tools_success_rate_with_tools(self, handler):
        metrics = _make_metrics(successful_tools=8, failed_tools=2)
        handler.data_processor.calculate_metrics = AsyncMock(return_value=metrics)
        path = await handler.export_dashboard_data([_make_result()])
        data = json.loads(Path(path).read_text())
        assert data["key_metrics"]["tools_success_rate"] == 80.0

    async def test_tools_success_rate_zero_tools(self, handler):
        metrics = _make_metrics(successful_tools=0, failed_tools=0)
        handler.data_processor.calculate_metrics = AsyncMock(return_value=metrics)
        path = await handler.export_dashboard_data([_make_result()])
        data = json.loads(Path(path).read_text())
        assert data["key_metrics"]["tools_success_rate"] == 100.0

    async def test_date_range(self, handler):
        t1 = datetime(2025, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 6, 15, tzinfo=UTC)
        results = [_make_result(created_at=t1), _make_result(created_at=t2)]
        path = await handler.export_dashboard_data(results)
        data = json.loads(Path(path).read_text())
        dr = data["dashboard_metadata"]["date_range"]
        assert dr["start"] == t1.isoformat()
        assert dr["end"] == t2.isoformat()

    async def test_error_propagation(self, handler):
        handler.data_processor.calculate_metrics = AsyncMock(
            side_effect=RuntimeError("dashboard fail")
        )
        with pytest.raises(RuntimeError, match="dashboard fail"):
            await handler.export_dashboard_data([_make_result()])


# ===========================================================================
# export_comparative_report
# ===========================================================================


class TestExportComparativeReport:
    async def test_basic_export(self, handler):
        current = [_make_result(analysis_id="c1")]
        baseline = [_make_result(analysis_id="b1")]
        path = await handler.export_comparative_report(current, baseline)
        assert path.endswith(".json")
        data = json.loads(Path(path).read_text())
        assert "comparison_metadata" in data
        assert "quality_changes" in data
        assert "recommendations" in data

    async def test_custom_filename(self, handler):
        path = await handler.export_comparative_report(
            [_make_result()], [_make_result()], filename="comp"
        )
        assert Path(path).name == "comp.json"

    async def test_custom_filename_with_extension(self, handler):
        path = await handler.export_comparative_report(
            [_make_result()], [_make_result()], filename="comp.json"
        )
        assert Path(path).name == "comp.json"

    async def test_with_settings(self, handler):
        settings = ReportSettings(severity_filter=["critical"])
        path = await handler.export_comparative_report(
            [_make_result()], [_make_result()], settings=settings
        )
        assert Path(path).exists()

    async def test_error_propagation(self, handler):
        handler.data_processor.calculate_comparative_metrics = AsyncMock(
            side_effect=RuntimeError("compare fail")
        )
        with pytest.raises(RuntimeError, match="compare fail"):
            await handler.export_comparative_report([_make_result()], [_make_result()])


# ===========================================================================
# _generate_quality_alerts
# ===========================================================================


class TestGenerateQualityAlerts:
    async def test_critical_issues_alert(self, handler):
        metrics = _make_metrics(critical_issues=3, overall_score=90.0, failed_tools=0)
        alerts = await handler._generate_quality_alerts(metrics, [])
        types_ = [a["type"] for a in alerts]
        assert "critical" in types_
        assert any("3 critical issues" in a["message"] for a in alerts)

    async def test_low_score_alert(self, handler):
        metrics = _make_metrics(critical_issues=0, overall_score=55.0, failed_tools=0)
        alerts = await handler._generate_quality_alerts(metrics, [])
        types_ = [a["type"] for a in alerts]
        assert "warning" in types_
        assert any("55.0" in a["message"] for a in alerts)

    async def test_tool_failure_alert(self, handler):
        metrics = _make_metrics(critical_issues=0, overall_score=90.0, failed_tools=2)
        alerts = await handler._generate_quality_alerts(metrics, [])
        types_ = [a["type"] for a in alerts]
        assert "info" in types_
        assert any("2 tool" in a["message"] for a in alerts)

    async def test_no_alerts(self, handler):
        metrics = _make_metrics(critical_issues=0, overall_score=90.0, failed_tools=0)
        alerts = await handler._generate_quality_alerts(metrics, [])
        assert alerts == []

    async def test_all_alerts(self, handler):
        metrics = _make_metrics(critical_issues=1, overall_score=50.0, failed_tools=1)
        alerts = await handler._generate_quality_alerts(metrics, [])
        types_ = [a["type"] for a in alerts]
        assert "critical" in types_
        assert "warning" in types_
        assert "info" in types_


# ===========================================================================
# _generate_comparative_recommendations
# ===========================================================================


class TestGenerateComparativeRecommendations:
    async def test_significant_regression(self, handler):
        data = {"score_change": -10, "issue_change_percentage": 0}
        recs = await handler._generate_comparative_recommendations(data, [], [])
        assert any("regression" in r.lower() for r in recs)

    async def test_significant_improvement(self, handler):
        data = {"score_change": 10, "issue_change_percentage": 0}
        recs = await handler._generate_comparative_recommendations(data, [], [])
        assert any("improvement" in r.lower() for r in recs)

    async def test_issue_increase(self, handler):
        data = {"score_change": 0, "issue_change_percentage": 25}
        recs = await handler._generate_comparative_recommendations(data, [], [])
        assert any("increased" in r.lower() or "issue count" in r.lower() for r in recs)

    async def test_stable(self, handler):
        data = {"score_change": 0, "issue_change_percentage": 0}
        recs = await handler._generate_comparative_recommendations(data, [], [])
        assert any("stable" in r.lower() for r in recs)

    async def test_regression_and_issue_increase(self, handler):
        data = {"score_change": -8, "issue_change_percentage": 30}
        recs = await handler._generate_comparative_recommendations(data, [], [])
        assert len(recs) == 2

    async def test_missing_keys_defaults_to_stable(self, handler):
        recs = await handler._generate_comparative_recommendations({}, [], [])
        assert any("stable" in r.lower() for r in recs)


# ===========================================================================
# get_supported_formats / set_output_directory
# ===========================================================================


class TestMiscMethods:
    def test_get_supported_formats(self, handler):
        formats = handler.get_supported_formats()
        assert "html" in formats
        assert "json" in formats
        assert "csv" in formats
        assert "dashboard" in formats
        assert "comparative" in formats

    def test_set_output_directory(self, handler, tmp_path):
        new_dir = str(tmp_path / "new_output")
        handler.set_output_directory(new_dir)
        assert handler.output_directory == Path(new_dir)
        assert Path(new_dir).exists()

    def test_set_output_directory_nested(self, handler, tmp_path):
        new_dir = str(tmp_path / "a" / "b" / "c")
        handler.set_output_directory(new_dir)
        assert Path(new_dir).exists()


# ===========================================================================
# create_output_handler factory
# ===========================================================================


class TestCreateOutputHandler:
    def test_with_custom_directory(self, tmp_path):
        d = str(tmp_path / "factory_out")
        h = create_output_handler(output_directory=d)
        assert h.output_directory == Path(d)

    def test_with_env_var(self, tmp_path, monkeypatch):
        d = str(tmp_path / "env_out")
        monkeypatch.setenv("QUALITY_REPORTS_DIR", d)
        h = create_output_handler()
        assert h.output_directory == Path(d)

    def test_default_fallback(self):
        h = create_output_handler()
        assert h.output_directory == Path("/var/tmp/quality_reports")

    def test_with_custom_managers(self, tmp_path, mock_template_manager, mock_data_processor):
        d = str(tmp_path / "custom")
        h = create_output_handler(
            output_directory=d,
            template_manager=mock_template_manager,
            data_processor=mock_data_processor,
        )
        assert h.template_manager is mock_template_manager
        assert h.data_processor is mock_data_processor
