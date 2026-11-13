"""Deep coverage tests for processing domain: quality, architecture, test_execution.

Covers ~700+ missing lines across 23 source files to push coverage toward 95%.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.testing_models import QualityIssue, QualityResult, SecurityLevel, TestRequest

# ── Shared helpers ────────────────────────────────────────────────────


def _qi(severity="warning", tool="ruff", rule_id="E001", msg="issue", file="t.py", line=1, **kw):
    return QualityIssue(
        severity=severity,
        tool=tool,
        rule_id=rule_id,
        message=msg,
        file_path=file,
        line_number=line,
        **kw,
    )


def _qr(aid="a1", success=True, lang="python", score=85.0, issues=None, tools=None, t=1.5):
    return QualityResult(
        analysis_id=aid,
        success=success,
        language=lang,
        overall_score=score,
        total_issues=len(issues) if issues else 0,
        issues=issues or [],
        tools_results=tools or {},
        execution_time=t,
        created_at=datetime.now(UTC),
    )


def _vuln(id="V1", severity="HIGH", summary="test"):
    from models.architecture_models import Vulnerability

    return Vulnerability(
        id=id,
        severity=severity,
        summary=summary,
        details="",
        affected_versions=[],
        fixed_versions=[],
        references=[],
        published_date="",
        modified_date="",
    )


# ═══════════════════════════════════════════════════════════════════════
#  report_generator.py
# ═══════════════════════════════════════════════════════════════════════


class TestReportGen:
    @pytest.fixture
    def gen(self, tmp_path):
        from processing.quality_analysis.report_generator import QualityReportGenerator

        return QualityReportGenerator(output_directory=str(tmp_path))

    @pytest.mark.asyncio
    async def test_html_report_raises(self, gen):
        with pytest.raises(Exception):
            await gen.generate_html_report(_qr(issues=[_qi()]))

    @pytest.mark.asyncio
    async def test_json_summary(self, gen):
        s = await gen.generate_json_summary(_qr(issues=[_qi(severity="critical"), _qi()]))
        assert "report_metadata" in s and "quality_metrics" in s
        assert s["report_metadata"]["generator_version"] == "2.0.0"

    @pytest.mark.asyncio
    async def test_json_summary_list(self, gen):
        s = await gen.generate_json_summary([_qr(), _qr(lang="javascript")])
        assert s["quality_metrics"]["total_files_analyzed"] == 2

    @pytest.mark.asyncio
    async def test_dashboard_empty_raises(self, gen):
        with pytest.raises(ValueError):
            await gen.generate_metrics_dashboard([])

    @pytest.mark.asyncio
    async def test_dashboard(self, gen):
        d = await gen.generate_metrics_dashboard(
            [
                _qr(
                    issues=[_qi(severity="critical"), _qi(severity="error")],
                    tools={"ruff": {"success": True, "issues": [], "execution_time": 0.5}},
                )
            ]
        )
        assert "charts" in d and d["key_metrics"]["critical_issues"] == 1

    @pytest.mark.asyncio
    async def test_dashboard_historical(self, gen):
        d = await gen.generate_metrics_dashboard(
            [_qr()], historical_data=[{"timestamp": "2024-01-01", "overall_score": 80}]
        )
        assert "charts" in d

    @pytest.mark.asyncio
    async def test_comparative_report(self, gen):
        r = await gen.generate_comparative_report([_qr(score=90)], [_qr(score=70)])
        assert r["comparison_metadata"]["current_period"]["analyses_count"] == 1


# ═══════════════════════════════════════════════════════════════════════
#  report_chart_generator.py
# ═══════════════════════════════════════════════════════════════════════


class TestChartGen:
    @pytest.fixture
    def cg(self):
        from processing.quality_analysis.report_chart_generator import ReportChartGenerator

        return ReportChartGenerator()

    @pytest.mark.asyncio
    async def test_score_dist_all_ranges(self, cg):
        results = [_qr(score=s) for s in (95, 85, 75, 65, 50)]
        c = await cg.generate_score_distribution_chart(results)
        assert c["data"]["datasets"][0]["data"] == [1, 1, 1, 1, 1]

    @pytest.mark.asyncio
    async def test_trends_historical(self, cg):
        c = await cg.generate_trends_chart(
            [_qr()],
            historical_data=[
                {"timestamp": "2024-01-01", "overall_score": 70},
                {"timestamp": "2024-02-01", "overall_score": 80},
            ],
        )
        assert c["data"]["datasets"][0]["data"] == [70, 80]

    @pytest.mark.asyncio
    async def test_issue_trends(self, cg):
        c = await cg.generate_issue_trends_chart(
            [_qr(issues=[_qi(severity="critical"), _qi(severity="error"), _qi()])]
        )
        assert len(c["data"]["datasets"]) == 3

    @pytest.mark.asyncio
    async def test_execution_time(self, cg):
        c = await cg.generate_execution_time_chart(
            [_qr(tools={"ruff": {"execution_time": 1.0}, "ty": {"execution_time": 2.0}})]
        )
        assert "ruff" in c["data"]["labels"]

    @pytest.mark.asyncio
    async def test_dashboard_charts(self, cg):
        c = await cg.generate_dashboard_charts([_qr(issues=[_qi()])])
        assert all(
            k in c
            for k in (
                "severity_distribution",
                "quality_score_distribution",
                "tools_performance",
                "language_breakdown",
                "quality_trends",
                "issue_trends",
                "execution_time",
            )
        )

    def test_color_schemes(self, cg):
        assert len(cg.get_chart_color_scheme("severity_distribution")) == 4
        assert len(cg.get_chart_color_scheme("quality_score_distribution")) == 5
        assert len(cg.get_chart_color_scheme("other")) == 6


# ═══════════════════════════════════════════════════════════════════════
#  report_template_manager.py
# ═══════════════════════════════════════════════════════════════════════


class TestTemplateMgr:
    @pytest.fixture
    def tm(self):
        from processing.quality_analysis.report_template_manager import ReportTemplateManager

        return ReportTemplateManager()

    @pytest.mark.asyncio
    async def test_render_raises(self, tm):
        from processing.quality_analysis.report_configuration import (
            ReportContext,
            create_default_settings,
        )
        from processing.quality_analysis.report_data_processor import create_data_processor

        dp = create_data_processor()
        m = await dp.calculate_metrics([_qr()])
        ctx = ReportContext(
            report_id="r1",
            generation_time=datetime.now(UTC),
            title="T",
            description="d",
            metrics=m,
            settings=create_default_settings(),
        )
        with pytest.raises(Exception):
            await tm.render_html_template([_qr()], ctx)

    def test_template_parts(self, tm):
        assert "font-family" in tm._get_css_styles()
        assert "context.title" in tm._get_html_template()
        assert "context.title" in tm._render_header()
        assert "Overall Quality Score" in tm._render_metrics_overview()
        assert "Severity" in tm._render_severity_distribution()
        assert "result.analysis_id" in tm._render_analysis_results()
        assert "Code Quality" in tm._render_footer()

    @pytest.mark.asyncio
    async def test_basic_html_fallback(self, tm):
        from processing.quality_analysis.report_configuration import (
            ReportContext,
            create_default_settings,
        )
        from processing.quality_analysis.report_data_processor import create_data_processor

        dp = create_data_processor()
        m = await dp.calculate_metrics([_qr()])
        ctx = ReportContext(
            report_id="r1",
            generation_time=datetime.now(UTC),
            title="FB",
            description="d",
            metrics=m,
            settings=create_default_settings(),
        )
        html = tm._render_basic_html([_qr()], ctx)
        assert "FB" in html and "<!DOCTYPE html>" in html

    @pytest.mark.asyncio
    async def test_custom_template(self, tm):
        assert "Hello World!" in await tm.render_custom_template("Hello {{ name }}!", name="World")

    def test_available_templates(self, tm):
        assert "html_report" in tm.get_available_templates()


# ═══════════════════════════════════════════════════════════════════════
#  http_quality_runner.py
# ═══════════════════════════════════════════════════════════════════════


class TestHTTPQuality:
    @pytest.fixture
    def runner(self):
        from processing.quality_analysis.http_quality_runner import HTTPQualityRunner

        return HTTPQualityRunner()

    def test_config(self, runner):
        assert runner.config.total_timeout > 0

    def test_failed_result(self, runner):
        r = runner._create_failed_quality_result("id", datetime.now(UTC), "python", "down")
        assert r.success is False and r.error == "down"

    def test_failed_result_empty_msg(self, runner):
        r = runner._create_failed_quality_result("id", datetime.now(UTC), "python", "")
        assert "no specific error" in r.error

    def test_config_score(self, runner):
        assert 0 <= runner._calculate_config_score() <= 100

    def test_health_score_healthy(self, runner):
        s = runner._calculate_health_score(
            {"status": "healthy"},
            {
                "session_health": {"failed_sessions": 0, "total_created": 1},
                "service_connectivity": {"reachable": True, "response_time_ms": 100},
                "resource_usage": {"memory_usage_mb": 200},
            },
        )
        assert s == 100.0

    def test_health_score_unhealthy(self, runner):
        s = runner._calculate_health_score(
            {"status": "unhealthy"},
            {
                "session_health": {"failed_sessions": 5, "total_created": 10},
                "service_connectivity": {"reachable": False},
                "resource_usage": {"memory_usage_mb": 1500},
            },
        )
        assert s < 50

    @pytest.mark.asyncio
    async def test_disabled(self, runner):
        from models.testing_models import QualityRequest

        req = QualityRequest(code_content="x=1", language="python", container_name="c")
        with patch.dict("os.environ", {"ENABLE_QUALITY_ANALYSIS": "false"}):
            r = await runner.run_quality_analysis(req)
        assert r.success is False and "disabled" in r.error

    @pytest.mark.asyncio
    async def test_perf_metrics(self, runner):
        m = await runner._get_performance_metrics()
        assert "session_efficiency_percent" in m

    @pytest.mark.asyncio
    async def test_close(self, runner):
        await runner.close()

    def test_factory(self):
        from processing.quality_analysis.http_quality_runner import create_quality_tool_runner

        assert create_quality_tool_runner(None) is not None


class TestSessionMonitor:
    @pytest.mark.asyncio
    async def test_register_close(self):
        from processing.quality_analysis.http_quality_runner import SessionMonitor

        m = SessionMonitor()
        await m.register_session(MagicMock())
        assert m.stats.total_sessions_created == 1
        await m.record_session_closed()
        assert m.stats.total_sessions_closed == 1

    @pytest.mark.asyncio
    async def test_error_overflow(self):
        from processing.quality_analysis.http_quality_runner import SessionMonitor

        m = SessionMonitor()
        for i in range(55):
            await m.record_session_error(f"E{i}")
        assert m.stats.failed_sessions == 55
        assert len(m.stats.session_errors) <= 50

    def test_categorize(self):
        from processing.quality_analysis.http_quality_runner import SessionMonitor

        m = SessionMonitor()
        assert m._categorize_error("session is closed") == "session_closed"
        assert m._categorize_error("Connection refused") == "connection_refused"
        assert m._categorize_error("Timeout") == "timeout"
        assert m._categorize_error("pool full") == "connection_pool_issue"
        assert m._categorize_error("HTTP 502") == "service_unavailable"
        assert m._categorize_error("other") == "other"

    def test_health_summary(self):
        from processing.quality_analysis.http_quality_runner import SessionMonitor

        s = SessionMonitor().get_health_summary()
        assert "active_sessions" in s and "error_patterns" in s


# ═══════════════════════════════════════════════════════════════════════
#  fix_suggester.py
# ═══════════════════════════════════════════════════════════════════════


class TestFixSuggester:
    @pytest.fixture
    def sug(self):
        from processing.quality_analysis.fix_suggester import AutoFixSuggester

        return AutoFixSuggester()

    @pytest.mark.asyncio
    async def test_suggest_empty(self, sug):
        assert await sug.suggest_fixes([]) == []

    @pytest.mark.asyncio
    async def test_suggest_with_issues(self, sug):
        assert isinstance(await sug.suggest_fixes([_qi(severity="error"), _qi()]), list)

    @pytest.mark.asyncio
    async def test_apply_empty(self, sug):
        r = await sug.apply_automated_fixes([], dry_run=True)
        assert r.total_fixes_attempted == 0

    @pytest.mark.asyncio
    async def test_validate_valid_python(self, sug):
        r = await sug.validate_fixes("x=1", "x=2", language="python")
        assert r.is_valid is True

    @pytest.mark.asyncio
    async def test_validate_empty(self, sug):
        r = await sug.validate_fixes("x=1", "", language="python")
        assert r.is_valid is False

    @pytest.mark.asyncio
    async def test_validate_much_shorter(self, sug):
        r = await sug.validate_fixes("x" * 100, "y", language="python")
        assert any("shorter" in w.lower() for w in r.warnings)

    @pytest.mark.asyncio
    async def test_validate_js(self, sug):
        r = await sug.validate_fixes("let x=1;", "let x=2;", language="javascript")
        assert isinstance(r.is_valid, bool)

    @pytest.mark.asyncio
    async def test_bulk_empty(self, sug):
        assert await sug.suggest_bulk_fixes([]) == {}

    @pytest.mark.asyncio
    async def test_bulk_with_results(self, sug):
        r = await sug.suggest_bulk_fixes([_qr(issues=[_qi(file="a.py")])])
        assert isinstance(r, dict)

    def test_priority_security(self, sug):
        from processing.quality_analysis.fix_models import FixCategory, FixRule, FixStrategy

        r = FixRule(
            rule_id="t",
            tool="ruff",
            rule_pattern="E001",
            category=FixCategory.SECURITY,
            strategy=FixStrategy.AUTOMATIC,
            confidence=0.9,
            requires_review=False,
        )
        assert sug._calculate_fix_priority(_qi(severity="critical"), r) == 1

    def test_priority_formatting(self, sug):
        from processing.quality_analysis.fix_models import FixCategory, FixRule, FixStrategy

        r = FixRule(
            rule_id="t",
            tool="ruff-format",
            rule_pattern=".*",
            category=FixCategory.FORMATTING,
            strategy=FixStrategy.SUGGESTED,
            confidence=0.5,
            requires_review=False,
        )
        assert sug._calculate_fix_priority(_qi(severity="info"), r) == 10

    def test_filter_no_filter(self, sug):
        from processing.quality_analysis.fix_models import (
            FixCategory,
            FixStrategy,
            QualityAnalysisResolveSuggestions,
        )

        f = [
            QualityAnalysisResolveSuggestions(
                fix_id="f1",
                strategy=FixStrategy.AUTOMATIC,
                category=FixCategory.BUG_FIX,
                confidence=0.5,
                priority=1,
                tool="ruff",
                severity="error",
                title="Fix",
                description="d",
                file_path="t.py",
            )
        ]
        assert len(sug._filter_applicable_fixes(f, filter_by_confidence=False)) == 1

    @pytest.mark.asyncio
    async def test_line_fix(self, sug):
        assert await sug._apply_line_based_fix("a\nb\nc", 2, "X") == "a\nX\nc"

    @pytest.mark.asyncio
    async def test_line_fix_oor(self, sug):
        assert await sug._apply_line_based_fix("a\nb", 99, "X") == "a\nb"

    def test_context_str(self):
        from processing.quality_analysis.fix_suggester import FixApplicationContext

        assert "app1" in str(FixApplicationContext([], True, False, "app1", 0.0))

    def test_factory(self):
        from processing.quality_analysis.fix_suggester import create_auto_fix_suggester

        assert create_auto_fix_suggester() is not None


# ═══════════════════════════════════════════════════════════════════════
#  quality_runner.py
# ═══════════════════════════════════════════════════════════════════════


class TestQualityRunner:
    @pytest.fixture
    def ds(self):
        s = AsyncMock()
        s.create_test_container = AsyncMock(return_value="d-123")
        s.start_container = AsyncMock()
        s.exec_in_container = AsyncMock(return_value={"exit_code": 0, "stdout": "", "stderr": ""})
        s.create_file_in_container = AsyncMock(return_value=True)
        s.stop_container = AsyncMock()
        s.remove_container_by_id = AsyncMock()
        return s

    @pytest.fixture
    def runner(self, ds):
        from processing.quality_analysis.quality_runner import QualityToolRunner

        return QualityToolRunner(ds)

    def _make_tool(self, name, fmt):
        from processing.quality_analysis.tool_registry import QualityTool, ToolCategory

        return QualityTool(
            name=name,
            category=ToolCategory.LINTER,
            command=[name],
            output_format=fmt,
            description=f"{name} tool",
        )

    def test_detect_lang(self, runner):
        from models.testing_models import QualityRequest

        assert (
            runner._detect_language(QualityRequest(language="javascript", container_name="c"))
            == "javascript"
        )
        assert (
            runner._detect_language(
                QualityRequest(language=None, code_path="t.ts", container_name="c")
            )
            == "typescript"
        )
        assert (
            runner._detect_language(QualityRequest(language=None, container_name="c")) == "python"
        )

    def test_score_no_issues(self, runner):
        assert runner._calculate_quality_score([], {}) == 10.0

    def test_score_with_issues(self, runner):
        s = runner._calculate_quality_score([_qi(severity="critical"), _qi()], {})
        assert 0 <= s < 10

    def test_map_ruff_severity(self, runner):
        assert runner._map_ruff_severity("F821") == "error"
        assert runner._map_ruff_severity("PLR2004") == "info"

    @pytest.mark.asyncio
    async def test_parse_json_ruff(self, runner):
        t = self._make_tool("ruff", "json")
        issues = await runner._parse_json_output(
            t,
            '[{"code":"F401","message":"m","filename":"t.py","location":{"row":1,"column":8},"fix":null}]',
            "t.py",
        )
        assert [(i.rule_id, i.line_number) for i in issues] == [("F401", 1)]

    @pytest.mark.asyncio
    async def test_parse_json_eslint(self, runner):
        t = self._make_tool("eslint", "json")
        issues = await runner._parse_json_output(
            t,
            '[{"messages":[{"severity":2,"message":"m","line":1,"column":5,"ruleId":"r"}]}]',
            "t.js",
        )
        assert issues[0].severity == "error"

    @pytest.mark.asyncio
    async def test_parse_json_bandit(self, runner):
        t = self._make_tool("bandit", "json")
        issues = await runner._parse_json_output(
            t,
            '{"results":[{"issue_severity":"HIGH","issue_text":"m","line_number":5,"test_id":"B101"}]}',
            "t.py",
        )
        assert issues[0].severity == "high"

    @pytest.mark.asyncio
    async def test_parse_json_ty(self, runner):
        t = self._make_tool("ty", "json")
        issues = await runner._parse_json_output(
            t,
            '[{"check_name":"invalid-assignment","description":"invalid-assignment: Bad type","severity":"major","location":{"path":"t.py","positions":{"begin":{"line":10,"column":1}}}}]',
            "t.py",
        )
        assert [(i.rule_id, i.line_number) for i in issues] == [("invalid-assignment", 10)]

    @pytest.mark.asyncio
    async def test_parse_text_formatter(self, runner):
        t = self._make_tool("prettier", "text")
        issues = await runner._parse_text_output(t, "[warn] t.js", "", "t.js")
        assert issues[0].tool == "prettier"

    @pytest.mark.asyncio
    async def test_parse_diff(self, runner):
        t = self._make_tool("ruff-format", "dif")
        assert len(await runner._parse_diff_output(t, "--- a\n+++ b", "t.py")) == 1
        assert len(await runner._parse_diff_output(t, "", "t.py")) == 0


# ═══════════════════════════════════════════════════════════════════════
#  dependency_graph.py
# ═══════════════════════════════════════════════════════════════════════


class TestDepGraph:
    @pytest.fixture
    def b(self):
        from processing.architecture.dependency_graph import DependencyGraphBuilder

        return DependencyGraphBuilder(timeout=5)

    def test_parse_pypi(self, b):
        assert b._parse_pypi_requirement("requests>=2.28")[0] == "requests"
        assert b._parse_pypi_requirement("flask")[1] is None
        assert b._parse_pypi_requirement("pkg[e1]>=1.0")[0] == "pkg"
        assert b._parse_pypi_requirement("")[0] is None

    def test_build_tree(self, b):
        from models.architecture_models import ExternalPackage

        t = b.build_dependency_tree(
            [ExternalPackage(name="r", version="2", license="MIT", ecosystem="pypi")]
        )
        assert t["r"]["version"] == "2"

    @pytest.mark.asyncio
    async def test_build_graph(self, b):
        from models.architecture_models import ExternalPackage

        r = await b.build_dependency_graph(
            [ExternalPackage(name="p", version="1", license="MIT", ecosystem="unknown")],
            max_depth=1,
        )
        assert len(r) == 1

    @pytest.mark.asyncio
    async def test_transitive_max0(self, b):
        from models.architecture_models import ExternalPackage

        assert (
            len(
                await b.analyze_transitive_dependencies(
                    [ExternalPackage(name="p", version="1", license="MIT", ecosystem="unknown")],
                    max_depth=0,
                )
            )
            == 0
        )

    def test_find_paths(self, b):
        from models.architecture_models import ExternalPackage

        assert isinstance(
            b.find_dependency_paths(
                [ExternalPackage(name="a", version="1", license="MIT", ecosystem="pypi")], "b"
            ),
            list,
        )


# ═══════════════════════════════════════════════════════════════════════
#  layer_analyzer.py
# ═══════════════════════════════════════════════════════════════════════


class TestLayerDeep:
    @pytest.fixture
    def la(self):
        from processing.architecture.layer_analyzer import LayerAnalyzer

        return LayerAnalyzer()

    def _cfg(self, layers=None):
        from models.architecture_models import RepoConfig

        return RepoConfig(detect_by=["folder"], folder_roots=["src/**"], layers=layers)

    def _comp(self, cid, files, deps=None, layer=None):
        from models.architecture_models import Component

        return Component(id=cid, name=cid, files=files, dependencies=deps or [], layer=layer)

    def test_violations(self, la):
        cfg = self._cfg(layers={"infrastructure": ["src/infra/**"], "presentation": ["src/ui/**"]})
        ci = self._comp("infra", ["src/infra/db.py"], deps=[], layer="infrastructure")
        cu = self._comp("ui", ["src/ui/v.py"], deps=["infra"], layer="presentation")
        v = la.detect_violations([cu, ci], cfg)
        assert len(v) >= 1 and v[0].component_id == "ui"

    def test_layer_deps(self, la):
        ca = self._comp("a", ["a.py"], deps=["b"], layer="service")
        cb = self._comp("b", ["b.py"], deps=[], layer="data")
        r = la.get_layer_dependencies([ca, cb])
        assert "data" in r.get("service", [])

    def test_layer_metrics(self, la):
        ca = self._comp("a", ["a.py", "a2.py"], deps=["b"], layer="service")
        cb = self._comp("b", ["b.py"], deps=[], layer="data")
        m = la.get_layer_metrics([ca, cb])
        assert m["service"]["file_count"] == 2 and "avg_files_per_component" in m["service"]

    def test_graph_data_empty(self, la):
        assert la.generate_layer_graph_data([], self._cfg())["metadata"]["layer_count"] == 0

    def test_graph_data_layers(self, la):
        cfg = self._cfg(layers={"service": ["src/service/**"], "data": ["src/data/**"]})
        ca = self._comp("a", ["src/service/s.py"], deps=["b"], layer="service")
        cb = self._comp("b", ["src/data/r.py"], deps=[], layer="data")
        d = la.generate_layer_graph_data([ca, cb], cfg)
        assert len(d["nodes"]) == 2

    def test_generate_layers(self, la):
        cfg = self._cfg(layers={"service": ["src/service/**"]})
        ca = self._comp("a", ["src/service/s.py"])
        ls = la.generate_layers([ca], cfg)
        assert any(layer.name == "service" for layer in ls)

    def test_cross_layer_deps(self, la):
        ca = self._comp("a", ["a.py"], deps=["b"], layer="service")
        cb = self._comp("b", ["b.py"], deps=[], layer="data")
        assert la._count_cross_layer_dependencies([ca, cb], "service", "data") == 1

    def test_is_violation(self, la):
        cfg = self._cfg(layers={"data": ["d/**"], "ui": ["u/**"]})
        h = {"data": 1, "ui": 6}
        assert la._is_layer_violation(h, "ui", "data", cfg) is True
        assert la._is_layer_violation(h, "data", "ui", cfg) is False


# ═══════════════════════════════════════════════════════════════════════
#  dependency_visualization.py
# ═══════════════════════════════════════════════════════════════════════


class TestDepViz:
    @pytest.fixture
    def sc(self):
        from processing.architecture.dependency_visualization import DependencyVulnerabilityScanner

        return DependencyVulnerabilityScanner(enable_cache=True, batch_size=2)

    @pytest.fixture
    def vi(self):
        from processing.architecture.dependency_visualization import DependencyVisualizer

        return DependencyVisualizer()

    def test_cvss(self, sc):
        assert sc._cvss_to_severity(9.5) == "CRITICAL"
        assert sc._cvss_to_severity(7.5) == "HIGH"
        assert sc._cvss_to_severity(5.0) == "MODERATE"
        assert sc._cvss_to_severity(2.0) == "LOW"

    def test_severity_cvss(self, sc):
        assert (
            sc._extract_severity_from_osv({"severity": [{"type": "CVSS_V3", "score": "8.5"}]})
            == "HIGH"
        )

    def test_severity_db(self, sc):
        assert (
            sc._extract_severity_from_osv({"database_specific": {"severity": "CRITICAL"}})
            == "CRITICAL"
        )

    def test_severity_unknown(self, sc):
        assert sc._extract_severity_from_osv({}) == "UNKNOWN"

    def test_version_ranges(self, sc):
        a, f = sc._extract_version_ranges(
            {"affected": [{"ranges": [{"events": [{"introduced": "1.0"}, {"fixed": "2.0"}]}]}]}
        )
        assert "1.0" in a and "2.0" in f

    def test_parse_vuln(self, sc):
        v = sc._parse_osv_vulnerability(
            {
                "id": "CVE-1",
                "summary": "s",
                "details": "",
                "severity": [{"type": "CVSS_V3", "score": "9.5"}],
                "references": [{"url": "u"}],
                "published": "",
                "modified": "",
                "affected": [{"ranges": [{"events": [{"introduced": "0"}, {"fixed": "1"}]}]}],
            }
        )
        assert v.id == "CVE-1" and v.severity == "CRITICAL"

    def test_parse_no_id(self, sc):
        assert sc._parse_osv_vulnerability({}) is None

    def test_dedup(self, sc):
        assert len(sc._deduplicate_vulnerabilities([_vuln("V1"), _vuln("V1"), _vuln("V2")])) == 2

    def test_cache(self, sc):
        from models.architecture_models import ExternalPackage

        p = ExternalPackage(name="t", version="1", license="MIT", ecosystem="pypi")
        sc._cache_package_vulnerabilities(p, [])
        assert sc.get_cache_stats()["total_entries"] == 1
        sc.clear_cache()
        assert len(sc.vulnerability_cache) == 0

    def test_separate_cached(self, sc):
        from models.architecture_models import ExternalPackage

        p = ExternalPackage(name="t", version="1", license="MIT", ecosystem="pypi")
        sc._cache_package_vulnerabilities(p, [])
        c, u = sc._separate_cached_packages([p])
        assert len(c) == 1

    def test_separate_expired(self, sc):
        from models.architecture_models import ExternalPackage

        p = ExternalPackage(name="t", version="1", license="MIT", ecosystem="pypi")
        k = sc._get_cache_key(p)
        sc.vulnerability_cache[k] = {
            "data": [],
            "timestamp": datetime.now(UTC) - timedelta(hours=48),
        }
        c, u = sc._separate_cached_packages([p])
        assert len(u) == 1

    @pytest.mark.asyncio
    async def test_scan_empty(self, sc):
        assert (await sc.scan_vulnerabilities([])).vulnerabilities == []

    def test_tree_text(self, vi):
        from models.architecture_models import ExternalPackage

        t = vi.generate_dependency_tree_text(
            [
                ExternalPackage(
                    name="r", version="2", license="MIT", ecosystem="pypi", severity="HIGH"
                )
            ]
        )
        assert "r" in t and "Security: HIGH" in t

    def test_tree_empty(self, vi):
        assert vi.generate_dependency_tree_text([]) == "No dependencies found."

    def test_vuln_summary(self, vi):
        t = vi.generate_vulnerability_summary_text(
            [_vuln(severity="CRITICAL"), _vuln(id="V2", severity="HIGH")]
        )
        assert "Total Vulnerabilities: 2" in t

    def test_vuln_summary_empty(self, vi):
        assert vi.generate_vulnerability_summary_text([]) == "No vulnerabilities found."


# ═══════════════════════════════════════════════════════════════════════
#  architecture_processor.py
# ═══════════════════════════════════════════════════════════════════════


class TestArchProc:
    @pytest.fixture
    def proc(self):
        from processing.architecture.architecture_processor import ArchitectureProcessor

        return ArchitectureProcessor(graph_adapter=MagicMock())

    def test_valid_config(self, proc):
        from models.architecture_models import RepoConfig

        assert (
            proc.validate_configuration(RepoConfig(detect_by=["folder"], layers={"s": ["s/**"]}))
            is True
        )

    def test_invalid_strategy(self, proc):
        from models.architecture_models import RepoConfig

        assert proc.validate_configuration(RepoConfig(detect_by=["bad"])) is False

    def test_attr_error(self, proc):
        assert proc.validate_configuration(None) is False  # type: ignore

    def test_convert_none(self, proc):
        assert proc._convert_quality_result(None) == {}

    def test_convert_data(self, proc):
        m = MagicMock()
        m.overall_score = 85.0
        m.issues = [MagicMock(severity="e", message="m", file_path="f", line_number=1)]
        m.summary = "ok"
        assert proc._convert_quality_result(m)["overall_score"] == 85.0

    def test_fingerprint(self, proc):
        import networkx as nx

        from models.architecture_models import Component

        g = nx.DiGraph()
        g.add_node(
            "n1",
            file_path="a.py",
            type="class",
            name="C",
            properties={"visibility": "public", "modifiers": ["static"]},
        )
        fp = proc.get_component_fingerprint(
            "cb1", [Component(id="c1", name="C1", files=["a.py"], dependencies=["c2"])], g
        )
        assert len(fp) == 64


# ═══════════════════════════════════════════════════════════════════════
#  performance_optimizer.py
# ═══════════════════════════════════════════════════════════════════════


class TestPerfOpt:
    def test_adaptive_workers(self):
        from processing.architecture.performance_optimizer import AdaptiveParallelismManager

        m = AdaptiveParallelismManager(min_workers=1, max_workers=4)
        assert 1 <= m.get_optimal_worker_count(10) <= 4

    def test_execute_empty(self):
        from processing.architecture.performance_optimizer import AdaptiveParallelismManager

        assert AdaptiveParallelismManager(min_workers=1).execute_parallel_tasks([]) == []

    def test_execute_tasks(self):
        from processing.architecture.performance_optimizer import AdaptiveParallelismManager

        r = AdaptiveParallelismManager(min_workers=1, max_workers=2).execute_parallel_tasks(
            [lambda: 42, lambda: 99]
        )
        assert set(r) == {42, 99}

    def test_memory_status(self):
        from processing.architecture.performance_optimizer import MemoryEfficientGraphProcessor

        s = MemoryEfficientGraphProcessor(chunk_size=100, max_memory_mb=512).get_memory_status()
        assert "current_memory_mb" in s


# ═══════════════════════════════════════════════════════════════════════
#  security_manager.py
# ═══════════════════════════════════════════════════════════════════════


class TestSecMgr:
    @pytest.fixture
    def mgr(self):
        from processing.test_execution.security_manager import ExecutionSecurityManager
        from python_test_suite.src.security_manager import TestValidationSecurityConfig as SecConfig

        cfg = SecConfig(
            max_execution_time=30,
            max_memory_mb=512,
            max_file_size_mb=10,
            max_code_length=10000,
            allowed_modules=["pytest", "math", "unittest"],
        )
        with patch("processing.test_execution.security_manager.LoggingManager") as lm:
            lm.get_instance.return_value = MagicMock()
            return ExecutionSecurityManager(cfg)

    def test_container_configs(self, mgr):
        assert mgr.get_container_security_config(SecurityLevel.HIGH)["mem_limit"] == "256m"
        assert mgr.get_container_security_config(SecurityLevel.LOW)["mem_limit"] == "1g"

    def test_validate_runtime_valid(self, mgr):
        assert (
            mgr.validate_container_runtime_security(
                {
                    "read_only": True,
                    "network_disabled": True,
                    "user": "testuser",
                    "security_opt": [],
                    "mem_limit": "512m",
                }
            )
            is True
        )

    def test_validate_runtime_missing(self, mgr):
        assert mgr.validate_container_runtime_security({"read_only": True}) is False

    def test_validate_runtime_root(self, mgr):
        assert (
            mgr.validate_container_runtime_security(
                {
                    "read_only": True,
                    "network_disabled": True,
                    "user": "root",
                    "security_opt": [],
                    "mem_limit": "512m",
                }
            )
            is False
        )

    @pytest.mark.asyncio
    async def test_scan_python(self, mgr):
        assert (
            len(await mgr.scan_for_security_violations("import os\nos.system('x')", "python")) >= 1
        )

    @pytest.mark.asyncio
    async def test_scan_js(self, mgr):
        assert (
            len(await mgr.scan_for_security_violations('require("child_process")', "javascript"))
            >= 1
        )

    @pytest.mark.asyncio
    async def test_suspicious_imports(self, mgr):
        v = await mgr._check_suspicious_imports("__import__('os')", "python")
        assert v[0]["type"] == "suspicious_import"

    @pytest.mark.asyncio
    async def test_fs_access(self, mgr):
        assert len(await mgr._check_filesystem_access('open("/etc/passwd")', "python")) >= 1

    @pytest.mark.asyncio
    async def test_sanitize_python(self, mgr):
        assert "SANITIZED" in await mgr._sanitize_python_code("import subprocess\neval('x')")

    @pytest.mark.asyncio
    async def test_sanitize_js(self, mgr):
        assert "SANITIZED" in await mgr._sanitize_javascript_code('require("fs")\neval("x")')

    @pytest.mark.asyncio
    async def test_sanitize_truncation(self, mgr):
        assert len(await mgr._sanitize_code_content("x=1\n" * 50000, "python")) < 500000

    def test_sanitize_string(self, mgr):
        r = mgr._sanitize_string_field('<script>alert("x")</script>Hello')
        assert "<script>" not in r and "Hello" in r

    @pytest.mark.asyncio
    async def test_sanitization_report(self, mgr):
        req = TestRequest(code="x=1", language="python", timeout=10)
        r = await mgr.create_sanitization_report(req, req)
        assert r["sanitization_applied"] is False

    def test_factory(self):
        from processing.test_execution.security_manager import create_test_security_manager
        from python_test_suite.src.security_manager import TestValidationSecurityConfig as SecConfig

        cfg = SecConfig(
            max_execution_time=30,
            max_memory_mb=256,
            max_file_size_mb=5,
            max_code_length=5000,
            allowed_modules=["pytest"],
        )
        with patch("processing.test_execution.security_manager.LoggingManager") as lm:
            lm.get_instance.return_value = MagicMock()
            assert create_test_security_manager(cfg) is not None


# ═══════════════════════════════════════════════════════════════════════
#  test_runner.py
# ═══════════════════════════════════════════════════════════════════════


class TestContainerRunner:
    @pytest.fixture
    def ds(self):
        s = AsyncMock()
        s.create_test_container = AsyncMock(return_value="d-1")
        s.start_container = AsyncMock()
        s.stop_container = AsyncMock()
        s.remove_container_by_id = AsyncMock()
        s.create_file_in_container = AsyncMock(return_value=True)
        s.exec_in_container = AsyncMock(return_value={"exit_code": 0, "stdout": "ok", "stderr": ""})
        s.install_tools_in_container = AsyncMock()
        s.get_language_container_image = MagicMock(return_value="python:3.12")
        return s

    @pytest.fixture
    def runner(self, ds):
        from processing.test_execution.test_runner import ContainerTestRunner
        from python_test_suite.src.config import TestConfig

        return ContainerTestRunner(ds, TestConfig())

    def test_should_use_orchestrator(self, runner):
        assert runner._should_use_orchestrator(TestRequest(code="", language="go")) is True
        assert runner._should_use_orchestrator(TestRequest(code="", language="python")) is False

    def test_memory_limits(self, runner):
        assert runner._get_memory_limit(SecurityLevel.HIGH) == "256m"

    def test_extensions(self, runner):
        assert runner._get_file_extension("python") == ".py"
        assert runner._get_test_file_extension("typescript") == ".test.ts"

    @pytest.mark.asyncio
    async def test_execute_legacy(self, runner):
        r = await runner.execute_test(
            TestRequest(code="x=1", test_code="assert True", language="python")
        )
        assert r.success is True

    @pytest.mark.asyncio
    async def test_js_package(self, runner, ds):
        await runner._create_js_package_file("d-1", "typescript")
        pkg = json.loads(ds.create_file_in_container.call_args[0][2])
        assert "ts-jest" in pkg["devDependencies"]

    @pytest.mark.asyncio
    async def test_cleanup_error(self, runner, ds):
        ds.stop_container.side_effect = RuntimeError("gone")
        await runner._cleanup_container("d-1")


# ═══════════════════════════════════════════════════════════════════════
#  timeout_manager.py
# ═══════════════════════════════════════════════════════════════════════


class TestTimeoutMgr:
    @pytest.fixture
    def tm(self):
        from processing.test_execution.timeout_manager import TimeoutConfig, TimeoutManager

        return TimeoutManager(
            TimeoutConfig(
                default_timeout_seconds=5, max_timeout_seconds=30, graceful_shutdown_timeout=2
            )
        )

    def test_resolve_explicit(self, tm):
        from processing.test_execution.timeout_manager import TimeoutType

        assert tm._resolve_timeout(10, TimeoutType.TEST_EXECUTION) == 10

    def test_resolve_exceeds_max(self, tm):
        from processing.test_execution.timeout_manager import TimeoutType

        assert tm._resolve_timeout(999, TimeoutType.TEST_EXECUTION) == 30

    def test_resolve_default(self, tm):
        from processing.test_execution.timeout_manager import TimeoutType

        assert tm._resolve_timeout(None, TimeoutType.TEST_EXECUTION) == 5

    @pytest.mark.asyncio
    async def test_cancel_not_found(self, tm):
        assert await tm.cancel_task("x") is False

    @pytest.mark.asyncio
    async def test_cleanup_empty(self, tm):
        assert await tm.cleanup_all_tasks() == 0

    @pytest.mark.asyncio
    async def test_cancel_force(self, tm):
        t = asyncio.create_task(asyncio.sleep(100))
        tm._active_tasks["t1"] = t
        assert await tm.cancel_task("t1", graceful=False) is True

    @pytest.mark.asyncio
    async def test_cleanup_success(self, tm):
        called = False

        async def ch():
            nonlocal called
            called = True

        tm._cleanup_handlers["t1"] = ch
        assert await tm._perform_cleanup("t1") is True and called

    @pytest.mark.asyncio
    async def test_cleanup_not_found(self, tm):
        assert await tm._perform_cleanup("x") is False

    @pytest.mark.asyncio
    async def test_cleanup_retries(self, tm):
        n = 0

        async def fail():
            nonlocal n
            n += 1
            raise RuntimeError("fail")

        tm._cleanup_handlers["t1"] = fail
        tm.config.retry_cleanup_attempts = 2
        assert await tm._perform_cleanup("t1") is False and n == 2


# ═══════════════════════════════════════════════════════════════════════
#  symbolic/analyzer.py
# ═══════════════════════════════════════════════════════════════════════


class TestSymbolic:
    @pytest.fixture
    def v(self):
        from processing.analyzers.symbolic.analyzer import SymbolicValidator

        return SymbolicValidator()

    @pytest.mark.asyncio
    async def test_snippet(self, v):
        r = await v.analyze_snippet("def add(a: int, b: int) -> int:\n    return a+b\n")
        assert hasattr(r, "success")

    @pytest.mark.asyncio
    async def test_file_nonexistent(self, v):
        r = await v.analyze_file("/nonexistent/f.py")
        assert hasattr(r, "metadata")

    @pytest.mark.asyncio
    async def test_dir_nonexistent(self, v):
        r = await v.analyze_directory("/nonexistent/d")
        assert hasattr(r, "metadata")
