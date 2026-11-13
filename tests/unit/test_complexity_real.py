"""Per-entity complexity is computed from the entity's real body.

Regression for the live defect where every entity reported cyclomatic=1.0 and
lines_of_code=0 (keywords were counted in the signature, never the body).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from models.core_models import CoreCodeEntity
from processing.analyzers.complexity_analyzers import (
    ComplexityAnalysisService,
    JavaScriptComplexityAnalyzer,
    PythonComplexityAnalyzer,
)
from processing.analyzers.complexity_analyzers.body_metrics import SourceContext
from processing.languages.base import ParseResult, ParserType

PY_SOURCE = """\
import os


def branchy(a, b, items):
    if a:
        x = 1

    if b and a:
        return x
    for item in items:
        if item:
            print(item)
    return 0


def straight(a):
    total = a + 1

    return total


class Box:
    def pick(self, v):
        if v > 1:
            return 1
        elif v < 0:
            return -1
        return 0
"""

JS_SOURCE = """\
function pick(a, b) {
  const t = a ? 1 : 2;
  if (a && b) {
    return t;
  }
  return 0;
}
"""


def _entity(name: str, line: int, etype: str = "function", language: str = "python"):
    return CoreCodeEntity(
        id=f"e:{name}",
        name=name,
        type=etype,
        language=language,
        file_path=f"mod.{'py' if language == 'python' else 'js'}",
        line_number=line,
    )


def _parse_result(language: str, entities: list, file_path: str) -> ParseResult:
    return ParseResult(
        success=True,
        file_path=file_path,
        language=language,
        parser_used=ParserType.AST_NATIVE,
        entities=entities,
        lines_of_code=30,
    )


def _line_of(source: str, needle: str) -> int:
    return next(i for i, line in enumerate(source.splitlines(), 1) if needle in line)


class TestPythonBodyMetrics:
    def _metrics(self, name: str, needle: str, etype: str = "function"):
        entity = _entity(name, _line_of(PY_SOURCE, needle), etype)
        source = SourceContext("python", "mod.py", PY_SOURCE)
        pr = _parse_result("python", [entity], "mod.py")
        return PythonComplexityAnalyzer().analyze_entity_complexity(entity, pr, source)

    def test_three_ifs_one_for_one_and_is_six(self):
        m = self._metrics("branchy", "def branchy")
        assert m.computed is True
        assert m.not_computed_reason is None
        # 1 + if a + if b + `and` + for + if item
        assert m.cyclomatic_complexity == 6.0
        assert m.nesting_depth == 2
        # Sonar: if(1) + if(1) + and(1) + for(1) + nested if(1+1)
        assert m.cognitive_complexity == 6.0

    def test_straight_line_function_is_one(self):
        m = self._metrics("straight", "def straight")
        assert m.cyclomatic_complexity == 1.0
        assert m.cognitive_complexity == 0.0
        assert m.nesting_depth == 0

    def test_loc_is_non_blank_lines_of_the_span(self):
        m = self._metrics("branchy", "def branchy")
        # branchy spans 10 physical lines, one of them blank.
        assert m.lines_of_code == 9
        assert self._metrics("straight", "def straight").lines_of_code == 3

    def test_elif_counts_without_nesting(self):
        m = self._metrics("pick", "def pick", etype="function")
        assert m.cyclomatic_complexity == 3.0
        assert m.cognitive_complexity == 2.0
        assert m.nesting_depth == 1

    def test_decorated_async_function_is_not_inflated(self):
        source = "@cache\nasync def f(a):\n    if a:\n        return 1\n    return 0\n"
        entity = _entity("f", 2)
        entity.is_async = True
        entity.properties = {"decorators": ["cache"]}
        pr = _parse_result("python", [entity], "f.py")
        m = PythonComplexityAnalyzer().analyze_entity_complexity(
            entity, pr, SourceContext("python", "f.py", source)
        )
        assert m.cyclomatic_complexity == 2.0  # McCabe, no multipliers
        assert {"async", "decorators"} <= set(m.risk_factors)


class TestJavaScriptBodyMetrics:
    def test_ternary_and_logical_and(self):
        entity = _entity("pick", 1, language="javascript")
        pr = _parse_result("javascript", [entity], "mod.js")
        m = JavaScriptComplexityAnalyzer().analyze_entity_complexity(
            entity, pr, SourceContext("javascript", "mod.js", JS_SOURCE)
        )
        assert m.computed is True
        # 1 + ternary + if + &&
        assert m.cyclomatic_complexity == 4.0
        assert m.lines_of_code == 7


class TestNotComputed:
    def test_unsupported_language_is_not_computed(self):
        ctx = SourceContext("cobol", "x.cbl", "IDENTIFICATION DIVISION.\n")
        body = ctx.metrics_for(1)
        assert body.computed is False
        assert "cobol" in (body.not_computed_reason or "")

    def test_missing_source_is_not_computed_never_one(self):
        entity = _entity("branchy", 4)
        pr = _parse_result("python", [entity], "mod.py")
        m = PythonComplexityAnalyzer().analyze_entity_complexity(entity, pr)
        assert m.computed is False
        assert m.not_computed_reason == "source content not available"
        assert m.cyclomatic_complexity == 0.0
        assert m.lines_of_code == 0

    def test_entity_without_location_is_not_computed(self):
        ctx = SourceContext("python", "mod.py", PY_SOURCE)
        body = ctx.metrics_for(0)
        assert body.computed is False
        assert "line 0" in (body.not_computed_reason or "")

    @pytest.mark.parametrize("line", [2, 999])
    def test_blank_or_out_of_range_line_is_not_computed(self, line):
        assert SourceContext("python", "mod.py", PY_SOURCE).metrics_for(line).computed is False


class TestServiceThreadsContent:
    def test_report_uses_real_metrics_and_counts_not_computed(self):
        entities = [
            _entity("branchy", _line_of(PY_SOURCE, "def branchy")),
            _entity("straight", _line_of(PY_SOURCE, "def straight")),
            _entity("ghost", 0),
        ]
        pr = _parse_result("python", entities, "mod.py")
        report = ComplexityAnalysisService().analyze_file_complexity(pr, PY_SOURCE)
        by_name = {m.entity_name: m for m in report.entity_metrics}
        assert by_name["branchy"].cyclomatic_complexity == 6.0
        assert by_name["straight"].cyclomatic_complexity == 1.0
        assert by_name["ghost"].computed is False
        assert report.entities_not_computed == 1
        assert report.max_complexity == 6.0
        assert report.average_complexity == pytest.approx(3.5)  # computed entities only

    def test_endpoint_response_carries_computed_flag(self):
        from app.api.endpoints.complexity import _convert_metrics_to_response

        entity = _entity("branchy", 4)
        pr = _parse_result("python", [entity], "mod.py")
        report = ComplexityAnalysisService().analyze_file_complexity(pr, None)
        resp = _convert_metrics_to_response(report.entity_metrics[0])
        assert resp.computed is False
        assert resp.not_computed_reason == "source content not available"

    def test_mock_metrics_default_to_computed(self):
        from app.api.endpoints.complexity import _convert_metrics_to_response

        metrics = MagicMock()
        metrics.cyclomatic_complexity = 2.0
        metrics.cognitive_complexity = 1.0
        metrics.lines_of_code = 3
        metrics.nesting_depth = 1
        metrics.parameter_count = 0
        metrics.dependency_count = 0
        metrics.halstead_difficulty = 0.0
        metrics.maintainability_index = 90.0
        metrics.coupling_factor = 0.0
        metrics.complexity_level.value = "trivial"
        metrics.risk_factors = []
        metrics.suggestions = []
        metrics.language = "python"
        metrics.entity_type = "function"
        metrics.entity_name = "f"
        resp = _convert_metrics_to_response(metrics)
        assert resp.computed is True
        assert resp.not_computed_reason is None


# --- first method of a class is measured alone, not the class body -------------

FIRST_METHOD_SOURCE = """\
class A:
    def first(self):
        return 1

    def second(self, x):
        if x:
            for i in x:
                if i and x:
                    return i
        return 0
"""


class TestFirstMethodOfClass:
    def test_trivial_first_method_is_one_not_the_class_body(self):
        ctx = SourceContext("python", "a.py", FIRST_METHOD_SOURCE)
        first = ctx.metrics_for(_line_of(FIRST_METHOD_SOURCE, "def first"))
        assert first.computed is True
        assert (first.cyclomatic, first.loc) == (1, 2)

    def test_branchy_second_method_control(self):
        """Negative control: the second method still carries its own branches."""
        ctx = SourceContext("python", "a.py", FIRST_METHOD_SOURCE)
        second = ctx.metrics_for(_line_of(FIRST_METHOD_SOURCE, "def second"))
        assert (second.cyclomatic, second.loc) == (5, 6)

    def test_class_itself_still_spans_its_body(self):
        ctx = SourceContext("python", "a.py", FIRST_METHOD_SOURCE)
        cls = ctx.metrics_for(1)
        assert cls.loc == 9


# --- syntax errors inside a span are not computed ------------------------------

BROKEN_SOURCE = """\
def ok(x):
    if x:
        return 1
    return 0

def broken(x):
    if x
        return (1
    return 0
"""


class TestParseErrorInSpan:
    def test_span_with_error_is_not_computed(self):
        ctx = SourceContext("python", "b.py", BROKEN_SOURCE)
        body = ctx.metrics_for(_line_of(BROKEN_SOURCE, "def broken"))
        assert body.computed is False
        assert body.not_computed_reason == "parse error in span"

    def test_clean_span_in_same_file_still_computes(self):
        """Negative control: an entity without errors in its span is measured."""
        ctx = SourceContext("python", "b.py", BROKEN_SOURCE)
        body = ctx.metrics_for(_line_of(BROKEN_SOURCE, "def ok"))
        assert body.computed is True
        assert body.cyclomatic == 2


# --- zero computed entities never yields a clean quality score -----------------


class TestNoComputedEntitiesHasNoScore:
    def test_quality_score_is_none_when_nothing_computed(self):
        entities = [_entity("ghost", 0), _entity("phantom", 999)]
        pr = _parse_result("python", entities, "mod.py")
        report = ComplexityAnalysisService().analyze_file_complexity(pr, PY_SOURCE)
        assert report.entities_not_computed == 2
        assert report.quality_score is None
        assert report.metrics_computed is False

        from app.api.endpoints.complexity import _create_file_complexity_response

        resp = _create_file_complexity_response(report, [])
        assert resp.quality_score is None
        assert resp.metrics_computed is False

    def test_one_computed_entity_keeps_a_score(self):
        """Negative control."""
        entities = [_entity("straight", _line_of(PY_SOURCE, "def straight")), _entity("g", 0)]
        pr = _parse_result("python", entities, "mod.py")
        report = ComplexityAnalysisService().analyze_file_complexity(pr, PY_SOURCE)
        assert report.metrics_computed is True
        assert isinstance(report.quality_score, float)

    def test_batch_summary_skips_unscored_files(self):
        from app.api.endpoints.complexity import (
            _create_file_complexity_response,
            _initialize_batch_summary,
            _update_batch_statistics,
        )

        pr = _parse_result("python", [_entity("ghost", 0)], "mod.py")
        report = ComplexityAnalysisService().analyze_file_complexity(pr, PY_SOURCE)
        summary = _initialize_batch_summary(1)
        _update_batch_statistics(_create_file_complexity_response(report, []), 70.0, summary)
        assert summary["not_computed_files"] == 1
        assert summary["scored_files"] == 0
        assert summary["below_threshold"] == 0
        assert sum(summary["quality_distribution"].values()) == 0
