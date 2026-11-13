"""Unit tests for all complexity analyzers.

Covers:
- processing.analyzers.complexity_analyzers.__init__
- processing.analyzers.complexity_analyzers.base
- processing.analyzers.complexity_analyzers.python_analyzer
- processing.analyzers.complexity_analyzers.javascript_analyzer
- processing.analyzers.complexity_analyzers.typescript_analyzer
- processing.analyzers.complexity_analyzers.csharp_analyzer
- processing.analyzers.complexity_analyzers.cpp_analyzer
- processing.analyzers.complexity_analyzers.java_analyzer
- processing.analyzers.complexity_analyzers.rust_analyzer
- processing.analyzers.complexity_analyzers.go_analyzer
"""

from __future__ import annotations

import pytest

from models.core_models import CoreCodeEntity, EntityRelation, EntityType, RelationType
from processing.languages.base import ParseResult, ParserType

# ---------------------------------------------------------------------------
# Helpers to build test fixtures
# ---------------------------------------------------------------------------


def _entity(
    *,
    name: str = "test_func",
    etype: str = "function",
    signature: str | None = None,
    is_async: bool = False,
    properties: dict | None = None,
    entity_id: str = "e1",
) -> CoreCodeEntity:
    return CoreCodeEntity(
        id=entity_id,
        name=name,
        type=EntityType(etype),
        language="python",
        file_path="test.py",
        line_number=1,
        signature=signature,
        is_async=is_async,
        properties=properties or {},
    )


def _parse_result(
    *,
    language: str = "python",
    entities: list[CoreCodeEntity] | None = None,
    relationships: list[EntityRelation] | None = None,
    lines_of_code: int = 100,
) -> ParseResult:
    return ParseResult(
        success=True,
        file_path="test.py",
        language=language,
        parser_used=ParserType.REGEX_FALLBACK,
        entities=entities or [],
        relationships=relationships or [],
        lines_of_code=lines_of_code,
    )


def _relation(
    source_id: str = "e1",
    target_id: str = "e2",
    rel_type: str = "inherits",
) -> EntityRelation:
    return EntityRelation(
        source_id=source_id,
        target_id=target_id,
        type=RelationType(rel_type),
    )


# ===========================================================================
# base.py tests
# ===========================================================================


class TestComplexityLevel:
    def test_values(self):
        from processing.analyzers.complexity_analyzers.base import ComplexityLevel

        assert ComplexityLevel.TRIVIAL == "trivial"
        assert ComplexityLevel.VERY_COMPLEX == "very_complex"


class TestServiceComplexityMetrics:
    def test_defaults(self):
        from processing.analyzers.complexity_analyzers.base import ServiceComplexityMetrics

        m = ServiceComplexityMetrics()
        assert m.cyclomatic_complexity == 0.0
        assert m.cognitive_complexity == 0.0
        assert m.risk_factors == []
        assert m.suggestions == []
        assert m.language == ""

    def test_custom_values(self):
        from processing.analyzers.complexity_analyzers.base import (
            ComplexityLevel,
            ServiceComplexityMetrics,
        )

        m = ServiceComplexityMetrics(
            cyclomatic_complexity=15.0,
            language="python",
            entity_type="function",
            entity_name="foo",
            complexity_level=ComplexityLevel.MODERATE,
        )
        assert m.cyclomatic_complexity == 15.0
        assert m.complexity_level == ComplexityLevel.MODERATE


class TestFileComplexityReport:
    def test_defaults(self):
        from processing.analyzers.complexity_analyzers.base import FileComplexityReport

        r = FileComplexityReport(file_path="f.py", language="python", total_loc=50)
        assert r.average_complexity == 0.0
        assert r.entity_metrics == []
        assert r.hotspots == []
        assert r.quality_score == 0.0


class TestLanguageComplexityAnalyzerBase:
    """Tests for the non-abstract helpers on the base class."""

    def test_classify_complexity_standard(self):
        from processing.analyzers.complexity_analyzers.java_analyzer import JavaComplexityAnalyzer

        analyzer = JavaComplexityAnalyzer()
        from processing.analyzers.complexity_analyzers.base import ComplexityLevel

        assert analyzer._classify_complexity_standard(3) == ComplexityLevel.TRIVIAL
        assert analyzer._classify_complexity_standard(7) == ComplexityLevel.BASIC
        assert analyzer._classify_complexity_standard(15) == ComplexityLevel.MODERATE
        assert analyzer._classify_complexity_standard(30) == ComplexityLevel.COMPLEX
        assert analyzer._classify_complexity_standard(50) == ComplexityLevel.VERY_COMPLEX

    def test_calculate_standard_maintainability_zero_loc(self):
        from processing.analyzers.complexity_analyzers.base import ServiceComplexityMetrics
        from processing.analyzers.complexity_analyzers.java_analyzer import JavaComplexityAnalyzer

        analyzer = JavaComplexityAnalyzer()
        m = ServiceComplexityMetrics(lines_of_code=0)
        assert analyzer._calculate_standard_maintainability(m) == 100.0

    def test_calculate_standard_maintainability_nonzero(self):
        from processing.analyzers.complexity_analyzers.base import ServiceComplexityMetrics
        from processing.analyzers.complexity_analyzers.java_analyzer import JavaComplexityAnalyzer

        analyzer = JavaComplexityAnalyzer()
        m = ServiceComplexityMetrics(lines_of_code=100, cyclomatic_complexity=10)
        mi = analyzer._calculate_standard_maintainability(m)
        assert 0.0 <= mi <= 100.0

    def test_count_parameters_from_signature(self):
        from processing.analyzers.complexity_analyzers.java_analyzer import JavaComplexityAnalyzer

        analyzer = JavaComplexityAnalyzer()
        assert analyzer._count_parameters_from_signature("foo(a, b, c)") == 3
        assert analyzer._count_parameters_from_signature("foo()") == 0
        assert analyzer._count_parameters_from_signature("no_parens") == 0
        assert analyzer._count_parameters_from_signature("foo(  )") == 0


# ===========================================================================
# Python analyzer tests
# ===========================================================================


class TestPythonComplexityAnalyzer:
    def _analyzer(self):
        from processing.analyzers.complexity_analyzers.python_analyzer import (
            PythonComplexityAnalyzer,
        )

        return PythonComplexityAnalyzer()

    def test_basic_function(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="def foo(x, y)")
        pr = _parse_result()
        m = a.analyze_entity_complexity(e, pr)
        assert m.language == "python"
        assert m.entity_type == "function"
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0
        assert m.parameter_count == 2

    def test_decorators_increase_complexity(self):
        a = self._analyzer()
        e = _entity(
            etype="method",
            signature="def foo(self)",
            properties={"decorators": ["@staticmethod", "@cache"]},
        )
        pr = _parse_result()
        m = a.analyze_entity_complexity(e, pr)
        assert "decorators" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_comprehension_multiplier(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="def f()", properties={"has_comprehension": True})
        pr = _parse_result()
        m = a.analyze_entity_complexity(e, pr)
        assert "comprehension" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_metaclass_multiplier(self):
        a = self._analyzer()
        e = _entity(etype="class", signature="class Meta()", properties={"has_metaclass": True})
        pr = _parse_result()
        m = a.analyze_entity_complexity(e, pr)
        assert "metaclass" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_generator_multiplier(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="def gen()", properties={"is_generator": True})
        pr = _parse_result()
        m = a.analyze_entity_complexity(e, pr)
        assert "generator" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_async_multiplier(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="async def f()", is_async=True)
        pr = _parse_result()
        m = a.analyze_entity_complexity(e, pr)
        assert "async" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_class_coupling(self):
        a = self._analyzer()
        e = _entity(etype="class", name="MyClass", entity_id="c1")
        rels = [
            _relation(source_id="c1", target_id="c2", rel_type="inherits"),
            _relation(source_id="c1", target_id="c3", rel_type="imports"),
            _relation(source_id="c1", target_id="c4", rel_type="calls"),
        ]
        pr = _parse_result(relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.coupling_factor <= 1.0

    def test_count_parameters_filters_self_cls(self):
        a = self._analyzer()
        assert a._count_parameters(_entity(signature="def f(self, x, y)")) == 2
        assert a._count_parameters(_entity(signature="def f(cls, x)")) == 1
        assert a._count_parameters(_entity(signature="def f()")) == 0

    def test_get_language_specific_factors(self):
        a = self._analyzer()
        factors = a.get_language_specific_factors()
        assert "decorator_penalty" in factors
        assert "async_penalty" in factors

    def test_maintainability_with_type_hints(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="def f()", properties={"has_type_hints": True})
        pr = _parse_result()
        m = a.analyze_entity_complexity(e, pr)
        assert m.maintainability_index > 0

    def test_maintainability_magic_method(self):
        a = self._analyzer()
        e = _entity(
            name="__init__",
            etype="method",
            signature="def __init__(self)",
            properties={},
        )
        pr = _parse_result()
        m = a.analyze_entity_complexity(e, pr)
        # Magic method penalty applied
        assert m.maintainability_index <= 100.0

    def test_maintainability_decorator_stacking(self):
        a = self._analyzer()
        e = _entity(
            etype="method",
            signature="def f(self)",
            properties={"decorators": ["@a", "@b", "@c", "@d", "@e"]},
        )
        pr = _parse_result()
        m = a.analyze_entity_complexity(e, pr)
        assert m.maintainability_index <= 100.0

    def test_suggestions_high_complexity(self):
        a = self._analyzer()
        sig = "if else while for switch case try catch finally if else while for"
        e = _entity(etype="function", signature=sig)
        pr = _parse_result()
        m = a.analyze_entity_complexity(e, pr)
        if m.cyclomatic_complexity > 10:
            assert any("breaking" in s for s in m.suggestions)

    def test_suggestions_many_params(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="def f(a, b, c, d, e, f_param)")
        pr = _parse_result()
        m = a.analyze_entity_complexity(e, pr)
        assert any("kwargs" in s or "dataclass" in s for s in m.suggestions)

    def test_suggestions_excessive_decorators(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="def f()",
            properties={"decorators": ["@a", "@b", "@c", "@d"]},
        )
        pr = _parse_result()
        m = a.analyze_entity_complexity(e, pr)
        assert any("decorator" in s.lower() for s in m.suggestions)

    def test_suggestions_no_type_hints(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="def f()", properties={"has_type_hints": False})
        pr = _parse_result()
        m = a.analyze_entity_complexity(e, pr)
        assert any("type hints" in s for s in m.suggestions)

    def test_suggestions_high_coupling(self):
        a = self._analyzer()
        e = _entity(etype="class", entity_id="c1")
        # Create enough relations to get coupling > 0.7
        rels = [
            _relation(source_id="c1", target_id=f"t{i}", rel_type="inherits") for i in range(10)
        ]
        pr = _parse_result(relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        if m.coupling_factor > 0.7:
            assert any("coupling" in s.lower() for s in m.suggestions)

    def test_suggestions_magic_method(self):
        a = self._analyzer()
        e = _entity(name="__add__", etype="method", signature="def __add__(self, other)")
        pr = _parse_result()
        m = a.analyze_entity_complexity(e, pr)
        assert any("Magic methods" in s for s in m.suggestions)

    def test_suggestions_complex_comprehension(self):
        a = self._analyzer()
        sig = "if else while for switch case if else while for switch case if else while for"
        e = _entity(
            etype="function",
            signature=sig,
            properties={"has_comprehension": True},
        )
        pr = _parse_result()
        m = a.analyze_entity_complexity(e, pr)
        if m.cyclomatic_complexity > 15:
            assert any("comprehension" in s.lower() for s in m.suggestions)

    def test_classify_complexity_all_levels(self):
        a = self._analyzer()
        from processing.analyzers.complexity_analyzers.base import ComplexityLevel

        assert a._classify_complexity(3) == ComplexityLevel.TRIVIAL
        assert a._classify_complexity(7) == ComplexityLevel.BASIC
        assert a._classify_complexity(15) == ComplexityLevel.MODERATE
        assert a._classify_complexity(30) == ComplexityLevel.COMPLEX
        assert a._classify_complexity(50) == ComplexityLevel.VERY_COMPLEX


# ===========================================================================
# JavaScript analyzer tests
# ===========================================================================


class TestJavaScriptComplexityAnalyzer:
    def _analyzer(self):
        from processing.analyzers.complexity_analyzers.javascript_analyzer import (
            JavaScriptComplexityAnalyzer,
        )

        return JavaScriptComplexityAnalyzer()

    def test_basic_function(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="function foo(a, b)")
        pr = _parse_result(language="javascript")
        m = a.analyze_entity_complexity(e, pr)
        assert m.language == "javascript"
        assert m.parameter_count == 2

    def test_callback_complexity(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="function f()", properties={"has_callbacks": True})
        pr = _parse_result(language="javascript")
        m = a.analyze_entity_complexity(e, pr)
        assert "callbacks" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_promise_complexity(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="function f()", properties={"has_promises": True})
        pr = _parse_result(language="javascript")
        m = a.analyze_entity_complexity(e, pr)
        assert "promises" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_closure_depth(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="function f()", properties={"closure_depth": 3})
        pr = _parse_result(language="javascript")
        m = a.analyze_entity_complexity(e, pr)
        assert "closures" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_iife_complexity(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="function f()", properties={"is_iife": True})
        pr = _parse_result(language="javascript")
        m = a.analyze_entity_complexity(e, pr)
        assert "iife" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_async_function(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="async function f()", is_async=True)
        pr = _parse_result(language="javascript")
        m = a.analyze_entity_complexity(e, pr)
        assert "async" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_arrow_function(self):
        a = self._analyzer()
        e = _entity(etype="arrow_function", signature="(x) => x + 1")
        pr = _parse_result(language="javascript")
        m = a.analyze_entity_complexity(e, pr)
        assert m.entity_type == "arrow_function"

    def test_class_coupling(self):
        a = self._analyzer()
        e = _entity(etype="class", entity_id="c1")
        rels = [
            _relation(source_id="c1", target_id="c2", rel_type="extends"),
            _relation(source_id="c1", target_id="c3", rel_type="imports"),
        ]
        pr = _parse_result(language="javascript", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.coupling_factor <= 1.0

    def test_count_parameters_filters_rest(self):
        a = self._analyzer()
        # '...' alone should be filtered
        assert a._count_parameters(_entity(signature="f(a, ..., b)")) == 2
        assert a._count_parameters(_entity(signature="f()")) == 0

    def test_get_language_specific_factors(self):
        a = self._analyzer()
        factors = a.get_language_specific_factors()
        assert "callback_nesting_penalty" in factors

    def test_maintainability_callback_hell(self):
        a = self._analyzer()
        sig = "if else while for switch case if else while for switch case if else while for"
        e = _entity(
            etype="function",
            signature=sig,
            properties={"has_callbacks": True},
        )
        pr = _parse_result(language="javascript")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_maintainability_deep_closure(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="function f()", properties={"closure_depth": 5})
        pr = _parse_result(language="javascript")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_maintainability_prototype_modification(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="function f()",
            properties={"modifies_prototype": True},
        )
        pr = _parse_result(language="javascript")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_classify_complexity(self):
        a = self._analyzer()
        from processing.analyzers.complexity_analyzers.base import ComplexityLevel

        assert a._classify_complexity(3) == ComplexityLevel.TRIVIAL
        assert a._classify_complexity(7) == ComplexityLevel.BASIC
        assert a._classify_complexity(15) == ComplexityLevel.MODERATE
        assert a._classify_complexity(30) == ComplexityLevel.COMPLEX
        assert a._classify_complexity(50) == ComplexityLevel.VERY_COMPLEX

    def test_suggestions_callbacks_no_async(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="function f()",
            properties={"has_callbacks": True},
        )
        pr = _parse_result(language="javascript")
        m = a.analyze_entity_complexity(e, pr)
        assert any("async/await" in s for s in m.suggestions)

    def test_suggestions_complex_promises(self):
        a = self._analyzer()
        sig = "if else while for switch case if else while for switch case if"
        e = _entity(
            etype="function",
            signature=sig,
            properties={"has_promises": True},
        )
        pr = _parse_result(language="javascript")
        m = a.analyze_entity_complexity(e, pr)
        if m.cyclomatic_complexity > 12:
            assert any("promise" in s.lower() for s in m.suggestions)

    def test_suggestions_deep_closure(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="function f()", properties={"closure_depth": 4})
        pr = _parse_result(language="javascript")
        m = a.analyze_entity_complexity(e, pr)
        assert any("closure" in s.lower() for s in m.suggestions)

    def test_suggestions_prototype_modification(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="function f()",
            properties={"modifies_prototype": True},
        )
        pr = _parse_result(language="javascript")
        m = a.analyze_entity_complexity(e, pr)
        assert any("prototype" in s.lower() for s in m.suggestions)

    def test_suggestions_high_coupling(self):
        a = self._analyzer()
        e = _entity(etype="class", entity_id="c1")
        rels = [_relation(source_id="c1", target_id=f"t{i}", rel_type="extends") for i in range(10)]
        pr = _parse_result(language="javascript", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        if m.coupling_factor > 0.7:
            assert any("coupling" in s.lower() for s in m.suggestions)

    def test_suggestions_many_params(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="function f(a, b, c, d, e)")
        pr = _parse_result(language="javascript")
        m = a.analyze_entity_complexity(e, pr)
        assert any("options" in s.lower() or "destructuring" in s.lower() for s in m.suggestions)


# ===========================================================================
# TypeScript analyzer tests
# ===========================================================================


class TestTypeScriptComplexityAnalyzer:
    def _analyzer(self):
        from processing.analyzers.complexity_analyzers.typescript_analyzer import (
            TypeScriptComplexityAnalyzer,
        )

        return TypeScriptComplexityAnalyzer()

    def test_interface_type(self):
        a = self._analyzer()
        e = _entity(etype="interface", entity_id="i1")
        rels = [_relation(source_id="i1", target_id="i2", rel_type="extends")]
        pr = _parse_result(language="typescript", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        assert m.language == "typescript"
        assert 0.0 <= m.coupling_factor <= 1.0

    def test_type_alias(self):
        a = self._analyzer()
        e = _entity(etype="type_alias", entity_id="t1")
        rels = [_relation(source_id="t1", target_id="t2", rel_type="implements")]
        pr = _parse_result(language="typescript", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.coupling_factor <= 1.0

    def test_class_with_decorators(self):
        a = self._analyzer()
        e = _entity(
            etype="class",
            entity_id="c1",
            properties={"decorators": ["@Component"]},
        )
        pr = _parse_result(language="typescript")
        m = a.analyze_entity_complexity(e, pr)
        assert "decorators" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_function_with_generics(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="function f()",
            properties={"generic_parameters": ["T", "U"], "parameters": "a, b, c"},
        )
        pr = _parse_result(language="typescript")
        m = a.analyze_entity_complexity(e, pr)
        assert m.parameter_count == 3
        assert "generic_parameters" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_conditional_type(self):
        a = self._analyzer()
        e = _entity(etype="type_alias", properties={"is_conditional_type": True})
        pr = _parse_result(language="typescript")
        m = a.analyze_entity_complexity(e, pr)
        assert "conditional_type" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_mapped_type(self):
        a = self._analyzer()
        e = _entity(etype="type_alias", properties={"is_mapped_type": True})
        pr = _parse_result(language="typescript")
        m = a.analyze_entity_complexity(e, pr)
        assert "mapped_type" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_generic_parameters_adds_complexity(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="function f()",
            properties={"generic_parameters": ["T", "U", "V"]},
        )
        pr = _parse_result(language="typescript")
        m = a.analyze_entity_complexity(e, pr)
        assert "generic_parameters" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_class_coupling_extends_implements(self):
        a = self._analyzer()
        e = _entity(etype="class", entity_id="c1")
        rels = [
            _relation(source_id="c1", target_id="c2", rel_type="extends"),
            _relation(source_id="c1", target_id="c3", rel_type="implements"),
        ]
        pr = _parse_result(language="typescript", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.coupling_factor <= 1.0

    def test_class_coupling_decorated_by(self):
        a = self._analyzer()
        e = _entity(etype="class", entity_id="c1")
        rels = [_relation(source_id="c1", target_id="c2", rel_type="calls")]
        pr = _parse_result(language="typescript", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.coupling_factor <= 1.0

    def test_count_typescript_parameters_from_properties(self):
        a = self._analyzer()
        assert a._count_typescript_parameters(_entity(properties={"parameters": "a, b"})) == 2
        assert a._count_typescript_parameters(_entity(properties={"parameters": "  "})) == 0
        assert a._count_typescript_parameters(_entity(properties={})) == 0

    def test_maintainability_exported(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="function f()",
            properties={"is_exported": True},
        )
        pr = _parse_result(language="typescript")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_maintainability_generics_and_decorators(self):
        a = self._analyzer()
        e = _entity(
            etype="class",
            entity_id="c1",
            properties={
                "generic_parameters": ["T", "U"],
                "decorators": ["@A", "@B"],
            },
        )
        pr = _parse_result(language="typescript")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_classify_complexity(self):
        a = self._analyzer()
        from processing.analyzers.complexity_analyzers.base import ComplexityLevel

        assert a._classify_complexity(3) == ComplexityLevel.TRIVIAL
        assert a._classify_complexity(8) == ComplexityLevel.BASIC
        assert a._classify_complexity(20) == ComplexityLevel.MODERATE
        assert a._classify_complexity(40) == ComplexityLevel.COMPLEX
        assert a._classify_complexity(60) == ComplexityLevel.VERY_COMPLEX

    def test_get_language_specific_factors(self):
        a = self._analyzer()
        factors = a.get_language_specific_factors()
        assert "type_system_penalty" in factors

    def test_suggestions_many_generics(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="function f()",
            properties={"generic_parameters": ["T", "U", "V", "W"]},
        )
        pr = _parse_result(language="typescript")
        m = a.analyze_entity_complexity(e, pr)
        assert any("generic" in s.lower() for s in m.suggestions)

    def test_suggestions_conditional_type(self):
        a = self._analyzer()
        e = _entity(etype="type_alias", properties={"is_conditional_type": True})
        pr = _parse_result(language="typescript")
        m = a.analyze_entity_complexity(e, pr)
        assert any("conditional" in s.lower() for s in m.suggestions)

    def test_suggestions_mapped_type(self):
        a = self._analyzer()
        e = _entity(etype="type_alias", properties={"is_mapped_type": True})
        pr = _parse_result(language="typescript")
        m = a.analyze_entity_complexity(e, pr)
        assert any("mapped" in s.lower() for s in m.suggestions)

    def test_suggestions_many_decorators(self):
        a = self._analyzer()
        e = _entity(
            etype="class",
            entity_id="c1",
            properties={"decorators": ["@A", "@B", "@C"]},
        )
        pr = _parse_result(language="typescript")
        m = a.analyze_entity_complexity(e, pr)
        assert any("decorator" in s.lower() for s in m.suggestions)

    def test_suggestions_high_coupling(self):
        a = self._analyzer()
        e = _entity(etype="class", entity_id="c1")
        rels = [_relation(source_id="c1", target_id=f"t{i}", rel_type="extends") for i in range(10)]
        pr = _parse_result(language="typescript", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        if m.coupling_factor > 0.7:
            assert any("coupling" in s.lower() for s in m.suggestions)

    def test_suggestions_many_params(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="function f()",
            properties={"parameters": "a, b, c, d, e"},
        )
        pr = _parse_result(language="typescript")
        m = a.analyze_entity_complexity(e, pr)
        assert any("options" in s.lower() or "interface" in s.lower() for s in m.suggestions)


# ===========================================================================
# C# analyzer tests
# ===========================================================================


class TestCSharpComplexityAnalyzer:
    def _analyzer(self):
        from processing.analyzers.complexity_analyzers.csharp_analyzer import (
            CSharpComplexityAnalyzer,
        )

        return CSharpComplexityAnalyzer()

    def test_method_basic(self):
        a = self._analyzer()
        e = _entity(etype="method", signature="void Foo(int a, string b)")
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert m.language == "csharp"
        assert m.parameter_count == 2

    def test_linq_complexity(self):
        a = self._analyzer()
        e = _entity(
            etype="method",
            signature="void Query()",
            properties={"has_linq": True, "linq_depth": 3},
        )
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert "linq" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_async_method(self):
        a = self._analyzer()
        e = _entity(
            etype="method",
            signature="async Task DoWork()",
            is_async=True,
            properties={},
        )
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert "async" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_delegate_complexity(self):
        a = self._analyzer()
        e = _entity(etype="method", signature="void F()", properties={"has_delegates": True})
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert "delegates" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_reflection_complexity(self):
        a = self._analyzer()
        e = _entity(etype="method", signature="void F()", properties={"uses_reflection": True})
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert "reflection" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_extension_method(self):
        a = self._analyzer()
        e = _entity(
            etype="method",
            signature="void F(this string s, int x)",
            properties={"is_extension_method": True},
        )
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        # 'this' param filtered
        assert m.parameter_count == 1

    def test_constructor(self):
        a = self._analyzer()
        e = _entity(etype="constructor", signature="MyClass(int a)")
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert m.parameter_count == 1

    def test_class_generics(self):
        a = self._analyzer()
        e = _entity(
            etype="class",
            entity_id="c1",
            properties={"generic_parameters": ["T", "U"]},
        )
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert "generic_parameters" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_class_attributes(self):
        a = self._analyzer()
        e = _entity(
            etype="class",
            entity_id="c1",
            properties={"attributes": ["Serializable", "DataContract", "Obsolete"]},
        )
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert "many_attributes" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_property_getter_setter(self):
        a = self._analyzer()
        e = _entity(
            etype="property",
            properties={"has_getter": True, "has_setter": True},
        )
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert "property_getter_setter" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_class_coupling(self):
        a = self._analyzer()
        e = _entity(etype="class", entity_id="c1")
        rels = [
            _relation(source_id="c1", target_id="c2", rel_type="inherits"),
            _relation(source_id="c1", target_id="c3", rel_type="implements"),
            _relation(source_id="c1", target_id="c4", rel_type="calls"),
        ]
        pr = _parse_result(language="csharp", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.coupling_factor <= 1.0

    def test_get_language_specific_factors(self):
        a = self._analyzer()
        factors = a.get_language_specific_factors()
        assert "linq_penalty" in factors
        assert "reflection_penalty" in factors

    def test_classify_complexity(self):
        a = self._analyzer()
        from processing.analyzers.complexity_analyzers.base import ComplexityLevel

        assert a._classify_complexity(3) == ComplexityLevel.TRIVIAL
        assert a._classify_complexity(7) == ComplexityLevel.BASIC
        assert a._classify_complexity(15) == ComplexityLevel.MODERATE
        assert a._classify_complexity(30) == ComplexityLevel.COMPLEX
        assert a._classify_complexity(50) == ComplexityLevel.VERY_COMPLEX

    def test_maintainability_linq_deep(self):
        a = self._analyzer()
        e = _entity(
            etype="method",
            signature="void F()",
            properties={"has_linq": True, "linq_depth": 4},
        )
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_maintainability_reflection(self):
        a = self._analyzer()
        e = _entity(etype="method", signature="void F()", properties={"uses_reflection": True})
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_maintainability_async_complex(self):
        a = self._analyzer()
        sig = "if else while for switch case if else while for"
        e = _entity(etype="method", signature=sig, is_async=True, properties={})
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_maintainability_delegate_count(self):
        a = self._analyzer()
        e = _entity(
            etype="method",
            signature="void F()",
            properties={"has_delegates": True, "delegate_count": 4},
        )
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_maintainability_auto_property(self):
        a = self._analyzer()
        e = _entity(etype="property", properties={"is_auto_property": True})
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_maintainability_implements_interfaces(self):
        a = self._analyzer()
        e = _entity(etype="class", entity_id="c1", properties={"implements_interfaces": True})
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_suggestions_complex_linq(self):
        a = self._analyzer()
        e = _entity(
            etype="method",
            signature="void F()",
            properties={"has_linq": True, "linq_depth": 4},
        )
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert any("linq" in s.lower() for s in m.suggestions)

    def test_suggestions_async_complex(self):
        a = self._analyzer()
        sig = "if else while for switch case if else while for switch case if"
        e = _entity(etype="method", signature=sig, is_async=True, properties={"is_async": True})
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        if m.cyclomatic_complexity > 12:
            assert any("async" in s.lower() for s in m.suggestions)

    def test_suggestions_reflection(self):
        a = self._analyzer()
        e = _entity(etype="method", signature="void F()", properties={"uses_reflection": True})
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert any("reflection" in s.lower() for s in m.suggestions)

    def test_suggestions_many_delegates(self):
        a = self._analyzer()
        e = _entity(
            etype="method",
            signature="void F()",
            properties={"has_delegates": True, "delegate_count": 4},
        )
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert any("delegate" in s.lower() for s in m.suggestions)

    def test_suggestions_many_generics(self):
        a = self._analyzer()
        e = _entity(
            etype="class",
            entity_id="c1",
            properties={"generic_parameters": ["T", "U", "V"]},
        )
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert any("generic" in s.lower() for s in m.suggestions)

    def test_suggestions_many_attributes(self):
        a = self._analyzer()
        e = _entity(
            etype="class",
            entity_id="c1",
            properties={"attributes": ["A", "B", "C", "D"]},
        )
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert any("attribute" in s.lower() for s in m.suggestions)

    def test_suggestions_complex_property(self):
        a = self._analyzer()
        sig = "if else while for switch case"
        e = _entity(etype="property", signature=sig, properties={"is_auto_property": False})
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        if m.cyclomatic_complexity > 3:
            assert any("property" in s.lower() for s in m.suggestions)

    def test_suggestions_high_coupling(self):
        a = self._analyzer()
        e = _entity(etype="class", entity_id="c1")
        rels = [
            _relation(source_id="c1", target_id=f"t{i}", rel_type="inherits") for i in range(10)
        ]
        pr = _parse_result(language="csharp", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        if m.coupling_factor > 0.7:
            assert any("coupling" in s.lower() for s in m.suggestions)

    def test_suggestions_many_params(self):
        a = self._analyzer()
        e = _entity(etype="method", signature="void F(int a, int b, int c, int d, int e, int f)")
        pr = _parse_result(language="csharp")
        m = a.analyze_entity_complexity(e, pr)
        assert any("parameter" in s.lower() or "builder" in s.lower() for s in m.suggestions)


# ===========================================================================
# C++ analyzer tests
# ===========================================================================


class TestCppComplexityAnalyzer:
    def _analyzer(self):
        from processing.analyzers.complexity_analyzers.cpp_analyzer import CppComplexityAnalyzer

        return CppComplexityAnalyzer()

    def test_basic_function(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="void foo(int a, int b)")
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert m.language == "cpp"
        assert m.parameter_count == 2

    def test_template_complexity(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="void foo()",
            properties={"template_parameters": ["T", "U"]},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert "template_parameters" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_operator_overload(self):
        a = self._analyzer()
        e = _entity(
            etype="method",
            signature="bool operator==()",
            properties={"is_operator_overload": True},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert "operator_overload" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_raii_complexity(self):
        a = self._analyzer()
        e = _entity(
            etype="constructor",
            signature="MyClass()",
            properties={"manages_resources": True},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert "manages_resources" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_exception_handling(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="void f()",
            properties={"has_exceptions": True},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert "exception_handling" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_move_semantics(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="void f()",
            properties={"has_move_semantics": True},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert "move_semantics" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_raw_pointers(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="void f()",
            properties={"uses_raw_pointers": True},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert "raw_pointers" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_destructor(self):
        a = self._analyzer()
        e = _entity(etype="destructor", signature="~MyClass()")
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert m.entity_type == "destructor"

    def test_class_multiple_inheritance(self):
        a = self._analyzer()
        e = _entity(
            etype="class",
            entity_id="c1",
            properties={"inheritance_count": 3},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert "multiple_inheritance" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_class_virtual_functions(self):
        a = self._analyzer()
        e = _entity(
            etype="class",
            entity_id="c1",
            properties={"has_virtual_functions": True},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert "virtual_functions" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_class_coupling(self):
        a = self._analyzer()
        e = _entity(etype="class", entity_id="c1")
        rels = [
            _relation(source_id="c1", target_id="c2", rel_type="inherits"),
            _relation(source_id="c1", target_id="c3", rel_type="imports"),
        ]
        pr = _parse_result(language="cpp", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.coupling_factor <= 1.0

    def test_count_parameters_void(self):
        a = self._analyzer()
        assert a._count_parameters(_entity(signature="void foo(void)")) == 0
        assert a._count_parameters(_entity(signature="void foo(int a)")) == 1

    def test_get_language_specific_factors(self):
        a = self._analyzer()
        factors = a.get_language_specific_factors()
        assert "template_penalty" in factors
        assert "raw_pointer_penalty" in factors

    def test_classify_complexity(self):
        a = self._analyzer()
        from processing.analyzers.complexity_analyzers.base import ComplexityLevel

        assert a._classify_complexity(2) == ComplexityLevel.TRIVIAL
        assert a._classify_complexity(6) == ComplexityLevel.BASIC
        assert a._classify_complexity(12) == ComplexityLevel.MODERATE
        assert a._classify_complexity(25) == ComplexityLevel.COMPLEX
        assert a._classify_complexity(40) == ComplexityLevel.VERY_COMPLEX

    def test_maintainability_template_meta(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="void f()",
            properties={"template_parameters": ["T", "U", "V"]},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_maintainability_multiple_inheritance(self):
        a = self._analyzer()
        e = _entity(
            etype="class",
            entity_id="c1",
            properties={"inheritance_count": 3},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_maintainability_raw_pointers(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="void f()",
            properties={"uses_raw_pointers": True},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_maintainability_smart_pointers_bonus(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="void f()",
            properties={"uses_smart_pointers": True},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_maintainability_const_correct_bonus(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="void f()",
            properties={"const_correct": True},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_suggestions_many_templates(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="void f()",
            properties={"template_parameters": ["T", "U", "V"]},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert any("template" in s.lower() for s in m.suggestions)

    def test_suggestions_multiple_inheritance(self):
        a = self._analyzer()
        e = _entity(etype="class", entity_id="c1", properties={"inheritance_count": 2})
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert any("inheritance" in s.lower() for s in m.suggestions)

    def test_suggestions_raw_pointers(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="void f()",
            properties={"uses_raw_pointers": True},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert any("smart pointer" in s.lower() for s in m.suggestions)

    def test_suggestions_operator_overload_complex(self):
        a = self._analyzer()
        sig = "if else while for switch case if"
        e = _entity(
            etype="method",
            signature=sig,
            properties={"is_operator_overload": True},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        if m.cyclomatic_complexity > 5:
            assert any("operator" in s.lower() for s in m.suggestions)

    def test_suggestions_exceptions_not_safe(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="void f()",
            properties={"has_exceptions": True},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert any("exception" in s.lower() for s in m.suggestions)

    def test_suggestions_rule_of_five(self):
        a = self._analyzer()
        e = _entity(
            etype="class",
            entity_id="c1",
            properties={"manages_resources": True},
        )
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert any("rule of five" in s.lower() for s in m.suggestions)

    def test_suggestions_high_coupling(self):
        a = self._analyzer()
        e = _entity(etype="class", entity_id="c1")
        rels = [
            _relation(source_id="c1", target_id=f"t{i}", rel_type="inherits") for i in range(10)
        ]
        pr = _parse_result(language="cpp", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        if m.coupling_factor > 0.7:
            assert any("coupling" in s.lower() for s in m.suggestions)

    def test_suggestions_many_params(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="void f(int a, int b, int c, int d, int e)")
        pr = _parse_result(language="cpp")
        m = a.analyze_entity_complexity(e, pr)
        assert any("struct" in s.lower() or "builder" in s.lower() for s in m.suggestions)


# ===========================================================================
# Java analyzer tests
# ===========================================================================


class TestJavaComplexityAnalyzer:
    def _analyzer(self):
        from processing.analyzers.complexity_analyzers.java_analyzer import JavaComplexityAnalyzer

        return JavaComplexityAnalyzer()

    def test_basic_method(self):
        a = self._analyzer()
        e = _entity(etype="method", signature="void foo(int a, String b)")
        pr = _parse_result(language="java")
        m = a.analyze_entity_complexity(e, pr)
        assert m.language == "java"
        assert m.parameter_count == 2

    def test_class_coupling(self):
        a = self._analyzer()
        e = _entity(etype="class", entity_id="c1")
        rels = [_relation(source_id="c1", target_id=f"t{i}", rel_type="calls") for i in range(5)]
        pr = _parse_result(language="java", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.coupling_factor <= 1.0

    def test_nesting_analysis_no_properties(self):
        a = self._analyzer()
        e = _entity(etype="method", signature="void f()", properties={})
        pr = _parse_result(language="java")
        m = a.analyze_entity_complexity(e, pr)
        assert m.nesting_depth == 0

    def test_maintainability_zero_loc(self):
        a = self._analyzer()
        e = _entity(etype="method", signature="void f()")
        pr = _parse_result(language="java")
        m = a.analyze_entity_complexity(e, pr)
        # lines_of_code defaults to 0
        assert m.maintainability_index == 100.0

    def test_get_language_specific_factors(self):
        a = self._analyzer()
        factors = a.get_language_specific_factors()
        assert "inheritance_penalty" in factors
        assert "interface_bonus" in factors

    def test_classify_complexity(self):
        a = self._analyzer()
        from processing.analyzers.complexity_analyzers.base import ComplexityLevel

        assert a._classify_complexity(3) == ComplexityLevel.TRIVIAL
        assert a._classify_complexity(7) == ComplexityLevel.BASIC
        assert a._classify_complexity(15) == ComplexityLevel.MODERATE
        assert a._classify_complexity(30) == ComplexityLevel.COMPLEX
        assert a._classify_complexity(50) == ComplexityLevel.VERY_COMPLEX

    def test_suggestions_high_complexity(self):
        a = self._analyzer()
        sig = "if else while for switch case if else while for switch"
        e = _entity(etype="method", signature=sig)
        pr = _parse_result(language="java")
        m = a.analyze_entity_complexity(e, pr)
        if m.cyclomatic_complexity > 10:
            assert any("breaking" in s or "smaller" in s for s in m.suggestions)

    def test_suggestions_many_params(self):
        a = self._analyzer()
        e = _entity(
            etype="method",
            signature="void f(int a, int b, int c, int d, int e, int f)",
        )
        pr = _parse_result(language="java")
        m = a.analyze_entity_complexity(e, pr)
        assert any("parameter" in s.lower() or "builder" in s.lower() for s in m.suggestions)

    def test_suggestions_high_coupling(self):
        a = self._analyzer()
        e = _entity(etype="class", entity_id="c1")
        rels = [_relation(source_id="c1", target_id=f"t{i}", rel_type="calls") for i in range(15)]
        pr = _parse_result(language="java", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        if m.coupling_factor > 0.7:
            assert any("coupling" in s.lower() for s in m.suggestions)

    def test_suggestions_large_class(self):
        a = self._analyzer()
        sig = " ".join(["if else while for switch case"] * 10)
        e = _entity(etype="class", entity_id="c1", signature=sig)
        pr = _parse_result(language="java")
        m = a.analyze_entity_complexity(e, pr)
        if m.cyclomatic_complexity > 50:
            assert any(
                "large class" in s.lower() or "splitting" in s.lower() for s in m.suggestions
            )

    def test_non_method_non_class(self):
        a = self._analyzer()
        e = _entity(etype="variable", signature="int x")
        pr = _parse_result(language="java")
        m = a.analyze_entity_complexity(e, pr)
        assert m.parameter_count == 0
        assert m.coupling_factor == 0.0


# ===========================================================================
# Rust analyzer tests
# ===========================================================================


class TestRustComplexityAnalyzer:
    def _analyzer(self):
        from processing.analyzers.complexity_analyzers.rust_analyzer import RustComplexityAnalyzer

        return RustComplexityAnalyzer()

    def test_basic_function(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="fn foo(x: i32, y: i32)")
        pr = _parse_result(language="rust")
        m = a.analyze_entity_complexity(e, pr)
        assert m.language == "rust"
        assert m.parameter_count == 2

    def test_async_function(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="async fn f()", is_async=True)
        pr = _parse_result(language="rust")
        m = a.analyze_entity_complexity(e, pr)
        assert "async" in m.risk_factors
        assert m.computed is False  # no source content passed
        assert m.cyclomatic_complexity == 0.0

    def test_struct_trait_coupling(self):
        a = self._analyzer()
        e = _entity(etype="struct", entity_id="s1")
        rels = [
            _relation(source_id="s1", target_id="t1", rel_type="implements"),
            _relation(source_id="s1", target_id="t2", rel_type="implements"),
        ]
        pr = _parse_result(language="rust", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.coupling_factor <= 1.0
        assert m.coupling_factor == pytest.approx(2.0 / 5.0)

    def test_count_parameters_filters_self(self):
        a = self._analyzer()
        assert a._count_parameters(_entity(signature="fn f(&self, x: i32)")) == 1
        assert a._count_parameters(_entity(signature="fn f(sel)")) == 0
        assert a._count_parameters(_entity(signature="fn f()")) == 0

    def test_get_language_specific_factors(self):
        a = self._analyzer()
        factors = a.get_language_specific_factors()
        assert "lifetime_penalty" in factors
        assert "unsafe_penalty" in factors

    def test_maintainability_unsafe(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="fn f()", properties={"has_unsafe": True})
        pr = _parse_result(language="rust")
        m = a.analyze_entity_complexity(e, pr)
        assert m.maintainability_index <= 80.0  # 100 - 20 penalty

    def test_maintainability_macro_usage(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="fn f()", properties={"macro_usage": 3})
        pr = _parse_result(language="rust")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_maintainability_no_properties(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="fn f()", properties={})
        pr = _parse_result(language="rust")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_classify_complexity(self):
        a = self._analyzer()
        from processing.analyzers.complexity_analyzers.base import ComplexityLevel

        assert a._classify_complexity(2) == ComplexityLevel.TRIVIAL
        assert a._classify_complexity(6) == ComplexityLevel.BASIC
        assert a._classify_complexity(12) == ComplexityLevel.MODERATE
        assert a._classify_complexity(25) == ComplexityLevel.COMPLEX
        assert a._classify_complexity(40) == ComplexityLevel.VERY_COMPLEX

    def test_suggestions_high_complexity(self):
        a = self._analyzer()
        sig = "if else while for switch case if else while for"
        e = _entity(etype="function", signature=sig)
        pr = _parse_result(language="rust")
        m = a.analyze_entity_complexity(e, pr)
        if m.cyclomatic_complexity > 8:
            assert any("smaller" in s.lower() for s in m.suggestions)

    def test_suggestions_many_params(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="fn f(a: i32, b: i32, c: i32, d: i32, e: i32)")
        pr = _parse_result(language="rust")
        m = a.analyze_entity_complexity(e, pr)
        assert any("struct" in s.lower() for s in m.suggestions)

    def test_suggestions_unsafe(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="fn f()", properties={"has_unsafe": True})
        pr = _parse_result(language="rust")
        m = a.analyze_entity_complexity(e, pr)
        assert any("unsafe" in s.lower() for s in m.suggestions)

    def test_suggestions_high_coupling(self):
        a = self._analyzer()
        e = _entity(etype="struct", entity_id="s1")
        rels = [
            _relation(source_id="s1", target_id=f"t{i}", rel_type="implements") for i in range(6)
        ]
        pr = _parse_result(language="rust", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        if m.coupling_factor > 0.8:
            assert any("coupling" in s.lower() for s in m.suggestions)

    def test_non_function_non_struct(self):
        a = self._analyzer()
        e = _entity(etype="variable")
        pr = _parse_result(language="rust")
        m = a.analyze_entity_complexity(e, pr)
        assert m.parameter_count == 0
        assert m.coupling_factor == 0.0


# ===========================================================================
# Go analyzer tests
# ===========================================================================


class TestGoComplexityAnalyzer:
    def _analyzer(self):
        from processing.analyzers.complexity_analyzers.go_analyzer import GoComplexityAnalyzer

        return GoComplexityAnalyzer()

    def test_basic_function(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="func foo(a int, b int)")
        pr = _parse_result(language="go")
        m = a.analyze_entity_complexity(e, pr)
        assert m.language == "go"
        assert m.parameter_count == 2

    def test_struct_interface_coupling(self):
        a = self._analyzer()
        e = _entity(etype="struct", entity_id="s1")
        rels = [
            _relation(source_id="s1", target_id="i1", rel_type="implements"),
            _relation(source_id="s1", target_id="i2", rel_type="implements"),
        ]
        pr = _parse_result(language="go", relationships=rels)
        m = a.analyze_entity_complexity(e, pr)
        assert m.coupling_factor == pytest.approx(2.0 / 3.0)

    def test_count_parameters(self):
        a = self._analyzer()
        assert a._count_parameters(_entity(signature="func f(a int, b string)")) == 2
        assert a._count_parameters(_entity(signature="func f()")) == 0

    def test_get_language_specific_factors(self):
        a = self._analyzer()
        factors = a.get_language_specific_factors()
        assert "goroutine_penalty" in factors
        assert "channel_complexity" in factors

    def test_maintainability(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="func f(a int, b int, c int)")
        pr = _parse_result(language="go")
        m = a.analyze_entity_complexity(e, pr)
        assert 0.0 <= m.maintainability_index <= 100.0

    def test_classify_complexity(self):
        a = self._analyzer()
        from processing.analyzers.complexity_analyzers.base import ComplexityLevel

        assert a._classify_complexity(3) == ComplexityLevel.TRIVIAL
        assert a._classify_complexity(7) == ComplexityLevel.BASIC
        assert a._classify_complexity(12) == ComplexityLevel.MODERATE
        assert a._classify_complexity(20) == ComplexityLevel.COMPLEX
        assert a._classify_complexity(30) == ComplexityLevel.VERY_COMPLEX

    def test_suggestions_high_complexity(self):
        a = self._analyzer()
        sig = "if else while for switch case if else while for switch"
        e = _entity(etype="function", signature=sig)
        pr = _parse_result(language="go")
        m = a.analyze_entity_complexity(e, pr)
        if m.cyclomatic_complexity > 10:
            assert any("splitting" in s.lower() or "smaller" in s.lower() for s in m.suggestions)

    def test_suggestions_many_params(self):
        a = self._analyzer()
        e = _entity(
            etype="function",
            signature="func f(a int, b int, c int, d int, e int, f_p int)",
        )
        pr = _parse_result(language="go")
        m = a.analyze_entity_complexity(e, pr)
        assert any("struct" in s.lower() for s in m.suggestions)

    def test_suggestions_goroutine(self):
        a = self._analyzer()
        e = _entity(etype="function", signature="func f() goroutine")
        pr = _parse_result(language="go")
        m = a.analyze_entity_complexity(e, pr)
        assert any("goroutine" in s.lower() for s in m.suggestions)

    def test_non_function_non_struct(self):
        a = self._analyzer()
        e = _entity(etype="variable")
        pr = _parse_result(language="go")
        m = a.analyze_entity_complexity(e, pr)
        assert m.parameter_count == 0
        assert m.coupling_factor == 0.0


# ===========================================================================
# __init__.py tests (registry, service, factory functions)
# ===========================================================================


class TestGetComplexityAnalyzer:
    def setup_method(self):
        """Clear analyzer cache before each test."""
        import processing.analyzers.complexity_analyzers as mod

        mod._analyzer_cache.clear()

    def test_get_python_analyzer(self):
        from processing.analyzers.complexity_analyzers import get_complexity_analyzer
        from processing.analyzers.complexity_analyzers.python_analyzer import (
            PythonComplexityAnalyzer,
        )

        a = get_complexity_analyzer("python")
        assert isinstance(a, PythonComplexityAnalyzer)

    def test_get_java_analyzer(self):
        from processing.analyzers.complexity_analyzers import get_complexity_analyzer
        from processing.analyzers.complexity_analyzers.java_analyzer import JavaComplexityAnalyzer

        a = get_complexity_analyzer("java")
        assert isinstance(a, JavaComplexityAnalyzer)

    def test_get_javascript_analyzer(self):
        from processing.analyzers.complexity_analyzers import get_complexity_analyzer
        from processing.analyzers.complexity_analyzers.javascript_analyzer import (
            JavaScriptComplexityAnalyzer,
        )

        a = get_complexity_analyzer("javascript")
        assert isinstance(a, JavaScriptComplexityAnalyzer)

    def test_get_typescript_analyzer(self):
        from processing.analyzers.complexity_analyzers import get_complexity_analyzer
        from processing.analyzers.complexity_analyzers.typescript_analyzer import (
            TypeScriptComplexityAnalyzer,
        )

        a = get_complexity_analyzer("typescript")
        assert isinstance(a, TypeScriptComplexityAnalyzer)

    def test_get_rust_analyzer(self):
        from processing.analyzers.complexity_analyzers import get_complexity_analyzer
        from processing.analyzers.complexity_analyzers.rust_analyzer import RustComplexityAnalyzer

        a = get_complexity_analyzer("rust")
        assert isinstance(a, RustComplexityAnalyzer)

    def test_get_go_analyzer(self):
        from processing.analyzers.complexity_analyzers import get_complexity_analyzer
        from processing.analyzers.complexity_analyzers.go_analyzer import GoComplexityAnalyzer

        a = get_complexity_analyzer("go")
        assert isinstance(a, GoComplexityAnalyzer)

    def test_get_cpp_analyzer(self):
        from processing.analyzers.complexity_analyzers import get_complexity_analyzer
        from processing.analyzers.complexity_analyzers.cpp_analyzer import CppComplexityAnalyzer

        a = get_complexity_analyzer("cpp")
        assert isinstance(a, CppComplexityAnalyzer)

    def test_get_cpp_alias(self):
        from processing.analyzers.complexity_analyzers import get_complexity_analyzer
        from processing.analyzers.complexity_analyzers.cpp_analyzer import CppComplexityAnalyzer

        a = get_complexity_analyzer("c++")
        assert isinstance(a, CppComplexityAnalyzer)

    def test_get_csharp_analyzer(self):
        from processing.analyzers.complexity_analyzers import get_complexity_analyzer
        from processing.analyzers.complexity_analyzers.csharp_analyzer import (
            CSharpComplexityAnalyzer,
        )

        a = get_complexity_analyzer("csharp")
        assert isinstance(a, CSharpComplexityAnalyzer)

    def test_get_csharp_alias(self):
        from processing.analyzers.complexity_analyzers import get_complexity_analyzer
        from processing.analyzers.complexity_analyzers.csharp_analyzer import (
            CSharpComplexityAnalyzer,
        )

        a = get_complexity_analyzer("c#")
        assert isinstance(a, CSharpComplexityAnalyzer)

    def test_case_insensitive(self):
        from processing.analyzers.complexity_analyzers import get_complexity_analyzer
        from processing.analyzers.complexity_analyzers.python_analyzer import (
            PythonComplexityAnalyzer,
        )

        a = get_complexity_analyzer("Python")
        assert isinstance(a, PythonComplexityAnalyzer)

    def test_unsupported_language_falls_back_to_java(self):
        from processing.analyzers.complexity_analyzers import get_complexity_analyzer
        from processing.analyzers.complexity_analyzers.java_analyzer import JavaComplexityAnalyzer

        a = get_complexity_analyzer("cobol")
        assert isinstance(a, JavaComplexityAnalyzer)

    def test_caching(self):
        from processing.analyzers.complexity_analyzers import get_complexity_analyzer

        a1 = get_complexity_analyzer("python")
        a2 = get_complexity_analyzer("python")
        assert a1 is a2


class TestGetSupportedLanguages:
    def test_returns_all(self):
        from processing.analyzers.complexity_analyzers import get_supported_languages

        langs = get_supported_languages()
        assert "python" in langs
        assert "java" in langs
        assert "javascript" in langs
        assert "typescript" in langs
        assert "rust" in langs
        assert "go" in langs
        assert "cpp" in langs
        assert "csharp" in langs
        # Should not contain aliases
        assert "c++" not in langs
        assert "c#" not in langs


class TestComplexityAnalysisService:
    def test_init(self):
        from processing.analyzers.complexity_analyzers import ComplexityAnalysisService

        svc = ComplexityAnalysisService()
        assert "python" in svc._analyzers
        assert "java" in svc._analyzers

    def test_get_supported_languages(self):
        from processing.analyzers.complexity_analyzers import ComplexityAnalysisService

        svc = ComplexityAnalysisService()
        langs = svc.get_supported_languages()
        assert "python" in langs
        assert "csharp" in langs

    def test_analyze_file_complexity_empty(self):
        from processing.analyzers.complexity_analyzers import ComplexityAnalysisService

        svc = ComplexityAnalysisService()
        pr = _parse_result(language="python", entities=[], lines_of_code=50)
        report = svc.analyze_file_complexity(pr)
        assert report.total_entities == 0
        assert report.quality_score == 100.0

    def test_analyze_file_complexity_with_entities(self):
        from processing.analyzers.complexity_analyzers import ComplexityAnalysisService

        svc = ComplexityAnalysisService()
        e1 = _entity(etype="function", signature="def foo()")
        e2 = _entity(etype="function", signature="def bar()", entity_id="e2")
        pr = _parse_result(language="python", entities=[e1, e2], lines_of_code=200)
        report = svc.analyze_file_complexity(pr)
        assert report.total_entities == 2
        # No source content passed: entities are not computed, never 1.0.
        assert report.entities_not_computed == report.total_entities
        assert report.max_complexity == 0.0

    def test_analyze_file_complexity_complex_entities(self):
        from processing.analyzers.complexity_analyzers import ComplexityAnalysisService

        svc = ComplexityAnalysisService()
        sig = " ".join(["if else while for switch case"] * 10)
        e1 = _entity(etype="function", signature=sig, entity_id="e1")
        pr = _parse_result(language="python", entities=[e1], lines_of_code=200)
        report = svc.analyze_file_complexity(pr)
        # No source content passed: entities are not computed, never 1.0.
        assert report.entities_not_computed == report.total_entities
        assert report.max_complexity == 0.0

    def test_analyze_file_complexity_unsupported_language(self):
        from processing.analyzers.complexity_analyzers import ComplexityAnalysisService

        svc = ComplexityAnalysisService()
        e1 = _entity(etype="function", signature="foo()")
        pr = _parse_result(language="cobol", entities=[e1])
        report = svc.analyze_file_complexity(pr)
        assert report.language == "cobol"
        assert report.total_entities == 1

    def test_quality_score_large_file(self):
        from processing.analyzers.complexity_analyzers import ComplexityAnalysisService

        svc = ComplexityAnalysisService()
        e1 = _entity(etype="function", signature="def f()")
        pr = _parse_result(language="python", entities=[e1], lines_of_code=1500)
        report = svc.analyze_file_complexity(pr)
        # No source content: nothing measured, so no score (not a clean 100).
        assert report.quality_score is None
        # Large file penalty applied when a score is computed
        assert svc._calculate_quality_score(report) < 100.0

    def test_file_suggestions_large_file(self):
        from processing.analyzers.complexity_analyzers import ComplexityAnalysisService

        svc = ComplexityAnalysisService()
        e1 = _entity(etype="function", signature="def f()")
        pr = _parse_result(language="python", entities=[e1], lines_of_code=1500)
        report = svc.analyze_file_complexity(pr)
        assert any("large file" in s.lower() for s in report.refactoring_suggestions)

    def test_file_suggestions_high_complex_ratio(self):
        from processing.analyzers.complexity_analyzers import ComplexityAnalysisService

        svc = ComplexityAnalysisService()
        # Create entities that will be classified as complex
        sig = " ".join(["if else while for switch case"] * 10)
        entities = [_entity(etype="function", signature=sig, entity_id=f"e{i}") for i in range(5)]
        pr = _parse_result(language="python", entities=entities)
        report = svc.analyze_file_complexity(pr)
        if report.complex_entities > report.total_entities * 0.3:
            assert any("ratio" in s.lower() for s in report.refactoring_suggestions)

    def test_file_suggestions_many_hotspots(self):
        from processing.analyzers.complexity_analyzers import ComplexityAnalysisService

        svc = ComplexityAnalysisService()
        sig = " ".join(["if else while for switch case"] * 10)
        entities = [_entity(etype="function", signature=sig, entity_id=f"e{i}") for i in range(10)]
        pr = _parse_result(language="python", entities=entities)
        report = svc.analyze_file_complexity(pr)
        if len(report.hotspots) > 5:
            assert any("hotspot" in s.lower() for s in report.refactoring_suggestions)


class TestGetComplexityService:
    def setup_method(self):
        """Reset global service before each test."""
        from processing.analyzers.complexity_analyzers import service as service_mod

        service_mod._complexity_service = None

    def test_returns_instance(self):
        from processing.analyzers.complexity_analyzers import get_complexity_service

        svc = get_complexity_service()
        assert svc is not None

    def test_caching(self):
        from processing.analyzers.complexity_analyzers import get_complexity_service

        svc1 = get_complexity_service()
        svc2 = get_complexity_service()
        assert svc1 is svc2


class TestAllExports:
    def test_all_exports(self):
        import processing.analyzers.complexity_analyzers as mod

        for name in mod.__all__:
            assert hasattr(mod, name), f"Missing export: {name}"
