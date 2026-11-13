"""Tests for processing.semantic_analysis.result_aggregator module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.core_models import (
    CoreCodeEntity,
    EntityRelation,
    EntityType,
    RelationType,
)
from processing.semantic_analysis.result_aggregator import (
    AnalysisMetric,
    AnalysisMetricType,
    DataFlowAnalyzer,
    ResultAggregator,
    SemanticAnalysisResult,
)
from processing.semantic_analysis.semantic_parser import SemanticContext
from processing.semantic_analysis.semantic_rules import (
    RuleSeverity,
    RuleViolation,
    RuleViolationType,
)
from processing.semantic_analysis.symbol_table import Scope, ScopeType, Symbol, SymbolType
from processing.semantic_analysis.type_inference import TypeConstraint, TypeInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(
    id: str = "e1",
    name: str = "my_func",
    etype: EntityType = EntityType.FUNCTION,
    language: str = "python",
    file_path: str = "src/app.py",
    line_number: int = 1,
    signature: str | None = None,
    properties: dict | None = None,
) -> CoreCodeEntity:
    return CoreCodeEntity(
        id=id,
        name=name,
        type=etype,
        language=language,
        file_path=file_path,
        line_number=line_number,
        signature=signature,
        properties=properties or {},
    )


def _make_relation(
    source_id: str = "e1",
    target_id: str = "e2",
    rel_type: RelationType = RelationType.CALLS,
) -> EntityRelation:
    return EntityRelation(source_id=source_id, target_id=target_id, type=rel_type)


def _make_violation(
    entity_id: str = "e1",
    file_path: str = "src/app.py",
    severity: RuleSeverity = RuleSeverity.HIGH,
) -> RuleViolation:
    return RuleViolation(
        rule_name="test_rule",
        violation_type=RuleViolationType.ARCHITECTURAL_BOUNDARY,
        severity=severity,
        message="test violation",
        entity_id=entity_id,
        file_path=file_path,
    )


def _make_symbol(
    name: str = "sym1",
    file_path: str = "src/app.py",
) -> Symbol:
    return Symbol(
        name=name,
        symbol_type=SymbolType.VARIABLE,
        entity_id="e1",
        scope_id="scope1",
        file_path=file_path,
    )


def _make_scope(
    scope_id: str = "scope1",
    file_path: str = "src/app.py",
) -> Scope:
    return Scope(
        scope_id=scope_id,
        scope_type=ScopeType.MODULE,
        name="module",
        file_path=file_path,
    )


def _make_type_info(name: str = "MyType") -> TypeInfo:
    return TypeInfo(name=name, kind="class")


def _make_type_constraint() -> TypeConstraint:
    return TypeConstraint(
        type_parameter="T",
        constraint_type="BaseClass",
        constraint_kind="extends",
    )


def _make_context(
    file_path: str = "src/app.py",
    imports: set[str] | None = None,
    exports: set[str] | None = None,
) -> SemanticContext:
    return SemanticContext(
        file_path=Path(file_path),
        language="python",
        imports=imports or set(),
        exports=exports or set(),
    )


# ---------------------------------------------------------------------------
# SemanticAnalysisResult dataclass
# ---------------------------------------------------------------------------


class TestSemanticAnalysisResult:
    def test_default_construction(self):
        result = SemanticAnalysisResult()
        assert result.entities == []
        assert result.relationships == []
        assert result.violations == []
        assert result.metrics == []
        assert result.symbols == {}
        assert result.scopes == {}
        assert result.types == {}
        assert result.type_constraints == []
        assert result.summary == {}

    def test_construction_with_data(self):
        entity = _make_entity()
        result = SemanticAnalysisResult(
            entities=[entity],
            symbols={"s1": _make_symbol()},
            types={"t1": _make_type_info()},
        )
        assert len(result.entities) == 1
        assert "s1" in result.symbols
        assert "t1" in result.types


# ---------------------------------------------------------------------------
# AnalysisMetric dataclass
# ---------------------------------------------------------------------------


class TestAnalysisMetric:
    def test_basic_construction(self):
        m = AnalysisMetric(
            name="test",
            metric_type=AnalysisMetricType.ENTITY_COUNT,
            value=5.0,
            description="test metric",
        )
        assert m.name == "test"
        assert m.value == 5.0
        assert m.file_path is None
        assert m.entity_id is None
        assert m.metadata == {}

    def test_construction_with_all_fields(self):
        m = AnalysisMetric(
            name="complexity",
            metric_type=AnalysisMetricType.COMPLEXITY_SCORE,
            value=12.0,
            description="complexity score",
            file_path="src/app.py",
            entity_id="e1",
            metadata={"detail": "high"},
        )
        assert m.file_path == "src/app.py"
        assert m.entity_id == "e1"
        assert m.metadata["detail"] == "high"


# ---------------------------------------------------------------------------
# AnalysisMetricType enum
# ---------------------------------------------------------------------------


class TestAnalysisMetricType:
    def test_enum_values(self):
        assert AnalysisMetricType.RELATIONSHIP_COUNT == "relationship_count"
        assert AnalysisMetricType.ENTITY_COUNT == "entity_count"
        assert AnalysisMetricType.VIOLATION_COUNT == "violation_count"
        assert AnalysisMetricType.COMPLEXITY_SCORE == "complexity_score"
        assert AnalysisMetricType.COUPLING_METRIC == "coupling_metric"
        assert AnalysisMetricType.COHESION_METRIC == "cohesion_metric"
        assert AnalysisMetricType.MAINTAINABILITY_INDEX == "maintainability_index"


# ---------------------------------------------------------------------------
# ResultAggregator
# ---------------------------------------------------------------------------


class TestResultAggregator:
    @pytest.fixture
    def aggregator(self) -> ResultAggregator:
        return ResultAggregator()

    def test_init_creates_data_flow_analyzer(self, aggregator):
        assert isinstance(aggregator.data_flow_analyzer, DataFlowAnalyzer)

    # ---- aggregate_results ----

    def test_aggregate_results_returns_semantic_analysis_result(self, aggregator):
        result = aggregator.aggregate_results(
            entities=[],
            relationships=[],
            violations=[],
            symbols={},
            scopes={},
            types={},
            type_constraints=[],
            contexts={},
        )
        assert isinstance(result, SemanticAnalysisResult)

    def test_aggregate_results_stores_inputs(self, aggregator):
        entity = _make_entity()
        rel = _make_relation()
        violation = _make_violation()
        sym = _make_symbol()
        scope = _make_scope()
        ti = _make_type_info()
        tc = _make_type_constraint()

        result = aggregator.aggregate_results(
            entities=[entity],
            relationships=[rel],
            violations=[violation],
            symbols={"s1": sym},
            scopes={"sc1": scope},
            types={"t1": ti},
            type_constraints=[tc],
            contexts={},
        )

        assert result.entities == [entity]
        assert result.relationships == [rel]
        assert result.violations == [violation]
        assert result.symbols == {"s1": sym}
        assert result.scopes == {"sc1": scope}
        assert result.types == {"t1": ti}
        assert result.type_constraints == [tc]

    def test_aggregate_results_generates_metrics(self, aggregator):
        entity = _make_entity()
        result = aggregator.aggregate_results(
            entities=[entity],
            relationships=[],
            violations=[],
            symbols={},
            scopes={},
            types={},
            type_constraints=[],
            contexts={},
        )
        assert len(result.metrics) > 0

    def test_aggregate_results_generates_summary(self, aggregator):
        entity = _make_entity()
        result = aggregator.aggregate_results(
            entities=[entity],
            relationships=[],
            violations=[],
            symbols={},
            scopes={},
            types={},
            type_constraints=[],
            contexts={},
        )
        assert "entity_count" in result.summary
        assert result.summary["entity_count"] == 1

    # ---- _calculate_count_metrics ----

    def test_count_metrics_entity_breakdown(self, aggregator):
        entities = [
            _make_entity(id="e1", name="f1", etype=EntityType.FUNCTION),
            _make_entity(id="e2", name="f2", etype=EntityType.FUNCTION),
            _make_entity(id="e3", name="C1", etype=EntityType.CLASS),
        ]
        result = SemanticAnalysisResult(entities=entities)
        metrics = aggregator._calculate_count_metrics(result)

        entity_metrics = [m for m in metrics if m.metric_type == AnalysisMetricType.ENTITY_COUNT]
        values_by_name = {m.name: m.value for m in entity_metrics}
        assert values_by_name["function_count"] == 2.0
        assert values_by_name["class_count"] == 1.0

    def test_count_metrics_relationship_breakdown(self, aggregator):
        rels = [
            _make_relation(source_id="e1", target_id="e2", rel_type=RelationType.CALLS),
            _make_relation(source_id="e2", target_id="e3", rel_type=RelationType.CALLS),
            _make_relation(source_id="e1", target_id="e3", rel_type=RelationType.IMPORTS),
        ]
        result = SemanticAnalysisResult(relationships=rels)
        metrics = aggregator._calculate_count_metrics(result)

        rel_metrics = [m for m in metrics if m.metric_type == AnalysisMetricType.RELATIONSHIP_COUNT]
        values_by_name = {m.name: m.value for m in rel_metrics}
        assert values_by_name["calls_relationship_count"] == 2.0
        assert values_by_name["imports_relationship_count"] == 1.0

    def test_count_metrics_violation_breakdown(self, aggregator):
        violations = [
            _make_violation(severity=RuleSeverity.CRITICAL),
            _make_violation(severity=RuleSeverity.CRITICAL),
            _make_violation(severity=RuleSeverity.LOW),
        ]
        result = SemanticAnalysisResult(violations=violations)
        metrics = aggregator._calculate_count_metrics(result)

        viol_metrics = [m for m in metrics if m.metric_type == AnalysisMetricType.VIOLATION_COUNT]
        values_by_name = {m.name: m.value for m in viol_metrics}
        assert values_by_name["critical_violation_count"] == 2.0
        assert values_by_name["low_violation_count"] == 1.0

    def test_count_metrics_empty_result(self, aggregator):
        result = SemanticAnalysisResult()
        metrics = aggregator._calculate_count_metrics(result)
        assert metrics == []

    # ---- _estimate_cyclomatic_complexity ----

    def test_cyclomatic_complexity_base(self, aggregator):
        entity = _make_entity(properties={"content": "return x"})
        score = aggregator._estimate_cyclomatic_complexity(entity)
        assert score == 1.0  # base only

    def test_cyclomatic_complexity_with_branches(self, aggregator):
        content = "if x:\n  while y:\n    for i in z:\n      pass"
        entity = _make_entity(properties={"content": content})
        score = aggregator._estimate_cyclomatic_complexity(entity)
        # base(1) + if(1) + while(1) + for(1) = 4
        assert score >= 4.0

    def test_cyclomatic_complexity_with_logical_operators(self, aggregator):
        content = "if x && y || z:\n  pass"
        entity = _make_entity(properties={"content": content})
        score = aggregator._estimate_cyclomatic_complexity(entity)
        # base(1) + if(1) + &&(1) + ||(1) = 4
        assert score >= 4.0

    def test_cyclomatic_complexity_empty_content(self, aggregator):
        entity = _make_entity(properties={})
        score = aggregator._estimate_cyclomatic_complexity(entity)
        assert score == 1.0  # base only

    # ---- _calculate_complexity_metrics ----

    def test_complexity_metrics_for_functions(self, aggregator):
        entities = [
            _make_entity(
                id="e1", name="f1", etype=EntityType.FUNCTION, properties={"content": "if x: pass"}
            ),
            _make_entity(id="e2", name="C1", etype=EntityType.CLASS),
        ]
        result = SemanticAnalysisResult(entities=entities)
        metrics = aggregator._calculate_complexity_metrics(result)

        complexity_metrics = [m for m in metrics if m.name == "cyclomatic_complexity"]
        assert len(complexity_metrics) == 1
        assert complexity_metrics[0].entity_id == "e1"

    # ---- _calculate_coupling_metrics ----

    def test_coupling_metrics_counts_relationships(self, aggregator):
        entities = [
            _make_entity(id="e1", name="f1", etype=EntityType.FUNCTION),
            _make_entity(id="e2", name="f2", etype=EntityType.FUNCTION),
        ]
        rels = [
            _make_relation(source_id="e1", target_id="e2"),
            _make_relation(source_id="e1", target_id="e3"),
        ]
        result = SemanticAnalysisResult(entities=entities, relationships=rels)
        metrics = aggregator._calculate_coupling_metrics(result)

        coupling = {m.entity_id: m for m in metrics if m.name == "coupling_metric"}
        # e1: 2 outgoing
        assert coupling["e1"].value == 2.0
        assert coupling["e1"].metadata["outgoing"] == 2
        # e2: 1 incoming
        assert coupling["e2"].value == 1.0
        assert coupling["e2"].metadata["incoming"] == 1

    def test_coupling_metrics_ignores_non_class_function(self, aggregator):
        entities = [
            _make_entity(id="e1", name="v1", etype=EntityType.VARIABLE),
        ]
        rels = [_make_relation(source_id="e1", target_id="e2")]
        result = SemanticAnalysisResult(entities=entities, relationships=rels)
        metrics = aggregator._calculate_coupling_metrics(result)
        coupling = [m for m in metrics if m.name == "coupling_metric"]
        assert len(coupling) == 0

    # ---- _calculate_cohesion_metrics ----

    def test_cohesion_metrics_for_class(self, aggregator):
        entities = [
            _make_entity(id="c1", name="MyClass", etype=EntityType.CLASS),
        ]
        result = SemanticAnalysisResult(entities=entities)
        metrics = aggregator._calculate_cohesion_metrics(result)
        cohesion = [m for m in metrics if m.name == "class_cohesion"]
        assert len(cohesion) == 1
        assert cohesion[0].entity_id == "c1"

    def test_cohesion_metrics_skips_non_class(self, aggregator):
        entities = [
            _make_entity(id="e1", name="f1", etype=EntityType.FUNCTION),
        ]
        result = SemanticAnalysisResult(entities=entities)
        metrics = aggregator._calculate_cohesion_metrics(result)
        cohesion = [m for m in metrics if m.name == "class_cohesion"]
        assert len(cohesion) == 0

    # ---- _estimate_class_cohesion ----

    def test_class_cohesion_no_members_returns_one(self, aggregator):
        class_entity = _make_entity(id="c1", name="MyClass", etype=EntityType.CLASS)
        result = SemanticAnalysisResult(entities=[class_entity])
        cohesion = aggregator._estimate_class_cohesion(class_entity, result)
        assert cohesion == 1.0

    def test_class_cohesion_single_member_returns_one(self, aggregator):
        class_entity = _make_entity(id="c1", name="MyClass", etype=EntityType.CLASS)
        method = _make_entity(
            id="m1",
            name="method1",
            etype=EntityType.FUNCTION,
            properties={"parent_class": "MyClass"},
        )
        result = SemanticAnalysisResult(entities=[class_entity, method])
        cohesion = aggregator._estimate_class_cohesion(class_entity, result)
        assert cohesion == 1.0

    def test_class_cohesion_with_internal_relationships(self, aggregator):
        class_entity = _make_entity(id="c1", name="MyClass", etype=EntityType.CLASS)
        m1 = _make_entity(
            id="m1",
            name="method1",
            etype=EntityType.FUNCTION,
            properties={"parent_class": "MyClass"},
        )
        m2 = _make_entity(
            id="m2",
            name="method2",
            etype=EntityType.FUNCTION,
            properties={"parent_class": "MyClass"},
        )
        rel = _make_relation(source_id="m1", target_id="m2")
        result = SemanticAnalysisResult(
            entities=[class_entity, m1, m2],
            relationships=[rel],
        )
        cohesion = aggregator._estimate_class_cohesion(class_entity, result)
        # 1 internal rel / (2 * 1) = 0.5
        assert cohesion == 0.5

    # ---- _calculate_quality_metrics ----

    def test_quality_metrics_for_function_and_class(self, aggregator):
        entities = [
            _make_entity(id="f1", name="func1", etype=EntityType.FUNCTION),
            _make_entity(id="c1", name="Class1", etype=EntityType.CLASS),
            _make_entity(id="v1", name="var1", etype=EntityType.VARIABLE),
        ]
        result = SemanticAnalysisResult(entities=entities)
        metrics = aggregator._calculate_quality_metrics(result)
        maint = [m for m in metrics if m.name == "maintainability_index"]
        # Should have metrics for function and class, not variable
        entity_ids = {m.entity_id for m in maint}
        assert "f1" in entity_ids
        assert "c1" in entity_ids
        assert "v1" not in entity_ids

    # ---- _estimate_maintainability_index ----

    def test_maintainability_base_score(self, aggregator):
        entity = _make_entity(etype=EntityType.CLASS, properties={"content": "x = 1"})
        result = SemanticAnalysisResult(entities=[entity])
        score = aggregator._estimate_maintainability_index(entity, result)
        # Base is 100, class doesn't get complexity penalty, short content
        assert score == 100.0

    def test_maintainability_reduced_by_complexity(self, aggregator):
        content = "if x:\n  if y:\n    while z:\n      pass"
        entity = _make_entity(id="f1", etype=EntityType.FUNCTION, properties={"content": content})
        result = SemanticAnalysisResult(entities=[entity])
        score = aggregator._estimate_maintainability_index(entity, result)
        # complexity >= 4, so penalty = (complexity - 1) * 2 >= 6
        assert score < 100.0

    def test_maintainability_reduced_by_size(self, aggregator):
        # 100 lines of code
        content = "\n".join([f"line {i}" for i in range(100)])
        entity = _make_entity(etype=EntityType.CLASS, properties={"content": content})
        result = SemanticAnalysisResult(entities=[entity])
        score = aggregator._estimate_maintainability_index(entity, result)
        # penalty for size: (100 - 50) * 0.1 = 5.0
        assert score == 95.0

    def test_maintainability_reduced_by_violations(self, aggregator):
        entity = _make_entity(id="e1", etype=EntityType.CLASS)
        violations = [
            _make_violation(entity_id="e1", severity=RuleSeverity.CRITICAL),
            _make_violation(entity_id="e1", severity=RuleSeverity.HIGH),
        ]
        result = SemanticAnalysisResult(entities=[entity], violations=violations)
        score = aggregator._estimate_maintainability_index(entity, result)
        # Base 100 - critical(20) - high(10) = 70
        assert score == 70.0

    def test_maintainability_never_below_zero(self, aggregator):
        entity = _make_entity(id="e1", etype=EntityType.CLASS)
        violations = [
            _make_violation(entity_id="e1", severity=RuleSeverity.CRITICAL) for _ in range(10)
        ]
        result = SemanticAnalysisResult(entities=[entity], violations=violations)
        score = aggregator._estimate_maintainability_index(entity, result)
        assert score == 0.0

    # ---- _calculate_file_level_metrics ----

    def test_file_level_metrics_lines_of_code(self, aggregator):
        entity = _make_entity(properties={"content": "line1\nline2\nline3"})
        result = SemanticAnalysisResult(entities=[entity])
        metrics = aggregator._calculate_file_level_metrics(result, {})
        loc_metrics = [m for m in metrics if m.name == "lines_of_code"]
        assert len(loc_metrics) == 1
        assert loc_metrics[0].value == 3.0

    def test_file_level_metrics_with_context(self, aggregator):
        entity = _make_entity(file_path="src/app.py", properties={"content": "x"})
        ctx = _make_context(
            file_path="src/app.py",
            imports={"os", "sys"},
            exports={"main"},
        )
        result = SemanticAnalysisResult(entities=[entity])
        metrics = aggregator._calculate_file_level_metrics(result, {"src/app.py": ctx})

        import_metrics = [m for m in metrics if m.name == "import_count"]
        assert len(import_metrics) == 1
        assert import_metrics[0].value == 2.0

        export_metrics = [m for m in metrics if m.name == "export_count"]
        assert len(export_metrics) == 1
        assert export_metrics[0].value == 1.0

    def test_file_level_metrics_no_context(self, aggregator):
        entity = _make_entity(file_path="src/app.py", properties={"content": "x"})
        result = SemanticAnalysisResult(entities=[entity])
        metrics = aggregator._calculate_file_level_metrics(result, {})
        # Should still have lines_of_code but no import/export metrics
        import_metrics = [m for m in metrics if m.name == "import_count"]
        assert len(import_metrics) == 0

    # ---- _generate_summary ----

    def test_summary_basic_counts(self, aggregator):
        entity = _make_entity()
        rel = _make_relation()
        violation = _make_violation()
        sym = _make_symbol()
        scope = _make_scope()
        ti = _make_type_info()

        result = SemanticAnalysisResult(
            entities=[entity],
            relationships=[rel],
            violations=[violation],
            symbols={"s1": sym},
            scopes={"sc1": scope},
            types={"t1": ti},
        )
        summary = aggregator._generate_summary(result)

        assert summary["entity_count"] == 1
        assert summary["relationship_count"] == 1
        assert summary["violation_count"] == 1
        assert summary["symbol_count"] == 1
        assert summary["scope_count"] == 1
        assert summary["type_count"] == 1

    def test_summary_entity_breakdown(self, aggregator):
        entities = [
            _make_entity(id="e1", etype=EntityType.FUNCTION),
            _make_entity(id="e2", etype=EntityType.FUNCTION),
            _make_entity(id="e3", etype=EntityType.CLASS),
        ]
        result = SemanticAnalysisResult(entities=entities)
        summary = aggregator._generate_summary(result)

        assert summary["entity_breakdown"]["function"] == 2
        assert summary["entity_breakdown"]["class"] == 1

    def test_summary_relationship_breakdown(self, aggregator):
        rels = [
            _make_relation(rel_type=RelationType.CALLS),
            _make_relation(rel_type=RelationType.IMPORTS),
        ]
        result = SemanticAnalysisResult(relationships=rels)
        summary = aggregator._generate_summary(result)

        assert summary["relationship_breakdown"]["calls"] == 1
        assert summary["relationship_breakdown"]["imports"] == 1

    def test_summary_violation_breakdown(self, aggregator):
        violations = [
            _make_violation(severity=RuleSeverity.CRITICAL),
            _make_violation(severity=RuleSeverity.LOW),
        ]
        result = SemanticAnalysisResult(violations=violations)
        summary = aggregator._generate_summary(result)

        assert summary["violation_breakdown"]["critical"] == 1
        assert summary["violation_breakdown"]["low"] == 1

    def test_summary_language_breakdown(self, aggregator):
        entities = [
            _make_entity(id="e1", language="python"),
            _make_entity(id="e2", language="java"),
            _make_entity(id="e3", language="python"),
        ]
        result = SemanticAnalysisResult(entities=entities)
        summary = aggregator._generate_summary(result)

        assert summary["language_breakdown"]["python"] == 2
        assert summary["language_breakdown"]["java"] == 1

    def test_summary_average_maintainability(self, aggregator):
        metrics = [
            AnalysisMetric(
                name="maintainability_index",
                metric_type=AnalysisMetricType.MAINTAINABILITY_INDEX,
                value=80.0,
                description="test",
            ),
            AnalysisMetric(
                name="maintainability_index",
                metric_type=AnalysisMetricType.MAINTAINABILITY_INDEX,
                value=60.0,
                description="test",
            ),
        ]
        result = SemanticAnalysisResult(metrics=metrics)
        summary = aggregator._generate_summary(result)
        assert summary["average_maintainability"] == 70.0

    def test_summary_average_complexity(self, aggregator):
        metrics = [
            AnalysisMetric(
                name="cyclomatic_complexity",
                metric_type=AnalysisMetricType.COMPLEXITY_SCORE,
                value=5.0,
                description="test",
            ),
            AnalysisMetric(
                name="complexity_other",
                metric_type=AnalysisMetricType.COMPLEXITY_SCORE,
                value=15.0,
                description="test",
            ),
        ]
        result = SemanticAnalysisResult(metrics=metrics)
        summary = aggregator._generate_summary(result)
        assert summary["average_complexity"] == 10.0

    def test_summary_critical_violation_count(self, aggregator):
        violations = [
            _make_violation(severity=RuleSeverity.CRITICAL),
            _make_violation(severity=RuleSeverity.CRITICAL),
            _make_violation(severity=RuleSeverity.LOW),
        ]
        result = SemanticAnalysisResult(violations=violations)
        summary = aggregator._generate_summary(result)
        assert summary["critical_violation_count"] == 2

    def test_summary_high_coupling_entity_count(self, aggregator):
        metrics = [
            AnalysisMetric(
                name="coupling",
                metric_type=AnalysisMetricType.COUPLING_METRIC,
                value=15.0,  # above threshold 10
                description="test",
            ),
            AnalysisMetric(
                name="coupling",
                metric_type=AnalysisMetricType.COUPLING_METRIC,
                value=5.0,  # below threshold
                description="test",
            ),
        ]
        result = SemanticAnalysisResult(metrics=metrics)
        summary = aggregator._generate_summary(result)
        assert summary["high_coupling_entity_count"] == 1

    def test_summary_no_quality_metrics(self, aggregator):
        result = SemanticAnalysisResult()
        summary = aggregator._generate_summary(result)
        assert "average_maintainability" not in summary
        assert "average_complexity" not in summary

    # ---- filter_results_by_file ----

    def test_filter_results_by_file_entities(self, aggregator):
        e1 = _make_entity(id="e1", file_path="src/app.py")
        e2 = _make_entity(id="e2", file_path="src/other.py")
        result = SemanticAnalysisResult(entities=[e1, e2])
        filtered = aggregator.filter_results_by_file(result, "src/app.py")

        assert len(filtered.entities) == 1
        assert filtered.entities[0].id == "e1"

    def test_filter_results_by_file_relationships(self, aggregator):
        e1 = _make_entity(id="e1", file_path="src/app.py")
        e2 = _make_entity(id="e2", file_path="src/other.py")
        # rel connecting e1 and e2: should be included since e1 is in the file
        rel = _make_relation(source_id="e1", target_id="e2")
        # rel only involving e2: should not be included
        rel2 = _make_relation(source_id="e2", target_id="e3")
        result = SemanticAnalysisResult(entities=[e1, e2], relationships=[rel, rel2])
        filtered = aggregator.filter_results_by_file(result, "src/app.py")

        assert len(filtered.relationships) == 1
        assert filtered.relationships[0].source_id == "e1"

    def test_filter_results_by_file_violations(self, aggregator):
        v1 = _make_violation(entity_id="e1", file_path="src/app.py")
        v2 = _make_violation(entity_id="e2", file_path="src/other.py")
        result = SemanticAnalysisResult(violations=[v1, v2])
        filtered = aggregator.filter_results_by_file(result, "src/app.py")

        assert len(filtered.violations) == 1
        assert filtered.violations[0].entity_id == "e1"

    def test_filter_results_by_file_symbols(self, aggregator):
        s1 = _make_symbol(name="sym1", file_path="src/app.py")
        s2 = _make_symbol(name="sym2", file_path="src/other.py")
        result = SemanticAnalysisResult(symbols={"s1": s1, "s2": s2})
        filtered = aggregator.filter_results_by_file(result, "src/app.py")

        assert len(filtered.symbols) == 1
        assert "s1" in filtered.symbols

    def test_filter_results_by_file_scopes(self, aggregator):
        sc1 = _make_scope(scope_id="sc1", file_path="src/app.py")
        sc2 = _make_scope(scope_id="sc2", file_path="src/other.py")
        result = SemanticAnalysisResult(scopes={"sc1": sc1, "sc2": sc2})
        filtered = aggregator.filter_results_by_file(result, "src/app.py")

        assert len(filtered.scopes) == 1
        assert "sc1" in filtered.scopes

    def test_filter_results_by_file_metrics(self, aggregator):
        m1 = AnalysisMetric(
            name="loc",
            metric_type=AnalysisMetricType.COMPLEXITY_SCORE,
            value=10.0,
            description="test",
            file_path="src/app.py",
        )
        m2 = AnalysisMetric(
            name="loc",
            metric_type=AnalysisMetricType.COMPLEXITY_SCORE,
            value=20.0,
            description="test",
            file_path="src/other.py",
        )
        m3 = AnalysisMetric(
            name="maintain",
            metric_type=AnalysisMetricType.MAINTAINABILITY_INDEX,
            value=90.0,
            description="test",
            entity_id="e1",
        )
        e1 = _make_entity(id="e1", file_path="src/app.py")
        result = SemanticAnalysisResult(entities=[e1], metrics=[m1, m2, m3])
        filtered = aggregator.filter_results_by_file(result, "src/app.py")

        # m1 matches by file_path, m3 matches by entity_id
        assert len(filtered.metrics) == 2

    def test_filter_results_preserves_types_and_constraints(self, aggregator):
        ti = _make_type_info()
        tc = _make_type_constraint()
        result = SemanticAnalysisResult(types={"t1": ti}, type_constraints=[tc])
        filtered = aggregator.filter_results_by_file(result, "src/app.py")

        # Types and constraints are global, should be preserved
        assert filtered.types == {"t1": ti}
        assert filtered.type_constraints == [tc]

    def test_filter_results_generates_summary(self, aggregator):
        e1 = _make_entity(id="e1", file_path="src/app.py")
        result = SemanticAnalysisResult(entities=[e1])
        filtered = aggregator.filter_results_by_file(result, "src/app.py")

        assert "entity_count" in filtered.summary
        assert filtered.summary["entity_count"] == 1

    def test_filter_results_empty_file(self, aggregator):
        e1 = _make_entity(id="e1", file_path="src/app.py")
        result = SemanticAnalysisResult(entities=[e1])
        filtered = aggregator.filter_results_by_file(result, "src/nonexistent.py")

        assert len(filtered.entities) == 0
        assert filtered.summary["entity_count"] == 0


# ---------------------------------------------------------------------------
# DataFlowAnalyzer
# ---------------------------------------------------------------------------


class TestDataFlowAnalyzer:
    @pytest.fixture
    def analyzer(self) -> DataFlowAnalyzer:
        return DataFlowAnalyzer()

    # ---- analyze_data_flow ----

    async def test_analyze_data_flow_calls_sub_analyzers(self, analyzer):
        """Verify analyze_data_flow delegates to all three sub-analyzers."""
        with (
            patch.object(
                analyzer, "_analyze_variable_flow", new_callable=AsyncMock, return_value=[]
            ) as mock_var,
            patch.object(
                analyzer, "_analyze_state_mutations", new_callable=AsyncMock, return_value=[]
            ) as mock_mut,
            patch.object(
                analyzer, "_analyze_memory_sharing", new_callable=AsyncMock, return_value=[]
            ) as mock_mem,
        ):
            result = await analyzer.analyze_data_flow([], [], {})

            mock_var.assert_called_once()
            mock_mut.assert_called_once()
            mock_mem.assert_called_once()
            assert result == []

    async def test_analyze_data_flow_combines_results(self, analyzer):
        """Verify results from sub-analyzers are combined."""
        mock_rel1 = MagicMock()
        mock_rel2 = MagicMock()
        mock_rel3 = MagicMock()

        with (
            patch.object(
                analyzer, "_analyze_variable_flow", new_callable=AsyncMock, return_value=[mock_rel1]
            ),
            patch.object(
                analyzer,
                "_analyze_state_mutations",
                new_callable=AsyncMock,
                return_value=[mock_rel2],
            ),
            patch.object(
                analyzer,
                "_analyze_memory_sharing",
                new_callable=AsyncMock,
                return_value=[mock_rel3],
            ),
        ):
            result = await analyzer.analyze_data_flow([], [], {})
            assert len(result) == 3

    # ---- _analyze_variable_flow ----

    async def test_variable_flow_no_functions(self, analyzer):
        """No functions means no variable flow relationships."""
        entities = [_make_entity(etype=EntityType.CLASS)]
        result = await analyzer._analyze_variable_flow(entities)
        assert result == []

    async def test_variable_flow_no_matches(self, analyzer):
        """Function with content that doesn't match patterns."""
        entity = _make_entity(
            id="f1",
            etype=EntityType.FUNCTION,
            properties={"content": "x = 42"},
        )
        result = await analyzer._analyze_variable_flow([entity])
        assert result == []

    async def test_variable_flow_func_call_pattern(self, analyzer):
        """Test var = func() pattern where func is another entity."""
        func_entity = _make_entity(
            id="f1",
            name="process",
            etype=EntityType.FUNCTION,
            properties={"content": "result = helper(args)"},
        )
        helper_entity = _make_entity(
            id="f2",
            name="helper",
            etype=EntityType.FUNCTION,
        )
        # Patch EntityRelation to accept SemanticRelationType values
        with patch(
            "processing.semantic_analysis.result_aggregator.EntityRelation",
            side_effect=lambda **kw: MagicMock(**kw),
        ):
            result = await analyzer._analyze_variable_flow([func_entity, helper_entity])
            assert len(result) == 1
            assert result[0].source_id == "f1"
            assert result[0].target_id == "f2"

    async def test_variable_flow_method_call_pattern(self, analyzer):
        """Test var = obj.method() pattern."""
        func_entity = _make_entity(
            id="f1",
            name="run",
            etype=EntityType.FUNCTION,
            properties={"content": "data = service.fetch()"},
        )
        service_entity = _make_entity(
            id="s1",
            name="service",
            etype=EntityType.FUNCTION,
        )
        with patch(
            "processing.semantic_analysis.result_aggregator.EntityRelation",
            side_effect=lambda **kw: MagicMock(**kw),
        ):
            result = await analyzer._analyze_variable_flow([func_entity, service_entity])
            assert len(result) == 1
            assert result[0].target_id == "s1"

    async def test_variable_flow_empty_entities(self, analyzer):
        result = await analyzer._analyze_variable_flow([])
        assert result == []

    # ---- _analyze_state_mutations ----

    async def test_state_mutations_no_functions(self, analyzer):
        entities = [_make_entity(etype=EntityType.CLASS)]
        result = await analyzer._analyze_state_mutations(entities)
        assert result == []

    async def test_state_mutations_field_assignment(self, analyzer):
        """Test obj.field = ... pattern."""
        func_entity = _make_entity(
            id="f1",
            name="update",
            etype=EntityType.FUNCTION,
            properties={"content": "user.name = 'test'"},
        )
        user_entity = _make_entity(
            id="u1",
            name="user",
            etype=EntityType.CLASS,
        )
        with patch(
            "processing.semantic_analysis.result_aggregator.EntityRelation",
            side_effect=lambda **kw: MagicMock(**kw),
        ):
            result = await analyzer._analyze_state_mutations([func_entity, user_entity])
            assert len(result) >= 1
            mutation_rels = [r for r in result if r.source_id == "f1" and r.target_id == "u1"]
            assert len(mutation_rels) >= 1

    async def test_state_mutations_array_assignment(self, analyzer):
        """Test obj[index] = ... pattern."""
        func_entity = _make_entity(
            id="f1",
            name="update",
            etype=EntityType.FUNCTION,
            properties={"content": "items[0] = 'new'"},
        )
        items_entity = _make_entity(
            id="v1",
            name="items",
            etype=EntityType.VARIABLE,
        )
        with patch(
            "processing.semantic_analysis.result_aggregator.EntityRelation",
            side_effect=lambda **kw: MagicMock(**kw),
        ):
            result = await analyzer._analyze_state_mutations([func_entity, items_entity])
            mutation_rels = [r for r in result if r.source_id == "f1" and r.target_id == "v1"]
            assert len(mutation_rels) >= 1

    async def test_state_mutations_empty(self, analyzer):
        result = await analyzer._analyze_state_mutations([])
        assert result == []

    # ---- _analyze_memory_sharing ----

    async def test_memory_sharing_no_patterns(self, analyzer):
        entities = [_make_entity(properties={"content": "x = 1"})]
        result = await analyzer._analyze_memory_sharing(entities)
        assert result == []

    async def test_memory_sharing_shared_ptr(self, analyzer):
        """Two entities using shared_ptr of the same type."""
        e1 = _make_entity(
            id="e1",
            name="f1",
            properties={"content": "shared_ptr<Widget> w;"},
        )
        e2 = _make_entity(
            id="e2",
            name="f2",
            properties={"content": "shared_ptr<Widget> w2;"},
        )
        with patch(
            "processing.semantic_analysis.result_aggregator.EntityRelation",
            side_effect=lambda **kw: MagicMock(**kw),
        ):
            result = await analyzer._analyze_memory_sharing([e1, e2])
            # Each entity should see the other has the same pattern
            assert len(result) >= 1
            # Verify memory sharing relationship
            ids_involved = {(r.source_id, r.target_id) for r in result}
            assert ("e1", "e2") in ids_involved or ("e2", "e1") in ids_involved

    async def test_memory_sharing_rust_arc(self, analyzer):
        """Two entities using Arc<Type>."""
        e1 = _make_entity(
            id="e1",
            name="f1",
            properties={"content": "let x: Arc<Mutex> = Arc::new(m);"},
        )
        e2 = _make_entity(
            id="e2",
            name="f2",
            properties={"content": "let y: Arc<Mutex> = x.clone();"},
        )
        with patch(
            "processing.semantic_analysis.result_aggregator.EntityRelation",
            side_effect=lambda **kw: MagicMock(**kw),
        ):
            result = await analyzer._analyze_memory_sharing([e1, e2])
            assert len(result) >= 1

    async def test_memory_sharing_rust_mut_ref(self, analyzer):
        """Entities using &mut reference."""
        e1 = _make_entity(
            id="e1",
            name="f1",
            signature="fn f1(x: &mut Data)",
            properties={"content": ""},
        )
        e2 = _make_entity(
            id="e2",
            name="f2",
            signature="fn f2(y: &mut Data)",
            properties={"content": ""},
        )
        with patch(
            "processing.semantic_analysis.result_aggregator.EntityRelation",
            side_effect=lambda **kw: MagicMock(**kw),
        ):
            result = await analyzer._analyze_memory_sharing([e1, e2])
            assert len(result) >= 1

    async def test_memory_sharing_no_match_across_entities(self, analyzer):
        """One entity uses shared_ptr, another doesn't."""
        e1 = _make_entity(
            id="e1",
            name="f1",
            properties={"content": "shared_ptr<Widget> w;"},
        )
        e2 = _make_entity(
            id="e2",
            name="f2",
            properties={"content": "int x = 42;"},
        )
        result = await analyzer._analyze_memory_sharing([e1, e2])
        assert result == []

    async def test_memory_sharing_empty(self, analyzer):
        result = await analyzer._analyze_memory_sharing([])
        assert result == []

    async def test_memory_sharing_single_entity(self, analyzer):
        """Single entity with shared_ptr -- no other entity to share with."""
        e1 = _make_entity(
            id="e1",
            name="f1",
            properties={"content": "shared_ptr<Widget> w;"},
        )
        result = await analyzer._analyze_memory_sharing([e1])
        assert result == []


# ---------------------------------------------------------------------------
# Integration-style: aggregate with multiple entity types
# ---------------------------------------------------------------------------


class TestResultAggregatorIntegration:
    def test_full_aggregation_with_mixed_entities(self):
        aggregator = ResultAggregator()
        entities = [
            _make_entity(id="c1", name="UserService", etype=EntityType.CLASS, language="python"),
            _make_entity(
                id="f1",
                name="get_user",
                etype=EntityType.FUNCTION,
                language="python",
                properties={
                    "content": "if user_id:\n  return db.query()",
                    "parent_class": "UserService",
                },
            ),
            _make_entity(
                id="f2",
                name="save_user",
                etype=EntityType.FUNCTION,
                language="python",
                properties={"content": "db.save(user)", "parent_class": "UserService"},
            ),
        ]
        rels = [
            _make_relation(source_id="c1", target_id="f1", rel_type=RelationType.CONTAINS),
            _make_relation(source_id="c1", target_id="f2", rel_type=RelationType.CONTAINS),
            _make_relation(source_id="f1", target_id="f2", rel_type=RelationType.CALLS),
        ]
        violations = [
            _make_violation(entity_id="f1", severity=RuleSeverity.MEDIUM),
        ]
        ctx = _make_context(file_path="src/app.py", imports={"os", "db"}, exports={"UserService"})

        result = aggregator.aggregate_results(
            entities=entities,
            relationships=rels,
            violations=violations,
            symbols={},
            scopes={},
            types={},
            type_constraints=[],
            contexts={"src/app.py": ctx},
        )

        assert result.summary["entity_count"] == 3
        assert result.summary["relationship_count"] == 3
        assert result.summary["violation_count"] == 1
        assert result.summary["language_breakdown"]["python"] == 3
        assert len(result.metrics) > 0

    def test_empty_aggregation(self):
        aggregator = ResultAggregator()
        result = aggregator.aggregate_results(
            entities=[],
            relationships=[],
            violations=[],
            symbols={},
            scopes={},
            types={},
            type_constraints=[],
            contexts={},
        )
        assert result.summary["entity_count"] == 0
        assert result.summary["relationship_count"] == 0
        assert result.metrics == []
