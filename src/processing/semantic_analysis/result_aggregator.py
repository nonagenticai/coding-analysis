"""
Analysis Result Aggregation Module

This module provides functionality for aggregating and combining results from
different semantic analysis components into a unified view.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    EntityRelation,
    EntityType,
)

from .semantic_parser import SemanticContext, SemanticRelationType
from .semantic_rules import RuleSeverity, RuleViolation
from .symbol_table import Scope, Symbol
from .type_inference import TypeConstraint, TypeInfo

logger = logging.getLogger(__name__)


class AnalysisMetricType(StrEnum):
    """Types of analysis metrics."""

    RELATIONSHIP_COUNT = "relationship_count"
    ENTITY_COUNT = "entity_count"
    VIOLATION_COUNT = "violation_count"
    COMPLEXITY_SCORE = "complexity_score"
    COUPLING_METRIC = "coupling_metric"
    COHESION_METRIC = "cohesion_metric"
    MAINTAINABILITY_INDEX = "maintainability_index"


@dataclass
class AnalysisMetric:
    """Represents an analysis metric."""

    name: str
    metric_type: AnalysisMetricType
    value: float
    description: str
    file_path: str | None = None
    entity_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticAnalysisResult:
    """Aggregated result of semantic analysis."""

    entities: list[CoreCodeEntity] = field(default_factory=list)
    relationships: list[EntityRelation] = field(default_factory=list)
    violations: list[RuleViolation] = field(default_factory=list)
    metrics: list[AnalysisMetric] = field(default_factory=list)

    # Detailed analysis results
    symbols: dict[str, Symbol] = field(default_factory=dict)
    scopes: dict[str, Scope] = field(default_factory=dict)
    types: dict[str, TypeInfo] = field(default_factory=dict)
    type_constraints: list[TypeConstraint] = field(default_factory=list)

    # Summary statistics
    summary: dict[str, Any] = field(default_factory=dict)


class DataFlowAnalyzer:
    """Analyzer for data flow and state mutation relationships."""

    async def analyze_data_flow(
        self,
        entities: list[CoreCodeEntity],
        basic_relationships: list[EntityRelation],
        contexts: dict[str, SemanticContext],
    ) -> list[EntityRelation]:
        """Analyze data flow relationships between entities.

        Args:
            entities: List of code entities to analyze
            basic_relationships: Basic syntactic relationships
            contexts: Language-specific semantic contexts

        Returns:
            List of EntityRelation objects representing data flow
        """
        relationships = []

        # Variable data flow
        var_flow_rels = await self._analyze_variable_flow(entities)
        relationships.extend(var_flow_rels)

        # State mutation analysis
        mutation_rels = await self._analyze_state_mutations(entities)
        relationships.extend(mutation_rels)

        # Memory sharing analysis
        memory_rels = await self._analyze_memory_sharing(entities)
        relationships.extend(memory_rels)

        return relationships

    async def _analyze_variable_flow(self, entities: list[CoreCodeEntity]) -> list[EntityRelation]:
        """Analyze how data flows between variables and functions."""
        relationships = []

        assignment_patterns = [
            r"(\w+)\s*=\s*(\w+)\.(\w+)\s*\(",  # var = obj.method()
            r"(\w+)\s*=\s*(\w+)\s*\(",  # var = func()
            r"return\s+(\w+)",  # return var
        ]

        for entity in entities:
            if entity.type == EntityType.FUNCTION:
                search_text = entity.properties.get("content", "")

                for pattern in assignment_patterns:
                    matches = re.finditer(pattern, search_text, re.IGNORECASE)
                    for match in matches:
                        if match.lastindex is not None and match.lastindex >= 2:
                            target_name = match.group(2)

                            for target_entity in entities:
                                if target_entity.name == target_name:
                                    relationships.append(
                                        EntityRelation(
                                            source_id=entity.id,
                                            target_id=target_entity.id,
                                            type=SemanticRelationType.DATA_DEPENDENCY.value,
                                            properties={
                                                "pattern": "data_flow",
                                                "flow_type": "assignment",
                                            },
                                        )
                                    )
                                    break

        return relationships

    async def _analyze_state_mutations(
        self, entities: list[CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Analyze state mutation patterns."""
        relationships = []

        mutation_patterns = [
            r"(\w+)\.(\w+)\s*=",  # obj.field =
            r"(\w+)\[.*\]\s*=",  # array[index] =
            r"(\w+)\.(\w+)\s*\(\s*.*\s*\)",  # obj.mutator_method()
        ]

        for entity in entities:
            if entity.type == EntityType.FUNCTION:
                search_text = entity.properties.get("content", "")

                for pattern in mutation_patterns:
                    matches = re.finditer(pattern, search_text, re.IGNORECASE)
                    for match in matches:
                        if match.lastindex is not None and match.lastindex >= 1:
                            mutated_object = match.group(1)

                            for target_entity in entities:
                                if (
                                    target_entity.type in [EntityType.CLASS, EntityType.VARIABLE]
                                    and mutated_object.lower() in target_entity.name.lower()
                                ):
                                    relationships.append(
                                        EntityRelation(
                                            source_id=entity.id,
                                            target_id=target_entity.id,
                                            type=SemanticRelationType.STATE_MUTATION.value,
                                            properties={
                                                "mutated_object": mutated_object,
                                                "mutation_pattern": pattern,
                                                "in_function": entity.name,
                                            },
                                        )
                                    )
                                    break

        return relationships

    async def _analyze_memory_sharing(self, entities: list[CoreCodeEntity]) -> list[EntityRelation]:
        """Analyze shared memory and resource access patterns."""
        relationships = []

        sharing_patterns = [
            r"shared_ptr<(\w+)>",  # C++ shared pointer
            r"Rc<(\w+)>",  # Rust reference counting
            r"Arc<(\w+)>",  # Rust atomic reference counting
            r"&mut\s+(\w+)",  # Rust mutable reference
        ]

        for entity in entities:
            search_text = f"{entity.signature or ''} {entity.properties.get('content', '')}"

            for pattern in sharing_patterns:
                matches = re.finditer(pattern, search_text, re.IGNORECASE)
                for match in matches:
                    shared_type = match.group(1)

                    for other_entity in entities:
                        if other_entity.id != entity.id:
                            other_text = f"{other_entity.signature or ''} {other_entity.properties.get('content', '')}"
                            if re.search(pattern, other_text, re.IGNORECASE):
                                relationships.append(
                                    EntityRelation(
                                        source_id=entity.id,
                                        target_id=other_entity.id,
                                        type=SemanticRelationType.MEMORY_SHARING.value,
                                        properties={
                                            "shared_type": shared_type,
                                            "sharing_pattern": pattern,
                                        },
                                    )
                                )

        return relationships


class ResultAggregator:
    """Aggregates results from different semantic analysis components."""

    def __init__(self):
        self.data_flow_analyzer = DataFlowAnalyzer()

    def aggregate_results(
        self,
        entities: list[CoreCodeEntity],
        relationships: list[EntityRelation],
        violations: list[RuleViolation],
        symbols: dict[str, Symbol],
        scopes: dict[str, Scope],
        types: dict[str, TypeInfo],
        type_constraints: list[TypeConstraint],
        contexts: dict[str, SemanticContext],
    ) -> SemanticAnalysisResult:
        """Aggregate all analysis results into a unified result.

        Args:
            entities: List of analyzed entities
            relationships: List of detected relationships
            violations: List of rule violations
            symbols: Symbol table entries
            scopes: Scope definitions
            types: Type information
            type_constraints: Type constraints
            contexts: Semantic contexts

        Returns:
            Aggregated semantic analysis result
        """
        result = SemanticAnalysisResult(
            entities=entities,
            relationships=relationships,
            violations=violations,
            symbols=symbols,
            scopes=scopes,
            types=types,
            type_constraints=type_constraints,
        )

        # Calculate metrics
        metrics = self._calculate_metrics(result, contexts)
        result.metrics = metrics

        # Generate summary
        summary = self._generate_summary(result)
        result.summary = summary

        return result

    def _calculate_metrics(
        self, result: SemanticAnalysisResult, contexts: dict[str, SemanticContext]
    ) -> list[AnalysisMetric]:
        """Calculate various analysis metrics.

        Args:
            result: Analysis result to calculate metrics for
            contexts: Semantic contexts

        Returns:
            List of calculated metrics
        """
        metrics = []

        # Basic counts
        metrics.extend(self._calculate_count_metrics(result))

        # Complexity metrics
        metrics.extend(self._calculate_complexity_metrics(result))

        # Quality metrics
        metrics.extend(self._calculate_quality_metrics(result))

        # File-level metrics
        metrics.extend(self._calculate_file_level_metrics(result, contexts))

        return metrics

    def _calculate_count_metrics(self, result: SemanticAnalysisResult) -> list[AnalysisMetric]:
        """Calculate basic count metrics."""
        metrics = []

        # Entity counts by type
        entity_counts = defaultdict(int)
        for entity in result.entities:
            entity_counts[entity.type.value] += 1

        for entity_type, count in entity_counts.items():
            metrics.append(
                AnalysisMetric(
                    name=f"{entity_type}_count",
                    metric_type=AnalysisMetricType.ENTITY_COUNT,
                    value=float(count),
                    description=f"Number of {entity_type} entities",
                )
            )

        # Relationship counts by type
        relationship_counts = defaultdict(int)
        for rel in result.relationships:
            relationship_counts[rel.type] += 1

        for rel_type, count in relationship_counts.items():
            metrics.append(
                AnalysisMetric(
                    name=f"{rel_type}_relationship_count",
                    metric_type=AnalysisMetricType.RELATIONSHIP_COUNT,
                    value=float(count),
                    description=f"Number of {rel_type} relationships",
                )
            )

        # Violation counts by severity
        violation_counts = defaultdict(int)
        for violation in result.violations:
            violation_counts[violation.severity.value] += 1

        for severity, count in violation_counts.items():
            metrics.append(
                AnalysisMetric(
                    name=f"{severity}_violation_count",
                    metric_type=AnalysisMetricType.VIOLATION_COUNT,
                    value=float(count),
                    description=f"Number of {severity} violations",
                )
            )

        return metrics

    def _calculate_complexity_metrics(self, result: SemanticAnalysisResult) -> list[AnalysisMetric]:
        """Calculate complexity metrics."""
        metrics = []

        # Calculate cyclomatic complexity approximation
        for entity in result.entities:
            if entity.type == EntityType.FUNCTION:
                complexity = self._estimate_cyclomatic_complexity(entity)
                metrics.append(
                    AnalysisMetric(
                        name="cyclomatic_complexity",
                        metric_type=AnalysisMetricType.COMPLEXITY_SCORE,
                        value=complexity,
                        description="Estimated cyclomatic complexity",
                        entity_id=entity.id,
                        file_path=entity.file_path,
                    )
                )

        # Calculate coupling metrics
        coupling_metrics = self._calculate_coupling_metrics(result)
        metrics.extend(coupling_metrics)

        # Calculate cohesion metrics
        cohesion_metrics = self._calculate_cohesion_metrics(result)
        metrics.extend(cohesion_metrics)

        return metrics

    def _calculate_quality_metrics(self, result: SemanticAnalysisResult) -> list[AnalysisMetric]:
        """Calculate code quality metrics."""
        metrics = []

        # Maintainability index approximation
        for entity in result.entities:
            if entity.type in [EntityType.CLASS, EntityType.FUNCTION]:
                maintainability = self._estimate_maintainability_index(entity, result)
                metrics.append(
                    AnalysisMetric(
                        name="maintainability_index",
                        metric_type=AnalysisMetricType.MAINTAINABILITY_INDEX,
                        value=maintainability,
                        description="Estimated maintainability index",
                        entity_id=entity.id,
                        file_path=entity.file_path,
                    )
                )

        return metrics

    def _calculate_file_level_metrics(
        self, result: SemanticAnalysisResult, contexts: dict[str, SemanticContext]
    ) -> list[AnalysisMetric]:
        """Calculate file-level metrics."""
        metrics = []

        # Group entities by file
        entities_by_file = defaultdict(list)
        for entity in result.entities:
            entities_by_file[entity.file_path].append(entity)

        # Calculate metrics for each file
        for file_path, file_entities in entities_by_file.items():
            # Lines of code approximation
            total_lines = sum(
                len(entity.properties.get("content", "").split("\n")) for entity in file_entities
            )

            metrics.append(
                AnalysisMetric(
                    name="lines_of_code",
                    metric_type=AnalysisMetricType.COMPLEXITY_SCORE,
                    value=float(total_lines),
                    description="Estimated lines of code",
                    file_path=file_path,
                )
            )

            # Number of dependencies (imports/exports)
            context = contexts.get(file_path)
            if context:
                import_count = len(context.imports)
                export_count = len(context.exports)

                metrics.append(
                    AnalysisMetric(
                        name="import_count",
                        metric_type=AnalysisMetricType.COUPLING_METRIC,
                        value=float(import_count),
                        description="Number of imports",
                        file_path=file_path,
                    )
                )

                metrics.append(
                    AnalysisMetric(
                        name="export_count",
                        metric_type=AnalysisMetricType.COUPLING_METRIC,
                        value=float(export_count),
                        description="Number of exports",
                        file_path=file_path,
                    )
                )

        return metrics

    def _estimate_cyclomatic_complexity(self, entity: CoreCodeEntity) -> float:
        """Estimate cyclomatic complexity for a function."""
        content = entity.properties.get("content", "")

        # Count decision points
        decision_keywords = [
            "if",
            "else",
            "while",
            "for",
            "switch",
            "case",
            "catch",
            "&&",
            "||",
            "?",
        ]
        complexity = 1  # Base complexity

        for keyword in decision_keywords:
            complexity += content.lower().count(keyword.lower())

        return float(complexity)

    def _calculate_coupling_metrics(self, result: SemanticAnalysisResult) -> list[AnalysisMetric]:
        """Calculate coupling metrics for entities."""
        metrics = []

        # Count incoming and outgoing relationships for each entity
        incoming_counts = defaultdict(int)
        outgoing_counts = defaultdict(int)

        for rel in result.relationships:
            outgoing_counts[rel.source_id] += 1
            incoming_counts[rel.target_id] += 1

        for entity in result.entities:
            if entity.type in [EntityType.CLASS, EntityType.FUNCTION]:
                incoming = incoming_counts[entity.id]
                outgoing = outgoing_counts[entity.id]
                total_coupling = incoming + outgoing

                metrics.append(
                    AnalysisMetric(
                        name="coupling_metric",
                        metric_type=AnalysisMetricType.COUPLING_METRIC,
                        value=float(total_coupling),
                        description="Total coupling (incoming + outgoing relationships)",
                        entity_id=entity.id,
                        file_path=entity.file_path,
                        metadata={"incoming": incoming, "outgoing": outgoing},
                    )
                )

        return metrics

    def _calculate_cohesion_metrics(self, result: SemanticAnalysisResult) -> list[AnalysisMetric]:
        """Calculate cohesion metrics for classes."""
        metrics = []

        for entity in result.entities:
            if entity.type == EntityType.CLASS:
                cohesion = self._estimate_class_cohesion(entity, result)
                metrics.append(
                    AnalysisMetric(
                        name="class_cohesion",
                        metric_type=AnalysisMetricType.COHESION_METRIC,
                        value=cohesion,
                        description="Estimated class cohesion",
                        entity_id=entity.id,
                        file_path=entity.file_path,
                    )
                )

        return metrics

    def _estimate_class_cohesion(
        self, class_entity: CoreCodeEntity, result: SemanticAnalysisResult
    ) -> float:
        """Estimate cohesion for a class based on internal relationships."""
        # Find methods and fields of this class
        class_members = [
            e
            for e in result.entities
            if e.file_path == class_entity.file_path
            and e.type in [EntityType.FUNCTION, EntityType.VARIABLE]
            and class_entity.name.lower() in e.properties.get("parent_class", "").lower()
        ]

        if len(class_members) <= 1:
            return 1.0

        # Count internal relationships between class members
        internal_relationships = 0
        for rel in result.relationships:
            source_is_member = any(m.id == rel.source_id for m in class_members)
            target_is_member = any(m.id == rel.target_id for m in class_members)

            if source_is_member and target_is_member:
                internal_relationships += 1

        # Calculate cohesion as ratio of internal relationships to possible relationships
        max_possible = len(class_members) * (len(class_members) - 1)
        if max_possible == 0:
            return 1.0

        return min(1.0, internal_relationships / max_possible)

    def _estimate_maintainability_index(
        self, entity: CoreCodeEntity, result: SemanticAnalysisResult
    ) -> float:
        """Estimate maintainability index for an entity."""
        # Simplified maintainability index calculation
        content = entity.properties.get("content", "")
        lines_of_code = len(content.split("\n"))

        # Count violations related to this entity
        entity_violations = [v for v in result.violations if v.entity_id == entity.id]

        # Base score
        maintainability = 100.0

        # Reduce for complexity
        if entity.type == EntityType.FUNCTION:
            complexity = self._estimate_cyclomatic_complexity(entity)
            maintainability -= (complexity - 1) * 2

        # Reduce for size
        maintainability -= max(0, (lines_of_code - 50) * 0.1)

        # Reduce for violations
        for violation in entity_violations:
            severity_penalties = {
                RuleSeverity.CRITICAL: 20,
                RuleSeverity.HIGH: 10,
                RuleSeverity.MEDIUM: 5,
                RuleSeverity.LOW: 2,
                RuleSeverity.INFO: 1,
            }
            maintainability -= severity_penalties.get(violation.severity, 1)

        return max(0.0, maintainability)

    def _generate_summary(self, result: SemanticAnalysisResult) -> dict[str, Any]:
        """Generate a summary of analysis results.

        Args:
            result: Analysis result to summarize

        Returns:
            Dictionary containing summary information
        """
        summary = {}

        # Basic counts
        summary["entity_count"] = len(result.entities)
        summary["relationship_count"] = len(result.relationships)
        summary["violation_count"] = len(result.violations)
        summary["symbol_count"] = len(result.symbols)
        summary["scope_count"] = len(result.scopes)
        summary["type_count"] = len(result.types)

        # Entity breakdown
        entity_breakdown = defaultdict(int)
        for entity in result.entities:
            entity_breakdown[entity.type.value] += 1
        summary["entity_breakdown"] = dict(entity_breakdown)

        # Relationship breakdown
        relationship_breakdown = defaultdict(int)
        for rel in result.relationships:
            relationship_breakdown[rel.type] += 1
        summary["relationship_breakdown"] = dict(relationship_breakdown)

        # Violation breakdown
        violation_breakdown = defaultdict(int)
        for violation in result.violations:
            violation_breakdown[violation.severity.value] += 1
        summary["violation_breakdown"] = dict(violation_breakdown)

        # Language breakdown
        language_breakdown = defaultdict(int)
        for entity in result.entities:
            language_breakdown[entity.language] += 1
        summary["language_breakdown"] = dict(language_breakdown)

        # Quality scores
        quality_metrics = [m for m in result.metrics if "maintainability" in m.name]
        if quality_metrics:
            avg_maintainability = sum(m.value for m in quality_metrics) / len(quality_metrics)
            summary["average_maintainability"] = avg_maintainability

        complexity_metrics = [m for m in result.metrics if "complexity" in m.name]
        if complexity_metrics:
            avg_complexity = sum(m.value for m in complexity_metrics) / len(complexity_metrics)
            summary["average_complexity"] = avg_complexity

        # Top issues
        critical_violations = [v for v in result.violations if v.severity == RuleSeverity.CRITICAL]
        summary["critical_violation_count"] = len(critical_violations)

        high_coupling_entities = [
            m
            for m in result.metrics
            if m.metric_type == AnalysisMetricType.COUPLING_METRIC and m.value > 10
        ]
        summary["high_coupling_entity_count"] = len(high_coupling_entities)

        return summary

    def filter_results_by_file(
        self, result: SemanticAnalysisResult, file_path: str
    ) -> SemanticAnalysisResult:
        """Filter analysis results to include only a specific file.

        Args:
            result: Analysis result to filter
            file_path: File path to filter by

        Returns:
            Filtered analysis result
        """
        filtered_entities = [e for e in result.entities if e.file_path == file_path]
        entity_ids = {e.id for e in filtered_entities}

        filtered_relationships = [
            r
            for r in result.relationships
            if r.source_id in entity_ids or r.target_id in entity_ids
        ]

        filtered_violations = [v for v in result.violations if v.file_path == file_path]

        filtered_symbols = {k: v for k, v in result.symbols.items() if v.file_path == file_path}

        filtered_scopes = {k: v for k, v in result.scopes.items() if v.file_path == file_path}

        filtered_metrics = [
            m for m in result.metrics if m.file_path == file_path or m.entity_id in entity_ids
        ]

        return SemanticAnalysisResult(
            entities=filtered_entities,
            relationships=filtered_relationships,
            violations=filtered_violations,
            metrics=filtered_metrics,
            symbols=filtered_symbols,
            scopes=filtered_scopes,
            types=result.types,  # Types are global
            type_constraints=result.type_constraints,  # Type constraints are global
            summary=self._generate_summary(
                SemanticAnalysisResult(
                    entities=filtered_entities,
                    relationships=filtered_relationships,
                    violations=filtered_violations,
                    metrics=filtered_metrics,
                )
            ),
        )
