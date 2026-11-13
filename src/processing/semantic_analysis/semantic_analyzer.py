"""
Semantic Analysis for Advanced Cross-Language Relationships

This module provides sophisticated semantic analysis capabilities that go beyond
basic syntax parsing to understand deeper relationships between code entities.

This is the main orchestrator that coordinates different semantic analysis components.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from models.core_models import (
    CoreCodeEntity,
    EntityRelation,
)
from processing.semantic_analysis.result_aggregator import (
    ResultAggregator,
    SemanticAnalysisResult,
)

# Import modular components
from processing.semantic_analysis.semantic_parser import (
    CrossLanguagePattern,
    SemanticContext,
    SemanticParser,
    SemanticRelationType,
)
from processing.semantic_analysis.semantic_rules import (
    ArchitecturalPatternAnalyzer,
    SemanticRuleValidator,
)
from processing.semantic_analysis.symbol_table import SymbolTableBuilder
from processing.semantic_analysis.type_inference import TypeSystemAnalyzer

logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """Advanced semantic analyzer for cross-language relationship detection.

    This is the main orchestrator that coordinates different semantic analysis components:
    - SemanticParser: Parses code patterns and extracts semantic information
    - SymbolTableBuilder: Builds and manages symbol tables
    - TypeSystemAnalyzer: Analyzes type relationships and inferences
    - SemanticRuleValidator: Validates semantic rules and detects violations
    - ArchitecturalPatternAnalyzer: Analyzes architectural patterns
    - ResultAggregator: Aggregates and summarizes all analysis results
    """

    def __init__(self) -> None:
        """Initialize the semantic analyzer with all component analyzers."""
        self.semantic_parser = SemanticParser()
        self.symbol_table_builder = SymbolTableBuilder()
        self.type_system_analyzer = TypeSystemAnalyzer()
        self.rule_validator = SemanticRuleValidator()
        self.architectural_analyzer = ArchitecturalPatternAnalyzer()
        self.result_aggregator = ResultAggregator()

    async def analyze_semantic_relationships(
        self,
        entities: list[CoreCodeEntity],
        basic_relationships: list[EntityRelation],
        contexts: dict[str, SemanticContext],
    ) -> SemanticAnalysisResult:
        """Analyze and detect advanced semantic relationships between code entities.

        Args:
            entities: List of code entities to analyze
            basic_relationships: List of basic syntactic relationships
            contexts: Language-specific semantic contexts

        Returns:
            SemanticAnalysisResult containing all analysis results and metrics

        Note:
            This is the main entry point for semantic analysis. It coordinates
            various specialized analyzers to detect complex relationships.
        """
        logger.info(f"Starting semantic analysis for {len(entities)} entities")

        # Step 1: Build symbol table
        logger.debug("Building symbol table")
        symbol_table = self.symbol_table_builder.build_from_entities(entities)

        # Step 2: Cross-language relationship analysis
        logger.debug("Analyzing cross-language relationships")
        cross_lang_rels = await self._analyze_cross_language_relationships(entities)

        # Step 3: Type system analysis
        logger.debug("Analyzing type relationships")
        type_rels = await self.type_system_analyzer.analyze_type_relationships(entities, contexts)

        # Step 4: Data flow analysis
        logger.debug("Analyzing data flow")
        data_flow_rels = await self.result_aggregator.data_flow_analyzer.analyze_data_flow(
            entities, basic_relationships, contexts
        )

        # Step 5: Architectural pattern analysis
        logger.debug("Analyzing architectural patterns")
        arch_rels = await self._analyze_architectural_patterns(entities, contexts)

        # Step 6: Combine all relationships
        all_relationships = []
        all_relationships.extend(cross_lang_rels)
        all_relationships.extend(type_rels)
        all_relationships.extend(data_flow_rels)
        all_relationships.extend(arch_rels)
        all_relationships.extend(basic_relationships)  # Include basic relationships

        # Step 7: Validate semantic rules
        logger.debug("Validating semantic rules")
        violations = self.rule_validator.validate_entities(entities, all_relationships, contexts)

        # Step 8: Aggregate results
        logger.debug("Aggregating analysis results")
        result = self.result_aggregator.aggregate_results(
            entities=entities,
            relationships=all_relationships,
            violations=violations,
            symbols=symbol_table.symbols,
            scopes=symbol_table.scopes,
            types=self.type_system_analyzer.type_registry,
            type_constraints=self.type_system_analyzer.type_constraints,
            contexts=contexts,
        )

        logger.info(
            f"Semantic analysis completed. Found {len(all_relationships)} relationships, {len(violations)} violations"
        )

        return result

    async def _analyze_cross_language_relationships(
        self, entities: list[CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Detect relationships that cross language boundaries.

        Args:
            entities: List of code entities to analyze

        Returns:
            List of EntityRelation objects representing cross-language relationships

        Note:
            Uses predefined cross-language patterns to identify relationships
            between entities written in different programming languages.
        """
        relationships = []

        # Group entities by language
        entities_by_language = defaultdict(list)
        for entity in entities:
            entities_by_language[entity.language].append(entity)

        # Analyze each cross-language pattern
        for pattern in self.semantic_parser.cross_language_patterns:
            pattern_relationships = await self._apply_cross_language_pattern(
                pattern, entities_by_language
            )
            relationships.extend(pattern_relationships)

        return relationships

    async def _apply_cross_language_pattern(
        self,
        pattern: CrossLanguagePattern,
        entities_by_language: dict[str, list[CoreCodeEntity]],
    ) -> list[EntityRelation]:
        """Apply a specific cross-language pattern to detect relationships.

        Args:
            pattern: CrossLanguagePattern to apply
            entities_by_language: Entities grouped by language

        Returns:
            List of EntityRelation objects matching the pattern

        Note:
            Each pattern defines source/target regex patterns and language constraints
            to identify specific types of cross-language relationships.
        """
        relationships = []

        # Find source entities matching source patterns
        source_entities = []
        for language, entities in entities_by_language.items():
            if pattern.source_languages and language not in pattern.source_languages:
                continue

            for entity in entities:
                if self.semantic_parser.parse_entity_patterns(entity, pattern.source_patterns):
                    source_entities.append(entity)

        # Find target entities matching target patterns
        target_entities = []
        for language, entities in entities_by_language.items():
            if pattern.target_languages and language not in pattern.target_languages:
                continue

            for entity in entities:
                if self.semantic_parser.parse_entity_patterns(entity, pattern.target_patterns):
                    target_entities.append(entity)

        # Create relationships between matching entities
        for source in source_entities:
            for target in target_entities:
                if self.semantic_parser.analyze_entity_relationships(source, target, pattern):
                    relationships.append(
                        EntityRelation(
                            source_id=source.id,
                            target_id=target.id,
                            type=pattern.relationship_type.value,
                            properties={
                                "pattern_name": pattern.name,
                                "confidence": pattern.confidence_score,
                                "description": pattern.description,
                                "source_language": source.language,
                                "target_language": target.language,
                            },
                        )
                    )

        return relationships

    async def _analyze_architectural_patterns(
        self,
        entities: list[CoreCodeEntity],
        contexts: dict[str, SemanticContext],
    ) -> list[EntityRelation]:
        """Analyze architectural patterns and service relationships.

        Args:
            entities: List of code entities to analyze
            contexts: Language-specific semantic contexts for each entity

        Note:
            Detects service boundaries, API gateways, and other architectural
            patterns that span multiple code entities and languages.
        """
        relationships = []

        # Service discovery patterns
        service_rels = self.architectural_analyzer.analyze_service_patterns(entities, contexts)
        relationships.extend(service_rels)

        # Configuration dependencies
        config_rels = self.architectural_analyzer.analyze_configuration_dependencies(entities)
        relationships.extend(config_rels)

        # Event-driven patterns
        event_rels = self.architectural_analyzer.analyze_event_patterns(entities)
        relationships.extend(event_rels)

        return relationships

    def add_custom_rule(self, rule):
        """Add a custom semantic validation rule.

        Args:
            rule: SemanticRule object to add
        """
        self.rule_validator.add_custom_rule(rule)

    def disable_rule(self, rule_name: str):
        """Disable a semantic validation rule.

        Args:
            rule_name: Name of the rule to disable
        """
        self.rule_validator.disable_rule(rule_name)

    def get_analysis_summary(self, result: SemanticAnalysisResult) -> dict:
        """Get a high-level summary of analysis results.

        Args:
            result: Analysis result to summarize

        Returns:
            Dictionary containing summary statistics
        """
        return result.summary

    def filter_results_by_file(
        self, result: SemanticAnalysisResult, file_path: str
    ) -> SemanticAnalysisResult:
        """Filter analysis results to a specific file.

        Args:
            result: Analysis result to filter
            file_path: File path to filter by

        Returns:
            Filtered analysis result
        """
        return self.result_aggregator.filter_results_by_file(result, file_path)

    def get_cross_language_patterns(self) -> list[CrossLanguagePattern]:
        """Get all cross-language patterns used for analysis.

        Returns:
            List of CrossLanguagePattern objects
        """
        return self.semantic_parser.cross_language_patterns

    def extract_architectural_patterns(
        self, entities: list[CoreCodeEntity]
    ) -> dict[str, list[CoreCodeEntity]]:
        """Extract architectural patterns from entities.

        Args:
            entities: List of entities to analyze

        Returns:
            Dictionary mapping pattern types to matching entities
        """
        return self.semantic_parser.extract_architectural_patterns(entities)


# Factory function for creating semantic analyzer
def create_semantic_analyzer() -> SemanticAnalyzer:
    """Create and configure a semantic analyzer instance.

    Returns:
        Configured SemanticAnalyzer instance
    """
    return SemanticAnalyzer()


# Legacy compatibility exports
# These are maintained for backward compatibility with existing code

# Re-export for backward compatibility
__all__ = [
    "SemanticAnalyzer",
    "SemanticAnalysisResult",
    "SemanticRelationType",
    "SemanticContext",
    "CrossLanguagePattern",
    "create_semantic_analyzer",
]
