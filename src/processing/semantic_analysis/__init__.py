"""
Semantic Analysis Package

This package provides sophisticated semantic analysis capabilities that go beyond
basic syntax parsing to understand deeper relationships between code entities.

Key Components:
- SemanticAnalyzer: Main orchestrator for semantic analysis
- SemanticParser: Parses code patterns and extracts semantic information
- SymbolTableBuilder: Builds and manages symbol tables
- TypeSystemAnalyzer: Analyzes type relationships and inferences
- SemanticRuleValidator: Validates semantic rules and detects violations
- ResultAggregator: Aggregates and summarizes all analysis results
"""

# Result aggregation
from .result_aggregator import (
    AnalysisMetric,
    AnalysisMetricType,
    DataFlowAnalyzer,
    ResultAggregator,
    SemanticAnalysisResult,
)

# Main analyzer and factory
from .semantic_analyzer import SemanticAnalyzer, create_semantic_analyzer

# Core types and enums from semantic parser
from .semantic_parser import (
    CrossLanguagePattern,
    SemanticContext,
    SemanticRelationType,
)

# Semantic rule validation
from .semantic_rules import (
    ArchitecturalPatternAnalyzer,
    RuleSeverity,
    RuleViolation,
    RuleViolationType,
    SemanticRule,
    SemanticRuleValidator,
)

# Symbol table types
from .symbol_table import (
    Scope,
    ScopeType,
    Symbol,
    SymbolTableBuilder,
    SymbolType,
)

# Type inference types
from .type_inference import (
    TypeConstraint,
    TypeInfo,
    TypeRelationship,
    TypeSystemAnalyzer,
)

__all__ = [
    # Main analyzer
    "SemanticAnalyzer",
    "create_semantic_analyzer",
    # Core types from semantic parser
    "SemanticRelationType",
    "SemanticContext",
    "CrossLanguagePattern",
    # Symbol table
    "SymbolTableBuilder",
    "Symbol",
    "Scope",
    "ScopeType",
    "SymbolType",
    # Type inference
    "TypeSystemAnalyzer",
    "TypeInfo",
    "TypeRelationship",
    "TypeConstraint",
    # Rule validation
    "SemanticRuleValidator",
    "ArchitecturalPatternAnalyzer",
    "SemanticRule",
    "RuleViolation",
    "RuleViolationType",
    "RuleSeverity",
    # Result aggregation
    "ResultAggregator",
    "SemanticAnalysisResult",
    "AnalysisMetric",
    "AnalysisMetricType",
    "DataFlowAnalyzer",
]
