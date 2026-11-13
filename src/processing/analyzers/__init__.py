"""Code Analysis Package.

This package provides various code analyzers for different programming languages
and analysis purposes including static analysis, documentation generation,
test coverage analysis, and symbolic validation.
"""

# Core analyzers
from .analyzer_factory import AnalyzerFactory

# Base classes
from .base_analyzer import BaseCodeProcessor

# Call graph analysis
from .call_graph_analyzer import CallGraphConfig, CallGraphGenerator
from .code_analyzer import CodeAnalyzer
from .doc_extractor import DocumentationExtractor
from .doc_generator import DocumentationGenerator
from .doc_validator import DocumentationValidator

# Documentation analyzers
from .documentation_engine import DocumentationConfig, DocumentationEngine
from .file_analyzer import FileAnalyzer
from .format_converter import FormatConverter

# Integration tests
from .integration_tests import StaticAnalysisIntegrationTests

# Language-specific analyzers
from .javascript_analyzer import JSAnalyzerProcessor
from .python_analyzer import PythonCodeProcessor

# Symbolic analysis (re-export from subpackage)
from .symbolic import SymbolicValidator, ValidationConfig

# Test analysis
from .test_analysis import (
    FailureCategory,
    QualityMetrics,
    RegressionDetector,
    TestAnalysisResult,
    TestPatternDetector,
    TestPatternMatch,
    TestQualityAnalyzer,
    TestResultAnalyzer,
    TestTrend,
)
from .test_coverage import (
    ComponentCoverageAnalyzer,
    CoverageAnalyzer,
    CoverageGap,
    CoverageLevel,
    CoverageMetric,
    CoverageReport,
    CoverageType,
    EdgeCaseCoverageAnalyzer,
    FunctionalCoverageAnalyzer,
    ScenarioCoverageAnalyzer,
)
from .test_discovery import (
    TestCodeGenerator,
    TestFileManager,
    TestProject,
    TestScenario,
    TestScenarioParser,
    TestScenarioType,
)
from .test_execution import (
    ComponentTestConfig,
    ComponentTestRunner,
    IntegrationTestExecutor,
    TestEnvironmentManager,
    TestExecutionResult,
    TestExecutionStatus,
)
from .test_reporting import (
    HTMLReportGenerator,
    JSONReportGenerator,
    MarkdownReportGenerator,
    ReportConfig,
    ReportFormat,
    ReportLevel,
    ReportMetadata,
    TestReport,
    TestReportGenerator,
)
from .typescript_analyzer import (
    APITypeScriptProcessor,
    TypeSafetyMetrics,
    TypeScriptComplexityMetrics,
)

__all__ = [
    # Base classes
    "BaseCodeProcessor",
    # Core analyzers
    "AnalyzerFactory",
    "CodeAnalyzer",
    "FileAnalyzer",
    # Language analyzers
    "JSAnalyzerProcessor",
    "PythonCodeProcessor",
    "APITypeScriptProcessor",
    "TypeSafetyMetrics",
    "TypeScriptComplexityMetrics",
    # Documentation
    "DocumentationEngine",
    "DocumentationConfig",
    "DocumentationExtractor",
    "DocumentationGenerator",
    "DocumentationValidator",
    "FormatConverter",
    # Call graph
    "CallGraphConfig",
    "CallGraphGenerator",
    # Test analysis
    "TestResultAnalyzer",
    "TestPatternDetector",
    "TestQualityAnalyzer",
    "RegressionDetector",
    "FailureCategory",
    "TestTrend",
    "TestAnalysisResult",
    "TestPatternMatch",
    "QualityMetrics",
    # Test coverage
    "CoverageAnalyzer",
    "FunctionalCoverageAnalyzer",
    "ComponentCoverageAnalyzer",
    "ScenarioCoverageAnalyzer",
    "EdgeCaseCoverageAnalyzer",
    "CoverageType",
    "CoverageLevel",
    "CoverageMetric",
    "CoverageGap",
    "CoverageReport",
    # Test discovery
    "TestCodeGenerator",
    "TestFileManager",
    "TestScenarioParser",
    "TestScenario",
    "TestProject",
    "TestScenarioType",
    # Test execution
    "ComponentTestRunner",
    "IntegrationTestExecutor",
    "TestEnvironmentManager",
    "TestExecutionStatus",
    "TestExecutionResult",
    "ComponentTestConfig",
    # Test reporting
    "TestReportGenerator",
    "MarkdownReportGenerator",
    "HTMLReportGenerator",
    "JSONReportGenerator",
    "ReportFormat",
    "ReportLevel",
    "ReportConfig",
    "ReportMetadata",
    "TestReport",
    # Symbolic analysis
    "SymbolicValidator",
    "ValidationConfig",
    # Integration tests
    "StaticAnalysisIntegrationTests",
]
