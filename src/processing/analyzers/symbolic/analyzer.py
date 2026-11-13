"""Symbolic Analysis and Advanced Validation Analyzer.

This module provides the main symbolic analyzer implementation for advanced
code validation and security analysis. It integrates multiple analysis tools
and techniques to provide comprehensive code safety assessment.

The analyzer supports:
- Symbolic execution using CrossHair
- Taint analysis using Pysa
- Formal verification techniques
- Safety score calculation
- Security vulnerability detection
- Code change impact assessment

The analyzer works by:
1. Accepting code files, directories, or snippets
2. Running configured validation types (symbolic execution, taint analysis, etc.)
3. Aggregating results from multiple tools
4. Calculating overall safety scores
5. Generating actionable recommendations

Classes:
    SymbolicValidator: Main validator class implementing BaseCodeProcessor interface

Example:
    # Basic usage
    validator = SymbolicValidator()
    result = await validator.analyze_file("suspicious_code.py")

    # Advanced usage with configuration
    config = ValidationConfig(
        validation_types=[ValidationType.SYMBOLIC_EXECUTION, ValidationType.TAINT_ANALYSIS],
        timeout=300
    )
    validator = SymbolicValidator(config)
    validation_result = await validator.validate_code_changes(changes)
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from models.core_models import CoreAnalysisResult
from models.static_analysis_models import (
    AdvancedValidationResponse,
    CodeChange,
    ValidationType,
)
from processing.analyzers.base_analyzer import BaseCodeProcessor
from processing.quality_analysis.tool_registry import get_tool_registry

from .config import SymbolicAnalysisConfig as ValidationConfig
from .exceptions import (
    CrossHairAnalysisError,
    FileProcessingError,
    FormalVerificationError,
    PysaAnalysisError,
    SymbolicAnalysisError,
)
from .exceptions import SymbolicValidationError as ValidationError
from .tools import CrossHairAnalyzer, FormalVerifier, PysaAnalyzer
from .utils import SafetyCalculator, SymbolicRecommendationGenerator, ValidationHelper

logger = logging.getLogger(__name__)


class SymbolicValidator(BaseCodeProcessor):
    """Validator for symbolic execution and advanced code validation.

    This validator provides comprehensive code safety assessment using multiple
    analysis techniques including symbolic execution, taint analysis, and
    formal verification. It integrates various tools to detect potential
    runtime errors, security vulnerabilities, and code quality issues.

    The validator supports:
    - Symbolic execution for runtime error detection
    - Taint analysis for security vulnerability detection
    - Formal verification for logical correctness
    - Safety score calculation and risk assessment
    - Code change validation and impact analysis

    Attributes:
        config (ValidationConfig): Configuration for analysis parameters
        tool_registry: Registry of available analysis tools
        crosshair_analyzer: CrossHair symbolic execution analyzer
        pysa_analyzer: Pysa taint analysis analyzer
        formal_verifier: Formal verification analyzer
        safety_calculator: Safety score calculation utility
        recommendation_generator: Recommendation generation utility
        validation_helper: Validation helper utilities

    Example:
        # Basic analysis
        validator = SymbolicValidator()
        result = await validator.analyze_file("code.py")

        # Code change validation
        changes = [CodeChange(...)]
        validation = await validator.validate_code_changes(changes)
    """

    def __init__(self, config: ValidationConfig | None = None):
        """Initialize symbolic validator with configuration.

        Args:
            config (Optional[ValidationConfig]): Configuration for analysis
                                               parameters. If None, uses default
                                               configuration.
        """
        self.config = config or ValidationConfig()
        self.tool_registry = get_tool_registry()

        # Initialize tools and utilities
        self.crosshair_analyzer = CrossHairAnalyzer(self.tool_registry)
        self.pysa_analyzer = PysaAnalyzer(self.tool_registry)
        self.formal_verifier = FormalVerifier()
        self.safety_calculator = SafetyCalculator()
        self.recommendation_generator = SymbolicRecommendationGenerator()
        self.validation_helper = ValidationHelper()

    def get_supported_extensions(self) -> list[str]:
        """Return supported file extensions for symbolic analysis.

        Returns:
            List[str]: List containing [".py"] as symbolic analysis
                      currently focuses on Python files.
        """
        return [".py"]  # Focus on Python for initial implementation

    def get_language_name(self) -> str:
        """Return the programming language name.

        Returns:
            str: Always returns "python" as the analyzer currently
                 supports Python files only.
        """
        return "python"

    async def analyze_file(self, file_path: str) -> CoreAnalysisResult:
        """Analyze a single file for symbolic execution and validation.

        Performs comprehensive symbolic analysis on a Python file using
        the configured validation types (symbolic execution, taint analysis,
        formal verification). Results are aggregated and returned with
        safety assessment.

        Args:
            file_path (str): Absolute path to the Python file to analyze.

        Returns:
            CoreAnalysisResult: Analysis results containing:
                - Success/failure status
                - Symbolic analysis results in metadata
                - Empty entities and relationships (not applicable for symbolic analysis)
                - Error information if analysis failed
        """
        try:
            validation_data = await self.run_symbolic_analysis_on_file(file_path)

            return CoreAnalysisResult(
                success=True,
                language="python",
                file_path=file_path,
                entities=[],
                relationships=[],
                complexity_metrics=None,
                error=None,
                parse_time_ms=0.0,
                total_lines=0,
                metadata={
                    "symbolic_analysis": validation_data,
                    "analyzer": "symbolic",
                    "file_path": file_path,
                },
            )
        except OSError as e:
            logger.error("Error reading file %s: %s", file_path, e)
            return self._create_error_result(file_path, str(e))
        except SymbolicAnalysisError as e:
            logger.error("Symbolic analysis error for %s: %s", file_path, e)
            return self._create_error_result(file_path, str(e))

    async def analyze_directory(self, directory_path: str) -> CoreAnalysisResult:
        """Analyze a directory for symbolic execution and validation."""
        try:
            validation_data = await self.run_symbolic_analysis_on_directory(directory_path)

            return CoreAnalysisResult(
                success=True,
                language="python",
                file_path=directory_path,
                entities=[],
                relationships=[],
                complexity_metrics=None,
                error=None,
                parse_time_ms=0.0,
                total_lines=0,
                metadata={
                    "symbolic_analysis": validation_data,
                    "analyzer": "symbolic",
                    "directory_path": directory_path,
                },
            )
        except OSError as e:
            logger.error("Error reading directory %s: %s", directory_path, e)
            return self._create_error_result(directory_path, str(e))
        except SymbolicAnalysisError as e:
            logger.error("Symbolic analysis error for %s: %s", directory_path, e)
            return self._create_error_result(directory_path, str(e))

    async def analyze_snippet(
        self, code_snippet: str, file_path: str | None = None
    ) -> CoreAnalysisResult:
        """Analyze a code snippet for symbolic execution and validation."""
        language_hint = file_path  # file_path used as language hint in this override
        if language_hint and language_hint != "python":
            return self._create_error_result("snippet", f"Language {language_hint} not supported")

        try:
            # Create temporary file for analysis
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as tmp_file:
                tmp_file.write(code_snippet)
                tmp_file_path = tmp_file.name

            try:
                # Run analysis on temporary file
                validation_data = await self.run_symbolic_analysis_on_file(tmp_file_path)

                return CoreAnalysisResult(
                    success=True,
                    language="python",
                    file_path="snippet",
                    entities=[],
                    relationships=[],
                    complexity_metrics=None,
                    error=None,
                    parse_time_ms=0.0,
                    total_lines=len(code_snippet.splitlines()),
                    metadata={
                        "symbolic_analysis": validation_data,
                        "analyzer": "symbolic",
                        "snippet": True,
                    },
                )
            finally:
                os.unlink(tmp_file_path)

        except OSError as e:
            logger.error("Error processing snippet: %s", e)
            return self._create_error_result("snippet", str(e))
        except SymbolicAnalysisError as e:
            logger.error("Symbolic analysis error for snippet: %s", e)
            return self._create_error_result("snippet", str(e))

    async def validate_code_changes(
        self,
        code_changes: list[CodeChange],
        config: ValidationConfig | None = None,
    ) -> AdvancedValidationResponse:
        """Validate code changes using symbolic execution and security analysis.

        This method performs comprehensive validation of code changes using
        multiple analysis techniques to assess safety and security implications.
        It provides detailed feedback on potential issues and recommendations.

        Args:
            code_changes (List[CodeChange]): List of code changes to validate.
                                           Each change contains original code,
                                           modified code, and metadata.
            config (Optional[ValidationConfig]): Optional configuration override
                                               for this validation run. If None,
                                               uses the analyzer's default config.

        Returns:
            AdvancedValidationResponse: Comprehensive validation results including:
                - Individual validation results for each change
                - Overall safety score calculation
                - Blocking issues that prevent deployment
                - Actionable recommendations for improvements
                - Analysis metadata and tool information

        Raises:
            ValidationError: If validation process fails or encounters
                           unrecoverable errors during analysis.
        """
        validation_config = config or self.config
        analysis_id = f"adv_{hash(str(code_changes)) % 1000000:06d}"

        try:
            validation_results = []
            safety_scores = []
            blocking_issues = []

            for change in code_changes:
                try:
                    # Create temporary file for the modified code
                    temp_file = self.validation_helper.create_temp_file(change.modified_code)
                    tmp_file_path = temp_file.name

                    try:
                        # Run validation analyses
                        validation_result = await self._validate_single_change(
                            tmp_file_path, change, validation_config
                        )
                        validation_results.append(validation_result)

                        # Calculate safety score for this change
                        safety_score = self.safety_calculator.calculate_score(validation_result)
                        safety_scores.append(safety_score)

                        # Check for blocking issues
                        blocking = self.safety_calculator.check_blocking_issues(validation_result)
                        if blocking:
                            blocking_issues.extend(blocking)

                    finally:
                        os.unlink(tmp_file_path)

                except FileProcessingError as e:
                    logger.error("Error validating change for %s: %s", change.file_path, e)
                    # Create error result for this change
                    error_result = self._create_error_validation_result(change.file_path, str(e))
                    validation_results.append(error_result)
                    safety_scores.append(0.0)

            # Calculate overall safety score
            overall_safety_score = sum(safety_scores) / len(safety_scores) if safety_scores else 0.0

            # Generate recommendations
            recommendations = self.recommendation_generator.generate_recommendations(
                validation_results, overall_safety_score, blocking_issues
            )

            return AdvancedValidationResponse(
                analysis_id=analysis_id,
                validation_results=validation_results,
                overall_safety_score=overall_safety_score,
                blocking_issues=blocking_issues,
                recommendations=recommendations,
            )

        except ValidationError as e:
            logger.error("Validation error: %s", e)
            raise

    async def run_symbolic_analysis_on_file(self, file_path: str) -> dict[str, Any]:
        """Run symbolic analysis on a single file."""
        analysis_results = {}

        # Run different analyses based on configuration
        validation_types = self.config.validation_types or []
        if ValidationType.SYMBOLIC_EXECUTION in validation_types:
            try:
                crosshair_result = await self.crosshair_analyzer.analyze(file_path)
                analysis_results["symbolic_execution"] = crosshair_result
            except CrossHairAnalysisError as e:
                logger.warning("CrossHair analysis failed: %s", e)
                analysis_results["symbolic_execution"] = {
                    "status": "error",
                    "error": str(e),
                }

        if ValidationType.TAINT_ANALYSIS in validation_types:
            try:
                pysa_result = self.pysa_analyzer.analyze(Path(file_path))
                analysis_results["taint_analysis"] = pysa_result
            except PysaAnalysisError as e:
                logger.warning("Pysa analysis failed: %s", e)
                analysis_results["taint_analysis"] = {
                    "status": "error",
                    "error": str(e),
                }

        if ValidationType.FORMAL_VERIFICATION in validation_types:
            try:
                formal_result = await self.formal_verifier.verify(file_path)
                analysis_results["formal_verification"] = formal_result
            except FormalVerificationError as e:
                logger.warning("Formal verification failed: %s", e)
                analysis_results["formal_verification"] = {
                    "status": "error",
                    "error": str(e),
                }

        return analysis_results

    async def run_symbolic_analysis_on_directory(self, directory_path: str) -> dict[str, Any]:
        """Run symbolic analysis on all Python files in a directory."""
        analysis_results = {"files": {}, "summary": {}}

        # Find all Python files
        python_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))

        # Analyze each file
        for file_path in python_files:
            try:
                file_results = await self.run_symbolic_analysis_on_file(file_path)
                analysis_results["files"][file_path] = file_results
            except SymbolicAnalysisError as e:
                logger.error("Error analyzing file %s: %s", file_path, e)
                analysis_results["files"][file_path] = {"error": str(e)}

        # Generate summary
        analysis_results["summary"] = {
            "total_files": len(python_files),
            "analyzed_files": len(
                [r for r in analysis_results["files"].values() if "error" not in r]
            ),
            "failed_files": len([r for r in analysis_results["files"].values() if "error" in r]),
        }

        return analysis_results

    async def _validate_single_change(
        self, tmp_file_path: str, change: CodeChange, config: ValidationConfig
    ) -> Any:
        """Validate a single code change."""
        # Run the specified validation types
        symbolic_result = {}
        taint_result = {}
        formal_result = {}
        cfg_validation_types = config.validation_types or []

        if ValidationType.SYMBOLIC_EXECUTION in cfg_validation_types:
            try:
                symbolic_result = await self.crosshair_analyzer.analyze(tmp_file_path)
            except CrossHairAnalysisError as e:
                logger.warning("CrossHair analysis failed for change: %s", e)
                symbolic_result = {"status": "error", "error": str(e)}

        if ValidationType.TAINT_ANALYSIS in cfg_validation_types:
            try:
                pysa_result = self.pysa_analyzer.analyze(Path(tmp_file_path))
                taint_result = dataclasses.asdict(pysa_result)
            except PysaAnalysisError as e:
                logger.warning("Pysa analysis failed for change: %s", e)
                taint_result = {"status": "error", "error": str(e)}

        if ValidationType.FORMAL_VERIFICATION in cfg_validation_types:
            try:
                formal_result = await self.formal_verifier.verify(tmp_file_path)
            except FormalVerificationError as e:
                logger.warning("Formal verification failed for change: %s", e)
                formal_result = {"status": "error", "error": str(e)}

        # Create validation result
        return self.validation_helper.create_validation_result(
            change.file_path,
            cfg_validation_types,
            symbolic_result,
            taint_result,
            formal_result,
        )

    def _create_error_result(self, file_path: str, error: str) -> CoreAnalysisResult:
        """Create an error analysis result."""
        return CoreAnalysisResult(
            success=False,
            language="python",
            file_path=file_path,
            entities=[],
            relationships=[],
            complexity_metrics=None,
            error=error,
            parse_time_ms=0.0,
            total_lines=0,
            metadata={},
        )

    def _create_error_validation_result(self, file_path: str, error: str) -> Any:
        """Create an error validation result."""
        return self.validation_helper.create_validation_result(
            file_path,
            self.config.validation_types or [],
            {"status": "error", "error": error},
            {"status": "error", "error": error},
            {"status": "error", "error": error},
        )
