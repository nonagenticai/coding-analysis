"""
Documentation Engine for AI Agent Static Analysis Augmentation
Generates architectural documentation and change summaries with modular components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from models.core_models import CoreAnalysisResult
from models.static_analysis_models import (
    DocumentationResponse,
    DocumentationSection,
    DocumentationType,
)
from processing.analyzers.code_analyzer import CodeAnalyzer
from processing.analyzers.doc_extractor import DocumentationExtractor
from processing.analyzers.doc_generator import DocumentationGenerator
from processing.analyzers.doc_validator import DocumentationValidator
from processing.analyzers.format_converter import FormatConverter

from .base_analyzer import BaseCodeProcessor

logger = logging.getLogger(__name__)


@dataclass
class DocumentationConfig:
    """Configuration for documentation generation."""

    doc_types: list[DocumentationType] = field(default_factory=list)
    include_diagrams: bool = True
    output_dir: str | None = None

    def __post_init__(self):
        if not self.doc_types:
            self.doc_types = [
                DocumentationType.ARCHITECTURE,
                DocumentationType.API_REFERENCE,
            ]


class DocumentationEngine(BaseCodeProcessor):
    """Engine for generating comprehensive documentation using modular components."""

    def __init__(self, config: DocumentationConfig | None = None):
        """Initialize documentation engine with configuration."""
        self.config = config or DocumentationConfig()

        # Initialize modular components
        self.doc_extractor = DocumentationExtractor()
        self.code_analyzer = CodeAnalyzer()
        self.doc_generator = DocumentationGenerator()
        self.doc_validator = DocumentationValidator()
        self.format_converter = FormatConverter(self.config.output_dir)

    def get_supported_extensions(self) -> list[str]:
        """Return supported file extensions."""
        return [".py", ".js", ".ts", ".md", ".rst"]

    def get_language_name(self) -> str:
        """Return supported languages."""
        return "multi-language"

    async def analyze_file(self, file_path: str) -> CoreAnalysisResult:
        """Analyze a single file for documentation extraction."""
        try:
            doc_data = await self.doc_extractor.extract_file_documentation(file_path)

            return CoreAnalysisResult(
                success=True,
                language="unknown",
                file_path=file_path,
                entities=[],
                relationships=[],
                complexity_metrics=None,
                error=None,
                parse_time_ms=0.0,
                total_lines=0,
                metadata={
                    "documentation": doc_data,
                    "analyzer": "documentation",
                    "file_path": file_path,
                },
            )
        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error analyzing file %s: %s", file_path, e)
            return CoreAnalysisResult(
                success=False,
                language="unknown",
                file_path=file_path,
                entities=[],
                relationships=[],
                complexity_metrics=None,
                error=str(e),
                parse_time_ms=0.0,
                total_lines=0,
                metadata={"error": str(e), "file_path": file_path},
            )

    async def analyze_directory(self, directory_path: str) -> CoreAnalysisResult:
        """Analyze directory for comprehensive documentation generation."""
        try:
            doc_data = await self.code_analyzer.generate_project_documentation_data(directory_path)

            return CoreAnalysisResult(
                success=True,
                language="unknown",
                file_path=directory_path,
                entities=[],
                relationships=[],
                complexity_metrics=None,
                error=None,
                parse_time_ms=0.0,
                total_lines=0,
                metadata={
                    "documentation": doc_data,
                    "analyzer": "documentation",
                    "directory_path": directory_path,
                },
            )
        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error analyzing directory %s: %s", directory_path, e)
            return CoreAnalysisResult(
                success=False,
                language="unknown",
                file_path=directory_path,
                entities=[],
                relationships=[],
                complexity_metrics=None,
                error=str(e),
                parse_time_ms=0.0,
                total_lines=0,
                metadata={"error": str(e), "directory_path": directory_path},
            )

    async def analyze_snippet(
        self, code_snippet: str, file_path: str | None = None
    ) -> CoreAnalysisResult:
        """Analyze code snippet for documentation extraction."""
        try:
            doc_data = await self.doc_extractor.extract_snippet_documentation(code_snippet)

            return CoreAnalysisResult(
                success=True,
                language="unknown",
                file_path=file_path or "snippet",
                entities=[],
                relationships=[],
                complexity_metrics=None,
                error=None,
                parse_time_ms=0.0,
                total_lines=0,
                metadata={
                    "documentation": doc_data,
                    "analyzer": "documentation",
                    "snippet": True,
                    "original_file": file_path,
                },
            )

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error analyzing code snippet: %s", e)
            return CoreAnalysisResult(
                success=False,
                language="unknown",
                file_path=file_path or "snippet",
                entities=[],
                relationships=[],
                complexity_metrics=None,
                error=str(e),
                parse_time_ms=0.0,
                total_lines=0,
                metadata={"error": str(e), "snippet": True},
            )

    async def generate_comprehensive_documentation(
        self,
        repository_path: str,
        config: DocumentationConfig | None = None,
    ) -> DocumentationResponse:
        """
        Generate comprehensive documentation as specified in the PRD.

        Args:
            repository_path: Path to the repository root
            config: Optional configuration override

        Returns:
            DocumentationResponse with generated documentation
        """
        doc_config = config or self.config
        analysis_id = f"doc_{hash(repository_path) % 1000000:06d}"

        try:
            documentation_sections = {}
            generated_files = []
            quality_scores = []

            # Generate each requested documentation type
            for doc_type in doc_config.doc_types:
                try:
                    section = await self.doc_generator.generate_documentation_by_type(
                        doc_type,
                        repository_path,
                        include_diagrams=doc_config.include_diagrams,
                    )

                    documentation_sections[doc_type.value] = section

                    # Collect generated files and quality scores
                    if hasattr(section, "diagrams"):
                        generated_files.extend(section.diagrams)
                    if hasattr(section, "coverage") and section.coverage:
                        quality_scores.append(section.coverage)

                except (OSError, ValueError, RuntimeError) as e:
                    logger.warning(
                        "Error generating %s documentation: %s",
                        doc_type.value,
                        e,
                    )
                    documentation_sections[doc_type.value] = DocumentationSection(
                        content=f"Error generating {doc_type.value}: {str(e)}",
                        diagrams=[],
                        coverage=0.0,
                    )

            # Calculate overall quality score using validator
            if documentation_sections:
                quality_report = self.doc_validator.assess_documentation_quality(
                    documentation_sections
                )
                overall_quality = quality_report["overall_score"]
            else:
                overall_quality = 0.0

            # Generate output files if needed
            if doc_config.output_dir:
                try:
                    output_files = await self.format_converter.save_documentation_files(
                        documentation_sections, "comprehensive_documentation"
                    )
                    generated_files.extend(output_files)
                except (OSError, ValueError, RuntimeError) as e:
                    logger.warning("Error saving documentation files: %s", e)

            return DocumentationResponse(
                analysis_id=analysis_id,
                documentation=documentation_sections,
                generated_files=generated_files,
                quality_score=round(overall_quality, 2),
            )

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error generating comprehensive documentation: %s", e)
            return DocumentationResponse(
                analysis_id=analysis_id,
                documentation={},
                generated_files=[],
                quality_score=0.0,
            )

    async def extract_file_documentation(self, file_path: str) -> dict[str, Any]:
        """Extract documentation from a single file."""
        return await self.doc_extractor.extract_file_documentation(file_path)

    async def generate_project_documentation(self, project_path: str) -> dict[str, Any]:
        """Generate comprehensive project documentation."""
        return await self.code_analyzer.generate_project_documentation_data(project_path)

    async def validate_documentation_quality(
        self, documentation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate documentation quality and completeness."""
        try:
            # Assess quality
            quality_report = self.doc_validator.assess_documentation_quality(documentation_data)

            # Check completeness
            completeness_report = self.doc_validator.validate_documentation_completeness(
                ".",
                documentation_data,  # Using current dir as fallback
            )

            # Analyze consistency
            consistency_report = self.doc_validator.analyze_documentation_consistency(
                documentation_data
            )

            # Generate metrics
            metrics = self.doc_validator.generate_quality_metrics(documentation_data)

            return {
                "quality": quality_report,
                "completeness": completeness_report,
                "consistency": consistency_report,
                "metrics": metrics,
                "overall_assessment": {
                    "score": (
                        quality_report.get("overall_score", 0.0)
                        + completeness_report.get("completeness_score", 0.0)
                        + consistency_report.get("overall_consistency", 0.0)
                        + metrics.get("overall_score", 0.0)
                    )
                    / 4,
                    "recommendations": (
                        quality_report.get("recommendations", [])
                        + [
                            f"Missing: {item}"
                            for item in completeness_report.get("missing_docs", [])
                        ]
                    ),
                },
            }

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error validating documentation quality: %s", e)
            return {
                "error": str(e),
                "overall_assessment": {
                    "score": 0.0,
                    "recommendations": ["Unable to validate documentation quality"],
                },
            }

    async def generate_diagrams(
        self, repository_path: str, diagram_types: list[str] | None = None
    ) -> list[str]:
        """Generate documentation diagrams."""
        try:
            diagrams = []

            # Generate architecture diagrams if requested
            if not diagram_types or "architecture" in diagram_types:
                from .call_graph_analyzer import CallGraphGenerator

                call_graph_analyzer = CallGraphGenerator()
                call_graph_result = await call_graph_analyzer.generate_comprehensive_call_graph(
                    repository_path
                )
                arch_diagrams = await self.format_converter.generate_architecture_diagrams(
                    repository_path, call_graph_result
                )
                diagrams.extend(arch_diagrams)

            return diagrams

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error generating diagrams: %s", e)
            return []

    def cleanup(self):
        """Clean up temporary files and resources."""
        try:
            self.format_converter.cleanup_temp_files()
        except (OSError, ValueError) as e:
            logger.warning("Error during cleanup: %s", e)
