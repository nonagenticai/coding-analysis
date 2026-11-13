"""Tests for processing.analyzers.documentation_engine module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.static_analysis_models import DocumentationSection, DocumentationType
from processing.analyzers.documentation_engine import (
    DocumentationConfig,
    DocumentationEngine,
)


class TestDocumentationConfig:
    def test_default_config(self):
        config = DocumentationConfig()
        assert DocumentationType.ARCHITECTURE in config.doc_types
        assert DocumentationType.API_REFERENCE in config.doc_types
        assert config.include_diagrams is True
        assert config.output_dir is None

    def test_custom_config(self):
        config = DocumentationConfig(
            doc_types=[DocumentationType.ARCHITECTURE],
            include_diagrams=False,
            output_dir="/tmp/docs",
        )
        assert config.doc_types == [DocumentationType.ARCHITECTURE]
        assert config.include_diagrams is False
        assert config.output_dir == "/tmp/docs"

    def test_empty_doc_types_gets_defaults(self):
        config = DocumentationConfig(doc_types=[])
        assert len(config.doc_types) == 2


class TestDocumentationEngineInit:
    def test_default_init(self):
        engine = DocumentationEngine()
        assert engine.config is not None
        assert engine.doc_extractor is not None
        assert engine.code_analyzer is not None
        assert engine.doc_generator is not None
        assert engine.doc_validator is not None
        assert engine.format_converter is not None

    def test_custom_config(self):
        config = DocumentationConfig(include_diagrams=False)
        engine = DocumentationEngine(config=config)
        assert engine.config.include_diagrams is False

    def test_supported_extensions(self):
        engine = DocumentationEngine()
        exts = engine.get_supported_extensions()
        assert ".py" in exts
        assert ".js" in exts
        assert ".ts" in exts
        assert ".md" in exts

    def test_language_name(self):
        engine = DocumentationEngine()
        assert engine.get_language_name() == "multi-language"


class TestDocumentationEngineAnalyzeFile:
    @pytest.mark.asyncio
    async def test_analyze_file_success(self):
        engine = DocumentationEngine()
        engine.doc_extractor = MagicMock()
        engine.doc_extractor.extract_file_documentation = AsyncMock(
            return_value={"functions": [], "classes": []}
        )

        result = await engine.analyze_file("/tmp/test.py")
        assert result.success is True
        assert result.file_path == "/tmp/test.py"
        assert result.metadata["analyzer"] == "documentation"

    @pytest.mark.asyncio
    async def test_analyze_file_error(self):
        engine = DocumentationEngine()
        engine.doc_extractor = MagicMock()
        engine.doc_extractor.extract_file_documentation = AsyncMock(
            side_effect=OSError("file not found")
        )

        result = await engine.analyze_file("/tmp/missing.py")
        assert result.success is False
        assert "file not found" in result.error


class TestDocumentationEngineAnalyzeDirectory:
    @pytest.mark.asyncio
    async def test_analyze_directory_success(self):
        engine = DocumentationEngine()
        engine.code_analyzer = MagicMock()
        engine.code_analyzer.generate_project_documentation_data = AsyncMock(
            return_value={"modules": [], "summary": {}}
        )

        result = await engine.analyze_directory("/tmp/project")
        assert result.success is True
        assert result.metadata["directory_path"] == "/tmp/project"

    @pytest.mark.asyncio
    async def test_analyze_directory_error(self):
        engine = DocumentationEngine()
        engine.code_analyzer = MagicMock()
        engine.code_analyzer.generate_project_documentation_data = AsyncMock(
            side_effect=ValueError("bad path")
        )

        result = await engine.analyze_directory("/bad")
        assert result.success is False
        assert "bad path" in result.error


class TestDocumentationEngineAnalyzeSnippet:
    @pytest.mark.asyncio
    async def test_analyze_snippet_success(self):
        engine = DocumentationEngine()
        engine.doc_extractor = MagicMock()
        engine.doc_extractor.extract_snippet_documentation = AsyncMock(
            return_value={"docstrings": []}
        )

        result = await engine.analyze_snippet("def foo(): pass")
        assert result.success is True
        assert result.metadata["snippet"] is True

    @pytest.mark.asyncio
    async def test_analyze_snippet_with_file_path(self):
        engine = DocumentationEngine()
        engine.doc_extractor = MagicMock()
        engine.doc_extractor.extract_snippet_documentation = AsyncMock(return_value={})

        result = await engine.analyze_snippet("x = 1", file_path="test.py")
        assert result.file_path == "test.py"
        assert result.metadata["original_file"] == "test.py"

    @pytest.mark.asyncio
    async def test_analyze_snippet_error(self):
        engine = DocumentationEngine()
        engine.doc_extractor = MagicMock()
        engine.doc_extractor.extract_snippet_documentation = AsyncMock(
            side_effect=RuntimeError("parse error")
        )

        result = await engine.analyze_snippet("broken code")
        assert result.success is False
        assert result.metadata["snippet"] is True


class TestDocumentationEngineGenerateComprehensive:
    @pytest.mark.asyncio
    async def test_generate_success(self):
        engine = DocumentationEngine()
        engine.doc_generator = MagicMock()
        section = DocumentationSection(
            content="Architecture docs",
            diagrams=["arch.png"],
            coverage=0.85,
        )
        engine.doc_generator.generate_documentation_by_type = AsyncMock(return_value=section)
        engine.doc_validator = MagicMock()
        engine.doc_validator.assess_documentation_quality = MagicMock(
            return_value={"overall_score": 0.9}
        )

        result = await engine.generate_comprehensive_documentation("/tmp/repo")
        assert result.quality_score > 0
        assert len(result.generated_files) >= 1

    @pytest.mark.asyncio
    async def test_generate_with_output_dir(self):
        config = DocumentationConfig(output_dir="/tmp/docs")
        engine = DocumentationEngine(config=config)
        engine.doc_generator = MagicMock()
        section = DocumentationSection(
            content="Architecture docs",
            diagrams=[],
            coverage=0.8,
        )
        engine.doc_generator.generate_documentation_by_type = AsyncMock(return_value=section)
        engine.doc_validator = MagicMock()
        engine.doc_validator.assess_documentation_quality = MagicMock(
            return_value={"overall_score": 0.8}
        )
        engine.format_converter = MagicMock()
        engine.format_converter.save_documentation_files = AsyncMock(return_value=["doc.html"])

        result = await engine.generate_comprehensive_documentation("/tmp/repo")
        assert "doc.html" in result.generated_files

    @pytest.mark.asyncio
    async def test_generate_section_error_handled(self):
        engine = DocumentationEngine()
        engine.doc_generator = MagicMock()
        engine.doc_generator.generate_documentation_by_type = AsyncMock(
            side_effect=ValueError("gen error")
        )
        engine.doc_validator = MagicMock()
        engine.doc_validator.assess_documentation_quality = MagicMock(
            return_value={"overall_score": 0.0}
        )

        result = await engine.generate_comprehensive_documentation("/tmp/repo")
        assert result.quality_score >= 0

    @pytest.mark.asyncio
    async def test_generate_section_error_returns_error_section(self):
        """When a doc type generation fails, an error section is created."""
        engine = DocumentationEngine()
        engine.doc_generator = MagicMock()
        engine.doc_generator.generate_documentation_by_type = AsyncMock(
            side_effect=ValueError("gen error")
        )
        engine.doc_validator = MagicMock()
        engine.doc_validator.assess_documentation_quality = MagicMock(
            return_value={"overall_score": 0.0}
        )

        result = await engine.generate_comprehensive_documentation("/tmp/repo")
        # Should contain error sections for each doc type
        for section in result.documentation.values():
            assert "Error generating" in section.content
        assert result.quality_score == 0.0


class TestDocumentationEngineValidateQuality:
    @pytest.mark.asyncio
    async def test_validate_success(self):
        engine = DocumentationEngine()
        engine.doc_validator = MagicMock()
        engine.doc_validator.assess_documentation_quality = MagicMock(
            return_value={"overall_score": 0.8, "recommendations": ["Add more docs"]}
        )
        engine.doc_validator.validate_documentation_completeness = MagicMock(
            return_value={"completeness_score": 0.7, "missing_docs": ["README"]}
        )
        engine.doc_validator.analyze_documentation_consistency = MagicMock(
            return_value={"overall_consistency": 0.9}
        )
        engine.doc_validator.generate_quality_metrics = MagicMock(
            return_value={"overall_score": 0.85}
        )

        result = await engine.validate_documentation_quality({"docs": "data"})
        assert "quality" in result
        assert "completeness" in result
        assert "consistency" in result
        assert "metrics" in result
        assert "overall_assessment" in result
        assert result["overall_assessment"]["score"] > 0
        assert "Missing: README" in result["overall_assessment"]["recommendations"]

    @pytest.mark.asyncio
    async def test_validate_error(self):
        engine = DocumentationEngine()
        engine.doc_validator = MagicMock()
        engine.doc_validator.assess_documentation_quality = MagicMock(
            side_effect=RuntimeError("validation failed")
        )

        result = await engine.validate_documentation_quality({})
        assert "error" in result
        assert result["overall_assessment"]["score"] == 0.0


class TestDocumentationEngineHelpers:
    @pytest.mark.asyncio
    async def test_extract_file_documentation(self):
        engine = DocumentationEngine()
        engine.doc_extractor = MagicMock()
        engine.doc_extractor.extract_file_documentation = AsyncMock(return_value={"functions": []})
        result = await engine.extract_file_documentation("test.py")
        assert result == {"functions": []}

    @pytest.mark.asyncio
    async def test_generate_project_documentation(self):
        engine = DocumentationEngine()
        engine.code_analyzer = MagicMock()
        engine.code_analyzer.generate_project_documentation_data = AsyncMock(
            return_value={"modules": []}
        )
        result = await engine.generate_project_documentation("/project")
        assert result == {"modules": []}


class TestDocumentationEngineGenerateDiagrams:
    @pytest.mark.asyncio
    async def test_generate_diagrams_success(self):
        engine = DocumentationEngine()
        engine.format_converter = MagicMock()
        engine.format_converter.generate_architecture_diagrams = AsyncMock(
            return_value=["diagram.svg"]
        )

        with patch("processing.analyzers.call_graph_analyzer.CallGraphGenerator") as mock_cgg:
            mock_gen = mock_cgg.return_value
            mock_gen.generate_comprehensive_call_graph = AsyncMock(
                return_value={"nodes": [], "edges": []}
            )
            result = await engine.generate_diagrams("/tmp/repo", ["architecture"])
            assert "diagram.svg" in result

    @pytest.mark.asyncio
    async def test_generate_diagrams_error(self):
        engine = DocumentationEngine()
        engine.format_converter = MagicMock()
        engine.format_converter.generate_architecture_diagrams = AsyncMock(
            side_effect=OSError("no disk")
        )

        with patch("processing.analyzers.call_graph_analyzer.CallGraphGenerator") as mock_cgg:
            mock_gen = mock_cgg.return_value
            mock_gen.generate_comprehensive_call_graph = AsyncMock(
                side_effect=RuntimeError("failed")
            )
            result = await engine.generate_diagrams("/tmp/repo")
            assert result == []


class TestDocumentationEngineCleanup:
    def test_cleanup_success(self):
        engine = DocumentationEngine()
        engine.format_converter = MagicMock()
        engine.format_converter.cleanup_temp_files = MagicMock()
        engine.cleanup()
        engine.format_converter.cleanup_temp_files.assert_called_once()

    def test_cleanup_error(self):
        engine = DocumentationEngine()
        engine.format_converter = MagicMock()
        engine.format_converter.cleanup_temp_files = MagicMock(
            side_effect=OSError("cleanup failed")
        )
        # Should not raise
        engine.cleanup()
