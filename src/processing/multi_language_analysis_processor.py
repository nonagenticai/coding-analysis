"""
Multi-language analysis processor using modular components.
Handles comprehensive multi-language codebase analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from db.facade import GraphDatabaseConnection

from .task_execution_engine import BaseTaskProcessor
from .task_result_processor import AnalysisMetadataManager, TaskResultProcessor
from .task_utilities import LanguageDetectionUtils, PerformanceUtils, ProcessingUtils
from .task_validator import (
    FileSystemValidator,
    TaskErrorHandler,
    TaskParameterValidator,
)

settings = get_settings()
logger = logging.getLogger(__name__)


class MultiLanguageAnalysisProcessor(BaseTaskProcessor):
    """Augmented processor for multi-language codebase analysis."""

    def __init__(self):
        self.result_processor = TaskResultProcessor()
        self.metadata_manager = AnalysisMetadataManager()
        self.error_handler = TaskErrorHandler()

    def get_supported_languages(self) -> list[str]:
        """Get list of supported programming languages."""
        return LanguageDetectionUtils.get_supported_languages()

    async def process(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Process multi-language analysis task."""
        # Validate parameters
        validation_result = TaskParameterValidator.validate_multi_language_analysis_params(
            task_data
        )
        if not validation_result["is_valid"]:
            return {
                "success": False,
                "errors": validation_result["errors"],
                "error": "Parameter validation failed",
            }

        validated_params = validation_result["validated_params"]
        code_graph_id = validated_params["code_graph_id"]
        codebase_path = validated_params["codebase_path"]

        # Validate filesystem
        fs_validation = FileSystemValidator.validate_codebase_path(codebase_path)
        if not fs_validation["is_valid"]:
            return {"success": False, "error": fs_validation["error"]}

        # Initialize dependencies
        multi_language_factory = await LanguageDetectionUtils.initialize_multi_language_factory()
        db_conn = GraphDatabaseConnection(code_graph_id)

        try:
            await db_conn.connect()
            conn = db_conn.get_connection()

            with PerformanceUtils.create_timer() as timer:
                logger.info("🚀 Starting multi-language analysis for: %s", code_graph_id)

                # Perform analysis
                analysis_data = await self._analyze_codebase(
                    codebase_path, multi_language_factory, conn, code_graph_id
                )

                # Build and return result
                return self.result_processor.build_multi_language_result(
                    analysis_data, timer.start_time
                )

        except Exception as e:
            self.error_handler.log_error(e, "multi-language analysis", "multi_language_analysis")
            return self.error_handler.create_error_response(e, "multi-language analysis")
        finally:
            await db_conn.close()

    async def _analyze_codebase(
        self, codebase_path: str, multi_language_factory, conn, code_graph_id: str
    ) -> dict[str, Any]:
        """Analyze codebase and return analysis data."""
        codebase_path_obj = Path(codebase_path)

        if not codebase_path_obj.exists():
            raise FileNotFoundError(f"Codebase path does not exist: {codebase_path}")

        # Scan for supported files
        supported_files, language_stats = await LanguageDetectionUtils.scan_supported_files(
            codebase_path_obj, multi_language_factory
        )

        logger.info(
            "📊 Found %d files across %d languages",
            len(supported_files),
            len(language_stats),
        )

        # Process languages
        languages_context = ProcessingUtils.create_processing_context(
            multi_language_factory,
            conn,
            code_graph_id,
            language_stats=language_stats,
            supported_files=supported_files,
        )
        language_analysis_results = await self._process_languages(languages_context)

        # Store metadata
        metadata_context = {
            "conn": conn,
            "code_graph_id": code_graph_id,
            "supported_files": supported_files,
            "language_stats": language_stats,
            "multi_language_factory": multi_language_factory,
        }
        await self.metadata_manager.store_analysis_metadata(metadata_context)

        return {
            "supported_files": supported_files,
            "language_stats": language_stats,
            "language_analysis_results": language_analysis_results,
            "multi_language_factory": multi_language_factory,
        }

    async def _process_languages(self, languages_context: dict[str, Any]) -> dict:
        """Process each language and return analysis results."""
        language_stats = languages_context["language_stats"]
        supported_files = languages_context["supported_files"]
        multi_language_factory = languages_context["factory"]
        conn = languages_context["conn"]
        code_graph_id = languages_context["code_graph_id"]

        language_analysis_results = {}

        for language, file_count in language_stats.items():
            try:
                processor = await multi_language_factory.get_processor(language)
                if processor:
                    # Process sample files for this language
                    language_context = ProcessingUtils.create_processing_context(
                        multi_language_factory,
                        conn,
                        code_graph_id,
                        language=language,
                        file_count=file_count,
                        supported_files=supported_files,
                        processor=processor,
                    )
                    language_results = await self._process_language_sample(language_context)
                    language_analysis_results[language] = language_results
                else:
                    language_analysis_results[language] = {
                        "error": "No processor available",
                        "total_files": file_count,
                    }

            except Exception as lang_error:
                self.error_handler.log_error(
                    lang_error,
                    f"processing language {language}",
                    "multi_language_analysis",
                )
                language_analysis_results[language] = {
                    "error": str(lang_error),
                    "total_files": file_count,
                }

        return language_analysis_results

    async def _process_language_sample(self, language_context: dict[str, Any]) -> dict:
        """Process a sample of files for a specific language."""
        language = language_context["language"]
        file_count = language_context["file_count"]
        supported_files = language_context["supported_files"]
        multi_language_factory = language_context["factory"]
        processor = language_context["processor"]
        conn = language_context["conn"]
        code_graph_id = language_context["code_graph_id"]

        # Get files for this language
        language_files = [
            f
            for f in supported_files
            if multi_language_factory.get_extension_language_map().get(f.suffix.lower()) == language
        ]

        # Sample first 10 files
        sample_files = language_files[: min(10, len(language_files))]

        language_results = {
            "total_files": file_count,
            "sampled_files": len(sample_files),
            "processor_type": processor.__class__.__name__,
            "language_info": multi_language_factory.get_language_info(language),
        }

        # Store language analysis in database
        await self.metadata_manager.store_language_analysis(
            conn, code_graph_id, language, language_results
        )

        return language_results
