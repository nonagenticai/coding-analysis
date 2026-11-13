"""
Codebase analysis processor using modular components.
Handles comprehensive codebase analysis with direct database access.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from app.core.config import get_settings
from db.facade import GraphDatabaseConnection

from .task_execution_engine import BaseTaskProcessor
from .task_result_processor import FileProcessingResultHandler, TaskResultProcessor
from .task_utilities import (
    FileOperationUtils,
    LanguageDetectionUtils,
    PerformanceUtils,
    ProcessingUtils,
)
from .task_validator import (
    FileSystemValidator,
    TaskErrorHandler,
    TaskParameterValidator,
)

settings = get_settings()
logger = logging.getLogger(__name__)


class CodebaseAnalysisProcessor(BaseTaskProcessor):
    """Processor for codebase analysis tasks with direct database access."""

    def __init__(self):
        self.result_processor = TaskResultProcessor()
        self.file_result_handler = FileProcessingResultHandler()
        self.error_handler = TaskErrorHandler()

    def get_supported_file_types(self) -> list[str]:
        """Get list of supported file extensions."""
        return FileOperationUtils.get_supported_file_types()

    async def process(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Process codebase analysis task."""
        # Validate parameters
        validation_result = TaskParameterValidator.validate_codebase_analysis_params(task_data)
        if not validation_result["is_valid"]:
            return {
                "success": False,
                "errors": validation_result["errors"],
                "error": "Parameter validation failed",
            }

        validated_params = validation_result["validated_params"]
        codebase_path = validated_params["codebase_path"]
        code_graph_id = validated_params["code_graph_id"]
        scan_depth = validated_params["scan_depth"]

        # Validate filesystem
        fs_validation = FileSystemValidator.validate_codebase_path(codebase_path)
        if not fs_validation["is_valid"]:
            return {"success": False, "error": fs_validation["error"]}

        # Initialize database connection
        db_conn = GraphDatabaseConnection(code_graph_id)

        try:
            # Establish async connection
            await db_conn.connect()
            conn = db_conn.get_connection()

            with PerformanceUtils.create_timer() as timer:
                # Process codebase
                processed_files = await self._perform_codebase_analysis(
                    codebase_path, conn, code_graph_id
                )

                # Calculate and return results
                return await self.result_processor.build_analysis_result(
                    conn, processed_files, scan_depth, timer.start_time, code_graph_id
                )

        except Exception as e:
            self.error_handler.log_error(e, "codebase analysis", "analyze_codebase")
            return self.error_handler.create_error_response(e, "codebase analysis")
        finally:
            await db_conn.close()

    async def _perform_codebase_analysis(self, codebase_path: str, conn, code_graph_id: str) -> int:
        """Perform the actual codebase analysis."""
        # Get multi-language processor factory
        factory = await LanguageDetectionUtils.initialize_graph_factory()

        # Scan codebase and process files
        return await self._scan_and_process_codebase(codebase_path, factory, conn, code_graph_id)

    async def _scan_and_process_codebase(
        self, codebase_path: str, factory, conn, code_graph_id: str
    ) -> int:
        """Scan codebase and process files using multi-language factory."""
        processed_files = 0

        try:
            # Walk through codebase directory
            processing_context = ProcessingUtils.create_processing_context(
                factory, conn, code_graph_id
            )

            for root, dirs, files in os.walk(codebase_path):
                FileOperationUtils.filter_directories(dirs)
                processed_count = await self._process_directory_files(
                    root, files, processing_context
                )
                processed_files += processed_count

                # Log progress for large codebases
                ProcessingUtils.log_progress(processed_files, 0)

            logger.info("Completed codebase scan. Processed %d files.", processed_files)
            return processed_files

        except Exception as e:
            self.error_handler.log_error(e, "codebase scanning", "analyze_codebase")
            return processed_files

    async def _process_directory_files(
        self, root: str, files: list[str], processing_context: dict[str, Any]
    ) -> int:
        """Process files in a directory."""
        factory = processing_context["factory"]
        conn = processing_context["conn"]
        code_graph_id = processing_context["code_graph_id"]

        processed_count = 0
        FileOperationUtils.get_language_extensions()

        for file in files:
            if not FileOperationUtils.should_process_file(file):
                continue

            file_path = os.path.join(root, file)
            language = FileOperationUtils.get_file_language(file_path)

            if not language:
                continue

            # Process the file
            file_context = ProcessingUtils.create_processing_context(
                factory, conn, code_graph_id, file_path=file_path, language=language
            )
            result = await self._process_single_file(file_context)

            if result:
                processed_count += 1

        return processed_count

    async def _process_single_file(self, file_context: dict[str, Any]) -> bool:
        """Process a single file and return success status."""
        factory = file_context["factory"]
        file_path = file_context["file_path"]
        language = file_context["language"]
        conn = file_context["conn"]
        code_graph_id = file_context["code_graph_id"]

        try:
            # Get appropriate processor for the language
            processor = factory.get_processor(language)

            if not processor:
                return False

            # Process file safely
            processing_result = await ProcessingUtils.process_file_safely(file_path, processor)

            if processing_result["success"]:
                # Store results in database
                await self.file_result_handler.store_processing_results(
                    conn,
                    code_graph_id,
                    {
                        "file_path": file_path,
                        "language": language,
                        "result": processing_result["result"],
                    },
                )
                return True
            return False

        except Exception as e:
            self.error_handler.log_error(e, f"processing file {file_path}", "analyze_codebase")
            return False
