#!/usr/bin/env python3
"""Quality analysis core module."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid7

from .quality_config import config, create_temp_dir, detect_language
from .quality_metrics import quality_metrics
from .quality_models import AppToolResult, QualityResult, QualityServiceRequest
from .tool_integration import tool_executor

logger = logging.getLogger(__name__)


class QualityAnalysisCore:
    """Core quality analysis orchestration service."""

    def __init__(self):
        """Initialize quality analysis core."""
        self.config = config
        self.tool_executor = tool_executor
        self.quality_metrics = quality_metrics

    def generate_analysis_id(self) -> str:
        """Generate unique analysis ID."""
        return str(uuid7())

    async def cleanup_temp_dir(self, temp_path: Path) -> None:
        """Clean up temporary directory."""
        try:
            if temp_path.exists():
                shutil.rmtree(temp_path)
                logger.debug(f"Cleaned up temporary directory: {temp_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {temp_path}: {e}")

    async def analyze_code_quality(self, request: QualityServiceRequest) -> QualityResult:
        """Perform comprehensive code quality analysis."""
        analysis_id = self.generate_analysis_id()
        start_time = time.time()
        temp_path = None

        try:
            logger.info(f"Starting quality analysis: {analysis_id}")

            # Detect language
            language = request.language or detect_language(
                request.code_path or "", request.code_content or ""
            )

            # Create temporary directory
            temp_path = await create_temp_dir(analysis_id)

            # Write code to temporary file
            file_path = await self._prepare_analysis_file(request, temp_path, language)

            # Get available tools for language
            available_tools = self.config.get_tools_for_language(language)
            tools_to_run = request.tools if request.tools else list(available_tools.keys())

            # Filter tools to only those available for the language
            tools_to_run = [tool for tool in tools_to_run if tool in available_tools]

            if not tools_to_run:
                raise ValueError(f"No quality tools available for language: {language}")

            # Run quality tools
            tools_results = await self._execute_quality_tools(
                tools_to_run, available_tools, str(file_path), language
            )

            # Calculate overall metrics
            overall_score = self.quality_metrics.calculate_quality_score(tools_results)
            total_issues = self.quality_metrics.calculate_total_issues(tools_results)
            execution_time = time.time() - start_time

            success = any(result.success for result in tools_results.values())

            logger.info(
                f"Quality analysis completed: {analysis_id}, score: {overall_score:.1f}, issues: {total_issues}"
            )

            return QualityResult(
                analysis_id=analysis_id,
                success=success,
                language=language,
                tools_results=tools_results,
                overall_score=overall_score,
                total_issues=total_issues,
                execution_time=execution_time,
                created_at=datetime.now(UTC).isoformat(),
                error=None,  # Successful analysis has no error
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e).strip() if e else ""
            if not error_message:
                error_message = "Quality analysis failed without specific error details"

            logger.error(f"Quality analysis failed: {analysis_id}, error: {error_message}")

            # Create a ToolResult with error details to provide context
            error_tool_result = AppToolResult(
                tool="analysis",
                success=False,
                issues=[],
                execution_time=execution_time,
                output="",
                error=error_message,
            )

            return QualityResult(
                analysis_id=analysis_id,
                success=False,
                language=(language if "language" in locals() and language else "unknown"),
                tools_results={"analysis_error": error_tool_result},
                overall_score=0.0,
                total_issues=0,
                execution_time=execution_time,
                created_at=datetime.now(UTC).isoformat(),
                error=error_message,  # Include detailed error message
            )

        finally:
            # Schedule cleanup
            if temp_path:
                asyncio.create_task(self.cleanup_temp_dir(temp_path))

    async def _prepare_analysis_file(
        self, request: QualityServiceRequest, temp_path: Path, language: str
    ) -> Path:
        """Prepare the code file for analysis in the temporary directory."""
        if request.code_content:
            file_ext = {
                "python": ".py",
                "javascript": ".js",
                "typescript": ".ts",
            }.get(language, ".py")
            file_path = temp_path / f"analysis{file_ext}"
            with open(file_path, "w") as f:
                f.write(request.code_content)
        elif request.code_path:
            # Copy file to temp directory
            file_path = temp_path / Path(request.code_path).name
            shutil.copy2(request.code_path, file_path)
        else:
            raise ValueError("Either 'code_path' or 'code_content' must be provided")

        return file_path

    async def _execute_quality_tools(
        self,
        tools_to_run: list[str],
        available_tools: dict[str, dict[str, Any]],
        file_path: str,
        language: str,
    ) -> dict[str, AppToolResult]:
        """Execute quality tools in parallel and return results."""
        tools_results: dict[str, AppToolResult] = {}
        tasks: list[tuple[str, Any]] = []

        for tool_name in tools_to_run:
            tool_config = available_tools[tool_name]
            task = self.tool_executor.run_quality_tool(tool_name, tool_config, file_path, language)
            tasks.append((tool_name, task))

        # Execute all tools concurrently - ACTUALLY PARALLEL NOW!
        logger.info(f"Running {len(tasks)} tools in parallel: {[name for name, _ in tasks]}")

        # Run all tools in parallel using asyncio.gather
        task_list = [task for _, task in tasks]
        tool_names = [name for name, _ in tasks]

        try:
            results = await asyncio.gather(*task_list, return_exceptions=True)

            # Process results
            for i, (tool_name, result) in enumerate(zip(tool_names, results)):
                if isinstance(result, Exception):
                    logger.error(f"Tool {tool_name} failed with exception: {result}")
                    tools_results[tool_name] = AppToolResult(
                        tool=tool_name,
                        success=False,
                        issues=[],
                        execution_time=0.0,
                        error=str(result),
                    )
                else:
                    # result is ToolResult at this point (we've checked it's not Exception)
                    assert isinstance(result, AppToolResult)
                    tools_results[tool_name] = result
                    logger.info(
                        f"Tool {tool_name} completed: {result.success}, {result.execution_time:.2f}s"
                    )

        except Exception as e:
            logger.error(f"Error in parallel tool execution: {e}")
            # Fallback to sequential execution if parallel fails
            logger.warning("Falling back to sequential tool execution...")

            for tool_name, task in tasks:
                try:
                    result = await task
                    tools_results[tool_name] = result
                except Exception as tool_error:
                    logger.error(f"Tool {tool_name} failed: {tool_error}")
                    tools_results[tool_name] = AppToolResult(
                        tool=tool_name,
                        success=False,
                        issues=[],
                        execution_time=0.0,
                        error=str(tool_error),
                    )

        return tools_results

    async def get_analysis_summary(self, tools_results: dict) -> dict:
        """Generate comprehensive analysis summary with recommendations."""
        return self.quality_metrics.generate_quality_summary(tools_results)

    async def validate_request(self, request: QualityServiceRequest) -> None:
        """Validate quality analysis request parameters."""
        if not request.code_path and not request.code_content:
            raise ValueError("Either 'code_path' or 'code_content' must be provided")

        if request.code_path and not Path(request.code_path).exists():
            raise ValueError(f"Code path does not exist: {request.code_path}")

        if request.language and request.language not in self.config.get_supported_languages():
            raise ValueError(f"Unsupported language: {request.language}")

        # Validate custom tools if specified
        if request.tools:
            language = request.language or detect_language(
                request.code_path or "", request.code_content or ""
            )
            available_tools = self.config.get_tools_for_language(language)
            invalid_tools = [tool for tool in request.tools if tool not in available_tools]
            if invalid_tools:
                raise ValueError(f"Invalid tools for {language}: {invalid_tools}")


# Create global quality analysis core instance
quality_analysis_core = QualityAnalysisCore()
