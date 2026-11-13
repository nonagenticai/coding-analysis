"""
Batch Analysis Endpoints

Batch processing for analyzing multiple files in containers efficiently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path as PathLib
from typing import Any
from uuid import uuid7

from coding_client.utils.async_timeout import AsyncTimeoutError
from fastapi import APIRouter, Body, Depends, HTTPException

from app.api.utils.workspace_files import release_materialized
from app.core.dependencies import get_multi_language_factory
from auth.tenant import require_tenant
from processing.multi_language_factory import (
    get_processor_for_language,
)

from ...models.analysis_api_models import BatchAnalysisRequest
from ...models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error406Response,
    Error422Response,
    Error429Response,
    Error500Response,
)
from .analysis_utils import resolve_container_files

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/batch",
    response_model=dict[str, Any],
    summary="📦 Batch Container Analysis",
    description="""
    🚀 **Batch Processing for Container Files** - Analyze multiple files in a container
    simultaneously with optimized performance

    ## Purpose
    This endpoint enables efficient batch processing of multiple source files within a container
    across different languages, providing consolidated results and performance optimizations for large codebases.

    ## How to Use
    **POST** `/api/analysis/batch`

    **Analyze All Files in Container:**
    ```json
    {
        "container_name": "my-python-project",
        "analysis_type": "augmented",
        "batch_options": {
            "max_concurrent": 5,
            "timeout_per_file": 30,
            "include_relationships": true,
            "aggregate_results": true
        }
    }
    ```

    **Analyze Specific Files in Container:**
    ```json
    {
        "container_name": "my-java-project",
        "file_paths": [
            "/app/data/repo/src/main/java/Main.java",
            "/app/data/repo/src/main/java/Utils.java",
            "/app/data/repo/src/test/java/MainTest.java"
        ],
        "analysis_type": "security",
        "batch_options": {
            "max_concurrent": 3,
            "timeout_per_file": 45,
            "include_relationships": true,
            "aggregate_results": true
        }
    }
    ```

    ## Features
    - ✅ **Container-Based Processing**: Analyze files within containers
    - ✅ **Concurrent Processing**: Process multiple files simultaneously
    - ✅ **Language Auto-Detection**: Automatic language detection when not specified
    - ✅ **Priority-Based Scheduling**: High-priority files processed first
    - ✅ **Aggregated Results**: Consolidated analysis across all files
    - ✅ **Performance Metrics**: Per-file and aggregate performance data
    - ✅ **Error Isolation**: Failed files don't affect others

    ## Response Format
    Returns comprehensive batch results with per-file analysis, aggregate metrics,
    and performance statistics for the entire batch operation.
    """,
    responses={
        200: {"description": "Batch analysis completed successfully"},
        400: {
            "model": Error400Response,
            "description": "Invalid batch request",
        },
        401: {
            "model": Error401Response,
            "description": "Authentication required",
        },
        403: {
            "model": Error403Response,
            "description": "Insufficient permissions",
        },
        404: {
            "model": Error404Response,
            "description": "Container not found",
        },
        406: {
            "model": Error406Response,
            "description": "Not acceptable - invalid Accept header",
        },
        422: {"model": Error422Response, "description": "Validation error"},
        429: {
            "model": Error429Response,
            "description": "Too many requests - rate limit exceeded",
        },
        500: {
            "model": Error500Response,
            "description": "Batch analysis failed",
        },
    },
    operation_id="analyze_files_batch",
)
async def analyze_files_batch(
    request: BatchAnalysisRequest = Body(..., description="Batch analysis request"),
    tenant_id: str = Depends(require_tenant),
) -> dict[str, Any]:
    """Perform batch analysis on multiple files in a container with optimized concurrent processing."""

    start_time = time.time()
    batch_id = str(uuid7())

    file_paths: list[Any] | None = None
    try:
        # Resolve file paths from container
        file_paths = await resolve_container_files(
            request.container_name, request.file_paths, tenant_id=tenant_id, repo=request.repo
        )

        if not file_paths:
            raise HTTPException(
                status_code=404,
                detail=f"No code files found in container '{request.container_name}'",
            )

        # Parse batch options
        options = request.batch_options or {}
        max_concurrent = options.get("max_concurrent", 5)
        timeout_per_file = options.get("timeout_per_file", 30)
        include_relationships = options.get("include_relationships", True)
        aggregate_results = options.get("aggregate_results", True)

        logger.info(
            "Starting batch analysis of %d files in container %s (batch_id: %s)",
            len(file_paths),
            request.container_name,
            batch_id,
        )

        # Prepare file analysis tasks
        analysis_tasks = []
        for i, file_path in enumerate(file_paths):
            task_info = {
                "index": i,
                "path": str(file_path),
                "language": None,  # Auto-detect
                "priority": "medium",  # Default priority
                "analysis_type": request.analysis_type,
                "batch_id": batch_id,
                "timeout": timeout_per_file,
            }
            analysis_tasks.append(task_info)

        # Process files in batches with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        file_results = await process_files_concurrent(analysis_tasks, semaphore)

        # Calculate batch metrics
        total_files = len(file_paths)
        successful_files = len([r for r in file_results if r["success"]])
        failed_files = total_files - successful_files

        batch_execution_time = time.time() - start_time

        # Aggregate results if requested
        aggregated_results = None
        if aggregate_results and successful_files > 0:
            aggregated_results = aggregate_analysis_results(file_results)

        # Build comprehensive response
        response = {
            "batch_id": batch_id,
            "container_name": request.container_name,
            "batch_summary": {
                "total_files": total_files,
                "successful_files": successful_files,
                "failed_files": failed_files,
                "success_rate": (successful_files / total_files if total_files > 0 else 0),
                "execution_time": batch_execution_time,
                "average_time_per_file": (
                    batch_execution_time / total_files if total_files > 0 else 0
                ),
            },
            "file_results": file_results,
            "aggregated_results": aggregated_results,
            "batch_options": {
                "max_concurrent": max_concurrent,
                "timeout_per_file": timeout_per_file,
                "include_relationships": include_relationships,
                "aggregate_results": aggregate_results,
            },
            "performance_metrics": {
                "fastest_file": min(
                    (r["execution_time"] for r in file_results if r["success"]),
                    default=0,
                ),
                "slowest_file": max(
                    (r["execution_time"] for r in file_results if r["success"]),
                    default=0,
                ),
                "total_entities": sum(r.get("entities_count", 0) for r in file_results),
                "total_relationships": sum(r.get("relationships_count", 0) for r in file_results),
                "languages_analyzed": len(
                    set(r.get("language") for r in file_results if r.get("language"))
                ),
            },
        }

        logger.info("Batch analysis completed: %s", response["batch_summary"])

        return response

    except HTTPException:  # Re-raise HTTP exceptions (including 404 for missing containers)
        raise
    except (TimeoutError, RuntimeError, OSError) as e:
        logger.exception("Batch analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}") from e
    except ValueError as e:
        # ValueError for missing code content should be 400, not 500
        logger.exception("Invalid batch analysis request: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid analysis request: {str(e)}") from e
    finally:
        release_materialized(file_paths)


async def process_files_concurrent(
    analysis_tasks: list[dict], semaphore: asyncio.Semaphore
) -> list[dict]:
    """Process files concurrently with semaphore control."""

    async def process_single_file(task_info):
        async with semaphore:
            return await analyze_single_file_batch(task_info)

    # Create and run concurrent tasks
    tasks = [process_single_file(task_info) for task_info in analysis_tasks]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions and convert to results
    file_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            error_result: dict[str, Any] = {
                "index": analysis_tasks[i]["index"],
                "path": analysis_tasks[i]["path"],
                "success": False,
                "error": str(result),
                "execution_time": 0,
            }
            file_results.append(error_result)
        else:
            if isinstance(result, dict):
                file_results.append(result)
            else:
                error_result = {
                    "index": analysis_tasks[i]["index"],
                    "path": analysis_tasks[i]["path"],
                    "success": False,
                    "error": "Unexpected result type",
                    "execution_time": 0,
                }
                file_results.append(error_result)

    return file_results


async def analyze_single_file_batch(task_info: dict) -> dict:
    """Analyze a single file as part of a batch operation."""

    start_time = time.time()

    try:
        file_path = PathLib(task_info["path"])
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get appropriate processor
        language = task_info.get("language")
        if language:
            processor = await get_processor_for_language(language.lower())
        else:
            # Use the factory to get processor for file
            factory = get_multi_language_factory()
            processor = await factory.get_processor_for_file(file_path)

        if not processor:
            raise HTTPException(
                status_code=422, detail=f"No processor available for file: {file_path}"
            )

        # Read file content
        content = file_path.read_text(encoding="utf-8", errors="ignore")

        # Perform analysis with timeout
        if hasattr(processor, "parse_file_with_metrics"):
            analysis_task = processor.parse_file_with_metrics(  # ty: ignore[call-non-callable]
                file_path, content, task_info.get("batch_id")
            )  # type: ignore[call-non-callable]
        elif hasattr(processor, "parse_file"):
            analysis_task = processor.parse_file(file_path, content)  # type: ignore[call-non-callable]  # ty: ignore[call-non-callable]
        else:
            raise ValueError("Processor does not support required parsing methods")

        try:
            parse_result = await asyncio.wait_for(
                analysis_task, timeout=task_info.get("timeout", 30)
            )
        except TimeoutError as exc:
            raise AsyncTimeoutError(
                f"Analysis timeout for file: {file_path}", task_info.get("timeout", 30)
            ) from exc

        execution_time = time.time() - start_time

        # Build result
        result = {
            "index": task_info["index"],
            "path": str(file_path),
            "language": parse_result.language,
            "success": parse_result.success,
            "entities_count": len(parse_result.entities),
            "relationships_count": len(parse_result.relationships),
            "lines_of_code": parse_result.lines_of_code,
            "complexity_score": parse_result.complexity_score,
            "parser_used": (
                parse_result.parser_used.value
                if hasattr(parse_result.parser_used, "value")
                else str(parse_result.parser_used)
            ),
            "execution_time": execution_time,
            "priority": task_info.get("priority"),
            "errors": parse_result.errors if parse_result.errors else [],
            "warnings": parse_result.warnings if parse_result.warnings else [],
        }

        return result

    except (RuntimeError, ValueError, ImportError, OSError) as e:
        execution_time = time.time() - start_time
        return {
            "index": task_info["index"],
            "path": task_info["path"],
            "success": False,
            "error": str(e),
            "execution_time": execution_time,
            "priority": task_info.get("priority"),
        }


def aggregate_analysis_results(file_results: list[dict]) -> dict:
    """Aggregate results from multiple file analyses."""

    successful_results = [r for r in file_results if r["success"]]

    if not successful_results:
        return {}

    # Aggregate metrics
    total_entities = sum(r.get("entities_count", 0) for r in successful_results)
    total_relationships = sum(r.get("relationships_count", 0) for r in successful_results)
    total_lines = sum(r.get("lines_of_code", 0) for r in successful_results)

    avg_complexity = sum(r.get("complexity_score", 0) for r in successful_results) / len(
        successful_results
    )
    avg_execution_time = sum(r.get("execution_time", 0) for r in successful_results) / len(
        successful_results
    )

    # Language distribution
    language_dist = {}
    for result in successful_results:
        lang = result.get("language", "unknown")
        if lang not in language_dist:
            language_dist[lang] = {"files": 0, "entities": 0, "lines": 0}
        language_dist[lang]["files"] += 1
        language_dist[lang]["entities"] += result.get("entities_count", 0)
        language_dist[lang]["lines"] += result.get("lines_of_code", 0)

    # Parser usage
    parser_usage: dict[str, int] = {}
    for result in successful_results:
        parser = result.get("parser_used", "unknown")
        parser_usage[parser] = parser_usage.get(parser, 0) + 1

    return {
        "summary": {
            "total_entities": total_entities,
            "total_relationships": total_relationships,
            "total_lines_of_code": total_lines,
            "average_complexity": avg_complexity,
            "average_execution_time": avg_execution_time,
            "files_processed": len(successful_results),
        },
        "language_distribution": language_dist,
        "parser_usage": parser_usage,
        "quality_metrics": {
            "entities_per_file": total_entities / len(successful_results),
            "relationships_per_file": total_relationships / len(successful_results),
            "lines_per_file": total_lines / len(successful_results),
            "processing_speed": total_lines
            / sum(r.get("execution_time", 1) for r in successful_results),  # lines per second
        },
    }
