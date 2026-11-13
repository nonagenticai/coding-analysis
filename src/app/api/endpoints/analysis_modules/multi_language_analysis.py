"""
Multi-Language Analysis Endpoints

Advanced multi-language analysis for 8+ programming languages within containers.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path as PathLib
from uuid import uuid7

from coding_client.auth.models import User
from fastapi import APIRouter, Body, Depends, HTTPException

from app.api.utils.workspace_files import release_materialized
from auth.tenant import get_current_user_with_tenant
from processing.languages.base import LanguageProcessor
from processing.multi_language_factory import (
    get_processor_for_language,
    get_supported_languages,
)

from ...models.analysis_api_models import (
    AnalysisIssue,
    AnalysisMetrics,
    AnalysisRequest,
    ApiAnalysisResponse,
    ApiAnalysisResult,
)
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
from .analysis_utils import (
    detect_language_from_code,
    get_extension_for_language,
    resolve_container_files,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/multi-language",
    response_model=ApiAnalysisResponse,
    summary="🌐 Multi-Language Container Analysis",
    description="""
    🚀 **Advanced Multi-Language Analysis Engine** - Analyze code in 8+ programming languages within containers

    ## Purpose
    This specialized endpoint leverages our advanced multi-language processing architecture
    to provide comprehensive analysis for Java, Rust, Go, C#, C/C++, Python, JavaScript,
    and TypeScript. It automatically detects the language and applies language-specific
    parsing strategies for maximum accuracy on container-based codebases.

    ## How to Use
    **POST** `/api/analysis/multi-language`

    **Analyze All Files in Container:**
    ```json
    {
        "container_name": "multi-lang-project",
        "analysis_type": "augmented"
    }
    ```

    **Analyze Specific Files in Container:**
    ```json
    {
        "container_name": "multi-lang-project",
        "file_paths": ["src/main.rs", "lib/helper.java", "api/service.go"],
        "analysis_type": "deep"
    }
    ```

    **Code Snippet Analysis (Legacy):**
    ```json
    {
        "code": "fn main() { println!(\"Hello Rust!\"); }",
        "language": "rust",
        "analysis_type": "deep"
    }
    ```

    ## Supported Languages
    - **Java** (.java) - Classes, interfaces, methods, inheritance
    - **Rust** (.rs) - Structs, traits, impl blocks, async functions
    - **Go** (.go) - Packages, structs, interfaces, goroutines
    - **C#** (.cs) - Classes, namespaces, properties, LINQ
    - **C/C++** (.c, .cpp, .h, .hpp) - Functions, classes, templates
    - **Python** (.py) - Augmented AST analysis
    - **JavaScript** (.js, .jsx) - ES6+, classes, modules
    - **TypeScript** (.ts, .tsx) - Types, interfaces, generics

    ## Analysis Types & What They Do

    ### 🔍 **basic** - Quick Static Analysis
    - **Best for**: Fast syntax validation and basic metrics
    - **Speed**: ⚡ Very Fast (< 1 second per file)
    - **Detects**: Syntax errors, basic code smells, line counts
    - **Use when**: You need quick validation or CI/CD integration

    ### 🎯 **augmented** - Comprehensive AST Analysis
    - **Best for**: Detailed code structure analysis
    - **Speed**: ⚡ Fast (1-3 seconds per file)
    - **Detects**: Functions, classes, complexity, documentation gaps
    - **Use when**: You want detailed code insights without deep analysis

    ### 🌳 **ast** - Abstract Syntax Tree Analysis
    - **Best for**: Understanding code structure and patterns
    - **Speed**: ⚡ Fast (2-5 seconds per file)
    - **Detects**: Import patterns, function signatures, class hierarchies
    - **Use when**: You need structural analysis for refactoring

    ### 🔬 **deep** - In-Depth Quality Analysis
    - **Best for**: Comprehensive code quality assessment
    - **Speed**: 🕐 Moderate (5-15 seconds per file)
    - **Detects**: Quality issues, maintainability problems, technical debt
    - **Use when**: You want thorough quality assessment and recommendations

    ### 🔒 **security** - Security Vulnerability Detection
    - **Best for**: Finding security vulnerabilities and risks
    - **Speed**: 🕐 Moderate (5-10 seconds per file)
    - **Detects**: Injection flaws, unsafe functions, credential leaks
    - **Use when**: Security is a priority or before deployment

    ### ⚡ **performance** - Performance Analysis
    - **Best for**: Identifying performance bottlenecks
    - **Speed**: 🕐 Moderate (3-8 seconds per file)
    - **Detects**: Inefficient loops, blocking calls, memory issues
    - **Use when**: Optimizing application performance

    ## Smart Analysis Modes

    ### 📁 **Container Mode (Recommended)**
    - Provide `container_name` parameter
    - Analyzes all supported files in containerized environment
    - Generates comprehensive multi-language repository insights
    - Supports language-specific optimizations

    ### 📄 **Code Snippet Mode (Legacy)**
    - Provide `code` parameter with your code snippet
    - Specify language explicitly for best results
    - Perfect for quick validation of new code

    ## Features
    - ✅ **8+ Language Support**: Java, Rust, Go, C#, C/C++, Python, JS/TS
    - ✅ **Language Auto-Detection**: Automatic language detection from file extensions
    - ✅ **Container Integration**: Seamless analysis of containerized codebases
    - ✅ **Cross-Language Insights**: Comprehensive analysis across multiple languages
    - ✅ **Performance Optimized**: Efficient processing for large multi-language projects
    - ✅ **Error Isolation**: Issues in one language don't affect analysis of others
    """,
    responses={
        200: {"description": "Multi-language analysis completed successfully"},
        400: {
            "model": Error400Response,
            "description": "Invalid analysis request",
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
            "description": "Multi-language analysis failed",
        },
    },
    operation_id="analyze_multi_language",
)
async def analyze_multi_language_code(
    request: AnalysisRequest = Body(..., description="Multi-language analysis request"),
    user: User = Depends(get_current_user_with_tenant),
) -> ApiAnalysisResponse:
    """Perform multi-language code analysis using our advanced language processors. Requires authentication."""
    analysis_id = str(uuid7())
    start_time = time.time()

    try:
        logger.info(f"Starting multi-language analysis for user {user.preferred_username}")
        # Validate input - now supports both container and code analysis
        if not request.code and not request.container_name:
            raise HTTPException(
                status_code=400,
                detail="Either 'code' content or 'container_name' is required for multi-language analysis",
            )

        # Handle container-based analysis with tenant isolation
        if request.container_name:
            tenant_id = None if user.is_admin() else user.effective_tenant_id
            return await analyze_multi_language_container(
                request, analysis_id, start_time, tenant_id=tenant_id
            )

        # Handle code snippet analysis (legacy mode)
        # At this point we know request.code is not None (validated above)
        assert request.code is not None  # Validated at line 206
        code_content: str = request.code

        # Detect or validate language
        language = request.language
        if not language:
            # Auto-detect language from code content
            language = detect_language_from_code(code_content)
            if not language:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Could not detect programming language. Please specify language explicitly."
                    ),
                )

        # Validate language support
        supported_languages = get_supported_languages()
        if language.lower() not in [lang.lower() for lang in supported_languages]:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Language '{language}' is not supported. Supported languages: {', '.join(supported_languages)}"
                ),
            )

        # Get language processor
        processor = await get_processor_for_language(language.lower())
        if not processor:
            raise HTTPException(
                status_code=500,
                detail=f"Could not initialize processor for language: {language}",
            )

        # Create temporary file for analysis
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=get_extension_for_language(language),
            delete=False,
        ) as tmp_file:
            tmp_file.write(code_content)
            tmp_file_path = PathLib(tmp_file.name)

        try:
            # Perform language-specific analysis
            if isinstance(processor, LanguageProcessor):
                # Use new multi-language processor with metrics tracking
                parse_result = await processor.parse_file_with_metrics(
                    tmp_file_path, code_content, analysis_id
                )

                # Convert to API response format
                issues = []
                for entity in parse_result.entities:
                    if hasattr(entity, "errors") and entity.errors:
                        for error in entity.errors:  # type: ignore[not-iterable]  # ty: ignore[not-iterable]
                            issues.append(
                                AnalysisIssue(
                                    type="syntax",
                                    severity="medium",
                                    message=str(error),
                                    line=entity.line_number or 1,
                                    column=1,
                                    file=str(tmp_file_path),
                                    rule="multi_language_parser",
                                )
                            )

                # Build metrics
                metrics = AnalysisMetrics(
                    lines_of_code=parse_result.lines_of_code,
                    complexity=parse_result.complexity_score or 0.0,
                    maintainability=max(0, 100 - (parse_result.complexity_score or 0) * 2),
                    test_coverage=None,
                    technical_debt=(
                        max(0, (parse_result.complexity_score or 0) - 10) * 0.5
                        if parse_result.complexity_score and parse_result.complexity_score > 10
                        else None
                    ),
                )

                # Generate suggestions
                suggestions = []
                if parse_result.complexity_score and parse_result.complexity_score > 10:
                    suggestions.append(
                        f"Consider refactoring to reduce complexity (current: {parse_result.complexity_score:.1f})"
                    )
                if len(parse_result.entities) == 0:
                    suggestions.append(
                        "No code entities detected. Verify code syntax and structure."
                    )
                if parse_result.errors:
                    suggestions.append("Address parsing errors to update analysis accuracy")
                if not suggestions:
                    suggestions.append(
                        "Code structure looks good. Consider adding documentation and tests."
                    )

                result = ApiAnalysisResult(
                    analysis_type=request.analysis_type,
                    success=parse_result.success,
                    issues=issues,
                    metrics=metrics,
                    suggestions=suggestions,
                    execution_time=time.time() - start_time,
                    metadata={
                        "analysis_id": analysis_id,
                        "language": language,
                        "parser_used": (
                            parse_result.parser_used.value
                            if hasattr(parse_result.parser_used, "value")
                            else str(parse_result.parser_used)
                        ),
                        "entities_found": len(parse_result.entities),
                        "relationships_found": len(parse_result.relationships),
                        "analysis_scope": "multi_language",
                    },
                )

            else:
                # Fall back to legacy processor
                raise HTTPException(
                    status_code=500,
                    detail=f"Legacy processor not supported for language: {language}",
                )

        finally:
            # Clean up temporary file
            if tmp_file_path.exists():
                os.unlink(tmp_file_path)

        return ApiAnalysisResponse(
            message=f"Multi-language analysis completed for {language}",
            result=result,
            request_id=analysis_id,
        )

    except HTTPException:  # Re-raise HTTP exceptions
        raise
    except (RuntimeError, ValueError, OSError, ImportError) as e:
        logger.exception("Multi-language analysis failed: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Multi-language analysis failed: {str(e)}"
        ) from e


async def analyze_multi_language_container(
    request: AnalysisRequest,
    analysis_id: str,
    start_time: float,
    tenant_id: str | None = None,
) -> ApiAnalysisResponse:
    """Analyze multiple files in a container with multi-language support."""
    file_paths: list[PathLib] | None = None
    try:
        # Exact, tenant-scoped lookup (404/403) + materialised live workspace.
        file_paths = await resolve_container_files(
            request.container_name,
            getattr(request, "file_paths", None),
            tenant_id=tenant_id,
        )

        if not file_paths:
            raise HTTPException(
                status_code=404,
                detail=f"No code files found in container '{request.container_name}'",
            )

        logger.info(
            "Starting multi-language analysis of %d files in container %s",
            len(file_paths),
            request.container_name,
        )

        # Group files by language for efficient processing
        files_by_language: dict[str, list[PathLib]] = {}
        for file_path in file_paths:
            # Detect language from file extension
            language = detect_language_from_file_path(str(file_path))
            if language:
                if language not in files_by_language:
                    files_by_language[language] = []
                files_by_language[language].append(file_path)

        if not files_by_language:
            raise HTTPException(
                status_code=422, detail="No supported language files found in container"
            )

        # Analyze files by language
        all_entities = []
        all_relationships = []
        all_issues = []
        total_lines = 0
        languages_analyzed = list(files_by_language.keys())

        for language, language_files in files_by_language.items():
            logger.info("Analyzing %d %s files", len(language_files), language)

            # Get processor for this language
            processor = await get_processor_for_language(language.lower())
            if not processor:
                logger.warning("No processor available for language: %s", language)
                continue

            # Analyze each file in this language
            for file_path in language_files:
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")

                    # Perform language-specific analysis
                    if isinstance(processor, LanguageProcessor):
                        parse_result = await processor.parse_file_with_metrics(
                            file_path, content, analysis_id
                        )
                    else:
                        # Skip unsupported processor types
                        logger.warning(
                            "Skipping file %s: processor type %s not supported",
                            file_path,
                            type(processor).__name__,
                        )
                        continue

                    if parse_result.success:
                        all_entities.extend(parse_result.entities)
                        all_relationships.extend(parse_result.relationships)
                        total_lines += parse_result.lines_of_code or 0

                        # Convert parsing errors to issues
                        for error in parse_result.errors or []:
                            all_issues.append(
                                AnalysisIssue(
                                    type="syntax",
                                    severity="medium",
                                    message=str(error),
                                    line=1,
                                    column=1,
                                    file=str(file_path),
                                    rule="multi_language_parser",
                                )
                            )

                except Exception as e:
                    logger.warning("Error analyzing %s: %s", file_path, e)
                    all_issues.append(
                        AnalysisIssue(
                            type="error",
                            severity="high",
                            message=f"Failed to analyze file: {str(e)}",
                            line=1,
                            column=1,
                            file=str(file_path),
                            rule="file_analysis_error",
                        )
                    )

        execution_time = time.time() - start_time

        # Build analysis metrics
        metrics = AnalysisMetrics(
            lines_of_code=total_lines,
            complexity=len(all_entities) * 0.1,  # Simple complexity estimate
            maintainability=max(0, 100 - len(all_issues) * 5),  # Simple maintainability score
            technical_debt=len(all_issues) * 0.5,  # Estimate technical debt in hours
        )

        # Build result - analysis_type is required so no fallback needed
        result = ApiAnalysisResult(
            analysis_type=request.analysis_type,
            success=True,
            issues=all_issues,
            metrics=metrics,
            suggestions=[
                f"Analyzed {len(file_paths)} files across {len(languages_analyzed)} languages",
                f"Found {len(all_entities)} entities and {len(all_relationships)} relationships",
                "Consider running language-specific analysis for deeper insights",
            ],
            execution_time=execution_time,
            metadata={
                "analysis_id": analysis_id,
                "entity_count": len(all_entities),
                "relationship_count": len(all_relationships),
                "languages_analyzed": languages_analyzed,
                "files_analyzed": len(file_paths),
                "container_name": request.container_name,
            },
        )

        return ApiAnalysisResponse(
            message=f"Multi-language analysis completed for container {request.container_name}: {len(file_paths)} files analyzed",
            result=result,
            request_id=analysis_id,
        )

    except HTTPException:  # Re-raise HTTP exceptions (including 404 for missing containers)
        raise
    except ValueError as e:
        # ValueError for missing code content should be 400, not 500
        logger.exception("Invalid multi-language analysis request: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid analysis request: {str(e)}") from e
    except Exception as e:
        logger.exception("Multi-language container analysis failed: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Multi-language analysis failed: {str(e)}"
        ) from e
    finally:
        release_materialized(file_paths)


def detect_language_from_file_path(file_path: str) -> str | None:
    """Detect programming language from file path/extension."""
    extension_map = {
        ".py": "python",
        ".java": "java",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".dart": "dart",
        ".vue": "vue",
        ".svelte": "svelte",
    }

    for ext, lang in extension_map.items():
        if file_path.endswith(ext):
            return lang

    return None
