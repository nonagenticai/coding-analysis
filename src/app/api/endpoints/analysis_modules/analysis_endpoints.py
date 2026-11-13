"""
Analysis Endpoints

Main analysis endpoint that handles different analysis types and routing.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
from pathlib import Path as PathLib
from typing import Any
from uuid import uuid7

import psycopg
from fastapi import APIRouter, Body, Depends, HTTPException, status

from auth.tenant import require_tenant
from clients.coding_containers import get_coding_containers_client
from processing.analyzers.deep import DeepCodeAnalyzer
from processing.analyzers.deep.multi_language_entities import collect_source_files
from processing.languages import language_registry
from services.github_provisioning_service import get_github_provisioning_service

from ...models.analysis_api_models import (
    AnalysisIssue,
    AnalysisMetrics,
    AnalysisRequest,
    AnalysisStatus,
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
from .analysis_utils import populate_graph_database
from .core_analysis import (
    ANALYSIS_RUNNERS,
    enhanced_analysis,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/analyze",
    response_model=ApiAnalysisResponse,
    summary="🔬 Unified Code Analysis",
    description="""
    🚀 **Comprehensive Code Analysis Engine** - Your one-stop solution for
    understanding and updating code quality

    ## Purpose
    This powerful endpoint provides comprehensive code analysis capabilities,
    consolidating all analysis
    functionality into a single, intelligent interface. Whether you're analyzing snippets, entire
    repositories, or validating code quality, this endpoint delivers actionable insights.

    ## How to Use
    **POST** `/api/analysis/analyze`

    **Basic Code Snippet Analysis:**
    ```json
    {
        "code": "def hello_world():\\n    print('Hello World!')",
        "language": "python",
        "analysis_type": "augmented"
    }
    ```

    **Repository Analysis (Containerized):**
    ```json
    {
        "container_name": "my-repo-container",
        "analysis_type": "deep"
    }
    ```

    **Code in Repository Context:**
    ```json
    {
        "code": "import my_module\\nresult = my_module.process_data()",
        "container_name": "my-repo-container",
        "analysis_type": "security"
    }
    ```

    ## Analysis Types & What They Do

    ### 🔍 **basic** - Quick Static Analysis
    - **Best for**: Fast syntax validation and basic metrics
    - **Speed**: ⚡ Very Fast (< 1 second)
    - **Detects**: Syntax errors, basic code smells, line counts
    - **Use when**: You need quick validation or CI/CD integration

    ### 🎯 **augmented** - Comprehensive AST Analysis
    - **Best for**: Detailed code structure analysis
    - **Speed**: ⚡ Fast (1-3 seconds)
    - **Detects**: Functions, classes, complexity, documentation gaps
    - **Use when**: You want detailed code insights without deep analysis

    ### 🌳 **ast** - Abstract Syntax Tree Analysis
    - **Best for**: Understanding code structure and patterns
    - **Speed**: ⚡ Fast (2-5 seconds)
    - **Detects**: Import patterns, function signatures, class hierarchies
    - **Use when**: You need structural analysis for refactoring

    ### 🔬 **deep** - In-Depth Quality Analysis
    - **Best for**: Comprehensive code quality assessment
    - **Speed**: 🕐 Moderate (5-15 seconds)
    - **Detects**: Quality issues, maintainability problems, technical debt
    - **Use when**: You want thorough quality assessment and recommendations

    ### 🔒 **security** - Security Vulnerability Detection
    - **Best for**: Finding security vulnerabilities and risks
    - **Speed**: 🕐 Moderate (5-10 seconds)
    - **Detects**: Injection flaws, unsafe functions, credential leaks
    - **Use when**: Security is a priority or before deployment

    ### ⚡ **performance** - Performance Analysis
    - **Best for**: Identifying performance bottlenecks
    - **Speed**: 🕐 Moderate (3-8 seconds)
    - **Detects**: Inefficient loops, blocking calls, memory issues
    - **Use when**: Optimizing application performance

    ## Smart Analysis Modes

    ### 📄 **Code Snippet Mode**
    - Provide `code` parameter with your code
    - Perfect for validating new code before committing
    - Works with any programming language

    ### 📁 **Repository Mode**
    - Provide `container_name` without code
    - Analyzes entire codebase in containerized environment
    - Generates comprehensive repository insights

    ### 🔗 **Context-Aware Mode**
    - Provide both `code` and `container_name`
    - Validates code snippet against repository context
    - Detects missing imports, undefined functions, type mismatches

    ## Parameters

    - **code** (string, optional): Code to analyze
      - Can be single line or multi-line
      - Supports all major programming languages
      - Use \\n for line breaks in JSON

    - **language** (string, optional): Programming language
      - `"python"`, `"javascript"`, `"java"`, `"cpp"`, etc.
      - Auto-detected if not specified

    - **analysis_type** (string, optional): Type of analysis to perform
      - Choose from: `basic`, `augmented`, `ast`, `deep`, `security`, `performance`
      - Default: `augmented` (recommended for most use cases)

    - **container_name** (string, optional): Container name for repository context
      - Use for repository or context-aware analysis
      - Get workspace names from `/api/workspaces/`
    """,
    responses={
        200: {"description": "Analysis completed successfully"},
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
            "description": "Container or resource not found",
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
            "description": "Analysis failed",
        },
    },
    operation_id="analyze_code",
)
async def analyze_code(
    request: AnalysisRequest = Body(..., description="Analysis request parameters"),
    tenant_id: str = Depends(require_tenant),
) -> ApiAnalysisResponse:
    """Perform unified code analysis based on the specified type."""
    analysis_id = None
    status_service = None

    try:
        start_time = time.time()

        # Generate unique analysis ID first
        analysis_id = str(uuid7())

        # Initialize status tracking service
        from services.analysis_status import AnalysisStatusService

        status_service = AnalysisStatusService()

        # Create analysis session BEFORE any validation that might fail
        try:
            logger.info("Creating analysis session for %s", request.analysis_type)
            analysis_type_str = request.analysis_type if request.analysis_type else "multi"
            await status_service.create_analysis_session(analysis_id, analysis_type_str)
        except (RuntimeError, ValueError, ConnectionError) as session_error:
            # If session creation fails, continue without status tracking but log the error
            logger.warning("Failed to create analysis session %s: %s", analysis_id, session_error)
            status_service = None

        # Basic input validation
        if not request.code and not request.container_name:
            error_msg = (
                "Must provide either 'code' content or 'container_name' for repository analysis"
            )
            if status_service:
                await status_service.update_status(
                    analysis_id,
                    AnalysisStatus.FAILED,
                    details={"error": error_msg},
                )
            raise ValueError(error_msg)

        # Handle repository context
        repository_context: dict[str, Any] | None = None
        if request.container_name:
            try:
                # Get container info for repository context
                github_service = get_github_provisioning_service()
                containers = await github_service.list_containers(tenant_id=tenant_id)

                # EXACT name match only. The old substring/prefix/suffix match let
                # "cv-ems" resolve to "cv-ems-p4" -- an answer about a different tree.
                container = next(
                    (c for c in containers if c.container_name == request.container_name),
                    None,
                )

                if not container:
                    # Better error message with debugging information
                    available_containers = [
                        c.container_name for c in containers[:5]
                    ]  # Show first 5
                    error_msg = (
                        f"Container '{request.container_name}' not found. "
                        f"Available containers: {available_containers}. "
                        "Please use a valid container name from the list above."
                    )
                    logger.warning(
                        "Container lookup failed for '%s'. Available: %s",
                        request.container_name,
                        available_containers,
                    )

                    if status_service:
                        await status_service.update_status(
                            analysis_id,
                            AnalysisStatus.FAILED,
                            details={
                                "error": error_msg,
                                "container_name_provided": request.container_name,
                            },
                        )
                    raise HTTPException(status_code=404, detail=error_msg)

                # Primary: shared volume path from container's codebase_path
                # (set to /app/provisioned_containers/{name}/repo by lifecycle manager)
                container_path = getattr(container, "codebase_path", None) or getattr(
                    container, "local_path", None
                )

                # Fallbacks for alternative mount layouts
                repo_container_path = f"/app/provisioned_containers/{container.container_name}/repo"
                fallback_path = "/app/analysis_workspace"

                repository_context = {
                    "container_path": container_path,
                    "repo_container_path": repo_container_path,
                    "fallback_path": fallback_path,
                    "repo_url": container.repo_url,
                    "language": request.language or "python",
                    "container_name": getattr(container, "container_name", request.container_name),
                    "tenant_id": tenant_id,
                }

                # Log the paths for debugging
                logger.info("Analysis using shared workspace path: %s", container_path)
                logger.info("Repository container path: %s", repo_container_path)
                if fallback_path:
                    logger.info("Fallback path available: %s", fallback_path)

                # Validate that we have a usable container path
                if not repository_context["container_path"]:
                    error_msg = (
                        f"Container '{request.container_name}' does not have a valid codebase path"
                    )
                    if status_service:
                        await status_service.update_status(
                            analysis_id,
                            AnalysisStatus.FAILED,
                            details={"error": error_msg},
                        )
                    raise HTTPException(status_code=404, detail=error_msg)

            except (RuntimeError, ValueError, ConnectionError) as container_error:
                error_msg = f"Failed to get container information: {str(container_error)}"
                if status_service:
                    await status_service.update_status(
                        analysis_id,
                        AnalysisStatus.FAILED,
                        details={"error": error_msg},
                    )
                raise HTTPException(status_code=404, detail=error_msg) from container_error

        # Determine what to analyze
        if not request.code and request.container_name:
            # Analyze entire repository
            if repository_context is None:
                error_msg = "Repository context is required but not available"
                if status_service:
                    await status_service.update_status(
                        analysis_id,
                        AnalysisStatus.FAILED,
                        details={"error": error_msg},
                    )
                raise HTTPException(status_code=404, detail=error_msg)
            result = await analyze_repository(request, repository_context, analysis_id, start_time)
        elif request.code and request.container_name:
            # Analyze code in repository context
            if repository_context is None:
                error_msg = "Repository context is required but not available"
                if status_service:
                    await status_service.update_status(
                        analysis_id,
                        AnalysisStatus.FAILED,
                        details={"error": error_msg},
                    )
                raise HTTPException(status_code=404, detail=error_msg)
            result = await analyze_code_with_context(
                request, repository_context, analysis_id, start_time
            )
        elif request.code:
            # Analyze standalone code
            result = await analyze_standalone_code(request, analysis_id, start_time)
        else:
            error_msg = (
                "Must provide either 'code' content or 'container_name' for repository analysis"
            )
            if status_service:
                await status_service.update_status(
                    analysis_id,
                    AnalysisStatus.FAILED,
                    details={"error": error_msg},
                )
            raise ValueError(error_msg)

        # Update status to completed if we have status service
        if status_service:
            await status_service.update_status(
                analysis_id, AnalysisStatus.COMPLETED, progress=100.0
            )

        return ApiAnalysisResponse(
            message=f"{request.analysis_type} analysis completed",
            result=result,
            request_id=analysis_id,
        )

    except ValueError as e:
        # Update status to failed if we have status service and analysis_id
        if status_service and analysis_id:
            try:
                await status_service.update_status(
                    analysis_id,
                    AnalysisStatus.FAILED,
                    details={"error": str(e)},
                )
            except (RuntimeError, ValueError, ConnectionError) as status_error:
                logger.warning(
                    "Failed to update analysis status for %s: %s",
                    analysis_id,
                    status_error,
                )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid analysis request: {str(e)}",
        ) from e
    except (RuntimeError, OSError, ImportError, psycopg.Error) as e:
        # Update status to failed if we have status service and analysis_id
        if status_service and analysis_id:
            try:
                await status_service.update_status(
                    analysis_id,
                    AnalysisStatus.FAILED,
                    details={"error": str(e)},
                )
            except (RuntimeError, ValueError, ConnectionError) as status_error:
                logger.warning(
                    "Failed to update analysis status for %s: %s",
                    analysis_id,
                    status_error,
                )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        ) from e


# Bounded retry for remote source discovery, to absorb the provision-time race
# where a just-created container's exec endpoint / repo clone is not ready yet.
_REMOTE_FIND_ATTEMPTS = 4
_REMOTE_FIND_BACKOFF_S = 2.0


def _build_remote_find_command() -> list[str]:
    """Build a ``find`` command that discovers every supported source file.

    Covers all languages registered in ``language_registry`` (Python,
    JavaScript, TypeScript, Java, Go, Rust, C/C++, C#) rather than ``.py`` only.
    """
    extensions = sorted(language_registry.get_supported_extensions())
    name_clauses: list[str] = []
    for index, ext in enumerate(extensions):
        if index:
            name_clauses.append("-o")
        name_clauses.extend(["-name", f"*{ext}"])
    return [
        "find",
        "/workspace",
        "-type",
        "f",
        "(",
        *name_clauses,
        ")",
        "-not",
        "-path",
        "*/.*",
        "-not",
        "-path",
        "*/__pycache__/*",
        "-not",
        "-path",
        "*/node_modules/*",
    ]


async def _fetch_remote_source_files(
    container_name: str,
    tenant_id: str | None,
) -> tuple[str | None, list[PathLib]]:
    """Fetch source files from a remote K8s container via coding-containers exec.

    Uses ``find`` to discover supported source files across all languages and
    ``cat`` to retrieve their content, writing them into a temporary directory
    that preserves relative paths.

    Returns:
        A tuple of (temp_directory_path, list_of_source_file_paths).
        If fetching fails or no files are found the temp dir is still returned
        (caller must clean up) but the list will be empty.
    """
    client = get_coding_containers_client()

    # Discover supported source files inside the container's /workspace.
    #
    # A freshly-provisioned container races the build: its exec endpoint may not
    # be ready yet (the client surfaces that as exit_code -1 from an HTTP 500),
    # or the repo clone into /workspace may not have landed (find exits non-zero
    # on a missing path). Both manifest as a non-zero find exit and must NOT be
    # conflated with a genuinely-empty repo (exit 0, no stdout) — otherwise the
    # graph build silently persists an empty graph and architecture analysis
    # returns zero components. Retry the discovery a bounded number of times with
    # backoff to absorb that readiness race before giving up.
    find_result: dict[str, Any] = {}
    for attempt in range(_REMOTE_FIND_ATTEMPTS):
        find_result = await client.execute(
            container_name,
            command=_build_remote_find_command(),
            workdir="/workspace",
            timeout=30,
            tenant_id=tenant_id,
        )
        if find_result.get("exit_code", -1) == 0:
            break
        if attempt < _REMOTE_FIND_ATTEMPTS - 1:
            logger.warning(
                "Remote find for %s not ready (exit %s, attempt %d/%d); retrying",
                container_name,
                find_result.get("exit_code"),
                attempt + 1,
                _REMOTE_FIND_ATTEMPTS,
            )
            await asyncio.sleep(_REMOTE_FIND_BACKOFF_S * (attempt + 1))

    if find_result.get("exit_code", -1) != 0:
        logger.warning(
            "Remote find failed for %s after %d attempts (exit %s): %s",
            container_name,
            _REMOTE_FIND_ATTEMPTS,
            find_result.get("exit_code"),
            find_result.get("stderr", ""),
        )
        return None, []

    stdout = find_result.get("stdout", "")
    remote_paths = [p.strip() for p in stdout.splitlines() if p.strip()]
    if not remote_paths:
        logger.info("Remote container %s has no source files under /workspace", container_name)
        return None, []

    logger.info(
        "Found %d source files in remote container %s",
        len(remote_paths),
        container_name,
    )

    # Create a temp directory and fetch each file
    temp_dir = tempfile.mkdtemp(prefix="analysis_remote_")
    fetched_files: list[PathLib] = []

    for remote_path in remote_paths:
        # Compute path relative to /workspace
        if remote_path.startswith("/workspace/"):
            rel_path = remote_path[len("/workspace/") :]
        elif remote_path.startswith("/workspace"):
            rel_path = remote_path[len("/workspace") :]
        else:
            rel_path = remote_path.lstrip("/")

        if not rel_path:
            continue

        cat_result = await client.execute(
            container_name,
            command=["cat", remote_path],
            workdir="/workspace",
            timeout=10,
            tenant_id=tenant_id,
        )

        if cat_result.get("exit_code", -1) != 0:
            logger.debug(
                "Failed to cat %s in %s: %s",
                remote_path,
                container_name,
                cat_result.get("stderr", ""),
            )
            continue

        content = cat_result.get("stdout", "")
        local_file = PathLib(temp_dir) / rel_path
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text(content, encoding="utf-8")
        fetched_files.append(local_file)

    logger.info(
        "Fetched %d/%d files from remote container %s into %s",
        len(fetched_files),
        len(remote_paths),
        container_name,
        temp_dir,
    )
    return temp_dir, fetched_files


async def analyze_repository(
    request: AnalysisRequest,
    context: dict[str, Any],
    analysis_id: str,
    start_time: float,
) -> ApiAnalysisResult:
    """Analyze entire repository in container."""
    if not context or not context.get("container_path"):
        raise HTTPException(
            status_code=404,
            detail="Container path not available for repository analysis",
        )

    # Try multiple paths to find the repository files
    path_candidates = [
        context["container_path"],  # Shared workspace path
        context.get("repo_container_path"),  # Repository container path
        context.get("fallback_path"),  # Container attribute path
    ]

    container_path = None
    source_files: list[PathLib] = []
    remote_temp_dir: str | None = None

    # Find the first path that contains supported source files
    for candidate_path in path_candidates:
        if not candidate_path:
            continue

        logger.info("Trying repository path: %s", candidate_path)

        try:
            path_obj = PathLib(candidate_path)
            if path_obj.exists():
                test_files = collect_source_files(path_obj)
                logger.info("Found %d source files at %s", len(test_files), candidate_path)

                if len(test_files) > 0:
                    container_path = candidate_path
                    source_files = test_files
                    logger.info(
                        "Selected path: %s with %d source files",
                        container_path,
                        len(source_files),
                    )
                    break
                logger.info("Path %s exists but contains no supported source files", candidate_path)
            else:
                logger.info("Path %s does not exist", candidate_path)
        except (OSError, PermissionError, RuntimeError) as e:
            logger.warning("Error checking path %s: %s", candidate_path, e)

    # Fallback: fetch files from the remote container via coding-containers exec
    if not container_path or len(source_files) == 0:
        container_name = context.get("container_name")
        tenant_id = context.get("tenant_id")
        if container_name:
            logger.info(
                "Local paths not found, fetching files from remote container %s",
                container_name,
            )
            remote_temp_dir, source_files = await _fetch_remote_source_files(
                container_name, tenant_id
            )
            if remote_temp_dir and source_files:
                container_path = remote_temp_dir

    try:
        return await _run_repository_analysis(
            request, context, analysis_id, start_time, container_path, source_files, path_candidates
        )
    finally:
        # Clean up the temporary directory if we created one
        if remote_temp_dir:
            try:
                shutil.rmtree(remote_temp_dir, ignore_errors=True)
                logger.debug("Cleaned up remote temp dir: %s", remote_temp_dir)
            except OSError as cleanup_err:
                logger.warning("Failed to clean up temp dir %s: %s", remote_temp_dir, cleanup_err)


async def _run_repository_analysis(
    request: AnalysisRequest,
    context: dict[str, Any],
    analysis_id: str,
    start_time: float,
    container_path: str | None,
    source_files: list[PathLib],
    path_candidates: list[str | None],
) -> ApiAnalysisResult:
    """Run the actual repository analysis on resolved files.

    Separated from ``analyze_repository`` so the caller can handle temp-dir
    cleanup in a ``finally`` block.

    Pure-Python repositories keep the existing Python deep-analysis path
    unchanged. Repositories containing any non-Python supported file get a
    merged multi-language graph (entities + relationships) so the architecture
    graph renders for any language the platform supports.
    """
    if not container_path or len(source_files) == 0:
        # Return a successful but empty analysis with helpful information
        logger.warning(
            "No supported source files found in any candidate paths: %s",
            [p for p in path_candidates if p],
        )
        return ApiAnalysisResult(
            analysis_type=request.analysis_type,
            success=True,
            issues=[],
            metrics=AnalysisMetrics(
                lines_of_code=0,
                complexity=0.0,
                maintainability=10.0,  # Base score for empty analysis
                test_coverage=None,
                technical_debt=None,
            ),
            suggestions=[
                "No supported source files found in the repository for analysis",
                f"Searched paths: {[p for p in path_candidates if p]}",
                "This may indicate a path configuration issue",
            ],
            execution_time=time.time() - start_time,
            metadata={
                "analysis_id": analysis_id,
                "analysis_scope": "full_repository",
                "repository_url": context.get("repo_url"),
                "total_issues": 0,
                "files_analyzed": 0,
                "searched_paths": [p for p in path_candidates if p],
                "selected_path": None,
            },
        )

    logger.info(
        "Analyzing repository at path: %s with %d source files",
        container_path,
        len(source_files),
    )

    # Use DeepCodeAnalyzer for full repository analysis
    # Get language from context or default to python
    language = context.get("language", "python")
    try:
        analyzer = DeepCodeAnalyzer(container_path, language=language)
        # Python deep-analysis (issues/metrics). For a non-Python repo this is
        # cheap — ``python_files`` is empty so the sub-analyzers no-op.
        deep_result = analyzer.analyze_codebase()
        # For polyglot / non-Python repos, replace the Python-only graph with
        # the merged multi-language graph so the architecture tab renders.
        is_polyglot = any(f.suffix.lower() != ".py" for f in source_files)
        if is_polyglot:
            ml_entities, ml_relationships = await analyzer.analyze_graph_multilang()
            deep_result.entities = ml_entities
            deep_result.relationships = ml_relationships
    except Exception as analyzer_error:
        logger.error("Failed to create or run DeepCodeAnalyzer: %s", analyzer_error)
        # Return a basic result if deep analysis fails
        return ApiAnalysisResult(
            analysis_type=request.analysis_type,
            success=True,
            issues=[],
            metrics=AnalysisMetrics(
                lines_of_code=sum(
                    len(f.read_text().splitlines()) for f in source_files if f.exists()
                ),
                complexity=0.0,
                maintainability=5.0,
                test_coverage=None,
                technical_debt=None,
            ),
            suggestions=[
                "Deep analysis failed - basic file count analysis performed instead",
                f"Error: {str(analyzer_error)}",
                f"Repository contains {len(source_files)} source files",
            ],
            execution_time=time.time() - start_time,
            metadata={
                "analysis_id": analysis_id,
                "analysis_scope": "failed_deep_fallback",
                "repository_url": context.get("repo_url"),
                "total_issues": 0,
                "files_analyzed": len(source_files),
                "selected_path": container_path,
                "error": str(analyzer_error),
            },
        )

    # Populate graph database with analysis results
    graph_nodes = 0
    graph_edges = 0
    try:
        # Get container name for database identification
        container_name = context.get("container_name", "unknown_container")
        logger.info("Populating graph database for container: %s", container_name)

        # Create analysis data from deep_result
        analysis_data: dict[str, list[dict[str, Any]]] = {
            "nodes": [],
            "edges": [],
        }

        # Convert deep analysis entities to graph nodes
        if hasattr(deep_result, "entities") and deep_result.entities:
            for entity in deep_result.entities:
                node_data = {
                    "id": entity.id,
                    "type": entity.type,
                    "name": entity.name,
                    "file_path": getattr(entity, "file_path", None),
                    "properties": {
                        "line_number": entity.line_number,
                        "docstring": getattr(entity, "docstring", None),
                        "metadata": getattr(entity, "metadata", {}),
                    },
                }
                analysis_data["nodes"].append(node_data)

        # Convert relationships to graph edges
        if hasattr(deep_result, "relationships") and deep_result.relationships:
            for relationship in deep_result.relationships:
                # ``EntityRelation`` keeps metadata in ``properties``; the
                # Python-path ``CodeRelationship`` uses ``metadata``.
                edge_properties = getattr(relationship, "properties", None) or getattr(
                    relationship, "metadata", {}
                )
                edge_data = {
                    "id": f"edge_{len(analysis_data['edges'])}",
                    "source_id": getattr(relationship, "source_id", ""),
                    "target_id": getattr(relationship, "target_id", ""),
                    "type": getattr(relationship, "type", "unknown"),
                    "properties": edge_properties,
                }
                analysis_data["edges"].append(edge_data)

        # Also create nodes from file structure if no entities found
        if len(analysis_data["nodes"]) == 0:
            logger.info("No entities found in deep analysis, creating basic file structure nodes")
            for i, file_path in enumerate(source_files):
                node_data = {
                    "id": f"file_{i}",
                    "type": "file",
                    "name": file_path.name,
                    "file_path": str(file_path),
                    "properties": {
                        "file_size": (file_path.stat().st_size if file_path.exists() else 0),
                        "file_type": language_registry.detect_language(file_path) or "unknown",
                    },
                }
                analysis_data["nodes"].append(node_data)

        # Populate the database using the function (tenant-attributed for isolation)
        success = await populate_graph_database(
            container_name, analysis_data, tenant_id=context.get("tenant_id")
        )

        if success:
            graph_nodes = len(analysis_data["nodes"])
            graph_edges = len(analysis_data["edges"])
            logger.info(
                "✅ Graph database populated: %d nodes, %d edges from %d source files",
                graph_nodes,
                graph_edges,
                len(source_files),
            )
        else:
            logger.error("❌ Graph database population failed!")

    except (RuntimeError, ValueError, psycopg.Error) as graph_error:
        logger.error("❌ Failed to populate graph database: %s", graph_error)
        logger.exception("Graph population error details:")
        # Continue with analysis even if graph population fails

    # Persist a dependency-report snapshot while the container is still live, so
    # the fr Dependency Report can degrade to stored data after the container is
    # deprovisioned (mirrors the graph's persisted-read fallback). Best-effort:
    # a failure here must never fail the analysis.
    try:
        from services.code_graph.dependency_snapshot import store_dependency_report_snapshot

        await store_dependency_report_snapshot(container_name, context.get("tenant_id"))
    except Exception as dep_snap_error:  # noqa: BLE001 - best-effort, never propagate
        logger.warning(
            "Dependency snapshot persistence skipped for %s: %s",
            container_name,
            dep_snap_error,
        )

    # Convert to API format
    issues = []
    for issue in deep_result.issues:
        # Defensive handling of enum values - could be enum or string
        type_value = issue.type.value if hasattr(issue.type, "value") else str(issue.type)
        severity_value = (
            issue.severity.value if hasattr(issue.severity, "value") else str(issue.severity)
        )

        issues.append(
            AnalysisIssue(
                type=type_value,
                severity=severity_value,
                message=issue.title,
                line=issue.line_number if issue.line_number is not None else 0,
                column=1,
                file=issue.file_path or "unknown",
                rule=getattr(issue, "rule", "deep-analysis"),
            )
        )

    # Extract metrics from the metrics dictionary
    metrics_data = deep_result.metrics if hasattr(deep_result, "metrics") else {}

    metrics = AnalysisMetrics(
        lines_of_code=metrics_data.get("total_lines", 0),
        complexity=10.0
        - deep_result.overall_score,  # Convert quality score to complexity (inverse)
        maintainability=deep_result.overall_score * 10,  # Convert 0-10 score to 0-100
        test_coverage=metrics_data.get("test_coverage", None),
        technical_debt=metrics_data.get("technical_debt", None),
    )

    # Use recommendations instead of suggestions
    suggestions = deep_result.recommendations if hasattr(deep_result, "recommendations") else []

    return ApiAnalysisResult(
        analysis_type=request.analysis_type,
        success=True,
        issues=issues,
        metrics=metrics,
        suggestions=suggestions,
        execution_time=time.time() - start_time,
        metadata={
            "analysis_id": analysis_id,
            "analysis_scope": "full_repository",
            "repository_url": context.get("repo_url"),
            "total_issues": len(issues),
            "files_analyzed": len(source_files),
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "selected_path": container_path,
        },
    )


async def analyze_code_with_context(
    request: AnalysisRequest,
    context: dict[str, Any],
    analysis_id: str,
    start_time: float,
) -> ApiAnalysisResult:
    """Analyze code snippet with repository context."""
    # For now, perform standalone analysis with context metadata
    # Future enhancement: validate imports against repository
    result = await analyze_standalone_code(request, analysis_id, start_time)

    # Add context metadata
    if result.metadata:
        result.metadata.update(
            {
                "repository_context": True,
                "repository_url": context.get("repo_url"),
                "container_name": context.get("container_name"),
            }
        )

    return result


async def analyze_standalone_code(
    request: AnalysisRequest, analysis_id: str, start_time: float
) -> ApiAnalysisResult:
    """Analyze standalone code snippet."""
    # Route through THE dispatch table (capabilities are computed from it).
    runner = ANALYSIS_RUNNERS.get(request.analysis_type, enhanced_analysis)
    return await runner(request, analysis_id, start_time)
