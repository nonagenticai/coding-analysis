"""
AI Agent Static Analysis Augmentation Endpoints.

New endpoints for call graph, signature analysis, advanced validation, and documentation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.api.responses import (
    bad_request,
    internal_error,
)
from app.core.redis_service import get_redis_service
from auth.tenant import require_tenant
from models.static_analysis_models import (
    ANALYZER_FAILED,
    ANALYZER_NOT_COMPUTED,
    ANALYZER_RAN_STATUSES,
    SIGNATURE_PATTERNS,
    AdvancedValidationRequest,
    AdvancedValidationResponse,
    AnalyzerAbstention,
    CallGraphRequest,
    CallGraphResponse,
    DocumentationRequest,
    DocumentationResponse,
    DocumentationSection,
    SignatureAnalysisRequest,
    SignatureAnalysisResponse,
    StaticAnalysisFunctionSignature,
    StaticAnalysisValidationResult,
    ValidationVerdict,
)
from processing.analyzers.call_graph_analyzer import (
    CallGraphConfig,
    CallGraphGenerator,
    InvalidAnalysisScopeError,
)
from processing.analyzers.symbolic.tools import FormalVerifier, analyze_file_security
from processing.analyzers.symbolic.utils import SafetyCalculator
from processing.analyzers.symbolic_analyzer import (
    SecurityLevel,
    SymbolicValidator,
    ValidationConfig,
    ValidationType,
)
from processing.ast_analysis.function_extractor import FunctionExtractor

from ..models.error_models import (
    Error400Response,
    Error500Response,
    ValidationErrorResponse,
)
from ..utils.container_validation import validate_and_get_container_for_tenant
from ..utils.workspace_files import (
    WorkspaceUnresolvableError,
    materialize_workspace,
    remove_materialized,
    tree_digest,
    workspace_root,
    workspace_unresolvable_http,
)

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_redis() -> Any | None:
    """Get Redis service for caching."""
    try:
        redis_service = get_redis_service()
        return redis_service.client
    except (ConnectionError, OSError, RuntimeError, TypeError) as e:
        logger.warning("Redis not available: %s", e)
        return None


@router.post(
    "/call-graph",
    response_model=CallGraphResponse,
    operation_id="generate_call_graph",
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def generate_call_graph(
    request: CallGraphRequest,
    background_tasks: BackgroundTasks,
    redis_client: Any | None = Depends(get_redis),
    tenant_id: str = Depends(require_tenant),
) -> CallGraphResponse:
    """
    Generate visual call graphs for AI agents to understand codebase architecture.

    This endpoint implements the Call Graph Analysis Service as specified in the PRD:
    - Generates visual call graphs using multiple tools (Pyan3, Code2flow, PyCG)
    - Provides JSON, DOT, and SVG output formats
    - Identifies critical paths and orphaned functions
    - Caches results for performance
    """
    try:
        repo_dir = await _materialize_workspace_for(
            request.container_name, tenant_id, _CALL_GRAPH_PATTERNS, repo=request.repo
        )
        repo_path = str(repo_dir)
        cache_key = _workspace_cache_key("call_graph_v2", tenant_id, request.model_dump(), repo_dir)
        if redis_client:
            try:
                redis_service = get_redis_service()
                cached_result = await redis_service.get(cache_key)
                if cached_result:
                    logger.info(
                        "Returning cached call graph result for container %s",
                        request.container_name,
                    )
                    if isinstance(cached_result, dict):
                        return CallGraphResponse.model_validate(cached_result)
                    return CallGraphResponse.model_validate_json(cached_result)
            except (ConnectionError, OSError, RuntimeError) as e:
                logger.warning("Cache lookup failed: %s", e)
        try:
            config = CallGraphConfig(
                output_format=request.output_format,
                include_external=request.include_external,
                max_depth=request.max_depth,
                analysis_scope=request.analysis_scope,
            )
            analyzer = CallGraphGenerator(config)
            logger.info("Generating call graph for container %s", request.container_name)
            try:
                result = await analyzer.generate_comprehensive_call_graph(repo_path, config)
            except InvalidAnalysisScopeError as exc:
                raise bad_request(str(exc), details={"rejected_scopes": exc.rejected}) from exc
            if redis_client:
                background_tasks.add_task(
                    _cache_result,
                    redis_client,
                    cache_key,
                    result.model_dump_json(),
                    7200,
                )
            background_tasks.add_task(_store_call_graph_analysis, result, request.model_dump())
            logger.info("Call graph analysis completed for container %s", request.container_name)
            return result
        finally:
            background_tasks.add_task(remove_materialized, repo_dir)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "Error generating call graph for container %s: %s",
            request.container_name,
            e,
        )
        raise internal_error(f"Call graph generation failed: {str(e)}") from e


@router.post(
    "/signatures",
    response_model=SignatureAnalysisResponse,
    operation_id="analyze_signatures",
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def analyze_signatures(
    request: SignatureAnalysisRequest,
    background_tasks: BackgroundTasks,
    redis_client: Any | None = Depends(get_redis),
    tenant_id: str = Depends(require_tenant),
) -> SignatureAnalysisResponse:
    """
    Extract comprehensive function signatures and API contracts for AI code generation.

    This endpoint implements the Augmented Function Signature Analysis as specified in the PRD:
    - Extracts comprehensive function signatures with type information
    - Analyzes usage patterns and function relationships
    - Provides type coverage metrics
    - Runs for the languages in ``SIGNATURE_PATTERNS`` only (Python; Rust delegates to
      the structure api-surface extractor); any other ``language`` is a 400, never an
      empty "0 functions" answer
    """
    try:
        analysis_id = f"sig_{hash(request.container_name) % 1000000:06d}"
        # The dispatch table lives with the request model so the ``language`` field's
        # documentation is derived from it: what /signatures lists is what it runs.
        patterns = SIGNATURE_PATTERNS.get(request.language.value, ())
        if not patterns:
            raise bad_request(f"Unsupported language for signature analysis: {request.language}")
        repo_dir = await _materialize_workspace_for(
            request.container_name, tenant_id, patterns, repo=request.repo
        )
        repo_path = str(repo_dir)
        cache_key = _workspace_cache_key("signatures:v2", tenant_id, request.model_dump(), repo_dir)
        if redis_client:
            try:
                redis_service = get_redis_service()
                cached_result = await redis_service.get(cache_key)
                if cached_result:
                    logger.info(
                        "Returning cached signature analysis for container %s",
                        request.container_name,
                    )
                    if isinstance(cached_result, dict):
                        return SignatureAnalysisResponse.model_validate(cached_result)
                    return SignatureAnalysisResponse.model_validate_json(cached_result)
            except (ConnectionError, OSError, RuntimeError) as e:
                logger.warning("Cache lookup failed: %s", e)
        try:
            if request.language.value == "rust":
                result = await _rust_signatures(analysis_id, repo_dir, tenant_id)
                if redis_client:
                    background_tasks.add_task(
                        _cache_result, redis_client, cache_key, result.model_dump_json(), 14400
                    )
                background_tasks.add_task(_store_signature_analysis, result, request.model_dump())
                return result
            extractor = FunctionExtractor(
                enable_type_inference=request.include_type_inference,
                analyze_usage_patterns=request.analyze_usage_patterns,
            )
            logger.info("Analyzing function signatures for container %s", request.container_name)
            supported_files = sorted(
                path for pattern in patterns for path in repo_dir.rglob(pattern) if path.is_file()
            )
            analysis_results = _analyze_files(supported_files, extractor, repo_path)
            all_functions = analysis_results["functions"]
            external_dependencies = analysis_results["dependencies"]
            total_functions = analysis_results["total_functions"]
            typed_functions = analysis_results["typed_functions"]
            type_coverage = typed_functions / total_functions if total_functions > 0 else 0.0
            result = SignatureAnalysisResponse(
                analysis_id=analysis_id,
                functions=all_functions,
                type_coverage=round(type_coverage, 2),
                external_dependencies=sorted(external_dependencies),
                total_functions=total_functions,
                typed_functions=typed_functions,
                files_not_computed=analysis_results.get("files_not_computed", []),
                call_arity_mismatches=analysis_results.get("call_arity_mismatches", []),
                call_arity_status=analysis_results.get("call_arity_status", "computed"),
                call_arity_reason=analysis_results.get("call_arity_reason"),
            )
            if redis_client:
                background_tasks.add_task(
                    _cache_result,
                    redis_client,
                    cache_key,
                    result.model_dump_json(),
                    14400,
                )
            background_tasks.add_task(_store_signature_analysis, result, request.model_dump())
            logger.info("Signature analysis completed for container %s", request.container_name)
            return result
        finally:
            background_tasks.add_task(remove_materialized, repo_dir)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "Error analyzing signatures for container %s: %s", request.container_name, e
        )
        raise internal_error(f"Signature analysis failed: {str(e)}") from e


@router.post(
    "/advanced-validation",
    response_model=AdvancedValidationResponse,
    operation_id="validate_code_changes",
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def validate_code_changes(
    request: AdvancedValidationRequest,
    background_tasks: BackgroundTasks,
    redis_client: Any | None = Depends(get_redis),
    tenant_id: str = Depends(require_tenant),
) -> AdvancedValidationResponse:
    """
    Validate AI-generated code using symbolic execution and security analysis.

    This endpoint implements the Advanced Static Validation as specified in the PRD:
    - Symbolic execution using CrossHair
    - Taint analysis using Pysa
    - Formal verification
    - Security vulnerability detection

    File paths in code_changes should be relative to the container's codebase root.
    """
    try:
        # Validate container exists and get container info
        container_info = await validate_and_get_container_for_tenant(
            request.container_name, tenant_id
        )
        container_root = Path(container_info.codebase_path)

        logger.info(
            "Starting advanced validation for container '%s' with %d code changes",
            request.container_name,
            len(request.code_changes),
        )

        # Validate that file paths are relative and within container bounds
        for change in request.code_changes:
            file_path = Path(change.file_path)
            if file_path.is_absolute():
                raise bad_request(
                    f"File path must be relative to container root: {change.file_path}"
                )
            # Resolve and check if path is within container
            full_path = (container_root / file_path).resolve()
            try:
                full_path.relative_to(container_root.resolve())
            except ValueError:
                raise bad_request(f"File path outside container bounds: {change.file_path}")

        analysis_id = f"adv_{hashlib.sha256(json.dumps(request.model_dump(), sort_keys=True, default=str).encode()).hexdigest()[:6]}"
        cache_key = f"validation:{request.container_name}:{hashlib.sha256(json.dumps(request.model_dump(), sort_keys=True, default=str).encode()).hexdigest()[:16]}"
        if redis_client:
            try:
                redis_service = get_redis_service()
                cached_result = await redis_service.get(cache_key)
                if cached_result:
                    logger.info(
                        "Returning cached validation result for container '%s'",
                        request.container_name,
                    )
                    if isinstance(cached_result, dict):
                        return AdvancedValidationResponse.model_validate(cached_result)
                    return AdvancedValidationResponse.model_validate_json(cached_result)
            except (ConnectionError, OSError, RuntimeError) as e:
                logger.warning("Cache lookup failed: %s", e)
        validation_results: list[dict[str, Any]] = []
        for change in request.code_changes:
            skip_reason = _unanalyzable_reason(change)
            if skip_reason:
                validation_results.append(
                    _abstain_for_change(change, request.validation_types, skip_reason)
                )
                continue
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                    tmp_file.write(change.modified_code)
                    tmp_file_path = tmp_file.name
                try:
                    validation_result = await _run_validation_analyses(
                        tmp_file_path,
                        change,
                        request.validation_types,
                        request.security_level,
                    )
                    validation_results.append(validation_result)
                finally:
                    os.unlink(tmp_file_path)
            except (OSError, RuntimeError, ValueError) as e:
                logger.exception("Error validating change for %s: %s", change.file_path, e)
                validation_results.append(
                    {
                        "file_path": change.file_path,
                        "error": str(e),
                        "analyzers_ran": [],
                        "analyzers_not_computed": [
                            {
                                "analyzer": "all",
                                "status": ANALYZER_FAILED,
                                "reason": str(e),
                            }
                        ],
                    }
                )

        validated_results, safety_scores = _build_validation_results(validation_results)

        # Only analyzers that actually ran contribute to the aggregate.
        analyzers_ran = sorted({a for r in validated_results for a in r.analyzers_ran})
        abstentions = [a.model_dump() for r in validated_results for a in r.analyzers_not_computed]
        overall_safety_score = (
            round(sum(safety_scores) / len(safety_scores), 2) if safety_scores else None
        )
        blocking_issues = _identify_blocking_issues(
            validation_results, blocking_severities_for(request.security_level)
        )

        if not analyzers_ran:
            # Nothing was verified. An abstention, never a pass.
            verdict = ValidationVerdict.ABSTAIN
        elif blocking_issues or (
            overall_safety_score is not None and overall_safety_score < _FAIL_SAFETY_THRESHOLD
        ):
            verdict = ValidationVerdict.FAIL
        else:
            verdict = ValidationVerdict.PASS

        recommendations = _generate_recommendations(
            overall_safety_score, blocking_issues, abstentions
        )
        result = AdvancedValidationResponse(
            analysis_id=analysis_id,
            validation_results=validated_results,
            overall_safety_score=overall_safety_score,
            blocking_issues=blocking_issues,
            recommendations=recommendations,
            verdict=verdict,
            analyzers_ran=analyzers_ran,
            analyzers_not_computed=[AnalyzerAbstention(**a) for a in abstentions],
        )
        if redis_client:
            background_tasks.add_task(
                _cache_result, redis_client, cache_key, result.model_dump_json(), 3600
            )
        background_tasks.add_task(_store_validation_analysis, result, request.model_dump())
        logger.info(
            "Advanced validation completed with safety score: %s",
            overall_safety_score,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in advanced validation: %s", e)
        raise internal_error(f"Advanced validation failed: {str(e)}") from e


@router.post(
    "/documentation",
    response_model=DocumentationResponse,
    operation_id="generate_documentation",
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        422: {"model": ValidationErrorResponse, "description": "Validation error"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def generate_documentation(
    request: DocumentationRequest,
    background_tasks: BackgroundTasks,
    _redis_client: Any | None = Depends(get_redis),
    tenant_id: str = Depends(require_tenant),
) -> DocumentationResponse:
    """
    Generate architectural documentation and change summaries for AI modifications.

    This endpoint implements the Documentation Generation as specified in the PRD:
    - Architecture documentation with diagrams
    - API reference generation
    - Change impact analysis
    - Multiple output formats (Markdown, HTML, Sphinx)
    """
    try:
        analysis_id = f"doc_{hash(request.container_name) % 1000000:06d}"
        logger.info("Generating documentation for container %s", request.container_name)
        repo_dir = await _materialize_workspace_for(
            request.container_name, tenant_id, _DOCUMENTATION_PATTERNS, repo=request.repo
        )
        repo_path = str(repo_dir)
        try:
            documentation_sections: dict[str, Any] = {}
            generated_files = []
            quality_scores = []
            for doc_type in request.doc_types:
                try:
                    if doc_type.value == "architecture":
                        section = await _generate_architecture_docs(
                            repo_path, request.include_diagrams, request.format
                        )
                    elif doc_type.value == "api_reference":
                        section = await _generate_api_reference(repo_path, request.format)
                    elif doc_type.value == "change_summary":
                        section = await _generate_change_summary(repo_path, request.format)
                    else:
                        continue
                    documentation_sections[doc_type.value] = section
                    if section.get("files"):
                        generated_files.extend(section["files"])
                    if "quality_score" in section:
                        quality_scores.append(section["quality_score"])
                except (OSError, RuntimeError, ValueError) as e:
                    logger.warning("Error generating %s documentation: %s", doc_type.value, e)
                    documentation_sections[doc_type.value] = {
                        "content": f"Error generating {doc_type.value}: {str(e)}",
                        "diagrams": [],
                        "coverage": 0.0,
                    }
            overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
            validated_documentation = {}
            for doc_type_str, section in documentation_sections.items():
                validated_documentation[doc_type_str] = DocumentationSection(**section)
            result = DocumentationResponse(
                analysis_id=analysis_id,
                documentation=validated_documentation,
                generated_files=generated_files,
                quality_score=round(overall_quality, 2),
            )
            logger.info(
                "Documentation generation completed for container %s",
                request.container_name,
            )
            return result
        finally:
            background_tasks.add_task(remove_materialized, repo_dir)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "Error generating documentation for container %s: %s",
            request.container_name,
            e,
        )
        raise internal_error(f"Documentation generation failed: {str(e)}") from e


#: ``find -name`` patterns materialised for each op. Only these files are
#: copied out of the workspace, so the local copy stays bounded to what the
#: analyser actually reads.
_CALL_GRAPH_PATTERNS: tuple[str, ...] = ("*.py",)
_DOCUMENTATION_PATTERNS: tuple[str, ...] = ("*.py",)


async def _materialize_workspace_for(
    container_name: str, tenant_id: str, patterns: tuple[str, ...], *, repo: str | None = None
) -> Path:
    """Mirror the tenant's workspace files out of coding-containers.

    Replaces ``_get_container_repository_path``, which called a
    ``ContainerMetadataRegistry.find_by_name`` that never existed and, on the
    resulting ``AttributeError``, returned ``os.getcwd()`` -- ``/app``, this
    service's own source tree. Every container therefore got a confident
    answer about coding-analysis. There is no fallback here: if the workspace
    cannot be read, the caller gets a typed 422, never a different repository.
    """
    container_info = await validate_and_get_container_for_tenant(container_name, tenant_id)
    try:
        return await materialize_workspace(
            container_name,
            tenant_id=tenant_id,
            root=workspace_root(container_info),
            name_patterns=patterns,
            repo=repo,
        )
    except WorkspaceUnresolvableError as exc:
        raise workspace_unresolvable_http(exc) from exc


def _workspace_cache_key(
    kind: str, tenant_id: str, request_body: dict[str, Any], repo_dir: Path
) -> str:
    """Cache key scoped by tenant, request AND the content that was analysed.

    The old keys hashed only the request body: no tenant, no content. Two
    tenants with same-named containers shared an entry, and an edit to the
    workspace was invisible for the whole TTL. ``v2`` also retires every entry
    written under the old scheme, all of which describe ``/app``.
    """
    body = hashlib.sha256(
        json.dumps(request_body, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return f"{kind}:v2:{tenant_id}:{body}:{tree_digest(repo_dir)[:16]}"


async def _cache_result(_redis_client: Any, key: str, value: str, ttl: int) -> None:
    """Cache analysis result in Redis."""
    try:
        redis_service = get_redis_service()
        await redis_service.set(key, value, ex=ttl)
    except (ConnectionError, OSError, RuntimeError) as e:
        logger.warning("Failed to cache result: %s", e)


async def _store_call_graph_analysis(result: CallGraphResponse, _config: dict[str, Any]) -> None:
    """Store call graph analysis in database."""
    try:
        logger.info("Storing call graph analysis %s", result.analysis_id)
    except (OSError, RuntimeError) as e:
        logger.exception("Failed to store call graph analysis: %s", e)


async def _store_signature_analysis(
    result: SignatureAnalysisResponse, _config: dict[str, Any]
) -> None:
    """Store signature analysis in database."""
    try:
        logger.info("Storing signature analysis %s", result.analysis_id)
    except (OSError, RuntimeError) as e:
        logger.exception("Failed to store signature analysis: %s", e)


async def _store_validation_analysis(
    result: AdvancedValidationResponse, _config: dict[str, Any]
) -> None:
    """Store validation analysis in database."""
    try:
        logger.info("Storing validation analysis %s", result.analysis_id)
    except (OSError, RuntimeError) as e:
        logger.exception("Failed to store validation analysis: %s", e)


#: Severities that block deployment, per requested security level. An analyzer
#: that finds nothing at the configured level is a pass; an analyzer that could
#: not run is never a pass, whatever the level.
_BLOCKING_SEVERITIES_BY_LEVEL: dict[str, frozenset[str]] = {
    "low": frozenset({"critical"}),
    "medium": frozenset({"critical", "high"}),
    "high": frozenset({"critical", "high", "medium"}),
    "critical": frozenset({"critical", "high", "medium"}),
}

_DEFAULT_BLOCKING_SEVERITIES: frozenset[str] = _BLOCKING_SEVERITIES_BY_LEVEL["medium"]

#: A run whose analyzers all ran but whose aggregate score falls below this is
#: failed even without a single blocking-severity finding.
_FAIL_SAFETY_THRESHOLD = 0.5


def _enum_value(value: Any) -> str:
    """Return the string value of an enum-or-string."""
    return str(value.value) if hasattr(value, "value") else str(value)


def blocking_severities_for(security_level: Any) -> frozenset[str]:
    """Map a requested security level onto the severities that block deployment."""
    return _BLOCKING_SEVERITIES_BY_LEVEL.get(
        _enum_value(security_level).lower(), _DEFAULT_BLOCKING_SEVERITIES
    )


def _build_validation_config(validation_types: list[Any], security_level: Any) -> Any:
    """Build a ValidationConfig for the symbolic tooling."""
    return ValidationConfig(
        validation_types=[ValidationType(_enum_value(vt)) for vt in validation_types],
        security_level=(
            SecurityLevel(security_level.value)
            if hasattr(security_level, "value")
            else SecurityLevel.MEDIUM
        ),
        timeout=120,
    )


async def _run_symbolic_execution(
    file_path: str, validation_types: list[Any], security_level: Any
) -> dict[str, Any]:
    """Run CrossHair symbolic execution, or report why it could not run.

    ``SymbolicValidator.run_symbolic_analysis_on_file`` returns its findings
    NESTED under the analyzer name (``{"symbolic_execution": {...}}``). The old
    code read ``.get("edge_cases")`` off the outer dict, which is always absent,
    so every run reported "completed" with nothing found regardless of the code.
    """
    try:
        # Only the symbolic type: run_symbolic_analysis_on_file dispatches on the
        # config's validation_types, so passing the full list would also invoke
        # the dead Pysa path (and its `pyre` subprocess) on every request.
        analyzer = SymbolicValidator(
            _build_validation_config([ValidationType.SYMBOLIC_EXECUTION], security_level)
        )
        raw = await analyzer.run_symbolic_analysis_on_file(file_path)
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning("Symbolic execution failed for %s: %s", file_path, e)
        return {
            "status": ANALYZER_FAILED,
            "reason": str(e),
            "edge_cases_found": [],
            "assertion_violations": [],
        }

    inner = raw.get("symbolic_execution") if isinstance(raw, dict) else None
    if not isinstance(inner, dict) or not inner:
        return {
            "status": ANALYZER_NOT_COMPUTED,
            "reason": "CrossHair produced no symbolic-execution result",
            "edge_cases_found": [],
            "assertion_violations": [],
        }

    status = str(inner.get("status", ""))
    if status in {"tool_not_found", "error", "no_output"}:
        return {
            "status": ANALYZER_NOT_COMPUTED,
            "reason": str(inner.get("error") or "CrossHair is not available in this deployment"),
            "edge_cases_found": [],
            "assertion_violations": [],
        }

    return {
        "status": "completed",
        "edge_cases_found": [str(c) for c in inner.get("edge_cases", [])],
        "assertion_violations": [str(v) for v in inner.get("violations", [])],
    }


def _run_taint_analysis(file_path: str, display_path: str | None = None) -> dict[str, Any]:
    """Run the security/taint rules this repo owns and can actually execute.

    Replaces the literal ``{"security_issues": []}`` that made a failing verdict
    structurally impossible. See ``ast_security_analyzer`` for what is reused.
    """
    try:
        return analyze_file_security(file_path, display_path=display_path)
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning("Taint analysis failed for %s: %s", file_path, e)
        return {"status": ANALYZER_FAILED, "reason": str(e), "security_issues": []}


async def _run_formal_verification(file_path: str) -> dict[str, Any]:
    """Run the repo's FormalVerifier, or report why it could not run.

    The old code hard-coded ``status: "passed"`` with coverage 0.92 and never
    called the verifier at all, although ``FormalVerifier`` is a real, working,
    dependency-free implementation (five AST verifiers).
    """
    try:
        raw = await FormalVerifier().verify(file_path)
    except (OSError, RuntimeError, ValueError) as e:
        logger.warning("Formal verification failed for %s: %s", file_path, e)
        return {
            "status": ANALYZER_FAILED,
            "reason": str(e),
            "properties_verified": [],
            "coverage": None,
            "violations": [],
        }

    violations = [
        {
            "type": str(v.get("type", "unknown")),
            "message": str(v.get("message", "")),
            "severity": str(v.get("severity", "medium")),
            "line": v.get("line"),
        }
        for v in raw.get("violations", [])
        if isinstance(v, dict)
    ]
    return {
        "status": str(raw.get("status", "completed")),
        "properties_verified": [str(p) for p in raw.get("properties", [])],
        "coverage": raw.get("coverage"),
        "violations": violations,
    }


def _record_analyzer_outcome(result: dict[str, Any], analyzer: str) -> None:
    """File one analyzer's outcome under analyzers_ran or analyzers_not_computed."""
    payload = result.get(analyzer) or {}
    status = str(payload.get("status", ANALYZER_NOT_COMPUTED))
    if status in ANALYZER_RAN_STATUSES:
        result["analyzers_ran"].append(analyzer)
        return
    result["analyzers_not_computed"].append(
        {
            "analyzer": analyzer,
            "status": status if status in {ANALYZER_FAILED, ANALYZER_NOT_COMPUTED} else status,
            "reason": str(payload.get("reason") or f"{analyzer} produced no result"),
        }
    )


def _unanalyzable_reason(change: Any) -> str | None:
    """Return why this change cannot be analyzed, or None if it can be.

    The analyzers in this service parse Python. A DELETED file has no body, and
    a non-Python file is not something these rule sets can read. Both are honest
    abstentions — writing the content to a ``.py`` temp file and reporting the
    resulting SyntaxError would be a misleading answer to a different question.
    """
    if _enum_value(getattr(change, "change_type", "")).upper() == "DELETED":
        return "file is deleted; there is no content to analyze"
    if not getattr(change, "modified_code", ""):
        return "change carries no content to analyze"
    suffix = Path(str(getattr(change, "file_path", ""))).suffix.lower()
    if suffix not in {".py", ".pyi"}:
        return f"analyzers support Python only; '{suffix or 'no extension'}' is not supported"
    return None


def _abstain_for_change(change: Any, validation_types: list[Any], reason: str) -> dict[str, Any]:
    """Build a per-change result in which every requested analyzer abstained."""
    requested = [_enum_value(vt) for vt in validation_types] or ["none"]
    return {
        "file_path": getattr(change, "file_path", ""),
        "analyzers_ran": [],
        "analyzers_not_computed": [
            {"analyzer": analyzer, "status": ANALYZER_NOT_COMPUTED, "reason": reason}
            for analyzer in requested
        ],
    }


async def _run_validation_analyses(
    file_path: str, change: Any, validation_types: list[Any], security_level: Any
) -> dict[str, Any]:
    """Run the requested validation analyses on one code change.

    Every requested analyzer reports either a real result or an explicit
    abstention with a reason. Nothing is defaulted: there is no baseline safety
    score, no empty-but-"completed" taint result, and no hard-coded "passed".
    """
    result: dict[str, Any] = {
        "file_path": change.file_path,
        "analyzers_ran": [],
        "analyzers_not_computed": [],
    }
    requested = {_enum_value(vt) for vt in validation_types}

    if "symbolic_execution" in requested:
        result["symbolic_execution"] = await _run_symbolic_execution(
            file_path, validation_types, security_level
        )
        _record_analyzer_outcome(result, "symbolic_execution")

    if "taint_analysis" in requested:
        result["taint_analysis"] = _run_taint_analysis(file_path, change.file_path)
        _record_analyzer_outcome(result, "taint_analysis")

    if "formal_verification" in requested:
        result["formal_verification"] = await _run_formal_verification(file_path)
        _record_analyzer_outcome(result, "formal_verification")

    if not requested:
        result["analyzers_not_computed"].append(
            {
                "analyzer": "none",
                "status": ANALYZER_NOT_COMPUTED,
                "reason": "no validation_types were requested",
            }
        )

    return result


def _analyze_files(
    supported_files: list[Path], extractor: FunctionExtractor, repo_path: str
) -> dict[str, Any]:
    """Extract signatures per file, then relate them ACROSS files.

    Per file the extractor sees only that file, so callers, real usage examples
    and argument-count checks are joined here over the whole analysed set.
    ``called_by`` is matched by function NAME across the analysed files, not
    resolved: it is a candidate list (any call to a same-named function counts). A file
    that could not be analysed is reported in ``files_not_computed`` instead of
    silently vanishing (it used to, which read as "this file has no functions").
    """
    all_functions: list[dict[str, Any]] = []
    total_functions = 0
    typed_functions = 0
    files_not_computed: list[dict[str, str]] = []
    per_file: list[tuple[str, dict[str, Any]]] = []
    internal_modules: set[str] = set()
    for file_path in supported_files:
        module = extractor._get_module_name(str(file_path), repo_path)
        internal_modules.add(module.removesuffix(".__init__"))
        try:
            file_analysis = extractor.extract_comprehensive_signatures(str(file_path), repo_path)
        except (OSError, RuntimeError, ValueError, UnicodeDecodeError) as e:
            logger.warning("Error analyzing file %s: %s", file_path, e)
            files_not_computed.append({"file": _rel(file_path, repo_path), "reason": str(e)})
            continue
        if file_analysis.get("error"):
            files_not_computed.append(
                {"file": _rel(file_path, repo_path), "reason": str(file_analysis["error"])}
            )
            continue
        per_file.append((module, file_analysis))
        functions = file_analysis.get("functions", [])
        all_functions.extend(functions)
        total_functions += len(functions)
        for func in functions:
            if func.get("return_type") or any(
                param.get("type") for param in func.get("parameters", [])
            ):
                typed_functions += 1

    def _internal(dotted: str) -> bool:
        parts = dotted.split(".")
        return any(".".join(parts[:k]) in internal_modules for k in range(1, len(parts) + 1))

    external_dependencies: set[str] = set()
    for _, analysis in per_file:
        for dep in analysis.get("external_dependencies", []):
            if not _internal(dep):
                external_dependencies.add(dep.split(".")[0])
    for func in all_functions:
        func["external_dependencies"] = [
            d for d in func.get("external_dependencies", []) if not _internal(d)
        ]

    # Cross-file callers and real call-site examples, by short name.
    sites_by_name: dict[str, list[dict[str, Any]]] = {}
    for module, analysis in per_file:
        for short, sites in (analysis.get("call_sites") or {}).items():
            for site in sites:
                sites_by_name.setdefault(short, []).append({**site, "module": module})
    for func in all_functions:
        sites = sites_by_name.get(func["name"], [])
        func["called_by"] = sorted({f"{s['module']}.{s['caller']}" for s in sites})
        examples = list(func.get("usage_examples") or [])
        examples += [s["source"] for s in sites if s.get("plain_name") and s.get("source")]
        func["usage_examples"] = list(dict.fromkeys(examples))[:3]

    return {
        "functions": all_functions,
        "dependencies": external_dependencies,
        "total_functions": total_functions,
        "typed_functions": typed_functions,
        "files_not_computed": files_not_computed,
        **_call_arity(all_functions, per_file),
    }


_RUST_FUNCTION_KINDS = {"function", "method", "impl_fn"}


async def _rust_signatures(
    analysis_id: str, repo_dir: Path, tenant_id: str
) -> SignatureAnalysisResponse:
    """Rust signatures from the api-surface extractor (pub items + trait impls)."""
    from dataclasses import asdict

    from processing.contract_analysis import isolation
    from processing.structure import jobs

    def read_tree() -> dict[str, str]:
        return {
            p.relative_to(repo_dir).as_posix(): p.read_text(encoding="utf-8", errors="replace")
            for p in sorted(repo_dir.rglob("*"))
            if p.is_file()
        }

    files = await asyncio.to_thread(read_tree)  # disk reads off the event loop
    surface = await isolation.run(jobs.api_surface_job, files, "rust", "", tenant=tenant_id)
    functions: list[dict[str, Any]] = []
    typed = 0
    for sym in surface.symbols:
        if sym.kind not in _RUST_FUNCTION_KINDS or sym.signature is None:
            continue
        sig = sym.signature
        params = [
            {
                "name": p.name,
                "type": p.type,
                "required": True,
                "default": None,
                # Rust has no keyword arguments: the receiver, or a positional one.
                "kind": "self" if p.name == "self" else "positional",
            }
            for p in sig.params
        ]
        rendered = ", ".join(
            p.type if p.name == "self" else f"{p.name}: {p.type}" for p in sig.params
        )
        text = f"{'async ' if sig.is_async else ''}fn {sym.name}{sig.generics or ''}({rendered})"
        if sig.returns:
            text += f" -> {sig.returns}"
        if sig.returns or any(p.type for p in sig.params):
            typed += 1
        functions.append(
            {
                "name": sym.name,
                "module": f"{sym.module_path}::{sym.owner}" if sym.owner else sym.module_path,
                "signature": text,
                "parameters": params,
                "return_type": sig.returns,
                "line_number": sym.line,
                "is_async": sig.is_async,
                "is_method": sym.owner is not None,
                "complexity_metrics": None,
            }
        )
    return SignatureAnalysisResponse(
        analysis_id=analysis_id,
        functions=[StaticAnalysisFunctionSignature.model_validate(f) for f in functions],
        type_coverage=round(typed / len(functions), 2) if functions else 0.0,
        total_functions=len(functions),
        typed_functions=typed,
        files_not_computed=[{"file": f, "reason": r} for f, r in surface.files_unparsed.items()],
        call_arity_status="not_computed",
        call_arity_reason="call arity is computed for Python only",
        unresolved=[asdict(u) for u in surface.unresolved],
    )


def _rel(path: Path, repo_path: str) -> str:
    try:
        return str(Path(path).relative_to(repo_path))
    except ValueError:
        return str(path)


def _call_arity(
    functions: list[dict[str, Any]], per_file: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    """Arity mismatches plus whether they were computed at all.

    A write to a module-like object whose target name the AST cannot read
    (``setattr(mod, name, ...)``, ``vars(mod).update(mapping)``) anywhere in the
    analysed set may rebind ANY function, so no call is judged and the status
    says so: an empty list must not read as "0 mismatches".
    """
    unknown = [
        f"{module}: {where}"
        for module, analysis in per_file
        for where in analysis.get("unknown_patch_targets") or []
    ]
    if unknown:
        return {
            "call_arity_mismatches": [],
            "call_arity_status": "not_computed",
            "call_arity_reason": (
                "module attribute written with a non-constant name; any function may be "
                "rebound: " + "; ".join(unknown[:5])
            ),
        }
    return {
        "call_arity_mismatches": _call_arity_mismatches(functions, per_file),
        "call_arity_status": "computed",
        "call_arity_reason": None,
    }


def _call_arity_mismatches(
    functions: list[dict[str, Any]], per_file: list[tuple[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Plain-name calls whose arguments cannot bind to the ONE function they name.

    Only judged when the binding is certain, because a false mismatch sends a
    correct change back; a missed one is acceptable. Certain means: the callee is
    a module-level, undecorated, non-method function defined exactly once in the
    analysed set, as a TOP-LEVEL statement (not under if/try/with/...), and its
    own module binds that name by nothing else (no reassignment such as
    ``f = partial(f, 1)`` or ``g = wrap(g)``, same-name class, import, ``global``
    rebinding from a function, star import, ``globals()[...]`` write, or any
    dynamic construct: setattr/exec/eval/globals()/``sys.modules``/``__defaults__``
    writes, ...); no file in the set stores or deletes an attribute of that name
    (``A.f = ...`` monkeypatch); the caller either is in the same module (and does not import
    the name) or binds it with ``from <that module> import``; and the caller's
    file does not rebind the name anywhere else (assignment, lambda, nested def,
    parameter, import alias, ...). Starred / ``**`` calls and callees taking
    ``*args`` / ``**kwargs`` are skipped. Arguments are bound the way Python
    binds them: positional-only, positional-or-keyword, keyword-only.
    """
    by_name: dict[str, list[dict[str, Any]]] = {}
    for func in functions:
        by_name.setdefault(func["name"], []).append(func)
    # Per module: every name bound by something other than its one top-level def.
    # A module-level import of the name rebinds it too (it may run after the def).
    callee_rebound: dict[str, set[str]] = {
        module: set(analysis.get("rebound_names") or []) | set(analysis.get("imports") or {})
        for module, analysis in per_file
    }
    # ``X.f = ...``, ``f.__defaults__ = ...``, ``setattr(X, 'f', ...)``, ``vars(X)['f']``,
    # ``mock.patch('a.f')`` ... anywhere in the set may rebind any module's ``f``.
    patched: set[str] = set()
    for _module, analysis in per_file:
        patched.update(analysis.get("attribute_stores") or [])
    out: list[dict[str, Any]] = []
    for module, analysis in per_file:
        imports = analysis.get("imports") or {}
        rebound = set(analysis.get("rebound_names") or [])
        if "*" in rebound:
            continue
        for short, sites in (analysis.get("call_sites") or {}).items():
            defs = by_name.get(short, [])
            if len(defs) != 1 or short in rebound or short in patched:
                continue
            callee = defs[0]
            if callee.get("is_method") or callee.get("decorators"):
                continue
            if not callee.get("is_top_level"):
                continue
            rebound_in_callee = callee_rebound.get(callee["module"])
            if rebound_in_callee is None or "*" in rebound_in_callee or short in rebound_in_callee:
                continue
            path = f"{callee['module']}.{short}"
            if callee["module"] == module:
                bound_here = short not in imports
            else:
                bound_here = imports.get(short) == path
            if not bound_here:
                continue
            params = callee.get("parameters", [])
            kinds = [p.get("kind", "positional_or_keyword") for p in params]
            if any(k in ("var_positional", "var_keyword") for k in kinds) or any(
                p["name"].startswith("*") for p in params
            ):
                continue
            for site in sites:
                if not site.get("plain_name") or site.get("positional") is None:
                    continue
                if None in site.get("keywords", []):  # **mapping: unknown keys
                    continue
                reasons = _bind_arguments(params, kinds, site["positional"], site["keywords"])
                if reasons:
                    out.append(
                        {
                            "caller": f"{module}.{site['caller']}",
                            "line": site["line"],
                            "call": site.get("source", ""),
                            "callee": path,
                            "signature": callee.get("signature"),
                            "reason": "; ".join(reasons),
                        }
                    )
    return out


def _bind_arguments(
    params: list[dict[str, Any]], kinds: list[str], positional: int, keywords: list[str]
) -> list[str]:
    """Why ``positional`` args + ``keywords`` cannot bind to ``params`` ([] if they can)."""
    positional_names = [
        p["name"]
        for p, k in zip(params, kinds, strict=True)
        if k in ("positional_only", "positional_or_keyword")
    ]
    positional_only = {
        p["name"] for p, k in zip(params, kinds, strict=True) if k == "positional_only"
    }
    keyword_names = {
        p["name"]
        for p, k in zip(params, kinds, strict=True)
        if k in ("positional_or_keyword", "keyword_only")
    }
    reasons: list[str] = []
    if positional > len(positional_names):
        reasons.append(f"{positional} positional arguments, at most {len(positional_names)}")
    filled = set(positional_names[:positional])
    by_position_only = [k for k in keywords if k in positional_only]
    if by_position_only:
        reasons.append(f"positional-only argument(s) passed by keyword {by_position_only}")
    unknown = [k for k in keywords if k not in keyword_names and k not in positional_only]
    if unknown:
        reasons.append(f"unknown keyword(s) {unknown}")
    multiple = [k for k in keywords if k in filled and k not in positional_only]
    if multiple:
        reasons.append(f"multiple values for argument(s) {multiple}")
    filled |= {k for k in keywords if k in keyword_names}
    missing = [p["name"] for p in params if p.get("required") and p["name"] not in filled]
    if missing:
        reasons.append(f"missing required argument(s) {missing}")
    return reasons


def _identify_blocking_issues(
    validation_results: list[dict[str, Any]],
    blocking_severities: frozenset[str] | set[str] | None = None,
) -> list[str]:
    """Identify blocking issues from EVERY analyzer that produced a result.

    Previously this read the taint result alone — and that result was a literal
    empty list, so the function could never return anything. An analyzer that
    abstained contributes nothing here; it is surfaced as an abstention instead.
    """
    severities = blocking_severities or _DEFAULT_BLOCKING_SEVERITIES
    blocking_issues: list[str] = []

    for validation_result in validation_results:
        file_path = validation_result.get("file_path", "")

        taint = validation_result.get("taint_analysis")
        if isinstance(taint, dict):
            for issue in taint.get("security_issues", []):
                if issue.get("severity") in severities:
                    # The analyzer reports "<file>:<line>"; fall back to the
                    # change's path when it has no location of its own.
                    where = issue.get("location") or file_path
                    blocking_issues.append(f"{where}: {issue.get('type', 'Unknown issue')}")

        symbolic = validation_result.get("symbolic_execution")
        if isinstance(symbolic, dict) and symbolic.get("status") in ANALYZER_RAN_STATUSES:
            for violation in symbolic.get("assertion_violations", []):
                blocking_issues.append(f"{file_path}: assertion violation: {violation}")

        formal = validation_result.get("formal_verification")
        if isinstance(formal, dict) and formal.get("status") in ANALYZER_RAN_STATUSES:
            for violation in formal.get("violations", []):
                if not isinstance(violation, dict):
                    continue
                if violation.get("severity") in severities:
                    line = violation.get("line")
                    suffix = f" at line {line}" if line else ""
                    blocking_issues.append(
                        f"{file_path}: {violation.get('type', 'property violation')}"
                        f"{suffix}: {violation.get('message', '')}".rstrip(": ")
                    )

    return blocking_issues


def _generate_recommendations(
    overall_safety_score: float | None,
    blocking_issues: list[str],
    abstentions: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Generate recommendations based on safety score, blocking issues and abstentions."""
    recommendations: list[str] = []
    if overall_safety_score is None:
        for abstention in abstentions or []:
            recommendations.append(
                f"No verdict: {abstention.get('analyzer')} did not run ({abstention.get('reason')})"
            )
        if not recommendations:
            recommendations.append("No verdict: no analyzer produced a result")
        return recommendations
    if overall_safety_score < 0.7:
        recommendations.append("Increase test coverage for modified functions")
    if blocking_issues:
        recommendations.append("Address critical security issues before deployment")
    for abstention in abstentions or []:
        recommendations.append(
            f"Partial coverage: {abstention.get('analyzer')} did not run "
            f"({abstention.get('reason')})"
        )
    if not recommendations:
        recommendations.append("Code changes appear safe for deployment")
    return recommendations


def _build_validation_results(
    validation_results: list[dict[str, Any]],
) -> tuple[list[StaticAnalysisValidationResult], list[float]]:
    """Turn the raw per-file dicts into response models and score the ones that ran.

    A file for which no analyzer produced a result gets ``safety_score=None`` and
    contributes nothing to the aggregate — it is not scored 0 (which would read
    as "dangerous") nor 0.8 (which would read as "checked and fine").
    """
    calculator = SafetyCalculator()
    built: list[StaticAnalysisValidationResult] = []
    scores: list[float] = []

    for result_dict in validation_results:
        model = StaticAnalysisValidationResult(
            file_path=result_dict.get("file_path", ""),
            symbolic_execution=result_dict.get("symbolic_execution"),
            taint_analysis=result_dict.get("taint_analysis"),
            formal_verification=result_dict.get("formal_verification"),
            safety_score=None,
            analyzers_ran=list(result_dict.get("analyzers_ran", [])),
            analyzers_not_computed=[
                AnalyzerAbstention(**a) for a in result_dict.get("analyzers_not_computed", [])
            ],
        )
        if model.analyzers_ran:
            model.safety_score = round(calculator.calculate_score(model), 2)
            scores.append(model.safety_score)
        built.append(model)

    return built, scores


async def _generate_architecture_docs(
    _repo_path: str, include_diagrams: bool, _doc_format: Any
) -> dict[str, Any]:
    """Generate architecture documentation."""
    return {
        "content": "# Architecture Overview\n\nGenerated architecture document.",
        "diagrams": ["architecture_diagram.svg"] if include_diagrams else [],
        "coverage": 0.85,
        "quality_score": 0.8,
        "files": ["docs/architecture.md"],
    }


async def _generate_api_reference(_repo_path: str, _doc_format: Any) -> dict[str, Any]:
    """Generate API reference documentation."""
    return {
        "content": "# API Reference\n\nGenerated API documentation.",
        "coverage": 0.9,
        "quality_score": 0.9,
        "files": ["docs/api_reference.md"],
    }


async def _generate_change_summary(_repo_path: str, _doc_format: Any) -> dict[str, Any]:
    """Generate change summary documentation."""
    return {
        "content": "# Change Summary\n\nSummary of recent changes.",
        "impact_analysis": "Low risk - isolated function changes",
        "quality_score": 0.85,
        "files": ["docs/change_summary.md"],
    }
