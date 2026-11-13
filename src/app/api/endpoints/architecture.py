"""Architecture module."""

from __future__ import annotations

"""Architecture analysis API endpoints.

This module provides FastAPI endpoints for analyzing software architecture,
including components, layers, metrics, patterns, and visualization.

Note: The functions decorated with @router.get/@router.post are FastAPI endpoints
that are automatically registered with the application router. They may appear
as unused to static analysis tools like vulture, but they are actually used
by the FastAPI framework.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.api.responses import (
    bad_request,
    internal_error,
    not_found,
)
from app.core.container_manager import get_container_manager
from auth.tenant import require_tenant
from models.architecture_models import (
    ArchitecturalMetric,
    ArchitectureAnalysisResult,
    ArchitectureGraphMetrics,
    ArchitectureGraphResponse,
    ArchitectureJobRequest,
    Component,
    DetectedPattern,
    GraphEdge,
    GraphNode,
    Layer,
    RepoConfig,
)
from processing.architecture.architecture_processor import ArchitectureProcessor
from processing.architecture.graph_adapter import GraphAdapter
from services.code_graph.multi_language_graph_service import (
    MultiLanguageGraphServiceFactory,
)

from ..models.error_models import (
    Error400Response,
    Error401Response,
    Error403Response,
    Error404Response,
    Error405Response,
    Error422Response,
    Error500Response,
)

logger = logging.getLogger(__name__)
router = APIRouter()


async def _resolve_container_to_codebase(container_name: str, tenant_id: str) -> tuple[str, Path]:
    """Resolve container name to codebase_id and repository path with tenant validation.

    Also performs a synchronous self-heal: architecture analysis is a *consumer*
    of the code graph (it loads, it does not build), so if the graph has not been
    populated yet — only the ``repository`` root node exists — the verdict would
    be vacuously empty. We build the graph on the spot (idempotent) so the
    analysis never runs against an empty graph, regardless of whether the
    provision-time eager build finished, was slow, or failed.
    """
    try:
        container_manager = get_container_manager()
        try:
            container = await container_manager.get_container(container_name, tenant_id=tenant_id)
        except ValueError:
            raise not_found("Container", container_name)
        if container is None:
            raise not_found("Container", container_name)
        codebase_id = container.container_name

        # Use the container's real codebase path (repos are cloned to /workspace
        # by the lifecycle manager), not a hardcoded /app/data/repo. This path is
        # used by the processor for config + optional dependency/quality analysis;
        # the graph itself is loaded by codebase_id.
        codebase_path = Path(getattr(container, "codebase_path", None) or "/workspace")

        # Self-heal: ensure the code graph is populated before the processor loads
        # it. Idempotent and best-effort — a build failure must not turn a read
        # endpoint into a 500 (an empty graph degrades to empty components).
        await _ensure_graph_populated(codebase_id, tenant_id)

        return (codebase_id, codebase_path)
    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        logger.error(f"Error resolving container {container_name}: {e}")
        raise internal_error(f"Failed to resolve container: {str(e)}") from e


async def _ensure_graph_populated(
    codebase_id: str, tenant_id: str, *, force: bool = False
) -> dict | None:
    """Build the code graph if it is empty (best-effort, never raises).

    ``force=True`` re-extracts even when a graph already exists. That matters for
    exactly one caller — the cycle-check BASELINE mode. ``build_code_graph`` opens
    with ``if not force_rebuild and await graph_is_populated(...): return {"skipped":
    True}``, and the graph is keyed on ``container_name``, which the AA drivers reuse
    across drives. So without ``force`` a "baseline" silently describes whatever
    earlier state built the graph — typically the PREVIOUS drive's POST-refactor
    tree — and the gate then compares that to the current post-refactor tree and can
    never fail. Measured on drive 6 (2026-09-01): a container holding a real
    module-level import ring reported ``cycles: []``.

    Returns ``build_code_graph``'s result dict, or ``None`` if the build raised.
    """
    try:
        from services.code_graph.graph_builder import build_code_graph

        # Keep the non-forced call shape byte-identical: the default path is the one
        # every other read endpoint takes, and its test is the control proving this
        # change did not touch it.
        if force:
            result = await build_code_graph(codebase_id, tenant_id, force_rebuild=True)
        else:
            result = await build_code_graph(codebase_id, tenant_id)
        if not result.get("skipped"):
            logger.info(
                "Self-heal built code graph for %s: %s nodes, %s edges",
                codebase_id,
                result.get("nodes"),
                result.get("edges"),
            )
        return result
    except Exception as exc:  # pragma: no cover - defensive; must not 500 a read
        logger.warning("Self-heal graph build failed for %s: %s", codebase_id, exc)
        return None


async def _try_get_architecture_processor() -> ArchitectureProcessor | None:
    """Try to build an ArchitectureProcessor.

    Returns None when the graph-service backend is unavailable (e.g. missing
    SQLite DB path in config).  Callers must check the return value and
    validate the container **first** so that a missing container still yields
    404 instead of 500.
    """
    try:
        graph_service = await MultiLanguageGraphServiceFactory.get_service()
        graph_adapter = GraphAdapter(graph_service)
        return ArchitectureProcessor(graph_adapter)
    except Exception as exc:
        logger.warning("Architecture processor unavailable: %s", exc)
        return None


@router.get(
    "/components/{container_name}",
    response_model=list[Component],
    operation_id="list_architecture_components",
    responses={
        400: {
            "model": Error400Response,
            "description": "Bad request - invalid parameters",
        },
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {
            "model": Error404Response,
            "description": "Container or resource not found",
        },
        405: {"model": Error405Response, "description": "Method not allowed"},
        422: {
            "model": Error422Response,
            "description": "Validation error in request parameters",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error during architecture analysis",
        },
    },
)
async def list_components(
    container_name: str,
    tenant_id: str = Depends(require_tenant),
):
    """Get architectural components for a container."""
    try:
        codebase_id, repo_path = await _resolve_container_to_codebase(container_name, tenant_id)
        processor = await _try_get_architecture_processor()
        if processor is None:
            return []
        job = ArchitectureJobRequest(
            codebase_id=codebase_id, repo_path=repo_path, tenant_id=tenant_id
        )
        # Graph-only read: skip the dependency manifest read.
        result = await processor.process(job, include_dependency_analysis=False)
        return result.components
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e)) from e
        logger.error("Failed to get components for container %s: %s", container_name, e)
        raise internal_error(f"Failed to get components: {str(e)}") from e


@router.get(
    "/dependencies/{container_name}",
    # Each entry is {"component": <id:str>, "dependencies": [<id:str>, ...]} — the
    # value is a str for "component" and a list for "dependencies", so the model
    # must allow both (the old list[dict[str, list[str]]] rejected the str id and
    # raised ResponseValidationError once components were non-empty).
    response_model=list[dict[str, str | list[str]]],
    operation_id="get_architecture_dependencies",
    responses={
        400: {
            "model": Error400Response,
            "description": "Bad request - invalid parameters",
        },
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {
            "model": Error404Response,
            "description": "Container or resource not found",
        },
        405: {"model": Error405Response, "description": "Method not allowed"},
        422: {
            "model": Error422Response,
            "description": "Validation error in request parameters",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error during architecture analysis",
        },
    },
)
async def get_dependencies(
    container_name: str,
    tenant_id: str = Depends(require_tenant),
):
    """Get component dependency mappings for a container."""
    try:
        codebase_id, repo_path = await _resolve_container_to_codebase(container_name, tenant_id)
        processor = await _try_get_architecture_processor()
        if processor is None:
            return []
        job = ArchitectureJobRequest(
            codebase_id=codebase_id, repo_path=repo_path, tenant_id=tenant_id
        )
        # Graph-only read: skip the dependency manifest read.
        result = await processor.process(job, include_dependency_analysis=False)
        return [
            {"component": comp.id, "dependencies": comp.dependencies} for comp in result.components
        ]
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e)) from e
        logger.error("Failed to get dependencies for container %s: %s", container_name, e)
        raise internal_error(f"Failed to get dependencies: {str(e)}") from e


@router.get(
    "/layers/{container_name}",
    response_model=dict[str, list[Layer]],
    operation_id="get_architecture_layers",
    responses={
        400: {
            "model": Error400Response,
            "description": "Bad request - invalid parameters",
        },
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {
            "model": Error404Response,
            "description": "Container or resource not found",
        },
        405: {"model": Error405Response, "description": "Method not allowed"},
        422: {
            "model": Error422Response,
            "description": "Validation error in request parameters",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error during architecture analysis",
        },
    },
)
async def get_layers(
    container_name: str,
    tenant_id: str = Depends(require_tenant),
):
    """Get layer assignments for a container."""
    try:
        codebase_id, repo_path = await _resolve_container_to_codebase(container_name, tenant_id)
        processor = await _try_get_architecture_processor()
        if processor is None:
            return {"layers": []}
        job = ArchitectureJobRequest(
            codebase_id=codebase_id, repo_path=repo_path, tenant_id=tenant_id
        )
        # Graph-only read: skip the dependency manifest read.
        result = await processor.process(job, include_dependency_analysis=False)
        return {"layers": result.layers}
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e)) from e
        logger.error("Failed to get layers for container %s: %s", container_name, e)
        raise internal_error(f"Failed to get layers: {str(e)}") from e


@router.get(
    "/metrics/{container_name}",
    response_model=dict[str, list[ArchitecturalMetric]],
    operation_id="get_architecture_metrics",
    responses={
        400: {
            "model": Error400Response,
            "description": "Bad request - invalid parameters",
        },
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {
            "model": Error404Response,
            "description": "Container or resource not found",
        },
        405: {"model": Error405Response, "description": "Method not allowed"},
        422: {
            "model": Error422Response,
            "description": "Validation error in request parameters",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error during architecture analysis",
        },
    },
)
async def get_metrics(
    container_name: str,
    tenant_id: str = Depends(require_tenant),
):
    """Get architectural metrics for a container."""
    try:
        codebase_id, repo_path = await _resolve_container_to_codebase(container_name, tenant_id)
        processor = await _try_get_architecture_processor()
        if processor is None:
            return {"metrics": []}
        job = ArchitectureJobRequest(
            codebase_id=codebase_id, repo_path=repo_path, tenant_id=tenant_id
        )
        # Graph-only read: skip the dependency manifest read.
        result = await processor.process(job, include_dependency_analysis=False)
        return {"metrics": result.metrics}
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e)) from e
        logger.error("Failed to get metrics for container %s: %s", container_name, e)
        raise internal_error(f"Failed to get metrics: {str(e)}") from e


@router.get(
    "/patterns/{container_name}",
    response_model=list[DetectedPattern],
    operation_id="get_architecture_patterns",
    responses={
        400: {
            "model": Error400Response,
            "description": "Bad request - invalid parameters",
        },
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {
            "model": Error404Response,
            "description": "Container or resource not found",
        },
        405: {"model": Error405Response, "description": "Method not allowed"},
        422: {
            "model": Error422Response,
            "description": "Validation error in request parameters",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error during architecture analysis",
        },
    },
)
async def get_patterns(
    container_name: str,
    tenant_id: str = Depends(require_tenant),
):
    """Get detected patterns for a container."""
    try:
        codebase_id, repo_path = await _resolve_container_to_codebase(container_name, tenant_id)
        processor = await _try_get_architecture_processor()
        if processor is None:
            return []
        job = ArchitectureJobRequest(
            codebase_id=codebase_id, repo_path=repo_path, tenant_id=tenant_id
        )
        # Graph-only read: skip the dependency manifest read.
        result = await processor.process(job, include_dependency_analysis=False)
        return result.patterns
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e)) from e
        logger.error("Failed to get patterns for container %s: %s", container_name, e)
        raise internal_error(f"Failed to get patterns: {str(e)}") from e


@router.get(
    "/analysis/{container_name}",
    response_model=ArchitectureAnalysisResult,
    operation_id="get_architecture_full_analysis",
    responses={
        400: {
            "model": Error400Response,
            "description": "Bad request - invalid parameters",
        },
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {
            "model": Error404Response,
            "description": "Container or resource not found",
        },
        405: {"model": Error405Response, "description": "Method not allowed"},
        422: {
            "model": Error422Response,
            "description": "Validation error in request parameters",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error during architecture analysis",
        },
    },
)
async def get_full_analysis(
    container_name: str,
    tenant_id: str = Depends(require_tenant),
):
    """Get full architectural analysis for a container."""
    try:
        codebase_id, repo_path = await _resolve_container_to_codebase(container_name, tenant_id)
        processor = await _try_get_architecture_processor()
        if processor is None:
            from datetime import UTC, datetime

            return ArchitectureAnalysisResult(
                codebase_id=codebase_id,
                generated_at=datetime.now(UTC).isoformat(),
                components=[],
                layers=[],
                metrics=[],
                patterns=[],
                summary={"status": 0},
                execution_time=0.0,
                config_used=RepoConfig(),
            )
        job = ArchitectureJobRequest(
            codebase_id=codebase_id, repo_path=repo_path, tenant_id=tenant_id
        )
        result = await processor.process(job)
        return result
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e)) from e
        logger.error("Failed to get full analysis for container %s: %s", container_name, e)
        raise internal_error(f"Failed to get full analysis: {str(e)}") from e


@router.post(
    "/analyze/{container_name}",
    response_model=ArchitectureAnalysisResult,
    operation_id="trigger_architecture_analysis",
    responses={
        400: {
            "model": Error400Response,
            "description": "Bad request - invalid parameters",
        },
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {
            "model": Error404Response,
            "description": "Container or resource not found",
        },
        405: {"model": Error405Response, "description": "Method not allowed"},
        422: {
            "model": Error422Response,
            "description": "Validation error in request parameters",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error during architecture analysis",
        },
    },
)
async def trigger_analysis(
    container_name: str,
    force_rebuild: bool = Query(False, description="Force rebuild of analysis"),
    tenant_id: str = Depends(require_tenant),
):
    """Trigger architectural analysis for a container."""
    try:
        if not container_name or not container_name.strip():
            raise bad_request("Container name cannot be empty")
        codebase_id, repo_path = await _resolve_container_to_codebase(container_name, tenant_id)
        processor = await _try_get_architecture_processor()
        if processor is None:
            from datetime import UTC, datetime

            return ArchitectureAnalysisResult(
                codebase_id=codebase_id,
                generated_at=datetime.now(UTC).isoformat(),
                components=[],
                layers=[],
                metrics=[],
                patterns=[],
                summary={"status": 0},
                execution_time=0.0,
                config_used=RepoConfig(),
            )
        # When force_rebuild is requested, RE-EXTRACT the graph before the processor
        # loads it. processor.process() only load_graph()s the existing structure and
        # never honors force_rebuild itself; the self-heal (_ensure_graph_populated) only
        # builds when EMPTY. So without this an explicit force_rebuild=true (e.g. the AA
        # DOING `rebuild_code_graph` keystone after a refactor) was a silent no-op and the
        # analyze served the STALE pre-refactor graph. build_code_graph(force_rebuild=True)
        # clears stale nodes/edges first, so the post-refactor structure is what gets read.
        if force_rebuild:
            from services.code_graph.graph_builder import build_code_graph

            rebuilt = await build_code_graph(codebase_id, tenant_id, force_rebuild=True)
            logger.info(
                "force_rebuild re-extracted graph for %s: %s nodes, %s edges",
                container_name,
                rebuilt.get("nodes"),
                rebuilt.get("edges"),
            )
        job = ArchitectureJobRequest(
            codebase_id=codebase_id,
            repo_path=repo_path,
            tenant_id=tenant_id,
            force_rebuild=force_rebuild,
        )
        result = await processor.process(job)
        logger.info("Architecture analysis completed for container %s", container_name)
        return result
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e)) from e
        logger.error("Failed to trigger analysis for container %s: %s", container_name, e)
        raise internal_error(f"Failed to trigger analysis: {str(e)}") from e


@router.get(
    "/graph/{container_name}",
    response_model=ArchitectureGraphResponse,
    operation_id="analysis_get_architecture_graph",
    summary="Get architecture visualization graph",
    description="""
    Get architecture graph data optimized for D3.js force-directed visualization.

    Returns nodes (components, services, modules), edges (dependencies),
    layers, and summary metrics for interactive graph visualization.
    """,
    responses={
        400: {"model": Error400Response, "description": "Bad request"},
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {"model": Error404Response, "description": "Container not found"},
        500: {"model": Error500Response, "description": "Internal server error"},
    },
)
async def get_architecture_graph(
    container_name: str,
    tenant_id: str = Depends(require_tenant),
):
    """Get architecture graph for visualization."""
    try:
        # Get architecture analysis results
        codebase_id, repo_path = await _resolve_container_to_codebase(container_name, tenant_id)
        processor = await _try_get_architecture_processor()
        if processor is None:
            return ArchitectureGraphResponse(
                nodes=[],
                edges=[],
                layers=[],
                metrics=ArchitectureGraphMetrics(
                    total_components=0,
                    total_dependencies=0,
                    avg_complexity=0.0,
                    hotspot_count=0,
                ),
            )
        job = ArchitectureJobRequest(
            codebase_id=codebase_id, repo_path=repo_path, tenant_id=tenant_id
        )
        # The graph is built from components, layers and metrics only: skip the
        # dependency manifest read (its count is not part of this response).
        result = await processor.process(job, include_dependency_analysis=False)

        # Build graph nodes from components
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        for component in result.components:
            # Determine component type based on naming/structure
            comp_type = "component"
            if "service" in component.name.lower():
                comp_type = "service"
            elif "module" in component.name.lower():
                comp_type = "module"
            elif "package" in component.name.lower() or "pkg" in component.name.lower():
                comp_type = "package"

            # Get metrics for this component
            comp_metrics = {}
            hotspot_score = 0.0
            for metric in result.metrics:
                if metric.component_id == component.id:
                    comp_metrics = {
                        "complexity": metric.value,
                        "instability": metric.instability or 0.0,
                        "abstractness": metric.abstractness or 0.0,
                        "distance": metric.distance or 0.0,
                    }
                    # Calculate hotspot score based on complexity and distance
                    hotspot_score = min(
                        1.0,
                        (metric.value / 10.0) * 0.5 + (metric.distance or 0.0) * 0.5,
                    )

            # Estimate issue count based on component metrics
            # High complexity and distance from main sequence correlate with quality issues
            # This is a heuristic until proper quality analysis integration is available
            estimated_issues: int | None = None
            if comp_metrics:
                complexity = comp_metrics.get("complexity", 0)
                distance = comp_metrics.get("distance", 0)
                # Complexity > 10 or distance > 0.5 suggests potential issues
                if complexity > 10 or distance > 0.5:
                    estimated_issues = int(complexity / 5) + int(distance * 10)
                else:
                    estimated_issues = 0

            node = GraphNode(
                id=component.id,
                name=component.name,
                type=comp_type,
                layer=component.layer,
                metrics=comp_metrics if comp_metrics else None,
                hotspotScore=hotspot_score if hotspot_score > 0 else None,
                issues=estimated_issues,
            )
            nodes.append(node)

        # Build edges from component dependencies
        for component in result.components:
            for dep_id in component.dependencies:
                edge = GraphEdge(
                    source=component.id,
                    target=dep_id,
                    type="dependency",
                    weight=1,
                )
                edges.append(edge)

        # Get unique layers
        layers = [layer.name for layer in result.layers]

        # Calculate summary metrics
        total_deps = len(edges)
        avg_complexity = (
            sum(m.value for m in result.metrics) / len(result.metrics) if result.metrics else 0.0
        )
        hotspot_count = sum(1 for n in nodes if n.hotspotScore and n.hotspotScore > 0.7)

        metrics = ArchitectureGraphMetrics(
            total_components=len(nodes),
            total_dependencies=total_deps,
            avg_complexity=round(avg_complexity, 2),
            hotspot_count=hotspot_count,
        )

        return ArchitectureGraphResponse(
            nodes=nodes,
            edges=edges,
            layers=layers,
            metrics=metrics,
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e)) from e
        logger.error("Failed to get architecture graph for %s: %s", container_name, e)
        raise internal_error(f"Failed to get architecture graph: {str(e)}") from e


def _cycle_signature(members: list[str]) -> str:
    """Order-independent canonical key for a cyclic component group."""
    return "|".join(sorted(set(members)))


def _find_cycle_signatures(adjacency: dict[str, list[str]]) -> list[list[str]]:
    """Enumerate every dependency cycle in a component graph.

    Runs Tarjan's strongly-connected-components over the ``{component_id:
    [dependency_id, ...]}`` adjacency (the same shape ``get_dependencies``
    returns). A cyclic SCC is any component with >1 member, plus any node that
    depends on itself (self-loop). Each cycle is returned as the sorted list of
    its member ids — a stable signature so a baseline->post set difference is
    meaningful.

    The QA ``verify-architecture`` oracle reports only the FIRST cycle path it
    finds (a DFS back-edge); enumerating ALL cyclic SCCs here is what lets the
    DOING delta-gate distinguish a newly-introduced cycle from a pre-existing
    one. Iterative (no recursion-depth limit on large graphs).
    """
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    stack: list[str] = []
    sccs: list[list[str]] = []
    counter = 0

    for root in list(adjacency.keys()):
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, pi = work[-1]
            if pi == 0:
                index[node] = lowlink[node] = counter
                counter += 1
                stack.append(node)
                on_stack[node] = True
            recursed = False
            neighbors = adjacency.get(node) or []
            i = pi
            while i < len(neighbors):
                w = neighbors[i]
                if w not in adjacency:
                    # Edge to a node with no outgoing entry — cannot close a cycle.
                    i += 1
                    continue
                if w not in index:
                    work[-1] = (node, i + 1)
                    work.append((w, 0))
                    recursed = True
                    break
                if on_stack.get(w):
                    lowlink[node] = min(lowlink[node], index[w])
                i += 1
            if recursed:
                continue
            if lowlink[node] == index[node]:
                comp: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == node:
                        break
                sccs.append(comp)
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])

    signatures: list[list[str]] = []
    for comp in sccs:
        if len(comp) > 1:
            signatures.append(sorted(set(comp)))
        else:
            n = comp[0]
            if n in (adjacency.get(n) or []):  # self-loop
                signatures.append([n])
    return signatures


class CycleCheckRequest(BaseModel):
    """Body for the architecture cycle-check gate.

    ``baseline_cycles`` is the cycle-signature list captured from an earlier
    report-mode call (before a refactor). Omit it to run in report mode.
    """

    baseline_cycles: list[list[str]] | None = None


@router.post(
    "/cycle-check/{container_name}",
    operation_id="architecture_cycle_check",
    responses={
        400: {
            "model": Error400Response,
            "description": "Bad request - invalid parameters",
        },
        401: {"model": Error401Response, "description": "Authentication required"},
        403: {"model": Error403Response, "description": "Insufficient permissions"},
        404: {
            "model": Error404Response,
            "description": "Container or resource not found",
        },
        405: {"model": Error405Response, "description": "Method not allowed"},
        422: {
            "model": Error422Response,
            "description": "Validation error in request parameters",
        },
        500: {
            "model": Error500Response,
            "description": "Internal server error during architecture analysis",
        },
    },
)
async def cycle_check(
    container_name: str,
    body: CycleCheckRequest | None = None,
    tenant_id: str = Depends(require_tenant),
):
    """Detect dependency cycles and gate on newly-introduced ones.

    This is the DOING-side architecture-delta gate. The QA ``verify-architecture``
    oracle hard-fails on ANY cycle (absolute); this endpoint instead reports a
    *delta* so a behavior-preserving refactor is blocked only when it introduces
    a cycle that was not present before — pre-existing unrelated cycles do not
    block the refactor in DOING (QA remains the absolute backstop).

    Report mode (no ``baseline_cycles``): returns the current cycle signatures so
    the caller can capture a pre-refactor baseline. Always ``success=True``. It
    FORCE-REBUILDS the graph first — see ``_ensure_graph_populated``; without that
    the baseline describes a stale tree and the gate below can never fail.

    Both modes carry provenance: ``grounded`` / ``ungrounded_reason`` /
    ``graph_built`` / ``component_count`` / ``node_count``. ``grounded=False`` means
    no usable graph was read, so the result is absence of evidence — graders must
    abstain on it rather than treat it as a clean pass.

    Gate mode (``baseline_cycles`` supplied): returns ``success=False`` when the
    current graph contains a cycle signature absent from the baseline. The HTTP
    status stays 200; the ``success`` flag is the gate the DOING orchestrator
    reads (a ``success=False`` coding step is a hard BLOCK).
    """
    baseline = body.baseline_cycles if body is not None else None
    mode = "baseline" if baseline is None else "gate"
    try:
        if not container_name or not container_name.strip():
            raise bad_request("Container name cannot be empty")
        codebase_id, repo_path = await _resolve_container_to_codebase(container_name, tenant_id)
        # BASELINE MODE MUST MEASURE THE TREE AS IT IS NOW. _resolve_container_to_codebase
        # already ran the best-effort self-heal, which NO-OPS on an already-populated
        # graph — so the baseline would otherwise describe whatever earlier state built
        # it (see _ensure_graph_populated). Force one extraction here. Gate mode does NOT
        # rebuild: the refactoring procedure's own `rebuild_code_graph` step already runs
        # triggerArchitectureAnalysis with forceRebuild=true immediately before the gate,
        # so a second extraction would be pure cost.
        baseline_build: dict | None = None
        if baseline is None:
            baseline_build = await _ensure_graph_populated(codebase_id, tenant_id, force=True)

        processor = await _try_get_architecture_processor()
        if processor is None:
            # No graph backend. `success` stays True on purpose: this is the DOING gate,
            # and hard-BLOCKing every card because a backend is down is over-blocking.
            # But `grounded: false` says plainly that NOTHING was measured, so the QA
            # oracle abstains instead of reading this as a clean bill of health.
            return {
                "success": True,
                "mode": mode,
                "grounded": False,
                "ungrounded_reason": (
                    "architecture graph backend unavailable - no graph was read, so "
                    "this result is absence of evidence, not evidence of absence"
                ),
                "graph_built": False,
                "component_count": 0,
                "node_count": 0,
                "cycles": [],
                "new_cycles": [],
                "current_cycles": [],
            }
        job = ArchitectureJobRequest(
            codebase_id=codebase_id, repo_path=repo_path, tenant_id=tenant_id
        )
        # Graph-only read: skip the dependency manifest read.
        result = await processor.process(job, include_dependency_analysis=False)
        adjacency = {comp.id: list(comp.dependencies or []) for comp in result.components}
        current = _find_cycle_signatures(adjacency)

        # PROVENANCE. Downstream graders cannot otherwise tell "measured a real graph and
        # found no cycle" from "read nothing". A cycle needs >= 2 components to exist at
        # all, so a smaller universe is not a finding.
        component_count = len(result.components)
        node_count = int((baseline_build or {}).get("nodes") or 0)
        graph_built = bool(baseline_build and not baseline_build.get("skipped"))
        grounded = component_count >= 2
        ungrounded_reason = None
        if baseline is None and baseline_build is None:
            grounded = False
            ungrounded_reason = (
                "the forced baseline graph build failed, so this baseline may describe a "
                "stale tree rather than the pre-refactor one"
            )
        elif not grounded:
            ungrounded_reason = (
                f"architecture analysis resolved only {component_count} component(s) - a "
                "dependency cycle cannot be expressed in a graph this small"
            )

        if baseline is None:
            return {
                "success": True,
                "mode": "baseline",
                "grounded": grounded,
                "ungrounded_reason": ungrounded_reason,
                "graph_built": graph_built,
                "component_count": component_count,
                "node_count": node_count,
                "cycles": current,
                "current_cycles": current,
            }

        baseline_set = {_cycle_signature(c) for c in baseline}
        new_cycles = [c for c in current if _cycle_signature(c) not in baseline_set]
        success = len(new_cycles) == 0
        response: dict = {
            "success": success,
            "mode": "gate",
            "grounded": grounded,
            "ungrounded_reason": ungrounded_reason,
            "graph_built": graph_built,
            "component_count": component_count,
            "node_count": node_count,
            "baseline_cycle_count": len(baseline),
            "new_cycles": new_cycles,
            "current_cycles": current,
        }
        if not success:
            response["error"] = (
                f"architecture refactor introduced {len(new_cycles)} new dependency "
                "cycle(s): " + "; ".join(" <-> ".join(c) for c in new_cycles)
            )
        return response
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        if hasattr(e, "status_code"):
            raise
        if "not found" in str(e).lower() or "does not exist" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e)) from e
        logger.error("Failed cycle-check for container %s: %s", container_name, e)
        raise internal_error(f"Failed cycle-check: {str(e)}") from e
