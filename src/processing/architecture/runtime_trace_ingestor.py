"""Runtime trace ingestion module for merging runtime execution data with static code graphs.

This module provides functionality to parse various trace formats (Jaeger, Zipkin, OpenTelemetry)
and merge runtime dependency information with static code analysis to create comprehensive
architecture graphs that reflect both static structure and runtime behavior.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict

from models.architecture_models import ArchitectureRuntimeEdge

# Type alias for clarity
RuntimeEdge = ArchitectureRuntimeEdge

logger = logging.getLogger(__name__)


class RuntimeSpan(BaseModel):
    """Represents a runtime execution span from tracing data."""

    span_id: str
    trace_id: str
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: datetime
    duration_ms: float
    parent_span_id: str | None = None
    tags: dict[str, Any] = {}

    model_config = ConfigDict(extra="forbid")


class RuntimeTraceIngestor:
    """Ingests runtime traces and merges them into the static architecture graph."""

    def __init__(self) -> None:
        """Initialize the runtime trace ingestor with supported trace formats."""
        self.supported_formats = ["jaeger", "zipkin", "opentelemetry"]

    async def ingest_traces(
        self,
        codebase_id: str,
        trace_files: list[Path],
        static_graph: nx.DiGraph,
    ) -> tuple[nx.DiGraph, list[RuntimeEdge]]:
        """
        Ingest runtime traces and merge with static graph.

        Args:
            codebase_id: Unique identifier for the codebase
            trace_files: List of trace files to process
            static_graph: Static code graph to augment

        Returns:
            Tuple of (merged_graph, runtime_edges)
        """
        logger.info("Ingesting %d trace files for %s", len(trace_files), codebase_id)

        all_spans = []
        runtime_edges = []

        try:
            # Parse all trace files
            for trace_file in trace_files:
                spans = await self._parse_trace_file(trace_file)
                all_spans.extend(spans)

            logger.info("Parsed %d spans from trace files", len(all_spans))

            # Build runtime dependency graph
            runtime_graph = self._build_runtime_graph(all_spans)

            # Extract runtime edges
            runtime_edges = self._extract_runtime_edges(runtime_graph, all_spans)

            # Merge with static graph
            merged_graph = self._merge_graphs(static_graph, runtime_graph, runtime_edges)

            logger.info("Augmented graph with %d runtime edges", len(runtime_edges))
            return merged_graph, runtime_edges

        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error("Trace ingestion failed for %s: %s", codebase_id, e)
            return static_graph, []

    async def _parse_trace_file(self, trace_file: Path) -> list[RuntimeSpan]:
        """Parse a single trace file based on its format."""
        try:
            with open(trace_file, encoding="utf-8") as f:
                data = json.load(f)

            # Detect format
            if self._is_jaeger_format(data):
                return self._parse_jaeger_traces(data)
            if self._is_zipkin_format(data):
                return self._parse_zipkin_traces(data)
            if self._is_opentelemetry_format(data):
                return self._parse_opentelemetry_traces(data)
            logger.warning("Unknown trace format in %s", trace_file)
            return []

        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error("Failed to parse trace file %s: %s", trace_file, e)
            return []

    def _is_jaeger_format(self, data: Any) -> bool:
        """Check if data is in Jaeger format."""
        return (
            isinstance(data, dict)
            and "data" in data
            and isinstance(data["data"], list)
            and len(data["data"]) > 0
            and "spans" in data["data"][0]
        )

    def _is_zipkin_format(self, data: Any) -> bool:
        """Check if data is in Zipkin format."""
        return (
            isinstance(data, list)
            and len(data) > 0
            and isinstance(data[0], dict)
            and "traceID" in data[0]
            and "spanID" in data[0]
        )

    def _is_opentelemetry_format(self, data: Any) -> bool:
        """Check if data is in OpenTelemetry format."""
        return isinstance(data, dict) and "resourceSpans" in data

    def _parse_jaeger_traces(self, data: dict[str, Any]) -> list[RuntimeSpan]:
        """Parse Jaeger trace format."""
        spans = []

        try:
            for trace_data in data.get("data", []):
                for span_data in trace_data.get("spans", []):
                    span = self._create_span_from_jaeger(span_data, trace_data.get("traceID", ""))
                    if span:
                        spans.append(span)
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Failed to parse Jaeger traces: %s", e)

        return spans

    def _parse_zipkin_traces(self, data: list[dict[str, Any]]) -> list[RuntimeSpan]:
        """Parse Zipkin trace format."""
        spans = []

        try:
            for span_data in data:
                span = self._create_span_from_zipkin(span_data)
                if span:
                    spans.append(span)
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Failed to parse Zipkin traces: %s", e)

        return spans

    def _parse_opentelemetry_traces(self, data: dict[str, Any]) -> list[RuntimeSpan]:
        """Parse OpenTelemetry trace format."""
        spans = []

        try:
            for resource_span in data.get("resourceSpans", []):
                for scope_span in resource_span.get("scopeSpans", []):
                    for span_data in scope_span.get("spans", []):
                        span = self._create_span_from_opentelemetry(span_data)
                        if span:
                            spans.append(span)
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Failed to parse OpenTelemetry traces: %s", e)

        return spans

    def _create_span_from_jaeger(
        self, span_data: dict[str, Any], trace_id: str
    ) -> RuntimeSpan | None:
        """Create RuntimeSpan from Jaeger span data."""
        try:
            start_time = datetime.fromtimestamp(span_data["startTime"] / 1_000_000)
            duration_ms = span_data["duration"] / 1000.0
            end_time = datetime.fromtimestamp(
                (span_data["startTime"] + span_data["duration"]) / 1_000_000
            )

            tags = {}
            for tag in span_data.get("tags", []):
                tags[tag["key"]] = tag["value"]

            return RuntimeSpan(
                span_id=span_data["spanID"],
                trace_id=trace_id,
                operation_name=span_data["operationName"],
                service_name=span_data.get("process", {}).get("serviceName", "unknown"),
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                parent_span_id=span_data.get("parentSpanID"),
                tags=tags,
            )
        except (KeyError, ValueError, TypeError, OSError) as e:
            logger.debug("Failed to create span from Jaeger data: %s", e)
            return None

    def _create_span_from_zipkin(self, span_data: dict[str, Any]) -> RuntimeSpan | None:
        """Create RuntimeSpan from Zipkin span data."""
        try:
            start_time = datetime.fromtimestamp(span_data["timestamp"] / 1_000_000)
            duration_ms = span_data["duration"] / 1000.0
            end_time = datetime.fromtimestamp(
                (span_data["timestamp"] + span_data["duration"]) / 1_000_000
            )

            return RuntimeSpan(
                span_id=span_data["id"],
                trace_id=span_data["traceId"],
                operation_name=span_data.get("name", "unknown"),
                service_name=span_data.get("localEndpoint", {}).get("serviceName", "unknown"),
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                parent_span_id=span_data.get("parentId"),
                tags=span_data.get("tags", {}),
            )
        except (KeyError, ValueError, TypeError, OSError) as e:
            logger.debug("Failed to create span from Zipkin data: %s", e)
            return None

    def _create_span_from_opentelemetry(self, span_data: dict[str, Any]) -> RuntimeSpan | None:
        """Create RuntimeSpan from OpenTelemetry span data."""
        try:
            start_time_ns = int(span_data["startTimeUnixNano"])
            end_time_ns = int(span_data["endTimeUnixNano"])

            start_time = datetime.fromtimestamp(start_time_ns / 1_000_000_000)
            end_time = datetime.fromtimestamp(end_time_ns / 1_000_000_000)
            duration_ms = (end_time_ns - start_time_ns) / 1_000_000

            attributes = {}
            for attr in span_data.get("attributes", []):
                if "stringValue" in attr["value"]:
                    attributes[attr["key"]] = attr["value"]["stringValue"]
                elif "intValue" in attr["value"]:
                    attributes[attr["key"]] = attr["value"]["intValue"]

            return RuntimeSpan(
                span_id=span_data["spanId"],
                trace_id=span_data["traceId"],
                operation_name=span_data.get("name", "unknown"),
                service_name=attributes.get("service.name", "unknown"),
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                parent_span_id=span_data.get("parentSpanId"),
                tags=attributes,
            )
        except (KeyError, ValueError, TypeError, OSError) as e:
            logger.debug("Failed to create span from OpenTelemetry data: %s", e)
            return None

    def _build_runtime_graph(self, spans: list[RuntimeSpan]) -> nx.DiGraph:
        """Build a runtime dependency graph from spans."""
        graph: nx.DiGraph = nx.DiGraph()

        # Group spans by trace
        traces: dict[str, list[RuntimeSpan]] = {}
        for span in spans:
            if span.trace_id not in traces:
                traces[span.trace_id] = []
            traces[span.trace_id].append(span)

        # Build edges from parent-child relationships
        for trace_spans in traces.values():
            span_map = {span.span_id: span for span in trace_spans}

            for span in trace_spans:
                # Add node
                graph.add_node(
                    span.service_name,
                    operations=graph.nodes.get(span.service_name, {}).get("operations", set())
                    | {span.operation_name},
                )

                # Add edge from parent
                if span.parent_span_id and span.parent_span_id in span_map:
                    parent_span = span_map[span.parent_span_id]

                    if parent_span.service_name != span.service_name:
                        edge_key = (
                            parent_span.service_name,
                            span.service_name,
                        )

                        if graph.has_edge(*edge_key):
                            edge_data = graph.edges[edge_key]
                            edge_data["call_count"] += 1
                            edge_data["total_duration"] += span.duration_ms
                        else:
                            graph.add_edge(
                                parent_span.service_name,
                                span.service_name,
                                call_count=1,
                                total_duration=span.duration_ms,
                                runtime=True,
                            )

        return graph

    def _extract_runtime_edges(
        self, runtime_graph: nx.DiGraph, spans: list[RuntimeSpan]
    ) -> list[RuntimeEdge]:
        """Extract runtime edges with metrics.

        Args:
            runtime_graph: Graph containing runtime dependencies
            spans: List of runtime spans (unused in current implementation,
                  reserved for future error rate calculations)
        """
        # spans parameter unused but kept for future error rate calculation
        _ = spans  # Mark as intentionally unused
        edges = []

        for source, target, data in runtime_graph.edges(data=True):
            call_count = data.get("call_count", 0)
            total_duration = data.get("total_duration", 0.0)
            avg_duration = total_duration / call_count if call_count > 0 else 0.0

            # Calculate error rate (streamlined - would need error span data)
            error_rate = 0.0

            # Calculate confidence based on call count
            # More calls = higher confidence
            confidence = min(1.0, call_count / 10.0)

            edge = ArchitectureRuntimeEdge(
                source=source,
                target=target,
                call_count=call_count,
                avg_duration_ms=avg_duration,
                error_rate=error_rate,
                confidence=confidence,
            )

            edges.append(edge)

        return edges

    def _merge_graphs(
        self,
        static_graph: nx.DiGraph,
        runtime_graph: nx.DiGraph,
        runtime_edges: list[RuntimeEdge],
    ) -> nx.DiGraph:
        """Merge static and runtime graphs.

        Args:
            static_graph: Original static code graph
            runtime_graph: Runtime dependency graph (unused in current implementation,
                          reserved for future node merging operations)
            runtime_edges: List of runtime edges to merge
        """
        # runtime_graph parameter unused but kept for future node merging
        _ = runtime_graph  # Mark as intentionally unused
        merged_graph = static_graph.copy()

        # Add runtime edges to static graph
        for edge in runtime_edges:
            if merged_graph.has_edge(edge.source, edge.target):
                # Augment existing edge with runtime data
                merged_graph.edges[edge.source, edge.target].update(
                    {
                        "runtime_call_count": edge.call_count,
                        "runtime_avg_duration_ms": edge.avg_duration_ms,
                        "runtime_error_rate": edge.error_rate,
                        "runtime_confidence": edge.confidence,
                        "has_runtime_data": True,
                    }
                )
            else:
                # Add new runtime-only edge
                merged_graph.add_edge(
                    edge.source,
                    edge.target,
                    runtime_call_count=edge.call_count,
                    runtime_avg_duration_ms=edge.avg_duration_ms,
                    runtime_error_rate=edge.error_rate,
                    runtime_confidence=edge.confidence,
                    runtime_only=True,
                )

        return merged_graph

    def get_runtime_metrics(self, runtime_edges: list[RuntimeEdge]) -> dict[str, Any]:
        """Get summary metrics from runtime edges."""
        if not runtime_edges:
            return {}

        total_calls = sum(edge.call_count for edge in runtime_edges)
        avg_duration = (
            sum(edge.avg_duration_ms * edge.call_count for edge in runtime_edges) / total_calls
            if total_calls > 0
            else 0.0
        )
        avg_error_rate = sum(edge.error_rate for edge in runtime_edges) / len(runtime_edges)

        # Find hottest paths
        hottest_edges = sorted(runtime_edges, key=lambda e: e.call_count, reverse=True)[:5]
        slowest_edges = sorted(runtime_edges, key=lambda e: e.avg_duration_ms, reverse=True)[:5]

        return {
            "total_runtime_edges": len(runtime_edges),
            "total_calls": total_calls,
            "avg_duration_ms": round(avg_duration, 2),
            "avg_error_rate": round(avg_error_rate, 4),
            "hottest_paths": [
                {"source": e.source, "target": e.target, "calls": e.call_count}
                for e in hottest_edges
            ],
            "slowest_paths": [
                {
                    "source": e.source,
                    "target": e.target,
                    "duration_ms": e.avg_duration_ms,
                }
                for e in slowest_edges
            ],
        }
