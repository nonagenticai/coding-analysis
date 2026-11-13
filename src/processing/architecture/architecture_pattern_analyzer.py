"""Architecture pattern analysis module for cloud-native and architectural patterns.

This module provides the ArchitecturePatternAnalyzer class which analyzes code graphs to
identify cloud-native patterns including Sidecar, Ambassador, Adapter, Circuit Breaker,
Service Mesh, and API Gateway patterns.
"""

from __future__ import annotations

import logging

import networkx as nx

from models.architecture_models import DetectedPattern

from .cloud_pattern_helpers import CloudPatternHelpers
from .code_pattern_recognition import CodePatternRecognizer

logger = logging.getLogger(__name__)


class ArchitecturePatternAnalyzer:
    """Analyze architectural and cloud-native patterns in code graphs."""

    def __init__(self, recognizer: CodePatternRecognizer) -> None:
        """Initialize the architecture pattern analyzer.

        Args:
            recognizer: Code pattern recognition utility instance
        """
        self._recognizer = recognizer
        self._cloud_helpers = CloudPatternHelpers(recognizer)

    def detect_all_architecture_patterns(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect all cloud-native and architectural patterns.

        Args:
            graph: The code graph to analyze for patterns

        Returns:
            List of detected architectural patterns
        """
        patterns = []

        # Detect cloud-native patterns
        patterns.extend(self._detect_sidecar_pattern(graph))
        patterns.extend(self._detect_ambassador_pattern(graph))
        patterns.extend(self._detect_adapter_pattern(graph))
        patterns.extend(self._detect_circuit_breaker_pattern(graph))
        patterns.extend(self._detect_service_mesh_pattern(graph))
        patterns.extend(self._detect_api_gateway_pattern(graph))

        return patterns

    def _detect_sidecar_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Sidecar pattern in cloud-native applications.

        The sidecar pattern involves a helper container that extends or enhances
        the main container. It typically includes shared resources and a common lifecycle.

        Args:
            graph: The code graph to analyze for patterns

        Returns:
            List of detected Sidecar patterns
        """
        patterns = []

        # Regex for explicit sidecar naming
        sidecar_regex = self._recognizer.get_compiled_pattern(
            r"(class|def|function|module)\s+\w*Sidecar\w*|\bsidecar\b"
        )

        # Look for explicit sidecar mentions
        for node_id, node_data in graph.nodes(data=True):
            if self._recognizer.node_matches_pattern(node_data, sidecar_regex):
                patterns.append(
                    DetectedPattern(
                        pattern_type="Sidecar",
                        confidence=0.8,
                        components=[node_id],
                        description="Sidecar cloud pattern detected",
                        name="Sidecar Pattern",
                        location=node_id,
                    )
                )

        # Look for structural sidecar patterns using helper
        for node_id, node_data in graph.nodes(data=True):
            structural_patterns = self._cloud_helpers.analyze_structural_patterns(
                graph, node_id, node_data
            )
            # Filter for sidecar patterns only
            sidecar_patterns = [p for p in structural_patterns if p.pattern_type == "Sidecar"]
            patterns.extend(sidecar_patterns)

        return patterns

    def _detect_ambassador_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Ambassador pattern in cloud-native applications.

        The ambassador pattern involves a proxy service that handles
        network communication on behalf of the main application.

        Args:
            graph: The code graph to analyze for patterns

        Returns:
            List of detected Ambassador patterns
        """
        patterns = []

        # Regex for explicit ambassador naming
        ambassador_regex = self._recognizer.get_compiled_pattern(
            r"(class|def|function|module)\s+\w*(Ambassador|Proxy|ApiGateway)\w*|"
            r"\b(ambassador|proxy|envoy)\b"
        )

        # Look for explicit ambassador mentions
        for node_id, node_data in graph.nodes(data=True):
            if self._recognizer.node_matches_pattern(node_data, ambassador_regex):
                confidence = 0.7
                if (
                    "ambassador" in str(node_id).lower()
                    or "ambassador" in self._recognizer.get_node_code(node_data).lower()
                ):
                    confidence = 0.9  # Higher confidence for explicit ambassador naming

                patterns.append(
                    DetectedPattern(
                        pattern_type="Ambassador",
                        confidence=confidence,
                        components=[node_id],
                        description="Ambassador cloud pattern detected",
                        name="Ambassador Pattern",
                        location=node_id,
                    )
                )

        # Look for structural ambassador patterns using helper
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get("type", "").lower() in ["service", "proxy", "client"]:
                structural_patterns = self._cloud_helpers.analyze_structural_patterns(
                    graph, node_id, node_data
                )
                # Filter for ambassador patterns only
                ambassador_patterns = [
                    p for p in structural_patterns if p.pattern_type == "Ambassador"
                ]
                patterns.extend(ambassador_patterns)

        return patterns

    def _detect_adapter_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Adapter pattern in cloud-native applications.

        The adapter pattern provides a consistent interface to varying
        underlying services or systems, often used for legacy integration.

        Args:
            graph: The code graph to analyze for patterns

        Returns:
            List of detected Adapter patterns
        """
        patterns = []

        # Regex for explicit adapter naming
        adapter_regex = self._recognizer.get_compiled_pattern(
            r"(class|def|function|module)\s+\w*(Adapter|Wrapper|Translator)\w*|"
            r"\b(adapt|translate|transform)\w*\b"
        )

        # Look for explicit adapter mentions
        for node_id, node_data in graph.nodes(data=True):
            if self._recognizer.node_matches_pattern(node_data, adapter_regex):
                patterns.append(
                    DetectedPattern(
                        pattern_type="Adapter",
                        confidence=0.8,
                        components=[node_id],
                        description="Adapter cloud pattern detected",
                        name="Adapter Pattern",
                        location=node_id,
                    )
                )

        # Look for structural adapter patterns (implements interface but delegates to another class)
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get("type", "").lower() == "class":
                # Look for implements edge and delegates edge
                implements_edges = []
                delegates_edges = []

                for _, target_id, edge_data in graph.out_edges(node_id, data=True):
                    if edge_data.get("type") == "implements":
                        implements_edges.append(target_id)
                    elif edge_data.get("type") in [
                        "calls",
                        "uses",
                        "delegates_to",
                    ]:
                        delegates_edges.append(target_id)

                # If class implements an interface and delegates to another component,
                # it's likely an adapter
                if implements_edges and delegates_edges and len(implements_edges) <= 3:
                    patterns.append(
                        DetectedPattern(
                            pattern_type="Adapter",
                            confidence=0.7,
                            components=[node_id] + implements_edges + delegates_edges,
                            description=(
                                "Inferred adapter pattern from implementation and delegation structure"
                            ),
                            name="Adapter Pattern",
                            location=node_id,
                        )
                    )

        return patterns

    def _detect_circuit_breaker_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Circuit Breaker pattern in cloud-native applications.

        The circuit breaker prevents cascading failures by failing fast when
        a downstream service is unavailable or responding slowly.

        Args:
            graph: The code graph to analyze for patterns

        Returns:
            List of detected Circuit Breaker patterns
        """
        patterns = []

        # Regex for circuit breaker naming and library usage
        circuit_regex = self._recognizer.get_compiled_pattern(
            r"(class|def|function|module)\s+\w*(Circuit|Breaker|Resilient|Fallback)\w*|"
            r"\b(circuitbreaker|resilience4j|hystrix|polly|timeout|retry|fallback)\b"
        )

        # Look for circuit breaker mentions or library imports
        for node_id, node_data in graph.nodes(data=True):
            if self._recognizer.node_matches_pattern(node_data, circuit_regex):
                confidence = 0.7
                code = self._recognizer.get_node_code(node_data).lower()

                # Increase confidence based on specific indicators
                if "circuitbreaker" in code or "circuit_breaker" in code:
                    confidence = 0.9
                elif any(lib in code for lib in ["resilience4j", "hystrix", "polly"]):
                    confidence = 0.85

                patterns.append(
                    DetectedPattern(
                        pattern_type="CircuitBreaker",
                        confidence=confidence,
                        components=[node_id],
                        description="Circuit Breaker cloud resilience pattern detected",
                        name="Circuit Breaker Pattern",
                        location=node_id,
                    )
                )

        return patterns

    def _detect_service_mesh_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Service Mesh pattern in cloud-native applications.

        The service mesh provides service-to-service communication with
        additional capabilities like observability, security, and traffic control.

        Args:
            graph: The code graph to analyze for patterns

        Returns:
            List of detected Service Mesh patterns
        """
        patterns = []

        # Regex for service mesh tech and components
        mesh_regex = self._recognizer.get_compiled_pattern(
            r"(class|def|function|module)\s+\w*(Mesh|Istio|Linkerd|Envoy|Consul|Proxy)\w*|"
            r"\b(mesh|istio|linkerd|envoy|consul|proxy|sidecar)\b"
        )

        # Find mesh technology components
        mesh_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            # Check for service mesh technology mentions
            if self._recognizer.node_matches_pattern(node_data, mesh_regex):
                mesh_nodes.append(node_id)

        # Find control plane and data plane components using helpers
        control_plane_nodes = self._cloud_helpers.find_mesh_control_plane_nodes(graph)
        data_plane_nodes = self._cloud_helpers.find_mesh_data_plane_nodes(graph)

        # Detect based on explicit technology mentions
        if mesh_nodes:
            patterns.append(
                DetectedPattern(
                    pattern_type="ServiceMesh",
                    confidence=0.8,
                    components=mesh_nodes,
                    description="Service Mesh pattern detected via mesh technology references",
                    name="Service Mesh Pattern",
                    location=mesh_nodes[0],
                )
            )

        # Detect based on control plane and data plane architecture
        if control_plane_nodes and data_plane_nodes:
            patterns.append(
                DetectedPattern(
                    pattern_type="ServiceMesh",
                    confidence=0.85,
                    components=control_plane_nodes + data_plane_nodes,
                    description=(
                        "Complete Service Mesh architecture detected with control and data planes"
                    ),
                    name="Service Mesh Pattern",
                    location=control_plane_nodes[0],
                )
            )

        # Check for Kubernetes-specific mesh patterns using helper
        k8s_patterns = self._cloud_helpers.detect_kubernetes_mesh_patterns(graph)
        patterns.extend(k8s_patterns)

        return patterns

    def _detect_api_gateway_pattern(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect API Gateway pattern in cloud-native applications.

        The API Gateway pattern provides a single entry point for clients to interact
        with a set of backend microservices.

        Args:
            graph: The code graph to analyze for patterns

        Returns:
            List of detected API Gateway patterns
        """
        patterns = []

        # Regex for API gateway naming
        gateway_regex = self._recognizer.get_compiled_pattern(
            r"(class|def|function|module)\s+\w*(Gateway|Router|ApiGateway|Bff)\w*|"
            r"\b(gateway|apigw|router|apigateway|bff|edge)\b"
        )

        # Look for explicit gateway mentions
        for node_id, node_data in graph.nodes(data=True):
            if self._recognizer.node_matches_pattern(node_data, gateway_regex):
                confidence = 0.75
                code = self._recognizer.get_node_code(node_data).lower()

                # Increase confidence based on specific indicators
                if "apigateway" in code or "api_gateway" in code:
                    confidence = 0.9
                elif "gateway" in node_id.lower() or "router" in node_id.lower():
                    confidence = 0.85

                patterns.append(
                    DetectedPattern(
                        pattern_type="ApiGateway",
                        confidence=confidence,
                        components=[node_id],
                        description="API Gateway pattern detected",
                        name="API Gateway Pattern",
                        location=node_id,
                    )
                )

        # Look for structural API gateway patterns using helper
        for node_id, node_data in graph.nodes(data=True):
            structural_patterns = self._cloud_helpers.analyze_structural_patterns(
                graph, node_id, node_data
            )
            # Filter for gateway patterns only
            gateway_patterns = [p for p in structural_patterns if p.pattern_type == "ApiGateway"]
            patterns.extend(gateway_patterns)

        # Check for Kubernetes-specific API Gateway patterns using helper
        k8s_patterns = self._cloud_helpers.detect_kubernetes_gateway_patterns(graph)
        patterns.extend(k8s_patterns)

        return patterns

    # Delegate helper methods to the cloud pattern helpers

    def _is_likely_helper(self, main_node_data: dict, helper_node_data: dict) -> bool:
        """Check if likely helper - delegate to cloud helpers."""
        return self._cloud_helpers.is_likely_helper(main_node_data, helper_node_data)

    def _has_proxy_behavior(self, node_data: dict) -> bool:
        """Check proxy behavior - delegate to cloud helpers."""
        return self._cloud_helpers.has_proxy_behavior(node_data)
