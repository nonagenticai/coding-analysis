"""Helper utilities for cloud-native pattern detection.

This module provides utility functions and classes for detecting cloud-native
patterns such as helper relationship detection, proxy behavior analysis, and
Kubernetes-specific pattern recognition.
"""

from __future__ import annotations

import logging

import networkx as nx

from models.architecture_models import DetectedPattern

from .code_pattern_recognition import CodePatternRecognizer

logger = logging.getLogger(__name__)


class CloudPatternHelpers:
    """Helper utilities for cloud-native pattern analysis."""

    def __init__(self, recognizer: CodePatternRecognizer) -> None:
        """Initialize cloud pattern helpers.

        Args:
            recognizer: Code pattern recognition utility instance
        """
        self._recognizer = recognizer

    def is_likely_helper(self, main_node_data: dict, helper_node_data: dict) -> bool:
        """Determine if one component is likely a helper/sidecar to another.

        Args:
            main_node_data: Main component data
            helper_node_data: Helper component data

        Returns:
            True if helper is likely a sidecar, False otherwise
        """
        main_code = self._recognizer.get_node_code(main_node_data)
        helper_code = self._recognizer.get_node_code(helper_node_data)

        # Helpers are typically smaller than the main component
        if len(helper_code) > len(main_code) * 0.8:
            return False

        # Helper names often contain suffixes like Agent, Helper, Sidecar
        helper_name = helper_node_data.get("id", "").lower()
        if any(
            suffix in helper_name for suffix in ["agent", "helper", "sidecar", "proxy", "adapter"]
        ):
            return True

        # Check if helper performs cross-cutting concerns
        return any(
            concern in helper_code.lower()
            for concern in [
                "log",
                "metric",
                "trace",
                "auth",
                "cache",
                "health",
                "monitor",
                "backup",
                "sync",
                "proxy",
                "adapter",
                "debug",
            ]
        )

    def has_proxy_behavior(self, node_data: dict) -> bool:
        """Check if a component has proxy-like behavior.

        Args:
            node_data: Node data to check

        Returns:
            True if node has proxy behavior, False otherwise
        """
        code = self._recognizer.get_node_code(node_data).lower()

        # Look for proxy-like behavior in code
        proxy_indicators = [
            "forward",
            "proxy",
            "redirect",
            "relay",
            "route",
            "send",
            "http.request",
            "fetch",
            "axios",
            "client.get",
            "client.post",
            "request(",
            "transport",
            "delegate",
            "pass",
            "transform",
        ]

        return any(indicator in code for indicator in proxy_indicators)

    def detect_kubernetes_mesh_patterns(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Kubernetes-specific Service Mesh patterns.

        Args:
            graph: The code graph to analyze

        Returns:
            List of detected Kubernetes mesh patterns
        """
        patterns = []

        # Check for Kubernetes-specific mesh patterns
        k8s_mesh_regex = self._recognizer.get_compiled_pattern(
            r"(kind:\s*(deployment|service|pod|daemonset)|apiVersion:\s*v1|apiVersion:\s*apps\/v1|"
            r"kubectl|k8s|kubernetes|helm|istio|mesh)"
        )

        k8s_mesh_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            code = self._recognizer.get_node_code(node_data).lower()

            # Look for Kubernetes-specific mesh indicators
            if self._recognizer.node_matches_pattern(node_data, k8s_mesh_regex) and any(
                term in code for term in ["istio", "linkerd", "mesh", "envoy", "proxy"]
            ):
                k8s_mesh_nodes.append(node_id)

        if k8s_mesh_nodes:
            patterns.append(
                DetectedPattern(
                    pattern_type="ServiceMesh",
                    confidence=0.9,  # Higher confidence for k8s-specific mesh
                    components=k8s_mesh_nodes,
                    description="Kubernetes-based Service Mesh pattern detected",
                    name="K8s Service Mesh Pattern",
                    # Use first node as primary location
                    location=k8s_mesh_nodes[0],
                )
            )

        return patterns

    def detect_kubernetes_gateway_patterns(self, graph: nx.DiGraph) -> list[DetectedPattern]:
        """Detect Kubernetes-specific API Gateway patterns.

        Args:
            graph: The code graph to analyze

        Returns:
            List of detected Kubernetes gateway patterns
        """
        patterns = []

        # Check for Kubernetes-specific API Gateway patterns
        k8s_gateway_regex = self._recognizer.get_compiled_pattern(
            r"(kind:\s*(ingress|gateway)|apiVersion:\s*networking.k8s.io|"
            r"\b(ingress|gateway|api-gateway|kong|ambassador|traefik)\b)"
        )

        k8s_gateway_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            code = self._recognizer.get_node_code(node_data).lower()

            # Look for Kubernetes-specific gateway indicators
            if self._recognizer.node_matches_pattern(node_data, k8s_gateway_regex) and any(
                term in code for term in ["route", "path", "host", "backend", "service"]
            ):
                k8s_gateway_nodes.append(node_id)

        if k8s_gateway_nodes:
            patterns.append(
                DetectedPattern(
                    pattern_type="ApiGateway",
                    confidence=0.85,  # High confidence for k8s-specific gateway
                    components=k8s_gateway_nodes,
                    description="Kubernetes-based API Gateway pattern detected",
                    name="K8s API Gateway Pattern",
                    # Use first node as primary location
                    location=k8s_gateway_nodes[0],
                )
            )

        return patterns

    def analyze_structural_patterns(
        self, graph: nx.DiGraph, node_id: str, node_data: dict
    ) -> list[DetectedPattern]:
        """Analyze structural patterns for a given node.

        Args:
            graph: The code graph to analyze
            node_id: ID of the node to analyze
            node_data: Data of the node to analyze

        Returns:
            List of detected structural patterns
        """
        patterns = []

        # Check for sidecar pattern (main service with helper service)
        if node_data.get("type", "").lower() in [
            "service",
            "container",
            "deployment",
            "pod",
        ]:
            sidecar_patterns = self._detect_structural_sidecar(graph, node_id, node_data)
            patterns.extend(sidecar_patterns)

        # Check for ambassador pattern (client → ambassador → external service)
        if node_data.get("type", "").lower() in ["service", "proxy", "client"]:
            ambassador_patterns = self._detect_structural_ambassador(graph, node_id, node_data)
            patterns.extend(ambassador_patterns)

        # Check for API gateway pattern (one entry point to multiple backends)
        gateway_patterns = self._detect_structural_gateway(graph, node_id, node_data)
        patterns.extend(gateway_patterns)

        return patterns

    def _detect_structural_sidecar(
        self, graph: nx.DiGraph, node_id: str, node_data: dict
    ) -> list[DetectedPattern]:
        """Detect structural sidecar patterns for a node."""
        patterns = []

        # Check if it has exactly one tightly coupled helper
        helpers = []
        for _, helper_id, edge_data in graph.out_edges(node_id, data=True):
            # Look for edges that suggest helper relationship
            if (
                edge_data.get("type") in ["uses", "depends_on", "mounts", "shares"]
                and helper_id != node_id
            ):
                helper_data = graph.nodes[helper_id]
                # Helper should be smaller/more straightforward than main component
                if self.is_likely_helper(node_data, helper_data):
                    helpers.append(helper_id)

        # If we found exactly one helper that looks like a sidecar
        if len(helpers) == 1:
            patterns.append(
                DetectedPattern(
                    pattern_type="Sidecar",
                    confidence=0.7,
                    components=[node_id, helpers[0]],
                    description="Inferred sidecar pattern from component relationships",
                    name="Sidecar Pattern",
                    location=node_id,
                )
            )

        return patterns

    def _detect_structural_ambassador(
        self, graph: nx.DiGraph, node_id: str, node_data: dict
    ) -> list[DetectedPattern]:
        """Detect structural ambassador patterns for a node."""
        patterns = []

        # Check if this component has incoming edges (clients) and outgoing edges
        # (external services)
        in_neighbors = list(graph.predecessors(node_id))
        out_neighbors = list(graph.successors(node_id))

        if in_neighbors and out_neighbors:
            # Check if this node acts as a communication proxy
            if (
                self.has_proxy_behavior(node_data) and len(out_neighbors) <= 3
            ):  # Typically proxies to few external services
                # Count how many clients use this as a proxy to reach external services
                client_count = len(in_neighbors)

                if client_count >= 2:  # Multiple clients using this proxy
                    patterns.append(
                        DetectedPattern(
                            pattern_type="Ambassador",
                            confidence=0.75,
                            components=[node_id] + out_neighbors,
                            description=(
                                "Inferred ambassador pattern from network communication structure"
                            ),
                            name="Ambassador Pattern",
                            location=node_id,
                        )
                    )

        return patterns

    def _detect_structural_gateway(
        self, graph: nx.DiGraph, node_id: str, node_data: dict
    ) -> list[DetectedPattern]:
        """Detect structural API gateway patterns for a node."""
        patterns = []

        # Check for services with many outgoing dependencies but few incoming
        in_edges = list(graph.in_edges(node_id))
        out_edges = list(graph.out_edges(node_id))

        if (
            len(in_edges) <= 3 and len(out_edges) >= 5
        ):  # Gateway typically has few clients but many backends
            # Check if node has routing, auth, or transformation logic
            code = self._recognizer.get_node_code(node_data).lower()
            if any(
                term in code
                for term in [
                    "route",
                    "auth",
                    "transform",
                    "filter",
                    "proxy",
                ]
            ):
                patterns.append(
                    DetectedPattern(
                        pattern_type="ApiGateway",
                        confidence=0.7,
                        components=[node_id],
                        description="Inferred API Gateway from service communication pattern",
                        name="API Gateway Pattern",
                        location=node_id,
                    )
                )

        return patterns

    def find_mesh_control_plane_nodes(self, graph: nx.DiGraph) -> list[str]:
        """Find nodes that appear to be service mesh control plane components.

        Args:
            graph: The code graph to search

        Returns:
            List of node IDs for control plane components
        """
        control_plane_regex = self._recognizer.get_compiled_pattern(
            r"(class|def|function|module)\s+\w*(ControlPlane|Controller|Config|Policy)\w*|"
            r"\b(control[\-_]?plane|controller|config|policy)\b"
        )

        control_plane_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            code = self._recognizer.get_node_code(node_data).lower()

            # Check for control plane components
            if self._recognizer.node_matches_pattern(node_data, control_plane_regex) and any(
                term in code
                for term in [
                    "config",
                    "control",
                    "policy",
                    "manage",
                    "orchestrate",
                ]
            ):
                control_plane_nodes.append(node_id)

        return control_plane_nodes

    def find_mesh_data_plane_nodes(self, graph: nx.DiGraph) -> list[str]:
        """Find nodes that appear to be service mesh data plane components.

        Args:
            graph: The code graph to search

        Returns:
            List of node IDs for data plane components
        """
        data_plane_regex = self._recognizer.get_compiled_pattern(
            r"(class|def|function|module)\s+\w*(DataPlane|Proxy|Sidecar|Ingress|Egress)\w*|"
            r"\b(data[\-_]?plane|proxy|sidecar|ingress|egress)\b"
        )

        data_plane_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            code = self._recognizer.get_node_code(node_data).lower()

            # Check for data plane components
            if self._recognizer.node_matches_pattern(node_data, data_plane_regex) and any(
                term in code
                for term in [
                    "proxy",
                    "route",
                    "traffic",
                    "forward",
                    "intercept",
                ]
            ):
                data_plane_nodes.append(node_id)

        return data_plane_nodes
