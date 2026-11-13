"""Module for analyzing architectural layers in code."""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Sequence
from typing import Any

from models.architecture_models import (
    Component,
    Layer,
    LayerViolation,
    RepoConfig,
)

logger = logging.getLogger(__name__)


class LayerAnalyzer:
    """Analyze and assign architectural layers to components.

    This class is responsible for analyzing component structures, assigning layers,
    detecting layer violations, and generating layer-based visualizations and metrics.
    """

    def assign_layers(
        self, components: list[Component], config: RepoConfig
    ) -> dict[str, list[str]]:
        """Assign components to architectural layers based on configuration."""
        if not config.layers:
            return {"unassigned": [comp.id for comp in components]}

        layer_map: dict[str, list[str]] = {layer: [] for layer in config.layers.keys()}

        # Add "unassigned" layer for components that don't match any pattern
        if "unassigned" not in layer_map:
            layer_map["unassigned"] = []

        for component in components:
            assigned_layer = self._match_component_to_layer(component, config)

            if assigned_layer:
                layer_map[assigned_layer].append(component.id)
                component.layer = assigned_layer
            else:
                layer_map["unassigned"].append(component.id)
                component.layer = "unassigned"

        # Remove empty layers
        return {layer: components for layer, components in layer_map.items() if components}

    def detect_violations(
        self, components: list[Component], config: RepoConfig
    ) -> list[LayerViolation]:
        """Detect violations of layer architecture rules."""
        violations: list[LayerViolation] = []

        # Define layer hierarchy (lower layers should not depend on higher layers)
        layer_hierarchy = self._get_layer_hierarchy(config)

        for component in components:
            if not component.layer or component.layer == "unassigned":
                continue

            component_layer_level = layer_hierarchy.get(component.layer, float("inf"))

            # Check each dependency
            for dep_id in component.dependencies:
                dep_component = next((c for c in components if c.id == dep_id), None)
                if not dep_component or not dep_component.layer:
                    continue

                dep_layer_level = layer_hierarchy.get(dep_component.layer, float("inf"))

                # Violation if component depends on a higher layer
                if component_layer_level > dep_layer_level:
                    violations.append(
                        LayerViolation(
                            component_id=component.id,
                            expected_layer=component.layer,
                            actual_layer=dep_component.layer,
                        )
                    )

        return violations

    def _match_component_to_layer(self, component: Component, config: RepoConfig) -> str | None:
        """Match a component to a layer based on file patterns."""
        if not config.layers:
            return None

        for layer_name, patterns in config.layers.items():
            if self._component_matches_patterns(component, patterns):
                return layer_name
        return None

    def _component_matches_patterns(self, component: Component, patterns: Sequence[str]) -> bool:
        """Check if any file in the component matches any of the patterns."""
        for file_path in component.files:
            for pattern in patterns:
                if self._matches_pattern(file_path, pattern):
                    return True
        return False

    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if a file path matches a glob pattern."""
        return fnmatch.fnmatch(file_path, pattern)

    def _get_layer_hierarchy(self, config: RepoConfig) -> dict[str, int]:
        """Get layer hierarchy levels (lower number = lower layer)."""
        # Default hierarchy for common layer names
        default_hierarchy = {
            "infrastructure": 0,
            "data": 1,
            "domain": 2,
            "service": 3,
            "application": 4,
            "presentation": 5,
            "ui": 6,
        }

        # Use configured layers or fall back to defaults
        hierarchy = {}
        if not config.layers:
            return default_hierarchy

        layer_names = list(config.layers.keys())

        for i, layer_name in enumerate(layer_names):
            if layer_name.lower() in default_hierarchy:
                hierarchy[layer_name] = default_hierarchy[layer_name.lower()]
            else:
                # Assign levels based on order in config
                hierarchy[layer_name] = i

        return hierarchy

    def get_layer_dependencies(self, components: list[Component]) -> dict[str, list[str]]:
        """Get dependencies between layers.

        Args:
            components: List of components to analyze

        Returns:
            Dictionary mapping layer names to their dependencies
        """
        layer_deps: dict[str, set] = {}

        for component in components:
            if not component.layer:
                logger.debug(f"Skipping component {component.id} - no layer assigned")
                continue

            if component.layer not in layer_deps:
                layer_deps[component.layer] = set()

            for dep_id in component.dependencies:
                dep_component = next((c for c in components if c.id == dep_id), None)
                if dep_component and dep_component.layer and dep_component.layer != component.layer:
                    layer_deps[component.layer].add(dep_component.layer)
                    logger.debug(f"Layer dependency: {component.layer} -> {dep_component.layer}")

        return {layer: sorted(deps) for layer, deps in layer_deps.items()}

    def get_layer_metrics(self, components: list[Component]) -> dict[str, dict[str, int | float]]:
        """Get metrics for each layer.

        Args:
            components: List of components to analyze

        Returns:
            Dictionary mapping layer names to their metrics
        """
        layer_metrics: dict[str, dict[str, int | float]] = {}

        for component in components:
            if not component.layer:
                continue

            if component.layer not in layer_metrics:
                layer_metrics[component.layer] = {
                    "component_count": 0,
                    "file_count": 0,
                    "dependency_count": 0,
                    "incoming_dependencies": 0,
                }

            layer_metrics[component.layer]["component_count"] += 1
            layer_metrics[component.layer]["file_count"] += len(component.files)
            layer_metrics[component.layer]["dependency_count"] += len(component.dependencies)

            # Count incoming dependencies
            for other_comp in components:
                if (
                    other_comp.layer
                    and other_comp.layer != component.layer
                    and component.id in other_comp.dependencies
                ):
                    layer_metrics[component.layer]["incoming_dependencies"] += 1

        # Calculate additional metrics
        for layer, metrics in layer_metrics.items():
            # Dependency ratio (outgoing/incoming)
            if metrics["incoming_dependencies"] > 0:
                metrics["dependency_ratio"] = round(
                    metrics["dependency_count"] / metrics["incoming_dependencies"],
                    2,
                )
            else:
                metrics["dependency_ratio"] = float("inf")

            # Files per component (average)
            if metrics["component_count"] > 0:
                metrics["avg_files_per_component"] = round(
                    metrics["file_count"] / metrics["component_count"], 2
                )
            else:
                metrics["avg_files_per_component"] = 0

        return layer_metrics

    def generate_layer_graph_data(
        self, components: list[Component], config: RepoConfig
    ) -> dict[str, Any]:
        """Generate visualization-ready graph data for layer dependencies.

        This method creates a complete graph representation suitable for visualization tools
        like D3.js, Sigma.js, or other graph visualization libraries.

        Args:
            components: List of components to analyze
            config: Repository configuration containing layer information

        Returns:
            Dictionary with nodes and edges suitable for graph visualization
        """
        logger.info("Generating layer graph data for visualization")

        # Get layer hierarchy and dependencies
        layer_hierarchy = self._get_layer_hierarchy(config)
        layer_dependencies = self.get_layer_dependencies(components)
        layer_metrics = self.get_layer_metrics(components)

        # Create nodes for each layer
        nodes = []
        if not config.layers:
            return {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "layer_count": 0,
                    "dependency_count": 0,
                    "violation_count": 0,
                    "hierarchy_levels": {},
                },
            }

        for layer_name in config.layers.keys():
            metrics = layer_metrics.get(
                layer_name,
                {"component_count": 0, "file_count": 0, "dependency_count": 0},
            )

            # Calculate node size based on component count
            size = min(100, max(30, metrics["component_count"] * 5))

            # Determine layer position based on hierarchy
            hierarchy_level = layer_hierarchy.get(layer_name, 0)

            nodes.append(
                {
                    "id": layer_name,
                    "label": layer_name,
                    "size": size,
                    "metrics": metrics,
                    "level": hierarchy_level,
                    "type": "layer",
                }
            )

        # Create edges between layers
        edges = []
        for source_layer, target_layers in layer_dependencies.items():
            for target_layer in target_layers:
                # Count the number of dependencies between components in these layers
                weight = self._count_cross_layer_dependencies(
                    components, source_layer, target_layer
                )

                # Determine if this is a valid or violating edge
                is_violation = self._is_layer_violation(
                    layer_hierarchy, source_layer, target_layer, config
                )

                edges.append(
                    {
                        "id": f"{source_layer}->{target_layer}",
                        "source": source_layer,
                        "target": target_layer,
                        "weight": weight,
                        # Scale edge thickness
                        "size": min(10, max(1, weight / 5)),
                        "violation": is_violation,
                        "type": "dependency",
                    }
                )

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "layer_count": len(nodes),
                "dependency_count": len(edges),
                "violation_count": sum(1 for edge in edges if edge["violation"]),
                "hierarchy_levels": layer_hierarchy,
            },
        }

    def generate_layers(self, components: list[Component], config: RepoConfig) -> list[Layer]:
        """Generate Layer objects from components.

        Args:
            components: List of components to analyze
            config: Repository configuration containing layer information

        Returns:
            List of Layer objects representing the architectural layers
        """
        logger.info(f"Generating Layer objects from {len(components)} components")

        # Assign components to layers
        layer_map = self.assign_layers(components, config)
        layer_deps = self.get_layer_dependencies(components)

        # Create Layer objects
        layers = []
        for layer_name, component_ids in layer_map.items():
            layers.append(
                Layer(
                    name=layer_name,
                    components=component_ids,
                    component_count=len(component_ids),
                    dependencies=layer_deps.get(layer_name, []),
                )
            )

        return layers

    def _count_cross_layer_dependencies(
        self, components: list[Component], source_layer: str, target_layer: str
    ) -> int:
        """Count the number of dependencies between components in two layers.

        Args:
            components: List of components to analyze
            source_layer: Source layer name
            target_layer: Target layer name

        Returns:
            Number of dependencies between components in these layers
        """
        count = 0
        for comp in components:
            if comp.layer != source_layer:
                continue

            for dep_id in comp.dependencies:
                dep_comp = next((c for c in components if c.id == dep_id), None)
                if dep_comp and dep_comp.layer == target_layer:
                    count += 1

        return count

    def _is_layer_violation(
        self,
        layer_hierarchy: dict[str, int],
        source_layer: str,
        target_layer: str,
        config: RepoConfig,
    ) -> bool:
        """Determine if a layer dependency is a violation of layer architecture rules.

        Args:
            layer_hierarchy: Dictionary mapping layer names to their hierarchy levels
            source_layer: Source layer name
            target_layer: Target layer name
            config: Repository configuration

        Returns:
            True if this dependency violates layer architecture rules, False otherwise
        """
        # Get layer levels
        source_level = layer_hierarchy.get(source_layer, float("inf"))
        target_level = layer_hierarchy.get(target_layer, float("inf"))

        # Check allowed layer dependencies if configured
        allowed_dependencies = getattr(config, "allowed_layer_dependencies", {})
        allowed_for_layer = allowed_dependencies.get(source_layer, [])

        # Violation if lower layer depends on higher layer
        # unless explicitly allowed in config
        return source_level > target_level and target_layer not in allowed_for_layer
