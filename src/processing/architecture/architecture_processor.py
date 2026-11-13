"""Module for processing architectural components of a codebase."""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx: Any = None

from config.yaml_config_loader import YamlConfigLoader
from models.architecture_models import (
    ArchitectureAnalysisResult,
    ArchitectureJobRequest,
    Component,
    Layer,
    RepoConfig,
)

from .component_extractor import ComponentExtractor
from .dependency_analyzer import ArchitectureDependencyAnalyzer
from .graph_adapter import GraphAdapter
from .layer_analyzer import LayerAnalyzer
from .metrics_calculator import MetricsCalculator
from .pattern_detector import ArchitecturePatternDetector

logger = logging.getLogger(__name__)


class ArchitectureProcessor:
    """High-level processor for architecture analysis orchestration."""

    def __init__(
        self,
        graph_adapter: GraphAdapter,
        config: RepoConfig | None = None,
    ):
        """
        Initialize the architecture processor with all necessary analyzers.

        Args:
            graph_adapter: Adapter for accessing the code graph
            config: Optional repository configuration
        """
        self.graph_adapter = graph_adapter
        self.config: RepoConfig = config or RepoConfig()

        # Initialize analysis components
        self.component_extractor = ComponentExtractor()
        self.layer_analyzer = LayerAnalyzer()
        self.metrics_calculator = MetricsCalculator()
        self.pattern_detector = ArchitecturePatternDetector()
        self.dependency_analyzer = ArchitectureDependencyAnalyzer()

        # Configuration loader
        self.config_loader = YamlConfigLoader()

    async def process(
        self,
        job: ArchitectureJobRequest,
        *,
        include_dependency_analysis: bool = True,
    ) -> ArchitectureAnalysisResult:
        """Process architecture analysis for the given job request.

        Every stage here is bounded by the repository's size, never by external
        I/O. The result carries components, layers, metrics, patterns and a
        summary; nothing else, so nothing else is computed:

        - The dependency stage only COUNTS the packages the manifests declare
          (``summary.dependencies_analyzed``). It used to run the full
          per-package ``api.osv.dev`` vulnerability scan and then discard
          everything but ``len(packages)``: 96 s of a 130 s request on a 37-file
          repo (cv-sweep-ems, 2026-10-12, egress to osv.dev reset), past the
          120 s edge timeout. ``include_dependency_analysis=False`` skips even
          the manifest read, for the graph-only reads that ignore the count.
        - There is no quality stage. It ran six container-exec linters (ruff,
          mypy, bandit, radon, pylint, black — 33 s on the same repo) and only
          logged the score; the result was never returned. Quality has its own
          endpoints (``/api/analysis/quality``).
        """
        logger.info("Starting architecture analysis for codebase: %s", job.codebase_id)
        start_time = datetime.now(UTC)

        try:
            # Load configuration from repository
            repo_config = await self._load_repository_config(job.repo_path, job.codebase_id)

            # Load the code graph (tenant-scoped — see ArchitectureJobRequest.tenant_id)
            graph = await self.graph_adapter.load_graph(job.codebase_id, job.tenant_id)
            logger.info(
                "Loaded graph with %s nodes and %s edges",
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )

            # Graph analysis (component/layer/metrics/pattern extraction). The
            # extractors return safe empty defaults for an empty graph, but a
            # malformed/partial graph can raise exception types beyond the
            # FileNotFoundError/ImportError/ValueError the optional steps below
            # catch. Guard the whole block so a parsing hiccup yields an honest
            # empty result rather than a 500 — the downstream verify-architecture
            # oracle needs a well-formed (possibly empty) input, never an error.
            try:
                # Extract components
                components = self.component_extractor.extract(graph, repo_config)
                logger.info("Extracted %s components", len(components))

                # Analyze layers
                layer_map = self.layer_analyzer.assign_layers(components, repo_config)
                logger.info("Assigned components to %s layers", len(layer_map))

                # Calculate metrics
                metrics_result = self.metrics_calculator.calc(graph, components)
                if isinstance(metrics_result, list):
                    metrics_list = metrics_result
                else:
                    metrics_list = [metrics_result]  # Ensure it's a list

                # Detect patterns
                patterns = self.pattern_detector.detect(graph, repo_config)
                logger.info("Detected %s patterns", len(patterns))
            except Exception as graph_analysis_error:
                logger.error(
                    "Graph analysis failed for codebase %s (%s nodes, %s edges); "
                    "returning empty architecture: %s",
                    job.codebase_id,
                    graph.number_of_nodes(),
                    graph.number_of_edges(),
                    graph_analysis_error,
                    exc_info=True,
                )
                components = []
                layer_map = {}
                metrics_list = []
                patterns = []

            # Count declared dependencies (optional; skipped for graph-only reads).
            # Manifest parse only — no vulnerability scan (see docstring).
            dependencies_declared = 0
            if include_dependency_analysis:
                try:
                    # Same tenant that scopes the graph reads above: the
                    # manifests are mirrored out of the tenant's container, and
                    # an untenanted call is 404'd by coding-containers' BOLA
                    # guard rather than reaching the workspace.
                    dependencies_declared = await self.dependency_analyzer.count_declared_packages(
                        job.repo_path, job.codebase_id, tenant_id=job.tenant_id
                    )
                    logger.info("Counted %s declared dependencies", dependencies_declared)
                except (FileNotFoundError, ImportError, ValueError) as e:
                    logger.warning("Dependency count failed: %s", e)

            # Convert layer map to Layer objects
            layers_list = [
                Layer(
                    name=layer_name,
                    components=component_ids,
                    component_count=len(component_ids),
                )
                for layer_name, component_ids in layer_map.items()
            ]

            # Create analysis result
            result = ArchitectureAnalysisResult(
                codebase_id=job.codebase_id,
                generated_at=datetime.now(UTC).isoformat(),
                components=components,
                layers=layers_list,
                metrics=metrics_list,
                patterns=patterns,
                summary={
                    "total_components": len(components),
                    "total_layers": len(layers_list),
                    "total_patterns": len(patterns),
                    "dependencies_analyzed": dependencies_declared,
                },
                execution_time=(datetime.now(UTC) - start_time).total_seconds(),
                config_used=repo_config,
            )

            logger.info("Architecture analysis completed for codebase: %s", job.codebase_id)
            return result

        except Exception as e:
            logger.error(
                "Architecture analysis failed for codebase %s: %s",
                job.codebase_id,
                e,
            )
            raise

    async def _load_repository_config(self, repo_path: Path, codebase_id: str) -> RepoConfig:
        """Load ``.codegraph.yml`` from the repo, falling back to defaults.

        The repo lives inside the analysis container (``repo_path`` is a path
        *inside* that container, e.g. ``/workspace``) — not on the analyzer pod's
        local filesystem. Read the config via the coding-containers exec API, the
        same model the quality tools and dependency analyzer use, instead of the
        local-fs ``YamlConfigLoader`` (which logged "Config file not found:
        /workspace"). Any failure (no file, bad YAML, exec/network error) falls
        back to defaults so analysis is never blocked on optional config.
        """
        import yaml

        from clients.coding_containers import get_coding_containers_client

        config_path = f"{str(repo_path).rstrip('/')}/.codegraph.yml"
        try:
            client = get_coding_containers_client()
            content = await client.read_file(codebase_id, config_path)
            if not content:
                logger.debug(
                    "No .codegraph.yml in container '%s'; using default config", codebase_id
                )
                return self.config
            raw_config = yaml.safe_load(content) or {}
            repo_config = RepoConfig(**raw_config) if raw_config else self.config
            logger.info("Loaded .codegraph.yml from container '%s'", codebase_id)
            return repo_config
        except Exception as e:
            # yaml errors, RepoConfig validation, or exec/network failures — none
            # should break architecture analysis; defaults are always valid.
            logger.debug(
                "No usable .codegraph.yml for container '%s', using defaults: %s",
                codebase_id,
                e,
            )
            return self.config

    def _convert_quality_result(self, quality_result) -> dict:
        """Convert quality result to dictionary for serialization."""
        if not quality_result:
            return {}

        return {
            "overall_score": quality_result.overall_score,
            "issues_count": len(quality_result.issues),
            "issues": [
                {
                    "severity": issue.severity,
                    "message": issue.message,
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                }
                # Limit to first 10 issues
                for issue in quality_result.issues[:10]
            ],
            "summary": quality_result.summary,
        }

    def get_component_fingerprint(
        self,
        codebase_id: str,
        components: list[Component] | None = None,
        graph: Any | None = None,
    ) -> str:
        """Calculate SHA256 fingerprint for incremental analysis."""

        fingerprint_data = []

        # Add codebase identifier
        fingerprint_data.append(f"codebase:{codebase_id}")

        if components and graph:
            # Include component structure in fingerprint
            for component in components:
                # Component structure
                comp_data = [
                    f"id:{component.id}",
                    f"name:{component.name}",
                    f"files:{','.join(sorted(component.files))}",
                    f"deps:{','.join(sorted(component.dependencies))}",
                ]

                # Include API surface and structure
                node_signatures = []
                for node_id, node_data in graph.nodes(data=True):
                    if node_data.get("file_path", "") in component.files:
                        # Create signature for each node
                        node_type = node_data.get("type", "unknown")
                        node_name = node_data.get("name", node_id)
                        properties = node_data.get("properties", {})

                        signature_parts = [node_type, node_name]

                        # Add visibility/access modifiers
                        if isinstance(properties, dict):
                            visibility = properties.get("visibility", "")
                            modifiers = properties.get("modifiers", [])
                            if visibility:
                                signature_parts.append(f"vis:{visibility}")
                            if modifiers:
                                signature_parts.append(f"mod:{','.join(sorted(modifiers))}")

                        node_signatures.append("|".join(signature_parts))

                comp_data.append(f"nodes:{';'.join(sorted(node_signatures))}")
                fingerprint_data.append(f"component:{';'.join(comp_data)}")

        # Include configuration if available
        if hasattr(self, "config") and self.config:
            config_data = [
                f"detect_by:{','.join(sorted(self.config.detect_by))}",
                f"folder_roots:{','.join(sorted(self.config.folder_roots))}",
                f"build_files:{','.join(sorted(self.config.build_files))}",
                f"layers:{len(self.config.layers) if self.config.layers is not None else 0}",
            ]
            fingerprint_data.append(f"config:{';'.join(config_data)}")

        # Generate SHA256 hash
        combined_data = "\n".join(fingerprint_data)
        fingerprint = hashlib.sha256(combined_data.encode("utf-8")).hexdigest()

        return fingerprint

    def validate_configuration(self, config: RepoConfig) -> bool:
        """Validate repository configuration."""
        try:
            # Check that detect_by strategies are valid
            valid_strategies = {"folder", "build_file", "annotation"}
            if not all(strategy in valid_strategies for strategy in config.detect_by):
                return False

            # Check that layers configuration is properly formatted
            if config.layers is not None:
                for patterns in config.layers.values():
                    if not isinstance(patterns, (list, tuple)):
                        return False

            return True

        except (AttributeError, TypeError, ValueError):  # fmt: skip
            return False
