"""Machine learning-based confidence calculator for architectural pattern detection.

This module implements ML-assisted confidence scoring for detected architectural patterns
using feature extraction and trained classifier models.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import networkx as nx

try:
    from sklearn.ensemble import RandomForestClassifier

    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback when sklearn is not available
    RandomForestClassifier: Any = None
    SKLEARN_AVAILABLE = False

from models.architecture_models import DetectedPattern, RepoConfig

logger = logging.getLogger(__name__)


class MLConfidenceCalculator:
    """Machine learning-based confidence calculator for architectural pattern detection.

    This class implements ML-assisted confidence scoring for detected architectural patterns
    using feature extraction and a trained classifier model.
    """

    def __init__(self, config: RepoConfig | None = None):
        """Initialize the ML confidence calculator.

        Args:
            config: Repository configuration with ML settings
        """
        self.config = config
        self.model_directory = Path("models")
        self.models: dict[str, Any] = {}  # Pattern type -> trained model
        self.initialized = False

        # Default confidence thresholds per pattern type
        self.confidence_thresholds = {
            "Singleton": 0.7,
            "Factory": 0.7,
            "Observer": 0.7,
            "MVC": 0.8,
            "Repository": 0.7,
            "Service": 0.6,
            "Sidecar": 0.7,
            "Ambassador": 0.7,
            "Adapter": 0.7,
            "CircuitBreaker": 0.8,
            "ServiceMesh": 0.8,
            "ApiGateway": 0.7,
            "Default": 0.7,  # Default threshold for any other patterns
        }

        # Only initialize if config is provided, ML is enabled, and sklearn is available
        if config and self._is_ml_enabled() and SKLEARN_AVAILABLE:
            self._initialize()
        elif not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, ML confidence calculation disabled")

    def _is_ml_enabled(self) -> bool:
        """Check if ML confidence scoring is enabled in config."""
        if not self.config:
            return False

        ml_settings = getattr(self.config, "ml_confidence_settings", None)
        if not ml_settings:
            return False

        return bool(ml_settings.get("enabled", False))

    def _initialize(self) -> None:
        """Initialize ML components and load pre-trained models.

        This method:
        - Creates the model directory if it doesn't exist
        - Updates confidence thresholds from config if provided
        - Loads pre-trained models for pattern detection

        Raises:
            OSError: If model directory creation fails
            ValueError: If model loading fails
            ImportError: If required ML dependencies are missing
        """
        try:
            # Create model directory if it doesn't exist
            if not self.model_directory.exists():
                self.model_directory.mkdir(parents=True, exist_ok=True)

            # Update confidence thresholds from config if provided
            if self.config and hasattr(self.config, "ml_confidence_settings"):
                ml_settings = self.config.ml_confidence_settings
                if ml_settings and "thresholds" in ml_settings:
                    self.confidence_thresholds.update(ml_settings["thresholds"])

            # Load pre-trained models
            self._load_models()
            self.initialized = True
            logger.info("ML confidence calculator initialized successfully")
        except (OSError, ValueError, ImportError) as e:
            logger.error("Failed to initialize ML confidence calculator: %s", e)
            self.initialized = False

    def _load_models(self) -> None:
        """Load pre-trained pattern classifier models.

        Iterates through all configured pattern types and loads their corresponding
        pre-trained models from disk (if they exist). Models are stored in the
        models directory as .pkl files with naming convention {pattern_type}_classifier.pkl

        Raises:
            OSError: If model file cannot be read
            ValueError: If model file is corrupted
            pickle.PickleError: If model deserialization fails
        """
        for pattern_type in self.confidence_thresholds:
            if pattern_type == "Default":
                continue

            model_path = self.model_directory / f"{pattern_type.lower()}_classifier.pkl"
            if model_path.exists():
                try:
                    with open(model_path, "rb") as f:
                        self.models[pattern_type] = pickle.load(f)  # nosec B301
                    logger.info("Loaded model for pattern type: %s", pattern_type)
                except (OSError, ValueError, pickle.PickleError) as e:
                    logger.warning("Failed to load model for %s: %s", pattern_type, e)

    def calculate_confidence(self, pattern: DetectedPattern, graph: nx.DiGraph) -> float:
        """Calculate confidence score for a detected pattern using ML features.

        This method:
        - Extracts features from the pattern and graph
        - Uses the trained model to predict the confidence score
        - Blends the predicted score with the original confidence for smoother transition

        Args:
            pattern: The detected pattern to calculate confidence for
            graph: The graph containing the components

        Returns:
            A confidence score between 0.0 and 1.0
        """
        if not self.initialized or pattern.pattern_type not in self.models:
            # Fall back to existing confidence if ML is not available
            return pattern.confidence

        try:
            # Extract features for the pattern
            features = self.extract_features(pattern, graph)

            # Get prediction probability from model
            model = self.models[pattern.pattern_type]
            confidence_score = model.predict_proba([features])[0][
                1
            ]  # Probability of positive class

            # Blend with original confidence for smoother transition
            blended_score = (confidence_score * 0.7) + (pattern.confidence * 0.3)

            # Ensure score is within valid range
            return float(max(0.0, min(1.0, blended_score)))
        except (ValueError, AttributeError, KeyError) as e:
            logger.error("Error calculating ML confidence: %s", e)
            return pattern.confidence  # Fall back to original confidence

    def extract_features(self, pattern: DetectedPattern, graph: nx.DiGraph) -> list[float]:
        """Extract features from pattern components for ML classification.

        This method:
        - Combines relationship, text, and structural features
        - Returns a list of numerical features for ML model

        Args:
            pattern: The pattern to extract features from
            graph: The graph containing the components

        Returns:
            List of numerical features for ML model
        """
        # Combine all feature types
        relationship_features = self._get_relationship_features(pattern.components, graph)
        text_features = self._get_text_features(pattern.components, graph)
        structural_features = self._get_structural_features(pattern.components, graph)

        # Combine all features
        features = relationship_features + text_features + structural_features

        # If features are empty (unlikely), return a default feature vector
        if not features:
            return [0.0] * 10

        return features

    def _get_relationship_features(self, components: list[str], graph: nx.DiGraph) -> list[float]:
        """Extract relationship features from the graph.

        This method:
        - Calculates average degree of components
        - Calculates subgraph density
        - Calculates clustering coefficient
        - Calculates edge counts (incoming and outgoing)
        - Calculates internal to external edge ratio

        Args:
            components: List of component IDs
            graph: The graph containing the components

        Returns:
            List of numerical features related to component relationships
        """
        return [
            self._calculate_average_degree(components, graph),
            self._calculate_subgraph_density(components, graph),
            self._calculate_clustering_coefficient(components, graph),
            *self._calculate_edge_counts(components, graph),
            self._calculate_internal_external_ratio(components, graph),
        ]

    def _calculate_average_degree(self, components: list[str], graph: nx.DiGraph) -> float:
        """Calculate average degree of components."""
        valid_nodes = [node for node in components if node in graph]
        if not valid_nodes:
            return 0.0

        degree_sum = sum(graph.degree[node] for node in valid_nodes)
        return degree_sum / len(valid_nodes)

    def _calculate_subgraph_density(self, components: list[str], graph: nx.DiGraph) -> float:
        """Calculate density of the subgraph formed by components."""
        if len(components) <= 1:
            return 0.0

        subgraph = graph.subgraph([n for n in components if n in graph])
        return nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0.0

    def _calculate_clustering_coefficient(self, components: list[str], graph: nx.DiGraph) -> float:
        """Calculate average clustering coefficient."""
        valid_nodes = [node for node in components if node in graph]
        if not valid_nodes:
            return 0.0

        try:
            return nx.average_clustering(graph, nodes=valid_nodes)
        except (ZeroDivisionError, nx.NetworkXError):
            return 0.0

    def _calculate_edge_counts(self, components: list[str], graph: nx.DiGraph) -> list[float]:
        """Calculate incoming and outgoing edge counts."""
        incoming = 0
        outgoing = 0
        for node in components:
            if node in graph:
                incoming += len(list(graph.predecessors(node)))
                outgoing += len(list(graph.successors(node)))
        return [float(incoming), float(outgoing)]

    def _calculate_internal_external_ratio(self, components: list[str], graph: nx.DiGraph) -> float:
        """Calculate ratio of internal to external edges."""
        internal_edges = sum(1 for u, v in graph.edges() if u in components and v in components)

        incoming, outgoing = self._calculate_edge_counts(components, graph)
        external_edges = (incoming + outgoing) - 2 * internal_edges
        return internal_edges / max(external_edges, 1)

    def _get_text_features(self, components: list[str], graph: nx.DiGraph) -> list[float]:
        """Extract text-based features from component names and code.

        This method:
        - Counts pattern-specific keywords in component names and code
        - Calculates average name length

        Args:
            components: List of component IDs
            graph: The graph containing the components

        Returns:
            List of numerical features related to text content
        """
        features = []

        # Keywords related to different patterns
        pattern_keywords = {
            "Singleton": [
                "singleton",
                "instance",
                "getInstance",
                "private constructor",
            ],
            "Factory": ["factory", "create", "build", "producer"],
            "Observer": [
                "observer",
                "notify",
                "subscribe",
                "event",
                "listener",
            ],
            "MVC": ["model", "view", "controller", "presenter"],
            "Repository": ["repository", "dao", "store", "persistence"],
            "Service": ["service", "manager", "handler", "processor"],
            "Sidecar": ["sidecar", "helper", "agent", "proxy"],
            "Ambassador": [
                "ambassador",
                "proxy",
                "delegate",
                "representative",
            ],
            "Adapter": ["adapter", "convert", "transform", "wrapper"],
            "CircuitBreaker": ["circuit", "breaker", "fallback", "timeout"],
            "ServiceMesh": ["mesh", "istio", "linkerd", "envoy"],
            "ApiGateway": ["gateway", "router", "facade", "bf"],
        }

        # Count pattern-specific keywords in component names and code
        for _, keywords in pattern_keywords.items():
            keyword_count = 0
            for node in components:
                if node in graph:
                    # Check node ID for keywords
                    node_lower = node.lower()
                    keyword_count += sum(keyword in node_lower for keyword in keywords)

                    # Check node code (if available)
                    node_data = graph.nodes[node]
                    if "code" in node_data:
                        code = node_data["code"].lower()
                        keyword_count += sum(keyword in code for keyword in keywords)

            features.append(float(keyword_count))

        # Calculate average name length as a feature
        avg_name_length = sum(len(node) for node in components) / max(len(components), 1)
        features.append(avg_name_length)

        return features

    def _get_structural_features(self, components: list[str], graph: nx.DiGraph) -> list[float]:
        """Extract structural features from components in the graph.

        This method:
        - Calculates number of components
        - Calculates average node attributes (size and complexity)
        - Calculates size variance

        Args:
            components: List of component IDs
            graph: The graph containing the components

        Returns:
            List of numerical features related to structural properties
        """
        features = []

        # Number of components
        features.append(float(len(components)))

        # Average node attributes
        node_sizes = []
        node_complexities = []

        for node in components:
            if node in graph:
                node_data = graph.nodes[node]

                # Size metric (lines of code if available)
                size = 0
                if "loc" in node_data:
                    size = node_data["loc"]
                elif "code" in node_data and isinstance(node_data["code"], str):
                    size = len(node_data["code"].split("\n"))
                node_sizes.append(size)

                # Complexity metric (if available)
                complexity = node_data.get("complexity", 0)
                node_complexities.append(complexity)

        # Add average size and complexity
        features.append(float(sum(node_sizes)) / max(len(node_sizes), 1))
        features.append(float(sum(node_complexities)) / max(len(node_complexities), 1))

        # Add size variance as a feature
        if len(node_sizes) > 1:
            variance = sum(
                (x - (sum(node_sizes) / len(node_sizes))) ** 2 for x in node_sizes
            ) / len(node_sizes)
            features.append(variance)
        else:
            features.append(0.0)

        return features

    def train_model(self, pattern_type: str, training_data: list[tuple[list[float], int]]) -> None:
        """Train a model for a specific pattern type.

        This method:
        - Extracts features and labels from training data
        - Creates and trains a classifier model
        - Saves the trained model to disk

        Args:
            pattern_type: The type of pattern to train for
            training_data: List of (features, label) tuples where label is 0 or 1

        Raises:
            ValueError: If training data is invalid
            OSError: If model file cannot be written
            pickle.PickleError: If model serialization fails
        """
        if not training_data:
            logger.warning("No training data provided for %s", pattern_type)
            return

        try:
            # Extract features and labels
            feature_vectors = [item[0] for item in training_data]
            labels = [item[1] for item in training_data]

            # Create and train classifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)  # type: ignore[call-non-callable]
            model.fit(feature_vectors, labels)

            # Save the trained model
            self.models[pattern_type] = model
            model_path = self.model_directory / f"{pattern_type.lower()}_classifier.pkl"

            # Create directory if it doesn't exist
            os.makedirs(self.model_directory, exist_ok=True)

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            logger.info("Trained and saved model for %s", pattern_type)
        except (ValueError, OSError, pickle.PickleError) as e:
            logger.error("Failed to train model for %s: %s", pattern_type, e)
