"""
Scoring module - compatibility layer.

This module provides backward compatibility for imports from app.scoring
by re-exporting the scoring functionality from app.services.pr_validation.scoring.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    # Re-export scoring components from services
    from services.pr_validation.scoring.recommendation_generator import (
        PRValidationRecommendationGenerator,
    )
    from services.pr_validation.scoring.risk_calculator import RiskCalculator

    logger.debug("Successfully imported scoring components from services")

    __all__ = [
        "RiskCalculator",
        "PRValidationRecommendationGenerator",
    ]

except ImportError as e:
    logger.warning(f"Could not import scoring components: {e}")

    # Create placeholder classes for graceful degradation
    class RiskCalculator:
        """Placeholder risk calculator when scoring module is not available."""

        def __init__(self, *args, **kwargs):
            logger.warning("Using placeholder RiskCalculator - scoring functionality disabled")

        def calculate_risk(self, *args, **kwargs):
            return {"overall_score": 0.0, "message": "Scoring unavailable"}

    class PRValidationRecommendationGenerator:
        """Placeholder recommendation generator when scoring module is not available."""

        def __init__(self, *args, **kwargs):
            logger.warning(
                "Using placeholder PRValidationRecommendationGenerator - recommendations disabled"
            )

        def generate_recommendations(self, *args, **kwargs):
            return []

    __all__ = [
        "RiskCalculator",
        "PRValidationRecommendationGenerator",
    ]
