"""
Recommendation generator compatibility module.

Re-exports PRValidationRecommendationGenerator from the actual services location.
"""

from __future__ import annotations

try:
    from services.pr_validation.scoring.recommendation_generator import (
        PRValidationRecommendationGenerator,
    )

    # Re-export all public components
    __all__ = ["PRValidationRecommendationGenerator"]

except ImportError:
    # Create placeholder if the real module is not available
    import logging

    logger = logging.getLogger(__name__)
    logger.warning("PRValidationRecommendationGenerator not available - using placeholder")

    class PRValidationRecommendationGenerator:
        """Placeholder recommendation generator when real module is not available."""

        def __init__(self, *args, **kwargs):
            pass

        def generate_recommendations(self, *args, **kwargs):
            return []

    __all__ = ["PRValidationRecommendationGenerator"]
