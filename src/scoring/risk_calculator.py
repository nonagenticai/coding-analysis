"""
Risk calculator compatibility module.

Re-exports RiskCalculator from the actual services location.
"""

from __future__ import annotations

try:
    from services.pr_validation.scoring.risk_calculator import RiskCalculator

    # Re-export all public components
    __all__ = ["RiskCalculator"]

except ImportError:
    # Create placeholder if the real module is not available
    import logging

    logger = logging.getLogger(__name__)
    logger.warning("RiskCalculator not available - using placeholder")

    class RiskCalculator:
        """Placeholder risk calculator when real module is not available."""

        def __init__(self, *args, **kwargs):
            pass

        def calculate_risk(self, *args, **kwargs):
            return {"overall_score": 0.0, "message": "Risk calculation unavailable"}

        def calculate_overall_risk(self, *args, **kwargs):
            return {"overall_score": 0.0, "message": "Risk calculation unavailable"}

    __all__ = ["RiskCalculator"]
