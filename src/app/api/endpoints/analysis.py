"""API analysis module."""

from __future__ import annotations

"""
V2 Unified Analysis Endpoint

Clean, modern analysis API that consolidates all analysis functionality.
Provides a single, consistent interface for all types of code analysis.

This is a refactored version of the original analysis.py file, split into
modular components for better maintainability.
"""

import logging

from fastapi import APIRouter

from .analysis_modules.analysis_endpoints import router as analysis_router
from .analysis_modules.batch_analysis import router as batch_router
from .analysis_modules.capabilities import router as capabilities_router
from .analysis_modules.multi_language_analysis import router as multi_language_router
from .analysis_modules.symbols import router as symbols_router

logger = logging.getLogger(__name__)

# Create the main router for analysis endpoints
router = APIRouter()

# Include all the modular routers (tags are set at main API level as "V2 🔬 Unified Analysis")
router.include_router(analysis_router)
router.include_router(batch_router)
router.include_router(capabilities_router)
router.include_router(multi_language_router)
router.include_router(symbols_router)

# Export the router for use in the main API
__all__ = ["router"]
