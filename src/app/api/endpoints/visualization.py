"""Visualization endpoint aggregator.

Combines all visualization sub-routers (graph, chart, export, configuration)
into a single router mounted by main.py.
"""

from fastapi import APIRouter

from .visualization_modules import chart_router, export_router, graph_router
from .visualization_modules.configuration import router as config_router

router = APIRouter()
router.include_router(graph_router)
router.include_router(config_router)
router.include_router(chart_router)
router.include_router(export_router)
