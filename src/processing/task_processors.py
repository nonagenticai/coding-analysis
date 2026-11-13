"""
Task processors for different types of background tasks.
Modular implementation using separated components for better maintainability.
"""

from __future__ import annotations

import logging

from processing.codebase_analysis_processor import CodebaseAnalysisProcessor
from processing.multi_language_analysis_processor import (
    MultiLanguageAnalysisProcessor,
)

# Import modular components
from processing.task_execution_engine import (
    BaseTaskProcessor,
    CentralityCalculationProcessor,
)
from processing.task_scheduler import TaskProcessorFactory as BaseTaskProcessorFactory

logger = logging.getLogger(__name__)

# Re-export centrality calculation functions for backward compatibility
from processing.task_execution_engine import (
    calculate_degree_centrality,
    calculate_pagerank,
)


class TaskProcessorFactory(BaseTaskProcessorFactory):
    """Enhanced factory for creating task processors with all available processors."""

    def __init__(self) -> None:
        """Initialize the factory with all available processors."""
        super().__init__()

        # Register additional processors
        self._processors.update(
            {
                "analyze_codebase": CodebaseAnalysisProcessor(),
                "multi_language_analysis": MultiLanguageAnalysisProcessor(),
            }
        )

    def get_processor_categories(self) -> dict[str, list]:
        """Get processors organized by category."""
        return {
            "analysis": ["analyze_codebase", "multi_language_analysis"],
            "metrics": ["calculate_centrality"],
        }

    def get_processor_by_category(self, category: str) -> dict[str, BaseTaskProcessor]:
        """Get all processors in a specific category."""
        categories = self.get_processor_categories()
        if category not in categories:
            return {}

        return {
            task_type: self._processors[task_type]
            for task_type in categories[category]
            if task_type in self._processors
        }


# Global factory instance for backward compatibility
_global_factory = None


def get_task_processor_factory() -> TaskProcessorFactory:
    """Get the global task processor factory instance."""
    global _global_factory
    if _global_factory is None:
        _global_factory = TaskProcessorFactory()
    return _global_factory


def get_processor(task_type: str) -> BaseTaskProcessor | None:
    """Get a processor for the given task type (convenience function)."""
    factory = get_task_processor_factory()
    return factory.get_processor(task_type)


def get_supported_task_types() -> list:
    """Get list of all supported task types (convenience function)."""
    factory = get_task_processor_factory()
    return factory.get_supported_task_types()


# Maintain backward compatibility by exposing the main classes
__all__ = [
    "BaseTaskProcessor",
    "CentralityCalculationProcessor",
    "CodebaseAnalysisProcessor",
    "MultiLanguageAnalysisProcessor",
    "TaskProcessorFactory",
    "get_task_processor_factory",
    "get_processor",
    "get_supported_task_types",
    "calculate_degree_centrality",
    "calculate_pagerank",
]
