"""
Task scheduling and management for background task processing.
Factory pattern implementation for task processor management.
"""

from __future__ import annotations

from .task_execution_engine import BaseTaskProcessor, CentralityCalculationProcessor


class TaskProcessorFactory:
    """Factory for creating task processors."""

    def __init__(self) -> None:
        """Initialize the factory with available processors."""
        self._processors: dict[str, BaseTaskProcessor] = {
            "calculate_centrality": CentralityCalculationProcessor(),
        }

    def get_processor(self, task_type: str) -> BaseTaskProcessor | None:
        """Get a processor for the given task type."""
        return self._processors.get(task_type)

    def register_processor(self, task_type: str, processor: BaseTaskProcessor):
        """Register a new processor for a task type."""
        self._processors[task_type] = processor

    def unregister_processor(self, task_type: str) -> bool:
        """Unregister a processor for a task type."""
        if task_type in self._processors:
            del self._processors[task_type]
            return True
        return False

    def get_supported_task_types(self) -> list:
        """Get list of supported task types."""
        return list(self._processors.keys())

    def get_processor_info(self, task_type: str) -> dict:
        """Get information about a specific processor."""
        processor = self._processors.get(task_type)
        if processor:
            return {
                "task_type": task_type,
                "processor_class": processor.__class__.__name__,
                "processor_type": processor.get_processor_type(),
            }
        return {"error": f"No processor found for task type: {task_type}"}

    def list_all_processors(self) -> dict:
        """List all registered processors with their information."""
        return {
            task_type: self.get_processor_info(task_type) for task_type in self._processors.keys()
        }

    def is_task_type_supported(self, task_type: str) -> bool:
        """Check if a task type is supported."""
        return task_type in self._processors
