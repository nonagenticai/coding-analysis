"""Unit tests for processing/task_processors.py and processing/task_scheduler.py.

Covers:
- TaskProcessorFactory (task_scheduler.py)
- TaskProcessorFactory (task_processors.py - enhanced version)
- get_task_processor_factory, get_processor, get_supported_task_types
"""

from __future__ import annotations

from unittest.mock import MagicMock

from processing.task_scheduler import TaskProcessorFactory as BaseFactory

# ---------------------------------------------------------------------------
# Tests for task_scheduler.TaskProcessorFactory
# ---------------------------------------------------------------------------


class TestBaseTaskProcessorFactory:
    """Tests for the base TaskProcessorFactory in task_scheduler.py."""

    def test_init_has_centrality_processor(self):
        factory = BaseFactory()
        assert factory.is_task_type_supported("calculate_centrality")

    def test_get_processor_returns_processor(self):
        factory = BaseFactory()
        proc = factory.get_processor("calculate_centrality")
        assert proc is not None

    def test_get_processor_unknown_returns_none(self):
        factory = BaseFactory()
        proc = factory.get_processor("nonexistent_task")
        assert proc is None

    def test_register_processor(self):
        factory = BaseFactory()
        mock_proc = MagicMock()
        mock_proc.get_processor_type = MagicMock(return_value="custom")
        factory.register_processor("custom_task", mock_proc)
        assert factory.get_processor("custom_task") is mock_proc

    def test_unregister_processor_exists(self):
        factory = BaseFactory()
        mock_proc = MagicMock()
        factory.register_processor("temp", mock_proc)
        assert factory.unregister_processor("temp") is True
        assert factory.get_processor("temp") is None

    def test_unregister_processor_not_exists(self):
        factory = BaseFactory()
        assert factory.unregister_processor("nonexistent") is False

    def test_get_supported_task_types(self):
        factory = BaseFactory()
        types = factory.get_supported_task_types()
        assert isinstance(types, list)
        assert "calculate_centrality" in types

    def test_get_processor_info_existing(self):
        factory = BaseFactory()
        info = factory.get_processor_info("calculate_centrality")
        assert info["task_type"] == "calculate_centrality"
        assert "processor_class" in info

    def test_get_processor_info_nonexistent(self):
        factory = BaseFactory()
        info = factory.get_processor_info("nonexistent")
        assert "error" in info

    def test_list_all_processors(self):
        factory = BaseFactory()
        all_procs = factory.list_all_processors()
        assert isinstance(all_procs, dict)
        assert "calculate_centrality" in all_procs

    def test_is_task_type_supported_true(self):
        factory = BaseFactory()
        assert factory.is_task_type_supported("calculate_centrality") is True

    def test_is_task_type_supported_false(self):
        factory = BaseFactory()
        assert factory.is_task_type_supported("nonexistent") is False


# ---------------------------------------------------------------------------
# Tests for task_processors.TaskProcessorFactory (enhanced)
# ---------------------------------------------------------------------------


class TestEnhancedTaskProcessorFactory:
    """Tests for the enhanced TaskProcessorFactory in task_processors.py."""

    def test_init_has_additional_processors(self):
        from processing.task_processors import TaskProcessorFactory

        factory = TaskProcessorFactory()
        assert factory.is_task_type_supported("analyze_codebase")
        assert factory.is_task_type_supported("multi_language_analysis")
        assert factory.is_task_type_supported("calculate_centrality")

    def test_get_processor_categories(self):
        from processing.task_processors import TaskProcessorFactory

        factory = TaskProcessorFactory()
        cats = factory.get_processor_categories()
        assert "analysis" in cats
        assert "metrics" in cats
        assert "analyze_codebase" in cats["analysis"]

    def test_get_processor_by_category_analysis(self):
        from processing.task_processors import TaskProcessorFactory

        factory = TaskProcessorFactory()
        procs = factory.get_processor_by_category("analysis")
        assert "analyze_codebase" in procs
        assert "multi_language_analysis" in procs

    def test_get_processor_by_category_metrics(self):
        from processing.task_processors import TaskProcessorFactory

        factory = TaskProcessorFactory()
        procs = factory.get_processor_by_category("metrics")
        assert "calculate_centrality" in procs

    def test_get_processor_by_category_unknown(self):
        from processing.task_processors import TaskProcessorFactory

        factory = TaskProcessorFactory()
        procs = factory.get_processor_by_category("unknown")
        assert procs == {}


class TestTaskProcessorConvenienceFunctions:
    """Tests for convenience functions in task_processors.py."""

    def test_get_task_processor_factory(self):
        import processing.task_processors as tp

        tp._global_factory = None
        factory = tp.get_task_processor_factory()
        assert factory is not None
        tp._global_factory = None

    def test_get_task_processor_factory_singleton(self):
        import processing.task_processors as tp

        tp._global_factory = None
        f1 = tp.get_task_processor_factory()
        f2 = tp.get_task_processor_factory()
        assert f1 is f2
        tp._global_factory = None

    def test_get_processor_found(self):
        import processing.task_processors as tp

        tp._global_factory = None
        proc = tp.get_processor("calculate_centrality")
        assert proc is not None
        tp._global_factory = None

    def test_get_processor_not_found(self):
        import processing.task_processors as tp

        tp._global_factory = None
        proc = tp.get_processor("nonexistent")
        assert proc is None
        tp._global_factory = None

    def test_get_supported_task_types(self):
        import processing.task_processors as tp

        tp._global_factory = None
        types = tp.get_supported_task_types()
        assert "calculate_centrality" in types
        assert "analyze_codebase" in types
        tp._global_factory = None

    def test_all_exports(self):
        import processing.task_processors as tp

        assert "BaseTaskProcessor" in tp.__all__
        assert "TaskProcessorFactory" in tp.__all__
        assert "get_task_processor_factory" in tp.__all__
        assert "calculate_degree_centrality" in tp.__all__
        assert "calculate_pagerank" in tp.__all__
