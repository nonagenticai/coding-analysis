"""Coverage boost tests for ComponentExtractor.

Focuses on:
- __init__: worker count, parallel processor initialization
- _get_default_worker_count: env var override, CPU count fallback
- _initialize_extraction: graph info
- _setup_progress_callback: combined callback
- _run_detection_strategies: folder, build_file, annotation strategies
- _estimate_graph_size
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from processing.architecture.component_extractor import ComponentExtractor


class TestComponentExtractorInit:
    def test_default_init(self):
        extractor = ComponentExtractor()
        assert extractor.max_workers >= 1
        assert extractor.use_processes is False
        assert extractor.min_parallel_threshold == 1000
        assert extractor.progress_callback is None
        assert extractor.parallel_processor is not None

    def test_custom_workers(self):
        extractor = ComponentExtractor(max_workers=4)
        assert extractor.max_workers == 4

    def test_use_processes(self):
        extractor = ComponentExtractor(use_processes=True)
        assert extractor.use_processes is True

    def test_custom_threshold(self):
        extractor = ComponentExtractor(min_parallel_threshold=500)
        assert extractor.min_parallel_threshold == 500

    def test_with_progress_callback(self):
        cb = MagicMock()
        extractor = ComponentExtractor(progress_callback=cb)
        assert extractor.progress_callback is cb


class TestGetDefaultWorkerCount:
    def test_env_var_override(self):
        with patch.dict(os.environ, {"ARCH_PROCESS_POOL_SIZE": "8"}):
            extractor = ComponentExtractor()
            count = extractor._get_default_worker_count()
            assert count == 8

    def test_env_var_invalid_fallback(self):
        with patch.dict(os.environ, {"ARCH_PROCESS_POOL_SIZE": "not_a_number"}):
            extractor = ComponentExtractor()
            count = extractor._get_default_worker_count()
            assert count >= 1

    def test_env_var_negative_clamps_to_1(self):
        with patch.dict(os.environ, {"ARCH_PROCESS_POOL_SIZE": "-5"}):
            extractor = ComponentExtractor()
            count = extractor._get_default_worker_count()
            assert count >= 1

    def test_cpu_count_based(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove ARCH_PROCESS_POOL_SIZE if present
            os.environ.pop("ARCH_PROCESS_POOL_SIZE", None)
            extractor = ComponentExtractor()
            count = extractor._get_default_worker_count()
            assert count >= 1


class TestInitializeExtraction:
    def test_initialize_extraction_returns_info(self):
        extractor = ComponentExtractor()
        mock_graph = MagicMock()
        mock_graph.nodes.return_value = ["a", "b", "c"]
        with patch.object(extractor, "_estimate_graph_size", return_value=1.5):
            info = extractor._initialize_extraction(mock_graph, None)
        assert info["total_nodes"] == 3
        assert info["graph_size_mb"] == 1.5

    def test_initialize_extraction_with_progress_reporter(self):
        extractor = ComponentExtractor()
        mock_graph = MagicMock()
        mock_graph.nodes.return_value = ["a", "b"]
        mock_reporter = MagicMock()
        mock_reporter.callback = None
        with patch.object(extractor, "_estimate_graph_size", return_value=0.5):
            extractor._initialize_extraction(mock_graph, mock_reporter)
        assert mock_reporter.total_steps == 5


class TestSetupProgressCallback:
    def test_combined_callback(self):
        progress_log = []
        extractor = ComponentExtractor(
            progress_callback=lambda c, t, m: progress_log.append(("ext", c, t))
        )
        mock_reporter = MagicMock()
        original_calls = []
        mock_reporter.callback = lambda c, t, m: original_calls.append(("orig", c, t))
        extractor._setup_progress_callback(mock_reporter)
        # Call the combined callback
        mock_reporter.callback(1, 5, {})
        assert len(original_calls) == 1
        assert len(progress_log) == 1

    def test_no_original_callback(self):
        progress_log = []
        extractor = ComponentExtractor(progress_callback=lambda c, t, m: progress_log.append(c))
        mock_reporter = MagicMock()
        mock_reporter.callback = None
        extractor._setup_progress_callback(mock_reporter)
        mock_reporter.callback(1, 5, {})
        assert len(progress_log) == 1


class TestRunDetectionStrategies:
    def test_folder_strategy(self):
        extractor = ComponentExtractor()
        mock_graph = MagicMock()
        mock_config = MagicMock()
        mock_config.detect_by = ["folder"]
        mock_config.folder_roots = ["src/"]
        with patch.object(extractor, "_extract_by_folder", return_value=[MagicMock()]):
            components, strategies = extractor._run_detection_strategies(
                mock_graph, mock_config, None
            )
        assert "folder" in strategies
        assert len(components) == 1

    def test_build_file_strategy(self):
        extractor = ComponentExtractor()
        mock_graph = MagicMock()
        mock_config = MagicMock()
        mock_config.detect_by = ["build_file"]
        mock_config.build_files = ["pom.xml"]
        with patch.object(extractor, "_extract_by_build_files", return_value=[]):
            components, strategies = extractor._run_detection_strategies(
                mock_graph, mock_config, None
            )
        assert "build_file" in strategies

    def test_annotation_strategy(self):
        extractor = ComponentExtractor()
        mock_graph = MagicMock()
        mock_config = MagicMock()
        mock_config.detect_by = ["annotation"]
        with patch.object(extractor, "_run_annotation_strategy", return_value=[]):
            components, strategies = extractor._run_detection_strategies(
                mock_graph, mock_config, None
            )
        assert "annotation" in strategies

    def test_multiple_strategies(self):
        extractor = ComponentExtractor()
        mock_graph = MagicMock()
        mock_config = MagicMock()
        mock_config.detect_by = ["folder", "build_file"]
        mock_config.folder_roots = ["src/"]
        mock_config.build_files = ["pom.xml"]
        with patch.object(extractor, "_extract_by_folder", return_value=[MagicMock()]):
            with patch.object(extractor, "_extract_by_build_files", return_value=[MagicMock()]):
                components, strategies = extractor._run_detection_strategies(
                    mock_graph, mock_config, None
                )
        assert "folder" in strategies
        assert "build_file" in strategies
        assert len(components) == 2

    def test_empty_detect_by(self):
        extractor = ComponentExtractor()
        mock_graph = MagicMock()
        mock_config = MagicMock()
        mock_config.detect_by = []
        components, strategies = extractor._run_detection_strategies(mock_graph, mock_config, None)
        assert len(strategies) == 0
        assert len(components) == 0
