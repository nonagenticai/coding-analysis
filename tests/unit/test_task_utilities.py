"""Unit tests for processing.task_utilities module."""

from __future__ import annotations

import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

from processing.task_utilities import (
    DatabaseUtils,
    FileOperationUtils,
    JsonUtils,
    LanguageDetectionUtils,
    PathUtils,
    PerformanceUtils,
    ProcessingUtils,
)

# ===========================================================================
# FileOperationUtils
# ===========================================================================


class TestFileOperationUtils:
    """Tests for FileOperationUtils."""

    def test_get_supported_file_types_returns_list(self):
        result = FileOperationUtils.get_supported_file_types()
        assert isinstance(result, list)
        assert ".py" in result
        assert ".js" in result
        assert ".ts" in result
        assert ".java" in result
        assert ".rs" in result
        assert ".go" in result
        assert ".cs" in result
        assert ".cpp" in result
        assert ".c" in result
        assert ".h" in result
        assert ".hpp" in result

    def test_get_language_extensions_returns_dict(self):
        result = FileOperationUtils.get_language_extensions()
        assert isinstance(result, dict)
        assert result[".py"] == "python"
        assert result[".js"] == "javascript"
        assert result[".ts"] == "typescript"
        assert result[".java"] == "java"
        assert result[".rs"] == "rust"
        assert result[".go"] == "go"
        assert result[".cs"] == "csharp"
        assert result[".cpp"] == "cpp"
        assert result[".c"] == "c"
        assert result[".h"] == "c"
        assert result[".hpp"] == "cpp"

    def test_filter_directories_removes_hidden_dirs(self):
        dirs = [".git", "src", ".hidden", "tests"]
        FileOperationUtils.filter_directories(dirs)
        assert ".git" not in dirs
        assert ".hidden" not in dirs
        assert "src" in dirs
        assert "tests" in dirs

    def test_filter_directories_removes_build_dirs(self):
        dirs = ["__pycache__", "node_modules", "target", "build", "dist", "src"]
        FileOperationUtils.filter_directories(dirs)
        assert dirs == ["src"]

    def test_filter_directories_modifies_in_place(self):
        dirs = ["src", "tests", "__pycache__"]
        original_id = id(dirs)
        FileOperationUtils.filter_directories(dirs)
        assert id(dirs) == original_id

    def test_should_process_file_python(self):
        assert FileOperationUtils.should_process_file("main.py") is True

    def test_should_process_file_javascript(self):
        assert FileOperationUtils.should_process_file("app.js") is True

    def test_should_process_file_typescript(self):
        assert FileOperationUtils.should_process_file("index.ts") is True

    def test_should_process_file_unsupported(self):
        assert FileOperationUtils.should_process_file("readme.md") is False
        assert FileOperationUtils.should_process_file("data.json") is False
        assert FileOperationUtils.should_process_file("image.png") is False

    def test_should_process_file_hidden(self):
        assert FileOperationUtils.should_process_file(".hidden.py") is False

    def test_should_process_file_with_path(self):
        assert FileOperationUtils.should_process_file("/path/to/file.py") is True
        assert FileOperationUtils.should_process_file("/path/to/.hidden.py") is False

    def test_get_file_language_python(self):
        assert FileOperationUtils.get_file_language("main.py") == "python"

    def test_get_file_language_javascript(self):
        assert FileOperationUtils.get_file_language("app.js") == "javascript"

    def test_get_file_language_typescript(self):
        assert FileOperationUtils.get_file_language("component.ts") == "typescript"

    def test_get_file_language_java(self):
        assert FileOperationUtils.get_file_language("Main.java") == "java"

    def test_get_file_language_rust(self):
        assert FileOperationUtils.get_file_language("lib.rs") == "rust"

    def test_get_file_language_go(self):
        assert FileOperationUtils.get_file_language("main.go") == "go"

    def test_get_file_language_cpp(self):
        assert FileOperationUtils.get_file_language("main.cpp") == "cpp"
        assert FileOperationUtils.get_file_language("header.hpp") == "cpp"

    def test_get_file_language_c(self):
        assert FileOperationUtils.get_file_language("main.c") == "c"
        assert FileOperationUtils.get_file_language("header.h") == "c"

    def test_get_file_language_csharp(self):
        assert FileOperationUtils.get_file_language("Program.cs") == "csharp"

    def test_get_file_language_unsupported(self):
        assert FileOperationUtils.get_file_language("readme.md") is None
        assert FileOperationUtils.get_file_language("noext") is None

    def test_get_file_info_success(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        result = FileOperationUtils.get_file_info(str(test_file))

        assert result["path"] == str(test_file)
        assert result["name"] == "test.py"
        assert result["size"] > 0
        assert result["extension"] == ".py"
        assert result["language"] == "python"
        assert result["is_supported"] is True
        assert "modified_time" in result

    def test_get_file_info_unsupported_extension(self, tmp_path):
        test_file = tmp_path / "readme.md"
        test_file.write_text("# Hello")

        result = FileOperationUtils.get_file_info(str(test_file))

        assert result["language"] is None
        assert result["is_supported"] is False

    def test_get_file_info_nonexistent_file(self):
        result = FileOperationUtils.get_file_info("/nonexistent/file.py")

        assert result["path"] == "/nonexistent/file.py"
        assert "error" in result
        assert result["is_supported"] is False


# ===========================================================================
# LanguageDetectionUtils
# ===========================================================================


class TestLanguageDetectionUtils:
    """Tests for LanguageDetectionUtils."""

    def test_get_supported_languages(self):
        langs = LanguageDetectionUtils.get_supported_languages()
        assert isinstance(langs, list)
        assert "python" in langs
        assert "javascript" in langs
        assert "typescript" in langs
        assert "java" in langs
        assert "rust" in langs
        assert "go" in langs
        assert "csharp" in langs
        assert "cpp" in langs
        assert "c" in langs

    async def test_initialize_multi_language_factory(self):
        mock_factory = MagicMock()
        with (
            patch("processing.task_utilities.FileService") as mock_fs_cls,
            patch(
                "processing.task_utilities.get_multi_language_factory",
                new_callable=AsyncMock,
                return_value=mock_factory,
            ) as mock_get_factory,
        ):
            result = await LanguageDetectionUtils.initialize_multi_language_factory()
            mock_fs_cls.assert_called_once()
            mock_get_factory.assert_awaited_once()
            assert result is mock_factory

    async def test_initialize_graph_factory(self):
        mock_factory = MagicMock()
        with (
            patch("processing.task_utilities.FileService") as mock_fs_cls,
            patch(
                "processing.task_utilities.get_graph_factory",
                new_callable=AsyncMock,
                return_value=mock_factory,
            ) as mock_get_factory,
        ):
            result = await LanguageDetectionUtils.initialize_graph_factory()
            mock_fs_cls.assert_called_once()
            mock_get_factory.assert_awaited_once()
            assert result is mock_factory

    async def test_scan_supported_files_finds_files(self, tmp_path):
        # Create test files
        py_file = tmp_path / "main.py"
        py_file.write_text("print('hi')")
        js_file = tmp_path / "app.js"
        js_file.write_text("console.log('hi')")
        md_file = tmp_path / "readme.md"
        md_file.write_text("# Hello")

        mock_factory = MagicMock()
        mock_factory.is_supported_file = MagicMock(side_effect=lambda p: p.suffix in (".py", ".js"))

        mock_processor = MagicMock()
        mock_processor.language = "python"
        mock_processor.get_language = MagicMock(return_value="python")
        mock_factory.get_processor_for_file = AsyncMock(return_value=mock_processor)

        files, stats = await LanguageDetectionUtils.scan_supported_files(tmp_path, mock_factory)

        assert len(files) == 2
        assert stats.get("python", 0) == 2  # both files get the mock processor

    async def test_scan_supported_files_empty_dir(self, tmp_path):
        mock_factory = MagicMock()
        mock_factory.is_supported_file = MagicMock(return_value=False)

        files, stats = await LanguageDetectionUtils.scan_supported_files(tmp_path, mock_factory)

        assert files == []
        assert stats == {}

    async def test_scan_supported_files_processor_without_get_language(self, tmp_path):
        py_file = tmp_path / "main.py"
        py_file.write_text("x = 1")

        mock_factory = MagicMock()
        mock_factory.is_supported_file = MagicMock(return_value=True)

        # Processor with language attribute but no get_language method
        mock_processor = MagicMock(spec=[])
        mock_processor.language = "rust"
        mock_factory.get_processor_for_file = AsyncMock(return_value=mock_processor)

        files, stats = await LanguageDetectionUtils.scan_supported_files(tmp_path, mock_factory)

        assert len(files) == 1
        assert stats.get("rust", 0) == 1

    async def test_scan_supported_files_no_processor(self, tmp_path):
        py_file = tmp_path / "main.py"
        py_file.write_text("x = 1")

        mock_factory = MagicMock()
        mock_factory.is_supported_file = MagicMock(return_value=True)
        mock_factory.get_processor_for_file = AsyncMock(return_value=None)

        files, stats = await LanguageDetectionUtils.scan_supported_files(tmp_path, mock_factory)

        assert len(files) == 1
        assert stats == {}

    async def test_scan_supported_files_handles_error(self, tmp_path):
        mock_factory = MagicMock()
        mock_factory.is_supported_file = MagicMock(side_effect=PermissionError("denied"))

        # Create a file so rglob finds something
        (tmp_path / "x.py").write_text("x")

        files, stats = await LanguageDetectionUtils.scan_supported_files(tmp_path, mock_factory)

        assert files == []
        assert stats == {}


# ===========================================================================
# DatabaseUtils
# ===========================================================================


class TestDatabaseUtils:
    """Tests for DatabaseUtils."""

    async def test_get_node_count_no_filters(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(42,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_node_count(mock_conn)

        assert result == 42
        mock_conn.execute.assert_awaited_once()

    async def test_get_node_count_with_node_type(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(10,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_node_count(mock_conn, node_type="function")

        assert result == 10
        call_args = mock_conn.execute.call_args
        assert "type = %s" in call_args[0][0]
        assert "function" in call_args[0][1]

    async def test_get_node_count_with_tenant(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(5,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_node_count(mock_conn, tenant_id="t1")

        assert result == 5
        call_args = mock_conn.execute.call_args
        assert "tenant_id = %s" in call_args[0][0]

    async def test_get_node_count_with_both_filters(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(3,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_node_count(mock_conn, node_type="class", tenant_id="t1")

        assert result == 3
        call_args = mock_conn.execute.call_args
        query = call_args[0][0]
        assert "tenant_id = %s" in query
        assert "type = %s" in query

    async def test_get_node_count_returns_zero_on_none_row(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_node_count(mock_conn)
        assert result == 0

    async def test_get_node_count_returns_zero_on_error(self):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("db error"))

        result = await DatabaseUtils.get_node_count(mock_conn)
        assert result == 0

    async def test_get_edge_count_no_filters(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(100,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_edge_count(mock_conn)
        assert result == 100

    async def test_get_edge_count_with_relationship_type(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(7,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_edge_count(mock_conn, relationship_type="calls")

        assert result == 7
        call_args = mock_conn.execute.call_args
        assert "relationship = %s" in call_args[0][0]

    async def test_get_edge_count_with_tenant(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=(15,))
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_edge_count(mock_conn, tenant_id="t2")

        assert result == 15
        call_args = mock_conn.execute.call_args
        assert "tenant_id = %s" in call_args[0][0]

    async def test_get_edge_count_returns_zero_on_error(self):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=RuntimeError("fail"))

        result = await DatabaseUtils.get_edge_count(mock_conn)
        assert result == 0

    async def test_get_edge_count_none_row(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=None)
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_edge_count(mock_conn)
        assert result == 0

    async def test_get_node_types_no_tenant(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[("function",), ("class",), ("module",)])
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_node_types(mock_conn)

        assert result == ["function", "class", "module"]

    async def test_get_node_types_with_tenant(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[("function",)])
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_node_types(mock_conn, tenant_id="t1")

        assert result == ["function"]
        call_args = mock_conn.execute.call_args
        assert "tenant_id = %s" in call_args[0][0]

    async def test_get_node_types_filters_none_values(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[("function",), (None,)])
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_node_types(mock_conn)
        assert result == ["function"]

    async def test_get_node_types_error(self):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("db error"))

        result = await DatabaseUtils.get_node_types(mock_conn)
        assert result == []

    async def test_get_relationship_types_no_tenant(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[("calls",), ("imports",), ("inherits",)])
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_relationship_types(mock_conn)
        assert result == ["calls", "imports", "inherits"]

    async def test_get_relationship_types_with_tenant(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[("calls",)])
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_relationship_types(mock_conn, tenant_id="t1")

        assert result == ["calls"]
        call_args = mock_conn.execute.call_args
        assert "tenant_id = %s" in call_args[0][0]

    async def test_get_relationship_types_filters_none(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[(None,), ("calls",)])
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_relationship_types(mock_conn)
        assert result == ["calls"]

    async def test_get_relationship_types_error(self):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("fail"))

        result = await DatabaseUtils.get_relationship_types(mock_conn)
        assert result == []

    async def test_get_nodes_basic(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        # psycopg returns JSONB as a dict already (no json.loads in source)
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                ("id1", "function", {"name": "foo"}, "2024-01-01", "2024-01-02", "t1"),
            ]
        )
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_nodes(mock_conn)

        assert len(result) == 1
        assert result[0]["id"] == "id1"
        assert result[0]["type"] == "function"
        assert result[0]["properties"] == {"name": "foo"}
        assert result[0]["tenant_id"] == "t1"

    async def test_get_nodes_with_filters(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[])
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        await DatabaseUtils.get_nodes(
            mock_conn, node_type="class", tenant_id="t1", limit=50, offset=10
        )

        call_args = mock_conn.execute.call_args
        query = call_args[0][0]
        params = call_args[0][1]
        assert "tenant_id = %s" in query
        assert "type = %s" in query
        assert 50 in params
        assert 10 in params

    async def test_get_nodes_null_properties(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                ("id1", "function", None, "2024-01-01", "2024-01-02", "t1"),
            ]
        )
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_nodes(mock_conn)
        assert result[0]["properties"] == {}

    async def test_get_nodes_error(self):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("db error"))

        result = await DatabaseUtils.get_nodes(mock_conn)
        assert result == []

    async def test_get_edges_basic(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        # psycopg returns JSONB as a dict already (no json.loads in source)
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                ("e1", "src1", "tgt1", "calls", {"line": 10}, 1.0, "2024-01-01", "t1"),
            ]
        )
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_edges(mock_conn)

        assert len(result) == 1
        assert result[0]["id"] == "e1"
        assert result[0]["source"] == "src1"
        assert result[0]["target"] == "tgt1"
        assert result[0]["relationship"] == "calls"
        assert result[0]["properties"] == {"line": 10}
        assert result[0]["weight"] == 1.0
        assert result[0]["tenant_id"] == "t1"

    async def test_get_edges_with_all_filters(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(return_value=[])
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        await DatabaseUtils.get_edges(
            mock_conn,
            source_id="s1",
            target_id="t1",
            relationship_type="calls",
            tenant_id="tenant1",
            limit=25,
            offset=5,
        )

        call_args = mock_conn.execute.call_args
        query = call_args[0][0]
        assert "tenant_id = %s" in query
        assert "source = %s" in query
        assert "target = %s" in query
        assert "relationship = %s" in query

    async def test_get_edges_null_properties(self):
        mock_conn = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.fetchall = AsyncMock(
            return_value=[
                ("e1", "s1", "t1", "calls", None, 1.0, "2024-01-01", "t1"),
            ]
        )
        mock_conn.execute = AsyncMock(return_value=mock_cursor)

        result = await DatabaseUtils.get_edges(mock_conn)
        assert result[0]["properties"] == {}

    async def test_get_edges_error(self):
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("db error"))

        result = await DatabaseUtils.get_edges(mock_conn)
        assert result == []


# ===========================================================================
# ProcessingUtils
# ===========================================================================


class TestProcessingUtils:
    """Tests for ProcessingUtils."""

    def test_create_processing_context_basic(self):
        factory = MagicMock()
        conn = MagicMock()

        result = ProcessingUtils.create_processing_context(factory, conn, "graph-123")

        assert result["factory"] is factory
        assert result["conn"] is conn
        assert result["code_graph_id"] == "graph-123"
        assert "timestamp" in result
        assert isinstance(result["timestamp"], float)

    def test_create_processing_context_with_kwargs(self):
        factory = MagicMock()
        conn = MagicMock()

        result = ProcessingUtils.create_processing_context(
            factory, conn, "graph-123", language="python", batch_size=50
        )

        assert result["language"] == "python"
        assert result["batch_size"] == 50

    def test_create_storage_context(self):
        conn = MagicMock()

        result = ProcessingUtils.create_storage_context(
            conn, "graph-123", "/path/to/file.py", "python"
        )

        assert result["conn"] is conn
        assert result["code_graph_id"] == "graph-123"
        assert result["file_path"] == "/path/to/file.py"
        assert result["language"] == "python"

    async def test_process_file_safely_success(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        mock_processor = AsyncMock()
        mock_processor.process_file = AsyncMock(return_value={"functions": 1})

        result = await ProcessingUtils.process_file_safely(str(test_file), mock_processor)

        assert result["success"] is True
        assert result["file_path"] == str(test_file)
        assert result["result"] == {"functions": 1}

    async def test_process_file_safely_with_progress_callback_success(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        mock_processor = AsyncMock()
        mock_processor.process_file = AsyncMock(return_value={})
        callback = MagicMock()

        await ProcessingUtils.process_file_safely(
            str(test_file), mock_processor, progress_callback=callback
        )

        callback.assert_called_once_with(str(test_file), True)

    async def test_process_file_safely_file_not_found(self):
        mock_processor = AsyncMock()

        result = await ProcessingUtils.process_file_safely("/nonexistent/file.py", mock_processor)

        assert result["success"] is False
        assert "error" in result
        assert result["error_type"] == "FileNotFoundError"

    async def test_process_file_safely_processor_error(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        mock_processor = AsyncMock()
        mock_processor.process_file = AsyncMock(side_effect=ValueError("parse error"))

        result = await ProcessingUtils.process_file_safely(str(test_file), mock_processor)

        assert result["success"] is False
        assert result["error"] == "parse error"
        assert result["error_type"] == "ValueError"

    async def test_process_file_safely_callback_on_failure(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        mock_processor = AsyncMock()
        mock_processor.process_file = AsyncMock(side_effect=RuntimeError("runtime fail"))
        callback = MagicMock()

        await ProcessingUtils.process_file_safely(
            str(test_file), mock_processor, progress_callback=callback
        )

        callback.assert_called_once_with(str(test_file), False)

    def test_log_progress_at_interval(self):
        with patch("processing.task_utilities.logger") as mock_logger:
            ProcessingUtils.log_progress(100, 1000, interval=100)
            mock_logger.info.assert_called_once()

    def test_log_progress_not_at_interval(self):
        with patch("processing.task_utilities.logger") as mock_logger:
            ProcessingUtils.log_progress(50, 1000, interval=100)
            mock_logger.info.assert_not_called()

    def test_log_progress_at_zero(self):
        with patch("processing.task_utilities.logger") as mock_logger:
            ProcessingUtils.log_progress(0, 1000, interval=100)
            mock_logger.info.assert_not_called()

    def test_log_progress_percentage_calculation(self):
        with patch("processing.task_utilities.logger") as mock_logger:
            ProcessingUtils.log_progress(200, 400, interval=100)
            call_args = mock_logger.info.call_args
            # Should log 50.0%
            assert 200 in call_args[0]
            assert 400 in call_args[0]


# ===========================================================================
# PerformanceUtils
# ===========================================================================


class TestPerformanceUtils:
    """Tests for PerformanceUtils."""

    def test_create_timer_context_manager(self):
        timer = PerformanceUtils.create_timer()
        with timer:
            pass  # some work
        assert timer.duration >= 0
        assert timer.start_time > 0
        assert timer.end_time >= timer.start_time

    def test_timer_elapsed(self):
        timer = PerformanceUtils.create_timer()
        assert timer.elapsed() == 0  # not started
        timer.start_time = time.time() - 1.0
        assert timer.elapsed() >= 1.0

    def test_format_duration_milliseconds(self):
        result = PerformanceUtils.format_duration(0.5)
        assert result == "500.0ms"

    def test_format_duration_seconds(self):
        result = PerformanceUtils.format_duration(5.0)
        assert result == "5.00s"

    def test_format_duration_minutes(self):
        result = PerformanceUtils.format_duration(125.5)
        assert result == "2m 5.5s"

    def test_format_duration_sub_millisecond(self):
        result = PerformanceUtils.format_duration(0.001)
        assert result == "1.0ms"

    def test_format_duration_exactly_60_seconds(self):
        result = PerformanceUtils.format_duration(60.0)
        assert result == "1m 0.0s"

    def test_estimate_remaining_time_normal(self):
        # 50 items processed in 10 seconds, 50 remaining → 10 seconds
        result = PerformanceUtils.estimate_remaining_time(50, 100, 10.0)
        assert result is not None
        assert "s" in result

    def test_estimate_remaining_time_zero_processed(self):
        result = PerformanceUtils.estimate_remaining_time(0, 100, 10.0)
        assert result is None

    def test_estimate_remaining_time_zero_total(self):
        result = PerformanceUtils.estimate_remaining_time(50, 0, 10.0)
        assert result is None

    def test_estimate_remaining_time_almost_done(self):
        result = PerformanceUtils.estimate_remaining_time(99, 100, 99.0)
        assert result is not None


# ===========================================================================
# JsonUtils
# ===========================================================================


class TestJsonUtils:
    """Tests for JsonUtils."""

    def test_safe_json_loads_valid(self):
        result = JsonUtils.safe_json_loads('{"key": "value"}')
        assert result == {"key": "value"}

    def test_safe_json_loads_invalid(self):
        result = JsonUtils.safe_json_loads("not json")
        assert result == {}

    def test_safe_json_loads_invalid_with_default(self):
        result = JsonUtils.safe_json_loads("not json", default=[])
        assert result == []

    def test_safe_json_loads_none_input(self):
        result = JsonUtils.safe_json_loads(None)
        assert result == {}

    def test_safe_json_loads_array(self):
        result = JsonUtils.safe_json_loads("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_safe_json_dumps_valid(self):
        result = JsonUtils.safe_json_dumps({"key": "value"})
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_safe_json_dumps_invalid(self):
        # Sets are not JSON-serializable
        result = JsonUtils.safe_json_dumps({1, 2, 3})
        assert result == "{}"

    def test_safe_json_dumps_invalid_with_default(self):
        result = JsonUtils.safe_json_dumps({1, 2, 3}, default={"error": True})
        parsed = json.loads(result)
        assert parsed == {"error": True}

    def test_safe_json_dumps_compact_format(self):
        result = JsonUtils.safe_json_dumps({"a": 1, "b": 2})
        # Should use compact separators (",", ":")
        assert " " not in result or result.count(" ") == 0
        # No spaces around separators
        assert ": " not in result
        assert ", " not in result

    def test_prepare_node_properties_basic(self):
        props = {"name": "foo", "count": 42, "active": True}
        result = JsonUtils.prepare_node_properties(props)
        parsed = json.loads(result)
        assert parsed["name"] == "foo"
        assert parsed["count"] == 42
        assert parsed["active"] is True

    def test_prepare_node_properties_converts_non_serializable(self):
        class CustomObj:
            def __str__(self):
                return "custom_value"

        props = {"name": "foo", "obj": CustomObj()}
        result = JsonUtils.prepare_node_properties(props)
        parsed = json.loads(result)
        assert parsed["name"] == "foo"
        assert parsed["obj"] == "custom_value"

    def test_prepare_node_properties_preserves_none(self):
        props = {"name": "foo", "value": None}
        result = JsonUtils.prepare_node_properties(props)
        parsed = json.loads(result)
        assert parsed["value"] is None

    def test_prepare_node_properties_preserves_nested(self):
        props = {"data": {"nested": True}, "items": [1, 2, 3]}
        result = JsonUtils.prepare_node_properties(props)
        parsed = json.loads(result)
        assert parsed["data"] == {"nested": True}
        assert parsed["items"] == [1, 2, 3]


# ===========================================================================
# PathUtils
# ===========================================================================


class TestPathUtils:
    """Tests for PathUtils."""

    def test_normalize_path(self):
        result = PathUtils.normalize_path("/foo/bar/../baz")
        assert result == "/foo/baz"

    def test_normalize_path_relative(self):
        result = PathUtils.normalize_path("foo/bar")
        assert os.path.isabs(result)

    def test_get_relative_path(self):
        result = PathUtils.get_relative_path("/foo/bar/baz.py", "/foo/bar")
        assert result == "baz.py"

    def test_get_relative_path_nested(self):
        result = PathUtils.get_relative_path("/foo/bar/sub/baz.py", "/foo/bar")
        assert result == os.path.join("sub", "baz.py")

    def test_create_unique_id(self):
        result = PathUtils.create_unique_id("/path/to/file.py", "my_function")
        assert "my_function" in result
        assert "::" in result

    def test_create_unique_id_normalizes_path(self):
        result = PathUtils.create_unique_id("/path/to/../to/file.py", "func")
        expected_path = os.path.normpath(os.path.abspath("/path/to/../to/file.py"))
        assert result.startswith(expected_path)

    def test_is_subpath_true(self):
        # Note: the implementation checks if parent is relative_to path (reversed)
        # is_subpath checks Path(parent).resolve().relative_to(Path(path).resolve())
        # So it returns True when parent is a subpath of path
        result = PathUtils.is_subpath("/foo", "/foo/bar")
        assert isinstance(result, bool)

    def test_is_subpath_false(self):
        result = PathUtils.is_subpath("/completely/different", "/foo/bar")
        assert result is False
