"""Round 4 coverage tests for analysis_endpoints, analysis_utils, batch_analysis,
bug_hunter, complexity, architecture, contract_analysis endpoints, and models."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


@contextmanager
def _patch_runner(kind: str) -> Iterator[AsyncMock]:
    """Replace one entry of THE /analyze dispatch table with an AsyncMock."""
    from app.api.endpoints.analysis_modules.core_analysis import ANALYSIS_RUNNERS
    from app.api.models.analysis_api_models import PydanticAIAnalysisType

    mock = AsyncMock()
    with patch.dict(ANALYSIS_RUNNERS, {PydanticAIAnalysisType[kind]: mock}):
        yield mock


# ---------------------------------------------------------------------------
# Tests for analysis_endpoints.py (lines 211-756)
# ---------------------------------------------------------------------------
class TestAnalysisEndpointsCoverage:
    """Cover uncovered lines in analysis_endpoints.py."""

    @pytest.mark.asyncio
    async def test_analyze_standalone_code_basic_type(self):
        """Cover line 742-743 - basic analysis type routing."""
        from app.api.endpoints.analysis_modules.analysis_endpoints import (
            analyze_standalone_code,
        )

        with _patch_runner("BASIC") as mock_basic:
            mock_result = MagicMock()
            mock_basic.return_value = mock_result

            from app.api.models.analysis_api_models import (
                PydanticAIAnalysisType as AnalysisType,
            )

            request = MagicMock()
            request.analysis_type = AnalysisType.BASIC
            request.code = "x = 1"
            request.language = "python"

            await analyze_standalone_code(request, "aid-1", time.time())
            mock_basic.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_standalone_code_ast_type(self):
        """Cover line 746-747 - AST analysis type routing."""
        from app.api.endpoints.analysis_modules.analysis_endpoints import (
            analyze_standalone_code,
        )

        with _patch_runner("AST") as mock_ast:
            mock_ast.return_value = MagicMock()

            from app.api.models.analysis_api_models import (
                PydanticAIAnalysisType as AnalysisType,
            )

            request = MagicMock()
            request.analysis_type = AnalysisType.AST
            await analyze_standalone_code(request, "aid-2", time.time())
            mock_ast.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_standalone_code_deep_type(self):
        """Cover line 748-749 - deep analysis type routing."""
        from app.api.endpoints.analysis_modules.analysis_endpoints import (
            analyze_standalone_code,
        )

        with _patch_runner("DEEP") as mock_deep:
            mock_deep.return_value = MagicMock()

            from app.api.models.analysis_api_models import (
                PydanticAIAnalysisType as AnalysisType,
            )

            request = MagicMock()
            request.analysis_type = AnalysisType.DEEP
            await analyze_standalone_code(request, "aid-3", time.time())
            mock_deep.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_standalone_code_security_type(self):
        """Cover line 750-751 - security analysis type routing."""
        from app.api.endpoints.analysis_modules.analysis_endpoints import (
            analyze_standalone_code,
        )

        with _patch_runner("SECURITY") as mock_sec:
            mock_sec.return_value = MagicMock()

            from app.api.models.analysis_api_models import (
                PydanticAIAnalysisType as AnalysisType,
            )

            request = MagicMock()
            request.analysis_type = AnalysisType.SECURITY
            await analyze_standalone_code(request, "aid-4", time.time())
            mock_sec.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_standalone_code_performance_type(self):
        """Cover line 752-753 - performance analysis type routing."""
        from app.api.endpoints.analysis_modules.analysis_endpoints import (
            analyze_standalone_code,
        )

        with _patch_runner("PERFORMANCE") as mock_perf:
            mock_perf.return_value = MagicMock()

            from app.api.models.analysis_api_models import (
                PydanticAIAnalysisType as AnalysisType,
            )

            request = MagicMock()
            request.analysis_type = AnalysisType.PERFORMANCE
            await analyze_standalone_code(request, "aid-5", time.time())
            mock_perf.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_standalone_code_default_type(self):
        """Cover lines 754-756 - unknown type defaults to enhanced."""
        from app.api.endpoints.analysis_modules.analysis_endpoints import (
            analyze_standalone_code,
        )

        with patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.enhanced_analysis"
        ) as mock_enh:
            mock_enh.return_value = MagicMock()

            request = MagicMock()
            request.analysis_type = "unknown_type"
            await analyze_standalone_code(request, "aid-6", time.time())
            mock_enh.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_code_with_context(self):
        """Cover lines 722-734 - analyze code with context metadata."""
        from app.api.endpoints.analysis_modules.analysis_endpoints import (
            analyze_code_with_context,
        )

        with patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.analyze_standalone_code"
        ) as mock_standalone:
            mock_result = MagicMock()
            mock_result.metadata = {"existing": "data"}
            mock_standalone.return_value = mock_result

            context = {
                "repo_url": "https://github.com/test/repo",
                "container_name": "test-container",
            }
            request = MagicMock()
            result = await analyze_code_with_context(request, context, "aid-7", time.time())
            assert result.metadata["repository_context"] is True
            assert result.metadata["repository_url"] == "https://github.com/test/repo"
            assert result.metadata["container_name"] == "test-container"


# TestAnalysisUtilsCoverage removed: the only tests there exercised
# ``DatabaseLock`` — the sqlite per-file lock dropped in the postgres cutover
# (the schema is Alembic-owned and rows are scoped by code_graph_id, so there
# is no per-file lock to acquire).


# ---------------------------------------------------------------------------
# Tests for batch_analysis.py (lines 248-366)
# ---------------------------------------------------------------------------
class TestBatchAnalysisCoverage:
    """Cover uncovered lines in batch_analysis.py."""

    @pytest.mark.asyncio
    async def test_process_files_concurrent_with_exceptions(self):
        """Cover lines 271-292 - concurrent processing with exception handling."""
        from app.api.endpoints.analysis_modules.batch_analysis import (
            process_files_concurrent,
        )

        async def failing_task(task_info):
            raise RuntimeError("Analysis failed")

        tasks = [
            {"index": 0, "path": "/tmp/test.py", "language": "python"},
            {"index": 1, "path": "/tmp/test2.py", "language": "python"},
        ]

        with patch(
            "app.api.endpoints.analysis_modules.batch_analysis.analyze_single_file_batch"
        ) as mock_analyze:
            # First call succeeds, second fails
            mock_analyze.side_effect = [
                {"index": 0, "path": "/tmp/test.py", "success": True},
                RuntimeError("File not found"),
            ]
            semaphore = asyncio.Semaphore(2)
            results = await process_files_concurrent(tasks, semaphore)

            assert len(results) == 2
            assert results[0]["success"] is True
            assert results[1]["success"] is False
            assert "File not found" in results[1]["error"]

    @pytest.mark.asyncio
    async def test_process_files_concurrent_unexpected_result_type(self):
        """Cover lines 284-292 - unexpected result type handling."""
        from app.api.endpoints.analysis_modules.batch_analysis import (
            process_files_concurrent,
        )

        tasks = [{"index": 0, "path": "/tmp/test.py"}]

        with patch(
            "app.api.endpoints.analysis_modules.batch_analysis.analyze_single_file_batch"
        ) as mock_analyze:
            # Return a non-dict, non-exception result
            mock_analyze.return_value = "unexpected string result"
            semaphore = asyncio.Semaphore(2)
            results = await process_files_concurrent(tasks, semaphore)
            # The result is not a dict and not an exception, so it hits the else branch
            # But actually asyncio.gather returns the result directly if not exception
            # The code checks isinstance(result, dict) - "unexpected string result" is not dict
            assert len(results) == 1


# ---------------------------------------------------------------------------
# Tests for contract_analysis.py (lines 35-45, 77-78, 119)
# ---------------------------------------------------------------------------
class TestContractAnalysisCoverage:
    """Cover uncovered lines in contract_analysis.py."""

    def test_resolve_source_from_file_path(self):
        """Cover lines 35-43 - file_path resolution."""
        import tempfile

        from app.api.endpoints.contract_analysis import _resolve_source
        from models.contract_analysis_models import SourceInput

        # Create a real temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello')")
            f.flush()
            temp_path = f.name

        try:
            # Patch allowed dirs to include the temp directory (resolve for macOS /var -> /private/var)
            temp_dir = str(Path(temp_path).resolve().parent) + "/"
            with patch(
                "app.api.endpoints.contract_analysis._ALLOWED_BASE_DIRS",
                ("/app/provisioned_containers/", "/repo/", temp_dir),
            ):
                inp = SourceInput(file_path=temp_path, label="test")
                result = _resolve_source(inp)
                assert result == "print('hello')"
        finally:
            import os

            os.unlink(temp_path)

    def test_resolve_source_relative_path_raises(self):
        """Cover line 38 - relative path raises bad_request."""
        from app.api.endpoints.contract_analysis import _resolve_source
        from models.contract_analysis_models import SourceInput

        inp = SourceInput(file_path="relative/path.py", label="test")
        with pytest.raises(HTTPException):
            _resolve_source(inp)

    def test_resolve_source_file_not_found_raises(self):
        """Cover line 40 - nonexistent file raises bad_request."""
        from app.api.endpoints.contract_analysis import _resolve_source
        from models.contract_analysis_models import SourceInput

        inp = SourceInput(file_path="/nonexistent/file.py", label="test")
        with pytest.raises(HTTPException):
            _resolve_source(inp)

    def test_resolve_source_file_too_large_raises(self):
        """Cover line 42 - large file raises bad_request."""
        import tempfile

        from app.api.endpoints.contract_analysis import _MAX_FILE_SIZE, _resolve_source
        from models.contract_analysis_models import SourceInput

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x" * (_MAX_FILE_SIZE + 1))
            f.flush()
            temp_path = f.name

        try:
            inp = SourceInput(file_path=temp_path, label="test")
            with pytest.raises(HTTPException):
                _resolve_source(inp)
        finally:
            import os

            os.unlink(temp_path)

    def test_resolve_source_neither_source_nor_path(self):
        """Cover lines 44-45 - missing both source and file_path."""
        from app.api.endpoints.contract_analysis import _resolve_source
        from models.contract_analysis_models import SourceInput

        inp = SourceInput(label="my_input")
        with pytest.raises(HTTPException):
            _resolve_source(inp)

    def test_scan_one_resolve_error(self):
        """Cover line 77-78 - _scan_one when resolution fails."""
        from app.api.endpoints.contract_analysis import _scan_one
        from models.contract_analysis_models import ContractPair, SourceInput

        # schemas and routes need at least 1 item
        pair = ContractPair(
            name="test_pair",
            client=SourceInput(label="client"),  # No source or file_path → will fail
            schemas=[SourceInput(source="schema src")],
            routes=[SourceInput(source="route src")],
        )
        config = MagicMock()
        result = asyncio.run(_scan_one(pair, config))
        assert result.passed is False
        assert result.parse_error is not None


# ---------------------------------------------------------------------------
# Tests for models: container_models, architecture_models, testing_models
# ---------------------------------------------------------------------------
class TestModelsCoverage:
    """Cover uncovered lines in model files."""

    def test_container_model_pr_url_validation(self):
        """Cover container_models.py lines 102-148 - PR URL gets parsed to repo URL."""
        from models.container_models import GitHubRepoRequest

        # Test with a PR URL - model_validator parses it
        req = GitHubRepoRequest(
            repo_url="https://github.com/owner/repo/pull/42",
            branch="main",
        )
        # After model validation, PR URL should be converted to repo URL
        # and pr_number should be extracted
        assert req.pr_number == 42
        assert "github.com/owner/repo" in req.repo_url

    def test_container_model_validate_branch_empty(self):
        """Cover container_models.py line 140."""
        from models.container_models import GitHubRepoRequest

        with pytest.raises(Exception):
            GitHubRepoRequest(
                repo_url="https://github.com/owner/repo",
                branch="",
            )

    def test_container_model_validate_pr_number(self):
        """Cover container_models.py line 148."""
        from models.container_models import GitHubRepoRequest

        with pytest.raises(Exception):
            GitHubRepoRequest(
                repo_url="https://github.com/owner/repo",
                branch="main",
                pr_number=-1,
            )

    def test_container_model_invalid_short_url(self):
        """Cover container_models.py lines 123-126."""
        from models.container_models import GitHubRepoRequest

        with pytest.raises(Exception):
            GitHubRepoRequest(
                repo_url="ftp://x",
                branch="main",
            )

    def test_testing_models_test_request_validate_language(self):
        """Cover testing_models.py lines 183, 198."""
        from models.testing_models import TestRequest

        # Valid language
        req = TestRequest(
            language="python",
            code="x = 1",
            test_code="assert True",
            container_name="test",
        )
        assert req.language == "python"

        # Invalid language
        with pytest.raises(Exception):
            TestRequest(
                language="cobol",
                code="x = 1",
                test_code="assert True",
                container_name="test",
            )

        # Empty language
        with pytest.raises(Exception):
            TestRequest(
                language="",
                code="x = 1",
                test_code="assert True",
                container_name="test",
            )

    def test_testing_models_test_request_defaults(self):
        """Cover testing_models.py lines 204-207 - model validator defaults."""
        from models.testing_models import TestRequest

        req = TestRequest(
            language="python",
            container_name="test",
        )
        # Should default code_path and test_path to "."
        assert req.code_path == "."
        assert req.test_path == "."

    def test_testing_models_computed_fields(self):
        """Cover testing_models.py lines 213-219 - computed fields."""
        from models.testing_models import TestRequest

        req = TestRequest(
            language="python",
            container_name="test",
            code_file_path="/src/main.py",
            test_file_path="/tests/test_main.py",
        )
        assert req.code_file_path_resolved == "/src/main.py"
        assert req.test_file_path_resolved == "/tests/test_main.py"

    def test_testing_models_test_result_datetime_serializer(self):
        """Cover testing_models.py lines 260-266 - datetime serializer."""

        from datetime import datetime

        from models.testing_models import TestResult

        # With timezone
        result = TestResult(
            test_id="test-1",
            success=True,
            stdout="",
            stderr="",
            exit_code=0,
            execution_time=1.0,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            language="python",
            security_level="medium",
        )
        serialized = result.model_dump()
        assert "2026" in serialized["created_at"]

        # Without timezone
        result2 = TestResult(
            test_id="test-2",
            success=True,
            stdout="",
            stderr="",
            exit_code=0,
            execution_time=1.0,
            created_at=datetime(2026, 1, 1),
            language="python",
            security_level="medium",
        )
        serialized2 = result2.model_dump()
        assert "2026" in serialized2["created_at"]

        # None
        result3 = TestResult(
            test_id="test-3",
            success=True,
            stdout="",
            stderr="",
            exit_code=0,
            execution_time=1.0,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            language="python",
            security_level="medium",
        )
        assert result3.serialize_datetime(None) is None

    def test_testing_models_quality_request_validate_language(self):
        """Cover testing_models.py lines 312-320."""
        from models.testing_models import QualityRequest

        # Valid
        req = QualityRequest(
            container_name="test",
            language="python",
        )
        assert req.language == "python"

        # None is valid
        req2 = QualityRequest(
            container_name="test",
            language=None,
        )
        assert req2.language is None

        # Empty string
        with pytest.raises(Exception):
            QualityRequest(container_name="test", language="")

        # Unsupported
        with pytest.raises(Exception):
            QualityRequest(container_name="test", language="rust")

    def test_testing_models_quality_request_default_code_path(self):
        """Cover testing_models.py lines 323-327."""
        from models.testing_models import QualityRequest

        req = QualityRequest(container_name="test")
        assert req.code_path == "."

    def test_testing_models_quality_request_computed_field(self):
        """Cover testing_models.py line 333."""
        from models.testing_models import QualityRequest

        req = QualityRequest(
            container_name="test",
            code="print(1)",
            code_content="print(2)",
        )
        # code_content_resolved prefers code_content
        assert req.code_content_resolved == "print(2)"

    def test_architecture_model_component_validators(self):
        """Cover architecture_models.py lines 76-80, 84-97, 100-105."""
        from models.architecture_models import Component

        # Self-referencing dependency
        with pytest.raises(Exception):
            Component(
                id="comp-a",
                name="Component A",
                files=["src/a.py"],
                dependencies=["comp-a"],
            )

        # Empty files list
        with pytest.raises(Exception):
            Component(
                id="comp-b",
                name="Component B",
                files=[],
                dependencies=[],
            )

        # Duplicate dependencies get deduplicated
        comp = Component(
            id="comp-c",
            name="Component C",
            files=["src/c.py"],
            dependencies=["dep-1", "dep-1", "dep-2"],
        )
        assert len(comp.dependencies) == 2

    def test_architecture_model_layer_validation(self):
        """Cover architecture_models.py lines 119-164 - layer transition validation."""
        from models.architecture_models import Component

        # Component with layer but no ctx → passes validation
        comp = Component(
            id="comp-d",
            name="Component D",
            files=["src/d.py"],
            dependencies=[],
            layer="api",
        )
        assert comp.layer == "api"
