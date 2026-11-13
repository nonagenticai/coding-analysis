"""Round 3 coverage: analysis_endpoints.py - the highest-impact uncovered file (180 lines).

Tests:
- analyze_code endpoint with code snippet (various analysis types)
- analyze_standalone_code routing for all analysis types
- analyze_repository with no python files
- analyze_repository with failed deep analysis
- analyze_code_with_context
- error handling paths (ValueError, RuntimeError)
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.endpoints.analysis_modules.analysis_endpoints import (
    _fetch_remote_source_files,
    analyze_code,
    analyze_code_with_context,
    analyze_repository,
    analyze_standalone_code,
)
from app.api.models.analysis_api_models import (
    AnalysisRequest,
    ApiAnalysisResult,
)
from app.api.models.analysis_api_models import PydanticAIAnalysisType as AnalysisType

# Helper to create requests (container_name is required by the model)
CN = "test-container"


@pytest.fixture(autouse=True)
def _mock_analysis_status_service():
    """Mock AnalysisStatusService to avoid Redis dependency in tests."""
    mock_service = MagicMock()
    mock_service.update_status = AsyncMock(return_value=MagicMock())
    mock_service.create_analysis_session = AsyncMock(return_value=MagicMock())
    with patch(
        "services.analysis_status.AnalysisStatusService",
        return_value=mock_service,
    ):
        yield mock_service


# ─── analyze_standalone_code ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_standalone_basic_analysis():
    request = AnalysisRequest(
        code="x = 1\nprint(x)\n",
        language="python",
        analysis_type=AnalysisType.BASIC,
        container_name=CN,
    )
    result = await analyze_standalone_code(request, "test-id", time.time())
    assert isinstance(result, ApiAnalysisResult)
    assert result.success is True
    assert result.analysis_type == AnalysisType.BASIC


@pytest.mark.asyncio
async def test_standalone_augmented_analysis():
    request = AnalysisRequest(
        code="def hello():\n    return 'world'\n",
        language="python",
        analysis_type=AnalysisType.AUGMENTED,
        container_name=CN,
    )
    result = await analyze_standalone_code(request, "test-id", time.time())
    assert result.success is True


@pytest.mark.asyncio
async def test_standalone_ast_analysis():
    request = AnalysisRequest(
        code="import os\nclass Foo:\n    pass\n",
        language="python",
        analysis_type=AnalysisType.AST,
        container_name=CN,
    )
    result = await analyze_standalone_code(request, "test-id", time.time())
    assert result.success is True


@pytest.mark.asyncio
async def test_standalone_deep_analysis():
    request = AnalysisRequest(
        code="def func():\n    return 42\n",
        language="python",
        analysis_type=AnalysisType.DEEP,
        container_name=CN,
    )
    result = await analyze_standalone_code(request, "test-id", time.time())
    assert isinstance(result, ApiAnalysisResult)


@pytest.mark.asyncio
async def test_standalone_security_analysis():
    request = AnalysisRequest(
        code="import os\nos.system('ls')\n",
        language="python",
        analysis_type=AnalysisType.SECURITY,
        container_name=CN,
    )
    result = await analyze_standalone_code(request, "test-id", time.time())
    assert isinstance(result, ApiAnalysisResult)


@pytest.mark.asyncio
async def test_standalone_performance_analysis():
    request = AnalysisRequest(
        code="for i in range(1000):\n    pass\n",
        language="python",
        analysis_type=AnalysisType.PERFORMANCE,
        container_name=CN,
    )
    result = await analyze_standalone_code(request, "test-id", time.time())
    assert isinstance(result, ApiAnalysisResult)


# ─── analyze_repository ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_analyze_repository_no_container_path():
    """Missing container path raises HTTPException."""
    from fastapi import HTTPException

    request = AnalysisRequest(
        analysis_type=AnalysisType.DEEP,
        container_name=CN,
    )
    with pytest.raises(HTTPException) as exc_info:
        await analyze_repository(request, {}, "test-id", time.time())
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_analyze_repository_no_python_files(tmp_path):
    """No Python files returns empty analysis result."""
    (tmp_path / "readme.txt").write_text("hello")
    context = {
        "container_path": str(tmp_path),
        "repo_container_path": None,
        "fallback_path": None,
        "repo_url": "https://github.com/test/repo",
        "container_name": CN,
    }
    request = AnalysisRequest(
        analysis_type=AnalysisType.DEEP,
        container_name=CN,
    )
    result = await analyze_repository(request, context, "test-id", time.time())
    assert result.success is True
    assert result.metrics.lines_of_code == 0
    assert "No supported source files" in result.suggestions[0]


@pytest.mark.asyncio
async def test_analyze_repository_empty_echoes_requested_type(tmp_path):
    """The empty-repo path must echo the requested analysis_type, not a
    hardcoded DEEP. Regression: previously returned AnalysisType.DEEP for every
    request, violating the API contract for basic/security/etc. requests."""
    (tmp_path / "readme.txt").write_text("hello")
    context = {
        "container_path": str(tmp_path),
        "repo_container_path": None,
        "fallback_path": None,
        "repo_url": "https://github.com/test/repo",
        "container_name": CN,
    }
    request = AnalysisRequest(
        analysis_type=AnalysisType.SECURITY,
        container_name=CN,
    )
    result = await analyze_repository(request, context, "test-id", time.time())
    assert result.success is True
    assert result.analysis_type == AnalysisType.SECURITY


@pytest.mark.asyncio
async def test_analyze_repository_nonexistent_paths():
    """All candidate paths don't exist returns empty result."""
    context = {
        "container_path": "/nonexistent/path1",
        "repo_container_path": "/nonexistent/path2",
        "fallback_path": "/nonexistent/path3",
        "repo_url": "https://github.com/test/repo",
        "container_name": CN,
    }
    request = AnalysisRequest(
        analysis_type=AnalysisType.DEEP,
        container_name=CN,
    )
    result = await analyze_repository(request, context, "test-id", time.time())
    assert result.success is True
    assert result.metrics.lines_of_code == 0


@pytest.mark.asyncio
async def test_analyze_repository_with_python_files(tmp_path):
    """Repository with Python files runs DeepCodeAnalyzer."""
    (tmp_path / "app.py").write_text("def main():\n    pass\n")
    (tmp_path / "utils.py").write_text("def helper():\n    return 1\n")
    context = {
        "container_path": str(tmp_path),
        "repo_container_path": None,
        "fallback_path": None,
        "repo_url": "https://github.com/test/repo",
        "container_name": CN,
        "language": "python",
    }
    request = AnalysisRequest(
        analysis_type=AnalysisType.DEEP,
        container_name=CN,
    )
    result = await analyze_repository(request, context, "test-id", time.time())
    assert isinstance(result, ApiAnalysisResult)
    assert result.metadata["files_analyzed"] == 2


@pytest.mark.asyncio
async def test_analyze_repository_deep_analyzer_fails(tmp_path):
    """When DeepCodeAnalyzer fails, returns fallback result."""
    (tmp_path / "app.py").write_text("def main():\n    pass\n")
    context = {
        "container_path": str(tmp_path),
        "repo_container_path": None,
        "fallback_path": None,
        "repo_url": "https://github.com/test/repo",
        "container_name": CN,
        "language": "python",
    }
    request = AnalysisRequest(
        analysis_type=AnalysisType.DEEP,
        container_name=CN,
    )

    with patch(
        "app.api.endpoints.analysis_modules.analysis_endpoints.DeepCodeAnalyzer"
    ) as mock_analyzer:
        mock_analyzer.side_effect = RuntimeError("Analyzer crashed")
        result = await analyze_repository(request, context, "test-id", time.time())
    assert result.success is True
    assert "Deep analysis failed" in result.suggestions[0]
    assert result.metadata["analysis_scope"] == "failed_deep_fallback"


@pytest.mark.asyncio
async def test_analyze_repository_uses_fallback_path(tmp_path):
    """When primary path has no files, falls through to fallback."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    (fallback_dir / "code.py").write_text("x = 1\n")

    context = {
        "container_path": str(empty_dir),
        "repo_container_path": None,
        "fallback_path": str(fallback_dir),
        "repo_url": "https://github.com/test/repo",
        "container_name": CN,
        "language": "python",
    }
    request = AnalysisRequest(
        analysis_type=AnalysisType.DEEP,
        container_name=CN,
    )
    result = await analyze_repository(request, context, "test-id", time.time())
    assert result.metadata["files_analyzed"] == 1
    assert result.metadata["selected_path"] == str(fallback_dir)


# ─── analyze_code_with_context ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_analyze_code_with_context():
    request = AnalysisRequest(
        code="import os\nprint(os.getcwd())\n",
        language="python",
        analysis_type=AnalysisType.BASIC,
        container_name=CN,
    )
    context = {
        "repo_url": "https://github.com/test/repo",
        "container_name": CN,
    }
    result = await analyze_code_with_context(request, context, "test-id", time.time())
    assert result.success is True
    assert result.metadata.get("repository_context") is True
    assert result.metadata.get("repository_url") == "https://github.com/test/repo"
    assert result.metadata.get("container_name") == CN


# ─── Full analyze_code endpoint ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_analyze_code_with_code_and_container():
    """Code snippet + container analysis should complete successfully."""
    request = AnalysisRequest(
        code="def hello():\n    return 'world'\n",
        language="python",
        analysis_type=AnalysisType.BASIC,
        container_name=CN,
    )

    mock_container = MagicMock()
    mock_container.container_name = CN
    mock_container.repo_url = "https://github.com/test/repo"
    mock_container.codebase_path = "/tmp/test"

    with patch(
        "app.api.endpoints.analysis_modules.analysis_endpoints.get_github_provisioning_service"
    ) as mock_svc:
        mock_github = AsyncMock()
        mock_github.list_containers = AsyncMock(return_value=[mock_container])
        mock_svc.return_value = mock_github

        response = await analyze_code(request, tenant_id="test-tenant")
        assert response.message is not None
        assert response.result is not None
        assert response.request_id is not None


@pytest.mark.asyncio
async def test_analyze_code_with_container_not_found():
    """Container analysis with invalid container should raise 404."""
    from fastapi import HTTPException

    request = AnalysisRequest(
        container_name="nonexistent-container",
        analysis_type=AnalysisType.DEEP,
    )

    with patch(
        "app.api.endpoints.analysis_modules.analysis_endpoints.get_github_provisioning_service"
    ) as mock_svc:
        mock_github = AsyncMock()
        mock_github.list_containers = AsyncMock(return_value=[])
        mock_svc.return_value = mock_github

        with pytest.raises(HTTPException) as exc_info:
            await analyze_code(request, tenant_id="test-tenant")
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_analyze_code_container_service_error():
    """Container service error should raise 404."""
    from fastapi import HTTPException

    request = AnalysisRequest(
        container_name="some-container",
        analysis_type=AnalysisType.DEEP,
    )

    with patch(
        "app.api.endpoints.analysis_modules.analysis_endpoints.get_github_provisioning_service"
    ) as mock_svc:
        mock_svc.side_effect = RuntimeError("Service unavailable")

        with pytest.raises(HTTPException) as exc_info:
            await analyze_code(request, tenant_id="test-tenant")
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_analyze_code_with_container_and_code():
    """Code + container analysis should work with repository context."""
    request = AnalysisRequest(
        code="x = 1",
        container_name=CN,
        language="python",
        analysis_type=AnalysisType.BASIC,
    )

    mock_container = MagicMock()
    mock_container.container_name = CN
    mock_container.repo_url = "https://github.com/test/repo"
    mock_container.codebase_path = "/tmp/test"

    with patch(
        "app.api.endpoints.analysis_modules.analysis_endpoints.get_github_provisioning_service"
    ) as mock_svc:
        mock_github = AsyncMock()
        mock_github.list_containers = AsyncMock(return_value=[mock_container])
        mock_svc.return_value = mock_github

        response = await analyze_code(request, tenant_id="test-tenant")
        assert response.result is not None
        assert response.result.metadata.get("repository_context") is True


@pytest.mark.asyncio
async def test_analyze_code_container_no_codebase_path():
    """Container without codebase_path should raise 404."""
    from fastapi import HTTPException

    request = AnalysisRequest(
        container_name=CN,
        analysis_type=AnalysisType.DEEP,
    )

    mock_container = MagicMock()
    mock_container.container_name = CN
    mock_container.repo_url = "https://github.com/test/repo"
    mock_container.codebase_path = None
    mock_container.local_path = None

    with patch(
        "app.api.endpoints.analysis_modules.analysis_endpoints.get_github_provisioning_service"
    ) as mock_svc:
        mock_github = AsyncMock()
        mock_github.list_containers = AsyncMock(return_value=[mock_container])
        mock_svc.return_value = mock_github

        with pytest.raises(HTTPException) as exc_info:
            await analyze_code(request, tenant_id="test-tenant")
        assert exc_info.value.status_code == 404
        assert "codebase path" in exc_info.value.detail


@pytest.mark.asyncio
async def test_analyze_code_session_creation_fails():
    """If analysis session creation fails, analysis continues without status tracking."""
    request = AnalysisRequest(
        code="x = 1\n",
        language="python",
        analysis_type=AnalysisType.BASIC,
        container_name=CN,
    )

    mock_container = MagicMock()
    mock_container.container_name = CN
    mock_container.repo_url = "https://github.com/test/repo"
    mock_container.codebase_path = "/tmp/test"

    with (
        patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.get_github_provisioning_service"
        ) as mock_svc,
        patch("services.analysis_status.AnalysisStatusService") as mock_status,
    ):
        mock_github = AsyncMock()
        mock_github.list_containers = AsyncMock(return_value=[mock_container])
        mock_svc.return_value = mock_github

        mock_instance = AsyncMock()
        mock_instance.create_analysis_session = AsyncMock(
            side_effect=RuntimeError("DB unavailable")
        )
        mock_status.return_value = mock_instance

        response = await analyze_code(request, tenant_id="test-tenant")
        assert response.result is not None


# ─── Uncovered lines: ValueError / RuntimeError exception handlers ─────────


@pytest.mark.asyncio
async def test_analyze_code_no_code_no_container_raises_value_error():
    """analyze_code with neither code nor container_name should raise 400 (lines 237-246)."""
    from fastapi import HTTPException

    # Create a request with minimal valid data then override fields to bypass pydantic
    request = MagicMock()
    request.code = None
    request.container_name = ""  # empty string triggers the validation path
    request.analysis_type = AnalysisType.BASIC
    request.language = None
    request.options = None

    # Mock status service to cover the status update path (lines 240-245)
    with patch("services.analysis_status.AnalysisStatusService") as mock_status:
        mock_instance = AsyncMock()
        mock_instance.create_analysis_session = AsyncMock()
        mock_instance.update_status = AsyncMock()
        mock_status.return_value = mock_instance

        with pytest.raises((HTTPException, ValueError)):
            await analyze_code(request, tenant_id="test-tenant")


@pytest.mark.asyncio
async def test_analyze_code_value_error_updates_status_and_raises_400():
    """ValueError during analysis updates status and raises 400 (lines 409-423)."""
    from fastapi import HTTPException

    request = AnalysisRequest(
        code="x = 1\n",
        language="python",
        analysis_type=AnalysisType.BASIC,
        container_name=CN,
    )

    mock_container = MagicMock()
    mock_container.container_name = CN
    mock_container.repo_url = "https://github.com/test/repo"
    mock_container.codebase_path = "/tmp/test"

    with (
        patch("services.analysis_status.AnalysisStatusService") as mock_status_cls,
        patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.get_github_provisioning_service"
        ) as mock_github_svc,
        patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.analyze_code_with_context",
            side_effect=ValueError("bad code"),
        ),
    ):
        mock_github = AsyncMock()
        mock_github.list_containers = AsyncMock(return_value=[mock_container])
        mock_github_svc.return_value = mock_github

        mock_svc = AsyncMock()
        mock_svc.create_analysis_session = AsyncMock()
        mock_svc.update_status = AsyncMock()
        mock_status_cls.return_value = mock_svc

        with pytest.raises(HTTPException) as exc_info:
            await analyze_code(request, tenant_id="test-tenant")
        assert exc_info.value.status_code == 400
        assert "bad code" in exc_info.value.detail


@pytest.mark.asyncio
async def test_analyze_code_runtime_error_updates_status_and_raises_500():
    """RuntimeError during analysis updates status and raises 500 (lines 429-443)."""
    from fastapi import HTTPException

    request = AnalysisRequest(
        code="x = 1\n",
        language="python",
        analysis_type=AnalysisType.BASIC,
        container_name=CN,
    )

    mock_container = MagicMock()
    mock_container.container_name = CN
    mock_container.repo_url = "https://github.com/test/repo"
    mock_container.codebase_path = "/tmp/test"

    with (
        patch("services.analysis_status.AnalysisStatusService") as mock_status_cls,
        patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.get_github_provisioning_service"
        ) as mock_github_svc,
        patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.analyze_code_with_context",
            side_effect=RuntimeError("engine crashed"),
        ),
    ):
        mock_github = AsyncMock()
        mock_github.list_containers = AsyncMock(return_value=[mock_container])
        mock_github_svc.return_value = mock_github

        mock_svc = AsyncMock()
        mock_svc.create_analysis_session = AsyncMock()
        mock_svc.update_status = AsyncMock()
        mock_status_cls.return_value = mock_svc

        with pytest.raises(HTTPException) as exc_info:
            await analyze_code(request, tenant_id="test-tenant")
        assert exc_info.value.status_code == 500
        assert "engine crashed" in exc_info.value.detail


@pytest.mark.asyncio
async def test_analyze_repository_path_error_during_scanning(tmp_path):
    """OSError while scanning path candidates (lines 497-498)."""
    context = {
        "container_path": str(tmp_path / "nonexistent"),
        "repo_container_path": str(tmp_path / "also_nonexistent"),
        "fallback_path": None,
        "repo_url": "https://github.com/test/repo",
        "container_name": CN,
    }
    request = AnalysisRequest(
        analysis_type=AnalysisType.DEEP,
        container_name=CN,
    )
    result = await analyze_repository(request, context, "test-id", time.time())
    assert result.success is True
    assert result.metrics.lines_of_code == 0


@pytest.mark.asyncio
async def test_analyze_repository_graph_db_failure(tmp_path):
    """Graph database population failure (lines 652-654)."""
    (tmp_path / "app.py").write_text("def main():\n    pass\n")
    context = {
        "container_path": str(tmp_path),
        "repo_container_path": None,
        "fallback_path": None,
        "repo_url": "https://github.com/test/repo",
        "container_name": CN,
        "language": "python",
    }
    request = AnalysisRequest(
        analysis_type=AnalysisType.DEEP,
        container_name=CN,
    )

    with patch(
        "app.api.endpoints.analysis_modules.analysis_endpoints.populate_graph_database",
        side_effect=RuntimeError("DB failure"),
    ):
        result = await analyze_repository(request, context, "test-id", time.time())
        # Analysis should still succeed even if graph population fails
        assert isinstance(result, ApiAnalysisResult)


@pytest.mark.asyncio
async def test_analyze_repository_no_entities_creates_file_nodes(tmp_path):
    """When deep analysis returns no entities, file structure nodes are created (lines 623-635)."""
    (tmp_path / "app.py").write_text("x = 1\n")
    context = {
        "container_path": str(tmp_path),
        "repo_container_path": None,
        "fallback_path": None,
        "repo_url": "https://github.com/test/repo",
        "container_name": CN,
        "language": "python",
    }
    request = AnalysisRequest(
        analysis_type=AnalysisType.DEEP,
        container_name=CN,
    )

    mock_deep_result = MagicMock()
    mock_deep_result.entities = []
    mock_deep_result.relationships = []
    mock_deep_result.issues = []
    mock_deep_result.metrics = MagicMock()
    mock_deep_result.metrics.lines_of_code = 1
    mock_deep_result.metrics.cyclomatic_complexity = 0.0
    mock_deep_result.metrics.maintainability_index = 10.0

    with (
        patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.DeepCodeAnalyzer"
        ) as mock_analyzer_cls,
        patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.populate_graph_database",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_pop,
    ):
        mock_analyzer_cls.return_value.analyze_codebase.return_value = mock_deep_result
        result = await analyze_repository(request, context, "test-id", time.time())
        assert isinstance(result, ApiAnalysisResult)
        # Graph population should have been called with file nodes
        if mock_pop.called:
            call_args = mock_pop.call_args[0]
            analysis_data = call_args[1]
            assert len(analysis_data["nodes"]) >= 1


@pytest.mark.asyncio
async def test_analyze_repository_graph_success_with_entities(tmp_path):
    """Graph database populated successfully (lines 641-643)."""
    (tmp_path / "app.py").write_text("def main():\n    pass\n")
    context = {
        "container_path": str(tmp_path),
        "repo_container_path": None,
        "fallback_path": None,
        "repo_url": "https://github.com/test/repo",
        "container_name": CN,
        "language": "python",
    }
    request = AnalysisRequest(
        analysis_type=AnalysisType.DEEP,
        container_name=CN,
    )

    with patch(
        "app.api.endpoints.analysis_modules.analysis_endpoints.populate_graph_database",
        new_callable=AsyncMock,
        return_value=True,
    ):
        result = await analyze_repository(request, context, "test-id", time.time())
        assert isinstance(result, ApiAnalysisResult)


# ─── _fetch_remote_source_files: provision-race retry ─────────────────────────


def _exec_result(exit_code, stdout=""):
    return {"exit_code": exit_code, "stdout": stdout, "stderr": "", "duration": 0.0}


@pytest.mark.asyncio
async def test_fetch_remote_retries_until_exec_ready(tmp_path):
    """A just-provisioned container whose exec endpoint 500s (exit_code -1) must
    be retried, not treated as an empty repo. Once the find succeeds, source is
    fetched. Regression for architecture analysis returning 0 components after a
    transient provision-time exec failure."""
    find_ok = _exec_result(0, "/workspace/main.py\n")
    cat_ok = _exec_result(0, "print('hi')\n")
    # exec sequence: find fails twice (exec not ready), then find succeeds, then cat.
    exec_mock = AsyncMock(side_effect=[_exec_result(-1), _exec_result(-1), find_ok, cat_ok])
    client = MagicMock()
    client.execute = exec_mock

    with (
        patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.get_coding_containers_client",
            return_value=client,
        ),
        patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.tempfile.mkdtemp",
            return_value=str(tmp_path),
        ),
        patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep_mock,
    ):
        temp_dir, files = await _fetch_remote_source_files("ctr", tenant_id="t")

    assert temp_dir == str(tmp_path)
    assert len(files) == 1
    # 3 find attempts (2 failures + 1 success) then 1 cat = 4 exec calls, 2 backoffs.
    assert exec_mock.await_count == 4
    assert sleep_mock.await_count == 2


@pytest.mark.asyncio
async def test_fetch_remote_gives_up_after_max_attempts():
    """If the exec endpoint never recovers, return empty after the bounded
    retries (graceful degrade) rather than hanging."""
    exec_mock = AsyncMock(return_value=_exec_result(-1))
    client = MagicMock()
    client.execute = exec_mock

    with (
        patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.get_coding_containers_client",
            return_value=client,
        ),
        patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        temp_dir, files = await _fetch_remote_source_files("ctr", tenant_id="t")

    assert temp_dir is None
    assert files == []
    # All attempts exhausted on the find (no cat, since find never succeeded).
    from app.api.endpoints.analysis_modules.analysis_endpoints import _REMOTE_FIND_ATTEMPTS

    assert exec_mock.await_count == _REMOTE_FIND_ATTEMPTS


@pytest.mark.asyncio
async def test_fetch_remote_empty_repo_does_not_retry():
    """A genuinely-empty repo (find succeeds, no matches) must return fast
    without burning the retry budget."""
    exec_mock = AsyncMock(return_value=_exec_result(0, ""))
    client = MagicMock()
    client.execute = exec_mock

    with (
        patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.get_coding_containers_client",
            return_value=client,
        ),
        patch(
            "app.api.endpoints.analysis_modules.analysis_endpoints.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep_mock,
    ):
        temp_dir, files = await _fetch_remote_source_files("ctr", tenant_id="t")

    assert temp_dir is None
    assert files == []
    assert exec_mock.await_count == 1  # no retries for a real empty result
    assert sleep_mock.await_count == 0
