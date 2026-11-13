"""Unit tests for repo_scanner_modules package.

Covers:
- scanner_core.py (192 lines)
- results_processor.py (199 lines)
- scanner_config.py (145 lines)
- file_discovery.py (125 lines)
- code_validator.py (83 lines)
- dependency_scanner.py (73 lines)
- import_scanner.py (52 lines)
- scanner_endpoints.py (66 lines)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.endpoints.repo_scanner_modules.file_discovery import (
    _classify_single_file,
    classify_files,
    discover_python_files,
    execute_in_container,
    get_file_metadata,
)
from app.api.endpoints.repo_scanner_modules.results_processor import (
    _analyze_cross_scan_correlations,
    _analyze_import_patterns,
    _analyze_validation_issues,
    _analyze_version_patterns,
    _calculate_performance_metrics,
    _calculate_quality_metrics,
    _categorize_imports,
    _extract_errors,
    _generate_recommendations,
    _process_dependency_results,
    _process_full_scan_results,
    _process_import_results,
    _process_validation_results,
    generate_scan_summary,
    process_scan_results,
)
from app.api.endpoints.repo_scanner_modules.scanner_config import (
    ScannerConfig,
    create_scan_context,
    get_container_manager_v2,
    get_scanner_config,
    reset_scanner_config,
    update_scanner_config,
    validate_container_access,
)
from app.core.container_manager import ContainerInfo

# ============================================================================
# file_discovery.py
# ============================================================================


class TestExecuteInContainer:
    """Tests for execute_in_container."""

    async def test_success(self):
        container = ContainerInfo(container_name="ctr-1", status="running")
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[container])
        cc = AsyncMock()
        cc.execute = AsyncMock(return_value={"exit_code": 0, "stdout": "hello", "stderr": ""})

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.file_discovery.get_container_manager",
                return_value=mgr,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.file_discovery.get_coding_containers_client",
                return_value=cc,
            ),
        ):
            result = await execute_in_container("ctr-1", "echo hello", tenant_id="tenant-a")

        assert result["success"] is True
        assert result["output"] == "hello"
        assert result["exit_code"] == 0
        # The caller's tenant scopes the lookup AND rides on the exec itself:
        # coding-containers' exec is tenant-scoped, so dropping it 404s every
        # customer workspace.
        mgr.list_containers.assert_awaited_once_with(tenant_id="tenant-a")
        assert cc.execute.await_args.kwargs["tenant_id"] == "tenant-a"

    async def test_container_not_found_raises_404(self):
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[])
        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.file_discovery.get_container_manager",
                return_value=mgr,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await execute_in_container("missing", "ls", tenant_id="tenant-a")
        assert exc_info.value.status_code == 404

    async def test_container_not_running_raises_400(self):
        container = ContainerInfo(container_name="ctr-1", status="stopped")
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[container])
        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.file_discovery.get_container_manager",
                return_value=mgr,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await execute_in_container("ctr-1", "ls", tenant_id="tenant-a")
        assert exc_info.value.status_code == 400

    async def test_execute_failure_raises_500(self):
        container = ContainerInfo(container_name="ctr-1", status="running")
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[container])
        cc = AsyncMock()
        cc.execute = AsyncMock(side_effect=RuntimeError("boom"))

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.file_discovery.get_container_manager",
                return_value=mgr,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.file_discovery.get_coding_containers_client",
                return_value=cc,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await execute_in_container("ctr-1", "ls", tenant_id="tenant-a")
        assert exc_info.value.status_code == 500

    async def test_nonzero_exit_code(self):
        container = ContainerInfo(container_name="ctr-1", status="running")
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[container])
        cc = AsyncMock()
        cc.execute = AsyncMock(return_value={"exit_code": 1, "stdout": "", "stderr": "error msg"})

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.file_discovery.get_container_manager",
                return_value=mgr,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.file_discovery.get_coding_containers_client",
                return_value=cc,
            ),
        ):
            result = await execute_in_container("ctr-1", "false", tenant_id="tenant-a")

        assert result["success"] is False
        assert result["output"] == "error msg"

    async def test_unexpected_error_raises_500(self):
        """Unexpected exception not from HTTPException."""
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(side_effect=RuntimeError("unexpected"))
        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.file_discovery.get_container_manager",
                return_value=mgr,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await execute_in_container("ctr-1", "ls", tenant_id="tenant-a")
        assert exc_info.value.status_code == 500


class TestDiscoverPythonFiles:
    """Tests for discover_python_files."""

    async def test_success_returns_files(self):
        with patch(
            "app.api.endpoints.repo_scanner_modules.file_discovery.execute_in_container",
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "output": "/workspace/src/main.py\n/workspace/tests/test_main.py\n",
            },
        ):
            result = await discover_python_files("ctr-1")

        assert result["file_count"] == 2
        assert "/workspace/src/main.py" in result["python_files"]

    async def test_failure_returns_error(self):
        with patch(
            "app.api.endpoints.repo_scanner_modules.file_discovery.execute_in_container",
            new_callable=AsyncMock,
            return_value={"success": False, "output": "find: error"},
        ):
            result = await discover_python_files("ctr-1")

        assert result["file_count"] == 0
        assert "error" in result

    async def test_empty_output(self):
        with patch(
            "app.api.endpoints.repo_scanner_modules.file_discovery.execute_in_container",
            new_callable=AsyncMock,
            return_value={"success": True, "output": ""},
        ):
            result = await discover_python_files("ctr-1")

        assert result["file_count"] == 0
        assert result["python_files"] == []

    async def test_exclusion_filtering(self):
        with patch(
            "app.api.endpoints.repo_scanner_modules.file_discovery.execute_in_container",
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "output": "/workspace/__pycache__/foo.py\n/workspace/src/bar.py\n",
            },
        ):
            result = await discover_python_files("ctr-1")

        assert result["file_count"] == 1
        assert "/workspace/src/bar.py" in result["python_files"]

    async def test_exception_handled(self):
        with patch(
            "app.api.endpoints.repo_scanner_modules.file_discovery.execute_in_container",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            result = await discover_python_files("ctr-1")

        assert result["file_count"] == 0
        assert "error" in result


class TestClassifySingleFile:
    """Tests for _classify_single_file."""

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("/workspace/tests/test_main.py", "test"),
            ("/workspace/test/conftest.py", "test"),
            ("/workspace/src/test_utils.py", "test"),
            ("/workspace/src/my_test.py", "test"),
            ("/workspace/setup.py", "config"),
            ("/workspace/__init__.py", "config"),
            ("/workspace/config.py", "config"),
            ("/workspace/bin/run.py", "script"),
            ("/workspace/scripts/deploy.py", "script"),
            ("/workspace/tools/lint.py", "script"),
            ("/workspace/utils/helper.py", "script"),
            ("/workspace/docs/tutorial.py", "documentation"),
            ("/workspace/examples/demo.py", "documentation"),
            ("/workspace/src/app/main.py", "source"),
            ("/workspace/lib/parser.py", "source"),
            ("/workspace/api/routes.py", "source"),
            ("/workspace/random.py", "other"),
        ],
    )
    def test_classification(self, path, expected):
        assert _classify_single_file(path) == expected


class TestClassifyFiles:
    """Tests for classify_files."""

    async def test_classifies_correctly(self):
        files = [
            "/workspace/src/main.py",
            "/workspace/tests/test_main.py",
            "/workspace/setup.py",
        ]
        result = await classify_files("ctr-1", files)

        assert result["total_files"] == 3
        assert "source" in result["classified_files"]
        assert "test" in result["classified_files"]
        assert "config" in result["classified_files"]
        assert result["classification_summary"]["source"] == 1
        assert result["classification_summary"]["test"] == 1
        assert result["classification_summary"]["config"] == 1

    async def test_empty_list(self):
        result = await classify_files("ctr-1", [])
        assert result["total_files"] == 0


class TestGetFileMetadata:
    """Tests for get_file_metadata."""

    async def test_success(self):
        async def mock_exec(container, cmd, tenant_id=None):
            if "wc -l" in cmd:
                return {"success": True, "output": "42 /file.py"}
            if "stat" in cmd:
                return {"success": True, "output": "1024"}
            if "head" in cmd:
                return {"success": True, "output": "# header"}
            return {"success": False, "output": ""}

        with patch(
            "app.api.endpoints.repo_scanner_modules.file_discovery.execute_in_container",
            new_callable=AsyncMock,
            side_effect=mock_exec,
        ):
            result = await get_file_metadata("ctr-1", ["/file.py"])

        assert result["total_lines"] == 42
        assert result["file_metadata"][0]["line_count"] == 42
        assert result["file_metadata"][0]["size_bytes"] == 1024
        assert result["file_metadata"][0]["is_readable"] is True

    async def test_empty_file_list(self):
        result = await get_file_metadata("ctr-1", [])
        assert result["total_lines"] == 0
        assert result["average_file_size"] == 0

    async def test_failed_wc(self):
        async def mock_exec(container, cmd, tenant_id=None):
            return {"success": False, "output": ""}

        with patch(
            "app.api.endpoints.repo_scanner_modules.file_discovery.execute_in_container",
            new_callable=AsyncMock,
            side_effect=mock_exec,
        ):
            result = await get_file_metadata("ctr-1", ["/file.py"])

        assert result["file_metadata"][0]["line_count"] == 0
        assert result["file_metadata"][0]["is_readable"] is False


# ============================================================================
# results_processor.py
# ============================================================================


class TestProcessScanResults:
    """Tests for process_scan_results."""

    def test_dependencies_scan_type(self):
        start = datetime.now(UTC) - timedelta(seconds=1)
        raw = {
            "dependencies": [{"name": "requests", "version": "2.28.1", "source": "pip"}],
            "dependency_count": 1,
        }
        result = process_scan_results("dependencies", raw, start, "ctr-1")
        assert result["metadata"]["scan_type"] == "dependencies"
        assert result["statistics"]["total_items"] == 1

    def test_imports_scan_type(self):
        start = datetime.now(UTC) - timedelta(seconds=1)
        raw = {
            "imports": [{"module": "os", "import_type": "import"}],
            "import_count": 1,
            "invalid_imports": [],
        }
        result = process_scan_results("imports", raw, start, "ctr-1")
        assert result["statistics"]["total_items"] == 1

    def test_validation_scan_type(self):
        start = datetime.now(UTC) - timedelta(seconds=1)
        raw = {
            "validation_results": [
                {
                    "file_path": "a.py",
                    "is_valid": True,
                    "issues": [],
                    "warnings": [],
                    "line_count": 10,
                }
            ],
            "valid_files": 1,
            "invalid_files": 0,
        }
        result = process_scan_results("validation", raw, start, "ctr-1")
        assert result["statistics"]["valid_files"] == 1

    def test_full_scan_type(self):
        start = datetime.now(UTC) - timedelta(seconds=1)
        raw = {
            "dependencies": {"dependencies": [], "dependency_count": 0},
            "imports": {"imports": [], "import_count": 0, "invalid_imports": []},
            "validation": {
                "validation_results": [],
                "valid_files": 0,
                "invalid_files": 0,
            },
            "summary": {},
        }
        result = process_scan_results("full_scan", raw, start, "ctr-1")
        assert "dependencies" in result["results"]

    def test_unknown_scan_type(self):
        start = datetime.now(UTC) - timedelta(seconds=1)
        result = process_scan_results("unknown", {"key": "val"}, start, "ctr-1")
        assert "Unknown scan type: unknown" in result["errors"]

    def test_with_file_count(self):
        start = datetime.now(UTC) - timedelta(seconds=1)
        result = process_scan_results(
            "dependencies", {"dependencies": []}, start, "ctr-1", file_count=5
        )
        assert result["metadata"]["file_count"] == 5
        assert result["performance"]["files_per_second"] > 0

    def test_error_in_raw_results(self):
        start = datetime.now(UTC) - timedelta(seconds=1)
        raw = {"dependencies": [], "error": "scan failed"}
        result = process_scan_results("dependencies", raw, start, "ctr-1")
        assert "scan failed" in result["errors"]


class TestProcessDependencyResults:
    """Tests for _process_dependency_results."""

    def test_groups_by_source(self):
        raw = {
            "dependencies": [
                {"name": "a", "version": "1.0", "source": "pip"},
                {"name": "b", "version": "2.0", "source": "requirements"},
            ],
            "dependency_count": 2,
        }
        result = _process_dependency_results(raw)
        assert "pip" in result["results"]["sources"]
        assert "requirements" in result["results"]["sources"]
        assert result["statistics"]["total_items"] == 2

    def test_empty_dependencies(self):
        result = _process_dependency_results({"dependencies": []})
        assert result["statistics"]["total_items"] == 0


class TestAnalyzeVersionPatterns:
    """Tests for _analyze_version_patterns."""

    def test_version_categorization(self):
        deps = [
            {"version": "==1.0"},
            {"version": ">=2.0"},
            {"version": "1.0.0"},
            {"version": "~=3.0"},
        ]
        result = _analyze_version_patterns(deps)
        assert result["version_types"]["exact"] == 1
        assert result["version_types"]["range"] == 2
        assert result["version_types"]["unknown"] == 1

    def test_empty(self):
        result = _analyze_version_patterns([])
        assert result["total_versioned"] == 0


class TestCategorizeImports:
    """Tests for _categorize_imports."""

    def test_categories(self):
        imports = [
            {"module": "os"},
            {"module": "requests"},
            {"module": ".relative"},
            {"module": "app.models"},
            {"module": ""},  # empty, should be skipped
        ]
        result = _categorize_imports(imports)
        assert "os" in result["standard_library"]
        assert ".relative" in result["relative"]
        assert "app.models" in result["local"]
        assert "requests" in result["third_party"]


class TestAnalyzeImportPatterns:
    """Tests for _analyze_import_patterns."""

    def test_patterns(self):
        imports = [
            {"import_type": "import", "alias": None, "module": "os"},
            {"import_type": "from_import", "alias": "np", "module": "numpy"},
            {"import_type": "from_import", "alias": None, "module": "numpy"},
        ]
        result = _analyze_import_patterns(imports)
        assert result["import_types"]["import"] == 1
        assert result["import_types"]["from_import"] == 2
        assert result["aliased_imports"] == 1
        assert "numpy" in result["most_imported_modules"]


class TestAnalyzeValidationIssues:
    """Tests for _analyze_validation_issues."""

    def test_issue_categorization(self):
        results = [
            {
                "issues": ["SyntaxError at line 5: bad syntax"],
                "warnings": ["Line too long (130 chars)"],
            },
            {"issues": ["Other error"], "warnings": []},
        ]
        analysis = _analyze_validation_issues(results)
        assert analysis["files_with_issues"] == 2
        assert analysis["total_issues"] == 2
        assert analysis["total_warnings"] == 1
        assert analysis["issue_types"]["SyntaxError"] == 1
        assert analysis["issue_types"]["Other"] == 1
        assert analysis["warning_types"]["LineLength"] == 1

    def test_no_issues(self):
        results = [{"issues": [], "warnings": []}]
        analysis = _analyze_validation_issues(results)
        assert analysis["files_with_issues"] == 0


class TestCalculateQualityMetrics:
    """Tests for _calculate_quality_metrics."""

    def test_basic_quality(self):
        results = [
            {"is_valid": True, "line_count": 100},
            {"is_valid": True, "line_count": 50},
            {"is_valid": False, "line_count": 30},
        ]
        metrics = _calculate_quality_metrics(results)
        assert metrics["quality_score"] == pytest.approx(66.67, abs=0.01)
        assert metrics["metrics"]["files_analyzed"] == 3

    def test_empty_results(self):
        metrics = _calculate_quality_metrics([])
        assert metrics["quality_score"] == 0


class TestExtractErrors:
    """Tests for _extract_errors."""

    def test_extracts_error(self):
        raw = {"error": "something failed"}
        assert "something failed" in _extract_errors(raw)

    def test_extracts_validation_error(self):
        raw = {"validation_error": "val err"}
        assert "val err" in _extract_errors(raw)

    def test_extracts_file_errors(self):
        raw = {"validation_results": [{"file_path": "a.py", "error": "parse fail"}]}
        errors = _extract_errors(raw)
        assert any("a.py" in e for e in errors)

    def test_no_errors(self):
        assert _extract_errors({}) == []


class TestCalculatePerformanceMetrics:
    """Tests for _calculate_performance_metrics."""

    def test_with_files(self):
        raw = {"dependencies": [{"name": "a"}]}
        result = _calculate_performance_metrics(raw, 2.0, 10)
        assert result["files_per_second"] == 5.0
        assert result["items_per_second"] == 0.5

    def test_no_files(self):
        result = _calculate_performance_metrics({}, 1.0, None)
        assert result["files_per_second"] == 0


class TestAnalyzeCrossScanCorrelations:
    """Tests for _analyze_cross_scan_correlations."""

    def test_basic_correlation(self):
        deps = {"statistics": {"total_dependencies": 10}}
        imports = {"statistics": {"total_imports": 20}}
        validation = {"statistics": {"valid_files": 8, "total_items": 10}}
        result = _analyze_cross_scan_correlations(deps, imports, validation)
        assert "overall_health_score" in result
        assert result["overall_health_score"] > 0

    def test_zero_imports(self):
        deps = {"statistics": {"total_dependencies": 0}}
        imports = {"statistics": {"total_imports": 0}}
        validation = {"statistics": {"valid_files": 0, "total_items": 1}}
        result = _analyze_cross_scan_correlations(deps, imports, validation)
        assert result["dependency_import_correlation"]["dependencies_per_import"] == 0


class TestGenerateScanSummary:
    """Tests for generate_scan_summary."""

    def test_success_status(self):
        processed = {
            "metadata": {
                "scan_type": "dependencies",
                "container_name": "ctr-1",
                "end_time": datetime.now(UTC).isoformat(),
                "duration_seconds": 1.0,
            },
            "statistics": {"total_items": 5},
            "errors": [],
            "performance": {"duration_seconds": 1.0},
        }
        summary = generate_scan_summary(processed)
        assert summary["status"] == "success"

    def test_error_status(self):
        processed = {
            "metadata": {"scan_type": "x", "duration_seconds": 1.0},
            "statistics": {},
            "errors": ["something went wrong"],
            "performance": {},
        }
        summary = generate_scan_summary(processed)
        assert summary["status"] == "error"

    def test_warning_status(self):
        processed = {
            "metadata": {"scan_type": "x", "duration_seconds": 1.0},
            "statistics": {"invalid_files": 1},
            "errors": [],
            "performance": {},
        }
        summary = generate_scan_summary(processed)
        assert summary["status"] == "warning"


class TestGenerateRecommendations:
    """Tests for _generate_recommendations."""

    def test_errors_recommendation(self):
        recs = _generate_recommendations({"statistics": {}, "errors": ["err"], "performance": {}})
        assert any("error" in r.lower() for r in recs)

    def test_invalid_deps_recommendation(self):
        recs = _generate_recommendations(
            {
                "statistics": {"dependencies": {"invalid_dependencies": 2}},
                "errors": [],
                "performance": {},
            }
        )
        assert any("dependencies" in r.lower() for r in recs)

    def test_invalid_imports_recommendation(self):
        recs = _generate_recommendations(
            {
                "statistics": {"imports": {"invalid_imports": 1}},
                "errors": [],
                "performance": {},
            }
        )
        assert any("import" in r.lower() for r in recs)

    def test_invalid_files_recommendation(self):
        recs = _generate_recommendations(
            {
                "statistics": {"validation": {"invalid_files": 3}},
                "errors": [],
                "performance": {},
            }
        )
        assert any("syntax" in r.lower() for r in recs)

    def test_slow_scan_recommendation(self):
        recs = _generate_recommendations(
            {
                "statistics": {},
                "errors": [],
                "performance": {"duration_seconds": 120},
            }
        )
        assert any("performance" in r.lower() for r in recs)

    def test_healthy_recommendation(self):
        recs = _generate_recommendations({"statistics": {}, "errors": [], "performance": {}})
        assert any("healthy" in r.lower() for r in recs)


class TestProcessImportResults:
    """Tests for _process_import_results."""

    def test_basic(self):
        raw = {
            "imports": [{"module": "os", "import_type": "import", "alias": None}],
            "import_count": 1,
            "invalid_imports": [],
        }
        result = _process_import_results(raw)
        assert result["statistics"]["total_items"] == 1
        assert result["statistics"]["unique_modules"] == 1


class TestProcessValidationResults:
    """Tests for _process_validation_results."""

    def test_basic(self):
        raw = {
            "validation_results": [
                {
                    "file_path": "a.py",
                    "is_valid": True,
                    "issues": [],
                    "warnings": [],
                    "line_count": 50,
                }
            ],
            "valid_files": 1,
            "invalid_files": 0,
        }
        result = _process_validation_results(raw)
        assert result["statistics"]["total_items"] == 1
        assert result["statistics"]["total_lines"] == 50


class TestProcessFullScanResults:
    """Tests for _process_full_scan_results."""

    def test_combines_all(self):
        raw = {
            "dependencies": {"dependencies": [{"name": "a", "source": "pip"}]},
            "imports": {
                "imports": [{"module": "os", "import_type": "import", "alias": None}],
                "import_count": 1,
                "invalid_imports": [],
            },
            "validation": {
                "validation_results": [],
                "valid_files": 0,
                "invalid_files": 0,
            },
        }
        result = _process_full_scan_results(raw)
        assert "dependencies" in result["results"]
        assert "imports" in result["results"]
        assert "validation" in result["results"]
        assert "correlations" in result["results"]


# ============================================================================
# scanner_config.py
# ============================================================================


class TestScannerConfig:
    """Tests for ScannerConfig class."""

    def test_defaults(self):
        config = ScannerConfig()
        assert config.container_timeout == 300
        assert config.max_files_per_scan == 1000
        assert config.enable_caching is True
        assert config.parallel_processing is False
        assert "__pycache__" in config.exclusion_patterns

    def test_update_from_dict(self):
        config = ScannerConfig()
        config.update_from_dict({"container_timeout": 600, "max_files_per_scan": 500})
        assert config.container_timeout == 600
        assert config.max_files_per_scan == 500

    def test_update_merge_dict(self):
        config = ScannerConfig()
        config.update_from_dict({"validation_rules": {"max_line_length": 80}})
        assert config.validation_rules["max_line_length"] == 80
        # Other keys preserved
        assert config.validation_rules["check_syntax"] is True

    def test_update_unknown_key(self):
        config = ScannerConfig()
        config.update_from_dict({"nonexistent_key": 42})
        assert not hasattr(config, "nonexistent_key")

    def test_validate_config_valid(self):
        config = ScannerConfig()
        issues = config.validate_config()
        assert issues == []

    def test_validate_config_invalid_timeout(self):
        config = ScannerConfig()
        config.container_timeout = -1
        issues = config.validate_config()
        assert any("container_timeout" in i for i in issues)

    def test_validate_config_invalid_max_files(self):
        config = ScannerConfig()
        config.max_files_per_scan = 0
        issues = config.validate_config()
        assert any("max_files_per_scan" in i for i in issues)

    def test_validate_config_max_files_too_high(self):
        config = ScannerConfig()
        config.max_files_per_scan = 99999
        issues = config.validate_config()
        assert any("exceeds recommended" in i for i in issues)

    def test_validate_config_invalid_line_length(self):
        config = ScannerConfig()
        config.validation_rules["max_line_length"] = -1
        issues = config.validate_config()
        assert any("max_line_length" in i for i in issues)

    def test_validate_config_invalid_file_size(self):
        config = ScannerConfig()
        config.security_settings["max_file_size_bytes"] = 0
        issues = config.validate_config()
        assert any("max_file_size_bytes" in i for i in issues)

    def test_validate_config_file_size_too_large(self):
        config = ScannerConfig()
        config.security_settings["max_file_size_bytes"] = 200 * 1024 * 1024
        issues = config.validate_config()
        assert any("max_file_size_bytes" in i and "exceeds" in i for i in issues)

    def test_get_container_execution_config(self):
        config = ScannerConfig()
        exec_config = config.get_container_execution_config()
        assert exec_config["timeout"] == 300
        assert exec_config["workdir"] == "/workspace"

    def test_get_file_discovery_config(self):
        config = ScannerConfig()
        fd_config = config.get_file_discovery_config()
        assert "exclusion_patterns" in fd_config
        assert "max_files" in fd_config

    def test_get_validation_config(self):
        config = ScannerConfig()
        val_config = config.get_validation_config()
        assert val_config["max_line_length"] == 120
        # Should be a copy
        val_config["max_line_length"] = 80
        assert config.validation_rules["max_line_length"] == 120

    def test_to_dict(self):
        config = ScannerConfig()
        d = config.to_dict()
        assert d["container_timeout"] == 300
        assert "exclusion_patterns" in d
        assert "dependency_sources" in d


class TestScannerConfigGlobal:
    """Tests for global config functions."""

    def test_get_scanner_config(self):
        config = get_scanner_config()
        assert isinstance(config, ScannerConfig)

    def test_update_scanner_config(self):
        reset_scanner_config()
        update_scanner_config({"container_timeout": 999})
        config = get_scanner_config()
        assert config.container_timeout == 999
        reset_scanner_config()

    def test_reset_scanner_config(self):
        update_scanner_config({"container_timeout": 999})
        reset_scanner_config()
        config = get_scanner_config()
        assert config.container_timeout == 300


class TestGetContainerManagerV2:
    """Tests for get_container_manager_v2."""

    def test_returns_container_manager(self):
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_config.ContainerManager"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            get_container_manager_v2()
            mock_cls.assert_called_once()


class TestValidateContainerAccess:
    """Tests for validate_container_access."""

    async def test_container_not_found(self):
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[])
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_config.get_container_manager",
            return_value=mgr,
        ):
            result = await validate_container_access("missing")
        assert result["is_valid"] is False
        assert any("not found" in c for c in result["checks_failed"])

    async def test_container_not_running(self):
        container = ContainerInfo(container_name="ctr-1", status="stopped")
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[container])
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_config.get_container_manager",
            return_value=mgr,
        ):
            result = await validate_container_access("ctr-1")
        assert result["is_valid"] is False
        assert any("not running" in c for c in result["checks_failed"])

    async def test_container_valid_full(self):
        container = ContainerInfo(container_name="ctr-1", status="running", work_dir="/workspace")
        detail = ContainerInfo(
            container_name="ctr-1",
            status="running",
            work_dir="/workspace",
            repo_url="https://github.com/test",
        )
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[container])
        mgr.get_container = AsyncMock(return_value=detail)
        mgr.execute_command = AsyncMock(return_value={"success": True, "output": "Python 3.12.0"})
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_config.get_container_manager",
            return_value=mgr,
        ):
            result = await validate_container_access("ctr-1")
        assert len(result["checks_passed"]) >= 3

    async def test_get_container_returns_none(self):
        container = ContainerInfo(container_name="ctr-1", status="running")
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[container])
        mgr.get_container = AsyncMock(return_value=None)
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_config.get_container_manager",
            return_value=mgr,
        ):
            result = await validate_container_access("ctr-1")
        assert result["is_valid"] is False

    async def test_list_containers_error(self):
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(side_effect=RuntimeError("fail"))
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_config.get_container_manager",
            return_value=mgr,
        ):
            result = await validate_container_access("ctr-1")
        assert result["is_valid"] is False

    async def test_unexpected_top_level_error(self):
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_config.get_container_manager",
            side_effect=RuntimeError("boom"),
        ):
            result = await validate_container_access("ctr-1")
        assert result["is_valid"] is False
        assert len(result["checks_failed"]) > 0


class TestCreateScanContext:
    """Tests for create_scan_context."""

    async def test_creates_context(self):
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_config.validate_container_access",
            new_callable=AsyncMock,
            return_value={
                "is_valid": True,
                "container_info": {},
                "checks_passed": [],
                "checks_failed": [],
                "recommendations": [],
            },
        ):
            reset_scanner_config()
            ctx = await create_scan_context("ctr-1", "dependencies")

        assert ctx["container_name"] == "ctr-1"
        assert ctx["scan_type"] == "dependencies"
        assert "scan_id" in ctx
        assert "config" in ctx
        assert ctx["validation"]["is_valid"] is True

    async def test_with_target_files(self):
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_config.validate_container_access",
            new_callable=AsyncMock,
            return_value={
                "is_valid": False,
                "container_info": {},
                "checks_passed": [],
                "checks_failed": ["missing"],
                "recommendations": [],
            },
        ):
            ctx = await create_scan_context("ctr-1", "imports", target_files=["a.py"])

        assert ctx["target_files"] == ["a.py"]
        assert ctx["validation"]["is_valid"] is False


# ============================================================================
# scanner_core.py — scan_dependencies (core)
# ============================================================================


class TestScannerCoreScanDependencies:
    """Tests for scanner_core.scan_dependencies."""

    async def test_success(self):
        deps_json = json.dumps([{"name": "requests", "version": "2.28.1", "source": "pip"}])

        async def mock_exec(container, cmd, **kwargs):
            if "scan_dependencies" in cmd and " -c " not in cmd:
                return {"success": True, "output": deps_json}
            return {"success": True, "output": "ok"}

        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.execute_in_container",
            new_callable=AsyncMock,
            side_effect=mock_exec,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                scan_dependencies as core_scan_deps,
            )

            result = await core_scan_deps("ctr-1")

        assert result["dependency_count"] == 1
        assert result["dependencies"][0].name == "requests"

    async def test_write_script_failure(self):
        async def mock_exec(container, cmd, **kwargs):
            return {"success": False, "output": "write failed"}

        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.execute_in_container",
            new_callable=AsyncMock,
            side_effect=mock_exec,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                scan_dependencies as core_scan_deps,
            )

            result = await core_scan_deps("ctr-1")

        assert "error" in result

    async def test_json_decode_error(self):
        async def mock_exec(container, cmd, **kwargs):
            if " -c " not in cmd and "scan_dependencies" in cmd:
                return {"success": True, "output": "not json"}
            return {"success": True, "output": "ok"}

        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.execute_in_container",
            new_callable=AsyncMock,
            side_effect=mock_exec,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                scan_dependencies as core_scan_deps,
            )

            result = await core_scan_deps("ctr-1")

        assert "error" in result

    async def test_execution_failure(self):
        call_count = 0

        async def mock_exec(container, cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": True, "output": "ok"}
            return {"success": False, "output": "script error"}

        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.execute_in_container",
            new_callable=AsyncMock,
            side_effect=mock_exec,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                scan_dependencies as core_scan_deps,
            )

            result = await core_scan_deps("ctr-1")

        assert "error" in result


# ============================================================================
# scanner_core.py — scan_imports (core)
# ============================================================================


class TestScannerCoreScanImports:
    """Tests for scanner_core.scan_imports."""

    async def test_success(self):
        imports_json = json.dumps(
            [
                {
                    "module": "os",
                    "name": None,
                    "alias": None,
                    "import_type": "import",
                    "file_path": "a.py",
                    "line_number": 1,
                    "is_valid": True,
                }
            ]
        )

        async def mock_exec(container, cmd, **kwargs):
            if "scan_imports" in cmd and " -c " not in cmd:
                return {"success": True, "output": imports_json}
            return {"success": True, "output": "ok"}

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_core.execute_in_container",
                new_callable=AsyncMock,
                side_effect=mock_exec,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.file_discovery.execute_in_container",
                new_callable=AsyncMock,
                return_value={"success": True, "output": "/workspace/a.py\n"},
            ),
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                scan_imports as core_scan_imports,
            )

            result = await core_scan_imports("ctr-1")

        assert result["import_count"] == 1
        assert "error" not in result

    async def test_with_target_files(self):
        imports_json = json.dumps([])

        async def mock_exec(container, cmd, **kwargs):
            if "scan_imports" in cmd and " -c " not in cmd:
                return {"success": True, "output": imports_json}
            return {"success": True, "output": "ok"}

        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.execute_in_container",
            new_callable=AsyncMock,
            side_effect=mock_exec,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                scan_imports as core_scan_imports,
            )

            result = await core_scan_imports("ctr-1", target_files=["a.py"])

        assert result["import_count"] == 0

    async def test_write_failure(self):
        async def mock_exec(container, cmd, **kwargs):
            return {"success": False, "output": "write fail"}

        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.execute_in_container",
            new_callable=AsyncMock,
            side_effect=mock_exec,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                scan_imports as core_scan_imports,
            )

            result = await core_scan_imports("ctr-1")

        assert "error" in result


# ============================================================================
# scanner_core.py — validate_code (core)
# ============================================================================


class TestScannerCoreValidateCode:
    """Tests for scanner_core.validate_code."""

    async def test_success(self):
        validation_json = json.dumps(
            [
                {
                    "file_path": "a.py",
                    "is_valid": True,
                    "issues": [],
                    "warnings": [],
                    "line_count": 10,
                }
            ]
        )

        async def mock_exec(container, cmd, **kwargs):
            if "validate_code" in cmd and " -c " not in cmd:
                return {"success": True, "output": validation_json}
            return {"success": True, "output": "ok"}

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_core.execute_in_container",
                new_callable=AsyncMock,
                side_effect=mock_exec,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.file_discovery.execute_in_container",
                new_callable=AsyncMock,
                return_value={"success": True, "output": "/workspace/a.py\n"},
            ),
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                validate_code as core_validate_code,
            )

            result = await core_validate_code("ctr-1")

        assert result["valid_files"] == 1
        assert result["invalid_files"] == 0
        assert "error" not in result

    async def test_write_failure(self):
        async def mock_exec(container, cmd, **kwargs):
            return {"success": False, "output": "fail"}

        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.execute_in_container",
            new_callable=AsyncMock,
            side_effect=mock_exec,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                validate_code as core_validate_code,
            )

            result = await core_validate_code("ctr-1")

        assert "error" in result


# ============================================================================
# scanner_core.py — validate_scanned_imports (core)
# ============================================================================


class TestValidateScannedImportsCore:
    """Tests for scanner_core.validate_scanned_imports."""

    async def test_success(self):
        mock_result = MagicMock()
        mock_result.valid_imports = [{"module": "os", "name": None}]
        mock_result.invalid_imports = []
        mock_result.stdlib_imports = [{"module": "os"}]
        mock_result.third_party_imports = []
        mock_result.local_imports = []
        mock_result.missing_dependencies = []
        mock_result.suggestions = []

        mock_svc = AsyncMock()
        mock_svc.validate_imports = AsyncMock(return_value=mock_result)

        mgr = AsyncMock()
        mgr.get_container = AsyncMock(return_value=MagicMock())

        imports = [{"module": "os", "name": None}]

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
                return_value=mgr,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_core.ImportValidationService",
                return_value=mock_svc,
            ),
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                validate_scanned_imports as core_validate_imports,
            )

            result = await core_validate_imports("ctr-1", imports)

        assert result["import_count"] == 1
        assert result["validation_summary"]["valid_count"] == 1

    async def test_container_not_found_fallback(self):
        mock_result = MagicMock()
        mock_result.valid_imports = []
        mock_result.invalid_imports = [{"module": "os"}]
        mock_result.stdlib_imports = []
        mock_result.third_party_imports = []
        mock_result.local_imports = []
        mock_result.missing_dependencies = []
        mock_result.suggestions = []

        mock_svc = AsyncMock()
        mock_svc.validate_imports = AsyncMock(return_value=mock_result)

        mgr = AsyncMock()
        mgr.get_container = AsyncMock(side_effect=ValueError("not found"))

        imports = [{"module": "os", "name": None}]

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
                return_value=mgr,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_core.ImportValidationService",
                return_value=mock_svc,
            ),
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                validate_scanned_imports as core_validate_imports,
            )

            result = await core_validate_imports("ctr-1", imports)

        # Should still succeed with fallback project root
        assert result["import_count"] == 1

    async def test_validation_error_returns_originals(self):
        mgr = AsyncMock()
        mgr.get_container = AsyncMock(side_effect=ValueError("err"))

        imports = [{"module": "os", "name": None}]

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
                return_value=mgr,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_core.ImportValidationService",
                side_effect=FileNotFoundError("no project"),
            ),
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                validate_scanned_imports as core_validate_imports,
            )

            result = await core_validate_imports("ctr-1", imports)

        assert "validation_error" in result
        assert result["imports"] == imports


# ============================================================================
# scanner_core.py — scan_dependencies_v2 (core)
# ============================================================================


class TestScannerCoreScanDependenciesV2:
    """Tests for scanner_core.scan_dependencies_v2."""

    async def test_pip_success(self):
        pip_json = json.dumps([{"name": "flask", "version": "3.0.0"}])
        mgr = AsyncMock()
        mgr.execute_command = AsyncMock(
            side_effect=[
                {"success": True, "output": pip_json},  # pip list
                {"success": False, "output": ""},  # requirements.txt
                {"success": False, "output": ""},  # pyproject.toml
            ]
        )
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
            return_value=mgr,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                scan_dependencies_v2 as core_scan_deps_v2,
            )

            result = await core_scan_deps_v2("ctr-1")

        assert result["dependency_count"] == 1
        assert "pip" in result["sources"]

    async def test_requirements_success(self):
        mgr = AsyncMock()
        mgr.execute_command = AsyncMock(
            side_effect=[
                {"success": False, "output": ""},  # pip list
                {"success": True, "output": "requests==2.28.1\nflask>=3.0"},  # req
                {"success": False, "output": ""},  # pyproject
            ]
        )
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
            return_value=mgr,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                scan_dependencies_v2 as core_scan_deps_v2,
            )

            result = await core_scan_deps_v2("ctr-1")

        assert result["dependency_count"] == 2
        assert "requirements.txt" in result["sources"]

    async def test_pip_json_with_prefix(self):
        """pip output with junk before JSON."""
        mgr = AsyncMock()
        mgr.execute_command = AsyncMock(
            side_effect=[
                {
                    "success": True,
                    "output": 'WARNING: blah\n[{"name": "a", "version": "1"}]',
                },
                {"success": False, "output": ""},
                {"success": False, "output": ""},
            ]
        )
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
            return_value=mgr,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                scan_dependencies_v2 as core_scan_deps_v2,
            )

            result = await core_scan_deps_v2("ctr-1")

        assert result["dependency_count"] == 1

    async def test_value_error(self):
        mgr = AsyncMock()
        mgr.execute_command = AsyncMock(side_effect=ValueError("fail"))
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
            return_value=mgr,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                scan_dependencies_v2 as core_scan_deps_v2,
            )

            result = await core_scan_deps_v2("ctr-1")

        assert "error" in result


# ============================================================================
# scanner_core.py — validate_code_v2 (core)
# ============================================================================


class TestScannerCoreValidateCodeV2:
    """Tests for scanner_core.validate_code_v2."""

    async def test_with_target_files(self):
        mgr = AsyncMock()
        mgr.execute_command = AsyncMock(return_value={"success": True, "output": "print('hello')"})
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
            return_value=mgr,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                validate_code_v2 as core_validate_v2,
            )

            result = await core_validate_v2("ctr-1", target_files=["a.py"])

        assert result["valid_files"] == 1
        assert result["invalid_files"] == 0

    async def test_syntax_error_detected(self):
        mgr = AsyncMock()
        mgr.execute_command = AsyncMock(return_value={"success": True, "output": "def bad(\n"})
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
            return_value=mgr,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                validate_code_v2 as core_validate_v2,
            )

            result = await core_validate_v2("ctr-1", target_files=["b.py"])

        assert result["invalid_files"] == 1

    async def test_file_discovery(self):
        mgr = AsyncMock()
        # First call: find command; second call: cat
        mgr.execute_command = AsyncMock(
            side_effect=[
                {"success": True, "output": "a.py\nb.py\n"},
                {"success": True, "output": "x = 1"},
                {"success": True, "output": "y = 2"},
            ]
        )
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
            return_value=mgr,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                validate_code_v2 as core_validate_v2,
            )

            result = await core_validate_v2("ctr-1")

        assert result["valid_files"] == 2

    async def test_find_fails(self):
        mgr = AsyncMock()
        mgr.execute_command = AsyncMock(return_value={"success": False, "output": ""})
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
            return_value=mgr,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                validate_code_v2 as core_validate_v2,
            )

            result = await core_validate_v2("ctr-1")

        assert result["valid_files"] == 0

    async def test_cat_fails(self):
        mgr = AsyncMock()
        mgr.execute_command = AsyncMock(
            side_effect=[
                {"success": True, "output": "a.py\n"},
                {"success": False, "output": "no such file"},
            ]
        )
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
            return_value=mgr,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                validate_code_v2 as core_validate_v2,
            )

            result = await core_validate_v2("ctr-1")

        assert result["valid_files"] == 0
        assert result["invalid_files"] == 0

    async def test_value_error_in_file_loop(self):
        """ValueError inside file processing loop is caught per-file."""
        mgr = AsyncMock()
        mgr.execute_command = AsyncMock(side_effect=ValueError("err"))
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
            return_value=mgr,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                validate_code_v2 as core_validate_v2,
            )

            result = await core_validate_v2("ctr-1", target_files=["a.py"])

        # Per-file ValueError is caught, not top-level error
        assert result["invalid_files"] == 1
        assert "ProcessingError" in result["validation_results"][0]["issues"][0]

    async def test_top_level_value_error(self):
        """ValueError in get_container_manager triggers top-level error."""
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
            side_effect=FileNotFoundError("missing"),
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                validate_code_v2 as core_validate_v2,
            )

            result = await core_validate_v2("ctr-1", target_files=["a.py"])

        assert "error" in result

    async def test_long_line_warning(self):
        long_code = "x = 1\n" + "a" * 130 + "\ny = 2\n"
        mgr = AsyncMock()
        mgr.execute_command = AsyncMock(return_value={"success": True, "output": long_code})
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
            return_value=mgr,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                validate_code_v2 as core_validate_v2,
            )

            result = await core_validate_v2("ctr-1", target_files=["a.py"])

        assert result["valid_files"] == 1
        assert len(result["validation_results"][0]["warnings"]) >= 1

    async def test_filters_non_py_files(self):
        mgr = AsyncMock()
        mgr.execute_command = AsyncMock(return_value={"success": True, "output": "print(1)"})
        with patch(
            "app.api.endpoints.repo_scanner_modules.scanner_core.get_container_manager",
            return_value=mgr,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                validate_code_v2 as core_validate_v2,
            )

            result = await core_validate_v2("ctr-1", target_files=["a.py", "b.txt"])

        # Only a.py should be validated
        assert len(result["validation_results"]) == 1


# ============================================================================
# code_validator.py — validate_code
# ============================================================================


class TestCodeValidatorValidateCode:
    """Tests for code_validator.validate_code."""

    async def test_success(self):
        validation_json = json.dumps(
            [
                {
                    "file_path": "x.py",
                    "is_valid": True,
                    "issues": [],
                    "warnings": [],
                    "line_count": 5,
                }
            ]
        )

        async def mock_exec(container, cmd, **kwargs):
            if "validate_code" in cmd and " -c " not in cmd:
                return {"success": True, "output": validation_json}
            return {"success": True, "output": "ok"}

        with patch(
            "app.api.endpoints.repo_scanner_modules.code_validator.execute_in_container",
            new_callable=AsyncMock,
            side_effect=mock_exec,
        ):
            from app.api.endpoints.repo_scanner_modules.code_validator import (
                validate_code as cv_validate_code,
            )

            result = await cv_validate_code("ctr-1")

        assert result["valid_files"] == 1

    async def test_with_target_files(self):
        validation_json = json.dumps(
            [
                {
                    "file_path": "x.py",
                    "is_valid": False,
                    "issues": ["SyntaxError"],
                    "warnings": [],
                    "line_count": 3,
                }
            ]
        )

        async def mock_exec(container, cmd, **kwargs):
            if "validate_code" in cmd and " -c " not in cmd:
                return {"success": True, "output": validation_json}
            return {"success": True, "output": "ok"}

        with patch(
            "app.api.endpoints.repo_scanner_modules.code_validator.execute_in_container",
            new_callable=AsyncMock,
            side_effect=mock_exec,
        ):
            from app.api.endpoints.repo_scanner_modules.code_validator import (
                validate_code as cv_validate_code,
            )

            result = await cv_validate_code("ctr-1", target_files=["x.py"])

        assert result["invalid_files"] == 1


# ============================================================================
# code_validator.py — validate_code_v2
# ============================================================================


class TestCodeValidatorV2:
    """Tests for code_validator.validate_code_v2."""

    async def test_success(self):
        mgr = AsyncMock()
        mgr.execute_command = AsyncMock(return_value={"success": True, "output": "x = 1"})
        with patch(
            "app.api.endpoints.repo_scanner_modules.code_validator.get_container_manager",
            return_value=mgr,
        ):
            from app.api.endpoints.repo_scanner_modules.code_validator import (
                validate_code_v2 as cv_validate_v2,
            )

            result = await cv_validate_v2("ctr-1", target_files=["a.py"])

        assert result["valid_files"] == 1


# ============================================================================
# dependency_scanner.py — scan_dependencies
# ============================================================================


class TestDepScannerScanDependencies:
    """Tests for dependency_scanner.scan_dependencies."""

    async def test_success(self):
        deps_json = json.dumps([{"name": "flask", "version": "3.0", "source": "pip"}])

        async def mock_exec(container, cmd, **kwargs):
            if "scan_dependencies" in cmd and " -c " not in cmd:
                return {"success": True, "output": deps_json}
            return {"success": True, "output": "ok"}

        with patch(
            "app.api.endpoints.repo_scanner_modules.dependency_scanner.execute_in_container",
            new_callable=AsyncMock,
            side_effect=mock_exec,
        ):
            from app.api.endpoints.repo_scanner_modules.dependency_scanner import (
                scan_dependencies as ds_scan_deps,
            )

            result = await ds_scan_deps("ctr-1")

        assert result["dependency_count"] == 1


# ============================================================================
# dependency_scanner.py — scan_dependencies_v2
# ============================================================================


class TestDepScannerV2:
    """Tests for dependency_scanner.scan_dependencies_v2."""

    async def test_requirements_parsed(self):
        mgr = AsyncMock()
        mgr.execute_command = AsyncMock(
            side_effect=[
                {"success": False, "output": ""},  # pip
                {
                    "success": True,
                    "output": "# comment\nflask==3.0\nrequests>=2.28",
                },  # req
                {"success": False, "output": ""},  # pyproject
            ]
        )
        with patch(
            "app.api.endpoints.repo_scanner_modules.dependency_scanner.get_container_manager",
            return_value=mgr,
        ):
            from app.api.endpoints.repo_scanner_modules.dependency_scanner import (
                scan_dependencies_v2 as ds_scan_v2,
            )

            result = await ds_scan_v2("ctr-1")

        assert result["dependency_count"] == 2
        assert "requirements.txt" in result["sources"]


# ============================================================================
# import_scanner.py — scan_imports
# ============================================================================


class TestImportScannerScanImports:
    """Tests for import_scanner.scan_imports."""

    async def test_success(self):
        imports_json = json.dumps(
            [
                {
                    "module": "json",
                    "name": None,
                    "alias": None,
                    "import_type": "import",
                    "file_path": "a.py",
                    "line_number": 1,
                    "is_valid": True,
                }
            ]
        )

        async def mock_exec(container, cmd, **kwargs):
            if "scan_imports" in cmd and " -c " not in cmd:
                return {"success": True, "output": imports_json}
            return {"success": True, "output": "ok"}

        with patch(
            "app.api.endpoints.repo_scanner_modules.import_scanner.execute_in_container",
            new_callable=AsyncMock,
            side_effect=mock_exec,
        ):
            from app.api.endpoints.repo_scanner_modules.import_scanner import (
                scan_imports as is_scan_imports,
            )

            result = await is_scan_imports("ctr-1")

        assert result["import_count"] == 1


# ============================================================================
# import_scanner.py — validate_scanned_imports
# ============================================================================


class TestImportScannerValidate:
    """Tests for import_scanner.validate_scanned_imports."""

    async def test_error_returns_originals(self):
        mgr = AsyncMock()
        mgr.get_container = AsyncMock(side_effect=ValueError("nope"))

        imports = [{"module": "os", "name": None}]

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.import_scanner.get_container_manager",
                return_value=mgr,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.import_scanner.ImportValidationService",
                side_effect=FileNotFoundError("no project"),
            ),
        ):
            from app.api.endpoints.repo_scanner_modules.import_scanner import (
                validate_scanned_imports as is_validate,
            )

            result = await is_validate("ctr-1", imports)

        assert "validation_error" in result


# ============================================================================
# scanner_endpoints.py
# ============================================================================


class TestScannerEndpoints:
    """Tests for scanner_endpoints.scan_repository."""

    def _make_request(self, action, container_name="ctr-1", target_files=None):
        from models.repo_scanner_models import RepoScanAction

        req = MagicMock()
        req.container_name = container_name
        req.action = getattr(RepoScanAction, action.upper())
        req.target_files = target_files
        return req

    async def test_container_not_found(self):
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[])

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_endpoints.get_container_manager",
                return_value=mgr,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_endpoints import (
                scan_repository,
            )

            await scan_repository(self._make_request("dependencies"), tenant_id="t1")
        assert exc_info.value.status_code == 404

    async def test_container_not_running(self):
        container = ContainerInfo(container_name="ctr-1", status="stopped")
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[container])

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_endpoints.get_container_manager",
                return_value=mgr,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_endpoints import (
                scan_repository,
            )

            await scan_repository(self._make_request("dependencies"), tenant_id="t1")
        assert exc_info.value.status_code == 400

    async def test_dependencies_scan(self):
        container = ContainerInfo(container_name="ctr-1", status="running")
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[container])

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_endpoints.get_container_manager",
                return_value=mgr,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_endpoints.scan_dependencies",
                new_callable=AsyncMock,
                return_value={
                    "dependencies": [],
                    "dependency_count": 0,
                    "invalid_dependencies": [],
                },
            ),
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_endpoints import (
                scan_repository,
            )

            result = await scan_repository(self._make_request("dependencies"), tenant_id="t1")
        assert result.action == "dependencies"

    async def test_imports_scan(self):
        container = ContainerInfo(container_name="ctr-1", status="running")
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[container])

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_endpoints.get_container_manager",
                return_value=mgr,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_endpoints.scan_imports",
                new_callable=AsyncMock,
                return_value={"imports": [], "import_count": 0, "invalid_imports": []},
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_endpoints.validate_scanned_imports",
                new_callable=AsyncMock,
                return_value={
                    "imports": [],
                    "import_count": 0,
                    "invalid_imports": [],
                },
            ),
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_endpoints import (
                scan_repository,
            )

            result = await scan_repository(self._make_request("imports"), tenant_id="t1")
        assert result.action == "imports"

    async def test_validation_scan(self):
        container = ContainerInfo(container_name="ctr-1", status="running")
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[container])

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_endpoints.get_container_manager",
                return_value=mgr,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_endpoints.validate_code",
                new_callable=AsyncMock,
                return_value={
                    "validation_results": [],
                    "valid_files": 0,
                    "invalid_files": 0,
                },
            ),
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_endpoints import (
                scan_repository,
            )

            result = await scan_repository(self._make_request("validation"), tenant_id="t1")
        assert result.action == "validation"

    async def test_full_scan(self):
        container = ContainerInfo(container_name="ctr-1", status="running")
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[container])

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_endpoints.get_container_manager",
                return_value=mgr,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_endpoints.scan_dependencies",
                new_callable=AsyncMock,
                return_value={
                    "dependencies": [],
                    "dependency_count": 0,
                },
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_endpoints.scan_imports",
                new_callable=AsyncMock,
                return_value={"imports": [], "import_count": 0},
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_endpoints.validate_code",
                new_callable=AsyncMock,
                return_value={
                    "validation_results": [],
                    "valid_files": 0,
                    "invalid_files": 0,
                },
            ),
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_endpoints import (
                scan_repository,
            )

            result = await scan_repository(self._make_request("full_scan"), tenant_id="t1")
        assert result.action == "full_scan"

    async def test_unexpected_error_raises_500(self):
        container = ContainerInfo(container_name="ctr-1", status="running")
        mgr = AsyncMock()
        mgr.list_containers = AsyncMock(return_value=[container])

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_endpoints.get_container_manager",
                return_value=mgr,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_endpoints.scan_dependencies",
                new_callable=AsyncMock,
                side_effect=RuntimeError("crash"),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_endpoints import (
                scan_repository,
            )

            await scan_repository(self._make_request("dependencies"), tenant_id="t1")
        assert exc_info.value.status_code == 500


# ============================================================================
# code_validator.py — validate_code & validate_code_v2
# ============================================================================
from app.api.endpoints.repo_scanner_modules.code_validator import (
    validate_code,
    validate_code_v2,
)

_EXEC_IN_CONTAINER = "app.api.endpoints.repo_scanner_modules.code_validator.execute_in_container"
_GET_CM_VALIDATOR = "app.api.endpoints.repo_scanner_modules.code_validator.get_container_manager"


class TestValidateCode:
    """Tests for validate_code function."""

    async def test_write_script_failure(self):
        with patch(
            _EXEC_IN_CONTAINER,
            new_callable=AsyncMock,
            return_value={"success": False, "output": "write failed"},
        ):
            result = await validate_code("ctr-1")
        assert result["validation_results"] == []
        assert "Failed to create validation script" in result["error"]

    async def test_successful_validation(self):
        call_count = 0

        async def mock_exec(container_name, cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": True, "output": ""}  # write script
            return {
                "success": True,
                "output": json.dumps(
                    [
                        {
                            "file_path": "/workspace/main.py",
                            "is_valid": True,
                            "issues": [],
                            "warnings": [],
                            "line_count": 10,
                        },
                        {
                            "file_path": "/workspace/bad.py",
                            "is_valid": False,
                            "issues": ["SyntaxError at line 5: invalid syntax"],
                            "warnings": [],
                            "line_count": 20,
                        },
                    ]
                ),
            }

        with patch(_EXEC_IN_CONTAINER, new_callable=AsyncMock, side_effect=mock_exec):
            result = await validate_code("ctr-1")

        assert result["valid_files"] == 1
        assert result["invalid_files"] == 1
        assert len(result["validation_results"]) == 2

    async def test_target_files_passed_to_command(self):
        call_count = 0

        async def mock_exec(container_name, cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": True, "output": ""}
            # Verify target files are in command
            assert "src/main.py" in cmd
            return {"success": True, "output": "[]"}

        with patch(_EXEC_IN_CONTAINER, new_callable=AsyncMock, side_effect=mock_exec):
            result = await validate_code("ctr-1", target_files=["src/main.py"])
        assert result["validation_results"] == []
        assert result["valid_files"] == 0

    async def test_json_parse_error(self):
        call_count = 0

        async def mock_exec(container_name, cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": True, "output": ""}
            return {"success": True, "output": "not valid json"}

        with patch(_EXEC_IN_CONTAINER, new_callable=AsyncMock, side_effect=mock_exec):
            result = await validate_code("ctr-1")
        assert result["validation_results"] == []
        assert "Failed to parse output" in result["error"]

    async def test_exec_failure(self):
        call_count = 0

        async def mock_exec(container_name, cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": True, "output": ""}
            return {"success": False, "output": "execution error"}

        with patch(_EXEC_IN_CONTAINER, new_callable=AsyncMock, side_effect=mock_exec):
            result = await validate_code("ctr-1")
        assert result["validation_results"] == []
        assert result["error"] == "execution error"


class TestValidateCodeV2:
    """Tests for validate_code_v2 function."""

    async def test_target_files_filters_py(self):
        mock_mgr = AsyncMock()
        mock_mgr.execute_command = AsyncMock(
            return_value={
                "success": True,
                "output": "x = 1\n",
            }
        )
        with patch(_GET_CM_VALIDATOR, return_value=mock_mgr):
            result = await validate_code_v2("ctr-1", target_files=["main.py", "data.txt"])
        # Only main.py should be validated (data.txt filtered out)
        assert result["valid_files"] + result["invalid_files"] <= 1

    async def test_find_python_files_failure(self):
        mock_mgr = AsyncMock()
        mock_mgr.execute_command = AsyncMock(
            return_value={"success": False, "error": "find failed"}
        )
        with patch(_GET_CM_VALIDATOR, return_value=mock_mgr):
            result = await validate_code_v2("ctr-1")
        assert result["valid_files"] == 0
        assert result["validation_results"] == []

    async def test_valid_file_with_long_line_warning(self):
        long_line = "x = " + "a" * 200
        mock_mgr = AsyncMock()

        async def mock_exec(container, cmd, **kwargs):
            if "find" in cmd:
                return {"success": True, "output": "./main.py\n"}
            if "cat" in cmd:
                return {"success": True, "output": f"x = 1\n{long_line}\n"}
            return {"success": False, "output": ""}

        mock_mgr.execute_command = AsyncMock(side_effect=mock_exec)
        with patch(_GET_CM_VALIDATOR, return_value=mock_mgr):
            result = await validate_code_v2("ctr-1")
        assert result["valid_files"] == 1
        # Should have at least 1 warning about long line
        validation = result["validation_results"][0]
        assert len(validation["warnings"]) >= 1
        assert "too long" in validation["warnings"][0].lower()

    async def test_invalid_syntax_file(self):
        mock_mgr = AsyncMock()

        async def mock_exec(container, cmd, **kwargs):
            if "find" in cmd:
                return {"success": True, "output": "./bad.py\n"}
            if "cat" in cmd:
                return {"success": True, "output": "def foo(\n"}
            return {"success": False, "output": ""}

        mock_mgr.execute_command = AsyncMock(side_effect=mock_exec)
        with patch(_GET_CM_VALIDATOR, return_value=mock_mgr):
            result = await validate_code_v2("ctr-1")
        assert result["invalid_files"] == 1
        validation = result["validation_results"][0]
        assert validation["is_valid"] is False
        assert validation["syntax_valid"] is False
        assert len(validation["issues"]) >= 1

    async def test_cat_failure_skips_file(self):
        mock_mgr = AsyncMock()

        async def mock_exec(container, cmd, **kwargs):
            if "find" in cmd:
                return {"success": True, "output": "./main.py\n"}
            if "cat" in cmd:
                return {"success": False, "output": "permission denied"}
            return {"success": False, "output": ""}

        mock_mgr.execute_command = AsyncMock(side_effect=mock_exec)
        with patch(_GET_CM_VALIDATOR, return_value=mock_mgr):
            result = await validate_code_v2("ctr-1")
        assert result["valid_files"] == 0
        assert result["invalid_files"] == 0

    async def test_value_error_during_file_processing(self):
        mock_mgr = AsyncMock()

        async def mock_exec(container, cmd, **kwargs):
            if "find" in cmd:
                return {"success": True, "output": "./main.py\n"}
            if "cat" in cmd:
                raise ValueError("mock value error")
            return {"success": False, "output": ""}

        mock_mgr.execute_command = AsyncMock(side_effect=mock_exec)
        with patch(_GET_CM_VALIDATOR, return_value=mock_mgr):
            result = await validate_code_v2("ctr-1")
        assert result["invalid_files"] == 1
        validation = result["validation_results"][0]
        assert "ProcessingError" in validation["issues"][0]

    async def test_container_error_returns_error(self):
        with patch(_GET_CM_VALIDATOR, side_effect=ValueError("no container")):
            result = await validate_code_v2("ctr-1")
        assert "error" in result
        assert result["valid_files"] == 0


# ============================================================================
# dependency_scanner.py — scan_dependencies & scan_dependencies_v2
# ============================================================================
from app.api.endpoints.repo_scanner_modules.dependency_scanner import (
    scan_dependencies as mod_scan_deps,
)
from app.api.endpoints.repo_scanner_modules.dependency_scanner import (
    scan_dependencies_v2 as mod_scan_deps_v2,
)

_EXEC_DEP = "app.api.endpoints.repo_scanner_modules.dependency_scanner.execute_in_container"
_GET_CM_DEP = "app.api.endpoints.repo_scanner_modules.dependency_scanner.get_container_manager"


class TestModuleScanDependencies:
    """Tests for dependency_scanner.scan_dependencies."""

    async def test_write_script_failure(self):
        with patch(
            _EXEC_DEP,
            new_callable=AsyncMock,
            return_value={"success": False, "output": "write failed"},
        ):
            result = await mod_scan_deps("ctr-1")
        assert result["dependencies"] == []
        assert "Failed to create scan script" in result["error"]

    async def test_successful_scan(self):
        call_count = 0

        async def mock_exec(container, cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": True, "output": ""}
            return {
                "success": True,
                "output": json.dumps(
                    [
                        {"name": "requests", "version": "2.28.1", "source": "pip"},
                        {"name": "flask", "version": "2.0.0", "source": "requirements"},
                    ]
                ),
            }

        with patch(_EXEC_DEP, new_callable=AsyncMock, side_effect=mock_exec):
            result = await mod_scan_deps("ctr-1")
        assert result["dependency_count"] == 2
        assert len(result["dependencies"]) == 2
        assert result["dependencies"][0].name == "requests"
        assert result["dependencies"][1].source == "requirements"

    async def test_json_parse_failure(self):
        call_count = 0

        async def mock_exec(container, cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": True, "output": ""}
            return {"success": True, "output": "not json"}

        with patch(_EXEC_DEP, new_callable=AsyncMock, side_effect=mock_exec):
            result = await mod_scan_deps("ctr-1")
        assert result["dependencies"] == []
        assert "Failed to parse output" in result["error"]

    async def test_exec_failure(self):
        call_count = 0

        async def mock_exec(container, cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": True, "output": ""}
            return {"success": False, "output": "script error"}

        with patch(_EXEC_DEP, new_callable=AsyncMock, side_effect=mock_exec):
            result = await mod_scan_deps("ctr-1")
        assert result["dependencies"] == []
        assert result["error"] == "script error"


class TestModuleScanDependenciesV2:
    """Tests for dependency_scanner.scan_dependencies_v2."""

    async def test_pip_and_requirements_scan(self):
        mock_mgr = AsyncMock()

        async def mock_exec(container, cmd, **kwargs):
            if "pip list" in cmd:
                return {
                    "success": True,
                    "output": json.dumps(
                        [
                            {"name": "requests", "version": "2.28.1"},
                        ]
                    ),
                }
            if "cat requirements.txt" in cmd:
                return {"success": True, "output": "flask==2.0.0\nnumpy>=1.21\n# comment\n"}
            if "cat pyproject.toml" in cmd:
                return {"success": True, "output": "[tool.poetry]"}
            return {"success": False, "output": ""}

        mock_mgr.execute_command = AsyncMock(side_effect=mock_exec)
        with patch(_GET_CM_DEP, return_value=mock_mgr):
            result = await mod_scan_deps_v2("ctr-1")

        assert result["dependency_count"] >= 3
        assert "pip" in result["sources"]
        assert "requirements.txt" in result["sources"]
        assert "pyproject.toml" in result["sources"]

    async def test_pip_json_in_mixed_output(self):
        """Test JSON extraction when pip output has non-JSON prefix."""
        mock_mgr = AsyncMock()

        async def mock_exec(container, cmd, **kwargs):
            if "pip list" in cmd:
                return {
                    "success": True,
                    "output": 'WARNING: some warning\n[{"name": "pkg1", "version": "1.0"}]',
                }
            return {"success": False, "output": ""}

        mock_mgr.execute_command = AsyncMock(side_effect=mock_exec)
        with patch(_GET_CM_DEP, return_value=mock_mgr):
            result = await mod_scan_deps_v2("ctr-1")
        assert result["dependency_count"] == 1
        assert result["dependencies"][0]["name"] == "pkg1"

    async def test_pip_json_decode_error(self):
        """Test handling of bad JSON from pip."""
        mock_mgr = AsyncMock()

        async def mock_exec(container, cmd, **kwargs):
            if "pip list" in cmd:
                return {"success": True, "output": "totally not json at all"}
            return {"success": False, "output": ""}

        mock_mgr.execute_command = AsyncMock(side_effect=mock_exec)
        with patch(_GET_CM_DEP, return_value=mock_mgr):
            result = await mod_scan_deps_v2("ctr-1")
        # Should not crash, just skip pip
        assert result["dependency_count"] == 0

    async def test_value_error_returns_error(self):
        with patch(_GET_CM_DEP, side_effect=ValueError("no container")):
            result = await mod_scan_deps_v2("ctr-1")
        assert "error" in result
        assert result["dependency_count"] == 0

    async def test_all_sources_fail(self):
        """All sources fail gracefully."""
        mock_mgr = AsyncMock()
        mock_mgr.execute_command = AsyncMock(return_value={"success": False, "output": "fail"})
        with patch(_GET_CM_DEP, return_value=mock_mgr):
            result = await mod_scan_deps_v2("ctr-1")
        assert result["dependency_count"] == 0
        assert result["sources"] == []


# ============================================================================
# import_scanner.py — scan_imports & validate_scanned_imports
# ============================================================================
from app.api.endpoints.repo_scanner_modules.import_scanner import (
    scan_imports as mod_scan_imports,
)
from app.api.endpoints.repo_scanner_modules.import_scanner import (
    validate_scanned_imports,
)

_EXEC_IMP = "app.api.endpoints.repo_scanner_modules.import_scanner.execute_in_container"
_GET_CM_IMP = "app.api.endpoints.repo_scanner_modules.import_scanner.get_container_manager"


class TestModuleScanImports:
    """Tests for import_scanner.scan_imports."""

    async def test_write_script_failure(self):
        with patch(
            _EXEC_IMP,
            new_callable=AsyncMock,
            return_value={"success": False, "output": "write failed"},
        ):
            result = await mod_scan_imports("ctr-1")
        assert result["imports"] == []
        assert "Failed to create scan script" in result["error"]

    async def test_successful_scan(self):
        call_count = 0

        async def mock_exec(container, cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": True, "output": ""}
            return {
                "success": True,
                "output": json.dumps(
                    [
                        {
                            "module": "os",
                            "name": None,
                            "alias": None,
                            "import_type": "import",
                            "file_path": "/workspace/main.py",
                            "line_number": 1,
                            "is_valid": True,
                        },
                        {
                            "module": "sys",
                            "name": "path",
                            "alias": None,
                            "import_type": "from_import",
                            "file_path": "/workspace/main.py",
                            "line_number": 2,
                            "is_valid": False,
                        },
                    ]
                ),
            }

        with patch(_EXEC_IMP, new_callable=AsyncMock, side_effect=mock_exec):
            result = await mod_scan_imports("ctr-1")
        assert result["import_count"] == 2
        assert len(result["imports"]) == 2
        assert len(result["invalid_imports"]) == 1

    async def test_target_files_passed(self):
        call_count = 0

        async def mock_exec(container, cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": True, "output": ""}
            assert "src/main.py" in cmd
            return {"success": True, "output": "[]"}

        with patch(_EXEC_IMP, new_callable=AsyncMock, side_effect=mock_exec):
            result = await mod_scan_imports("ctr-1", target_files=["src/main.py"])
        assert result["import_count"] == 0

    async def test_json_parse_failure(self):
        call_count = 0

        async def mock_exec(container, cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": True, "output": ""}
            return {"success": True, "output": "invalid json"}

        with patch(_EXEC_IMP, new_callable=AsyncMock, side_effect=mock_exec):
            result = await mod_scan_imports("ctr-1")
        assert result["imports"] == []
        assert "Failed to parse output" in result["error"]

    async def test_exec_failure(self):
        call_count = 0

        async def mock_exec(container, cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"success": True, "output": ""}
            return {"success": False, "output": "execution error"}

        with patch(_EXEC_IMP, new_callable=AsyncMock, side_effect=mock_exec):
            result = await mod_scan_imports("ctr-1")
        assert result["imports"] == []
        assert result["error"] == "execution error"


class TestValidateScannedImports:
    """Tests for import_scanner.validate_scanned_imports."""

    async def test_successful_validation(self):
        mock_mgr = AsyncMock()
        mock_mgr.get_container = AsyncMock(return_value=MagicMock())

        mock_validation_result = MagicMock()
        mock_validation_result.valid_imports = [
            {"module": "os", "name": None},
        ]
        mock_validation_result.invalid_imports = [
            {"module": "nonexistent", "name": None},
        ]
        mock_validation_result.stdlib_imports = [{"module": "os"}]
        mock_validation_result.third_party_imports = []
        mock_validation_result.local_imports = []
        mock_validation_result.missing_dependencies = ["nonexistent"]
        mock_validation_result.suggestions = ["Install nonexistent"]

        mock_validation_service = MagicMock()
        mock_validation_service.validate_imports = AsyncMock(return_value=mock_validation_result)

        with (
            patch(_GET_CM_IMP, return_value=mock_mgr),
            patch(
                "app.api.endpoints.repo_scanner_modules.import_scanner.ImportValidationService",
                return_value=mock_validation_service,
            ),
        ):
            imports = [
                {"module": "os", "name": None, "file_path": "main.py", "line_number": 1},
                {"module": "nonexistent", "name": None, "file_path": "main.py", "line_number": 2},
            ]
            result = await validate_scanned_imports("ctr-1", imports)

        assert result["import_count"] == 2
        assert result["validation_summary"]["valid_count"] == 1
        assert result["validation_summary"]["invalid_count"] == 1
        assert "nonexistent" in result["validation_summary"]["missing_dependencies"]

    async def test_container_not_found_uses_default(self):
        mock_mgr = AsyncMock()
        mock_mgr.get_container = AsyncMock(side_effect=ValueError("not found"))

        mock_validation_result = MagicMock()
        mock_validation_result.valid_imports = []
        mock_validation_result.invalid_imports = []
        mock_validation_result.stdlib_imports = []
        mock_validation_result.third_party_imports = []
        mock_validation_result.local_imports = []
        mock_validation_result.missing_dependencies = []
        mock_validation_result.suggestions = []

        mock_validation_service = MagicMock()
        mock_validation_service.validate_imports = AsyncMock(return_value=mock_validation_result)

        with (
            patch(_GET_CM_IMP, return_value=mock_mgr),
            patch(
                "app.api.endpoints.repo_scanner_modules.import_scanner.ImportValidationService",
                return_value=mock_validation_service,
            ),
        ):
            result = await validate_scanned_imports("ctr-1", [])
        assert result["import_count"] == 0

    async def test_error_returns_original_imports(self):
        with patch(_GET_CM_IMP, side_effect=ValueError("broken")):
            imports = [{"module": "os", "name": None}]
            result = await validate_scanned_imports("ctr-1", imports)
        assert result["import_count"] == 1
        assert "validation_error" in result


# ============================================================================
# Multi-language scanning — file_discovery.discover_source_files
# ============================================================================


class TestDiscoverSourceFiles:
    """Tests for file_discovery.discover_source_files (all supported languages)."""

    async def test_success_returns_polyglot_files(self):
        from app.api.endpoints.repo_scanner_modules.file_discovery import (
            discover_source_files,
        )

        with patch(
            "app.api.endpoints.repo_scanner_modules.file_discovery.execute_in_container",
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "output": "/workspace/main.py\n/workspace/app.ts\n/workspace/svc.go\n",
            },
        ):
            result = await discover_source_files("ctr-1")

        assert result["file_count"] == 3
        assert "/workspace/app.ts" in result["source_files"]
        assert "/workspace/svc.go" in result["source_files"]

    async def test_exclusion_filtering(self):
        from app.api.endpoints.repo_scanner_modules.file_discovery import (
            discover_source_files,
        )

        with patch(
            "app.api.endpoints.repo_scanner_modules.file_discovery.execute_in_container",
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "output": "/workspace/node_modules/x.js\n/workspace/src/bar.ts\n",
            },
        ):
            result = await discover_source_files("ctr-1")

        assert result["file_count"] == 1
        assert result["source_files"] == ["/workspace/src/bar.ts"]

    async def test_failure_returns_error(self):
        from app.api.endpoints.repo_scanner_modules.file_discovery import (
            discover_source_files,
        )

        with patch(
            "app.api.endpoints.repo_scanner_modules.file_discovery.execute_in_container",
            new_callable=AsyncMock,
            return_value={"success": False, "output": "find: error"},
        ):
            result = await discover_source_files("ctr-1")

        assert result["file_count"] == 0
        assert "error" in result


# ============================================================================
# Multi-language scanning — multi_language_scan
# ============================================================================


class TestPartitionByPython:
    """Tests for multi_language_scan.partition_by_python."""

    def test_splits_python_from_other(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            partition_by_python,
        )

        python_files, other_files = partition_by_python(
            ["a.py", "b.ts", "c.go", "d.pyw", "readme.md", "e.rs"]
        )

        assert sorted(python_files) == ["a.py", "d.pyw"]
        assert sorted(other_files) == ["b.ts", "c.go", "e.rs"]
        # Unsupported extensions (.md) are dropped from both lists.
        assert "readme.md" not in python_files
        assert "readme.md" not in other_files


class TestScanImportsMultilang:
    """Tests for multi_language_scan.scan_imports_multilang."""

    async def test_extracts_typescript_imports(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            scan_imports_multilang,
        )

        ts_source = 'import { readFile } from "fs";\nimport * as path from "path";\n'

        with patch(
            "app.api.endpoints.repo_scanner_modules.multi_language_scan.execute_in_container",
            new_callable=AsyncMock,
            return_value={"success": True, "output": ts_source},
        ):
            imports = await scan_imports_multilang("ctr-1", ["/workspace/app.ts"])

        modules = {imp.module for imp in imports}
        assert {"fs", "path"} <= modules
        assert all(imp.file_path == "/workspace/app.ts" for imp in imports)

    async def test_extracts_go_imports(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            scan_imports_multilang,
        )

        go_source = 'package main\n\nimport (\n\t"fmt"\n\t"os"\n)\n'

        with patch(
            "app.api.endpoints.repo_scanner_modules.multi_language_scan.execute_in_container",
            new_callable=AsyncMock,
            return_value={"success": True, "output": go_source},
        ):
            imports = await scan_imports_multilang("ctr-1", ["/workspace/svc.go"])

        modules = {imp.module for imp in imports}
        assert {"fmt", "os"} <= modules

    async def test_unreadable_file_is_skipped(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            scan_imports_multilang,
        )

        with patch(
            "app.api.endpoints.repo_scanner_modules.multi_language_scan.execute_in_container",
            new_callable=AsyncMock,
            return_value={"success": False, "output": "cat: no such file"},
        ):
            imports = await scan_imports_multilang("ctr-1", ["/workspace/gone.ts"])

        assert imports == []


class TestValidateCodeMultilang:
    """Tests for multi_language_scan.validate_code_multilang."""

    async def test_valid_typescript_file(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            validate_code_multilang,
        )

        with patch(
            "app.api.endpoints.repo_scanner_modules.multi_language_scan.execute_in_container",
            new_callable=AsyncMock,
            return_value={
                "success": True,
                "output": "function f(): number { return 1; }\n",
            },
        ):
            results = await validate_code_multilang("ctr-1", ["/workspace/ok.ts"])

        assert len(results) == 1
        assert results[0].is_valid is True
        assert results[0].file_path == "/workspace/ok.ts"

    async def test_long_line_warning(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            validate_code_multilang,
        )

        long_line = "const x = " + '"' + ("a" * 200) + '";\n'

        with patch(
            "app.api.endpoints.repo_scanner_modules.multi_language_scan.execute_in_container",
            new_callable=AsyncMock,
            return_value={"success": True, "output": long_line},
        ):
            results = await validate_code_multilang("ctr-1", ["/workspace/long.ts"])

        assert results[0].warnings
        assert "too long" in results[0].warnings[0]

    async def test_unreadable_file_reported(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            validate_code_multilang,
        )

        with patch(
            "app.api.endpoints.repo_scanner_modules.multi_language_scan.execute_in_container",
            new_callable=AsyncMock,
            return_value={"success": False, "output": "cat: missing"},
        ):
            results = await validate_code_multilang("ctr-1", ["/workspace/gone.go"])

        assert len(results) == 1
        assert results[0].is_valid is False
        assert results[0].issues


class TestScanImportsPolyglotDispatch:
    """scanner_core.scan_imports dispatches Python and non-Python correctly."""

    async def test_target_files_split_between_paths(self):
        ts_source = 'import { readFile } from "fs";\n'
        python_imports_json = json.dumps(
            [
                {
                    "module": "os",
                    "name": None,
                    "alias": None,
                    "import_type": "import",
                    "file_path": "main.py",
                    "line_number": 1,
                    "is_valid": True,
                }
            ]
        )

        async def mock_core_exec(container, cmd, **kwargs):
            # Python in-container script execution.
            if "scan_imports" in cmd and " -c " not in cmd:
                return {"success": True, "output": python_imports_json}
            return {"success": True, "output": "ok"}

        async def mock_ml_exec(container, cmd, **kwargs):
            # Non-Python files are read out via ``cat``.
            return {"success": True, "output": ts_source}

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_core.execute_in_container",
                new_callable=AsyncMock,
                side_effect=mock_core_exec,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.multi_language_scan.execute_in_container",
                new_callable=AsyncMock,
                side_effect=mock_ml_exec,
            ),
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                scan_imports as core_scan_imports,
            )

            result = await core_scan_imports("ctr-1", target_files=["main.py", "app.ts"])

        modules = {imp.module for imp in result["imports"]}
        # One import from the Python path, one from the framework path.
        assert "os" in modules
        assert "fs" in modules

    async def test_pure_python_repo_unchanged(self):
        """A repo with only Python files never touches the framework path."""
        python_imports_json = json.dumps([])

        async def mock_core_exec(container, cmd, **kwargs):
            if "scan_imports" in cmd and " -c " not in cmd:
                return {"success": True, "output": python_imports_json}
            return {"success": True, "output": "ok"}

        async def mock_discovery_exec(container, cmd, **kwargs):
            # discover_source_files finds only Python files in this repo.
            return {"success": True, "output": "/workspace/main.py\n"}

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_core.execute_in_container",
                new_callable=AsyncMock,
                side_effect=mock_core_exec,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.file_discovery.execute_in_container",
                new_callable=AsyncMock,
                side_effect=mock_discovery_exec,
            ),
        ):
            from app.api.endpoints.repo_scanner_modules.scanner_core import (
                scan_imports as core_scan_imports,
            )

            result = await core_scan_imports("ctr-1")

        assert result["import_count"] == 0
        assert "error" not in result


# ---------------------------------------------------------------------------
# Multi-language dependency-manifest parsing — multi_language_scan
# ---------------------------------------------------------------------------


class TestManifestParsers:
    """Tests for the pure stdlib manifest parsers (no container needed)."""

    def test_parse_package_json(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            _parse_package_json,
        )

        content = json.dumps(
            {
                "name": "demo",
                "dependencies": {"express": "^4.18.0", "lodash": "4.17.21"},
                "devDependencies": {"jest": "^29.0.0"},
            }
        )
        deps = _parse_package_json(content)
        by_name = {d.name: d for d in deps}
        # Caret ranges are stripped to satisfy the VERSION_PATTERN charset.
        assert by_name["express"].version == "4.18.0"
        assert by_name["lodash"].version == "4.17.21"
        assert by_name["jest"].version == "29.0.0"
        assert all(d.source == "package.json" and d.is_valid for d in deps)

    def test_parse_package_json_malformed_returns_empty(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            _parse_package_json,
        )

        assert _parse_package_json("{not json") == []

    def test_parse_go_mod_block_and_inline(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            _parse_go_mod,
        )

        content = (
            "module example.com/demo\n\n"
            "go 1.22\n\n"
            "require github.com/gin-gonic/gin v1.9.1\n\n"
            "require (\n"
            "\tgithub.com/stretchr/testify v1.8.4\n"
            "\tgolang.org/x/sync v0.5.0 // indirect\n"
            ")\n"
        )
        deps = _parse_go_mod(content)
        by_name = {d.name: d.version for d in deps}
        assert by_name["github.com/gin-gonic/gin"] == "v1.9.1"
        assert by_name["github.com/stretchr/testify"] == "v1.8.4"
        assert by_name["golang.org/x/sync"] == "v0.5.0"
        assert all(d.source == "go.mod" for d in deps)

    def test_parse_cargo_toml_string_and_table(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            _parse_cargo_toml,
        )

        content = (
            '[package]\nname = "demo"\n\n'
            "[dependencies]\n"
            'serde = "1.0"\n'
            'tokio = { version = "1.35", features = ["full"] }\n\n'
            "[dev-dependencies]\n"
            'criterion = "0.5"\n'
        )
        deps = _parse_cargo_toml(content)
        by_name = {d.name: d.version for d in deps}
        assert by_name["serde"] == "1.0"
        assert by_name["tokio"] == "1.35"
        assert by_name["criterion"] == "0.5"
        assert all(d.source == "Cargo.toml" for d in deps)

    def test_parse_pom_xml(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            _parse_pom_xml,
        )

        content = (
            '<project xmlns="http://maven.apache.org/POM/4.0.0">\n'
            "  <dependencies>\n"
            "    <dependency>\n"
            "      <groupId>com.google.guava</groupId>\n"
            "      <artifactId>guava</artifactId>\n"
            "      <version>32.1.3-jre</version>\n"
            "    </dependency>\n"
            "    <dependency>\n"
            "      <groupId>junit</groupId>\n"
            "      <artifactId>junit</artifactId>\n"
            "      <version>4.13.2</version>\n"
            "    </dependency>\n"
            "  </dependencies>\n"
            "</project>\n"
        )
        deps = _parse_pom_xml(content)
        by_name = {d.name: d.version for d in deps}
        assert by_name["com.google.guava:guava"] == "32.1.3-jre"
        assert by_name["junit:junit"] == "4.13.2"
        assert all(d.source == "pom.xml" for d in deps)

    def test_parse_pom_xml_malformed_returns_empty(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            _parse_pom_xml,
        )

        assert _parse_pom_xml("<project><unclosed>") == []


class TestScanDependenciesMultilang:
    """Tests for multi_language_scan.scan_dependencies_multilang."""

    async def test_discovers_and_parses_manifests(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            scan_dependencies_multilang,
        )

        package_json = json.dumps({"dependencies": {"express": "^4.18.0"}})
        go_mod = "module demo\n\nrequire github.com/gin-gonic/gin v1.9.1\n"

        async def mock_exec(container, cmd, **kwargs):
            if cmd.startswith("find "):
                return {
                    "success": True,
                    "output": "/workspace/package.json\n/workspace/svc/go.mod\n",
                }
            if "package.json" in cmd:
                return {"success": True, "output": package_json}
            if "go.mod" in cmd:
                return {"success": True, "output": go_mod}
            return {"success": False, "output": "unexpected"}

        with patch(
            "app.api.endpoints.repo_scanner_modules.multi_language_scan.execute_in_container",
            new_callable=AsyncMock,
            side_effect=mock_exec,
        ):
            deps = await scan_dependencies_multilang("ctr-1")

        by_name = {d.name: d for d in deps}
        assert by_name["express"].source == "package.json"
        assert by_name["github.com/gin-gonic/gin"].source == "go.mod"

    async def test_no_manifests_returns_empty(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            scan_dependencies_multilang,
        )

        with patch(
            "app.api.endpoints.repo_scanner_modules.multi_language_scan.execute_in_container",
            new_callable=AsyncMock,
            return_value={"success": True, "output": ""},
        ):
            deps = await scan_dependencies_multilang("ctr-1")

        assert deps == []

    async def test_unreadable_manifest_is_skipped(self):
        from app.api.endpoints.repo_scanner_modules.multi_language_scan import (
            scan_dependencies_multilang,
        )

        async def mock_exec(container, cmd, **kwargs):
            if cmd.startswith("find "):
                return {"success": True, "output": "/workspace/package.json\n"}
            return {"success": False, "output": "cat: no such file"}

        with patch(
            "app.api.endpoints.repo_scanner_modules.multi_language_scan.execute_in_container",
            new_callable=AsyncMock,
            side_effect=mock_exec,
        ):
            deps = await scan_dependencies_multilang("ctr-1")

        assert deps == []


class TestScanDependenciesCoreMerge:
    """scanner_core.scan_dependencies merges Python + multi-language deps."""

    async def test_merges_python_and_manifest_deps(self):
        from app.api.endpoints.repo_scanner_modules.scanner_core import (
            scan_dependencies as core_scan_dependencies,
        )

        package_json = json.dumps({"dependencies": {"express": "^4.18.0"}})

        async def mock_core_exec(container, cmd, **kwargs):
            # Script write, then the python3 execution returning pip deps.
            if cmd.startswith("python3 -I -P /tmp/scan_dependencies.py /workspace"):
                return {
                    "success": True,
                    "output": json.dumps(
                        [{"name": "requests", "version": "2.31.0", "source": "pip"}]
                    ),
                }
            return {"success": True, "output": ""}

        async def mock_ml_exec(container, cmd, **kwargs):
            if cmd.startswith("find "):
                return {"success": True, "output": "/workspace/package.json\n"}
            if "package.json" in cmd:
                return {"success": True, "output": package_json}
            return {"success": False, "output": "unexpected"}

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_core.execute_in_container",
                new_callable=AsyncMock,
                side_effect=mock_core_exec,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.multi_language_scan.execute_in_container",
                new_callable=AsyncMock,
                side_effect=mock_ml_exec,
            ),
        ):
            result = await core_scan_dependencies("ctr-1")

        sources = {d.source for d in result["dependencies"]}
        assert {"pip", "package.json"} <= sources
        assert result["dependency_count"] == len(result["dependencies"])
        assert "error" not in result

    async def test_python_path_unchanged_for_pure_python_repo(self):
        from app.api.endpoints.repo_scanner_modules.scanner_core import (
            scan_dependencies as core_scan_dependencies,
        )

        async def mock_core_exec(container, cmd, **kwargs):
            if cmd.startswith("python3 -I -P /tmp/scan_dependencies.py /workspace"):
                return {
                    "success": True,
                    "output": json.dumps(
                        [{"name": "requests", "version": "2.31.0", "source": "pip"}]
                    ),
                }
            return {"success": True, "output": ""}

        async def mock_ml_exec(container, cmd, **kwargs):
            # No non-Python manifests in this repo.
            return {"success": True, "output": ""}

        with (
            patch(
                "app.api.endpoints.repo_scanner_modules.scanner_core.execute_in_container",
                new_callable=AsyncMock,
                side_effect=mock_core_exec,
            ),
            patch(
                "app.api.endpoints.repo_scanner_modules.multi_language_scan.execute_in_container",
                new_callable=AsyncMock,
                side_effect=mock_ml_exec,
            ),
        ):
            result = await core_scan_dependencies("ctr-1")

        assert result["dependency_count"] == 1
        assert result["dependencies"][0].name == "requests"
        assert result["dependencies"][0].source == "pip"
        assert "error" not in result
