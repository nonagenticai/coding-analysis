"""Tests for repo_scanner_modules: code_validator, dependency_scanner, import_scanner.

These modules execute scripts inside containers. Tests mock the container execution
and validate parsing/processing logic.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.endpoints.repo_scanner_modules.code_validator import (
    validate_code,
    validate_code_v2,
)
from app.api.endpoints.repo_scanner_modules.dependency_scanner import (
    scan_dependencies,
    scan_dependencies_v2,
)
from app.api.endpoints.repo_scanner_modules.import_scanner import (
    scan_imports,
    validate_scanned_imports,
)

_EXEC_CV = "app.api.endpoints.repo_scanner_modules.code_validator.execute_in_container"
_EXEC_DS = "app.api.endpoints.repo_scanner_modules.dependency_scanner.execute_in_container"
_EXEC_IS = "app.api.endpoints.repo_scanner_modules.import_scanner.execute_in_container"
_GET_CM = "app.api.endpoints.repo_scanner_modules.code_validator.get_container_manager"
_GET_CM_DEP = "app.api.endpoints.repo_scanner_modules.dependency_scanner.get_container_manager"
_GET_CM_IMP = "app.api.endpoints.repo_scanner_modules.import_scanner.get_container_manager"


# ===================================================================
# validate_code tests
# ===================================================================


class TestValidateCode:
    """Tests for validate_code function."""

    @pytest.mark.asyncio
    async def test_script_write_failure_returns_error(self):
        with patch(_EXEC_CV, new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"success": False, "output": "write failed"}
            result = await validate_code("test-container")
            assert result["validation_results"] == []
            assert "error" in result

    @pytest.mark.asyncio
    async def test_successful_validation(self):
        validation_data = [
            {
                "file_path": "/workspace/main.py",
                "is_valid": True,
                "issues": [],
                "warnings": [],
                "line_count": 50,
            }
        ]
        with patch(_EXEC_CV, new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = [
                {"success": True, "output": ""},  # write script
                {"success": True, "output": json.dumps(validation_data)},  # execute
            ]
            result = await validate_code("test-container")
            assert result["valid_files"] == 1
            assert result["invalid_files"] == 0
            assert len(result["validation_results"]) == 1

    @pytest.mark.asyncio
    async def test_invalid_file_counts(self):
        validation_data = [
            {
                "file_path": "/workspace/bad.py",
                "is_valid": False,
                "issues": ["SyntaxError at line 5: invalid syntax"],
                "warnings": [],
                "line_count": 10,
            },
            {
                "file_path": "/workspace/good.py",
                "is_valid": True,
                "issues": [],
                "warnings": [],
                "line_count": 20,
            },
        ]
        with patch(_EXEC_CV, new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = [
                {"success": True, "output": ""},
                {"success": True, "output": json.dumps(validation_data)},
            ]
            result = await validate_code("test-container")
            assert result["valid_files"] == 1
            assert result["invalid_files"] == 1

    @pytest.mark.asyncio
    async def test_json_parse_failure(self):
        with patch(_EXEC_CV, new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = [
                {"success": True, "output": ""},
                {"success": True, "output": "NOT JSON"},
            ]
            result = await validate_code("test-container")
            assert result["validation_results"] == []
            assert "error" in result

    @pytest.mark.asyncio
    async def test_execution_failure(self):
        with patch(_EXEC_CV, new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = [
                {"success": True, "output": ""},
                {"success": False, "output": "command failed"},
            ]
            result = await validate_code("test-container")
            assert result["validation_results"] == []
            assert result["error"] == "command failed"

    @pytest.mark.asyncio
    async def test_target_files_parameter_appended(self):
        with patch(_EXEC_CV, new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = [
                {"success": True, "output": ""},
                {"success": True, "output": "[]"},
            ]
            await validate_code("test-container", target_files=["a.py", "b.py"])
            # The second call should include the files in the command
            second_call = mock_exec.call_args_list[1]
            cmd = second_call[0][1]
            assert "a.py" in cmd
            assert "b.py" in cmd


# ===================================================================
# validate_code_v2 tests
# ===================================================================


class TestValidateCodeV2:
    """Tests for validate_code_v2 function."""

    @pytest.mark.asyncio
    async def test_with_target_files(self):
        mock_cm = MagicMock()
        mock_cm.execute_command = AsyncMock(return_value={"success": True, "output": "x = 1\n"})
        with patch(_GET_CM, return_value=mock_cm):
            result = await validate_code_v2("test-container", target_files=["main.py"])
            assert result["valid_files"] == 1
            assert result["invalid_files"] == 0
            assert result["total_lines"] > 0

    @pytest.mark.asyncio
    async def test_filters_non_python_files(self):
        mock_cm = MagicMock()
        mock_cm.execute_command = AsyncMock(return_value={"success": True, "output": "x = 1\n"})
        with patch(_GET_CM, return_value=mock_cm):
            result = await validate_code_v2("test-container", target_files=["main.py", "readme.md"])
            # readme.md is filtered out since it's not .py
            assert len(result["validation_results"]) == 1

    @pytest.mark.asyncio
    async def test_syntax_error_detected(self):
        mock_cm = MagicMock()
        mock_cm.execute_command = AsyncMock(return_value={"success": True, "output": "def foo(\n"})
        with patch(_GET_CM, return_value=mock_cm):
            result = await validate_code_v2("test-container", target_files=["bad.py"])
            assert result["invalid_files"] == 1
            assert any(
                "SyntaxError" in issue
                for r in result["validation_results"]
                for issue in r.get("issues", [])
            )

    @pytest.mark.asyncio
    async def test_long_line_warnings(self):
        long_line = "x = " + "a" * 200 + "\n"
        mock_cm = MagicMock()
        mock_cm.execute_command = AsyncMock(return_value={"success": True, "output": long_line})
        with patch(_GET_CM, return_value=mock_cm):
            result = await validate_code_v2("test-container", target_files=["main.py"])
            assert result["valid_files"] == 1
            warnings = result["validation_results"][0].get("warnings", [])
            assert any("too long" in w.lower() for w in warnings)

    @pytest.mark.asyncio
    async def test_cat_failure_skips_file(self):
        mock_cm = MagicMock()
        mock_cm.execute_command = AsyncMock(
            return_value={"success": False, "output": "file not found"}
        )
        with patch(_GET_CM, return_value=mock_cm):
            result = await validate_code_v2("test-container", target_files=["missing.py"])
            assert len(result["validation_results"]) == 0

    @pytest.mark.asyncio
    async def test_find_command_failure_yields_empty(self):
        mock_cm = MagicMock()
        mock_cm.execute_command = AsyncMock(return_value={"success": False, "output": ""})
        with patch(_GET_CM, return_value=mock_cm):
            result = await validate_code_v2("test-container")
            assert result["validation_results"] == []

    @pytest.mark.asyncio
    async def test_find_command_discovers_files(self):
        call_count = 0

        async def mock_exec(container_name, cmd):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # find command returns file list
                return {"success": True, "output": "./main.py\n./utils.py\n"}
            else:
                # cat returns valid python
                return {"success": True, "output": "x = 1\n"}

        mock_cm = MagicMock()
        mock_cm.execute_command = AsyncMock(side_effect=mock_exec)
        with patch(_GET_CM, return_value=mock_cm):
            result = await validate_code_v2("test-container")
            assert result["valid_files"] == 2

    @pytest.mark.asyncio
    async def test_value_error_returns_error_result(self):
        with patch(_GET_CM, side_effect=ValueError("no container manager")):
            result = await validate_code_v2("test-container")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_file_not_found_returns_error_result(self):
        with patch(_GET_CM, side_effect=FileNotFoundError("manager not found")):
            result = await validate_code_v2("test-container")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_processing_error_handled(self):
        """ValueError/KeyError/TypeError during file processing appends invalid result."""
        mock_cm = MagicMock()

        async def mock_exec(container_name, cmd):
            if "cat" in cmd:
                raise ValueError("unexpected parsing error")
            return {"success": True, "output": "./main.py\n"}

        mock_cm.execute_command = AsyncMock(side_effect=mock_exec)
        with patch(_GET_CM, return_value=mock_cm):
            result = await validate_code_v2("test-container", target_files=["main.py"])
            # The ValueError is caught and file is marked invalid
            assert result["invalid_files"] == 1


# ===================================================================
# scan_dependencies tests
# ===================================================================


class TestScanDependencies:
    """Tests for scan_dependencies function."""

    @pytest.mark.asyncio
    async def test_script_write_failure(self):
        with patch(_EXEC_DS, new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"success": False, "output": "failed"}
            result = await scan_dependencies("test-container")
            assert result["dependencies"] == []
            assert "error" in result

    @pytest.mark.asyncio
    async def test_successful_scan(self):
        deps_data = [
            {"name": "requests", "version": "2.28.0", "source": "pip"},
            {"name": "flask", "version": "2.3.0", "source": "pip"},
        ]
        with patch(_EXEC_DS, new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = [
                {"success": True, "output": ""},
                {"success": True, "output": json.dumps(deps_data)},
            ]
            result = await scan_dependencies("test-container")
            assert result["dependency_count"] == 2
            assert len(result["dependencies"]) == 2
            assert result["dependencies"][0].name == "requests"

    @pytest.mark.asyncio
    async def test_json_decode_error(self):
        with patch(_EXEC_DS, new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = [
                {"success": True, "output": ""},
                {"success": True, "output": "NOT JSON"},
            ]
            result = await scan_dependencies("test-container")
            assert result["dependencies"] == []
            assert "error" in result

    @pytest.mark.asyncio
    async def test_execution_failure(self):
        with patch(_EXEC_DS, new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = [
                {"success": True, "output": ""},
                {"success": False, "output": "pip not found"},
            ]
            result = await scan_dependencies("test-container")
            assert result["dependencies"] == []
            assert result["error"] == "pip not found"


# ===================================================================
# scan_dependencies_v2 tests
# ===================================================================


class TestScanDependenciesV2:
    """Tests for scan_dependencies_v2 function."""

    @pytest.mark.asyncio
    async def test_pip_list_success(self):
        pip_output = json.dumps(
            [
                {"name": "requests", "version": "2.28.0"},
                {"name": "flask", "version": "2.3.0"},
            ]
        )
        mock_cm = MagicMock()
        mock_cm.execute_command = AsyncMock(
            side_effect=[
                {"success": True, "output": pip_output},  # pip list
                {"success": False, "output": ""},  # requirements.txt
                {"success": False, "output": ""},  # pyproject.toml
            ]
        )
        with patch(_GET_CM_DEP, return_value=mock_cm):
            result = await scan_dependencies_v2("test-container")
            assert result["dependency_count"] == 2
            assert "pip" in result["sources"]

    @pytest.mark.asyncio
    async def test_requirements_txt_parsing(self):
        req_content = "requests==2.28.0\nflask>=2.3.0\n# comment\nnumpy\n"
        mock_cm = MagicMock()
        mock_cm.execute_command = AsyncMock(
            side_effect=[
                {"success": False, "output": ""},  # pip list fails
                {"success": True, "output": req_content},  # requirements.txt
                {"success": False, "output": ""},  # pyproject.toml
            ]
        )
        with patch(_GET_CM_DEP, return_value=mock_cm):
            result = await scan_dependencies_v2("test-container")
            assert "requirements.txt" in result["sources"]
            # Should find requests, flask, numpy
            assert result["dependency_count"] >= 2

    @pytest.mark.asyncio
    async def test_pyproject_toml_detected(self):
        mock_cm = MagicMock()
        mock_cm.execute_command = AsyncMock(
            side_effect=[
                {"success": False, "output": ""},  # pip list
                {"success": False, "output": ""},  # requirements.txt
                {"success": True, "output": "[tool.poetry]"},  # pyproject.toml
            ]
        )
        with patch(_GET_CM_DEP, return_value=mock_cm):
            result = await scan_dependencies_v2("test-container")
            assert "pyproject.toml" in result["sources"]

    @pytest.mark.asyncio
    async def test_pip_json_with_extra_output(self):
        """Pip may output warnings before JSON; the function extracts JSON from brackets."""
        pip_output = 'WARNING: something\n[{"name": "pkg", "version": "1.0"}]'
        mock_cm = MagicMock()
        mock_cm.execute_command = AsyncMock(
            side_effect=[
                {"success": True, "output": pip_output},
                {"success": False, "output": ""},
                {"success": False, "output": ""},
            ]
        )
        with patch(_GET_CM_DEP, return_value=mock_cm):
            result = await scan_dependencies_v2("test-container")
            assert result["dependency_count"] == 1

    @pytest.mark.asyncio
    async def test_pip_json_decode_error_handled(self):
        mock_cm = MagicMock()
        mock_cm.execute_command = AsyncMock(
            side_effect=[
                {"success": True, "output": "totally broken output"},
                {"success": False, "output": ""},
                {"success": False, "output": ""},
            ]
        )
        with patch(_GET_CM_DEP, return_value=mock_cm):
            result = await scan_dependencies_v2("test-container")
            # No crash; just 0 dependencies
            assert result["dependency_count"] == 0

    @pytest.mark.asyncio
    async def test_value_error_returns_error(self):
        with patch(_GET_CM_DEP, side_effect=ValueError("broken")):
            result = await scan_dependencies_v2("test-container")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_file_not_found_returns_error(self):
        with patch(_GET_CM_DEP, side_effect=FileNotFoundError("missing")):
            result = await scan_dependencies_v2("test-container")
            assert "error" in result

    @pytest.mark.asyncio
    async def test_all_sources_empty(self):
        mock_cm = MagicMock()
        mock_cm.execute_command = AsyncMock(return_value={"success": False, "output": ""})
        with patch(_GET_CM_DEP, return_value=mock_cm):
            result = await scan_dependencies_v2("test-container")
            assert result["dependency_count"] == 0
            assert result["sources"] == []


# ===================================================================
# scan_imports tests
# ===================================================================


class TestScanImports:
    """Tests for scan_imports function."""

    @pytest.mark.asyncio
    async def test_script_write_failure(self):
        with patch(_EXEC_IS, new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"success": False, "output": "failed"}
            result = await scan_imports("test-container")
            assert result["imports"] == []
            assert "error" in result

    @pytest.mark.asyncio
    async def test_successful_scan(self):
        imports_data = [
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
                "is_valid": True,
            },
        ]
        with patch(_EXEC_IS, new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = [
                {"success": True, "output": ""},
                {"success": True, "output": json.dumps(imports_data)},
            ]
            result = await scan_imports("test-container")
            assert result["import_count"] == 2
            assert len(result["imports"]) == 2
            assert len(result["invalid_imports"]) == 0

    @pytest.mark.asyncio
    async def test_invalid_imports_tracked(self):
        imports_data = [
            {
                "module": "broken_module",
                "name": None,
                "alias": None,
                "import_type": "import",
                "file_path": "/workspace/main.py",
                "line_number": 1,
                "is_valid": False,
            },
        ]
        with patch(_EXEC_IS, new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = [
                {"success": True, "output": ""},
                {"success": True, "output": json.dumps(imports_data)},
            ]
            result = await scan_imports("test-container")
            assert len(result["invalid_imports"]) == 1

    @pytest.mark.asyncio
    async def test_json_decode_error(self):
        with patch(_EXEC_IS, new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = [
                {"success": True, "output": ""},
                {"success": True, "output": "NOT JSON"},
            ]
            result = await scan_imports("test-container")
            assert result["imports"] == []
            assert "error" in result

    @pytest.mark.asyncio
    async def test_execution_failure(self):
        with patch(_EXEC_IS, new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = [
                {"success": True, "output": ""},
                {"success": False, "output": "script crashed"},
            ]
            result = await scan_imports("test-container")
            assert result["imports"] == []
            assert result["error"] == "script crashed"

    @pytest.mark.asyncio
    async def test_target_files_passed(self):
        with patch(_EXEC_IS, new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = [
                {"success": True, "output": ""},
                {"success": True, "output": "[]"},
            ]
            await scan_imports("test-container", target_files=["src/app.py"])
            second_call = mock_exec.call_args_list[1]
            cmd = second_call[0][1]
            assert "src/app.py" in cmd


# ===================================================================
# validate_scanned_imports tests
# ===================================================================


class TestValidateScannedImports:
    """Tests for validate_scanned_imports function."""

    @pytest.mark.asyncio
    async def test_value_error_returns_original(self):
        with patch(_GET_CM_IMP, side_effect=ValueError("no manager")):
            imports = [{"module": "os", "name": None}]
            result = await validate_scanned_imports("test-container", imports)
            assert result["import_count"] == 1
            assert "validation_error" in result

    @pytest.mark.asyncio
    async def test_file_not_found_returns_original(self):
        with patch(_GET_CM_IMP, side_effect=FileNotFoundError("missing")):
            imports = [{"module": "os", "name": None}]
            result = await validate_scanned_imports("test-container", imports)
            assert result["import_count"] == 1
            assert "validation_error" in result

    @pytest.mark.asyncio
    async def test_successful_validation(self):
        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(return_value=MagicMock())

        mock_validation_result = MagicMock()
        mock_validation_result.valid_imports = [{"module": "os", "name": None}]
        mock_validation_result.invalid_imports = []
        mock_validation_result.stdlib_imports = [{"module": "os"}]
        mock_validation_result.third_party_imports = []
        mock_validation_result.local_imports = []
        mock_validation_result.missing_dependencies = []
        mock_validation_result.suggestions = []

        mock_ivs = MagicMock()
        mock_ivs.validate_imports = AsyncMock(return_value=mock_validation_result)

        with (
            patch(_GET_CM_IMP, return_value=mock_cm),
            patch(
                "app.api.endpoints.repo_scanner_modules.import_scanner.ImportValidationService",
                return_value=mock_ivs,
            ),
        ):
            imports = [{"module": "os", "name": None}]
            result = await validate_scanned_imports("test-container", imports)
            assert result["import_count"] == 1
            assert "validation_summary" in result
            assert result["validation_summary"]["stdlib_count"] == 1

    @pytest.mark.asyncio
    async def test_container_not_found_uses_default_root(self):
        mock_cm = MagicMock()
        mock_cm.get_container = AsyncMock(side_effect=ValueError("not found"))

        mock_validation_result = MagicMock()
        mock_validation_result.valid_imports = []
        mock_validation_result.invalid_imports = []
        mock_validation_result.stdlib_imports = []
        mock_validation_result.third_party_imports = []
        mock_validation_result.local_imports = []
        mock_validation_result.missing_dependencies = []
        mock_validation_result.suggestions = []

        mock_ivs = MagicMock()
        mock_ivs.validate_imports = AsyncMock(return_value=mock_validation_result)

        with (
            patch(_GET_CM_IMP, return_value=mock_cm),
            patch(
                "app.api.endpoints.repo_scanner_modules.import_scanner.ImportValidationService",
                return_value=mock_ivs,
            ),
        ):
            result = await validate_scanned_imports("test-container", [])
            assert result["import_count"] == 0
