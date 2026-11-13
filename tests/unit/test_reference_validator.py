"""Unit tests for reference_validator module."""

from __future__ import annotations

import os
import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_codebase(tmp_path, files: dict[str, str] | None = None):
    """Create a temporary codebase directory with given Python files.

    Args:
        tmp_path: pytest tmp_path fixture
        files: mapping of relative path -> content.  Defaults to a small sample.
    """
    if files is None:
        files = {
            "utils.py": textwrap.dedent("""\
                import os
                from pathlib import Path

                def helper_func(x):
                    return x + 1

                async def async_helper():
                    pass

                class HelperClass:
                    def do_stuff(self):
                        pass

                    async def async_do(self):
                        pass
            """),
            "models/user.py": textwrap.dedent("""\
                from typing import Any

                class UserModel:
                    name: str = ""
                    def greet(self):
                        return f"Hello {self.name}"

                class AdminModel(UserModel):
                    def admin_action(self):
                        pass

                CONSTANT_VAR = 42
            """),
        }
    base = tmp_path / "codebase"
    base.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        fpath = base / rel_path
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content, encoding="utf-8")
    return str(base)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def codebase_path(tmp_path):
    """Return a path to a small temporary codebase."""
    return _make_codebase(tmp_path)


@pytest.fixture
def validator(codebase_path):
    """Return a ReferenceValidator over the sample codebase."""
    from services.code_graph.reference_validator import ReferenceValidator

    return ReferenceValidator(codebase_path)


@pytest.fixture
def empty_validator(tmp_path):
    """Return a ReferenceValidator over an empty directory."""
    empty = tmp_path / "empty"
    empty.mkdir()
    from services.code_graph.reference_validator import ReferenceValidator

    return ReferenceValidator(str(empty))


# ===================================================================
# timeout context-manager
# ===================================================================


class TestTimeout:
    def test_timeout_does_not_raise_when_fast(self):
        from services.code_graph.reference_validator import timeout

        with timeout(5):
            x = 1 + 1
        assert x == 2

    def test_timeout_restores_handler(self):
        import signal

        from services.code_graph.reference_validator import timeout

        if not hasattr(signal, "SIGALRM"):
            pytest.skip("No SIGALRM on this platform")

        old = signal.getsignal(signal.SIGALRM)
        with timeout(5):
            pass
        after = signal.getsignal(signal.SIGALRM)
        assert after == old


# ===================================================================
# ValidationTimeoutError
# ===================================================================


class TestValidationTimeoutError:
    def test_is_exception(self):
        from services.code_graph.reference_validator import ValidationTimeoutError

        err = ValidationTimeoutError("test")
        assert isinstance(err, Exception)
        assert str(err) == "test"


# ===================================================================
# ReferenceValidator._build_indexes / _index_entities
# ===================================================================


class TestBuildIndexes:
    def test_indexes_python_files(self, validator):
        assert "utils.py" in validator.file_index
        assert "models/user.py" in validator.file_index

    def test_indexes_functions(self, validator):
        assert "helper_func" in validator.entity_index["functions"]
        assert "async_helper" in validator.entity_index["functions"]

    def test_indexes_classes(self, validator):
        assert "HelperClass" in validator.entity_index["classes"]
        assert "UserModel" in validator.entity_index["classes"]
        assert "AdminModel" in validator.entity_index["classes"]

    def test_class_methods_indexed(self, validator):
        assert "HelperClass.do_stuff" in validator.entity_index["methods"]
        assert "HelperClass.async_do" in validator.entity_index["methods"]
        assert "UserModel.greet" in validator.entity_index["methods"]
        assert "AdminModel.admin_action" in validator.entity_index["methods"]

    def test_indexes_imports(self, validator):
        assert "os" in validator.entity_index["imports"]
        assert "pathlib.Path" in validator.entity_index["imports"]
        assert "typing.Any" in validator.entity_index["imports"]

    def test_skips_non_python_files(self, tmp_path):
        base = tmp_path / "cb"
        base.mkdir()
        (base / "readme.txt").write_text("hello")
        (base / "data.json").write_text("{}")
        from services.code_graph.reference_validator import ReferenceValidator

        v = ReferenceValidator(str(base))
        assert len(v.file_index) == 0

    def test_handles_syntax_error_file(self, tmp_path):
        base = tmp_path / "cb"
        base.mkdir()
        (base / "bad.py").write_text("def (broken syntax!!")
        from services.code_graph.reference_validator import ReferenceValidator

        v = ReferenceValidator(str(base))
        # File content is still indexed, but no entities should be extracted
        assert "bad.py" in v.file_index
        # No functions/classes should have been extracted from a broken file
        for category in ("functions", "classes", "methods"):
            for entity_info in v.entity_index[category].values():
                assert entity_info.get("file") != "bad.py"

    def test_handles_unicode_error_file(self, tmp_path):
        base = tmp_path / "cb"
        base.mkdir()
        (base / "binary.py").write_bytes(b"\x80\x81\x82\x83")
        from services.code_graph.reference_validator import ReferenceValidator

        v = ReferenceValidator(str(base))
        assert "binary.py" not in v.file_index

    def test_empty_directory(self, empty_validator):
        assert len(empty_validator.file_index) == 0
        for category in empty_validator.entity_index.values():
            assert len(category) == 0


# ===================================================================
# validate_file_reference
# ===================================================================


class TestValidateFileReference:
    def test_existing_relative_path(self, validator):
        exists, path = validator.validate_file_reference("utils.py")
        assert exists is True
        assert path == "utils.py"

    def test_existing_nested_path(self, validator):
        exists, path = validator.validate_file_reference("models/user.py")
        assert exists is True
        assert path == "models/user.py"

    def test_nonexistent_relative_path_no_similar(self, validator):
        exists, path = validator.validate_file_reference("nonexistent.py")
        assert exists is False
        assert path is None

    def test_nonexistent_relative_path_similar_basename(self, validator):
        # "user.py" is the basename of "models/user.py" which should be suggested
        exists, path = validator.validate_file_reference("user.py")
        assert exists is False
        assert path is not None
        assert "models/user.py" in path

    def test_nonexistent_relative_path_partial_match(self, validator):
        # "models/" is a substring of "models/user.py"
        exists, path = validator.validate_file_reference("models/")
        assert exists is False
        assert path is not None
        assert "models/user.py" in path

    def test_absolute_path_existing(self, validator, codebase_path):
        abs_path = os.path.join(codebase_path, "utils.py")
        exists, path = validator.validate_file_reference(abs_path)
        assert exists is True
        assert path == "utils.py"

    def test_absolute_path_nonexistent(self, validator):
        exists, path = validator.validate_file_reference("/nonexistent/path.py")
        assert exists is False
        assert path is None


# ===================================================================
# validate_function_reference
# ===================================================================


class TestValidateFunctionReference:
    def test_existing_function(self, validator):
        exists, details = validator.validate_function_reference("helper_func")
        assert exists is True
        assert details is not None
        assert details["file"] == "utils.py"

    def test_existing_async_function(self, validator):
        exists, details = validator.validate_function_reference("async_helper")
        assert exists is True

    def test_method_with_dot_notation(self, validator):
        exists, details = validator.validate_function_reference("HelperClass.do_stuff")
        assert exists is True
        assert details is not None
        assert details["class"] == "HelperClass"

    def test_nonexistent_function_no_similar(self, validator):
        exists, details = validator.validate_function_reference("zzz_totally_unknown")
        assert exists is False
        assert details is None

    def test_nonexistent_function_with_similar(self, validator):
        # "helper" is a substring of "helper_func"
        exists, details = validator.validate_function_reference("helper")
        assert exists is False
        assert details is not None
        assert "similar_functions" in details
        assert "helper_func" in details["similar_functions"]


# ===================================================================
# validate_class_reference
# ===================================================================


class TestValidateClassReference:
    def test_existing_class(self, validator):
        exists, details = validator.validate_class_reference("HelperClass")
        assert exists is True
        assert details is not None
        assert "do_stuff" in details["methods"]

    def test_nonexistent_class_no_similar(self, validator):
        exists, details = validator.validate_class_reference("ZZZUnknownClass")
        assert exists is False
        assert details is None

    def test_nonexistent_class_with_similar(self, validator):
        exists, details = validator.validate_class_reference("Helper")
        assert exists is False
        assert details is not None
        assert "similar_classes" in details
        assert "HelperClass" in details["similar_classes"]

    def test_class_with_methods_list(self, validator):
        exists, details = validator.validate_class_reference("UserModel")
        assert exists is True
        assert "greet" in details["methods"]


# ===================================================================
# _create_error_response
# ===================================================================


class TestCreateErrorResponse:
    def test_structure(self, validator):
        resp = validator._create_error_response(
            "test_type", "test_name", "test_message", "test_suggestion"
        )
        assert resp["valid"] is False
        assert resp["overall_score"] == 0.0
        assert resp["suggestion"] == "test_suggestion"
        assert len(resp["issues"]) == 1
        issue = resp["issues"][0]
        assert issue["type"] == "test_type"
        assert issue["name"] == "test_name"
        assert issue["valid"] is False
        assert issue["suggestion"] == "test_message"


# ===================================================================
# _suggest_similar_import / _suggest_similar_function / _suggest_similar_method
# ===================================================================


class TestSuggestSimilar:
    def test_suggest_similar_import_found(self, validator):
        # "os" is in imports
        result = validator._suggest_similar_import("os")
        assert result is not None
        assert "os" in result

    def test_suggest_similar_import_not_found(self, validator):
        result = validator._suggest_similar_import("zzz_no_match_zzz")
        assert result is None

    def test_suggest_similar_function_found(self, validator):
        result = validator._suggest_similar_function("helper")
        assert result is not None
        assert "helper_func" in result

    def test_suggest_similar_function_not_found(self, validator):
        result = validator._suggest_similar_function("zzz_no_match_zzz")
        assert result is None

    def test_suggest_similar_method_found(self, validator):
        result = validator._suggest_similar_method("do_stuff")
        assert result is not None
        assert "do_stuff" in result

    def test_suggest_similar_method_not_found(self, validator):
        result = validator._suggest_similar_method("zzz_no_match_zzz")
        assert result is None


# ===================================================================
# validate_code_snippet
# ===================================================================


class TestValidateCodeSnippet:
    def test_valid_snippet_with_known_function(self, validator):
        snippet = "helper_func(42)"
        result = validator.validate_code_snippet(snippet)
        assert result["valid"] is True
        assert result["overall_score"] == 1.0
        assert result["suggestion"] is None

    def test_empty_snippet(self, validator):
        result = validator.validate_code_snippet("")
        assert result["valid"] is False
        assert result["issues"][0]["type"] == "input"

    def test_none_snippet(self, validator):
        result = validator.validate_code_snippet(None)
        assert result["valid"] is False
        assert result["issues"][0]["type"] == "input"

    def test_non_string_snippet(self, validator):
        result = validator.validate_code_snippet(123)
        assert result["valid"] is False

    def test_syntax_error_snippet(self, validator):
        result = validator.validate_code_snippet("def (broken!!")
        assert result["valid"] is False
        assert result["issues"][0]["type"] == "syntax"
        assert result["overall_score"] == 0.0

    def test_unknown_import(self, validator):
        snippet = "import nonexistent_module_xyz"
        result = validator.validate_code_snippet(snippet)
        assert result["valid"] is False
        assert len(result["issues"]) > 0
        assert result["issues"][0]["type"] == "import"

    def test_known_import(self, validator):
        snippet = "import os"
        result = validator.validate_code_snippet(snippet)
        assert result["valid"] is True
        assert result["overall_score"] == 1.0

    def test_unknown_from_import(self, validator):
        snippet = "from unknown_module import something"
        result = validator.validate_code_snippet(snippet)
        assert result["valid"] is False

    def test_known_from_import(self, validator):
        snippet = "from pathlib import Path"
        result = validator.validate_code_snippet(snippet)
        assert result["valid"] is True

    def test_unknown_function_call(self, validator):
        snippet = "nonexistent_function()"
        result = validator.validate_code_snippet(snippet)
        assert result["valid"] is False
        assert any(i["type"] == "function" for i in result["issues"])

    def test_unknown_method_call(self, validator):
        snippet = "obj.nonexistent_method()"
        result = validator.validate_code_snippet(snippet)
        assert result["valid"] is False
        assert any(i["type"] == "method" for i in result["issues"])

    def test_no_references_gives_score_1(self, validator):
        snippet = "x = 1 + 2"
        result = validator.validate_code_snippet(snippet)
        assert result["valid"] is True
        assert result["overall_score"] == 1.0

    def test_mixed_valid_invalid(self, validator):
        snippet = "import os\nimport nonexistent_xyz"
        result = validator.validate_code_snippet(snippet)
        assert result["valid"] is False
        assert 0 < result["overall_score"] < 1.0
        assert result["suggestion"] is not None

    def test_snippet_with_issues_has_suggestion(self, validator):
        snippet = "import fake_module_xyz"
        result = validator.validate_code_snippet(snippet)
        assert result["suggestion"] is not None
        assert "validation issue" in result["suggestion"]


# ===================================================================
# _validate_import_node / _validate_import_from_node / _validate_call_node
# ===================================================================


class TestValidateNodes:
    def test_validate_import_node_known(self, validator):
        import ast

        tree = ast.parse("import os")
        import_node = [n for n in ast.walk(tree) if isinstance(n, ast.Import)][0]
        result = {"valid": True, "issues": []}
        checked, issues = validator._validate_import_node(import_node, result)
        assert checked == 1
        assert issues == 0
        assert result["valid"] is True

    def test_validate_import_node_unknown(self, validator):
        import ast

        tree = ast.parse("import zzz_unknown_pkg")
        import_node = [n for n in ast.walk(tree) if isinstance(n, ast.Import)][0]
        result = {"valid": True, "issues": []}
        checked, issues = validator._validate_import_node(import_node, result)
        assert checked == 1
        assert issues == 1
        assert result["valid"] is False

    def test_validate_import_from_node_known(self, validator):
        import ast

        tree = ast.parse("from pathlib import Path")
        node = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)][0]
        result = {"valid": True, "issues": []}
        checked, issues = validator._validate_import_from_node(node, result)
        assert checked == 1
        assert issues == 0

    def test_validate_import_from_node_unknown(self, validator):
        import ast

        tree = ast.parse("from unknown_pkg import thing")
        node = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)][0]
        result = {"valid": True, "issues": []}
        checked, issues = validator._validate_import_from_node(node, result)
        assert checked == 1
        assert issues == 1

    def test_validate_call_node_known_function(self, validator):
        import ast

        tree = ast.parse("helper_func(42)")
        call_node = [n for n in ast.walk(tree) if isinstance(n, ast.Call)][0]
        result = {"valid": True, "issues": []}
        checked, issues = validator._validate_call_node(call_node, result)
        assert checked == 1
        assert issues == 0

    def test_validate_call_node_unknown_function(self, validator):
        import ast

        tree = ast.parse("unknown_func()")
        call_node = [n for n in ast.walk(tree) if isinstance(n, ast.Call)][0]
        result = {"valid": True, "issues": []}
        checked, issues = validator._validate_call_node(call_node, result)
        assert checked == 1
        assert issues == 1

    def test_validate_call_node_method_call(self, validator):
        import ast

        tree = ast.parse("obj.some_method()")
        call_node = [n for n in ast.walk(tree) if isinstance(n, ast.Call)][0]
        result = {"valid": True, "issues": []}
        checked, issues = validator._validate_call_node(call_node, result)
        assert checked == 1
        # "obj.some_method" is not in methods index so it should be an issue
        assert issues == 1


# ===================================================================
# PackageReferenceValidator
# ===================================================================


class TestPackageReferenceValidator:
    @pytest.fixture
    def pkg_validator(self, codebase_path):
        from services.code_graph.reference_validator import PackageReferenceValidator

        return PackageReferenceValidator(codebase_path)

    @pytest.fixture
    def pkg_validator_with_discovery(self, codebase_path):
        from services.code_graph.reference_validator import PackageReferenceValidator

        mock_discovery = MagicMock()
        mock_discovery.is_standard_library_module = MagicMock(return_value=False)
        mock_discovery.discover_container_packages = AsyncMock(
            return_value={
                "total_packages": 5,
                "standard_library_modules": 200,
                "total_importable_modules": 300,
                "modules_index": {
                    "requests": {
                        "version": "2.31.0",
                        "type": "pip",
                        "modules": ["requests", "requests.auth", "requests.sessions"],
                    },
                    "numpy": {
                        "version": "1.26.0",
                        "type": "pip",
                        "modules": ["numpy", "numpy.core", "numpy.linalg"],
                    },
                },
            }
        )
        return PackageReferenceValidator(codebase_path, package_discovery=mock_discovery)

    # --- initialize_for_container ---

    async def test_initialize_for_container_with_discovery(self, pkg_validator_with_discovery):
        await pkg_validator_with_discovery.initialize_for_container("test-container")
        assert pkg_validator_with_discovery.container_packages is not None
        assert pkg_validator_with_discovery.container_packages["total_packages"] == 5

    async def test_initialize_for_container_without_discovery(self, pkg_validator):
        await pkg_validator.initialize_for_container("test-container")
        assert pkg_validator.container_packages == {}

    # --- _is_package_importable ---

    def test_is_package_importable_no_packages(self, pkg_validator):
        assert pkg_validator._is_package_importable("requests") is False

    def test_is_package_importable_direct_match(self, pkg_validator_with_discovery):
        # Need to set container_packages first
        pkg_validator_with_discovery.container_packages = {
            "modules_index": {
                "requests": {"modules": ["requests"]},
            }
        }
        assert pkg_validator_with_discovery._is_package_importable("requests") is True

    def test_is_package_importable_module_match(self, pkg_validator_with_discovery):
        pkg_validator_with_discovery.container_packages = {
            "modules_index": {
                "numpy": {"modules": ["numpy", "numpy.core"]},
            }
        }
        assert pkg_validator_with_discovery._is_package_importable("numpy.core") is True

    def test_is_package_importable_submodule_prefix(self, pkg_validator_with_discovery):
        pkg_validator_with_discovery.container_packages = {
            "modules_index": {
                "numpy": {"modules": ["numpy"]},
            }
        }
        assert pkg_validator_with_discovery._is_package_importable("numpy.sub") is True

    def test_is_package_importable_not_found(self, pkg_validator_with_discovery):
        pkg_validator_with_discovery.container_packages = {
            "modules_index": {
                "numpy": {"modules": ["numpy"]},
            }
        }
        assert pkg_validator_with_discovery._is_package_importable("pandas") is False

    # --- _get_package_info ---

    def test_get_package_info_no_packages(self, pkg_validator):
        assert pkg_validator._get_package_info("requests") == {}

    def test_get_package_info_direct_match(self, pkg_validator):
        pkg_validator.container_packages = {
            "modules_index": {
                "requests": {"version": "2.31.0", "modules": ["requests"]},
            }
        }
        info = pkg_validator._get_package_info("requests")
        assert info["version"] == "2.31.0"

    def test_get_package_info_via_module(self, pkg_validator):
        pkg_validator.container_packages = {
            "modules_index": {
                "numpy_pkg": {
                    "version": "1.26",
                    "type": "pip",
                    "modules": ["numpy", "numpy.core"],
                },
            }
        }
        info = pkg_validator._get_package_info("numpy")
        assert info["providing_package"] == "numpy_pkg"
        assert info["package_version"] == "1.26"

    def test_get_package_info_via_submodule_prefix(self, pkg_validator):
        pkg_validator.container_packages = {
            "modules_index": {
                "numpy_pkg": {
                    "version": "1.26",
                    "type": "pip",
                    "modules": ["numpy"],
                },
            }
        }
        info = pkg_validator._get_package_info("numpy.linalg")
        assert info["providing_package"] == "numpy_pkg"

    def test_get_package_info_not_found(self, pkg_validator):
        pkg_validator.container_packages = {
            "modules_index": {
                "numpy": {"modules": ["numpy"]},
            }
        }
        assert pkg_validator._get_package_info("pandas") == {}

    # --- validate_import_statement ---

    def test_validate_import_source_found(self, pkg_validator):
        # "os" is indexed from utils.py
        is_valid, details = pkg_validator.validate_import_statement("os")
        assert is_valid is True
        assert details["found_in_source"] is True
        assert details["resolution"] == "source"

    def test_validate_import_not_found_anywhere(self, pkg_validator):
        is_valid, details = pkg_validator.validate_import_statement("zzz_unknown_pkg")
        assert is_valid is False
        assert details["resolution"] == "not_found"

    def test_validate_import_stdlib(self, codebase_path):
        from services.code_graph.reference_validator import PackageReferenceValidator

        mock_discovery = MagicMock()
        mock_discovery.is_standard_library_module = MagicMock(return_value=True)
        v = PackageReferenceValidator(codebase_path, package_discovery=mock_discovery)
        is_valid, details = v.validate_import_statement("zzz_stdlib_fake")
        assert is_valid is True
        assert details["found_in_stdlib"] is True
        assert details["resolution"] == "standard_library"

    def test_validate_import_package(self, codebase_path):
        from services.code_graph.reference_validator import PackageReferenceValidator

        mock_discovery = MagicMock()
        mock_discovery.is_standard_library_module = MagicMock(return_value=False)
        v = PackageReferenceValidator(codebase_path, package_discovery=mock_discovery)
        v.container_packages = {
            "modules_index": {
                "requests": {"modules": ["requests"]},
            }
        }
        is_valid, details = v.validate_import_statement("requests")
        assert is_valid is True
        assert details["found_in_packages"] is True
        assert details["resolution"] == "package"

    # --- validate_code_snippet_comprehensive ---

    async def test_comprehensive_syntax_error(self, pkg_validator):
        result = await pkg_validator.validate_code_snippet_comprehensive("def (broken!!")
        assert result["valid"] is False
        assert result["overall_score"] == 0.0
        assert any(i["type"] == "syntax_error" for i in result["issues"])

    async def test_comprehensive_valid_import(self, pkg_validator):
        result = await pkg_validator.validate_code_snippet_comprehensive("import os")
        assert result["overall_score"] > 0

    async def test_comprehensive_with_container(self, pkg_validator_with_discovery):
        result = await pkg_validator_with_discovery.validate_code_snippet_comprehensive(
            "import os", container_name="test-container"
        )
        assert "total_checks" in result
        assert "container_packages_info" in result

    async def test_comprehensive_from_import(self, pkg_validator):
        result = await pkg_validator.validate_code_snippet_comprehensive("from pathlib import Path")
        assert "issues" in result
        # Should have from_import type issues
        types_found = {i["type"] for i in result["issues"]}
        assert "from_import" in types_found

    async def test_comprehensive_valid_from_import_items(self, pkg_validator):
        # "os" is in source, so from_import should be valid and items inherited
        result = await pkg_validator.validate_code_snippet_comprehensive("from os import path")
        # Check that from_import_item is present when parent module valid
        types_found = {i["type"] for i in result["issues"]}
        # "os" is in entity_index["imports"] as a source import
        # We need to check the validate_import_statement result
        assert "from_import" in types_found

    async def test_comprehensive_no_checks(self, pkg_validator):
        result = await pkg_validator.validate_code_snippet_comprehensive("x = 1 + 2")
        assert result["valid"] is True
        assert result["overall_score"] == 1.0

    async def test_comprehensive_no_container_packages_info(self, pkg_validator):
        result = await pkg_validator.validate_code_snippet_comprehensive("x = 1")
        assert result["container_packages_info"] is None

    async def test_comprehensive_with_container_packages_info(self, pkg_validator_with_discovery):
        await pkg_validator_with_discovery.initialize_for_container("test")
        result = await pkg_validator_with_discovery.validate_code_snippet_comprehensive("import os")
        assert result["container_packages_info"] is not None
        assert result["container_packages_info"]["total_packages"] == 5

    async def test_comprehensive_overall_valid_threshold(self, pkg_validator):
        # Test that overall_valid is True when score >= 0.5
        result = await pkg_validator.validate_code_snippet_comprehensive("import os")
        # os is in source so valid
        assert result["valid"] is True

    async def test_comprehensive_mixed_imports(self, pkg_validator):
        snippet = "import os\nimport zzz_fake_pkg"
        result = await pkg_validator.validate_code_snippet_comprehensive(snippet)
        assert result["total_checks"] == 2
        # One passes (os is in source), one fails
        assert result["passed_checks"] == 1
        assert result["overall_score"] == 0.5
        assert result["valid"] is True  # 0.5 >= 0.5
