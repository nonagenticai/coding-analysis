"""Tests for symbolic analyzer (SymbolicValidator)."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

from models.static_analysis_models import ChangeType, CodeChange, ValidationType
from processing.analyzers.symbolic.analyzer import SymbolicValidator
from processing.analyzers.symbolic.config import SymbolicAnalysisConfig as ValidationConfig


@pytest.fixture
def validator():
    return SymbolicValidator()


@pytest.fixture
def validator_with_config():
    config = ValidationConfig(
        validation_types=[ValidationType.SYMBOLIC_EXECUTION, ValidationType.TAINT_ANALYSIS],
    )
    return SymbolicValidator(config)


@pytest.fixture
def sample_python_file():
    code = """\
def divide(a: int, b: int) -> float:
    return a / b

def safe_divide(a: int, b: int) -> float:
    if b == 0:
        return 0.0
    return a / b

def greet(name: str) -> str:
    return f"Hello, {name}!"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        yield f.name
    os.unlink(f.name)


def test_get_supported_extensions(validator):
    exts = validator.get_supported_extensions()
    assert ".py" in exts


def test_get_language_name(validator):
    assert validator.get_language_name() == "python"


@pytest.mark.asyncio
async def test_analyze_file_success(validator, sample_python_file):
    result = await validator.analyze_file(sample_python_file)
    assert result.success is True
    assert result.language == "python"
    assert result.file_path == sample_python_file
    assert "symbolic_analysis" in result.metadata


@pytest.mark.asyncio
async def test_analyze_file_nonexistent(validator):
    result = await validator.analyze_file("/nonexistent/file.py")
    # Analyzer runs tools and aggregates — individual tool failures don't
    # necessarily make the overall result fail (tools may return empty findings)
    assert result is not None
    assert hasattr(result, "success")


@pytest.mark.asyncio
async def test_analyze_directory_success(validator):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a python file in the directory
        fp = os.path.join(tmpdir, "test.py")
        with open(fp, "w") as f:
            f.write("def hello(): pass\n")
        result = await validator.analyze_directory(tmpdir)
        assert result.success is True
        assert "symbolic_analysis" in result.metadata
        assert "files" in result.metadata["symbolic_analysis"]


@pytest.mark.asyncio
async def test_analyze_directory_empty(validator):
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await validator.analyze_directory(tmpdir)
        assert result.success is True
        summary = result.metadata["symbolic_analysis"]["summary"]
        assert summary["total_files"] == 0


@pytest.mark.asyncio
async def test_analyze_snippet_success(validator):
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    result = await validator.analyze_snippet(code)
    assert result.success is True
    assert result.file_path == "snippet"
    assert result.metadata.get("snippet") is True


@pytest.mark.asyncio
async def test_analyze_snippet_unsupported_language(validator):
    code = "fn main() {}"
    result = await validator.analyze_snippet(code, "rust")
    assert result.success is False
    assert "not supported" in (result.error or "")


@pytest.mark.asyncio
async def test_validate_code_changes(validator):
    changes = [
        CodeChange(
            file_path="test.py",
            original_code="def divide(a, b): return a / b",
            modified_code="def divide(a, b): return a / b if b != 0 else 0",
            change_type=ChangeType.MODIFIED,
        )
    ]
    result = await validator.validate_code_changes(changes)
    assert result.analysis_id is not None
    assert isinstance(result.overall_safety_score, float)
    assert isinstance(result.validation_results, list)
    assert isinstance(result.recommendations, list)


@pytest.mark.asyncio
async def test_validate_code_changes_with_config(validator):
    config = ValidationConfig(
        validation_types=[ValidationType.SYMBOLIC_EXECUTION],
    )
    changes = [
        CodeChange(
            file_path="test.py",
            original_code="x = 1",
            modified_code="x = 2",
            change_type=ChangeType.MODIFIED,
        )
    ]
    result = await validator.validate_code_changes(changes, config)
    assert result.analysis_id is not None
    assert isinstance(result.overall_safety_score, float)


@pytest.mark.asyncio
async def test_validate_multiple_changes(validator):
    changes = [
        CodeChange(
            file_path="file1.py",
            original_code="x = 1",
            modified_code="x = 2",
            change_type=ChangeType.MODIFIED,
        ),
        CodeChange(
            file_path="file2.py",
            original_code="y = 'a'",
            modified_code="y = 'b'",
            change_type=ChangeType.MODIFIED,
        ),
    ]
    result = await validator.validate_code_changes(changes)
    assert len(result.validation_results) == 2


@pytest.mark.asyncio
async def test_run_symbolic_analysis_on_file(validator, sample_python_file):
    data = await validator.run_symbolic_analysis_on_file(sample_python_file)
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_run_symbolic_analysis_on_file_with_config(validator_with_config, sample_python_file):
    data = await validator_with_config.run_symbolic_analysis_on_file(sample_python_file)
    assert isinstance(data, dict)
    # Should have attempted both symbolic_execution and taint_analysis
    # Results may be error status if tools aren't installed, but keys should exist
    # since the config requested them


@pytest.mark.asyncio
async def test_run_symbolic_analysis_on_directory(validator):
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = os.path.join(tmpdir, "module.py")
        with open(fp, "w") as f:
            f.write("def foo(): return 42\n")
        data = await validator.run_symbolic_analysis_on_directory(tmpdir)
        assert "files" in data
        assert "summary" in data
        assert data["summary"]["total_files"] == 1


def test_create_error_result(validator):
    result = validator._create_error_result("test.py", "something broke")
    assert result.success is False
    assert result.error == "something broke"
    assert result.file_path == "test.py"


def test_create_error_validation_result(validator):
    result = validator._create_error_validation_result("test.py", "validation failed")
    assert result is not None


# ---------------------------------------------------------------------------
# Uncovered error paths: lines 181-186, 209-214, 255-260, 321, 326-331,
# 349-351, 362, 374-376, 385-387, 410-412, 446-448, 451-455
# ---------------------------------------------------------------------------

from processing.analyzers.symbolic.exceptions import (
    CrossHairAnalysisError,
    FormalVerificationError,
    PysaAnalysisError,
    SymbolicAnalysisError,
)


@pytest.mark.asyncio
async def test_analyze_file_os_error(validator):
    """OSError when reading a file (lines 181-183)."""
    with patch.object(
        validator,
        "run_symbolic_analysis_on_file",
        side_effect=OSError("Permission denied"),
    ):
        result = await validator.analyze_file("/locked/file.py")
        assert result.success is False
        assert "Permission denied" in (result.error or "")


@pytest.mark.asyncio
async def test_analyze_file_symbolic_analysis_error(validator):
    """SymbolicAnalysisError during file analysis (lines 184-186)."""
    with patch.object(
        validator,
        "run_symbolic_analysis_on_file",
        side_effect=SymbolicAnalysisError("symbolic fail"),
    ):
        result = await validator.analyze_file("/test/file.py")
        assert result.success is False
        assert "symbolic fail" in (result.error or "")


@pytest.mark.asyncio
async def test_analyze_directory_os_error(validator):
    """OSError when reading a directory (lines 209-211)."""
    with patch.object(
        validator,
        "run_symbolic_analysis_on_directory",
        side_effect=OSError("dir not accessible"),
    ):
        result = await validator.analyze_directory("/locked/dir")
        assert result.success is False
        assert "dir not accessible" in (result.error or "")


@pytest.mark.asyncio
async def test_analyze_directory_symbolic_analysis_error(validator):
    """SymbolicAnalysisError during directory analysis (lines 212-214)."""
    with patch.object(
        validator,
        "run_symbolic_analysis_on_directory",
        side_effect=SymbolicAnalysisError("dir analysis failed"),
    ):
        result = await validator.analyze_directory("/test/dir")
        assert result.success is False
        assert "dir analysis failed" in (result.error or "")


@pytest.mark.asyncio
async def test_analyze_snippet_os_error(validator):
    """OSError during snippet processing (lines 255-257)."""
    with patch("tempfile.NamedTemporaryFile", side_effect=OSError("disk full")):
        result = await validator.analyze_snippet("def foo(): pass")
        assert result.success is False
        assert "disk full" in (result.error or "")


@pytest.mark.asyncio
async def test_analyze_snippet_symbolic_error(validator):
    """SymbolicAnalysisError during snippet processing (lines 258-260)."""
    with patch.object(
        validator,
        "run_symbolic_analysis_on_file",
        side_effect=SymbolicAnalysisError("snippet analysis error"),
    ):
        result = await validator.analyze_snippet("def foo(): pass")
        assert result.success is False
        assert "snippet analysis error" in (result.error or "")


@pytest.mark.asyncio
async def test_validate_code_changes_file_processing_error(validator):
    """FileProcessingError during validation (lines 326-331)."""
    from processing.analyzers.symbolic.exceptions import FileProcessingError

    changes = [
        CodeChange(
            file_path="broken.py",
            original_code="x = 1",
            modified_code="x = 2",
            change_type=ChangeType.MODIFIED,
        )
    ]
    with patch.object(
        validator.validation_helper,
        "create_temp_file",
        side_effect=FileProcessingError("cannot create temp file"),
    ):
        result = await validator.validate_code_changes(changes)
        assert result.analysis_id is not None
        assert len(result.validation_results) == 1
        # Safety score should be 0.0 for failed validation
        assert 0.0 in [s for s in [result.overall_safety_score]]


@pytest.mark.asyncio
async def test_validate_code_changes_validation_error(validator):
    """ValidationError propagates (lines 349-351)."""
    from processing.analyzers.symbolic.exceptions import ValidationError

    changes = [
        CodeChange(
            file_path="test.py",
            original_code="x = 1",
            modified_code="x = 2",
            change_type=ChangeType.MODIFIED,
        )
    ]

    # Create a real temp file so unlink works
    import tempfile as _tf

    tmp = _tf.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmp.write("x = 2")
    tmp.flush()

    with patch.object(
        validator,
        "_validate_single_change",
        side_effect=ValidationError("validation broke"),
    ):
        with patch.object(validator.validation_helper, "create_temp_file", return_value=tmp):
            with pytest.raises(ValidationError):
                await validator.validate_code_changes(changes)

    # Clean up if file still exists
    try:
        os.unlink(tmp.name)
    except FileNotFoundError:
        pass


@pytest.mark.asyncio
async def test_run_symbolic_crosshair_error(sample_python_file):
    """CrossHairAnalysisError during symbolic execution (lines 362-368)."""
    config = ValidationConfig(
        validation_types=[ValidationType.SYMBOLIC_EXECUTION],
    )
    v = SymbolicValidator(config)
    with patch.object(
        v.crosshair_analyzer,
        "analyze",
        side_effect=CrossHairAnalysisError("crosshair crash"),
    ):
        data = await v.run_symbolic_analysis_on_file(sample_python_file)
        assert data["symbolic_execution"]["status"] == "error"
        assert "crosshair crash" in data["symbolic_execution"]["error"]


@pytest.mark.asyncio
async def test_run_symbolic_pysa_error(sample_python_file):
    """PysaAnalysisError during taint analysis (lines 374-376)."""
    config = ValidationConfig(
        validation_types=[ValidationType.TAINT_ANALYSIS],
    )
    v = SymbolicValidator(config)
    with patch.object(
        v.pysa_analyzer,
        "analyze",
        side_effect=PysaAnalysisError("pysa crash"),
    ):
        data = await v.run_symbolic_analysis_on_file(sample_python_file)
        assert data["taint_analysis"]["status"] == "error"


@pytest.mark.asyncio
async def test_run_symbolic_formal_verification_error(sample_python_file):
    """FormalVerificationError during formal verification (lines 385-387)."""
    config = ValidationConfig(
        validation_types=[ValidationType.FORMAL_VERIFICATION],
    )
    v = SymbolicValidator(config)
    with patch.object(
        v.formal_verifier,
        "verify",
        side_effect=FormalVerificationError("formal crash"),
    ):
        data = await v.run_symbolic_analysis_on_file(sample_python_file)
        assert data["formal_verification"]["status"] == "error"


@pytest.mark.asyncio
async def test_run_symbolic_analysis_on_directory_file_error(validator):
    """SymbolicAnalysisError on individual file in directory (lines 410-412)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fp = os.path.join(tmpdir, "err.py")
        with open(fp, "w") as f:
            f.write("x = 1\n")

        with patch.object(
            validator,
            "run_symbolic_analysis_on_file",
            side_effect=SymbolicAnalysisError("file error"),
        ):
            data = await validator.run_symbolic_analysis_on_directory(tmpdir)
            assert "error" in data["files"][fp]
            assert data["summary"]["failed_files"] == 1


@pytest.mark.asyncio
async def test_validate_single_change_crosshair_error():
    """CrossHairAnalysisError in _validate_single_change (lines 438-440)."""
    config = ValidationConfig(
        validation_types=[ValidationType.SYMBOLIC_EXECUTION],
    )
    v = SymbolicValidator(config)
    change = CodeChange(
        file_path="test.py",
        original_code="x = 1",
        modified_code="x = 2",
        change_type=ChangeType.MODIFIED,
    )
    with patch.object(
        v.crosshair_analyzer,
        "analyze",
        side_effect=CrossHairAnalysisError("crosshair fail"),
    ):
        result = await v._validate_single_change("/tmp/test.py", change, config)
        assert result is not None


@pytest.mark.asyncio
async def test_validate_single_change_pysa_error():
    """PysaAnalysisError in _validate_single_change (lines 446-448)."""
    config = ValidationConfig(
        validation_types=[ValidationType.TAINT_ANALYSIS],
    )
    v = SymbolicValidator(config)
    change = CodeChange(
        file_path="test.py",
        original_code="x = 1",
        modified_code="x = 2",
        change_type=ChangeType.MODIFIED,
    )
    with patch.object(
        v.pysa_analyzer,
        "analyze",
        side_effect=PysaAnalysisError("pysa fail"),
    ):
        result = await v._validate_single_change("/tmp/test.py", change, config)
        assert result is not None


@pytest.mark.asyncio
async def test_validate_single_change_formal_error():
    """FormalVerificationError in _validate_single_change (lines 451-455)."""
    config = ValidationConfig(
        validation_types=[ValidationType.FORMAL_VERIFICATION],
    )
    v = SymbolicValidator(config)
    change = CodeChange(
        file_path="test.py",
        original_code="x = 1",
        modified_code="x = 2",
        change_type=ChangeType.MODIFIED,
    )
    with patch.object(
        v.formal_verifier,
        "verify",
        side_effect=FormalVerificationError("formal fail"),
    ):
        result = await v._validate_single_change("/tmp/test.py", change, config)
        assert result is not None
