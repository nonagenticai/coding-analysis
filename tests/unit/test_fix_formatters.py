"""Tests for fix_formatters (CodeFormatter + QualityAnalysisCodeValidator).

Tests verify that formatting actually transforms code, subprocess commands
are invoked correctly, and error paths return original content.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from processing.quality_analysis.fix_formatters import (
    CodeFormatter,
    FixFormatter,
    QualityAnalysisCodeValidator,
    RuffFixError,
)


@pytest.fixture
def formatter():
    return CodeFormatter()


@pytest.fixture
def validator():
    return QualityAnalysisCodeValidator()


# --- CodeFormatter tests ---


class TestCodeFormatterRuffFormat:
    @pytest.mark.asyncio
    async def test_apply_ruff_format_formats_unformatted_code(self, formatter):
        """Real ruff (a dev dependency, so on PATH under uv run) reformats."""
        result = await formatter.apply_ruff_formatting(
            "test.py", "x=1\ndef f( a,b ):\n  return a+b\n"
        )
        assert result == "x = 1\n\n\ndef f(a, b):\n    return a + b\n"

    @pytest.mark.asyncio
    async def test_apply_ruff_format_calls_correct_command(self, formatter):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="x = 1\n", stderr="")
            result = await formatter.apply_ruff_formatting("test.py", "x=1\n")
        argv = mock_run.call_args[0][0]
        assert argv == ["ruff", "format", "--isolated", "--stdin-filename", "fix_input.py", "-"]
        assert mock_run.call_args.kwargs["input"] == "x=1\n"
        assert result == "x = 1\n"

    @pytest.mark.asyncio
    async def test_apply_ruff_format_unparseable_raises(self, formatter):
        """ruff format exits 2 on source it cannot parse (real ruff): an error."""
        with pytest.raises(RuffFixError, match="exited 2"):
            await formatter.apply_ruff_formatting("test.py", "def (:\n")

    @pytest.mark.asyncio
    async def test_apply_ruff_format_not_installed_raises(self, formatter):
        """ruff missing must surface, never return the content as 'applied'."""
        with patch("subprocess.run", side_effect=FileNotFoundError("ruff")):
            with pytest.raises(RuffFixError, match="not installed"):
                await formatter.apply_ruff_formatting("test.py", "x=1\n")

    @pytest.mark.asyncio
    async def test_apply_ruff_format_os_error_raises(self, formatter):
        with patch("subprocess.run", side_effect=OSError("boom")):
            with pytest.raises(RuffFixError, match="boom"):
                await formatter.apply_ruff_formatting("test.py", "x=1\n")

    @pytest.mark.asyncio
    async def test_isolated_ignores_a_discovered_ruff_config(
        self, formatter, tmp_path, monkeypatch
    ):
        """Real ruff: a pyproject in the cwd (the service's own) never applies."""
        (tmp_path / "pyproject.toml").write_text('[tool.ruff.format]\nquote-style = "single"\n')
        monkeypatch.chdir(tmp_path)
        assert await formatter.apply_ruff_formatting("test.py", "x = 'a'\n") == 'x = "a"\n'


class TestCodeFormatterPrettier:
    @pytest.mark.asyncio
    async def test_apply_prettier_calls_correct_command(self, formatter):
        """Verify prettier is invoked with --write and correct file extension."""
        code = "const x = 1;\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            await formatter.apply_prettier_formatting("test.js", code)
            assert mock_run.called
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "prettier"
            assert "--write" in call_args

    @pytest.mark.asyncio
    async def test_apply_prettier_returns_original_on_failure(self, formatter):
        """When prettier fails, original code is returned."""
        code = "const x = 1;\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="prettier error")
            result = await formatter.apply_prettier_formatting("test.js", code)
        assert result == code

    @pytest.mark.asyncio
    async def test_apply_prettier_file_not_found(self, formatter):
        """When prettier is not installed, original code is returned."""
        code = "const x = 1;\n"
        with patch("subprocess.run", side_effect=FileNotFoundError("No prettier")):
            result = await formatter.apply_prettier_formatting("test.js", code)
        assert result == code

    @pytest.mark.asyncio
    async def test_apply_prettier_os_error(self, formatter):
        """When OS-level error occurs, original code is returned."""
        code = "const x = 1;\n"
        with patch("subprocess.run", side_effect=OSError("disk full")):
            result = await formatter.apply_prettier_formatting("test.ts", code)
        assert result == code


class TestCodeFormatterRuffFixes:
    @pytest.mark.asyncio
    async def test_import_sorting_sorts_imports(self, formatter):
        result = await formatter.apply_ruff_import_sorting(
            "test.py", "import sys\nimport os\n\nprint(os, sys)\n"
        )
        assert result == "import os\nimport sys\n\nprint(os, sys)\n"

    @pytest.mark.asyncio
    async def test_unused_removal_drops_unused_import(self, formatter):
        result = await formatter.apply_ruff_unused_removal(
            "test.py", "import os\nimport sys\n\nprint(sys)\n"
        )
        assert result == "import sys\n\nprint(sys)\n"

    @pytest.mark.asyncio
    async def test_import_sorting_command(self, formatter):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="fixed\n", stderr="1 remaining")
            result = await formatter.apply_ruff_import_sorting("test.py", "code\n")
        assert mock_run.call_args[0][0] == [
            "ruff",
            "check",
            "--isolated",
            "--select",
            "I",
            "--fix",
            "--stdin-filename",
            "fix_input.py",
            "-",
        ]
        # exit 1 = unfixable violations remain; the applied fixes are in stdout.
        assert result == "fixed\n"

    @pytest.mark.asyncio
    async def test_unused_removal_command(self, formatter):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
            await formatter.apply_ruff_unused_removal("test.py", "code\n")
        assert mock_run.call_args[0][0] == [
            "ruff",
            "check",
            "--isolated",
            "--select",
            "F401,F841",
            "--fix",
            "--unsafe-fixes",
            "--stdin-filename",
            "fix_input.py",
            "-",
        ]

    @pytest.mark.asyncio
    async def test_unused_removal_drops_unused_variable_real_ruff(self, formatter):
        """autoflake parity: F841's fix is unsafe in ruff; this path applies it.

        Runs the real ``ruff`` on PATH (the dev pin in CI; verified locally
        against the product pin ruff==0.17.3 by putting that binary first on
        PATH). Without ``--unsafe-fixes`` the assignment survives.
        """
        code = "def f():\n    unused = compute()\n    return 1\n"
        result = await formatter.apply_ruff_unused_removal("test.py", code)
        assert "unused" not in result
        assert result == "def f():\n    compute()\n    return 1\n"

    @pytest.mark.asyncio
    async def test_import_sorting_never_applies_unsafe_fixes(self, formatter):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok\n", stderr="")
            await formatter.apply_ruff_import_sorting("test.py", "code\n")
            await formatter.apply_ruff_formatting("test.py", "code\n")
        for call in mock_run.call_args_list:
            assert "--unsafe-fixes" not in call[0][0]

    @pytest.mark.asyncio
    async def test_ruff_fix_abnormal_exit_raises(self, formatter):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=2, stdout="", stderr="ruff failed")
            with pytest.raises(RuffFixError, match="ruff failed"):
                await formatter.apply_ruff_import_sorting("test.py", "import os\n")


class TestCodeValidator:
    @pytest.mark.asyncio
    async def test_validate_syntax_python_valid(self, validator):
        assert await validator.validate_syntax("x = 1\n", "python") is True

    @pytest.mark.asyncio
    async def test_validate_syntax_python_invalid(self, validator):
        assert await validator.validate_syntax("def foo(\n", "python") is False

    @pytest.mark.asyncio
    async def test_validate_syntax_javascript(self, validator):
        assert await validator.validate_syntax("const x = 1;", "javascript") is True

    @pytest.mark.asyncio
    async def test_validate_syntax_typescript(self, validator):
        assert await validator.validate_syntax("const x: number = 1;", "typescript") is True

    @pytest.mark.asyncio
    async def test_validate_syntax_unknown_language(self, validator):
        assert await validator.validate_syntax("anything", "rust") is True

    @pytest.mark.asyncio
    async def test_validate_python_code_clean(self, validator):
        errors = await validator.validate_python_code("x = 1\ny = 2\n")
        assert errors == []

    @pytest.mark.asyncio
    async def test_validate_python_code_eval(self, validator):
        errors = await validator.validate_python_code("result = eval('2+2')")
        assert any("eval" in e for e in errors)

    @pytest.mark.asyncio
    async def test_validate_python_code_exec(self, validator):
        errors = await validator.validate_python_code("exec('print(1)')")
        assert any("exec" in e for e in errors)

    @pytest.mark.asyncio
    async def test_validate_python_code_incomplete_assignment(self, validator):
        errors = await validator.validate_python_code("x =")
        assert any("incomplete" in e.lower() for e in errors)

    @pytest.mark.asyncio
    async def test_validate_javascript_code_clean(self, validator):
        errors = await validator.validate_javascript_code("const x = 1;")
        assert errors == []

    @pytest.mark.asyncio
    async def test_validate_javascript_code_eval(self, validator):
        errors = await validator.validate_javascript_code("eval('2+2');")
        assert any("eval" in e for e in errors)

    @pytest.mark.asyncio
    async def test_validate_javascript_mismatched_braces(self, validator):
        errors = await validator.validate_javascript_code("function f() { if (true) {}")
        assert any("braces" in e.lower() for e in errors)

    @pytest.mark.asyncio
    async def test_validate_javascript_mismatched_parens(self, validator):
        errors = await validator.validate_javascript_code("f((x)")
        assert any("parentheses" in e.lower() for e in errors)

    @pytest.mark.asyncio
    async def test_validate_javascript_balanced(self, validator):
        errors = await validator.validate_javascript_code("function f() { return (1 + 2); }")
        assert errors == []


class TestFixFormatterAlias:
    def test_backward_compat_alias(self):
        assert FixFormatter is CodeFormatter
