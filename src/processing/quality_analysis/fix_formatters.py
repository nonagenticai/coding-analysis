"""
Code formatting utilities for fix application.
"""

from __future__ import annotations

import ast
import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class RuffFixError(RuntimeError):
    """A ruff autofix could not be applied (ruff missing, crashed, or refused the source).

    Raised, never swallowed: returning the content unchanged would read as
    "applied, nothing to change" -- a silent no-op reported as success.
    """


class CodeFormatter:
    """Handles code formatting for different languages and tools."""

    @staticmethod
    def _run_ruff_stdin(argv: list[str], content: str, ok_codes: tuple[int, ...]) -> str:
        """Run a ruff subcommand over ``content`` on stdin; return the rewritten source.

        ``--isolated``: the analysed (user) code must never pick up a
        ``pyproject.toml`` / ``ruff.toml`` discovered from this service's own
        working directory. ruff writes the rewritten source to stdout and its
        summary to stderr. ``ruff check --fix`` exits 1 when unfixable
        violations remain -- the fixes it could apply are still in stdout --
        while 2 is an abnormal termination (unparseable source, bad argv).

        ruff is installed in the service image outside uv.lock (``uv tool
        install ruff==0.17.3``, see Dockerfile); the repo's own dev pin is
        separate.

        Raises:
            RuffFixError: ruff is not installed, could not run, or exited abnormally.
        """
        command = [
            "ruff",
            argv[0],
            "--isolated",
            *argv[1:],
            "--stdin-filename",
            "fix_input.py",
            "-",
        ]
        try:
            result = subprocess.run(
                command,
                input=content,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as e:
            raise RuffFixError(f"ruff is not installed in this service image: {e}") from e
        except (subprocess.SubprocessError, OSError) as e:
            raise RuffFixError(f"ruff {argv[0]} could not run: {e}") from e
        if result.returncode not in ok_codes:
            raise RuffFixError(
                f"ruff {argv[0]} exited {result.returncode}: {(result.stderr or '').strip()[-500:]}"
            )
        return result.stdout

    async def _apply_ruff(
        self, label: str, argv: list[str], content: str, ok_codes: tuple[int, ...]
    ) -> str:
        try:
            return self._run_ruff_stdin(argv, content, ok_codes)
        except RuffFixError as e:
            logger.error("%s failed: %s", label, e)
            raise

    async def apply_ruff_formatting(self, _file_path: str, content: str) -> str:
        """Apply ``ruff format`` to Python code (replaces black)."""
        return await self._apply_ruff("ruff format", ["format"], content, (0,))

    async def apply_ruff_import_sorting(self, _file_path: str, content: str) -> str:
        """Sort imports with ``ruff check --select I --fix`` (replaces isort)."""
        return await self._apply_ruff(
            "ruff import sorting", ["check", "--select", "I", "--fix"], content, (0, 1)
        )

    async def apply_ruff_unused_removal(self, _file_path: str, content: str) -> str:
        """Remove unused imports/variables with ``ruff check --select F401,F841 --fix
        --unsafe-fixes`` (replaces autoflake).

        ruff marks the F841 (unused variable) fix as *unsafe*, so without
        ``--unsafe-fixes`` only unused imports go -- less than autoflake's
        ``--remove-unused-variables`` did. Scoped to this autoflake path only;
        no other ruff run applies unsafe fixes.
        """
        return await self._apply_ruff(
            "ruff unused removal",
            ["check", "--select", "F401,F841", "--fix", "--unsafe-fixes"],
            content,
            (0, 1),
        )

    async def apply_prettier_formatting(self, file_path: str, content: str) -> str:
        """Apply Prettier formatting to JavaScript/TypeScript code."""
        try:
            # Determine file extension for prettier
            ext = os.path.splitext(file_path)[1]

            # Write content to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            try:
                # Run prettier on the temporary file
                result = subprocess.run(
                    ["prettier", "--write", tmp_path],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if result.returncode == 0:
                    # Read the formatted content
                    with open(tmp_path, encoding="utf-8") as f:
                        formatted_content = f.read()
                    return formatted_content

                logger.warning("Prettier formatting failed: %s", result.stderr)
                return content
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

        except FileNotFoundError:
            logger.warning("Prettier not installed, skipping formatting")
            return content
        except (subprocess.SubprocessError, OSError) as e:
            logger.error("Error applying Prettier formatting: %s", e)
            return content


class QualityAnalysisCodeValidator:
    """Validates code syntax and patterns."""

    async def validate_syntax(self, code: str, language: str | None) -> bool:
        """Validate syntax for given code and language."""
        try:
            if language == "python":
                ast.parse(code)
                return True
            if language in ["javascript", "typescript"]:
                # Would need a JavaScript parser or external tool
                return True  # Streamlined validation
            # Unknown language, assume valid
            return True
        except SyntaxError:
            return False
        except (ValueError, TypeError) as e:
            logger.warning("Syntax validation error: %s", e)
            return False

    async def validate_python_code(self, code: str) -> list[str]:
        """Validate Python-specific code patterns."""
        errors = []

        # Check for basic issues
        if "eval(" in code:
            errors.append("Code contains potentially dangerous eval() call")

        if "exec(" in code:
            errors.append("Code contains potentially dangerous exec() call")

        # Check for common mistakes
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            if line.strip().endswith("="):
                errors.append(f"Line {i}: Assignment statement appears incomplete")

        return errors

    async def validate_javascript_code(self, code: str) -> list[str]:
        """Validate JavaScript-specific code patterns."""
        errors = []

        # Check for basic issues
        if "eval(" in code:
            errors.append("Code contains potentially dangerous eval() call")

        # Check for syntax patterns
        if code.count("{") != code.count("}"):
            errors.append("Mismatched braces in JavaScript code")

        if code.count("(") != code.count(")"):
            errors.append("Mismatched parentheses in JavaScript code")

        return errors


# Backward compatibility alias
FixFormatter = CodeFormatter
