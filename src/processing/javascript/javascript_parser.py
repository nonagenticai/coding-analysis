"""Module for parsing JavaScript code."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Any

from app.core.security_config import CommandConfig, SecurityError, secure_runner

from ..base_parser import CodeParser

logger = logging.getLogger(__name__)


class JavaScriptParser(CodeParser):
    """Specialized parser for JavaScript and TypeScript code using ESLint parser."""

    def __init__(self):
        """Initialize the parser and check dependencies."""
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check if required Node.js dependencies are installed."""
        try:
            # Check for Node.js using secure runner
            result = secure_runner.run_command(["node", "--version"], CommandConfig(timeout=10))
            if result.returncode != 0:
                logger.warning("Node.js not found. JavaScript/TypeScript processing may not work.")
                return

            # Check for ESLint parser (globally installed)
            result = secure_runner.run_command(
                ["npm", "list", "-g", "@typescript-eslint/parser", "--json"],
                CommandConfig(timeout=15),
            )

            if result.returncode != 0:
                logger.warning(
                    "@typescript-eslint/parser not found. Some JavaScript/TypeScript features may not work."
                )
                logger.info("Install with: npm install -g @typescript-eslint/parser")

        except SecurityError as e:
            logger.error("Security error checking JavaScript dependencies: %s", e)
            logger.warning("JavaScript/TypeScript processing may not work correctly")
        except (OSError, subprocess.SubprocessError) as e:
            logger.error("Error checking JavaScript dependencies: %s", e)
            logger.warning("JavaScript/TypeScript processing may not work correctly")

    async def parse_code(
        self,
        code: str,
        file_path: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Parse JavaScript or TypeScript code using Node.js and ESLint parser.

        Args:
            code: Code content to parse
            file_path: Path of the original file (for error reporting)
            options: Optional parser options, including is_typescript flag

        Returns:
            Parsed AST as a dictionary, or None if parsing failed
        """

        code_path = None
        script_path = None

        # Extract options
        is_typescript = options.get("is_typescript", False) if options else False

        try:
            # Create temporary files for the parsing operation
            with tempfile.NamedTemporaryFile(
                suffix=".js" if not is_typescript else ".ts",
                mode="w+",
                delete=False,
                encoding="utf-8",
            ) as code_file:
                code_file.write(code)
                code_file.flush()
                code_path = code_file.name

            with tempfile.NamedTemporaryFile(
                suffix=".js", mode="w+", delete=False, encoding="utf-8"
            ) as script_file:
                parser_script = self._generate_parser_script(code_path, file_path, is_typescript)
                script_file.write(parser_script)
                script_file.flush()
                script_path = script_file.name

            try:
                # Run the parser script using secure runner
                result = secure_runner.run_command(["node", script_path], CommandConfig(timeout=30))

                if result.returncode != 0:
                    self._handle_parse_error(result, file_path)
                    return None

                # Parse the AST from JSON
                parsed_result = json.loads(result.stdout)
                return parsed_result if isinstance(parsed_result, dict) else None

            finally:
                # Clean up temporary files
                self._cleanup_temp_files(script_path, code_path)

        except SecurityError as e:
            logger.error(
                "Security error parsing JavaScript/TypeScript code for %s: %s",
                file_path,
                e,
            )
            return None
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.error(
                "Error parsing JavaScript/TypeScript code for %s: %s",
                file_path,
                e,
            )
            return None

    def _generate_parser_script(self, code_path: str, file_path: str, is_typescript: bool) -> str:
        """Generate the Node.js parser script."""
        jsx_support = "ecmaFeatures: { jsx: true }," if file_path.endswith(("x", "sx")) else ""
        ts_config = "project: './tsconfig.json'," if is_typescript else ""

        return f"""
        const {{ parse }} = require('@typescript-eslint/parser');
        const fs = require('fs');

        try {{
            // Read the code from the temporary file
            const code = fs.readFileSync('{code_path}', 'utf8');

            // Parse the code
            const ast = parse(code, {{
                ecmaVersion: 'latest',
                sourceType: 'module',
                range: true,
                loc: true,
                {jsx_support}
                {ts_config}
            }});

            // Output the AST as JSON
            console.log(JSON.stringify(ast));
        }} catch (error) {{
            console.error(JSON.stringify({{ error: error.message }}));
            process.exit(1);
        }}
        """

    def _handle_parse_error(self, result: subprocess.CompletedProcess, file_path: str) -> None:
        """Handle parsing errors with proper logging."""
        try:
            error_data = json.loads(result.stderr)
            logger.error(
                "Parser error for %s: %s",
                file_path,
                error_data.get("error", "Unknown error"),
            )
        except json.JSONDecodeError:
            logger.error("Parser error for %s: %s", file_path, result.stderr)

    def _cleanup_temp_files(self, script_path: str | None, code_path: str | None) -> None:
        """Clean up temporary files safely."""
        for path in [script_path, code_path]:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass  # Files might already be deleted

    async def validate_syntax(
        self,
        code: str,
        file_path: str,
    ) -> bool:
        """Validate if the JavaScript/TypeScript code has valid syntax.

        Args:
            code: JavaScript/TypeScript source code content to validate
            file_path: Path to the file (for error reporting)

        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            # Use the same parsing logic but just return success/failure
            result = await self.parse_code(code, file_path)
            return result is not None
        except (OSError, ValueError) as e:
            logger.error(
                "Error validating JavaScript/TypeScript file %s: %s",
                file_path,
                str(e),
            )
            return False
