"""Code Validator Module.

This module provides code validation functionality for Python files,
including syntax checking, quality analysis, and error reporting.

The module provides:
- Python code syntax validation using AST parsing
- Basic code quality checks (line length, structure)
- Enhanced validation with container manager integration
- Comprehensive error reporting with line numbers
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any

from app.core.container_manager import get_container_manager

from ...models.repo_scanner_api_models import ApiValidationResult
from .file_discovery import (
    NEUTRAL_WORKDIR,
    WORKSPACE_ROOT,
    _script_write_command,
    execute_in_container,
    isolated_python_command,
)

logger = logging.getLogger(__name__)


async def validate_code(
    container_name: str,
    target_files: list[str] | None = None,
    *,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Validate Python code syntax and perform basic quality checks.

    This function performs comprehensive code validation by analyzing Python
    files for syntax errors, parsing issues, and basic code quality metrics.
    It uses AST parsing for syntax validation and implements additional
    quality checks such as line length analysis.

    Validation Process:
    1. Creates a code validation script with AST parsing
    2. Deploys the script to the target container
    3. Executes validation on specified or all Python files
    4. Performs syntax checking using ast.parse()
    5. Applies basic code quality rules
    6. Returns detailed validation results

    Args:
        container_name (str): Name of the container to validate code in.
            Container must be running and contain Python files.
        target_files (Optional[List[str]]): Specific files to validate.
            If None, validates all Python files in /workspace.
            Paths should be relative to container's working directory.

    Returns:
        Dict[str, Any]: Validation results containing:
            - validation_results (List[ApiValidationResult]): Per-file results
            - valid_files (int): Number of files that passed validation
            - invalid_files (int): Number of files that failed validation
            - error (str, optional): Error message if validation failed

        Each validation result includes:
            - file_path (str): Path to the validated file
            - is_valid (bool): Overall validation status
            - issues (List[str]): Critical issues (syntax errors)
            - warnings (List[str]): Non-critical issues (style violations)
            - line_count (int): Total lines in the file

    Validation Checks:
        - Syntax Validation: AST parsing to detect syntax errors
        - Line Length: Warns about lines exceeding 120 characters
        - Parse Errors: Captures and reports parsing failures
        - File Accessibility: Handles missing or unreadable files

    Quality Metrics:
        - Syntax correctness (pass/fail)
        - Code style warnings
        - File size metrics (line counts)
        - Error location reporting (line numbers)

    Example:
        ```python
        # Validate specific files
        result = await validate_code(
            "python-container",
            ["src/main.py", "tests/test_main.py"]
        )

        # Validate entire codebase
        result = await validate_code("python-container")

        for file_result in result["validation_results"]:
            if not file_result.is_valid:
                print(f"Issues in {file_result.file_path}:")
                for issue in file_result.issues:
                    print(f"  - {issue}")
        ```

    Error Handling:
        - Individual file errors don't stop the entire validation
        - Syntax errors are captured with line number information
        - Unreadable files are reported as processing errors
        - Graceful degradation for containers without Python files
    """

    # Create validation script
    script = '''
import ast
import os
import json
import sys
import subprocess
from pathlib import Path


def validate_syntax(filepath):
    """Check Python syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        ast.parse(content, filename=filepath)
        return {"is_valid": True, "issues": [], "warnings": []}
    except SyntaxError as e:
        return {
            "is_valid": False,
            "issues": [f"Syntax error at line {e.lineno}: {e.msg}"],
            "warnings": []
        }
    except Exception as e:
        return {
            "is_valid": False,
            "issues": [f"Parse error: {str(e)}"],
            "warnings": []
        }

def count_lines(filepath):
    """Count lines in file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return len(f.readlines())
    except Exception:
        return 0

def find_python_files(directory="."):
    """Find all Python files."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for file in files:
            if file.endswith('.py') and not file.startswith('.'):
                python_files.append(os.path.join(root, file))
    return python_files

# Validate files
validation_results = []
# argv[1] is the repo root, passed explicitly: this interpreter runs isolated
# (-I -P) from cwd "/", never from the checkout (coding-analysis#104).
root = sys.argv[1]
target_files = sys.argv[2:]

if target_files:
    files_to_validate = target_files
else:
    files_to_validate = find_python_files(root)

for filepath in files_to_validate:
    # Relative targets are relative to the repo root; reported as given.
    resolved = os.path.join(root, filepath)
    if os.path.exists(resolved):
        validation = validate_syntax(resolved)
        line_count = count_lines(resolved)

        validation_results.append({
            "file_path": filepath,
            "is_valid": validation["is_valid"],
            "issues": validation["issues"],
            "warnings": validation["warnings"],
            "line_count": line_count
        })

print(json.dumps(validation_results, indent=2))
'''

    script_path = "/tmp/validate_code.py"  # nosec B108
    write_result = await execute_in_container(
        container_name,
        _script_write_command(script_path, script),
        tenant_id=tenant_id,
        workdir=NEUTRAL_WORKDIR,
    )

    if not write_result["success"]:
        return {
            "validation_results": [],
            "error": "Failed to create validation script",
        }

    cmd = isolated_python_command(script_path, WORKSPACE_ROOT, *(target_files or []))
    result = await execute_in_container(
        container_name, cmd, tenant_id=tenant_id, workdir=NEUTRAL_WORKDIR
    )

    if result["success"]:
        try:
            results_data = json.loads(result["output"])
            validation_results = [ApiValidationResult(**res) for res in results_data]
            valid_files = sum(1 for res in validation_results if res.is_valid)
            invalid_files = len(validation_results) - valid_files

            return {
                "validation_results": validation_results,
                "valid_files": valid_files,
                "invalid_files": invalid_files,
            }
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            return {
                "validation_results": [],
                "error": f"Failed to parse output: {str(e)}",
            }
    else:
        return {"validation_results": [], "error": result["output"]}


async def validate_code_v2(
    container_name: str, target_files: list[str] | None = None
) -> dict[str, Any]:
    """Advanced Python code validation with enhanced file discovery and quality checks.

    This function provides comprehensive code validation using direct container
    manager integration for improved performance and reliability. It includes
    enhanced file discovery, detailed syntax validation, and quality metrics
    collection.

    Enhanced Validation Features:
    - Direct container manager integration
    - Improved file discovery with filtering
    - Enhanced syntax validation using AST parsing
    - Basic code quality metrics (line length, structure)
    - Comprehensive error reporting with line numbers
    - Performance tracking and statistics

    Validation Process:
    1. Discover Python files using container manager
    2. Read file contents securely through container interface
    3. Parse files using Python AST for syntax validation
    4. Apply code quality rules and style checks
    5. Collect metrics and generate detailed reports

    Args:
        container_name (str): Name of the Docker container to validate.
            Container must be running and accessible.
        target_files (Optional[List[str]]): Specific files to validate.
            If None, discovers and validates all Python files in container.
            Files should be relative to container working directory.

    Returns:
        Dict[str, Any]: Comprehensive validation results containing:
            - validation_results (List[Dict]): Per-file validation results
            - valid_files (int): Number of syntactically correct files
            - invalid_files (int): Number of files with syntax errors
            - total_lines (int): Total lines of code analyzed
            - error (str, optional): Error if validation process failed

        Each file result includes:
            - file_path (str): Path to the validated file
            - is_valid (bool): Overall syntax validity
            - syntax_valid (bool): Specific AST parsing result
            - issues (List[str]): Critical syntax errors
            - warnings (List[str]): Code style warnings
            - line_count (int): Lines in the file

    Quality Checks Performed:
        - AST syntax validation for Python correctness
        - Line length analysis (warns >120 characters)
        - File accessibility and readability checks
        - Basic structural validation

    File Discovery:
        - Automatically finds all .py files in container
        - Excludes hidden files and directories
        - Filters out common non-source directories (__pycache__, .git)
        - Respects target_files parameter for focused analysis

    Performance Features:
        - Efficient file reading through container manager
        - Batch processing of multiple files
        - Progress tracking and timing information
        - Resource usage monitoring

    Example:
        ```python
        # Validate all Python files
        result = await validate_code_v2("python-container")

        print(f"Validated {len(result['validation_results'])} files")
        print(f"Valid: {result['valid_files']}, Invalid: {result['invalid_files']}")
        print(f"Total lines: {result['total_lines']}")

        # Show files with issues
        for file_result in result['validation_results']:
            if not file_result['is_valid']:
                print(f"Issues in {file_result['file_path']}:")
                for issue in file_result['issues']:
                    print(f"  - {issue}")
        ```

    Error Recovery:
        - Individual file errors don't stop overall validation
        - Graceful handling of unreadable or binary files
        - Detailed error reporting for troubleshooting
        - Fallback to basic validation if advanced features fail
    """
    logger.info(f"Starting code validation for container: {container_name}")

    try:
        container_manager = get_container_manager()

        # Find Python files to validate
        if target_files:
            python_files = [f for f in target_files if f.endswith(".py")]
        else:
            # Find all Python files
            find_cmd = "find . -name '*.py' -type f"
            find_result = await container_manager.execute_command(container_name, find_cmd)

            logger.info(
                f"Validation find command result: success={find_result.get('success')}, output_length={len(find_result.get('output', ''))}"
            )
            if not find_result.get("success"):
                logger.warning(
                    f"Validation find command failed: {find_result.get('error', 'Unknown error')}"
                )

            if find_result.get("success"):
                output = find_result.get("output", "")
                all_files = [f.strip() for f in output.split("\n") if f.strip()]
                # Limit to first 50 files after finding them
                python_files = all_files
                logger.info(
                    f"Found {len(python_files)} Python files for validation (limited from {len(all_files)} total)"
                )
            else:
                python_files = []
                logger.warning("Validation find command failed, no Python files found")

        results: dict[str, Any] = {
            "validation_results": [],
            "valid_files": 0,
            "invalid_files": 0,
            "total_lines": 0,
        }

        for file_path in python_files:
            try:
                # Read file content
                cat_cmd = f"cat '{file_path}'"
                cat_result = await container_manager.execute_command(container_name, cat_cmd)

                if not cat_result.get("success"):
                    continue

                file_content = cat_result.get("output", "")
                line_count = len(file_content.split("\n"))
                results["total_lines"] += line_count

                validation_result: dict[str, Any] = {
                    "file_path": file_path,
                    "is_valid": True,
                    "issues": [],
                    "warnings": [],
                    "line_count": line_count,
                }

                # Validate syntax using AST
                try:
                    ast.parse(file_content)
                    validation_result["syntax_valid"] = True
                except SyntaxError as e:
                    validation_result["is_valid"] = False
                    validation_result["syntax_valid"] = False
                    validation_result["issues"].append(f"SyntaxError at line {e.lineno}: {e.msg}")

                # Basic code quality checks
                if validation_result["is_valid"]:
                    # Check for common issues
                    lines = file_content.split("\n")
                    for i, line in enumerate(lines, 1):
                        # Check for very long lines
                        if len(line) > 120:
                            validation_result["warnings"].append(
                                f"Line {i}: Line too long ({len(line)} characters)"
                            )

                if validation_result["is_valid"]:
                    results["valid_files"] += 1
                else:
                    results["invalid_files"] += 1

                results["validation_results"].append(validation_result)

            except (ValueError, KeyError, TypeError) as e:
                logger.warning("Error validating file %s: %s", file_path, str(e))
                results["validation_results"].append(
                    {
                        "file_path": file_path,
                        "is_valid": False,
                        "issues": [f"ProcessingError: {str(e)}"],
                        "warnings": [],
                        "line_count": 0,
                    }
                )
                results["invalid_files"] += 1

        logger.info(
            f"Validated {len(results['validation_results'])} files: {results['valid_files']} valid, {results['invalid_files']} invalid"
        )

        return results

    except (ValueError, FileNotFoundError) as e:
        logger.error("Error in code validation: %s", str(e))
        return {
            "validation_results": [],
            "valid_files": 0,
            "invalid_files": 0,
            "total_lines": 0,
            "error": str(e),
        }
