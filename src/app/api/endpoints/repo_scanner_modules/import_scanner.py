"""Import Scanner Module.

This module provides import analysis functionality for Python code,
including AST-based import parsing and validation.

The module provides:
- Python import statement analysis using AST parsing
- Import validation against available packages
- Import categorization and relationship analysis
- Error handling for malformed import statements
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.container_manager import get_container_manager
from services.code_analysis.import_validation_service import (
    ImportValidationService,
)

from ...models.repo_scanner_api_models import RepoScannerImportInfo
from .file_discovery import (
    NEUTRAL_WORKDIR,
    WORKSPACE_ROOT,
    _script_write_command,
    execute_in_container,
    isolated_python_command,
)

logger = logging.getLogger(__name__)


async def scan_imports(
    container_name: str,
    target_files: list[str] | None = None,
    *,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Scan and analyze Python import statements using AST parsing.

    This function performs comprehensive import analysis by parsing Python files
    using the Abstract Syntax Tree (AST) to extract import statements. It can
    analyze either specific target files or scan the entire repository.

    The scanning process:
    1. Creates an AST-based import visitor script
    2. Deploys the script to the container
    3. Executes the script to parse Python files
    4. Collects detailed import information
    5. Returns structured import data with metadata

    Args:
        container_name (str): Name of the Docker container to scan.
            Container must be running and contain Python files.
        target_files (Optional[List[str]]): Specific files to analyze.
            If None, scans all Python files in /workspace.
            Files should be relative to container's working directory.

    Returns:
        Dict[str, Any]: Import scan results containing:
            - imports (List[RepoScannerImportInfo]): List of found import statements
            - import_count (int): Total number of imports found
            - invalid_imports (List[RepoScannerImportInfo]): Imports that failed validation
            - error (str, optional): Error message if scan failed

        Each import includes:
            - module (str): Module name being imported
            - name (str): Specific name imported (for 'from' imports)
            - alias (str): Import alias (if using 'as')
            - import_type (str): Type ('import' or 'from_import')
            - file_path (str): File containing the import
            - line_number (int): Line number of import statement
            - is_valid (bool): Whether import is valid/resolvable

    Import Types Detected:
        - Standard imports: `import module`
        - From imports: `from module import name`
        - Aliased imports: `import module as alias`
        - Complex from imports: `from package.module import name as alias`

    Example:
        ```python
        # Scan specific files
        result = await scan_imports(
            "python-container",
            ["src/main.py", "tests/test_main.py"]
        )

        # Scan entire repository
        result = await scan_imports("python-container")

        for imp in result["imports"]:
            print(f"{imp.file_path}:{imp.line_number} - {imp.module}")
        ```

    Error Handling:
        - Syntax errors in files are captured as error entries
        - Missing files are skipped gracefully
        - Parse failures are logged with error details
        - Invalid imports are tracked separately for analysis
    """

    # Create Python script for import scanning using AST
    script = '''
import ast
import os
import json
import sys
from pathlib import Path

class ImportVisitor(ast.NodeVisitor):
    def __init__(self, filepath) -> None:
        self.filepath = filepath
        self.imports = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append({
                "module": alias.name,
                "name": None,
                "alias": alias.asname,
                "import_type": "import",
                "file_path": self.filepath,
                "line_number": node.lineno,
                "is_valid": True
            })

    def visit_ImportFrom(self, node):
        module = node.module if node.module else ""
        for alias in node.names:
            self.imports.append({
                "module": module,
                "name": alias.name,
                "alias": alias.asname,
                "import_type": "from_import",
                "file_path": self.filepath,
                "line_number": node.lineno,
                "is_valid": True
            })

def scan_file_imports(filepath, display_path=None):
    """Scan imports in a single Python file."""
    display_path = display_path or filepath
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=display_path)
        visitor = ImportVisitor(display_path)
        visitor.visit(tree)
        return visitor.imports
    except Exception as e:
        return [{
            "module": "ERROR",
            "name": None,
            "alias": None,
            "import_type": "error",
            "file_path": display_path,
            "line_number": 0,
            "is_valid": False,
            "error_message": str(e)
        }]

def find_python_files(directory="."):
    """Find all Python files in directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip common ignore patterns
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

        for file in files:
            if file.endswith('.py') and not file.startswith('.'):
                python_files.append(os.path.join(root, file))

    return python_files

# Find and scan all Python files
all_imports = []
# argv[1] is the repo root, passed explicitly: this interpreter runs isolated
# (-I -P) from cwd "/", never from the checkout (coding-analysis#104).
root = sys.argv[1]
target_files = sys.argv[2:]

if target_files:
    files_to_scan = target_files
else:
    files_to_scan = find_python_files(root)

for filepath in files_to_scan:
    # Relative targets are relative to the repo root; reported as given.
    resolved = os.path.join(root, filepath)
    if os.path.exists(resolved):
        file_imports = scan_file_imports(resolved, filepath)
        all_imports.extend(file_imports)

print(json.dumps(all_imports, indent=2))
'''

    # Write and execute the import scanning script
    script_path = "/tmp/scan_imports.py"  # nosec B108
    write_result = await execute_in_container(
        container_name,
        _script_write_command(script_path, script),
        tenant_id=tenant_id,
        workdir=NEUTRAL_WORKDIR,
    )

    if not write_result["success"]:
        return {"imports": [], "error": "Failed to create scan script"}

    # Execute with optional target files
    cmd = isolated_python_command(script_path, WORKSPACE_ROOT, *(target_files or []))
    result = await execute_in_container(
        container_name, cmd, tenant_id=tenant_id, workdir=NEUTRAL_WORKDIR
    )

    if result["success"]:
        try:
            imports_data = json.loads(result["output"])
            logger.info(f"Import data sample: {imports_data[:1] if imports_data else 'None'}")
            imports = [RepoScannerImportInfo(**imp) for imp in imports_data]
            invalid_imports = [imp for imp in imports if not imp.is_valid]

            return {
                "imports": imports,
                "import_count": len(imports),
                "invalid_imports": invalid_imports,
            }
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            return {
                "imports": [],
                "error": f"Failed to parse output: {str(e)}",
            }
    else:
        return {"imports": [], "error": result["output"]}


async def validate_scanned_imports(
    container_name: str,
    imports: list[dict[str, Any]],
    *,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Validate scanned import statements against available dependencies and modules.

    This function takes raw import data from the import scanner and validates
    each import against the container's Python environment. It uses the
    ImportValidationService to check import resolvability and categorize
    imports by type (stdlib, third-party, local).

    Validation Process:
    1. Initialize ImportValidationService with container project root
    2. Validate imports against available packages and modules
    3. Categorize imports (standard library, third-party, local)
    4. Identify missing dependencies and resolution issues
    5. Generate suggestions for import improvements

    Args:
        container_name (str): Name of the container containing the imports.
            Used to determine project root and available packages.
        imports (List[Dict[str, Any]]): Raw import data from scan_imports.
            Each Dict[str, Any] should contain module, name, file_path, line_number, etc.

    Returns:
        Dict[str, Any]: Validation results containing:
            - imports (List[Dict]): All imports with updated validation status
            - import_count (int): Total number of imports processed
            - invalid_imports (List[Dict]): Imports that failed validation
            - validation_summary (Dict): Detailed validation statistics including:
                - valid_count (int): Number of valid imports
                - invalid_count (int): Number of invalid imports
                - stdlib_count (int): Standard library imports
                - third_party_count (int): Third-party package imports
                - local_count (int): Local module imports
                - missing_dependencies (List[str]): Unresolvable dependencies
                - suggestions (List[str]): Improvement recommendations
            - validation_error (str, optional): Error if validation failed

    Import Categories:
        - Standard Library: Built-in Python modules (os, sys, json, etc.)
        - Third Party: Installed packages (requests, numpy, etc.)
        - Local: Project-specific modules and packages
        - Invalid: Unresolvable or missing imports

    Example:
        ```python
        # Validate imports after scanning
        raw_imports = await scan_imports("container-name")
        validated = await validate_scanned_imports(
            "container-name",
            raw_imports["imports"]
        )

        print(f"Valid: {validated['validation_summary']['valid_count']}")
        print(f"Invalid: {validated['validation_summary']['invalid_count']}")
        for missing in validated['validation_summary']['missing_dependencies']:
            print(f"Missing: {missing}")
        ```

    Error Recovery:
        If validation service fails, returns original imports with default
        validation status to ensure the API remains functional.
    """
    try:
        # Use the container's working directory as project root
        # This assumes dependencies are defined in the container
        container_manager = get_container_manager()
        try:
            await container_manager.get_container(container_name, tenant_id)
            # Try to determine project root from container working directory
            project_root = "/app"  # Most containers use /app as working directory
        except ValueError:
            logger.warning(f"Container {container_name} not found, using default project root")
            project_root = "/app"  # Default container path

        # Initialize import validation service
        validation_service = ImportValidationService(project_root)

        # Validate the imports
        # Note: For now, we use a placeholder code_graph_id since we don't have
        # a proper graph ID context in this endpoint
        validation_result = await validation_service.validate_imports(
            code_graph_id=f"container_{container_name}", imports=imports
        )

        # Update the original import objects with validation results
        validated_imports = []
        invalid_imports = []

        for import_info in imports:
            # Check if this import is in the validation results
            is_valid = any(
                vi["module"] == import_info.get("module")
                and vi.get("name") == import_info.get("name")
                for vi in validation_result.valid_imports
            )

            # Update the import info with validation result
            updated_import = {**import_info, "is_valid": is_valid}

            if is_valid:
                validated_imports.append(updated_import)
            else:
                invalid_imports.append(updated_import)

        return {
            "imports": validated_imports + invalid_imports,
            "import_count": len(validated_imports + invalid_imports),
            "invalid_imports": invalid_imports,
            "validation_summary": {
                "valid_count": len(validation_result.valid_imports),
                "invalid_count": len(validation_result.invalid_imports),
                "stdlib_count": len(validation_result.stdlib_imports),
                "third_party_count": len(validation_result.third_party_imports),
                "local_count": len(validation_result.local_imports),
                "missing_dependencies": validation_result.missing_dependencies,
                "suggestions": validation_result.suggestions,
            },
        }

    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Error validating imports: {e}")
        # Return original imports with default validation
        return {
            "imports": imports,
            "import_count": len(imports),
            "invalid_imports": [],
            "validation_error": str(e),
        }
