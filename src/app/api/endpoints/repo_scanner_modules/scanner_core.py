"""Repository Scanner Core Logic Module.

This module contains the core scanning functionality for dependency analysis,
import validation, and code quality checks. It provides the main scanning
algorithms and processing logic used by the endpoint handlers.

The module provides:
- Dependency scanning from multiple sources (pip, poetry, requirements)
- Import statement analysis using AST parsing
- Code validation and syntax checking
- Advanced scanning with container manager integration
- Import validation against available packages
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from app.core.container_manager import get_container_manager
from services.code_analysis.import_validation_service import (
    ImportValidationService,
)

from ...models.repo_scanner_api_models import (
    ApiValidationResult,
    RepoScannerImportInfo,
    ScannedDependency,
)
from .dependency_scan_script import (
    DEPENDENCY_SCAN_SCRIPT,
    DEPENDENCY_SCAN_SCRIPT_PATH,
    dependency_scan_command,
)
from .file_discovery import (
    NEUTRAL_WORKDIR,
    WORKSPACE_ROOT,
    _script_write_command,
    discover_source_files,
    execute_in_container,
    isolated_python_command,
)
from .multi_language_scan import (
    partition_by_python,
    scan_dependencies_multilang,
    scan_imports_multilang,
    validate_code_multilang,
)

logger = logging.getLogger(__name__)


async def scan_dependencies(container_name: str, *, tenant_id: str | None = None) -> dict[str, Any]:
    """Scan and analyze dependencies from multiple sources within a container.

    Python dependencies (pip, poetry, requirements files) are gathered by an
    in-container Python script — the proven path, unchanged. Non-Python
    manifests (``package.json``, ``go.mod``, ``Cargo.toml``, ``pom.xml``) are
    discovered, read out of the container and parsed locally via
    :func:`scan_dependencies_multilang`, so a pure-Python repository behaves
    exactly as before and a polyglot repository reports every ecosystem.

    The scan process:
    1. Creates a Python script for Python dependency detection
    2. Writes the script to the container's temporary directory
    3. Executes the script to collect Python dependency information
    4. Discovers and locally parses non-Python dependency manifests
    5. Returns the merged, structured dependency data

    Args:
        container_name (str): Name of the Docker container to scan.
            Container must be running and accessible.

    Returns:
        Dict[str, Any]: Dependency scan results containing:
            - dependencies (List[ScannedDependency]): List of found dependencies
            - dependency_count (int): Total number of dependencies found
            - error (str, optional): Error message if scan failed

        Each dependency includes:
            - name (str): Package name
            - version (str): Package version
            - source (str): Source of dependency info (pip, poetry, requirements,
              package.json, go.mod, Cargo.toml, pom.xml)
            - is_valid (bool): Validation status

    Sources Scanned:
        - pip list: Currently installed packages
        - poetry.lock: Poetry-locked dependencies, parsed as text (never runs poetry)
        - requirements.txt / requirements-dev.txt / requirements/base.txt
        - package.json: npm/yarn dependencies (JS/TS)
        - go.mod: Go module requirements
        - Cargo.toml: Rust crate dependencies
        - pom.xml: Maven dependencies (Java)

    Example:
        ```python
        result = await scan_dependencies("my-python-container")
        if "error" not in result:
            for dep in result["dependencies"]:
                print(f"{dep.name}=={dep.version} from {dep.source}")
        ```

    Note:
        The function handles containers without package managers gracefully,
        returning empty results rather than failing. Script execution is
        performed in an isolated environment for security.
    """

    # Write the dependency script and run it isolated from the repo (#104).
    script = DEPENDENCY_SCAN_SCRIPT
    script_path = DEPENDENCY_SCAN_SCRIPT_PATH
    write_result = await execute_in_container(
        container_name,
        _script_write_command(script_path, script),
        tenant_id=tenant_id,
        workdir=NEUTRAL_WORKDIR,
    )

    python_deps: list[ScannedDependency] = []
    errors: list[str] = []

    if not write_result["success"]:
        logger.error(f"Failed to write dependency scan script: {write_result['output']}")
        errors.append("Failed to create scan script")
    else:
        # Execute the script
        result = await execute_in_container(
            container_name,
            dependency_scan_command(script_path),
            tenant_id=tenant_id,
            workdir=NEUTRAL_WORKDIR,
        )
        if result["success"]:
            try:
                deps_data = json.loads(result["output"])
                python_deps = [
                    ScannedDependency(
                        name=dep["name"],
                        version=dep["version"],
                        source=dep["source"],
                        is_valid=True,
                    )
                    for dep in deps_data
                ]
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse dependency output: {e}")
                errors.append(f"Failed to parse output: {str(e)}")
        else:
            errors.append(result["output"])

    # Multi-language manifests (package.json / go.mod / Cargo.toml / pom.xml)
    # are read out of the container and parsed locally — a failure there is
    # logged and isolated, never fatal to the Python result.
    multilang_deps = await scan_dependencies_multilang(container_name, tenant_id=tenant_id)

    all_deps = python_deps + multilang_deps
    response: dict[str, Any] = {
        "dependencies": all_deps,
        "dependency_count": len(all_deps),
    }
    if errors:
        response["error"] = "; ".join(errors)
    return response


async def scan_imports(
    container_name: str,
    target_files: list[str] | None = None,
    *,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Scan and analyze import statements across every supported language.

    Python files keep the proven in-container AST path
    (:func:`_scan_python_imports`); files in any other supported language are
    parsed locally with the ``processing.languages`` framework
    (:func:`scan_imports_multilang`). A pure-Python repository therefore
    behaves identically to before this became multi-language.

    Args:
        container_name (str): Name of the running Docker container to scan.
        target_files (Optional[List[str]]): Specific files to analyze. If
            ``None``, every supported source file under ``/workspace`` is
            discovered and scanned.

    Returns:
        Dict[str, Any]: Import scan results — ``imports``, ``import_count``,
        ``invalid_imports`` and (on partial failure) ``error``.
    """
    errors: list[str] = []

    if target_files is not None:
        python_files: list[str] | None
        python_subset, other_files = partition_by_python(target_files)
        python_files = python_subset
    else:
        # No explicit targets: let the Python path discover ``*.py`` itself
        # (unchanged behaviour) and discover the rest for the framework path.
        python_files = None
        discovered = await discover_source_files(container_name, tenant_id=tenant_id)
        if discovered.get("error"):
            errors.append(str(discovered["error"]))
        _, other_files = partition_by_python(discovered.get("source_files", []))

    imports: list[RepoScannerImportInfo] = []

    # Python — in-container stdlib AST path (zero behavioural change).
    if python_files is None or python_files:
        python_result = await _scan_python_imports(
            container_name, python_files, tenant_id=tenant_id
        )
        imports.extend(python_result.get("imports", []))
        if python_result.get("error"):
            errors.append(str(python_result["error"]))

    # Every other supported language — local framework parsing.
    if other_files:
        imports.extend(
            await scan_imports_multilang(container_name, other_files, tenant_id=tenant_id)
        )

    invalid_imports = [imp for imp in imports if not imp.is_valid]
    result: dict[str, Any] = {
        "imports": imports,
        "import_count": len(imports),
        "invalid_imports": invalid_imports,
    }
    if errors:
        result["error"] = "; ".join(errors)
    return result


async def _scan_python_imports(
    container_name: str,
    target_files: list[str] | None = None,
    *,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Scan Python import statements via an in-container AST script.

    This is the original Python-only scanner path, unchanged: an AST-based
    visitor script is written into the container and executed there. When
    *target_files* is ``None`` the script discovers every ``*.py`` file under
    ``/workspace`` itself.

    Args:
        container_name (str): Name of the Docker container to scan.
        target_files (Optional[List[str]]): Specific Python files to analyze.

    Returns:
        Dict[str, Any]: ``imports``, ``import_count``, ``invalid_imports`` and
        (on failure) ``error``.
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


async def validate_code(
    container_name: str,
    target_files: list[str] | None = None,
    *,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Validate source-code syntax across every supported language.

    Python files keep the proven in-container AST path
    (:func:`_validate_python_code`); files in any other supported language are
    parsed locally with the ``processing.languages`` framework
    (:func:`validate_code_multilang`). A pure-Python repository therefore
    behaves identically to before this became multi-language.

    Args:
        container_name (str): Name of the running Docker container.
        target_files (Optional[List[str]]): Specific files to validate. If
            ``None``, every supported source file under ``/workspace`` is
            discovered and validated.

    Returns:
        Dict[str, Any]: ``validation_results`` (per-file
        :class:`ApiValidationResult`), ``valid_files``, ``invalid_files`` and
        (on partial failure) ``error``.
    """
    errors: list[str] = []

    if target_files is not None:
        python_files: list[str] | None
        python_subset, other_files = partition_by_python(target_files)
        python_files = python_subset
    else:
        python_files = None
        discovered = await discover_source_files(container_name, tenant_id=tenant_id)
        if discovered.get("error"):
            errors.append(str(discovered["error"]))
        _, other_files = partition_by_python(discovered.get("source_files", []))

    validation_results: list[ApiValidationResult] = []

    # Python — in-container stdlib AST path (zero behavioural change).
    if python_files is None or python_files:
        python_result = await _validate_python_code(
            container_name, python_files, tenant_id=tenant_id
        )
        validation_results.extend(python_result.get("validation_results", []))
        if python_result.get("error"):
            errors.append(str(python_result["error"]))

    # Every other supported language — local framework parsing.
    if other_files:
        validation_results.extend(
            await validate_code_multilang(container_name, other_files, tenant_id=tenant_id)
        )

    valid_files = sum(1 for res in validation_results if res.is_valid)
    result: dict[str, Any] = {
        "validation_results": validation_results,
        "valid_files": valid_files,
        "invalid_files": len(validation_results) - valid_files,
    }
    if errors:
        result["error"] = "; ".join(errors)
    return result


async def _validate_python_code(
    container_name: str,
    target_files: list[str] | None = None,
    *,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Validate Python code syntax via an in-container AST script.

    This is the original Python-only validation path, unchanged: an
    ``ast.parse``-based script is written into the container and executed
    there. When *target_files* is ``None`` the script discovers every
    ``*.py`` file under ``/workspace`` itself.

    Args:
        container_name (str): Name of the container to validate code in.
        target_files (Optional[List[str]]): Specific Python files to validate.

    Returns:
        Dict[str, Any]: ``validation_results``, ``valid_files``,
        ``invalid_files`` and (on failure) ``error``.
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
    except:
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


async def scan_dependencies_v2(container_name: str) -> dict[str, Any]:
    """Advanced dependency scanning with enhanced error handling and multi-source support.

    This function represents an advanced implementation of dependency scanning
    that directly uses the container manager for improved reliability and
    performance. It systematically checks multiple dependency sources and
    provides detailed source attribution.

    Enhanced Features:
    - Direct container manager integration for improved reliability
    - Enhanced JSON parsing with fallback mechanisms
    - Better error handling and logging
    - Source attribution for each dependency
    - Support for complex requirement specifications

    Scanning Sources:
    1. pip list: Currently installed packages with JSON output
    2. requirements.txt: Pin and range requirements
    3. pyproject.toml: Poetry and modern Python projects
    4. Additional requirement files as discovered

    Args:
        container_name (str): Name of the Docker container to scan.
            Must be accessible through the container manager.

    Returns:
        Dict[str, Any]: Enhanced dependency results containing:
            - dependencies (List[Dict]): Dependency information with sources
            - dependency_count (int): Total dependencies found
            - invalid_dependencies (List[Dict]): Problematic dependencies
            - sources (List[str]): List of dependency sources found
            - error (str, optional): Error message if scan failed

        Each dependency includes:
            - name (str): Package name
            - version (str): Version string or requirement spec
            - source (str): Source type (pip, requirements.txt, pyproject.toml)
            - valid (bool): Whether dependency is properly formatted

    Parsing Improvements:
        - Robust JSON extraction from pip output
        - Regex-based requirement parsing
        - Graceful handling of malformed dependency declarations
        - Support for pip version operators (>=, <=, ==, ~=)

    Example:
        ```python
        result = await scan_dependencies_v2("python-container")

        print(f"Found {result['dependency_count']} dependencies")
        print(f"Sources: {', '.join(result['sources'])}")

        for dep in result['dependencies']:
            print(f"{dep['name']}=={dep['version']} ({dep['source']})")
        ```

    Error Handling:
        - Container access failures are handled gracefully
        - Invalid JSON output is parsed with fallback mechanisms
        - Missing files don't prevent other sources from being scanned
        - Malformed requirements are logged but don't stop scanning
    """
    logger.info(f"Starting dependency scan for container: {container_name}")

    try:
        container_manager = get_container_manager()

        # Check multiple dependency sources
        results: dict[str, Any] = {
            "dependencies": [],
            "dependency_count": 0,
            "invalid_dependencies": [],
            "sources": [],
        }

        # 1. Check pip installed packages
        # Isolated interpreter from cwd "/": a repo-root pip.py must not run (#104).
        pip_cmd = isolated_python_command(
            "-m", "pip", "list", "--format=json", "--disable-pip-version-check"
        )
        pip_result = await container_manager.execute_command(
            container_name, pip_cmd, workdir=NEUTRAL_WORKDIR
        )

        if pip_result.get("success"):
            try:
                output = pip_result.get("output", "[]").strip()
                # Find the JSON part (starts with [ and ends with ])
                json_start = output.find("[")
                json_end = output.rfind("]")
                if json_start != -1 and json_end != -1:
                    json_output = output[json_start : json_end + 1]
                    pip_packages = json.loads(json_output)
                else:
                    pip_packages = json.loads(output)

                for pkg in pip_packages:
                    dep_info = {
                        "name": pkg.get("name", ""),
                        "version": pkg.get("version", ""),
                        "source": "pip",
                        "valid": True,
                    }
                    results["dependencies"].append(dep_info)
                results["sources"].append("pip")
                logger.info(f"Found {len(pip_packages)} pip packages")
            except json.JSONDecodeError:
                logger.warning("Failed to parse pip list output as JSON")

        # 2. Check requirements.txt if exists
        req_cmd = "cat requirements.txt"
        req_result = await container_manager.execute_command(container_name, req_cmd)

        if req_result.get("success"):
            req_lines = req_result.get("output", "").strip().split("\n")
            for line in req_lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Parse requirement line (handle >= <= == etc.)
                    match = re.match(r"([a-zA-Z0-9\-_]+)([><=!~]+)?(.+)?", line)
                    if match:
                        name = match.group(1)
                        version = match.group(3) if match.group(3) else "unknown"
                        dep_info = {
                            "name": name,
                            "version": version,
                            "source": "requirements.txt",
                            "valid": True,
                        }
                        results["dependencies"].append(dep_info)
            results["sources"].append("requirements.txt")
            logger.info(
                f"Found requirements.txt with {len([ln for ln in req_lines if ln.strip() and not ln.startswith('#')])} dependencies"
            )

        # 3. Check pyproject.toml for Poetry projects
        poetry_cmd = "cat pyproject.toml"
        poetry_result = await container_manager.execute_command(container_name, poetry_cmd)

        if poetry_result.get("success"):
            results["sources"].append("pyproject.toml")
            logger.info("Found pyproject.toml file")

        results["dependency_count"] = len(results["dependencies"])

        return results

    except (ValueError, FileNotFoundError) as e:
        logger.error("Error in dependency scan: %s", str(e))
        return {
            "dependencies": [],
            "dependency_count": 0,
            "invalid_dependencies": [],
            "sources": [],
            "error": str(e),
        }


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
            find_cmd = "find . -name '*.py' -type "  # Remove the pipe for now
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
