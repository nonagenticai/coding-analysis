"""File Discovery and Classification Module.

This module provides functionality for discovering and classifying files within
Docker containers, including container command execution capabilities and file
system operations for scanning purposes.

The module provides:
- Container command execution with error handling
- File discovery and filtering utilities
- File classification by type and purpose
- Security-focused container interaction
"""

from __future__ import annotations

import base64
import logging
import shlex
from typing import Any

from fastapi import HTTPException, status

from app.core.container_manager import get_container_manager
from clients.coding_containers import get_coding_containers_client
from processing.workspace_repos import repos_exclude_globs

logger = logging.getLogger(__name__)


#: Interpreter argv prefix for EVERY Python process the scanner starts in a
#: workspace (coding-analysis#104). The workspace holds an untrusted, freshly
#: imported repository, and auto-index scans it with no user action:
#:
#: * ``-I`` (isolated) ignores ``PYTHON*`` env vars and the user site dir.
#: * ``-P`` stops Python prepending the script's directory (``/tmp``) or, for
#:   ``-c`` / ``-m``, the cwd to ``sys.path``. Without it a repo-root
#:   ``base64.py`` or ``pip.py`` shadows the stdlib/pip module and executes.
#:   Needs Python >= 3.11 (workspace images are python:3.13+-slim). On 3.11+
#:   ``-I`` already implies ``-P``; it is spelled out so the guarantee does not
#:   silently depend on that implication if ``-I`` is ever dropped.
ISOLATED_PYTHON: tuple[str, ...] = ("python3", "-I", "-P")

#: cwd for every interpreter run: never the repository checkout, so nothing
#: resolves against repo files by accident. Paths go in as explicit arguments.
NEUTRAL_WORKDIR = "/"

#: The repository checkout inside the workspace, passed to scripts explicitly.
WORKSPACE_ROOT = "/workspace"


def isolated_python_command(*args: str) -> str:
    """Build an ``execute_in_container`` command running isolated ``python3``.

    Run the result with ``workdir=NEUTRAL_WORKDIR``. ``shlex.join`` quotes each
    argument, so file names with spaces survive the client's ``shlex.split``.
    """
    return shlex.join([*ISOLATED_PYTHON, *args])


def _script_write_command(script_path: str, script: str) -> str:
    """Build an ``execute_in_container`` command that writes *script* to *script_path*.

    Callers used to construct ``python3 -c "import json; open(\\"{path}\\", \\"w\\")
    .write({json.dumps(script)})"`` inline, which nests two layers of double
    quotes through an f-string and breaks under client-side shlex.split (the
    inner ``json.dumps`` output re-opens quotes mid-token). Base64-encoding the
    script body removes every shell-significant character from the inner
    literal, so shlex splits cleanly into ``["python3", "-I", "-P", "-c",
    "<decoder>"]`` and the coding-containers allowlist is satisfied.

    The decoder imports ``base64``: run it isolated and from
    ``NEUTRAL_WORKDIR``, or a repo-root ``base64.py`` would be imported instead.
    """
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    decoder = (
        "import base64; "
        f"open('{script_path}', 'w').write(base64.b64decode('{encoded}').decode('utf-8'))"
    )
    return isolated_python_command("-c", decoder)


async def execute_in_container(
    container_name: str,
    command: str,
    timeout: int = 300,
    *,
    tenant_id: str | None,
    workdir: str = WORKSPACE_ROOT,
) -> dict[str, Any]:
    """Execute a command inside a dockerized container with comprehensive error handling.

    This function provides a secure interface for executing commands within
    Docker containers. It validates container existence and status before
    execution, and handles various error conditions gracefully.

    Args:
        container_name (str): Name of the Docker container to execute command in.
            Must be an existing, running container managed by the container manager.
        command (str): Shell command to execute inside the container.
            Commands are executed in *workdir* with restricted privileges.
        timeout (int, optional): Command execution timeout in seconds.
            Defaults to 300 seconds. Currently unused but reserved for future use.
        tenant_id (str | None): The CALLER's tenant. Required, keyword-only and
            without a default so no call site can forget it: it scopes the
            container lookup AND is forwarded on the exec, because
            coding-containers' exec is tenant-scoped and an untenanted exec is
            resolved as this service's own account and 404'd for every
            customer workspace. ``None`` stays absent (never defaulted).
        workdir (str): cwd for the command. Defaults to the repository
            checkout; Python interpreter runs must pass ``NEUTRAL_WORKDIR``.

    Returns:
        Dict[str, Any]: Execution result containing:
            - exit_code (int): Command exit code (0 for success)
            - output (str): Command stdout/stderr output decoded as UTF-8
            - success (bool): True if exit_code is 0, False otherwise

    Raises:
        HTTPException:
            - 404 if container not found
            - 400 if container is not running
            - 500 for Docker API errors or unexpected failures

    Security:
        - Commands execute with restricted privileges (privileged=False)
        - Execution isolated to container environment
        - No TTY allocation for security
        - Working directory is /workspace unless *workdir* says otherwise

    Example:
        ```python
        result = await execute_in_container(
            container_name="analysis-container",
            command="python --version"
        )
        if result["success"]:
            print(f"Python version: {result['output']}")
        else:
            print(f"Command failed with code {result['exit_code']}")
        ```
    """
    try:
        container_manager = get_container_manager()

        # Check if container exists and is running
        container_info = None
        containers = await container_manager.list_containers(tenant_id=tenant_id)
        for container in containers:
            if container_name == container.container_name:
                container_info = container
                break

        if not container_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Container '{container_name}' not found",
            )

        if container_info.status != "running":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Container '{container_name}' is not running (status: {container_info.status})",
            )

        # Execute command in container via coding-containers HTTP API
        # (coding-analysis does NOT have the Docker socket; coding-containers does)
        cc_client = get_coding_containers_client()

        try:
            result = await cc_client.execute(
                container_name=container_info.container_name,
                command=command,
                workdir=workdir,
                timeout=timeout,
                tenant_id=tenant_id,
            )

            exit_code = result.get("exit_code", -1)
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            output = stdout if stdout else stderr

            return {
                "exit_code": exit_code,
                "output": output,
                "success": exit_code == 0,
            }

        except Exception as e:
            logger.error("Error executing command in container: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to execute command in container: {str(e)}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error in execute_in_container: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        )


def _repos_exclude_args(base_directory: str) -> str:
    """``find`` args keeping a walk from *base_directory* out of ``.repos/``.

    The source checkouts of a multi-repo workspace are not part of the
    repository rooted there (``processing.workspace_repos``).
    """
    return "".join(f" -not -path '{glob}'" for glob in repos_exclude_globs(base_directory))


async def discover_python_files(
    container_name: str,
    base_directory: str = "/workspace",
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Discover Python files within a container directory structure.

    This function performs intelligent discovery of Python files within a
    container, applying filtering rules to exclude common non-source files
    and directories. It provides detailed file classification and metadata.

    Discovery Process:
    1. Execute find command within the container
    2. Filter results to exclude hidden and cache directories
    3. Classify files by type and purpose
    4. Return structured file information

    Args:
        container_name (str): Name of the Docker container to scan.
            Container must be running and accessible.
        base_directory (str): Directory to start discovery from.
            Defaults to "/workspace" which is the standard container working directory.

    Returns:
        Dict[str, Any]: Discovery results containing:
            - python_files (List[str]): List of discovered Python file paths
            - file_count (int): Total number of Python files found
            - directories_scanned (List[str]): Directories that were searched
            - excluded_patterns (List[str]): Patterns used for exclusion
            - error (str, optional): Error message if discovery failed

    File Classification:
        - Source files: Regular .py files in src/, app/, lib/ directories
        - Test files: Files in test/, tests/ directories or with test_ prefix
        - Configuration: setup.py, __init__.py, config files
        - Scripts: Executable Python files, utility scripts

    Exclusion Patterns:
        - Hidden directories (starting with .)
        - Cache directories (__pycache__, .pytest_cache)
        - Build directories (build/, dist/, .eggs/)
        - Virtual environments (venv/, env/, .venv/)
        - IDE directories (.vscode/, .idea/)

    Example:
        ```python
        result = await discover_python_files("python-container")

        print(f"Found {result['file_count']} Python files")
        for file_path in result['python_files']:
            print(f"  {file_path}")
        ```

    Error Handling:
        - Container access failures are reported clearly
        - Permission issues are handled gracefully
        - Missing directories don't cause complete failure
        - Invalid file paths are filtered out automatically
    """
    try:
        # Build find command with exclusion patterns
        exclusion_patterns = [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "env",
            ".pytest_cache",
            ".mypy_cache",
            "build",
            "dist",
            ".eggs",
            ".tox",
            ".coverage",
            "htmlcov",
            ".vscode",
            ".idea",
            "node_modules",
        ]

        # Create find command with exclusions
        exclude_args = " ".join([f"-not -path '*/{pattern}/*'" for pattern in exclusion_patterns])
        exclude_args += _repos_exclude_args(base_directory)
        find_command = f"find {base_directory} -name '*.py' -type f {exclude_args}"

        logger.info(f"Discovering Python files in {container_name}:{base_directory}")

        result = await execute_in_container(container_name, find_command, tenant_id=tenant_id)

        if not result["success"]:
            return {
                "python_files": [],
                "file_count": 0,
                "directories_scanned": [base_directory],
                "excluded_patterns": exclusion_patterns,
                "error": f"File discovery failed: {result['output']}",
            }

        # Parse find output
        output = result["output"].strip()
        if not output:
            python_files = []
        else:
            python_files = [
                path.strip()
                for path in output.split("\n")
                if path.strip() and path.strip().endswith(".py")
            ]

        # Filter out any remaining unwanted files
        filtered_files = []
        for file_path in python_files:
            # Skip files with exclusion patterns in path
            skip_file = False
            for pattern in exclusion_patterns:
                if pattern in file_path:
                    skip_file = True
                    break

            if not skip_file:
                filtered_files.append(file_path)

        logger.info(f"Discovered {len(filtered_files)} Python files in {container_name}")

        return {
            "python_files": filtered_files,
            "file_count": len(filtered_files),
            "directories_scanned": [base_directory],
            "excluded_patterns": exclusion_patterns,
        }

    except Exception as e:
        logger.error(f"Error discovering Python files: {e}")
        return {
            "python_files": [],
            "file_count": 0,
            "directories_scanned": [],
            "excluded_patterns": [],
            "error": str(e),
        }


async def discover_source_files(
    container_name: str,
    base_directory: str = "/workspace",
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Discover source files in any supported language within a container.

    The multi-language counterpart of :func:`discover_python_files`: the
    ``find`` glob is built from every file extension the language framework
    supports (plus Python's ``.py``/``.pyw``), so polyglot repositories are
    fully covered rather than only their Python subset.

    Args:
        container_name (str): Name of the running Docker container to scan.
        base_directory (str): Directory to start discovery from. Defaults to
            ``/workspace``, the standard container working directory.
        tenant_id (str, optional): Tenant ID for container filtering.

    Returns:
        Dict[str, Any]: Discovery results containing:
            - source_files (List[str]): Discovered source file paths
            - file_count (int): Total number of source files found
            - directories_scanned (List[str]): Directories that were searched
            - excluded_patterns (List[str]): Patterns used for exclusion
            - error (str, optional): Error message if discovery failed
    """
    try:
        # Lazy import: registers the language framework only when a
        # multi-language scan is actually requested.
        from processing.languages import language_registry

        exclusion_patterns = [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "env",
            ".pytest_cache",
            ".mypy_cache",
            "build",
            "dist",
            ".eggs",
            ".tox",
            ".coverage",
            "htmlcov",
            ".vscode",
            ".idea",
            "node_modules",
        ]

        # Build the ``find`` name-group from every supported extension. The
        # alternation must be parenthesised so the ``-type``/``-not`` filters
        # apply to every extension, not just the last one.
        extensions = sorted(language_registry.get_supported_extensions() | {".py", ".pyw"})
        name_clause = " -o ".join(f"-name '*{ext}'" for ext in extensions)
        exclude_args = " ".join([f"-not -path '*/{pattern}/*'" for pattern in exclusion_patterns])
        exclude_args += _repos_exclude_args(base_directory)
        find_command = rf"find {base_directory} \( {name_clause} \) -type f {exclude_args}"

        logger.info(f"Discovering source files in {container_name}:{base_directory}")

        result = await execute_in_container(container_name, find_command, tenant_id=tenant_id)

        if not result["success"]:
            return {
                "source_files": [],
                "file_count": 0,
                "directories_scanned": [base_directory],
                "excluded_patterns": exclusion_patterns,
                "error": f"Source file discovery failed: {result['output']}",
            }

        output = result["output"].strip()
        candidates = [path.strip() for path in output.split("\n") if path.strip()] if output else []

        # Drop anything that still slipped through with an excluded path
        # component (defends against symlinks / unusual layouts).
        filtered_files = [
            file_path
            for file_path in candidates
            if not any(pattern in file_path for pattern in exclusion_patterns)
        ]

        logger.info(f"Discovered {len(filtered_files)} source files in {container_name}")

        return {
            "source_files": filtered_files,
            "file_count": len(filtered_files),
            "directories_scanned": [base_directory],
            "excluded_patterns": exclusion_patterns,
        }

    except Exception as e:
        logger.error(f"Error discovering source files: {e}")
        return {
            "source_files": [],
            "file_count": 0,
            "directories_scanned": [],
            "excluded_patterns": [],
            "error": str(e),
        }


async def classify_files(container_name: str, file_paths: list[str]) -> dict[str, Any]:
    """Classify discovered files by type and purpose.

    This function analyzes a list of file paths and classifies them into
    categories based on their location, naming patterns, and content hints.
    This helps understand the structure and purpose of different files.

    Classification Categories:
    - source: Main application source code
    - test: Unit tests, integration tests, test utilities
    - config: Configuration files, setup scripts
    - script: Utility scripts, command-line tools
    - documentation: README files, docs, examples
    - other: Files that don't fit other categories

    Args:
        container_name (str): Name of the container containing the files.
            Used for logging and error reporting.
        file_paths (List[str]): List of file paths to classify.
            Paths should be absolute or relative to container working directory.

    Returns:
        Dict[str, Any]: Classification results containing:
            - classified_files (Dict[str, List[str]]): Files grouped by category
            - classification_summary (Dict[str, int]): Count of files per category
            - total_files (int): Total number of files classified
            - error (str, optional): Error message if classification failed

    Classification Rules:
        - test: Files in test/ or tests/ directories, files starting with test_
        - config: setup.py, __init__.py, config.py, settings.py files
        - script: Files in bin/, scripts/, tools/ directories
        - source: Files in src/, app/, lib/ directories
        - documentation: Files named README, docs/, examples/

    Example:
        ```python
        files = ["src/main.py", "tests/test_main.py", "setup.py"]
        result = await classify_files("container-name", files)

        for category, file_list in result['classified_files'].items():
            print(f"{category}: {len(file_list)} files")
            for file_path in file_list:
                print(f"  {file_path}")
        ```

    Error Handling:
        - Invalid file paths are logged but don't stop classification
        - Empty file lists are handled gracefully
        - Classification continues even if some files can't be categorized
    """
    try:
        classified: dict[str, list[str]] = {
            "source": [],
            "test": [],
            "config": [],
            "script": [],
            "documentation": [],
            "other": [],
        }

        for file_path in file_paths:
            category = _classify_single_file(file_path)
            classified[category].append(file_path)

        # Generate summary statistics
        summary = {category: len(files) for category, files in classified.items()}

        logger.info(
            f"Classified {len(file_paths)} files in {container_name}: "
            f"source={summary['source']}, test={summary['test']}, "
            f"config={summary['config']}, other={summary['other']}"
        )

        return {
            "classified_files": classified,
            "classification_summary": summary,
            "total_files": len(file_paths),
        }

    except Exception as e:
        logger.error(f"Error classifying files: {e}")
        return {
            "classified_files": {},
            "classification_summary": {},
            "total_files": 0,
            "error": str(e),
        }


def _classify_single_file(file_path: str) -> str:
    """Classify a single file based on its path and name.

    This helper function applies classification rules to determine
    the category of a single file based on its path components and filename.

    Args:
        file_path (str): Path to the file to classify.

    Returns:
        str: Classification category ('source', 'test', 'config', 'script',
             'documentation', or 'other').
    """
    path_lower = file_path.lower()
    filename = path_lower.split("/")[-1]

    # Test files
    if (
        "/test/" in path_lower
        or "/tests/" in path_lower
        or filename.startswith("test_")
        or filename.startswith("conftest")
        or "_test.py" in filename
    ):
        return "test"

    # Configuration files
    if filename in [
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "config.py",
        "settings.py",
        "configuration.py",
        "__init__.py",
    ]:
        return "config"

    # Script files
    if (
        "/bin/" in path_lower
        or "/scripts/" in path_lower
        or "/tools/" in path_lower
        or "/utils/" in path_lower
        or "script" in filename
    ):
        return "script"

    # Documentation files
    if (
        "/docs/" in path_lower
        or "/doc/" in path_lower
        or "/examples/" in path_lower
        or "readme" in filename
        or "example" in filename
    ):
        return "documentation"

    # Source files (main application code)
    if (
        "/src/" in path_lower
        or "/app/" in path_lower
        or "/lib/" in path_lower
        or "/core/" in path_lower
        or "/models/" in path_lower
        or "/services/" in path_lower
        or "/api/" in path_lower
    ):
        return "source"

    # Default to other
    return "other"


async def get_file_metadata(
    container_name: str, file_paths: list[str], tenant_id: str | None = None
) -> dict[str, Any]:
    """Get metadata information for files within a container.

    This function collects detailed metadata about files including size,
    line counts, and basic content analysis. This information is useful
    for understanding code complexity and file characteristics.

    Args:
        container_name (str): Name of the container containing the files.
        file_paths (List[str]): List of file paths to analyze.

    Returns:
        Dict[str, Any]: Metadata results containing:
            - file_metadata (List[Dict]): Detailed metadata per file
            - total_lines (int): Total lines across all files
            - average_file_size (float): Average lines per file
            - error (str, optional): Error message if metadata collection failed

        Each file metadata includes:
            - file_path (str): Path to the file
            - line_count (int): Number of lines in the file
            - size_bytes (int): File size in bytes
            - is_readable (bool): Whether file could be read
            - content_preview (str): First few lines of the file

    Example:
        ```python
        files = ["src/main.py", "src/utils.py"]
        result = await get_file_metadata("container-name", files)

        print(f"Total lines: {result['total_lines']}")
        for file_meta in result['file_metadata']:
            print(f"{file_meta['file_path']}: {file_meta['line_count']} lines")
        ```
    """
    try:
        file_metadata = []
        total_lines = 0

        for file_path in file_paths:
            try:
                # Get file line count and basic info
                wc_result = await execute_in_container(
                    container_name, f"wc -l '{file_path}'", tenant_id=tenant_id
                )

                line_count = 0
                if wc_result["success"]:
                    output = wc_result["output"].strip()
                    try:
                        line_count = int(output.split()[0])
                    except (ValueError, IndexError):  # fmt: skip
                        line_count = 0

                # Get file size
                size_result = await execute_in_container(
                    container_name, f"stat -c %s '{file_path}'", tenant_id=tenant_id
                )

                size_bytes = 0
                if size_result["success"]:
                    try:
                        size_bytes = int(size_result["output"].strip())
                    except ValueError:
                        size_bytes = 0

                # Get content preview (first 3 lines)
                preview_result = await execute_in_container(
                    container_name, f"head -n 3 '{file_path}'", tenant_id=tenant_id
                )

                content_preview = ""
                if preview_result["success"]:
                    content_preview = preview_result["output"].strip()

                metadata = {
                    "file_path": file_path,
                    "line_count": line_count,
                    "size_bytes": size_bytes,
                    "is_readable": wc_result["success"],
                    "content_preview": content_preview,
                }

                file_metadata.append(metadata)
                total_lines += line_count

            except Exception as e:
                logger.warning(f"Error getting metadata for {file_path}: {e}")
                file_metadata.append(
                    {
                        "file_path": file_path,
                        "line_count": 0,
                        "size_bytes": 0,
                        "is_readable": False,
                        "content_preview": "",
                        "error": str(e),
                    }
                )

        average_file_size = total_lines / len(file_paths) if file_paths else 0

        return {
            "file_metadata": file_metadata,
            "total_lines": total_lines,
            "average_file_size": average_file_size,
        }

    except Exception as e:
        logger.error(f"Error collecting file metadata: {e}")
        return {
            "file_metadata": [],
            "total_lines": 0,
            "average_file_size": 0,
            "error": str(e),
        }
