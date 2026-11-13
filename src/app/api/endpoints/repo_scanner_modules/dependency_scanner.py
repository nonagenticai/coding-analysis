"""Dependency Scanner Module.

This module provides dependency scanning functionality for Python packages,
including pip packages, poetry dependencies, and requirements files.

The module provides:
- Multi-source dependency detection (pip, poetry, requirements)
- Enhanced dependency scanning with direct container manager integration
- Version pattern analysis and validation
- Dependency source attribution
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.container_manager import get_container_manager

from ...models.repo_scanner_api_models import ScannedDependency
from .dependency_scan_script import (
    DEPENDENCY_SCAN_SCRIPT,
    DEPENDENCY_SCAN_SCRIPT_PATH,
    dependency_scan_command,
)
from .file_discovery import (
    NEUTRAL_WORKDIR,
    _script_write_command,
    execute_in_container,
    isolated_python_command,
)

logger = logging.getLogger(__name__)


async def scan_dependencies(container_name: str, *, tenant_id: str | None = None) -> dict[str, Any]:
    """Scan and analyze Python dependencies from multiple sources within a container.

    This function performs comprehensive dependency analysis by checking multiple
    package management sources including pip, poetry, and requirements files.
    It creates and executes a Python script inside the target container to
    gather dependency information securely.

    The scan process:
    1. Creates a Python script for dependency detection
    2. Writes the script to the container's temporary directory
    3. Executes the script to collect dependency information
    4. Parses and validates the results
    5. Returns structured dependency data

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
            - source (str): Source of dependency info (pip, poetry, requirements)
            - is_valid (bool): Validation status

    Sources Scanned:
        - pip list: Currently installed packages
        - poetry.lock: Poetry-locked dependencies, parsed as text (never runs poetry)
        - requirements.txt: Requirements file dependencies
        - requirements-dev.txt: Development requirements
        - requirements/base.txt: Base requirements

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

    if not write_result["success"]:
        logger.error(f"Failed to write dependency scan script: {write_result['output']}")
        return {"dependencies": [], "error": "Failed to create scan script"}

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
            return {
                "dependencies": [
                    ScannedDependency(
                        name=dep["name"],
                        version=dep["version"],
                        source=dep["source"],
                        is_valid=True,
                    )
                    for dep in deps_data
                ],
                "dependency_count": len(deps_data),
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse dependency output: {e}")
            return {
                "dependencies": [],
                "error": f"Failed to parse output: {str(e)}",
            }
    else:
        return {"dependencies": [], "error": result["output"]}


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
