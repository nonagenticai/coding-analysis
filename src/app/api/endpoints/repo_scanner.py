"""Repository Scanner API Endpoint Module.

This module provides comprehensive code analysis and dependency scanning capabilities
for dockerized repositories. It enables advanced repository analysis including:

- Dependency scanning and analysis (pip, poetry, requirements.txt)
- Python import statement analysis and validation
- Code syntax and quality validation
- Comprehensive repository health assessment

The module operates by executing analysis scripts inside Docker containers,
providing isolated and secure analysis environments. All scanning operations
are performed through containerized execution to ensure reproducibility and
security isolation.

Key Features:
    - Multi-source dependency detection (pip, poetry, requirements files)
    - AST-based import analysis with validation
    - Syntax validation and basic code quality checks
    - Containerized execution for security and isolation
    - Comprehensive error handling and logging
    - Support for targeted file analysis or full repository scans

Typical Usage:
    ```python
    # Scan dependencies in a container
    request = RepoScannerScanRequest(
        container_name="my-repo-container",
        action="dependencies"
    )
    result = await scan_repository(request)

    # Validate imports with specific files
    request = RepoScannerScanRequest(
        container_name="my-repo-container",
        action="imports",
        target_files=["src/main.py", "src/utils.py"]
    )
    result = await scan_repository(request)
    ```

Author: Code Graph Helper Team
Version: 2.1 (Modularized)

Note: This module has been refactored into focused sub-modules for better
maintainability and separation of concerns. The main functionality is now
distributed across:

- repo_scanner_modules.scanner_endpoints: FastAPI endpoint definitions
- repo_scanner_modules.scanner_core: Core scanning logic
- repo_scanner_modules.file_discovery: File discovery and container operations
- repo_scanner_modules.results_processor: Results processing and formatting
- repo_scanner_modules.scanner_config: Configuration management

The public API remains unchanged, but the internal implementation is now
modular and easier to maintain.
"""

# Import the main router from the modularized endpoints
from .repo_scanner_modules import router

# Re-export the router for backward compatibility
__all__ = ["router"]

# For debugging and introspection, also export key functions

# Module metadata
__version__ = "2.1.0"
__author__ = "Code Graph Helper Team"
__description__ = "Modularized Repository Scanner for Dockerized Code Analysis"

# Backward compatibility exports (if needed by other modules)
__all__.extend(
    [
        "scan_dependencies",
        "scan_imports",
        "validate_code",
        "scan_dependencies_v2",
        "validate_code_v2",
        "validate_scanned_imports",
        "execute_in_container",
        "process_scan_results",
        "get_container_manager_v2",
    ]
)
