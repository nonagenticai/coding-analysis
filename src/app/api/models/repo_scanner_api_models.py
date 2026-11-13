"""Repository Scanner API Models for V2 API.

This module contains Pydantic models for repository scanning operations in the V2 API.
These models support comprehensive repository analysis including dependency scanning,
import validation, code quality checks, and version management.

The models provide:
    - Dependency discovery and validation across multiple package managers
    - Import statement analysis and validation
    - Code quality assessment and validation results
    - Version checking and compatibility analysis
    - Configurable scanning options with include/exclude patterns
    - Comprehensive scan results with detailed metrics

Supported scan types:
    - Dependencies: Discover and validate project dependencies
    - Imports: Analyze import statements and validate availability
    - Validation: Perform code quality and syntax validation
    - Versions: Check package versions and compatibility
    - Full scan: Comprehensive analysis combining all scan types

Example:
    ```python
    from repo_scanner_api_models import RepoScannerScanRequest, ScannedDependency

    # Create a dependency scan request
    scan_request = RepoScannerScanRequest(
        container_name="my-python-project",
        action="dependencies",
        include_patterns=["*.py"],
        exclude_patterns=["__pycache__", "*.pyc"],
        deep_scan=True
    )

    # Dependency information
    dependency = ScannedDependency(
        name="requests",
        version="2.28.1",
        source="pip",
        is_valid=True
    )
    ```

Note:
    All scanning operations are performed on containerized repositories to ensure
    isolated and reproducible analysis environments.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from models.repo_scanner_models import RepoScanAction

from ...types import SAFE_STRING_PATTERN, VERSION_PATTERN
from ..models.common_api_models import (
    BaseResponse,
)


# Pydantic Models for Repository Scanner API
class APIScanAction(str):
    """Enumeration of available repository scanning actions for API models.

    This class defines the types of analysis that can be performed on a repository.
    Each action focuses on a specific aspect of repository health and quality.

    Attributes:
        DEPENDENCIES: Scan for project dependencies and their validity
        IMPORTS: Analyze import statements and their availability
        VALIDATION: Perform code quality and syntax validation
        VERSIONS: Check package versions and compatibility
        FULL_SCAN: Comprehensive scan combining all analysis types

    Example:
        ```python
        # Use specific scan action
        scan_request = RepoScannerScanRequest(
            container_name="my-project",
            action=ScanAction.DEPENDENCIES
        )

        # Use string value directly
        scan_request = RepoScannerScanRequest(
            container_name="my-project",
            action="dependencies"
        )
        ```

    Note:
        FULL_SCAN provides the most comprehensive analysis but takes longer to complete.
        Use specific actions when you need focused analysis results.
    """

    DEPENDENCIES = "dependencies"
    IMPORTS = "imports"
    VALIDATION = "validation"
    VERSIONS = "versions"
    FULL_SCAN = "full_scan"


class ScannedDependency(BaseModel):
    """Information about a project dependency and its installation status.

    This model represents a single dependency discovered in a repository,
    including its source, version, and validation status. It provides
    comprehensive information for dependency management and troubleshooting.

    Attributes:
        name (str): Package name as specified in dependency files
        version (str): Currently installed version of the package
        source (str): Package manager or source ("pip", "poetry", "conda", etc.)
        is_valid (bool): Whether the dependency is properly installed and available

    Example:
        ```python
        dependency = ScannedDependency(
            name="requests",
            version="2.28.1",
            source="pip",
            is_valid=True
        )

        # Invalid dependency example
        broken_dependency = ScannedDependency(
            name="missing-package",
            version="unknown",
            source="pip",
            is_valid=False
        )
        ```

    Note:
        The is_valid field indicates whether the dependency can be imported
        and used in the current environment. Invalid dependencies may need
        reinstallation or version updates.
    """

    name: str = Field(..., description="Package name", pattern=SAFE_STRING_PATTERN)
    version: str = Field(..., description="Installed version", pattern=VERSION_PATTERN)
    source: str = Field(..., description="Source (pip, poetry, etc.)", pattern=SAFE_STRING_PATTERN)
    is_valid: bool = Field(True, description="Whether dependency is properly installed")


class RepoScannerImportInfo(BaseModel):
    """Information about import statements found in code.

    This model represents a single import statement discovered during repository
    analysis, including its location, type, and validity status. It provides
    detailed information for import resolution and dependency tracking.

    Attributes:
        module (str): The module being imported (e.g., "requests", "os.path")
        name (Optional[str]): Specific name being imported (e.g., "get" from "requests.get")
        alias (Optional[str]): Import alias if specified (e.g., "np" in "import numpy as np")
        import_type (str): Type of import statement ("import", "from", "from_relative")
        file_path (str): Path to the file containing the import statement
        line_number (int): Line number where the import appears
        is_valid (bool): Whether the import can be resolved and used
        error_message (Optional[str]): Error details if the import is invalid

    Example:
        ```python
        # Standard import
        import_info = RepoScannerImportInfo(
            module="requests",
            name=None,
            alias=None,
            import_type="import",
            file_path="src/api_client.py",
            line_number=3,
            is_valid=True
        )

        # From import with alias
        from_import = RepoScannerImportInfo(
            module="numpy",
            name="array",
            alias="np_array",
            import_type="from",
            file_path="src/data_processor.py",
            line_number=5,
            is_valid=True
        )

        # Invalid import
        invalid_import = RepoScannerImportInfo(
            module="missing_module",
            name=None,
            alias=None,
            import_type="import",
            file_path="src/broken.py",
            line_number=1,
            is_valid=False,
            error_message="ModuleNotFoundError: No module named 'missing_module'"
        )
        ```

    Note:
        The is_valid field indicates whether the import can be resolved in the
        current environment. Invalid imports may indicate missing dependencies.
    """

    module: str = Field(..., description="Module being imported", pattern=SAFE_STRING_PATTERN)
    name: str | None = Field(None, description="Specific import name", pattern=SAFE_STRING_PATTERN)
    alias: str | None = Field(None, description="Import alias if any", pattern=SAFE_STRING_PATTERN)
    import_type: str = Field(
        ..., description="Type of import (from, import)", pattern=SAFE_STRING_PATTERN
    )
    file_path: str = Field(
        ..., description="File containing the import", pattern=SAFE_STRING_PATTERN
    )
    line_number: int = Field(..., description="Line number of import")
    is_valid: bool = Field(True, description="Whether import is valid")
    error_message: str | None = Field(
        None, description="Error if invalid", pattern=SAFE_STRING_PATTERN
    )


class ApiValidationResult(BaseModel):
    """Code validation result for individual files.

    This model represents the validation outcome for a single file in the repository,
    including syntax checking, code quality assessment, and issue reporting.

    Attributes:
        file_path (str): Path to the validated file relative to repository root
        is_valid (bool): Whether the file passes all validation checks
        issues (List[str]): List of validation issues that must be addressed
        warnings (List[str]): List of validation warnings that should be reviewed
        line_count (int): Total number of lines in the file

    Example:
        ```python
        # Valid file
        validation_result = ApiValidationResult(
            file_path="src/utils.py",
            is_valid=True,
            issues=[],
            warnings=["Line 42: Consider using f-string for better readability"],
            line_count=150
        )

        # Invalid file
        invalid_result = ApiValidationResult(
            file_path="src/broken.py",
            is_valid=False,
            issues=[
                "Line 15: SyntaxError: invalid syntax",
                "Line 23: NameError: undefined variable 'x'"
            ],
            warnings=["Line 5: Unused import 'os'"],
            line_count=45
        )
        ```

    Note:
        Issues are critical problems that prevent the code from running correctly.
        Warnings are suggestions for code improvement that don't affect functionality.
    """

    file_path: str = Field(..., description="Path to validated file", pattern=SAFE_STRING_PATTERN)
    is_valid: bool = Field(..., description="Whether file passes validation")
    issues: list[str] = Field(default_factory=list, description="Validation issues")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")
    line_count: int = Field(0, description="Number of lines in file")


class VersionInfo(BaseModel):
    """Package version information and compatibility data.

    This model provides comprehensive version information for packages,
    including current and available versions, and symbol availability
    across different versions.

    Attributes:
        package (str): Name of the package being analyzed
        symbol (Optional[str]): Specific symbol/function to check availability for
        current_version (Optional[str]): Currently installed version in the environment
        latest_version (Optional[str]): Latest available version from package repository
        available_versions (List[str]): List of all available versions for the package
        symbol_available_in (List[str]): Versions where the specified symbol is available

    Example:
        ```python
        # Package version info
        version_info = VersionInfo(
            package="requests",
            symbol="Session",
            current_version="2.28.1",
            latest_version="2.31.0",
            available_versions=["2.25.1", "2.26.0", "2.27.0", "2.28.1", "2.31.0"],
            symbol_available_in=["2.25.1", "2.26.0", "2.27.0", "2.28.1", "2.31.0"]
        )

        # Package without specific symbol check
        basic_version = VersionInfo(
            package="numpy",
            symbol=None,
            current_version="1.21.0",
            latest_version="1.24.3",
            available_versions=["1.20.0", "1.21.0", "1.22.0", "1.23.0", "1.24.3"],
            symbol_available_in=[]
        )
        ```

    Note:
        Symbol availability checking helps ensure compatibility when upgrading packages.
        Empty symbol_available_in list indicates no symbol-specific analysis was performed.
    """

    package: str = Field(..., description="Package name", pattern=SAFE_STRING_PATTERN)
    symbol: str | None = Field(None, description="Symbol to check", pattern=SAFE_STRING_PATTERN)
    current_version: str | None = Field(
        None, description="Currently installed version", pattern=VERSION_PATTERN
    )
    latest_version: str | None = Field(
        None, description="Latest available version", pattern=VERSION_PATTERN
    )
    available_versions: list[str] = Field(
        default_factory=list, description="All available versions"
    )
    symbol_available_in: list[str] = Field(
        default_factory=list, description="Versions containing symbol"
    )


class RepoScannerScanRequest(BaseModel):
    """Request model for repository scanning operations.

    This model defines the parameters for repository scanning requests,
    including the target container, scan type, and filtering options.

    Attributes:
        container_name (str): Name of the container containing the repository to scan
        action (str): Type of scan to perform (dependencies, imports, validation, etc.)
        target_files (Optional[List[str]]): Specific files to scan (overrides patterns)
        include_patterns (Optional[List[str]]): File glob patterns to include in scan
        exclude_patterns (Optional[List[str]]): File glob patterns to exclude from scan
        deep_scan (bool): Whether to perform comprehensive deep analysis

    Example:
        ```python
        # Basic dependency scan
        scan_request = RepoScannerScanRequest(
            container_name="my-python-project",
            action="dependencies",
            include_patterns=["*.py"],
            exclude_patterns=["__pycache__", "*.pyc", ".git"],
            deep_scan=False
        )

        # Deep scan with specific files
        deep_scan_request = RepoScannerScanRequest(
            container_name="web-app",
            action="validation",
            target_files=["src/main.py", "src/utils.py"],
            deep_scan=True
        )

        # Full comprehensive scan
        full_scan = RepoScannerScanRequest(
            container_name="backend-service",
            action="full_scan",
            include_patterns=["*.py", "*.js", "*.ts"],
            exclude_patterns=["node_modules", "__pycache__", "*.pyc"],
            deep_scan=True
        )
        ```

    Note:
        When target_files is specified, include_patterns and exclude_patterns are ignored.
        Deep scan mode provides more thorough analysis but takes longer to complete.
    """

    container_name: str = Field(
        ..., description="Container name to scan", pattern=SAFE_STRING_PATTERN
    )
    action: RepoScanAction = Field(..., description="Scan action to perform")
    target_files: list[str] | None = Field(None, description="Specific files to scan (optional)")
    include_patterns: list[str] | None = Field(["*.py"], description="File patterns to include")
    exclude_patterns: list[str] | None = Field(
        ["__pycache__", "*.pyc", ".git"], description="Patterns to exclude"
    )
    deep_scan: bool = Field(False, description="Whether to perform deep analysis")


class RepoScannerScanResponse(BaseResponse):
    """Base response model for repository scanning operations.

    This model provides the foundation for all scanning operation responses,
    including timing information, file counts, and basic result data.

    Attributes:
        container_name (str): Name of the container that was scanned
        action (str): Type of scan that was performed
        scan_duration (float): Time taken to complete the scan in seconds
        file_count (int): Number of files that were analyzed during the scan
        results (Dict[str, Any]): Basic scan results in dictionary format

    Example:
        ```python
        response = RepoScannerScanResponse(
            status=ResponseStatus.SUCCESS,
            message="Scan completed successfully",
            container_name="my-project",
            action="dependencies",
            scan_duration=12.5,
            file_count=45,
            results={"summary": "Found 23 dependencies, 2 invalid"}
        )
        ```

    Note:
        This is a base class that is extended by specific scan response types.
        The results field contains basic information; specialized responses
        provide detailed results in specific fields.
    """

    container_name: str = Field(
        ..., description="Scanned container name", pattern=SAFE_STRING_PATTERN
    )
    action: str = Field(..., description="Performed scan action", pattern=SAFE_STRING_PATTERN)
    scan_duration: float = Field(..., description="Scan duration in seconds")
    file_count: int = Field(0, description="Number of files scanned")
    results: dict[str, Any] = Field(default_factory=dict, description="Scan results")


class RepoScannerDependencyScanResponse(RepoScannerScanResponse):
    """Response model for dependency scanning operations.

    This model extends ScanResponse to provide detailed dependency information,
    including valid and invalid dependencies discovered in the repository.

    Attributes:
        dependencies (List[ScannedDependency]): List of all discovered dependencies
        dependency_count (int): Total number of dependencies found
        invalid_dependencies (List[str]): Names of dependencies that failed validation

    Example:
        ```python
        response = RepoScannerDependencyScanResponse(
            status=ResponseStatus.SUCCESS,
            message="Dependency scan completed",
            container_name="my-project",
            action="dependencies",
            scan_duration=8.2,
            file_count=15,
            dependencies=[
                ScannedDependency(name="requests", version="2.28.1", source="pip", is_valid=True),
                ScannedDependency(name="numpy", version="1.21.0", source="pip", is_valid=True),
                ScannedDependency(name="missing-pkg", version="unknown", source="pip", is_valid=False)
            ],
            dependency_count=3,
            invalid_dependencies=["missing-pkg"]
        )
        ```

    Note:
        Invalid dependencies indicate packages that are listed in dependency files
        but are not properly installed or available in the current environment.
    """

    dependencies: list[ScannedDependency] = Field(
        default_factory=list, description="Found dependencies"
    )
    dependency_count: int = Field(0, description="Total number of dependencies")
    invalid_dependencies: list[str] = Field(
        default_factory=list, description="Invalid dependencies"
    )


class RepoScannerImportScanResponse(RepoScannerScanResponse):
    """Response model for import scanning operations.

    This model extends ScanResponse to provide detailed import information,
    including all discovered imports and those that failed validation.

    Attributes:
        imports (List[RepoScannerImportInfo]): List of all discovered import statements
        import_count (int): Total number of import statements found
        invalid_imports (List[RepoScannerImportInfo]): List of imports that failed validation

    Example:
        ```python
        response = RepoScannerImportScanResponse(
            status=ResponseStatus.SUCCESS,
            message="Import scan completed",
            container_name="my-project",
            action="imports",
            scan_duration=5.7,
            file_count=20,
            imports=[
                RepoScannerImportInfo(module="requests", file_path="src/api.py", line_number=1, is_valid=True),
                RepoScannerImportInfo(module="numpy", file_path="src/data.py", line_number=3, is_valid=True),
                RepoScannerImportInfo(module="missing_module", file_path="src/broken.py", line_number=5, is_valid=False)
            ],
            import_count=3,
            invalid_imports=[
                RepoScannerImportInfo(module="missing_module", file_path="src/broken.py", line_number=5, is_valid=False)
            ]
        )
        ```

    Note:
        Invalid imports indicate import statements that reference modules or
        packages that are not available in the current environment.
    """

    imports: list[RepoScannerImportInfo] = Field(default_factory=list, description="Found imports")
    import_count: int = Field(0, description="Total number of imports")
    invalid_imports: list[RepoScannerImportInfo] = Field(
        default_factory=list, description="Invalid imports"
    )


class RepoScannerValidationScanResponse(RepoScannerScanResponse):
    """Response model for code validation scanning operations.

    This model extends ScanResponse to provide detailed validation results
    for each file, including syntax checking and code quality assessment.

    Attributes:
        validation_results (List[ApiValidationResult]): Detailed validation results for each file
        valid_files (int): Number of files that passed all validation checks
        invalid_files (int): Number of files that failed validation

    Example:
        ```python
        response = RepoScannerValidationScanResponse(
            status=ResponseStatus.SUCCESS,
            message="Validation scan completed",
            container_name="my-project",
            action="validation",
            scan_duration=15.3,
            file_count=25,
            validation_results=[
                ApiValidationResult(
                    file_path="src/main.py",
                    is_valid=True,
                    issues=[],
                    warnings=["Line 10: Consider using f-string"],
                    line_count=50
                ),
                ApiValidationResult(
                    file_path="src/broken.py",
                    is_valid=False,
                    issues=["Line 5: SyntaxError: invalid syntax"],
                    warnings=[],
                    line_count=20
                )
            ],
            valid_files=24,
            invalid_files=1
        )
        ```

    Note:
        Valid files have no critical issues but may have warnings for improvement.
        Invalid files have syntax errors or other critical issues that prevent execution.
    """

    validation_results: list[ApiValidationResult] = Field(
        default_factory=list, description="Validation results"
    )
    valid_files: int = Field(0, description="Number of valid files")
    invalid_files: int = Field(0, description="Number of invalid files")


class VersionScanResponse(RepoScannerScanResponse):
    """Response model for package version scanning operations.

    This model extends ScanResponse to provide detailed version information
    for packages, including current versions, latest versions, and outdated packages.

    Attributes:
        version_info (List[VersionInfo]): Detailed version information for each package
        outdated_packages (List[str]): Names of packages with newer versions available

    Example:
        ```python
        response = VersionScanResponse(
            status=ResponseStatus.SUCCESS,
            message="Version scan completed",
            container_name="my-project",
            action="versions",
            scan_duration=22.8,
            file_count=18,
            version_info=[
                VersionInfo(
                    package="requests",
                    current_version="2.28.1",
                    latest_version="2.31.0",
                    available_versions=["2.28.1", "2.29.0", "2.30.0", "2.31.0"]
                ),
                VersionInfo(
                    package="numpy",
                    current_version="1.21.0",
                    latest_version="1.24.3",
                    available_versions=["1.21.0", "1.22.0", "1.23.0", "1.24.3"]
                )
            ],
            outdated_packages=["requests", "numpy"]
        )
        ```

    Note:
        Outdated packages indicate dependencies that have newer versions available.
        Consider upgrading these packages while checking for breaking changes.
    """

    version_info: list[VersionInfo] = Field(default_factory=list, description="Version information")
    outdated_packages: list[str] = Field(
        default_factory=list, description="Packages with newer versions"
    )
