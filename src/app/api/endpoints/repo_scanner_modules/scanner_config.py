"""Scanner Configuration Management Module.

This module provides configuration management for repository scanning operations,
including container manager setup, scanning parameters, and environment settings.

The module provides:
- Container manager instance management
- Scanning configuration validation
- Environment-specific settings
- Performance tuning parameters
- Security configuration options
"""

from __future__ import annotations

import logging
from datetime import UTC
from typing import Any, TypedDict
from uuid import uuid7

from app.core.container_manager import ContainerManager, get_container_manager

logger = logging.getLogger(__name__)


class ValidationResultDict(TypedDict):
    """TypedDict for container validation results."""

    is_valid: bool
    container_info: dict[str, Any]
    checks_passed: list[str]
    checks_failed: list[str]
    recommendations: list[str]


class ScannerConfig:
    """Configuration class for repository scanner operations.

    This class encapsulates all configuration settings for scanning operations,
    providing a centralized way to manage scanner behavior, performance tuning,
    and environment-specific settings.

    Attributes:
        container_timeout (int): Default timeout for container operations in seconds
        max_files_per_scan (int): Maximum number of files to process in a single scan
        enable_caching (bool): Whether to enable result caching
        parallel_processing (bool): Whether to enable parallel file processing
        exclusion_patterns (List[str]): Default file/directory exclusion patterns
        dependency_sources (List[str]): Package management sources to scan
        validation_rules (Dict[str, Any]): Code validation rules and thresholds
    """

    def __init__(self):
        """Initialize scanner configuration with default values."""
        # Performance settings
        self.container_timeout = 300  # 5 minutes
        self.max_files_per_scan = 1000
        self.enable_caching = True
        self.parallel_processing = False  # Disabled by default for stability

        # File discovery settings
        self.exclusion_patterns = [
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
            "*.pyc",
            "*.pyo",
            "*.pyd",
        ]

        # Dependency scanning settings
        self.dependency_sources = [
            "pip",
            "poetry",
            "requirements.txt",
            "requirements-dev.txt",
            "requirements/base.txt",
            "requirements/development.txt",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
        ]

        # Validation settings
        self.validation_rules = {
            "max_line_length": 120,
            "check_syntax": True,
            "check_imports": True,
            "enable_style_warnings": True,
            "fail_fast": False,  # Continue processing even if errors found
        }

        # Security settings
        self.security_settings = {
            "restrict_command_execution": True,
            "allowed_file_extensions": [
                ".py",
                ".txt",
                ".toml",
                ".cfg",
                ".yml",
                ".yaml",
            ],
            "max_file_size_bytes": 10 * 1024 * 1024,  # 10MB
            "sandbox_execution": True,
        }

        # Logging settings
        self.logging_config = {
            "log_level": "INFO",
            "enable_performance_logging": True,
            "log_scan_details": True,
        }

    def update_from_dict(self, config_dict: dict[str, Any]) -> None:
        """Update configuration from a dictionary.

        Args:
            config_dict: Dictionary containing configuration updates.
                Keys should match attribute names.

        Example:
            ```python
            config = ScannerConfig()
            config.update_from_dict({
                "container_timeout": 600,
                "max_files_per_scan": 500,
                "validation_rules": {"max_line_length": 100}
            })
            ```
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), dict) and isinstance(value, dict):
                    # Merge dictionaries
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)
                logger.debug(f"Updated scanner config: {key} = {value}")
            else:
                logger.warning(f"Unknown configuration key: {key}")

    def validate_config(self) -> list[str]:
        """Validate configuration settings and return any issues.

        Returns:
            List[str]: List of validation error messages. Empty if valid.

        Example:
            ```python
            config = ScannerConfig()
            issues = config.validate_config()
            if issues:
                for issue in issues:
                    print(f"Config issue: {issue}")
            ```
        """
        issues = []

        # Validate timeout settings
        if self.container_timeout <= 0:
            issues.append("container_timeout must be positive")

        # Validate file limits
        if self.max_files_per_scan <= 0:
            issues.append("max_files_per_scan must be positive")
        elif self.max_files_per_scan > 10000:
            issues.append("max_files_per_scan exceeds recommended maximum (10000)")

        # Validate validation rules
        if self.validation_rules.get("max_line_length", 0) <= 0:
            issues.append("max_line_length must be positive")

        # Validate security settings
        max_size: int = self.security_settings.get("max_file_size_bytes", 0)  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        if max_size <= 0:
            issues.append("max_file_size_bytes must be positive")
        elif max_size > 100 * 1024 * 1024:  # 100MB
            issues.append("max_file_size_bytes exceeds recommended maximum (100MB)")

        return issues

    def get_container_execution_config(self) -> dict[str, Any]:
        """Get configuration for container command execution.

        Returns:
            Dict[str, Any]: Configuration for container operations.
        """
        return {
            "timeout": self.container_timeout,
            "workdir": "/workspace",
            "privileged": False,
            "tty": False,
            "stream": False,
            "detach": False,
            "environment": {},
        }

    def get_file_discovery_config(self) -> dict[str, Any]:
        """Get configuration for file discovery operations.

        Returns:
            Dict[str, Any]: Configuration for file discovery.
        """
        return {
            "exclusion_patterns": self.exclusion_patterns,
            "max_files": self.max_files_per_scan,
            "allowed_extensions": self.security_settings["allowed_file_extensions"],
            "max_file_size": self.security_settings["max_file_size_bytes"],
        }

    def get_validation_config(self) -> dict[str, Any]:
        """Get configuration for code validation operations.

        Returns:
            Dict[str, Any]: Configuration for validation.
        """
        return self.validation_rules.copy()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary format.

        Returns:
            Dict[str, Any]: Complete configuration as dictionary.
        """
        return {
            "container_timeout": self.container_timeout,
            "max_files_per_scan": self.max_files_per_scan,
            "enable_caching": self.enable_caching,
            "parallel_processing": self.parallel_processing,
            "exclusion_patterns": self.exclusion_patterns,
            "dependency_sources": self.dependency_sources,
            "validation_rules": self.validation_rules,
            "security_settings": self.security_settings,
            "logging_config": self.logging_config,
        }


# Global configuration instance
_scanner_config = ScannerConfig()


def get_scanner_config() -> ScannerConfig:
    """Get the global scanner configuration instance.

    Returns:
        ScannerConfig: Global configuration instance.

    Example:
        ```python
        config = get_scanner_config()
        timeout = config.container_timeout
        patterns = config.exclusion_patterns
        ```
    """
    return _scanner_config


def update_scanner_config(config_updates: dict[str, Any]) -> None:
    """Update the global scanner configuration.

    Args:
        config_updates: Dictionary of configuration updates.

    Example:
        ```python
        update_scanner_config({
            "container_timeout": 600,
            "max_files_per_scan": 500
        })
        ```
    """
    global _scanner_config
    _scanner_config.update_from_dict(config_updates)

    # Validate after update
    issues = _scanner_config.validate_config()
    if issues:
        logger.warning(f"Configuration validation issues after update: {issues}")


def reset_scanner_config() -> None:
    """Reset scanner configuration to default values.

    Example:
        ```python
        reset_scanner_config()
        config = get_scanner_config()
        # config now has default values
        ```
    """
    global _scanner_config
    _scanner_config = ScannerConfig()


def get_container_manager_v2() -> ContainerManager:
    """Get the container manager instance for direct container operations.

    This function provides access to the container management system used
    for advanced scanning operations. It creates a new ContainerManager
    instance configured for the current system environment.

    Returns:
        ContainerManager: Configured container manager instance capable of:
            - Container discovery and status checking
            - Command execution within containers
            - File operations and content retrieval
            - Resource monitoring and management

    Usage:
        This function is used internally by the v2 scanning functions to
        access containers directly rather than through the global container
        manager instance. This provides better isolation and error handling
        for scanning operations.

    Note:
        The import path should be adjusted based on the actual codebase
        structure. This function serves as a factory for container manager
        instances used in scanning operations.
    """
    # This should import the actual container manager used in your system
    # Adjust import path based on your codebase structure
    return ContainerManager()


async def validate_container_access(
    container_name: str, tenant_id: str | None = None
) -> ValidationResultDict:
    """Validate that a container is accessible and ready for scanning.

    This function performs comprehensive validation of container status,
    accessibility, and readiness for scanning operations. It checks both
    the container manager's view and the actual Docker container state.

    Args:
        container_name (str): Name of the container to validate.

    Returns:
        ValidationResultDict: Validation results containing:
            - is_valid (bool): Whether container is ready for scanning
            - container_info (Dict): Container information if available
            - checks_passed (List[str]): List of validation checks that passed
            - checks_failed (List[str]): List of validation checks that failed
            - recommendations (List[str]): Suggested actions if validation fails

    Validation Checks:
        - Container exists in container manager
        - Container is in running state
        - Container is accessible via Docker API
        - Container has required working directory
        - Container has Python runtime (for Python scanning)

    Example:
        ```python
        validation = await validate_container_access("my-container")
        if validation["is_valid"]:
            print("Container ready for scanning")
        else:
            print("Issues:", validation["checks_failed"])
            for rec in validation["recommendations"]:
                print(f"- {rec}")
        ```

    Error Handling:
        - Container manager failures are captured and reported
        - Docker API errors are handled gracefully
        - Individual check failures don't stop other checks
        - Detailed error information is provided for troubleshooting
    """
    validation_result: ValidationResultDict = {
        "is_valid": False,
        "container_info": {},
        "checks_passed": [],
        "checks_failed": [],
        "recommendations": [],
    }

    try:
        container_manager = get_container_manager()

        # Check 1: Container exists in manager
        try:
            containers = await container_manager.list_containers(tenant_id=tenant_id)
            container_info = None
            for container in containers:
                if container_name == container.container_name:
                    container_info = container
                    break

            if container_info:
                validation_result["checks_passed"].append("Container found in manager")
                validation_result["container_info"] = {
                    "name": container_info.container_name,
                    "status": container_info.status,
                    "image": getattr(container_info, "image", "unknown"),
                }
            else:
                validation_result["checks_failed"].append("Container not found in manager")
                validation_result["recommendations"].append(
                    "Verify container name and ensure it's managed by the system"
                )
                return validation_result

        except Exception as e:
            validation_result["checks_failed"].append(f"Container manager access failed: {e}")
            validation_result["recommendations"].append("Check container manager configuration")
            return validation_result

        # Check 2: Container is running
        if container_info.status == "running":
            validation_result["checks_passed"].append("Container is running")
        else:
            validation_result["checks_failed"].append(
                f"Container not running (status: {container_info.status})"
            )
            validation_result["recommendations"].append("Start the container before scanning")
            return validation_result

        # Check 3: Container accessible via coding-containers service
        try:
            container_detail = await container_manager.get_container(container_name)
            if container_detail:
                validation_result["checks_passed"].append(
                    "Container accessible via coding-containers service"
                )
                validation_result["container_info"].update(
                    {
                        "work_dir": container_detail.work_dir,
                        "repo_url": container_detail.repo_url,
                    }
                )
            else:
                validation_result["checks_failed"].append(
                    "Container not found via coding-containers service"
                )
                validation_result["recommendations"].append(
                    "Check coding-containers service and container status"
                )
                return validation_result

        except Exception as e:
            validation_result["checks_failed"].append(
                f"Coding-containers service access failed: {e}"
            )
            validation_result["recommendations"].append(
                "Check coding-containers service connectivity"
            )
            return validation_result

        # Check 4: Working directory exists
        try:
            result = await container_manager.execute_command(container_name, "test -d /workspace")
            if result.get("success"):
                validation_result["checks_passed"].append("Working directory /workspace exists")
            else:
                validation_result["checks_failed"].append("Working directory /workspace not found")
                validation_result["recommendations"].append(
                    "Ensure container has /workspace directory"
                )
        except Exception as e:
            validation_result["checks_failed"].append(f"Working directory check failed: {e}")

        # Check 5: Python runtime available (for Python scanning)
        try:
            result = await container_manager.execute_command(container_name, "python3 --version")
            if result.get("success"):
                validation_result["checks_passed"].append("Python runtime available")
                output = result.get("output")
                python_version = output.strip() if output else ""
                validation_result["container_info"]["python_version"] = python_version
            else:
                validation_result["checks_failed"].append("Python runtime not found")
                validation_result["recommendations"].append(
                    "Install Python in container for Python scanning"
                )
        except Exception as e:
            validation_result["checks_failed"].append(f"Python runtime check failed: {e}")

        # Overall validation
        critical_checks = [
            "Container found in manager",
            "Container is running",
            "Container accessible via Docker API",
        ]

        critical_passed = all(
            check in validation_result["checks_passed"] for check in critical_checks
        )
        validation_result["is_valid"] = (
            critical_passed and len(validation_result["checks_failed"]) == 0
        )

        if validation_result["is_valid"]:
            logger.info(f"Container {container_name} validated successfully for scanning")
        else:
            logger.warning(
                f"Container {container_name} validation failed: {validation_result['checks_failed']}"
            )

        return validation_result

    except Exception as e:
        logger.error(f"Unexpected error validating container {container_name}: {e}")
        validation_result["checks_failed"].append(f"Validation error: {e}")
        validation_result["recommendations"].append("Check logs for detailed error information")
        return validation_result


async def create_scan_context(
    container_name: str,
    scan_type: str,
    target_files: list[str] | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Create a context object for scan operations.

    This function creates a comprehensive context that includes all necessary
    information and configuration for a scanning operation. The context is
    used to ensure consistent configuration and provide traceability.

    Args:
        container_name (str): Name of the container to scan.
        scan_type (str): Type of scan to perform.
        target_files (Optional[List[str]]): Specific files to scan.

    Returns:
        Dict[str, Any]: Scan context containing:
            - container_name (str): Target container name
            - scan_type (str): Scan operation type
            - target_files (List[str]): Files to scan (empty for full scan)
            - config (Dict): Scanner configuration
            - validation (Dict): Container validation results
            - timestamp (str): Context creation timestamp
            - scan_id (str): Unique identifier for this scan

    Example:
        ```python
        context = await create_scan_context(
            container_name="my-app",
            scan_type="dependencies"
        )

        if context["validation"]["is_valid"]:
            # Proceed with scan using context
            config = context["config"]
            timeout = config["container_timeout"]
        ```
    """
    from datetime import datetime

    # Get scanner configuration
    config = get_scanner_config()

    # Validate container
    validation = await validate_container_access(container_name, tenant_id=tenant_id)

    # Create unique scan ID
    scan_id = str(uuid7())[:8]

    context = {
        "container_name": container_name,
        "scan_type": scan_type,
        "target_files": target_files or [],
        "config": config.to_dict(),
        "validation": validation,
        "timestamp": datetime.now(UTC).isoformat(),
        "scan_id": scan_id,
    }

    logger.info(f"Created scan context {scan_id} for {scan_type} scan of {container_name}")

    return context
