"""Python test suite security manager."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Patterns that are always considered dangerous regardless of config
_DANGEROUS_PATTERNS: list[str] = [
    "os.system(",
    "subprocess.call(",
    "subprocess.run(",
    "subprocess.Popen(",
    "eval(",
    "exec(",
    "__import__(",
]


class TestValidationSecurityConfig(BaseModel):
    __test__ = False
    """Security configuration for test validation."""

    allow_network: bool = False
    allow_filesystem: bool = True
    allow_subprocess: bool = False
    max_file_size_mb: int = 10
    allowed_paths: list[str] = []
    blocked_modules: list[str] = []

    # Extra fields used by test_runner.py
    max_execution_time: int = 30
    max_memory_mb: int = 512
    max_code_length: int = 10000
    allowed_modules: list[str] = []


class SecurityManager:
    """Security manager for test execution."""

    def __init__(
        self,
        logger_or_config: Any = None,
        config: TestValidationSecurityConfig | None = None,
    ) -> None:
        """Initialize security manager.

        Supports two call patterns:
          - SecurityManager(config)                  — single-arg, config first
          - SecurityManager(logger, config)           — logger first, config second
        """
        if isinstance(logger_or_config, TestValidationSecurityConfig):
            # Called as SecurityManager(config)
            self.logger = logger
            self.config = logger_or_config
        elif logger_or_config is None and config is None:
            self.logger = logger
            self.config = TestValidationSecurityConfig()
        else:
            # Called as SecurityManager(some_logger, config)
            self.logger = logger_or_config if logger_or_config is not None else logger
            self.config = config if config is not None else TestValidationSecurityConfig()

        logging.getLogger(__name__).debug("SecurityManager initialized")

    def validate_code(self, code: str) -> dict[str, Any]:
        """Validate code for security issues.

        Checks for dangerous built-ins and blocked module imports.

        Returns:
            Dict with 'safe' (bool) and 'issues' (list[str]).
        """
        issues: list[str] = []

        # 1. Check hard-coded dangerous patterns
        for pattern in _DANGEROUS_PATTERNS:
            if pattern in code:
                issues.append(f"Dangerous pattern detected: {pattern}")

        # 2. Check blocked modules from config
        for module in self.config.blocked_modules:
            # Match both "import <module>" and "from <module>"
            if f"import {module}" in code or f"from {module}" in code:
                issues.append(f"Blocked module used: {module}")

        return {"safe": len(issues) == 0, "issues": issues}

    def validate_test_request(self, request: Any) -> bool:
        """Validate a test execution request against configured limits.

        Returns:
            True if request is within limits, False otherwise.
        """
        # Check timeout
        timeout = getattr(request, "timeout", None) or getattr(request, "timeout_seconds", None)
        if timeout is not None and self.config.max_execution_time:
            if timeout > self.config.max_execution_time:
                return False

        # Check code length
        code = getattr(request, "code", "") or getattr(request, "source_code", "") or ""
        if self.config.max_code_length and len(code) > self.config.max_code_length:
            return False

        return True

    def get_sandbox_config(self) -> dict[str, Any]:
        """Get sandbox configuration."""
        return {
            "allow_network": self.config.allow_network,
            "allow_filesystem": self.config.allow_filesystem,
            "allow_subprocess": self.config.allow_subprocess,
        }
