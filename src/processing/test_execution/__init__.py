"""
Test execution module for running tests in isolated containers.

This module provides comprehensive test execution capabilities including:
- Container-based test execution with security isolation
- HTTP-based communication with test service containers
- Timeout management and execution control
- Security validation and sanitization
- Result processing and analysis

Key Components:
- ContainerTestRunner: Main test runner for container execution
- ExecutionSecurityManager: Security validation and sanitization
- TimeoutManager: Execution timeout handling and monitoring
"""

from __future__ import annotations

# Security management
from .security_manager import ExecutionSecurityManager

# Test execution runners
from .test_runner import ContainerTestRunner

# Timeout management
from .timeout_manager import (
    ExecutionParams,
    TimeoutConfig,
    TimeoutManager,
    TimeoutResult,
    TimeoutSeverity,
    TimeoutType,
)

# Note: The following modules are placeholders and will be implemented as needed:
# - hypothesis_manager: Property-based testing integration
# - result_processor: Test result analysis and formatting

__all__ = [
    # Runners
    "ContainerTestRunner",
    # Security
    "ExecutionSecurityManager",
    # Timeout Management
    "TimeoutManager",
    "TimeoutConfig",
    "TimeoutType",
    "TimeoutSeverity",
    "TimeoutResult",
    "ExecutionParams",
]
