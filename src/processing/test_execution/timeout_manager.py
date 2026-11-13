"""
Execution timeout handling for test execution systems.

This module provides comprehensive timeout management for test execution,
including timeout configuration, execution monitoring, and cleanup handling.
Supports both Docker container timeouts and general async operation timeouts.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import (
    Any,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field, field_validator

ExitCode = int

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TimeoutType(StrEnum):
    """Types of timeout operations."""

    TEST_EXECUTION = "test_execution"
    CONTAINER_SETUP = "container_setup"
    CONTAINER_CLEANUP = "container_cleanup"
    FILE_OPERATIONS = "file_operations"
    NETWORK_OPERATIONS = "network_operations"


class TimeoutSeverity(StrEnum):
    """Severity levels for timeout handling."""

    SOFT = "soft"  # Graceful shutdown attempt
    HARD = "hard"  # Forced termination
    CRITICAL = "critical"  # System-level intervention


@dataclass
class TimeoutResult:
    """Result of a timeout operation."""

    success: bool
    timed_out: bool
    execution_time_seconds: float
    error_message: str | None = None
    exit_code: ExitCode | None = None
    cleanup_performed: bool = False


@dataclass
class ExecutionParams:
    """Parameters for timeout execution."""

    timeout_seconds: int | None = None
    timeout_type: TimeoutType = TimeoutType.TEST_EXECUTION
    task_id: str | None = None
    cleanup_handler: Callable[[], Awaitable[None]] | None = None


class TimeoutConfig(BaseModel):
    """Configuration for timeout operations."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    # Core timeout settings
    default_timeout_seconds: int = Field(
        default=30, ge=1, le=3600, description="Default timeout in seconds"
    )
    max_timeout_seconds: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Maximum allowed timeout in seconds",
    )

    # Graceful shutdown settings
    graceful_shutdown_timeout: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Time to wait for graceful shutdown",
    )

    # Container-specific timeouts
    container_start_timeout: int = Field(
        default=60, ge=5, le=300, description="Timeout for container startup"
    )
    container_stop_timeout: int = Field(
        default=30, ge=5, le=120, description="Timeout for container shutdown"
    )

    # Cleanup settings
    cleanup_on_timeout: bool = Field(
        default=True, description="Whether to perform cleanup on timeout"
    )
    retry_cleanup_attempts: int = Field(
        default=3, ge=1, le=10, description="Number of cleanup retry attempts"
    )

    @field_validator("max_timeout_seconds")
    @classmethod
    def validate_max_timeout(cls, v: int, info: Any) -> int:
        """Ensure max timeout is not less than default."""
        default_timeout = info.data.get("default_timeout_seconds", 30)
        if v < default_timeout:
            raise ValueError(
                f"max_timeout_seconds ({v}) must be >= default_timeout_seconds ({default_timeout})"
            )
        return v


class TimeoutManager:
    """
    Manages execution timeouts for test operations with comprehensive error handling.

    Provides timeout management for various operations including test execution,
    container management, and cleanup operations with configurable behavior.
    """

    def __init__(self, config: TimeoutConfig | None = None):
        """
        Initialize timeout manager with configuration.

        Args:
            config: Timeout configuration. Uses defaults if not provided.
        """
        self.config = config or TimeoutConfig()
        self._active_tasks: dict[str, asyncio.Task[Any]] = {}
        self._cleanup_handlers: dict[str, Callable[[], Awaitable[None]]] = {}

        logger.info(
            "TimeoutManager initialized with default_timeout=%ss, max_timeout=%ss",
            self.config.default_timeout_seconds,
            self.config.max_timeout_seconds,
        )

    async def execute_with_timeout(
        self,
        coro: Coroutine[Any, Any, T],
        params: ExecutionParams | None = None,
    ) -> TimeoutResult:
        """
        Execute a coroutine with timeout management.

        Args:
            coro: Coroutine to execute
            params: Execution parameters including timeout, type, and cleanup handler

        Returns:
            TimeoutResult with execution details
        """
        params = params or ExecutionParams()
        timeout = self._resolve_timeout(params.timeout_seconds, params.timeout_type)
        task_id = params.task_id or f"{params.timeout_type.value}_{int(time.time())}"
        start_time = time.time()

        logger.info(
            "Starting %s execution with %ss timeout (task_id: %s)",
            params.timeout_type.value,
            timeout,
            task_id,
        )

        if params.cleanup_handler:
            self._cleanup_handlers[task_id] = params.cleanup_handler

        try:
            # Create task and track it
            task: asyncio.Task[T] = asyncio.create_task(coro)
            self._active_tasks[task_id] = task

            # Wait for completion or timeout
            await asyncio.wait_for(task, timeout=timeout)
            execution_time = time.time() - start_time

            logger.info("Task %s completed successfully in %.2fs", task_id, execution_time)

            return TimeoutResult(
                success=True,
                timed_out=False,
                execution_time_seconds=execution_time,
                exit_code=0,
            )

        except TimeoutError:
            execution_time = time.time() - start_time
            logger.warning("Task %s timed out after %.2fs", task_id, execution_time)

            # Perform cleanup if configured
            cleanup_performed = False
            if self.config.cleanup_on_timeout and task_id in self._cleanup_handlers:
                cleanup_performed = await self._perform_cleanup(task_id)

            return TimeoutResult(
                success=False,
                timed_out=True,
                execution_time_seconds=execution_time,
                error_message=f"Operation timed out after {timeout}s",
                exit_code=124,
                cleanup_performed=cleanup_performed,
            )

        except (OSError, ValueError, RuntimeError) as e:
            execution_time = time.time() - start_time
            logger.error("Task %s failed with error: %s", task_id, e)

            return TimeoutResult(
                success=False,
                timed_out=False,
                execution_time_seconds=execution_time,
                error_message=str(e),
                exit_code=1,
            )

        finally:
            # Clean up task tracking
            self._active_tasks.pop(task_id, None)
            self._cleanup_handlers.pop(task_id, None)

    @asynccontextmanager
    async def timeout_context(
        self,
        timeout_seconds: int | None = None,
        timeout_type: TimeoutType = TimeoutType.TEST_EXECUTION,
        task_id: str | None = None,
    ) -> Any:
        """
        Async context manager for timeout handling.

        Args:
            timeout_seconds: Timeout in seconds
            timeout_type: Type of timeout operation
            task_id: Unique identifier for this task

        Yields:
            Context manager for timeout handling

        Raises:
            asyncio.TimeoutError: If operation times out
        """
        timeout = self._resolve_timeout(timeout_seconds, timeout_type)
        task_id = task_id or f"{timeout_type.value}_{int(time.time())}"

        logger.debug("Entering timeout context for %s with %ss timeout", task_id, timeout)

        try:
            async with asyncio.timeout(timeout):
                yield
            logger.debug("Timeout context for %s completed successfully", task_id)

        except TimeoutError:
            logger.warning("Timeout context for %s timed out after %ss", task_id, timeout)
            raise

    async def cancel_task(self, task_id: str, graceful: bool = True) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: Task identifier to cancel
            graceful: Whether to attempt graceful cancellation

        Returns:
            True if task was cancelled successfully
        """
        if task_id not in self._active_tasks:
            logger.warning("Task %s not found in active tasks", task_id)
            return False

        task = self._active_tasks[task_id]

        if graceful:
            logger.info("Attempting graceful cancellation of task %s", task_id)
            task.cancel()

            try:
                await asyncio.wait_for(task, timeout=self.config.graceful_shutdown_timeout)
                logger.info("Task %s cancelled gracefully", task_id)
                return True

            except (TimeoutError, asyncio.CancelledError):
                logger.warning("Graceful cancellation of %s timed out", task_id)

        # Force cancellation
        logger.info("Force cancelling task %s", task_id)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Perform cleanup
        if task_id in self._cleanup_handlers:
            await self._perform_cleanup(task_id)

        return True

    async def get_active_tasks(self) -> dict[str, dict[str, Any]]:
        """
        Get information about currently active tasks.

        Returns:
            Dictionary of active task information
        """
        active_info = {}

        for task_id, task in self._active_tasks.items():
            active_info[task_id] = {
                "done": task.done(),
                "cancelled": task.cancelled(),
                "has_cleanup": task_id in self._cleanup_handlers,
            }

        return active_info

    async def cleanup_all_tasks(self) -> int:
        """
        Cancel and cleanup all active tasks.

        Returns:
            Number of tasks that were cleaned up
        """
        active_task_ids = list(self._active_tasks.keys())

        if not active_task_ids:
            logger.info("No active tasks to cleanup")
            return 0

        logger.info("Cleaning up %d active tasks", len(active_task_ids))

        cleanup_count = 0
        for task_id in active_task_ids:
            if await self.cancel_task(task_id, graceful=False):
                cleanup_count += 1

        logger.info("Cleaned up %d tasks", cleanup_count)
        return cleanup_count

    def _resolve_timeout(self, timeout_seconds: int | None, timeout_type: TimeoutType) -> int:
        """
        Resolve timeout value based on type and configuration.

        Args:
            timeout_seconds: Requested timeout
            timeout_type: Type of timeout operation

        Returns:
            Resolved timeout value in seconds
        """
        if timeout_seconds is not None:
            # Validate against maximum
            if timeout_seconds > self.config.max_timeout_seconds:
                logger.warning(
                    "Requested timeout %ss exceeds maximum %ss, using maximum",
                    timeout_seconds,
                    self.config.max_timeout_seconds,
                )
                return self.config.max_timeout_seconds
            return timeout_seconds

        # Use type-specific defaults
        type_defaults = {
            TimeoutType.TEST_EXECUTION: self.config.default_timeout_seconds,
            TimeoutType.CONTAINER_SETUP: self.config.container_start_timeout,
            TimeoutType.CONTAINER_CLEANUP: self.config.container_stop_timeout,
            TimeoutType.FILE_OPERATIONS: min(30, self.config.default_timeout_seconds),
            TimeoutType.NETWORK_OPERATIONS: min(60, self.config.default_timeout_seconds),
        }

        return type_defaults.get(timeout_type, self.config.default_timeout_seconds)

    async def _perform_cleanup(self, task_id: str) -> bool:
        """
        Perform cleanup for a task.

        Args:
            task_id: Task identifier

        Returns:
            True if cleanup was successful
        """
        if task_id not in self._cleanup_handlers:
            return False

        cleanup_handler = self._cleanup_handlers[task_id]

        for attempt in range(self.config.retry_cleanup_attempts):
            try:
                logger.info("Performing cleanup for task %s (attempt %d)", task_id, attempt + 1)
                await asyncio.wait_for(
                    cleanup_handler(),
                    timeout=self.config.graceful_shutdown_timeout,
                )
                logger.info("Cleanup successful for task %s", task_id)
                return True

            except (TimeoutError, OSError, RuntimeError) as e:
                logger.warning("Cleanup attempt %d failed for task %s: %s", attempt + 1, task_id, e)
                if attempt < self.config.retry_cleanup_attempts - 1:
                    await asyncio.sleep(1)  # Brief delay before retry

        logger.error("All cleanup attempts failed for task %s", task_id)
        return False


# Factory functions for common timeout configurations


def create_test_timeout_manager() -> TimeoutManager:
    """Create a timeout manager optimized for test execution."""
    config = TimeoutConfig(
        default_timeout_seconds=30,
        max_timeout_seconds=300,
        graceful_shutdown_timeout=10,
        cleanup_on_timeout=True,
    )
    return TimeoutManager(config)


def create_container_timeout_manager() -> TimeoutManager:
    """Create a timeout manager optimized for container operations."""
    config = TimeoutConfig(
        default_timeout_seconds=60,
        max_timeout_seconds=600,
        container_start_timeout=120,
        container_stop_timeout=60,
        graceful_shutdown_timeout=15,
        cleanup_on_timeout=True,
        retry_cleanup_attempts=5,
    )
    return TimeoutManager(config)


def create_production_timeout_manager() -> TimeoutManager:
    """Create a timeout manager with conservative production settings."""
    config = TimeoutConfig(
        default_timeout_seconds=60,
        max_timeout_seconds=900,
        container_start_timeout=180,
        container_stop_timeout=90,
        graceful_shutdown_timeout=30,
        cleanup_on_timeout=True,
        retry_cleanup_attempts=3,
    )
    return TimeoutManager(config)
