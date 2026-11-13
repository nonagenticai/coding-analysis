"""
HTTP Quality Runner with proper session lifecycle management.
Resolves session leaks and improves connection pooling efficiency.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import weakref
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any
from uuid import uuid7

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from models.testing_models import QualityRequest, QualityResult
from services.cache_manager import RedisCacheManager

from .http_quality_metrics import (
    PROMETHEUS_AVAILABLE,
    quality_connection_pool_size,
    quality_duration_seconds,
    quality_requests_total,
    quality_retries_total,
    quality_service_info,
    quality_sessions_active,
    quality_sessions_created_total,
    quality_sessions_errors_total,
)
from .http_quality_models import (
    QualityServiceConfig,
    RequestMetrics,
    SessionManager,
    SessionStats,
)

logger = logging.getLogger(__name__)


class SessionMonitor:
    """Monitor and track HTTP session health and performance.

    Attributes:
        stats (SessionStats): Instance attribute stats.
        _lock: Instance attribute _lock."""

    def __init__(self) -> None:
        self.stats = SessionStats()
        self.request_metrics: list[RequestMetrics] = []
        self.error_patterns: dict[str, int] = defaultdict(int)
        self.session_registry: weakref.WeakSet[ClientSession] = weakref.WeakSet()
        self._lock = asyncio.Lock()

    async def register_session(self, session: ClientSession) -> None:
        """Register a new session for monitoring.

        Args:
            session (ClientSession): The session parameter.

        Returns:
            None: The none result."""
        async with self._lock:
            self.session_registry.add(session)
            self.stats.total_sessions_created += 1
            self.stats.active_sessions = len(self.session_registry)
            logger.debug("Session registered. Total active: %d", self.stats.active_sessions)

    async def record_session_closed(self) -> None:
        """Record that a session was closed.

        Returns:
            None: The none result."""
        async with self._lock:
            self.stats.total_sessions_closed += 1
            self.stats.active_sessions = len(self.session_registry)

    async def record_session_error(self, error_message: str) -> None:
        """Record a session error for analysis.

        Args:
            error_message (str): The error_message parameter.

        Returns:
            None: The none result."""
        async with self._lock:
            self.stats.failed_sessions += 1
            self.stats.session_errors.append(f"{datetime.now(UTC).isoformat()}: {error_message}")

            # Keep only last 50 errors
            if len(self.stats.session_errors) > 50:
                self.stats.session_errors = self.stats.session_errors[-25:]

            # Track error patterns
            error_type = self._categorize_error(error_message)
            self.error_patterns[error_type] += 1

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error messages for pattern analysis."""
        error_lower = error_message.lower()

        if "session is closed" in error_lower:
            return "session_closed"
        if "connection" in error_lower and "refused" in error_lower:
            return "connection_refused"
        if "timeout" in error_lower:
            return "timeout"
        if "pool" in error_lower:
            return "connection_pool_issue"
        if "502" in error_message or "503" in error_message:
            return "service_unavailable"
        return "other"

    def get_health_summary(self) -> dict[str, Any]:
        """Get session health summary.

        Returns:
            Dict[str, Any]: The dict[str, any] result."""
        return {
            "active_sessions": self.stats.active_sessions,
            "total_created": self.stats.total_sessions_created,
            "total_closed": self.stats.total_sessions_closed,
            "failed_sessions": self.stats.failed_sessions,
            "recent_errors": len(self.stats.session_errors),
            "error_patterns": dict(self.error_patterns),
        }


class HTTPQualityRunner:
    """Quality runner with proper session lifecycle management.
    Resolves session leaks and improves connection pooling efficiency.

    Attributes:
        cache_manager: Instance attribute cache_manager.
        config: Instance attribute config.
        timeout (ClientTimeout): Instance attribute timeout.
        _session_manager (SessionManager): Instance attribute _session_manager.
        _request_semaphore: Instance attribute _request_semaphore.
        _batch_semaphore: Instance attribute _batch_semaphore.
        monitor (SessionMonitor): Instance attribute monitor.

    Example:
        >>> obj = HTTPQualityRunner("cache_manager_value")"""

    def __init__(self, cache_manager: RedisCacheManager | None = None):
        self.cache_manager = cache_manager

        # Initialize configuration with validation
        self.config = self._load_and_validate_config()
        self.timeout = ClientTimeout(
            total=self.config.total_timeout,
            connect=self.config.connect_timeout,
        )

        # Session management
        self._session_manager = SessionManager()

        # Rate limiting and batch management
        self._request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._batch_semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)

        # Session monitoring
        self.monitor = SessionMonitor()

        # Initialize Prometheus metrics with service info
        if PROMETHEUS_AVAILABLE:
            quality_service_info.info(
                {
                    "service_url": self.config.service_url,
                    "max_connections": str(self.config.max_concurrent_connections),
                    "max_requests": str(self.config.max_concurrent_requests),
                    "total_timeout": str(self.config.total_timeout),
                    "version": "1.0.0",
                }
            )

        logger.info(
            "HTTP Quality Runner initialized with validated config: max_connections=%d, max_requests=%d, timeout=%ds",
            self.config.max_concurrent_connections,
            self.config.max_concurrent_requests,
            self.config.total_timeout,
        )

    def _load_and_validate_config(self) -> QualityServiceConfig:
        """Load configuration from environment variables and validate."""
        try:
            from config.settings import get_settings

            service_url = get_settings().QUALITY_ANALYSIS_SERVICE_URL
            if not service_url:
                raise RuntimeError(
                    "QUALITY_ANALYSIS_SERVICE_URL is required; set it in the Deployment env block."
                )
            config = QualityServiceConfig(
                service_url=service_url,
                total_timeout=int(os.getenv("QUALITY_HTTP_TIMEOUT", "3600")),
                connect_timeout=int(os.getenv("QUALITY_HTTP_CONNECT_TIMEOUT", "60")),
                max_concurrent_connections=int(os.getenv("QUALITY_MAX_CONNECTIONS", "50")),
                max_concurrent_requests=int(os.getenv("QUALITY_MAX_CONCURRENT_REQUESTS", "32")),
                max_concurrent_batches=int(os.getenv("QUALITY_MAX_CONCURRENT_BATCHES", "12")),
                max_retries=int(os.getenv("QUALITY_CONNECTION_RETRY_ATTEMPTS", "3")),
                keepalive_timeout=int(os.getenv("QUALITY_CONNECTION_KEEPALIVE", "120")),
                max_requests_per_session=int(os.getenv("QUALITY_MAX_REQUESTS_PER_SESSION", "1000")),
            )

            logger.info("Quality service configuration validated successfully")
            return config

        except (ValueError, TypeError) as e:
            logger.error("Configuration validation failed: %s", e)
            raise

    async def _create_session(self) -> ClientSession:
        """Create a new HTTP session with optimal configuration."""
        try:
            connector = TCPConnector(
                limit=self.config.max_concurrent_connections,
                limit_per_host=self.config.max_concurrent_connections,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=self.config.keepalive_timeout,
                enable_cleanup_closed=True,
            )

            session = ClientSession(
                timeout=self.timeout,
                connector=connector,
                raise_for_status=False,  # We'll handle status codes manually
            )

            # Register session with monitor
            await self.monitor.register_session(session)

            # Update metrics
            quality_sessions_created_total.inc()
            quality_sessions_active.set(self.monitor.stats.active_sessions)
            quality_connection_pool_size.set(self.config.max_concurrent_connections)

            logger.debug("Created new HTTP session with optimized configuration")
            return session

        except (TimeoutError, ClientError, OSError) as e:
            logger.error("Failed to create session: %s", e)
            await self.monitor.record_session_error(f"Session creation failed: {e}")
            quality_sessions_errors_total.labels(error_type="creation_failed").inc()
            raise

    async def _ensure_session(self) -> ClientSession:
        """Ensure we have a healthy session, creating new one if needed."""
        async with self._session_manager.lock:
            now = datetime.now(UTC)

            # Check if session needs replacement
            needs_new_session = (
                self._session_manager.session is None
                or self._session_manager.session.closed
                or (
                    self._session_manager.created_at
                    and (now - self._session_manager.created_at).total_seconds() > 3600
                )  # 1 hour max age
                or self._session_manager.request_count >= self.config.max_requests_per_session
            )

            if needs_new_session:
                # Clean up old session
                if self._session_manager.session and not self._session_manager.session.closed:
                    await self._cleanup_session(self._session_manager.session)

                # Create new session
                self._session_manager.session = await self._create_session()
                self._session_manager.created_at = now
                self._session_manager.request_count = 0
                logger.debug("Session refreshed")

            # At this point, self._session should never be None
            if self._session_manager.session is None:
                raise RuntimeError("Session is None after creation - this should never happen")
            return self._session_manager.session

    async def _cleanup_session(self, session: ClientSession) -> None:
        """Properly cleanup a session with sufficient time for connection cleanup."""
        try:
            if session and not session.closed:
                await session.close()
                # Give more time for proper cleanup - critical for preventing leaks
                await asyncio.sleep(0.5)
                await self.monitor.record_session_closed()
                logger.debug("Session cleaned up successfully")
        except (TimeoutError, ClientError, OSError) as e:
            logger.warning("Error during session cleanup: %s", e)
            await self.monitor.record_session_error(f"Session cleanup error: {e}")

    async def _health_check_with_retry(self, max_retries: int = 3, delay: int = 2) -> bool:
        """Check service health with retry logic."""
        for attempt in range(max_retries):
            try:
                session = await self._ensure_session()
                health_url = f"{self.config.service_url}/health"
                async with session.get(health_url, timeout=ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        logger.debug("Health check passed for %s", self.config.service_url)
                        return True
                    logger.warning("Health check failed with status %d", response.status)
            except (TimeoutError, ClientError, OSError) as e:
                logger.debug("Health check attempt %d failed: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
        return False

    async def close(self) -> None:
        """Close HTTP session and clean up resources.

        Returns:
            None: The none result."""
        async with self._session_manager.lock:
            if self._session_manager.session:
                await self._cleanup_session(self._session_manager.session)
                self._session_manager.session = None
            logger.info("HTTPQualityRunner closed")

    async def run_quality_analysis(self, request: QualityRequest) -> QualityResult:
        """Execute quality analysis with optimized session management and metrics.

        Args:
            request (QualityRequest): The request parameter.

        Returns:
            QualityResult: The qualityresult result.

        Raises:
            RuntimeError: Raised under certain conditions.

        Example:
            >>> await run_quality_analysis("request_value")"""
        analysis_id = str(uuid7())
        start_time = datetime.now(UTC)
        language = request.language or "unknown"

        # Check if quality analysis is enabled
        if os.getenv("ENABLE_QUALITY_ANALYSIS", "true").lower() == "false":
            logger.info(
                "Quality analysis is disabled via ENABLE_QUALITY_ANALYSIS environment variable"
            )
            return self._create_failed_quality_result(
                analysis_id=analysis_id,
                start_time=start_time,
                language=language,
                error_message="Quality analysis is disabled",
            )

        # Record request start
        start_timestamp = time.time()

        # Use batch semaphore for request-level rate limiting
        async with self._batch_semaphore:
            # Use request semaphore for connection-level rate limiting
            async with self._request_semaphore:
                try:
                    logger.debug("HTTP quality analysis %s: %s", analysis_id, language)

                    # Health check before proceeding
                    if not await self._health_check_with_retry():
                        error_msg = (
                            f"Quality analyzer service at {self.config.service_url} is not healthy"
                        )
                        logger.warning(error_msg)
                        # Return empty result when service is unavailable
                        return self._create_failed_quality_result(
                            analysis_id=analysis_id,
                            start_time=start_time,
                            language=request.language or "unknown",
                            error_message="Quality analyzer service unavailable",
                        )

                    # Prepare request payload
                    payload = {
                        "code_path": request.code_path,
                        "code_content": request.code_content,
                        "language": request.language,
                        "tools": request.tools or [],
                        "fix_issues": getattr(request, "fix_issues", False),
                        "custom_rules": getattr(request, "custom_rules", None),
                    }

                    # Remove None values
                    payload = {k: v for k, v in payload.items() if v is not None}

                    # Get session (reuse existing or create new)
                    session = await self._ensure_session()

                    # Increment request counter
                    async with self._session_manager.lock:
                        self._session_manager.request_count += 1

                    # Execute request with retry logic
                    result = await self._execute_request_with_retry(
                        session,
                        "analyze",
                        payload,
                        analysis_id=analysis_id,
                        start_time=start_time,
                    )

                    # Record metrics
                    execution_time = time.time() - start_timestamp
                    quality_duration_seconds.labels(language=language).observe(execution_time)
                    quality_requests_total.labels(
                        language=language,
                        status="success" if result.success else "failure",
                    ).inc()

                    return result

                except (TimeoutError, ClientError, ValueError, OSError) as e:
                    logger.error("Quality analysis %s failed: %s", analysis_id, e)
                    await self.monitor.record_session_error(str(e))

                    # Record error metrics
                    execution_time = time.time() - start_timestamp
                    quality_duration_seconds.labels(language=language).observe(execution_time)
                    quality_requests_total.labels(language=language, status="error").inc()
                    quality_sessions_errors_total.labels(error_type="request_failed").inc()

                    return self._create_failed_quality_result(
                        analysis_id=analysis_id,
                        start_time=start_time,
                        language=request.language or "unknown",
                        error_message=str(e),
                    )

    async def _execute_request_with_retry(
        self,
        session: ClientSession,
        endpoint: str,
        payload: dict[str, Any],
        *,
        analysis_id: str,
        start_time: datetime,
    ) -> QualityResult:
        """Execute HTTP request with exponential backoff retry logic."""

        for attempt in range(self.config.max_retries):
            try:
                # Record retry attempt
                if attempt > 0:
                    quality_retries_total.labels(attempt=str(attempt)).inc()

                async with session.post(
                    f"{self.config.service_url}/{endpoint}",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        result_data = await response.json()

                        # Convert response to QualityResult
                        return QualityResult(
                            analysis_id=result_data.get("analysis_id", analysis_id),
                            success=result_data.get("success", False),
                            language=result_data.get(
                                "language", payload.get("language", "unknown")
                            ),
                            tools_results=result_data.get("tools_results", {}),
                            overall_score=result_data.get("overall_score", 0.0),
                            total_issues=result_data.get("total_issues", 0),
                            issues=result_data.get("issues", []),
                            execution_time=result_data.get("execution_time", 0.0),
                            created_at=datetime.fromisoformat(
                                result_data.get("created_at", start_time.isoformat())
                            ),
                            error=result_data.get("error", None),
                        )
                    # Handle HTTP error
                    error_text = await response.text()
                    logger.warning("Quality service HTTP error %d: %s", response.status, error_text)

                    # Don't retry on 4xx errors (client errors)
                    if 400 <= response.status < 500:
                        break

                    # Retry on 5xx errors (server errors)
                    if attempt < self.config.max_retries - 1:
                        wait_time = (2**attempt) + (
                            attempt * 0.1
                        )  # Exponential backoff with jitter
                        logger.info(
                            "Retrying in %.1f seconds... (attempt %d)",
                            wait_time,
                            attempt + 1,
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    error_message = (
                        error_text
                        if error_text.strip()
                        else f"HTTP {response.status} error with no details"
                    )
                    return self._create_failed_quality_result(
                        analysis_id=analysis_id,
                        start_time=start_time,
                        language=payload.get("language", "unknown"),
                        error_message=f"HTTP {response.status}: {error_message}",
                    )

            except (TimeoutError, ClientError) as e:
                error_msg = (
                    f"Connection error on attempt {attempt + 1}/{self.config.max_retries}: {e}"
                )
                logger.warning(error_msg)
                await self.monitor.record_session_error(error_msg)

                if attempt < self.config.max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2**attempt) + (attempt * 0.1)
                    logger.info("Retrying in %.1f seconds...", wait_time)
                    await asyncio.sleep(wait_time)

                    # Force session recreation on connection errors
                    if "session" in str(e).lower() or isinstance(e, asyncio.TimeoutError):
                        logger.info("Forcing session recreation due to connection issue")
                        async with self._session_manager.lock:
                            if self._session_manager.session:
                                await self._cleanup_session(self._session_manager.session)
                                self._session_manager.session = None
                        # Get new session for retry
                        session = await self._ensure_session()

                    continue
                # Final attempt failed
                final_error = f"All {self.config.max_retries} connection attempts failed: {e}"
                logger.error("All %d connection attempts failed: %s", self.config.max_retries, e)
                await self.monitor.record_session_error(final_error)

                return self._create_failed_quality_result(
                    analysis_id=analysis_id,
                    start_time=start_time,
                    language=payload.get("language", "unknown"),
                    error_message=final_error,
                )

        # Should not reach here, but just in case
        return self._create_failed_quality_result(
            analysis_id=analysis_id,
            start_time=start_time,
            language=payload.get("language", "unknown"),
            error_message="Unexpected retry loop exit",
        )

    async def run_single_tool(self, tool_name: str, language: str, code: str) -> dict[str, Any]:
        """Run a single quality tool with optimized session management.

        Args:
            tool_name (str): The tool_name parameter.
            language (str): The language parameter.
            code (str): The code parameter.

        Returns:
            Dict[str, Any]: The dict[str, any] result.

        Example:
            >>> await run_single_tool("example", "language_value", "code_value")"""
        async with self._request_semaphore:
            try:
                logger.debug("Running single tool %s for %s", tool_name, language)

                payload = {
                    "code_content": code,
                    "language": language,
                    "tools": [tool_name],
                }

                session = await self._ensure_session()

                async with self._session_manager.lock:
                    self._session_manager.request_count += 1

                # Execute with retry logic
                result = await self._execute_request_with_retry(
                    session,
                    "analyze",
                    payload,
                    analysis_id=f"single_tool_{tool_name}",
                    start_time=datetime.now(UTC),
                )

                # Extract tool-specific result
                if result.success and tool_name in result.tools_results:
                    return result.tools_results[tool_name]
                return {
                    "success": False,
                    "error": result.error or f"Tool {tool_name} not found in results",
                    "issues": [],
                    "execution_time": result.execution_time,
                }

            except (TimeoutError, ClientError, ValueError, OSError) as e:
                logger.error("Single tool execution failed: %s", e)
                await self.monitor.record_session_error(f"Single tool error: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "issues": [],
                    "execution_time": 0.0,
                }

    async def get_available_tools(self, language: str | None = None) -> dict[str, list[str]]:
        """Get available quality tools with optimized session management.

        Args:
            language (Optional[str], optional): The language parameter. Defaults to None.

        Returns:
            Dict[str, List[str]]: The dict[str, list[str]] result.

        Example:
            >>> await get_available_tools("language_value")"""
        async with self._request_semaphore:
            try:
                session = await self._ensure_session()

                async with session.get(f"{self.config.service_url}/info") as response:
                    if response.status == 200:
                        info_data = await response.json()
                        available_tools: dict[str, list[str]] = info_data.get("available_tools", {})

                        if language:
                            return {language: available_tools.get(language, [])}
                        return available_tools
                    logger.error("Failed to get available tools: HTTP %d", response.status)
                    return {}

            except (TimeoutError, ClientError, OSError) as e:
                logger.error("Failed to get available tools: %s", e)
                await self.monitor.record_session_error(f"Get tools error: {e}")
                return {}

    async def detailed_health_check(self) -> dict[str, Any]:
        """Perform detailed health check with comprehensive diagnostics.

        Returns:
            Dict[str, Any]: The dict[str, any] result.

        Example:
            >>> await detailed_health_check()"""
        try:
            health_start = time.time()

            # Basic service health check
            basic_health = await self.health_check()

            # Extended diagnostics
            diagnostics = {
                "configuration": {
                    "service_url": self.config.service_url,
                    "max_connections": self.config.max_concurrent_connections,
                    "max_requests": self.config.max_concurrent_requests,
                    "timeout_config": {
                        "total": self.config.total_timeout,
                        "connect": self.config.connect_timeout,
                    },
                    "retry_config": {
                        "max_retries": self.config.max_retries,
                        "keepalive_timeout": self.config.keepalive_timeout,
                    },
                },
                "session_health": self.monitor.get_health_summary(),
                "resource_usage": await self._get_resource_metrics(),
                "service_connectivity": await self._test_service_connectivity(),
                "performance_metrics": await self._get_performance_metrics(),
            }

            # Overall health assessment
            health_score = self._calculate_health_score(basic_health, diagnostics)

            diagnostics_time = time.time() - health_start

            return {
                **basic_health,
                "health_score": health_score,
                "detailed_diagnostics": diagnostics,
                "diagnostics_execution_time": round(diagnostics_time, 3),
            }

        except (TimeoutError, ClientError, OSError) as e:
            logger.error("Detailed health check failed: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    async def health_check(self) -> dict[str, Any]:
        """Check health of quality analysis service and session management.

        Returns:
            Dict[str, Any]: The dict[str, any] result.

        Example:
            >>> await health_check()"""
        try:
            session = await self._ensure_session()

            async with session.get(
                f"{self.config.service_url}/health",
                timeout=ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    health_data = await response.json()

                    # Add session health information
                    session_health = self.monitor.get_health_summary()

                    return {
                        "status": "healthy",
                        "url": self.config.service_url,
                        "service_details": health_data,
                        "session_health": session_health,
                    }
                return {
                    "status": "unhealthy",
                    "url": self.config.service_url,
                    "error": f"HTTP {response.status}",
                    "session_health": self.monitor.get_health_summary(),
                }

        except (TimeoutError, ClientError, OSError) as e:
            await self.monitor.record_session_error(f"Health check error: {e}")
            return {
                "status": "error",
                "url": self.config.service_url,
                "error": str(e),
                "session_health": self.monitor.get_health_summary(),
            }

    async def get_service_info(self) -> dict[str, Any]:
        """Get quality service information with optimized session management.

        Returns:
            Dict[str, Any]: The dict[str, any] result.

        Raises:
            RuntimeError: Raised under certain conditions.

        Example:
            >>> await get_service_info()"""
        async with self._request_semaphore:
            try:
                session = await self._ensure_session()

                async with session.get(f"{self.config.service_url}/info") as response:
                    if response.status == 200:
                        service_info: dict[str, Any] = await response.json()
                        return service_info
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {error_text}")

            except (TimeoutError, ClientError, OSError) as e:
                logger.error("Failed to get quality service info: %s", e)
                await self.monitor.record_session_error(f"Get service info error: {e}")
                raise

    # Additional diagnostic methods

    async def _get_resource_metrics(self) -> dict[str, Any]:
        """Get current resource usage metrics."""
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil not available"}

        try:
            process = psutil.Process()

            return {
                "memory_usage_mb": round(process.memory_info().rss / 1024 / 1024, 2),
                "cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files()),
                "connections": len(process.net_connections()),
                "threads": process.num_threads(),
            }
        except OSError as e:
            return {"error": str(e)}

    async def _test_service_connectivity(self) -> dict[str, Any]:
        """Test connectivity to the quality service."""
        try:
            session = await self._ensure_session()
            start_time = time.time()

            async with session.get(
                f"{self.config.service_url}/health",
                timeout=ClientTimeout(total=10),
            ) as response:
                response_time = round((time.time() - start_time) * 1000, 2)  # ms

                return {
                    "reachable": True,
                    "response_time_ms": response_time,
                    "status_code": response.status,
                    "response_size": len(await response.text()),
                }

        except (TimeoutError, ClientError, OSError) as e:
            return {
                "reachable": False,
                "error": str(e),
                "response_time_ms": None,
            }

    async def _get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics for the quality runner."""
        try:
            session_stats = self.monitor.get_health_summary()

            # Calculate derived metrics
            if session_stats["total_created"] > 0:
                session_efficiency = (
                    session_stats["total_closed"] / session_stats["total_created"] * 100
                )
            else:
                session_efficiency = 100.0

            error_rate = 0.0
            if session_stats["total_created"] > 0:
                error_rate = session_stats["failed_sessions"] / session_stats["total_created"] * 100

            return {
                "session_efficiency_percent": round(session_efficiency, 2),
                "error_rate_percent": round(error_rate, 2),
                "active_sessions": session_stats["active_sessions"],
                "connection_pool_utilization": round(
                    session_stats["active_sessions"] / self.config.max_concurrent_connections * 100,
                    2,
                ),
                "configuration_score": self._calculate_config_score(),
            }

        except (ArithmeticError, ZeroDivisionError, ValueError) as e:
            return {"error": str(e)}

    def _calculate_health_score(
        self, basic_health: dict[str, Any], diagnostics: dict[str, Any]
    ) -> float:
        """Calculate overall health score (0-100)."""
        score = 100.0

        # Basic health check
        if basic_health.get("status") != "healthy":
            score -= 50

        # Session health
        session_health = diagnostics.get("session_health", {})
        if session_health.get("failed_sessions", 0) > 0:
            failure_rate = session_health["failed_sessions"] / max(
                session_health.get("total_created", 1), 1
            )
            score -= failure_rate * 30

        # Service connectivity
        connectivity = diagnostics.get("service_connectivity", {})
        if not connectivity.get("reachable", False):
            score -= 30
        elif connectivity.get("response_time_ms", 0) > 5000:  # > 5 seconds
            score -= 10

        # Resource usage
        resource_usage = diagnostics.get("resource_usage", {})
        memory_mb = resource_usage.get("memory_usage_mb", 0)
        if memory_mb > 1000:  # > 1GB
            score -= 10

        return max(0.0, min(100.0, score))

    def _calculate_config_score(self) -> float:
        """Calculate configuration optimality score (0-100)."""
        score = 100.0

        # Check for optimal ratios
        request_to_connection_ratio = (
            self.config.max_concurrent_requests / self.config.max_concurrent_connections
        )
        if request_to_connection_ratio > 0.8:  # Too high utilization
            score -= 10
        elif request_to_connection_ratio < 0.3:  # Under-utilization
            score -= 5

        # Check timeout configuration
        if self.config.total_timeout < 300:  # < 5 minutes might be too short
            score -= 5
        elif self.config.total_timeout > 7200:  # > 2 hours might be too long
            score -= 10

        # Check retry configuration
        if self.config.max_retries < 2:
            score -= 5
        elif self.config.max_retries > 5:
            score -= 5

        return max(0.0, min(100.0, score))

    def _create_failed_quality_result(
        self,
        analysis_id: str,
        start_time: datetime,
        language: str,
        error_message: str | None = None,
    ) -> QualityResult:
        """Create a consistent failed quality result with proper error handling."""
        # Ensure error message is never None or empty
        if not error_message or error_message.strip() == "":
            error_message = "Quality analysis failed - no specific error details available"

        return QualityResult(
            analysis_id=analysis_id,
            success=False,
            language=language,
            tools_results={},
            overall_score=0.0,
            total_issues=0,
            execution_time=(datetime.now(UTC) - start_time).total_seconds(),
            created_at=start_time,
            error=error_message,
        )


# Factory function for compatibility with existing code
def create_quality_tool_runner(_docker_service: Any) -> HTTPQualityRunner:
    """Factory function to create optimized HTTP quality runner.

    Args:
        _docker_service (Any): The _docker_service parameter.

    Returns:
        HTTPQualityRunner: The httpqualityrunner result."""
    logger.info("Creating optimized HTTP-based quality runner with proper session management")
    return HTTPQualityRunner()
