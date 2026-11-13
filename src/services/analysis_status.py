"""
Analysis Status Tracking Service

Provides real-time status tracking for analysis operations using Redis.
Integrates with the existing task queue system for comprehensive monitoring.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

import redis.exceptions as redis_exceptions

from app.api.models.analysis_api_models import (
    AnalysisPhase,
    AnalysisStatus,
    AnalysisStatusInfo,
)
from app.core.redis_service import RedisService, get_redis_service

logger = logging.getLogger(__name__)


class AnalysisStatusService:
    """Service for tracking analysis operation status and progress."""

    def __init__(self) -> None:
        """Initialize the AnalysisStatusService service.

        This initializes the AnalysisStatusService with required dependencies
        and configuration for service operations.
        """
        self.redis_service: RedisService | None = None
        self.status_prefix = "analysis_status:"
        self.default_ttl = 3600  # 1 hour
        # Fallback in-memory storage when Redis is unavailable
        self._memory_store: dict[str, AnalysisStatusInfo] = {}

    async def _get_redis_service(self) -> Any:
        """Lazy initialization of Redis service."""
        if self.redis_service is None:
            self.redis_service = get_redis_service()
        return self.redis_service

    async def create_analysis_session(
        self,
        analysis_id: str,
        analysis_type: str,
        estimated_duration: float | None = None,
    ) -> AnalysisStatusInfo:
        """Create a new analysis session with initial status."""
        try:
            now = datetime.now(UTC)

            # Estimate completion time based on analysis type
            if not estimated_duration:
                type_durations = {
                    "basic": 30.0,
                    "augmented": 60.0,
                    "ast": 45.0,
                    "deep": 120.0,
                    "security": 90.0,
                    "performance": 100.0,
                }
                estimated_duration = type_durations.get(analysis_type, 60.0)

            estimated_completion = datetime.fromtimestamp(
                now.timestamp() + estimated_duration, tz=UTC
            )

            status_info = AnalysisStatusInfo(
                analysis_id=analysis_id,
                status=AnalysisStatus.PENDING,
                progress=0.0,
                started_at=now.isoformat(),
                completed_at=None,
                estimated_completion=estimated_completion.isoformat(),
                metadata={},
            )

            # Store in Redis
            await self._store_status(analysis_id, status_info)

            logger.info("Created analysis session %s (%s)", analysis_id, analysis_type)
            return status_info

        except Exception as e:
            logger.error("Failed to create analysis session %s: %s", analysis_id, e)
            raise

    async def update_status(
        self,
        analysis_id: str,
        status: AnalysisStatus,
        progress: float | None = None,
        **kwargs: Any,
    ) -> AnalysisStatusInfo:
        """Update analysis status and progress."""
        try:
            current_status = await self.get_status(analysis_id)
            if not current_status:
                # Auto-create session if it doesn't exist (e.g. Redis unavailable)
                current_status = AnalysisStatusInfo(
                    analysis_id=analysis_id,
                    status=AnalysisStatus.PENDING,
                    progress=0.0,
                    started_at=datetime.now(UTC).isoformat(),
                    completed_at=None,
                    estimated_completion=None,
                    metadata={},
                )

            # Update fields
            current_status.status = status
            if progress is not None:
                current_status.progress = min(100.0, max(0.0, progress))

            # Set completion time if finished
            if status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]:
                current_status.completed_at = datetime.now(UTC).isoformat()
                current_status.progress = (
                    100.0 if status == AnalysisStatus.COMPLETED else current_status.progress
                )

            # Add phase and details to metadata
            if current_status.metadata is None:
                current_status.metadata = {}

            phase = kwargs.get("phase")
            details = kwargs.get("details")

            if phase and isinstance(phase, AnalysisPhase):
                current_status.metadata["current_phase"] = phase.value
            if details and isinstance(details, dict):
                current_status.metadata.update(details)

            # Store updated status
            await self._store_status(analysis_id, current_status)

            logger.debug(
                "Updated analysis %s: %s (%.1f%%)",
                analysis_id,
                status.value,
                current_status.progress,
            )
            return current_status

        except Exception as e:
            logger.error("Failed to update status for %s: %s", analysis_id, e)
            # Return a fallback status instead of raising, so callers can proceed
            return AnalysisStatusInfo(
                analysis_id=analysis_id,
                status=status,
                progress=progress or 0.0,
                started_at=datetime.now(UTC).isoformat(),
                completed_at=None,
                estimated_completion=None,
                metadata={},
            )

    async def get_status(self, analysis_id: str) -> AnalysisStatusInfo | None:
        """Get current status for an analysis operation."""
        try:
            key = f"{self.status_prefix}{analysis_id}"
            redis_service = await self._get_redis_service()
            data = await redis_service.get(key)

            if data:
                # Handle both pre-deserialized dict and JSON string
                if isinstance(data, dict):
                    status_dict = data
                elif isinstance(data, (str, bytes)):
                    status_dict = json.loads(data)
                else:
                    logger.warning(
                        "Unexpected data type from Redis for %s: %s",
                        analysis_id,
                        type(data),
                    )
                    return None

                return AnalysisStatusInfo(**status_dict)

            # Redis failed or no data, check memory store
            if analysis_id in self._memory_store:
                logger.debug("Retrieved analysis %s from memory store", analysis_id)
                return self._memory_store[analysis_id]

            return None

        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ConnectionError,
            redis_exceptions.ConnectionError,
            redis_exceptions.RedisError,
            OSError,
        ) as e:
            logger.error("Failed to get status for %s: %s", analysis_id, e)
            # Try memory store as last resort
            return self._memory_store.get(analysis_id)

    async def get_active_analyses(self) -> list[AnalysisStatusInfo]:
        """Get all currently active analysis operations."""
        try:
            pattern = f"{self.status_prefix}*"
            redis_service = await self._get_redis_service()
            keys: list[str] = []
            try:
                keys = await redis_service.async_scan_keys(pattern)
            except Exception as exc:
                logger.warning(
                    "Redis key scan failed for active analyses (pattern %s); "
                    "falling back to memory store only: %s",
                    pattern,
                    exc,
                )

            active_analyses: list[Any] = []

            # Check Redis data
            for key in keys:
                data = await redis_service.get(key)
                if data:
                    # Handle both pre-deserialized dict and JSON string
                    if isinstance(data, dict):
                        status_dict = data
                    elif isinstance(data, (str, bytes)):
                        status_dict = json.loads(data)
                    else:
                        logger.warning(
                            "Unexpected data type from Redis for key %s: %s",
                            key,
                            type(data),
                        )
                        continue

                    status_info = AnalysisStatusInfo(**status_dict)

                    # Only include active analyses
                    if status_info.status in [
                        AnalysisStatus.PENDING,
                        AnalysisStatus.RUNNING,
                    ]:
                        active_analyses.append(status_info)

            # Also check memory store for active analyses
            for analysis_id, status_info in self._memory_store.items():
                if status_info.status in [
                    AnalysisStatus.PENDING,
                    AnalysisStatus.RUNNING,
                ]:
                    # Avoid duplicates (in case we got it from Redis too)
                    if not any(a.analysis_id == analysis_id for a in active_analyses):
                        active_analyses.append(status_info)

            return active_analyses

        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ConnectionError,
            redis_exceptions.ConnectionError,
            redis_exceptions.RedisError,
            OSError,
        ) as e:
            logger.error("Failed to get active analyses: %s", e)
            # Fallback to memory store only
            return [
                status_info
                for status_info in self._memory_store.values()
                if status_info.status in [AnalysisStatus.PENDING, AnalysisStatus.RUNNING]
            ]

    async def cleanup_completed_analyses(self, older_than_hours: int = 24) -> int:
        """Clean up completed analysis status records older than specified hours."""
        try:
            pattern = f"{self.status_prefix}*"
            redis_service = await self._get_redis_service()
            keys: list[str] = []
            try:
                keys = await redis_service.async_scan_keys(pattern)
            except Exception as exc:
                logger.warning(
                    "Redis key scan failed during cleanup (pattern %s); "
                    "no keys will be cleaned this run: %s",
                    pattern,
                    exc,
                )

            cleaned_count = 0
            cutoff_time = time.time() - (older_than_hours * 3600)

            for key in keys:
                data = await redis_service.get(key)
                if data:
                    # Handle both pre-deserialized dict and JSON string
                    if isinstance(data, dict):
                        status_dict = data
                    elif isinstance(data, (str, bytes)):
                        status_dict = json.loads(data)
                    else:
                        logger.warning(
                            "Unexpected data type from Redis for key %s: %s",
                            key,
                            type(data),
                        )
                        continue

                    status_info = AnalysisStatusInfo(**status_dict)

                    # Check if completed and old enough
                    if (
                        status_info.status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]
                        and status_info.completed_at
                    ):
                        # Handle completed_at as string
                        completed_at_str = status_info.completed_at
                        if isinstance(completed_at_str, str):
                            completed_at_str = completed_at_str.replace("Z", "+00:00")
                        completed_time = datetime.fromisoformat(completed_at_str)
                        if completed_time.timestamp() < cutoff_time:
                            await redis_service.delete(key)
                            cleaned_count += 1

            logger.info("Cleaned up %d old analysis records", cleaned_count)
            return cleaned_count

        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ConnectionError,
            redis_exceptions.ConnectionError,
            redis_exceptions.RedisError,
            OSError,
            ValueError,
        ) as e:
            logger.error("Failed to cleanup analyses: %s", e)
            return 0

    async def _store_status(self, analysis_id: str, status_info: AnalysisStatusInfo) -> None:
        """Store status information in Redis with fallback to memory."""
        key = f"{self.status_prefix}{analysis_id}"

        # Convert to dict for JSON serialization
        status_dict = status_info.model_dump()

        # Try Redis first
        try:
            redis_service = await self._get_redis_service()
            await redis_service.set(key, json.dumps(status_dict), ex=self.default_ttl)
            self._memory_store[analysis_id] = status_info
        except Exception:
            logger.debug("Redis unavailable, using memory store for analysis %s", analysis_id)
            self._memory_store[analysis_id] = status_info


# Singleton instance holder
class _AnalysisStatusServiceHolder:
    """Thread-safe singleton holder for AnalysisStatusService."""

    def __init__(self) -> None:
        """Initialize the _AnalysisStatusServiceHolder service.

        This initializes the _AnalysisStatusServiceHolder with required dependencies
        and configuration for service operations.
        """
        self._instance: AnalysisStatusService | None = None

    def get_instance(self) -> AnalysisStatusService:
        """Get or create the singleton instance."""
        if self._instance is None:
            self._instance = AnalysisStatusService()
        return self._instance

    def reset_instance(self) -> None:
        """Reset the singleton instance (useful for testing)."""
        self._instance = None


_service_holder = _AnalysisStatusServiceHolder()


def get_analysis_status_service() -> AnalysisStatusService:
    """Get global analysis status service instance."""
    return _service_holder.get_instance()
