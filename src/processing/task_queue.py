"""
Task queue management for background processing.
Clean, focused implementation for task queue operations with updated Redis handling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any

from app.core.config.core_config_settings import get_settings
from app.core.redis_service import RedisService

settings = get_settings()
logger = logging.getLogger(__name__)


@dataclass
class TaskQueueConfig:
    """Configuration for task queue operations."""

    max_retries: int = 3
    retry_delay: float = 1.0
    connection_timeout: float = 10.0
    operation_timeout: float = 5.0
    namespace: str = "code_graph:"


class ProcessingTaskQueue:
    """Clean task queue implementation with Redis backend and updated error handling."""

    def __init__(self, redis_service: RedisService, config: TaskQueueConfig | None = None):
        """Initialize task queue with Redis service."""
        self.redis_service = redis_service
        self.config = config or TaskQueueConfig()

        # Use namespace from config or settings
        namespace = getattr(settings, "TASK_QUEUE_NAMESPACE", self.config.namespace)
        self.keys = {
            "queue": f"{namespace}task_queue",
            "processing": f"{namespace}processing",
            "results_prefix": f"{namespace}task_result:",
        }

        # Fallback for when Redis is unavailable
        self.local_queue: asyncio.Queue = asyncio.Queue()
        self.local_results: dict[str, Any] = {}

    async def _get_redis_client_with_retry(self):
        """Get Redis client with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                if self.redis_service:
                    client = await asyncio.wait_for(
                        self.redis_service.client,  # type: ignore[invalid-argument-type]  # ty: ignore[invalid-argument-type]
                        timeout=self.config.connection_timeout,
                    )
                    if client:
                        # Test connection
                        await asyncio.wait_for(client.ping(), timeout=2.0)
                        return client
            except (TimeoutError, ConnectionError, OSError) as e:
                if attempt < self.config.max_retries - 1:
                    logger.warning(
                        "Redis connection attempt %d failed: %s. Retrying in %ss...",
                        attempt + 1,
                        e,
                        self.config.retry_delay,
                    )
                    await asyncio.sleep(
                        self.config.retry_delay * (attempt + 1)
                    )  # Exponential backoff
                else:
                    logger.error(
                        "Redis connection failed after %d attempts: %s",
                        self.config.max_retries,
                        e,
                    )
        return None

    async def add_task(self, task: dict[str, Any]) -> bool:
        """Add a task to the queue."""

        try:
            client = await self._get_redis_client_with_retry()
            if client:
                # Store task data
                task_key = f"{self.config.namespace}task:{task['id']}"
                await asyncio.wait_for(
                    client.set(task_key, json.dumps(task), ex=86400),
                    timeout=self.config.operation_timeout,
                )

                # Add to queue
                await asyncio.wait_for(
                    client.rpush(self.keys["queue"], json.dumps(task)),
                    timeout=self.config.operation_timeout,
                )
                return True

            # Fallback to local queue
            await self.local_queue.put(task)
            logger.info("Added task %s to local queue (Redis unavailable)", task["id"])
            return True

        except (TimeoutError, ConnectionError, OSError, JSONDecodeError) as e:
            logger.error("Error adding task to queue: %s", e)
            # Fallback to local queue
            try:
                await self.local_queue.put(task)
                return True
            except (asyncio.QueueFull, RuntimeError) as fallback_error:
                logger.error("Fallback queue also failed: %s", fallback_error)
                return False

    async def get_next_task(self, timeout: int = 30) -> dict[str, Any] | None:
        """Get the next task from the queue with updated timeout handling."""

        # Try Redis first
        task = await self._get_task_from_redis(timeout)
        if task is not None:
            return task

        # Fallback to local queue
        return await self._get_task_from_local_queue(min(timeout, 5))

    async def _get_task_from_redis(self, timeout: int) -> dict[str, Any] | None:
        """Get task from Redis with error handling."""
        try:
            client = await self._get_redis_client_with_retry()
            if not client:
                return None

            # Use longer timeout for blpop to reduce error frequency
            result = await asyncio.wait_for(
                client.blpop(self.keys["queue"], timeout=timeout),
                timeout=timeout + 5,  # Add buffer to asyncio timeout
            )
            if not result:
                return None

            _, task_json = result
            task = json.loads(task_json)

            # Mark as processing
            await asyncio.wait_for(
                client.hset(
                    self.keys["processing"],
                    task["id"],
                    json.dumps({"started_at": time.time()}),
                ),
                timeout=self.config.operation_timeout,
            )
            return task if isinstance(task, dict) else None

        except TimeoutError:
            # This is normal - no tasks available
            return None
        except (ConnectionError, OSError, JSONDecodeError) as e:
            logger.error("Error getting next task: %s", e)
            return None

    async def _get_task_from_local_queue(self, timeout: int) -> dict[str, Any] | None:
        """Get task from local queue with error handling."""
        try:
            return await asyncio.wait_for(self.local_queue.get(), timeout=timeout)
        except (TimeoutError, asyncio.QueueEmpty, RuntimeError):
            return None

    async def update_task_status(self, task_id: str, status: str) -> bool:
        """Update task status."""
        try:
            status_data = {
                "status": status,
                "updated_at": time.time(),
            }

            if self.redis_service:
                client = self.redis_service.client
                if client:
                    status_key = f"{self.config.namespace}task:{task_id}:status"
                    await client.set(
                        status_key, json.dumps(status_data)
                    )  # ty resolves root sync RedisService instead of async
                    return True

            # Fallback to local storage
            self.local_results[f"{task_id}:status"] = status_data
            return True

        except (ConnectionError, OSError, JSONDecodeError) as e:
            logger.error("Error updating task status: %s", e)
            return False

    async def complete_task(self, task_id: str, result: dict[str, Any]) -> bool:
        """Mark task as completed and store result."""
        try:
            if self.redis_service:
                client = self.redis_service.client
                if client:
                    # Store result
                    result_key = f"{self.keys['results_prefix']}{task_id}"
                    await client.set(result_key, json.dumps(result), ex=86400)
                    # Update status
                    await self.update_task_status(task_id, "completed")

                    # Remove from processing
                    hdel_result = client.hdel(self.keys["processing"], task_id)
                    if asyncio.iscoroutine(hdel_result):
                        await hdel_result
                    return True

            # Fallback to local storage
            self.local_results[f"{self.keys['results_prefix']}{task_id}"] = result
            await self.update_task_status(task_id, "completed")
            return True

        except (ConnectionError, OSError, JSONDecodeError) as e:
            logger.error("Error completing task: %s", e)
            return False

    async def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed with error message."""
        try:
            error_result = {
                "error": error,
                "failed_at": time.time(),
            }

            if self.redis_service:
                client = self.redis_service.client
                if client:
                    # Store error result
                    result_key = f"{self.keys['results_prefix']}{task_id}"
                    await client.set(result_key, json.dumps(error_result), ex=86400)
                    # Update status
                    await self.update_task_status(task_id, "failed")

                    # Remove from processing
                    hdel_result = client.hdel(self.keys["processing"], task_id)
                    if asyncio.iscoroutine(hdel_result):
                        await hdel_result
                    return True

            # Fallback to local storage
            self.local_results[f"{self.keys['results_prefix']}{task_id}"] = error_result
            await self.update_task_status(task_id, "failed")
            return True

        except (ConnectionError, OSError, JSONDecodeError) as e:
            logger.error("Error failing task: %s", e)
            return False

    async def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Get task status."""
        try:
            if self.redis_service:
                client = self.redis_service.client
                if client:
                    status_key = f"{self.config.namespace}task:{task_id}:status"
                    status_data = await client.get(status_key)
                    if status_data:
                        parsed_data = json.loads(status_data)
                        return parsed_data if isinstance(parsed_data, dict) else None

            # Fallback to local storage
            return self.local_results.get(f"{task_id}:status")

        except (ConnectionError, OSError, JSONDecodeError) as e:
            logger.error("Error getting task status: %s", e)
            return None

    async def get_task_result(self, task_id: str) -> dict[str, Any] | None:
        """Get task result."""
        try:
            if self.redis_service:
                client = self.redis_service.client
                if client:
                    result_key = f"{self.keys['results_prefix']}{task_id}"
                    result_data = await client.get(result_key)
                    if result_data:
                        parsed_data = json.loads(result_data)
                        return parsed_data if isinstance(parsed_data, dict) else None

            # Fallback to local storage
            return self.local_results.get(f"{self.keys['results_prefix']}{task_id}")

        except (ConnectionError, OSError, JSONDecodeError) as e:
            logger.error("Error getting task result: %s", e)
            return None

    async def get_all_tasks(self) -> list[dict[str, Any]]:
        """Get all tasks."""
        try:
            if self.redis_service:
                client = self.redis_service.client
                if client:
                    keys = await client.keys(f"{self.config.namespace}task:*")
                    return [{"key": str(key)} for key in keys] if keys else []

                # Return empty list if client connection fails
                return []
        except (ConnectionError, OSError) as e:
            logger.error("Error getting all tasks: %s", e)
            return []

        # Return empty list if Redis service is unavailable
        return []
