"""Change Detection and Tracking Module for Incremental Updates.

This module provides functionality for detecting and tracking file changes
in a repository. It handles change detection through checksums and timestamps,
tracking file modifications, additions, and deletions.

The module supports:
- File checksum-based change tracking
- Modification time tracking for performance
- File categorization (added, modified, deleted)
- Persistent checksum caching
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ChangeDetector:
    """Handles change detection for incremental updates."""

    def __init__(self):
        """Initialize the change detector."""
        self.checksums_cache: dict[str, str] = {}
        self.last_update_time: dict[str, float] = {}

    async def detect_changes(self, repo_path: Path) -> dict[str, list[Path]]:
        """Detect changes in the repository since last update.

        Args:
            repo_path: Path to the repository

        Returns:
            Dictionary with 'added', 'modified', and 'deleted' files
        """
        changes: dict[str, list[Path]] = {
            "added": [],
            "modified": [],
            "deleted": [],
        }

        # Load previous file checksums if available
        await self._load_checksums(repo_path)

        # Get all current Python files
        current_files = await self._scan_current_files(repo_path, changes)

        # Check for deleted files
        await self._detect_deleted_files(repo_path, current_files, changes)

        return changes

    async def _load_checksums(self, repo_path: Path) -> None:
        """Load previous file checksums from cache file."""
        checksums_file = repo_path / ".code_graph" / "checksums.json"
        if checksums_file.exists():
            try:
                with open(checksums_file, encoding="utf-8") as f:
                    self.checksums_cache = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Failed to load checksums file: %s", e)
                self.checksums_cache = {}

    async def _scan_current_files(
        self, repo_path: Path, changes: dict[str, list[Path]]
    ) -> set[str]:
        """Scan current files and detect added/modified files.

        Args:
            repo_path: Path to the repository
            changes: Dictionary to update with detected changes

        Returns:
            Set of current file paths (relative to repo)
        """
        current_files = set()

        for root, _, files in os.walk(repo_path):
            # Skip hidden directories and virtual environments
            if self._should_skip_directory(Path(root)):
                continue

            for file in files:
                if self._is_trackable_file(file):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(repo_path)
                    current_files.add(str(rel_path))

                    await self._check_file_status(file_path, rel_path, changes)

        return current_files

    def _should_skip_directory(self, dir_path: Path) -> bool:
        """Check if directory should be skipped during scanning."""
        # Skip hidden directories and virtual environments
        return any(part.startswith(".") for part in dir_path.parts) or any(
            part in ["venv", "env", ".venv", ".env", "__pycache__"] for part in dir_path.parts
        )

    def _is_trackable_file(self, filename: str) -> bool:
        """Check if file should be tracked for changes."""
        return filename.endswith(".py")

    async def _check_file_status(
        self, file_path: Path, rel_path: Path, changes: dict[str, list[Path]]
    ) -> None:
        """Check if file is new or modified and update changes accordingly."""
        rel_path_str = str(rel_path)

        # Check if file is new
        if rel_path_str not in self.checksums_cache:
            changes["added"].append(file_path)
            # Calculate and cache checksum for new file
            self.checksums_cache[rel_path_str] = self._calculate_file_checksum(file_path)
        else:
            # Check if file was modified
            await self._check_file_modification(file_path, rel_path_str, changes)

    async def _check_file_modification(
        self, file_path: Path, rel_path_str: str, changes: dict[str, list[Path]]
    ) -> None:
        """Check if an existing file has been modified."""
        # Check modification time first (faster)
        mtime = file_path.stat().st_mtime

        # If we have a cached update time and file is newer, check checksum
        if rel_path_str in self.last_update_time and mtime > self.last_update_time[rel_path_str]:
            # Verify with checksum (more accurate)
            current_checksum = self._calculate_file_checksum(file_path)
            if current_checksum != self.checksums_cache[rel_path_str]:
                changes["modified"].append(file_path)
                self.checksums_cache[rel_path_str] = current_checksum

            self.last_update_time[rel_path_str] = mtime
        else:
            # File hasn't been modified since last update based on mtime
            self.last_update_time[rel_path_str] = mtime

    async def _detect_deleted_files(
        self, repo_path: Path, current_files: set[str], changes: dict[str, list[Path]]
    ) -> None:
        """Detect files that have been deleted since last scan."""
        for rel_path_str in self.checksums_cache:
            if rel_path_str not in current_files:
                # Convert string path to Path object to match expected type
                changes["deleted"].append(repo_path / Path(rel_path_str))

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file.

        Args:
            file_path: Path to the file

        Returns:
            MD5 checksum as hex string
        """
        hash_md5 = hashlib.md5()  # nosec B324
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Failed to calculate checksum for %s: %s", file_path, e)
            # Return a timestamp-based fallback checksum
            return f"fallback_{int(time.time())}"

        return hash_md5.hexdigest()

    async def save_checksums(self, repo_path: Path) -> None:
        """Save current checksums to cache file.

        Args:
            repo_path: Path to the repository
        """
        checksums_dir = repo_path / ".code_graph"
        checksums_dir.mkdir(exist_ok=True)

        try:
            with open(checksums_dir / "checksums.json", "w", encoding="utf-8") as f:
                json.dump(self.checksums_cache, f, indent=2)
        except (OSError, TypeError) as e:
            logger.error("Failed to save checksums file: %s", e)

    def update_file_tracking(self, file_path: Path, repo_path: Path) -> None:
        """Update tracking information for a specific file.

        Args:
            file_path: Path to the file
            repo_path: Path to the repository root
        """
        try:
            rel_path = str(file_path.relative_to(repo_path))
            self.checksums_cache[rel_path] = self._calculate_file_checksum(file_path)
            self.last_update_time[rel_path] = file_path.stat().st_mtime
        except (ValueError, OSError) as e:
            logger.warning("Failed to update tracking for %s: %s", file_path, e)

    def get_file_status(self, file_path: Path, repo_path: Path) -> str:
        """Get the current status of a file.

        Args:
            file_path: Path to the file
            repo_path: Path to the repository root

        Returns:
            Status string: 'new', 'modified', 'unchanged', or 'unknown'
        """
        try:
            rel_path = str(file_path.relative_to(repo_path))

            if rel_path not in self.checksums_cache:
                return "new"

            current_checksum = self._calculate_file_checksum(file_path)
            cached_checksum = self.checksums_cache[rel_path]

            if current_checksum != cached_checksum:
                return "modified"
            else:
                return "unchanged"

        except (ValueError, OSError) as e:
            logger.warning("Failed to get status for %s: %s", file_path, e)
            return "unknown"

    def get_change_statistics(self, changes: dict[str, list[Path]]) -> dict[str, Any]:
        """Get statistics about detected changes.

        Args:
            changes: Dictionary with detected changes

        Returns:
            Dictionary with change statistics
        """
        total_files = sum(len(file_list) for file_list in changes.values())

        return {
            "total_changes": total_files,
            "added_files": len(changes["added"]),
            "modified_files": len(changes["modified"]),
            "deleted_files": len(changes["deleted"]),
            "change_breakdown": {
                "additions": [str(f) for f in changes["added"]],
                "modifications": [str(f) for f in changes["modified"]],
                "deletions": [str(f) for f in changes["deleted"]],
            },
            "detected_at": datetime.now(UTC).isoformat(),
        }

    def clear_cache(self) -> None:
        """Clear the internal cache of checksums and update times."""
        self.checksums_cache.clear()
        self.last_update_time.clear()
        logger.info("Change detector cache cleared")
