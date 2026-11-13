"""Module for analyzing change frequency patterns from Git history."""

from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ChangeMetrics:
    """Tracks change metrics for a file based on Git history analysis."""

    def __init__(self) -> None:
        """Initialize change metrics with default values."""
        self.change_count: int = 0
        self.unique_authors: set[str] = set()
        self.bug_fix_count: int = 0
        self.last_change_date: datetime | None = None
        self.first_change_date: datetime | None = None
        self.lines_added: int = 0
        self.lines_removed: int = 0
        self.commits: list[str] = []

    def add_commit(
        self,
        commit_hash: str,
        author: str,
        date: datetime,
        message: str,
        lines_added: int = 0,
        lines_removed: int = 0,
    ) -> None:
        """Add a commit record to the metrics.

        Args:
            commit_hash: Git commit hash
            author: Commit author name
            date: Commit date
            message: Commit message
            lines_added: Number of lines added in this commit
            lines_removed: Number of lines removed in this commit
        """
        self.change_count += 1
        self.unique_authors.add(author)
        self.commits.append(commit_hash)
        self.lines_added += lines_added
        self.lines_removed += lines_removed

        # Update dates
        if self.last_change_date is None or date > self.last_change_date:
            self.last_change_date = date
        if self.first_change_date is None or date < self.first_change_date:
            self.first_change_date = date

        # Check if it's a bug fix
        if self._is_bug_fix(message):
            self.bug_fix_count += 1

    def _is_bug_fix(self, message: str) -> bool:
        """Determine if a commit message indicates a bug fix.

        Args:
            message: Commit message to analyze

        Returns:
            True if the message indicates a bug fix
        """
        bug_keywords = ["resolve", "bug", "issue", "error", "crash", "fix"]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in bug_keywords)

    def get_age_days(self) -> int:
        """Get the age of the file in days since first commit.

        Returns:
            Number of days since the file was first committed
        """
        if self.first_change_date:
            age = datetime.now(UTC) - self.first_change_date
            return age.days
        return 0

    def get_code_churn(self) -> int:
        """Get total code churn (lines added + removed).

        Returns:
            Total number of lines changed across all commits
        """
        return self.lines_added + self.lines_removed


class GitAnalyzer:
    """Analyzes Git history to extract change frequency patterns and metrics.

    This class analyzes Git repository history to extract metrics about file
    changes, including change frequency, author patterns, bug fix frequency,
    and code churn metrics. It provides the historical context needed for
    hotspot analysis.
    """

    def __init__(self, repo_path: Path | None = None) -> None:
        """Initialize the Git analyzer.

        Args:
            repo_path: Path to the git repository to analyze
        """
        self.repo_path = repo_path
        self.metrics_cache: dict[str, ChangeMetrics] = {}
        if repo_path:
            self._analyze_repository()

    def _analyze_repository(self) -> None:
        """Analyze the git repository and populate metrics cache."""
        if not self.repo_path:
            return

        try:
            # Get git log with file changes
            cmd = [
                "git",
                "log",
                "--all",
                "--numstat",
                "--format=%H|%an|%ae|%at|%s",
                "--no-renames",
                "--no-merges",
            ]

            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )

            if result.returncode != 0:
                logger.warning("Git log failed: %s", result.stderr)
                return

            # Parse git log output
            self._parse_git_log(result.stdout)
            logger.info(f"Analyzed {len(self.metrics_cache)} files from git history")

        except subprocess.TimeoutExpired:
            logger.error("Git analysis timed out")
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            logger.error("Failed to analyze git history: %s", e)

    def _parse_git_log(self, git_log_output: str) -> None:
        """Parse git log output and populate metrics cache.

        Args:
            git_log_output: Raw output from git log command
        """
        lines = git_log_output.strip().split("\n")
        current_commit = None
        current_author = None
        current_date = None
        current_message = None

        for line in lines:
            if not line:
                continue

            if "|" in line and not line.startswith("\t"):
                # Commit info line
                parts = line.split("|", 4)
                if len(parts) >= 5:
                    current_commit = parts[0]
                    current_author = parts[1]
                    current_date = datetime.fromtimestamp(int(parts[3]), tz=UTC)
                    current_message = parts[4]
            elif line[0].isdigit() or line.startswith("-"):
                # File change line (numstat format)
                self._process_file_change(
                    line, current_commit, current_author, current_date, current_message
                )

    def _process_file_change(
        self,
        line: str,
        commit_hash: str | None,
        author: str | None,
        date: datetime | None,
        message: str | None,
    ) -> None:
        """Process a single file change from git log.

        Args:
            line: Numstat line showing file changes
            commit_hash: Current commit hash
            author: Commit author
            date: Commit date
            message: Commit message
        """
        parts = line.split("\t")
        if len(parts) < 3:
            return

        added_str, removed_str, file_path = parts[0], parts[1], parts[2]

        # Initialize metrics for new files
        if file_path not in self.metrics_cache:
            self.metrics_cache[file_path] = ChangeMetrics()

        # Parse line counts
        lines_added = 0 if added_str == "-" else int(added_str)
        lines_removed = 0 if removed_str == "-" else int(removed_str)

        # Add commit to metrics
        if (
            commit_hash is not None
            and author is not None
            and date is not None
            and message is not None
        ):
            self.metrics_cache[file_path].add_commit(
                commit_hash, author, date, message, lines_added, lines_removed
            )

    def get_change_frequency(self, file_path: str) -> int:
        """Get the number of times a file has been changed.

        Args:
            file_path: Path to the file relative to repository root

        Returns:
            Integer count of changes made to the file
        """
        if file_path in self.metrics_cache:
            return self.metrics_cache[file_path].change_count
        return 0

    def get_unique_authors(self, file_path: str) -> int:
        """Get the number of unique authors who have modified a file.

        Args:
            file_path: Path to the file relative to repository root

        Returns:
            Integer count of unique authors who modified the file
        """
        if file_path in self.metrics_cache:
            return len(self.metrics_cache[file_path].unique_authors)
        return 0

    def get_bug_fix_count(self, file_path: str) -> int:
        """Get the number of bug fixes associated with a file.

        Args:
            file_path: Path to the file relative to repository root

        Returns:
            Integer count of bug fix commits affecting the file
        """
        if file_path in self.metrics_cache:
            return self.metrics_cache[file_path].bug_fix_count
        return 0

    def get_code_churn(self, file_path: str) -> int:
        """Get total code churn (lines added + removed) for a file.

        Args:
            file_path: Path to the file relative to repository root

        Returns:
            Integer representing total lines added and removed
        """
        if file_path in self.metrics_cache:
            return self.metrics_cache[file_path].get_code_churn()
        return 0

    def get_age_days(self, file_path: str) -> int:
        """Get the age of a file in days.

        Args:
            file_path: Path to the file relative to repository root

        Returns:
            Integer representing the file's age in days
        """
        if file_path in self.metrics_cache:
            return self.metrics_cache[file_path].get_age_days()
        return 0

    def get_recency_score(self, file_path: str) -> float:
        """Get recency score (0-1) based on last modification date.

        Calculates a score representing how recently the file was modified,
        using an exponential decay function where more recent changes receive
        higher scores (closer to 1.0).

        Args:
            file_path: Path to the file relative to repository root

        Returns:
            Float between 0.0 and 1.0 representing recency (1.0 = very recent)
        """
        if file_path in self.metrics_cache:
            metrics = self.metrics_cache[file_path]
            if metrics.last_change_date:
                days_since_change = (datetime.now(UTC) - metrics.last_change_date).days
                # Exponential decay: more recent changes get higher scores
                return max(0.0, min(1.0, np.exp(-0.05 * days_since_change)))
        return 0.0

    def get_author_list(self, file_path: str) -> list[str]:
        """Get list of authors who have modified a file.

        Args:
            file_path: Path to the file relative to repository root

        Returns:
            List of author names who have committed changes to the file
        """
        if file_path in self.metrics_cache:
            return list(self.metrics_cache[file_path].unique_authors)
        return []

    def get_commit_list(self, file_path: str) -> list[str]:
        """Get list of commit hashes that modified a file.

        Args:
            file_path: Path to the file relative to repository root

        Returns:
            List of commit hashes that affected the file
        """
        if file_path in self.metrics_cache:
            return self.metrics_cache[file_path].commits.copy()
        return []

    def get_metrics(self, file_path: str) -> ChangeMetrics | None:
        """Get complete change metrics for a file.

        Args:
            file_path: Path to the file relative to repository root

        Returns:
            ChangeMetrics object with all metrics, or None if file not found
        """
        return self.metrics_cache.get(file_path)

    def get_files_with_changes(self) -> list[str]:
        """Get list of all files that have changes in the repository.

        Returns:
            List of file paths that have been modified according to Git history
        """
        return list(self.metrics_cache.keys())

    def clear_cache(self) -> None:
        """Clear the metrics cache."""
        self.metrics_cache.clear()

    def get_cache_size(self) -> int:
        """Get the number of files in the metrics cache.

        Returns:
            Number of files currently cached
        """
        return len(self.metrics_cache)
