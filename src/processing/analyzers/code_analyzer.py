"""
Code Analysis for Documentation Module
Handles project structure analysis, dependency extraction, and change analysis.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """Handles code analysis for documentation generation."""

    def __init__(self):
        """Initialize the code analyzer."""
        pass

    def analyze_project_structure(self, project_path: str) -> dict[str, Any]:
        """Analyze project structure for documentation."""
        structure: dict[str, Any] = {
            "directories": [],
            "files": [],
            "languages": set(),
            "size_metrics": {},
            "file_types": {},
        }

        try:
            file_type_counts = {}
            total_size = 0

            for root, dirs, files in os.walk(project_path):
                # Skip hidden directories and common ignore patterns
                dirs[:] = [
                    d
                    for d in dirs
                    if not d.startswith(".")
                    and d not in ["__pycache__", "node_modules", "venv", "env"]
                ]

                rel_root = os.path.relpath(root, project_path)
                if rel_root != ".":
                    structure["directories"].append(rel_root)

                for file in files:
                    if file.startswith("."):
                        continue

                    file_path = os.path.join(rel_root, file) if rel_root != "." else file
                    structure["files"].append(file_path)

                    # Get file size
                    try:
                        full_path = os.path.join(root, file)
                        size = os.path.getsize(full_path)
                        total_size += size
                    except OSError:
                        continue

                    # Detect language and count file types
                    ext = Path(file).suffix.lower()
                    file_type_counts[ext] = file_type_counts.get(ext, 0) + 1

                    if ext == ".py":
                        structure["languages"].add("Python")
                    elif ext == ".js":
                        structure["languages"].add("JavaScript")
                    elif ext == ".ts":
                        structure["languages"].add("TypeScript")
                    elif ext == ".java":
                        structure["languages"].add("Java")
                    elif ext in [".c", ".h"]:
                        structure["languages"].add("C")
                    elif ext in [".cpp", ".cc", ".cxx", ".hpp"]:
                        structure["languages"].add("C++")
                    elif ext == ".go":
                        structure["languages"].add("Go")
                    elif ext == ".rs":
                        structure["languages"].add("Rust")
                    elif ext in [".md", ".rst"]:
                        structure["languages"].add("Documentation")

            structure["languages"] = list(structure["languages"])
            structure["file_types"] = file_type_counts
            structure["size_metrics"] = {
                "total_files": len(structure["files"]),
                "total_directories": len(structure["directories"]),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            }

        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Error analyzing project structure: %s", e)

        return structure

    async def analyze_recent_changes(self, repository_path: str) -> dict[str, Any]:
        """Analyze recent changes using git integration."""
        try:
            # Check if it's a git repository
            git_dir = Path(repository_path) / ".git"
            if not git_dir.exists():
                return {
                    "error": "Not a git repository",
                    "recent_commits": [],
                    "files_changed": [],
                    "impact_analysis": "Unable to analyze - not a git repository",
                }

            # Get recent commits (last 30 days or 50 commits, whichever is fewer)
            since_date = (datetime.now(UTC) - timedelta(days=30)).strftime("%Y-%m-%d")

            # Get commit information
            commit_cmd = [
                "git",
                "-C",
                repository_path,
                "log",
                "--since",
                since_date,
                "--max-count",
                "50",
                "--pretty=format:%H|%an|%ae|%ad|%s",
                "--date=iso",
            ]

            try:
                commit_output = subprocess.run(
                    commit_cmd, capture_output=True, text=True, check=True
                ).stdout

                recent_commits = []
                for line in commit_output.strip().split("\n"):
                    if line:
                        parts = line.split("|")
                        if len(parts) >= 5:
                            recent_commits.append(
                                {
                                    "hash": parts[0][:7],  # Short hash
                                    "author": parts[1],
                                    "email": parts[2],
                                    "date": parts[3],
                                    "message": parts[4],
                                }
                            )

                # Get files changed in recent commits
                files_cmd = (
                    [
                        "git",
                        "-C",
                        repository_path,
                        "diff",
                        "--name-only",
                        f"HEAD~{min(len(recent_commits), 10)}..HEAD",
                    ]
                    if recent_commits
                    else []
                )

                files_changed = []
                if files_cmd:
                    try:
                        files_output = subprocess.run(
                            files_cmd,
                            capture_output=True,
                            text=True,
                            check=True,
                        ).stdout
                        files_changed = [f for f in files_output.strip().split("\n") if f]
                    except subprocess.CalledProcessError:
                        # Fallback to showing files from the last commit
                        files_cmd = [
                            "git",
                            "-C",
                            repository_path,
                            "diff-tree",
                            "--no-commit-id",
                            "--name-only",
                            "-r",
                            "HEAD",
                        ]
                        try:
                            files_output = subprocess.run(
                                files_cmd, capture_output=True, text=True, check=False
                            ).stdout
                            files_changed = [f for f in files_output.strip().split("\n") if f]
                        except (subprocess.CalledProcessError, OSError):
                            files_changed = []

                # Analyze impact based on files changed
                impact_analysis = self._assess_change_impact(files_changed, recent_commits)

                # Get change statistics
                stats_cmd = (
                    [
                        "git",
                        "-C",
                        repository_path,
                        "diff",
                        "--shortstat",
                        f"HEAD~{min(len(recent_commits), 10)}..HEAD",
                    ]
                    if recent_commits
                    else []
                )

                change_stats = {}
                if stats_cmd:
                    try:
                        stats_output = subprocess.run(
                            stats_cmd, capture_output=True, text=True, check=False
                        ).stdout.strip()

                        # Parse stats like "10 files changed, 250 insertions(+), 100 deletions(-)"
                        files_match = re.search(r"(\d+) files? changed", stats_output)
                        insertions_match = re.search(r"(\d+) insertions?\(\+\)", stats_output)
                        deletions_match = re.search(r"(\d+) deletions?\(-\)", stats_output)

                        change_stats = {
                            "files_changed": (int(files_match.group(1)) if files_match else 0),
                            "insertions": (
                                int(insertions_match.group(1)) if insertions_match else 0
                            ),
                            "deletions": (int(deletions_match.group(1)) if deletions_match else 0),
                        }
                    except (subprocess.CalledProcessError, OSError, ValueError):
                        change_stats = {}

                # Get branch information
                branch_cmd = [
                    "git",
                    "-C",
                    repository_path,
                    "branch",
                    "--show-current",
                ]
                try:
                    current_branch = subprocess.run(
                        branch_cmd, capture_output=True, text=True, check=False
                    ).stdout.strip()
                except (subprocess.CalledProcessError, OSError):
                    current_branch = "unknown"

                return {
                    "recent_commits": recent_commits[:10],  # Limit to 10 most recent
                    "files_changed": files_changed,
                    "impact_analysis": impact_analysis,
                    "change_statistics": change_stats,
                    "current_branch": current_branch,
                    "analysis_period": f"Last 30 days or {len(recent_commits)} commits",
                }

            except subprocess.CalledProcessError as e:
                logger.warning("Git command failed: %s", e)
                return {
                    "error": f"Git analysis failed: {str(e)}",
                    "recent_commits": [],
                    "files_changed": [],
                    "impact_analysis": "Unable to analyze git history",
                }

        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Error analyzing recent changes: %s", e)
            return {
                "error": str(e),
                "recent_commits": [],
                "files_changed": [],
                "impact_analysis": "Error occurred during analysis",
            }

    async def extract_python_dependencies(self, repository_path: str) -> dict[str, Any]:
        """Extract Python dependencies from various sources."""
        dependencies = {}

        try:
            # Check pyproject.toml
            pyproject_path = Path(repository_path) / "pyproject.toml"
            if pyproject_path.exists():
                try:
                    import toml

                    with open(pyproject_path, encoding="utf-8") as f:
                        pyproject = toml.load(f)

                    # Extract dependencies
                    if "tool" in pyproject and "poetry" in pyproject["tool"]:
                        poetry_deps = pyproject["tool"]["poetry"].get("dependencies", {})
                        for dep, version in poetry_deps.items():
                            if dep != "python":
                                dependencies[dep] = {
                                    "version": version,
                                    "dev": False,
                                }

                        dev_deps = pyproject["tool"]["poetry"].get("dev-dependencies", {})
                        for dep, version in dev_deps.items():
                            dependencies[dep] = {"version": version, "dev": True}

                    elif "project" in pyproject:
                        # PEP 621 format
                        proj_deps = pyproject["project"].get("dependencies", [])
                        for dep in proj_deps:
                            name = dep.split("==")[0].split(">=")[0].split("<")[0].strip()
                            dependencies[name] = {"version": dep, "dev": False}
                except ImportError:
                    logger.warning("toml package not available for pyproject.toml parsing")

            # Check requirements.txt
            req_path = Path(repository_path) / "requirements.txt"
            if req_path.exists() and not dependencies:
                with open(req_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            name = line.split("==")[0].split(">=")[0].split("<")[0].strip()
                            dependencies[name] = {
                                "version": line,
                                "dev": False,
                            }

            # Check setup.py
            setup_path = Path(repository_path) / "setup.py"
            if setup_path.exists() and not dependencies:
                # Basic regex-based extraction
                with open(setup_path, encoding="utf-8") as f:
                    content = f.read()

                # Find install_requires
                install_match = re.search(r"install_requires\s*=\s*\[(.*?)\]", content, re.DOTALL)
                if install_match:
                    deps_str = install_match.group(1)
                    for dep in re.findall(r'["\']([^"\']+)["\']', deps_str):
                        name = dep.split("==")[0].split(">=")[0].split("<")[0].strip()
                        dependencies[name] = {"version": dep, "dev": False}

        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Error extracting Python dependencies: %s", e)

        return dependencies

    async def extract_javascript_dependencies(self, repository_path: str) -> dict[str, Any]:
        """Extract JavaScript/TypeScript dependencies from package.json."""
        dependencies = {}

        try:
            package_path = Path(repository_path) / "package.json"
            if package_path.exists():
                with open(package_path, encoding="utf-8") as f:
                    package_data = json.load(f)

                # Regular dependencies
                for dep, version in package_data.get("dependencies", {}).items():
                    dependencies[dep] = {"version": version, "dev": False}

                # Dev dependencies
                for dep, version in package_data.get("devDependencies", {}).items():
                    dependencies[dep] = {"version": version, "dev": True}

                # Optional dependencies
                for dep, version in package_data.get("optionalDependencies", {}).items():
                    dependencies[dep] = {"version": version, "optional": True}

        except (OSError, ValueError, RuntimeError) as e:
            logger.warning("Error extracting JavaScript dependencies: %s", e)

        return dependencies

    async def generate_project_documentation_data(self, project_path: str) -> dict[str, Any]:
        """Generate comprehensive project documentation data."""
        try:
            # Analyze project structure
            structure = self.analyze_project_structure(project_path)

            # Extract documentation from key files
            documentation = {}

            # Find and process documentation files
            doc_files = list(Path(project_path).rglob("*.md")) + list(
                Path(project_path).rglob("*.rst")
            )

            # Limit to prevent excessive processing
            for doc_file in doc_files[:20]:
                try:
                    with open(doc_file, encoding="utf-8") as f:
                        content = f.read()
                    documentation[str(doc_file.relative_to(project_path))] = {
                        "content": content,
                        "size": len(content),
                        "word_count": len(content.split()),
                    }
                except (OSError, UnicodeDecodeError):  # fmt: skip
                    continue

            # Analyze dependencies
            python_deps = await self.extract_python_dependencies(project_path)
            js_deps = await self.extract_javascript_dependencies(project_path)

            dependencies = {}
            if python_deps:
                dependencies["python"] = python_deps
            if js_deps:
                dependencies["javascript"] = js_deps

            return {
                "project_structure": structure,
                "documentation_files": documentation,
                "dependencies": dependencies,
                "analysis_timestamp": datetime.now(UTC).isoformat(),
            }

        except (OSError, ValueError, RuntimeError) as e:
            logger.exception("Error generating project documentation data: %s", e)
            return {"error": str(e), "project_path": project_path}

    def _assess_change_impact(self, files_changed: list[str], commits: list[dict[str, Any]]) -> str:
        """Assess the impact level of recent changes."""
        if not files_changed:
            return "No changes detected"

        # Analyze file types and locations
        critical_files = []
        test_files = []
        doc_files = []
        config_files = []
        source_files = []

        for file in files_changed:
            file_lower = file.lower()
            if "test" in file_lower or file_lower.endswith("_test.py"):
                test_files.append(file)
            elif file_lower.endswith((".md", ".rst", ".txt")):
                doc_files.append(file)
            elif file_lower in [
                "setup.py",
                "pyproject.toml",
                "package.json",
                "requirements.txt",
                "Dockerfile",
                "docker-compose.yml",
                ".github/workflows",
            ]:
                config_files.append(file)
                critical_files.append(file)
            elif any(
                critical in file_lower for critical in ["auth", "security", "database", "api"]
            ):
                critical_files.append(file)
                source_files.append(file)
            else:
                source_files.append(file)

        # Determine impact level
        impact_parts = []

        if critical_files:
            impact_parts.append(f"HIGH RISK - {len(critical_files)} critical files modified")

        if len(source_files) > 10:
            impact_parts.append(f"Large change scope - {len(source_files)} source files")
        elif source_files:
            impact_parts.append(f"Moderate scope - {len(source_files)} source files")

        if test_files:
            impact_parts.append(f"{len(test_files)} test files updated")

        if config_files:
            impact_parts.append(f"{len(config_files)} configuration files changed")

        if doc_files:
            impact_parts.append(f"{len(doc_files)} documentation files updated")

        # Check commit messages for risk indicators
        risk_keywords = [
            "breaking",
            "major",
            "security",
            "critical",
            "urgent",
            "resolve",
            "bug",
        ]
        risky_commits = [
            commit
            for commit in commits
            if any(keyword in commit.get("message", "").lower() for keyword in risk_keywords)
        ]

        if risky_commits:
            impact_parts.append(f"{len(risky_commits)} commits with risk indicators")

        if not impact_parts:
            return "Low risk - minor changes"

        return " | ".join(impact_parts)
