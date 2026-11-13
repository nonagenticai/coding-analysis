"""
Similarity Configuration and Utilities

Configuration models, validation functions, and utility methods for similarity analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import ConfigDict, Field

from app.api.utils.workspace_files import (
    MaterializedFiles,
    materialize_tenant_workspace,
    materialized_files,
    release_materialized,
)
from processing.workspace_repos import RepoSelector
from services.similarity_detector import (
    SimilarityScope,
    SimilarityType,
    get_similarity_detector,
)

logger = logging.getLogger(__name__)


class SimilarityAnalysisRequest(RepoSelector):
    """Request for similarity analysis."""

    container_name: str | None = Field(
        None,
        description="Container name to analyze (analyzes all files in container)",
    )
    file_paths: list[str] | None = Field(
        None,
        description="Specific file paths within container to analyze (optional)",
    )
    scope: str = Field(
        "function",
        description="Scope of analysis: file, function, class, module, project",
    )
    min_similarity_score: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to report (0.0 - 1.0)",
    )
    detect_cross_language: bool = Field(
        True,
        description="Whether to detect similarities across different languages",
    )
    max_files: int | None = Field(
        None,
        ge=1,
        le=1000,
        description="Maximum number of files to analyze (performance limit)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "container_name": "my-app-backend",
                "file_paths": [
                    "src/utils.py",
                    "lib/helpers.java",
                    "common/tools.rs",
                ],
                "scope": "function",
                "min_similarity_score": 0.6,
                "detect_cross_language": True,
                "max_files": 100,
            },
        }
    )


class CloneDetectionRequest(RepoSelector):
    """Request for code clone detection."""

    container_name: str | None = Field(None, description="Container name to analyze for clones")
    file_paths: list[str] | None = Field(
        None, description="Specific file paths within container to analyze"
    )
    clone_threshold: float = Field(
        0.85,
        ge=0.5,
        le=1.0,
        description="Minimum similarity score to consider as a clone",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "container_name": "my-app-backend",
                "file_paths": [
                    "src/module1.py",
                    "src/module2.py",
                    "tests/test_utils.py",
                ],
                "clone_threshold": 0.85,
            },
        }
    )


class CrossLanguagePatternRequest(RepoSelector):
    """Request for cross-language pattern detection."""

    container_name: str | None = Field(None, description="Container name to analyze for patterns")
    file_paths: list[str] | None = Field(
        None, description="Specific file paths within container to analyze"
    )
    pattern_threshold: float = Field(
        0.60,
        ge=0.3,
        le=1.0,
        description="Minimum similarity score for cross-language patterns",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "container_name": "fullstack-app",
                "file_paths": [
                    "backend/auth.java",
                    "frontend/auth.ts",
                    "mobile/auth.swift",
                ],
                "pattern_threshold": 0.60,
            },
        }
    )


# Kept as the public name used by callers/tests of this module.
ResolvedFiles = MaterializedFiles


class SimilarityConfig:
    """Configuration and utility methods for similarity analysis."""

    # Supported file extensions for similarity analysis
    SUPPORTED_EXTENSIONS = [
        ".py",
        ".java",
        ".js",
        ".ts",
        ".go",
        ".rs",
        ".cs",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
    ]

    # Directories to skip during file discovery
    SKIP_DIRECTORIES = [
        "node_modules",
        "venv",
        "__pycache__",
        ".git",
        ".svn",
        "target",
        "build",
        "dist",
        ".next",
        ".nuxt",
    ]

    @staticmethod
    def validate_analysis_request(request: SimilarityAnalysisRequest) -> None:
        """Validate similarity analysis request parameters."""
        if not request.container_name and not request.file_paths:
            raise HTTPException(
                status_code=400,
                detail="Either container_name or file_paths must be provided",
            )

    @staticmethod
    async def resolve_container_files(
        container_name: str,
        file_paths: list[str] | None,
        max_files: int | None,
        *,
        tenant_id: str | None,
        repo: str | None = None,
    ) -> ResolvedFiles:
        """Resolve a container's code files to readable LOCAL paths.

        The code lives inside the tenant's workspace pod; this pod cannot see it.
        The old implementation looked the container up in a local registry (404
        for every real tenant workspace, e.g. ``cv-ems-p4``) and then walked
        ``codebase_path`` on THIS pod's disk. Now: exact, tenant-scoped lookup and
        a materialised copy of the live workspace tree, never a local directory.

        The returned list carries ``materialized_root``; the caller must
        :func:`remove_materialized` it when done.
        """
        if file_paths:
            root = await materialize_tenant_workspace(
                container_name, tenant_id, rel_paths=file_paths, repo=repo
            )
        else:
            root = await materialize_tenant_workspace(
                container_name,
                tenant_id,
                name_patterns=tuple(f"*{ext}" for ext in SimilarityConfig.SUPPORTED_EXTENSIONS),
                repo=repo,
            )
        files = materialized_files(root)
        if max_files and len(files) > max_files:
            logger.warning("Found %s files, limiting to %s", len(files), max_files)
            files = files[:max_files]
        return ResolvedFiles(files, materialized_root=root)

    @staticmethod
    async def resolve_file_paths(
        container_name: str | None,
        file_paths: list[str] | None,
        max_files: int | None,
        *,
        tenant_id: str | None = None,
        repo: str | None = None,
    ) -> list[str]:
        """Resolve file paths from container or direct file paths."""
        if container_name:
            return await SimilarityConfig.resolve_container_files(
                container_name, file_paths, max_files, tenant_id=tenant_id, repo=repo
            )
        else:
            return file_paths or []

    @staticmethod
    def release_files(files: list[str] | None) -> None:
        """Remove the materialised workspace behind ``files`` (no-op otherwise)."""
        release_materialized(files)

    @staticmethod
    def validate_file_paths(file_paths: list[str]) -> list[str]:
        """Validate file paths and return only existing files."""
        valid_files = []
        for file_path_str in file_paths:
            file_path_obj = Path(file_path_str)
            if file_path_obj.exists():
                valid_files.append(str(file_path_obj))
            else:
                logger.warning("File not found: %s", file_path_str)
        return valid_files

    @staticmethod
    def get_capabilities() -> dict[str, Any]:
        """Get similarity detection system capabilities."""
        detector = get_similarity_detector()

        capabilities = {
            "similarity_types": [t.value for t in SimilarityType],
            "analysis_scopes": [s.value for s in SimilarityScope],
            "supported_languages": [
                "java",
                "rust",
                "go",
                "python",
                "javascript",
                "typescript",
                "csharp",
                "cpp",
                "c",
            ],
            "algorithms": {
                "structural": "Entity and relationship pattern matching",
                "semantic": "Function signature and algorithm comparison",
                "syntactic": "Code style and naming convention analysis",
                "cosine_similarity": "Vector-based feature comparison",
                "jaccard_similarity": "Set-based pattern matching",
            },
            "default_thresholds": {
                sim_type.value: threshold for sim_type, threshold in detector.thresholds.items()
            },
            "features": {
                "cross_language_detection": True,
                "clone_detection": True,
                "pattern_recognition": True,
                "refactoring_insights": True,
                "performance_optimization": True,
                "confidence_scoring": True,
            },
            "limitations": {
                "max_recommended_files": 1000,
                "min_similarity_score": 0.0,
                "max_similarity_score": 1.0,
                "supported_file_extensions": SimilarityConfig.SUPPORTED_EXTENSIONS,
            },
            "performance": {
                "complexity": "O(n²) for pairwise comparison",
                "optimization": "Intelligent feature extraction and caching",
                "memory_efficient": True,
                "parallel_processing": False,  # Future augmentation
            },
        }

        return capabilities

    @staticmethod
    def get_file_language(file_path: str) -> str:
        """Determine programming language from file extension."""
        extension = Path(file_path).suffix.lower()
        language_map = {
            ".py": "python",
            ".java": "java",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".rb": "ruby",
            ".php": "php",
            ".kt": "kotlin",
            ".swift": "swift",
            ".dart": "dart",
            ".scala": "scala",
            ".clj": "clojure",
            ".hs": "haskell",
            ".ml": "ocaml",
            ".fs": "fsharp",
            ".vb": "vbnet",
        }
        return language_map.get(extension, "unknown")

    @staticmethod
    def is_supported_file(file_path: str) -> bool:
        """Check if file is supported for similarity analysis."""
        return any(file_path.endswith(ext) for ext in SimilarityConfig.SUPPORTED_EXTENSIONS)

    @staticmethod
    def get_default_thresholds() -> dict[str, float]:
        """Get default similarity thresholds for different analysis types."""
        return {
            "exact_duplicates": 0.95,
            "near_duplicates": 0.85,
            "structural_similarity": 0.75,
            "semantic_similarity": 0.70,
            "syntactic_similarity": 0.65,
            "cross_language_patterns": 0.60,
            "code_clones": 0.85,
            "refactoring_candidates": 0.80,
        }

    @staticmethod
    def get_performance_recommendations(num_files: int) -> dict[str, Any]:
        """Get performance recommendations based on number of files."""
        if num_files < 50:
            return {
                "recommended_scope": "function_level",
                "max_files_suggestion": None,
                "analysis_time_estimate": "< 1 minute",
                "memory_usage": "Low",
                "recommendations": [
                    "Full analysis recommended",
                    "All similarity types can be enabled",
                    "Cross-language detection recommended",
                ],
            }
        elif num_files < 200:
            return {
                "recommended_scope": "function_level",
                "max_files_suggestion": None,
                "analysis_time_estimate": "1-5 minutes",
                "memory_usage": "Medium",
                "recommendations": [
                    "Function-level scope optimal",
                    "Consider limiting similarity types",
                    "Monitor analysis time",
                ],
            }
        elif num_files < 1000:
            return {
                "recommended_scope": "function_level",
                "max_files_suggestion": 500,
                "analysis_time_estimate": "5-15 minutes",
                "memory_usage": "High",
                "recommendations": [
                    "Use max_files limit",
                    "Consider batch processing",
                    "Focus on specific file types",
                    "Increase min_similarity_score",
                ],
            }
        else:
            return {
                "recommended_scope": "file_level",
                "max_files_suggestion": 200,
                "analysis_time_estimate": "> 15 minutes",
                "memory_usage": "Very High",
                "recommendations": [
                    "Batch processing strongly recommended",
                    "Use file-level scope for initial analysis",
                    "Focus on specific directories",
                    "Consider distributed processing",
                    "Use high similarity thresholds",
                ],
            }


def get_similarity_config() -> dict[str, Any]:
    """
    Get similarity analysis configuration.

    Returns:
        Dictionary containing configuration settings
    """
    return {
        "supported_extensions": SimilarityConfig.SUPPORTED_EXTENSIONS,
        "skip_directories": SimilarityConfig.SKIP_DIRECTORIES,
        "default_thresholds": SimilarityConfig.get_default_thresholds(),
        "capabilities": SimilarityConfig.get_capabilities(),
        "algorithms": {
            "structural": {
                "description": "Entity and relationship pattern matching",
                "suitable_for": ["refactoring", "architecture_analysis"],
                "weight": 0.4,
            },
            "semantic": {
                "description": "Function signature and algorithm comparison",
                "suitable_for": ["code_clones", "cross_language"],
                "weight": 0.3,
            },
            "syntactic": {
                "description": "Code style and naming convention analysis",
                "suitable_for": ["style_consistency", "formatting"],
                "weight": 0.2,
            },
            "exact": {
                "description": "Exact duplicate detection",
                "suitable_for": ["duplicate_detection", "copy_paste"],
                "weight": 0.1,
            },
        },
        "analysis_types": {
            "general": {
                "description": "Balanced analysis for general purposes",
                "recommended_for": "Most use cases",
            },
            "refactoring": {
                "description": "Optimized for finding refactoring opportunities",
                "recommended_for": "Code cleanup and optimization",
            },
            "clone_detection": {
                "description": "Specialized for detecting code clones",
                "recommended_for": "Finding duplicated code",
            },
            "cross_language": {
                "description": "Cross-language pattern detection",
                "recommended_for": "Multi-language codebases",
            },
        },
        "performance": {
            "max_recommended_files": 1000,
            "optimal_batch_size": 100,
            "memory_efficient": True,
            "parallel_processing": False,
        },
    }
