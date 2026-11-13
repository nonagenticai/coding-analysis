"""
Quality Analysis Delegate - Handles quality/metrics logic for CodeExecutionService.

This delegate encapsulates all quality analysis functionality including:
- Code quality analysis
- Container quality scanning
- File batch analysis
- Quality scoring and metrics
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import UTC, datetime
from os import getenv
from typing import TYPE_CHECKING, Any
from uuid import uuid7

from models.testing_models import (
    QualityIssue,
    QualityRequest,
    QualityResult,
    QualityScanRequest,
)

if TYPE_CHECKING:
    from processing.quality_analysis.http_quality_runner import (
        HTTPQualityToolRunner,
    )
    from processing.quality_analysis.tool_registry import ToolRegistry
    from services.cache_manager import RedisCacheManager
    from services.docker_service import DockerService

logger = logging.getLogger(__name__)


class ContainerScanContext:
    """Context object for container scan operations."""

    def __init__(self, files_to_scan: list[str], files_by_language: dict[str, list[str]]):
        """Initialize the ContainerScanContext.

        Args:
            files_to_scan: List of files to scan
            files_by_language: Files grouped by language
        """
        self.files_to_scan = files_to_scan
        self.files_by_language = files_by_language
        self.has_files = bool(files_to_scan)
        self.total_files = len(files_to_scan)
        self.language_count = len(files_by_language)


class ContainerAnalysisResults:
    """Results from container analysis operations."""

    def __init__(
        self,
        all_issues: list[dict[str, Any]],
        tools_results: dict[str, Any],
        files_by_language: dict[str, list[str]],
    ):
        """Initialize the ContainerAnalysisResults.

        Args:
            all_issues: All issues found
            tools_results: Results from each tool
            files_by_language: Files grouped by language
        """
        self.all_issues = all_issues
        self.tools_results = tools_results
        self.files_by_language = files_by_language
        self.total_issues = len(all_issues)


class QualityAnalysisDelegate:
    """
    Delegate class handling quality analysis logic.

    Responsibilities:
    - Run quality analysis on code
    - Handle container quality scanning
    - Manage quality result caching
    - Calculate quality scores and metrics
    """

    def __init__(
        self,
        docker_service: DockerService,
        quality_runner: HTTPQualityToolRunner,
        tool_registry: ToolRegistry,
        cache_manager: RedisCacheManager | None,
    ):
        """Initialize the QualityAnalysisDelegate.

        Args:
            docker_service: Docker service for container operations
            quality_runner: Quality tool runner
            tool_registry: Registry of available quality tools
            cache_manager: Optional cache manager for result caching
        """
        self.docker_service = docker_service
        self.quality_runner = quality_runner
        self.tool_registry = tool_registry
        self.cache_manager = cache_manager
        logger.debug("QualityAnalysisDelegate initialized")

    # -------------------------------------------------------------------------
    # Cache Methods
    # -------------------------------------------------------------------------

    def generate_cache_key(self, request: QualityRequest) -> str:
        """Generate cache key for quality request.

        Args:
            request: The quality request

        Returns:
            A unique cache key string
        """
        tools_str = ",".join(sorted(request.tools or []))
        cache_data = f"{request.code or ''}{request.language or ''}{tools_str}"
        return f"quality_{hashlib.md5(cache_data.encode()).hexdigest()}"  # nosec B324

    async def get_cached_result(self, cache_key: str) -> QualityResult | None:
        """Get cached quality result.

        Args:
            cache_key: The cache key to look up

        Returns:
            Cached QualityResult if found, None otherwise
        """
        try:
            if not self.cache_manager:
                return None
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for quality key: {cache_key}")
                try:
                    return QualityResult(**cached_data)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to deserialize cached quality result: {e}")
            return None
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning(f"Quality cache retrieval failed: {e}")
            return None

    async def cache_result(self, cache_key: str, result: QualityResult) -> None:
        """Cache quality result.

        Args:
            cache_key: The cache key to store under
            result: The quality result to cache
        """
        try:
            if not self.cache_manager:
                return
            await self.cache_manager.set(cache_key, result.model_dump(), ttl=3600)
            logger.debug(f"Cached quality result: {cache_key}")
        except (RuntimeError, OSError, ValueError) as e:
            logger.warning(f"Quality cache storage failed: {e}")

    # -------------------------------------------------------------------------
    # Tool Registry Methods
    # -------------------------------------------------------------------------

    def get_supported_tools(self, language: str) -> list[str]:
        """Get list of supported quality tools for language.

        Args:
            language: Programming language

        Returns:
            List of tool names
        """
        return self.tool_registry.list_available_tools(language)

    def get_tool_info(self, tool_name: str, language: str) -> dict[str, Any] | None:
        """Get information about a specific tool.

        Args:
            tool_name: Name of the tool
            language: Programming language

        Returns:
            Tool information dictionary or None
        """
        return self.tool_registry.get_tool_info(language, tool_name)

    def get_supported_languages(self) -> list[str]:
        """Get list of supported programming languages.

        Returns:
            List of language names
        """
        return self.tool_registry.list_supported_languages()

    # -------------------------------------------------------------------------
    # Quality Analysis Methods
    # -------------------------------------------------------------------------

    async def analyze_quality(self, request: QualityRequest) -> QualityResult:
        """Run quality analysis on code using available tools.

        Args:
            request: Quality analysis request

        Returns:
            QualityResult with analysis details
        """
        try:
            logger.info(
                f"Running quality analysis for language: {request.language or 'auto-detect'}"
            )

            result = await self.quality_runner.run_quality_analysis(request)

            if not result.success:
                logger.error(f"Quality analysis failed with error: {result.error}")
                self._enhance_error_details(result)

            return result

        except (TimeoutError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Quality analysis failed with exception: {e}")
            logger.exception("Full exception details:")
            error_details = f"Exception in analyze_quality: {type(e).__name__}: {str(e)}"
            return QualityResult(
                analysis_id=str(uuid7()),
                success=False,
                overall_score=0.0,
                total_issues=0,
                issues=[],
                tools_results={},
                execution_time=0.0,
                created_at=datetime.now(UTC),
                language=request.language or "unknown",
                error=error_details,
            )

    def _enhance_error_details(self, result: QualityResult) -> None:
        """Enhance error details in quality result.

        Args:
            result: The quality result to enhance
        """
        specific_errors = []
        tool_diagnostics = []

        for tool_name, tool_result in (result.tools_results or {}).items():
            if isinstance(tool_result, dict):
                tool_success = tool_result.get("success", True)
                tool_error = tool_result.get("error", "")
                if not tool_success:
                    if tool_error and tool_error.strip():
                        specific_errors.append(f"{tool_name}: {tool_error}")
                    else:
                        specific_errors.append(f"{tool_name}: failed without error details")
                exec_time = tool_result.get("execution_time", 0)
                issues_count = len(tool_result.get("issues", []))
                tool_diagnostics.append(
                    f"{tool_name}(success={tool_success}, time={exec_time:.2f}s, issues={issues_count})"
                )

        if specific_errors:
            detailed_error = f"Tool failures - {'; '.join(specific_errors)}"
            logger.error(f"Tool diagnostics: {'; '.join(tool_diagnostics)}")
            if result.error and "without specific error details" not in result.error:
                result.error = f"{result.error}. Details: {detailed_error}"
            else:
                result.error = detailed_error
        elif not result.error or "without specific error details" in result.error:
            if tool_diagnostics:
                result.error = f"Quality analysis failed - tools may not be installed or accessible. Tool status: {'; '.join(tool_diagnostics)}"
            else:
                result.error = "Quality analysis failed - no tool results available, likely due to missing dependencies or configuration issues"

    async def run_single_tool(self, tool_name: str, language: str, code: str) -> dict[str, Any]:
        """Run a single quality tool on code.

        Args:
            tool_name: Name of the tool to run
            language: Programming language
            code: Code to analyze

        Returns:
            Tool result dictionary
        """
        try:
            logger.info(f"Running single tool {tool_name} for {language}")
            return await self.quality_runner.run_single_tool(tool_name, language, code)
        except (TimeoutError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Single tool execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "issues": [],
                "execution_time": 0.0,
            }

    # -------------------------------------------------------------------------
    # Container Analysis Methods
    # -------------------------------------------------------------------------

    async def analyze_container_quality(self, request: QualityScanRequest) -> QualityResult:
        """Run quality analysis on entire container filesystem.

        Args:
            request: Container quality scan request

        Returns:
            QualityResult with analysis details
        """
        analysis_id = str(uuid7())
        start_time = time.time()

        try:
            logger.info(f"Starting container quality scan: {analysis_id}")
            logger.info(f"Container: {request.container_name}, Path: {request.scan_path}")

            if not await self._validate_container_access(request.container_name):
                raise ValueError(f"Container {request.container_name} not found or not accessible")

            scan_context = await self._prepare_container_scan(request, analysis_id)
            if not scan_context.has_files:
                return self._create_empty_quality_result(analysis_id, start_time)

            analysis_results = await self._execute_container_analysis(
                request, scan_context, analysis_id
            )

            return await self._finalize_container_analysis(
                request, analysis_results, analysis_id, start_time
            )

        except (TimeoutError, RuntimeError, OSError, ValueError) as e:
            execution_time = time.time() - start_time
            logger.error(f"Container quality scan {analysis_id} failed: {e}")
            error_message = str(e).strip() if e else ""
            if not error_message:
                error_message = "Unknown error occurred during container quality analysis"
            return QualityResult(
                analysis_id=analysis_id,
                success=False,
                overall_score=0.0,
                total_issues=0,
                issues=[],
                tools_results={},
                execution_time=execution_time,
                created_at=datetime.now(UTC),
                language=request.language or "unknown",
                error=error_message,
            )

    async def _validate_container_access(
        self, container_name: str, tenant_id: str | None = None
    ) -> bool:
        """Validate that container exists and is accessible.

        Args:
            container_name: Name of the container
            tenant_id: Optional tenant ID to filter containers by tenant.

        Returns:
            True if accessible, False otherwise
        """
        try:
            if not self.docker_service.is_docker_available():
                logger.warning("Docker service is not available")
                return False

            logger.debug(f"Validating container access for: {container_name}")

            try:
                result = await self.docker_service.exec_in_container(
                    container_name, ["echo", "test"], timeout=10
                )
                success = result.get("exit_code") == 0
                if success:
                    logger.debug(f"Container {container_name} validation successful")
                    return True
                else:
                    logger.debug(
                        f"Container {container_name} validation failed - exit_code: {result.get('exit_code')}"
                    )
            except (TimeoutError, RuntimeError, OSError) as e:
                logger.debug(f"Direct container access failed: {e}")

            try:
                available_containers = await self.docker_service.list_containers(
                    tenant_id=tenant_id
                )
                for container in available_containers:
                    if (
                        hasattr(container, "container_name")
                        and container.container_name == container_name
                    ):
                        logger.debug(f"Found container by name: {container.container_name}")
                        return True
            except (TimeoutError, RuntimeError, OSError) as e:
                logger.debug(f"Container listing failed: {e}")

            logger.warning(f"Container {container_name} not found in any available containers")
            return False

        except (TimeoutError, RuntimeError, OSError) as e:
            logger.warning(f"Container {container_name} validation failed: {e}")
            return False

    async def _prepare_container_scan(
        self, request: QualityScanRequest, _analysis_id: str
    ) -> ContainerScanContext:
        """Prepare container scan by discovering and organizing files.

        Args:
            request: The scan request
            _analysis_id: Analysis ID for logging

        Returns:
            ContainerScanContext with scan preparation data
        """
        files_to_scan = await self._discover_container_files(
            request.container_name,
            request.scan_path or "/app/data",
            request.include_patterns,
            request.exclude_patterns,
            request.recursive,
            request.max_files,
        )

        if not files_to_scan:
            logger.warning(f"No files found to scan in container {request.container_name}")
            return ContainerScanContext([], {})

        logger.info(f"Found {len(files_to_scan)} files to analyze")
        files_by_language = self._group_files_by_language(files_to_scan)
        logger.info(
            f"Files grouped by language: {[(lang, len(files)) for lang, files in files_by_language.items()]}"
        )

        return ContainerScanContext(files_to_scan, files_by_language)

    async def _discover_container_files(
        self,
        container_name: str,
        scan_path: str,
        include_patterns: list[str],
        exclude_patterns: list[str],
        recursive: bool,
        max_files: int,
    ) -> list[str]:
        """Discover files in container matching patterns.

        Args:
            container_name: Name of the container
            scan_path: Path to scan
            include_patterns: Patterns to include
            exclude_patterns: Patterns to exclude
            recursive: Whether to scan recursively
            max_files: Maximum number of files

        Returns:
            List of file paths
        """
        try:
            find_cmd = ["find", scan_path, "-type", "f"]
            if not recursive:
                find_cmd.extend(["-maxdepth", "1"])

            if include_patterns:
                pattern_args = []
                for i, pattern in enumerate(include_patterns):
                    if i > 0:
                        pattern_args.append("-o")
                    pattern_args.extend(["-name", pattern])
                find_cmd.append("(")
                find_cmd.extend(pattern_args)
                find_cmd.append(")")

            result = await self.docker_service.exec_in_container(
                container_name, find_cmd, timeout=60
            )

            if result.get("exit_code") != 0:
                logger.warning(
                    f"Find command failed in container {container_name}: {result.get('stderr')}"
                )
                return []

            files = result.get("stdout", "").strip().split("\n")
            files = [f for f in files if f.strip()]

            filtered_files = []
            for file_path in files:
                should_exclude = False
                for exclude_pattern in exclude_patterns:
                    if self._matches_pattern(file_path, exclude_pattern):
                        should_exclude = True
                        break
                if not should_exclude:
                    filtered_files.append(file_path)

            return filtered_files[:max_files]

        except (TimeoutError, RuntimeError, OSError) as e:
            logger.error(f"Failed to discover files in container {container_name}: {e}")
            return []

    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches pattern.

        Args:
            file_path: Path to check
            pattern: Pattern to match against

        Returns:
            True if matches, False otherwise
        """
        import fnmatch
        from os import path, sep

        if fnmatch.fnmatch(file_path, pattern):
            return True
        if fnmatch.fnmatch(path.basename(file_path), pattern):
            return True

        path_parts = file_path.split(sep)
        for part in path_parts:
            if fnmatch.fnmatch(part, pattern):
                return True

        return False

    def _group_files_by_language(self, files: list[str]) -> dict[str, list[str]]:
        """Group files by their programming language.

        Args:
            files: List of file paths

        Returns:
            Dictionary mapping language to file lists
        """
        from os.path import splitext

        ext_language_map = {
            ".py": "python",
            ".pyw": "python",
            ".pyi": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".mjs": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".d.ts": "typescript",
        }

        files_by_language: dict[str, list[str]] = {}
        for file_path in files:
            ext = splitext(file_path)[1].lower()
            if file_path.lower().endswith(".d.ts"):
                language: str | None = "typescript"
            else:
                language = ext_language_map.get(ext)

            if language:
                if language not in files_by_language:
                    files_by_language[language] = []
                files_by_language[language].append(file_path)
            else:
                logger.debug(f"Skipping file with unknown extension: {file_path}")

        return files_by_language

    async def _execute_container_analysis(
        self,
        request: QualityScanRequest,
        scan_context: ContainerScanContext,
        analysis_id: str,
    ) -> ContainerAnalysisResults:
        """Execute the container analysis for all languages.

        Args:
            request: The scan request
            scan_context: Scan context with files
            analysis_id: Analysis ID

        Returns:
            ContainerAnalysisResults with all findings
        """
        all_issues: list[dict[str, Any]] = []
        tools_results: dict[str, Any] = {}

        for file_language, language_files in scan_context.files_by_language.items():
            if not language_files:
                continue

            logger.info(f"Processing {len(language_files)} {file_language} files")
            language_tools = request.tools or self.get_supported_tools(file_language)
            logger.info(f"Tools for {file_language}: {language_tools}")

            if not language_tools:
                logger.warning(f"No quality tools available for language: {file_language}")
                continue

            try:
                language_issues, language_tools_results = await self._process_language_files(
                    request,
                    analysis_id,
                    language_files,
                    file_language,
                    language_tools,
                )
                all_issues.extend(language_issues)
                self._merge_language_tools_results(tools_results, language_tools_results)
                logger.info(
                    f"Completed {file_language} processing: {len(language_issues)} issues found"
                )
            except (TimeoutError, RuntimeError, OSError, ValueError) as e:
                logger.error(f"Failed to process {file_language} files: {e}")
                continue

        logger.info(
            f"Multi-language processing completed: {len(all_issues)} total issues found across all languages"
        )

        return ContainerAnalysisResults(all_issues, tools_results, scan_context.files_by_language)

    async def _process_language_files(
        self,
        request: QualityScanRequest,
        analysis_id: str,
        language_files: list[str],
        file_language: str,
        language_tools: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Process files for a specific language with appropriate tools.

        Args:
            request: The scan request
            analysis_id: Analysis ID
            language_files: Files for this language
            file_language: The language
            language_tools: Tools to use

        Returns:
            Tuple of (issues list, tools results dict)
        """

        if request.parallel_analysis:
            return await self._process_files_parallel(
                request,
                analysis_id,
                language_files,
                file_language,
                language_tools,
            )
        else:
            return await self._process_files_sequential(
                request,
                analysis_id,
                language_files,
                file_language,
                language_tools,
            )

    async def _process_files_parallel(
        self,
        request: QualityScanRequest,
        analysis_id: str,
        language_files: list[str],
        file_language: str,
        language_tools: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Process files in parallel batches.

        Args:
            request: The scan request
            analysis_id: Analysis ID
            language_files: Files for this language
            file_language: The language
            language_tools: Tools to use

        Returns:
            Tuple of (issues list, tools results dict)
        """
        language_issues: list[dict[str, Any]] = []
        language_tools_results: dict[str, dict[str, Any]] = {}

        max_concurrent_batches = min(2, int(getenv("QUALITY_MAX_CONCURRENT_BATCHES", "2")))
        batch_size = max(3, min(5, len(language_files) // max_concurrent_batches))
        file_batches = [
            language_files[i : i + batch_size] for i in range(0, len(language_files), batch_size)
        ]

        logger.info(
            f"Processing {len(file_batches)} {file_language} batches with max {max_concurrent_batches} concurrent batches"
        )

        batch_semaphore = asyncio.Semaphore(max_concurrent_batches)

        async def process_batch_safely(
            batch_index: int, file_batch: list[str]
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            """Process a batch with proper error handling."""
            async with batch_semaphore:
                try:
                    logger.debug(
                        f"Starting {file_language} batch {batch_index} with {len(file_batch)} files"
                    )
                    result = await self._analyze_file_batch(
                        request.container_name,
                        file_batch,
                        file_language,
                        language_tools,
                        f"{analysis_id}_{file_language}_batch_{batch_index}",
                    )
                    logger.debug(f"Completed {file_language} batch {batch_index}")
                    return result
                except (TimeoutError, RuntimeError, OSError, ValueError) as e:
                    logger.error(f"{file_language} batch {batch_index} failed: {e}")
                    return ([], {})
                finally:
                    await asyncio.sleep(0.1)

        batch_timeout = int(getenv("QUALITY_BATCH_TIMEOUT", "1800"))
        tasks = [
            asyncio.create_task(process_batch_safely(i, batch))
            for i, batch in enumerate(file_batches)
        ]

        try:
            batch_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=batch_timeout,
            )
        except TimeoutError:
            logger.error(
                f"{file_language} batch processing timed out after {batch_timeout} seconds"
            )
            for task in tasks:
                if not task.done():
                    task.cancel()
            batch_results = []

        successful_batches = 0
        for batch_index, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                logger.error(
                    f"{file_language} batch {batch_index} processing failed: {str(batch_result)}"
                )
                continue
            if isinstance(batch_result, tuple) and len(batch_result) == 2:
                batch_issues, batch_tools_results = batch_result
            else:
                logger.error(f"Unexpected batch result format: {batch_result}")
                continue

            language_issues.extend(batch_issues)
            successful_batches += 1

            for tool, result in batch_tools_results.items():
                if tool not in language_tools_results:
                    language_tools_results[tool] = {
                        "success": True,
                        "issues": [],
                        "execution_time": 0.0,
                    }
                issues_list = language_tools_results[tool]["issues"]
                assert isinstance(issues_list, list)
                issues_list.extend(result.get("issues", []))
                language_tools_results[tool]["execution_time"] += result.get("execution_time", 0)

        logger.info(
            f"{file_language} parallel processing completed: {successful_batches}/{len(file_batches)} batches successful"
        )

        return (language_issues, language_tools_results)

    async def _process_files_sequential(
        self,
        request: QualityScanRequest,
        analysis_id: str,
        language_files: list[str],
        file_language: str,
        language_tools: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Process files sequentially.

        Args:
            request: The scan request
            analysis_id: Analysis ID
            language_files: Files for this language
            file_language: The language
            language_tools: Tools to use

        Returns:
            Tuple of (issues list, tools results dict)
        """
        language_issues: list[dict[str, Any]] = []
        language_tools_results: dict[str, dict[str, Any]] = {}

        batch_size = 3
        total_batches = (len(language_files) + batch_size - 1) // batch_size

        logger.info(
            f"Processing {total_batches} sequential {file_language} batches of {batch_size} files each"
        )

        for i in range(0, len(language_files), batch_size):
            batch_files = language_files[i : i + batch_size]
            batch_index = i // batch_size

            logger.debug(
                f"Processing sequential {file_language} batch {batch_index + 1}/{total_batches}"
            )

            try:
                batch_issues, batch_tools_results = await self._analyze_file_batch(
                    request.container_name,
                    batch_files,
                    file_language,
                    language_tools,
                    f"{analysis_id}_{file_language}_seq_{batch_index}",
                )

                language_issues.extend(batch_issues)

                for tool, result in batch_tools_results.items():
                    if tool not in language_tools_results:
                        language_tools_results[tool] = {
                            "success": True,
                            "issues": [],
                            "execution_time": 0.0,
                        }
                    issues_list = language_tools_results[tool]["issues"]
                    assert isinstance(issues_list, list)
                    issues_list.extend(result.get("issues", []))
                    language_tools_results[tool]["execution_time"] += result.get(
                        "execution_time", 0
                    )

                if batch_index < total_batches - 1:
                    await asyncio.sleep(0.5)

            except (TimeoutError, RuntimeError, OSError, ValueError) as e:
                logger.error(
                    f"Sequential {file_language} batch {batch_index} processing failed: {e}"
                )

        return (language_issues, language_tools_results)

    async def _analyze_file_batch(
        self,
        container_name: str,
        files: list[str],
        language: str,
        tools: list[str],
        batch_id: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Analyze a batch of files using quality-analyzer service.

        Args:
            container_name: Name of the container
            files: Files to analyze
            language: Programming language
            tools: Tools to use
            batch_id: Batch identifier

        Returns:
            Tuple of (issues list, tools results dict)
        """
        try:
            logger.debug(f"Analyzing batch {batch_id}: {len(files)} files")

            individual_files = []
            grouped_files = []
            file_contents: dict[str, str] = {}

            for file_path in files:
                try:
                    copy_result = await self.docker_service.exec_in_container(
                        container_name, ["cat", file_path], timeout=30
                    )
                    if copy_result.get("exit_code") == 0 and copy_result.get("stdout"):
                        content = copy_result["stdout"]

                        # Skip files with deletion markers
                        if any(
                            marker in content
                            for marker in [
                                "# This file has been deleted",
                                "# Please remove this file",
                                "File deleted - ",
                                "This file is no longer needed",
                            ]
                        ):
                            logger.info(f"Skipping {file_path} - contains deletion marker")
                            continue

                        file_contents[file_path] = content
                        if len(content) > 51200:
                            individual_files.append(file_path)
                        else:
                            grouped_files.append(file_path)
                        logger.debug(f"Copied {file_path} ({len(content)} bytes)")
                    else:
                        stderr = copy_result.get("stderr", "").strip()
                        if "__init__.py" in file_path and (not stderr or "No such file" in stderr):
                            logger.debug(f"Skipping empty/missing {file_path}")
                        else:
                            logger.warning(
                                "Failed to copy %s: %s",
                                file_path,
                                stderr or "Unknown error",
                            )
                except (TimeoutError, OSError, RuntimeError) as e:
                    logger.warning(f"Error copying file {file_path}: {e}")
                    continue

            if not file_contents:
                logger.warning(f"No files successfully copied for batch {batch_id}")
                return ([], {})

            logger.info(
                f"Batch {batch_id}: {len(individual_files)} individual files, {len(grouped_files)} small files to group"
            )

            all_batch_issues: list[dict[str, Any]] = []
            batch_tools_results: dict[str, Any] = {}

            # Process individual large files
            for file_path in individual_files:
                try:
                    quality_request = QualityRequest(
                        container_name=container_name,
                        code=file_contents[file_path],
                        code_content=file_contents[file_path],
                        code_path=file_path,
                        language=language,
                        tools=tools,
                        custom_rules=None,
                    )
                    file_result = await self.quality_runner.run_quality_analysis(quality_request)

                    if file_result.success:
                        issues, tools_results = self._process_file_analysis_result(
                            file_result, file_path, language
                        )
                        all_batch_issues.extend(issues)
                        self._merge_tools_results(batch_tools_results, tools_results)
                    else:
                        self._log_file_analysis_failure(file_path, file_result)

                except (TimeoutError, RuntimeError, OSError, ValueError) as e:
                    logger.error(f"Error analyzing individual file {file_path}: {e}")
                    continue

            # Process grouped small files
            if grouped_files:
                chunk_size = 5
                for i in range(0, len(grouped_files), chunk_size):
                    chunk_files = grouped_files[i : i + chunk_size]
                    try:
                        combined_content = self._create_combined_analysis_content(
                            chunk_files, file_contents, language
                        )
                        quality_request = QualityRequest(
                            container_name=container_name,
                            code=combined_content,
                            code_content=combined_content,
                            code_path=None,
                            language=language,
                            tools=tools,
                            custom_rules=None,
                        )
                        chunk_result = await self.quality_runner.run_quality_analysis(
                            quality_request
                        )

                        if chunk_result.success:
                            issues, tools_results = self._process_grouped_analysis_result(
                                chunk_result, chunk_files, file_contents, language
                            )
                            all_batch_issues.extend(issues)
                            self._merge_tools_results(batch_tools_results, tools_results)
                        else:
                            self._log_grouped_analysis_failure(chunk_result)

                    except (TimeoutError, RuntimeError, OSError, ValueError) as e:
                        logger.error(f"Error analyzing file chunk: {e}")
                        continue

                    await asyncio.sleep(0.1)

            logger.info(
                f"Batch {batch_id} completed: {len(all_batch_issues)} issues found across {len(file_contents)} files"
            )
            return (all_batch_issues, batch_tools_results)

        except (TimeoutError, RuntimeError, OSError, ValueError) as e:
            logger.error(f"Batch analysis failed for {batch_id}: {e}")
            return ([], {})

    def _log_file_analysis_failure(self, file_path: str, file_result: QualityResult) -> None:
        """Log file analysis failure details."""
        specific_errors = []
        for tool_name, tool_result in (file_result.tools_results or {}).items():
            if isinstance(tool_result, dict) and not tool_result.get("success", True):
                tool_error = tool_result.get("error", "Unknown error")
                specific_errors.append(f"{tool_name}: {tool_error}")

        if specific_errors:
            error_msg = f"Tool failures - {'; '.join(specific_errors)}"
        elif file_result.error and file_result.error.strip():
            error_msg = file_result.error
        else:
            error_msg = "Quality analysis failed - tools may not be installed or accessible"

        logger.warning(f"Analysis failed for {file_path}: {error_msg}")

    def _log_grouped_analysis_failure(self, chunk_result: QualityResult) -> None:
        """Log grouped analysis failure details."""
        specific_errors = []
        for tool_name, tool_result in (chunk_result.tools_results or {}).items():
            if isinstance(tool_result, dict) and not tool_result.get("success", True):
                tool_error = tool_result.get("error", "Unknown error")
                specific_errors.append(f"{tool_name}: {tool_error}")

        if specific_errors:
            error_msg = f"Tool failures - {'; '.join(specific_errors)}"
        elif chunk_result.error and chunk_result.error.strip():
            error_msg = chunk_result.error
        else:
            error_msg = "Grouped quality analysis failed - tools may not be installed or accessible"

        logger.warning(f"Grouped analysis failed for chunk: {error_msg}")

    def _create_combined_analysis_content(
        self, files: list[str], file_contents: dict[str, str], language: str = "python"
    ) -> str:
        """Create combined content for grouped file analysis.

        Args:
            files: List of file paths
            file_contents: Map of file paths to contents
            language: Programming language

        Returns:
            Combined content string
        """
        combined_parts = []
        comment_syntax = {
            "python": "# FILE: {}",
            "javascript": "// FILE: {}",
            "typescript": "// FILE: {}",
        }
        marker_template = comment_syntax.get(language.lower(), "# FILE: {}")

        for file_path in files:
            content = file_contents.get(file_path, "")
            marker = f"\n{marker_template.format(file_path)}\n"
            combined_parts.append(marker + content)

        return "\n".join(combined_parts)

    def _process_file_analysis_result(
        self, result: QualityResult, file_path: str, _language: str = "python"
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Process quality analysis result for a single file.

        Args:
            result: Quality analysis result
            file_path: Path of the analyzed file
            _language: Programming language

        Returns:
            Tuple of (issues list, tools results dict)
        """
        issues: list[dict[str, Any]] = []
        tools_results: dict[str, Any] = {}

        logger.info(f"Processing file analysis result for {file_path}")
        logger.info(f"Result success: {result.success}, total_issues: {result.total_issues}")
        logger.info(f"Result.issues count: {len(result.issues)}")
        logger.info(f"Tools_results keys: {list(result.tools_results.keys())}")

        for issue in result.issues:
            issue_dict = {
                "tool": issue.tool,
                "severity": issue.severity,
                "message": issue.message,
                "file_path": file_path,
                "line_number": getattr(issue, "line_number", None),
                "column_number": getattr(issue, "column_number", None),
                "rule_id": getattr(issue, "rule_id", None),
                "suggested_fix": getattr(issue, "suggested_fix", None),
            }
            issues.append(issue_dict)

        for tool, tool_result in result.tools_results.items():
            tool_issues_count = len(tool_result.get("issues", []))
            logger.info(f"Processing tool {tool}: {tool_issues_count} issues")
            tools_results[tool] = {
                "success": tool_result.get("success", True),
                "issues": [],
                "execution_time": tool_result.get("execution_time", 0.0),
            }

            for issue in tool_result.get("issues", []):
                issue_dict = {
                    "tool": tool,
                    "severity": issue.get("severity", "info"),
                    "message": issue.get("message", ""),
                    "file_path": file_path,
                    "line_number": issue.get("line"),
                    "column_number": issue.get("column"),
                    "rule_id": issue.get("rule_id"),
                }
                tools_results[tool]["issues"].append(issue_dict)
                issues.append(issue_dict)

        logger.info(f"Processed result: {len(issues)} total issues extracted")
        return (issues, tools_results)

    def _process_grouped_analysis_result(
        self,
        result: QualityResult,
        files: list[str],
        file_contents: dict[str, str],
        _language: str = "python",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Process quality analysis result for grouped files.

        Args:
            result: Quality analysis result
            files: List of file paths
            file_contents: Map of file paths to contents
            _language: Programming language

        Returns:
            Tuple of (issues list, tools results dict)
        """
        issues: list[dict[str, Any]] = []
        tools_results: dict[str, Any] = {}

        logger.info(f"Processing grouped analysis result for {len(files)} files")
        logger.info(f"Result success: {result.success}, total_issues: {result.total_issues}")
        logger.info(f"Result.issues count: {len(result.issues)}")
        logger.info(f"Tools_results keys: {list(result.tools_results.keys())}")

        line_offsets = self._calculate_line_offsets(files, file_contents)

        for issue in result.issues:
            target_file = self._determine_issue_file(issue, line_offsets, files)
            adjusted_line = self._adjust_line_number(issue, line_offsets, target_file)
            issue_dict = {
                "tool": issue.tool,
                "severity": issue.severity,
                "message": issue.message,
                "file_path": target_file,
                "line_number": adjusted_line,
                "column_number": getattr(issue, "column_number", None),
                "rule_id": getattr(issue, "rule_id", None),
                "suggested_fix": getattr(issue, "suggested_fix", None),
            }
            issues.append(issue_dict)

        for tool, tool_result in result.tools_results.items():
            tool_issues_count = len(tool_result.get("issues", []))
            logger.info(f"Processing tool {tool}: {tool_issues_count} issues")
            tools_results[tool] = {
                "success": tool_result.get("success", True),
                "issues": [],
                "execution_time": tool_result.get("execution_time", 0.0),
            }

            for issue in tool_result.get("issues", []):
                target_file = self._determine_tool_issue_file(issue, line_offsets, files)
                adjusted_line = self._adjust_tool_issue_line(issue, line_offsets, target_file)
                issue_dict = {
                    "tool": tool,
                    "severity": issue.get("severity", "info"),
                    "message": issue.get("message", ""),
                    "file_path": target_file,
                    "line_number": adjusted_line,
                    "column_number": issue.get("column"),
                    "rule_id": issue.get("rule_id"),
                }
                tools_results[tool]["issues"].append(issue_dict)
                issues.append(issue_dict)

        logger.info(f"Processed grouped result: {len(issues)} total issues extracted")
        return (issues, tools_results)

    def _calculate_line_offsets(
        self, files: list[str], file_contents: dict[str, str]
    ) -> dict[str, int]:
        """Calculate line number offsets for each file in combined content.

        Args:
            files: List of file paths
            file_contents: Map of file paths to contents

        Returns:
            Dictionary mapping file paths to line offsets
        """
        offsets = {}
        current_line = 1
        for file_path in files:
            offsets[file_path] = current_line
            content = file_contents.get(file_path, "")
            current_line += content.count("\n") + 2
        return offsets

    def _determine_issue_file(
        self, issue: Any, line_offsets: dict[str, int], files: list[str]
    ) -> str:
        """Determine which file an issue belongs to based on line number.

        Args:
            issue: The issue object
            line_offsets: Line offset mapping
            files: List of file paths

        Returns:
            File path for the issue
        """
        issue_line = getattr(issue, "line_number", 1) or 1
        for file_path in reversed(files):
            if line_offsets[file_path] <= issue_line:
                return file_path
        return files[0] if files else "unknown"

    def _adjust_line_number(
        self, issue: Any, line_offsets: dict[str, int], target_file: str
    ) -> int | None:
        """Adjust line number to be relative to individual file.

        Args:
            issue: The issue object
            line_offsets: Line offset mapping
            target_file: Target file path

        Returns:
            Adjusted line number or None
        """
        issue_line = getattr(issue, "line_number", None)
        if issue_line is None:
            return None
        if not isinstance(issue_line, int):
            return None
        file_offset = line_offsets.get(target_file, 1)
        adjusted_line = issue_line - file_offset
        return max(1, adjusted_line)

    def _determine_tool_issue_file(
        self, issue: dict[str, Any], line_offsets: dict[str, int], files: list[str]
    ) -> str:
        """Determine which file a tool issue belongs to.

        Args:
            issue: The issue dictionary
            line_offsets: Line offset mapping
            files: List of file paths

        Returns:
            File path for the issue
        """
        issue_line = issue.get("line", 1) or 1
        for file_path in reversed(files):
            if line_offsets[file_path] <= issue_line:
                return file_path
        return files[0] if files else "unknown"

    def _adjust_tool_issue_line(
        self, issue: dict[str, Any], line_offsets: dict[str, int], target_file: str
    ) -> int | None:
        """Adjust tool issue line number to be relative to individual file.

        Args:
            issue: The issue dictionary
            line_offsets: Line offset mapping
            target_file: Target file path

        Returns:
            Adjusted line number or None
        """
        issue_line = issue.get("line")
        if issue_line is None:
            return None
        if not isinstance(issue_line, int):
            return None
        file_offset = line_offsets.get(target_file, 1)
        adjusted_line = issue_line - file_offset
        return max(1, adjusted_line)

    def _merge_tools_results(self, target: dict[str, Any], source: dict[str, Any]) -> None:
        """Merge tools results from source into target.

        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for tool, result in source.items():
            if tool not in target:
                target[tool] = {"success": True, "issues": [], "execution_time": 0.0}
            target[tool]["issues"].extend(result.get("issues", []))
            target[tool]["execution_time"] += result.get("execution_time", 0)
            if not result.get("success", True):
                target[tool]["success"] = False

    def _merge_language_tools_results(
        self, tools_results: dict[str, Any], language_tools_results: dict[str, Any]
    ) -> None:
        """Merge tools results from language processing.

        Args:
            tools_results: Target dictionary
            language_tools_results: Source dictionary
        """
        for tool, result in language_tools_results.items():
            if tool not in tools_results:
                tools_results[tool] = {
                    "success": True,
                    "issues": [],
                    "execution_time": 0.0,
                }
            tools_results[tool]["issues"].extend(result.get("issues", []))
            tools_results[tool]["execution_time"] += result.get("execution_time", 0)

    async def _finalize_container_analysis(
        self,
        request: QualityScanRequest,
        analysis_results: ContainerAnalysisResults,
        analysis_id: str,
        start_time: float,
    ) -> QualityResult:
        """Finalize container analysis and create result.

        Args:
            request: The scan request
            analysis_results: Analysis results
            analysis_id: Analysis ID
            start_time: Start time of analysis

        Returns:
            QualityResult with final analysis
        """
        overall_score = self._calculate_container_quality_score(
            analysis_results.all_issues, analysis_results.tools_results
        )
        logger.info(
            f"Quality analysis summary: {analysis_results.total_issues} total issues, score: {overall_score:.1f}"
        )

        self._validate_quality_thresholds(request, analysis_results.all_issues, overall_score)

        execution_time = time.time() - start_time
        primary_language = self._determine_primary_language(analysis_results.files_by_language)

        result = QualityResult(
            analysis_id=analysis_id,
            success=True,
            overall_score=overall_score,
            total_issues=analysis_results.total_issues,
            issues=self._convert_to_quality_issues(analysis_results.all_issues),
            tools_results=analysis_results.tools_results,
            execution_time=execution_time,
            created_at=datetime.now(UTC),
            language=primary_language,
            error=None,
        )

        logger.info(
            f"Container quality scan completed: {analysis_id}, score: {overall_score:.1f}, total issues: {analysis_results.total_issues}"
        )

        return result

    def _validate_quality_thresholds(
        self,
        request: QualityScanRequest,
        all_issues: list[dict[str, Any]],
        overall_score: float,
    ) -> None:
        """Validate quality thresholds and raise errors if not met.

        Args:
            request: The scan request
            all_issues: All issues found
            overall_score: Overall quality score
        """
        if request.fail_on_critical:
            critical_issues = [i for i in all_issues if i.get("severity") == "critical"]
            if critical_issues:
                raise ValueError(f"Critical issues found: {len(critical_issues)}")

        if request.min_quality_score and overall_score < request.min_quality_score:
            logger.warning(
                f"Quality score {overall_score:.1f} below minimum {request.min_quality_score}, but continuing with results"
            )

    def _determine_primary_language(self, files_by_language: dict[str, list[str]]) -> str:
        """Determine primary language for reporting.

        Args:
            files_by_language: Files grouped by language

        Returns:
            Primary language string
        """
        if not files_by_language:
            return "unknown"
        if len(files_by_language) == 1:
            return list(files_by_language.keys())[0]
        primary_language = max(files_by_language.keys(), key=lambda k: len(files_by_language[k]))
        return f"mixed ({primary_language} primary)"

    def _calculate_container_quality_score(
        self, issues: list[dict[str, Any]], tools_results: dict[str, Any]
    ) -> float:
        """Calculate overall quality score for container scan.

        Args:
            issues: List of issues
            tools_results: Tools results

        Returns:
            Quality score (0-100)
        """
        logger.debug(
            f"Calculating quality score for {len(issues)} issues and {len(tools_results)} tools"
        )

        if not issues and not tools_results:
            logger.debug("No issues and no tools results, returning baseline score")
            return 90.0

        critical_count = len([i for i in issues if i.get("severity") == "critical"])
        error_count = len([i for i in issues if i.get("severity") == "error"])
        warning_count = len([i for i in issues if i.get("severity") == "warning"])
        info_count = len([i for i in issues if i.get("severity") == "info"])

        logger.debug(
            f"Issue counts: critical={critical_count}, error={error_count}, warning={warning_count}, info={info_count}"
        )

        score = 100.0
        score -= critical_count * 10.0
        score -= error_count * 5.0
        score -= warning_count * 2.0
        score -= info_count * 0.5

        logger.debug(f"Score after issue deductions: {score}")

        successful_tools = len([r for r in tools_results.values() if r.get("success", False)])
        total_tools = len(tools_results)
        logger.debug(f"Tool success rate: {successful_tools}/{total_tools}")

        if total_tools > 0:
            tool_success_rate = successful_tools / total_tools
            score *= tool_success_rate
            logger.debug(f"Score after tool success rate adjustment: {score}")

        final_score = max(0.0, min(100.0, score))
        logger.debug(f"Final quality score: {final_score}")
        return final_score

    def _convert_to_quality_issues(self, issues: list[dict[str, Any]]) -> list[QualityIssue]:
        """Convert issue dictionaries to QualityIssue objects.

        Args:
            issues: List of issue dictionaries

        Returns:
            List of QualityIssue objects
        """
        quality_issues = []
        for issue in issues:
            try:
                quality_issue = QualityIssue(
                    tool=issue.get("tool", "unknown"),
                    severity=issue.get("severity", "info"),
                    message=issue.get("message", "No message"),
                    file_path=issue.get("file_path", "unknown"),
                    line_number=issue.get("line_number"),
                    column_number=issue.get("column_number"),
                    rule_id=issue.get("rule_id"),
                    suggested_fix=issue.get("suggested_fix"),
                    fix_confidence=issue.get("fix_confidence"),
                )
                quality_issues.append(quality_issue)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to convert issue to QualityIssue: {e}")
                continue

        return quality_issues

    def _create_empty_quality_result(self, analysis_id: str, start_time: float) -> QualityResult:
        """Create an empty quality result for cases with no files found.

        Args:
            analysis_id: Analysis ID
            start_time: Start time

        Returns:
            Empty QualityResult
        """
        return QualityResult(
            analysis_id=analysis_id,
            success=True,
            overall_score=90.0,
            total_issues=0,
            issues=[],
            tools_results={},
            execution_time=time.time() - start_time,
            created_at=datetime.now(UTC),
            language="unknown",
            error=None,
        )

    def _detect_container_language(self, files: list[str]) -> str:
        """Detect primary language from file extensions.

        Args:
            files: List of file paths

        Returns:
            Detected language string
        """
        from os.path import splitext

        extensions: dict[str, int] = {}
        for file_path in files:
            ext = splitext(file_path)[1].lower()
            extensions[ext] = extensions.get(ext, 0) + 1

        logger.debug(f"File extensions found: {extensions}")

        ext_language_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
        }

        language_counts: dict[str, int] = {}
        for ext, count in extensions.items():
            lang = ext_language_map.get(ext)
            if lang:
                language_counts[lang] = language_counts.get(lang, 0) + count

        logger.debug(f"Language counts: {language_counts}")

        if language_counts:
            detected_lang = max(language_counts, key=lambda x: language_counts[x])
            logger.debug(f"Detected language: {detected_lang}")
            return detected_lang

        logger.debug("No recognized language extensions found, defaulting to python")
        return "python"

    def _get_compatible_files_for_tool(
        self, tool: str, language: str, files: list[str]
    ) -> list[str]:
        """Filter files to only include those compatible with the tool's language.

        Args:
            tool: Tool name
            language: Programming language
            files: List of file paths

        Returns:
            List of compatible file paths
        """
        language_extensions = {
            "python": [".py", ".pyw", ".pyi"],
            "javascript": [".js", ".jsx", ".mjs"],
            "typescript": [".ts", ".tsx", ".d.ts"],
        }

        extensions = language_extensions.get(language.lower(), [])
        if not extensions:
            logger.warning(f"No file extensions defined for language: {language}")
            return files

        compatible_files = []
        for file_path in files:
            file_lower = file_path.lower()
            if any(file_lower.endswith(ext) for ext in extensions):
                compatible_files.append(file_path)

        logger.debug(
            f"Tool {tool} ({language}): filtered {len(files)} files to {len(compatible_files)} compatible files"
        )
        return compatible_files
