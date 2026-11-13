"""Module for analyzing the quality of architectural components."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clients.coding_containers import get_coding_containers_client
from models.architecture_models import ArchitectureQualityIssue, QualityReport
from processing.architecture.architecture_pattern_analysis import (
    ArchitecturePatternAnalysis,
)
from processing.architecture.code_quality_metrics import CodeQualityMetrics
from processing.architecture.complexity_analysis import ComplexityAnalysis
from processing.architecture.container_fs import (
    ScopeMatchesNoFilesError,
    materialize_from_container,
)
from processing.architecture.performance_analysis import PerformanceAnalysis
from processing.architecture.quality_reporting import QualityReporting
from processing.architecture.tool_execution import ToolExecution
from processing.workspace_repos import (
    PRIMARY_REPO,
    is_repos_path,
    normalize_repo,
    repo_of_path,
    repo_root,
)

logger = logging.getLogger(__name__)

# ``find -name`` globs for every source extension the detection / perf scanners
# below recognise (union of the maps in ArchitecturePatternAnalysis and
# PerformanceAnalysis). Used to mirror source files out of the analysis container.
SOURCE_GLOBS: tuple[str, ...] = (
    "*.py",
    "*.js",
    "*.jsx",
    "*.ts",
    "*.tsx",
    "*.java",
    "*.go",
    "*.rs",
    "*.cpp",
    "*.cc",
    "*.cxx",
    "*.hpp",
    "*.h",
    "*.cs",
)


class ToolsNotApplicableError(ValueError):
    """Requested ``tools`` that cannot run on the detected language.

    Raised (never swallowed into a ``None`` report) so the caller answers 422
    instead of running a different tool set, or nothing, and calling it clean.
    """

    def __init__(self, language: str, unknown: Sequence[str], accepted: Sequence[str]) -> None:
        super().__init__(
            f"tools not available for detected language {language!r}: {list(unknown)}; "
            f"accepted: {list(accepted)}"
        )
        self.language = language
        self.unknown = list(unknown)
        self.accepted = list(accepted)


def _normalize_issue_path(file_path: str, roots: tuple[str, ...]) -> str:
    """Reduce a tool-reported path to repo-relative form (strip a known root
    prefix and any leading ``./``) so it can be compared against a scope."""
    path = file_path.strip()
    for root in roots:
        prefix = root.rstrip("/") + "/"
        if root and path.startswith(prefix):
            path = path[len(prefix) :]
            break
    while path.startswith("./"):
        path = path[2:]
    return path


def _issue_in_scope(file_path: str | None, scope: list[str], roots: tuple[str, ...]) -> bool:
    """True iff ``file_path`` is one of the scoped files or under a scoped dir."""
    if not file_path:
        return False
    rel = _normalize_issue_path(file_path, roots)
    return any(rel == p or rel.startswith(p + "/") for p in scope)


def _refuse_unverified_scoped_clippy(
    tools_results: dict[str, dict[str, Any]], issues: list[ArchitectureQualityIssue]
) -> None:
    """A scoped clippy run that exited non-zero with no finding left in scope failed.

    A non-zero exit means a crate did not build; cargo then never lints the
    crates that depend on it, so the scoped files may not have been linted at
    all and the scope filter would turn the build error elsewhere into "clean".
    """
    result = tools_results.get("clippy")
    if (
        result
        and result.get("success")
        and result.get("exit_code") not in (0, None)
        and not any(i.tool == "clippy" for i in issues)
    ):
        result["success"] = False
        result["error"] = (
            f"clippy exited {result.get('exit_code')} on findings outside the requested "
            "paths: the scoped files may not have been linted"
        )


class ArchitectureQualityAnalyzer:
    """Comprehensive quality analyzer for architecture processing."""

    def __init__(self) -> None:
        """Initialize the quality analyzer with specialized modules."""
        self.pattern_analyzer = ArchitecturePatternAnalysis()
        self.metrics_calculator = CodeQualityMetrics()
        self.complexity_analyzer = ComplexityAnalysis()
        self.performance_analyzer = PerformanceAnalysis()
        self.reporter = QualityReporting()
        self.tool_executor = ToolExecution()

    async def analyze(
        self,
        repo_path: Path,
        codebase_id: str = "",
        tenant_id: str | None = None,
        scope_paths: list[str] | None = None,
        repo: str | None = None,
        language: str | None = None,
        workdir: str | None = None,
        tools: list[str] | None = None,
    ) -> QualityReport | None:
        """Analyze code quality in the given repository.

        ``repo`` picks the repository of a multi-repo workspace
        (``processing.workspace_repos``): ``None``/``"primary"`` analyses the
        workspace root and DROPS everything under ``.repos/`` (the source
        checkouts are not part of it) unless ``scope_paths`` names ``.repos/``
        paths explicitly; ``".repos/<name>"`` moves the whole run -- mirroring,
        tool cwd, reported paths -- to that checkout. Every issue carries the
        ``repo`` it belongs to.

        ``language`` (a ``tools_config`` key) overrides detection, so the
        language's tools run whatever the file mix is. ``workdir`` (repo-relative,
        pre-validated) is the directory project tools run in -- the crate dir
        for cargo -- and only applies to them.

        ``scope_paths`` (repo-relative files/dirs, pre-validated) restricts the
        run: only files under those paths are mirrored out of the container,
        every path-taking tool is pointed at them (tools that cannot be scoped
        are skipped, not run project-wide), and any issue outside the scope is
        dropped. A scope matching no source file raises
        :class:`ScopeMatchesNoFilesError`. ``None`` = whole repo, unchanged.

        ``tenant_id`` is the tenant that OWNS ``codebase_id``, threaded from the
        authenticated request. Every container hop below (the materialization
        ``find``/``read_file`` and each tool exec) is tenant-scoped: without it
        coding-containers resolves this service as its own account and its BOLA
        guard 404s the workspace, which is how six Python tools "failed" in 25 ms.

        ``tools`` restricts the run to exactly those registry tools for the
        detected language: no other tool runs, and the built-in performance
        pattern scan (``performance_analyzer``) is skipped too, so the report
        holds only results from tools that were asked for. A name that is not an
        enabled tool for the detected language (or an empty list) raises
        :class:`ToolsNotApplicableError`. ``None`` = the full default set.
        """
        start_time = datetime.now(UTC)
        scope_not_analysed: list[str] = []
        materialized: Path | None = None
        selected_repo = normalize_repo(repo)
        container_root = repo_root("/workspace", selected_repo)
        repo_path = Path(repo_root(str(repo_path), selected_repo))

        try:
            logger.info(f"Starting quality analysis for {repo_path}")

            # The repo lives inside the analysis container at /workspace, not on
            # the analyzer pod's local fs. Mirror its source files out so the sync
            # Path-based language detection / perf scanners below see real files;
            # otherwise a non-Python repo is misdetected as the "python" default
            # and silently scored N/A. Fall back to the local path when there is
            # no container id (unit tests / local callers) or nothing materialized.
            if codebase_id:
                client = get_coding_containers_client()
                materialized = await materialize_from_container(
                    client,
                    codebase_id,
                    container_root,
                    SOURCE_GLOBS,
                    tenant_id=tenant_id,
                    scope_paths=scope_paths,
                    existing_scope_ok=True,
                )
                if materialized is None and scope_paths:
                    # Every path exists (a missing one was refused above) but none
                    # holds a source file: nothing for the tools to read.
                    raise ScopeMatchesNoFilesError(
                        f"paths hold no readable source file to analyse: {scope_paths}", []
                    )
                if materialized is not None and scope_paths:
                    # A scope path that exists but holds no source file
                    # (``pyproject.toml``) was left out of the copy; the tools
                    # only get the paths that did hold one.
                    analysable = [p for p in scope_paths if (materialized / p).exists()]
                    scope_not_analysed = [p for p in scope_paths if p not in analysable]
                    scope_paths = analysable
                if materialized is None:
                    logger.warning(
                        "No source files materialized from container '%s'; "
                        "falling back to local path %s",
                        codebase_id,
                        repo_path,
                    )
            elif scope_paths:
                missing = [p for p in scope_paths if not (repo_path / p).exists()]
                if missing:
                    raise ScopeMatchesNoFilesError(
                        f"paths match no files under {repo_path}: {missing}", missing
                    )
            effective_repo = materialized or repo_path

            # An explicit language wins over detection: a Rust target with a few
            # Python helper scripts must still get clippy/rustfmt.
            if language:
                logger.info("Using requested language: %s", language)
            else:
                language = self.pattern_analyzer.detect_primary_language(effective_repo)
                logger.info(f"Detected primary language: {language}")

            # Run quality tools based on language
            issues: list[ArchitectureQualityIssue] = []
            tools_results: dict[str, dict[str, Any]] = {}

            # Get available tools for the language
            available_tools = self.pattern_analyzer.get_available_tools(language)
            if tools is not None:
                requested = list(dict.fromkeys(tools))
                enabled = [
                    t for t in available_tools if self.pattern_analyzer.is_tool_enabled(language, t)
                ]
                unknown = [t for t in requested if t not in enabled]
                if unknown or not requested:
                    raise ToolsNotApplicableError(language, unknown, enabled)
                # Registry order, restricted to exactly the requested set.
                available_tools = [t for t in enabled if t in requested]

            for tool_name in available_tools:
                if self.pattern_analyzer.is_tool_enabled(language, tool_name):
                    tool_config = self.pattern_analyzer.get_tool_config(language, tool_name)
                    command = tool_config.get("command", [])
                    if isinstance(command, list):
                        tool_issues, tool_result = await self.tool_executor.run_tool(
                            tool_name,
                            command,
                            repo_path,
                            language,
                            container_name=codebase_id,
                            tenant_id=tenant_id,
                            scope_paths=scope_paths,
                            workdir=workdir,
                        )
                        if tool_result.get("skipped"):
                            # Not run (cannot be path-scoped): not attempted.
                            continue
                        issues.extend(tool_issues)
                        tools_results[tool_name] = tool_result

            # Add performance analysis -- not a requested tool, so a run restricted
            # to named tools must not carry its findings.
            if tools is None:
                performance_issues = self.performance_analyzer.analyze_performance_patterns(
                    effective_repo, language
                )
                issues.extend(performance_issues)

            roots = (str(repo_path), container_root, str(effective_repo))
            if scope_paths:
                # Safety net: tools were pointed at the scope, but a tool may
                # still report on files it pulled in (imports, configs), and a
                # whole-crate tool (cargo) reports on everything. Only issues
                # on the requested paths belong in a scoped report.
                issues = [i for i in issues if _issue_in_scope(i.file_path, scope_paths, roots)]
                _refuse_unverified_scoped_clippy(tools_results, issues)
            elif selected_repo == PRIMARY_REPO:
                # Tools walk the whole workspace root, so a linter that recurses
                # also reports on the source checkouts under .repos/. They are
                # not the primary repository: drop them.
                issues = [
                    i
                    for i in issues
                    if not is_repos_path(_normalize_issue_path(i.file_path or "", roots))
                ]
            for issue in issues:
                issue.repo = repo_of_path(
                    _normalize_issue_path(issue.file_path or "", roots), selected_repo
                )

            # Honest abstain: if no static-analysis tool ran successfully, an empty
            # issue list means "unknown", not "perfect". Reporting 100/A here would
            # be a vacuous pass. Signal inconclusive via score -1.0 / grade "N/A"
            # (the score field has no ge=0 bound, so the sentinel is model-valid).
            # Per-tool provenance (the source of truth is tools_results, which is
            # otherwise discarded). `tools_attempted` are the tools we actually
            # ran; a tool "succeeded" iff its result carries success=True (the
            # dispatcher leaves success=False + an `error` on any failure — see
            # tool_execution/_base.py). `partial` is the middle of the spectrum
            # whose ends are all-passed and inconclusive (0 succeeded).
            tools_attempted = list(tools_results.keys())
            tools_succeeded = [k for k, r in tools_results.items() if r.get("success")]
            tools_failed = [k for k, r in tools_results.items() if not r.get("success")]
            successful_tools = len(tools_succeeded)
            partial = 0 < successful_tools < len(tools_attempted)
            if successful_tools == 0:
                logger.warning(
                    "Quality analysis inconclusive for %s: 0/%d tools succeeded",
                    repo_path,
                    len(tools_results),
                )
                overall_score = -1.0
                grade = "N/A"
            else:
                # Calculate overall score and grade
                overall_score = self.metrics_calculator.calculate_quality_score(issues)
                grade = self.metrics_calculator.score_to_grade(overall_score)

            # Generate comprehensive summary
            total_files = self.pattern_analyzer.count_files_by_language(effective_repo, language)
            summary = self.reporter.generate_summary_statistics(
                issues, total_files, {"tools_run": len(tools_results)}
            )

            execution_time = (datetime.now(UTC) - start_time).total_seconds()

            # Generate comprehensive quality report
            report = self.reporter.generate_quality_report(
                codebase_id=codebase_id,
                language=language,
                issues=issues,
                overall_score=overall_score,
                grade=grade,
                summary=summary,
                execution_time=execution_time,
                inconclusive=successful_tools == 0,
                partial=partial,
                tools_attempted=tools_attempted,
                tools_succeeded=tools_succeeded,
                tools_failed=tools_failed,
            )
            report.scope_not_analysed = scope_not_analysed

            return report

        except (ScopeMatchesNoFilesError, ToolsNotApplicableError):
            raise
        except Exception as e:
            logger.error(f"Quality analysis failed for {repo_path}: {e}")
            return None
        finally:
            if materialized is not None:
                shutil.rmtree(materialized, ignore_errors=True)

    # Legacy method - now delegated to pattern analyzer
    def _detect_primary_language(self, repo_path: Path) -> str:
        """Detect the primary programming language in the repository."""
        return self.pattern_analyzer.detect_primary_language(repo_path)

    # Legacy method - now delegated to tool executor
    async def _run_tool(
        self,
        tool_name: str,
        command: list[str],
        repo_path: Path,
        language: str,
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run a specific quality analysis tool."""
        return await self.tool_executor.run_tool(tool_name, command, repo_path, language)

    # Legacy method - now delegated to metrics calculator
    def calculate_quality_score(self, issues: list[ArchitectureQualityIssue]) -> float:
        """Calculate overall quality score from issues."""
        return self.metrics_calculator.calculate_quality_score(issues)

    # Legacy methods - functionality moved to specialized modules
    # These methods are kept for backward compatibility but delegate to new modules

    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        return self.metrics_calculator.score_to_grade(score)

    def _count_issues_by_severity(self, issues: list[ArchitectureQualityIssue]) -> dict[str, int]:
        """Count issues by severity level."""
        return self.metrics_calculator.count_issues_by_severity(issues)

    def _count_issues_by_tool(self, issues: list[ArchitectureQualityIssue]) -> dict[str, int]:
        """Count issues by tool."""
        return self.metrics_calculator.count_issues_by_tool(issues)

    def _count_files(self, repo_path: Path, language: str) -> int:
        """Count files for the given language."""
        return self.pattern_analyzer.count_files_by_language(repo_path, language)

    def _generate_recommendations(
        self, issues: list[ArchitectureQualityIssue], score: float
    ) -> list[str]:
        """Generate recommendations based on issues found."""
        return self.reporter.generate_recommendations(issues, score)

    # Additional convenience methods for advanced analysis

    def analyze_complexity(self, repo_path: Path, language: str = "python") -> dict[str, Any]:
        """Analyze code complexity metrics."""
        return {
            "radon_analysis": self.complexity_analyzer.run_radon_complexity(repo_path),
            "architecture_patterns": self.pattern_analyzer.detect_architecture_patterns(repo_path),
            "dependency_structure": self.pattern_analyzer.analyze_dependency_structure(
                repo_path, language
            ),
        }

    def analyze_performance(self, repo_path: Path, language: str = "python") -> dict[str, Any]:
        """Analyze performance patterns and issues."""
        return {
            "performance_patterns": self.performance_analyzer.analyze_performance_patterns(
                repo_path, language
            ),
            "memory_analysis": self.performance_analyzer.analyze_memory_usage_patterns(
                repo_path, language
            ),
            "profiling_analysis": self.performance_analyzer.run_profiling_analysis(
                repo_path, language
            ),
        }

    async def generate_detailed_report(
        self,
        repo_path: Path,
        codebase_id: str = "",
        include_performance: bool = False,
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Generate a comprehensive detailed quality report."""
        # Run main analysis
        main_report = await self.analyze(repo_path, codebase_id, tenant_id=tenant_id)
        if not main_report:
            return None

        # Add detailed breakdown
        detailed_data = {
            "main_report": main_report,
            "detailed_breakdown": self.reporter.generate_detailed_breakdown(main_report.issues),
            "executive_summary": self.reporter.generate_executive_summary(main_report),
        }

        # Add performance analysis if requested. These sync scanners read the
        # filesystem too, so mirror the container's source files out the same way
        # analyze() does (fall back to the local path for tests / local callers).
        if include_performance:
            materialized: Path | None = None
            try:
                if codebase_id:
                    client = get_coding_containers_client()
                    materialized = await materialize_from_container(
                        client, codebase_id, "/workspace", SOURCE_GLOBS, tenant_id=tenant_id
                    )
                effective_repo = materialized or repo_path
                language = self.pattern_analyzer.detect_primary_language(effective_repo)
                detailed_data["performance_analysis"] = self.analyze_performance(
                    effective_repo, language
                )
                detailed_data["complexity_analysis"] = self.analyze_complexity(
                    effective_repo, language
                )
            finally:
                if materialized is not None:
                    shutil.rmtree(materialized, ignore_errors=True)

        return detailed_data

    # All tool execution methods have been moved to ToolExecution module
    # For backward compatibility, tool methods delegate to ToolExecution
    async def _run_ruff(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run Ruff linter - delegated to ToolExecution."""
        return await self.tool_executor.run_tool("ruff", [], repo_path, "python")

    async def _run_ty(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run the ty type checker - delegated to ToolExecution."""
        return await self.tool_executor.run_tool("ty", [], repo_path, "python")

    async def _run_bandit(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run Bandit security scanner - delegated to ToolExecution."""
        return await self.tool_executor.run_tool("bandit", [], repo_path, "python")

    async def _run_radon(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run Radon complexity analyzer - delegated to ComplexityAnalysis."""
        return self.complexity_analyzer.run_radon_complexity(repo_path)

    async def _run_eslint(
        self, repo_path: Path, language: str
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run ESLint - delegated to ToolExecution."""
        return await self.tool_executor.run_tool("eslint", [], repo_path, language)

    async def _run_typescript_compiler(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        """Run TypeScript compiler - delegated to ToolExecution."""
        return await self.tool_executor.run_tool("tsc", [], repo_path, "typescript")

    # Additional tool stubs - all delegate to ToolExecution module
    async def _run_ruff_format(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        return await self.tool_executor.run_tool("ruff-format", [], repo_path, "python")

    async def _run_prettier(
        self, repo_path: Path, language: str
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        return await self.tool_executor.run_tool("prettier", [], repo_path, language)

    async def _run_jshint(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        return await self.tool_executor.run_tool("jshint", [], repo_path, "javascript")

    async def _run_checkstyle(
        self, repo_path: Path
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        return await self.tool_executor.run_tool("checkstyle", [], repo_path, "java")

    async def _run_semgrep(
        self, repo_path: Path, language: str
    ) -> tuple[list[ArchitectureQualityIssue], dict[str, Any]]:
        return await self.tool_executor.run_tool("semgrep", [], repo_path, language)

    # Severity mapping methods - kept for compatibility
    def _ruff_severity_to_standard(self, code: str) -> str:
        """Convert Ruff error codes to standard severity."""
        if code.startswith("F"):  # Fatal errors
            return "critical"
        if code.startswith("E"):  # Errors
            return "high"
        if code.startswith("W"):  # Warnings
            return "medium"
        return "low"

    def _eslint_severity_to_standard(self, severity: int) -> str:
        """Convert ESLint severity to standard severity."""
        return "high" if severity == 2 else "medium"
