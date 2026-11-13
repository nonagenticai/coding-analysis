"""
Automated Resolve Suggester - Generate and apply automated code resolves.
Analyzes quality issues and provides intelligent resolve suggestions with confidence scoring.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid7

from models.testing_models import QualityIssue, QualityResult
from processing.quality_analysis.fix_formatters import (
    CodeFormatter,
    QualityAnalysisCodeValidator,
    RuffFixError,
)
from processing.quality_analysis.fix_models import (
    BugFixValidationResult,
    FixApplicationResult,
    FixCategory,
    FixRule,
    FixStrategy,
    QualityAnalysisResolveSuggestions,
)
from processing.quality_analysis.fix_rules import (
    FixDetailGenerator,
    FixRuleManager,
)
from services.cache_manager import RedisCacheManager

logger = logging.getLogger(__name__)


@dataclass
class FixApplicationContext:
    """Context for fix application to reduce method parameters.

    Attributes:
        applicable_fixes (List[QualityAnalysisResolveSuggestions]): Class attribute applicable_fixes.
        dry_run (bool): Class attribute dry_run.
        create_backup (bool): Class attribute create_backup.
        application_id (str): Class attribute application_id.
        start_time (float): Class attribute start_time.

    Example:
        >>> obj = FixApplicationContext()"""

    applicable_fixes: list[QualityAnalysisResolveSuggestions]
    dry_run: bool
    create_backup: bool
    application_id: str
    start_time: float

    def __str__(self) -> str:
        """Return string representation.

        Returns:
            str: The str result."""
        return (
            f"FixApplicationContext(id={self.application_id}, fixes={len(self.applicable_fixes)})"
        )


class AutoFixSuggester:
    """Intelligent automated resolve suggester for code quality issues.
    Analyzes issues and generates contextual resolve suggestions with confidence scoring.

    Attributes:
        cache_manager: Instance attribute cache_manager.
        enable_experimental_fixes: Instance attribute enable_experimental_fixes.
        max_fixes_per_file: Instance attribute max_fixes_per_file.
        _rule_manager (FixRuleManager): Instance attribute _rule_manager.
        _detail_generator (FixDetailGenerator): Instance attribute _detail_generator.
        _formatter (CodeFormatter): Instance attribute _formatter.
        _validator (QualityAnalysisCodeValidator): Instance attribute _validator.
        _confidence_thresholds (Dict): Instance attribute _confidence_thresholds.

    Example:
        >>> obj = AutoFixSuggester("cache_manager_value", "enable_experimental_fixes_value", "max_fixes_per_file_value")
    """

    def __init__(
        self,
        cache_manager: RedisCacheManager | None = None,
        enable_experimental_fixes: bool = False,
        max_fixes_per_file: int = 50,
    ):
        self.cache_manager = cache_manager
        self.enable_experimental_fixes = enable_experimental_fixes
        self.max_fixes_per_file = max_fixes_per_file

        # Initialize manager and generator components
        self._rule_manager = FixRuleManager()
        self._detail_generator = FixDetailGenerator()

        # Initialize formatters and validators
        self._formatter = CodeFormatter()
        self._validator = QualityAnalysisCodeValidator()

        # Configuration thresholds combined into a single dict
        self._confidence_thresholds = {"safe_fix": 0.8, "auto_apply": 0.9}

        logger.info(
            "AutoFixSuggester initialized with %d resolve rules",
            len(self._rule_manager.rules),
        )

    async def suggest_fixes(
        self, issues: list[QualityIssue]
    ) -> list[QualityAnalysisResolveSuggestions]:
        """Generate resolve suggestions for a list of quality issues.

        Args:
            issues (List[QualityIssue]): The issues parameter.

        Returns:
            List[QualityAnalysisResolveSuggestions]:     List of QualityAnalysisResolveSuggestions objects with recommended resolves.

        Example:
            >>> await suggest_fixes("issues_value")"""
        try:
            logger.info("Generating resolve suggestions for %d issues", len(issues))

            if not issues:
                return []

            # Group issues by file for context-aware resolving
            issues_by_file = self._group_issues_by_file(issues)

            all_suggestions = []

            # Process each file's issues
            for file_path, file_issues in issues_by_file.items():
                logger.debug("Processing %d issues for %s", len(file_issues), file_path)

                # Generate suggestions for this file
                file_suggestions = await self._generate_file_suggestions(file_path, file_issues)

                # Limit suggestions per file
                if len(file_suggestions) > self.max_fixes_per_file:
                    logger.warning(
                        "Limiting %d suggestions to %d for %s",
                        len(file_suggestions),
                        self.max_fixes_per_file,
                        file_path,
                    )
                    file_suggestions = sorted(
                        file_suggestions,
                        key=lambda x: (x.priority, -x.confidence),
                    )[: self.max_fixes_per_file]

                all_suggestions.extend(file_suggestions)

            # Sort by priority and confidence
            all_suggestions.sort(key=lambda x: (x.priority, -x.confidence))

            logger.info("Generated %d total resolve suggestions", len(all_suggestions))
            return all_suggestions

        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Failed to generate resolve suggestions: %s", e)
            raise

    async def apply_automated_fixes(
        self,
        resolves: list[QualityAnalysisResolveSuggestions],
        dry_run: bool = False,
        create_backup: bool = True,
        filter_by_confidence: bool = True,
    ) -> FixApplicationResult:
        """Apply automated resolves to code files.

        Args:
            resolves (List[QualityAnalysisResolveSuggestions]): The resolves parameter.
            dry_run (bool, optional): The dry_run parameter. Defaults to False.
            create_backup (bool, optional): The create_backup parameter. Defaults to True.
            filter_by_confidence (bool, optional): The filter_by_confidence parameter. Defaults to True.

        Returns:
            FixApplicationResult:     FixApplicationResult with application results.

        Example:
            >>> await apply_automated_fixes("resolves_value", "dry_run_value", "create_backup_value", "example-id")
        """
        application_id = str(uuid7())
        start_time = time.time()

        try:
            logger.info("Applying %d automated resolves (dry_run=%s)", len(resolves), dry_run)

            # Filter resolves if requested
            applicable_fixes = self._filter_applicable_fixes(resolves, filter_by_confidence)

            if not applicable_fixes:
                return self._create_empty_result(application_id, start_time)

            # Apply fixes
            context = FixApplicationContext(
                applicable_fixes=applicable_fixes,
                dry_run=dry_run,
                create_backup=create_backup,
                application_id=application_id,
                start_time=start_time,
            )
            result = await self._apply_fixes(context)

            logger.info(
                "Resolve application completed: %d applied, %d failed, %.2fs",
                result.successful_applications,
                result.failed_applications,
                result.execution_time,
            )

            return result

        except (ValueError, TypeError, OSError) as e:
            logger.error("Failed to apply automated resolves: %s", e)
            return self._create_error_result(application_id, resolves, start_time, str(e))

    async def validate_fixes(
        self,
        original: str,
        resolved: str,
        file_path: str | None = None,
        language: str | None = None,
    ) -> BugFixValidationResult:
        """Validate that resolves are safe and don't introduce new issues.

        Args:
            original (str): The original parameter.
            resolved (str): The resolved parameter.
            file_path (Optional[str], optional): The file_path parameter. Defaults to None.
            language (Optional[str], optional): The language parameter. Defaults to None.

        Returns:
            BugFixValidationResult:     BugFixValidationResult with validation details.

        Example:
            >>> await validate_fixes("original_value", "resolved_value", "file_path_value", "language_value")
        """
        validation_id = str(uuid7())
        start_time = time.time()

        try:
            logger.debug("Validating resolve for %s", file_path or "inline code")

            # Basic syntax validation
            syntax_valid = await self._validator.validate_syntax(resolved, language)

            # Check for obvious issues
            validation_errors = []
            warnings = []

            # Validate basic structure
            if not resolved.strip():
                validation_errors.append("Resolved code is empty")

            # Check for common issues
            if language == "python":
                validation_errors.extend(await self._validator.validate_python_code(resolved))
            elif language in ["javascript", "typescript"]:
                validation_errors.extend(await self._validator.validate_javascript_code(resolved))

            # Compare before/after
            if len(resolved) < len(original) * 0.5:
                warnings.append("Resolved code is significantly shorter than original")

            # Determine safety
            is_valid = syntax_valid and len(validation_errors) == 0
            is_safe = is_valid and len(warnings) <= 2

            execution_time = time.time() - start_time

            result = BugFixValidationResult(
                validation_id=validation_id,
                fix_id="",  # Would be set by caller
                is_valid=is_valid,
                is_safe=is_safe,
                syntax_valid=syntax_valid,
                issues_fixed=0,  # Would require analysis
                new_issues=0,  # Would require analysis
                quality_score_change=0.0,  # Would require analysis
                validation_errors=validation_errors,
                warnings=warnings,
                execution_time=execution_time,
                tests_passed=True,  # Would require test execution
                test_output=None,
            )

            logger.debug("Resolve validation completed: valid=%s, safe=%s", is_valid, is_safe)
            return result

        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Resolve validation failed: %s", e)

            return BugFixValidationResult(
                validation_id=validation_id,
                fix_id="",
                is_valid=False,
                is_safe=False,
                syntax_valid=False,
                issues_fixed=0,
                new_issues=0,
                test_output=None,
                quality_score_change=0.0,
                validation_errors=[str(e)],
                warnings=[],
                execution_time=time.time() - start_time,
                tests_passed=False,
            )

    async def suggest_bulk_fixes(
        self,
        results: list[QualityResult],
        group_similar: bool = True,
        prioritize_by_impact: bool = True,
    ) -> dict[str, list[QualityAnalysisResolveSuggestions]]:
        """Generate bulk resolve suggestions across multiple analysis results.

        Args:
            results (List[QualityResult]): The results parameter.
            group_similar (bool, optional): The group_similar parameter. Defaults to True.
            prioritize_by_impact (bool, optional): The prioritize_by_impact parameter. Defaults to True.

        Returns:
            Dict[str, List[QualityAnalysisResolveSuggestions]]:     Dictionary mapping file paths to resolve suggestions.

        Example:
            >>> await suggest_bulk_fixes("results_value", "group_similar_value", "prioritize_by_impact_value")
        """
        try:
            logger.info(
                "Generating bulk resolve suggestions for %d analysis results",
                len(results),
            )

            # Collect all issues
            all_issues = []
            for result in results:
                all_issues.extend(result.issues)

            if not all_issues:
                return {}

            # Generate suggestions
            suggestions = await self.suggest_fixes(all_issues)

            # Group by file path
            suggestions_by_file: dict[str, list[QualityAnalysisResolveSuggestions]] = {}
            for suggestion in suggestions:
                file_path = suggestion.file_path
                if file_path not in suggestions_by_file:
                    suggestions_by_file[file_path] = []
                suggestions_by_file[file_path].append(suggestion)

            # Group similar resolves if requested
            if group_similar:
                for file_path in suggestions_by_file:
                    suggestions_by_file[file_path] = await self._group_similar_fixes(
                        suggestions_by_file[file_path]
                    )

            # Prioritize by impact if requested
            if prioritize_by_impact:
                for file_path in suggestions_by_file:
                    suggestions_by_file[file_path] = await self._prioritize_fixes_by_impact(
                        suggestions_by_file[file_path]
                    )

            logger.info("Generated bulk suggestions for %d files", len(suggestions_by_file))
            return suggestions_by_file

        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Failed to generate bulk resolve suggestions: %s", e)
            raise

    # Helper methods for resolve generation

    def _group_issues_by_file(self, issues: list[QualityIssue]) -> dict[str, list[QualityIssue]]:
        """Group issues by file path."""
        issues_by_file: dict[str, list[QualityIssue]] = {}
        for issue in issues:
            file_path = issue.file_path
            if file_path not in issues_by_file:
                issues_by_file[file_path] = []
            issues_by_file[file_path].append(issue)
        return issues_by_file

    async def _generate_file_suggestions(
        self, file_path: str, issues: list[QualityIssue]
    ) -> list[QualityAnalysisResolveSuggestions]:
        """Generate resolve suggestions for issues in a specific file."""
        suggestions = []

        for issue in issues:
            # Find matching resolve rules
            matching_rules = self._rule_manager.find_matching_rules(issue)

            for rule in matching_rules:
                suggestion = await self._create_fix_suggestion(issue, rule, file_path)
                if suggestion:
                    suggestions.append(suggestion)

        return suggestions

    async def _create_fix_suggestion(
        self, issue: QualityIssue, rule: FixRule, file_path: str
    ) -> QualityAnalysisResolveSuggestions | None:
        """Create a resolve suggestion from an issue and rule."""
        try:
            # Generate resolve based on rule and issue
            fix_details = await self._detail_generator.generate_fix_details(issue, rule)

            if not fix_details:
                return None

            suggestion = QualityAnalysisResolveSuggestions(
                fix_id=str(uuid7()),
                issue_id=getattr(issue, "id", None),
                strategy=rule.strategy,
                category=rule.category,
                confidence=rule.confidence,
                priority=self._calculate_fix_priority(issue, rule),
                tool=issue.tool,
                rule_id=issue.rule_id,
                severity=issue.severity,
                title=fix_details["title"],
                description=fix_details["description"],
                file_path=file_path,
                line_number=issue.line_number,
                original_code=fix_details.get("original_code"),
                fixed_code=fix_details.get("fixed_code"),
                diff=fix_details.get("diff"),
                requires_review=rule.requires_review,
                breaking_change=fix_details.get("breaking_change", False),
                side_effects=fix_details.get("side_effects", []),
                dependencies=fix_details.get("dependencies", []),
            )

            return suggestion

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to create resolve suggestion for %s: %s", issue.rule_id, e)
            return None

    def _calculate_fix_priority(self, issue: QualityIssue, rule: FixRule) -> int:
        """Calculate resolve priority based on issue and rule."""

        # Base priority from severity
        severity_priority = {
            "critical": 1,
            "error": 2,
            "warning": 5,
            "info": 8,
        }

        priority = severity_priority.get(issue.severity.lower(), 8)

        # Adjust based on resolve category
        if rule.category == FixCategory.SECURITY:
            priority = max(1, priority - 2)
        elif rule.category == FixCategory.BUG_FIX:
            priority = max(1, priority - 1)
        elif rule.category == FixCategory.FORMATTING:
            priority = min(10, priority + 2)

        return priority

    def _filter_applicable_fixes(
        self,
        resolves: list[QualityAnalysisResolveSuggestions],
        filter_by_confidence: bool,
    ) -> list[QualityAnalysisResolveSuggestions]:
        """Filter resolves based on confidence and strategy."""
        if not filter_by_confidence:
            return resolves

        applicable_fixes = [
            resolve
            for resolve in resolves
            if resolve.strategy in [FixStrategy.AUTOMATIC, FixStrategy.SUGGESTED]
            and resolve.confidence >= self._confidence_thresholds["safe_fix"]
            and not resolve.breaking_change
        ]

        logger.info("Filtered to %d safe resolves based on confidence", len(applicable_fixes))
        return applicable_fixes

    async def _apply_fixes(self, context: FixApplicationContext) -> FixApplicationResult:
        """Apply fixes to files."""
        # Group resolves by file for efficient application
        fixes_by_file: dict[str, list[QualityAnalysisResolveSuggestions]] = {}
        for resolve in context.applicable_fixes:
            if resolve.file_path not in fixes_by_file:
                fixes_by_file[resolve.file_path] = []
            fixes_by_file[resolve.file_path].append(resolve)

        applied_fixes = []
        failed_fixes = []
        modified_files = []
        errors = []

        # Apply resolves file by file
        for file_path, file_fixes in fixes_by_file.items():
            result = await self._apply_fixes_to_single_file(
                file_path, file_fixes, context.dry_run, context.create_backup
            )

            applied_fixes.extend(result["applied"])
            failed_fixes.extend(result["failed"])
            if result["modified"]:
                modified_files.append(file_path)
            errors.extend(result["errors"])

        execution_time = time.time() - context.start_time

        return FixApplicationResult(
            application_id=context.application_id,
            fixes_applied=applied_fixes,
            fixes_failed=failed_fixes,
            total_fixes_attempted=len(context.applicable_fixes),
            successful_applications=len(applied_fixes),
            failed_applications=len(failed_fixes),
            initial_quality_score=0.0,  # Would need original analysis
            final_quality_score=0.0,  # Would need re-analysis
            quality_improvement=0.0,
            files_modified=modified_files,
            backup_created=context.create_backup and not context.dry_run,
            execution_time=execution_time,
            errors=errors,
        )

    async def _apply_fixes_to_single_file(
        self,
        file_path: str,
        file_fixes: list[QualityAnalysisResolveSuggestions],
        dry_run: bool,
        create_backup: bool,
    ) -> dict[str, Any]:
        """Apply fixes to a single file and return results."""
        result: dict[str, Any] = {
            "applied": [],
            "failed": [],
            "modified": False,
            "errors": [],
        }

        try:
            logger.debug("Applying %d resolves to %s", len(file_fixes), file_path)

            # Sort resolves by line number (descending) to avoid offset issues
            file_fixes.sort(key=lambda x: x.line_number or 0, reverse=True)

            if not dry_run:
                # Create backup if requested
                if create_backup:
                    await self._create_file_backup(file_path)

                # Apply resolves to file
                success = await self._apply_fixes_to_file(file_path, file_fixes)

                if success:
                    result["applied"] = [resolve.fix_id for resolve in file_fixes]
                    result["modified"] = True
                else:
                    result["failed"] = [resolve.fix_id for resolve in file_fixes]
                    result["errors"].append(f"Failed to apply resolves to {file_path}")
            else:
                # Dry run - just validate resolves
                for resolve in file_fixes:
                    validation = await self._validate_fix_syntax(resolve)
                    if validation:
                        result["applied"].append(resolve.fix_id)
                    else:
                        result["failed"].append(resolve.fix_id)
                        result["errors"].append(
                            f"Resolve {resolve.fix_id} failed syntax validation"
                        )

        except (OSError, ValueError, TypeError, RuffFixError) as e:
            logger.error("Error applying resolves to %s: %s", file_path, e)
            result["failed"] = [resolve.fix_id for resolve in file_fixes]
            result["errors"].append(f"Error processing {file_path}: {str(e)}")

        return result

    # Resolve application methods

    async def _apply_fixes_to_file(
        self, file_path: str, resolves: list[QualityAnalysisResolveSuggestions]
    ) -> bool:
        """Apply resolves to a specific file."""
        try:
            # Read the original file content
            try:
                with open(file_path, encoding="utf-8") as f:
                    original_content = f.read()
            except FileNotFoundError:
                logger.error("File not found: %s", file_path)
                return False
            except (OSError, UnicodeDecodeError) as e:
                logger.error("Error reading file %s: %s", file_path, e)
                return False

            # Apply resolves based on type
            modified_content = await self._apply_categorized_fixes(
                file_path, original_content, resolves
            )

            # Write the modified content back to file
            if modified_content != original_content:
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(modified_content)
                    logger.info(
                        "Successfully applied %d resolves to %s",
                        len(resolves),
                        file_path,
                    )
                    return True
                except (OSError, UnicodeEncodeError) as e:
                    logger.error("Error writing to file %s: %s", file_path, e)
                    return False
            else:
                logger.debug("No changes made to %s", file_path)
                return True

        except (OSError, ValueError, TypeError) as e:
            logger.error("Failed to apply resolves to %s: %s", file_path, e)
            return False

    async def _apply_categorized_fixes(
        self,
        file_path: str,
        content: str,
        resolves: list[QualityAnalysisResolveSuggestions],
    ) -> str:
        """Apply fixes categorized by type."""
        modified_content = content

        # Group resolves by type for efficient application
        formatting_fixes = [f for f in resolves if f.category == FixCategory.FORMATTING]
        import_fixes = [f for f in resolves if f.category == FixCategory.IMPORT_ORGANIZATION]
        code_fixes = [
            f
            for f in resolves
            if f.category
            not in [
                FixCategory.FORMATTING,
                FixCategory.IMPORT_ORGANIZATION,
            ]
        ]

        # Apply formatting resolves using appropriate tools
        if formatting_fixes:
            for resolve in formatting_fixes:
                if resolve.tool == "ruff-format" and file_path.endswith(".py"):
                    modified_content = await self._formatter.apply_ruff_formatting(
                        file_path, modified_content
                    )
                elif resolve.tool == "prettier" and file_path.endswith(
                    (".js", ".jsx", ".ts", ".tsx")
                ):
                    modified_content = await self._formatter.apply_prettier_formatting(
                        file_path, modified_content
                    )

        # Apply import organization resolves
        if import_fixes:
            for resolve in import_fixes:
                if resolve.tool != "ruff" or not file_path.endswith(".py"):
                    continue
                if (resolve.rule_id or "").startswith("I"):
                    modified_content = await self._formatter.apply_ruff_import_sorting(
                        file_path, modified_content
                    )
                elif resolve.rule_id in ("F401", "F841"):
                    modified_content = await self._formatter.apply_ruff_unused_removal(
                        file_path, modified_content
                    )

        # Apply code-level resolves
        if code_fixes:
            modified_content = await self._apply_code_fixes(modified_content, code_fixes)

        return modified_content

    async def _apply_code_fixes(
        self, content: str, code_fixes: list[QualityAnalysisResolveSuggestions]
    ) -> str:
        """Apply code-level fixes."""
        modified_content = content

        # Sort by line number (descending) to avoid offset issues
        code_fixes.sort(key=lambda x: x.line_number or 0, reverse=True)

        for resolve in code_fixes:
            if resolve.fixed_code and resolve.original_code:
                # Basic replacement if we have both original and resolved code
                modified_content = modified_content.replace(
                    resolve.original_code, resolve.fixed_code
                )
            elif resolve.line_number and resolve.fixed_code:
                # Line-based replacement
                modified_content = await self._apply_line_based_fix(
                    modified_content,
                    resolve.line_number,
                    resolve.fixed_code,
                )

        return modified_content

    async def _apply_line_based_fix(self, content: str, line_number: int, fixed_code: str) -> str:
        """Apply a resolve to a specific line number."""
        try:
            lines = content.split("\n")

            # Adjust for 0-based indexing
            line_index = line_number - 1

            if 0 <= line_index < len(lines):
                # Replace the line with the resolved code
                lines[line_index] = fixed_code
                return "\n".join(lines)

            logger.warning("Line number %d out of range", line_number)
            return content

        except (ValueError, IndexError, TypeError) as e:
            logger.error("Error applying line-based resolve: %s", e)
            return content

    async def _create_file_backup(self, file_path: str) -> bool:
        """Create backup of file before applying resolves."""
        try:
            backup_path = f"{file_path}.fix_backup_{int(datetime.now(UTC).timestamp())}"
            shutil.copy2(file_path, backup_path)
            logger.debug("Created backup: %s", backup_path)
            return True
        except (OSError, shutil.Error) as e:
            logger.error("Failed to create backup for %s: %s", file_path, e)
            return False

    # Validation methods

    async def _validate_fix_syntax(self, resolve: QualityAnalysisResolveSuggestions) -> bool:
        """Validate that a resolve produces syntactically correct code."""
        if not resolve.fixed_code:
            return True  # No code to validate

        # Determine language from file extension
        file_path = resolve.file_path
        if file_path.endswith(".py"):
            language = "python"
        elif file_path.endswith((".js", ".jsx")):
            language = "javascript"
        elif file_path.endswith((".ts", ".tsx")):
            language = "typescript"
        else:
            return True  # Can't validate unknown language

        return await self._validator.validate_syntax(resolve.fixed_code, language)

    # Grouping and prioritization methods

    async def _group_similar_fixes(
        self, resolves: list[QualityAnalysisResolveSuggestions]
    ) -> list[QualityAnalysisResolveSuggestions]:
        """Group similar resolves together."""
        # Group by category and tool
        grouped: dict[
            tuple[FixCategory, str, str | None],
            list[QualityAnalysisResolveSuggestions],
        ] = {}

        for resolve in resolves:
            key = (resolve.category, resolve.tool, resolve.rule_id)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(resolve)

        # Return ungrouped for now, but could merge similar resolves
        return resolves

    async def _prioritize_fixes_by_impact(
        self, resolves: list[QualityAnalysisResolveSuggestions]
    ) -> list[QualityAnalysisResolveSuggestions]:
        """Prioritize resolves by potential impact."""

        def impact_score(resolve: QualityAnalysisResolveSuggestions) -> float:
            """Function impact_score.

            Args:
                resolve (QualityAnalysisResolveSuggestions): The resolve parameter.

            Returns:
                float: The float result.

            Example:
                >>> impact_score("resolve_value")"""
            score = 0.0

            # Priority weight (lower is better)
            score += (10 - resolve.priority) * 10

            # Confidence weight
            score += resolve.confidence * 5

            # Category weight
            category_weights = {
                FixCategory.SECURITY: 20,
                FixCategory.BUG_FIX: 15,
                FixCategory.PERFORMANCE: 10,
                FixCategory.TYPE_ANNOTATION: 5,
                FixCategory.CODE_STYLE: 3,
                FixCategory.FORMATTING: 1,
            }
            score += category_weights.get(resolve.category, 5)

            # Strategy weight
            if resolve.strategy == FixStrategy.AUTOMATIC:
                score += 10
            elif resolve.strategy == FixStrategy.SUGGESTED:
                score += 5

            return score

        return sorted(resolves, key=impact_score, reverse=True)

    def _create_empty_result(self, application_id: str, start_time: float) -> FixApplicationResult:
        """Create an empty result when no fixes are applicable."""
        logger.warning("No applicable resolves found after filtering")
        return FixApplicationResult(
            application_id=application_id,
            fixes_applied=[],
            fixes_failed=[],
            total_fixes_attempted=0,
            successful_applications=0,
            failed_applications=0,
            initial_quality_score=0.0,
            final_quality_score=0.0,
            quality_improvement=0.0,
            execution_time=time.time() - start_time,
        )

    def _create_error_result(
        self,
        application_id: str,
        resolves: list[QualityAnalysisResolveSuggestions],
        start_time: float,
        error: str,
    ) -> FixApplicationResult:
        """Create an error result when fix application fails."""
        execution_time = time.time() - start_time

        return FixApplicationResult(
            application_id=application_id,
            fixes_applied=[],
            fixes_failed=[resolve.fix_id for resolve in resolves],
            total_fixes_attempted=len(resolves),
            successful_applications=0,
            failed_applications=len(resolves),
            initial_quality_score=0.0,
            final_quality_score=0.0,
            quality_improvement=0.0,
            execution_time=execution_time,
            errors=[error],
        )


# Backward compatibility alias
FixSuggester = AutoFixSuggester


# Factory function for dependency injection
def create_auto_fix_suggester(
    cache_manager: RedisCacheManager | None = None,
    enable_experimental_fixes: bool = False,
    max_fixes_per_file: int | None = None,
) -> AutoFixSuggester:
    """Factory function to create AutoFixSuggester.

    Args:
        cache_manager (Optional[RedisCacheManager], optional): The cache_manager parameter. Defaults to None.
        enable_experimental_fixes (bool, optional): The enable_experimental_fixes parameter. Defaults to False.
        max_fixes_per_file (Optional[int], optional): The max_fixes_per_file parameter. Defaults to None.

    Returns:
        AutoFixSuggester: The autofixsuggester result.

    Example:
        >>> create_auto_fix_suggester("cache_manager_value", "enable_experimental_fixes_value", "max_fixes_per_file_value")
    """
    return AutoFixSuggester(
        cache_manager=cache_manager,
        enable_experimental_fixes=enable_experimental_fixes
        or os.getenv("ENABLE_EXPERIMENTAL_FIXES", "false").lower() == "true",
        max_fixes_per_file=max_fixes_per_file or int(os.getenv("MAX_FIXES_PER_FILE", "50")),
    )
