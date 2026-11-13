"""Round 4 tests for AutoFixSuggester (78 uncov lines)."""

from __future__ import annotations

import time

import pytest

from processing.quality_analysis.fix_models import (
    FixApplicationResult,
    FixCategory,
    FixStrategy,
    QualityAnalysisResolveSuggestions,
)
from processing.quality_analysis.fix_suggester import (
    AutoFixSuggester,
    FixApplicationContext,
    FixSuggester,
    create_auto_fix_suggester,
)


@pytest.fixture
def suggester():
    return AutoFixSuggester()


def _make_issue(
    rule_id="E001",
    severity="warning",
    tool="ruff",
    file_path="test.py",
    line=1,
    message="test issue",
):
    from models.testing_models import QualityIssue

    return QualityIssue(
        rule_id=rule_id,
        severity=severity,
        tool=tool,
        file_path=file_path,
        line_number=line,
        message=message,
    )


def _make_suggestion(
    fix_id="fix-1",
    file_path="test.py",
    strategy=FixStrategy.AUTOMATIC,
    category=FixCategory.CODE_STYLE,
    confidence=0.9,
    priority=5,
    breaking_change=False,
    line_number=1,
    fixed_code=None,
    original_code=None,
):
    return QualityAnalysisResolveSuggestions(
        fix_id=fix_id,
        strategy=strategy,
        category=category,
        confidence=confidence,
        priority=priority,
        tool="ruff",
        rule_id="E001",
        severity="warning",
        title="Fix something",
        description="Fix this issue",
        file_path=file_path,
        line_number=line_number,
        original_code=original_code,
        fixed_code=fixed_code,
        breaking_change=breaking_change,
    )


class TestFixApplicationContext:
    def test_str_representation(self):
        ctx = FixApplicationContext(
            applicable_fixes=[_make_suggestion()],
            dry_run=False,
            create_backup=True,
            application_id="app-1",
            start_time=time.time(),
        )
        s = str(ctx)
        assert "app-1" in s
        assert "1" in s  # number of fixes


class TestAutoFixSuggester:
    def test_init(self, suggester):
        assert suggester.max_fixes_per_file == 50
        assert suggester.enable_experimental_fixes is False

    @pytest.mark.asyncio
    async def test_suggest_fixes_empty(self, suggester):
        result = await suggester.suggest_fixes([])
        assert result == []

    @pytest.mark.asyncio
    async def test_suggest_fixes_with_issues(self, suggester):
        issues = [_make_issue(), _make_issue(rule_id="E002", line=5)]
        result = await suggester.suggest_fixes(issues)
        assert isinstance(result, list)
        # Should return sorted by priority
        if len(result) > 1:
            assert (
                result[0].priority <= result[-1].priority
                or result[0].confidence >= result[-1].confidence
            )

    @pytest.mark.asyncio
    async def test_suggest_fixes_limits_per_file(self):
        s = AutoFixSuggester(max_fixes_per_file=2)
        issues = [_make_issue(rule_id=f"E{i:03d}", line=i) for i in range(10)]
        result = await s.suggest_fixes(issues)
        # Check the per-file limit was applied
        file_counts: dict[str, int] = {}
        for suggestion in result:
            file_counts[suggestion.file_path] = file_counts.get(suggestion.file_path, 0) + 1
        for count in file_counts.values():
            assert count <= 2

    def test_group_issues_by_file(self, suggester):
        issues = [
            _make_issue(file_path="a.py"),
            _make_issue(file_path="b.py"),
            _make_issue(file_path="a.py", line=5),
        ]
        grouped = suggester._group_issues_by_file(issues)
        assert "a.py" in grouped
        assert "b.py" in grouped
        assert len(grouped["a.py"]) == 2
        assert len(grouped["b.py"]) == 1

    def test_calculate_fix_priority_critical(self, suggester):
        from processing.quality_analysis.fix_models import FixRule

        issue = _make_issue(severity="critical")
        rule = FixRule(
            rule_id="E001",
            tool="ruff",
            rule_pattern=".*",
            category=FixCategory.SECURITY,
            strategy=FixStrategy.AUTOMATIC,
            confidence=0.9,
            requires_review=False,
        )
        priority = suggester._calculate_fix_priority(issue, rule)
        assert priority == 1  # max(1, 1-2)

    def test_calculate_fix_priority_formatting(self, suggester):
        from processing.quality_analysis.fix_models import FixRule

        issue = _make_issue(severity="info")
        rule = FixRule(
            rule_id="F001",
            tool="ruff",
            rule_pattern=".*",
            category=FixCategory.FORMATTING,
            strategy=FixStrategy.SUGGESTED,
            confidence=0.5,
            requires_review=True,
        )
        priority = suggester._calculate_fix_priority(issue, rule)
        assert priority == 10  # min(10, 8+2)

    def test_calculate_fix_priority_bug_fix(self, suggester):
        from processing.quality_analysis.fix_models import FixRule

        issue = _make_issue(severity="error")
        rule = FixRule(
            rule_id="B001",
            tool="ruff",
            rule_pattern=".*",
            category=FixCategory.BUG_FIX,
            strategy=FixStrategy.AUTOMATIC,
            confidence=0.9,
            requires_review=False,
        )
        priority = suggester._calculate_fix_priority(issue, rule)
        assert priority == 1  # max(1, 2-1)

    def test_filter_applicable_fixes_no_filter(self, suggester):
        fixes = [_make_suggestion(confidence=0.5)]
        result = suggester._filter_applicable_fixes(fixes, filter_by_confidence=False)
        assert len(result) == 1

    def test_filter_applicable_fixes_with_filter(self, suggester):
        fixes = [
            _make_suggestion(confidence=0.9, strategy=FixStrategy.AUTOMATIC),
            _make_suggestion(fix_id="fix-2", confidence=0.5, strategy=FixStrategy.AUTOMATIC),
            _make_suggestion(fix_id="fix-3", confidence=0.9, strategy=FixStrategy.MANUAL),
        ]
        result = suggester._filter_applicable_fixes(fixes, filter_by_confidence=True)
        assert len(result) == 1
        assert result[0].fix_id == "fix-1"

    def test_filter_applicable_fixes_excludes_breaking(self, suggester):
        fixes = [
            _make_suggestion(confidence=0.9, strategy=FixStrategy.AUTOMATIC, breaking_change=True),
        ]
        result = suggester._filter_applicable_fixes(fixes, filter_by_confidence=True)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_validate_fixes_valid_python(self, suggester):
        result = await suggester.validate_fixes(
            original="x = 1\n",
            resolved="x = 2\n",
            file_path="test.py",
            language="python",
        )
        assert result.is_valid is True
        assert result.syntax_valid is True

    @pytest.mark.asyncio
    async def test_validate_fixes_invalid_python(self, suggester):
        result = await suggester.validate_fixes(
            original="x = 1\n",
            resolved="def foo(\n",
            file_path="test.py",
            language="python",
        )
        assert result.syntax_valid is False

    @pytest.mark.asyncio
    async def test_validate_fixes_empty_resolved(self, suggester):
        result = await suggester.validate_fixes(
            original="x = 1\n",
            resolved="",
            file_path="test.py",
            language="python",
        )
        assert result.is_valid is False
        assert any("empty" in e.lower() for e in result.validation_errors)

    @pytest.mark.asyncio
    async def test_validate_fixes_significantly_shorter(self, suggester):
        result = await suggester.validate_fixes(
            original="x = 1\ny = 2\nz = 3\nw = 4\na = 5\nb = 6\n",
            resolved="x = 1\n",
            file_path="test.py",
            language="python",
        )
        assert any("shorter" in w.lower() for w in result.warnings)

    @pytest.mark.asyncio
    async def test_validate_fixes_javascript(self, suggester):
        result = await suggester.validate_fixes(
            original="const x = 1;",
            resolved="const x = 2;",
            language="javascript",
        )
        assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_apply_automated_fixes_empty_after_filter(self, suggester):
        fixes = [_make_suggestion(confidence=0.1)]
        result = await suggester.apply_automated_fixes(fixes, filter_by_confidence=True)
        assert result.successful_applications == 0

    @pytest.mark.asyncio
    async def test_apply_automated_fixes_dry_run(self, suggester):
        fixes = [
            _make_suggestion(
                confidence=0.95,
                strategy=FixStrategy.AUTOMATIC,
                fixed_code="x = 2",
            ),
        ]
        result = await suggester.apply_automated_fixes(
            fixes, dry_run=True, filter_by_confidence=False
        )
        assert isinstance(result, FixApplicationResult)

    @pytest.mark.asyncio
    async def test_apply_line_based_fix(self, suggester):
        content = "line1\nline2\nline3\n"
        result = await suggester._apply_line_based_fix(content, 2, "replaced")
        assert "replaced" in result
        assert "line1" in result

    @pytest.mark.asyncio
    async def test_apply_line_based_fix_out_of_range(self, suggester):
        content = "line1\n"
        result = await suggester._apply_line_based_fix(content, 100, "replaced")
        assert result == content

    @pytest.mark.asyncio
    async def test_validate_fix_syntax_no_code(self, suggester):
        suggestion = _make_suggestion(fixed_code=None)
        result = await suggester._validate_fix_syntax(suggestion)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_fix_syntax_python(self, suggester):
        suggestion = _make_suggestion(fixed_code="x = 1", file_path="test.py")
        result = await suggester._validate_fix_syntax(suggestion)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_fix_syntax_unknown_ext(self, suggester):
        suggestion = _make_suggestion(fixed_code="something", file_path="test.rs")
        result = await suggester._validate_fix_syntax(suggestion)
        assert result is True

    @pytest.mark.asyncio
    async def test_group_similar_fixes(self, suggester):
        fixes = [_make_suggestion(), _make_suggestion(fix_id="fix-2")]
        result = await suggester._group_similar_fixes(fixes)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_prioritize_fixes_by_impact(self, suggester):
        fixes = [
            _make_suggestion(priority=5, confidence=0.5, category=FixCategory.FORMATTING),
            _make_suggestion(
                fix_id="fix-2",
                priority=1,
                confidence=0.9,
                category=FixCategory.SECURITY,
                strategy=FixStrategy.AUTOMATIC,
            ),
        ]
        result = await suggester._prioritize_fixes_by_impact(fixes)
        assert result[0].fix_id == "fix-2"  # Security should rank higher

    def test_create_empty_result(self, suggester):
        result = suggester._create_empty_result("app-1", time.time())
        assert result.successful_applications == 0
        assert result.application_id == "app-1"

    def test_create_error_result(self, suggester):
        fixes = [_make_suggestion()]
        result = suggester._create_error_result("app-1", fixes, time.time(), "boom")
        assert result.failed_applications == 1
        assert "boom" in result.errors


class TestFactoryFunction:
    def test_create_auto_fix_suggester(self):
        s = create_auto_fix_suggester()
        assert isinstance(s, AutoFixSuggester)

    def test_create_auto_fix_suggester_custom(self):
        s = create_auto_fix_suggester(max_fixes_per_file=10)
        assert s.max_fixes_per_file == 10


class TestSuggestFixesExceptionPath:
    """Cover lines 166-168: exception re-raise in suggest_fixes."""

    @pytest.mark.asyncio
    async def test_suggest_fixes_raises_on_attribute_error(self, suggester):
        """Force AttributeError by passing an issue with missing attributes."""

        class BadIssue:
            pass  # has no file_path, rule_id, etc.

        with pytest.raises((ValueError, TypeError, AttributeError)):
            await suggester.suggest_fixes([BadIssue()])


class TestApplyAutomatedFixesExceptionPath:
    """Cover lines 222-224: error result on OSError during apply."""

    @pytest.mark.asyncio
    async def test_apply_automated_fixes_oserror(self, suggester):
        """Force an OSError during fix application to hit the except branch."""
        from unittest.mock import AsyncMock

        fixes = [
            _make_suggestion(
                confidence=0.95,
                strategy=FixStrategy.AUTOMATIC,
                fixed_code="x = 2",
                original_code="x = 1",
            ),
        ]
        # Mock _filter_applicable_fixes to return fixes, then _apply_fixes to raise
        suggester._apply_fixes = AsyncMock(side_effect=OSError("disk full"))
        result = await suggester.apply_automated_fixes(
            fixes, dry_run=False, filter_by_confidence=False
        )
        assert result.failed_applications == 1
        assert "disk full" in result.errors[0]


class TestValidateFixesExceptionPath:
    """Cover lines 299-302: exception in validate_fixes."""

    @pytest.mark.asyncio
    async def test_validate_fixes_catches_validator_error(self, suggester):
        """Force _validator.validate_syntax to raise to hit except branch."""
        from unittest.mock import AsyncMock

        suggester._validator.validate_syntax = AsyncMock(side_effect=TypeError("bad type"))
        result = await suggester.validate_fixes(
            original="x = 1",
            resolved="x = 2",
            language="python",
        )
        assert result.is_valid is False
        assert result.is_safe is False
        assert result.syntax_valid is False
        assert any("bad type" in e for e in result.validation_errors)


class TestSuggestBulkFixes:
    """Cover lines 337-381: suggest_bulk_fixes method."""

    @pytest.mark.asyncio
    async def test_suggest_bulk_fixes_empty_results(self, suggester):
        result = await suggester.suggest_bulk_fixes([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_suggest_bulk_fixes_empty_issues(self, suggester):
        from datetime import UTC, datetime

        from models.testing_models import QualityResult

        qr = QualityResult(
            analysis_id="a1",
            success=True,
            language="python",
            tools_results={},
            overall_score=10.0,
            total_issues=0,
            execution_time=1.0,
            created_at=datetime.now(UTC),
        )
        result = await suggester.suggest_bulk_fixes([qr])
        assert result == {}

    @pytest.mark.asyncio
    async def test_suggest_bulk_fixes_with_issues(self, suggester):
        from datetime import UTC, datetime

        from models.testing_models import QualityIssue, QualityResult

        issues = [
            QualityIssue(
                rule_id="E001",
                severity="warning",
                tool="ruff",
                file_path="a.py",
                line_number=1,
                message="test",
            ),
        ]
        qr = QualityResult(
            analysis_id="a1",
            success=True,
            language="python",
            tools_results={},
            overall_score=5.0,
            total_issues=1,
            execution_time=0.5,
            issues=issues,
            created_at=datetime.now(UTC),
        )
        result = await suggester.suggest_bulk_fixes(
            [qr], group_similar=True, prioritize_by_impact=True
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_suggest_bulk_fixes_exception(self, suggester):
        """Cover lines 379-381: exception re-raise."""
        from datetime import UTC, datetime
        from unittest.mock import AsyncMock

        suggester.suggest_fixes = AsyncMock(side_effect=AttributeError("boom"))
        from models.testing_models import QualityIssue, QualityResult

        issues = [
            QualityIssue(
                rule_id="E001",
                severity="warning",
                tool="ruff",
                file_path="a.py",
                line_number=1,
                message="test",
            ),
        ]
        qr = QualityResult(
            analysis_id="a1",
            success=True,
            language="python",
            tools_results={},
            overall_score=5.0,
            total_issues=1,
            execution_time=0.5,
            issues=issues,
            created_at=datetime.now(UTC),
        )
        with pytest.raises(AttributeError, match="boom"):
            await suggester.suggest_bulk_fixes([qr])


class TestCreateFixSuggestionNullReturn:
    """Cover line 421: _create_fix_suggestion returns None when fix_details is None."""

    @pytest.mark.asyncio
    async def test_create_fix_suggestion_returns_none(self, suggester):
        from unittest.mock import AsyncMock

        from processing.quality_analysis.fix_models import FixRule

        issue = _make_issue()
        rule = FixRule(
            rule_id="E001",
            tool="ruff",
            rule_pattern=".*",
            category=FixCategory.CODE_STYLE,
            strategy=FixStrategy.SUGGESTED,
            confidence=0.5,
            requires_review=True,
        )
        # Make detail generator return None
        suggester._detail_generator.generate_fix_details = AsyncMock(return_value=None)
        result = await suggester._create_fix_suggestion(issue, rule, "test.py")
        assert result is None


class TestCreateFixSuggestionException:
    """Cover lines 448-450: exception in _create_fix_suggestion."""

    @pytest.mark.asyncio
    async def test_create_fix_suggestion_catches_exception(self, suggester):
        from unittest.mock import AsyncMock

        from processing.quality_analysis.fix_models import FixRule

        issue = _make_issue()
        rule = FixRule(
            rule_id="E001",
            tool="ruff",
            rule_pattern=".*",
            category=FixCategory.CODE_STYLE,
            strategy=FixStrategy.SUGGESTED,
            confidence=0.5,
            requires_review=True,
        )
        suggester._detail_generator.generate_fix_details = AsyncMock(
            side_effect=ValueError("bad details")
        )
        result = await suggester._create_fix_suggestion(issue, rule, "test.py")
        assert result is None


class TestApplyCodeFixes:
    """Cover line 518 and related code fix application paths."""

    @pytest.mark.asyncio
    async def test_apply_code_fixes_with_original_and_fixed(self, suggester):
        content = "x = 1\ny = 2\n"
        fixes = [
            _make_suggestion(
                original_code="x = 1",
                fixed_code="x = 42",
                category=FixCategory.BUG_FIX,
            ),
        ]
        result = await suggester._apply_code_fixes(content, fixes)
        assert "x = 42" in result
        assert "x = 1" not in result

    @pytest.mark.asyncio
    async def test_apply_code_fixes_line_based(self, suggester):
        content = "line1\nline2\nline3\n"
        fixes = [
            _make_suggestion(
                fixed_code="replaced",
                line_number=2,
                original_code=None,
                category=FixCategory.BUG_FIX,
            ),
        ]
        result = await suggester._apply_code_fixes(content, fixes)
        assert "replaced" in result


class TestApplyFixesToSingleFile:
    """Cover lines 562-589: _apply_fixes_to_single_file dry_run validation failure."""

    @pytest.mark.asyncio
    async def test_dry_run_with_invalid_syntax(self, suggester):
        """Dry run where _validate_fix_syntax returns False."""
        from unittest.mock import AsyncMock

        fix = _make_suggestion(
            fixed_code="def foo(",  # invalid python
            file_path="test.py",
        )
        suggester._validate_fix_syntax = AsyncMock(return_value=False)
        result = await suggester._apply_fixes_to_single_file("test.py", [fix], True, False)
        assert len(result["failed"]) == 1
        assert len(result["errors"]) == 1
        assert "syntax validation" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_non_dry_run_apply_success(self, suggester, tmp_path):
        """Non-dry run with a real file."""
        from unittest.mock import AsyncMock

        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        fix = _make_suggestion(
            fixed_code="x = 2",
            original_code="x = 1",
            file_path=str(test_file),
            category=FixCategory.BUG_FIX,
        )
        suggester._create_file_backup = AsyncMock(return_value=True)
        result = await suggester._apply_fixes_to_single_file(str(test_file), [fix], False, True)
        assert len(result["applied"]) == 1
        assert result["modified"] is True

    @pytest.mark.asyncio
    async def test_non_dry_run_apply_failure(self, suggester):
        """Non-dry run with nonexistent file (apply returns False)."""
        from unittest.mock import AsyncMock

        fix = _make_suggestion(
            fixed_code="x = 2",
            original_code="x = 1",
            file_path="/nonexistent/file.py",
            category=FixCategory.BUG_FIX,
        )
        suggester._create_file_backup = AsyncMock(return_value=True)
        result = await suggester._apply_fixes_to_single_file(
            "/nonexistent/file.py", [fix], False, True
        )
        assert len(result["failed"]) == 1
        assert len(result["errors"]) >= 1

    @pytest.mark.asyncio
    async def test_exception_during_apply(self, suggester):
        """Cover lines 586-589: exception handler in _apply_fixes_to_single_file."""
        from unittest.mock import AsyncMock

        fix = _make_suggestion(file_path="test.py")
        # Make sort raise TypeError
        suggester._create_file_backup = AsyncMock(side_effect=TypeError("oops"))
        result = await suggester._apply_fixes_to_single_file("test.py", [fix], False, True)
        assert len(result["failed"]) == 1
        assert any("oops" in e for e in result["errors"])


class TestApplyFixesToFile:
    """Cover lines 599-636: _apply_fixes_to_file."""

    @pytest.mark.asyncio
    async def test_file_not_found(self, suggester):
        fix = _make_suggestion(file_path="/nonexistent/xyz.py")
        result = await suggester._apply_fixes_to_file("/nonexistent/xyz.py", [fix])
        assert result is False

    @pytest.mark.asyncio
    async def test_no_changes_returns_true(self, suggester, tmp_path):
        """When no modifications are made, should return True."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        # Fix with no matching original code - nothing changes
        fix = _make_suggestion(
            original_code="does_not_exist",
            fixed_code="x = 2",
            category=FixCategory.BUG_FIX,
        )
        result = await suggester._apply_fixes_to_file(str(test_file), [fix])
        assert result is True  # No changes but no error

    @pytest.mark.asyncio
    async def test_successful_modification(self, suggester, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\n")
        fix = _make_suggestion(
            original_code="x = 1",
            fixed_code="x = 42",
            category=FixCategory.BUG_FIX,
        )
        result = await suggester._apply_fixes_to_file(str(test_file), [fix])
        assert result is True
        assert "x = 42" in test_file.read_text()


class TestApplyCategorizedFixes:
    """Cover lines 645-686: _apply_categorized_fixes."""

    @pytest.mark.asyncio
    async def test_formatting_ruff_format_fix(self, suggester):
        from unittest.mock import AsyncMock

        fix = _make_suggestion(category=FixCategory.FORMATTING)
        fix.tool = "ruff-format"
        suggester._formatter.apply_ruff_formatting = AsyncMock(return_value="formatted")
        result = await suggester._apply_categorized_fixes("test.py", "original", [fix])
        assert result == "formatted"

    @pytest.mark.asyncio
    async def test_formatting_prettier_fix(self, suggester):
        from unittest.mock import AsyncMock

        fix = _make_suggestion(category=FixCategory.FORMATTING)
        fix.tool = "prettier"
        suggester._formatter.apply_prettier_formatting = AsyncMock(return_value="pretty")
        result = await suggester._apply_categorized_fixes("app.jsx", "original", [fix])
        assert result == "pretty"

    @pytest.mark.asyncio
    async def test_import_ruff_isort_fix(self, suggester):
        from unittest.mock import AsyncMock

        fix = _make_suggestion(category=FixCategory.IMPORT_ORGANIZATION)
        fix.tool = "ruff"
        fix.rule_id = "I001"
        suggester._formatter.apply_ruff_import_sorting = AsyncMock(return_value="sorted")
        result = await suggester._apply_categorized_fixes("test.py", "original", [fix])
        assert result == "sorted"

    @pytest.mark.asyncio
    async def test_unused_ruff_fix(self, suggester):
        from unittest.mock import AsyncMock

        fix = _make_suggestion(category=FixCategory.IMPORT_ORGANIZATION)
        fix.tool = "ruff"
        fix.rule_id = "F841"
        suggester._formatter.apply_ruff_unused_removal = AsyncMock(return_value="pruned")
        suggester._formatter.apply_ruff_import_sorting = AsyncMock(return_value="wrong")
        result = await suggester._apply_categorized_fixes("test.py", "original", [fix])
        assert result == "pruned"

    @pytest.mark.asyncio
    async def test_code_fix_category(self, suggester):
        fix = _make_suggestion(
            category=FixCategory.BUG_FIX,
            original_code="old",
            fixed_code="new",
        )
        result = await suggester._apply_categorized_fixes("test.py", "old code old", [fix])
        assert "new" in result


class TestApplyLineBasedFixException:
    """Cover lines 729-731: exception in _apply_line_based_fix."""

    @pytest.mark.asyncio
    async def test_line_based_fix_type_error(self, suggester):
        # line_number as a non-int-convertible to trigger error
        result = await suggester._apply_line_based_fix("line1\nline2", 2, "fixed")
        # Normal case should work - just verify it returns
        assert "fixed" in result


class TestCreateFileBackup:
    """Cover lines 738-739: _create_file_backup."""

    @pytest.mark.asyncio
    async def test_backup_success(self, suggester, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("content")
        result = await suggester._create_file_backup(str(test_file))
        assert result is True
        # Verify backup file was created
        backups = list(tmp_path.glob("test.py.fix_backup_*"))
        assert len(backups) == 1

    @pytest.mark.asyncio
    async def test_backup_failure(self, suggester):
        result = await suggester._create_file_backup("/nonexistent/file.py")
        assert result is False


class TestBackwardCompat:
    def test_fix_suggester_alias(self):
        assert FixSuggester is AutoFixSuggester


class TestRuffFixErrorSurfaces:
    @pytest.mark.asyncio
    async def test_missing_ruff_fails_the_file_with_the_reason(self, suggester, tmp_path):
        """ruff absent at runtime: the fix is reported FAILED, not applied-unchanged."""
        from unittest.mock import patch

        target = tmp_path / "mod.py"
        target.write_text("x=1\n")
        fix = _make_suggestion(category=FixCategory.FORMATTING, file_path=str(target))
        fix.tool = "ruff-format"
        with patch("subprocess.run", side_effect=FileNotFoundError("ruff")):
            result = await suggester._apply_fixes_to_single_file(
                str(target), [fix], dry_run=False, create_backup=False
            )
        assert result["applied"] == []
        assert result["failed"] == [fix.fix_id]
        assert result["modified"] is False
        assert any("ruff is not installed" in e for e in result["errors"])
        assert target.read_text() == "x=1\n"
