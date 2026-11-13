"""Tests for processing.quality_analysis.fix_rules module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from processing.quality_analysis.fix_rules import FixDetailGenerator, FixRuleManager


def _make_issue(
    tool: str = "ruff",
    message: str = "some issue",
    rule_id: str = "D103",
    severity: str = "warning",
    file_path: str = "test.py",
    line_number: int = 1,
) -> MagicMock:
    issue = MagicMock()
    issue.tool = tool
    issue.message = message
    issue.rule_id = rule_id
    issue.severity = severity
    issue.file_path = file_path
    issue.line_number = line_number
    return issue


# ---------------------------------------------------------------------------
# FixRuleManager
# ---------------------------------------------------------------------------


class TestFixRuleManager:
    def test_initialization(self):
        manager = FixRuleManager()
        assert manager.get_rule_count() > 0

    def test_has_python_rules(self):
        manager = FixRuleManager()
        rule_ids = [r.rule_id for r in manager.rules]
        assert {"ruff_formatting", "ruff_imports", "ruff_unused", "ty_return_type"} <= set(rule_ids)
        assert not {r.tool for r in manager.rules} & {
            "black",
            "isort",
            "mypy",
            "pylint",
            "flake8",
            "autoflake",
            "pyright",
        }

    def test_has_js_rules(self):
        manager = FixRuleManager()
        rule_ids = [r.rule_id for r in manager.rules]
        assert "eslint_auto_fixable" in rule_ids
        assert "prettier_formatting" in rule_ids

    def test_find_matching_rules_ruff_docstring(self):
        manager = FixRuleManager()
        rules = manager.find_matching_rules(_make_issue(tool="ruff", rule_id="D103"))
        assert [r.rule_id for r in rules] == ["ruff_docstring"]

    def test_find_matching_rules_ruff_format(self):
        manager = FixRuleManager()
        rules = manager.find_matching_rules(_make_issue(tool="ruff-format", rule_id="unformatted"))
        assert [r.rule_id for r in rules] == ["ruff_formatting"]

    @pytest.mark.parametrize(
        ("rule_id", "expected"),
        [("I001", "ruff_imports"), ("F401", "ruff_unused"), ("F841", "ruff_unused")],
    )
    def test_find_matching_rules_keyed_on_ruff_codes(self, rule_id, expected):
        manager = FixRuleManager()
        rules = manager.find_matching_rules(_make_issue(tool="ruff", rule_id=rule_id))
        assert [r.rule_id for r in rules] == [expected]

    def test_ruff_codes_outside_the_patterns_do_not_match(self):
        manager = FixRuleManager()
        for code in ("F4011", "E501", "PLR0913", "I"):
            assert manager.find_matching_rules(_make_issue(tool="ruff", rule_id=code)) == []

    def test_find_matching_rules_eslint(self):
        manager = FixRuleManager()
        issue = _make_issue(tool="eslint", rule_id="no-unused-vars")
        rules = manager.find_matching_rules(issue)
        assert len(rules) >= 1

    def test_find_matching_rules_no_match(self):
        manager = FixRuleManager()
        issue = _make_issue(tool="unknown_tool", rule_id="X999")
        rules = manager.find_matching_rules(issue)
        assert len(rules) == 0

    def test_find_matching_rules_ty_rule_name(self):
        manager = FixRuleManager()
        issue = _make_issue(tool="ty", rule_id="invalid-return-type")
        assert [r.rule_id for r in manager.find_matching_rules(issue)] == ["ty_return_type"]
        other = _make_issue(tool="ty", rule_id="invalid-argument-type")
        assert manager.find_matching_rules(other) == []

    def test_find_matching_rules_bandit(self):
        manager = FixRuleManager()
        issue = _make_issue(tool="bandit", rule_id="B101")
        rules = manager.find_matching_rules(issue)
        assert len(rules) >= 1


# ---------------------------------------------------------------------------
# FixDetailGenerator
# ---------------------------------------------------------------------------


class TestFixDetailGenerator:
    def test_get_supported_tools(self):
        gen = FixDetailGenerator()
        tools = gen.get_supported_tools()
        assert {"ruff", "ruff-format", "ty", "eslint"} <= set(tools)
        assert not set(tools) & {"black", "isort", "mypy", "pylint"}

    @pytest.mark.asyncio
    async def test_generate_ruff_format_fix(self):
        gen = FixDetailGenerator()
        result = await gen.generate_fix_details(
            _make_issue(tool="ruff-format", rule_id="unformatted"), MagicMock()
        )
        assert "ruff" in result["title"].lower()
        assert result["breaking_change"] is False

    @pytest.mark.asyncio
    async def test_generate_ruff_import_fix(self):
        gen = FixDetailGenerator()
        result = await gen.generate_fix_details(
            _make_issue(tool="ruff", rule_id="I001"), MagicMock()
        )
        assert "import" in result["title"].lower()

    @pytest.mark.asyncio
    async def test_generate_ruff_unused_fix(self):
        gen = FixDetailGenerator()
        result = await gen.generate_fix_details(
            _make_issue(tool="ruff", rule_id="F841"), MagicMock()
        )
        assert "unused" in result["title"].lower()
        assert "F401,F841" in result["description"]

    @pytest.mark.asyncio
    async def test_generate_ruff_naming_fix(self):
        gen = FixDetailGenerator()
        result = await gen.generate_fix_details(
            _make_issue(tool="ruff", rule_id="N802"), MagicMock()
        )
        assert result["breaking_change"] is True
        assert "naming" in result["title"].lower()

    @pytest.mark.asyncio
    async def test_generate_ruff_docstring_fix(self):
        gen = FixDetailGenerator()
        result = await gen.generate_fix_details(
            _make_issue(tool="ruff", rule_id="D103"), MagicMock()
        )
        assert "docstring" in result["title"].lower()

    @pytest.mark.asyncio
    async def test_generate_ruff_unused_arg_fix(self):
        gen = FixDetailGenerator()
        result = await gen.generate_fix_details(
            _make_issue(tool="ruff", rule_id="ARG001"), MagicMock()
        )
        assert result["breaking_change"] is True

    @pytest.mark.asyncio
    async def test_generate_ruff_generic_fix(self):
        gen = FixDetailGenerator()
        result = await gen.generate_fix_details(
            _make_issue(tool="ruff", rule_id="PLR0913"), MagicMock()
        )
        assert "PLR0913" in result["title"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rule", "needle"),
        [
            ("invalid-return-type", "return type"),
            ("invalid-argument-type", "argument type"),
            ("unresolved-import", "import"),
            ("possibly-unresolved-reference", "possibly-unresolved-reference"),
        ],
    )
    async def test_generate_ty_fix_keyed_on_rule_name(self, rule, needle):
        gen = FixDetailGenerator()
        result = await gen.generate_fix_details(_make_issue(tool="ty", rule_id=rule), MagicMock())
        assert needle in result["title"].lower()

    @pytest.mark.asyncio
    async def test_generate_bandit_fix(self):
        gen = FixDetailGenerator()
        issue = _make_issue(tool="bandit", message="Use of exec")
        rule = MagicMock()
        result = await gen.generate_fix_details(issue, rule)
        assert result["breaking_change"] is True
        assert "security" in result["title"].lower()

    @pytest.mark.asyncio
    async def test_generate_eslint_fix(self):
        gen = FixDetailGenerator()
        issue = _make_issue(tool="eslint", message="no-unused-vars")
        rule = MagicMock()
        result = await gen.generate_fix_details(issue, rule)
        assert "ESLint" in result["title"]

    @pytest.mark.asyncio
    async def test_generate_prettier_fix(self):
        gen = FixDetailGenerator()
        issue = _make_issue(tool="prettier")
        rule = MagicMock()
        result = await gen.generate_fix_details(issue, rule)
        assert "Prettier" in result["title"]
        assert result["breaking_change"] is False

    @pytest.mark.asyncio
    async def test_generate_generic_fix(self):
        gen = FixDetailGenerator()
        issue = _make_issue(tool="unknown_tool")
        rule = MagicMock()
        rule.requires_review = True
        result = await gen.generate_fix_details(issue, rule)
        assert result is not None
        assert result["breaking_change"] is True
