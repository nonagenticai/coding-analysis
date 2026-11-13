"""Unit tests for recovery_mechanisms module."""

from __future__ import annotations

import ast

import pytest

from services.code_graph.recovery_mechanisms import (
    DefaultSuggestionStrategy,
    ExceptionSuggestionStrategy,
    LogicSuggestionStrategy,
    NamingSuggestionStrategy,
    ParameterSuggestionStrategy,
    RecoveryMechanisms,
    ReturnSuggestionStrategy,
    StructuralSuggestionStrategy,
    SuggestionStrategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_issue(
    issue_type: str = "parameter",
    description: str = "some issue",
    severity: str = "medium",
    **extra,
) -> dict:
    """Create an issue dict with sensible defaults."""
    issue = {"type": issue_type, "description": description, "severity": severity}
    issue.update(extra)
    return issue


# ---------------------------------------------------------------------------
# ParameterSuggestionStrategy
# ---------------------------------------------------------------------------


class TestParameterSuggestionStrategy:
    @pytest.fixture
    def strategy(self):
        return ParameterSuggestionStrategy()

    def test_get_strategy_name(self, strategy):
        assert strategy.get_strategy_name() == "parameter_suggestion"

    def test_get_applicable_types(self, strategy):
        assert "parameter" in strategy.get_applicable_types()
        assert "missing_parameters" in strategy.get_applicable_types()

    def test_single_missing_parameter(self, strategy):
        issues = [_make_issue("parameter", "Parameter 'x' is not documented")]
        suggestions = strategy.generate(issues, {})
        assert len(suggestions) == 1
        assert "'x'" in suggestions[0]

    def test_multiple_missing_parameters(self, strategy):
        issues = [
            _make_issue("parameter", "Parameter 'x' is not documented"),
            _make_issue("parameter", "Parameter 'y' is not documented"),
        ]
        suggestions = strategy.generate(issues, {})
        assert len(suggestions) == 1
        assert "'x'" in suggestions[0]
        assert "'y'" in suggestions[0]

    def test_duplicate_parameters_deduped(self, strategy):
        issues = [
            _make_issue("parameter", "Parameter 'x' is not documented"),
            _make_issue("parameter", "Parameter 'x' is missing again"),
        ]
        suggestions = strategy.generate(issues, {})
        # Only one param, so single-param message
        assert len(suggestions) == 1
        assert "Add explanation for parameter 'x'" in suggestions[0]

    def test_non_parameter_issues_ignored(self, strategy):
        issues = [_make_issue("return", "Parameter 'x' is not documented")]
        suggestions = strategy.generate(issues, {})
        assert suggestions == []

    def test_fallback_to_parameter_name_field(self, strategy):
        # When description doesn't contain "parameter '" but parameter_name is set
        issues = [
            _make_issue("parameter", "Missing documentation", parameter_name="foo"),
        ]
        # The description doesn't contain "parameter '" so extraction skips,
        # but fallback to parameter_name fires via IndexError path
        suggestions = strategy.generate(issues, {})
        # parameter_name won't be used because the description doesn't match
        # "parameter '" pattern at all - the conditional check fails
        assert suggestions == []

    def test_with_function_node_typed_params(self, strategy):
        code = "def foo(x: int, y): pass"
        tree = ast.parse(code)
        func_node = tree.body[0]
        issues = [
            _make_issue("parameter", "Parameter 'x' is not documented"),
        ]
        suggestions = strategy.generate(issues, {"function_node": func_node})
        assert any("type information" in s for s in suggestions)

    def test_with_function_node_no_typed_params(self, strategy):
        code = "def foo(x, y): pass"
        tree = ast.parse(code)
        func_node = tree.body[0]
        issues = [
            _make_issue("parameter", "Parameter 'x' is not documented"),
        ]
        suggestions = strategy.generate(issues, {"function_node": func_node})
        # x has no annotation, so no type-info suggestion
        assert not any("type information" in s for s in suggestions)

    def test_empty_issues(self, strategy):
        suggestions = strategy.generate([], {})
        assert suggestions == []


# ---------------------------------------------------------------------------
# ReturnSuggestionStrategy
# ---------------------------------------------------------------------------


class TestReturnSuggestionStrategy:
    @pytest.fixture
    def strategy(self):
        return ReturnSuggestionStrategy()

    def test_get_strategy_name(self, strategy):
        assert strategy.get_strategy_name() == "return_suggestion"

    def test_get_applicable_types(self, strategy):
        assert "return" in strategy.get_applicable_types()
        assert "missing_return_info" in strategy.get_applicable_types()

    def test_missing_return_not_mentioned(self, strategy):
        issues = [_make_issue("return", "Return value is not mentioned")]
        suggestions = strategy.generate(issues, {})
        assert any("return values" in s for s in suggestions)

    def test_missing_return_returns_values(self, strategy):
        issues = [_make_issue("return", "Function returns values but undocumented")]
        suggestions = strategy.generate(issues, {})
        assert any("return values" in s for s in suggestions)

    def test_missing_return_type(self, strategy):
        issues = [_make_issue("return", "The return type is not specified")]
        suggestions = strategy.generate(issues, {})
        assert any("return type" in s for s in suggestions)

    def test_specific_return_type_single(self, strategy):
        issues = [_make_issue("return", "desc", return_type="str")]
        suggestions = strategy.generate(issues, {})
        assert any("returns str" in s for s in suggestions)

    def test_specific_return_type_multiple(self, strategy):
        issues = [
            _make_issue("return", "desc", return_type="str"),
            _make_issue("return", "desc", return_type="int"),
        ]
        suggestions = strategy.generate(issues, {})
        assert any("return types" in s for s in suggestions)

    def test_empty_issues(self, strategy):
        suggestions = strategy.generate([], {})
        assert suggestions == []

    def test_non_return_issues_ignored(self, strategy):
        issues = [_make_issue("parameter", "Return value is not mentioned")]
        suggestions = strategy.generate(issues, {})
        assert suggestions == []


# ---------------------------------------------------------------------------
# ExceptionSuggestionStrategy
# ---------------------------------------------------------------------------


class TestExceptionSuggestionStrategy:
    @pytest.fixture
    def strategy(self):
        return ExceptionSuggestionStrategy()

    def test_get_strategy_name(self, strategy):
        assert strategy.get_strategy_name() == "exception_suggestion"

    def test_get_applicable_types(self, strategy):
        assert strategy.get_applicable_types() == ["exception"]

    def test_missing_exception_not_mentioned(self, strategy):
        issues = [_make_issue("exception", "Exception handling is not mentioned")]
        suggestions = strategy.generate(issues, {})
        assert any("exception handling" in s.lower() for s in suggestions)

    def test_missing_exception_handles(self, strategy):
        issues = [_make_issue("exception", "Function handles exceptions but undocumented")]
        suggestions = strategy.generate(issues, {})
        assert any("exception handling" in s.lower() for s in suggestions)

    def test_specific_exception_type_field(self, strategy):
        issues = [_make_issue("exception", "desc", exception_type="ValueError")]
        suggestions = strategy.generate(issues, {})
        assert any("ValueError" in s for s in suggestions)

    def test_specific_exception_from_description(self, strategy):
        issues = [_make_issue("exception", "handles 'TypeError' but not documented")]
        suggestions = strategy.generate(issues, {})
        assert any("TypeError" in s for s in suggestions)

    def test_empty_issues(self, strategy):
        suggestions = strategy.generate([], {})
        assert suggestions == []


# ---------------------------------------------------------------------------
# LogicSuggestionStrategy
# ---------------------------------------------------------------------------


class TestLogicSuggestionStrategy:
    @pytest.fixture
    def strategy(self):
        return LogicSuggestionStrategy()

    def test_get_strategy_name(self, strategy):
        assert strategy.get_strategy_name() == "logic_suggestion"

    def test_get_applicable_types(self, strategy):
        assert strategy.get_applicable_types() == ["logic"]

    def test_database_operation(self, strategy):
        issues = [_make_issue("logic", "desc", operation_type="database")]
        suggestions = strategy.generate(issues, {})
        assert any("database" in s.lower() for s in suggestions)

    def test_file_operation(self, strategy):
        issues = [_make_issue("logic", "desc", operation_type="file")]
        suggestions = strategy.generate(issues, {})
        assert any("file" in s.lower() for s in suggestions)

    def test_network_operation(self, strategy):
        issues = [_make_issue("logic", "desc", operation_type="network")]
        suggestions = strategy.generate(issues, {})
        assert any("network" in s.lower() for s in suggestions)

    def test_async_operation(self, strategy):
        issues = [_make_issue("logic", "desc", operation_type="async")]
        suggestions = strategy.generate(issues, {})
        assert any("asynchronously" in s for s in suggestions)

    def test_recursion_operation(self, strategy):
        issues = [_make_issue("logic", "desc", operation_type="recursion")]
        suggestions = strategy.generate(issues, {})
        assert any("recursive" in s for s in suggestions)

    def test_operation_type_from_description(self, strategy):
        issues = [_make_issue("logic", "Uses database queries")]
        suggestions = strategy.generate(issues, {})
        assert any("database" in s.lower() for s in suggestions)

    def test_unknown_operation_type(self, strategy):
        issues = [_make_issue("logic", "some generic issue", operation_type="unknown_op")]
        suggestions = strategy.generate(issues, {})
        assert suggestions == []

    def test_empty_issues(self, strategy):
        suggestions = strategy.generate([], {})
        assert suggestions == []


# ---------------------------------------------------------------------------
# StructuralSuggestionStrategy
# ---------------------------------------------------------------------------


class TestStructuralSuggestionStrategy:
    @pytest.fixture
    def strategy(self):
        return StructuralSuggestionStrategy()

    def test_get_strategy_name(self, strategy):
        assert strategy.get_strategy_name() == "structural_suggestion"

    def test_get_applicable_types(self, strategy):
        types = strategy.get_applicable_types()
        assert "structural" in types
        assert "logic" in types

    def test_empty_function(self, strategy):
        issues = [_make_issue("structural", "desc", category="empty_function")]
        suggestions = strategy.generate(issues, {})
        assert any("empty function" in s.lower() for s in suggestions)

    def test_empty_class(self, strategy):
        issues = [_make_issue("structural", "desc", category="empty_class")]
        suggestions = strategy.generate(issues, {})
        assert any("empty class" in s.lower() for s in suggestions)

    def test_unreachable_code(self, strategy):
        issues = [_make_issue("structural", "desc", category="unreachable_code")]
        suggestions = strategy.generate(issues, {})
        assert any("unreachable" in s.lower() for s in suggestions)

    def test_infinite_loop(self, strategy):
        issues = [_make_issue("structural", "desc", category="infinite_loop")]
        suggestions = strategy.generate(issues, {})
        assert any("infinite loop" in s.lower() for s in suggestions)

    def test_unused_variable(self, strategy):
        issues = [_make_issue("structural", "desc", category="unused_variable")]
        suggestions = strategy.generate(issues, {})
        assert any("unused variable" in s.lower() for s in suggestions)

    def test_multiple_categories(self, strategy):
        issues = [
            _make_issue("structural", "desc", category="empty_function"),
            _make_issue("structural", "desc", category="unreachable_code"),
        ]
        suggestions = strategy.generate(issues, {})
        assert len(suggestions) == 2

    def test_no_matching_category(self, strategy):
        issues = [_make_issue("structural", "desc", category="other")]
        suggestions = strategy.generate(issues, {})
        assert suggestions == []

    def test_empty_issues(self, strategy):
        suggestions = strategy.generate([], {})
        assert suggestions == []


# ---------------------------------------------------------------------------
# NamingSuggestionStrategy
# ---------------------------------------------------------------------------


class TestNamingSuggestionStrategy:
    @pytest.fixture
    def strategy(self):
        return NamingSuggestionStrategy()

    def test_get_strategy_name(self, strategy):
        assert strategy.get_strategy_name() == "naming_suggestion"

    def test_get_applicable_types(self, strategy):
        types = strategy.get_applicable_types()
        assert "naming" in types

    def test_single_function_name(self, strategy):
        issues = [_make_issue("naming", "desc", entity_type="function", entity_name="do_work")]
        suggestions = strategy.generate(issues, {})
        assert len(suggestions) == 1
        assert "'do_work'" in suggestions[0]

    def test_multiple_function_names(self, strategy):
        issues = [
            _make_issue("naming", "desc", entity_type="function", entity_name="foo"),
            _make_issue("naming", "desc", entity_type="function", entity_name="bar"),
        ]
        suggestions = strategy.generate(issues, {})
        assert any("'foo'" in s and "'bar'" in s for s in suggestions)

    def test_single_class_name(self, strategy):
        issues = [_make_issue("naming", "desc", entity_type="class", entity_name="MyClass")]
        suggestions = strategy.generate(issues, {})
        assert len(suggestions) == 1
        assert "'MyClass'" in suggestions[0]

    def test_multiple_class_names(self, strategy):
        issues = [
            _make_issue("naming", "desc", entity_type="class", entity_name="A"),
            _make_issue("naming", "desc", entity_type="class", entity_name="B"),
        ]
        suggestions = strategy.generate(issues, {})
        assert any("'A'" in s and "'B'" in s for s in suggestions)

    def test_mixed_functions_and_classes(self, strategy):
        issues = [
            _make_issue("naming", "desc", entity_type="function", entity_name="foo"),
            _make_issue("naming", "desc", entity_type="class", entity_name="Bar"),
        ]
        suggestions = strategy.generate(issues, {})
        assert len(suggestions) == 2

    def test_no_entity_name(self, strategy):
        issues = [_make_issue("naming", "desc", entity_type="function")]
        suggestions = strategy.generate(issues, {})
        assert suggestions == []

    def test_empty_issues(self, strategy):
        suggestions = strategy.generate([], {})
        assert suggestions == []


# ---------------------------------------------------------------------------
# DefaultSuggestionStrategy
# ---------------------------------------------------------------------------


class TestDefaultSuggestionStrategy:
    @pytest.fixture
    def strategy(self):
        return DefaultSuggestionStrategy()

    def test_get_strategy_name(self, strategy):
        assert strategy.get_strategy_name() == "default_suggestion"

    def test_get_applicable_types(self, strategy):
        assert "unknown" in strategy.get_applicable_types()
        assert "default" in strategy.get_applicable_types()

    def test_no_issues(self, strategy):
        suggestions = strategy.generate([], {})
        assert len(suggestions) == 1
        assert "well-aligned" in suggestions[0]

    def test_critical_severity(self, strategy):
        issues = [_make_issue("other", "desc", severity="critical")]
        suggestions = strategy.generate(issues, {})
        assert any("critical" in s.lower() for s in suggestions)

    def test_high_severity(self, strategy):
        issues = [_make_issue("other", "desc", severity="high")]
        suggestions = strategy.generate(issues, {})
        assert any("high-priority" in s.lower() for s in suggestions)

    def test_medium_severity(self, strategy):
        issues = [_make_issue("other", "desc", severity="medium")]
        suggestions = strategy.generate(issues, {})
        assert any("review" in s.lower() for s in suggestions)


# ---------------------------------------------------------------------------
# RecoveryMechanisms
# ---------------------------------------------------------------------------


class TestRecoveryMechanisms:
    @pytest.fixture
    def rm(self):
        return RecoveryMechanisms()

    # -- init / strategy management --

    def test_init_has_all_strategies(self, rm):
        names = rm.get_available_strategies()
        assert "parameter_suggestion" in names
        assert "return_suggestion" in names
        assert "exception_suggestion" in names
        assert "logic_suggestion" in names
        assert "structural_suggestion" in names
        assert "naming_suggestion" in names
        assert "default_suggestion" in names

    def test_add_custom_strategy(self, rm):
        class Custom(SuggestionStrategy):
            def generate(self, issues, context):
                return ["custom"]

            def get_strategy_name(self):
                return "custom"

            def get_applicable_types(self):
                return ["custom_type"]

        rm.add_custom_strategy(Custom())
        assert "custom" in rm.get_available_strategies()

    def test_remove_strategy(self, rm):
        assert rm.remove_strategy("parameter_suggestion") is True
        assert "parameter_suggestion" not in rm.get_available_strategies()

    def test_remove_nonexistent_strategy(self, rm):
        assert rm.remove_strategy("nonexistent") is False

    # -- _validate_input --

    def test_validate_input_valid(self, rm):
        issues = [_make_issue()]
        assert rm._validate_input(issues, {}) is True

    def test_validate_input_not_list(self, rm):
        assert rm._validate_input("not a list", {}) is False

    def test_validate_input_bad_context(self, rm):
        assert rm._validate_input([], "not a dict") is False

    def test_validate_input_issue_not_dict(self, rm):
        assert rm._validate_input(["not a dict"], {}) is False

    def test_validate_input_missing_type(self, rm):
        assert rm._validate_input([{"description": "x"}], {}) is False

    def test_validate_input_missing_description(self, rm):
        assert rm._validate_input([{"type": "x"}], {}) is False

    # -- _handle_invalid_input --

    def test_handle_invalid_input_no_issues(self, rm):
        result = rm._handle_invalid_input(issues=None)
        assert any("No issues" in s for s in result)

    def test_handle_invalid_input_not_list(self, rm):
        result = rm._handle_invalid_input(issues="bad")
        assert any("Invalid" in s for s in result)

    def test_handle_invalid_input_malformed_list(self, rm):
        result = rm._handle_invalid_input(issues=["bad"])
        assert any("invalid input" in s.lower() for s in result)

    # -- _group_issues_by_type --

    def test_group_by_type(self, rm):
        issues = [
            _make_issue("parameter", "a"),
            _make_issue("return", "b"),
            _make_issue("parameter", "c"),
        ]
        grouped = rm._group_issues_by_type(issues)
        assert len(grouped["parameter"]) == 2
        assert len(grouped["return"]) == 1

    def test_group_by_type_missing_type(self, rm):
        issues = [{"description": "x"}]
        grouped = rm._group_issues_by_type(issues)
        assert "unknown" in grouped

    # -- _group_issues_by_severity --

    def test_group_by_severity(self, rm):
        issues = [
            _make_issue(severity="critical"),
            _make_issue(severity="high"),
            _make_issue(severity="critical"),
        ]
        grouped = rm._group_issues_by_severity(issues)
        assert len(grouped["critical"]) == 2
        assert len(grouped["high"]) == 1

    def test_group_by_severity_default(self, rm):
        issues = [{"type": "x", "description": "y"}]
        grouped = rm._group_issues_by_severity(issues)
        assert "medium" in grouped

    # -- _find_applicable_strategies --

    def test_find_applicable_strategies_parameter(self, rm):
        strategies = rm._find_applicable_strategies("parameter")
        assert any(s.get_strategy_name() == "parameter_suggestion" for s in strategies)

    def test_find_applicable_strategies_unknown(self, rm):
        strategies = rm._find_applicable_strategies("unknown")
        assert any(s.get_strategy_name() == "default_suggestion" for s in strategies)

    def test_find_applicable_strategies_no_match(self, rm):
        strategies = rm._find_applicable_strategies("completely_made_up_type_xyz")
        assert strategies == []

    # -- _get_default_strategy --

    def test_get_default_strategy(self, rm):
        strategy = rm._get_default_strategy()
        assert strategy.get_strategy_name() == "default_suggestion"

    # -- _prioritize_suggestions --

    def test_prioritize_deduplicates(self, rm):
        suggestions = ["a", "b", "a", "c"]
        result = rm._prioritize_suggestions(suggestions, [])
        assert result == ["a", "b", "c"]

    def test_prioritize_critical_first(self, rm):
        suggestions = [
            "Update description",
            "Fix critical syntax error",
        ]
        result = rm._prioritize_suggestions(suggestions, [])
        assert result[0] == "Fix critical syntax error"

    def test_prioritize_high_before_normal(self, rm):
        suggestions = [
            "Update description",
            "Fix structural issue",
        ]
        result = rm._prioritize_suggestions(suggestions, [])
        assert result[0] == "Fix structural issue"

    # -- _estimate_effort --

    def test_estimate_effort_empty(self, rm):
        assert rm._estimate_effort([]) == "minimal"

    def test_estimate_effort_critical(self, rm):
        issues = [_make_issue(severity="critical")]
        assert rm._estimate_effort(issues) == "high"

    def test_estimate_effort_many_high(self, rm):
        issues = [_make_issue(severity="high") for _ in range(4)]
        assert rm._estimate_effort(issues) == "medium"

    def test_estimate_effort_many_total(self, rm):
        issues = [_make_issue(severity="medium") for _ in range(11)]
        assert rm._estimate_effort(issues) == "medium"

    def test_estimate_effort_moderate_total(self, rm):
        issues = [_make_issue(severity="low") for _ in range(6)]
        assert rm._estimate_effort(issues) == "low"

    def test_estimate_effort_few(self, rm):
        issues = [_make_issue(severity="low") for _ in range(3)]
        assert rm._estimate_effort(issues) == "minimal"

    # -- generate_suggestions --

    def test_generate_suggestions_parameter_issues(self, rm):
        issues = [_make_issue("parameter", "Parameter 'x' is not documented")]
        suggestions = rm.generate_suggestions(issues)
        assert len(suggestions) > 0

    def test_generate_suggestions_none_context(self, rm):
        issues = [_make_issue("parameter", "Parameter 'x' is not documented")]
        suggestions = rm.generate_suggestions(issues, None)
        assert isinstance(suggestions, list)

    def test_generate_suggestions_invalid_input(self, rm):
        result = rm.generate_suggestions("not a list")
        assert isinstance(result, list)
        assert any("invalid" in s.lower() or "Invalid" in s for s in result)

    def test_generate_suggestions_empty_issues(self, rm):
        result = rm.generate_suggestions([])
        assert isinstance(result, list)

    def test_generate_suggestions_no_matching_strategy(self, rm):
        issues = [_make_issue("completely_unknown_type", "some desc")]
        suggestions = rm.generate_suggestions(issues)
        # Should fall back to default strategy
        assert len(suggestions) > 0

    def test_generate_suggestions_multiple_types(self, rm):
        issues = [
            _make_issue("parameter", "Parameter 'x' is not documented"),
            _make_issue("return", "Return value is not mentioned"),
        ]
        suggestions = rm.generate_suggestions(issues)
        assert len(suggestions) >= 2

    # -- generate_recovery_plan --

    def test_recovery_plan_structure(self, rm):
        issues = [_make_issue("parameter", "Parameter 'x' is not documented")]
        plan = rm.generate_recovery_plan(issues)
        assert "total_issues" in plan
        assert "total_suggestions" in plan
        assert "estimated_effort" in plan
        assert "action_plan" in plan
        assert "all_suggestions" in plan
        assert "issue_breakdown" in plan
        assert plan["total_issues"] == 1

    def test_recovery_plan_critical_issues(self, rm):
        issues = [_make_issue("parameter", "Critical syntax error", severity="critical")]
        plan = rm.generate_recovery_plan(issues)
        assert plan["estimated_effort"] == "high"
        assert plan["issue_breakdown"]["by_severity"].get("critical", 0) == 1

    def test_recovery_plan_high_issues(self, rm):
        issues = [_make_issue("parameter", "Parameter 'x' is not documented", severity="high")]
        plan = rm.generate_recovery_plan(issues)
        assert plan["issue_breakdown"]["by_severity"].get("high", 0) == 1

    def test_recovery_plan_medium_issues(self, rm):
        issues = [_make_issue("parameter", "Parameter 'x' is not documented", severity="medium")]
        plan = rm.generate_recovery_plan(issues)
        assert plan["issue_breakdown"]["by_severity"].get("medium", 0) == 1

    def test_recovery_plan_low_issues(self, rm):
        issues = [_make_issue("parameter", "Parameter 'x' is not documented", severity="low")]
        plan = rm.generate_recovery_plan(issues)
        assert plan["issue_breakdown"]["by_severity"].get("low", 0) == 1

    def test_recovery_plan_none_context(self, rm):
        issues = [_make_issue("parameter", "Parameter 'x' is not documented")]
        plan = rm.generate_recovery_plan(issues, None)
        assert isinstance(plan, dict)

    def test_recovery_plan_empty_issues(self, rm):
        plan = rm.generate_recovery_plan([])
        assert plan["total_issues"] == 0

    def test_recovery_plan_all_severities(self, rm):
        issues = [
            _make_issue("parameter", "Critical syntax error", severity="critical"),
            _make_issue("return", "Return value not mentioned in description", severity="high"),
            _make_issue("naming", "Function name description mismatch", severity="medium"),
            _make_issue("logic", "Minor style issue", severity="low"),
        ]
        plan = rm.generate_recovery_plan(issues)
        assert plan["total_issues"] == 4
        assert len(plan["action_plan"]) >= 1
        assert plan["issue_breakdown"]["by_severity"]["critical"] == 1
        assert plan["issue_breakdown"]["by_severity"]["high"] == 1
        assert plan["issue_breakdown"]["by_severity"]["medium"] == 1
        assert plan["issue_breakdown"]["by_severity"]["low"] == 1

    def test_recovery_plan_by_type_breakdown(self, rm):
        issues = [
            _make_issue("parameter", "Parameter 'x' is not documented"),
            _make_issue("parameter", "Parameter 'y' is not documented"),
            _make_issue("return", "Return value not mentioned"),
        ]
        plan = rm.generate_recovery_plan(issues)
        assert plan["issue_breakdown"]["by_type"]["parameter"] == 2
        assert plan["issue_breakdown"]["by_type"]["return"] == 1
