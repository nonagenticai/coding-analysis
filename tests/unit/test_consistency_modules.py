"""Unit tests for consistency_checker, consistency_rules,
consistency_rules_engine, and graph_validation modules.

Covers the public methods of:
- CodeConsistencyChecker  (consistency_checker.py)
- ParameterConsistencyRule, ReturnConsistencyRule,
  ExceptionConsistencyRule, LogicOperationConsistencyRule,
  NamingConsistencyRule  (consistency_rules.py)
- ConsistencyRulesEngine, ModuleConsistencyChecker
  (consistency_rules_engine.py)
- GraphValidator  (graph_validation.py)
"""

from __future__ import annotations

import ast
from unittest.mock import MagicMock

import networkx as nx

# We import CodeConsistencyChecker last because it pulls in many more
# internal modules (error_detection_reporting, recovery_mechanisms, etc.).
from services.code_graph.consistency_checker import CodeConsistencyChecker
from services.code_graph.consistency_rules import (
    ExceptionConsistencyRule,
    LogicOperationConsistencyRule,
    NamingConsistencyRule,
    ParameterConsistencyRule,
    ReturnConsistencyRule,
)
from services.code_graph.consistency_rules_engine import (
    ConsistencyRulesEngine,
    ModuleConsistencyChecker,
)
from services.code_graph.graph_validation import GraphValidator

# =====================================================================
# Helpers
# =====================================================================

SIMPLE_FUNC = """\
def add(a, b):
    return a + b
"""

FUNC_WITH_RETURN_TYPE = """\
def greet(name: str) -> str:
    return f"Hello, {name}"
"""

FUNC_WITH_TRY_EXCEPT = """\
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
"""

FUNC_WITH_DB_OPS = """\
def fetch_user(user_id):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()
"""

RECURSIVE_FUNC = """\
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

SIMPLE_CLASS = """\
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"""


# =====================================================================
# consistency_rules.py
# =====================================================================


class TestParameterConsistencyRule:
    """Tests for ParameterConsistencyRule."""

    def setup_method(self):
        self.rule = ParameterConsistencyRule()

    def test_rule_name(self):
        assert self.rule.get_rule_name() == "parameter_consistency"

    def test_severity(self):
        assert self.rule.get_severity() == "medium"

    def test_no_issues_when_params_mentioned(self):
        desc = "Add two numbers a and b and return the sum."
        issues = self.rule.check(SIMPLE_FUNC, desc, {})
        assert len(issues) == 0

    def test_issue_when_param_missing_from_desc(self):
        code = "def compute(x_val, y_val):\n    return x_val + y_val\n"
        desc = "Does some work."
        issues = self.rule.check(code, desc, {})
        # 'x_val' and 'y_val' not mentioned
        assert len(issues) == 2
        assert all(i["type"] == "parameter" for i in issues)

    def test_self_param_excluded(self):
        code = "def greet(self, name):\n    return name"
        desc = "Greet with name."
        issues = self.rule.check(code, desc, {})
        # 'self' should not generate an issue, 'name' is mentioned
        assert len(issues) == 0

    def test_syntax_error_code(self):
        bad_code = "def oops(:"
        issues = self.rule.check(bad_code, "whatever", {})
        assert len(issues) == 1
        assert issues[0]["type"] == "syntax"

    def test_no_function_returns_empty(self):
        code = "x = 1"
        issues = self.rule.check(code, "assign x", {})
        assert issues == []

    def test_no_params_returns_empty(self):
        code = "def noop():\n    pass"
        issues = self.rule.check(code, "does nothing", {})
        assert issues == []


class TestReturnConsistencyRule:
    """Tests for ReturnConsistencyRule."""

    def setup_method(self):
        self.rule = ReturnConsistencyRule()

    def test_rule_name(self):
        assert self.rule.get_rule_name() == "return_consistency"

    def test_severity(self):
        assert self.rule.get_severity() == "medium"

    def test_no_issue_when_return_mentioned(self):
        desc = "Add a and b and return the sum."
        issues = self.rule.check(SIMPLE_FUNC, desc, {})
        assert len(issues) == 0

    def test_issue_when_return_not_mentioned(self):
        desc = "Add two numbers together."
        issues = self.rule.check(SIMPLE_FUNC, desc, {})
        assert any("returns values" in i["description"] for i in issues)

    def test_return_type_not_mentioned(self):
        desc = "Greet the user by name and return the greeting."
        issues = self.rule.check(FUNC_WITH_RETURN_TYPE, desc, {})
        # 'str' return type may or may not be in desc; check rule fires about type
        type_issues = [i for i in issues if "return_type" in i]
        assert len(type_issues) == 1
        assert type_issues[0]["return_type"] == "str"

    def test_no_function_returns_empty(self):
        code = "x = 1"
        issues = self.rule.check(code, "assign", {})
        assert issues == []


class TestExceptionConsistencyRule:
    """Tests for ExceptionConsistencyRule."""

    def setup_method(self):
        self.rule = ExceptionConsistencyRule()

    def test_rule_name(self):
        assert self.rule.get_rule_name() == "exception_consistency"

    def test_severity(self):
        assert self.rule.get_severity() == "low"

    def test_no_issue_when_exception_handling_mentioned(self):
        desc = "Divide a by b, handles ZeroDivisionError exception."
        issues = self.rule.check(FUNC_WITH_TRY_EXCEPT, desc, {})
        assert len(issues) == 0

    def test_issue_when_exception_not_mentioned(self):
        desc = "Divide a by b."
        issues = self.rule.check(FUNC_WITH_TRY_EXCEPT, desc, {})
        assert len(issues) >= 1

    def test_specific_exception_type_reported(self):
        desc = "Divide a by b, handles errors."
        issues = self.rule.check(FUNC_WITH_TRY_EXCEPT, desc, {})
        type_issues = [i for i in issues if "exception_type" in i]
        assert any(i["exception_type"] == "ZeroDivisionError" for i in type_issues)

    def test_no_function_returns_empty(self):
        code = "x = 1"
        issues = self.rule.check(code, "assign", {})
        assert issues == []


class TestLogicOperationConsistencyRule:
    """Tests for LogicOperationConsistencyRule."""

    def setup_method(self):
        self.rule = LogicOperationConsistencyRule()

    def test_rule_name(self):
        assert self.rule.get_rule_name() == "logic_operation_consistency"

    def test_severity(self):
        assert self.rule.get_severity() == "medium"

    def test_database_op_not_mentioned(self):
        desc = "Fetch the user."
        issues = self.rule.check(FUNC_WITH_DB_OPS, desc, {})
        db_issues = [i for i in issues if i.get("operation_type") == "database"]
        assert len(db_issues) == 1

    def test_database_op_mentioned(self):
        desc = "Query the database for a user by id."
        issues = self.rule.check(FUNC_WITH_DB_OPS, desc, {})
        db_issues = [i for i in issues if i.get("operation_type") == "database"]
        assert len(db_issues) == 0

    def test_recursion_detected_not_mentioned(self):
        desc = "Calculate factorial of n."
        issues = self.rule.check(RECURSIVE_FUNC, desc, {})
        rec_issues = [i for i in issues if i.get("operation_type") == "recursion"]
        assert len(rec_issues) == 1

    def test_recursion_mentioned(self):
        desc = "Calculate factorial of n using recursion."
        issues = self.rule.check(RECURSIVE_FUNC, desc, {})
        rec_issues = [i for i in issues if i.get("operation_type") == "recursion"]
        assert len(rec_issues) == 0

    def test_no_function_returns_empty(self):
        code = "x = 1"
        issues = self.rule.check(code, "assign", {})
        assert issues == []


class TestNamingConsistencyRule:
    """Tests for NamingConsistencyRule."""

    def setup_method(self):
        self.rule = NamingConsistencyRule()

    def test_rule_name(self):
        assert self.rule.get_rule_name() == "naming_consistency"

    def test_severity(self):
        assert self.rule.get_severity() == "low"

    def test_no_issue_when_name_mentioned(self):
        desc = "The add function adds two numbers."
        issues = self.rule.check(SIMPLE_FUNC, desc, {})
        assert len(issues) == 0

    def test_issue_when_name_not_mentioned(self):
        desc = "Sums two numbers."
        issues = self.rule.check(SIMPLE_FUNC, desc, {})
        assert len(issues) == 1
        assert issues[0]["entity_name"] == "add"

    def test_class_name_not_mentioned(self):
        desc = "A class that does math."
        issues = self.rule.check(SIMPLE_CLASS, desc, {})
        name_issues = [i for i in issues if i.get("entity_type") == "class"]
        assert any(i["entity_name"] == "Calculator" for i in name_issues)


# =====================================================================
# consistency_rules_engine.py
# =====================================================================


class TestConsistencyRulesEngine:
    """Tests for ConsistencyRulesEngine."""

    def setup_method(self):
        self.engine = ConsistencyRulesEngine()

    def test_default_rules_loaded(self):
        names = self.engine.get_enabled_rules()
        assert "parameter_consistency" in names
        assert "return_consistency" in names
        assert "exception_consistency" in names
        assert "logic_operation_consistency" in names
        assert "naming_consistency" in names

    def test_add_rule(self):
        custom = MagicMock()
        custom.get_rule_name.return_value = "custom_rule"
        custom.get_severity.return_value = "low"
        custom.check.return_value = []
        self.engine.add_rule(custom)
        assert "custom_rule" in self.engine.get_enabled_rules()

    def test_remove_rule(self):
        removed = self.engine.remove_rule("parameter_consistency")
        assert removed is True
        assert "parameter_consistency" not in self.engine.get_enabled_rules()

    def test_remove_nonexistent_rule(self):
        removed = self.engine.remove_rule("does_not_exist")
        assert removed is False

    def test_check_consistency_no_issues(self):
        desc = "The add function takes a and b and returns their sum."
        result = self.engine.check_consistency(SIMPLE_FUNC, desc)
        assert "consistency_score" in result
        assert "all_issues" in result
        assert "total_rules" in result
        assert result["total_rules"] == len(self.engine.rules)

    def test_check_consistency_with_issues(self):
        desc = "Does something."
        result = self.engine.check_consistency(SIMPLE_FUNC, desc)
        assert len(result["all_issues"]) > 0
        assert result["consistency_score"] < 1.0

    def test_check_consistency_perfect(self):
        """All rules pass => score == 1.0."""
        # Remove all rules to guarantee zero issues
        self.engine.rules = []
        result = self.engine.check_consistency(SIMPLE_FUNC, "anything")
        assert result["consistency_score"] == 1.0
        assert result["is_consistent"] is True

    def test_group_issues_by_severity(self):
        issues = [
            {"severity": "high"},
            {"severity": "medium"},
            {"severity": "medium"},
            {"severity": "low"},
        ]
        grouped = self.engine._group_issues_by_severity(issues)
        assert grouped == {"high": 1, "medium": 2, "low": 1}

    def test_calculate_consistency_score_empty(self):
        assert self.engine._calculate_consistency_score([]) == 1.0

    def test_calculate_consistency_score_penalty(self):
        issues = [{"severity": "high"}, {"severity": "high"}]
        score = self.engine._calculate_consistency_score(issues)
        assert 0.0 <= score < 1.0

    def test_get_rule_info_found(self):
        info = self.engine.get_rule_info("parameter_consistency")
        assert info["name"] == "parameter_consistency"
        assert info["severity"] == "medium"

    def test_get_rule_info_not_found(self):
        info = self.engine.get_rule_info("nonexistent")
        assert info == {}

    def test_enable_rule(self):
        assert self.engine.enable_rule("parameter_consistency") is True
        assert self.engine.enable_rule("nonexistent") is False

    def test_disable_rule(self):
        assert self.engine.disable_rule("parameter_consistency") is True
        assert "parameter_consistency" not in self.engine.get_enabled_rules()

    def test_rule_exception_handled(self):
        """If a rule raises, the engine logs and continues."""
        bad_rule = MagicMock()
        bad_rule.get_rule_name.return_value = "bad_rule"
        bad_rule.get_severity.return_value = "medium"
        bad_rule.check.side_effect = RuntimeError("boom")
        self.engine.rules = [bad_rule]
        result = self.engine.check_consistency(SIMPLE_FUNC, "desc")
        assert result["rule_results"]["bad_rule"]["passed"] is False
        assert "error" in result["rule_results"]["bad_rule"]


class TestModuleConsistencyChecker:
    """Tests for ModuleConsistencyChecker."""

    def setup_method(self):
        self.checker = ModuleConsistencyChecker()

    def test_check_naming_conventions_snake_case_only(self):
        code = "def my_func():\n    pass\n"
        tree = ast.parse(code)
        issues = self.checker.check_naming_conventions({"mod_a": tree})
        # pure snake_case, no inconsistencies
        assert len(issues) == 0

    def test_check_naming_conventions_mixed(self):
        code = "def my_func():\n    pass\ndef myFunc():\n    pass\n"
        tree = ast.parse(code)
        issues = self.checker.check_naming_conventions({"mod_a": tree})
        assert any("Inconsistent" in i["description"] for i in issues)

    def test_check_naming_conventions_non_pascal_class(self):
        code = "class my_class:\n    pass\n"
        tree = ast.parse(code)
        issues = self.checker.check_naming_conventions({"mod_a": tree})
        assert any("PascalCase" in i["description"] for i in issues)

    def test_check_naming_conventions_across_modules(self):
        snake = ast.parse("def my_func():\n    pass\n")
        camel = ast.parse("def myFunc():\n    pass\n")
        issues = self.checker.check_naming_conventions({"mod_snake": snake, "mod_camel": camel})
        assert any("across modules" in i["description"] for i in issues)

    def test_check_import_patterns_all_absolute(self):
        code = "import os\nimport sys\n"
        tree = ast.parse(code)
        issues = self.checker.check_import_patterns({"mod_a": tree})
        assert len(issues) == 0

    def test_check_import_patterns_mixed(self):
        code = "import os\nfrom . import sibling\n"
        tree = ast.parse(code)
        issues = self.checker.check_import_patterns({"mod_a": tree})
        assert len(issues) == 1
        assert "Mixed import" in issues[0]["description"]


# =====================================================================
# graph_validation.py
# =====================================================================


class TestGraphValidator:
    """Tests for GraphValidator."""

    def setup_method(self):
        self.validator = GraphValidator()

    # -- validate_graph_structure --

    def test_valid_simple_graph(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C")])
        result = self.validator.validate_graph_structure(g)
        assert result["is_valid"] is True
        assert result["connectivity_score"] == 1.0
        assert result["metrics"]["node_count"] == 3
        assert result["metrics"]["edge_count"] == 2

    def test_isolated_node(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B")])
        g.add_node("Z")  # isolated
        result = self.validator.validate_graph_structure(g)
        assert result["connectivity_score"] < 1.0
        iso_issues = [i for i in result["structural_issues"] if i["type"] == "connectivity"]
        assert len(iso_issues) == 1

    def test_circular_dependency(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
        result = self.validator.validate_graph_structure(g)
        assert result["is_valid"] is False
        cycle_issues = [
            i for i in result["structural_issues"] if i["type"] == "circular_dependency"
        ]
        assert len(cycle_issues) >= 1

    def test_empty_graph(self):
        g = nx.DiGraph()
        result = self.validator.validate_graph_structure(g)
        # networkx raises on is_weakly_connected for null graph,
        # so the code catches it and marks is_valid=False
        assert result["is_valid"] is False
        assert len(result["structural_issues"]) >= 1

    def test_cache_works(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B")])
        r1 = self.validator.validate_graph_structure(g)
        r2 = self.validator.validate_graph_structure(g)
        assert r1 == r2

    def test_dangling_reference(self):
        g = nx.DiGraph()
        g.add_node("A", target_ref="MISSING")
        g.add_node("B")
        g.add_edge("A", "B")
        result = self.validator.validate_graph_structure(g)
        dang = [i for i in result["structural_issues"] if i["type"] == "dangling_reference"]
        assert len(dang) == 1

    # -- validate_node_relationships --

    def test_node_not_in_graph(self):
        g = nx.DiGraph()
        g.add_node("A")
        result = self.validator.validate_node_relationships(g, "X")
        assert result["is_valid"] is False
        assert result["issues"][0]["type"] == "missing_node"

    def test_self_loop(self):
        g = nx.DiGraph()
        g.add_edge("A", "A")
        result = self.validator.validate_node_relationships(g, "A")
        assert any(i["type"] == "self_reference" for i in result["issues"])

    def test_high_fan_out(self):
        g = nx.DiGraph()
        for i in range(25):
            g.add_edge("hub", f"target_{i}")
        result = self.validator.validate_node_relationships(g, "hub")
        assert any(i["type"] == "high_coupling" for i in result["issues"])

    def test_high_fan_in(self):
        g = nx.DiGraph()
        for i in range(20):
            g.add_edge(f"dep_{i}", "sink")
        result = self.validator.validate_node_relationships(g, "sink")
        assert any(i["type"] == "high_dependency" for i in result["issues"])

    def test_normal_node_valid(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C")])
        result = self.validator.validate_node_relationships(g, "B")
        assert result["is_valid"] is True
        assert result["relationships"]["incoming_count"] == 1
        assert result["relationships"]["outgoing_count"] == 1

    # -- check_dependency_cycles --

    def test_no_cycles(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C")])
        result = self.validator.check_dependency_cycles(g)
        assert result["has_cycles"] is False
        assert result["cycle_count"] == 0
        assert result["severity_assessment"] == "none"

    def test_single_cycle(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "A")])
        result = self.validator.check_dependency_cycles(g)
        assert result["has_cycles"] is True
        assert result["cycle_count"] >= 1
        assert result["severity_assessment"] in ("low", "medium")

    def test_many_cycles_high_severity(self):
        g = nx.DiGraph()
        # Create many small cycles
        for i in range(10):
            a, b = f"X{i}", f"Y{i}"
            g.add_edges_from([(a, b), (b, a)])
        result = self.validator.check_dependency_cycles(g)
        assert result["has_cycles"] is True
        assert result["severity_assessment"] == "high"

    # -- validate_graph_integrity --

    def test_integrity_all_expected_present(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C")])
        result = self.validator.validate_graph_integrity(g, expected_nodes=["A", "B", "C"])
        assert result["is_intact"] is True
        assert result["completeness_score"] == 1.0

    def test_integrity_missing_nodes(self):
        g = nx.DiGraph()
        g.add_node("A")
        result = self.validator.validate_graph_integrity(g, expected_nodes=["A", "B", "C"])
        assert result["is_intact"] is False
        assert result["completeness_score"] < 1.0
        assert "B" in result["missing_nodes"]
        assert "C" in result["missing_nodes"]

    def test_integrity_unexpected_nodes(self):
        g = nx.DiGraph()
        g.add_nodes_from(["A", "B", "EXTRA"])
        result = self.validator.validate_graph_integrity(g, expected_nodes=["A", "B"])
        assert "EXTRA" in result["unexpected_nodes"]

    def test_integrity_no_expected(self):
        g = nx.DiGraph()
        g.add_node("A")
        result = self.validator.validate_graph_integrity(g)
        assert result["is_intact"] is True

    # -- _hash_graph --

    def test_hash_deterministic(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B"), ("B", "C")])
        h1 = self.validator._hash_graph(g)
        h2 = self.validator._hash_graph(g)
        assert h1 == h2

    # -- _find_dangling_references --

    def test_find_dangling_ref_attribute(self):
        g = nx.DiGraph()
        g.add_node("A", dep_ref="GONE")
        result = self.validator._find_dangling_references(g)
        assert len(result) == 1
        assert "A->GONE" in result[0]

    def test_find_dangling_ref_list(self):
        g = nx.DiGraph()
        g.add_node("A", deps_reference=["B", "MISSING"])
        g.add_node("B")
        result = self.validator._find_dangling_references(g)
        assert any("MISSING" in r for r in result)

    def test_no_dangling_refs(self):
        g = nx.DiGraph()
        g.add_edges_from([("A", "B")])
        result = self.validator._find_dangling_references(g)
        assert result == []


# =====================================================================
# consistency_checker.py  (CodeConsistencyChecker -- facade)
# =====================================================================


class TestCodeConsistencyChecker:
    """Tests for CodeConsistencyChecker facade."""

    def setup_method(self):
        self.checker = CodeConsistencyChecker()
        # Disable caching to keep tests isolated
        self.checker.config["enable_caching"] = False

    # -- check_function_description --

    def test_empty_code(self):
        result = self.checker.check_function_description("   ", "desc")
        assert result["is_consistent"] is False
        assert "Empty function code" in result["error"]

    def test_empty_description(self):
        result = self.checker.check_function_description(SIMPLE_FUNC, "   ")
        assert result["is_consistent"] is False
        assert "Empty description" in result["error"]

    def test_unparseable_code(self):
        result = self.checker.check_function_description("def oops(:", "desc")
        assert result["is_consistent"] is False
        assert "parse" in result["error"].lower()

    def test_code_without_function(self):
        result = self.checker.check_function_description("x = 1", "desc")
        assert result["is_consistent"] is False
        assert "No function" in result["error"]

    def test_valid_function(self):
        desc = "The add function takes parameters a and b and returns their sum."
        result = self.checker.check_function_description(SIMPLE_FUNC, desc)
        assert "consistency_score" in result
        assert "issues" in result
        assert "analysis_time_ms" in result
        assert result["context"] == "internal"
        assert "function_info" in result

    def test_public_api_context(self):
        desc = "The add function takes a and b and returns their sum."
        result = self.checker.check_function_description(SIMPLE_FUNC, desc, context="public_api")
        assert result["context"] == "public_api"

    def test_rules_summary_present(self):
        desc = "The add function."
        result = self.checker.check_function_description(SIMPLE_FUNC, desc)
        assert "rules_summary" in result
        assert "total_rules" in result["rules_summary"]

    # -- check_class_description --

    def test_empty_class_code(self):
        result = self.checker.check_class_description("  ", "desc")
        assert result["is_consistent"] is False

    def test_class_no_class_def(self):
        result = self.checker.check_class_description("x = 1", "desc")
        assert result["is_consistent"] is False

    def test_valid_class(self):
        desc = "The Calculator class provides add and subtract methods."
        result = self.checker.check_class_description(SIMPLE_CLASS, desc)
        assert "consistency_score" in result
        assert "class_info" in result
        assert result["class_name"] == "Calculator"

    # -- check_module_consistency --

    def test_single_valid_module(self):
        modules = {"mod_a": SIMPLE_FUNC}
        result = self.checker.check_module_consistency(modules)
        assert result["module_count"] == 1
        assert result["parsed_modules"] == 1
        assert "consistency_score" in result

    def test_unparseable_module(self):
        modules = {"bad": "def oops(:"}
        result = self.checker.check_module_consistency(modules)
        assert result["parsed_modules"] == 0
        assert len(result["issues"]) >= 1

    def test_multiple_modules(self):
        modules = {"a": SIMPLE_FUNC, "b": SIMPLE_CLASS}
        result = self.checker.check_module_consistency(modules)
        assert result["module_count"] == 2
        assert result["parsed_modules"] == 2

    # -- generate_comprehensive_report --

    def test_report_empty_results(self):
        result = self.checker.generate_comprehensive_report([])
        assert "error" in result

    def test_report_with_results(self):
        analysis = {
            "consistency_score": 0.9,
            "is_consistent": True,
            "issues": [{"type": "test", "severity": "low", "description": "minor"}],
        }
        result = self.checker.generate_comprehensive_report([analysis])
        assert "summary" in result
        assert "overall_metrics" in result
        assert result["overall_metrics"]["total_analyses"] == 1

    # -- utility methods --

    def test_create_error_result(self):
        result = self.checker._create_error_result("boom", 42, "public_api")
        assert result["consistency_score"] == 0.0
        assert result["is_consistent"] is False
        assert result["error"] == "boom"
        assert result["context"] == "public_api"
        assert result["analysis_time_ms"] == 42

    def test_configure(self):
        self.checker.configure(fast_mode=True)
        assert self.checker.config["fast_mode"] is True

    def test_get_available_rules(self):
        rules = self.checker.get_available_rules()
        assert isinstance(rules, list)
        assert len(rules) > 0

    def test_add_custom_rule(self):
        custom = MagicMock()
        custom.get_rule_name.return_value = "custom"
        self.checker.add_custom_rule(custom)
        assert "custom" in self.checker.get_available_rules()

    def test_clear_cache(self):
        # Should not raise
        self.checker.clear_cache()

    def test_get_cache_stats(self):
        stats = self.checker.get_cache_stats()
        assert "cache_size" in stats
        assert "max_size" in stats

    def test_caching_returns_cached_result(self):
        """When caching is enabled, the second call returns the cached result."""
        self.checker.config["enable_caching"] = True
        self.checker.clear_cache()
        desc = "The add function takes a and b and returns the sum."
        r1 = self.checker.check_function_description(SIMPLE_FUNC, desc)
        r2 = self.checker.check_function_description(SIMPLE_FUNC, desc)
        assert r1 == r2
