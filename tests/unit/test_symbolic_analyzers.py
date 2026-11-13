"""Unit tests for symbolic analysis tools: CrossHairAnalyzer and PysaAnalyzer.

Covers:
  - src/processing/analyzers/symbolic/tools/crosshair_analyzer.py
  - src/processing/analyzers/symbolic/tools/pysa_analyzer.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from processing.analyzers.symbolic.exceptions import (
    CrossHairAnalysisError,
    PysaAnalysisError,
)
from processing.analyzers.symbolic.tools.crosshair_analyzer import (
    ContractViolation,
    CrosshairAnalysisResult,
    CrossHairAnalyzer,
    PropertyTestResult,
)
from processing.analyzers.symbolic.tools.pysa_analyzer import PysaAnalyzer

# ===========================================================================
#  Helpers / Fixtures
# ===========================================================================


@pytest.fixture()
def mock_registry():
    """Return a mock tool registry."""
    reg = MagicMock()
    tool = MagicMock()
    tool.command = ["crosshair", "check"]
    tool.timeout = 30
    reg.get_tool.return_value = tool
    return reg


@pytest.fixture()
def crosshair(mock_registry):
    """Return a CrossHairAnalyzer with mocked registry."""
    return CrossHairAnalyzer(tool_registry=mock_registry, timeout=10, max_iterations=50)


@pytest.fixture()
def pysa_analyzer(tmp_path):
    """Return a PysaAnalyzer with a temp project root."""
    reg = MagicMock()
    tool = MagicMock()
    tool.command = ["pyre", "analyze"]
    tool.timeout = 60
    reg.get_tool.return_value = tool
    return PysaAnalyzer(tool_registry=reg, project_root=tmp_path)


PYTHON_SOURCE_WITH_CONTRACTS = '''\
def add(x: int, y: int) -> int:
    """Add two numbers.

    pre: x >= 0
    post: __return__ >= 0
    """
    return x + y

@requires
def mul(a, b):
    return a * b

def plain_func():
    pass

def prop_check():
    """Examples:
    >>> prop_check()
    """
    pass

@property
def prop_decorated():
    pass
'''

PYTHON_SOURCE_NO_CONTRACTS = """\
def foo():
    pass

def bar():
    pass
"""


# ===========================================================================
#  CrossHairAnalyzer — dataclass sanity
# ===========================================================================


class TestDataclasses:
    def test_contract_violation_defaults(self):
        v = ContractViolation(
            function_name="f", line_number=1, violation_type="precondition", message="bad"
        )
        assert v.counterexample is None
        assert v.contract_expression is None
        assert v.file_path is None

    def test_property_test_result_defaults(self):
        r = PropertyTestResult(function_name="f", property_name="p", passed=True, num_examples=10)
        assert r.counterexample is None
        assert r.error_message is None
        assert r.execution_time == 0.0

    def test_crosshair_analysis_result_defaults(self):
        r = CrosshairAnalysisResult(file_path="/a.py")
        assert r.contract_violations == []
        assert r.property_test_results == []
        assert r.verified_functions == []
        assert r.analysis_errors == []
        assert r.total_contracts_checked == 0
        assert r.total_properties_tested == 0
        assert r.execution_time == 0.0


# ===========================================================================
#  CrossHairAnalyzer._check_crosshair_availability
# ===========================================================================


class TestCrossHairAvailability:
    def test_returns_cached_value(self, crosshair):
        crosshair._crosshair_available = True
        assert crosshair._check_crosshair_availability() is True

    def test_available_when_tool_and_subprocess_succeed(self, crosshair):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert crosshair._check_crosshair_availability() is True
        assert crosshair._crosshair_available is True

    def test_unavailable_when_returncode_nonzero(self, crosshair):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert crosshair._check_crosshair_availability() is False

    def test_unavailable_when_tool_not_found(self, crosshair):
        crosshair.tool_registry.get_tool.return_value = None
        assert crosshair._check_crosshair_availability() is False

    def test_unavailable_on_timeout(self, crosshair):
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            assert crosshair._check_crosshair_availability() is False

    def test_unavailable_on_file_not_found(self, crosshair):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert crosshair._check_crosshair_availability() is False


# ===========================================================================
#  CrossHairAnalyzer.analyze (legacy async)
# ===========================================================================


class TestCrossHairAnalyzeLegacy:
    async def test_tool_not_found(self, crosshair):
        crosshair.tool_registry.get_tool.return_value = None
        result = await crosshair.analyze("test.py")
        assert result["status"] == "tool_not_found"

    async def test_passed(self, crosshair):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            result = await crosshair.analyze("test.py")
        assert result["status"] == "passed"
        assert result["violations"] == []

    async def test_failed_with_violations(self, crosshair):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="error: some violation\nother line\n",
            )
            result = await crosshair.analyze("test.py")
        assert result["status"] == "failed"
        assert len(result["violations"]) == 1

    async def test_warning_when_no_violations_parsed(self, crosshair):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="just a warning, nothing found\n",
            )
            result = await crosshair.analyze("test.py")
        assert result["status"] == "warning"

    async def test_timeout_raises(self, crosshair):
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
            with pytest.raises(CrossHairAnalysisError, match="timeout"):
                await crosshair.analyze("test.py")

    async def test_called_process_error_raises(self, crosshair):
        import subprocess

        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(2, "crosshair"),
        ):
            with pytest.raises(CrossHairAnalysisError, match="process error"):
                await crosshair.analyze("test.py")

    async def test_os_error_raises(self, crosshair):
        with patch("subprocess.run", side_effect=OSError("boom")):
            with pytest.raises(CrossHairAnalysisError, match="OS error"):
                await crosshair.analyze("test.py")


# ===========================================================================
#  CrossHairAnalyzer.analyze_file
# ===========================================================================


class TestCrossHairAnalyzeFile:
    def test_not_available(self, crosshair, tmp_path):
        crosshair._crosshair_available = False
        result = crosshair.analyze_file(tmp_path / "test.py")
        assert "CrossHair is not available" in result.analysis_errors

    def test_file_parse_error(self, crosshair, tmp_path):
        crosshair._crosshair_available = True
        bad = tmp_path / "bad.py"
        bad.write_text("def (broken syntax")
        result = crosshair.analyze_file(bad)
        assert any("Failed to parse" in e for e in result.analysis_errors)

    def test_happy_path(self, crosshair, tmp_path):
        crosshair._crosshair_available = True
        f = tmp_path / "good.py"
        f.write_text(PYTHON_SOURCE_WITH_CONTRACTS)
        with patch.object(crosshair, "_run_crosshair_check", return_value=[]):
            with patch.object(
                crosshair,
                "_test_property",
                return_value=PropertyTestResult(
                    function_name="prop_check",
                    property_name="prop_check",
                    passed=True,
                    num_examples=50,
                ),
            ):
                result = crosshair.analyze_file(f)
        assert result.total_contracts_checked > 0
        assert result.analysis_errors == []

    def test_verified_functions_exclude_violated(self, crosshair, tmp_path):
        crosshair._crosshair_available = True
        f = tmp_path / "test.py"
        f.write_text(PYTHON_SOURCE_WITH_CONTRACTS)
        violation = ContractViolation(
            function_name="add",
            line_number=1,
            violation_type="postcondition",
            message="bad",
        )
        with patch.object(crosshair, "_run_crosshair_check", return_value=[violation]):
            with patch.object(
                crosshair,
                "_test_property",
                return_value=PropertyTestResult(
                    function_name="x", property_name="x", passed=True, num_examples=1
                ),
            ):
                result = crosshair.analyze_file(f)
        assert "add" not in result.verified_functions


# ===========================================================================
#  CrossHairAnalyzer._extract_contracted_functions
# ===========================================================================


class TestExtractContractedFunctions:
    def test_finds_contracts(self, crosshair):
        tree = ast.parse(PYTHON_SOURCE_WITH_CONTRACTS)
        funcs = crosshair._extract_contracted_functions(tree, PYTHON_SOURCE_WITH_CONTRACTS)
        assert "add" in funcs  # docstring contract
        assert "mul" in funcs  # decorator contract
        assert "plain_func" not in funcs

    def test_empty_source(self, crosshair):
        tree = ast.parse("")
        assert crosshair._extract_contracted_functions(tree, "") == []


# ===========================================================================
#  CrossHairAnalyzer._run_crosshair_check
# ===========================================================================


class TestRunCrossHairCheck:
    def test_tool_not_found(self, crosshair, tmp_path):
        crosshair.tool_registry.get_tool.return_value = None
        assert crosshair._run_crosshair_check(tmp_path / "test.py") == []

    def test_parses_stdout_and_stderr(self, crosshair, tmp_path):
        f = tmp_path / "test.py"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="",
                stderr="some error line\nWARNING: ignored\n",
            )
            violations = crosshair._run_crosshair_check(f)
        # One error line, WARNING line skipped
        assert len(violations) == 1
        assert violations[0].message == "some error line"

    def test_timeout_produces_violation(self, crosshair, tmp_path):
        import subprocess

        f = tmp_path / "test.py"
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("c", 10)):
            violations = crosshair._run_crosshair_check(f)
        assert len(violations) == 1
        assert violations[0].violation_type == "timeout"

    def test_generic_exception_produces_violation(self, crosshair, tmp_path):
        f = tmp_path / "test.py"
        with patch("subprocess.run", side_effect=RuntimeError("boom")):
            violations = crosshair._run_crosshair_check(f)
        assert len(violations) == 1
        assert "boom" in violations[0].message


# ===========================================================================
#  CrossHairAnalyzer._parse_crosshair_output
# ===========================================================================


class TestParseCrossHairOutput:
    def test_empty_output(self, crosshair, tmp_path):
        assert crosshair._parse_crosshair_output("", tmp_path / "f.py") == []

    def test_counterexample_found(self, crosshair, tmp_path):
        output = (
            "module.my_func: counterexample found: postcondition violated\n"
            "  * x = 5\n"
            "  * y = -3\n"
            "\n"
        )
        violations = crosshair._parse_crosshair_output(output, tmp_path / "f.py")
        assert len(violations) == 1
        assert violations[0].function_name == "my_func"
        assert violations[0].counterexample == {"x": "5", "y": "-3"}
        assert violations[0].violation_type == "postcondition"

    def test_multiple_violations(self, crosshair, tmp_path):
        output = (
            "module.func_a: counterexample found: precondition requires\n"
            "  * a = 0\n"
            "\n"
            "module.func_b: counterexample found: invariant broken\n"
            "  * b = 1\n"
        )
        violations = crosshair._parse_crosshair_output(output, tmp_path / "f.py")
        assert len(violations) == 2

    def test_trailing_violation_no_blank_line(self, crosshair, tmp_path):
        output = "mod.f: counterexample found: bad ensures\n  * z = 9\n"
        violations = crosshair._parse_crosshair_output(output, tmp_path / "f.py")
        assert len(violations) == 1
        assert violations[0].counterexample == {"z": "9"}


# ===========================================================================
#  CrossHairAnalyzer._parse_output (legacy)
# ===========================================================================


class TestParseOutputLegacy:
    def test_empty(self, crosshair):
        assert crosshair._parse_output("") == []

    def test_finds_errors_and_violations(self, crosshair):
        text = "line 1 error here\nnothing\nViolation detected\n"
        result = crosshair._parse_output(text)
        assert len(result) == 2

    def test_none_output(self, crosshair):
        assert crosshair._parse_output(None) == []


# ===========================================================================
#  CrossHairAnalyzer helper methods
# ===========================================================================


class TestCrossHairHelpers:
    def test_extract_function_name_dotted(self, crosshair):
        assert crosshair._extract_function_name("module.submod.func") == "func"

    def test_extract_function_name_at(self, crosshair):
        assert crosshair._extract_function_name("func at line 5") == "func"

    def test_extract_function_name_plain(self, crosshair):
        assert crosshair._extract_function_name("myfunc") == "myfunc"

    def test_extract_line_number_present(self, crosshair):
        assert crosshair._extract_line_number("func at line 42 stuff") == 42

    def test_extract_line_number_missing(self, crosshair):
        assert crosshair._extract_line_number("no line info") == 0

    def test_extract_line_number_bad_value(self, crosshair):
        assert crosshair._extract_line_number("at line xyz") == 0

    def test_determine_violation_precondition(self, crosshair):
        assert crosshair._determine_violation_type("Precondition failed") == "precondition"

    def test_determine_violation_postcondition(self, crosshair):
        assert crosshair._determine_violation_type("ensures not met") == "postcondition"

    def test_determine_violation_invariant(self, crosshair):
        assert crosshair._determine_violation_type("invariant broken") == "invariant"

    def test_determine_violation_requires(self, crosshair):
        assert crosshair._determine_violation_type("requires x > 0") == "precondition"

    def test_determine_violation_unknown(self, crosshair):
        assert crosshair._determine_violation_type("something else") == "contract"


# ===========================================================================
#  CrossHairAnalyzer._extract_properties
# ===========================================================================


class TestExtractProperties:
    def test_finds_property_by_name(self, crosshair):
        src = "def prop_my_test():\n    pass\n"
        tree = ast.parse(src)
        props = crosshair._extract_properties(tree, src)
        assert any(name == "prop_my_test" for name, _ in props)

    def test_finds_test_prop_prefix(self, crosshair):
        src = "def test_prop_foo():\n    pass\n"
        tree = ast.parse(src)
        props = crosshair._extract_properties(tree, src)
        assert len(props) == 1

    def test_finds_property_decorator(self, crosshair):
        src = "@property\ndef my_prop():\n    pass\n"
        tree = ast.parse(src)
        props = crosshair._extract_properties(tree, src)
        assert len(props) == 1

    def test_finds_hypothesis_given_attribute(self, crosshair):
        src = "@hypothesis.given\ndef my_test():\n    pass\n"
        tree = ast.parse(src)
        props = crosshair._extract_properties(tree, src)
        assert len(props) == 1

    def test_skips_regular_functions(self, crosshair):
        src = "def regular():\n    pass\n"
        tree = ast.parse(src)
        assert crosshair._extract_properties(tree, src) == []


# ===========================================================================
#  CrossHairAnalyzer._test_property
# ===========================================================================


class TestTestProperty:
    def test_tool_not_available(self, crosshair, tmp_path):
        crosshair.tool_registry.get_tool.return_value = None
        node = ast.parse("def prop_x():\n    pass\n").body[0]
        result = crosshair._test_property(("prop_x", node), tmp_path / "test.py")
        assert result.passed is False
        assert "not available" in result.error_message

    def test_passed_property(self, crosshair, tmp_path):
        node = ast.parse("def prop_x():\n    pass\n").body[0]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="all good", stderr="")
            result = crosshair._test_property(("prop_x", node), tmp_path / "test.py")
        assert result.passed is True
        assert result.num_examples == 50

    def test_failed_property_with_counterexample(self, crosshair, tmp_path):
        node = ast.parse("def prop_x():\n    pass\n").body[0]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="counterexample found\n  * x = 5\n", stderr="")
            result = crosshair._test_property(("prop_x", node), tmp_path / "test.py")
        assert result.passed is False
        assert result.counterexample is not None

    def test_exception_during_test(self, crosshair, tmp_path):
        node = ast.parse("def prop_x():\n    pass\n").body[0]
        with patch("subprocess.run", side_effect=RuntimeError("boom")):
            result = crosshair._test_property(("prop_x", node), tmp_path / "test.py")
        assert result.passed is False
        assert "boom" in result.error_message


# ===========================================================================
#  CrossHairAnalyzer._parse_counterexample
# ===========================================================================


class TestParseCounterexample:
    def test_parses_literal_values(self, crosshair):
        output = "  * x = 42\n  * y = 'hello'\n  other stuff\n"
        result = crosshair._parse_counterexample(output)
        assert result["x"] == 42
        assert result["y"] == "hello"

    def test_non_literal_kept_as_string(self, crosshair):
        output = "  * obj = <SomeClass>\n"
        result = crosshair._parse_counterexample(output)
        assert result["obj"] == "<SomeClass>"

    def test_empty_output(self, crosshair):
        assert crosshair._parse_counterexample("") == {}


# ===========================================================================
#  CrossHairAnalyzer.verify_function_contract
# ===========================================================================


class TestVerifyFunctionContract:
    def test_with_source_segment(self, crosshair):
        source = "def add(x, y):\n    return x + y\n"
        tree = ast.parse(source)
        func_node = tree.body[0]
        with patch.object(crosshair, "_run_crosshair_check", return_value=[]):
            violations = crosshair.verify_function_contract(func_node, source)
        assert violations == []

    def test_no_source_segment_fallback_error(self, crosshair):
        source = "def add(x, y):\n    return x + y\n"
        tree = ast.parse(source)
        func_node = tree.body[0]
        with patch("ast.get_source_segment", return_value=None):
            violations = crosshair.verify_function_contract(func_node, source)
        assert len(violations) == 1
        assert violations[0].violation_type == "error"


# ===========================================================================
#  CrossHairAnalyzer.generate_contract_report
# ===========================================================================


class TestGenerateContractReport:
    def test_empty_result(self, crosshair):
        result = CrosshairAnalysisResult(file_path="/test.py")
        report = crosshair.generate_contract_report(result)
        assert "/test.py" in report
        assert "Total contracts checked: 0" in report

    def test_with_violations(self, crosshair):
        v = ContractViolation(
            function_name="bad_func",
            line_number=10,
            violation_type="postcondition",
            message="contract broken",
            counterexample={"x": 5, "y": -1},
        )
        result = CrosshairAnalysisResult(
            file_path="/test.py",
            contract_violations=[v],
            total_contracts_checked=3,
        )
        report = crosshair.generate_contract_report(result)
        assert "bad_func" in report
        assert "postcondition" in report
        assert "x = 5" in report

    def test_with_property_results(self, crosshair):
        passed = PropertyTestResult(
            function_name="prop_ok", property_name="prop_ok", passed=True, num_examples=100
        )
        failed = PropertyTestResult(
            function_name="prop_bad",
            property_name="prop_bad",
            passed=False,
            num_examples=1,
            counterexample={"a": 0},
        )
        result = CrosshairAnalysisResult(
            file_path="/test.py",
            property_test_results=[passed, failed],
            total_properties_tested=2,
        )
        report = crosshair.generate_contract_report(result)
        assert "PASSED" in report
        assert "FAILED" in report
        assert "a = 0" in report

    def test_with_verified_functions(self, crosshair):
        result = CrosshairAnalysisResult(
            file_path="/test.py",
            verified_functions=["safe_add", "safe_mul"],
        )
        report = crosshair.generate_contract_report(result)
        assert "safe_add" in report
        assert "safe_mul" in report


# ===========================================================================
#  PysaAnalyzer.__init__ / _setup_pysa_config
# ===========================================================================


class TestPysaInit:
    def test_creates_config_if_missing(self, tmp_path):
        PysaAnalyzer(project_root=tmp_path)
        config_path = tmp_path / ".pyre_configuration"
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert "source_directories" in data

    def test_does_not_overwrite_existing_config(self, tmp_path):
        config_path = tmp_path / ".pyre_configuration"
        config_path.write_text('{"custom": true}')
        PysaAnalyzer(project_root=tmp_path)
        assert json.loads(config_path.read_text()) == {"custom": True}

    def test_defaults_to_cwd(self):
        with patch.object(Path, "exists", return_value=True):
            analyzer = PysaAnalyzer()
            assert analyzer.project_root == Path.cwd()


# ===========================================================================
#  PysaAnalyzer.analyze (non-legacy)
# ===========================================================================


class TestPysaAnalyze:
    def test_skips_non_python(self, pysa_analyzer):
        result = pysa_analyzer.analyze(Path("/test.js"))
        assert result.file_path == "/test.js"
        assert result.security_issues == []

    def test_successful_analysis(self, pysa_analyzer):
        pysa_output = json.dumps(
            {
                "errors": [
                    {
                        "path": "target.py",
                        "code": "5001",
                        "level": "error",
                        "description": "SQL injection",
                        "line": 10,
                        "column": 5,
                        "code_snippet": "cursor.execute(query)",
                    }
                ],
                "taint_flows": [],
            }
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=pysa_output, stderr="")
            result = pysa_analyzer.analyze(Path("/project/target.py"))
        assert result.metrics["total_issues"] >= 0

    def test_analysis_failure_returns_empty(self, pysa_analyzer):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="pyre error")
            result = pysa_analyzer.analyze(Path("/test.py"))
        assert result.security_issues == []

    def test_analysis_timeout(self, pysa_analyzer):
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("c", 300)):
            result = pysa_analyzer.analyze(Path("/test.py"))
        assert result.security_issues == []

    def test_analysis_generic_exception(self, pysa_analyzer):
        with patch("subprocess.run", side_effect=RuntimeError("boom")):
            result = pysa_analyzer.analyze(Path("/test.py"))
        assert result.security_issues == []


# ===========================================================================
#  PysaAnalyzer.analyze_legacy (async)
# ===========================================================================


class TestPysaAnalyzeLegacy:
    async def test_tool_not_found(self, pysa_analyzer):
        pysa_analyzer.tool_registry.get_tool.return_value = None
        result = await pysa_analyzer.analyze_legacy("/test.py")
        assert result["status"] == "tool_not_found"

    async def test_successful_json_output(self, pysa_analyzer, tmp_path):
        src = tmp_path / "input.py"
        src.write_text("x = 1\n")
        pysa_data = {
            "errors": [
                {"name": "SQLInjection", "line": 5, "column": 1, "description": "tainted flow"}
            ]
        }
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=json.dumps(pysa_data), stderr="")
            result = await pysa_analyzer.analyze_legacy(str(src))
        assert result["status"] == "completed"

    async def test_non_json_output(self, pysa_analyzer, tmp_path):
        src = tmp_path / "input.py"
        src.write_text("x = 1\n")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="not json", stderr="")
            result = await pysa_analyzer.analyze_legacy(str(src))
        assert result["status"] == "completed"
        assert "raw_output" in result

    async def test_no_output(self, pysa_analyzer, tmp_path):
        src = tmp_path / "input.py"
        src.write_text("x = 1\n")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="err")
            result = await pysa_analyzer.analyze_legacy(str(src))
        assert result["status"] == "no_output"

    async def test_timeout_raises(self, pysa_analyzer, tmp_path):
        import subprocess

        src = tmp_path / "input.py"
        src.write_text("x = 1\n")
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("c", 60)):
            with pytest.raises(PysaAnalysisError, match="timeout"):
                await pysa_analyzer.analyze_legacy(str(src))

    async def test_called_process_error_raises(self, pysa_analyzer, tmp_path):
        import subprocess

        src = tmp_path / "input.py"
        src.write_text("x = 1\n")
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "pyre"),
        ):
            with pytest.raises(PysaAnalysisError, match="process error"):
                await pysa_analyzer.analyze_legacy(str(src))

    async def test_os_error_raises(self, pysa_analyzer, tmp_path):
        src = tmp_path / "input.py"
        src.write_text("x = 1\n")
        with patch("subprocess.run", side_effect=OSError("disk full")):
            with pytest.raises(PysaAnalysisError, match="IO error"):
                await pysa_analyzer.analyze_legacy(str(src))


# ===========================================================================
#  PysaAnalyzer._parse_output
# ===========================================================================


class TestPysaParseOutput:
    def test_with_errors(self, pysa_analyzer):
        data = {
            "errors": [
                {"name": "SQLInjection", "line": 10, "column": 5, "description": "tainted"},
                {"name": "CommandInjection", "line": 20, "column": 3, "description": "cmd"},
            ]
        }
        result = pysa_analyzer._parse_output(data)
        assert result["status"] == "completed"
        assert len(result["issues"]) == 2
        assert result["safety_score"] <= 100

    def test_empty_errors(self, pysa_analyzer):
        result = pysa_analyzer._parse_output({"errors": []})
        assert result["safety_score"] == 100
        assert result["issues"] == []

    def test_no_errors_key(self, pysa_analyzer):
        result = pysa_analyzer._parse_output({})
        assert result["safety_score"] == 100


# ===========================================================================
#  PysaAnalyzer._parse_security_issue
# ===========================================================================


class TestPysaParseSecurityIssue:
    def test_basic_issue(self, pysa_analyzer):
        error = {
            "name": "SQLInjection",
            "line": 42,
            "column": 8,
            "description": "tainted data flows to sql",
        }
        issue = pysa_analyzer._parse_security_issue(error)
        assert issue["type"] == "SQLInjection"
        assert issue["severity"] == "high"
        assert issue["line"] == 42

    def test_defaults(self, pysa_analyzer):
        issue = pysa_analyzer._parse_security_issue({})
        assert issue["type"] == "Unknown"
        assert issue["line"] == 0


# ===========================================================================
#  PysaAnalyzer._determine_severity
# ===========================================================================


class TestDetermineSeverity:
    @pytest.mark.parametrize(
        "issue_type,expected",
        [
            ("SQLInjection", "high"),
            ("CommandExecution", "high"),
            ("XSSAttack", "high"),
            ("RCEVulnerability", "high"),
            ("TaintFlow", "medium"),
            ("PathTraversal", "medium"),
            ("SSRFIssue", "medium"),
            ("SomethingElse", "low"),
        ],
    )
    def test_severity_mapping(self, pysa_analyzer, issue_type, expected):
        assert pysa_analyzer._determine_severity(issue_type) == expected


# ===========================================================================
#  PysaAnalyzer._get_recommendation
# ===========================================================================


class TestGetRecommendation:
    def test_sql(self, pysa_analyzer):
        assert "parameterized" in pysa_analyzer._get_recommendation("SQLInjection")

    def test_command(self, pysa_analyzer):
        assert "subprocess" in pysa_analyzer._get_recommendation("CommandExec")

    def test_xss(self, pysa_analyzer):
        assert "Sanitize" in pysa_analyzer._get_recommendation("XSSAttack")

    def test_path(self, pysa_analyzer):
        assert "paths" in pysa_analyzer._get_recommendation("PathTraversal")

    def test_taint(self, pysa_analyzer):
        assert "Track" in pysa_analyzer._get_recommendation("TaintFlow")

    def test_unknown(self, pysa_analyzer):
        rec = pysa_analyzer._get_recommendation("UnknownIssue")
        assert "Review" in rec


# ===========================================================================
#  PysaAnalyzer._generate_recommendations
# ===========================================================================


class TestGenerateRecommendations:
    def test_deduplicates(self, pysa_analyzer):
        issues = [
            {"type": "SQLInjection"},
            {"type": "SQLInjection"},
        ]
        recs = pysa_analyzer._generate_recommendations(issues)
        assert len(recs) == 1

    def test_multiple_types(self, pysa_analyzer):
        issues = [
            {"type": "SQLInjection"},
            {"type": "CommandExec"},
        ]
        recs = pysa_analyzer._generate_recommendations(issues)
        assert len(recs) == 2


# ===========================================================================
#  PysaAnalyzer._parse_pysa_output
# ===========================================================================


class TestParsePysaOutput:
    def test_valid_json_with_matching_file(self, pysa_analyzer):
        output = json.dumps(
            {
                "errors": [
                    {
                        "path": "/project/target.py",
                        "code": "5001",
                        "level": "error",
                        "description": "SQL injection",
                        "line": 10,
                        "column": 5,
                        "code_snippet": "cursor.execute(q)",
                    }
                ]
            }
        )
        issues = pysa_analyzer._parse_pysa_output(output, Path("/project/target.py"))
        assert len(issues) == 1
        assert issues[0].type == "SQL Injection"

    def test_valid_json_no_matching_file(self, pysa_analyzer):
        output = json.dumps(
            {
                "errors": [
                    {
                        "path": "/other/file.py",
                        "code": "5001",
                        "description": "x",
                        "line": 1,
                        "column": 1,
                    }
                ]
            }
        )
        issues = pysa_analyzer._parse_pysa_output(output, Path("/project/target.py"))
        assert len(issues) == 0

    def test_invalid_json_fallback(self, pysa_analyzer):
        output = "target.py:10:5: SQL injection found\nnot_matching line\n"
        issues = pysa_analyzer._parse_pysa_output(output, Path("/project/target.py"))
        assert len(issues) == 1

    def test_empty_output(self, pysa_analyzer):
        issues = pysa_analyzer._parse_pysa_output("{}", Path("/test.py"))
        assert issues == []


# ===========================================================================
#  PysaAnalyzer._extract_taint_flows
# ===========================================================================


class TestExtractTaintFlows:
    def test_valid_flows(self, pysa_analyzer):
        output = json.dumps(
            {
                "taint_flows": [
                    {
                        "source": {"name": "user_input", "location": {"file": "a.py", "line": 1}},
                        "sink": {"name": "sql_exec", "location": {"file": "b.py", "line": 10}},
                        "path": ["a.py:1", "b.py:10"],
                        "issue_type": "sql_injection",
                    }
                ]
            }
        )
        flows = pysa_analyzer._extract_taint_flows(output)
        assert len(flows) == 1
        assert flows[0]["source"] == "user_input"
        assert flows[0]["sink"] == "sql_exec"

    def test_empty_flows(self, pysa_analyzer):
        assert pysa_analyzer._extract_taint_flows(json.dumps({"taint_flows": []})) == []

    def test_invalid_json(self, pysa_analyzer):
        assert pysa_analyzer._extract_taint_flows("not json") == []


# ===========================================================================
#  PysaAnalyzer._parse_issue_line
# ===========================================================================


class TestParseIssueLine:
    def test_valid_line(self, pysa_analyzer):
        issue = pysa_analyzer._parse_issue_line(
            "target.py:10:5: SQL injection found", Path("/project/target.py")
        )
        assert issue is not None
        assert issue.line_number == 10
        assert issue.column_number == 5

    def test_invalid_line_format(self, pysa_analyzer):
        assert pysa_analyzer._parse_issue_line("not:a:valid", Path("/f.py")) is None

    def test_non_numeric_line(self, pysa_analyzer):
        assert pysa_analyzer._parse_issue_line("f.py:abc:5: msg", Path("/f.py")) is None


# ===========================================================================
#  PysaAnalyzer mapping helpers
# ===========================================================================


class TestPysaMappingHelpers:
    @pytest.mark.parametrize(
        "code,expected",
        [
            ("5001", "SQL Injection"),
            ("5002", "Command Injection"),
            ("5003", "Path Traversal"),
            ("5004", "Cross-Site Scripting (XSS)"),
            ("5005", "Remote Code Execution"),
            ("5006", "Server-Side Request Forgery (SSRF)"),
            ("5007", "XML External Entity (XXE)"),
            ("5008", "Insecure Deserialization"),
            ("5009", "LDAP Injection"),
            ("5010", "NoSQL Injection"),
            ("9999", "Security Issue (9999)"),
        ],
    )
    def test_map_issue_type(self, pysa_analyzer, code, expected):
        assert pysa_analyzer._map_issue_type(code) == expected

    @pytest.mark.parametrize(
        "level,expected",
        [
            ("error", "critical"),
            ("warning", "high"),
            ("advice", "medium"),
            ("info", "low"),
            ("ERROR", "critical"),
            ("unknown", "medium"),
        ],
    )
    def test_map_severity(self, pysa_analyzer, level, expected):
        assert pysa_analyzer._map_severity(level) == expected

    @pytest.mark.parametrize(
        "code,expected",
        [
            ("5001", "CWE-89"),
            ("5002", "CWE-78"),
            ("5003", "CWE-22"),
            ("5004", "CWE-79"),
            ("5005", "CWE-94"),
            ("5006", "CWE-918"),
            ("5007", "CWE-611"),
            ("5008", "CWE-502"),
            ("5009", "CWE-90"),
            ("5010", "CWE-943"),
            ("9999", None),
        ],
    )
    def test_map_cwe_id(self, pysa_analyzer, code, expected):
        assert pysa_analyzer._map_cwe_id(code) == expected


# ===========================================================================
#  PysaAnalyzer._get_recommendation_enhanced
# ===========================================================================


class TestGetRecommendationEnhanced:
    def test_known_code(self, pysa_analyzer):
        rec = pysa_analyzer._get_recommendation_enhanced("5001")
        assert "parameterized" in rec

    def test_unknown_code(self, pysa_analyzer):
        rec = pysa_analyzer._get_recommendation_enhanced("0000")
        assert "Review" in rec


# ===========================================================================
#  PysaAnalyzer._extract_issue_type
# ===========================================================================


class TestExtractIssueType:
    @pytest.mark.parametrize(
        "message,expected",
        [
            ("SQL injection detected", "SQL Injection"),
            ("Command injection found", "Command Injection"),
            ("Path traversal vulnerability", "Path Traversal"),
            ("Directory traversal attack", "Path Traversal"),
            ("XSS vulnerability", "Cross-Site Scripting (XSS)"),
            ("Cross-Site Scripting found", "Cross-Site Scripting (XSS)"),
            ("Remote Code Execution", "Remote Code Execution"),
            ("RCE detected", "Remote Code Execution"),
            ("SSRF issue", "Server-Side Request Forgery (SSRF)"),
            ("server-side request forgery", "Server-Side Request Forgery (SSRF)"),
            ("XXE vulnerability", "XML External Entity (XXE)"),
            ("XML External Entity attack", "XML External Entity (XXE)"),
            ("Insecure deserialization", "Insecure Deserialization"),
            ("LDAP injection found", "LDAP Injection"),
            ("NoSQL injection detected", "NoSQL Injection"),
            ("Some unknown issue", "Security Issue"),
        ],
    )
    def test_extract_issue_type(self, pysa_analyzer, message, expected):
        assert pysa_analyzer._extract_issue_type(message) == expected


# ===========================================================================
#  PysaAnalyzer._get_recommendation_from_message
# ===========================================================================


class TestGetRecommendationFromMessage:
    def test_sql_message(self, pysa_analyzer):
        assert "parameterized" in pysa_analyzer._get_recommendation_from_message("SQL injection")

    def test_command_message(self, pysa_analyzer):
        assert "command" in pysa_analyzer._get_recommendation_from_message("command injection")

    def test_path_message(self, pysa_analyzer):
        assert "path" in pysa_analyzer._get_recommendation_from_message("path traversal").lower()

    def test_directory_message(self, pysa_analyzer):
        rec = pysa_analyzer._get_recommendation_from_message("directory traversal")
        assert "path" in rec.lower() or "directory" in rec.lower()

    def test_xss_message(self, pysa_analyzer):
        assert "Escape" in pysa_analyzer._get_recommendation_from_message("XSS attack")

    def test_script_message(self, pysa_analyzer):
        assert "Escape" in pysa_analyzer._get_recommendation_from_message("script injection")

    def test_unknown_message(self, pysa_analyzer):
        assert "Review" in pysa_analyzer._get_recommendation_from_message("something unknown")


# ===========================================================================
#  PysaAnalyzer._extract_cwe_from_message
# ===========================================================================


class TestExtractCweFromMessage:
    def test_cwe_present(self, pysa_analyzer):
        assert pysa_analyzer._extract_cwe_from_message("Found CWE-89 issue") == "CWE-89"

    def test_cwe_absent(self, pysa_analyzer):
        assert pysa_analyzer._extract_cwe_from_message("no cwe here") is None


# ===========================================================================
#  PysaAnalyzer._create_empty_result
# ===========================================================================


class TestCreateEmptyResult:
    def test_empty_result(self, pysa_analyzer):
        result = pysa_analyzer._create_empty_result(Path("/test.py"))
        assert result.file_path == "/test.py"
        assert result.security_issues == []
        assert result.taint_flows == []
        assert result.metrics["total_issues"] == 0


# ===========================================================================
#  PysaAnalyzer.get_custom_taint_models
# ===========================================================================


class TestGetCustomTaintModels:
    def test_returns_models(self, pysa_analyzer):
        models = pysa_analyzer.get_custom_taint_models()
        assert len(models) == 1
        assert "sources" in models[0]
        assert "sinks" in models[0]
        assert "rules" in models[0]


# ===========================================================================
#  PysaAnalyzer.generate_taint_config
# ===========================================================================


class TestGenerateTaintConfig:
    def test_creates_config_file(self, pysa_analyzer, tmp_path):
        output = tmp_path / "subdir" / "taint.json"
        pysa_analyzer.generate_taint_config(output)
        assert output.exists()
        data = json.loads(output.read_text())
        assert "sources" in data
        assert "sinks" in data
        assert "features" in data
        assert "rules" in data


# ===========================================================================
#  PysaAnalyzer project config helpers
# ===========================================================================


class TestPysaProjectHelpers:
    def test_get_project_sources(self, pysa_analyzer):
        sources = pysa_analyzer._get_project_sources()
        assert len(sources) == 4
        names = {s["name"] for s in sources}
        assert "UserControlled" in names

    def test_get_project_sinks(self, pysa_analyzer):
        sinks = pysa_analyzer._get_project_sinks()
        assert len(sinks) == 5
        names = {s["name"] for s in sinks}
        assert "SQL" in names

    def test_get_project_features(self, pysa_analyzer):
        features = pysa_analyzer._get_project_features()
        assert len(features) == 3

    def test_get_project_rules(self, pysa_analyzer):
        rules = pysa_analyzer._get_project_rules()
        assert len(rules) == 3
        codes = {r["code"] for r in rules}
        assert codes == {5001, 5002, 5003}
