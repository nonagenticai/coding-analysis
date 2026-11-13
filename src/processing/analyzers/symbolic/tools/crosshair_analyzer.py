"""CrossHair analyzer for property-based testing and contract verification.

This module provides integration with CrossHair, a tool that uses symbolic execution
to verify function contracts and perform property-based testing.
"""

from __future__ import annotations

import ast
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..exceptions import CrossHairAnalysisError

logger = logging.getLogger(__name__)


@dataclass
class ContractViolation:
    """Represents a contract violation found by CrossHair."""

    function_name: str
    line_number: int
    violation_type: str  # 'precondition', 'postcondition', 'invariant'
    message: str
    counterexample: dict[str, Any] | None = None
    contract_expression: str | None = None
    file_path: str | None = None


@dataclass
class PropertyTestResult:
    """Result of property-based testing."""

    function_name: str
    property_name: str
    passed: bool
    num_examples: int
    counterexample: dict[str, Any] | None = None
    error_message: str | None = None
    execution_time: float = 0.0


@dataclass
class CrosshairAnalysisResult:
    """Complete result of CrossHair analysis."""

    file_path: str
    contract_violations: list[ContractViolation] = field(default_factory=list)
    property_test_results: list[PropertyTestResult] = field(default_factory=list)
    verified_functions: list[str] = field(default_factory=list)
    analysis_errors: list[str] = field(default_factory=list)
    total_contracts_checked: int = 0
    total_properties_tested: int = 0
    execution_time: float = 0.0


class CrossHairAnalyzer:
    """Analyzer for property-based testing and contract verification using CrossHair."""

    def __init__(self, tool_registry, timeout: int = 60, max_iterations: int = 1000):
        """Initialize with tool registry and analysis parameters.

        Args:
            tool_registry: Registry for tool access
            timeout: Maximum time in seconds for each function analysis
            max_iterations: Maximum number of symbolic execution iterations
        """
        self.tool_registry = tool_registry
        self.timeout = timeout
        self.max_iterations = max_iterations
        self._crosshair_available = None

    def _check_crosshair_availability(self) -> bool:
        """Check if CrossHair is available."""
        if self._crosshair_available is not None:
            return self._crosshair_available

        try:
            tool = self.tool_registry.get_tool("python", "crosshair")
            if tool:
                result = subprocess.run(
                    ["crosshair", "--help"], capture_output=True, text=True, timeout=5
                )
                self._crosshair_available = result.returncode == 0
            else:
                self._crosshair_available = False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._crosshair_available = False

        return self._crosshair_available

    async def analyze(self, file_path: str) -> dict[str, Any]:
        """Run CrossHair symbolic execution analysis (legacy interface)."""
        try:
            tool = self.tool_registry.get_tool("python", "crosshair")
            if not tool:
                return {
                    "status": "tool_not_found",
                    "error": "CrossHair not available",
                }

            # Build CrossHair command
            cmd = tool.command.copy()
            cmd.append(file_path)

            # Run CrossHair
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=tool.timeout, check=False
            )

            # Parse CrossHair output
            if result.returncode == 0:
                return {
                    "status": "passed",
                    "edge_cases": [],
                    "violations": [],
                    "output": result.stdout,
                }

            # Parse errors/violations from stderr
            violations = self._parse_output(result.stderr)
            return {
                "status": "failed" if violations else "warning",
                "edge_cases": [],
                "violations": violations,
                "output": result.stderr,
            }

        except subprocess.TimeoutExpired as e:
            logger.warning("CrossHair analysis timeout for %s", file_path)
            raise CrossHairAnalysisError("CrossHair analysis timeout") from e
        except subprocess.CalledProcessError as e:
            logger.error("CrossHair process error: %s", e)
            raise CrossHairAnalysisError(f"CrossHair process error: {e}") from e
        except OSError as e:
            logger.error("OS error running CrossHair: %s", e)
            raise CrossHairAnalysisError(f"OS error: {e}") from e

    def analyze_file(self, file_path: Path) -> CrosshairAnalysisResult:
        """Analyze a Python file for contract violations and property tests.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            CrosshairAnalysisResult containing all findings
        """
        if not self._check_crosshair_availability():
            return CrosshairAnalysisResult(
                file_path=str(file_path), analysis_errors=["CrossHair is not available"]
            )

        result = CrosshairAnalysisResult(file_path=str(file_path))

        # Parse the file to extract contracts and properties
        try:
            with open(file_path) as f:
                source = f.read()
            tree = ast.parse(source, filename=str(file_path))
        except Exception as e:
            result.analysis_errors.append(f"Failed to parse file: {str(e)}")
            return result

        # Extract functions with contracts
        contracted_functions = self._extract_contracted_functions(tree, source)
        result.total_contracts_checked = len(contracted_functions)

        # Run CrossHair check
        violations = self._run_crosshair_check(file_path)
        result.contract_violations.extend(violations)

        # Extract and test properties
        properties = self._extract_properties(tree, source)
        result.total_properties_tested = len(properties)

        for prop in properties:
            test_result = self._test_property(prop, file_path)
            result.property_test_results.append(test_result)

        # Identify verified functions
        violated_functions = {v.function_name for v in result.contract_violations}
        result.verified_functions = [f for f in contracted_functions if f not in violated_functions]

        return result

    def _extract_contracted_functions(self, tree: ast.AST, source: str) -> list[str]:
        """Extract functions that have contracts (pre/post conditions)."""
        contracted_functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for contract decorators or docstring contracts
                has_contract = False

                # Check decorators
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id in [
                        "contract",
                        "requires",
                        "ensures",
                    ]:
                        has_contract = True
                        break

                # Check docstring for contract syntax
                docstring = ast.get_docstring(node)
                if docstring and any(
                    marker in docstring
                    for marker in [
                        "pre:",
                        "post:",
                        "requires:",
                        "ensures:",
                        ">>>",
                        "Examples:",
                    ]
                ):
                    has_contract = True

                if has_contract:
                    contracted_functions.append(node.name)

        return contracted_functions

    def _run_crosshair_check(self, file_path: Path) -> list[ContractViolation]:
        """Run CrossHair check command and parse results."""
        violations = []

        try:
            tool = self.tool_registry.get_tool("python", "crosshair")
            if not tool:
                return violations

            cmd = tool.command.copy()
            cmd.extend(
                [
                    str(file_path),
                    f"--timeout={self.timeout}",
                    f"--max-iterations={self.max_iterations}",
                ]
            )

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout + 10)

            # Parse CrossHair output
            violations.extend(self._parse_crosshair_output(result.stdout, file_path))

            if result.stderr:
                # Parse any errors as analysis errors
                for line in result.stderr.strip().split("\n"):
                    if line and not line.startswith("WARNING"):
                        violations.append(
                            ContractViolation(
                                function_name="<unknown>",
                                line_number=0,
                                violation_type="error",
                                message=line,
                                file_path=str(file_path),
                            )
                        )

        except subprocess.TimeoutExpired:
            violations.append(
                ContractViolation(
                    function_name="<timeout>",
                    line_number=0,
                    violation_type="timeout",
                    message=f"CrossHair analysis timed out after {self.timeout} seconds",
                    file_path=str(file_path),
                )
            )
        except Exception as e:
            violations.append(
                ContractViolation(
                    function_name="<error>",
                    line_number=0,
                    violation_type="error",
                    message=f"CrossHair analysis failed: {str(e)}",
                    file_path=str(file_path),
                )
            )

        return violations

    def _parse_crosshair_output(self, output: str, file_path: Path) -> list[ContractViolation]:
        """Parse CrossHair output to extract violations."""
        violations = []
        current_violation = None

        for line in output.strip().split("\n"):
            # Parse violation headers
            if "counterexample found" in line.lower():
                # Extract function name and violation type
                parts = line.split(":")
                if len(parts) >= 2:
                    func_info = parts[0].strip()
                    message = ":".join(parts[1:]).strip()

                    current_violation = ContractViolation(
                        function_name=self._extract_function_name(func_info),
                        line_number=self._extract_line_number(func_info),
                        violation_type=self._determine_violation_type(message),
                        message=message,
                        file_path=str(file_path),
                    )

            # Parse counterexample details
            elif current_violation and line.strip().startswith("*"):
                # Parse counterexample values
                if not current_violation.counterexample:
                    current_violation.counterexample = {}

                parts = line.strip("* ").split("=")
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    var_value = parts[1].strip()
                    current_violation.counterexample[var_name] = var_value

            # End of current violation
            elif current_violation and (not line.strip() or "found" in line):
                violations.append(current_violation)
                current_violation = None

        # Add any remaining violation
        if current_violation:
            violations.append(current_violation)

        return violations

    def _parse_output(self, output: str) -> list[str]:
        """Parse CrossHair output for violations (legacy method)."""
        violations: list[str] = []
        if not output:
            return violations

        lines = output.strip().split("\n")
        for line in lines:
            if "error" in line.lower() or "violation" in line.lower():
                violations.append(line.strip())

        return violations

    def _extract_function_name(self, func_info: str) -> str:
        """Extract function name from CrossHair output."""
        # Format: "module.function" or "function at line X"
        if "." in func_info:
            return func_info.split(".")[-1].split()[0]
        elif " at " in func_info:
            return func_info.split(" at ")[0].strip()
        else:
            return func_info.strip()

    def _extract_line_number(self, func_info: str) -> int:
        """Extract line number from CrossHair output."""
        if " line " in func_info:
            try:
                return int(func_info.split(" line ")[-1].split()[0])
            except (ValueError, IndexError):  # fmt: skip
                pass
        return 0

    def _determine_violation_type(self, message: str) -> str:
        """Determine the type of contract violation from the message."""
        message_lower = message.lower()
        if "precondition" in message_lower or "requires" in message_lower:
            return "precondition"
        elif "postcondition" in message_lower or "ensures" in message_lower:
            return "postcondition"
        elif "invariant" in message_lower:
            return "invariant"
        else:
            return "contract"

    def _extract_properties(self, tree: ast.AST, source: str) -> list[tuple[str, ast.FunctionDef]]:
        """Extract property-based test functions."""
        properties = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if it's a property test
                is_property = False

                # Check for property decorators
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id in [
                        "property",
                        "hypothesis.given",
                    ]:
                        is_property = True
                        break
                    elif isinstance(decorator, ast.Attribute) and decorator.attr == "given":
                        is_property = True
                        break

                # Check if function name suggests property test
                if node.name.startswith("prop_") or node.name.startswith("test_prop_"):
                    is_property = True

                if is_property:
                    properties.append((node.name, node))

        return properties

    def _test_property(
        self, property_info: tuple[str, ast.FunctionDef], file_path: Path
    ) -> PropertyTestResult:
        """Test a property using CrossHair."""
        prop_name, prop_node = property_info

        # Create a temporary test file for CrossHair
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
            # Write the property test
            tf.write(f"# Property test for {prop_name}\n")
            tf.write(f"from {file_path.stem} import {prop_name}\n\n")
            tf.write("# Run property test\n")
            tf.write(f"crosshair_test_{prop_name}()\n")
            tf.flush()

            # Run CrossHair on the property
            try:
                tool = self.tool_registry.get_tool("python", "crosshair")
                if not tool:
                    return PropertyTestResult(
                        function_name=prop_name,
                        property_name=prop_name,
                        passed=False,
                        num_examples=0,
                        error_message="CrossHair tool not available",
                    )

                cmd = tool.command.copy()
                cmd.extend([tf.name, f"--timeout={self.timeout}"])

                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout + 10
                )

                # Parse results
                if "counterexample found" in result.stdout:
                    return PropertyTestResult(
                        function_name=prop_name,
                        property_name=prop_name,
                        passed=False,
                        num_examples=1,
                        counterexample=self._parse_counterexample(result.stdout),
                        error_message="Property violation found",
                    )
                else:
                    return PropertyTestResult(
                        function_name=prop_name,
                        property_name=prop_name,
                        passed=True,
                        num_examples=self.max_iterations,
                    )

            except Exception as e:
                return PropertyTestResult(
                    function_name=prop_name,
                    property_name=prop_name,
                    passed=False,
                    num_examples=0,
                    error_message=f"Property test failed: {str(e)}",
                )
            finally:
                Path(tf.name).unlink(missing_ok=True)

    def _parse_counterexample(self, output: str) -> dict[str, Any]:
        """Parse counterexample from CrossHair output."""
        counterexample = {}

        for line in output.split("\n"):
            if line.strip().startswith("*"):
                parts = line.strip("* ").split("=")
                if len(parts) == 2:
                    var_name = parts[0].strip()
                    var_value = parts[1].strip()

                    # Try to evaluate the value
                    try:
                        counterexample[var_name] = ast.literal_eval(var_value)
                    except Exception:
                        counterexample[var_name] = var_value

        return counterexample

    def verify_function_contract(
        self, func_ast: ast.FunctionDef, source_code: str
    ) -> list[ContractViolation]:
        """Verify contracts for a specific function.

        Args:
            func_ast: AST node of the function
            source_code: Complete source code containing the function

        Returns:
            List of contract violations found
        """
        # Create a temporary file with just this function
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
            # Write necessary imports and the function
            func_source = ast.get_source_segment(source_code, func_ast)
            if not func_source:
                # Fallback for older Python versions
                import inspect

                try:
                    func_source = inspect.getsource(func_ast)  # type: ignore[invalid-argument-type]  # ty: ignore[invalid-argument-type]
                except Exception:
                    return [
                        ContractViolation(
                            function_name=func_ast.name,
                            line_number=func_ast.lineno,
                            violation_type="error",
                            message="Could not extract function source",
                        )
                    ]

            tf.write("# Isolated function for contract verification\n")
            tf.write(func_source)
            tf.flush()

            # Run CrossHair check
            violations = self._run_crosshair_check(Path(tf.name))

            # Clean up
            Path(tf.name).unlink(missing_ok=True)

            return violations

    def generate_contract_report(self, result: CrosshairAnalysisResult) -> str:
        """Generate a human-readable report of the analysis results."""
        lines = []
        lines.append(f"CrossHair Analysis Report for {result.file_path}")
        lines.append("=" * 60)
        lines.append(f"Total contracts checked: {result.total_contracts_checked}")
        lines.append(f"Total properties tested: {result.total_properties_tested}")
        lines.append(f"Contract violations found: {len(result.contract_violations)}")
        lines.append(f"Verified functions: {len(result.verified_functions)}")
        lines.append("")

        if result.contract_violations:
            lines.append("Contract Violations:")
            lines.append("-" * 40)
            for violation in result.contract_violations:
                lines.append(f"  Function: {violation.function_name}")
                lines.append(f"  Type: {violation.violation_type}")
                lines.append(f"  Message: {violation.message}")
                if violation.counterexample:
                    lines.append("  Counterexample:")
                    for var, val in violation.counterexample.items():
                        lines.append(f"    {var} = {val}")
                lines.append("")

        if result.property_test_results:
            lines.append("Property Test Results:")
            lines.append("-" * 40)
            for test in result.property_test_results:
                status = "PASSED" if test.passed else "FAILED"
                lines.append(f"  Property: {test.property_name} - {status}")
                if not test.passed and test.counterexample:
                    lines.append("  Counterexample:")
                    for var, val in test.counterexample.items():
                        lines.append(f"    {var} = {val}")
                lines.append("")

        if result.verified_functions:
            lines.append("Verified Functions:")
            lines.append("-" * 40)
            for func in result.verified_functions:
                lines.append(f"  ✓ {func}")

        return "\n".join(lines)
