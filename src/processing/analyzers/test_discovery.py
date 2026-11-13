"""
Test Discovery and Parsing Module for Integration Tests.

This module provides functionality for discovering test files, parsing test code,
and extracting test metadata. It serves as the foundation for the integration
testing system by identifying test scenarios and test data structures.

The module supports:
- Test code generation for various analysis scenarios
- Test file and project structure creation
- Test parameter parsing and validation
- Test scenario identification and categorization

Classes:
    TestCodeGenerator: Generates synthetic test code for analysis
    TestFileManager: Manages test file creation and cleanup
    TestScenarioParser: Parses and categorizes test scenarios

Functions:
    create_test_code_samples: Creates standardized test code samples
    parse_test_requirements: Parses test configuration requirements
"""

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TestScenarioType(Enum):
    """Types of test scenarios supported by the system."""

    CALL_GRAPH = "call_graph"
    SIGNATURE_ANALYSIS = "signature_analysis"
    SYMBOLIC_VALIDATION = "symbolic_validation"
    DOCUMENTATION = "documentation"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"


@dataclass
class TestScenario:
    """Represents a test scenario configuration."""

    name: str
    scenario_type: TestScenarioType
    test_code: str
    expected_results: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class TestProject:
    """Represents a test project structure."""

    root_dir: str
    files: dict[str, str]  # filename -> content
    cleanup_required: bool = True


class TestCodeGenerator:
    """Generates synthetic test code for various analysis scenarios.

    This class creates standardized test code that exercises different
    aspects of the static analysis system. Each generated code sample
    is designed to test specific functionality and edge cases.
    """

    def __init__(self):
        """Initialize the test code generator."""
        self.generated_scenarios: list[TestScenario] = []

    def generate_call_graph_test_code(self) -> str:
        """Generate test code for call graph analysis.

        Creates Python code with function call chains, orphaned functions,
        and various call patterns to test call graph generation capabilities.

        Returns:
            str: Python code suitable for call graph analysis testing
        """
        return '''
def main():
    """Main entry point with call chain."""
    result = process_data(get_input())
    save_result(result)
    return result

def get_input():
    """Get input data."""
    return "test input"

def process_data(data):
    """Process input data through validation and transformation."""
    validated = validate_data(data)
    return transform_data(validated)

def validate_data(data):
    """Validate input data."""
    if not data:
        raise ValueError("Invalid data")
    return data

def transform_data(data):
    """Transform validated data."""
    return data.upper()

def save_result(result):
    """Save processing result."""
    print(f"Saving: {result}")

def orphaned_function():
    """This function is never called - should be detected as orphaned."""
    pass

def recursive_function(n):
    """Example of recursive function."""
    if n <= 0:
        return 1
    return n * recursive_function(n - 1)

def complex_call_chain():
    """Complex call chain for testing."""
    step1 = initial_processing()
    step2 = intermediate_processing(step1)
    return final_processing(step2)

def initial_processing():
    """Initial processing step."""
    return "initial"

def intermediate_processing(data):
    """Intermediate processing step."""
    return f"intermediate_{data}"

def final_processing(data):
    """Final processing step."""
    return f"final_{data}"
'''

    def generate_signature_analysis_test_code(self) -> str:
        """Generate test code for function signature analysis.

        Creates Python code with various function signatures, type annotations,
        decorators, and docstrings to test signature extraction capabilities.

        Returns:
            str: Python code suitable for signature analysis testing
        """
        return '''
from typing import List, Dict, Optional, Any, Union, Callable
import json
import asyncio
from functools import wraps

def process_users(users: List[Dict[str, str]], config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Process a list of users with optional configuration.

    Args:
        users: List of user dictionaries with name and email
        config: Optional configuration dictionary

    Returns:
        Dictionary containing processed user data

    Raises:
        ValueError: If users list is invalid
    """
    if not users:
        return {"count": 0, "users": []}

    processed = []
    for user in users:
        processed_user = validate_user(user)
        if processed_user:
            processed.append(processed_user)

    return {
        "count": len(processed),
        "users": processed,
        "config": config or {}
    }

def validate_user(user: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Validate a single user dictionary."""
    if "name" not in user or "email" not in user:
        return None

    return {
        "name": user["name"].strip(),
        "email": user["email"].lower(),
        "validated": True
    }

def legacy_function(data):
    """Legacy function without type annotations."""
    return str(data)

async def async_process(data: str) -> str:
    """Example async function with type annotations."""
    await asyncio.sleep(0.1)
    return data.upper()

@property
def computed_property(self) -> int:
    """Example property with decorator."""
    return 42

def decorator_function(func: Callable) -> Callable:
    """Example decorator function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return wrapper

@decorator_function
def decorated_function(x: int, y: int = 10) -> int:
    """Function with decorator applied."""
    return x + y

def complex_signature(
    mandatory: str,
    optional: Optional[int] = None,
    *args: Any,
    keyword_only: bool = False,
    **kwargs: Any
) -> Union[str, int]:
    """Function with complex signature for testing."""
    if keyword_only:
        return len(mandatory)
    return mandatory

class ExampleClass:
    """Example class for method signature testing."""

    def __init__(self, name: str, value: int = 0):
        """Initialize the example class."""
        self.name = name
        self.value = value

    def instance_method(self, param: str) -> str:
        """Instance method with type annotations."""
        return f"{self.name}: {param}"

    @classmethod
    def class_method(cls, data: Dict[str, Any]) -> 'ExampleClass':
        """Class method example."""
        return cls(data["name"], data.get("value", 0))

    @staticmethod
    def static_method(x: int, y: int) -> int:
        """Static method example."""
        return x * y
'''

    def generate_symbolic_validation_test_code(self) -> str:
        """Generate test code for symbolic validation analysis.

        Creates Python code with potential security issues, edge cases,
        and validation scenarios for symbolic execution testing.

        Returns:
            str: Python code suitable for symbolic validation testing
        """
        return '''
def divide_numbers(a: int, b: int) -> float:
    """Divide two numbers - potential division by zero."""
    return a / b

def safe_divide_numbers(a: int, b: int) -> float:
    """Safe division with zero check."""
    if b == 0:
        raise ValueError("Division by zero")
    return a / b

def process_list(items: list) -> int:
    """Process a list - potential index error."""
    if not items:
        return 0
    return items[0] + items[-1]

def safe_process_list(items: list) -> int:
    """Safe list processing with bounds checking."""
    if not items:
        return 0
    if len(items) == 1:
        return items[0] * 2
    return items[0] + items[-1]

def unsafe_input_processing(user_input: str) -> str:
    """Process user input - potential SQL injection."""
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    return query

def safe_input_processing(user_input: str) -> str:
    """Safe input processing with parameterized queries."""
    # This would use proper parameterized queries
    if not user_input.replace(" ", "").isalnum():
        raise ValueError("Invalid input characters")
    return f"SELECT * FROM users WHERE name = ?"

def buffer_overflow_risk(data: str, max_size: int = 100) -> str:
    """Function with potential buffer overflow."""
    if len(data) > max_size:
        # Potential issue: truncation without warning
        return data[:max_size]
    return data

def safe_buffer_handling(data: str, max_size: int = 100) -> str:
    """Safe buffer handling with validation."""
    if len(data) > max_size:
        raise ValueError(f"Data too large: {len(data)} > {max_size}")
    return data

def null_pointer_risk(obj: dict, key: str) -> str:
    """Function with potential null pointer access."""
    return obj[key].upper()

def safe_null_handling(obj: dict, key: str) -> str:
    """Safe null handling with validation."""
    if obj is None:
        raise ValueError("Object is None")
    if key not in obj:
        raise KeyError(f"Key '{key}' not found")
    value = obj[key]
    if value is None:
        raise ValueError(f"Value for key '{key}' is None")
    return str(value).upper()

def resource_leak_risk():
    """Function with potential resource leak."""
    file = open("/tmp/test.txt", "w")
    file.write("test data")
    # File not closed - potential leak

def safe_resource_handling():
    """Safe resource handling with context manager."""
    with open("/tmp/test.txt", "w") as file:
        file.write("test data")
    # File automatically closed
'''

    def generate_performance_test_code(self, function_count: int = 50) -> str:
        """Generate test code for performance analysis.

        Creates a large codebase with multiple functions to test
        performance characteristics of the analysis tools.

        Args:
            function_count: Number of functions to generate

        Returns:
            str: Python code suitable for performance testing
        """
        base_imports = """
from typing import List, Dict, Optional, Any
import json
import asyncio
import time

"""

        functions = []
        for func_id in range(function_count):
            function_code = f'''
def function_{func_id}(param1: str, param2: int = {func_id}) -> Dict[str, Any]:
    """Function {func_id} for performance testing."""
    result = process_function_{func_id}(param1, param2)
    return validate_result_{func_id}(result)

def process_function_{func_id}(data: str, num: int) -> str:
    """Process function {func_id}."""
    return f"{{data}}_{{num}}"

def validate_result_{func_id}(result: str) -> Dict[str, Any]:
    """Validate result for function {func_id}."""
    return {{"valid": bool(result), "length": len(result), "function_id": {func_id}}}
'''
            functions.append(function_code)

        return base_imports + "\n".join(functions)

    def create_test_scenario(self, scenario_type: TestScenarioType, **kwargs) -> TestScenario:
        """Create a test scenario of the specified type.

        Args:
            scenario_type: Type of test scenario to create
            **kwargs: Additional parameters for scenario customization

        Returns:
            TestScenario: Configured test scenario
        """
        scenario_generators = {
            TestScenarioType.CALL_GRAPH: self.generate_call_graph_test_code,
            TestScenarioType.SIGNATURE_ANALYSIS: self.generate_signature_analysis_test_code,
            TestScenarioType.SYMBOLIC_VALIDATION: self.generate_symbolic_validation_test_code,
            TestScenarioType.PERFORMANCE: lambda: self.generate_performance_test_code(
                kwargs.get("function_count", 50)
            ),
        }

        if scenario_type not in scenario_generators:
            raise ValueError(f"Unsupported scenario type: {scenario_type}")

        test_code = scenario_generators[scenario_type]()

        scenario = TestScenario(
            name=f"{scenario_type.value}_test",
            scenario_type=scenario_type,
            test_code=test_code,
            expected_results={},
            metadata=kwargs,
        )

        self.generated_scenarios.append(scenario)
        return scenario


class TestFileManager:
    """Manages test file creation, organization, and cleanup.

    This class handles the creation of temporary test files and directories,
    manages test project structures, and ensures proper cleanup of test resources.
    """

    def __init__(self):
        """Initialize the test file manager."""
        self.created_files: list[str] = []
        self.created_directories: list[str] = []

    def create_test_file(self, content: str, suffix: str = ".py") -> str:
        """Create a temporary test file with specified content.

        Args:
            content: File content to write
            suffix: File extension/suffix

        Returns:
            str: Path to created test file
        """
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)

        try:
            temp_file.write(content)
            temp_file.flush()
            file_path = temp_file.name
            self.created_files.append(file_path)
            return file_path
        finally:
            temp_file.close()

    def create_test_project(self, project_structure: dict[str, str]) -> TestProject:
        """Create a test project with specified structure.

        Args:
            project_structure: Dictionary mapping file paths to content

        Returns:
            TestProject: Created test project information
        """
        temp_dir = tempfile.mkdtemp()
        self.created_directories.append(temp_dir)

        created_files = {}

        for relative_path, content in project_structure.items():
            full_path = os.path.join(temp_dir, relative_path)

            # Create directory if it doesn't exist
            dir_path = os.path.dirname(full_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            # Write file content
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

            created_files[relative_path] = full_path

        return TestProject(root_dir=temp_dir, files=created_files, cleanup_required=True)

    def create_integration_test_project(self) -> TestProject:
        """Create a complete integration test project.

        Returns:
            TestProject: Complete test project for integration testing
        """
        project_structure = {
            "src/main.py": '''
"""Main application module."""

from typing import List, Dict, Optional, Any
import logging

def main() -> int:
    """Main entry point for the application."""
    try:
        users = load_users()
        processed = process_users(users)
        save_results(processed)
        return 0
    except Exception as e:
        logging.error(f"Application error: {e}")
        return 1

def load_users() -> List[Dict[str, str]]:
    """Load users from data source."""
    return [
        {"name": "Alice", "email": "alice@example.com"},
        {"name": "Bob", "email": "bob@example.com"}
    ]

def process_users(users: List[Dict[str, str]]) -> Dict[str, Any]:
    """Process loaded users."""
    return {
        "count": len(users),
        "users": [validate_user(user) for user in users]
    }

def validate_user(user: Dict[str, str]) -> Dict[str, str]:
    """Validate a single user."""
    if not user.get("name") or not user.get("email"):
        raise ValueError("Invalid user data")
    return user

def save_results(results: Dict[str, Any]) -> None:
    """Save processing results."""
    print(f"Processed {results['count']} users")

if __name__ == "__main__":
    exit(main())
''',
            "src/utils.py": '''
"""Utility functions."""

def format_email(email: str) -> str:
    """Format email address."""
    return email.lower().strip()

def validate_email(email: str) -> bool:
    """Validate email format."""
    return "@" in email and "." in email
''',
            "README.md": """# Integration Test Project

Complete test project for integration testing of static analysis tools.

## Features
- User processing pipeline
- Email validation utilities
- Error handling examples
""",
            "requirements.txt": """# No external dependencies for test project
""",
        }

        return self.create_test_project(project_structure)

    def cleanup_test_files(self):
        """Clean up all created test files and directories."""
        # Remove individual files
        for file_path in self.created_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except OSError as e:
                logger.warning(f"Failed to remove test file {file_path}: {e}")

        # Remove directories
        for dir_path in self.created_directories:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path, ignore_errors=True)
            except OSError as e:
                logger.warning(f"Failed to remove test directory {dir_path}: {e}")

        self.created_files.clear()
        self.created_directories.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_test_files()


class TestScenarioParser:
    """Parses and categorizes test scenarios for analysis.

    This class analyzes test scenarios to determine their characteristics,
    requirements, and expected outcomes for proper test orchestration.
    """

    def __init__(self):
        """Initialize the test scenario parser."""
        self.parsed_scenarios: dict[str, TestScenario] = {}

    def parse_test_requirements(self, scenario: TestScenario) -> dict[str, Any]:
        """Parse test requirements from a scenario.

        Args:
            scenario: Test scenario to analyze

        Returns:
            Dict containing parsed requirements and capabilities needed
        """
        requirements = {
            "tools_needed": [],
            "expected_outputs": [],
            "validation_criteria": [],
            "performance_expectations": {},
            "resource_requirements": {},
        }

        # Analyze based on scenario type
        if scenario.scenario_type == TestScenarioType.CALL_GRAPH:
            requirements["tools_needed"] = ["call_graph_analyzer"]
            requirements["expected_outputs"] = ["nodes", "edges", "orphaned_functions"]
            requirements["validation_criteria"] = [
                "has_call_graph",
                "has_nodes",
                "has_edges",
                "found_orphaned",
            ]

        elif scenario.scenario_type == TestScenarioType.SIGNATURE_ANALYSIS:
            requirements["tools_needed"] = ["function_extractor"]
            requirements["expected_outputs"] = [
                "signatures",
                "type_annotations",
                "docstrings",
            ]
            requirements["validation_criteria"] = [
                "function_count",
                "type_coverage",
                "has_docstrings",
            ]

        elif scenario.scenario_type == TestScenarioType.SYMBOLIC_VALIDATION:
            requirements["tools_needed"] = ["symbolic_validator"]
            requirements["expected_outputs"] = [
                "safety_score",
                "validation_results",
                "recommendations",
            ]
            requirements["validation_criteria"] = [
                "has_analysis",
                "safety_score",
                "blocking_issues",
            ]

        elif scenario.scenario_type == TestScenarioType.PERFORMANCE:
            requirements["performance_expectations"] = {
                "max_execution_time": 30.0,
                "memory_limit": "1GB",
            }
            requirements["resource_requirements"] = {
                "function_count": scenario.metadata.get("function_count", 50)
            }

        return requirements

    def categorize_test_scenarios(
        self, scenarios: list[TestScenario]
    ) -> dict[str, list[TestScenario]]:
        """Categorize test scenarios by type and complexity.

        Args:
            scenarios: List of test scenarios to categorize

        Returns:
            Dict mapping categories to scenario lists
        """
        categories = {
            "unit_tests": [],
            "integration_tests": [],
            "performance_tests": [],
            "security_tests": [],
        }

        for scenario in scenarios:
            if scenario.scenario_type == TestScenarioType.INTEGRATION:
                categories["integration_tests"].append(scenario)
            elif scenario.scenario_type == TestScenarioType.PERFORMANCE:
                categories["performance_tests"].append(scenario)
            elif scenario.scenario_type == TestScenarioType.SYMBOLIC_VALIDATION:
                categories["security_tests"].append(scenario)
            else:
                categories["unit_tests"].append(scenario)

        return categories

    def validate_test_scenario(self, scenario: TestScenario) -> tuple[bool, list[str]]:
        """Validate a test scenario for completeness and correctness.

        Args:
            scenario: Test scenario to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check required fields
        if not scenario.name:
            issues.append("Scenario name is required")

        if not scenario.test_code:
            issues.append("Test code is required")

        # Validate test code syntax
        try:
            compile(scenario.test_code, f"<{scenario.name}>", "exec")
        except SyntaxError as e:
            issues.append(f"Test code syntax error: {e}")

        # Check scenario-specific requirements
        if scenario.scenario_type == TestScenarioType.CALL_GRAPH:
            if "def " not in scenario.test_code:
                issues.append("Call graph test must contain function definitions")

        elif scenario.scenario_type == TestScenarioType.SIGNATURE_ANALYSIS:
            if ":" not in scenario.test_code or "->" not in scenario.test_code:
                issues.append("Signature analysis test should contain type annotations")

        return len(issues) == 0, issues


# Convenience functions for common test discovery operations
def create_test_code_samples() -> dict[str, str]:
    """Create standardized test code samples for all scenario types.

    Returns:
        Dict mapping scenario names to test code
    """
    generator = TestCodeGenerator()

    return {
        "call_graph": generator.generate_call_graph_test_code(),
        "signature_analysis": generator.generate_signature_analysis_test_code(),
        "symbolic_validation": generator.generate_symbolic_validation_test_code(),
        "performance": generator.generate_performance_test_code(),
    }


def parse_test_requirements(scenario_type: TestScenarioType) -> dict[str, Any]:
    """Parse test requirements for a specific scenario type.

    Args:
        scenario_type: Type of test scenario

    Returns:
        Dict containing parsed requirements
    """
    parser = TestScenarioParser()
    generator = TestCodeGenerator()

    scenario = generator.create_test_scenario(scenario_type)
    return parser.parse_test_requirements(scenario)
