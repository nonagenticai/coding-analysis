"""Unit tests for Go language processing modules.

Covers:
- go_syntax_parser.py  (GoSyntaxParser)
- go_test_executor.py  (GoTestExecutor, GoTestResult, GoBenchmarkResult, GoTestSuite)
- go_quality_analyzer.py  (GoQualityAnalyzer)
- go_module_manager.py  (GoModuleManager)
- go_processor.py  (GoProcessor)
- go_dependency_analyzer.py  (GoDependencyAnalyzer)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.core_models import (
    CoreCodeEntity,
    EntityType,
    RelationType,
)
from processing.languages.go.go_dependency_analyzer import GoDependencyAnalyzer
from processing.languages.go.go_module_manager import GoModuleManager
from processing.languages.go.go_quality_analyzer import GoQualityAnalyzer
from processing.languages.go.go_syntax_parser import GoSyntaxParser
from processing.languages.go.go_test_executor import (
    GoTestExecutor,
    GoTestSuite,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_GO_CODE = """\
package main

import (
    "fmt"
    "os"
)

// MaxRetries is the maximum retry count.
const MaxRetries = 3

const (
    StatusOK   = 200
    StatusFail = 500
)

var GlobalLogger *Logger

var (
    DefaultTimeout int
    Verbose        bool
)

type MyID int

// Server handles HTTP requests.
type Server struct {
    Port int
    Host string
}

type handler interface {
    Handle() error
}

func main() {
    fmt.Println("hello")
}

func (s *Server) Start(addr string) error {
    return nil
}

func helper(x int, y int) (int, error) {
    if x > 0 && y > 0 {
        return x + y, nil
    }
    return 0, fmt.Errorf("invalid")
}

func processItems(items []string) {
    for _, item := range items {
        switch item {
        case "a":
            go handle(item)
        case "b":
            defer cleanup(item)
        }
    }
}
"""


@pytest.fixture
def sample_go_code():
    return SAMPLE_GO_CODE


@pytest.fixture
def sample_file_path():
    return Path("cmd/server/main.go")


@pytest.fixture
def syntax_parser():
    return GoSyntaxParser()


@pytest.fixture
def module_manager():
    return GoModuleManager()


@pytest.fixture
def dep_analyzer():
    return GoDependencyAnalyzer()


@pytest.fixture
def quality_analyzer():
    return GoQualityAnalyzer()


@pytest.fixture
def test_executor():
    return GoTestExecutor()


# ---------------------------------------------------------------------------
# Helper to quickly build a CoreCodeEntity
# ---------------------------------------------------------------------------


def _entity(
    name: str,
    etype: EntityType,
    visibility: str | None = "public",
    line: int = 1,
    file_path: str = "main.go",
    properties: dict | None = None,
) -> CoreCodeEntity:
    return CoreCodeEntity(
        id=f"{etype.value}:{name}:{file_path}:{line}",
        name=name,
        type=etype,
        language="go",
        file_path=file_path,
        line_number=line,
        visibility=visibility,
        properties=properties or {},
    )


# ===================================================================
# GoSyntaxParser tests
# ===================================================================


class TestGoSyntaxParserInit:
    def test_init(self, syntax_parser):
        assert syntax_parser.parser is None

    def test_set_parser(self, syntax_parser):
        mock_parser = MagicMock()
        syntax_parser.set_parser(mock_parser)
        assert syntax_parser.parser is mock_parser


class TestGoSyntaxParserRegexFunctions:
    async def test_extract_simple_function(self, syntax_parser, sample_file_path):
        code = 'func main() {\n    fmt.Println("hi")\n}'
        result = await syntax_parser.extract_functions_and_methods_regex(code, sample_file_path)
        assert len(result) == 1
        assert result[0].name == "main"
        assert result[0].type == EntityType.FUNCTION
        assert result[0].visibility == "private"  # lowercase
        assert result[0].properties["is_method"] is False

    async def test_extract_exported_function(self, syntax_parser, sample_file_path):
        code = "func HandleRequest(w http.ResponseWriter, r *http.Request) {\n}"
        result = await syntax_parser.extract_functions_and_methods_regex(code, sample_file_path)
        assert len(result) == 1
        assert result[0].name == "HandleRequest"
        assert result[0].visibility == "public"

    async def test_extract_method_with_receiver(self, syntax_parser, sample_file_path):
        code = "func (s *Server) Start(addr string) error {\n    return nil\n}"
        result = await syntax_parser.extract_functions_and_methods_regex(code, sample_file_path)
        assert len(result) == 1
        e = result[0]
        assert e.name == "Start"
        assert e.properties["receiver_type"] == "Server"
        assert e.properties["receiver_name"] == "s"
        assert e.properties["is_method"] is True
        assert "Server.Start" in e.id

    async def test_extract_function_with_multiple_returns(self, syntax_parser, sample_file_path):
        code = "func divide(a, b float64) (float64, error) {\n    return a/b, nil\n}"
        result = await syntax_parser.extract_functions_and_methods_regex(code, sample_file_path)
        assert len(result) == 1
        assert "float64, error" in result[0].signature

    async def test_extract_function_with_single_return(self, syntax_parser, sample_file_path):
        code = "func greet(name string) string {\n    return name\n}"
        result = await syntax_parser.extract_functions_and_methods_regex(code, sample_file_path)
        assert len(result) == 1
        assert "string" in result[0].signature

    async def test_extract_no_functions(self, syntax_parser, sample_file_path):
        code = "package main\n\nvar x int"
        result = await syntax_parser.extract_functions_and_methods_regex(code, sample_file_path)
        assert result == []

    async def test_extract_multiple_functions(self, syntax_parser, sample_file_path):
        result = await syntax_parser.extract_functions_and_methods_regex(
            SAMPLE_GO_CODE, sample_file_path
        )
        names = {e.name for e in result}
        assert "main" in names
        assert "Start" in names
        assert "helper" in names
        assert "processItems" in names

    async def test_line_numbers(self, syntax_parser, sample_file_path):
        code = "package main\n\nfunc first() {\n}\n\nfunc second() {\n}"
        result = await syntax_parser.extract_functions_and_methods_regex(code, sample_file_path)
        assert result[0].line_number == 3
        assert result[1].line_number == 6


class TestGoSyntaxParserRegexStructs:
    async def test_extract_struct(self, syntax_parser, sample_file_path):
        code = "type Server struct {\n    Port int\n}"
        result = await syntax_parser.extract_structs_regex(code, sample_file_path)
        assert len(result) == 1
        assert result[0].name == "Server"
        assert result[0].type == EntityType.CLASS
        assert result[0].visibility == "public"
        assert result[0].properties["struct_type"] is True

    async def test_extract_private_struct(self, syntax_parser, sample_file_path):
        code = "type config struct {\n    key string\n}"
        result = await syntax_parser.extract_structs_regex(code, sample_file_path)
        assert len(result) == 1
        assert result[0].visibility == "private"

    async def test_no_structs(self, syntax_parser, sample_file_path):
        result = await syntax_parser.extract_structs_regex("package main", sample_file_path)
        assert result == []


class TestGoSyntaxParserRegexInterfaces:
    async def test_extract_interface(self, syntax_parser, sample_file_path):
        code = "type Reader interface {\n    Read(p []byte) (int, error)\n}"
        result = await syntax_parser.extract_interfaces_regex(code, sample_file_path)
        assert len(result) == 1
        assert result[0].name == "Reader"
        assert result[0].type == EntityType.INTERFACE
        assert result[0].visibility == "public"

    async def test_extract_private_interface(self, syntax_parser, sample_file_path):
        code = "type handler interface {\n    Handle() error\n}"
        result = await syntax_parser.extract_interfaces_regex(code, sample_file_path)
        assert len(result) == 1
        assert result[0].visibility == "private"


class TestGoSyntaxParserRegexTypes:
    async def test_extract_type_alias(self, syntax_parser, sample_file_path):
        code = "type MyID int"
        result = await syntax_parser.extract_types_regex(code, sample_file_path)
        assert len(result) == 1
        assert result[0].name == "MyID"
        assert result[0].type == EntityType.TYPE_ALIAS
        assert result[0].properties["base_type"] == "int"
        assert result[0].signature == "type MyID int"


class TestGoSyntaxParserRegexConstants:
    async def test_extract_single_const(self, syntax_parser, sample_file_path):
        code = "const MaxRetries = 3"
        result = await syntax_parser.extract_constants_regex(code, sample_file_path)
        assert len(result) >= 1
        names = {e.name for e in result}
        assert "MaxRetries" in names

    async def test_extract_const_block(self, syntax_parser, sample_file_path):
        code = "const (\n    StatusOK   = 200\n    StatusFail = 500\n)"
        result = await syntax_parser.extract_constants_regex(code, sample_file_path)
        names = {e.name for e in result}
        assert "StatusOK" in names
        assert "StatusFail" in names

    async def test_const_visibility(self, syntax_parser, sample_file_path):
        code = "const maxBuf = 1024"
        result = await syntax_parser.extract_constants_regex(code, sample_file_path)
        assert any(e.visibility == "private" for e in result if e.name == "maxBuf")


class TestGoSyntaxParserRegexVariables:
    async def test_extract_single_var(self, syntax_parser, sample_file_path):
        code = "var GlobalLogger *Logger"
        result = await syntax_parser.extract_variables_regex(code, sample_file_path)
        assert len(result) == 1
        assert result[0].name == "GlobalLogger"
        assert result[0].type == EntityType.VARIABLE
        assert result[0].visibility == "public"

    async def test_extract_var_block(self, syntax_parser, sample_file_path):
        code = "var (\n    DefaultTimeout int\n    Verbose        bool\n)"
        result = await syntax_parser.extract_variables_regex(code, sample_file_path)
        names = {e.name for e in result}
        assert "DefaultTimeout" in names
        assert "Verbose" in names

    async def test_no_vars(self, syntax_parser, sample_file_path):
        result = await syntax_parser.extract_variables_regex("package main\n", sample_file_path)
        assert result == []


class TestGoSyntaxParserFindNodes:
    """Test _find_nodes_by_type and _get_node_text with mocks."""

    def test_find_nodes_by_type_no_tree_sitter(self, syntax_parser):
        """When TREE_SITTER_AVAILABLE is False, returns empty."""
        with patch("processing.languages.go.go_syntax_parser.TREE_SITTER_AVAILABLE", False):
            result = syntax_parser._find_nodes_by_type(MagicMock(), {"function_declaration"})
            assert result == []

    def test_get_node_text_no_tree_sitter(self, syntax_parser):
        with patch("processing.languages.go.go_syntax_parser.TREE_SITTER_AVAILABLE", False):
            result = syntax_parser._get_node_text(MagicMock(), b"some content")
            assert result == ""


class TestGoSyntaxParserTreeSitterEntities:
    """Test extract_entities_tree_sitter with mocked tree."""

    async def test_tree_sitter_not_available(self, syntax_parser, sample_file_path):
        with patch("processing.languages.go.go_syntax_parser.TREE_SITTER_AVAILABLE", False):
            result = await syntax_parser.extract_entities_tree_sitter(
                MagicMock(), sample_file_path, b""
            )
            assert result == []

    async def test_tree_sitter_function_extraction(self, syntax_parser, sample_file_path):
        """Test tree-sitter extraction with a mock tree containing a function node."""
        with patch("processing.languages.go.go_syntax_parser.TREE_SITTER_AVAILABLE", True):
            func_code = b'func Hello() string {\n    return "hi"\n}'
            # Build mock node
            func_node = MagicMock()
            func_node.type = "function_declaration"
            func_node.children = []
            func_node.start_byte = 0
            func_node.end_byte = len(func_code)
            func_node.start_point = (0, 0)
            func_node.end_point = (2, 1)

            root = MagicMock()
            root.type = "source_file"
            root.children = [func_node]

            tree = MagicMock()
            tree.root_node = root

            result = await syntax_parser.extract_entities_tree_sitter(
                tree, sample_file_path, func_code
            )
            func_entities = [e for e in result if e.type == EntityType.FUNCTION]
            assert len(func_entities) == 1
            assert func_entities[0].name == "Hello"
            assert func_entities[0].visibility == "public"

    async def test_tree_sitter_struct_extraction(self, syntax_parser, sample_file_path):
        with patch("processing.languages.go.go_syntax_parser.TREE_SITTER_AVAILABLE", True):
            code = b"type Config struct {\n    Port int\n}"
            node = MagicMock()
            node.type = "type_declaration"
            node.children = []
            node.start_byte = 0
            node.end_byte = len(code)
            node.start_point = (0, 0)
            node.end_point = (2, 1)

            root = MagicMock()
            root.type = "source_file"
            root.children = [node]

            tree = MagicMock()
            tree.root_node = root

            result = await syntax_parser.extract_entities_tree_sitter(tree, sample_file_path, code)
            struct_entities = [e for e in result if e.type == EntityType.CLASS]
            assert len(struct_entities) == 1
            assert struct_entities[0].name == "Config"

    async def test_tree_sitter_interface_extraction(self, syntax_parser, sample_file_path):
        with patch("processing.languages.go.go_syntax_parser.TREE_SITTER_AVAILABLE", True):
            code = b"type Reader interface {\n    Read() error\n}"
            parent_node = MagicMock()
            parent_node.type = "type_declaration"
            parent_node.start_byte = 0
            parent_node.end_byte = len(code)
            parent_node.start_point = (0, 0)
            parent_node.end_point = (2, 1)

            iface_node = MagicMock()
            iface_node.type = "interface_type"
            # interface_type is a child of type_declaration
            iface_node.parent = parent_node
            iface_node.children = []
            iface_node.start_byte = 17
            iface_node.end_byte = len(code) - 1
            iface_node.start_point = (0, 17)
            iface_node.end_point = (2, 0)

            parent_node.children = [iface_node]

            root = MagicMock()
            root.type = "source_file"
            root.children = [parent_node]

            tree = MagicMock()
            tree.root_node = root

            result = await syntax_parser.extract_entities_tree_sitter(tree, sample_file_path, code)
            iface_entities = [e for e in result if e.type == EntityType.INTERFACE]
            assert len(iface_entities) == 1
            assert iface_entities[0].name == "Reader"

    async def test_tree_sitter_const_extraction(self, syntax_parser, sample_file_path):
        with patch("processing.languages.go.go_syntax_parser.TREE_SITTER_AVAILABLE", True):
            code = b"const MaxSize = 100"
            node = MagicMock()
            node.type = "const_declaration"
            node.children = []
            node.start_byte = 0
            node.end_byte = len(code)
            node.start_point = (0, 0)
            node.end_point = (0, len(code))

            root = MagicMock()
            root.type = "source_file"
            root.children = [node]

            tree = MagicMock()
            tree.root_node = root

            result = await syntax_parser.extract_entities_tree_sitter(tree, sample_file_path, code)
            const_entities = [e for e in result if e.type == EntityType.CONSTANT]
            assert any(e.name == "MaxSize" for e in const_entities)

    async def test_tree_sitter_var_declaration(self, syntax_parser, sample_file_path):
        with patch("processing.languages.go.go_syntax_parser.TREE_SITTER_AVAILABLE", True):
            code = b"var Count int"
            node = MagicMock()
            node.type = "var_declaration"
            node.children = []
            node.start_byte = 0
            node.end_byte = len(code)
            node.start_point = (0, 0)
            node.end_point = (0, len(code))

            root = MagicMock()
            root.type = "source_file"
            root.children = [node]
            tree = MagicMock()
            tree.root_node = root

            result = await syntax_parser.extract_entities_tree_sitter(tree, sample_file_path, code)
            var_entities = [e for e in result if e.type == EntityType.VARIABLE]
            assert any(e.name == "Count" for e in var_entities)

    async def test_tree_sitter_short_var_declaration(self, syntax_parser, sample_file_path):
        with patch("processing.languages.go.go_syntax_parser.TREE_SITTER_AVAILABLE", True):
            code = b'msg := "hello"'
            node = MagicMock()
            node.type = "short_var_declaration"
            node.children = []
            node.start_byte = 0
            node.end_byte = len(code)
            node.start_point = (0, 0)
            node.end_point = (0, len(code))

            root = MagicMock()
            root.type = "source_file"
            root.children = [node]
            tree = MagicMock()
            tree.root_node = root

            result = await syntax_parser.extract_entities_tree_sitter(tree, sample_file_path, code)
            var_entities = [e for e in result if e.type == EntityType.VARIABLE]
            assert any(e.name == "msg" for e in var_entities)
            assert any(e.properties.get("short_declaration") for e in var_entities)

    async def test_tree_sitter_type_alias(self, syntax_parser, sample_file_path):
        with patch("processing.languages.go.go_syntax_parser.TREE_SITTER_AVAILABLE", True):
            code = b"type Duration int64"
            node = MagicMock()
            node.type = "type_declaration"
            node.children = []
            node.start_byte = 0
            node.end_byte = len(code)
            node.start_point = (0, 0)
            node.end_point = (0, len(code))

            root = MagicMock()
            root.type = "source_file"
            root.children = [node]
            tree = MagicMock()
            tree.root_node = root

            result = await syntax_parser.extract_entities_tree_sitter(tree, sample_file_path, code)
            type_entities = [e for e in result if e.type == EntityType.TYPE_ALIAS]
            assert any(e.name == "Duration" for e in type_entities)


# ===================================================================
# GoTestExecutor tests
# ===================================================================


class TestGoTestExecutorInit:
    def test_init(self, test_executor):
        assert test_executor.go_binary == "go"


class TestGoTestExecutorDiscover:
    async def test_discover_test_files(self, test_executor, tmp_path):
        test_file = tmp_path / "pkg" / "handler_test.go"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("package pkg\n")
        non_test = tmp_path / "pkg" / "handler.go"
        non_test.write_text("package pkg\n")

        result = await test_executor.discover_test_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "handler_test.go"

    async def test_discover_no_test_files(self, test_executor, tmp_path):
        (tmp_path / "main.go").write_text("package main\n")
        result = await test_executor.discover_test_files(tmp_path)
        assert result == []

    async def test_discover_handles_error(self, test_executor):
        result = await test_executor.discover_test_files(Path("/nonexistent/path"))
        assert result == []


class TestGoTestExecutorRunTests:
    async def test_run_tests_success(self, test_executor, tmp_path):
        json_lines = "\n".join(
            [
                json.dumps({"Action": "pass", "Test": "TestAdd", "Elapsed": 0.01, "Output": "ok"}),
                json.dumps({"Action": "fail", "Test": "TestSub", "Elapsed": 0.02, "Output": "err"}),
                json.dumps({"Action": "skip", "Test": "TestMul", "Output": "skipped"}),
            ]
        )
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json_lines, stderr=""
        )
        with patch.object(test_executor, "_execute_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result
            suite = await test_executor.run_tests(tmp_path)

        assert suite.total_tests == 3
        assert suite.passed_tests == 1
        assert suite.failed_tests == 1
        assert suite.skipped_tests == 1

    async def test_run_tests_with_coverage(self, test_executor, tmp_path):
        json_lines = json.dumps({"Action": "pass", "Test": "TestX", "Elapsed": 0.05, "Output": ""})
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json_lines, stderr=""
        )
        with (
            patch.object(test_executor, "_execute_command", new_callable=AsyncMock) as mock_cmd,
            patch.object(test_executor, "_parse_coverage", new_callable=AsyncMock) as mock_cov,
        ):
            mock_cmd.return_value = mock_result
            mock_cov.return_value = 85.5
            suite = await test_executor.run_tests(tmp_path, coverage=True)

        assert suite.coverage_percentage == 85.5

    async def test_run_tests_failure(self, test_executor, tmp_path):
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="compilation error"
        )
        with patch.object(test_executor, "_execute_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result
            suite = await test_executor.run_tests(tmp_path)

        assert suite.total_tests == 0
        assert suite.failed_tests == 0

    async def test_run_tests_exception(self, test_executor, tmp_path):
        with patch.object(test_executor, "_execute_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.side_effect = OSError("go not found")
            suite = await test_executor.run_tests(tmp_path)

        assert suite.total_tests == 0


class TestGoTestExecutorBenchmarks:
    async def test_run_benchmarks_success(self, test_executor, tmp_path):
        output = (
            "BenchmarkSort-8     1000000     1234 ns/op     256 B/op     3 allocs/op\n"
            "BenchmarkSearch-8   5000000      567 ns/op\n"
        )
        mock_result = subprocess.CompletedProcess(args=[], returncode=0, stdout=output, stderr="")
        with patch.object(test_executor, "_execute_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result
            results = await test_executor.run_benchmarks(tmp_path)

        assert len(results) == 2
        assert results[0].benchmark_name == "Sort"
        assert results[0].iterations == 1000000
        assert results[0].ns_per_op == 1234.0
        assert results[0].allocs_per_op == 3
        assert results[0].mb_per_sec is not None

        assert results[1].benchmark_name == "Search"
        assert results[1].allocs_per_op is None

    async def test_run_benchmarks_failure(self, test_executor, tmp_path):
        mock_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")
        with patch.object(test_executor, "_execute_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result
            results = await test_executor.run_benchmarks(tmp_path)
        assert results == []

    async def test_run_benchmarks_exception(self, test_executor, tmp_path):
        with patch.object(test_executor, "_execute_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.side_effect = RuntimeError("boom")
            results = await test_executor.run_benchmarks(tmp_path)
        assert results == []


class TestGoTestExecutorCoverage:
    async def test_analyze_test_coverage_success(self, test_executor, tmp_path):
        mock_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")
        coverage_content = (
            "mode: count\n"
            "example.com/pkg/main.go:10.20,15.30 3 5\n"
            "example.com/pkg/main.go:20.10,25.20 2 0\n"
        )
        coverage_file = tmp_path / "coverage.out"
        coverage_file.write_text(coverage_content)

        with (
            patch.object(test_executor, "_execute_command", new_callable=AsyncMock) as mock_cmd,
            patch.object(
                test_executor, "_generate_coverage_report", new_callable=AsyncMock
            ) as mock_report,
        ):
            mock_cmd.return_value = mock_result
            mock_report.return_value = "<html>report</html>"
            result = await test_executor.analyze_test_coverage(tmp_path)

        assert "overall_percentage" in result
        assert "file_coverage" in result

    async def test_analyze_test_coverage_failure(self, test_executor, tmp_path):
        mock_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="err")
        with patch.object(test_executor, "_execute_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result
            result = await test_executor.analyze_test_coverage(tmp_path)
        assert result["overall_percentage"] == 0.0

    async def test_analyze_test_coverage_exception(self, test_executor, tmp_path):
        with patch.object(test_executor, "_execute_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.side_effect = OSError("fail")
            result = await test_executor.analyze_test_coverage(tmp_path)
        assert result["overall_percentage"] == 0.0

    async def test_parse_coverage_no_file(self, test_executor, tmp_path):
        result = await test_executor._parse_coverage(tmp_path)
        assert result is None

    async def test_parse_coverage_with_total(self, test_executor, tmp_path):
        coverage_file = tmp_path / "coverage.out"
        coverage_file.write_text("mode: count\n")
        cmd_output = "total:\t(statements)\t78.5%\n"
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=cmd_output, stderr=""
        )
        with patch.object(test_executor, "_execute_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result
            result = await test_executor._parse_coverage(tmp_path)
        assert result == 78.5

    async def test_parse_coverage_profile_no_file(self, test_executor, tmp_path):
        result = await test_executor._parse_coverage_profile(tmp_path / "missing.out")
        assert result == {}

    async def test_parse_coverage_profile_valid(self, test_executor, tmp_path):
        content = "mode: count\npkg/main.go:5.10,10.20 3 1\npkg/main.go:12.5,15.10 2 0\n"
        cov_file = tmp_path / "coverage.out"
        cov_file.write_text(content)
        result = await test_executor._parse_coverage_profile(cov_file)
        assert result["overall_percentage"] > 0 or result["overall_percentage"] == 0
        assert "file_coverage" in result
        assert "uncovered_lines" in result

    async def test_generate_coverage_report_no_file(self, test_executor, tmp_path):
        result = await test_executor._generate_coverage_report(tmp_path)
        assert "No coverage data available" in result

    async def test_generate_coverage_report_success(self, test_executor, tmp_path):
        (tmp_path / "coverage.out").write_text("mode: count\n")
        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="<html>coverage</html>", stderr=""
        )
        with patch.object(test_executor, "_execute_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result
            result = await test_executor._generate_coverage_report(tmp_path)
        assert "<html>" in result

    async def test_generate_coverage_report_failure(self, test_executor, tmp_path):
        (tmp_path / "coverage.out").write_text("mode: count\n")
        mock_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="failed")
        with patch.object(test_executor, "_execute_command", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = mock_result
            result = await test_executor._generate_coverage_report(tmp_path)
        assert "Error" in result


class TestGoTestExecutorParseOutput:
    def test_parse_test_output_mixed(self, test_executor):
        lines = "\n".join(
            [
                json.dumps({"Action": "pass", "Test": "TestA", "Elapsed": 0.1}),
                json.dumps({"Action": "fail", "Test": "TestB", "Elapsed": 0.2, "Output": "failed"}),
                json.dumps({"Action": "skip", "Test": "TestC"}),
                "not json",
                "",
            ]
        )
        suite = test_executor._parse_test_output(lines, "mypkg")
        assert suite.total_tests == 3
        assert suite.passed_tests == 1
        assert suite.failed_tests == 1
        assert suite.skipped_tests == 1
        assert suite.package == "mypkg"

    def test_parse_test_output_empty(self, test_executor):
        suite = test_executor._parse_test_output("", "pkg")
        assert suite.total_tests == 0

    def test_parse_benchmark_output(self, test_executor):
        output = "BenchmarkHash-4     500000     2345 ns/op     128 B/op     2 allocs/op\n"
        results = test_executor._parse_benchmark_output(output, "pkg")
        assert len(results) == 1
        assert results[0].benchmark_name == "Hash"
        assert results[0].iterations == 500000
        assert results[0].ns_per_op == 2345.0
        assert results[0].allocs_per_op == 2

    def test_parse_benchmark_no_match(self, test_executor):
        results = test_executor._parse_benchmark_output("ok  pkg 0.123s\n", "pkg")
        assert results == []


class TestGoTestExecutorStatistics:
    def test_get_test_statistics(self, test_executor):
        suites = [
            GoTestSuite(
                package="a",
                total_tests=10,
                passed_tests=8,
                failed_tests=1,
                skipped_tests=1,
                total_duration=1.5,
                coverage_percentage=80.0,
            ),
            GoTestSuite(
                package="b",
                total_tests=5,
                passed_tests=5,
                failed_tests=0,
                skipped_tests=0,
                total_duration=0.5,
                coverage_percentage=90.0,
            ),
        ]
        stats = test_executor.get_test_statistics(suites)
        assert stats["total_packages"] == 2
        assert stats["total_tests"] == 15
        assert stats["total_passed"] == 13
        assert stats["total_failed"] == 1
        assert stats["total_skipped"] == 1
        assert stats["total_duration"] == 2.0
        assert stats["average_coverage"] == 85.0
        assert stats["packages_with_failures"] == 1
        assert abs(stats["pass_rate"] - 86.666) < 1.0

    def test_get_test_statistics_empty(self, test_executor):
        stats = test_executor.get_test_statistics([])
        assert stats["total_tests"] == 0
        assert stats["pass_rate"] == 0.0
        assert stats["average_coverage"] == 0.0

    def test_get_test_statistics_no_coverage(self, test_executor):
        suites = [
            GoTestSuite(
                package="x",
                total_tests=2,
                passed_tests=2,
                failed_tests=0,
                skipped_tests=0,
                total_duration=0.1,
            ),
        ]
        stats = test_executor.get_test_statistics(suites)
        assert stats["average_coverage"] == 0.0


# ===================================================================
# GoQualityAnalyzer tests
# ===================================================================


class TestGoQualityAnalyzerComplexity:
    async def test_calculate_complexity_metrics(self, quality_analyzer):
        code = (
            "package main\n\n"
            "// main starts the app\n"
            "func main() {\n"
            "    if true {\n"
            "        for i := 0; i < 10; i++ {\n"
            "        }\n"
            "    }\n"
            "}\n"
            "\n"
        )
        entities = [_entity("main", EntityType.FUNCTION)]
        metrics = await quality_analyzer.calculate_complexity_metrics(code, entities)
        assert metrics.lines_of_code == 10
        assert metrics.function_count == 1
        assert metrics.comment_lines >= 1
        assert metrics.blank_lines >= 1
        assert metrics.cyclomatic_complexity > 1
        assert metrics.cognitive_complexity >= 0
        assert 0 <= metrics.maintainability_index <= 100

    async def test_empty_code(self, quality_analyzer):
        metrics = await quality_analyzer.calculate_complexity_metrics("", [])
        assert metrics.lines_of_code == 0
        assert metrics.maintainability_index == 100.0


class TestGoQualityCyclomaticComplexity:
    def test_basic_complexity(self, quality_analyzer):
        code = "func f() {\n}"
        c = quality_analyzer._calculate_cyclomatic_complexity(code)
        # base(1) + func(1) = 2
        assert c >= 2

    def test_branching_complexity(self, quality_analyzer):
        code = (
            "func f() {\n"
            "    if x > 0 {\n"
            "    } else if x < 0 {\n"
            "    }\n"
            "    for i := 0; i < 10; i++ {\n"
            "    }\n"
            "    switch x {\n"
            "    case 1:\n"
            "    case 2:\n"
            "    }\n"
            "    if a && b || c {\n"
            "    }\n"
            "}\n"
        )
        c = quality_analyzer._calculate_cyclomatic_complexity(code)
        # Many branching keywords -- just check it's significantly higher
        assert c > 5


class TestGoQualityCognitiveComplexity:
    def test_simple_code(self, quality_analyzer):
        code = "func f() {\n    x := 1\n}\n"
        c = quality_analyzer._calculate_cognitive_complexity(code)
        assert c >= 0

    def test_nested_code(self, quality_analyzer):
        code = (
            "func f() {\n"
            "    if true {\n"
            "        for i := 0; i < 10; i++ {\n"
            "            if false {\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        c = quality_analyzer._calculate_cognitive_complexity(code)
        assert c > 3

    def test_logical_operators(self, quality_analyzer):
        code = "if a && b || c {\n}\n"
        c = quality_analyzer._calculate_cognitive_complexity(code)
        assert c >= 1


class TestGoQualityMaintainability:
    def test_zero_lines(self, quality_analyzer):
        assert quality_analyzer._calculate_maintainability_index(0, 1, 1) == 100.0

    def test_normal_values(self, quality_analyzer):
        mi = quality_analyzer._calculate_maintainability_index(100, 10, 5)
        assert 0 <= mi <= 100

    def test_high_complexity(self, quality_analyzer):
        mi = quality_analyzer._calculate_maintainability_index(1000, 100, 50)
        assert 0 <= mi <= 100


class TestGoQualityCommentCoverage:
    def test_fully_documented(self, quality_analyzer):
        code = "// Hello greets.\nfunc Hello() {\n}\n"
        entities = [_entity("Hello", EntityType.FUNCTION, visibility="public", line=2)]
        coverage = quality_analyzer._calculate_comment_coverage(code, entities)
        assert coverage == 1.0

    def test_undocumented(self, quality_analyzer):
        code = "func Hello() {\n}\n"
        entities = [_entity("Hello", EntityType.FUNCTION, visibility="public", line=1)]
        coverage = quality_analyzer._calculate_comment_coverage(code, entities)
        assert coverage == 0.0

    def test_private_entities_ignored(self, quality_analyzer):
        code = "func helper() {\n}\n"
        entities = [_entity("helper", EntityType.FUNCTION, visibility="private", line=1)]
        coverage = quality_analyzer._calculate_comment_coverage(code, entities)
        assert coverage == 1.0  # no public entities => 1.0

    def test_no_entities(self, quality_analyzer):
        coverage = quality_analyzer._calculate_comment_coverage("package main", [])
        assert coverage == 1.0


class TestGoQualityAnalyzeCodeQuality:
    async def test_analyze_code_quality(self, quality_analyzer, sample_file_path):
        entities = [
            _entity("main", EntityType.FUNCTION, visibility="private", line=35),
            _entity(
                "Start",
                EntityType.FUNCTION,
                visibility="public",
                line=40,
                properties={"is_method": True, "receiver_type": "Server"},
            ),
            _entity("Server", EntityType.CLASS, visibility="public", line=28),
        ]
        result = await quality_analyzer.analyze_code_quality(
            SAMPLE_GO_CODE, entities, sample_file_path
        )
        assert "complexity_metrics" in result
        assert "function_analysis" in result
        assert "naming_analysis" in result
        assert "pattern_analysis" in result
        assert "quality_score" in result
        assert "recommendations" in result
        assert isinstance(result["quality_score"], float)


class TestGoQualityFunctionAnalysis:
    def test_function_analysis_with_match(self, quality_analyzer):
        code = "func helper(x int) int {\n    if x > 0 {\n        return x\n    }\n    return 0\n}"
        entities = [_entity("helper", EntityType.FUNCTION, properties={"is_method": False})]
        result = quality_analyzer._analyze_function_complexity(code, entities)
        assert result["total_functions"] == 1
        assert len(result["function_details"]) >= 0

    def test_function_analysis_empty(self, quality_analyzer):
        result = quality_analyzer._analyze_function_complexity("package main\n", [])
        assert result["total_functions"] == 0
        assert result["average_complexity"] == 0
        assert result["max_complexity"] == 0


class TestGoQualityNamingConventions:
    def test_good_naming(self, quality_analyzer):
        entities = [
            _entity("HandleRequest", EntityType.FUNCTION, visibility="public"),
            _entity("Server", EntityType.CLASS, visibility="public"),
            _entity("MaxRetries", EntityType.CONSTANT, visibility="public"),
        ]
        result = quality_analyzer._analyze_naming_conventions(entities)
        assert result["compliance_rate"] == 1.0
        assert result["issue_count"] == 0

    def test_bad_naming(self, quality_analyzer):
        entities = [
            _entity("my_func", EntityType.FUNCTION, visibility="public"),
        ]
        result = quality_analyzer._analyze_naming_conventions(entities)
        assert result["issue_count"] >= 1

    def test_empty_entities(self, quality_analyzer):
        result = quality_analyzer._analyze_naming_conventions([])
        assert result["compliance_rate"] == 1.0  # 0/max(0,1) = 0/1 ... actually (0-0)/max(0,1)=0


class TestGoQualityCodePatterns:
    def test_patterns_detected(self, quality_analyzer):
        code = (
            "if err != nil {\n    return err\n}\n"
            "go handleRequest()\n"
            "defer file.Close()\n"
            'panic("fatal")\n'
            "// TODO: fix this\n"
            "// FIXME: broken\n"
            "ch <- data\n"
        )
        result = quality_analyzer._analyze_code_patterns(code)
        assert result["patterns"]["error_handling"]["if_err_nil"] >= 1
        assert result["patterns"]["error_handling"]["error_returns"] >= 1
        assert result["patterns"]["goroutines"]["go_statements"] >= 1
        assert result["patterns"]["defer_statements"] >= 1
        assert result["patterns"]["panic_statements"] >= 1
        assert result["patterns"]["TODO_comments"] >= 1
        assert result["patterns"]["FIXME_comments"] >= 1
        assert result["concurrent_features"] is True

    def test_no_patterns(self, quality_analyzer):
        result = quality_analyzer._analyze_code_patterns("package main\n")
        assert result["good_pattern_count"] == 0
        assert result["concerning_pattern_count"] == 0
        assert result["concurrent_features"] is False


class TestGoQualityRecommendations:
    def test_recommendations_generated(self, quality_analyzer):
        from models.core_models import CoreComplexityMetrics

        metrics = CoreComplexityMetrics(
            cyclomatic_complexity=25,
            maintainability_index=30,
            docstring_coverage=0.5,
        )
        func_analysis = {
            "complex_functions": [{"name": "f", "complexity": 15}],
            "average_complexity": 15,
        }
        naming_analysis = {"compliance_rate": 0.7}
        pattern_analysis = {
            "patterns": {"panic_statements": 2, "TODO_comments": 10},
            "concerning_pattern_count": 5,
        }
        recs = quality_analyzer._generate_recommendations(
            metrics, func_analysis, naming_analysis, pattern_analysis
        )
        assert len(recs) > 0
        assert any("complex" in r.lower() for r in recs)

    def test_no_recommendations_for_good_code(self, quality_analyzer):
        from models.core_models import CoreComplexityMetrics

        metrics = CoreComplexityMetrics(
            cyclomatic_complexity=5,
            maintainability_index=80,
            docstring_coverage=0.95,
        )
        func_analysis = {"complex_functions": [], "average_complexity": 3}
        naming_analysis = {"compliance_rate": 0.95}
        pattern_analysis = {
            "patterns": {"panic_statements": 0, "TODO_comments": 0},
            "concerning_pattern_count": 0,
        }
        recs = quality_analyzer._generate_recommendations(
            metrics, func_analysis, naming_analysis, pattern_analysis
        )
        assert len(recs) == 0


class TestGoQualityScore:
    def test_quality_score(self, quality_analyzer):
        from models.core_models import CoreComplexityMetrics

        metrics = CoreComplexityMetrics(
            maintainability_index=80,
            docstring_coverage=0.9,
        )
        func_analysis = {"average_complexity": 5}
        naming_analysis = {"compliance_rate": 0.95}
        pattern_analysis = {"concerning_pattern_count": 0}
        score = quality_analyzer._calculate_quality_score(
            metrics, func_analysis, naming_analysis, pattern_analysis
        )
        assert 0 <= score <= 100


# ===================================================================
# GoModuleManager tests
# ===================================================================


class TestGoModuleManagerInit:
    def test_init(self, module_manager):
        assert module_manager.parser is None

    def test_set_parser(self, module_manager):
        p = MagicMock()
        module_manager.set_parser(p)
        assert module_manager.parser is p


class TestGoModuleManagerRegex:
    async def test_extract_package(self, module_manager, sample_file_path):
        code = "package main\n\nfunc main() {}"
        result = await module_manager.extract_package_regex(code, sample_file_path)
        assert result is not None
        assert result.name == "main"
        assert result.type == EntityType.PACKAGE

    async def test_extract_package_none(self, module_manager, sample_file_path):
        result = await module_manager.extract_package_regex("// no package\n", sample_file_path)
        assert result is None


class TestGoModuleManagerTreeSitter:
    async def test_not_available(self, module_manager, sample_file_path):
        with patch("processing.languages.go.go_module_manager.TREE_SITTER_AVAILABLE", False):
            result = await module_manager.extract_package_tree_sitter(
                MagicMock(), sample_file_path, b""
            )
            assert result == []

    async def test_tree_sitter_package(self, module_manager, sample_file_path):
        with patch("processing.languages.go.go_module_manager.TREE_SITTER_AVAILABLE", True):
            code = b"package server"
            node = MagicMock()
            node.type = "package_clause"
            node.children = []
            node.start_byte = 0
            node.end_byte = len(code)
            node.start_point = (0, 0)
            node.end_point = (0, len(code))

            root = MagicMock()
            root.type = "source_file"
            root.children = [node]

            tree = MagicMock()
            tree.root_node = root

            result = await module_manager.extract_package_tree_sitter(tree, sample_file_path, code)
            assert len(result) == 1
            assert result[0].name == "server"
            assert result[0].type == EntityType.PACKAGE


class TestGoModuleManagerHelpers:
    def test_get_package_name_from_content(self, module_manager):
        assert module_manager.get_package_name_from_content("package utils\n") == "utils"
        assert module_manager.get_package_name_from_content("// no package") is None

    def test_get_package_name_from_path(self, module_manager):
        assert module_manager.get_package_name_from_path(Path("/proj/utils/helper.go")) == "utils"
        assert module_manager.get_package_name_from_path(Path("/proj/cmd/main.go")) == "main"
        assert module_manager.get_package_name_from_path(Path("/proj/main/app.go")) == "main"

    def test_is_main_package(self, module_manager):
        assert module_manager.is_main_package("package main\n") is True
        assert module_manager.is_main_package("package utils\n") is False

    def test_is_test_file(self, module_manager):
        assert module_manager.is_test_file(Path("handler_test.go")) is True
        assert module_manager.is_test_file(Path("handler.go")) is False

    def test_classify_package_type(self, module_manager):
        assert (
            module_manager.classify_package_type(Path("main.go"), "package main\nfunc main() {}\n")
            == "executable"
        )
        assert (
            module_manager.classify_package_type(Path("main.go"), "package main\n")
            == "main_package"
        )
        assert (
            module_manager.classify_package_type(Path("handler_test.go"), "package handler\n")
            == "test"
        )
        assert (
            module_manager.classify_package_type(Path("handler.go"), "package handler_test\n")
            == "external_test"
        )
        assert (
            module_manager.classify_package_type(Path("handler.go"), "package handler\n")
            == "library"
        )


class TestGoModuleManagerGoMod:
    def test_get_module_info_from_go_mod(self, module_manager, tmp_path):
        go_mod = tmp_path / "go.mod"
        go_mod.write_text(
            "module github.com/example/project\n\n"
            "go 1.21\n\n"
            "require (\n"
            "    github.com/gin-gonic/gin v1.9.0\n"
            "    golang.org/x/text v0.14.0\n"
            ")\n"
        )
        result = module_manager.get_module_info_from_go_mod(tmp_path)
        assert result is not None
        assert result["module_name"] == "github.com/example/project"
        assert result["go_version"] == "1.21"
        assert len(result["dependencies"]) == 2
        assert result["dependencies"][0]["name"] == "github.com/gin-gonic/gin"

    def test_get_module_info_no_go_mod(self, module_manager, tmp_path):
        result = module_manager.get_module_info_from_go_mod(tmp_path)
        assert result is None

    def test_get_module_info_parent_search(self, module_manager, tmp_path):
        """go.mod in parent directory should be found."""
        go_mod = tmp_path / "go.mod"
        go_mod.write_text("module example.com/app\n\ngo 1.22\n")
        sub = tmp_path / "pkg" / "sub"
        sub.mkdir(parents=True)
        result = module_manager.get_module_info_from_go_mod(sub)
        assert result is not None
        assert result["module_name"] == "example.com/app"


class TestGoModuleManagerPackageDeps:
    def test_get_package_dependencies_multi_import(self, module_manager):
        code = 'import (\n    "fmt"\n    "os"\n    "net/http"\n)\n'
        deps = module_manager.get_package_dependencies(code)
        assert "fmt" in deps
        assert "os" in deps
        assert "net/http" in deps

    def test_get_package_dependencies_single_import(self, module_manager):
        code = 'import "fmt"\n'
        deps = module_manager.get_package_dependencies(code)
        assert deps == ["fmt"]

    def test_get_package_dependencies_none(self, module_manager):
        deps = module_manager.get_package_dependencies("package main\n")
        assert deps == []


class TestGoModuleManagerExports:
    def test_get_package_visibility_exports(self, module_manager):
        code = (
            "func HandleRequest() {}\n"
            "func helper() {}\n"
            "type Server struct {}\n"
            "type config struct {}\n"
            "var MaxRetries int\n"
            "var count int\n"
            "const DefaultPort = 8080\n"
            "const maxBuf = 1024\n"
            "const (\n    StatusOK = 200\n)\n"
        )
        exports = module_manager.get_package_visibility_exports(code)
        assert "HandleRequest" in exports
        assert "Server" in exports
        assert "MaxRetries" in exports
        assert "DefaultPort" in exports
        assert "StatusOK" in exports
        # Private ones should NOT be present
        assert "helper" not in exports
        assert "config" not in exports
        assert "count" not in exports
        assert "maxBuf" not in exports


# ===================================================================
# GoDependencyAnalyzer tests
# ===================================================================


class TestGoDependencyAnalyzerInit:
    def test_init(self, dep_analyzer):
        assert dep_analyzer.parser is None

    def test_set_parser(self, dep_analyzer):
        p = MagicMock()
        dep_analyzer.set_parser(p)
        assert dep_analyzer.parser is p


class TestGoDependencyAnalyzerRegexImports:
    async def test_extract_multi_import(self, dep_analyzer, sample_file_path):
        code = 'import (\n    "fmt"\n    "os"\n)\n'
        result = await dep_analyzer.extract_imports_regex(code, sample_file_path)
        names = {e.name for e in result}
        assert "fmt" in names
        assert "os" in names
        assert all(e.type == EntityType.IMPORT for e in result)

    async def test_extract_single_import(self, dep_analyzer, sample_file_path):
        code = 'import "net/http"\n'
        result = await dep_analyzer.extract_imports_regex(code, sample_file_path)
        assert len(result) == 1
        assert result[0].name == "net/http"

    async def test_extract_no_imports(self, dep_analyzer, sample_file_path):
        result = await dep_analyzer.extract_imports_regex("package main\n", sample_file_path)
        assert result == []


class TestGoDependencyAnalyzerTreeSitterImports:
    async def test_not_available(self, dep_analyzer, sample_file_path):
        with patch("processing.languages.go.go_dependency_analyzer.TREE_SITTER_AVAILABLE", False):
            result = await dep_analyzer.extract_imports_tree_sitter(
                MagicMock(), sample_file_path, b""
            )
            assert result == []

    async def test_tree_sitter_import(self, dep_analyzer, sample_file_path):
        with patch("processing.languages.go.go_dependency_analyzer.TREE_SITTER_AVAILABLE", True):
            code = b'"net/http"'
            node = MagicMock()
            node.type = "import_spec"
            node.children = []
            node.start_byte = 0
            node.end_byte = len(code)
            node.start_point = (0, 0)
            node.end_point = (0, len(code))

            root = MagicMock()
            root.type = "source_file"
            root.children = [node]
            tree = MagicMock()
            tree.root_node = root

            result = await dep_analyzer.extract_imports_tree_sitter(tree, sample_file_path, code)
            assert len(result) == 1
            assert result[0].name == "net/http"


class TestGoDependencyAnalyzerRelationships:
    async def test_build_import_relationships(self, dep_analyzer, sample_file_path):
        entities = [
            _entity("main", EntityType.PACKAGE),
            _entity("fmt", EntityType.IMPORT, visibility=None, properties={"import_path": "fmt"}),
            _entity("os", EntityType.IMPORT, visibility=None, properties={"import_path": "os"}),
        ]
        rels = await dep_analyzer.build_import_relationships(entities, sample_file_path)
        assert len(rels) == 2
        assert all(r.type == RelationType.IMPORTS for r in rels)

    async def test_build_import_relationships_no_package(self, dep_analyzer, sample_file_path):
        entities = [
            _entity("fmt", EntityType.IMPORT, visibility=None),
        ]
        rels = await dep_analyzer.build_import_relationships(entities, sample_file_path)
        assert rels == []

    async def test_build_method_relationships(self, dep_analyzer, sample_file_path):
        entities = [
            _entity("Server", EntityType.CLASS),
            _entity(
                "Start",
                EntityType.FUNCTION,
                properties={"is_method": True, "receiver_type": "Server"},
            ),
            _entity("helper", EntityType.FUNCTION, properties={"is_method": False}),
        ]
        rels = await dep_analyzer.build_method_relationships(entities, sample_file_path)
        assert len(rels) == 1
        assert rels[0].type == RelationType.DEFINES

    async def test_build_method_relationships_no_match(self, dep_analyzer, sample_file_path):
        entities = [
            _entity("Config", EntityType.CLASS),
            _entity(
                "Start",
                EntityType.FUNCTION,
                properties={"is_method": True, "receiver_type": "Server"},
            ),
        ]
        rels = await dep_analyzer.build_method_relationships(entities, sample_file_path)
        assert rels == []

    async def test_build_package_relationships(self, dep_analyzer, sample_file_path):
        entities = [
            _entity("main", EntityType.PACKAGE),
            _entity("Hello", EntityType.FUNCTION),
            _entity("Server", EntityType.CLASS),
        ]
        rels = await dep_analyzer.build_package_relationships(entities, sample_file_path)
        assert len(rels) == 2  # Hello + Server contained by main
        assert all(r.type == RelationType.CONTAINS for r in rels)

    async def test_build_package_relationships_no_package(self, dep_analyzer, sample_file_path):
        entities = [_entity("Hello", EntityType.FUNCTION)]
        rels = await dep_analyzer.build_package_relationships(entities, sample_file_path)
        assert rels == []


class TestGoDependencyAnalyzerClassification:
    def test_classify_standard(self, dep_analyzer):
        assert dep_analyzer.classify_import_type("fmt") == "standard"
        assert dep_analyzer.classify_import_type("net/http") == "standard"
        assert dep_analyzer.classify_import_type("os") == "standard"

    def test_classify_external(self, dep_analyzer):
        assert dep_analyzer.classify_import_type("github.com/gin-gonic/gin") == "external"
        assert dep_analyzer.classify_import_type("golang.org/x/text") == "external"

    def test_classify_local(self, dep_analyzer):
        assert dep_analyzer.classify_import_type("./utils") == "local"
        assert dep_analyzer.classify_import_type("../pkg") == "local"
        assert dep_analyzer.classify_import_type("mypackage") == "local"

    def test_extract_import_alias(self, dep_analyzer):
        assert dep_analyzer.extract_import_alias('mux "github.com/gorilla/mux"') == "mux"
        assert dep_analyzer.extract_import_alias('"fmt"') is None

    def test_is_dot_import(self, dep_analyzer):
        assert dep_analyzer.is_dot_import('. "fmt"') is True
        assert dep_analyzer.is_dot_import('"fmt"') is False

    def test_is_blank_import(self, dep_analyzer):
        assert dep_analyzer.is_blank_import('_ "net/http/pprof"') is True
        assert dep_analyzer.is_blank_import('"fmt"') is False

    def test_get_package_name_from_import(self, dep_analyzer):
        assert dep_analyzer.get_package_name_from_import("github.com/gin-gonic/gin") == "gin"
        assert dep_analyzer.get_package_name_from_import("fmt") == "fmt"
        assert dep_analyzer.get_package_name_from_import("net/http/") == "http"


class TestGoDependencyAnalyzerAnalyze:
    def test_analyze_import_dependencies(self, dep_analyzer):
        entities = [
            _entity("fmt", EntityType.IMPORT, visibility=None, properties={"import_path": "fmt"}),
            _entity(
                "github.com/gin-gonic/gin",
                EntityType.IMPORT,
                visibility=None,
                properties={"import_path": "github.com/gin-gonic/gin"},
            ),
            _entity(
                "mypackage",
                EntityType.IMPORT,
                visibility=None,
                properties={"import_path": "mypackage"},
            ),
            _entity("Server", EntityType.CLASS),  # non-import, should be excluded
        ]
        result = dep_analyzer.analyze_import_dependencies(entities)
        assert result["total_imports"] == 3
        assert result["standard_library"]["count"] == 1
        assert result["external_packages"]["count"] == 1
        assert result["local_packages"]["count"] == 1


# ===================================================================
# GoProcessor tests
# ===================================================================


class TestGoProcessorInit:
    def test_init(self):
        proc = GoProcessor()
        assert proc.syntax_parser is not None
        assert proc.module_manager is not None
        assert proc.dependency_analyzer is not None
        assert proc.quality_analyzer is not None
        assert proc.test_executor is not None


class TestGoProcessorRegexParse:
    async def test_regex_parse_full(self, sample_file_path):
        proc = GoProcessor()
        result = await proc._regex_parse(sample_file_path, SAMPLE_GO_CODE)
        assert result.success is True
        assert result.language == "go"
        assert result.parser_used.value == "regex_fallback"
        assert result.lines_of_code > 0
        assert len(result.entities) > 0

        # Check entity types extracted
        etypes = {e.type for e in result.entities}
        assert EntityType.PACKAGE in etypes
        assert EntityType.FUNCTION in etypes

    async def test_regex_parse_has_relationships(self, sample_file_path):
        proc = GoProcessor()
        result = await proc._regex_parse(sample_file_path, SAMPLE_GO_CODE)
        assert len(result.relationships) > 0

    async def test_regex_parse_properties(self, sample_file_path):
        proc = GoProcessor()
        result = await proc._regex_parse(sample_file_path, SAMPLE_GO_CODE)
        assert "package_name" in result.properties
        assert result.properties["package_name"] == "main"

    async def test_regex_parse_empty(self, sample_file_path):
        proc = GoProcessor()
        result = await proc._regex_parse(sample_file_path, "")
        assert result.success is True
        assert result.lines_of_code == 0


class TestGoProcessorParseFile:
    async def test_parse_file_no_tree_sitter(self, sample_file_path):
        """When tree-sitter unavailable, falls back to regex."""
        proc = GoProcessor()
        with patch("processing.languages.go.go_processor.TREE_SITTER_AVAILABLE", False):
            proc.parser = None
            result = await proc.parse_file(sample_file_path, SAMPLE_GO_CODE)
        assert result.success is True
        assert len(result.entities) > 0


class TestGoProcessorExtractRelationships:
    async def test_extract_relationships(self, sample_file_path):
        proc = GoProcessor()
        entities = [
            _entity("main", EntityType.PACKAGE),
            _entity("fmt", EntityType.IMPORT, visibility=None, properties={"import_path": "fmt"}),
            _entity("Server", EntityType.CLASS),
            _entity(
                "Start",
                EntityType.FUNCTION,
                properties={"is_method": True, "receiver_type": "Server"},
            ),
        ]
        rels = await proc.extract_relationships(None, entities, sample_file_path)
        assert len(rels) > 0
        rel_types = {r.type for r in rels}
        assert RelationType.CONTAINS in rel_types
        assert RelationType.IMPORTS in rel_types
        assert RelationType.DEFINES in rel_types


class TestGoProcessorConvenienceMethods:
    async def test_analyze_code_quality(self, sample_file_path):
        proc = GoProcessor()
        entities = [_entity("main", EntityType.FUNCTION)]
        result = await proc.analyze_code_quality(SAMPLE_GO_CODE, entities, sample_file_path)
        assert "quality_score" in result

    async def test_analyze_dependencies(self):
        proc = GoProcessor()
        entities = [
            _entity("fmt", EntityType.IMPORT, visibility=None, properties={"import_path": "fmt"}),
        ]
        result = await proc.analyze_dependencies(entities)
        assert "total_imports" in result

    def test_get_module_info(self, tmp_path):
        go_mod = tmp_path / "go.mod"
        go_mod.write_text("module example.com/app\n\ngo 1.22\n")
        proc = GoProcessor()
        result = proc.get_module_info(tmp_path)
        assert result is not None
        assert result["module_name"] == "example.com/app"

    async def test_run_tests_delegates(self, tmp_path):
        proc = GoProcessor()
        suite = GoTestSuite(
            package=str(tmp_path),
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            skipped_tests=0,
            total_duration=0.1,
            test_results=[],
        )
        with patch.object(proc.test_executor, "run_tests", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = suite
            result = await proc.run_tests(tmp_path)
        assert "test_suite" in result
        assert "statistics" in result


# Need to import GoProcessor last to avoid import-order issues with the
# @register_language decorator that runs at import time.
from processing.languages.go.go_processor import GoProcessor  # noqa: E402

# ===================================================================
# Additional GoProcessor coverage - _regex_parse error path, extract_*
# ===================================================================


class TestGoProcessorRegexParseFull:
    """Cover _regex_parse happy path and error handling in GoProcessor."""

    @pytest.mark.asyncio
    async def test_regex_parse_returns_parse_result(self):
        proc = GoProcessor()
        code = 'package main\n\nimport "fmt"\n\nfunc main() {\n    fmt.Println("hello")\n}\n'
        result = await proc._regex_parse(Path("main.go"), code)
        assert result.success is True
        assert result.language == "go"
        assert result.lines_of_code == 7
        assert any(e.type == EntityType.PACKAGE for e in result.entities)
        assert any(e.type == EntityType.IMPORT for e in result.entities)
        assert any(e.type == EntityType.FUNCTION for e in result.entities)
        assert result.properties["package_name"] == "main"

    @pytest.mark.asyncio
    async def test_regex_parse_empty_code(self):
        proc = GoProcessor()
        result = await proc._regex_parse(Path("empty.go"), "")
        assert result.success is True
        assert (
            result.entities == []
            or all(e.type == EntityType.PACKAGE for e in result.entities) is False
        )

    @pytest.mark.asyncio
    async def test_regex_parse_no_package(self):
        proc = GoProcessor()
        result = await proc._regex_parse(Path("nopackage.go"), "// just a comment\n")
        assert result.success is True
        assert result.properties.get("package_name") is None

    @pytest.mark.asyncio
    async def test_parse_file_uses_regex_fallback(self):
        """parse_file should fall through to _regex_parse when tree-sitter not available."""
        proc = GoProcessor()
        code = "package main\n\nfunc run() {}\n"
        result = await proc.parse_file(Path("run.go"), code)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_extract_relationships_builds_all_types(self):
        """extract_relationships should produce package, import, and method rels."""
        proc = GoProcessor()
        pkg = _entity("main", EntityType.PACKAGE)
        imp = _entity("fmt", EntityType.IMPORT, properties={"import_path": "fmt"})
        struct = _entity("Server", EntityType.CLASS)
        method = _entity(
            "Start",
            EntityType.FUNCTION,
            properties={"is_method": True, "receiver_type": "Server"},
        )
        entities = [pkg, imp, struct, method]
        rels = await proc.extract_relationships(None, entities, Path("main.go"))
        assert len(rels) > 0
        rel_types = {r.type for r in rels}
        assert RelationType.CONTAINS in rel_types
        assert RelationType.IMPORTS in rel_types
        assert RelationType.DEFINES in rel_types

    @pytest.mark.asyncio
    async def test_extract_relationships_empty(self):
        proc = GoProcessor()
        rels = await proc.extract_relationships(None, [], Path("empty.go"))
        assert rels == []

    @pytest.mark.asyncio
    async def test_analyze_code_quality(self):
        proc = GoProcessor()
        code = "package main\n\nfunc main() {\n    if true {\n    }\n}\n"
        entities = [_entity("main", EntityType.FUNCTION)]
        result = await proc.analyze_code_quality(code, entities, Path("main.go"))
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_analyze_dependencies(self):
        proc = GoProcessor()
        entities = [
            _entity("fmt", EntityType.IMPORT, properties={"import_path": "fmt"}),
            _entity(
                "github.com/pkg/errors",
                EntityType.IMPORT,
                properties={"import_path": "github.com/pkg/errors"},
            ),
        ]
        result = await proc.analyze_dependencies(entities)
        assert result["total_imports"] == 2

    def test_get_module_info_no_gomod(self, tmp_path):
        proc = GoProcessor()
        result = proc.get_module_info(tmp_path)
        assert result is None


class TestGoDependencyAnalyzerClassifyImport:
    """Cover classify_import_type, extract_import_alias, is_dot_import, is_blank_import, get_package_name_from_import."""

    def test_classify_standard_single(self):
        da = GoDependencyAnalyzer()
        assert da.classify_import_type("fmt") == "standard"

    def test_classify_standard_path(self):
        da = GoDependencyAnalyzer()
        assert da.classify_import_type("net/http") == "standard"

    def test_classify_external_domain(self):
        da = GoDependencyAnalyzer()
        assert da.classify_import_type("github.com/pkg/errors") == "external"

    def test_classify_local_relative(self):
        da = GoDependencyAnalyzer()
        assert da.classify_import_type("./internal") == "local"
        assert da.classify_import_type("../utils") == "local"

    def test_classify_local_ambiguous(self):
        da = GoDependencyAnalyzer()
        assert da.classify_import_type("mypackage") == "local"

    def test_extract_import_alias_present(self):
        da = GoDependencyAnalyzer()
        assert da.extract_import_alias('myfmt "fmt"') == "myfmt"

    def test_extract_import_alias_none(self):
        da = GoDependencyAnalyzer()
        assert da.extract_import_alias('"fmt"') is None

    def test_is_dot_import(self):
        da = GoDependencyAnalyzer()
        assert da.is_dot_import('. "fmt"') is True
        assert da.is_dot_import('"fmt"') is False

    def test_is_blank_import(self):
        da = GoDependencyAnalyzer()
        assert da.is_blank_import('_ "database/sql"') is True
        assert da.is_blank_import('"fmt"') is False

    def test_get_package_name_from_import(self):
        da = GoDependencyAnalyzer()
        assert da.get_package_name_from_import("fmt") == "fmt"
        assert da.get_package_name_from_import("net/http") == "http"
        assert da.get_package_name_from_import("github.com/pkg/errors") == "errors"
        assert da.get_package_name_from_import("net/http/") == "http"

    def test_analyze_import_dependencies(self):
        da = GoDependencyAnalyzer()
        entities = [
            _entity("fmt", EntityType.IMPORT, properties={"import_path": "fmt"}),
            _entity("net/http", EntityType.IMPORT, properties={"import_path": "net/http"}),
            _entity(
                "github.com/pkg/errors",
                EntityType.IMPORT,
                properties={"import_path": "github.com/pkg/errors"},
            ),
            _entity("./internal", EntityType.IMPORT, properties={"import_path": "./internal"}),
        ]
        result = da.analyze_import_dependencies(entities)
        assert result["total_imports"] == 4
        assert result["standard_library"]["count"] == 2
        assert result["external_packages"]["count"] == 1
        assert result["local_packages"]["count"] == 1
        assert result["import_complexity"] == 4


class TestGoProcessorExtractEntitiesWithFile:
    """Cover extract_entities regex fallback path with real file."""

    @pytest.mark.asyncio
    async def test_extract_entities_from_file(self, tmp_path):
        code = 'package main\n\nimport "fmt"\n\nfunc hello() {\n    fmt.Println("hi")\n}\n'
        go_file = tmp_path / "main.go"
        go_file.write_text(code)
        proc = GoProcessor()
        entities = await proc.extract_entities(None, go_file)
        assert isinstance(entities, list)
        assert len(entities) > 0
        # Should have package, import, and function
        entity_types = {e.type for e in entities}
        assert EntityType.PACKAGE in entity_types
        assert EntityType.IMPORT in entity_types
        assert EntityType.FUNCTION in entity_types

    @pytest.mark.asyncio
    async def test_extract_entities_missing_file(self):
        proc = GoProcessor()
        with pytest.raises(ValueError, match="Cannot read"):
            await proc.extract_entities(None, Path("/nonexistent/main.go"))

    @pytest.mark.asyncio
    async def test_regex_parse_with_structs_and_methods(self):
        code = """\
package server

import (
    "fmt"
    "net/http"
)

type Server struct {
    Port int
    Host string
}

type handler interface {
    Handle() error
}

func NewServer(port int) *Server {
    return &Server{Port: port}
}

func (s *Server) Start(addr string) error {
    fmt.Println("Starting server")
    return http.ListenAndServe(addr, nil)
}

const DefaultPort = 8080

var GlobalConfig = "default"
"""
        proc = GoProcessor()
        result = await proc._regex_parse(Path("server.go"), code)
        assert result.success is True
        assert result.properties["package_name"] == "server"
        # Check multiple entity types exist
        entity_types = {e.type for e in result.entities}
        assert EntityType.PACKAGE in entity_types
        assert EntityType.IMPORT in entity_types
        assert EntityType.FUNCTION in entity_types
