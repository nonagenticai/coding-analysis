"""Unit tests for the four big analyzer modules:

- src/processing/analyzers/code_analyzer.py
- src/processing/analyzers/test_execution.py
- src/processing/analyzers/integration_tests.py
- src/processing/analyzers/format_converter.py
"""

from __future__ import annotations

import asyncio
import base64

# ---------------------------------------------------------------------------
# Stub heavy optional deps that may not be installed in the test env
# ---------------------------------------------------------------------------
import importlib.machinery
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _install_stub(name: str, mod: types.ModuleType) -> None:
    """Install a fake module stub that survives conftest _purge_stubs().

    The unit conftest removes modules with no __spec__, __file__, or __loader__
    between test files.  Setting __spec__ prevents the stub from being purged.
    """
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules[name] = mod


if "graphviz" not in sys.modules:
    _gv = types.ModuleType("graphviz")

    class _FakeDigraph:
        def __init__(self, **kw):
            self.nodes_list = []
            self.edges_list = []

        def attr(self, *a, **kw):
            pass

        def node(self, *a, **kw):
            self.nodes_list.append(a)

        def edge(self, *a, **kw):
            self.edges_list.append(a)

        def render(self, path, cleanup=False):
            # Create a dummy SVG file
            with open(f"{path}.svg", "w") as f:
                f.write("<svg></svg>")

    _gv.Digraph = _FakeDigraph
    _install_stub("graphviz", _gv)

if "markdown" not in sys.modules:
    _md_mod = types.ModuleType("markdown")

    class _FakeMarkdown:
        def __init__(self, extensions=None):
            pass

        def convert(self, text):
            return f"<p>{text}</p>"

    _md_mod.Markdown = _FakeMarkdown
    _install_stub("markdown", _md_mod)

if "weasyprint" not in sys.modules:
    _wp = types.ModuleType("weasyprint")

    class _FakeHTML:
        def __init__(self, string=""):
            self._string = string

        def write_pdf(self, path):
            with open(path, "wb") as f:
                f.write(b"%PDF-fake")

    _wp.HTML = _FakeHTML
    _install_stub("weasyprint", _wp)

if "toml" not in sys.modules:
    _toml = types.ModuleType("toml")
    _toml.load = lambda f: {}
    _install_stub("toml", _toml)

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from processing.analyzers.code_analyzer import CodeAnalyzer
from processing.analyzers.format_converter import FormatConverter

# integration_tests.py imports many heavy deps; stub the classes it instantiates
# so we never need real analysis backends.
from processing.analyzers.integration_tests import (
    StaticAnalysisIntegrationTests,
    run_integration_tests,
)
from processing.analyzers.test_discovery import (
    TestFileManager,
    TestProject,
    TestScenario,
    TestScenarioType,
)
from processing.analyzers.test_execution import (
    ComponentTestConfig,
    ComponentTestRunner,
    IntegrationTestExecutor,
    TestEnvironmentManager,
    TestExecutionResult,
    TestExecutionStatus,
    execute_component_tests,
    run_integration_test_suite,
)

# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def _make_scenario(
    name: str = "test_scenario",
    scenario_type: TestScenarioType = TestScenarioType.CALL_GRAPH,
    test_code: str = "def main():\n    pass\n",
    expected_results: dict | None = None,
    metadata: dict | None = None,
) -> TestScenario:
    return TestScenario(
        name=name,
        scenario_type=scenario_type,
        test_code=test_code,
        expected_results=expected_results or {},
        metadata=metadata or {},
    )


def _tmp_project_dir() -> str:
    """Create a small temp project with a Python file and return the path."""
    d = tempfile.mkdtemp()
    py = os.path.join(d, "hello.py")
    with open(py, "w") as f:
        f.write("def greet(name: str) -> str:\n    return f'Hello {name}'\n")
    return d


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — CodeAnalyzer
# ═══════════════════════════════════════════════════════════════════════════


class TestCodeAnalyzerInit:
    def test_init(self):
        analyzer = CodeAnalyzer()
        assert analyzer is not None


class TestAnalyzeProjectStructure:
    """Tests for CodeAnalyzer.analyze_project_structure."""

    def test_basic_structure(self, tmp_path):
        # Create a tiny project
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print(1)")
        (tmp_path / "README.md").write_text("# hi")

        analyzer = CodeAnalyzer()
        result = analyzer.analyze_project_structure(str(tmp_path))

        assert "src" in result["directories"]
        assert any("main.py" in f for f in result["files"])
        assert "Python" in result["languages"]
        assert "Documentation" in result["languages"]
        assert result["size_metrics"]["total_files"] >= 2
        assert ".py" in result["file_types"]

    def test_skip_hidden_dirs_and_files(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("x")
        (tmp_path / ".hidden").write_text("x")
        (tmp_path / "visible.py").write_text("x=1")

        analyzer = CodeAnalyzer()
        result = analyzer.analyze_project_structure(str(tmp_path))

        assert ".git" not in result["directories"]
        assert not any(".hidden" in f for f in result["files"])
        assert any("visible.py" in f for f in result["files"])

    def test_skip_pycache_node_modules(self, tmp_path):
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "venv").mkdir()
        (tmp_path / "app.js").write_text("1")

        analyzer = CodeAnalyzer()
        result = analyzer.analyze_project_structure(str(tmp_path))

        for skipped in ["__pycache__", "node_modules", "venv"]:
            assert skipped not in result["directories"]

    def test_language_detection(self, tmp_path):
        exts = {
            "a.py": "Python",
            "b.js": "JavaScript",
            "c.ts": "TypeScript",
            "d.java": "Java",
            "e.c": "C",
            "f.cpp": "C++",
            "g.go": "Go",
            "h.rs": "Rust",
        }
        for fname in exts:
            (tmp_path / fname).write_text("x")

        analyzer = CodeAnalyzer()
        result = analyzer.analyze_project_structure(str(tmp_path))

        for lang in exts.values():
            assert lang in result["languages"], f"{lang} not detected"

    def test_empty_directory(self, tmp_path):
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_project_structure(str(tmp_path))

        assert result["files"] == []
        assert result["directories"] == []
        assert result["size_metrics"]["total_files"] == 0

    def test_oserror_returns_empty(self):
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_project_structure("/nonexistent/path/xyz")

        # Should not crash; returns the default structure
        assert isinstance(result, dict)
        assert "files" in result

    def test_size_metrics_calculated(self, tmp_path):
        content = "x" * 1024  # 1 KB
        (tmp_path / "data.py").write_text(content)

        analyzer = CodeAnalyzer()
        result = analyzer.analyze_project_structure(str(tmp_path))

        assert result["size_metrics"]["total_size_bytes"] >= 1024
        assert result["size_metrics"]["total_size_mb"] >= 0

    def test_c_header_detection(self, tmp_path):
        (tmp_path / "header.h").write_text("int x;")
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_project_structure(str(tmp_path))
        assert "C" in result["languages"]

    def test_cpp_variants(self, tmp_path):
        for ext in [".cc", ".cxx", ".hpp"]:
            (tmp_path / f"file{ext}").write_text("x")
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_project_structure(str(tmp_path))
        assert "C++" in result["languages"]

    def test_rst_detected_as_documentation(self, tmp_path):
        (tmp_path / "index.rst").write_text("====\nHi\n====")
        analyzer = CodeAnalyzer()
        result = analyzer.analyze_project_structure(str(tmp_path))
        assert "Documentation" in result["languages"]


class TestAnalyzeRecentChanges:
    """Tests for CodeAnalyzer.analyze_recent_changes (async)."""

    async def test_not_a_git_repo(self, tmp_path):
        analyzer = CodeAnalyzer()
        result = await analyzer.analyze_recent_changes(str(tmp_path))

        assert result["error"] == "Not a git repository"
        assert result["recent_commits"] == []

    async def test_git_repo_happy_path(self, tmp_path):
        (tmp_path / ".git").mkdir()

        commit_output = "abc1234|Alice|a@b.com|2025-01-01|Fix stuff"
        files_output = "src/main.py\nREADME.md"
        stats_output = "2 files changed, 10 insertions(+), 3 deletions(-)"
        branch_output = "main"

        def fake_run(cmd, **kwargs):
            joined = " ".join(cmd)
            result = MagicMock()
            result.returncode = 0
            if "log" in joined:
                result.stdout = commit_output
            elif "diff --name-only" in joined:
                result.stdout = files_output
            elif "diff --shortstat" in joined:
                result.stdout = stats_output
            elif "branch --show-current" in joined:
                result.stdout = branch_output
            else:
                result.stdout = ""
            return result

        with patch("processing.analyzers.code_analyzer.subprocess.run", side_effect=fake_run):
            analyzer = CodeAnalyzer()
            result = await analyzer.analyze_recent_changes(str(tmp_path))

        assert len(result["recent_commits"]) == 1
        assert result["recent_commits"][0]["author"] == "Alice"
        assert result["recent_commits"][0]["hash"] == "abc1234"[:7]
        assert "src/main.py" in result["files_changed"]
        assert result["change_statistics"]["files_changed"] == 2
        assert result["change_statistics"]["insertions"] == 10
        assert result["change_statistics"]["deletions"] == 3
        assert result["current_branch"] == "main"

    async def test_git_command_failure(self, tmp_path):
        (tmp_path / ".git").mkdir()

        with patch(
            "processing.analyzers.code_analyzer.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "git"),
        ):
            analyzer = CodeAnalyzer()
            result = await analyzer.analyze_recent_changes(str(tmp_path))

        assert "error" in result
        assert result["recent_commits"] == []

    async def test_no_recent_commits(self, tmp_path):
        (tmp_path / ".git").mkdir()

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            return result

        with patch("processing.analyzers.code_analyzer.subprocess.run", side_effect=fake_run):
            analyzer = CodeAnalyzer()
            result = await analyzer.analyze_recent_changes(str(tmp_path))

        assert result["recent_commits"] == []

    async def test_diff_name_only_fallback(self, tmp_path):
        """When diff --name-only fails, falls back to diff-tree."""
        (tmp_path / ".git").mkdir()

        call_count = 0

        def fake_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            joined = " ".join(cmd)
            result = MagicMock()
            result.returncode = 0

            if "log" in joined:
                result.stdout = "abc1234|Alice|a@b.com|2025-01-01|Fix"
            elif "diff --name-only" in joined:
                raise subprocess.CalledProcessError(1, "git diff")
            elif "diff-tree" in joined:
                result.stdout = "fallback.py"
            elif "diff --shortstat" in joined:
                result.stdout = ""
            elif "branch" in joined:
                result.stdout = "main"
            else:
                result.stdout = ""
            return result

        with patch("processing.analyzers.code_analyzer.subprocess.run", side_effect=fake_run):
            analyzer = CodeAnalyzer()
            result = await analyzer.analyze_recent_changes(str(tmp_path))

        assert "fallback.py" in result["files_changed"]


class TestAssessChangeImpact:
    """Tests for CodeAnalyzer._assess_change_impact."""

    def setup_method(self):
        self.analyzer = CodeAnalyzer()

    def test_no_changes(self):
        assert self.analyzer._assess_change_impact([], []) == "No changes detected"

    def test_critical_files(self):
        files = ["setup.py", "src/auth.py"]
        result = self.analyzer._assess_change_impact(files, [])
        assert "HIGH RISK" in result

    def test_test_files(self):
        files = ["tests/test_main.py", "test_utils.py"]
        result = self.analyzer._assess_change_impact(files, [])
        assert "test files updated" in result

    def test_doc_files(self):
        files = ["README.md", "docs/guide.rst"]
        result = self.analyzer._assess_change_impact(files, [])
        assert "documentation files updated" in result

    def test_config_files(self):
        files = ["pyproject.toml", "package.json"]
        result = self.analyzer._assess_change_impact(files, [])
        assert "configuration files changed" in result

    def test_large_scope(self):
        files = [f"src/module_{i}.py" for i in range(15)]
        result = self.analyzer._assess_change_impact(files, [])
        assert "Large change scope" in result

    def test_moderate_scope(self):
        files = ["src/a.py", "src/b.py"]
        result = self.analyzer._assess_change_impact(files, [])
        assert "Moderate scope" in result

    def test_risky_commits(self):
        commits = [{"message": "breaking change to API"}]
        files = ["src/app.py"]
        result = self.analyzer._assess_change_impact(files, commits)
        assert "risk indicators" in result

    def test_low_risk(self):
        # Files with unknown extensions don't match any category
        files = ["data.csv"]
        result = self.analyzer._assess_change_impact(files, [])
        # data.csv goes into source_files -> moderate scope
        assert "Moderate scope" in result

    def test_security_keyword_in_path(self):
        files = ["src/security/validator.py"]
        result = self.analyzer._assess_change_impact(files, [])
        assert "HIGH RISK" in result

    def test_database_keyword_in_path(self):
        files = ["src/database/models.py"]
        result = self.analyzer._assess_change_impact(files, [])
        assert "HIGH RISK" in result


class TestExtractPythonDependencies:
    """Tests for CodeAnalyzer.extract_python_dependencies."""

    async def test_requirements_txt(self, tmp_path):
        req = tmp_path / "requirements.txt"
        req.write_text("flask>=2.0\nrequests==2.28.0\n# comment\n")

        analyzer = CodeAnalyzer()
        deps = await analyzer.extract_python_dependencies(str(tmp_path))

        assert "flask" in deps
        assert deps["flask"]["dev"] is False
        assert "requests" in deps

    async def test_setup_py(self, tmp_path):
        setup = tmp_path / "setup.py"
        setup.write_text(
            "from setuptools import setup\nsetup(install_requires=['numpy>=1.0', 'pandas'])\n"
        )

        analyzer = CodeAnalyzer()
        deps = await analyzer.extract_python_dependencies(str(tmp_path))

        assert "numpy" in deps
        assert "pandas" in deps

    async def test_pyproject_toml_pep621(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\ndependencies = ['flask>=2.0']\n")

        # Override toml.load to return parsed data
        parsed = {"project": {"dependencies": ["flask>=2.0", "requests<3"]}}
        with patch("toml.load", return_value=parsed):
            analyzer = CodeAnalyzer()
            deps = await analyzer.extract_python_dependencies(str(tmp_path))

        assert "flask" in deps
        assert "requests" in deps

    async def test_pyproject_toml_poetry(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.poetry]\n")

        parsed = {
            "tool": {
                "poetry": {
                    "dependencies": {"python": "^3.10", "flask": "^2.0"},
                    "dev-dependencies": {"pytest": "^7.0"},
                }
            }
        }
        with patch("toml.load", return_value=parsed):
            analyzer = CodeAnalyzer()
            deps = await analyzer.extract_python_dependencies(str(tmp_path))

        assert "flask" in deps
        assert deps["flask"]["dev"] is False
        assert "pytest" in deps
        assert deps["pytest"]["dev"] is True
        # python should be excluded
        assert "python" not in deps

    async def test_empty_project(self, tmp_path):
        analyzer = CodeAnalyzer()
        deps = await analyzer.extract_python_dependencies(str(tmp_path))
        assert deps == {}

    async def test_toml_import_error(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\n")

        with patch("builtins.__import__", side_effect=ImportError("no toml")):
            # This will catch the ImportError inside the method
            analyzer = CodeAnalyzer()
            # Since import toml fails, it should skip pyproject.toml parsing
            deps = await analyzer.extract_python_dependencies(str(tmp_path))
            assert isinstance(deps, dict)


class TestExtractJavascriptDependencies:
    """Tests for CodeAnalyzer.extract_javascript_dependencies."""

    async def test_package_json(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps(
                {
                    "dependencies": {"express": "^4.18"},
                    "devDependencies": {"jest": "^29.0"},
                    "optionalDependencies": {"fsevents": "^2.3"},
                }
            )
        )

        analyzer = CodeAnalyzer()
        deps = await analyzer.extract_javascript_dependencies(str(tmp_path))

        assert deps["express"]["dev"] is False
        assert deps["jest"]["dev"] is True
        assert deps["fsevents"]["optional"] is True

    async def test_no_package_json(self, tmp_path):
        analyzer = CodeAnalyzer()
        deps = await analyzer.extract_javascript_dependencies(str(tmp_path))
        assert deps == {}

    async def test_invalid_json(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text("{invalid json")

        analyzer = CodeAnalyzer()
        deps = await analyzer.extract_javascript_dependencies(str(tmp_path))
        assert deps == {}


class TestGenerateProjectDocumentationData:
    """Tests for CodeAnalyzer.generate_project_documentation_data."""

    async def test_basic_project(self, tmp_path):
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "README.md").write_text("# Project\nDescription here.")

        analyzer = CodeAnalyzer()
        result = await analyzer.generate_project_documentation_data(str(tmp_path))

        assert "project_structure" in result
        assert "documentation_files" in result
        assert "dependencies" in result
        assert "analysis_timestamp" in result

        # README should be found
        assert any("README.md" in k for k in result["documentation_files"])

    async def test_with_python_deps(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask>=2.0\n")
        (tmp_path / "app.py").write_text("import flask")

        analyzer = CodeAnalyzer()
        result = await analyzer.generate_project_documentation_data(str(tmp_path))

        assert "python" in result["dependencies"]

    async def test_error_handling(self):
        analyzer = CodeAnalyzer()
        # Passing a path that doesn't exist should handle gracefully
        with patch.object(
            analyzer,
            "analyze_project_structure",
            side_effect=RuntimeError("boom"),
        ):
            result = await analyzer.generate_project_documentation_data("/fake/path")
            assert "error" in result


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — test_execution.py
# ═══════════════════════════════════════════════════════════════════════════


class TestTestExecutionStatus:
    def test_enum_values(self):
        assert TestExecutionStatus.PENDING.value == "pending"
        assert TestExecutionStatus.RUNNING.value == "running"
        assert TestExecutionStatus.COMPLETED.value == "completed"
        assert TestExecutionStatus.FAILED.value == "failed"
        assert TestExecutionStatus.SKIPPED.value == "skipped"


class TestTestExecutionResult:
    def test_create_result(self):
        r = TestExecutionResult(
            component="cg",
            test_name="t1",
            status=TestExecutionStatus.COMPLETED,
            execution_time=1.23,
            results={"key": "val"},
        )
        assert r.component == "cg"
        assert r.error_message is None
        assert r.metadata is None

    def test_with_error(self):
        r = TestExecutionResult(
            component="cg",
            test_name="t1",
            status=TestExecutionStatus.FAILED,
            execution_time=0.0,
            results={},
            error_message="something broke",
        )
        assert r.error_message == "something broke"


class TestComponentTestRunner:
    """Tests for ComponentTestRunner methods."""

    def _make_runner(self):
        fm = MagicMock(spec=TestFileManager)
        fm.create_test_file.return_value = "/tmp/test.py"
        project = TestProject(root_dir="/tmp/proj", files={"src/main.py": "/tmp/proj/src/main.py"})
        fm.create_integration_test_project.return_value = project
        return ComponentTestRunner(fm)

    async def test_run_call_graph_tests_success(self):
        runner = self._make_runner()
        scenario = _make_scenario(name="cg_test")

        # Mock CallGraphGenerator
        mock_file_result = MagicMock()
        mock_file_result.success = True
        mock_file_result.metadata = {"call_graph": {}}

        mock_comp_result = MagicMock()
        mock_comp_result.analysis_id = "xyz"
        mock_comp_result.call_graph = {"nodes": [1], "edges": [2]}
        mock_comp_result.metrics = MagicMock(total_functions=5)
        mock_comp_result.orphaned_functions = ["orphan"]

        with patch("processing.analyzers.test_execution.CallGraphGenerator") as mock_cg:
            inst = mock_cg.return_value
            inst.analyze_file = AsyncMock(return_value=mock_file_result)
            inst.generate_comprehensive_call_graph = AsyncMock(return_value=mock_comp_result)

            result = await runner.run_call_graph_tests(scenario)

        assert result.status == TestExecutionStatus.COMPLETED
        assert result.component == "call_graph"
        assert result.results["single_file"]["success"] is True
        assert result.results["comprehensive"]["has_nodes"] is True

    async def test_run_call_graph_tests_failure(self):
        runner = self._make_runner()
        scenario = _make_scenario(name="cg_fail")

        with patch(
            "processing.analyzers.test_execution.CallGraphGenerator",
            side_effect=RuntimeError("boom"),
        ):
            result = await runner.run_call_graph_tests(scenario)

        assert result.status == TestExecutionStatus.FAILED
        assert "boom" in result.error_message

    async def test_run_signature_analysis_tests_success(self):
        runner = self._make_runner()
        scenario = _make_scenario(name="sig_test")

        mock_sigs = {"func1": {"docstring": "hi"}}
        mock_comp = {
            "analysis_id": "abc",
            "functions": [{"usage_examples": True, "complexity_metrics": True}],
            "type_coverage": 0.8,
            "external_dependencies": ["os"],
        }

        with patch("processing.analyzers.test_execution.FunctionExtractor") as mock_fe:
            inst = mock_fe.return_value
            inst.extract_function_signatures.return_value = mock_sigs
            inst.extract_comprehensive_signatures.return_value = mock_comp

            result = await runner.run_signature_analysis_tests(scenario)

        assert result.status == TestExecutionStatus.COMPLETED
        assert result.component == "signature_analysis"

    async def test_run_signature_analysis_tests_failure(self):
        runner = self._make_runner()
        scenario = _make_scenario(name="sig_fail")

        with patch(
            "processing.analyzers.test_execution.FunctionExtractor",
            side_effect=ValueError("bad"),
        ):
            result = await runner.run_signature_analysis_tests(scenario)

        assert result.status == TestExecutionStatus.FAILED

    async def test_run_symbolic_validation_tests_success(self):
        runner = self._make_runner()
        scenario = _make_scenario(name="sym_test", test_code="def f(a, b):\n    return a / b\n")

        mock_file_result = MagicMock()
        mock_file_result.success = True
        mock_file_result.metadata = {"symbolic_analysis": {}}

        mock_val_result = MagicMock()
        mock_val_result.analysis_id = "v1"
        mock_val_result.validation_results = [{"ok": True}]
        mock_val_result.overall_safety_score = 85
        mock_val_result.recommendations = ["Add null check"]
        mock_val_result.blocking_issues = []

        with patch("processing.analyzers.test_execution.SymbolicValidator") as mock_sv:
            inst = mock_sv.return_value
            inst.analyze_file = AsyncMock(return_value=mock_file_result)
            inst.validate_code_changes = AsyncMock(return_value=mock_val_result)

            result = await runner.run_symbolic_validation_tests(scenario)

        assert result.status == TestExecutionStatus.COMPLETED
        assert result.results["code_changes"]["safety_score"] == 85

    async def test_run_documentation_tests_success(self):
        runner = self._make_runner()
        scenario = _make_scenario(name="doc_test")

        mock_file_result = MagicMock()
        mock_file_result.success = True
        mock_file_result.metadata = {"documentation": {}}

        mock_proj_result = {"project_structure": {}, "api_documentation": {}, "coverage": 0.9}

        mock_comp_result = MagicMock()
        mock_comp_result.analysis_id = "d1"
        mock_comp_result.documentation = {"section": "content"}
        mock_comp_result.quality_score = 88
        mock_comp_result.generated_files = ["a.md", "b.md"]

        with patch("processing.analyzers.test_execution.DocumentationEngine") as mock_de:
            inst = mock_de.return_value
            inst.analyze_file = AsyncMock(return_value=mock_file_result)
            inst.generate_project_documentation = AsyncMock(return_value=mock_proj_result)
            inst.generate_comprehensive_documentation = AsyncMock(return_value=mock_comp_result)

            result = await runner.run_documentation_tests(scenario)

        assert result.status == TestExecutionStatus.COMPLETED
        assert result.results["comprehensive"]["quality_score"] == 88

    async def test_run_performance_tests_success(self):
        runner = self._make_runner()
        scenario = _make_scenario(name="perf_test")

        mock_file_result = MagicMock()
        mock_file_result.success = True
        mock_file_result.metadata = {}

        # Patch all four analysis components
        with (
            patch("processing.analyzers.test_execution.CallGraphGenerator") as mock_cg,
            patch("processing.analyzers.test_execution.FunctionExtractor") as mock_fe,
            patch("processing.analyzers.test_execution.SymbolicValidator") as mock_sv,
            patch("processing.analyzers.test_execution.DocumentationEngine") as mock_de,
        ):
            for mock_cls in [mock_cg, mock_sv, mock_de]:
                mock_cls.return_value.analyze_file = AsyncMock(return_value=mock_file_result)

            mock_fe.return_value.extract_comprehensive_signatures.return_value = {}

            result = await runner.run_performance_tests(scenario)

        assert result.status == TestExecutionStatus.COMPLETED
        assert "individual_times" in result.results
        assert result.results["acceptable_performance"] is True

    async def test_run_component_tests_unknown_component(self):
        runner = self._make_runner()
        config = ComponentTestConfig(
            component_name="nonexistent",
            test_scenarios=[_make_scenario()],
        )
        with pytest.raises(ValueError, match="Unknown component"):
            await runner.run_component_tests(config)

    async def test_run_component_tests_sequential(self):
        runner = self._make_runner()

        mock_file_result = MagicMock()
        mock_file_result.success = True
        mock_file_result.metadata = {"call_graph": {}}

        mock_comp_result = MagicMock()
        mock_comp_result.analysis_id = "x"
        mock_comp_result.call_graph = {"nodes": [], "edges": []}
        mock_comp_result.metrics = MagicMock(total_functions=0)
        mock_comp_result.orphaned_functions = []

        config = ComponentTestConfig(
            component_name="call_graph",
            test_scenarios=[_make_scenario(), _make_scenario(name="test2")],
            parallel_execution=False,
        )

        with patch("processing.analyzers.test_execution.CallGraphGenerator") as mock_cg:
            inst = mock_cg.return_value
            inst.analyze_file = AsyncMock(return_value=mock_file_result)
            inst.generate_comprehensive_call_graph = AsyncMock(return_value=mock_comp_result)

            results = await runner.run_component_tests(config)

        assert len(results) == 2
        assert all(r.status == TestExecutionStatus.COMPLETED for r in results)

    async def test_run_component_tests_parallel(self):
        runner = self._make_runner()

        mock_file_result = MagicMock()
        mock_file_result.success = True
        mock_file_result.metadata = {"call_graph": {}}

        mock_comp_result = MagicMock()
        mock_comp_result.analysis_id = "x"
        mock_comp_result.call_graph = {"nodes": [], "edges": []}
        mock_comp_result.metrics = MagicMock(total_functions=0)
        mock_comp_result.orphaned_functions = []

        config = ComponentTestConfig(
            component_name="call_graph",
            test_scenarios=[_make_scenario(), _make_scenario(name="test2")],
            parallel_execution=True,
        )

        with patch("processing.analyzers.test_execution.CallGraphGenerator") as mock_cg:
            inst = mock_cg.return_value
            inst.analyze_file = AsyncMock(return_value=mock_file_result)
            inst.generate_comprehensive_call_graph = AsyncMock(return_value=mock_comp_result)

            results = await runner.run_component_tests(config)

        assert len(results) == 2

    async def test_run_component_tests_timeout(self):
        runner = self._make_runner()

        async def slow_runner(scenario):
            await asyncio.sleep(10)

        config = ComponentTestConfig(
            component_name="call_graph",
            test_scenarios=[_make_scenario()],
            timeout=0.01,  # Very short timeout
        )

        with patch.object(runner, "run_call_graph_tests", side_effect=slow_runner):
            results = await runner.run_component_tests(config)

        assert len(results) == 1
        assert results[0].status == TestExecutionStatus.FAILED
        assert "timed out" in results[0].error_message

    async def test_run_component_tests_exception_in_sequential(self):
        runner = self._make_runner()

        config = ComponentTestConfig(
            component_name="call_graph",
            test_scenarios=[_make_scenario()],
        )

        with patch.object(runner, "run_call_graph_tests", side_effect=RuntimeError("unexpected")):
            results = await runner.run_component_tests(config)

        assert len(results) == 1
        assert results[0].status == TestExecutionStatus.FAILED
        assert "unexpected" in results[0].error_message


class TestIntegrationTestExecutor:
    """Tests for IntegrationTestExecutor."""

    async def test_run_integration_tests_success(self):
        executor = IntegrationTestExecutor()

        mock_result = TestExecutionResult(
            component="call_graph",
            test_name="test1",
            status=TestExecutionStatus.COMPLETED,
            execution_time=0.5,
            results={"ok": True},
        )

        with patch.object(
            executor.component_runner,
            "run_component_tests",
            new=AsyncMock(return_value=[mock_result]),
        ):
            with patch.object(executor.file_manager, "__enter__", return_value=None):
                with patch.object(executor.file_manager, "__exit__", return_value=False):
                    config = ComponentTestConfig(
                        component_name="call_graph",
                        test_scenarios=[_make_scenario()],
                    )
                    result = await executor.run_integration_tests([config])

        assert "summary" in result
        assert result["summary"]["total_tests"] == 1
        assert result["summary"]["passed_tests"] == 1
        assert result["summary"]["all_tests_passed"] is True

    async def test_run_integration_tests_error(self):
        executor = IntegrationTestExecutor()

        with patch.object(
            executor.component_runner,
            "run_component_tests",
            new=AsyncMock(side_effect=RuntimeError("fail")),
        ):
            with patch.object(executor.file_manager, "__enter__", return_value=None):
                with patch.object(executor.file_manager, "__exit__", return_value=False):
                    config = ComponentTestConfig(
                        component_name="call_graph",
                        test_scenarios=[_make_scenario()],
                    )
                    result = await executor.run_integration_tests([config])

        assert "error" in result

    async def test_run_end_to_end_integration(self):
        executor = IntegrationTestExecutor()

        mock_project = TestProject(
            root_dir="/tmp/test_proj",
            files={"src/main.py": "/tmp/test_proj/src/main.py"},
        )

        integration_results = {
            "call_graph": {"success": True, "functions_found": 5},
            "signatures": {"success": True, "functions_analyzed": 3},
            "validation": {"success": True, "safety_score": 90},
            "documentation": {"success": True, "quality_score": 85},
        }

        with (
            patch.object(executor.file_manager, "__enter__", return_value=None),
            patch.object(executor.file_manager, "__exit__", return_value=False),
            patch.object(
                executor.file_manager,
                "create_integration_test_project",
                return_value=mock_project,
            ),
            patch.object(
                executor,
                "_run_integration_components",
                new=AsyncMock(return_value=integration_results),
            ),
            patch("processing.analyzers.test_execution.Path") as mock_path,
        ):
            mock_path.return_value.rglob.return_value = ["a.py", "b.py"]
            result = await executor.run_end_to_end_integration()

        assert result["all_components_working"] is True

    async def test_compile_test_summary_mixed(self):
        executor = IntegrationTestExecutor()

        results = {
            "cg": [
                TestExecutionResult(
                    component="cg",
                    test_name="t1",
                    status=TestExecutionStatus.COMPLETED,
                    execution_time=1.0,
                    results={},
                ),
                TestExecutionResult(
                    component="cg",
                    test_name="t2",
                    status=TestExecutionStatus.FAILED,
                    execution_time=0.5,
                    results={},
                    error_message="err",
                ),
            ]
        }
        summary = executor._compile_test_summary(results, 2.0)

        assert summary["total_tests"] == 2
        assert summary["passed_tests"] == 1
        assert summary["failed_tests"] == 1
        assert summary["all_tests_passed"] is False
        assert summary["success_rate"] == 0.5

    async def test_compile_test_summary_empty(self):
        executor = IntegrationTestExecutor()
        summary = executor._compile_test_summary({}, 0.0)

        assert summary["total_tests"] == 0
        assert summary["success_rate"] == 0.0


class TestTestEnvironmentManager:
    """Tests for TestEnvironmentManager."""

    async def test_setup_environment(self):
        mgr = TestEnvironmentManager()
        env = await mgr.setup_test_environment("test-1")

        assert env["test_id"] == "test-1"
        assert "test-1" in mgr.active_environments

    async def test_teardown_environment(self):
        mgr = TestEnvironmentManager()
        await mgr.setup_test_environment("test-1")
        assert "test-1" in mgr.active_environments

        await mgr.teardown_test_environment("test-1")
        assert "test-1" not in mgr.active_environments

    async def test_teardown_nonexistent(self):
        mgr = TestEnvironmentManager()
        # Should not raise
        await mgr.teardown_test_environment("nonexistent")

    def test_check_resource_limits_valid(self):
        mgr = TestEnvironmentManager()
        # Manually add environment
        mgr.active_environments["test-1"] = {
            "start_time": time.time(),
            "resource_usage": {"temp_files_count": 5},
        }
        assert mgr.check_resource_limits("test-1") is True

    def test_check_resource_limits_nonexistent(self):
        mgr = TestEnvironmentManager()
        assert mgr.check_resource_limits("nope") is False

    def test_check_resource_limits_exceeded_time(self):
        mgr = TestEnvironmentManager()
        mgr.active_environments["test-1"] = {
            "start_time": time.time() - 600,  # 10 minutes ago
            "resource_usage": {"temp_files_count": 0},
        }
        assert mgr.check_resource_limits("test-1") is False

    def test_check_resource_limits_exceeded_files(self):
        mgr = TestEnvironmentManager()
        mgr.active_environments["test-1"] = {
            "start_time": time.time(),
            "resource_usage": {"temp_files_count": 999},
        }
        assert mgr.check_resource_limits("test-1") is False

    async def test_cleanup_all(self):
        mgr = TestEnvironmentManager()
        await mgr.setup_test_environment("a")
        await mgr.setup_test_environment("b")
        assert len(mgr.active_environments) == 2

        await mgr.cleanup_all_environments()
        assert len(mgr.active_environments) == 0


class TestConvenienceFunctions:
    """Tests for execute_component_tests and run_integration_test_suite."""

    async def test_execute_component_tests_success(self):
        mock_result = TestExecutionResult(
            component="call_graph",
            test_name="t1",
            status=TestExecutionStatus.COMPLETED,
            execution_time=0.5,
            results={},
        )

        with patch.object(
            ComponentTestRunner,
            "run_component_tests",
            new=AsyncMock(return_value=[mock_result]),
        ):
            with patch.object(TestFileManager, "__enter__", return_value=None):
                with patch.object(TestFileManager, "__exit__", return_value=False):
                    results = await execute_component_tests("call_graph", [_make_scenario()])

        assert len(results) == 1

    async def test_execute_component_tests_error(self):
        with patch.object(
            ComponentTestRunner,
            "run_component_tests",
            new=AsyncMock(side_effect=RuntimeError("fail")),
        ):
            with patch.object(TestFileManager, "__enter__", return_value=None):
                with patch.object(TestFileManager, "__exit__", return_value=False):
                    results = await execute_component_tests("call_graph", [_make_scenario()])

        assert results == []

    async def test_run_integration_test_suite(self):
        with patch.object(
            IntegrationTestExecutor,
            "run_integration_tests",
            new=AsyncMock(return_value={"summary": {"all_passed": True}}),
        ):
            result = await run_integration_test_suite([])

        assert "summary" in result


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — integration_tests.py (StaticAnalysisIntegrationTests)
# ═══════════════════════════════════════════════════════════════════════════


class TestStaticAnalysisIntegrationTestsInit:
    def test_init_creates_result_structure(self):
        suite = StaticAnalysisIntegrationTests()
        assert "call_graph" in suite.test_results
        assert "signature_analysis" in suite.test_results
        assert "symbolic_validation" in suite.test_results
        assert "documentation" in suite.test_results
        assert "performance" in suite.test_results
        assert "integration" in suite.test_results


class TestCheckAllTestsPassed:
    def test_all_passed(self):
        suite = StaticAnalysisIntegrationTests()
        suite.test_results = {
            "call_graph": {"single_file": {"success": True}},
            "documentation": {"file": {"success": True}},
        }
        assert suite._check_all_tests_passed() is True

    def test_has_error(self):
        suite = StaticAnalysisIntegrationTests()
        suite.test_results = {
            "call_graph": {"error": "something failed"},
        }
        assert suite._check_all_tests_passed() is False

    def test_has_false_success(self):
        suite = StaticAnalysisIntegrationTests()
        suite.test_results = {
            "call_graph": {"single_file": {"success": False}},
        }
        assert suite._check_all_tests_passed() is False

    def test_nested_error(self):
        suite = StaticAnalysisIntegrationTests()
        suite.test_results = {
            "call_graph": {"single_file": {"error": "fail", "success": True}},
        }
        assert suite._check_all_tests_passed() is False

    def test_summary_is_skipped(self):
        suite = StaticAnalysisIntegrationTests()
        suite.test_results = {
            "summary": {"all_tests_passed": False},
            "call_graph": {"single_file": {"success": True}},
        }
        assert suite._check_all_tests_passed() is True

    def test_invalid_results_returns_false(self):
        suite = StaticAnalysisIntegrationTests()
        suite.test_results = None  # type: ignore
        assert suite._check_all_tests_passed() is False


class TestGenerateTestReport:
    def test_report_with_summary(self):
        suite = StaticAnalysisIntegrationTests()
        suite.test_results = {
            "summary": {"total_time": 1.5, "all_tests_passed": True},
            "call_graph": {"single_file": {"success": True}},
        }
        report = suite.generate_test_report()

        assert "Integration Test Report" in report
        assert "1.50 seconds" in report
        assert "call_graph" in report.lower() or "Call Graph" in report

    def test_report_with_error(self):
        suite = StaticAnalysisIntegrationTests()
        suite.test_results = {
            "call_graph": {"error": "something broke"},
        }
        report = suite.generate_test_report()
        assert "something broke" in report

    def test_report_without_summary(self):
        suite = StaticAnalysisIntegrationTests()
        suite.test_results = {
            "call_graph": {"test1": {"success": True}},
        }
        report = suite.generate_test_report()
        assert "Integration Test Report" in report


class TestRunAllTests:
    async def test_run_all_tests_success(self):
        suite = StaticAnalysisIntegrationTests()

        # Patch all private test methods
        for method in [
            "_test_call_graph_analysis",
            "_test_signature_analysis",
            "_test_symbolic_validation",
            "_test_documentation_generation",
            "_test_performance",
            "_test_end_to_end_integration",
        ]:
            setattr(suite, method, AsyncMock())

        with patch.object(suite, "_check_all_tests_passed", return_value=True):
            results = await suite.run_all_tests()

        assert "summary" in results
        assert results["summary"]["all_tests_passed"] is True

    async def test_run_all_tests_with_error(self):
        suite = StaticAnalysisIntegrationTests()

        # Make first test method raise
        suite._test_call_graph_analysis = AsyncMock(side_effect=RuntimeError("boom"))

        results = await suite.run_all_tests()
        assert "error" in results


class TestRunIntegrationTestsStandalone:
    async def test_run_integration_tests_func(self):
        with patch.object(
            StaticAnalysisIntegrationTests,
            "run_all_tests",
            new=AsyncMock(
                return_value={
                    "summary": {"all_tests_passed": True, "total_time": 0.1},
                    "call_graph": {},
                }
            ),
        ):
            with patch.object(
                StaticAnalysisIntegrationTests,
                "generate_test_report",
                return_value="report",
            ):
                results = await run_integration_tests()

        assert results["summary"]["all_tests_passed"] is True


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — format_converter.py
# ═══════════════════════════════════════════════════════════════════════════


class TestFormatConverterInit:
    def test_default_output_dir(self):
        fc = FormatConverter()
        assert fc.output_dir.startswith(tempfile.gettempdir())

    def test_custom_output_dir(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))
        assert fc.output_dir == str(tmp_path)


class TestSaveDiagram:
    async def test_save_base64_diagram(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))
        content = base64.b64encode(b"<svg>hello</svg>").decode()

        path = await fc.save_diagram(content, "test.svg")

        assert path is not None
        assert os.path.exists(path)
        with open(path, "rb") as f:
            assert f.read() == b"<svg>hello</svg>"

    async def test_save_invalid_base64(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))

        path = await fc.save_diagram("not-valid-base64!!!", "bad.svg")
        assert path is None

    async def test_save_creates_dir(self):
        out_dir = os.path.join(tempfile.gettempdir(), "test_fc_save_" + str(time.time()))
        try:
            fc = FormatConverter(output_dir=out_dir)
            content = base64.b64encode(b"data").decode()
            path = await fc.save_diagram(content, "test.svg")
            assert path is not None
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


class TestGenerateArchitectureDiagrams:
    async def test_with_visual_output(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))

        call_graph_result = MagicMock()
        call_graph_result.visual_output = base64.b64encode(b"<svg/>").decode()

        with (
            patch.object(fc, "generate_dependency_diagram", new=AsyncMock(return_value=None)),
            patch.object(fc, "generate_structure_diagram", new=AsyncMock(return_value=None)),
        ):
            diagrams = await fc.generate_architecture_diagrams("/fake", call_graph_result)

        assert len(diagrams) >= 1  # At least the call graph diagram

    async def test_without_visual_output(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))

        call_graph_result = MagicMock(spec=[])  # no visual_output attr

        with (
            patch.object(fc, "generate_dependency_diagram", new=AsyncMock(return_value="/dep.svg")),
            patch.object(
                fc, "generate_structure_diagram", new=AsyncMock(return_value="/struct.svg")
            ),
        ):
            diagrams = await fc.generate_architecture_diagrams("/fake", call_graph_result)

        assert "/dep.svg" in diagrams
        assert "/struct.svg" in diagrams

    async def test_error_handling(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))

        call_graph_result = MagicMock(spec=[])

        with (
            patch.object(
                fc, "generate_dependency_diagram", new=AsyncMock(side_effect=OSError("fail"))
            ),
        ):
            diagrams = await fc.generate_architecture_diagrams("/fake", call_graph_result)
            assert isinstance(diagrams, list)


class TestGenerateDependencyDiagram:
    async def test_with_dependencies(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))

        python_deps = {"flask": {"version": "^2.0", "dev": False}}
        js_deps = {"express": {"version": "^4.18", "dev": True, "optional": False}}

        with (
            patch("processing.analyzers.code_analyzer.CodeAnalyzer") as mock_ca,
        ):
            inst = mock_ca.return_value
            inst.extract_python_dependencies = AsyncMock(return_value=python_deps)
            inst.extract_javascript_dependencies = AsyncMock(return_value=js_deps)

            result = await fc.generate_dependency_diagram(str(tmp_path))

        # Should produce a file path ending in .svg
        assert result is not None
        assert result.endswith(".svg")

    async def test_no_dependencies(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))

        with patch("processing.analyzers.code_analyzer.CodeAnalyzer") as mock_ca:
            inst = mock_ca.return_value
            inst.extract_python_dependencies = AsyncMock(return_value={})
            inst.extract_javascript_dependencies = AsyncMock(return_value={})

            result = await fc.generate_dependency_diagram(str(tmp_path))

        assert result is not None  # Still renders an empty graph

    async def test_more_than_10_deps(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))

        many_deps = {f"pkg{i}": {"version": "1.0", "dev": False} for i in range(15)}

        with patch("processing.analyzers.code_analyzer.CodeAnalyzer") as mock_ca:
            inst = mock_ca.return_value
            inst.extract_python_dependencies = AsyncMock(return_value=many_deps)
            inst.extract_javascript_dependencies = AsyncMock(return_value={})

            result = await fc.generate_dependency_diagram(str(tmp_path))

        assert result is not None

    async def test_optional_dep_coloring(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))

        deps = {"optpkg": {"version": "1.0", "optional": True}}

        with patch("processing.analyzers.code_analyzer.CodeAnalyzer") as mock_ca:
            inst = mock_ca.return_value
            inst.extract_python_dependencies = AsyncMock(return_value=deps)
            inst.extract_javascript_dependencies = AsyncMock(return_value={})

            result = await fc.generate_dependency_diagram(str(tmp_path))

        assert result is not None


class TestGenerateStructureDiagram:
    async def test_basic_structure(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))

        structure = {
            "directories": ["src", "tests", "docs", "lib"],
            "files": ["main.py", "test_main.py"],
            "file_types": {".py": 5, ".md": 2, ".js": 3, ".ts": 1, ".txt": 1},
        }

        with patch("processing.analyzers.code_analyzer.CodeAnalyzer") as mock_ca:
            mock_ca.return_value.analyze_project_structure.return_value = structure

            result = await fc.generate_structure_diagram(str(tmp_path))

        assert result is not None
        assert result.endswith(".svg")

    async def test_deep_directory_paths(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))

        structure = {
            "directories": [os.path.join("a", "b", "c", "d")],
            "files": [],
            "file_types": {},
        }

        with patch("processing.analyzers.code_analyzer.CodeAnalyzer") as mock_ca:
            mock_ca.return_value.analyze_project_structure.return_value = structure

            result = await fc.generate_structure_diagram(str(tmp_path))

        assert result is not None

    async def test_empty_structure(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))

        structure = {"directories": [], "files": [], "file_types": {}}

        with patch("processing.analyzers.code_analyzer.CodeAnalyzer") as mock_ca:
            mock_ca.return_value.analyze_project_structure.return_value = structure

            result = await fc.generate_structure_diagram(str(tmp_path))

        assert result is not None


class TestConvertMarkdownToHtml:
    async def test_with_markdown_lib(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))

        result = await fc.convert_markdown_to_html("# Hello\nWorld")
        assert "<p>" in result or "<h1>" in result

    async def test_fallback_basic_conversion(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))

        # Force ImportError for markdown by setting its sys.modules entry to None
        with patch.dict(sys.modules, {"markdown": None}):
            # The method catches the ImportError and falls back
            result = await fc.convert_markdown_to_html("# Hello\n**bold**")

        # Whether it uses the library or fallback, should return some HTML
        assert isinstance(result, str)


class TestBasicMarkdownToHtml:
    def test_headers(self):
        fc = FormatConverter()
        result = fc._basic_markdown_to_html("# H1\n## H2\n### H3")
        assert "<h1>" in result
        assert "<h2>" in result
        assert "<h3>" in result

    def test_bold_and_italic(self):
        fc = FormatConverter()
        result = fc._basic_markdown_to_html("**bold** and *italic*")
        assert "<strong>bold</strong>" in result
        assert "<em>italic</em>" in result

    def test_inline_code(self):
        fc = FormatConverter()
        result = fc._basic_markdown_to_html("Use `code` here")
        assert "<code>code</code>" in result

    def test_links(self):
        fc = FormatConverter()
        result = fc._basic_markdown_to_html("[click](http://example.com)")
        assert '<a href="http://example.com">click</a>' in result

    def test_paragraphs(self):
        fc = FormatConverter()
        result = fc._basic_markdown_to_html("para1\n\npara2")
        assert "</p><p>" in result


class TestSaveDocumentationFiles:
    async def test_saves_json_md_html(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))
        doc = {"overview": {"content": "Hello world"}, "api": "Endpoints"}

        saved = await fc.save_documentation_files(doc, "test_doc")

        assert len(saved) == 3
        # Check file extensions
        extensions = {os.path.splitext(f)[1] for f in saved}
        assert {".json", ".md", ".html"} == extensions

        # Verify JSON is valid
        json_file = [f for f in saved if f.endswith(".json")][0]
        with open(json_file) as f:
            data = json.load(f)
        assert data == doc

    async def test_error_handling(self):
        fc = FormatConverter(output_dir="/nonexistent/dir/xyz")
        saved = await fc.save_documentation_files({"key": "val"})
        # Should return whatever was saved before error (could be empty)
        assert isinstance(saved, list)


class TestConvertToMarkdown:
    async def test_dict_sections(self):
        fc = FormatConverter()
        doc = {
            "overview": {"title": "My Project", "description": "A description"},
            "details": "Just a string",
            "metrics": 42,
        }
        md = await fc._convert_to_markdown(doc)

        assert "# Project Documentation" in md
        assert "Overview" in md
        assert "My Project" in md
        assert "Just a string" in md
        assert "42" in md

    async def test_dict_with_content_key(self):
        fc = FormatConverter()
        doc = {"section": {"content": "The content here"}}
        md = await fc._convert_to_markdown(doc)
        assert "The content here" in md

    async def test_dict_with_list_value(self):
        fc = FormatConverter()
        doc = {"section": {"items": ["a", "b", "c"]}}
        md = await fc._convert_to_markdown(doc)
        assert "```json" in md

    async def test_empty_doc(self):
        fc = FormatConverter()
        md = await fc._convert_to_markdown({})
        assert "# Project Documentation" in md


class TestWrapHtml:
    def test_basic_wrap(self):
        fc = FormatConverter()
        html = fc._wrap_html("<p>Hello</p>", "Test Page")

        assert "<!DOCTYPE html>" in html
        assert "<title>Test Page</title>" in html
        assert "<p>Hello</p>" in html
        assert "font-family" in html


class TestGeneratePdfOutput:
    async def test_with_weasyprint(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))

        result = await fc.generate_pdf_output("<p>Hello</p>", "test")

        assert result is not None
        assert result.endswith(".pdf")
        assert os.path.exists(result)

    async def test_without_weasyprint(self, tmp_path):
        fc = FormatConverter(output_dir=str(tmp_path))

        with patch.dict(sys.modules, {"weasyprint": None}):
            with patch("builtins.__import__", side_effect=ImportError("no weasyprint")):
                result = await fc.generate_pdf_output("<p>Hello</p>", "test")

        # Whether it works or not depends on stub; just ensure no crash
        assert result is None or isinstance(result, str)


class TestCleanupTempFiles:
    def test_cleanup_temp_dir(self):
        temp_dir = tempfile.mkdtemp(prefix="docs_output_")
        # Create a file in it
        with open(os.path.join(temp_dir, "test.txt"), "w") as f:
            f.write("data")

        fc = FormatConverter(output_dir=temp_dir)
        fc.cleanup_temp_files()

        assert not os.path.exists(temp_dir)

    def test_cleanup_non_temp_dir(self):
        # Use a path that is NOT under tempfile.gettempdir()
        # On Linux /tmp, on macOS /var/folders — ensure we test outside it
        non_temp = os.path.join(os.path.dirname(__file__), "_test_non_temp_cleanup")
        os.makedirs(non_temp, exist_ok=True)
        try:
            fc = FormatConverter(output_dir=non_temp)
            fc.cleanup_temp_files()
            # Should NOT delete directories outside tempdir
            assert os.path.exists(non_temp)
        finally:
            os.rmdir(non_temp)

    def test_cleanup_error_handling(self):
        fc = FormatConverter(output_dir="/nonexistent/path")
        # Should not raise
        fc.cleanup_temp_files()
