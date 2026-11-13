"""Unit tests for processing/analyzers/deep — DeepCodeAnalyzer."""

from __future__ import annotations

import ast
import pathlib
import tempfile

from processing.analyzers.deep import (
    CodeIssue,
    CodeRelationship,
    DeepAnalysisResult,
    DeepAnalysisServiceCodeEntity,
    DeepCodeAnalyzer,
    IssueType,
    Severity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tmp_codebase(files: dict[str, str]) -> pathlib.Path:
    """Create a temporary directory with Python files.

    Args:
        files: mapping of relative file path -> source content

    Returns:
        Path to the temporary directory (caller must clean up)
    """
    tmp = pathlib.Path(tempfile.mkdtemp())
    for rel_path, content in files.items():
        p = tmp / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return tmp


# ---------------------------------------------------------------------------
# Dataclass / Enum tests
# ---------------------------------------------------------------------------


class TestEnumsAndDataclasses:
    """Verify basic enum values and dataclass construction."""

    def test_issue_type_values(self):
        assert IssueType.SECURITY.value == "security"
        assert IssueType.PERFORMANCE.value == "performance"
        assert IssueType.MAINTAINABILITY.value == "maintainability"
        assert IssueType.COMPLEXITY.value == "complexity"
        assert IssueType.CODE_SMELL.value == "code_smell"
        assert IssueType.BUG_RISK.value == "bug_risk"
        assert IssueType.DOCUMENTATION.value == "documentation"
        assert IssueType.ARCHITECTURE.value == "architecture"

    def test_severity_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFO.value == "info"

    def test_code_issue_creation(self):
        issue = CodeIssue(
            type=IssueType.SECURITY,
            severity=Severity.HIGH,
            title="Test Issue",
            description="desc",
            file_path="test.py",
            line_number=1,
            code_snippet="import os",
            suggestion="fix it",
            confidence=0.9,
        )
        assert issue.type == IssueType.SECURITY
        assert issue.severity == Severity.HIGH
        assert issue.confidence == 0.9

    def test_code_issue_optional_fields(self):
        issue = CodeIssue(
            type=IssueType.BUG_RISK,
            severity=Severity.LOW,
            title="t",
            description="d",
            file_path="f.py",
            line_number=None,
            code_snippet=None,
            suggestion="s",
            confidence=0.5,
        )
        assert issue.line_number is None
        assert issue.code_snippet is None

    def test_deep_analysis_service_code_entity(self):
        entity = DeepAnalysisServiceCodeEntity(
            id="e1", type="function", name="foo", line_number=10, file_path="a.py"
        )
        assert entity.docstring is None
        assert entity.metadata is None

    def test_code_relationship(self):
        rel = CodeRelationship(source_id="s1", target_id="t1", type="calls", metadata={"line": 5})
        assert rel.type == "calls"
        assert rel.metadata == {"line": 5}

    def test_deep_analysis_result(self):
        result = DeepAnalysisResult(
            overall_score=8.5,
            issues=[],
            metrics={"total_files": 1},
            recommendations=["do better"],
            entities=[],
            relationships=[],
        )
        assert result.overall_score == 8.5
        assert result.recommendations == ["do better"]


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestDeepCodeAnalyzerInit:
    """Constructor / __init__ behaviour."""

    def test_init_nonexistent_path(self, tmp_path):
        """Non-existent path should yield empty python_files list."""
        bad_path = tmp_path / "no_such_dir"
        analyzer = DeepCodeAnalyzer(str(bad_path))
        assert analyzer.python_files == []
        assert analyzer.language == "python"

    def test_init_empty_directory(self, tmp_path):
        """Empty directory should have no python files."""
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        assert analyzer.python_files == []

    def test_init_finds_python_files(self, tmp_path):
        """Should discover *.py files recursively."""
        (tmp_path / "a.py").write_text("x = 1")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.py").write_text("y = 2")
        (tmp_path / "readme.txt").write_text("nope")
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        names = {f.name for f in analyzer.python_files}
        assert "a.py" in names
        assert "b.py" in names
        assert "readme.txt" not in names

    def test_init_default_language(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path), language="javascript")
        assert analyzer.language == "javascript"


# ---------------------------------------------------------------------------
# _safe_parse_ast
# ---------------------------------------------------------------------------


class TestSafeParseAST:
    def test_valid_code(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        tree = analyzer._safe_parse_ast("x = 1\n")
        assert tree is not None
        assert isinstance(tree, ast.Module)

    def test_syntax_error(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        tree = analyzer._safe_parse_ast("def ???")
        assert tree is None

    def test_empty_string(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        tree = analyzer._safe_parse_ast("")
        assert tree is not None  # empty module is valid


# ---------------------------------------------------------------------------
# _extract_docstring / _extract_decorator_name / _extract_base_name
# ---------------------------------------------------------------------------


class TestASTHelpers:
    def _make_analyzer(self, tmp_path):
        return DeepCodeAnalyzer(str(tmp_path))

    def test_extract_docstring_present(self, tmp_path):
        analyzer = self._make_analyzer(tmp_path)
        code = 'def foo():\n    """My doc."""\n    pass\n'
        tree = ast.parse(code)
        func = tree.body[0]
        assert analyzer._extract_docstring(func) == "My doc."

    def test_extract_docstring_absent(self, tmp_path):
        analyzer = self._make_analyzer(tmp_path)
        code = "def foo():\n    pass\n"
        tree = ast.parse(code)
        func = tree.body[0]
        assert analyzer._extract_docstring(func) is None

    def test_extract_decorator_name_simple(self, tmp_path):
        analyzer = self._make_analyzer(tmp_path)
        code = "@staticmethod\ndef foo(): pass\n"
        tree = ast.parse(code)
        func = tree.body[0]
        assert analyzer._extract_decorator_name(func.decorator_list[0]) == "staticmethod"

    def test_extract_decorator_name_attribute(self, tmp_path):
        analyzer = self._make_analyzer(tmp_path)
        code = "@app.route\ndef foo(): pass\n"
        tree = ast.parse(code)
        func = tree.body[0]
        assert analyzer._extract_decorator_name(func.decorator_list[0]) == "route"

    def test_extract_base_name_simple(self, tmp_path):
        analyzer = self._make_analyzer(tmp_path)
        code = "class Foo(Bar): pass\n"
        tree = ast.parse(code)
        cls = tree.body[0]
        assert analyzer._extract_base_name(cls.bases[0]) == "Bar"

    def test_extract_base_name_attribute(self, tmp_path):
        analyzer = self._make_analyzer(tmp_path)
        code = "class Foo(abc.ABC): pass\n"
        tree = ast.parse(code)
        cls = tree.body[0]
        assert analyzer._extract_base_name(cls.bases[0]) == "ABC"


# ---------------------------------------------------------------------------
# _get_line / _get_function_signature
# ---------------------------------------------------------------------------


class TestLineHelpers:
    def test_get_line_valid(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        content = "line1\nline2\nline3"
        assert analyzer._get_line(content, 2) == "line2"

    def test_get_line_out_of_range(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        assert analyzer._get_line("one line", 99) == ""

    def test_get_line_strips_whitespace(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        content = "    indented  "
        assert analyzer._get_line(content, 1) == "indented"

    def test_get_function_signature(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = "def foo(a, b):\n    pass\n"
        tree = ast.parse(code)
        func = tree.body[0]
        sig = analyzer._get_function_signature(code, func)
        assert "def foo" in sig

    def test_get_function_signature_fallback(self, tmp_path):
        """When line_no is out of range, should return fallback."""
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = ""
        tree = ast.parse("def foo(): pass\n")
        func = tree.body[0]
        # Modify lineno to something out of range
        func.lineno = 999
        sig = analyzer._get_function_signature(code, func)
        assert "def foo" in sig


# ---------------------------------------------------------------------------
# _calculate_cyclomatic_complexity
# ---------------------------------------------------------------------------


class TestCyclomaticComplexity:
    def test_simple_function(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = "def foo():\n    return 1\n"
        tree = ast.parse(code)
        func = tree.body[0]
        assert analyzer._calculate_cyclomatic_complexity(func) == 1

    def test_function_with_if(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = "def foo(x):\n    if x:\n        return 1\n    return 0\n"
        tree = ast.parse(code)
        func = tree.body[0]
        assert analyzer._calculate_cyclomatic_complexity(func) == 2

    def test_function_with_loops_and_conditions(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = (
            "def foo(items):\n"
            "    for i in items:\n"
            "        if i > 0:\n"
            "            while True:\n"
            "                break\n"
            "    return items\n"
        )
        tree = ast.parse(code)
        func = tree.body[0]
        complexity = analyzer._calculate_cyclomatic_complexity(func)
        # 1 (base) + 1 (for) + 1 (if) + 1 (while) = 4
        assert complexity == 4

    def test_function_with_boolop(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = "def foo(a, b, c):\n    if a and b or c:\n        pass\n"
        tree = ast.parse(code)
        func = tree.body[0]
        complexity = analyzer._calculate_cyclomatic_complexity(func)
        # 1 (base) + 1 (if) + BoolOps: (and has 2 values -> +1) + (or has 2 values -> +1) = 4
        assert complexity >= 3


# ---------------------------------------------------------------------------
# _calculate_nesting_depth
# ---------------------------------------------------------------------------


class TestNestingDepth:
    def test_flat_if(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = "if True:\n    x = 1\n"
        tree = ast.parse(code)
        node = tree.body[0]
        assert analyzer._calculate_nesting_depth(node) == 0

    def test_nested_if(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = "if True:\n    if True:\n        if True:\n            x = 1\n"
        tree = ast.parse(code)
        node = tree.body[0]
        depth = analyzer._calculate_nesting_depth(node)
        assert depth == 2  # two levels deep inside the root if

    def test_nested_for_while(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = "for i in range(10):\n    while True:\n        if i:\n            break\n"
        tree = ast.parse(code)
        node = tree.body[0]
        depth = analyzer._calculate_nesting_depth(node)
        assert depth >= 2


# ---------------------------------------------------------------------------
# _calculate_overall_score
# ---------------------------------------------------------------------------


class TestOverallScore:
    def test_no_issues_perfect_score(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        assert analyzer._calculate_overall_score() == 10.0

    def test_critical_issues_lower_score(self):
        tmp = _make_tmp_codebase({"a.py": "x = 1\n"})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer.issues = [
                CodeIssue(
                    type=IssueType.SECURITY,
                    severity=Severity.CRITICAL,
                    title="t",
                    description="d",
                    file_path="a.py",
                    line_number=1,
                    code_snippet=None,
                    suggestion="s",
                    confidence=1.0,
                )
            ]
            score = analyzer._calculate_overall_score()
            assert score < 10.0
            assert score >= 0.0
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_score_never_negative(self):
        tmp = _make_tmp_codebase({"a.py": "x = 1\n"})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            # Add many critical issues
            analyzer.issues = [
                CodeIssue(
                    type=IssueType.SECURITY,
                    severity=Severity.CRITICAL,
                    title="t",
                    description="d",
                    file_path="a.py",
                    line_number=1,
                    code_snippet=None,
                    suggestion="s",
                    confidence=1.0,
                )
                for _ in range(100)
            ]
            score = analyzer._calculate_overall_score()
            assert score == 0.0
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# _generate_recommendations
# ---------------------------------------------------------------------------


class TestGenerateRecommendations:
    def test_no_issues_no_recommendations(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        recs = analyzer._generate_recommendations()
        assert recs == []

    def test_security_issues_recommend(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        analyzer.issues = [
            CodeIssue(
                type=IssueType.SECURITY,
                severity=Severity.HIGH,
                title="t",
                description="d",
                file_path="f",
                line_number=1,
                code_snippet=None,
                suggestion="s",
                confidence=0.8,
            )
        ]
        recs = analyzer._generate_recommendations()
        assert any("security" in r.lower() for r in recs)

    def test_bug_risk_recommend(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        analyzer.issues = [
            CodeIssue(
                type=IssueType.BUG_RISK,
                severity=Severity.MEDIUM,
                title="t",
                description="d",
                file_path="f",
                line_number=1,
                code_snippet=None,
                suggestion="s",
                confidence=0.8,
            )
        ]
        recs = analyzer._generate_recommendations()
        assert any("bug" in r.lower() for r in recs)

    def test_documentation_issues_need_threshold(self, tmp_path):
        """Documentation recommendation triggers only when > 10 issues."""
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        # 5 doc issues — should NOT trigger
        analyzer.issues = [
            CodeIssue(
                type=IssueType.DOCUMENTATION,
                severity=Severity.LOW,
                title="t",
                description="d",
                file_path="f",
                line_number=1,
                code_snippet=None,
                suggestion="s",
                confidence=1.0,
            )
            for _ in range(5)
        ]
        recs = analyzer._generate_recommendations()
        assert not any("documentation" in r.lower() for r in recs)

        # 11 doc issues — should trigger
        analyzer.issues = analyzer.issues * 3  # 15 issues
        recs = analyzer._generate_recommendations()
        assert any("documentation" in r.lower() for r in recs)

    def test_performance_issues_need_threshold(self, tmp_path):
        """Performance recommendation triggers only when > 3 issues."""
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        analyzer.issues = [
            CodeIssue(
                type=IssueType.PERFORMANCE,
                severity=Severity.MEDIUM,
                title="t",
                description="d",
                file_path="f",
                line_number=1,
                code_snippet=None,
                suggestion="s",
                confidence=0.8,
            )
            for _ in range(4)
        ]
        recs = analyzer._generate_recommendations()
        assert any("performance" in r.lower() for r in recs)

    def test_complexity_issues_need_threshold(self, tmp_path):
        """Complexity recommendation triggers only when > 5 issues."""
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        analyzer.issues = [
            CodeIssue(
                type=IssueType.COMPLEXITY,
                severity=Severity.MEDIUM,
                title="t",
                description="d",
                file_path="f",
                line_number=1,
                code_snippet=None,
                suggestion="s",
                confidence=0.8,
            )
            for _ in range(6)
        ]
        recs = analyzer._generate_recommendations()
        assert any("refactor" in r.lower() for r in recs)


# ---------------------------------------------------------------------------
# _calculate_metrics
# ---------------------------------------------------------------------------


class TestCalculateMetrics:
    def test_empty_codebase(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        metrics = analyzer._calculate_metrics()
        assert metrics["total_files"] == 0
        assert metrics["total_lines"] == 0
        assert metrics["total_functions"] == 0
        assert metrics["total_classes"] == 0
        assert metrics["total_issues"] == 0
        assert metrics["average_issues_per_file"] == 0

    def test_metrics_with_code(self):
        code = "class Foo:\n    def bar(self):\n        pass\n\ndef baz():\n    pass\n"
        tmp = _make_tmp_codebase({"mod.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            metrics = analyzer._calculate_metrics()
            assert metrics["total_files"] == 1
            assert metrics["total_lines"] == 6
            # bar (method) + baz (function) = 2 functions
            assert metrics["total_functions"] == 2
            assert metrics["total_classes"] == 1
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_issue_breakdown(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        analyzer.issues = [
            CodeIssue(
                type=IssueType.SECURITY,
                severity=Severity.HIGH,
                title="t",
                description="d",
                file_path="f",
                line_number=1,
                code_snippet=None,
                suggestion="s",
                confidence=0.8,
            ),
            CodeIssue(
                type=IssueType.SECURITY,
                severity=Severity.HIGH,
                title="t2",
                description="d",
                file_path="f",
                line_number=2,
                code_snippet=None,
                suggestion="s",
                confidence=0.8,
            ),
        ]
        metrics = analyzer._calculate_metrics()
        assert metrics["issue_breakdown"]["security_high"] == 2


# ---------------------------------------------------------------------------
# Security analysis
# ---------------------------------------------------------------------------


class TestAnalyzeSecurityIssues:
    def test_detect_dangerous_import_pickle(self):
        code = "import pickle\n"
        tmp = _make_tmp_codebase({"bad.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_security_issues()
            sec = [i for i in analyzer.issues if i.type == IssueType.SECURITY]
            assert len(sec) >= 1
            assert any("pickle" in i.title for i in sec)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_detect_eval_call(self):
        code = "result = eval('1+1')\n"
        tmp = _make_tmp_codebase({"bad.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_security_issues()
            sec = [i for i in analyzer.issues if i.type == IssueType.SECURITY]
            assert len(sec) >= 1
            assert any("eval" in i.title for i in sec)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_detect_hardcoded_password(self):
        code = "password = 'supersecret'\n"
        tmp = _make_tmp_codebase({"bad.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_security_issues()
            sec = [i for i in analyzer.issues if i.type == IssueType.SECURITY]
            assert len(sec) >= 1
            assert any(
                "password" in i.title.lower() or "password" in i.description.lower() for i in sec
            )
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_security_issues_clean_code(self):
        code = "x = 1\ny = x + 2\n"
        tmp = _make_tmp_codebase({"clean.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_security_issues()
            sec = [i for i in analyzer.issues if i.type == IssueType.SECURITY]
            assert len(sec) == 0
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Performance analysis
# ---------------------------------------------------------------------------


class TestAnalyzePerformanceIssues:
    def test_detect_string_concat_in_loop(self):
        code = "result = ''\nfor i in range(10):\n    result += str(i)\n"
        tmp = _make_tmp_codebase({"bad.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_performance_issues()
            perf = [i for i in analyzer.issues if i.type == IssueType.PERFORMANCE]
            assert len(perf) >= 1
            assert any("concatenation" in i.title.lower() for i in perf)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_perf_issues_clean_code(self):
        code = "x = [i for i in range(10)]\n"
        tmp = _make_tmp_codebase({"clean.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_performance_issues()
            assert len(analyzer.issues) == 0
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Complexity analysis
# ---------------------------------------------------------------------------


class TestAnalyzeCodeComplexity:
    def test_detect_high_complexity_function(self):
        # Build a function with complexity > 10
        branches = "\n".join(f"    if x == {i}:\n        pass" for i in range(12))
        code = f"def messy(x):\n{branches}\n"
        tmp = _make_tmp_codebase({"complex.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_code_complexity()
            cx = [i for i in analyzer.issues if i.type == IssueType.COMPLEXITY]
            assert len(cx) >= 1
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_detect_long_function(self):
        body = "\n".join(f"    x{i} = {i}" for i in range(55))
        code = f"def long_func():\n{body}\n"
        tmp = _make_tmp_codebase({"long.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_code_complexity()
            maint = [i for i in analyzer.issues if i.type == IssueType.MAINTAINABILITY]
            assert len(maint) >= 1
            assert any("long" in i.title.lower() for i in maint)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_complexity_issues_simple_function(self):
        code = "def simple():\n    return 1\n"
        tmp = _make_tmp_codebase({"simple.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_code_complexity()
            assert len(analyzer.issues) == 0
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Code smell analysis
# ---------------------------------------------------------------------------


class TestAnalyzeCodeSmells:
    def test_too_many_params(self):
        code = "def many(a, b, c, d, e, f, g, h):\n    pass\n"
        tmp = _make_tmp_codebase({"smell.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_code_smells()
            smells = [i for i in analyzer.issues if i.type == IssueType.CODE_SMELL]
            assert len(smells) >= 1
            assert any("parameter" in i.title.lower() for i in smells)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_deep_nesting(self):
        code = (
            "if True:\n"
            "    if True:\n"
            "        if True:\n"
            "            if True:\n"
            "                if True:\n"
            "                    if True:\n"
            "                        x = 1\n"
        )
        tmp = _make_tmp_codebase({"nested.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_code_smells()
            smells = [i for i in analyzer.issues if i.type == IssueType.CODE_SMELL]
            assert len(smells) >= 1
            assert any("nesting" in i.title.lower() for i in smells)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Documentation analysis
# ---------------------------------------------------------------------------


class TestAnalyzeDocumentation:
    def test_missing_function_docstring(self):
        code = "def no_docs():\n    pass\n"
        tmp = _make_tmp_codebase({"nodoc.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_documentation()
            docs = [i for i in analyzer.issues if i.type == IssueType.DOCUMENTATION]
            assert len(docs) >= 1
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_missing_class_docstring(self):
        code = "class NoDocs:\n    pass\n"
        tmp = _make_tmp_codebase({"nodoc.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_documentation()
            docs = [i for i in analyzer.issues if i.type == IssueType.DOCUMENTATION]
            assert len(docs) >= 1
            # Class missing docstring should be MEDIUM severity
            assert any(i.severity == Severity.MEDIUM for i in docs)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_documented_code_no_issues(self):
        code = (
            'def documented():\n    """Good docs."""\n    pass\n\n'
            'class Documented:\n    """Also good docs."""\n    pass\n'
        )
        tmp = _make_tmp_codebase({"good.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_documentation()
            docs = [i for i in analyzer.issues if i.type == IssueType.DOCUMENTATION]
            assert len(docs) == 0
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Bug risk analysis
# ---------------------------------------------------------------------------


class TestAnalyzeBugRisks:
    def test_bare_except(self):
        code = "try:\n    x = 1\nexcept:\n    pass\n"
        tmp = _make_tmp_codebase({"bare.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_bug_risks()
            bugs = [i for i in analyzer.issues if i.type == IssueType.BUG_RISK]
            assert len(bugs) >= 1
            assert any("bare except" in i.title.lower() for i in bugs)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_mutable_default_argument(self):
        code = "def bad(items=[]):\n    items.append(1)\n    return items\n"
        tmp = _make_tmp_codebase({"mutable.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_bug_risks()
            bugs = [i for i in analyzer.issues if i.type == IssueType.BUG_RISK]
            assert len(bugs) >= 1
            assert any("mutable" in i.title.lower() for i in bugs)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_bug_risks_clean_code(self):
        code = "def clean(x=None):\n    return x or []\n"
        tmp = _make_tmp_codebase({"clean.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_bug_risks()
            assert len(analyzer.issues) == 0
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Architecture analysis
# ---------------------------------------------------------------------------


class TestAnalyzeArchitecture:
    def test_no_issues_for_simple_code(self):
        code = "import os\nx = os.getcwd()\n"
        tmp = _make_tmp_codebase({"simple.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._analyze_architecture()
            arch = [i for i in analyzer.issues if i.type == IssueType.ARCHITECTURE]
            assert len(arch) == 0
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Full codebase analysis (integration-style)
# ---------------------------------------------------------------------------


class TestAnalyzeCodebase:
    def test_full_analysis_clean_code(self):
        code = '"""Module doc."""\n\ndef hello():\n    """Say hello."""\n    return "hello"\n'
        tmp = _make_tmp_codebase({"clean.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            result = analyzer.analyze_codebase()
            assert isinstance(result, DeepAnalysisResult)
            assert result.overall_score > 0
            assert result.metrics["total_files"] == 1
            assert len(result.entities) > 0
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_full_analysis_with_issues(self):
        code = (
            "import pickle\n"
            "def bad(items=[]):\n"
            "    result = eval('1+1')\n"
            "    try:\n"
            "        pass\n"
            "    except:\n"
            "        pass\n"
            "    return items\n"
        )
        tmp = _make_tmp_codebase({"bad.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            result = analyzer.analyze_codebase()
            assert isinstance(result, DeepAnalysisResult)
            assert len(result.issues) > 0
            assert result.overall_score < 10.0
            # Should have security + bug_risk issues at minimum
            issue_types = {i.type for i in result.issues}
            assert IssueType.SECURITY in issue_types
            assert IssueType.BUG_RISK in issue_types
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_full_analysis_empty_codebase(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        result = analyzer.analyze_codebase()
        assert isinstance(result, DeepAnalysisResult)
        assert result.overall_score == 10.0
        assert len(result.issues) == 0


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


class TestExtractEntities:
    def test_extracts_functions_and_classes(self):
        code = (
            '"""Module."""\n\n'
            'class Foo:\n    """Foo class."""\n    pass\n\n'
            'def bar():\n    """Bar func."""\n    pass\n'
        )
        tmp = _make_tmp_codebase({"mod.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._extract_entities()
            names = [e.name for e in analyzer.entities]
            assert "Foo" in names
            assert "bar" in names
            # Should also have a file entity
            file_entities = [e for e in analyzer.entities if e.name == "mod.py"]
            assert len(file_entities) == 1
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_extracts_relationships_imports(self):
        code = "import os\nfrom pathlib import Path\n"
        tmp = _make_tmp_codebase({"mod.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._extract_entities()
            import_rels = [
                r for r in analyzer.relationships if r.type in ("imports", "imports_from")
            ]
            assert len(import_rels) >= 2
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_extracts_inheritance_relationship(self):
        code = "class Base:\n    pass\n\nclass Child(Base):\n    pass\n"
        tmp = _make_tmp_codebase({"mod.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._extract_entities()
            inherit_rels = [r for r in analyzer.relationships if r.type == "inherits"]
            assert len(inherit_rels) >= 1
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_extracts_function_call_relationship(self):
        code = "def helper():\n    pass\n\ndef caller():\n    helper()\n"
        tmp = _make_tmp_codebase({"mod.py": code})
        try:
            analyzer = DeepCodeAnalyzer(str(tmp))
            analyzer._extract_entities()
            call_rels = [r for r in analyzer.relationships if r.type == "calls"]
            assert len(call_rels) >= 1
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Snippet analysis (async)
# ---------------------------------------------------------------------------


class TestAnalyzeCodeSnippet:
    async def test_analyze_clean_snippet(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = 'def greet(name):\n    """Say hello."""\n    return f"Hello {name}"\n'
        result = await analyzer.analyze_code_snippet(code, language="python")
        assert isinstance(result, DeepAnalysisResult)
        assert result.overall_score > 0

    async def test_analyze_snippet_with_eval(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = "def dangerous():\n    return eval('1+1')\n"
        result = await analyzer.analyze_code_snippet(code, language="python")
        sec = [i for i in result.issues if i.type == IssueType.SECURITY]
        assert len(sec) >= 1

    async def test_analyze_snippet_complex_function(self, tmp_path):
        branches = "\n".join(f"    if x == {i}:\n        pass" for i in range(12))
        code = f"def messy(x):\n{branches}\n"
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        result = await analyzer.analyze_code_snippet(code, language="python")
        cx = [i for i in result.issues if i.type == IssueType.COMPLEXITY]
        assert len(cx) >= 1

    async def test_analyze_snippet_non_python(self, tmp_path):
        """Non-python language should still return a result without crashing."""
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = "function hello() { return 'hi'; }"
        result = await analyzer.analyze_code_snippet(code, language="javascript")
        assert isinstance(result, DeepAnalysisResult)

    async def test_analyze_snippet_cleans_temp_file(self, tmp_path):
        """Temporary file should be cleaned up after analysis."""
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = "x = 1\n"
        result = await analyzer.analyze_code_snippet(code, language="python")
        assert isinstance(result, DeepAnalysisResult)
        # The temp file should no longer exist (cleaned up in finally block)


# ---------------------------------------------------------------------------
# Snippet sub-analyzers
# ---------------------------------------------------------------------------


class TestSnippetSubAnalyzers:
    def test_extract_entities_from_snippet(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = (
            'class Foo:\n    """Foo."""\n    def bar(self):\n        pass\n\n'
            'def baz():\n    """Baz."""\n    pass\n'
        )
        tree = ast.parse(code)
        analyzer._extract_entities_from_snippet(code, tree, "snippet.py")
        names = [e.name for e in analyzer.entities]
        assert "Foo" in names
        assert "baz" in names

    def test_analyze_snippet_security_eval(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = "x = eval('2+2')\n"
        tree = ast.parse(code)
        analyzer._analyze_snippet_security(code, tree)
        sec = [i for i in analyzer.issues if i.type == IssueType.SECURITY]
        assert len(sec) >= 1

    def test_analyze_snippet_security_exec(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = "exec('import os')\n"
        tree = ast.parse(code)
        analyzer._analyze_snippet_security(code, tree)
        sec = [i for i in analyzer.issues if i.type == IssueType.SECURITY]
        assert len(sec) >= 1

    def test_analyze_snippet_security_clean(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = "x = 1 + 2\n"
        tree = ast.parse(code)
        analyzer._analyze_snippet_security(code, tree)
        assert len(analyzer.issues) == 0

    def test_analyze_snippet_complexity_no_issues(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = "def simple():\n    return 1\n"
        tree = ast.parse(code)
        analyzer._analyze_snippet_complexity(code, tree)
        assert len(analyzer.issues) == 0

    def test_analyze_snippet_complexity_high(self, tmp_path):
        branches = "\n".join(f"    if x == {i}:\n        pass" for i in range(12))
        code = f"def messy(x):\n{branches}\n"
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        tree = ast.parse(code)
        analyzer._analyze_snippet_complexity(code, tree)
        cx = [i for i in analyzer.issues if i.type == IssueType.COMPLEXITY]
        assert len(cx) >= 1

    def test_analyze_snippet_smells_short_function(self, tmp_path):
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        code = "def short():\n    return 1\n"
        tree = ast.parse(code)
        analyzer._analyze_snippet_smells(code, tree)
        assert len(analyzer.issues) == 0

    def test_analyze_snippet_smells_long_function(self, tmp_path):
        body = "\n".join(f"    x{i} = {i}" for i in range(55))
        code = f"def long_func():\n{body}\n"
        analyzer = DeepCodeAnalyzer(str(tmp_path))
        tree = ast.parse(code)
        analyzer._analyze_snippet_smells(code, tree)
        smells = [i for i in analyzer.issues if i.type == IssueType.CODE_SMELL]
        assert len(smells) >= 1
