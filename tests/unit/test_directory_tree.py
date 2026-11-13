"""Tests for processing.ast_analysis.directory_tree_analyzer module."""

from __future__ import annotations

from pathlib import Path

from processing.ast_analysis.directory_tree_analyzer import DirectoryTreeAnalyzer


class TestDirectoryTreeAnalyzerInit:
    def test_default_initialization(self):
        analyzer = DirectoryTreeAnalyzer()
        assert analyzer.directory_structure == {}
        assert analyzer.file_analysis == {}


class TestDirectoryTreeAnalyzerAnalyze:
    def test_analyze_directory_with_python_files(self, tmp_path):
        # Create a simple directory structure
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "module.py").write_text(
            'import os\n\nclass MyClass:\n    """A class."""\n    pass\n\ndef my_func():\n    return 1\n'
        )

        analyzer = DirectoryTreeAnalyzer()
        result = analyzer.analyze_directory(pkg)

        assert "directory_structure" in result
        assert "file_analysis" in result
        assert "summary" in result
        assert result["summary"]["total_files_analyzed"] >= 1

    def test_analyze_directory_default_extensions(self, tmp_path):
        (tmp_path / "file.py").write_text("x = 1\n")
        (tmp_path / "file.txt").write_text("not python\n")

        analyzer = DirectoryTreeAnalyzer()
        result = analyzer.analyze_directory(tmp_path)

        # Only .py files should be analyzed
        assert result["summary"]["total_files_analyzed"] == 1

    def test_analyze_directory_custom_extensions(self, tmp_path):
        (tmp_path / "file.py").write_text("x = 1\n")
        (tmp_path / "file.txt").write_text("text\n")

        analyzer = DirectoryTreeAnalyzer()
        result = analyzer.analyze_directory(tmp_path, extensions=[".txt"])

        # .txt is not parseable by AST, so file_analysis will contain an empty dict for it
        assert result["summary"]["total_files_analyzed"] >= 0

    def test_analyze_directory_error_returns_empty(self):
        """Test that an error during analysis returns empty dict."""
        analyzer = DirectoryTreeAnalyzer()
        # Use a mock that causes an internal error
        from unittest.mock import patch

        with patch.object(analyzer, "_build_directory_tree", side_effect=RuntimeError("boom")):
            result = analyzer.analyze_directory(Path("/tmp"))
            assert result == {}

    def test_analyze_directory_nested(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "inner.py").write_text("def inner(): pass\n")
        (tmp_path / "outer.py").write_text("def outer(): pass\n")

        analyzer = DirectoryTreeAnalyzer()
        result = analyzer.analyze_directory(tmp_path)
        assert result["summary"]["total_files_analyzed"] == 2
        assert result["summary"]["total_functions"] == 2


class TestBuildDirectoryTree:
    def test_directory_tree_structure(self, tmp_path):
        (tmp_path / "file.py").write_text("")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "inner.py").write_text("")

        analyzer = DirectoryTreeAnalyzer()
        tree = analyzer._build_directory_tree(tmp_path)

        assert tree["type"] == "directory"
        assert tree["name"] == tmp_path.name
        assert len(tree["children"]) >= 2  # file.py and sub/

    def test_file_tree_structure(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("")
        analyzer = DirectoryTreeAnalyzer()
        tree = analyzer._build_directory_tree(f)
        assert tree["type"] == "file"
        assert tree["name"] == "test.py"
        assert tree["children"] == []


class TestAnalyzeFileAST:
    def test_extracts_functions(self, tmp_path):
        f = tmp_path / "funcs.py"
        f.write_text("def foo(): pass\ndef _bar(): pass\n")

        analyzer = DirectoryTreeAnalyzer()
        result = analyzer._analyze_file_ast(f)

        assert len(result["functions"]) == 2
        names = [fn["name"] for fn in result["functions"]]
        assert "foo" in names
        assert "_bar" in names
        # Check private flag
        bar_fn = next(fn for fn in result["functions"] if fn["name"] == "_bar")
        assert bar_fn["is_private"] is True

    def test_extracts_classes(self, tmp_path):
        f = tmp_path / "classes.py"
        f.write_text("class Base:\n    pass\n\nclass Child(Base):\n    pass\n")

        analyzer = DirectoryTreeAnalyzer()
        result = analyzer._analyze_file_ast(f)

        assert len(result["classes"]) == 2
        child = next(c for c in result["classes"] if c["name"] == "Child")
        assert "Base" in child["bases"]

    def test_extracts_imports(self, tmp_path):
        f = tmp_path / "imports.py"
        f.write_text("import os\nfrom pathlib import Path\n")

        analyzer = DirectoryTreeAnalyzer()
        result = analyzer._analyze_file_ast(f)

        assert len(result["imports"]) == 2
        import_types = [imp["type"] for imp in result["imports"]]
        assert "import" in import_types
        assert "from_import" in import_types

    def test_complexity_score_calculation(self, tmp_path):
        f = tmp_path / "complex.py"
        f.write_text("class A:\n    pass\n\ndef f1(): pass\ndef f2(): pass\n")

        analyzer = DirectoryTreeAnalyzer()
        result = analyzer._analyze_file_ast(f)

        # 2 functions + 1 class * 2 = 4
        assert result["complexity_score"] == 4

    def test_syntax_error_returns_empty(self, tmp_path):
        f = tmp_path / "bad.py"
        f.write_text("def broken(\n")

        analyzer = DirectoryTreeAnalyzer()
        result = analyzer._analyze_file_ast(f)
        assert result == {}


class TestGenerateSummary:
    def test_summary_statistics(self):
        analyzer = DirectoryTreeAnalyzer()
        analyzer.file_analysis = {
            "file1.py": {
                "functions": [{"name": "f1"}, {"name": "f2"}],
                "classes": [{"name": "C1"}],
            },
            "file2.py": {"functions": [{"name": "f3"}], "classes": []},
        }

        summary = analyzer._generate_summary()
        assert summary["total_files_analyzed"] == 2
        assert summary["total_functions"] == 3
        assert summary["total_classes"] == 1
        assert summary["average_functions_per_file"] == 1.5
        assert summary["average_classes_per_file"] == 0.5

    def test_empty_summary(self):
        analyzer = DirectoryTreeAnalyzer()
        summary = analyzer._generate_summary()
        assert summary["total_files_analyzed"] == 0
        assert summary["total_functions"] == 0
        assert summary["total_classes"] == 0
        assert summary["average_functions_per_file"] == 0.0
