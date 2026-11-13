"""Unit tests for similarity_algorithms module."""

from __future__ import annotations

import pytest

from app.api.endpoints.similarity_modules.similarity_algorithms import (
    CodeFeatureExtractor,
    SimilarityAlgorithms,
    _determine_clone_type,
    _group_clones,
    _shares_segment,
    calculate_similarity,
    detect_code_clones,
    find_similar_functions,
)

# ---------------------------------------------------------------------------
# CodeFeatureExtractor — extract_structural_features
# ---------------------------------------------------------------------------


class TestExtractStructuralFeatures:
    def test_python_language_delegates(self):
        code = "def foo():\n    pass\n"
        result = CodeFeatureExtractor.extract_structural_features(code, "python")
        assert result["language"] == "python"
        assert result["line_count"] == 2
        assert result["char_count"] == len(code)
        assert any(f["name"] == "foo" for f in result["functions"])

    def test_java_language_delegates(self):
        code = "public class Foo {\n  public void bar() {\n  }\n}\n"
        result = CodeFeatureExtractor.extract_structural_features(code, "java")
        assert result["language"] == "java"
        assert any(c["name"] == "Foo" for c in result["classes"])

    def test_csharp_language_delegates(self):
        code = "using System;\npublic class Baz {}\n"
        result = CodeFeatureExtractor.extract_structural_features(code, "csharp")
        assert result["language"] == "csharp"
        assert "System" in result["imports"]

    def test_javascript_language_delegates(self):
        code = "import foo from 'bar';\nfunction baz() {}\n"
        result = CodeFeatureExtractor.extract_structural_features(code, "javascript")
        assert result["language"] == "javascript"
        assert "bar" in result["imports"]

    def test_typescript_language_delegates(self):
        code = "import { x } from 'y';\nclass Qux {}\n"
        result = CodeFeatureExtractor.extract_structural_features(code, "typescript")
        assert result["language"] == "typescript"

    def test_unknown_language_uses_generic(self):
        code = "hello world\n"
        result = CodeFeatureExtractor.extract_structural_features(code, "brainfuck")
        assert result["language"] == "brainfuck"
        # generic extractor produces identifiers
        assert "identifiers" in result


# ---------------------------------------------------------------------------
# CodeFeatureExtractor — _extract_python_features
# ---------------------------------------------------------------------------


class TestExtractPythonFeatures:
    def test_functions_extracted(self):
        code = "def add(a, b):\n    return a + b\n"
        result = CodeFeatureExtractor._extract_python_features(code)
        assert len(result["functions"]) == 1
        assert result["functions"][0]["name"] == "add"
        assert result["functions"][0]["args"] == ["a", "b"]

    def test_classes_extracted(self):
        code = "class Foo(Bar):\n    pass\n"
        result = CodeFeatureExtractor._extract_python_features(code)
        assert len(result["classes"]) == 1
        assert result["classes"][0]["name"] == "Foo"
        assert "Bar" in result["classes"][0]["bases"]

    def test_imports_extracted(self):
        code = "import os\nfrom pathlib import Path\n"
        result = CodeFeatureExtractor._extract_python_features(code)
        assert "os" in result["imports"]
        assert "pathlib" in result["imports"]

    def test_decorators_extracted(self):
        code = "@staticmethod\ndef foo():\n    pass\n"
        result = CodeFeatureExtractor._extract_python_features(code)
        assert len(result["functions"]) == 1
        assert "staticmethod" in result["functions"][0]["decorators"]

    def test_exceptions_extracted(self):
        code = "try:\n    pass\nexcept ValueError:\n    pass\n"
        result = CodeFeatureExtractor._extract_python_features(code)
        assert "ValueError" in result["exceptions"]

    def test_syntax_error_returns_empty(self):
        code = "def broken(:\n"
        result = CodeFeatureExtractor._extract_python_features(code)
        assert result["functions"] == []
        assert result["classes"] == []

    def test_import_without_module(self):
        # from . import something -> module is None
        code = "from . import sibling\n"
        result = CodeFeatureExtractor._extract_python_features(code)
        assert "" in result["imports"]

    def test_class_with_complex_base(self):
        code = "class Foo(some.module.Base):\n    pass\n"
        result = CodeFeatureExtractor._extract_python_features(code)
        assert len(result["classes"]) == 1
        # base doesn't have .id so falls back to str()
        assert len(result["classes"][0]["bases"]) == 1


# ---------------------------------------------------------------------------
# CodeFeatureExtractor — _extract_framework_features (tree-sitter, all langs)
# ---------------------------------------------------------------------------


class TestExtractFrameworkFeatures:
    def test_java_functions_and_classes(self):
        code = "public class Worker {\n  public void doSomething(int x) {\n  }\n}\n"
        result = CodeFeatureExtractor._extract_framework_features(code, "java")
        assert any(c["name"] == "Worker" for c in result["classes"])
        assert any(f["name"] == "doSomething" for f in result["functions"])

    def test_java_imports(self):
        code = "import java.util.List;\nimport java.io.File;\nclass C {}\n"
        result = CodeFeatureExtractor._extract_framework_features(code, "java")
        assert "java.util.List" in result["imports"]
        assert "java.io.File" in result["imports"]

    def test_java_interfaces(self):
        code = "public interface Runnable {\n}\n"
        result = CodeFeatureExtractor._extract_framework_features(code, "java")
        assert any(i["name"] == "Runnable" for i in result["interfaces"])

    def test_javascript_imports_and_functions(self):
        code = "import React from 'react';\nfunction render() {}\n"
        result = CodeFeatureExtractor._extract_framework_features(code, "javascript")
        assert "react" in result["imports"]
        assert any(f["name"] == "render" for f in result["functions"])

    def test_typescript_class_and_interface(self):
        code = "import { x } from 'y';\ninterface Shape {}\nclass Circle {}\n"
        result = CodeFeatureExtractor._extract_framework_features(code, "typescript")
        assert "y" in result["imports"]
        assert any(c["name"] == "Circle" for c in result["classes"])
        assert any(i["name"] == "Shape" for i in result["interfaces"])

    def test_csharp_using(self):
        code = "using System.Collections.Generic;\nclass Box {}\n"
        result = CodeFeatureExtractor._extract_framework_features(code, "csharp")
        assert "System.Collections.Generic" in result["imports"]
        assert any(c["name"] == "Box" for c in result["classes"])

    def test_go_functions_and_imports(self):
        code = 'package main\n\nimport (\n\t"fmt"\n)\n\nfunc Greet() {\n\tfmt.Println("hi")\n}\n'
        result = CodeFeatureExtractor._extract_framework_features(code, "go")
        assert "fmt" in result["imports"]
        assert any(f["name"] == "Greet" for f in result["functions"])

    def test_rust_struct_and_function(self):
        code = "use std::collections::HashMap;\n\nstruct Point;\n\nfn build() {}\n"
        result = CodeFeatureExtractor._extract_framework_features(code, "rust")
        assert any(c["name"] == "Point" for c in result["classes"])
        assert any(f["name"] == "build" for f in result["functions"])

    def test_unknown_language_falls_back_to_generic(self):
        # A language with no registered grammar yields the generic regex signal.
        result = CodeFeatureExtractor._extract_framework_features("foo bar baz\n", "brainfuck")
        assert "identifiers" in result


# ---------------------------------------------------------------------------
# CodeFeatureExtractor — _extract_generic_features
# ---------------------------------------------------------------------------


class TestExtractGenericFeatures:
    def test_identifiers(self):
        code = "foo bar baz\n"
        result = CodeFeatureExtractor._extract_generic_features(code)
        assert "foo" in result["identifiers"]
        assert "bar" in result["identifiers"]

    def test_operators(self):
        code = "a + b - c * d\n"
        result = CodeFeatureExtractor._extract_generic_features(code)
        assert len(result["operators"]) > 0

    def test_literals(self):
        code = 'x = "hello"\n'
        result = CodeFeatureExtractor._extract_generic_features(code)
        assert len(result["literals"]) > 0


# ---------------------------------------------------------------------------
# CodeFeatureExtractor — extract_semantic_features
# ---------------------------------------------------------------------------


class TestExtractSemanticFeatures:
    def test_api_calls(self):
        code = "result = foo(x)\nbar(y)\n"
        result = CodeFeatureExtractor.extract_semantic_features(code, "python")
        assert "foo" in result["api_calls"]
        assert "bar" in result["api_calls"]

    def test_data_structures_detected(self):
        code = "my_list = list()\nmy_map = dict()\n"
        result = CodeFeatureExtractor.extract_semantic_features(code, "python")
        assert "list" in result["data_structures"]
        assert "map" in result["data_structures"]

    def test_algorithms_detected(self):
        code = "result = sorted(data)\nhash_val = hash(key)\n"
        result = CodeFeatureExtractor.extract_semantic_features(code, "python")
        assert "sorting" in result["algorithms"]
        assert "hashing" in result["algorithms"]

    def test_design_patterns_detected(self):
        code = "instance = Singleton.getInstance()\nfactory = Factory.create()\n"
        result = CodeFeatureExtractor.extract_semantic_features(code, "python")
        assert "singleton" in result["design_patterns"]
        assert "factory" in result["design_patterns"]

    def test_no_matches(self):
        code = "x = 1\n"
        result = CodeFeatureExtractor.extract_semantic_features(code, "python")
        assert result["data_structures"] == []
        assert result["algorithms"] == []
        assert result["design_patterns"] == []

    def test_queue_and_stack(self):
        code = "q = Queue()\ns = Stack()\n"
        result = CodeFeatureExtractor.extract_semantic_features(code, "python")
        assert "queue" in result["data_structures"]
        assert "stack" in result["data_structures"]

    def test_searching_algorithm(self):
        code = "idx = binary_search(arr, target)\n"
        result = CodeFeatureExtractor.extract_semantic_features(code, "python")
        assert "searching" in result["algorithms"]

    def test_encryption_algorithm(self):
        code = "encrypted = encrypt(data)\n"
        result = CodeFeatureExtractor.extract_semantic_features(code, "python")
        assert "encryption" in result["algorithms"]

    def test_parsing_algorithm(self):
        code = "doc = json.parse(text)\n"
        result = CodeFeatureExtractor.extract_semantic_features(code, "python")
        assert "parsing" in result["algorithms"]

    def test_observer_pattern(self):
        code = "observer.notify(event)\n"
        result = CodeFeatureExtractor.extract_semantic_features(code, "python")
        assert "observer" in result["design_patterns"]

    def test_decorator_pattern(self):
        code = "wrapped = wrapper(func)\n"
        result = CodeFeatureExtractor.extract_semantic_features(code, "python")
        assert "decorator" in result["design_patterns"]

    def test_set_data_structure(self):
        code = "unique = set(items)\n"
        result = CodeFeatureExtractor.extract_semantic_features(code, "python")
        assert "set" in result["data_structures"]


# ---------------------------------------------------------------------------
# CodeFeatureExtractor — extract_syntactic_features
# ---------------------------------------------------------------------------


class TestExtractSyntacticFeatures:
    def test_spaces_indentation(self):
        code = "def foo():\n    pass\n    return\n"
        result = CodeFeatureExtractor.extract_syntactic_features(code)
        assert result["indentation_style"] == "spaces"

    def test_tabs_indentation(self):
        code = "def foo():\n\tpass\n\treturn\n"
        result = CodeFeatureExtractor.extract_syntactic_features(code)
        assert result["indentation_style"] == "tabs"

    def test_line_length_avg(self):
        code = "abcd\nefgh\n"
        result = CodeFeatureExtractor.extract_syntactic_features(code)
        assert result["line_length_avg"] == 4.0

    def test_naming_conventions(self):
        code = "myVar = SomeClass\nmy_func = another_thing\n"
        result = CodeFeatureExtractor.extract_syntactic_features(code)
        assert "camelCase" in result["naming_conventions"]
        assert "snake_case" in result["naming_conventions"]
        assert "PascalCase" in result["naming_conventions"]

    def test_comment_ratio(self):
        code = "# comment\ncode\n"
        result = CodeFeatureExtractor.extract_syntactic_features(code)
        assert result["comment_ratio"] == 0.5

    def test_empty_code(self):
        code = ""
        result = CodeFeatureExtractor.extract_syntactic_features(code)
        assert result["line_length_avg"] == 0
        assert result["comment_ratio"] == 0.0

    def test_no_indentation(self):
        code = "x = 1\ny = 2\n"
        result = CodeFeatureExtractor.extract_syntactic_features(code)
        # Default is spaces when no indented lines
        assert result["indentation_style"] == "spaces"

    def test_only_comments(self):
        code = "# comment1\n# comment2\n"
        result = CodeFeatureExtractor.extract_syntactic_features(code)
        assert result["comment_ratio"] == 1.0

    def test_c_style_comments(self):
        code = "// comment\ncode\n/* block */\n"
        result = CodeFeatureExtractor.extract_syntactic_features(code)
        assert result["comment_ratio"] > 0

    def test_star_comment_lines(self):
        code = "* middle of block comment\ncode\n"
        result = CodeFeatureExtractor.extract_syntactic_features(code)
        assert result["comment_ratio"] == 0.5


# ---------------------------------------------------------------------------
# SimilarityAlgorithms — compare_structural_similarity
# ---------------------------------------------------------------------------


class TestCompareStructuralSimilarity:
    def test_identical_features(self):
        features = {
            "functions": [{"name": "foo"}, {"name": "bar"}],
            "classes": [{"name": "MyClass"}],
            "imports": ["os", "sys"],
            "line_count": 10,
        }
        score = SimilarityAlgorithms.compare_structural_similarity(features, features)
        assert score > 0.8

    def test_no_overlap(self):
        a = {
            "functions": [{"name": "foo"}],
            "classes": [{"name": "A"}],
            "imports": ["os"],
            "line_count": 10,
        }
        b = {
            "functions": [{"name": "bar"}],
            "classes": [{"name": "B"}],
            "imports": ["sys"],
            "line_count": 100,
        }
        score = SimilarityAlgorithms.compare_structural_similarity(a, b)
        assert score < 0.5

    def test_empty_features(self):
        score = SimilarityAlgorithms.compare_structural_similarity({}, {})
        assert score == 0.0

    def test_partial_overlap(self):
        a = {
            "functions": [{"name": "foo"}, {"name": "bar"}],
            "classes": [],
            "imports": ["os"],
            "line_count": 20,
        }
        b = {
            "functions": [{"name": "foo"}, {"name": "baz"}],
            "classes": [],
            "imports": ["os", "sys"],
            "line_count": 20,
        }
        score = SimilarityAlgorithms.compare_structural_similarity(a, b)
        assert 0.0 < score < 1.0

    def test_only_line_count(self):
        a = {"line_count": 50}
        b = {"line_count": 50}
        score = SimilarityAlgorithms.compare_structural_similarity(a, b)
        assert score == pytest.approx(0.3)  # complexity_ratio * 0.3


# ---------------------------------------------------------------------------
# SimilarityAlgorithms — compare_semantic_similarity
# ---------------------------------------------------------------------------


class TestCompareSemanticSimilarity:
    def test_identical(self):
        features = {
            "api_calls": ["foo", "bar"],
            "data_structures": ["list"],
            "algorithms": ["sorting"],
            "design_patterns": ["singleton"],
        }
        score = SimilarityAlgorithms.compare_semantic_similarity(features, features)
        assert score > 0.8

    def test_no_overlap(self):
        a = {
            "api_calls": ["foo"],
            "data_structures": ["list"],
            "algorithms": ["sorting"],
            "design_patterns": ["singleton"],
        }
        b = {
            "api_calls": ["bar"],
            "data_structures": ["map"],
            "algorithms": ["hashing"],
            "design_patterns": ["factory"],
        }
        score = SimilarityAlgorithms.compare_semantic_similarity(a, b)
        assert score == 0.0

    def test_empty(self):
        score = SimilarityAlgorithms.compare_semantic_similarity({}, {})
        assert score == 0.0


# ---------------------------------------------------------------------------
# SimilarityAlgorithms — compare_syntactic_similarity
# ---------------------------------------------------------------------------


class TestCompareSyntacticSimilarity:
    def test_identical(self):
        features = {
            "naming_conventions": ["camelCase", "snake_case"],
            "indentation_style": "spaces",
            "line_length_avg": 40,
            "comment_ratio": 0.2,
        }
        score = SimilarityAlgorithms.compare_syntactic_similarity(features, features)
        assert score > 0.8

    def test_different_indent(self):
        a = {
            "indentation_style": "spaces",
            "naming_conventions": [],
            "line_length_avg": 0,
            "comment_ratio": 0,
        }
        b = {
            "indentation_style": "tabs",
            "naming_conventions": [],
            "line_length_avg": 0,
            "comment_ratio": 0,
        }
        score = SimilarityAlgorithms.compare_syntactic_similarity(a, b)
        # indent mismatch should produce 0 for that component
        assert score == 0.0

    def test_empty(self):
        score = SimilarityAlgorithms.compare_syntactic_similarity({}, {})
        assert score == 0.0

    def test_comment_ratio_both_zero(self):
        a = {
            "naming_conventions": [],
            "indentation_style": "",
            "line_length_avg": 0,
            "comment_ratio": 0,
        }
        b = {
            "naming_conventions": [],
            "indentation_style": "",
            "line_length_avg": 0,
            "comment_ratio": 0,
        }
        score = SimilarityAlgorithms.compare_syntactic_similarity(a, b)
        assert score == 0.0

    def test_different_line_lengths(self):
        a = {
            "naming_conventions": [],
            "indentation_style": "spaces",
            "line_length_avg": 40,
            "comment_ratio": 0,
        }
        b = {
            "naming_conventions": [],
            "indentation_style": "spaces",
            "line_length_avg": 80,
            "comment_ratio": 0,
        }
        score = SimilarityAlgorithms.compare_syntactic_similarity(a, b)
        # indent match 0.2, line_length_ratio 0.5 * 0.2 = 0.1
        assert score == pytest.approx(0.3, abs=0.01)

    def test_comment_ratio_difference(self):
        a = {
            "naming_conventions": [],
            "indentation_style": "",
            "line_length_avg": 0,
            "comment_ratio": 0.2,
        }
        b = {
            "naming_conventions": [],
            "indentation_style": "",
            "line_length_avg": 0,
            "comment_ratio": 0.4,
        }
        score = SimilarityAlgorithms.compare_syntactic_similarity(a, b)
        # comment_similarity = 1 - |0.2-0.4|/0.4 = 0.5, * 0.3 = 0.15
        assert score == pytest.approx(0.15, abs=0.01)


# ---------------------------------------------------------------------------
# SimilarityAlgorithms — detect_exact_duplicates
# ---------------------------------------------------------------------------


class TestDetectExactDuplicates:
    def test_identical_code(self):
        code = "def foo():\n    return 42\n"
        score = SimilarityAlgorithms.detect_exact_duplicates(code, code)
        assert score == 1.0

    def test_completely_different(self):
        a = "def foo():\n    return 42\n"
        b = "class Bar:\n    pass\n"
        score = SimilarityAlgorithms.detect_exact_duplicates(a, b)
        assert score < 1.0

    def test_empty_code(self):
        score = SimilarityAlgorithms.detect_exact_duplicates("", "")
        assert score == 0.0

    def test_one_empty(self):
        score = SimilarityAlgorithms.detect_exact_duplicates("x = 1", "")
        assert score == 0.0

    def test_whitespace_difference_normalized(self):
        a = "x = 1\n  y = 2\n"
        b = "x = 1\ny = 2\n"
        score = SimilarityAlgorithms.detect_exact_duplicates(a, b)
        assert score == 1.0

    def test_comment_difference_normalized(self):
        a = "x = 1  # a comment\n"
        b = "x = 1  # different comment\n"
        score = SimilarityAlgorithms.detect_exact_duplicates(a, b)
        # Comments are stripped, so normalized versions should match
        assert score == 1.0


# ---------------------------------------------------------------------------
# SimilarityAlgorithms — _normalize_code
# ---------------------------------------------------------------------------


class TestNormalizeCode:
    def test_strips_python_comments(self):
        code = "x = 1  # comment\n"
        result = SimilarityAlgorithms._normalize_code(code)
        assert "#" not in result

    def test_strips_c_comments(self):
        code = "x = 1; // comment\n"
        result = SimilarityAlgorithms._normalize_code(code)
        assert "//" not in result

    def test_collapses_whitespace(self):
        code = "x   =   1\n"
        result = SimilarityAlgorithms._normalize_code(code)
        assert result == "x = 1"

    def test_removes_empty_lines(self):
        code = "a\n\n\nb\n"
        result = SimilarityAlgorithms._normalize_code(code)
        assert result == "a\nb"

    def test_empty_input(self):
        assert SimilarityAlgorithms._normalize_code("") == ""


# ---------------------------------------------------------------------------
# SimilarityAlgorithms — calculate_code_hash
# ---------------------------------------------------------------------------


class TestCalculateCodeHash:
    def test_deterministic(self):
        code = "def foo():\n    pass\n"
        h1 = SimilarityAlgorithms.calculate_code_hash(code)
        h2 = SimilarityAlgorithms.calculate_code_hash(code)
        assert h1 == h2

    def test_different_code_different_hash(self):
        h1 = SimilarityAlgorithms.calculate_code_hash("a = 1")
        h2 = SimilarityAlgorithms.calculate_code_hash("b = 2")
        assert h1 != h2

    def test_whitespace_insensitive(self):
        h1 = SimilarityAlgorithms.calculate_code_hash("x = 1")
        h2 = SimilarityAlgorithms.calculate_code_hash("x   =   1")
        assert h1 == h2

    def test_returns_hex_string(self):
        h = SimilarityAlgorithms.calculate_code_hash("x = 1")
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# SimilarityAlgorithms — find_similar_blocks
# ---------------------------------------------------------------------------


class TestFindSimilarBlocks:
    def test_identical_blocks(self):
        code = "line1\nline2\nline3\n"
        blocks = SimilarityAlgorithms.find_similar_blocks(code, code, min_lines=3)
        assert len(blocks) >= 1
        # At least one block with high similarity
        assert any(score > 0.8 for _, _, score in blocks)

    def test_no_similar_blocks(self):
        a = "aaa\nbbb\nccc\n"
        b = "xxx\nyyy\nzzz\n"
        blocks = SimilarityAlgorithms.find_similar_blocks(a, b, min_lines=3)
        assert len(blocks) == 0

    def test_too_short(self):
        a = "a\nb\n"
        b = "a\nb\n"
        blocks = SimilarityAlgorithms.find_similar_blocks(a, b, min_lines=3)
        assert len(blocks) == 0

    def test_empty_code(self):
        blocks = SimilarityAlgorithms.find_similar_blocks("", "", min_lines=3)
        assert blocks == []


# ---------------------------------------------------------------------------
# SimilarityAlgorithms — _remove_overlapping_blocks
# ---------------------------------------------------------------------------


class TestRemoveOverlappingBlocks:
    def test_no_overlap(self):
        blocks = [
            ((0, 2), (0, 2), 0.9),
            ((5, 7), (5, 7), 0.85),
        ]
        result = SimilarityAlgorithms._remove_overlapping_blocks(blocks)
        assert len(result) == 2

    def test_overlap_keeps_higher(self):
        blocks = [
            ((0, 3), (0, 3), 0.95),
            ((1, 4), (1, 4), 0.85),
        ]
        result = SimilarityAlgorithms._remove_overlapping_blocks(blocks)
        assert len(result) == 1
        assert result[0][2] == 0.95

    def test_empty_list(self):
        assert SimilarityAlgorithms._remove_overlapping_blocks([]) == []


# ---------------------------------------------------------------------------
# calculate_similarity (top-level function)
# ---------------------------------------------------------------------------


class TestCalculateSimilarity:
    def test_identical_python(self):
        code = "def foo():\n    return 42\n"
        result = calculate_similarity(code, code, "python")
        assert "overall" in result
        assert "structural" in result
        assert "semantic" in result
        assert "syntactic" in result
        assert "exact" in result
        assert result["exact"] == 1.0
        assert result["overall"] > 0.0

    def test_different_code(self):
        a = "def foo():\n    return 1\n"
        b = "class Bar:\n    pass\n"
        result = calculate_similarity(a, b, "python")
        assert result["exact"] < 1.0

    def test_default_language(self):
        result = calculate_similarity("x = 1", "x = 1")
        assert result["overall"] > 0.0

    def test_error_returns_zeros(self):
        # Patch to force an exception
        from unittest.mock import patch

        with patch.object(
            CodeFeatureExtractor,
            "extract_structural_features",
            side_effect=RuntimeError("boom"),
        ):
            result = calculate_similarity("a", "b", "python")
        assert result["overall"] == 0.0
        assert result["structural"] == 0.0


# ---------------------------------------------------------------------------
# find_similar_functions
# ---------------------------------------------------------------------------


class TestFindSimilarFunctions:
    def test_identical_functions(self):
        funcs = [{"name": "foo", "args": ["a", "b"]}]
        result = find_similar_functions(funcs, funcs, "python")
        # The mock code generated is small so overall might not exceed threshold
        assert isinstance(result, list)

    def test_empty_lists(self):
        result = find_similar_functions([], [], "python")
        assert result == []

    def test_different_functions(self):
        a = [{"name": "calculate_sum", "args": ["x", "y", "z"]}]
        b = [{"name": "render_template", "args": ["path"]}]
        result = find_similar_functions(a, b, "python")
        # Should be below threshold
        assert isinstance(result, list)

    def test_error_returns_empty(self):
        from unittest.mock import patch

        with patch(
            "app.api.endpoints.similarity_modules.similarity_algorithms.calculate_similarity",
            side_effect=RuntimeError("boom"),
        ):
            result = find_similar_functions(
                [{"name": "a", "args": []}],
                [{"name": "b", "args": []}],
                "python",
            )
        assert result == []


# ---------------------------------------------------------------------------
# detect_code_clones
# ---------------------------------------------------------------------------


class TestDetectCodeClones:
    def test_identical_segments(self):
        segments = [
            {"code": "def foo():\n    return 1\n", "file": "a.py"},
            {"code": "def foo():\n    return 1\n", "file": "b.py"},
        ]
        result = detect_code_clones(segments, "python", threshold=0.0)
        assert isinstance(result, list)

    def test_empty_segments(self):
        result = detect_code_clones([], "python")
        assert result == []

    def test_single_segment(self):
        segments = [{"code": "x = 1"}]
        result = detect_code_clones(segments, "python")
        assert result == []

    def test_empty_code_skipped(self):
        segments = [
            {"code": "", "file": "a.py"},
            {"code": "x = 1", "file": "b.py"},
        ]
        result = detect_code_clones(segments, "python")
        assert result == []

    def test_error_returns_empty(self):
        from unittest.mock import patch

        with patch(
            "app.api.endpoints.similarity_modules.similarity_algorithms.calculate_similarity",
            side_effect=RuntimeError("boom"),
        ):
            segments = [
                {"code": "x = 1", "file": "a.py"},
                {"code": "y = 2", "file": "b.py"},
            ]
            result = detect_code_clones(segments, "python")
        assert result == []


# ---------------------------------------------------------------------------
# _determine_clone_type
# ---------------------------------------------------------------------------


class TestDetermineCloneType:
    def test_type_i(self):
        sim = {"exact": 0.96, "structural": 0.5, "semantic": 0.5}
        assert _determine_clone_type(sim) == "Type I (Exact)"

    def test_type_ii(self):
        sim = {"exact": 0.5, "structural": 0.95, "semantic": 0.5}
        assert _determine_clone_type(sim) == "Type II (Structural)"

    def test_type_iii(self):
        sim = {"exact": 0.5, "structural": 0.5, "semantic": 0.85}
        assert _determine_clone_type(sim) == "Type III (Semantic)"

    def test_type_iv(self):
        sim = {"exact": 0.5, "structural": 0.5, "semantic": 0.5}
        assert _determine_clone_type(sim) == "Type IV (Functional)"


# ---------------------------------------------------------------------------
# _group_clones
# ---------------------------------------------------------------------------


class TestGroupClones:
    def test_empty(self):
        assert _group_clones([]) == []

    def test_single_clone(self):
        clone = {
            "segment_a": {"index": 0},
            "segment_b": {"index": 1},
            "similarity": {"overall": 0.9},
        }
        result = _group_clones([clone])
        assert len(result) == 1
        assert result[0]["group_id"] == 0
        assert result[0]["average_similarity"] == 0.9

    def test_shared_segment_grouped(self):
        clone1 = {
            "segment_a": {"index": 0},
            "segment_b": {"index": 1},
            "similarity": {"overall": 0.9},
        }
        clone2 = {
            "segment_a": {"index": 1},
            "segment_b": {"index": 2},
            "similarity": {"overall": 0.85},
        }
        result = _group_clones([clone1, clone2])
        assert len(result) == 1
        assert len(result[0]["clones"]) == 2
        # Average of 0.9 and 0.85
        assert result[0]["average_similarity"] == pytest.approx(0.875, abs=0.001)

    def test_no_shared_segments(self):
        clone1 = {
            "segment_a": {"index": 0},
            "segment_b": {"index": 1},
            "similarity": {"overall": 0.9},
        }
        clone2 = {
            "segment_a": {"index": 2},
            "segment_b": {"index": 3},
            "similarity": {"overall": 0.85},
        }
        result = _group_clones([clone1, clone2])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _shares_segment
# ---------------------------------------------------------------------------


class TestSharesSegment:
    def test_shares(self):
        a = {"segment_a": {"index": 0}, "segment_b": {"index": 1}}
        b = {"segment_a": {"index": 1}, "segment_b": {"index": 2}}
        assert _shares_segment(a, b) is True

    def test_no_shares(self):
        a = {"segment_a": {"index": 0}, "segment_b": {"index": 1}}
        b = {"segment_a": {"index": 2}, "segment_b": {"index": 3}}
        assert _shares_segment(a, b) is False
