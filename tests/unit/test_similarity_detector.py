"""Tests for SimilarityDetector.detect()."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.similarity_detector import (
    SimilarityAnalysisResult,
    SimilarityDetector,
    SimilarityScope,
    SimilarityType,
    get_similarity_detector,
)


@pytest.fixture
def detector() -> SimilarityDetector:
    return SimilarityDetector()


# ---------------------------------------------------------------------------
# EXACT similarity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exact_identical_files(detector: SimilarityDetector, tmp_path: Path):
    content = "def hello():\n    return 'world'\n"
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text(content)
    f2.write_text(content)

    result = await detector.detect(
        [str(f1), str(f2)],
        similarity_type=SimilarityType.EXACT,
        threshold=0.5,
    )

    assert result.total_comparisons == 1
    assert len(result.matches) == 1
    assert result.matches[0].similarity_score == 1.0


@pytest.mark.asyncio
async def test_exact_different_files(detector: SimilarityDetector, tmp_path: Path):
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text("def a(): pass")
    f2.write_text("def b(): pass")

    result = await detector.detect(
        [str(f1), str(f2)],
        similarity_type=SimilarityType.EXACT,
        threshold=0.5,
    )

    assert result.total_comparisons == 1
    assert len(result.matches) == 0


# ---------------------------------------------------------------------------
# STRUCTURAL similarity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structural_identical(detector: SimilarityDetector, tmp_path: Path):
    code = "class Foo:\n    def bar(self): pass\n"
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text(code)
    f2.write_text(code)

    result = await detector.detect(
        [str(f1), str(f2)],
        similarity_type=SimilarityType.STRUCTURAL,
        threshold=0.5,
    )

    assert len(result.matches) == 1
    assert result.matches[0].similarity_score == 1.0


@pytest.mark.asyncio
async def test_structural_similar(detector: SimilarityDetector, tmp_path: Path):
    """Two files with the same structure but different names should score high."""
    code_a = "def alpha(x):\n    return x + 1\n"
    code_b = "def beta(y):\n    return y + 2\n"
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text(code_a)
    f2.write_text(code_b)

    result = await detector.detect(
        [str(f1), str(f2)],
        similarity_type=SimilarityType.STRUCTURAL,
        threshold=0.5,
    )

    # Same AST structure (FunctionDef, arguments, arg, Return, BinOp, ...)
    assert len(result.matches) == 1
    assert result.matches[0].similarity_score > 0.7


@pytest.mark.asyncio
async def test_structural_very_different(detector: SimilarityDetector, tmp_path: Path):
    code_a = "x = 1\n"
    code_b = """\
class A:
    def m1(self): pass
    def m2(self): pass
    def m3(self): pass
class B(A):
    def m4(self): pass
"""
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text(code_a)
    f2.write_text(code_b)

    result = await detector.detect(
        [str(f1), str(f2)],
        similarity_type=SimilarityType.STRUCTURAL,
        threshold=0.8,
    )

    assert len(result.matches) == 0


# ---------------------------------------------------------------------------
# TOKEN similarity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_identical(detector: SimilarityDetector, tmp_path: Path):
    code = "for i in range(10):\n    print(i)\n"
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text(code)
    f2.write_text(code)

    result = await detector.detect(
        [str(f1), str(f2)],
        similarity_type=SimilarityType.TOKEN,
        threshold=0.5,
    )

    assert len(result.matches) == 1
    assert result.matches[0].similarity_score == 1.0


@pytest.mark.asyncio
async def test_token_below_threshold(detector: SimilarityDetector, tmp_path: Path):
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text("alpha beta gamma delta epsilon")
    f2.write_text("one two three four five six seven eight nine ten")

    result = await detector.detect(
        [str(f1), str(f2)],
        similarity_type=SimilarityType.TOKEN,
        threshold=0.8,
    )

    assert len(result.matches) == 0


# ---------------------------------------------------------------------------
# SEMANTIC similarity (falls back to STRUCTURAL)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semantic_identical(detector: SimilarityDetector, tmp_path: Path):
    code = "def foo(): return 42\n"
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text(code)
    f2.write_text(code)

    result = await detector.detect(
        [str(f1), str(f2)],
        similarity_type=SimilarityType.SEMANTIC,
        threshold=0.5,
    )

    assert len(result.matches) == 1
    assert result.matches[0].similarity_score == 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unreadable_file_skipped(detector: SimilarityDetector, tmp_path: Path):
    f1 = tmp_path / "exists.py"
    f1.write_text("x = 1")
    missing = tmp_path / "no_such_file.py"

    result = await detector.detect(
        [str(f1), str(missing)],
        similarity_type=SimilarityType.EXACT,
        threshold=0.0,
    )

    # Only 1 readable file => 0 pairs
    assert result.total_comparisons == 0
    assert len(result.matches) == 0


@pytest.mark.asyncio
async def test_empty_file_list(detector: SimilarityDetector):
    result = await detector.detect([], similarity_type=SimilarityType.EXACT, threshold=0.5)

    assert result.total_comparisons == 0
    assert len(result.matches) == 0
    assert result.analysis_time_ms >= 0


@pytest.mark.asyncio
async def test_single_file(detector: SimilarityDetector, tmp_path: Path):
    f1 = tmp_path / "only.py"
    f1.write_text("pass")

    result = await detector.detect(
        [str(f1)],
        similarity_type=SimilarityType.TOKEN,
        threshold=0.0,
    )

    assert result.total_comparisons == 0


@pytest.mark.asyncio
async def test_three_files_pairs(detector: SimilarityDetector, tmp_path: Path):
    """Three files should produce 3 pair-wise comparisons."""
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("pass")

    result = await detector.detect(
        [str(tmp_path / n) for n in ("a.py", "b.py", "c.py")],
        similarity_type=SimilarityType.EXACT,
        threshold=0.5,
    )

    assert result.total_comparisons == 3  # C(3,2) = 3
    # All identical so all should match
    assert len(result.matches) == 3


@pytest.mark.asyncio
async def test_result_metadata(detector: SimilarityDetector, tmp_path: Path):
    f1 = tmp_path / "a.py"
    f1.write_text("x = 1")

    result = await detector.detect(
        [str(f1)],
        similarity_type=SimilarityType.TOKEN,
        scope=SimilarityScope.FUNCTION,
        threshold=0.5,
    )

    assert result.similarity_type == SimilarityType.TOKEN
    assert result.scope == SimilarityScope.FUNCTION
    assert isinstance(result, SimilarityAnalysisResult)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------


def test_get_similarity_detector_singleton():
    # Reset global state first
    import services.similarity_detector as mod

    mod._detector = None
    d1 = get_similarity_detector()
    d2 = get_similarity_detector()
    assert d1 is d2


# ---------------------------------------------------------------------------
# Multi-language STRUCTURAL similarity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_structural_typescript_identical(detector: SimilarityDetector, tmp_path: Path):
    """STRUCTURAL comparison works for non-Python languages (tree-sitter path)."""
    code = "function add(a: number, b: number): number {\n  return a + b;\n}\n"
    f1 = tmp_path / "a.ts"
    f2 = tmp_path / "b.ts"
    f1.write_text(code)
    f2.write_text(code)

    result = await detector.detect(
        [str(f1), str(f2)],
        similarity_type=SimilarityType.STRUCTURAL,
        threshold=0.5,
    )

    assert len(result.matches) == 1
    assert result.matches[0].similarity_score == 1.0


@pytest.mark.asyncio
async def test_structural_go_similar_structure(detector: SimilarityDetector, tmp_path: Path):
    """Two Go functions with the same shape but different names score high."""
    f1 = tmp_path / "a.go"
    f2 = tmp_path / "b.go"
    f1.write_text("package main\n\nfunc alpha(x int) int {\n\treturn x + 1\n}\n")
    f2.write_text("package main\n\nfunc beta(y int) int {\n\treturn y + 2\n}\n")

    result = await detector.detect(
        [str(f1), str(f2)],
        similarity_type=SimilarityType.STRUCTURAL,
        threshold=0.5,
    )

    assert len(result.matches) == 1
    assert result.matches[0].similarity_score > 0.7


@pytest.mark.asyncio
async def test_structural_rust_distinct_files_do_not_match(
    detector: SimilarityDetector, tmp_path: Path
):
    """Structurally different Rust files stay below a high threshold."""
    f1 = tmp_path / "a.rs"
    f2 = tmp_path / "b.rs"
    f1.write_text("fn main() {}\n")
    f2.write_text(
        "struct Big;\nimpl Big {\n    fn m1(&self) {}\n    fn m2(&self) {}\n"
        "    fn m3(&self) {}\n}\n"
    )

    result = await detector.detect(
        [str(f1), str(f2)],
        similarity_type=SimilarityType.STRUCTURAL,
        threshold=0.8,
    )

    assert len(result.matches) == 0
