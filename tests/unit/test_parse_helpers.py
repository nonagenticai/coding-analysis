"""Tests for the shared multi-language parsing helpers.

Covers ``parse_source`` and ``node_type_sequence`` — the language-agnostic
primitives the repo-scanner, bug-hunter and similarity analyzers build on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from processing.languages.parse_helpers import node_type_sequence, parse_source

# ── Per-language fixture snippets ────────────────────────────────────────────
# Each snippet has one class/struct + a function + an import so every language
# exercises entity and relationship extraction.
_SNIPPETS: dict[str, str] = {
    ".py": "import os\n\n\nclass Widget:\n    def render(self) -> int:\n        return 1\n\n\ndef build() -> int:\n    return 2\n",
    ".ts": 'import { readFile } from "fs";\n\nclass Widget {\n  render(): number { return 1; }\n}\n\nfunction build(): number { return 2; }\n',
    ".js": 'import fs from "fs";\n\nclass Widget {\n  render() { return 1; }\n}\n\nfunction build() { return 2; }\n',
    ".go": 'package main\n\nimport "fmt"\n\ntype Widget struct{}\n\nfunc (w Widget) Render() int { return 1 }\n\nfunc build() int { return 2 }\n',
    ".rs": "use std::fmt;\n\nstruct Widget;\n\nimpl Widget {\n    fn render(&self) -> i32 { 1 }\n}\n\nfn build() -> i32 { 2 }\n",
    ".java": "import java.util.List;\n\nclass Widget {\n    int render() { return 1; }\n}\n",
    ".cpp": "#include <vector>\n\nstruct Widget {\n    int render() { return 1; }\n};\n\nint build() { return 2; }\n",
    ".cs": "using System;\n\nclass Widget {\n    int Render() { return 1; }\n}\n",
}


@pytest.mark.parametrize("suffix", sorted(_SNIPPETS))
async def test_parse_source_extracts_entities_for_every_language(
    suffix: str, tmp_path: Path
) -> None:
    """``parse_source`` returns entities for a real on-disk file in any language."""
    source = tmp_path / f"sample{suffix}"
    source.write_text(_SNIPPETS[suffix], encoding="utf-8")

    result = await parse_source(source, _SNIPPETS[suffix])

    assert result is not None, f"{suffix} should parse"
    assert result.entities, f"{suffix} should yield at least one entity"


@pytest.mark.parametrize("suffix", sorted(_SNIPPETS))
async def test_parse_source_works_for_paths_not_on_disk(suffix: str) -> None:
    """Content pulled from a container (path not on disk) still parses.

    Exercises the temporary-file materialisation path used for processors
    (Go, C++, C#) that re-read the file from disk.
    """
    virtual = Path("/container/workspace") / f"sample{suffix}"

    result = await parse_source(virtual, _SNIPPETS[suffix])

    assert result is not None, f"{suffix} should parse from virtual path"
    assert result.entities
    # Entity paths are rewritten back to the caller-supplied path.
    assert result.file_path == str(virtual)
    for entity in result.entities:
        assert entity.file_path == str(virtual)


async def test_parse_source_returns_none_for_unknown_extension(tmp_path: Path) -> None:
    """A ``.svelte`` file (no registered grammar) is skipped, not an error."""
    source = tmp_path / "Component.svelte"
    source.write_text("<script>let x = 1;</script>", encoding="utf-8")

    assert await parse_source(source, "<script>let x = 1;</script>") is None


@pytest.mark.parametrize("suffix", sorted(_SNIPPETS))
def test_node_type_sequence_non_empty_for_every_language(suffix: str) -> None:
    """``node_type_sequence`` produces a structural fingerprint per language."""
    sequence = node_type_sequence(Path(f"sample{suffix}"), _SNIPPETS[suffix])

    assert sequence, f"{suffix} should yield a node-type sequence"
    assert all(isinstance(node_type, str) for node_type in sequence)


def test_node_type_sequence_empty_for_unknown_extension() -> None:
    """An unknown extension yields an empty sequence — no structural signal."""
    assert node_type_sequence(Path("Component.svelte"), "<script>1</script>") == []


def test_node_type_sequence_distinguishes_structure() -> None:
    """Structurally different files produce different sequences."""
    flat = node_type_sequence(Path("a.py"), "x = 1\ny = 2\n")
    nested = node_type_sequence(Path("b.py"), "def f():\n    if x:\n        return 1\n")

    assert flat != nested
