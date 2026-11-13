"""Tests for multi-language code-graph extraction.

Covers ``collect_source_files`` and ``extract_graph_multilang`` — the
language-agnostic graph extraction path that lets the architecture graph
render for any language the platform supports, not just Python.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from models.core_models import EntityType
from processing.analyzers.deep.multi_language_entities import (
    collect_source_files,
    extract_graph_multilang,
)

# ── Per-language fixture snippets ────────────────────────────────────────────
# Each snippet defines a class/struct + functions + an import + (where the
# language supports it) an inheritance relationship.

_SNIPPETS: dict[str, tuple[str, str]] = {
    "typescript": (
        "app.ts",
        """
import { readFile } from "fs";

class Base {
    greet(): string { return "hi"; }
}

class Service extends Base {
    run(): number { return helper(); }
}

function helper(): number { return 42; }
""",
    ),
    "javascript": (
        "app.js",
        """
import { readFile } from "fs";

class Base {
    greet() { return "hi"; }
}

class Service extends Base {
    run() { return helper(); }
}

function helper() { return 42; }
""",
    ),
    "go": (
        "main.go",
        """
package main

import "fmt"

type Service struct {
    name string
}

func (s Service) Run() int {
    return helper()
}

func helper() int {
    return 42
}
""",
    ),
    "java": (
        "App.java",
        """
import java.util.List;

class Base {
    String greet() { return "hi"; }
}

class Service extends Base {
    int run() { return helper(); }
    int helper() { return 42; }
}
""",
    ),
    "rust": (
        "lib.rs",
        """
use std::collections::HashMap;

struct Service {
    name: String,
}

impl Service {
    fn run(&self) -> i32 { helper() }
}

fn helper() -> i32 { 42 }
""",
    ),
    "cpp": (
        "main.cpp",
        """
#include <vector>

class Base {
public:
    int greet() { return 1; }
};

class Service : public Base {
public:
    int run() { return helper(); }
};

int helper() { return 42; }
""",
    ),
    "csharp": (
        "App.cs",
        """
using System;

class Base {
    public string Greet() { return "hi"; }
}

class Service : Base {
    public int Run() { return Helper(); }
    public int Helper() { return 42; }
}
""",
    ),
}

# Entity types that count as a "type-like" declaration across languages.
_TYPE_LIKE = {
    EntityType.CLASS,
    EntityType.STRUCT,
    EntityType.INTERFACE,
    EntityType.ENUM,
}
# Entity types that count as a "callable" across languages.
_CALLABLE = {EntityType.FUNCTION, EntityType.METHOD}


@pytest.mark.parametrize("language", sorted(_SNIPPETS))
@pytest.mark.asyncio
async def test_extract_graph_single_language(language: str, tmp_path: Path):
    """Each supported language yields entities + relationships from a snippet."""
    filename, source = _SNIPPETS[language]
    (tmp_path / filename).write_text(source)

    files = collect_source_files(tmp_path)
    assert len(files) == 1, f"expected to collect {filename}"

    entities, relationships = await extract_graph_multilang(tmp_path, files)

    assert entities, f"{language}: expected at least one entity"
    # The detected language is recorded on every entity.
    assert all(e.language for e in entities)
    # At least one type-like declaration and one callable were found.
    entity_types = {e.type for e in entities}
    assert entity_types & _TYPE_LIKE, f"{language}: no class/struct/interface entity"
    assert entity_types & _CALLABLE, f"{language}: no function/method entity"
    # File paths are relative to the codebase root, not absolute.
    assert all(not Path(e.file_path).is_absolute() for e in entities)
    # Relationships should be produced (imports / contains / inherits / calls).
    assert relationships, f"{language}: expected at least one relationship"


@pytest.mark.asyncio
async def test_extract_graph_polyglot_repo(tmp_path: Path):
    """A polyglot repo produces ONE merged graph spanning every language."""
    (tmp_path / "app.py").write_text("def main():\n    return 1\n")
    (tmp_path / "main.go").write_text(_SNIPPETS["go"][1])
    (tmp_path / "app.ts").write_text(_SNIPPETS["typescript"][1])

    files = collect_source_files(tmp_path)
    assert len(files) == 3

    entities, _relationships = await extract_graph_multilang(tmp_path, files)

    languages = {e.language for e in entities}
    # All three languages contributed entities to the single merged graph.
    assert "go" in languages
    assert "typescript" in languages
    assert "python" in languages


def test_collect_source_files_skips_unsupported_and_vendored(tmp_path: Path):
    """Unsupported extensions and dependency dirs are excluded from collection."""
    (tmp_path / "app.ts").write_text("export const x = 1;\n")
    # No tree-sitter grammar for Svelte → must be silently skipped.
    (tmp_path / "Widget.svelte").write_text("<script>let x = 1;</script>\n")
    (tmp_path / "readme.md").write_text("# docs\n")
    vendored = tmp_path / "node_modules" / "pkg"
    vendored.mkdir(parents=True)
    (vendored / "index.js").write_text("module.exports = {};\n")

    files = collect_source_files(tmp_path)

    names = {f.name for f in files}
    assert names == {"app.ts"}


@pytest.mark.asyncio
async def test_extract_graph_skips_unreadable_file(tmp_path: Path):
    """A file that disappears between collection and parsing is skipped, not fatal."""
    good = tmp_path / "good.ts"
    good.write_text(_SNIPPETS["typescript"][1])
    missing = tmp_path / "missing.ts"

    # ``missing`` never gets created on disk but is passed in explicitly.
    entities, _relationships = await extract_graph_multilang(tmp_path, [good, missing])

    # The good file still produced entities; the missing one was skipped.
    assert entities


def test_collect_source_files_nonexistent_root(tmp_path: Path):
    """A nonexistent root returns an empty list rather than raising."""
    assert collect_source_files(tmp_path / "does-not-exist") == []
