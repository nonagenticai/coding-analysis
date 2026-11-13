"""Round 4 coverage tests for Rust language processor regex/fallback paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models.core_models import CoreCodeEntity, EntityType


class TestRustProcessorRegexParsing:
    """Test regex fallback parsing for Rust processor."""

    @pytest.fixture
    def processor(self):
        """Create a RustProcessor instance."""
        from processing.languages.rust.rust_processor import RustProcessor

        proc = RustProcessor()
        return proc

    @pytest.mark.asyncio
    async def test_regex_parse_full(self, processor):
        """Cover lines 171-197 - full regex parse."""
        content = """\
mod utils;
pub mod helpers;
use std::io;
use std::collections::HashMap;
pub struct Point {
    x: f64,
    y: f64,
}
pub enum Color {
    Red,
    Green,
    Blue,
}
pub trait Drawable {
    fn draw(&self);
}
pub async fn process_data(data: Vec<u8>) -> Result<(), Error> {
    // process
}
fn helper_func(x: i32) -> i32 {
    x + 1
}
impl Drawable for Point {
    fn draw(&self) {}
}
impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }
}
macro_rules! my_macro {
    ($x:expr) => { $x + 1 };
}
"""
        file_path = Path("test.rs")
        result = await processor._regex_parse(file_path, content)

        assert result.success is True
        assert result.language == "rust"
        assert result.parser_used.value == "regex_fallback"
        assert result.lines_of_code > 0

        entity_names = {e.name for e in result.entities}

        # Modules
        assert "utils" in entity_names
        assert "helpers" in entity_names

        # Use statements
        assert "std::io" in entity_names
        assert "std::collections::HashMap" in entity_names

        # Structs
        assert "Point" in entity_names

        # Enums
        assert "Color" in entity_names

        # Traits
        assert "Drawable" in entity_names

        # Functions (main, new, default are skipped)
        assert "process_data" in entity_names
        assert "helper_func" in entity_names

        # Impl blocks
        impl_names = [
            e.name for e in result.entities if "impl" in e.name.lower() or "for" in e.name.lower()
        ]
        assert len(impl_names) >= 1

        # Macros
        assert "my_macro" in entity_names

    def test_extract_modules_regex(self, processor):
        """Cover lines 199-226."""
        content = "mod utils;\npub mod helpers;\nmod tests;"
        file_path = Path("test.rs")
        entities = processor._extract_modules_regex(content, file_path)

        assert len(entities) == 3
        # Check visibility
        names_vis = {e.name: e.visibility for e in entities}
        assert names_vis["utils"] == "private"
        assert names_vis["helpers"] == "public"
        assert names_vis["tests"] == "private"

    def test_extract_use_statements_regex(self, processor):
        """Cover lines 228-255."""
        content = "use std::io;\nuse std::collections::HashMap;\nuse crate::my_mod;"
        file_path = Path("test.rs")
        entities = processor._extract_use_statements_regex(content, file_path)

        assert len(entities) == 3
        names = {e.name for e in entities}
        assert "std::io" in names
        assert "std::collections::HashMap" in names
        assert "crate::my_mod" in names

    def test_extract_structs_regex(self, processor):
        """Cover lines 257-283."""
        content = "pub struct Point {\n    x: f64,\n}\nstruct Internal;"
        file_path = Path("test.rs")
        entities = processor._extract_structs_regex(content, file_path)

        assert len(entities) == 2
        names = {e.name for e in entities}
        assert "Point" in names
        assert "Internal" in names

    def test_extract_enums_regex(self, processor):
        """Cover lines 285-311."""
        content = "pub enum Color {\n    Red,\n    Blue,\n}\nenum Shape { Circle, Square }"
        file_path = Path("test.rs")
        entities = processor._extract_enums_regex(content, file_path)

        assert len(entities) == 2
        names = {e.name for e in entities}
        assert "Color" in names
        assert "Shape" in names

    def test_extract_traits_regex(self, processor):
        """Cover lines 313-339."""
        content = "pub trait Drawable {\n    fn draw(&self);\n}\ntrait Printable {}"
        file_path = Path("test.rs")
        entities = processor._extract_traits_regex(content, file_path)

        assert len(entities) == 2
        names_vis = {e.name: e.visibility for e in entities}
        assert names_vis["Drawable"] == "public"
        assert names_vis["Printable"] == "private"

    def test_extract_functions_regex(self, processor):
        """Cover lines 341-370."""
        content = (
            "pub fn process(x: i32) -> i32 { x }\n"
            "pub async fn fetch_data() -> Result<()> { Ok(()) }\n"
            "fn helper(y: &str) -> String { y.to_string() }\n"
            "fn main() {}\n"  # Should be skipped
            "fn new() {}\n"  # Should be skipped
        )
        file_path = Path("test.rs")
        entities = processor._extract_functions_regex(content, file_path)

        names = {e.name for e in entities}
        # main and new should be skipped
        assert "main" not in names
        assert "new" not in names
        assert "process" in names
        assert "fetch_data" in names
        assert "helper" in names

        # Check async detection
        async_entities = [e for e in entities if getattr(e, "is_async", False)]
        assert len(async_entities) >= 1

    def test_extract_impl_blocks_regex(self, processor):
        """Cover lines 372-433."""
        content = (
            "impl Display for Point {\n    fn fmt(&self) {}\n}\n"
            "impl Point {\n    fn new() {}\n}\n"
            "impl<T> Clone for MyStruct<T> {}\n"
        )
        file_path = Path("test.rs")
        entities = processor._extract_impl_blocks_regex(content, file_path)

        assert len(entities) >= 2
        # Check trait impl
        trait_impls = [
            e for e in entities if e.properties and e.properties.get("impl_type") == "trait"
        ]
        assert len(trait_impls) >= 1

        # Check inherent impl
        inherent_impls = [
            e for e in entities if e.properties and e.properties.get("impl_type") == "inherent"
        ]
        assert len(inherent_impls) >= 1

    def test_extract_macros_regex(self, processor):
        """Cover lines 435-461."""
        content = (
            "macro_rules! vec_of {\n    ($($x:expr),*) => { vec![$($x),*] };\n}\n"
            'macro_rules! debug_print {\n    ($x:expr) => { println!("{:?}", $x) };\n}\n'
        )
        file_path = Path("test.rs")
        entities = processor._extract_macros_regex(content, file_path)

        assert len(entities) == 2
        names = {e.name for e in entities}
        assert "vec_of" in names
        assert "debug_print" in names

    @pytest.mark.asyncio
    async def test_parse_file_uses_regex_fallback(self, processor):
        """Cover lines 156-169 - parse_file falls back to regex when tree-sitter unavailable."""
        content = "pub fn add(a: i32, b: i32) -> i32 { a + b }"
        file_path = Path("add.rs")

        # Force regex fallback
        processor.parser = None
        result = await processor.parse_file(file_path, content)
        assert result.success is True
        assert result.parser_used.value == "regex_fallback"

    @pytest.mark.asyncio
    async def test_initialize_impl_without_tree_sitter(self, processor):
        """Cover lines 148-154 - initialization without tree-sitter."""
        processor.parser = None
        await processor._initialize_impl()
        # Should not raise


class TestRustDependencyAnalyzerCoverage:
    """Test Rust dependency analyzer regex/fallback paths."""

    @pytest.fixture
    def analyzer(self):
        from processing.languages.rust.rust_dependency_analyzer import (
            RustDependencyAnalyzer,
        )

        return RustDependencyAnalyzer()

    def test_extract_use_regex(self, analyzer):
        """Cover regex-based use extraction."""
        content = "use std::io;\nuse std::collections::HashMap;"
        file_path = Path("test.rs")
        entities = analyzer.extract_use_statements_regex(content, file_path)
        assert len(entities) == 2

    def test_analyze_dependency_graph(self, analyzer):
        """Cover analyze_dependency_graph method."""
        entities = [
            CoreCodeEntity(
                id="use:std::io:test.rs",
                name="std::io",
                type=EntityType.IMPORT,
                language="rust",
                file_path="test.rs",
                line_number=1,
                properties={"use_path": "std::io"},
            ),
            CoreCodeEntity(
                id="fn:process:test.rs:0",
                name="process",
                type=EntityType.FUNCTION,
                language="rust",
                file_path="test.rs",
                line_number=5,
            ),
        ]
        result = analyzer.analyze_dependency_graph(entities)
        assert isinstance(result, dict)
        assert "total_imports" in result

    def test_fallback_tree_sitter_methods(self, analyzer):
        """Cover fallback methods when tree-sitter is unavailable (lines 382-414)."""
        file_path = Path("test.rs")
        entities = [
            CoreCodeEntity(
                id="fn:test:test.rs:0",
                name="test",
                type=EntityType.FUNCTION,
                language="rust",
                file_path="test.rs",
                line_number=1,
            ),
        ]

        # These should return empty lists when tree-sitter is unavailable
        result1 = analyzer.extract_use_declarations_tree_sitter(None, file_path, b"")
        assert result1 == []

        result2 = analyzer.extract_use_relationships_tree_sitter(None, entities, file_path)
        assert result2 == []

        result3 = analyzer.extract_function_calls_tree_sitter(None, entities, file_path)
        assert result3 == []

        result4 = analyzer._find_containing_function(MagicMock(), entities)
        assert result4 is None

        loc = analyzer._node_to_location(MagicMock(), file_path)
        assert loc.file_path == str(file_path)

    def test_extract_use_statements_regex(self, analyzer):
        """Cover extract_use_statements_regex."""
        content = "use std::io;\nuse std::collections::HashMap;"
        file_path = Path("test.rs")
        entities = analyzer.extract_use_statements_regex(content, file_path)
        assert len(entities) == 2

    def test_extract_external_dependencies(self, analyzer):
        """Cover extract_external_dependencies."""
        entities = [
            CoreCodeEntity(
                id="use:serde::Serialize:test.rs",
                name="serde::Serialize",
                type=EntityType.IMPORT,
                language="rust",
                file_path="test.rs",
                line_number=1,
                properties={"use_path": "serde::Serialize"},
            ),
            CoreCodeEntity(
                id="use:std::io:test.rs",
                name="std::io",
                type=EntityType.IMPORT,
                language="rust",
                file_path="test.rs",
                line_number=2,
                properties={"use_path": "std::io"},
            ),
        ]
        result = analyzer.extract_external_dependencies(entities)
        assert "serde" in result
        assert "std" not in result  # std is not external

    def test_extract_internal_dependencies(self, analyzer):
        """Cover extract_internal_dependencies."""
        entities = [
            CoreCodeEntity(
                id="use:crate::utils:test.rs",
                name="crate::utils",
                type=EntityType.IMPORT,
                language="rust",
                file_path="test.rs",
                line_number=1,
                properties={"use_path": "crate::utils"},
            ),
        ]
        result = analyzer.extract_internal_dependencies(entities)
        assert len(result) >= 1
        paths = [d["path"] for d in result]
        assert "crate::utils" in paths

    def test_parse_use_statement(self, analyzer):
        """Cover parse_use_statement."""
        result = analyzer.parse_use_statement("use std::collections::HashMap;")
        assert result["is_std"] is True
        assert result["base_path"] == "std::collections::HashMap"
        assert "HashMap" in result["imported_items"]

    def test_suggest_import_optimizations(self, analyzer):
        """Cover suggest_import_optimizations."""
        entities = [
            CoreCodeEntity(
                id="use:std::io:test.rs",
                name="std::io",
                type=EntityType.IMPORT,
                language="rust",
                file_path="test.rs",
                line_number=1,
            ),
        ]
        result = analyzer.suggest_import_optimizations(entities)
        assert isinstance(result, list)
        # With only 1 import, no optimization suggestions expected
        for suggestion in result:
            assert isinstance(suggestion, dict)
            assert "type" in suggestion
            assert "message" in suggestion

    def test_get_dependency_complexity_score(self, analyzer):
        """Cover get_dependency_complexity_score."""
        entities = [
            CoreCodeEntity(
                id="use:std::io:test.rs",
                name="std::io",
                type=EntityType.IMPORT,
                language="rust",
                file_path="test.rs",
                line_number=1,
            ),
        ]
        score = analyzer.get_dependency_complexity_score(entities)
        assert isinstance(score, (int, float))
        assert score >= 0
