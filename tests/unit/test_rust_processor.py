"""Unit tests for processing.languages.rust.rust_processor.RustProcessor."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.core_models import (
    EntityType,
)
from processing.languages.base import LanguageConfig, ParseResult, ParserType
from processing.languages.rust.rust_config import RUST_CONFIG
from processing.languages.rust.rust_processor import TREE_SITTER_AVAILABLE, RustProcessor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def processor():
    """Create a RustProcessor instance."""
    return RustProcessor()


@pytest.fixture
def sample_file_path():
    return Path("src/main.rs")


@pytest.fixture
def simple_rust_code():
    return """\
use std::io;
use std::collections::HashMap;

pub mod utils;
mod internal;

pub struct MyStruct {
    pub name: String,
    age: u32,
}

struct PrivateStruct {
    data: Vec<u8>,
}

pub enum Color {
    Red,
    Green,
    Blue,
}

enum Direction {
    Up,
    Down,
}

pub trait Drawable {
    fn draw(&self);
}

trait Printable {
    fn print(&self);
}

pub fn calculate(x: i32, y: i32) -> i32 {
    x + y
}

fn helper(s: &str) -> String {
    s.to_string()
}

pub async fn fetch_data(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    Ok(String::new())
}

impl MyStruct {
    pub fn new(name: String, age: u32) -> Self {
        Self { name, age }
    }
}

impl Drawable for MyStruct {
    fn draw(&self) {
        println!("{}", self.name);
    }
}

macro_rules! my_macro {
    ($x:expr) => {
        println!("{}", $x);
    };
}
"""


@pytest.fixture
def minimal_rust_code():
    return "fn helper() {}\n"


@pytest.fixture
def empty_rust_code():
    return ""


# ---------------------------------------------------------------------------
# Constructor / init tests
# ---------------------------------------------------------------------------


class TestRustProcessorInit:
    def test_constructor_default_config(self, processor):
        assert processor.config == RUST_CONFIG
        assert processor.config.name == "rust"

    def test_constructor_custom_config(self):
        custom = LanguageConfig(
            name="rust_custom",
            display_name="Custom Rust",
            extensions=[".rs"],
            parser_type=ParserType.REGEX_FALLBACK,
            features=RUST_CONFIG.features,
        )
        proc = RustProcessor(config=custom)
        assert proc.config.name == "rust_custom"

    async def test_initialize_impl(self, processor):
        """_initialize_impl should run without error."""
        await processor._initialize_impl()

    def test_class_attributes(self, processor):
        assert processor.STRUCT_NODE_TYPES == {"struct_item"}
        assert processor.ENUM_NODE_TYPES == {"enum_item"}
        assert processor.TRAIT_NODE_TYPES == {"trait_item"}
        assert processor.IMPL_NODE_TYPES == {"impl_item"}
        assert processor.FUNCTION_NODE_TYPES == {"function_item"}
        assert processor.MOD_NODE_TYPES == {"mod_item"}
        assert processor.USE_NODE_TYPES == {"use_declaration"}
        assert processor.CONST_NODE_TYPES == {"const_item"}
        assert processor.STATIC_NODE_TYPES == {"static_item"}
        assert processor.TYPE_NODE_TYPES == {"type_item"}
        assert processor.MACRO_NODE_TYPES == {"macro_definition", "macro_invocation"}


# ---------------------------------------------------------------------------
# parse_file tests
# ---------------------------------------------------------------------------


class TestParseFile:
    async def test_parse_file_returns_parse_result(
        self, processor, simple_rust_code, sample_file_path
    ):
        result = await processor.parse_file(sample_file_path, simple_rust_code)
        assert isinstance(result, ParseResult)
        assert result.success is True
        assert result.language == "rust"
        assert result.file_path == str(sample_file_path)

    async def test_parse_file_counts_lines(self, processor, simple_rust_code, sample_file_path):
        result = await processor.parse_file(sample_file_path, simple_rust_code)
        expected_lines = len(simple_rust_code.splitlines())
        assert result.lines_of_code == expected_lines

    async def test_parse_file_empty_content(self, processor, sample_file_path):
        result = await processor.parse_file(sample_file_path, "")
        assert result.success is True
        assert result.entities == []

    async def test_parse_file_extracts_entities(
        self, processor, simple_rust_code, sample_file_path
    ):
        result = await processor.parse_file(sample_file_path, simple_rust_code)
        assert len(result.entities) > 0
        entity_types = {e.type for e in result.entities}
        # Should find at least structs, enums, traits, functions, modules, imports
        assert EntityType.STRUCT in entity_types
        assert EntityType.ENUM in entity_types
        assert EntityType.FUNCTION in entity_types


# ---------------------------------------------------------------------------
# Regex extraction tests
# ---------------------------------------------------------------------------


class TestExtractModulesRegex:
    def test_extract_public_module(self, processor, sample_file_path):
        content = "pub mod utils;"
        entities = processor._extract_modules_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "utils"
        assert entities[0].type == EntityType.MODULE
        assert entities[0].visibility == "public"
        assert entities[0].properties["is_public"] is True

    def test_extract_private_module(self, processor, sample_file_path):
        content = "mod internal;"
        entities = processor._extract_modules_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "internal"
        assert entities[0].visibility == "private"
        assert entities[0].properties["is_public"] is False

    def test_extract_multiple_modules(self, processor, sample_file_path):
        content = "pub mod utils;\nmod internal;\npub mod config;"
        entities = processor._extract_modules_regex(content, sample_file_path)
        assert len(entities) == 3
        names = [e.name for e in entities]
        assert "utils" in names
        assert "internal" in names
        assert "config" in names

    def test_extract_no_modules(self, processor, sample_file_path):
        content = "fn main() {}"
        entities = processor._extract_modules_regex(content, sample_file_path)
        assert entities == []

    def test_module_line_number(self, processor, sample_file_path):
        content = "// comment\n\npub mod utils;"
        entities = processor._extract_modules_regex(content, sample_file_path)
        assert entities[0].line_number == 3

    def test_module_language(self, processor, sample_file_path):
        content = "mod foo;"
        entities = processor._extract_modules_regex(content, sample_file_path)
        assert entities[0].language == "rust"


class TestExtractUseStatementsRegex:
    def test_extract_simple_use(self, processor, sample_file_path):
        content = "use std::io;"
        entities = processor._extract_use_statements_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "std::io"
        assert entities[0].type == EntityType.IMPORT
        assert entities[0].properties["use_path"] == "std::io"

    def test_extract_nested_use(self, processor, sample_file_path):
        content = "use std::collections::HashMap;"
        entities = processor._extract_use_statements_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "std::collections::HashMap"

    def test_extract_multiple_uses(self, processor, sample_file_path):
        content = "use std::io;\nuse std::fs;\nuse serde::Deserialize;"
        entities = processor._extract_use_statements_regex(content, sample_file_path)
        assert len(entities) == 3

    def test_no_use_statements(self, processor, sample_file_path):
        content = "fn main() {}"
        entities = processor._extract_use_statements_regex(content, sample_file_path)
        assert entities == []

    def test_use_line_number(self, processor, sample_file_path):
        content = "// header\nuse std::io;"
        entities = processor._extract_use_statements_regex(content, sample_file_path)
        assert entities[0].line_number == 2

    def test_use_id_format(self, processor, sample_file_path):
        content = "use std::io;"
        entities = processor._extract_use_statements_regex(content, sample_file_path)
        assert entities[0].id == f"use:std::io:{sample_file_path}"


class TestExtractStructsRegex:
    def test_extract_public_struct(self, processor, sample_file_path):
        content = "pub struct MyStruct { name: String }"
        entities = processor._extract_structs_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "MyStruct"
        assert entities[0].type == EntityType.STRUCT

    def test_extract_private_struct(self, processor, sample_file_path):
        content = "struct InternalStruct { data: Vec<u8> }"
        entities = processor._extract_structs_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "InternalStruct"

    def test_extract_multiple_structs(self, processor, sample_file_path):
        content = "pub struct A {}\nstruct B {}\npub struct C {}"
        entities = processor._extract_structs_regex(content, sample_file_path)
        assert len(entities) == 3
        names = {e.name for e in entities}
        assert names == {"A", "B", "C"}

    def test_struct_id_format(self, processor, sample_file_path):
        content = "struct Foo {}"
        entities = processor._extract_structs_regex(content, sample_file_path)
        assert entities[0].id == f"struct:Foo:{sample_file_path}"

    def test_struct_line_number(self, processor, sample_file_path):
        content = "\n\nstruct Late {}"
        entities = processor._extract_structs_regex(content, sample_file_path)
        assert entities[0].line_number == 3

    def test_no_structs(self, processor, sample_file_path):
        content = "fn main() {}"
        entities = processor._extract_structs_regex(content, sample_file_path)
        assert entities == []


class TestExtractEnumsRegex:
    def test_extract_public_enum(self, processor, sample_file_path):
        content = "pub enum Color { Red, Green, Blue }"
        entities = processor._extract_enums_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "Color"
        assert entities[0].type == EntityType.ENUM

    def test_extract_private_enum(self, processor, sample_file_path):
        content = "enum Direction { Up, Down }"
        entities = processor._extract_enums_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "Direction"

    def test_extract_multiple_enums(self, processor, sample_file_path):
        content = "pub enum A { X }\nenum B { Y }"
        entities = processor._extract_enums_regex(content, sample_file_path)
        assert len(entities) == 2

    def test_enum_id_format(self, processor, sample_file_path):
        content = "enum Status { Ok, Err }"
        entities = processor._extract_enums_regex(content, sample_file_path)
        assert entities[0].id == f"enum:Status:{sample_file_path}"

    def test_no_enums(self, processor, sample_file_path):
        content = "struct Foo {}"
        entities = processor._extract_enums_regex(content, sample_file_path)
        assert entities == []


class TestExtractTraitsRegex:
    def test_extract_public_trait(self, processor, sample_file_path):
        content = "pub trait Drawable { fn draw(&self); }"
        entities = processor._extract_traits_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "Drawable"
        assert entities[0].type == EntityType.TRAIT
        assert entities[0].visibility == "public"

    def test_extract_private_trait(self, processor, sample_file_path):
        content = "trait Printable { fn print(&self); }"
        entities = processor._extract_traits_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "Printable"
        assert entities[0].visibility == "private"

    def test_extract_multiple_traits(self, processor, sample_file_path):
        content = "pub trait A {}\ntrait B {}\npub trait C {}"
        entities = processor._extract_traits_regex(content, sample_file_path)
        assert len(entities) == 3

    def test_trait_id_format(self, processor, sample_file_path):
        content = "trait Foo {}"
        entities = processor._extract_traits_regex(content, sample_file_path)
        assert entities[0].id == f"trait:Foo:{sample_file_path}"

    def test_no_traits(self, processor, sample_file_path):
        content = "fn main() {}"
        entities = processor._extract_traits_regex(content, sample_file_path)
        assert entities == []


class TestExtractFunctionsRegex:
    def test_extract_public_function(self, processor, sample_file_path):
        content = "pub fn calculate(x: i32) -> i32 { x }"
        entities = processor._extract_functions_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "calculate"
        assert entities[0].type == EntityType.FUNCTION
        assert entities[0].visibility == "public"

    def test_extract_private_function(self, processor, sample_file_path):
        content = "fn helper(s: &str) -> String { s.to_string() }"
        entities = processor._extract_functions_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "helper"
        assert entities[0].visibility == "private"

    def test_extract_async_function(self, processor, sample_file_path):
        content = (
            "pub async fn fetch_data(url: &str) -> Result<String, Error> { Ok(String::new()) }"
        )
        entities = processor._extract_functions_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "fetch_data"
        assert entities[0].is_async is True

    def test_extract_sync_function(self, processor, sample_file_path):
        content = "fn sync_work(x: i32) -> i32 { x }"
        entities = processor._extract_functions_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].is_async is False

    def test_skip_main(self, processor, sample_file_path):
        content = "fn main() {}"
        entities = processor._extract_functions_regex(content, sample_file_path)
        assert len(entities) == 0

    def test_skip_new(self, processor, sample_file_path):
        content = "fn new() -> Self { Self {} }"
        entities = processor._extract_functions_regex(content, sample_file_path)
        assert len(entities) == 0

    def test_skip_default(self, processor, sample_file_path):
        content = "fn default() -> Self { Self {} }"
        entities = processor._extract_functions_regex(content, sample_file_path)
        assert len(entities) == 0

    def test_extract_multiple_functions(self, processor, sample_file_path):
        content = "pub fn foo(x: i32) {}\nfn bar(y: i32) {}\npub async fn baz(z: i32) {}"
        entities = processor._extract_functions_regex(content, sample_file_path)
        assert len(entities) == 3
        names = {e.name for e in entities}
        assert names == {"foo", "bar", "baz"}

    def test_no_functions(self, processor, sample_file_path):
        content = "struct Foo {}"
        entities = processor._extract_functions_regex(content, sample_file_path)
        assert entities == []

    def test_function_line_number(self, processor, sample_file_path):
        content = "// comment\n\npub fn third_line(x: i32) {}"
        entities = processor._extract_functions_regex(content, sample_file_path)
        assert entities[0].line_number == 3


class TestExtractImplBlocksRegex:
    def test_extract_inherent_impl(self, processor, sample_file_path):
        content = "impl MyStruct { fn do_thing(&self) {} }"
        entities = processor._extract_impl_blocks_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "impl MyStruct"
        assert entities[0].type == EntityType.CLASS
        assert entities[0].properties["impl_type"] == "inherent"
        assert entities[0].properties["type_name"] == "MyStruct"

    def test_extract_trait_impl(self, processor, sample_file_path):
        content = "impl Drawable for MyStruct { fn draw(&self) {} }"
        entities = processor._extract_impl_blocks_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "Drawable for MyStruct"
        assert entities[0].properties["impl_type"] == "trait"
        assert entities[0].properties["trait_name"] == "Drawable"
        assert entities[0].properties["type_name"] == "MyStruct"

    def test_extract_generic_impl(self, processor, sample_file_path):
        content = "impl<T> Container<T> { fn contents(&self) {} }"
        entities = processor._extract_impl_blocks_regex(content, sample_file_path)
        # Should match something (the regex allows generics)
        assert len(entities) >= 1

    def test_extract_multiple_impl_blocks(self, processor, sample_file_path):
        content = "impl Foo {}\nimpl Bar for Foo {}\nimpl Baz {}"
        entities = processor._extract_impl_blocks_regex(content, sample_file_path)
        assert len(entities) == 3

    def test_no_impl_blocks(self, processor, sample_file_path):
        content = "struct Foo {}"
        entities = processor._extract_impl_blocks_regex(content, sample_file_path)
        assert entities == []

    def test_impl_visibility(self, processor, sample_file_path):
        content = "impl MyStruct {}"
        entities = processor._extract_impl_blocks_regex(content, sample_file_path)
        assert entities[0].visibility == "private"


class TestExtractMacrosRegex:
    def test_extract_macro(self, processor, sample_file_path):
        content = 'macro_rules! my_macro { ($x:expr) => { println!("{}", $x); }; }'
        entities = processor._extract_macros_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "my_macro"
        assert entities[0].type == EntityType.MACRO
        assert entities[0].visibility == "private"

    def test_extract_multiple_macros(self, processor, sample_file_path):
        content = "macro_rules! mac_a { () => {}; }\nmacro_rules! mac_b { () => {}; }"
        entities = processor._extract_macros_regex(content, sample_file_path)
        assert len(entities) == 2
        names = {e.name for e in entities}
        assert names == {"mac_a", "mac_b"}

    def test_no_macros(self, processor, sample_file_path):
        content = "fn main() {}"
        entities = processor._extract_macros_regex(content, sample_file_path)
        assert entities == []

    def test_macro_id_format(self, processor, sample_file_path):
        content = "macro_rules! test_mac { () => {}; }"
        entities = processor._extract_macros_regex(content, sample_file_path)
        assert entities[0].id == f"macro:test_mac:{sample_file_path}"

    def test_macro_line_number(self, processor, sample_file_path):
        content = "// line 1\n// line 2\nmacro_rules! late { () => {}; }"
        entities = processor._extract_macros_regex(content, sample_file_path)
        assert entities[0].line_number == 3


# ---------------------------------------------------------------------------
# _regex_parse integration test
# ---------------------------------------------------------------------------


class TestRegexParse:
    async def test_regex_parse_success(self, processor, simple_rust_code, sample_file_path):
        result = await processor._regex_parse(sample_file_path, simple_rust_code)
        assert result.success is True
        assert result.language == "rust"
        assert result.parser_used == ParserType.REGEX_FALLBACK
        assert result.file_path == str(sample_file_path)
        assert result.ast is None
        assert result.complexity_score == 1.0
        assert result.parse_time_ms == 0.0

    async def test_regex_parse_finds_all_entity_types(
        self, processor, simple_rust_code, sample_file_path
    ):
        result = await processor._regex_parse(sample_file_path, simple_rust_code)
        entity_types = {e.type for e in result.entities}
        assert EntityType.MODULE in entity_types
        assert EntityType.IMPORT in entity_types
        assert EntityType.STRUCT in entity_types
        assert EntityType.ENUM in entity_types
        assert EntityType.TRAIT in entity_types
        assert EntityType.FUNCTION in entity_types
        assert EntityType.CLASS in entity_types  # impl blocks
        assert EntityType.MACRO in entity_types

    async def test_regex_parse_entity_count(self, processor, simple_rust_code, sample_file_path):
        result = await processor._regex_parse(sample_file_path, simple_rust_code)
        # Count expected: 2 modules, 2 uses, 2 structs, 2 enums, 2 traits,
        # some functions (main/new/default skipped), impl blocks, 1 macro
        assert len(result.entities) > 10  # broad check

    async def test_regex_parse_empty_content(self, processor, sample_file_path):
        result = await processor._regex_parse(sample_file_path, "")
        assert result.success is True
        assert result.entities == []
        assert result.relationships == []
        assert result.lines_of_code == 0

    async def test_regex_parse_lines_of_code(self, processor, sample_file_path):
        content = "line1\nline2\nline3\n"
        result = await processor._regex_parse(sample_file_path, content)
        assert result.lines_of_code == 3

    async def test_regex_parse_relationships_empty(
        self, processor, simple_rust_code, sample_file_path
    ):
        result = await processor._regex_parse(sample_file_path, simple_rust_code)
        # _regex_parse creates an empty relationship list
        assert result.relationships == []


# ---------------------------------------------------------------------------
# extract_entities / extract_relationships fallback tests
# ---------------------------------------------------------------------------


class TestExtractEntitiesAndRelationships:
    async def test_extract_entities_via_parse_file(
        self, processor, simple_rust_code, sample_file_path
    ):
        """parse_file populates entities regardless of tree-sitter availability."""
        result = await processor.parse_file(sample_file_path, simple_rust_code)
        assert isinstance(result.entities, list)
        assert len(result.entities) > 0

    async def test_extract_relationships_via_parse_file(
        self, processor, simple_rust_code, sample_file_path
    ):
        """parse_file populates relationships list."""
        result = await processor.parse_file(sample_file_path, simple_rust_code)
        assert isinstance(result.relationships, list)

    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")
    async def test_extract_entities_tree_sitter_with_mock_tree(self, processor, sample_file_path):
        """extract_entities works with a mock tree when tree-sitter is available."""
        mock_tree = MagicMock()
        mock_tree.text = b"pub struct Foo {}\npub fn bar() {}"
        mock_tree.root_node = MagicMock()
        mock_tree.root_node.children = []

        # Patch _find_nodes_by_type to return empty lists so it doesn't crash
        with patch.object(processor, "_find_nodes_by_type", return_value=[]):
            entities = await processor.extract_entities(mock_tree, sample_file_path)
            assert isinstance(entities, list)

    @pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")
    async def test_extract_relationships_tree_sitter_with_mock_tree(
        self, processor, sample_file_path
    ):
        """extract_relationships works with a mock tree when tree-sitter is available."""
        mock_tree = MagicMock()
        mock_tree.text = b"impl Foo {}"
        mock_tree.root_node = MagicMock()
        mock_tree.root_node.children = []

        with patch.object(processor, "_find_nodes_by_type", return_value=[]):
            rels = await processor.extract_relationships(mock_tree, [], sample_file_path)
            assert isinstance(rels, list)


# ---------------------------------------------------------------------------
# Edge cases and combined scenarios
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_struct_with_generic_params(self, processor, sample_file_path):
        content = "pub struct Container<T> { data: Vec<T> }"
        entities = processor._extract_structs_regex(content, sample_file_path)
        # The regex captures "Container" from "struct Container<T>"
        found = [e for e in entities if e.name == "Container"]
        assert len(found) == 1

    def test_function_with_lifetime_not_matched_by_regex(self, processor, sample_file_path):
        # The regex requires `fn name(` with no generic params between name and paren,
        # so lifetime-parameterized functions are not matched by the regex fallback.
        content = "fn longest<'a>(x: &'a str, y: &'a str) -> &'a str { x }"
        entities = processor._extract_functions_regex(content, sample_file_path)
        assert len(entities) == 0

    def test_function_without_generics(self, processor, sample_file_path):
        content = "fn longest(x: &str, y: &str) -> &str { x }"
        entities = processor._extract_functions_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "longest"

    def test_multiline_struct(self, processor, sample_file_path):
        content = "pub struct Point {\n    x: f64,\n    y: f64,\n}"
        entities = processor._extract_structs_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "Point"

    def test_enum_with_data(self, processor, sample_file_path):
        content = (
            "pub enum Message {\n    Quit,\n    Move { x: i32, y: i32 },\n    Write(String),\n}"
        )
        entities = processor._extract_enums_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "Message"

    def test_trait_with_default_impl(self, processor, sample_file_path):
        content = 'pub trait Summary {\n    fn summarize(&self) -> String {\n        String::from("...")\n    }\n}'
        entities = processor._extract_traits_regex(content, sample_file_path)
        assert len(entities) == 1
        assert entities[0].name == "Summary"

    def test_use_with_braces(self, processor, sample_file_path):
        content = "use std::io::{self, Read, Write};"
        entities = processor._extract_use_statements_regex(content, sample_file_path)
        assert len(entities) == 1
        assert "std::io::" in entities[0].name

    def test_impl_with_trait_for_generic(self, processor, sample_file_path):
        content = "impl<T> Display for Container<T> { fn fmt(&self) {} }"
        entities = processor._extract_impl_blocks_regex(content, sample_file_path)
        # Should find at least one impl block
        assert len(entities) >= 1

    async def test_full_parse_comprehensive(self, processor, sample_file_path):
        """Full parse of comprehensive Rust code finds all major entity types."""
        code = """\
use std::fmt;

pub mod network;

pub struct Server {
    port: u16,
}

enum Status {
    Running,
    Stopped,
}

pub trait Service {
    fn start(&self);
}

pub fn run_server(port: u16) {}

async fn background_task(id: u32) {}

impl Server {
    fn bind(port: u16) -> Self { Self { port } }
}

impl Service for Server {
    fn start(&self) {}
}

macro_rules! log {
    ($msg:expr) => { println!("{}", $msg); };
}
"""
        result = await processor.parse_file(sample_file_path, code)
        assert result.success is True
        entity_names = {e.name for e in result.entities}
        assert "Server" in entity_names
        assert "Status" in entity_names
        assert "Service" in entity_names
        assert "network" in entity_names
        assert "std::fmt" in entity_names

    def test_entity_file_path_set_correctly(self, processor, sample_file_path):
        content = "pub struct Foo {}"
        entities = processor._extract_structs_regex(content, sample_file_path)
        assert entities[0].file_path == str(sample_file_path)

    def test_entity_language_set_correctly(self, processor, sample_file_path):
        content = "pub enum Bar { X }"
        entities = processor._extract_enums_regex(content, sample_file_path)
        assert entities[0].language == "rust"

    async def test_parse_file_with_only_comments(self, processor, sample_file_path):
        content = "// This is a comment\n// Another comment\n"
        result = await processor.parse_file(sample_file_path, content)
        assert result.success is True
        assert result.entities == []

    async def test_parse_file_single_function(self, processor, sample_file_path):
        content = 'pub fn greet(name: &str) -> String { format!("Hello, {}!", name) }'
        result = await processor.parse_file(sample_file_path, content)
        assert result.success is True
        fns = [e for e in result.entities if e.type == EntityType.FUNCTION]
        assert len(fns) == 1
        assert fns[0].name == "greet"


# ---------------------------------------------------------------------------
# Config checks
# ---------------------------------------------------------------------------


class TestRustConfig:
    def test_config_name(self):
        assert RUST_CONFIG.name == "rust"

    def test_config_extensions(self):
        assert ".rs" in RUST_CONFIG.extensions

    def test_config_parser_type(self):
        assert RUST_CONFIG.parser_type == ParserType.TREE_SITTER

    def test_config_fallback(self):
        assert ParserType.REGEX_FALLBACK in RUST_CONFIG.fallback_parsers

    def test_config_features(self):
        f = RUST_CONFIG.features
        assert f.static_typing is True
        assert f.generics is True
        assert f.async_support is True
        assert f.macros is True
        assert f.closures is True
        assert f.garbage_collected is False
        assert f.manual_memory is True


# ---------------------------------------------------------------------------
# Tree-sitter parsing tests (call internal methods directly)
# ---------------------------------------------------------------------------


SAMPLE_RUST_TS = """\
use std::io;
use std::collections::HashMap;

pub mod utils;
mod internal;

pub struct MyStruct {
    pub name: String,
    age: u32,
}

struct PrivateStruct {
    data: Vec<u8>,
}

pub enum Color {
    Red,
    Green,
    Blue,
}

enum Direction {
    Up,
    Down,
}

pub trait Drawable {
    fn draw(&self);
}

trait Printable {
    fn print_it(&self);
}

pub fn calculate(x: i32, y: i32) -> i32 {
    x + y
}

fn helper(s: &str) -> String {
    s.to_string()
}

pub async fn fetch_data(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    Ok(String::new())
}

impl MyStruct {
    pub fn create(name: String, age: u32) -> Self {
        Self { name, age }
    }
}

impl Drawable for MyStruct {
    fn draw(&self) {
        println!("{}", self.name);
    }
}

const MAX_SIZE: u32 = 1024;
static COUNTER: u32 = 0;
"""


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")
class TestRustTreeSitterParsing:
    """Tests exercising tree-sitter parsing on internal methods directly."""

    @pytest.fixture
    async def ts_processor(self):
        p = RustProcessor()
        await p._initialize_impl()
        return p

    @pytest.fixture
    async def rust_tree(self, ts_processor):
        """Parse sample Rust code into a tree-sitter Tree."""
        tree = ts_processor.parser.parse(SAMPLE_RUST_TS.encode("utf-8"))
        return tree

    @pytest.fixture
    def content_bytes(self):
        return SAMPLE_RUST_TS.encode("utf-8")

    @pytest.fixture
    def sample_path(self):
        return Path("src/lib.rs")

    async def test_extract_modules(self, ts_processor, rust_tree, sample_path, content_bytes):
        entities = ts_processor._extract_modules(rust_tree, sample_path, content_bytes)
        names = {e.name for e in entities}
        assert "utils" in names
        assert "internal" in names
        assert all(e.type == EntityType.MODULE for e in entities)

    async def test_extract_modules_visibility(
        self, ts_processor, rust_tree, sample_path, content_bytes
    ):
        entities = ts_processor._extract_modules(rust_tree, sample_path, content_bytes)
        utils = next(e for e in entities if e.name == "utils")
        internal = next(e for e in entities if e.name == "internal")
        assert utils.visibility == "public"
        assert internal.visibility == "private"

    async def test_extract_use_declarations(
        self, ts_processor, rust_tree, sample_path, content_bytes
    ):
        entities = ts_processor._extract_use_declarations(rust_tree, sample_path, content_bytes)
        assert len(entities) >= 2
        assert all(e.type == EntityType.IMPORT for e in entities)

    async def test_extract_structs(self, ts_processor, rust_tree, sample_path, content_bytes):
        entities = ts_processor._extract_structs(rust_tree, sample_path, content_bytes)
        names = {e.name for e in entities}
        assert "MyStruct" in names
        assert "PrivateStruct" in names
        assert all(e.type == EntityType.STRUCT for e in entities)

    async def test_extract_structs_visibility(
        self, ts_processor, rust_tree, sample_path, content_bytes
    ):
        entities = ts_processor._extract_structs(rust_tree, sample_path, content_bytes)
        my_struct = next(e for e in entities if e.name == "MyStruct")
        priv_struct = next(e for e in entities if e.name == "PrivateStruct")
        assert my_struct.visibility == "public"
        assert priv_struct.visibility == "private"

    async def test_extract_enums(self, ts_processor, rust_tree, sample_path, content_bytes):
        entities = ts_processor._extract_enums(rust_tree, sample_path, content_bytes)
        names = {e.name for e in entities}
        assert "Color" in names
        assert "Direction" in names
        assert all(e.type == EntityType.ENUM for e in entities)

    async def test_extract_traits(self, ts_processor, rust_tree, sample_path, content_bytes):
        entities = ts_processor._extract_traits(rust_tree, sample_path, content_bytes)
        names = {e.name for e in entities}
        assert "Drawable" in names
        assert "Printable" in names
        assert all(e.type == EntityType.TRAIT for e in entities)

    async def test_extract_traits_visibility(
        self, ts_processor, rust_tree, sample_path, content_bytes
    ):
        entities = ts_processor._extract_traits(rust_tree, sample_path, content_bytes)
        drawable = next(e for e in entities if e.name == "Drawable")
        printable = next(e for e in entities if e.name == "Printable")
        assert drawable.visibility == "public"
        assert printable.visibility == "private"

    async def test_extract_functions(self, ts_processor, rust_tree, sample_path, content_bytes):
        entities = ts_processor._extract_functions(rust_tree, sample_path, content_bytes)
        names = {e.name for e in entities}
        assert "calculate" in names
        assert "helper" in names
        assert "fetch_data" in names
        assert all(e.type == EntityType.FUNCTION for e in entities)

    async def test_extract_functions_async_flag(
        self, ts_processor, rust_tree, sample_path, content_bytes
    ):
        entities = ts_processor._extract_functions(rust_tree, sample_path, content_bytes)
        fetch = next(e for e in entities if e.name == "fetch_data")
        calc = next(e for e in entities if e.name == "calculate")
        assert fetch.is_async is True
        assert calc.is_async is False

    async def test_extract_impl_blocks(self, ts_processor, rust_tree, sample_path, content_bytes):
        entities = ts_processor._extract_impl_blocks(rust_tree, sample_path, content_bytes)
        assert len(entities) >= 2
        trait_impls = [e for e in entities if e.properties.get("impl_type") == "trait"]
        inherent_impls = [e for e in entities if e.properties.get("impl_type") == "inherent"]
        assert len(trait_impls) >= 1
        assert len(inherent_impls) >= 1

    async def test_extract_constants(self, ts_processor, rust_tree, sample_path, content_bytes):
        entities = ts_processor._extract_constants(rust_tree, sample_path, content_bytes)
        names = {e.name for e in entities}
        assert "MAX_SIZE" in names
        const_entity = next(e for e in entities if e.name == "MAX_SIZE")
        assert const_entity.type == EntityType.CONSTANT

    async def test_extract_statics(self, ts_processor, rust_tree, sample_path, content_bytes):
        entities = ts_processor._extract_constants(rust_tree, sample_path, content_bytes)
        names = {e.name for e in entities}
        assert "COUNTER" in names
        static_entity = next(e for e in entities if e.name == "COUNTER")
        assert static_entity.type == EntityType.VARIABLE
        assert static_entity.is_static is True

    async def test_extract_visibility(self, ts_processor, rust_tree, sample_path, content_bytes):
        struct_nodes = ts_processor._find_nodes_by_type(rust_tree, ts_processor.STRUCT_NODE_TYPES)
        assert len(struct_nodes) >= 2
        for node in struct_nodes:
            vis = ts_processor._extract_visibility(node, content_bytes)
            assert vis in ("public", "private")

    async def test_entity_line_numbers(self, ts_processor, rust_tree, sample_path, content_bytes):
        entities = ts_processor._extract_structs(rust_tree, sample_path, content_bytes)
        for entity in entities:
            assert entity.line_number >= 1

    async def test_entity_file_path(self, ts_processor, rust_tree, sample_path, content_bytes):
        entities = ts_processor._extract_enums(rust_tree, sample_path, content_bytes)
        for entity in entities:
            assert entity.file_path == str(sample_path)

    async def test_entity_language(self, ts_processor, rust_tree, sample_path, content_bytes):
        entities = ts_processor._extract_traits(rust_tree, sample_path, content_bytes)
        for entity in entities:
            assert entity.language == "rust"

    async def test_extract_entities_all_types(
        self, ts_processor, rust_tree, sample_path, content_bytes
    ):
        """Comprehensive extraction covers all entity types."""
        entities = []
        entities.extend(ts_processor._extract_modules(rust_tree, sample_path, content_bytes))
        entities.extend(
            ts_processor._extract_use_declarations(rust_tree, sample_path, content_bytes)
        )
        entities.extend(ts_processor._extract_structs(rust_tree, sample_path, content_bytes))
        entities.extend(ts_processor._extract_enums(rust_tree, sample_path, content_bytes))
        entities.extend(ts_processor._extract_traits(rust_tree, sample_path, content_bytes))
        entities.extend(ts_processor._extract_functions(rust_tree, sample_path, content_bytes))
        entities.extend(ts_processor._extract_impl_blocks(rust_tree, sample_path, content_bytes))
        entities.extend(ts_processor._extract_constants(rust_tree, sample_path, content_bytes))
        assert len(entities) > 10
        entity_types = {e.type for e in entities}
        assert EntityType.MODULE in entity_types
        assert EntityType.IMPORT in entity_types
        assert EntityType.STRUCT in entity_types
        assert EntityType.ENUM in entity_types
        assert EntityType.TRAIT in entity_types
        assert EntityType.FUNCTION in entity_types
        assert EntityType.CLASS in entity_types  # impl blocks
        assert EntityType.CONSTANT in entity_types
