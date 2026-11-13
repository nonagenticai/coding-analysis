"""Coverage boost tests for RustProcessor and RustSyntaxParser.

Focuses on:
- _regex_parse: modules, use statements, structs, enums, traits, functions,
  impl blocks (trait + inherent), macros
- parse_file fallback dispatch
- extract_entities/relationships fallback
- RustSyntaxParser regex methods
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models.core_models import EntityType
from processing.languages.base import ParserType
from processing.languages.rust.rust_processor import RustProcessor
from processing.languages.rust.rust_syntax_parser import RustSyntaxParser


@pytest.fixture
def processor():
    return RustProcessor()


@pytest.fixture
def syntax_parser():
    return RustSyntaxParser()


RUST_FULL = """\
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


# ── RustProcessor._regex_parse ───────────────────────────────────────


class TestRustRegexParse:
    @pytest.mark.asyncio
    async def test_full_parse_success(self, processor):
        result = await processor._regex_parse(Path("main.rs"), RUST_FULL)
        assert result.success is True
        assert result.language == "rust"
        assert result.parser_used == ParserType.REGEX_FALLBACK

    @pytest.mark.asyncio
    async def test_extracts_modules(self, processor):
        result = await processor._regex_parse(Path("main.rs"), RUST_FULL)
        mods = [e for e in result.entities if e.type == EntityType.MODULE]
        mod_names = {m.name for m in mods}
        assert "utils" in mod_names
        assert "internal" in mod_names

    @pytest.mark.asyncio
    async def test_module_visibility(self, processor):
        result = await processor._regex_parse(Path("main.rs"), RUST_FULL)
        mods = [e for e in result.entities if e.type == EntityType.MODULE]
        utils = [m for m in mods if m.name == "utils"][0]
        internal = [m for m in mods if m.name == "internal"][0]
        assert utils.visibility == "public"
        assert internal.visibility == "private"

    @pytest.mark.asyncio
    async def test_extracts_use_statements(self, processor):
        result = await processor._regex_parse(Path("main.rs"), RUST_FULL)
        imports = [e for e in result.entities if e.type == EntityType.IMPORT]
        import_names = {i.name for i in imports}
        assert "std::io" in import_names
        assert "std::collections::HashMap" in import_names

    @pytest.mark.asyncio
    async def test_extracts_structs(self, processor):
        result = await processor._regex_parse(Path("main.rs"), RUST_FULL)
        structs = [e for e in result.entities if e.type == EntityType.STRUCT]
        struct_names = {s.name for s in structs}
        assert "MyStruct" in struct_names
        assert "PrivateStruct" in struct_names

    @pytest.mark.asyncio
    async def test_extracts_enums(self, processor):
        result = await processor._regex_parse(Path("main.rs"), RUST_FULL)
        enums = [e for e in result.entities if e.type == EntityType.ENUM]
        enum_names = {e.name for e in enums}
        assert "Color" in enum_names
        assert "Direction" in enum_names

    @pytest.mark.asyncio
    async def test_extracts_traits(self, processor):
        result = await processor._regex_parse(Path("main.rs"), RUST_FULL)
        traits = [e for e in result.entities if e.type == EntityType.TRAIT]
        trait_names = {t.name for t in traits}
        assert "Drawable" in trait_names
        assert "Printable" in trait_names

    @pytest.mark.asyncio
    async def test_trait_visibility(self, processor):
        result = await processor._regex_parse(Path("main.rs"), RUST_FULL)
        traits = [e for e in result.entities if e.type == EntityType.TRAIT]
        drawable = [t for t in traits if t.name == "Drawable"][0]
        printable = [t for t in traits if t.name == "Printable"][0]
        assert drawable.visibility == "public"
        assert printable.visibility == "private"

    @pytest.mark.asyncio
    async def test_extracts_functions(self, processor):
        result = await processor._regex_parse(Path("main.rs"), RUST_FULL)
        fns = [e for e in result.entities if e.type == EntityType.FUNCTION]
        fn_names = {f.name for f in fns}
        assert "calculate" in fn_names
        assert "helper" in fn_names
        assert "fetch_data" in fn_names

    @pytest.mark.asyncio
    async def test_async_function_detected(self, processor):
        result = await processor._regex_parse(Path("main.rs"), RUST_FULL)
        fns = [e for e in result.entities if e.type == EntityType.FUNCTION]
        fetch = [f for f in fns if f.name == "fetch_data"]
        assert len(fetch) >= 1
        assert fetch[0].is_async is True

    @pytest.mark.asyncio
    async def test_function_visibility(self, processor):
        result = await processor._regex_parse(Path("main.rs"), RUST_FULL)
        fns = [e for e in result.entities if e.type == EntityType.FUNCTION]
        calc = [f for f in fns if f.name == "calculate"]
        if calc:
            assert calc[0].visibility == "public"

    @pytest.mark.asyncio
    async def test_extracts_impl_blocks(self, processor):
        result = await processor._regex_parse(Path("main.rs"), RUST_FULL)
        impls = [
            e
            for e in result.entities
            if e.type == EntityType.CLASS and e.properties and "impl_type" in e.properties
        ]
        assert len(impls) >= 2
        # Check trait impl
        trait_impls = [i for i in impls if i.properties.get("impl_type") == "trait"]
        assert len(trait_impls) >= 1
        assert trait_impls[0].properties["trait_name"] == "Drawable"
        assert trait_impls[0].properties["type_name"] == "MyStruct"
        # Check inherent impl
        inherent_impls = [i for i in impls if i.properties.get("impl_type") == "inherent"]
        assert len(inherent_impls) >= 1

    @pytest.mark.asyncio
    async def test_extracts_macros(self, processor):
        result = await processor._regex_parse(Path("main.rs"), RUST_FULL)
        macros = [e for e in result.entities if e.type == EntityType.MACRO]
        macro_names = {m.name for m in macros}
        assert "my_macro" in macro_names

    @pytest.mark.asyncio
    async def test_empty_code(self, processor):
        result = await processor._regex_parse(Path("empty.rs"), "")
        assert result.success is True
        assert len(result.entities) == 0

    @pytest.mark.asyncio
    async def test_functions_skip_keywords(self, processor):
        """main, new, default are skipped by regex."""
        code = """\
fn main() {}
fn new() {}
fn default() {}
fn actual_fn(x: i32) -> i32 { x }
"""
        result = await processor._regex_parse(Path("test.rs"), code)
        fns = [e for e in result.entities if e.type == EntityType.FUNCTION]
        fn_names = {f.name for f in fns}
        assert "main" not in fn_names
        assert "new" not in fn_names
        assert "default" not in fn_names
        assert "actual_fn" in fn_names

    @pytest.mark.asyncio
    async def test_impl_with_generics(self, processor):
        code = """\
impl<T> Container<T> {
    fn get(&self) -> &T { &self.value }
}

impl<T: Display> Display for Container<T> {
    fn fmt(&self) -> String { String::new() }
}
"""
        result = await processor._regex_parse(Path("generic.rs"), code)
        impls = [
            e
            for e in result.entities
            if e.type == EntityType.CLASS and e.properties and "impl_type" in e.properties
        ]
        assert len(impls) >= 2


# ── RustProcessor dispatch ───────────────────────────────────────────


class TestRustParseFileDispatch:
    @pytest.mark.asyncio
    async def test_parse_file_uses_regex_when_no_parser(self, processor):
        processor.parser = None
        result = await processor.parse_file(Path("test.rs"), RUST_FULL)
        assert result.success is True
        assert result.parser_used == ParserType.REGEX_FALLBACK

    @pytest.mark.asyncio
    async def test_initialize_impl(self, processor):
        await processor._initialize_impl()

    @pytest.mark.asyncio
    async def test_extract_entities_with_mock_tree(self, processor):
        """extract_entities with a mock tree returns entities."""
        from unittest.mock import MagicMock

        mock_tree = MagicMock()
        mock_tree.text = b""
        mock_tree.root_node.children = []
        mock_tree.root_node.type = "source_file"
        entities = await processor.extract_entities(mock_tree, Path("test.rs"))
        assert isinstance(entities, list)

    @pytest.mark.asyncio
    async def test_extract_relationships_with_mock_tree(self, processor):
        """extract_relationships with a mock tree returns relationships."""
        from unittest.mock import MagicMock

        mock_tree = MagicMock()
        mock_tree.text = b""
        mock_tree.root_node.children = []
        mock_tree.root_node.type = "source_file"
        rels = await processor.extract_relationships(mock_tree, [], Path("test.rs"))
        assert isinstance(rels, list)


# ── RustSyntaxParser ─────────────────────────────────────────────────


class TestRustSyntaxParserRegex:
    def test_extract_modules(self, syntax_parser):
        entities = syntax_parser.extract_modules_regex(RUST_FULL, Path("main.rs"))
        mod_names = {e.name for e in entities}
        assert "utils" in mod_names
        assert "internal" in mod_names

    def test_extract_structs(self, syntax_parser):
        entities = syntax_parser.extract_structs_regex(RUST_FULL, Path("main.rs"))
        names = {e.name for e in entities}
        assert "MyStruct" in names
        assert "PrivateStruct" in names

    def test_extract_enums(self, syntax_parser):
        entities = syntax_parser.extract_enums_regex(RUST_FULL, Path("main.rs"))
        names = {e.name for e in entities}
        assert "Color" in names
        assert "Direction" in names

    def test_extract_traits(self, syntax_parser):
        entities = syntax_parser.extract_traits_regex(RUST_FULL, Path("main.rs"))
        names = {e.name for e in entities}
        assert "Drawable" in names
        assert "Printable" in names

    def test_extract_functions(self, syntax_parser):
        entities = syntax_parser.extract_functions_regex(RUST_FULL, Path("main.rs"))
        names = {e.name for e in entities}
        assert "calculate" in names
        assert "helper" in names
        assert "fetch_data" in names

    def test_extract_impl_blocks(self, syntax_parser):
        entities = syntax_parser.extract_impl_blocks_regex(RUST_FULL, Path("main.rs"))
        assert len(entities) >= 2

    def test_extract_macros(self, syntax_parser):
        entities = syntax_parser.extract_macros_regex(RUST_FULL, Path("main.rs"))
        names = {e.name for e in entities}
        assert "my_macro" in names

    def test_impl_inherent_properties(self, syntax_parser):
        entities = syntax_parser.extract_impl_blocks_regex(RUST_FULL, Path("main.rs"))
        inherent = [e for e in entities if e.properties.get("impl_type") == "inherent"]
        assert len(inherent) >= 1
        assert inherent[0].properties["type_name"] == "MyStruct"

    def test_impl_trait_properties(self, syntax_parser):
        entities = syntax_parser.extract_impl_blocks_regex(RUST_FULL, Path("main.rs"))
        trait_impls = [e for e in entities if e.properties.get("impl_type") == "trait"]
        assert len(trait_impls) >= 1


# ── RustSyntaxParser tree-sitter fallback methods ────────────────────


class TestRustSyntaxParserTreeSitterFallback:
    def test_extract_modules_tree_sitter_no_processor(self, syntax_parser):
        result = syntax_parser.extract_modules_tree_sitter(MagicMock(), Path("t.rs"), b"")
        assert result == []

    def test_extract_structs_tree_sitter_no_processor(self, syntax_parser):
        result = syntax_parser.extract_structs_tree_sitter(MagicMock(), Path("t.rs"), b"")
        assert result == []

    def test_extract_enums_tree_sitter_no_processor(self, syntax_parser):
        result = syntax_parser.extract_enums_tree_sitter(MagicMock(), Path("t.rs"), b"")
        assert result == []

    def test_extract_traits_tree_sitter_no_processor(self, syntax_parser):
        result = syntax_parser.extract_traits_tree_sitter(MagicMock(), Path("t.rs"), b"")
        assert result == []

    def test_extract_functions_tree_sitter_no_processor(self, syntax_parser):
        result = syntax_parser.extract_functions_tree_sitter(MagicMock(), Path("t.rs"), b"")
        assert result == []

    def test_extract_impl_blocks_tree_sitter_no_processor(self, syntax_parser):
        result = syntax_parser.extract_impl_blocks_tree_sitter(MagicMock(), Path("t.rs"), b"")
        assert result == []

    def test_extract_constants_tree_sitter_no_processor(self, syntax_parser):
        result = syntax_parser.extract_constants_tree_sitter(MagicMock(), Path("t.rs"), b"")
        assert result == []

    def test_extract_visibility_no_processor(self, syntax_parser):
        result = syntax_parser._extract_visibility(MagicMock(), b"pub fn test()")
        assert result == "private"  # fallback
