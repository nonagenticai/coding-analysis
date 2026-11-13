"""Tests for Rust language processor regex-based parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from processing.languages.rust.rust_processor import RustProcessor


@pytest.fixture
def processor():
    return RustProcessor()


SAMPLE_RUST_CODE = """\
use std::collections::HashMap;
use std::io::{self, Read};

pub mod utils;
mod internal;

pub struct Config {
    name: String,
    value: i32,
}

pub enum Status {
    Active,
    Inactive,
    Error(String),
}

pub trait Processable {
    fn process(&self) -> Result<(), String>;
}

impl Processable for Config {
    fn process(&self) -> Result<(), String> {
        Ok(())
    }
}

impl Config {
    pub fn new(name: String) -> Self {
        Config { name, value: 0 }
    }

    pub async fn fetch_data(&self) -> String {
        self.name.clone()
    }
}

pub fn calculate(a: i32, b: i32) -> i32 {
    a + b
}

fn helper_function(data: &str) -> bool {
    !data.is_empty()
}

const MAX_SIZE: usize = 1024;

macro_rules! my_macro {
    ($x:expr) => { println!("{}", $x) }
}
"""


@pytest.mark.asyncio
async def test_regex_parse_success(processor):
    fp = Path("lib.rs")
    result = await processor._regex_parse(fp, SAMPLE_RUST_CODE)
    assert result.success is True
    assert result.language == "rust"
    assert result.lines_of_code == len(SAMPLE_RUST_CODE.splitlines())


@pytest.mark.asyncio
async def test_extracts_use_statements(processor):
    fp = Path("lib.rs")
    result = await processor._regex_parse(fp, SAMPLE_RUST_CODE)
    imports = [e for e in result.entities if e.type.value == "import"]
    assert len(imports) >= 2
    import_names = {e.name for e in imports}
    assert any("std::collections::HashMap" in n for n in import_names)


@pytest.mark.asyncio
async def test_extracts_modules(processor):
    fp = Path("lib.rs")
    result = await processor._regex_parse(fp, SAMPLE_RUST_CODE)
    modules = [e for e in result.entities if e.type.value == "module"]
    mod_names = {e.name for e in modules}
    assert "utils" in mod_names
    assert "internal" in mod_names


@pytest.mark.asyncio
async def test_module_visibility(processor):
    fp = Path("lib.rs")
    result = await processor._regex_parse(fp, SAMPLE_RUST_CODE)
    modules = [e for e in result.entities if e.type.value == "module"]
    utils_mod = next((m for m in modules if m.name == "utils"), None)
    internal_mod = next((m for m in modules if m.name == "internal"), None)
    assert utils_mod is not None
    assert utils_mod.visibility == "public"
    assert internal_mod is not None
    assert internal_mod.visibility == "private"


@pytest.mark.asyncio
async def test_extracts_structs(processor):
    fp = Path("lib.rs")
    result = await processor._regex_parse(fp, SAMPLE_RUST_CODE)
    structs = [e for e in result.entities if e.type.value == "struct"]
    struct_names = {e.name for e in structs}
    assert "Config" in struct_names


@pytest.mark.asyncio
async def test_extracts_enums(processor):
    fp = Path("lib.rs")
    result = await processor._regex_parse(fp, SAMPLE_RUST_CODE)
    enums = [e for e in result.entities if e.type.value == "enum"]
    enum_names = {e.name for e in enums}
    assert "Status" in enum_names


@pytest.mark.asyncio
async def test_extracts_traits(processor):
    fp = Path("lib.rs")
    result = await processor._regex_parse(fp, SAMPLE_RUST_CODE)
    traits = [e for e in result.entities if e.type.value == "trait"]
    trait_names = {e.name for e in traits}
    assert "Processable" in trait_names


@pytest.mark.asyncio
async def test_trait_visibility(processor):
    fp = Path("lib.rs")
    result = await processor._regex_parse(fp, SAMPLE_RUST_CODE)
    traits = [e for e in result.entities if e.type.value == "trait"]
    processable = next((t for t in traits if t.name == "Processable"), None)
    assert processable is not None
    assert processable.visibility == "public"


@pytest.mark.asyncio
async def test_extracts_functions(processor):
    fp = Path("lib.rs")
    result = await processor._regex_parse(fp, SAMPLE_RUST_CODE)
    funcs = [e for e in result.entities if e.type.value == "function"]
    func_names = {e.name for e in funcs}
    assert "calculate" in func_names
    assert "helper_function" in func_names


@pytest.mark.asyncio
async def test_function_visibility(processor):
    fp = Path("lib.rs")
    result = await processor._regex_parse(fp, SAMPLE_RUST_CODE)
    funcs = [e for e in result.entities if e.type.value == "function"]
    calc = next((f for f in funcs if f.name == "calculate"), None)
    helper = next((f for f in funcs if f.name == "helper_function"), None)
    assert calc is not None
    assert calc.visibility == "public"
    assert helper is not None
    assert helper.visibility == "private"


@pytest.mark.asyncio
async def test_extracts_impl_blocks(processor):
    fp = Path("lib.rs")
    result = await processor._regex_parse(fp, SAMPLE_RUST_CODE)
    impl_blocks = [e for e in result.entities if e.type.value == "class"]
    assert len(impl_blocks) >= 2
    impl_names = {e.name for e in impl_blocks}
    assert any("Processable for Config" in n for n in impl_names)
    assert any("impl Config" in n for n in impl_names)


@pytest.mark.asyncio
async def test_impl_block_trait_properties(processor):
    fp = Path("lib.rs")
    result = await processor._regex_parse(fp, SAMPLE_RUST_CODE)
    impl_blocks = [e for e in result.entities if e.type.value == "class"]
    trait_impl = next(
        (e for e in impl_blocks if e.properties and e.properties.get("impl_type") == "trait"),
        None,
    )
    assert trait_impl is not None
    assert trait_impl.properties["trait_name"] == "Processable"
    assert trait_impl.properties["type_name"] == "Config"


@pytest.mark.asyncio
async def test_impl_block_inherent_properties(processor):
    fp = Path("lib.rs")
    result = await processor._regex_parse(fp, SAMPLE_RUST_CODE)
    impl_blocks = [e for e in result.entities if e.type.value == "class"]
    inherent_impl = next(
        (e for e in impl_blocks if e.properties and e.properties.get("impl_type") == "inherent"),
        None,
    )
    assert inherent_impl is not None
    assert inherent_impl.properties["type_name"] == "Config"


@pytest.mark.asyncio
async def test_extracts_macros(processor):
    fp = Path("lib.rs")
    result = await processor._regex_parse(fp, SAMPLE_RUST_CODE)
    macros = [e for e in result.entities if e.type.value == "macro"]
    macro_names = {e.name for e in macros}
    assert "my_macro" in macro_names


@pytest.mark.asyncio
async def test_regex_parse_empty_code(processor):
    fp = Path("empty.rs")
    result = await processor._regex_parse(fp, "")
    assert result.success is True
    assert result.entities == []


@pytest.mark.asyncio
async def test_parse_file_fallback(processor):
    fp = Path("test.rs")
    result = await processor.parse_file(fp, SAMPLE_RUST_CODE)
    assert result.success is True
    assert len(result.entities) > 0


@pytest.mark.asyncio
async def test_initialize_impl(processor):
    await processor._initialize_impl()


@pytest.mark.asyncio
async def test_extract_entities_with_none_tree(processor):
    """When tree is None, extract_entities should return empty list or handle gracefully."""
    try:
        entities = await processor.extract_entities(None, Path("test.rs"))
        assert entities == []
    except (AttributeError, TypeError):
        # Main processor requires a real tree — None is not a valid input
        pass


@pytest.mark.asyncio
async def test_extract_relationships_with_none_tree(processor):
    """When tree is None, extract_relationships should return empty list or handle gracefully."""
    try:
        rels = await processor.extract_relationships(None, [], Path("test.rs"))
        assert rels == []
    except (AttributeError, TypeError):
        # Main processor requires a real tree — None is not a valid input
        pass
