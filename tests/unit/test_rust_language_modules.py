"""Unit tests for Rust language processing modules.

Covers:
- rust_processor.py (RustProcessor regex-based parsing)
- rust_syntax_parser.py (RustSyntaxParser regex + tree-sitter-fallback paths)
- rust_dependency_analyzer.py (RustDependencyAnalyzer use-statement / dep analysis)
- rust_cargo_manager.py (RustCargoManager Cargo.toml parsing & project structure)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models.core_models import (
    CoreCodeEntity,
    EntityType,
    SourceLocation,
)
from processing.languages.base import ParserType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RUST = """\
use std::io;
use serde::Deserialize;
use crate::models::User;

pub mod network;
mod internal;

pub struct Config {
    host: String,
    port: u16,
}

struct Private {
    data: Vec<u8>,
}

pub enum Status {
    Active,
    Inactive,
}

enum InternalState {
    Ready,
    Error,
}

pub trait Handler {
    fn handle(&self);
}

trait InternalTrait {
    fn process(&self);
}

pub fn serve(cfg: &Config) {
    println!("serving");
}

pub async fn async_serve(cfg: &Config) {
    println!("async serving");
}

fn helper() {
    // private helper
}

impl Handler for Config {
    fn handle(&self) {
        println!("handling");
    }
}

impl Config {
    fn build() {
        // builder
    }
}

impl<T> Clone for Config {
    fn clone(&self) -> Self {
        Config { host: self.host.clone(), port: self.port }
    }
}

macro_rules! my_macro {
    () => {};
}

macro_rules! another_macro {
    ($x:expr) => { $x + 1 };
}
"""

FP = Path("/fake/project/src/lib.rs")


def _make_import_entity(use_path: str, fp: str = "test.rs") -> CoreCodeEntity:
    """Helper to create an IMPORT entity for dependency analysis tests."""
    return CoreCodeEntity(
        id=f"use:{use_path}:{fp}",
        name=use_path,
        type=EntityType.IMPORT,
        language="rust",
        file_path=fp,
        line_number=1,
        end_line_number=None,
        column_start=0,
        column_end=0,
        visibility="public",
        code_snippet=None,
        docstring=None,
        signature=None,
        source_code=None,
        created_at=None,
        updated_at=None,
        properties={"use_path": use_path},
    )


# ============================================================================
# RustProcessor tests
# ============================================================================


class TestRustProcessor:
    """Tests for RustProcessor (regex fallback path)."""

    @pytest.fixture
    def processor(self):
        from processing.languages.rust.rust_processor import RustProcessor

        return RustProcessor()

    # --- _regex_parse ---

    async def test_regex_parse_returns_success(self, processor):
        result = await processor._regex_parse(FP, SAMPLE_RUST)
        assert result.success is True
        assert result.language == "rust"
        assert result.parser_used == ParserType.REGEX_FALLBACK

    async def test_regex_parse_extracts_modules(self, processor):
        result = await processor._regex_parse(FP, SAMPLE_RUST)
        mod_entities = [e for e in result.entities if e.type == EntityType.MODULE]
        names = {e.name for e in mod_entities}
        assert "network" in names
        assert "internal" in names

    async def test_regex_parse_extracts_structs(self, processor):
        result = await processor._regex_parse(FP, SAMPLE_RUST)
        struct_entities = [e for e in result.entities if e.type == EntityType.STRUCT]
        names = {e.name for e in struct_entities}
        assert "Config" in names
        assert "Private" in names

    async def test_regex_parse_extracts_enums(self, processor):
        result = await processor._regex_parse(FP, SAMPLE_RUST)
        enum_entities = [e for e in result.entities if e.type == EntityType.ENUM]
        names = {e.name for e in enum_entities}
        assert "Status" in names
        assert "InternalState" in names

    async def test_regex_parse_extracts_traits(self, processor):
        result = await processor._regex_parse(FP, SAMPLE_RUST)
        trait_entities = [e for e in result.entities if e.type == EntityType.TRAIT]
        names = {e.name for e in trait_entities}
        assert "Handler" in names
        assert "InternalTrait" in names

    async def test_regex_parse_extracts_functions(self, processor):
        result = await processor._regex_parse(FP, SAMPLE_RUST)
        fn_entities = [e for e in result.entities if e.type == EntityType.FUNCTION]
        names = {e.name for e in fn_entities}
        assert "serve" in names
        assert "async_serve" in names
        assert "helper" in names

    async def test_regex_parse_skips_main_new_default_functions(self, processor):
        code = "fn main() {}\nfn new() {}\nfn default() {}\nfn real_fn() {}"
        result = await processor._regex_parse(FP, code)
        fn_names = {e.name for e in result.entities if e.type == EntityType.FUNCTION}
        assert "main" not in fn_names
        assert "new" not in fn_names
        assert "default" not in fn_names
        assert "real_fn" in fn_names

    async def test_regex_parse_async_flag_on_functions(self, processor):
        result = await processor._regex_parse(FP, SAMPLE_RUST)
        fn_entities = [e for e in result.entities if e.type == EntityType.FUNCTION]
        async_fns = {e.name for e in fn_entities if e.is_async}
        non_async_fns = {e.name for e in fn_entities if not e.is_async}
        assert "async_serve" in async_fns
        assert "serve" in non_async_fns

    async def test_regex_parse_extracts_impl_blocks(self, processor):
        result = await processor._regex_parse(FP, SAMPLE_RUST)
        impl_entities = [e for e in result.entities if e.type == EntityType.CLASS]
        names = {e.name for e in impl_entities}
        # Trait impl
        assert "Handler for Config" in names
        # Inherent impl
        assert "impl Config" in names

    async def test_regex_parse_impl_properties(self, processor):
        result = await processor._regex_parse(FP, SAMPLE_RUST)
        impl_entities = [e for e in result.entities if e.type == EntityType.CLASS]
        trait_impls = [e for e in impl_entities if e.properties.get("impl_type") == "trait"]
        inherent_impls = [e for e in impl_entities if e.properties.get("impl_type") == "inherent"]
        assert len(trait_impls) >= 1
        assert len(inherent_impls) >= 1
        assert trait_impls[0].properties["trait_name"] == "Handler"
        assert trait_impls[0].properties["type_name"] == "Config"

    async def test_regex_parse_extracts_macros(self, processor):
        result = await processor._regex_parse(FP, SAMPLE_RUST)
        macro_entities = [e for e in result.entities if e.type == EntityType.MACRO]
        names = {e.name for e in macro_entities}
        assert "my_macro" in names
        assert "another_macro" in names

    async def test_regex_parse_extracts_use_statements(self, processor):
        result = await processor._regex_parse(FP, SAMPLE_RUST)
        import_entities = [e for e in result.entities if e.type == EntityType.IMPORT]
        names = {e.name for e in import_entities}
        assert "std::io" in names
        assert "serde::Deserialize" in names
        assert "crate::models::User" in names

    async def test_regex_parse_line_numbers(self, processor):
        result = await processor._regex_parse(FP, SAMPLE_RUST)
        # First use statement "use std::io;" is on line 1
        import_entities = [e for e in result.entities if e.type == EntityType.IMPORT]
        first_import = next(e for e in import_entities if e.name == "std::io")
        assert first_import.line_number == 1

    async def test_regex_parse_lines_of_code(self, processor):
        result = await processor._regex_parse(FP, SAMPLE_RUST)
        assert result.lines_of_code == len(SAMPLE_RUST.splitlines())

    async def test_regex_parse_visibility_public_module(self, processor):
        result = await processor._regex_parse(FP, SAMPLE_RUST)
        mod_entities = [e for e in result.entities if e.type == EntityType.MODULE]
        network = next(e for e in mod_entities if e.name == "network")
        internal = next(e for e in mod_entities if e.name == "internal")
        assert network.visibility == "public"
        assert internal.visibility == "private"

    async def test_regex_parse_visibility_public_trait(self, processor):
        result = await processor._regex_parse(FP, SAMPLE_RUST)
        trait_entities = [e for e in result.entities if e.type == EntityType.TRAIT]
        handler = next(e for e in trait_entities if e.name == "Handler")
        internal_trait = next(e for e in trait_entities if e.name == "InternalTrait")
        assert handler.visibility == "public"
        assert internal_trait.visibility == "private"

    async def test_regex_parse_empty_content(self, processor):
        result = await processor._regex_parse(FP, "")
        assert result.success is True
        assert result.entities == []

    async def test_parse_file_uses_regex_fallback(self, processor):
        """parse_file should fall back to regex when tree-sitter is not available."""
        # The processor.parser attribute may or may not exist depending on tree-sitter
        # availability. Force the regex path.
        processor.parser = None
        result = await processor.parse_file(FP, SAMPLE_RUST)
        assert result.success is True
        assert result.parser_used == ParserType.REGEX_FALLBACK

    async def test_initialize_impl_no_crash(self, processor):
        """_initialize_impl should not crash."""
        await processor._initialize_impl()


# ============================================================================
# RustSyntaxParser tests
# ============================================================================


class TestRustSyntaxParser:
    """Tests for RustSyntaxParser regex methods."""

    @pytest.fixture
    def parser(self):
        from processing.languages.rust.rust_syntax_parser import RustSyntaxParser

        return RustSyntaxParser()

    # --- extract_modules_regex ---

    def test_extract_modules_regex(self, parser):
        entities = parser.extract_modules_regex(SAMPLE_RUST, FP)
        names = {e.name for e in entities}
        assert "network" in names
        assert "internal" in names
        assert all(e.type == EntityType.MODULE for e in entities)

    def test_extract_modules_regex_visibility(self, parser):
        entities = parser.extract_modules_regex(SAMPLE_RUST, FP)
        network = next(e for e in entities if e.name == "network")
        internal = next(e for e in entities if e.name == "internal")
        assert network.visibility == "public"
        assert network.properties["is_public"] is True
        assert internal.visibility == "private"
        assert internal.properties["is_public"] is False

    def test_extract_modules_regex_empty(self, parser):
        assert parser.extract_modules_regex("", FP) == []

    # --- extract_structs_regex ---

    def test_extract_structs_regex(self, parser):
        entities = parser.extract_structs_regex(SAMPLE_RUST, FP)
        names = {e.name for e in entities}
        assert "Config" in names
        assert "Private" in names
        assert all(e.type == EntityType.STRUCT for e in entities)

    def test_extract_structs_regex_visibility(self, parser):
        entities = parser.extract_structs_regex(SAMPLE_RUST, FP)
        config = next(e for e in entities if e.name == "Config")
        private = next(e for e in entities if e.name == "Private")
        assert config.visibility == "public"
        assert private.visibility == "private"

    def test_extract_structs_regex_empty(self, parser):
        assert parser.extract_structs_regex("fn main() {}", FP) == []

    # --- extract_enums_regex ---

    def test_extract_enums_regex(self, parser):
        entities = parser.extract_enums_regex(SAMPLE_RUST, FP)
        names = {e.name for e in entities}
        assert "Status" in names
        assert "InternalState" in names
        assert all(e.type == EntityType.ENUM for e in entities)

    def test_extract_enums_regex_visibility(self, parser):
        entities = parser.extract_enums_regex(SAMPLE_RUST, FP)
        status = next(e for e in entities if e.name == "Status")
        internal = next(e for e in entities if e.name == "InternalState")
        assert status.visibility == "public"
        assert internal.visibility == "private"

    def test_extract_enums_regex_empty(self, parser):
        assert parser.extract_enums_regex("", FP) == []

    # --- extract_traits_regex ---

    def test_extract_traits_regex(self, parser):
        entities = parser.extract_traits_regex(SAMPLE_RUST, FP)
        names = {e.name for e in entities}
        assert "Handler" in names
        assert "InternalTrait" in names
        assert all(e.type == EntityType.TRAIT for e in entities)

    def test_extract_traits_regex_visibility(self, parser):
        entities = parser.extract_traits_regex(SAMPLE_RUST, FP)
        handler = next(e for e in entities if e.name == "Handler")
        internal = next(e for e in entities if e.name == "InternalTrait")
        assert handler.visibility == "public"
        assert internal.visibility == "private"

    def test_extract_traits_regex_empty(self, parser):
        assert parser.extract_traits_regex("", FP) == []

    # --- extract_functions_regex ---

    def test_extract_functions_regex(self, parser):
        entities = parser.extract_functions_regex(SAMPLE_RUST, FP)
        names = {e.name for e in entities}
        assert "serve" in names
        assert "async_serve" in names
        assert "helper" in names
        assert all(e.type == EntityType.FUNCTION for e in entities)

    def test_extract_functions_regex_skips_main_new_default(self, parser):
        code = "fn main() {}\npub fn new() {}\nfn default() {}\nfn ok_fn() {}"
        entities = parser.extract_functions_regex(code, FP)
        names = {e.name for e in entities}
        assert "main" not in names
        assert "new" not in names
        assert "default" not in names
        assert "ok_fn" in names

    def test_extract_functions_regex_async_flag(self, parser):
        entities = parser.extract_functions_regex(SAMPLE_RUST, FP)
        async_fn = next(e for e in entities if e.name == "async_serve")
        sync_fn = next(e for e in entities if e.name == "serve")
        assert async_fn.is_async is True
        assert sync_fn.is_async is False

    def test_extract_functions_regex_visibility(self, parser):
        entities = parser.extract_functions_regex(SAMPLE_RUST, FP)
        serve = next(e for e in entities if e.name == "serve")
        helper_fn = next(e for e in entities if e.name == "helper")
        assert serve.visibility == "public"
        assert helper_fn.visibility == "private"

    def test_extract_functions_regex_empty(self, parser):
        assert parser.extract_functions_regex("", FP) == []

    # --- extract_impl_blocks_regex ---

    def test_extract_impl_blocks_regex_trait_impl(self, parser):
        entities = parser.extract_impl_blocks_regex(SAMPLE_RUST, FP)
        trait_impls = [e for e in entities if e.properties.get("impl_type") == "trait"]
        assert len(trait_impls) >= 1
        handler_impl = next(
            (e for e in trait_impls if e.properties.get("trait_name") == "Handler"), None
        )
        assert handler_impl is not None
        assert handler_impl.properties["type_name"] == "Config"
        assert handler_impl.type == EntityType.CLASS

    def test_extract_impl_blocks_regex_inherent_impl(self, parser):
        entities = parser.extract_impl_blocks_regex(SAMPLE_RUST, FP)
        inherent_impls = [e for e in entities if e.properties.get("impl_type") == "inherent"]
        assert len(inherent_impls) >= 1
        config_impl = next(
            (e for e in inherent_impls if e.properties.get("type_name") == "Config"), None
        )
        assert config_impl is not None
        assert config_impl.name == "impl Config"

    def test_extract_impl_blocks_regex_empty(self, parser):
        assert parser.extract_impl_blocks_regex("fn foo() {}", FP) == []

    # --- extract_macros_regex ---

    def test_extract_macros_regex(self, parser):
        entities = parser.extract_macros_regex(SAMPLE_RUST, FP)
        names = {e.name for e in entities}
        assert "my_macro" in names
        assert "another_macro" in names
        assert all(e.type == EntityType.MACRO for e in entities)

    def test_extract_macros_regex_empty(self, parser):
        assert parser.extract_macros_regex("fn foo() {}", FP) == []

    # --- tree-sitter fallback methods ---

    def test_tree_sitter_fallback_methods_return_empty(self, parser):
        """When tree-sitter is not loaded, fallback methods return []."""
        tree = MagicMock()
        content = b""
        assert parser.extract_modules_tree_sitter(tree, FP, content) == []
        assert parser.extract_structs_tree_sitter(tree, FP, content) == []
        assert parser.extract_enums_tree_sitter(tree, FP, content) == []
        assert parser.extract_traits_tree_sitter(tree, FP, content) == []
        assert parser.extract_functions_tree_sitter(tree, FP, content) == []
        assert parser.extract_impl_blocks_tree_sitter(tree, FP, content) == []
        assert parser.extract_constants_tree_sitter(tree, FP, content) == []

    def test_extract_visibility_fallback(self, parser):
        """Fallback _extract_visibility should return 'private'."""
        assert parser._extract_visibility(MagicMock(), b"") == "private"


# ============================================================================
# RustSyntaxParser tree-sitter tests
# ============================================================================

from processing.languages.rust.rust_syntax_parser import (
    TREE_SITTER_AVAILABLE as SYNTAX_TS_AVAILABLE,
)

RUST_TS_SAMPLE = """\
use std::io;
use std::collections::HashMap;

pub mod network;
mod internal;

pub struct Config {
    pub host: String,
    port: u16,
}

struct Private {
    data: Vec<u8>,
}

pub enum Status {
    Active,
    Inactive,
}

enum InternalState {
    Ready,
    Error,
}

pub trait Handler {
    fn handle(&self);
}

trait InternalTrait {
    fn process(&self);
}

pub fn serve(cfg: &Config) {
    println!("serving");
}

pub async fn async_serve(cfg: &Config) {
    println!("async serving");
}

fn helper_fn() {
}

impl Handler for Config {
    fn handle(&self) {
        println!("handling");
    }
}

impl Config {
    fn build() {
    }
}

const MAX: u32 = 100;
static COUNTER: u32 = 0;
"""


@pytest.mark.skipif(not SYNTAX_TS_AVAILABLE, reason="tree-sitter not installed")
class TestRustSyntaxParserTreeSitter:
    """Tests for RustSyntaxParser tree-sitter methods."""

    @pytest.fixture
    async def ts_processor(self):
        from processing.languages.rust.rust_processor import RustProcessor

        p = RustProcessor()
        await p._initialize_impl()
        return p

    @pytest.fixture
    async def parser_with_ts(self, ts_processor):
        from processing.languages.rust.rust_syntax_parser import RustSyntaxParser

        return RustSyntaxParser(tree_sitter_processor=ts_processor)

    @pytest.fixture
    async def rust_tree(self, ts_processor):
        return ts_processor.parser.parse(RUST_TS_SAMPLE.encode("utf-8"))

    @pytest.fixture
    def content_bytes(self):
        return RUST_TS_SAMPLE.encode("utf-8")

    async def test_extract_modules_tree_sitter(self, parser_with_ts, rust_tree, content_bytes):
        entities = parser_with_ts.extract_modules_tree_sitter(rust_tree, FP, content_bytes)
        names = {e.name for e in entities}
        assert "network" in names
        assert "internal" in names
        assert all(e.type == EntityType.MODULE for e in entities)

    async def test_extract_modules_tree_sitter_visibility(
        self, parser_with_ts, rust_tree, content_bytes
    ):
        entities = parser_with_ts.extract_modules_tree_sitter(rust_tree, FP, content_bytes)
        network = next(e for e in entities if e.name == "network")
        internal = next(e for e in entities if e.name == "internal")
        assert network.visibility == "public"
        assert internal.visibility == "private"

    async def test_extract_structs_tree_sitter(self, parser_with_ts, rust_tree, content_bytes):
        entities = parser_with_ts.extract_structs_tree_sitter(rust_tree, FP, content_bytes)
        names = {e.name for e in entities}
        assert "Config" in names
        assert "Private" in names
        assert all(e.type == EntityType.STRUCT for e in entities)

    async def test_extract_structs_tree_sitter_visibility(
        self, parser_with_ts, rust_tree, content_bytes
    ):
        entities = parser_with_ts.extract_structs_tree_sitter(rust_tree, FP, content_bytes)
        config = next(e for e in entities if e.name == "Config")
        private = next(e for e in entities if e.name == "Private")
        assert config.visibility == "public"
        assert private.visibility == "private"

    async def test_extract_enums_tree_sitter(self, parser_with_ts, rust_tree, content_bytes):
        entities = parser_with_ts.extract_enums_tree_sitter(rust_tree, FP, content_bytes)
        names = {e.name for e in entities}
        assert "Status" in names
        assert "InternalState" in names
        assert all(e.type == EntityType.ENUM for e in entities)

    async def test_extract_traits_tree_sitter(self, parser_with_ts, rust_tree, content_bytes):
        entities = parser_with_ts.extract_traits_tree_sitter(rust_tree, FP, content_bytes)
        names = {e.name for e in entities}
        assert "Handler" in names
        assert "InternalTrait" in names
        assert all(e.type == EntityType.TRAIT for e in entities)

    async def test_extract_traits_tree_sitter_visibility(
        self, parser_with_ts, rust_tree, content_bytes
    ):
        entities = parser_with_ts.extract_traits_tree_sitter(rust_tree, FP, content_bytes)
        handler = next(e for e in entities if e.name == "Handler")
        internal = next(e for e in entities if e.name == "InternalTrait")
        assert handler.visibility == "public"
        assert internal.visibility == "private"

    async def test_extract_functions_tree_sitter(self, parser_with_ts, rust_tree, content_bytes):
        entities = parser_with_ts.extract_functions_tree_sitter(rust_tree, FP, content_bytes)
        names = {e.name for e in entities}
        assert "serve" in names
        assert "async_serve" in names
        assert "helper_fn" in names
        assert all(e.type == EntityType.FUNCTION for e in entities)

    async def test_extract_functions_tree_sitter_async(
        self, parser_with_ts, rust_tree, content_bytes
    ):
        entities = parser_with_ts.extract_functions_tree_sitter(rust_tree, FP, content_bytes)
        async_fn = next(e for e in entities if e.name == "async_serve")
        sync_fn = next(e for e in entities if e.name == "serve")
        assert async_fn.is_async is True
        assert sync_fn.is_async is False

    async def test_extract_impl_blocks_tree_sitter(self, parser_with_ts, rust_tree, content_bytes):
        entities = parser_with_ts.extract_impl_blocks_tree_sitter(rust_tree, FP, content_bytes)
        assert len(entities) >= 2
        trait_impls = [e for e in entities if e.properties.get("impl_type") == "trait"]
        inherent_impls = [e for e in entities if e.properties.get("impl_type") == "inherent"]
        assert len(trait_impls) >= 1
        assert len(inherent_impls) >= 1

    async def test_extract_constants_tree_sitter(self, parser_with_ts, rust_tree, content_bytes):
        entities = parser_with_ts.extract_constants_tree_sitter(rust_tree, FP, content_bytes)
        names = {e.name for e in entities}
        assert "MAX" in names
        assert "COUNTER" in names
        const_entity = next(e for e in entities if e.name == "MAX")
        assert const_entity.type == EntityType.CONSTANT
        static_entity = next(e for e in entities if e.name == "COUNTER")
        assert static_entity.type == EntityType.VARIABLE
        assert static_entity.is_static is True

    async def test_extract_visibility(self, parser_with_ts, rust_tree, content_bytes):
        struct_nodes = parser_with_ts.tree_sitter_processor._find_nodes_by_type(
            rust_tree, parser_with_ts.STRUCT_NODE_TYPES
        )
        for node in struct_nodes:
            vis = parser_with_ts._extract_visibility(node, content_bytes)
            assert vis in ("public", "private")

    def test_no_processor_returns_empty(self):
        from processing.languages.rust.rust_syntax_parser import RustSyntaxParser

        parser = RustSyntaxParser(tree_sitter_processor=None)
        tree = MagicMock()
        content = b""
        assert parser.extract_modules_tree_sitter(tree, FP, content) == []
        assert parser.extract_structs_tree_sitter(tree, FP, content) == []
        assert parser.extract_enums_tree_sitter(tree, FP, content) == []
        assert parser.extract_traits_tree_sitter(tree, FP, content) == []
        assert parser.extract_functions_tree_sitter(tree, FP, content) == []
        assert parser.extract_impl_blocks_tree_sitter(tree, FP, content) == []
        assert parser.extract_constants_tree_sitter(tree, FP, content) == []

    async def test_entity_line_numbers(self, parser_with_ts, rust_tree, content_bytes):
        entities = parser_with_ts.extract_structs_tree_sitter(rust_tree, FP, content_bytes)
        for entity in entities:
            assert entity.line_number >= 1

    async def test_entity_file_path(self, parser_with_ts, rust_tree, content_bytes):
        entities = parser_with_ts.extract_enums_tree_sitter(rust_tree, FP, content_bytes)
        for entity in entities:
            assert entity.file_path == str(FP)


# ============================================================================
# RustDependencyAnalyzer tests
# ============================================================================


class TestRustDependencyAnalyzer:
    """Tests for RustDependencyAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        from processing.languages.rust.rust_dependency_analyzer import RustDependencyAnalyzer

        return RustDependencyAnalyzer()

    # --- extract_use_statements_regex ---

    def test_extract_use_statements_regex(self, analyzer):
        entities = analyzer.extract_use_statements_regex(SAMPLE_RUST, FP)
        names = {e.name for e in entities}
        assert "std::io" in names
        assert "serde::Deserialize" in names
        assert "crate::models::User" in names
        assert all(e.type == EntityType.IMPORT for e in entities)
        assert all("use_path" in e.properties for e in entities)

    def test_extract_use_statements_regex_empty(self, analyzer):
        assert analyzer.extract_use_statements_regex("fn main() {}", FP) == []

    # --- parse_use_statement ---

    def test_parse_use_std(self, analyzer):
        info = analyzer.parse_use_statement("use std::io;")
        assert info["is_std"] is True
        assert info["is_external"] is False
        assert info["is_relative"] is False
        assert info["base_path"] == "std::io"
        assert "io" in info["imported_items"]

    def test_parse_use_external(self, analyzer):
        info = analyzer.parse_use_statement("use serde::Deserialize;")
        assert info["is_external"] is True
        assert info["is_std"] is False
        assert info["base_path"] == "serde::Deserialize"

    def test_parse_use_crate_relative(self, analyzer):
        info = analyzer.parse_use_statement("use crate::models::User;")
        assert info["is_relative"] is True
        assert info["is_external"] is False
        assert info["is_std"] is False

    def test_parse_use_super_relative(self, analyzer):
        info = analyzer.parse_use_statement("use super::parent::Module;")
        assert info["is_relative"] is True

    def test_parse_use_self_relative(self, analyzer):
        info = analyzer.parse_use_statement("use self::local::Item;")
        assert info["is_relative"] is True

    def test_parse_use_glob(self, analyzer):
        info = analyzer.parse_use_statement("use std::io::*;")
        assert info["is_glob"] is True
        assert info["base_path"] == "std::io"

    def test_parse_use_alias(self, analyzer):
        info = analyzer.parse_use_statement("use std::io::Result as IoResult;")
        assert info["is_aliased"] is True
        assert info["alias"] == "IoResult"

    def test_parse_use_group(self, analyzer):
        info = analyzer.parse_use_statement("use std::{io, fs};")
        assert info["base_path"] == "std"
        assert "io" in info["imported_items"]
        assert "fs" in info["imported_items"]

    def test_parse_use_simple_no_colons(self, analyzer):
        info = analyzer.parse_use_statement("use tokio;")
        assert info["base_path"] == "tokio"
        assert info["imported_items"] == ["tokio"]

    # --- extract_external_dependencies ---

    def test_extract_external_dependencies(self, analyzer):
        entities = [
            _make_import_entity("serde::Deserialize"),
            _make_import_entity("tokio::runtime::Runtime"),
            _make_import_entity("std::io"),
            _make_import_entity("crate::models::User"),
        ]
        deps = analyzer.extract_external_dependencies(entities)
        assert "serde" in deps
        assert "tokio" in deps
        # std and crate are not external
        assert "std" not in deps
        assert "crate" not in deps

    def test_extract_external_dependencies_empty(self, analyzer):
        assert analyzer.extract_external_dependencies([]) == []

    def test_extract_external_dependencies_no_external(self, analyzer):
        entities = [
            _make_import_entity("std::io"),
            _make_import_entity("crate::models"),
        ]
        assert analyzer.extract_external_dependencies(entities) == []

    # --- extract_internal_dependencies ---

    def test_extract_internal_dependencies(self, analyzer):
        entities = [
            _make_import_entity("crate::models::User"),
            _make_import_entity("super::parent::Helper"),
            _make_import_entity("serde::Deserialize"),
        ]
        deps = analyzer.extract_internal_dependencies(entities)
        paths = [d["path"] for d in deps]
        assert "crate::models::User" in paths
        assert "super::parent::Helper" in paths
        # serde is external, should NOT be in internal deps
        assert "serde::Deserialize" not in paths

    def test_extract_internal_dependencies_empty(self, analyzer):
        assert analyzer.extract_internal_dependencies([]) == []

    # --- analyze_dependency_graph ---

    def test_analyze_dependency_graph(self, analyzer):
        entities = [
            _make_import_entity("std::io"),
            _make_import_entity("serde::Deserialize"),
            _make_import_entity("crate::models::User"),
        ]
        analysis = analyzer.analyze_dependency_graph(entities)
        assert analysis["total_imports"] == 3
        assert len(analysis["std_imports"]) == 1
        assert len(analysis["external_crates"]) == 1
        assert len(analysis["relative_imports"]) == 1

    def test_analyze_dependency_graph_glob(self, analyzer):
        entities = [_make_import_entity("std::io::*")]
        analysis = analyzer.analyze_dependency_graph(entities)
        assert len(analysis["glob_imports"]) == 1

    def test_analyze_dependency_graph_aliased(self, analyzer):
        entities = [_make_import_entity("std::io::Result as IoResult")]
        analysis = analyzer.analyze_dependency_graph(entities)
        assert len(analysis["aliased_imports"]) == 1

    def test_analyze_dependency_graph_empty(self, analyzer):
        analysis = analyzer.analyze_dependency_graph([])
        assert analysis["total_imports"] == 0

    # --- get_dependency_complexity_score ---

    def test_dependency_complexity_score_zero(self, analyzer):
        score = analyzer.get_dependency_complexity_score([])
        assert score == 0.0

    def test_dependency_complexity_score_positive(self, analyzer):
        entities = [
            _make_import_entity("serde::Deserialize"),  # external: 0.3
            _make_import_entity("std::io"),  # std: 0.1
            _make_import_entity("crate::models::User"),  # relative: 0.2
        ]
        score = analyzer.get_dependency_complexity_score(entities)
        assert abs(score - 0.6) < 0.01

    def test_dependency_complexity_score_glob_adds_more(self, analyzer):
        entities_without_glob = [_make_import_entity("std::io")]
        entities_with_glob = [_make_import_entity("std::io::*")]
        score_without = analyzer.get_dependency_complexity_score(entities_without_glob)
        score_with = analyzer.get_dependency_complexity_score(entities_with_glob)
        assert score_with > score_without

    # --- suggest_import_optimizations ---

    def test_suggest_no_optimizations(self, analyzer):
        entities = [_make_import_entity("std::io")]
        suggestions = analyzer.suggest_import_optimizations(entities)
        assert suggestions == []

    def test_suggest_reduce_glob_imports(self, analyzer):
        entities = [_make_import_entity(f"mod{i}::*") for i in range(5)]
        suggestions = analyzer.suggest_import_optimizations(entities)
        types = [s["type"] for s in suggestions]
        assert "reduce_glob_imports" in types

    def test_suggest_review_aliases(self, analyzer):
        entities = [_make_import_entity(f"mod{i}::Item as Alias{i}") for i in range(7)]
        suggestions = analyzer.suggest_import_optimizations(entities)
        types = [s["type"] for s in suggestions]
        assert "review_aliases" in types

    def test_suggest_reduce_external_deps(self, analyzer):
        entities = [_make_import_entity(f"crate{i}::Thing") for i in range(12)]
        suggestions = analyzer.suggest_import_optimizations(entities)
        types = [s["type"] for s in suggestions]
        assert "reduce_external_deps" in types

    # --- tree-sitter fallback methods ---

    def test_tree_sitter_fallback_extract_use_declarations(self, analyzer):
        assert analyzer.extract_use_declarations_tree_sitter(MagicMock(), FP, b"") == []

    def test_tree_sitter_fallback_extract_use_relationships(self, analyzer):
        assert analyzer.extract_use_relationships_tree_sitter(MagicMock(), [], FP) == []

    def test_tree_sitter_fallback_extract_function_calls(self, analyzer):
        assert analyzer.extract_function_calls_tree_sitter(MagicMock(), [], FP) == []

    def test_tree_sitter_fallback_find_containing_function(self, analyzer):
        assert analyzer._find_containing_function(MagicMock(), []) is None

    def test_tree_sitter_fallback_node_to_location(self, analyzer):
        loc = analyzer._node_to_location(MagicMock(), FP)
        assert isinstance(loc, SourceLocation)
        assert loc.file_path == str(FP)


# ============================================================================
# RustCargoManager tests
# ============================================================================


class TestRustCargoManager:
    """Tests for RustCargoManager."""

    @pytest.fixture
    def manager(self):
        from processing.languages.rust.rust_cargo_manager import RustCargoManager

        return RustCargoManager()

    # --- find_cargo_toml ---

    def test_find_cargo_toml_found(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text("[package]\nname = 'test'\n")
        src = tmp_path / "src"
        src.mkdir()
        rs_file = src / "main.rs"
        rs_file.write_text("fn main() {}")

        result = manager.find_cargo_toml(rs_file)
        assert result == cargo

    def test_find_cargo_toml_not_found(self, manager, tmp_path):
        rs_file = tmp_path / "main.rs"
        rs_file.write_text("fn main() {}")
        result = manager.find_cargo_toml(rs_file)
        assert result is None

    def test_find_cargo_toml_directory_input(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text("[package]\nname = 'test'\n")
        sub = tmp_path / "src"
        sub.mkdir()
        result = manager.find_cargo_toml(sub)
        assert result == cargo

    # --- parse_cargo_toml ---

    def test_parse_cargo_toml(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[package]\nname = "myapp"\nversion = "1.0.0"\n')
        data = manager.parse_cargo_toml(cargo)
        assert data["package"]["name"] == "myapp"
        assert data["package"]["version"] == "1.0.0"

    def test_parse_cargo_toml_caching(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[package]\nname = "cached"\n')
        data1 = manager.parse_cargo_toml(cargo)
        data2 = manager.parse_cargo_toml(cargo)
        assert data1 is data2  # Same object from cache

    def test_parse_cargo_toml_bad_file(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text("{{invalid toml")
        data = manager.parse_cargo_toml(cargo)
        assert data == {}

    def test_parse_cargo_toml_missing_file(self, manager):
        data = manager.parse_cargo_toml(Path("/nonexistent/Cargo.toml"))
        assert data == {}

    # --- get_project_metadata ---

    def test_get_project_metadata(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text(
            '[package]\nname = "myapp"\nversion = "2.0.0"\nedition = "2021"\n'
            'authors = ["Alice"]\ndescription = "desc"\nlicense = "MIT"\n'
        )
        meta = manager.get_project_metadata(cargo)
        assert meta["name"] == "myapp"
        assert meta["version"] == "2.0.0"
        assert meta["edition"] == "2021"
        assert "Alice" in meta["authors"]
        assert meta["description"] == "desc"
        assert meta["license"] == "MIT"
        assert meta["is_workspace"] is False

    def test_get_project_metadata_defaults(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text("")
        meta = manager.get_project_metadata(cargo)
        assert meta["name"] == "unknown"
        assert meta["version"] == "0.0.0"
        assert meta["edition"] == "2021"

    def test_get_project_metadata_workspace(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[workspace]\nmembers = ["crate_a", "crate_b"]\n')
        meta = manager.get_project_metadata(cargo)
        assert meta["is_workspace"] is True
        assert "crate_a" in meta["workspace_members"]

    # --- get_dependencies ---

    def test_get_dependencies(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text(
            '[dependencies]\nserde = "1.0"\n'
            '[dev-dependencies]\ntokio = "1.0"\n'
            '[build-dependencies]\ncc = "1.0"\n'
        )
        deps = manager.get_dependencies(cargo)
        assert len(deps["dependencies"]) == 1
        assert deps["dependencies"][0]["name"] == "serde"
        assert deps["dependencies"][0]["version"] == "1.0"
        assert len(deps["dev_dependencies"]) == 1
        assert deps["dev_dependencies"][0]["name"] == "tokio"
        assert len(deps["build_dependencies"]) == 1
        assert deps["build_dependencies"][0]["name"] == "cc"

    def test_get_dependencies_empty(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text("")
        deps = manager.get_dependencies(cargo)
        assert deps["dependencies"] == []
        assert deps["dev_dependencies"] == []
        assert deps["build_dependencies"] == []

    # --- _parse_dependency_spec ---

    def test_parse_dependency_spec_string(self, manager):
        info = manager._parse_dependency_spec("serde", "1.0")
        assert info["name"] == "serde"
        assert info["version"] == "1.0"
        assert info["source"] == "crates.io"

    def test_parse_dependency_spec_dict_version(self, manager):
        spec = {"version": "1.0", "features": ["derive"], "optional": True}
        info = manager._parse_dependency_spec("serde", spec)
        assert info["version"] == "1.0"
        assert info["features"] == ["derive"]
        assert info["optional"] is True
        assert info["source"] == "crates.io"

    def test_parse_dependency_spec_git(self, manager):
        spec = {"git": "https://github.com/user/repo.git", "branch": "main"}
        info = manager._parse_dependency_spec("mylib", spec)
        assert info["source"] == "git"
        assert info["git"] == "https://github.com/user/repo.git"
        assert info["branch"] == "main"

    def test_parse_dependency_spec_path(self, manager):
        spec = {"path": "../my-crate"}
        info = manager._parse_dependency_spec("mylib", spec)
        assert info["source"] == "path"
        assert info["path"] == "../my-crate"

    def test_parse_dependency_spec_tag_and_rev(self, manager):
        spec = {"git": "https://example.com/repo", "tag": "v1.0", "rev": "abc123"}
        info = manager._parse_dependency_spec("mylib", spec)
        assert info["tag"] == "v1.0"
        assert info["rev"] == "abc123"

    def test_parse_dependency_spec_default_features_false(self, manager):
        spec = {"version": "1.0", "default-features": False}
        info = manager._parse_dependency_spec("serde", spec)
        assert info["default_features"] is False

    # --- get_build_configuration ---

    def test_get_build_configuration(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text(
            '[[bin]]\nname = "cli"\npath = "src/main.rs"\n'
            "[features]\nfull = []\n"
            "[profile.release]\nopt-level = 3\n"
        )
        config = manager.get_build_configuration(cargo)
        assert len(config["bin"]) == 1
        assert config["bin"][0]["name"] == "cli"
        assert "full" in config["features"]
        assert config["profile"]["release"]["opt-level"] == 3

    def test_get_build_configuration_empty(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text("")
        config = manager.get_build_configuration(cargo)
        assert config["bin"] == []
        assert config["lib"] == {}
        assert config["features"] == {}

    # --- get_workspace_info ---

    def test_get_workspace_info_not_workspace(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[package]\nname = "single"\n')
        result = manager.get_workspace_info(cargo)
        assert result is None

    def test_get_workspace_info_workspace(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text(
            '[workspace]\nmembers = ["a", "b"]\nexclude = ["c"]\ndefault-members = ["a"]\n'
        )
        info = manager.get_workspace_info(cargo)
        assert info is not None
        assert "a" in info["members"]
        assert "b" in info["members"]
        assert "c" in info["exclude"]
        assert "a" in info["default_members"]

    # --- find_source_directories ---

    def test_find_source_directories_lib(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[package]\nname = "mylib"\n')
        src = tmp_path / "src"
        src.mkdir()
        (src / "lib.rs").write_text("// lib")
        dirs = manager.find_source_directories(cargo)
        assert "lib" in dirs
        assert dirs["lib"] == src

    def test_find_source_directories_bin(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[package]\nname = "mybin"\n')
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.rs").write_text("fn main() {}")
        dirs = manager.find_source_directories(cargo)
        assert "bin" in dirs

    def test_find_source_directories_custom_bin(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[[bin]]\nname = "cli"\npath = "src/cli.rs"\n')
        src = tmp_path / "src"
        src.mkdir()
        (src / "cli.rs").write_text("fn main() {}")
        dirs = manager.find_source_directories(cargo)
        assert "bin_cli" in dirs

    def test_find_source_directories_empty(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[package]\nname = "empty"\n')
        dirs = manager.find_source_directories(cargo)
        assert dirs == {}

    # --- get_manifest_entities ---

    def test_get_manifest_entities_project(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text(
            '[package]\nname = "myapp"\nversion = "1.0.0"\n'
            '[dependencies]\nserde = "1.0"\ntokio = "1.0"\n'
        )
        entities = manager.get_manifest_entities(cargo)
        project_entities = [e for e in entities if e.properties.get("project_type")]
        assert len(project_entities) == 1
        assert project_entities[0].name == "myapp"
        assert project_entities[0].properties["version"] == "1.0.0"

    def test_get_manifest_entities_dependencies(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text(
            '[package]\nname = "myapp"\nversion = "1.0.0"\n'
            '[dependencies]\nserde = "1.0"\n'
            '[dev-dependencies]\ntokio = "1.0"\n'
        )
        entities = manager.get_manifest_entities(cargo)
        dep_entities = [e for e in entities if e.type == EntityType.IMPORT]
        dep_names = {e.name for e in dep_entities}
        assert "serde" in dep_names
        assert "tokio" in dep_names

    def test_get_manifest_entities_empty(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text("")
        entities = manager.get_manifest_entities(cargo)
        # Should have at least the project entity with defaults
        assert len(entities) >= 1

    # --- is_test_file ---

    def test_is_test_file_in_tests_dir(self, manager, tmp_path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "integration.rs"
        test_file.write_text("fn test() {}")
        assert manager.is_test_file(test_file) is True

    def test_is_test_file_suffix_pattern(self, manager, tmp_path):
        f = tmp_path / "foo_test.rs"
        f.write_text("fn test() {}")
        assert manager.is_test_file(f) is True

    def test_is_test_file_prefix_pattern(self, manager, tmp_path):
        f = tmp_path / "test_bar.rs"
        f.write_text("fn test() {}")
        assert manager.is_test_file(f) is True

    def test_is_test_file_content_test_attr(self, manager, tmp_path):
        f = tmp_path / "normal.rs"
        f.write_text("#[test]\nfn it_works() { assert!(true); }")
        assert manager.is_test_file(f) is True

    def test_is_test_file_content_cfg_test(self, manager, tmp_path):
        f = tmp_path / "normal.rs"
        f.write_text("#[cfg(test)]\nmod tests { fn it_works() {} }")
        assert manager.is_test_file(f) is True

    def test_is_test_file_regular_file(self, manager, tmp_path):
        f = tmp_path / "lib.rs"
        f.write_text("pub fn add(a: i32, b: i32) -> i32 { a + b }")
        assert manager.is_test_file(f) is False

    def test_is_test_file_unreadable(self, manager):
        """is_test_file should return False for unreadable/missing files."""
        assert manager.is_test_file(Path("/nonexistent/file.rs")) is False

    # --- get_crate_type ---

    def test_get_crate_type_lib(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[package]\nname = "test"\n')
        src = tmp_path / "src"
        src.mkdir()
        lib_rs = src / "lib.rs"
        lib_rs.write_text("pub fn add() {}")
        assert manager.get_crate_type(cargo, lib_rs) == "lib"

    def test_get_crate_type_bin(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[package]\nname = "test"\n')
        src = tmp_path / "src"
        src.mkdir()
        main_rs = src / "main.rs"
        main_rs.write_text("fn main() {}")
        assert manager.get_crate_type(cargo, main_rs) == "bin"

    def test_get_crate_type_example(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[package]\nname = "test"\n')
        examples = tmp_path / "examples"
        examples.mkdir()
        ex_file = examples / "demo.rs"
        ex_file.write_text("fn main() {}")
        assert manager.get_crate_type(cargo, ex_file) == "example"

    def test_get_crate_type_bench(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[package]\nname = "test"\n')
        benches = tmp_path / "benches"
        benches.mkdir()
        bench_file = benches / "mybench.rs"
        bench_file.write_text("fn main() {}")
        assert manager.get_crate_type(cargo, bench_file) == "bench"

    def test_get_crate_type_test(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[package]\nname = "test"\n')
        tests = tmp_path / "tests"
        tests.mkdir()
        test_file = tests / "integration.rs"
        test_file.write_text("fn test_something() {}")
        assert manager.get_crate_type(cargo, test_file) == "test"

    def test_get_crate_type_custom_bin(self, manager, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[[bin]]\nname = "cli"\npath = "src/bin/cli.rs"\n')
        src = tmp_path / "src" / "bin"
        src.mkdir(parents=True)
        cli = src / "cli.rs"
        cli.write_text("fn main() {}")
        assert manager.get_crate_type(cargo, cli) == "bin"

    def test_get_crate_type_default_lib(self, manager, tmp_path):
        """Files not matching any pattern should default to lib."""
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text('[package]\nname = "test"\n')
        src = tmp_path / "src"
        src.mkdir()
        some_file = src / "utils.rs"
        some_file.write_text("pub fn helper() {}")
        assert manager.get_crate_type(cargo, some_file) == "lib"


# ============================================================================
# RustSyntaxParser fallback coverage
# ============================================================================


class TestRustSyntaxParserFallbacks:
    """Cover the fallback (non-tree-sitter) methods of RustSyntaxParser."""

    @pytest.fixture
    def syntax_parser(self):
        from processing.languages.rust.rust_syntax_parser import RustSyntaxParser

        return RustSyntaxParser()

    def test_extract_modules_tree_sitter_fallback(self, syntax_parser):
        result = syntax_parser.extract_modules_tree_sitter(MagicMock(), FP, b"pub mod test;")
        assert result == []

    def test_extract_structs_tree_sitter_fallback(self, syntax_parser):
        result = syntax_parser.extract_structs_tree_sitter(MagicMock(), FP, b"struct Foo {}")
        assert result == []

    def test_extract_enums_tree_sitter_fallback(self, syntax_parser):
        result = syntax_parser.extract_enums_tree_sitter(MagicMock(), FP, b"enum Bar {}")
        assert result == []

    def test_extract_traits_tree_sitter_fallback(self, syntax_parser):
        result = syntax_parser.extract_traits_tree_sitter(MagicMock(), FP, b"trait Baz {}")
        assert result == []

    def test_extract_functions_tree_sitter_fallback(self, syntax_parser):
        result = syntax_parser.extract_functions_tree_sitter(MagicMock(), FP, b"fn foo() {}")
        assert result == []

    def test_extract_impl_blocks_tree_sitter_fallback(self, syntax_parser):
        result = syntax_parser.extract_impl_blocks_tree_sitter(MagicMock(), FP, b"impl Foo {}")
        assert result == []

    def test_extract_constants_tree_sitter_fallback(self, syntax_parser):
        result = syntax_parser.extract_constants_tree_sitter(MagicMock(), FP, b"const X: i32 = 1;")
        assert result == []

    def test_extract_visibility_fallback(self, syntax_parser):
        result = syntax_parser._extract_visibility(MagicMock(), b"pub struct Foo {}")
        assert result == "private"

    # --- Regex parsing methods ---

    def test_extract_modules_regex(self, syntax_parser):
        content = "pub mod network;\nmod internal;"
        entities = syntax_parser.extract_modules_regex(content, FP)
        assert len(entities) == 2
        names = {e.name for e in entities}
        assert "network" in names
        assert "internal" in names
        pub_mod = next(e for e in entities if e.name == "network")
        assert pub_mod.visibility == "public"
        priv_mod = next(e for e in entities if e.name == "internal")
        assert priv_mod.visibility == "private"

    def test_extract_structs_regex(self, syntax_parser):
        content = "pub struct Config {\n    name: String,\n}\nstruct Private {}"
        entities = syntax_parser.extract_structs_regex(content, FP)
        assert len(entities) == 2
        names = {e.name for e in entities}
        assert "Config" in names
        assert "Private" in names

    def test_extract_enums_regex(self, syntax_parser):
        content = "pub enum Color {\n    Red,\n    Green,\n}\nenum Internal { A }"
        entities = syntax_parser.extract_enums_regex(content, FP)
        assert len(entities) == 2

    def test_extract_traits_regex(self, syntax_parser):
        content = "pub trait Handler {\n    fn handle(&self);\n}\ntrait Private {}"
        entities = syntax_parser.extract_traits_regex(content, FP)
        assert len(entities) == 2
        pub_trait = next(e for e in entities if e.name == "Handler")
        assert pub_trait.visibility == "public"

    def test_extract_functions_regex(self, syntax_parser):
        content = "pub fn calculate(x: i32) -> i32 { x }\nfn helper() {}\npub async fn fetch() {}"
        entities = syntax_parser.extract_functions_regex(content, FP)
        names = {e.name for e in entities}
        assert "calculate" in names
        assert "helper" in names
        assert "fetch" in names
        fetch_fn = next(e for e in entities if e.name == "fetch")
        assert fetch_fn.is_async is True

    def test_extract_functions_regex_skips_keywords(self, syntax_parser):
        content = "fn main() {}\nfn new() {}\nfn default() {}"
        entities = syntax_parser.extract_functions_regex(content, FP)
        names = {e.name for e in entities}
        assert "main" not in names
        assert "new" not in names
        assert "default" not in names

    def test_extract_impl_blocks_regex_trait(self, syntax_parser):
        content = "impl Display for Config {\n    fn fmt(&self) -> String { String::new() }\n}"
        entities = syntax_parser.extract_impl_blocks_regex(content, FP)
        assert len(entities) >= 1
        trait_impl = next((e for e in entities if e.properties.get("impl_type") == "trait"), None)
        assert trait_impl is not None
        assert trait_impl.properties["trait_name"] == "Display"
        assert trait_impl.properties["type_name"] == "Config"

    def test_extract_impl_blocks_regex_inherent(self, syntax_parser):
        content = "impl Config {\n    pub fn new() -> Self { Config {} }\n}"
        entities = syntax_parser.extract_impl_blocks_regex(content, FP)
        assert len(entities) >= 1
        inherent = next((e for e in entities if e.properties.get("impl_type") == "inherent"), None)
        assert inherent is not None
        assert inherent.properties["type_name"] == "Config"

    def test_extract_macros_regex(self, syntax_parser):
        content = "macro_rules! my_vec {\n    () => { Vec::new() };\n}"
        entities = syntax_parser.extract_macros_regex(content, FP)
        assert len(entities) == 1
        assert entities[0].name == "my_vec"
        assert entities[0].type == EntityType.MACRO


# ============================================================================
# RustDependencyAnalyzer tree-sitter method tests (covers lines 227-368)
# ============================================================================


class TestRustDependencyAnalyzerTreeSitter:
    """Tests for tree-sitter methods of RustDependencyAnalyzer.

    When tree-sitter IS available, these are real methods that need
    to be exercised with mocked tree-sitter nodes.
    """

    @pytest.fixture
    def ts_processor(self):
        """Create a mock tree-sitter processor."""
        proc = MagicMock()
        return proc

    @pytest.fixture
    def analyzer(self, ts_processor):
        from processing.languages.rust.rust_dependency_analyzer import (
            RustDependencyAnalyzer,
        )

        return RustDependencyAnalyzer(tree_sitter_processor=ts_processor)

    def test_extract_use_declarations_tree_sitter_no_processor(self):
        """extract_use_declarations_tree_sitter returns [] when no processor."""
        from processing.languages.rust.rust_dependency_analyzer import RustDependencyAnalyzer

        a = RustDependencyAnalyzer(tree_sitter_processor=None)
        result = a.extract_use_declarations_tree_sitter(MagicMock(), FP, b"use std::io;")
        assert result == []

    def test_extract_use_declarations_tree_sitter_with_nodes(self, analyzer, ts_processor):
        """extract_use_declarations_tree_sitter extracts entities from use nodes."""
        from processing.languages.rust.rust_dependency_analyzer import TREE_SITTER_AVAILABLE

        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")

        # Create a mock use node
        use_node = MagicMock()
        use_node.start_point = (0, 0)
        use_node.end_point = (0, 15)

        ts_processor._find_nodes_by_type.return_value = [use_node]
        ts_processor._get_node_text.return_value = "use std::io;"

        tree = MagicMock()
        content = b"use std::io;"

        result = analyzer.extract_use_declarations_tree_sitter(tree, FP, content)
        assert len(result) == 1
        assert result[0].type == EntityType.IMPORT
        assert result[0].name == "std::io"
        assert result[0].properties["use_path"] == "std::io"

    def test_extract_use_relationships_tree_sitter_with_entities(self, analyzer):
        """extract_use_relationships_tree_sitter creates IMPORTS relationships."""
        from processing.languages.rust.rust_dependency_analyzer import TREE_SITTER_AVAILABLE

        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")

        file_entity = CoreCodeEntity(
            id=f"file:{FP}",
            name=FP.name,
            type=EntityType.FILE,
            language="rust",
            file_path=str(FP),
            line_number=1,
        )
        import_entity = CoreCodeEntity(
            id="use:std::io:test.rs",
            name="std::io",
            type=EntityType.IMPORT,
            language="rust",
            file_path=str(FP),
            line_number=1,
            column_start=0,
            column_end=15,
        )

        tree = MagicMock()
        result = analyzer.extract_use_relationships_tree_sitter(
            tree, [file_entity, import_entity], FP
        )
        assert len(result) == 1
        assert result[0].source_id == file_entity.id
        assert result[0].target_id == import_entity.id

    def test_extract_use_relationships_tree_sitter_no_file_entity(self, analyzer):
        """When no FILE entity exists, one is created automatically."""
        from processing.languages.rust.rust_dependency_analyzer import TREE_SITTER_AVAILABLE

        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")

        import_entity = CoreCodeEntity(
            id="use:std::io:test.rs",
            name="std::io",
            type=EntityType.IMPORT,
            language="rust",
            file_path=str(FP),
            line_number=1,
            column_start=0,
            column_end=15,
        )

        tree = MagicMock()
        result = analyzer.extract_use_relationships_tree_sitter(tree, [import_entity], FP)
        assert len(result) == 1
        assert result[0].source_id == f"file:{FP}"

    def test_extract_function_calls_tree_sitter_no_processor(self):
        """extract_function_calls_tree_sitter returns [] when no processor."""
        from processing.languages.rust.rust_dependency_analyzer import (
            TREE_SITTER_AVAILABLE,
            RustDependencyAnalyzer,
        )

        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")

        a = RustDependencyAnalyzer(tree_sitter_processor=None)
        result = a.extract_function_calls_tree_sitter(MagicMock(), [], FP)
        assert result == []

    def test_extract_function_calls_tree_sitter_with_calls(self, analyzer, ts_processor):
        """extract_function_calls_tree_sitter extracts call relationships."""
        from processing.languages.rust.rust_dependency_analyzer import TREE_SITTER_AVAILABLE

        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")

        # Create a function entity
        fn_entity = CoreCodeEntity(
            id="fn:main:test.rs",
            name="main",
            type=EntityType.FUNCTION,
            language="rust",
            file_path=str(FP),
            line_number=5,
        )

        # Mock the call node hierarchy
        call_node = MagicMock()
        call_node.start_point = (5, 4)
        call_node.end_point = (5, 20)

        # Mock _find_containing_function by setting up parent chain
        fn_item_node = MagicMock()
        fn_item_node.type = "function_item"
        fn_item_node.start_point = (4, 0)
        fn_item_node.parent = None
        call_node.parent = fn_item_node

        # Mock identifier child
        identifier_node = MagicMock()
        identifier_node.type = "identifier"

        ts_processor._find_nodes_by_type.return_value = [call_node]
        ts_processor._get_child_by_type.return_value = identifier_node
        ts_processor._get_node_text.return_value = "helper_fn"

        tree = MagicMock()
        tree.text = b"helper_fn()"

        result = analyzer.extract_function_calls_tree_sitter(tree, [fn_entity], FP)
        assert len(result) == 1
        assert result[0].properties["function_name"] == "helper_fn"

    def test_extract_function_calls_no_identifier(self, analyzer, ts_processor):
        """When call_node has no identifier or field_expression child, skip it."""
        from processing.languages.rust.rust_dependency_analyzer import TREE_SITTER_AVAILABLE

        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")

        fn_entity = CoreCodeEntity(
            id="fn:main:test.rs",
            name="main",
            type=EntityType.FUNCTION,
            language="rust",
            file_path=str(FP),
            line_number=5,
        )

        call_node = MagicMock()
        fn_item = MagicMock()
        fn_item.type = "function_item"
        fn_item.start_point = (4, 0)
        fn_item.parent = None
        call_node.parent = fn_item

        ts_processor._find_nodes_by_type.return_value = [call_node]
        ts_processor._get_child_by_type.return_value = None  # No identifier

        tree = MagicMock()
        tree.text = b""

        result = analyzer.extract_function_calls_tree_sitter(tree, [fn_entity], FP)
        assert result == []

    def test_extract_function_calls_no_containing_function(self, analyzer, ts_processor):
        """When call is not inside a function, skip it."""
        from processing.languages.rust.rust_dependency_analyzer import TREE_SITTER_AVAILABLE

        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")

        call_node = MagicMock()
        call_node.parent = None  # No parent function

        ts_processor._find_nodes_by_type.return_value = [call_node]

        tree = MagicMock()
        tree.text = b""

        result = analyzer.extract_function_calls_tree_sitter(tree, [], FP)
        assert result == []

    def test_find_containing_function_no_processor(self):
        """_find_containing_function returns None when no processor."""
        from processing.languages.rust.rust_dependency_analyzer import (
            TREE_SITTER_AVAILABLE,
            RustDependencyAnalyzer,
        )

        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")

        a = RustDependencyAnalyzer(tree_sitter_processor=None)
        result = a._find_containing_function(MagicMock(), [])
        assert result is None

    def test_find_containing_function_walks_up(self, analyzer):
        """_find_containing_function walks up node tree to find function."""
        from processing.languages.rust.rust_dependency_analyzer import TREE_SITTER_AVAILABLE

        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")

        fn_entity = CoreCodeEntity(
            id="fn:test:test.rs",
            name="test",
            type=EntityType.FUNCTION,
            language="rust",
            file_path=str(FP),
            line_number=10,
        )

        fn_node = MagicMock()
        fn_node.type = "function_item"
        fn_node.start_point = (9, 0)  # line 10, 0-indexed = 9
        fn_node.parent = None

        inner_node = MagicMock()
        inner_node.parent = fn_node

        result = analyzer._find_containing_function(inner_node, [fn_entity])
        assert result is fn_entity

    def test_find_containing_function_no_match(self, analyzer):
        """_find_containing_function returns None if no entity matches."""
        from processing.languages.rust.rust_dependency_analyzer import TREE_SITTER_AVAILABLE

        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")

        fn_node = MagicMock()
        fn_node.type = "function_item"
        fn_node.start_point = (99, 0)
        fn_node.parent = None

        inner_node = MagicMock()
        inner_node.parent = fn_node

        result = analyzer._find_containing_function(inner_node, [])
        assert result is None

    def test_node_to_location(self, analyzer):
        """_node_to_location converts tree-sitter node to SourceLocation."""
        from processing.languages.rust.rust_dependency_analyzer import TREE_SITTER_AVAILABLE

        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")

        node = MagicMock()
        node.start_point = (4, 2)
        node.end_point = (6, 10)

        loc = analyzer._node_to_location(node, FP)
        assert loc.file_path == str(FP)
        assert loc.line_start == 5
        assert loc.line_end == 7
        assert loc.column_start == 2
        assert loc.column_end == 10

    def test_extract_function_calls_field_expression(self, analyzer, ts_processor):
        """When identifier child not found, uses field_expression child."""
        from processing.languages.rust.rust_dependency_analyzer import TREE_SITTER_AVAILABLE

        if not TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")

        fn_entity = CoreCodeEntity(
            id="fn:main:test.rs",
            name="main",
            type=EntityType.FUNCTION,
            language="rust",
            file_path=str(FP),
            line_number=5,
        )

        call_node = MagicMock()
        call_node.start_point = (5, 4)
        call_node.end_point = (5, 20)
        fn_item = MagicMock()
        fn_item.type = "function_item"
        fn_item.start_point = (4, 0)
        fn_item.parent = None
        call_node.parent = fn_item

        field_expr = MagicMock()
        field_expr.type = "field_expression"

        # First call for "identifier" returns None, second for "field_expression" returns field_expr
        ts_processor._get_child_by_type.side_effect = [None, field_expr]
        ts_processor._get_node_text.return_value = "self.method"
        ts_processor._find_nodes_by_type.return_value = [call_node]

        tree = MagicMock()
        tree.text = b"self.method()"

        result = analyzer.extract_function_calls_tree_sitter(tree, [fn_entity], FP)
        assert len(result) == 1
        assert result[0].properties["function_name"] == "self.method"
