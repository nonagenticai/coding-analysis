"""Coverage boost tests for BaseTreeSitterProcessor — targets uncovered paths.

Focuses on:
- _initialize_impl: ImportError handling, general Exception
- _ensure_grammar_available: unknown grammar
- _download_and_compile_grammar: paths for wasm, gcc fallback, TypeScript special case
- parse_file: UnicodeDecodeError, TypeError
- calculate_complexity: with many node types
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from models.core_models import CoreCodeEntity, EntityRelation, EntityType
from processing.languages.base import LanguageConfig, ParserType
from processing.parsers.tree_sitter_base import BaseTreeSitterProcessor


class ConcreteProcessor(BaseTreeSitterProcessor):
    async def extract_entities(self, ast: Any, file_path: Path) -> list[CoreCodeEntity]:
        return []

    async def extract_relationships(
        self, ast: Any, entities: list[CoreCodeEntity], file_path: Path
    ) -> list[EntityRelation]:
        return []


def _make_config(name="python", grammar="python"):
    from processing.languages.base import LanguageFeatures

    return LanguageConfig(
        name=name,
        display_name=name.capitalize(),
        extensions=[f".{name[:2]}"],
        tree_sitter_grammar=grammar,
        parser_type=ParserType.TREE_SITTER,
        features=LanguageFeatures(),
    )


# ── _initialize_impl ────────────────────────────────────────────────


class TestInitializeImpl:
    @pytest.mark.asyncio
    async def test_initialize_with_nonexistent_grammar_module(self):
        """ImportError path when language module doesn't exist."""
        config = _make_config(name="fakeLang", grammar="nonexistent_language")
        p = ConcreteProcessor(config)
        with pytest.raises(RuntimeError, match="Failed to initialize"):
            await p._initialize_impl()

    @pytest.mark.asyncio
    async def test_initialize_no_grammar_configured(self):
        from processing.languages.base import LanguageFeatures

        config = LanguageConfig(
            name="test",
            display_name="Test",
            extensions=[".tst"],
            tree_sitter_grammar=None,
            parser_type=ParserType.TREE_SITTER,
            features=LanguageFeatures(),
        )
        p = ConcreteProcessor(config)
        with pytest.raises(ValueError, match="No Tree-sitter grammar"):
            await p._initialize_impl()


# ── _ensure_grammar_available ────────────────────────────────────────


class TestEnsureGrammarAvailable:
    @pytest.mark.asyncio
    async def test_grammar_none_raises(self):
        from processing.languages.base import LanguageFeatures

        config = LanguageConfig(
            name="test",
            display_name="Test",
            extensions=[".t"],
            tree_sitter_grammar=None,
            parser_type=ParserType.TREE_SITTER,
            features=LanguageFeatures(),
        )
        p = ConcreteProcessor(config)
        with pytest.raises(ValueError, match="not configured"):
            await p._ensure_grammar_available()


# ── _download_and_compile_grammar ────────────────────────────────────


class TestDownloadAndCompileGrammar:
    @pytest.mark.asyncio
    async def test_unknown_grammar_returns_early(self):
        config = _make_config()
        p = ConcreteProcessor(config)
        grammar_dir = Path("/tmp/test_grammars_unknown")
        grammar_dir.mkdir(exist_ok=True)
        # "xyz_unknown_grammar" is not in GRAMMAR_REPOSITORIES
        await p._download_and_compile_grammar(grammar_dir, "xyz_unknown_grammar")
        # Should return without creating anything

    @pytest.mark.asyncio
    async def test_git_clone_failure(self):
        config = _make_config()
        p = ConcreteProcessor(config)
        grammar_dir = Path("/tmp/test_grammars_fail")
        grammar_dir.mkdir(exist_ok=True)
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "git")):
            # Should not raise, just log
            await p._download_and_compile_grammar(grammar_dir, "python")


# ── parse_file ───────────────────────────────────────────────────────


class TestParseFileEdgeCases:
    @pytest.fixture
    def processor(self):
        p = ConcreteProcessor(_make_config())
        mock_tree = MagicMock()
        mock_tree.root_node.type = "program"
        mock_tree.root_node.children = []
        mock_parser = MagicMock()
        mock_parser.parse.return_value = mock_tree
        p.parser = mock_parser
        return p

    @pytest.mark.asyncio
    async def test_parse_file_unicode_error(self, processor):
        processor.parser.parse.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "bad")
        result = await processor.parse_file(Path("test.py"), "bad content")
        assert result.success is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_parse_file_type_error(self, processor):
        processor.parser.parse.side_effect = TypeError("unexpected type")
        result = await processor.parse_file(Path("test.py"), "x = 1")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_parse_file_os_error(self, processor):
        processor.parser.parse.side_effect = OSError("disk error")
        result = await processor.parse_file(Path("test.py"), "x = 1")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_parse_file_records_parse_time(self, processor):
        result = await processor.parse_file(Path("test.py"), "x = 1")
        assert result.parse_time_ms >= 0

    @pytest.mark.asyncio
    async def test_parse_file_multiline(self, processor):
        code = "x = 1\ny = 2\nz = x + y\n"
        result = await processor.parse_file(Path("test.py"), code)
        assert result.success is True
        assert result.lines_of_code == 3


# ── calculate_complexity ─────────────────────────────────────────────


class TestCalculateComplexityExtended:
    @pytest.fixture
    def processor(self):
        return ConcreteProcessor(_make_config())

    @pytest.mark.asyncio
    async def test_complexity_with_all_node_types(self, processor):
        """Build a tree with many different complexity-contributing node types."""
        node_types = [
            "if_statement",
            "while_statement",
            "for_statement",
            "switch_statement",
            "case_statement",
            "catch_clause",
            "conditional_expression",
            "try_statement",
            "do_statement",
            "break_statement",
            "continue_statement",
            "return_statement",
            "throw_statement",
            "assert_statement",
            "lambda_expression",
        ]

        children = []
        for nt in node_types:
            child = MagicMock()
            child.type = nt
            child.children = []
            children.append(child)

        root = MagicMock()
        root.type = "program"
        root.children = children

        tree = MagicMock()
        tree.root_node = root

        complexity = await processor.calculate_complexity(tree)
        # base(1) + 15 nodes
        assert complexity == 16.0

    @pytest.mark.asyncio
    async def test_complexity_with_non_complexity_nodes(self, processor):
        """Nodes that don't contribute to complexity."""
        child = MagicMock()
        child.type = "identifier"
        child.children = []

        root = MagicMock()
        root.type = "program"
        root.children = [child]

        tree = MagicMock()
        tree.root_node = root

        complexity = await processor.calculate_complexity(tree)
        assert complexity == 1.0  # base only

    @pytest.mark.asyncio
    async def test_complexity_deeply_nested(self, processor):
        """Deeply nested tree still counts all complexity nodes."""
        leaf = MagicMock()
        leaf.type = "return_statement"
        leaf.children = []

        inner = MagicMock()
        inner.type = "if_statement"
        inner.children = [leaf]

        middle = MagicMock()
        middle.type = "for_statement"
        middle.children = [inner]

        outer = MagicMock()
        outer.type = "while_statement"
        outer.children = [middle]

        root = MagicMock()
        root.type = "program"
        root.children = [outer]

        tree = MagicMock()
        tree.root_node = root

        complexity = await processor.calculate_complexity(tree)
        # base(1) + while(1) + for(1) + if(1) + return(1)
        assert complexity == 5.0


# ── Helper method edge cases ─────────────────────────────────────────


class TestHelperMethodsExtended:
    @pytest.fixture
    def processor(self):
        return ConcreteProcessor(_make_config())

    def test_get_node_text_empty_node(self, processor):
        node = MagicMock()
        node.start_byte = 5
        node.end_byte = 5
        content = b"hello world"
        result = processor._get_node_text(node, content)
        assert result == ""

    def test_find_entity_by_location_multiple_entities(self, processor):
        e1 = CoreCodeEntity(
            id="e1",
            name="a",
            type=EntityType.FUNCTION,
            language="py",
            file_path="t.py",
            line_number=1,
            column_start=0,
        )
        e2 = CoreCodeEntity(
            id="e2",
            name="b",
            type=EntityType.FUNCTION,
            language="py",
            file_path="t.py",
            line_number=5,
            column_start=4,
        )
        node = MagicMock()
        node.start_point = (4, 4)  # line 5 (1-indexed), col 4
        result = processor._find_entity_by_location([e1, e2], node)
        assert result is e2

    def test_find_containing_method_deep_nesting(self, processor):
        entity = CoreCodeEntity(
            id="m1",
            name="outer",
            type=EntityType.FUNCTION,
            language="py",
            file_path="t.py",
            line_number=1,
            column_start=0,
        )

        method_node = MagicMock()
        method_node.type = "method_declaration"
        method_node.start_point = (0, 0)
        method_node.parent = None

        middle = MagicMock()
        middle.type = "block"
        middle.parent = method_node

        inner = MagicMock()
        inner.type = "expression"
        inner.parent = middle

        deepest = MagicMock()
        deepest.parent = inner

        result = processor._find_containing_method(deepest, [entity], {"method_declaration"})
        assert result is entity

    def test_get_children_by_type_empty(self, processor):
        parent = MagicMock()
        parent.children = []
        result = processor._get_children_by_type(parent, "identifier")
        assert result == []
