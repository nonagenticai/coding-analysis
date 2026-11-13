"""Round 4 tests for BaseTreeSitterProcessor (107 uncov lines)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from models.core_models import CoreCodeEntity, EntityRelation, EntityType, SourceLocation
from processing.languages.base import LanguageConfig, LanguageFeatures, ParserType

# Import with fallback for circular import scenarios in full test suite
try:
    from processing.parsers.tree_sitter_base import TREE_SITTER_AVAILABLE, BaseTreeSitterProcessor
except ImportError:
    pytest.skip("Cannot import tree_sitter_base (circular import)", allow_module_level=True)


# Create a concrete subclass for testing
class ConcreteTreeSitterProcessor(BaseTreeSitterProcessor):
    """Concrete implementation for testing the base class."""

    async def extract_entities(self, ast: Any, file_path: Path) -> list[CoreCodeEntity]:
        return []

    async def extract_relationships(
        self, ast: Any, entities: list[CoreCodeEntity], file_path: Path
    ) -> list[EntityRelation]:
        return []


def _make_config(name="python", grammar="python"):
    return LanguageConfig(
        name=name,
        display_name=name.capitalize(),
        extensions=[f".{name[:2]}"],
        parser_type=ParserType.TREE_SITTER,
        features=LanguageFeatures(),
        tree_sitter_grammar=grammar,
    )


class TestBaseTreeSitterProcessorInit:
    def test_init(self):
        config = _make_config()
        p = ConcreteTreeSitterProcessor(config)
        assert p.parser is None
        assert p.language is None
        assert p._grammar_path is None
        assert p.config.name == "python"


class TestGrammarRepositories:
    def test_grammar_repos_exist(self):
        repos = BaseTreeSitterProcessor.GRAMMAR_REPOSITORIES
        assert "python" in repos
        assert "javascript" in repos
        assert "java" in repos
        assert "rust" in repos
        assert "cpp" in repos
        assert "c_sharp" in repos
        assert "go" in repos
        assert "typescript" in repos


class TestHelperMethods:
    """Test utility methods that work with mock tree-sitter nodes."""

    @pytest.fixture
    def processor(self):
        return ConcreteTreeSitterProcessor(_make_config())

    def test_get_node_text(self, processor):
        node = MagicMock()
        node.start_byte = 0
        node.end_byte = 5
        content = b"hello world"
        result = processor._get_node_text(node, content)
        assert result == "hello"

    def test_get_node_text_unicode(self, processor):
        node = MagicMock()
        node.start_byte = 0
        node.end_byte = 6
        content = "héllo".encode()
        result = processor._get_node_text(node, content)
        assert "h" in result

    def test_node_to_location(self, processor):
        node = MagicMock()
        node.start_point = (5, 10)
        node.end_point = (8, 20)
        loc = processor._node_to_location(node, Path("test.py"))
        assert isinstance(loc, SourceLocation)
        assert loc.file_path == "test.py"
        assert loc.line_start == 6  # 1-indexed
        assert loc.line_end == 9
        assert loc.column_start == 10
        assert loc.column_end == 20

    def test_get_child_by_type_found(self, processor):
        child1 = MagicMock()
        child1.type = "identifier"
        child2 = MagicMock()
        child2.type = "string"
        parent = MagicMock()
        parent.children = [child1, child2]
        result = processor._get_child_by_type(parent, "identifier")
        assert result is child1

    def test_get_child_by_type_not_found(self, processor):
        parent = MagicMock()
        parent.children = []
        result = processor._get_child_by_type(parent, "identifier")
        assert result is None

    def test_get_children_by_type(self, processor):
        child1 = MagicMock()
        child1.type = "identifier"
        child2 = MagicMock()
        child2.type = "identifier"
        child3 = MagicMock()
        child3.type = "string"
        parent = MagicMock()
        parent.children = [child1, child2, child3]
        result = processor._get_children_by_type(parent, "identifier")
        assert len(result) == 2

    def test_find_nodes_by_type(self, processor):
        # Build a simple tree: root -> child1 (identifier), child2 -> grandchild (identifier)
        grandchild = MagicMock()
        grandchild.type = "identifier"
        grandchild.children = []

        child1 = MagicMock()
        child1.type = "identifier"
        child1.children = []

        child2 = MagicMock()
        child2.type = "method_declaration"
        child2.children = [grandchild]

        root = MagicMock()
        root.children = [child1, child2]

        tree = MagicMock()
        tree.root_node = root

        result = processor._find_nodes_by_type(tree, {"identifier"})
        assert len(result) == 2

    def test_find_nodes_by_type_empty(self, processor):
        root = MagicMock()
        root.type = "program"
        root.children = []
        tree = MagicMock()
        tree.root_node = root

        result = processor._find_nodes_by_type(tree, {"nonexistent"})
        assert result == []

    def test_find_entity_by_location_found(self, processor):
        entity = CoreCodeEntity(
            id="e1",
            name="test",
            type=EntityType.FUNCTION,
            language="python",
            file_path="test.py",
            line_number=6,
            column_start=10,
        )
        node = MagicMock()
        node.start_point = (5, 10)  # line 5 (0-indexed) = line 6 (1-indexed)

        result = processor._find_entity_by_location([entity], node)
        assert result is entity

    def test_find_entity_by_location_not_found(self, processor):
        node = MagicMock()
        node.start_point = (99, 0)
        result = processor._find_entity_by_location([], node)
        assert result is None

    def test_find_containing_method(self, processor):
        entity = CoreCodeEntity(
            id="m1",
            name="my_method",
            type=EntityType.FUNCTION,
            language="python",
            file_path="test.py",
            line_number=3,
            column_start=4,
        )

        # method_node at line 2 (0-indexed), col 4
        method_node = MagicMock()
        method_node.type = "method_declaration"
        method_node.start_point = (2, 4)
        method_node.parent = None

        # inner_node's parent is method_node
        inner_node = MagicMock()
        inner_node.parent = method_node

        result = processor._find_containing_method(inner_node, [entity], {"method_declaration"})
        assert result is entity

    def test_find_containing_method_not_found(self, processor):
        node = MagicMock()
        node.parent = None
        result = processor._find_containing_method(node, [], {"method_declaration"})
        assert result is None


class TestParseFileWithoutTreeSitter:
    @pytest.fixture
    def processor(self):
        return ConcreteTreeSitterProcessor(_make_config())

    @pytest.mark.asyncio
    async def test_parse_file_no_parser(self, processor):
        """Should raise RuntimeError when parser not initialized."""
        with pytest.raises(RuntimeError, match="Parser not initialized"):
            await processor.parse_file(Path("test.py"), "x = 1")


class TestParseFileWithMockedParser:
    @pytest.fixture
    def processor(self):
        p = ConcreteTreeSitterProcessor(_make_config())
        # Mock out the parser
        mock_tree = MagicMock()
        mock_tree.root_node.type = "program"
        mock_tree.root_node.children = []
        mock_parser = MagicMock()
        mock_parser.parse.return_value = mock_tree
        p.parser = mock_parser
        return p

    @pytest.mark.asyncio
    async def test_parse_file_success(self, processor):
        result = await processor.parse_file(Path("test.py"), "x = 1")
        assert result.success is True
        assert result.parser_used == ParserType.TREE_SITTER
        assert result.language == "python"

    @pytest.mark.asyncio
    async def test_parse_file_error(self, processor):
        processor.parser.parse.side_effect = RuntimeError("parse error")
        result = await processor.parse_file(Path("test.py"), "bad code")
        assert result.success is False
        assert len(result.errors) > 0


class TestCalculateComplexity:
    @pytest.fixture
    def processor(self):
        return ConcreteTreeSitterProcessor(_make_config())

    @pytest.mark.asyncio
    async def test_calculate_complexity_simple(self, processor):
        # Build a simple tree with some complexity nodes
        if_node = MagicMock()
        if_node.type = "if_statement"
        if_node.children = []

        for_node = MagicMock()
        for_node.type = "for_statement"
        for_node.children = []

        root = MagicMock()
        root.type = "program"
        root.children = [if_node, for_node]

        tree = MagicMock()
        tree.root_node = root

        complexity = await processor.calculate_complexity(tree)
        assert complexity == 3.0  # base(1) + if(1) + for(1)

    @pytest.mark.asyncio
    async def test_calculate_complexity_empty(self, processor):
        root = MagicMock()
        root.type = "program"
        root.children = []

        tree = MagicMock()
        tree.root_node = root

        complexity = await processor.calculate_complexity(tree)
        assert complexity == 1.0  # base only

    @pytest.mark.asyncio
    async def test_calculate_complexity_nested(self, processor):
        inner_if = MagicMock()
        inner_if.type = "if_statement"
        inner_if.children = []

        outer_if = MagicMock()
        outer_if.type = "if_statement"
        outer_if.children = [inner_if]

        root = MagicMock()
        root.type = "program"
        root.children = [outer_if]

        tree = MagicMock()
        tree.root_node = root

        complexity = await processor.calculate_complexity(tree)
        assert complexity == 3.0  # base(1) + outer(1) + inner(1)


class TestInitializeImpl:
    @pytest.mark.asyncio
    async def test_initialize_no_tree_sitter_available(self):
        p = ConcreteTreeSitterProcessor(_make_config())
        with patch("processing.parsers.tree_sitter_base.TREE_SITTER_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="not available"):
                await p._initialize_impl()

    @pytest.mark.asyncio
    async def test_initialize_no_grammar(self):
        config = LanguageConfig(
            name="test",
            display_name="Test",
            extensions=[".tst"],
            parser_type=ParserType.TREE_SITTER,
            features=LanguageFeatures(),
            tree_sitter_grammar=None,
        )
        p = ConcreteTreeSitterProcessor(config)
        if TREE_SITTER_AVAILABLE:
            with pytest.raises(ValueError, match="No Tree-sitter grammar"):
                await p._initialize_impl()


class TestEnsureGrammarAvailable:
    @pytest.mark.asyncio
    async def test_ensure_grammar_no_grammar_config(self):
        config = LanguageConfig(
            name="test",
            display_name="Test",
            extensions=[".tst"],
            parser_type=ParserType.TREE_SITTER,
            features=LanguageFeatures(),
            tree_sitter_grammar=None,
        )
        p = ConcreteTreeSitterProcessor(config)
        with pytest.raises(ValueError, match="not configured"):
            await p._ensure_grammar_available()

    @pytest.mark.asyncio
    async def test_ensure_grammar_existing_file(self, tmp_path):
        """When the .so file exists, it sets _grammar_path."""
        config = _make_config(name="python", grammar="python")
        p = ConcreteTreeSitterProcessor(config)
        grammar_dir = tmp_path / "grammars"
        grammar_dir.mkdir(parents=True)
        (grammar_dir / "python.so").write_text("fake")
        with patch("pathlib.Path.expanduser", return_value=grammar_dir):
            await p._ensure_grammar_available()
        assert p._grammar_path == grammar_dir / "python.so"

    @pytest.mark.asyncio
    async def test_ensure_grammar_downloads_when_missing(self, tmp_path):
        """When .so missing, _download_and_compile_grammar is called."""
        config = _make_config(name="python", grammar="python")
        p = ConcreteTreeSitterProcessor(config)
        grammar_dir = tmp_path / "grammars"
        grammar_dir.mkdir(parents=True)
        with (
            patch("pathlib.Path.expanduser", return_value=grammar_dir),
            patch.object(p, "_download_and_compile_grammar") as mock_download,
        ):
            await p._ensure_grammar_available()
        mock_download.assert_awaited_once()


class TestDownloadAndCompileGrammar:
    @pytest.fixture
    def processor(self):
        return ConcreteTreeSitterProcessor(_make_config())

    @pytest.mark.asyncio
    async def test_unknown_grammar_name(self, processor, tmp_path):
        """Grammar not in GRAMMAR_REPOSITORIES just logs warning and returns."""
        await processor._download_and_compile_grammar(tmp_path, "unknown_grammar")
        # Should not raise, just return silently

    @pytest.mark.asyncio
    async def test_clone_failure(self, processor, tmp_path):
        """If git clone fails, should not raise (OSError path)."""
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "git")):
            await processor._download_and_compile_grammar(tmp_path, "python")
        # Should not raise

    @pytest.mark.asyncio
    async def test_os_error(self, processor, tmp_path):
        """OSError during download should be caught."""
        with patch("subprocess.run", side_effect=OSError("disk full")):
            await processor._download_and_compile_grammar(tmp_path, "python")


class TestInitializeImplAdvanced:
    @pytest.mark.asyncio
    async def test_initialize_import_error(self):
        """When the language module cannot be imported, RuntimeError is raised."""
        config = _make_config(name="fake_lang", grammar="nonexistent_grammar")
        p = ConcreteTreeSitterProcessor(config)
        if TREE_SITTER_AVAILABLE:
            with pytest.raises(RuntimeError, match="Failed to initialize"):
                await p._initialize_impl()

    @pytest.mark.asyncio
    async def test_initialize_generic_error(self):
        """When initialization raises a generic error."""
        config = _make_config()
        p = ConcreteTreeSitterProcessor(config)
        if TREE_SITTER_AVAILABLE:
            with patch("importlib.import_module", side_effect=RuntimeError("broken")):
                with pytest.raises(RuntimeError, match="Failed to initialize"):
                    await p._initialize_impl()


class TestTreeSitterImportFallback:
    """Test tree-sitter import fallback (lines 39-44)."""

    def test_tree_sitter_available_is_bool(self):
        """TREE_SITTER_AVAILABLE should be a boolean."""
        assert isinstance(TREE_SITTER_AVAILABLE, bool)

    def test_tree_sitter_is_available(self):
        """In this environment, tree-sitter should be available."""
        assert TREE_SITTER_AVAILABLE is True


class TestDownloadAndCompileGrammarAdvanced:
    """Additional tests for _download_and_compile_grammar (lines 153-207)."""

    @pytest.fixture
    def processor(self):
        return ConcreteTreeSitterProcessor(_make_config())

    @pytest.mark.asyncio
    async def test_typescript_grammar_uses_subdirectory(self, processor, tmp_path):
        """TypeScript grammar uses typescript subdirectory if it exists (lines 156-161)."""
        import subprocess

        # Mock successful git clone that creates a typescript subdirectory
        def mock_run(cmd, **kwargs):
            if "git" in cmd[0]:
                grammar_dir = tmp_path / "typescript"
                grammar_dir.mkdir(exist_ok=True)
                ts_sub = grammar_dir / "typescript"
                ts_sub.mkdir(exist_ok=True)
                return MagicMock(returncode=0, stdout="", stderr="")
            # Fail all subsequent compilation attempts
            raise subprocess.CalledProcessError(1, cmd)

        with patch("subprocess.run", side_effect=mock_run):
            await processor._download_and_compile_grammar(tmp_path, "typescript")
        # Should not raise, just log warning

    @pytest.mark.asyncio
    async def test_wasm_compilation_success(self, processor, tmp_path):
        """Successful tree-sitter build-wasm compilation (lines 164-178)."""
        import subprocess

        clone_called = False

        def mock_run(cmd, **kwargs):
            nonlocal clone_called
            if "git" in cmd[0]:
                clone_called = True
                grammar_dir = tmp_path / "python"
                grammar_dir.mkdir(exist_ok=True)
                # Create the compiled file
                wasm = grammar_dir / "tree-sitter-python.wasm"
                wasm.write_text("fake wasm")
                return MagicMock(returncode=0, stdout="", stderr="")
            if "tree-sitter" in cmd[0]:
                return MagicMock(returncode=0, stdout="", stderr="")
            raise subprocess.CalledProcessError(1, cmd)

        with patch("subprocess.run", side_effect=mock_run):
            await processor._download_and_compile_grammar(tmp_path, "python")
        assert clone_called

    @pytest.mark.asyncio
    async def test_gcc_fallback_compilation(self, processor, tmp_path):
        """GCC fallback compilation when tree-sitter CLI fails (lines 184-202)."""
        import subprocess

        call_count = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if "git" in cmd[0]:
                grammar_dir = tmp_path / "python"
                grammar_dir.mkdir(exist_ok=True)
                src_dir = grammar_dir / "src"
                src_dir.mkdir(exist_ok=True)
                (src_dir / "parser.c").write_text("// fake")
                return MagicMock(returncode=0, stdout="", stderr="")
            if "tree-sitter" in cmd[0]:
                raise subprocess.CalledProcessError(1, cmd)
            if "gcc" in cmd[0]:
                return MagicMock(returncode=0, stdout="", stderr="")
            raise subprocess.CalledProcessError(1, cmd)

        with patch("subprocess.run", side_effect=mock_run):
            await processor._download_and_compile_grammar(tmp_path, "python")
        # At minimum git + tree-sitter attempt = 2 calls, gcc may or may not run
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_gcc_fallback_fails(self, processor, tmp_path):
        """Both tree-sitter CLI and GCC fail (lines 204-207)."""
        import subprocess

        def mock_run(cmd, **kwargs):
            if "git" in cmd[0]:
                grammar_dir = tmp_path / "python"
                grammar_dir.mkdir(exist_ok=True)
                src_dir = grammar_dir / "src"
                src_dir.mkdir(exist_ok=True)
                (src_dir / "parser.c").write_text("// fake")
                return MagicMock(returncode=0, stdout="", stderr="")
            raise subprocess.CalledProcessError(1, cmd)

        with patch("subprocess.run", side_effect=mock_run):
            # Should not raise
            await processor._download_and_compile_grammar(tmp_path, "python")
