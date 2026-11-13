"""Coverage boost for tree-sitter extraction methods on Java/Rust/C# processors.

Since tree-sitter IS available, the tree-sitter code paths (extract_entities,
extract_relationships, _extract_*, _extract_modifiers, etc.) are the active
branches. We test them with mock tree-sitter nodes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models.core_models import CoreCodeEntity, EntityType

# ── Helper to create mock nodes ──────────────────────────────────────


def _mock_node(
    node_type: str,
    text: str = "",
    start_point: tuple = (0, 0),
    end_point: tuple = (0, 10),
    children: list | None = None,
    start_byte: int = 0,
    end_byte: int = 0,
):
    node = MagicMock()
    node.type = node_type
    node.start_point = start_point
    node.end_point = end_point
    node.children = children or []
    node.start_byte = start_byte
    node.end_byte = end_byte
    node.parent = None
    return node


def _mock_tree(root_children: list, content: bytes = b""):
    root = _mock_node("program", children=root_children)
    tree = MagicMock()
    tree.root_node = root
    tree.text = content
    tree.root_node.text = content
    return tree


# ═══════════════════════════════════════════════════════════════════════
# Java tree-sitter methods
# ═══════════════════════════════════════════════════════════════════════


class TestJavaTreeSitterExtraction:
    @pytest.fixture
    def processor(self):
        from processing.languages.java.java_processor import JavaProcessor

        return JavaProcessor()

    @pytest.mark.asyncio
    async def test_extract_entities_with_empty_tree(self, processor):
        tree = _mock_tree([], b"")
        entities = await processor.extract_entities(tree, Path("Test.java"))
        assert isinstance(entities, list)

    @pytest.mark.asyncio
    async def test_extract_entities_returns_none_for_none_ast(self, processor):
        entities = await processor.extract_entities(None, Path("Test.java"))
        assert entities == []

    @pytest.mark.asyncio
    async def test_extract_relationships_none_ast(self, processor):
        rels = await processor.extract_relationships(None, [], Path("Test.java"))
        assert rels == []

    @pytest.mark.asyncio
    async def test_extract_relationships_empty_tree(self, processor):
        tree = _mock_tree([], b"")
        rels = await processor.extract_relationships(tree, [], Path("Test.java"))
        assert isinstance(rels, list)

    def test_extract_modifiers_public(self, processor):
        content = b"public static final void main() {}"
        node = _mock_node("method_declaration", start_byte=0, end_byte=len(content))
        modifiers = processor._extract_modifiers(node, content)
        assert modifiers["visibility"] == "public"
        assert modifiers["static"] is True
        assert modifiers["final"] is True

    def test_extract_modifiers_private(self, processor):
        content = b"private volatile int count;"
        node = _mock_node("field_declaration", start_byte=0, end_byte=len(content))
        modifiers = processor._extract_modifiers(node, content)
        assert modifiers["visibility"] == "private"
        assert modifiers["volatile"] is True

    def test_extract_modifiers_protected(self, processor):
        content = b"protected synchronized void sync() {}"
        node = _mock_node("method_declaration", start_byte=0, end_byte=len(content))
        modifiers = processor._extract_modifiers(node, content)
        assert modifiers["visibility"] == "protected"
        assert modifiers["synchronized"] is True

    def test_extract_modifiers_package_default(self, processor):
        content = b"void helper() {}"
        node = _mock_node("method_declaration", start_byte=0, end_byte=len(content))
        modifiers = processor._extract_modifiers(node, content)
        assert modifiers["visibility"] == "package"

    def test_extract_modifiers_abstract(self, processor):
        content = b"public abstract class Base {}"
        node = _mock_node("class_declaration", start_byte=0, end_byte=len(content))
        modifiers = processor._extract_modifiers(node, content)
        assert modifiers["abstract"] is True

    def test_extract_modifiers_native_transient(self, processor):
        content = b"private transient native int data;"
        node = _mock_node("field_declaration", start_byte=0, end_byte=len(content))
        modifiers = processor._extract_modifiers(node, content)
        assert modifiers["transient"] is True
        assert modifiers["native"] is True

    def test_extract_package_no_package_nodes(self, processor):
        tree = _mock_tree([], b"public class Foo {}")
        result = processor._extract_package(tree, Path("Foo.java"), b"public class Foo {}")
        assert result is None

    def test_extract_imports_empty_tree(self, processor):
        tree = _mock_tree([], b"")
        imports = processor._extract_imports(tree, Path("T.java"), b"")
        assert imports == []

    def test_extract_classes_empty_tree(self, processor):
        tree = _mock_tree([], b"")
        classes = processor._extract_classes(tree, Path("T.java"), b"")
        assert classes == []

    def test_extract_methods_empty_tree(self, processor):
        tree = _mock_tree([], b"")
        methods = processor._extract_methods(tree, Path("T.java"), b"")
        assert methods == []

    def test_extract_fields_empty_tree(self, processor):
        tree = _mock_tree([], b"")
        fields = processor._extract_fields(tree, Path("T.java"), b"")
        assert fields == []

    def test_extract_import_relationships_no_file_entity(self, processor):
        tree = _mock_tree([], b"")
        import_entity = CoreCodeEntity(
            id="import:java.util.List:T.java",
            name="java.util.List",
            type=EntityType.IMPORT,
            language="java",
            file_path="T.java",
            line_number=1,
            column_start=0,
            column_end=20,
        )
        rels = processor._extract_import_relationships(tree, [import_entity], Path("T.java"))
        assert len(rels) == 1

    def test_extract_import_relationships_with_file_entity(self, processor):
        tree = _mock_tree([], b"")
        file_entity = CoreCodeEntity(
            id="file:T.java",
            name="T.java",
            type=EntityType.FILE,
            language="java",
            file_path="T.java",
            line_number=1,
        )
        import_entity = CoreCodeEntity(
            id="import:java.util.List:T.java",
            name="java.util.List",
            type=EntityType.IMPORT,
            language="java",
            file_path="T.java",
            line_number=2,
            column_start=0,
            column_end=20,
        )
        rels = processor._extract_import_relationships(
            tree, [file_entity, import_entity], Path("T.java")
        )
        assert len(rels) == 1
        assert rels[0].source_id == "file:T.java"

    def test_extract_inheritance_empty(self, processor):
        tree = _mock_tree([], b"")
        rels = processor._extract_inheritance_relationships(tree, [], Path("T.java"))
        assert rels == []

    def test_extract_call_relationships_empty(self, processor):
        tree = _mock_tree([], b"")
        rels = processor._extract_call_relationships(tree, [], Path("T.java"))
        assert rels == []


# ═══════════════════════════════════════════════════════════════════════
# Rust tree-sitter methods
# ═══════════════════════════════════════════════════════════════════════


class TestRustTreeSitterExtraction:
    @pytest.fixture
    def processor(self):
        from processing.languages.rust.rust_processor import RustProcessor

        return RustProcessor()

    @pytest.mark.asyncio
    async def test_extract_entities_empty_tree(self, processor):
        tree = _mock_tree([], b"")
        entities = await processor.extract_entities(tree, Path("main.rs"))
        assert isinstance(entities, list)

    @pytest.mark.asyncio
    async def test_extract_relationships_empty_tree(self, processor):
        tree = _mock_tree([], b"")
        rels = await processor.extract_relationships(tree, [], Path("main.rs"))
        assert isinstance(rels, list)

    def test_extract_visibility_pub(self, processor):
        content = b"pub fn main() {}"
        node = _mock_node("function_item", start_byte=0, end_byte=len(content))
        vis = processor._extract_visibility(node, content)
        assert vis == "public"

    def test_extract_visibility_private(self, processor):
        content = b"fn helper() {}"
        node = _mock_node("function_item", start_byte=0, end_byte=len(content))
        vis = processor._extract_visibility(node, content)
        assert vis == "private"

    def test_extract_modules_empty_tree(self, processor):
        tree = _mock_tree([], b"")
        mods = processor._extract_modules(tree, Path("main.rs"), b"")
        assert mods == []

    def test_extract_use_declarations_empty(self, processor):
        tree = _mock_tree([], b"")
        uses = processor._extract_use_declarations(tree, Path("main.rs"), b"")
        assert uses == []

    def test_extract_structs_empty(self, processor):
        tree = _mock_tree([], b"")
        structs = processor._extract_structs(tree, Path("main.rs"), b"")
        assert structs == []

    def test_extract_enums_empty(self, processor):
        tree = _mock_tree([], b"")
        enums = processor._extract_enums(tree, Path("main.rs"), b"")
        assert enums == []

    def test_extract_traits_empty(self, processor):
        tree = _mock_tree([], b"")
        traits = processor._extract_traits(tree, Path("main.rs"), b"")
        assert traits == []

    def test_extract_functions_empty(self, processor):
        tree = _mock_tree([], b"")
        fns = processor._extract_functions(tree, Path("main.rs"), b"")
        assert fns == []

    def test_extract_impl_blocks_empty(self, processor):
        tree = _mock_tree([], b"")
        impls = processor._extract_impl_blocks(tree, Path("main.rs"), b"")
        assert impls == []

    def test_extract_constants_empty(self, processor):
        tree = _mock_tree([], b"")
        consts = processor._extract_constants(tree, Path("main.rs"), b"")
        assert consts == []

    def test_extract_trait_implementations_empty(self, processor):
        tree = _mock_tree([], b"")
        rels = processor._extract_trait_implementations(tree, [], Path("main.rs"))
        assert rels == []

    def test_extract_function_calls_empty(self, processor):
        tree = _mock_tree([], b"")
        rels = processor._extract_function_calls(tree, [], Path("main.rs"))
        assert rels == []

    def test_extract_use_relationships_no_file_entity(self, processor):
        tree = _mock_tree([], b"")
        import_entity = CoreCodeEntity(
            id="use:std::io:main.rs",
            name="std::io",
            type=EntityType.IMPORT,
            language="rust",
            file_path="main.rs",
            line_number=1,
            column_start=0,
            column_end=10,
        )
        rels = processor._extract_use_relationships(tree, [import_entity], Path("main.rs"))
        assert len(rels) == 1

    def test_extract_use_relationships_with_file_entity(self, processor):
        tree = _mock_tree([], b"")
        file_entity = CoreCodeEntity(
            id="file:main.rs",
            name="main.rs",
            type=EntityType.FILE,
            language="rust",
            file_path="main.rs",
            line_number=1,
        )
        import_entity = CoreCodeEntity(
            id="use:std::io:main.rs",
            name="std::io",
            type=EntityType.IMPORT,
            language="rust",
            file_path="main.rs",
            line_number=2,
            column_start=0,
            column_end=10,
        )
        rels = processor._extract_use_relationships(
            tree, [file_entity, import_entity], Path("main.rs")
        )
        assert len(rels) == 1
        assert rels[0].source_id == "file:main.rs"

    def test_extract_impl_blocks_inherent(self, processor):
        """Test inherent impl block extraction with mock nodes."""
        content = b"impl MyStruct {\n    fn new() {}\n}"
        impl_node = _mock_node(
            "impl_item",
            start_byte=0,
            end_byte=len(content),
            start_point=(0, 0),
            end_point=(2, 1),
            children=[],
        )
        tree = _mock_tree([impl_node], content)
        impls = processor._extract_impl_blocks(tree, Path("main.rs"), content)
        assert len(impls) >= 1
        assert impls[0].properties["impl_type"] == "inherent"

    def test_extract_impl_blocks_trait(self, processor):
        """Test trait impl block extraction."""
        content = b"impl Display for MyStruct {\n    fn fmt() {}\n}"
        impl_node = _mock_node(
            "impl_item",
            start_byte=0,
            end_byte=len(content),
            start_point=(0, 0),
            end_point=(2, 1),
            children=[],
        )
        tree = _mock_tree([impl_node], content)
        impls = processor._extract_impl_blocks(tree, Path("main.rs"), content)
        assert len(impls) >= 1
        assert impls[0].properties["impl_type"] == "trait"


# ═══════════════════════════════════════════════════════════════════════
# HTTP Quality Runner _execute_request_with_retry
# ═══════════════════════════════════════════════════════════════════════


class TestHttpQualityExecuteRequestRetry:
    @pytest.fixture
    def runner(self):
        from processing.quality_analysis.http_quality_runner import HTTPQualityRunner

        return HTTPQualityRunner()

    @pytest.mark.asyncio
    async def test_execute_request_200_success(self, runner):
        from datetime import UTC, datetime
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "success": True,
                "analysis_id": "test-123",
                "language": "python",
                "tools_results": {"ruff": {"score": 9.0}},
                "overall_score": 9.0,
                "total_issues": 0,
                "issues": [],
                "execution_time": 1.5,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        result = await runner._execute_request_with_retry(
            mock_session,
            "analyze",
            {"language": "python", "code_content": "x=1"},
            analysis_id="test-123",
            start_time=datetime.now(UTC),
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_request_400_no_retry(self, runner):
        from datetime import UTC, datetime
        from unittest.mock import AsyncMock

        mock_response = MagicMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Bad Request")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        result = await runner._execute_request_with_retry(
            mock_session,
            "analyze",
            {"language": "python"},
            analysis_id="test-400",
            start_time=datetime.now(UTC),
        )
        # Should break on 4xx without retrying
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_request_500_retries(self, runner):
        from datetime import UTC, datetime
        from unittest.mock import AsyncMock

        runner.config.max_retries = 2

        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)

        result = await runner._execute_request_with_retry(
            mock_session,
            "analyze",
            {"language": "python"},
            analysis_id="test-500",
            start_time=datetime.now(UTC),
        )
        assert result.success is False
        assert "500" in result.error


# ═══════════════════════════════════════════════════════════════════════
# test_runner additional paths
# ═══════════════════════════════════════════════════════════════════════


class TestContainerTestRunnerLegacy:
    @pytest.fixture
    def mock_docker(self):
        svc = MagicMock()
        svc.get_language_container_image = MagicMock(return_value="python:3.12")
        svc.create_test_container = MagicMock(return_value="cid")
        svc.start_container = MagicMock()
        svc.stop_container = MagicMock()
        svc.remove_container_by_id = MagicMock()
        svc.create_file_in_container = MagicMock(return_value=True)
        svc.install_tools_in_container = MagicMock()
        svc.exec_in_container = MagicMock(
            return_value={"exit_code": 0, "stdout": "OK", "stderr": ""}
        )
        return svc

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.max_execution_time = 30
        config.max_memory_mb = 512
        config.max_file_size_mb = 10
        config.max_code_length = 10000
        config.allowed_imports = ["pytest"]
        return config

    @pytest.fixture
    def runner(self, mock_docker, mock_config):
        from processing.test_execution.test_runner import ContainerTestRunner

        return ContainerTestRunner(mock_docker, mock_config)

    def test_get_memory_limit_default(self, runner):
        """Unknown security level returns default."""
        result = runner._get_memory_limit("unknown")
        assert result == "512m"

    def test_get_cpu_quota_default(self, runner):
        """Unknown security level returns default."""
        result = runner._get_cpu_quota("unknown")
        assert result == 50000
