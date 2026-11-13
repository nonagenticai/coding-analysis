"""
JavaScript/TypeScript language processor using Tree-sitter with compatibility layer.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    CoreComplexityMetrics,
    EntityRelation,
    EntityType,
    RelationType,
)

from ..base import LanguageProcessor, ParseResult, ParserType
from ..registry import register_language
from .javascript_config import JAVASCRIPT_CONFIG

# Import Tree-sitter components if available
try:
    from tree_sitter import Tree

    from processing.parsers.tree_sitter_base import BaseTreeSitterProcessor

    # Create JavaScriptTreeSitterProcessor that properly inherits from BaseTreeSitterProcessor
    class JavaScriptTreeSitterProcessor(BaseTreeSitterProcessor):
        """JavaScript Tree-sitter processor that inherits from BaseTreeSitterProcessor."""

        pass

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

    # Create a dummy JavaScriptTreeSitterProcessor that inherits from LanguageProcessor
    class JavaScriptTreeSitterProcessor(LanguageProcessor):
        """Dummy JavaScriptTreeSitterProcessor for when tree-sitter is not available."""

        async def _initialize_impl(self) -> None:
            pass

        async def parse_file(self, file_path: Path, content: str) -> ParseResult:
            """Async Function parse_file.

            Args:
                file_path (Path): The file_path parameter.
                content (str): The content parameter.

            Returns:
                ParseResult: The parseresult result.

            Raises:
                NotImplementedError: Raised under certain conditions."""
            """Async Method parse_file.

            Args:
                file_path (Path): The file_path parameter.
                content (str): The content parameter.

            Returns:
                ParseResult: The parseresult result.

            Raises:
                NotImplementedError: Raised under certain conditions."""
            # This should be overridden by subclasses
            raise NotImplementedError("Tree-sitter not available")

        async def extract_entities(self, ast: Any, file_path: Path) -> list[CoreCodeEntity]:
            """Async Function extract_entities.

            Args:
                ast (Any): The ast parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[CoreCodeEntity]: The list[corecodeentity] result."""
            """Async Method extract_entities.

            Args:
                ast (Any): The ast parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[CoreCodeEntity]: The list[corecodeentity] result."""
            return []

        async def extract_relationships(
            self, ast: Any, entities: list[CoreCodeEntity], file_path: Path
        ) -> list[EntityRelation]:
            """Async Function extract_relationships.

            Args:
                ast (Any): The ast parameter.
                entities (List[CoreCodeEntity]): The entities parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[EntityRelation]: The list[entityrelation] result.

            Example:
                >>> await extract_relationships("ast_value", "entities_value", "file_path_value")
            """
            """Async Method extract_relationships.

            Args:
                ast (Any): The ast parameter.
                entities (List[CoreCodeEntity]): The entities parameter.
                file_path (Path): The file_path parameter.

            Returns:
                List[EntityRelation]: The list[entityrelation] result.

            Example:
                >>> await extract_relationships("ast_value", "entities_value", "file_path_value")"""
            return []

    # Tree/Node types not available - use Any in method signatures

# Import existing JavaScript processing components
try:
    from processing.javascript.javascript_parser import JavaScriptParser

    LEGACY_COMPONENTS_AVAILABLE = True
except ImportError:
    LEGACY_COMPONENTS_AVAILABLE = False

    # Create a dummy JavaScriptParser class that can be instantiated
    class JavaScriptParser:
        """Dummy JavaScript parser when legacy components are not available."""

        def __init__(self):
            pass


logger = logging.getLogger(__name__)


@register_language(JAVASCRIPT_CONFIG)
class JSLanguageProcessor(JavaScriptTreeSitterProcessor):  # type: ignore[unsupported-base]
    """JavaScript language processor using Tree-sitter with legacy compatibility."""

    def __init__(self, config=None):
        super().__init__(config or JAVASCRIPT_CONFIG)

        # Try to use legacy components if available
        if LEGACY_COMPONENTS_AVAILABLE:
            try:
                self.legacy_parser = JavaScriptParser()
                self._use_legacy = True
            except (ImportError, AttributeError, TypeError) as e:
                logger.warning("Failed to initialize legacy JavaScript components: %s", e)
                self._use_legacy = False
        else:
            self._use_legacy = False

    async def _initialize_impl(self) -> None:
        """Initialize JavaScript processor."""
        if TREE_SITTER_AVAILABLE:
            await super()._initialize_impl()
        else:
            # Fallback initialization without Tree-sitter
            pass

    async def parse_file(self, file_path: Path, content: str) -> ParseResult:
        """Parse JavaScript file.

        Args:
            file_path (Path): The file_path parameter.
            content (str): The content parameter.

        Returns:
            ParseResult: The parseresult result.

        Example:
            >>> await parse_file("file_path_value", "content_value")"""
        file_path = Path(file_path)

        # Determine if this is TypeScript
        is_typescript = file_path.suffix.lower() in [".ts", ".tsx"]
        language = "typescript" if is_typescript else "javascript"

        # Tree-sitter is the primary parser (no external Node dependency).
        if TREE_SITTER_AVAILABLE and self.parser:
            try:
                result: ParseResult = await super().parse_file(file_path, content)
                if result.success and result.entities:
                    return result
                logger.debug("Tree-sitter produced no entities for %s, trying fallbacks", file_path)
            except (ImportError, AttributeError, TypeError, ValueError) as e:
                logger.warning("Tree-sitter parsing failed, falling back: %s", e)

        # Legacy parser is only a fallback when Tree-sitter is unavailable.
        if self._use_legacy and not is_typescript and hasattr(self, "legacy_parser"):
            try:
                return await self._parse_with_legacy(file_path, content, language)
            except (ImportError, AttributeError, TypeError, ValueError) as e:
                logger.warning("Legacy parser failed, falling back to regex: %s", e)

        # Fall back to regex parsing
        return await self._regex_parse(file_path, content, language)

    async def _parse_with_legacy(self, file_path: Path, content: str, language: str) -> ParseResult:
        """Parse using legacy JavaScript components."""
        entities = []
        relationships: list[EntityRelation] = []
        errors = []

        try:
            # Parse the code
            parse_result = await self.legacy_parser.parse_code(content, str(file_path))

            if parse_result and parse_result.get("success", False):
                # Convert legacy entities to new format
                legacy_entities = parse_result.get("entities", [])
                for le in legacy_entities:
                    entity = CoreCodeEntity(
                        id=getattr(le, "id", f"{le.name}:{file_path}"),
                        name=getattr(le, "name", ""),
                        type=self._map_entity_type(getattr(le, "type", "function")),
                        language=language,
                        file_path=str(file_path),
                        line_number=getattr(le, "line_number", 1),
                        end_line_number=None,
                        column_start=None,
                        column_end=None,
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties=getattr(le, "metadata", {}),
                    )
                    entities.append(entity)
            else:
                error_msg = "Unknown parse error"
                if parse_result:
                    error_msg = parse_result.get("error", "Unknown parse error")
                errors.append(error_msg)

        except (ImportError, AttributeError, TypeError, ValueError) as e:
            logger.error("Legacy parsing failed: %s", e)
            raise

        # Calculate complexity using common method
        complexity = self.calculate_basic_complexity_metrics(content, entities)

        return ParseResult(
            ast=None,
            language=language,
            file_path=str(file_path),
            parser_used=ParserType.EXTERNAL_TOOL,
            success=len(errors) == 0,
            entities=entities,
            relationships=relationships,
            lines_of_code=len(content.splitlines()),
            complexity_score=complexity.cyclomatic_complexity,
            errors=errors,
            parse_time_ms=0.0,
        )

    async def _regex_parse(self, file_path: Path, content: str, language: str) -> ParseResult:
        """Fallback regex-based parsing for JavaScript/TypeScript."""
        entities = []
        relationships: list[EntityRelation] = []

        # Extract import/require statements
        import_patterns = [
            r'import\s+(?:{[^}]+}|\w+|\*\s+as\s+\w+)\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+[\'"]([^\'"]+)[\'"]',
            r'const\s+\w+\s*=\s*require\([\'"]([^\'"]+)[\'"]\)',
            r'require\([\'"]([^\'"]+)[\'"]\)',
        ]

        for pattern in import_patterns:
            for match in re.finditer(pattern, content):
                module_name = match.group(1)
                line_num = content[: match.start()].count("\n") + 1

                entities.append(
                    CoreCodeEntity(
                        id=f"import:{module_name}:{file_path}",
                        name=module_name,
                        type=EntityType.IMPORT,
                        language=language,
                        file_path=str(file_path),
                        line_number=line_num,
                        end_line_number=None,
                        column_start=None,
                        column_end=None,
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"import_path": module_name},
                    )
                )

        # Calculate basic complexity for regex parsing
        complexity = CoreComplexityMetrics(
            lines_of_code=len(content.splitlines()),
            logical_lines=len(
                [
                    line
                    for line in content.splitlines()
                    if line.strip() and not line.strip().startswith("//")
                ]
            ),
            function_count=len([e for e in entities if e.type == EntityType.FUNCTION]),
            class_count=len([e for e in entities if e.type == EntityType.CLASS]),
            comment_lines=len(
                [
                    line
                    for line in content.splitlines()
                    if line.strip().startswith("//") or "/*" in line
                ]
            ),
            blank_lines=len([line for line in content.splitlines() if not line.strip()]),
            docstring_coverage=0.0,
        )

        return ParseResult(
            ast=None,
            language=language,
            file_path=str(file_path),
            parser_used=ParserType.REGEX_FALLBACK,
            success=True,
            entities=entities,
            relationships=relationships,
            lines_of_code=len(content.splitlines()),
            complexity_score=complexity.cyclomatic_complexity,
            errors=[],
            parse_time_ms=0.0,
        )

    def _map_entity_type(self, legacy_type: str) -> EntityType:
        """Map legacy entity type to new EntityType enum."""
        mapping = {
            "class": EntityType.CLASS,
            "function": EntityType.FUNCTION,
            "method": EntityType.FUNCTION,
            "variable": EntityType.VARIABLE,
            "constant": EntityType.CONSTANT,
            "import": EntityType.IMPORT,
            "interface": EntityType.INTERFACE,
            "type": EntityType.TYPE_ALIAS,
            "enum": EntityType.ENUM,
        }
        return mapping.get(legacy_type.lower(), EntityType.FUNCTION)

    # Tree-sitter node types for JavaScript entity extraction.
    _DECL_NODE_TYPES = {
        "class_declaration": EntityType.CLASS,
        "function_declaration": EntityType.FUNCTION,
        "generator_function_declaration": EntityType.FUNCTION,
        "method_definition": EntityType.METHOD,
    }

    @staticmethod
    def _node_name(node: Any) -> str | None:
        """Return the text of a node's ``name`` field, if present."""
        name_node = node.child_by_field_name("name")
        if name_node is None or name_node.text is None:
            return None
        return name_node.text.decode("utf-8", errors="ignore")

    # Required abstract methods from LanguageProcessor
    async def extract_entities(self, ast: Any, file_path: Path) -> list[CoreCodeEntity]:
        """Extract code entities from a Tree-sitter AST.

        Walks the JavaScript syntax tree for classes, functions, methods and
        imports. Falls back to regex extraction when Tree-sitter is unavailable.

        Args:
            ast: The parsed Tree-sitter ``Tree`` (or ``None``/other when
                Tree-sitter is unavailable).
            file_path: Path of the source file being analysed.

        Returns:
            List of extracted code entities.
        """
        if not (TREE_SITTER_AVAILABLE and isinstance(ast, Tree)):
            # Fallback to regex parsing
            try:
                content_str = file_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, FileNotFoundError):
                return []
            is_typescript = file_path.suffix.lower() in [".ts", ".tsx"]
            language = "typescript" if is_typescript else "javascript"
            result = await self._regex_parse(file_path, content_str, language)
            return result.entities

        language = "typescript" if file_path.suffix.lower() in [".ts", ".tsx"] else "javascript"
        entities: list[CoreCodeEntity] = [
            CoreCodeEntity(
                id=f"file:{file_path}",
                name=Path(file_path).name,
                type=EntityType.FILE,
                language=language,
                file_path=str(file_path),
                line_number=1,
                properties={"extension": Path(file_path).suffix},
            )
        ]

        for node in self._find_nodes_by_type(ast, set(self._DECL_NODE_TYPES)):
            name = self._node_name(node)
            if not name:
                continue
            entity_type = self._DECL_NODE_TYPES[node.type]
            line = node.start_point[0] + 1
            entities.append(
                CoreCodeEntity(
                    id=f"{entity_type.value}:{name}:{file_path}:{line}",
                    name=name,
                    type=entity_type,
                    language=language,
                    file_path=str(file_path),
                    line_number=line,
                    end_line_number=node.end_point[0] + 1,
                )
            )

        for node in self._find_nodes_by_type(ast, {"import_statement"}):
            source = node.child_by_field_name("source")
            if source is None or source.text is None:
                continue
            module_name = source.text.decode("utf-8", errors="ignore").strip("'\"`")
            line = node.start_point[0] + 1
            entities.append(
                CoreCodeEntity(
                    id=f"import:{module_name}:{file_path}:{line}",
                    name=module_name,
                    type=EntityType.IMPORT,
                    language=language,
                    file_path=str(file_path),
                    line_number=line,
                    properties={"import_path": module_name},
                )
            )

        return entities

    async def extract_relationships(
        self, ast: Any, entities: list[CoreCodeEntity], file_path: Path
    ) -> list[EntityRelation]:
        """Extract relationships between JavaScript entities.

        Produces ``CONTAINS`` edges (file → declarations/imports), ``IMPORTS``
        edges (file → imported module) and ``INHERITS`` edges (class → base
        class) from the Tree-sitter AST.

        Args:
            ast: The parsed Tree-sitter ``Tree`` (or other when unavailable).
            entities: Entities previously extracted from ``ast``.
            file_path: Path of the source file being analysed.

        Returns:
            List of relationships between the supplied entities.
        """
        relationships: list[EntityRelation] = []

        file_entity = next((e for e in entities if e.type == EntityType.FILE), None)
        if file_entity is not None:
            for entity in entities:
                if entity.id == file_entity.id:
                    continue
                rel_type = (
                    RelationType.IMPORTS
                    if entity.type == EntityType.IMPORT
                    else RelationType.CONTAINS
                )
                relationships.append(
                    EntityRelation(
                        source_id=file_entity.id,
                        target_id=entity.id,
                        type=rel_type,
                    )
                )

        if not (TREE_SITTER_AVAILABLE and isinstance(ast, Tree)):
            return relationships

        class_id_by_name = {e.name: e.id for e in entities if e.type == EntityType.CLASS}
        for node in self._find_nodes_by_type(ast, {"class_declaration"}):
            name = self._node_name(node)
            if not name or name not in class_id_by_name:
                continue
            heritage = self._get_child_by_type(node, "class_heritage")
            if heritage is None:
                continue
            base_node = self._get_child_by_type(heritage, "identifier")
            if base_node is None or base_node.text is None:
                continue
            base_name = base_node.text.decode("utf-8", errors="ignore")
            relationships.append(
                EntityRelation(
                    source_id=class_id_by_name[name],
                    target_id=class_id_by_name.get(base_name, f"class:{base_name}:external"),
                    type=RelationType.INHERITS,
                )
            )

        return relationships


# TypeScript processor is now handled by the dedicated TypeScript processor module
# @register_language(TYPESCRIPT_CONFIG) - REMOVED: Duplicate registration
# class TypeScriptProcessor(JavaScriptProcessor):
#     """TypeScript language processor, extends JavaScript processor."""
#
#     def __init__(self, config=None):
#         super().__init__(config or TYPESCRIPT_CONFIG)
