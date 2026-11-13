"""
Python language processor using native AST with compatibility layer.
"""

from __future__ import annotations

import ast as ast_module
import logging
from pathlib import Path
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    CoreComplexityMetrics,
    EntityRelation,
    EntityType,
    RelationType,
)

from ..base import (
    LanguageProcessor,
    ParseResult,
    ParserType,
)
from ..registry import register_language
from .python_config import PYTHON_CONFIG

# Import existing Python processing components
try:
    from processing.python.python_entity_extractor import PythonEntityExtractor
    from processing.python.python_parser import PythonParser
    from processing.python.python_relation_extractor import (
        PythonRelationExtractor,
    )

    LEGACY_COMPONENTS_AVAILABLE = True
except ImportError:
    LEGACY_COMPONENTS_AVAILABLE = False
    PythonParser = None  # type: ignore
    PythonEntityExtractor = None  # type: ignore
    PythonRelationExtractor = None  # type: ignore

logger = logging.getLogger(__name__)


@register_language(PYTHON_CONFIG)
class PyLanguageProcessor(LanguageProcessor):
    """Python language processor using native AST."""

    def __init__(self, config=None):
        super().__init__(config or PYTHON_CONFIG)

        # Try to use legacy components if available
        if LEGACY_COMPONENTS_AVAILABLE:
            try:
                self.parser = PythonParser()
                self.entity_extractor = PythonEntityExtractor()
                self.relation_extractor = PythonRelationExtractor()
                self._use_legacy = True
            except (ImportError, AttributeError, TypeError) as e:
                logger.warning("Failed to initialize legacy Python components: %s", str(e))
                self._use_legacy = False
        else:
            self._use_legacy = False

    async def _initialize_impl(self) -> None:
        """Initialize Python processor."""
        # Python's AST is built-in, no special initialization needed

    async def parse_file(self, file_path: Path, content: str) -> ParseResult:
        """Parse Python file using native AST or legacy parser."""
        file_path = Path(file_path)

        # Try legacy parser first if available
        if self._use_legacy and hasattr(self, "parser"):
            try:
                return await self._parse_with_legacy(file_path, content)
            except (SyntaxError, ValueError, TypeError, AttributeError) as e:
                logger.warning("Legacy parser failed, falling back to AST: %s", str(e))

        # Use native AST parsing
        return await self._parse_with_ast(file_path, content)

    async def _parse_with_legacy(self, file_path: Path, content: str) -> ParseResult:
        """Parse using legacy Python components."""
        entities: list[CoreCodeEntity] = []
        relationships: list[EntityRelation] = []
        errors: list[str] = []

        try:
            tree = await self._try_legacy_parsing(file_path, content, errors)
            if tree:
                entities, relationships = await self._extract_with_legacy_components(
                    tree, file_path, content
                )

        except (SyntaxError, ValueError, TypeError, AttributeError) as e:
            logger.error("Legacy parsing failed: %s", str(e))
            errors.append(str(e))

        # Calculate complexity
        complexity = self._calculate_complexity(content, entities)

        return ParseResult(
            language="python",
            file_path=str(file_path),
            parser_used=ParserType.AST_NATIVE,
            success=len(errors) == 0,
            ast=tree,
            entities=entities,
            relationships=relationships,
            lines_of_code=len(content.splitlines()),
            complexity_score=complexity.cyclomatic_complexity,
            errors=errors,
            parse_time_ms=0.0,
        )

    async def _try_legacy_parsing(
        self, file_path: Path, content: str, errors: list[str]
    ) -> ast_module.AST | None:
        """Try different legacy parser interfaces to get AST."""
        tree = None

        # Method 1: Try parse method
        if hasattr(self.parser, "parse"):
            try:
                parse_result = self.parser.parse(content, str(file_path))  # type: ignore[call-non-callable]  # ty: ignore[call-non-callable]
                if isinstance(parse_result, dict) and parse_result.get("success"):
                    tree = parse_result.get("tree")
            except (SyntaxError, ValueError, TypeError, AttributeError) as e:
                logger.debug("parse() method failed: %s", str(e))

        # Method 2: Try parse_code method
        if tree is None and hasattr(self.parser, "parse_code"):
            try:
                tree = await self.parser.parse_code(content, str(file_path))
            except (SyntaxError, ValueError, TypeError, AttributeError) as e:
                logger.debug("parse_code() method failed: %s", str(e))

        # Method 3: Use Python AST directly
        if tree is None:
            try:
                tree = ast_module.parse(content, filename=str(file_path))
            except SyntaxError as e:
                logger.debug("ast_module.parse failed: %s", str(e))
                errors.append(f"AST parse error: {str(e)}")

        return tree

    async def _extract_with_legacy_components(
        self, tree: ast_module.AST, file_path: Path, content: str
    ) -> tuple[list[CoreCodeEntity], list[EntityRelation]]:
        """Extract entities and relationships using legacy components."""
        entities = []
        relationships = []

        try:
            legacy_entities = await self.entity_extractor.extract_entities(tree, str(file_path), "")  # type: ignore[invalid-argument-type]  # ty: ignore[invalid-argument-type]

            # Convert legacy entities to new format
            entities = [self._convert_legacy_entity(le, file_path) for le in legacy_entities]

            # Extract relationships using legacy extractor
            if hasattr(self.relation_extractor, "extract_all_relations"):
                legacy_relations = await self.relation_extractor.extract_all_relations(
                    tree,  # type: ignore[invalid-argument-type]  # ty: ignore[invalid-argument-type]
                    str(file_path),
                    "",
                )
                relationships = [self._convert_legacy_relation(lr) for lr in legacy_relations]

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("Legacy entity extraction failed: %s", str(e))
            # Fall back to direct AST parsing
            entities = await self._extract_entities_from_ast(tree, file_path, content)
            relationships = self._extract_relationships(tree, entities, file_path)

        return entities, relationships

    def _convert_legacy_entity(self, le: Any, file_path: Path) -> CoreCodeEntity:
        """Convert legacy entity to new CoreCodeEntity format."""
        if isinstance(le, dict):
            entity_name = le.get("name", "unknown")
            entity_id = le.get("id", f"entity:{entity_name}:{file_path}")
            entity_type = le.get("type", "function")
            # PythonEntityExtractor emits start_line/end_line, not *_number.
            line_number = le.get("line_number") or le.get("start_line") or 0
        else:
            entity_name = getattr(le, "name", "unknown")
            entity_id = getattr(le, "id", f"entity:{entity_name}:{file_path}")
            entity_type = getattr(le, "type", "function")
            line_number = getattr(le, "line_number", 0)

        return CoreCodeEntity(
            id=entity_id,
            name=entity_name,
            type=self._map_entity_type(entity_type),
            language="python",
            file_path=str(file_path),
            line_number=line_number,
            end_line_number=(
                (le.get("end_line_number") or le.get("end_line"))
                if isinstance(le, dict)
                else getattr(le, "end_line_number", None)
            ),
            column_start=(
                le.get("column_start")
                if isinstance(le, dict)
                else getattr(le, "column_start", None)
            ),
            column_end=(
                le.get("column_end") if isinstance(le, dict) else getattr(le, "column_end", None)
            ),
            visibility=(
                le.get("visibility") if isinstance(le, dict) else getattr(le, "visibility", None)
            ),
            code_snippet=(
                le.get("code_snippet", "")
                if isinstance(le, dict)
                else getattr(le, "code_snippet", "")
            ),
            docstring=(
                le.get("docstring") if isinstance(le, dict) else getattr(le, "docstring", None)
            ),
            signature=(
                le.get("signature", "") if isinstance(le, dict) else getattr(le, "signature", "")
            ),
            source_code=(
                le.get("source_code", "")
                if isinstance(le, dict)
                else getattr(le, "source_code", "")
            ),
            created_at=None,
            updated_at=None,
            properties=(
                (le.get("properties") or le.get("metadata") or {})
                if isinstance(le, dict)
                else getattr(le, "metadata", {}) or {}
            ),
        )

    def _convert_legacy_relation(self, lr: Any) -> EntityRelation:
        """Convert legacy relation to new EntityRelation format."""
        return EntityRelation(
            source_id=getattr(lr, "source_id", ""),
            target_id=getattr(lr, "target_id", ""),
            type=self._map_relation_type(getattr(lr, "type", "references")),
            properties=getattr(lr, "metadata", {}) or {},
            metadata=getattr(lr, "metadata", {}) or {},
        )

    def _calculate_complexity(
        self, content: str, entities: list[CoreCodeEntity]
    ) -> CoreComplexityMetrics:
        """Calculate complexity metrics for the code."""
        return CoreComplexityMetrics(
            lines_of_code=len(content.splitlines()),
            function_count=len([e for e in entities if e.type == EntityType.FUNCTION]),
            class_count=len([e for e in entities if e.type == EntityType.CLASS]),
            comment_lines=0,
            blank_lines=0,
            docstring_coverage=0.0,
        )

    async def _parse_with_ast(self, file_path: Path, content: str) -> ParseResult:
        """Parse using Python's native AST."""
        entities = []
        relationships = []
        errors = []
        tree = None

        try:
            # Parse the Python code
            tree = ast_module.parse(content, filename=str(file_path))

            # Extract entities
            for node in ast_module.walk(tree):
                entity = self._extract_entity_from_node(node, file_path, content)
                if entity:
                    entities.append(entity)

            # Extract imports
            for node in ast_module.walk(tree):
                if isinstance(node, ast_module.Import):
                    for alias in node.names:
                        entities.append(
                            CoreCodeEntity(
                                id=f"import:{alias.name}:{file_path}",
                                name=alias.name,
                                type=EntityType.IMPORT,
                                language="python",
                                file_path=str(file_path),
                                line_number=node.lineno,
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
                                properties={
                                    "import_type": "import",
                                    "alias": alias.asname,
                                },
                            )
                        )
                elif isinstance(node, ast_module.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        import_name = f"{module}.{alias.name}" if module else alias.name
                        entities.append(
                            CoreCodeEntity(
                                id=f"import:{import_name}:{file_path}",
                                name=import_name,
                                type=EntityType.IMPORT,
                                language="python",
                                file_path=str(file_path),
                                line_number=node.lineno,
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
                                properties={
                                    "import_type": "from_import",
                                    "module": module,
                                    "name": alias.name,
                                    "alias": alias.asname,
                                },
                            )
                        )

            # Extract basic relationships
            relationships = self._extract_relationships(tree, entities, file_path)

        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
        except (ValueError, TypeError, AttributeError) as e:
            errors.append(f"Parse error: {str(e)}")

        # Calculate complexity
        complexity = CoreComplexityMetrics(
            lines_of_code=len(content.splitlines()),
            function_count=len([e for e in entities if e.type == EntityType.FUNCTION]),
            class_count=len([e for e in entities if e.type == EntityType.CLASS]),
            comment_lines=len(
                [line for line in content.splitlines() if line.strip().startswith("#")]
            ),
            blank_lines=len([line for line in content.splitlines() if not line.strip()]),
            docstring_coverage=0.0,  # Could be calculated if needed
        )

        return ParseResult(
            language="python",
            file_path=str(file_path),
            parser_used=ParserType.AST_NATIVE,
            success=len(errors) == 0,
            entities=entities,
            relationships=relationships,
            lines_of_code=len(content.splitlines()),
            complexity_score=complexity.cyclomatic_complexity,
            errors=errors,
            parse_time_ms=0.0,
            ast=tree,  # Include the AST
        )

    def _extract_entity_from_node(
        self,
        node: ast_module.AST,
        file_path: Path,
        content: str,
    ) -> CoreCodeEntity | None:
        """Extract entity from AST node."""
        if isinstance(node, ast_module.ClassDef):
            # Extract class
            is_abstract = any(
                isinstance(decorator, ast_module.Name) and decorator.id == "abstractmethod"
                for decorator in node.decorator_list
                if isinstance(decorator, ast_module.Name)
            )

            return CoreCodeEntity(
                id=f"class:{node.name}:{file_path}",
                name=node.name,
                type=EntityType.CLASS,
                language="python",
                file_path=str(file_path),
                line_number=node.lineno,
                end_line_number=getattr(node, "end_lineno", None),
                column_start=getattr(node, "col_offset", None),
                column_end=getattr(node, "end_col_offset", None),
                visibility=None,
                code_snippet=None,
                docstring=ast_module.get_docstring(node),
                signature=None,
                source_code=None,
                created_at=None,
                updated_at=None,
                is_abstract=is_abstract,
                properties={
                    "bases": [self._get_node_name(base) for base in node.bases],
                    "decorators": [self._get_node_name(dec) for dec in node.decorator_list],
                },
            )

        if isinstance(node, ast_module.FunctionDef):
            # Extract function/method
            is_async = isinstance(node, ast_module.AsyncFunctionDef)
            is_method = hasattr(node, "_parent") and isinstance(
                getattr(node, "_parent", None), ast_module.ClassDef
            )

            # Get decorators
            decorators = [self._get_node_name(dec) for dec in node.decorator_list]
            is_static = "staticmethod" in decorators
            is_property = "property" in decorators

            return CoreCodeEntity(
                id=f"function:{node.name}:{file_path}:{node.lineno}",
                name=node.name,
                type=EntityType.FUNCTION,
                language="python",
                file_path=str(file_path),
                line_number=node.lineno,
                end_line_number=getattr(node, "end_lineno", None),
                column_start=getattr(node, "col_offset", None),
                column_end=getattr(node, "end_col_offset", None),
                visibility=None,
                code_snippet=None,
                docstring=ast_module.get_docstring(node),
                signature=self._build_function_signature(node),
                source_code=None,
                created_at=None,
                updated_at=None,
                is_async=is_async,
                is_static=is_static,
                properties={
                    "is_method": is_method,
                    "is_property": is_property,
                    "decorators": decorators,
                    "arguments": [arg.arg for arg in node.args.args],
                },
            )

        if isinstance(node, ast_module.AsyncFunctionDef):
            # Async functions are handled above
            pass

        return None

    def _get_node_name(self, node: ast_module.AST) -> str:
        """Get name from AST node."""
        if isinstance(node, ast_module.Name):
            return node.id
        if isinstance(node, ast_module.Attribute):
            return f"{self._get_node_name(node.value)}.{node.attr}"
        if isinstance(node, ast_module.Call) and isinstance(node.func, ast_module.Name):
            return node.func.id
        return ast_module.dump(node)

    def _build_function_signature(self, node: ast_module.FunctionDef) -> str:
        """Build function signature from AST node."""
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_node_name(arg.annotation)}"
            args.append(arg_str)

        signature = f"{node.name}({', '.join(args)})"

        if node.returns:
            signature += f" -> {self._get_node_name(node.returns)}"

        return signature

    def _extract_relationships(
        self, tree: ast_module.AST, entities: list[CoreCodeEntity], file_path: Path
    ) -> list[EntityRelation]:
        """Extract relationships from AST."""
        relationships = []

        # Extract class inheritance
        for node in ast_module.walk(tree):
            if isinstance(node, ast_module.ClassDef):
                class_entity = next(
                    (e for e in entities if e.name == node.name and e.type == EntityType.CLASS),
                    None,
                )
                if class_entity:
                    for base in node.bases:
                        base_name = self._get_node_name(base)
                        # Look for base class in entities
                        base_entity = next(
                            (
                                e
                                for e in entities
                                if e.name == base_name and e.type == EntityType.CLASS
                            ),
                            None,
                        )
                        if base_entity:
                            relationships.append(
                                EntityRelation(
                                    source_id=class_entity.id,
                                    target_id=base_entity.id,
                                    type=RelationType.INHERITS,
                                    properties={"inheritance_type": "class"},
                                )
                            )

        # Module contains entities
        module_name = file_path.stem
        for entity in entities:
            if entity.type in [EntityType.CLASS, EntityType.FUNCTION]:
                relationships.append(
                    EntityRelation(
                        source_id=f"module:{module_name}:{file_path}",
                        target_id=entity.id,
                        type=RelationType.CONTAINS,
                        properties={"relationship_context": "module_member"},
                    )
                )

        return relationships

    async def _extract_entities_from_ast(
        self, tree: ast_module.AST, file_path: Path, content: str
    ) -> list[CoreCodeEntity]:
        """Extract entities directly from AST."""
        entities = []

        for node in ast_module.walk(tree):
            entity = self._extract_entity_from_node(node, file_path, content)
            if entity:
                entities.append(entity)

        # Add imports
        for node in ast_module.walk(tree):
            if isinstance(node, ast_module.Import):
                for alias in node.names:
                    entities.append(
                        CoreCodeEntity(
                            id=f"import:{alias.name}:{file_path}",
                            name=alias.name,
                            type=EntityType.IMPORT,
                            language="python",
                            file_path=str(file_path),
                            line_number=node.lineno,
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
                            properties={
                                "import_type": "import",
                                "alias": alias.asname,
                            },
                        )
                    )
            elif isinstance(node, ast_module.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    import_name = f"{module}.{alias.name}" if module else alias.name
                    entities.append(
                        CoreCodeEntity(
                            id=f"import:{import_name}:{file_path}",
                            name=import_name,
                            type=EntityType.IMPORT,
                            language="python",
                            file_path=str(file_path),
                            line_number=node.lineno,
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
                            properties={
                                "import_type": "from_import",
                                "module": module,
                                "name": alias.name,
                                "alias": alias.asname,
                            },
                        )
                    )

        return entities

    def _map_entity_type(self, legacy_type: str) -> EntityType:
        """Map legacy entity type to new EntityType enum."""
        mapping = {
            "class": EntityType.CLASS,
            "function": EntityType.FUNCTION,
            "method": EntityType.FUNCTION,
            "module": EntityType.MODULE,
            "variable": EntityType.VARIABLE,
            "constant": EntityType.CONSTANT,
            "import": EntityType.IMPORT,
            "decorator": EntityType.DECORATOR,
        }
        return mapping.get(legacy_type.lower(), EntityType.FUNCTION)

    def _map_relation_type(self, legacy_type: str) -> RelationType:
        """Map legacy relation type to new RelationType enum."""
        mapping = {
            "inheritance": RelationType.INHERITS,
            "composition": RelationType.CONTAINS,
            "dependency": RelationType.DEPENDS_ON,
            "call": RelationType.CALLS,
            "import": RelationType.IMPORTS,
            "usage": RelationType.REFERENCES,
        }
        return mapping.get(legacy_type.lower(), RelationType.REFERENCES)

    # Required abstract methods from LanguageProcessor
    async def extract_entities(self, ast: Any, file_path: Path) -> list[CoreCodeEntity]:
        """Extract code entities from AST."""
        entities = []

        if isinstance(ast, ast_module.AST):
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except (OSError, FileNotFoundError):
                content = ""
            for node in ast_module.walk(ast):
                entity = self._extract_entity_from_node(node, file_path, content)
                if entity:
                    entities.append(entity)

        return entities

    async def extract_relationships(
        self, ast: Any, entities: list[CoreCodeEntity], file_path: Path
    ) -> list[EntityRelation]:
        """Extract relationships between entities."""
        if isinstance(ast, ast_module.AST):
            return self._extract_relationships(ast, entities, file_path)
        return []
