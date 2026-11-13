"""
Advanced TypeScript Language Processor

Advanced TypeScript processor with comprehensive type system analysis,
including generics, decorators, modules, and advanced TypeScript features.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from models.core_models import CoreCodeEntity, EntityRelation, EntityType

from ..base import (
    LanguageConfig,
    LanguageFeatures,
    LanguageProcessor,
    ParseResult,
    ParserType,
)
from ..registry import register_language

logger = logging.getLogger(__name__)


@dataclass
class TypeScriptTypeInfo:
    """Information about TypeScript types."""

    name: str
    kind: str  # interface, type, class, enum, generic
    parameters: list[str] = field(default_factory=list)
    extends: list[str] = field(default_factory=list)
    implements: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    is_generic: bool = False
    is_abstract: bool = False
    is_exported: bool = False


class TSLanguageProcessor(LanguageProcessor):
    """Advanced TypeScript language processor with comprehensive type system analysis."""

    def __init__(self, config: LanguageConfig | None = None):
        if config is None:
            config = LanguageConfig(
                name="typescript",
                display_name="TypeScript",
                extensions=[".ts", ".tsx"],
                parser_type=ParserType.EXTERNAL_TOOL,
                fallback_parsers=[ParserType.REGEX_FALLBACK],
                external_command="tsc",
                features=LanguageFeatures(
                    classes=True,
                    inheritance=True,
                    interfaces=True,
                    abstract_classes=True,
                    functions=True,
                    static_typing=True,
                    generics=True,
                    modules=True,
                    annotations=True,
                    async_support=True,
                ),
            )
        super().__init__(config)
        self.type_registry: dict[str, TypeScriptTypeInfo] = {}
        self._time_module = time

    @property
    def patterns(self) -> dict[str, re.Pattern[str]]:
        """Get TypeScript-specific regex patterns."""
        return {
            "interface": re.compile(
                r"(?:export\s+)?interface\s+(\w+)(?:<([^>]+)>)?\s*(?:extends\s+([^{]+))?\s*{",
                re.MULTILINE,
            ),
            "type_alias": re.compile(
                r"(?:export\s+)?type\s+(\w+)(?:<([^>]+)>)?\s*=\s*([^;]+);",
                re.MULTILINE,
            ),
            "class": re.compile(
                r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:<([^>]+)>)?\s*"
                r"(?:extends\s+(\w+))?\s*(?:implements\s+([^{]+))?\s*{",
                re.MULTILINE,
            ),
            "enum": re.compile(r"(?:export\s+)?(?:const\s+)?enum\s+(\w+)\s*{", re.MULTILINE),
            "function": re.compile(
                r"(?:export\s+)?(?:async\s+)?function\s+(\w+)(?:<([^>]+)>)?\s*"
                r"\(([^)]*)\)\s*:\s*([^{;]+)",
                re.MULTILINE,
            ),
            "arrow_function": re.compile(
                r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*:\s*[^=]*=\s*"
                r"(?:async\s+)?\(([^)]*)\)\s*:\s*([^=]+)\s*=>",
                re.MULTILINE,
            ),
            "method": re.compile(
                r"(?:public|private|protected|static)?\s*(?:async\s+)?(\w+)"
                r"(?:<([^>]+)>)?\s*\(([^)]*)\)\s*:\s*([^{;]+)",
                re.MULTILINE,
            ),
            "property": re.compile(
                r"(?:public|private|protected|static|readonly)?\s*(\w+)\s*:"
                r"\s*([^;=]+)(?:[;=]|$)",
                re.MULTILINE,
            ),
            "decorator": re.compile(r"@(\w+)(?:\(([^)]*)\))?", re.MULTILINE),
            "import": re.compile(
                r"import\s+(?:(?:\*\s+as\s+\w+)|(?:{\s*[^}]+\s*})|(?:\w+))\s+"
                r'from\s+[\'"]([^\'"]+)[\'"]',
                re.MULTILINE,
            ),
            "export": re.compile(
                r'export\s+(?:\*\s+from\s+[\'"]([^\'"]+)[\'"]|(?:default\s+)?'
                r"(?:class|interface|function|const|let|var|type|enum)\s+(\w+))",
                re.MULTILINE,
            ),
            "generic_constraint": re.compile(r"(\w+)\s+extends\s+([^,>]+)", re.MULTILINE),
            "conditional_type": re.compile(
                r"(\w+)\s+extends\s+([^?]+)\s*\?\s*([^:]+)\s*:\s*(.+)",
                re.MULTILINE,
            ),
            "mapped_type": re.compile(
                r"{\s*\[(\w+)\s+in\s+([^\]]+)\]\s*:\s*([^}]+)\s*}",
                re.MULTILINE,
            ),
            "namespace": re.compile(r"(?:export\s+)?namespace\s+(\w+)\s*{", re.MULTILINE),
            "module": re.compile(r'declare\s+module\s+[\'"]([^\'"]+)[\'"]', re.MULTILINE),
        }

    async def _initialize_impl(self) -> None:
        """Initialize TypeScript processor."""
        logger.debug("TypeScript processor initialized with comprehensive type analysis")

    async def parse_file(self, file_path: Path, content: str) -> ParseResult:
        """Parse TypeScript file with comprehensive type system analysis."""

        start_time = self._get_time()
        entities = []
        relationships: list[EntityRelation] = []
        errors: list[str] = []
        warnings: list[str] = []

        try:
            # Clear type registry for this file
            self.type_registry.clear()

            # Extract entities with type information
            entities.extend(await self._extract_interfaces(content, file_path))
            entities.extend(await self._extract_type_aliases(content, file_path))
            entities.extend(await self._extract_classes(content, file_path))
            entities.extend(await self._extract_enums(content, file_path))
            entities.extend(await self._extract_functions(content, file_path))
            entities.extend(await self._extract_namespaces(content, file_path))
            entities.extend(await self._extract_modules(content, file_path))

            # Extract relationships
            relationships.extend(await self._extract_type_relationships(content, entities))
            relationships.extend(await self._extract_import_relationships(content, file_path))
            relationships.extend(await self._extract_generic_relationships(content, entities))
            relationships.extend(await self._extract_decorator_relationships(content, entities))

            # Calculate complexity
            complexity = self._calculate_typescript_complexity(content, entities)

            parse_time = self._get_time() - start_time

            return ParseResult(
                success=True,
                file_path=str(file_path),
                language="typescript",
                parser_used=ParserType.REGEX_FALLBACK,
                entities=entities,
                relationships=relationships,
                lines_of_code=len(content.splitlines()),
                complexity_score=complexity,
                parse_time_ms=parse_time * 1000,
                errors=errors,
                warnings=warnings,
                ast=None,
                properties={
                    "type_count": len(self.type_registry),
                    "has_generics": any(t.is_generic for t in self.type_registry.values()),
                    "has_decorators": any(t.decorators for t in self.type_registry.values()),
                    "export_count": sum(1 for t in self.type_registry.values() if t.is_exported),
                },
            )

        except (ValueError, TypeError, AttributeError) as e:
            errors.append(f"Parse error: {str(e)}")
            logger.error("TypeScript parsing failed for %s: %s", file_path, e)

            return ParseResult(
                success=False,
                file_path=str(file_path),
                language="typescript",
                parser_used=ParserType.REGEX_FALLBACK,
                entities=[],
                relationships=[],
                lines_of_code=len(content.splitlines()) if content else 0,
                complexity_score=0.0,
                parse_time_ms=(self._get_time() - start_time) * 1000,
                errors=errors,
                warnings=warnings,
                ast=None,
            )

    async def _extract_interfaces(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract TypeScript interfaces with detailed type information."""
        entities = []

        for match in self.patterns["interface"].finditer(content):
            name = match.group(1)
            type_params = match.group(2)
            extends_clause = match.group(3)

            # Parse generic parameters
            generic_params = []
            if type_params:
                generic_params = [p.strip() for p in type_params.split(",")]

            # Parse extends clause
            extends_types = []
            if extends_clause:
                extends_types = [t.strip() for t in extends_clause.split(",")]

            # Create type info
            type_info = TypeScriptTypeInfo(
                name=name,
                kind="interface",
                parameters=generic_params,
                extends=extends_types,
                is_generic=bool(generic_params),
                is_exported="export" in match.group(0),
            )
            self.type_registry[name] = type_info

            entity = CoreCodeEntity(
                id=f"{file_path}:{name}",
                name=name,
                type=EntityType.INTERFACE,
                language="typescript",
                file_path=str(file_path),
                line_number=content[: match.start()].count("\n") + 1,
                end_line_number=None,
                column_start=None,
                column_end=None,
                visibility=None,
                code_snippet=None,
                docstring=None,
                source_code=None,
                created_at=None,
                updated_at=None,
                signature=match.group(0).split("{")[0].strip(),
                is_abstract=False,
                is_static=False,
                properties={
                    "generic_parameters": generic_params,
                    "extends": extends_types,
                    "is_exported": type_info.is_exported,
                    "type_kind": "interface",
                },
            )
            entities.append(entity)

        return entities

    async def _extract_type_aliases(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract TypeScript type aliases."""
        entities = []

        for match in self.patterns["type_alias"].finditer(content):
            name = match.group(1)
            type_params = match.group(2)
            type_definition = match.group(3)

            # Parse generic parameters
            generic_params = []
            if type_params:
                generic_params = [p.strip() for p in type_params.split(",")]

            # Analyze type definition
            is_union = "|" in type_definition
            is_conditional = "?" in type_definition and ":" in type_definition
            is_mapped = re.search(r"{\s*\[.*\].*}", type_definition)

            # Create type info
            type_info = TypeScriptTypeInfo(
                name=name,
                kind="type_alias",
                parameters=generic_params,
                is_generic=bool(generic_params),
                is_exported="export" in match.group(0),
            )
            self.type_registry[name] = type_info

            entity = CoreCodeEntity(
                id=f"{file_path}:{name}",
                name=name,
                type=EntityType.TYPE_ALIAS,
                language="typescript",
                file_path=str(file_path),
                line_number=content[: match.start()].count("\n") + 1,
                end_line_number=None,
                column_start=None,
                column_end=None,
                visibility=None,
                code_snippet=None,
                docstring=None,
                source_code=None,
                created_at=None,
                updated_at=None,
                signature=match.group(0).strip(),
                properties={
                    "generic_parameters": generic_params,
                    "type_definition": type_definition.strip(),
                    "is_union_type": is_union,
                    "is_conditional_type": is_conditional,
                    "is_mapped_type": bool(is_mapped),
                    "is_exported": type_info.is_exported,
                    "type_kind": "type_alias",
                },
            )
            entities.append(entity)

        return entities

    def _extract_class_decorators(self, content: str) -> dict[int, list[str]]:
        """Extract decorators that might apply to classes."""
        decorators_by_line: dict[int, list[str]] = {}
        for decorator_match in self.patterns["decorator"].finditer(content):
            line_num = content[: decorator_match.start()].count("\n") + 1
            decorator_name = decorator_match.group(1)
            decorators_by_line[line_num] = decorators_by_line.get(line_num, []) + [decorator_name]
        return decorators_by_line

    def _get_class_decorators(
        self, decorators_by_line: dict[int, list[str]], line_num: int
    ) -> list[str]:
        """Get decorators for a specific class."""
        class_decorators = []
        for decorator_line in range(max(1, line_num - 5), line_num):
            if decorator_line in decorators_by_line:
                class_decorators.extend(decorators_by_line[decorator_line])
        return class_decorators

    async def _extract_classes(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract TypeScript classes with inheritance and decorator information."""
        entities = []
        decorators_by_line = self._extract_class_decorators(content)

        for match in self.patterns["class"].finditer(content):
            name = match.group(1)
            type_params = match.group(2)
            extends_class = match.group(3)
            implements_clause = match.group(4)

            line_num = content[: match.start()].count("\n") + 1
            is_abstract = "abstract" in match.group(0)

            # Parse generic parameters
            generic_params = []
            if type_params:
                generic_params = [p.strip() for p in type_params.split(",")]

            # Parse implements clause
            implements_types = []
            if implements_clause:
                implements_types = [t.strip() for t in implements_clause.split(",")]

            # Get decorators for this class
            class_decorators = self._get_class_decorators(decorators_by_line, line_num)

            # Create type info
            type_info = TypeScriptTypeInfo(
                name=name,
                kind="class",
                parameters=generic_params,
                extends=[extends_class] if extends_class else [],
                implements=implements_types,
                decorators=class_decorators,
                is_generic=bool(generic_params),
                is_abstract=is_abstract,
                is_exported="export" in match.group(0),
            )
            self.type_registry[name] = type_info

            entity = CoreCodeEntity(
                id=f"{file_path}:{name}",
                name=name,
                type=EntityType.CLASS,
                language="typescript",
                file_path=str(file_path),
                line_number=line_num,
                end_line_number=None,
                column_start=None,
                column_end=None,
                visibility=None,
                code_snippet=None,
                docstring=None,
                source_code=None,
                created_at=None,
                updated_at=None,
                signature=match.group(0).split("{")[0].strip(),
                is_abstract=is_abstract,
                is_static=False,
                properties={
                    "generic_parameters": generic_params,
                    "extends": extends_class,
                    "implements": implements_types,
                    "decorators": class_decorators,
                    "is_exported": type_info.is_exported,
                    "type_kind": "class",
                },
            )
            entities.append(entity)

        return entities

    async def _extract_enums(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract TypeScript enums."""
        entities = []

        for match in self.patterns["enum"].finditer(content):
            name = match.group(1)
            is_const = "const" in match.group(0)

            # Create type info
            type_info = TypeScriptTypeInfo(
                name=name, kind="enum", is_exported="export" in match.group(0)
            )
            self.type_registry[name] = type_info

            entity = CoreCodeEntity(
                id=f"{file_path}:{name}",
                name=name,
                type=EntityType.ENUM,
                language="typescript",
                file_path=str(file_path),
                line_number=content[: match.start()].count("\n") + 1,
                end_line_number=None,
                column_start=None,
                column_end=None,
                visibility=None,
                code_snippet=None,
                docstring=None,
                source_code=None,
                created_at=None,
                updated_at=None,
                signature=match.group(0).replace("{", "").strip(),
                properties={
                    "is_const_enum": is_const,
                    "is_exported": type_info.is_exported,
                    "type_kind": "enum",
                },
            )
            entities.append(entity)

        return entities

    async def _extract_functions(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract TypeScript functions with type information."""
        entities = []

        # Regular functions
        for match in self.patterns["function"].finditer(content):
            name = match.group(1)
            type_params = match.group(2)
            params = match.group(3)
            return_type = match.group(4)

            # Parse generic parameters
            generic_params = []
            if type_params:
                generic_params = [p.strip() for p in type_params.split(",")]

            is_async = "async" in match.group(0)

            entity = CoreCodeEntity(
                id=f"{file_path}:{name}",
                name=name,
                type=EntityType.FUNCTION,
                language="typescript",
                file_path=str(file_path),
                line_number=content[: match.start()].count("\n") + 1,
                end_line_number=None,
                column_start=None,
                column_end=None,
                visibility=None,
                code_snippet=None,
                docstring=None,
                source_code=None,
                created_at=None,
                updated_at=None,
                signature=match.group(0).strip(),
                is_async=is_async,
                properties={
                    "generic_parameters": generic_params,
                    "parameters": params.strip(),
                    "return_type": return_type.strip(),
                    "is_exported": "export" in match.group(0),
                    "type_kind": "function",
                },
            )
            entities.append(entity)

        # Arrow functions
        for match in self.patterns["arrow_function"].finditer(content):
            name = match.group(1)
            params = match.group(2)
            return_type = match.group(3)

            is_async = "async" in match.group(0)

            entity = CoreCodeEntity(
                id=f"{file_path}:{name}",
                name=name,
                type=EntityType.FUNCTION,
                language="typescript",
                file_path=str(file_path),
                line_number=content[: match.start()].count("\n") + 1,
                end_line_number=None,
                column_start=None,
                column_end=None,
                visibility=None,
                code_snippet=None,
                docstring=None,
                source_code=None,
                created_at=None,
                updated_at=None,
                signature=f"{name} = ({params}) : {return_type} =>",
                is_async=is_async,
                properties={
                    "parameters": params.strip(),
                    "return_type": return_type.strip(),
                    "is_arrow_function": True,
                    "is_exported": "export" in match.group(0),
                    "type_kind": "arrow_function",
                },
            )
            entities.append(entity)

        return entities

    async def _extract_namespaces(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract TypeScript namespaces."""
        entities = []

        for match in self.patterns["namespace"].finditer(content):
            name = match.group(1)

            entity = CoreCodeEntity(
                id=f"{file_path}:{name}",
                name=name,
                type=EntityType.NAMESPACE,
                language="typescript",
                file_path=str(file_path),
                line_number=content[: match.start()].count("\n") + 1,
                end_line_number=None,
                column_start=None,
                column_end=None,
                visibility=None,
                code_snippet=None,
                docstring=None,
                source_code=None,
                created_at=None,
                updated_at=None,
                signature=match.group(0).replace("{", "").strip(),
                properties={
                    "is_exported": "export" in match.group(0),
                    "type_kind": "namespace",
                },
            )
            entities.append(entity)

        return entities

    async def _extract_modules(self, content: str, file_path: Path) -> list[CoreCodeEntity]:
        """Extract TypeScript module declarations."""
        entities = []

        for match in self.patterns["module"].finditer(content):
            name = match.group(1)

            entity = CoreCodeEntity(
                id=f"{file_path}:{name}",
                name=name,
                type=EntityType.MODULE,
                language="typescript",
                file_path=str(file_path),
                line_number=content[: match.start()].count("\n") + 1,
                end_line_number=None,
                column_start=None,
                column_end=None,
                visibility=None,
                code_snippet=None,
                docstring=None,
                source_code=None,
                created_at=None,
                updated_at=None,
                signature=match.group(0).strip(),
                properties={
                    "module_path": name,
                    "type_kind": "module_declaration",
                },
            )
            entities.append(entity)

        return entities

    async def _extract_type_relationships(
        self, _content: str, entities: list[CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Extract type-based relationships (extends, implements, etc.)."""
        relationships: list[EntityRelation] = []

        for entity in entities:
            # Inheritance relationships
            if entity.properties and "extends" in entity.properties:
                extends_type = entity.properties["extends"]
                if extends_type:
                    relationships.append(
                        EntityRelation(
                            source_id=entity.id,
                            target_id=f"{entity.file_path}:{extends_type}",
                            type="extends",
                            properties={"relationship_kind": "inheritance"},
                        )
                    )

            # Interface implementation relationships
            if entity.properties and "implements" in entity.properties:
                implements_list = entity.properties["implements"]
                if implements_list:
                    for interface_name in implements_list:
                        relationships.append(
                            EntityRelation(
                                source_id=entity.id,
                                target_id=f"{entity.file_path}:{interface_name}",
                                type="implements",
                                properties={"relationship_kind": "interface_implementation"},
                            )
                        )

        return relationships

    async def _extract_import_relationships(
        self, content: str, file_path: Path
    ) -> list[EntityRelation]:
        """Extract import/export relationships."""
        relationships: list[EntityRelation] = []

        for match in self.patterns["import"].finditer(content):
            module_path = match.group(1)

            relationships.append(
                EntityRelation(
                    source_id=str(file_path),
                    target_id=module_path,
                    type="imports",
                    properties={
                        "relationship_kind": "module_dependency",
                        "import_statement": match.group(0).strip(),
                    },
                )
            )

        return relationships

    async def _extract_generic_relationships(
        self, content: str, entities: list[CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Extract generic type constraint relationships."""
        relationships: list[EntityRelation] = []

        for match in self.patterns["generic_constraint"].finditer(content):
            type_param = match.group(1)
            constraint = match.group(2)

            # Find the entity this constraint belongs to
            for entity in entities:
                if (
                    entity.properties
                    and "generic_parameters" in entity.properties
                    and any(
                        gp == type_param or gp.split()[0] == type_param
                        for gp in entity.properties["generic_parameters"]
                    )
                ):
                    relationships.append(
                        EntityRelation(
                            source_id=entity.id,
                            target_id=f"{entity.file_path}:{constraint.strip()}",
                            type="generic_of",
                            properties={
                                "relationship_kind": "type_constraint",
                                "type_parameter": type_param,
                                "constraint": constraint.strip(),
                            },
                        )
                    )
                    break

        return relationships

    async def _extract_decorator_relationships(
        self, _content: str, entities: list[CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Extract decorator relationships."""
        relationships: list[EntityRelation] = []

        for entity in entities:
            if entity.properties and "decorators" in entity.properties:
                for decorator in entity.properties["decorators"]:
                    relationships.append(
                        EntityRelation(
                            source_id=entity.id,
                            target_id=f"decorator:{decorator}",
                            type="annotates",
                            properties={
                                "relationship_kind": "decoration",
                                "decorator_name": decorator,
                            },
                        )
                    )

        return relationships

    def _calculate_typescript_complexity(
        self, content: str, _entities: list[CoreCodeEntity]
    ) -> float:
        """Calculate TypeScript-specific complexity."""
        complexity = 1.0  # Base complexity

        # Type system complexity
        complexity += len(self.type_registry) * 0.1

        # Generic complexity
        generic_count = sum(1 for t in self.type_registry.values() if t.is_generic)
        complexity += generic_count * 0.5

        # Decorator complexity
        decorator_count = sum(len(t.decorators) for t in self.type_registry.values())
        complexity += decorator_count * 0.3

        # Advanced type features
        conditional_types = len(self.patterns["conditional_type"].findall(content))
        mapped_types = len(self.patterns["mapped_type"].findall(content))
        complexity += (conditional_types + mapped_types) * 0.7

        # Control flow complexity
        control_keywords = [
            "i",
            "else",
            "while",
            "for",
            "switch",
            "try",
            "catch",
        ]
        for keyword in control_keywords:
            complexity += content.lower().count(keyword) * 0.2

        return complexity

    def _get_time(self) -> float:
        """Get current time for performance measurement."""
        return self._time_module.perf_counter()

    async def extract_entities(self, ast: Any, file_path: Path) -> list[CoreCodeEntity]:
        """Extract entities from AST (not used in regex implementation)."""
        return []

    async def extract_relationships(
        self, ast: Any, entities: list[CoreCodeEntity], file_path: Path
    ) -> list[EntityRelation]:
        """Extract relationships from AST (not used in regex implementation)."""
        return []


# Configuration for TypeScript
TYPESCRIPT_CONFIG = LanguageConfig(
    name="typescript",
    display_name="TypeScript",
    extensions=[".ts", ".tsx", ".d.ts"],
    parser_type=ParserType.REGEX_FALLBACK,
    fallback_parsers=[ParserType.TREE_SITTER],
    tree_sitter_grammar="typescript",
    external_command=None,
    language_server_command=None,
    features=LanguageFeatures(
        classes=True,
        interfaces=True,
        functions=True,
        generics=True,
        decorators=True,
        namespaces=True,
        modules=True,
        static_typing=True,
        type_inference=True,
        async_support=True,
        nullable_types=True,
        inheritance=True,
        abstract_classes=True,
        lambdas=True,
        closures=True,
        higher_order_functions=True,
        packages=True,
        imports=True,
        annotations=True,
        garbage_collected=True,
    ),
    ignore_patterns=["node_modules/**", "**/*.test.ts", "**/*.spec.ts"],
    test_patterns=["**/*.test.ts", "**/*.spec.ts", "**/*.test.tsx"],
    max_file_size_mb=5,
    timeout_seconds=30,
)

# This will automatically register the processor when the module is imported
register_language(TYPESCRIPT_CONFIG, TSLanguageProcessor)
