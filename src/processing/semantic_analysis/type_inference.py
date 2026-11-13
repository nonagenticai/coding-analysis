"""
Type Inference and Checking Module

This module provides functionality for analyzing type systems, inferring types,
and detecting type-related relationships between code entities.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    EntityRelation,
    EntityType,
)

from .semantic_parser import SemanticContext, SemanticRelationType

logger = logging.getLogger(__name__)


class TypeRelationship(StrEnum):
    """Types of relationships between types."""

    INHERITANCE = "inheritance"
    IMPLEMENTATION = "implementation"
    COMPOSITION = "composition"
    AGGREGATION = "aggregation"
    DEPENDENCY = "dependency"
    GENERIC_CONSTRAINT = "generic_constraint"
    TYPE_ALIAS = "type_alias"
    CONVERSION = "conversion"


@dataclass
class TypeInfo:
    """Information about a type."""

    name: str
    kind: str  # class, interface, enum, type_alias, primitive
    namespace: str | None = None
    generic_parameters: list[str] = field(default_factory=list)
    base_types: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    # Language-specific information
    language: str | None = None
    is_nullable: bool = False
    is_mutable: bool = True


@dataclass
class TypeConstraint:
    """Represents a type constraint or bound."""

    type_parameter: str
    constraint_type: str
    constraint_kind: str  # extends, implements, where, trait_bound
    properties: dict[str, Any] = field(default_factory=dict)


class TypeSystemAnalyzer:
    """Analyzer for advanced type system relationships."""

    def __init__(self):
        self.type_registry: dict[str, TypeInfo] = {}
        self.type_constraints: list[TypeConstraint] = []
        self.type_relationships: list[EntityRelation] = []

    async def analyze_type_relationships(
        self, entities: list[CoreCodeEntity], contexts: dict[str, SemanticContext]
    ) -> list[EntityRelation]:
        """Analyze type system relationships between entities.

        Args:
            entities: List of code entities to analyze
            contexts: Language-specific semantic contexts

        Returns:
            List of EntityRelation objects representing type relationships
        """
        relationships = []

        # Build type registry
        self._build_type_registry(entities, contexts)

        # Generic type constraints
        generic_rels = await self._analyze_generic_constraints(entities)
        relationships.extend(generic_rels)

        # Type alias resolution
        alias_rels = await self._analyze_type_aliases(entities, contexts)
        relationships.extend(alias_rels)

        # Polymorphic relationships
        poly_rels = await self._analyze_polymorphic_usage(entities)
        relationships.extend(poly_rels)

        # Type conversion relationships
        conversion_rels = await self._analyze_type_conversions(entities)
        relationships.extend(conversion_rels)

        # Interface binding relationships
        interface_rels = await self._analyze_interface_bindings(entities)
        relationships.extend(interface_rels)

        return relationships

    def _build_type_registry(
        self, entities: list[CoreCodeEntity], contexts: dict[str, SemanticContext]
    ):
        """Build a registry of all types in the codebase.

        Args:
            entities: List of code entities
            contexts: Language-specific contexts
        """
        for entity in entities:
            if entity.type in [
                EntityType.CLASS,
                EntityType.INTERFACE,
                EntityType.ENUM,
                EntityType.TYPE_ALIAS,
            ]:
                type_info = self._extract_type_info(entity, contexts.get(entity.file_path))
                self.type_registry[entity.name] = type_info

    def _extract_type_info(
        self, entity: CoreCodeEntity, context: SemanticContext | None
    ) -> TypeInfo:
        """Extract type information from an entity.

        Args:
            entity: Code entity to analyze
            context: Semantic context for the entity

        Returns:
            TypeInfo object containing extracted type information
        """
        # Map entity types to type kinds
        type_kind_map = {
            EntityType.CLASS: "class",
            EntityType.INTERFACE: "interface",
            EntityType.ENUM: "enum",
            EntityType.TYPE_ALIAS: "type_alias",
        }

        type_info = TypeInfo(
            name=entity.name,
            kind=type_kind_map.get(entity.type, "unknown"),
            language=entity.language,
        )

        # Extract generic parameters
        if entity.signature:
            generic_match = re.search(r"<([^>]+)>", entity.signature)
            if generic_match:
                params = [p.strip() for p in generic_match.group(1).split(",")]
                type_info.generic_parameters = params

        # Extract base types/inheritance
        if entity.signature:
            # Java/C# style inheritance
            extends_match = re.search(r"extends\s+([^\s{]+)", entity.signature)
            if extends_match:
                type_info.base_types.append(extends_match.group(1))

            # Java/C# style implementation
            implements_match = re.search(r"implements\s+([^{]+)", entity.signature)
            if implements_match:
                interfaces = [i.strip() for i in implements_match.group(1).split(",")]
                type_info.base_types.extend(interfaces)

            # Rust trait implementation
            trait_match = re.search(r"impl\s+([^\s]+)\s+for", entity.signature)
            if trait_match:
                type_info.base_types.append(trait_match.group(1))

        # Extract namespace from context
        if context:
            type_info.namespace = list(context.namespaces)[0] if context.namespaces else None

        # Extract additional properties
        type_info.properties = entity.properties.copy()

        return type_info

    async def _analyze_generic_constraints(
        self, entities: list[CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Analyze generic type constraints and bounds.

        Args:
            entities: List of code entities to analyze

        Returns:
            List of EntityRelation objects representing type constraints
        """
        relationships = []

        constraint_patterns = [
            r"<(\w+)\s+extends\s+(\w+)>",  # Java
            r"<(\w+):\s*(\w+)>",  # Rust trait bounds
            r"template\s*<[^>]*class\s+(\w+)[^>]*>",  # C++ templates
            r"where\s+(\w+):\s*(\w+)",  # C# generic constraints
            r"<(\w+)\s+super\s+(\w+)>",  # Java wildcards
        ]

        for entity in entities:
            if entity.type in [
                EntityType.CLASS,
                EntityType.FUNCTION,
                EntityType.INTERFACE,
            ]:
                search_text = f"{entity.signature or ''} {entity.properties.get('content', '')}"

                for pattern in constraint_patterns:
                    matches = re.finditer(pattern, search_text, re.IGNORECASE)
                    for match in matches:
                        type_param = match.group(1)
                        constraint = match.group(2)

                        # Find the constraint entity
                        constraint_entity = self._find_entity_by_name(entities, constraint)
                        if constraint_entity:
                            relationships.append(
                                EntityRelation(
                                    source_id=entity.id,
                                    target_id=constraint_entity.id,
                                    type=SemanticRelationType.TYPE_CONSTRAINT.value,
                                    properties={
                                        "type_parameter": type_param,
                                        "constraint_type": constraint,
                                        "constraint_pattern": pattern,
                                    },
                                )
                            )

                            # Store constraint for later analysis
                            self.type_constraints.append(
                                TypeConstraint(
                                    type_parameter=type_param,
                                    constraint_type=constraint,
                                    constraint_kind="bound",
                                    properties={"entity_id": entity.id},
                                )
                            )

        return relationships

    async def _analyze_type_aliases(
        self,
        entities: list[CoreCodeEntity],
        contexts: dict[str, SemanticContext],
    ) -> list[EntityRelation]:
        """Analyze type alias usage and resolution.

        Args:
            entities: List of code entities to analyze
            contexts: Language-specific semantic contexts

        Returns:
            List of EntityRelation objects representing type alias relationships
        """
        relationships = []

        # Find type aliases
        type_aliases = {}
        for entity in entities:
            if entity.type == EntityType.TYPE_ALIAS:
                # Extract the underlying type
                if entity.signature:
                    # Rust type alias
                    rust_alias = re.search(r"type\s+\w+\s*=\s*(.+)", entity.signature)
                    if rust_alias:
                        underlying_type = rust_alias.group(1).strip()
                        type_aliases[entity.name] = underlying_type
                        continue

                    # TypeScript type alias
                    ts_alias = re.search(r"type\s+\w+\s*=\s*(.+)", entity.signature)
                    if ts_alias:
                        underlying_type = ts_alias.group(1).strip()
                        type_aliases[entity.name] = underlying_type
                        continue

                    # C++ typedef/using
                    cpp_alias = re.search(r"(?:typedef|using)\s+\w+\s*=\s*(.+)", entity.signature)
                    if cpp_alias:
                        underlying_type = cpp_alias.group(1).strip()
                        type_aliases[entity.name] = underlying_type

        # Find usage of type aliases
        for entity in entities:
            search_text = f"{entity.signature or ''} {entity.properties.get('content', '')}"

            for alias_name, underlying_type in type_aliases.items():
                if re.search(rf"\b{alias_name}\b", search_text):
                    # Find the alias entity
                    alias_entity = self._find_entity_by_name(entities, alias_name)
                    if alias_entity and alias_entity.id != entity.id:
                        relationships.append(
                            EntityRelation(
                                source_id=entity.id,
                                target_id=alias_entity.id,
                                type=SemanticRelationType.TYPE_ALIAS_RESOLUTION.value,
                                properties={
                                    "underlying_type": underlying_type,
                                    "alias_name": alias_name,
                                },
                            )
                        )

        return relationships

    async def _analyze_polymorphic_usage(
        self, entities: list[CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Analyze polymorphic method calls and virtual dispatch.

        Args:
            entities: List of code entities to analyze

        Returns:
            List of EntityRelation objects representing polymorphic relationships
        """
        relationships = []

        # Find virtual methods and interface methods
        virtual_methods = []
        for entity in entities:
            if entity.type == EntityType.FUNCTION:
                if (
                    entity.properties.get("is_virtual")
                    or entity.properties.get("is_abstract")
                    or entity.properties.get("is_interface_method")
                ):
                    virtual_methods.append(entity)

        # Find calls that might be polymorphic
        for entity in entities:
            if entity.type == EntityType.FUNCTION:
                search_text = f"{entity.signature or ''} {entity.properties.get('content', '')}"

                for virtual_method in virtual_methods:
                    # Look for calls to the virtual method
                    call_patterns = [
                        rf"\b{virtual_method.name}\s*\(",  # Direct call
                        rf"\.{virtual_method.name}\s*\(",  # Method call
                        rf"->{virtual_method.name}\s*\(",  # Pointer method call (C++)
                    ]

                    for pattern in call_patterns:
                        if re.search(pattern, search_text):
                            relationships.append(
                                EntityRelation(
                                    source_id=entity.id,
                                    target_id=virtual_method.id,
                                    type=SemanticRelationType.POLYMORPHIC_CALL.value,
                                    properties={
                                        "method_name": virtual_method.name,
                                        "call_type": "virtual_dispatch",
                                        "call_pattern": pattern,
                                    },
                                )
                            )
                            break

        return relationships

    async def _analyze_type_conversions(
        self, entities: list[CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Analyze type casting and conversion patterns.

        Args:
            entities: List of code entities to analyze

        Returns:
            List of EntityRelation objects representing type conversions
        """
        relationships = []

        conversion_patterns = [
            # Rust conversions
            (r"\.into\(\)", "into_conversion"),
            (r"(\w+)::from\(", "from_conversion"),
            (r"as\s+(\w+)", "as_conversion"),
            # C++ conversions
            (r"static_cast<(\w+)>", "static_cast"),
            (r"dynamic_cast<(\w+)>", "dynamic_cast"),
            (r"reinterpret_cast<(\w+)>", "reinterpret_cast"),
            # C-style cast
            (r"\((\w+)\)\s*\w+", "c_style_cast"),
            # Java/C# conversions
            (r"(\w+)\.valueOf\(", "value_of_conversion"),
            (r"Integer\.parseInt\(", "parse_conversion"),
            # TypeScript/JavaScript
            (r"Number\(", "number_conversion"),
            (r"String\(", "string_conversion"),
        ]

        for entity in entities:
            search_text = f"{entity.signature or ''} {entity.properties.get('content', '')}"

            for pattern, conversion_type in conversion_patterns:
                matches = re.finditer(pattern, search_text, re.IGNORECASE)
                for match in matches:
                    target_type = match.group(1) if match.groups() else "unknown"

                    # Find target type entity
                    target_entity = self._find_entity_by_name(entities, target_type)
                    if target_entity:
                        relationships.append(
                            EntityRelation(
                                source_id=entity.id,
                                target_id=target_entity.id,
                                type=SemanticRelationType.TYPE_CONVERSION.value,
                                properties={
                                    "conversion_type": conversion_type,
                                    "target_type": target_type,
                                    "pattern": pattern,
                                },
                            )
                        )

        return relationships

    async def _analyze_interface_bindings(
        self, entities: list[CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Analyze interface implementation and binding patterns.

        Args:
            entities: List of code entities to analyze

        Returns:
            List of EntityRelation objects representing interface bindings
        """
        relationships = []

        # Find interface definitions
        interfaces = [e for e in entities if e.type == EntityType.INTERFACE]

        # Find implementations
        for entity in entities:
            if entity.type == EntityType.CLASS:
                search_text = f"{entity.signature or ''}"

                # Check for interface implementations
                for interface in interfaces:
                    implementation_patterns = [
                        rf"implements\s+[^{{]*\b{interface.name}\b",  # Java/C#
                        rf":\s*[^{{]*\b{interface.name}\b",  # C++/Swift
                        rf"impl\s+{interface.name}\s+for",  # Rust trait impl
                    ]

                    for pattern in implementation_patterns:
                        if re.search(pattern, search_text, re.IGNORECASE):
                            relationships.append(
                                EntityRelation(
                                    source_id=entity.id,
                                    target_id=interface.id,
                                    type=SemanticRelationType.INTERFACE_BINDING.value,
                                    properties={
                                        "interface_name": interface.name,
                                        "implementation_pattern": pattern,
                                        "binding_type": "implementation",
                                    },
                                )
                            )
                            break

        return relationships

    def _find_entity_by_name(
        self, entities: list[CoreCodeEntity], name: str
    ) -> CoreCodeEntity | None:
        """Find an entity by name.

        Args:
            entities: List of entities to search
            name: Name to search for

        Returns:
            Found entity or None
        """
        for entity in entities:
            if entity.name == name:
                return entity
        return None

    def get_type_hierarchy(self, type_name: str) -> list[str]:
        """Get the inheritance hierarchy for a type.

        Args:
            type_name: Name of the type

        Returns:
            List of type names in the hierarchy from root to the given type
        """
        hierarchy = []
        current_type = type_name
        visited = set()

        while current_type and current_type not in visited:
            visited.add(current_type)
            hierarchy.append(current_type)

            type_info = self.type_registry.get(current_type)
            if type_info and type_info.base_types:
                # Take the first base type (primary inheritance)
                current_type = type_info.base_types[0]
            else:
                break

        return list(reversed(hierarchy))

    def infer_type_from_usage(
        self, entity: CoreCodeEntity, entities: list[CoreCodeEntity]
    ) -> str | None:
        """Infer the type of an entity from its usage patterns.

        Args:
            entity: Entity to infer type for
            entities: All entities for context

        Returns:
            Inferred type name or None
        """
        if entity.type != EntityType.VARIABLE:
            return None

        content = entity.properties.get("content", "")

        # Look for assignment patterns
        assignment_patterns = [
            (r"=\s*new\s+(\w+)", "constructor"),
            (r"=\s*(\w+)\.create", "factory_method"),
            (r"=\s*(\w+)\(", "function_call"),
            (r":\s*(\w+)\s*=", "type_annotation"),
        ]

        for pattern, inference_type in assignment_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                potential_type = match.group(1)

                # Check if this is a known type
                if potential_type in self.type_registry:
                    return potential_type

                # Check if there's an entity with this name
                type_entity = self._find_entity_by_name(entities, potential_type)
                if type_entity and type_entity.type in [
                    EntityType.CLASS,
                    EntityType.INTERFACE,
                ]:
                    return potential_type

        return None
