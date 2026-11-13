"""
Symbol Table Management Module

This module provides functionality for managing symbol tables and tracking
symbol relationships across different code entities and scopes.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    EntityType,
)

logger = logging.getLogger(__name__)


class ScopeType(StrEnum):
    """Types of symbol scopes."""

    GLOBAL = "global"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    BLOCK = "block"
    NAMESPACE = "namespace"


class SymbolType(StrEnum):
    """Types of symbols in the symbol table."""

    VARIABLE = "variable"
    FUNCTION = "function"
    CLASS = "class"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    CONSTANT = "constant"
    ENUM = "enum"
    NAMESPACE = "namespace"
    MODULE = "module"


@dataclass
class Symbol:
    """Represents a symbol in the symbol table."""

    name: str
    symbol_type: SymbolType
    entity_id: str
    scope_id: str
    file_path: str
    line_number: int | None = None

    # Type information
    declared_type: str | None = None
    inferred_type: str | None = None

    # Metadata
    is_exported: bool = False
    is_imported: bool = False
    access_modifier: str | None = None  # public, private, protected

    # Additional properties
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scope:
    """Represents a scope in the symbol table."""

    scope_id: str
    scope_type: ScopeType
    name: str
    parent_scope_id: str | None = None
    file_path: str = ""

    # Symbols defined in this scope
    symbols: dict[str, Symbol] = field(default_factory=dict)

    # Child scopes
    child_scopes: set[str] = field(default_factory=set)

    # Imports and exports
    imports: set[str] = field(default_factory=set)
    exports: set[str] = field(default_factory=set)


class SymbolTable:
    """Manages symbols and their scopes across the codebase."""

    def __init__(self):
        self.scopes: dict[str, Scope] = {}
        self.symbols: dict[str, Symbol] = {}
        self.file_scopes: dict[str, set[str]] = defaultdict(set)
        self.symbol_references: dict[str, set[str]] = defaultdict(set)

    def create_scope(
        self,
        scope_id: str,
        scope_type: ScopeType,
        name: str,
        file_path: str,
        parent_scope_id: str | None = None,
    ) -> Scope:
        """Create a new scope in the symbol table.

        Args:
            scope_id: Unique identifier for the scope
            scope_type: Type of the scope
            name: Human-readable name of the scope
            file_path: Path to the file containing this scope
            parent_scope_id: ID of the parent scope if any

        Returns:
            The created Scope object
        """
        scope = Scope(
            scope_id=scope_id,
            scope_type=scope_type,
            name=name,
            parent_scope_id=parent_scope_id,
            file_path=file_path,
        )

        self.scopes[scope_id] = scope
        self.file_scopes[file_path].add(scope_id)

        # Add to parent's child scopes
        if parent_scope_id and parent_scope_id in self.scopes:
            self.scopes[parent_scope_id].child_scopes.add(scope_id)

        return scope

    def add_symbol(
        self,
        name: str,
        symbol_type: SymbolType,
        entity_id: str,
        scope_id: str,
        file_path: str,
        **kwargs,
    ) -> Symbol:
        """Add a symbol to the symbol table.

        Args:
            name: Name of the symbol
            symbol_type: Type of the symbol
            entity_id: ID of the associated entity
            scope_id: ID of the scope containing this symbol
            file_path: Path to the file containing this symbol
            **kwargs: Additional properties for the symbol

        Returns:
            The created Symbol object
        """
        symbol = Symbol(
            name=name,
            symbol_type=symbol_type,
            entity_id=entity_id,
            scope_id=scope_id,
            file_path=file_path,
            **kwargs,
        )

        symbol_id = f"{scope_id}::{name}"
        self.symbols[symbol_id] = symbol

        # Add to scope
        if scope_id in self.scopes:
            self.scopes[scope_id].symbols[name] = symbol

        return symbol

    def resolve_symbol(self, name: str, scope_id: str) -> Symbol | None:
        """Resolve a symbol name in a given scope.

        Args:
            name: Name of the symbol to resolve
            scope_id: ID of the scope to search from

        Returns:
            The resolved Symbol object or None if not found
        """
        # Check current scope
        if scope_id in self.scopes:
            scope = self.scopes[scope_id]
            if name in scope.symbols:
                return scope.symbols[name]

        # Check parent scopes
        current_scope_id = scope_id
        while current_scope_id:
            if current_scope_id in self.scopes:
                scope = self.scopes[current_scope_id]
                if name in scope.symbols:
                    return scope.symbols[name]
                current_scope_id = scope.parent_scope_id
            else:
                break

        return None

    def find_symbols_by_type(self, symbol_type: SymbolType) -> list[Symbol]:
        """Find all symbols of a given type.

        Args:
            symbol_type: Type of symbols to find

        Returns:
            List of matching Symbol objects
        """
        return [symbol for symbol in self.symbols.values() if symbol.symbol_type == symbol_type]

    def find_symbols_in_file(self, file_path: str) -> list[Symbol]:
        """Find all symbols defined in a specific file.

        Args:
            file_path: Path to the file

        Returns:
            List of Symbol objects defined in the file
        """
        symbols = []
        for scope_id in self.file_scopes.get(file_path, set()):
            if scope_id in self.scopes:
                symbols.extend(self.scopes[scope_id].symbols.values())
        return symbols

    def get_scope_hierarchy(self, scope_id: str) -> list[str]:
        """Get the hierarchy of scopes from root to the given scope.

        Args:
            scope_id: ID of the scope

        Returns:
            List of scope IDs from root to the given scope
        """
        hierarchy = []
        current_scope_id = scope_id

        while current_scope_id and current_scope_id in self.scopes:
            hierarchy.insert(0, current_scope_id)
            current_scope_id = self.scopes[current_scope_id].parent_scope_id

        return hierarchy

    def add_symbol_reference(self, symbol_id: str, referencing_entity_id: str):
        """Add a reference to a symbol from another entity.

        Args:
            symbol_id: ID of the symbol being referenced
            referencing_entity_id: ID of the entity making the reference
        """
        self.symbol_references[symbol_id].add(referencing_entity_id)

    def get_symbol_references(self, symbol_id: str) -> set[str]:
        """Get all entities that reference a symbol.

        Args:
            symbol_id: ID of the symbol

        Returns:
            Set of entity IDs that reference the symbol
        """
        return self.symbol_references.get(symbol_id, set())

    def get_exported_symbols(self, file_path: str) -> list[Symbol]:
        """Get all exported symbols from a file.

        Args:
            file_path: Path to the file

        Returns:
            List of exported Symbol objects
        """
        return [symbol for symbol in self.find_symbols_in_file(file_path) if symbol.is_exported]

    def get_imported_symbols(self, file_path: str) -> list[Symbol]:
        """Get all imported symbols in a file.

        Args:
            file_path: Path to the file

        Returns:
            List of imported Symbol objects
        """
        return [symbol for symbol in self.find_symbols_in_file(file_path) if symbol.is_imported]


class SymbolTableBuilder:
    """Builds symbol tables from code entities."""

    def __init__(self):
        self.symbol_table = SymbolTable()

    def build_from_entities(self, entities: list[CoreCodeEntity]) -> SymbolTable:
        """Build a symbol table from a list of code entities.

        Args:
            entities: List of code entities to process

        Returns:
            The built SymbolTable object
        """
        # Group entities by file
        entities_by_file = defaultdict(list)
        for entity in entities:
            entities_by_file[entity.file_path].append(entity)

        # Process each file
        for file_path, file_entities in entities_by_file.items():
            self._process_file_entities(file_path, file_entities)

        return self.symbol_table

    def _process_file_entities(self, file_path: str, entities: list[CoreCodeEntity]):
        """Process entities from a single file.

        Args:
            file_path: Path to the file
            entities: List of entities in the file
        """
        # Create global scope for the file
        global_scope_id = f"global:{file_path}"
        self.symbol_table.create_scope(
            scope_id=global_scope_id,
            scope_type=ScopeType.GLOBAL,
            name=Path(file_path).name,
            file_path=file_path,
        )

        # Sort entities to process in dependency order
        sorted_entities = self._sort_entities_by_dependency(entities)

        for entity in sorted_entities:
            self._process_entity(entity, global_scope_id)

    def _sort_entities_by_dependency(self, entities: list[CoreCodeEntity]) -> list[CoreCodeEntity]:
        """Sort entities by dependency order (classes before functions, etc.).

        Args:
            entities: List of entities to sort

        Returns:
            Sorted list of entities
        """
        # Define processing order
        type_order = {
            EntityType.NAMESPACE: 0,
            EntityType.MODULE: 1,
            EntityType.CONSTANT: 2,
            EntityType.ENUM: 3,
            EntityType.TYPE_ALIAS: 4,
            EntityType.INTERFACE: 5,
            EntityType.CLASS: 6,
            EntityType.FUNCTION: 7,
            EntityType.VARIABLE: 8,
        }

        return sorted(entities, key=lambda e: type_order.get(e.type, 999))

    def _process_entity(self, entity: CoreCodeEntity, parent_scope_id: str):
        """Process a single entity and add it to the symbol table.

        Args:
            entity: Entity to process
            parent_scope_id: ID of the parent scope
        """
        # Map entity types to symbol types
        entity_to_symbol_type = {
            EntityType.VARIABLE: SymbolType.VARIABLE,
            EntityType.FUNCTION: SymbolType.FUNCTION,
            EntityType.CLASS: SymbolType.CLASS,
            EntityType.INTERFACE: SymbolType.INTERFACE,
            EntityType.TYPE_ALIAS: SymbolType.TYPE_ALIAS,
            EntityType.CONSTANT: SymbolType.CONSTANT,
            EntityType.ENUM: SymbolType.ENUM,
            EntityType.NAMESPACE: SymbolType.NAMESPACE,
            EntityType.MODULE: SymbolType.MODULE,
        }

        symbol_type = entity_to_symbol_type.get(entity.type, SymbolType.VARIABLE)

        # Create scope for entities that define scopes
        scope_defining_types = {
            EntityType.CLASS,
            EntityType.FUNCTION,
            EntityType.NAMESPACE,
        }
        current_scope_id = parent_scope_id

        if entity.type in scope_defining_types:
            scope_type_map = {
                EntityType.CLASS: ScopeType.CLASS,
                EntityType.FUNCTION: ScopeType.FUNCTION,
                EntityType.NAMESPACE: ScopeType.NAMESPACE,
            }

            scope_id = f"{scope_type_map[entity.type]}:{entity.id}"
            self.symbol_table.create_scope(
                scope_id=scope_id,
                scope_type=scope_type_map[entity.type],
                name=entity.name,
                file_path=entity.file_path,
                parent_scope_id=parent_scope_id,
            )
            current_scope_id = scope_id

        # Add symbol to symbol table
        self.symbol_table.add_symbol(
            name=entity.name,
            symbol_type=symbol_type,
            entity_id=entity.id,
            scope_id=current_scope_id,
            file_path=entity.file_path,
            declared_type=entity.properties.get("type"),
            access_modifier=entity.properties.get("access_modifier"),
            is_exported=entity.properties.get("is_exported", False),
            is_imported=entity.properties.get("is_imported", False),
            properties=entity.properties,
        )
