"""Module for extracting relationships from code structures."""

from __future__ import annotations

import ast
import logging
from typing import Any

from models.core_models import CoreCodeEntity, EntityRelation, RelationType

from ..utils.ast_utils import ASTUtils

"""Relationship Extractor Module for Code Graph Analysis.

This module provides comprehensive functionality for extracting relationships between code entities
from Abstract Syntax Trees (AST). It analyzes various types of relationships including
inheritance hierarchies, function call dependencies, import dependencies, variable usage patterns,
and composition relationships.

The relationship extraction process is fundamental to building a comprehensive code graph that
represents the interconnections and dependencies within a codebase. This module supports
the analysis of Python code by walking through AST nodes and identifying patterns that
indicate relationships between different code entities such as classes, functions, methods,
variables, and modules.

Key relationship types extracted:
    - INHERITANCE: Class inheritance relationships (A inherits from B)
    - CALLS: Function call relationships (A calls B)
    - IMPORT: Module import dependencies (A imports B)
    - USAGE: Variable and attribute usage patterns (A uses B)
    - COMPOSITION: Class composition relationships (A contains instance of B)

The extracted relationships are represented as EntityRelation objects that include
metadata such as line numbers, relationship context, and type-specific information.
This data is used to build dependency graphs, analyze code complexity, detect patterns,
and support various code analysis features.

Example:
    ```python
    extractor = RelationshipExtractor()
    tree = ast.parse(source_code)
    entities = [...]  # List of CoreCodeEntity objects
    relationships = await extractor.extract_relationships(tree, entities)
    ```

Typical use cases:
    - Dependency analysis and visualization
    - Code complexity metrics calculation
    - Dead code detection
    - Impact analysis for code changes
    - Architecture validation and pattern detection
"""

logger = logging.getLogger(__name__)


class RelationshipExtractor:
    """Extracts relationships between code entities from Abstract Syntax Trees.

    This class provides comprehensive methods to analyze Python code and extract various types of
    relationships between code entities. It uses AST traversal techniques to identify patterns
    that indicate relationships such as inheritance hierarchies, function call dependencies,
    import relationships, variable usage patterns, and composition relationships.

    The extractor operates by walking through AST nodes and applying pattern matching algorithms
    to identify different types of relationships. Each relationship type has specialized extraction
    logic that captures the specific patterns and metadata relevant to that relationship.

    Key capabilities:
        - Inheritance relationship detection through base class analysis
        - Function call relationship mapping via call node analysis
        - Import dependency tracking through import statement parsing
        - Variable usage pattern detection via name reference analysis
        - Composition relationship identification through attribute assignment analysis

    The extracted relationships include rich metadata such as line numbers, context information,
    and relationship-specific details that enable sophisticated code analysis and visualization.

    Attributes:
        ast_utils (ASTUtils): Utility class for AST operations and ID generation.

    Example:
        ```python
        extractor = RelationshipExtractor()

        # Parse source code into AST
        tree = ast.parse(source_code)

        # Extract entities first (classes, functions, etc.)
        entities = entity_extractor.extract_entities(tree)

        # Extract relationships between entities
        relationships = await extractor.extract_relationships(tree, entities)

        # Access specific relationship types
        inheritance_rels = await extractor._extract_inheritance_relationships(tree, entity_lookup)
        composition_rels = await extractor.extract_composition_relationships(tree, entity_lookup)
        ```
    """

    def __init__(self) -> None:
        """Initialize the RelationshipExtractor with required utilities.

        Sets up the RelationshipExtractor with an ASTUtils instance that provides
        utility methods for AST operations, attribute name extraction, and
        relationship ID generation.

        The AST utilities are essential for:
            - Generating unique relationship identifiers
            - Extracting attribute names from complex AST nodes
            - Performing common AST traversal operations
        """
        self.ast_utils = ASTUtils()

    async def extract_relationships(
        self, tree: ast.AST, entities: list[CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Extract all relationships from Python AST.

        This is the main entry point for comprehensive relationship extraction. It coordinates
        the extraction of different types of relationships by delegating to specialized
        extraction methods and combines them into a single comprehensive list.

        The method creates an entity lookup dictionary for efficient access during relationship
        extraction, then processes the AST to identify inheritance, function call, import,
        and usage relationships. Each relationship type is extracted using specialized
        algorithms optimized for that particular pattern.

        The extraction process is asynchronous to support large codebases and complex
        analysis operations without blocking the main thread.

        Args:
            tree (ast.AST): The Abstract Syntax Tree of the Python code to analyze.
                Must be a valid AST generated from parseable Python source code.
            entities (List[CoreCodeEntity]): List of code entities that have been previously
                extracted from the same AST. These entities serve as the nodes between
                which relationships will be established. The list should include all
                relevant entities like classes, functions, methods, variables, and modules.

        Returns:
            List[EntityRelation]: A comprehensive list of EntityRelation objects representing
                all the relationships found in the code. Each relationship includes:
                - Unique relationship ID
                - Source and target entity IDs
                - Relationship type (inheritance, calls, import, usage)
                - Rich metadata including line numbers and context information

        Note:
            The relationships are only established between entities that exist in the
            provided entities list. External dependencies or undefined references
            will not generate relationships.

        Example:
            ```python
            # Parse source code
            tree = ast.parse("class Child(Parent): pass")

            # Extract entities first
            entities = [parent_entity, child_entity]

            # Extract relationships
            relationships = await extractor.extract_relationships(tree, entities)
            # Returns [EntityRelation(child -> parent, type=INHERITANCE)]
            ```
        """
        relationships: list[Any] = []
        # Create entity lookup for quick access
        entity_lookup = {entity.name: entity for entity in entities}

        # Extract different types of relationships
        inheritance_relations = await self._extract_inheritance_relationships(tree, entity_lookup)
        call_relations = await self._extract_function_call_relationships(tree, entity_lookup)
        import_relations = await self._extract_import_relationships(tree, entity_lookup)
        usage_relations = await self._extract_usage_relationships(tree, entity_lookup)

        relationships.extend(inheritance_relations)
        relationships.extend(call_relations)
        relationships.extend(import_relations)
        relationships.extend(usage_relations)

        return relationships

    async def _extract_inheritance_relationships(
        self, tree: ast.AST, entity_lookup: dict[str, CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Extract class inheritance relationships from the AST.

        Identifies class definitions and their base classes to establish inheritance
        relationships. This method walks through all class definitions in the AST
        and analyzes their base class declarations to create INHERITANCE type
        relationships.

        The algorithm handles various inheritance patterns:
            - Simple inheritance: class Child(Parent)
            - Multiple inheritance: class Child(Parent1, Parent2)
            - Attribute-based inheritance: class Child(module.Parent)
            - Complex base class expressions

        For each inheritance relationship found, the method creates an EntityRelation
        with the child class as the source and the parent class as the target.

        Args:
            tree (ast.AST): The Abstract Syntax Tree of the Python code to analyze.
                Should contain class definitions with base class specifications.
            entity_lookup (Dict[str, CoreCodeEntity]): Dictionary mapping entity names
                to CoreCodeEntity objects for efficient lookup during relationship creation.
                Both child and parent classes must exist in this lookup for a
                relationship to be created.

        Returns:
            List[EntityRelation]: A list of EntityRelation objects representing
                inheritance relationships. Each relationship includes:
                - source_id: The ID of the child class
                - target_id: The ID of the parent class
                - relation_type: RelationType.INHERITANCE
                - metadata: Contains source/target names and line number

        Note:
            Only inheritance relationships between entities that exist in the
            entity_lookup will be created. External or undefined base classes
            will be silently ignored.

        Example:
            For code "class Dog(Animal): pass", this method will create:
            EntityRelation(source_id=dog_id, target_id=animal_id, type=INHERITANCE)
        """
        relationships: list[Any] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_entity = entity_lookup.get(node.name)
                if not class_entity:
                    continue

                for base in node.bases:
                    base_name = None
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = self.ast_utils.get_attribute_name(base)

                    if base_name:
                        # Look for base class entity
                        base_entity = entity_lookup.get(base_name)
                        if base_entity:
                            rel_id = self.ast_utils.generate_relationship_id(
                                class_entity.id, base_entity.id, "inheritance"
                            )

                            relationships.append(
                                EntityRelation(
                                    id=rel_id,
                                    source_id=class_entity.id,
                                    target_id=base_entity.id,
                                    type=RelationType.INHERITANCE,
                                    metadata={
                                        "source_name": class_entity.name,
                                        "target_name": base_entity.name,
                                        "line_number": node.lineno,
                                    },
                                )
                            )

        return relationships

    async def _extract_function_call_relationships(
        self, tree: ast.AST, entity_lookup: dict[str, CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Extract function call relationships from the AST.

        Identifies function calls within functions and methods to establish caller-callee
        relationships. This method uses a specialized AST visitor that tracks the current
        function context and identifies all function calls within that context.

        The algorithm handles various call patterns:
            - Direct function calls: func()
            - Method calls: obj.method()
            - Nested calls: func1(func2())
            - Attribute-based calls: module.function()
            - Async function calls within async functions

        The visitor maintains context awareness by tracking the current function being
        analyzed and creates relationships from that function to all functions it calls.
        This enables the construction of call graphs for dependency analysis.

        Args:
            tree (ast.AST): The Abstract Syntax Tree of the Python code to analyze.
                Should contain function definitions with function call statements.
            entity_lookup (Dict[str, CoreCodeEntity]): Dictionary mapping entity names
                to CoreCodeEntity objects. Both calling and called functions must exist
                in this lookup for a relationship to be created.

        Returns:
            List[EntityRelation]: A list of EntityRelation objects representing
                function call relationships. Each relationship includes:
                - source_id: The ID of the calling function
                - target_id: The ID of the called function
                - relation_type: RelationType.CALLS
                - metadata: Contains function names and call location

        Note:
            Only relationships between functions that exist in the entity_lookup
            will be created. Calls to external libraries or undefined functions
            will be ignored. Self-recursive calls are included.

        Example:
            For code "def process(): validate(); save()", this method will create:
            - EntityRelation(process -> validate, type=CALLS)
            - EntityRelation(process -> save, type=CALLS)
        """
        relationships: list[Any] = []

        class RelationshipCallVisitor(ast.NodeVisitor):
            """AST visitor for extracting function call relationships.

            This specialized visitor traverses the AST and identifies function calls within
            function definitions. It maintains context awareness by tracking the current
            function being analyzed and creates relationships between the current function
            and any functions it calls.

            The visitor handles both regular and async function definitions and processes
            all types of function calls including direct calls, method calls, and
            attribute-based calls.

            Attributes:
                extractor (RelationshipExtractor): Reference to the parent extractor instance
                entity_lookup (Dict[str, CoreCodeEntity]): Entity lookup dictionary for relationship creation
                current_function (CoreCodeEntity): The function currently being analyzed for calls
                relationships (List[EntityRelation]): Accumulated list of call relationships found
            """

            def __init__(self, extractor, entity_lookup):
                """Initialize the relationship call visitor.

                Sets up the visitor with references to the parent extractor and entity lookup,
                and initializes the context tracking variables.

                Args:
                    extractor (RelationshipExtractor): The parent RelationshipExtractor instance
                        that provides access to AST utilities and relationship creation methods.
                    entity_lookup (Dict[str, CoreCodeEntity]): Dictionary mapping entity names to
                        CoreCodeEntity objects for efficient lookup during relationship creation.
                """
                self.extractor = extractor
                self.entity_lookup = entity_lookup
                self.current_function = None
                self.relationships = []

            def visit_FunctionDef(self, node):
                """Visit function definitions and track the current function context.

                This method is called whenever a function definition is encountered in the AST.
                It updates the current function context to enable tracking of function calls
                within this function scope. The previous context is preserved and restored
                after visiting the function body to handle nested function definitions correctly.

                Args:
                    node (ast.FunctionDef): The AST node representing a function definition.
                        Contains the function name, parameters, body, and decorators.
                """
                old_function = self.current_function
                self.current_function = self.entity_lookup.get(node.name)
                self.generic_visit(node)
                self.current_function = old_function

            def visit_AsyncFunctionDef(self, node):
                """Visit async function definitions and track the current function context.

                Handles async function definitions the same way as regular function definitions
                since the call relationship extraction logic is identical for both types.

                Args:
                    node (ast.AsyncFunctionDef): The AST node representing an async function
                        definition. Contains the same structure as regular function definitions.
                """
                self.visit_FunctionDef(node)  # Handle same as regular function

            def visit_Call(self, node):
                """Visit function call nodes and extract call relationships.

                This method is called for every function call encountered in the AST.
                If there is a current function context, it attempts to identify the called
                function and create a CALLS relationship between the current function
                and the called function.

                The method handles various call patterns by delegating name extraction
                to a helper method that can handle direct calls, method calls, and
                attribute-based calls.

                Args:
                    node (ast.Call): The AST node representing a function call.
                        Contains the function being called (func), arguments (args),
                        and keyword arguments (keywords).
                """
                if not self.current_function:
                    return

                call_name = self._get_call_name(node)
                if not call_name:
                    return

                # Check if the called function exists in our entity lookup
                called_entity = self.entity_lookup.get(call_name)
                if called_entity:
                    # Create a relationship from current function to called function
                    rel_id = self.extractor.ast_utils.generate_relationship_id(
                        self.current_function.id, called_entity.id, "calls"
                    )

                    self.relationships.append(
                        EntityRelation(
                            id=rel_id,
                            source_id=self.current_function.id,
                            target_id=called_entity.id,
                            type=RelationType.CALLS,
                            metadata={
                                "source_name": self.current_function.name,
                                "target_name": called_entity.name,
                                "line_number": node.lineno,
                            },
                        )
                    )

                self.generic_visit(node)

            def _get_call_name(self, node):
                """Extract the name of the called function from a call node.

                This helper method analyzes the function part of a call node to extract
                the name of the function being called. It handles different call patterns:
                - Direct calls: func() -> "func"
                - Method calls: obj.method() -> "method"
                - Attribute calls: module.func() -> "module.func"
                - Complex attribute chains: a.b.c.func() -> "a.b.c.func"

                Args:
                    node (ast.Call): The AST node representing a function call.
                        The func attribute contains the callable being invoked.

                Returns:
                    str or None: The name of the called function if it can be determined,
                        or None if the call pattern is too complex to extract a simple name.

                Example:
                    For "print('hello')" returns "print"
                    For "obj.save()" returns "save"
                    For "utils.helpers.format()" returns "utils.helpers.format"
                """
                if isinstance(node.func, ast.Name):
                    return node.func.id
                elif isinstance(node.func, ast.Attribute):
                    return self.extractor.ast_utils.get_attribute_name(node.func)
                return None

        visitor = RelationshipCallVisitor(self, entity_lookup)
        visitor.visit(tree)
        relationships.extend(visitor.relationships)

        return relationships

    async def _extract_import_relationships(
        self, tree: ast.AST, entity_lookup: dict[str, CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Extract import relationships from the AST.

        Identifies import statements and establishes relationships between the importing
        module and the imported modules or entities. This method processes both regular
        import statements (import module) and from-import statements (from module import item).

        The algorithm handles various import patterns:
            - Simple imports: import os
            - Module imports: import package.module
            - From imports: from package import function
            - Multiple imports: import os, sys
            - Aliased imports: import numpy as np
            - Relative imports: from .module import function

        For each import found, the method creates an IMPORT relationship from the current
        module to the imported entity, enabling dependency tracking and analysis.

        Args:
            tree (ast.AST): The Abstract Syntax Tree of the Python code to analyze.
                Should contain import statements (ast.Import or ast.ImportFrom nodes).
            entity_lookup (Dict[str, CoreCodeEntity]): Dictionary mapping entity names to
                CoreCodeEntity objects. The importing module and imported entities must
                exist in this lookup for relationships to be created.

        Returns:
            List[EntityRelation]: A list of EntityRelation objects representing
                import relationships. Each relationship includes:
                - source_id: The ID of the importing module
                - target_id: The ID of the imported entity
                - relation_type: RelationType.IMPORT
                - metadata: Contains import details and type information

        Note:
            Only creates relationships for imports where both the importing module
            and the imported entity exist in the entity_lookup. External dependencies
            that haven't been processed will be ignored.

        Example:
            For "import json; from os import path", this method will create:
            - EntityRelation(current_module -> json, type=IMPORT)
            - EntityRelation(current_module -> path, type=IMPORT)
        """
        relationships: list[EntityRelation] = []

        # Find module entity
        module_entities = [e for e in entity_lookup.values() if e.type.value == "module"]
        if not module_entities:
            return relationships

        # Should only be one module per file
        module_entity = module_entities[0]

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Create relationships from module to imported entities
                import_names: list[Any] = []
                if isinstance(node, ast.Import):
                    import_names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    import_names = [
                        (f"{module_name}.{alias.name}" if module_name else alias.name)
                        for alias in node.names
                    ]

                for import_name in import_names:
                    # Look for import entity
                    import_entity = None
                    for entity in entity_lookup.values():
                        if entity.type.value == "import" and (
                            entity.name == import_name
                            or entity.metadata.get("full_name") == import_name
                        ):
                            import_entity = entity
                            break

                    if import_entity:
                        rel_id = self.ast_utils.generate_relationship_id(
                            module_entity.id, import_entity.id, "import"
                        )

                        relationships.append(
                            EntityRelation(
                                id=rel_id,
                                source_id=module_entity.id,
                                target_id=import_entity.id,
                                type=RelationType.IMPORT,
                                metadata={
                                    "source_name": module_entity.name,
                                    "target_name": import_entity.name,
                                    "line_number": node.lineno,
                                    "import_type": (
                                        "import" if isinstance(node, ast.Import) else "from_import"
                                    ),
                                },
                            )
                        )

        return relationships

    async def _extract_usage_relationships(
        self, tree: ast.AST, entity_lookup: dict[str, CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Extract variable and attribute usage relationships from the AST.

        Identifies where variables, attributes, and other entities are referenced within
        functions and classes to establish usage relationships. This method uses a
        specialized visitor that tracks the current context and identifies name references
        that correspond to known entities.

        The algorithm focuses on Load contexts (variable reads) rather than Store contexts
        (variable assignments) to identify actual usage rather than definitions. It handles
        usage patterns within both function and class contexts.

        The usage relationships are essential for:
            - Dead code detection (unused variables/functions)
            - Dependency analysis (what uses what)
            - Impact analysis (what might be affected by changes)
            - Code complexity metrics

        Args:
            tree (ast.AST): The Abstract Syntax Tree of the Python code to analyze.
                Should contain functions and classes with variable references.
            entity_lookup (Dict[str, CoreCodeEntity]): Dictionary mapping entity names to
                CoreCodeEntity objects. Both the using context and used entities must
                exist in this lookup for relationships to be created.

        Returns:
            List[EntityRelation]: A list of EntityRelation objects representing
                usage relationships. Each relationship includes:
                - source_id: The ID of the function/class that uses the entity
                - target_id: The ID of the entity being used
                - relation_type: RelationType.USAGE
                - metadata: Contains usage context and location information

        Note:
            Self-references (entity using itself) are automatically filtered out.
            Only creates relationships for entities that exist in the entity_lookup.
            Focuses on Load contexts to avoid creating relationships for assignments.

        Example:
            For code "def process(): return format(data)", this method will create:
            - EntityRelation(process -> format, type=USAGE)
            - EntityRelation(process -> data, type=USAGE)
        """
        relationships: list[Any] = []

        class UsageVisitor(ast.NodeVisitor):
            """AST visitor for extracting variable and attribute usage relationships.

            This specialized visitor traverses the AST and identifies variable and entity
            references within function and class contexts. It maintains awareness of the
            current scope and creates usage relationships between the current context
            and any entities that are referenced.

            The visitor specifically looks for Name nodes in Load contexts to identify
            actual usage rather than variable assignments or definitions.

            Attributes:
                extractor (RelationshipExtractor): Reference to the parent extractor instance
                entity_lookup (Dict[str, CoreCodeEntity]): Entity lookup dictionary for relationship creation
                current_context (CoreCodeEntity): The current function or class context being analyzed
                relationships (List[EntityRelation]): Accumulated list of usage relationships found
            """

            def __init__(self, extractor, entity_lookup):
                """Initialize the usage visitor.

                Sets up the visitor with references to the parent extractor and entity lookup,
                and initializes the context tracking variables.

                Args:
                    extractor (RelationshipExtractor): The parent RelationshipExtractor instance
                        that provides access to AST utilities and relationship creation methods.
                    entity_lookup (Dict[str, CoreCodeEntity]): Dictionary mapping entity names to
                        CoreCodeEntity objects for efficient lookup during relationship creation.
                """
                self.extractor = extractor
                self.entity_lookup = entity_lookup
                self.current_context = None  # Current function or class
                self.relationships = []

            def visit_FunctionDef(self, node):
                """Visit function definitions and track the current context.

                Updates the current context to the function being visited, enabling
                tracking of variable usage within that function scope. The previous
                context is preserved and restored to handle nested functions correctly.

                Args:
                    node (ast.FunctionDef): The AST node representing a function definition.
                        Contains the function name, parameters, body, and decorators.
                """
                old_context = self.current_context
                self.current_context = self.entity_lookup.get(node.name)
                self.generic_visit(node)
                self.current_context = old_context

            def visit_AsyncFunctionDef(self, node):
                """Visit async function definitions and track the current context.

                Handles async function definitions the same way as regular function
                definitions since usage tracking logic is identical for both types.

                Args:
                    node (ast.AsyncFunctionDef): The AST node representing an async function
                        definition. Contains the same structure as regular function definitions.
                """
                self.visit_FunctionDef(node)

            def visit_ClassDef(self, node):
                """Visit class definitions and track the current context.

                Updates the current context to the class being visited, enabling
                tracking of variable usage within that class scope. This captures
                usage relationships at the class level.

                Args:
                    node (ast.ClassDef): The AST node representing a class definition.
                        Contains the class name, base classes, body, and decorators.
                """
                old_context = self.current_context
                self.current_context = self.entity_lookup.get(node.name)
                self.generic_visit(node)
                self.current_context = old_context

            def visit_Name(self, node):
                """Visit name nodes and extract usage relationships.

                This method is called for every name reference in the AST. It identifies
                variable and entity usage by checking for Load contexts (reading variables)
                and creates relationships between the current context and any referenced
                entities that exist in the entity lookup.

                The method automatically filters out self-references to avoid creating
                relationships where an entity references itself.

                Args:
                    node (ast.Name): The AST node representing a name reference.
                        Contains the variable name (id) and context (Load, Store, Del).
                """
                if not self.current_context:
                    return

                # Check if this name refers to a known entity
                if isinstance(node.ctx, ast.Load) and node.id in self.entity_lookup:
                    used_entity = self.entity_lookup[node.id]

                    # Don't create self-references
                    if used_entity.id == self.current_context.id:
                        return

                    rel_id = self.extractor.ast_utils.generate_relationship_id(
                        self.current_context.id, used_entity.id, "usage"
                    )

                    self.relationships.append(
                        EntityRelation(
                            id=rel_id,
                            source_id=self.current_context.id,
                            target_id=used_entity.id,
                            type=RelationType.USAGE,
                            metadata={
                                "source_name": self.current_context.name,
                                "target_name": used_entity.name,
                                "line_number": node.lineno,
                                "usage_type": "variable_reference",
                            },
                        )
                    )

                self.generic_visit(node)

        visitor = UsageVisitor(self, entity_lookup)
        visitor.visit(tree)
        relationships.extend(visitor.relationships)

        return relationships

    async def extract_composition_relationships(
        self, tree: ast.AST, entity_lookup: dict[str, CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Extract composition relationships from the AST.

        Identifies where classes contain instances of other classes as attributes,
        particularly focusing on instance variable assignments in __init__ methods.
        This method detects "has-a" relationships where one class contains instances
        of another class, which is fundamental for understanding object composition
        and architectural patterns.

        The algorithm specifically looks for patterns in __init__ methods where
        instance attributes are assigned new instances of other classes:
        - self.attribute = ClassName()
        - self.service = ServiceClass(params)
        - Complex initialization patterns

        Composition relationships are crucial for:
            - Understanding object hierarchies and dependencies
            - Analyzing architectural patterns (dependency injection, etc.)
            - Identifying tight coupling between classes
            - Supporting refactoring and impact analysis

        Args:
            tree (ast.AST): The Abstract Syntax Tree of the Python code to analyze.
                Should contain class definitions with __init__ methods that perform
                instance variable assignments.
            entity_lookup (Dict[str, CoreCodeEntity]): Dictionary mapping entity names to
                CoreCodeEntity objects. Both the containing class and the composed class
                must exist in this lookup for relationships to be created.

        Returns:
            List[EntityRelation]: A list of EntityRelation objects representing
                composition relationships. Each relationship includes:
                - source_id: The ID of the containing class
                - target_id: The ID of the composed class
                - relation_type: RelationType.COMPOSITION
                - metadata: Contains attribute name and composition details

        Note:
            This method focuses on explicit object instantiation patterns in __init__
            methods. Other composition patterns (factory methods, dependency injection)
            may not be detected by this algorithm.

        Example:
            For code "class Car: def __init__(self): self.engine = Engine()",
            this method will create:
            EntityRelation(Car -> Engine, type=COMPOSITION, metadata={attribute_name: "engine"})
        """
        relationships: list[Any] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_entity = entity_lookup.get(node.name)
                if not class_entity:
                    continue

                # Look for instance variable assignments in __init__
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        for stmt in ast.walk(item):
                            if isinstance(stmt, ast.Assign):
                                # Look for self.attr = SomeClass()
                                for target in stmt.targets:
                                    if (
                                        isinstance(target, ast.Attribute)
                                        and isinstance(target.value, ast.Name)
                                        and target.value.id == "self"
                                    ):
                                        # Check if value is a class instantiation
                                        if isinstance(stmt.value, ast.Call):
                                            if isinstance(stmt.value.func, ast.Name):
                                                composed_class_name = stmt.value.func.id
                                                composed_entity = entity_lookup.get(
                                                    composed_class_name
                                                )

                                                if composed_entity:
                                                    rel_id = (
                                                        self.ast_utils.generate_relationship_id(
                                                            class_entity.id,
                                                            composed_entity.id,
                                                            "composition",
                                                        )
                                                    )

                                                    relationships.append(
                                                        EntityRelation(
                                                            id=rel_id,
                                                            source_id=class_entity.id,
                                                            target_id=composed_entity.id,
                                                            type=RelationType.COMPOSITION,
                                                            metadata={
                                                                "source_name": class_entity.name,
                                                                "target_name": composed_entity.name,
                                                                "attribute_name": target.attr,
                                                                "line_number": stmt.lineno,
                                                            },
                                                        )
                                                    )

        return relationships
