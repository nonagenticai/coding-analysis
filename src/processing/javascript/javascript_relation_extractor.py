"""Module for extracting relations from JavaScript code."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid7

from models.core_models import EntityRelation

from ..base_extractor import RelationExtractor

logger = logging.getLogger(__name__)


class JavaScriptRelationExtractor(RelationExtractor):
    """Specialized extractor for JavaScript/TypeScript code relationships."""

    async def extract_all_relations(
        self, ast: dict[str, Any], file_path: str, module_name: str
    ) -> list[dict[str, Any]]:
        """Extract all types of relationships from JavaScript/TypeScript AST."""
        relations: list[Any] = []
        # Extract different types of relationships
        basic_relations = await self._extract_basic_relationships(ast, file_path, module_name)
        call_relations = await self._extract_function_calls(ast, file_path, module_name)
        import_relations = await self._extract_imports(ast, file_path, module_name)

        # Combine all relations
        relations.extend(basic_relations)
        relations.extend(call_relations)
        relations.extend(import_relations)

        return relations

    async def _extract_basic_relationships(
        self, ast: dict[str, Any], _file_path: str, module_name: str
    ) -> list[dict[str, Any]]:
        """Extract basic relationships between JavaScript/TypeScript entities."""
        relationships: list[Any] = []
        # Process the AST body for relationships
        body = ast.get("body", [])
        for node in body:
            node_type = node.get("type")

            # Class inheritance relationships
            if node_type == "ClassDeclaration":
                class_relations = self._extract_class_relationships(node, module_name)
                relationships.extend(class_relations)

        return relationships

    def _extract_class_relationships(
        self, node: dict[str, Any], module_name: str
    ) -> list[EntityRelation]:
        """Extract relationships for a class declaration."""
        relationships: list[EntityRelation] = []
        class_name = node.get("id", {}).get("name")

        if not class_name:
            return relationships

        qualified_class_name = f"{module_name}.{class_name}" if module_name else class_name

        # Process inheritance
        super_class = node.get("superClass")
        if super_class and super_class.get("type") == "Identifier":
            base_name = super_class.get("name")
            try:
                relationships.append(
                    EntityRelation(
                        id=str(uuid7()),
                        source_id=qualified_class_name,
                        target_id=base_name,
                        type="INHERITS_FROM",
                        metadata={
                            "source_name": qualified_class_name,
                            "target_name": base_name,
                            "line_number": node.get("loc", {}).get("start", {}).get("line", 0),
                        },
                    )
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.error("Error creating inheritance relation: %s", e)  # Process class methods
        class_body = node.get("body", {}).get("body", [])
        for method_node in class_body:
            if method_node.get("type") == "MethodDefinition":
                method_relations = self._extract_method_relationships(
                    method_node, qualified_class_name
                )
                relationships.extend(method_relations)

        return relationships

    def _extract_method_relationships(
        self, method_node: dict[str, Any], qualified_class_name: str
    ) -> list[EntityRelation]:
        """Extract relationships for a method."""
        relationships: list[Any] = []
        method_name = method_node.get("key", {}).get("name")

        if method_name:
            qualified_method_name = f"{qualified_class_name}.{method_name}"
            try:
                relationships.append(
                    EntityRelation(
                        id=str(uuid7()),
                        source_id=qualified_method_name,
                        target_id=qualified_class_name,
                        type="MEMBER_OF",
                        metadata={
                            "source_name": qualified_method_name,
                            "target_name": qualified_class_name,
                            "line_number": method_node.get("loc", {})
                            .get("start", {})
                            .get("line", 0),
                        },
                    )
                )
            except (ValueError, TypeError, KeyError) as e:
                logger.error("Error creating method relationship: %s", e)

        return relationships

    async def _extract_function_calls(
        self, ast: dict[str, Any], _file_path: str, module_name: str
    ) -> list[EntityRelation]:
        """Extract function call relationships from JavaScript/TypeScript AST."""
        call_relations: list[Any] = []

        # Define a recursive function to extract calls from nodes
        def extract_calls_from_node(node, context=None):
            if not node or not isinstance(node, dict):
                return

            node_type = node.get("type")

            # Extract function context for calls
            if node_type in ["FunctionDeclaration", "MethodDefinition"]:
                new_context = self._get_function_context(node, module_name)
                body = self._get_function_body(node)

                if body:
                    # Process function body for calls
                    for child in body:
                        extract_calls_from_node(child, new_context)

            # Process call expressions
            elif node_type == "CallExpression" and context:
                call_relation = self._create_call_relation(node, context)
                if call_relation:
                    call_relations.append(call_relation)

            # Recursively process children
            for _, value in node.items():
                if isinstance(value, dict):
                    extract_calls_from_node(value, context)
                elif isinstance(value, list):
                    for item in value:
                        extract_calls_from_node(item, context)

        # Start processing from the root
        body = ast.get("body", [])
        for node in body:
            extract_calls_from_node(node)

        return call_relations

    def _create_call_relation(self, node: dict[str, Any], context: str) -> EntityRelation | None:
        """Create a function call relation from a call expression node."""
        callee = node.get("callee", {})
        callee_type = callee.get("type")

        target_name = None

        if callee_type == "Identifier":
            # Basic function call (e.g., foo())
            target_name = callee.get("name")
        elif callee_type == "MemberExpression":
            # Method call (e.g., obj.method())
            object_name = callee.get("object", {}).get("name")
            property_name = callee.get("property", {}).get("name")

            if object_name and property_name:
                target_name = f"{object_name}.{property_name}"

        if target_name and context:
            try:
                relation = EntityRelation(
                    id=str(uuid7()),
                    source_id=context,
                    target_id=target_name,
                    type="CALLS",
                    metadata={
                        "source_name": context,
                        "target_name": target_name,
                        "line_number": node.get("loc", {}).get("start", {}).get("line", 0),
                    },
                )
                return relation
            except (ValueError, TypeError, KeyError) as e:
                logger.error("Error creating call relation: %s", e)
                return None

        return None

    def _get_function_context(self, node: dict[str, Any], module_name: str) -> str:
        """Get the qualified function name for context."""
        node_type = node.get("type")

        if node_type == "FunctionDeclaration":
            func_name = node.get("id", {}).get("name")
            return f"{module_name}.{func_name}" if module_name and func_name else func_name

        if node_type == "MethodDefinition":
            method_name = node.get("key", {}).get("name")

            # Find parent class
            parent_class = None
            parent = node.get("parent")
            if parent and parent.get("type") == "ClassBody":
                class_parent = parent.get("parent")
                if class_parent and class_parent.get("type") == "ClassDeclaration":
                    class_name = class_parent.get("id", {}).get("name")
                    parent_class = f"{module_name}.{class_name}" if module_name else class_name

            if parent_class and method_name:
                return f"{parent_class}.{method_name}"

        return module_name  # Default to module context

    def _get_function_body(self, node: dict[str, Any]) -> list[dict[str, Any]]:
        """Get the function body from a function or method node."""
        result = (
            node.get("body", {}).get("body", [])
            if node.get("type") == "FunctionDeclaration"
            else node.get("value", {}).get("body", {}).get("body", [])
        )
        return result if isinstance(result, list) else []

    async def _extract_imports(
        self, ast: dict[str, Any], _file_path: str, module_name: str
    ) -> list[EntityRelation]:
        """Extract import relationships from JavaScript/TypeScript AST."""
        imports: list[Any] = []
        # Process the AST body for imports
        body = ast.get("body", [])
        for node in body:
            try:
                node_type = node.get("type")
                if node_type == "ImportDeclaration":
                    import_relations = self._extract_es6_imports(node, module_name)
                    imports.extend(import_relations)
                elif node_type == "VariableDeclaration":
                    require_relations = self._extract_require_imports(node, module_name)
                    imports.extend(require_relations)
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.error(
                    "Error extracting import from line %s: %s",
                    node.get("loc", {}).get("start", {}).get("line", "unknown"),
                    e,
                )

        return imports

    def _extract_es6_imports(self, node: dict[str, Any], module_name: str) -> list[EntityRelation]:
        """Extract ES6 import relationships."""
        imports: list[Any] = []
        source = node.get("source", {}).get("value")

        if source:
            specifiers = node.get("specifiers", [])
            for spec in specifiers:
                import_relation = self._create_es6_import_relation(spec, source, module_name, node)
                if import_relation:
                    imports.append(import_relation)

        return imports

    def _create_es6_import_relation(
        self,
        spec: dict[str, Any],
        source: str,
        module_name: str,
        node: dict[str, Any],
    ) -> EntityRelation | None:
        """Create an ES6 import relation."""
        spec_type = spec.get("type")
        imported_name = None

        if spec_type == "ImportDefaultSpecifier":
            imported_name = spec.get("local", {}).get("name")
        elif spec_type == "ImportSpecifier":
            imported_name = spec.get("imported", {}).get("name")
        elif spec_type == "ImportNamespaceSpecifier":
            imported_name = spec.get("local", {}).get("name")

        if imported_name:
            return EntityRelation(
                id=str(uuid7()),
                source_id=module_name,
                target_id=f"{source}.{imported_name}",
                type="import",
                metadata={
                    "source_name": module_name,
                    "target_name": f"{source}.{imported_name}",
                    "import_source": source,
                    "line_number": node.get("loc", {}).get("start", {}).get("line", 0),
                    "type": "es6_import",
                },
            )
        return None

    def _extract_require_imports(
        self, node: dict[str, Any], module_name: str
    ) -> list[EntityRelation]:
        """Extract CommonJS require() import relationships."""
        imports: list[Any] = []
        for declarator in node.get("declarations", []):
            require_relation = self._create_require_import_relation(declarator, module_name, node)
            if require_relation:
                imports.append(require_relation)

        return imports

    def _create_require_import_relation(
        self,
        declarator: dict[str, Any],
        module_name: str,
        node: dict[str, Any],
    ) -> EntityRelation | None:
        """Create a require() import relation."""
        init = declarator.get("init", {})

        if init.get("type") == "CallExpression" and init.get("callee", {}).get("name") == "require":
            source_arg = init.get("arguments", [])
            if source_arg and source_arg[0].get("type") == "Literal":
                source = source_arg[0].get("value")
                var_name = declarator.get("id", {}).get("name")

                if source and var_name:
                    return EntityRelation(
                        id=str(uuid7()),
                        source_id=module_name,
                        target_id=source,
                        type="import",
                        metadata={
                            "source_name": module_name,
                            "target_name": source,
                            "alias": var_name,
                            "line_number": node.get("loc", {}).get("start", {}).get("line", 0),
                            "type": "require",
                        },
                    )
        return None
