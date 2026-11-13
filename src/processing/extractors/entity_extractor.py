"""Extractor for entities from source code using AST."""

from __future__ import annotations

import ast
import logging
from datetime import UTC, datetime
from typing import Any

from models.core_models import CoreCodeEntity, EntityType

from ..utils.ast_utils import ASTUtils

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts code entities (classes, functions, variables) from AST"""

    def __init__(self) -> None:
        self.ast_utils = ASTUtils()

    def _create_entity_with_defaults(
        self,
        entity_id: str,
        name: str,
        entity_type: EntityType,
        file_path: str,
        line_number: int,
        language: str = "python",
        **kwargs,
    ) -> CoreCodeEntity:
        """Helper to create CoreCodeEntity with required fields."""
        return CoreCodeEntity(
            id=entity_id,
            name=name,
            type=entity_type,
            language=language,
            file_path=file_path,
            line_number=line_number,
            column_start=kwargs.get("column_start", 0),
            column_end=kwargs.get("column_end", 0),
            visibility=kwargs.get("visibility", "public"),
            signature=kwargs.get("signature"),
            source_code=kwargs.get("source_code"),
            end_line_number=kwargs.get("end_line_number"),
            code_snippet=kwargs.get("code_snippet"),
            docstring=kwargs.get("docstring"),
            metadata=kwargs.get("metadata", {}),
            created_at=kwargs.get("created_at", datetime.now(UTC)),
            updated_at=kwargs.get("updated_at", datetime.now(UTC)),
        )

    async def extract_python_entities(
        self,
        tree: ast.AST,
        file_path: str,
        file_lines: list[str] | None = None,
    ) -> list[CoreCodeEntity]:
        """Extract all entities from Python AST"""
        entities: list[Any] = []
        # Add module entity first
        module_entity = self._create_module_entity(tree, file_path, file_lines)
        entities.append(module_entity)

        # Extract other entities
        for node in ast.walk(tree):
            entity = None

            if isinstance(node, ast.ClassDef):
                entity = self._create_class_entity(node, file_path, file_lines)
            elif isinstance(node, ast.FunctionDef):
                entity = self._create_function_entity(node, file_path, file_lines)
            elif isinstance(node, ast.AsyncFunctionDef):
                entity = self._create_async_function_entity(node, file_path, file_lines)
            elif isinstance(node, ast.Assign):
                entities.extend(self._create_variable_entities(node, file_path))
            elif isinstance(node, ast.AnnAssign):
                entities.extend(self._create_annotated_variable_entities(node, file_path))

            if entity:
                entities.append(entity)

        return entities

    async def extract_import_entities(self, tree: ast.AST, file_path: str) -> list[CoreCodeEntity]:
        """Extract import entities separately for better organization"""
        entities: list[Any] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                entities.extend(self._create_import_entities(node, file_path))
            elif isinstance(node, ast.ImportFrom):
                entities.extend(self._create_import_from_entities(node, file_path))

        return entities

    def _create_module_entity(
        self,
        tree: ast.AST,
        file_path: str,
        file_lines: list[str] | None = None,
    ) -> CoreCodeEntity:
        """Create entity for module"""
        module_name = self.ast_utils.get_module_name(file_path)
        entity_id = self.ast_utils.generate_entity_id("module", module_name, file_path, 1)

        # Extract module-level docstring
        docstring = None
        if isinstance(tree, ast.Module):
            docstring = self.ast_utils.extract_docstring(tree)

        # Calculate module metrics
        total_lines = len(file_lines) if file_lines else 0

        return self._create_entity_with_defaults(
            entity_id=entity_id,
            name=module_name,
            entity_type=EntityType.MODULE,
            file_path=file_path,
            line_number=1,
            end_line_number=total_lines,
            docstring=docstring,
            metadata={
                "docstring": docstring,
                "total_lines": total_lines,
                "is_package": file_path.endswith("__init__.py"),
            },
        )

    def _create_class_entity(
        self,
        node: ast.ClassDef,
        file_path: str,
        file_lines: list[str] | None = None,
    ) -> CoreCodeEntity:
        """Create entity for class definition"""
        entity_id = self.ast_utils.generate_entity_id("class", node.name, file_path, node.lineno)

        # Extract class information
        base_classes = self.ast_utils.extract_base_classes(node)
        decorators = self.ast_utils.extract_decorators(node)
        docstring = self.ast_utils.extract_docstring(node)

        # Extract methods
        methods: list[Any] = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(
                    {
                        "name": item.name,
                        "line_number": item.lineno,
                        "is_async": isinstance(item, ast.AsyncFunctionDef),
                        "is_property": any(
                            (d.id == "property" if isinstance(d, ast.Name) else False)
                            for d in item.decorator_list
                        ),
                        "is_classmethod": any(
                            (d.id == "classmethod" if isinstance(d, ast.Name) else False)
                            for d in item.decorator_list
                        ),
                        "is_staticmethod": any(
                            (d.id == "staticmethod" if isinstance(d, ast.Name) else False)
                            for d in item.decorator_list
                        ),
                    }
                )

        # Extract class variables
        class_variables: list[Any] = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_variables.append(target.id)

        # Get code snippet if file_lines available
        code_snippet = None
        if file_lines and hasattr(node, "end_lineno"):
            code_snippet = self.ast_utils.get_code_snippet(file_lines, node.lineno, node.end_lineno)

        return self._create_entity_with_defaults(
            entity_id=entity_id,
            name=node.name,
            entity_type=EntityType.CLASS,
            file_path=file_path,
            line_number=node.lineno,
            end_line_number=getattr(node, "end_lineno", None),
            code_snippet=code_snippet,
            docstring=docstring,
            metadata={
                "base_classes": base_classes,
                "decorators": decorators,
                "docstring": docstring,
                "methods": methods,
                "class_variables": class_variables,
                "code_snippet": code_snippet,
                "end_line_number": getattr(node, "end_lineno", None),
            },
        )

    def _create_function_entity(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        file_lines: list[str] | None = None,
    ) -> CoreCodeEntity:
        """Create entity for function definition"""
        entity_id = self.ast_utils.generate_entity_id("function", node.name, file_path, node.lineno)

        # Extract function information
        parameters = self.ast_utils.extract_function_parameters(node)
        decorators = self.ast_utils.extract_decorators(node)
        docstring = self.ast_utils.extract_docstring(node)

        # Extract return annotation
        return_annotation = None
        if node.returns:
            return_annotation = self.ast_utils._get_annotation_string(node.returns)

        # Calculate complexity
        cyclomatic_complexity = self.ast_utils.calculate_cyclomatic_complexity(node)
        cognitive_complexity = self.ast_utils.calculate_cognitive_complexity(node)

        # Check if it's a method (has 'sel' or 'cls' as first parameter)
        is_method = len(parameters) > 0 and parameters[0]["name"] in [
            "sel",
            "cls",
        ]
        is_classmethod = "classmethod" in decorators
        is_staticmethod = "staticmethod" in decorators
        is_property = "property" in decorators

        # Get code snippet if file_lines available
        code_snippet = None
        if file_lines and hasattr(node, "end_lineno"):
            code_snippet = self.ast_utils.get_code_snippet(file_lines, node.lineno, node.end_lineno)

        return self._create_entity_with_defaults(
            entity_id=entity_id,
            name=node.name,
            entity_type=EntityType.FUNCTION,
            file_path=file_path,
            line_number=node.lineno,
            end_line_number=getattr(node, "end_lineno", None),
            code_snippet=code_snippet,
            docstring=docstring,
            metadata={
                "parameters": parameters,
                "decorators": decorators,
                "docstring": docstring,
                "return_annotation": return_annotation,
                "cyclomatic_complexity": cyclomatic_complexity,
                "cognitive_complexity": cognitive_complexity,
                "is_method": is_method,
                "is_classmethod": is_classmethod,
                "is_staticmethod": is_staticmethod,
                "is_property": is_property,
                "is_async": False,
                "code_snippet": code_snippet,
                "end_line_number": getattr(node, "end_lineno", None),
            },
        )

    def _create_async_function_entity(
        self,
        node: ast.AsyncFunctionDef,
        file_path: str,
        file_lines: list[str] | None = None,
    ) -> CoreCodeEntity:
        """Create entity for async function definition"""
        # Reuse function logic but mark as async
        entity = self._create_function_entity(node, file_path, file_lines)
        # Create new metadata dict to avoid mutation issues
        new_metadata = entity.metadata.copy()
        new_metadata["is_async"] = True
        entity = entity.model_copy(update={"metadata": new_metadata})
        return entity

    def _create_import_entities(self, node: ast.Import, file_path: str) -> list[CoreCodeEntity]:
        """Create entities for import statements"""
        entities: list[Any] = []
        for alias in node.names:
            entity_id = self.ast_utils.generate_entity_id(
                "import", alias.name, file_path, node.lineno
            )

            entities.append(
                self._create_entity_with_defaults(
                    entity_id=entity_id,
                    name=alias.name,
                    entity_type=EntityType.IMPORT,
                    file_path=file_path,
                    line_number=node.lineno,
                    end_line_number=node.lineno,
                    metadata={
                        "alias": alias.asname,
                        "module": alias.name,
                        "import_type": "import",
                        "is_external": self._is_external_import(alias.name),
                    },
                )
            )

        return entities

    def _create_import_from_entities(
        self, node: ast.ImportFrom, file_path: str
    ) -> list[CoreCodeEntity]:
        """Create entities for from...import statements"""
        entities: list[Any] = []
        module_name = node.module or ""

        for alias in node.names:
            import_name = f"{module_name}.{alias.name}" if module_name else alias.name
            entity_id = self.ast_utils.generate_entity_id(
                "import", import_name, file_path, node.lineno
            )

            entities.append(
                self._create_entity_with_defaults(
                    entity_id=entity_id,
                    name=alias.name,
                    entity_type=EntityType.IMPORT,
                    file_path=file_path,
                    line_number=node.lineno,
                    end_line_number=node.lineno,
                    metadata={
                        "alias": alias.asname,
                        "module": module_name,
                        "import_type": "from_import",
                        "full_name": import_name,
                        "is_external": self._is_external_import(module_name),
                        "level": node.level,  # For relative imports
                    },
                )
            )

        return entities

    def _create_variable_entities(self, node: ast.Assign, file_path: str) -> list[CoreCodeEntity]:
        """Create entities for variable assignments"""
        entities: list[Any] = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                entity_id = self.ast_utils.generate_entity_id(
                    "variable", target.id, file_path, node.lineno
                )

                # Try to determine variable type from value
                value_type = self._get_value_type(node.value)

                entities.append(
                    self._create_entity_with_defaults(
                        entity_id=entity_id,
                        name=target.id,
                        entity_type=EntityType.VARIABLE,
                        file_path=file_path,
                        line_number=node.lineno,
                        end_line_number=node.lineno,
                        metadata={
                            "value_type": value_type,
                            "is_constant": target.id.isupper(),  # Convention for constants
                            "annotation": self._get_target_annotation(target),
                        },
                    )
                )

        return entities

    def _create_annotated_variable_entities(
        self, node: ast.AnnAssign, file_path: str
    ) -> list[CoreCodeEntity]:
        """Create entities for annotated variable assignments"""
        entities: list[Any] = []

        # Annotated assignments have a single target
        if isinstance(node.target, ast.Name):
            entity_id = self.ast_utils.generate_entity_id(
                "variable", node.target.id, file_path, node.lineno
            )

            # Extract type annotation
            annotation_str = self._extract_annotation_string(node.annotation)

            # Try to determine variable type from value if present
            value_type = None
            if node.value:
                value_type = self._get_value_type(node.value)

            entities.append(
                self._create_entity_with_defaults(
                    entity_id=entity_id,
                    name=node.target.id,
                    entity_type=EntityType.VARIABLE,
                    file_path=file_path,
                    line_number=node.lineno,
                    end_line_number=node.lineno,
                    metadata={
                        "annotation": annotation_str,
                        "value_type": value_type,
                        "is_annotated": True,
                        "is_constant": node.target.id.isupper(),  # Convention for constants
                        "has_value": node.value is not None,
                    },
                )
            )

        return entities

    def _is_external_import(self, module_name: str) -> bool:
        """Determine if an import is external (not part of the project)"""
        if not module_name:
            return False

        # Common standard library modules
        stdlib_modules = {
            "os",
            "sys",
            "json",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "typing",
            "pathlib",
            "logging",
            "unittest",
            "asyncio",
            "concurrent",
            "multiprocessing",
            "threading",
            "urllib",
            "http",
            "email",
            "xml",
            "html",
            "sqlite3",
        }

        # Check if it's a standard library module
        root_module = module_name.split(".")[0]
        if root_module in stdlib_modules:
            return True

        # Check if it starts with common third-party prefixes
        third_party_prefixes = [
            "numpy",
            "pandas",
            "requests",
            "flask",
            "django",
            "fastapi",
        ]
        if any(module_name.startswith(prefix) for prefix in third_party_prefixes):
            return True

        # If it starts with a dot, it's a relative import (internal)
        if module_name.startswith("."):
            return False

        # Default to external if we can't determine
        return True

    def _get_value_type(self, node: ast.expr) -> str:
        """Determine the type of a value from its AST node"""
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        if isinstance(node, ast.List):
            return "list"
        if isinstance(node, ast.Dict):
            return "dict"
        if isinstance(node, ast.Set):
            return "set"
        if isinstance(node, ast.Tuple):
            return "tuple"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
        return "unknown"

    def _get_target_annotation(self, target: ast.expr) -> str | None:
        """Get type annotation for assignment target if available"""
        # For regular assignments, there's no annotation on the target
        # Annotations are handled in visit_AnnAssign method for annotated assignments
        if hasattr(target, "annotation"):
            return self._extract_annotation_string(target.annotation)  # type: ignore[invalid-argument-type]  # ty: ignore[invalid-argument-type]
        return None

    def _extract_annotation_string(self, annotation_node: ast.expr) -> str:
        """Extract string representation of type annotation from AST node"""
        try:
            if isinstance(annotation_node, ast.Name):
                # Basic name like 'int', 'str', 'List'
                return annotation_node.id
            if isinstance(annotation_node, ast.Constant):
                # String annotations like 'List[str]'
                return str(annotation_node.value)
            if isinstance(annotation_node, ast.Attribute):
                # Qualified names like 'typing.List'
                return f"{self._extract_annotation_string(annotation_node.value)}.{annotation_node.attr}"
            if isinstance(annotation_node, ast.Subscript):
                # Generic types like 'List[str]', 'Dict[str, int]'
                base = self._extract_annotation_string(annotation_node.value)
                slice_value = self._extract_annotation_string(annotation_node.slice)
                return f"{base}[{slice_value}]"
            if isinstance(annotation_node, ast.Tuple):
                # Multiple types in tuple for generics like 'Dict[str, int]'
                elements = [self._extract_annotation_string(elt) for elt in annotation_node.elts]
                return ", ".join(elements)
            if isinstance(annotation_node, ast.BinOp) and isinstance(annotation_node.op, ast.BitOr):
                # Union types using | syntax (Python 3.10+) like 'str | int'
                left = self._extract_annotation_string(annotation_node.left)
                right = self._extract_annotation_string(annotation_node.right)
                return f"{left} | {right}"
            if hasattr(annotation_node, "id"):
                # Fallback for basic names
                return str(annotation_node.id)
            # For complex annotations, try to get source representation
            import ast as ast_module

            try:
                return ast_module.unparse(annotation_node)
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(f"Failed to unparse annotation node: {e}")
                return "unknown"
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Error extracting annotation string: {e}")
            return "unknown"
