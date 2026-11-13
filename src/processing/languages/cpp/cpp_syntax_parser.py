"""
C++ syntax parsing and analysis module.

This module handles parsing C++ source code using Tree-sitter and regex fallback,
extracting syntactic elements like classes, functions, namespaces, etc.
"""

from __future__ import annotations

import re
from pathlib import Path

from models.core_models import (
    CoreCodeEntity,
    EntityType,
)

# Import Tree-sitter components if available
try:
    from tree_sitter import Node, Tree

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    # Tree/Node types not available - use Any in method signatures


class CppSyntaxParser:
    """C++ syntax parser that handles both Tree-sitter and regex-based parsing."""

    # C++-specific node types for Tree-sitter
    NAMESPACE_NODE_TYPES = {"namespace_definition"}
    INCLUDE_NODE_TYPES = {"preproc_include"}
    CLASS_NODE_TYPES = {"class_specifier"}
    STRUCT_NODE_TYPES = {"struct_specifier"}
    ENUM_NODE_TYPES = {"enum_specifier"}
    FUNCTION_NODE_TYPES = {"function_definition", "function_declarator"}
    METHOD_NODE_TYPES = {"function_definition"}
    TEMPLATE_NODE_TYPES = {"template_declaration"}
    TYPEDEF_NODE_TYPES = {"type_definition"}
    MACRO_NODE_TYPES = {"preproc_de"}

    def __init__(self):
        """Initialize the C++ syntax parser."""
        pass

    def is_cpp_file(self, file_path: Path, content: str) -> bool:
        """Determine if a file is C++ (vs C) based on extension and content."""
        return (
            file_path.suffix.lower()
            in [".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++"]
            or "namespace" in content
            or "class" in content
            or "#include <iostream>" in content
        )

    async def parse_with_regex(self, file_path: Path, content: str) -> list[CoreCodeEntity]:
        """Parse C++ code using regex patterns as fallback."""
        entities = []
        is_cpp = self.is_cpp_file(file_path, content)
        lang_name = "cpp" if is_cpp else "c"

        # Extract all different types of entities
        entities.extend(self._extract_includes_regex(file_path, content, lang_name))

        if is_cpp:
            entities.extend(self._extract_namespaces_regex(file_path, content, lang_name))
            entities.extend(self._extract_classes_regex(file_path, content, lang_name))
            entities.extend(self._extract_templates_regex(file_path, content, lang_name))

        entities.extend(self._extract_structs_regex(file_path, content, lang_name))
        entities.extend(self._extract_enums_regex(file_path, content, lang_name))
        entities.extend(self._extract_functions_regex(file_path, content, lang_name))
        entities.extend(self._extract_typedefs_regex(file_path, content, lang_name))
        entities.extend(self._extract_macros_regex(file_path, content, lang_name))
        entities.extend(self._extract_variables_regex(file_path, content, lang_name))

        return entities

    def _extract_includes_regex(
        self, file_path: Path, content: str, lang_name: str
    ) -> list[CoreCodeEntity]:
        """Extract #include statements using regex."""
        entities = []
        include_pattern = r'#include\s*[<"]([^>"]+)[>"]'

        for match in re.finditer(include_pattern, content):
            include_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"include:{include_name}:{file_path}",
                    name=include_name,
                    type=EntityType.IMPORT,
                    language=lang_name,
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
                    properties={"include_path": include_name},
                )
            )
        return entities

    def _extract_namespaces_regex(
        self, file_path: Path, content: str, lang_name: str
    ) -> list[CoreCodeEntity]:
        """Extract namespace declarations using regex."""
        entities = []
        namespace_pattern = r"namespace\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{"

        for match in re.finditer(namespace_pattern, content):
            namespace_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"namespace:{namespace_name}:{file_path}",
                    name=namespace_name,
                    type=EntityType.NAMESPACE,
                    language=lang_name,
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
                    properties={"namespace_name": namespace_name},
                )
            )
        return entities

    def _extract_classes_regex(
        self, file_path: Path, content: str, lang_name: str
    ) -> list[CoreCodeEntity]:
        """Extract class declarations using regex."""
        entities = []
        class_pattern = (
            r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*(?:public|private|protected)\s+[^{]+)?\s*\{"
        )

        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            # Extract inheritance information
            full_match = match.group(0)
            inheritance_info = []
            if ":" in full_match:
                inheritance_part = full_match.split(":")[1].split("{")[0]
                inheritance_info = [part.strip() for part in inheritance_part.split(",")]

            entities.append(
                CoreCodeEntity(
                    id=f"class:{class_name}:{file_path}",
                    name=class_name,
                    type=EntityType.CLASS,
                    language=lang_name,
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
                    properties={
                        "inheritance": inheritance_info,
                        "is_cpp_class": True,
                    },
                )
            )
        return entities

    def _extract_structs_regex(
        self, file_path: Path, content: str, lang_name: str
    ) -> list[CoreCodeEntity]:
        """Extract struct declarations using regex."""
        entities = []
        struct_pattern = r"(?:typedef\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{"

        for match in re.finditer(struct_pattern, content):
            struct_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"struct:{struct_name}:{file_path}",
                    name=struct_name,
                    type=EntityType.STRUCT,
                    language=lang_name,
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
                    properties={"is_struct": True},
                )
            )
        return entities

    def _extract_enums_regex(
        self, file_path: Path, content: str, lang_name: str
    ) -> list[CoreCodeEntity]:
        """Extract enum declarations using regex."""
        entities = []
        enum_pattern = r"(?:typedef\s+)?enum\s+(?:class\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\{"

        for match in re.finditer(enum_pattern, content):
            enum_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"enum:{enum_name}:{file_path}",
                    name=enum_name,
                    type=EntityType.ENUM,
                    language=lang_name,
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
                    properties={"is_enum": True},
                )
            )
        return entities

    def _extract_functions_regex(
        self, file_path: Path, content: str, lang_name: str
    ) -> list[CoreCodeEntity]:
        """Extract function declarations/definitions using regex."""
        entities = []
        function_pattern = r"(?:(?:static|inline|virtual|extern)\s+)?(?:([A-Za-z_][A-Za-z0-9_*&\s]*)\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*(?:const\s*)?(?:\s*:\s*[^{]+)?\s*\{"

        for match in re.finditer(function_pattern, content, re.MULTILINE):
            return_type = match.group(1)
            function_name = match.group(2)
            line_num = content[: match.start()].count("\n") + 1

            # Skip some common non-function patterns
            if function_name.lower() in [
                "i",
                "for",
                "while",
                "switch",
                "catch",
                "else",
                "namespace",
            ]:
                continue

            # Skip constructors and destructors for now (they need special handling)
            if function_name.startswith("~") or (return_type is None and lang_name == "cpp"):
                continue

            entities.append(
                CoreCodeEntity(
                    id=f"function:{function_name}:{file_path}:{line_num}",
                    name=function_name,
                    type=EntityType.FUNCTION,
                    language=lang_name,
                    file_path=str(file_path),
                    line_number=line_num,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility=None,
                    code_snippet=None,
                    docstring=None,
                    signature=match.group(0).strip(),
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={
                        "return_type": (return_type.strip() if return_type else "void"),
                        "signature": match.group(0).strip(),
                    },
                )
            )
        return entities

    def _extract_templates_regex(
        self, file_path: Path, content: str, lang_name: str
    ) -> list[CoreCodeEntity]:
        """Extract template declarations using regex."""
        entities = []
        template_pattern = (
            r"template\s*<[^>]+>\s*(?:class|struct|typename)\s+([A-Za-z_][A-Za-z0-9_]*)"
        )

        for match in re.finditer(template_pattern, content):
            template_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"template:{template_name}:{file_path}",
                    name=template_name,
                    type=EntityType.GENERIC,
                    language=lang_name,
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
                    properties={"is_template": True},
                )
            )
        return entities

    def _extract_typedefs_regex(
        self, file_path: Path, content: str, lang_name: str
    ) -> list[CoreCodeEntity]:
        """Extract typedef declarations using regex."""
        entities = []
        typedef_pattern = r"typedef\s+([^;]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*;"

        for match in re.finditer(typedef_pattern, content):
            base_type = match.group(1).strip()
            typedef_name = match.group(2)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"typedef:{typedef_name}:{file_path}",
                    name=typedef_name,
                    type=EntityType.TYPE_ALIAS,
                    language=lang_name,
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
                    properties={"base_type": base_type, "is_typedef": True},
                )
            )
        return entities

    def _extract_macros_regex(
        self, file_path: Path, content: str, lang_name: str
    ) -> list[CoreCodeEntity]:
        """Extract #define macros using regex."""
        entities = []
        define_pattern = r"#define\s+([A-Za-z_][A-Za-z0-9_]*)"

        for match in re.finditer(define_pattern, content):
            macro_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            entities.append(
                CoreCodeEntity(
                    id=f"macro:{macro_name}:{file_path}",
                    name=macro_name,
                    type=EntityType.MACRO,
                    language=lang_name,
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
                    properties={"is_macro": True},
                )
            )
        return entities

    def _extract_variables_regex(
        self, file_path: Path, content: str, lang_name: str
    ) -> list[CoreCodeEntity]:
        """Extract global variable declarations using regex."""
        entities = []
        global_var_pattern = r"^(?:extern\s+|static\s+)?([A-Za-z_][A-Za-z0-9_*&\s]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:=.*)?;"

        for match in re.finditer(global_var_pattern, content, re.MULTILINE):
            var_type = match.group(1).strip()
            var_name = match.group(2)
            line_num = content[: match.start()].count("\n") + 1

            # Skip function declarations and some patterns
            if "(" in var_name or var_name.lower() in [
                "int",
                "char",
                "float",
                "double",
                "void",
                "class",
                "struct",
            ]:
                continue

            entities.append(
                CoreCodeEntity(
                    id=f"variable:{var_name}:{file_path}:{line_num}",
                    name=var_name,
                    type=EntityType.VARIABLE,
                    language=lang_name,
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
                    properties={"variable_type": var_type, "is_global": True},
                )
            )
        return entities

    # Tree-sitter parsing methods
    async def parse_with_tree_sitter(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Parse C++ code using Tree-sitter."""
        if not TREE_SITTER_AVAILABLE:
            return []

        entities = []

        # Extract different types of entities using Tree-sitter
        entities.extend(await self._extract_includes_ts(tree, file_path, content))
        entities.extend(await self._extract_namespaces_ts(tree, file_path, content))
        entities.extend(await self._extract_classes_ts(tree, file_path, content))
        entities.extend(await self._extract_structs_ts(tree, file_path, content))
        entities.extend(await self._extract_enums_ts(tree, file_path, content))
        entities.extend(await self._extract_functions_ts(tree, file_path, content))
        entities.extend(await self._extract_templates_ts(tree, file_path, content))
        entities.extend(await self._extract_typedefs_ts(tree, file_path, content))
        entities.extend(await self._extract_macros_ts(tree, file_path, content))

        return entities

    def _find_nodes_by_type(self, tree: Tree, node_types: set) -> list[Node]:
        """Find all nodes of specific types in the tree."""
        if not TREE_SITTER_AVAILABLE:
            return []

        def traverse(node):
            nodes = []
            if node.type in node_types:
                nodes.append(node)
            for child in node.children:
                nodes.extend(traverse(child))
            return nodes

        return traverse(tree.root_node)

    def _get_node_text(self, node: Node, content: bytes) -> str:
        """Get the text content of a node."""
        return content[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

    async def _extract_includes_ts(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract #include statements using Tree-sitter."""
        entities = []
        include_nodes = self._find_nodes_by_type(tree, self.INCLUDE_NODE_TYPES)

        for node in include_nodes:
            include_text = self._get_node_text(node, content)
            match = re.search(r'#include\s*[<"]([^>"]+)[>"]', include_text)
            if match:
                include_name = match.group(1)

                entities.append(
                    CoreCodeEntity(
                        id=f"include:{include_name}:{file_path}",
                        name=include_name,
                        type=EntityType.IMPORT,
                        language="cpp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"include_path": include_name},
                    )
                )
        return entities

    async def _extract_namespaces_ts(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract namespace declarations using Tree-sitter."""
        entities = []
        namespace_nodes = self._find_nodes_by_type(tree, self.NAMESPACE_NODE_TYPES)

        for node in namespace_nodes:
            namespace_text = self._get_node_text(node, content)
            match = re.search(r"namespace\s+([A-Za-z_][A-Za-z0-9_]*)", namespace_text)
            if match:
                namespace_name = match.group(1)

                entities.append(
                    CoreCodeEntity(
                        id=f"namespace:{namespace_name}:{file_path}",
                        name=namespace_name,
                        type=EntityType.NAMESPACE,
                        language="cpp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"namespace_name": namespace_name},
                    )
                )
        return entities

    async def _extract_classes_ts(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract class declarations using Tree-sitter."""
        entities = []
        class_nodes = self._find_nodes_by_type(tree, self.CLASS_NODE_TYPES)

        for node in class_nodes:
            class_text = self._get_node_text(node, content)
            match = re.search(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", class_text)
            if match:
                class_name = match.group(1)

                entities.append(
                    CoreCodeEntity(
                        id=f"class:{class_name}:{file_path}",
                        name=class_name,
                        type=EntityType.CLASS,
                        language="cpp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"is_cpp_class": True},
                    )
                )
        return entities

    async def _extract_structs_ts(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract struct declarations using Tree-sitter."""
        entities = []
        struct_nodes = self._find_nodes_by_type(tree, self.STRUCT_NODE_TYPES)

        for node in struct_nodes:
            struct_text = self._get_node_text(node, content)
            match = re.search(r"struct\s+([A-Za-z_][A-Za-z0-9_]*)", struct_text)
            if match:
                struct_name = match.group(1)

                entities.append(
                    CoreCodeEntity(
                        id=f"struct:{struct_name}:{file_path}",
                        name=struct_name,
                        type=EntityType.STRUCT,
                        language="cpp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"is_struct": True},
                    )
                )
        return entities

    async def _extract_enums_ts(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract enum declarations using Tree-sitter."""
        entities = []
        enum_nodes = self._find_nodes_by_type(tree, self.ENUM_NODE_TYPES)

        for node in enum_nodes:
            enum_text = self._get_node_text(node, content)
            match = re.search(r"enum\s+(?:class\s+)?([A-Za-z_][A-Za-z0-9_]*)", enum_text)
            if match:
                enum_name = match.group(1)

                entities.append(
                    CoreCodeEntity(
                        id=f"enum:{enum_name}:{file_path}",
                        name=enum_name,
                        type=EntityType.ENUM,
                        language="cpp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"is_enum": True},
                    )
                )
        return entities

    async def _extract_functions_ts(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract function declarations/definitions using Tree-sitter."""
        entities = []
        function_nodes = self._find_nodes_by_type(tree, self.FUNCTION_NODE_TYPES)

        for node in function_nodes:
            function_text = self._get_node_text(node, content)

            # Extract function name
            match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", function_text)
            if match:
                function_name = match.group(1)

                entities.append(
                    CoreCodeEntity(
                        id=f"function:{function_name}:{file_path}:{node.start_point[0] + 1}",
                        name=function_name,
                        type=EntityType.FUNCTION,
                        language="cpp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=function_text.split("\n")[0].strip(),
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"signature": function_text.split("\n")[0].strip()},
                    )
                )
        return entities

    async def _extract_templates_ts(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract template declarations using Tree-sitter."""
        entities = []
        template_nodes = self._find_nodes_by_type(tree, self.TEMPLATE_NODE_TYPES)

        for node in template_nodes:
            template_text = self._get_node_text(node, content)
            match = re.search(
                r"template\s*<[^>]+>\s*(?:class|struct|typename)\s+([A-Za-z_][A-Za-z0-9_]*)",
                template_text,
            )
            if match:
                template_name = match.group(1)

                entities.append(
                    CoreCodeEntity(
                        id=f"template:{template_name}:{file_path}",
                        name=template_name,
                        type=EntityType.GENERIC,
                        language="cpp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"is_template": True},
                    )
                )
        return entities

    async def _extract_typedefs_ts(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract typedef declarations using Tree-sitter."""
        entities = []
        typedef_nodes = self._find_nodes_by_type(tree, self.TYPEDEF_NODE_TYPES)

        for node in typedef_nodes:
            typedef_text = self._get_node_text(node, content)
            match = re.search(r"typedef\s+([^;]+)\s+([A-Za-z_][A-Za-z0-9_]*)", typedef_text)
            if match:
                base_type = match.group(1).strip()
                typedef_name = match.group(2)

                entities.append(
                    CoreCodeEntity(
                        id=f"typedef:{typedef_name}:{file_path}",
                        name=typedef_name,
                        type=EntityType.TYPE_ALIAS,
                        language="cpp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"base_type": base_type, "is_typedef": True},
                    )
                )
        return entities

    async def _extract_macros_ts(
        self, tree: Tree, file_path: Path, content: bytes
    ) -> list[CoreCodeEntity]:
        """Extract #define macros using Tree-sitter."""
        entities = []
        macro_nodes = self._find_nodes_by_type(tree, self.MACRO_NODE_TYPES)

        for node in macro_nodes:
            macro_text = self._get_node_text(node, content)
            match = re.search(r"#define\s+([A-Za-z_][A-Za-z0-9_]*)", macro_text)
            if match:
                macro_name = match.group(1)

                entities.append(
                    CoreCodeEntity(
                        id=f"macro:{macro_name}:{file_path}",
                        name=macro_name,
                        type=EntityType.MACRO,
                        language="cpp",
                        file_path=str(file_path),
                        line_number=node.start_point[0] + 1,
                        end_line_number=None,
                        column_start=node.start_point[1],
                        column_end=node.end_point[1],
                        visibility=None,
                        code_snippet=None,
                        docstring=None,
                        signature=None,
                        source_code=None,
                        created_at=None,
                        updated_at=None,
                        properties={"is_macro": True},
                    )
                )
        return entities
