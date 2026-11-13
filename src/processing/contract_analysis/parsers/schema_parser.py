"""Parse Pydantic BaseModel classes from Python source using tree-sitter."""

from __future__ import annotations

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from .types import ModelField, PydanticModel, ServiceSchema

PY_LANGUAGE = Language(tspython.language())


def _make_parser() -> Parser:
    return Parser(PY_LANGUAGE)


def _node_text(node) -> str:
    """Extract text from a tree-sitter node."""
    return node.text.decode() if node.text else ""


def _unwrap_optional(type_node) -> str | None:
    """If *type_node* represents ``Optional[X]``, return the inner type text."""
    for child in type_node.children:
        if child.type != "generic_type":
            continue
        ident = None
        type_param = None
        for gc in child.children:
            if gc.type == "identifier" and ident is None:
                ident = gc
            elif gc.type == "type_parameter":
                type_param = gc
        if ident is None or _node_text(ident) != "Optional" or type_param is None:
            continue
        # The inner type lives inside the type_parameter between [ and ]
        for tp_child in type_param.children:
            if tp_child.type == "type":
                return _extract_type_annotation(tp_child)
    return None


def _extract_type_annotation(type_node) -> str:
    """Extract the type annotation string, normalizing Optional[X] to X | None."""
    inner = _unwrap_optional(type_node)
    if inner is not None:
        return f"{inner} | None"
    return _node_text(type_node)


def _is_field_required(assignment_node) -> tuple[bool, str | None]:
    """Determine if a field assignment is required and extract default.

    Returns (required, default_string).
    """
    # Find the value part of the assignment (after the =)
    children = assignment_node.children
    eq_idx = None
    for i, child in enumerate(children):
        if child.type == "=":
            eq_idx = i
            break

    if eq_idx is None:
        # No = sign means no default -> required (e.g., `field: str`)
        return True, None

    value_node = children[eq_idx + 1] if eq_idx + 1 < len(children) else None
    if value_node is None:
        return True, None

    value_text = _node_text(value_node)

    # Check for Field(...) call
    if value_node.type == "call":
        func_node = value_node.child_by_field_name("function")
        if func_node and _node_text(func_node) == "Field":
            args = value_node.child_by_field_name("arguments")
            if args:
                # Check positional args for ellipsis
                for arg in args.children:
                    if arg.type == "ellipsis":
                        return True, None
                    if arg.type == "keyword_argument":
                        key = arg.children[0]
                        if _node_text(key) in ("default", "default_factory"):
                            return False, value_text
                # If Field() has no positional ellipsis and no default/default_factory,
                # check if there are any positional args at all
                positional = [
                    a for a in args.children if a.type not in ("(", ")", ",", "keyword_argument")
                ]
                if not positional:
                    # Field() with only keyword args but no default -> required
                    return True, None
                # Non-ellipsis positional arg = default value (e.g. Field(None, ...))
                return False, value_text
            return True, None

    # Any other default value -> optional
    return False, value_text


def _check_annotated_field(type_node) -> tuple[bool, str | None] | None:
    """Check if type_node is Annotated[X, Field(...)], return (required, default) or None."""
    for child in type_node.children:
        if child.type != "generic_type":
            continue
        ident = None
        type_param = None
        for gc in child.children:
            if gc.type == "identifier" and ident is None:
                ident = gc
            elif gc.type == "type_parameter":
                type_param = gc
        if ident is None or _node_text(ident) != "Annotated" or type_param is None:
            continue
        # Look for Field(...) call inside type_parameter (may be wrapped in a type node)
        for tp_child in type_param.children:
            call_node = None
            if tp_child.type == "call":
                call_node = tp_child
            elif tp_child.type == "type":
                for sub in tp_child.children:
                    if sub.type == "call":
                        call_node = sub
                        break
            if call_node is None:
                continue
            func_node = call_node.child_by_field_name("function")
            if func_node and _node_text(func_node) == "Field":
                args = call_node.child_by_field_name("arguments")
                if args:
                    for arg in args.children:
                        if arg.type == "ellipsis":
                            return True, None
                        if arg.type == "keyword_argument":
                            key = arg.children[0]
                            if _node_text(key) in ("default", "default_factory"):
                                return False, _node_text(call_node)
                    positional = [
                        a
                        for a in args.children
                        if a.type not in ("(", ")", ",", "keyword_argument")
                    ]
                    if not positional:
                        return True, None
                    return False, _node_text(call_node)
                return True, None
    return None


def _extract_annotated_inner_type(type_node) -> str:
    """Extract the inner type X from Annotated[X, ...]."""
    for child in type_node.children:
        if child.type != "generic_type":
            continue
        ident = None
        type_param = None
        for gc in child.children:
            if gc.type == "identifier" and ident is None:
                ident = gc
            elif gc.type == "type_parameter":
                type_param = gc
        if ident is None or _node_text(ident) != "Annotated" or type_param is None:
            continue
        # First type child inside type_parameter is the inner type
        for tp_child in type_param.children:
            if tp_child.type == "type":
                return _extract_type_annotation(tp_child)
    return _node_text(type_node)


def _extract_fields_from_body(body_node) -> list[ModelField]:
    """Extract ModelField instances from a class body block node."""
    fields = []
    for stmt in body_node.children:
        if stmt.type != "expression_statement":
            continue
        # Look for assignment with type annotation
        for child in stmt.children:
            if child.type != "assignment":
                continue
            # Check if this assignment has a type annotation
            name_node = None
            type_node = None
            for sub in child.children:
                if sub.type == "identifier" and name_node is None:
                    name_node = sub
                elif sub.type == "type":
                    type_node = sub

            if name_node is None or type_node is None:
                continue

            name = _node_text(name_node)

            # Check for Annotated[X, Field(...)] pattern
            annotated = _check_annotated_field(type_node)
            if annotated is not None:
                required, default = annotated
                type_ann = _extract_annotated_inner_type(type_node)
            else:
                type_ann = _extract_type_annotation(type_node)
                required, default = _is_field_required(child)

            fields.append(
                ModelField(
                    name=name,
                    type_annotation=type_ann,
                    required=required,
                    default=default,
                )
            )
    return fields


def _has_model_validator(body_node) -> bool:
    """Check if the class body contains a @model_validator decorated method."""
    for stmt in body_node.children:
        if stmt.type != "decorated_definition":
            continue
        for child in stmt.children:
            if child.type == "decorator":
                dec_text = _node_text(child)
                if "model_validator" in dec_text:
                    return True
    return False


def _get_base_classes(class_node) -> list[str]:
    """Extract base class names from a class_definition node."""
    arg_list = class_node.child_by_field_name("superclasses")
    if arg_list is None:
        return []
    bases = []
    for child in arg_list.children:
        if child.type == "identifier":
            bases.append(_node_text(child))
        elif child.type == "attribute":
            bases.append(_node_text(child))
    return bases


def parse_schemas(source: str, *, resolve: bool = True) -> ServiceSchema:
    """Parse all Pydantic BaseModel classes from Python source code.

    Args:
        source: Python source code string.
        resolve: If True, resolve inheritance after parsing. Set to False when
            parsing multiple files that will be merged before resolution.

    Returns:
        ServiceSchema with parsed models and any errors.
    """
    if not source.strip():
        return ServiceSchema()

    parser = _make_parser()
    tree = parser.parse(source.encode())
    root = tree.root_node

    result = ServiceSchema()
    model_names: set[str] = set()

    # First pass: find all class definitions that inherit from BaseModel
    # or from another model we've already found
    class_nodes = []
    for node in root.children:
        if node.type == "class_definition":
            class_nodes.append(node)

    # We need multiple passes to resolve transitive inheritance
    # First, collect all classes and their bases
    class_info: list[tuple] = []  # (node, name, bases)
    for node in class_nodes:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            continue
        name = _node_text(name_node)
        bases = _get_base_classes(node)
        class_info.append((node, name, bases))

    # Determine which classes are BaseModel subclasses (direct or transitive)
    is_model: dict[str, bool] = {}
    changed = True
    while changed:
        changed = False
        for _node, name, bases in class_info:
            if name in is_model:
                continue
            if "BaseModel" in bases:
                is_model[name] = True
                model_names.add(name)
                changed = True
            elif any(b in model_names for b in bases):
                is_model[name] = True
                model_names.add(name)
                changed = True

    # Second pass: extract fields for each model
    for node, name, bases in class_info:
        if name not in model_names:
            continue
        body = node.child_by_field_name("body")
        if body is None:
            continue
        fields = _extract_fields_from_body(body)
        has_validator = _has_model_validator(body)
        result.models[name] = PydanticModel(
            name=name, bases=bases, fields=fields, has_model_validator=has_validator
        )

    # Third pass: resolve inheritance -- merge parent fields into children
    if resolve:
        resolve_inheritance(result)

    return result


def resolve_inheritance(schema: ServiceSchema) -> None:
    """Merge parent model fields into child models."""
    resolved: set[str] = set()

    def resolve(name: str) -> list[ModelField]:
        if name in resolved:
            return schema.models[name].fields
        model = schema.models.get(name)
        if model is None:
            return []

        resolved.add(name)
        own_field_names = {f.name for f in model.fields}
        parent_fields: list[ModelField] = []

        for base in model.bases:
            if base == "BaseModel":
                continue
            if base in schema.models:
                pfields = resolve(base)
                for pf in pfields:
                    if pf.name not in own_field_names:
                        parent_fields.append(pf)
                        own_field_names.add(pf.name)

        model.fields = parent_fields + model.fields
        return model.fields

    for model_name in list(schema.models.keys()):
        resolve(model_name)
