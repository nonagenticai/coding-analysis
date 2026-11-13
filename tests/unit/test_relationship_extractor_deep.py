"""Deep unit tests for RelationshipExtractor — targets 80%+ coverage."""

from __future__ import annotations

import ast
from typing import Any

import pytest

from models.core_models import CoreCodeEntity, EntityRelation, EntityType, RelationType
from processing.extractors.relationship_extractor import RelationshipExtractor
from processing.utils.ast_utils import ASTUtils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(
    name: str,
    entity_type: EntityType = EntityType.FUNCTION,
    entity_id: str | None = None,
    file_path: str = "test.py",
    line_number: int = 1,
    metadata: dict[str, Any] | None = None,
) -> CoreCodeEntity:
    """Create a minimal CoreCodeEntity for testing."""
    eid = entity_id or f"id_{name}"
    return CoreCodeEntity(
        id=eid,
        name=name,
        type=entity_type,
        language="python",
        file_path=file_path,
        line_number=line_number,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor() -> RelationshipExtractor:
    return RelationshipExtractor()


# ===========================================================================
# 1. Basic construction
# ===========================================================================


class TestRelationshipExtractorInit:
    def test_init_creates_ast_utils(self, extractor: RelationshipExtractor) -> None:
        assert isinstance(extractor.ast_utils, ASTUtils)


# ===========================================================================
# 2. extract_relationships — top-level coordinator
# ===========================================================================


class TestExtractRelationships:
    async def test_returns_empty_for_empty_entities(self, extractor: RelationshipExtractor) -> None:
        tree = ast.parse("")
        result = await extractor.extract_relationships(tree, [])
        assert result == []

    async def test_combines_all_relationship_types(self, extractor: RelationshipExtractor) -> None:
        code = """
import json

class Base:
    pass

class Child(Base):
    pass

def helper():
    pass

def caller():
    helper()
    x = Base
"""
        tree = ast.parse(code)
        entities = [
            _make_entity("json", EntityType.IMPORT, metadata={"full_name": "json"}),
            _make_entity("Base", EntityType.CLASS),
            _make_entity("Child", EntityType.CLASS),
            _make_entity("helper", EntityType.FUNCTION),
            _make_entity("caller", EntityType.FUNCTION),
            _make_entity("test", EntityType.MODULE),
        ]
        rels = await extractor.extract_relationships(tree, entities)
        types_found = {r.type for r in rels}
        # Should include at least inheritance and calls
        assert RelationType.INHERITANCE in types_found
        assert RelationType.CALLS in types_found

    async def test_no_duplicate_entity_lookup_keys(self, extractor: RelationshipExtractor) -> None:
        """When two entities share a name, last one wins in lookup (dict behavior)."""
        code = "class A: pass"
        tree = ast.parse(code)
        e1 = _make_entity("A", EntityType.CLASS, entity_id="id_A_1")
        e2 = _make_entity("A", EntityType.CLASS, entity_id="id_A_2")
        rels = await extractor.extract_relationships(tree, [e1, e2])
        # The last entity in the list overwrites; no crash
        assert isinstance(rels, list)


# ===========================================================================
# 3. Inheritance relationships
# ===========================================================================


class TestInheritanceRelationships:
    async def test_simple_inheritance(self, extractor: RelationshipExtractor) -> None:
        code = "class Child(Parent): pass"
        tree = ast.parse(code)
        entities = {
            "Child": _make_entity("Child", EntityType.CLASS),
            "Parent": _make_entity("Parent", EntityType.CLASS),
        }
        rels = await extractor._extract_inheritance_relationships(tree, entities)
        assert len(rels) == 1
        rel = rels[0]
        assert rel.type == RelationType.INHERITANCE
        assert rel.source_id == "id_Child"
        assert rel.target_id == "id_Parent"
        assert rel.metadata["source_name"] == "Child"
        assert rel.metadata["target_name"] == "Parent"

    async def test_multiple_inheritance(self, extractor: RelationshipExtractor) -> None:
        code = "class C(A, B): pass"
        tree = ast.parse(code)
        entities = {
            "C": _make_entity("C", EntityType.CLASS),
            "A": _make_entity("A", EntityType.CLASS),
            "B": _make_entity("B", EntityType.CLASS),
        }
        rels = await extractor._extract_inheritance_relationships(tree, entities)
        assert len(rels) == 2
        target_ids = {r.target_id for r in rels}
        assert target_ids == {"id_A", "id_B"}

    async def test_attribute_base_class(self, extractor: RelationshipExtractor) -> None:
        """class Child(mod.Parent) — base is an ast.Attribute."""
        code = "class Child(mod.Parent): pass"
        tree = ast.parse(code)
        entities = {
            "Child": _make_entity("Child", EntityType.CLASS),
            "mod.Parent": _make_entity("mod.Parent", EntityType.CLASS),
        }
        rels = await extractor._extract_inheritance_relationships(tree, entities)
        assert len(rels) == 1
        assert rels[0].target_id == "id_mod.Parent"

    async def test_class_not_in_lookup_skipped(self, extractor: RelationshipExtractor) -> None:
        code = "class Child(Parent): pass"
        tree = ast.parse(code)
        # Only Parent in lookup, not Child
        entities = {"Parent": _make_entity("Parent", EntityType.CLASS)}
        rels = await extractor._extract_inheritance_relationships(tree, entities)
        assert rels == []

    async def test_base_not_in_lookup_skipped(self, extractor: RelationshipExtractor) -> None:
        code = "class Child(ExternalBase): pass"
        tree = ast.parse(code)
        entities = {"Child": _make_entity("Child", EntityType.CLASS)}
        rels = await extractor._extract_inheritance_relationships(tree, entities)
        assert rels == []

    async def test_no_base_classes(self, extractor: RelationshipExtractor) -> None:
        code = "class Standalone: pass"
        tree = ast.parse(code)
        entities = {"Standalone": _make_entity("Standalone", EntityType.CLASS)}
        rels = await extractor._extract_inheritance_relationships(tree, entities)
        assert rels == []

    async def test_complex_base_expression_ignored(self, extractor: RelationshipExtractor) -> None:
        """Base class as a subscript like Generic[T] is neither Name nor Attribute."""
        code = "class Foo(list[int]): pass"
        tree = ast.parse(code)
        entities = {"Foo": _make_entity("Foo", EntityType.CLASS)}
        rels = await extractor._extract_inheritance_relationships(tree, entities)
        assert rels == []

    async def test_inheritance_metadata_has_line_number(
        self, extractor: RelationshipExtractor
    ) -> None:
        code = "\n\nclass Child(Parent): pass"
        tree = ast.parse(code)
        entities = {
            "Child": _make_entity("Child", EntityType.CLASS),
            "Parent": _make_entity("Parent", EntityType.CLASS),
        }
        rels = await extractor._extract_inheritance_relationships(tree, entities)
        assert rels[0].metadata["line_number"] == 3


# ===========================================================================
# 4. Function call relationships
# ===========================================================================


class TestFunctionCallRelationships:
    async def test_direct_function_call(self, extractor: RelationshipExtractor) -> None:
        code = """
def helper():
    pass

def main():
    helper()
"""
        tree = ast.parse(code)
        entities = {
            "helper": _make_entity("helper"),
            "main": _make_entity("main"),
        }
        rels = await extractor._extract_function_call_relationships(tree, entities)
        assert len(rels) == 1
        assert rels[0].type == RelationType.CALLS
        assert rels[0].source_id == "id_main"
        assert rels[0].target_id == "id_helper"

    async def test_no_call_outside_function(self, extractor: RelationshipExtractor) -> None:
        """Top-level calls (outside any function) should be ignored."""
        code = "helper()"
        tree = ast.parse(code)
        entities = {"helper": _make_entity("helper")}
        rels = await extractor._extract_function_call_relationships(tree, entities)
        assert rels == []

    async def test_call_to_unknown_function_ignored(self, extractor: RelationshipExtractor) -> None:
        code = """
def main():
    unknown()
"""
        tree = ast.parse(code)
        entities = {"main": _make_entity("main")}
        rels = await extractor._extract_function_call_relationships(tree, entities)
        assert rels == []

    async def test_method_call_via_attribute(self, extractor: RelationshipExtractor) -> None:
        """obj.method() call where the full attribute name is in the lookup."""
        code = """
def caller():
    obj.save()
"""
        tree = ast.parse(code)
        entities = {
            "caller": _make_entity("caller"),
            "obj.save": _make_entity("obj.save"),
        }
        rels = await extractor._extract_function_call_relationships(tree, entities)
        assert len(rels) == 1
        assert rels[0].target_id == "id_obj.save"

    async def test_async_function_def_calls(self, extractor: RelationshipExtractor) -> None:
        code = """
async def fetch():
    pass

async def run():
    fetch()
"""
        tree = ast.parse(code)
        entities = {
            "fetch": _make_entity("fetch"),
            "run": _make_entity("run"),
        }
        rels = await extractor._extract_function_call_relationships(tree, entities)
        assert len(rels) == 1
        assert rels[0].source_id == "id_run"

    async def test_nested_function_context_restored(self, extractor: RelationshipExtractor) -> None:
        """Inner function calls should be attributed to the inner function."""
        code = """
def outer():
    def inner():
        target()
    other()
"""
        tree = ast.parse(code)
        entities = {
            "outer": _make_entity("outer"),
            "inner": _make_entity("inner"),
            "target": _make_entity("target"),
            "other": _make_entity("other"),
        }
        rels = await extractor._extract_function_call_relationships(tree, entities)
        sources_targets = [(r.source_id, r.target_id) for r in rels]
        assert ("id_inner", "id_target") in sources_targets
        assert ("id_outer", "id_other") in sources_targets

    async def test_call_name_returns_none_for_complex_node(
        self, extractor: RelationshipExtractor
    ) -> None:
        """Call where func is not Name or Attribute (e.g., subscript)."""
        code = """
def f():
    funcs[0]()
"""
        tree = ast.parse(code)
        entities = {"f": _make_entity("f")}
        rels = await extractor._extract_function_call_relationships(tree, entities)
        assert rels == []

    async def test_multiple_calls_in_one_function(self, extractor: RelationshipExtractor) -> None:
        code = """
def workflow():
    a()
    b()
    c()
"""
        tree = ast.parse(code)
        entities = {
            "workflow": _make_entity("workflow"),
            "a": _make_entity("a"),
            "b": _make_entity("b"),
            "c": _make_entity("c"),
        }
        rels = await extractor._extract_function_call_relationships(tree, entities)
        assert len(rels) == 3

    async def test_recursive_call(self, extractor: RelationshipExtractor) -> None:
        code = """
def recurse():
    recurse()
"""
        tree = ast.parse(code)
        entities = {"recurse": _make_entity("recurse")}
        rels = await extractor._extract_function_call_relationships(tree, entities)
        assert len(rels) == 1
        assert rels[0].source_id == rels[0].target_id


# ===========================================================================
# 5. Import relationships
# ===========================================================================


class TestImportRelationships:
    async def test_simple_import(self, extractor: RelationshipExtractor) -> None:
        code = "import json"
        tree = ast.parse(code)
        entities = {
            "mymodule": _make_entity("mymodule", EntityType.MODULE),
            "json": _make_entity("json", EntityType.IMPORT),
        }
        rels = await extractor._extract_import_relationships(tree, entities)
        assert len(rels) == 1
        assert rels[0].type == RelationType.IMPORT
        assert rels[0].metadata["import_type"] == "import"

    async def test_from_import(self, extractor: RelationshipExtractor) -> None:
        code = "from os import path"
        tree = ast.parse(code)
        entities = {
            "mod": _make_entity("mod", EntityType.MODULE),
            "os.path": _make_entity("os.path", EntityType.IMPORT),
        }
        rels = await extractor._extract_import_relationships(tree, entities)
        assert len(rels) == 1
        assert rels[0].metadata["import_type"] == "from_import"

    async def test_from_import_no_module_prefix(self, extractor: RelationshipExtractor) -> None:
        """from . import something — node.module is None."""
        code = "from . import something"
        tree = ast.parse(code)
        entities = {
            "mod": _make_entity("mod", EntityType.MODULE),
            "something": _make_entity("something", EntityType.IMPORT),
        }
        rels = await extractor._extract_import_relationships(tree, entities)
        assert len(rels) == 1

    async def test_no_module_entity_returns_empty(self, extractor: RelationshipExtractor) -> None:
        code = "import os"
        tree = ast.parse(code)
        # No module entity in lookup
        entities = {"os": _make_entity("os", EntityType.IMPORT)}
        rels = await extractor._extract_import_relationships(tree, entities)
        assert rels == []

    async def test_import_entity_not_in_lookup_skipped(
        self, extractor: RelationshipExtractor
    ) -> None:
        code = "import unknown_lib"
        tree = ast.parse(code)
        entities = {"mod": _make_entity("mod", EntityType.MODULE)}
        rels = await extractor._extract_import_relationships(tree, entities)
        assert rels == []

    async def test_import_matched_by_full_name_metadata(
        self, extractor: RelationshipExtractor
    ) -> None:
        """Import entity matched via metadata.full_name instead of name."""
        code = "import foo"
        tree = ast.parse(code)
        entities = {
            "mod": _make_entity("mod", EntityType.MODULE),
            "foo_alias": _make_entity(
                "foo_alias",
                EntityType.IMPORT,
                metadata={"full_name": "foo"},
            ),
        }
        rels = await extractor._extract_import_relationships(tree, entities)
        assert len(rels) == 1

    async def test_multiple_imports(self, extractor: RelationshipExtractor) -> None:
        code = "import os, sys"
        tree = ast.parse(code)
        entities = {
            "mod": _make_entity("mod", EntityType.MODULE),
            "os": _make_entity("os", EntityType.IMPORT),
            "sys": _make_entity("sys", EntityType.IMPORT),
        }
        rels = await extractor._extract_import_relationships(tree, entities)
        assert len(rels) == 2


# ===========================================================================
# 6. Usage relationships
# ===========================================================================


class TestUsageRelationships:
    async def test_variable_usage_in_function(self, extractor: RelationshipExtractor) -> None:
        code = """
x = 1

def f():
    return x
"""
        tree = ast.parse(code)
        entities = {
            "x": _make_entity("x", EntityType.VARIABLE),
            "f": _make_entity("f"),
        }
        rels = await extractor._extract_usage_relationships(tree, entities)
        assert len(rels) >= 1
        assert any(r.target_id == "id_x" and r.source_id == "id_f" for r in rels)

    async def test_self_reference_filtered(self, extractor: RelationshipExtractor) -> None:
        code = """
def f():
    f
"""
        tree = ast.parse(code)
        entities = {"f": _make_entity("f")}
        rels = await extractor._extract_usage_relationships(tree, entities)
        # Self-reference should be filtered
        assert rels == []

    async def test_usage_outside_context_ignored(self, extractor: RelationshipExtractor) -> None:
        """Name at module level (no function/class context) should be ignored."""
        code = "x"
        tree = ast.parse(code)
        entities = {"x": _make_entity("x", EntityType.VARIABLE)}
        rels = await extractor._extract_usage_relationships(tree, entities)
        assert rels == []

    async def test_store_context_ignored(self, extractor: RelationshipExtractor) -> None:
        """Variable assignment (Store) should not create usage relationship."""
        code = """
def f():
    x = 1
"""
        tree = ast.parse(code)
        entities = {
            "x": _make_entity("x", EntityType.VARIABLE),
            "f": _make_entity("f"),
        }
        rels = await extractor._extract_usage_relationships(tree, entities)
        # 'x' in Store context should not generate usage
        assert not any(r.target_id == "id_x" for r in rels)

    async def test_usage_in_class_body(self, extractor: RelationshipExtractor) -> None:
        code = """
shared = 1

class MyClass:
    val = shared
"""
        tree = ast.parse(code)
        entities = {
            "shared": _make_entity("shared", EntityType.VARIABLE),
            "MyClass": _make_entity("MyClass", EntityType.CLASS),
        }
        rels = await extractor._extract_usage_relationships(tree, entities)
        assert len(rels) >= 1
        assert any(r.source_id == "id_MyClass" for r in rels)

    async def test_async_function_usage(self, extractor: RelationshipExtractor) -> None:
        code = """
data = []

async def process():
    return data
"""
        tree = ast.parse(code)
        entities = {
            "data": _make_entity("data", EntityType.VARIABLE),
            "process": _make_entity("process"),
        }
        rels = await extractor._extract_usage_relationships(tree, entities)
        assert any(r.source_id == "id_process" and r.target_id == "id_data" for r in rels)

    async def test_usage_metadata_fields(self, extractor: RelationshipExtractor) -> None:
        code = """
val = 42

def f():
    return val
"""
        tree = ast.parse(code)
        entities = {
            "val": _make_entity("val", EntityType.VARIABLE),
            "f": _make_entity("f"),
        }
        rels = await extractor._extract_usage_relationships(tree, entities)
        rel = rels[0]
        assert rel.metadata["usage_type"] == "variable_reference"
        assert "line_number" in rel.metadata
        assert rel.metadata["source_name"] == "f"
        assert rel.metadata["target_name"] == "val"

    async def test_nested_function_context_for_usage(
        self, extractor: RelationshipExtractor
    ) -> None:
        code = """
val = 1

def outer():
    def inner():
        return val
    return val
"""
        tree = ast.parse(code)
        entities = {
            "val": _make_entity("val", EntityType.VARIABLE),
            "outer": _make_entity("outer"),
            "inner": _make_entity("inner"),
        }
        rels = await extractor._extract_usage_relationships(tree, entities)
        sources = {r.source_id for r in rels if r.target_id == "id_val"}
        assert "id_inner" in sources
        assert "id_outer" in sources

    async def test_unknown_name_ignored(self, extractor: RelationshipExtractor) -> None:
        code = """
def f():
    return unknown_var
"""
        tree = ast.parse(code)
        entities = {"f": _make_entity("f")}
        rels = await extractor._extract_usage_relationships(tree, entities)
        assert rels == []


# ===========================================================================
# 7. Composition relationships
# ===========================================================================


class TestCompositionRelationships:
    async def test_simple_composition(self, extractor: RelationshipExtractor) -> None:
        code = """
class Engine:
    pass

class Car:
    def __init__(self):
        self.engine = Engine()
"""
        tree = ast.parse(code)
        entities = {
            "Engine": _make_entity("Engine", EntityType.CLASS),
            "Car": _make_entity("Car", EntityType.CLASS),
        }
        rels = await extractor.extract_composition_relationships(tree, entities)
        assert len(rels) == 1
        rel = rels[0]
        assert rel.type == RelationType.COMPOSITION
        assert rel.source_id == "id_Car"
        assert rel.target_id == "id_Engine"
        assert rel.metadata["attribute_name"] == "engine"

    async def test_class_not_in_lookup_skipped(self, extractor: RelationshipExtractor) -> None:
        code = """
class Car:
    def __init__(self):
        self.engine = Engine()
"""
        tree = ast.parse(code)
        entities = {"Engine": _make_entity("Engine", EntityType.CLASS)}
        rels = await extractor.extract_composition_relationships(tree, entities)
        assert rels == []

    async def test_composed_class_not_in_lookup_skipped(
        self, extractor: RelationshipExtractor
    ) -> None:
        code = """
class Car:
    def __init__(self):
        self.engine = Engine()
"""
        tree = ast.parse(code)
        entities = {"Car": _make_entity("Car", EntityType.CLASS)}
        rels = await extractor.extract_composition_relationships(tree, entities)
        assert rels == []

    async def test_non_init_method_ignored(self, extractor: RelationshipExtractor) -> None:
        """Only __init__ body should be scanned for composition."""
        code = """
class Car:
    def setup(self):
        self.engine = Engine()
"""
        tree = ast.parse(code)
        entities = {
            "Car": _make_entity("Car", EntityType.CLASS),
            "Engine": _make_entity("Engine", EntityType.CLASS),
        }
        rels = await extractor.extract_composition_relationships(tree, entities)
        assert rels == []

    async def test_non_self_attribute_ignored(self, extractor: RelationshipExtractor) -> None:
        """cls.engine = Engine() should be ignored (target.value.id != 'self')."""
        code = """
class Car:
    def __init__(self):
        cls = type(self)
        cls.engine = Engine()
"""
        tree = ast.parse(code)
        entities = {
            "Car": _make_entity("Car", EntityType.CLASS),
            "Engine": _make_entity("Engine", EntityType.CLASS),
        }
        rels = await extractor.extract_composition_relationships(tree, entities)
        assert rels == []

    async def test_non_call_assignment_ignored(self, extractor: RelationshipExtractor) -> None:
        """self.x = 42 (not a call) should be ignored."""
        code = """
class Car:
    def __init__(self):
        self.x = 42
"""
        tree = ast.parse(code)
        entities = {"Car": _make_entity("Car", EntityType.CLASS)}
        rels = await extractor.extract_composition_relationships(tree, entities)
        assert rels == []

    async def test_attribute_call_func_ignored(self, extractor: RelationshipExtractor) -> None:
        """self.x = mod.Engine() — func is Attribute, not Name — should not match."""
        code = """
class Car:
    def __init__(self):
        self.engine = mod.Engine()
"""
        tree = ast.parse(code)
        entities = {
            "Car": _make_entity("Car", EntityType.CLASS),
            "mod.Engine": _make_entity("mod.Engine", EntityType.CLASS),
        }
        rels = await extractor.extract_composition_relationships(tree, entities)
        assert rels == []

    async def test_multiple_compositions(self, extractor: RelationshipExtractor) -> None:
        code = """
class Wheel:
    pass

class Engine:
    pass

class Car:
    def __init__(self):
        self.engine = Engine()
        self.wheel = Wheel()
"""
        tree = ast.parse(code)
        entities = {
            "Car": _make_entity("Car", EntityType.CLASS),
            "Engine": _make_entity("Engine", EntityType.CLASS),
            "Wheel": _make_entity("Wheel", EntityType.CLASS),
        }
        rels = await extractor.extract_composition_relationships(tree, entities)
        assert len(rels) == 2
        target_ids = {r.target_id for r in rels}
        assert target_ids == {"id_Engine", "id_Wheel"}

    async def test_composition_metadata_has_line_number(
        self, extractor: RelationshipExtractor
    ) -> None:
        code = """
class Engine:
    pass

class Car:
    def __init__(self):
        self.engine = Engine()
"""
        tree = ast.parse(code)
        entities = {
            "Car": _make_entity("Car", EntityType.CLASS),
            "Engine": _make_entity("Engine", EntityType.CLASS),
        }
        rels = await extractor.extract_composition_relationships(tree, entities)
        assert "line_number" in rels[0].metadata

    async def test_empty_init_no_composition(self, extractor: RelationshipExtractor) -> None:
        code = """
class Car:
    def __init__(self):
        pass
"""
        tree = ast.parse(code)
        entities = {"Car": _make_entity("Car", EntityType.CLASS)}
        rels = await extractor.extract_composition_relationships(tree, entities)
        assert rels == []

    async def test_no_classes_returns_empty(self, extractor: RelationshipExtractor) -> None:
        code = "def f(): pass"
        tree = ast.parse(code)
        entities = {"f": _make_entity("f")}
        rels = await extractor.extract_composition_relationships(tree, entities)
        assert rels == []


# ===========================================================================
# 8. Edge cases & integration-style tests
# ===========================================================================


class TestEdgeCases:
    async def test_empty_source_code(self, extractor: RelationshipExtractor) -> None:
        tree = ast.parse("")
        rels = await extractor.extract_relationships(tree, [])
        assert rels == []

    async def test_all_rel_types_return_entity_relation_objects(
        self, extractor: RelationshipExtractor
    ) -> None:
        code = """
import os

class Base:
    pass

class Child(Base):
    def __init__(self):
        self.base = Base()

def helper():
    pass

def main():
    helper()
    x = Base
"""
        tree = ast.parse(code)
        entities = [
            _make_entity("os", EntityType.IMPORT),
            _make_entity("Base", EntityType.CLASS),
            _make_entity("Child", EntityType.CLASS),
            _make_entity("helper"),
            _make_entity("main"),
            _make_entity("test_module", EntityType.MODULE),
        ]
        rels = await extractor.extract_relationships(tree, entities)
        for r in rels:
            assert isinstance(r, EntityRelation)

    async def test_deeply_nested_attribute_base(self, extractor: RelationshipExtractor) -> None:
        """class X(a.b.c.D) — deeply nested attribute name."""
        code = "class X(a.b.c.D): pass"
        tree = ast.parse(code)
        entities = {
            "X": _make_entity("X", EntityType.CLASS),
            "a.b.c.D": _make_entity("a.b.c.D", EntityType.CLASS),
        }
        rels = await extractor._extract_inheritance_relationships(tree, entities)
        assert len(rels) == 1
        assert rels[0].target_id == "id_a.b.c.D"

    async def test_function_calling_function_not_in_lookup(
        self, extractor: RelationshipExtractor
    ) -> None:
        code = """
def main():
    external_lib_call()
"""
        tree = ast.parse(code)
        entities = {"main": _make_entity("main")}
        rels = await extractor._extract_function_call_relationships(tree, entities)
        assert rels == []

    async def test_class_with_no_body_functions(self, extractor: RelationshipExtractor) -> None:
        """Class with only pass statement — no methods at all."""
        code = "class Empty: pass"
        tree = ast.parse(code)
        entities = {"Empty": _make_entity("Empty", EntityType.CLASS)}
        rels = await extractor.extract_composition_relationships(tree, entities)
        assert rels == []
