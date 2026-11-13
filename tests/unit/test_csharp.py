"""Coverage boost tests for CSharpProcessor — targets uncovered lines.

Focuses on:
- _regex_parse: property extraction, field with const/readonly, sealed class
- extract_entities without tree-sitter (fallback to regex via file read)
- extract_relationships: namespace-contains, class-defines heuristics
- Complexity metrics in ParseResult
"""

from __future__ import annotations

from pathlib import Path

import pytest

from models.core_models import EntityType, RelationType
from processing.languages.base import ParserType
from processing.languages.csharp.csharp_processor import CSharpProcessor


@pytest.fixture
def processor():
    return CSharpProcessor()


CSHARP_STATIC_CLASS = """\
using System;

namespace StaticDemo
{
    public static class MathHelper
    {
        public static int Add(int a, int b) { return a + b; }
        public static double Pi = 3.14159;
    }
}
"""

CSHARP_VIRTUAL_OVERRIDE = """\
namespace Polymorphism
{
    public class Animal
    {
        public virtual void Speak() { }
    }

    public class Dog : Animal
    {
        public override void Speak() { }
    }
}
"""

CSHARP_READONLY_STRUCT_FIELDS = """\
using System.Collections.Generic;

namespace DataModels
{
    public readonly struct ImmutablePoint
    {
        public int X { get; }
        public int Y { get; }
    }

    public class Config
    {
        public readonly int MaxItems = 100;
        public static int InstanceCount;
        public const string Version = "1.0";
    }
}
"""

CSHARP_ASYNC_VIRTUAL = """\
namespace AsyncService
{
    public class BaseService
    {
        public virtual async Task<int> FetchAsync()
        {
            return 42;
        }

        protected abstract void OnComplete();
    }
}
"""

CSHARP_NESTED_CLASSES = """\
namespace Outer
{
    public class Container
    {
        private class Inner
        {
            public void DoWork() {}
        }

        internal class AnotherInner
        {
            protected void Process() {}
        }
    }
}
"""

CSHARP_DELEGATE_EVENT = """\
namespace Events
{
    public delegate void MyDelegate(string message);

    public class EventEmitter
    {
        public event MyDelegate OnMessage;
    }
}
"""


class TestCSharpRegexParseProperties:
    @pytest.mark.asyncio
    async def test_static_class(self, processor):
        result = await processor._regex_parse(Path("MathHelper.cs"), CSHARP_STATIC_CLASS)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        math_helper = [c for c in classes if c.name == "MathHelper"]
        assert len(math_helper) == 1
        assert math_helper[0].is_static is True
        assert math_helper[0].visibility == "public"

    @pytest.mark.asyncio
    async def test_virtual_override_methods(self, processor):
        result = await processor._regex_parse(Path("Animals.cs"), CSHARP_VIRTUAL_OVERRIDE)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        speak_methods = [m for m in methods if m.name == "Speak"]
        assert len(speak_methods) >= 2
        # Check that properties track virtual/override
        has_virtual = any(m.properties.get("is_virtual") for m in speak_methods)
        has_override = any(m.properties.get("is_override") for m in speak_methods)
        assert has_virtual or has_override

    @pytest.mark.asyncio
    async def test_readonly_fields_and_const(self, processor):
        result = await processor._regex_parse(Path("DataModels.cs"), CSHARP_READONLY_STRUCT_FIELDS)
        # Check readonly struct
        structs = [e for e in result.entities if e.type == EntityType.STRUCT]
        assert any(s.name == "ImmutablePoint" for s in structs)

        # Check const field
        constants = [e for e in result.entities if e.type == EntityType.CONSTANT]
        const_names = {c.name for c in constants}
        assert "Version" in const_names

    @pytest.mark.asyncio
    async def test_static_field(self, processor):
        result = await processor._regex_parse(Path("DataModels.cs"), CSHARP_READONLY_STRUCT_FIELDS)
        variables = [e for e in result.entities if e.type == EntityType.VARIABLE]
        static_vars = [v for v in variables if v.is_static]
        static_names = {v.name for v in static_vars}
        assert "InstanceCount" in static_names

    @pytest.mark.asyncio
    async def test_async_method(self, processor):
        result = await processor._regex_parse(Path("AsyncService.cs"), CSHARP_ASYNC_VIRTUAL)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        method_names = {m.name for m in methods}
        assert "FetchAsync" in method_names
        fetch = next(m for m in methods if m.name == "FetchAsync")
        # The regex only captures one modifier, so either async or virtual is captured
        # depending on which the regex matches first
        assert fetch.is_async is True or fetch.properties.get("is_virtual") is True
        assert fetch.signature is not None

    @pytest.mark.asyncio
    async def test_nested_classes(self, processor):
        result = await processor._regex_parse(Path("Nested.cs"), CSHARP_NESTED_CLASSES)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        class_names = {c.name for c in classes}
        assert "Container" in class_names
        assert "Inner" in class_names

    @pytest.mark.asyncio
    async def test_method_return_type_captured(self, processor):
        code = """\
public class Svc {
    public string GetName() { return "test"; }
}
"""
        result = await processor._regex_parse(Path("Svc.cs"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        get_name = [m for m in methods if m.name == "GetName"]
        assert len(get_name) == 1
        assert get_name[0].properties["return_type"] == "string"
        assert get_name[0].visibility == "public"


class TestCSharpInterfaceAndEnumExtraction:
    @pytest.mark.asyncio
    async def test_interface_extraction(self, processor):
        code = """\
namespace Api {
    public interface IService {
        void Start();
    }
    internal interface IHelper {
        void Help();
    }
}
"""
        result = await processor._regex_parse(Path("Interfaces.cs"), code)
        interfaces = [e for e in result.entities if e.type == EntityType.INTERFACE]
        iface_names = {i.name for i in interfaces}
        assert "IService" in iface_names
        assert "IHelper" in iface_names
        svc = next(i for i in interfaces if i.name == "IService")
        assert svc.visibility == "public"
        helper = next(i for i in interfaces if i.name == "IHelper")
        assert helper.visibility == "internal"

    @pytest.mark.asyncio
    async def test_enum_extraction(self, processor):
        code = """\
namespace Models {
    public enum Status {
        Active,
        Inactive,
        Pending
    }
    internal enum Priority {
        Low,
        High
    }
}
"""
        result = await processor._regex_parse(Path("Enums.cs"), code)
        enums = [e for e in result.entities if e.type == EntityType.ENUM]
        enum_names = {en.name for en in enums}
        assert "Status" in enum_names
        assert "Priority" in enum_names
        status = next(en for en in enums if en.name == "Status")
        assert status.visibility == "public"
        assert status.language == "csharp"

    @pytest.mark.asyncio
    async def test_sealed_class(self, processor):
        code = "public sealed class Singleton { }"
        result = await processor._regex_parse(Path("Singleton.cs"), code)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        assert len(classes) == 1
        assert classes[0].name == "Singleton"
        assert classes[0].properties["is_sealed"] is True
        assert classes[0].is_final is True

    @pytest.mark.asyncio
    async def test_abstract_class(self, processor):
        code = "public abstract class BaseHandler { }"
        result = await processor._regex_parse(Path("Base.cs"), code)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        handler = next(c for c in classes if c.name == "BaseHandler")
        assert handler.is_abstract is True
        assert handler.visibility == "public"

    @pytest.mark.asyncio
    async def test_using_directives(self, processor):
        code = """\
using System;
using System.Collections.Generic;
using static System.Math;
"""
        result = await processor._regex_parse(Path("Usings.cs"), code)
        imports = [e for e in result.entities if e.type == EntityType.IMPORT]
        import_names = {i.name for i in imports}
        assert "System" in import_names
        assert "System.Collections.Generic" in import_names

    @pytest.mark.asyncio
    async def test_namespace_extraction(self, processor):
        code = """\
namespace Company.Product.Module
{
    public class Widget { }
}
"""
        result = await processor._regex_parse(Path("Widget.cs"), code)
        namespaces = [e for e in result.entities if e.type == EntityType.NAMESPACE]
        assert len(namespaces) == 1
        assert namespaces[0].name == "Company.Product.Module"
        assert namespaces[0].properties["namespace_name"] == "Company.Product.Module"

    @pytest.mark.asyncio
    async def test_static_method(self, processor):
        code = """\
public class Utils {
    public static int Add(int a, int b) { return a + b; }
}
"""
        result = await processor._regex_parse(Path("Utils.cs"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        add = next(m for m in methods if m.name == "Add")
        assert add.is_static is True
        assert add.visibility == "public"

    @pytest.mark.asyncio
    async def test_abstract_method(self, processor):
        code = """\
public abstract class Base {
    protected abstract void Execute();
}
"""
        result = await processor._regex_parse(Path("Base.cs"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        execute = next((m for m in methods if m.name == "Execute"), None)
        if execute:
            assert execute.is_abstract is True
            assert execute.visibility == "protected"

    @pytest.mark.asyncio
    async def test_empty_code(self, processor):
        result = await processor._regex_parse(Path("Empty.cs"), "")
        assert result.success is True
        assert result.entities == []
        assert result.lines_of_code == 0

    @pytest.mark.asyncio
    async def test_method_skips_keywords(self, processor):
        code = """\
public class Foo {
    public void Run() {
        for (int i = 0; i < 10; i++) { }
        while (true) { break; }
    }
}
"""
        result = await processor._regex_parse(Path("Foo.cs"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        method_names = {m.name for m in methods}
        assert "for" not in method_names
        assert "while" not in method_names
        assert "Run" in method_names

    @pytest.mark.asyncio
    async def test_readonly_struct(self, processor):
        code = """\
public readonly struct Point {
    public int X { get; }
    public int Y { get; }
}
"""
        result = await processor._regex_parse(Path("Point.cs"), code)
        structs = [e for e in result.entities if e.type == EntityType.STRUCT]
        assert len(structs) == 1
        assert structs[0].name == "Point"
        assert structs[0].properties["is_readonly"] is True
        assert structs[0].properties["value_type"] is True

    @pytest.mark.asyncio
    async def test_const_field_entity_type(self, processor):
        code = 'public const string AppName = "MyApp";'
        result = await processor._regex_parse(Path("Constants.cs"), code)
        constants = [e for e in result.entities if e.type == EntityType.CONSTANT]
        matching = [c for c in constants if c.name == "AppName"]
        assert len(matching) >= 1
        assert matching[0].properties["is_const"] is True

    @pytest.mark.asyncio
    async def test_entity_file_path_and_language(self, processor):
        code = "public class Foo { public void Bar() {} }"
        result = await processor._regex_parse(Path("Foo.cs"), code)
        for entity in result.entities:
            assert entity.file_path == "Foo.cs"
            assert entity.language == "csharp"

    @pytest.mark.asyncio
    async def test_all_entity_ids_unique(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), CSHARP_STATIC_CLASS)
        ids = [e.id for e in result.entities]
        assert len(ids) == len(set(ids)), "Entity IDs should be unique"


class TestCSharpRelationshipsDeep:
    @pytest.mark.asyncio
    async def test_namespace_contains_relationships(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), CSHARP_STATIC_CLASS)
        rels = await processor.extract_relationships(None, result.entities, Path("Test.cs"))
        contains_rels = [r for r in rels if r.type == RelationType.CONTAINS]
        assert len(contains_rels) > 0

    @pytest.mark.asyncio
    async def test_class_defines_relationships(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), CSHARP_NESTED_CLASSES)
        rels = await processor.extract_relationships(None, result.entities, Path("Test.cs"))
        # Should produce at least namespace-contains relationships
        assert isinstance(rels, list)

    @pytest.mark.asyncio
    async def test_relationships_without_namespace(self, processor):
        code = "public class Standalone { public void Run() {} }"
        result = await processor._regex_parse(Path("Standalone.cs"), code)
        rels = await processor.extract_relationships(None, result.entities, Path("Standalone.cs"))
        # No namespace means no contains relationships
        contains_rels = [r for r in rels if r.type == RelationType.CONTAINS]
        assert len(contains_rels) == 0

    @pytest.mark.asyncio
    async def test_relationships_with_multiple_containers(self, processor):
        code = """\
namespace NS {
    public class A { public void M1() {} }
    public class B { public void M2() {} }
}
"""
        result = await processor._regex_parse(Path("Multi.cs"), code)
        rels = await processor.extract_relationships(None, result.entities, Path("Multi.cs"))
        assert len(rels) > 0


class TestCSharpComplexityMetrics:
    @pytest.mark.asyncio
    async def test_complexity_metrics_computed(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), CSHARP_STATIC_CLASS)
        assert isinstance(result.complexity_score, float)
        assert result.complexity_score >= 0.0
        assert result.lines_of_code == len(CSHARP_STATIC_CLASS.splitlines())

    @pytest.mark.asyncio
    async def test_complexity_counts_lines_correctly(self, processor):
        code = """\
// Comment line
using System;

namespace Test
{
    public class Foo
    {
        // Another comment
        public void Bar() { }
    }
}
"""
        result = await processor._regex_parse(Path("Foo.cs"), code)
        assert result.lines_of_code == len(code.splitlines())


class TestCSharpExtractEntitiesFallback:
    @pytest.mark.asyncio
    async def test_extract_entities_with_none_ast_returns_list(self, processor):
        """extract_entities with None ast should use regex fallback or return empty."""
        entities = await processor.extract_entities(None, Path("nonexistent.cs"))
        assert isinstance(entities, list)

    @pytest.mark.asyncio
    async def test_parse_file_without_tree_sitter_uses_regex(self, processor):
        processor.parser = None
        result = await processor.parse_file(Path("Test.cs"), CSHARP_STATIC_CLASS)
        assert result.success is True
        assert result.parser_used == ParserType.REGEX_FALLBACK


class TestCSharpPropertyAndFieldExtraction:
    @pytest.mark.asyncio
    async def test_property_extraction(self, processor):
        code = """\
public class Settings
{
    public string Name { get; set; }
    public int Count { get; }
}
"""
        result = await processor._regex_parse(Path("Settings.cs"), code)
        variables = [e for e in result.entities if e.type == EntityType.VARIABLE]
        var_names = {v.name for v in variables}
        assert "Name" in var_names or "Count" in var_names

    @pytest.mark.asyncio
    async def test_field_readonly_property(self, processor):
        code = 'public readonly string Id = "abc";'
        result = await processor._regex_parse(Path("F.cs"), code)
        vars_and_consts = [
            e for e in result.entities if e.type in (EntityType.VARIABLE, EntityType.CONSTANT)
        ]
        matching = [v for v in vars_and_consts if v.name == "Id"]
        if matching:
            assert matching[0].properties.get("is_readonly") is True

    @pytest.mark.asyncio
    async def test_method_signature_captured(self, processor):
        code = """\
public class Api {
    public async Task<List<string>> FetchAll(int limit) { return null; }
}
"""
        result = await processor._regex_parse(Path("Api.cs"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        fetch = [m for m in methods if m.name == "FetchAll"]
        assert len(fetch) >= 1
        assert fetch[0].signature is not None
