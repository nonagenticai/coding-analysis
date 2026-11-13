"""Tests for C# language processor regex-based parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from models.core_models import EntityType
from processing.languages.csharp.csharp_processor import CSharpProcessor


@pytest.fixture
def processor():
    return CSharpProcessor()


SAMPLE_CSHARP_CODE = """\
using System;
using System.Collections.Generic;
using static System.Math;

namespace MyApp.Services
{
    public abstract class UserService
    {
        private readonly string _name;
        public const int MaxRetries = 3;

        public async void ProcessUsers(List<string> users)
        {
            foreach (var user in users)
            {
                ValidateUser(user);
            }
        }

        private bool ValidateUser(string user)
        {
            return !string.IsNullOrEmpty(user);
        }

        public static int Calculate(int a, int b)
        {
            return a + b;
        }

        public string Name { get; set; }
    }

    public interface IUserRepository
    {
        void Save(string user);
    }

    public readonly struct Point
    {
        public int X { get; }
        public int Y { get; }
    }

    public enum UserStatus
    {
        Active,
        Inactive,
        Banned
    }
}
"""


@pytest.mark.asyncio
async def test_regex_parse_success(processor):
    fp = Path("UserService.cs")
    result = await processor._regex_parse(fp, SAMPLE_CSHARP_CODE)
    assert result.success is True
    assert result.language == "csharp"
    assert result.lines_of_code == len(SAMPLE_CSHARP_CODE.splitlines())


@pytest.mark.asyncio
async def test_regex_parse_extracts_namespace(processor):
    fp = Path("UserService.cs")
    result = await processor._regex_parse(fp, SAMPLE_CSHARP_CODE)
    namespaces = [e for e in result.entities if e.type.value == "namespace"]
    assert len(namespaces) >= 1
    ns_names = {e.name for e in namespaces}
    assert "MyApp.Services" in ns_names


@pytest.mark.asyncio
async def test_regex_parse_extracts_usings(processor):
    fp = Path("UserService.cs")
    result = await processor._regex_parse(fp, SAMPLE_CSHARP_CODE)
    usings = [e for e in result.entities if e.type.value == "import"]
    using_names = {e.name for e in usings}
    assert "System" in using_names
    assert "System.Collections.Generic" in using_names


@pytest.mark.asyncio
async def test_regex_parse_extracts_class(processor):
    fp = Path("UserService.cs")
    result = await processor._regex_parse(fp, SAMPLE_CSHARP_CODE)
    classes = [e for e in result.entities if e.type.value == "class"]
    class_names = {e.name for e in classes}
    assert "UserService" in class_names


@pytest.mark.asyncio
async def test_regex_parse_class_modifiers(processor):
    fp = Path("UserService.cs")
    result = await processor._regex_parse(fp, SAMPLE_CSHARP_CODE)
    classes = [e for e in result.entities if e.type.value == "class"]
    user_service = next((c for c in classes if c.name == "UserService"), None)
    assert user_service is not None
    assert user_service.visibility == "public"
    assert user_service.is_abstract is True


@pytest.mark.asyncio
async def test_regex_parse_extracts_interface(processor):
    fp = Path("UserService.cs")
    result = await processor._regex_parse(fp, SAMPLE_CSHARP_CODE)
    interfaces = [e for e in result.entities if e.type.value == "interface"]
    assert len(interfaces) >= 1
    iface_names = {e.name for e in interfaces}
    assert "IUserRepository" in iface_names


@pytest.mark.asyncio
async def test_regex_parse_extracts_struct(processor):
    fp = Path("UserService.cs")
    result = await processor._regex_parse(fp, SAMPLE_CSHARP_CODE)
    structs = [e for e in result.entities if e.type.value == "struct"]
    assert len(structs) >= 1
    struct_names = {e.name for e in structs}
    assert "Point" in struct_names


@pytest.mark.asyncio
async def test_regex_parse_struct_readonly(processor):
    fp = Path("UserService.cs")
    result = await processor._regex_parse(fp, SAMPLE_CSHARP_CODE)
    structs = [e for e in result.entities if e.type.value == "struct"]
    point = next((s for s in structs if s.name == "Point"), None)
    assert point is not None
    assert point.properties.get("is_readonly") is True


@pytest.mark.asyncio
async def test_regex_parse_extracts_enum(processor):
    fp = Path("UserService.cs")
    result = await processor._regex_parse(fp, SAMPLE_CSHARP_CODE)
    enums = [e for e in result.entities if e.type.value == "enum"]
    assert len(enums) >= 1
    enum_names = {e.name for e in enums}
    assert "UserStatus" in enum_names


@pytest.mark.asyncio
async def test_regex_parse_extracts_methods(processor):
    fp = Path("UserService.cs")
    result = await processor._regex_parse(fp, SAMPLE_CSHARP_CODE)
    funcs = [e for e in result.entities if e.type.value == "function"]
    func_names = {e.name for e in funcs}
    assert "ProcessUsers" in func_names
    assert "ValidateUser" in func_names
    assert "Calculate" in func_names


@pytest.mark.asyncio
async def test_regex_parse_method_skips_keywords(processor):
    fp = Path("UserService.cs")
    result = await processor._regex_parse(fp, SAMPLE_CSHARP_CODE)
    funcs = [e for e in result.entities if e.type.value == "function"]
    func_names = {e.name for e in funcs}
    assert "for" not in func_names
    assert "while" not in func_names
    assert "switch" not in func_names


@pytest.mark.asyncio
async def test_regex_parse_const_field(processor):
    fp = Path("UserService.cs")
    result = await processor._regex_parse(fp, SAMPLE_CSHARP_CODE)
    constants = [e for e in result.entities if e.type.value == "constant"]
    if constants:
        const_names = {e.name for e in constants}
        assert "MaxRetries" in const_names


@pytest.mark.asyncio
async def test_regex_parse_complexity_metrics(processor):
    fp = Path("UserService.cs")
    result = await processor._regex_parse(fp, SAMPLE_CSHARP_CODE)
    assert result.complexity_score is not None


@pytest.mark.asyncio
async def test_regex_parse_empty_code(processor):
    fp = Path("Empty.cs")
    result = await processor._regex_parse(fp, "")
    assert result.success is True
    assert len([e for e in result.entities if e.type.value == "namespace"]) == 0


@pytest.mark.asyncio
async def test_parse_file_without_tree_sitter(processor):
    fp = Path("Test.cs")
    result = await processor.parse_file(fp, SAMPLE_CSHARP_CODE)
    assert result.success is True
    assert len(result.entities) > 0


@pytest.mark.asyncio
async def test_extract_relationships(processor):
    fp = Path("Test.cs")
    result = await processor._regex_parse(fp, SAMPLE_CSHARP_CODE)
    relationships = await processor.extract_relationships(None, result.entities, fp)
    # Should produce namespace-contains and class-defines relationships
    assert isinstance(relationships, list)
    if len(result.entities) > 1:
        assert len(relationships) > 0


@pytest.mark.asyncio
async def test_initialize_impl(processor):
    await processor._initialize_impl()


def test_node_types_defined(processor):
    assert len(processor.CLASS_NODE_TYPES) > 0
    assert len(processor.METHOD_NODE_TYPES) > 0
    assert len(processor.FIELD_NODE_TYPES) > 0


# ---------------------------------------------------------------------------
# Tests merged from test_round4_csharp_processor.py
# ---------------------------------------------------------------------------

SAMPLE_CSHARP_R4 = """\
using System;
using System.Collections.Generic;
using static System.Math;

namespace MyApp.Services
{
    public abstract class UserService
    {
        private readonly string _name;
        public static int Counter;
        public const int MaxRetries = 3;

        public UserService(string name)
        {
            _name = name;
        }

        public virtual List<User> GetUsers()
        {
            return new List<User>();
        }

        private async void ProcessData(Dictionary<string, object> data)
        {
            // do stuff
        }

        protected static bool IsValid(string input)
        {
            return input != null;
        }

        public override string ToString()
        {
            return _name;
        }
    }

    public interface IRepository
    {
        void Save(object entity);
        object FindById(string id);
    }

    public readonly struct Point
    {
        public int X { get; }
        public int Y { get; }
    }

    public enum Status
    {
        Active,
        Inactive,
        Pending
    }

    internal sealed class Config
    {
        public string DbUrl { get; set; }
    }
}
"""


class TestCSharpRegexParse:
    """Test _regex_parse which covers the bulk of uncovered lines."""

    @pytest.mark.asyncio
    async def test_extracts_namespace(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        assert result.success is True
        assert result.language == "csharp"
        ns = [e for e in result.entities if e.type == EntityType.NAMESPACE]
        assert len(ns) == 1
        assert ns[0].name == "MyApp.Services"
        assert ns[0].properties["namespace_name"] == "MyApp.Services"

    @pytest.mark.asyncio
    async def test_extracts_using_statements(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        usings = [e for e in result.entities if e.type == EntityType.IMPORT]
        using_names = {e.name for e in usings}
        assert "System" in using_names
        assert "System.Collections.Generic" in using_names
        assert "System.Math" in using_names

    @pytest.mark.asyncio
    async def test_extracts_class(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        class_names = {c.name for c in classes}
        assert "UserService" in class_names

    @pytest.mark.asyncio
    async def test_class_modifiers(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        user_svc = [c for c in classes if c.name == "UserService"][0]
        assert user_svc.visibility == "public"
        assert user_svc.is_abstract is True

    @pytest.mark.asyncio
    async def test_sealed_class(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        config = [c for c in classes if c.name == "Config"]
        assert len(config) == 1
        assert config[0].properties["is_sealed"] is True

    @pytest.mark.asyncio
    async def test_extracts_interface(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        interfaces = [e for e in result.entities if e.type == EntityType.INTERFACE]
        assert any(i.name == "IRepository" for i in interfaces)
        iface = [i for i in interfaces if i.name == "IRepository"][0]
        assert iface.visibility == "public"
        assert iface.properties["interface_type"] is True

    @pytest.mark.asyncio
    async def test_extracts_struct(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        structs = [e for e in result.entities if e.type == EntityType.STRUCT]
        assert any(s.name == "Point" for s in structs)
        point = [s for s in structs if s.name == "Point"][0]
        assert point.properties["is_readonly"] is True
        assert point.properties["value_type"] is True

    @pytest.mark.asyncio
    async def test_extracts_enum(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        enums = [e for e in result.entities if e.type == EntityType.ENUM]
        assert any(e.name == "Status" for e in enums)
        status = [e for e in enums if e.name == "Status"][0]
        assert status.properties["enum_type"] is True

    @pytest.mark.asyncio
    async def test_extracts_methods(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        method_names = {m.name for m in methods}
        assert "GetUsers" in method_names
        assert "ProcessData" in method_names
        assert "IsValid" in method_names

    @pytest.mark.asyncio
    async def test_method_modifiers(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        is_valid = [m for m in methods if m.name == "IsValid"]
        assert len(is_valid) >= 1
        m = is_valid[0]
        assert m.is_static is True

    @pytest.mark.asyncio
    async def test_methods_skip_keywords(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        method_names = {m.name for m in methods}
        for kw in ["get", "set", "for", "while", "switch", "using", "var", "new"]:
            assert kw not in method_names

    @pytest.mark.asyncio
    async def test_extracts_fields(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        variables = [e for e in result.entities if e.type == EntityType.VARIABLE]
        assert len(variables) > 0
        assert all(v.name for v in variables)

    @pytest.mark.asyncio
    async def test_const_field(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        constants = [e for e in result.entities if e.type == EntityType.CONSTANT]
        const_names = {c.name for c in constants}
        assert "MaxRetries" in const_names

    @pytest.mark.asyncio
    async def test_complexity_metrics(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        assert result.lines_of_code == len(SAMPLE_CSHARP_R4.splitlines())

    @pytest.mark.asyncio
    async def test_parser_type(self, processor):
        from processing.languages.base import ParserType

        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        assert result.parser_used == ParserType.REGEX_FALLBACK

    @pytest.mark.asyncio
    async def test_empty_code(self, processor):
        result = await processor._regex_parse(Path("Empty.cs"), "")
        assert result.success is True
        assert len(result.entities) == 0

    @pytest.mark.asyncio
    async def test_parse_file_fallback(self, processor):
        """parse_file should fall back to regex when tree-sitter not available."""
        result = await processor.parse_file(Path("Test.cs"), SAMPLE_CSHARP_R4)
        assert result.success is True
        assert len(result.entities) > 0


class TestCSharpExtractRelationships:
    @pytest.mark.asyncio
    async def test_extract_relationships_namespace_contains(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        entities = result.entities
        rels = await processor.extract_relationships(None, entities, Path("Test.cs"))
        from models.core_models import RelationType

        contains_rels = [r for r in rels if r.type == RelationType.CONTAINS]
        assert len(contains_rels) > 0

    @pytest.mark.asyncio
    async def test_extract_relationships_defines(self, processor):
        result = await processor._regex_parse(Path("Test.cs"), SAMPLE_CSHARP_R4)
        entities = result.entities
        rels = await processor.extract_relationships(None, entities, Path("Test.cs"))
        from models.core_models import RelationType

        defines_rels = [r for r in rels if r.type == RelationType.DEFINES]
        assert len(defines_rels) > 0

    @pytest.mark.asyncio
    async def test_extract_relationships_empty(self, processor):
        rels = await processor.extract_relationships(None, [], Path("Test.cs"))
        assert rels == []

    @pytest.mark.asyncio
    async def test_extract_entities_fallback_regex(self, processor):
        """extract_entities without tree-sitter should fallback to regex."""
        entities = await processor.extract_entities(None, Path("Test.cs"))
        assert isinstance(entities, list)


class TestCSharpInit:
    def test_constructor_default_config(self):
        p = CSharpProcessor()
        assert p.config is not None

    def test_node_types(self):
        p = CSharpProcessor()
        assert "class_declaration" in p.CLASS_NODE_TYPES
        assert "method_declaration" in p.METHOD_NODE_TYPES
        assert "property_declaration" in p.PROPERTY_NODE_TYPES

    @pytest.mark.asyncio
    async def test_initialize_impl(self):
        p = CSharpProcessor()
        await p._initialize_impl()

    @pytest.mark.asyncio
    async def test_private_interface(self):
        code = "private interface IInternal { }"
        p = CSharpProcessor()
        result = await p._regex_parse(Path("Int.cs"), code)
        interfaces = [e for e in result.entities if e.type == EntityType.INTERFACE]
        assert any(i.name == "IInternal" for i in interfaces)

    @pytest.mark.asyncio
    async def test_internal_visibility_default(self):
        code = "class Foo { }"
        p = CSharpProcessor()
        result = await p._regex_parse(Path("Foo.cs"), code)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        foo = [c for c in classes if c.name == "Foo"][0]
        assert foo.visibility == "internal"

    @pytest.mark.asyncio
    async def test_readonly_field(self):
        code = "public readonly int Value = 5;"
        p = CSharpProcessor()
        result = await p._regex_parse(Path("F.cs"), code)
        variables = [
            e for e in result.entities if e.type in (EntityType.VARIABLE, EntityType.CONSTANT)
        ]
        matching = [v for v in variables if v.name == "Value"]
        assert len(matching) >= 1
        assert matching[0].properties.get("is_readonly") is True

    @pytest.mark.asyncio
    async def test_extract_entities_fallback_with_file(self, tmp_path):
        """extract_entities should fallback to regex when given None ast and real file."""
        code = """\
using System;

namespace Test {
    public class Foo {
        public void Bar() { }
    }
}
"""
        cs_file = tmp_path / "Test.cs"
        cs_file.write_text(code)
        p = CSharpProcessor()
        entities = await p.extract_entities(None, cs_file)
        assert isinstance(entities, list)
        assert len(entities) > 0
        entity_names = {e.name for e in entities}
        assert "Foo" in entity_names

    @pytest.mark.asyncio
    async def test_extract_entities_fallback_missing_file(self):
        """extract_entities should return empty list for missing file."""
        p = CSharpProcessor()
        entities = await p.extract_entities(None, Path("/nonexistent/Test.cs"))
        assert entities == []

    @pytest.mark.asyncio
    async def test_property_regex_extraction(self):
        code = """\
public class Config {
    public string Name { get; set; }
    public static int Count { get; }
    public virtual string Description { get; set; }
}
"""
        p = CSharpProcessor()
        result = await p._regex_parse(Path("Config.cs"), code)
        variables = [e for e in result.entities if e.type == EntityType.VARIABLE]
        # At least some properties should be captured
        assert len(variables) > 0
        assert any(v.name for v in variables)

    @pytest.mark.asyncio
    async def test_field_readonly_and_const_types(self):
        code = """\
public class Fields {
    private readonly string _name;
    public const int MAX = 100;
    public static int counter;
}
"""
        p = CSharpProcessor()
        result = await p._regex_parse(Path("Fields.cs"), code)
        constants = [e for e in result.entities if e.type == EntityType.CONSTANT]
        variables = [e for e in result.entities if e.type == EntityType.VARIABLE]
        const_names = {c.name for c in constants}
        assert "MAX" in const_names
        # readonly fields should be VARIABLE with is_readonly
        all_fields = constants + variables
        readonly_fields = [f for f in all_fields if f.properties.get("is_readonly")]
        assert len(readonly_fields) >= 1

    @pytest.mark.asyncio
    async def test_method_virtual_override_properties(self):
        code = """\
public class Base {
    public virtual void Process() { }
    public override string ToString() { return ""; }
}
"""
        p = CSharpProcessor()
        result = await p._regex_parse(Path("Base.cs"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        process = [m for m in methods if m.name == "Process"]
        if process:
            assert process[0].properties.get("is_virtual") is True
        tostring = [m for m in methods if m.name == "ToString"]
        if tostring:
            assert tostring[0].properties.get("is_override") is True

    @pytest.mark.asyncio
    async def test_static_method(self):
        code = """\
public class Util {
    public static void DoWork() { }
}
"""
        p = CSharpProcessor()
        result = await p._regex_parse(Path("Util.cs"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        do_work = [m for m in methods if m.name == "DoWork"]
        assert len(do_work) >= 1
        assert do_work[0].is_static is True

    @pytest.mark.asyncio
    async def test_async_method(self):
        code = """\
public class Svc {
    public async Task<int> FetchAsync() { return 1; }
}
"""
        p = CSharpProcessor()
        result = await p._regex_parse(Path("Svc.cs"), code)
        methods = [e for e in result.entities if e.type == EntityType.FUNCTION]
        fetch = [m for m in methods if m.name == "FetchAsync"]
        assert len(fetch) >= 1


# ---------------------------------------------------------------------------
# Tree-sitter parsing tests
# ---------------------------------------------------------------------------

from processing.languages.csharp.csharp_processor import TREE_SITTER_AVAILABLE

SAMPLE_CSHARP_TS = """\
using System;
using System.Collections.Generic;

namespace MyApp.Services
{
    public abstract class UserService
    {
        private readonly string _name;
        public const int MaxRetries = 3;

        public async void ProcessUsers(List<string> users)
        {
        }

        private bool ValidateUser(string user)
        {
            return !string.IsNullOrEmpty(user);
        }

        public static int Calculate(int a, int b)
        {
            return a + b;
        }

        public string Name { get; set; }
    }

    public interface IUserRepository
    {
        void Save(string user);
    }

    public readonly struct Point
    {
        public int X { get; }
        public int Y { get; }
    }

    public enum UserStatus
    {
        Active,
        Inactive,
        Banned
    }

    public delegate void EventHandler(object sender);
}
"""


@pytest.mark.skipif(not TREE_SITTER_AVAILABLE, reason="tree-sitter not installed")
class TestCSharpTreeSitterParsing:
    """Tests exercising tree-sitter parsing (not regex fallback)."""

    @pytest.fixture
    async def ts_processor(self):
        p = CSharpProcessor()
        await p._initialize_impl()
        return p

    @pytest.fixture
    def cs_file(self, tmp_path):
        fp = tmp_path / "Test.cs"
        fp.write_text(SAMPLE_CSHARP_TS)
        return fp

    @pytest.mark.asyncio
    async def test_tree_sitter_parse_success(self, ts_processor, cs_file):
        result = await ts_processor.parse_file(cs_file, SAMPLE_CSHARP_TS)
        assert result.success is True
        assert result.parser_used.value == "tree_sitter"
        assert len(result.entities) > 0

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_namespace(self, ts_processor, cs_file):
        result = await ts_processor.parse_file(cs_file, SAMPLE_CSHARP_TS)
        namespaces = [e for e in result.entities if e.type == EntityType.NAMESPACE]
        ns_names = {e.name for e in namespaces}
        assert "MyApp.Services" in ns_names

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_using(self, ts_processor, cs_file):
        result = await ts_processor.parse_file(cs_file, SAMPLE_CSHARP_TS)
        usings = [e for e in result.entities if e.type == EntityType.IMPORT]
        using_names = {e.name for e in usings}
        assert "System" in using_names
        assert "System.Collections.Generic" in using_names

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_class(self, ts_processor, cs_file):
        result = await ts_processor.parse_file(cs_file, SAMPLE_CSHARP_TS)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        class_names = {c.name for c in classes}
        assert "UserService" in class_names

    @pytest.mark.asyncio
    async def test_tree_sitter_class_modifiers(self, ts_processor, cs_file):
        result = await ts_processor.parse_file(cs_file, SAMPLE_CSHARP_TS)
        classes = [e for e in result.entities if e.type == EntityType.CLASS]
        user_svc = next((c for c in classes if c.name == "UserService"), None)
        assert user_svc is not None
        assert user_svc.visibility == "public"
        assert user_svc.is_abstract is True

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_interface(self, ts_processor, cs_file):
        result = await ts_processor.parse_file(cs_file, SAMPLE_CSHARP_TS)
        interfaces = [e for e in result.entities if e.type == EntityType.INTERFACE]
        iface_names = {i.name for i in interfaces}
        assert "IUserRepository" in iface_names

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_struct(self, ts_processor, cs_file):
        result = await ts_processor.parse_file(cs_file, SAMPLE_CSHARP_TS)
        structs = [e for e in result.entities if e.type == EntityType.STRUCT]
        struct_names = {s.name for s in structs}
        assert "Point" in struct_names

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_enum(self, ts_processor, cs_file):
        result = await ts_processor.parse_file(cs_file, SAMPLE_CSHARP_TS)
        enums = [e for e in result.entities if e.type == EntityType.ENUM]
        enum_names = {e.name for e in enums}
        assert "UserStatus" in enum_names

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_methods(self, ts_processor, cs_file):
        result = await ts_processor.parse_file(cs_file, SAMPLE_CSHARP_TS)
        funcs = [e for e in result.entities if e.type == EntityType.FUNCTION]
        func_names = {f.name for f in funcs}
        assert "ProcessUsers" in func_names
        assert "ValidateUser" in func_names
        assert "Calculate" in func_names

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_properties(self, ts_processor, cs_file):
        result = await ts_processor.parse_file(cs_file, SAMPLE_CSHARP_TS)
        variables = [e for e in result.entities if e.type == EntityType.VARIABLE]
        var_names = {v.name for v in variables}
        assert "Name" in var_names or len(variables) > 0

    @pytest.mark.asyncio
    async def test_tree_sitter_extracts_fields(self, ts_processor, cs_file):
        result = await ts_processor.parse_file(cs_file, SAMPLE_CSHARP_TS)
        all_vars = [
            e for e in result.entities if e.type in (EntityType.VARIABLE, EntityType.CONSTANT)
        ]
        assert len(all_vars) > 0

    @pytest.mark.asyncio
    async def test_tree_sitter_entity_line_numbers(self, ts_processor, cs_file):
        result = await ts_processor.parse_file(cs_file, SAMPLE_CSHARP_TS)
        for entity in result.entities:
            assert entity.line_number >= 1

    @pytest.mark.asyncio
    async def test_tree_sitter_entity_language(self, ts_processor, cs_file):
        result = await ts_processor.parse_file(cs_file, SAMPLE_CSHARP_TS)
        for entity in result.entities:
            assert entity.language == "csharp"

    @pytest.mark.asyncio
    async def test_tree_sitter_relationships(self, ts_processor, cs_file):
        result = await ts_processor.parse_file(cs_file, SAMPLE_CSHARP_TS)
        assert isinstance(result.relationships, list)

    @pytest.mark.asyncio
    async def test_tree_sitter_empty_code(self, ts_processor, tmp_path):
        fp = tmp_path / "Empty.cs"
        fp.write_text("")
        result = await ts_processor.parse_file(fp, "")
        assert result.success is True
        assert len(result.entities) == 0
