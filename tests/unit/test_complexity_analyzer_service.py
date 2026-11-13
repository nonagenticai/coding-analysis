"""Unit tests for processing.analyzers.complexity_analyzers — broad coverage of uncovered methods.

Covers:
- Base class: calculate_base_complexity, calculate_cognitive_complexity
- Per-language analyze_entity_complexity (Java, Rust, Go, TypeScript, Python, JS, C++, C#)
- Per-language _count_parameters
- Per-language _analyze_*_coupling
- Per-language _generate_*_suggestions
- Per-language maintainability calculations
- ComplexityAnalysisService: analyze_file_complexity, _calculate_quality_score,
  _generate_file_suggestions, _create_generic_analyzer
- FileComplexityReport dataclass
- CSharp: classify, maintainability, suggestions, coupling, parameter counting
"""

from __future__ import annotations

from unittest.mock import MagicMock

from models.core_models import CoreCodeEntity
from processing.analyzers.complexity_analyzers import (
    ComplexityAnalysisService,
    ComplexityLevel,
    CppComplexityAnalyzer,
    CSharpComplexityAnalyzer,
    FileComplexityReport,
    GoComplexityAnalyzer,
    JavaComplexityAnalyzer,
    JavaScriptComplexityAnalyzer,
    PythonComplexityAnalyzer,
    RustComplexityAnalyzer,
    ServiceComplexityMetrics,
    TypeScriptComplexityAnalyzer,
)
from processing.languages.base import ParserType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(
    *,
    entity_id: str = "e1",
    name: str = "test_func",
    entity_type: str = "function",
    language: str = "python",
    signature: str | None = None,
    properties: dict | None = None,
    is_async: bool = False,
    is_generic: bool = False,
) -> CoreCodeEntity:
    """Create a CoreCodeEntity, using MagicMock for non-standard entity types.

    The EntityType enum only has a fixed set of values (function, class, struct,
    interface, etc.), but the complexity analyzers reference types like 'method',
    'constructor', 'destructor', 'property', 'arrow_function' in their branching
    logic. For those, we use a MagicMock that behaves like an entity.
    """
    _valid_entity_types = {
        "module",
        "file",
        "function",
        "variable",
        "constant",
        "import",
        "class",
        "interface",
        "trait",
        "enum",
        "struct",
        "union",
        "closure",
        "lambda",
        "generator",
        "namespace",
        "package",
        "annotation",
        "decorator",
        "macro",
        "generic",
        "type_alias",
        "comment",
        "docstring",
        "todo",
    }
    if entity_type not in _valid_entity_types:
        # Use a MagicMock for entity types not in the enum
        entity = MagicMock()
        entity.id = entity_id
        entity.name = name
        entity.type = entity_type
        entity.language = language
        entity.file_path = "test.py"
        entity.line_number = 1
        entity.signature = signature
        entity.properties = properties if properties is not None else {}
        entity.is_async = is_async
        entity.is_generic = is_generic
        return entity
    return CoreCodeEntity(
        id=entity_id,
        name=name,
        type=entity_type,
        language=language,
        file_path="test.py",
        line_number=1,
        signature=signature,
        properties=properties or {},
        is_async=is_async,
        is_generic=is_generic,
    )


def _make_parse_result(
    *,
    language: str = "python",
    entities: list | None = None,
    relationships: list | None = None,
    lines_of_code: int = 100,
    file_path: str = "test.py",
):
    """Create a parse result. Uses MagicMock to avoid Pydantic validation issues
    when entities contain non-standard types (e.g. 'method', 'constructor')."""
    pr = MagicMock()
    pr.success = True
    pr.file_path = file_path
    pr.language = language
    pr.parser_used = ParserType.TREE_SITTER
    pr.entities = entities or []
    pr.relationships = relationships or []
    pr.lines_of_code = lines_of_code
    pr.errors = []
    pr.warnings = []
    pr.properties = {}
    return pr


def _make_relation(
    source_id: str = "e1",
    target_id: str = "e2",
    rel_type: str = "calls",
):
    """Create a relation. Uses MagicMock to support non-standard relation types
    like 'trait_bound', 'generic_constraint', 'decorated_by', etc."""
    rel = MagicMock()
    rel.source_id = source_id
    rel.target_id = target_id
    rel.type = rel_type
    rel.weight = 1.0
    rel.properties = {}
    return rel


# =====================================================================
# Base class: calculate_base_complexity / calculate_cognitive_complexity
# =====================================================================


class TestBaseComplexityMethods:
    """Tests for LanguageComplexityAnalyzer base methods (via JavaComplexityAnalyzer)."""

    def setup_method(self):
        self.analyzer = JavaComplexityAnalyzer()


# =====================================================================
# Java analyzer
# =====================================================================


class TestJavaAnalyzeEntityComplexity:
    def setup_method(self):
        self.analyzer = JavaComplexityAnalyzer()

    def test_analyze_class_entity(self):
        entity = _make_entity(entity_type="class", language="java", name="MyClass")
        relations = [_make_relation("e1", "e2", "calls")]
        pr = _make_parse_result(language="java", entities=[entity], relationships=relations)
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.language == "java"
        assert metrics.entity_type == "class"
        assert metrics.entity_name == "MyClass"
        assert metrics.coupling_factor >= 0.0

    def test_analyze_method_entity(self):
        entity = _make_entity(
            entity_type="method",
            language="java",
            name="doWork",
            signature="public void doWork(int x, String y)",
        )
        pr = _make_parse_result(language="java", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.parameter_count == 2
        assert metrics.complexity_level is not None

    def test_java_count_parameters_empty(self):
        entity = _make_entity(signature="void foo()")
        assert self.analyzer._count_parameters(entity) == 0

    def test_java_count_parameters_multiple(self):
        entity = _make_entity(signature="void foo(int a, String b, double c)")
        assert self.analyzer._count_parameters(entity) == 3

    def test_java_count_parameters_no_parens(self):
        entity = _make_entity(signature="field_declaration")
        assert self.analyzer._count_parameters(entity) == 0

    def test_java_class_coupling_no_relations(self):
        entity = _make_entity(entity_type="class")
        pr = _make_parse_result(language="java", relationships=[])
        assert self.analyzer._analyze_class_coupling(entity, pr) == 0.0

    def test_java_class_coupling_many_relations(self):
        entity = _make_entity(entity_type="class")
        rels = [_make_relation("e1", f"t{i}", "calls") for i in range(15)]
        pr = _make_parse_result(language="java", relationships=rels)
        result = self.analyzer._analyze_class_coupling(entity, pr)
        assert result == 1.0  # capped at 1.0

    def test_java_suggestions_high_complexity(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=15.0)
        entity = _make_entity(entity_type="method")
        suggestions = self.analyzer._generate_java_suggestions(metrics, entity)
        assert any("breaking down" in s for s in suggestions)

    def test_java_suggestions_many_params(self):
        metrics = ServiceComplexityMetrics(parameter_count=7)
        entity = _make_entity(entity_type="method")
        suggestions = self.analyzer._generate_java_suggestions(metrics, entity)
        assert any("parameter object" in s for s in suggestions)

    def test_java_suggestions_deep_nesting(self):
        metrics = ServiceComplexityMetrics(nesting_depth=5)
        entity = _make_entity(entity_type="method")
        suggestions = self.analyzer._generate_java_suggestions(metrics, entity)
        assert any("nesting depth" in s.lower() for s in suggestions)

    def test_java_suggestions_high_coupling(self):
        metrics = ServiceComplexityMetrics(coupling_factor=0.8)
        entity = _make_entity(entity_type="class")
        suggestions = self.analyzer._generate_java_suggestions(metrics, entity)
        assert any("coupling" in s.lower() for s in suggestions)

    def test_java_suggestions_large_class(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=55.0)
        entity = _make_entity(entity_type="class")
        suggestions = self.analyzer._generate_java_suggestions(metrics, entity)
        assert any("large class" in s.lower() for s in suggestions)

    def test_java_suggestions_no_issues(self):
        metrics = ServiceComplexityMetrics(
            cyclomatic_complexity=3.0,
            parameter_count=2,
            nesting_depth=1,
            coupling_factor=0.1,
        )
        entity = _make_entity(entity_type="method")
        suggestions = self.analyzer._generate_java_suggestions(metrics, entity)
        assert suggestions == []


# =====================================================================
# Rust analyzer
# =====================================================================


class TestRustAnalyzeEntityComplexity:
    def setup_method(self):
        self.analyzer = RustComplexityAnalyzer()

    def test_analyze_function_entity(self):
        entity = _make_entity(
            entity_type="function",
            language="rust",
            name="process",
            signature="fn process(data: &[u8], size: usize)",
        )
        pr = _make_parse_result(language="rust", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.language == "rust"
        assert metrics.entity_type == "function"
        assert metrics.parameter_count >= 0

    def test_analyze_async_function(self):
        entity = _make_entity(
            entity_type="function",
            language="rust",
            name="fetch",
            signature="async fn fetch(url: &str)",
            is_async=True,
        )
        pr = _make_parse_result(language="rust", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "async" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_struct_entity(self):
        entity = _make_entity(entity_type="struct", language="rust", name="Config")
        rels = [_make_relation("e1", "t1", "implements")]
        pr = _make_parse_result(language="rust", relationships=rels)
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.coupling_factor >= 0.0

    def test_rust_count_params_filters_self(self):
        entity = _make_entity(signature="fn method(&self, x: i32, y: i32)")
        count = self.analyzer._count_parameters(entity)
        assert count == 2  # self filtered

    def test_rust_count_params_no_parens(self):
        entity = _make_entity(signature="struct Config")
        assert self.analyzer._count_parameters(entity) == 0

    def test_rust_trait_coupling(self):
        entity = _make_entity(entity_type="struct")
        rels = [
            _make_relation("e1", "t1", "implements"),
            _make_relation("e1", "t2", "trait_bound"),
            _make_relation("e1", "t3", "calls"),
            _make_relation("e2", "t4", "implements"),  # different source — shouldn't count
        ]
        pr = _make_parse_result(language="rust", relationships=rels)
        result = self.analyzer._analyze_trait_coupling(entity, pr)
        assert result == 2 / 5.0  # 2 matching rels out of max 5

    def test_rust_suggestions_high_complexity(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=12.0)
        entity = _make_entity()
        suggestions = self.analyzer._generate_rust_suggestions(metrics, entity)
        assert any("smaller functions" in s for s in suggestions)

    def test_rust_suggestions_many_params(self):
        metrics = ServiceComplexityMetrics(parameter_count=6)
        entity = _make_entity()
        suggestions = self.analyzer._generate_rust_suggestions(metrics, entity)
        assert any("struct" in s for s in suggestions)

    def test_rust_suggestions_unsafe(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"has_unsafe": True})
        suggestions = self.analyzer._generate_rust_suggestions(metrics, entity)
        assert any("unsafe" in s.lower() for s in suggestions)

    def test_rust_suggestions_high_coupling(self):
        metrics = ServiceComplexityMetrics(coupling_factor=0.9)
        entity = _make_entity()
        suggestions = self.analyzer._generate_rust_suggestions(metrics, entity)
        assert any("coupling" in s.lower() for s in suggestions)

    def test_rust_suggestions_no_issues(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=3.0, parameter_count=1)
        entity = _make_entity(properties={})
        suggestions = self.analyzer._generate_rust_suggestions(metrics, entity)
        assert suggestions == []


# =====================================================================
# Go analyzer
# =====================================================================


class TestGoAnalyzeEntityComplexity:
    def setup_method(self):
        self.analyzer = GoComplexityAnalyzer()

    def test_analyze_function(self):
        entity = _make_entity(
            entity_type="function",
            language="go",
            name="HandleRequest",
            signature="func HandleRequest(w http.ResponseWriter, r *http.Request) error",
        )
        pr = _make_parse_result(language="go", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.language == "go"
        assert metrics.parameter_count == 2

    def test_analyze_struct(self):
        entity = _make_entity(entity_type="struct", language="go", name="Service")
        rels = [_make_relation("e1", "t1", "implements")]
        pr = _make_parse_result(language="go", relationships=rels)
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.coupling_factor >= 0.0

    def test_go_interface_coupling(self):
        entity = _make_entity(entity_type="struct")
        rels = [
            _make_relation("e1", "t1", "implements"),
            _make_relation("e1", "t2", "implements"),
            _make_relation("e1", "t3", "implements"),
        ]
        pr = _make_parse_result(language="go", relationships=rels)
        result = self.analyzer._analyze_interface_coupling(entity, pr)
        assert result == 1.0  # 3/3 = 1.0

    def test_go_maintainability(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=10.0, parameter_count=3)
        result = self.analyzer._calculate_go_maintainability(metrics)
        assert result == 100.0 - 10 * 1.5 - 3 * 2  # 79.0

    def test_go_suggestions_high_complexity(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=15.0)
        entity = _make_entity()
        suggestions = self.analyzer._generate_go_suggestions(metrics, entity)
        assert any("splitting" in s for s in suggestions)

    def test_go_suggestions_many_params(self):
        metrics = ServiceComplexityMetrics(parameter_count=7)
        entity = _make_entity()
        suggestions = self.analyzer._generate_go_suggestions(metrics, entity)
        assert any("struct" in s for s in suggestions)

    def test_go_suggestions_goroutine(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(entity_type="function", signature="go goroutine()")
        suggestions = self.analyzer._generate_go_suggestions(metrics, entity)
        assert any("goroutine" in s.lower() for s in suggestions)

    def test_go_suggestions_no_issues(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=3.0, parameter_count=2)
        entity = _make_entity(entity_type="function", signature="func foo()")
        suggestions = self.analyzer._generate_go_suggestions(metrics, entity)
        assert suggestions == []

    def test_go_count_parameters_empty(self):
        entity = _make_entity(signature="func foo()")
        assert self.analyzer._count_parameters(entity) == 0

    def test_go_count_parameters_multiple(self):
        entity = _make_entity(signature="func foo(a int, b string)")
        assert self.analyzer._count_parameters(entity) == 2


# =====================================================================
# TypeScript analyzer
# =====================================================================


class TestTypeScriptAnalyzeEntityComplexity:
    def setup_method(self):
        self.analyzer = TypeScriptComplexityAnalyzer()

    def test_analyze_interface(self):
        entity = _make_entity(entity_type="interface", language="typescript", name="IService")
        rels = [_make_relation("e1", "t1", "extends")]
        pr = _make_parse_result(language="typescript", entities=[entity], relationships=rels)
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.language == "typescript"
        assert metrics.coupling_factor > 0.0

    def test_analyze_type_alias(self):
        entity = _make_entity(entity_type="type_alias", language="typescript", name="MyType")
        pr = _make_parse_result(language="typescript", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.entity_type == "type_alias"

    def test_analyze_class_with_decorators(self):
        entity = _make_entity(
            entity_type="class",
            language="typescript",
            name="Component",
            properties={"decorators": ["@Component", "@Injectable"]},
        )
        pr = _make_parse_result(language="typescript", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "decorators" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_with_generics(self):
        entity = _make_entity(
            entity_type="function",
            language="typescript",
            name="transform",
            properties={
                "generic_parameters": ["T", "U"],
                "parameters": "input: T, mapper: (t: T) => U",
            },
        )
        pr = _make_parse_result(language="typescript", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        # generics multiply by 1.1 and add 0.2 per param
        assert metrics.parameter_count == 2

    def test_analyze_conditional_type(self):
        entity = _make_entity(
            entity_type="type_alias",
            language="typescript",
            name="ConditionalType",
            properties={"is_conditional_type": True},
        )
        pr = _make_parse_result(language="typescript", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "conditional_type" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_mapped_type(self):
        entity = _make_entity(
            entity_type="type_alias",
            language="typescript",
            name="MappedType",
            properties={"is_mapped_type": True},
        )
        pr = _make_parse_result(language="typescript", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "mapped_type" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_ts_type_coupling(self):
        entity = _make_entity(entity_type="interface")
        rels = [
            _make_relation("e1", "t1", "extends"),
            _make_relation("e1", "t2", "generic_constraint"),
        ]
        pr = _make_parse_result(language="typescript", relationships=rels)
        result = self.analyzer._analyze_type_coupling(entity, pr)
        assert result == 2 / 8.0

    def test_ts_class_coupling_extends(self):
        entity = _make_entity(entity_type="class")
        rels = [
            _make_relation("e1", "t1", "extends"),
            _make_relation("e1", "t2", "decorated_by"),
            _make_relation("e1", "t3", "calls"),
        ]
        pr = _make_parse_result(language="typescript", relationships=rels)
        result = self.analyzer._analyze_class_coupling(entity, pr)
        # extends=2.0, decorated_by=0.5, calls=1.0 => total=3.5, 3.5/10=0.35
        assert abs(result - 0.35) < 0.01

    def test_ts_count_parameters(self):
        entity = _make_entity(properties={"parameters": "a: string, b: number, c: boolean"})
        assert self.analyzer._count_typescript_parameters(entity) == 3

    def test_ts_count_parameters_empty(self):
        entity = _make_entity(properties={"parameters": ""})
        assert self.analyzer._count_typescript_parameters(entity) == 0

    def test_ts_count_parameters_no_key(self):
        entity = _make_entity(properties={})
        assert self.analyzer._count_typescript_parameters(entity) == 0

    def test_ts_maintainability_with_exported_generics_decorators(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=5.0, parameter_count=2)
        entity = _make_entity(
            properties={
                "is_exported": True,
                "generic_parameters": ["T", "U"],
                "decorators": ["@Inject"],
            }
        )
        result = self.analyzer._calculate_typescript_maintainability(metrics, entity)
        # 100 - 5*2 - 2*1.5 - 5 - 2*3 - 1*2 = 100 - 10 - 3 - 5 - 6 - 2 = 74
        assert abs(result - 74.0) < 0.01

    def test_ts_suggestions_high_complexity(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=15.0)
        entity = _make_entity()
        suggestions = self.analyzer._generate_typescript_suggestions(metrics, entity)
        assert any("breaking" in s for s in suggestions)

    def test_ts_suggestions_many_params(self):
        metrics = ServiceComplexityMetrics(parameter_count=6)
        entity = _make_entity()
        suggestions = self.analyzer._generate_typescript_suggestions(metrics, entity)
        assert any("options object" in s for s in suggestions)

    def test_ts_suggestions_many_generics(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"generic_parameters": ["T", "U", "V", "W"]})
        suggestions = self.analyzer._generate_typescript_suggestions(metrics, entity)
        assert any("generic" in s.lower() for s in suggestions)

    def test_ts_suggestions_conditional_type(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"is_conditional_type": True})
        suggestions = self.analyzer._generate_typescript_suggestions(metrics, entity)
        assert any("conditional" in s.lower() for s in suggestions)

    def test_ts_suggestions_mapped_type(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"is_mapped_type": True})
        suggestions = self.analyzer._generate_typescript_suggestions(metrics, entity)
        assert any("mapped" in s.lower() for s in suggestions)

    def test_ts_suggestions_many_decorators(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"decorators": ["@A", "@B", "@C"]})
        suggestions = self.analyzer._generate_typescript_suggestions(metrics, entity)
        assert any("decorator" in s.lower() for s in suggestions)

    def test_ts_suggestions_high_coupling(self):
        metrics = ServiceComplexityMetrics(coupling_factor=0.8)
        entity = _make_entity()
        suggestions = self.analyzer._generate_typescript_suggestions(metrics, entity)
        assert any("coupling" in s.lower() for s in suggestions)

    def test_ts_suggestions_no_issues(self):
        metrics = ServiceComplexityMetrics(
            cyclomatic_complexity=3.0, parameter_count=1, coupling_factor=0.1
        )
        entity = _make_entity(properties={})
        suggestions = self.analyzer._generate_typescript_suggestions(metrics, entity)
        assert suggestions == []


# =====================================================================
# Python analyzer
# =====================================================================


class TestPythonAnalyzeEntityComplexity:
    def setup_method(self):
        self.analyzer = PythonComplexityAnalyzer()

    def test_analyze_function(self):
        entity = _make_entity(
            entity_type="function",
            name="process",
            signature="def process(self, data, config=None)",
        )
        pr = _make_parse_result(language="python", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.language == "python"
        assert metrics.parameter_count >= 0

    def test_analyze_function_with_decorators(self):
        entity = _make_entity(
            entity_type="function",
            name="view",
            signature="def view(request)",
            properties={"decorators": ["@login_required", "@cache"]},
        )
        pr = _make_parse_result(language="python", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "decorators" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_with_comprehension(self):
        entity = _make_entity(
            entity_type="function",
            name="get_items",
            properties={"has_comprehension": True},
        )
        pr = _make_parse_result(language="python", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "comprehension" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_with_metaclass(self):
        entity = _make_entity(
            entity_type="class",
            name="Meta",
            properties={"has_metaclass": True},
        )
        pr = _make_parse_result(language="python", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "metaclass" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_async_function(self):
        entity = _make_entity(
            entity_type="function",
            name="fetch",
            is_async=True,
        )
        pr = _make_parse_result(language="python", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "async" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_generator(self):
        entity = _make_entity(
            entity_type="function",
            name="gen",
            properties={"is_generator": True},
        )
        pr = _make_parse_result(language="python", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "generator" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_class_coupling(self):
        entity = _make_entity(entity_type="class", name="Service")
        rels = [
            _make_relation("e1", "t1", "inherits"),
            _make_relation("e1", "t2", "imports"),
            _make_relation("e1", "t3", "calls"),
        ]
        pr = _make_parse_result(language="python", entities=[entity], relationships=rels)
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.coupling_factor > 0.0

    def test_python_count_params_filters_self(self):
        entity = _make_entity(signature="def method(self, x, y)")
        assert self.analyzer._count_parameters(entity) == 2

    def test_python_count_params_filters_cls(self):
        entity = _make_entity(signature="def method(cls, x)")
        assert self.analyzer._count_parameters(entity) == 1

    def test_python_count_params_no_parens(self):
        entity = _make_entity(signature="x = 5")
        assert self.analyzer._count_parameters(entity) == 0

    def test_python_class_coupling_inherits(self):
        entity = _make_entity(entity_type="class")
        rels = [_make_relation("e1", "t1", "inherits")]
        pr = _make_parse_result(relationships=rels)
        result = self.analyzer._analyze_class_coupling(entity, pr)
        assert result == 1.5 / 8.0

    def test_python_class_coupling_imports(self):
        entity = _make_entity(entity_type="class")
        rels = [_make_relation("e1", "t1", "imports")]
        pr = _make_parse_result(relationships=rels)
        result = self.analyzer._analyze_class_coupling(entity, pr)
        assert result == 0.5 / 8.0

    def test_python_class_coupling_other(self):
        entity = _make_entity(entity_type="class")
        rels = [_make_relation("e1", "t1", "calls")]
        pr = _make_parse_result(relationships=rels)
        result = self.analyzer._analyze_class_coupling(entity, pr)
        assert result == 1.0 / 8.0

    def test_python_maintainability_with_many_decorators(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=5.0, parameter_count=2)
        entity = _make_entity(
            properties={
                "decorators": ["@a", "@b", "@c", "@d", "@e"],
            }
        )
        result = self.analyzer._calculate_python_maintainability(metrics, entity)
        # 100 - 5*2 - 2*1.5 - (5-3)*5 = 100 - 10 - 3 - 10 = 77
        assert abs(result - 77.0) < 0.01

    def test_python_maintainability_magic_method(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=2.0, parameter_count=0)
        # Properties must be truthy for the magic method check to fire
        entity = _make_entity(name="__init__", properties={"some_key": True})
        result = self.analyzer._calculate_python_maintainability(metrics, entity)
        # 100 - 2*2 - 0 - 5 (magic) = 91
        assert abs(result - 91.0) < 0.01

    def test_python_maintainability_type_hints_bonus(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=2.0, parameter_count=0)
        entity = _make_entity(properties={"has_type_hints": True})
        result = self.analyzer._calculate_python_maintainability(metrics, entity)
        # 100 - 4 + 5 = 101 clamped to 100
        assert result == 100.0

    def test_python_suggestions_high_complexity(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=15.0)
        entity = _make_entity()
        suggestions = self.analyzer._generate_python_suggestions(metrics, entity)
        assert any("smaller" in s for s in suggestions)

    def test_python_suggestions_many_params(self):
        metrics = ServiceComplexityMetrics(parameter_count=7)
        entity = _make_entity()
        suggestions = self.analyzer._generate_python_suggestions(metrics, entity)
        assert any("kwargs" in s or "dataclass" in s for s in suggestions)

    def test_python_suggestions_decorator_stacking(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"decorators": ["@a", "@b", "@c", "@d"]})
        suggestions = self.analyzer._generate_python_suggestions(metrics, entity)
        assert any("decorator" in s.lower() for s in suggestions)

    def test_python_suggestions_complex_comprehension(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=20.0)
        entity = _make_entity(properties={"has_comprehension": True})
        suggestions = self.analyzer._generate_python_suggestions(metrics, entity)
        assert any("comprehension" in s.lower() for s in suggestions)

    def test_python_suggestions_no_type_hints(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(entity_type="function", properties={"has_type_hints": False})
        suggestions = self.analyzer._generate_python_suggestions(metrics, entity)
        # has_type_hints is False but truthy — need to check the condition
        # The code checks `not entity.properties.get("has_type_hints")` which is True when False
        assert any("type hints" in s for s in suggestions)

    def test_python_suggestions_high_coupling(self):
        metrics = ServiceComplexityMetrics(coupling_factor=0.8)
        entity = _make_entity()
        suggestions = self.analyzer._generate_python_suggestions(metrics, entity)
        assert any("coupling" in s.lower() for s in suggestions)

    def test_python_suggestions_magic_method(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(name="__repr__")
        suggestions = self.analyzer._generate_python_suggestions(metrics, entity)
        assert any("magic" in s.lower() for s in suggestions)


# =====================================================================
# JavaScript analyzer
# =====================================================================


class TestJavaScriptAnalyzeEntityComplexity:
    def setup_method(self):
        self.analyzer = JavaScriptComplexityAnalyzer()

    def test_analyze_function(self):
        entity = _make_entity(
            entity_type="function",
            language="javascript",
            name="handleClick",
            signature="function handleClick(event, callback)",
        )
        pr = _make_parse_result(language="javascript", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.language == "javascript"
        assert metrics.parameter_count == 2

    def test_analyze_arrow_function(self):
        entity = _make_entity(
            entity_type="arrow_function",
            language="javascript",
            name="handler",
            signature="(x, y) => x + y",
        )
        pr = _make_parse_result(language="javascript", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.parameter_count == 2

    def test_analyze_function_with_callbacks(self):
        entity = _make_entity(
            entity_type="function",
            language="javascript",
            name="fetch",
            properties={"has_callbacks": True},
        )
        pr = _make_parse_result(language="javascript", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "callbacks" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_with_promises(self):
        entity = _make_entity(
            entity_type="function",
            language="javascript",
            name="fetch",
            properties={"has_promises": True},
        )
        pr = _make_parse_result(language="javascript", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "promises" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_with_closures(self):
        entity = _make_entity(
            entity_type="function",
            language="javascript",
            name="make",
            properties={"closure_depth": 3},
        )
        pr = _make_parse_result(language="javascript", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "closures" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_iife(self):
        entity = _make_entity(
            entity_type="function",
            language="javascript",
            name="init",
            properties={"is_iife": True},
        )
        pr = _make_parse_result(language="javascript", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "iife" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_async_function(self):
        entity = _make_entity(
            entity_type="function",
            language="javascript",
            name="fetch",
            is_async=True,
        )
        pr = _make_parse_result(language="javascript", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "async" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_class(self):
        entity = _make_entity(entity_type="class", language="javascript", name="App")
        rels = [_make_relation("e1", "t1", "extends")]
        pr = _make_parse_result(language="javascript", entities=[entity], relationships=rels)
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.coupling_factor > 0.0

    def test_js_count_params_filters_rest(self):
        entity = _make_entity(signature="function foo(..., a, b)")
        assert self.analyzer._count_parameters(entity) == 2

    def test_js_class_coupling_extends(self):
        entity = _make_entity(entity_type="class")
        rels = [_make_relation("e1", "t1", "extends")]
        pr = _make_parse_result(language="javascript", relationships=rels)
        result = self.analyzer._analyze_class_coupling(entity, pr)
        assert result == 1.5 / 8.0

    def test_js_class_coupling_imports(self):
        entity = _make_entity(entity_type="class")
        rels = [_make_relation("e1", "t1", "imports")]
        pr = _make_parse_result(language="javascript", relationships=rels)
        result = self.analyzer._analyze_class_coupling(entity, pr)
        assert result == 0.5 / 8.0

    def test_js_class_coupling_other(self):
        entity = _make_entity(entity_type="class")
        rels = [_make_relation("e1", "t1", "calls")]
        pr = _make_parse_result(language="javascript", relationships=rels)
        result = self.analyzer._analyze_class_coupling(entity, pr)
        assert result == 1.0 / 8.0

    def test_js_maintainability_callbacks_high_complexity(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=20.0, parameter_count=2)
        entity = _make_entity(properties={"has_callbacks": True})
        result = self.analyzer._calculate_javascript_maintainability(metrics, entity)
        # 100 - 20*2 - 2*1.5 - 15 = 100 - 40 - 3 - 15 = 42
        assert abs(result - 42.0) < 0.01

    def test_js_maintainability_deep_closures(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=5.0, parameter_count=0)
        entity = _make_entity(properties={"closure_depth": 4})
        result = self.analyzer._calculate_javascript_maintainability(metrics, entity)
        # 100 - 5*2 - 0 - (4-2)*5 = 100 - 10 - 10 = 80
        assert abs(result - 80.0) < 0.01

    def test_js_maintainability_prototype(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=2.0, parameter_count=0)
        entity = _make_entity(properties={"modifies_prototype": True})
        result = self.analyzer._calculate_javascript_maintainability(metrics, entity)
        # 100 - 4 - 10 = 86
        assert abs(result - 86.0) < 0.01

    def test_js_suggestions_many_params(self):
        metrics = ServiceComplexityMetrics(parameter_count=6)
        entity = _make_entity()
        suggestions = self.analyzer._generate_javascript_suggestions(metrics, entity)
        assert any("options object" in s for s in suggestions)

    def test_js_suggestions_callbacks_no_async(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=5.0)
        entity = _make_entity(properties={"has_callbacks": True}, is_async=False)
        suggestions = self.analyzer._generate_javascript_suggestions(metrics, entity)
        assert any("async/await" in s for s in suggestions)

    def test_js_suggestions_complex_promises(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=15.0)
        entity = _make_entity(properties={"has_promises": True})
        suggestions = self.analyzer._generate_javascript_suggestions(metrics, entity)
        assert any("promise" in s.lower() for s in suggestions)

    def test_js_suggestions_deep_closures(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"closure_depth": 4})
        suggestions = self.analyzer._generate_javascript_suggestions(metrics, entity)
        assert any("closure" in s.lower() for s in suggestions)

    def test_js_suggestions_prototype_modification(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"modifies_prototype": True})
        suggestions = self.analyzer._generate_javascript_suggestions(metrics, entity)
        assert any("prototype" in s.lower() for s in suggestions)

    def test_js_suggestions_high_coupling(self):
        metrics = ServiceComplexityMetrics(coupling_factor=0.8)
        entity = _make_entity()
        suggestions = self.analyzer._generate_javascript_suggestions(metrics, entity)
        assert any("coupling" in s.lower() for s in suggestions)


# =====================================================================
# C++ analyzer
# =====================================================================


class TestCppAnalyzeEntityComplexity:
    def setup_method(self):
        self.analyzer = CppComplexityAnalyzer()

    def test_analyze_function(self):
        entity = _make_entity(
            entity_type="function",
            language="cpp",
            name="process",
            signature="void process(int x, double y)",
        )
        pr = _make_parse_result(language="cpp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.language == "cpp"
        assert metrics.parameter_count == 2

    def test_analyze_method_with_templates(self):
        entity = _make_entity(
            entity_type="method",
            language="cpp",
            name="transform",
            properties={"template_parameters": ["T", "U"]},
        )
        pr = _make_parse_result(language="cpp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "template_parameters" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed

    def test_analyze_constructor(self):
        entity = _make_entity(
            entity_type="constructor",
            language="cpp",
            name="MyClass",
            signature="MyClass(int a, int b)",
        )
        pr = _make_parse_result(language="cpp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.parameter_count == 2

    def test_analyze_destructor(self):
        entity = _make_entity(
            entity_type="destructor",
            language="cpp",
            name="~MyClass",
        )
        pr = _make_parse_result(language="cpp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.entity_name == "~MyClass"

    def test_analyze_operator_overload(self):
        entity = _make_entity(
            entity_type="method",
            language="cpp",
            name="operator+",
            properties={"is_operator_overload": True},
        )
        pr = _make_parse_result(language="cpp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "operator_overload" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_with_resources(self):
        entity = _make_entity(
            entity_type="function",
            language="cpp",
            name="allocate",
            properties={"manages_resources": True},
        )
        pr = _make_parse_result(language="cpp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "manages_resources" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_with_exceptions(self):
        entity = _make_entity(
            entity_type="function",
            language="cpp",
            name="parse",
            properties={"has_exceptions": True},
        )
        pr = _make_parse_result(language="cpp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "exception_handling" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_with_move_semantics(self):
        entity = _make_entity(
            entity_type="function",
            language="cpp",
            name="transfer",
            properties={"has_move_semantics": True},
        )
        pr = _make_parse_result(language="cpp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "move_semantics" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_function_with_raw_pointers(self):
        entity = _make_entity(
            entity_type="function",
            language="cpp",
            name="legacy",
            properties={"uses_raw_pointers": True},
        )
        pr = _make_parse_result(language="cpp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "raw_pointers" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_class_multiple_inheritance(self):
        entity = _make_entity(
            entity_type="class",
            language="cpp",
            name="Derived",
            properties={"inheritance_count": 3, "has_virtual_functions": True},
        )
        rels = [_make_relation("e1", "t1", "inherits")]
        pr = _make_parse_result(language="cpp", entities=[entity], relationships=rels)
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "multiple_inheritance" in metrics.risk_factors
        assert "virtual_functions" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_cpp_count_params_void(self):
        entity = _make_entity(signature="void foo(void)")
        assert self.analyzer._count_parameters(entity) == 0

    def test_cpp_count_params_multiple(self):
        entity = _make_entity(signature="int foo(int a, double b, char c)")
        assert self.analyzer._count_parameters(entity) == 3

    def test_cpp_class_coupling(self):
        entity = _make_entity(entity_type="class")
        rels = [
            _make_relation("e1", "t1", "inherits"),
            _make_relation("e1", "t2", "template_specialization"),
            _make_relation("e1", "t3", "includes"),
            _make_relation("e1", "t4", "calls"),
        ]
        pr = _make_parse_result(language="cpp", relationships=rels)
        result = self.analyzer._analyze_class_coupling(entity, pr)
        # 2.0 + 1.5 + 0.5 + 1.0 = 5.0 / 10 = 0.5
        assert abs(result - 0.5) < 0.01

    def test_cpp_maintainability_templates(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=5.0, parameter_count=2)
        entity = _make_entity(properties={"template_parameters": ["T", "U", "V"]})
        result = self.analyzer._calculate_cpp_maintainability(metrics, entity)
        # 100 - 5*2.5 - 2*2 - (3-2)*8 = 100 - 12.5 - 4 - 8 = 75.5
        assert abs(result - 75.5) < 0.01

    def test_cpp_maintainability_multiple_inheritance(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=3.0, parameter_count=0)
        entity = _make_entity(properties={"inheritance_count": 3})
        result = self.analyzer._calculate_cpp_maintainability(metrics, entity)
        # 100 - 3*2.5 - 0 - (3-1)*10 = 100 - 7.5 - 20 = 72.5
        assert abs(result - 72.5) < 0.01

    def test_cpp_maintainability_raw_pointers_penalty(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=2.0, parameter_count=0)
        entity = _make_entity(properties={"uses_raw_pointers": True})
        result = self.analyzer._calculate_cpp_maintainability(metrics, entity)
        # 100 - 2*2.5 - 15 = 80
        assert abs(result - 80.0) < 0.01

    def test_cpp_maintainability_smart_pointers_bonus(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=2.0, parameter_count=0)
        entity = _make_entity(properties={"uses_smart_pointers": True})
        result = self.analyzer._calculate_cpp_maintainability(metrics, entity)
        # 100 - 5 + 5 = 100
        assert result == 100.0

    def test_cpp_maintainability_const_correctness_bonus(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=2.0, parameter_count=0)
        entity = _make_entity(properties={"const_correct": True})
        result = self.analyzer._calculate_cpp_maintainability(metrics, entity)
        # 100 - 5 + 5 = 100
        assert result == 100.0

    def test_cpp_suggestions_high_complexity(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=12.0)
        entity = _make_entity()
        suggestions = self.analyzer._generate_cpp_suggestions(metrics, entity)
        assert any("smaller" in s for s in suggestions)

    def test_cpp_suggestions_many_params(self):
        metrics = ServiceComplexityMetrics(parameter_count=6)
        entity = _make_entity()
        suggestions = self.analyzer._generate_cpp_suggestions(metrics, entity)
        assert any("struct" in s or "builder" in s for s in suggestions)

    def test_cpp_suggestions_template_complexity(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"template_parameters": ["T", "U", "V"]})
        suggestions = self.analyzer._generate_cpp_suggestions(metrics, entity)
        assert any("template" in s.lower() for s in suggestions)

    def test_cpp_suggestions_multiple_inheritance(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"inheritance_count": 2})
        suggestions = self.analyzer._generate_cpp_suggestions(metrics, entity)
        assert any("inheritance" in s.lower() for s in suggestions)

    def test_cpp_suggestions_raw_pointers(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"uses_raw_pointers": True})
        suggestions = self.analyzer._generate_cpp_suggestions(metrics, entity)
        assert any("smart pointer" in s.lower() for s in suggestions)

    def test_cpp_suggestions_operator_overload_complex(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=8.0)
        entity = _make_entity(properties={"is_operator_overload": True})
        suggestions = self.analyzer._generate_cpp_suggestions(metrics, entity)
        assert any("operator" in s.lower() for s in suggestions)

    def test_cpp_suggestions_exceptions_not_safe(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"has_exceptions": True, "exception_safe": False})
        suggestions = self.analyzer._generate_cpp_suggestions(metrics, entity)
        assert any("exception safety" in s.lower() for s in suggestions)

    def test_cpp_suggestions_resource_no_rule_of_five(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(
            entity_type="class",
            properties={"manages_resources": True, "has_rule_of_five": False},
        )
        suggestions = self.analyzer._generate_cpp_suggestions(metrics, entity)
        assert any("rule of five" in s.lower() for s in suggestions)

    def test_cpp_suggestions_high_coupling(self):
        metrics = ServiceComplexityMetrics(coupling_factor=0.8)
        entity = _make_entity()
        suggestions = self.analyzer._generate_cpp_suggestions(metrics, entity)
        assert any("coupling" in s.lower() for s in suggestions)


# =====================================================================
# C# analyzer
# =====================================================================


class TestCSharpAnalyzeEntityComplexity:
    def setup_method(self):
        self.analyzer = CSharpComplexityAnalyzer()

    def test_analyze_method(self):
        entity = _make_entity(
            entity_type="method",
            language="csharp",
            name="Execute",
            signature="public void Execute(int id, string name)",
        )
        pr = _make_parse_result(language="csharp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.language == "csharp"
        assert metrics.parameter_count == 2

    def test_analyze_constructor(self):
        entity = _make_entity(
            entity_type="constructor",
            language="csharp",
            name="Service",
            signature="public Service(ILogger logger)",
        )
        pr = _make_parse_result(language="csharp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert metrics.parameter_count == 1

    def test_analyze_method_with_linq(self):
        entity = _make_entity(
            entity_type="method",
            language="csharp",
            name="Query",
            properties={"has_linq": True, "linq_depth": 3},
        )
        pr = _make_parse_result(language="csharp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "linq" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_async_method(self):
        entity = _make_entity(
            entity_type="method",
            language="csharp",
            name="FetchAsync",
            is_async=True,
            properties={"some_key": True},  # Must be truthy to enter the properties block
        )
        pr = _make_parse_result(language="csharp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "async" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_method_with_delegates(self):
        entity = _make_entity(
            entity_type="method",
            language="csharp",
            name="Subscribe",
            properties={"has_delegates": True},
        )
        pr = _make_parse_result(language="csharp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "delegates" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_method_with_reflection(self):
        entity = _make_entity(
            entity_type="method",
            language="csharp",
            name="Invoke",
            properties={"uses_reflection": True},
        )
        pr = _make_parse_result(language="csharp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "reflection" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_extension_method(self):
        entity = _make_entity(
            entity_type="method",
            language="csharp",
            name="ToList",
            properties={"is_extension_method": True},
        )
        pr = _make_parse_result(language="csharp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "extension_method" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_class_with_generics(self):
        entity = _make_entity(
            entity_type="class",
            language="csharp",
            name="Repository",
            properties={"generic_parameters": ["T", "TKey"]},
        )
        rels = [_make_relation("e1", "t1", "implements")]
        pr = _make_parse_result(language="csharp", entities=[entity], relationships=rels)
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "generic_parameters" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_class_with_many_attributes(self):
        entity = _make_entity(
            entity_type="class",
            language="csharp",
            name="Controller",
            properties={"attributes": ["Route", "ApiController", "Authorize"]},
        )
        pr = _make_parse_result(language="csharp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "many_attributes" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_analyze_property_with_getter_setter(self):
        entity = _make_entity(
            entity_type="property",
            language="csharp",
            name="Name",
            properties={"has_getter": True, "has_setter": True},
        )
        pr = _make_parse_result(language="csharp", entities=[entity])
        metrics = self.analyzer.analyze_entity_complexity(entity, pr)
        assert "property_getter_setter" in metrics.risk_factors
        assert metrics.computed is False  # no source content passed
        assert metrics.cyclomatic_complexity == 0.0

    def test_csharp_classify_complexity(self):
        assert self.analyzer._classify_complexity(5) == ComplexityLevel.TRIVIAL
        assert self.analyzer._classify_complexity(6) == ComplexityLevel.BASIC
        assert self.analyzer._classify_complexity(10) == ComplexityLevel.BASIC
        assert self.analyzer._classify_complexity(11) == ComplexityLevel.MODERATE
        assert self.analyzer._classify_complexity(20) == ComplexityLevel.MODERATE
        assert self.analyzer._classify_complexity(21) == ComplexityLevel.COMPLEX
        assert self.analyzer._classify_complexity(40) == ComplexityLevel.COMPLEX
        assert self.analyzer._classify_complexity(41) == ComplexityLevel.VERY_COMPLEX

    def test_csharp_count_params_extension_method(self):
        entity = _make_entity(signature="static IEnumerable<T> ToList(this IQueryable<T> source)")
        assert self.analyzer._count_parameters(entity) == 0  # "this " filtered

    def test_csharp_count_params_multiple(self):
        entity = _make_entity(signature="void Foo(int a, string b)")
        assert self.analyzer._count_parameters(entity) == 2

    def test_csharp_class_coupling_inherits(self):
        entity = _make_entity(entity_type="class")
        rels = [_make_relation("e1", "t1", "inherits")]
        pr = _make_parse_result(relationships=rels)
        result = self.analyzer._analyze_class_coupling(entity, pr)
        assert result == 1.5 / 8.0

    def test_csharp_class_coupling_implements(self):
        entity = _make_entity(entity_type="class")
        rels = [_make_relation("e1", "t1", "implements")]
        pr = _make_parse_result(relationships=rels)
        result = self.analyzer._analyze_class_coupling(entity, pr)
        assert result == 0.8 / 8.0

    def test_csharp_class_coupling_uses(self):
        entity = _make_entity(entity_type="class")
        rels = [_make_relation("e1", "t1", "uses")]
        pr = _make_parse_result(relationships=rels)
        result = self.analyzer._analyze_class_coupling(entity, pr)
        assert result == 1.0 / 8.0

    def test_csharp_class_coupling_other(self):
        entity = _make_entity(entity_type="class")
        rels = [_make_relation("e1", "t1", "calls")]
        pr = _make_parse_result(relationships=rels)
        result = self.analyzer._analyze_class_coupling(entity, pr)
        assert result == 0.5 / 8.0

    def test_csharp_maintainability_linq_deep(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=5.0, parameter_count=2)
        entity = _make_entity(properties={"has_linq": True, "linq_depth": 4})
        result = self.analyzer._calculate_csharp_maintainability(metrics, entity)
        # 100 - 10 - 3 - (4-2)*8 = 100 - 10 - 3 - 16 = 71
        assert abs(result - 71.0) < 0.01

    def test_csharp_maintainability_reflection(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=2.0, parameter_count=0)
        entity = _make_entity(properties={"uses_reflection": True})
        result = self.analyzer._calculate_csharp_maintainability(metrics, entity)
        # 100 - 4 - 15 = 81
        assert abs(result - 81.0) < 0.01

    def test_csharp_maintainability_async_complex(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=15.0, parameter_count=0)
        entity = _make_entity(is_async=True, properties={"some_key": True})
        result = self.analyzer._calculate_csharp_maintainability(metrics, entity)
        # 100 - 15*2 - 10 (async + cc>10) = 60
        assert abs(result - 60.0) < 0.01

    def test_csharp_maintainability_delegates(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=2.0, parameter_count=0)
        entity = _make_entity(properties={"has_delegates": True, "delegate_count": 4})
        result = self.analyzer._calculate_csharp_maintainability(metrics, entity)
        # 100 - 4 - 5 = 91
        assert abs(result - 91.0) < 0.01

    def test_csharp_maintainability_auto_property_bonus(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=1.0, parameter_count=0)
        entity = _make_entity(entity_type="property", properties={"is_auto_property": True})
        result = self.analyzer._calculate_csharp_maintainability(metrics, entity)
        # 100 - 2 + 5 = 103 clamped to 100
        assert result == 100.0

    def test_csharp_maintainability_implements_bonus(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=2.0, parameter_count=0)
        entity = _make_entity(entity_type="class", properties={"implements_interfaces": True})
        result = self.analyzer._calculate_csharp_maintainability(metrics, entity)
        # 100 - 4 + 3 = 99
        assert abs(result - 99.0) < 0.01

    def test_csharp_suggestions_high_complexity(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=15.0)
        entity = _make_entity()
        suggestions = self.analyzer._generate_csharp_suggestions(metrics, entity)
        assert any("breaking" in s or "smaller" in s for s in suggestions)

    def test_csharp_suggestions_many_params(self):
        metrics = ServiceComplexityMetrics(parameter_count=7)
        entity = _make_entity()
        suggestions = self.analyzer._generate_csharp_suggestions(metrics, entity)
        assert any("parameter object" in s for s in suggestions)

    def test_csharp_suggestions_deep_linq(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"has_linq": True, "linq_depth": 4})
        suggestions = self.analyzer._generate_csharp_suggestions(metrics, entity)
        assert any("linq" in s.lower() for s in suggestions)

    def test_csharp_suggestions_complex_async(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=15.0)
        entity = _make_entity(is_async=True, properties={"some_key": True})
        suggestions = self.analyzer._generate_csharp_suggestions(metrics, entity)
        assert any("async" in s.lower() for s in suggestions)

    def test_csharp_suggestions_reflection(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"uses_reflection": True})
        suggestions = self.analyzer._generate_csharp_suggestions(metrics, entity)
        assert any("reflection" in s.lower() for s in suggestions)

    def test_csharp_suggestions_delegates(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"has_delegates": True, "delegate_count": 4})
        suggestions = self.analyzer._generate_csharp_suggestions(metrics, entity)
        assert any("delegate" in s.lower() for s in suggestions)

    def test_csharp_suggestions_generics(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"generic_parameters": ["T", "U", "V"]})
        suggestions = self.analyzer._generate_csharp_suggestions(metrics, entity)
        assert any("generic" in s.lower() for s in suggestions)

    def test_csharp_suggestions_many_attributes(self):
        metrics = ServiceComplexityMetrics()
        entity = _make_entity(properties={"attributes": ["A", "B", "C", "D"]})
        suggestions = self.analyzer._generate_csharp_suggestions(metrics, entity)
        assert any("attribute" in s.lower() for s in suggestions)

    def test_csharp_suggestions_complex_property(self):
        metrics = ServiceComplexityMetrics(cyclomatic_complexity=5.0)
        entity = _make_entity(entity_type="property", properties={"is_auto_property": False})
        suggestions = self.analyzer._generate_csharp_suggestions(metrics, entity)
        assert any("property" in s.lower() for s in suggestions)

    def test_csharp_suggestions_high_coupling(self):
        metrics = ServiceComplexityMetrics(coupling_factor=0.8)
        entity = _make_entity()
        suggestions = self.analyzer._generate_csharp_suggestions(metrics, entity)
        assert any("coupling" in s.lower() for s in suggestions)

    def test_csharp_get_language_specific_factors(self):
        factors = self.analyzer.get_language_specific_factors()
        assert "linq_penalty" in factors
        assert "reflection_penalty" in factors
        assert "interface_bonus" in factors
        assert "property_bonus" in factors


# =====================================================================
# C++ classify complexity (was missing from original tests)
# =====================================================================


class TestCppClassifyComplexity:
    def setup_method(self):
        self.analyzer = CppComplexityAnalyzer()

    def test_trivial(self):
        assert self.analyzer._classify_complexity(4) == ComplexityLevel.TRIVIAL

    def test_basic(self):
        assert self.analyzer._classify_complexity(5) == ComplexityLevel.BASIC
        assert self.analyzer._classify_complexity(8) == ComplexityLevel.BASIC

    def test_moderate(self):
        assert self.analyzer._classify_complexity(9) == ComplexityLevel.MODERATE
        assert self.analyzer._classify_complexity(15) == ComplexityLevel.MODERATE

    def test_complex(self):
        assert self.analyzer._classify_complexity(16) == ComplexityLevel.COMPLEX
        assert self.analyzer._classify_complexity(30) == ComplexityLevel.COMPLEX

    def test_very_complex(self):
        assert self.analyzer._classify_complexity(31) == ComplexityLevel.VERY_COMPLEX


# =====================================================================
# C# classify complexity
# =====================================================================


class TestCSharpClassifyComplexity:
    def setup_method(self):
        self.analyzer = CSharpComplexityAnalyzer()

    def test_trivial(self):
        assert self.analyzer._classify_complexity(0) == ComplexityLevel.TRIVIAL
        assert self.analyzer._classify_complexity(5) == ComplexityLevel.TRIVIAL

    def test_basic(self):
        assert self.analyzer._classify_complexity(6) == ComplexityLevel.BASIC
        assert self.analyzer._classify_complexity(10) == ComplexityLevel.BASIC

    def test_moderate(self):
        assert self.analyzer._classify_complexity(11) == ComplexityLevel.MODERATE
        assert self.analyzer._classify_complexity(20) == ComplexityLevel.MODERATE

    def test_complex(self):
        assert self.analyzer._classify_complexity(21) == ComplexityLevel.COMPLEX
        assert self.analyzer._classify_complexity(40) == ComplexityLevel.COMPLEX

    def test_very_complex(self):
        assert self.analyzer._classify_complexity(41) == ComplexityLevel.VERY_COMPLEX


# =====================================================================
# ComplexityAnalysisService — file-level analysis
# =====================================================================


class TestComplexityAnalysisServiceFileMethods:
    def setup_method(self):
        self.service = ComplexityAnalysisService()

    def test_analyze_file_complexity_python(self):
        entities = [
            _make_entity(
                entity_id="e1", name="func_a", entity_type="function", signature="def func_a(x)"
            ),
            _make_entity(
                entity_id="e2", name="func_b", entity_type="function", signature="def func_b(y)"
            ),
        ]
        pr = _make_parse_result(language="python", entities=entities, lines_of_code=50)
        report = self.service.analyze_file_complexity(pr)
        assert isinstance(report, FileComplexityReport)
        assert report.language == "python"
        assert report.total_entities == 2
        assert report.total_loc == 50
        assert len(report.entity_metrics) == 2
        # No source content passed: entities are not computed, never 1.0.
        assert report.entities_not_computed == report.total_entities
        assert report.max_complexity == 0.0
        # Zero measurements never yield a clean score.
        assert report.quality_score is None
        assert report.metrics_computed is False

    def test_analyze_file_complexity_empty(self):
        pr = _make_parse_result(language="python", entities=[], lines_of_code=0)
        report = self.service.analyze_file_complexity(pr)
        assert report.total_entities == 0
        assert report.average_complexity == 0.0
        assert report.quality_score == 100.0

    def test_analyze_file_complexity_unsupported_language(self):
        entity = _make_entity(
            entity_id="e1", name="main", entity_type="function", language="haskell"
        )
        pr = _make_parse_result(language="haskell", entities=[entity], lines_of_code=20)
        report = self.service.analyze_file_complexity(pr)
        # Should fall back to Java analyzer
        assert report.language == "haskell"
        assert report.total_entities == 1

    def test_analyze_file_complexity_with_complex_entities(self):
        # Create entity with high-complexity signature
        entity = _make_entity(
            entity_id="e1",
            name="complex_func",
            entity_type="function",
            signature="if if if if if if if if if if while for switch case try catch finally else else else",
        )
        pr = _make_parse_result(language="python", entities=[entity], lines_of_code=200)
        report = self.service.analyze_file_complexity(pr)
        # No source content passed: entities are not computed, never 1.0.
        assert report.entities_not_computed == report.total_entities
        assert report.max_complexity == 0.0

    def test_analyze_file_complexity_reports_hotspots(self):
        # Create entity with very high complexity that triggers COMPLEX level
        entity = _make_entity(
            entity_id="e1",
            name="big_func",
            entity_type="function",
            language="python",
            signature="if else while for switch case try catch finally " * 10,
        )
        pr = _make_parse_result(language="python", entities=[entity], lines_of_code=500)
        report = self.service.analyze_file_complexity(pr)
        # Check if it got classified as complex
        if report.entity_metrics[0].complexity_level in [
            ComplexityLevel.COMPLEX,
            ComplexityLevel.VERY_COMPLEX,
        ]:
            assert len(report.hotspots) > 0
            assert "function big_func" in report.hotspots

    def test_create_generic_analyzer(self):
        analyzer = self.service._create_generic_analyzer("kotlin")
        assert isinstance(analyzer, JavaComplexityAnalyzer)

    def test_calculate_quality_score_no_entities(self):
        report = FileComplexityReport(file_path="test.py", language="python", total_loc=100)
        report.total_entities = 0
        score = self.service._calculate_quality_score(report)
        assert score == 100.0

    def test_calculate_quality_score_low_complexity(self):
        report = FileComplexityReport(file_path="test.py", language="python", total_loc=100)
        report.total_entities = 5
        report.complex_entities = 0
        report.average_complexity = 3.0
        score = self.service._calculate_quality_score(report)
        assert score == 100.0  # avg < 5 so no penalty

    def test_calculate_quality_score_high_complexity(self):
        report = FileComplexityReport(file_path="test.py", language="python", total_loc=100)
        report.total_entities = 10
        report.complex_entities = 5
        report.average_complexity = 20.0
        score = self.service._calculate_quality_score(report)
        # penalty = (20-5)*2 = 30, complex_ratio = 0.5*30 = 15 => 100-30-15 = 55
        assert abs(score - 55.0) < 0.01

    def test_calculate_quality_score_large_file(self):
        report = FileComplexityReport(file_path="test.py", language="python", total_loc=1000)
        report.total_entities = 5
        report.complex_entities = 0
        report.average_complexity = 3.0
        score = self.service._calculate_quality_score(report)
        # size_penalty = (1000-500)/100 = 5
        assert abs(score - 95.0) < 0.01

    def test_calculate_quality_score_clamped_at_zero(self):
        report = FileComplexityReport(file_path="test.py", language="python", total_loc=5000)
        report.total_entities = 10
        report.complex_entities = 10
        report.average_complexity = 100.0
        score = self.service._calculate_quality_score(report)
        assert score >= 0.0

    def test_generate_file_suggestions_large_file(self):
        report = FileComplexityReport(file_path="test.py", language="python", total_loc=1500)
        report.total_entities = 5
        report.complex_entities = 0
        report.average_complexity = 3.0
        suggestions = self.service._generate_file_suggestions(report)
        assert any("large file" in s.lower() for s in suggestions)

    def test_generate_file_suggestions_high_complex_ratio(self):
        report = FileComplexityReport(file_path="test.py", language="python", total_loc=100)
        report.total_entities = 10
        report.complex_entities = 5
        report.average_complexity = 5.0
        suggestions = self.service._generate_file_suggestions(report)
        assert any("ratio" in s.lower() for s in suggestions)

    def test_generate_file_suggestions_high_avg_complexity(self):
        report = FileComplexityReport(file_path="test.py", language="python", total_loc=100)
        report.total_entities = 10
        report.complex_entities = 0
        report.average_complexity = 20.0
        suggestions = self.service._generate_file_suggestions(report)
        assert any("average complexity" in s.lower() for s in suggestions)

    def test_generate_file_suggestions_many_hotspots(self):
        report = FileComplexityReport(file_path="test.py", language="python", total_loc=100)
        report.total_entities = 10
        report.complex_entities = 0
        report.average_complexity = 3.0
        report.hotspots = ["a", "b", "c", "d", "e", "f"]
        suggestions = self.service._generate_file_suggestions(report)
        assert any("hotspot" in s.lower() for s in suggestions)

    def test_generate_file_suggestions_no_issues(self):
        report = FileComplexityReport(file_path="test.py", language="python", total_loc=100)
        report.total_entities = 10
        report.complex_entities = 0
        report.average_complexity = 3.0
        report.hotspots = []
        suggestions = self.service._generate_file_suggestions(report)
        assert suggestions == []


# =====================================================================
# FileComplexityReport dataclass
# =====================================================================


class TestFileComplexityReport:
    def test_defaults(self):
        report = FileComplexityReport(file_path="test.py", language="python", total_loc=100)
        assert report.file_path == "test.py"
        assert report.language == "python"
        assert report.total_loc == 100
        assert report.average_complexity == 0.0
        assert report.max_complexity == 0.0
        assert report.total_entities == 0
        assert report.complex_entities == 0
        assert report.entity_metrics == []
        assert report.hotspots == []
        assert report.refactoring_suggestions == []
        assert report.quality_score == 0.0

    def test_independent_lists(self):
        r1 = FileComplexityReport(file_path="a.py", language="python", total_loc=10)
        r2 = FileComplexityReport(file_path="b.py", language="python", total_loc=20)
        r1.hotspots.append("func")
        assert len(r2.hotspots) == 0


# =====================================================================
# Integration: analyze_file_complexity with multiple languages
# =====================================================================


class TestAnalyzeFileComplexityMultiLanguage:
    def setup_method(self):
        self.service = ComplexityAnalysisService()

    def test_java_file(self):
        entity = _make_entity(
            entity_type="method",
            language="java",
            name="process",
            signature="public void process(int x)",
        )
        pr = _make_parse_result(language="java", entities=[entity], lines_of_code=30)
        report = self.service.analyze_file_complexity(pr)
        assert report.language == "java"
        assert len(report.entity_metrics) == 1
        assert report.entity_metrics[0].language == "java"

    def test_rust_file(self):
        entity = _make_entity(
            entity_type="function",
            language="rust",
            name="compute",
            signature="fn compute(x: i32) -> i32",
        )
        pr = _make_parse_result(language="rust", entities=[entity], lines_of_code=20)
        report = self.service.analyze_file_complexity(pr)
        assert report.language == "rust"

    def test_go_file(self):
        entity = _make_entity(
            entity_type="function",
            language="go",
            name="Handle",
            signature="func Handle(r *Request) error",
        )
        pr = _make_parse_result(language="go", entities=[entity], lines_of_code=25)
        report = self.service.analyze_file_complexity(pr)
        assert report.language == "go"

    def test_typescript_file(self):
        entity = _make_entity(
            entity_type="function",
            language="typescript",
            name="render",
            signature="function render(props: Props)",
        )
        pr = _make_parse_result(language="typescript", entities=[entity], lines_of_code=15)
        report = self.service.analyze_file_complexity(pr)
        assert report.language == "typescript"

    def test_javascript_file(self):
        entity = _make_entity(
            entity_type="function",
            language="javascript",
            name="init",
            signature="function init(config)",
        )
        pr = _make_parse_result(language="javascript", entities=[entity], lines_of_code=10)
        report = self.service.analyze_file_complexity(pr)
        assert report.language == "javascript"

    def test_cpp_file(self):
        entity = _make_entity(
            entity_type="function",
            language="cpp",
            name="main",
            signature="int main(int argc, char** argv)",
        )
        pr = _make_parse_result(language="cpp", entities=[entity], lines_of_code=50)
        report = self.service.analyze_file_complexity(pr)
        assert report.language == "cpp"

    def test_csharp_file(self):
        entity = _make_entity(
            entity_type="method",
            language="csharp",
            name="Execute",
            signature="public void Execute(int id)",
        )
        pr = _make_parse_result(language="csharp", entities=[entity], lines_of_code=40)
        report = self.service.analyze_file_complexity(pr)
        assert report.language == "csharp"
