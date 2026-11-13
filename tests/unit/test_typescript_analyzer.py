"""Unit tests for processing.analyzers.typescript_analyzer.APITypeScriptProcessor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.core_models import (
    CoreCodeEntity,
    EntityRelation,
    EntityType,
    RelationType,
)
from processing.languages.base import ParseResult, ParserType

# ---------------------------------------------------------------------------
# Sample TypeScript code snippets for testing
# ---------------------------------------------------------------------------

SIMPLE_TS_CODE = """\
import { Component } from '@angular/core';
import axios from 'axios';
import('./lazy-module');
const config = require('config');

@Component({ selector: 'app-root' })
export class AppComponent {
    title: string = 'Hello';
    data: any = null;
    value: unknown = undefined;

    async fetchData(): Promise<void> {
        const result = await axios.get('/api/data');
        this.data = result.data;
    }
}

interface User {
    id: number;
    name: string;
    email?: string;
}

type Status = 'active' | 'inactive';

function greet(user: User): string {
    if (user.name == 'admin') {
        console.log('Admin detected');
        return user.name;
    }
    return `Hello, ${user.name}`;
}

const dangerousCode = eval('alert(1)');
document.write('<script>alert(1)</script>');
element.innerHTML = '<div>unsafe</div>';
"""

TYPED_TS_CODE = """\
interface Repository<T> {
    findById(id: string): Promise<T>;
    save(entity: T): Promise<void>;
}

type Nullable<T> = T | null;
type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };

class UserService implements Repository<User> {
    private cache: Map<string, User> = new Map();

    async findById(id: string): Promise<User> {
        const cached = this.cache.get(id);
        if (cached !== null && cached !== undefined) {
            return cached;
        }
        return this.fetchFromDb(id);
    }

    async save(entity: User): Promise<void> {
        this.cache.set(entity.id, entity);
    }

    private fetchFromDb(id: string): User {
        return { id, name: 'unknown' } as User;
    }
}

export class AdminService extends UserService {
    getAdmins(): Partial<User>[] {
        return [];
    }
}
"""

EMPTY_TS_CODE = ""

MINIMAL_TS_CODE = "const x: number = 42;\n"


# ---------------------------------------------------------------------------
# Helper: build mock ParseResult
# ---------------------------------------------------------------------------


def _make_parse_result(
    success: bool = True,
    entities: list | None = None,
    relationships: list | None = None,
    errors: list | None = None,
    properties: dict | None = None,
    lines_of_code: int = 10,
    parse_time_ms: float = 1.0,
) -> ParseResult:
    return ParseResult(
        success=success,
        file_path="test.ts",
        language="typescript",
        parser_used=ParserType.REGEX_FALLBACK,
        entities=entities or [],
        relationships=relationships or [],
        errors=errors or [],
        lines_of_code=lines_of_code,
        parse_time_ms=parse_time_ms,
        properties=properties or {},
    )


def _make_entity(
    entity_id: str = "e1",
    name: str = "TestEntity",
    entity_type: EntityType = EntityType.FUNCTION,
    properties: dict | None = None,
    signature: str | None = None,
) -> CoreCodeEntity:
    return CoreCodeEntity(
        id=entity_id,
        name=name,
        type=entity_type,
        language="typescript",
        file_path="test.ts",
        line_number=1,
        properties=properties or {},
        signature=signature,
    )


def _make_relationship(
    source_id: str = "e1",
    target_id: str = "e2",
    rel_type: RelationType = RelationType.CALLS,
    properties: dict | None = None,
) -> EntityRelation:
    return EntityRelation(
        source_id=source_id,
        target_id=target_id,
        type=rel_type,
        properties=properties or {},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ts_processor():
    """Create a mock TSLanguageProcessor."""
    proc = MagicMock()
    proc.initialize = AsyncMock()
    proc.parse_file = AsyncMock(return_value=_make_parse_result())
    return proc


@pytest.fixture
def processor(mock_ts_processor):
    """Create an APITypeScriptProcessor with mocked internals."""
    with patch(
        "processing.analyzers.typescript_analyzer.TSLanguageProcessor",
        return_value=mock_ts_processor,
    ):
        from processing.analyzers.typescript_analyzer import APITypeScriptProcessor

        p = APITypeScriptProcessor(file_service=None)
        p.typescript_processor = mock_ts_processor
        return p


@pytest.fixture
def processor_with_file_service(mock_ts_processor, mock_file_service):
    """Processor backed by a mock FileService."""
    with patch(
        "processing.analyzers.typescript_analyzer.TSLanguageProcessor",
        return_value=mock_ts_processor,
    ):
        from processing.analyzers.typescript_analyzer import APITypeScriptProcessor

        p = APITypeScriptProcessor(file_service=mock_file_service)
        p.typescript_processor = mock_ts_processor
        return p


# ---------------------------------------------------------------------------
# Tests: basic properties
# ---------------------------------------------------------------------------


class TestBasicProperties:
    def test_get_supported_extensions(self, processor):
        exts = processor.get_supported_extensions()
        assert ".ts" in exts
        assert ".tsx" in exts
        assert ".d.ts" in exts

    def test_get_language_name(self, processor):
        assert processor.get_language_name() == "typescript"

    def test_can_analyze_ts_file(self, processor):
        assert processor.can_analyze_file("src/app.ts") is True
        assert processor.can_analyze_file("component.tsx") is True
        assert processor.can_analyze_file("types.d.ts") is True

    def test_cannot_analyze_non_ts_file(self, processor):
        assert processor.can_analyze_file("main.py") is False
        assert processor.can_analyze_file("index.js") is False

    async def test_initialize(self, processor, mock_ts_processor):
        await processor.initialize()
        mock_ts_processor.initialize.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests: analyze_snippet
# ---------------------------------------------------------------------------


class TestAnalyzeSnippet:
    async def test_analyze_snippet_success(self, processor, mock_ts_processor):
        mock_ts_processor.parse_file.return_value = _make_parse_result(
            success=True, lines_of_code=20
        )
        result = await processor.analyze_snippet(MINIMAL_TS_CODE, "test.ts")
        assert result.success is True
        assert result.language == "typescript"
        assert result.file_path == "test.ts"

    async def test_analyze_snippet_default_file_path(self, processor, mock_ts_processor):
        mock_ts_processor.parse_file.return_value = _make_parse_result(success=True)
        result = await processor.analyze_snippet(MINIMAL_TS_CODE)
        assert result.file_path == "snippet.ts"

    async def test_analyze_snippet_parse_failure(self, processor, mock_ts_processor):
        mock_ts_processor.parse_file.return_value = _make_parse_result(
            success=False, errors=["Syntax error at line 5"]
        )
        result = await processor.analyze_snippet("invalid{{{", "bad.ts")
        assert result.success is False
        assert "Syntax error at line 5" in result.error

    async def test_analyze_snippet_returns_metadata(self, processor, mock_ts_processor):
        mock_ts_processor.parse_file.return_value = _make_parse_result(
            success=True,
            properties={
                "has_generics": True,
                "has_decorators": False,
                "export_count": 3,
                "type_count": 2,
            },
        )
        result = await processor.analyze_snippet(TYPED_TS_CODE, "typed.ts")
        assert result.success is True
        assert "type_safety" in result.metadata
        assert "complexity" in result.metadata
        assert "typescript_specific" in result.metadata
        ts_specific = result.metadata["typescript_specific"]
        assert ts_specific["has_generics"] is True
        assert ts_specific["export_count"] == 3

    async def test_analyze_snippet_with_entities(self, processor, mock_ts_processor):
        entities = [
            _make_entity("cls1", "UserService", EntityType.CLASS, properties={"is_exported": True}),
            _make_entity("iface1", "Repository", EntityType.INTERFACE),
        ]
        mock_ts_processor.parse_file.return_value = _make_parse_result(
            success=True, entities=entities
        )
        result = await processor.analyze_snippet(TYPED_TS_CODE, "service.ts")
        assert result.success is True
        assert len(result.entities) == 2

    async def test_analyze_snippet_with_relationships(self, processor, mock_ts_processor):
        rels = [
            _make_relationship("cls1", "iface1", RelationType.IMPLEMENTS),
            _make_relationship("cls2", "cls1", RelationType.EXTENDS),
        ]
        entities = [
            _make_entity("cls1", "UserService", EntityType.CLASS),
            _make_entity("cls2", "AdminService", EntityType.CLASS),
            _make_entity("iface1", "Repository", EntityType.INTERFACE),
        ]
        mock_ts_processor.parse_file.return_value = _make_parse_result(
            success=True, entities=entities, relationships=rels
        )
        result = await processor.analyze_snippet(TYPED_TS_CODE, "service.ts")
        assert result.success is True
        assert len(result.relationships) == 2

    async def test_analyze_snippet_catches_exception(self, processor, mock_ts_processor):
        mock_ts_processor.parse_file.side_effect = ValueError("boom")
        result = await processor.analyze_snippet("code", "test.ts")
        assert result.success is False
        assert "boom" in result.error


# ---------------------------------------------------------------------------
# Tests: analyze_file
# ---------------------------------------------------------------------------


class TestAnalyzeFile:
    async def test_analyze_file_reads_with_open(self, processor, mock_ts_processor, tmp_path):
        ts_file = tmp_path / "app.ts"
        ts_file.write_text(MINIMAL_TS_CODE, encoding="utf-8")
        mock_ts_processor.parse_file.return_value = _make_parse_result(success=True)
        result = await processor.analyze_file(str(ts_file))
        assert result.success is True

    async def test_analyze_file_uses_file_service(
        self, processor_with_file_service, mock_ts_processor, mock_file_service
    ):
        mock_file_service.read_file.return_value = MINIMAL_TS_CODE
        mock_ts_processor.parse_file.return_value = _make_parse_result(success=True)
        result = await processor_with_file_service.analyze_file("src/app.ts")
        mock_file_service.read_file.assert_called_once_with("src/app.ts")
        assert result.success is True

    async def test_analyze_file_handles_os_error(self, processor, mock_ts_processor):
        result = await processor.analyze_file("/nonexistent/path/file.ts")
        assert result.success is False
        assert result.error is not None

    async def test_analyze_file_handles_unexpected_error(self, processor, mock_ts_processor):
        mock_ts_processor.parse_file.side_effect = RuntimeError("unexpected")
        # Need to provide a real file so reading works, but parsing fails
        with patch("builtins.open", MagicMock()):
            with patch("builtins.open") as mock_open:
                mock_open.return_value.__enter__ = MagicMock(
                    return_value=MagicMock(read=MagicMock(return_value="code"))
                )
                mock_open.return_value.__exit__ = MagicMock(return_value=False)
                result = await processor.analyze_file("test.ts")
        assert result.success is False


# ---------------------------------------------------------------------------
# Tests: _analyze_type_safety
# ---------------------------------------------------------------------------


class TestAnalyzeTypeSafety:
    async def test_counts_any_types(self, processor):
        code = "let x: any = 1;\nlet y: any = 2;\n"
        metrics = await processor._analyze_type_safety(code, [])
        assert metrics.any_types == 2

    async def test_counts_unknown_types(self, processor):
        code = "let x: unknown = 1;\n"
        metrics = await processor._analyze_type_safety(code, [])
        assert metrics.unknown_types == 1

    async def test_counts_type_assertions(self, processor):
        code = "const x = value as string;\nconst y = <number>value;\n"
        metrics = await processor._analyze_type_safety(code, [])
        assert metrics.type_assertions >= 2

    async def test_counts_explicit_types(self, processor):
        code = "let x: string = 'hello';\nlet y: number = 42;\n"
        metrics = await processor._analyze_type_safety(code, [])
        assert metrics.explicit_types >= 2

    async def test_counts_generic_usage(self, processor):
        code = "const x: Array<T> = [];\nconst y: Map<K, V> = new Map();\n"
        metrics = await processor._analyze_type_safety(code, [])
        assert metrics.generic_usage_count >= 1

    async def test_counts_decorators(self, processor):
        code = "@Component\n@Injectable\nclass MyService {}\n"
        metrics = await processor._analyze_type_safety(code, [])
        assert metrics.decorator_count == 2

    async def test_type_safety_score_no_types(self, processor):
        # No typed elements at all -> score should be 1.0
        code = "// just a comment\n"
        metrics = await processor._analyze_type_safety(code, [])
        assert metrics.type_safety_score == 1.0

    async def test_type_safety_score_with_any(self, processor):
        code = "let x: any = 1;\nlet y: string = 'hi';\n"
        metrics = await processor._analyze_type_safety(code, [])
        # any_types > 0 should penalize score
        assert metrics.type_safety_score < 1.0

    async def test_total_types_counts_entities(self, processor):
        entities = [
            _make_entity("i1", "IUser", EntityType.INTERFACE),
            _make_entity("c1", "MyClass", EntityType.CLASS),
            _make_entity("e1", "Status", EntityType.ENUM),
            _make_entity("f1", "doStuff", EntityType.FUNCTION),  # not counted
        ]
        metrics = await processor._analyze_type_safety("", entities)
        assert metrics.total_types == 3  # interface + class + enum


# ---------------------------------------------------------------------------
# Tests: _analyze_complexity
# ---------------------------------------------------------------------------


class TestAnalyzeComplexity:
    async def test_cyclomatic_complexity_base(self, processor):
        code = "const x = 1;\n"
        metrics = await processor._analyze_complexity(code, [])
        # base complexity is 1
        assert metrics.cyclomatic_complexity >= 1.0

    async def test_cyclomatic_complexity_branches(self, processor):
        code = "if (x) { } else { } while (y) { } for (;;) { }\n"
        metrics = await processor._analyze_complexity(code, [])
        assert metrics.cyclomatic_complexity > 1.0

    async def test_type_complexity_unions(self, processor):
        code = "type A = string | number | boolean;\n"
        metrics = await processor._analyze_complexity(code, [])
        assert metrics.type_complexity > 0

    async def test_generic_complexity(self, processor):
        code = "function f<T>(x: T): T { return x; }\nfunction g<U>(y: U): U { return y; }\n"
        metrics = await processor._analyze_complexity(code, [])
        assert metrics.generic_complexity > 0

    async def test_inheritance_depth_with_extends(self, processor):
        entities = [
            _make_entity("c1", "Base", EntityType.CLASS, properties={"extends": "Object"}),
        ]
        metrics = await processor._analyze_complexity("", entities)
        assert metrics.inheritance_depth == 1

    async def test_inheritance_depth_no_extends(self, processor):
        entities = [
            _make_entity("c1", "Base", EntityType.CLASS, properties={}),
        ]
        metrics = await processor._analyze_complexity("", entities)
        assert metrics.inheritance_depth == 0

    async def test_coupling_score(self, processor):
        code = "import { A } from 'a';\nimport { B } from 'b';\nexport class C {}\n"
        metrics = await processor._analyze_complexity(code, [])
        assert metrics.coupling_score > 0

    async def test_maintainability_index(self, processor):
        code = "const x = 1;\n"
        metrics = await processor._analyze_complexity(code, [])
        assert 0 <= metrics.maintainability_index <= 100


# ---------------------------------------------------------------------------
# Tests: _analyze_security_patterns
# ---------------------------------------------------------------------------


class TestAnalyzeSecurityPatterns:
    async def test_detects_eval(self, processor):
        code = "eval('alert(1)');\n"
        patterns = await processor._analyze_security_patterns(code)
        eval_patterns = [p for p in patterns if p.type == "eval_usage"]
        assert len(eval_patterns) == 1
        assert eval_patterns[0].severity == "high"

    async def test_detects_inner_html(self, processor):
        code = "element.innerHTML = '<div>test</div>';\n"
        patterns = await processor._analyze_security_patterns(code)
        inner_html = [p for p in patterns if p.type == "inner_html"]
        assert len(inner_html) == 1
        assert inner_html[0].severity == "high"

    async def test_detects_document_write(self, processor):
        code = "document.write('hello');\n"
        patterns = await processor._analyze_security_patterns(code)
        doc_write = [p for p in patterns if p.type == "document_write"]
        assert len(doc_write) == 1

    async def test_detects_loose_equality(self, processor):
        code = "if (x == y) { }\nif (a != b) { }\n"
        patterns = await processor._analyze_security_patterns(code)
        loose = [p for p in patterns if p.type == "loose_equality"]
        assert len(loose) >= 2
        assert all(p.severity == "medium" for p in loose)

    async def test_detects_console_log(self, processor):
        code = "console.log('debug');\n"
        patterns = await processor._analyze_security_patterns(code)
        logs = [p for p in patterns if p.type == "console_log"]
        assert len(logs) == 1
        assert logs[0].severity == "medium"

    async def test_detects_any_type(self, processor):
        code = "let x: any = 1;\n"
        patterns = await processor._analyze_security_patterns(code)
        any_patterns = [p for p in patterns if p.type == "any_type_usage"]
        assert len(any_patterns) == 1
        assert any_patterns[0].severity == "medium"

    async def test_line_numbers_correct(self, processor):
        code = "const a = 1;\nconst b = 2;\neval('x');\n"
        patterns = await processor._analyze_security_patterns(code)
        eval_patterns = [p for p in patterns if p.type == "eval_usage"]
        assert len(eval_patterns) == 1
        assert eval_patterns[0].line_number == 3

    async def test_no_patterns_in_clean_code(self, processor):
        code = "const x: string = 'hello';\nif (x === 'world') { }\n"
        patterns = await processor._analyze_security_patterns(code)
        # Only potentially console_log or loose_equality would match - this code has neither
        dangerous = [
            p for p in patterns if p.type in ("eval_usage", "inner_html", "document_write")
        ]
        assert len(dangerous) == 0

    async def test_match_text_in_metadata(self, processor):
        code = "eval('code');\n"
        patterns = await processor._analyze_security_patterns(code)
        assert len(patterns) >= 1
        assert "match_text" in patterns[0].metadata

    async def test_multiple_patterns_in_same_code(self, processor):
        patterns = await processor._analyze_security_patterns(SIMPLE_TS_CODE)
        pattern_types = {p.type for p in patterns}
        assert "eval_usage" in pattern_types
        assert "inner_html" in pattern_types
        assert "document_write" in pattern_types
        assert "console_log" in pattern_types


# ---------------------------------------------------------------------------
# Tests: _extract_dependencies
# ---------------------------------------------------------------------------


class TestExtractDependencies:
    async def test_extracts_import_from(self, processor):
        code = "import { Component } from '@angular/core';\n"
        deps = await processor._extract_dependencies(code, "test.ts")
        assert len(deps) == 1
        assert deps[0].module_name == "@angular/core"
        assert deps[0].import_type == "import"
        assert deps[0].is_external is True

    async def test_extracts_default_import(self, processor):
        code = "import axios from 'axios';\n"
        deps = await processor._extract_dependencies(code, "test.ts")
        assert len(deps) == 1
        assert deps[0].module_name == "axios"

    async def test_extracts_dynamic_import(self, processor):
        code = "const mod = import('./local-module');\nconst ext = import('lodash');\n"
        deps = await processor._extract_dependencies(code, "test.ts")
        # Only external (non-relative) deps are included
        external = [d for d in deps if d.is_external]
        assert any(d.module_name == "lodash" for d in external)

    async def test_extracts_require(self, processor):
        code = "const config = require('config');\n"
        deps = await processor._extract_dependencies(code, "test.ts")
        assert len(deps) == 1
        assert deps[0].module_name == "config"
        assert deps[0].import_type == "require"

    async def test_ignores_relative_imports(self, processor):
        code = "import { helper } from './utils';\nimport { Base } from '../base';\n"
        deps = await processor._extract_dependencies(code, "test.ts")
        assert len(deps) == 0

    async def test_scoped_packages(self, processor):
        code = "import { Injectable } from '@nestjs/common';\n"
        deps = await processor._extract_dependencies(code, "test.ts")
        assert len(deps) == 1
        assert deps[0].module_name == "@nestjs/common"

    async def test_deduplicates_imports(self, processor):
        code = "import { A } from 'lodash';\nimport { B } from 'lodash';\n"
        deps = await processor._extract_dependencies(code, "test.ts")
        assert len(deps) == 1

    async def test_line_numbers(self, processor):
        code = "// header\nimport axios from 'axios';\n"
        deps = await processor._extract_dependencies(code, "test.ts")
        assert deps[0].line_number == 2

    async def test_full_path_in_metadata(self, processor):
        code = "import { map } from 'rxjs/operators';\n"
        deps = await processor._extract_dependencies(code, "test.ts")
        assert deps[0].metadata["full_path"] == "rxjs/operators"

    async def test_multiple_dependency_types(self, processor):
        deps = await processor._extract_dependencies(SIMPLE_TS_CODE, "test.ts")
        import_types = {d.import_type for d in deps}
        assert "import" in import_types


# ---------------------------------------------------------------------------
# Tests: _process_entities
# ---------------------------------------------------------------------------


class TestProcessEntities:
    async def test_enriches_class_entity(self, processor):
        from processing.analyzers.typescript_analyzer import TypeSafetyMetrics

        entities = [
            _make_entity("c1", "MyClass", EntityType.CLASS, properties={"is_exported": True}),
        ]
        metrics = TypeSafetyMetrics()
        processed = await processor._process_entities(entities, metrics)
        assert len(processed) == 1
        assert "type_safety_impact" in processed[0].properties

    async def test_enriches_interface_entity(self, processor):
        from processing.analyzers.typescript_analyzer import TypeSafetyMetrics

        entities = [
            _make_entity(
                "i1",
                "IUser",
                EntityType.INTERFACE,
                properties={"generic_parameters": ["T"]},
            ),
        ]
        metrics = TypeSafetyMetrics()
        processed = await processor._process_entities(entities, metrics)
        assert processed[0].properties["type_safety_impact"]["uses_generics"] is True

    async def test_does_not_enrich_function(self, processor):
        from processing.analyzers.typescript_analyzer import TypeSafetyMetrics

        entities = [
            _make_entity("f1", "doStuff", EntityType.FUNCTION),
        ]
        metrics = TypeSafetyMetrics()
        processed = await processor._process_entities(entities, metrics)
        assert "type_safety_impact" not in processed[0].properties

    async def test_preserves_entity_fields(self, processor):
        from processing.analyzers.typescript_analyzer import TypeSafetyMetrics

        entities = [
            _make_entity("e1", "TestFunc", EntityType.FUNCTION),
        ]
        metrics = TypeSafetyMetrics()
        processed = await processor._process_entities(entities, metrics)
        assert processed[0].id == "e1"
        assert processed[0].name == "TestFunc"
        assert processed[0].type == EntityType.FUNCTION
        assert processed[0].language == "typescript"


# ---------------------------------------------------------------------------
# Tests: _process_relationships
# ---------------------------------------------------------------------------


class TestProcessRelationships:
    async def test_implements_coupling_strength(self, processor):
        rels = [_make_relationship("c1", "i1", RelationType.IMPLEMENTS)]
        entities = [_make_entity("c1", "C", EntityType.CLASS)]
        processed = await processor._process_relationships(rels, entities)
        assert processed[0].properties["coupling_strength"] == "high"

    async def test_extends_coupling_strength(self, processor):
        rels = [_make_relationship("c2", "c1", RelationType.EXTENDS)]
        entities = []
        processed = await processor._process_relationships(rels, entities)
        assert processed[0].properties["coupling_strength"] == "very_high"

    async def test_imports_coupling_strength(self, processor):
        rels = [_make_relationship("m1", "m2", RelationType.IMPORTS)]
        processed = await processor._process_relationships(rels, [])
        assert processed[0].properties["coupling_strength"] == "medium"

    async def test_generic_constraint_relationship(self, processor):
        rels = [_make_relationship("g1", "t1", RelationType.GENERIC_OF)]
        processed = await processor._process_relationships(rels, [])
        # GENERIC_OF is not "generic_constraint" so no coupling_strength set
        assert len(processed) == 1

    async def test_preserves_existing_properties(self, processor):
        rels = [
            _make_relationship("c1", "i1", RelationType.IMPLEMENTS, properties={"custom": "value"})
        ]
        processed = await processor._process_relationships(rels, [])
        assert processed[0].properties["custom"] == "value"
        assert processed[0].properties["coupling_strength"] == "high"


# ---------------------------------------------------------------------------
# Tests: _aggregate_metrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_aggregates_type_safety(self, processor):
        from processing.analyzers.typescript_analyzer import (
            TypeSafetyMetrics,
            TypeScriptComplexityMetrics,
        )

        total_ts = TypeSafetyMetrics()
        total_cx = TypeScriptComplexityMetrics()
        file_metrics = {
            "type_safety": {
                "total_types": 5,
                "explicit_types": 10,
                "any_types": 2,
                "unknown_types": 1,
                "type_assertions": 3,
                "generic_usage_count": 4,
                "decorator_count": 1,
            },
            "complexity": {
                "cyclomatic_complexity": 8.0,
                "type_complexity": 3.0,
                "generic_complexity": 1.5,
                "inheritance_depth": 2,
                "coupling_score": 0.5,
            },
        }
        processor._aggregate_metrics(file_metrics, total_ts, total_cx)
        assert total_ts.total_types == 5
        assert total_ts.explicit_types == 10
        assert total_ts.any_types == 2
        assert total_cx.cyclomatic_complexity == 8.0
        assert total_cx.inheritance_depth == 2

    def test_aggregates_multiple_files(self, processor):
        from processing.analyzers.typescript_analyzer import (
            TypeSafetyMetrics,
            TypeScriptComplexityMetrics,
        )

        total_ts = TypeSafetyMetrics()
        total_cx = TypeScriptComplexityMetrics()
        for _ in range(3):
            processor._aggregate_metrics(
                {
                    "type_safety": {"total_types": 2, "any_types": 1},
                    "complexity": {"cyclomatic_complexity": 5.0, "inheritance_depth": 1},
                },
                total_ts,
                total_cx,
            )
        assert total_ts.total_types == 6
        assert total_ts.any_types == 3
        assert total_cx.cyclomatic_complexity == 15.0
        assert total_cx.inheritance_depth == 1  # max, not sum

    def test_handles_missing_keys(self, processor):
        from processing.analyzers.typescript_analyzer import (
            TypeSafetyMetrics,
            TypeScriptComplexityMetrics,
        )

        total_ts = TypeSafetyMetrics()
        total_cx = TypeScriptComplexityMetrics()
        processor._aggregate_metrics({}, total_ts, total_cx)
        assert total_ts.total_types == 0
        assert total_cx.cyclomatic_complexity == 0.0


# ---------------------------------------------------------------------------
# Tests: _calculate_project_metrics
# ---------------------------------------------------------------------------


class TestCalculateProjectMetrics:
    async def test_project_metrics_basic(self, processor):
        from processing.analyzers.typescript_analyzer import (
            TypeSafetyMetrics,
            TypeScriptComplexityMetrics,
        )

        entities = [
            _make_entity("c1", "A", EntityType.CLASS),
            _make_entity("f1", "fn", EntityType.FUNCTION),
        ]
        rels = [_make_relationship("c1", "f1", RelationType.CALLS)]
        ts = TypeSafetyMetrics(total_types=2, explicit_types=5, any_types=1)
        cx = TypeScriptComplexityMetrics(cyclomatic_complexity=10.0)

        result = await processor._calculate_project_metrics(entities, rels, ts, cx)
        assert "type_safety" in result
        assert "complexity" in result
        assert "project_overview" in result
        assert result["project_overview"]["total_entities"] == 2
        assert result["project_overview"]["total_relationships"] == 1

    async def test_entity_type_counts(self, processor):
        from processing.analyzers.typescript_analyzer import (
            TypeSafetyMetrics,
            TypeScriptComplexityMetrics,
        )

        entities = [
            _make_entity("c1", "A", EntityType.CLASS),
            _make_entity("c2", "B", EntityType.CLASS),
            _make_entity("f1", "fn", EntityType.FUNCTION),
        ]
        result = await processor._calculate_project_metrics(
            entities, [], TypeSafetyMetrics(), TypeScriptComplexityMetrics()
        )
        entity_types = result["project_overview"]["entity_types"]
        assert entity_types.get("class", 0) == 2
        assert entity_types.get("function", 0) == 1

    async def test_empty_project(self, processor):
        from processing.analyzers.typescript_analyzer import (
            TypeSafetyMetrics,
            TypeScriptComplexityMetrics,
        )

        result = await processor._calculate_project_metrics(
            [], [], TypeSafetyMetrics(), TypeScriptComplexityMetrics()
        )
        assert result["project_overview"]["total_entities"] == 0
        assert result["project_overview"]["total_relationships"] == 0


# ---------------------------------------------------------------------------
# Tests: _count_entity_types and _count_relationship_types
# ---------------------------------------------------------------------------


class TestCountHelpers:
    def test_count_entity_types(self, processor):
        entities = [
            _make_entity("c1", "A", EntityType.CLASS),
            _make_entity("c2", "B", EntityType.CLASS),
            _make_entity("f1", "fn", EntityType.FUNCTION),
            _make_entity("i1", "I", EntityType.INTERFACE),
        ]
        counts = processor._count_entity_types(entities)
        assert counts["class"] == 2
        assert counts["function"] == 1
        assert counts["interface"] == 1

    def test_count_relationship_types(self, processor):
        rels = [
            _make_relationship("a", "b", RelationType.CALLS),
            _make_relationship("c", "d", RelationType.CALLS),
            _make_relationship("e", "f", RelationType.IMPORTS),
        ]
        counts = processor._count_relationship_types(rels)
        assert counts["calls"] == 2
        assert counts["imports"] == 1


# ---------------------------------------------------------------------------
# Tests: _create_error_result
# ---------------------------------------------------------------------------


class TestCreateErrorResult:
    def test_error_result_fields(self, processor):
        result = processor._create_error_result("bad_file.ts", "Something went wrong")
        assert result.success is False
        assert result.language == "typescript"
        assert result.file_path == "bad_file.ts"
        assert result.error == "Something went wrong"
        assert result.entities == []
        assert result.relationships == []
        assert "error_timestamp" in result.metadata


# ---------------------------------------------------------------------------
# Tests: analyze_directory
# ---------------------------------------------------------------------------


class TestAnalyzeDirectory:
    async def test_analyze_directory_empty(self, processor, tmp_path):
        result = await processor.analyze_directory(str(tmp_path))
        assert result.success is True
        assert result.metadata["files_analyzed"] == 0

    async def test_analyze_directory_with_files(self, processor, mock_ts_processor, tmp_path):
        (tmp_path / "app.ts").write_text(MINIMAL_TS_CODE, encoding="utf-8")
        (tmp_path / "util.ts").write_text("export const x = 1;\n", encoding="utf-8")
        mock_ts_processor.parse_file.return_value = _make_parse_result(
            success=True,
            entities=[_make_entity("e1", "x", EntityType.VARIABLE)],
        )
        result = await processor.analyze_directory(str(tmp_path))
        assert result.success is True
        assert result.metadata["total_files"] == 2

    async def test_analyze_directory_with_file_service(
        self, processor_with_file_service, mock_ts_processor, mock_file_service
    ):
        mock_file_service.find_files = MagicMock(return_value=["a.ts", "b.tsx"])
        mock_file_service.read_file.return_value = MINIMAL_TS_CODE
        mock_ts_processor.parse_file.return_value = _make_parse_result(success=True)
        result = await processor_with_file_service.analyze_directory("/some/dir")
        assert result.success is True

    async def test_analyze_directory_handles_error(self, processor):
        result = await processor.analyze_directory("/nonexistent/dir/that/should/not/exist")
        # Should still return a result (possibly with 0 files analyzed)
        assert result is not None

    async def test_analyze_directory_tracks_failed_files(
        self, processor, mock_ts_processor, tmp_path
    ):
        (tmp_path / "good.ts").write_text(MINIMAL_TS_CODE, encoding="utf-8")
        (tmp_path / "bad.ts").write_text("invalid", encoding="utf-8")

        call_count = 0

        async def alternating_parse(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                return _make_parse_result(success=False, errors=["parse error"])
            return _make_parse_result(success=True)

        mock_ts_processor.parse_file.side_effect = alternating_parse
        result = await processor.analyze_directory(str(tmp_path))
        assert result.success is True


# ---------------------------------------------------------------------------
# Tests: validate_syntax (inherited from BaseCodeProcessor)
# ---------------------------------------------------------------------------


class TestValidateSyntax:
    async def test_valid_syntax(self, processor, mock_ts_processor):
        mock_ts_processor.parse_file.return_value = _make_parse_result(success=True)
        valid = await processor.validate_syntax(MINIMAL_TS_CODE, "test.ts")
        assert valid is True

    async def test_invalid_syntax(self, processor, mock_ts_processor):
        mock_ts_processor.parse_file.return_value = _make_parse_result(
            success=False, errors=["Syntax error"]
        )
        valid = await processor.validate_syntax("{{invalid", "test.ts")
        assert valid is False


# ---------------------------------------------------------------------------
# Tests: dataclass models
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_type_safety_metrics_defaults(self):
        from processing.analyzers.typescript_analyzer import TypeSafetyMetrics

        m = TypeSafetyMetrics()
        assert m.total_types == 0
        assert m.type_safety_score == 0.0
        assert m.decorator_count == 0

    def test_complexity_metrics_defaults(self):
        from processing.analyzers.typescript_analyzer import TypeScriptComplexityMetrics

        m = TypeScriptComplexityMetrics()
        assert m.cyclomatic_complexity == 0.0
        assert m.maintainability_index == 0.0
        assert m.inheritance_depth == 0

    def test_type_safety_metrics_dict(self):
        from processing.analyzers.typescript_analyzer import TypeSafetyMetrics

        m = TypeSafetyMetrics(total_types=5, any_types=2, type_safety_score=0.8)
        d = m.__dict__
        assert d["total_types"] == 5
        assert d["any_types"] == 2
        assert d["type_safety_score"] == 0.8
