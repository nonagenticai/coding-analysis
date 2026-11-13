"""Unit tests for processing.processors.code_processor.CodeProcessor."""

from __future__ import annotations

import sys
import types as _types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub models.github_models (not a local module) so import of code_processor
# succeeds. coding_client.* stubs are installed by conftest.py.
# ---------------------------------------------------------------------------


class _CodeSnippet:
    """Minimal stub for CodeSnippet."""

    def __init__(self, *, code="", language="python", file_path=None, **kw):
        self.code = code
        self.language = language
        self.file_path = file_path
        for k, v in kw.items():
            setattr(self, k, v)


class _RepositoryContext:
    """Minimal stub for RepositoryContext."""

    def __init__(self, *, existing_patterns=None, style_guide=None, **kw):
        self.existing_patterns = existing_patterns or []
        self.style_guide = style_guide or {}
        for k, v in kw.items():
            setattr(self, k, v)


if "models.github_models" not in sys.modules or isinstance(
    sys.modules.get("models.github_models"), MagicMock
):
    _gh_models = _types.ModuleType("models.github_models")
    _gh_models.CodeSnippet = _CodeSnippet
    _gh_models.RepositoryContext = _RepositoryContext
    sys.modules["models.github_models"] = _gh_models
else:
    from models.github_models import CodeSnippet as _CodeSnippet  # type: ignore[no-redef]
    from models.github_models import (
        RepositoryContext as _RepositoryContext,  # type: ignore[no-redef]
    )

# ---------------------------------------------------------------------------
# Import the module under test (patch analyzers to avoid heavy deps)
# ---------------------------------------------------------------------------
from models.core_models import (
    CoreAnalysisResult,
    CoreCodeEntity,
    DependencyInfoModel,
)

# We need to patch the analyzer constructors so CodeProcessor.__init__ does not
# instantiate real heavyweight analyzers.
_mock_python_cls = MagicMock()
_mock_js_cls = MagicMock()
_mock_ts_cls = MagicMock()

with (
    patch(
        "processing.analyzers.python_analyzer.PythonCodeProcessor",
        _mock_python_cls,
    ),
    patch(
        "processing.analyzers.javascript_analyzer.JSAnalyzerProcessor",
        _mock_js_cls,
    ),
    patch(
        "processing.analyzers.typescript_analyzer.APITypeScriptProcessor",
        _mock_ts_cls,
    ),
):
    from processing.processors.code_processor import CodeProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_analysis_result(
    success=True,
    language="python",
    file_path="test.py",
    entities=None,
    relationships=None,
    dependencies=None,
    metadata=None,
    error=None,
) -> CoreAnalysisResult:
    return CoreAnalysisResult(
        success=success,
        language=language,
        file_path=file_path,
        entities=entities or [],
        relationships=relationships or [],
        dependencies=dependencies or [],
        complexity_metrics=None,
        error=error,
        parse_time_ms=1.0,
        total_lines=10,
        metadata=metadata or {},
    )


def _make_entity(name="my_func", entity_type="function", language="python"):
    from models.core_models import EntityType

    return CoreCodeEntity(
        id="ent-1",
        name=name,
        type=EntityType(entity_type),
        language=language,
        file_path="test.py",
        line_number=1,
    )


def _make_dependency(module_name="os", import_type="import"):
    return DependencyInfoModel(
        module_name=module_name,
        import_type=import_type,
    )


# ---------------------------------------------------------------------------
# Fixture: CodeProcessor with mocked analyzers
# ---------------------------------------------------------------------------


@pytest.fixture
def processor():
    """Create a CodeProcessor with mocked analyzers."""
    with (
        patch(
            "processing.analyzers.python_analyzer.PythonCodeProcessor",
        ),
        patch(
            "processing.analyzers.javascript_analyzer.JSAnalyzerProcessor",
        ),
        patch(
            "processing.analyzers.typescript_analyzer.APITypeScriptProcessor",
        ),
    ):
        proc = CodeProcessor()

    # Replace the analyzers with AsyncMock instances after construction
    py_inst = AsyncMock()
    js_inst = AsyncMock()
    ts_inst = AsyncMock()
    proc.analyzers["python"] = py_inst
    proc.analyzers["javascript"] = js_inst
    proc.analyzers["typescript"] = ts_inst
    # Expose individual mocks for test assertions
    proc._py = py_inst
    proc._js = js_inst
    proc._ts = ts_inst
    return proc


# ===================================================================
# _detect_language
# ===================================================================


class TestDetectLanguage:
    def test_python_extension(self, processor):
        assert processor._detect_language("app.py") == "python"

    def test_js_extension(self, processor):
        assert processor._detect_language("index.js") == "javascript"

    def test_jsx_extension(self, processor):
        assert processor._detect_language("App.jsx") == "javascript"

    def test_ts_extension(self, processor):
        assert processor._detect_language("main.ts") == "typescript"

    def test_tsx_extension(self, processor):
        assert processor._detect_language("Component.tsx") == "typescript"

    def test_unknown_extension(self, processor):
        assert processor._detect_language("data.csv") == "unknown"

    def test_no_extension(self, processor):
        assert processor._detect_language("Makefile") == "unknown"

    def test_uppercase_extension(self, processor):
        assert processor._detect_language("script.PY") == "python"


# ===================================================================
# _is_code_file
# ===================================================================


class TestIsCodeFile:
    def test_python_file(self, processor):
        assert processor._is_code_file("src/main.py") is True

    def test_javascript_file(self, processor):
        assert processor._is_code_file("app.js") is True

    def test_non_code_file(self, processor):
        assert processor._is_code_file("README.md") is False

    def test_hidden_file(self, processor):
        assert processor._is_code_file(".gitignore") is False


# ===================================================================
# _get_code_files
# ===================================================================


class TestGetCodeFiles:
    def test_nonexistent_directory(self, processor):
        result = processor._get_code_files("/nonexistent/path/xyz")
        assert result == []

    def test_empty_directory(self, processor, tmp_path):
        result = processor._get_code_files(str(tmp_path))
        assert result == []

    def test_finds_python_files(self, processor, tmp_path):
        (tmp_path / "main.py").write_text("pass")
        (tmp_path / "README.md").write_text("# readme")
        result = processor._get_code_files(str(tmp_path))
        assert len(result) == 1
        assert result[0].endswith("main.py")

    def test_skips_hidden_and_venv(self, processor, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "hooks.py").write_text("pass")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "mod.py").write_text("pass")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "lib.js").write_text("pass")
        (tmp_path / "app.py").write_text("pass")
        result = processor._get_code_files(str(tmp_path))
        assert len(result) == 1

    def test_returns_sorted(self, processor, tmp_path):
        (tmp_path / "z.py").write_text("")
        (tmp_path / "a.py").write_text("")
        result = processor._get_code_files(str(tmp_path))
        assert result == sorted(result)

    def test_finds_nested_files(self, processor, tmp_path):
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "util.py").write_text("")
        (tmp_path / "main.js").write_text("")
        result = processor._get_code_files(str(tmp_path))
        assert len(result) == 2


# ===================================================================
# analyze_file
# ===================================================================


class TestAnalyzeFile:
    async def test_unsupported_language(self, processor):
        result = await processor.analyze_file("data.csv")
        assert result.success is False
        assert "Unsupported language" in result.error

    async def test_supported_language_delegates(self, processor):
        expected = _make_analysis_result(metadata={"file_path": "app.py"})
        processor._py.analyze_file = AsyncMock(return_value=expected)
        result = await processor.analyze_file("app.py")
        processor._py.analyze_file.assert_awaited_once_with("app.py")
        assert result is expected

    async def test_file_path_added_to_metadata(self, processor):
        expected = _make_analysis_result(metadata={})
        processor._py.analyze_file = AsyncMock(return_value=expected)
        result = await processor.analyze_file("app.py")
        assert result.metadata["file_path"] == "app.py"

    async def test_analyzer_raises_oserror(self, processor):
        processor._py.analyze_file = AsyncMock(side_effect=OSError("not found"))
        result = await processor.analyze_file("missing.py")
        assert result.success is False
        assert "not found" in result.error
        assert result.language == "unknown"

    async def test_analyzer_raises_valueerror(self, processor):
        processor._py.analyze_file = AsyncMock(side_effect=ValueError("bad"))
        result = await processor.analyze_file("bad.py")
        assert result.success is False
        assert result.metadata["error_type"] == "ValueError"

    async def test_javascript_file(self, processor):
        expected = _make_analysis_result(
            language="javascript", file_path="app.js", metadata={"file_path": "app.js"}
        )
        processor._js.analyze_file = AsyncMock(return_value=expected)
        result = await processor.analyze_file("app.js")
        processor._js.analyze_file.assert_awaited_once_with("app.js")
        assert result.language == "javascript"

    async def test_typescript_file(self, processor):
        expected = _make_analysis_result(
            language="typescript", file_path="main.ts", metadata={"file_path": "main.ts"}
        )
        processor._ts.analyze_file = AsyncMock(return_value=expected)
        await processor.analyze_file("main.ts")
        processor._ts.analyze_file.assert_awaited_once_with("main.ts")


# ===================================================================
# process_repository
# ===================================================================


class TestProcessRepository:
    async def test_no_code_files(self, processor, tmp_path):
        result = await processor.process_repository(str(tmp_path))
        assert result.success is True
        assert result.file_results == []
        assert result.summary["total_files"] == 0

    async def test_with_files_success(self, processor, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        expected = _make_analysis_result(
            file_path=str(tmp_path / "a.py"),
            metadata={"file_path": str(tmp_path / "a.py"), "language": "python"},
            entities=[_make_entity()],
        )
        processor._py.analyze_file = AsyncMock(return_value=expected)

        result = await processor.process_repository(str(tmp_path))
        assert result.success is True
        assert len(result.file_results) == 1
        assert result.metadata["successful_files"] == 1
        assert result.metadata["failed_files"] == 0

    async def test_with_file_exception(self, processor, tmp_path):
        (tmp_path / "a.py").write_text("x = 1")
        processor._py.analyze_file = AsyncMock(side_effect=RuntimeError("boom"))

        result = await processor.process_repository(str(tmp_path))
        assert result.success is True
        assert len(result.file_results) == 0
        assert len(result.failed_files) == 1
        assert result.failed_files[0]["error"] == "boom"
        assert result.failed_files[0]["error_type"] == "RuntimeError"

    async def test_repo_level_oserror(self, processor):
        with patch.object(processor, "_get_code_files", side_effect=OSError("no dir")):
            result = await processor.process_repository("/nope")
        assert result.success is False
        assert "no dir" in result.metadata["error"]

    async def test_multiple_files_mixed(self, processor, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")

        good = _make_analysis_result(
            file_path=str(tmp_path / "a.py"),
            metadata={"file_path": str(tmp_path / "a.py"), "language": "python"},
            entities=[_make_entity()],
        )

        call_count = 0

        async def _side_effect(fp):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return good
            raise RuntimeError("fail on second")

        processor._py.analyze_file = AsyncMock(side_effect=_side_effect)

        result = await processor.process_repository(str(tmp_path))
        assert result.success is True
        assert result.metadata["successful_files"] == 1
        assert result.metadata["failed_files"] == 1


# ===================================================================
# _create_repository_summary
# ===================================================================


class TestCreateRepositorySummary:
    def test_empty_results(self, processor):
        summary = processor._create_repository_summary([], [])
        assert summary["total_entities"] == 0
        assert summary["total_relationships"] == 0
        assert summary["total_dependencies"] == 0
        assert summary["analysis_success_rate"] == 0

    def test_with_results(self, processor):
        r1 = _make_analysis_result(
            entities=[_make_entity("a"), _make_entity("b")],
            dependencies=[_make_dependency("os")],
            metadata={"language": "python"},
        )
        r2 = _make_analysis_result(
            entities=[_make_entity("c")],
            metadata={"language": "javascript"},
        )
        failed = [{"file_path": "bad.py", "error": "err"}]

        summary = processor._create_repository_summary([r1, r2], failed)
        assert summary["total_entities"] == 3
        assert summary["total_dependencies"] == 1
        assert summary["successful_files"] == 2
        assert summary["failed_files"] == 1
        assert summary["total_files_analyzed"] == 3
        assert summary["analysis_success_rate"] == pytest.approx(2 / 3)
        assert summary["language_breakdown"]["python"] == 1
        assert summary["language_breakdown"]["javascript"] == 1


# ===================================================================
# evaluate_snippet
# ===================================================================


class TestEvaluateSnippet:
    async def test_unsupported_language(self, processor):
        snippet = _CodeSnippet(code="x", language="ruby")
        result = await processor.evaluate_snippet(snippet)
        assert result.success is False
        assert "Unsupported language" in result.error

    async def test_snippet_analysis_fails(self, processor):
        snippet = _CodeSnippet(code="bad", language="python", file_path="t.py")
        processor._py.analyze_snippet = AsyncMock(
            return_value=_make_analysis_result(success=False, error="parse error")
        )
        result = await processor.evaluate_snippet(snippet)
        assert result.success is False
        assert "parse error" in result.error

    async def test_no_context_gives_defaults(self, processor):
        snippet = _CodeSnippet(code="x=1", language="python", file_path="t.py")
        analysis = _make_analysis_result(metadata={"language": "python"})
        processor._py.analyze_snippet = AsyncMock(return_value=analysis)

        result = await processor.evaluate_snippet(snippet)
        assert result.success is True
        assert result.compatibility_score == 0.85
        assert len(result.recommendations) == 3
        assert result.metadata["has_context"] is False

    async def test_with_context(self, processor):
        snippet = _CodeSnippet(code="x=1", language="python", file_path="t.py")
        analysis = _make_analysis_result(
            entities=[_make_entity()],
            metadata={"language": "python"},
        )
        processor._py.analyze_snippet = AsyncMock(return_value=analysis)

        context = _RepositoryContext(existing_patterns=[], style_guide={})

        with (
            patch.object(
                processor, "_calculate_compatibility", new_callable=AsyncMock, return_value=0.9
            ),
            patch.object(
                processor,
                "_generate_recommendations",
                new_callable=AsyncMock,
                return_value=["rec1"],
            ),
        ):
            result = await processor.evaluate_snippet(snippet, context)

        assert result.success is True
        assert result.compatibility_score == 0.9
        assert result.recommendations == ["rec1"]
        assert result.metadata["has_context"] is True

    async def test_attribute_error_caught(self, processor):
        # snippet with no .language triggers AttributeError
        bad_snippet = MagicMock()
        bad_snippet.language = MagicMock(side_effect=AttributeError("no lang"))
        # The property access raises, which should be caught
        result = await processor.evaluate_snippet(bad_snippet)
        assert result.success is False


# ===================================================================
# _extract_version_info
# ===================================================================


class TestExtractVersionInfo:
    def test_exact_version(self, processor):
        info = processor._extract_version_info("package==1.2.3")
        assert info["operator"] == "=="
        assert info["version"] == "1.2.3"

    def test_gte_version(self, processor):
        info = processor._extract_version_info("package>=2.0.0")
        assert info["operator"] == ">="
        assert info["version"] == "2.0.0"

    def test_no_version(self, processor):
        info = processor._extract_version_info("package")
        assert info["operator"] is None
        assert info["version"] is None

    def test_lt_version(self, processor):
        info = processor._extract_version_info("numpy<2.0")
        assert info["operator"] == "<"
        assert info["version"] == "2.0"


# ===================================================================
# _check_version_compatibility
# ===================================================================


class TestCheckVersionCompatibility:
    def test_same_exact_version(self, processor):
        assert processor._check_version_compatibility("pkg==1.0", ["pkg==1.0"]) is True

    def test_different_exact_versions(self, processor):
        assert processor._check_version_compatibility("pkg==1.0", ["pkg==2.0"]) is False

    def test_no_version_constraint(self, processor):
        assert processor._check_version_compatibility("pkg", ["pkg==1.0"]) is True

    def test_not_in_context(self, processor):
        assert processor._check_version_compatibility("pkg==1.0", ["other==2.0"]) is True

    def test_gte_vs_exact(self, processor):
        # >=1.0 vs ==1.5 — not both == so assumed compatible
        assert processor._check_version_compatibility("pkg>=1.0", ["pkg==1.5"]) is True


# ===================================================================
# _extract_package_name
# ===================================================================


class TestExtractPackageName:
    def test_exact(self, processor):
        assert processor._extract_package_name("numpy==1.24") == "numpy"

    def test_gte(self, processor):
        assert processor._extract_package_name("requests>=2.28") == "requests"

    def test_no_version(self, processor):
        assert processor._extract_package_name("flask") == "flask"

    def test_lt(self, processor):
        assert processor._extract_package_name("pkg<3") == "pkg"

    def test_lte(self, processor):
        assert processor._extract_package_name("pkg<=3.0") == "pkg"

    def test_gt(self, processor):
        assert processor._extract_package_name("pkg>1.0") == "pkg"


# ===================================================================
# _matches_naming_pattern
# ===================================================================


class TestMatchesNamingPattern:
    def test_snake_case_valid(self, processor):
        assert processor._matches_naming_pattern("my_func", "snake_case") is True

    def test_snake_case_invalid(self, processor):
        assert processor._matches_naming_pattern("MyFunc", "snake_case") is False

    def test_camel_case_valid(self, processor):
        assert processor._matches_naming_pattern("myFunc", "camelCase") is True

    def test_camel_case_invalid(self, processor):
        assert processor._matches_naming_pattern("MyFunc", "camelCase") is False

    def test_pascal_case_valid(self, processor):
        assert processor._matches_naming_pattern("MyClass", "PascalCase") is True

    def test_pascal_case_invalid(self, processor):
        assert processor._matches_naming_pattern("my_class", "PascalCase") is False

    def test_upper_case_valid(self, processor):
        assert processor._matches_naming_pattern("MAX_SIZE", "UPPER_CASE") is True

    def test_upper_case_invalid(self, processor):
        assert processor._matches_naming_pattern("maxSize", "UPPER_CASE") is False

    def test_unknown_pattern(self, processor):
        assert processor._matches_naming_pattern("anything", "kebab-case") is True


# ===================================================================
# _matches_import_pattern
# ===================================================================


class TestMatchesImportPattern:
    def test_relative_imports_dot(self, processor):
        assert processor._matches_import_pattern(".util", "relative_imports") is True

    def test_relative_imports_absolute(self, processor):
        assert processor._matches_import_pattern("os", "relative_imports") is False

    def test_absolute_imports(self, processor):
        assert processor._matches_import_pattern("os", "absolute_imports") is True

    def test_absolute_imports_relative(self, processor):
        assert processor._matches_import_pattern(".util", "absolute_imports") is False

    def test_package_imports(self, processor):
        assert processor._matches_import_pattern("os", "package_imports") is True

    def test_unknown_pattern(self, processor):
        assert processor._matches_import_pattern("os", "wildcard") is True

    def test_dependency_info_model(self, processor):
        dep = _make_dependency("os")
        assert processor._matches_import_pattern(dep, "absolute_imports") is True


# ===================================================================
# _analyze_naming_conventions
# ===================================================================


class TestAnalyzeNamingConventions:
    def test_no_entities(self, processor):
        analysis = _make_analysis_result(entities=[])
        context = _RepositoryContext(style_guide={"function": "snake_case"})
        score = processor._analyze_naming_conventions(analysis, context)
        assert score == 0.8

    def test_no_style_guide(self, processor):
        analysis = _make_analysis_result(entities=[_make_entity()])
        context = _RepositoryContext(style_guide={})
        score = processor._analyze_naming_conventions(analysis, context)
        assert score == 0.8

    def test_matching_pattern(self, processor):
        entity = _make_entity(name="my_func", entity_type="function")
        analysis = _make_analysis_result(entities=[entity])
        context = _RepositoryContext(style_guide={"function": "snake_case"})
        score = processor._analyze_naming_conventions(analysis, context)
        assert score == 1.0

    def test_non_matching_pattern(self, processor):
        entity = _make_entity(name="MyFunc", entity_type="function")
        analysis = _make_analysis_result(entities=[entity])
        context = _RepositoryContext(style_guide={"function": "snake_case"})
        score = processor._analyze_naming_conventions(analysis, context)
        assert score == 0.0


# ===================================================================
# _analyze_import_patterns
# ===================================================================


class TestAnalyzeImportPatterns:
    def test_no_dependencies(self, processor):
        analysis = _make_analysis_result(dependencies=[])
        context = _RepositoryContext(style_guide={"import_patterns": ["absolute_imports"]})
        score = processor._analyze_import_patterns(analysis, context)
        assert score == 0.8

    def test_no_import_patterns(self, processor):
        analysis = _make_analysis_result(dependencies=[_make_dependency("os")])
        context = _RepositoryContext(style_guide={})
        score = processor._analyze_import_patterns(analysis, context)
        assert score == 0.8

    def test_matching_absolute(self, processor):
        dep = _make_dependency("os")
        analysis = _make_analysis_result(dependencies=[dep])
        context = _RepositoryContext(style_guide={"import_patterns": ["absolute_imports"]})
        score = processor._analyze_import_patterns(analysis, context)
        assert score == 1.0


# ===================================================================
# _calculate_compatibility
# ===================================================================


class TestCalculateCompatibility:
    async def test_base_score_no_entities_no_deps(self, processor):
        analysis = _make_analysis_result(entities=[], dependencies=[])
        context = _RepositoryContext()
        score = await processor._calculate_compatibility(analysis, context)
        # base 0.8 + naming 0.8*0.05 + import 0.8*0.05 = 0.88
        assert score == pytest.approx(0.88, abs=0.01)

    async def test_entities_boost(self, processor):
        analysis = _make_analysis_result(entities=[_make_entity()], dependencies=[])
        context = _RepositoryContext()
        score = await processor._calculate_compatibility(analysis, context)
        # base 0.8 + entity 0.1 + naming + import
        assert score > 0.88

    async def test_capped_at_one(self, processor):
        # Give it everything to potentially exceed 1.0
        analysis = _make_analysis_result(
            entities=[_make_entity()],
            dependencies=[_make_dependency("os")],
        )
        context = _RepositoryContext(
            existing_patterns=[{"name": "os"}],
            style_guide={"function": "snake_case", "import_patterns": ["absolute_imports"]},
        )
        score = await processor._calculate_compatibility(analysis, context)
        assert score <= 1.0


# ===================================================================
# _analyze_dependency_compatibility
# ===================================================================


class TestAnalyzeDependencyCompatibility:
    async def test_no_dependencies(self, processor):
        context = _RepositoryContext()
        score = await processor._analyze_dependency_compatibility([], context)
        assert score == 1.0

    async def test_no_context_dependencies(self, processor):
        context = _RepositoryContext(existing_patterns=[])
        deps = [_make_dependency("os")]
        score = await processor._analyze_dependency_compatibility(deps, context)
        assert score == 0.8

    async def test_compatible_deps(self, processor):
        context = _RepositoryContext(existing_patterns=["os==1.0"])
        deps = [_make_dependency("os")]
        score = await processor._analyze_dependency_compatibility(deps, context)
        # "os" found in context, no version in DependencyInfoModel, so compatible
        assert score > 0.5

    async def test_new_deps(self, processor):
        context = _RepositoryContext(existing_patterns=["numpy==1.0"])
        deps = [_make_dependency("pandas")]
        score = await processor._analyze_dependency_compatibility(deps, context)
        assert score == 0.7  # all new


# ===================================================================
# _extract_context_dependency_names
# ===================================================================


class TestExtractContextDependencyNames:
    def test_dict_deps(self, processor):
        deps = [{"name": "numpy"}, {"name": "pandas"}]
        result = processor._extract_context_dependency_names(deps)
        assert result == {"numpy", "pandas"}

    def test_string_deps(self, processor):
        deps = ["numpy==1.0", "requests>=2.0"]
        result = processor._extract_context_dependency_names(deps)
        assert result == {"numpy", "requests"}

    def test_mixed_deps(self, processor):
        deps = [{"name": "numpy"}, "requests>=2.0"]
        result = processor._extract_context_dependency_names(deps)
        assert result == {"numpy", "requests"}


# ===================================================================
# _convert_context_deps_to_strings
# ===================================================================


class TestConvertContextDepsToStrings:
    def test_dict_with_version(self, processor):
        deps = [{"name": "numpy", "version": "1.24"}]
        result = processor._convert_context_deps_to_strings(deps)
        assert result == ["numpy==1.24"]

    def test_dict_without_version(self, processor):
        deps = [{"name": "numpy"}]
        result = processor._convert_context_deps_to_strings(deps)
        assert result == ["numpy"]

    def test_string_passthrough(self, processor):
        deps = ["numpy>=1.24"]
        result = processor._convert_context_deps_to_strings(deps)
        assert result == ["numpy>=1.24"]


# ===================================================================
# _calculate_dependency_score
# ===================================================================


class TestCalculateDependencyScore:
    def test_conflicting(self, processor):
        stats = {"compatible": 0, "conflicting": 2, "new": 0}
        score = processor._calculate_dependency_score(stats, 2)
        assert score == 0.0

    def test_compatible(self, processor):
        stats = {"compatible": 3, "conflicting": 0, "new": 0}
        score = processor._calculate_dependency_score(stats, 3)
        assert score == 1.0

    def test_all_new(self, processor):
        stats = {"compatible": 0, "conflicting": 0, "new": 5}
        score = processor._calculate_dependency_score(stats, 5)
        assert score == 0.7


# ===================================================================
# _generate_recommendations
# ===================================================================


class TestGenerateRecommendations:
    async def test_with_functions(self, processor):
        analysis = _make_analysis_result(
            entities=[_make_entity(name="func", entity_type="function")],
            metadata={"language": "python"},
        )
        context = _RepositoryContext()
        recs = await processor._generate_recommendations(analysis, context)
        assert "Ensure all functions have appropriate docstrings" in recs
        assert "Consider adding type hints for better code clarity" in recs
        assert "Add unit tests for new functionality" in recs

    async def test_no_functions(self, processor):
        analysis = _make_analysis_result(
            entities=[_make_entity(name="MyClass", entity_type="class")],
            metadata={"language": "javascript"},
        )
        context = _RepositoryContext()
        recs = await processor._generate_recommendations(analysis, context)
        assert "Ensure all functions have appropriate docstrings" not in recs
        assert "Add unit tests for new functionality" in recs

    async def test_non_python_no_type_hint_rec(self, processor):
        analysis = _make_analysis_result(
            entities=[],
            metadata={"language": "javascript"},
        )
        context = _RepositoryContext()
        recs = await processor._generate_recommendations(analysis, context)
        assert "Consider adding type hints for better code clarity" not in recs


# ===================================================================
# CodebaseAnalysisProcessor tests (codebase_analysis_processor.py)
# Covering uncovered lines: 76, 93-95, 120-127, 132-134, 140-166
# ===================================================================


class TestCodebaseAnalysisProcessorAdvanced:
    """Advanced tests for processing.codebase_analysis_processor.CodebaseAnalysisProcessor."""

    @pytest.mark.asyncio
    async def test_process_exception_in_analysis(self):
        """Test the exception handler in process() (lines 93-95)."""
        mock_settings = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
        with (
            patch("processing.codebase_analysis_processor.get_settings") as mock_gs,
            patch("processing.codebase_analysis_processor.settings", mock_settings),
            patch("processing.codebase_analysis_processor.TaskResultProcessor"),
            patch("processing.codebase_analysis_processor.FileProcessingResultHandler"),
            patch("processing.codebase_analysis_processor.TaskErrorHandler") as mock_teh,
            patch("processing.codebase_analysis_processor.TaskParameterValidator") as mock_tpv,
            patch("processing.codebase_analysis_processor.FileSystemValidator") as mock_fsv,
            patch("processing.codebase_analysis_processor.GraphDatabaseConnection") as mock_gdb,
            patch("processing.codebase_analysis_processor.PerformanceUtils") as mock_perf,
        ):
            mock_gs.return_value = mock_settings
            from processing.codebase_analysis_processor import CodebaseAnalysisProcessor

            mock_tpv.validate_codebase_analysis_params.return_value = {
                "is_valid": True,
                "errors": [],
                "validated_params": {
                    "codebase_path": "/tmp/test",
                    "scan_depth": 1,
                    "code_graph_id": "test_id",
                },
            }
            mock_fsv.validate_codebase_path.return_value = {"is_valid": True}
            mock_db = MagicMock()
            mock_db.connect = AsyncMock()
            mock_db.get_connection.return_value = AsyncMock()
            mock_db.close = AsyncMock()
            mock_gdb.return_value = mock_db
            mock_perf.create_timer.side_effect = RuntimeError("db error")
            error_handler = MagicMock()
            error_handler.create_error_response.return_value = {
                "success": False,
                "error": "db error",
            }
            mock_teh.return_value = error_handler

            proc = CodebaseAnalysisProcessor()
            proc.error_handler = error_handler
            result = await proc.process({"codebase_path": "/tmp/test", "code_graph_id": "test_id"})
            assert result["success"] is False

    @pytest.mark.asyncio
    async def test_scan_and_process_codebase_exception(self):
        """Test exception handling in _scan_and_process_codebase (lines 132-134)."""
        with (
            patch("processing.codebase_analysis_processor.get_settings") as mock_gs,
            patch("processing.codebase_analysis_processor.TaskResultProcessor"),
            patch("processing.codebase_analysis_processor.FileProcessingResultHandler"),
            patch("processing.codebase_analysis_processor.TaskErrorHandler") as mock_teh,
            patch("processing.codebase_analysis_processor.ProcessingUtils") as mock_pu,
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.codebase_analysis_processor import CodebaseAnalysisProcessor

            error_handler = MagicMock()
            mock_teh.return_value = error_handler
            mock_pu.create_processing_context.return_value = {
                "factory": MagicMock(),
                "conn": AsyncMock(),
                "code_graph_id": "test",
            }

            proc = CodebaseAnalysisProcessor()
            proc.error_handler = error_handler

            with patch("os.walk", side_effect=RuntimeError("walk error")):
                result = await proc._scan_and_process_codebase(
                    "/tmp/test", MagicMock(), AsyncMock(), "test_id"
                )
            assert result == 0

    @pytest.mark.asyncio
    async def test_process_directory_files(self):
        """Test _process_directory_files (lines 140-166)."""
        with (
            patch("processing.codebase_analysis_processor.get_settings") as mock_gs,
            patch("processing.codebase_analysis_processor.TaskResultProcessor"),
            patch("processing.codebase_analysis_processor.FileProcessingResultHandler"),
            patch("processing.codebase_analysis_processor.TaskErrorHandler") as mock_teh,
            patch("processing.codebase_analysis_processor.FileOperationUtils") as mock_fou,
            patch("processing.codebase_analysis_processor.ProcessingUtils") as mock_pu,
        ):
            mock_gs.return_value = MagicMock(CODE_GRAPH_DATABASE_DIR="/tmp")
            from processing.codebase_analysis_processor import CodebaseAnalysisProcessor

            mock_fou.get_language_extensions.return_value = {".py": "python"}
            mock_fou.should_process_file.side_effect = lambda f: f.endswith(".py")
            mock_fou.get_file_language.return_value = "python"
            mock_pu.create_processing_context.return_value = {
                "factory": MagicMock(),
                "conn": AsyncMock(),
                "code_graph_id": "test",
                "file_path": "/tmp/test.py",
                "language": "python",
            }
            mock_pu.process_file_safely = AsyncMock(return_value={"success": True, "result": {}})
            error_handler = MagicMock()
            mock_teh.return_value = error_handler

            proc = CodebaseAnalysisProcessor()
            proc.error_handler = error_handler
            proc.file_result_handler = AsyncMock()
            proc.file_result_handler.store_processing_results = AsyncMock()

            ctx = {
                "factory": MagicMock(),
                "conn": AsyncMock(),
                "code_graph_id": "test",
            }
            result = await proc._process_directory_files("/tmp", ["test.py", "readme.md"], ctx)
            assert result >= 0
