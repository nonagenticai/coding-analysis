"""Unit tests for processing.analyzers.complexity_analyzers package.

Focuses on pure-computation tests that need no mocking:
- ComplexityLevel enum values
- ServiceComplexityMetrics dataclass defaults
- ComplexityAnalysisService.get_supported_languages()
- Language-specific _classify_complexity() thresholds
- Language-specific _calculate_maintainability_index() / maintainability
"""

from __future__ import annotations

from processing.analyzers.complexity_analyzers import (
    ComplexityAnalysisService,
    ComplexityLevel,
    CppComplexityAnalyzer,
    GoComplexityAnalyzer,
    JavaComplexityAnalyzer,
    JavaScriptComplexityAnalyzer,
    PythonComplexityAnalyzer,
    RustComplexityAnalyzer,
    ServiceComplexityMetrics,
    TypeScriptComplexityAnalyzer,
    get_complexity_service,
)

# ---------------------------------------------------------------------------
# ComplexityLevel enum
# ---------------------------------------------------------------------------


class TestComplexityLevel:
    """Tests for the ComplexityLevel enum."""

    def test_trivial_value(self):
        assert ComplexityLevel.TRIVIAL == "trivial"
        assert ComplexityLevel.TRIVIAL.value == "trivial"

    def test_basic_value(self):
        assert ComplexityLevel.BASIC == "basic"

    def test_moderate_value(self):
        assert ComplexityLevel.MODERATE == "moderate"

    def test_complex_value(self):
        assert ComplexityLevel.COMPLEX == "complex"

    def test_very_complex_value(self):
        assert ComplexityLevel.VERY_COMPLEX == "very_complex"

    def test_all_members(self):
        members = [level.value for level in ComplexityLevel]
        assert members == ["trivial", "basic", "moderate", "complex", "very_complex"]

    def test_is_str_enum(self):
        assert isinstance(ComplexityLevel.TRIVIAL, str)


# ---------------------------------------------------------------------------
# ServiceComplexityMetrics dataclass
# ---------------------------------------------------------------------------


class TestServiceComplexityMetrics:
    """Tests for ServiceComplexityMetrics dataclass defaults."""

    def test_default_values(self):
        m = ServiceComplexityMetrics()
        assert m.cyclomatic_complexity == 0.0
        assert m.cognitive_complexity == 0.0
        assert m.lines_of_code == 0
        assert m.nesting_depth == 0
        assert m.parameter_count == 0
        assert m.dependency_count == 0
        assert m.halstead_difficulty == 0.0
        assert m.maintainability_index == 0.0
        assert m.coupling_factor == 0.0
        assert m.complexity_level == ComplexityLevel.TRIVIAL
        assert m.risk_factors == []
        assert m.suggestions == []
        assert m.language == ""
        assert m.entity_type == ""
        assert m.entity_name == ""

    def test_custom_values(self):
        m = ServiceComplexityMetrics(
            cyclomatic_complexity=15.0,
            cognitive_complexity=10.0,
            lines_of_code=200,
            language="python",
            entity_type="function",
            entity_name="process_data",
            complexity_level=ComplexityLevel.MODERATE,
        )
        assert m.cyclomatic_complexity == 15.0
        assert m.language == "python"
        assert m.complexity_level == ComplexityLevel.MODERATE

    def test_risk_factors_default_is_independent(self):
        """Each instance should have its own list, not a shared reference."""
        m1 = ServiceComplexityMetrics()
        m2 = ServiceComplexityMetrics()
        m1.risk_factors.append("test")
        assert len(m2.risk_factors) == 0

    def test_suggestions_default_is_independent(self):
        m1 = ServiceComplexityMetrics()
        m2 = ServiceComplexityMetrics()
        m1.suggestions.append("refactor")
        assert len(m2.suggestions) == 0


# ---------------------------------------------------------------------------
# ComplexityAnalysisService.get_supported_languages
# ---------------------------------------------------------------------------


class TestComplexityAnalysisService:
    """Tests for ComplexityAnalysisService."""

    def test_get_supported_languages(self):
        svc = ComplexityAnalysisService()
        languages = svc.get_supported_languages()
        assert isinstance(languages, list)
        assert "python" in languages
        assert "java" in languages
        assert "go" in languages
        assert "rust" in languages
        assert "javascript" in languages
        assert "typescript" in languages
        assert "cpp" in languages
        assert "csharp" in languages

    def test_supported_languages_count(self):
        svc = ComplexityAnalysisService()
        languages = svc.get_supported_languages()
        assert len(languages) == 8


# ---------------------------------------------------------------------------
# Java _classify_complexity
# ---------------------------------------------------------------------------


class TestJavaClassifyComplexity:
    """Tests for JavaComplexityAnalyzer._classify_complexity thresholds."""

    def setup_method(self):
        self.analyzer = JavaComplexityAnalyzer()

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
        assert self.analyzer._classify_complexity(100) == ComplexityLevel.VERY_COMPLEX


# ---------------------------------------------------------------------------
# Rust _classify_complexity (different thresholds)
# ---------------------------------------------------------------------------


class TestRustClassifyComplexity:
    """Tests for RustComplexityAnalyzer._classify_complexity thresholds."""

    def setup_method(self):
        self.analyzer = RustComplexityAnalyzer()

    def test_trivial(self):
        assert self.analyzer._classify_complexity(0) == ComplexityLevel.TRIVIAL
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


# ---------------------------------------------------------------------------
# Go _classify_complexity
# ---------------------------------------------------------------------------


class TestGoClassifyComplexity:
    """Tests for GoComplexityAnalyzer._classify_complexity thresholds."""

    def setup_method(self):
        self.analyzer = GoComplexityAnalyzer()

    def test_trivial(self):
        assert self.analyzer._classify_complexity(5) == ComplexityLevel.TRIVIAL

    def test_basic(self):
        assert self.analyzer._classify_complexity(6) == ComplexityLevel.BASIC
        assert self.analyzer._classify_complexity(10) == ComplexityLevel.BASIC

    def test_moderate(self):
        assert self.analyzer._classify_complexity(11) == ComplexityLevel.MODERATE
        assert self.analyzer._classify_complexity(15) == ComplexityLevel.MODERATE

    def test_complex(self):
        assert self.analyzer._classify_complexity(16) == ComplexityLevel.COMPLEX
        assert self.analyzer._classify_complexity(25) == ComplexityLevel.COMPLEX

    def test_very_complex(self):
        assert self.analyzer._classify_complexity(26) == ComplexityLevel.VERY_COMPLEX


# ---------------------------------------------------------------------------
# Python _classify_complexity
# ---------------------------------------------------------------------------


class TestPythonClassifyComplexity:
    """Tests for PythonComplexityAnalyzer._classify_complexity thresholds."""

    def setup_method(self):
        self.analyzer = PythonComplexityAnalyzer()

    def test_trivial(self):
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


# ---------------------------------------------------------------------------
# JavaScript _classify_complexity
# ---------------------------------------------------------------------------


class TestJavaScriptClassifyComplexity:
    """Tests for JavaScriptComplexityAnalyzer._classify_complexity thresholds."""

    def setup_method(self):
        self.analyzer = JavaScriptComplexityAnalyzer()

    def test_trivial(self):
        assert self.analyzer._classify_complexity(5) == ComplexityLevel.TRIVIAL

    def test_basic(self):
        assert self.analyzer._classify_complexity(6) == ComplexityLevel.BASIC
        assert self.analyzer._classify_complexity(10) == ComplexityLevel.BASIC

    def test_moderate(self):
        assert self.analyzer._classify_complexity(11) == ComplexityLevel.MODERATE
        assert self.analyzer._classify_complexity(18) == ComplexityLevel.MODERATE

    def test_complex(self):
        assert self.analyzer._classify_complexity(19) == ComplexityLevel.COMPLEX
        assert self.analyzer._classify_complexity(35) == ComplexityLevel.COMPLEX

    def test_very_complex(self):
        assert self.analyzer._classify_complexity(36) == ComplexityLevel.VERY_COMPLEX


# ---------------------------------------------------------------------------
# TypeScript _classify_complexity
# ---------------------------------------------------------------------------


class TestTypeScriptClassifyComplexity:
    """Tests for TypeScriptComplexityAnalyzer._classify_complexity thresholds."""

    def setup_method(self):
        self.analyzer = TypeScriptComplexityAnalyzer()

    def test_trivial(self):
        assert self.analyzer._classify_complexity(6) == ComplexityLevel.TRIVIAL

    def test_basic(self):
        assert self.analyzer._classify_complexity(7) == ComplexityLevel.BASIC
        assert self.analyzer._classify_complexity(12) == ComplexityLevel.BASIC

    def test_moderate(self):
        assert self.analyzer._classify_complexity(13) == ComplexityLevel.MODERATE
        assert self.analyzer._classify_complexity(25) == ComplexityLevel.MODERATE

    def test_complex(self):
        assert self.analyzer._classify_complexity(26) == ComplexityLevel.COMPLEX
        assert self.analyzer._classify_complexity(50) == ComplexityLevel.COMPLEX

    def test_very_complex(self):
        assert self.analyzer._classify_complexity(51) == ComplexityLevel.VERY_COMPLEX


# ---------------------------------------------------------------------------
# Java _calculate_maintainability_index
# ---------------------------------------------------------------------------


class TestJavaMaintainabilityIndex:
    """Tests for JavaComplexityAnalyzer._calculate_maintainability_index."""

    def setup_method(self):
        self.analyzer = JavaComplexityAnalyzer()

    def test_zero_loc_returns_100(self):
        m = ServiceComplexityMetrics(lines_of_code=0, cyclomatic_complexity=5.0)
        result = self.analyzer._calculate_maintainability_index(m)
        assert result == 100.0

    def test_small_function(self):
        m = ServiceComplexityMetrics(lines_of_code=10, cyclomatic_complexity=2.0)
        result = self.analyzer._calculate_maintainability_index(m)
        # 171 - 5.2*log(10) - 0.23*2 - 16.2*log(10)
        # = 171 - 5.2*2.302 - 0.46 - 16.2*2.302
        # = 171 - 11.97 - 0.46 - 37.29 = ~121.28 -> clamped to 100
        assert result == 100.0  # clamped

    def test_large_complex_function(self):
        m = ServiceComplexityMetrics(lines_of_code=500, cyclomatic_complexity=50.0)
        result = self.analyzer._calculate_maintainability_index(m)
        # Should be low but >= 0
        assert 0.0 <= result <= 100.0

    def test_result_clamped_to_0_100(self):
        # Very large file with very high complexity
        m = ServiceComplexityMetrics(lines_of_code=10000, cyclomatic_complexity=200.0)
        result = self.analyzer._calculate_maintainability_index(m)
        assert result >= 0.0
        assert result <= 100.0

    def test_moderate_function(self):
        m = ServiceComplexityMetrics(lines_of_code=50, cyclomatic_complexity=10.0)
        result = self.analyzer._calculate_maintainability_index(m)
        assert 0.0 <= result <= 100.0


# ---------------------------------------------------------------------------
# Rust _calculate_rust_maintainability
# ---------------------------------------------------------------------------


class TestRustMaintainability:
    """Tests for RustComplexityAnalyzer._calculate_rust_maintainability."""

    def setup_method(self):
        self.analyzer = RustComplexityAnalyzer()

    def _make_entity(self, properties=None):
        from unittest.mock import MagicMock

        entity = MagicMock()
        entity.properties = properties
        return entity

    def test_simple_function(self):
        m = ServiceComplexityMetrics(cyclomatic_complexity=2.0)
        entity = self._make_entity()
        result = self.analyzer._calculate_rust_maintainability(m, entity)
        # 100 - 2*2 = 96
        assert result == 96.0

    def test_unsafe_penalty(self):
        m = ServiceComplexityMetrics(cyclomatic_complexity=2.0)
        entity = self._make_entity(properties={"has_unsafe": True})
        result = self.analyzer._calculate_rust_maintainability(m, entity)
        # 100 - 20 - 4 = 76
        assert result == 76.0

    def test_macro_penalty(self):
        m = ServiceComplexityMetrics(cyclomatic_complexity=2.0)
        entity = self._make_entity(properties={"macro_usage": 3})
        result = self.analyzer._calculate_rust_maintainability(m, entity)
        # 100 - 10 - 4 = 86
        assert result == 86.0

    def test_clamped_at_zero(self):
        m = ServiceComplexityMetrics(cyclomatic_complexity=100.0)
        entity = self._make_entity(properties={"has_unsafe": True, "macro_usage": 5})
        result = self.analyzer._calculate_rust_maintainability(m, entity)
        assert result == 0.0


# ---------------------------------------------------------------------------
# get_complexity_service singleton
# ---------------------------------------------------------------------------


class TestGetComplexityService:
    """Tests for the get_complexity_service singleton function."""

    def setup_method(self):
        from processing.analyzers.complexity_analyzers import service as mod

        mod._complexity_service = None

    def teardown_method(self):
        from processing.analyzers.complexity_analyzers import service as mod

        mod._complexity_service = None

    def test_returns_instance(self):
        svc = get_complexity_service()
        assert isinstance(svc, ComplexityAnalysisService)

    def test_returns_same_instance(self):
        svc1 = get_complexity_service()
        svc2 = get_complexity_service()
        assert svc1 is svc2


# ---------------------------------------------------------------------------
# Language-specific factors
# ---------------------------------------------------------------------------


class TestLanguageSpecificFactors:
    """Tests that each analyzer returns expected factor keys."""

    def test_java_factors(self):
        factors = JavaComplexityAnalyzer().get_language_specific_factors()
        assert "inheritance_penalty" in factors
        assert "exception_handling" in factors

    def test_rust_factors(self):
        factors = RustComplexityAnalyzer().get_language_specific_factors()
        assert "unsafe_penalty" in factors
        assert "ownership_complexity" in factors

    def test_go_factors(self):
        factors = GoComplexityAnalyzer().get_language_specific_factors()
        assert "goroutine_penalty" in factors
        assert "error_handling" in factors

    def test_python_factors(self):
        factors = PythonComplexityAnalyzer().get_language_specific_factors()
        assert "decorator_penalty" in factors
        assert "async_penalty" in factors

    def test_javascript_factors(self):
        factors = JavaScriptComplexityAnalyzer().get_language_specific_factors()
        assert "callback_nesting_penalty" in factors
        assert "closure_penalty" in factors

    def test_typescript_factors(self):
        factors = TypeScriptComplexityAnalyzer().get_language_specific_factors()
        assert "type_system_penalty" in factors
        assert "generic_penalty" in factors

    def test_cpp_factors(self):
        factors = CppComplexityAnalyzer().get_language_specific_factors()
        assert "template_penalty" in factors
        assert "raw_pointer_penalty" in factors
