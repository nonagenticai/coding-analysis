"""Unit tests for scoring module (compatibility layer).

Covers:
- scoring/__init__.py (placeholder classes)
- scoring/recommendation_generator.py (placeholder PRValidationRecommendationGenerator)
- scoring/risk_calculator.py (placeholder RiskCalculator)

The services.pr_validation.scoring package does not exist in this codebase,
so all three modules fall through to their placeholder implementations.
These tests verify the placeholder API contracts that the rest of the codebase
depends on.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Tests for scoring/__init__.py
# ---------------------------------------------------------------------------


class TestScoringInit:
    """Tests for the scoring package __init__.py compatibility layer."""

    def test_risk_calculator_importable_and_is_class(self):
        """RiskCalculator should be importable and be a class."""
        from scoring import RiskCalculator

        assert isinstance(RiskCalculator, type)

    def test_recommendation_generator_importable_and_is_class(self):
        """PRValidationRecommendationGenerator should be importable and be a class."""
        from scoring import PRValidationRecommendationGenerator

        assert isinstance(PRValidationRecommendationGenerator, type)

    def test_risk_calculator_calculate_risk_returns_expected_structure(self):
        """RiskCalculator.calculate_risk should return dict with overall_score=0.0 and a message."""
        from scoring import RiskCalculator

        calc = RiskCalculator()
        result = calc.calculate_risk()
        assert result == {"overall_score": 0.0, "message": "Scoring unavailable"}

    def test_recommendation_generator_generate_returns_empty_list(self):
        """PRValidationRecommendationGenerator.generate_recommendations should return []."""
        from scoring import PRValidationRecommendationGenerator

        gen = PRValidationRecommendationGenerator()
        result = gen.generate_recommendations()
        assert result == []

    def test_all_exports_match_expected_set(self):
        """__all__ should contain exactly RiskCalculator and PRValidationRecommendationGenerator."""
        import scoring

        assert set(scoring.__all__) == {"RiskCalculator", "PRValidationRecommendationGenerator"}

    def test_risk_calculator_accepts_arbitrary_args(self):
        """Placeholder RiskCalculator should accept arbitrary args and still return the stub."""
        from scoring import RiskCalculator

        calc = RiskCalculator("arg1", key="value")
        result = calc.calculate_risk("data", extra=True)
        assert result["overall_score"] == 0.0
        assert "message" in result

    def test_recommendation_generator_accepts_arbitrary_args(self):
        """Placeholder PRValidationRecommendationGenerator should accept arbitrary args."""
        from scoring import PRValidationRecommendationGenerator

        gen = PRValidationRecommendationGenerator("arg1", key="value")
        result = gen.generate_recommendations("data", key="val")
        assert result == []


# ---------------------------------------------------------------------------
# Tests for scoring/recommendation_generator.py
# ---------------------------------------------------------------------------


class TestRecommendationGeneratorModule:
    """Tests for scoring/recommendation_generator.py."""

    def test_is_importable_class(self):
        """PRValidationRecommendationGenerator should be importable from the submodule."""
        from scoring.recommendation_generator import PRValidationRecommendationGenerator

        assert isinstance(PRValidationRecommendationGenerator, type)

    def test_generate_recommendations_returns_empty_list(self):
        """generate_recommendations should return an empty list (placeholder behavior)."""
        from scoring.recommendation_generator import PRValidationRecommendationGenerator

        gen = PRValidationRecommendationGenerator()
        result = gen.generate_recommendations()
        assert result == []

    def test_generate_with_args_returns_empty_list(self):
        """generate_recommendations should accept and ignore arbitrary args."""
        from scoring.recommendation_generator import PRValidationRecommendationGenerator

        gen = PRValidationRecommendationGenerator()
        result = gen.generate_recommendations("data", key="val")
        assert result == []

    def test_all_exports(self):
        """__all__ should contain PRValidationRecommendationGenerator."""
        from scoring import recommendation_generator

        assert recommendation_generator.__all__ == ["PRValidationRecommendationGenerator"]


# ---------------------------------------------------------------------------
# Tests for scoring/risk_calculator.py
# ---------------------------------------------------------------------------


class TestRiskCalculatorModule:
    """Tests for scoring/risk_calculator.py."""

    def test_is_importable_class(self):
        """RiskCalculator should be importable from the submodule."""
        from scoring.risk_calculator import RiskCalculator

        assert isinstance(RiskCalculator, type)

    def test_calculate_risk_returns_expected_structure(self):
        """calculate_risk should return dict with overall_score=0.0 and message."""
        from scoring.risk_calculator import RiskCalculator

        calc = RiskCalculator()
        result = calc.calculate_risk()
        assert result == {"overall_score": 0.0, "message": "Risk calculation unavailable"}

    def test_calculate_overall_risk_returns_expected_structure(self):
        """calculate_overall_risk should return the same structure as calculate_risk."""
        from scoring.risk_calculator import RiskCalculator

        calc = RiskCalculator()
        result = calc.calculate_overall_risk()
        assert result == {"overall_score": 0.0, "message": "Risk calculation unavailable"}

    def test_calculate_risk_and_overall_risk_return_same_result(self):
        """Both calculate methods should return identical structures."""
        from scoring.risk_calculator import RiskCalculator

        calc = RiskCalculator()
        risk = calc.calculate_risk()
        overall = calc.calculate_overall_risk()
        assert risk == overall

    def test_calculate_risk_with_args_preserves_structure(self):
        """calculate_risk should accept arbitrary args and still return the stub dict."""
        from scoring.risk_calculator import RiskCalculator

        calc = RiskCalculator()
        result = calc.calculate_risk("data", key="val")
        assert result["overall_score"] == 0.0
        assert result["message"] == "Risk calculation unavailable"

    def test_constructor_accepts_arbitrary_args(self):
        """Constructor should accept arbitrary args for compatibility."""
        from scoring.risk_calculator import RiskCalculator

        calc = RiskCalculator("config", setting=True)
        result = calc.calculate_risk()
        assert result["overall_score"] == 0.0

    def test_all_exports(self):
        """__all__ should contain RiskCalculator."""
        from scoring import risk_calculator

        assert risk_calculator.__all__ == ["RiskCalculator"]
