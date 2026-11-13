"""
Fix rules and generators for different tools and languages.
"""

from __future__ import annotations

import re
from typing import Any

from models.testing_models import QualityIssue
from processing.quality_analysis.fix_models import (
    FixCategory,
    FixRule,
    FixStrategy,
)


class FixRuleManager:
    """Manages fix rules for different tools and issue types."""

    def __init__(self):
        self.rules = self._initialize_fix_rules()

    def get_rule_count(self) -> int:
        """Get the number of fix rules."""
        return len(self.rules)

    def _initialize_fix_rules(self) -> list[FixRule]:
        """Initialize resolve rules for different tools and issues."""
        rules = []

        # Python rules
        rules.extend(
            [
                # ruff format (formatter check; rule_id ``unformatted``)
                FixRule(
                    rule_id="ruff_formatting",
                    tool="ruff-format",
                    rule_pattern=".*",
                    category=FixCategory.FORMATTING,
                    strategy=FixStrategy.AUTOMATIC,
                    confidence=0.95,
                    requires_review=False,
                ),
                # ruff isort rules (I001 unsorted, I002 missing required import)
                FixRule(
                    rule_id="ruff_imports",
                    tool="ruff",
                    rule_pattern=r"^I\d{3}$",
                    category=FixCategory.IMPORT_ORGANIZATION,
                    strategy=FixStrategy.AUTOMATIC,
                    confidence=0.9,
                    requires_review=False,
                ),
                # ruff unused import / variable (what autoflake removed)
                FixRule(
                    rule_id="ruff_unused",
                    tool="ruff",
                    rule_pattern=r"^(F401|F841)$",
                    category=FixCategory.IMPORT_ORGANIZATION,
                    strategy=FixStrategy.AUTOMATIC,
                    confidence=0.85,
                    requires_review=False,
                ),
                # ty: returned value does not match the declared return type
                FixRule(
                    rule_id="ty_return_type",
                    tool="ty",
                    rule_pattern=r"^invalid-return-type$",
                    category=FixCategory.TYPE_ANNOTATION,
                    strategy=FixStrategy.SUGGESTED,
                    confidence=0.7,
                    requires_review=True,
                ),
                # ruff pydocstyle rules (pylint C0111 missing-docstring)
                FixRule(
                    rule_id="ruff_docstring",
                    tool="ruff",
                    rule_pattern=r"^D10[0-7]$",
                    category=FixCategory.CODE_STYLE,
                    strategy=FixStrategy.SUGGESTED,
                    confidence=0.8,
                    requires_review=True,
                ),
                # Security resolves
                FixRule(
                    rule_id="bandit_security",
                    tool="bandit",
                    rule_pattern=".*",
                    category=FixCategory.SECURITY,
                    strategy=FixStrategy.MANUAL,
                    confidence=0.6,
                    requires_review=True,
                ),
            ]
        )

        # JavaScript/TypeScript rules
        rules.extend(
            [
                # ESLint resolves
                FixRule(
                    rule_id="eslint_auto_fixable",
                    tool="eslint",
                    rule_pattern=".*",
                    category=FixCategory.CODE_STYLE,
                    strategy=FixStrategy.SUGGESTED,
                    confidence=0.8,
                    requires_review=False,
                ),
                # Prettier formatting
                FixRule(
                    rule_id="prettier_formatting",
                    tool="prettier",
                    rule_pattern=".*",
                    category=FixCategory.FORMATTING,
                    strategy=FixStrategy.AUTOMATIC,
                    confidence=0.95,
                    requires_review=False,
                ),
            ]
        )

        return rules

    def find_matching_rules(self, issue: QualityIssue) -> list[FixRule]:
        """Find resolve rules that match the given issue."""
        matching_rules = []

        for rule in self.rules:
            # Check tool match
            if rule.tool not in (issue.tool, "*"):
                continue

            # Check rule pattern match
            if rule.rule_pattern and issue.rule_id:
                if not re.search(rule.rule_pattern, issue.rule_id):
                    continue

            # Check message pattern match
            if rule.message_pattern:
                if not re.search(rule.message_pattern, issue.message, re.IGNORECASE):
                    continue

            matching_rules.append(rule)

        return matching_rules


class FixDetailGenerator:
    """Generates fix details for different tools and issues."""

    def get_supported_tools(self) -> list[str]:
        """Get list of supported tools."""
        return ["ruff", "ruff-format", "ty", "bandit", "eslint", "prettier"]

    async def generate_fix_details(
        self, issue: QualityIssue, rule: FixRule
    ) -> dict[str, Any] | None:
        """Generate specific resolve details based on issue and rule."""
        # Use a dispatch table for cleaner code
        tool_generators = {
            "ruff": self._generate_ruff_fix,
            "ruff-format": self._generate_ruff_format_fix,
            "ty": self._generate_ty_fix,
            "bandit": self._generate_bandit_fix,
            "eslint": self._generate_eslint_fix,
            "prettier": self._generate_prettier_fix,
        }

        generator = tool_generators.get(issue.tool)
        if generator:
            return await generator(issue)

        # Generic resolve
        return await self._generate_generic_fix(issue, rule)

    # Tool-specific resolve generators

    async def _generate_ruff_format_fix(self, _issue: QualityIssue) -> dict[str, Any]:
        """Generate ruff format resolves."""
        return {
            "title": "Apply ruff formatting",
            "description": "Automatically format code using ruff format",
            "original_code": None,  # Would need file content
            "fixed_code": None,  # Would run ruff format
            "dif": None,
            "breaking_change": False,
            "side_effects": ["Code formatting will change"],
            "dependencies": ["ruff"],
        }

    async def _generate_ruff_fix(self, issue: QualityIssue) -> dict[str, Any]:
        """Generate resolves for ruff lint codes (flake8 E/W/F, isort I, pylint PL*)."""
        code = issue.rule_id or ""
        if code.startswith("I"):
            return {
                "title": "Sort and organize imports",
                "description": "Automatically sort and organize imports (ruff check --select I --fix)",
                "original_code": None,
                "fixed_code": None,
                "dif": None,
                "breaking_change": False,
                "side_effects": ["Import order will change"],
                "dependencies": ["ruff"],
            }
        if code in ("F401", "F841"):
            return {
                "title": "Remove unused import or variable",
                "description": (
                    "Remove the unused binding (ruff check --select F401,F841 --fix): "
                    f"{issue.message}"
                ),
                "original_code": None,
                "fixed_code": None,
                "dif": None,
                "breaking_change": False,
                "side_effects": ["Unused imports/variables will be removed"],
                "dependencies": ["ruff"],
            }
        if code.startswith("N8"):  # pep8-naming (pylint C0103 invalid-name)
            return {
                "title": "Resolve naming convention",
                "description": "Rename to follow Python naming conventions",
                "original_code": None,
                "fixed_code": None,
                "dif": None,
                "breaking_change": True,  # Renaming can break references
                "side_effects": ["Name will change"],
                "dependencies": [],
            }
        if code.startswith("D10"):  # pydocstyle (pylint C0111 missing-docstring)
            return {
                "title": "Add missing docstring",
                "description": "Add docstring to function, class, or module",
                "original_code": None,
                "fixed_code": None,
                "dif": None,
                "breaking_change": False,
                "side_effects": ["Docstring will be added"],
                "dependencies": [],
            }
        if code.startswith("ARG"):  # unused argument (pylint W0613)
            return {
                "title": "Handle unused argument",
                "description": "Add underscore prefix to unused argument or remove if possible",
                "original_code": None,
                "fixed_code": None,
                "dif": None,
                "breaking_change": True,  # Changing function signature
                "side_effects": ["Function signature may change"],
                "dependencies": [],
            }
        return {
            "title": f"Resolve ruff issue ({code})",
            "description": f"Address ruff issue: {issue.message}",
            "original_code": None,
            "fixed_code": None,
            "dif": None,
            "breaking_change": False,
            "side_effects": [],
            "dependencies": [],
        }

    async def _generate_ty_fix(self, issue: QualityIssue) -> dict[str, Any]:
        """Generate resolves for ty rules (keyed on the ty rule name)."""
        rule = issue.rule_id or ""
        if rule == "invalid-return-type":
            return {
                "title": "Fix return type mismatch",
                "description": "Make the returned value match the declared return type, or fix the annotation",
                "original_code": None,
                "fixed_code": None,
                "dif": None,
                "breaking_change": False,
                "side_effects": ["Function signature or return value may change"],
                "dependencies": [],
            }
        if rule == "invalid-argument-type":
            return {
                "title": "Fix argument type mismatch",
                "description": f"Pass a value of the expected type: {issue.message}",
                "original_code": None,
                "fixed_code": None,
                "dif": None,
                "breaking_change": False,
                "side_effects": ["Call site will change"],
                "dependencies": [],
            }
        if rule == "unresolved-import":
            return {
                "title": "Resolve missing import",
                "description": f"Install the dependency or fix the module path: {issue.message}",
                "original_code": None,
                "fixed_code": None,
                "dif": None,
                "breaking_change": False,
                "side_effects": ["Dependencies or imports may change"],
                "dependencies": [],
            }
        return {
            "title": f"Resolve type checking issue ({rule})",
            "description": f"Address ty type checking issue: {issue.message}",
            "original_code": None,
            "fixed_code": None,
            "dif": None,
            "breaking_change": False,
            "side_effects": ["Type annotations may be added or modified"],
            "dependencies": [],
        }

    async def _generate_bandit_fix(self, issue: QualityIssue) -> dict[str, Any]:
        """Generate Bandit security resolves."""
        return {
            "title": "Resolve security issue",
            "description": f"Address security vulnerability: {issue.message}",
            "original_code": None,
            "fixed_code": None,
            "dif": None,
            "breaking_change": True,  # Security resolves often change behavior
            "side_effects": ["Security behavior will change"],
            "dependencies": [],
        }

    async def _generate_eslint_fix(self, issue: QualityIssue) -> dict[str, Any]:
        """Generate ESLint resolves."""
        return {
            "title": "Resolve ESLint issue",
            "description": f"Apply ESLint auto-resolve: {issue.message}",
            "original_code": None,
            "fixed_code": None,
            "dif": None,
            "breaking_change": False,
            "side_effects": ["Code style will change"],
            "dependencies": ["eslint"],
        }

    async def _generate_prettier_fix(self, _issue: QualityIssue) -> dict[str, Any]:
        """Generate Prettier formatting resolves."""
        return {
            "title": "Apply Prettier formatting",
            "description": "Automatically format code using Prettier",
            "original_code": None,
            "fixed_code": None,
            "dif": None,
            "breaking_change": False,
            "side_effects": ["Code formatting will change"],
            "dependencies": ["prettier"],
        }

    async def _generate_generic_fix(self, issue: QualityIssue, rule: FixRule) -> dict[str, Any]:
        """Generate generic resolve based on rule template."""
        return {
            "title": f"Resolve {issue.tool} issue",
            "description": f"Address {issue.tool} issue: {issue.message}",
            "original_code": None,
            "fixed_code": None,
            "dif": None,
            "breaking_change": rule.requires_review,
            "side_effects": [],
            "dependencies": [],
        }
