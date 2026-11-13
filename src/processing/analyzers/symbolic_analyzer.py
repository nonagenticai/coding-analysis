"""
Symbolic Analysis and Advanced Validation Analyzer

This module provides symbolic analysis capabilities for code validation and security assessment.
It has been refactored into a package structure for better maintainability while preserving
all functionality through backward-compatible re-exports.

The module exposes:
- SymbolicValidator: Main validator class for performing symbolic analysis
- ValidationConfig: Configuration class for analysis parameters
- SecurityLevel: Enumeration of security risk levels
- ValidationType: Enumeration of validation types

Example:
    from symbolic_analyzer import SymbolicValidator, ValidationConfig, SecurityLevel

    config = ValidationConfig(security_level=SecurityLevel.HIGH)
    validator = SymbolicValidator(config)
    results = validator.analyze(code_snippet)
"""

# Re-export enums from models for backward compatibility
from models.static_analysis_models import (
    StaticAnalysisSecurityLevel as SecurityLevel,
)
from models.static_analysis_models import (
    ValidationType,
)

# Re-export main components for backward compatibility
from .symbolic import SymbolicValidator, ValidationConfig

__all__ = ["SymbolicValidator", "ValidationConfig", "SecurityLevel", "ValidationType"]
