"""Configuration for symbolic analysis and validation.

This module provides configuration classes for controlling symbolic analysis
and code validation behaviors. It defines parameters for different validation
types, security levels, and analysis constraints.

The configuration system allows fine-tuning of:
- Validation types (symbolic execution, taint analysis, formal verification)
- Security assessment levels
- Analysis timeouts and depth limits
- Tool-specific parameters

Classes:
    SymbolicAnalysisConfig: Main configuration class for validation parameters

Example:
    # Default configuration
    config = SymbolicAnalysisConfig()

    # Custom configuration
    config = SymbolicAnalysisConfig(
        validation_types=[ValidationType.SYMBOLIC_EXECUTION],
        security_level=StaticAnalysisSecurityLevel.HIGH,
        timeout=300
    )
"""

from __future__ import annotations

from dataclasses import dataclass

from models.static_analysis_models import (
    StaticAnalysisSecurityLevel as SecurityLevel,
)
from models.static_analysis_models import (
    ValidationType,
)


@dataclass
class SymbolicAnalysisConfig:
    """Configuration for symbolic analysis and validation operations.

    This class encapsulates all configuration parameters for symbolic analysis
    and code validation. It provides sensible defaults while allowing
    customization for specific analysis requirements.

    Attributes:
        validation_types (Optional[List[ValidationType]]): List of validation
            types to perform. Defaults to symbolic execution and taint analysis.
        security_level (StaticAnalysisSecurityLevel): Security assessment level for analysis.
            Reserved for future use. Defaults to MEDIUM.
        timeout (int): Maximum time in seconds for analysis operations.
            Defaults to 180 seconds.
        max_depth (int): Maximum analysis depth for recursive operations.
            Reserved for future use. Defaults to 10.

    Example:
        # Use default configuration
        config = SymbolicAnalysisConfig()

        # Custom configuration for high security
        config = SymbolicAnalysisConfig(
            validation_types=[
                ValidationType.SYMBOLIC_EXECUTION,
                ValidationType.TAINT_ANALYSIS,
                ValidationType.FORMAL_VERIFICATION
            ],
            security_level=StaticAnalysisSecurityLevel.HIGH,
            timeout=300,
            max_depth=15
        )
    """

    validation_types: list[ValidationType] | None = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM  # Reserved for future use
    timeout: int = 180
    max_depth: int = 10  # Reserved for future use

    def __post_init__(self) -> None:
        """Initialize default validation types if not provided.

        Sets up default validation types (symbolic execution and taint analysis)
        if no specific types were provided during initialization.
        """
        if self.validation_types is None:
            self.validation_types = [
                ValidationType.SYMBOLIC_EXECUTION,
                ValidationType.TAINT_ANALYSIS,
            ]


# Alias for backward compatibility
ValidationConfig = SymbolicAnalysisConfig
