"""Custom exceptions for symbolic analysis."""


class SymbolicAnalysisError(Exception):
    """Base exception for symbolic analysis errors."""

    pass


class CrossHairAnalysisError(SymbolicAnalysisError):
    """Exception raised when CrossHair analysis fails."""

    pass


class PysaAnalysisError(SymbolicAnalysisError):
    """Exception raised when Pysa analysis fails."""

    pass


class FormalVerificationError(SymbolicAnalysisError):
    """Exception raised when formal verification fails."""

    pass


class SymbolicValidationError(SymbolicAnalysisError):
    """Exception raised when symbolic code validation fails."""

    pass


class FileProcessingError(SymbolicAnalysisError):
    """Exception raised when file processing fails."""

    pass


# Alias for backward compatibility
ValidationError = SymbolicValidationError
