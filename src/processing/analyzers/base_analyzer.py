"""Base module for implementing various code processors.

This module provides the abstract base class for all code processors in the system.
Each language-specific processor must inherit from BaseCodeProcessor and implement all
abstract methods to provide consistent analysis capabilities across different
programming languages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.core_models import (
    CoreAnalysisResult,
)


class BaseCodeProcessor(ABC):
    """Base interface for all code processors.

    This abstract class defines the standard interface that all language-specific
    code processors must implement. It ensures consistent behavior across different
    programming language processors and provides a uniform API for code analysis.

    Concrete implementations should handle language-specific parsing, entity
    extraction, relationship detection, and other analysis tasks.
    """

    @abstractmethod
    async def analyze_file(self, file_path: str) -> CoreAnalysisResult:
        """Analyze a single file and return comprehensive analysis results.

        This method performs complete analysis on a source code file, including
        parsing, entity extraction, relationship detection, complexity calculation,
        and security pattern identification.

        Args:
            file_path (str): The absolute path to the file to analyze.
                           Must be a valid path to an existing file.

        Returns:
            CoreAnalysisResult: A comprehensive analysis result containing:
                - success: Whether the analysis completed successfully
                - language: The detected programming language
                - entities: List of code entities (classes, functions, variables)
                - relationships: List of relationships between entities
                - dependencies: External and internal dependencies
                - complexity_metrics: Code complexity measurements
                - security_patterns: Detected security-related patterns
                - error: Error message if analysis failed
                - metadata: Additional language-specific information

        Raises:
            OSError: If the file cannot be read or accessed
            ValueError: If the file path is invalid
        """

    @abstractmethod
    async def analyze_directory(self, directory_path: str) -> CoreAnalysisResult:
        """Analyze all supported files in a directory.

        Recursively analyzes all files with supported extensions within the
        specified directory. Results are aggregated to provide directory-level
        insights and metrics.

        Args:
            directory_path (str): The absolute path to the directory to analyze.
                                Must be a valid directory path.

        Returns:
            CoreAnalysisResult: Aggregated analysis results for all analyzed files,
                          including directory-level metrics and summaries.

        Raises:
            OSError: If the directory cannot be accessed
            ValueError: If the directory path is invalid
        """

    @abstractmethod
    def get_supported_extensions(self) -> list[str]:
        """Return list of supported file extensions.

        Returns:
            List[str]: List of file extensions this analyzer can process,
                      including the leading dot (e.g., ['.py', '.pyx']).
                      Extensions should be lowercase.
        """

    @abstractmethod
    async def analyze_snippet(
        self, code_snippet: str, file_path: str | None = None
    ) -> CoreAnalysisResult:
        """Analyze a code snippet without requiring a file.

        This method allows analysis of code fragments or dynamically generated
        code without needing to write to disk. Useful for real-time analysis,
        code generation validation, or analyzing code from external sources.

        Args:
            code_snippet (str): The source code to analyze as a string.
            file_path (Optional[str]): Optional file path for context.
                                     Used for relative import resolution
                                     and error reporting. Defaults to None.

        Returns:
            CoreAnalysisResult: Analysis results for the code snippet, with the
                          same structure as file-based analysis.

        Raises:
            SyntaxError: If the code snippet contains syntax errors
            ValueError: If the code snippet is empty or invalid
        """

    @abstractmethod
    def get_language_name(self) -> str:
        """Return the name of the programming language this analyzer handles.

        Returns:
            str: The canonical name of the programming language
                (e.g., 'python', 'javascript', 'typescript').
                Should be lowercase.
        """

    def can_analyze_file(self, file_path: str) -> bool:
        """Check if this code processor can handle the given file.

        Determines whether the file has an extension supported by this processor.
        This method is used by the processor factory to route files to the
        appropriate processor.

        Args:
            file_path (str): The path to the file to check.
                           Can be absolute or relative.

        Returns:
            bool: True if the file extension is in the list of supported
                 extensions, False otherwise. Case-insensitive comparison.
        """
        return any(file_path.endswith(ext) for ext in self.get_supported_extensions())

    async def validate_syntax(self, content: str, file_path: str | None = None) -> bool:
        """Validate if the code has correct syntax.

        Performs a syntax check on the provided code without full analysis.
        This is a lightweight operation compared to full analysis and is useful
        for quick validation in editors or pre-commit hooks.

        Args:
            content (str): The source code to validate.
            file_path (Optional[str]): Optional file path for context.
                                     Used for error reporting and module
                                     resolution. Defaults to None.

        Returns:
            bool: True if the code has valid syntax, False if there are
                 syntax errors or if validation fails for any reason.

        Note:
            This method catches all exceptions and returns False for any
            failure, making it safe to use for validation without error handling.
        """
        try:
            result = await self.analyze_snippet(content, file_path)
            return result.success
        except Exception:
            return False
