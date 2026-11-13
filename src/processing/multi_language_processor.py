"""Multi-language processing with version detection and fallback strategies.

This module provides language processors that can detect language versions
and apply fallback parsing strategies when primary parsing fails.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from models.core_models import CoreCodeEntity, EntityRelation, EntityType

from .languages.base import (
    LanguageConfig,
    LanguageProcessor,
    ParseResult,
    ParserType,
)


def ensure_parse_result(
    result: Any, file_path: Path, language: str, error_context: str = ""
) -> ParseResult:
    """Ensure any result is converted to a proper ParseResult object.

    Args:
        result: Result object to validate/convert
        file_path: Path to the file being processed
        language: Programming language name
        error_context: Additional context for error reporting

    Returns:
        ParseResult: Properly typed result object
    """
    if isinstance(result, ParseResult):
        return result

    if isinstance(result, dict):
        logger.warning(
            "Converting dict result to ParseResult for %s (%s): %s",
            file_path,
            error_context,
            result,
        )
        # Try to extract meaningful data from the dict
        success = result.get("success", False)
        entities = result.get("entities", [])
        relationships = result.get("relationships", [])
        errors = result.get("errors", [f"Dict result converted: {result}"])

        return ParseResult(
            success=success,
            file_path=str(file_path),
            language=language,
            parser_used=ParserType.REGEX_FALLBACK,
            ast=result.get("ast"),
            entities=entities,
            relationships=relationships,
            lines_of_code=result.get("lines_of_code", 0),
            complexity_score=result.get("complexity_score", 0.0),
            parse_time_ms=result.get("parse_time_ms", 0.0),
            errors=errors,
            warnings=result.get("warnings", []),
            properties=result.get("properties", {}),
        )

    # For any other type, create a minimal ParseResult
    logger.warning(
        "Converting %s result to ParseResult for %s (%s)",
        type(result),
        file_path,
        error_context,
    )
    return ParseResult(
        success=False,
        file_path=str(file_path),
        language=language,
        parser_used=ParserType.REGEX_FALLBACK,
        ast=None,
        entities=[],
        relationships=[],
        errors=[f"Invalid result type {type(result)}: {result}"],
    )


logger = logging.getLogger(__name__)


class VersionDetectorRegistry:
    """Registry for language version detection strategies."""

    def __init__(self) -> None:
        """Initialize the version detector registry."""
        self._detectors: dict[str, Callable] = {}

    def register_detector(self, language: str, detector_func: Callable):
        """Register a version detector for a language.

        Args:
            language: Name of the programming language
            detector_func: Callable that takes (file_path, content) and returns version string
        """
        self._detectors[language] = detector_func

    async def detect_version(self, language: str, file_path: Path, content: str) -> str | None:
        """Detect the version of a language from file content.

        Args:
            language: Programming language name
            file_path: Path to the file being analyzed
            content: File content as string

        Returns:
            Optional[str]: Detected version string or None if detection fails
        """
        if language not in self._detectors:
            return None

        try:
            result = await self._detectors[language](file_path, content)
            return result if isinstance(result, str) else None
        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Version detection failed for %s: %s", language, e)
            return None


class FallbackManager:
    """Manages fallback strategies when primary parsing fails."""

    def __init__(self) -> None:
        """Initialize the fallback manager."""
        self._fallback_strategies: dict[str, list[Callable]] = {}

    def register_fallback(self, language: str, strategy_func: Callable):
        """Register a fallback strategy for a language.

        Args:
            language: Name of the programming language
            strategy_func: Callable that takes (file_path, content, error) and returns ParseResult
        """
        if language not in self._fallback_strategies:
            self._fallback_strategies[language] = []
        self._fallback_strategies[language].append(strategy_func)

    async def handle_failure(
        self, language: str, file_path: Path, content: str, error: Exception
    ) -> ParseResult:
        """Handle parsing failure using fallback strategies.

        Args:
            language: Programming language name
            file_path: Path to the file being parsed
            content: File content as string
            error: Exception that caused the primary parsing to fail

        Returns:
            ParseResult: Result from successful fallback strategy or failure result
        """
        strategies = self._fallback_strategies.get(language, [])

        for i, strategy in enumerate(strategies):
            try:
                logger.info("Trying fallback strategy %d for %s", i + 1, language)
                # Try to pass language parameter if the strategy supports it
                try:
                    result = await strategy(file_path, content, error, language)
                except TypeError:
                    # Fallback for strategies that don't accept language parameter
                    result = await strategy(file_path, content, error)

                # Use utility function to ensure proper type
                result = ensure_parse_result(
                    result, file_path, language, f"fallback strategy {i + 1}"
                )

                if result.success:
                    logger.info("Fallback strategy %d succeeded for %s", i + 1, language)
                    return result

            except (RuntimeError, ValueError, OSError) as fallback_error:
                logger.warning(
                    "Fallback strategy %d failed for %s: %s",
                    i + 1,
                    language,
                    fallback_error,
                )
                continue

        # All fallback strategies failed
        return ParseResult(
            success=False,
            file_path=str(file_path),
            language=language,
            parser_used=ParserType.REGEX_FALLBACK,
            ast=None,
            errors=[
                f"Primary parser failed: {error}",
                "All fallback strategies failed",
            ],
        )


class MultiLanguageProcessor(LanguageProcessor):
    """Language processor with version detection and fallback support."""

    def __init__(
        self,
        config: LanguageConfig,
        version_detector: VersionDetectorRegistry,
        fallback_mgr: FallbackManager,
    ):
        """Initialize the multi-language processor.

        Args:
            config: Language configuration settings
            version_detector: Registry for version detection strategies
            fallback_mgr: Manager for fallback parsing strategies
        """
        super().__init__(config)
        self.version_detector = version_detector
        self.fallback_manager = fallback_mgr

    async def _initialize_impl(self) -> None:
        """Initialize multi-language processor."""
        # This processor doesn't need specific initialization

    async def parse_file(self, file_path: Path, content: str) -> ParseResult:
        """Parse with version detection and fallback support.

        Args:
            file_path: Path to the file being parsed
            content: File content as string

        Returns:
            ParseResult: Parsing result with entities, relationships, and metadata
        """

        # Detect language version first
        detected_version = await self.version_detector.detect_version(
            self.config.name, file_path, content
        )

        try:
            # Try primary parsing
            result = await self._primary_parse(file_path, content, detected_version)
            # Ensure result is properly typed
            result = ensure_parse_result(result, file_path, self.config.name, "primary parse")
            if hasattr(result, "properties") and hasattr(result.properties, "__setitem__"):
                result.properties["detected_version"] = detected_version
            return result

        except (RuntimeError, ValueError, OSError) as e:
            logger.warning("Primary parser failed for %s: %s", file_path, e)

            # Try fallback strategies
            result = await self.fallback_manager.handle_failure(
                self.config.name, file_path, content, e
            )
            # Ensure result is properly typed (should already be handled in handle_failure)
            result = ensure_parse_result(result, file_path, self.config.name, "fallback")
            if hasattr(result, "properties") and hasattr(result.properties, "__setitem__"):
                result.properties["detected_version"] = detected_version
            return result

    async def _primary_parse(
        self, file_path: Path, content: str, version: str | None
    ) -> ParseResult:
        """Primary parsing method - to be implemented by language processors.

        Args:
            file_path: Path to the file being parsed
            content: File content as string
            version: Detected language version (may be None)

        Returns:
            ParseResult containing parsing results

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        # Parameters are used by subclasses
        _ = file_path, content, version
        raise NotImplementedError("Subclasses must implement _primary_parse")

    async def extract_entities(self, ast: Any, file_path: Path) -> list[CoreCodeEntity]:
        """Extract entities - implemented by specific processors.

        Args:
            ast: Abstract syntax tree or parsed representation
            file_path: Path to the file being processed

        Returns:
            List of extracted code entities

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        # Parameters are used by subclasses
        _ = ast, file_path
        raise NotImplementedError("Subclasses must implement extract_entities")

    async def extract_relationships(
        self, ast: Any, entities: list[CoreCodeEntity], file_path: Path
    ) -> list[EntityRelation]:
        """Extract relationships - implemented by specific processors.

        Args:
            ast: Abstract syntax tree or parsed representation
            entities: List of previously extracted entities
            file_path: Path to the file being processed

        Returns:
            List of extracted entity relationships

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        # Parameters are used by subclasses
        _ = ast, entities, file_path
        raise NotImplementedError("Subclasses must implement extract_relationships")


# Global instances
version_detector_registry = VersionDetectorRegistry()
fallback_manager = FallbackManager()


# Default version detectors
async def detect_python_version(_file_path: Path, content: str) -> str | None:
    """Detect Python version from file content."""
    # Look for version hints in shebang or encoding declarations
    lines = content.split("\n")[:5]  # Check first 5 lines

    for line in lines:
        if "python3" in line.lower():
            return "3.x"
        if "python2" in line.lower():
            return "2.x"
        if line.startswith("#") and "python" in line.lower():
            # Extract version from shebang if possible
            if "3." in line:
                return "3.x"
            if "2." in line:
                return "2.x"

    # Check for Python 3 specific syntax features
    version_features = [
        ('"', "3.6+"),  # f-strings
        ("async de", "3.5+"),
        ("await ", "3.5+"),
        ("yield from", "3.3+"),
    ]

    for feature, version in version_features:
        if feature in content:
            return version

    return None


async def detect_java_version(_file_path: Path, content: str) -> str | None:
    """Detect Java version from file content."""
    # Look for version-specific features
    if "var " in content:  # Local variable type inference (Java 10+)
        return "10+"
    if "record " in content:  # Records (Java 14+)
        return "14+"
    if "sealed " in content:  # Sealed classes (Java 17+)
        return "17+"
    # Switch expressions (Java 14+)
    if "switch" in content and "->" in content:
        return "14+"
    if "module " in content:  # Module system (Java 9+)
        return "9+"

    return "8+"  # Default to Java 8+


async def detect_rust_version(file_path: Path, content: str) -> str | None:
    """Detect Rust version from file content."""
    # Look for edition-specific or version-specific features
    version_features = [
        (["async fn", "async move"], "1.39+"),
        (["const fn"], "1.31+"),
        (["dyn "], "1.27+"),
        (["impl Trait"], "1.26+"),
        (["?", "Result"], "1.13+"),  # ? operator with Result
    ]

    for features, version in version_features:
        if all(feature in content for feature in features):
            return version

    # Check for edition in Cargo.toml (if it's a lib.rs or main.rs)
    if file_path.name in ["lib.rs", "main.rs"]:
        cargo_toml = file_path.parent / "Cargo.toml"
        if cargo_toml.exists():
            try:
                cargo_content = cargo_toml.read_text()
                if 'edition = "2021"' in cargo_content:
                    return "1.56+"  # Rust 2021 edition
                if 'edition = "2018"' in cargo_content:
                    return "1.31+"  # Rust 2018 edition
            except OSError:
                pass

    return "1.0+"  # Default to Rust 1.0+


# Register default detectors
version_detector_registry.register_detector("python", detect_python_version)
version_detector_registry.register_detector("java", detect_java_version)
version_detector_registry.register_detector("rust", detect_rust_version)


# Default fallback strategies
async def regex_fallback_strategy(
    file_path: Path, content: str, _error: Exception, language: str = "unknown"
) -> ParseResult:
    """Basic regex-based fallback parsing strategy.

    Uses regular expressions to extract basic code entities like functions
    and classes when primary parsing fails.

    Args:
        file_path: Path to the file being parsed
        content: File content as string
        _error: Exception that caused primary parsing to fail (unused)
        language: Programming language name to use for entities

    Returns:
        ParseResult: Result with basic entities extracted via regex
    """

    entities: list[CoreCodeEntity] = []
    relationships: list[EntityRelation] = []

    # Very basic regex-based entity extraction
    # Function definitions
    func_pattern = r"(?:def|function|func)\s+(\w+)\s*\("
    for match in re.finditer(func_pattern, content, re.IGNORECASE):
        entities.append(
            CoreCodeEntity(
                id=f"func:{match.group(1)}:{file_path}",
                name=match.group(1),
                type=EntityType.FUNCTION,
                language=language,
                file_path=str(file_path),
                line_number=content[: match.start()].count("\n") + 1,
                end_line_number=None,
                column_start=0,
                column_end=0,
                visibility="public",
                code_snippet=None,
                docstring=None,
                signature=None,
                source_code=None,
                created_at=None,
                updated_at=None,
            )
        )

    # Class definitions
    class_pattern = r"(?:class|struct)\s+(\w+)"
    for match in re.finditer(class_pattern, content, re.IGNORECASE):
        entities.append(
            CoreCodeEntity(
                id=f"class:{match.group(1)}:{file_path}",
                name=match.group(1),
                type=EntityType.CLASS,
                language=language,
                file_path=str(file_path),
                line_number=content[: match.start()].count("\n") + 1,
                end_line_number=None,
                column_start=0,
                column_end=0,
                visibility="public",
                code_snippet=None,
                docstring=None,
                signature=None,
                source_code=None,
                created_at=None,
                updated_at=None,
            )
        )

    return ParseResult(
        success=True,
        file_path=str(file_path),
        language=language,
        parser_used=ParserType.REGEX_FALLBACK,
        ast=None,
        entities=entities,
        relationships=relationships,
        lines_of_code=len(content.splitlines()),
    )


async def structure_only_fallback(
    file_path: Path, content: str, error: Exception, language: str = "unknown"
) -> ParseResult:
    """Emergency fallback that only captures file metadata.

    Args:
        file_path: Path to the file being parsed
        content: File content as string
        error: Exception that caused parsing to fail
        language: Programming language name to use for entities

    Returns:
        ParseResult: Basic result with file metadata entity
    """
    file_entity = CoreCodeEntity(
        id=f"file:{file_path}",
        name=file_path.name,
        type=EntityType.FILE,
        language=language,
        file_path=str(file_path),
        line_number=1,
        end_line_number=None,
        column_start=0,
        column_end=0,
        visibility="public",
        code_snippet=None,
        docstring=None,
        signature=None,
        source_code=None,
        created_at=None,
        updated_at=None,
        properties={
            "file_size": len(content),
            "line_count": len(content.splitlines()),
            "parse_error": str(error),
        },
    )

    return ParseResult(
        success=True,
        file_path=str(file_path),
        language=language,
        parser_used=ParserType.REGEX_FALLBACK,
        ast=None,
        entities=[file_entity],
        relationships=[],
        lines_of_code=len(content.splitlines()),
    )


# Register default fallback strategies
fallback_manager.register_fallback("python", regex_fallback_strategy)
fallback_manager.register_fallback("python", structure_only_fallback)
fallback_manager.register_fallback("java", regex_fallback_strategy)
fallback_manager.register_fallback("java", structure_only_fallback)
fallback_manager.register_fallback("rust", regex_fallback_strategy)
fallback_manager.register_fallback("rust", structure_only_fallback)
