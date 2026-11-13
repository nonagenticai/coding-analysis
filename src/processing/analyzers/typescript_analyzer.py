"""
Advanced TypeScript Processor for Comprehensive Type System Analysis.

This module provides deep analysis capabilities specifically for TypeScript code,
extending beyond basic JavaScript analysis to include type system analysis,
generic type checking, decorator analysis, and TypeScript-specific features.

The processor builds on the TypeScript processor to provide comprehensive
analysis including:

Type Safety Analysis:
- Explicit type annotations vs implicit types
- Usage of 'any' and 'unknown' types
- Type assertion patterns
- Null safety and optional chaining usage
- Generic type parameter analysis

Code Quality Metrics:
- Type safety score calculation
- Complexity metrics including type complexity
- Maintainability index with TypeScript considerations
- Coupling analysis based on type dependencies

Security Analysis:
- TypeScript-specific security patterns
- Type-related vulnerabilities (any type usage)
- Unsafe DOM manipulation patterns
- Strict equality enforcement

Advanced Features:
- Decorator usage analysis
- Interface implementation tracking
- Generic constraint analysis
- Utility type usage patterns
- Async/await pattern analysis

The processor provides both file-level and project-level analysis with
detailed metrics and recommendations for improving TypeScript code quality.

Classes:
    APITypeScriptProcessor: Main processor for TypeScript code analysis
    TypeSafetyMetrics: Metrics for type safety analysis
    TypeScriptComplexityMetrics: TypeScript-specific complexity measurements

Example:
    # Analyze a TypeScript file
    processor = APITypeScriptProcessor()
    result = await processor.analyze_file("src/app.ts")

    # Get type safety metrics
    type_safety = result.metadata["type_safety"]
    print(f"Type safety score: {type_safety['type_safety_score']}")

    # Analyze entire project
    project_result = await processor.analyze_directory("src/")
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.file_service import FileService
from models.core_models import CodeSecurityPattern as SecurityPattern
from models.core_models import (
    CoreAnalysisResult,
    CoreCodeEntity,
    DependencyInfoModel,
    EntityRelation,
)

from ..languages.typescript.typescript_processor import (
    TYPESCRIPT_CONFIG,
    TSLanguageProcessor,
)
from .base_analyzer import BaseCodeProcessor

logger = logging.getLogger(__name__)


@dataclass
class TypeSafetyMetrics:
    """Metrics for TypeScript type safety analysis.

    This class encapsulates various metrics related to TypeScript type safety
    and usage patterns. It provides a comprehensive view of how well types
    are used in TypeScript code.

    Attributes:
        total_types (int): Total number of type definitions found
        explicit_types (int): Number of explicit type annotations
        any_types (int): Number of 'any' type usages (reduces type safety)
        unknown_types (int): Number of 'unknown' type usages (safer than 'any')
        type_assertions (int): Number of type assertions (as, <Type>)
        nullable_safety_violations (int): Potential null safety issues
        generic_usage_count (int): Number of generic type usages
        decorator_count (int): Number of decorators used
        interface_implementations (int): Number of interface implementations
        type_safety_score (float): Overall type safety score (0.0 to 1.0)

    Note:
        Higher type_safety_score indicates better type safety practices.
        Heavy use of 'any' types will lower the score.
    """

    total_types: int = 0
    explicit_types: int = 0
    any_types: int = 0
    unknown_types: int = 0
    type_assertions: int = 0
    nullable_safety_violations: int = 0
    generic_usage_count: int = 0
    decorator_count: int = 0
    interface_implementations: int = 0
    type_safety_score: float = 0.0


@dataclass
class TypeScriptComplexityMetrics:
    """TypeScript-specific complexity metrics for code quality assessment.

    This class provides metrics specifically tailored to TypeScript code
    complexity, extending beyond traditional cyclomatic complexity to include
    type system complexity and TypeScript-specific patterns.

    Attributes:
        cyclomatic_complexity (float): Traditional cyclomatic complexity
        type_complexity (float): Complexity introduced by type definitions
        generic_complexity (float): Complexity from generic type parameters
        inheritance_depth (int): Maximum inheritance depth in class hierarchies
        coupling_score (float): Coupling between modules and types
        maintainability_index (float): Overall maintainability score (0-100)

    Note:
        Lower complexity values generally indicate more maintainable code.
        Type complexity considers union types, intersections, and generic constraints.
    """

    cyclomatic_complexity: float = 0.0
    type_complexity: float = 0.0
    generic_complexity: float = 0.0
    inheritance_depth: int = 0
    coupling_score: float = 0.0
    maintainability_index: float = 0.0


class APITypeScriptProcessor(BaseCodeProcessor):
    """Advanced TypeScript analyzer processor with comprehensive type system analysis.

    This processor provides deep analysis of TypeScript code with focus on type
    safety, complexity metrics, and TypeScript-specific patterns. It extends
    the base processor to include TypeScript-specific features and metrics.

    The processor performs:
    - Type safety analysis with scoring
    - Generic type complexity assessment
    - Decorator usage analysis
    - Interface implementation tracking
    - Security pattern detection
    - Maintainability index calculation

    Attributes:
        file_service (Optional[FileService]): File service for file operations
        typescript_processor (TSLanguageProcessor): Core TypeScript processor
        language (str): Language identifier ("typescript")
        analysis_patterns (Dict[str, str]): Regex patterns for TypeScript analysis

    Example:
        # Analyze single file
        processor = APITypeScriptProcessor()
        result = await processor.analyze_file("app.ts")

        # Get type safety metrics
        type_safety = result.metadata["type_safety"]
        print(f"Type safety score: {type_safety['type_safety_score']}")
    """

    def __init__(self, file_service: FileService | None = None):
        """Initialize the TypeScript analyzer with optional file service.

        Args:
            file_service (Optional[FileService]): File service for advanced
                                                file operations. If None, uses
                                                standard file operations.
        """
        self.file_service = file_service
        self.typescript_processor = TSLanguageProcessor(TYPESCRIPT_CONFIG)
        self.language = "typescript"

        # Analysis patterns for advanced features
        self.analysis_patterns = {
            "any_type": r"\b(\w+)\s*:\s*any\b",
            "unknown_type": r"\b(\w+)\s*:\s*unknown\b",
            "type_assertion": r"<[^>]+>|\s+as\s+\w+",
            "non_null_assertion": r"!\s*[\.\[;,)]",
            "optional_chaining": r"\?\.",
            "nullish_coalescing": r"\?\?",
            "strict_equality": r"===|!==",
            "loose_equality": r"==|!=",
            "promise_usage": r"Promise<[^>]+>|async\s+|await\s+",
            "utility_types": r"\b(Partial|Required|Pick|Omit|Record|Exclude|Extract|"
            r"ReturnType|Parameters)\s*<",
        }

    async def initialize(self) -> None:
        """Initialize the TypeScript processor for parsing operations.

        This method must be called before using the analyzer to ensure the
        underlying TypeScript processor is properly initialized.
        """
        await self.typescript_processor.initialize()

    def get_supported_extensions(self) -> list[str]:
        """Get supported TypeScript file extensions.

        Returns:
            List[str]: List of supported file extensions:
                - .ts: Standard TypeScript files
                - .tsx: TypeScript React files with JSX
                - .d.ts: TypeScript declaration files
        """
        return [".ts", ".tsx", ".d.ts"]

    def get_language_name(self) -> str:
        """Return the name of the programming language this analyzer handles.

        Returns:
            str: Always returns "typescript" for this analyzer.
        """
        return self.language

    async def analyze_file(self, file_path: str) -> CoreAnalysisResult:
        """Analyze a TypeScript file with comprehensive type system analysis.

        Performs deep analysis of a TypeScript file including type safety assessment,
        complexity metrics, security pattern detection, and entity extraction.

        Args:
            file_path (str): Absolute path to the TypeScript file to analyze.
                           Must have a supported TypeScript extension.

        Returns:
            CoreAnalysisResult: Comprehensive analysis results including:
                - Type safety metrics and scoring
                - TypeScript-specific complexity measurements
                - Security pattern detection results
                - Extracted entities with type information
                - Processed relationships with coupling analysis
                - Dependency information from imports

        Raises:
            The method catches exceptions and returns them in CoreAnalysisResult:
            - OSError/IOError: For file access issues
            - UnicodeDecodeError: For encoding problems
            - Exception: For any other analysis errors
        """
        try:
            # Initialize if not already done
            if not hasattr(self.typescript_processor, "_initialized"):
                await self.initialize()

            # Read the file
            if self.file_service:
                content = self.file_service.read_file(file_path)
            else:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

            return await self.analyze_snippet(content, file_path)

        except (OSError, UnicodeDecodeError) as e:
            logger.error("Error analyzing TypeScript file %s: %s", file_path, e)
            return self._create_error_result(file_path, str(e))
        except Exception as e:
            logger.error("Unexpected error analyzing TypeScript file %s: %s", file_path, e)
            return self._create_error_result(file_path, str(e))

    async def analyze_snippet(
        self, code_snippet: str, file_path: str | None = None
    ) -> CoreAnalysisResult:
        """Analyze a TypeScript code snippet with advanced type system analysis.

        Performs comprehensive analysis of TypeScript code including type safety
        assessment, complexity metrics, security patterns, and entity extraction.
        This method provides the core analysis functionality for both files and
        code snippets.

        Args:
            code_snippet (str): TypeScript source code to analyze.
            file_path (Optional[str]): Optional file path for context.
                                     Defaults to "snippet.ts" if not provided.

        Returns:
            CoreAnalysisResult: Detailed analysis results including:
                - Type safety metrics with scoring
                - TypeScript complexity measurements
                - Security pattern detection
                - Entity extraction with type information
                - Relationship analysis with coupling metrics
                - Import dependency analysis
                - Comprehensive metadata with TypeScript-specific insights

        Raises:
            The method catches exceptions and returns them in CoreAnalysisResult:
            - ValueError/TypeError: For invalid input parameters
            - AttributeError: For missing required attributes
            - Exception: For any other analysis errors
        """
        try:
            if file_path is None:
                file_path = "snippet.ts"

            # Use TypeScript processor for basic parsing
            parse_result = await self.typescript_processor.parse_file(Path(file_path), code_snippet)

            if not parse_result.success:
                return self._create_error_result(
                    file_path,
                    f"TypeScript parsing failed: {'; '.join(parse_result.errors)}",
                )

            # Perform advanced analysis
            type_safety_metrics = await self._analyze_type_safety(
                code_snippet, parse_result.entities
            )
            complexity_metrics = await self._analyze_complexity(code_snippet, parse_result.entities)
            security_patterns = await self._analyze_security_patterns(code_snippet)
            dependencies = await self._extract_dependencies(code_snippet, file_path)

            # Process entities with additional analysis
            processed_entities = await self._process_entities(
                parse_result.entities, type_safety_metrics
            )

            # Process relationships with type information
            processed_relationships = await self._process_relationships(
                parse_result.relationships, parse_result.entities
            )

            return CoreAnalysisResult(
                success=True,
                language=self.language,
                file_path=file_path,
                entities=processed_entities,
                relationships=processed_relationships,
                dependencies=dependencies,
                complexity_metrics=None,
                security_patterns=security_patterns,
                error=None,
                parse_time_ms=parse_result.parse_time_ms,
                total_lines=parse_result.lines_of_code,
                metadata={
                    "parser_used": str(parse_result.parser_used),
                    "typescript_version": "4.9+",  # Assumed version support
                    "type_safety": type_safety_metrics.__dict__,
                    "complexity": complexity_metrics.__dict__,
                    "typescript_specific": {
                        "has_generics": parse_result.properties.get("has_generics", False),
                        "has_decorators": parse_result.properties.get("has_decorators", False),
                        "export_count": parse_result.properties.get("export_count", 0),
                        "type_count": parse_result.properties.get("type_count", 0),
                    },
                },
            )

        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Error analyzing TypeScript snippet: %s", e)
            return self._create_error_result(file_path or "unknown", str(e))
        except Exception as e:
            logger.error("Unexpected error analyzing TypeScript snippet: %s", e)
            return self._create_error_result(file_path or "unknown", str(e))

    async def analyze_directory(self, directory_path: str) -> CoreAnalysisResult:
        """Analyze all TypeScript files in a directory."""
        try:
            files: list[str] = []

            # Find TypeScript files
            if self.file_service:
                for ext in self.get_supported_extensions():
                    files.extend(self.file_service.find_files(directory_path, f"*{ext}"))
            else:
                for root, _, filenames in os.walk(directory_path):
                    for filename in filenames:
                        if any(filename.endswith(ext) for ext in self.get_supported_extensions()):
                            files.append(os.path.join(root, filename))

            all_entities: list[CoreCodeEntity] = []
            all_relationships: list[EntityRelation] = []
            all_dependencies: list[DependencyInfoModel] = []
            failed_files: list[str] = []

            # Aggregate metrics
            total_type_safety = TypeSafetyMetrics()
            total_complexity = TypeScriptComplexityMetrics()

            # Analyze files in parallel
            semaphore = asyncio.Semaphore(10)  # Limit concurrent analysis

            async def analyze_single_file(file_path: str):
                async with semaphore:
                    try:
                        result = await self.analyze_file(file_path)
                        return file_path, result
                    except (OSError, UnicodeDecodeError) as e:
                        logger.error("Error analyzing file %s: %s", file_path, e)
                        return file_path, None
                    except Exception as e:
                        logger.error("Unexpected error analyzing file %s: %s", file_path, e)
                        return file_path, None

            # Process files concurrently
            tasks = [analyze_single_file(file_path) for file_path in files]
            results = await asyncio.gather(*tasks)

            for file_path, result in results:
                if result and result.success:
                    all_entities.extend(result.entities)
                    all_relationships.extend(result.relationships)
                    if result.dependencies:
                        all_dependencies.extend(result.dependencies)

                    # Aggregate metrics
                    if result.complexity_metrics:
                        self._aggregate_metrics(
                            result.complexity_metrics,
                            total_type_safety,
                            total_complexity,
                        )
                else:
                    failed_files.append(file_path)

            # Calculate project-level metrics
            project_metrics = await self._calculate_project_metrics(
                all_entities,
                all_relationships,
                total_type_safety,
                total_complexity,
            )

            return CoreAnalysisResult(
                success=True,
                language=self.language,
                file_path=directory_path,
                entities=all_entities,
                relationships=all_relationships,
                dependencies=all_dependencies,
                complexity_metrics=None,
                security_patterns=[],
                error=None,
                parse_time_ms=0.0,
                total_lines=sum(
                    len(e.signature.splitlines()) if e.signature else 1 for e in all_entities
                ),
                metadata={
                    "files_analyzed": len(files) - len(failed_files),
                    "failed_files": failed_files,
                    "total_files": len(files),
                    "analysis_timestamp": datetime.now(UTC).isoformat(),
                    **project_metrics,  # Spread the project metrics into metadata
                },
            )

        except (OSError, PermissionError) as e:
            logger.error("Error analyzing TypeScript directory %s: %s", directory_path, e)
            return self._create_error_result(directory_path, str(e))
        except Exception as e:
            logger.error(
                "Unexpected error analyzing TypeScript directory %s: %s",
                directory_path,
                e,
            )
            return self._create_error_result(directory_path, str(e))

    async def _analyze_type_safety(
        self, code: str, entities: list[CoreCodeEntity]
    ) -> TypeSafetyMetrics:
        """Analyze TypeScript type safety metrics."""

        metrics = TypeSafetyMetrics()

        # Count type annotations and usage patterns
        metrics.any_types = len(re.findall(self.analysis_patterns["any_type"], code))
        metrics.unknown_types = len(re.findall(self.analysis_patterns["unknown_type"], code))
        metrics.type_assertions = len(re.findall(self.analysis_patterns["type_assertion"], code))

        # Count explicit type annotations
        type_annotation_pattern = r":\s*[a-zA-Z][a-zA-Z0-9<>\[\]|&,\s]*"
        metrics.explicit_types = len(re.findall(type_annotation_pattern, code))

        # Count generic usage
        generic_pattern = r"<[A-Z][a-zA-Z0-9,\s]*>"
        metrics.generic_usage_count = len(re.findall(generic_pattern, code))

        # Count decorators
        decorator_pattern = r"@[a-zA-Z][a-zA-Z0-9]*"
        metrics.decorator_count = len(re.findall(decorator_pattern, code))

        # Calculate type safety score
        total_typed_elements = metrics.explicit_types + metrics.any_types + metrics.unknown_types
        if total_typed_elements > 0:
            type_safety_penalty = metrics.any_types * 0.5 + metrics.unknown_types * 0.2
            metrics.type_safety_score = max(
                0.0,
                (metrics.explicit_types - type_safety_penalty) / total_typed_elements,
            )
        else:
            metrics.type_safety_score = 1.0  # No typing issues found

        metrics.total_types = len(
            [e for e in entities if e.type in ["interface", "type_alias", "class", "enum"]]
        )

        return metrics

    async def _analyze_complexity(
        self, code: str, entities: list[CoreCodeEntity]
    ) -> TypeScriptComplexityMetrics:
        """Analyze TypeScript-specific complexity metrics."""

        metrics = TypeScriptComplexityMetrics()

        # Cyclomatic complexity (basic calculation)
        complexity_keywords = [
            r"\bif\b",
            r"\belse\b",
            r"\bwhile\b",
            r"\bfor\b",
            r"\bswitch\b",
            r"\btry\b",
            r"\bcatch\b",
            r"\&&",
            r"\|\|",
        ]
        cyclomatic_count = sum(
            len(re.findall(pattern, code, re.IGNORECASE)) for pattern in complexity_keywords
        )
        metrics.cyclomatic_complexity = cyclomatic_count + 1.0

        # Type complexity
        generic_depth = 0
        nested_generics = re.findall(r"<[^<>]*<[^<>]*>[^<>]*>", code)
        generic_depth = len(nested_generics)

        union_types = len(re.findall(r"\|", code))
        intersection_types = len(re.findall(r"\&", code))

        metrics.type_complexity = generic_depth * 2.0 + union_types * 0.5 + intersection_types * 0.3

        # Generic complexity
        metrics.generic_complexity = len(re.findall(r"<[A-Z]", code)) * 0.5

        # Inheritance depth (approximate)
        class_entities = [e for e in entities if e.type == "class"]
        max_depth = 0
        for entity in class_entities:
            if entity.properties and "extends" in entity.properties:
                extends = entity.properties["extends"]
                if extends:
                    max_depth = max(max_depth, 1)  # Streamlined calculation
        metrics.inheritance_depth = max_depth

        # Coupling score (based on imports and dependencies)
        import_count = len(re.findall(r"import\s+", code))
        export_count = len(re.findall(r"export\s+", code))
        metrics.coupling_score = (import_count + export_count) * 0.1

        # Maintainability index (streamlined formula)
        lines_of_code = len(code.splitlines())
        if lines_of_code > 0:
            complexity_factor = metrics.cyclomatic_complexity + metrics.type_complexity
            metrics.maintainability_index = max(
                0.0, 100.0 - (complexity_factor * 2.0) - (lines_of_code * 0.1)
            )

        return metrics

    async def _analyze_security_patterns(self, code: str) -> list[SecurityPattern]:
        """Analyze security patterns in TypeScript code."""

        security_patterns = []

        # Check for dangerous patterns
        patterns = {
            "eval_usage": (r"\beval\s*\(", "Use of eval() can be dangerous"),
            "inner_html": (
                r"\.innerHTML\s*=",
                "Direct innerHTML assignment can lead to XSS",
            ),
            "document_write": (
                r"document\.write\s*\(",
                "document.write can be dangerous",
            ),
            "loose_equality": (
                r"==(?!=)|!=(?!=)",
                "Use strict equality (=== / !==) for type safety",
            ),
            "console_log": (
                r"console\.log\s*\(",
                "Remove console.log statements in production",
            ),
            "any_type_usage": (
                r":\s*any\b",
                "Avoid 'any' type for better type safety",
            ),
        }

        for pattern_name, (regex, description) in patterns.items():
            matches = list(re.finditer(regex, code))
            for match in matches:
                line_number = code[: match.start()].count("\n") + 1
                severity = (
                    "medium"
                    if pattern_name in ["loose_equality", "console_log", "any_type_usage"]
                    else "high"
                )

                security_patterns.append(
                    SecurityPattern(
                        type=pattern_name,
                        description=description,
                        line_number=line_number,
                        severity=severity,
                        cwe_id=None,
                        metadata={"match_text": match.group(0)},
                    )
                )

        return security_patterns

    async def _extract_dependencies(
        self,
        code: str,
        file_path: str,
    ) -> list[DependencyInfoModel]:
        """Extract dependencies from TypeScript imports."""

        dependencies = []

        # Match import statements with line numbers
        import_patterns = [
            (r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', "import"),
            (
                r'import\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
                "dynamic_import",
            ),  # Dynamic imports
            (
                r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
                "require",
            ),  # CommonJS requires
        ]

        seen_dependencies = set()  # Track to avoid duplicates

        for pattern, import_type in import_patterns:
            for match in re.finditer(pattern, code):
                module_path = match.group(1)
                line_number = code[: match.start()].count("\n") + 1

                # External dependencies only
                if not module_path.startswith("."):
                    # Extract package name (remove sub-paths)
                    package_name = module_path.split("/")[0]
                    if package_name.startswith("@"):  # Scoped packages
                        parts = module_path.split("/")
                        if len(parts) >= 2:
                            package_name = f"{parts[0]}/{parts[1]}"

                    # Avoid duplicates
                    if package_name not in seen_dependencies:
                        seen_dependencies.add(package_name)
                        dependencies.append(
                            DependencyInfoModel(
                                module_name=package_name,
                                import_type=import_type,
                                line_number=line_number,
                                is_external=True,
                                alias=None,
                                metadata={"full_path": module_path},
                            )
                        )

        return dependencies

    async def _process_entities(
        self,
        entities: list[CoreCodeEntity],
        type_safety_metrics: TypeSafetyMetrics,
    ) -> list[CoreCodeEntity]:
        """Process entities with additional TypeScript-specific analysis."""
        processed_entities = []

        for entity in entities:
            # Add type safety information
            enriched_properties = entity.properties or {}

            # Add type safety score for type-related entities
            if entity.type in ["interface", "type_alias", "class"]:
                enriched_properties["type_safety_impact"] = {
                    "uses_generics": "generic_parameters" in enriched_properties,
                    "has_decorators": bool(enriched_properties.get("decorators", [])),
                    "is_exported": enriched_properties.get("is_exported", False),
                }

            # Create processed entity
            processed_entity = CoreCodeEntity(
                id=entity.id,
                name=entity.name,
                type=entity.type,
                language=entity.language,
                file_path=entity.file_path,
                line_number=entity.line_number,
                end_line_number=entity.end_line_number,
                column_start=entity.column_start,
                column_end=entity.column_end,
                visibility=entity.visibility,
                code_snippet=entity.code_snippet,
                docstring=entity.docstring,
                signature=entity.signature,
                source_code=entity.source_code,
                created_at=entity.created_at,
                updated_at=entity.updated_at,
                is_abstract=getattr(entity, "is_abstract", False),
                is_static=getattr(entity, "is_static", False),
                is_async=getattr(entity, "is_async", False),
                properties=enriched_properties,
            )
            processed_entities.append(processed_entity)

        return processed_entities

    async def _process_relationships(
        self,
        relationships: list[EntityRelation],
        entities: list[CoreCodeEntity],
    ) -> list[EntityRelation]:
        """Process relationships with TypeScript-specific information."""
        processed_relationships = []

        for relationship in relationships:
            # Add strength indicators for different relationship types
            enriched_properties = relationship.properties or {}

            if relationship.type == "implements":
                enriched_properties["coupling_strength"] = "high"
            elif relationship.type == "extends":
                enriched_properties["coupling_strength"] = "very_high"
            elif relationship.type == "imports":
                enriched_properties["coupling_strength"] = "medium"
            elif relationship.type == "generic_constraint":
                enriched_properties["coupling_strength"] = "medium"
                enriched_properties["is_type_relationship"] = True

            # Create processed relationship
            processed_relationship = EntityRelation(
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                type=relationship.type,
                properties=enriched_properties,
            )
            processed_relationships.append(processed_relationship)

        return processed_relationships

    def _aggregate_metrics(
        self,
        file_metrics: dict[str, Any],
        total_type_safety: TypeSafetyMetrics,
        total_complexity: TypeScriptComplexityMetrics,
    ) -> None:
        """Aggregate metrics from individual file analysis."""
        if "type_safety" in file_metrics:
            ts_metrics = file_metrics["type_safety"]
            total_type_safety.total_types += ts_metrics.get("total_types", 0)
            total_type_safety.explicit_types += ts_metrics.get("explicit_types", 0)
            total_type_safety.any_types += ts_metrics.get("any_types", 0)
            total_type_safety.unknown_types += ts_metrics.get("unknown_types", 0)
            total_type_safety.type_assertions += ts_metrics.get("type_assertions", 0)
            total_type_safety.generic_usage_count += ts_metrics.get("generic_usage_count", 0)
            total_type_safety.decorator_count += ts_metrics.get("decorator_count", 0)

        if "complexity" in file_metrics:
            comp_metrics = file_metrics["complexity"]
            total_complexity.cyclomatic_complexity += comp_metrics.get("cyclomatic_complexity", 0)
            total_complexity.type_complexity += comp_metrics.get("type_complexity", 0)
            total_complexity.generic_complexity += comp_metrics.get("generic_complexity", 0)
            total_complexity.inheritance_depth = max(
                total_complexity.inheritance_depth,
                comp_metrics.get("inheritance_depth", 0),
            )
            total_complexity.coupling_score += comp_metrics.get("coupling_score", 0)

    async def _calculate_project_metrics(
        self,
        entities: list[CoreCodeEntity],
        relationships: list[EntityRelation],
        type_safety_metrics: TypeSafetyMetrics,
        complexity_metrics: TypeScriptComplexityMetrics,
    ) -> dict[str, Any]:
        """Calculate project-level metrics."""

        # Calculate overall type safety score
        if type_safety_metrics.total_types > 0:
            total_elements = (
                type_safety_metrics.explicit_types
                + type_safety_metrics.any_types
                + type_safety_metrics.unknown_types
            )
            if total_elements > 0:
                penalty = (
                    type_safety_metrics.any_types * 0.5 + type_safety_metrics.unknown_types * 0.2
                )
                type_safety_metrics.type_safety_score = max(
                    0.0,
                    (type_safety_metrics.explicit_types - penalty) / total_elements,
                )

        # Calculate average maintainability index
        entity_count = len(entities)
        if entity_count > 0:
            complexity_metrics.maintainability_index = max(
                0.0,
                100.0 - (complexity_metrics.cyclomatic_complexity / entity_count * 2.0),
            )

        return {
            "type_safety": type_safety_metrics.__dict__,
            "complexity": complexity_metrics.__dict__,
            "project_overview": {
                "total_entities": len(entities),
                "total_relationships": len(relationships),
                "entity_types": self._count_entity_types(entities),
                "relationship_types": self._count_relationship_types(relationships),
            },
        }

    def _count_entity_types(self, entities: list[CoreCodeEntity]) -> dict[str, int]:
        """Count entities by type."""
        counts: dict[str, int] = {}
        for entity in entities:
            entity_type_str = (
                str(entity.type) if hasattr(entity.type, "value") else str(entity.type)
            )
            counts[entity_type_str] = counts.get(entity_type_str, 0) + 1
        return counts

    def _count_relationship_types(self, relationships: list[EntityRelation]) -> dict[str, int]:
        """Count relationships by type."""
        counts: dict[str, int] = {}
        for relationship in relationships:
            relationship_type_str = (
                str(relationship.type)
                if hasattr(relationship.type, "value")
                else str(relationship.type)
            )
            counts[relationship_type_str] = counts.get(relationship_type_str, 0) + 1
        return counts

    def _create_error_result(self, file_path: str, error: str) -> CoreAnalysisResult:
        """Create an error result."""
        return CoreAnalysisResult(
            success=False,
            language=self.language,
            file_path=file_path,
            entities=[],
            relationships=[],
            dependencies=[],
            complexity_metrics=None,
            security_patterns=[],
            error=error,
            parse_time_ms=0.0,
            total_lines=0,
            metadata={
                "error_timestamp": datetime.now(UTC).isoformat(),
            },
        )


__all__ = ["APITypeScriptProcessor", "TypeSafetyMetrics", "TypeScriptComplexityMetrics"]
