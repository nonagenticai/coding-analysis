"""
Semantic Parsing and Code Analysis Module

This module provides functionality for parsing code semantically and extracting
meaningful patterns for cross-language relationship detection.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from models.core_models import (
    CoreCodeEntity,
)

logger = logging.getLogger(__name__)


class SemanticRelationType(StrEnum):
    """Advanced semantic relationship types."""

    # Cross-language relationships
    FFI_CALL = "ffi_call"  # Foreign function interface calls
    API_ENDPOINT = "api_endpoint"  # REST/GraphQL API endpoints
    RPC_CALL = "rpc_call"  # Remote procedure calls

    # Data flow relationships
    DATA_DEPENDENCY = "data_dependency"  # Data flows from A to B
    STATE_MUTATION = "state_mutation"  # A modifies state of B
    EVENT_EMISSION = "event_emission"  # A emits events consumed by B

    # Architectural relationships
    SERVICE_CALL = "service_call"  # Microservice/service calls
    DATABASE_ACCESS = "database_access"  # Database operations
    CONFIGURATION_USE = "configuration_use"  # Configuration dependencies

    # Type system relationships
    TYPE_CONSTRAINT = "type_constraint"  # Generic type constraints
    TYPE_CONVERSION = "type_conversion"  # Type casting/conversion
    TYPE_ALIAS_RESOLUTION = "type_alias_resolution"  # Type alias usage

    # Advanced OOP relationships
    POLYMORPHIC_CALL = "polymorphic_call"  # Virtual method calls
    INTERFACE_BINDING = "interface_binding"  # Interface implementations
    MIXIN_COMPOSITION = "mixin_composition"  # Mixin/trait composition

    # Functional relationships
    HIGHER_ORDER_USAGE = "higher_order_usage"  # Higher-order function usage
    CLOSURE_CAPTURE = "closure_capture"  # Variable capture in closures
    MONAD_BINDING = "monad_binding"  # Monadic operations

    # Resource relationships
    RESOURCE_ACQUISITION = "resource_acquisition"  # Resource management
    MEMORY_SHARING = "memory_sharing"  # Shared memory access
    LOCK_DEPENDENCY = "lock_dependency"  # Synchronization dependencies


@dataclass
class SemanticContext:
    """Context information for semantic analysis."""

    file_path: Path
    language: str
    imports: set[str] = field(default_factory=set)
    exports: set[str] = field(default_factory=set)
    external_references: set[str] = field(default_factory=set)
    type_definitions: dict[str, str] = field(default_factory=dict)
    function_signatures: dict[str, str] = field(default_factory=dict)
    class_hierarchies: dict[str, list[str]] = field(default_factory=dict)

    # Language-specific context
    namespaces: set[str] = field(default_factory=set)
    packages: set[str] = field(default_factory=set)
    modules: set[str] = field(default_factory=set)


@dataclass
class CrossLanguagePattern:
    """Pattern for detecting cross-language relationships."""

    name: str
    description: str
    source_patterns: list[str]  # Regex patterns for source language
    target_patterns: list[str]  # Regex patterns for target language
    relationship_type: SemanticRelationType
    confidence_score: float = 1.0

    # Language constraints
    source_languages: set[str] | None = None
    target_languages: set[str] | None = None


class SemanticParser:
    """Parser for extracting semantic patterns from code entities."""

    def __init__(self) -> None:
        self.cross_language_patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> list[CrossLanguagePattern]:
        """Initialize cross-language relationship patterns."""
        return [
            # Rust FFI patterns
            CrossLanguagePattern(
                name="rust_c_ffi",
                description="Rust calling C functions via FFI",
                source_patterns=[
                    r'extern\s+"C"\s*\{[^}]*fn\s+(\w+)',
                    r"unsafe\s*\{[^}]*(\w+)\s*\(",
                ],
                target_patterns=[
                    r"(?:extern\s+)?(?:__declspec\([^)]+\)\s+)?[\w\s\*]+\s+(\w+)\s*\(",
                ],
                relationship_type=SemanticRelationType.FFI_CALL,
                source_languages={"rust"},
                target_languages={"c", "cpp"},
            ),
            # Java JNI patterns
            CrossLanguagePattern(
                name="java_jni",
                description="Java Native Interface calls",
                source_patterns=[
                    r"native\s+[\w\s<>\[\]]+\s+(\w+)\s*\(",
                    r'System\.loadLibrary\s*\(\s*["\']([^"\']+)["\']',
                ],
                target_patterns=[
                    r"JNIEXPORT\s+[\w\s\*]+\s+JNICALL\s+Java_[\w_]+_(\w+)",
                ],
                relationship_type=SemanticRelationType.FFI_CALL,
                source_languages={"java"},
                target_languages={"c", "cpp"},
            ),
            # Go CGO patterns
            CrossLanguagePattern(
                name="go_cgo",
                description="Go calling C code via CGO",
                source_patterns=[
                    r'\/\*\s*#include\s+[<"]([^>"]+)[>"]\s*\*\/',
                    r"C\.(\w+)\s*\(",
                ],
                target_patterns=[
                    r"(?:extern\s+)?[\w\s\*]+\s+(\w+)\s*\(",
                ],
                relationship_type=SemanticRelationType.FFI_CALL,
                source_languages={"go"},
                target_languages={"c", "cpp"},
            ),
            # REST API patterns
            CrossLanguagePattern(
                name="rest_api_calls",
                description="REST API endpoint calls",
                source_patterns=[
                    r'(?:fetch|axios|requests|http)[\w.]*\(\s*["\']([^"\']+)["\']',
                    r'@(?:GET|POST|PUT|DELETE|PATCH)\s*\(\s*["\']([^"\']+)["\']',
                    r'curl\s+[^"\']*["\']([^"\']+)["\']',
                ],
                target_patterns=[
                    r'@(?:RequestMapping|GetMapping|PostMapping|PutMapping|DeleteMapping)\s*\(\s*["\']([^"\']+)["\']',
                    r'@app\.(?:route|get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']',
                    r'router\.(?:get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']',
                ],
                relationship_type=SemanticRelationType.API_ENDPOINT,
            ),
            # Database access patterns
            CrossLanguagePattern(
                name="database_access",
                description="Database operations and queries",
                source_patterns=[
                    r"(?:SELECT|INSERT|UPDATE|DELETE)\s+.*FROM\s+(\w+)",
                    r"db\.(\w+)\.(?:find|insert|update|delete)",
                    r"@Entity\s*(?:\([^)]*\))?\s*(?:public\s+)?class\s+(\w+)",
                ],
                target_patterns=[
                    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
                    r'@Table\s*\(\s*name\s*=\s*["\']([^"\']+)["\']',
                ],
                relationship_type=SemanticRelationType.DATABASE_ACCESS,
            ),
            # Type conversion patterns
            CrossLanguagePattern(
                name="type_conversions",
                description="Type casting and conversions",
                source_patterns=[
                    r"(?:as|into|from)\s+(\w+)",
                    r"\((\w+)\)\s*\w+",  # C-style cast
                    r"static_cast<(\w+)>",  # C++ cast
                    r"\.to(\w+)\(\)",  # Conversion methods
                ],
                target_patterns=[
                    r"(?:struct|class|enum|type)\s+(\w+)",
                ],
                relationship_type=SemanticRelationType.TYPE_CONVERSION,
            ),
        ]

    def parse_entity_patterns(self, entity: CoreCodeEntity, patterns: list[str]) -> bool:
        """Check if an entity matches any of the given patterns.

        Args:
            entity: The code entity to check against patterns
            patterns: List of regex patterns to match against the entity

        Returns:
            bool: True if the entity matches any pattern, False otherwise

        Note:
            Patterns are matched against entity name, signature and content
            using case-insensitive comparison.
        """
        search_text = (
            f"{entity.name} {entity.signature or ''} {entity.properties.get('content', '')}"
        )

        for pattern in patterns:
            if re.search(pattern, search_text, re.IGNORECASE | re.MULTILINE):
                return True

        return False

    def extract_semantic_patterns(
        self,
        entities: list[CoreCodeEntity],
        pattern_types: list[SemanticRelationType] | None = None,
    ) -> dict[str, list[CoreCodeEntity]]:
        """Extract entities that match specific semantic patterns.

        Args:
            entities: List of code entities to analyze
            pattern_types: Optional list of pattern types to filter by

        Returns:
            Dictionary mapping pattern names to matching entities
        """
        pattern_matches = {}

        patterns_to_check = self.cross_language_patterns
        if pattern_types:
            patterns_to_check = [
                p for p in self.cross_language_patterns if p.relationship_type in pattern_types
            ]

        for pattern in patterns_to_check:
            matching_entities = []

            for entity in entities:
                # Check source patterns
                if self.parse_entity_patterns(entity, pattern.source_patterns):
                    matching_entities.append(entity)
                # Check target patterns
                elif self.parse_entity_patterns(entity, pattern.target_patterns):
                    matching_entities.append(entity)

            if matching_entities:
                pattern_matches[pattern.name] = matching_entities

        return pattern_matches

    def analyze_entity_relationships(
        self,
        source: CoreCodeEntity,
        target: CoreCodeEntity,
        pattern: CrossLanguagePattern,
    ) -> bool:
        """Determine if two entities are related based on the pattern context.

        Args:
            source: Source code entity in the relationship
            target: Target code entity in the relationship
            pattern: Cross-language pattern defining the relationship type

        Returns:
            bool: True if entities are related according to pattern rules

        Note:
            Checks both structural and semantic relationships between entities
            based on the pattern's source/target language constraints.
        """
        # Name-based matching (common for FFI and API calls)
        if source.name.lower() in target.name.lower() or target.name.lower() in source.name.lower():
            return True

        # Signature-based matching
        if source.signature and target.signature:
            source_sig = source.signature.lower()
            target_sig = target.signature.lower()

            # Extract function names from signatures
            source_func = re.search(r"(\w+)\s*\(", source_sig)
            target_func = re.search(r"(\w+)\s*\(", target_sig)

            if source_func and target_func:
                if source_func.group(1) == target_func.group(1):
                    return True

        # Context-based matching (same file paths, related modules)
        source_path = Path(source.file_path)
        target_path = Path(target.file_path)

        # Check if files are in related directories
        if source_path.parent == target_path.parent:
            return True

        # Check for naming conventions (e.g., same_name.rs and same_name.c)
        if source_path.stem == target_path.stem:
            return True

        return False

    def extract_architectural_patterns(
        self, entities: list[CoreCodeEntity]
    ) -> dict[str, list[CoreCodeEntity]]:
        """Extract architectural patterns from entities.

        Args:
            entities: List of code entities to analyze

        Returns:
            Dictionary mapping pattern types to matching entities
        """
        architectural_patterns = {}

        # Service patterns
        service_patterns = [
            r"@Service\s*(?:\([^)]*\))?\s*(?:public\s+)?class\s+(\w+)",
            r"@RestController\s*(?:\([^)]*\))?\s*(?:public\s+)?class\s+(\w+)",
            r"@Component\s*(?:\([^)]*\))?\s*(?:public\s+)?class\s+(\w+)",
            r"class\s+(\w+Service)\b",
            r"class\s+(\w+Controller)\b",
        ]

        # Event patterns
        event_emission_patterns = [
            r'emit\s*\(\s*["\']([^"\']+)["\']',
            r'publish\s*\(\s*["\']([^"\']+)["\']',
            r'trigger\s*\(\s*["\']([^"\']+)["\']',
            r'addEventListener\s*\(\s*["\']([^"\']+)["\']',
        ]

        event_consumption_patterns = [
            r'on\s*\(\s*["\']([^"\']+)["\']',
            r'addEventListener\s*\(\s*["\']([^"\']+)["\']',
            r'subscribe\s*\(\s*["\']([^"\']+)["\']',
            r'@EventListener\s*(?:\([^)]*["\']([^"\']+)["\'][^)]*\))?',
        ]

        # Configuration patterns
        config_patterns = [
            r'@Value\s*\(\s*["\']([^"\']+)["\']',
            r'getenv\s*\(\s*["\']([^"\']+)["\']',
            r'os\.environ\[[\'"\'](.*?)[\'"\']\]',
            r"process\.env\.(\w+)",
            r'config\.get\s*\(\s*["\']([^"\']+)["\']',
        ]

        # Categorize entities
        for entity in entities:
            search_text = (
                f"{entity.name} {entity.signature or ''} {entity.properties.get('content', '')}"
            )

            # Check service patterns
            for pattern in service_patterns:
                if re.search(pattern, search_text, re.IGNORECASE):
                    architectural_patterns.setdefault("services", []).append(entity)
                    break

            # Check event patterns
            for pattern in event_emission_patterns:
                if re.search(pattern, search_text, re.IGNORECASE):
                    architectural_patterns.setdefault("event_emitters", []).append(entity)
                    break

            for pattern in event_consumption_patterns:
                if re.search(pattern, search_text, re.IGNORECASE):
                    architectural_patterns.setdefault("event_consumers", []).append(entity)
                    break

            # Check config patterns
            for pattern in config_patterns:
                if re.search(pattern, search_text, re.IGNORECASE):
                    architectural_patterns.setdefault("config_users", []).append(entity)
                    break

        return architectural_patterns
