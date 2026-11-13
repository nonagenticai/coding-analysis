"""
Semantic Rule Validation Module

This module provides functionality for validating semantic rules and patterns
in code, including architectural patterns, service boundaries, and data flow rules.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from models.core_models import (
    CoreCodeEntity,
    EntityRelation,
    EntityType,
)

from .semantic_parser import SemanticContext, SemanticRelationType

logger = logging.getLogger(__name__)


class RuleViolationType(StrEnum):
    """Types of semantic rule violations."""

    ARCHITECTURAL_BOUNDARY = "architectural_boundary"
    DATA_FLOW_VIOLATION = "data_flow_violation"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    LAYER_VIOLATION = "layer_violation"
    SERVICE_BOUNDARY = "service_boundary"
    SECURITY_VIOLATION = "security_violation"
    CONCURRENCY_ISSUE = "concurrency_issue"
    RESOURCE_LEAK = "resource_leak"


class RuleSeverity(StrEnum):
    """Severity levels for rule violations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SemanticRule:
    """Represents a semantic validation rule."""

    name: str
    description: str
    violation_type: RuleViolationType
    severity: RuleSeverity
    patterns: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)

    # Rule configuration
    enabled: bool = True
    languages: set[str] | None = None
    entity_types: set[EntityType] | None = None

    # Validation function (if needed)
    validator: str | None = None


@dataclass
class RuleViolation:
    """Represents a violation of a semantic rule."""

    rule_name: str
    violation_type: RuleViolationType
    severity: RuleSeverity
    message: str
    entity_id: str
    file_path: str
    line_number: int | None = None

    # Related entities
    related_entities: list[str] = field(default_factory=list)

    # Evidence
    evidence: dict[str, Any] = field(default_factory=dict)


class SemanticRuleValidator:
    """Validates semantic rules and detects violations."""

    def __init__(self):
        self.rules = self._initialize_rules()
        self.violations: list[RuleViolation] = []

    def _initialize_rules(self) -> list[SemanticRule]:
        """Initialize default semantic rules."""
        return [
            # Architectural boundary rules
            SemanticRule(
                name="no_ui_to_data_access",
                description="UI components should not directly access data access layer",
                violation_type=RuleViolationType.ARCHITECTURAL_BOUNDARY,
                severity=RuleSeverity.HIGH,
                anti_patterns=[
                    r"Controller.*Repository",
                    r"View.*DAO",
                    r"Component.*Database",
                ],
            ),
            # Layer violation rules
            SemanticRule(
                name="no_skip_service_layer",
                description="Controllers should not bypass service layer",
                violation_type=RuleViolationType.LAYER_VIOLATION,
                severity=RuleSeverity.MEDIUM,
                anti_patterns=[
                    r"Controller.*Repository.*(?!Service)",
                    r"Controller.*DAO.*(?!Service)",
                ],
            ),
            # Service boundary rules
            SemanticRule(
                name="no_cross_service_database_access",
                description="Services should not access other services' databases",
                violation_type=RuleViolationType.SERVICE_BOUNDARY,
                severity=RuleSeverity.CRITICAL,
                anti_patterns=[
                    r"UserService.*OrderDatabase",
                    r"PaymentService.*UserRepository",
                ],
            ),
            # Data flow rules
            SemanticRule(
                name="no_circular_data_dependency",
                description="Entities should not have circular data dependencies",
                violation_type=RuleViolationType.CIRCULAR_DEPENDENCY,
                severity=RuleSeverity.HIGH,
                validator="check_circular_dependencies",
            ),
            # Security rules
            SemanticRule(
                name="no_sensitive_data_logging",
                description="Sensitive data should not be logged",
                violation_type=RuleViolationType.SECURITY_VIOLATION,
                severity=RuleSeverity.CRITICAL,
                anti_patterns=[
                    r"log.*password",
                    r"log.*secret",
                    r"log.*token",
                    r"console\.log.*password",
                ],
            ),
            # Concurrency rules
            SemanticRule(
                name="no_shared_mutable_state_without_sync",
                description="Shared mutable state should be properly synchronized",
                violation_type=RuleViolationType.CONCURRENCY_ISSUE,
                severity=RuleSeverity.HIGH,
                anti_patterns=[
                    r"static.*(?!final).*=.*(?!synchronized)",
                    r"global.*=.*(?!lock)",
                ],
            ),
            # Resource management rules
            SemanticRule(
                name="proper_resource_cleanup",
                description="Resources should be properly closed/disposed",
                violation_type=RuleViolationType.RESOURCE_LEAK,
                severity=RuleSeverity.MEDIUM,
                patterns=[
                    r"try.*finally.*close",
                    r"with.*:",
                    r"using.*\{",
                    r"defer.*close",
                ],
                anti_patterns=[
                    r"new\s+File.*(?!close)",
                    r"open\(.*(?!close)",
                ],
            ),
        ]

    def validate_entities(
        self,
        entities: list[CoreCodeEntity],
        relationships: list[EntityRelation],
        contexts: dict[str, SemanticContext],
    ) -> list[RuleViolation]:
        """Validate entities against semantic rules.

        Args:
            entities: List of code entities to validate
            relationships: List of relationships between entities
            contexts: Language-specific semantic contexts

        Returns:
            List of rule violations found
        """
        violations = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            rule_violations = self._validate_rule(rule, entities, relationships, contexts)
            violations.extend(rule_violations)

        self.violations = violations
        return violations

    def _validate_rule(
        self,
        rule: SemanticRule,
        entities: list[CoreCodeEntity],
        relationships: list[EntityRelation],
        contexts: dict[str, SemanticContext],
    ) -> list[RuleViolation]:
        """Validate a specific rule against entities.

        Args:
            rule: Rule to validate
            entities: List of entities to check
            relationships: List of relationships
            contexts: Semantic contexts

        Returns:
            List of violations for this rule
        """
        violations = []

        # Use custom validator if specified
        if rule.validator:
            validator_method = getattr(self, rule.validator, None)
            if validator_method:
                custom_violations = validator_method(rule, entities, relationships, contexts)
                violations.extend(custom_violations)
                return violations

        # Pattern-based validation
        for entity in entities:
            # Filter by entity type if specified
            if rule.entity_types and entity.type not in rule.entity_types:
                continue

            # Filter by language if specified
            if rule.languages and entity.language not in rule.languages:
                continue

            entity_violations = self._check_entity_patterns(rule, entity, entities, relationships)
            violations.extend(entity_violations)

        return violations

    def _check_entity_patterns(
        self,
        rule: SemanticRule,
        entity: CoreCodeEntity,
        entities: list[CoreCodeEntity],
        relationships: list[EntityRelation],
    ) -> list[RuleViolation]:
        """Check an entity against rule patterns.

        Args:
            rule: Rule to check
            entity: Entity to validate
            entities: All entities for context
            relationships: All relationships for context

        Returns:
            List of violations found
        """
        violations = []
        search_text = (
            f"{entity.name} {entity.signature or ''} {entity.properties.get('content', '')}"
        )

        # Check anti-patterns (violations)
        for anti_pattern in rule.anti_patterns:
            if re.search(anti_pattern, search_text, re.IGNORECASE):
                violations.append(
                    RuleViolation(
                        rule_name=rule.name,
                        violation_type=rule.violation_type,
                        severity=rule.severity,
                        message=f"Anti-pattern detected: {anti_pattern}",
                        entity_id=entity.id,
                        file_path=entity.file_path,
                        evidence={"pattern": anti_pattern, "matched_text": search_text},
                    )
                )

        # Check if required patterns are missing
        if rule.patterns:
            has_required_pattern = False
            for pattern in rule.patterns:
                if re.search(pattern, search_text, re.IGNORECASE):
                    has_required_pattern = True
                    break

            if not has_required_pattern:
                violations.append(
                    RuleViolation(
                        rule_name=rule.name,
                        violation_type=rule.violation_type,
                        severity=rule.severity,
                        message=f"Required pattern not found: {rule.patterns}",
                        entity_id=entity.id,
                        file_path=entity.file_path,
                        evidence={"required_patterns": rule.patterns},
                    )
                )

        return violations

    def check_circular_dependencies(
        self,
        rule: SemanticRule,
        entities: list[CoreCodeEntity],
        relationships: list[EntityRelation],
        contexts: dict[str, SemanticContext],
    ) -> list[RuleViolation]:
        """Check for circular dependencies between entities.

        Args:
            rule: Rule being validated
            entities: List of entities
            relationships: List of relationships
            contexts: Semantic contexts

        Returns:
            List of circular dependency violations
        """
        violations = []

        # Build dependency graph
        dependency_graph = defaultdict(set)
        for rel in relationships:
            if rel.type in [
                SemanticRelationType.DATA_DEPENDENCY.value,
                SemanticRelationType.SERVICE_CALL.value,
                "imports",
                "calls",
            ]:
                dependency_graph[rel.source_id].add(rel.target_id)

        # Find cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node, path):
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                return cycle

            if node in visited:
                return None

            visited.add(node)
            rec_stack.add(node)

            for neighbor in dependency_graph.get(node, set()):
                cycle = has_cycle(neighbor, path + [node])
                if cycle:
                    return cycle

            rec_stack.remove(node)
            return None

        # Check each entity for cycles
        entity_map = {e.id: e for e in entities}
        for entity in entities:
            if entity.id not in visited:
                cycle = has_cycle(entity.id, [])
                if cycle:
                    # Create violation for the cycle
                    cycle_entities = [entity_map.get(eid) for eid in cycle if eid in entity_map]
                    cycle_names = [e.name for e in cycle_entities if e]

                    violations.append(
                        RuleViolation(
                            rule_name=rule.name,
                            violation_type=rule.violation_type,
                            severity=rule.severity,
                            message=f"Circular dependency detected: {' -> '.join(cycle_names)}",
                            entity_id=entity.id,
                            file_path=entity.file_path,
                            related_entities=cycle,
                            evidence={"cycle": cycle, "cycle_names": cycle_names},
                        )
                    )

        return violations

    def add_custom_rule(self, rule: SemanticRule):
        """Add a custom validation rule.

        Args:
            rule: Custom rule to add
        """
        self.rules.append(rule)

    def disable_rule(self, rule_name: str):
        """Disable a validation rule.

        Args:
            rule_name: Name of the rule to disable
        """
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
                break

    def get_violations_by_severity(self, severity: RuleSeverity) -> list[RuleViolation]:
        """Get violations filtered by severity.

        Args:
            severity: Severity level to filter by

        Returns:
            List of violations with the specified severity
        """
        return [v for v in self.violations if v.severity == severity]

    def get_violations_by_type(self, violation_type: RuleViolationType) -> list[RuleViolation]:
        """Get violations filtered by type.

        Args:
            violation_type: Type of violations to filter by

        Returns:
            List of violations of the specified type
        """
        return [v for v in self.violations if v.violation_type == violation_type]

    def get_violation_summary(self) -> dict[str, int]:
        """Get a summary of violations by severity.

        Returns:
            Dictionary mapping severity levels to violation counts
        """
        summary = defaultdict(int)
        for violation in self.violations:
            summary[violation.severity.value] += 1
        return dict(summary)


class ArchitecturalPatternAnalyzer:
    """Analyzes architectural patterns and service relationships."""

    def __init__(self):
        self.service_boundaries = {}
        self.layer_definitions = {}

    def analyze_service_patterns(
        self, entities: list[CoreCodeEntity], contexts: dict[str, SemanticContext]
    ) -> list[EntityRelation]:
        """Analyze service patterns and boundaries.

        Args:
            entities: List of entities to analyze
            contexts: Semantic contexts

        Returns:
            List of service-related relationships
        """
        relationships = []

        # Detect service definitions
        services = self._detect_services(entities)

        # Detect service clients
        service_clients = self._detect_service_clients(entities, services)

        # Create service call relationships
        for client_entity, service_entity in service_clients:
            relationships.append(
                EntityRelation(
                    source_id=client_entity.id,
                    target_id=service_entity.id,
                    type=SemanticRelationType.SERVICE_CALL.value,
                    properties={
                        "pattern": "service_call",
                        "service_name": service_entity.name,
                        "client_name": client_entity.name,
                    },
                )
            )

        return relationships

    def _detect_services(self, entities: list[CoreCodeEntity]) -> list[CoreCodeEntity]:
        """Detect service entities.

        Args:
            entities: List of entities to search

        Returns:
            List of entities that are services
        """
        service_patterns = [
            r"@Service\b",
            r"@RestController\b",
            r"@Component\b",
            r"Service$",
            r"Controller$",
            r"Handler$",
        ]

        services = []
        for entity in entities:
            if entity.type == EntityType.CLASS:
                search_text = f"{entity.name} {entity.signature or ''}"

                for pattern in service_patterns:
                    if re.search(pattern, search_text, re.IGNORECASE):
                        services.append(entity)
                        break

        return services

    def _detect_service_clients(
        self, entities: list[CoreCodeEntity], services: list[CoreCodeEntity]
    ) -> list[tuple]:
        """Detect entities that call services.

        Args:
            entities: All entities
            services: List of service entities

        Returns:
            List of (client_entity, service_entity) tuples
        """
        service_clients = []

        client_patterns = [
            r"@Autowired\s+(\w*Service\w*)",
            r"@Inject\s+(\w*Service\w*)",
            r"private\s+(\w*Service\w*)",
            r"\.(\w*Service\w*)\s*\(",
        ]

        for entity in entities:
            if entity.type in [EntityType.CLASS, EntityType.FUNCTION]:
                search_text = f"{entity.signature or ''} {entity.properties.get('content', '')}"

                for pattern in client_patterns:
                    matches = re.finditer(pattern, search_text, re.IGNORECASE)
                    for match in matches:
                        service_name = match.group(1)

                        # Find matching service
                        for service in services:
                            if service_name.lower() in service.name.lower():
                                service_clients.append((entity, service))
                                break

        return service_clients

    def analyze_configuration_dependencies(
        self, entities: list[CoreCodeEntity]
    ) -> list[EntityRelation]:
        """Analyze configuration dependencies.

        Args:
            entities: List of entities to analyze

        Returns:
            List of configuration dependency relationships
        """
        relationships = []

        config_patterns = [
            r'@Value\s*\(\s*["\']([^"\']+)["\']',
            r'getenv\s*\(\s*["\']([^"\']+)["\']',
            r'os\.environ\[[\'"\'](.*?)[\'"\']\]',
            r"process\.env\.(\w+)",
            r'config\.get\s*\(\s*["\']([^"\']+)["\']',
        ]

        for entity in entities:
            search_text = f"{entity.signature or ''} {entity.properties.get('content', '')}"

            for pattern in config_patterns:
                matches = re.finditer(pattern, search_text, re.IGNORECASE)
                for match in matches:
                    config_key = match.group(1)

                    # Create virtual configuration entity
                    config_entity_id = f"config:{config_key}"

                    relationships.append(
                        EntityRelation(
                            source_id=entity.id,
                            target_id=config_entity_id,
                            type=SemanticRelationType.CONFIGURATION_USE.value,
                            properties={
                                "config_key": config_key,
                                "pattern": pattern,
                            },
                        )
                    )

        return relationships

    def analyze_event_patterns(self, entities: list[CoreCodeEntity]) -> list[EntityRelation]:
        """Analyze event emission and consumption patterns.

        Args:
            entities: List of entities to analyze

        Returns:
            List of event-related relationships
        """
        relationships = []

        # Find event emitters and consumers
        emitters = defaultdict(list)
        consumers = defaultdict(list)

        emission_patterns = [
            r'emit\s*\(\s*["\']([^"\']+)["\']',
            r'publish\s*\(\s*["\']([^"\']+)["\']',
            r'trigger\s*\(\s*["\']([^"\']+)["\']',
        ]

        consumption_patterns = [
            r'on\s*\(\s*["\']([^"\']+)["\']',
            r'addEventListener\s*\(\s*["\']([^"\']+)["\']',
            r'subscribe\s*\(\s*["\']([^"\']+)["\']',
            r'@EventListener\s*(?:\([^)]*["\']([^"\']+)["\'][^)]*\))?',
        ]

        for entity in entities:
            search_text = f"{entity.signature or ''} {entity.properties.get('content', '')}"

            # Check for event emission
            for pattern in emission_patterns:
                matches = re.finditer(pattern, search_text, re.IGNORECASE)
                for match in matches:
                    event_name = match.group(1)
                    emitters[event_name].append(entity)

            # Check for event consumption
            for pattern in consumption_patterns:
                matches = re.finditer(pattern, search_text, re.IGNORECASE)
                for match in matches:
                    event_name = match.group(1) if match.group(1) else "unknown"
                    consumers[event_name].append(entity)

        # Create relationships between emitters and consumers
        for event_name in set(emitters.keys()) & set(consumers.keys()):
            for emitter in emitters[event_name]:
                for consumer in consumers[event_name]:
                    if emitter.id != consumer.id:
                        relationships.append(
                            EntityRelation(
                                source_id=emitter.id,
                                target_id=consumer.id,
                                type=SemanticRelationType.EVENT_EMISSION.value,
                                properties={
                                    "event_name": event_name,
                                    "pattern": "event_driven",
                                },
                            )
                        )

        return relationships
