"""Contract scanner orchestrator -- wires parsers and checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .checks.query_check import check_query_params
from .checks.request_check import check_request_fields
from .checks.response_check import check_response_fields
from .checks.route_check import check_routes
from .checks.status_check import check_status_codes
from .checks.syntax_check import check_syntax
from .checks.type_check import check_types
from .checks.types import Violation
from .enrichment.enricher import enrich_violations
from .parsers.client_parser import parse_client
from .parsers.route_parser import parse_routes
from .parsers.schema_parser import parse_schemas
from .parsers.ts_route_parser import parse_ts_routes
from .parsers.types import ServiceRoutes

logger = logging.getLogger(__name__)


@dataclass
class ContractResult:
    name: str
    passed: bool = True
    violations: list[Violation] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    parse_error: str | None = None


class ContractScanner:
    def scan_sources(
        self,
        name: str,
        client_source: str,
        schema_sources: list[str],
        route_sources: list[str],
        *,
        service_sources: dict[str, str] | None = None,
        route_prefix: str | list[str] = "",
        ts_route_sources: dict[str, str] | None = None,
        ts_routes: ServiceRoutes | None = None,
        check_syntax_enabled: bool = True,
        check_routes_enabled: bool = True,
        check_methods_enabled: bool = True,
        check_request_fields_enabled: bool = True,
        check_response_fields_enabled: bool = True,
        check_types_enabled: bool = True,
        check_query_params_enabled: bool = True,
        check_status_codes_enabled: bool = True,
    ) -> ContractResult:
        violations: list[Violation] = []

        # Syntax check FIRST -- catches files that can't import at runtime
        syntax_errors = 0
        if check_syntax_enabled:
            if service_sources is not None:
                syntax_violations = check_syntax(service_sources)
            else:
                # Fall back to checking sources we already have
                fallback_sources: dict[str, str] = {"client.py": client_source}
                for i, s in enumerate(schema_sources):
                    fallback_sources[f"schema_{i}.py"] = s
                for i, r in enumerate(route_sources):
                    fallback_sources[f"route_{i}.py"] = r
                syntax_violations = check_syntax(fallback_sources)
            syntax_errors = len(syntax_violations)
            violations.extend(syntax_violations)

        client = parse_client(client_source)
        combined_source = "\n\n".join(schema_sources)
        combined_schema = parse_schemas(combined_source)
        combined_routes = ServiceRoutes()
        for i, src in enumerate(route_sources):
            pfx = route_prefix[i] if isinstance(route_prefix, list) else route_prefix
            parsed = parse_routes(src, prefix=pfx)
            combined_routes.routes.extend(parsed.routes)
            combined_routes.parse_errors.extend(parsed.parse_errors)
        if ts_routes is None and ts_route_sources:
            # TypeScript/JavaScript providers (fr's Router.add, Express, Fastify, Hono,
            # pathname dispatch). Parsed together so cross-file mounts resolve.
            ts_routes = parse_ts_routes(ts_route_sources).as_service_routes()
        if ts_routes is not None:  # or pre-parsed: the API does it in an isolated process
            combined_routes.routes.extend(ts_routes.routes)
            combined_routes.parse_errors.extend(ts_routes.parse_errors)

        if check_routes_enabled or check_methods_enabled:
            violations.extend(check_routes(client.calls, combined_routes.routes))
        if check_request_fields_enabled:
            violations.extend(
                check_request_fields(client.calls, combined_routes.routes, combined_schema)
            )
        if check_response_fields_enabled:
            violations.extend(
                check_response_fields(client.calls, combined_routes.routes, combined_schema)
            )
        if check_types_enabled:
            violations.extend(check_types(client.calls, combined_schema))
        if check_query_params_enabled:
            violations.extend(check_query_params(client.calls, combined_routes.routes))
        if check_status_codes_enabled:
            violations.extend(check_status_codes(client.calls, combined_routes.routes))

        enrich_violations(violations, combined_routes.routes)

        fail_count = sum(1 for v in violations if v.severity == "FAIL")
        return ContractResult(
            name=name,
            passed=fail_count == 0,
            violations=violations,
            stats={
                "client_calls": len(client.calls),
                "service_routes": len(combined_routes.routes),
                "schema_models": len(combined_schema.models),
                "violations": len(violations),
                "failures": fail_count,
                "syntax_errors": syntax_errors,
            },
        )
