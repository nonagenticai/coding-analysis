"""Service configuration via Pydantic Settings.

Service URLs MUST come from gitops env (Deployment env block);
the source defaults are intentionally empty placeholders. Missing values
fail loud at startup in production via ``_REQUIRED_IN_PRODUCTION``.
See: gitops-production/docs/config-externalization-policy.md.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """coding-analysis service configuration."""

    model_config = {"env_prefix": "", "case_sensitive": False}

    # ── Service identity ──────────────────────────────────────────────
    SERVICE_NAME: str = "coding-analysis"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    PORT: int = 8688

    # ── Database ──────────────────────────────────────────────────────
    DATABASE_URL: str | None = None
    PGBOUNCER_URL: str | None = None

    # ── Redis ─────────────────────────────────────────────────────────
    # Placeholder; real value comes from the Deployment env block in
    # gitops-production. Required in production (see _REQUIRED_IN_PRODUCTION).
    REDIS_URL: str = ""

    # ── Authentication & authorisation ────────────────────────────────
    ENABLE_AUTH: bool = True
    # NB there is deliberately no ``DEFAULT_TENANT_ID`` field here. It existed
    # with a ``"n8n-dev"`` default and had ZERO readers — ``auth/tenant.py``
    # imports the name from ``coding_client.auth.tenant``, where it is ``None``
    # when unset. A dead field is not inert: ``"n8n-dev"`` is the stale
    # pre-uuid7 label of a tenant that really exists, so the next person to
    # write ``settings.DEFAULT_TENANT_ID`` would have silently got a tenant
    # nobody chose rather than a failure. Read it from the SDK.
    LICENSE_INTROSPECT_URL: str = ""

    # ── API docs ──────────────────────────────────────────────────────
    ENABLE_DOCS: bool = False

    # ── Logging ───────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # "json" or "text"

    # ── CORS ──────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["*"])

    # ── OpenTelemetry (OTLP) ──────────────────────────────────────────
    OTLP_ENDPOINT: str | None = None
    OTLP_INSECURE: bool = True
    OTLP_SERVICE_NAME: str = "coding-analysis"

    # ── Inter-service URLs ────────────────────────────────────────────
    # All inter-service URLs are placeholders; real values come from the
    # Deployment env block in gitops-production. Required in production.
    CODING_CONTAINERS_URL: str = ""
    # Optional (best-effort): used only to persist dependency-report snapshots at
    # analysis time (see services/code_graph/dependency_snapshot.py). If unset the
    # snapshot write no-ops with a warning — it is NOT required for core analysis,
    # so it is deliberately kept out of _REQUIRED_IN_PRODUCTION to avoid coupling
    # the coding-analysis rollout to a gitops env change.
    CODING_SECURITY_URL: str = ""
    QUALITY_ANALYSIS_SERVICE_URL: str = ""
    PYTHON_TEST_SERVICE_URL: str = ""
    JAVASCRIPT_TEST_SERVICE_URL: str = ""
    JAVA_TEST_SERVICE_URL: str = ""
    RUST_TEST_SERVICE_URL: str = ""
    CSHARP_TEST_SERVICE_URL: str = ""
    GO_TEST_SERVICE_URL: str = ""

    # ── Bug hunt completion callback ──────────────────────────────────
    # Comma-separated URL prefixes a queued hunt's ``callback_url`` must start
    # with (e.g. an n8n Wait-node resume base ``http://host:5678/webhook-waiting/``).
    # EMPTY = every callback_url is refused (fail closed): the worker POSTs to
    # this URL from inside the cluster, so an open value would be an SSRF
    # primitive. Optional, so deliberately not in _REQUIRED_IN_PRODUCTION.
    BUG_HUNT_CALLBACK_URL_PREFIXES: str = ""

    # ── Contract extraction isolation ─────────────────────────────────
    # scan_call_sites / scan_routes / scan_contracts' TypeScript routes parse
    # tenant source. Each request's extraction runs in its own worker process
    # (processing/contract_analysis/isolation.py), SIGKILLed at this deadline so a
    # pathological input can never stall the single uvicorn process. call-sites
    # returns what finished (truncation_reason "deadline"); routes returns 422.
    CONTRACT_EXTRACTION_DEADLINE_SECONDS: float = Field(default=60.0, gt=0)
    # RLIMIT_AS of each worker process in MiB (0 = uncapped). The pod limit is
    # 2Gi; MAX_WORKERS x this must leave room for the API process itself.
    CONTRACT_EXTRACTION_MEMORY_MB: int = Field(default=768, ge=0)
    # Concurrent extraction workers; further requests wait (see QUEUE_WAIT).
    CONTRACT_EXTRACTION_MAX_WORKERS: int = Field(default=2, ge=1)
    # Longest a job waits for a free worker before the request fails busy
    # (HTTP 503 + Retry-After). Waiting counts against the deadline.
    CONTRACT_EXTRACTION_QUEUE_WAIT_SECONDS: float = Field(default=10.0, ge=0)
    # A job is never started with less than this much of its deadline left:
    # it fails busy instead of starting only to be killed with nothing done.
    CONTRACT_EXTRACTION_MIN_RUN_SECONDS: float = Field(default=5.0, ge=0)
    # Extraction requests in flight (running + waiting) before new ones are
    # refused busy at once, before their files are loaded. Each holds its
    # request body (up to 20 MiB inline, ~50 MB parsed) while it waits.
    CONTRACT_EXTRACTION_MAX_PENDING: int = Field(default=4, ge=1)
    # Running jobs per tenant, so one tenant cannot hold every worker.
    CONTRACT_EXTRACTION_MAX_PER_TENANT: int = Field(default=1, ge=1)
    # False = extract in a thread of this process (tests only; no deadline).
    CONTRACT_EXTRACTION_ISOLATED: bool = True

    _REQUIRED_IN_PRODUCTION: ClassVar[tuple[str, ...]] = (
        "REDIS_URL",
        "LICENSE_INTROSPECT_URL",
        "CODING_CONTAINERS_URL",
        "QUALITY_ANALYSIS_SERVICE_URL",
        "PYTHON_TEST_SERVICE_URL",
        "JAVASCRIPT_TEST_SERVICE_URL",
        "JAVA_TEST_SERVICE_URL",
        "RUST_TEST_SERVICE_URL",
        "CSHARP_TEST_SERVICE_URL",
        "GO_TEST_SERVICE_URL",
    )

    @property
    def bug_hunt_callback_prefixes(self) -> tuple[str, ...]:
        """The allowed callback URL prefixes; empty means callbacks are off."""
        return tuple(p.strip() for p in self.BUG_HUNT_CALLBACK_URL_PREFIXES.split(",") if p.strip())

    @property
    def docs_enabled(self) -> bool:
        """Docs are on in dev or when explicitly enabled."""
        return self.ENVIRONMENT == "development" or self.ENABLE_DOCS

    def model_post_init(self, __context: object) -> None:
        if self.ENVIRONMENT != "production":
            return
        missing = [n for n in self._REQUIRED_IN_PRODUCTION if not getattr(self, n)]
        if missing:
            raise RuntimeError(
                f"coding-analysis settings missing required fields in production: {missing}. "
                "Set them via env vars on the Deployment (see gitops-production)."
            )


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return _settings


_settings = Settings()
