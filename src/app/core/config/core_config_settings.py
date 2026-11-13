"""Core configuration settings for coding-analysis.

Service URLs MUST come from gitops env (Deployment env block); source
defaults are empty placeholders. Missing required values fail loud at
startup in production. See: gitops-production/docs/config-externalization-policy.md.
"""

from __future__ import annotations

from functools import lru_cache
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    service_name: str = "coding-analysis"
    version: str = "1.0.0"
    environment: str = "production"
    debug: bool = False

    # Database — placeholders; real values come from gitops env.
    database_url: str = ""
    pgbouncer_url: str = ""

    # Graph-build analysis config (read by CodeStructureStorage)
    include_tests_in_graphs: bool = False
    ignore_patterns: list[str] = Field(default_factory=list)

    # Redis — placeholder; real value from gitops env.
    redis_url: str = ""

    # Auth
    enable_auth: bool = True
    # NB no ``default_tenant_id`` field. It existed with a ``"n8n-dev"`` default
    # and had zero readers, but ``model_config`` below is ``case_sensitive:
    # False``, so it WAS being populated from the ``DEFAULT_TENANT_ID`` env var
    # — a live-looking, unread tenant carrying the stale pre-uuid7 label of a
    # tenant that really exists. Read the tenant from
    # ``coding_client.auth.tenant.DEFAULT_TENANT_ID`` (``None`` when unset).

    # Telemetry — placeholder; real value from gitops env.
    otlp_traces_endpoint: str = ""

    model_config = {"env_prefix": "", "case_sensitive": False}

    _REQUIRED_IN_PRODUCTION: ClassVar[tuple[str, ...]] = (
        "database_url",
        "pgbouncer_url",
        "redis_url",
        "otlp_traces_endpoint",
    )

    def model_post_init(self, __context: object) -> None:
        if self.environment != "production":
            return
        missing = [n for n in self._REQUIRED_IN_PRODUCTION if not getattr(self, n)]
        if missing:
            raise RuntimeError(
                f"coding-analysis core settings missing required fields in production: {missing}. "
                "Set them via env vars on the Deployment (see gitops-production)."
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
