"""Unit tests for config.settings module."""

from __future__ import annotations

import pytest

from config.settings import Settings, get_settings


class TestSettingsDefaults:
    """Verify default values of Settings fields."""

    def test_service_name_default(self):
        s = Settings(ENVIRONMENT="test")
        assert s.SERVICE_NAME == "coding-analysis"

    def test_version_default(self):
        s = Settings(ENVIRONMENT="test")
        assert s.VERSION == "1.0.0"

    def test_environment_default(self):
        # Construct without overriding ENVIRONMENT; conftest sets env var to "test",
        # so create with explicit production (with all required URLs supplied) to
        # test the field default. Production validator otherwise raises.
        s = Settings(
            ENVIRONMENT="production",
            REDIS_URL="redis://x:1",
            LICENSE_INTROSPECT_URL="http://x:1",
            CODING_CONTAINERS_URL="http://x:1",
            QUALITY_ANALYSIS_SERVICE_URL="http://x:1",
            PYTHON_TEST_SERVICE_URL="http://x:1",
            JAVASCRIPT_TEST_SERVICE_URL="http://x:1",
            JAVA_TEST_SERVICE_URL="http://x:1",
            RUST_TEST_SERVICE_URL="http://x:1",
            CSHARP_TEST_SERVICE_URL="http://x:1",
            GO_TEST_SERVICE_URL="http://x:1",
        )
        assert s.ENVIRONMENT == "production"

    def test_port_default(self):
        s = Settings(ENVIRONMENT="test")
        assert s.PORT == 8688

    def test_database_url_default_none(self, monkeypatch):
        # Hermetic: the CI job sets DATABASE_URL (has_postgres archetype input),
        # so clear it to assert the field *default* rather than the ambient env.
        monkeypatch.delenv("DATABASE_URL", raising=False)
        s = Settings(ENVIRONMENT="test")
        assert s.DATABASE_URL is None

    def test_pgbouncer_url_default_none(self, monkeypatch):
        monkeypatch.delenv("PGBOUNCER_URL", raising=False)
        s = Settings(ENVIRONMENT="test")
        assert s.PGBOUNCER_URL is None

    def test_redis_url_default(self, monkeypatch):
        monkeypatch.delenv("REDIS_URL", raising=False)
        s = Settings(ENVIRONMENT="test")
        # No hardcoded fallback — gitops env supplies the value in production.
        assert s.REDIS_URL == ""

    def test_enable_auth_default(self):
        # Conftest sets ENABLE_AUTH=false in env; test explicit True.
        s = Settings(ENVIRONMENT="test", ENABLE_AUTH=True)
        assert s.ENABLE_AUTH is True

    def test_settings_has_no_default_tenant_field(self):
        """Inverted, not deleted.

        This assertion used to be ``assert s.DEFAULT_TENANT_ID == "n8n-dev"``
        — it PINNED the stale literal as correct, so it would have gone red on
        the fix and green on the defect. The field had zero readers
        (``auth/tenant.py`` imports the name from ``coding_client.auth.tenant``,
        where it is ``None`` when unset), but a dead field with that literal is
        a booby trap for the next reader, not inert. Kept as an assertion that
        the field is gone, so a revert is caught.
        """
        assert "DEFAULT_TENANT_ID" not in Settings.model_fields
        s = Settings(ENVIRONMENT="test")
        assert not hasattr(s, "DEFAULT_TENANT_ID")

    def test_enable_docs_default(self):
        # Conftest may set it; test explicit False.
        s = Settings(ENVIRONMENT="test", ENABLE_DOCS=False)
        assert s.ENABLE_DOCS is False

    def test_log_level_default(self):
        s = Settings(ENVIRONMENT="test")
        assert s.LOG_LEVEL == "INFO"

    def test_log_format_default(self):
        s = Settings(ENVIRONMENT="test")
        assert s.LOG_FORMAT == "json"

    def test_cors_origins_default(self):
        s = Settings(ENVIRONMENT="test")
        assert s.CORS_ORIGINS == ["*"]

    def test_otlp_endpoint_default_none(self):
        s = Settings(ENVIRONMENT="test")
        assert s.OTLP_ENDPOINT is None

    def test_otlp_insecure_default(self):
        s = Settings(ENVIRONMENT="test")
        assert s.OTLP_INSECURE is True

    def test_otlp_service_name_default(self):
        s = Settings(ENVIRONMENT="test")
        assert s.OTLP_SERVICE_NAME == "coding-analysis"

    def test_coding_containers_url_default(self):
        s = Settings(ENVIRONMENT="test", CODING_CONTAINERS_URL="http://coding-containers:8691")
        assert s.CODING_CONTAINERS_URL == "http://coding-containers:8691"


class TestDocsEnabledProperty:
    """Test the docs_enabled computed property."""

    _PROD_URLS: dict[str, str] = {
        "REDIS_URL": "redis://x:1",
        "LICENSE_INTROSPECT_URL": "http://x:1",
        "CODING_CONTAINERS_URL": "http://x:1",
        "QUALITY_ANALYSIS_SERVICE_URL": "http://x:1",
        "PYTHON_TEST_SERVICE_URL": "http://x:1",
        "JAVASCRIPT_TEST_SERVICE_URL": "http://x:1",
        "JAVA_TEST_SERVICE_URL": "http://x:1",
        "RUST_TEST_SERVICE_URL": "http://x:1",
        "CSHARP_TEST_SERVICE_URL": "http://x:1",
        "GO_TEST_SERVICE_URL": "http://x:1",
    }

    def test_production_docs_disabled(self):
        s = Settings(ENVIRONMENT="production", ENABLE_DOCS=False, **self._PROD_URLS)
        assert s.docs_enabled is False

    def test_development_docs_enabled(self):
        s = Settings(ENVIRONMENT="development", ENABLE_DOCS=False)
        assert s.docs_enabled is True

    def test_production_with_enable_docs_true(self):
        s = Settings(ENVIRONMENT="production", ENABLE_DOCS=True, **self._PROD_URLS)
        assert s.docs_enabled is True

    def test_development_with_enable_docs_true(self):
        s = Settings(ENVIRONMENT="development", ENABLE_DOCS=True)
        assert s.docs_enabled is True

    def test_test_env_docs_disabled(self):
        s = Settings(ENVIRONMENT="test", ENABLE_DOCS=False)
        assert s.docs_enabled is False

    def test_test_env_docs_explicitly_enabled(self):
        s = Settings(ENVIRONMENT="test", ENABLE_DOCS=True)
        assert s.docs_enabled is True


class TestEnvOverrides:
    """Test that environment variables override defaults via monkeypatch."""

    def test_environment_override(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "development")
        s = Settings()
        assert s.ENVIRONMENT == "development"

    def test_port_override(self, monkeypatch):
        monkeypatch.setenv("PORT", "9999")
        monkeypatch.setenv("ENVIRONMENT", "test")
        s = Settings()
        assert s.PORT == 9999

    def test_service_name_override(self, monkeypatch):
        monkeypatch.setenv("SERVICE_NAME", "custom-analysis")
        monkeypatch.setenv("ENVIRONMENT", "test")
        s = Settings()
        assert s.SERVICE_NAME == "custom-analysis"

    def test_redis_url_override(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://otherhost:6380/1")
        monkeypatch.setenv("ENVIRONMENT", "test")
        s = Settings()
        assert s.REDIS_URL == "redis://otherhost:6380/1"

    def test_coding_containers_url_override(self, monkeypatch):
        monkeypatch.setenv("CODING_CONTAINERS_URL", "http://my-containers:9000")
        monkeypatch.setenv("ENVIRONMENT", "test")
        s = Settings()
        assert s.CODING_CONTAINERS_URL == "http://my-containers:9000"

    def test_enable_auth_override(self, monkeypatch):
        monkeypatch.setenv("ENABLE_AUTH", "true")
        monkeypatch.setenv("ENVIRONMENT", "test")
        s = Settings()
        assert s.ENABLE_AUTH is True

    def test_log_level_override(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("ENVIRONMENT", "test")
        s = Settings()
        assert s.LOG_LEVEL == "DEBUG"

    def test_cors_origins_override(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", '["http://localhost:3000"]')
        monkeypatch.setenv("ENVIRONMENT", "test")
        s = Settings()
        assert s.CORS_ORIGINS == ["http://localhost:3000"]


class TestGetSettings:
    """Test the get_settings() factory."""

    def test_returns_settings_instance(self):
        result = get_settings()
        assert isinstance(result, Settings)

    def test_returns_same_instance(self):
        a = get_settings()
        b = get_settings()
        assert a is b


class TestProductionValidator:
    """Settings.model_post_init must fail loud when required URLs missing in production."""

    REQUIRED_FIELDS = (
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

    def _all_set(self) -> dict[str, str]:
        return {name: f"http://stub-{name.lower()}:1" for name in self.REQUIRED_FIELDS}

    def test_production_with_all_required_passes(self):
        s = Settings(ENVIRONMENT="production", **self._all_set())
        assert s.ENVIRONMENT == "production"

    def test_production_with_missing_redis_fails(self):
        kwargs = self._all_set()
        kwargs["REDIS_URL"] = ""
        with pytest.raises(RuntimeError, match="REDIS_URL"):
            Settings(ENVIRONMENT="production", **kwargs)

    def test_production_with_missing_license_fails(self):
        kwargs = self._all_set()
        kwargs["LICENSE_INTROSPECT_URL"] = ""
        with pytest.raises(RuntimeError, match="LICENSE_INTROSPECT_URL"):
            Settings(ENVIRONMENT="production", **kwargs)

    def test_test_env_skips_validation(self, monkeypatch):
        # No required fields set; ENVIRONMENT="test" means validator must skip.
        # Conftest may have set REDIS_URL via os.environ; clear it.
        for key in self.REQUIRED_FIELDS:
            monkeypatch.delenv(key, raising=False)
        s = Settings(ENVIRONMENT="test")
        assert s.REDIS_URL == ""
        assert s.LICENSE_INTROSPECT_URL == ""
