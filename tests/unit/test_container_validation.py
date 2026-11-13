"""Unit tests for app.api.utils.container_validation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.utils.container_validation import validate_and_get_container_for_tenant


def _patch_docker(find_result):
    mock_service = MagicMock()
    mock_service.find_container_by_name = AsyncMock(return_value=find_result)

    patcher = patch(
        "services.docker_service.get_docker_service",
        return_value=mock_service,
    )
    return patcher, mock_service


class TestContainerFound:
    async def test_returns_container_info_with_validated_true(self):
        container = {"name": "my-ctr", "id": "abc-123", "status": "running"}
        patcher, _ = _patch_docker(container)
        with patcher:
            result = await validate_and_get_container_for_tenant("my-ctr")
        assert result.validated is True
        assert result.name == "my-ctr"
        assert result.id == "abc-123"
        assert result.status == "running"
        assert vars(result) == {**container, "validated": True}

    async def test_original_fields_preserved(self):
        container = {"name": "ctr", "image": "python:3.12", "port": 8080}
        patcher, _ = _patch_docker(container)
        with patcher:
            result = await validate_and_get_container_for_tenant("ctr")
        assert result.validated is True
        assert result.image == "python:3.12"
        assert result.port == 8080
        assert result.name == "ctr"
        assert set(vars(result).keys()) == {"name", "image", "port", "validated"}

    async def test_calls_find_container_by_name_with_correct_arg(self):
        container = {"name": "target"}
        patcher, mock_service = _patch_docker(container)
        with patcher:
            result = await validate_and_get_container_for_tenant("target")
        mock_service.find_container_by_name.assert_awaited_once_with("target", None)
        assert result.validated is True
        assert result.name == "target"

    async def test_forwards_tenant_to_lookup(self):
        # The tenant must reach the container lookup so coding-containers resolves
        # the container under the caller's tenant (else an other-tenant container
        # 404s and the graph read fails despite the graph existing).
        container = {"name": "target", "tenant_id": "tenant-9"}
        patcher, mock_service = _patch_docker(container)
        with patcher:
            await validate_and_get_container_for_tenant("target", "tenant-9")
        mock_service.find_container_by_name.assert_awaited_once_with("target", "tenant-9")


class TestContainerNotFound:
    async def test_raises_http_404(self):
        patcher, _ = _patch_docker(None)
        with patcher, pytest.raises(HTTPException) as exc_info:
            await validate_and_get_container_for_tenant("missing")
        assert exc_info.value.status_code == 404
        assert "missing" in exc_info.value.detail
        assert "not found" in exc_info.value.detail.lower()

    async def test_error_message_contains_container_name(self):
        patcher, _ = _patch_docker(None)
        with patcher:
            with pytest.raises(HTTPException) as exc_info:
                await validate_and_get_container_for_tenant("special-ctr")
        assert exc_info.value.status_code == 404
        assert "special-ctr" in exc_info.value.detail
        assert exc_info.value.detail == "Container 'special-ctr' not found"


class TestNoTenantId:
    async def test_no_tenant_id_skips_tenant_check(self):
        container = {"name": "ctr", "tenant_id": "tenant-A"}
        patcher, _ = _patch_docker(container)
        with patcher:
            result = await validate_and_get_container_for_tenant("ctr", tenant_id=None)
        assert result.validated is True
        assert result.tenant_id == "tenant-A"
        assert result.name == "ctr"

    async def test_no_tenant_id_default_parameter(self):
        container = {"name": "ctr", "tenant_id": "tenant-X"}
        patcher, _ = _patch_docker(container)
        with patcher:
            result = await validate_and_get_container_for_tenant("ctr")
        assert result.validated is True
        assert result.tenant_id == "tenant-X"
        assert result.name == "ctr"
        assert vars(result) == {"name": "ctr", "tenant_id": "tenant-X", "validated": True}


class TestTenantMatch:
    async def test_matching_tenant_id_field(self):
        container = {"name": "ctr", "tenant_id": "tenant-1"}
        patcher, _ = _patch_docker(container)
        with patcher:
            result = await validate_and_get_container_for_tenant("ctr", tenant_id="tenant-1")
        assert result.validated is True
        assert result.tenant_id == "tenant-1"
        assert result.name == "ctr"
        assert vars(result) == {"name": "ctr", "tenant_id": "tenant-1", "validated": True}

    async def test_matching_tenant_via_labels(self):
        container = {"name": "ctr", "labels": {"tenant_id": "tenant-2"}}
        patcher, _ = _patch_docker(container)
        with patcher:
            result = await validate_and_get_container_for_tenant("ctr", tenant_id="tenant-2")
        assert result.validated is True
        assert result.labels == {"tenant_id": "tenant-2"}
        assert result.name == "ctr"
        assert not hasattr(result, "tenant_id")

    async def test_container_without_any_tenant_metadata_passes(self):
        container = {"name": "ctr"}
        patcher, _ = _patch_docker(container)
        with patcher:
            result = await validate_and_get_container_for_tenant("ctr", tenant_id="any-tenant")
        assert result.validated is True
        assert result.name == "ctr"
        assert not hasattr(result, "tenant_id")
        assert vars(result) == {"name": "ctr", "validated": True}


class TestTenantMismatch:
    async def test_tenant_id_field_mismatch_raises(self):
        container = {"name": "ctr", "tenant_id": "tenant-A"}
        patcher, _ = _patch_docker(container)
        with patcher, pytest.raises(HTTPException) as exc_info:
            await validate_and_get_container_for_tenant("ctr", tenant_id="tenant-B")
        assert exc_info.value.status_code == 403
        assert "Access denied" in exc_info.value.detail
        assert "tenant" in exc_info.value.detail.lower()

    async def test_labels_tenant_mismatch_raises(self):
        container = {"name": "ctr", "labels": {"tenant_id": "tenant-A"}}
        patcher, _ = _patch_docker(container)
        with patcher, pytest.raises(HTTPException) as exc_info:
            await validate_and_get_container_for_tenant("ctr", tenant_id="tenant-Z")
        assert exc_info.value.status_code == 403
        assert "Access denied" in exc_info.value.detail
        assert exc_info.value.detail == "Access denied: container does not belong to tenant"

    async def test_tenant_id_field_takes_precedence_over_labels(self):
        container = {
            "name": "ctr",
            "tenant_id": "real-tenant",
            "labels": {"tenant_id": "real-tenant"},
        }
        patcher, _ = _patch_docker(container)
        with patcher, pytest.raises(HTTPException) as exc_info:
            await validate_and_get_container_for_tenant("ctr", tenant_id="wrong-tenant")
        assert exc_info.value.status_code == 403
        assert "Access denied" in exc_info.value.detail

        container_precedence = {
            "name": "ctr",
            "tenant_id": "top-level",
            "labels": {"tenant_id": "label-level"},
        }
        patcher2, _ = _patch_docker(container_precedence)
        with patcher2:
            result = await validate_and_get_container_for_tenant("ctr", tenant_id="top-level")
        assert result.validated is True
        assert result.tenant_id == "top-level"


class TestExtraKwargs:
    async def test_extra_kwargs_ignored(self):
        container = {"name": "ctr"}
        patcher, _ = _patch_docker(container)
        with patcher:
            result = await validate_and_get_container_for_tenant(
                "ctr",
                tenant_id=None,
                extra_param="ignored",
                another=123,
            )
        assert result.validated is True
        assert result.name == "ctr"
        assert not hasattr(result, "extra_param")
        assert not hasattr(result, "another")
        assert vars(result) == {"name": "ctr", "validated": True}
