"""Unit tests for contract analysis endpoints."""

from __future__ import annotations

import pathlib
import sys
import types as _types
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

# ---------------------------------------------------------------------------
# sys.modules surgery
# ---------------------------------------------------------------------------

_src_dir = pathlib.Path(__file__).resolve().parents[2] / "src"

for _key in list(sys.modules):
    if _key in ("app.api.endpoints.contract_analysis",):
        del sys.modules[_key]

for _pkg, _subpath in (
    ("app", "app"),
    ("app.api", "app/api"),
    ("app.api.endpoints", "app/api/endpoints"),
    ("app.api.models", "app/api/models"),
    ("app.core", "app/core"),
):
    if _pkg not in sys.modules:
        _m = _types.ModuleType(_pkg)
        _m.__path__ = [str(_src_dir / _subpath)]
        _m.__package__ = _pkg
        sys.modules[_pkg] = _m
    elif not getattr(sys.modules[_pkg], "__path__", None):
        sys.modules[_pkg].__path__ = [str(_src_dir / _subpath)]

# services stubs
if "services" not in sys.modules:
    _svc = _types.ModuleType("services")
    _svc.__path__ = [str(_src_dir / "services")]
    sys.modules["services"] = _svc
if "services.code_graph_metrics" not in sys.modules:
    _scgm = _types.ModuleType("services.code_graph_metrics")
    _scgm.CodeGraphMetricsService = MagicMock
    sys.modules["services.code_graph_metrics"] = _scgm

# processing.contract_analysis stubs
if "processing" not in sys.modules:
    _proc = _types.ModuleType("processing")
    _proc.__path__ = [str(_src_dir / "processing")]
    sys.modules["processing"] = _proc
if "processing.contract_analysis" not in sys.modules:
    _pca = _types.ModuleType("processing.contract_analysis")
    _pca.__path__ = [str(_src_dir / "processing" / "contract_analysis")]
    sys.modules["processing.contract_analysis"] = _pca
if "processing.contract_analysis.scanner" not in sys.modules:
    _pcs_stub = _types.ModuleType("processing.contract_analysis.scanner")

    class _FakeContractResult:
        def __init__(self, name="test", passed=True, violations=None, stats=None, parse_error=None):
            self.name = name
            self.passed = passed
            self.violations = violations or []
            self.stats = stats or {}
            self.parse_error = parse_error

    class _FakeContractScanner:
        def scan_sources(self, **kwargs):
            return _FakeContractResult(name=kwargs.get("name", "test"), passed=True)

    _pcs_stub.ContractResult = _FakeContractResult
    _pcs_stub.ContractScanner = _FakeContractScanner
    sys.modules["processing.contract_analysis.scanner"] = _pcs_stub

# Now import
from app.api.endpoints.contract_analysis import (  # noqa: E402
    scan_capabilities,
    scan_contracts,
)
from models.contract_analysis_models import (  # noqa: E402
    ContractPair,
    ScanConfig,
    ScanRequest,
    SourceInput,
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestContractAnalysisEndpoints:
    """Tests for contract analysis endpoints."""

    async def test_scan_success_with_representative_input(self):
        request = ScanRequest(
            contracts=[
                ContractPair(
                    name="test-contract",
                    client=SourceInput(
                        source="import requests\nrequests.get('/api/users')", label="client"
                    ),
                    schemas=[SourceInput(source="class User:\n    name: str", label="schema")],
                    routes=[
                        SourceInput(
                            source="@app.get('/api/users')\ndef get_users(): pass", label="route"
                        )
                    ],
                ),
            ],
        )
        result = await scan_contracts(body=request, tenant_id="t")
        assert result.total_contracts == 1
        assert result.passed is True
        assert result.total_violations == 0

    async def test_capabilities_success(self):
        result = await scan_capabilities(tenant_id="t")
        assert "route_path" in result.checks
        assert "http_method" in result.checks
        assert result.supports_file_paths is True

    async def test_parser_check_failure_handling(self):
        """When no contracts are provided, endpoint raises 400."""
        # Bypass Pydantic's min_length=1 validator to reach the endpoint's own check
        request = ScanRequest.model_construct(contracts=[], config=ScanConfig())
        with pytest.raises(HTTPException) as exc_info:
            await scan_contracts(body=request, tenant_id="t")
        assert exc_info.value.status_code == 400
