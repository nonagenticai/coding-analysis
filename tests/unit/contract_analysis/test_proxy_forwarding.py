from processing.contract_analysis.parsers.client_parser import parse_client

PROXY_SOURCE = """
class GatewayRouter:
    async def handle_artifacts(self, request):
        await proxy.forward_request(
            service_url=settings.ARTIFACTS_URL,
            path="/api/artifacts",
            method="POST",
            data={"name": "test", "type": "file"},
        )

    async def handle_search(self, query: str):
        await proxy.forward_request(
            service_url=settings.SEARCH_URL,
            path="/api/search",
            method="GET",
        )
"""


def test_forward_request_detected():
    """proxy.forward_request(path=..., method=...) produces correct ClientCall."""
    result = parse_client(PROXY_SOURCE)
    artifact_calls = [c for c in result.calls if c.path == "/api/artifacts"]
    assert len(artifact_calls) == 1
    assert artifact_calls[0].method == "POST"


def test_forward_request_get_detected():
    """GET method proxy calls also detected."""
    result = parse_client(PROXY_SOURCE)
    search_calls = [c for c in result.calls if c.path == "/api/search"]
    assert len(search_calls) == 1
    assert search_calls[0].method == "GET"


def test_proxy_data_payload_extracted():
    """data={"key": "val"} extracts payload fields."""
    result = parse_client(PROXY_SOURCE)
    artifact_calls = [c for c in result.calls if c.path == "/api/artifacts"]
    assert len(artifact_calls) == 1
    assert "name" in artifact_calls[0].payload_fields
    assert "type" in artifact_calls[0].payload_fields


def test_proxy_data_opaque():
    """data=some_var sets payload_opaque=True when fields can't be extracted."""
    source = """
class Router:
    async def forward(self, data):
        await proxy.forward_request(
            path="/api/items",
            method="POST",
            data=data,
        )
"""
    result = parse_client(source)
    calls = [c for c in result.calls if c.path == "/api/items"]
    assert len(calls) == 1
    # data=data is a variable, not a dict literal, but _find_dict_literal_keys
    # won't find it as a dict assignment in the same scope, so payload_fields
    # will be empty. payload_opaque depends on _has_kwarg("data") which is True.
    assert calls[0].payload_opaque is True


def test_proxy_no_match_without_path_or_method():
    """Missing path= or method= kwarg means no call detected."""
    source = """
class Router:
    async def bad(self):
        await proxy.forward_request(service_url=settings.URL)
"""
    result = parse_client(source)
    assert len(result.calls) == 0


def test_forward_request_raw_detected():
    """forward_request_raw is also detected."""
    source = """
class Router:
    async def raw_forward(self):
        await proxy.forward_request_raw(
            path="/api/raw",
            method="PUT",
        )
"""
    result = parse_client(source)
    calls = [c for c in result.calls if c.path == "/api/raw"]
    assert len(calls) == 1
    assert calls[0].method == "PUT"
