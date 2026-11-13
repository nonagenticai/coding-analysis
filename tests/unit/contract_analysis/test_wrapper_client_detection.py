from processing.contract_analysis.parsers.client_parser import parse_client

WRAPPER_CLIENT_SOURCE = """
class AnalysisService:
    async def run_scan(self, code: str, language: str):
        client = _get_http_client()
        result = await client.scan(json={"code": code, "language": language})
        return result

    async def check_status(self):
        client = _get_http_client()
        return await client.status()
"""


def test_wrapper_client_factory_detected():
    """Factory pattern client.scan(json=...) produces POST @wrapper/scan."""
    result = parse_client(WRAPPER_CLIENT_SOURCE)
    scan_calls = [c for c in result.calls if c.path == "@wrapper/scan"]
    assert len(scan_calls) == 1
    assert scan_calls[0].method == "POST"


def test_wrapper_client_no_json_infers_get():
    """client.status() without json= infers GET."""
    result = parse_client(WRAPPER_CLIENT_SOURCE)
    status_calls = [c for c in result.calls if c.path == "@wrapper/status"]
    assert len(status_calls) == 1
    assert status_calls[0].method == "GET"


def test_wrapper_client_payload_fields_extracted():
    """json= dict literal keys are extracted from wrapper calls."""
    result = parse_client(WRAPPER_CLIENT_SOURCE)
    scan_calls = [c for c in result.calls if c.path == "@wrapper/scan"]
    assert len(scan_calls) == 1
    assert "code" in scan_calls[0].payload_fields
    assert "language" in scan_calls[0].payload_fields


def test_wrapper_client_await_factory():
    """Await on factory function also detected."""
    source = """
class Svc:
    async def broadcast(self, event: str):
        client = await _get_client()
        await client.notify(json={"event": event})
"""
    result = parse_client(source)
    notify_calls = [c for c in result.calls if c.path == "@wrapper/notify"]
    assert len(notify_calls) == 1
    assert notify_calls[0].method == "POST"


def test_wrapper_does_not_shadow_httpx():
    """Wrapper var calling .get() (an HTTPX method) is NOT treated as wrapper."""
    source = """
class Svc:
    async def fetch(self):
        client = _get_http_client()
        await client.get("/api/items")
"""
    result = parse_client(source)
    # .get() is an HTTPX method, should be detected as generic HTTP call, not wrapper
    wrapper_calls = [c for c in result.calls if c.path.startswith("@wrapper/")]
    assert len(wrapper_calls) == 0


def test_non_factory_not_detected():
    """Variables from non-factory functions are not treated as wrappers."""
    source = """
class Svc:
    async def fetch(self):
        client = get_something_else()
        await client.do_work(json={"key": "val"})
"""
    result = parse_client(source)
    wrapper_calls = [c for c in result.calls if c.path.startswith("@wrapper/")]
    assert len(wrapper_calls) == 0
