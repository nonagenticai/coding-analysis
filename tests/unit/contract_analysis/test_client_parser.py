from processing.contract_analysis.parsers.client_parser import parse_client

SAMPLE_CLIENT = """
class AnalyzerClient(BaseServiceClient):
    def __init__(self, base_url="http://localhost:6303", **kwargs):
        super().__init__(base_url=base_url, service_name="analyzer", **kwargs)

    async def start_analysis(self, pages, user_context=None):
        payload = {"pages": pages}
        if user_context:
            payload["user_context"] = user_context
        return await self.submit_job("/api/analyze", payload)

    async def get_analysis(self, job_id):
        return await self._request("GET", f"/api/analyze/{job_id}")

    async def wait_for_analysis(self, job_id, timeout=300.0):
        return await self.poll_job("/api/analyze", job_id, timeout=timeout)
"""


def test_parse_calls():
    result = parse_client(SAMPLE_CLIENT)
    assert len(result.calls) >= 2


def test_submit_job_detected():
    result = parse_client(SAMPLE_CLIENT)
    submit_calls = [c for c in result.calls if c.path == "/api/analyze" and c.method == "POST"]
    assert len(submit_calls) == 1


def test_get_request_detected():
    result = parse_client(SAMPLE_CLIENT)
    get_calls = [c for c in result.calls if c.method == "GET"]
    assert len(get_calls) >= 1
    assert any("/api/analyze/" in c.path for c in get_calls)


def test_payload_fields():
    result = parse_client(SAMPLE_CLIENT)
    submit_call = [c for c in result.calls if c.path == "/api/analyze" and c.method == "POST"][0]
    assert "pages" in submit_call.payload_fields


def test_fstring_path_normalized():
    result = parse_client(SAMPLE_CLIENT)
    get_calls = [c for c in result.calls if c.method == "GET"]
    paths = [c.path for c in get_calls]
    assert any("{" in p for p in paths)


def test_empty_source():
    result = parse_client("")
    assert len(result.calls) == 0


def test_response_reads_get_pattern():
    """Detect result.get("field") response reads."""
    source = """
class FooClient(BaseServiceClient):
    async def get_status(self, job_id):
        result = await self._request("GET", f"/api/foo/{job_id}")
        return result.get("status", "unknown")
"""
    result = parse_client(source)
    assert len(result.calls) == 1
    assert "status" in result.calls[0].response_reads


def test_response_reads_subscript_pattern():
    """Detect result["field"] response reads."""
    source = """
class FooClient(BaseServiceClient):
    async def create(self):
        result = await self.submit_job("/api/foo", {"name": "test"})
        return result["job_id"]
"""
    result = parse_client(source)
    submit_calls = [c for c in result.calls if c.method == "POST"]
    assert len(submit_calls) == 1
    assert "job_id" in submit_calls[0].response_reads


def test_response_reads_multiple_fields():
    """Detect multiple response reads from same call."""
    source = """
class FooClient(BaseServiceClient):
    async def get_info(self, job_id):
        data = await self._request("GET", f"/api/foo/{job_id}")
        status = data.get("status")
        name = data["name"]
        return {"status": status, "name": name}
"""
    result = parse_client(source)
    assert len(result.calls) == 1
    reads = result.calls[0].response_reads
    assert "status" in reads
    assert "name" in reads


def test_httpx_async_client_context_manager():
    """Detect httpx.AsyncClient context manager calls."""
    source = """
class NarrativeClient:
    async def generate(self, prompt):
        payload = {"prompt": prompt, "max_tokens": 500}
        async with httpx.AsyncClient(base_url=self.service_url) as client:
            response = await client.post("/api/generate", json=payload)
            return response.json()
"""
    result = parse_client(source)
    post_calls = [c for c in result.calls if c.method == "POST"]
    assert len(post_calls) == 1
    assert post_calls[0].path == "/api/generate"
    assert "prompt" in post_calls[0].payload_fields


def test_httpx_direct_get():
    """Detect direct httpx GET calls."""
    source = """
class HealthChecker:
    async def check(self, url):
        async with httpx.AsyncClient() as client:
            resp = await client.get("/health")
"""
    result = parse_client(source)
    get_calls = [c for c in result.calls if c.method == "GET"]
    assert len(get_calls) == 1
    assert get_calls[0].path == "/health"


def test_poll_job_appends_param():
    """poll_job path should include /{param} for route matching."""
    source = """
class FooClient(BaseServiceClient):
    async def wait_for_result(self, job_id):
        return await self.poll_job("/api/foo/jobs", job_id, timeout=300)
"""
    result = parse_client(source)
    poll_calls = [c for c in result.calls if c.method == "GET"]
    assert len(poll_calls) == 1
    assert poll_calls[0].path == "/api/foo/jobs/{param}"


def test_request_extracts_json_payload():
    """_request() should extract json= keyword argument payload fields."""
    source = """
class FooClient(BaseServiceClient):
    async def create(self, name, url):
        payload = {"name": name, "url": url}
        return await self._request("POST", "/api/foo", json=payload)
"""
    result = parse_client(source)
    post_calls = [c for c in result.calls if c.method == "POST"]
    assert len(post_calls) == 1
    assert "name" in post_calls[0].payload_fields
    assert "url" in post_calls[0].payload_fields


def test_request_extracts_inline_json_dict():
    """_request() should extract inline json={...} dict payload fields."""
    source = """
class FooClient(BaseServiceClient):
    async def create(self, text):
        return await self._request("POST", "/api/bar", json={"text": text, "mode": "fast"})
"""
    result = parse_client(source)
    post_calls = [c for c in result.calls if c.method == "POST"]
    assert len(post_calls) == 1
    assert "text" in post_calls[0].payload_fields
    assert "mode" in post_calls[0].payload_fields


def test_stored_httpx_self_client():
    """Detect self._client.post("/path") stored client pattern."""
    source = """
class FooClient:
    async def create(self, data):
        response = await self._client.post("/api/foo", json=data)
        return response.json()
"""
    result = parse_client(source)
    post_calls = [c for c in result.calls if c.method == "POST"]
    assert len(post_calls) == 1
    assert post_calls[0].path == "/api/foo"


def test_stored_httpx_get_client_helper():
    """Detect client = self._get_client() then client.get("/path")."""
    source = """
class FooClient:
    async def get_item(self, item_id):
        client = self._get_client()
        response = await client.get(f"/api/items/{item_id}")
        return response.json()
"""
    result = parse_client(source)
    get_calls = [c for c in result.calls if c.method == "GET"]
    assert len(get_calls) == 1
    assert "/api/items/" in get_calls[0].path


def test_stored_httpx_stream():
    """Detect self._client.stream("POST", "/path") streaming pattern."""
    source = """
class FooClient:
    async def stream_data(self, session_id):
        async with self._client.stream(
            "POST", f"/api/stream/{session_id}", json={"mode": "live"}
        ) as response:
            async for line in response.aiter_lines():
                yield line
"""
    result = parse_client(source)
    post_calls = [c for c in result.calls if c.method == "POST"]
    assert len(post_calls) == 1
    assert "/api/stream/" in post_calls[0].path


def test_stored_httpx_with_base_url():
    """Detect paths with {self.base_url} prefix and strip it."""
    source = """
class FooClient:
    async def get_product(self, product_id):
        resp = await self._client.get(
            f"{self.base_url}/api/products/{product_id}"
        )
        return resp.json()
"""
    result = parse_client(source)
    get_calls = [c for c in result.calls if c.method == "GET"]
    assert len(get_calls) == 1
    assert get_calls[0].path == "/api/products/{product_id}"


def test_stored_httpx_payload_opaque():
    """payload_opaque set when json=data (variable, not dict literal)."""
    source = """
class FooClient:
    async def suggest(self, product_id, data: dict):
        return await self._client.post(f"/api/products/{product_id}/suggest", json=data)
"""
    result = parse_client(source)
    post_calls = [c for c in result.calls if c.method == "POST"]
    assert len(post_calls) == 1
    assert post_calls[0].payload_opaque is True
    assert post_calls[0].payload_fields == []


def test_no_false_positive_on_dict_get():
    """self.cache.get("key") should NOT be detected as HTTP call."""
    source = """
class FooService:
    async def get_cached(self, key):
        return self.cache.get("user_data")
"""
    result = parse_client(source)
    assert len(result.calls) == 0


def test_head_and_options_methods():
    """Detect self._client.head() and self._client.options() calls."""
    source = """
class FooClient:
    async def check_exists(self):
        return await self._client.head("/api/foo")

    async def get_options(self):
        return await self._client.options("/api/foo")
"""
    result = parse_client(source)
    assert len(result.calls) == 2
    methods = sorted(c.method for c in result.calls)
    assert methods == ["HEAD", "OPTIONS"]
    assert all(c.path == "/api/foo" for c in result.calls)


def test_strip_base_url_internal_path():
    """Paths with {self.base_url}/_internal/... should strip the prefix."""
    source = """
class FooClient:
    async def get_items(self, item_id):
        resp = await self._client.get(
            f"{self.base_url}/_internal/v1/items/{item_id}"
        )
        return resp.json()
"""
    result = parse_client(source)
    get_calls = [c for c in result.calls if c.method == "GET"]
    assert len(get_calls) == 1
    assert get_calls[0].path == "/_internal/v1/items/{item_id}"


def test_model_dump_payload_detected():
    """Detect payload.model_dump() and trace to parameter type."""
    source = """
class FooClient:
    async def create(self, payload: CreateRequest):
        await self._client.post("/api/foo", json=payload.model_dump())
"""
    result = parse_client(source)
    post_calls = [c for c in result.calls if c.method == "POST"]
    assert len(post_calls) == 1
    assert post_calls[0].payload_model == "CreateRequest"
    assert post_calls[0].payload_opaque is True


def test_dict_method_payload_detected():
    """Detect payload.dict() (Pydantic v1 compat) and trace to parameter type."""
    source = """
class FooClient:
    async def create(self, payload: CreateRequest):
        await self._client.post("/api/foo", json=payload.dict())
"""
    result = parse_client(source)
    post_calls = [c for c in result.calls if c.method == "POST"]
    assert len(post_calls) == 1
    assert post_calls[0].payload_model == "CreateRequest"


def test_query_params_from_params_kwarg():
    """Extract query params from params= dict kwarg."""
    source = """
class FooClient:
    async def list_items(self, page, limit):
        return await self._client.get("/api/items", params={"page": page, "limit": limit})
"""
    result = parse_client(source)
    get_calls = [c for c in result.calls if c.method == "GET"]
    assert len(get_calls) == 1
    assert "page" in get_calls[0].query_params
    assert "limit" in get_calls[0].query_params
