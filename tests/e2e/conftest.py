"""E2E test configuration.

Requires a running coding-analysis container at localhost:8688.
Start with:
    docker run -d --name coding-analysis-e2e -p 8688:8688 \
        -e ENABLE_AUTH=false -e ENVIRONMENT=test \
        coding-analysis
"""

from __future__ import annotations

import os

import httpx
import pytest

BASE_URL = os.getenv("E2E_BASE_URL", "http://localhost:8688")


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as c:
        yield c


@pytest.fixture(scope="session")
def async_client() -> httpx.AsyncClient:
    # Caller uses async with; we just provide the configured client
    return httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)
