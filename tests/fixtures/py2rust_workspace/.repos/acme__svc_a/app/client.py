import os

import httpx

__all__ = ["fetch_item", "SERVICE_B_URL"]

SERVICE_B_URL = os.environ.get("SERVICE_B_URL", "http://svc-b:8002")


def _helper():
    return 1


async def fetch_item(item_id: int, *, timeout: float = 5.0) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{SERVICE_B_URL}/items/{item_id}")
        await client.post(f"{SERVICE_B_URL}/jobs", json={})
        await client.get(f"{SERVICE_B_URL}/nowhere")
        return resp.json()
