from fastapi import APIRouter

from app.client import fetch_item

router = APIRouter(prefix="/orders")


@router.get("/{order_id}")
async def get_order(order_id: int):
    return await fetch_item(order_id)


@router.post("/events/{event_id}")
async def post_event(event_id: int):
    return {}
