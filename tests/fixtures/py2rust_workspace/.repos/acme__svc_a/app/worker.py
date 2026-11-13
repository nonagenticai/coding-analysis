import redis.asyncio as redis


async def consume_events():
    r = redis.Redis()
    while True:
        await r.xreadgroup("grp", "c1", {"orders.events": ">"})
