import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket

from app.api import router as api_router
from app.worker import consume_events


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(consume_events())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)
app.include_router(api_router, prefix="/api")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()


def run():
    pass


if __name__ == "__main__":
    run()
