from fastapi import FastAPI

app = FastAPI()


@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"id": item_id}


@app.post("/jobs")
async def create_job():
    return {}
