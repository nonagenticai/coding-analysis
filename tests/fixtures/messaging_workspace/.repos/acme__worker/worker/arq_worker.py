async def crunch(ctx, order_id):
    return order_id


class WorkerSettings:
    functions = [crunch]
