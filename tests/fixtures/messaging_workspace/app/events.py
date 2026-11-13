import os
from multiprocessing import Pool

import redis
from arq import create_pool
from celery import Celery
from rq import Queue
from worker import tasks
from worker.jobs import resize_image

ORDERS = "orders.created"
r = redis.Redis()
celery_app = Celery("publisher")
q = Queue(connection=r)


def order_created(order_id):
    r.publish(ORDERS, order_id)
    r.publish(f"user.{order_id}.events", "changed")
    r.publish(os.getenv("AUDIT_CHANNEL"), order_id)
    r.publish("nobody.listens", order_id)
    r.xadd("payments", {"id": order_id})
    celery_app.send_task("billing.charge", args=[order_id])
    tasks.send_receipt.delay(order_id)
    q.enqueue(resize_image, order_id)
    Pool().apply_async(print, (order_id,))


async def crunch_later(order_id):
    pool = await create_pool()
    await pool.enqueue_job("crunch", order_id)


async def receipt_later(order_id):
    pool = await create_pool()
    await pool.enqueue_job("send_receipt", order_id)


def process_order(order_id):
    tasks.process.delay(order_id)


def refund_order(order_id):
    from billing.tasks import refund

    refund.delay(order_id)
