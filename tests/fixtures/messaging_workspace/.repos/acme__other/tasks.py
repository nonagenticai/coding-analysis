from celery import shared_task


@shared_task
def process(order_id):
    return order_id
