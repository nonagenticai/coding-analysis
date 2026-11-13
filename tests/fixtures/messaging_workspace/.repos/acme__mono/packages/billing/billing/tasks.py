from celery import shared_task


@shared_task
def refund(order_id):
    return order_id
