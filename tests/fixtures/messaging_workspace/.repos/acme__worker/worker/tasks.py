from celery import shared_task


@shared_task(name="billing.charge")
def charge(order_id):
    return order_id


@shared_task
def send_receipt(order_id):
    return order_id


@shared_task
def never_called():
    return None
