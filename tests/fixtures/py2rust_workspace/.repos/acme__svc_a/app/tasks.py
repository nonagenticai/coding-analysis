from celery import Celery

celery_app = Celery("svc_a")


@celery_app.task
def send_email(order_id):
    return order_id
