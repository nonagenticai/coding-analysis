import redis
from rx import of

r = redis.Redis()
p = r.pubsub()


def listen():
    p.subscribe("orders.created")
    p.psubscribe("user.*")
    p.subscribe("lonely")
    r.xreadgroup("grp", "c1", {"payments": ">"})
    of(1).subscribe(on_next=print)
