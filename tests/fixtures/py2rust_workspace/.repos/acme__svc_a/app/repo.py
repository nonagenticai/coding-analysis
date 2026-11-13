from sqlalchemy import select

from app.models import Order


def list_orders(session):
    return session.scalars(select(Order)).all()


def add_order(session):
    session.add(Order(id=1))


def raw(conn, table):
    conn.execute("UPDATE orders SET id = 2 WHERE id = 1")
    conn.execute(f"SELECT * FROM {table}")
