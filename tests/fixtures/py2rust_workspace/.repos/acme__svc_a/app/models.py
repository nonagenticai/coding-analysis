from sqlalchemy import Column, Integer, MetaData, Table
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)


metadata = MetaData()
audit = Table("audit_log", metadata, Column("id", Integer))
