import sqlalchemy as sa
from alembic import op


def upgrade():
    op.create_table("orders", sa.Column("id", sa.Integer))
    op.add_column("orders", sa.Column("note", sa.String))
