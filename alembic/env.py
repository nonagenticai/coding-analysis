"""Alembic environment configuration for coding-analysis."""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Ensure src/ is on the path so ORM models can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from db.models import Base  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Allow DATABASE_URL env var to override alembic.ini sqlalchemy.url.
# In-cluster the migrate initContainer points DATABASE_URL at a pooler
# (pgbouncer:6432) — direct database egress is blocked from pods by the
# ambient mesh / NetworkPolicy. NullPool + a single short-lived DDL transaction
# is pgbouncer-txn-mode-safe (matches coding-gateway).
_db_url = os.getenv("DATABASE_URL")
if _db_url:
    # Fleet standard: psycopg v3. SQLAlchemy picks psycopg2 for bare
    # postgresql:// URLs; force the v3 driver since psycopg2 isn't installed.
    if _db_url.startswith("postgresql://"):
        _db_url = _db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    # Escape % for configparser interpolation (e.g. %40 in passwords).
    config.set_main_option("sqlalchemy.url", _db_url.replace("%", "%%"))


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
