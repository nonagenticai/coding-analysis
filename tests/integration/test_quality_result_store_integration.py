"""Round-trip QualityResultStore against a real PostgreSQL + migration 0007 (#106).

Skipped unless ``TEST_DATABASE_URL`` points at a reachable postgres (the CI
``postgres:18`` service provides it). Marked ``integration`` so it is excluded
from the default ``-m 'not integration'`` run.

Tenant literals must be real uuids: ``quality_results.tenant_id`` is a native
``uuid`` column like every other tenant column since migration 0005.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_DSN = os.getenv("TEST_DATABASE_URL")
if not _DSN:
    pytest.skip("TEST_DATABASE_URL not set", allow_module_level=True)

TENANT_A = "0195f48b-0000-7000-8000-00000000000a"
TENANT_B = "0195f48b-0000-7000-8000-00000000000b"
SHA_1 = "1" * 40
SHA_2 = "2" * 40


@pytest.fixture(scope="module", autouse=True)
def _migrated_db():
    os.environ["DATABASE_URL"] = _DSN
    os.environ["PGBOUNCER_URL"] = ""

    from app.core.config.core_config_settings import get_settings

    get_settings.cache_clear()

    from alembic.config import Config

    from alembic import command

    repo_root = Path(__file__).resolve().parents[2]
    cfg = Config(str(repo_root / "alembic.ini"))
    command.upgrade(cfg, "head")
    yield
    command.downgrade(cfg, "base")


@pytest.fixture(autouse=True)
async def _clean_table():
    from db import session

    async with session.get_raw_connection() as conn:
        await conn.execute("TRUNCATE quality_results")
        await conn.commit()
    yield


def _report(**kw):
    from tests.unit.quality_store_fakes import make_report

    return make_report(**kw)


async def test_save_then_latest_round_trip():
    from services.quality_result_store import QualityResultStore

    store = QualityResultStore()
    first = await store.save(
        _report(score=70.0),
        tenant_id=TENANT_A,
        container_name="c1",
        repo_url="https://github.com/o/r",
        commit_sha=SHA_1,
    )
    second = await store.save(
        _report(score=90.0),
        tenant_id=TENANT_A,
        container_name="c1",
        repo_url="https://github.com/o/r",
        commit_sha=SHA_1,  # same sha re-analysed: a NEW row, no unique violation
    )
    assert first != second

    row = await store.latest(TENANT_A, "c1")
    assert row is not None
    assert row["id"] == second
    assert row["tenant_id"] == TENANT_A  # ::text cast, a str not uuid.UUID
    assert isinstance(row["tenant_id"], str)
    assert row["overall_score"] == 90.0
    assert row["commit_sha"] == SHA_1
    assert row["status"] == "complete"
    assert isinstance(row["issues"], list)

    history = await store.history(TENANT_A, "c1")
    assert [r["id"] for r in history] == [second, first]


async def test_tenant_scoping():
    from services.quality_result_store import QualityResultStore

    store = QualityResultStore()
    await store.save(
        _report(), tenant_id=TENANT_B, container_name="c1", repo_url=None, commit_sha=SHA_2
    )
    assert await store.latest(TENANT_A, "c1") is None
    assert await store.history(TENANT_A, "c1") == []


async def test_failed_and_inconclusive_rows_store_no_score():
    from services.quality_result_store import QualityResultStore

    store = QualityResultStore()
    await store.save(None, tenant_id=TENANT_A, container_name="c1", repo_url=None, commit_sha=None)
    failed = await store.latest(TENANT_A, "c1")
    assert failed["status"] == "failed"
    assert failed["overall_score"] is None
    assert failed["commit_sha"] is None

    await store.save(
        _report(inconclusive=True, tools_succeeded=[], tools_failed=["ruff"]),
        tenant_id=TENANT_A,
        container_name="c1",
        repo_url=None,
        commit_sha=None,
    )
    inconclusive = await store.latest(TENANT_A, "c1")
    assert inconclusive["status"] == "inconclusive"
    assert inconclusive["overall_score"] is None


async def test_check_constraints_refuse_bad_rows():
    import psycopg

    from db import session

    async with session.get_raw_connection() as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            await conn.execute(
                "INSERT INTO quality_results (id, tenant_id, container_name, status, commit_sha) "
                "VALUES ('x1', %s, 'c1', 'complete', 'main')",
                (TENANT_A,),
            )
        await conn.rollback()
        with pytest.raises(psycopg.errors.CheckViolation):
            await conn.execute(
                "INSERT INTO quality_results (id, tenant_id, container_name, status) "
                "VALUES ('x2', %s, 'c1', 'great')",
                (TENANT_A,),
            )
        await conn.rollback()


async def test_latest_for_sha_is_keyed_by_commit_and_tenant():
    from services.quality_result_store import QualityResultStore

    store = QualityResultStore()
    old = await store.save(
        _report(score=50.0),
        tenant_id=TENANT_A,
        container_name="c1",
        repo_url=None,
        commit_sha=SHA_1,
    )
    await store.save(
        _report(score=99.0),
        tenant_id=TENANT_B,
        container_name="c1",
        repo_url=None,
        commit_sha=SHA_2,
    )
    assert (await store.latest_for_sha(TENANT_A, "c1", SHA_1))["id"] == old
    # Another tenant's row at SHA_2 is invisible; no row for SHA_2 in tenant A.
    assert await store.latest_for_sha(TENANT_A, "c1", SHA_2) is None


async def test_statement_timeout_is_transaction_local_and_does_not_leak():
    """The store's 5s bound is SET LOCAL: after a save the SAME pooled
    connection (min_size=1) is back at the server default (pgbouncer txn-pooling safety)."""
    from db import session
    from services.quality_result_store import QualityResultStore

    pool = await session.get_pool()
    async with pool.connection() as conn:
        default = (await (await conn.execute("SHOW statement_timeout")).fetchone())[0]
        await conn.rollback()
    await QualityResultStore().save(
        _report(), tenant_id=TENANT_A, container_name="c1", repo_url=None, commit_sha=SHA_1
    )
    async with pool.connection() as conn:
        after = (await (await conn.execute("SHOW statement_timeout")).fetchone())[0]
        await conn.rollback()
    assert default != "5s"
    assert after == default
    assert (await QualityResultStore().latest(TENANT_A, "c1")) is not None
