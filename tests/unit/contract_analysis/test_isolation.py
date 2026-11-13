"""The per-request extraction worker: deadline kill, partial results, failures, memory cap."""

from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest

from processing.contract_analysis import isolation
from processing.contract_analysis.isolation import IsolationFailedError, Limits

from . import isolation_jobs

FAST = Limits(deadline_s=20.0, memory_mb=0, max_workers=2)


def _gone(pid: int, within_s: float = 3.0) -> bool:
    end = time.monotonic() + within_s
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.02)
    return False


def test_runs_the_job_in_another_process():
    assert asyncio.run(isolation.run(isolation_jobs.add, 2, 3, limits=FAST)) == 5
    pid = asyncio.run(isolation.run(os.getpid, limits=FAST))
    assert pid != os.getpid()


@pytest.mark.timeout(30)
def test_deadline_kills_the_worker_and_keeps_partial_results():
    limits = Limits(deadline_s=1.0, memory_mb=0, max_workers=2)
    start = time.monotonic()
    items, failure = asyncio.run(isolation.stream(isolation_jobs.pid_then_spin, limits=limits))
    elapsed = time.monotonic() - start
    assert elapsed < 3.0, f"{elapsed:.2f}s"
    assert failure is not None and failure.reason == "deadline"
    assert len(items) == 1 and items[0] != os.getpid()
    assert _gone(items[0]), "the spinning worker was not killed"


def test_job_exception_is_a_worker_error():
    with pytest.raises(IsolationFailedError) as exc:
        asyncio.run(isolation.run(isolation_jobs.fail, limits=FAST))
    assert exc.value.reason == "worker_error"
    assert exc.value.detail == "ValueError: boom"


def test_worker_exit_without_result_is_worker_died():
    with pytest.raises(IsolationFailedError) as exc:
        asyncio.run(isolation.run(isolation_jobs.die, limits=FAST))
    assert exc.value.reason == "worker_died"


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is enforced on Linux only")
def test_memory_cap_fails_the_job_not_the_service():
    limits = Limits(deadline_s=20.0, memory_mb=512, max_workers=2)
    with pytest.raises(IsolationFailedError) as exc:
        asyncio.run(isolation.run(isolation_jobs.allocate, 1024, limits=limits))
    assert exc.value.reason in ("worker_error", "worker_died")
    # Under the cap the same job runs.
    assert asyncio.run(isolation.run(isolation_jobs.allocate, 16, limits=limits)) == 16 << 20


@pytest.mark.timeout(30)
def test_no_free_worker_within_the_deadline_is_busy():
    hold = Limits(deadline_s=1.5, memory_mb=0, max_workers=1)
    short = Limits(deadline_s=0.3, memory_mb=0, max_workers=1)

    async def both():
        spinner = asyncio.create_task(isolation.stream(isolation_jobs.pid_then_spin, limits=hold))
        await asyncio.sleep(0.3)  # the spinner holds the only slot
        with pytest.raises(IsolationFailedError) as exc:
            await isolation.run(isolation_jobs.add, 1, 1, limits=short)
        _, failure = await spinner
        return exc.value, failure

    busy, failure = asyncio.run(both())
    assert busy.reason == "busy"
    assert failure is not None and failure.reason == "deadline"


def test_inline_mode_runs_in_process():
    limits = Limits(deadline_s=1.0, memory_mb=0, max_workers=1, isolated=False)
    assert asyncio.run(isolation.run(os.getpid, limits=limits)) == os.getpid()


@pytest.mark.timeout(30)
def test_a_slot_freed_too_late_is_busy_not_a_doomed_start():
    """Never start a job with (almost) none of its deadline left: answer busy promptly."""
    hold = Limits(deadline_s=1.0, memory_mb=0, max_workers=1)
    late = Limits(deadline_s=1.4, memory_mb=0, max_workers=1, min_run_s=0.6)

    async def both():
        spinner = asyncio.create_task(isolation.stream(isolation_jobs.pid_then_spin, limits=hold))
        await asyncio.sleep(0.1)  # the spinner holds the only slot until ~1.0s
        start = time.monotonic()
        with pytest.raises(IsolationFailedError) as exc:
            await isolation.run(isolation_jobs.sleep_then, 0.5, 7, limits=late)
        elapsed = time.monotonic() - start
        await spinner
        return exc.value, elapsed

    failure, elapsed = asyncio.run(both())
    # Waiting until the slot freed (~0.9s) would have left 0.5s: a certain "deadline".
    assert failure.reason == "busy", failure
    assert elapsed < 0.95, f"{elapsed:.2f}s"


@pytest.mark.timeout(30)
def test_queue_wait_bounds_the_wait_for_a_slot():
    hold = Limits(deadline_s=2.0, memory_mb=0, max_workers=1)
    queued = Limits(deadline_s=10.0, memory_mb=0, max_workers=1, queue_wait_s=0.3)

    async def both():
        spinner = asyncio.create_task(isolation.stream(isolation_jobs.pid_then_spin, limits=hold))
        await asyncio.sleep(0.1)
        start = time.monotonic()
        with pytest.raises(IsolationFailedError) as exc:
            await isolation.run(isolation_jobs.add, 1, 1, limits=queued)
        elapsed = time.monotonic() - start
        await spinner
        return exc.value, elapsed

    failure, elapsed = asyncio.run(both())
    assert failure.reason == "busy"
    assert elapsed < 1.0, f"{elapsed:.2f}s"


@pytest.mark.timeout(30)
def test_one_tenant_cannot_hold_every_slot():
    limits = Limits(deadline_s=2.0, memory_mb=0, max_workers=2, queue_wait_s=0.3, max_per_tenant=1)

    async def scenario():
        spinner = asyncio.create_task(
            isolation.stream(isolation_jobs.pid_then_spin, limits=limits, tenant="a")
        )
        await asyncio.sleep(0.1)  # tenant a runs one job; one slot is still free
        with pytest.raises(IsolationFailedError) as exc:
            await isolation.run(isolation_jobs.add, 1, 1, limits=limits, tenant="a")
        other = await isolation.run(isolation_jobs.add, 2, 2, limits=limits, tenant="b")
        await spinner
        return exc.value, other

    second_a, other = asyncio.run(scenario())
    assert second_a.reason == "busy"
    assert other == 4  # another tenant still gets the free slot


def test_admission_cap_refuses_at_once():
    limits = Limits(deadline_s=5.0, memory_mb=0, max_workers=2, max_pending=1)

    async def scenario():
        async with isolation.admit(limits):
            start = time.monotonic()
            with pytest.raises(IsolationFailedError) as exc:
                async with isolation.admit(limits):
                    pass
            assert time.monotonic() - start < 0.1
        async with isolation.admit(limits):  # released on exit
            pass
        return exc.value

    assert asyncio.run(scenario()).reason == "busy"


def test_self_check_passes_when_a_worker_starts():
    asyncio.run(isolation.self_check(FAST))


def test_self_check_fails_loudly_when_no_worker_can_start(monkeypatch):
    real = isolation._context()

    class Broken:
        def Pipe(self, duplex):  # noqa: N802 - multiprocessing context API
            return real.Pipe(duplex=duplex)

        def Process(self, **kwargs):  # noqa: N802
            proc = real.Process(**kwargs)

            def start():
                raise OSError("forkserver socket: read-only file system")

            proc.start = start
            return proc

    monkeypatch.setattr(isolation, "_context", lambda: Broken())
    with pytest.raises(RuntimeError, match="cannot start"):
        asyncio.run(isolation.self_check(FAST))
