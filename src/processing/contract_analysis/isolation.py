"""Run a CPU-bound extraction job in a separate process with a hard deadline.

Why: the contract endpoints parse attacker-controlled tenant source, and the
service is ONE uvicorn process. In a thread, a super-linear path in any parser
holds the GIL and stalls every tenant's requests, and nothing can stop it. Every
parser is meant to be linear (and is tested to be), but this guard does not
depend on that: the job runs in its own process, which is SIGKILLed at the
deadline, under an address-space cap.

Design:

* One process per job, forked from a ``forkserver`` (spawned clean at first use,
  with the extraction modules preloaded), so a job starts in milliseconds without
  inheriting uvicorn's threads, sockets or event loop. Arguments and results are
  pickled; job functions live in :mod:`processing.contract_analysis.jobs`. The
  usual multiprocessing contract applies: the process's ``__main__`` must be
  import-safe (guarded by ``if __name__ == "__main__"``), which the uvicorn
  console script (the image's CMD) is.
* Results come back over a pipe. A streaming job (a generator) sends one message
  per item, so a job killed at the deadline still returns what it finished.
* The parent waits in a worker thread (blocked in ``poll``, not holding the GIL),
  so the event loop stays free.
* Scheduling (all in the parent, all configurable, see ``config.settings``):

  - **Admission** (:func:`admit`, entered by each endpoint before it loads any
    file): at most ``max_pending`` extraction requests in flight (running +
    waiting for a worker). Past it the request fails ``busy`` at once (HTTP 503 +
    ``Retry-After``), so queued requests cannot pile up request bodies until the
    pod is OOM-killed.
  - **Slots**: at most ``max_workers`` jobs run at once, and at most
    ``max_per_tenant`` of them for one tenant, so one tenant cannot hold every
    worker while others wait.
  - **Queue wait**: a job waits for a slot at most ``queue_wait_s`` and never
    longer than leaves it ``min_run_s`` (at most half) of its deadline (waiting counts against
    the deadline). A job that cannot start with that budget fails ``busy``
    promptly; it is never started with no time left only to be reported
    ``deadline`` with nothing done.

* The child sets ``RLIMIT_AS`` to ``memory_mb`` before running the job (where the
  platform supports it), so a blow-up fails the job with MemoryError instead of
  the pod being OOM-killed.
* A client that disconnects does not stop its job: the worker runs on until it
  finishes or the deadline kills it, so an abandoned request costs at most one
  slot for one deadline.

Entrypoint contract (forkserver): every process that imports this module and
runs a job must guard its ``__main__`` with ``if __name__ == "__main__":``. The
forkserver preloads ``__main__`` (see ``_PRELOAD``); an unguarded main re-runs
in the forkserver and either fails every job or starts a second server. The
image's CMD (the uvicorn console script) is guarded. :func:`self_check` runs a
trivial job at startup (``main.lifespan``) and fails startup loudly when no
worker can be started, rather than failing every extraction request later.

Failure modes, all reported as :class:`IsolationFailedError` with a ``reason``:
``deadline`` (killed), ``busy`` (not admitted, or no slot within the queue wait),
``worker_error`` (the job raised; message attached), ``worker_died`` (no result
and no error: killed by the OS, crashed, or out of memory while reporting).

``isolated=False`` runs the job in a thread of this process instead (tests that
patch a job in-process; never the default).
"""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

#: Imported once in the forkserver, so every job's process starts warm.
#: ``__main__`` first: each child otherwise RE-RUNS the parent's main script as
#: ``__mp_main__`` (multiprocessing's spawn preparation). Under the uvicorn
#: console script that is a guarded no-op, but it costs an import per job, and a
#: main that cannot be re-run would kill every job. Preloaded in the forkserver,
#: the children find it already imported and skip it.
_PRELOAD = [
    "__main__",
    "processing.contract_analysis.isolation",
    "processing.contract_analysis.jobs",
]
_JOIN_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class Limits:
    deadline_s: float
    memory_mb: int  # 0 = no address-space cap
    max_workers: int
    isolated: bool = True
    queue_wait_s: float | None = None  # None = as long as min_run_s allows
    min_run_s: float = 0.0  # never start a job with less of its deadline left
    max_pending: int = 0  # admitted requests (running + waiting); 0 = unlimited
    max_per_tenant: int = 0  # running jobs per tenant; 0 = unlimited

    def run_floor_s(self) -> float:
        """``min_run_s``, but never more than half the deadline (short test deadlines)."""
        return min(self.min_run_s, self.deadline_s / 2)

    def slot_wait_s(self) -> float:
        """How long a job may wait for a slot and still keep its run floor."""
        wait = self.deadline_s - self.run_floor_s()
        if self.queue_wait_s is not None:
            wait = min(wait, self.queue_wait_s)
        return max(wait, 0.0)

    def retry_after_s(self) -> int:
        """A ``Retry-After`` for a busy response: about one job's run time."""
        return max(1, math.ceil(self.queue_wait_s or self.deadline_s))


class IsolationFailedError(Exception):
    """The job produced no (complete) result; ``reason`` is machine-readable."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason  # deadline | busy | worker_error | worker_died
        self.detail = detail


def limits_from_settings() -> Limits:
    from config.settings import get_settings

    settings = get_settings()
    return Limits(
        deadline_s=float(settings.CONTRACT_EXTRACTION_DEADLINE_SECONDS),
        memory_mb=int(settings.CONTRACT_EXTRACTION_MEMORY_MB),
        max_workers=int(settings.CONTRACT_EXTRACTION_MAX_WORKERS),
        isolated=bool(settings.CONTRACT_EXTRACTION_ISOLATED),
        queue_wait_s=float(settings.CONTRACT_EXTRACTION_QUEUE_WAIT_SECONDS),
        min_run_s=float(settings.CONTRACT_EXTRACTION_MIN_RUN_SECONDS),
        max_pending=int(settings.CONTRACT_EXTRACTION_MAX_PENDING),
        max_per_tenant=int(settings.CONTRACT_EXTRACTION_MAX_PER_TENANT),
    )


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------


def _limit_memory(memory_mb: int) -> None:
    if memory_mb <= 0:
        return
    try:
        import resource

        cap = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except (ImportError, ValueError, OSError):  # not enforceable here (e.g. macOS)
        pass


def _child(conn: Any, fn: Callable[..., Any], args: tuple, streaming: bool, memory_mb: int) -> None:
    _limit_memory(memory_mb)
    try:
        if streaming:
            for item in fn(*args):
                conn.send(("item", item))
        else:
            conn.send(("item", fn(*args)))
        conn.send(("done", None))
    except BaseException as exc:  # report everything; the parent decides
        try:
            conn.send(("error", f"{type(exc).__name__}: {str(exc)[:500]}"))
        except BaseException:  # nosec B110 - out of memory/pipe gone: parent sees EOF
            pass
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_CONTEXT: Any = None


def _context() -> Any:
    global _CONTEXT
    with _LOCK:
        if _CONTEXT is None:
            methods = multiprocessing.get_all_start_methods()
            ctx = multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")
            if ctx.get_start_method() == "forkserver":
                ctx.set_forkserver_preload(_PRELOAD)
            _CONTEXT = ctx
        return _CONTEXT


class _Slots:
    """Running jobs, in total and per tenant. Thread-safe (waiters block in threads)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.running = 0
        self.by_tenant: Counter[str] = Counter()

    def _free(self, tenant: str | None, limits: Limits) -> bool:
        if self.running >= max(limits.max_workers, 1):
            return False
        if tenant is None or limits.max_per_tenant <= 0:
            return True
        return self.by_tenant[tenant] < limits.max_per_tenant

    def acquire(self, tenant: str | None, limits: Limits, timeout: float) -> bool:
        end = time.monotonic() + timeout
        with self._cond:
            while not self._free(tenant, limits):
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            self.running += 1
            if tenant is not None:
                self.by_tenant[tenant] += 1
            return True

    def release(self, tenant: str | None) -> None:
        with self._cond:
            self.running -= 1
            if tenant is not None:
                self.by_tenant[tenant] -= 1
                if self.by_tenant[tenant] <= 0:
                    del self.by_tenant[tenant]
            self._cond.notify_all()


_SLOTS = _Slots()
_pending = 0  # admitted requests; mutated on the event loop only (under _LOCK anyway)


@asynccontextmanager
async def admit(limits: Limits | None = None) -> AsyncIterator[None]:
    """Admit one extraction request, or fail ``busy`` at once (see the module doc)."""
    global _pending
    limits = limits or limits_from_settings()
    with _LOCK:
        if limits.max_pending > 0 and _pending >= limits.max_pending:
            raise IsolationFailedError(
                "busy",
                f"{_pending} extraction requests already in flight (max {limits.max_pending})",
            )
        _pending += 1
    try:
        yield
    finally:
        with _LOCK:
            _pending -= 1


def _run_blocking(
    fn: Callable[..., Any],
    args: tuple,
    streaming: bool,
    limits: Limits,
    tenant: str | None = None,
) -> tuple[list[Any], IsolationFailedError | None]:
    """Run *fn* in a child process; ``(items received, failure or None)``. Blocking."""
    start = time.monotonic()
    wait = limits.slot_wait_s()
    if not _SLOTS.acquire(tenant, limits, wait):
        return [], IsolationFailedError("busy", f"no extraction worker free within {wait:g}s")
    try:
        remaining = limits.deadline_s - (time.monotonic() - start)
        if remaining <= 0 or remaining < limits.run_floor_s():
            return [], IsolationFailedError(
                "busy", f"only {max(remaining, 0):.1f}s of the deadline left after waiting"
            )
        end = time.monotonic() + remaining
        ctx = _context()
        recv, send = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_child, args=(send, fn, args, streaming, limits.memory_mb), daemon=True
        )
        try:
            proc.start()
        except Exception as exc:  # e.g. the forkserver could not start
            recv.close()
            send.close()
            return [], IsolationFailedError(
                "worker_died", f"could not start an extraction worker: {type(exc).__name__}: {exc}"
            )
        send.close()  # the child holds the only writer: its exit reads as EOF here
        items: list[Any] = []
        try:
            while True:
                remaining = end - time.monotonic()
                if remaining <= 0 or not recv.poll(remaining):
                    return items, IsolationFailedError(
                        "deadline", f"extraction exceeded the {limits.deadline_s:g}s deadline"
                    )
                try:
                    kind, payload = recv.recv()
                except EOFError:
                    proc.join(_JOIN_TIMEOUT_S)
                    return items, IsolationFailedError(
                        "worker_died",
                        f"extraction worker exited ({proc.exitcode}) without a result",
                    )
                if kind == "item":
                    items.append(payload)
                elif kind == "done":
                    return items, None
                else:
                    return items, IsolationFailedError("worker_error", str(payload))
        finally:
            recv.close()
            if proc.is_alive():
                proc.kill()
            proc.join(_JOIN_TIMEOUT_S)
    finally:
        _SLOTS.release(tenant)


def _run_inline(
    fn: Callable[..., Any], args: tuple, streaming: bool
) -> tuple[list[Any], IsolationFailedError | None]:
    try:
        return (list(fn(*args)) if streaming else [fn(*args)]), None
    except Exception as exc:
        return [], IsolationFailedError("worker_error", f"{type(exc).__name__}: {str(exc)[:500]}")


async def _run(
    fn: Callable[..., Any],
    args: tuple,
    streaming: bool,
    limits: Limits | None,
    tenant: str | None,
) -> tuple[list[Any], IsolationFailedError | None]:
    limits = limits or limits_from_settings()
    if not limits.isolated:
        return await run_in_threadpool(_run_inline, fn, args, streaming)
    items, failure = await run_in_threadpool(_run_blocking, fn, args, streaming, limits, tenant)
    if failure is not None:
        logger.warning("isolated %s failed: %s", getattr(fn, "__name__", fn), failure)
    return items, failure


async def run(
    fn: Callable[..., Any], *args: Any, limits: Limits | None = None, tenant: str | None = None
) -> Any:
    """``fn(*args)`` in a worker process; raises :class:`IsolationFailedError`."""
    items, failure = await _run(fn, args, False, limits, tenant)
    if failure is not None:
        raise failure
    return items[0]


async def stream(
    fn: Callable[..., Iterable[Any]],
    *args: Any,
    limits: Limits | None = None,
    tenant: str | None = None,
) -> tuple[list[Any], IsolationFailedError | None]:
    """Every item a generator job yields, plus why it stopped early (None = finished)."""
    return await _run(fn, args, True, limits, tenant)


async def self_check(limits: Limits | None = None) -> None:
    """Start one worker and run a trivial job; raise RuntimeError if that fails.

    Called at startup, so a forkserver that cannot start (an unguarded
    ``__main__``, no writable temp dir for its socket, a broken preload) fails
    the deploy instead of every extraction request.
    """
    limits = limits or limits_from_settings()
    if not limits.isolated:
        return
    probe = Limits(
        deadline_s=max(limits.deadline_s, 30.0),
        memory_mb=limits.memory_mb,
        max_workers=limits.max_workers,
        queue_wait_s=max(limits.deadline_s, 30.0),
    )
    items, failure = await run_in_threadpool(_run_blocking, os.getpid, (), False, probe, None)
    if failure is not None or not items or items[0] == os.getpid():
        raise RuntimeError(
            f"contract extraction worker cannot start ({failure or 'ran in-process'}); "
            "see processing/contract_analysis/isolation.py"
        )
    logger.info("contract extraction worker self-check passed (pid %s)", items[0])
