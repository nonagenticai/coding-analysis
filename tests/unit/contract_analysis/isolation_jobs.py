"""Jobs for the isolation tests. A module (not a test file) so a worker process can import it."""

from __future__ import annotations

import os
import time

from processing.contract_analysis.jobs import call_sites_job


def add(a: int, b: int) -> int:
    return a + b


def fail() -> None:
    raise ValueError("boom")


def die() -> None:
    os._exit(3)


def allocate(mb: int) -> int:
    block = bytearray(mb * 1024 * 1024)
    return len(block)


def pid_then_spin():
    """Yield the worker's pid, then spin forever holding the CPU (and its own GIL)."""
    yield os.getpid()
    while True:
        pass


def first_file_then_spin(files, kinds, max_call_sites):
    """Extract the first file for real, then spin on the second like a pathological parser."""
    yield from call_sites_job(files[:1], kinds, max_call_sites)
    while True:
        pass


def spin_routes(files, max_routes):
    while True:
        pass


def spin_ts_routes(files):
    while True:
        pass


def spin_scan_pair(*args):
    while True:
        pass


def spin_call_sites(files, kinds, max_call_sites):
    """A job that never finishes its first file (e.g. started with no time left)."""
    while True:
        pass


def sleep_then(seconds: float, value: int) -> int:
    time.sleep(seconds)
    return value
