"""Message coupling (#135) on real fixture source files: Redis pub/sub, streams and
celery / rq / arq tasks, Python and the Rust redis crate, matched by channel name."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from models.structure_models import ServiceGraphResponse
from processing.structure.messaging import extract_messaging
from processing.structure.service_graph import build_service_graph, split_by_repo

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "messaging_workspace"
WORKER = ".repos/acme__worker"
RS = ".repos/acme__rs"
OTHER = ".repos/acme__other"
MONO = ".repos/acme__mono"


def _files() -> dict[str, str]:
    return {
        p.relative_to(FIXTURE).as_posix(): p.read_text()
        for p in sorted(FIXTURE.rglob("*"))
        if p.is_file()
    }


@pytest.fixture(scope="module")
def graph():
    return build_service_graph(split_by_repo(_files()))


def test_message_edges(graph):
    got = [
        (e.from_repo, e.from_file, e.from_line, e.kind, e.from_channel, e.from_via, e.match,
         e.to_repo, e.to_file, e.to_line, e.to_via)
        for e in graph.message_edges
    ]  # fmt: skip
    assert got == [
        (RS, "src/main.rs", 6, "pubsub", "alerts", "redis-rs.publish", "pattern",
         RS, "src/main.rs", 15, "redis-rs.psubscribe"),
        (RS, "src/main.rs", 7, "pubsub", "user.*", "redis-rs.publish", "pattern",
         WORKER, "worker/listener.py", 10, "redis.psubscribe"),
        (RS, "src/main.rs", 8, "stream", "payments", "redis-rs.cmd(XADD)", "exact",
         WORKER, "worker/listener.py", 12, "redis.xreadgroup"),
        (RS, "src/main.rs", 9, "pubsub", "RS_CHAN", "redis-rs.publish", "unresolved",
         None, None, None, None),
        ("primary", "app/events.py", 18, "pubsub", "orders.created", "redis.publish", "exact",
         RS, "src/main.rs", 14, "redis-rs.subscribe"),
        ("primary", "app/events.py", 18, "pubsub", "orders.created", "redis.publish", "exact",
         WORKER, "worker/listener.py", 9, "redis.subscribe"),
        ("primary", "app/events.py", 19, "pubsub", "user.*", "redis.publish", "pattern",
         WORKER, "worker/listener.py", 10, "redis.psubscribe"),
        ("primary", "app/events.py", 20, "pubsub", "AUDIT_CHANNEL", "redis.publish", "unresolved",
         None, None, None, None),
        ("primary", "app/events.py", 22, "stream", "payments", "redis.xadd", "exact",
         WORKER, "worker/listener.py", 12, "redis.xreadgroup"),
        ("primary", "app/events.py", 23, "task", "billing.charge", "celery.send_task", "exact",
         WORKER, "worker/tasks.py", 5, "celery.task"),
        ("primary", "app/events.py", 24, "task", "worker.tasks.send_receipt", "celery.delay",
         "exact", WORKER, "worker/tasks.py", 10, "celery.task"),
        ("primary", "app/events.py", 25, "task", "worker.jobs.resize_image", "rq.enqueue",
         "exact", WORKER, "worker/jobs.py", 1, "rq.function"),
        ("primary", "app/events.py", 31, "task", "crunch", "arq.enqueue_job", "exact",
         WORKER, "worker/arq_worker.py", 6, "arq.function"),
        # non-src layout: the module name follows __init__.py packages, not the path
        ("primary", "app/events.py", 46, "task", "billing.tasks.refund", "celery.delay", "exact",
         MONO, "packages/billing/billing/tasks.py", 5, "celery.task"),
    ]  # fmt: skip
    unresolved = [e for e in graph.message_edges if e.match == "unresolved"]
    assert {e.reason for e in unresolved} == {"channel name comes from an environment variable"}
    assert graph.message_edges[0].from_symbol == "send"
    assert graph.message_edges[4].from_symbol == "order_created"


def test_unmatched_sides(graph):
    assert [(e.repo, e.file, e.line, e.channel, e.via) for e in graph.unmatched_publishers] == [
        ("primary", "app/events.py", 21, "nobody.listens", "redis.publish"),
        # arq job named like a CELERY task (worker.tasks.send_receipt): no cross-framework match
        ("primary", "app/events.py", 36, "send_receipt", "arq.enqueue_job"),
        # worker.tasks.process exists nowhere; acme__other's root tasks.process is not it
        ("primary", "app/events.py", 40, "worker.tasks.process", "celery.delay"),
    ]
    assert [(e.repo, e.file, e.line, e.channel) for e in graph.unmatched_consumers] == [
        (OTHER, "tasks.py", 5, "tasks.process"),
        (WORKER, "worker/listener.py", 11, "lonely"),
        (WORKER, "worker/tasks.py", 15, "worker.tasks.never_called"),
    ]


def test_repo_counts(graph):
    counts = {s.repo: (s.message_publishers, s.message_consumers) for s in graph.repos}
    # acme__other: only the celery task; the Invoke @task is not a consumer.
    # acme__worker: rx ``of(1).subscribe(on_next=print)`` is not a channel.
    assert counts == {"primary": (12, 0), WORKER: (0, 8), RS: (4, 2), OTHER: (0, 1), MONO: (0, 1)}


def test_negative_controls_are_not_endpoints():
    """``post.publish`` in a file without a redis import and ``Pool().apply_async``
    are not messaging; a file that does not parse is reported, not dropped."""
    files = {p: t for p, t in split_by_repo(_files())["primary"].items()}
    files["app/broken.py"] = "def (:\n"
    found = extract_messaging("primary", files)
    assert not [e for e in found.endpoints if e.file == "app/blog.py"]
    assert not [e for e in found.endpoints if e.via == "celery.apply_async"]
    assert list(found.unparsed) == ["app/broken.py"]


def test_invoke_task_and_keyword_subscribe_are_not_consumers():
    repos = split_by_repo(_files())
    other = extract_messaging(OTHER, repos[OTHER])
    assert [(e.file, e.channel, e.via) for e in other.endpoints] == [
        ("tasks.py", "tasks.process", "celery.task")
    ]
    worker = extract_messaging(WORKER, repos[WORKER])
    assert not [e for e in worker.endpoints if e.line == 13]  # of(1).subscribe(on_next=print)
    assert "on_next" not in {e.channel for e in worker.endpoints}


def test_response_model_carries_message_fields(graph):
    body = ServiceGraphResponse.model_validate({**asdict(graph), "files_read": 0}).model_dump()
    assert len(body["message_edges"]) == 14
    assert "aliases" not in body["unmatched_consumers"][0]
    assert body["message_edges"][3]["from_channel"] == "RS_CHAN"
    assert body["unmatched_publishers"][0]["channel"] == "nobody.listens"
    assert body["unmatched_consumers"][0]["role"] == "consumer"
