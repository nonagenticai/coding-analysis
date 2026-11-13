"""Bug hunt completion callback: allowlist (fail closed), exactly-once, never fails the hunt."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException, Response

from app.api.endpoints import bug_hunter
from app.api.endpoints.bug_hunter import _run_claimed_hunt, callback_url_allowed, queue_bug_hunt
from app.api.models.bug_hunter_api_models import BugHuntRequest
from models.analysis_models import BugHuntingResponse
from services.bug_hunt_report_store import BugHuntReportStore

PREFIX = "http://203.0.113.10:5678/webhook-waiting/"
GOOD = PREFIX + "4711"


def _settings(prefixes: str) -> SimpleNamespace:
    return SimpleNamespace(
        bug_hunt_callback_prefixes=tuple(p.strip() for p in prefixes.split(",") if p.strip())
    )


@pytest.fixture
def allow_prefix():
    with patch.object(bug_hunter, "get_settings", return_value=_settings(PREFIX)):
        yield


def _terminal_job(status: str = "completed") -> dict:
    now = datetime.now(UTC)
    return {
        "hunt_id": "hunt-1",
        "container_name": "repo-1",
        "tenant_id": "tenant-1",
        "status": status,
        "errors_found": 2,
        "fixes_generated": 1,
        "error_message": None if status == "completed" else "boom",
        "created_at": now,
        "started_at": now,
        "completed_at": now,
        "updated_at": now,
    }


def _store() -> MagicMock:
    store = MagicMock(spec=BugHuntReportStore)
    store.renew_lease = AsyncMock(return_value=True)
    store.complete_hunt = AsyncMock()
    store.fail_hunt = AsyncMock(return_value=True)
    store.get_hunt = AsyncMock(return_value=_terminal_job())
    store.create_hunt = AsyncMock(return_value=_terminal_job("queued"))
    return store


def _claimed(callback_url: str | None) -> dict:
    job = _terminal_job("running")
    job["request_payload"] = BugHuntRequest(callback_url=callback_url).model_dump(mode="json")
    return job


def _report() -> BugHuntingResponse:
    return BugHuntingResponse(
        errors_found=0,
        fixes_generated=0,
        root_cause_analysis="clean",
        fix_summary="none",
        estimated_effort="none",
        risk_assessment="low",
        errors=[],
        resolves=[],
        pull_request=None,
        analysis_time=0.1,
    )


class _Recorder:
    """Stands in for httpx.AsyncClient; records POSTs and returns scripted results."""

    def __init__(self, results: list):
        self.results = list(results)
        self.posts: list[tuple[str, dict]] = []

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json):
        self.posts.append((url, json))
        r = self.results.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(status_code=r)


# ── allowlist ─────────────────────────────────────────────────────────


def test_no_prefixes_configured_refuses_everything():
    with patch.object(bug_hunter, "get_settings", return_value=_settings("")):
        assert callback_url_allowed(GOOD) is False


@pytest.mark.parametrize(
    "url",
    [
        GOOD,
        PREFIX + "4711?signature=abc123",  # n8n appends its resume token as a query param
        PREFIX + "4711/my-suffix",  # Wait node with a webhook suffix
    ],
)
def test_allowed_urls(allow_prefix, url):
    assert callback_url_allowed(url) is True


@pytest.mark.parametrize(
    "tail",
    [
        "../webhook/x",  # httpx collapses this and POSTs /webhook/x (unauthenticated n8n webhook)
        "./../rest/x",
        "4711/../../rest/x",
        "%2e%2e/webhook/x",
        "4711%2F..%2Fx",
        "..\\webhook\\x",
        "4711//x",  # empty segment
        "",  # the bare prefix, no execution id
        "4711/",
    ],
)
def test_tail_after_prefix_must_be_plain_segments(allow_prefix, tail):
    assert callback_url_allowed(PREFIX + tail) is False


def test_fragment_is_refused(allow_prefix):
    assert callback_url_allowed(GOOD + "#frag") is False


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "http://10.43.0.10:8080/internal",  # off-prefix: an in-cluster SSRF target
        "http://203.0.113.10:5678/rest/workflows",  # same host, other path
        "http://203.0.113.10:5678@evil.example/webhook-waiting/1",  # userinfo trick
        "file:///etc/passwd",
        "ftp://203.0.113.10:5678/webhook-waiting/1",
    ],
)
def test_refused_urls(allow_prefix, url):
    assert callback_url_allowed(url) is False


@pytest.mark.parametrize(
    "url",
    [
        "http://203.0.113.10:5678@evil.example/x",  # userinfo: the real host is evil.example
        "http://203.0.113.10:56789/x",  # port extended past the prefix
        "http://203.0.113.10:5678.evil.example/x",  # host extended past the prefix
        "https://203.0.113.10:5678/x",  # scheme differs
    ],
)
def test_origin_only_prefix_cannot_be_extended(url):
    # A prefix with no path is where a raw string startswith() would be exploitable.
    with patch.object(
        bug_hunter, "get_settings", return_value=_settings("http://203.0.113.10:5678")
    ):
        assert callback_url_allowed(url) is False
        assert callback_url_allowed("http://203.0.113.10:5678/webhook-waiting/1") is True


async def test_queue_refuses_disallowed_callback_before_persisting(allow_prefix):
    store = _store()
    with pytest.raises(HTTPException) as exc:
        await queue_bug_hunt(
            response=Response(),
            container_name="repo-1",
            request=BugHuntRequest(callback_url="http://10.43.0.10/x"),
            tenant_id="tenant-1",
            store=store,
        )
    assert exc.value.status_code == 422
    store.create_hunt.assert_not_awaited()


async def test_queue_persists_allowed_callback(allow_prefix):
    store = _store()
    manager = MagicMock()
    manager.get_container = AsyncMock(return_value=SimpleNamespace(work_dir="/repo"))
    with patch.object(bug_hunter, "get_container_manager", return_value=manager):
        await queue_bug_hunt(
            response=Response(),
            container_name="repo-1",
            request=BugHuntRequest(callback_url=GOOD),
            tenant_id="tenant-1",
            store=store,
        )
    assert store.create_hunt.await_args.kwargs["request_payload"]["callback_url"] == GOOD


# ── worker: exactly once, never fails the hunt ────────────────────────


async def _run(job: dict, store: MagicMock, recorder: _Recorder, *, hunt=None):
    manager = MagicMock()
    manager.get_container = AsyncMock(return_value=SimpleNamespace(work_dir="/repo"))
    with (
        patch.object(bug_hunter, "hunt_bugs", new=hunt or AsyncMock(return_value=_report())),
        patch.object(bug_hunter, "get_container_manager", return_value=manager),
        patch.object(bug_hunter.httpx, "AsyncClient", new=recorder),
        patch.object(bug_hunter.asyncio, "sleep", new=AsyncMock()),
    ):
        await _run_claimed_hunt(job=job, worker_id="w1", store=store)


async def test_completed_hunt_posts_final_status_once(allow_prefix):
    store, rec = _store(), _Recorder([200])
    await _run(_claimed(GOOD), store, rec)
    assert len(rec.posts) == 1
    url, body = rec.posts[0]
    assert url == GOOD
    assert body["status"] == "completed"
    assert body["hunt_id"] == "hunt-1"
    assert body["links"]["summary"].endswith("/hunts/hunt-1/report")


async def test_failed_hunt_posts_failed_status(allow_prefix):
    store, rec = _store(), _Recorder([200])
    store.get_hunt.return_value = _terminal_job("failed")
    await _run(_claimed(GOOD), store, rec, hunt=AsyncMock(side_effect=RuntimeError("boom")))
    assert [b["status"] for _, b in rec.posts] == ["failed"]


async def test_no_callback_when_another_worker_owns_the_transition(allow_prefix):
    store, rec = _store(), _Recorder([200])
    store.fail_hunt.return_value = False  # lease moved: this worker changed nothing
    await _run(_claimed(GOOD), store, rec, hunt=AsyncMock(side_effect=RuntimeError("boom")))
    assert rec.posts == []


async def test_no_callback_when_complete_did_not_persist(allow_prefix):
    store, rec = _store(), _Recorder([200])
    store.complete_hunt.side_effect = RuntimeError("Running bug hunt 'hunt-1' was not found")
    store.fail_hunt.return_value = False
    await _run(_claimed(GOOD), store, rec)
    assert rec.posts == []


async def test_no_callback_url_means_no_post(allow_prefix):
    store, rec = _store(), _Recorder([200])
    await _run(_claimed(None), store, rec)
    assert rec.posts == []
    store.get_hunt.assert_not_awaited()


async def test_retries_5xx_and_transport_errors_then_stops(allow_prefix):
    store, rec = _store(), _Recorder([503, httpx.ConnectError("down"), 200])
    await _run(_claimed(GOOD), store, rec)
    assert len(rec.posts) == 3


async def test_undeliverable_callback_never_fails_the_hunt(allow_prefix):
    store, rec = _store(), _Recorder([500, 500, 500])
    await _run(_claimed(GOOD), store, rec)
    assert len(rec.posts) == 3
    store.complete_hunt.assert_awaited_once()
    store.fail_hunt.assert_not_awaited()


async def test_4xx_is_final_not_retried(allow_prefix):
    store, rec = _store(), _Recorder([404])
    await _run(_claimed(GOOD), store, rec)
    assert len(rec.posts) == 1


async def test_prefix_narrowed_after_queue_blocks_send():
    store, rec = _store(), _Recorder([200])
    with patch.object(bug_hunter, "get_settings", return_value=_settings("http://other/")):
        await _run(_claimed(GOOD), store, rec)
    assert rec.posts == []


async def test_3xx_is_not_followed_or_counted_as_delivered(allow_prefix, caplog):
    caplog.set_level(logging.INFO, logger=bug_hunter.logger.name)
    store, rec = _store(), _Recorder([302])
    await _run(_claimed(GOOD), store, rec)
    assert len(rec.posts) == 1
    assert "callback refused (HTTP 302)" in caplog.text
    assert "delivered" not in caplog.text


def test_dot_segment_really_is_collapsed_by_httpx():
    # Pins the premise of the traversal guard: if httpx ever stops normalising
    # dot-segments this still documents why the tail regex exists.
    req = httpx.Request("POST", PREFIX + "../webhook/x")
    assert req.url.path == "/webhook/x"


@pytest.mark.parametrize(
    "url",
    [
        GOOD + "\n",
        GOOD + "\t",
        " " + GOOD,
        "\t" + GOOD,
        GOOD + "\x7f",
        GOOD + "?q=a b",
        "http://[::1/webhook-waiting/1",  # urlsplit raises ValueError: must be False, not a 500
        PREFIX + "1" * 2048,  # over the length cap
    ],
)
def test_malformed_or_unsendable_urls_are_refused(allow_prefix, url):
    assert callback_url_allowed(url) is False


async def test_malformed_callback_is_a_422_not_a_500(allow_prefix):
    with pytest.raises(HTTPException) as exc:
        await queue_bug_hunt(
            response=Response(),
            container_name="repo-1",
            request=BugHuntRequest(callback_url="http://[::1/webhook-waiting/1"),
            tenant_id="tenant-1",
            store=_store(),
        )
    assert exc.value.status_code == 422
