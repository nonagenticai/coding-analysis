"""py2rust #11: Rust quality checks as developer feedback.

``language: "rust"`` (explicit) runs ``cargo fmt --all -- --check`` and
``cargo clippy --workspace --all-targets --message-format=json`` in the requested
crate dir and parses them into the findings schema; a path-scoped request filters
the findings instead of skipping the tools. The authoritative Rust evaluation is
the rust test runner Job (coding-testing), not this.

The analyzer tests run the REAL toolchain on the small crates in
``tests/fixtures/rust_quality``: ``dirty`` plants clippy::needless_return in
library code, clippy::bool_assert_comparison in TEST code (only
``--all-targets`` lints it), two lints on one line (``two_lints.rs``), a one-line
and a multi-line unformatted file; ``clean`` is the negative
control. They are skipped where cargo is not installed.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.api.endpoints.quality import QualityAnalysisSyncRequest
from processing.architecture.quality_analyzer import ArchitectureQualityAnalyzer
from tests.unit.quality_store_fakes import FakeQualityResultStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "rust_quality"
_TENANT = "0195f48b-0000-7000-8000-00000000000a"
_ANALYZER_CLIENT = "processing.architecture.quality_analyzer.get_coding_containers_client"
_TOOL_CLIENT = "processing.architecture.tool_execution._base.get_coding_containers_client"
_VALIDATE_CONTAINER = "app.api.endpoints.quality.validate_and_get_container_for_tenant"
_RESOLVE_SHA = "app.api.endpoints.quality.resolve_commit_sha"

LIB = "crates/core/src/lib.rs"
UNFORMATTED = "crates/core/src/unformatted.rs"
SPREAD = "crates/core/src/spread.rs"  # formatted, then one unformatted fn on line 9
TWO = "crates/core/src/two_lints.rs"  # len_zero + needless_return on line 2
MAC = "crates/core/src/macro_calls.rs"  # deprecated call in a macro, invoked on 17/21/25
DIRTY = {
    ("clippy", LIB, 6),
    ("clippy", LIB, 17),
    ("clippy", TWO, 2),
    ("clippy", MAC, 17),
    ("clippy", MAC, 21),
    ("clippy", MAC, 25),
    ("rustfmt", UNFORMATTED, 1),
    ("rustfmt", SPREAD, 6),  # rustfmt's hunk starts 3 context lines above line 9
    ("rustfmt", TWO, 1),
}


def _cargo() -> str | None:
    """The cargo this host can actually run, or None.

    Presence is not usability. On this estate ``~/.local/bin/cargo`` is a shim that
    REFUSES inside any Cargo project (Rust builds and tests belong on m1), so a
    ``Path.exists()`` probe says "present" on a host where every one of these tests
    then fails on the shim's refusal — the class was un-runnable off m1 and CI yet
    did not skip it. Probing inside a throwaway manifest is the only honest test:
    it skips exactly where the shim applies.
    """
    found = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
    if not Path(found).exists():
        return None
    with tempfile.TemporaryDirectory() as probe:
        probe_path = Path(probe)
        (probe_path / "Cargo.toml").write_text(
            '[package]\nname = "cargo-probe"\nversion = "0.0.0"\nedition = "2021"\n'
        )
        (probe_path / "src").mkdir()
        (probe_path / "src" / "lib.rs").write_text("")
        try:
            result = subprocess.run(
                [found, "--version"],
                cwd=probe_path,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
    return found if result.returncode == 0 else None


class _LocalExec:
    """coding-containers' exec, run for real on this host with /workspace mapped
    onto a local copy of a fixture (a crate, or a whole multi-repo workspace)."""

    def __init__(self, root: Path, target_dir: Path) -> None:
        self.root = root
        self.env = {**os.environ, "CARGO_TARGET_DIR": str(target_dir)}
        cargo = _cargo()
        if cargo:
            self.env["PATH"] = f"{Path(cargo).parent}:{self.env.get('PATH', '')}"
        self.calls: list[tuple[list[str], str]] = []

    def _map(self, value: str) -> str:
        return value.replace("/workspace", str(self.root))

    async def execute(self, container_name: str, command: list[str], **kw: Any) -> dict[str, Any]:
        workdir = kw.get("workdir", "/workspace")
        self.calls.append((list(command), workdir))
        proc = await asyncio.create_subprocess_exec(
            *[self._map(a) for a in command],
            cwd=self._map(workdir),
            env=self.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=280)
        unmap = lambda b: b.decode().replace(str(self.root), "/workspace")  # noqa: E731
        return {"exit_code": proc.returncode, "stdout": unmap(out), "stderr": unmap(err)}

    async def read_file(self, container_name: str, file_path: str, **_: Any) -> str | None:
        p = Path(self._map(file_path))
        return p.read_text() if p.is_file() else None

    def cargo_cwds(self, sub: str) -> set[str]:
        return {w for c, w in self.calls if c[:2] == ["cargo", sub]}


def _workspace(tmp_path: Path, fixture: str) -> _LocalExec:
    root = tmp_path / fixture
    shutil.copytree(FIXTURES / fixture, root)
    return _LocalExec(root, tmp_path / f"target-{fixture}")


async def _analyze(ex: _LocalExec, **kwargs: Any) -> Any:
    analyzer = ArchitectureQualityAnalyzer()
    with patch(_ANALYZER_CLIENT, return_value=ex), patch(_TOOL_CLIENT, return_value=ex):
        return await analyzer.analyze(Path("/workspace"), "ws-1", _TENANT, **kwargs)


def _gate(report: Any) -> set[tuple[str, str, int]]:
    return {
        (i.tool, i.file_path, i.line_number)
        for i in report.issues
        if i.tool in ("clippy", "rustfmt")
    }


def _container() -> MagicMock:
    c = MagicMock()
    c.codebase_path = "/workspace"
    c.repo_url = "https://github.com/acme/repo"
    return c


def _client(analyzer: Any, store: FakeQualityResultStore):
    from coding_client.api.exception_handlers import register_exception_handlers
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.api.endpoints.quality import get_quality_analyzer, router
    from auth.tenant import require_tenant
    from services.quality_result_store import get_quality_result_store

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router, prefix="/quality")
    app.dependency_overrides[require_tenant] = lambda: _TENANT
    app.dependency_overrides[get_quality_analyzer] = lambda: analyzer
    app.dependency_overrides[get_quality_result_store] = lambda: store
    return TestClient(app)


_live = pytest.mark.skipif(_cargo() is None, reason="no cargo toolchain on this host")


@_live
class TestLiveToolchain:
    @pytest.mark.asyncio
    async def test_dirty_crate_findings_and_clean_control(self, tmp_path: Path):
        dirty = await _analyze(_workspace(tmp_path, "dirty"), language="rust")
        clean = await _analyze(_workspace(tmp_path, "clean"), language="rust")
        assert _gate(dirty) == DIRTY
        assert {i.repo for i in dirty.issues} == {"primary"}
        assert _gate(clean) == set()
        assert {"clippy", "rustfmt"} <= set(clean.tools_succeeded)
        assert clean.inconclusive is False

    @pytest.mark.asyncio
    async def test_two_lints_on_one_line_are_both_reported(self, tmp_path: Path):
        report = await _analyze(_workspace(tmp_path, "dirty"), language="rust")
        rules = {i.rule_id for i in report.issues if i.file_path == TWO and i.line_number == 2}
        assert rules == {"clippy::len_zero", "clippy::needless_return"}

    @pytest.mark.asyncio
    async def test_macro_diagnostics_are_reported_at_each_call_site(self, tmp_path: Path):
        """rustc points each diagnostic at the macro body (line 12); every
        invocation is its own finding, at its call site."""
        report = await _analyze(_workspace(tmp_path, "dirty"), language="rust")
        assert sorted((i.line_number, i.rule_id) for i in report.issues if i.file_path == MAC) == [
            (17, "deprecated"),
            (21, "deprecated"),
            (25, "deprecated"),
        ]

    @pytest.mark.asyncio
    async def test_unbuildable_crate_is_a_failed_clippy_not_a_clean_one(self, tmp_path: Path):
        """``broken`` names a missing path dependency: cargo exits 101 with no
        finding, which is no verdict, never a 0-issue pass."""
        report = await _analyze(_workspace(tmp_path, "broken"), language="rust")
        assert "clippy" in report.tools_failed
        assert not any(i.tool == "clippy" for i in report.issues)

    @pytest.mark.asyncio
    async def test_out_of_repo_path_dependency_is_not_reported(self, tmp_path: Path):
        """``pathdep/repo`` (the repo) depends on ``pathdep/outside``, which is not
        formatted. ``cargo fmt --all`` checks it too; its hunks are not the repo's."""
        shutil.copytree(FIXTURES / "pathdep", tmp_path / "pathdep")
        ex = _LocalExec(tmp_path / "pathdep" / "repo", tmp_path / "target-pathdep")
        report = await _analyze(ex, language="rust")
        assert [i.file_path for i in report.issues if i.tool == "rustfmt"] == []
        assert "rustfmt" in report.tools_succeeded

    @pytest.mark.asyncio
    async def test_detection_alone_does_not_run_cargo(self, tmp_path: Path):
        """Negative control: a Python majority wins detection; ``language`` overrides it."""
        ex = _workspace(tmp_path, "dirty")
        for n in range(5):
            (ex.root / f"helper_{n}.py").write_text("x = 1\n")
        report = await _analyze(ex)
        assert report.language == "python" and ex.cargo_cwds("clippy") == set()
        assert _gate(await _analyze(ex, language="rust")) == DIRTY

    @pytest.mark.asyncio
    async def test_workdir_is_the_cargo_cwd_and_paths_stay_repo_relative(self, tmp_path: Path):
        ex = _workspace(tmp_path, "dirty")
        report = await _analyze(ex, language="rust", workdir="crates/core")
        assert ex.cargo_cwds("clippy") == ex.cargo_cwds("fmt") == {"/workspace/crates/core"}
        assert _gate(report) == DIRTY

    @pytest.mark.asyncio
    async def test_path_scope_filters_findings_instead_of_skipping(self, tmp_path: Path):
        report = await _analyze(
            _workspace(tmp_path, "dirty"), language="rust", scope_paths=[UNFORMATTED]
        )
        assert _gate(report) == {("rustfmt", UNFORMATTED, 1)}
        assert {"clippy", "rustfmt"} <= set(report.tools_succeeded)

    @pytest.mark.asyncio
    async def test_scoped_clean_after_a_build_error_is_a_failure(self, tmp_path: Path):
        """A compile error outside the scope stops cargo before the scope is linted:
        zero in-scope findings is then not "clean"."""
        ex = _workspace(tmp_path, "dirty")
        lib = ex.root / LIB
        lib.write_text(lib.read_text().replace("return x * 2;", 'return x * "2";'))
        report = await _analyze(ex, language="rust", scope_paths=[UNFORMATTED])
        assert "clippy" in report.tools_failed
        unscoped = await _analyze(ex, language="rust")
        assert "clippy" in unscoped.tools_succeeded
        assert any(i.tool == "clippy" and i.severity == "high" for i in unscoped.issues)

    def test_analyze_sync_route(self, tmp_path: Path):
        ex = _workspace(tmp_path, "dirty")
        store = FakeQualityResultStore()
        with (
            patch(_VALIDATE_CONTAINER, new_callable=AsyncMock, return_value=_container()),
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=None),
            patch(_ANALYZER_CLIENT, return_value=ex),
            patch(_TOOL_CLIENT, return_value=ex),
        ):
            r = _client(ArchitectureQualityAnalyzer(), store).post(
                "/quality/analyze-sync",
                json={
                    "container_name": "ws-1",
                    "language": "rust",
                    "workdir": "crates/core",
                    "paths": [LIB],
                },
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert (body["language"], body["workdir"], body["repo"]) == (
            "rust",
            "crates/core",
            "primary",
        )
        assert sorted(
            (i["tool"], i["file_path"], i["line_number"], i["repo"])
            for i in body["issues"]
            if i["tool"] in ("clippy", "rustfmt")
        ) == [("clippy", LIB, 6, "primary"), ("clippy", LIB, 17, "primary")]
        assert body["persisted"] is False and store.save_calls == []

    def test_workdir_run_is_not_stored(self, tmp_path: Path):
        """``workdir`` alone (no ``paths``, no ``tools``) keeps the run out of the store."""
        ex = _workspace(tmp_path, "dirty")
        store = FakeQualityResultStore()
        with (
            patch(_VALIDATE_CONTAINER, new_callable=AsyncMock, return_value=_container()),
            patch(_RESOLVE_SHA, new_callable=AsyncMock, return_value=None),
            patch(_ANALYZER_CLIENT, return_value=ex),
            patch(_TOOL_CLIENT, return_value=ex),
        ):
            client = _client(ArchitectureQualityAnalyzer(), store)
            body = {"container_name": "ws-1", "language": "rust"}
            r = client.post("/quality/analyze-sync", json={**body, "workdir": "crates/core"})
            plain = client.post("/quality/analyze-sync", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["persisted"] is False
        assert any(i["file_path"] == LIB for i in r.json()["issues"])
        # Negative control: the same run without workdir is stored.
        assert plain.json()["persisted"] is True and len(store.save_calls) == 1


class TestAnalyzeSyncRequest:
    @pytest.mark.parametrize(
        "payload",
        [
            {"container_name": "ws-1", "workdir": "crates/core"},  # workdir needs rust
            {"container_name": "ws-1", "language": "python", "workdir": "x"},
            {"container_name": "ws-1", "language": "rust", "workdir": "../up"},
            {"container_name": "ws-1", "language": "rust", "workdir": "/abs"},
        ],
    )
    def test_invalid_workdir_is_rejected(self, payload: dict[str, Any]):
        with pytest.raises(ValidationError):
            QualityAnalysisSyncRequest(**payload)

    def test_unchanged_default(self):
        req = QualityAnalysisSyncRequest(container_name="w")
        assert (req.language, req.workdir, req.repo) == (None, None, "primary")
        assert not req.rust_override
        assert QualityAnalysisSyncRequest(container_name="w", language=" Rust ").rust_override
