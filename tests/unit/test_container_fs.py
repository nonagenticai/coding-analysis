"""Unit tests for processing.architecture.container_fs.materialize_from_container.

The repo under analysis lives inside the per-job coding-containers container, not
on the analyzer pod's local fs. This helper mirrors selected files out via the
container's whitelisted ``find`` + ``read_file`` so sync Path-based scanners work.
"""

from __future__ import annotations

import shutil
from unittest.mock import AsyncMock, MagicMock

import pytest

from processing.architecture.container_fs import (
    DEFAULT_EXCLUDE_GLOBS,
    materialize_from_container,
    safe_relative,
)


def _client(find_stdout: str, files: dict[str, str | None]) -> MagicMock:
    client = MagicMock()
    client.execute = AsyncMock(return_value={"exit_code": 0, "stdout": find_stdout})

    async def _read(container_name: str, file_path: str, tenant_id: str | None = None):
        return files.get(file_path)

    client.read_file = AsyncMock(side_effect=_read)
    return client


class TestMaterializeFromContainer:
    async def test_copies_files_preserving_relative_paths(self):
        client = _client(
            "/workspace/main.go\n/workspace/pkg/util.go\n",
            {
                "/workspace/main.go": "package main\n",
                "/workspace/pkg/util.go": "package pkg\n",
            },
        )
        tmp = await materialize_from_container(client, "cid", "/workspace", ("*.go",))
        try:
            assert tmp is not None
            assert (tmp / "main.go").read_text() == "package main\n"
            assert (tmp / "pkg" / "util.go").read_text() == "package pkg\n"
        finally:
            import shutil

            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)

    async def test_find_command_is_parenthesised_with_excludes(self):
        client = _client("", {})
        await materialize_from_container(
            client, "cid", "/srv", ("*.py", "*.js"), ("*/node_modules/*",)
        )
        container_arg = client.execute.call_args.args[0]
        find_cmd = client.execute.call_args.args[1]
        assert container_arg == "cid"
        assert find_cmd[0] == "find"
        assert find_cmd[1] == "/srv"
        # name alternation parenthesised so -not filters apply to every pattern
        assert "(" in find_cmd and ")" in find_cmd
        assert "*.py" in find_cmd and "*.js" in find_cmd
        assert "-o" in find_cmd
        assert "-not" in find_cmd and "*/node_modules/*" in find_cmd
        # workdir pinned to the scan root
        assert client.execute.call_args.kwargs.get("workdir") == "/srv"

    async def test_returns_none_when_find_empty(self):
        client = _client("", {})
        assert await materialize_from_container(client, "cid", "/workspace", ("*.py",)) is None

    async def test_returns_none_when_no_patterns(self):
        client = _client("/workspace/a.py\n", {"/workspace/a.py": "x = 1\n"})
        assert await materialize_from_container(client, "cid", "/workspace", ()) is None
        client.execute.assert_not_called()

    async def test_returns_none_when_all_reads_empty(self):
        client = _client(
            "/workspace/a.py\n/workspace/b.py\n",
            {"/workspace/a.py": None, "/workspace/b.py": ""},
        )
        assert await materialize_from_container(client, "cid", "/workspace", ("*.py",)) is None

    async def test_default_excludes_applied_when_unspecified(self):
        client = _client("", {})
        await materialize_from_container(client, "cid", "/workspace", ("*.py",))
        find_cmd = client.execute.call_args.args[1]
        for excl in DEFAULT_EXCLUDE_GLOBS:
            assert excl in find_cmd

    async def test_exec_and_read_carry_the_owning_tenant(self):
        """Every container call must present the tenant that OWNS the workspace.

        coding-containers resolves a call with no ``X-Tenant-Id`` as the caller's
        own service account (``svc-coding-analysis``), and its BOLA guard then
        answers 404 "Container not found" for a workspace owned by a customer —
        which is how every static-analysis exec failed in ~1.2 ms while
        ``GET /api/workspaces/<name>`` returned 200 for the same name.
        """
        tenant = "0195f48b-d000-7000-8000-000000000003"
        client = _client("/workspace/a.py\n", {"/workspace/a.py": "x = 1\n"})
        tmp = await materialize_from_container(
            client, "cid", "/workspace", ("*.py",), tenant_id=tenant
        )
        try:
            assert client.execute.call_args.kwargs.get("tenant_id") == tenant, (
                "materialize_from_container exec'd `find` with NO tenant: "
                "coding-containers resolves that as the service account and its "
                "BOLA guard returns 404 for the customer's workspace"
            )
            assert client.read_file.call_args.kwargs.get("tenant_id") == tenant, (
                "materialize_from_container read a file with NO tenant: "
                "the same BOLA guard 404s the read"
            )
        finally:
            import shutil

            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)

    async def test_absent_tenant_stays_absent(self):
        """No tenant supplied means none asserted -- never a substituted literal.

        ``optional_tenant_headers`` omits ``X-Tenant-Id`` for a falsy tenant; a
        defaulted one would attribute the call to a tenant that does not own the
        container.
        """
        client = _client("/workspace/a.py\n", {"/workspace/a.py": "x = 1\n"})
        tmp = await materialize_from_container(client, "cid", "/workspace", ("*.py",))
        try:
            assert client.execute.call_args.kwargs.get("tenant_id") is None
            assert client.read_file.call_args.kwargs.get("tenant_id") is None
        finally:
            import shutil

            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)

    @pytest.mark.parametrize(
        "evil",
        ["/workspace/../../escape.py", "/workspace/a/../../escape.py", "/elsewhere/escape.py"],
    )
    async def test_listed_path_outside_root_is_never_written(self, evil, tmp_path, monkeypatch):
        """``find`` output comes from the tenant's container: a listed ``../`` path
        used to be joined onto the temp dir and written outside it."""
        work = tmp_path / "tmpdirs"
        work.mkdir()
        monkeypatch.setattr("tempfile.tempdir", str(work))
        client = _client(
            f"{evil}\n/workspace/ok.py\n", {evil: "pwned\n", "/workspace/ok.py": "1\n"}
        )
        tmp = await materialize_from_container(client, "cid", "/workspace", ("*.py",))
        try:
            assert tmp is not None
            assert [p.name for p in tmp_path.rglob("*.py")] == ["ok.py"]
            assert [c.args[1] for c in client.read_file.call_args_list] == ["/workspace/ok.py"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_safe_relative(self):
        assert safe_relative("/workspace/a/b.py", "/workspace") == "a/b.py"
        assert safe_relative("/workspace/a/./b.py", "/workspace/") == "a/b.py"
        assert safe_relative("/workspace/a/../b.py", "/workspace") == "b.py"
        assert safe_relative("/workspace/../x", "/workspace") is None
        assert safe_relative("/workspace", "/workspace") is None
        assert safe_relative("/workspacex/y", "/workspace") is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
