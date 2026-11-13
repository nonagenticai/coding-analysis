"""Workspace reads go through coding-containers, confined and tenant-scoped.

Covers ``app.api.utils.workspace_files`` and the three static-analysis ops that
used to fall back to ``os.getcwd()`` (``/app`` -- this service's own source):
``analyze_signatures``, ``generate_call_graph``, ``generate_documentation``.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException

from app.api.endpoints.static_analysis import (
    analyze_signatures,
    generate_call_graph,
    generate_documentation,
)
from app.api.utils.workspace_files import (
    INVALID_WORKSPACE_PATH,
    WORKSPACE_UNRESOLVABLE,
    InvalidWorkspacePathError,
    WorkspaceUnresolvableError,
    invalid_workspace_path_http,
    materialize_workspace,
    normalize_workspace_path,
    remove_materialized,
    tree_digest,
    workspace_root,
    workspace_unresolvable_http,
)
from models.static_analysis_models import (
    CallGraphRequest,
    DocumentationRequest,
    SignatureAnalysisRequest,
)

_WS_CLIENT = "app.api.utils.workspace_files.get_coding_containers_client"
_VALIDATE = "app.api.endpoints.static_analysis.validate_and_get_container_for_tenant"

TENANT_A = "0195f48b-d000-7000-8000-00000000000a"

#: A function name that exists nowhere in coding-analysis's own tree, so seeing
#: it in the output proves the WORKSPACE was analysed and not ``/app``.
WORKSPACE_SOURCE = (
    "def zz_workspace_only_marker(a: int, b: int) -> int:\n"
    '    """Only the tenant workspace has this."""\n'
    "    return a + b\n"
)


def _fake_client(files: dict[str, str]) -> MagicMock:
    """A coding-containers client whose workspace holds exactly *files*."""
    client = MagicMock()

    async def _execute(container_name, command, workdir="/workspace", timeout=30, tenant_id=None):
        if command and command[0] == "find":
            return {"exit_code": 0, "stdout": "\n".join(files)}
        return {"exit_code": 1, "stdout": "", "stderr": "unexpected"}

    async def _read_file(container_name, file_path, tenant_id=None):
        return files.get(file_path)

    client.execute = AsyncMock(side_effect=_execute)
    client.read_file = AsyncMock(side_effect=_read_file)
    return client


# ---------------------------------------------------------------------------
# normalize_workspace_path -- the traversal guard
# ---------------------------------------------------------------------------


class TestNormalizeWorkspacePath:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("src/a.py", "src/a.py"),
            ("./src/a.py", "src/a.py"),
            ("src//a.py", "src/a.py"),
            ("/workspace/src/a.py", "src/a.py"),
            ("a.py", "a.py"),
        ],
    )
    def test_accepts_confined_paths(self, raw: str, expected: str) -> None:
        assert normalize_workspace_path(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "../etc/passwd",
            "src/../../etc/passwd",
            "src/..",
            "/etc/passwd",
            "/app/src/main.py",
            "/workspace/../etc/passwd",
            "/workspacex/a.py",
            "src\\..\\a.py",
            "a\x00.py",
            "",
            "   ",
            ".",
            "/workspace",
            "a" * 2000,
        ],
    )
    def test_refuses_escapes(self, raw: str) -> None:
        with pytest.raises(InvalidWorkspacePathError):
            normalize_workspace_path(raw)

    def test_custom_root(self) -> None:
        assert normalize_workspace_path("/repo/x/y.py", "/repo") == "x/y.py"
        with pytest.raises(InvalidWorkspacePathError):
            normalize_workspace_path("/workspace/x.py", "/repo")


class TestHelpers:
    def test_workspace_root(self) -> None:
        assert workspace_root(SimpleNamespace(codebase_path="/repo/")) == "/repo"
        assert workspace_root(SimpleNamespace(codebase_path=None)) == "/workspace"
        assert workspace_root(None) == "/workspace"

    def test_http_forms_are_typed(self) -> None:
        exc = workspace_unresolvable_http(WorkspaceUnresolvableError("ws", "nothing"))
        assert exc.status_code == 422
        assert exc.detail["error_code"] == WORKSPACE_UNRESOLVABLE
        assert exc.detail["details"] == {"container_name": "ws", "reason": "nothing"}
        bad = invalid_workspace_path_http(InvalidWorkspacePathError("nope"))
        assert bad.status_code == 400
        assert bad.detail["error_code"] == INVALID_WORKSPACE_PATH

    def test_tree_digest_tracks_content(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1\n")
        first = tree_digest(tmp_path)
        (tmp_path / "a.py").write_text("x = 2\n")
        assert tree_digest(tmp_path) != first

    def test_remove_materialized_tolerates_none(self, tmp_path: Path) -> None:
        remove_materialized(None)
        d = tmp_path / "d"
        d.mkdir()
        remove_materialized(d)
        assert not d.exists()


# ---------------------------------------------------------------------------
# materialize_workspace
# ---------------------------------------------------------------------------


class TestMaterializeWorkspace:
    async def test_rel_paths_read_with_tenant(self) -> None:
        client = _fake_client({"/workspace/src/a.py": "x = 1\n"})
        with patch(_WS_CLIENT, return_value=client):
            root = await materialize_workspace("ws", tenant_id=TENANT_A, rel_paths=["src/a.py"])
        try:
            assert (root / "src" / "a.py").read_text() == "x = 1\n"
            client.read_file.assert_awaited_once_with(
                "ws", "/workspace/src/a.py", tenant_id=TENANT_A
            )
        finally:
            remove_materialized(root)

    async def test_traversal_refused_before_any_read(self) -> None:
        client = _fake_client({})
        with patch(_WS_CLIENT, return_value=client), pytest.raises(InvalidWorkspacePathError):
            await materialize_workspace("ws", tenant_id=TENANT_A, rel_paths=["../../etc/passwd"])
        client.read_file.assert_not_awaited()

    async def test_nothing_readable_is_typed_error(self) -> None:
        with (
            patch(_WS_CLIENT, return_value=_fake_client({})),
            pytest.raises(WorkspaceUnresolvableError),
        ):
            await materialize_workspace("ws", tenant_id=TENANT_A, rel_paths=["a.py"])

    async def test_one_requested_file_missing_is_404_not_partial(self) -> None:
        from app.api.utils.workspace_files import (
            WORKSPACE_FILE_NOT_FOUND,
            WorkspaceFileNotFoundError,
            workspace_file_not_found_http,
        )

        client = _fake_client({"/workspace/a.py": "x = 1\n"})
        created: list[str] = []
        real_mkdtemp = __import__("tempfile").mkdtemp

        def _mkdtemp(*args, **kwargs):
            created.append(real_mkdtemp(*args, **kwargs))
            return created[-1]

        with (
            patch(_WS_CLIENT, return_value=client),
            patch("app.api.utils.workspace_files.tempfile.mkdtemp", side_effect=_mkdtemp),
            pytest.raises(WorkspaceFileNotFoundError) as exc_info,
        ):
            await materialize_workspace("ws", tenant_id=TENANT_A, rel_paths=["a.py", "gone.py"])
        assert exc_info.value.missing == ["gone.py"]
        assert created and not Path(created[0]).exists()  # temp dir removed
        http = workspace_file_not_found_http(exc_info.value)
        assert http.status_code == 404
        assert http.detail["error_code"] == WORKSPACE_FILE_NOT_FOUND

    async def test_read_raising_midway_removes_temp_dir(self) -> None:
        client = _fake_client({"/workspace/a.py": "x = 1\n"})
        client.read_file = AsyncMock(side_effect=["x = 1\n", RuntimeError("boom")])
        created: list[str] = []
        real_mkdtemp = __import__("tempfile").mkdtemp

        def _mkdtemp(*args, **kwargs):
            created.append(real_mkdtemp(*args, **kwargs))
            return created[-1]

        with (
            patch(_WS_CLIENT, return_value=client),
            patch("app.api.utils.workspace_files.tempfile.mkdtemp", side_effect=_mkdtemp),
            pytest.raises(RuntimeError),
        ):
            await materialize_workspace("ws", tenant_id=TENANT_A, rel_paths=["a.py", "b.py"])
        assert created and not Path(created[0]).exists()

    async def test_pattern_read_raising_midway_removes_temp_dir(self) -> None:
        client = _fake_client({"/workspace/a.py": "x = 1\n", "/workspace/b.py": "y = 2\n"})
        client.read_file = AsyncMock(side_effect=["x = 1\n", RuntimeError("boom")])
        created: list[str] = []
        real_mkdtemp = __import__("tempfile").mkdtemp

        def _mkdtemp(*args, **kwargs):
            created.append(real_mkdtemp(*args, **kwargs))
            return created[-1]

        with (
            patch(_WS_CLIENT, return_value=client),
            patch("processing.architecture.container_fs.tempfile.mkdtemp", side_effect=_mkdtemp),
            pytest.raises(RuntimeError),
        ):
            await materialize_workspace("ws", tenant_id=TENANT_A, name_patterns=("*.py",))
        assert created and not Path(created[0]).exists()

    async def test_pattern_discovered_unreadable_file_is_skipped_and_counted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Control: pattern discovery may skip, but logs how many it skipped."""
        client = _fake_client({"/workspace/a.py": "x = 1\n", "/workspace/b.py": ""})
        with patch(_WS_CLIENT, return_value=client), caplog.at_level("WARNING"):
            root = await materialize_workspace("ws", tenant_id=TENANT_A, name_patterns=("*.py",))
        try:
            assert (root / "a.py").exists()
            assert not (root / "b.py").exists()
            assert any("Skipped 1 of 2" in r.getMessage() for r in caplog.records)
        finally:
            remove_materialized(root)

    async def test_empty_rel_paths_is_typed_error(self) -> None:
        with (
            patch(_WS_CLIENT, return_value=_fake_client({})),
            pytest.raises(WorkspaceUnresolvableError),
        ):
            await materialize_workspace("ws", tenant_id=TENANT_A, rel_paths=[])

    async def test_patterns_mirror_with_tenant(self) -> None:
        client = _fake_client({"/workspace/pkg/m.py": "y = 2\n"})
        with patch(_WS_CLIENT, return_value=client):
            root = await materialize_workspace("ws", tenant_id=TENANT_A, name_patterns=("*.py",))
        try:
            assert (root / "pkg" / "m.py").read_text() == "y = 2\n"
            assert client.execute.await_args.kwargs["tenant_id"] == TENANT_A
            assert client.read_file.await_args.kwargs["tenant_id"] == TENANT_A
        finally:
            remove_materialized(root)

    async def test_patterns_nothing_found_is_typed_error(self) -> None:
        with (
            patch(_WS_CLIENT, return_value=_fake_client({})),
            pytest.raises(WorkspaceUnresolvableError),
        ):
            await materialize_workspace("ws", tenant_id=TENANT_A, name_patterns=("*.py",))

    async def test_no_patterns_is_typed_error(self) -> None:
        with (
            patch(_WS_CLIENT, return_value=_fake_client({})),
            pytest.raises(WorkspaceUnresolvableError),
        ):
            await materialize_workspace("ws", tenant_id=TENANT_A)


# ---------------------------------------------------------------------------
# The three ops analyse the WORKSPACE, never /app
# ---------------------------------------------------------------------------


def _container() -> SimpleNamespace:
    return SimpleNamespace(codebase_path="/workspace", validated=True)


class TestStaticOpsAnalyseTheWorkspace:
    async def test_signatures_come_from_the_workspace(self) -> None:
        client = _fake_client({"/workspace/pkg/mod.py": WORKSPACE_SOURCE})
        with (
            patch(_VALIDATE, new=AsyncMock(return_value=_container())),
            patch(_WS_CLIENT, return_value=client),
        ):
            result = await analyze_signatures(
                request=SignatureAnalysisRequest(container_name="ws", language="python"),
                background_tasks=BackgroundTasks(),
                redis_client=None,
                tenant_id=TENANT_A,
            )
        names = {f.name if hasattr(f, "name") else f["name"] for f in result.functions}
        assert names == {"zz_workspace_only_marker"}
        assert result.total_functions == 1
        # Every read carried the caller's tenant.
        assert client.execute.await_args.kwargs["tenant_id"] == TENANT_A
        assert client.read_file.await_args.kwargs["tenant_id"] == TENANT_A

    async def test_signatures_unresolvable_workspace_is_422_not_cwd(self) -> None:
        with (
            patch(_VALIDATE, new=AsyncMock(return_value=_container())),
            patch(_WS_CLIENT, return_value=_fake_client({})),
            patch("os.getcwd", side_effect=AssertionError("must never fall back to cwd")),
            pytest.raises(HTTPException) as exc_info,
        ):
            await analyze_signatures(
                request=SignatureAnalysisRequest(container_name="ws", language="python"),
                background_tasks=BackgroundTasks(),
                redis_client=None,
                tenant_id=TENANT_A,
            )
        assert exc_info.value.status_code == 422
        assert exc_info.value.detail["error_code"] == WORKSPACE_UNRESOLVABLE

    async def test_call_graph_is_fed_the_materialised_workspace(self) -> None:
        client = _fake_client({"/workspace/pkg/mod.py": WORKSPACE_SOURCE})
        seen: dict[str, str] = {}

        async def _generate(repo_path, config):
            seen["repo_path"] = repo_path
            seen["listing"] = sorted(
                str(p.relative_to(repo_path)) for p in Path(repo_path).rglob("*.py")
            )
            response = MagicMock()
            response.model_dump_json.return_value = "{}"
            return response

        generator = MagicMock()
        generator.generate_comprehensive_call_graph = AsyncMock(side_effect=_generate)
        with (
            patch(_VALIDATE, new=AsyncMock(return_value=_container())),
            patch(_WS_CLIENT, return_value=client),
            patch(
                "app.api.endpoints.static_analysis.CallGraphGenerator",
                return_value=generator,
            ),
        ):
            await generate_call_graph(
                request=CallGraphRequest(container_name="ws"),
                background_tasks=BackgroundTasks(),
                redis_client=None,
                tenant_id=TENANT_A,
            )
        assert seen["repo_path"] != os.getcwd()
        assert seen["listing"] == [str(Path("pkg") / "mod.py")]

    async def test_call_graph_unresolvable_is_422(self) -> None:
        with (
            patch(_VALIDATE, new=AsyncMock(return_value=_container())),
            patch(_WS_CLIENT, return_value=_fake_client({})),
            pytest.raises(HTTPException) as exc_info,
        ):
            await generate_call_graph(
                request=CallGraphRequest(container_name="ws"),
                background_tasks=BackgroundTasks(),
                redis_client=None,
                tenant_id=TENANT_A,
            )
        assert exc_info.value.status_code == 422

    async def test_documentation_unresolvable_is_422(self) -> None:
        with (
            patch(_VALIDATE, new=AsyncMock(return_value=_container())),
            patch(_WS_CLIENT, return_value=_fake_client({})),
            pytest.raises(HTTPException) as exc_info,
        ):
            await generate_documentation(
                request=DocumentationRequest(container_name="ws", doc_types=["architecture"]),
                background_tasks=BackgroundTasks(),
                _redis_client=None,
                tenant_id=TENANT_A,
            )
        assert exc_info.value.status_code == 422

    async def test_cross_tenant_container_never_materialised(self) -> None:
        client = _fake_client({"/workspace/pkg/mod.py": WORKSPACE_SOURCE})
        with (
            patch(
                _VALIDATE,
                new=AsyncMock(side_effect=HTTPException(status_code=404, detail="not found")),
            ),
            patch(_WS_CLIENT, return_value=client),
            pytest.raises(HTTPException) as exc_info,
        ):
            await analyze_signatures(
                request=SignatureAnalysisRequest(container_name="ws", language="python"),
                background_tasks=BackgroundTasks(),
                redis_client=None,
                tenant_id="0195f48b-d000-7000-8000-00000000000b",
            )
        assert exc_info.value.status_code == 404
        client.read_file.assert_not_awaited()
        client.execute.assert_not_awaited()


# --- materialize_tenant_workspace: the shared container-mode entry point ---


async def test_materialize_tenant_workspace_exact_lookup_then_materialise(tmp_path):
    from app.api.utils import workspace_files as wf

    info = SimpleNamespace(codebase_path="/workspace", validated=True)
    with (
        patch(
            "app.api.utils.container_validation.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            return_value=info,
        ) as validate,
        patch.object(wf, "materialize_workspace", new_callable=AsyncMock, return_value=tmp_path),
    ):
        root = await wf.materialize_tenant_workspace("cv-ems-p4", "t1", name_patterns=("*.py",))
    assert root == tmp_path
    validate.assert_awaited_once_with("cv-ems-p4", "t1")


async def test_materialize_tenant_workspace_reuses_given_container_info(tmp_path):
    from app.api.utils import workspace_files as wf

    info = SimpleNamespace(codebase_path="/srv/repo")
    with (
        patch(
            "app.api.utils.container_validation.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
        ) as validate,
        patch.object(
            wf, "materialize_workspace", new_callable=AsyncMock, return_value=tmp_path
        ) as mat,
    ):
        await wf.materialize_tenant_workspace(
            "c", "t1", name_patterns=("*.py",), container_info=info
        )
    validate.assert_not_awaited()
    assert mat.call_args.kwargs["root"] == "/srv/repo"


async def test_materialize_tenant_workspace_unresolvable_is_422_no_fallback():
    from app.api.utils import workspace_files as wf

    with (
        patch(
            "app.api.utils.container_validation.validate_and_get_container_for_tenant",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(codebase_path="/workspace"),
        ),
        patch.object(
            wf,
            "materialize_workspace",
            new_callable=AsyncMock,
            side_effect=wf.WorkspaceUnresolvableError("c", "nothing readable"),
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            await wf.materialize_tenant_workspace("c", "t1", name_patterns=("*.py",))
    assert exc.value.status_code == 422
