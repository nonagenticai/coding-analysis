"""Read a tenant's workspace through coding-containers -- never this pod's filesystem.

The repository under analysis lives inside the tenant's per-repo workspace pod
(``/workspace``), which coding-containers owns. This service cannot see it on
its own disk. Two defects followed from forgetting that:

* ``analyzeSignatures`` / ``generateCallGraph`` / ``generateDocumentation``
  called a registry method that does not exist, caught the ``AttributeError``
  and fell back to ``os.getcwd()`` -- ``/app``, this service's OWN source -- so
  every container got a confident answer about coding-analysis.
* ``createDependencyVisualization`` parsed whatever absolute path the caller
  named, on this pod.

Everything that needs workspace content goes through here instead:

* paths are repo-RELATIVE and confined (:func:`normalize_workspace_path`) --
  no absolute paths, no ``..``, no NUL;
* every read carries the caller's tenant, because coding-containers' BOLA
  guard answers 404 for an untenanted read of a customer workspace;
* nothing readable is a typed failure (:class:`WorkspaceUnresolvableError`),
  never a silent fallback to some local directory.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from fastapi import HTTPException, status

from app.api.responses import error_response
from clients.coding_containers import get_coding_containers_client
from processing.workspace_repos import repo_root

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE_ROOT = "/workspace"
WORKSPACE_UNRESOLVABLE = "WORKSPACE_UNRESOLVABLE"
WORKSPACE_FILE_NOT_FOUND = "WORKSPACE_FILE_NOT_FOUND"
INVALID_WORKSPACE_PATH = "INVALID_WORKSPACE_PATH"
_MAX_PATH_LEN = 1024


class InvalidWorkspacePathError(ValueError):
    """A path that is not a confined, repo-relative workspace path."""


class WorkspaceUnresolvableError(RuntimeError):
    """The workspace could not be read through coding-containers.

    Raised instead of falling back to any local directory: a fallback answers a
    different question with the same confidence, which is worse than no answer.
    """

    def __init__(self, container_name: str, reason: str) -> None:
        super().__init__(f"Workspace '{container_name}' could not be resolved: {reason}")
        self.container_name = container_name
        self.reason = reason


class WorkspaceFileNotFoundError(WorkspaceUnresolvableError):
    """One or more explicitly REQUESTED files could not be read from the workspace.

    Never a silent skip: analysing the rest would answer a narrower question than
    the one asked, with the same confidence.
    """

    def __init__(self, container_name: str, missing: list[str]) -> None:
        super().__init__(container_name, f"file not found in workspace: {', '.join(missing)}")
        self.missing = missing


def workspace_file_not_found_http(exc: WorkspaceFileNotFoundError) -> HTTPException:
    """The HTTP form of :class:`WorkspaceFileNotFoundError` (404, typed code)."""
    return error_response(
        status_code=status.HTTP_404_NOT_FOUND,
        message=str(exc),
        error_code=WORKSPACE_FILE_NOT_FOUND,
        details={"container_name": exc.container_name, "missing": exc.missing},
    )


def workspace_unresolvable_http(exc: WorkspaceUnresolvableError) -> HTTPException:
    """The HTTP form of :class:`WorkspaceUnresolvableError` (422, typed code)."""
    return error_response(
        status_code=422,
        message=str(exc),
        error_code=WORKSPACE_UNRESOLVABLE,
        details={"container_name": exc.container_name, "reason": exc.reason},
    )


def invalid_workspace_path_http(exc: InvalidWorkspacePathError) -> HTTPException:
    """The HTTP form of :class:`InvalidWorkspacePathError` (400, typed code)."""
    return error_response(
        status_code=status.HTTP_400_BAD_REQUEST,
        message=str(exc),
        error_code=INVALID_WORKSPACE_PATH,
    )


def workspace_root(container_info: object | None) -> str:
    """The workspace root inside the container (``codebase_path``, else ``/workspace``)."""
    raw = getattr(container_info, "codebase_path", None) if container_info is not None else None
    root = str(raw or DEFAULT_WORKSPACE_ROOT).rstrip("/")
    return root or DEFAULT_WORKSPACE_ROOT


def normalize_workspace_path(path: str, root: str = DEFAULT_WORKSPACE_ROOT) -> str:
    """Return *path* as a normalised repo-relative POSIX path, or raise.

    Accepted: ``src/a.py``, ``./src/a.py``, and an absolute path that is
    already under *root* (``/workspace/src/a.py`` -> ``src/a.py``).
    Refused: any other absolute path, any ``..`` segment, NUL, backslashes,
    empty, over-long.
    """
    if not isinstance(path, str) or not path.strip():
        raise InvalidWorkspacePathError("Workspace path must be a non-empty string")
    if len(path) > _MAX_PATH_LEN:
        raise InvalidWorkspacePathError("Workspace path is too long")
    if "\x00" in path or "\\" in path:
        raise InvalidWorkspacePathError(f"Workspace path contains a forbidden character: {path!r}")

    candidate = PurePosixPath(path)
    if candidate.is_absolute():
        root_path = PurePosixPath(root.rstrip("/") or DEFAULT_WORKSPACE_ROOT)
        try:
            candidate = candidate.relative_to(root_path)
        except ValueError:
            raise InvalidWorkspacePathError(
                f"Workspace path must be relative to the workspace root: {path}"
            ) from None

    parts = [p for p in candidate.parts if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise InvalidWorkspacePathError(f"Workspace path escapes the workspace: {path}")
    if not parts:
        raise InvalidWorkspacePathError(f"Workspace path names no file: {path}")
    return "/".join(parts)


def tree_digest(local_root: Path) -> str:
    """Content digest of a materialised tree (relative path + bytes, sorted).

    Used in cache keys so a cached result can never outlive an edit to the
    workspace and can never be served for different content.
    """
    digest = hashlib.sha256()
    for file_path in sorted(p for p in local_root.rglob("*") if p.is_file()):
        digest.update(file_path.relative_to(local_root).as_posix().encode())
        digest.update(b"\x00")
        digest.update(file_path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


async def materialize_workspace(
    container_name: str,
    *,
    tenant_id: str | None,
    root: str = DEFAULT_WORKSPACE_ROOT,
    name_patterns: Iterable[str] = (),
    rel_paths: Iterable[str] | None = None,
    repo: str | None = None,
) -> Path:
    """Copy workspace files into a fresh local temp dir and return it.

    With *rel_paths*, exactly those (normalised, confined) files are read.
    Otherwise every file matching *name_patterns* under *root* is mirrored,
    EXCLUDING the source checkouts under ``.repos/``.

    *repo* (``processing.workspace_repos``) picks which repository of the
    workspace: ``None``/``"primary"`` = *root* itself, ``".repos/<name>"`` = that
    checkout, which then becomes the root every path is relative to.
    The caller owns the returned directory and must remove it
    (:func:`remove_materialized`).

    Raises:
        InvalidWorkspacePathError: a requested path is not confined.
        WorkspaceFileNotFoundError: a requested *rel_paths* file could not be read
            (the temp dir is removed first).
        WorkspaceUnresolvableError: nothing could be read.
    """
    client = get_coding_containers_client()
    root = repo_root(root, repo)

    if rel_paths is not None:
        normalized = [normalize_workspace_path(p, root) for p in rel_paths]
        if not normalized:
            raise WorkspaceUnresolvableError(container_name, "no file paths were requested")
        local_root = Path(tempfile.mkdtemp(prefix="ws-"))
        try:
            missing: list[str] = []
            for rel in dict.fromkeys(normalized):
                content = await client.read_file(
                    container_name, f"{root}/{rel}", tenant_id=tenant_id
                )
                if content is None:
                    missing.append(rel)
                    continue
                dest = local_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            if missing:
                # Every requested file is required: no partial analysis.
                raise WorkspaceFileNotFoundError(container_name, missing)
        except BaseException:
            remove_materialized(local_root)
            raise
        return local_root

    # Imported here, not at module import: ``processing.architecture`` pulls in
    # the language-processor factory, which endpoint modules (and their tests)
    # that only need the path guard must not have to load.
    from processing.architecture.container_fs import materialize_from_container

    patterns = tuple(name_patterns)
    if not patterns:
        raise WorkspaceUnresolvableError(container_name, "no file patterns to match")
    materialized = await materialize_from_container(
        client, container_name, root, patterns, tmp_prefix="ws-", tenant_id=tenant_id
    )
    if materialized is None:
        raise WorkspaceUnresolvableError(
            container_name, f"no readable files matching {', '.join(patterns)}"
        )
    return materialized


def remove_materialized(path: Path | str | None) -> None:
    """Remove a directory created by :func:`materialize_workspace`."""
    if path:
        shutil.rmtree(path, ignore_errors=True)


async def materialize_tenant_workspace(
    container_name: str,
    tenant_id: str | None,
    *,
    name_patterns: Iterable[str] = (),
    rel_paths: Iterable[str] | None = None,
    container_info: object | None = None,
    repo: str | None = None,
) -> Path:
    """Resolve ``container_name`` for ``tenant_id`` EXACTLY and mirror its workspace.

    The one path every container-mode analysis endpoint should take:

    * the container is looked up by exact name under the caller's tenant
      (coding-containers' tenant-scoped lookup; 404 if absent, 403 if another
      tenant's) -- never a substring/prefix match;
    * its files are read out of the workspace pod through coding-containers and
      written to a fresh local temp dir -- never this pod's filesystem, and
      never a fallback directory.

    Pass ``container_info`` when the caller already ran
    ``validate_and_get_container_for_tenant`` (avoids a second lookup).

    The caller owns the returned directory and must :func:`remove_materialized` it.

    Raises:
        HTTPException: 404/403 from the lookup, 400 for an unconfined path,
            404 ``WORKSPACE_FILE_NOT_FOUND`` when a requested file cannot be read,
            422 ``WORKSPACE_UNRESOLVABLE`` when nothing could be read.
    """
    if container_info is None:
        from .container_validation import validate_and_get_container_for_tenant

        container_info = await validate_and_get_container_for_tenant(container_name, tenant_id)
    try:
        return await materialize_workspace(
            container_name,
            tenant_id=tenant_id,
            root=workspace_root(container_info),
            name_patterns=name_patterns,
            rel_paths=rel_paths,
            repo=repo,
        )
    except InvalidWorkspacePathError as exc:
        raise invalid_workspace_path_http(exc) from exc
    except WorkspaceFileNotFoundError as exc:
        raise workspace_file_not_found_http(exc) from exc
    except WorkspaceUnresolvableError as exc:
        raise workspace_unresolvable_http(exc) from exc


def materialized_files(local_root: Path) -> list[str]:
    """Every file under a materialised tree, sorted (deterministic order)."""
    return sorted(str(p) for p in local_root.rglob("*") if p.is_file())


class MaterializedFiles(list):
    """Local file paths that remember the materialised temp dir they live in.

    A plain ``list`` to every consumer; the owner calls :meth:`release` (or
    :func:`remove_materialized` on ``materialized_root``) when done.
    """

    def __init__(self, files: Iterable, *, materialized_root: Path | None = None) -> None:
        super().__init__(files)
        self.materialized_root = materialized_root

    def release(self) -> None:
        remove_materialized(self.materialized_root)


def release_materialized(files: object) -> None:
    """Remove the temp dir behind ``files`` if it is :class:`MaterializedFiles`."""
    remove_materialized(getattr(files, "materialized_root", None))
