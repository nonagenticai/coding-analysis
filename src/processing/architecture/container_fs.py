"""Shared helper for reading repository files out of the analysis container.

`coding-analysis` runs as a pod; the repository under analysis lives inside a
per-job container managed by `coding-containers` (at ``/workspace``), **not** on
the analyzer pod's local filesystem. Any code that needs to scan the repo with
sync, ``Path``-based logic must therefore mirror the relevant files out of the
container into a local temp dir first, then run against that copy.

This module factors out that mechanic — a whitelisted ``find`` to enumerate the
target files, ``read_file`` per match to fetch content — so both the dependency
analyzer (manifests) and the architecture-quality analyzer (source files) share
one implementation. The coding-containers client is passed in by the caller so
each call site keeps its own patchable ``get_coding_containers_client`` symbol.
"""

from __future__ import annotations

import logging
import posixpath
import shutil
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from processing.workspace_repos import repos_exclude_globs, scope_enters_repos

if TYPE_CHECKING:
    from clients.coding_containers import CodingContainersClient

logger = logging.getLogger(__name__)

# Directories never worth copying out of the container — build artefacts, vendored
# trees and VCS metadata that would balloon the copy without informing analysis.
DEFAULT_EXCLUDE_GLOBS: tuple[str, ...] = (
    "*/node_modules/*",
    "*/.git/*",
    "*/target/*",
    "*/.venv/*",
    "*/dist/*",
    "*/build/*",
    "*/vendor/*",
)


def safe_relative(abs_path: str, root: str) -> str | None:
    """*abs_path* relative to *root*, or None when it is not strictly inside *root*.

    The path is ``find`` output from a tenant container, so it is untrusted:
    ``/workspace/../../etc/x`` normalises outside *root* and is rejected.
    """
    prefix = posixpath.normpath(root).rstrip("/") + "/"
    norm = posixpath.normpath(abs_path)
    return norm[len(prefix) :] if norm.startswith(prefix) else None


class ScopeMatchesNoFilesError(ValueError):
    """A path-scoped analysis where one or more paths match no analysable file.

    Raised (never swallowed into a ``None`` report) so the caller can answer
    "your scope is empty" instead of returning an empty-but-successful report.
    ``unmatched`` names the requested paths that matched nothing — a scope with
    one real file and one typo is refused, not silently narrowed to the real one
    (radon, for one, prints ``{}`` and exits 0 for a missing target).
    """

    def __init__(self, message: str, unmatched: Sequence[str] | None = None) -> None:
        super().__init__(message)
        self.unmatched: list[str] | None = list(unmatched) if unmatched is not None else None


def _build_find_command(
    root: str | Sequence[str], name_patterns: Iterable[str], exclude_globs: Iterable[str]
) -> list[str]:
    """Build a whitelisted ``find`` argv: matching files of any given name
    pattern, minus excluded path components.

    ``coding-containers`` whitelists ``find`` but not ``sh``/pipes, so the whole
    query is expressed as a single argv. The name alternation is parenthesised so
    the ``-type``/``-not`` filters apply to every pattern, not just the last one.
    ``root`` may be several starting points (a path-scoped run): ``find`` walks
    each, so files outside them are never enumerated, let alone fetched.
    """
    roots = [root] if isinstance(root, str) else list(root)
    find_cmd: list[str] = ["find", *roots, "-type", "f", "("]
    for idx, name in enumerate(name_patterns):
        if idx:
            find_cmd.append("-o")
        find_cmd += ["-name", name]
    find_cmd.append(")")
    for excl in exclude_globs:
        find_cmd += ["-not", "-path", excl]
    return find_cmd


async def materialize_from_container(
    client: CodingContainersClient,
    container_name: str,
    root: str = "/workspace",
    name_patterns: Iterable[str] = (),
    exclude_globs: Iterable[str] = DEFAULT_EXCLUDE_GLOBS,
    *,
    tmp_prefix: str = "matrepo-",
    tenant_id: str | None = None,
    scope_paths: Sequence[str] | None = None,
    exclude_repos: bool = True,
    existing_scope_ok: bool = False,
) -> Path | None:
    """Copy files matching ``name_patterns`` out of ``container_name`` into a
    local temp dir, preserving paths relative to ``root``.

    Args:
        client: A coding-containers client (resolved/patched by the caller).
        container_name: Target container (the codebase id).
        root: Directory inside the container to scan. Defaults to ``/workspace``.
        name_patterns: ``find -name`` values (e.g. ``"*.py"``, ``"pyproject.toml"``).
        exclude_globs: ``find -not -path`` values for directories to skip.
        tmp_prefix: Prefix for the created temp dir.
        tenant_id: Tenant that OWNS the container, threaded from the
            authenticated caller. The workspace is tenant-scoped, so an exec /
            read that arrives without it is resolved by coding-containers as
            this service's own account and refused by its BOLA guard with a 404
            ("Container not found") — indistinguishable from a missing
            container. Absent stays absent (never defaulted to a literal): see
            ``clients.coding_containers._identity_headers``.
        scope_paths: Optional ``root``-relative files/directories (already
            validated: relative, no ``..``). When given, ``find`` starts at
            those paths instead of ``root``, so only files inside the scope are
            enumerated and read. Paths keep their ``root``-relative layout.
        exclude_repos: Keep the walk out of ``<root>/.repos/`` -- the source
            checkouts of a multi-repo workspace are NOT part of the repository
            rooted at ``root`` (``processing.workspace_repos``). Skipped when
            ``scope_paths`` explicitly names something under ``.repos/``: an
            explicit scope is the caller choosing those files.

    Every listed path is untrusted (it is ``find`` output from the tenant's
    container): one that is not strictly inside ``root`` after normalisation
    (``/workspace/../../etc/x``) is never written.

    Returns:
        The temp dir path (the **caller must clean it up**), or ``None`` if the
        container yielded no matching files / no readable content.

    Raises:
        ScopeMatchesNoFilesError: ``scope_paths`` was given and at least one of
            them matched no file; ``unmatched`` lists those paths. Nothing is
            read in that case.
    """
    patterns = tuple(name_patterns)
    if not patterns:
        return None

    roots: list[str] = [f"{root.rstrip('/')}/{p}" for p in scope_paths] if scope_paths else [root]
    excludes = list(exclude_globs)
    if exclude_repos and not scope_enters_repos(scope_paths):
        excludes += repos_exclude_globs(root)
    find_cmd = _build_find_command(roots, patterns, excludes)
    result = await client.execute(container_name, find_cmd, workdir=root, tenant_id=tenant_id)
    stdout = result.get("stdout") or ""
    paths = [line.strip() for line in stdout.splitlines() if line.strip()]
    if scope_paths:
        # Check every requested path, not just the union: a single typo among
        # real paths must not degrade into a quietly narrower run.
        unmatched = [
            p
            for p, r in zip(scope_paths, roots, strict=True)
            if not any(f == r.rstrip("/") or f.startswith(r.rstrip("/") + "/") for f in paths)
        ]
        if unmatched and existing_scope_ok:
            # One more listing, of the unmatched paths themselves (not their files).
            gone = {p: f"{root.rstrip('/')}/{p}".rstrip("/") for p in unmatched}
            probe = await client.execute(
                container_name,
                [
                    "find",
                    *gone.values(),
                    "-maxdepth",
                    "0",
                    "(",
                    "-type",
                    "f",
                    "-o",
                    "-type",
                    "d",
                    ")",
                ],
                workdir=root,
                tenant_id=tenant_id,
            )
            seen = [x.strip() for x in (probe.get("stdout") or "").splitlines() if x.strip()]
            unmatched = [
                p for p, r in gone.items() if not any(f == r or f.startswith(r + "/") for f in seen)
            ]
        if unmatched:
            raise ScopeMatchesNoFilesError(
                f"paths match no files in container '{container_name}': {unmatched}",
                unmatched,
            )
    if not paths:
        return None

    tmp = Path(tempfile.mkdtemp(prefix=tmp_prefix))
    written = 0
    skipped: list[str] = []
    rejected: list[str] = []
    try:
        for abs_path in paths:
            rel = safe_relative(abs_path, root)
            if rel is None:
                rejected.append(abs_path)
                continue
            content = await client.read_file(container_name, abs_path, tenant_id=tenant_id)
            if not content:
                # Discovered by pattern, not requested: skippable, but counted.
                skipped.append(abs_path)
                continue
            dest = tmp / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            written += 1
    except BaseException:
        # Never leak the temp dir when a read/write fails midway.
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    if rejected:
        logger.warning(
            "Rejected %d listed path(s) outside %s in container '%s': %s",
            len(rejected),
            root,
            container_name,
            rejected[:5],
        )
    if skipped:
        logger.warning(
            "Skipped %d of %d discovered file(s) in container '%s' (unreadable or empty): %s",
            len(skipped),
            len(paths),
            container_name,
            skipped[:20],
        )

    if written == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        return None

    logger.info(
        "Materialized %d file(s) from container '%s' into %s (%d skipped)",
        written,
        container_name,
        tmp,
        len(skipped),
    )
    return tmp
