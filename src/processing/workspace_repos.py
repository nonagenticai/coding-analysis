"""Which repository of a multi-repo workspace an analysis is about.

A py2rust workspace holds ONE primary repository at the workspace root (the
Rust target) plus any number of read-only source checkouts that coding-containers
places at ``<root>/.repos/<owner>__<repo>`` (``secondary_repos.secondary_path``).

Before this module every workspace-wide walk (``find`` listings, tool runs)
descended into ``.repos/`` and lumped the source repos in with the primary one,
with nothing saying which file came from which repo. The rules now are:

* the default selector is :data:`PRIMARY_REPO`: the walk starts at the
  workspace root and EXCLUDES ``.repos/`` (:func:`repos_exclude_globs`);
* ``repo=".repos/<name>"`` (or the bare ``<name>``) moves the analysis root to
  that checkout, so an agent can analyse one source repo explicitly; paths in
  the answer are then relative to that checkout;
* :func:`repo_of_path` attributes a root-relative path to the repo it belongs
  to, for the one case that stays ambiguous: a primary-repo request that names
  ``.repos/...`` paths explicitly (accepted for backward compatibility).
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, Field, field_validator

#: Selector value for the workspace's own repository (the workspace root).
PRIMARY_REPO = "primary"
#: Directory, relative to the workspace root, that holds the source checkouts.
REPOS_DIR = ".repos"

#: One path segment naming a checkout under ``.repos/`` (``<owner>__<repo>``
#: in practice; any safe single segment is accepted). No ``/``, no ``..``,
#: not hidden, so it can never escape ``.repos/``.
_REPO_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")

REPO_FIELD_DESCRIPTION = (
    "Which repository of the workspace to analyse. Omitted or 'primary' = the "
    "workspace's own repository at the workspace root, with the source checkouts "
    "under '.repos/' EXCLUDED. '.repos/<name>' (or the bare '<name>') = that one "
    "source checkout; file paths in the answer are then relative to it."
)


def normalize_repo(value: str | None) -> str:
    """Return the canonical selector: ``"primary"`` or ``".repos/<name>"``.

    Raises:
        ValueError: anything that is not ``primary``, ``<name>`` or
            ``.repos/<name>`` with a single safe path segment as ``<name>``.
    """
    if value is None:
        return PRIMARY_REPO
    if not isinstance(value, str):
        raise ValueError("repo must be a string")
    raw = value.strip().strip("/")
    if raw in ("", PRIMARY_REPO, "."):
        return PRIMARY_REPO
    name = raw[len(REPOS_DIR) + 1 :] if raw.startswith(REPOS_DIR + "/") else raw
    if not _REPO_NAME_RE.match(name) or ".." in name:
        raise ValueError(
            f"repo must be 'primary' or '{REPOS_DIR}/<name>' with a single-segment name, "
            f"got {value!r}"
        )
    return f"{REPOS_DIR}/{name}"


def repo_root(workspace_root: str, repo: str | None) -> str:
    """Absolute analysis root inside the container for *repo*."""
    root = workspace_root.rstrip("/") or "/"
    canonical = normalize_repo(repo)
    if canonical == PRIMARY_REPO:
        return root
    return f"{root.rstrip('/')}/{canonical}"


def repos_exclude_globs(root: str) -> tuple[str, ...]:
    """``find -not -path`` globs that keep a walk from *root* out of ``.repos/``.

    Anchored at *root* (``find -path`` matches the whole path, and ``*`` there
    crosses ``/``): an unanchored ``*/.repos/*`` would also match every file of
    a source checkout when *root* itself is ``<ws>/.repos/<name>``.
    """
    base = root.rstrip("/")
    return (f"{base}/{REPOS_DIR}", f"{base}/{REPOS_DIR}/*")


def is_repos_path(rel_path: str) -> bool:
    """True iff a root-relative path lies under ``.repos/``."""
    rel = rel_path
    while rel.startswith("./"):
        rel = rel[2:]
    return rel == REPOS_DIR or rel.startswith(REPOS_DIR + "/")


def scope_enters_repos(scope_paths: Iterable[str] | None) -> bool:
    """True iff an explicit scope names something under ``.repos/``."""
    return any(is_repos_path(p) for p in (scope_paths or ()))


def repo_of_path(rel_path: str, selected: str | None = None) -> str:
    """The repo a path (relative to the selected repo's root) belongs to.

    For a primary-repo request a path under ``.repos/<name>/`` belongs to
    ``.repos/<name>``; every other path belongs to the selected repo.
    """
    canonical = normalize_repo(selected)
    if canonical != PRIMARY_REPO:
        return canonical
    rel = rel_path
    while rel.startswith("./"):
        rel = rel[2:]
    if rel.startswith(REPOS_DIR + "/"):
        name = rel.split("/", 2)[1]
        if name:
            return f"{REPOS_DIR}/{name}"
    return PRIMARY_REPO


class RepoSelector(BaseModel):
    """Mixin for request models: the optional ``repo`` selector, canonicalised.

    Omitted, ``null``, ``""`` and ``"primary"`` all mean the primary repo, so the
    pre-existing request shape is unchanged.
    """

    repo: str | None = Field(
        default=None, description=REPO_FIELD_DESCRIPTION, validate_default=True
    )

    @field_validator("repo")
    @classmethod
    def _canonical_repo(cls, value: str | None) -> str:
        return normalize_repo(value)
