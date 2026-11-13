"""The self-contained hunt agent (src/hunt_agent) on real temporary git repositories.

The agent runs inside the tenant workspace, so these tests drive it the way
production does: as ``python3 -I -P hunt.py`` from cwd ``/``, plus direct calls
into its functions for the finer-grained contracts.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hunt_agent import hunt as agent

AGENT_DIR = Path(agent.__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parent.parent
CLEAN_PY = "def add(a, b):\n    return a + b\n"
BROKEN_PY = "def broken(:\n    pass\n"


def _env_without_git() -> dict[str, str]:
    # Tests may run under a git hook that exports GIT_DIR & co.
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=repo,
        env=_env_without_git(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-P", str(AGENT_DIR / "hunt.py"), *args],
        cwd="/",
        env=_env_without_git(),
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# HEAD, not the working tree
# ---------------------------------------------------------------------------


def test_committed_files_are_scanned_and_counts_agree(tmp_path):
    repo = _repo(tmp_path, {"a.py": CLEAN_PY, "src/b.js": "const x = 1;\n", "README.md": "x"})

    result = agent.hunt(repo)

    assert result["commit_sha"] == _git(repo, "rev-parse", "HEAD")
    assert result["worktree_dirty"] is False
    assert result["tracked_source_files"] == 2  # README.md is not a source file
    assert result["scanned_files"] == 2
    assert result["findings"] == []
    assert result["not_scanned"] == []


def test_uncommitted_edit_is_not_scanned_but_marks_the_tree_dirty(tmp_path):
    repo = _repo(tmp_path, {"a.py": CLEAN_PY})
    head = _git(repo, "rev-parse", "HEAD")
    (repo / "a.py").write_text(BROKEN_PY, encoding="utf-8")  # edited, not committed

    result = agent.hunt(repo)

    assert result["commit_sha"] == head
    assert result["worktree_dirty"] is True
    assert result["findings"] == []  # the broken bytes exist only in the working tree
    assert (result["scanned_files"], result["tracked_source_files"]) == (1, 1)


def test_untracked_file_is_neither_counted_nor_scanned_but_is_dirty(tmp_path):
    repo = _repo(tmp_path, {"a.py": CLEAN_PY})
    (repo / "new.py").write_text(BROKEN_PY, encoding="utf-8")

    result = agent.hunt(repo)

    assert result["worktree_dirty"] is True
    assert (result["scanned_files"], result["tracked_source_files"]) == (1, 1)
    assert result["findings"] == []


def test_committed_syntax_error_and_oversized_function_are_found(tmp_path):
    long_fn = "def big():\n" + "".join(f"    x{i} = {i}\n" for i in range(110))
    repo = _repo(
        tmp_path,
        {"bad.py": BROKEN_PY, "big.py": long_fn, "bad.js": "function f( {\n", "ok.go": ""},
    )

    result = agent.hunt(repo)

    by_file = {(f["file_path"], f["error_type"]) for f in result["findings"]}
    assert ("bad.py", "syntax_error") in by_file
    assert ("bad.js", "syntax_error") in by_file
    assert ("big.py", "logic_error") in by_file
    syntax = next(f for f in result["findings"] if f["file_path"] == "bad.py")
    assert syntax["line_number"] == 1
    # An empty file is parsed and clean: scanned, no finding.
    assert result["scanned_files"] == result["tracked_source_files"] == 4


# ---------------------------------------------------------------------------
# Scope: both counts share it; the size cap is NOT a scope rule
# ---------------------------------------------------------------------------


def test_excluded_dirs_are_out_of_both_counts_and_declared(tmp_path):
    repo = _repo(
        tmp_path,
        {
            "a.py": CLEAN_PY,
            "vendor/lib.py": BROKEN_PY,
            "node_modules/m.js": "function f( {\n",
            "src/build/gen.py": BROKEN_PY,
        },
    )

    result = agent.hunt(repo)

    assert (result["scanned_files"], result["tracked_source_files"]) == (1, 1)
    assert result["findings"] == []
    assert result["excluded_dirs"] == sorted(agent.EXCLUDED_DIRS)


def test_symlinks_and_submodules_are_out_of_both_counts_and_declared(tmp_path):
    """scanned and tracked share ONE scope, or fr refuses every such repo."""
    repo = _repo(tmp_path, {"ok.py": CLEAN_PY})
    head = _git(repo, "rev-parse", "HEAD")
    (repo / "link.py").symlink_to("ok.py")
    _git(repo, "add", "link.py")
    # A gitlink whose path would otherwise pass the source-file predicate.
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{head},sub.py")
    _git(repo, "commit", "-q", "-m", "symlink + submodule")
    modes = _git(repo, "ls-tree", "HEAD")
    assert "120000 blob" in modes and "160000 commit" in modes  # fixture really has both

    result = agent.hunt(repo)

    assert result["scanned_files"] == result["tracked_source_files"] == 1
    assert result["not_scanned"] == []
    assert result["excluded_entry_kinds"] == ["non_utf8_path", "submodule", "symlink"]
    assert result["findings"] == []


def test_non_utf8_path_is_out_of_both_counts_and_declared(tmp_path):
    """A path that is not UTF-8 cannot be stored; excluding it keeps scanned == tracked."""
    repo = _repo(tmp_path, {"ok.py": CLEAN_PY})
    oid = _git(repo, "hash-object", "-w", "ok.py")
    subprocess.run(
        [
            b"git",
            b"update-index",
            b"--add",
            b"--cacheinfo",
            f"100644,{oid},".encode() + b"x\xff.py",
        ],
        cwd=repo,
        env=_env_without_git(),
        check=True,
    )
    _git(repo, "commit", "-q", "-m", "non-utf8 path")
    assert len(agent.list_head_blobs(repo, _git(repo, "rev-parse", "HEAD"))) == 2  # fixture has it

    result = agent.hunt(repo)

    assert result["scanned_files"] == result["tracked_source_files"] == 1
    assert result["not_scanned"] == []
    assert "non_utf8_path" in result["excluded_entry_kinds"]
    json.dumps(result, ensure_ascii=False).encode("utf-8")  # storable: no lone surrogate


def test_oversized_tracked_file_stays_in_the_denominator(tmp_path, monkeypatch):
    repo = _repo(tmp_path, {"a.py": CLEAN_PY, "b.js": "const x = 1;\n"})
    monkeypatch.setitem(agent.SOURCE_EXTENSIONS, ".py", ("python", 0))

    result = agent.hunt(repo)

    assert result["tracked_source_files"] == 2
    assert result["scanned_files"] == 1
    assert result["not_scanned"] == [{"path": "a.py", "reason": "oversized"}]


def test_inherited_git_env_does_not_redirect_the_measurement(tmp_path, monkeypatch):
    outer = _repo(tmp_path / "outer", {"x.py": CLEAN_PY, "y.py": CLEAN_PY, "z.py": CLEAN_PY})
    inner = _repo(tmp_path / "inner", {"only.py": CLEAN_PY})
    monkeypatch.setenv("GIT_DIR", str(outer / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(outer))

    result = agent.hunt(inner)

    assert result["commit_sha"] == _git(inner, "rev-parse", "HEAD")
    assert result["tracked_source_files"] == 1


def test_not_a_repository_is_an_error_never_an_empty_scan(tmp_path):
    (tmp_path / "a.py").write_text(CLEAN_PY, encoding="utf-8")

    with pytest.raises(agent.AgentError):
        agent.hunt(tmp_path)


# ---------------------------------------------------------------------------
# The CLI exactly as the worker drives it
# ---------------------------------------------------------------------------


def test_cli_writes_result_prints_summary_and_serves_chunks(tmp_path):
    repo = _repo(tmp_path, {"bad.py": BROKEN_PY, "ok.py": CLEAN_PY})
    out = tmp_path / "hunt-agent-test-1.json"

    proc = _run_cli("--repo", str(repo), "--out", str(out))

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    data = out.read_bytes()
    assert summary["ok"] is True
    assert summary["result_bytes"] == len(data)
    assert summary["result_sha256"] == hashlib.sha256(data).hexdigest()
    assert summary["agent_sha"] == agent.compute_agent_sha()
    assert summary["findings"] == 1
    document = json.loads(data)
    assert document["agent_sha"] == summary["agent_sha"]
    assert document["tree_sitter_version"] == agent.tree_sitter_versions()["tree-sitter"]
    assert document["grammar_versions"] == agent.tree_sitter_versions()

    first = _run_cli("--read-result", str(out), "--offset", "0", "--length", "10")
    rest = _run_cli("--read-result", str(out), "--offset", "10", "--length", str(len(data)))
    assert (first.stdout + rest.stdout).encode("ascii") == data

    assert _run_cli("--remove-result", str(out)).returncode == 0
    assert not out.exists()


def test_cli_refuses_a_result_inside_the_repository(tmp_path):
    repo = _repo(tmp_path, {"a.py": CLEAN_PY})

    proc = _run_cli("--repo", str(repo), "--out", str(repo / "hunt-agent-x.json"))

    assert proc.returncode == 3
    assert json.loads(proc.stdout)["ok"] is False
    assert _git(repo, "status", "--porcelain") == ""


def test_cli_reader_refuses_files_that_are_not_results(tmp_path):
    secret = tmp_path / "other.json"
    secret.write_text("{}", encoding="utf-8")

    proc = _run_cli("--read-result", str(secret), "--length", "2")

    assert proc.returncode == 3
    assert json.loads(proc.stdout)["ok"] is False


# ---------------------------------------------------------------------------
# Identity and isolation guards
# ---------------------------------------------------------------------------


def test_committed_manifest_matches_the_agent_files():
    """Editing the agent without refreshing MANIFEST.sha256 must red CI."""
    committed = (AGENT_DIR / agent.MANIFEST_NAME).read_text(encoding="utf-8")
    assert committed == agent.manifest_text(), "run: scripts/hunt-agent-sha.py --write"
    assert agent.compute_agent_sha() == hashlib.sha256(committed.encode()).hexdigest()


def test_sha_script_check_passes_and_prints_the_agent_sha():
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "hunt-agent-sha.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == agent.compute_agent_sha()


def test_manifest_covers_every_agent_file():
    names = [p.name for p in agent.agent_files()]
    assert names == sorted(names)
    assert {"__init__.py", "hunt.py"} <= set(names)


def test_agent_imports_only_stdlib_and_tree_sitter():
    allowed_third_party = ("tree_sitter",)
    for path in agent.agent_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, f"{path.name}: relative import"
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                top = module.split(".")[0]
                assert top in sys.stdlib_module_names or top.startswith(allowed_third_party), (
                    f"{path.name} imports {module}; the agent must stay self-contained"
                )
    # Grammar modules are imported dynamically; they must be tree-sitter ones.
    assert all(m.startswith("tree_sitter_") for m, _f, _d in agent.GRAMMARS.values())


def test_source_table_matches_the_language_registry():
    """The agent cannot import the registry, so the two tables are pinned equal."""
    from processing.languages import language_registry

    registry = {
        ext: (
            language_registry.get_language_config(lang).tree_sitter_grammar,
            language_registry.get_language_config(lang).max_file_size_mb,
        )
        for ext, lang in language_registry._extension_map.items()
    }
    assert agent.SOURCE_EXTENSIONS == registry
