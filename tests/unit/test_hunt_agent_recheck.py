"""Hunt agent ``--recheck`` and the finding fingerprint, on real temporary git repos.

A recheck re-runs the SAME detection on ONE file as it is in a named commit's
tree. It must reproduce a finding that is there, report it gone when the file
at that commit no longer has it, ignore the working tree entirely, and turn
every "could not examine" case into a distinct status, never an empty pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hunt_agent import hunt as agent
from tests.unit.test_hunt_agent import BROKEN_PY, CLEAN_PY, _git, _repo, _run_cli

LONG_FN = "def big():\n" + "".join(f"    x{i} = {i}\n" for i in range(110))


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _fingerprints(result: dict) -> set[str]:
    return {f["fingerprint"] for f in result["findings"]}


# ---------------------------------------------------------------------------
# Recheck semantics
# ---------------------------------------------------------------------------


def test_finding_at_commit_a_reproduces_and_is_gone_at_fixed_commit_b(tmp_path):
    repo = _repo(tmp_path, {"bad.py": BROKEN_PY, "ok.py": CLEAN_PY})
    commit_a = _git(repo, "rev-parse", "HEAD")
    hunted = agent.hunt(repo)
    (finding,) = [f for f in hunted["findings"] if f["file_path"] == "bad.py"]

    (repo / "bad.py").write_text(CLEAN_PY, encoding="utf-8")
    commit_b = _commit_all(repo, "fix")

    at_a = agent.recheck(repo, commit_a, "bad.py")
    at_b = agent.recheck(repo, commit_b, "bad.py")

    assert at_a["status"] == "scanned"
    assert at_a["commit_sha"] == commit_a
    assert finding["fingerprint"] in _fingerprints(at_a)
    # Same detection, same shape: the recheck finding IS the hunt finding.
    assert [f for f in at_a["findings"] if f["fingerprint"] == finding["fingerprint"]] == [finding]
    assert at_b["status"] == "scanned"
    assert finding["fingerprint"] not in _fingerprints(at_b)
    assert at_b["findings"] == []


def test_working_tree_edit_is_ignored_in_both_directions(tmp_path):
    repo = _repo(tmp_path, {"bad.py": BROKEN_PY, "ok.py": CLEAN_PY})
    commit = _git(repo, "rev-parse", "HEAD")
    expected = _fingerprints(agent.recheck(repo, commit, "bad.py"))
    assert expected

    # Fix bad.py and break ok.py ONLY in the working tree.
    (repo / "bad.py").write_text(CLEAN_PY, encoding="utf-8")
    (repo / "ok.py").write_text(BROKEN_PY, encoding="utf-8")

    assert _fingerprints(agent.recheck(repo, commit, "bad.py")) == expected
    clean = agent.recheck(repo, commit, "ok.py")
    assert (clean["status"], clean["findings"]) == ("scanned", [])


@pytest.mark.parametrize(
    ("commit", "path", "status"),
    [
        ("not-a-sha", "bad.py", "invalid_commit"),
        ("ABCDEF" + "0" * 34, "bad.py", "invalid_commit"),  # upper case is not a full sha
        ("0" * 40, "bad.py", "commit_not_found"),
        (None, "missing.py", "path_not_found"),
        (None, "./bad.py", "path_not_found"),  # exact hunt-relative path only
        (None, "README.md", "path_out_of_scope"),
        (None, "vendor/v.py", "path_out_of_scope"),
    ],
)
def test_bad_commit_or_path_is_a_distinct_status_never_an_empty_pass(
    tmp_path, commit, path, status
):
    repo = _repo(tmp_path, {"bad.py": BROKEN_PY, "README.md": "x", "vendor/v.py": BROKEN_PY})
    head = _git(repo, "rev-parse", "HEAD")

    result = agent.recheck(repo, commit or head, path)

    assert result["status"] == status
    assert result["status"] != agent.RECHECK_SCANNED
    assert result["findings"] == []


def test_a_tree_sha_is_not_a_commit(tmp_path):
    repo = _repo(tmp_path, {"bad.py": BROKEN_PY})
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    assert agent.recheck(repo, tree, "bad.py")["status"] == "commit_not_found"


def test_unscannable_file_at_commit_reports_why(tmp_path, monkeypatch):
    repo = _repo(tmp_path, {"bad.py": BROKEN_PY})
    head = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setitem(agent.SOURCE_EXTENSIONS, ".py", ("python", 0))  # 0 MiB cap

    result = agent.recheck(repo, head, "bad.py")

    assert (result["status"], result["findings"]) == ("oversized", [])


def test_not_a_repository_is_an_error_not_a_status(tmp_path):
    with pytest.raises(agent.AgentError):
        agent.recheck(tmp_path, "0" * 40, "bad.py")


def test_cli_recheck_writes_result_and_summary(tmp_path):
    repo = _repo(tmp_path, {"bad.py": BROKEN_PY})
    head = _git(repo, "rev-parse", "HEAD")
    out = tmp_path / "hunt-agent-recheck-1.json"

    proc = _run_cli(
        "--recheck", "--repo", str(repo), "--commit", head, "--path=bad.py", "--out", str(out)
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    document = json.loads(out.read_text())
    assert summary["ok"] is True
    assert (summary["status"], summary["path"], summary["commit_sha"]) == (
        "scanned",
        "bad.py",
        head,
    )
    assert summary["findings"] == len(document["findings"]) == 1
    assert document["schema"] == agent.RECHECK_SCHEMA
    assert document["agent_sha"] == agent.compute_agent_sha()


def test_cli_recheck_requires_commit_and_path(tmp_path):
    repo = _repo(tmp_path, {"bad.py": BROKEN_PY})
    out = tmp_path / "hunt-agent-recheck-2.json"
    proc = _run_cli("--recheck", "--repo", str(repo), "--out", str(out))
    assert proc.returncode != 0
    assert not out.exists()


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def _single(repo: Path, path: str) -> dict:
    (finding,) = [f for f in agent.hunt(repo)["findings"] if f["file_path"] == path]
    return finding


def test_every_hunt_finding_carries_a_sha256_fingerprint(tmp_path):
    repo = _repo(tmp_path, {"bad.py": BROKEN_PY, "big.py": LONG_FN, "bad.js": "function f( {\n"})
    findings = agent.hunt(repo)["findings"]
    assert findings
    for finding in findings:
        fp = finding["fingerprint"]
        assert isinstance(fp, str) and len(fp) == 64 and int(fp, 16) >= 0


def test_unrelated_edit_above_does_not_change_the_fingerprint(tmp_path):
    repo = _repo(tmp_path, {"big.py": LONG_FN})
    before = _single(repo, "big.py")

    (repo / "big.py").write_text("import os\n\n\nX = 1\n" + LONG_FN, encoding="utf-8")
    _commit_all(repo, "unrelated edit above")
    after = _single(repo, "big.py")

    assert after["line_number"] == before["line_number"] + 4  # the line moved ...
    assert after["fingerprint"] == before["fingerprint"]  # ... the identity did not


def test_reindenting_or_growing_the_function_keeps_the_fingerprint(tmp_path):
    repo = _repo(tmp_path, {"big.py": LONG_FN})
    before = _single(repo, "big.py")

    grown = LONG_FN.replace("def big():", "def   big():") + "    tail = 1\n"
    (repo / "big.py").write_text(grown, encoding="utf-8")
    _commit_all(repo, "grow")

    assert _single(repo, "big.py")["fingerprint"] == before["fingerprint"]


def test_editing_the_flagged_line_changes_the_fingerprint(tmp_path):
    repo = _repo(tmp_path, {"big.py": LONG_FN})
    before = _single(repo, "big.py")

    (repo / "big.py").write_text(LONG_FN.replace("def big():", "def huge():"), encoding="utf-8")
    _commit_all(repo, "rename")

    assert _single(repo, "big.py")["fingerprint"] != before["fingerprint"]


def test_identical_anchors_in_one_file_get_distinct_fingerprints(tmp_path):
    body = "".join(f"    x{i} = {i}\n" for i in range(110))
    repo = _repo(tmp_path, {"two.py": f"def big():\n{body}\n\ndef big():\n{body}"})
    findings = [f for f in agent.hunt(repo)["findings"] if f["file_path"] == "two.py"]
    assert len(findings) == 2
    assert len({f["fingerprint"] for f in findings}) == 2


def test_same_line_in_another_file_is_another_finding(tmp_path):
    repo = _repo(tmp_path, {"a/bad.py": BROKEN_PY, "b/bad.py": BROKEN_PY})
    fps = {f["file_path"]: f["fingerprint"] for f in agent.hunt(repo)["findings"]}
    assert fps["a/bad.py"] != fps["b/bad.py"]


def test_fingerprint_recipe_is_pinned():
    """Changing the recipe silently would orphan every stored fingerprint."""
    assert agent.finding_fingerprint("syntax_error", "bad.py", "def broken(:", 0) == (
        __import__("hashlib")
        .sha256(b"bugfp/v1\0syntax_error\0bad.py\0def broken(:\0" + b"0")
        .hexdigest()
    )
    assert agent.anchor_text(b"a\n   def  x( y ):  \n", 2) == "def x( y ):"
    assert agent.anchor_text(b"a\n", 9) == ""
    assert agent.anchor_text(b"a\n", None) == ""
