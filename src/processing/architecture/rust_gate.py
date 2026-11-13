"""The argv of the Rust quality checks (developer feedback, not the authoritative
evaluation: that is the rust test runner Job in coding-testing).

Shared by the runners (``tool_execution._rust``) and the tool configuration
(``architecture_pattern_analysis``), so what is configured is what runs.
"""

from __future__ import annotations

#: ``--workspace`` lints every member of the cargo workspace, including when the
#: root is itself a package with members (without it only the root package is
#: linted); ``--all-targets`` lints tests/benches/examples too. No ``-D warnings``:
#: findings are the verdict, and promoting warnings to errors makes cargo stop
#: before linting the crates that depend on the first one that warns.
CLIPPY_COMMAND: tuple[str, ...] = (
    "cargo",
    "clippy",
    "--workspace",
    "--all-targets",
    "--message-format=json",
)
#: ``--all`` checks every package of the cargo workspace, not just the root one.
RUSTFMT_COMMAND: tuple[str, ...] = ("cargo", "fmt", "--all", "--", "--check")
#: cargo-audit's machine-readable report, run in the crate dir.
CARGO_AUDIT_COMMAND: tuple[str, ...] = ("cargo", "audit", "--json")
