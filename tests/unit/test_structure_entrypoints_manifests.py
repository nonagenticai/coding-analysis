"""Cargo bin targets, console scripts and entrypoints, read directly from real
manifest text by the structure extractors."""

from __future__ import annotations

from processing.structure.entrypoints import entrypoint_inventory
from processing.structure.rust_facts import discover_crates

CARGO_BINS = (
    '[package]\nname = "tool"\n\n'
    '[[bin]]\nname = "first"\npath = "src/first.rs"\n\n'
    '[[bin]]\n# comment\nname = "second"\n\n'
    '[[bin]]\nname = "nested"\n'
)


def test_every_bin_target_has_its_own_line():
    files = {
        "Cargo.toml": CARGO_BINS,
        "src/first.rs": "fn main() {}\n",
        "src/bin/second.rs": "fn main() {}\n",
        "src/bin/nested/main.rs": "fn main() {}\n",
    }
    entries = [e for e in entrypoint_inventory(files).entrypoints if e.kind == "bin_target"]
    assert [(e.name, e.line, e.detail) for e in entries] == [
        ("first", 5, "src/first.rs"),
        ("second", 10, "src/bin/second.rs"),
        ("nested", 13, "src/bin/nested/main.rs"),
    ]


def test_bin_without_name_line_and_inline_bin_array_lines():
    # A [[bin]] whose name is in a later table is reported at its header line;
    # an inline ``bin = [...]`` (no header at all) falls back to the first name key.
    headed = '[package]\nname = "t"\n[[bin]]\npath = "src/x.rs"\n[lib]\nname = "lib"\n'
    inline = '[package]\nname = "t"\nbin = [{name = "a", path = "src/a.rs"}]\n'
    for text, line in ((headed, 3), (inline, 2)):
        files = {"Cargo.toml": text, "src/x.rs": "fn main() {}\n", "src/a.rs": "fn main() {}\n"}
        bins = [e for e in entrypoint_inventory(files).entrypoints if e.kind == "bin_target"]
        assert all(e.line == line for e in bins), (text, [(e.name, e.line) for e in bins])


def test_pyproject_script_lines_quoted_and_inline():
    files = {
        "pyproject.toml": (
            '[project]\nname = "x"\n[project.scripts]\n"my-cli" = "x.cli:main"\n'
            'serve = "x.srv:run"\n[tool.poetry.scripts]\nother = {reference = "x", type = "file"}\n'
        )
    }
    got = {
        e.name: e.line
        for e in entrypoint_inventory(files).entrypoints
        if e.kind == "console_script"
    }
    assert got == {"my-cli": 4, "serve": 5}


def test_python_entrypoints_attribute_calls_to_the_enclosing_def_only():
    src = (
        "import redis\n"
        "def outer():\n"
        "    def inner():\n"
        "        r.subscribe('a')\n"
        "    r.subscribe('b')\n"
        "@app.subscribe('deco')\n"  # decorator: module scope, not `handler`
        "def handler():\n"
        "    pass\n"
        "r.subscribe('top')\n"
    )
    got = {
        e.detail: e.name
        for e in entrypoint_inventory({"w.py": src}).entrypoints
        if e.kind == "redis_pubsub_subscriber"
    }
    assert got == {"'deco'": None, "'top'": None, "'a'": "inner", "'b'": "outer"}


def test_rust_clap_parser_derive_found_through_comments():
    src = (
        "/// The CLI\n#[derive(Debug, clap::Parser)]\n// options\n#[command(version)]\n"
        "struct Args { x: u8 }\n"
        "#[derive(Debug)] // Parser in a comment is not a derive\nstruct NotCli;\n"
    )
    cli = [
        (e.name, e.line)
        for e in entrypoint_inventory({"src/main.rs": src}).entrypoints
        if e.kind == "cli"
    ]
    assert cli == [("Args", 5)]


def test_src_bin_dir_main_binaries_are_detected():
    files = {
        "Cargo.toml": '[package]\nname = "tool"\n',
        "src/main.rs": "fn main() {}\n",
        "src/bin/worker/main.rs": "fn main() {}\n",
        "src/bin/worker/helper.rs": "pub fn h() {}\n",
        "src/bin/cli.rs": "fn main() {}\n",
        "examples/bin/x/main.rs": "fn main() {}\n",
    }
    binaries = sorted(
        (e.name, e.file) for e in entrypoint_inventory(files).entrypoints if e.kind == "binary"
    )
    assert binaries == [
        ("cli", "src/bin/cli.rs"),
        ("main", "src/main.rs"),
        ("worker", "src/bin/worker/main.rs"),
    ]
    crates, _ = discover_crates(files)
    assert sorted(c.name for c in crates) == ["cli", "tool", "worker"]
