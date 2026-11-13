#!/usr/bin/env python3
"""Print, check or refresh the hunt agent's sha (src/hunt_agent/MANIFEST.sha256).

The agent sha is sha256(MANIFEST.sha256), where the manifest is the
``sha256sum``-format listing of the agent's ``*.py`` files in name order. The
agent computes the same value about itself at runtime and reports it; the
coding-analysis worker fails a hunt whose reported sha differs from
sha256(the committed manifest).

    scripts/hunt-agent-sha.py            # print the sha of the files on disk
    scripts/hunt-agent-sha.py --check    # exit 1 if MANIFEST.sha256 is stale
    scripts/hunt-agent-sha.py --write    # refresh MANIFEST.sha256
    scripts/hunt-agent-sha.py --agent-dir /opt/hunt-agent --check   # a vendored copy

Stdlib only; it never imports coding-analysis. coding-containers' drift check
can run it (or ``python3 hunt.py --print-manifest | diff - MANIFEST.sha256``)
against its vendored copy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

DEFAULT_AGENT_DIR = Path(__file__).resolve().parent.parent / "src" / "hunt_agent"


def _load_agent(agent_dir: Path):
    spec = importlib.util.spec_from_file_location("_hunt_agent_for_sha", agent_dir / "hunt.py")
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {agent_dir / 'hunt.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent-dir", type=Path, default=DEFAULT_AGENT_DIR)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()

    agent_dir = args.agent_dir.resolve()
    agent = _load_agent(agent_dir)
    expected = agent.manifest_text(agent_dir)
    manifest = agent_dir / agent.MANIFEST_NAME

    if args.write:
        manifest.write_text(expected, encoding="utf-8")
    elif args.check:
        committed = manifest.read_text(encoding="utf-8") if manifest.is_file() else ""
        if committed != expected:
            sys.stderr.write(
                f"{manifest} is stale; run scripts/hunt-agent-sha.py --write\n"
                f"--- committed\n{committed}+++ computed\n{expected}"
            )
            return 1
    sys.stdout.write(hashlib.sha256(expected.encode("utf-8")).hexdigest() + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
