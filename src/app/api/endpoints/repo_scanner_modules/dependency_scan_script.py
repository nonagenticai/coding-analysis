"""In-workspace Python dependency discovery script (coding-analysis#104).

Auto-index runs this inside the tenant's workspace pod against a freshly
imported, untrusted repository. It must never execute repository code:

* it is started with ``python3 -I -P`` from cwd ``/`` (see
  :data:`file_discovery.ISOLATED_PYTHON`), and receives the repository root as
  ``argv[1]`` instead of reading it from the cwd;
* its one child process, ``pip list``, gets the same ``-I -P`` flags and
  ``cwd="/"`` explicitly, because flags are not inherited;
* Poetry dependencies come from ``poetry.lock`` parsed as TOML text. ``poetry``
  is never executed: ``poetry show`` in the repo dir loads the repo's
  ``pyproject.toml`` plugins/config and runs with the checkout as cwd.

The printed shape is a JSON list of ``{"name", "version", "source"}`` where
``source`` is ``pip``, ``poetry`` or ``requirements`` (first source wins).
"""

from __future__ import annotations

from .file_discovery import WORKSPACE_ROOT, isolated_python_command

DEPENDENCY_SCAN_SCRIPT_PATH = "/tmp/scan_dependencies.py"  # nosec B108

DEPENDENCY_SCAN_SCRIPT = r'''
import json
import subprocess
import sys
from pathlib import Path


def get_pip_dependencies():
    """Installed packages of the interpreter, listed by an isolated pip."""
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-P", "-m", "pip", "list",
             "--format=json", "--disable-pip-version-check"],
            capture_output=True, text=True, cwd="/",
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return []


def parse_poetry_lock(text):
    """``[{"name", "version"}]`` for every locked package, as ``poetry show`` listed them."""
    import tomllib

    deps = []
    for pkg in tomllib.loads(text).get("package", []):
        name = pkg.get("name") if isinstance(pkg, dict) else None
        version = pkg.get("version") if isinstance(pkg, dict) else None
        if name and version:
            deps.append({"name": str(name), "version": str(version)})
    return deps


def get_poetry_dependencies(root):
    """Dependencies from poetry.lock, read as text. poetry is never executed."""
    lock = root / "poetry.lock"
    if not lock.is_file():
        return []
    try:
        return parse_poetry_lock(lock.read_text(encoding="utf-8"))
    except Exception:
        return []


def get_requirements_dependencies(root):
    """Get dependencies from requirements files."""
    deps = []
    req_files = ["requirements.txt", "requirements-dev.txt", "requirements/base.txt"]
    for req_file in req_files:
        path = root / req_file
        if path.is_file():
            try:
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Parse package==version format
                            if '==' in line:
                                name, version = line.split('==', 1)
                                deps.append({"name": name.strip(), "version": version.strip()})
                            elif '>=' in line:
                                name = line.split('>=')[0].strip()
                                deps.append({"name": name, "version": ">=requirement"})
            except Exception:
                pass
    return deps


def main():
    # The repo root arrives as argv[1]; the cwd is "/", never the checkout.
    root = Path(sys.argv[1])
    all_deps = {}
    for source, deps in (
        ("pip", get_pip_dependencies()),
        ("poetry", get_poetry_dependencies(root)),
        ("requirements", get_requirements_dependencies(root)),
    ):
        for dep in deps:
            name = dep["name"]
            if name not in all_deps:
                all_deps[name] = {"name": name, "version": dep["version"], "source": source}
    print(json.dumps(list(all_deps.values()), indent=2))


if __name__ == "__main__":
    main()
'''


def dependency_scan_command(
    script_path: str = DEPENDENCY_SCAN_SCRIPT_PATH, root: str = WORKSPACE_ROOT
) -> str:
    """Command that runs the written script isolated, with *root* as argv[1].

    Run it with ``workdir=NEUTRAL_WORKDIR``.
    """
    return isolated_python_command(script_path, root)
