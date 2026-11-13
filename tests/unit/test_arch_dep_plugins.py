"""Unit tests for architecture dependency loader plugins.

Covers:
- js_loader.py        (JSDependencyLoader)
- rust_loader.py      (RustDependencyLoader)
- system_package_loader.py (SystemPackageLoader)
- poetry_loader.py    (PoetryLoader)
- uv_loader.py        (UvLoader)
- pip_loader.py       (PipLoader)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from models.architecture_models import ExternalPackage
from models.system_models import PackageManager
from processing.architecture.dependency_plugins.js_loader import (
    JSDependencyLoader,
)
from processing.architecture.dependency_plugins.pip_loader import PipLoader
from processing.architecture.dependency_plugins.poetry_loader import PoetryLoader
from processing.architecture.dependency_plugins.rust_loader import (
    RustDependencyLoader,
)
from processing.architecture.dependency_plugins.system_package_loader import (
    DetectedSystemPackage,
    SystemPackageLoader,
)
from processing.architecture.dependency_plugins.uv_loader import UvLoader


# ============================================================================
# JSDependencyLoader
# ============================================================================
class TestJSDependencyLoader:
    """Tests for JSDependencyLoader."""

    def setup_method(self):
        self.loader = JSDependencyLoader()

    # -- detect ---------------------------------------------------------------

    def test_detect_package_json(self, tmp_path: Path):
        (tmp_path / "package.json").write_text("{}")
        assert self.loader.detect(tmp_path) is True

    def test_detect_lockfile_without_package_json(self, tmp_path: Path):
        (tmp_path / "yarn.lock").write_text("")
        assert self.loader.detect(tmp_path) is True

    def test_detect_pnpm_lockfile(self, tmp_path: Path):
        (tmp_path / "pnpm-lock.yaml").write_text("")
        assert self.loader.detect(tmp_path) is True

    def test_detect_package_lock(self, tmp_path: Path):
        (tmp_path / "package-lock.json").write_text("")
        assert self.loader.detect(tmp_path) is True

    def test_detect_nested_package_json(self, tmp_path: Path):
        nested = tmp_path / "packages" / "app"
        nested.mkdir(parents=True)
        (nested / "package.json").write_text("{}")
        assert self.loader.detect(tmp_path) is True

    def test_detect_no_js_project(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("print()")
        assert self.loader.detect(tmp_path) is False

    # -- _detect_package_manager ----------------------------------------------

    def test_detect_package_manager_pnpm(self, tmp_path: Path):
        (tmp_path / "pnpm-lock.yaml").write_text("")
        assert self.loader._detect_package_manager(tmp_path) == "pnpm"

    def test_detect_package_manager_yarn(self, tmp_path: Path):
        (tmp_path / "yarn.lock").write_text("")
        assert self.loader._detect_package_manager(tmp_path) == "yarn"

    def test_detect_package_manager_npm(self, tmp_path: Path):
        (tmp_path / "package-lock.json").write_text("{}")
        assert self.loader._detect_package_manager(tmp_path) == "npm"

    def test_detect_package_manager_default(self, tmp_path: Path):
        assert self.loader._detect_package_manager(tmp_path) == "npm"

    # -- normalize_version ----------------------------------------------------

    def test_normalize_version_empty(self):
        assert self.loader.normalize_version("") == "*"

    def test_normalize_version_git(self):
        assert self.loader.normalize_version("git+https://repo") == "git:git+https://repo"

    def test_normalize_version_github(self):
        assert (
            self.loader.normalize_version("https://github.com/org/repo")
            == "git:https://github.com/org/repo"
        )

    def test_normalize_version_file(self):
        assert self.loader.normalize_version("file:../local") == "file:file:../local"

    def test_normalize_version_relative(self):
        assert self.loader.normalize_version("./local") == "file:./local"

    def test_normalize_version_url(self):
        assert (
            self.loader.normalize_version("https://example.com/pkg.tgz")
            == "url:https://example.com/pkg.tgz"
        )

    def test_normalize_version_workspace(self):
        assert self.loader.normalize_version("workspace:*") == "workspace:*"

    def test_normalize_version_tag(self):
        assert self.loader.normalize_version("latest") == "tag:latest"

    def test_normalize_version_semver(self):
        assert self.loader.normalize_version("^1.2.3") == "^1.2.3"

    # -- _create_package_from_spec -------------------------------------------

    def test_create_package_scoped(self):
        pkg = self.loader._create_package_from_spec("@scope/pkg", "^1.0.0", "dependencies")
        assert pkg is not None
        assert pkg.name == "@scope/pkg"
        assert pkg.ecosystem == "npm"

    def test_create_package_regular(self):
        pkg = self.loader._create_package_from_spec("lodash", "^4.17.21", "devDependencies")
        assert pkg is not None
        assert pkg.name == "lodash"
        assert "devDependencies" in pkg.description

    def test_create_package_invalid_name(self):
        pkg = self.loader._create_package_from_spec("inv@lid!name", "1.0", "dependencies")
        assert pkg is None

    # -- _parse_package_json --------------------------------------------------

    def test_parse_package_json_happy(self, tmp_path: Path):
        data = {
            "dependencies": {"express": "^4.18.0"},
            "devDependencies": {"jest": "^29.0.0"},
            "peerDependencies": {"react": ">=17"},
            "optionalDependencies": {"fsevents": "~2.3.2"},
        }
        pj = tmp_path / "package.json"
        pj.write_text(json.dumps(data))
        pkgs = self.loader._parse_package_json(pj)
        names = {p.name for p in pkgs}
        assert {"express", "jest", "react", "fsevents"} == names

    def test_parse_package_json_empty_file(self, tmp_path: Path):
        pj = tmp_path / "package.json"
        pj.write_text("")
        assert self.loader._parse_package_json(pj) == []

    def test_parse_package_json_invalid_json(self, tmp_path: Path):
        pj = tmp_path / "package.json"
        pj.write_text("{bad json")
        assert self.loader._parse_package_json(pj) == []

    # -- _parse_npm_lockfile --------------------------------------------------

    def test_parse_npm_lockfile_v2(self, tmp_path: Path):
        data = {
            "packages": {
                "": {"name": "root"},
                "node_modules/express": {"version": "4.18.2", "license": "MIT"},
            }
        }
        lf = tmp_path / "package-lock.json"
        lf.write_text(json.dumps(data))
        pkgs = self.loader._parse_npm_lockfile(lf)
        assert len(pkgs) == 1
        assert pkgs[0].name == "express"
        assert pkgs[0].version == "4.18.2"
        assert pkgs[0].license == "MIT"

    def test_parse_npm_lockfile_v1(self, tmp_path: Path):
        data = {
            "dependencies": {
                "lodash": {"version": "4.17.21"},
                "nested": {
                    "version": "1.0.0",
                    "dependencies": {
                        "inner": {"version": "0.1.0"},
                    },
                },
            }
        }
        lf = tmp_path / "package-lock.json"
        lf.write_text(json.dumps(data))
        pkgs = self.loader._parse_npm_lockfile(lf)
        names = {p.name for p in pkgs}
        assert {"lodash", "nested", "inner"} == names

    def test_parse_npm_lockfile_empty(self, tmp_path: Path):
        lf = tmp_path / "package-lock.json"
        lf.write_text("")
        assert self.loader._parse_npm_lockfile(lf) == []

    # -- _parse_yarn_lockfile -------------------------------------------------

    def test_parse_yarn_lockfile(self, tmp_path: Path):
        content = (
            '# yarn lockfile v1\n\nlodash@^4.17.21:\n  version "4.17.21"\n  resolved "..."\n\n'
        )
        yf = tmp_path / "yarn.lock"
        yf.write_text(content)
        pkgs = self.loader._parse_yarn_lockfile(yf)
        assert len(pkgs) == 1
        assert pkgs[0].name == "lodash"
        assert pkgs[0].version == "4.17.21"

    def test_parse_yarn_lockfile_empty(self, tmp_path: Path):
        yf = tmp_path / "yarn.lock"
        yf.write_text("")
        assert self.loader._parse_yarn_lockfile(yf) == []

    def test_parse_yarn_lockfile_scoped(self, tmp_path: Path):
        content = '"@babel/core@^7.0.0":\n  version "7.22.0"\n'
        yf = tmp_path / "yarn.lock"
        yf.write_text(content)
        pkgs = self.loader._parse_yarn_lockfile(yf)
        assert len(pkgs) == 1

    # -- _parse_pnpm_lockfile -------------------------------------------------

    def test_parse_pnpm_lockfile(self, tmp_path: Path):
        import yaml

        data = {
            "packages": {
                "/express/4.18.2": {"license": "MIT"},
                "/@scope/pkg/1.0.0": {"license": "ISC"},
            }
        }
        pf = tmp_path / "pnpm-lock.yaml"
        pf.write_text(yaml.dump(data))
        pkgs = self.loader._parse_pnpm_lockfile(pf)
        names = {p.name for p in pkgs}
        assert "express" in names

    def test_parse_pnpm_lockfile_empty(self, tmp_path: Path):
        pf = tmp_path / "pnpm-lock.yaml"
        pf.write_text("")
        assert self.loader._parse_pnpm_lockfile(pf) == []

    # -- _extract_package_name_from_path --------------------------------------

    def test_extract_name_regular(self):
        assert self.loader._extract_package_name_from_path("node_modules/express") == "express"

    def test_extract_name_scoped(self):
        assert (
            self.loader._extract_package_name_from_path("node_modules/@types/node") == "@types/node"
        )

    def test_extract_name_no_prefix(self):
        assert self.loader._extract_package_name_from_path("lodash") == "lodash"

    def test_extract_name_scoped_short(self):
        # scoped package with only one part after @
        assert self.loader._extract_package_name_from_path("@scope") is None

    # -- _extract_yarn_package_name -------------------------------------------

    def test_extract_yarn_regular(self):
        assert self.loader._extract_yarn_package_name("lodash@^4.17.21") == "lodash"

    def test_extract_yarn_scoped(self):
        assert self.loader._extract_yarn_package_name("@babel/core@^7.0.0") == "@babel/core"

    def test_extract_yarn_no_at(self):
        assert self.loader._extract_yarn_package_name("lodash") == "lodash"

    # -- _parse_pnpm_package_spec ---------------------------------------------

    def test_pnpm_spec_regular(self):
        name, version = self.loader._parse_pnpm_package_spec("/express/4.18.2")
        assert name == "express"
        assert version == "4.18.2"

    def test_pnpm_spec_scoped(self):
        name, version = self.loader._parse_pnpm_package_spec("/@scope/pkg/1.0.0")
        assert name is not None
        assert version == "1.0.0"

    def test_pnpm_spec_no_version(self):
        name, version = self.loader._parse_pnpm_package_spec("justname")
        assert name is None
        assert version is None

    # -- _merge_package_data --------------------------------------------------

    def test_merge_package_data(self):
        manifest = [
            ExternalPackage(name="express", version="^4.18", license="Unknown", ecosystem="npm"),
        ]
        lock = [
            ExternalPackage(name="express", version="4.18.2", license="MIT", ecosystem="npm"),
        ]
        merged = self.loader._merge_package_data(manifest, lock)
        assert len(merged) == 1
        assert merged[0].version == "4.18.2"
        assert merged[0].license == "MIT"

    def test_merge_package_data_no_match(self):
        manifest = [
            ExternalPackage(name="lodash", version="^4.0", license="Unknown", ecosystem="npm"),
        ]
        merged = self.loader._merge_package_data(manifest, [])
        assert len(merged) == 1
        assert merged[0].name == "lodash"

    # -- _deduplicate_packages ------------------------------------------------

    def test_deduplicate_packages(self):
        pkgs = [
            ExternalPackage(name="a", version="1.0", license="MIT", ecosystem="npm"),
            ExternalPackage(name="a", version="2.0", license="MIT", ecosystem="npm"),
            ExternalPackage(name="b", version="1.0", license="MIT", ecosystem="npm"),
        ]
        result = self.loader._deduplicate_packages(pkgs)
        assert len(result) == 2
        assert result[0].version == "1.0"  # first wins

    # -- _parse_workspaces ----------------------------------------------------

    def test_parse_workspaces_list(self, tmp_path: Path):
        pkg = {"workspaces": ["packages/*"]}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        ws_dir = tmp_path / "packages" / "app"
        ws_dir.mkdir(parents=True)
        ws_pkg = {"dependencies": {"react": "^18.0.0"}}
        (ws_dir / "package.json").write_text(json.dumps(ws_pkg))
        pkgs = self.loader._parse_workspaces(tmp_path)
        assert any(p.name == "react" for p in pkgs)

    def test_parse_workspaces_dict_format(self, tmp_path: Path):
        pkg = {"workspaces": {"packages": ["libs/*"]}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        ws_dir = tmp_path / "libs" / "core"
        ws_dir.mkdir(parents=True)
        ws_pkg = {"dependencies": {"axios": "^1.0"}}
        (ws_dir / "package.json").write_text(json.dumps(ws_pkg))
        pkgs = self.loader._parse_workspaces(tmp_path)
        assert any(p.name == "axios" for p in pkgs)

    def test_parse_workspaces_no_workspaces(self, tmp_path: Path):
        (tmp_path / "package.json").write_text("{}")
        assert self.loader._parse_workspaces(tmp_path) == []

    def test_parse_workspaces_no_package_json(self, tmp_path: Path):
        assert self.loader._parse_workspaces(tmp_path) == []

    # -- parse (integration) --------------------------------------------------

    def test_parse_full_npm(self, tmp_path: Path):
        pkg = {"dependencies": {"express": "^4.18.0"}}
        lock = {
            "packages": {
                "": {"name": "root"},
                "node_modules/express": {"version": "4.18.2", "license": "MIT"},
            }
        }
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        (tmp_path / "package-lock.json").write_text(json.dumps(lock))
        pkgs = self.loader.parse(tmp_path)
        assert len(pkgs) >= 1
        expr = next((p for p in pkgs if p.name == "express"), None)
        assert expr is not None
        assert expr.version == "4.18.2"

    def test_parse_filesystem_error(self, tmp_path: Path):
        # No files at all - should still return empty without raising
        assert self.loader.parse(tmp_path) == []

    def test_parse_lockfile_dispatch(self, tmp_path: Path):
        # yarn lockfile path
        pkg = {"dependencies": {"left-pad": "^1.0.0"}}
        (tmp_path / "package.json").write_text(json.dumps(pkg))
        content = 'left-pad@^1.0.0:\n  version "1.3.0"\n'
        (tmp_path / "yarn.lock").write_text(content)
        pkgs = self.loader.parse(tmp_path)
        lp = next((p for p in pkgs if p.name == "left-pad"), None)
        assert lp is not None
        assert lp.version == "1.3.0"


# ============================================================================
# RustDependencyLoader
# ============================================================================
class TestRustDependencyLoader:
    """Tests for RustDependencyLoader."""

    def setup_method(self):
        self.loader = RustDependencyLoader()

    # -- detect ---------------------------------------------------------------

    def test_detect_cargo_toml(self, tmp_path: Path):
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "hello"')
        assert self.loader.detect(tmp_path) is True

    def test_detect_cargo_lock_only(self, tmp_path: Path):
        (tmp_path / "Cargo.lock").write_text("")
        assert self.loader.detect(tmp_path) is True

    def test_detect_nested_cargo_toml(self, tmp_path: Path):
        nested = tmp_path / "crates" / "core"
        nested.mkdir(parents=True)
        (nested / "Cargo.toml").write_text("")
        assert self.loader.detect(tmp_path) is True

    def test_detect_no_rust_project(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("")
        assert self.loader.detect(tmp_path) is False

    # -- normalize_version ----------------------------------------------------

    def test_normalize_version_empty(self):
        assert self.loader.normalize_version("") == "*"

    def test_normalize_version_git(self):
        assert self.loader.normalize_version("git:https://repo") == "git:https://repo"

    def test_normalize_version_path(self):
        assert self.loader.normalize_version("path:../local") == "path:../local"

    def test_normalize_version_semver(self):
        assert self.loader.normalize_version("^1.0") == "^1.0"

    # -- _create_package_from_spec -------------------------------------------

    def test_create_package_string_version(self):
        pkg = self.loader._create_package_from_spec("serde", "1.0.0", "dependencies")
        assert pkg is not None
        assert pkg.name == "serde"
        assert pkg.ecosystem == "crates.io"

    def test_create_package_dict_version(self):
        spec = {"version": "1.0", "features": ["derive"]}
        pkg = self.loader._create_package_from_spec("serde", spec, "dependencies")
        assert pkg is not None
        assert "features: derive" in pkg.description

    def test_create_package_dict_git(self):
        spec = {"git": "https://github.com/org/repo", "branch": "main"}
        pkg = self.loader._create_package_from_spec("my-crate", spec, "dependencies")
        assert pkg is not None
        assert pkg.version.startswith("git:")

    def test_create_package_dict_git_tag(self):
        spec = {"git": "https://github.com/org/repo", "tag": "v1.0"}
        pkg = self.loader._create_package_from_spec("my-crate", spec, "dependencies")
        assert pkg is not None
        assert "@v1.0" in pkg.version

    def test_create_package_dict_git_rev(self):
        spec = {"git": "https://github.com/org/repo", "rev": "abc123"}
        pkg = self.loader._create_package_from_spec("my-crate", spec, "dependencies")
        assert pkg is not None
        assert "@abc123" in pkg.version

    def test_create_package_dict_path(self):
        spec = {"path": "../my-local-crate"}
        pkg = self.loader._create_package_from_spec("local-crate", spec, "dependencies")
        assert pkg is not None
        assert pkg.version == "path:../my-local-crate"

    def test_create_package_dict_no_version(self):
        spec = {"optional": True}
        pkg = self.loader._create_package_from_spec("opt-crate", spec, "dependencies")
        assert pkg is not None
        assert pkg.version == "*"
        assert "optional" in pkg.description

    def test_create_package_invalid_name(self):
        pkg = self.loader._create_package_from_spec("123-bad", "1.0", "dependencies")
        assert pkg is None

    def test_create_package_unknown_spec(self):
        pkg = self.loader._create_package_from_spec("crate", 42, "dependencies")
        assert pkg is None

    # -- _parse_standard_dependencies -----------------------------------------

    def test_parse_standard_dependencies(self):
        data = {
            "dependencies": {"serde": "1.0"},
            "dev-dependencies": {"criterion": "0.5"},
            "build-dependencies": {"cc": "1.0"},
        }
        pkgs = self.loader._parse_standard_dependencies(data)
        names = {p.name for p in pkgs}
        assert {"serde", "criterion", "cc"} == names

    def test_parse_standard_dependencies_missing_sections(self):
        assert self.loader._parse_standard_dependencies({}) == []

    # -- _parse_target_dependencies -------------------------------------------

    def test_parse_target_dependencies(self):
        data = {
            "target": {
                'cfg(target_os = "linux")': {
                    "dependencies": {"nix": "0.27"},
                }
            }
        }
        pkgs = self.loader._parse_target_dependencies(data)
        assert len(pkgs) == 1
        assert pkgs[0].name == "nix"
        assert "target:" in pkgs[0].description

    def test_parse_target_dependencies_empty(self):
        assert self.loader._parse_target_dependencies({}) == []

    def test_parse_target_dependencies_non_dict_value(self):
        data = {"target": {"linux": "not_a_dict"}}
        assert self.loader._parse_target_dependencies(data) == []

    # -- _parse_cargo_lock ----------------------------------------------------

    def test_parse_cargo_lock(self, tmp_path: Path):
        toml_content = (
            '[[package]]\nname = "serde"\nversion = "1.0.197"\n'
            'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
            "\n"
            '[[package]]\nname = "my-project"\nversion = "0.1.0"\n'
        )
        cl = tmp_path / "Cargo.lock"
        cl.write_text(toml_content)
        pkgs = self.loader._parse_cargo_lock(cl)
        # root package (no source) should be skipped
        assert len(pkgs) == 1
        assert pkgs[0].name == "serde"

    def test_parse_cargo_lock_empty(self, tmp_path: Path):
        cl = tmp_path / "Cargo.lock"
        cl.write_text("")
        assert self.loader._parse_cargo_lock(cl) == []

    # -- _is_root_package -----------------------------------------------------

    def test_is_root_package_true(self):
        assert self.loader._is_root_package({"name": "proj", "version": "0.1"}) is True

    def test_is_root_package_false(self):
        assert (
            self.loader._is_root_package({"name": "serde", "version": "1.0", "source": "registry+"})
            is False
        )

    # -- _determine_ecosystem_from_source ------------------------------------

    def test_ecosystem_crates_io(self):
        assert (
            self.loader._determine_ecosystem_from_source(
                "registry+https://github.com/rust-lang/crates.io-index"
            )
            == "crates.io"
        )

    def test_ecosystem_git(self):
        assert (
            self.loader._determine_ecosystem_from_source("git+https://github.com/org/repo") == "git"
        )

    def test_ecosystem_alt_registry(self):
        assert (
            self.loader._determine_ecosystem_from_source("registry+https://my-registry.com")
            == "crates.io"
        )

    def test_ecosystem_unknown(self):
        assert self.loader._determine_ecosystem_from_source("local") == "unknown"

    # -- _parse_workspace_members ---------------------------------------------

    def test_parse_workspace_members(self, tmp_path: Path):
        main_toml = '[workspace]\nmembers = ["crates/core"]\n'
        (tmp_path / "Cargo.toml").write_text(main_toml)
        member_dir = tmp_path / "crates" / "core"
        member_dir.mkdir(parents=True)
        member_toml = '[package]\nname = "core"\n\n[dependencies]\ntokio = "1.0"\n'
        (member_dir / "Cargo.toml").write_text(member_toml)
        pkgs = self.loader._parse_workspace_members(tmp_path, tmp_path / "Cargo.toml")
        assert any(p.name == "tokio" for p in pkgs)

    def test_parse_workspace_members_glob(self, tmp_path: Path):
        main_toml = '[workspace]\nmembers = ["crates/*"]\n'
        (tmp_path / "Cargo.toml").write_text(main_toml)
        member_dir = tmp_path / "crates" / "util"
        member_dir.mkdir(parents=True)
        member_toml = '[dependencies]\nlog = "0.4"\n'
        (member_dir / "Cargo.toml").write_text(member_toml)
        pkgs = self.loader._parse_workspace_members(tmp_path, tmp_path / "Cargo.toml")
        assert any(p.name == "log" for p in pkgs)

    def test_parse_workspace_members_no_workspace(self, tmp_path: Path):
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "app"\n')
        pkgs = self.loader._parse_workspace_members(tmp_path, tmp_path / "Cargo.toml")
        assert pkgs == []

    def test_parse_workspace_members_empty_file(self, tmp_path: Path):
        (tmp_path / "Cargo.toml").write_text("")
        pkgs = self.loader._parse_workspace_members(tmp_path, tmp_path / "Cargo.toml")
        assert pkgs == []

    def test_parse_workspace_exclude(self, tmp_path: Path):
        main_toml = '[workspace]\nmembers = ["crates/*"]\nexclude = ["crates/skip"]\n'
        (tmp_path / "Cargo.toml").write_text(main_toml)
        # included member
        inc = tmp_path / "crates" / "inc"
        inc.mkdir(parents=True)
        (inc / "Cargo.toml").write_text('[dependencies]\nlog = "0.4"\n')
        # excluded member
        exc = tmp_path / "crates" / "skip"
        exc.mkdir(parents=True)
        (exc / "Cargo.toml").write_text('[dependencies]\nsecret = "0.1"\n')
        pkgs = self.loader._parse_workspace_members(tmp_path, tmp_path / "Cargo.toml")
        names = {p.name for p in pkgs}
        assert "log" in names
        assert "secret" not in names

    # -- _merge_package_data --------------------------------------------------

    def test_merge_adds_transitive(self):
        manifest = [ExternalPackage(name="serde", version="^1.0", ecosystem="crates.io")]
        lock = [
            ExternalPackage(name="serde", version="1.0.197", ecosystem="crates.io"),
            ExternalPackage(name="serde_derive", version="1.0.197", ecosystem="crates.io"),
        ]
        merged = self.loader._merge_package_data(manifest, lock)
        names = {p.name for p in merged}
        assert "serde" in names
        assert "serde_derive" in names
        serde = next(p for p in merged if p.name == "serde")
        assert serde.version == "1.0.197"

    # -- _deduplicate_packages ------------------------------------------------

    def test_deduplicate(self):
        pkgs = [
            ExternalPackage(name="x", version="1"),
            ExternalPackage(name="x", version="2"),
        ]
        result = self.loader._deduplicate_packages(pkgs)
        assert len(result) == 1

    # -- parse (integration) --------------------------------------------------

    def test_parse_full(self, tmp_path: Path):
        cargo_toml = '[package]\nname = "app"\n\n[dependencies]\nserde = "1.0"\n'
        cargo_lock = (
            '[[package]]\nname = "app"\nversion = "0.1.0"\n\n'
            '[[package]]\nname = "serde"\nversion = "1.0.197"\n'
            'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        )
        (tmp_path / "Cargo.toml").write_text(cargo_toml)
        (tmp_path / "Cargo.lock").write_text(cargo_lock)
        pkgs = self.loader.parse(tmp_path)
        assert any(p.name == "serde" for p in pkgs)

    def test_parse_no_tomllib(self, tmp_path: Path):
        (tmp_path / "Cargo.toml").write_text('[dependencies]\nserde = "1.0"\n')
        with patch(
            "processing.architecture.dependency_plugins.rust_loader.TOMLLIB_AVAILABLE", False
        ):
            with patch("processing.architecture.dependency_plugins.rust_loader.tomllib", None):
                pkgs = self.loader.parse(tmp_path)
                assert pkgs == []


# ============================================================================
# SystemPackageLoader
# ============================================================================
class TestSystemPackageLoader:
    """Tests for SystemPackageLoader."""

    def setup_method(self):
        self.loader = SystemPackageLoader()

    # -- detect ---------------------------------------------------------------

    def test_detect_always_true(self, tmp_path: Path):
        assert self.loader.detect(tmp_path) is True

    # -- _detect_package_manager ----------------------------------------------

    def test_detect_pm_apt(self):
        with patch.object(self.loader, "_command_exists", side_effect=lambda c: c == "apt-get"):
            self.loader._detect_package_manager()
            assert self.loader._package_manager == PackageManager.APT

    def test_detect_pm_yum(self):
        def exists(c):
            return c == "yum"

        with patch.object(self.loader, "_command_exists", side_effect=exists):
            self.loader._detect_package_manager()
            assert self.loader._package_manager == PackageManager.YUM

    def test_detect_pm_dnf(self):
        with patch.object(self.loader, "_command_exists", side_effect=lambda c: c == "dnf"):
            self.loader._detect_package_manager()
            assert self.loader._package_manager == PackageManager.DNF

    def test_detect_pm_brew(self):
        with patch.object(self.loader, "_command_exists", side_effect=lambda c: c == "brew"):
            self.loader._detect_package_manager()
            assert self.loader._package_manager == PackageManager.BREW

    def test_detect_pm_pacman(self):
        with patch.object(self.loader, "_command_exists", side_effect=lambda c: c == "pacman"):
            self.loader._detect_package_manager()
            assert self.loader._package_manager == PackageManager.PACMAN

    def test_detect_pm_apk(self):
        with patch.object(self.loader, "_command_exists", side_effect=lambda c: c == "apk"):
            self.loader._detect_package_manager()
            assert self.loader._package_manager == PackageManager.APK

    def test_detect_pm_default(self):
        with patch.object(self.loader, "_command_exists", return_value=False):
            self.loader._detect_package_manager()
            assert self.loader._package_manager == PackageManager.APT
            assert self.loader._detected_os == "unknown"

    # -- _command_exists ------------------------------------------------------

    def test_command_exists_true(self):
        with patch(
            "processing.architecture.dependency_plugins.system_package_loader.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock()
            assert self.loader._command_exists("ls") is True

    def test_command_exists_false(self):
        import subprocess

        with patch(
            "processing.architecture.dependency_plugins.system_package_loader.subprocess.run"
        ) as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "which")
            assert self.loader._command_exists("nonexistent") is False

    def test_command_exists_file_not_found(self):
        with patch(
            "processing.architecture.dependency_plugins.system_package_loader.subprocess.run"
        ) as mock_run:
            mock_run.side_effect = FileNotFoundError
            assert self.loader._command_exists("nonexistent") is False

    # -- _extract_python_packages ---------------------------------------------

    def test_extract_from_requirements_txt(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("numpy>=1.21\npandas==1.5.0\n# comment\n")
        pkgs = self.loader._extract_python_packages(tmp_path)
        assert "numpy" in pkgs
        assert "pandas" in pkgs

    def test_extract_from_setup_py(self, tmp_path: Path):
        setup_content = "setup(\n    install_requires=['requests>=2.28', 'psycopg2'],\n)"
        (tmp_path / "setup.py").write_text(setup_content)
        pkgs = self.loader._extract_python_packages(tmp_path)
        assert "requests" in pkgs
        assert "psycopg2" in pkgs

    def test_extract_from_pyproject(self, tmp_path: Path):
        content = '[project]\ndependencies = [\n"numpy" >= "1.21",\n]\n'
        (tmp_path / "pyproject.toml").write_text(content)
        pkgs = self.loader._extract_python_packages(tmp_path)
        assert "numpy" in pkgs

    # -- _parse_setup_py ------------------------------------------------------

    def test_parse_setup_py_system_requirements(self, tmp_path: Path):
        content = "# system-requirements: libpq-dev build-essential\nsetup()"
        (tmp_path / "setup.py").write_text(content)
        deps = self.loader._parse_setup_py(tmp_path)
        names = {d.name for d in deps}
        assert "libpq-dev" in names
        assert "build-essential" in names

    def test_parse_setup_py_apt_get(self, tmp_path: Path):
        content = "# apt-get install libssl-dev\nsetup()"
        (tmp_path / "setup.py").write_text(content)
        deps = self.loader._parse_setup_py(tmp_path)
        names = {d.name for d in deps}
        assert "libssl-dev" in names

    def test_parse_setup_py_no_file(self, tmp_path: Path):
        deps = self.loader._parse_setup_py(tmp_path)
        assert deps == set()

    # -- _parse_pyproject_toml ------------------------------------------------

    def test_parse_pyproject_manylinux(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            "[build-system]\nrequires = ['setuptools', 'wheel']\n"
        )
        deps = self.loader._parse_pyproject_toml(tmp_path)
        names = {d.name for d in deps}
        assert "build-essential" in names

    def test_parse_pyproject_maturin(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text('[build-system]\nrequires = ["maturin"]\n')
        deps = self.loader._parse_pyproject_toml(tmp_path)
        names = {d.name for d in deps}
        assert "cargo" in names
        assert "rustc" in names

    def test_parse_pyproject_no_file(self, tmp_path: Path):
        assert self.loader._parse_pyproject_toml(tmp_path) == set()

    def test_parse_pyproject_no_indicators(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 120\n")
        assert self.loader._parse_pyproject_toml(tmp_path) == set()

    # -- _check_error_logs ----------------------------------------------------

    def test_check_error_logs_python_h(self, tmp_path: Path):
        (tmp_path / "build.log").write_text("fatal error: Python.h: No such file or directory")
        deps = self.loader._check_error_logs(tmp_path)
        names = {d.name for d in deps}
        assert "python3-dev" in names

    def test_check_error_logs_openssl(self, tmp_path: Path):
        (tmp_path / "install.log").write_text(
            "fatal error: openssl/ssl.h: No such file or directory"
        )
        deps = self.loader._check_error_logs(tmp_path)
        names = {d.name for d in deps}
        assert "libssl-dev" in names

    def test_check_error_logs_no_files(self, tmp_path: Path):
        assert self.loader._check_error_logs(tmp_path) == set()

    def test_check_error_logs_pg_config(self, tmp_path: Path):
        (tmp_path / "pip-log.txt").write_text("pg_config executable not found")
        deps = self.loader._check_error_logs(tmp_path)
        names = {d.name for d in deps}
        assert "libpq-dev" in names

    # -- find_package_for_library ---------------------------------------------

    def test_find_package_not_apt(self):
        self.loader._package_manager = PackageManager.BREW
        assert self.loader.find_package_for_library("libfoo") is None

    def test_find_package_cached(self):
        self.loader._package_manager = PackageManager.APT
        self.loader._apt_file_cache["libfoo.so"] = ["libfoo-dev"]
        assert self.loader.find_package_for_library("libfoo.so") == "libfoo-dev"

    def test_find_package_cached_empty(self):
        self.loader._package_manager = PackageManager.APT
        self.loader._apt_file_cache["libfoo.so"] = []
        assert self.loader.find_package_for_library("libfoo.so") is None

    def test_find_package_apt_file_success(self):
        self.loader._package_manager = PackageManager.APT
        mock_result = MagicMock()
        mock_result.stdout = "libfoo-dev: /usr/lib/libfoo.so\nlibfoo1: /usr/lib/libfoo.so.1\n"
        with patch(
            "processing.architecture.dependency_plugins.system_package_loader.subprocess.run",
            return_value=mock_result,
        ):
            result = self.loader.find_package_for_library("libfoo.so")
            assert result == "libfoo-dev"

    def test_find_package_apt_file_no_dev(self):
        self.loader._package_manager = PackageManager.APT
        mock_result = MagicMock()
        mock_result.stdout = "libfoo1: /usr/lib/libfoo.so.1\n"
        with patch(
            "processing.architecture.dependency_plugins.system_package_loader.subprocess.run",
            return_value=mock_result,
        ):
            result = self.loader.find_package_for_library("libfoo.so")
            assert result == "libfoo1"

    def test_find_package_apt_file_failure(self):
        import subprocess

        self.loader._package_manager = PackageManager.APT
        with patch(
            "processing.architecture.dependency_plugins.system_package_loader.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "apt-file"),
        ):
            assert self.loader.find_package_for_library("libfoo.so") is None

    # -- generate_install_command ---------------------------------------------

    def test_generate_install_command_apt(self):
        self.loader._package_manager = PackageManager.APT
        cmd = self.loader.generate_install_command(["libfoo-dev", "libbar-dev"])
        assert "apt-get" in cmd
        assert "libfoo-dev" in cmd

    def test_generate_install_command_brew(self):
        self.loader._package_manager = PackageManager.BREW
        cmd = self.loader.generate_install_command(["openssl"])
        assert "brew install" in cmd

    def test_generate_install_command_yum(self):
        self.loader._package_manager = PackageManager.YUM
        cmd = self.loader.generate_install_command(["gcc"])
        assert "yum install" in cmd

    def test_generate_install_command_dnf(self):
        self.loader._package_manager = PackageManager.DNF
        cmd = self.loader.generate_install_command(["gcc"])
        assert "dnf install" in cmd

    def test_generate_install_command_pacman(self):
        self.loader._package_manager = PackageManager.PACMAN
        cmd = self.loader.generate_install_command(["gcc"])
        assert "pacman -S" in cmd

    def test_generate_install_command_apk(self):
        self.loader._package_manager = PackageManager.APK
        cmd = self.loader.generate_install_command(["gcc"])
        assert "apk add" in cmd

    def test_generate_install_command_empty(self):
        assert self.loader.generate_install_command([]) == ""

    # -- parse (integration) --------------------------------------------------

    def test_parse_with_known_deps(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("psycopg2>=2.9\n")
        with patch.object(self.loader, "_command_exists", return_value=False):
            pkgs = self.loader.parse(tmp_path)
        names = {p.name for p in pkgs}
        assert "libpq-dev" in names

    # -- DetectedSystemPackage ------------------------------------------------

    def test_detected_system_package_hashable(self):
        d1 = DetectedSystemPackage(name="libfoo", package_manager=PackageManager.APT)
        d2 = DetectedSystemPackage(name="libfoo", package_manager=PackageManager.APT)
        assert d1 == d2
        assert len({d1, d2}) == 1

    def test_detected_system_package_fields(self):
        d = DetectedSystemPackage(
            name="libfoo-dev",
            package_manager=PackageManager.APT,
            library_name="libfoo.so",
            header_files=("foo.h",),
            description="test",
            required_by=("pkg",),
        )
        assert d.name == "libfoo-dev"
        assert d.library_name == "libfoo.so"
        assert d.header_files == ("foo.h",)
        assert d.required_by == ("pkg",)


# ============================================================================
# PoetryLoader
# ============================================================================
class TestPoetryLoader:
    """Tests for PoetryLoader."""

    def setup_method(self):
        self.loader = PoetryLoader()

    # -- detect ---------------------------------------------------------------

    def test_detect_poetry_project(self, tmp_path: Path):
        content = '[tool.poetry]\nname = "myproject"\nversion = "1.0"\n'
        (tmp_path / "pyproject.toml").write_text(content)
        assert self.loader.detect(tmp_path) is True

    def test_detect_not_poetry(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text('[build-system]\nrequires = ["setuptools"]\n')
        assert self.loader.detect(tmp_path) is False

    def test_detect_no_pyproject(self, tmp_path: Path):
        assert self.loader.detect(tmp_path) is False

    # -- _parse_dependencies_section -----------------------------------------

    def test_parse_deps_string_version(self):
        deps = {"requests": "^2.28.0", "python": "^3.11"}
        pkgs = self.loader._parse_dependencies_section(deps)
        names = {p.name for p in pkgs}
        assert "requests" in names
        assert "python" not in names  # python is skipped

    def test_parse_deps_dict_version(self):
        deps = {"numpy": {"version": "^1.21", "optional": True}}
        pkgs = self.loader._parse_dependencies_section(deps)
        assert len(pkgs) == 1
        assert pkgs[0].version == "^1.21"

    def test_parse_deps_dict_no_version(self):
        deps = {"local-pkg": {"path": "../local"}}
        pkgs = self.loader._parse_dependencies_section(deps)
        assert len(pkgs) == 1
        assert pkgs[0].version == "*"

    # -- _parse_project_dependencies -----------------------------------------

    def test_parse_project_deps_with_version(self):
        deps = ["requests>=2.28.0", "flask~=2.0"]
        pkgs = self.loader._parse_project_dependencies(deps)
        assert len(pkgs) == 2
        assert pkgs[0].name == "requests"
        assert pkgs[0].version == ">=2.28.0"
        assert pkgs[1].name == "flask"
        assert pkgs[1].version == "~=2.0"

    def test_parse_project_deps_no_version(self):
        deps = ["requests"]
        pkgs = self.loader._parse_project_dependencies(deps)
        assert len(pkgs) == 1
        assert pkgs[0].version == "*"

    def test_parse_project_deps_all_operators(self):
        ops = [">=", "<=", "==", "~=", "!=", ">", "<"]
        for op in ops:
            deps = [f"pkg{op}1.0"]
            pkgs = self.loader._parse_project_dependencies(deps)
            assert pkgs[0].version == f"{op}1.0", f"Failed for operator {op}"

    # -- _parse_pyproject_toml ------------------------------------------------

    def test_parse_pyproject_full(self, tmp_path: Path):
        content = (
            "[tool.poetry]\n"
            'name = "app"\n'
            "\n"
            "[tool.poetry.dependencies]\n"
            'python = "^3.11"\n'
            'requests = "^2.28"\n'
            "\n"
            "[tool.poetry.dev-dependencies]\n"
            'pytest = "^7.0"\n'
            "\n"
            "[tool.poetry.group.lint.dependencies]\n"
            'ruff = "^0.1"\n'
        )
        (tmp_path / "pyproject.toml").write_text(content)
        pkgs = self.loader._parse_pyproject_toml(tmp_path)
        names = {p.name for p in pkgs}
        assert "requests" in names
        assert "pytest" in names
        assert "ruff" in names
        assert "python" not in names

    def test_parse_pyproject_project_format(self, tmp_path: Path):
        content = '[tool.poetry]\nname = "app"\n\n[project]\ndependencies = ["requests>=2.28"]\n'
        (tmp_path / "pyproject.toml").write_text(content)
        pkgs = self.loader._parse_pyproject_toml(tmp_path)
        assert any(p.name == "requests" for p in pkgs)

    def test_parse_pyproject_no_file(self, tmp_path: Path):
        assert self.loader._parse_pyproject_toml(tmp_path) == []

    def test_parse_pyproject_exception_fallback(self, tmp_path: Path):
        # Write intentionally malformed content that tomli can't parse
        (tmp_path / "pyproject.toml").write_text(
            "[tool.poetry]\nname = 'app'\n\n[tool.poetry.dependencies]\nrequests = '^2.28'\n"
        )
        # The parser will fallback to _parse_simple_pyproject
        pkgs = self.loader._parse_pyproject_toml(tmp_path)
        # Should still get results from fallback
        assert isinstance(pkgs, list)

    # -- _parse_poetry_lock ---------------------------------------------------

    def test_parse_poetry_lock(self, tmp_path: Path):
        content = (
            '[[package]]\nname = "requests"\nversion = "2.31.0"\n'
            "\n"
            '[[package]]\nname = "urllib3"\nversion = "2.0.4"\n'
        )
        (tmp_path / "poetry.lock").write_text(content)
        pkgs = self.loader._parse_poetry_lock(tmp_path)
        assert len(pkgs) == 2
        assert pkgs[0].name == "requests"

    def test_parse_poetry_lock_no_file(self, tmp_path: Path):
        assert self.loader._parse_poetry_lock(tmp_path) == []

    def test_parse_poetry_lock_empty_entries(self, tmp_path: Path):
        content = '[[package]]\nname = ""\nversion = ""\n'
        (tmp_path / "poetry.lock").write_text(content)
        pkgs = self.loader._parse_poetry_lock(tmp_path)
        assert len(pkgs) == 0

    # -- _parse_simple (no tomli) --------------------------------------------

    def test_parse_simple(self, tmp_path: Path):
        content = '[tool.poetry.dependencies]\npython = "^3.11"\nrequests = "^2.28"\n'
        (tmp_path / "pyproject.toml").write_text(content)
        pkgs = self.loader._parse_simple(tmp_path)
        names = {p.name for p in pkgs}
        assert "requests" in names
        assert "python" not in names

    # -- _parse_simple_pyproject ----------------------------------------------

    def test_parse_simple_pyproject(self, tmp_path: Path):
        content = (
            "[tool.poetry.dependencies]\n"
            'requests = "^2.28"\n'
            "# a comment\n"
            "[tool.poetry.dev-dependencies]\n"
            'pytest = "^7.0"\n'
            "[tool.other]\n"
            'foo = "bar"\n'
        )
        (tmp_path / "pyproject.toml").write_text(content)
        pkgs = self.loader._parse_simple_pyproject(tmp_path)
        names = {p.name for p in pkgs}
        assert "requests" in names
        assert "pytest" in names
        assert "foo" not in names

    def test_parse_simple_pyproject_no_file(self, tmp_path: Path):
        assert self.loader._parse_simple_pyproject(tmp_path) == []

    def test_parse_simple_pyproject_group_deps(self, tmp_path: Path):
        content = '[tool.poetry.group.test.dependencies]\npytest-cov = "^4.0"\n'
        (tmp_path / "pyproject.toml").write_text(content)
        pkgs = self.loader._parse_simple_pyproject(tmp_path)
        assert any(p.name == "pytest-cov" for p in pkgs)

    # -- _parse_simple_lock ---------------------------------------------------

    def test_parse_simple_lock(self, tmp_path: Path):
        content = (
            "[[package]]\n"
            'name = "requests"\n'
            'version = "2.31.0"\n'
            "\n"
            "[[package]]\n"
            'name = "urllib3"\n'
            'version = "2.0.4"\n'
        )
        (tmp_path / "poetry.lock").write_text(content)
        pkgs = self.loader._parse_simple_lock(tmp_path)
        assert len(pkgs) == 2

    def test_parse_simple_lock_no_file(self, tmp_path: Path):
        assert self.loader._parse_simple_lock(tmp_path) == []

    # -- parse (integration) --------------------------------------------------

    def test_parse_with_lock(self, tmp_path: Path):
        pyproject = (
            "[tool.poetry]\nname = 'app'\n\n"
            "[tool.poetry.dependencies]\n"
            'python = "^3.11"\n'
            'requests = "^2.28"\n'
        )
        lock = '[[package]]\nname = "requests"\nversion = "2.31.0"\n'
        (tmp_path / "pyproject.toml").write_text(pyproject)
        (tmp_path / "poetry.lock").write_text(lock)
        pkgs = self.loader.parse(tmp_path)
        req = next((p for p in pkgs if p.name == "requests"), None)
        assert req is not None
        assert req.version == "2.31.0"  # locked version
        assert req.current_version == "2.31.0"

    def test_parse_no_tomli(self, tmp_path: Path):
        content = '[tool.poetry.dependencies]\nrequests = "^2.28"\n'
        (tmp_path / "pyproject.toml").write_text(content)
        with patch("processing.architecture.dependency_plugins.poetry_loader.tomli", None):
            pkgs = self.loader.parse(tmp_path)
            assert any(p.name == "requests" for p in pkgs)


# ============================================================================
# UvLoader
# ============================================================================
class TestUvLoader:
    """Tests for UvLoader."""

    def setup_method(self):
        self.loader = UvLoader()

    # -- detect ---------------------------------------------------------------

    def test_detect_uv_lock(self, tmp_path: Path):
        (tmp_path / "uv.lock").write_text("")
        assert self.loader.detect(tmp_path) is True

    def test_detect_no_lock(self, tmp_path: Path):
        assert self.loader.detect(tmp_path) is False

    # -- _parse_toml_simple ---------------------------------------------------

    def test_parse_toml_simple(self):
        content = (
            "[[package]]\n"
            'name = "requests"\n'
            'version = "2.31.0"\n'
            "\n"
            "[[package]]\n"
            'name = "urllib3"\n'
            'version = "2.0.4"\n'
        )
        result = self.loader._parse_toml_simple(content)
        assert len(result["package"]) == 2
        assert result["package"][0]["name"] == "requests"
        assert result["package"][1]["version"] == "2.0.4"

    def test_parse_toml_simple_empty(self):
        result = self.loader._parse_toml_simple("")
        assert result == {"package": []}

    def test_parse_toml_simple_bad_lines(self):
        content = '[[package]]\nbadline_no_equals\nname = "pkg"\n'
        result = self.loader._parse_toml_simple(content)
        assert len(result["package"]) == 1
        assert result["package"][0]["name"] == "pkg"

    # -- _parse_uv_lock_content -----------------------------------------------

    def test_parse_uv_lock_content_with_tomli(self):
        content = '[[package]]\nname = "foo"\nversion = "1.0"\n'
        result = self.loader._parse_uv_lock_content(content)
        assert len(result["package"]) == 1

    def test_parse_uv_lock_content_no_tomli(self):
        content = '[[package]]\nname = "foo"\nversion = "1.0"\n'
        with patch.dict(sys.modules, {"tomli": None}):
            # Force ImportError by patching the import
            def mock_parse(c):
                # Simulate tomli import failure
                return self.loader._parse_toml_simple(c)

            with patch.object(self.loader, "_parse_uv_lock_content", side_effect=mock_parse):
                result = mock_parse(content)
                assert len(result["package"]) == 1

    # -- _parse_simple_lock ---------------------------------------------------

    def test_parse_simple_lock(self, tmp_path: Path):
        content = (
            "[[package]]\n"
            'name = "requests"\n'
            'version = "2.31.0"\n'
            "\n"
            "[[package]]\n"
            'name = "urllib3"\n'
            'version = "2.0.4"\n'
        )
        (tmp_path / "uv.lock").write_text(content)
        pkgs = self.loader._parse_simple_lock(tmp_path)
        assert len(pkgs) == 2
        assert pkgs[0].name == "requests"
        assert pkgs[1].version == "2.0.4"

    def test_parse_simple_lock_no_file(self, tmp_path: Path):
        assert self.loader._parse_simple_lock(tmp_path) == []

    def test_parse_simple_lock_name_without_version(self, tmp_path: Path):
        content = 'name = "incomplete"\n'
        (tmp_path / "uv.lock").write_text(content)
        pkgs = self.loader._parse_simple_lock(tmp_path)
        # No version line follows, so it should not be added
        assert len(pkgs) == 0

    # -- parse (integration) --------------------------------------------------

    def test_parse_happy(self, tmp_path: Path):
        content = '[[package]]\nname = "requests"\nversion = "2.31.0"\n'
        (tmp_path / "uv.lock").write_text(content)
        pkgs = self.loader.parse(tmp_path)
        assert len(pkgs) == 1
        assert pkgs[0].name == "requests"
        assert pkgs[0].ecosystem == "pypi"

    def test_parse_no_lock_file(self, tmp_path: Path):
        assert self.loader.parse(tmp_path) == []

    def test_parse_empty_entries(self, tmp_path: Path):
        content = '[[package]]\nname = ""\nversion = ""\n'
        (tmp_path / "uv.lock").write_text(content)
        pkgs = self.loader.parse(tmp_path)
        assert len(pkgs) == 0

    def test_parse_exception_fallback(self, tmp_path: Path):
        content = '[[package]]\nname = "requests"\nversion = "2.31.0"\n'
        (tmp_path / "uv.lock").write_text(content)
        with patch.object(
            self.loader, "_parse_uv_lock_content", side_effect=Exception("parse error")
        ):
            pkgs = self.loader.parse(tmp_path)
            # Falls back to _parse_simple_lock
            assert len(pkgs) == 1
            assert pkgs[0].name == "requests"


# ============================================================================
# PipLoader
# ============================================================================
class TestPipLoader:
    """Tests for PipLoader."""

    def setup_method(self):
        self.loader = PipLoader()

    # -- detect ---------------------------------------------------------------

    def test_detect_requirements_txt(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("flask\n")
        assert self.loader.detect(tmp_path) is True

    def test_detect_requirements_dir(self, tmp_path: Path):
        req_dir = tmp_path / "requirements"
        req_dir.mkdir()
        (req_dir / "base.txt").write_text("flask\n")
        assert self.loader.detect(tmp_path) is True

    def test_detect_requirements_dev(self, tmp_path: Path):
        req_dir = tmp_path / "requirements"
        req_dir.mkdir()
        (req_dir / "dev.txt").write_text("pytest\n")
        assert self.loader.detect(tmp_path) is True

    def test_detect_nested_requirements(self, tmp_path: Path):
        nested = tmp_path / "subdir"
        nested.mkdir()
        (nested / "requirements-test.txt").write_text("pytest\n")
        assert self.loader.detect(tmp_path) is True

    def test_detect_no_requirements(self, tmp_path: Path):
        (tmp_path / "setup.py").write_text("setup()")
        assert self.loader.detect(tmp_path) is False

    # -- _parse_package_spec --------------------------------------------------

    def test_parse_spec_exact_version(self):
        pkg = self.loader._parse_package_spec("requests==2.31.0")
        assert pkg is not None
        assert pkg.name == "requests"
        assert pkg.version == "2.31.0"

    def test_parse_spec_gte(self):
        pkg = self.loader._parse_package_spec("flask>=2.0.0")
        assert pkg is not None
        assert pkg.name == "flask"
        assert pkg.version == ">=2.0.0"

    def test_parse_spec_compatible(self):
        pkg = self.loader._parse_package_spec("django~=4.2")
        assert pkg is not None
        assert pkg.name == "django"
        assert pkg.version == "~=4.2"

    def test_parse_spec_no_version(self):
        pkg = self.loader._parse_package_spec("requests")
        assert pkg is not None
        assert pkg.name == "requests"
        assert pkg.version == "*"

    def test_parse_spec_with_extras(self):
        pkg = self.loader._parse_package_spec("uvicorn[standard]==0.23.0")
        assert pkg is not None
        assert pkg.name == "uvicorn"
        assert pkg.version == "0.23.0"

    def test_parse_spec_inline_comment(self):
        pkg = self.loader._parse_package_spec("requests==2.31.0  # needed for API")
        assert pkg is not None
        assert pkg.name == "requests"

    def test_parse_spec_empty_after_comment(self):
        pkg = self.loader._parse_package_spec("# just a comment")
        # After stripping comment, spec is empty
        assert pkg is None

    def test_parse_spec_empty(self):
        assert self.loader._parse_package_spec("") is None

    # -- _parse_requirements_file --------------------------------------------

    def test_parse_requirements_file(self, tmp_path: Path):
        content = (
            "# dependencies\n"
            "requests==2.31.0\n"
            "flask>=2.0\n"
            "\n"
            "-r base.txt\n"
            "-e git+https://...\n"
            "--editable .\n"
            "gunicorn\n"
        )
        f = tmp_path / "requirements.txt"
        f.write_text(content)
        pkgs = self.loader._parse_requirements_file(f)
        names = {p.name for p in pkgs}
        assert "requests" in names
        assert "flask" in names
        assert "gunicorn" in names
        # -r, -e, --editable lines should be skipped
        assert len(pkgs) == 3

    def test_parse_requirements_file_empty(self, tmp_path: Path):
        f = tmp_path / "requirements.txt"
        f.write_text("")
        assert self.loader._parse_requirements_file(f) == []

    # -- parse (integration) --------------------------------------------------

    def test_parse_deduplicates(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\nrequests>=2.28\n")
        pkgs = self.loader.parse(tmp_path)
        assert len(pkgs) == 1
        assert pkgs[0].name == "requests"

    def test_parse_multiple_files(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("flask\n")
        req_dir = tmp_path / "requirements"
        req_dir.mkdir()
        (req_dir / "dev.txt").write_text("pytest\n")
        pkgs = self.loader.parse(tmp_path)
        names = {p.name for p in pkgs}
        assert "flask" in names
        assert "pytest" in names

    def test_parse_no_files(self, tmp_path: Path):
        assert self.loader.parse(tmp_path) == []

    def test_parse_prod_requirements(self, tmp_path: Path):
        req_dir = tmp_path / "requirements"
        req_dir.mkdir()
        (req_dir / "prod.txt").write_text("gunicorn\n")
        (req_dir / "production.txt").write_text("uvicorn\n")
        pkgs = self.loader.parse(tmp_path)
        names = {p.name for p in pkgs}
        assert "gunicorn" in names
        assert "uvicorn" in names
