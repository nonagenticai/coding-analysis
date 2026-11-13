"""Rust Cargo project management module.

This module handles Cargo project structure analysis, dependency management,
and project metadata extraction for Rust projects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from models.core_models import CoreCodeEntity, EntityType


class RustCargoManager:
    """Handles Cargo project management and analysis."""

    def __init__(self):
        """Initialize the Cargo manager."""
        self.cargo_toml_cache: dict[str, dict[str, Any]] = {}

    def find_cargo_toml(self, file_path: Path) -> Path | None:
        """Find the nearest Cargo.toml file for a given Rust file.

        Args:
            file_path: Path to a Rust source file

        Returns:
            Path to the nearest Cargo.toml file, or None if not found
        """
        current = file_path.parent if file_path.is_file() else file_path

        while current != current.parent:
            cargo_toml = current / "Cargo.toml"
            if cargo_toml.exists():
                return cargo_toml
            current = current.parent

        return None

    def parse_cargo_toml(self, cargo_toml_path: Path) -> dict[str, Any]:
        """Parse a Cargo.toml file and extract project metadata.

        Args:
            cargo_toml_path: Path to the Cargo.toml file

        Returns:
            Dictionary containing parsed Cargo.toml content
        """
        if str(cargo_toml_path) in self.cargo_toml_cache:
            return self.cargo_toml_cache[str(cargo_toml_path)]

        try:
            with open(cargo_toml_path, encoding="utf-8") as f:
                cargo_data = toml.load(f)

            self.cargo_toml_cache[str(cargo_toml_path)] = cargo_data
            return cargo_data
        except (OSError, toml.TomlDecodeError):
            # Return empty dict if parsing fails
            return {}

    def get_project_metadata(self, cargo_toml_path: Path) -> dict[str, Any]:
        """Extract project metadata from Cargo.toml.

        Args:
            cargo_toml_path: Path to the Cargo.toml file

        Returns:
            Dictionary containing project metadata
        """
        cargo_data = self.parse_cargo_toml(cargo_toml_path)

        package_info = cargo_data.get("package", {})
        workspace_info = cargo_data.get("workspace", {})

        metadata = {
            "name": package_info.get("name", "unknown"),
            "version": package_info.get("version", "0.0.0"),
            "edition": package_info.get("edition", "2021"),
            "authors": package_info.get("authors", []),
            "description": package_info.get("description", ""),
            "license": package_info.get("license", ""),
            "repository": package_info.get("repository", ""),
            "homepage": package_info.get("homepage", ""),
            "documentation": package_info.get("documentation", ""),
            "keywords": package_info.get("keywords", []),
            "categories": package_info.get("categories", []),
            "is_workspace": "workspace" in cargo_data,
            "workspace_members": workspace_info.get("members", []),
        }

        return metadata

    def get_dependencies(self, cargo_toml_path: Path) -> dict[str, list[dict[str, Any]]]:
        """Extract dependencies from Cargo.toml.

        Args:
            cargo_toml_path: Path to the Cargo.toml file

        Returns:
            Dictionary categorizing dependencies by type
        """
        cargo_data = self.parse_cargo_toml(cargo_toml_path)

        dependencies = {
            "dependencies": [],
            "dev_dependencies": [],
            "build_dependencies": [],
        }

        # Parse regular dependencies
        deps = cargo_data.get("dependencies", {})
        for name, spec in deps.items():
            dep_info = self._parse_dependency_spec(name, spec)
            dependencies["dependencies"].append(dep_info)

        # Parse dev dependencies
        dev_deps = cargo_data.get("dev-dependencies", {})
        for name, spec in dev_deps.items():
            dep_info = self._parse_dependency_spec(name, spec)
            dependencies["dev_dependencies"].append(dep_info)

        # Parse build dependencies
        build_deps = cargo_data.get("build-dependencies", {})
        for name, spec in build_deps.items():
            dep_info = self._parse_dependency_spec(name, spec)
            dependencies["build_dependencies"].append(dep_info)

        return dependencies

    def _parse_dependency_spec(self, name: str, spec: Any) -> dict[str, Any]:
        """Parse a dependency specification.

        Args:
            name: Dependency name
            spec: Dependency specification (string, dict, etc.)

        Returns:
            Dictionary with parsed dependency information
        """
        dep_info = {
            "name": name,
            "version": None,
            "source": "crates.io",
            "git": None,
            "branch": None,
            "tag": None,
            "rev": None,
            "path": None,
            "features": [],
            "optional": False,
            "default_features": True,
        }

        if isinstance(spec, str):
            # Simple version specification
            dep_info["version"] = spec
        elif isinstance(spec, dict):
            # Complex specification
            dep_info["version"] = spec.get("version")
            dep_info["git"] = spec.get("git")
            dep_info["branch"] = spec.get("branch")
            dep_info["tag"] = spec.get("tag")
            dep_info["rev"] = spec.get("rev")
            dep_info["path"] = spec.get("path")
            dep_info["features"] = spec.get("features", [])
            dep_info["optional"] = spec.get("optional", False)
            dep_info["default_features"] = spec.get("default-features", True)

            if dep_info["git"]:
                dep_info["source"] = "git"
            elif dep_info["path"]:
                dep_info["source"] = "path"

        return dep_info

    def get_build_configuration(self, cargo_toml_path: Path) -> dict[str, Any]:
        """Extract build configuration from Cargo.toml.

        Args:
            cargo_toml_path: Path to the Cargo.toml file

        Returns:
            Dictionary containing build configuration
        """
        cargo_data = self.parse_cargo_toml(cargo_toml_path)

        build_config = {
            "bin": cargo_data.get("bin", []),
            "lib": cargo_data.get("lib", {}),
            "test": cargo_data.get("test", []),
            "bench": cargo_data.get("bench", []),
            "example": cargo_data.get("example", []),
            "profile": cargo_data.get("profile", {}),
            "features": cargo_data.get("features", {}),
            "target": cargo_data.get("target", {}),
        }

        return build_config

    def get_workspace_info(self, cargo_toml_path: Path) -> dict[str, Any] | None:
        """Get workspace information if this is a workspace.

        Args:
            cargo_toml_path: Path to the Cargo.toml file

        Returns:
            Workspace information or None if not a workspace
        """
        cargo_data = self.parse_cargo_toml(cargo_toml_path)
        workspace_data = cargo_data.get("workspace")

        if not workspace_data:
            return None

        workspace_info = {
            "members": workspace_data.get("members", []),
            "exclude": workspace_data.get("exclude", []),
            "default_members": workspace_data.get("default-members", []),
            "dependencies": workspace_data.get("dependencies", {}),
            "metadata": workspace_data.get("metadata", {}),
        }

        return workspace_info

    def find_source_directories(self, cargo_toml_path: Path) -> dict[str, Path]:
        """Find source directories for a Cargo project.

        Args:
            cargo_toml_path: Path to the Cargo.toml file

        Returns:
            Dictionary mapping source types to their directories
        """
        project_root = cargo_toml_path.parent
        cargo_data = self.parse_cargo_toml(cargo_toml_path)

        source_dirs = {}

        # Library source
        lib_config = cargo_data.get("lib", {})
        lib_path = lib_config.get("path", "src/lib.rs")
        if not Path(lib_path).is_absolute():
            lib_path = project_root / lib_path
        else:
            lib_path = Path(lib_path)

        if lib_path.exists():
            source_dirs["lib"] = lib_path.parent

        # Binary sources
        bin_configs = cargo_data.get("bin", [])
        if not bin_configs and (project_root / "src" / "main.rs").exists():
            # Default binary
            source_dirs["bin"] = project_root / "src"
        else:
            for bin_config in bin_configs:
                bin_path = bin_config.get("path", f"src/bin/{bin_config['name']}.rs")
                if not Path(bin_path).is_absolute():
                    bin_path = project_root / bin_path
                else:
                    bin_path = Path(bin_path)

                if bin_path.exists():
                    source_dirs[f"bin_{bin_config['name']}"] = bin_path.parent

        # Test sources
        test_configs = cargo_data.get("test", [])
        for test_config in test_configs:
            test_path = test_config.get("path", f"tests/{test_config['name']}.rs")
            if not Path(test_path).is_absolute():
                test_path = project_root / test_path
            else:
                test_path = Path(test_path)

            if test_path.exists():
                source_dirs[f"test_{test_config['name']}"] = test_path.parent

        # Examples
        example_configs = cargo_data.get("example", [])
        for example_config in example_configs:
            example_path = example_config.get("path", f"examples/{example_config['name']}.rs")
            if not Path(example_path).is_absolute():
                example_path = project_root / example_path
            else:
                example_path = Path(example_path)

            if example_path.exists():
                source_dirs[f"example_{example_config['name']}"] = example_path.parent

        return source_dirs

    def get_manifest_entities(self, cargo_toml_path: Path) -> list[CoreCodeEntity]:
        """Create entities representing the Cargo manifest structure.

        Args:
            cargo_toml_path: Path to the Cargo.toml file

        Returns:
            List of entities representing manifest components
        """
        entities = []
        metadata = self.get_project_metadata(cargo_toml_path)
        dependencies = self.get_dependencies(cargo_toml_path)

        # Create project entity
        project_entity = CoreCodeEntity(
            id=f"project:{metadata['name']}:{cargo_toml_path}",
            name=metadata["name"],
            type=EntityType.MODULE,  # Representing project as a module
            language="rust",
            file_path=str(cargo_toml_path),
            line_number=1,
            end_line_number=None,
            column_start=0,
            column_end=0,
            visibility="public",
            code_snippet=None,
            docstring=metadata.get("description"),
            signature=None,
            source_code=None,
            created_at=None,
            updated_at=None,
            properties={
                "project_type": "cargo_project",
                "version": metadata["version"],
                "edition": metadata["edition"],
                "authors": metadata["authors"],
                "license": metadata["license"],
                "repository": metadata["repository"],
                "is_workspace": metadata["is_workspace"],
            },
        )
        entities.append(project_entity)

        # Create dependency entities
        for dep_type, deps in dependencies.items():
            for dep in deps:
                dep_entity = CoreCodeEntity(
                    id=f"dependency:{dep['name']}:{cargo_toml_path}",
                    name=dep["name"],
                    type=EntityType.IMPORT,  # Dependencies are like imports
                    language="rust",
                    file_path=str(cargo_toml_path),
                    line_number=1,  # Would need more sophisticated parsing for exact line
                    end_line_number=None,
                    column_start=0,
                    column_end=0,
                    visibility="private",
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={
                        "dependency_type": dep_type,
                        "version": dep["version"],
                        "source": dep["source"],
                        "git": dep["git"],
                        "path": dep["path"],
                        "features": dep["features"],
                        "optional": dep["optional"],
                    },
                )
                entities.append(dep_entity)

        return entities

    def is_test_file(self, file_path: Path) -> bool:
        """Determine if a Rust file is a test file.

        Args:
            file_path: Path to the Rust file

        Returns:
            True if the file is a test file
        """
        # Check if file is in tests directory
        if "tests" in file_path.parts:
            return True

        # Check for test naming patterns
        if file_path.name.endswith("_test.rs") or file_path.name.startswith("test_"):
            return True

        # Check if file contains test attributes (would need content analysis)
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                # Simple heuristic: file contains #[test] or #[cfg(test)]
                if "#[test]" in content or "#[cfg(test)]" in content:
                    return True
        except (OSError, UnicodeDecodeError):  # fmt: skip
            pass

        return False

    def get_crate_type(self, cargo_toml_path: Path, file_path: Path) -> str:
        """Determine the crate type for a given file.

        Args:
            cargo_toml_path: Path to the Cargo.toml file
            file_path: Path to the Rust file

        Returns:
            String indicating crate type ('lib', 'bin', 'test', 'example', 'bench')
        """
        cargo_data = self.parse_cargo_toml(cargo_toml_path)
        project_root = cargo_toml_path.parent
        relative_path = file_path.relative_to(project_root)

        # Check if it's a library file
        lib_config = cargo_data.get("lib", {})
        lib_path = Path(lib_config.get("path", "src/lib.rs"))
        if relative_path == lib_path:
            return "lib"

        # Check if it's a binary file
        bin_configs = cargo_data.get("bin", [])
        for bin_config in bin_configs:
            bin_path = Path(bin_config.get("path", f"src/bin/{bin_config['name']}.rs"))
            if relative_path == bin_path:
                return "bin"

        # Check default binary location
        if relative_path == Path("src/main.rs"):
            return "bin"

        # Check if it's a test file
        if self.is_test_file(file_path):
            return "test"

        # Check if it's an example
        if "examples" in relative_path.parts:
            return "example"

        # Check if it's a benchmark
        if "benches" in relative_path.parts:
            return "bench"

        # Default to library
        return "lib"
