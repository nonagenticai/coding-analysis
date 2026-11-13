"""
CMake project management module for C++ projects.

This module handles CMake file parsing, project structure analysis,
and build system configuration detection for C++ projects.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, TypedDict


class CMakeTarget(TypedDict, total=False):
    """Type definition for a CMake target."""

    type: str
    name: str
    sources: list[str]
    library_type: str


class CMakeDependency(TypedDict, total=False):
    """Type definition for a CMake dependency."""

    type: str
    name: str
    options: str
    target: str
    libraries: list[str]


class CMakeInfo(TypedDict):
    """Type definition for parsed CMake file info."""

    file_path: str
    project_name: str | None
    cmake_minimum_required: str | None
    targets: list[CMakeTarget]
    dependencies: list[CMakeDependency]
    subdirectories: list[str]
    variables: dict[str, str]
    includes: list[str]
    libraries: list[str]


class ProjectInfo(TypedDict):
    """Type definition for project structure info."""

    root_path: str
    cmake_files: list[str]
    projects: list[CMakeInfo]
    total_targets: int
    executables: list[CMakeTarget]
    libraries: list[CMakeTarget]
    dependencies: list[str]
    subdirectories: list[str]


class BuildFiles(TypedDict):
    """Type definition for build files."""

    cmake: list[Path]
    makefiles: list[Path]
    ninja: list[Path]
    conan: list[Path]
    vcpkg: list[Path]
    pkg_config: list[Path]


class CompilerSettings(TypedDict):
    """Type definition for compiler settings."""

    cxx_standard: str | None
    c_standard: str | None
    compile_options: list[str]
    compile_definitions: list[str]
    compile_features: list[str]


class InstallConfiguration(TypedDict):
    """Type definition for install configuration."""

    install_targets: list[str]
    install_directories: list[str]
    install_files: list[str]


from models.core_models import (
    CoreCodeEntity,
    EntityType,
)


class CppCMakeManager:
    """CMake project manager for C++ projects."""

    def __init__(self):
        """Initialize the CMake manager."""
        self.cmake_cache: dict[str, Any] = {}

    def find_cmake_files(self, project_root: Path) -> list[Path]:
        """Find all CMake files in the project."""
        cmake_files = []

        # Look for CMakeLists.txt files
        for cmake_file in project_root.rglob("CMakeLists.txt"):
            cmake_files.append(cmake_file)

        # Look for .cmake files
        for cmake_file in project_root.rglob("*.cmake"):
            cmake_files.append(cmake_file)

        return cmake_files

    def parse_cmake_file(self, cmake_file: Path) -> CMakeInfo:
        """Parse a CMake file and extract project information."""
        empty_info: CMakeInfo = {
            "file_path": "",
            "project_name": None,
            "cmake_minimum_required": None,
            "targets": [],
            "dependencies": [],
            "subdirectories": [],
            "variables": {},
            "includes": [],
            "libraries": [],
        }
        if not cmake_file.exists():
            return empty_info

        try:
            content = cmake_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return empty_info

        targets: list[CMakeTarget] = []
        dependencies: list[CMakeDependency] = []
        subdirectories: list[str] = []
        variables: dict[str, str] = {}
        includes: list[str] = []

        # Extract project name
        project_name: str | None = None
        project_match = re.search(
            r"project\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)", content, re.IGNORECASE
        )
        if project_match:
            project_name = project_match.group(1)

        # Extract minimum CMake version
        cmake_minimum_required: str | None = None
        version_match = re.search(
            r"cmake_minimum_required\s*\(\s*VERSION\s+([0-9.]+)", content, re.IGNORECASE
        )
        if version_match:
            cmake_minimum_required = version_match.group(1)

        # Extract targets (executables and libraries)
        targets.extend(self._extract_executables(content))
        targets.extend(self._extract_libraries(content))

        # Extract dependencies
        dependencies.extend(self._extract_find_package(content))
        dependencies.extend(self._extract_target_link_libraries(content))

        # Extract subdirectories
        subdirectories.extend(self._extract_subdirectories(content))

        # Extract variables
        variables.update(self._extract_variables(content))

        # Extract include directories
        includes.extend(self._extract_include_directories(content))

        cmake_info: CMakeInfo = {
            "file_path": str(cmake_file),
            "project_name": project_name,
            "cmake_minimum_required": cmake_minimum_required,
            "targets": targets,
            "dependencies": dependencies,
            "subdirectories": subdirectories,
            "variables": variables,
            "includes": includes,
            "libraries": [],
        }

        return cmake_info

    def _extract_executables(self, content: str) -> list[CMakeTarget]:
        """Extract executable targets from CMake content."""
        executables: list[CMakeTarget] = []

        # Match add_executable commands
        exe_pattern = r"add_executable\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+([^)]+)\)"
        for match in re.finditer(exe_pattern, content, re.IGNORECASE | re.MULTILINE):
            target_name = match.group(1)
            sources = [s.strip() for s in match.group(2).split() if s.strip()]

            executables.append(
                CMakeTarget(
                    type="executable",
                    name=target_name,
                    sources=sources,
                )
            )

        return executables

    def _extract_libraries(self, content: str) -> list[CMakeTarget]:
        """Extract library targets from CMake content."""
        libraries: list[CMakeTarget] = []

        # Match add_library commands
        lib_patterns = [
            r"add_library\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+(STATIC|SHARED|MODULE)?\s*([^)]+)\)",
            r"add_library\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+([^)]+)\)",
        ]

        for pattern in lib_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE):
                groups = match.groups()
                target_name = str(groups[0]) if groups[0] else ""

                if len(groups) == 3 and groups[1] in ["STATIC", "SHARED", "MODULE"]:
                    lib_type = str(groups[1]).lower()
                    sources = [s.strip() for s in str(groups[2]).split() if s.strip()]
                else:
                    lib_type = "static"  # default
                    last_group = groups[-1]
                    sources = (
                        [s.strip() for s in str(last_group).split() if s.strip()]
                        if last_group
                        else []
                    )

                libraries.append(
                    CMakeTarget(
                        type="library",
                        library_type=lib_type,
                        name=target_name,
                        sources=sources,
                    )
                )

        return libraries

    def _extract_find_package(self, content: str) -> list[CMakeDependency]:
        """Extract find_package dependencies."""
        dependencies: list[CMakeDependency] = []

        # Match find_package commands
        find_pattern = r"find_package\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*([^)]*)\)"
        for match in re.finditer(find_pattern, content, re.IGNORECASE):
            package_name = match.group(1)
            options = match.group(2).strip()

            dependencies.append(
                CMakeDependency(
                    type="find_package",
                    name=package_name,
                    options=options,
                )
            )

        return dependencies

    def _extract_target_link_libraries(self, content: str) -> list[CMakeDependency]:
        """Extract target_link_libraries dependencies."""
        dependencies: list[CMakeDependency] = []

        # Match target_link_libraries commands
        link_pattern = r"target_link_libraries\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+([^)]+)\)"
        for match in re.finditer(link_pattern, content, re.IGNORECASE | re.MULTILINE):
            target_name = match.group(1)
            libraries = [lib.strip() for lib in match.group(2).split() if lib.strip()]

            dependencies.append(
                CMakeDependency(
                    type="target_link_libraries",
                    target=target_name,
                    libraries=libraries,
                )
            )

        return dependencies

    def _extract_subdirectories(self, content: str) -> list[str]:
        """Extract subdirectory declarations."""
        subdirs = []

        # Match add_subdirectory commands
        subdir_pattern = r"add_subdirectory\s*\(\s*([^)]+)\)"
        for match in re.finditer(subdir_pattern, content, re.IGNORECASE):
            subdir = match.group(1).strip()
            subdirs.append(subdir)

        return subdirs

    def _extract_variables(self, content: str) -> dict[str, str]:
        """Extract CMake variable definitions."""
        variables = {}

        # Match set commands
        set_pattern = r"set\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+([^)]+)\)"
        for match in re.finditer(set_pattern, content, re.IGNORECASE | re.MULTILINE):
            var_name = match.group(1)
            var_value = match.group(2).strip()
            variables[var_name] = var_value

        return variables

    def _extract_include_directories(self, content: str) -> list[str]:
        """Extract include directory declarations."""
        includes = []

        # Match include_directories commands
        include_pattern = r"include_directories\s*\(\s*([^)]+)\)"
        for match in re.finditer(include_pattern, content, re.IGNORECASE):
            dirs = [d.strip() for d in match.group(1).split() if d.strip()]
            includes.extend(dirs)

        # Match target_include_directories commands
        target_include_pattern = r"target_include_directories\s*\(\s*[A-Za-z_][A-Za-z0-9_]*\s+(?:PUBLIC|PRIVATE|INTERFACE)?\s*([^)]+)\)"
        for match in re.finditer(target_include_pattern, content, re.IGNORECASE | re.MULTILINE):
            dirs = [d.strip() for d in match.group(1).split() if d.strip()]
            includes.extend(dirs)

        return includes

    def analyze_project_structure(self, project_root: Path) -> ProjectInfo:
        """Analyze the overall CMake project structure."""
        cmake_files = self.find_cmake_files(project_root)

        projects: list[CMakeInfo] = []
        total_targets = 0
        executables: list[CMakeTarget] = []
        libraries: list[CMakeTarget] = []
        dependencies_set: set[str] = set()
        subdirectories: list[str] = []

        for cmake_file in cmake_files:
            cmake_data = self.parse_cmake_file(cmake_file)
            if cmake_data:
                projects.append(cmake_data)

                # Aggregate information
                for target in cmake_data.get("targets", []):
                    total_targets += 1
                    target_type = target.get("type", "")
                    if target_type == "executable":
                        executables.append(target)
                    elif target_type == "library":
                        libraries.append(target)

                # Collect dependencies
                for dep in cmake_data.get("dependencies", []):
                    dep_type = dep.get("type", "")
                    if dep_type == "find_package":
                        dep_name = dep.get("name")
                        if dep_name:
                            dependencies_set.add(dep_name)

                # Collect subdirectories
                subdirectories.extend(cmake_data.get("subdirectories", []))

        project_info: ProjectInfo = {
            "root_path": str(project_root),
            "cmake_files": [str(f) for f in cmake_files],
            "projects": projects,
            "total_targets": total_targets,
            "executables": executables,
            "libraries": libraries,
            "dependencies": list(dependencies_set),
            "subdirectories": subdirectories,
        }
        return project_info

    def extract_cmake_entities(self, cmake_file: Path) -> list[CoreCodeEntity]:
        """Extract CMake entities as code entities."""
        entities: list[CoreCodeEntity] = []
        cmake_data = self.parse_cmake_file(cmake_file)

        if not cmake_data.get("file_path"):
            return entities

        # Create entity for the project
        project_name = cmake_data.get("project_name")
        if project_name:
            entities.append(
                CoreCodeEntity(
                    id=f"cmake_project:{project_name}:{cmake_file}",
                    name=project_name,
                    type=EntityType.MODULE,
                    language="cmake",
                    file_path=str(cmake_file),
                    line_number=1,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility=None,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={
                        "cmake_version": cmake_data.get("cmake_minimum_required"),
                        "is_cmake_project": True,
                    },
                )
            )

        # Create entities for targets
        for target in cmake_data.get("targets", []):
            target_type_str = target.get("type", "")
            entity_type = (
                EntityType.FUNCTION if target_type_str == "executable" else EntityType.CLASS
            )
            target_name = target.get("name", "unknown")

            entities.append(
                CoreCodeEntity(
                    id=f"cmake_target:{target_name}:{cmake_file}",
                    name=target_name,
                    type=entity_type,
                    language="cmake",
                    file_path=str(cmake_file),
                    line_number=1,
                    end_line_number=None,
                    column_start=None,
                    column_end=None,
                    visibility=None,
                    code_snippet=None,
                    docstring=None,
                    signature=None,
                    source_code=None,
                    created_at=None,
                    updated_at=None,
                    properties={
                        "target_type": target_type_str,
                        "sources": target.get("sources", []),
                        "library_type": target.get("library_type"),
                        "is_cmake_target": True,
                    },
                )
            )

        return entities

    def find_build_files(self, project_root: Path) -> BuildFiles:
        """Find various build-related files in the project."""
        cmake_files: list[Path] = []
        makefiles: list[Path] = []
        ninja_files: list[Path] = []
        conan_files: list[Path] = []
        vcpkg_files: list[Path] = []
        pkg_config_files: list[Path] = []

        # CMake files
        cmake_files.extend(self.find_cmake_files(project_root))

        # Makefiles
        for makefile in project_root.rglob("Makefile*"):
            makefiles.append(makefile)

        # Ninja files
        for ninja_file in project_root.rglob("*.ninja"):
            ninja_files.append(ninja_file)
        for ninja_file in project_root.rglob("build.ninja"):
            ninja_files.append(ninja_file)

        # Conan files
        for conan_file in project_root.rglob("conanfile.*"):
            conan_files.append(conan_file)

        # vcpkg files
        for vcpkg_file in project_root.rglob("vcpkg.json"):
            vcpkg_files.append(vcpkg_file)

        # pkg-config files
        for pc_file in project_root.rglob("*.pc.in"):
            pkg_config_files.append(pc_file)

        build_files: BuildFiles = {
            "cmake": cmake_files,
            "makefiles": makefiles,
            "ninja": ninja_files,
            "conan": conan_files,
            "vcpkg": vcpkg_files,
            "pkg_config": pkg_config_files,
        }

        return build_files

    def detect_build_system(self, project_root: Path) -> dict[str, Any]:
        """Detect the build system used in the project."""
        build_files = self.find_build_files(project_root)

        build_systems = {
            "cmake": len(build_files["cmake"]) > 0,
            "make": len(build_files["makefiles"]) > 0,
            "ninja": len(build_files["ninja"]) > 0,
            "conan": len(build_files["conan"]) > 0,
            "vcpkg": len(build_files["vcpkg"]) > 0,
        }

        primary_build_system = None
        if build_systems["cmake"]:
            primary_build_system = "cmake"
        elif build_systems["make"]:
            primary_build_system = "make"
        elif build_systems["ninja"]:
            primary_build_system = "ninja"

        return {
            "primary": primary_build_system,
            "detected_systems": build_systems,
            "build_files": build_files,
        }

    def get_compiler_settings(self, cmake_data: CMakeInfo) -> CompilerSettings:
        """Extract compiler settings from CMake data."""
        cxx_standard: str | None = None
        c_standard: str | None = None
        compile_options: list[str] = []
        compile_definitions: list[str] = []
        compile_features: list[str] = []

        variables = cmake_data.get("variables", {})

        # Extract C++ standard
        if "CMAKE_CXX_STANDARD" in variables:
            cxx_standard = variables["CMAKE_CXX_STANDARD"]

        # Extract C standard
        if "CMAKE_C_STANDARD" in variables:
            c_standard = variables["CMAKE_C_STANDARD"]

        # Look for other compiler-related variables
        for var_name, var_value in variables.items():
            if "COMPILE_OPTIONS" in var_name:
                compile_options.append(var_value)
            elif "COMPILE_DEFINITIONS" in var_name:
                compile_definitions.append(var_value)
            elif "COMPILE_FEATURES" in var_name:
                compile_features.append(var_value)

        compiler_settings: CompilerSettings = {
            "cxx_standard": cxx_standard,
            "c_standard": c_standard,
            "compile_options": compile_options,
            "compile_definitions": compile_definitions,
            "compile_features": compile_features,
        }

        return compiler_settings

    def get_install_configuration(self, cmake_data: CMakeInfo) -> InstallConfiguration:
        """Extract install configuration from CMake data."""
        # This would parse install() commands and other installation-related settings
        # Implementation would be similar to other extraction methods
        install_config: InstallConfiguration = {
            "install_targets": [],
            "install_directories": [],
            "install_files": [],
        }

        # Placeholder for install command parsing
        # In a full implementation, this would parse install() commands

        return install_config
