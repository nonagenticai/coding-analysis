"""Multi-language import and syntax scanning for the repository scanner.

The legacy scanner path parses files with Python's stdlib ``ast`` inside the
target container. This module covers every *other* language the platform
supports: it reads file content out of the container and parses it locally
with the shared ``processing.languages`` framework -- so no tree-sitter
runtime has to be installed inside the scanned container.

Python files keep their proven in-container AST path in ``scanner_core`` (the
framework's Python import extraction is inconsistent across its processors);
this module is only invoked for non-Python source files, leaving pure-Python
repositories byte-for-byte unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import tomllib
from pathlib import Path

from defusedxml.ElementTree import ParseError as XMLParseError
from defusedxml.ElementTree import fromstring as xml_fromstring
from fastapi import HTTPException

from models.core_models import EntityType, RelationType
from processing.languages import language_registry
from processing.languages.parse_helpers import parse_source

from ...models.repo_scanner_api_models import (
    ApiValidationResult,
    RepoScannerImportInfo,
    ScannedDependency,
)
from .file_discovery import execute_in_container

logger = logging.getLogger(__name__)

# Extensions handled by the legacy in-container Python AST path.
_PYTHON_EXTENSIONS = frozenset({".py", ".pyw"})

# Lines longer than this raise a style warning, matching the Python path.
_MAX_LINE_LENGTH = 120

# Non-Python dependency manifests, by file name. The Python path
# (pip / poetry / requirements.txt) stays in ``scanner_core.scan_dependencies``.
_MANIFEST_FILE_NAMES = frozenset({"package.json", "go.mod", "Cargo.toml", "pom.xml"})


def partition_by_python(files: list[str]) -> tuple[list[str], list[str]]:
    """Split *files* into ``(python_files, other_supported_files)``.

    Files whose extension is neither Python nor a framework-supported language
    are dropped from both lists.

    Args:
        files: Candidate file paths.

    Returns:
        A ``(python_files, other_supported_files)`` tuple.
    """
    supported = language_registry.get_supported_extensions()
    python_files: list[str] = []
    other_files: list[str] = []
    for file_path in files:
        suffix = Path(file_path).suffix.lower()
        if suffix in _PYTHON_EXTENSIONS:
            python_files.append(file_path)
        elif suffix in supported:
            other_files.append(file_path)
    return python_files, other_files


async def _read_container_file(
    container_name: str, file_path: str, *, tenant_id: str | None = None
) -> str | None:
    """Read a single file's text content out of a container, or ``None``."""
    try:
        result = await execute_in_container(
            container_name, f"cat '{file_path}'", tenant_id=tenant_id
        )
    except HTTPException as exc:
        # Container vanished / stopped mid-scan: skip this file, do not abort.
        logger.warning("Could not read %s from %s: %s", file_path, container_name, exc.detail)
        return None
    if not result["success"]:
        logger.warning("Could not read %s from %s: %s", file_path, container_name, result["output"])
        return None
    return result["output"]


def _module_from_relation_target(target_id: str) -> str:
    """Normalise an ``IMPORTS`` relation target into a bare module name.

    Processors encode import targets differently — Go uses
    ``import:<module>:<path>:<line>`` ids, TypeScript uses the raw module name.
    """
    if target_id.startswith("import:"):
        parts = target_id.split(":")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return target_id


def _imports_from_parse_result(file_path: str, result: object) -> list[RepoScannerImportInfo]:
    """Map a :class:`ParseResult` to ``RepoScannerImportInfo`` records.

    Prefers explicit ``IMPORT`` entities (Go, C#, ...); falls back to
    ``IMPORTS`` relations (TypeScript, ...) when a processor only emits those.
    """
    entities = getattr(result, "entities", [])
    relationships = getattr(result, "relationships", [])

    import_entities = [e for e in entities if e.type == EntityType.IMPORT]
    if import_entities:
        return [
            RepoScannerImportInfo(
                module=entity.name,
                name=None,
                alias=None,
                import_type=str(entity.properties.get("import_type", "import")),
                file_path=file_path,
                line_number=entity.line_number or 0,
                is_valid=True,
            )
            for entity in import_entities
        ]

    imports: list[RepoScannerImportInfo] = []
    seen: set[str] = set()
    for relation in relationships:
        if relation.type != RelationType.IMPORTS:
            continue
        module = _module_from_relation_target(relation.target_id)
        if not module or module in seen:
            continue
        seen.add(module)
        imports.append(
            RepoScannerImportInfo(
                module=module,
                name=None,
                alias=None,
                import_type="import",
                file_path=file_path,
                line_number=relation.line_number or 0,
                is_valid=True,
            )
        )
    return imports


async def scan_imports_multilang(
    container_name: str,
    files: list[str],
    *,
    tenant_id: str | None = None,
) -> list[RepoScannerImportInfo]:
    """Extract import statements from non-Python source files.

    Args:
        container_name: Container whose files are being scanned.
        files: Non-Python source file paths (see :func:`partition_by_python`).

    Returns:
        Import records across every successfully parsed file. A file that
        cannot be read or parsed is logged and skipped, never fatal.
    """
    imports: list[RepoScannerImportInfo] = []
    for file_path in files:
        content = await _read_container_file(container_name, file_path, tenant_id=tenant_id)
        if content is None:
            continue
        result = await parse_source(Path(file_path), content)
        if result is None:
            logger.warning("No parser produced a result for %s", file_path)
            continue
        imports.extend(_imports_from_parse_result(file_path, result))
    return imports


async def validate_code_multilang(
    container_name: str,
    files: list[str],
    *,
    tenant_id: str | None = None,
) -> list[ApiValidationResult]:
    """Syntax-validate non-Python source files via the language framework.

    Each file is read out of the container and parsed; ``ParseResult.success``
    becomes ``is_valid`` and ``ParseResult.errors`` populate ``issues``. The
    language-agnostic long-line style check from the Python path is preserved.

    Args:
        container_name: Container whose files are being validated.
        files: Non-Python source file paths (see :func:`partition_by_python`).

    Returns:
        One :class:`ApiValidationResult` per file.
    """
    results: list[ApiValidationResult] = []
    for file_path in files:
        content = await _read_container_file(container_name, file_path, tenant_id=tenant_id)
        if content is None:
            results.append(
                ApiValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    issues=["Could not read file from container"],
                    warnings=[],
                    line_count=0,
                )
            )
            continue

        lines = content.splitlines()
        line_count = len(lines)
        result = await parse_source(Path(file_path), content)

        if result is None:
            results.append(
                ApiValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    issues=["Parser unavailable or failed for this file"],
                    warnings=[],
                    line_count=line_count,
                )
            )
            continue

        issues = [] if result.success else (result.errors or ["Syntax error"])
        warnings = [
            f"Line {index}: Line too long ({len(line)} characters)"
            for index, line in enumerate(lines, 1)
            if len(line) > _MAX_LINE_LENGTH
        ]
        results.append(
            ApiValidationResult(
                file_path=file_path,
                is_valid=result.success,
                issues=issues,
                warnings=warnings,
                line_count=line_count,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Multi-language dependency-manifest parsing
#
# The Python path (pip / poetry / requirements.txt) keeps its proven
# in-container script in ``scanner_core.scan_dependencies``. This section adds
# JS/TS, Go, Rust and Java by reading the ecosystem manifest out of the
# container and parsing it locally with the stdlib — no per-ecosystem tooling
# has to be installed inside the scanned container.
# ---------------------------------------------------------------------------


# ``ScannedDependency.version`` is constrained to ``VERSION_PATTERN`` — which
# does not allow ``^`` (npm/cargo caret ranges). Strip any character outside
# that set so a real caret range like ``^4.18.0`` still validates as ``4.18.0``.
_VERSION_ALLOWED = re.compile(r"[A-Za-z0-9._\-,<>=!*~+ ]")


def _safe_version(version: str) -> str:
    """Coerce *version* into the ``VERSION_PATTERN`` charset, or ``"unknown"``."""
    cleaned = "".join(_VERSION_ALLOWED.findall(version)).strip()
    return cleaned or "unknown"


def _parse_package_json(content: str) -> list[ScannedDependency]:
    """Parse an npm/yarn ``package.json`` into dependency records."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Could not parse package.json: %s", exc)
        return []
    deps: list[ScannedDependency] = []
    for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        block = data.get(section)
        if not isinstance(block, dict):
            continue
        for name, version in block.items():
            deps.append(
                ScannedDependency(
                    name=str(name),
                    version=_safe_version(str(version)) if version else "unknown",
                    source="package.json",
                    is_valid=True,
                )
            )
    return deps


def _parse_go_mod(content: str) -> list[ScannedDependency]:
    """Parse a Go ``go.mod`` into dependency records.

    Handles both the single-line ``require module v1.2.3`` form and the
    block ``require ( ... )`` form; trailing ``// indirect`` markers are
    ignored for the version.
    """
    deps: list[ScannedDependency] = []
    in_block = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if in_block:
            if line.startswith(")"):
                in_block = False
                continue
            spec = line
        elif line.startswith("require ("):
            in_block = True
            continue
        elif line.startswith("require "):
            spec = line[len("require ") :].strip()
        else:
            continue

        spec = spec.split("//", 1)[0].strip()
        parts = spec.split()
        if len(parts) >= 2:
            deps.append(
                ScannedDependency(
                    name=parts[0],
                    version=_safe_version(parts[1]),
                    source="go.mod",
                    is_valid=True,
                )
            )
    return deps


def _parse_cargo_toml(content: str) -> list[ScannedDependency]:
    """Parse a Rust ``Cargo.toml`` into dependency records."""
    try:
        data = tomllib.loads(content)
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        logger.warning("Could not parse Cargo.toml: %s", exc)
        return []
    deps: list[ScannedDependency] = []
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        block = data.get(section)
        if not isinstance(block, dict):
            continue
        for name, spec in block.items():
            # A dependency is either a bare version string or a table that
            # may carry a ``version`` key (git/path deps may omit it).
            if isinstance(spec, str):
                version = spec
            elif isinstance(spec, dict):
                version = str(spec.get("version", "unknown"))
            else:
                version = "unknown"
            deps.append(
                ScannedDependency(
                    name=str(name),
                    version=_safe_version(version) if version else "unknown",
                    source="Cargo.toml",
                    is_valid=True,
                )
            )
    return deps


def _parse_pom_xml(content: str) -> list[ScannedDependency]:
    """Parse a Maven ``pom.xml`` into dependency records.

    Maven namespaces every tag, so tags are matched by their local name
    (the part after any ``{namespace}`` prefix) rather than a fixed path.
    """

    def _local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    try:
        root = xml_fromstring(content)
    except XMLParseError as exc:
        logger.warning("Could not parse pom.xml: %s", exc)
        return []

    deps: list[ScannedDependency] = []
    for dependency in root.iter():
        if _local(dependency.tag) != "dependency":
            continue
        fields = {_local(child.tag): (child.text or "").strip() for child in dependency}
        group_id = fields.get("groupId", "")
        artifact_id = fields.get("artifactId", "")
        if not artifact_id:
            continue
        name = f"{group_id}:{artifact_id}" if group_id else artifact_id
        deps.append(
            ScannedDependency(
                name=name,
                version=_safe_version(fields.get("version", "")),
                source="pom.xml",
                is_valid=True,
            )
        )
    return deps


# Manifest file name -> parser. Keyed by the exact file name so the same
# parser runs regardless of where in the repo the manifest lives.
_MANIFEST_PARSERS = {
    "package.json": _parse_package_json,
    "go.mod": _parse_go_mod,
    "Cargo.toml": _parse_cargo_toml,
    "pom.xml": _parse_pom_xml,
}


async def _discover_manifest_files(
    container_name: str, *, tenant_id: str | None = None
) -> list[str]:
    """Find non-Python dependency manifests under ``/workspace`` in a container."""
    names = sorted(_MANIFEST_FILE_NAMES)
    name_expr = " -o ".join(f"-name '{name}'" for name in names)
    # Skip vendored trees so we report the project's own manifests, not its
    # dependencies' manifests.
    command = (
        f"find /workspace -maxdepth 6 -type f \\( {name_expr} \\) "
        "-not -path '*/node_modules/*' -not -path '*/vendor/*' "
        "-not -path '*/target/*' -not -path '*/.git/*' "
        # The source checkouts of a multi-repo workspace are not this repo's.
        "-not -path '/workspace/.repos' -not -path '/workspace/.repos/*'"
    )
    try:
        result = await execute_in_container(container_name, command, tenant_id=tenant_id)
    except HTTPException as exc:
        logger.warning("Manifest discovery failed for %s: %s", container_name, exc.detail)
        return []
    if not result["success"]:
        # ``find`` exits non-zero only on real errors here; an empty repo
        # simply yields no output with exit 0.
        logger.warning("Manifest discovery failed for %s: %s", container_name, result["output"])
        return []
    return [line.strip() for line in result["output"].splitlines() if line.strip()]


async def scan_dependencies_multilang(
    container_name: str, *, tenant_id: str | None = None
) -> list[ScannedDependency]:
    """Scan non-Python dependency manifests in a container.

    Discovers ``package.json`` / ``go.mod`` / ``Cargo.toml`` / ``pom.xml``,
    reads each out of the container and parses it locally with the stdlib.
    A manifest that cannot be read or parsed is logged and skipped, never
    fatal — mirroring the failure isolation of the import/validation paths.

    Args:
        container_name: Container whose manifests are being scanned.

    Returns:
        Dependency records across every successfully parsed manifest.
    """
    manifest_files = await _discover_manifest_files(container_name, tenant_id=tenant_id)
    deps: list[ScannedDependency] = []
    for file_path in manifest_files:
        file_name = Path(file_path).name
        parser = _MANIFEST_PARSERS.get(file_name)
        if parser is None:
            continue
        content = await _read_container_file(container_name, file_path, tenant_id=tenant_id)
        if content is None:
            continue
        deps.extend(parser(content))
    return deps
