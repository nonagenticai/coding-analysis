"""Entity + relationship extraction for the deep code analyzer.

Free-function form of the original ``DeepCodeAnalyzer._extract_entities`` and
``_extract_entities_from_snippet`` methods. The first argument ``analyzer`` is
the ``DeepCodeAnalyzer`` instance whose ``entities``/``relationships`` lists
get appended to.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from models.core_models import CoreCodeEntity, EntityType

from .base import (
    CodeRelationship,
    extract_base_name,
    extract_decorator_name,
    extract_docstring,
    safe_parse_ast,
)

if TYPE_CHECKING:
    from .analyzer import DeepCodeAnalyzer

logger = logging.getLogger(__name__)


def _module_keys(rel_path: str) -> list[str]:
    """Canonical dotted module path(s) a Python file is importable as.

    ``pkg/core.py`` -> ``["pkg.core"]``; ``pkg/__init__.py`` -> ``["pkg"]`` (the
    package itself). Path separators are normalised to ``/`` first.
    """
    norm = rel_path.replace("\\", "/")
    if not norm.endswith(".py"):
        return []
    stem = norm[: -len(".py")]
    parts = [p for p in stem.split("/") if p]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return []
    return [".".join(parts)]


# Conventional Python source-root directory names (``src``-layout). A package
# physically at ``src/pkg/`` is imported as ``pkg`` (``src/`` is on ``sys.path``
# via ``[tool.setuptools] package-dir = {"" = "src"}`` / hatch / pytest
# ``pythonpath``), never as ``src.pkg``. Without stripping the root, imports in a
# src-layout repo resolve to nothing and the architecture analysis is vacuous.
_SOURCE_ROOT_NAMES = ("src",)


def _source_roots(codebase_path: Path) -> set[str]:
    """Top-level directory names under ``codebase_path`` that act as import
    source roots (``src``-layout).

    A directory only qualifies when it has **no** ``__init__.py`` of its own — a
    real package literally named ``src`` (``src/__init__.py`` present) is imported
    as ``src.foo`` and must NOT be stripped. This guard makes the convention safe.
    """
    roots: set[str] = set()
    for name in _SOURCE_ROOT_NAMES:
        d = codebase_path / name
        if d.is_dir() and not (d / "__init__.py").exists():
            roots.add(name)
    return roots


def _resolve_import_target(
    module_index: dict[str, str],
    module: str | None,
    names: list[str],
    level: int,
    rel_path: str,
) -> str | None:
    """Resolve an import to a LOCAL file entity id by full dotted module path.

    Returns the file entity id of the imported module when it lives in this
    repository, else ``None`` (external/stdlib import — left unresolved so it
    forms no intra-repo edge). Resolution is by the *full* dotted path, never a
    bare basename, so same-named modules across packages (``a/core.py`` vs
    ``b/core.py``) never collide. Handles ``from pkg import submodule`` (the
    imported name is itself a submodule) and relative imports (``level > 0``).
    """
    norm = rel_path.replace("\\", "/")
    pkg_parts = [p for p in norm.split("/")[:-1] if p]  # the file's package dir
    prefix = ""
    if level and level > 0:
        # `from .` is the current package; each extra dot strips one parent.
        keep = len(pkg_parts) - (level - 1)
        prefix = ".".join(pkg_parts[:keep]) if keep > 0 else ""

    base = ".".join(p for p in (prefix, module or "") if p)

    candidates: list[str] = []
    if base:
        # Most specific first: `from base import name` may pull a submodule base.name.
        candidates += [f"{base}.{n}" for n in names if n and n != "*"]
        candidates.append(base)
    elif prefix:
        # `from . import name` (relative, no module): each name is a submodule.
        candidates += [f"{prefix}.{n}" for n in names if n and n != "*"]

    for cand in candidates:
        hit = module_index.get(cand)
        if hit:
            return hit
    return None


def extract_entities(analyzer: DeepCodeAnalyzer) -> None:
    """Extract code entities (functions, classes, etc.) from Python files."""
    logger.info(
        f"🔍 Extracting entities and relationships from {len(analyzer.python_files)} Python files"
    )
    entity_count = 0
    relationship_count = 0
    entity_registry = {}
    # Map every local module's full dotted path -> its file entity id, so import
    # statements resolve to real file nodes (by full path, never basename). Keyed
    # on the canonical dotted path => same-named modules in different packages do
    # not collide. file_id is deterministic in rel_path, so it matches the file
    # entity created in the second pass below.
    module_index: dict[str, str] = {}
    # src-layout repos import ``src/pkg/mod.py`` as ``pkg.mod`` (not ``src.pkg.mod``);
    # register the source-root-stripped dotted path too so those imports resolve.
    source_roots = _source_roots(analyzer.codebase_path)
    for file_path in analyzer.python_files:
        try:
            rel_path = str(file_path.relative_to(analyzer.codebase_path))
        except ValueError:
            continue
        file_id = analyzer.ast_utils.generate_entity_id("file", rel_path, rel_path, 0)
        for key in _module_keys(rel_path):
            module_index.setdefault(key, file_id)
            for root in source_roots:
                if key.startswith(root + "."):
                    # ``setdefault`` => a genuine flat-layout module of the same
                    # dotted name always wins; the stripped alias never overrides.
                    module_index.setdefault(key[len(root) + 1 :], file_id)
    for file_path in analyzer.python_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = safe_parse_ast(content, str(file_path))
            if tree is None:
                continue
            rel_path = str(file_path.relative_to(analyzer.codebase_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    entity_id = analyzer.ast_utils.generate_entity_id(
                        "function", node.name, rel_path, node.lineno
                    )
                    entity_registry[node.name] = entity_id
                elif isinstance(node, ast.ClassDef):
                    entity_id = analyzer.ast_utils.generate_entity_id(
                        "class", node.name, rel_path, node.lineno
                    )
                    entity_registry[node.name] = entity_id
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"⚠️  Failed to register entities from {file_path}: {e}")
    for file_path in analyzer.python_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = safe_parse_ast(content, str(file_path))
            if tree is None:
                continue
            rel_path = str(file_path.relative_to(analyzer.codebase_path))
            file_id = analyzer.ast_utils.generate_entity_id("file", rel_path, rel_path, 0)
            file_entities = set()
            # Deferred (function-local) imports are the standard Python idiom for
            # breaking an import-time cycle; they do not execute at module import
            # time, so they must NOT count as an architecture dependency-cycle edge
            # (the exclusion is applied downstream in ComponentExtractor
            # ._add_node_dependencies, which reads this ``deferred`` flag). Scope =
            # any import nested inside a function/method body (covers nested
            # functions and methods); a class-body import still runs at import time
            # and is therefore NOT deferred.
            deferred_import_ids: set[int] = set()
            for _fn in ast.walk(tree):
                if isinstance(_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for _sub in ast.walk(_fn):
                        if isinstance(_sub, (ast.Import, ast.ImportFrom)):
                            deferred_import_ids.add(id(_sub))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            resolved = _resolve_import_target(
                                module_index, alias.name, [], 0, rel_path
                            )
                            relationship = CodeRelationship(
                                source_id=file_id,
                                target_id=resolved or f"module_{alias.name}",
                                type="imports",
                                metadata={
                                    "line": node.lineno,
                                    "import_type": "direct",
                                    "module": alias.name,
                                    "resolved": bool(resolved),
                                    "deferred": id(node) in deferred_import_ids,
                                },
                            )
                            analyzer.relationships.append(relationship)
                            relationship_count += 1
                    elif isinstance(node, ast.ImportFrom) and (node.module or node.level):
                        names = [a.name for a in node.names]
                        resolved = _resolve_import_target(
                            module_index, node.module, names, node.level, rel_path
                        )
                        relationship = CodeRelationship(
                            source_id=file_id,
                            target_id=resolved or f"module_{node.module}",
                            type="imports_from",
                            metadata={
                                "line": node.lineno,
                                "import_type": "from",
                                "module": node.module,
                                "level": node.level,
                                "resolved": bool(resolved),
                                "deferred": id(node) in deferred_import_ids,
                            },
                        )
                        analyzer.relationships.append(relationship)
                        relationship_count += 1
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    entity_id = analyzer.ast_utils.generate_entity_id(
                        "function", node.name, rel_path, node.lineno
                    )
                    docstring = extract_docstring(node)
                    entity = CoreCodeEntity(
                        id=entity_id,
                        type=EntityType.FUNCTION,
                        name=node.name,
                        language=analyzer.language,  # Use language from init
                        line_number=node.lineno,
                        file_path=rel_path,
                        docstring=docstring,
                        metadata={
                            "args": [arg.arg for arg in node.args.args],
                            "returns": (
                                getattr(node.returns, "id", None) if node.returns else None
                            ),
                            "is_async": isinstance(node, ast.AsyncFunctionDef),
                            "decorators": [
                                extract_decorator_name(dec) for dec in node.decorator_list
                            ],
                        },
                    )
                    analyzer.entities.append(entity)
                    entity_count += 1
                    file_entities.add(node.name)
                    for call_node in ast.walk(node):
                        if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name):
                            target_name = call_node.func.id
                            target_id = entity_registry.get(target_name, f"function_{target_name}")
                            relationship = CodeRelationship(
                                source_id=entity_id,
                                target_id=target_id,
                                type="calls",
                                metadata={
                                    "line": call_node.lineno,
                                    "in_file": rel_path,
                                },
                            )
                            analyzer.relationships.append(relationship)
                            relationship_count += 1
                elif isinstance(node, ast.ClassDef):
                    entity_id = analyzer.ast_utils.generate_entity_id(
                        "class", node.name, rel_path, node.lineno
                    )
                    docstring = extract_docstring(node)
                    entity = CoreCodeEntity(
                        id=entity_id,
                        type=EntityType.CLASS,
                        name=node.name,
                        language=analyzer.language,  # Use language from init
                        line_number=node.lineno,
                        file_path=rel_path,
                        docstring=docstring,
                        metadata={
                            "bases": [extract_base_name(base) for base in node.bases],
                            "decorators": [
                                extract_decorator_name(dec) for dec in node.decorator_list
                            ],
                            "methods": [
                                method.name
                                for method in node.body
                                if isinstance(method, ast.FunctionDef)
                            ],
                        },
                    )
                    analyzer.entities.append(entity)
                    entity_count += 1
                    file_entities.add(node.name)
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            target_name = base.id
                            target_id = entity_registry.get(target_name, f"class_{target_name}")
                            relationship = CodeRelationship(
                                source_id=entity_id,
                                target_id=target_id,
                                type="inherits",
                                metadata={"line": node.lineno, "in_file": rel_path},
                            )
                            analyzer.relationships.append(relationship)
                            relationship_count += 1
            file_entity = CoreCodeEntity(
                id=file_id,
                type=EntityType.FILE,
                name=file_path.name,
                language="python",  # Add language field
                line_number=1,
                file_path=rel_path,
                docstring=None,
                metadata={
                    "file_size": file_path.stat().st_size,
                    "extension": file_path.suffix,
                    "entities_in_file": list(file_entities),
                },
            )
            analyzer.entities.append(file_entity)
            entity_count += 1
            for entity_name in file_entities:
                if entity_name in entity_registry:
                    relationship = CodeRelationship(
                        source_id=file_id,
                        target_id=entity_registry[entity_name],
                        type="contains",
                        metadata={"entity_type": "code_element"},
                    )
                    analyzer.relationships.append(relationship)
                    relationship_count += 1
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"⚠️  Failed to extract entities from {file_path}: {e}")
    logger.info(f"✅ Extracted {entity_count} entities and {relationship_count} relationships")
    logger.info(
        f"   📊 {len([e for e in analyzer.entities if e.type == 'function'])} functions, {len([e for e in analyzer.entities if e.type == 'class'])} classes, {len([e for e in analyzer.entities if e.type == 'file'])} files"
    )


def extract_entities_from_snippet(
    analyzer: DeepCodeAnalyzer,
    code: str,
    tree: ast.AST,
    filename: str = "snippet.py",
) -> None:
    """Extract entities from a code snippet."""
    _ = code  # unused but preserved for parity with the original signature
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            entity_id = analyzer.ast_utils.generate_entity_id(
                "function", node.name, filename, node.lineno
            )
            docstring = extract_docstring(node)
            entity = CoreCodeEntity(
                id=entity_id,
                type=EntityType.FUNCTION,
                name=node.name,
                language=analyzer.language,
                line_number=node.lineno,
                file_path=filename,
                docstring=docstring,
                metadata={
                    "args": [arg.arg for arg in node.args.args],
                    "returns": (getattr(node.returns, "id", None) if node.returns else None),
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "decorators": [extract_decorator_name(dec) for dec in node.decorator_list],
                },
            )
            analyzer.entities.append(entity)
        elif isinstance(node, ast.ClassDef):
            entity_id = analyzer.ast_utils.generate_entity_id(
                "class", node.name, filename, node.lineno
            )
            docstring = extract_docstring(node)
            entity = CoreCodeEntity(
                id=entity_id,
                type=EntityType.CLASS,
                name=node.name,
                language=analyzer.language,
                line_number=node.lineno,
                file_path=filename,
                docstring=docstring,
                metadata={
                    "bases": [extract_base_name(base) for base in node.bases],
                    "decorators": [extract_decorator_name(dec) for dec in node.decorator_list],
                    "methods": [
                        method.name for method in node.body if isinstance(method, ast.FunctionDef)
                    ],
                },
            )
            analyzer.entities.append(entity)
