"""Semantic analysis service.

Parses source code and extracts entities and relationships
using the Python ``ast`` module (for Python sources) or a
simple token-based fallback for other languages.
"""

from __future__ import annotations

import ast
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_python(code: str) -> dict[str, Any]:
    """Parse Python source and return entities + relationships."""
    tree = ast.parse(code)

    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            entity: dict[str, Any] = {
                "name": node.name,
                "type": "function",
                "line_number": node.lineno,
                "end_line_number": node.end_lineno,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "decorators": [_decorator_name(d) for d in node.decorator_list],
                "args": [a.arg for a in node.args.args],
            }
            entities.append(entity)

            # Detect calls inside the function body
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    callee = _call_name(child)
                    if callee:
                        relationships.append(
                            {
                                "source": node.name,
                                "target": callee,
                                "type": "calls",
                            }
                        )

        elif isinstance(node, ast.ClassDef):
            bases = [_name_of(b) for b in node.bases]
            entity = {
                "name": node.name,
                "type": "class",
                "line_number": node.lineno,
                "end_line_number": node.end_lineno,
                "bases": bases,
                "decorators": [_decorator_name(d) for d in node.decorator_list],
            }
            entities.append(entity)

            for base in bases:
                if base:
                    relationships.append(
                        {
                            "source": node.name,
                            "target": base,
                            "type": "inherits",
                        }
                    )

        elif isinstance(node, ast.Import):
            for alias in node.names:
                entities.append(
                    {
                        "name": alias.name,
                        "type": "import",
                        "line_number": node.lineno,
                        "alias": alias.asname,
                    }
                )

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                entities.append(
                    {
                        "name": f"{module}.{alias.name}" if module else alias.name,
                        "type": "import",
                        "line_number": node.lineno,
                        "alias": alias.asname,
                        "module": module,
                    }
                )

    return {"entities": entities, "relationships": relationships}


def _parse_generic(code: str, language: str) -> dict[str, Any]:
    """Lightweight regex-based extraction for non-Python languages."""
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, str]] = []

    func_pattern = re.compile(
        r"(?:(?:public|private|protected|static|async|export)\s+)*"
        r"(?:function|def|fn|func)\s+(\w+)",
    )
    class_pattern = re.compile(
        r"(?:(?:public|private|protected|abstract|final|export)\s+)*"
        r"class\s+(\w+)(?:\s*(?:extends|:)\s*(\w+))?",
    )
    import_pattern = re.compile(
        r"(?:import|using|require|from)\s+[\"']?([^\s\"';]+)",
    )

    for lineno, line in enumerate(code.splitlines(), start=1):
        m = func_pattern.search(line)
        if m:
            entities.append({"name": m.group(1), "type": "function", "line_number": lineno})

        m = class_pattern.search(line)
        if m:
            entity: dict[str, Any] = {
                "name": m.group(1),
                "type": "class",
                "line_number": lineno,
            }
            entities.append(entity)
            if m.group(2):
                relationships.append(
                    {
                        "source": m.group(1),
                        "target": m.group(2),
                        "type": "inherits",
                    }
                )

        m = import_pattern.search(line)
        if m and not func_pattern.search(line) and not class_pattern.search(line):
            entities.append({"name": m.group(1), "type": "import", "line_number": lineno})

    return {"entities": entities, "relationships": relationships}


def _decorator_name(node: ast.expr) -> str:
    """Extract decorator name from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_name_of(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""


def _call_name(node: ast.Call) -> str:
    """Extract callee name from an ``ast.Call`` node."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _name_of(node: ast.expr) -> str:
    """Return a string representation of a simple AST name node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = _name_of(node.value)
        return f"{value}.{node.attr}" if value else node.attr
    return ""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class SemanticAnalysisService:
    """Parses source code and returns entities, relationships, and metrics."""

    async def analyze(self, code: str, language: str = "python") -> dict[str, Any]:
        """Perform semantic analysis on *code*.

        Parameters
        ----------
        code:
            The source code to analyse.
        language:
            Programming language of *code* (default ``"python"``).

        Returns
        -------
        dict with keys ``language``, ``entities``, ``relationships``,
        ``metrics``, and ``errors``.
        """
        start = time.monotonic()
        errors: list[str] = []

        try:
            if language.lower() == "python":
                parsed = _parse_python(code)
            else:
                parsed = _parse_generic(code, language)
        except SyntaxError as exc:
            logger.warning("SyntaxError while parsing %s code: %s", language, exc)
            errors.append(f"SyntaxError: {exc}")
            parsed = {"entities": [], "relationships": []}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error while parsing %s code: %s", language, exc)
            errors.append(f"{type(exc).__name__}: {exc}")
            parsed = {"entities": [], "relationships": []}

        elapsed_ms = (time.monotonic() - start) * 1000

        entities = parsed["entities"]
        relationships = parsed["relationships"]

        # De-duplicate relationships (same triple seen inside nested walks)
        seen: set[tuple[str, str, str]] = set()
        unique_rels: list[dict[str, str]] = []
        for rel in relationships:
            key = (rel["source"], rel["target"], rel["type"])
            if key not in seen:
                seen.add(key)
                unique_rels.append(rel)

        return {
            "language": language,
            "entities": entities,
            "relationships": unique_rels,
            "metrics": {
                "total_entities": len(entities),
                "total_relationships": len(unique_rels),
                "parse_time_ms": round(elapsed_ms, 3),
            },
            "errors": errors,
        }

    async def analyze_codebase_semantics(
        self, file_paths: list[Path], analysis_id: str
    ) -> dict[str, Any]:
        """Analyse multiple files and aggregate results.

        Parameters
        ----------
        file_paths:
            Paths to source files to analyse.
        analysis_id:
            Unique identifier for this analysis run.

        Returns
        -------
        dict with ``analysis_summary``, ``entities``, ``basic_relationships``,
        and ``semantic_relationships``.
        """
        all_entities: list[dict[str, Any]] = []
        all_relationships: list[dict[str, Any]] = []
        language_counts: dict[str, int] = {}
        errors: list[str] = []

        ext_to_lang = {
            ".py": "python",
            ".java": "java",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".cs": "csharp",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
        }

        for file_path in file_paths:
            try:
                code = file_path.read_text(encoding="utf-8", errors="ignore")
                language = ext_to_lang.get(file_path.suffix.lower(), "unknown")
                result = await self.analyze(code, language)

                for entity in result.get("entities", []):
                    entity["file_path"] = str(file_path)
                    all_entities.append(entity)

                for rel in result.get("relationships", []):
                    rel["file_path"] = str(file_path)
                    all_relationships.append(rel)

                language_counts[language] = language_counts.get(language, 0) + 1
                if result.get("errors"):
                    errors.extend(result["errors"])
            except Exception as exc:
                logger.warning("Error analysing %s: %s", file_path, exc)
                errors.append(f"{file_path}: {exc}")

        return {
            "analysis_summary": {
                "analysis_id": analysis_id,
                "total_files": len(file_paths),
                "total_entities": len(all_entities),
                "total_relationships": len(all_relationships),
                "languages": language_counts,
                "errors": errors,
            },
            "entities": all_entities,
            "basic_relationships": all_relationships,
            "semantic_relationships": {},
        }

    async def get_semantic_insights(self, analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Derive high-level insights from aggregated analysis results.

        Parameters
        ----------
        analysis_results:
            Output of :meth:`analyze_codebase_semantics`.

        Returns
        -------
        dict with ``language_distribution``, ``cross_language_interactions``,
        ``architectural_patterns``, ``complexity_hotspots``, and
        ``coupling_analysis``.
        """
        entities = analysis_results.get("entities", [])
        summary = analysis_results.get("analysis_summary", {})
        languages = summary.get("languages", {})

        # Language distribution
        language_distribution: dict[str, dict[str, Any]] = {}
        for lang, count in languages.items():
            lang_entities = [
                e
                for e in entities
                if e.get("file_path", "").endswith(
                    tuple(
                        k
                        for k, v in {
                            ".py": "python",
                            ".java": "java",
                            ".js": "javascript",
                            ".ts": "typescript",
                            ".go": "go",
                            ".rs": "rust",
                            ".cs": "csharp",
                            ".cpp": "cpp",
                            ".c": "c",
                        }.items()
                        if v == lang
                    )
                )
            ]
            entity_types: dict[str, int] = {}
            for e in lang_entities:
                t = e.get("type", "unknown")
                entity_types[t] = entity_types.get(t, 0) + 1
            language_distribution[lang] = {
                "file_count": count,
                "entity_count": len(lang_entities),
                "entity_types": entity_types,
            }

        # Complexity hotspots - entities with many relationships
        relationships = analysis_results.get("basic_relationships", [])
        entity_rel_count: dict[str, int] = {}
        for rel in relationships:
            src = rel.get("source", "")
            entity_rel_count[src] = entity_rel_count.get(src, 0) + 1

        complexity_hotspots: list[dict[str, Any]] = []
        for entity in entities:
            name = entity.get("name", "")
            rel_count = entity_rel_count.get(name, 0)
            if rel_count >= 3:
                complexity_hotspots.append(
                    {
                        "entity_name": name,
                        "entity_type": entity.get("type", "unknown"),
                        "language": "unknown",
                        "relationship_count": rel_count,
                        "file_path": entity.get("file_path", ""),
                    }
                )
        complexity_hotspots.sort(key=lambda x: x["relationship_count"], reverse=True)

        return {
            "language_distribution": language_distribution,
            "cross_language_interactions": [],
            "architectural_patterns": [],
            "complexity_hotspots": complexity_hotspots[:20],
            "coupling_analysis": {"language_pairs": []},
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_semantic_analysis_service: SemanticAnalysisService | None = None


def get_semantic_analysis_service() -> SemanticAnalysisService:
    """Return a singleton SemanticAnalysisService instance."""
    global _semantic_analysis_service  # noqa: PLW0603
    if _semantic_analysis_service is None:
        _semantic_analysis_service = SemanticAnalysisService()
    return _semantic_analysis_service
